"""Phase 1: run the *existing* stacked CR training under observation.

This runner deliberately calls the production code path
(``step_3_preprocessing.get_step_3_tinfo_events`` -> ``train_stacked_fvt`` ->
``StackedFvTClassifier.fit``) without editing any of it.  The only changes are
made by two wrappers installed at runtime:

1. a ``FingerprintCallback`` is appended to the callback list handed to
   ``fit``, and
2. the process changes directory into a per-run scratch workdir for the
   duration of ``fit`` only, so that the relative paths ``./data/checkpoints``
   and ``./tb_logs`` written by ``fit`` land in the run directory instead of
   the production cache.

Neither wrapper touches the model, loss, optimizer, scheduler, sampler or
batch-size schedule.  ``TrainingInfo`` records are never saved.

Usage (on a compute node, one GPU):

    python phase1/run_baseline.py --tag A --num-stacks 5 --max-epochs 20 \
        --out-root phase1_runs/dev
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
SOHEUN = HERE.parent
sys.path.insert(0, str(SOHEUN))
os.chdir(SOHEUN)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from constants import FEATURES  # noqa: E402
from fvt_classifier import FvTClassifier  # noqa: E402
from step_3_preprocessing import (  # noqa: E402
    check_and_get_CR_fvt_hparams,
    get_step_3_tinfo_events,
)
from stacked_fvt import StackedFvTClassifier  # noqa: E402
from train_stacked_fvts_from_tinfo import train_stacked_fvt  # noqa: E402

from fingerprint import FingerprintCallback, env_record, tensor_digest  # noqa: E402

DEFAULT_PATTERN = (
    "configs/tmp/CR_fvt_training_ensemble_max_{seed}_0.0_0_0.1_0.2_0.8.yml"
)


@contextlib.contextmanager
def pushd(path: pathlib.Path):
    prev = pathlib.Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def install_fit_wrapper(callback, workdir: pathlib.Path):
    """Inject the fingerprint callback and redirect fit's relative paths."""
    original_fit = StackedFvTClassifier.fit

    def wrapped_fit(self, *args, **kwargs):
        kwargs["callbacks"] = list(kwargs.get("callbacks") or []) + [callback]
        with pushd(workdir):
            return original_fit(self, *args, **kwargs)

    StackedFvTClassifier.fit = wrapped_fit
    return original_fit


def load_configs(pattern: str, seeds: list[int], max_epochs: int | None) -> list[dict]:
    configs = []
    for s in seeds:
        path = SOHEUN / pattern.format(seed=s)
        if not path.exists():
            raise FileNotFoundError(f"config not found: {path}")
        with open(path) as f:
            cfg = yaml.safe_load(f)
        if max_epochs is not None:
            cfg["CR_fvt"]["max_epochs"] = int(max_epochs)
        cfg["_phase1_source_config"] = str(path)
        configs.append(cfg)
    return configs


def comparable_hparams(hparams: dict) -> dict:
    """hparams with the non-deterministic pieces removed, for run-to-run
    comparison (TrainingInfo hashes are timestamp-based, not content-based)."""
    out = {}
    for k, v in sorted(hparams.items()):
        if k.startswith("aux_info"):
            continue
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="run label, e.g. A or B")
    ap.add_argument("--out-root", default="phase1_runs/dev")
    ap.add_argument("--config-pattern", default=DEFAULT_PATTERN)
    ap.add_argument("--num-stacks", type=int, default=5)
    ap.add_argument("--seeds", default=None, help="comma separated dataset seeds")
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--probe-size", type=int, default=20000)
    ap.add_argument("--keep-checkpoints", action="store_true")
    args = ap.parse_args()

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = list(range(args.num_stacks))

    out_dir = (SOHEUN / args.out_root / args.tag).resolve()
    workdir = out_dir / "workdir"
    workdir.mkdir(parents=True, exist_ok=True)

    print(f"[phase1] tag={args.tag} seeds={seeds} max_epochs={args.max_epochs}")
    print(f"[phase1] out_dir={out_dir}")
    print(f"[phase1] env={json.dumps(env_record(), indent=2)}", flush=True)

    configs = load_configs(args.config_pattern, seeds, args.max_epochs)
    hparams_list = [check_and_get_CR_fvt_hparams(c) for c in configs]

    t0 = time.perf_counter()
    tinfos = []
    for i, hp in enumerate(hparams_list):
        ti = time.perf_counter()
        tinfos.append(get_step_3_tinfo_events(hp)[0])
        print(
            f"[phase1] step-3 tinfo {i+1}/{len(hparams_list)} "
            f"in {time.perf_counter() - ti:.1f}s",
            flush=True,
        )
    print(f"[phase1] preprocessing took {time.perf_counter() - t0:.1f}s", flush=True)

    callback = FingerprintCallback(
        num_stacks=len(tinfos), out_path=out_dir / "fingerprint.json"
    )
    original_fit = install_fit_wrapper(callback, workdir)
    try:
        t1 = time.perf_counter()
        stacked_model = train_stacked_fvt(tinfos, None)
        train_seconds = time.perf_counter() - t1
    finally:
        StackedFvTClassifier.fit = original_fit
    print(f"[phase1] training took {train_seconds:.1f}s", flush=True)

    # ------------------------------------------------------------- probe set
    dm = stacked_model.datamodule
    x_val = dm.stacked_val_dataset.tensors[0]
    y_val = dm.stacked_val_dataset.tensors[1]
    w_val = dm.stacked_val_dataset.tensors[2]
    n_probe = min(args.probe_size, x_val.shape[0])
    probe_x = x_val[:n_probe].contiguous()

    probe_info = {
        "n_probe": int(n_probe),
        "val_rows": int(x_val.shape[0]),
        "train_rows": int(dm.stacked_train_dataset.tensors[0].shape[0]),
        "probe_x_digest": tensor_digest(probe_x),
        "val_y_digest": tensor_digest(y_val[:n_probe]),
        "val_w_digest": tensor_digest(w_val[:n_probe]),
    }

    stacked_model.eval()
    preds_final = np.zeros((n_probe, len(tinfos)), dtype=np.float64)
    for i in range(len(tinfos)):
        p = stacked_model.fvt_classifiers[i].predict(probe_x[:, i, :])
        preds_final[:, i] = p[:, 1].detach().cpu().numpy().astype(np.float64)

    # predictions from the checkpoints the saver selected as "best"
    ckpt_dir = workdir / "data" / "checkpoints"
    preds_best = np.zeros_like(preds_final)
    best_ckpt_digests = []
    for i, tinfo in enumerate(tinfos):
        path = ckpt_dir / f"{tinfo.hash}_best.pt"
        sd = torch.load(path, map_location="cpu")
        model = FvTClassifier(
            num_classes=2,
            dim_input_jet_features=4,
            dim_dijet_features=tinfo.hparams["dim_dijet_features"],
            dim_quadjet_features=tinfo.hparams["dim_quadjet_features"],
            run_name=tinfo.hash,
            depth=tinfo.hparams["depth"],
            repr_norm=tinfo.hparams["repr_norm"],
        )
        model.load_state_dict(sd)
        model.eval()
        model.to(stacked_model.device)
        p = model.predict(probe_x[:, i, :])
        preds_best[:, i] = p[:, 1].detach().cpu().numpy().astype(np.float64)
        best_ckpt_digests.append(
            tensor_digest(torch.cat([v.flatten().float() for _, v in sorted(sd.items())]))
        )

    np.save(out_dir / "probe_preds_final.npy", preds_final)
    np.save(out_dir / "probe_preds_best.npy", preds_best)

    callback.record["run"] = {
        "tag": args.tag,
        "seeds": seeds,
        "config_pattern": args.config_pattern,
        "source_configs": [c["_phase1_source_config"] for c in configs],
        "max_epochs": args.max_epochs,
        "train_seconds": train_seconds,
        "tinfo_hashes": [t.hash for t in tinfos],
        "ms_hash": tinfos[0].ms_hash,
        "ms_idx_digests": [
            tensor_digest(torch.from_numpy(t.ms_idx.astype(np.uint8))) for t in tinfos
        ],
        "hparams": [comparable_hparams(t.hparams) for t in tinfos],
    }
    callback.record["probe"] = probe_info
    callback.record["best_ckpt_digests"] = best_ckpt_digests
    callback.record["probe_preds_final_digest"] = tensor_digest(
        torch.from_numpy(preds_final)
    )
    callback.record["probe_preds_best_digest"] = tensor_digest(
        torch.from_numpy(preds_best)
    )
    callback.dump()

    if not args.keep_checkpoints:
        for p in ckpt_dir.glob("*_last.pt"):
            p.unlink()

    print(f"[phase1] wrote {out_dir / 'fingerprint.json'}", flush=True)


if __name__ == "__main__":
    main()
