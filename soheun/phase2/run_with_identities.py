"""Phase 2 end-to-end: train a stack whose initialization comes from identity.

Issue #2 asks that reordering a stack change neither initial parameters nor
predictions. The unit gate in ``test_identity.py`` covers initial parameters.
This runner covers predictions, by training the same five estimators twice in
opposite stack orders and keying every result by identity instead of position.

Like the Phase 1 runner it edits no production file. The wrapper around
``StackedFvTClassifier.fit`` now does three things: apply identities to the
freshly constructed stack, attach the Phase 1 fingerprint callback, and chdir
into a scratch workdir so ``./data/checkpoints`` and ``./tb_logs`` land there.

    python phase2/run_with_identities.py --tag fwd --order forward
    python phase2/run_with_identities.py --tag rev --order reverse
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
for p in (str(SOHEUN), str(HERE), str(SOHEUN / "phase1")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(SOHEUN)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from step_3_preprocessing import (  # noqa: E402
    check_and_get_CR_fvt_hparams,
    get_step_3_tinfo_events,
)
from stacked_fvt import StackedFvTClassifier  # noqa: E402
from train_stacked_fvts_from_tinfo import train_stacked_fvt  # noqa: E402

from fingerprint import FingerprintCallback, env_record, tensor_digest  # noqa: E402
from identity import group_fingerprint, identity_from_step3_config  # noqa: E402
from init_from_identity import apply_identities  # noqa: E402

CONFIG_PATTERN = (
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


def install_fit_wrapper(identities, hparams_list, callback, workdir: pathlib.Path):
    original_fit = StackedFvTClassifier.fit
    applied: dict = {}

    def wrapped_fit(self, *args, **kwargs):
        # identity-derived initialization, before any optimizer is built
        applied["seeds"] = apply_identities(self, identities, hparams_list)
        applied["init_digests"] = [
            tensor_digest(
                torch.cat(
                    [v.flatten().float() for _, v in sorted(m.state_dict().items())]
                )
            )
            for m in self.fvt_classifiers
        ]
        kwargs["callbacks"] = list(kwargs.get("callbacks") or []) + [callback]
        with pushd(workdir):
            return original_fit(self, *args, **kwargs)

    StackedFvTClassifier.fit = wrapped_fit
    return original_fit, applied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--order", choices=["forward", "reverse"], default="forward")
    ap.add_argument("--out-root", default="phase2_runs/perm")
    ap.add_argument("--num-stacks", type=int, default=5)
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--probe-size", type=int, default=20000)
    args = ap.parse_args()

    seeds = list(range(args.num_stacks))
    if args.order == "reverse":
        seeds = list(reversed(seeds))

    out_dir = (SOHEUN / args.out_root / args.tag).resolve()
    workdir = out_dir / "workdir"
    workdir.mkdir(parents=True, exist_ok=True)

    print(f"[phase2] tag={args.tag} order={args.order} seeds={seeds}")
    print(f"[phase2] env={json.dumps(env_record())}", flush=True)

    configs = []
    for s in seeds:
        with open(SOHEUN / CONFIG_PATTERN.format(seed=s)) as f:
            cfg = yaml.safe_load(f)
        cfg["CR_fvt"]["max_epochs"] = args.max_epochs
        configs.append(cfg)

    identities = [identity_from_step3_config(c) for c in configs]
    hparams_list = [check_and_get_CR_fvt_hparams(c) for c in configs]

    print(f"[phase2] group fingerprint = {group_fingerprint(identities)}")
    for pos, ident in enumerate(identities):
        print(
            f"[phase2]   pos {pos}: ms_seed={ident.mother_sample_seed} "
            f"id={ident.fingerprint[:16]} init_seed={ident.seed('model_init')}"
        )

    tinfos = []
    for i, hp in enumerate(hparams_list):
        t0 = time.perf_counter()
        tinfos.append(get_step_3_tinfo_events(hp)[0])
        print(
            f"[phase2] step-3 tinfo {i+1}/{len(hparams_list)} "
            f"in {time.perf_counter()-t0:.1f}s",
            flush=True,
        )

    callback = FingerprintCallback(
        num_stacks=len(tinfos), out_path=out_dir / "fingerprint.json"
    )
    original_fit, applied = install_fit_wrapper(
        identities, hparams_list, callback, workdir
    )
    try:
        t1 = time.perf_counter()
        stacked_model = train_stacked_fvt(tinfos, None)
        train_seconds = time.perf_counter() - t1
    finally:
        StackedFvTClassifier.fit = original_fit
    print(f"[phase2] training took {train_seconds:.1f}s", flush=True)

    dm = stacked_model.datamodule
    x_val = dm.stacked_val_dataset.tensors[0]
    n_probe = min(args.probe_size, x_val.shape[0])
    probe_x = x_val[:n_probe].contiguous()

    stacked_model.eval()
    preds: dict[str, np.ndarray] = {}
    for pos, ident in enumerate(identities):
        p = stacked_model.fvt_classifiers[pos].predict(probe_x[:, pos, :])
        preds[ident.fingerprint] = p[:, 1].detach().cpu().numpy().astype(np.float64)
    np.savez(out_dir / "probe_preds_by_identity.npz", **preds)

    callback.record["phase2"] = {
        "tag": args.tag,
        "order": args.order,
        "seeds": seeds,
        "group_fingerprint": group_fingerprint(identities),
        "identities": [
            {
                "position": pos,
                "fingerprint": ident.fingerprint,
                "mother_sample_seed": ident.mother_sample_seed,
                "seeds": ident.seeds(),
                "canonical": ident.canonical(),
            }
            for pos, ident in enumerate(identities)
        ],
        "applied_init_seeds": applied.get("seeds"),
        "init_digest_by_identity": {
            identities[pos].fingerprint: d
            for pos, d in enumerate(applied.get("init_digests", []))
        },
        "final_digest_by_identity": {
            identities[pos].fingerprint: tensor_digest(
                torch.cat(
                    [
                        v.flatten().float()
                        for _, v in sorted(
                            stacked_model.fvt_classifiers[pos].state_dict().items()
                        )
                    ]
                )
            )
            for pos in range(len(identities))
        },
        "probe_x_digest_by_position": [
            tensor_digest(probe_x[:, pos, :]) for pos in range(len(identities))
        ],
        "train_seconds": train_seconds,
        "n_probe": int(n_probe),
    }
    callback.dump()
    print(f"[phase2] wrote {out_dir / 'fingerprint.json'}", flush=True)


if __name__ == "__main__":
    main()
