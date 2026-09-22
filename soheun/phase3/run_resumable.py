"""Phase 3: run a stacked CR group with a resumable checkpoint.

Layout per issue #2:

    training_runs/<campaign_id>/<group_id>/
      manifest.json
      last.ckpt
      completion.json
      individual_models/
      predictions/
      metrics/
      logs/

``completion.json`` is written only after predictions are exported, so a group
that died between the last checkpoint and its exports is not mistaken for a
finished one.

    # uninterrupted reference
    python phase3/run_resumable.py --group-id whole --max-epochs 20

    # interrupt after epoch 7, then resume
    python phase3/run_resumable.py --group-id split --max-epochs 20 --stop-after-epoch 7
    python phase3/run_resumable.py --group-id split --max-epochs 20 --resume
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
SOHEUN = HERE.parent
for p in (str(SOHEUN), str(HERE), str(SOHEUN / "phase1"), str(SOHEUN / "phase2")):
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
from identity import identity_from_step3_config  # noqa: E402
from patches import install  # noqa: E402
from resumable import atomic_torch_save, group_dirs  # noqa: E402

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


def digest_obj(obj) -> str:
    """Stable digest of an optimizer/scheduler state_dict."""
    h = hashlib.sha256()

    def walk(x):
        if isinstance(x, torch.Tensor):
            a = x.detach().cpu().contiguous()
            h.update(str(tuple(a.shape)).encode())
            h.update(a.numpy().tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=repr):
                h.update(repr(k).encode())
                walk(x[k])
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
        else:
            h.update(repr(x).encode())

    walk(obj)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-id", default="phase3_resume")
    ap.add_argument("--group-id", required=True)
    ap.add_argument("--out-root", default="training_runs")
    ap.add_argument("--num-stacks", type=int, default=5)
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--stop-after-epoch", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--probe-size", type=int, default=20000)
    ap.add_argument(
        "--lr-patience",
        type=int,
        default=None,
        help="override ReduceLROnPlateau patience. Test-only: the shipped "
             "config has patience 10, which never fires a reduction inside "
             "20 epochs, so a resume test cannot show that a reduction "
             "survives a restart. Both sides of a comparison must use the "
             "same value.",
    )
    args = ap.parse_args()

    dirs = group_dirs(SOHEUN / args.out_root, args.campaign_id, args.group_id)
    base = dirs["base"]
    last_ckpt = base / "last.ckpt"
    workdir = base / "workdir"
    workdir.mkdir(parents=True, exist_ok=True)

    resume_from = None
    if args.resume:
        if not last_ckpt.exists():
            raise SystemExit(f"--resume given but {last_ckpt} does not exist")
        resume_from = last_ckpt
        print(f"[phase3] resuming from {last_ckpt}", flush=True)

    seeds = list(range(args.num_stacks))
    configs = []
    for s in seeds:
        with open(SOHEUN / CONFIG_PATTERN.format(seed=s)) as f:
            cfg = yaml.safe_load(f)
        cfg["CR_fvt"]["max_epochs"] = args.max_epochs
        if args.lr_patience is not None:
            cfg["CR_fvt"]["lr_scheduler"]["patience"] = args.lr_patience
        configs.append(cfg)

    identities = [identity_from_step3_config(c) for c in configs]
    hparams_list = [check_and_get_CR_fvt_hparams(c) for c in configs]

    tinfos = []
    for i, hp in enumerate(hparams_list):
        t0 = time.perf_counter()
        tinfos.append(get_step_3_tinfo_events(hp)[0])
        print(
            f"[phase3] step-3 tinfo {i+1}/{len(hparams_list)} "
            f"in {time.perf_counter()-t0:.1f}s",
            flush=True,
        )

    part = len(list(dirs["metrics"].glob("fingerprint.part*.json")))
    callback = FingerprintCallback(
        num_stacks=len(tinfos),
        out_path=dirs["metrics"] / f"fingerprint.part{part}.json",
    )

    manifest = {
        "campaign_id": args.campaign_id,
        "group_id": args.group_id,
        "seeds": seeds,
        "max_epochs": args.max_epochs,
        "num_stacks": args.num_stacks,
        "env": env_record(),
        "identities": [
            {
                "position": k,
                "fingerprint": ident.fingerprint,
                "mother_sample_seed": ident.mother_sample_seed,
                "canonical": ident.canonical(),
            }
            for k, ident in enumerate(identities)
        ],
        "tinfo_hashes": [t.hash for t in tinfos],
        "started_parts": part + 1,
        "resumed": bool(args.resume),
        "stop_after_epoch": args.stop_after_epoch,
        "lr_patience_override": args.lr_patience,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    manifest_path = base / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            previous = json.load(f)
        prev_ids = [d["fingerprint"] for d in previous.get("identities", [])]
        now_ids = [i.fingerprint for i in identities]
        if prev_ids and prev_ids != now_ids:
            raise SystemExit(
                "refusing to continue: this group was started with a different "
                f"set or order of estimators.\n  was: {prev_ids}\n  now: {now_ids}"
            )
        manifest["history"] = previous.get("history", []) + [
            {k: previous.get(k) for k in ("started_at", "resumed", "slurm_job_id",
                                          "stop_after_epoch")}
        ]
    tmp_manifest = base / "manifest.json.tmp"
    with open(tmp_manifest, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(tmp_manifest, manifest_path)

    uninstall = install(
        ckpt_dir=base,
        individual_models_dir=dirs["individual_models"],
        run_names=[i.fingerprint for i in identities],
        shuffle_seed=int(hparams_list[0]["train_seed"]),
        resume_from=resume_from,
        stop_after_epoch=args.stop_after_epoch,
    )
    orig_fit = StackedFvTClassifier.fit

    def wrapped_fit(self, *a, **kw):
        kw["callbacks"] = list(kw.get("callbacks") or []) + [callback]
        with pushd(workdir):
            return orig_fit(self, *a, **kw)

    StackedFvTClassifier.fit = wrapped_fit
    try:
        t1 = time.perf_counter()
        stacked_model = train_stacked_fvt(tinfos, None)
        train_seconds = time.perf_counter() - t1
    finally:
        StackedFvTClassifier.fit = orig_fit
        uninstall()

    print(f"[phase3] training segment took {train_seconds:.1f}s", flush=True)

    # optimizer / scheduler state, which the resume test has to match
    trainer = getattr(stacked_model, "trainer", None)
    opt_digests, sched_states = [], []
    if trainer is not None:
        for opt in trainer.optimizers:
            opt_digests.append(digest_obj(opt.state_dict()))
        for cfg in trainer.lr_scheduler_configs:
            s = cfg.scheduler.state_dict()
            sched_states.append({k: v for k, v in s.items() if k != "_last_lr"})
    callback.record["phase3"] = {
        "group_id": args.group_id,
        "part": part,
        "resumed": bool(args.resume),
        "stop_after_epoch": args.stop_after_epoch,
        "train_seconds": train_seconds,
        "optimizer_state_digests": opt_digests,
        "scheduler_states": sched_states,
        "lrs": [float(o.param_groups[0]["lr"]) for o in (trainer.optimizers if trainer else [])],
        "identity_fingerprints": [i.fingerprint for i in identities],
    }
    callback.dump()

    if args.stop_after_epoch is not None:
        print(
            f"[phase3] stopped after epoch {args.stop_after_epoch}; "
            f"no completion.json written",
            flush=True,
        )
        return

    # ------------------------------------------------------------- exports
    dm = stacked_model.datamodule
    x_val = dm.stacked_val_dataset.tensors[0]
    n_probe = min(args.probe_size, x_val.shape[0])
    probe_x = x_val[:n_probe].contiguous()

    stacked_model.eval()
    preds = {}
    for pos, ident in enumerate(identities):
        p = stacked_model.fvt_classifiers[pos].predict(probe_x[:, pos, :])
        preds[ident.fingerprint] = p[:, 1].detach().cpu().numpy().astype(np.float64)
    np.savez(dirs["predictions"] / "probe_preds_by_identity.npz", **preds)

    final_digests = {
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
    }

    completion = {
        "group_id": args.group_id,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "parts": part + 1,
        "max_epochs": args.max_epochs,
        "final_digest_by_identity": final_digests,
        "optimizer_state_digests": opt_digests,
        "scheduler_states": sched_states,
        "probe_x_digest_by_position": [
            tensor_digest(probe_x[:, pos, :]) for pos in range(len(identities))
        ],
        "n_probe": int(n_probe),
    }
    tmp = base / "completion.json.tmp"
    with open(tmp, "w") as f:
        json.dump(completion, f, indent=2, sort_keys=True)
    os.replace(tmp, base / "completion.json")
    print(f"[phase3] wrote {base / 'completion.json'}", flush=True)


if __name__ == "__main__":
    main()
