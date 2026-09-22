"""Phase 5: benchmark attached-model count K on one L40.

Issue #2 asks for completed estimators per GPU-hour, seconds per epoch, peak
GPU and host memory, GPU utilisation and DataLoader wait, checkpoint size,
write and resume time, and prediction export time.

Runs the real training path (the Phase 3 resumable stack) for one K, for enough
epochs to reach the largest scheduled batch size (milestone 15 -> 32768, so 20
epochs), and extrapolates the 100-epoch group cost from the steady-state epochs.

    python phase5/benchmark.py --num-stacks 100
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import resource
import subprocess
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
SOHEUN = HERE.parent
for p in (str(SOHEUN), str(HERE), str(SOHEUN / "phase1"),
          str(SOHEUN / "phase2"), str(SOHEUN / "phase3")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(SOHEUN)

import numpy as np  # noqa: E402
import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from step_3_preprocessing import (  # noqa: E402
    check_and_get_CR_fvt_hparams,
    get_step_3_tinfo_events,
)
from stacked_fvt import StackedFvTClassifier  # noqa: E402
from train_stacked_fvts_from_tinfo import train_stacked_fvt  # noqa: E402

from fingerprint import env_record  # noqa: E402
from identity import identity_from_step3_config  # noqa: E402
from patches import install  # noqa: E402

# eta = 0.1, s_SR = 0.20: 300 configs, eps in {0.0, 0.01, 0.02} x 100 seeds.
# Ordered by (eps, seed) so that K=50 is a subset of K=100 is a subset of
# K=200, which isolates the effect of K from the effect of which data.
PATTERN = "configs/tmp/CR_fvt_training_ensemble_max_{seed}_{eps}_0_0.1_0.2_0.8.yml"
EPS_ORDER = ["0.0", "0.01", "0.02"]
SEEDS = range(100)


def config_paths(k: int) -> list[pathlib.Path]:
    out = []
    for eps in EPS_ORDER:
        for s in SEEDS:
            p = SOHEUN / PATTERN.format(seed=s, eps=eps)
            if p.exists():
                out.append(p)
            if len(out) == k:
                return out
    if len(out) < k:
        raise SystemExit(f"only {len(out)} configs available, need {k}")
    return out


class GpuSampler(threading.Thread):
    """Poll utilisation and memory for the one GPU slurm gave us."""

    def __init__(self, interval: float = 2.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.util: list[float] = []
        self.mem: list[float] = []
        self._stop = threading.Event()
        self.index = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]

    def run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "-i", self.index, "--query-gpu="
                     "utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                )
                u, m = out.stdout.strip().split(",")
                self.util.append(float(u))
                self.mem.append(float(m))
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


class EpochTimer(pl.Callback):
    """Per-epoch wall time, plus in-step compute time so the gap attributable
    to waiting on the DataLoader can be reported."""

    def __init__(self):
        self.epochs: list[dict] = []
        self._t0 = None
        self._batch_t0 = None
        self._compute = 0.0
        self._nbatch = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self._t0 = time.perf_counter()
        self._compute = 0.0
        self._nbatch = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._batch_t0 = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._batch_t0 is not None:
            self._compute += time.perf_counter() - self._batch_t0
        self._nbatch += 1

    def on_train_epoch_end(self, trainer, pl_module):
        wall = time.perf_counter() - self._t0
        dm = getattr(pl_module, "datamodule", None)
        self.epochs.append({
            "epoch": int(trainer.current_epoch),
            "wall_s": wall,
            "compute_s": self._compute,
            "wait_s": max(0.0, wall - self._compute),
            "num_batches": self._nbatch,
            "batch_size": int(dm.batch_size) if dm is not None else None,
        })


@contextlib.contextmanager
def pushd(path: pathlib.Path):
    prev = pathlib.Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def peak_rss_gb() -> float:
    me = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return (me + kids) / 1e6  # ru_maxrss is KB on Linux


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-stacks", type=int, required=True)
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--out-root", default="phase5_runs")
    ap.add_argument("--probe-size", type=int, default=20000)
    ap.add_argument(
        "--no-anomaly",
        action="store_true",
        help="neutralise the torch.autograd.set_detect_anomaly(True) that "
             "StackedFvTClassifier.fit turns on. Anomaly mode attaches a Python "
             "traceback to every autograd node and NaN-checks every backward "
             "output; a microbenchmark puts it at 4.9x on the training step. "
             "It is a debugging instrument, so disabling it does not change "
             "results at all.",
    )
    ap.add_argument("--tag", default="", help="suffix for the output directory")
    args = ap.parse_args()

    # Patch rather than edit: intercept the call fit() makes and force it off.
    if args.no_anomaly:
        _orig_set = torch.autograd.set_detect_anomaly

        def _forced_off(mode=True, check_nan=True):
            return _orig_set(False, False)

        torch.autograd.set_detect_anomaly = _forced_off
        torch.autograd.set_detect_anomaly(False)
        print("[phase5] anomaly detection forced OFF", flush=True)

    K = args.num_stacks
    name = f"K{K:03d}" + (f"_{args.tag}" if args.tag else "")
    out_dir = (SOHEUN / args.out_root / name).resolve()
    workdir = out_dir / "workdir"
    workdir.mkdir(parents=True, exist_ok=True)

    print(f"[phase5] K={K} max_epochs={args.max_epochs}", flush=True)
    print(f"[phase5] env={json.dumps(env_record())}", flush=True)

    paths = config_paths(K)
    configs = []
    for p in paths:
        with open(p) as f:
            cfg = yaml.safe_load(f)
        cfg["CR_fvt"]["max_epochs"] = args.max_epochs
        configs.append(cfg)
    identities = [identity_from_step3_config(c) for c in configs]
    hparams_list = [check_and_get_CR_fvt_hparams(c) for c in configs]

    t_pre = time.perf_counter()
    tinfos = []
    for i, hp in enumerate(hparams_list):
        tinfos.append(get_step_3_tinfo_events(hp)[0])
        if (i + 1) % 25 == 0 or i + 1 == K:
            print(f"[phase5] preprocessed {i+1}/{K} "
                  f"({time.perf_counter()-t_pre:.0f}s)", flush=True)
    preprocess_s = time.perf_counter() - t_pre
    rss_after_pre = peak_rss_gb()

    timer = EpochTimer()
    sampler = GpuSampler()
    torch.cuda.reset_peak_memory_stats()
    uninstall = install(
        ckpt_dir=out_dir,
        individual_models_dir=out_dir / "individual_models",
        run_names=[i.fingerprint for i in identities],
        shuffle_seed=int(hparams_list[0]["train_seed"]),
    )
    orig_fit = StackedFvTClassifier.fit

    def wrapped_fit(self, *a, **kw):
        kw["callbacks"] = list(kw.get("callbacks") or []) + [timer]
        with pushd(workdir):
            return orig_fit(self, *a, **kw)

    StackedFvTClassifier.fit = wrapped_fit
    sampler.start()
    try:
        t0 = time.perf_counter()
        model = train_stacked_fvt(tinfos, None)
        train_s = time.perf_counter() - t0
    finally:
        sampler.stop()
        StackedFvTClassifier.fit = orig_fit
        uninstall()

    peak_alloc = torch.cuda.max_memory_allocated() / 1e9
    peak_reserved = torch.cuda.max_memory_reserved() / 1e9

    # ---------------------------------------------------- checkpoint metrics
    ckpt = out_dir / "last.ckpt"
    ckpt_bytes = ckpt.stat().st_size if ckpt.exists() else 0
    t0 = time.perf_counter()
    if ckpt.exists():
        loaded = torch.load(ckpt, map_location="cpu")
        load_s = time.perf_counter() - t0
        del loaded
    else:
        load_s = float("nan")

    # ------------------------------------------------- prediction export
    dm = model.datamodule
    x_val = dm.stacked_val_dataset.tensors[0]
    n_probe = min(args.probe_size, x_val.shape[0])
    probe_x = x_val[:n_probe].contiguous()
    model.eval()
    t0 = time.perf_counter()
    preds = np.zeros((n_probe, K), dtype=np.float32)
    for pos in range(K):
        p = model.fvt_classifiers[pos].predict(probe_x[:, pos, :])
        preds[:, pos] = p[:, 1].detach().cpu().numpy()
    export_s = time.perf_counter() - t0

    # ---------------------------------------------------------- accounting
    eps = timer.epochs
    steady = [e for e in eps if e["batch_size"] == max(
        (x["batch_size"] or 0) for x in eps)]
    steady_s = float(np.mean([e["wall_s"] for e in steady])) if steady else float("nan")
    measured_epochs = len(eps)
    # a 100-epoch group: the measured schedule, then 80 more steady epochs
    projected_100 = train_s + max(0, 100 - measured_epochs) * steady_s
    per_gpu_hour = K / (projected_100 / 3600.0)

    record = {
        "K": K,
        "anomaly_detection": not args.no_anomaly,
        "tag": args.tag,
        "max_epochs": args.max_epochs,
        "env": env_record(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "train_rows": int(dm.stacked_train_dataset.tensors[0].shape[0]),
        "val_rows": int(x_val.shape[0]),
        "preprocess_s": preprocess_s,
        "train_s": train_s,
        "epochs": eps,
        "steady_epoch_s": steady_s,
        "steady_batch_size": steady[0]["batch_size"] if steady else None,
        "projected_100epoch_group_s": projected_100,
        "estimators_per_gpu_hour": per_gpu_hour,
        "peak_gpu_allocated_gb": peak_alloc,
        "peak_gpu_reserved_gb": peak_reserved,
        "peak_host_rss_gb": peak_rss_gb(),
        "host_rss_after_preprocess_gb": rss_after_pre,
        "gpu_util_mean": float(np.mean(sampler.util)) if sampler.util else None,
        "gpu_util_p90": float(np.quantile(sampler.util, 0.9)) if sampler.util else None,
        "gpu_mem_used_max_mb": float(np.max(sampler.mem)) if sampler.mem else None,
        "gpu_samples": len(sampler.util),
        "dataloader_wait_frac": (
            sum(e["wait_s"] for e in eps) / sum(e["wall_s"] for e in eps)
            if eps else None
        ),
        "checkpoint_bytes": ckpt_bytes,
        "checkpoint_load_s": load_s,
        "prediction_export_s": export_s,
        "n_probe": int(n_probe),
    }
    tmp = out_dir / "benchmark.json.tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
    os.replace(tmp, out_dir / "benchmark.json")

    print(f"\n[phase5] K={K}: {train_s:.0f}s for {measured_epochs} epochs, "
          f"steady {steady_s:.1f}s/epoch, "
          f"projected 100-epoch group {projected_100/3600:.2f} h, "
          f"{per_gpu_hour:.1f} estimators/GPU-hour", flush=True)
    print(f"[phase5] peak GPU {peak_alloc:.2f} GB alloc / {peak_reserved:.2f} GB "
          f"reserved, peak host RSS {record['peak_host_rss_gb']:.1f} GB, "
          f"GPU util mean {record['gpu_util_mean']}%", flush=True)


if __name__ == "__main__":
    main()
