"""Phase 5 report: pick a stack size, and say whether K=300 is worth trying.

Issue #2: "Choose the stack size maximizing completed estimators per GPU-hour
subject to safe memory, checkpoint latency, and one logical configuration per
stack."

    python phase5/report.py phase5_runs
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

SIGNIFICANT = 0.10  # relative gain that would justify testing a larger K


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=pathlib.Path, nargs="?",
                    default=pathlib.Path("phase5_runs"))
    ap.add_argument("--host-mem-gb", type=float, default=56.0,
                    help="the --mem the job requested, for a headroom check")
    args = ap.parse_args()

    runs = []
    for d in sorted(args.root.glob("K*")):
        f = d / "benchmark.json"
        if f.exists():
            runs.append(json.load(open(f)))
    if not runs:
        print(f"no benchmark.json under {args.root}")
        return 1
    runs.sort(key=lambda r: r["K"])

    print("Phase 5: attached-model count benchmark")
    print(f"  gpu  : {runs[0]['env'].get('gpu_name')}")
    print(f"  rows : {runs[0]['train_rows']:,} train / {runs[0]['val_rows']:,} val"
          f"  per estimator")
    print(f"  epochs measured: {runs[0]['max_epochs']}"
          f"  (100-epoch group cost is extrapolated from steady-state epochs)\n")

    hdr = (f"{'K':>4} {'est/GPU-h':>10} {'100ep grp':>10} {'steady s/ep':>12} "
           f"{'GPU util':>9} {'wait':>6} {'GPU GB':>8} {'host GB':>8} "
           f"{'ckpt MB':>8} {'load s':>7} {'export s':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in runs:
        print(f"{r['K']:>4} {r['estimators_per_gpu_hour']:>10.1f} "
              f"{r['projected_100epoch_group_s']/3600:>9.2f}h "
              f"{r['steady_epoch_s']:>12.1f} "
              f"{(r['gpu_util_mean'] or 0):>8.0f}% "
              f"{(r['dataloader_wait_frac'] or 0)*100:>5.0f}% "
              f"{r['peak_gpu_reserved_gb']:>8.2f} "
              f"{r['peak_host_rss_gb']:>8.1f} "
              f"{r['checkpoint_bytes']/1e6:>8.1f} "
              f"{r['checkpoint_load_s']:>7.1f} "
              f"{r['prediction_export_s']:>9.1f}")

    print("\nthroughput relative to the smallest K measured:")
    base = runs[0]
    for r in runs:
        rel = r["estimators_per_gpu_hour"] / base["estimators_per_gpu_hour"]
        print(f"  K={r['K']:<4} {rel:5.2f}x")

    best = max(runs, key=lambda r: r["estimators_per_gpu_hour"])
    print(f"\nbest measured: K={best['K']} at "
          f"{best['estimators_per_gpu_hour']:.1f} estimators/GPU-hour")

    # memory headroom.  Host RSS is fixed + marginal, not proportional: at
    # small K it is dominated by per-tinfo DataFrame loading, and only the
    # stacked (N, K, 16) tensors scale with K.  Dividing total RSS by K would
    # therefore wildly over-project a larger K.
    print("\nmemory headroom:")
    for r in runs:
        frac = r["peak_host_rss_gb"] / args.host_mem_gb
        flag = "  <-- tight" if frac > 0.8 else ""
        print(f"  K={r['K']:<4} host {r['peak_host_rss_gb']:5.1f} GB of "
              f"{args.host_mem_gb:.0f} GB ({frac*100:.0f}%){flag}")

    fixed = marginal = None
    if len(runs) >= 2:
        ks = [r["K"] for r in runs]
        ms = [r["peak_host_rss_gb"] for r in runs]
        kbar, mbar = sum(ks) / len(ks), sum(ms) / len(ms)
        denom = sum((k - kbar) ** 2 for k in ks)
        if denom > 0:
            marginal = sum((k - kbar) * (m - mbar) for k, m in zip(ks, ms)) / denom
            fixed = mbar - marginal * kbar
            print(f"\n  fit: RSS ~ {fixed:.1f} GB fixed + "
                  f"{marginal*1000:.0f} MB per estimator")
            if marginal <= 0:
                print("  marginal cost is not resolvable at these K; the runs "
                      "are all in the fixed-overhead regime, so treat any "
                      "projection to a larger K as unknown rather than small.")
            else:
                for kk in (200, 300):
                    if kk > max(ks):
                        proj = fixed + marginal * kk
                        print(f"  projected at K={kk}: {proj:.0f} GB "
                              f"({'fits' if proj < 0.8*args.host_mem_gb else 'needs more'}"
                              f" in {args.host_mem_gb:.0f} GB)")

    # should we try a larger K?
    print("\nverdict:")
    if len(runs) >= 2:
        top, prev = runs[-1], runs[-2]
        gain = (top["estimators_per_gpu_hour"] / prev["estimators_per_gpu_hour"]) - 1
        print(f"  K={top['K']} vs K={prev['K']}: {gain*100:+.1f}% throughput")
        if gain >= SIGNIFICANT:
            print(f"  still improving by more than {SIGNIFICANT*100:.0f}% at the "
                  f"largest K measured, so K=300 is worth testing.")
            if fixed is not None and marginal and marginal > 0:
                proj = fixed + marginal * 300
                print(f"  projected host RSS at K=300: {proj:.0f} GB "
                      f"({'fits' if proj < 0.8*args.host_mem_gb else 'needs more'} "
                      f"in {args.host_mem_gb:.0f} GB)")
        else:
            print(f"  gain is below {SIGNIFICANT*100:.0f}%, so K={top['K']} is at "
                  f"or past the knee. Testing K=300 is not justified on "
                  f"throughput grounds.")
    if best["gpu_util_mean"] is not None and best["gpu_util_mean"] < 50:
        print(f"\n  NOTE: GPU utilisation is only {best['gpu_util_mean']:.0f}% at "
              f"the best K. The per-batch Python loop over estimators, not the "
              f"stack size, is then the limit; vectorising that loop would be a "
              f"larger win than any choice of K.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
