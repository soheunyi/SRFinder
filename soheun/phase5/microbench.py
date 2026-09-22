"""Phase 5 follow-up: where does the per-batch time actually go?

Phase 5 measured 18-24% GPU utilisation at 1.2 GB of a 46 GB card, so the
stack size is not what limits throughput. This isolates the candidates on
synthetic data of the production shape, without touching any production code:

  a) baseline          the current loop, with set_detect_anomaly(True) as
                       stacked_fvt.fit leaves it
  b) no anomaly        the same loop with anomaly detection off
  c) compiled          b) plus torch.compile on the per-model step
  d) vmap ensemble     one vmapped forward/backward over stacked parameters

(a) vs (b) prices a one-line change. (b) vs (d) prices the redesign.

    python phase5/microbench.py --num-stacks 20 --batch-size 32768
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
SOHEUN = HERE.parent
for p in (str(SOHEUN), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(SOHEUN)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from fvt_classifier import FvTClassifier  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build(k: int) -> list[FvTClassifier]:
    models = []
    for i in range(k):
        torch.manual_seed(1000 + i)
        m = FvTClassifier(
            num_classes=2,
            dim_input_jet_features=4,
            dim_dijet_features=6,
            dim_quadjet_features=6,
            run_name=f"micro{i}",
            device=DEV,
            depth={"encoder": 4, "decoder": 1},
        ).to(DEV)
        m.train()
        models.append(m)
    return models


def timed(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def loop_step(models, opts, x, y, w):
    def step():
        for i, m in enumerate(models):
            opts[i].zero_grad(set_to_none=True)
            logits = m(x[:, i, :])
            loss = (F.cross_entropy(logits, y[:, i], reduction="none") * w[:, i]).mean()
            loss.backward()
            opts[i].step()
    return step


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-stacks", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32768)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default="phase5_runs/microbench.json")
    args = ap.parse_args()

    K, B = args.num_stacks, args.batch_size
    torch.manual_seed(0)
    x = torch.randn(B, K, 16, device=DEV)
    y = torch.randint(0, 2, (B, K), device=DEV)
    w = torch.rand(B, K, device=DEV) + 0.5

    print(f"K={K}  batch={B}  device={DEV}")
    if DEV.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    results: dict[str, float] = {}

    # ------------------------------------------------ a) baseline, anomaly on
    models = build(K)
    opts = [torch.optim.Adam(m.parameters(), lr=0.01) for m in models]
    torch.autograd.set_detect_anomaly(True)
    results["a_baseline_anomaly_on"] = timed(
        loop_step(models, opts, x, y, w), args.warmup, args.iters
    )
    torch.autograd.set_detect_anomaly(False)
    print(f"  a) loop, anomaly ON      {results['a_baseline_anomaly_on']*1000:9.1f} ms/batch")

    # ------------------------------------------------------ b) anomaly off
    results["b_anomaly_off"] = timed(
        loop_step(models, opts, x, y, w), args.warmup, args.iters
    )
    speedup = results["a_baseline_anomaly_on"] / results["b_anomaly_off"]
    print(f"  b) loop, anomaly OFF     {results['b_anomaly_off']*1000:9.1f} ms/batch"
          f"   {speedup:.2f}x vs (a)")

    # ------------------------------------------------------- c) torch.compile
    try:
        compiled = [torch.compile(m, dynamic=False) for m in models]
        results["c_compiled"] = timed(
            loop_step(compiled, opts, x, y, w), args.warmup, args.iters
        )
        print(f"  c) + torch.compile       {results['c_compiled']*1000:9.1f} ms/batch"
              f"   {results['b_anomaly_off']/results['c_compiled']:.2f}x vs (b)")
    except Exception as e:  # noqa: BLE001
        results["c_compiled"] = None
        results["c_error"] = f"{type(e).__name__}: {e}"
        print(f"  c) torch.compile FAILED  {type(e).__name__}: {str(e)[:160]}")

    # --------------------------------------------------------- d) vmap ensemble
    try:
        from torch.func import functional_call, stack_module_state, vmap

        params, buffers = stack_module_state(models)
        base = build(1)[0].to("meta")

        def one(p, b, xi, yi, wi):
            logits = functional_call(base, (p, b), (xi,))
            return (F.cross_entropy(logits, yi, reduction="none") * wi).mean()

        xs = x.permute(1, 0, 2).contiguous()   # (K, B, 16)
        ys = y.permute(1, 0).contiguous()
        ws = w.permute(1, 0).contiguous()
        flat = [v for v in params.values()]
        for v in flat:
            v.requires_grad_(True)
        ens_opt = torch.optim.Adam(flat, lr=0.01, foreach=True)

        def step():
            ens_opt.zero_grad(set_to_none=True)
            losses = vmap(one)(params, buffers, xs, ys, ws)
            losses.sum().backward()
            ens_opt.step()

        results["d_vmap"] = timed(step, args.warmup, args.iters)
        print(f"  d) vmap ensemble         {results['d_vmap']*1000:9.1f} ms/batch"
              f"   {results['b_anomaly_off']/results['d_vmap']:.2f}x vs (b)")
    except Exception as e:  # noqa: BLE001
        results["d_vmap"] = None
        results["d_error"] = f"{type(e).__name__}: {e}"
        print(f"  d) vmap FAILED           {type(e).__name__}: {str(e)[:300]}")
        print("     (a failure here is itself the finding: the model as written "
              "is not vmap-compatible, most likely the in-place GhostBatchNorm "
              "buffer updates)")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"K": K, "batch": B, "results": results,
                   "gpu": torch.cuda.get_device_name(0) if DEV.type == "cuda" else None},
                  f, indent=2, sort_keys=True)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
