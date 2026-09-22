"""Is the jet-interleaving slice pattern the bottleneck, and does a reshape fix it?

Profiling a single FvT training step showed ~11,253 CUDA ops for a
2,024-parameter model, with SliceBackward0 alone at 31% of CPU time and real
convolution work at 12% of GPU time. The source is the interleave in the
reinforce layers, which is expressed as many slices plus a cat:

    DijetReinforceLayer    12 slices + cat -> |j0|j1|d0|j2|j3|d1|...
    QuadjetReinforceLayer   9 slices + cat -> |sym0|anti0|q0|sym1|...

Both are pure reshapes. Each slice's backward allocates a zeros tensor of the
full input and scatters into it; a cat's backward is a cheap split.

This script verifies the rewrite is numerically identical (outputs *and*
gradients) and then measures it, on the real model.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, "/home/export/soheuny/SRFinder/soheun")
os.chdir("/home/export/soheuny/SRFinder/soheun")

import torch
import torch.nn.functional as F

import network_blocks as nb
from fvt_classifier import FvTClassifier

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ rewrites

def dijet_forward_fast(self, j, d):
    """|j0|j1|d0|j2|j3|d1|... as (6,2) jets interleaved with (6,1) dijets.

    position 3k = j[2k], 3k+1 = j[2k+1], 3k+2 = d[k] -- the original order.
    """
    n = j.shape[0]
    d = torch.cat(
        (j.reshape(n, self.dim_d, 6, 2), d.unsqueeze(-1)), dim=3
    ).reshape(n, self.dim_d, 18)
    return self.conv(d)


def quadjet_forward_fast(self, d, q):
    """|sym0|anti0|q0|sym1|anti1|q1|... as three (.,3) tensors stacked."""
    n = d.shape[0]
    d_sym = self.sym(d)
    d_antisym = torch.abs(self.antisym(d))
    q = torch.stack((d_sym, d_antisym, q), dim=3).reshape(n, self.dim_q, 9)
    return self.conv(q)


ORIG = {
    "dijet": nb.DijetReinforceLayer.forward,
    "quadjet": nb.QuadjetReinforceLayer.forward,
}


def patch():
    nb.DijetReinforceLayer.forward = dijet_forward_fast
    nb.QuadjetReinforceLayer.forward = quadjet_forward_fast


def unpatch():
    nb.DijetReinforceLayer.forward = ORIG["dijet"]
    nb.QuadjetReinforceLayer.forward = ORIG["quadjet"]


# --------------------------------------------------------------------- model

def build(seed: int = 0) -> FvTClassifier:
    torch.manual_seed(seed)
    return FvTClassifier(
        num_classes=2, dim_input_jet_features=4, dim_dijet_features=6,
        dim_quadjet_features=6, run_name="sr", device=DEV,
        depth={"encoder": 4, "decoder": 1},
    ).to(DEV).train()


def fwd_bwd(m, x, y):
    m.zero_grad(set_to_none=True)
    loss = F.cross_entropy(m(x), y)
    loss.backward()
    return loss


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32768)
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    torch.set_float32_matmul_precision("medium")
    B = args.batch
    torch.manual_seed(7)
    x = torch.randn(B, 16, device=DEV)
    y = torch.randint(0, 2, (B,), device=DEV)

    # ------------------------------------------------ 1. numerical equivalence
    # eval mode: GhostBatchNorm mutates running stats in train mode, so two
    # forward passes would not be comparable without resetting them.
    unpatch()
    a = build(0).eval()
    with torch.no_grad():
        out_a = a(x).clone()
    patch()
    b = build(0).eval()
    with torch.no_grad():
        out_b = b(x).clone()
    unpatch()
    dmax = (out_a - out_b).abs().max().item()
    print(f"forward equivalence      max|diff| = {dmax:.3e}  "
          f"({'EXACT' if dmax == 0 else 'differs'})")

    # gradients too
    unpatch()
    a = build(0).train()
    fwd_bwd(a, x, y)
    ga = torch.cat([p.grad.flatten() for p in a.parameters() if p.grad is not None])
    patch()
    b = build(0).train()
    fwd_bwd(b, x, y)
    gb = torch.cat([p.grad.flatten() for p in b.parameters() if p.grad is not None])
    unpatch()
    gmax = (ga - gb).abs().max().item()
    print(f"gradient equivalence     max|diff| = {gmax:.3e}  "
          f"({'EXACT' if gmax == 0 else 'differs'})")

    # ------------------------------------------------------- 2. op count + time
    from torch.profiler import ProfilerActivity, profile

    def measure(label: str):
        m = build(0)
        opt = torch.optim.Adam(m.parameters(), lr=0.01)

        def step():
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(m(x), y).backward()
            opt.step()

        for _ in range(3):
            step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            step()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / args.iters * 1000

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
            step()
            torch.cuda.synchronize()
        ka = pr.key_averages()
        ops = sum(e.count for e in ka)
        slices = sum(e.count for e in ka if "slice" in e.key.lower())
        zeros = sum(e.count for e in ka if e.key in ("aten::zeros", "aten::zero_"))
        cpu_ms = sum(e.self_cpu_time_total for e in ka) / 1000
        print(f"  {label:<10} {ms:8.1f} ms/step   ops {ops:>6}   "
              f"slice-ops {slices:>5}   zeros {zeros:>5}   self-CPU {cpu_ms:7.1f} ms")
        return ms, ops

    print(f"\nbatch={B}, one model, {args.iters} timed steps")
    unpatch()
    ms_a, ops_a = measure("original")
    patch()
    ms_b, ops_b = measure("rewritten")
    unpatch()

    print(f"\nspeedup {ms_a/ms_b:.2f}x    ops {ops_a} -> {ops_b} "
          f"({100*(ops_a-ops_b)/max(ops_a,1):.0f}% fewer)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
