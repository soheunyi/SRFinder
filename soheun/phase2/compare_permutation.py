"""Phase 2 gate, end to end: reordering a stack changes nothing.

Compares two runs of the same five estimators trained in opposite stack
orders. Everything is keyed by identity fingerprint, so position drops out.

    python phase2/compare_permutation.py phase2_runs/perm/fwd phase2_runs/perm/rev
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np


def load(run_dir: pathlib.Path):
    with open(run_dir / "fingerprint.json") as f:
        fp = json.load(f)
    preds = dict(np.load(run_dir / "probe_preds_by_identity.npz"))
    return fp, preds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a", type=pathlib.Path)
    ap.add_argument("run_b", type=pathlib.Path)
    ap.add_argument("--pred-tol", type=float, default=0.0)
    args = ap.parse_args()

    a, pa = load(args.run_a)
    b, pb = load(args.run_b)
    pa2, pb2 = a["phase2"], b["phase2"]

    print("Phase 2 permutation gate")
    print(f"  A = {args.run_a}  order={pa2['order']}  seeds={pa2['seeds']}")
    print(f"  B = {args.run_b}  order={pb2['order']}  seeds={pb2['seeds']}")
    print()

    rows: list[tuple[str, bool, str]] = []

    rows.append(
        (
            "stack orders really differ",
            pa2["seeds"] != pb2["seeds"],
            f"{pa2['seeds']} vs {pb2['seeds']}",
        )
    )
    rows.append(
        (
            "same group fingerprint",
            pa2["group_fingerprint"] == pb2["group_fingerprint"],
            pa2["group_fingerprint"][:16],
        )
    )

    ids_a = {i["fingerprint"] for i in pa2["identities"]}
    ids_b = {i["fingerprint"] for i in pb2["identities"]}
    rows.append(("same estimator set", ids_a == ids_b, f"{len(ids_a)} estimators"))

    rows.append(
        (
            "initial parameters equal per identity",
            pa2["init_digest_by_identity"] == pb2["init_digest_by_identity"],
            "",
        )
    )
    rows.append(
        (
            "final parameters equal per identity",
            pa2["final_digest_by_identity"] == pb2["final_digest_by_identity"],
            "",
        )
    )

    seeds_a = {i["fingerprint"]: i["seeds"] for i in pa2["identities"]}
    seeds_b = {i["fingerprint"]: i["seeds"] for i in pb2["identities"]}
    rows.append(("derived seeds equal per identity", seeds_a == seeds_b, ""))

    if ids_a == ids_b:
        diffs = {k: float(np.abs(pa[k] - pb[k]).max()) for k in sorted(ids_a)}
        worst = max(diffs.values())
        rows.append(
            (
                "predictions equal per identity",
                worst <= args.pred_tol,
                f"max |diff| = {worst:.3e} over {len(diffs)} estimators",
            )
        )
    else:
        rows.append(("predictions equal per identity", False, "estimator sets differ"))

    w = max(len(r[0]) for r in rows)
    ok = True
    for name, passed, detail in rows:
        ok = ok and passed
        print(f"  {'PASS' if passed else 'FAIL'}  {name.ljust(w)}  {detail}")

    print()
    print("per-identity detail")
    for ident in pa2["identities"]:
        f = ident["fingerprint"]
        pos_a = ident["position"]
        pos_b = next(i["position"] for i in pb2["identities"] if i["fingerprint"] == f)
        d = float(np.abs(pa[f] - pb[f]).max()) if f in pb else float("nan")
        print(
            f"  ms_seed={ident['mother_sample_seed']:<3} id={f[:16]}  "
            f"pos {pos_a} -> {pos_b}   max|dpred| = {d:.3e}"
        )

    print()
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
