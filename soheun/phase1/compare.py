"""Phase 1 gate: compare two fingerprints of the same stacked configuration.

Checks the five criteria the issue lists for the Phase 1 reference:

    1. identical initial parameters for every estimator
    2. identical minibatch order
    3. identical validation losses
    4. identical selected best checkpoints
    5. predictions equal within numerical tolerance

A precondition block first confirms the two runs really did train the same
thing (same configs, same mother-sample subsets, same probe rows).  Estimator
identity is compared by stack position, not by ``TrainingInfo`` hash, because
those hashes are timestamp-based rather than content-based.

    python phase1/compare.py phase1_runs/dev/A phase1_runs/dev/B
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

PASS = "PASS"
FAIL = "FAIL"


class Report:
    def __init__(self):
        self.rows: list[tuple[str, str, str]] = []
        self.ok = True

    def add(self, name: str, ok: bool, detail: str = ""):
        self.rows.append((name, PASS if ok else FAIL, detail))
        self.ok = self.ok and ok

    def print(self):
        w = max(len(r[0]) for r in self.rows)
        for name, status, detail in self.rows:
            print(f"  {status}  {name.ljust(w)}  {detail}")


def load(run_dir: pathlib.Path):
    with open(run_dir / "fingerprint.json") as f:
        fp = json.load(f)
    final = np.load(run_dir / "probe_preds_final.npy")
    best = np.load(run_dir / "probe_preds_best.npy")
    return fp, final, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a", type=pathlib.Path)
    ap.add_argument("run_b", type=pathlib.Path)
    ap.add_argument("--val-loss-tol", type=float, default=0.0)
    ap.add_argument("--pred-tol", type=float, default=0.0)
    args = ap.parse_args()

    a, a_final, a_best = load(args.run_a)
    b, b_final, b_best = load(args.run_b)

    print(f"Phase 1 reference comparison")
    print(f"  A = {args.run_a}  (tag {a['run']['tag']})")
    print(f"  B = {args.run_b}  (tag {b['run']['tag']})")
    print(f"  A gpu = {a['env'].get('gpu_name')}   B gpu = {b['env'].get('gpu_name')}")
    print()

    pre = Report()
    pre.add("same number of estimators", a["num_stacks"] == b["num_stacks"])
    pre.add("same source configs", a["run"]["source_configs"] == b["run"]["source_configs"])
    pre.add("same hparams", a["run"]["hparams"] == b["run"]["hparams"])
    pre.add("same mother sample", a["run"]["ms_hash"] == b["run"]["ms_hash"])
    pre.add("same CR subsets", a["run"]["ms_idx_digests"] == b["run"]["ms_idx_digests"])
    pre.add("same probe rows", a["probe"]["probe_x_digest"] == b["probe"]["probe_x_digest"])
    pre.add(
        "same dataset sizes",
        (a["probe"]["train_rows"], a["probe"]["val_rows"])
        == (b["probe"]["train_rows"], b["probe"]["val_rows"]),
        f"train={a['probe']['train_rows']} val={a['probe']['val_rows']}",
    )
    print("Preconditions")
    pre.print()
    print()

    rep = Report()

    # 1 -------------------------------------------------------------- init
    same_init = a["init_param_digests"] == b["init_param_digests"]
    n_diff = sum(
        1
        for x, y in zip(a["init_param_digests"], b["init_param_digests"])
        if x != y
    )
    rep.add(
        "1. identical initial parameters",
        same_init,
        "all estimators match" if same_init else f"{n_diff} estimator(s) differ",
    )

    # 2 --------------------------------------------------------- batch order
    ea, eb = a["epochs"], b["epochs"]
    same_len = len(ea) == len(eb)
    bad_epochs = [
        ea[i]["epoch"]
        for i in range(min(len(ea), len(eb)))
        if ea[i]["batch_order_digest"] != eb[i]["batch_order_digest"]
    ]
    bad_sizes = [
        ea[i]["epoch"]
        for i in range(min(len(ea), len(eb)))
        if ea[i]["batch_sizes"] != eb[i]["batch_sizes"]
    ]
    rep.add(
        "2. identical minibatch order",
        same_len and not bad_epochs,
        f"{len(ea)} epochs"
        + ("" if not bad_epochs else f"; differ at epochs {bad_epochs[:10]}"),
    )
    rep.add(
        "2b. identical batch-size schedule",
        same_len and not bad_sizes,
        "sizes "
        + str(sorted({s for e in ea for s in e["batch_sizes"]}))
        + ("" if not bad_sizes else f"; differ at epochs {bad_sizes[:10]}"),
    )

    # 3 --------------------------------------------------------- val losses
    va = np.array(
        [[np.nan if v is None else v for v in e["val_loss_per_stack"]] for e in a["val_epochs"]]
    )
    vb = np.array(
        [[np.nan if v is None else v for v in e["val_loss_per_stack"]] for e in b["val_epochs"]]
    )
    if va.shape == vb.shape and va.size:
        dv = np.nanmax(np.abs(va - vb))
        rep.add(
            "3. identical validation losses",
            bool(dv <= args.val_loss_tol),
            f"max |diff| = {dv:.3e} over {va.shape[0]} epochs x {va.shape[1]} estimators",
        )
    else:
        rep.add("3. identical validation losses", False, f"shape {va.shape} vs {vb.shape}")

    # 4 ------------------------------------------------------ best selection
    rep.add(
        "4. identical best epochs",
        a["best"]["epochs"] == b["best"]["epochs"],
        f"A={a['best']['epochs']} B={b['best']['epochs']}",
    )
    rep.add(
        "4b. identical best scores",
        a["saver_best_scores"] == b["saver_best_scores"],
    )
    rep.add(
        "4c. identical best checkpoints (bitwise)",
        a["best_ckpt_digests"] == b["best_ckpt_digests"],
    )

    # 5 ----------------------------------------------------------- predictions
    for label, pa, pb in (("final", a_final, b_final), ("best", a_best, b_best)):
        if pa.shape != pb.shape:
            rep.add(f"5. predictions ({label})", False, f"{pa.shape} vs {pb.shape}")
            continue
        d = np.abs(pa - pb)
        rep.add(
            f"5. predictions equal ({label} weights)",
            bool(d.max() <= args.pred_tol),
            f"max |diff| = {d.max():.3e}, mean |diff| = {d.mean():.3e}",
        )

    print("Phase 1 criteria")
    rep.print()
    print()

    # -------------------------------------------- informational observations
    stale_mismatch = sum(
        1
        for e in a["val_epochs"]
        if e["val_loss_per_stack_as_seen_by_saver"] != e["val_loss_per_stack"]
    )
    print("Observations (not gates)")
    print(
        f"  checkpoint selection consumed a stale metric in "
        f"{stale_mismatch}/{len(a['val_epochs'])} validation epochs"
    )
    print(f"  train seconds: A={a['run']['train_seconds']:.0f}  B={b['run']['train_seconds']:.0f}")
    print()

    overall = pre.ok and rep.ok
    print(f"RESULT: {PASS if overall else FAIL}")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
