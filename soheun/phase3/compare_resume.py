"""Phase 3 gate: an interrupted-and-resumed group matches an uninterrupted one.

The five resume tests from issue #2:

    [ ] Interrupt after a prescribed epoch
    [ ] Resume from last.ckpt
    [ ] Match uninterrupted optimizer and scheduler states
    [ ] Match final predictions within deterministic tolerance
    [ ] Preserve best checkpoints across interruption

    python phase3/compare_resume.py training_runs/<campaign>/whole \
                                    training_runs/<campaign>/split
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np
import torch


def load_group(d: pathlib.Path):
    manifest = json.load(open(d / "manifest.json"))
    completion_path = d / "completion.json"
    completion = json.load(open(completion_path)) if completion_path.exists() else None
    parts = []
    for p in sorted((d / "metrics").glob("fingerprint.part*.json"),
                    key=lambda q: int(q.stem.split("part")[1])):
        parts.append(json.load(open(p)))
    preds_path = d / "predictions" / "probe_preds_by_identity.npz"
    preds = dict(np.load(preds_path)) if preds_path.exists() else {}
    return manifest, completion, parts, preds


def merged_val_epochs(parts) -> dict[int, list]:
    out: dict[int, list] = {}
    for p in parts:
        for v in p["val_epochs"]:
            out[int(v["epoch"])] = v["val_loss_per_stack"]
    return out


def file_digest(path: pathlib.Path) -> str:
    sd = torch.load(path, map_location="cpu")
    h = hashlib.sha256()
    for k in sorted(sd):
        v = sd[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("whole", type=pathlib.Path, help="uninterrupted reference group")
    ap.add_argument("split", type=pathlib.Path, help="interrupted and resumed group")
    ap.add_argument("--pred-tol", type=float, default=0.0)
    args = ap.parse_args()

    wm, wc, wp, wpred = load_group(args.whole)
    sm, sc, sp, spred = load_group(args.split)

    print("Phase 3 resume gate")
    print(f"  uninterrupted = {args.whole}   parts={len(wp)}")
    print(f"  resumed       = {args.split}   parts={len(sp)}")
    print()

    rows: list[tuple[str, bool, str]] = []

    # 1 -------------------------------------------------- interrupt happened
    stop_epochs = [p["phase3"].get("stop_after_epoch") for p in sp]
    prescribed = next((e for e in stop_epochs if e is not None), None)
    rows.append(
        (
            "1. interrupted after a prescribed epoch",
            len(sp) > 1 and prescribed is not None,
            f"{len(sp)} segments, first stopped after epoch {prescribed}",
        )
    )

    # 2 --------------------------------------------------- resumed from ckpt
    rows.append(
        (
            "2. resumed from last.ckpt",
            any(p["phase3"]["resumed"] for p in sp) and (args.split / "last.ckpt").exists(),
            f"last.ckpt present: {(args.split / 'last.ckpt').exists()}",
        )
    )

    # epochs covered exactly once, no gap and no repeat
    w_ep, s_ep = merged_val_epochs(wp), merged_val_epochs(sp)
    seen = [int(v["epoch"]) for p in sp for v in p["val_epochs"]]
    rows.append(
        (
            "2b. epochs covered once, no gap or repeat",
            sorted(seen) == sorted(set(seen)) == sorted(w_ep),
            f"{len(seen)} epochs across segments, {len(w_ep)} in reference",
        )
    )

    # 3 ------------------------------------- optimizer and scheduler states
    if wc and sc:
        rows.append(
            (
                "3. optimizer states match",
                wc["optimizer_state_digests"] == sc["optimizer_state_digests"],
                f"{len(wc['optimizer_state_digests'])} optimizers",
            )
        )
        rows.append(
            (
                "3b. scheduler states match",
                wc["scheduler_states"] == sc["scheduler_states"],
                "",
            )
        )
    else:
        rows.append(("3. optimizer states match", False, "missing completion.json"))
        rows.append(("3b. scheduler states match", False, "missing completion.json"))

    # per-epoch validation losses along the way
    common = sorted(set(w_ep) & set(s_ep))
    if common:
        d = max(
            abs(a - b)
            for e in common
            for a, b in zip(w_ep[e], s_ep[e])
            if a is not None and b is not None
        )
        rows.append(
            (
                "3c. validation losses match every epoch",
                d <= 0.0,
                f"max |diff| = {d:.3e} over {len(common)} epochs",
            )
        )

    # 4 ------------------------------------------------------- predictions
    if wpred and spred and set(wpred) == set(spred):
        worst = max(float(np.abs(wpred[k] - spred[k]).max()) for k in wpred)
        rows.append(
            (
                "4. final predictions match",
                worst <= args.pred_tol,
                f"max |diff| = {worst:.3e} over {len(wpred)} estimators",
            )
        )
    else:
        rows.append(("4. final predictions match", False, "prediction sets differ"))

    # 5 -------------------------------------------------- best checkpoints
    w_best = {p.stem[: -len("_best")]: p
              for p in (args.whole / "individual_models").glob("*_best.pt")}
    s_best = {p.stem[: -len("_best")]: p
              for p in (args.split / "individual_models").glob("*_best.pt")}
    same_names = set(w_best) == set(s_best)
    rows.append(
        (
            "5. best checkpoints exist under stable names",
            same_names and len(w_best) > 0,
            f"{len(w_best)} vs {len(s_best)} files",
        )
    )
    if same_names and w_best:
        mismatched = [k for k in w_best if file_digest(w_best[k]) != file_digest(s_best[k])]
        rows.append(
            (
                "5b. best checkpoints identical",
                not mismatched,
                "all match" if not mismatched else f"{len(mismatched)} differ",
            )
        )

    w = max(len(r[0]) for r in rows)
    ok = True
    for name, passed, detail in rows:
        ok = ok and passed
        print(f"  {'PASS' if passed else 'FAIL'}  {name.ljust(w)}  {detail}")

    print()
    if wc and sc:
        fw, fs = wc["final_digest_by_identity"], sc["final_digest_by_identity"]
        for k in sorted(fw):
            print(f"  {k[:16]}  final weights {'match' if fw.get(k) == fs.get(k) else 'DIFFER'}")
    print()
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
