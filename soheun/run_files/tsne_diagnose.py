#!/usr/bin/env python3
"""Why doesn't t-SNE show the concentration the maximum ensemble clearly achieves?

Reports, for one seed:
  * signal fraction in the full test split, in the SR, and in the CR
  * how strongly the ensemble-max statistic psi concentrates signal
  * kNN signal lift in the RAW representation space (no t-SNE), in the CR and
    over all test events -- this separates "t-SNE cannot show it" from "the
    concentration is not in the representation geometry at all"
"""
import os
import sys

REPO = "/home/export/soheuny/SRFinder/soheun"
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from constants import FEATURES
from dataset import MotherSamples
from events_data import events_from_scdinfo
from signal_region import get_SR_CR_cut
from training_info import TrainingInfo

EXPERIMENT = "smeared_fvt_training_ensemble"
SIGNAL_FILENAME = "HH4b_picoAOD.h5"
N_3B = 100_0000
NOISE_SCALE = 2.0
SIGNAL_RATIO = 0.02
SRCR = {"4b_in_SR": 0.05, "4b_in_CR": 0.95}
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 9
N_SAMPLE = 20000
K = 10


def lift(points, is_signal, k=K):
    base = is_signal.mean()
    if base == 0 or is_signal.sum() < 3:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=k + 1).fit(points)
    _, idx = nn.kneighbors(points[is_signal])
    return float(is_signal[idx[:, 1:]].mean() / base)


hashes = TrainingInfo.find(
    {
        "experiment_name": EXPERIMENT,
        "dataset": lambda x: (x["signal_ratio"] == SIGNAL_RATIO and x["seed"] == SEED
                              and x["n_3b"] == N_3B),
        "smearing": lambda x: x["noise_scale"] == NOISE_SCALE,
    },
    use_cached_metadata=True,
)
smeared = [TrainingInfo.load(h) for h in hashes]
bases = [TrainingInfo.load(t.hparams["encoder_hash"]) for t in smeared]

psi_tst = np.max([b.aux_info["base_fvt_logit_tst"] - s.aux_info["smeared_fvt_logit_tst"]
                  for b, s in zip(bases, smeared)], axis=0)
psi_train = np.max([b.aux_info["base_fvt_logit_train"] - s.aux_info["smeared_fvt_logit_train"]
                    for b, s in zip(bases, smeared)], axis=0)

msamples = MotherSamples.load(bases[0].ms_hash)
ms_idx = bases[0].ms_idx
events_train = events_from_scdinfo(msamples.scdinfo[ms_idx], FEATURES, SIGNAL_FILENAME)
events_tst = events_from_scdinfo(msamples.scdinfo[~ms_idx], FEATURES, SIGNAL_FILENAME)
SR_cut, CR_cut = get_SR_CR_cut(psi_train, events_train, SRCR)

sig = np.asarray(events_tst.is_signal)
in_SR = psi_tst >= SR_cut
in_CR = (psi_tst >= CR_cut) & (psi_tst < SR_cut)

print(f"=== seed {SEED}, signal_ratio={SIGNAL_RATIO}, eta={NOISE_SCALE} ===")
print(f"test events        n={len(sig):>9,}  signal={sig.sum():>7,}  frac={sig.mean():.5f}")
for name, mask in [("SR (psi top 5% of 4b)", in_SR), ("CR (5-95%)", in_CR)]:
    n = int(mask.sum())
    s = int(sig[mask].sum())
    print(f"{name:<18} n={n:>9,}  signal={s:>7,}  frac={s/max(n,1):.5f}  "
          f"lift={(s/max(n,1))/sig.mean():.2f}  holds {100*s/max(sig.sum(),1):.1f}% of all signal")

# is the concentration present in the representation geometry itself?
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = bases[0].load_trained_model("best")
model.eval()
model.to(device)
rng = np.random.default_rng(0)
for label, mask in [("CR only", in_CR), ("all test", np.ones(len(sig), bool))]:
    idx = np.where(mask)[0]
    take = idx[rng.choice(len(idx), size=min(N_SAMPLE, len(idx)), replace=False)]
    ev = events_tst[take]
    with torch.no_grad():
        q, _ = model.representations(ev.X_torch)
    repr_flat = q.reshape(len(q), -1).cpu().numpy()
    raw_flat = ev.X_torch.reshape(len(ev.X_torch), -1).cpu().numpy()
    s_mask = np.asarray(ev.is_signal)
    print(f"\n[{label}] n={len(s_mask):,} signal={int(s_mask.sum())} "
          f"(frac={s_mask.mean():.5f})")
    print(f"  kNN lift, RAW input space      = {lift(raw_flat, s_mask):.2f}")
    print(f"  kNN lift, REPRESENTATION space = {lift(repr_flat, s_mask):.2f}")
    print(f"  kNN lift, psi alone (1-D)      = {lift(psi_tst[take].reshape(-1, 1), s_mask):.2f}")
