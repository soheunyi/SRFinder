import numpy as np
from events_data import EventsData
from fvt_classifier import FvTClassifier
from attention_classifier import AttentionClassifier


def get_DRs(
    events: EventsData,
    base_fvt_model: FvTClassifier,
    smeared_fvt_model: AttentionClassifier,
    return_repr: bool = False,
) -> tuple[np.ndarray, np.ndarray] | dict[str, np.ndarray]:
    """
    Get density ratios from base and smeared models
    """
    X_torch = events.X_torch
    fvt_base_score, q_repr_base = base_fvt_model.predict_and_representations(X_torch)
    fvt_base_score = fvt_base_score[:, 1].cpu().numpy()
    fvt_smeared_score = smeared_fvt_model.predict(q_repr_base)[:, 1].cpu().numpy()
    gamma = fvt_base_score / (1 - fvt_base_score)
    gamma_smeared = fvt_smeared_score / (1 - fvt_smeared_score)
    if return_repr:
        return gamma, gamma_smeared, q_repr_base
    return gamma, gamma_smeared


def get_SR_CR_cut(SR_stats: np.ndarray, events_train: EventsData, SRCR_hparams: dict):
    assert len(SR_stats) == len(events_train)
    assert "4b_in_SR" in SRCR_hparams and "4b_in_CR" in SRCR_hparams

    W_4B_CUT_MIN = 0.001
    W_4B_CUT_MAX = 0.999

    SR_stats_argsort = np.argsort(SR_stats)[::-1]
    SR_stats_sorted = SR_stats[SR_stats_argsort]
    weights = events_train.weights[SR_stats_argsort]
    is_4b = events_train.is_4b[SR_stats_argsort]
    cumul_4b_ratio = np.cumsum(weights * is_4b) / np.sum(weights * is_4b)

    w_4b_SR_ratio = np.clip(SRCR_hparams["4b_in_SR"], W_4B_CUT_MIN, W_4B_CUT_MAX)
    w_4b_CR_ratio = np.clip(
        SRCR_hparams["4b_in_CR"] + SRCR_hparams["4b_in_SR"], W_4B_CUT_MIN, W_4B_CUT_MAX
    )

    SR_cut, CR_cut = None, None
    for i in range(1, len(cumul_4b_ratio)):
        if cumul_4b_ratio[i] > w_4b_SR_ratio and SR_cut is None:
            SR_cut = SR_stats_sorted[i - 1]
        if cumul_4b_ratio[i] > w_4b_CR_ratio and CR_cut is None:
            CR_cut = SR_stats_sorted[i - 1]
        if SR_cut is not None and CR_cut is not None:
            break

    # If the cut is not found, set the cut to the minimum value
    # Both SR and CR cuts should be different
    if SR_cut is None:
        SR_cut = SR_stats_sorted[-1]
    if CR_cut is None:
        CR_cut = SR_stats_sorted[-1]
    if SR_cut == CR_cut:
        raise ValueError("SR and CR cuts are the same")

    return SR_cut, CR_cut
