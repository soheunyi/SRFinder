import os
from typing import Literal
import numpy as np
import tqdm
from joblib import Parallel, delayed
import cvxpy as cp
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def auc_score(clf_scores: np.ndarray, is_4b: np.ndarray, weights: np.ndarray):
    assert len(clf_scores) == len(is_4b) == len(weights)

    clf_scores_argsort = np.argsort(clf_scores)
    is_4b_sorted = is_4b[clf_scores_argsort]
    weights_sorted = weights[clf_scores_argsort]
    weights_3b_cumsum = np.cumsum(np.where(~is_4b_sorted, weights_sorted, 0))
    weights_3b_sum = weights_3b_cumsum[-1]
    weights_4b_sum = np.sum(weights_sorted[is_4b_sorted])

    return np.sum(weights_3b_cumsum[is_4b_sorted] * weights_sorted[is_4b_sorted]) / (
        weights_3b_sum * weights_4b_sum
    )


def mce_score(
    clf_scores: np.ndarray, is_4b: np.ndarray, weights: np.ndarray, pi: float
):
    assert len(clf_scores) == len(is_4b) == len(weights)
    weights_4b = weights[is_4b]
    weights_3b = weights[~is_4b]
    clf_scores_4b = clf_scores[is_4b]
    clf_scores_3b = clf_scores[~is_4b]
    err_1 = np.sum(weights_4b * (clf_scores_4b < pi)) / np.sum(weights_4b)
    err_2 = np.sum(weights_3b * (clf_scores_3b >= pi)) / np.sum(weights_3b)
    return 0.5 * (err_1 + err_2)


def process_iteration(
    i: int,
    seed: int,
    clf_scores: np.ndarray,
    is_4b: np.ndarray,
    weights: np.ndarray,
    pi: float,
    alpha: float,
    method: str,
) -> tuple[float, float, float]:
    """Process single iteration of the bootstrap/permutation test."""
    # Initialize RNG with seed unique to this iteration
    rng = np.random.default_rng(seed + i)
    n_samples = len(clf_scores)
    sum_is_4b = np.sum(is_4b)

    # Generate indices for scores/weights
    if method == "bootstrap":
        indices = rng.choice(n_samples, n_samples, replace=True)
    elif method == "permutation":
        indices = rng.permutation(n_samples)
    else:
        raise ValueError(f"Invalid method: {method}")

    # Shuffle scores/weights
    clf_scores_rnd = clf_scores[indices]
    weights_rnd = weights[indices]

    # Randomly assign 4b labels
    is_4b_rnd_indices = rng.choice(n_samples, sum_is_4b, replace=False)
    is_4b_rnd = np.isin(np.arange(n_samples), is_4b_rnd_indices)

    # Compute scores
    auc = auc_score(clf_scores_rnd, is_4b_rnd, weights_rnd)
    mce = mce_score(clf_scores_rnd, is_4b_rnd, weights_rnd, pi)
    renyi = renyi_divergence(clf_scores_rnd, is_4b_rnd, weights_rnd, pi, alpha)

    return auc, mce, renyi


def renyi_divergence(
    clf_scores: np.ndarray,
    is_4b: np.ndarray,
    weights: np.ndarray,
    pi: float,
    alpha: float,
):
    # estimate the reverse renyi divergence using 4b samples
    # i.e. calculates D_alpha(p_4b || p_3b)
    # using clf_scores as a proxy for p_4b / (p_3b + p_4b)
    assert len(clf_scores) == len(is_4b) == len(weights)
    clf_scores_4b = clf_scores[is_4b]
    dr_4b = clf_scores_4b / (1 - clf_scores_4b) * (1 - pi) / pi
    weights_4b = weights[is_4b]

    if alpha == 1:
        return np.sum(np.log(dr_4b) * weights_4b) / np.sum(weights_4b)
    else:
        return (1 / (alpha - 1)) * np.log(
            np.sum(dr_4b ** (alpha - 1) * weights_4b) / np.sum(weights_4b)
        )


def minimize_renyi_divergence_cvxpy(
    clf_scores: np.ndarray,
    is_4b: np.ndarray,
    weights: np.ndarray,
    pi: float,
    alpha: float,
    correction_features: np.ndarray,
):
    assert alpha > 1
    logits = np.log(clf_scores / (1 - clf_scores) * (1 - pi) / pi)
    logits_4b = logits[is_4b]
    logits_3b = logits[~is_4b]
    weights_4b = weights[is_4b]
    weights_3b = weights[~is_4b]
    correction_features_4b = correction_features[is_4b]
    correction_features_3b = correction_features[~is_4b]

    beta = cp.Variable(correction_features_4b.shape[1])
    logit_4b_corr = logits_4b + correction_features_4b @ beta
    logit_3b_corr = logits_3b + correction_features_3b @ beta

    constraints = [
        cp.sum(cp.multiply(weights_3b, cp.exp(-logit_3b_corr))) <= np.sum(weights_3b),
        beta[1] >= -0.5,
    ]
    objective = cp.Minimize(
        (1 / (alpha - 1))
        * cp.log_sum_exp(
            logit_4b_corr * (alpha - 1) + np.log(weights_4b / np.sum(weights_4b))
        )
    )
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # print value when beta = 0
    print(
        (1 / (alpha - 1))
        * np.log(
            np.sum(np.exp(logits_4b * (alpha - 1)) * weights_4b) / np.sum(weights_4b)
        )
    )

    return problem.value, beta.value


def mi_test(
    clf_scores: np.ndarray,
    is_4b: np.ndarray,
    weights: np.ndarray,
    pi: float,
    method: Literal["bootstrap", "permutation"],
    n_reps: int,
    seed: int,
    do_tqdm: bool = True,
    alpha: float = 2,
):
    auc_score_orig = auc_score(clf_scores, is_4b, weights)
    mce_score_orig = mce_score(clf_scores, is_4b, weights, pi)
    renyi_score_orig = renyi_divergence(clf_scores, is_4b, weights, pi, alpha)

    # Parallel execution
    n_jobs = np.clip(os.cpu_count() // 2, 1, 64)
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(process_iteration)(
            i, seed, clf_scores, is_4b, weights, pi, alpha, method
        )
        for i in tqdm.tqdm(
            range(n_reps), desc="Processing iterations", disable=not do_tqdm
        )
    )

    # Unpack results
    auc_score_null, mce_score_null, renyi_score_null = zip(*results)
    auc_score_null = np.array(auc_score_null)
    mce_score_null = np.array(mce_score_null)
    renyi_score_null = np.array(renyi_score_null)

    return (
        auc_score_orig,
        mce_score_orig,
        renyi_score_orig,
        auc_score_null,
        mce_score_null,
        renyi_score_null,
    )


def calibrate_fvt_scores_cv(
    fvt_scores: np.ndarray,
    is_4b: np.ndarray,
    weights: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
    calibrator: str = "isotonic",
    renyi_alpha: float = 2,
    pi: float = 0.5,
) -> np.ndarray:
    """
    Calibrate classifier scores using Platt scaling (logistic regression) with cross-validation.

    Args:
        fvt_scores: Raw classifier scores (shape: [n_samples]).
        is_4b: Binary labels (0 or 1, shape: [n_samples]).
        weights: Sample weights (shape: [n_samples]).
        n_folds: Number of cross-validation folds.
        random_state: Random seed for reproducibility.

    Returns:
        Calibrated probabilities (shape: [n_samples]).
    """
    if calibrator is None:
        return fvt_scores

    if calibrator == "minimize_renyi":
        correction_features = np.log(
            (fvt_scores / (1 - fvt_scores) * (1 - pi) / pi).reshape(-1, 1)
        )
        correction_features = np.concatenate(
            [np.ones((correction_features.shape[0], 1)), correction_features], axis=1
        )
        # correction_features = np.ones_like(fvt_scores).reshape(-1, 1)
        min_renyi_divergence, beta_value = minimize_renyi_divergence_cvxpy(
            fvt_scores, is_4b, weights, pi, renyi_alpha, correction_features
        )

        print(min_renyi_divergence, beta_value)
        corrected_logits = (
            np.log(fvt_scores / (1 - fvt_scores) * (1 - pi) / pi)
            + correction_features @ beta_value
        )
        corrected_fvt_scores = np.exp(corrected_logits) / (1 + np.exp(corrected_logits))
        return corrected_fvt_scores

    calibrated_probs = np.zeros_like(fvt_scores)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for train_idx, test_idx in skf.split(fvt_scores, is_4b):
        # Reshape scores to 2D (required by scikit-learn)
        scores_train = fvt_scores[train_idx].reshape(-1, 1)
        scores_test = fvt_scores[test_idx].reshape(-1, 1)

        if calibrator == "isotonic":
            y_min = np.min(fvt_scores)
            y_max = np.max(fvt_scores)
            regressor = IsotonicRegression(
                out_of_bounds="clip", y_min=y_min, y_max=y_max
            )
        elif calibrator == "platt":
            # Fit logistic regression (Platt scaling)
            regressor = LogisticRegression(
                penalty=None,  # Disable regularization (Platt scaling uses unregularized LR)
                solver="lbfgs",  # Solver for unconstrained optimization
                max_iter=1000,  # Ensure convergence
            )
        else:
            raise ValueError(f"Invalid calibrator: {calibrator}")

        regressor.fit(scores_train, is_4b[train_idx], sample_weight=weights[train_idx])

        if calibrator == "isotonic":
            calibrated_probs[test_idx] = regressor.predict(scores_test)
        else:
            calibrated_probs[test_idx] = regressor.predict_proba(scores_test)[:, 1]

    return calibrated_probs
