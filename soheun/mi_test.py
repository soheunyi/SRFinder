import os
from typing import Callable
import click
import numpy as np
import pandas as pd
import tqdm

from counting_test_v1 import events_from_scdinfo
from fvt_classifier import FvTClassifier
from events_data import EventsData
from training_info import TrainingInfoV2
from tst_info import TSTInfo


def auc_score_fn(clf_scores: np.ndarray, is_4b: np.ndarray, weights: np.ndarray):

    assert len(clf_scores) == len(is_4b) == len(weights)

    clf_scores_3b = clf_scores[~is_4b].reshape(-1, 1)
    clf_scores_4b = clf_scores[is_4b].reshape(1, -1)
    weights_3b = weights[~is_4b].reshape(-1, 1)
    weights_4b = weights[is_4b].reshape(1, -1)

    score_diff = clf_scores_4b - clf_scores_3b
    weights = weights_3b * weights_4b

    return np.sum(weights * (score_diff > 0)) / np.sum(weights)


def mce_score_fn(
    clf_scores: np.ndarray, is_4b: np.ndarray, weights: np.ndarray, pi: float
):
    assert len(clf_scores) == len(is_4b) == len(weights)

    clf_scores_3b = clf_scores[~is_4b]
    clf_scores_4b = clf_scores[is_4b]
    weights_3b = weights[~is_4b]
    weights_4b = weights[is_4b]

    return 0.5 * (
        np.sum(weights_3b * (clf_scores_3b > pi)) / np.sum(weights_3b)
        + np.sum(weights_4b * (clf_scores_4b < pi)) / np.sum(weights_4b)
    )


def calculate_null_score(
    clf_scores: np.ndarray,
    is_4b: np.ndarray,
    weights: np.ndarray,
    bootstrap: bool,
    score_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    rnd_seed: int,
):
    np.random.seed(rnd_seed)
    if bootstrap:
        indices = np.random.choice(len(clf_scores), len(clf_scores), replace=True)
    else:
        indices = np.arange(len(clf_scores))

    clf_scores_rnd = clf_scores[indices]
    weights_rnd = weights[indices]
    is_4b_rnd = np.random.choice(len(clf_scores), np.sum(is_4b), replace=False)
    is_4b_rnd = np.isin(np.arange(len(clf_scores)), is_4b_rnd)

    return score_fn(clf_scores_rnd, is_4b_rnd, weights_rnd)


def test_via_classifier(
    events: EventsData,
    clf_scores: np.ndarray,
    method: str,
    bootstrap: bool = True,
    n_samples: int = 1000,
    p_value_type: str = "greater",
):
    assert p_value_type in [
        "greater",
        "less",
        "two-sided",
    ], f"p_value_type {p_value_type} is not supported"

    pi = events.total_weight_4b / events.total_weight
    is_4b = events.is_4b
    weights = events.weights

    if method == "auc":
        score_func = auc_score_fn
    elif method == "mce":
        score_func = lambda clf_scores, is_4b, weights: mce_score_fn(
            clf_scores, is_4b, weights, pi
        )
    else:
        raise ValueError(f"Method {method} is not supported")

    score_0 = score_func(clf_scores, is_4b, weights)

    null_scores = np.zeros(n_samples)
    for rnd_seed in range(n_samples):
        null_scores[rnd_seed] = calculate_null_score(
            clf_scores, is_4b, weights, bootstrap, score_func, rnd_seed
        )

    if p_value_type == "greater":
        p_value = np.mean(null_scores > score_0)
    elif p_value_type == "less":
        p_value = np.mean(null_scores < score_0)
    elif p_value_type == "two-sided":
        p_value = np.mean(np.abs(null_scores - score_0) > np.abs(score_0))

    return score_0, null_scores, p_value


def mi_test(
    tstinfo_hash: str,
    n_samples: int = 1000,
):

    features = [
        "sym_Jet0_pt",
        "sym_Jet1_pt",
        "sym_Jet2_pt",
        "sym_Jet3_pt",
        "sym_Jet0_eta",
        "sym_Jet1_eta",
        "sym_Jet2_eta",
        "sym_Jet3_eta",
        "sym_Jet0_phi",
        "sym_Jet1_phi",
        "sym_Jet2_phi",
        "sym_Jet3_phi",
        "sym_Jet0_m",
        "sym_Jet1_m",
        "sym_Jet2_m",
        "sym_Jet3_m",
    ]
    device = "cuda"

    tstinfo = TSTInfo.load(tstinfo_hash)
    signal_filename = tstinfo.hparams["signal_filename"]
    signal_ratio = tstinfo.hparams["signal_ratio"]
    seed = tstinfo.hparams["seed"]
    ratio_4b = tstinfo.hparams["ratio_4b"]
    batch_size = 1024

    CR_fvt_tinfo_hash = tstinfo.CR_fvt_tinfo_hash
    CR_fvt_tinfo = TrainingInfoV2.load(CR_fvt_tinfo_hash)
    CR_model = FvTClassifier.load_from_checkpoint(
        f"data/checkpoints/{CR_fvt_tinfo.hash}_best.ckpt"
    )
    CR_model.to(device)
    CR_model.eval()

    events_tst = events_from_scdinfo(tstinfo.scdinfo_tst, features, signal_filename)

    tst_fvt_scores = CR_model.predict(events_tst.X_torch).detach().cpu().numpy()[:, 1]
    SR_stat = tstinfo.SR_stats
    reweights = tst_fvt_scores / (1 - tst_fvt_scores) * (ratio_4b / (1 - ratio_4b))
    SR_cut = tstinfo.SR_cut
    in_SR = SR_stat > SR_cut

    events_tst_clone = events_tst.clone()
    events_tst_clone.reweight(
        np.where(
            events_tst_clone.is_4b,
            events_tst_clone.weights,
            events_tst_clone.weights * reweights,
        )
    )

    in_SR = SR_stat >= SR_cut
    events_tst_clone_SR = events_tst_clone[in_SR]

    SR_classifier = FvTClassifier(
        num_classes=2,
        dim_input_jet_features=4,
        dim_dijet_features=6,
        dim_quadjet_features=6,
        run_name="",
        device=device,
        lr=0.001,
    )

    events_tst_SR_train, events_tst_SR_test = events_tst_clone_SR.split(0.9, seed=seed)
    events_tst_SR_train, events_tst_SR_val = events_tst_SR_train.split(2 / 3, seed=seed)
    events_tst_SR_train.fit_batch_size(batch_size=batch_size)
    events_tst_SR_val.fit_batch_size(batch_size=batch_size)

    print(len(events_tst_SR_train), len(events_tst_SR_val), len(events_tst_SR_test))

    SR_classifier.fit(
        events_tst_SR_train.to_tensor_dataset(),
        events_tst_SR_val.to_tensor_dataset(),
        max_epochs=10,
        train_seed=seed,
        save_checkpoint=False,
        callbacks=[],
        batch_size=batch_size,
    )

    SR_classifier.eval()
    SR_classifier.to(device)

    SR_classifier_scores = (
        SR_classifier.predict(events_tst_SR_test.X_torch).detach().cpu().numpy()[:, 1]
    )

    auc_score_0, _, p_value_auc_bootstrap = test_via_classifier(
        events_tst_SR_test,
        SR_classifier_scores,
        "auc",
        bootstrap=True,
        n_samples=n_samples,
        p_value_type="greater",
    )

    mce_score_0, _, p_value_mce_bootstrap = test_via_classifier(
        events_tst_SR_test,
        SR_classifier_scores,
        "mce",
        bootstrap=True,
        n_samples=n_samples,
        p_value_type="less",
    )

    _, _, p_value_auc_permutation = test_via_classifier(
        events_tst_SR_test,
        SR_classifier_scores,
        "auc",
        bootstrap=False,
        n_samples=n_samples,
        p_value_type="greater",
    )

    _, _, p_value_mce_permutation = test_via_classifier(
        events_tst_SR_test,
        SR_classifier_scores,
        "mce",
        bootstrap=False,
        n_samples=n_samples,
        p_value_type="less",
    )

    results = {
        "tstinfo_hash": tstinfo_hash,
        "seed": seed,
        "signal_ratio": signal_ratio,
        "auc_score_0": auc_score_0,
        "p_value_auc_bootstrap": p_value_auc_bootstrap,
        "p_value_auc_permutation": p_value_auc_permutation,
        "mce_score_0": mce_score_0,
        "p_value_mce_bootstrap": p_value_mce_bootstrap,
        "p_value_mce_permutation": p_value_mce_permutation,
    }

    return results


@click.command()
@click.option("--experiment-name", type=str, default="counting_test_high_4b_in_CR")
@click.option("--n-3b", type=int, default=140_0000)
@click.option("--n-samples", type=int, default=1000)
@click.option("--seed-start", type=int)
@click.option("--seed-end", type=int)
def main(experiment_name, n_3b, n_samples, seed_start, seed_end):
    df_name = f"data/tsv/tst_results_summary_{experiment_name}_n_3b={n_3b}_mi_test_seed={seed_start}_to_{seed_end}.tsv"

    if os.path.exists(df_name):
        df = pd.read_csv(df_name, sep="\t")
    else:
        df = pd.DataFrame(
            columns=[
                "tstinfo_hash",
                "seed",
                "signal_ratio",
                "auc_score_0",
                "p_value_auc_bootstrap",
                "p_value_auc_permutation",
                "mce_score_0",
                "p_value_mce_bootstrap",
                "p_value_mce_permutation",
            ]
        )

    hparam_filter = {
        "experiment_name": experiment_name,
        "n_3b": n_3b,
        "seed": lambda x: seed_start <= x < seed_end,
    }
    hashes = TSTInfo.find(hparam_filter, sort_by=["signal_ratio", "seed"])
    existing_hashes = df["tstinfo_hash"].values if len(df) > 0 else []
    hashes = [h for h in hashes if h not in existing_hashes]

    for tstinfo_hash in tqdm.tqdm(hashes):
        print(f"Testing {tstinfo_hash}")
        results = mi_test(tstinfo_hash, n_samples)
        df = pd.concat([df, pd.DataFrame([results])], ignore_index=True)
        df.to_csv(df_name, sep="\t", index=False)


if __name__ == "__main__":
    main()
