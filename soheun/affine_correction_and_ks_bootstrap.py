from typing import Literal
import numpy as np
from constants import FEATURES
from dataset import MotherSamples
from signal_region import compute_sr_stats, get_SR_CR_cut
from training_info import TrainingInfo
from events_data import events_from_scdinfo
import click
from ks_test import affine_correction, affine_tilt, normalize, max_cdf_diff
from ks_poisson_bootstrap import null_max_cdf_diff_bootstrap
from tqdm import tqdm


@click.command()
@click.option("--experiment_name", type=str)
@click.option("--n_reps", type=int)
@click.option("--signal_ratio", type=float)
@click.option("--cdf_mode", type=str)
def main(
    experiment_name: str,
    n_reps: int,
    signal_ratio: float,
    cdf_mode: Literal["max", "mean"],
):
    print(f"experiment_name: {experiment_name}, signal_ratio: {signal_ratio}")
    hashes = TrainingInfo.find(
        {
            "experiment_name": experiment_name,
            "dataset": (lambda x: x["signal_ratio"] == signal_ratio),
        }
    )
    # assert isinstance(hashes, list), "Expected list return type"

    random_seed = 0
    grid_size = 0.005

    np.random.seed(random_seed)

    for hash_ in tqdm(hashes):
        CR_fvt_tinfo = TrainingInfo.load(hash_)

        if (
            f"affine_correction_and_ks_poisson_bootstrap_n_reps={n_reps}_cdf_mode={cdf_mode}"
            in CR_fvt_tinfo.aux_info
        ):
            continue

        SR_stats_hashes = CR_fvt_tinfo.hparams["signal_region"]["SR_stats_hashes"]
        ensemble_mode = CR_fvt_tinfo.hparams["signal_region"]["ensemble_mode"]
        stats_type = CR_fvt_tinfo.hparams["signal_region"]["stats_type"]
        signal_filename = CR_fvt_tinfo.hparams["dataset"]["signal_filename"]

        SR_stats_train, SR_stats_tst = compute_sr_stats(
            SR_stats_hashes,
            signal_filename,
            ensemble_mode,
            stats_type,
        )

        smeared_tinfo = TrainingInfo.load(SR_stats_hashes[0])
        msamples = MotherSamples.load(smeared_tinfo.ms_hash)
        events_train = events_from_scdinfo(
            msamples.scdinfo[smeared_tinfo.ms_idx], FEATURES, signal_filename
        )
        events_tst = events_from_scdinfo(
            msamples.scdinfo[~smeared_tinfo.ms_idx], FEATURES, signal_filename
        )
        SR_size = CR_fvt_tinfo.hparams["signal_region"]["4b_in_SR"]
        CR_size = CR_fvt_tinfo.hparams["signal_region"]["4b_in_CR"]

        SR_cut, CR_cut = get_SR_CR_cut(
            SR_stats_train,
            events_train,
            {"4b_in_SR": SR_size, "4b_in_CR": CR_size},
        )

        SR_stats_tst_SR = SR_stats_tst[SR_stats_tst >= SR_cut]
        events_tst_SR = events_tst[SR_stats_tst >= SR_cut]
        fvt_scores_tst_SR = CR_fvt_tinfo.aux_info["fvt_scores_tst_SR"]
        reweights_tst_SR = fvt_scores_tst_SR / (1 - fvt_scores_tst_SR)
        rw_tst_SR = np.where(
            events_tst_SR.is_4b,
            events_tst_SR.weights,
            reweights_tst_SR * events_tst_SR.weights,
        )
        is_4b_tst_SR = events_tst_SR.is_4b

        stats_3b = SR_stats_tst_SR[~is_4b_tst_SR]
        stats_4b = SR_stats_tst_SR[is_4b_tst_SR]
        weights_3b_rw = rw_tst_SR[~is_4b_tst_SR]
        weights_4b_rw = rw_tst_SR[is_4b_tst_SR]

        correction_slope, correction_intercept = affine_correction(
            stats_3b,
            stats_4b,
            weights_3b_rw,
            weights_4b_rw,
            grid_size=grid_size,
            cdf_mode=cdf_mode,
        )

        stats_3b_mean = np.sum(weights_3b_rw * stats_3b) / np.sum(weights_3b_rw)
        stats_3b_std = np.sqrt(
            np.sum(weights_3b_rw * (stats_3b - stats_3b_mean) ** 2)
            / np.sum(weights_3b_rw)
        )

        results = {
            "correction_slope": correction_slope,
            "correction_intercept": correction_intercept,
            "stats_3b_mean": stats_3b_mean,
            "stats_3b_std": stats_3b_std,
            "stats_3b_min": np.min(stats_3b),
            "stats_3b_max": np.max(stats_3b),
        }

        null_max_cdf_diffs_no_correction = null_max_cdf_diff_bootstrap(
            stats_3b,
            stats_4b,
            normalize(weights_3b_rw),
            normalize(weights_4b_rw),
            n_reps=n_reps,
            random_seed=random_seed,
            do_tqdm=False,
            n_jobs=4,
        )

        alt_max_cdf_diff_no_correction = max_cdf_diff(
            stats_3b,
            stats_4b,
            normalize(weights_3b_rw),
            normalize(weights_4b_rw),
        )

        null_max_cdf_diffs_correction = null_max_cdf_diff_bootstrap(
            stats_3b,
            stats_4b,
            affine_tilt(
                stats_3b, weights_3b_rw, correction_slope, correction_intercept
            ),
            normalize(weights_4b_rw),
            n_reps=n_reps,
            random_seed=random_seed,
            do_tqdm=False,
            n_jobs=4,
        )

        alt_max_cdf_diff_correction = max_cdf_diff(
            stats_3b,
            stats_4b,
            affine_tilt(
                stats_3b, weights_3b_rw, correction_slope, correction_intercept
            ),
            normalize(weights_4b_rw),
        )

        results["alt_value_no_correction"] = alt_max_cdf_diff_no_correction
        results["alt_value_correction"] = alt_max_cdf_diff_correction
        results["null_values_no_correction"] = null_max_cdf_diffs_no_correction
        results["null_values_correction"] = null_max_cdf_diffs_correction

        p_value_no_correction = np.sum(
            null_max_cdf_diffs_no_correction >= alt_max_cdf_diff_no_correction
        ) / len(null_max_cdf_diffs_no_correction)
        p_value_correction = np.sum(
            null_max_cdf_diffs_correction >= alt_max_cdf_diff_correction
        ) / len(null_max_cdf_diffs_correction)

        print(f"p_value_no_correction: {p_value_no_correction}")
        print(f"p_value_correction: {p_value_correction}")

        results["p_value_no_correction"] = p_value_no_correction
        results["p_value_correction"] = p_value_correction

        CR_fvt_tinfo.aux_info[
            f"affine_correction_and_ks_poisson_bootstrap_n_reps={n_reps}_cdf_mode={cdf_mode}"
        ] = results
        CR_fvt_tinfo.save()


if __name__ == "__main__":
    main()  # type: ignore
