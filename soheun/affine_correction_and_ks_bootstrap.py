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
def main(experiment_name: str, n_reps: int):

    hashes = TrainingInfo.find(
        {
            "experiment_name": experiment_name,
            "dataset": lambda x: x["signal_ratio"] != 0.0,
        }
    )

    random_seed = 0
    grid_size = 0.001

    np.random.seed(random_seed)

    for hash_ in tqdm(hashes):
        CR_fvt_tinfo = TrainingInfo.load(hash_)
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
            stats_3b, stats_4b, weights_3b_rw, weights_4b_rw, grid_size=grid_size
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
        }

        null_max_cdf_diffs = null_max_cdf_diff_bootstrap(
            stats_3b,
            stats_4b,
            affine_tilt(
                stats_3b, weights_3b_rw, correction_slope, correction_intercept
            ),
            normalize(weights_4b_rw),
            n_samples_1=n_samples,
            n_samples_2=n_samples,
            n_reps=n_reps,
            random_seed=random_seed,
            do_tqdm=False,
        )

        alt_max_cdf_diff = max_cdf_diff(
            stats_3b,
            stats_4b,
            affine_tilt(
                stats_3b, weights_3b_rw, correction_slope, correction_intercept
            ),
            normalize(weights_4b_rw),
        )

        results["alt_value"] = alt_max_cdf_diff
        results["null_values"] = null_max_cdf_diffs

        p_value = np.sum(null_max_cdf_diffs >= alt_max_cdf_diff) / len(
            null_max_cdf_diffs
        )
        results["p_value"] = p_value

        CR_fvt_tinfo.aux_info[
            f"affine_correction_and_ks_poisson_bootstrap_n_reps={n_reps}"
        ] = results
        CR_fvt_tinfo.save()


if __name__ == "__main__":
    main()
