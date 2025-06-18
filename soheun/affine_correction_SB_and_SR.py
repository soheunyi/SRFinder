import numpy as np
from constants import FEATURES
from dataset import MotherSamples
from signal_region import compute_sr_stats, get_SR_CR_cut
from training_info import TrainingInfo
from events_data import events_from_scdinfo
import click
from ks_test import affine_correction
from tqdm import tqdm


@click.command()
@click.option("--experiment_name", type=str)
def main(experiment_name: str):
    SR_size = 0.2
    SB_sizes = [0.05, 0.1, 0.15]

    hashes = TrainingInfo.find(
        {
            "experiment_name": experiment_name,
            "signal_region": lambda x: x["4b_in_SR"] == SR_size,
        }
    )

    grid_size = 0.01

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

        for SB_size in SB_sizes:
            SR_cut, SB_cut = get_SR_CR_cut(
                SR_stats_train,
                events_train,
                {"4b_in_SR": SR_size - SB_size, "4b_in_CR": SB_size},
            )
            SR_SB_idx = SR_stats_tst >= SB_cut
            SR_stats_tst_SR_SB = SR_stats_tst[SR_SB_idx]
            events_tst_SR_SB = events_tst[SR_SB_idx]
            fvt_scores_tst_SR_SB = CR_fvt_tinfo.aux_info["fvt_scores_tst_SR"]
            reweights_tst_SR_SB = fvt_scores_tst_SR_SB / (1 - fvt_scores_tst_SR_SB)
            rw_tst_SR_SB = np.where(
                events_tst_SR_SB.is_4b,
                events_tst_SR_SB.weights,
                reweights_tst_SR_SB * events_tst_SR_SB.weights,
            )

            SB_idx_int = np.where((SR_stats_tst >= SB_cut) & (SR_stats_tst < SR_cut))[0]
            SR_idx_int = np.where(SR_stats_tst >= SR_cut)[0]
            SR_SB_idx_int = np.where(SR_SB_idx)[0]

            SR_idx = np.isin(SR_SB_idx_int, SR_idx_int)
            SB_idx = np.isin(SR_SB_idx_int, SB_idx_int)

            is_4b_tst_SR = events_tst_SR_SB.is_4b[SR_idx]
            SR_stats_SR = SR_stats_tst_SR_SB[SR_idx]
            rw_tst_SR = rw_tst_SR_SB[SR_idx]

            stats_3b_SR = SR_stats_SR[~is_4b_tst_SR]
            stats_4b_SR = SR_stats_SR[is_4b_tst_SR]
            weights_3b_rw_SR = rw_tst_SR[~is_4b_tst_SR]
            weights_4b_rw_SR = rw_tst_SR[is_4b_tst_SR]

            correction_slope_SR, correction_intercept_SR = affine_correction(
                stats_3b_SR,
                stats_4b_SR,
                weights_3b_rw_SR,
                weights_4b_rw_SR,
                grid_size=grid_size,
            )

            stats_3b_mean_SR = np.sum(weights_3b_rw_SR * stats_3b_SR) / np.sum(
                weights_3b_rw_SR
            )
            stats_3b_std_SR = np.sqrt(
                np.sum(weights_3b_rw_SR * (stats_3b_SR - stats_3b_mean_SR) ** 2)
                / np.sum(weights_3b_rw_SR)
            )

            is_4b_tst_SB = events_tst_SR_SB.is_4b[SB_idx]
            SR_stats_SB = SR_stats_tst_SR_SB[SB_idx]
            rw_tst_SB = rw_tst_SR_SB[SB_idx]

            stats_3b_SB = SR_stats_SB[~is_4b_tst_SB]
            stats_4b_SB = SR_stats_SB[is_4b_tst_SB]
            weights_3b_rw_SB = rw_tst_SB[~is_4b_tst_SB]
            weights_4b_rw_SB = rw_tst_SB[is_4b_tst_SB]

            correction_slope_SB, correction_intercept_SB = affine_correction(
                stats_3b_SB,
                stats_4b_SB,
                weights_3b_rw_SB,
                weights_4b_rw_SB,
                grid_size=grid_size,
            )

            stats_3b_mean_SB = np.sum(weights_3b_rw_SB * stats_3b_SB) / np.sum(
                weights_3b_rw_SB
            )
            stats_3b_std_SB = np.sqrt(
                np.sum(weights_3b_rw_SB * (stats_3b_SB - stats_3b_mean_SB) ** 2)
                / np.sum(weights_3b_rw_SB)
            )

            CR_fvt_tinfo.aux_info[f"affine_correction_SB_size={SB_size}"] = {
                "correction_slope_SR": correction_slope_SR,
                "correction_intercept_SR": correction_intercept_SR,
                "stats_3b_mean_SR": stats_3b_mean_SR,
                "stats_3b_std_SR": stats_3b_std_SR,
                "correction_slope_SB": correction_slope_SB,
                "correction_intercept_SB": correction_intercept_SB,
                "stats_3b_mean_SB": stats_3b_mean_SB,
                "stats_3b_std_SB": stats_3b_std_SB,
            }

        CR_fvt_tinfo.save()


if __name__ == "__main__":
    main()
