import numpy as np
from constants import FEATURES
from dataset import MotherSamples
from ks_test import (
    max_cdf_diff,
    max_cdf_diff_permutation,
    max_cdf_diff_tilted,
    tilt_correction_via_first_order_approximation,
    tilt_correction_iterative,
)
from signal_region import compute_sr_stats, get_SR_CR_cut
from training_info import TrainingInfo
from events_data import events_from_scdinfo
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from constants import FEATURES
import click


@click.command()
@click.option("--experiment_name", type=str)
def main(experiment_name: str):
    SR_size = 0.2
    for correction_width in [0.5, 0.25, 0.1]:
        hashes = TrainingInfo.find(
            {
                "experiment_name": experiment_name,
                "signal_region": lambda x: x["4b_in_SR"] == SR_size,
            }
        )

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

            for SB_size in [0.05, 0.1, 0.15]:
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

                SB_idx_int = np.where(
                    (SR_stats_tst >= SB_cut) & (SR_stats_tst < SR_cut)
                )[0]
                SR_idx_int = np.where(SR_stats_tst >= SR_cut)[0]
                SR_SB_idx_int = np.where(SR_SB_idx)[0]

                SR_idx = np.isin(SR_SB_idx_int, SR_idx_int)
                SB_idx = np.isin(SR_SB_idx_int, SB_idx_int)

                # SR_stats_SB = SR_stats_tst_SR_SB[SB_idx]
                SR_stats_SR = SR_stats_tst_SR_SB[SR_idx]

                is_4b_tst_SB = events_tst_SR_SB.is_4b[SB_idx]
                is_4b_tst_SR = events_tst_SR_SB.is_4b[SR_idx]

                rw_tst_SR = rw_tst_SR_SB[SR_idx]
                rw_tst_SB = rw_tst_SR_SB[SB_idx]

                # is_4b_tst_SB_int = np.where(is_4b_tst_SB)[0]
                # isnt_4b_tst_SB_int = np.where(~is_4b_tst_SB)[0]

                # theta_SB_list = []

                # for rnd in range(200):
                #     np.random.seed(rnd)
                #     is_4b_tst_SB_rnd = np.random.choice(
                #         is_4b_tst_SB_int, size=len(is_4b_tst_SB_int) // 2, replace=False
                #     )
                #     isnt_4b_tst_SB_rnd = np.random.choice(
                #         isnt_4b_tst_SB_int, size=len(isnt_4b_tst_SB_int) // 2, replace=False
                #     )

                #     theta_SB = tilt_correction_iterative(
                #         SR_stats_SB[is_4b_tst_SB_rnd],
                #         SR_stats_SB[isnt_4b_tst_SB_rnd],
                #         rw_tst_SB[is_4b_tst_SB_rnd],
                #         rw_tst_SB[isnt_4b_tst_SB_rnd],
                #     )
                #     theta_SB_list.append(theta_SB)

                # theta_SB_mean = np.mean(theta_SB_list)
                # theta_SB_std = np.std(theta_SB_list)

                theta_SB_mean = CR_fvt_tinfo.aux_info[
                    f"correction_2_SB_size={SB_size}"
                ]["theta_SB_mean"]
                theta_SB_std = CR_fvt_tinfo.aux_info[f"correction_2_SB_size={SB_size}"][
                    "theta_SB_std"
                ]

                max_diff_SR = max_cdf_diff(
                    SR_stats_SR[is_4b_tst_SR],
                    SR_stats_SR[~is_4b_tst_SR],
                    rw_tst_SR[is_4b_tst_SR],
                    rw_tst_SR[~is_4b_tst_SR],
                )
                theta_SR = tilt_correction_iterative(
                    SR_stats_SR[is_4b_tst_SR],
                    SR_stats_SR[~is_4b_tst_SR],
                    rw_tst_SR[is_4b_tst_SR],
                    rw_tst_SR[~is_4b_tst_SR],
                )

                theta_SR_clipped = np.clip(
                    theta_SR,
                    theta_SB_mean - correction_width * theta_SB_std,
                    theta_SB_mean + correction_width * theta_SB_std,
                )

                rw_tst_SR_corrected = np.copy(rw_tst_SR)
                rw_tst_SR_corrected[~is_4b_tst_SR] = rw_tst_SR[~is_4b_tst_SR] * np.exp(
                    theta_SR_clipped * SR_stats_SR[~is_4b_tst_SR]
                )

                max_diff_SR_corrected = max_cdf_diff(
                    SR_stats_SR[is_4b_tst_SR],
                    SR_stats_SR[~is_4b_tst_SR],
                    rw_tst_SR_corrected[is_4b_tst_SR],
                    rw_tst_SR_corrected[~is_4b_tst_SR],
                )

                SB_correction_results = {
                    "theta_SB_mean": theta_SB_mean,
                    "theta_SB_std": theta_SB_std,
                    "correction_width": correction_width,
                    "theta_SR": theta_SR,
                    "theta_SR_clipped": theta_SR_clipped,
                }

                SB_correction_results["original"] = {
                    "max_diff_SR": max_diff_SR,
                    "W_4b_SB": np.sum(rw_tst_SB[is_4b_tst_SB]),
                    "W2_4b_SB": np.sum(rw_tst_SB[is_4b_tst_SB] ** 2),
                    "W_3b_SB": np.sum(rw_tst_SB[~is_4b_tst_SB]),
                    "W2_3b_SB": np.sum(rw_tst_SB[~is_4b_tst_SB] ** 2),
                    "W_4b_SR": np.sum(rw_tst_SR[is_4b_tst_SR]),
                    "W2_4b_SR": np.sum(rw_tst_SR[is_4b_tst_SR] ** 2),
                    "W_3b_SR": np.sum(rw_tst_SR[~is_4b_tst_SR]),
                    "W2_3b_SR": np.sum(rw_tst_SR[~is_4b_tst_SR] ** 2),
                }

                SB_correction_results["corrected"] = {
                    "max_diff_SR": max_diff_SR_corrected,
                    "W_4b_SR": np.sum(rw_tst_SR_corrected[is_4b_tst_SR]),
                    "W2_4b_SR": np.sum(rw_tst_SR_corrected[is_4b_tst_SR] ** 2),
                    "W_3b_SR": np.sum(rw_tst_SR_corrected[~is_4b_tst_SR]),
                    "W2_3b_SR": np.sum(rw_tst_SR_corrected[~is_4b_tst_SR] ** 2),
                }

                CR_fvt_tinfo.aux_info.update(
                    {
                        f"correction_{correction_width}_SB_size={SB_size}": SB_correction_results
                    }
                )

                CR_fvt_tinfo.save()


if __name__ == "__main__":
    main()
