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

            SB_idx_int = np.where((SR_stats_tst >= SB_cut) & (SR_stats_tst < SR_cut))[0]
            SR_idx_int = np.where(SR_stats_tst >= SR_cut)[0]
            SR_SB_idx_int = np.where(SR_SB_idx)[0]

            SR_idx = np.isin(SR_SB_idx_int, SR_idx_int)
            SB_idx = np.isin(SR_SB_idx_int, SB_idx_int)

            SR_stats_SB = SR_stats_tst_SR_SB[SB_idx]
            SR_stats_SR = SR_stats_tst_SR_SB[SR_idx]

            is_4b_tst_SB = events_tst_SR_SB.is_4b[SB_idx]
            is_4b_tst_SR = events_tst_SR_SB.is_4b[SR_idx]

            rw_tst_SR = rw_tst_SR_SB[SR_idx]
            rw_tst_SB = rw_tst_SR_SB[SB_idx]

            max_diff_SB = max_cdf_diff(
                SR_stats_SB[is_4b_tst_SB],
                SR_stats_SB[~is_4b_tst_SB],
                rw_tst_SB[is_4b_tst_SB],
                rw_tst_SB[~is_4b_tst_SB],
            )
            max_diff_SR = max_cdf_diff(
                SR_stats_SR[is_4b_tst_SR],
                SR_stats_SR[~is_4b_tst_SR],
                rw_tst_SR[is_4b_tst_SR],
                rw_tst_SR[~is_4b_tst_SR],
            )
            theta_SB = tilt_correction_iterative(
                SR_stats_SB[is_4b_tst_SB],
                SR_stats_SB[~is_4b_tst_SB],
                rw_tst_SB[is_4b_tst_SB],
                rw_tst_SB[~is_4b_tst_SB],
            )
            theta_SR = tilt_correction_iterative(
                SR_stats_SR[is_4b_tst_SR],
                SR_stats_SR[~is_4b_tst_SR],
                rw_tst_SR[is_4b_tst_SR],
                rw_tst_SR[~is_4b_tst_SR],
            )

            theta_SR_clipped = np.clip(theta_SR, -np.abs(theta_SB), np.abs(theta_SB))

            rw_tst_SB_corrected = np.copy(rw_tst_SB)
            rw_tst_SB_corrected[~is_4b_tst_SB] = rw_tst_SB[~is_4b_tst_SB] * np.exp(
                theta_SB * SR_stats_SB[~is_4b_tst_SB]
            )
            rw_tst_SR_corrected = np.copy(rw_tst_SR)
            rw_tst_SR_corrected[~is_4b_tst_SR] = rw_tst_SR[~is_4b_tst_SR] * np.exp(
                theta_SR_clipped * SR_stats_SR[~is_4b_tst_SR]
            )

            max_diff_SB_corrected = max_cdf_diff(
                SR_stats_SB[is_4b_tst_SB],
                SR_stats_SB[~is_4b_tst_SB],
                rw_tst_SB_corrected[is_4b_tst_SB],
                rw_tst_SB_corrected[~is_4b_tst_SB],
            )
            max_diff_SR_corrected = max_cdf_diff(
                SR_stats_SR[is_4b_tst_SR],
                SR_stats_SR[~is_4b_tst_SR],
                rw_tst_SR_corrected[is_4b_tst_SR],
                rw_tst_SR_corrected[~is_4b_tst_SR],
            )

            SB_correction_results = {
                "theta_SB": theta_SB,
                "theta_SR": theta_SR,
                "theta_SR_clipped": theta_SR_clipped,
            }

            SB_correction_results["original"] = {
                "max_diff_SB": max_diff_SB,
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
                "max_diff_SB": max_diff_SB_corrected,
                "max_diff_SR": max_diff_SR_corrected,
                "W_4b_SB": np.sum(rw_tst_SB_corrected[is_4b_tst_SB]),
                "W2_4b_SB": np.sum(rw_tst_SB_corrected[is_4b_tst_SB] ** 2),
                "W_3b_SB": np.sum(rw_tst_SB_corrected[~is_4b_tst_SB]),
                "W2_3b_SB": np.sum(rw_tst_SB_corrected[~is_4b_tst_SB] ** 2),
                "W_4b_SR": np.sum(rw_tst_SR_corrected[is_4b_tst_SR]),
                "W2_4b_SR": np.sum(rw_tst_SR_corrected[is_4b_tst_SR] ** 2),
                "W_3b_SR": np.sum(rw_tst_SR_corrected[~is_4b_tst_SR]),
                "W2_3b_SR": np.sum(rw_tst_SR_corrected[~is_4b_tst_SR] ** 2),
            }

            CR_fvt_tinfo.aux_info.update(
                {f"correction_SB_size={SB_size}": SB_correction_results}
            )

            # SR_cut, CR_cut = get_SR_CR_cut(
            #     SR_stats_train, events_train, CR_fvt_tinfo.hparams["signal_region"]
            # )
            # SR_idx = SR_stats_tst >= SR_cut

            # SR_stats_tst_SR = SR_stats_tst[SR_idx]
            # events_tst_SR = events_tst[SR_idx]
            # fvt_scores_tst_SR = CR_fvt_tinfo.aux_info["fvt_scores_tst_SR"]
            # reweights_tst_SR = fvt_scores_tst_SR / (1 - fvt_scores_tst_SR)
            # rw_tst_SR = np.where(
            #     events_tst_SR.is_4b,
            #     events_tst_SR.weights,
            #     reweights_tst_SR * events_tst_SR.weights,
            # )
            # is_4b_tst_SR = events_tst_SR.is_4b

            # _, _, p_value = max_cdf_diff_permutation(
            #     SR_stats_tst_SR[is_4b_tst_SR],
            #     SR_stats_tst_SR[~is_4b_tst_SR],
            #     rw_tst_SR[is_4b_tst_SR],
            #     rw_tst_SR[~is_4b_tst_SR],
            #     n_permutations=1000,
            # )
            # CR_fvt_tinfo.aux_info.update({"max_cdf_diff_permutation_p_value": p_value})

            # theta = tilt_correction_iterative(
            #     SR_stats_tst_SR[is_4b_tst_SR],
            #     SR_stats_tst_SR[~is_4b_tst_SR],
            #     rw_tst_SR[is_4b_tst_SR],
            #     rw_tst_SR[~is_4b_tst_SR],
            #     verbose=False,
            # )
            # # corrections = np.exp(theta * SR_stats_tst_SR)
            # max_diff_tilted = max_cdf_diff_tilted(
            #     SR_stats_tst_SR[is_4b_tst_SR],
            #     SR_stats_tst_SR[~is_4b_tst_SR],
            #     rw_tst_SR[is_4b_tst_SR],
            #     rw_tst_SR[~is_4b_tst_SR],
            #     theta,
            #     mode="exponential",
            # )

            # max_diff = max_cdf_diff(
            #     SR_stats_tst_SR[is_4b_tst_SR],
            #     SR_stats_tst_SR[~is_4b_tst_SR],
            #     rw_tst_SR[is_4b_tst_SR],
            #     rw_tst_SR[~is_4b_tst_SR],
            # )

            # events_tst_SR = events_tst[SR_idx]
            # N_3b_SR = np.sum(events_tst_SR.is_3b)
            # N_4b_SR = np.sum(events_tst_SR.is_4b)
            # W_3b_SR = np.sum(rw_tst_SR[events_tst_SR.is_3b])
            # W_4b_SR = np.sum(rw_tst_SR[events_tst_SR.is_4b])
            # W2_3b_SR = np.sum(rw_tst_SR[events_tst_SR.is_3b]**2)
            # W2_4b_SR = np.sum(rw_tst_SR[events_tst_SR.is_4b]**2)

            # CR_fvt_tinfo.aux_info.update({"max_cdf_diff": max_diff})
            # CR_fvt_tinfo.aux_info.update({"N_3b_SR": N_3b_SR, "N_4b_SR": N_4b_SR})
            # CR_fvt_tinfo.aux_info.update({"W_3b_SR": W_3b_SR, "W_4b_SR": W_4b_SR,
            #                               "W2_3b_SR": W2_3b_SR, "W2_4b_SR": W2_4b_SR})
            # CR_fvt_tinfo.aux_info.update(
            #     {
            #         "tilt_correction_iterative": theta,
            #         "max_diff_tilted_iterative": max_diff_tilted,
            #     }
            # )

            CR_fvt_tinfo.save()


if __name__ == "__main__":
    main()
