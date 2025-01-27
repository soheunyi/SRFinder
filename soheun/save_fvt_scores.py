import datetime
from itertools import cycle
import os
import pathlib
import numpy as np
import torch
import tqdm
from training_info import TrainingInfo
from fvt_classifier import FvTClassifier
from dataset import MotherSamples
from events_data import EventsData
import pandas as pd
import click
import logging
from correct_systematic_error import compute_sr_stats, get_SR_CR_cut, get_is_signal
from concurrent.futures import ProcessPoolExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logging.info("Packages loaded")

path_3b = pathlib.Path("../events/MG3/dataframes/threeTag_picoAOD.h5")
path_bg4b = pathlib.Path("../events/MG3/dataframes/fourTag_10x_picoAOD.h5")
path_signal = pathlib.Path("../events/MG3/dataframes/HH4b_picoAOD.h5")
df_3b = pd.read_hdf(path_3b)
df_bg4b = pd.read_hdf(path_bg4b)
df_signal = pd.read_hdf(path_signal)
df_3b["signal"] = False
df_bg4b["signal"] = False
df_signal["signal"] = True
loaded_df = {path_3b: df_3b, path_bg4b: df_bg4b, path_signal: df_signal}

logging.info("Datasets loaded")

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


def process_hash(args: tuple[int, int, str]):
    process_id, n_processes, CR_fvt_hash = args

    CR_fvt_tinfo = TrainingInfo.load(CR_fvt_hash)

    tst_calculated, train_calculated = False, False

    # if isinstance(CR_fvt_tinfo.aux_info.get("fvt_scores_tst_SR"), np.ndarray):
    #     tst_calculated = True
    # if isinstance(CR_fvt_tinfo.aux_info.get("fvt_scores_train_SR"), np.ndarray):
    #     train_calculated = True

    # if tst_calculated and train_calculated:
    #     file_handler.stream.write(f"FVT scores already computed for {CR_fvt_hash}\n")
    #     file_handler.stream.flush()
    #     return

    SR_stats_hashes = CR_fvt_tinfo.hparams["signal_region"]["SR_stats_hashes"]
    ensemble_mode = CR_fvt_tinfo.hparams["signal_region"]["ensemble_mode"]
    stats_type = CR_fvt_tinfo.hparams["signal_region"]["stats_type"]
    signal_filename = CR_fvt_tinfo.hparams["dataset"]["signal_filename"]

    SR_stats_train, SR_stats_tst = compute_sr_stats(
        SR_stats_hashes, signal_filename, ensemble_mode, stats_type
    )
    ms_idx = TrainingInfo.load(SR_stats_hashes[0]).ms_idx
    msamples = MotherSamples.load(CR_fvt_tinfo.ms_hash)
    train_scdinfo = msamples.scdinfo[ms_idx]
    tst_scdinfo = msamples.scdinfo[~ms_idx]

    df_train = train_scdinfo.fetch_data(loaded_df)
    df_train["signal"] = get_is_signal(train_scdinfo, signal_filename)
    events_train = EventsData.from_dataframe(df_train, features)

    df_tst = tst_scdinfo.fetch_data(loaded_df)
    df_tst["signal"] = get_is_signal(tst_scdinfo, signal_filename)

    SR_cut, _ = get_SR_CR_cut(
        SR_stats_train, events_train, CR_fvt_tinfo.hparams["signal_region"]
    )
    events_train_SR = events_train[SR_stats_train >= SR_cut]
    SR_idx = SR_stats_tst >= SR_cut
    scdinfo_tst_SR = tst_scdinfo[SR_idx]
    events_tst_SR = EventsData.from_dataframe(
        scdinfo_tst_SR.fetch_data(loaded_df), features
    )

    CR_fvt_model = CR_fvt_tinfo.load_trained_model("best")
    CR_fvt_model.eval()
    CR_fvt_model.to(
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    CR_fvt_model: FvTClassifier

    if not train_calculated:
        fvt_scores_train_SR = (
            CR_fvt_model.predict(events_train_SR.X_torch)[:, 1].detach().cpu().numpy()
        )
        CR_fvt_tinfo.update_aux_info(fvt_scores_train_SR=fvt_scores_train_SR)

    if not tst_calculated:
        fvt_scores_tst_SR = (
            CR_fvt_model.predict(events_tst_SR.X_torch)[:, 1].detach().cpu().numpy()
        )
        CR_fvt_tinfo.update_aux_info(fvt_scores_tst_SR=fvt_scores_tst_SR)

    if not (train_calculated and tst_calculated):
        CR_fvt_tinfo.save()


@click.command()
@click.option("--experiment-name", type=str, help="Name of the experiment")
def main(experiment_name: str):
    target_hashes = TrainingInfo.find({"experiment_name": experiment_name})
    logging.info(f"Found {len(target_hashes)} target hashes")

    # create 8 file handlers
    n_processes = 1
    process_args = [(i, n_processes, hash) for i, hash in enumerate(target_hashes)]

    # with ProcessPoolExecutor(max_workers=n_processes) as executor:
    #     results = list(
    #         tqdm.tqdm(
    #             executor.map(process_hash, process_args),
    #             total=len(target_hashes),
    #             desc="Processing hashes",
    #         )
    #     )

    for args in tqdm.tqdm(process_args):
        process_hash(args)


if __name__ == "__main__":
    main()
