import itertools
import click
import numpy as np
from attention_classifier import AttentionClassifier
from training_info import TrainingInfo
from fvt_classifier import FvTClassifier
from dataset import MotherSamples
from events_data import EventsData, get_is_signal
from constants import FEATURES
from pathlib import Path
import pandas as pd
import tqdm
import torch
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)


def step_1_save_aux_info(
    base_fvt_tinfo: TrainingInfo, loaded_df: dict[Path, pd.DataFrame] = {}
):
    if (
        "base_fvt_logit_train" in base_fvt_tinfo.aux_info
        and "base_fvt_logit_tst" in base_fvt_tinfo.aux_info
    ):
        return

    if "step" not in base_fvt_tinfo.hparams:
        base_fvt_tinfo._hparams["step"] = base_fvt_tinfo.aux_info["step"]
    assert (
        base_fvt_tinfo.hparams["step"] == 1
    ), f"Step should be 1, but is {base_fvt_tinfo.hparams['step']}"
    base_fvt_tinfo.update_aux_info(description=f"Step 1: base_FvT", step=1)

    base_fvt_model = base_fvt_tinfo.load_trained_model("best")
    base_fvt_model: FvTClassifier
    base_fvt_model.eval()

    ms_hash = base_fvt_tinfo.ms_hash
    ms_idx = base_fvt_tinfo.ms_idx
    mother_samples = MotherSamples.load(ms_hash)
    signal_filename = base_fvt_tinfo.hparams["dataset"]["signal_filename"]

    train_scdinfo = mother_samples.scdinfo[ms_idx]
    df_train = train_scdinfo.fetch_data(loaded_df)
    df_train["signal"] = get_is_signal(train_scdinfo, signal_filename)
    events_train = EventsData.from_dataframe(df_train, FEATURES)
    base_fvt_score_train, _ = base_fvt_model.predict_and_representations(
        events_train.X_torch
    )
    base_fvt_score_train = base_fvt_score_train[:, 1].numpy()
    base_fvt_logit_train = np.log(base_fvt_score_train / (1 - base_fvt_score_train))

    tst_scdinfo = mother_samples.scdinfo[~ms_idx]
    df_tst = tst_scdinfo.fetch_data(loaded_df)
    df_tst["signal"] = get_is_signal(tst_scdinfo, signal_filename)
    events_tst = EventsData.from_dataframe(df_tst, FEATURES)
    base_fvt_score_tst, _ = base_fvt_model.predict_and_representations(
        events_tst.X_torch
    )
    base_fvt_score_tst = base_fvt_score_tst[:, 1].numpy()
    base_fvt_logit_tst = np.log(base_fvt_score_tst / (1 - base_fvt_score_tst))

    base_fvt_tinfo.aux_info.update(
        {
            "base_fvt_logit_train": base_fvt_logit_train,
            "base_fvt_logit_tst": base_fvt_logit_tst,
        }
    )
    base_fvt_tinfo.save()


def step_1_save_aux_info_later(base_hashes: list[str]):
    path_3b = Path("../events/MG3/dataframes/threeTag_picoAOD.h5")
    path_4b = Path("../events/MG3/dataframes/fourTag_10x_picoAOD.h5")
    path_HH4b = Path("../events/MG3/dataframes/HH4b_picoAOD.h5")
    path_HH4b_400 = Path("../events/MG3/dataframes/HH4b_400.h5")
    path_HH4b_800 = Path("../events/MG3/dataframes/HH4b_800.h5")

    df_3b = pd.read_hdf(path_3b)
    df_bg4b = pd.read_hdf(path_4b)
    df_HH4b = pd.read_hdf(path_HH4b)
    df_HH4b_400 = pd.read_hdf(path_HH4b_400)
    df_HH4b_800 = pd.read_hdf(path_HH4b_800)

    df_3b["signal"] = False
    df_bg4b["signal"] = False
    df_HH4b["signal"] = True
    df_HH4b_400["signal"] = True
    df_HH4b_800["signal"] = True

    loaded_df = {
        path_3b: df_3b,
        path_4b: df_bg4b,
        path_HH4b: df_HH4b,
        path_HH4b_400: df_HH4b_400,
        path_HH4b_800: df_HH4b_800,
    }

    for base_hash in tqdm.tqdm(base_hashes):
        base_fvt_tinfo = TrainingInfo.load(base_hash)
        step_1_save_aux_info(base_fvt_tinfo, loaded_df)


def step_2_save_aux_info(
    smeared_fvt_tinfo: TrainingInfo, loaded_df: dict[Path, pd.DataFrame] = {}
):
    if (
        "smeared_fvt_logit_train" in smeared_fvt_tinfo.aux_info
        and "smeared_fvt_logit_tst" in smeared_fvt_tinfo.aux_info
    ):
        return

    assert (
        smeared_fvt_tinfo.hparams["step"] == 2
    ), f"Step should be 2, but is {smeared_fvt_tinfo.hparams['step']}"
    smeared_fvt_tinfo.update_aux_info(
        description=f"Step 2: smeared_FvT_based_on_{smeared_fvt_tinfo.hparams['encoder_hash']}",
        step=2,
    )
    signal_filename = smeared_fvt_tinfo.hparams["dataset"]["signal_filename"]
    mother_samples = MotherSamples.load(smeared_fvt_tinfo.ms_hash)
    train_scdinfo = mother_samples.scdinfo[smeared_fvt_tinfo.ms_idx]
    df_train = train_scdinfo.fetch_data(loaded_df)
    df_train["signal"] = get_is_signal(train_scdinfo, signal_filename)
    events_train = EventsData.from_dataframe(df_train, FEATURES)

    tst_scdinfo = mother_samples.scdinfo[~smeared_fvt_tinfo.ms_idx]
    df_tst = tst_scdinfo.fetch_data(loaded_df)
    df_tst["signal"] = get_is_signal(tst_scdinfo, signal_filename)
    events_tst = EventsData.from_dataframe(df_tst, FEATURES)

    base_fvt_tinfo = TrainingInfo.load(smeared_fvt_tinfo.hparams["encoder_hash"])
    base_fvt_model = base_fvt_tinfo.load_trained_model("best")
    base_fvt_model: FvTClassifier
    base_fvt_model.eval()

    _, base_q_repr_train = base_fvt_model.predict_and_representations(
        events_train.X_torch
    )
    base_q_repr_train = base_q_repr_train.numpy()

    _, base_q_repr_tst = base_fvt_model.predict_and_representations(events_tst.X_torch)
    base_q_repr_tst = base_q_repr_tst.numpy()

    smeared_fvt_model = smeared_fvt_tinfo.load_trained_model("best")
    smeared_fvt_model: AttentionClassifier
    smeared_fvt_model.eval()

    smeared_fvt_score_train = smeared_fvt_model.predict(base_q_repr_train)[:, 1].numpy()
    smeared_fvt_logit_train = np.log(
        smeared_fvt_score_train / (1 - smeared_fvt_score_train)
    )

    smeared_fvt_score_tst = smeared_fvt_model.predict(base_q_repr_tst)[:, 1].numpy()
    smeared_fvt_logit_tst = np.log(smeared_fvt_score_tst / (1 - smeared_fvt_score_tst))

    smeared_fvt_tinfo.aux_info.update(
        {
            "smeared_fvt_logit_train": smeared_fvt_logit_train,
            "smeared_fvt_logit_tst": smeared_fvt_logit_tst,
        }
    )
    smeared_fvt_tinfo.save()


def process_single_hash(args):
    start_time = datetime.now()
    parent_pid, pid, smeared_hash, loaded_df = args

    # Set up file logging for this process
    # log_file = f"run_files/logs/{parent_pid}_process_{pid}.log"
    # file_handler = logging.FileHandler(log_file)
    # file_handler.setLevel(logging.INFO)
    # formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # file_handler.setFormatter(formatter)
    # logging.getLogger().addHandler(file_handler)

    # Log process start with GPU info
    gpu_info = (
        f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"
    )
    logging.info(
        f"""
======================================
Process {pid} starting on {gpu_info}
Hash: {smeared_hash}
Current time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
======================================
"""
    )

    try:
        if torch.cuda.is_available():
            # Set memory limit for this process
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()
        smeared_fvt_tinfo = TrainingInfo.load(smeared_hash)
        step_2_save_aux_info(smeared_fvt_tinfo, loaded_df)
        end_time = datetime.now()
        secs_taken = (end_time - start_time).total_seconds()

        # Fix the time calculation
        hours = int(secs_taken // 3600)
        minutes = int((secs_taken % 3600) // 60)
        seconds = secs_taken % 60

        logging.info(
            f"""
======================================
Process {pid} finished
Current time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
Time taken: {hours:02d}:{minutes:02d}:{seconds:05.2f}
======================================
"""
        )
    except Exception as e:
        logging.error(f"Process {pid} failed with error: {str(e)}", exc_info=True)
        raise
    finally:
        torch.cuda.empty_cache()
        # Remove the handler when done
        # logging.getLogger().removeHandler(file_handler)


def step_2_save_aux_info_later(smeared_hashes: list[str], nprocs: int = 4):
    path_3b = Path("../events/MG3/dataframes/threeTag_picoAOD.h5")
    path_4b = Path("../events/MG3/dataframes/fourTag_10x_picoAOD.h5")
    path_HH4b = Path("../events/MG3/dataframes/HH4b_picoAOD.h5")
    path_HH4b_400 = Path("../events/MG3/dataframes/HH4b_400.h5")
    path_HH4b_800 = Path("../events/MG3/dataframes/HH4b_800.h5")

    df_3b = pd.read_hdf(path_3b)
    df_bg4b = pd.read_hdf(path_4b)
    df_HH4b = pd.read_hdf(path_HH4b)
    df_HH4b_400 = pd.read_hdf(path_HH4b_400)
    df_HH4b_800 = pd.read_hdf(path_HH4b_800)

    df_3b["signal"] = False
    df_bg4b["signal"] = False
    df_HH4b["signal"] = True
    df_HH4b_400["signal"] = True
    df_HH4b_800["signal"] = True

    loaded_df = {
        path_3b: df_3b,
        path_4b: df_bg4b,
        path_HH4b: df_HH4b,
        path_HH4b_400: df_HH4b_400,
        path_HH4b_800: df_HH4b_800,
    }

    # Get parent pid
    parent_pid = os.getpid()

    with ProcessPoolExecutor(max_workers=nprocs) as executor:
        # Enumerate hashes to pair each with a process ID
        process_args = list(enumerate(smeared_hashes))
        process_args = [
            (parent_pid, pid, hash, loaded_df) for pid, hash in process_args
        ]
        # Use executor.map with the new function and args
        list(
            tqdm.tqdm(
                executor.map(process_single_hash, process_args),
                total=len(smeared_hashes),
            )
        )


def step_2_save_aux_info_later_accelerated(
    experiment_name: str, dataset_seed: int, signal_ratio: float, num_workers: int = 8
):
    start_time = datetime.now()
    logger = logging.getLogger()
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info(f"Starting step 2 of {experiment_name}")
    logger.info(f"Dataset seed: {dataset_seed}")
    logger.info(f"Signal ratio: {signal_ratio}")
    logger.info(f"Number of workers: {num_workers}")

    path_3b = Path("../events/MG3/dataframes/threeTag_picoAOD.h5")
    path_4b = Path("../events/MG3/dataframes/fourTag_10x_picoAOD.h5")
    path_HH4b = Path("../events/MG3/dataframes/HH4b_picoAOD.h5")
    path_HH4b_400 = Path("../events/MG3/dataframes/HH4b_400.h5")
    path_HH4b_800 = Path("../events/MG3/dataframes/HH4b_800.h5")

    df_3b = pd.read_hdf(path_3b)
    df_bg4b = pd.read_hdf(path_4b)
    df_HH4b = pd.read_hdf(path_HH4b)
    df_HH4b_400 = pd.read_hdf(path_HH4b_400)
    df_HH4b_800 = pd.read_hdf(path_HH4b_800)

    df_3b["signal"] = False
    df_bg4b["signal"] = False
    df_HH4b["signal"] = True
    df_HH4b_400["signal"] = True
    df_HH4b_800["signal"] = True

    loaded_df = {
        path_3b: df_3b,
        path_4b: df_bg4b,
        path_HH4b: df_HH4b,
        path_HH4b_400: df_HH4b_400,
        path_HH4b_800: df_HH4b_800,
    }

    # Get hashes
    hashes = TrainingInfo.find(
        {
            "experiment_name": experiment_name,
            "dataset": lambda x: (
                x["seed"] == dataset_seed and x["signal_ratio"] == signal_ratio
            ),
        }
    )

    tinfos = [TrainingInfo.load(h) for h in hashes]
    assert all(tinfo.hparams["step"] == 2 for tinfo in tinfos)
    assert all(
        tinfo.hparams["dataset"]["signal_ratio"] == signal_ratio for tinfo in tinfos
    )
    assert all(tinfo.hparams["dataset"]["seed"] == dataset_seed for tinfo in tinfos)

    ms_hash = tinfos[0].ms_hash
    ms_idx = tinfos[0].ms_idx
    signal_filename = tinfos[0].hparams["dataset"]["signal_filename"]
    for tinfo in tinfos:
        assert tinfo.ms_hash == ms_hash
        assert np.all(tinfo.ms_idx == ms_idx)
        assert tinfo.hparams["dataset"]["signal_filename"] == signal_filename

    train_seeds = np.unique([tinfo.hparams["train_seed"] for tinfo in tinfos])
    assert len(train_seeds) == 15
    encoder_hashes = np.unique([tinfo.hparams["encoder_hash"] for tinfo in tinfos])
    assert len(encoder_hashes) == 15

    base_fvt_model_dict = {
        encoder_hash: TrainingInfo.load(encoder_hash).load_trained_model("best")
        for encoder_hash in encoder_hashes
    }
    for encoder_hash in encoder_hashes:
        base_fvt_model_dict[encoder_hash].eval()

    mother_samples = MotherSamples.load(ms_hash)
    scdinfo_all = mother_samples.scdinfo
    df_all = scdinfo_all.fetch_data(loaded_df)
    df_all["signal"] = get_is_signal(scdinfo_all, signal_filename)
    events_all = EventsData.from_dataframe(df_all, FEATURES)

    base_q_repr_all_dict = {}

    logger.info("Computing base_q_repr_all_dict")

    for encoder_hash in tqdm.tqdm(encoder_hashes, total=len(encoder_hashes)):
        base_q_repr_all_dict[encoder_hash] = (
            base_fvt_model_dict[encoder_hash]
            .predict_and_representations(events_all.X_torch, num_workers=num_workers)[1]
            .numpy()
        )
        torch.cuda.empty_cache()
        logger.info(f"Done computing base_q_repr_all_dict for {encoder_hash}")

    logger.info("Computing smeared_fvt_logit_all_dict")

    for tinfo in tqdm.tqdm(tinfos, total=len(tinfos)):
        step_2_compute_logit_routine(
            tinfo, base_q_repr_all_dict, num_workers=num_workers, logger=logger
        )
        torch.cuda.empty_cache()

    logger.info("Done computing smeared_fvt_logit_all_dict")
    logger.info(
        f"Done step 2 of {experiment_name}, dataset_seed: {dataset_seed}, signal_ratio: {signal_ratio}"
    )
    end_time = datetime.now()
    secs_taken = (end_time - start_time).total_seconds()
    hours = int(secs_taken // 3600)
    minutes = int((secs_taken % 3600) // 60)
    seconds = secs_taken % 60
    logger.info(f"Time taken: {hours:02d}:{minutes:02d}:{seconds:05.2f}")


def step_2_compute_logit_routine(
    tinfo: TrainingInfo,
    base_q_repr_all_dict: dict[str, np.ndarray],
    num_workers: int = 0,
    logger: logging.Logger = None,
):
    encoder_hash = tinfo.hparams["encoder_hash"]
    ms_idx = tinfo.ms_idx
    smeared_fvt_model = tinfo.load_trained_model("best")
    smeared_fvt_model: AttentionClassifier
    smeared_fvt_model.eval()

    smeared_fvt_score_all = smeared_fvt_model.predict(
        base_q_repr_all_dict[encoder_hash], num_workers=num_workers
    )[:, 1].numpy()
    smeared_fvt_logit_all = np.log(smeared_fvt_score_all / (1 - smeared_fvt_score_all))
    tinfo.aux_info.update(
        {
            "smeared_fvt_logit_train": smeared_fvt_logit_all[ms_idx],
            "smeared_fvt_logit_tst": smeared_fvt_logit_all[~ms_idx],
        }
    )
    tinfo.save()
    if logger is not None:
        logger.info(f"Done processing {tinfo.hash}")


def step_2_sanity_check(experiment_name: str, previous_experiment_name: str):
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "unknown")
    slurm_node_id = os.environ.get("SLURM_NODEID", "unknown")
    logger_name = f"step_2_sanity_check_{slurm_job_id}_{slurm_node_id}"
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        f"%(asctime)s - Node {slurm_node_id} - %(levelname)s - %(message)s"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    hashes, hparams = TrainingInfo.find(
        {"experiment_name": experiment_name}, return_hparams=True
    )

    dataset_seeds = range(50)
    train_seeds = range(15)
    signal_ratios = np.unique([hp["dataset"]["signal_ratio"] for hp in hparams])

    for hash_, hp in tqdm.tqdm(zip(hashes, hparams), total=len(hashes)):
        if "step" not in hp:
            logger.info(f"{hash_} because step is not in hp")
            logger.info(hp)
            tinfo = TrainingInfo.load(hash_)
            tinfo._hparams["step"] = 2
            tinfo.save()
        if hp.get("step", 0) != 2:
            logger.info(f"{hash_} because step is not 2")
            logger.info(hp)

    for dataset_seed, signal_ratio in tqdm.tqdm(
        itertools.product(dataset_seeds, signal_ratios),
        total=len(dataset_seeds) * len(signal_ratios),
    ):
        logger.info(
            f"Sanity checking Dataset seed: {dataset_seed}, Signal ratio: {signal_ratio}"
        )
        hashes, hparams = TrainingInfo.find(
            {
                "experiment_name": experiment_name,
                "dataset": lambda x: (
                    x["seed"] == dataset_seed and x["signal_ratio"] == signal_ratio
                ),
            },
            use_cached_metadata=True,
            return_hparams=True,
        )

        tinfos = [TrainingInfo.load(h) for h in hashes]
        ms_hash = tinfos[0].ms_hash
        ms_idx = tinfos[0].ms_idx
        signal_filename = tinfos[0].hparams["dataset"]["signal_filename"]
        if any(tinfo.ms_hash != ms_hash for tinfo in tinfos):
            logger.info(f"ms_hash mismatch for {hash_}")
            logger.info("Dataset seed: ", dataset_seed, "Signal ratio: ", signal_ratio)
        if any(np.any(tinfo.ms_idx != ms_idx) for tinfo in tinfos):
            logger.info(f"ms_idx mismatch for {hash_}")
            logger.info("Dataset seed: ", dataset_seed, "Signal ratio: ", signal_ratio)
        if any(
            tinfo.hparams["dataset"]["signal_filename"] != signal_filename
            for tinfo in tinfos
        ):
            logger.info(f"signal_filename mismatch for {hash_}")
            logger.info("Dataset seed: ", dataset_seed, "Signal ratio: ", signal_ratio)

        train_seeds = np.unique([tinfo.hparams["train_seed"] for tinfo in tinfos])
        if len(train_seeds) != 15:
            logger.info(f"train_seeds mismatch for {hash_}")
            logger.info("Dataset seed: ", dataset_seed, "Signal ratio: ", signal_ratio)
        encoder_hashes = np.unique([tinfo.hparams["encoder_hash"] for tinfo in tinfos])
        if len(encoder_hashes) != 15:
            logger.info(f"encoder_hashes mismatch for {hash_}")
            logger.info("Dataset seed: ", dataset_seed, "Signal ratio: ", signal_ratio)
        for encoder_hash in encoder_hashes:
            tinfo = TrainingInfo.load(encoder_hash)
            if tinfo.hparams["experiment_name"] != previous_experiment_name:
                logger.info(f"experiment_name mismatch for {encoder_hash}")
                logger.info(
                    "Dataset seed: ", dataset_seed, "Signal ratio: ", signal_ratio
                )


def step_1_delete_unnecessary_aux_info(experiment_name: str):
    necessary_aux_info = [
        "description",
        "step",
        "base_fvt_logit_train",
        "base_fvt_logit_tst",
    ]
    hashes = TrainingInfo.find({"experiment_name": experiment_name})
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "unknown")
    slurm_node_id = os.environ.get("SLURM_NODEID", "unknown")
    logger_name = f"step_1_delete_unnecessary_aux_info_{slurm_job_id}_{slurm_node_id}"
    logger = logging.getLogger(logger_name)
    for hash_ in tqdm.tqdm(hashes, total=len(hashes)):
        tinfo = TrainingInfo.load(hash_)
        assert tinfo.hparams["step"] == 1
        delete_keys = []
        for key in tinfo.aux_info.keys():
            if key not in necessary_aux_info:
                delete_keys.append(key)
        if len(delete_keys) > 0:
            logger.info(f"Deleting {len(delete_keys)} keys for {hash_}")
            for key in delete_keys:
                tinfo.aux_info.pop(key)
            tinfo.save()


def step_2_delete_unnecessary_aux_info(experiment_name: str):
    necessary_aux_info = [
        "description",
        "step",
        "smeared_fvt_logit_train",
        "smeared_fvt_logit_tst",
    ]
    hashes = TrainingInfo.find({"experiment_name": experiment_name})
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "unknown")
    slurm_node_id = os.environ.get("SLURM_NODEID", "unknown")
    logger_name = f"step_2_delete_unnecessary_aux_info_{slurm_job_id}_{slurm_node_id}"
    logger = logging.getLogger(logger_name)
    for hash_ in tqdm.tqdm(hashes, total=len(hashes)):
        tinfo = TrainingInfo.load(hash_)
        assert tinfo.hparams["step"] == 2
        delete_keys = []
        for key in tinfo.aux_info.keys():
            if key not in necessary_aux_info:
                delete_keys.append(key)
        if len(delete_keys) > 0:
            logger.info(f"Deleting {len(delete_keys)} keys for {hash_}")
            for key in delete_keys:
                tinfo.aux_info.pop(key)
            tinfo.save()


def main():
    experiment_name = "base_fvt_training_ensemble_HH4b_800"
    path_3b = Path("../events/MG3/dataframes/threeTag_picoAOD.h5")
    path_bg4b = Path("../events/MG3/dataframes/fourTag_10x_picoAOD.h5")
    path_HH4b = Path("../events/MG3/dataframes/HH4b_picoAOD.h5")
    path_HH4b_400 = Path("../events/MG3/dataframes/HH4b_400.h5")
    path_HH4b_800 = Path("../events/MG3/dataframes/HH4b_800.h5")
    loaded_df = {
        path_3b: pd.read_hdf(path_3b),
        path_bg4b: pd.read_hdf(path_bg4b),
        path_HH4b: pd.read_hdf(path_HH4b),
        path_HH4b_400: pd.read_hdf(path_HH4b_400),
        path_HH4b_800: pd.read_hdf(path_HH4b_800),
    }
    hashes = TrainingInfo.find({"experiment_name": experiment_name})
    for hash_ in tqdm.tqdm(hashes, total=len(hashes)):
        tinfo = TrainingInfo.load(hash_)
        step_1_save_aux_info(tinfo, loaded_df)
    # for dataset_seed in range(ds_start, ds_end):
    #     for signal_ratio in [0.0, 0.02]:
    #         step_2_save_aux_info_later_accelerated(
    #             experiment_name, dataset_seed, signal_ratio, num_workers=8
    #         )


if __name__ == "__main__":
    main()
