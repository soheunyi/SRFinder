#!/usr/bin/env python3
"""
Test that MultiTaskFvTClassifier can train 50 tasks in one shot.
Uses the same MotherSamples→train/val split logic as test_training_speed,
but stacks 50 random seeds/labels into one (N,50) Y tensor.
"""

import logging
import os
import pickle
import time
import sys
import pathlib
from typing import Iterable
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
import tqdm

# allow imports from project root
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.absolute()))
from dataset import MotherSamples
from constants import FEATURES
from training_info import TrainingInfo

# from data_modules import FvTDataModule # Not used directly in this script
from stacked_fvt import StackedFvTClassifier

# Setup basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],  # Log to console
)
logger = logging.getLogger(__name__)  # Get logger for this module


def make_tinfos(
    ms_hparams: dict,
    base_fvt_train_ratio: float,
    seeds: Iterable[int],
    base_fvt_hparams: dict,
):
    """
    Loads or creates TrainingInfo objects and corresponding datasets for multiple seeds.
    """
    # Use the logger instance obtained above
    logger.info(f"Starting make_tinfos for {len(list(seeds))} seeds.")
    logger.info(f"Using MotherSamples parameters: {ms_hparams}")

    # Constants for MotherSamples creation
    hashes = MotherSamples.find(ms_hparams, from_metadata=False)
    if not hashes:
        logger.error(f"No MotherSamples found for parameters: {ms_hparams}")
        raise ValueError("MotherSamples not found.")
    ms_hash = hashes[0]
    logger.info(f"Found MotherSamples hash: {ms_hash}")

    t_load_start = time.perf_counter()
    mother_samples = MotherSamples.load(ms_hash)
    t_load_end = time.perf_counter()
    logger.info(f"Loaded MotherSamples in {t_load_end - t_load_start:.2f}s")

    ms_len = len(mother_samples.scdinfo)
    logger.info(f"MotherSamples length: {ms_len}")

    base_fvt_train_dsets = []
    base_fvt_val_dsets = []
    tinfos = []

    logger.info(f"Generating TrainingInfo and datasets for seeds: {list(seeds)}")
    for seed in tqdm.tqdm(seeds, desc="Processing Seeds"):
        logger.debug(f"Processing seed: {seed}")
        ms_idx = np.zeros(ms_len, dtype=bool)
        train_count = int(ms_len * base_fvt_train_ratio)
        ms_idx[:train_count] = True
        np.random.seed(seed)  # Seed for shuffling index
        np.random.shuffle(ms_idx)
        logger.debug(
            f"Created train/val split for seed {seed} (Train size: {train_count}, Val size: {ms_len - train_count})"
        )

        # Update seed-specific hparams
        current_base_fvt_hparams = base_fvt_hparams.copy()
        current_base_fvt_hparams["model_seed"] = seed
        current_base_fvt_hparams["train_seed"] = seed
        current_base_fvt_hparams["data_seed"] = seed
        logger.debug(f"Hparams for seed {seed}: {current_base_fvt_hparams}")

        try:
            t_tinfo_start = time.perf_counter()
            base_fvt_tinfo = TrainingInfo(
                current_base_fvt_hparams, ms_hash=ms_hash, ms_idx=ms_idx
            )
            t_tinfo_end = time.perf_counter()
            logger.debug(
                f"Created TrainingInfo for seed {seed} in {t_tinfo_end - t_tinfo_start:.4f}s"
            )

            # Fetch datasets using the TrainingInfo object
            # This ensures consistency with how data might be loaded elsewhere
            t_fetch_start = time.perf_counter()
            base_fvt_train_dset, base_fvt_val_dset = (
                base_fvt_tinfo.fetch_train_val_tensor_datasets(
                    FEATURES, "fourTag", "weight"
                )
            )
            t_fetch_end = time.perf_counter()
            logger.debug(
                f"Fetched datasets for seed {seed} in {t_fetch_end - t_fetch_start:.4f}s. Train size: {len(base_fvt_train_dset)}, Val size: {len(base_fvt_val_dset)}"
            )

            base_fvt_train_dsets.append(base_fvt_train_dset)
            base_fvt_val_dsets.append(base_fvt_val_dset)
            tinfos.append(base_fvt_tinfo)
        except Exception as e:
            logger.error(
                f"Failed to create TrainingInfo or fetch dataset for seed {seed}: {e}",
                exc_info=True,
            )  # Log traceback
            # Decide whether to raise, continue, or handle differently
            raise

    logger.info(
        f"Finished make_tinfos. Created {len(tinfos)} TrainingInfo objects and dataset pairs."
    )
    return tinfos, base_fvt_train_dsets, base_fvt_val_dsets


def run_multitask_test():
    # Use the logger instance obtained above
    logger.info("--- Starting Multi-Task Test --- ")
    log_file = "test_stacked_fvt.log"
    # Setup file logging specifically for this test run
    file_handler = logging.FileHandler(
        log_file, mode="w"
    )  # Overwrite log file each run
    # Use a more detailed formatter for the file log
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s"
    )
    file_handler.setFormatter(formatter)
    # Add file handler to the root logger to capture logs from all modules
    # Note: Adding handler to root logger might capture logs from imported libraries too.
    # Consider adding only to specific loggers if needed.
    root_logger = logging.getLogger("")
    root_logger.addHandler(file_handler)
    # Optionally set file log level higher if needed, e.g., logging.DEBUG
    file_handler.setLevel(logging.DEBUG)
    logger.info(f"Logging detailed output to: {log_file}")

    ms_hparams = {
        "n_3b": 1_000_000,
        "ratio_4b": 0.5,
        "signal_ratio": 0.0,
        "signal_filename": "HH4b_picoAOD.h5",
        "seed": 0,  # Base seed to find the initial MotherSamples
    }
    logger.info(f"MotherSamples search parameters: {ms_hparams}")

    # Define Base FvT Hyperparameters (used within make_tinfos)
    base_fvt_hparams = {
        "model": "FvTClassifier",  # Placeholder, not used for model itself
        "dim_dijet_features": 6,
        "dim_quadjet_features": 6,
        "repr_norm": False,
        "depth": {
            "encoder": 4,
            "decoder": 1,
        },
        "fit_batch_size": 1024,  # Used by fetch_train_val_tensor_datasets if > 0
        "model_seed": 0,  # Will be overridden per seed
        "train_seed": 0,  # Will be overridden per seed
        "data_seed": 0,  # Will be overridden per seed
        "max_epochs": 100,  # Passed to custom model.fit
        "val_ratio": 0.33,  # Used by fetch_train_val_tensor_datasets
        "early_stop_patience": None,  # Passed to custom model.fit
        "optimizer": {
            "type": "Adam",
            "lr": 0.01,
        },
        "lr_scheduler": {
            "type": "ReduceLROnPlateau",
            "factor": 0.5,
            "threshold": 0.0001,
            "patience": 10,
            "cooldown": 1,
            "min_lr": 0.0002,
        },
        "dataloader": {
            "batch_size": 1024,
            "batch_size_multiplier": 2,
            "batch_size_milestones": [1, 3, 6, 10, 15],
            # Add num_workers, pin_memory etc here if needed by model.fit's datamodule
            "num_workers": 8,
            "pin_memory": False,
            "persistent_workers": False,
        },
        "experiment_name": "base_fvt_placeholder",  # Required by TrainingInfo
    }
    logger.info(f"Base FvT hyperparameters template: {base_fvt_hparams}")

    base_fvt_train_ratio = 0.5
    seeds = range(50)
    logger.info(
        f"Generating datasets for {len(seeds)} seeds with train ratio: {base_fvt_train_ratio}"
    )

    t_data_start = time.perf_counter()
    # Make sure make_tinfos uses the logger instance
    if os.path.exists("test/tmp_data/stacked_fvt_train_dsets.pkl") and os.path.exists(
        "test/tmp_data/stacked_fvt_val_dsets.pkl"
    ):
        with open("test/tmp_data/stacked_fvt_train_dsets.pkl", "rb") as f:
            base_fvt_train_dsets = pickle.load(f)
        with open("test/tmp_data/stacked_fvt_val_dsets.pkl", "rb") as f:
            base_fvt_val_dsets = pickle.load(f)
    else:
        tinfos, base_fvt_train_dsets, base_fvt_val_dsets = make_tinfos(
            ms_hparams,
            base_fvt_train_ratio=base_fvt_train_ratio,
            seeds=seeds,
            base_fvt_hparams=base_fvt_hparams,
        )
        with open("test/tmp_data/stacked_fvt_train_dsets.pkl", "wb") as f:
            pickle.dump(base_fvt_train_dsets, f)
        with open("test/tmp_data/stacked_fvt_val_dsets.pkl", "wb") as f:
            pickle.dump(base_fvt_val_dsets, f)

    t_data_end = time.perf_counter()
    logger.info(f"Dataset generation took {t_data_end - t_data_start:.2f} seconds.")

    num_classes = 2  # Assuming binary classification
    # Infer input features from the first dataset
    # Assumes all datasets have the same feature dimension
    if not base_fvt_train_dsets:
        logger.error("No training datasets were generated.")
        raise ValueError("Dataset generation failed.")
    input_features_dim = base_fvt_train_dsets[0].tensors[0].shape[1]
    logger.info(f"Inferred input feature dimension: {input_features_dim}")

    # Determine dim_input_jet_features assuming 4 jets contribute equally
    # This might need adjustment based on actual feature engineering
    if input_features_dim % 4 == 0:
        dim_input_jet_features = input_features_dim // 4
        logger.info(
            f"Assuming 4 jets, inferred dim_input_jet_features: {dim_input_jet_features}"
        )
    else:
        logger.warning(
            f"Input feature dimension {input_features_dim} not divisible by 4. Setting dim_input_jet_features to 4 as default."
        )
        dim_input_jet_features = 4  # Default fallback

    # Instantiate StackedFvTClassifier
    stacked_hparams = {
        "num_stacks": len(seeds),  # Use actual number of seeds
        "num_classes": num_classes,
        "dim_input_jet_features": dim_input_jet_features,
        "dim_dijet_features": base_fvt_hparams["dim_dijet_features"],
        "dim_quadjet_features": base_fvt_hparams["dim_quadjet_features"],
        "run_name": "test_stacked_fvt_run",
        "depth": base_fvt_hparams["depth"],
        "repr_norm": base_fvt_hparams["repr_norm"],
        # Ensure device is passed if model __init__ expects it explicitly
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    logger.info(f"Instantiating StackedFvTClassifier with hparams: {stacked_hparams}")
    try:
        model = StackedFvTClassifier(**stacked_hparams)
        logger.info("Model instantiated successfully.")
    except Exception as e:
        logger.error(f"Failed to instantiate StackedFvTClassifier: {e}", exc_info=True)
        # Clean up file handler before raising
        root_logger.removeHandler(file_handler)
        file_handler.close()
        raise

    # Prepare arguments for the custom model.fit method
    fit_args = {
        "train_datasets": base_fvt_train_dsets,
        "val_datasets": base_fvt_val_dsets,
        "max_epochs": base_fvt_hparams["max_epochs"],
        "train_seed": base_fvt_hparams[
            "train_seed"
        ],  # Use the base seed for overall training run
        "save_checkpoint": True,  # As per original code
        "callbacks": [],  # Pass any custom callbacks if needed
        "tb_log_dir": "test_stacked_fvt_tb_logs",  # TensorBoard log dir
        "optimizer_config": base_fvt_hparams["optimizer"],
        "lr_scheduler_config": base_fvt_hparams["lr_scheduler"],
        "early_stop_patience": base_fvt_hparams["early_stop_patience"],
        "dataloader_config": base_fvt_hparams["dataloader"],
        "file_handler": file_handler,  # Pass the file handler
    }
    # Avoid logging potentially huge datasets in fit_args
    fit_args_loggable = {
        k: v for k, v in fit_args.items() if k not in ["train_datasets", "val_datasets"]
    }
    logger.info(f"Calling custom model.fit with arguments: {fit_args_loggable}")

    t0 = time.perf_counter()
    logger.info("Starting model.fit...")
    try:
        model.fit(**fit_args)
        t1 = time.perf_counter()
        logger.info("model.fit completed successfully.")
    except Exception as e:
        logger.exception(
            "Exception occurred during model.fit"
        )  # Automatically includes traceback
        # Clean up file handler so pytest doesn't hang on it
        root_logger.removeHandler(file_handler)
        file_handler.close()
        raise  # Re-raise the exception to fail the test

    elapsed = t1 - t0
    logger.info(f"Training {len(seeds)} tasks took {elapsed:.2f} seconds.")

    # --- Final Check (Optional) ---
    # Could add assertions here about final logged metrics if the custom fit method returns them
    # or logs them in a predictable way accessible after training.
    # For now, success is defined as completing without errors.
    logger.info("--- Multi-Task Test Completed Successfully --- ")

    # Clean up file handler
    root_logger.removeHandler(file_handler)
    file_handler.close()


if __name__ == "__main__":
    run_multitask_test()
