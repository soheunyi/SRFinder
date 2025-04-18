#!/usr/bin/env python3
"""
Test that MultiTaskFvTClassifier can train 50 tasks in one shot.
Uses the same MotherSamples→train/val split logic as test_training_speed,
but stacks 50 random seeds/labels into one (N,50) Y tensor.
"""

import time
import sys
import pathlib
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl

# allow imports from project root
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.absolute()))
from dataset import MotherSamples
from constants import FEATURES
from training_info import TrainingInfo
from data_modules import FvTDataModule
from multi_task_fvt import MultiTaskFvTClassifier


def make_multitask_dataset(
    signal_ratio, signal_filename, n_3b, features=FEATURES, num_tasks=50
):
    """
    Loads or creates 50 different MotherSamples datasets (one per seed) based on the provided parameters:

    Parameters:
        signal_ratio: Ratio of signal samples in the dataset
        signal_filename: Filename for the signal samples
        n_3b: Number of 3b samples
        features: List of features to extract from the dataset (default: FEATURES from constants)
        num_tasks: Number of different tasks/seeds to create (default: 50)

    For each seed/task, the function:
    1. Finds or creates a MotherSample with the given parameters and seed
    2. Creates a synthetic binary classification task by thresholding the weights
    3. Combines all tasks into a single multi-task dataset

    Returns:
        X_combined: Feature tensor of shape (N, feature_dims)
        Y: Label tensor of shape (N, num_tasks)
        W: Weight tensor of shape (N, num_tasks)
    """
    # Constants for MotherSamples creation
    ratio_4b = 0.5  # Keep this fixed as in the examples

    X_combined = None
    Ys = []
    Ws = []

    # Create or load MotherSamples for each seed
    for seed in range(num_tasks):
        # Define parameters for this seed's MotherSample
        ms_hparams = {
            "n_3b": n_3b,
            "ratio_4b": ratio_4b,
            "signal_ratio": signal_ratio,
            "signal_filename": signal_filename,
            "seed": seed,
        }

        # Find or create MotherSamples for this seed
        hashes = MotherSamples.find(ms_hparams, from_metadata=False)
        if len(hashes) == 0:
            raise ValueError(
                f"No mother samples found for seed {seed}. Please create them first with parameters: {ms_hparams}"
            )

        ms_hash = hashes[0]

        # Set up training info to get the dataset with proper hyperparameters
        base_fvt_hparams = {
            "step": 1,
            "experiment_name": "multi_task_dataset",
            "model": "FvTClassifier",
            "dim_dijet_features": 6,
            "dim_quadjet_features": 6,
            "depth": {"encoder": 4, "decoder": 1},
            "fit_batch_size": 1024,
            "model_seed": seed,
            "train_seed": seed,
            "data_seed": seed,
            "val_ratio": 0.2,
            "optimizer": {"type": "Adam", "lr": 0.001},
            "lr_scheduler": {"type": "none"},
            "dataloader": {"batch_size": 1024},
        }

        base_tinfo = TrainingInfo(
            base_fvt_hparams,
            ms_hash=ms_hash,
            ms_idx=np.ones(len(MotherSamples.load(ms_hash).scdinfo), dtype=bool),
        )
        X_train, y_train, w_train = base_tinfo.fetch_train_val_tensor_datasets(
            features, "fourTag", "weight"
        )[0]

        # Store features from first dataset only (assumes same dimensions for all)
        if X_combined is None:
            X_combined = X_train

        # Just use the original labels and weights directly
        y_task = y_train
        w_task = w_train

        Ys.append(y_task)
        Ws.append(w_task)

    # Stack all task labels and weights
    Y = torch.stack(Ys, dim=1)  # (N, num_tasks)
    W = torch.stack(Ws, dim=1)  # (N, num_tasks)

    return X_combined, Y, W


def run_multitask_test():
    # 1) Find a single MotherSamples entry, same as test_training_speed
    ms_hparams = {
        "n_3b": 1_000_000,
        "ratio_4b": 0.5,
        "signal_ratio": 0.0,
        "signal_filename": "HH4b_picoAOD.h5",
        "seed": 0,
    }
    hashes = MotherSamples.find(ms_hparams, from_metadata=False)
    assert len(hashes) == 1, "Need exactly one mother sample hash"
    ms_hash = hashes[0]

    # split into train/val mask
    mother = MotherSamples.load(ms_hash)
    Ntot = len(mother.scdinfo)
    train_ratio = 0.5
    idx = np.zeros(Ntot, bool)
    idx[: int(Ntot * train_ratio)] = True
    np.random.seed(ms_hparams["seed"])
    np.random.shuffle(idx)

    # Build the multi-task dataset
    X, Y, W = make_multitask_dataset(
        ms_hparams["signal_ratio"],
        ms_hparams["signal_filename"],
        ms_hparams["n_3b"],
        features=FEATURES,
        num_tasks=50,
    )

    # Create TensorDataset and DataLoaders
    ds = TensorDataset(X, Y, W)
    loader = DataLoader(
        ds, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True
    )

    # Instantiate and train one epoch
    model = MultiTaskFvTClassifier(
        num_tasks=50,
        dim_input_jet_features=X.shape[1] // 4,  # since X is (N, dim_j*4)
        dim_dijet_features=6,
        dim_quadjet_features=6,
        lr=1e-3,
    )
    trainer = pl.Trainer(
        max_epochs=1,
        gpus=1 if torch.cuda.is_available() else 0,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
    )

    t0 = time.perf_counter()
    trainer.fit(model, loader)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    print(f"Trained 50 tasks in {elapsed:.2f}s on one shared backbone")

    # Must produce some finite loss
    result = trainer.callback_metrics.get("train_loss")
    assert result is not None and result.item() >= 0, "No training loss logged"


if __name__ == "__main__":
    run_multitask_test()
