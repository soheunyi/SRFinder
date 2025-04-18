#!/usr/bin/env python3
"""
Quick speed test for FvTClassifier training: compares baseline vs. optimized DataLoader settings.
Run with:
  python soheun/test_training_speed.py
"""
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset

# Import classifier and data module
import sys
import pathlib

# allow imports from project root
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.absolute()))
from fvt_classifier import FvTClassifier
from data_modules import FvTDataModule
from training_info import TrainingInfo
from dataset import MotherSamples
from constants import FEATURES


def make_synthetic_dataset(N=4096, dim_j=2):
    """Create a random dataset of size N for testing."""
    # x shape: (N, dim_j * 4) flattened jet features
    x = torch.randn(N, dim_j * 4)
    y = torch.randint(0, 2, (N,))
    w = torch.ones(N)
    return TensorDataset(x, y, w)


def train_one_epoch(model, datamodule):
    """Run one training epoch with a simple loop."""
    # Load entire dataset into GPU memory
    loader = datamodule.train_dataloader()
    all_data = []
    for x, y, w in loader:
        all_data.append((x.to(model.device), y.to(model.device)))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    t0 = time.perf_counter()

    # Train on GPU-resident data
    for x, y in all_data:
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

    t1 = time.perf_counter()
    return t1 - t0


def run_test(name, num_workers, pin_memory, persistent_workers, preload_to_gpu):
    # Create config similar to better_fvt_training.yml
    config = {
        "experiment_name": "speed_test",
        "dataset": {
            "signal_filename": "HH4b_picoAOD.h5",
            "signal_ratio": 0.0,
            "n_3b": 100_0000,
            "ratio_4b": 0.5,
            "seed": 0,
            "base_fvt_train_ratio": 0.5,
        },
        "base_fvt": {
            "model": "FvTClassifier",
            "dim_dijet_features": 6,
            "dim_quadjet_features": 6,
            "depth": {"encoder": 4, "decoder": 1},
            "fit_batch_size": 1024,
            "model_seed": 0,
            "train_seed": 0,
            "data_seed": 0,
            "max_epochs": 10,  # Just test one epoch
            "val_ratio": 0.33,
            "early_stop_patience": None,
            "optimizer": {"type": "Adam", "lr": 0.01},
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
            },
        },
    }

    # Create mother samples
    ms_hparams = {
        "n_3b": config["dataset"]["n_3b"],
        "ratio_4b": config["dataset"]["ratio_4b"],
        "signal_ratio": config["dataset"]["signal_ratio"],
        "signal_filename": config["dataset"]["signal_filename"],
        "seed": config["dataset"]["seed"],
    }

    # Find or create mother samples
    hashes = MotherSamples.find(ms_hparams, from_metadata=False)
    if len(hashes) == 0:
        raise ValueError("No mother samples found for the given parameters")
    elif len(hashes) > 1:
        raise ValueError("Number of mother samples must be one")
    ms_hash = hashes[0]
    mother_samples = MotherSamples.load(ms_hash)

    # Split the mother dataset into train and test
    ms_len = len(mother_samples.scdinfo)
    ms_idx = np.zeros(ms_len, dtype=bool)
    ms_idx[: int(ms_len * config["dataset"]["base_fvt_train_ratio"])] = True
    np.random.seed(config["dataset"]["seed"])
    np.random.shuffle(ms_idx)

    # Create training info
    base_fvt_hparams = config["base_fvt"]
    base_fvt_hparams["experiment_name"] = config["experiment_name"]
    base_fvt_hparams["dataset"] = config["dataset"]
    base_fvt_hparams["step"] = 1

    base_fvt_tinfo = TrainingInfo(base_fvt_hparams, ms_hash=ms_hash, ms_idx=ms_idx)
    print("Base FvT Training Hash: ", base_fvt_tinfo.hash)

    # Get train and validation datasets
    base_fvt_train_dset, base_fvt_val_dset = (
        base_fvt_tinfo.fetch_train_val_tensor_datasets(FEATURES, "fourTag", "weight")
    )

    # Initialize data module with desired settings
    dm = FvTDataModule(
        train_dataset=base_fvt_train_dset,
        val_dataset=base_fvt_val_dset,
        batch_size=config["base_fvt"]["dataloader"]["batch_size"],
        num_workers=num_workers,
        batch_size_milestones=config["base_fvt"]["dataloader"]["batch_size_milestones"],
        batch_size_multiplier=config["base_fvt"]["dataloader"]["batch_size_multiplier"],
    )
    # Override performance settings
    dm.pin_memory = pin_memory
    dm.persistent_workers = persistent_workers

    # Initialize model
    model = FvTClassifier(
        num_classes=2,
        dim_input_jet_features=4,
        dim_dijet_features=config["base_fvt"]["dim_dijet_features"],
        dim_quadjet_features=config["base_fvt"]["dim_quadjet_features"],
        run_name=name,
        depth=config["base_fvt"]["depth"],
    )

    # Time the training
    t0 = time.perf_counter()
    model.fit(
        base_fvt_train_dset,
        base_fvt_val_dset,
        max_epochs=config["base_fvt"]["max_epochs"],
        train_seed=config["base_fvt"]["train_seed"],
        save_checkpoint=False,
        callbacks=[],
        tb_log_dir="_".join(
            [config["experiment_name"], str(config["dataset"]["signal_ratio"])]
        ),
        optimizer_config=config["base_fvt"]["optimizer"],
        lr_scheduler_config=config["base_fvt"]["lr_scheduler"],
        early_stop_patience=config["base_fvt"]["early_stop_patience"],
        dataloader_config=config["base_fvt"]["dataloader"],
        preload_to_gpu=preload_to_gpu,
    )
    t1 = time.perf_counter()
    return t1 - t0


def main():
    # Test with preload_to_gpu=False (standard loading)
    t_standard = run_test(
        name="standard",
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        preload_to_gpu=False,
    )

    # Test with preload_to_gpu=True (GPU resident data)
    t_gpu = run_test(
        name="gpu_resident",
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        preload_to_gpu=True,
    )

    print(f"Standard loading epoch time: {t_standard:.3f}s")
    print(f"GPU resident data epoch time: {t_gpu:.3f}s")
    if t_gpu > 0:
        print(f"Speedup: {t_standard/t_gpu:.2f}×")


if __name__ == "__main__":
    main()
