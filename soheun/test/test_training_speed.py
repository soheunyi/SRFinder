#!/usr/bin/env python3
"""
Quick speed test for FvTClassifier training: compares baseline vs. optimized DataLoader settings.
Run with:
  python soheun/test_training_speed.py
"""
import time
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

# Import classifier and data module
import sys
import pathlib
# allow imports from project root
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.absolute()))
from soheun.fvt_classifier import FvTClassifier
from soheun.data_modules import FvTDataModule


def make_synthetic_dataset(N=4096, dim_j=2):
    """Create a random dataset of size N for testing."""
    # x shape: (N, dim_j * 4) flattened jet features
    x = torch.randn(N, dim_j * 4)
    y = torch.randint(0, 2, (N,))
    w = torch.ones(N)
    return TensorDataset(x, y, w)


def train_one_epoch(model, datamodule):
    """Run one training epoch with a simple loop."""
    loader = datamodule.train_dataloader()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    t0 = time.perf_counter()
    for x, y, w in loader:
        x, y = x.to(model.device), y.to(model.device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
    t1 = time.perf_counter()
    return t1 - t0


def run_test(name, num_workers, pin_memory, persistent_workers, prefetch_factor):
    ds = make_synthetic_dataset(N=4096, dim_j=2)
    # Initialize data module with desired settings
    dm = FvTDataModule(
        train_dataset=ds,
        val_dataset=ds,
        batch_size=128,
        num_workers=num_workers,
        batch_size_milestones=[],
        batch_size_multiplier=1,
    )
    # Override performance settings
    dm.pin_memory = pin_memory
    dm.persistent_workers = persistent_workers
    dm.prefetch_factor = prefetch_factor

    # Initialize model
    model = FvTClassifier(
        num_classes=2,
        dim_input_jet_features=2,
        dim_dijet_features=1,
        dim_quadjet_features=1,
        run_name=name,
    )
    # Run one epoch and return elapsed time
    elapsed = train_one_epoch(model, dm)
    return elapsed


def main():
    # Baseline: single-worker, no pinning, no prefetch
    t_base = run_test(
        name="baseline",
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=0,
    )
    # Optimized: multi-worker, pinned memory, persistent workers, prefetch
    t_opt = run_test(
        name="optimized",
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    print(f"Baseline epoch time: {t_base:.3f}s")
    print(f"Optimized epoch time: {t_opt:.3f}s")
    if t_opt > 0:
        print(f"Speedup: {t_base/t_opt:.2f}×")


if __name__ == "__main__":
    main()