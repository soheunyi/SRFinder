import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def split_dataset(
    X: torch.Tensor,
    y: torch.Tensor,
    valid_split: float,
    train_batch_size: int = 32,
    valid_batch_size: int = 32,
    seed: int = 42,
):
    np.random.seed(seed)
    n_samples = X.shape[0]
    n_valid = int(n_samples * valid_split)
    valid_indices = np.random.choice(n_samples, n_valid, replace=False)

    X_train, y_train = X[~valid_indices], y[~valid_indices]
    X_valid, y_valid = X[valid_indices], y[valid_indices]

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=train_batch_size,
        shuffle=True,
    )

    valid_loader = DataLoader(
        TensorDataset(X_valid, y_valid),
        batch_size=valid_batch_size,
        shuffle=True,
    )

    return train_loader, valid_loader
