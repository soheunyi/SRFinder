import numpy as np
import torch


def smear_features(
    X: np.ndarray | torch.Tensor,
    noise_scale: float,
    seed: int,
    hard_cutoff: bool = False,
    scale_mode: str = "std",
):
    X_type = type(X)
    if X_type == torch.Tensor:
        X = X.detach().cpu().numpy()
    elif X_type != np.ndarray:
        raise ValueError(f"Invalid type for X: {X_type}")

    if scale_mode == "std":
        base_scale = np.std(X, axis=0)
    elif scale_mode == "range":
        features_min = np.min(X, axis=0)
        features_max = np.max(X, axis=0)
        base_scale = features_max - features_min
    else:
        raise ValueError(f"Invalid scale_mode: {scale_mode}")

    accept_mask = np.zeros_like(X, dtype=bool)
    X_smeared = np.zeros_like(X)

    np.random.seed(seed)
    if hard_cutoff:
        while True:
            X_smeared[~accept_mask] = (
                X + noise_scale * base_scale * np.random.randn(*X.shape)
            )[~accept_mask]
            accept_mask = (X_smeared >= features_min) & (X_smeared <= features_max)
            if np.all(accept_mask):
                break
    else:
        X_smeared = X + noise_scale * base_scale * np.random.randn(*X.shape)

    if X_type == torch.Tensor:
        X_smeared = torch.tensor(X_smeared, dtype=torch.float32)

    return X_smeared
