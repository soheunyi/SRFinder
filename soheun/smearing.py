# from typing import Literal
import numpy as np
from sklearn.model_selection import train_test_split
import torch

from utils import require_keys
from dataset import SCDatasetInfo

# import torch.nn.functional as F
# import pytorch_lightning as pl
# from network_blocks import conv1d
# from torch.utils.data import DataLoader, TensorDataset
# from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar

# from fvt_classifier import FvTClassifier


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


# class AttentionClassifier(pl.LightningModule):
#     def __init__(
#         self,
#         input_size,
#         num_classes,
#         learning_rate=1e-3,
#         random_state=42,
#         fvt_hash=None,
#         hidden_size=None,
#     ):
#         super().__init__()
#         self.save_hyperparameters()

#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_classes = num_classes
#         self.learning_rate = learning_rate
#         self.random_state = random_state

#         torch.manual_seed(random_state)
#         np.random.seed(random_state)

#         if fvt_hash is not None:
#             fvt_model = FvTClassifier.load_from_checkpoint(
#                 f"./data/checkpoints/{fvt_hash}_best.ckpt"
#             )
#             fvt_model.eval()
#             self.select_q = fvt_model.select_q
#             self.out = fvt_model.out
#         else:
#             self.select_q = conv1d(
#                 input_size, 1, 1, name="quadjet selector", batchNorm=True
#             )
#             self.out = conv1d(input_size, num_classes, 1, name="out", batchNorm=True)

#         self.val_losses = []
#         self.val_weights = []

#     def forward(self, q: torch.Tensor):
#         n = q.shape[0]
#         q_score = self.select_q(q)
#         q_score = F.softmax(q_score, dim=-1)
#         event = torch.matmul(q, q_score.transpose(1, 2))
#         event = event.view(n, self.input_size, 1)

#         # project the final event-level pixel into the class score space
#         class_score = self.out(event)
#         class_score = class_score.view(n, self.num_classes)

#         return class_score

#     def training_step(self, batch, batch_idx):
#         x, y, w = batch
#         logits = self(x)
#         loss = torch.sum(F.cross_entropy(logits, y) * w)
#         self.log("train_loss", loss)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y, w = batch
#         logits = self(x)
#         loss = torch.sum(F.cross_entropy(logits, y) * w)
#         self.log("val_loss", loss)
#         self.val_losses.append(loss)
#         self.val_weights.append(w.sum())
#         return loss

#     def on_validation_epoch_end(self):
#         val_losses = torch.tensor(self.val_losses)
#         val_weights = torch.tensor(self.val_weights)
#         avg_loss = torch.sum(val_losses) / torch.sum(val_weights)
#         self.log("avg_val_loss", avg_loss, prog_bar=True)

#         self.val_losses = []
#         self.val_weights = []

#         self.train()

#     def test_step(self, batch, batch_idx):
#         x, y, w = batch
#         logits = self(x)
#         loss = torch.sum(F.cross_entropy(logits, y) * w)
#         self.log("test_loss", loss)
#         return loss

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
#         return optimizer

#     def predict(self, x) -> np.ndarray:
#         self.eval()
#         if isinstance(x, np.ndarray):
#             x = torch.tensor(x, dtype=torch.float32).to(self.device)
#         logits = self(x)
#         return torch.softmax(logits, dim=1).detach().cpu().numpy()

#     def fit(self, X, y, w, max_epochs=10):
#         self.train()
#         X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
#             X, y, w, test_size=1 / 3, random_state=self.hparams.random_state
#         )

#         # fit batch size to the number of samples
#         batch_size = 2**10

#         X_train = X_train[: len(X_train) // batch_size * batch_size]
#         X_train = torch.tensor(X_train, dtype=torch.float32)
#         y_train = y_train[: len(y_train) // batch_size * batch_size]
#         y_train = torch.tensor(y_train, dtype=torch.long)
#         w_train = w_train[: len(w_train) // batch_size * batch_size]
#         w_train = torch.tensor(w_train, dtype=torch.float32)

#         X_val = X_val[: len(X_val) // batch_size * batch_size]
#         X_val = torch.tensor(X_val, dtype=torch.float32)
#         y_val = y_val[: len(y_val) // batch_size * batch_size]
#         y_val = torch.tensor(y_val, dtype=torch.long)
#         w_val = w_val[: len(w_val) // batch_size * batch_size]
#         w_val = torch.tensor(w_val, dtype=torch.float32)

#         train_dataset = TensorDataset(X_train, y_train, w_train)
#         val_dataset = TensorDataset(X_val, y_val, w_val)

#         train_loader = DataLoader(
#             train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
#         )
#         val_loader = DataLoader(
#             val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
#         )
#         early_stop_callback = EarlyStopping(
#             monitor="avg_val_loss", min_delta=0.0, patience=5, mode="min"
#         )
#         progress_bar = TQDMProgressBar(
#             refresh_rate=max(1, (len(train_loader) // batch_size) // 10)
#         )
#         trainer = pl.Trainer(
#             max_epochs=max_epochs,
#             callbacks=[early_stop_callback, progress_bar],
#             logger=False,
#             enable_model_summary=False,
#             enable_checkpointing=False,
#         )

#         trainer.fit(self, train_loader, val_loader)


# def smeared_density_ratio(
#     X: np.ndarray,
#     y: np.ndarray,
#     w: np.ndarray,
#     noise_scale: float,
#     base_noise_scale: Literal["minmax", "std"] = "minmax",
#     pretrained_fvt_hash: str = None,
#     max_epochs: int = 10,
#     seed: int = 42,
# ):
#     """
#     Calculate the smeared density ratio.

#     Parameters
#     ----------
#     X : np.ndarray
#         Training data.
#     y : np.ndarray
#         Training labels.
#     w : np.ndarray
#         Training weights.
#     noise_scale : float
#         Scale of the noise.
#     base_noise_scale : Literal["minmax", "std"], optional
#         Base scale of the noise, by default "minmax".
#     pretrained_fvt_hash : str, optional
#         Hash of the pretrained FvT model, by default None.
#     training : bool, optional
#         Whether to train the model, by default True.
#     """

#     assert len(X) == len(y) == len(w)

#     n = X.shape[0]
#     n_classes = 2

#     # add noise to the data
#     base_noise_scale = (
#         np.std(X, axis=0)
#         if base_noise_scale == "std"
#         else np.max(X, axis=0) - np.min(X, axis=0)
#     )
#     X = X + np.random.normal(0, noise_scale, X.shape) * base_noise_scale

#     # create the model
#     model = AttentionClassifier(
#         input_size=X.shape[1],
#         num_classes=n_classes,
#         learning_rate=1e-3,
#         random_state=seed,
#         fvt_hash=pretrained_fvt_hash,
#     )

#     model.fit(X, y, w, max_epochs=max_epochs)
#     model.eval()

#     def density_ratio(x: np.ndarray) -> np.ndarray:
#         prob_4b = model.predict(x)
#         return prob_4b[:, 1] / prob_4b[:, 0]

#     return density_ratio


# import time
# import pytorch_lightning as pl
# from events_data import events_from_scdinfo
# from training_info import TrainingInfo

# n_3b = 100_0000
# device = torch.device("cuda")
# experiment_name = "better_base_fvt_training"
# signal_filename = "HH4b_picoAOD.h5"
# ratio_4b = 0.5

# seeds = [0]
# hparam_filter = {
#     "experiment_name": lambda x: x in [experiment_name],
#     "dataset": lambda x: all(
#         [x["seed"] in seeds, x["n_3b"] == n_3b, x["signal_ratio"] == 0.02]
#     ),
# }
# hashes = TrainingInfo.find(hparam_filter)

# noise_scale = 0.5
# depth = 8
# batch_size = 1024
# max_epochs = 30

# gamma_dict = {}

# for tinfo_hash in hashes:
#     tinfo = TrainingInfo.load(tinfo_hash)
#     model = tinfo.load_trained_model("best")
#     seed = tinfo.hparams["dataset"]["seed"]
#     n_3b = tinfo.hparams["dataset"]["n_3b"]
#     signal_ratio = tinfo.hparams["dataset"]["signal_ratio"]
#     print(f"seed={seed}, n_3b={n_3b}, signal_ratio={signal_ratio}")

#     model.eval()
#     model.to(device)
#     events_train = events_from_scdinfo(tinfo.scdinfo, features, signal_filename)
#     events_train.shuffle(seed=seed)
#     q_repr, view_scores = model.representations(events_train.X_torch)

#     from attention_classifier import AttentionClassifier
#     from sklearn.model_selection import train_test_split
#     from torch.utils.data import TensorDataset

#     tb_log_dir = "smeared_fvt_training_tmp"
#     optimizer_config = {
#         "type": "Adam",
#         "lr": 1e-2,
#     }
#     lr_scheduler_config = {
#         "type": "ReduceLROnPlateau",
#         "factor": 0.25,
#         "patience": 3,
#         "min_lr": 2e-4,
#         "cooldown": 1,
#         "threshold": 1e-4,
#     }
#     early_stop_patience = None
#     dataloader_config = {
#         "batch_size": batch_size,
#         "batch_size_milestones": [1, 3, 6, 10, 15],
#         "batch_size_multiplier": 2,
#     }

#     from dataset import MotherSamples

#     ms_scdinfo = MotherSamples.load(tinfo._ms_hash).scdinfo
#     events_tst = events_from_scdinfo(
#         ms_scdinfo[~tinfo._ms_idx], features, signal_filename
#     )
#     events_tst.shuffle(seed=seed)
#     probs_4b_tst = (
#         model.predict(events_tst.X_torch, do_tqdm=True)[:, 1].to("cpu").numpy()
#     )
#     q_repr_tst, _ = model.representations(events_tst.X_torch)
#     gamma_tst = probs_4b_tst / (1 - probs_4b_tst)
#     gamma_dict[0.0] = gamma_tst

#     for noise_scale in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
#         q_repr_smear = smear_features(
#             q_repr.numpy(), noise_scale, seed, hard_cutoff=False, scale_mode="std"
#         )
#         q_sm_train, q_sm_val, is_4b_train, is_4b_val, weights_train, weights_val = (
#             train_test_split(
#                 q_repr_smear,
#                 events_train.is_4b,
#                 events_train.weights,
#                 test_size=0.33,
#                 random_state=seed,
#             )
#         )
#         q_sm_train = torch.tensor(q_sm_train, dtype=torch.float32)
#         q_sm_val = torch.tensor(q_sm_val, dtype=torch.float32)
#         is_4b_train = torch.tensor(is_4b_train, dtype=torch.long)
#         is_4b_val = torch.tensor(is_4b_val, dtype=torch.long)
#         weights_train = torch.tensor(weights_train, dtype=torch.float32)
#         weights_val = torch.tensor(weights_val, dtype=torch.float32)

#         q_sm_train = q_sm_train[: batch_size * (len(q_sm_train) // batch_size)]
#         is_4b_train = is_4b_train[: batch_size * (len(is_4b_train) // batch_size)]
#         weights_train = weights_train[: batch_size * (len(weights_train) // batch_size)]

#         q_sm_val = q_sm_val[: batch_size * (len(q_sm_val) // batch_size)]
#         is_4b_val = is_4b_val[: batch_size * (len(is_4b_val) // batch_size)]
#         weights_val = weights_val[: batch_size * (len(weights_val) // batch_size)]

#         train_dataset = TensorDataset(q_sm_train, is_4b_train, weights_train)
#         val_dataset = TensorDataset(q_sm_val, is_4b_val, weights_val)

#         pl.seed_everything(seed)
#         att_classifier = AttentionClassifier(
#             dim_q=q_repr.shape[1],
#             num_classes=2,
#             depth=depth,
#             run_name=f"smeared_fvt_training_tmp_depth={depth}_seed={seed}_lrs=True_bsch=True_noise_scale={noise_scale}",
#         )

#         att_classifier.to(device)
#         att_classifier.fit(
#             train_dataset,
#             val_dataset,
#             max_epochs=max_epochs,
#             optimizer_config=optimizer_config,
#             lr_scheduler_config=lr_scheduler_config,
#             early_stop_patience=early_stop_patience,
#             dataloader_config=dataloader_config,
#             tb_log_dir=tb_log_dir,
#         )

#         att_classifier.eval()
#         att_classifier.to(device)
#         probs_4b_smeared_tst = (
#             att_classifier.predict(q_repr_tst, do_tqdm=True)[:, 1].to("cpu").numpy()
#         )

#         gamma_tilde_tst = probs_4b_smeared_tst / (1 - probs_4b_smeared_tst)

#         from plots import plot_sr_stats

#         fig, ax = plt.subplots(1, 1, figsize=(6, 6))
#         fig.suptitle(f"Noise Scale: {noise_scale}")
#         plot_sr_stats(events_tst, gamma_tst, ax, label="Original")
#         plot_sr_stats(events_tst, gamma_tilde_tst, ax, label="Smeared")
#         plot_sr_stats(
#             events_tst, gamma_tst / gamma_tilde_tst, ax, label="Original / Smeared"
#         )
#         plt.legend()
#         plt.show()

#         gamma_dict[noise_scale] = gamma_tilde_tst

import torch
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from attention_classifier import AttentionClassifier
from fvt_classifier import FvTClassifier
from dataset import MotherSamples
from events_data import EventsData


def generate_smeared_density_ratio(
    model: FvTClassifier,
    events_train: EventsData,
    smearing_config: dict,
    device: torch.device = torch.device("cuda"),
):
    require_keys(
        smearing_config,
        ["noise_scale", "seed", "hard_cutoff", "scale_mode"],
    )
    model.eval()
    model.to(device)

    q_repr_train, _ = model.representations(events_train.X_torch)
    q_repr_smear = smear_features(
        q_repr_train.numpy(),
        noise_scale=smearing_config["noise_scale"],
        seed=smearing_config["seed"],
        hard_cutoff=smearing_config["hard_cutoff"],
        scale_mode=smearing_config["scale_mode"],
    )

    q_sm_train, q_sm_val, is_4b_train, is_4b_val, weights_train, weights_val = (
        train_test_split(
            q_repr_smear,
            events_train.is_4b,
            events_train.weights,
            test_size=0.33,
            random_state=seed,
        )
    )

    # Convert to tensors and trim to batch size
    q_sm_train, is_4b_train, weights_train = [
        torch.tensor(x[: batch_size * (len(x) // batch_size)], dtype=torch.float32)
        for x in (q_sm_train, is_4b_train, weights_train)
    ]
    q_sm_val, is_4b_val, weights_val = [
        torch.tensor(x[: batch_size * (len(x) // batch_size)], dtype=torch.float32)
        for x in (q_sm_val, is_4b_val, weights_val)
    ]
    is_4b_train, is_4b_val = is_4b_train.long(), is_4b_val.long()

    # Create datasets
    train_dataset = TensorDataset(q_sm_train, is_4b_train, weights_train)
    val_dataset = TensorDataset(q_sm_val, is_4b_val, weights_val)

    # Initialize and train AttentionClassifier
    pl.seed_everything(seed)
    att_classifier = AttentionClassifier(
        dim_q=q_repr_train.shape[1],
        num_classes=2,
        depth=depth,
        run_name=f"smeared_fvt_training_tmp_depth={depth}_seed={seed}_lrs=True_bsch=True_noise_scale={noise_scale}",
    )

    att_classifier.to(device)
    att_classifier.fit(
        train_dataset,
        val_dataset,
        max_epochs=max_epochs,
        optimizer_config={"type": "Adam", "lr": 1e-2},
        lr_scheduler_config={
            "type": "ReduceLROnPlateau",
            "factor": 0.25,
            "patience": 3,
            "min_lr": 2e-4,
            "cooldown": 1,
            "threshold": 1e-4,
        },
        early_stop_patience=None,
        dataloader_config={
            "batch_size": batch_size,
            "batch_size_milestones": [1, 3, 6, 10, 15],
            "batch_size_multiplier": 2,
        },
        tb_log_dir="smeared_fvt_training_tmp",
    )

    # Generate smeared predictions
    att_classifier.eval()
    att_classifier.to(device)
    probs_4b_smeared_test = (
        att_classifier.predict(q_repr_test, do_tqdm=True)[:, 1].to("cpu").numpy()
    )

    # Calculate smeared gamma
    gamma_tilde_test = probs_4b_smeared_test / (1 - probs_4b_smeared_test)

    return gamma_test, gamma_tilde_test


# Usage example:
# gamma_test, gamma_tilde_test = generate_smeared_density_ratio(
#     model, events_train, events_test, noise_scale, seed
# )
