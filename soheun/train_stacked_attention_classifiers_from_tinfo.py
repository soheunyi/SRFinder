from member_initialization import initialize_members
from independent_data import stream_identity
from copy import deepcopy
import logging
import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset


from stacked_attention_classifier import StackedAttentionClassifier
from training_info import TrainingInfo
from utils import validate_consistent_hparams
from constants import FEATURES

DIM_Q = 6


###########################################################################################
###########################################################################################
# Get multiple TrainingInfo objects and train FvTClassifier models based on them
###########################################################################################


def train_stacked_attention_classifiers(
    tinfos: list[TrainingInfo],
    file_handler: logging.FileHandler | None = None,
):

    critical_hparams = [
        "experiment_name",
        "step",
        "model",
        "dim_quadjet_features",
        "depth",
        "fit_batch_size",
        "max_epochs",
        "val_ratio",
        "early_stop_patience",
        "optimizer.type",
        "optimizer.lr",
        "lr_scheduler.type",
        "lr_scheduler.factor",
        "lr_scheduler.threshold",
        "lr_scheduler.patience",
        "lr_scheduler.cooldown",
        "lr_scheduler.min_lr",
        "dataloader.batch_size",
        "dataloader.batch_size_multiplier",
        "dataloader.batch_size_milestones",
        "encoder_mode",
    ]

    num_stacks = len(tinfos)

    # Architecture and schedule must match; member seeds remain independent.
    consistent, mismatches = validate_consistent_hparams(
        [tinfo.hparams for tinfo in tinfos], critical_hparams
    )
    if not consistent:
        raise ValueError(f"Critical hyperparameters are not consistent: {mismatches}")

    model_seed = tinfos[0].hparams["model_seed"]
    depth = tinfos[0].hparams["depth"]
    run_names = [tinfo.hash for tinfo in tinfos]

    stacked_hparams = {
        "num_stacks": num_stacks,  # Use actual number of seeds
        "num_classes": 2,
        "dim_quadjet_features": DIM_Q,
        "run_names": run_names,
        "stacked_run_name": "_".join([run_names[0], str(num_stacks)]),
        "depth": depth,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    train_datasets = []
    train_lengths = []
    val_datasets = []
    val_lengths = []
    for tinfo in tinfos:
        train_dset, val_dset = tinfo.fetch_train_val_smeared_features(
            FEATURES, "fourTag", "weight", training_alignment=32, retain_validation=True
        )
        tinfo.update_aux_info(data_batching_policy={
            'version': 1, 'training_alignment': 32, 'retain_validation': True,
            'train_rows': len(train_dset), 'val_rows': len(val_dset),
            'shuffle_seed': int(tinfo.hparams['train_seed']),
        })
        train_datasets.append(train_dset)
        train_lengths.append(len(train_dset))
        val_datasets.append(val_dset)
        val_lengths.append(len(val_dset))

    print(train_lengths, flush=True)
    print(val_lengths, flush=True)
    # Align training rows to GBN groups (32), retain all validation rows, and
    # never trim further according to another estimator's dataset length.

    max_epochs = tinfos[0].hparams["max_epochs"]
    train_seed = tinfos[0].hparams["train_seed"]
    experiment_name = tinfos[0].hparams["experiment_name"]
    optimizer_config = tinfos[0].hparams["optimizer"]
    lr_scheduler_config = tinfos[0].hparams["lr_scheduler"]
    early_stop_patience = tinfos[0].hparams["early_stop_patience"]
    dataloader_config = deepcopy(tinfos[0].hparams["dataloader"])
    dataloader_config.setdefault("num_workers", 8)

    fit_args = {
        "independent_batches": True,
        "estimator_train_seeds": [int(t.hparams['train_seed']) for t in tinfos],
        "estimator_ids": [stream_identity(t.hparams) for t in tinfos],
        "execution_chunk_size": dataloader_config.get('execution_chunk_size', 0),
        "train_datasets": train_datasets,
        "val_datasets": val_datasets,
        "max_epochs": max_epochs,
        "train_seed": train_seed,
        "save_checkpoint": True,
        "callbacks": [],
        "tb_log_dir": "_".join([experiment_name, stacked_hparams["stacked_run_name"]]),
        "optimizer_config": optimizer_config,
        "lr_scheduler_config": lr_scheduler_config,
        "early_stop_patience": early_stop_patience,
        "dataloader_config": dataloader_config,
        "file_handler": file_handler,
    }

    pl.seed_everything(model_seed)
    stacked_model = StackedAttentionClassifier(**stacked_hparams)
    initialization = initialize_members(stacked_model.attention_classifiers, [t.hparams for t in tinfos])
    for tinfo, record in zip(tinfos, initialization):
        tinfo.update_aux_info(initialization_policy=record)
    stacked_model.fit(**fit_args)

    stacked_model.eval()
    stacked_model.to(
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    stacked_model: StackedAttentionClassifier

    return stacked_model
