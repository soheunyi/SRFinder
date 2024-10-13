import pathlib
import time
from itertools import product

import numpy as np
import torch

from fvt_classifier import FvTClassifier
from training_info import TrainingInfoV2
from tst_info import TSTInfo
from events_data import events_from_scdinfo
import pytorch_lightning as pl

n_3b = 100_0000
device = torch.device("cuda")
experiment_name = "counting_test_v2"
signal_filename = "HH4b_picoAOD.h5"
ratio_4b = 0.5

seeds = [0, 1, 2]

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

hparam_filter = {
    "experiment_name": lambda x: x in [experiment_name],
    "n_3b": n_3b,
    "seed": lambda x: x in seeds,
    "signal_ratio": 0.0,
}
hashes = TSTInfo.find(hparam_filter, sort_by=["seed", "signal_ratio"])

for tstinfo_hash in hashes:
    tstinfo = TSTInfo.load(tstinfo_hash)
    seed = tstinfo.hparams["seed"]
    print(
        f"n_3b={tstinfo.hparams['n_3b']}, signal_ratio={tstinfo.hparams['signal_ratio']}, seed={tstinfo.hparams['seed']}"
    )
    base_fvt_tinfo_hash = tstinfo.base_fvt_tinfo_hash
    base_fvt_tinfo = TrainingInfoV2.load(base_fvt_tinfo_hash)

    train_scdinfo, val_scdinfo = base_fvt_tinfo.fetch_train_val_scdinfo()
    events_train = events_from_scdinfo(train_scdinfo, features, signal_filename)
    events_val = events_from_scdinfo(val_scdinfo, features, signal_filename)
    events_tst = events_from_scdinfo(tstinfo.scdinfo_tst, features, signal_filename)
    events_train.shuffle(seed=seed)
    events_val.shuffle(seed=seed)
    events_tst.shuffle(seed=seed)

    # batch_size = base_fvt_tinfo.hparams["batch_size"] # double the batch size to fit the kernel matrix
    batch_size = 2**10

    events_train.fit_batch_size(batch_size)
    events_val.fit_batch_size(batch_size)

    timestamp = int(time.time())

    configs = [
        {
            "batch_schedule": True,
            "batch_milestones": (1, 3, 6, 10, 15),
            "init_lr": 1e-2,
            "lr_schedule": True,
            "min_lr": 1e-3,
            "lr_factor": 0.5,
            "lr_patience": 10,
            "depth": {"encoder": 4, "decoder": 4},
        },
        {
            "batch_schedule": True,
            "batch_milestones": (1, 3, 6, 10, 15),
            "init_lr": 1e-2,
            "lr_schedule": True,
            "min_lr": 1e-3,
            "lr_factor": 0.5,
            "lr_patience": 10,
            "depth": {"encoder": 4, "decoder": 1},
        },
        {
            "batch_schedule": True,
            "batch_milestones": (1, 3, 6, 10, 15),
            "init_lr": 1e-2,
            "lr_schedule": True,
            "min_lr": 2e-4,
            "lr_factor": 0.5,
            "lr_patience": 10,
            "depth": {"encoder": 4, "decoder": 1},
        },
    ]

    for config in configs:
        batch_schedule = config["batch_schedule"]
        batch_milestones = config["batch_milestones"]
        init_lr = config["init_lr"]
        lr_schedule = config["lr_schedule"]
        min_lr = config["min_lr"]
        lr_factor = config["lr_factor"]
        lr_patience = config["lr_patience"]
        run_name = f"bs={batch_schedule}_bs_milestones={batch_milestones}_init_lr={init_lr}_lrs={lr_schedule}_min_lr={min_lr}_lr_factor={lr_factor}_lr_patience={lr_patience}"

        # IMPORTANT: For reproducibility, weight initialization is fixed
        pl.seed_everything(seed)
        base_model_new = FvTClassifier(
            num_classes=2,
            dim_input_jet_features=4,
            dim_dijet_features=base_fvt_tinfo.hparams["dim_dijet_features"],
            dim_quadjet_features=base_fvt_tinfo.hparams["dim_quadjet_features"],
            run_name=f"{run_name}_seed={seed}_timestamp={timestamp}",
            device=device,
            depth={"encoder": 4, "decoder": 1},
        )

        if lr_schedule:
            lr_scheduler_config = {
                "type": "ReduceLROnPlateau",
                "factor": lr_factor,
                "threshold": 0.0001,
                "patience": lr_patience,
                "cooldown": 1,
                "min_lr": min_lr,
            }
        else:
            lr_scheduler_config = {"type": "none"}

        if batch_schedule:
            dataloader_config = {
                "batch_size": batch_size,
                "batch_size_milestones": batch_milestones,
                "batch_size_multiplier": 2,
            }
        else:
            dataloader_config = {"batch_size": batch_size}

        base_model_new.fit(
            events_train.to_tensor_dataset(),
            events_val.to_tensor_dataset(),
            max_epochs=200,
            train_seed=seed,
            save_checkpoint=False,
            optimizer_config={"type": "Adam", "lr": init_lr},
            lr_scheduler_config=lr_scheduler_config,
            dataloader_config=dataloader_config,
            tb_log_dir="training_ablation_2",
        )
