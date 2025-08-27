import logging
import time
import pandas as pd
import click
import yaml

from step_2_preprocessing import check_and_get_smeared_fvt_hparams, get_step_2_tinfo
from train_stacked_attention_classifiers_from_tinfo import (
    train_stacked_attention_classifiers,
)
from save_aux_info import step_2_save_aux_info
from training_info import TrainingInfo

###########################################################################################
###########################################################################################
# For each instance of experiment, there would be two FvTClassifier models to be trained:
# 1. base_fvt_model: to learn 3b vs 4b on the mother (training) dataset
# 2. CR_fvt_model: to learn 3b vs 4b on the control region for background estimation
# We require YAML file to specify the hyperparameters separately for each model.
# We will save the trained models in the checkpoints.
# For smear-based SR definition, there is a AttentionClassifier to be trained, but we first
# do not save them in the checkpoints.


def routine(configs: list[dict], file_handler: logging.FileHandler | None = None):
    print("Current Time: ", pd.Timestamp.now(), flush=True)
    print(f"Checking smeared FvT configs for {len(configs)} configs", flush=True)
    smeared_fvt_hparams_list = [
        check_and_get_smeared_fvt_hparams(config) for config in configs
    ]
    smeared_fvt_tinfos: list[TrainingInfo] = []
    for smeared_fvt_hparams in smeared_fvt_hparams_list:
        smeared_fvt_tinfo = get_step_2_tinfo(smeared_fvt_hparams)
        smeared_fvt_tinfos.append(smeared_fvt_tinfo)

    print(
        "Training stacked smeared FvT models with {} configs".format(
            len(smeared_fvt_tinfos)
        ),
        flush=True,
    )
    start_time = time.perf_counter()
    train_stacked_attention_classifiers(smeared_fvt_tinfos, file_handler)
    end_time = time.perf_counter()
    h, m, s = (
        (end_time - start_time) // 3600,
        ((end_time - start_time) % 3600) // 60,
        (end_time - start_time) % 60,
    )
    print(
        f"Time taken to train stacked smeared FvT models: {h:.0f}h {m:.0f}m {s:.0f}s",
        flush=True,
    )

    for smeared_fvt_tinfo in smeared_fvt_tinfos:
        step_2_save_aux_info(smeared_fvt_tinfo)


@click.command()
@click.option("--config", type=str)
def main(config):
    with open(config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    routine(config)


if __name__ == "__main__":
    # load base yml file

    main()
