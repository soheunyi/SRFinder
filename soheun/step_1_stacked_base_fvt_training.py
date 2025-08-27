import logging
import time
import pandas as pd
import click
import yaml


from step_1_preprocessing import check_and_get_base_fvt_hparams, get_step_1_tinfo
from train_stacked_fvts_from_tinfo import train_stacked_fvt
from training_info import TrainingInfo
from save_aux_info import step_1_save_aux_info

###########################################################################################
###########################################################################################
# For each instance of experiment, there would be two FvTClassifier models to be trained:
# 1. base_fvt_model: to learn 3b vs 4b on the mother (training) dataset
# 2. CR_fvt_model: to learn 3b vs 4b on the control region for background estimation
# We require YAML file to specify the hyperparameters separately for each model.
# We will save the trained models in the checkpoints.
# For smear-based SR definition, there is a AttentionClassifier to be trained, but we first
# do not save them in the checkpoints.
###########################################################################################
W_4B_CUT_MIN = 0.001
W_4B_CUT_MAX = 0.999


def routine(configs: list[dict], file_handler: logging.FileHandler | None = None):
    print("Current Time: ", pd.Timestamp.now(), flush=True)
    print(f"Checking base FvT configs for {len(configs)} configs", flush=True)
    base_fvt_hparams_list = [
        check_and_get_base_fvt_hparams(config) for config in configs
    ]
    base_fvt_tinfos: list[TrainingInfo] = []
    for base_fvt_hparams in base_fvt_hparams_list:
        base_fvt_tinfo = get_step_1_tinfo(base_fvt_hparams)
        base_fvt_tinfos.append(base_fvt_tinfo)

    print(
        "Training stacked base FvT models with {} configs".format(len(base_fvt_tinfos)),
        flush=True,
    )
    start_time = time.perf_counter()
    train_stacked_fvt(base_fvt_tinfos, file_handler)
    end_time = time.perf_counter()
    h, m, s = (
        (end_time - start_time) // 3600,
        ((end_time - start_time) % 3600) // 60,
        (end_time - start_time) % 60,
    )
    print(
        f"Time taken to train stacked base FvT models: {h:.0f}h {m:.0f}m {s:.0f}s",
        flush=True,
    )

    for base_fvt_tinfo in base_fvt_tinfos:
        step_1_save_aux_info(base_fvt_tinfo)


@click.command()
@click.option("--configs", type=str)
def main(configs):
    with open(configs, "r") as ymlfile:
        configs = yaml.safe_load(ymlfile)

    routine(configs)


if __name__ == "__main__":
    # load base yml file

    main()
