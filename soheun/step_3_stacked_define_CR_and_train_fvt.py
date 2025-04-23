import logging
import pandas as pd
import click
import yaml
from tqdm import tqdm
from fvt_classifier import FvTClassifier
from step_3_preprocessing import check_and_get_CR_fvt_hparams, get_step_3_tinfo_events
from train_stacked_fvts_from_tinfo import train_stacked_fvt

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
    print("Experiment Configuration")
    print("Current Time: ", pd.Timestamp.now())

    CR_fvt_hparams_list = [check_and_get_CR_fvt_hparams(config) for config in configs]
    CR_fvt_tinfos = [
        get_step_3_tinfo_events(CR_fvt_hparams)[0]
        for CR_fvt_hparams in CR_fvt_hparams_list
    ]

    stacked_model = train_stacked_fvt(CR_fvt_tinfos, file_handler)

    for i, CR_fvt_tinfo in tqdm(enumerate(CR_fvt_tinfos), total=len(CR_fvt_tinfos)):
        _, events_train, events_tst, SR_idx_train, SR_idx = get_step_3_tinfo_events(
            CR_fvt_tinfo
        )
        events_train_SR = events_train[SR_idx_train]
        events_tst_SR = events_tst[SR_idx]

        CR_fvt_model: FvTClassifier = stacked_model.fvt_classifiers[i]
        fvt_scores_train_SR = (
            CR_fvt_model.predict(events_train_SR.X_torch)[:, 1].detach().cpu().numpy()
        )
        fvt_scores_tst_SR = (
            CR_fvt_model.predict(events_tst_SR.X_torch)[:, 1].detach().cpu().numpy()
        )

        CR_fvt_tinfo.update_aux_info(
            description=f"FvT on Control Region",
            step=3,
            fvt_scores_train_SR=fvt_scores_train_SR,
            fvt_scores_tst_SR=fvt_scores_tst_SR,
        )
        CR_fvt_tinfo.save()


@click.command()
@click.option("--configs", type=str)
def main(configs):
    with open(configs, "r") as ymlfile:
        configs = yaml.safe_load(ymlfile)

    routine(configs)


if __name__ == "__main__":
    main()
