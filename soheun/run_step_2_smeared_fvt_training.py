# ad hoc script to run counting test
import click
import numpy as np
from step_2_smeared_fvt_training import routine
import yaml


def edit_config(
    config,
    seed,
    signal_ratio,
):
    config["smeared_fvt"]["data_seed"] = seed
    config["smeared_fvt"]["train_seed"] = seed
    config["smeared_fvt"]["model_seed"] = seed
    config["smearing"]["seed"] = seed
    config["dataset"]["seed"] = seed
    config["dataset"]["signal_ratio"] = signal_ratio

    return config


@click.command()
@click.option("--config", type=str)
@click.option("--seed-start", type=int, default=0)
@click.option("--seed-end", type=int, default=10)
@click.option("--signal-ratio", type=float, default=0.0)
def main(config, seed_start, seed_end, signal_ratio):
    seeds = np.arange(seed_start, seed_end)
    print("seeds: ", seeds)
    print("config: ", config)

    with open(config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    for seed in seeds:
        config = edit_config(config, seed, signal_ratio)
        routine(config)


if __name__ == "__main__":
    main()
