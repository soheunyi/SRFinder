# ad hoc script to run counting test
import click
import numpy as np
from counting_test_v2 import routine
import yaml


def edit_config(
    config,
    seed,
    signal_ratio,
):
    config["seed"] = seed
    config["base_fvt"]["data_seed"] = seed
    config["base_fvt"]["train_seed"] = seed
    config["CR_fvt"]["data_seed"] = seed
    config["CR_fvt"]["train_seed"] = seed
    config["signal_ratio"] = signal_ratio

    return config


@click.command()
@click.option("--config", type=str)
def main(config):
    signal_ratios = [0.0, 0.01, 0.02]
    seeds = np.arange(0, 10)

    with open(config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    for seed in seeds:
        for signal_ratio in signal_ratios:
            config = edit_config(config, seed, signal_ratio)
            routine(config)


if __name__ == "__main__":
    main()
