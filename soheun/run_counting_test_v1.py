# ad hoc script to run counting test
import time
import click
import numpy as np
from tqdm import tqdm
from counting_test_v1 import routine
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
@click.option("--seed-start", type=int, default=0)
@click.option("--seed-end", type=int, default=10)
def main(config, seed_start, seed_end):
    signal_ratios = [0.0, 0.01, 0.02]
    seeds = np.arange(seed_start, seed_end)

    print("signal_ratios: ", signal_ratios)
    print("seeds: ", seeds)
    print("config: ", config)

    pbar = tqdm(range(10, 0, -1))
    for i in pbar:
        pbar.set_description(f"Experiment starts in {i} seconds")
        time.sleep(1)

    with open(config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    for seed in seeds:
        for signal_ratio in signal_ratios:
            config = edit_config(config, seed, signal_ratio)
            routine(config)


if __name__ == "__main__":
    main()
