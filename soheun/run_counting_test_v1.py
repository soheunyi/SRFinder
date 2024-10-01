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
    noise_scale,
):
    config["seed"] = seed
    config["base_fvt"]["data_seed"] = seed
    config["base_fvt"]["train_seed"] = seed
    config["CR_fvt"]["data_seed"] = seed
    config["CR_fvt"]["train_seed"] = seed
    config["signal_ratio"] = signal_ratio
    config["SRCR"]["noise_scale"] = noise_scale

    return config


@click.command()
@click.option("--config", type=str)
@click.option("--seed-start", type=int, default=0)
@click.option("--seed-end", type=int, default=10)
@click.option("--noise-scale", type=float, default=0.1)
def main(config, seed_start, seed_end, noise_scale):
    signal_ratios = [0.05]
    seeds = np.arange(seed_start, seed_end)

    print("signal_ratios: ", signal_ratios)
    print("seeds: ", seeds)
    print("config: ", config)

    # pbar = tqdm(range(10, 0, -1))
    # for i in pbar:
    #     pbar.set_description(f"Experiment starts in {i} seconds")
    #     time.sleep(1)

    with open(config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    for seed in seeds:
        for signal_ratio in signal_ratios:
            config = edit_config(config, seed, signal_ratio, noise_scale)
            routine(config)


if __name__ == "__main__":
    main()
