import json
import os
import click
import yaml
from step_3_stacked_define_CR_and_train_fvt import routine as step_3_stacked_routine
import logging
from training_info import TrainingInfo

logging.basicConfig(level=logging.INFO)


def run_stacked_process_with_id(configs: list[dict]):
    assert len(configs) > 1, "Need at least 2 configs to run stacked process"
    assert all(
        config["step"] == configs[0]["step"] for config in configs
    ), "All configs must have the same step"
    step = configs[0]["step"]
    parent_pid = os.getpid()
    print(f"Running stacked process with PID: {parent_pid}", flush=True)
    log_file = f"run_files/logs/{parent_pid}_stacked.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)

    if step == 3:
        step_3_stacked_routine(configs, file_handler)
    else:
        raise ValueError(f"Step {step} not supported for stacked process")


@click.command()
@click.option("--args-filename", "-a", type=str)
def main(args_filename):
    print(f"Hello 1", flush=True)
    with open(args_filename, "r") as f:
        args = json.load(f)
    print(f"Hello 2", flush=True)
    config_filenames = args["config_filenames"]
    groups = args["groups"]
    print(f"Hello 3", flush=True)
    assert len(config_filenames) == len(
        groups
    ), "Configs and groups must have the same length"
    print(f"Hello 4", flush=True)
    config_dicts = []
    for cfg_filename in config_filenames:
        with open(cfg_filename, "r") as ymlfile:
            config_dicts.append(yaml.safe_load(ymlfile))
    print(f"Hello 5", flush=True)
    config_groups = {}
    for config, g in zip(config_dicts, groups):
        if g not in config_groups:
            config_groups[g] = []
        config_groups[g].append(config)
    print(f"Hello 6", flush=True)
    for i, config_group in enumerate(config_groups.values()):
        print(
            f"Start running {i+1}th stacked process with PID: {os.getpid()}", flush=True
        )
        run_stacked_process_with_id(config_group)
    TrainingInfo.update_metadata()


if __name__ == "__main__":
    main()
