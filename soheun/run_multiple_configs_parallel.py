from concurrent.futures import ProcessPoolExecutor
import os
import click
import yaml
from step_1_base_fvt_training import routine as step_1_routine
from step_2_smeared_fvt_training import routine as step_2_routine
from step_3_define_CR_and_train_fvt import routine as step_3_routine
from step_4_mi_test import routine as step_4_routine
from step_3_stacked_define_CR_and_train_fvt import routine as step_3_stacked_routine
import torch
import logging
from datetime import datetime
from training_info import TrainingInfo

logging.basicConfig(level=logging.INFO)


def run_single_config(config: dict, file_handler: logging.FileHandler):

    assert "step" in config, f"step is not in the config: {config}"
    step = config["step"]

    if step == 1:
        step_1_routine(config, file_handler)
    elif step == 2:
        step_2_routine(config, file_handler)
    elif step == 3:
        step_3_routine(config, file_handler)
    elif step == 4:
        step_4_routine(config, file_handler)
    else:
        raise ValueError(f"Step {step} not supported")


def run_stacked_process_with_id(configs: list[dict], file_handler: logging.FileHandler):
    assert len(configs) > 1, "Need at least 2 configs to run stacked process"
    assert all(
        config["step"] == configs[0]["step"] for config in configs
    ), "All configs must have the same step"
    step = configs[0]["step"]
    if step == 3:
        step_3_stacked_routine(configs, file_handler)
    else:
        raise ValueError(f"Step {step} not supported for stacked process")


def run_process_with_id(args):
    parent_pid, pid, config, nprocs = args

    # Set up file logging for this process
    log_file = f"run_files/logs/{parent_pid}_process_{pid}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)

    start_time = datetime.now()
    # Log process start with GPU info
    gpu_info = (
        f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"
    )
    logging.info(
        f"""
======================================
Process {pid} starting on {gpu_info}
Config: {yaml.dump(config, default_flow_style=False)}
Current time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
======================================
"""
    )

    try:
        if torch.cuda.is_available():
            # Set memory limit for this process
            torch.cuda.set_device(0)
            torch.cuda.set_per_process_memory_fraction(
                0.95 / nprocs
            )  # Leave some headroom
            torch.cuda.empty_cache()
        run_single_config(config, file_handler)
        end_time = datetime.now()
        secs_taken = (end_time - start_time).total_seconds()

        # Fix the time calculation
        hours = int(secs_taken // 3600)
        minutes = int((secs_taken % 3600) // 60)
        seconds = secs_taken % 60

        logging.info(
            f"""
======================================
Process {pid} finished
Current time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
Time taken: {hours:02d}:{minutes:02d}:{seconds:05.2f}
======================================
"""
        )
    except Exception as e:
        logging.error(f"Process {pid} failed with error: {str(e)}", exc_info=True)
        raise
    finally:
        torch.cuda.empty_cache()
        # Remove the handler when done
        logging.getLogger().removeHandler(file_handler)


@click.command()
@click.option("--config", "-c", type=str, multiple=True)
@click.option("--nprocs", "-p", type=int, default=4)
def main(config, nprocs):
    cfg_list = config
    config_dicts = []
    # get parent pid
    parent_pid = os.getpid()

    for cfg in cfg_list:
        with open(cfg, "r") as ymlfile:
            config_dicts.append(yaml.safe_load(ymlfile))

    with ProcessPoolExecutor(max_workers=nprocs) as executor:
        # Enumerate configs to pair each with a process ID
        process_args = list(enumerate(config_dicts))
        process_args = [(parent_pid, pid, cfg, nprocs) for pid, cfg in process_args]
        # Use executor.map with the new function and args
        executor.map(run_process_with_id, process_args)

    # update metadata
    TrainingInfo.update_metadata()


if __name__ == "__main__":
    main()
