import yaml
import os
import stat
import time
from itertools import product
import numpy as np
import sys

sys.path.append("..")
from training_info import TrainingInfo

EXPERIMENT_NAME = "better_base_fvt_training"
BASE_CONFIG_FILENAME = "better_fvt_training.yml"
N_RUNFILES = 2


seeds = range(50)
signal_ratios = [0.005]

hparams_filter = {
    "experiment_name": EXPERIMENT_NAME,
    "dataset": lambda x: x["signal_ratio"] == signal_ratios[0],
}
_, hparams = TrainingInfo.find(hparams_filter, return_hparams=True)
existing_seeds_and_srs = {
    (hparam["dataset"]["seed"], hparam["dataset"]["signal_ratio"]) for hparam in hparams
}
target_seeds_and_srs = set(product(seeds, signal_ratios)) - existing_seeds_and_srs
print(len(target_seeds_and_srs))


# noise_scale = 4.0

CONFIGS_TO_RUN = []
for seed, signal_ratio in target_seeds_and_srs:
    # postfix = f"{seed}_{noise_scale}_{signal_ratio}"
    postfix = f"{seed}_{signal_ratio}"

    NEW_CONFIG_FILENAME = f"{EXPERIMENT_NAME}_{postfix}.yml"

    # Update config file
    config = yaml.safe_load(open(f"../configs/{BASE_CONFIG_FILENAME}", "r"))
    config["experiment_name"] = EXPERIMENT_NAME
    config["dataset"]["signal_ratio"] = signal_ratio
    config["dataset"]["seed"] = seed
    config["base_fvt"]["train_seed"] = seed
    config["base_fvt"]["model_seed"] = seed
    config["base_fvt"]["data_seed"] = seed
    # config["smearing"]["seed"] = seed
    # config["smearing"]["noise_scale"] = noise_scale

    with open(f"../configs/tmp/{NEW_CONFIG_FILENAME}", "w") as f:
        yaml.dump(config, f)

    CONFIGS_TO_RUN.append(f"-c configs/tmp/{NEW_CONFIG_FILENAME}")


# divide CONFIGS_TO_RUN into N_RUNFILES chunks
CONFIG_CHUNKS = {i: [] for i in range(N_RUNFILES)}
for i, config in enumerate(CONFIGS_TO_RUN):
    CONFIG_CHUNKS[i % N_RUNFILES].append(config)

RUN_PARTITIONS = []
PROB_STATDS = 0.0
for _ in range(N_RUNFILES):
    RUN_PARTITIONS.append("statds" if np.random.rand() < PROB_STATDS else "phil_condo")

print(
    f"n_tasks (runfiles) {len(CONFIG_CHUNKS)}, max n_tasks per runfile {max([len(chunk) for chunk in CONFIG_CHUNKS.values()])}"
)
print(f"Partitions: {RUN_PARTITIONS}")
# Get input
input("Press Enter to continue...")

for k, configs in CONFIG_CHUNKS.items():
    if len(configs) == 0:
        continue

    RUN_PARTITION = RUN_PARTITIONS[k]
    RUN_FILENAME = f"run_{EXPERIMENT_NAME}_{k}.sh"

    # Update run script
    with open("run_with_multiple_configs.sh", "r") as f:
        run_script = f.read()

    run_script = run_script.replace(
        'CONFIGS_LIST=""', f'CONFIGS_LIST="{" ".join(configs)}"'
    )

    run_script = run_script.replace(
        "#SBATCH --partition statds",
        f"#SBATCH --partition {RUN_PARTITION}",
    )

    with open(RUN_FILENAME, "w") as f:
        f.write(run_script)

    # Make run script executable and run it
    st = os.stat(RUN_FILENAME)
    os.chmod(RUN_FILENAME, st.st_mode | stat.S_IEXEC)
    os.system(f"sbatch {RUN_FILENAME}")
    time.sleep(0.5)
    os.remove(RUN_FILENAME)
