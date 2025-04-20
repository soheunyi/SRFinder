import shutil
import tqdm
import concurrent.futures
import os
import pickle


# Function to copy a single file
def copy_file(source_path, dest_path):
    try:
        # Use copy2 to preserve metadata if needed, otherwise copy is fine
        shutil.copy2(source_path, dest_path)
    except FileNotFoundError:
        # Optionally log or print a warning if a source file doesn't exist
        # Using print here for simplicity in notebook context
        print(f"Warning: Source file not found: {source_path}")
    except Exception as e:
        print(f"Error copying {source_path} to {dest_path}: {e}")


# Determine a reasonable number of workers based on CPU cores
# Adjust this number based on your system's I/O capabilities
MAX_WORKERS = os.cpu_count() or 4  # Default to 4 if os.cpu_count() is None

experiment_names = [
    "CR_fvt_training",
    "CR_fvt_training_ensemble_max",
    "CR_fvt_training_ensemble_max_HH4b_400",
    "CR_fvt_training_ensemble_max_HH4b_800",
    "CR_fvt_training_ensemble_max_fvt",
    "CR_fvt_training_ensemble_max_fvt_HH4b_400",
    "CR_fvt_training_ensemble_max_smeared",
    "CR_fvt_training_ensemble_max_smeared_HH4b_400",
    "CR_fvt_training_ensemble_mean",
    "CR_fvt_training_repr_norm",
    "CR_fvt_training_same_data",
    "CR_fvt_training_schedulefree",
    "CR_fvt_training_schedulefree_ablation",
    "CR_fvt_training_schedulefree_ablation2",
    "CR_fvt_training_v2",
    "base_fvt_training_ensemble",
    "base_fvt_training_ensemble_HH4b_400",
    "base_fvt_training_ensemble_HH4b_800",
    "base_fvt_training_ensemble_HH4b_resonant",
    "base_fvt_training_fixed_data_split",
    "base_fvt_with_repr_norm",
    "base_fvt_with_repr_norm_smaller_repr_dim",
    "better_base_fvt_training",
    "better_base_fvt_training_small",
    "mi_test",
    "smeared_fvt_training",
    "smeared_fvt_training_ensemble",
    "smeared_fvt_training_ensemble_HH4b_400",
    "smeared_fvt_training_ensemble_HH4b_800",
    "smeared_fvt_training_ensemble_HH4b_resonant",
    "smeared_fvt_training_noise_scale",
    "smeared_fvt_training_repr_norm",
    "smeared_fvt_training_small",
]
metadata = pickle.load(open("data/metadata/TrainingInfo.pkl", "rb"))
metadata: dict[str, dict[str, any]]
# copy each file in the directory to the new directory
for experiment_name in experiment_names:

    hashes = [
        hash_
        for hash_, hparams in metadata.items()
        if hparams["experiment_name"] == experiment_name
    ]

    # Construct the destination directory path
    dest_dir = f"data/training_info/{experiment_name}"
    # Ensure the destination directory exists (might be redundant if created in previous cell)
    os.makedirs(dest_dir, exist_ok=True)

    # Use ThreadPoolExecutor for parallel I/O operations
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit copy tasks to the executor
        futures = []
        for hash_ in hashes:
            source_path = f"data/TrainingInfo/{hash_}"
            # Use os.path.join for robust path creation
            dest_path = os.path.join(dest_dir, hash_)
            futures.append(executor.submit(copy_file, source_path, dest_path))

        # Use tqdm to show progress on the completion of futures
        # The description provides context about the current experiment
        print(
            f"Submitting {len(futures)} copy tasks for {experiment_name}..."
        )  # Added print statement
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"Copying for {experiment_name}",
        ):
            # Check for exceptions during the copy process
            try:
                future.result()  # Raises exceptions if any occurred in the worker thread
            except Exception as e:
                # Error logging already happens in copy_file, could add aggregation here if needed
                # print(f"An error occurred during copy: {e}") # Example of additional logging
                pass
        print(f"Finished copying for {experiment_name}.")  # Added print statement
