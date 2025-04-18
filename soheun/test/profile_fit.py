import torch
import numpy as np
from torch.utils.data import TensorDataset
from fvt_classifier import FvTClassifier
import time
from tqdm import tqdm
from training_info import TrainingInfo
from dataset import MotherSamples
from constants import FEATURES
import matplotlib.pyplot as plt
import seaborn as sns


def create_dummy_data(num_samples=1000, feature_dim=100):
    """Create dummy data for profiling"""
    X = torch.randn(num_samples, feature_dim)
    y = torch.randint(0, 2, (num_samples,))
    w = torch.ones(num_samples)
    return TensorDataset(X, y, w)


def load_data():
    """Load the actual dataset using the same logic as test_training_speed.py"""
    # Create config similar to better_fvt_training.yml
    config = {
        "experiment_name": "speed_test",
        "dataset": {
            "signal_filename": "HH4b_picoAOD.h5",
            "signal_ratio": 0.0,
            "n_3b": 100_0000,
            "ratio_4b": 0.5,
            "seed": 0,
            "base_fvt_train_ratio": 0.5,
        },
        "base_fvt": {
            "model": "FvTClassifier",
            "dim_dijet_features": 6,
            "dim_quadjet_features": 6,
            "depth": {"encoder": 4, "decoder": 1},
            "fit_batch_size": 1024,
            "model_seed": 0,
            "train_seed": 0,
            "data_seed": 0,
            "max_epochs": 10,
            "val_ratio": 0.33,
            "early_stop_patience": None,
            "optimizer": {"type": "Adam", "lr": 0.01},
            "lr_scheduler": {
                "type": "ReduceLROnPlateau",
                "factor": 0.5,
                "threshold": 0.0001,
                "patience": 10,
                "cooldown": 1,
                "min_lr": 0.0002,
            },
            "dataloader": {
                "batch_size": 1024,
                "batch_size_multiplier": 2,
                "batch_size_milestones": [1, 3, 6, 10, 15],
            },
        },
    }

    # Create mother samples
    ms_hparams = {
        "n_3b": config["dataset"]["n_3b"],
        "ratio_4b": config["dataset"]["ratio_4b"],
        "signal_ratio": config["dataset"]["signal_ratio"],
        "signal_filename": config["dataset"]["signal_filename"],
        "seed": config["dataset"]["seed"],
    }

    # Find or create mother samples
    hashes = MotherSamples.find(ms_hparams, from_metadata=False)
    if len(hashes) == 0:
        raise ValueError("No mother samples found for the given parameters")
    elif len(hashes) > 1:
        raise ValueError("Number of mother samples must be one")
    ms_hash = hashes[0]
    mother_samples = MotherSamples.load(ms_hash)

    # Split the mother dataset into train and test
    ms_len = len(mother_samples.scdinfo)
    ms_idx = np.zeros(ms_len, dtype=bool)
    ms_idx[: int(ms_len * config["dataset"]["base_fvt_train_ratio"])] = True
    np.random.seed(config["dataset"]["seed"])
    np.random.shuffle(ms_idx)

    # Create training info
    base_fvt_hparams = config["base_fvt"]
    base_fvt_hparams["experiment_name"] = config["experiment_name"]
    base_fvt_hparams["dataset"] = config["dataset"]
    base_fvt_hparams["step"] = 1

    base_fvt_tinfo = TrainingInfo(base_fvt_hparams, ms_hash=ms_hash, ms_idx=ms_idx)
    print("Base FvT Training Hash: ", base_fvt_tinfo.hash)

    # Get train and validation datasets
    train_dset, val_dset = base_fvt_tinfo.fetch_train_val_tensor_datasets(
        FEATURES, "fourTag", "weight"
    )

    return train_dset, val_dset


def measure_performance():
    """Measure performance metrics with minimal overhead"""
    print("Loading data...")
    train_data, val_data = load_data()

    # Test only the most important configurations
    configs = [
        {"batch_size": 1024, "num_workers": 0, "preload_to_gpu": False},  # Baseline
        {
            "batch_size": 1024,
            "num_workers": 0,
            "preload_to_gpu": True,
        },  # Test multi-worker
    ]

    results = []

    print("Testing configurations...")
    for config in tqdm(configs):
        # Initialize model
        model = FvTClassifier(
            num_classes=2,
            dim_input_jet_features=4,
            dim_dijet_features=6,
            dim_quadjet_features=6,
            run_name=f"perf_test_{config['batch_size']}_{config['num_workers']}",
        )

        # Measure one epoch
        start_time = time.time()
        model.fit(
            train_dataset=train_data,
            val_dataset=val_data,
            max_epochs=10,
            train_seed=42,
            save_checkpoint=False,
            optimizer_config={"type": "Adam", "lr": 0.001},
            lr_scheduler_config={
                "type": "ReduceLROnPlateau",
                "factor": 0.5,
                "threshold": 0.0001,
                "patience": 10,
                "cooldown": 1,
                "min_lr": 0.0002,
            },
            dataloader_config={
                "batch_size": config["batch_size"],
                "num_workers": config["num_workers"],
                "pin_memory": True,
                "persistent_workers": True,
            },
            callbacks=[],
            tb_log_dir="tmp",
            early_stop_patience=None,
            preload_to_gpu=config["preload_to_gpu"],
            progress_bar_epochs=1,
        )
        end_time = time.time()

        # Calculate metrics
        throughput = len(train_data) / (end_time - start_time)  # samples per second
        results.append(
            {
                "batch_size": config["batch_size"],
                "num_workers": config["num_workers"],
                "throughput": throughput,
                "time_per_epoch": (end_time - start_time) / 10,
            }
        )

        # Print immediate results
        print(
            f"\nConfiguration: batch_size={config['batch_size']}, num_workers={config['num_workers']}"
        )
        print(f"Throughput: {throughput:.2f} samples/second")
        print(f"Time per epoch: {(end_time - start_time) / 10:.2f} seconds")

    # Plot results
    plt.figure(figsize=(12, 6))

    # Throughput plot
    plt.subplot(1, 2, 1)
    for num_workers in [0, 4]:
        mask = [r["num_workers"] == num_workers for r in results]
        batch_sizes = [
            r["batch_size"] for r in results if r["num_workers"] == num_workers
        ]
        throughputs = [
            r["throughput"] for r in results if r["num_workers"] == num_workers
        ]
        plt.plot(batch_sizes, throughputs, label=f"num_workers={num_workers}")
    plt.xlabel("Batch Size")
    plt.ylabel("Throughput (samples/second)")
    plt.title("Training Throughput")
    plt.legend()

    # Time per epoch plot
    plt.subplot(1, 2, 2)
    for num_workers in [0, 4]:
        mask = [r["num_workers"] == num_workers for r in results]
        batch_sizes = [
            r["batch_size"] for r in results if r["num_workers"] == num_workers
        ]
        times = [
            r["time_per_epoch"] for r in results if r["num_workers"] == num_workers
        ]
        plt.plot(batch_sizes, times, label=f"num_workers={num_workers}")
    plt.xlabel("Batch Size")
    plt.ylabel("Time per Epoch (seconds)")
    plt.title("Training Time")
    plt.legend()

    plt.tight_layout()
    plt.savefig("performance_analysis.png")
    plt.close()

    # Save results to file
    with open("performance_results.txt", "w") as f:
        f.write("Configuration\tThroughput (samples/sec)\tTime per Epoch (sec)\n")
        for r in results:
            f.write(
                f"batch_size={r['batch_size']}, num_workers={r['num_workers']}\t{r['throughput']:.2f}\t{r['time_per_epoch']:.2f}\n"
            )


def main():
    print("Starting performance measurement...")
    print(f"Using CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    measure_performance()

    print("\nPerformance measurement complete! Check the following files for results:")
    print("- performance_results.txt: Detailed performance metrics")
    print("- performance_analysis.png: Performance plots")


if __name__ == "__main__":
    main()
