import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pickle
import sys
from pathlib import Path


REPO = Path("/home/export/soheuny/SRFinder/soheun")
sys.path.insert(0, str(REPO))

from run_files.recompute_null_rejection_rates import compute_one
from training_info import TrainingInfo


OUTPUT_DIR = REPO / "data/refit_bootstrap/power_pilot_eta2_logitcap10_v1"
RESULT_DIR = OUTPUT_DIR / "results"
MANIFEST = OUTPUT_DIR / "manifest.pkl"
OLD_KEY = "affine_correction_and_ks_poisson_bootstrap_n_reps=1000_cdf_mode=mean"


def compute_with_old(hash_):
    result = compute_one(hash_)
    tinfo = TrainingInfo.load(hash_)
    old = tinfo.aux_info.get(OLD_KEY)
    if old is None:
        raise KeyError(f"{hash_}: missing old bootstrap result {OLD_KEY}")
    result["old_p_value"] = old["p_value_correction"]
    result["old_observed"] = old["alt_value_correction"]
    return result


def atomic_pickle(result, output_path):
    temp_path = output_path.parent / f".{output_path.name}.{os.getpid()}.tmp"
    with open(temp_path, "wb") as handle:
        pickle.dump(result, handle)
    os.replace(temp_path, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--n-shards", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    with open(MANIFEST, "rb") as handle:
        rows = pickle.load(handle)
    shard = rows[args.shard_index :: args.n_shards]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    row_by_hash = {row["hash"]: row for row in shard}
    pending = [
        row["hash"]
        for row in shard
        if not (RESULT_DIR / f"{row['hash']}.pkl").exists()
    ]
    print(
        f"shard={args.shard_index}/{args.n_shards} targets={len(shard)} "
        f"pending={len(pending)}",
        flush=True,
    )

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(compute_with_old, hash_): hash_ for hash_ in pending
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            hash_ = futures[future]
            result = future.result()
            row = row_by_hash[hash_]
            result["experiment_name"] = row["experiment_name"]
            result["noise_scale"] = row["noise_scale"]
            result["stats_type"] = "smeared"
            atomic_pickle(result, RESULT_DIR / f"{hash_}.pkl")
            print(
                f"done {completed}/{len(pending)} experiment={row['experiment_name']} "
                f"ratio={row['signal_ratio']} sr={row['sr_size']} seed={row['seed']} "
                f"old_p={result['old_p_value']:.4f} new_p={result['p_value']:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
