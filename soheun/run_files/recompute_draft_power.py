import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pickle
import sys
from pathlib import Path


REPO = Path("/home/export/soheuny/SRFinder/soheun")
sys.path.insert(0, str(REPO))

from run_files.recompute_power_pilot import compute_with_old
from training_info import TrainingInfo


OUTPUT_DIR = REPO / "data/refit_bootstrap/draft_eta2_logitcap10_v1"
RESULT_DIR = OUTPUT_DIR / "results"
MANIFEST = OUTPUT_DIR / "manifest.pkl"
IMPORT_DIRS = [
    REPO / "data/refit_bootstrap/null_eta2_logitcap10_v1",
    REPO / "data/refit_bootstrap/power_pilot_eta2_logitcap10_v1/results",
]
OLD_KEY = "affine_correction_and_ks_poisson_bootstrap_n_reps=1000_cdf_mode=mean"


def atomic_pickle(result, output_path):
    temp_path = output_path.parent / f".{output_path.name}.{os.getpid()}.tmp"
    with open(temp_path, "wb") as handle:
        pickle.dump(result, handle)
    os.replace(temp_path, output_path)


def enrich(result, row):
    if "old_p_value" not in result:
        tinfo = TrainingInfo.load(row["hash"])
        old = tinfo.aux_info.get(OLD_KEY)
        if old is None:
            raise KeyError(f"{row['hash']}: missing old result {OLD_KEY}")
        result["old_p_value"] = old["p_value_correction"]
        result["old_observed"] = old["alt_value_correction"]
    result["experiment_name"] = row["experiment_name"]
    result["noise_scale"] = row["noise_scale"]
    result["stats_type"] = "smeared"
    return result


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

    pending = []
    imported = 0
    skipped = 0
    for row in shard:
        hash_ = row["hash"]
        output_path = RESULT_DIR / f"{hash_}.pkl"
        if output_path.exists():
            skipped += 1
            continue
        source_path = next(
            (directory / f"{hash_}.pkl" for directory in IMPORT_DIRS if (directory / f"{hash_}.pkl").exists()),
            None,
        )
        if source_path is not None:
            with open(source_path, "rb") as handle:
                result = pickle.load(handle)
            atomic_pickle(enrich(result, row), output_path)
            imported += 1
            continue
        pending.append(hash_)

    print(
        f"shard={args.shard_index}/{args.n_shards} targets={len(shard)} "
        f"skipped={skipped} imported={imported} pending={len(pending)}",
        flush=True,
    )

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(compute_with_old, hash_): hash_ for hash_ in pending
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            hash_ = futures[future]
            result = enrich(future.result(), row_by_hash[hash_])
            atomic_pickle(result, RESULT_DIR / f"{hash_}.pkl")
            print(
                f"done {completed}/{len(pending)} experiment={result['experiment_name']} "
                f"ratio={result['signal_ratio']} sr={result['sr_size']} "
                f"seed={result['seed']} old_p={result['old_p_value']:.4f} "
                f"new_p={result['p_value']:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
