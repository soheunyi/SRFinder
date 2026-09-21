#!/bin/bash
set -euo pipefail

code_dir=/home/export/soheuny/SRFinder/issue1-centered-poisson/soheun
output_dir=/home/export/soheuny/SRFinder/soheun/data/refit_bootstrap/centered_poisson_null_v1
cd "$code_dir"

smoke_job=$(
    sbatch --parsable \
        --job-name=poisson-smoke \
        --partition=all \
        --ntasks=1 \
        --cpus-per-task=1 \
        --mem=4G \
        --time=00:10:00 \
        --output="$output_dir/smoke-slurm-%j.out" \
        --wrap="cd $code_dir && export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && /home/export/soheuny/.conda/envs/coffea_torch/bin/python run_files/run_centered_poisson_null.py --shard-index 0 --n-shards 1 --limit 1"
)
array_job=$(
    sbatch --parsable \
        --dependency="afterok:$smoke_job" \
        --kill-on-invalid-dep=yes \
        run_files/run_centered_poisson_null_array.sh
)
aggregate_job=$(
    sbatch --parsable \
        --dependency="afterok:$array_job" \
        --kill-on-invalid-dep=yes \
        run_files/run_aggregate_centered_poisson_null.sh
)

printf 'smoke_job=%s\narray_job=%s\naggregate_job=%s\n' \
    "$smoke_job" "$array_job" "$aggregate_job" \
    | tee "$output_dir/pipeline_jobs.txt"
