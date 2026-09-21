#!/bin/bash
set -euo pipefail

cd /home/export/soheuny/SRFinder/soheun
output_dir=data/refit_bootstrap/continuous_affine_full_v1

smoke_job=$(
    sbatch --parsable \
        --job-name=affine-signed-smoke \
        --partition=all \
        --ntasks=1 \
        --cpus-per-task=1 \
        --mem=4G \
        --time=00:15:00 \
        --output="$output_dir/signed-smoke-slurm-%j.out" \
        --wrap='cd /home/export/soheuny/SRFinder/soheun && export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && /home/export/soheuny/.conda/envs/coffea_torch/bin/python run_files/run_continuous_affine_full.py --shard-index 15 --n-shards 120 --max-new 1'
)
array_job=$(
    sbatch --parsable \
        --dependency="afterok:$smoke_job" \
        --kill-on-invalid-dep=yes \
        run_files/run_continuous_affine_full_array.sh
)
aggregate_job=$(
    sbatch --parsable \
        --dependency="afterok:$array_job" \
        --kill-on-invalid-dep=yes \
        run_files/run_aggregate_continuous_affine_full.sh
)
figure_job=$(
    sbatch --parsable \
        --dependency="afterok:$aggregate_job" \
        --kill-on-invalid-dep=yes \
        run_files/run_generate_continuous_affine_power_figures.sh
)

printf 'signed_smoke_job=%s\narray_job=%s\naggregate_job=%s\nfigure_job=%s\n' \
    "$smoke_job" "$array_job" "$aggregate_job" "$figure_job" \
    | tee "$output_dir/signed_resume_jobs.txt"
