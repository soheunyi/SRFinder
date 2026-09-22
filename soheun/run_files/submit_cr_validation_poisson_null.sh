#!/bin/bash
set -euo pipefail

code_dir=/home/export/soheuny/SRFinder/issue1-centered-poisson/soheun
output_dir=/home/export/soheuny/SRFinder/soheun/data/refit_bootstrap/cr_validation_poisson_null_v1
cd "$code_dir"

array_job=$(sbatch --parsable run_files/run_cr_validation_poisson_null_array.sh)
aggregate_job=$(
    sbatch --parsable \
        --dependency="afterok:$array_job" \
        --kill-on-invalid-dep=yes \
        run_files/run_aggregate_cr_validation_poisson_null.sh
)
printf 'array_job=%s\naggregate_job=%s\n' "$array_job" "$aggregate_job" \
    | tee "$output_dir/pipeline_jobs.txt"
