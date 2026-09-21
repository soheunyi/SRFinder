#!/usr/bin/env python3
"""Master entry point for corrected-bootstrap tests and draft figures.

Run from the SRFinder repository root on Wright. Expensive work is always
submitted to Slurm; this command only prepares manifests, submits jobs, reports
status, and verifies outputs.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent
PYTHON = Path("/home/export/soheuny/.conda/envs/coffea_torch/bin/python")

SCOPES = {
    "pilot": {
        "total": 560,
        "output_dir": REPO / "data/refit_bootstrap/power_pilot_eta2_logitcap10_v1",
        "prepare": REPO / "run_files/prepare_power_pilot_manifest.py",
        "launcher": REPO / "run_files/run_recompute_power_pilot.sh",
        "aggregate_launcher": REPO / "run_files/run_aggregate_power_pilot.sh",
    },
    "draft": {
        "total": 6000,
        "output_dir": REPO / "data/refit_bootstrap/draft_eta2_logitcap10_v1",
        "prepare": REPO / "run_files/prepare_draft_power_manifest.py",
        "launcher": REPO / "run_files/run_recompute_draft_power.sh",
        "aggregate_launcher": REPO / "run_files/run_aggregate_draft_power.sh",
    },
}

MANAGED_FIGURES = [
    "classifier_example_hist_exp=smeared_fvt_training_ensemble_signal=0.01.pdf",
    "4b_vs_signal_exp=smeared_fvt_training_ensemble_signal=0.01_max_vs_mean_sr_stats_eta=inf.pdf",
    "4b_vs_signal_exp=smeared_fvt_training_ensemble_signal=0.01_max_vs_mean_sr_stats_eta=2.0.pdf",
    "smearing_overlap_signal_ratio_0.0_seed_5.pdf",
    "base_and_CR_fvt_scores_hist_seed_5.pdf",
    "pull_vs_sr_stats_SR_size_0.2.pdf",
    "on_which_to_learn.pdf",
    "tsne_original_repr.pdf",
    "smear_toy_updated.pdf",
    "smearing_and_tails_signal_ratio_0.01_seed_50.pdf",
    "4b_vs_signal_exp=smeared_fvt_training_ensemble_signal=0.01_different_noise_scales.pdf",
    "4b_vs_signal_exp=smeared_fvt_training_ensemble_signal=0.02_different_noise_scales.pdf",
    "4b_vs_signal_exp=smeared_fvt_training_ensemble_signal=0.01_vs_baseline.pdf",
    "4b_vs_signal_exp=smeared_fvt_training_ensemble_signal=0.02_vs_baseline.pdf",
    "power_plot_HH4b_noise_scale=2.0.pdf",
    "4b_vs_signal_exp=smeared_fvt_training_ensemble_HH4b_400_signal=0.0075_different_noise_scales.pdf",
    "4b_vs_signal_exp=smeared_fvt_training_ensemble_ZH4b_signal=0.03_different_noise_scales.pdf",
    "power_plot_HH4b_400_noise_scale=2.0.pdf",
    "power_plot_ZH4b_noise_scale=2.0.pdf",
]

EXTERNAL_MANUSCRIPT_ASSETS = [
    "contribution_diagram.pdf",
    "workflow.png",
    "ABCD schematic embedded in LaTeX",
    "classifier architecture embedded in LaTeX",
]


def run(command, capture=False):
    print("+", " ".join(str(part) for part in command), flush=True)
    return subprocess.run(
        [str(part) for part in command],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=capture,
    )


def scope_config(name):
    return SCOPES[name]


def ensure_manifest(name):
    config = scope_config(name)
    manifest = config["output_dir"] / "manifest.pkl"
    if manifest.exists():
        print(f"using frozen manifest: {manifest}")
        return manifest
    run(
        [
            "srun",
            "-p",
            "all",
            "-n",
            "1",
            "--cpus-per-task=1",
            "--mem=2G",
            "--time=00:05:00",
            PYTHON,
            config["prepare"],
        ]
    )
    return manifest


def save_state(name, **updates):
    output_dir = scope_config(name)["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "master_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    state.update(updates)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def submit(name, with_followups=False):
    ensure_manifest(name)
    config = scope_config(name)
    result = run(["sbatch", "--parsable", config["launcher"]], capture=True)
    test_job = result.stdout.strip().split(";")[0]
    print(f"submitted {name} tests: {test_job}")
    updates = {"test_job": test_job}
    if with_followups:
        aggregate = run(
            [
                "sbatch",
                "--parsable",
                f"--dependency=afterok:{test_job}",
                config["aggregate_launcher"],
            ],
            capture=True,
        ).stdout.strip().split(";")[0]
        updates["aggregate_job"] = aggregate
        print(f"submitted dependent aggregation: {aggregate}")
        if name == "draft":
            figures = run(
                [
                    "sbatch",
                    "--parsable",
                    f"--dependency=afterok:{aggregate}",
                    REPO / "run_files/run_generate_all_draft_figures.sh",
                ],
                capture=True,
            ).stdout.strip().split(";")[0]
            updates["figures_job"] = figures
            print(f"submitted dependent all-figures job: {figures}")
    save_state(name, **updates)


def status(name):
    config = scope_config(name)
    result_dir = config["output_dir"] / "results"
    completed = len(list(result_dir.glob("*.pkl"))) if result_dir.exists() else 0
    total = config["total"]
    print(f"{name}: {completed}/{total} complete; {total - completed} pending")
    state_path = config["output_dir"] / "master_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        job_ids = [str(value) for key, value in state.items() if key.endswith("_job")]
        if job_ids:
            run(["squeue", "-j", ",".join(job_ids), "-o", "%.18i %.28j %.10T %.10M %R"])


def aggregate(name):
    run(["sbatch", scope_config(name)["aggregate_launcher"]])


def submit_figures(dependency=None):
    command = ["sbatch", "--parsable"]
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    command.append(REPO / "run_files/run_generate_all_draft_figures.sh")
    result = run(command, capture=True)
    print(f"submitted all-figures job: {result.stdout.strip()}")


def verify_figures():
    figure_dir = REPO / "figures"
    missing = []
    empty = []
    for filename in MANAGED_FIGURES:
        path = figure_dir / filename
        if not path.exists():
            missing.append(filename)
        elif path.stat().st_size == 0:
            empty.append(filename)
    print(f"managed draft figures: {len(MANAGED_FIGURES)}")
    print(f"missing: {missing}")
    print(f"empty: {empty}")
    print("external/hand-authored manuscript assets:")
    for asset in EXTERNAL_MANUSCRIPT_ASSETS:
        print(f"  - {asset}")
    if missing or empty:
        raise SystemExit(1)


def plan():
    print("Corrected-bootstrap draft scope")
    print("  alternatives: (4 + 4 + 6) signal settings x 4 SR sizes x 100 seeds = 5,600")
    print("  null:          1 setting x 4 SR sizes x 100 seeds = 400")
    print("  total:         6,000 SR-specific tests at eta=2")
    print("  bootstrap:     1,000 Poisson replicates per test")
    print("  statistic:     clip psi to [-10, 10] after fixing SR membership")
    print("  compatible logitcap10 pilot checkpoints are imported by hash")
    print(f"  managed generated figures: {len(MANAGED_FIGURES)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    for command in ["prepare", "submit", "resume", "status", "aggregate"]:
        child = subparsers.add_parser(command)
        child.add_argument("--scope", choices=SCOPES, required=True)
        if command in {"submit", "resume"}:
            child.add_argument("--with-followups", action="store_true")
    figures = subparsers.add_parser("submit-figures")
    figures.add_argument("--dependency")
    subparsers.add_parser("verify-figures")
    args = parser.parse_args()

    if args.command == "plan":
        plan()
    elif args.command == "prepare":
        ensure_manifest(args.scope)
    elif args.command in {"submit", "resume"}:
        submit(args.scope, with_followups=args.with_followups)
    elif args.command == "status":
        status(args.scope)
    elif args.command == "aggregate":
        aggregate(args.scope)
    elif args.command == "submit-figures":
        submit_figures(args.dependency)
    elif args.command == "verify-figures":
        verify_figures()


if __name__ == "__main__":
    main()
