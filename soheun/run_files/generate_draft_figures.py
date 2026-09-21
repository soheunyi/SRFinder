#!/usr/bin/env python3
"""Master runner for every code-generated figure in the manuscript.

main.tex has 21 live \\includegraphics: 19 come from code, 2 are hand-authored
(contribution_diagram.pdf, workflow.png); 3 further schematics are inline TikZ.

Each task is a standalone script -- no notebook is executed. The per-notebook
scripts under run_files/figure_scripts/ are produced by
run_files/extract_figure_scripts.py, which copies out only the cells that feed a
manuscript figure and narrows their sweeps. The tasks are independent, so
--submit runs them as parallel Slurm jobs rather than one long sequential job.

    python run_files/generate_draft_figures.py --list
    python run_files/generate_draft_figures.py --submit
    python run_files/generate_draft_figures.py --submit --only classifier,smearing
    python run_files/generate_draft_figures.py --run --only toys,power
    python run_files/generate_draft_figures.py --verify
"""

import argparse
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

REPO = Path("/home/export/soheuny/SRFinder/soheun")
FIGURE_DIR = REPO / "figures"
PYTHON = "/home/export/soheuny/.conda/envs/coffea_torch/bin/python"
LOG_DIR = REPO / "data/refit_bootstrap/draft_eta2_logitcap10_v1/figure_logs"


@dataclass
class Task:
    key: str
    script: str
    figures: List[str]
    mem: str = "64G"
    cpus: int = 4
    time_limit: str = "04:00:00"
    gpu: bool = False
    note: str = ""


TASKS = [
    Task("toys", "run_files/figure_scripts/toys.py",
         ["smear_toy_updated.pdf"],
         mem="8G", cpus=2, time_limit="00:20:00",
         note="pure matplotlib toy; no model or event loading"),
    Task("power", "run_files/generate_corrected_power_figures.py",
         ["power_plot_HH4b_noise_scale=2.0.pdf",
          "power_plot_HH4b_400_noise_scale=2.0.pdf",
          "power_plot_ZH4b_noise_scale=2.0.pdf"],
         mem="4G", cpus=1, time_limit="00:15:00",
         note="reads draft_eta2_old_vs_refit_summary.csv only"),
    Task("classifier", "run_files/figure_scripts/classifier.py",
         ["classifier_example_hist_exp=smeared_fvt_training_ensemble_signal=0.01.pdf"]),
    Task("smearing", "run_files/figure_scripts/smearing.py",
         ["smearing_overlap_signal_ratio_0.0_seed_5.pdf",
          "base_and_CR_fvt_scores_hist_seed_5.pdf",
          "smearing_and_tails_signal_ratio_0.01_seed_50.pdf"]),
    Task("null_case", "run_files/figure_scripts/null_case.py",
         ["pull_vs_sr_stats_SR_size_0.2.pdf"],
         mem="32G", time_limit="02:00:00"),
    Task("on_which_to_learn", "run_files/figure_scripts/on_which_to_learn.py",
         ["on_which_to_learn.pdf", "tsne_original_repr.pdf"],
         gpu=True, note="notebook code calls .to('cuda') explicitly; t-SNE fit dominates"),
    Task("signal_concentration", "run_files/figure_scripts/signal_concentration.py",
         ["4b_vs_signal_exp=smeared_fvt_training_ensemble_signal=0.01_max_vs_mean_sr_stats_eta=inf.pdf",
          "4b_vs_signal_exp=smeared_fvt_training_ensemble_signal=0.01_max_vs_mean_sr_stats_eta=2.0.pdf",
          "4b_vs_signal_exp=smeared_fvt_training_ensemble_signal=0.01_different_noise_scales.pdf",
          "4b_vs_signal_exp=smeared_fvt_training_ensemble_signal=0.02_different_noise_scales.pdf",
          "4b_vs_signal_exp=smeared_fvt_training_ensemble_signal=0.01_vs_baseline.pdf",
          "4b_vs_signal_exp=smeared_fvt_training_ensemble_signal=0.02_vs_baseline.pdf",
          "4b_vs_signal_exp=smeared_fvt_training_ensemble_HH4b_400_signal=0.0075_different_noise_scales.pdf",
          "4b_vs_signal_exp=smeared_fvt_training_ensemble_ZH4b_signal=0.03_different_noise_scales.pdf"],
         time_limit="08:00:00",
         note="heaviest task: cells 4, 5 and 8, four 100-seed sweeps"),
]

# Manuscript figures whose generating code is commented out in the notebook.
# Restoring them means re-enabling plotting blocks that were deliberately
# disabled, which changes what the cell draws, so it is left as a decision.
# Every manuscript figure now has a generator.
UNGENERATED = {}

HAND_AUTHORED = ["contribution_diagram.pdf", "workflow.png"]

DRAFT_FIGURES = {
    "4b_vs_signal_exp=smeared_fvt_training_ensemble_HH4b_400_signal=0.0075_different_noise_scales.pdf",
    "4b_vs_signal_exp=smeared_fvt_training_ensemble_ZH4b_signal=0.03_different_noise_scales.pdf",
    "4b_vs_signal_exp=smeared_fvt_training_ensemble_signal=0.01_different_noise_scales.pdf",
    "4b_vs_signal_exp=smeared_fvt_training_ensemble_signal=0.01_max_vs_mean_sr_stats_eta=2.0.pdf",
    "4b_vs_signal_exp=smeared_fvt_training_ensemble_signal=0.01_max_vs_mean_sr_stats_eta=inf.pdf",
    "4b_vs_signal_exp=smeared_fvt_training_ensemble_signal=0.01_vs_baseline.pdf",
    "4b_vs_signal_exp=smeared_fvt_training_ensemble_signal=0.02_different_noise_scales.pdf",
    "4b_vs_signal_exp=smeared_fvt_training_ensemble_signal=0.02_vs_baseline.pdf",
    "base_and_CR_fvt_scores_hist_seed_5.pdf",
    "classifier_example_hist_exp=smeared_fvt_training_ensemble_signal=0.01.pdf",
    "on_which_to_learn.pdf",
    "power_plot_HH4b_400_noise_scale=2.0.pdf",
    "power_plot_HH4b_noise_scale=2.0.pdf",
    "power_plot_ZH4b_noise_scale=2.0.pdf",
    "pull_vs_sr_stats_SR_size_0.2.pdf",
    "smear_toy_updated.pdf",
    "smearing_and_tails_signal_ratio_0.01_seed_50.pdf",
    "smearing_overlap_signal_ratio_0.0_seed_5.pdf",
    "tsne_original_repr.pdf",
}


def assert_coverage():
    planned = {name for task in TASKS for name in task.figures}
    both = planned & set(UNGENERATED)
    if both:
        raise SystemExit(f"figure both planned and marked ungenerated: {sorted(both)}")
    uncovered = DRAFT_FIGURES - planned - set(UNGENERATED)
    if uncovered:
        raise SystemExit(f"manuscript figures no task produces: {sorted(uncovered)}")
    unknown = planned - DRAFT_FIGURES
    if unknown:
        raise SystemExit(f"task produces figures the manuscript does not use: {sorted(unknown)}")


def stamp(name):
    path = FIGURE_DIR / name
    if not path.exists():
        return "MISSING         "
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime))


def submit(task):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    directives = [
        f"#SBATCH --job-name=fig-{task.key}",
        "#SBATCH --partition=all",
        f"#SBATCH --cpus-per-task={task.cpus}",
        f"#SBATCH --mem={task.mem}",
        f"#SBATCH --time={task.time_limit}",
        f"#SBATCH --output={LOG_DIR.relative_to(REPO)}/{task.key}-%j.out",
    ]
    if task.gpu:
        directives.append("#SBATCH --gres=gpu:1")
    script = "\n".join(
        ["#!/bin/bash"] + directives + [
            "", "set -euo pipefail", f"cd {REPO}", f"{PYTHON} {task.script}", ""
        ]
    )
    result = subprocess.run(
        ["sbatch", "--parsable"], input=script, text=True,
        capture_output=True, cwd=REPO, check=True,
    )
    return result.stdout.strip().split(";")[0]


def run_local(task):
    started = time.time()
    subprocess.run([PYTHON, str(REPO / task.script)], cwd=REPO, check=True)
    missing = [f for f in task.figures
               if not (FIGURE_DIR / f).exists() or (FIGURE_DIR / f).stat().st_size == 0]
    stale = [f for f in task.figures
             if (FIGURE_DIR / f).exists() and (FIGURE_DIR / f).stat().st_mtime < started]
    if missing or stale:
        raise RuntimeError(f"{task.key}: missing={missing} not_regenerated={stale}")
    print(f"--- {task.key} ok in {time.time() - started:.0f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--submit", action="store_true", help="one Slurm job per task, in parallel")
    mode.add_argument("--run", action="store_true", help="run sequentially here")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--only")
    parser.add_argument("--skip")
    args = parser.parse_args()
    assert_coverage()

    tasks = TASKS
    if args.only:
        wanted = {k.strip() for k in args.only.split(",")}
        unknown = wanted - {t.key for t in TASKS}
        if unknown:
            parser.error(f"unknown task(s): {sorted(unknown)}")
        tasks = [t for t in TASKS if t.key in wanted]
    if args.skip:
        dropped = {k.strip() for k in args.skip.split(",")}
        tasks = [t for t in tasks if t.key not in dropped]

    if args.verify:
        bad = [f for t in TASKS for f in t.figures if not (FIGURE_DIR / f).exists()]
        print(f"covered by a task: {len({f for t in TASKS for f in t.figures})}")
        print(f"no live generator: {len(UNGENERATED)}")
        print(f"missing on disk:   {bad}")
        raise SystemExit(1 if bad else 0)

    if args.submit:
        for task in tasks:
            job = submit(task)
            print(f"{task.key:22s} job {job}  ({len(task.figures)} figure(s), {task.mem}, {task.time_limit})")
        print(f"\n{len(tasks)} independent jobs submitted; watch with: squeue -u $USER")
        return

    if args.run:
        for task in tasks:
            print(f"=== {task.key} ===", flush=True)
            run_local(task)
        return

    for task in tasks:
        print(f"{task.key:22s} {task.mem:>5s} {task.time_limit}  {task.script}")
        if task.note:
            print(f"{'':22s} note: {task.note}")
        for name in task.figures:
            print(f"{'':22s}   {stamp(name)}  {name}")
    print(f"\nno live generator ({len(UNGENERATED)}):")
    for name, where in UNGENERATED.items():
        print(f"  {name}\n      {where}")
    print("hand-authored: " + ", ".join(HAND_AUTHORED))


if __name__ == "__main__":
    main()
