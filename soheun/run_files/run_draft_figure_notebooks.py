import time
from pathlib import Path

import nbformat
from nbclient.exceptions import CellExecutionError
from nbconvert.preprocessors import ExecutePreprocessor


REPO = Path("/home/export/soheuny/SRFinder/soheun")
OUTPUT_DIR = REPO / "data/refit_bootstrap/draft_eta2_logitcap10_v1/executed_notebooks"
FIGURE_DIR = REPO / "figures"

# notebooks/figures.ipynb is not listed: it fails with KeyError '3b_sq' and its
# only output, systematic_error_correction.pdf, is not used in the manuscript.
NOTEBOOKS = [
    (
        REPO / "notebooks/draft/toys.ipynb",
        ["smear_toy_updated.pdf"],
    ),
    (
        REPO / "notebooks/draft/classifier.ipynb",
        ["classifier_example_hist_exp=smeared_fvt_training_ensemble_signal=0.01.pdf"],
    ),
    (
        REPO / "notebooks/draft/smearing.ipynb",
        [
            "smearing_overlap_signal_ratio_0.0_seed_5.pdf",
            "base_and_CR_fvt_scores_hist_seed_5.pdf",
            "smearing_and_tails_signal_ratio_0.01_seed_50.pdf",
        ],
    ),
    (
        REPO / "notebooks/draft/null_case.ipynb",
        ["pull_vs_sr_stats_SR_size_0.2.pdf"],
    ),
    (
        REPO / "notebooks/draft/on_which_to_learn.ipynb",
        ["on_which_to_learn.pdf", "tsne_original_repr.pdf"],
    ),
    (
        REPO / "notebooks/draft/signal_concentration.ipynb",
        [
            "4b_vs_signal_exp=smeared_fvt_training_ensemble_signal=0.01_max_vs_mean_sr_stats_eta=inf.pdf",
            "4b_vs_signal_exp=smeared_fvt_training_ensemble_signal=0.01_max_vs_mean_sr_stats_eta=2.0.pdf",
            "4b_vs_signal_exp=smeared_fvt_training_ensemble_signal=0.01_different_noise_scales.pdf",
            "4b_vs_signal_exp=smeared_fvt_training_ensemble_signal=0.02_different_noise_scales.pdf",
            "4b_vs_signal_exp=smeared_fvt_training_ensemble_signal=0.01_vs_baseline.pdf",
            "4b_vs_signal_exp=smeared_fvt_training_ensemble_signal=0.02_vs_baseline.pdf",
            "4b_vs_signal_exp=smeared_fvt_training_ensemble_HH4b_400_signal=0.0075_different_noise_scales.pdf",
            "4b_vs_signal_exp=smeared_fvt_training_ensemble_ZH4b_signal=0.03_different_noise_scales.pdf",
        ],
    ),
]


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for notebook_path, expected_figures in NOTEBOOKS:
    started = time.time()
    with open(notebook_path, encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)
    executor = ExecutePreprocessor(timeout=3600, kernel_name="python3")
    execution_error = None
    try:
        executor.preprocess(
            notebook,
            {"metadata": {"path": str(notebook_path.parent)}},
        )
    except CellExecutionError as error:
        execution_error = error
    output_path = OUTPUT_DIR / notebook_path.name
    with open(output_path, "w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)

    missing = []
    stale = []
    for filename in expected_figures:
        figure = FIGURE_DIR / filename
        if not figure.exists() or figure.stat().st_size == 0:
            missing.append(filename)
        elif figure.stat().st_mtime < started:
            stale.append(filename)
    if missing or stale:
        raise RuntimeError(
            f"{notebook_path.name}: missing={missing}, not regenerated={stale}"
        ) from execution_error
    if execution_error is not None:
        print(
            f"warning: {notebook_path.name} stopped after its required figures "
            f"were regenerated: {execution_error.__class__.__name__}",
            flush=True,
        )
    print(f"executed {notebook_path} -> {output_path}")

print("All non-power draft notebooks executed and required figures regenerated.")
