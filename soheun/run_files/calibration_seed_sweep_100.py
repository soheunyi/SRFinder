"""Run the calibration-figure audit over all 100 seeds."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import calibration_seed_sweep as sweep


sweep.OUT = sweep.DATA_REPO / "data/refit_bootstrap/calibration_seed_sweep_100_v1"
sweep.CAMPAIGN_NAME = "calibration_seed_sweep_100_v1"
sweep.SEEDS = list(range(100))


if __name__ == "__main__":
    sweep.main()
