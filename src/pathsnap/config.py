"""Project-wide paths and constants."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Folder holding one sub-folder per partition (train/, student_test/, ...).
# Override with the PATHSNAP_DATA environment variable.
DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA", REPO_ROOT / "dataset" / "Student folder"))

RESULTS_DIR = REPO_ROOT / "results"

# Partitions available to us for development. The evaluator partition is
# never read by training or tuning code.
DEV_PARTITIONS = ("train", "student_test")

SEED = 42
