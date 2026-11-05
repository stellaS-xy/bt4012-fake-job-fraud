"""Shared paths and constants."""
from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"

D1_SLUG = "shivamb/real-or-fake-fake-jobposting-prediction"
D2_SLUG = "srisaisuhassanisetty/fake-job-postings"

D1_DIR = RAW_DIR / "d1"
D2_DIR = RAW_DIR / "d2"
