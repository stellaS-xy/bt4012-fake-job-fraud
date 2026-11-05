"""Sanity check: libraries import, data loads, shapes and fraud rate look right."""
import importlib

import pandas as pd

from config import D1_DIR, D2_DIR

PACKAGES = ["pandas", "numpy", "sklearn", "xgboost", "shap", "imblearn", "nltk"]


def check_packages():
    for name in PACKAGES:
        try:
            importlib.import_module(name)
            print(f"[ok] {name}")
        except ImportError:
            print(f"[MISSING] {name} - run: pip install -r requirements.txt")


def load_first_csv(folder):
    files = sorted(folder.glob("*.csv"))
    if not files:
        print(f"[MISSING] no CSV in {folder} - run python src/download_data.py")
        return None
    return pd.read_csv(files[0])


if __name__ == "__main__":
    check_packages()

    d1 = load_first_csv(D1_DIR)
    if d1 is not None:
        print(f"\nD1 shape: {d1.shape}")
        print("D1 columns:", list(d1.columns))
        if "fraudulent" in d1.columns:
            print(f"D1 fraud rate: {d1['fraudulent'].mean():.2%} (expected ~4.8%)")

    d2 = load_first_csv(D2_DIR)
    if d2 is not None:
        print(f"\nD2 shape: {d2.shape}")
        print("D2 columns:", list(d2.columns))
