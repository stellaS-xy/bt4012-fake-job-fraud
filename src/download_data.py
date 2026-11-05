"""Download both Kaggle datasets into data/raw/. Requires ~/.kaggle/kaggle.json."""
import subprocess
import sys

from config import D1_DIR, D1_SLUG, D2_DIR, D2_SLUG


def download(slug, dest):
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.glob("*.csv")):
        print(f"[skip] {slug}: CSV already present in {dest}")
        return
    print(f"[download] {slug} -> {dest}")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"],
        check=True,
    )


if __name__ == "__main__":
    try:
        download(D1_SLUG, D1_DIR)
        download(D2_SLUG, D2_DIR)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"Download failed: {exc}")
        print("Check that the kaggle CLI is installed and ~/.kaggle/kaggle.json exists,")
        print("or download the CSVs manually into data/raw/d1/ and data/raw/d2/.")
        sys.exit(1)
