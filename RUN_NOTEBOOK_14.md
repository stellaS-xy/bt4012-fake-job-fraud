# Run notebook 14

Use the `model/fraud-detection` branch. Notebook 14 is independent of the other
notebooks: it downloads the fixed training inputs, checks the original report
folds and runs all selected trials. It does not score the reserved test set.

## Setup

Clone the branch and open the complete repository folder in VS Code:

```bash
git clone --branch model/fraud-detection https://github.com/stellaS-xy/bt4012-fake-job-fraud.git
cd bt4012-fake-job-fraud
```

Create a **Python 3.11.9** virtual environment. For macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-14.txt
```

For Windows PowerShell, with Python 3.11.9 installed:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-14.txt
```

Install VS Code's Python and Jupyter extensions, open
`notebooks/14_d1_mlp_development.ipynb`, select this `.venv` kernel and **Run All**.
`RUN_TRAINING=True` and `OUTER_LIMIT=None` already request the full run.

## Fixed data and evaluation

The first cell downloads the 13.2 MB training-input ZIP from the
[notebook 14 inputs release](https://github.com/stellaS-xy/bt4012-fake-job-fraud/releases/tag/notebook14-inputs-v1).
It verifies the archive checksum, recovered training-file checksum, source
hashes and all 25 original outer-fold signatures. No Kaggle credentials or
earlier notebook executions are needed. For an offline machine, download the
same `14_mlp_training_inputs.zip` and put it in the repository root before running.

The training pool contains **14,439 postings, 693 frauds and 3,555 groups**.
The comparison uses **5 repeats × 5 grouped outer folds**, with 3 grouped inner
folds for selection. **PR-AUC (computed as average precision) is primary**;
fraud precision, recall and F1 and separate top-5% review measures are reported.
All thresholds, epochs and blend weights are selected within training data.

The recovered data matches the report's IDs, labels, groups and outer fold
positions; the original training CSV bytes were unavailable. Historical Colab
scores are not substituted for new results. This notebook evaluates complete
pipelines and does not guarantee that MLP will rank first.

## Runtime, resuming and results

This runs 11 trial procedures on 25 folds and can take many hours. A GPU may
help neural fitting. Keep the machine awake and preserve its output folder.
Completed inner stages and outer predictions are cached. After an interruption,
use the same environment/settings and Run All again to resume. An interrupted
individual fit restarts. Do not run two copies in the same output folder.

Results are written under `data/processed/14_mlp_development/<fingerprint>/`:

- `fold_metrics.csv`: completed model/fold measurements.
- `progress.json`: counts of completed folds per trial.
- `manifest.json`: exact settings, data, package and code fingerprints.
- `matched_summary.csv`: mean and SD using folds completed by every selected trial.
- `paired_differences.csv`: paired comparisons against the matched A2 LR baseline.
- `matched_report.html`: the comparison table for reading outside Jupyter.

Save the executed notebook. Return it along with the CSV/JSON/HTML files above.
The latter comparison files are generated when execution reaches the summary
cells; if interrupted, `fold_metrics.csv` and the cached stages preserve progress.
Model caches, datasets and virtual environments stay outside Git.
