# Notebook 13: completed LightGBM versus calibrated MLP–Extra Trees comparison

Use the `model/fraud-detection` branch. No other notebook needs to run first.

## Read the completed results

Open `notebooks/13_d1_reference_comparison.ipynb`. It includes the completed
25-fold interpretation. With the dependencies installed, **Run All displays
the saved tables without starting training** (`RUN_TRAINING=False`).

The result files are in `reports/13_reference_comparison/`:

- `summary.csv`: means and sample SDs across all 25 folds per pipeline.
- `fold_metrics.csv`: all 50 model/fold rows.
- `paired_differences.csv`: ensemble minus fresh LightGBM comparisons.
- `historical_replay_check.csv`: fresh LightGBM versus the supplied historical run.
- `comparison.html`: the completed report; download and open it in a browser.
- `manifest.json`: provenance and package/source hashes from the original run.
- `completion.json`: checksums and the original run fingerprint.

| Mean metric | LightGBM | Calibrated MLP + Extra Trees |
|---|---:|---:|
| PR-AUC (average precision) | 0.6142 | 0.6891 |
| Fraud precision | 0.6019 | 0.7454 |
| Fraud recall | 0.5482 | 0.5329 |
| Fraud F1 | 0.5515 | 0.5808 |

The ensemble led on AP in 18/25 folds. The approximate corrected AP-difference
interval is [-0.0385, 0.1883], so its observed advantage is not proof of
superiority. This compares complete pipelines (LightGBM 2k versus ensemble 30k
TF-IDF), not standalone MLP versus LightGBM. No final-test performance is claimed.

## Reproduce the comparison

Clone the branch and open the complete folder:

```bash
git clone --branch model/fraud-detection https://github.com/stellaS-xy/bt4012-fake-job-fraud.git
cd bt4012-fake-job-fraud
```

Use **Python 3.11.9**. On macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-13.txt
```

On Windows PowerShell, with Python 3.11.9 installed:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-13.txt
```

Select this environment as the notebook kernel, set `RUN_TRAINING=True` and
Run All. Training runs in a background process; the last cell monitors it.
The first run downloads the verified training inputs from the
[notebook 13 inputs release](https://github.com/stellaS-xy/bt4012-fake-job-fraud/releases/tag/notebook13-inputs-v1).
For offline use, place `13_reference_training_inputs.zip` from that release in
the repository root. Datasets and model caches remain outside Git.

You can also use the activated environment in a terminal:

```bash
python -u -m src.d1_reference_comparison
```

The recovered development pool contains **14,439 postings, 693 frauds and
3,555 groups**. All **25 saved outer-fold signatures** must match. Three grouped
inner folds choose settings by mean AP and thresholds by inner OOF F1.
Ensemble calibration and neural epoch selection use further training-only
group partitions. Group IDs never become predictors.

The original CSV byte checksum could not be reconstructed, which is why the
completed comparison uses a fresh LightGBM replay as the paired reference.
The original training postings, labels, groups and outer fold positions match.
Some original reserved-test IDs may have entered earlier development under a
different split; do not claim a completely untouched project-wide final test.

## Progress and resuming

A complete CPU run can take several hours. Preserve
`data/processed/13_reference_comparison/`. Each completed inner stage and outer
prediction is cached. Run the same command/environment/settings after an
interruption to reuse compatible checkpoints. A partially completed fit restarts.

```bash
python -m src.run_reference13 --status
```

The status command checks the worker lock as well as saved metadata; an old
"running" entry does not prove a worker is alive. Interrupting the notebook's
monitor only stops the display. Stop the worker itself if you intend to stop
training. Do not start multiple copies.

New runs save CSV/HTML reports and prediction caches under a new fingerprinted
folder in `data/processed/13_reference_comparison/runs/`. The original completed
results in `reports/13_reference_comparison/` remain unchanged. Packaging changes
to input loading and platform support change the fingerprint; they do not alter
the estimator, calibration or metric implementations used for the original run.

The helpers named for experiments 10–12 are imported implementation dependencies;
their notebooks, datasets, CatBoost and semantic-encoder packages are not needed
to run this comparison.
