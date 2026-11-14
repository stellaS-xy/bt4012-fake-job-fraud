# Fake Job Posting Fraud Detection (BT4012 Group Project)

A three-phase fraud analytics pipeline for online job postings:

1. **Phase 1 - Exploratory characterization** of scam patterns (fraud-only dataset) -> red-flag lexicon
2. **Phase 2 - Feature engineering + supervised detection** on the labeled EMSCAD dataset (class imbalance, SHAP, PR-AUC)
3. **Phase 3 - Agentic layer** that wraps the classifier and lexicon as tools for an LLM agent

## Team

| Name | Role |
|------|------|
| [Name 1] | EDA and feature engineering |
| [Name 2] | Modeling |
| [Name 3] | Agent layer |
| [Name 4] | Report / evaluation |

## Datasets

| Dataset | Kaggle slug | Use |
|---------|-------------|-----|
| D1 (labeled, ~17.9k rows, ~4.8% fraud) | `shivamb/real-or-fake-fake-jobposting-prediction` | Training / evaluation |
| D2 (fraud-only, ~10k rows) | `srisaisuhassanisetty/fake-job-postings` | Phase 1 pattern discovery only |

Raw data is **not committed** (see `.gitignore`). Download it with the steps below.

## Setup (reproduce locally)

```bash
# 1. Clone
git clone <REPO_URL>
cd bt4012-fake-job-fraud

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows (PowerShell/cmd)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Get the data (needs a Kaggle API token, see below)
python src/download_data.py

# 5. Sanity check
python src/check_setup.py
```

### Kaggle API token

1. Kaggle -> Settings -> API -> **Create New Token** (downloads `kaggle.json`).
2. Place it at `~/.kaggle/kaggle.json` (macOS/Linux) or `C:\Users\<you>\.kaggle\kaggle.json` (Windows).
3. Or skip the API: download both datasets manually from Kaggle and unzip the CSVs into `data/raw/d1/` and `data/raw/d2/`.

## Model evaluation

Use `notebooks/02_d1_modelling_setup.ipynb` after notebook 00 has produced the
frozen training and test files. It uses `src/evaluation.py` to keep the testing
process separate from choosing model settings:

1. Set aside a group of training ads for scoring, keeping related ads together.
2. Use smaller grouped splits of the remaining ads to compare settings and
   choose the flagging cutoff from predictions on ads unseen by each fitted model.
3. Fit the chosen pipeline on the remaining ads, then score the ads set aside
   in step 1 with the cutoff already fixed.

Every model uses the same scoring rounds. The notebook saves per-round scores,
chosen settings and cutoffs, and an approximate paired uncertainty interval.
Settings are selected by mean inner PR-AUC; the cutoff maximises F1 on the
selected setting's inner out-of-fold predictions. Text vocabularies and other
learned features are fitted again in every split. Supply the complete unfitted
feature-and-model pipeline, not features fitted on all the data in advance.

The default is five repeats of five scoring folds, three inner folds and three
candidate logistic-regression settings. For a quick development check, reduce
the repeats and candidate grid; use the full, shared plan for reported results.
The reserved test file is opened only when the final-test switch is enabled,
after the chosen model's settings and cutoff have been fixed from training data.

Run the checks from the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

### Compare logistic regression, LightGBM and XGBoost

Run the complete comparison, then open `notebooks/03_d1_model_comparison.ipynb`
to read the results:

```powershell
.\.venv\Scripts\python.exe -u -m src.model_comparison
```

All four pipelines use the same five repeats of five grouped scoring folds.
Three inner grouped folds select between two settings and choose the flagging
cutoff. The text-only logistic reference retains its 30,000-term vocabulary
limit. A2 logistic regression, LightGBM and XGBoost share identical A2 features
with a 2,000-term limit, so the tree models have a reference with matching inputs.
Class-balancing weights use only the rows in each individual fit. The grids and
tree counts are fixed in advance, and outer validation is not used for early stopping.

`reports/model_comparison/report.html` presents the average number of fake ads
caught, genuine ads wrongly flagged, and fake ads found when only 5% of each
scoring batch is reviewed. It also shows each repeat and approximate paired
uncertainty intervals. Counts are averaged across repeats; the same ads are
reused, so repeats are not additional independent datasets.

The output folder also contains per-fold scores, settings tried, the saved run
plan, package versions and data/code fingerprints. Exact row assignments and
predictions are saved under `data/processed/model_comparison/` and ignored by Git.
The final test set is not read. Completed models can be reused if the run plan
matches; an interrupted partial model starts again. To rebuild the report from
completed results without training, run:

```powershell
.\.venv\Scripts\python.exe -m src.model_comparison --report-only
```

## Repo structure

```
data/raw/          original downloads (git-ignored)
data/processed/    cleaned / feature tables (git-ignored)
notebooks/         exploratory notebooks (prefix with your initials, e.g. ab_01_eda.ipynb)
src/               reusable code (download, features, models, agent)
reports/figures/   exported figures for the report and slides
```

## Team workflow

- `main` is always runnable. Work on a branch: `eda/<topic>`, `model/<topic>`, `agent/<topic>`.
- Open a pull request and get one teammate to review before merging.
- Commit small and often, with clear messages. Contribution history is part of the peer evaluation.
- Do not commit data, API keys or `.env` files.
- Set a global random seed (`SEED = 42` in `src/config.py`) so results are reproducible.

## AI / external code acknowledgement

List any generative AI tools, external code or public notebooks used here (required by the course):

- [ ] ...
