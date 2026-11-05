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
