# Final Ensemble v1 classifier evaluation

The selected model is a fixed 50/50 average of separately sigmoid-calibrated MLP
and Extra Trees scores. A2 preprocessing includes structured and text-style
features plus up to 30,000 word TF-IDF terms. The threshold
**0.20923746948802296** was selected by maximum F1 on three grouped development
OOF folds. Calibration uses a development group holdout; there is no component
refit after calibration.

## Selection and test results

Notebook 13 reports mean AP **0.689087** and fold SD **0.056569** over five repeats
of five grouped outer folds. The fresh LightGBM replay mean is **0.614218**.
The corrected paired AP interval includes zero, so the higher ensemble mean does
not establish statistical superiority. These are development results, separate
from the final fixed-split evaluation below.

Final-fit/test job **1177635** completed in **6m31s**. It evaluated 3,441 postings,
including 173 frauds, from 885 groups. The [saved aggregate JSON](test_metrics.json)
preserves the original metrics and 1,000 group-bootstrap samples' percentile
intervals; it omits host-specific artifact paths and posting-level data.

| Metric | Value | Group-bootstrap 95% interval |
|---|---|---|
| AP, labelled PR-AUC in project outputs | 0.784770 | 0.623595–0.896200 |
| ROC-AUC | 0.950306 | 0.907939–0.979872 |
| Precision | 0.677249 | 0.537385–0.780966 |
| Recall | 0.739884 | 0.568684–0.876069 |
| F1 | 0.707182 | 0.573808–0.809772 |
| False-positive rate | 0.018666 | 0.012109–0.028062 |

At the frozen threshold: **128** frauds caught, **45** missed, **61** genuine ads
flagged and **3,207** genuine ads below cutoff. Ranking the highest-scoring 5%
selects 172 postings, including 124 frauds and 48 genuine ads: precision 0.720930,
recall 0.716763. The ranking budget and fixed-threshold decisions are different
operating rules.

Current posting/group separation passed audit. Historical project-wide test
exposure remains unverified; retain the original history note. The bootstrap
intervals describe this model's uncertainty, not paired superiority over another
model. Scores are calibrated within the development protocol, not guaranteed
real-world probabilities.

## Provenance and usage

The JSON retains original creation time, frozen-plan hash, prediction-file hash,
model ZIP hash and a checksum of its source metrics file. It does not contain
posting IDs, labels per posting, model weights or raw predictions. Training/test
source hashes are recorded in the final runner and linked experiment manifests.
The original data files remain ignored under `data/processed/`.

Use [Notebook 15](../../notebooks/15_final_ensemble_test_results.ipynb) to view the
original downloaded metrics without training. Model deployment and release
instructions are in the [deployment guide](../../docs/DEPLOYMENT.md). The
[agent audit](../agent_evaluation/ensemble_v1_100cases/README.md) separately
assesses tool execution and explanation quality.
