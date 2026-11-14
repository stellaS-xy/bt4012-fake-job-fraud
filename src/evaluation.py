"""Grouped nested evaluation. This module never reads the reserved test file.

An outer fold measures performance. Only its training rows enter the inner
folds that choose model settings and the decision threshold. Pass the complete
unfitted feature/model pipeline as ``make_model`` so text vocabularies, category
encodings and any label-derived features are rebuilt for every fit.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from numbers import Real

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold


def _check_data(X, y, groups):
    y, groups = np.asarray(y), np.asarray(groups)
    if not hasattr(X, "iloc"):
        raise ValueError("X must be a pandas table so rows can be selected by position.")
    if y.ndim != 1 or groups.ndim != 1 or len(X) != len(y) or len(y) != len(groups):
        raise ValueError("X, y and groups must contain the same number of rows.")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Both classes (0 = legitimate, 1 = fraud) are required.")
    if pd.isna(groups).any():
        raise ValueError("Every row needs a group identifier.")
    return y, groups


def _check_scores(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("Labels and probabilities must be non-empty, matching vectors.")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Both classes are required for evaluation.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Predicted probabilities must be finite and between 0 and 1.")
    return y, p


def validate_folds(folds, y, groups):
    """Reject overlapping groups, missing predictions or invalid folds before fitting."""
    y, groups = np.asarray(y), np.asarray(groups)
    checked, seen, keys = [], {}, set()
    for repeat, fold, fit_idx, val_idx in folds:
        if not isinstance(repeat, (int, np.integer)) or not isinstance(fold, (int, np.integer)):
            raise ValueError("Repeat and fold identifiers must be integers.")
        if repeat < 0 or fold < 0 or (repeat, fold) in keys:
            raise ValueError("Repeat/fold identifiers must be unique and non-negative.")
        keys.add((repeat, fold))
        fit_idx, val_idx = np.asarray(fit_idx), np.asarray(val_idx)
        for idx in (fit_idx, val_idx):
            if idx.ndim != 1 or not len(idx) or idx.dtype.kind not in "iu":
                raise ValueError("Fold indices must be non-empty integer vectors.")
            if (idx < 0).any() or (idx >= len(y)).any() or len(np.unique(idx)) != len(idx):
                raise ValueError("Fold indices are duplicated or outside the data.")
        if len(np.intersect1d(fit_idx, val_idx)) or len(fit_idx) + len(val_idx) != len(y):
            raise ValueError("Each fold must partition all rows into training and validation.")
        if not set(groups[fit_idx]).isdisjoint(groups[val_idx]):
            raise ValueError("Related ads appear in both training and validation.")
        if len(np.unique(y[fit_idx])) != 2 or len(np.unique(y[val_idx])) != 2:
            raise ValueError("Every training and validation fold must contain both classes; use fewer folds.")
        seen.setdefault(repeat, np.zeros(len(y), dtype=int))[val_idx] += 1
        checked.append((repeat, fold, fit_idx, val_idx))
    if not checked or sorted(seen) != list(range(len(seen))):
        raise ValueError("Repeat identifiers must start at zero without gaps.")
    if any(not np.all(counts == 1) for counts in seen.values()):
        raise ValueError("Every row must be validated exactly once per repeat.")
    return checked


def make_folds(groups, y, n_splits=5, n_repeats=5, seed=42):
    """Build reproducible splits, keeping all ads in a group together."""
    y, groups = np.asarray(y), np.asarray(groups)
    _check_data(pd.DataFrame(index=range(len(y))), y, groups)
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("Use at least two folds.")
    if not isinstance(n_repeats, int) or n_repeats < 1:
        raise ValueError("Use at least one repeat.")
    if min(len(np.unique(groups[y == label])) for label in (0, 1)) < n_splits:
        raise ValueError("Too few groups containing each class; use fewer folds.")
    folds = []
    for repeat in range(n_repeats):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat)
        for fold, (fit_idx, val_idx) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
            folds.append((repeat, fold, fit_idx, val_idx))
    return validate_folds(folds, y, groups)


def best_f1_threshold(y, p):
    """Choose a cutoff using training-only out-of-fold predictions, never evaluation labels."""
    y, p = _check_scores(y, p)
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[np.argmax(f1)])


def score_summary(y, p, threshold=None, review_share=0.05):
    """Ranking/review metrics; cutoff metrics require an explicitly supplied threshold.

    Omitting a threshold never chooses one from the evaluation answers.
    """
    y, p = _check_scores(y, p)
    if not 0 < review_share <= 1:
        raise ValueError("review_share must be greater than zero and at most one.")
    top = np.argsort(-p, kind="stable")[:max(int(review_share * len(p)), 1)]
    out = {"pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p),
           "precision_at_budget": float(y[top].mean()),
           "recall_at_budget": float(y[top].sum() / y.sum()),
           "review_share": review_share, "review_count": len(top),
           "review_fraud_caught": int(y[top].sum()),
           "review_genuine_flagged": int(len(top) - y[top].sum())}
    if review_share == 0.05:
        out.update(precision_top5pct=out["precision_at_budget"], recall_top5pct=out["recall_at_budget"])
    if threshold is not None:
        if not isinstance(threshold, Real) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("The threshold must be a finite number between zero and one.")
        flagged = p >= threshold
        caught = int(((y == 1) & flagged).sum())
        false_alarms = int(((y == 0) & flagged).sum())
        out.update(threshold=float(threshold), precision=precision_score(y, flagged, zero_division=0),
                   recall=recall_score(y, flagged), f1=f1_score(y, flagged, zero_division=0),
                   share_flagged=float(flagged.mean()), fraud_caught=caught,
                   fraud_missed=int(y.sum()) - caught, genuine_flagged=false_alarms,
                   genuine_passed=int((y == 0).sum()) - false_alarms,
                   false_positive_rate=false_alarms / int((y == 0).sum()))
    return out


def _probabilities(model, X):
    classes = np.asarray(model.classes_)
    if set(classes) != {0, 1}:
        raise ValueError("Models must predict classes 0 and 1 with predict_proba.")
    return np.asarray(model.predict_proba(X))[:, np.flatnonzero(classes == 1)[0]]


@dataclass
class InnerSelection:
    params: dict
    threshold: float
    oof: np.ndarray
    candidates: pd.DataFrame


def select_settings(make_model, X, y, groups, param_grid=None, inner_splits=3, seed=42):
    """Choose settings and cutoff using only the supplied training rows.

    Every candidate uses the same grouped inner folds. Its feature pipeline is
    fitted afresh on each inner training part. Select the highest mean inner
    PR-AUC, then choose the cutoff from that candidate's held-out predictions.
    Inner scores are selection diagnostics, not reported performance estimates.
    """
    y, groups = _check_data(X, y, groups)
    folds = make_folds(groups, y, n_splits=inner_splits, n_repeats=1, seed=seed)
    grid = list(ParameterGrid({} if param_grid is None else param_grid))
    if not grid:
        raise ValueError("The parameter grid must contain at least one candidate.")
    rows, best_score, best_params, best_oof = [], -np.inf, None, None
    for params in grid:
        oof = np.full(len(y), np.nan)
        scores = []
        for _, _, fit_idx, val_idx in folds:
            model = clone(make_model()).set_params(**params)
            model.fit(X.iloc[fit_idx], y[fit_idx])
            p = _probabilities(model, X.iloc[val_idx])
            _check_scores(y[val_idx], p)
            oof[val_idx] = p
            scores.append(average_precision_score(y[val_idx], p))
        mean = float(np.mean(scores))
        rows.append({"params": dict(params), "mean_inner_pr_auc": mean})
        if mean > best_score:  # deterministic: first candidate wins a tie
            best_score, best_params, best_oof = mean, dict(params), oof
    return InnerSelection(best_params, best_f1_threshold(y, best_oof), best_oof, pd.DataFrame(rows))


def fold_signature(y, groups, fit_idx, val_idx):
    """Identify the actual rows and their split so mismatched comparisons fail."""
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(pd.DataFrame({"y": y, "group": groups}), index=False).to_numpy().tobytes())
    for idx in (fit_idx, val_idx):
        digest.update(np.asarray(sorted(idx), dtype="<i8").tobytes())
        digest.update(b"|")
    return digest.hexdigest()


def cross_validate(make_model, X, y, groups, folds=None, param_grid=None,
                   inner_splits=3, seed=42, threshold="inner", review_share=0.05, verbose=True,
                   on_fold=None):
    """Evaluate on outer folds after choosing settings and cutoff inside training.

    Returns (fold_scores, oof). Each row records the chosen settings, cutoff,
    and fold signature. The returned OOF array has one prediction per row per
    repeat. ``threshold=None`` skips cutoff metrics but still permits tuning;
    a numeric threshold uses that predeclared cutoff instead of choosing one.
    ``on_fold(row, validation_indices, probabilities)`` optionally saves progress
    or clears a feature cache after a completed outer fold.
    """
    y, groups = _check_data(X, y, groups)
    folds = make_folds(groups, y, seed=seed) if folds is None else validate_folds(list(folds), y, groups)
    if threshold is not None and threshold != "inner":
        if not isinstance(threshold, Real) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be 'inner', None, or a number between zero and one.")
    if not 0 < review_share <= 1:
        raise ValueError("review_share must be greater than zero and at most one.")
    oof = np.full((max(r for r, *_ in folds) + 1, len(y)), np.nan)
    rows, candidate_tables = [], []
    for repeat, fold, fit_idx, val_idx in folds:
        X_fit, y_fit, g_fit = X.iloc[fit_idx], y[fit_idx], groups[fit_idx]
        params, cutoff, inner_score = {}, threshold, np.nan
        if threshold == "inner" or param_grid is not None:
            selection = select_settings(make_model, X_fit, y_fit, g_fit, param_grid,
                                        inner_splits=inner_splits, seed=seed + repeat * 1000 + fold)
            params = selection.params
            cutoff = selection.threshold if threshold == "inner" else threshold
            inner_score = selection.candidates["mean_inner_pr_auc"].max()
            candidate_tables.append(selection.candidates.assign(repeat=repeat, fold=fold))
        model = clone(make_model()).set_params(**params).fit(X_fit, y_fit)
        p = _probabilities(model, X.iloc[val_idx])
        oof[repeat, val_idx] = p
        row = score_summary(y[val_idx], p, threshold=cutoff, review_share=review_share)
        row.update(repeat=repeat, fold=fold, train_rows=len(fit_idx), val_rows=len(val_idx),
                   val_fraud=int(y[val_idx].sum()), selected_params=params,
                   selection_pr_auc=inner_score,
                   fold_signature=fold_signature(y, groups, fit_idx, val_idx))
        rows.append(row)
        if verbose:
            cutoff_text = "none" if cutoff is None else f"{cutoff:.3f}"
            print(f"Repeat {repeat + 1}, fold {fold + 1}: PR-AUC {row['pr_auc']:.3f}, "
                  f"cutoff {cutoff_text}, settings {params}", flush=True)
        if on_fold is not None:
            on_fold(row.copy(), val_idx.copy(), p.copy())
    if not np.isfinite(oof).all():
        raise ValueError("Some validation predictions are missing or non-finite.")
    scores = pd.DataFrame(rows)
    scores.attrs["inner_candidates"] = pd.concat(candidate_tables, ignore_index=True) if candidate_tables else pd.DataFrame()
    return scores, oof


def summarise(fold_scores, name):
    """Average outer-fold results; never mix selection diagnostics into performance."""
    metrics = ["pr_auc", "roc_auc", "precision", "recall", "f1", "share_flagged", "false_positive_rate",
               "precision_at_budget", "recall_at_budget", "precision_top5pct", "recall_top5pct"]
    out = {m: fold_scores[m].mean() for m in metrics if m in fold_scores}
    out["pr_auc_sd"] = fold_scores["pr_auc"].std(ddof=1)
    return pd.Series(out, name=name)


def pooled_metrics(oof, y):
    """Ranking/review metrics only. Pooled validation labels never select a cutoff."""
    oof = np.asarray(oof)
    if oof.ndim != 2:
        raise ValueError("OOF predictions must have shape (repeats, rows).")
    return pd.DataFrame([score_summary(y, p) for p in oof]).mean()


def paired_comparison(scores_a, scores_b, metric="pr_auc", higher_is_better=True):
    """Approximate corrected 95% interval for a metric difference, a minus b.

    Fold identities must match, regardless of table order. The correction uses
    actual validation/training sizes rather than assuming exactly equal folds.
    This interval is approximate; repeated folds are not independent samples.
    """
    required = {"repeat", "fold", "fold_signature", metric, "train_rows", "val_rows"}
    if not required.issubset(scores_a) or not required.issubset(scores_b):
        raise ValueError("Both score tables must include fold identities and sizes.")
    a, b = (s.set_index(["repeat", "fold"]).sort_index() for s in (scores_a, scores_b))
    if a.index.has_duplicates or b.index.has_duplicates or not a.index.equals(b.index) or len(a) < 2:
        raise ValueError("Compare the same, unique outer folds (at least two).")
    if not a[["fold_signature", "train_rows", "val_rows"]].equals(b[["fold_signature", "train_rows", "val_rows"]]):
        raise ValueError("The models were evaluated on different rows or folds.")
    d = a[metric].to_numpy() - b[metric].to_numpy()
    if not np.isfinite(d).all():
        raise ValueError("Fold scores must be finite.")
    ratio = (a["val_rows"] / a["train_rows"]).mean()
    se = float(np.sqrt((1 / len(d) + ratio) * d.var(ddof=1)))
    return {"mean_diff": d.mean(), "se": se, "low": d.mean() - 2 * se,
            "high": d.mean() + 2 * se,
            "share_folds_a_better": ((d > 0) if higher_is_better else (d < 0)).mean()}
