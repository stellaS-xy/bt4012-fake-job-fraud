"""Fold-fitted hierarchical frequency features for the preregistered geo ablation."""
from collections import Counter
import re

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.features import FEATURE_SETS, GROUP, logistic_pipeline


def location_keys(values):
    """Parse country,state,city; a missing parent makes a descendant key missing.

    Country is uppercased, state/city casefolded, and whitespace normalised.
    No geocoding, inferred locations, spelling correction or label information.
    """
    rows = []
    for value in values:
        parts = [] if pd.isna(value) else str(value).split(',', 2)
        parts += [''] * (3 - len(parts))
        parts = [re.sub(r'\s+', ' ', p).strip() or None for p in parts]
        country = parts[0].upper() if parts[0] else None
        state = parts[1].casefold() if parts[1] else None
        city = parts[2].casefold() if parts[2] else None
        rows.append((None if country is None else (country,),
                     None if country is None or state is None else (country, state),
                     None if country is None or state is None or city is None else (country, state, city)))
    return rows


class HierarchicalLocationCount(BaseEstimator, TransformerMixin):
    """Train-only counts, with thresholded backoff and separate missing flags.

    An unseen flag describes the requested non-missing key, before backoff.
    Rare but seen keys back off with unseen=0. Missing outputs have count=0,
    unseen=0, missing=1, and do not borrow a parent's count. A non-missing key
    with no supported ancestor has count=0, unseen as above, missing=0.
    Country counts are backoff support only, never an extra model feature.
    """

    def __init__(self, levels=('state',), min_count=5, group_deduplicate=False):
        self.levels = levels
        self.min_count = min_count
        self.group_deduplicate = group_deduplicate

    def fit(self, X, y=None):
        if not self.levels or set(self.levels) - {'state', 'city'}:
            raise ValueError('Output levels must be state and/or city.')
        if self.min_count < 1:
            raise ValueError('min_count must be positive.')
        keys = location_keys(X['location'])
        if self.group_deduplicate and (GROUP not in X or X[GROUP].isna().any()):
            raise ValueError('Group-deduplicated counts require a non-missing group_id.')
        self.counts_ = []
        for level in range(3):
            if self.group_deduplicate:
                pairs = {(group, row[level]) for group, row in zip(X[GROUP], keys)
                         if row[level] is not None}
                counts = Counter(key for _, key in pairs)
            else:
                counts = Counter(row[level] for row in keys if row[level] is not None)
            self.counts_.append(counts)
        return self

    def describe(self, X):
        check_is_fitted(self, 'counts_')
        records = []
        for row in location_keys(X['location']):
            record = {}
            for name in self.levels:
                index = {'state': 1, 'city': 2}[name]
                key = row[index]
                missing = key is None
                unseen = not missing and self.counts_[index].get(key, 0) == 0
                resolved, support, selected_key = 'MISSING' if missing else 'UNSEEN', 0, None
                if not missing:
                    for ancestor in range(index, -1, -1):
                        count = self.counts_[ancestor].get(row[ancestor], 0)
                        if count >= self.min_count:
                            resolved = ('country', 'state', 'city')[ancestor]
                            support, selected_key = count, row[ancestor]
                            break
                prefix = f'geo_{name}'
                record.update({f'{prefix}_log_count': np.log1p(support),
                               f'{prefix}_unseen': float(unseen),
                               f'{prefix}_missing': float(missing),
                               f'{prefix}_key': 'MISSING' if missing else '|'.join(key),
                               f'{prefix}_resolved_level': resolved,
                               f'{prefix}_resolved_key': resolved if selected_key is None else '|'.join(selected_key)})
            records.append(record)
        return pd.DataFrame(records, index=X.index)

    def transform(self, X):
        return self.describe(X)[self.get_feature_names_out()].to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array([f'geo_{level}_{suffix}' for level in self.levels
                         for suffix in ('log_count', 'unseen', 'missing')], dtype=object)


def geographic_pipeline(variant, memory=None, seed=42):
    """Unchanged A2 30k balanced logistic plus only the requested geo block."""
    if variant not in ('V0', 'V1', 'V2', 'V3'):
        raise ValueError(variant)
    pipe = logistic_pipeline(FEATURE_SETS['A2'], tfidf_max_features=30000)
    pipe.set_params(memory=memory, model__random_state=seed)
    if variant != 'V0':
        encoder = HierarchicalLocationCount(
            levels=('state',) if variant == 'V1' else ('state', 'city'),
            min_count=5, group_deduplicate=variant == 'V3')
        geo = Pipeline([('count', encoder), ('scale', StandardScaler())])
        features = pipe.named_steps['features']
        features.set_params(transformers=[*features.transformers,
                            ('geography', geo, ['location', GROUP])])
    return pipe


def auxiliary_metrics(y, probabilities):
    """Descriptive PR envelope at recall>=.60; no operational cutoff is selected."""
    y, probabilities = np.asarray(y), np.asarray(probabilities)
    precision, recall, _ = precision_recall_curve(y, probabilities)
    values = {'precision_at_recall60': float(precision[recall >= .60].max())}
    for share, label in ((.02, '2'), (.05, '5')):
        selected = np.argsort(-probabilities, kind='stable')[:max(1, int(len(y) * share))]
        values[f'precision_top{label}pct'] = float(y[selected].mean())
        values[f'recall_top{label}pct'] = float(y[selected].sum() / y.sum())
    return values


def corrected_difference(candidate, reference, metric='pr_auc'):
    """Paired Nadeau–Bengio correction with a two-sided Student-t 95% interval."""
    a, b = (frame.set_index(['repeat', 'fold']).sort_index() for frame in (candidate, reference))
    if a.index.has_duplicates or b.index.has_duplicates or not a.index.equals(b.index) or len(a) < 2:
        raise ValueError('Require the same unique folds, with at least two pairs.')
    cols = ['fold_signature', 'train_rows', 'val_rows']
    if not a[cols].equals(b[cols]):
        raise ValueError('Paired comparisons require matching held-out rows and fold sizes.')
    differences = a[metric].to_numpy() - b[metric].to_numpy()
    if not np.isfinite(differences).all():
        raise ValueError('Fold differences must be finite.')
    sd = float(differences.std(ddof=1))
    ratio = float((a.val_rows / a.train_rows).mean())
    se = float(np.sqrt((1 / len(a) + ratio) * sd ** 2))
    critical = float(t.ppf(.975, df=len(a)-1))
    return dict(mean_diff=float(differences.mean()), sd_diff=sd, corrected_se=se,
                low=float(differences.mean()-critical*se), high=float(differences.mean()+critical*se),
                pairs=len(a), mean_validation_training_ratio=ratio, t_critical=critical)
