"""Notebook 10: memory-conscious DistilBERT experiment; imports never train/download.

Frozen outer assignments; grouped inner OOF threshold selection; grouped stopping
inside every fit, then a fresh fixed-duration refit. Development data only.
"""
from dataclasses import asdict, dataclass
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.d1_a2_mlp import GroupedConfig, load_development, split_plan
from src.evaluation import (best_f1_threshold, fold_signature, make_folds,
                            paired_comparison, score_summary, summarise)

MODEL = 'distilbert/distilbert-base-uncased'
REVISION = '12040accade4e8a0f71eabdb258fecc2e7e948be'
TRAIN_SHA = '61117d0309733e4dcf28d1f9836588ffaf72b7b44479113a742533134cad57ce'
# Budgets INCLUDE each field's name; the remaining tokens are separators/CLS.
FIELDS = ('title', 'company_profile', 'description', 'requirements', 'benefits')
BUDGETS = (24, 48, 96, 56, 24)  # 248 content + CLS and five SEP = 254 <= 256
NAME = '10 DistilBERT text-only'


@dataclass(frozen=True)
class Config:
    seed: int = 42
    repeats: int = 5
    outer_limit: int | None = 1
    inner_splits: int = 3
    stopping_splits: int = 5
    max_epochs: int = 3
    patience: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = .01
    batch_size: int = 2
    accumulation: int = 8
    max_length: int = 256
    device: str = 'auto'

    def __post_init__(self):
        if min(self.repeats, self.max_epochs, self.patience, self.batch_size, self.accumulation) < 1:
            raise ValueError('Training sizes must be positive.')
        if min(self.inner_splits, self.stopping_splits) < 2:
            raise ValueError('Use at least two grouped folds.')
        if self.outer_limit is not None and self.outer_limit < 1:
            raise ValueError('outer_limit must be positive or None.')
        if not 254 <= self.max_length <= 512:
            raise ValueError('Fixed field budgets need 254 tokens; DistilBERT supports at most 512.')
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('learning_rate must be positive and finite.')
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError('weight_decay must be nonnegative and finite.')
        if self.device not in {'auto', 'mps', 'cuda', 'cpu'}:
            raise ValueError('Select auto, mps, cuda or cpu.')


def device_name(request='auto'):
    import torch
    if request == 'auto':
        return 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if request == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA device is unavailable.')
    if request == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('Requested Mac MPS device is unavailable.')
    return request


def versions():
    return {name: importlib.metadata.version(name) for name in
            ['numpy', 'pandas', 'scipy', 'scikit-learn', 'torch', 'transformers']}


def load_inputs(root, config):
    path = Path(root) / 'data/processed/d1_train.csv'
    if hashlib.sha256(path.read_bytes()).hexdigest() != TRAIN_SHA:
        raise ValueError('Restore the agreed frozen d1_train.csv before continuing.')
    frame = load_development(path)
    if not set(FIELDS).issubset(frame):
        raise ValueError('Development file lacks required text fields.')
    folds = split_plan(frame, GroupedConfig(repeats=config.repeats, outer_limit=None),
                       Path(root) / 'data/processed/model_cv_assignments.csv')
    return frame, folds


def head_tail(ids, budget):
    """Keep both ends of long fields; no learnt vocabulary or label access."""
    if len(ids) <= budget:
        return list(ids)
    head = (budget + 1) // 2
    return list(ids[:head]) + list(ids[-(budget - head):]) if budget > 1 else list(ids[:1])


def encode_posting(row, tokenizer):
    tokens = [tokenizer.cls_token_id]
    truncated = []
    for field, budget in zip(FIELDS, BUDGETS):
        value = row[field]
        text = 'missing' if pd.isna(value) or not str(value).strip() else str(value)
        # Prefixes are retained even when head/tail truncation is necessary.
        prefix = tokenizer.encode(field.replace('_', ' ') + ': ', add_special_tokens=False)
        body = tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)
        available = budget - len(prefix)
        if available < 1:
            raise ValueError('Tokenizer field prefix exceeds its fixed budget.')
        truncated.append(len(body) > available)
        tokens.extend(prefix + head_tail(body, available) + [tokenizer.sep_token_id])
    return tokens, truncated


def prepare_tokens(frame, config, directory):
    """Uses a fixed pretrained tokenizer; no labels or train-fitted text statistics."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    path = Path(directory) / 'tokens.npz'
    if path.is_file():
        with np.load(path, allow_pickle=False) as saved:
            if not np.array_equal(saved['job_id'], frame.job_id):
                raise ValueError('Token cache job order differs.')
            return saved['ids'].copy(), saved['truncated'].copy(), tokenizer.pad_token_id
    ids = np.full((len(frame), config.max_length), tokenizer.pad_token_id, dtype=np.int32)
    clipped = np.zeros((len(frame), len(FIELDS)), dtype=bool)
    for i, (_, row) in enumerate(frame.iterrows()):
        values, clipped[i] = encode_posting(row, tokenizer)
        if len(values) > config.max_length:
            raise ValueError('Encoded posting exceeds configured token budget.')
        ids[i, :len(values)] = values
    _save_npz(path, ids=ids, truncated=clipped, job_id=frame.job_id.to_numpy())
    return ids, clipped, tokenizer.pad_token_id


def cache_directory(root, config):
    settings = asdict(config)
    settings.pop('outer_limit')
    settings.pop('repeats')
    settings['actual_device'] = device_name(config.device)
    identity = {'data_sha': TRAIN_SHA, 'model': MODEL, 'revision': REVISION,
                'config': settings, 'versions': versions()}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
    for path in [Path(__file__), Path(__file__).with_name('d1_a2_mlp.py'),
                 Path(__file__).with_name('evaluation.py'),
                 Path(root) / 'data/processed/model_cv_assignments.csv']:
        digest.update(path.read_bytes())
    return Path(root) / 'data/processed/10_distilbert' / digest.hexdigest()[:20]


def _save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _release():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _new_model(config, device, seed):
    import torch
    from transformers import AutoModelForSequenceClassification, set_seed
    set_seed(seed)
    torch.set_num_threads(2)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, revision=REVISION, num_labels=2, use_safetensors=True,
        attn_implementation='eager', id2label={0: 'LEGITIMATE', 1: 'FRAUD'},
        label2id={'LEGITIMATE': 0, 'FRAUD': 1})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    return model, optimizer


def _batch(ids, indices, pad, device):
    import torch
    values = ids[indices]
    width = max(1, int((values != pad).sum(axis=1).max()))
    values = values[:, :width]
    return {'input_ids': torch.as_tensor(values.copy(), dtype=torch.long, device=device),
            'attention_mask': torch.as_tensor(values != pad, dtype=torch.long, device=device)}


def _train_epoch(model, optimizer, ids, y, indices, pad, config, device, seed, max_steps=None):
    import torch
    model.train()
    indices = np.random.default_rng(seed).permutation(indices)
    counts = np.bincount(y[indices], minlength=2)
    if not counts.min():
        raise ValueError('Every fitting subset must contain both classes.')
    weights = torch.tensor(len(indices) / (2 * counts), dtype=torch.float32, device=device)
    batches = [indices[i:i + config.batch_size] for i in range(0, len(indices), config.batch_size)]
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    steps = 0
    for first in range(0, len(batches), config.accumulation):
        window = batches[first:first + config.accumulation]
        count = sum(map(len, window))
        for selected in window:
            labels = torch.as_tensor(y[selected].copy(), dtype=torch.long, device=device)
            logits = model(**_batch(ids, selected, pad, device)).logits
            # Weighted sample mean preserves class weighting even for batch size 1/2.
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=weights, reduction='sum') / count
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        steps += 1
        if steps % 100 == 0:
            print(f'    {steps} optimizer steps; {(time.monotonic()-started)/60:.1f} minutes', flush=True)
        if max_steps is not None and steps >= max_steps:
            break
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()
    return {'optimizer_steps': steps, 'seconds': time.monotonic() - started}


def _predict(model, ids, indices, pad, config, device):
    import torch
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(indices), config.batch_size):
            logits = model(**_batch(ids, indices[start:start + config.batch_size], pad, device)).logits
            result.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.asarray(result, dtype=float)


def resource_check(frame, ids, pad, config, folds):
    """Three optimizer updates on training rows only; never a performance result."""
    import torch
    _, _, fitting, _ = folds[0]
    model, optimizer = _new_model(config, device_name(config.device), config.seed)
    try:
        timing = _train_epoch(model, optimizer, ids, frame.fraudulent.to_numpy(), fitting,
                              pad, config, device_name(config.device), config.seed, max_steps=3)
        import psutil
        timing['process_rss_mb'] = psutil.Process().memory_info().rss / 2**20
        timing['available_system_ram_gb'] = psutil.virtual_memory().available / 2**30
        timing['device'] = device_name(config.device)
        timing['note'] = 'Resource diagnostic only. Startup and warm-up affect timing; no performance estimate.'
        if timing['device'] == 'mps':
            timing['mps_allocated_mb'] = torch.mps.current_allocated_memory() / 2**20
        return timing
    finally:
        del model, optimizer
        _release()


def _fit_predict(frame, ids, pad, fitting, scoring, config, seed, path):
    if path.is_file():
        with np.load(path, allow_pickle=False) as saved:
            if not np.array_equal(saved['job_id'], frame.iloc[scoring].job_id):
                raise ValueError('Cached scoring jobs differ.')
            print(f'  Reusing {path.name}', flush=True)
            return saved['p'].copy(), json.loads(str(saved['details'].item()))
    y, groups = frame.fraudulent.to_numpy(), frame.group_id.to_numpy()
    _, _, a, b = make_folds(groups[fitting], y[fitting], n_splits=config.stopping_splits,
                             n_repeats=1, seed=seed)[0]
    train_idx, stop_idx = fitting[a], fitting[b]
    device = device_name(config.device)
    best, epoch_chosen, stale, history = -np.inf, 1, 0, []
    model, optimizer = _new_model(config, device, seed)
    try:
        for epoch in range(1, config.max_epochs + 1):
            timing = _train_epoch(model, optimizer, ids, y, train_idx, pad, config, device, seed + epoch)
            p = _predict(model, ids, stop_idx, pad, config, device)
            ap = float(average_precision_score(y[stop_idx], p))
            history.append({'epoch': epoch, 'stopping_pr_auc': ap, **timing})
            print(f'  Stopping epoch {epoch}: AP={ap:.4f}; {timing["seconds"]/60:.1f} minutes', flush=True)
            if ap > best:
                best, epoch_chosen, stale = ap, epoch, 0
            else:
                stale += 1
            if stale >= config.patience:
                break
    finally:
        del model, optimizer
        _release()
    model, optimizer = _new_model(config, device, seed)
    try:
        print(f'  Fresh refit: {len(fitting)} rows, {epoch_chosen} fixed epochs', flush=True)
        for epoch in range(1, epoch_chosen + 1):
            _train_epoch(model, optimizer, ids, y, fitting, pad, config, device, seed + epoch)
        p = _predict(model, ids, scoring, pad, config, device)
    finally:
        del model, optimizer
        _release()
    details = {'selected_epoch': epoch_chosen, 'history': history, 'seed': seed}
    _save_npz(path, p=p, job_id=frame.iloc[scoring].job_id.to_numpy(), details=json.dumps(details))
    return p, details


def run(root, frame, folds, ids, pad, config, directory):
    """One outer fold by default; inner/outer prediction caches resume separately."""
    root, directory = Path(root), Path(directory)
    y, groups = frame.fraudulent.to_numpy(), frame.group_id.to_numpy()
    requested = folds if config.outer_limit is None else folds[:config.outer_limit]
    rows = []
    output = root / 'reports/classifier_distilbert' / directory.name
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'configuration': asdict(config), 'versions': versions(), 'device': device_name(config.device),
                'model': MODEL, 'revision': REVISION, 'training_sha256': TRAIN_SHA,
                'assignments_sha256': hashlib.sha256((root / 'data/processed/model_cv_assignments.csv').read_bytes()).hexdigest(),
                'final_test_accessed': False, 'features': 'text only; field-budget head/tail truncation',
                'field_budgets': dict(zip(FIELDS, BUDGETS)), 'completed_outer_folds': 0,
                'requested_outer_folds': len(requested), 'full_protocol_outer_folds': len(folds),
                'selection_note': 'Single-fold results are exploratory; no adoption interval is computed from one fold.'}
    for repeat, fold, fitting, scoring in requested:
        print(f'Outer repeat {repeat+1}, fold {fold+1}', flush=True)
        signature = fold_signature(y, groups, fitting, scoring)
        seed = config.seed + repeat * 1000 + fold
        inner = make_folds(groups[fitting], y[fitting], n_splits=config.inner_splits, n_repeats=1, seed=seed)
        inner_oof = np.full(len(fitting), np.nan)
        inner_scores = []
        for _, inner_fold, a, b in inner:
            print(f' Inner fold {inner_fold+1}/{config.inner_splits}', flush=True)
            p, _ = _fit_predict(frame, ids, pad, fitting[a], fitting[b], config, seed,
                                directory / f'r{repeat}_f{fold}_{signature[:12]}_inner{inner_fold}.npz')
            inner_oof[b] = p
            inner_scores.append(average_precision_score(y[fitting[b]], p))
        cutoff = best_f1_threshold(y[fitting], inner_oof)
        p, details = _fit_predict(frame, ids, pad, fitting, scoring, config, seed,
                                  directory / f'r{repeat}_f{fold}_{signature[:12]}_outer.npz')
        row = {'model': NAME, 'repeat': repeat, 'fold': fold, 'fold_signature': signature,
               'train_rows': len(fitting), 'val_rows': len(scoring),
               'selected_epoch': details['selected_epoch'], 'selection_pr_auc': float(np.mean(inner_scores)),
               **score_summary(y[scoring], p, threshold=cutoff)}
        rows.append(row)
        scores = pd.DataFrame(rows)
        scores.to_csv(output / 'fold_scores.csv', index=False)
        comparisons, sources = compare_saved(root, scores)
        comparisons.to_csv(output / 'model_comparison.csv', index=False)
        sources.to_csv(output / 'comparison_sources.csv', index=False)
        manifest['completed_outer_folds'] = len(rows)
        (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
        print(f'Outer AP={row["pr_auc"]:.4f}; F1={row["f1"]:.4f}; cutoff={cutoff:.4f}', flush=True)
    return {'scores': scores, 'comparison': comparisons, 'sources': sources,
            'output_dir': str(output), 'cache_dir': str(directory)}


def compare_saved(root, scores):
    """Match historical outer folds; never fit any baseline."""
    root = Path(root)
    summary = [dict(summarise(scores, NAME), model=NAME, matched_folds=len(scores))]
    sources = []
    references = [('02 A2 logistic 30k', root / 'reports/tables/model_logreg_a2_fold_scores.csv'),
                  ('03 revised A2 LightGBM 2k', root / 'reports/model_comparison_revised/lightgbm_a2_2k_fold_scores.csv')]
    for name, path in references:
        try:
            historical = pd.read_csv(path).set_index(['repeat', 'fold'])
            current = scores.set_index(['repeat', 'fold'])
            if historical.index.has_duplicates:
                raise ValueError('Historical fold keys are duplicated.')
            selected = historical.loc[current.index].reset_index()
            identity = ['fold_signature', 'train_rows', 'val_rows']
            if not selected[identity].equals(scores.reset_index(drop=True)[identity]):
                raise ValueError('Historical fold identities differ.')
            row = dict(summarise(selected, name), model=name, matched_folds=len(selected))
            row['distilbert_minus_reference_pr_auc'] = float(scores.pr_auc.mean() - selected.pr_auc.mean())
            if len(scores) >= 2:
                paired = paired_comparison(scores, selected)
                row.update(delta_corrected_low=paired['low'], delta_corrected_high=paired['high'])
            summary.append(row)
            sources.append({'model': name, 'status': 'included', 'reason': 'Matching saved outer folds; historical training methods may differ.'})
        except (OSError, ValueError, KeyError) as exc:
            sources.append({'model': name, 'status': 'excluded', 'reason': str(exc)})
    return pd.DataFrame(summary), pd.DataFrame(sources)
