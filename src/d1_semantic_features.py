"""Frozen MiniLM vectors with corrected whitespace cleaning and bounded windows.

Only row-local text and a pinned pretrained encoder are used. No corpus fitting
or labels enter this cache; downstream scaling/classification stays in folds.
"""
from dataclasses import asdict, dataclass
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import re

import numpy as np

TEXT_FIELDS = ('title', 'company_profile', 'description', 'requirements', 'benefits')


@dataclass(frozen=True)
class EncoderConfig:
    chunk_tokens: int = 192
    overlap: int = 32
    max_chunks: int = 4
    batch_size: int = 32
    row_batch: int = 64
    threads: int = 2

    def __post_init__(self):
        if not 0 <= self.overlap < self.chunk_tokens or min(self.max_chunks, self.batch_size, self.row_batch, self.threads) < 1:
            raise ValueError('Invalid encoder budget.')


def clean_text(value):
    value = html.unescape(str(value))
    value = re.sub(r'#(URL|EMAIL|PHONE)_[A-Za-z0-9]+#',
                   lambda m: ' ' + m.group(1).lower() + 'ph ', value)
    value = re.sub(r'<[^>]+>', ' ', value)
    return re.sub(r'\s+', ' ', value).strip()


def text_windows(text, tokenizer, model_limit, config):
    ids = tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)
    capacity = min(config.chunk_tokens, model_limit - tokenizer.num_special_tokens_to_add(pair=False))
    if capacity <= config.overlap:
        raise ValueError('Encoder token limit too small.')
    if not ids:
        return [''], False
    starts = list(range(0, max(1, len(ids) - capacity + 1), capacity - config.overlap))
    if starts[-1] + capacity < len(ids):
        starts.append(max(0, len(ids) - capacity))
    capped = len(starts) > config.max_chunks
    if capped:
        starts = [starts[i] for i in np.linspace(0, len(starts) - 1, config.max_chunks, dtype=int)]
    windows = []
    for start in starts:
        end = min(start + capacity, len(ids))
        value = tokenizer.decode(ids[start:end], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        while len(tokenizer.encode(value, add_special_tokens=True, truncation=False, verbose=False)) > model_limit:
            end -= 1
            if end <= start:
                raise ValueError('Unable to encode a bounded window.')
            value = tokenizer.decode(ids[start:end], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        windows.append(value)
    return windows, capped


def prepare_embeddings(root, frame, config=EncoderConfig()):
    root = Path(root)
    model_info_path = root / 'data/processed/12_models/minilm_revision.json'
    if not model_info_path.is_file():
        raise FileNotFoundError('Download the pinned MiniLM encoder first; no automatic network call is made here.')
    model_info = json.loads(model_info_path.read_text())
    raw = frame[list(TEXT_FIELDS)].fillna('').astype(str).agg(' '.join, axis=1)
    texts = raw.map(clean_text)
    versions = {name: importlib.metadata.version(name)
                for name in ('numpy', 'sentence-transformers', 'transformers', 'torch')}
    request = {'config': asdict(config), 'encoder': model_info, 'versions': versions,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'texts_sha256': hashlib.sha256(texts.to_csv(index=False, lineterminator='\n').encode()).hexdigest(),
        'job_ids_sha256': hashlib.sha256(frame.job_id.to_numpy(dtype='<i8').tobytes()).hexdigest()}
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:20]
    directory = root / 'data/processed/12_semantic_embeddings' / key
    directory.mkdir(parents=True, exist_ok=True)
    complete, output = directory / 'complete.json', directory / 'embeddings.npy'
    if complete.is_file():
        manifest = json.loads(complete.read_text())
        if manifest['request'] != request or hashlib.sha256(output.read_bytes()).hexdigest() != manifest['array_sha256']:
            raise ValueError('Embedding cache integrity mismatch.')
        return np.load(output, allow_pickle=False), manifest, directory
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(config.threads)
    encoder = SentenceTransformer(model_info['local_dir'], device='cpu', local_files_only=True,
                                  trust_remote_code=False)
    width = encoder.get_embedding_dimension()
    # Per-row-batch checkpoints resume the expensive frozen inference stage.
    values = np.zeros((len(frame), width), dtype=np.float32)
    capped_total, window_total = 0, 0
    for start in range(0, len(frame), config.row_batch):
        stop = min(start + config.row_batch, len(frame))
        checkpoint = directory / f'rows_{start}_{stop}.npz'
        if checkpoint.is_file():
            with np.load(checkpoint, allow_pickle=False) as saved:
                if not np.array_equal(saved['job_id'], frame.iloc[start:stop].job_id.to_numpy()):
                    raise ValueError('Embedding checkpoint posting IDs differ.')
                values[start:stop] = saved['vectors']
                capped_total += int(saved['capped'])
                window_total += int(saved['windows'])
            continue
        windows, owners, capped = [], [], 0
        for local, text in enumerate(texts.iloc[start:stop]):
            chunks, limited = text_windows(text, encoder.tokenizer, encoder.max_seq_length, config)
            windows.extend(chunks)
            owners.extend([local] * len(chunks))
            capped += int(limited)
        vectors = encoder.encode(windows, batch_size=config.batch_size, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False)
        pooled = np.zeros((stop - start, width), dtype=np.float32)
        counts = np.zeros(stop - start, dtype=int)
        for owner, vector in zip(owners, vectors):
            pooled[owner] += vector
            counts[owner] += 1
        pooled /= counts[:, None]
        pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
        tmp = checkpoint.with_suffix('.tmp')
        with tmp.open('wb') as stream:
            np.savez_compressed(stream, vectors=pooled, job_id=frame.iloc[start:stop].job_id.to_numpy(),
                                capped=capped, windows=len(windows))
        tmp.replace(checkpoint)
        values[start:stop] = pooled
        capped_total += capped
        window_total += len(windows)
        if start == 0 or (start // config.row_batch) % 8 == 0 or stop == len(frame):
            print(f'Frozen embeddings: {stop:,}/{len(frame):,} postings', flush=True)
    tmp = output.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        np.save(stream, values, allow_pickle=False)
    tmp.replace(output)
    manifest = {'request': request, 'rows': len(frame), 'dimensions': width,
        'sampled_long_postings': capped_total, 'encoded_windows': window_total,
        'array_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'labels_used': False, 'final_test_accessed': False}
    complete.write_text(json.dumps(manifest, indent=2))
    return values, manifest, directory
