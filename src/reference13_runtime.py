"""Portable locking and verified fixed-input download for notebook 13."""
from contextlib import contextmanager
import hashlib
import io
import os
from pathlib import Path
import urllib.request
import zipfile


@contextmanager
def training_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        if os.name == 'nt':
            import msvcrt
            handle.seek(0, 2)
            if handle.tell() == 0:
                handle.write(b'0')
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise BlockingIOError('A notebook 13 worker already holds the lock.') from error
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def worker_active(base):
    try:
        with training_lock(Path(base) / 'run.lock'):
            return False
    except BlockingIOError:
        return True


def ensure_inputs(root, expected_sha):
    root = Path(root)
    reference = root / 'data/processed/13_reference_comparison/reference'
    names = {'recovered_train.csv', 'outer_assignments.csv', 'recovery.json',
             'lightgbm_a2_2k_plan.json', 'lightgbm_a2_2k_complete.json',
             'lightgbm_a2_2k_fold_scores.csv'}
    if all((reference / name).is_file() for name in names):
        return reference
    archive_name = '13_reference_training_inputs.zip'
    local = next((p for p in (root / archive_name, root / 'data/processed' / archive_name) if p.is_file()), None)
    if local is None:
        url = ('https://github.com/stellaS-xy/bt4012-fake-job-fraud/releases/download/'
               'notebook13-inputs-v1/' + archive_name)
        print('Downloading notebook 13 fixed training inputs...', flush=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = response.read()
    else:
        payload = local.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise ValueError('Notebook 13 training archive checksum mismatch.')
    allowed = {str((reference / name).relative_to(root)).replace('\\', '/') for name in names}
    with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
        if set(bundle.namelist()) != allowed or len(bundle.namelist()) != len(allowed):
            raise ValueError('Unexpected notebook 13 training archive contents.')
        for name in allowed:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.read(name))
    return reference
