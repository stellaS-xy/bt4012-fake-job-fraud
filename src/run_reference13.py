"""Start/resume experiment 13 as a durable background job, or show its status.

From the repository root:
    .venv/bin/python -m src.run_reference13
    .venv/bin/python -m src.run_reference13 --status
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from src.reference13_runtime import worker_active

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/processed/13_reference_comparison'


def status():
    active = BASE / 'active_run.json'
    if active.exists():
        directory = Path(json.loads(active.read_text())['directory'])
        print(f'Results: {directory}')
        if (directory / 'progress.json').exists():
            progress = json.loads((directory / 'progress.json').read_text())
            if not progress.get('complete') and not worker_active(BASE):
                progress['status'] = 'stopped; saved status was stale; resume with the launch command'
            print(json.dumps(progress, indent=2))
    log = BASE / 'run.log'
    if log.exists():
        print('\nLatest log lines:')
        print('\n'.join(log.read_text(errors='replace').splitlines()[-15:]))


def start():
    BASE.mkdir(parents=True, exist_ok=True)
    if worker_active(BASE):
        print('The comparison is already running.')
        status()
        return
    isolated = BASE / 'runtime/venv/bin/python'
    python = Path(os.environ.get('REFERENCE13_PYTHON') or (str(isolated) if isolated.exists() else sys.executable))
    cache = BASE / 'runtime/cache'
    cache.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED='1', TF_CPP_MIN_LOG_LEVEL='2',
               MPLCONFIGDIR=str(cache / 'matplotlib'), XDG_CACHE_HOME=str(cache),
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', VECLIB_MAXIMUM_THREADS='2')
    with (BASE / 'run.log').open('a') as log:
        process = subprocess.Popen([str(python), '-u', '-m', 'src.d1_reference_comparison'],
            cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
    (BASE / 'launcher.json').write_text(json.dumps({'pid': process.pid, 'log': str(BASE / 'run.log')}, indent=2))
    # Keep this Mac awake only while the training process exists.
    caffeinate = shutil.which('caffeinate')
    if caffeinate:
        subprocess.Popen([caffeinate, '-i', '-w', str(process.pid)], stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    print(f'Started/resumed all 25 folds per model; PID {process.pid}. Log: {BASE / "run.log"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status', action='store_true')
    if parser.parse_args().status:
        status()
    else:
        start()
