"""Run a copy of notebook 10 for ONE outer fold on a Slurm CUDA node.

Only the executed copy is changed: installation off, tokens/training on,
resource diagnostic off, outer_limit=1, device=cuda. Never writes the source
notebook or any MLP code/environment/cache. Imports do not execute training.
"""
import ast
from copy import deepcopy
from pathlib import Path
import os

FLAGS = {'INSTALL_DEPENDENCIES': False, 'OUTER_LIMIT': 1,
         'PREPARE_TOKENS': True, 'RUN_RESOURCE_CHECK': False, 'TRAIN_MODEL': True}


def prepare_notebook(notebook):
    """Validate and override known top-level settings in an in-memory copy."""
    notebook = deepcopy(notebook)
    counts = dict.fromkeys(FLAGS, 0)
    configs = 0
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = cell['source']
        source = ''.join(source) if isinstance(source, list) else source
        cell['source'] = source
        tree = ast.parse(source)
        changed = False
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            name = node.targets[0].id
            if name in FLAGS:
                counts[name] += 1
                node.value = ast.Constant(FLAGS[name])
                changed = True
            if name == 'CONFIG':
                if not (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                        and isinstance(node.value.func.value, ast.Name)
                        and node.value.func.value.id == 'experiment' and node.value.func.attr == 'Config'):
                    raise ValueError('Notebook CONFIG layout changed; review the launcher before training.')
                keys = [key for key in node.value.keywords if key.arg == 'device']
                if len(keys) != 1:
                    raise ValueError('Expected exactly one CONFIG device setting.')
                keys[0].value = ast.Constant('cuda')
                configs += 1
                changed = True
        if changed:
            cell['source'] = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
        cell['outputs'] = []
        cell['execution_count'] = None
    if any(count != 1 for count in counts.values()) or configs != 1:
        raise ValueError(f'Unexpected notebook settings; refusing to run: {counts}, CONFIG={configs}')
    return notebook


def main():
    import nbformat
    from nbclient import NotebookClient

    job = os.environ.get('SLURM_JOB_ID')
    if not job:
        raise SystemExit('Submit cluster/distilbert_single_fold.slurm from the repository root.')
    root = Path(__file__).resolve().parents[1]
    output = root / 'reports/10_distilbert_runs' / job
    output.mkdir(parents=True, exist_ok=True)
    source = root / 'notebooks/10_distilbert_finetuning.ipynb'
    notebook = prepare_notebook(nbformat.read(source, as_version=4))
    # Tee kernel output to a file for live tail -f, while retaining notebook output.
    prelude = '''import sys
class _DistilbertTee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log
    def write(self, text):
        self.log.write(text)
        self.log.flush()
        return self.stream.write(text)
    def flush(self):
        self.stream.flush()
        self.log.flush()
    def __getattr__(self, name):
        return getattr(self.stream, name)
_distilbert_log = open(PROGRESS_PATH, 'a', buffering=1)
sys.stdout = _DistilbertTee(sys.stdout, _distilbert_log)
sys.stderr = _DistilbertTee(sys.stderr, _distilbert_log)
print('Cluster notebook 10: one outer fold; CUDA required; no dependency installation.')
'''.replace('PROGRESS_PATH', repr(str(output / 'progress.log')))
    notebook.cells.insert(0, nbformat.v4.new_code_cell(prelude))
    destination = output / '10_distilbert.executed.ipynb'

    def checkpoint_notebook(**kwargs):
        nbformat.write(notebook, destination)

    client = NotebookClient(notebook, timeout=None, kernel_name='python3',
                            resources={'metadata': {'path': str(root)}},
                            on_cell_executed=checkpoint_notebook)
    print(f'Live progress: {output / "progress.log"}', flush=True)
    print(f'Executed notebook: {destination}', flush=True)
    try:
        client.execute()
    finally:
        nbformat.write(notebook, destination)
        print(f'Saved notebook outputs: {destination}', flush=True)


if __name__ == '__main__':
    main()
