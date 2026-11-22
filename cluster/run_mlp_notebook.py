"""Execute the MLP notebook on an allocated GPU; never import to launch work."""
from pathlib import Path
import os


def main():
    import nbformat
    from nbclient import NotebookClient

    root = Path(__file__).resolve().parents[1]
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Launch through cluster/mlp_model.slurm on a compute node.')
    # Keep historical output directory/experiment IDs compatible with saved runs.
    output = root / 'reports/09_mlp_runs' / os.environ['SLURM_JOB_ID']
    output.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(root / 'notebooks/10_mlp_model.ipynb', as_version=4)
    client = NotebookClient(notebook, timeout=None, kernel_name='python3',
                            resources={'metadata': {'path': str(root)}})
    try:
        client.execute()
    finally:
        # Preserve outputs and traceback even when a cell fails.
        destination = output / '09_MLP_model.executed.ipynb'
        nbformat.write(notebook, destination)
        print(f'Executed notebook saved to {destination}', flush=True)


if __name__ == '__main__':
    main()
