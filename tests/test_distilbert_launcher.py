"""Cluster launch preparation only: no notebook execution or ML imports."""
import ast
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('distilbert_launcher', ROOT / 'cluster/run_distilbert_notebook.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class DistilbertLaunchTests(unittest.TestCase):
    def setUp(self):
        self.original = json.loads((ROOT / 'notebooks/12_distilbert_finetuning.ipynb').read_text())

    def test_copy_enforces_single_cuda_fold_and_safe_flags(self):
        original_text = json.dumps(self.original)
        prepared = launcher.prepare_notebook(self.original)
        assignments = {}
        device = None
        for cell in prepared['cells']:
            if cell['cell_type'] != 'code':
                continue
            self.assertEqual(cell['outputs'], [])
            self.assertIsNone(cell['execution_count'])
            for node in ast.parse(cell['source']).body:
                if not isinstance(node, ast.Assign) or not isinstance(node.targets[0], ast.Name):
                    continue
                name = node.targets[0].id
                if name in launcher.FLAGS:
                    assignments[name] = ast.literal_eval(node.value)
                if name == 'CONFIG':
                    device = ast.literal_eval(next(k.value for k in node.value.keywords if k.arg == 'device'))
        self.assertEqual(assignments, launcher.FLAGS)
        self.assertEqual(device, 'cuda')
        self.assertEqual(json.dumps(self.original), original_text)

    def test_refuses_missing_training_flag(self):
        for cell in self.original['cells']:
            if cell['cell_type'] == 'code':
                cell['source'] = ''.join(cell['source']).replace('TRAIN_MODEL = False', 'DIFFERENT_FLAG = False')
        with self.assertRaisesRegex(ValueError, 'Unexpected notebook settings'):
            launcher.prepare_notebook(self.original)

    def test_refuses_duplicate_limit(self):
        self.original['cells'].append({'cell_type': 'code', 'source': 'OUTER_LIMIT = None',
                                       'outputs': [], 'execution_count': None})
        with self.assertRaisesRegex(ValueError, 'Unexpected notebook settings'):
            launcher.prepare_notebook(self.original)

    def test_refuses_changed_config_api(self):
        for cell in self.original['cells']:
            if cell['cell_type'] == 'code':
                cell['source'] = ''.join(cell['source']).replace('experiment.Config(', 'experiment.OtherConfig(')
        with self.assertRaisesRegex(ValueError, 'CONFIG layout changed'):
            launcher.prepare_notebook(self.original)


if __name__ == '__main__':
    unittest.main()
