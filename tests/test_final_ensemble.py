"""Final protocol and portable packaging checks; no TensorFlow/models/test data run."""
import contextlib
import ast
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

from src import d1_final_ensemble as final
from src import final_ensemble_artifact as bundle
from src.d1_candidate_comparison import ComparisonConfig


class FinalProtocolTests(unittest.TestCase):
    def test_selected_estimator_sources_match_published_implementation(self):
        self.assertIn('features.py', final.verify_selected_sources(final.ROOT))

    def test_python312_empty_type_params_do_not_change_estimator_fingerprint(self):
        source = (final.ROOT/'src/d1_a2_mlp.py').read_text()
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'A2MLP')
        baseline = final.ast_fingerprint(node)
        changed = copy.deepcopy(node)
        for child in ast.walk(changed):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if 'type_params' not in child._fields:
                    child._fields = (*child._fields, 'type_params')
                child.type_params = []
        self.assertEqual(final.ast_fingerprint(changed), baseline)
        self.assertEqual(baseline, final.MLP_CLASS_SHA)

    def test_real_estimator_changes_still_change_fingerprint(self):
        source = (final.ROOT/'src/d1_a2_mlp.py').read_text()
        baseline = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'A2MLP')
        changed = next(n for n in ast.parse(source.replace('max_epochs=50', 'max_epochs=49')).body
                       if isinstance(n, ast.ClassDef) and n.name == 'A2MLP')
        self.assertNotEqual(final.ast_fingerprint(changed), final.ast_fingerprint(baseline))
        generic = copy.deepcopy(baseline)
        if 'type_params' not in generic._fields:
            generic._fields = (*generic._fields, 'type_params')
        generic.type_params = [ast.Name(id='T', ctx=ast.Load())]
        self.assertNotEqual(final.ast_fingerprint(generic), final.ast_fingerprint(baseline))

    def test_inner_predictions_never_receive_outer_or_test_labels(self):
        frame = pd.DataFrame({'job_id': np.arange(120), 'group_id': np.repeat(np.arange(60), 2),
                              'fraudulent': np.tile([0, 1], 60)})
        seen = []
        def stage(training, evaluation, config, path):
            self.assertNotIn('fraudulent', evaluation)
            self.assertTrue(set(training.group_id).isdisjoint(evaluation.group_id))
            seen.extend(evaluation.job_id.tolist())
            return np.full(len(evaluation), .4), {}
        with tempfile.TemporaryDirectory() as folder, patch.object(final, 'calibrated_stage', side_effect=stage), \
             patch.object(final, '_release'), contextlib.redirect_stdout(io.StringIO()):
            result = final.select_threshold(frame, ComparisonConfig(), Path(folder))
            self.assertTrue((Path(folder)/'threshold_oof.csv').exists())
        self.assertEqual(sorted(seen), list(frame.job_id))
        self.assertEqual(result['source'], 'development data only')

    def test_split_overlap_rejected_on_synthetic_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); p=root/'data/processed'; p.mkdir(parents=True)
            pd.DataFrame({'job_id':[1,2], 'group_id':[10,20], 'fraudulent':[0,1]}).to_csv(p/'d1_train.csv',index=False)
            pd.DataFrame({'job_id':[3,4], 'group_id':[10,30], 'fraudulent':[0,1]}).to_csv(p/'d1_test.csv',index=False)
            def checksum(path):return final.TRAIN_SHA if Path(path).name=='d1_train.csv' else final.TEST_SHA
            with patch.object(final,'file_sha',side_effect=checksum):
                with self.assertRaisesRegex(ValueError, 'overlap'):
                    final.load_fixed_split(root)

    def test_group_bootstrap_is_reproducible_and_handles_whole_groups(self):
        y=np.tile([0,1],8); p=np.tile([.1,.9],8); groups=np.repeat(np.arange(8),2)
        a=final.group_bootstrap(y,p,groups,.5,repeats=30,seed=2026)
        b=final.group_bootstrap(y,p,groups,.5,repeats=30,seed=2026)
        self.assertEqual(a,b)
        self.assertEqual(a['valid'],30)
        self.assertEqual(a['intervals']['pr_auc'],{'low':1.,'high':1.})

    def test_final_components_fit_only_core_groups_and_export_before_test(self):
        import sys
        from types import SimpleNamespace
        seen = []
        class FakeComponent:
            def fit(self, X, y):
                self.features_ = {'fixture': True}
                self.model_ = {'fixture': True}
                self.selected_epoch_ = 1
                self.adapter_ = SimpleNamespace(model=SimpleNamespace(input='input', output='output'))
                seen.append(set(X.group_id))
                return self
            def predict_proba(self, X):
                p = .2 + .6*(X.job_id.to_numpy()%2)
                return np.column_stack([1-p,p])
        class FakeFunctional:
            def __init__(self, *args): pass
            def save(self, path): Path(path).write_bytes(b'synthetic-network')
        def restored(artifact, probe):
            raw = .2 + .6*(probe.job_id.to_numpy()%2)
            p = np.zeros(len(probe))
            for name in ['mlp','trees']:
                calibrator = final.joblib.load(Path(artifact)/f'{name}_calibrator.joblib')
                p += .5*calibrator.predict_proba(final.logit(raw)[:,None])[:,1]
            return p
        train = pd.DataFrame({'job_id': np.arange(200), 'group_id': np.repeat(np.arange(100),2),
                              'fraudulent': np.tile([0,1],100)})
        fake_tf = SimpleNamespace(keras=SimpleNamespace(Model=FakeFunctional))
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(final,'_components',side_effect=lambda c: (FakeComponent(),FakeComponent())), \
             patch.object(final,'build_inputs',side_effect=lambda f: f.drop(columns='fraudulent')), \
             patch.object(final,'predict_artifact',side_effect=restored), patch.object(final,'_release'), \
             patch.object(final.importlib.metadata,'version',return_value='fixture'), \
             patch.dict(sys.modules,{'tensorflow':fake_tf}), contextlib.redirect_stdout(io.StringIO()):
            artifact=Path(folder)/'artifact'
            metadata=final.fit_export(train,ComparisonConfig(),artifact,.73)
            core=pd.read_csv(artifact/'core_ids.csv')
            cal=pd.read_csv(artifact/'calibration_ids.csv')
            self.assertTrue(set(core.job_id).isdisjoint(cal.job_id))
            self.assertEqual(len(core)+len(cal),len(train))
            self.assertEqual(seen[0],seen[1])
            cal_groups=set(train.loc[train.job_id.isin(cal.job_id),'group_id'])
            self.assertTrue(seen[0].isdisjoint(cal_groups))
            self.assertEqual(metadata['threshold'],.73)
            self.assertFalse(metadata['refit_after_calibration'])
            self.assertEqual(bundle.verify_artifact(artifact)['weights'],[.5,.5])

    def test_completed_result_is_read_without_refitting_or_reading_test(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder); (path/'test_metrics.json').write_text('{"fixture": true}')
            with patch.object(final,'load_fixed_split',side_effect=AssertionError('No test access')), \
                 patch.object(final,'fit_export',side_effect=AssertionError('No refit')), \
                 contextlib.redirect_stdout(io.StringIO()):
                final.run(path,path)


class ArtifactTests(unittest.TestCase):
    def create_artifact(self, path):
        path.mkdir()
        for name in bundle.CORE:(path/name).write_bytes(('synthetic file '+name).encode())
        metadata={'model':'calibrated_mlp_extra_trees','weights':[.5,.5],
                  'file_hashes':{name:bundle.digest(path/name) for name in bundle.CORE}}
        (path/'metadata.json').write_text(json.dumps(metadata))
        (path/'runtime_requirements.txt').write_text('fixture==1\n')
        (path/'export_verified.json').write_text('{"development_probe_rows": 32, "test_used": false}')
        (path/'core_ids.csv').write_text('private training IDs must stay out of release')

    def test_package_excludes_data_and_roundtrips_with_checksum(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); artifact=root/'artifact'; self.create_artifact(artifact)
            checksum=bundle.package_artifact(artifact,root/'model.zip')
            with zipfile.ZipFile(root/'model.zip') as archive:
                self.assertEqual(set(archive.namelist()),bundle.FILES)
                self.assertNotIn('core_ids.csv',archive.namelist())
            restored=bundle.unpack_artifact(root/'model.zip',root/'restored',checksum)
            self.assertEqual(bundle.verify_artifact(restored)['weights'],[.5,.5])
            self.assertTrue((root/'model.sha256').exists())
            self.assertEqual(checksum,bundle.package_artifact(artifact,root/'model.zip'))

    def test_wrong_download_checksum_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); artifact=root/'artifact'; self.create_artifact(artifact)
            bundle.package_artifact(artifact,root/'model.zip')
            with self.assertRaisesRegex(ValueError,'SHA256'):
                bundle.unpack_artifact(root/'model.zip',root/'restored','0'*64)
            self.assertFalse((root/'restored').exists())

    def test_tampered_model_and_unsafe_zip_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); artifact=root/'artifact'; self.create_artifact(artifact)
            (artifact/'mlp.keras').write_text('changed')
            with self.assertRaisesRegex(ValueError,'checksums'):
                bundle.package_artifact(artifact,root/'model.zip')
            with zipfile.ZipFile(root/'unsafe.zip','w') as archive:
                archive.writestr('../outside.txt','bad')
            with self.assertRaisesRegex(ValueError,'ZIP contents'):
                bundle.unpack_artifact(root/'unsafe.zip',root/'restored',bundle.digest(root/'unsafe.zip'))
            self.assertFalse((root/'outside.txt').exists())


if __name__=='__main__':unittest.main()
