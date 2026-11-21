"""Synthetic sampling, process deadline and accounting tests; no external calls."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from evaluator import balanced
from evaluator.schemas import Example, DIMENSIONS
from agent.core.schemas import Posting


def records(n=60):
    return [dict(job_id=f'{label}-{i}', group_id=f'g-{label}-{i}', fraudulent=label,
                 title='Example role', description='Ordinary example posting text.',
                 company_profile='', hidden_target_feature='do not forward')
            for label in (0, 1) for i in range(n)]


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def example():
    return Example(case_id='case_1', fraudulent=1,
                   posting=Posting(posting_id='case_1', title='Example', description='A test posting.'))


class SamplingTests(unittest.TestCase):
    def test_balanced_reproducible_distinct_groups(self):
        source = records()
        a = balanced.balanced_sample(source)
        self.assertEqual(a, balanced.balanced_sample(source))
        self.assertEqual(len(a), 100)
        self.assertEqual(sum(r['fraudulent'] for r in a), 50)
        self.assertEqual(len({r['group_id'] for r in a}), 100)
        self.assertTrue(all(a[i]['fraudulent'] != a[i+1]['fraudulent'] for i in range(99)))

    def test_mixed_groups_cannot_exhaust_other_label(self):
        # Genuine exists only in shared groups; fraud can use exclusive groups.
        rows = [dict(job_id=f'{label}_{i}', group_id=f'shared_{i}', fraudulent=label)
                for label in (0, 1) for i in range(10)]
        rows += [dict(job_id=f'fraud_{i}', group_id=f'exclusive_{i}', fraudulent=1) for i in range(10)]
        for seed in range(30):
            selected = balanced.balanced_sample(rows, per_label=10, seed=seed)
            self.assertEqual(len({r['group_id'] for r in selected}), 20)
            self.assertEqual(sum(r['fraudulent'] for r in selected), 10)

    def test_duplicate_missing_and_insufficient_groups_rejected(self):
        for source in [records() + records()[:1], records()[:14]]:
            with self.assertRaises(ValueError):
                balanced.balanced_sample(source)
        source = records(); source[0]['group_id'] = None
        with self.assertRaises(ValueError):
            balanced.balanced_sample(source)

    def test_prepare_locks_sample_and_manual_audit_without_engineered_targets(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root/'development.csv'
            write_csv(source, records())
            manifest = balanced.prepare(source, root/'locked')
            sample = (root/'locked/sample.jsonl').read_text()
            self.assertNotIn('hidden_target_feature', sample)
            self.assertEqual(manifest['manual_review_status'], 'pending')
            self.assertEqual(len(manifest['manual_review_case_ids']), 20)
            mapping = {r['case_id']: r for r in manifest['cases']}
            self.assertEqual(sum(mapping[k]['fraudulent'] for k in manifest['manual_review_case_ids']), 10)
            with self.assertRaises(FileExistsError):
                balanced.prepare(source, root/'locked')

    def test_reserved_test_requires_freeze_and_group_isolation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root/'d1_test.csv'; train = root/'train.csv'
            write_csv(source, records()); write_csv(train, records())
            with self.assertRaisesRegex(ValueError, 'Reserved test'):
                balanced.prepare(source, root/'a')
            with self.assertRaisesRegex(ValueError, 'frozen-version'):
                balanced.prepare(source, root/'b', split='heldout')
            with self.assertRaisesRegex(ValueError, 'share groups'):
                balanced.prepare(source, root/'c', split='heldout', frozen_version='frozen-v1', training_data=train)
            train_rows = records()
            for row in train_rows:
                row['group_id'] = 'training-' + row['group_id']
            write_csv(train, train_rows)
            result = balanced.prepare(source, root/'d', split='heldout', frozen_version='frozen-v1', training_data=train)
            self.assertEqual(result['split'], 'heldout')
            self.assertIsNotNone(result['training_sha256'])


class BoundedRunTests(unittest.TestCase):
    def test_whole_worker_timeout_keeps_denominator(self):
        with tempfile.TemporaryDirectory() as folder:
            row = balanced.execute_case(example(), Path(folder)/'case', .15,
                command=[sys.executable, '-c', 'import time; time.sleep(5)'])
        self.assertEqual(row['evaluation_status'], 'timeout')
        self.assertEqual(row['agent_status'], 'failed')
        self.assertLess(row['elapsed_seconds'], 4)
        q = balanced.quality_summary([row])
        self.assertEqual(q['planned_cases'], 1)
        self.assertEqual(q['timeout_cases'], 1)
        self.assertEqual(q['judge_coverage'], 0)
        self.assertTrue(all(v is None for v in q['mean_effective_scores'].values()))

    def test_judge_timeout_preserves_agent_and_classifier_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/'case'
            checkpoint = balanced.empty_row(example())
            checkpoint.update(agent_status='completed', case={'fixture': True}, classifier_prediction=True)
            script = ('import pathlib,time; '
                      f'pathlib.Path({str(target / "agent_checkpoint.json")!r}).write_text({json.dumps(checkpoint)!r}); '
                      'time.sleep(5)')
            row = balanced.execute_case(example(), target, .5, command=[sys.executable, '-c', script])
        self.assertEqual(row['evaluation_status'], 'timeout')
        self.assertEqual(row['timeout_stage'], 'judge')
        self.assertEqual(row['agent_status'], 'completed')
        self.assertTrue(row['classifier_prediction'])
        self.assertEqual(row['judge_status'], 'failed')

    def test_successful_worker_read(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/'case'
            result = balanced.empty_row(example()); result['agent_status'] = 'fallback'
            script = f'import pathlib; pathlib.Path({str(target / "result.json")!r}).write_text({json.dumps(result)!r})'
            row = balanced.execute_case(example(), target, 3, command=[sys.executable, '-c', script])
        self.assertEqual(row['evaluation_status'], 'finished')
        self.assertEqual(row['agent_status'], 'fallback')

    def test_quality_means_and_failure_coverage_are_separate(self):
        complete = balanced.empty_row(example())
        complete.update(agent_status='completed', judge_status='completed',
            scoring={'effective_scores': {d: 4 for d in DIMENSIONS}},
            judgment={'unsupported_claims': [{'quote': 'fixture'}]}, elapsed_seconds=2)
        timeout = balanced.empty_row(example())
        timeout.update(evaluation_status='timeout', agent_status='failed', elapsed_seconds=3)
        q = balanced.quality_summary([complete, timeout])
        self.assertEqual(q['judge_coverage'], .5)
        self.assertEqual(q['agent_completion_rate'], .5)
        self.assertEqual(q['mean_effective_scores']['evidence_grounding'], 4)
        self.assertEqual(q['scored_count_by_dimension']['evidence_grounding'], 1)
        self.assertEqual(q['unsupported_claim_case_denominator'], 1)
        self.assertEqual(q['cases_with_judge_reported_unsupported_claims'], 1)

    def test_budget_exhaustion_exports_all_hundred_without_any_agent_call(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root/'development.csv'; locked=root/'sample'
            write_csv(source, records()); balanced.prepare(source, locked)
            runtime = dict(judge={'model': 'fixture', 'backend': 'fixture'}, rubric_version='fixture', rubric_sha256='fixture')
            with patch.object(balanced, 'runtime_manifest', return_value=runtime), \
                 patch.object(balanced, 'execute_case', side_effect=AssertionError('Must not run')):
                report = balanced.run(locked, root/'run', max_seconds=.001)
            q = report['quality_summary']['overall']
            self.assertEqual(q['planned_cases'], 100)
            self.assertEqual(q['not_started_cases'], 100)
            self.assertEqual(q['judge_coverage'], 0)
            self.assertTrue((root/'run/review_report.html').is_file())
            self.assertEqual(len((root/'run/results.jsonl').read_text().splitlines()), 100)
            self.assertTrue(report['final_report'])

    def test_manifest_cannot_silently_reduce_the_hundred_case_design(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root/'development.csv'; locked = root/'sample'
            write_csv(source, records()); manifest = balanced.prepare(source, locked)
            self.assertEqual(manifest['protocol'], 'balanced-agent-quality-v2')
            manifest['selected_cases'] = 20
            balanced.atomic_json(locked/'sample_manifest.json', manifest)
            with patch.object(balanced, 'runtime_manifest', side_effect=AssertionError('No runtime access')):
                with self.assertRaisesRegex(ValueError, 'matching case counts'):
                    balanced.run(locked, root/'run')

    def test_historical_twenty_case_sample_remains_readable_without_expansion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root/'development.csv'; locked = root/'sample'
            write_csv(source, records()); manifest = balanced.prepare(source, locked)
            selected = [json.loads(line) for line in (locked/'sample.jsonl').read_text().splitlines()][:20]
            (locked/'sample.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in selected))
            manifest.update(protocol='balanced-agent-quality-v1', per_label=10,
                            selected_cases=20, distinct_groups=20, cases=manifest['cases'][:20],
                            sample_sha256=balanced.sha256(locked/'sample.jsonl'))
            manifest['manual_review_case_ids'] = [row['job_id'] for row in selected[:4]]
            balanced.atomic_json(locked/'sample_manifest.json', manifest)
            runtime = dict(judge={}, rubric_version='fixture', rubric_sha256='fixture')
            with patch.object(balanced, 'runtime_manifest', return_value=runtime), \
                 patch.object(balanced, 'execute_case', side_effect=AssertionError('Must not run')):
                report = balanced.run(locked, root/'run', max_seconds=.001)
            self.assertEqual(report['selected_cases'], 20)
            self.assertEqual(report['quality_summary']['overall']['planned_cases'], 20)

    def test_changed_sample_is_rejected_before_runtime_or_model_access(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); source=root/'development.csv'; locked=root/'sample'
            write_csv(source, records()); balanced.prepare(source, locked)
            with (locked/'sample.jsonl').open('a') as stream:
                stream.write('{}\n')
            with patch.object(balanced, 'runtime_manifest', side_effect=AssertionError('No runtime access')):
                with self.assertRaisesRegex(ValueError, 'fingerprint'):
                    balanced.run(locked, root/'run')


class ArtifactProvenanceTests(unittest.TestCase):
    def test_heldout_reference_must_match_frozen_classifier_training_file(self):
        with tempfile.TemporaryDirectory() as folder:
            artifact = Path(folder)
            (artifact/'pipeline.joblib').write_bytes(b'synthetic-artifact-not-a-real-model')
            metadata = dict(artifact_sha256=balanced.sha256(artifact/'pipeline.joblib'),
                training_sha256='training-file-a', features_sha256=balanced.sha256(balanced.ROOT/'src/features.py'))
            balanced.atomic_json(artifact/'metadata.json', metadata)
            with self.assertRaisesRegex(ValueError, 'training file differs'):
                balanced.artifact_provenance(artifact, {'split': 'heldout', 'training_sha256': 'training-file-b'})
            result = balanced.artifact_provenance(artifact, {'split': 'heldout', 'training_sha256': 'training-file-a'})
            self.assertEqual(result['training_sha256'], 'training-file-a')
            (artifact/'pipeline.joblib').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum differs'):
                balanced.artifact_provenance(artifact, {'split': 'development'})


class AgentCheckpointTests(unittest.TestCase):
    def test_agent_checkpoint_precedes_judge_without_annotations_in_posting(self):
        import importlib.util
        import types
        from types import SimpleNamespace
        # Replace only model adapters. Execute the actual runner without loading
        # LangChain or making external model calls in this CPU-only environment.
        actor_module = types.ModuleType('agent.core.runner')
        actor_module.run_review = lambda *args: None
        judge_module = types.ModuleType('evaluator.judge')
        judge_module.judge_case = lambda *args: None
        judge_module.validate_judgment = lambda *args: None
        judge_module.judge_error_code = lambda exc: 'synthetic_judge_failure'
        judge_module.RUBRIC_VERSION = 'fixture'
        judge_module.SYSTEM_PROMPT = 'fixture'
        checks_module = types.ModuleType('evaluator.checks')
        checks_module.code_checks = lambda *args: {}
        spec = importlib.util.spec_from_file_location('bounded_runner_fixture', balanced.ROOT/'evaluator/runner.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'agent.core.runner': actor_module, 'evaluator.judge': judge_module, 'evaluator.checks': checks_module}):
            spec.loader.exec_module(module)
        ex = example()
        case = SimpleNamespace(status='completed', error_code=None,
            assessment=SimpleNamespace(recommendation='review_needed'), tool_trace=[],
            model_dump=lambda **kwargs: dict(model_backend='fixture', model_name='fixture'))
        checkpoints = []

        def judge(posting, actual_case, settings):
            self.assertEqual(len(checkpoints), 1)
            self.assertNotIn('fraudulent', posting.model_dump())
            raise ValueError('Synthetic judge failure')

        module.code_checks = lambda *args: dict(all_passed=True, expected_action_match=None,
                                                 checks={'context_review_completed': True})
        settings = SimpleNamespace(backend='fixture', model='fixture')
        report = module.run_evaluation([ex], settings, settings, agent_fn=lambda *args: case,
            judge_fn=judge, on_agent_result=checkpoints.append)
        self.assertEqual(checkpoints[0]['agent_status'], 'completed')
        self.assertEqual(checkpoints[0]['judge_status'], 'skipped')
        self.assertEqual(report['rows'][0]['judge_status'], 'failed')


if __name__ == '__main__':
    unittest.main()
