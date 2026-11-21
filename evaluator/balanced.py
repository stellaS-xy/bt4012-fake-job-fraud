"""Prepare a locked 50+50 sample and run a time-bounded agent-quality audit.

No default dataset or automatic test access. Imports never invoke any model.
The parent process enforces whole-case deadlines on isolated worker groups.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TEXT_FIELDS = ('title', 'company_profile', 'description', 'requirements', 'benefits')
WARNING = ('Balanced, group-distinct quality audit, not a prevalence-representative accuracy test. '
           'Judge scores are automated opinions. Review-needed is not a fraud label. '
           'Failures and timeouts remain in the planned denominator.')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def balanced_sample(records, per_label=50, seed=2026):
    """Random groups, then random posting; no group repeats, including mixed groups.

    Each draw excludes choices that would make the other label quota impossible.
    Thus mixed-label groups do not cause a silent redraw, omission or duplicate.
    """
    if per_label < 1:
        raise ValueError('per_label must be positive.')
    pools, seen = {0: {}, 1: {}}, set()
    for raw in records:
        record = dict(raw)
        job, group = (str(record.get(k) if record.get(k) is not None else '').strip()
                      for k in ('job_id', 'group_id'))
        label = str(record.get('fraudulent', '')).strip()
        if not job or not group or job in seen or label not in {'0', '1'}:
            raise ValueError('Source requires unique job_id, nonmissing group_id and binary fraudulent labels.')
        seen.add(job)
        record.update(job_id=job, group_id=group, fraudulent=int(label))
        pools[int(label)].setdefault(group, []).append(record)
    groups = {label: set(pool) for label, pool in pools.items()}
    if (any(len(groups[label]) < per_label for label in (0, 1))
            or len(groups[0] | groups[1]) < 2 * per_label):
        raise ValueError('Insufficient distinct groups for the requested balanced sample.')
    rng = random.Random(seed)
    needed, used, selected = {0: per_label, 1: per_label}, set(), []
    first = rng.randrange(2)
    for index in range(2 * per_label):
        label = (first + index) % 2  # interleaved labels prevent a class-first time-budget bias
        other = 1 - label
        other_available = groups[other] - used
        candidates = [g for g in sorted(groups[label] - used)
                      if len(other_available) - int(g in other_available) >= needed[other]]
        if not candidates:
            raise ValueError('No feasible group allocation; no replacement sample was generated.')
        group = rng.choice(candidates)
        record = dict(rng.choice(sorted(pools[label][group], key=lambda r: r['job_id'])))
        selected.append(record)
        used.add(group)
        needed[label] -= 1
    return selected


def prepare(data, output, *, split='development', frozen_version=None, training_data=None):
    """Explicit user action only; tests call this with temporary synthetic CSVs."""
    data, output = Path(data), Path(output)
    if split not in {'development', 'heldout'}:
        raise ValueError('Select development or heldout.')
    if data.name in {'d1_test.csv', 'd1_test_feature_engineered.csv'} and split != 'heldout':
        raise ValueError('Reserved test preparation requires heldout mode and a frozen version.')
    if split == 'heldout' and (not frozen_version or not str(frozen_version).strip() or training_data is None):
        raise ValueError('Heldout sampling requires --frozen-version and --training-data for group-disjointness checks.')
    if output.exists():
        raise FileExistsError('Sample destination already exists. Locked samples are never overwritten.')
    with data.open(encoding='utf-8-sig', newline='') as stream:
        records = list(csv.DictReader(stream))
    selected = balanced_sample(records)
    training_sha = None
    if split == 'heldout':
        with Path(training_data).open(encoding='utf-8-sig', newline='') as stream:
            training_groups = {str(row.get('group_id', '')).strip() for row in csv.DictReader(stream)}
        if '' in training_groups or any(not str(r.get('group_id', '')).strip() for r in records):
            raise ValueError('Missing group identifiers prevent a heldout overlap check.')
        if training_groups.intersection(str(row['group_id']).strip() for row in records):
            raise ValueError('Heldout source and training data share groups.')
        training_sha = sha256(training_data)
    # Retain only supported posting fields plus audit metadata, not engineered targets.
    from evaluator.data import POSTING_FIELDS
    allowed = {'job_id', 'group_id', 'fraudulent', *POSTING_FIELDS}
    selected = [{k: v for k, v in row.items() if k in allowed} for row in selected]
    audit_rng = random.Random(2027)
    manual_ids = []
    for label in (0, 1):
        manual_ids.extend(audit_rng.sample([r['job_id'] for r in selected if r['fraudulent'] == label], 10))
    output.mkdir(parents=True)
    sample = output / 'sample.jsonl'
    sample.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected), encoding='utf-8')
    metadata = []
    for row in selected:
        words = sum(len(str(row.get(f) or '').split()) for f in TEXT_FIELDS)
        metadata.append({'case_id': row['job_id'], 'group_id': row['group_id'],
                         'fraudulent': row['fraudulent'], 'text_words': words,
                         'manual_review_selected': row['job_id'] in manual_ids})
    manifest = dict(protocol='balanced-agent-quality-v2', created_at=datetime.now(timezone.utc).isoformat(),
        source=str(data.resolve()), source_sha256=sha256(data), sample_sha256=sha256(sample),
        training_sha256=training_sha, split=split, frozen_version=frozen_version,
        seed=2026, per_label=50, selected_cases=100, distinct_groups=100,
        sampling_unit='One group per draw, then one random posting of the requested label; mixed-label group feasibility constrained.',
        run_order='Alternating labels, random starting label.', length_policy='No length filtering or forced length quotas.',
        manual_review_seed=2027, manual_review_case_ids=manual_ids,
        manual_review_status='pending', source_records=len(records), cases=metadata,
        limitations=[WARNING, '100 cases, one execution per case; no population-level or statistical-superiority claim.'])
    atomic_json(output / 'sample_manifest.json', manifest)
    with (output / 'manual_review.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['case_id', 'group_id', 'fraudulent', 'quote_support',
            'interpretation_correct', 'judge_agreement', 'notes', 'review_status'])
        writer.writeheader()
        for row in metadata:
            if row['manual_review_selected']:
                writer.writerow({k: v for k, v in row.items() if k in {'case_id', 'group_id', 'fraudulent'}} | {'review_status': 'pending'})
    return manifest


def empty_row(example):
    return dict(case_id=example.case_id, repeat=0, fraudulent=example.fraudulent,
        expected_action=example.expected_action, input_error=example.input_error,
        agent_status='not_started', agent_error=None, code=None, judge_status='skipped',
        judge_error=None, judgment=None, recommendation=None, scoring=None,
        classifier_prediction=None, case=None, evaluation_status='pending', elapsed_seconds=None)


def quality_summary(rows):
    from evaluator.schemas import DIMENSIONS
    n = len(rows)
    scored = [r for r in rows if r.get('judge_status') == 'completed' and r.get('scoring')]
    judged = [r for r in rows if r.get('judge_status') == 'completed']
    scores = {d: [r['scoring']['effective_scores'][d] for r in scored
                  if r['scoring']['effective_scores'].get(d) is not None] for d in DIMENSIONS}
    times = [r['elapsed_seconds'] for r in rows if r.get('elapsed_seconds') is not None]
    return dict(planned_cases=n,
        agent_completed=sum(r.get('agent_status') == 'completed' for r in rows),
        agent_completion_rate=sum(r.get('agent_status') == 'completed' for r in rows) / n if n else None,
        code_checks_passed=sum(bool(r.get('code') and r['code']['all_passed']) for r in rows),
        code_pass_rate=sum(bool(r.get('code') and r['code']['all_passed']) for r in rows) / n if n else None,
        judge_completed=len(judged), judge_coverage=len(judged)/n if n else None,
        fallback_cases=sum(r.get('agent_status') == 'fallback' for r in rows),
        timeout_cases=sum(r.get('evaluation_status') == 'timeout' for r in rows),
        not_started_cases=sum(r.get('agent_status') == 'not_started' for r in rows),
        input_invalid_cases=sum(r.get('agent_status') == 'input_invalid' for r in rows),
        mean_effective_scores={d: statistics.mean(v) if v else None for d, v in scores.items()},
        scored_count_by_dimension={d: len(v) for d, v in scores.items()},
        cases_with_judge_reported_unsupported_claims=sum(bool((r.get('judgment') or {}).get('unsupported_claims')) for r in judged),
        unsupported_claim_case_denominator=len(judged),
        median_seconds=statistics.median(times) if times else None, max_seconds=max(times) if times else None)


def terminate_group(process):
    """Kill only this worker's new session, including its classifier subprocess."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=2)


def execute_case(example, folder, timeout, *, command=None, stop_requested=lambda: False):
    """Child wall time includes imports, agent, context calls, judge and corrections."""
    if os.name != 'posix':
        raise RuntimeError('The bounded worker launcher requires a POSIX cluster/Linux or macOS host.')
    folder = Path(folder).resolve()
    folder.mkdir()
    result_path, stage_path = folder / 'result.json', folder / 'agent_checkpoint.json'
    command = command or [sys.executable, '-m', 'evaluator.balanced', 'worker',
                          '--result', str(result_path), '--stage', str(stage_path)]
    started = time.monotonic()
    row = empty_row(example)
    input_path = folder / 'input.json'
    input_path.write_text(example.model_dump_json(), encoding='utf-8')
    with input_path.open(encoding='utf-8') as incoming, (folder / 'worker.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdin=incoming, stdout=log, stderr=subprocess.STDOUT,
                                   text=True, cwd=ROOT, start_new_session=True)
        try:
            while process.poll() is None:
                if stop_requested() or time.monotonic() - started >= timeout:
                    terminate_group(process)
                    if stage_path.is_file():
                        row = json.loads(stage_path.read_text())
                    row['evaluation_status'] = 'timeout'
                    row['timeout_stage'] = 'judge' if row.get('case') else 'agent_or_worker_startup'
                    if row.get('case'):
                        row.update(judge_status='failed', judge_error='case_deadline_exceeded')
                    else:
                        row.update(agent_status='failed', agent_error='case_deadline_exceeded')
                    break
                time.sleep(.05)
            else:
                if process.returncode == 0 and result_path.is_file():
                    row = json.loads(result_path.read_text())
                    row['evaluation_status'] = 'finished'
                else:
                    if stage_path.is_file():
                        row = json.loads(stage_path.read_text())
                    row['evaluation_status'] = 'worker_failed'
                    if row.get('case'):
                        row.update(judge_status='failed', judge_error='worker_failed')
                    else:
                        row.update(agent_status='failed', agent_error='worker_failed')
        finally:
            terminate_group(process)
    if row.get('case_id') != example.case_id:
        raise ValueError('Worker returned the wrong case identity.')
    row['elapsed_seconds'] = time.monotonic() - started
    return row


def worker(result_path, stage_path):
    from agent.core.config import load_settings
    from evaluator.config import load_judge_settings
    from evaluator.runner import run_evaluation
    from evaluator.schemas import Example
    example = Example.model_validate(json.loads(sys.stdin.read()))
    report = run_evaluation([example], load_settings(), load_judge_settings(),
        progress=lambda text: print(text, flush=True),
        on_agent_result=lambda row: atomic_json(stage_path, row))
    atomic_json(result_path, report['rows'][0])


def artifact_provenance(artifact, sample_manifest):
    """Validate the frozen binary and, for heldout runs, its training-file identity."""
    artifact = Path(artifact)
    metadata_sha = sha256(artifact/'metadata.json')
    metadata = json.loads((artifact/'metadata.json').read_text(encoding='utf-8'))
    if metadata.get('model') == 'calibrated_mlp_extra_trees':
        from src.final_ensemble_artifact import verify_artifact
        verify_artifact(artifact)
        binaries = dict(file_hashes=metadata['file_hashes'], threshold=metadata['threshold'],
                        model_version='calibrated-mlp-extra-trees-v1')
        features_sha = metadata.get('features_source_sha256')
    else:
        pipeline_sha = sha256(artifact/'pipeline.joblib')
        if pipeline_sha != metadata.get('artifact_sha256'):
            raise ValueError('Classifier artifact checksum differs from its metadata.')
        binaries = dict(pipeline_sha256=pipeline_sha)
        features_sha = metadata.get('features_sha256')
    if (sample_manifest['split'] == 'heldout'
            and metadata.get('training_sha256') != sample_manifest.get('training_sha256')):
        raise ValueError('Classifier training file differs from the heldout group-overlap reference.')
    if features_sha != sha256(ROOT/'src/features.py'):
        raise ValueError('Classifier preprocessing source changed; freeze/fit a consistent artifact first.')
    return dict(artifact=str(artifact), **binaries, metadata_sha256=metadata_sha,
                training_sha256=metadata.get('training_sha256'))


def runtime_manifest(sample_manifest):
    """Export allowlisted provenance only: never serialise settings/API keys."""
    import importlib.metadata
    from agent.core.config import load_settings
    from evaluator.config import load_judge_settings
    from evaluator.judge import RUBRIC_VERSION, SYSTEM_PROMPT
    actor, judge = load_settings(), load_judge_settings()
    artifact = actor.classifier_artifact
    paths = sorted({*ROOT.glob('agent/core/*.py'), *ROOT.glob('agent/tools/*.py'), *ROOT.glob('evaluator/*.py'),
                    ROOT/'src/ensemble_inference.py', ROOT/'src/d1_final_ensemble.py',
                    ROOT/'src/final_ensemble_artifact.py', ROOT/'src/features.py'})
    return dict(protocol=sample_manifest['protocol'], sample_sha256=sample_manifest['sample_sha256'],
        actor=dict(backend=actor.backend, model=actor.model, timeout=actor.request_timeout),
        judge=dict(backend=judge.backend, model=judge.model, timeout=judge.request_timeout,
                   same_model_as_agent=(actor.backend, actor.model) == (judge.backend, judge.model)),
        classifier=dict(backend=actor.classifier_backend, **artifact_provenance(artifact, sample_manifest)),
        rubric_version=RUBRIC_VERSION, rubric_sha256=hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in paths},
        versions={name: importlib.metadata.version(name) for name in
                  ['langchain', 'langchain-core', 'langchain-openai', 'pydantic', 'openai']})


def run(sample_dir, output_dir, *, max_seconds=28200, case_seconds=240):
    from evaluator.data import load_examples
    from evaluator.report import summarize
    from evaluator.export_html import render
    sample_dir, output = Path(sample_dir).resolve(), Path(output_dir).resolve()
    if not (math.isfinite(max_seconds) and math.isfinite(case_seconds)
            and 0 < max_seconds <= 28500 and 0 < case_seconds <= 240):
        raise ValueError('Use a total evaluation budget <=28500 seconds and whole-case budget <=240 seconds.')
    started = time.monotonic()
    manifest = json.loads((sample_dir/'sample_manifest.json').read_text())
    sample = sample_dir/'sample.jsonl'
    if sha256(sample) != manifest['sample_sha256']:
        raise ValueError('Locked sample fingerprint differs; no evaluation started.')
    examples = load_examples(sample)
    metadata = manifest['cases']
    raw_records = [json.loads(line) for line in sample.read_text(encoding='utf-8').splitlines() if line.strip()]
    if [str(r.get('group_id')) for r in raw_records] != [str(m['group_id']) for m in metadata]:
        raise ValueError('Sample groups differ from locked metadata.')
    # Existing locked v1 samples remain readable, but new preparation always uses v2.
    per_label = {'balanced-agent-quality-v1': 10, 'balanced-agent-quality-v2': 50}.get(manifest.get('protocol'))
    total = 2 * per_label if per_label else 0
    if (not total or len(examples) != total or len(metadata) != total
            or manifest.get('selected_cases') != total or manifest.get('per_label') != per_label
            or manifest.get('distinct_groups') != total
            or len({m['group_id'] for m in metadata}) != total
            or len({e.case_id for e in examples}) != total
            or [e.case_id for e in examples] != [m['case_id'] for m in metadata]
            or any(e.fraudulent != m['fraudulent'] for e, m in zip(examples, metadata))
            or sorted(e.fraudulent for e in examples) != [0]*per_label + [1]*per_label):
        raise ValueError('Expected the unchanged balanced, group-distinct protocol and matching case counts.')
    if manifest['split'] == 'heldout' and not manifest.get('frozen_version'):
        raise ValueError('Missing heldout frozen-version declaration.')
    if output.exists():
        raise FileExistsError('Run directory exists; previous results must not be overwritten.')
    runtime = runtime_manifest(manifest)  # verifies real classifier artifact before any cases
    output.mkdir(parents=True)
    atomic_json(output/'runtime_manifest.json', runtime)
    atomic_json(output/'sample_manifest.json', manifest)
    rows = [empty_row(e) | {'group_id': m['group_id']} for e, m in zip(examples, metadata)]
    stopped = [False]
    previous = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.signal(signum, lambda *_: stopped.__setitem__(0, True))

    def save_report(final=False):
        summaries = {'overall': quality_summary(rows),
                     'genuine': quality_summary([r for r in rows if r['fraudulent'] == 0]),
                     'fraudulent': quality_summary([r for r in rows if r['fraudulent'] == 1])}
        report = dict(schema_version=2, created_at=datetime.now(timezone.utc).isoformat(),
            source=str(sample.resolve()), split=manifest['split'], frozen_version=manifest.get('frozen_version'),
            protocol=manifest['protocol'], input_sha256=manifest['sample_sha256'],
            judge=runtime['judge'], rubric_version=runtime['rubric_version'],
            rubric_sha256=runtime['rubric_sha256'], repeats=1, selected_cases=total,
            limitations=[WARNING, f'A {total}-case audit cannot establish production reliability.',
                         'The same actor/judge model may reproduce shared errors.',
                         'Quality means exclude unavailable ratings; read coverage and failure counts.'],
            summary=summarize(rows), quality_summary=summaries, rows=rows,
            elapsed_seconds=time.monotonic()-started, final_report=final,
            budget=dict(max_seconds=max_seconds, case_seconds=case_seconds),
            manual_review_case_ids=manifest['manual_review_case_ids'], manual_review_status='pending')
        atomic_json(output/'report.json', report)
        atomic_json(output/'quality_summary.json', summaries)
        with (output/'quality_summary.csv').open('w', newline='', encoding='utf-8') as stream:
            flat = []
            for name, summary in summaries.items():
                record = {'subset': name} | {k: v for k, v in summary.items() if not isinstance(v, dict)}
                for d, value in summary['mean_effective_scores'].items():
                    record[d+'_mean'] = value
                    record[d+'_scored_count'] = summary['scored_count_by_dimension'][d]
                flat.append(record)
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader(); writer.writerows(flat)
        warning = '<section class="panel"><h2>Balanced quality audit</h2><p>'+html.escape(WARNING)+'</p><pre>'+html.escape(json.dumps(summaries, indent=2))+'</pre></section>'
        page = render(report).replace('</header>', '</header>'+warning, 1)
        temporary = output/'review_report.html.tmp'
        temporary.write_text(page, encoding='utf-8')
        temporary.replace(output/'review_report.html')
        return report

    try:
        save_report()
        with (output/'results.jsonl').open('w', encoding='utf-8') as results, (output/'cases.jsonl').open('w', encoding='utf-8') as cases:
            for index, example in enumerate(examples):
                remaining = max_seconds - (time.monotonic()-started)
                if stopped[0] or remaining < case_seconds + 5:
                    rows[index].update(evaluation_status='budget_not_started', agent_error='run_budget_or_signal')
                else:
                    print(f'Case {index+1}/{total}: {example.case_id}; whole-case limit {case_seconds:g}s', flush=True)
                    case_started = time.monotonic()
                    try:
                        row = execute_case(example, output/f'case_{index+1:02d}', case_seconds,
                                           stop_requested=lambda: stopped[0])
                    except Exception as exc:
                        # Preserve the selected case even if launching/reading its worker fails.
                        # Record only the exception type, never provider messages or credentials.
                        row = empty_row(example) | dict(agent_status='failed', agent_error='worker_failed',
                            evaluation_status='worker_failed', worker_error_type=type(exc).__name__,
                            elapsed_seconds=time.monotonic()-case_started)
                    rows[index] = row | {'group_id': metadata[index]['group_id']}
                    print(f'  {row["evaluation_status"]}; agent={row["agent_status"]}; judge={row["judge_status"]}; {row["elapsed_seconds"]:.1f}s', flush=True)
                results.write(json.dumps(rows[index], ensure_ascii=False)+'\n'); results.flush()
                if rows[index].get('case'):
                    cases.write(json.dumps(dict(case_id=example.case_id, fraudulent=example.fraudulent,
                        expected_action=example.expected_action, case=rows[index]['case']), ensure_ascii=False)+'\n'); cases.flush()
                save_report()
        return save_report(final=True)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare', help='Explicitly prepare a locked sample; no models run.')
    prep.add_argument('--data', type=Path, required=True)
    prep.add_argument('--output-dir', type=Path, required=True)
    prep.add_argument('--split', choices=['development', 'heldout'], default='development')
    prep.add_argument('--frozen-version')
    prep.add_argument('--training-data', type=Path)
    audit = sub.add_parser('run', help='Run only an explicitly prepared sample.')
    audit.add_argument('--sample-dir', type=Path, required=True)
    audit.add_argument('--output-dir', type=Path, required=True)
    audit.add_argument('--max-seconds', type=float, default=28200)
    audit.add_argument('--case-seconds', type=float, default=240)
    child = sub.add_parser('worker', help=argparse.SUPPRESS)
    child.add_argument('--result', type=Path, required=True)
    child.add_argument('--stage', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        manifest = prepare(args.data, args.output_dir, split=args.split,
                           frozen_version=args.frozen_version, training_data=args.training_data)
        print(f'Locked {manifest["selected_cases"]} cases in {args.output_dir}. No model calls made.')
    elif args.command == 'worker':
        worker(args.result, args.stage)
    else:
        report = run(args.sample_dir, args.output_dir, max_seconds=args.max_seconds, case_seconds=args.case_seconds)
        print('Saved:', args.output_dir/'report.json')
        print('Quality summary:', args.output_dir/'quality_summary.csv')
        q = report['quality_summary']['overall']
        if (q['agent_completed'] != q['planned_cases'] or q['code_checks_passed'] != q['planned_cases'] or q['judge_completed'] != q['planned_cases']):
            raise SystemExit(1)  # report remains valid; never hide partial/failed evaluations


if __name__ == '__main__':
    main()
