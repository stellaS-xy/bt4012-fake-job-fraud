"""python -m evaluator --data rows.csv, or --cases saved_cases.jsonl."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from agent.core.config import Settings, load_settings
from evaluator.config import load_judge_settings
from evaluator.data import load_examples
from evaluator.runner import run_evaluation
from evaluator.export_html import render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--data', type=Path, help='CSV/JSONL postings and optional annotations; run the agent.')
    source.add_argument('--cases', type=Path, help='Saved agent cases as JSON/JSONL; do not rerun the agent.')
    parser.add_argument('--code-only', action='store_true', help='Skip the judge; generating new cases still uses the agent.')
    parser.add_argument('--judge-model', help='Override JUDGE_MODEL; select a model already available to your chosen provider.')
    parser.add_argument('--limit', type=int, help='Evaluate a prefix for a smoke run; not a representative sample.')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--split', choices=('development', 'heldout'), default='development')
    parser.add_argument('--frozen-version', help='Record the frozen agent/prompt/rule version for a held-out evaluation.')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    path = args.data or args.cases
    if args.repeats < 1 or (args.limit is not None and args.limit < 1):
        parser.error('--repeats and --limit must be positive')
    reserved_names = {'d1_test.csv', 'd1_test_feature_engineered.csv'}
    if path.name in reserved_names and args.split != 'heldout':
        parser.error('For the reserved test file, select --split heldout and provide --frozen-version.')
    if args.split == 'heldout' and not args.frozen_version:
        parser.error('Freeze the implementation before a held-out evaluation and record --frozen-version.')
    examples = load_examples(path, replay=bool(args.cases))
    original_count = len(examples)
    if args.limit:
        examples = examples[:args.limit]
    agent_settings = Settings() if args.cases else load_settings()
    judge_settings = None if args.code_only else load_judge_settings()
    if args.judge_model and not args.code_only:
        from dataclasses import replace
        judge_settings = replace(judge_settings, model=args.judge_model)
    output = args.output_dir or Path('evaluator/outputs') / datetime.now(timezone.utc).strftime('run_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=True)
    if any((output/name).exists() for name in ('report.json', 'results.jsonl', 'cases.jsonl')):
        parser.error('Output already contains evaluation files. Choose a new --output-dir to preserve the previous run.')
    with (output/'results.jsonl').open('w', encoding='utf-8') as results, \
         (output/'cases.jsonl').open('w', encoding='utf-8') as snapshots:
        def save(row):
            results.write(json.dumps(row)+'\n')
            results.flush()
            if row['case'] and row['repeat'] == 0:
                snapshots.write(json.dumps(dict(case_id=row['case_id'],
                    fraudulent=row['fraudulent'], expected_action=row['expected_action'],
                    case=row['case']))+'\n')
                snapshots.flush()
        report = run_evaluation(examples, agent_settings, judge_settings,
            code_only=args.code_only, repeats=args.repeats,
            progress=lambda text: print(text, flush=True), on_result=save)
    report.update(source=str(path.resolve()), split=args.split, frozen_version=args.frozen_version,
                  available_cases=original_count, selected_cases=len(examples), repeats=args.repeats,
                  replay=bool(args.cases), offline_agent=False)
    (output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (output/'review_report.html').write_text(render(report), encoding='utf-8')
    print(json.dumps(report['summary'], indent=2))
    print('Saved:', output/'report.json')
    print('HTML report:', output/'review_report.html')
    if (report['summary']['judge_failed'] or report['summary']['agent_failures']
            or report['summary']['input_failures'] or report['summary']['code_checks_failed']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
