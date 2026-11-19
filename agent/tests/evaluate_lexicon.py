"""Compare evidence extraction on shared, annotated development examples.

Run from the repository root: python -m agent.tests.evaluate_lexicon --mode all
This measures risky-request extraction, NOT job-fraud classification.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

from agent.core.config import load_settings
from agent.tools.context_review import review_context, validate_context, interpreted_findings
from agent.core.errors import error_code_for
from agent.tools.risk_rules import VERSION, candidate_matches, rule_requests
from agent.core.schemas import Posting

DEFAULT_CASES = Path(__file__).resolve().parents[1] / "resources" / "risk_examples.jsonl"


def load_cases(path):
    cases, seen = [], set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        if case['case_id'] in seen:
            raise ValueError("Development case IDs must be unique.")
        seen.add(case['case_id'])
        posting = Posting.model_validate(case['posting'])
        for gold in case['gold_requests']:
            if gold['category'] not in ('payment_request', 'credential_request'):
                raise ValueError("Unknown gold request category.")
            if not gold['quote'] or gold['quote'] not in (getattr(posting, gold['field']) or ''):
                raise ValueError("Gold evidence must be an exact quote from its field.")
        case['posting'] = posting
        cases.append(case)
    if not cases:
        raise ValueError("Provide at least one annotated development example.")
    return cases


def report_requests(report, candidates, posting):
    found = [f for f in interpreted_findings(report, posting, candidates) if f['interpretation'] == 'request']
    return [dict(field=f['field'], quote=f['quote'], category=f['category']) for f in found]


def metrics(rows):
    tp = sum(r['predicted_request'] is True and r['expected_request'] for r in rows)
    fp = sum(r['predicted_request'] is True and not r['expected_request'] for r in rows)
    positives = sum(r['expected_request'] for r in rows)
    negatives = len(rows) - positives
    findings = [f for r in rows for f in r['findings']]
    unmatched = sum(not any(f['field'] == g['field'] and f['category'] == g['category']
                             and g['quote'] in f['quote'] for g in r['gold_requests'])
                      for r in rows for f in r['findings'])
    unsupported = sum(not any(f['field'] == g['field'] and f['category'] == g['category']
                             for g in r['gold_requests']) for r in rows for f in r['findings'])
    return dict(examples=len(rows), positives=positives, negatives=negatives,
        requests_caught=tp, requests_missed_or_unavailable=positives-tp,
        false_alarms=fp, failed_reviews=sum(r['status'] != 'completed' for r in rows),
        precision=tp/(tp+fp) if tp+fp else None,
        recall=tp/positives if positives else None,
        false_alarm_rate=fp/negatives if negatives else None,
        emitted_findings=len(findings), unsupported_findings=unsupported,
        findings_without_gold_span_match=unmatched,
        accuracy_counting_failures_as_wrong=sum(r['status'] == 'completed' and
            r['predicted_request'] == r['expected_request'] for r in rows)/len(rows))


def paired_intervals(rows, draws=2000):
    """Resample cases, keeping all repeats/methods of a case together."""
    by_method = defaultdict(lambda: defaultdict(list))
    for row in rows:
        correct = float(row['status'] == 'completed' and row['predicted_request'] == row['expected_request'])
        by_method[row['method']][row['case_id']].append(correct)
    comparisons = []
    for left, right in [('llm', 'rules'), ('combined', 'rules'), ('combined', 'llm')]:
        ids = sorted(set(by_method[left]) & set(by_method[right]))
        if not ids:
            continue
        deltas = [sum(by_method[left][i])/len(by_method[left][i]) -
                  sum(by_method[right][i])/len(by_method[right][i]) for i in ids]
        rng = random.Random(42)
        samples = sorted(sum(rng.choice(deltas) for _ in deltas)/len(deltas) for _ in range(draws))
        comparisons.append(dict(comparison=f'{left} minus {right}', independent_cases=len(ids),
            mean_accuracy_difference=sum(deltas)/len(deltas),
            bootstrap_95_low=samples[int(.025*draws)], bootstrap_95_high=samples[int(.975*draws)]))
    return comparisons


def evaluate(cases, settings, repeats=1, mode='all', reviewer=review_context, progress=None):
    if repeats < 1 or mode not in ('rules', 'all'):
        raise ValueError("Choose a valid mode and a positive repeat count.")
    methods = ('rules',) if mode == 'rules' else ('rules', 'llm', 'combined')
    rows = []
    for repeat in range(repeats):
        for case in cases:
            posting = case['posting']
            candidates = candidate_matches(posting)
            for method in methods:
                if progress:
                    progress(f"Repeat {repeat+1}: {case['case_id']} / {method}")
                row = dict(case_id=case['case_id'], repeat=repeat, method=method,
                    expected_request=bool(case['gold_requests']), gold_requests=case['gold_requests'],
                    status='completed', error_code=None, predicted_request=None, findings=[])
                try:
                    if method == 'rules':
                        found = [dict(field=c['field'], quote=c['quote'], category=c['category'])
                                 for c in candidates if rule_requests(c)]
                    else:
                        supplied = candidates if method == 'combined' else []
                        report = validate_context(reviewer(posting, supplied, settings), posting, supplied)
                        found = report_requests(report, supplied, posting)
                    row.update(predicted_request=bool(found), findings=found)
                except Exception as exc:
                    row.update(status='failed', error_code=error_code_for(exc))
                rows.append(row)
    inputs = [dict(case_id=c['case_id'], posting=c['posting'].model_dump(),
                   gold_requests=c['gold_requests']) for c in cases]
    return dict(created_at=datetime.now(timezone.utc).isoformat(),
        purpose='Development evidence extraction; not fraud detection accuracy or held-out test performance.',
        warning='Included examples are synthetic regression cases, not a representative dataset. Small-sample intervals do not establish deployment quality.',
        rules_version=VERSION, backend=settings.backend, model=settings.model,
        input_sha256=hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
        independent_cases=len(cases), repeats=repeats,
        metrics={m: metrics([r for r in rows if r['method'] == m]) for m in methods},
        paired_comparisons=paired_intervals(rows), rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=DEFAULT_CASES)
    parser.add_argument('--mode', choices=('rules', 'all'), default='rules')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--max-cases', type=int, help='Limit a smoke run; never interpret a subset as full evaluation.')
    parser.add_argument('--output', type=Path, default=Path('agent/runtime/lexicon_evaluation.json'))
    args = parser.parse_args()
    from agent.core.config import Settings
    settings = Settings(backend='ollama_local') if args.mode == 'rules' else load_settings()
    cases = load_cases(args.cases)
    if args.max_cases is not None:
        if args.max_cases < 1:
            parser.error('--max-cases must be positive')
        cases = cases[:args.max_cases]
    result = evaluate(cases, settings, args.repeats, args.mode, progress=lambda s: print(s, flush=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result['metrics'], indent=2))
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
