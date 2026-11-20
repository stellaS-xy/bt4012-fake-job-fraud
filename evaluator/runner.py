"""Run/replay the agent, then apply code checks and independent judge calls."""
from datetime import datetime, timezone
import hashlib
import json
import math
from agent.core.errors import error_code_for
from agent.core.runner import run_review
from evaluator.checks import code_checks
from evaluator.judge import judge_case, validate_judgment, judge_error_code, RUBRIC_VERSION, SYSTEM_PROMPT
from evaluator.report import summarize
from evaluator.scoring import scorecard


def run_evaluation(examples, agent_settings, judge_settings=None, *, code_only=False,
                   repeats=1, agent_fn=run_review, judge_fn=judge_case,
                   progress=None, on_result=None):
    if repeats < 1 or not examples:
        raise ValueError('Provide examples and a positive repeat count.')
    if not code_only and (judge_settings is None):
        raise ValueError('Select a real judge or explicitly use code-only mode.')
    rows = []
    for repeat in range(repeats):
        for example in examples:
            if progress:
                progress(f"Round {repeat+1}: {example.case_id}")
            row = dict(case_id=example.case_id, repeat=repeat, fraudulent=example.fraudulent,
                expected_action=example.expected_action,
                input_error=example.input_error,
                agent_status='failed', agent_error=None, code=None, judge_status='skipped',
                judge_error=None, judgment=None, recommendation=None,
                scoring=None,
                classifier_prediction=None, case=None)
            if example.posting is None:
                row.update(agent_status='input_invalid', input_error=example.input_error or 'invalid_posting_fields')
                rows.append(row)
                if on_result:
                    on_result(row)
                continue
            try:
                case = example.case or agent_fn(example.posting, 'ordinary', agent_settings)
                row.update(case=case.model_dump(mode='json'), agent_status=case.status,
                           agent_error=case.error_code, recommendation=case.assessment.recommendation)
                row['code'] = code_checks(example.posting, case, example.expected_action)
                if progress:
                    progress(f"  Agent: {case.status}; code checks: {'passed' if row['code']['all_passed'] else 'failed'}")
                score = next((t.output for t in case.tool_trace if t.name == 'score_posting'
                              and t.status == 'success'), {})
                numbers = [score.get('model_score'), score.get('cutoff')]
                if (score.get('is_test_fixture') is False
                        and score.get('posting_id') == example.posting.posting_id
                        and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                and math.isfinite(v) and 0 <= v <= 1 for v in numbers)
                        and type(score.get('flagged')) is bool
                        and score['flagged'] == (numbers[0] >= numbers[1])):
                    # Evaluate the classifier tool independently of final-answer
                    # failures. Otherwise dropping failed LLM cases biases metrics.
                    row['classifier_prediction'] = score['flagged']
                if not code_only:
                    if progress:
                        progress('  Running LLM judge...')
                    try:
                        judgment = validate_judgment(judge_fn(example.posting, case, judge_settings),
                                                     example.posting, case)
                        row.update(judge_status='completed', judgment=judgment.model_dump())
                        row['scoring'] = scorecard(case, row['code'], judgment)
                        if progress:
                            progress('  Judge: completed')
                    except Exception as exc:
                        row.update(judge_status='failed', judge_error=judge_error_code(exc))
                        if progress:
                            progress('  Judge failed: '+row['judge_error'])
            except Exception as exc:
                row.update(agent_status='failed', agent_error=error_code_for(exc))
            rows.append(row)
            if on_result:
                on_result(row)
    sources = [e.model_dump(mode='json') for e in examples]
    models = {(r['case']['model_backend'], r['case']['model_name']) for r in rows if r['case']}
    same_model = (not code_only and
                  (judge_settings.backend, judge_settings.model) in models)
    return dict(schema_version=2, created_at=datetime.now(timezone.utc).isoformat(),
        input_sha256=hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
        rubric_version=RUBRIC_VERSION,
        rubric_sha256=hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        judge=None if code_only else dict(backend=judge_settings.backend, model=judge_settings.model,
                                          same_model_as_agent=same_model),
        limitations=['The judge can make mistakes; no human validation is implied.',
                     'Using the same model for acting and judging can reproduce the same mistakes.'
                     if same_model else 'A different judge model is not a guarantee of correctness.',
                     'Exact output schemas and evidence references are checked; semantic correctness is judged heuristically.'],
        summary=summarize(rows), rows=rows)
