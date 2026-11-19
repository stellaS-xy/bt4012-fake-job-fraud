"""Aggregate code results, judge ratings and eligible classifier predictions."""
from evaluator.schemas import DIMENSIONS


def classification_metrics(rows):
    eligible = [r for r in rows if r.get('fraudulent') is not None
                and r.get('classifier_prediction') is not None]
    if not eligible:
        return dict(status='unavailable', evaluated=0,
                    reason='No labelled cases with a validated, real classifier prediction. Recommendations are not fraud labels.')
    tp = sum(r['classifier_prediction'] and r['fraudulent'] == 1 for r in eligible)
    fp = sum(r['classifier_prediction'] and r['fraudulent'] == 0 for r in eligible)
    fn = sum(not r['classifier_prediction'] and r['fraudulent'] == 1 for r in eligible)
    tn = len(eligible)-tp-fp-fn
    return dict(status='available', evaluated=len(eligible), true_positive=tp,
        false_positive=fp, false_negative=fn, true_negative=tn,
        precision=tp/(tp+fp) if tp+fp else None,
        recall=tp/(tp+fn) if tp+fn else None,
        false_positive_rate=fp/(fp+tn) if fp+tn else None)


def summarize(rows):
    judged = [r for r in rows if r['judge_status'] == 'completed']
    checks = [r['code'] for r in rows if r.get('code') is not None]
    scored = [r for r in judged if r.get('scoring')]
    eligible_scores = {d: [r['scoring']['effective_scores'][d] for r in scored
                          if r['scoring']['effective_scores'][d] is not None] for d in DIMENSIONS}
    matches = [r['code']['expected_action_match'] for r in rows
               if r.get('code') and r['code']['expected_action_match'] is not None]
    table = {str(label): {a: 0 for a in ('pass_no_review_needed', 'review_needed')}
             for label in (0, 1)}
    for row in rows:
        if row.get('fraudulent') is not None and row.get('recommendation'):
            table[str(row['fraudulent'])][row['recommendation']] += 1
    return dict(evaluations=len(rows), independent_cases=len({r['case_id'] for r in rows}),
        recommendation_counts={action: sum(r.get('recommendation') == action for r in rows)
                               for action in ('pass_no_review_needed', 'review_needed')},
        input_failures=sum(r['agent_status'] == 'input_invalid' for r in rows),
        agent_failures=sum(r['agent_status'] in ('failed', 'fallback') for r in rows),
        agent_completed=sum(r['agent_status'] == 'completed' for r in rows),
        safe_fallbacks=sum(r['agent_status'] == 'fallback' and r.get('code', {}).get('safety_checks_passed', False) for r in rows if r.get('code')),
        context_review_failures=sum(not c['checks'].get('context_review_completed', True) for c in checks),
        code_checks_passed=sum(c['all_passed'] for c in checks),
        code_checks_failed=sum(not c['all_passed'] for c in checks),
        code_checks_unavailable=len(rows)-len(checks),
        judge_completed=len(judged), judge_failed=sum(r['judge_status'] == 'failed' for r in rows),
        judge_skipped=sum(r['judge_status'] == 'skipped' for r in rows),
        quality_scored_cases=sum(r['scoring']['answer_quality_eligible'] for r in scored),
        scored_count_by_dimension={d: len(eligible_scores[d]) for d in DIMENSIONS},
        mean_judge_scores={d: sum(eligible_scores[d])/len(eligible_scores[d])
                           if eligible_scores[d] else None for d in DIMENSIONS},
        judge_recommendation_acceptable=sum(r['scoring']['recommendation_acceptable'] for r in scored),
        judge_reported_unsupported_claims=sum(len(r['judgment']['unsupported_claims']) for r in judged),
        expected_action_matches=sum(matches), expected_action_annotations=len(matches),
        recommendations_by_fraud_label=table,
        classifier_metrics=classification_metrics(rows),
        notes=['Judge means use effective scores with explicit per-dimension denominators; incomplete reviews have no completed-answer quality scores.',
               'Recommendations by fraud label are descriptive, not action accuracy. A review-needed recommendation can apply to either label.',
               'Judge ratings are automated opinions, not verified truth or fraud predictions.',
               'Repeated cases are not new independent examples.'])
