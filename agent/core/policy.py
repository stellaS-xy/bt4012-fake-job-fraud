"""Deterministic recommendations from the two tools; never execute an action."""
import math


def classifier_check(output):
    numbers = [output.get('model_score'), output.get('cutoff')]
    valid = (all(isinstance(v, (int, float)) and not isinstance(v, bool)
                 and math.isfinite(v) and 0 <= v <= 1 for v in numbers)
             and type(output.get('flagged')) is bool
             and output['flagged'] == (numbers[0] >= numbers[1]))
    if not valid:
        return {'passed': False, 'review_reasons': ['The classifier did not return a valid score and cutoff.']}
    if output.get('is_test_fixture') is not False:
        return {'passed': False, 'review_reasons': ['The classifier score is a test fixture, so this check cannot pass.']}
    if output['flagged']:
        return {'passed': False, 'review_reasons': ['The classifier score meets or exceeds its review cutoff.']}
    return {'passed': True, 'review_reasons': []}


def lexicon_check(output):
    if output.get('is_test_fixture') is not False or output.get('context_status') != 'completed':
        return {'passed': False, 'review_reasons': ['The lexicon context check did not complete successfully.']}
    reasons = []
    for evidence in output.get('evidence', []):
        if evidence.get('source') not in ('risk_lexicon', 'llm_context'):
            continue
        context = evidence.get('context')
        if context == 'request':
            reasons.append('A payment or secret-credential request was identified: '+str(evidence.get('quote') or evidence.get('observation')))
        elif context not in ('warning', 'negated') or evidence.get('reviewed_by') != 'llm':
            reasons.append('Potentially risky wording remains unresolved: '+str(evidence.get('quote') or evidence.get('observation')))
    return {'passed': not reasons, 'review_reasons': reasons}


def recommendation_policy(trace):
    outputs = {t.name: t.output for t in trace if t.status == 'success'}
    checks = {}
    for name, check in (('score_posting', classifier_check), ('match_lexicon', lexicon_check)):
        checks[name] = check(outputs[name]) if name in outputs else {
            'passed': False, 'review_reasons': [f'The {name} tool did not complete successfully.']}
    passed = all(check['passed'] for check in checks.values())
    return {'recommendation': 'pass_no_review_needed' if passed else 'review_needed',
            'tool_checks': checks,
            'review_reasons': [reason for check in checks.values() for reason in check['review_reasons']]}
