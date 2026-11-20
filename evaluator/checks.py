"""Deterministic checks, independent of judge preferences and gold fraud labels."""
from agent.core.validation import collect_evidence, validate_assessment
from agent.core.policy import recommendation_policy


def code_checks(posting, case, expected_action=None):
    evidence = case.evidence
    by_id = {e.evidence_id: e for e in evidence}
    quoted = [e for e in evidence if e.quote is not None]
    quotes_ok = all(bool(e.quote.strip()) and e.quote in (getattr(posting, e.field) or '') for e in quoted)
    citations_ok = set(case.assessment.evidence_ids).issubset(by_id)
    ids_ok = len(by_id) == len(evidence)
    try:
        trace_evidence = collect_evidence(case.tool_trace)
        trace_ok = {e.evidence_id: e for e in trace_evidence} == by_id
    except (ValueError, TypeError, KeyError):
        trace_ok = False
    try:
        validate_assessment(case.assessment, posting, case.tool_trace)
        assessment_ok = True
    except (ValueError, TypeError, KeyError):
        assessment_ok = False
    cited = [by_id[i] for i in case.assessment.evidence_ids if i in by_id]
    lexical = next((t.output for t in case.tool_trace if t.name == 'match_lexicon' and t.status == 'success'), {})
    policy = recommendation_policy(case.tool_trace)
    expected = policy['recommendation'] if case.status == 'completed' else 'review_needed'
    policy_ok = case.assessment.recommendation == expected
    checks = dict(review_completed=case.status == 'completed',
        context_review_completed=lexical.get('context_status') == 'completed',
        input_preserved=case.posting.model_dump() == posting.model_dump(),
        posting_id_consistent=case.assessment.posting_id == posting.posting_id,
        evidence_ids_unique=ids_ok, quotes_exact=quotes_ok, citations_exist=citations_ok,
        evidence_matches_tool_trace=trace_ok, assessment_valid=assessment_ok,
        recommendation_policy=policy_ok)
    return dict(checks=checks, all_passed=all(checks.values()),
        safety_checks_passed=all(v for k,v in checks.items() if k not in ('review_completed', 'context_review_completed')),
        failed_checks=[k for k, passed in checks.items() if not passed],
        tool_policy=policy, quoted_evidence_count=len(quoted), cited_evidence_count=len(cited),
        expected_action_match=None if expected_action is None else case.assessment.recommendation == expected_action)
