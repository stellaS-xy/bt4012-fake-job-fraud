"""Keep raw judge opinions separate from execution/grounding eligibility."""
from evaluator.schemas import DIMENSIONS


def scorecard(case, code, judgment):
    scores = {d: getattr(judgment, d).score for d in DIMENSIONS}
    adjustments = []
    if case.status != 'completed' or not code['checks']['context_review_completed']:
        for d in ('evidence_grounding', 'context_understanding', 'explanation_clarity'):
            scores[d] = None
        adjustments.append('Agent answer or required context review is incomplete: no completed-answer quality scores. Safe handling is evaluated separately.')
    if not code['checks']['recommendation_policy']:
        scores['recommendation_consistency'] = 0
        adjustments.append('Recommendation violates the code-enforced policy.')
    if not all(code['checks'][k] for k in ('input_preserved', 'quotes_exact', 'citations_exist', 'evidence_matches_tool_trace')):
        scores['evidence_grounding'] = 0 if case.status == 'completed' else None
        adjustments.append('Hard grounding checks failed.')
    return dict(effective_scores=scores, adjustments=adjustments,
                answer_quality_eligible=case.status == 'completed' and code['all_passed'],
                recommendation_acceptable=judgment.recommendation_acceptable and code['checks']['recommendation_policy'])
