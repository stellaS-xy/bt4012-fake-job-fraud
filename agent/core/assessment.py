"""One bounded, real-model recovery using completed tool evidence."""
import json
import math
from typing import Literal
from typing import Annotated
from pydantic import Field, create_model
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import tracing_context

from agent.core.middleware import REQUIRED_TOOLS
from agent.core.prompts import SYSTEM_PROMPT
from agent.core.schemas import Assessment
from agent.core.validation import collect_evidence
from agent.core.policy import recommendation_policy


def assessment_schema(test_fixtures=None):
    """Bound generated answer length and optionally constrain test provenance."""
    item = Annotated[str, Field(max_length=500)]
    flags = {} if test_fixtures is None else {'uses_test_fixtures': (Literal[test_fixtures], Field(...))}
    return create_model('Assessment', __base__=Assessment,
        rationale=(str, Field(..., min_length=1, max_length=1600)),
        limitations=(list[item], Field(..., max_length=3)),
        next_steps=(list[item], Field(..., max_length=3)), **flags)


def completed_tools(trace):
    return set(REQUIRED_TOOLS).issubset({t.name for t in trace if t.status == 'success'})


def recover_assessment(model, settings, posting, trace, messages=None):
    if not completed_tools(trace):
        raise ValueError('Cannot recover an assessment before the required tools complete.')
    outputs = {t.name: t.output for t in trace if t.status == 'success'}
    score = outputs['score_posting']
    numbers = [score.get('model_score'), score.get('cutoff')]
    if (not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                and 0 <= v <= 1 for v in numbers)
            or type(score.get('flagged')) is not bool or score['flagged'] != (numbers[0] >= numbers[1])
            or score.get('posting_id') != posting.posting_id):
        raise ValueError('The classifier result is invalid; do not invent a replacement.')
    evidence = collect_evidence(trace)
    ids = [e.evidence_id for e in evidence]
    decision = recommendation_policy(trace)['recommendation']
    response_type = create_model('Assessment', __base__=assessment_schema(score.get('is_test_fixture', True)),
        posting_id=(Literal[posting.posting_id], Field(...)),
        model_score=(Literal[score['model_score']], Field(...)),
        cutoff=(Literal[score['cutoff']], Field(...)), flagged=(Literal[score['flagged']], Field(...)),
        evidence_ids=(list[Literal[tuple(ids)]] if ids else list[str], Field(..., max_length=len(ids))),
        recommendation=(Literal[decision], Field(...)))
    method = 'json_schema' if settings.backend == 'ollama_local' else 'function_calling'
    structured = model.with_structured_output(response_type, method=method)
    history = messages or [HumanMessage(content=json.dumps(dict(posting=posting.model_dump(), tool_results=outputs)))]
    with tracing_context(enabled=False):
        result = structured.invoke([SystemMessage(content=SYSTEM_PROMPT), *history,
            HumanMessage(content='The previous final answer was missing or failed validation. '
                'Use the completed tool results to return one valid assessment. Do not call business tools again. '
                'Explain the actual quoted wording and context, including warnings or negations where present. '
                'Copy the fixed score values and cite only supplied IDs. Do not add outside claims about '
                'when hiring fees are legitimate, normal, lawful or customary. '
                'Explain why both checks pass, or state each reason review is needed. '
                'The recommendation is fixed by the two-tool policy: '+json.dumps(recommendation_policy(trace)))])
    return Assessment.model_validate(result)
