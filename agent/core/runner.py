"""LangChain orchestration, response validation and safe failure handling."""
import logging
import math

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain_core.messages import AIMessage, HumanMessage
from langsmith import tracing_context

from agent.core.config import Settings, load_settings
from agent.core.models import create_review_model
from agent.core.middleware import require_review_tools
from agent.core.prompts import SYSTEM_PROMPT
from agent.core.schemas import Assessment, Posting, ReviewCase
from agent.tools import build_tools
from agent.core.validation import collect_evidence, validate_assessment
from agent.core.tracing import ToolCallRecorder
from agent.core.errors import error_code_for
from agent.core.assessment import recover_assessment, completed_tools, assessment_schema

logger = logging.getLogger(__name__)


def run_review(posting: Posting, scenario: str, settings: Settings | None = None):
    settings = settings or load_settings()
    trace = []
    recorder = ToolCallRecorder()
    status, error_code = "completed", None
    stage, recovery_attempted = 'tool_calling', False
    result = None
    try:
        tools = build_tools(posting, scenario, trace, settings)
        options = {}
        middleware = [ModelCallLimitMiddleware(run_limit=4, exit_behavior="error"),
                      ToolCallLimitMiddleware(run_limit=8, exit_behavior="error")]
        output_type = assessment_schema()
        options["response_format"] = (ProviderStrategy(output_type) if settings.backend == "ollama_local"
                                      else ToolStrategy(output_type, handle_errors=False))
        middleware.append(require_review_tools)
        model = create_review_model(settings)
        graph = create_agent(model=model, tools=tools,
                             system_prompt=SYSTEM_PROMPT, middleware=middleware, **options)
        # Disable optional LangSmith uploads even if tracing is enabled globally.
        with tracing_context(enabled=False):
            result = graph.invoke(
                {"messages": [HumanMessage(content=posting.model_dump_json())]},
                config={"recursion_limit": settings.recursion_limit, "callbacks": [recorder]},
            )
        try:
            stage = 'assessment_generation'
            assessment = Assessment.model_validate(result.get("structured_response"))
            stage = 'assessment_validation'
            evidence = validate_assessment(assessment, posting, trace)
        except (ValueError, KeyError):
            if not completed_tools(trace):
                raise
            stage, recovery_attempted = 'assessment_recovery', True
            assessment = recover_assessment(model, settings, posting, trace, result['messages'])
            evidence = validate_assessment(assessment, posting, trace)
    except Exception as exc:
        if (not recovery_attempted and completed_tools(trace)
                and error_code_for(exc) == 'review_incomplete'):
            stage, recovery_attempted = 'assessment_recovery', True
            try:
                assessment = recover_assessment(model, settings, posting, trace,
                                                result.get('messages') if result else None)
                evidence = validate_assessment(assessment, posting, trace)
                recorder.attach_ids(trace)
                return ReviewCase(posting=posting, scenario=scenario, assessment=assessment,
                    evidence=evidence, tool_trace=trace, model_backend=settings.backend,
                    model_name=settings.model, status='completed', llm_tool_calls=recorder.calls,
                    recovery_attempted=True)
            except Exception as recovery_error:
                exc = recovery_error
        # Do not expose provider errors, credentials or arbitrary exception text in exports.
        error_code = error_code_for(exc)
        if error_code == 'review_incomplete' and stage.startswith('assessment'):
            error_code = 'assessment_invalid'
        status = "fallback"
        logger.warning("Review incomplete (%s).", error_code)
        try:
            evidence = [e for e in collect_evidence(trace)
                        if e.quote is None or e.quote in (getattr(posting, e.field) or "")]
        except (ValueError, TypeError, KeyError):
            evidence = []
        score = next((t.output for t in trace if t.name == "score_posting" and t.status == "success"), {})
        # A malformed adapter must not break the review-needed fallback.
        values = [score.get("model_score"), score.get("cutoff")]
        valid_score = all(isinstance(v, (int, float)) and not isinstance(v, bool)
                          and math.isfinite(v) and 0 <= v <= 1 for v in values)
        if (not valid_score or type(score.get("flagged")) is not bool
                or score["flagged"] != (values[0] >= values[1])
                or score.get("posting_id") != posting.posting_id):
            score = {}
        assessment = Assessment(
            posting_id=posting.posting_id, model_score=score.get("model_score"),
            cutoff=score.get("cutoff"), flagged=score.get("flagged"), recommendation="review_needed",
            rationale="Review needed: the checks or final explanation did not complete successfully. This is a recommendation only; no action was taken.",
            evidence_ids=[e.evidence_id for e in evidence],
            uses_test_fixtures=score.get('is_test_fixture', False),
            limitations=["Classifier unavailable or incomplete checks; inspect the tool trace.", "Review incomplete; do not rely on an automated assessment.",
                         "Employer legitimacy and market salary were not checked."],
            next_steps=["Inspect the supplied posting manually.", "Retry the demo after checking the local tool setup."],
        )
    recorder.attach_ids(trace)
    return ReviewCase(posting=posting, scenario=scenario, assessment=assessment,
                      evidence=evidence, tool_trace=trace, model_backend=settings.backend,
                      model_name=settings.model,
                      status=status, error_code=error_code, llm_tool_calls=recorder.calls,
                      recovery_attempted=recovery_attempted, failure_stage=stage if status == 'fallback' else None)
