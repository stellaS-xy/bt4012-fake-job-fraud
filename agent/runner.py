"""LangChain orchestration, response validation and safe failure handling."""
import logging
import math

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain_core.messages import AIMessage, HumanMessage
from langsmith import tracing_context

from agent.config import Settings, load_settings
from agent.models import create_review_model
from agent.middleware import require_review_tools
from agent.fixtures import SCENARIOS
from agent.prompts import SYSTEM_PROMPT
from agent.schemas import Assessment, Posting, ReviewCase
from agent.tools import build_tools
from agent.validation import collect_evidence, validate_assessment
from agent.tracing import ToolCallRecorder
from agent.errors import error_code_for

logger = logging.getLogger(__name__)


def run_review(posting: Posting, scenario: str, settings: Settings | None = None):
    settings = settings or load_settings()
    if scenario not in SCENARIOS:
        raise ValueError("Choose one of the defined dummy scenarios.")
    trace = []
    recorder = ToolCallRecorder()
    status, error_code = "completed", None
    try:
        tools = build_tools(posting, scenario, trace, settings)
        options = {}
        middleware = [ModelCallLimitMiddleware(run_limit=6, exit_behavior="error"),
                      ToolCallLimitMiddleware(run_limit=8, exit_behavior="error")]
        if settings.backend != "dummy":
            options["response_format"] = (ProviderStrategy(Assessment) if settings.backend == "ollama_local"
                                          else ToolStrategy(Assessment, handle_errors=False))
            middleware.append(require_review_tools)
        graph = create_agent(model=create_review_model(settings), tools=tools,
                             system_prompt=SYSTEM_PROMPT, middleware=middleware, **options)
        # Disable optional LangSmith uploads even if tracing is enabled globally.
        with tracing_context(enabled=False):
            result = graph.invoke(
                {"messages": [HumanMessage(content=posting.model_dump_json())]},
                config={"recursion_limit": settings.recursion_limit, "callbacks": [recorder]},
            )
        if settings.backend != "dummy":
            assessment = Assessment.model_validate(result["structured_response"])
        else:
            final = next(m for m in reversed(result["messages"]) if isinstance(m, AIMessage))
            assessment = Assessment.model_validate_json(final.content)
        try:
            evidence = validate_assessment(assessment, posting, trace)
        except ValueError as validation_error:
            if settings.backend == "dummy":
                raise
            # Small local models may miscopy tool values. Give concrete feedback
            # once; never alter their answer or accept an unsupported assessment.
            with tracing_context(enabled=False):
                result = graph.invoke(
                    {"messages": result["messages"] + [HumanMessage(content=(
                        "Your assessment failed validation: " + str(validation_error)
                        + " Read the existing tool results again. Copy posting_id, model_score, cutoff and flagged exactly from score_posting. "
                        "Return a corrected Assessment using only returned evidence IDs."))]},
                    config={"recursion_limit": settings.recursion_limit, "callbacks": [recorder]},
                )
            assessment = Assessment.model_validate(result["structured_response"])
            evidence = validate_assessment(assessment, posting, trace)
    except Exception as exc:
        # Do not expose provider errors, credentials or arbitrary exception text in exports.
        error_code = error_code_for(exc)
        status = "fallback"
        logger.warning("Review escalated (%s).", error_code)
        try:
            evidence = [e for e in collect_evidence(trace)
                        if e.quote is None or e.quote in (getattr(posting, e.field) or "")]
        except (ValueError, TypeError, KeyError):
            evidence = []
        score = next((t.output for t in trace if t.name == "score_posting" and t.status == "success"), {})
        # A malformed future tool adapter must not break the escalation fallback.
        values = [score.get("model_score"), score.get("cutoff")]
        valid_score = all(isinstance(v, (int, float)) and not isinstance(v, bool)
                          and math.isfinite(v) and 0 <= v <= 1 for v in values)
        if (not valid_score or type(score.get("flagged")) is not bool
                or score["flagged"] != (values[0] >= values[1])
                or score.get("posting_id") != posting.posting_id):
            score = {}
        assessment = Assessment(
            posting_id=posting.posting_id, model_score=score.get("model_score"),
            cutoff=score.get("cutoff"), flagged=score.get("flagged"), recommendation="escalate_to_review",
            rationale="The review did not complete its checks. Escalate for manual review; no rejection was executed.",
            evidence_ids=[e.evidence_id for e in evidence],
            limitations=["The classifier still returns a placeholder score.", "Review incomplete; do not rely on an automated assessment.",
                         "Employer legitimacy and market salary were not checked."],
            next_steps=["Inspect the supplied posting manually.", "Retry the demo after checking the local tool setup."],
        )
    recorder.attach_ids(trace)
    return ReviewCase(posting=posting, scenario=scenario, assessment=assessment,
                      evidence=evidence, tool_trace=trace, model_backend=settings.backend,
                      model_name=settings.model if settings.backend != "dummy" else "offline-test-double",
                      status=status, error_code=error_code, llm_tool_calls=recorder.calls)
