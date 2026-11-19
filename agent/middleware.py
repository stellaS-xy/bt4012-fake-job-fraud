"""Require genuine model-issued evidence calls before accepting final output."""
from langchain.agents.middleware import wrap_model_call
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import ToolMessage

REQUIRED_TOOLS = ("score_posting", "match_lexicon", "inspect_posting")


@wrap_model_call
def require_review_tools(request, handler):
    completed = {m.name for m in request.messages if isinstance(m, ToolMessage)
                 and m.name in REQUIRED_TOOLS and getattr(m, "status", "success") != "error"}
    missing = set(REQUIRED_TOOLS) - completed
    if missing:
        # Do not offer the final-output tool until the evidence is collected.
        tools = [t for t in request.tools if getattr(t, "name", None) in missing]
        return handler(request.override(tools=tools, response_format=None, tool_choice="required"))
    # ToolStrategy supplies Assessment as the sole remaining function tool.
    if isinstance(request.response_format, ProviderStrategy):
        return handler(request.override(tools=[], tool_choice="auto"))
    return handler(request.override(tools=[], tool_choice="required"))
