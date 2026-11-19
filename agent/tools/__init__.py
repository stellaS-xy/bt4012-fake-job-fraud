"""Case-scoped LangChain tool registration and execution traces."""
import json
from functools import lru_cache

from langchain.tools import tool

from agent.schemas import Posting, ToolTrace
from agent.tools.classifier import score_posting_stub
from agent.tools.lexicon import match_lexicon as check_lexicon
from agent.tools.posting import inspect_posting_stub


def build_tools(posting: Posting, scenario: str, trace: list[ToolTrace], settings=None):
    @lru_cache(maxsize=1)
    def lexical_result():
        return check_lexicon(posting, settings)

    def execute(name, posting_id, operation):
        args = {"posting_id": posting_id}
        try:
            if posting_id != posting.posting_id:
                raise ValueError("Tools can access only the current posting.")
            result = operation()
            trace.append(ToolTrace(name=name, arguments=args, status="success", output=result))
            return json.dumps(result)
        except Exception as exc:
            trace.append(ToolTrace(name=name, arguments=args, status="error",
                                   output={"error": "Tool failed.", "error_type": type(exc).__name__}))
            raise

    @tool
    def score_posting(posting_id: str) -> str:
        """Return the current posting's synthetic classifier score. This is a dummy tool."""
        return execute("score_posting", posting_id, lambda: score_posting_stub(posting, scenario))

    @tool
    def match_lexicon(posting_id: str) -> str:
        """Find risk patterns and use an LLM to check context, with exact quotes; dataset terms are background only."""
        return execute("match_lexicon", posting_id, lexical_result)

    @tool
    def inspect_posting(posting_id: str) -> str:
        """Inspect supplied fields and narrow demo patterns. No employer or salary verification."""
        return execute("inspect_posting", posting_id, lambda: inspect_posting_stub(posting))

    return [score_posting, match_lexicon, inspect_posting]
