"""Case-scoped LangChain tool registration and execution traces."""
import json
from functools import lru_cache

from langchain.tools import tool

from agent.core.schemas import Posting, ToolTrace
from agent.core.config import load_settings
from agent.tools.classifier import score_posting as score_real
from agent.tools.lexicon import match_lexicon as check_lexicon


def build_tools(posting: Posting, scenario: str, trace: list[ToolTrace], settings=None):
    settings = settings or load_settings()
    classifier_adapter, lexicon_adapter = score_real, check_lexicon
    @lru_cache(maxsize=1)
    def lexical_result():
        return lexicon_adapter(posting, settings)

    @lru_cache(maxsize=1)
    def classifier_result():
        return classifier_adapter(posting, settings)

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
        """Score this posting with real LightGBM; return passed, review_reasons, score, cutoff, flag and limitations."""
        return execute("score_posting", posting_id, classifier_result)

    @tool
    def match_lexicon(posting_id: str) -> str:
        """Check risk wording with rules and LLM context; return passed, review_reasons and exact quotes. Historical terms are background."""
        return execute("match_lexicon", posting_id, lexical_result)

    return [score_posting, match_lexicon]
