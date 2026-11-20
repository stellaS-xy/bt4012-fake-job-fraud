"""A local LangChain chat model that simulates tool calls and a final answer.

This is a scripted model, not a trained or hosted language model. It exercises
the same create_agent/tool-message loop without an API key or network access.
"""
import json
from typing import Any, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent.core.schemas import Assessment

from agent.core.middleware import REQUIRED_TOOLS


class ScriptedChatModel(BaseChatModel):
    @property
    def _llm_type(self):
        return "offline-job-review-simulator"

    @property
    def _identifying_params(self):
        return {"mode": "test_fixture", "version": "scripted-chat-v1"}

    def bind_tools(self, tools: Sequence[Any], *, tool_choice=None, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        human = next(m for m in reversed(messages) if isinstance(m, HumanMessage))
        posting_id = json.loads(human.content)["posting_id"]
        results = {m.name: json.loads(m.content) for m in messages
                   if isinstance(m, ToolMessage) and m.name in REQUIRED_TOOLS}
        missing = [name for name in REQUIRED_TOOLS if name not in results]
        if missing:
            message = AIMessage(content="", tool_calls=[
                {"name": name, "args": {"posting_id": posting_id}, "id": f"demo_{name}"}
                for name in missing
            ])
        else:
            score = results["score_posting"]
            evidence = results["match_lexicon"]["evidence"]
            decision = "review_needed"
            reason = ("Review needed: this offline test uses a placeholder classifier and "
                      "the lexicon context review did not run. Both checks must pass "
                      "before recommending no review. This is a recommendation only.")
            assessment = Assessment(
                posting_id=posting_id, model_score=score["model_score"], cutoff=score["cutoff"],
                flagged=score["flagged"], recommendation=decision, rationale=reason,
                uses_test_fixtures=True,
                evidence_ids=[e["evidence_id"] for e in evidence],
                limitations=["Offline test model and placeholder classifier score; LLM context review was not run.",
                             "Rule candidates and historical background terms do not confirm risky requests.",
                             "Employer legitimacy and market salary were not checked.",
],
                next_steps=["Read the cited advertisement wording in context.",
                            "Verify the employer through an independently found official contact.",
                            "Run the real classifier and context checks to resolve the incomplete checks."],
            )
            message = AIMessage(content=assessment.model_dump_json())
        return ChatResult(generations=[ChatGeneration(message=message)])
