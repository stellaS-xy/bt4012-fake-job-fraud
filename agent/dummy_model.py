"""A local LangChain chat model that simulates tool calls and a final answer.

This is a scripted model, not a trained or hosted language model. It exercises
the same create_agent/tool-message loop without an API key or network access.
"""
import json
from typing import Any, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent.schemas import Assessment

from agent.middleware import REQUIRED_TOOLS
from agent.validation import supports_rejection
from agent.schemas import Evidence


class DummyReviewModel(BaseChatModel):
    @property
    def _llm_type(self):
        return "offline-job-review-simulator"

    @property
    def _identifying_params(self):
        return {"mode": "dummy", "version": "dummy-review-v1"}

    def bind_tools(self, tools: Sequence[Any], *, tool_choice=None, **kwargs):
        names = {tool.name for tool in tools}
        if not set(REQUIRED_TOOLS).issubset(names):
            raise ValueError("The demo requires all three review tools.")
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
            evidence = results["match_lexicon"]["evidence"] + results["inspect_posting"]["evidence"]
            strong = [e for e in evidence if supports_rejection(Evidence.model_validate(e))]
            decision = "propose_rejection" if strong else "escalate_to_review"
            if strong:
                reason = ("Simulated rejection proposal: the supplied posting contains an explicit "
                          "payment or credential request confirmed by the context reviewer. "
                          "A reviewer must read the quoted context and confirm the decision.")
            else:
                reason = ("Simulated escalation: the dummy evidence does not support a rejection proposal. "
                          "A score, ordinary job terminology, urgency or missing supplied fields alone "
                          "does not establish fraud. The original flag still needs human review.")
            assessment = Assessment(
                posting_id=posting_id, model_score=score["model_score"], cutoff=score["cutoff"],
                flagged=score["flagged"], recommendation=decision, rationale=reason,
                evidence_ids=[e["evidence_id"] for e in evidence],
                limitations=["Offline test model and placeholder classifier score; LLM context review was not run.",
                             "Rule candidates and historical background terms do not confirm risky requests.",
                             "Employer legitimacy and market salary were not checked.",
                             "The synthetic score is not a probability of fraud."],
                next_steps=["Read the cited advertisement wording in context.",
                            "Verify the employer through an independently found official contact.",
                            "Have a named reviewer confirm rejection or escalate the case."],
            )
            message = AIMessage(content=assessment.model_dump_json())
        return ChatResult(generations=[ChatGeneration(message=message)])
