"""Shared input, evidence, output and reviewer contracts."""
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Posting(Contract):
    posting_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = Field(min_length=1, max_length=500)
    description: str = Field(min_length=1, max_length=20000)
    company_profile: str | None = Field(default=None, max_length=10000)
    salary_range: str | None = Field(default=None, max_length=200)
    requirements: str | None = Field(default=None, max_length=10000)
    benefits: str | None = Field(default=None, max_length=10000)

    @field_validator("title", "description")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("A title and description are required.")
        return value.strip()

    @field_validator("company_profile", "salary_range", "requirements", "benefits")
    @classmethod
    def optional_text(cls, value):
        return value.strip() or None if value is not None else None


class Evidence(Contract):
    evidence_id: str
    source: Literal["risk_lexicon", "llm_context", "dataset_lexicon", "posting"]
    field: Literal["title", "description", "company_profile", "salary_range", "requirements", "benefits"]
    kind: str
    observation: str
    quote: str | None = None
    context: Literal["candidate", "request", "negated", "warning", "uncertain", "background"] = "background"
    reviewed_by: Literal["llm", "not_checked"] = "not_checked"
    rule_id: str | None = None
    model_interpretation: Literal["request", "negated", "warning", "uncertain"] | None = None
    context_guard: Literal["negation", "warning"] | None = None


class ToolTrace(Contract):
    call_id: str | None = None
    name: str
    arguments: dict
    status: Literal["success", "error"]
    output: dict


class Assessment(Contract):
    posting_id: str
    model_score: float | None = Field(..., ge=0, le=1, description="Required: copy model_score exactly from the score_posting tool result.")
    cutoff: float | None = Field(..., ge=0, le=1, description="Required: copy cutoff exactly from the score_posting tool result.")
    flagged: bool | None = Field(..., description="Required: copy flagged exactly from the score_posting tool result.")
    recommendation: Literal["propose_rejection", "escalate_to_review"]
    rationale: str
    evidence_ids: list[str]
    limitations: list[str]
    next_steps: list[str]
    requires_confirmation: Literal[True] = True
    tool_results_are_placeholders: Literal[True] = True


class ReviewCase(Contract):
    schema_version: Literal[3] = 3
    posting: Posting
    scenario: str
    assessment: Assessment
    evidence: list[Evidence]
    tool_trace: list[ToolTrace]
    model_backend: Literal["dummy", "ollama", "ollama_local"] = "dummy"
    model_name: str = "offline-test-double"
    status: Literal["completed", "fallback"] = "completed"
    error_code: str | None = None
    llm_tool_calls: list[dict] = Field(default_factory=list)


class ReviewerDecision(Contract):
    posting_id: str
    decision: Literal["confirmed_rejection", "escalated_to_review"]
    reviewer: str = Field(min_length=1, max_length=100)
    recorded_at: str
    tool_results_are_placeholders: Literal[True] = True
