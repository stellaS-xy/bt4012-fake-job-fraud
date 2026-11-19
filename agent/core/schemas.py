"""Shared input, evidence and recommendation contracts."""
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
    location: str | None = Field(default=None, max_length=1000)
    department: str | None = Field(default=None, max_length=1000)
    employment_type: str | None = Field(default=None, max_length=500)
    required_experience: str | None = Field(default=None, max_length=500)
    required_education: str | None = Field(default=None, max_length=500)
    industry: str | None = Field(default=None, max_length=1000)
    function: str | None = Field(default=None, max_length=1000)
    telecommuting: Literal[0, 1] | None = None
    has_company_logo: Literal[0, 1] | None = None
    has_questions: Literal[0, 1] | None = None

    @field_validator("title", "description")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("A title and description are required.")
        return value.strip()

    @field_validator("company_profile", "salary_range", "requirements", "benefits", 'location', 'department',
                     'employment_type', 'required_experience', 'required_education', 'industry', 'function')
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
    recommendation: Literal["pass_no_review_needed", "review_needed"]
    rationale: str
    evidence_ids: list[str]
    limitations: list[str]
    next_steps: list[str]
    uses_test_fixtures: bool = False


class ReviewCase(Contract):
    schema_version: Literal[5] = 5
    posting: Posting
    scenario: str
    assessment: Assessment
    evidence: list[Evidence]
    tool_trace: list[ToolTrace]
    model_backend: Literal["ollama", "ollama_local"]
    model_name: str = Field(min_length=1)
    status: Literal["completed", "fallback"] = "completed"
    error_code: str | None = None
    llm_tool_calls: list[dict] = Field(default_factory=list)
    recovery_attempted: bool = False
    failure_stage: str | None = None

