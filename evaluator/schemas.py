"""Evaluation inputs and structured judge outputs."""
from typing import Literal
from pydantic import Field
from agent.core.schemas import Contract, Posting, ReviewCase


class Example(Contract):
    case_id: str = Field(min_length=1)
    posting: Posting | None = None
    input_error: str | None = None
    fraudulent: Literal[0, 1] | None = None
    expected_action: Literal['pass_no_review_needed', 'review_needed'] | None = None
    case: ReviewCase | None = None


class Rating(Contract):
    score: int = Field(ge=0, le=4, strict=True)
    reason: str = Field(min_length=1)
    evidence_ids: list[str]
    answer_quotes: list[str] = Field(default_factory=list, description='Exact excerpts of what the agent actually wrote, not tool observations.')


class UnsupportedClaim(Contract):
    quote: str = Field(min_length=1, description='Exact substring of the agent rationale.')
    reason: str = Field(min_length=1)


class Judgment(Contract):
    posting_id: str
    evidence_grounding: Rating
    context_understanding: Rating
    recommendation_consistency: Rating
    explanation_clarity: Rating
    recommendation_acceptable: bool
    unsupported_claims: list[UnsupportedClaim]
    uncertainties: list[str]


DIMENSIONS = ('evidence_grounding', 'context_understanding',
              'recommendation_consistency', 'explanation_clarity')
