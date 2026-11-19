"""Manual reviewer action, kept outside the agent's tool registry."""
from datetime import datetime, timezone

from agent.audit import save_review
from agent.schemas import ReviewCase, ReviewerDecision


def confirm_decision(case: ReviewCase, decision: str, reviewer: str, audit_path):
    reviewer = reviewer.strip()
    if not reviewer:
        raise ValueError("Enter the reviewer's name before saving a decision.")
    result = ReviewerDecision(posting_id=case.posting.posting_id, decision=decision,
                              reviewer=reviewer, recorded_at=datetime.now(timezone.utc).isoformat())
    return save_review(audit_path, case, result)
