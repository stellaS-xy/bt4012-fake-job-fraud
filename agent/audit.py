"""Local append-only audit export; never called by an agent tool."""
import json
from pathlib import Path
from threading import Lock

from agent.schemas import ReviewCase, ReviewerDecision

_LOCK = Lock()


def save_review(path: Path, case: ReviewCase, decision: ReviewerDecision):
    if decision.posting_id != case.posting.posting_id:
        raise ValueError("Reviewer decision belongs to a different case.")
    record = {"case": case.model_dump(mode="json"), "reviewer_decision": decision.model_dump(mode="json")}
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record
