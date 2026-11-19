"""Replacement point for the future real classifier service."""
from agent.fixtures import SCENARIOS
from agent.schemas import Posting


def score_posting_stub(posting: Posting, scenario: str):
    score = SCENARIOS[scenario]["score"]
    return {"posting_id": posting.posting_id, "model_score": score, "cutoff": 0.7,
            "flagged": score >= 0.7, "model_version": "dummy-classifier-v1",
            "is_dummy": True,
            "note": "Synthetic fixture score selected by scenario, not calculated from this advertisement."}
