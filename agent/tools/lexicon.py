"""Real risk candidates, dataset background and optional LLM context review."""
import csv
from functools import lru_cache
from pathlib import Path
import re

from agent.context_review import review_context, validate_context, interpreted_findings
from agent.errors import error_code_for
from agent.risk_rules import VERSION, TEXT_FIELDS, candidate_matches
from agent.schemas import Evidence

BACKGROUND_PATH = Path(__file__).resolve().parents[1] / "resources" / "dataset_terms.csv"


@lru_cache(maxsize=1)
def background_terms():
    with BACKGROUND_PATH.open(encoding="utf-8", newline="") as stream:
        return tuple(csv.DictReader(stream))


def match_lexicon(posting, settings=None, context_reviewer=None):
    candidates = candidate_matches(posting)
    evidence = [Evidence(evidence_id=c["evidence_id"], source="risk_lexicon", field=c["field"],
                         kind=c["category"], quote=c["quote"], context="candidate",
                         observation="Rule candidate; context has not yet been confirmed.",
                         rule_id=c["rule_id"]).model_dump() for c in candidates]
    status, error = "not_run", None
    if settings is not None and settings.backend != "dummy":
        try:
            report = (context_reviewer or review_context)(posting, candidates, settings)
            report = validate_context(report, posting, candidates)
            reviewed_evidence = []
            for finding in interpreted_findings(report, posting, candidates):
                reviewed_evidence.append(Evidence(evidence_id=finding['evidence_id'],
                    source='risk_lexicon' if 'rule_id' in finding else 'llm_context',
                    field=finding['field'], kind=finding['category'], quote=finding['quote'],
                    rule_id=finding.get('rule_id'), context=finding['interpretation'],
                    model_interpretation=finding['model_interpretation'], context_guard=finding['context_guard'],
                    reviewed_by='llm', observation=finding['reason']).model_dump())
            evidence = reviewed_evidence
            status = "completed"
        except Exception as exc:
            status, error = "failed", error_code_for(exc)
    for field in TEXT_FIELDS:
        text = getattr(posting, field) or ""
        for row in background_terms():
            term = row["term"]
            pattern = r"\b" + r"\s+".join(re.escape(t) for t in term.split()) + r"\b"
            match = re.search(pattern, text, re.I)
            if match:
                evidence.append(Evidence(evidence_id=f"background_{len(evidence)+1}", source="dataset_lexicon",
                    field=field, kind="dataset_association", quote=match.group(), context="background",
                    observation=f"'{term}' was associated with labels in the historical training analysis. Background only; not evidence of a risky request.").model_dump())
    return {"is_dummy": False, "lexicon_version": VERSION, "context_status": status,
            "context_error": error, "context_model": settings.model if status == "completed" else None,
            "evidence": evidence,
            "note": "Only LLM-reviewed request findings can support a rejection proposal. Background terms and unmatched wording do not establish fraud or safety."}
