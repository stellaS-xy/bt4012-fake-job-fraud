"""Real risk candidates, dataset background and optional LLM context review."""
import csv
from functools import lru_cache
from pathlib import Path
import re

from agent.tools.context_review import review_context, validate_context, interpreted_findings
from agent.core.errors import error_code_for
from agent.tools.risk_rules import VERSION, TEXT_FIELDS, candidate_matches
from agent.core.schemas import Evidence
from agent.tools.posting import posting_observations
from agent.core.policy import lexicon_check

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
    if settings is not None:
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
    basic = posting_observations(posting)
    evidence.extend(basic['evidence'])
    output = {"is_test_fixture": False, "lexicon_version": VERSION, "context_status": status,
            'external_checks': basic['external_checks'],
            "context_error": error, "context_model": settings.model if status == "completed" else None,
            "evidence": evidence,
            "note": "Pass requires a completed context review with no risky or unresolved findings. Background words and missing optional fields alone do not fail this check. No external facts were verified."}
    return {**output, **lexicon_check(output)}
