"""Application checks that do not rely on the language model's compliance."""
from agent.schemas import Assessment, Evidence, Posting, ToolTrace
from agent.middleware import REQUIRED_TOOLS


def supports_rejection(item):
    return (item.kind in ("payment_request", "credential_request")
            and item.source in ("risk_lexicon", "llm_context")
            and item.context == "request" and item.reviewed_by == "llm"
            and item.context_guard is None and bool(item.quote))


def collect_evidence(trace: list[ToolTrace]):
    evidence = {}
    for event in trace:
        if event.status == "success":
            for value in event.output.get("evidence", []):
                item = Evidence.model_validate(value)
                if item.evidence_id in evidence and evidence[item.evidence_id] != item:
                    raise ValueError("A repeated evidence ID changed its contents.")
                evidence[item.evidence_id] = item
    return list(evidence.values())


def validate_assessment(assessment: Assessment, posting: Posting, trace: list[ToolTrace]):
    results = {t.name: t.output for t in trace if t.status == "success"}
    if not set(REQUIRED_TOOLS).issubset(results):
        raise ValueError("All required tools must finish before an assessment is accepted.")
    if any(t.arguments.get("posting_id") != posting.posting_id for t in trace):
        raise ValueError("A tool referred to a different posting.")
    if assessment.posting_id != posting.posting_id:
        raise ValueError("Assessment refers to a different posting.")
    score = results["score_posting"]
    if score.get("posting_id") != posting.posting_id:
        raise ValueError("Score belongs to another posting.")
    for name in ("model_score", "cutoff", "flagged"):
        if getattr(assessment, name) != score[name]:
            raise ValueError(f"Assessment changed the tool's {name}.")
    evidence = collect_evidence(trace)
    by_id = {e.evidence_id: e for e in evidence}
    if len(by_id) != len(evidence):
        raise ValueError("Duplicate evidence IDs.")
    if not set(assessment.evidence_ids).issubset(by_id):
        raise ValueError("Assessment cited evidence that was not returned by a tool.")
    for item in evidence:
        if item.quote is not None and item.quote not in (getattr(posting, item.field) or ""):
            raise ValueError("A quoted passage does not occur in the supplied field.")
    if assessment.recommendation == "propose_rejection" and not any(
        supports_rejection(by_id[e]) for e in assessment.evidence_ids
    ):
        raise ValueError("Rejection proposals require a cited, LLM-reviewed request; candidates, warnings, background terms and scores cannot justify them.")
    if (assessment.recommendation == "propose_rejection"
            and results['match_lexicon'].get('context_status') != 'completed'):
        raise ValueError("A failed or missing context review requires escalation.")
    return evidence
