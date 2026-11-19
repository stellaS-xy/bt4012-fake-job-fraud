"""Transparent local observations behind the simulated inspection tool."""
import re
from agent.schemas import Evidence, Posting


def inspect_posting_stub(posting: Posting):
    evidence = []

    def add(field, kind, observation, quote=None):
        evidence.append(Evidence(evidence_id=f"posting_{len(evidence)+1}", source="posting",
                                 field=field, kind=kind, observation=observation, quote=quote).model_dump())

    for field in ("company_profile", "salary_range"):
        if getattr(posting, field) is None:
            add(field, "missing_information", f"{field.replace('_', ' ').capitalize()} was not supplied; this does not verify that the employer lacks it.")
    # Risk requests are handled only by the lexicon/context tool, so this tool
    # cannot bypass its interpretation of negations or educational quotations.
    patterns = {
        "urgency_wording": r"\b(?:immediate start|apply now)\b",
    }
    for kind, pattern in patterns.items():
        for match in re.finditer(pattern, posting.description, flags=re.IGNORECASE):
            add("description", kind, f"Demo rule matched {kind.replace('_', ' ')}; confirm its context manually.", match.group())
    return {"is_dummy": True, "evidence": evidence,
            "external_checks": {"market_salary": "not_checked", "employer_legitimacy": "not_checked"},
            "note": "Only narrow local demo patterns are checked. Unmatched wording does not establish safety."}
