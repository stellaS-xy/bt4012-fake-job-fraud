"""Versioned English candidate patterns, not a fraud classifier."""
from dataclasses import dataclass
import re

VERSION = "risk-patterns-v2"
TEXT_FIELDS = ("title", "description", "company_profile", "requirements", "benefits")


@dataclass(frozen=True)
class RiskRule:
    rule_id: str
    category: str
    pattern: str


RULES = (
    RiskRule("recruitment_fee", "payment_request", r"\b(?:pay|send|transfer|deposit|wire)\b.{0,100}\b(?:application fee|registration (?:fee|deposit)|interview (?:fee|deposit)|recruitment fee|refundable (?:fee|deposit)|security deposit)\b"),
    RiskRule("payment_to_recruiter", "payment_request", r"\b(?:pay|send|transfer|wire)\b.{0,70}\b(?:fee|deposit|money|funds|\$\s*\d+)\b.{0,80}\b(?:recruiter|hiring manager|interview|job offer)\b"),
    RiskRule("equipment_payment", "payment_request", r"\b(?:pay|send|transfer|wire)\b.{0,80}\b(?:equipment|training|starter kit)\b.{0,65}\b(?:before (?:starting|you start)|to (?:start|secure)|our (?:vendor|supplier))\b"),
    RiskRule("bank_password", "credential_request", r"\b(?:share|send|provide|give|disclose)\b.{0,75}\b(?:bank(?:ing)?|online bank(?:ing)?)\b.{0,30}\b(?:password|pin|login credentials)\b"),
    RiskRule("bank_code", "credential_request", r"\b(?:share|send|provide|give|forward)\b.{0,70}\b(?:one[- ]time (?:login |verification )?(?:code|password)|otp|verification code)\b.{0,90}\b(?:bank(?:ing)?|account|recruiter|hiring manager)\b"),
)


def candidate_matches(posting):
    """Quote the full sentence so negation and warning context stay visible."""
    matches, seen = [], set()
    for field in TEXT_FIELDS:
        text = getattr(posting, field) or ""
        for sentence in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", text):
            quote = sentence.group().strip()
            for rule in RULES:
                if not re.search(rule.pattern, quote, re.IGNORECASE):
                    continue
                key = (field, quote, rule.category)
                if key in seen:
                    continue
                seen.add(key)
                matches.append(dict(evidence_id=f"risk_{len(matches)+1}", field=field,
                                    quote=quote, category=rule.category, rule_id=rule.rule_id))
    return matches


def rule_requests(candidate):
    """Conservative rules-only baseline, never an auto-reject rule."""
    return guard_text(candidate['quote']) is None


def guard_text(text):
    """A conservative contradiction check, not a general context interpreter."""
    verbs = r"(?:pay|send|share|provide|give|transfer|disclose|forward)"
    if re.search(r"\b(?:not|never|don't|do not)\b.{0,35}\b" + verbs + r"\b", text, re.I):
        return 'negation'
    if re.search(r"\b(?:warnings?|beware|scams?|fraudulent|teach(?:ing)?)\b", text, re.I):
        return 'warning'
    return None


def context_guard(posting, field, quote):
    """Inspect the containing sentence, including text outside a short LLM quote."""
    text = getattr(posting, field) or ''
    for match in re.finditer(re.escape(quote), text):
        prefix = text[:match.start()]
        start = max(prefix.rfind(c) for c in '.!?\n') + 1
        end = match.end()
        if not quote.rstrip().endswith(('.', '!', '?')):
            boundary = re.search(r'[.!?\n]', text[end:])
            end = end + boundary.end() if boundary else len(text)
        guard = guard_text(text[start:end])
        if guard:
            return guard
    return None
