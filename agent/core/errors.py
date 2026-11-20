"""Safe provider error classification; never return raw exception messages."""
ERROR_MESSAGES = {
    "assessment_invalid": "The model did not return a valid final assessment. Review the evidence manually.",
    "credits_required": "Ollama requires usage credits or a subscription for this request. Check your Ollama account billing.",
    "authentication_failed": "Ollama rejected the API key. Check OLLAMA_API_KEY.",
    "access_denied": "This Ollama account does not have access to the requested model.",
    "model_unavailable": "Ollama could not find the configured model. Check OLLAMA_MODEL.",
    "rate_limited": "Ollama's rate or usage limit was reached. Try again later.",
    "provider_timeout": "The Ollama request timed out.",
    "connection_failed": "The app could not connect to Ollama.",
    "provider_request_rejected": "Ollama rejected the request or tool schema.",
    "provider_unavailable": "Ollama returned a server error. Try again later.",
    "review_incomplete": "The review failed validation or exceeded its step limit. Inspect the tool trace and retry.",
}


def error_code_for(exc):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        status = getattr(exc, "status_code", None)
        code = {400: "provider_request_rejected", 401: "authentication_failed", 402: "credits_required", 403: "access_denied",
                404: "model_unavailable", 429: "rate_limited"}.get(status)
        if code:
            return code
        if isinstance(status, int) and status >= 500:
            return "provider_unavailable"
        kind = type(exc).__name__
        if "Timeout" in kind:
            return "provider_timeout"
        if "ConnectionError" in kind:
            return "connection_failed"
        exc = exc.__cause__ or exc.__context__
    return "review_incomplete"
