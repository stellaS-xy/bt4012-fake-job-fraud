"""Model creation. Only explicit dummy mode uses the offline test double."""
from agent.config import Settings


def create_review_model(settings: Settings):
    if settings.backend == "dummy":
        from agent.dummy_model import DummyReviewModel
        return DummyReviewModel()
    from langchain_openai import ChatOpenAI
    options = {"reasoning_effort": "none"} if settings.backend == "ollama" else {}
    return ChatOpenAI(
        model=settings.model, base_url=settings.base_url, api_key=settings.api_key,
        use_responses_api=False, timeout=settings.request_timeout, max_retries=1,
        max_tokens=4096, temperature=0, **options,
    )
