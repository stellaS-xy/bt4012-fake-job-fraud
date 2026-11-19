"""Create real local or cloud Ollama models."""
from agent.core.config import Settings


def create_review_model(settings: Settings):
    from langchain_openai import ChatOpenAI
    options = {"reasoning_effort": "none"} if settings.backend == "ollama" else {}
    return ChatOpenAI(
        model=settings.model, base_url=settings.base_url, api_key=settings.api_key,
        use_responses_api=False, timeout=settings.request_timeout,
        max_retries=0 if settings.backend == 'ollama_local' else 1,
        max_tokens=4096, temperature=0, **options,
    )
