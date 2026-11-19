"""Ollama local/cloud configuration; secrets are excluded from repr and exports."""
from dataclasses import dataclass, field
import os
from pathlib import Path

from dotenv import dotenv_values

AGENT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    backend: str = "ollama"
    api_key: str = field(default="", repr=False)
    model: str = ""
    base_url: str = ""
    request_timeout: float = 60.0
    recursion_limit: int = 18
    audit_path: Path = AGENT_DIR / "runtime" / "reviews.jsonl"

    def __post_init__(self):
        if self.backend not in ("dummy", "ollama", "ollama_local"):
            raise ValueError("AGENT_BACKEND must be ollama_local (local model), ollama (cloud), or dummy (offline tests).")
        local = self.backend == "ollama_local"
        if not self.model:
            object.__setattr__(self, "model", "qwen3:4b-instruct" if local else "deepseek-v4.1-flash")
        if not self.base_url:
            object.__setattr__(self, "base_url", "http://localhost:11434/v1" if local else "https://ollama.com/v1")
        if local:
            if self.base_url != "http://localhost:11434/v1":
                raise ValueError("Local mode uses http://localhost:11434/v1 only.")
            if self.model.endswith("-cloud") or ":cloud" in self.model:
                raise ValueError("Choose a downloaded local model, not a cloud model.")
            # Never send the saved cloud credential to the local service.
            object.__setattr__(self, "api_key", "ollama")
        if self.backend == "ollama" and (not self.api_key.strip() or self.api_key in ("DUMMY_NOT_USED", "your-ollama-api-key")):
            raise ValueError("Set OLLAMA_API_KEY in agent/.env or the project-root .env to use AGENT_BACKEND=ollama.")
        if self.backend == "ollama" and self.base_url != "https://ollama.com/v1":
            raise ValueError("This backend uses https://ollama.com/v1; the Ollama key is not sent to other endpoints.")


def load_settings():
    values = {}
    for path in (AGENT_DIR.parent / ".env", AGENT_DIR / ".env"):
        for key, value in dotenv_values(path).items():
            if value is not None and value.strip() and value not in ("DUMMY_NOT_USED", "your-ollama-api-key"):
                values[key] = value
    values.update({k: v for k, v in os.environ.items() if k in ("AGENT_BACKEND", "OLLAMA_API_KEY", "OLLAMA_MODEL")})
    return Settings(backend=values.get("AGENT_BACKEND", "ollama"),
                    api_key=values.get("OLLAMA_API_KEY", ""),
                    model=values.get("OLLAMA_MODEL", ""))
