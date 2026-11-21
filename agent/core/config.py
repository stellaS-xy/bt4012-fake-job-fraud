"""Ollama local/cloud configuration; secrets are excluded from repr and exports."""
from dataclasses import dataclass, field
import os
import sys
from pathlib import Path

from dotenv import dotenv_values

AGENT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    backend: str = "ollama_local"
    api_key: str = field(default="", repr=False)
    model: str = ""
    base_url: str = ""
    request_timeout: float = 60.0
    recursion_limit: int = 18
    classifier_backend: str = 'auto'
    classifier_artifact: Path | None = None
    classifier_python: Path | None = None

    def __post_init__(self):
        if self.classifier_backend == 'auto':
            object.__setattr__(self, 'classifier_backend', 'ensemble')
        if self.classifier_backend not in ('ensemble', 'lightgbm'):
            raise ValueError('CLASSIFIER_BACKEND must be ensemble or lightgbm; simulated classifiers are not supported.')
        ensemble = self.classifier_backend == 'ensemble'
        defaults = {
            'classifier_artifact': AGENT_DIR.parent/'models'/('final_ensemble_v1' if ensemble else 'agent_lightgbm'),
            'classifier_python': AGENT_DIR.parent/('.venv-mlp' if ensemble else '.venv')/('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python'),
        }
        for field_name, default in defaults.items():
            value = getattr(self, field_name)
            path = Path(default if value is None else value).expanduser()
            object.__setattr__(self, field_name, path if path.is_absolute() else AGENT_DIR.parent/path)
        if self.backend not in ("ollama", "ollama_local"):
            raise ValueError("AGENT_BACKEND must be ollama_local (local model) or ollama (cloud).")
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
        if self.backend == "ollama" and (not self.api_key.strip() or self.api_key in ("NOT_CONFIGURED", "your-ollama-api-key")):
            raise ValueError("Set OLLAMA_API_KEY in agent/.env or the project-root .env to use AGENT_BACKEND=ollama.")
        if self.backend == "ollama" and self.base_url != "https://ollama.com/v1":
            raise ValueError("This backend uses https://ollama.com/v1; the Ollama key is not sent to other endpoints.")


def load_settings():
    values = {}
    for path in (AGENT_DIR.parent / ".env", AGENT_DIR / ".env"):
        for key, value in dotenv_values(path).items():
            if value is not None and value.strip() and value not in ("NOT_CONFIGURED", "your-ollama-api-key"):
                values[key] = value
    values.update({k: v for k, v in os.environ.items() if k in ("AGENT_BACKEND", "OLLAMA_API_KEY", "OLLAMA_MODEL", 'CLASSIFIER_BACKEND', 'CLASSIFIER_ARTIFACT', 'CLASSIFIER_PYTHON')})
    return Settings(backend=values.get("AGENT_BACKEND", "ollama_local"),
                    api_key=values.get("OLLAMA_API_KEY", ""),
                    model=values.get("OLLAMA_MODEL", ""),
                    classifier_backend=values.get('CLASSIFIER_BACKEND', 'auto'),
                    classifier_artifact=values.get('CLASSIFIER_ARTIFACT'),
                    classifier_python=values.get('CLASSIFIER_PYTHON'))
