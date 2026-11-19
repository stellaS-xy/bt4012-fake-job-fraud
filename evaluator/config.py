"""Independent judge settings; local judging is the default."""
import os
import math
from pathlib import Path
from dotenv import dotenv_values
from agent.core.config import Settings

EVALUATOR_DIR = Path(__file__).resolve().parent


def load_judge_settings():
    values = {}
    for path in (EVALUATOR_DIR.parent / '.env', EVALUATOR_DIR / '.env'):
        values.update({k: v for k, v in dotenv_values(path).items() if v and v.strip()})
    values.update({k: v for k, v in os.environ.items() if k.startswith('JUDGE_')})
    backend = values.get('JUDGE_BACKEND', 'ollama_local')
    if backend not in ('ollama_local', 'ollama'):
        raise ValueError('JUDGE_BACKEND must be ollama_local or ollama; the judge must be a real model.')
    if backend == 'ollama' and not values.get('JUDGE_API_KEY'):
        raise ValueError('Set JUDGE_API_KEY in evaluator/.env for a cloud judge.')
    timeout = float(values.get('JUDGE_TIMEOUT_SECONDS', '180' if backend == 'ollama_local' else '60'))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('JUDGE_TIMEOUT_SECONDS must be a finite positive number.')
    return Settings(backend=backend, model=values.get('JUDGE_MODEL', ''),
                    api_key=values.get('JUDGE_API_KEY', ''), request_timeout=timeout)
