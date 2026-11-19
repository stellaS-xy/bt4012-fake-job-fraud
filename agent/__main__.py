"""Headless review: python -m agent --scenario suspicious."""
import argparse

from agent.config import load_settings
from agent.fixtures import SCENARIOS
from agent.runner import run_review
from agent.schemas import Posting


def main():
    parser = argparse.ArgumentParser(description="LangChain review using Ollama cloud with placeholder business tools.")
    parser.add_argument("--backend", choices=("ollama", "ollama_local", "dummy"), help="Select cloud, local Ollama, or offline tests.")
    parser.add_argument("--scenario", choices=SCENARIOS, default="suspicious")
    parser.add_argument("--description", help="Replace the sample description; the selected score remains dummy.")
    args = parser.parse_args()
    values = dict(SCENARIOS[args.scenario]["posting"])
    if args.description is not None:
        values["description"] = args.description
    from dataclasses import replace
    # An explicit offline test must not require a cloud API key.
    from agent.config import Settings
    settings = Settings(backend=args.backend) if args.backend in ("dummy", "ollama_local") else load_settings()
    if args.backend is not None:
        settings = replace(settings, backend=args.backend)
    case = run_review(Posting(**values), args.scenario, settings)
    print(case.model_dump_json(indent=2))
    if case.status == "fallback":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
