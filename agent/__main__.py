"""Headless review: python -m agent --scenario suspicious."""
import argparse

from agent.core.config import load_settings
from agent.resources.fixtures import SCENARIOS
from agent.core.runner import run_review
from agent.core.schemas import Posting


def main():
    parser = argparse.ArgumentParser(description="LangChain job-ad recommendations using Ollama, LightGBM and lexicon context checks.")
    parser.add_argument("--backend", choices=("ollama", "ollama_local"), help="Select Ollama cloud or local Ollama.")
    parser.add_argument("--scenario", choices=SCENARIOS, default="suspicious")
    parser.add_argument("--description", help="Replace the sample description; real classifier mode scores the submitted text.")
    args = parser.parse_args()
    values = dict(SCENARIOS[args.scenario]["posting"])
    if args.description is not None:
        values["description"] = args.description
    from dataclasses import replace
    # An explicit local run does not require a cloud API key.
    from agent.core.config import Settings
    settings = Settings(backend=args.backend) if args.backend == "ollama_local" else load_settings()
    if args.backend is not None:
        settings = replace(settings, backend=args.backend)
    case = run_review(Posting(**values), args.scenario, settings)
    print(case.model_dump_json(indent=2))
    if case.status == "fallback":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
