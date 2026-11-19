"""Streamlit interface. Launch from the repository root; see frontend/README.md."""
import json
from pathlib import Path
import sys

# Streamlit executes this as a script, so make the sibling package importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from pydantic import ValidationError

from agent.config import load_settings
from agent.fixtures import SCENARIOS
from agent.reviewer import confirm_decision
from agent.runner import run_review
from agent.schemas import Posting
from agent.errors import ERROR_MESSAGES


def clear_case():
    for key in ("case", "saved_review"):
        st.session_state.pop(key, None)


def main():
    st.set_page_config(page_title="Job-ad review demo", page_icon="🔎", layout="wide")
    st.title("Job-ad review assistant")
    try:
        settings = load_settings()
    except ValueError as exc:
        st.error(str(exc))
        st.code("AGENT_BACKEND=ollama\nOLLAMA_MODEL=deepseek-v4.1-flash\nOLLAMA_API_KEY=<your Ollama key>", language="text")
        return
    if settings.backend == "ollama_local":
        st.info(f"Local Ollama model: {settings.model}. Runs on this computer with no Ollama cloud API charges.")
        st.warning("The classifier score is still a placeholder. Risk-pattern matching and LLM context review are real checks, but their accuracy is not yet validated.")
    elif settings.backend == "ollama":
        st.info(f"Live Ollama model: {settings.model}. LangChain sends real model requests and executes the model's tool calls.")
        st.warning("The classifier score is still a placeholder. Risk-pattern matching and LLM context review are real checks, but their accuracy is not yet validated.")
    else:
        st.warning("OFFLINE DEMO — explicit test mode; no hosted language model is called.")
    scenario = st.selectbox("Demo scenario", list(SCENARIOS), format_func=lambda k: SCENARIOS[k]["label"],
                            key="scenario", on_change=clear_case)
    example = SCENARIOS[scenario]["posting"]
    st.caption("The scenario controls the placeholder classifier score. The risk checks analyse the text you submit.")
    with st.form("posting_form"):
        title = st.text_input("Job title", value=example["title"], key=f"title_{scenario}")
        description = st.text_area("Job advertisement", value=example["description"], height=180,
                                   key=f"description_{scenario}")
        profile = st.text_area("Company profile, if supplied", value=example["company_profile"] or "",
                               key=f"profile_{scenario}")
        salary = st.text_input("Salary range, if supplied", value=example["salary_range"] or "",
                               key=f"salary_{scenario}")
        requirements = st.text_area("Requirements, if supplied", key=f"requirements_{scenario}")
        benefits = st.text_area("Benefits, if supplied", key=f"benefits_{scenario}")
        submitted = st.form_submit_button("Review posting", type="primary")
    if submitted:
        clear_case()
        try:
            posting = Posting(title=title, description=description, company_profile=profile,
                              salary_range=salary, requirements=requirements, benefits=benefits)
            with st.spinner("Running the LangChain review and tool calls…"):
                st.session_state.case = run_review(posting, scenario, settings)
        except ValidationError:
            st.error("Enter a non-empty job title and advertisement. Keep the title under 500 characters and the advertisement under 20,000 characters.")
    case = st.session_state.get("case")
    if case is not None and (getattr(case, "schema_version", 0) != 3 or case.model_backend != settings.backend):
        clear_case()
        case = None
    if case is None:
        return
    a = case.assessment
    if case.status == "fallback":
        st.error(f"Review incomplete: {ERROR_MESSAGES.get(case.error_code, ERROR_MESSAGES['review_incomplete'])} No offline model was substituted; any scores below came only from placeholder tools.")
        if settings.backend == "ollama_local":
            st.caption("Keep the Ollama app running and check that ollama list includes the configured model. Local models can take longer to respond.")
    st.divider()
    st.subheader(f"Review: {case.posting.title}")
    st.caption("Results belong to the last submitted posting below. Submit the form again after editing.")
    left, middle, right = st.columns(3)
    left.metric("Placeholder classifier score", "Unavailable" if a.model_score is None else f"{a.model_score:.2f}")
    middle.metric("Placeholder cutoff", "Unavailable" if a.cutoff is None else f"{a.cutoff:.2f}")
    right.metric("Placeholder classifier flag", "Unavailable" if a.flagged is None else ("Flagged" if a.flagged else "Below cutoff"))
    st.caption("A below-cutoff dummy score does not clear an upstream flag or establish that the advertisement is safe.")
    labels = {"propose_rejection": "Propose rejection — reviewer confirmation required",
              "escalate_to_review": "Escalate for review"}
    st.info(labels[a.recommendation])
    st.write(a.rationale)
    lexical = next((t.output for t in case.tool_trace if t.name == "match_lexicon" and t.status == "success"), {})
    if lexical.get("context_status") in ("failed", "not_run"):
        st.warning("LLM context review did not complete. Risk-pattern matches are unconfirmed and require human review.")
        if lexical.get('context_error'):
            st.caption(ERROR_MESSAGES.get(lexical['context_error'], ERROR_MESSAGES['review_incomplete']))
    st.subheader("Risk findings and context")
    if not case.evidence:
        st.write("No supported evidence was returned. A reviewer must inspect the posting.")
    for item in case.evidence:
        if item.source == "dataset_lexicon":
            continue
        with st.expander(f"[{item.evidence_id}] {item.field.replace('_', ' ')} · {item.kind.replace('_', ' ')}"):
            st.write(item.observation)
            if item.quote is not None:
                st.text(item.quote)
            st.caption(f"Context: {item.context}. Reviewed by: {item.reviewed_by}.")
            if item.context_guard:
                st.caption(f"Application check: {item.context_guard} wording detected. Original model interpretation: {item.model_interpretation}.")
            st.caption(f"Source: {item.source}. Contextual evidence is not a verified cause of fraud.")
    background = [e for e in case.evidence if e.source == "dataset_lexicon"]
    if background:
        with st.expander("Historical dataset terms — background only"):
            st.caption("These words were associated with labels in the old dataset. They do not support rejection.")
            st.json([e.model_dump() for e in background])
    st.subheader("Not verified / limitations")
    for limitation in a.limitations:
        st.write(f"• {limitation}")
    st.subheader("Suggested next steps")
    for step in a.next_steps:
        st.write(f"• {step}")
    with st.expander("Posting used for this assessment"):
        st.json(case.posting.model_dump())
    with st.expander("Model-issued tool calls"):
        st.json(case.llm_tool_calls)
    with st.expander("Executed tool-call trace"):
        st.json([event.model_dump() for event in case.tool_trace])
    st.subheader("Reviewer decision")
    st.caption("Saving records a local demonstration decision only. No advertisement is removed or rejected on another platform.")
    reviewer = st.text_input("Reviewer name", key="reviewer_name")
    decision = st.selectbox("Decision", ["escalated_to_review", "confirmed_rejection"],
                            format_func=lambda value: value.replace("_", " ").capitalize())
    saved = st.session_state.get("saved_review")
    if st.button("Confirm and save local decision", disabled=saved is not None):
        try:
            saved = confirm_decision(case, decision, reviewer, settings.audit_path)
            st.session_state.saved_review = saved
            st.success("Demonstration decision saved locally.")
        except ValueError as exc:
            st.error(str(exc))
        except OSError:
            st.error("The audit record could not be saved. No decision was recorded; check the local runtime folder.")
    if saved is not None:
        st.write("Saved reviewer decision:")
        st.json(saved["reviewer_decision"])
    export = saved if saved is not None else {"case": case.model_dump(mode="json"), "reviewer_decision": None}
    st.download_button("Download case report", data=json.dumps(export, indent=2, ensure_ascii=False),
                       file_name=f"review-{case.posting.posting_id}.json", mime="application/json")


if __name__ == "__main__":
    main()
