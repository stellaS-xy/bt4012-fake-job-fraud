"""Capture provider-issued business tool calls, without prompts or credentials."""
from langchain_core.callbacks import BaseCallbackHandler
from agent.core.middleware import REQUIRED_TOOLS


class ToolCallRecorder(BaseCallbackHandler):
    def __init__(self):
        self.calls = []

    def on_llm_end(self, response, **kwargs):
        for batch in response.generations:
            for generation in batch:
                message = getattr(generation, "message", None)
                for call in getattr(message, "tool_calls", []):
                    if call["name"] in REQUIRED_TOOLS:
                        self.calls.append({"name": call["name"], "args": call["args"], "id": call["id"]})

    def attach_ids(self, trace):
        remaining = list(self.calls)
        for event in trace:
            match = next((c for c in remaining if c["name"] == event.name and c["args"] == event.arguments), None)
            if match:
                event.call_id = match["id"]
                remaining.remove(match)
