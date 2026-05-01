import os
from .base import BaseModelClient


class OpenAIClient(BaseModelClient):
    """GPT-5.x reasoning models via the Responses API. Captures reasoning summary
    when thinking is enabled (raw CoT is not exposed by OpenAI policy)."""

    def __init__(self, model_id: str = "gpt-5.5", mode: str = "light"):
        super().__init__(model_id)
        self.mode = mode
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")

    def _call_api(self, prompt: str):
        # NOTE on apples-to-apples: GPT-5.5 with effort='low' produces ZERO
        # reasoning tokens on moral-prompt inputs (verified against
        # usage.output_tokens_details.reasoning_tokens). To get a non-trivial
        # reasoning signal, the 'light' condition uses effort='medium'.
        # This is documented as a known asymmetry in the paper: each provider
        # exposes the lightest non-zero reasoning level differently.
        effort = "medium" if self.mode == "light" else "none"
        reasoning = {"effort": effort}
        if self.mode == "light":
            reasoning["summary"] = "detailed"
        response = self._client.responses.create(
            model=self.model_id,
            input=prompt,
            reasoning=reasoning,
            max_output_tokens=4096,
        )
        text = response.output_text or ""
        trace_parts = []
        for item in (response.output or []):
            if getattr(item, "type", None) == "reasoning":
                for s in getattr(item, "summary", []) or []:
                    txt = getattr(s, "text", None)
                    if txt:
                        trace_parts.append(txt)
        trace = "\n\n".join(trace_parts) if trace_parts else None
        # OpenAI exposes reasoning_tokens directly via usage.output_tokens_details
        rt = 0
        try:
            rt = getattr(response.usage.output_tokens_details, "reasoning_tokens", 0) or 0
        except AttributeError:
            pass
        return (text, trace, rt)
