import os
import anthropic
from .base import BaseModelClient


class AnthropicClient(BaseModelClient):
    """Claude Sonnet 4.6 with adaptive thinking. Captures full thinking trace."""

    def __init__(self, model_id: str = "claude-sonnet-4-6", mode: str = "light",
                 effort: str = "low", thinking_budget: int = 1024):
        super().__init__(model_id)
        self.mode = mode
        self.effort = effort
        self.thinking_budget = thinking_budget
        self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def _call_api(self, prompt: str):
        kwargs = {
            "model": self.model_id,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.mode == "light":
            # Use explicit "enabled" thinking with a fixed budget — guaranteed
            # to produce a non-empty thinking trace, unlike adaptive which can
            # decide to skip thinking on simple-looking prompts.
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_budget}
        msg = self._client.messages.create(**kwargs)
        text_parts = []
        thinking_parts = []
        for block in msg.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "thinking":
                thinking_parts.append(getattr(block, "thinking", "") or "")
        text = "".join(text_parts)
        trace = "\n\n".join(t for t in thinking_parts if t) if thinking_parts else None
        # IMPORTANT: Anthropic does NOT expose a separate thinking_tokens count.
        # The configured budget_tokens=N is an upper bound for what may be charged
        # to output_tokens, NOT what the model actually thinks/exposes. The
        # visible thinking text is a redacted SUMMARY of the model's thinking, often
        # just a short descriptive sentence. We therefore report the chars/4 estimate
        # of the *visible* thinking text — i.e., what's actually auditable from
        # outside. This is the same convention used for Together-AI providers.
        # Caveat: this likely undercounts the model's true internal reasoning effort.
        rt = (sum(len(t) for t in thinking_parts) // 4) if (self.mode == "light" and thinking_parts) else 0
        return (text, trace, rt)
