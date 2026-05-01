import os
import re
from .base import BaseModelClient


class TogetherClient(BaseModelClient):
    """Hybrid open-weight reasoning models via Together AI. Captures full CoT trace
    from message.reasoning when thinking is enabled."""

    def __init__(self, model_id: str, mode: str = "light"):
        super().__init__(model_id)
        self.mode = mode
        try:
            from together import Together
            self._client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        except ImportError:
            raise RuntimeError("together package not installed. Run: pip install together")

    def _call_api(self, prompt: str):
        enabled = (self.mode == "light")
        # Together's API has no separate reasoning-token cap; max_tokens caps the
        # combined reasoning+answer output. We use 4096 (matching the closed-API
        # providers) so the JSON answer is never truncated. Reasoning tokens
        # used are tracked per call and reported in the paper to document
        # any per-provider asymmetry.
        kwargs = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
            "temperature": 0.0,
            "extra_body": {"reasoning": {"enabled": enabled}},
        }
        response = self._client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        text = msg.content or ""
        # Defense in depth: strip any leaked thinking blocks if a model
        # ignores the toggle and inlines them into content.
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL).strip()
        # Together exposes the trace separately at message.reasoning
        trace = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None) or None
        if trace and not trace.strip():
            trace = None
        # Together does not separately count reasoning tokens; estimate from char count
        # (rough: 1 token ≈ 4 chars for English). Documented as estimate in the paper.
        rt = (len(trace) // 4) if trace else 0
        return (text, trace, rt)
