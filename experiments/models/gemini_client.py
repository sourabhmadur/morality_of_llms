import os
from .base import BaseModelClient


class GeminiClient(BaseModelClient):
    """Gemini 3.x via google-genai using thinking_level. Captures thought summaries
    when thinking is enabled."""

    def __init__(self, model_id: str = "gemini-3-flash-preview", mode: str = "light"):
        super().__init__(model_id)
        self.mode = mode
        try:
            from google import genai
            self._client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        except ImportError:
            raise RuntimeError("google-genai package not installed. Run: pip install google-genai")

    def _call_api(self, prompt: str):
        from google.genai import types
        level = "low" if self.mode == "light" else "minimal"
        response = self._client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=4096,
                thinking_config=types.ThinkingConfig(
                    thinking_level=level,
                    include_thoughts=(self.mode == "light"),
                ),
            ),
        )
        # Walk the response parts to separate thought summaries from final text
        text_parts = []
        thought_parts = []
        try:
            for cand in (response.candidates or []):
                for part in (cand.content.parts or []):
                    txt = getattr(part, "text", None)
                    if txt is None:
                        continue
                    if getattr(part, "thought", False):
                        thought_parts.append(txt)
                    else:
                        text_parts.append(txt)
        except AttributeError:
            pass
        text = "".join(text_parts) if text_parts else (response.text or "")
        trace = "\n\n".join(thought_parts) if thought_parts else None
        # Gemini exposes thoughts_token_count in usage_metadata
        rt = 0
        try:
            rt = getattr(response.usage_metadata, "thoughts_token_count", 0) or 0
        except AttributeError:
            pass
        return (text, trace, rt)
