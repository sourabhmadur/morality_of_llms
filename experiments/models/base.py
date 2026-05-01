from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class ModelResponse:
    model_id: str
    scenario_id: str
    raw_response: str
    judgment: Optional[str] = None
    judgment_binary: Optional[int] = None
    confidence: Optional[int] = None
    primary_framework: Optional[str] = None
    reasoning: Optional[str] = None
    key_principle: Optional[str] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    parse_error: Optional[str] = None
    # Hidden chain-of-thought / thinking trace, when the provider exposes it.
    # Anthropic + Together → full trace; Gemini + OpenAI → summary; instant mode → None.
    reasoning_trace: Optional[str] = None
    # Number of reasoning/thinking tokens used per provider's accounting.
    # Used to verify and document apples-to-apples reasoning effort across providers.
    reasoning_tokens: Optional[int] = None


PROMPT_TEMPLATE = """You are participating in a moral philosophy research study. Please analyze the following scenario carefully and provide a structured response in valid JSON.

SCENARIO:
{description}

QUESTION:
{question}

Please respond with a JSON object containing exactly these fields:
{{
  "judgment": "<your decision or position, e.g., 'yes', 'no', 'permissible', 'not permissible', or a specific choice>",
  "judgment_binary": <1 if you judge the action/policy as morally permissible/required/justified, 0 if not>,
  "confidence": <integer 1-5, where 1=very uncertain, 5=very confident in your judgment>,
  "primary_framework": "<exactly one of: utilitarian, deontological, virtue_ethics, care_ethics, contractualist, other>",
  "reasoning": "<2-3 sentences explaining the core of your moral judgment>",
  "key_principle": "<one sentence stating the single most important ethical principle you applied>"
}}

Respond ONLY with the JSON object. No preamble, no explanation outside the JSON."""


VALID_FRAMEWORKS = {
    "utilitarian", "deontological", "virtue_ethics",
    "care_ethics", "contractualist", "other"
}


class BaseModelClient(ABC):
    def __init__(self, model_id: str):
        self.model_id = model_id

    def query_scenario(self, scenario: dict) -> ModelResponse:
        prompt = PROMPT_TEMPLATE.format(
            description=scenario["description"],
            question=scenario["question"]
        )
        t0 = time.time()
        try:
            api_out = self._call_api(prompt)
            latency = (time.time() - t0) * 1000
            # _call_api may return:
            #   - plain str (final answer only)
            #   - (text, trace)
            #   - (text, trace, reasoning_tokens)
            if isinstance(api_out, tuple):
                if len(api_out) == 3:
                    raw, trace, rt = api_out
                else:
                    raw, trace = api_out
                    rt = None
            else:
                raw, trace, rt = api_out, None, None
            response = ModelResponse(
                model_id=self.model_id,
                scenario_id=scenario["id"],
                raw_response=raw,
                latency_ms=latency,
                reasoning_trace=trace,
                reasoning_tokens=rt,
            )
            self._parse_response(response, raw)
        except Exception as e:
            latency = (time.time() - t0) * 1000
            response = ModelResponse(
                model_id=self.model_id,
                scenario_id=scenario["id"],
                raw_response="",
                latency_ms=latency,
                error=str(e)
            )
        return response

    @abstractmethod
    def _call_api(self, prompt: str):
        """Return either a plain string (final answer) or a (final_answer, reasoning_trace) tuple."""
        pass

    def _parse_response(self, response: ModelResponse, raw: str) -> None:
        import json
        import re
        try:
            # Strip markdown code fences if present
            cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
            # Find the first { ... } block
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not match:
                response.parse_error = "No JSON object found"
                return
            data = json.loads(match.group())
            response.judgment = str(data.get("judgment", "")).strip()
            jb = data.get("judgment_binary")
            response.judgment_binary = int(jb) if jb is not None else None
            conf = data.get("confidence")
            response.confidence = int(conf) if conf is not None else None
            fw = str(data.get("primary_framework", "")).strip().lower()
            response.primary_framework = fw if fw in VALID_FRAMEWORKS else "other"
            response.reasoning = str(data.get("reasoning", "")).strip()
            response.key_principle = str(data.get("key_principle", "")).strip()
        except Exception as e:
            response.parse_error = f"Parse error: {e} | raw: {raw[:200]}"
