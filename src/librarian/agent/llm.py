"""LLM access: loaded-model preference + validated JSON completions via LiteLLM.

ModelSelector: asks Ollama's /api/ps (cached) which candidate from
config.librarian_models is already in VRAM and picks that one (no model
swap). If none is loaded, the first in the list is loaded on purpose.
If Ollama is unreachable, the LiteLLM fallback alias applies
(librarian-chat: local -> cloud-pool -> copilot).

chat_json: a narrowly scoped completion call, Pydantic-validated, with
error-feedback retries. Raises LlmUnavailable if everything fails - the
caller ALWAYS has a deterministic fallback.
"""
from __future__ import annotations

import json
import re
import time
from typing import TypeVar

import httpx
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from ..config import Settings

T = TypeVar("T", bound=BaseModel)


class LlmUnavailable(Exception):
    pass


class ModelSelector:
    def __init__(self, settings: Settings):
        self.s = settings
        self._cached: tuple[float, str] | None = None

    def _loaded_models(self) -> list[str]:
        r = httpx.get(f"{self.s.env.ollama_base_url}/api/ps", timeout=5)
        r.raise_for_status()
        return [m.get("name", "") for m in r.json().get("models", [])]

    def pick(self) -> str:
        """LiteLLM model_name of the best candidate (cached, model_ps_ttl)."""
        # Emergency override: FORCE_MODEL in .env disables local selection
        # entirely (e.g. cloud-pool when ComfyUI/training blocks the GPU)
        if self.s.env.force_model:
            return self.s.env.force_model
        now = time.monotonic()
        if self._cached and now - self._cached[0] < self.s.cfg.model_ps_ttl:
            return self._cached[1]
        cfg = self.s.cfg
        try:
            loaded = self._loaded_models()
            chosen_tag = next(
                (tag for tag in cfg.librarian_models if tag in loaded),
                cfg.librarian_models[0],  # none loaded -> load the favorite on purpose
            )
            model = cfg.litellm_model_map.get(chosen_tag, cfg.litellm_fallback_model)
        except Exception:
            # Ollama unreachable -> LiteLLM's fallback chain takes over
            model = cfg.litellm_fallback_model
        self._cached = (now, model)
        return model


def _extract_json(text: str) -> str:
    """Peel JSON out of an LLM reply (tolerate thinking remnants, code fences)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start:end + 1]
    return text.strip()


class Llm:
    def __init__(self, settings: Settings, selector: ModelSelector | None = None):
        self.s = settings
        self.selector = selector or ModelSelector(settings)
        self.client = OpenAI(
            base_url=settings.env.litellm_base_url,
            api_key=settings.env.litellm_api_key,
            timeout=settings.cfg.llm_timeout,
        )

    def chat_json(self, system: str, user: str, schema: type[T],
                  model: str | None = None) -> T:
        """One completion call with a hard schema. Retries with error feedback.

        model: explicit LiteLLM model name (for A/B tests); default = selector.
        """
        model = model or self.selector.pick()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_err = "empty"
        for attempt in range(1 + self.s.cfg.llm_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                raw = (resp.choices[0].message.content or "").strip()
            except Exception as e:
                last_err = f"API error: {e}"
                continue
            if not raw:
                last_err = "empty response"  # known qwen3 thinking symptom
            else:
                try:
                    return schema.model_validate_json(_extract_json(raw))
                except (ValidationError, json.JSONDecodeError, ValueError) as e:
                    last_err = f"invalid JSON: {e}"
            # error feedback for the retry
            messages.append({"role": "assistant", "content": raw or "(empty)"})
            messages.append({
                "role": "user",
                "content": f"Your reply was unusable ({last_err}). Reply ONLY with "
                           f"valid JSON matching the required schema, no explanatory text.",
            })
        raise LlmUnavailable(f"LLM call failed after retries: {last_err}")
