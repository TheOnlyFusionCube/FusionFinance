"""Claude provider for the originator and the analyst desk.

Uses the official ``anthropic`` SDK (optional extra: ``pip install -e '.[claude]'``)
with structured outputs, so each response is one JSON object matching the
role's schema. The response is still treated as untrusted: it is re-validated
by the strict pydantic contracts, the evidence audit, and the ML gate.

Refusals are handled explicitly, and requests opt into the server-side
``fallbacks: "default"`` so a declined request is retried on Anthropic's
recommended fallback model instead of silently failing the idea.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from alpha.agents.models import ANALYST_ROLES, AnalystRequest
from alpha.agents.providers import ProviderError, analyst_system_prompt
from alpha.fund.ideas import IdeaRequest, originator_system_prompt
from alpha.verifier.contract import CLAIM_TARGET

DEFAULT_MODEL = "claude-opus-5-5"
Effort = Literal["low", "medium", "high", "xhigh", "max"]

_STRINGS = {"type": "array", "items": {"type": "string"}}
ANALYST_SCHEMA = {
    "type": "object",
    "properties": {
        "role": {"type": "string", "enum": list(ANALYST_ROLES)},
        "direction": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "confidence": {"type": "number"},
        "summary": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"},
                    "quoted_text": {"type": "string"},
                    "numeric_key": {"type": "string"},
                    "asserted_value": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                },
                "required": ["document_id", "quoted_text", "numeric_key", "asserted_value"],
                "additionalProperties": False,
            },
        },
        "causal_chain": _STRINGS,
        "falsifiers": _STRINGS,
        "risk_flags": _STRINGS,
        "veto": {"type": "boolean"},
    },
    "required": [
        "role", "direction", "confidence", "summary", "citations",
        "causal_chain", "falsifiers", "risk_flags", "veto",
    ],
    "additionalProperties": False,
}
IDEA_SCHEMA = {
    "type": "object",
    "properties": {
        "ticker": {"type": "string"},
        "action": {"type": "string", "enum": ["long", "short", "pass"]},
        "claim_type": {"type": "string", "enum": sorted(CLAIM_TARGET)},
        "horizon_days": {"type": "integer", "enum": [1, 3, 5, 10]},
        "expected_move_bps": {"type": "number"},
        "conviction": {"type": "number"},
        "thesis": {"type": "string"},
        "catalysts": _STRINGS,
        "falsifiers": _STRINGS,
        "cited_document_ids": _STRINGS,
    },
    "required": [
        "ticker", "action", "claim_type", "horizon_days", "expected_move_bps",
        "conviction", "thesis", "catalysts", "falsifiers", "cited_document_ids",
    ],
    "additionalProperties": False,
}


@dataclass
class ClaudeProvider:
    """One provider object serves both ``originate`` and ``analyze``."""

    model: str = DEFAULT_MODEL
    effort: Effort = "medium"
    max_tokens: int = 16_000
    timeout_seconds: float = 120.0
    client: object | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.model or any(ch.isspace() for ch in self.model):
            raise ValueError("model id is invalid")
        if self.client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise ProviderError(
                    "install the optional Claude extra: pip install -e '.[claude]'"
                ) from exc
            self.client = anthropic.Anthropic(timeout=self.timeout_seconds)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "ClaudeProvider":
        values = os.environ if environ is None else environ
        return cls(
            model=values.get("FUSION_CLAUDE_MODEL", DEFAULT_MODEL),
            effort=values.get("FUSION_CLAUDE_EFFORT", "medium"),  # type: ignore[arg-type]
        )

    @property
    def model_id(self) -> str:
        return f"anthropic:{self.model}"

    def originate(self, request: IdeaRequest) -> str:
        return self._call(originator_system_prompt(), request.model_dump_json(), IDEA_SCHEMA)

    def analyze(self, request: AnalystRequest) -> str:
        instruction = (
            f"{analyst_system_prompt()} You are the {request.role} analyst. Quote only "
            "sentences that appear verbatim in documents eligible for your role, and "
            "use numeric_key values exactly as listed in those documents. Use an empty "
            "numeric_key and null asserted_value when citing text only."
        )
        return self._call(instruction, request.model_dump_json(), ANALYST_SCHEMA)

    def _call(self, system: str, payload: str, schema: dict) -> str:
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": payload}],
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
                betas=["server-side-fallback-2026-07-01"],
                extra_body={"fallbacks": "default"},
            )
        except Exception:
            raise ProviderError("Claude request failed") from None
        if getattr(response, "stop_reason", None) == "refusal":
            raise ProviderError("Claude declined the request")
        if getattr(response, "stop_reason", None) == "max_tokens":
            raise ProviderError("Claude response was truncated")
        text = next(
            (block.text for block in response.content if getattr(block, "type", "") == "text"), ""
        )
        if not text:
            raise ProviderError("Claude returned no text content")
        json.loads(text)  # fail fast on malformed JSON; strict validation happens downstream
        return text


__all__ = ["ANALYST_SCHEMA", "ClaudeProvider", "DEFAULT_MODEL", "IDEA_SCHEMA"]
