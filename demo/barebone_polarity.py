"""Frozen point-in-time polarity for the Barebone narrative arm.

The scorer is a sealed lexicon. It reads text that was already available at
``available_ts`` and does not call a live model. Scores are gitignored. The
events digest stays the locked Hacker News bind.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from demo.barebone_comparison import (
    BAREBONE_EXPERIMENT_ID,
    BAREBONE_UNIVERSE,
    LOCKED_NARRATIVE_SHA256,
    NARRATIVE_LEXICON,
    NARRATIVE_POLARITY_PROVENANCE,
    NARRATIVE_SCORES,
    load_barebone_comparison_config,
    validate_barebone_payload,
)
from demo.barebone_narrative import decision_session_for, sessions_from_ohlcv

POLARITY_MODEL_ID = "fusionfinance-narrative-lexicon-v1"
POLARITY_AGGREGATION = "mean_polarity_minus_cross_sectional_median"
POLARITY_SKILL_THRESHOLD = 0.0
_NEGATION_WINDOW = 3
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_PROVENANCE_SCHEMA = "fusionfinance-barebone-narrative-polarity-v1"
_BANNED_FIELDS = frozenset(
    {
        "total_return",
        "forward_return",
        "next_return",
        "residual",
        "label",
        "sharpe_ratio",
        "sortino_ratio",
        "annualized_return",
    }
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class PolarityLexicon:
    """Sealed word lists. The file bytes are the model."""

    model_id: str
    negation_window: int
    negations: frozenset[str]
    weights: Mapping[str, int]
    sha256: str


def load_polarity_lexicon(path: Path | None = None) -> PolarityLexicon:
    """Load the frozen lexicon. A drifted rule or a return label raises."""

    lexicon_path = path or (_repo_root() / NARRATIVE_LEXICON)
    payload_bytes = lexicon_path.read_bytes()
    payload = json.loads(payload_bytes.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("polarity lexicon must be a JSON object")
    if payload.get("schema") != POLARITY_MODEL_ID or payload.get("model_id") != POLARITY_MODEL_ID:
        raise ValueError("polarity model id is not the frozen lexicon")
    if payload.get("comparable_performance_claim") is not False:
        raise ValueError("comparable_performance_claim must be false")
    if payload.get("aggregation") != POLARITY_AGGREGATION:
        raise ValueError("polarity aggregation is not the frozen cross-sectional median")
    if payload.get("skill_threshold") != POLARITY_SKILL_THRESHOLD:
        raise ValueError("polarity skill threshold is locked at zero")
    if payload.get("negation_window") != _NEGATION_WINDOW:
        raise ValueError("polarity negation window is locked at 3 tokens")
    if _BANNED_FIELDS.intersection(payload):
        raise ValueError("polarity lexicon must not carry a return label")
    negations = payload.get("negations")
    positive = payload.get("positive")
    negative = payload.get("negative")
    if not isinstance(negations, list) or not isinstance(positive, list) or not isinstance(negative, list):
        raise ValueError("polarity lexicon lists are missing")
    negation_set = frozenset(str(token) for token in negations)
    weights: dict[str, int] = {}
    for token in positive:
        name = str(token)
        if name in weights or name in negation_set:
            raise ValueError(f"polarity token {name} is repeated or negated")
        weights[name] = 1
    for token in negative:
        name = str(token)
        if name in weights or name in negation_set:
            raise ValueError(f"polarity token {name} is repeated or negated")
        weights[name] = -1
    if not weights:
        raise ValueError("polarity lexicon has no tokens")
    return PolarityLexicon(
        model_id=POLARITY_MODEL_ID,
        negation_window=_NEGATION_WINDOW,
        negations=negation_set,
        weights=weights,
        sha256=hashlib.sha256(payload_bytes).hexdigest(),
    )


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN.findall(text.lower()))


def _negated(tokens: Sequence[str], index: int, lexicon: PolarityLexicon) -> bool:
    start = max(0, index - lexicon.negation_window)
    for token in tokens[start:index]:
        if token in lexicon.negations or token.endswith("n't"):
            return True
    return False


def score_text(text: str, lexicon: PolarityLexicon | None = None) -> float:
    """Mean signed lexicon hit in ``[-1, 1]``. No hits score 0."""

    model = lexicon or load_polarity_lexicon()
    tokens = _tokens(text)
    signed = 0
    hits = 0
    for index, token in enumerate(tokens):
        weight = model.weights.get(token)
        if weight is None:
            continue
        value = -weight if _negated(tokens, index, model) else weight
        signed += value
        hits += 1
    if hits == 0:
        return 0.0
    polarity = signed / hits
    if polarity > 1.0:
        return 1.0
    if polarity < -1.0:
        return -1.0
    return float(polarity)


def _aware_utc(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"available_ts is not a timestamp: {value}") from exc
    if parsed.tzinfo is None:
        raise ValueError("available_ts must be timezone-aware UTC")
    return parsed.astimezone(timezone.utc)


def score_events(
    events: Sequence[Mapping[str, object]],
    sessions: Sequence[date],
    lexicon: PolarityLexicon | None = None,
) -> list[dict[str, object]]:
    """Score events already mapped to a later session. Look-ahead raises."""

    model = lexicon or load_polarity_lexicon()
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for event in events:
        if not isinstance(event, Mapping):
            raise ValueError("narrative event must be an object")
        banned = _BANNED_FIELDS.intersection(event)
        if banned:
            raise ValueError("narrative event carries a return label: " + ", ".join(sorted(banned)))
        if event.get("available_ts") in (None, ""):
            raise ValueError("narrative event is missing available_ts")
        available = _aware_utc(str(event["available_ts"]))
        ticker = str(event.get("ticker", "")).strip().upper()
        if ticker not in BAREBONE_UNIVERSE:
            raise ValueError(f"narrative event ticker {ticker or '(blank)'} is not in the sealed map universe")
        session_text = event.get("decision_session")
        if not isinstance(session_text, str) or not session_text:
            raise ValueError("narrative event is missing decision_session")
        session = date.fromisoformat(session_text)
        expected = decision_session_for(available, sessions)
        if expected is None or session != expected:
            raise ValueError("decision_session must be the first session strictly after available_ts")
        if session <= available.date():
            raise ValueError("same-session narrative text is refused")
        event_id = str(event.get("event_id", "")).strip()
        if not event_id:
            raise ValueError("narrative event is missing event_id")
        if event_id in seen:
            raise ValueError(f"narrative event_id {event_id} is repeated")
        seen.add(event_id)
        text = event.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("narrative polarity requires the text available at available_ts")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        rows.append(
            {
                "event_id": event_id,
                "ticker": ticker,
                "available_ts": available.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "decision_session": session.isoformat(),
                "polarity": score_text(text, model),
                "polarity_model_id": model.model_id,
                "polarity_text_hash": digest,
            }
        )
    rows.sort(key=lambda row: (str(row["available_ts"]), str(row["event_id"])))
    return rows


def render_scores_jsonl(rows: Sequence[Mapping[str, object]]) -> bytes:
    lines = [json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) for row in rows]
    if not lines:
        return b""
    return ("\n".join(lines) + "\n").encode("utf-8")


def mean_polarity_by_session(
    rows: Sequence[Mapping[str, object]],
) -> dict[date, dict[str, float]]:
    """Mean event polarity at each already-assigned decision session."""

    grouped: dict[date, dict[str, list[float]]] = {}
    for row in rows:
        session = date.fromisoformat(str(row["decision_session"]))
        ticker = str(row["ticker"])
        grouped.setdefault(session, {}).setdefault(ticker, []).append(float(row["polarity"]))
    return {
        session: {
            ticker: sum(values) / len(values)
            for ticker, values in sorted(names.items())
        }
        for session, names in grouped.items()
    }


def demean_cross_section(means: Mapping[str, float]) -> dict[str, float]:
    """Subtract the cross-sectional median. Fewer than two names is empty."""

    if len(means) < 2:
        return {}
    ordered = sorted(float(value) for value in means.values())
    mid = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[mid]
    else:
        median = (ordered[mid - 1] + ordered[mid]) / 2.0
    return {ticker: float(value) - median for ticker, value in means.items()}


def polarity_provenance(
    *,
    lexicon: PolarityLexicon,
    scores_sha256: str,
    narrative_sha256: str,
    event_count: int,
    scored_at: datetime,
) -> dict[str, object]:
    if narrative_sha256 != LOCKED_NARRATIVE_SHA256:
        raise ValueError("polarity provenance refuses a changed narrative_sha256")
    if lexicon.model_id != POLARITY_MODEL_ID:
        raise ValueError("polarity model id is not the frozen lexicon")
    return {
        "schema": _PROVENANCE_SCHEMA,
        "experiment_id": BAREBONE_EXPERIMENT_ID,
        "provider": "frozen lexicon; no live model",
        "model_id": lexicon.model_id,
        "aggregation": POLARITY_AGGREGATION,
        "skill_threshold": POLARITY_SKILL_THRESHOLD,
        "scored_at": scored_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lexicon": NARRATIVE_LEXICON,
        "lexicon_sha256": lexicon.sha256,
        "scores": NARRATIVE_SCORES,
        "polarity_sha256": scores_sha256,
        "narrative_events_sha256": narrative_sha256,
        "event_count": event_count,
        "license_note": "not redistributed; local bind only",
        "comparable_performance_claim": False,
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def write_locked_polarity_sha256(config_path: Path, digest: str) -> None:
    """Record the scores-file digest without touching the narrative or tape hash."""

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("barebone comparison config must be a JSON object")
    if payload.get("comparable_performance_claim") is not False:
        raise ValueError("comparable_performance_claim must be false")
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("evidence path is required")
    if evidence.get("narrative_sha256") != LOCKED_NARRATIVE_SHA256:
        raise ValueError("refusing to lock polarity against a changed narrative_sha256")
    evidence["narrative_scores"] = NARRATIVE_SCORES
    evidence["polarity_sha256"] = digest
    validate_barebone_payload(payload)
    config_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def score_locked_events(
    *,
    config_path: Path,
    root: Path | None = None,
    lock_config: bool = False,
    sessions: Sequence[date] | None = None,
    now: datetime | None = None,
) -> str:
    """Score the locked events file. Return the scores digest. Do not rewrite events."""

    base = root or _repo_root()
    config = load_barebone_comparison_config(config_path)
    if config.comparable_performance_claim is not False:
        raise ValueError("comparable_performance_claim must be false")
    if config.evidence.narrative_sha256 != LOCKED_NARRATIVE_SHA256:
        raise ValueError("narrative_sha256 is locked; refusing a new digest without a new experiment_id")
    events_path = base / config.evidence.narrative_events
    payload = events_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != LOCKED_NARRATIVE_SHA256:
        raise ValueError("narrative events bytes do not match the locked narrative_sha256")
    calendar = sessions if sessions is not None else sessions_from_ohlcv(base / config.evidence.ohlcv)
    events = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    lexicon = load_polarity_lexicon(base / NARRATIVE_LEXICON)
    rows = score_events(events, calendar, lexicon)
    if not rows:
        raise ValueError("refusing an empty polarity tape")
    rendered = render_scores_jsonl(rows)
    scores_digest = hashlib.sha256(rendered).hexdigest()
    _atomic_write(base / NARRATIVE_SCORES, rendered)
    sidecar = polarity_provenance(
        lexicon=lexicon,
        scores_sha256=scores_digest,
        narrative_sha256=LOCKED_NARRATIVE_SHA256,
        event_count=len(rows),
        scored_at=now or datetime.now(timezone.utc),
    )
    _atomic_write(
        base / NARRATIVE_POLARITY_PROVENANCE,
        (json.dumps(sidecar, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
    )
    if lock_config:
        write_locked_polarity_sha256(config_path, scores_digest)
    return scores_digest


def main(argv: list[str] | None = None) -> int:
    """Score the locked local events file. Does not call a live model."""

    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--lock-config", action="store_true")
    args = parser.parse_args(argv)
    config_path = args.config or (_repo_root() / "configs" / "barebone-comparison-v1.json")
    try:
        digest = score_locked_events(config_path=config_path, lock_config=args.lock_config)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"polarity_sha256={digest}")
    if args.lock_config:
        print(f"locked polarity_sha256={digest} in {config_path}")
    else:
        print("polarity_sha256 remains unset; re-run with --lock-config to record this digest")
    return 0
