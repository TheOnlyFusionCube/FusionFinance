"""Point-in-time Hacker News attention for the locked Barebone window.

The score is ``log1p`` of the mapped event count in a fixed lookback that ends
at the decision session, minus the cross-sectional median. The lexicon is not
read. Counts are gitignored. ``narrative_sha256`` and ``tape_sha256`` stay the
locked binds.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from pathlib import Path

from demo.barebone_comparison import (
    ATTENTION_PROVENANCE,
    ATTENTION_SCORES,
    BAREBONE_EXPERIMENT_ID,
    BAREBONE_UNIVERSE,
    LOCKED_NARRATIVE_SHA256,
    LOCKED_POLARITY_SHA256,
    LOCKED_TAPE_SHA256,
    NARRATIVE_LEXICON,
    load_barebone_comparison_config,
    validate_barebone_payload,
)
from demo.barebone_narrative import _aware_utc, decision_session_for, sessions_from_ohlcv

ATTENTION_SIGNAL_ID = "hn-attention-log1p-21s-v1"
HYBRID_SIGNAL_ID = "hn-attention-momentum-intersection-v1"
ATTENTION_LOOKBACK_SESSIONS = 21
ATTENTION_TRANSFORM = "log1p_count_minus_cross_sectional_median"
ATTENTION_SKILL_THRESHOLD = 0.0
FROZEN_LEXICON_SHA256 = "6fc38675c0b159da067bf2bd87d3ad93e7998a36f6a7095cdefa88bfc2e57a12"
_PROVENANCE_SCHEMA = "fusionfinance-barebone-attention-v1"
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
        "polarity",
        "polarity_model_id",
    }
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def demean_cross_section(values: Mapping[str, float]) -> dict[str, float]:
    """Subtract the cross-sectional median. Fewer than two names is empty."""

    if len(values) < 2:
        return {}
    ordered = sorted(float(value) for value in values.values())
    mid = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[mid]
    else:
        median = (ordered[mid - 1] + ordered[mid]) / 2.0
    return {ticker: float(value) - median for ticker, value in values.items()}


def _validate_event(
    event: Mapping[str, object],
    sessions: Sequence[date],
    universe: Sequence[str],
    seen: set[str],
) -> tuple[str, date]:
    if not isinstance(event, Mapping):
        raise ValueError("narrative event must be an object")
    banned = _BANNED_FIELDS.intersection(event)
    if banned:
        raise ValueError("narrative event carries a return label: " + ", ".join(sorted(banned)))
    if event.get("available_ts") in (None, ""):
        raise ValueError("narrative event is missing available_ts")
    available = _aware_utc(str(event["available_ts"]))
    ticker = str(event.get("ticker", "")).strip().upper()
    allowed = set(universe)
    if ticker not in allowed:
        raise ValueError(f"narrative event ticker {ticker or '(blank)'} is not in the attention universe")
    session_text = event.get("decision_session")
    if not isinstance(session_text, str) or not session_text:
        raise ValueError("narrative event is missing decision_session")
    session = date.fromisoformat(session_text)
    if session not in set(sessions):
        raise ValueError(f"decision_session {session.isoformat()} is not on the barebone calendar")
    if session <= available.date():
        raise ValueError("same-session narrative text is refused")
    expected = decision_session_for(available, sessions)
    if expected is None or session != expected:
        raise ValueError("decision_session must be the first session strictly after available_ts")
    event_id = str(event.get("event_id", "")).strip()
    if not event_id:
        raise ValueError("narrative event is missing event_id")
    if event_id in seen:
        raise ValueError(f"narrative event_id {event_id} is repeated")
    seen.add(event_id)
    return ticker, session


def attention_score_rows(
    events: Sequence[Mapping[str, object]],
    sessions: Sequence[date],
    universe: Sequence[str] | None = None,
    *,
    lookback: int = ATTENTION_LOOKBACK_SESSIONS,
) -> list[dict[str, object]]:
    """Count mapped events in ``[j - lookback + 1, j]`` and demean ``log1p``.

    The last session in the window is the decision session. An event whose
    decision session is later is not a feature. Zero counts stay in the
    cross-section. The lexicon is not consulted.
    """

    if lookback < 1:
        raise ValueError("attention lookback must be a positive number of sessions")
    if lookback != ATTENTION_LOOKBACK_SESSIONS:
        raise ValueError("attention lookback is locked at 21 sessions")
    names = tuple(universe) if universe is not None else BAREBONE_UNIVERSE
    if len(names) < 2:
        raise ValueError("attention cross-section needs at least two names")
    if len(set(names)) != len(names):
        raise ValueError("attention universe repeats a ticker")
    ordered = tuple(sessions)
    for earlier, later in zip(ordered, ordered[1:]):
        if later <= earlier:
            raise ValueError("attention calendar must be strictly increasing")
    index = {day: offset for offset, day in enumerate(ordered)}
    counts = {ticker: [0] * len(ordered) for ticker in names}
    seen: set[str] = set()
    for event in events:
        ticker, session = _validate_event(event, ordered, names, seen)
        counts[ticker][index[session]] += 1
    prefix = {ticker: [0] for ticker in names}
    for ticker in names:
        running = 0
        for count in counts[ticker]:
            running += count
            prefix[ticker].append(running)
    rows: list[dict[str, object]] = []
    for session_index in range(lookback - 1, len(ordered)):
        start = session_index - lookback + 1
        raw = {
            ticker: math.log1p(prefix[ticker][session_index + 1] - prefix[ticker][start])
            for ticker in names
        }
        scores = demean_cross_section(raw)
        if len(scores) != len(names):
            raise ValueError("attention demean dropped a name")
        day = ordered[session_index]
        for ticker in names:
            event_count = prefix[ticker][session_index + 1] - prefix[ticker][start]
            rows.append(
                {
                    "session": day.isoformat(),
                    "session_index": session_index,
                    "ticker": ticker,
                    "event_count": event_count,
                    "log1p_count": raw[ticker],
                    "score": scores[ticker],
                    "signal_id": ATTENTION_SIGNAL_ID,
                    "lookback_sessions": lookback,
                }
            )
    return rows


def render_attention_jsonl(rows: Sequence[Mapping[str, object]]) -> bytes:
    lines = [json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) for row in rows]
    if not lines:
        return b""
    return ("\n".join(lines) + "\n").encode("utf-8")


def attention_provenance(
    *,
    scores_sha256: str,
    narrative_sha256: str,
    tape_sha256: str,
    event_count: int,
    row_count: int,
    scored_at: datetime,
) -> dict[str, object]:
    if narrative_sha256 != LOCKED_NARRATIVE_SHA256:
        raise ValueError("attention provenance refuses a changed narrative_sha256")
    if tape_sha256 != LOCKED_TAPE_SHA256:
        raise ValueError("attention provenance refuses a changed tape_sha256")
    return {
        "schema": _PROVENANCE_SCHEMA,
        "experiment_id": BAREBONE_EXPERIMENT_ID,
        "provider": "Hacker News event counts; lexicon not used",
        "signal_id": ATTENTION_SIGNAL_ID,
        "lookback_sessions": ATTENTION_LOOKBACK_SESSIONS,
        "transform": ATTENTION_TRANSFORM,
        "skill_threshold": ATTENTION_SKILL_THRESHOLD,
        "scored_at": scored_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scores": ATTENTION_SCORES,
        "attention_sha256": scores_sha256,
        "narrative_events_sha256": narrative_sha256,
        "tape_sha256": tape_sha256,
        "polarity_sha256": LOCKED_POLARITY_SHA256,
        "lexicon_used_for_sizing": False,
        "event_count": event_count,
        "row_count": row_count,
        "license_note": "not redistributed; local bind only",
        "comparable_performance_claim": False,
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def write_locked_attention_sha256(config_path: Path, digest: str) -> None:
    """Record the attention digest without touching narrative, tape, or polarity."""

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("barebone comparison config must be a JSON object")
    if payload.get("comparable_performance_claim") is not False:
        raise ValueError("comparable_performance_claim must be false")
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("evidence path is required")
    if evidence.get("narrative_sha256") != LOCKED_NARRATIVE_SHA256:
        raise ValueError("refusing to lock attention against a changed narrative_sha256")
    if evidence.get("tape_sha256") != LOCKED_TAPE_SHA256:
        raise ValueError("refusing to lock attention against a changed tape_sha256")
    if evidence.get("polarity_sha256") != LOCKED_POLARITY_SHA256:
        raise ValueError("refusing to lock attention against a changed polarity_sha256")
    evidence["attention_scores"] = ATTENTION_SCORES
    evidence["attention_sha256"] = digest
    validate_barebone_payload(payload)
    config_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _lexicon_unchanged(root: Path) -> None:
    digest = hashlib.sha256((root / NARRATIVE_LEXICON).read_bytes()).hexdigest()
    if digest != FROZEN_LEXICON_SHA256:
        raise ValueError("polarity lexicon bytes changed; refusing a retune")


def score_locked_attention(
    *,
    config_path: Path,
    root: Path | None = None,
    lock_config: bool = False,
    sessions: Sequence[date] | None = None,
    now: datetime | None = None,
) -> str:
    """Count the locked events file. Return the scores digest. Do not rewrite events."""

    base = root or _repo_root()
    config = load_barebone_comparison_config(config_path)
    if config.comparable_performance_claim is not False:
        raise ValueError("comparable_performance_claim must be false")
    if config.evidence.narrative_sha256 != LOCKED_NARRATIVE_SHA256:
        raise ValueError("narrative_sha256 is locked; refusing a new digest without a new experiment_id")
    if config.evidence.tape_sha256 != LOCKED_TAPE_SHA256:
        raise ValueError("tape_sha256 is locked; refusing a new digest without a new experiment_id")
    if config.evidence.polarity_sha256 != LOCKED_POLARITY_SHA256:
        raise ValueError("polarity_sha256 is locked; refusing a lexicon retune")
    _lexicon_unchanged(base)
    events_path = base / config.evidence.narrative_events
    payload = events_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != LOCKED_NARRATIVE_SHA256:
        raise ValueError("narrative events bytes do not match the locked narrative_sha256")
    calendar = sessions if sessions is not None else sessions_from_ohlcv(base / config.evidence.ohlcv)
    events = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    if not events:
        raise ValueError("refusing an empty attention tape")
    rows = attention_score_rows(events, calendar, BAREBONE_UNIVERSE)
    if not rows:
        raise ValueError("refusing an empty attention tape")
    for row in rows:
        if row.get("signal_id") != ATTENTION_SIGNAL_ID:
            raise ValueError("attention row is not the locked signal")
        if "text" in row or "polarity" in row:
            raise ValueError("attention row carries text or lexicon polarity")
    rendered = render_attention_jsonl(rows)
    scores_digest = hashlib.sha256(rendered).hexdigest()
    _atomic_write(base / ATTENTION_SCORES, rendered)
    sidecar = attention_provenance(
        scores_sha256=scores_digest,
        narrative_sha256=LOCKED_NARRATIVE_SHA256,
        tape_sha256=LOCKED_TAPE_SHA256,
        event_count=len(events),
        row_count=len(rows),
        scored_at=now or datetime.now(timezone.utc),
    )
    _atomic_write(
        base / ATTENTION_PROVENANCE,
        (json.dumps(sidecar, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
    )
    if lock_config:
        write_locked_attention_sha256(config_path, scores_digest)
    return scores_digest


def main(argv: list[str] | None = None) -> int:
    """Score the locked local events file. Does not call a live model or the lexicon."""

    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--lock-config", action="store_true")
    args = parser.parse_args(argv)
    config_path = args.config or (_repo_root() / "configs" / "barebone-comparison-v1.json")
    try:
        digest = score_locked_attention(config_path=config_path, lock_config=args.lock_config)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"attention_sha256={digest}")
    print(f"signal_id={ATTENTION_SIGNAL_ID}")
    if args.lock_config:
        print(f"locked attention_sha256={digest} in {config_path}")
    else:
        print("attention_sha256 remains unset; re-run with --lock-config to record this digest")
    return 0
