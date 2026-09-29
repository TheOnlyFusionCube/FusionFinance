"""Deterministic longitudinal filing-text features.

Adapted from filing-alpha-cleanroom 0.6.0 under the MIT License.  See NOTICE.
These features complement LLM research with reproducible, auditable evidence.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

import numpy as np
import pandas as pd


_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z'-]*|\d+(?:\.\d+)?%?")
_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")

ADVERSE = {
    "adverse", "breach", "decline", "default", "deteriorate", "deterioration",
    "disruption", "impairment", "investigation", "lawsuit", "loss", "material weakness",
    "restatement", "shortfall", "uncertain", "weakness", "warning", "write-down",
    "going concern",
}
POSITIVE = {
    "accelerate", "benefit", "improve", "improvement", "opportunity", "profitable",
    "recovery", "resilient", "strength", "strong", "successful", "upside",
}
UNCERTAINTY = {
    "approximately", "could", "estimate", "may", "might", "possible", "potential",
    "uncertain", "uncertainty", "unknown", "unlikely", "variability",
}
CONSTRAINING = {
    "cannot", "covenant", "limit", "limited", "must", "prohibit", "restrict",
    "restricted", "requirement", "unable",
}


def _normalize_text(text: object) -> str:
    if text is None:
        return ""
    try:
        if bool(pd.isna(text)):
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]


def _sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in _SENTENCE_PATTERN.split(text) if sentence.strip()]


def _phrase_count(text: str, lexicon: Iterable[str]) -> int:
    count = 0
    token_counts = Counter(_tokens(text))
    for term in lexicon:
        count += text.count(term) if " " in term or "-" in term else token_counts[term]
    return count


def extract_text_statistics(text: object) -> dict[str, float]:
    """Return deterministic readability, tone, and warning statistics."""
    normalized = _normalize_text(text)
    tokens = _tokens(normalized)
    sentences = _sentences(normalized)
    word_denominator = max(len(tokens), 1)
    sentence_denominator = max(len(sentences), 1)
    long_words = sum(len(token) >= 7 for token in tokens)
    numeric_tokens = sum(any(character.isdigit() for character in token) for token in tokens)
    return {
        "word_count": float(len(tokens)),
        "sentence_count": float(len(sentences)),
        "avg_sentence_length": float(len(tokens) / sentence_denominator),
        "long_word_rate": float(long_words / word_denominator),
        "numeric_token_rate": float(numeric_tokens / word_denominator),
        "adverse_rate": float(_phrase_count(normalized, ADVERSE) / word_denominator),
        "positive_rate": float(_phrase_count(normalized, POSITIVE) / word_denominator),
        "uncertainty_rate": float(_phrase_count(normalized, UNCERTAINTY) / word_denominator),
        "constraining_rate": float(_phrase_count(normalized, CONSTRAINING) / word_denominator),
        "material_weakness_flag": float("material weakness" in normalized),
        "going_concern_flag": float("going concern" in normalized),
        "restatement_flag": float("restate" in normalized),
        "liquidity_warning_flag": float(
            "liquidity" in normalized and _phrase_count(normalized, ADVERSE) > 0
        ),
    }


def _jaccard_novelty(current: str, prior: str) -> float:
    current_set = set(_tokens(current))
    prior_set = set(_tokens(prior))
    union = current_set | prior_set
    return 0.0 if not union else 1.0 - len(current_set & prior_set) / len(union)


def _sentence_change_rates(current: str, prior: str) -> tuple[float, float]:
    current_sentences = {re.sub(r"\W+", " ", value).strip() for value in _sentences(current)}
    prior_sentences = {re.sub(r"\W+", " ", value).strip() for value in _sentences(prior)}
    current_sentences.discard("")
    prior_sentences.discard("")
    new_rate = len(current_sentences - prior_sentences) / max(len(current_sentences), 1)
    deleted_rate = len(prior_sentences - current_sentences) / max(len(prior_sentences), 1)
    return float(new_rate), float(deleted_rate)


def build_longitudinal_text_features(section_text: pd.DataFrame) -> pd.DataFrame:
    """Return filing-delta features from section-level filing text copies.

    Comparisons are isolated within ticker/form/section, preventing a 10-Q
    risk section from being compared with a 10-K MD&A section.
    """
    required = {"ticker", "accepted_at", "form", "section", "text"}
    missing = sorted(required.difference(section_text.columns))
    if missing:
        raise ValueError(f"Section text is missing required columns: {missing}")
    frame = section_text.copy(deep=True)
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame["form"] = frame["form"].astype(str).str.upper().str.strip()
    frame["section"] = (
        frame["section"].astype(str).str.lower().str.replace(r"\W+", "_", regex=True).str.strip("_")
    )
    if (frame["section"] == "").any():
        raise ValueError("section names must contain at least one alphanumeric character")
    frame["accepted_at"] = pd.to_datetime(frame["accepted_at"], errors="raise", utc=True)
    frame["normalized_text"] = frame["text"].map(_normalize_text)
    frame = frame.sort_values(["ticker", "form", "section", "accepted_at"]).reset_index(drop=True)

    statistic_rows = pd.DataFrame(
        [extract_text_statistics(text) for text in frame["normalized_text"]]
    )
    frame = pd.concat([frame.reset_index(drop=True), statistic_rows], axis=1)
    base_stats = list(statistic_rows.columns)
    group_keys = ["ticker", "form", "section"]
    frame["prior_text"] = frame.groupby(group_keys, sort=False)["normalized_text"].shift(1)
    for column in base_stats:
        frame[f"delta_{column}"] = frame[column] - frame.groupby(group_keys, sort=False)[column].shift(1)
    frame["text_novelty"] = [
        _jaccard_novelty(current, prior) if isinstance(prior, str) else np.nan
        for current, prior in zip(frame["normalized_text"], frame["prior_text"], strict=False)
    ]
    sentence_changes = [
        _sentence_change_rates(current, prior) if isinstance(prior, str) else (np.nan, np.nan)
        for current, prior in zip(frame["normalized_text"], frame["prior_text"], strict=False)
    ]
    frame["new_sentence_rate"] = [value[0] for value in sentence_changes]
    frame["deleted_sentence_rate"] = [value[1] for value in sentence_changes]
    frame["adverse_deterioration_flag"] = (
        frame["delta_adverse_rate"].fillna(0.0) > 0.0
    ).astype(float)
    for window in (2, 3, 4, 6):
        frame[f"adverse_persistence_{window}"] = (
            frame.assign(_flag=frame["adverse_deterioration_flag"])
            .groupby(group_keys, sort=False)["_flag"]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=group_keys, drop=True)
        )

    identifiers = ["ticker", "accepted_at", "form"]
    if "period_end" in frame.columns:
        identifiers.append("period_end")
    value_columns = [
        *base_stats,
        *(f"delta_{column}" for column in base_stats),
        "text_novelty", "new_sentence_rate", "deleted_sentence_rate",
        "adverse_deterioration_flag", "adverse_persistence_2", "adverse_persistence_3",
        "adverse_persistence_4", "adverse_persistence_6",
    ]
    wide_parts: list[pd.DataFrame] = []
    for section, group in frame.groupby("section", sort=False):
        section_frame = group[identifiers + value_columns].copy()
        section_frame = section_frame.rename(
            columns={column: f"{section}__{column}" for column in value_columns}
        )
        wide_parts.append(section_frame)
    if not wide_parts:
        return pd.DataFrame(columns=identifiers)
    result = wide_parts[0]
    for part in wide_parts[1:]:
        result = result.merge(part, on=identifiers, how="outer", validate="one_to_one")
    return result.sort_values(["ticker", "accepted_at"]).reset_index(drop=True)
