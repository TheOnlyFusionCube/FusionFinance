"""Deterministic evidence checks for committed FusionFinance theses.

The LLM is not trusted to validate its own citations or numbers.  This module
compares immutable evidence records with immutable, point-in-time source
records and emits the four fields consumed by :class:`VerifierOutput`.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from numbers import Real
from types import MappingProxyType


_DOCUMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_NUMERIC_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


@dataclass(frozen=True, slots=True)
class EvidenceSource:
    """A source snapshot and the time at which it became usable.

    ``numeric_values`` is copied before being wrapped in a read-only proxy, so
    later mutation of the caller's dictionary cannot rewrite the evidence.
    Keys identify trusted, pre-extracted values at locations in the document.
    """

    document_id: str
    available_at: str
    text: str
    numeric_values: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "numeric_values",
            MappingProxyType(dict(self.numeric_values)),
        )


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """One exact citation and, optionally, one reconciled numeric claim."""

    document_id: str
    quoted_text: str
    numeric_key: str = ""
    asserted_value: float | None = None


@dataclass(frozen=True, slots=True)
class EvidenceAuditResult:
    """Immutable result shaped for ``VerifierOutput`` evidence fields."""

    evidence_valid: bool
    citation_coverage: float
    numeric_reconciliation: float
    timestamp_integrity: bool
    errors: tuple[str, ...]
    valid_citations: int
    total_citations: int
    reconciled_numbers: int
    total_numbers: int


@dataclass(frozen=True, slots=True)
class _RecordAudit:
    citation_valid: bool = False
    timestamp_valid: bool = False
    total_numbers: int = 0
    reconciled_numbers: int = 0
    errors: tuple[str, ...] = ()


def audit_evidence(
    evidence: Iterable[EvidenceRecord],
    sources: Iterable[EvidenceSource],
    *,
    thesis_as_of: str,
    relative_tolerance: float = 1e-6,
    absolute_tolerance: float = 1e-9,
) -> EvidenceAuditResult:
    """Audit citations, numbers, and source availability without side effects.

    Exact quotes are case-sensitive byte-for-text matches. Numeric assertions
    are compared only with trusted values already extracted into the source
    record. Any malformed, missing, future, or ambiguous input fails closed.
    """

    records, record_errors = _snapshot_records(evidence)
    source_records, source_errors = _snapshot_sources(sources)
    tolerances_ok = _valid_tolerance(relative_tolerance) and _valid_tolerance(
        absolute_tolerance
    )
    as_of = _timestamp(thesis_as_of)
    source_by_id, index_errors = _index_sources(source_records)
    audits = tuple(
        _audit_record(
            record,
            index=index,
            source_by_id=source_by_id,
            as_of=as_of,
            tolerances_ok=tolerances_ok,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        for index, record in enumerate(records)
    )
    errors = (
        record_errors
        + source_errors
        + (() if tolerances_ok else ("INVALID_NUMERIC_TOLERANCE",))
        + (() if as_of is not None else ("INVALID_THESIS_AS_OF",))
        + index_errors
        + tuple(error for audit in audits for error in audit.errors)
    )
    return _build_result(
        audits,
        total_citations=len(records),
        errors=errors,
        as_of_valid=as_of is not None,
        collection_valid=not record_errors,
    )


def _build_result(
    audits: tuple[_RecordAudit, ...],
    *,
    total_citations: int,
    errors: tuple[str, ...],
    as_of_valid: bool,
    collection_valid: bool,
) -> EvidenceAuditResult:
    valid_citations = sum(audit.citation_valid for audit in audits)
    valid_timestamps = sum(audit.timestamp_valid for audit in audits)
    total_numbers = sum(audit.total_numbers for audit in audits)
    reconciled_numbers = sum(audit.reconciled_numbers for audit in audits)
    citation_coverage = valid_citations / total_citations if total_citations else 0.0
    numeric_reconciliation = (
        reconciled_numbers / total_numbers
        if total_numbers
        else (1.0 if total_citations else 0.0)
    )
    timestamp_integrity = bool(
        total_citations and as_of_valid and valid_timestamps == total_citations
    )
    final_errors = errors + (
        ("NO_EVIDENCE",) if not total_citations and collection_valid else ()
    )
    evidence_valid = bool(
        total_citations
        and not final_errors
        and citation_coverage == 1.0
        and numeric_reconciliation == 1.0
        and timestamp_integrity
    )
    return EvidenceAuditResult(
        evidence_valid=evidence_valid,
        citation_coverage=citation_coverage,
        numeric_reconciliation=numeric_reconciliation,
        timestamp_integrity=timestamp_integrity,
        errors=final_errors,
        valid_citations=valid_citations,
        total_citations=total_citations,
        reconciled_numbers=reconciled_numbers,
        total_numbers=total_numbers,
    )


def _audit_record(
    record: object,
    *,
    index: int,
    source_by_id: Mapping[str, EvidenceSource],
    as_of: datetime | None,
    tolerances_ok: bool,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> _RecordAudit:
    if not isinstance(record, EvidenceRecord):
        return _RecordAudit()
    numeric_total = int(record.asserted_value is not None)
    if not _valid_identifier(record.document_id, _DOCUMENT_ID):
        return _RecordAudit(
            total_numbers=numeric_total,
            errors=(f"INVALID_DOCUMENT_ID evidence[{index}]",),
        )
    source = source_by_id.get(record.document_id)
    if source is None:
        return _RecordAudit(
            total_numbers=numeric_total,
            errors=(f"UNKNOWN_DOCUMENT {record.document_id}",),
        )
    citation_valid, citation_errors = _audit_quote(record, source)
    timestamp_valid, timestamp_errors = _audit_timestamp(source, as_of)
    number_valid, number_errors = _audit_number(
        record,
        source,
        tolerances_ok=tolerances_ok,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    return _RecordAudit(
        citation_valid=citation_valid,
        timestamp_valid=timestamp_valid,
        total_numbers=numeric_total,
        reconciled_numbers=int(number_valid),
        errors=citation_errors + timestamp_errors + number_errors,
    )


def _snapshot_records(
    evidence: Iterable[EvidenceRecord],
) -> tuple[tuple[EvidenceRecord, ...], tuple[str, ...]]:
    try:
        records = tuple(evidence)
    except Exception:
        return (), ("INVALID_EVIDENCE_COLLECTION",)
    errors = tuple(
        f"INVALID_EVIDENCE_RECORD evidence[{index}]"
        for index, record in enumerate(records)
        if not isinstance(record, EvidenceRecord)
    )
    return records, errors


def _snapshot_sources(
    sources: Iterable[EvidenceSource],
) -> tuple[tuple[EvidenceSource, ...], tuple[str, ...]]:
    try:
        records = tuple(sources)
    except Exception:
        return (), ("INVALID_SOURCE_COLLECTION",)
    errors = tuple(
        f"INVALID_SOURCE_RECORD source[{index}]"
        for index, source in enumerate(records)
        if not isinstance(source, EvidenceSource)
    )
    return records, errors


def _index_sources(
    sources: tuple[EvidenceSource, ...],
) -> tuple[Mapping[str, EvidenceSource], tuple[str, ...]]:
    result: dict[str, EvidenceSource] = {}
    errors: tuple[str, ...] = ()
    for index, source in enumerate(sources):
        if not isinstance(source, EvidenceSource):
            continue
        if not _valid_identifier(source.document_id, _DOCUMENT_ID):
            errors = errors + (f"INVALID_DOCUMENT_ID source[{index}]",)
            continue
        if source.document_id in result:
            errors = errors + (f"DUPLICATE_DOCUMENT_ID {source.document_id}",)
            continue
        result = {**result, source.document_id: source}
    return MappingProxyType(result), errors


def _audit_quote(
    record: EvidenceRecord, source: EvidenceSource
) -> tuple[bool, tuple[str, ...]]:
    if _quote_is_exact(record.quoted_text, source.text):
        return True, ()
    if not isinstance(record.quoted_text, str) or not record.quoted_text:
        return False, (f"MISSING_QUOTE {record.document_id}",)
    return False, (f"QUOTE_NOT_FOUND {record.document_id}",)


def _audit_timestamp(
    source: EvidenceSource, as_of: datetime | None
) -> tuple[bool, tuple[str, ...]]:
    available_at = _timestamp(source.available_at)
    if available_at is None:
        return False, (f"INVALID_SOURCE_TIMESTAMP {source.document_id}",)
    if as_of is None:
        return False, ()
    if available_at > as_of:
        return False, (f"SOURCE_AFTER_AS_OF {source.document_id}",)
    return True, ()


def _audit_number(
    record: EvidenceRecord,
    source: EvidenceSource,
    *,
    tolerances_ok: bool,
    relative_tolerance: float,
    absolute_tolerance: float,
 ) -> tuple[bool, tuple[str, ...]]:
    if record.asserted_value is None:
        return False, ()
    if not tolerances_ok:
        return False, ()
    key = record.numeric_key
    if not _valid_identifier(key, _NUMERIC_KEY):
        return False, (f"INVALID_NUMERIC_KEY {record.document_id}",)
    if key not in source.numeric_values:
        return False, (f"UNKNOWN_NUMERIC_VALUE {record.document_id} {key}",)
    asserted = _finite_real(record.asserted_value)
    trusted = _finite_real(source.numeric_values[key])
    if asserted is None:
        return False, (f"INVALID_ASSERTED_NUMERIC {record.document_id} {key}",)
    if trusted is None:
        return False, (f"INVALID_SOURCE_NUMERIC {record.document_id} {key}",)
    if not math.isclose(
        asserted,
        trusted,
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    ):
        return False, (f"NUMERIC_MISMATCH {record.document_id} {key}",)
    return True, ()


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _quote_is_exact(quote: object, text: object) -> bool:
    return bool(
        isinstance(quote, str)
        and quote
        and isinstance(text, str)
        and quote in text
    )


def _valid_identifier(value: object, pattern: re.Pattern[str]) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _valid_tolerance(value: object) -> bool:
    return bool(
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _finite_real(value: object) -> float | None:
    if not isinstance(value, Real) or isinstance(value, bool):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


__all__ = [
    "EvidenceAuditResult",
    "EvidenceRecord",
    "EvidenceSource",
    "audit_evidence",
]
