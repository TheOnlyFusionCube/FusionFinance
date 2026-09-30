"""Local-hash-only ingest for barebone-comparison-v1."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from demo.barebone_comparison import (
    BAREBONE_OHLCV,
    BAREBONE_PROVENANCE,
    FAIR_RACE_TAPE_HASH,
    load_barebone_comparison_config,
    refuse_barebone_performance_claim,
    require_barebone_evidence,
)
from demo.barebone_tape import (
    fetch_provider_rows,
    ingest_barebone_tape,
    main,
    read_ohlcv_csv,
    refuse_software_mark_source,
    required_tickers,
    write_locked_tape_sha256,
)
from demo.barebone_tape import _refuse_fair_race_bytes as refuse_fair_race_tape_bytes

_LEGACY_METRICS_SHA256 = "e7e5055ce9b4409d7941a71929f66261416b8d3c3f62423190edafd5bf5b1411"
_DATES = ("2025-01-02", "2026-01-12")
_PRICE_KEYS = {"open", "high", "low", "close", "adjclose", "volume", "bars"}


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _config_copy(tmp_path: Path) -> Path:
    source = _root() / "configs" / "barebone-comparison-v1.json"
    destination = tmp_path / "barebone-comparison-v1.json"
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return destination


def _tickers() -> tuple[str, ...]:
    return required_tickers(load_barebone_comparison_config())


def _csv_bytes(tickers: tuple[str, ...], *, include_adj: bool, drop: str | None = None) -> bytes:
    header = ["date", "ticker", "open", "high", "low", "close", "volume"]
    if include_adj:
        header.append("adjclose")
    lines = [",".join(header)]
    for day in _DATES:
        for index, ticker in enumerate(tickers):
            if ticker == drop and day == _DATES[-1]:
                continue
            base = 50.0 + index
            cells = [
                day,
                ticker,
                f"{base:.1f}",
                f"{base + 1:.1f}",
                f"{base - 1:.1f}",
                f"{base:.1f}",
                str(1000 + index),
            ]
            if include_adj:
                cells.append(f"{base - 0.25:.2f}")
            lines.append(",".join(cells))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _write_csv(tmp_path: Path, payload: bytes, name: str = "local.csv") -> Path:
    path = tmp_path / name
    path.write_bytes(payload)
    return path


def test_barebone_tape_names_are_the_config_universe_plus_spy_and_qqq() -> None:
    config = load_barebone_comparison_config()
    names = required_tickers(config)

    assert len(config.experiment.universe) == 16
    assert names == (*config.experiment.universe, "SPY", "QQQ")
    assert config.evidence.tape_sha256 is None
    assert (
        hashlib.sha256((_root() / "results" / "metrics.json").read_bytes()).hexdigest()
        == _LEGACY_METRICS_SHA256
    )


def test_barebone_ohlcv_is_gitignored_and_provenance_is_not() -> None:
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", BAREBONE_OHLCV],
        cwd=_root(),
        check=False,
    )
    provenance = subprocess.run(
        ["git", "check-ignore", "-q", BAREBONE_PROVENANCE],
        cwd=_root(),
        check=False,
    )

    assert ignored.returncode == 0
    assert provenance.returncode == 1


def test_submission_archive_skips_a_local_barebone_ohlcv_file() -> None:
    root = _root()
    spec = importlib.util.spec_from_file_location(
        "build_submission_archive", root / "scripts" / "build_submission_archive.py"
    )
    assert spec and spec.loader
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    path = root / BAREBONE_OHLCV
    path.write_text('{"bars": []}\n', encoding="utf-8")
    try:
        assert path not in builder.collect_files()
    finally:
        path.unlink()


def test_from_csv_locks_the_real_digest_and_writes_provenance_without_prices(
    tmp_path: Path,
) -> None:
    config_path = _config_copy(tmp_path)
    csv_path = _write_csv(tmp_path, _csv_bytes(_tickers(), include_adj=True))
    fetched = datetime(2026, 9, 30, 3, 51, tzinfo=timezone.utc)

    digest = ingest_barebone_tape(
        config_path=config_path,
        source="local-csv",
        csv_path=csv_path,
        now=fetched,
        root=tmp_path,
    )
    ohlcv = tmp_path / BAREBONE_OHLCV
    sidecar = json.loads((tmp_path / BAREBONE_PROVENANCE).read_text(encoding="utf-8"))
    unlocked = load_barebone_comparison_config(config_path)

    assert hashlib.sha256(ohlcv.read_bytes()).hexdigest() == digest
    assert unlocked.evidence.tape_sha256 is None
    with pytest.raises(ValueError, match="unlocked evidence tape"):
        require_barebone_evidence(unlocked, root=tmp_path)
    digest = ingest_barebone_tape(
        config_path=config_path,
        source="local-csv",
        csv_path=csv_path,
        lock_config=True,
        now=fetched,
        root=tmp_path,
    )
    assert sidecar["byte_sha256"] == digest
    assert sidecar["provider"] == "local-csv"
    assert sidecar["license_note"] == "not redistributed; local bind only"
    assert sidecar["comparable_performance_claim"] is False
    assert sidecar["window"] == ["2025-01-02", "2026-01-12"]
    assert sidecar["tickers"] == list(_tickers())
    assert _PRICE_KEYS.isdisjoint(sidecar)
    tape = json.loads(ohlcv.read_text(encoding="utf-8"))
    assert tape["schema"] == "fusionfinance-evidence-ohlcv-v1"
    assert tape["adjustment"].startswith("raw OHLC plus source adjclose")
    assert tape["bars"][0]["adjclose"] != tape["bars"][0]["close"]

    locked = load_barebone_comparison_config(config_path)
    assert require_barebone_evidence(locked, root=tmp_path) == ohlcv
    with pytest.raises(ValueError, match="keeps comparable_performance_claim false"):
        refuse_barebone_performance_claim(locked, True)
    ohlcv.write_bytes(ohlcv.read_bytes() + b" ")
    with pytest.raises(ValueError, match="does not match the locked tape_sha256"):
        require_barebone_evidence(locked, root=tmp_path)


def test_cli_from_csv_prints_the_digest_and_leaves_tape_sha256_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io
    import sys

    config_path = _config_copy(tmp_path)
    csv_path = _write_csv(tmp_path, _csv_bytes(_tickers(), include_adj=True))
    output = tmp_path / "out.json"
    provenance = tmp_path / "provenance.json"
    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    code = main(
        [
            "--config",
            str(config_path),
            "--from-csv",
            str(csv_path),
            "--output",
            str(output),
            "--provenance",
            str(provenance),
        ]
    )
    printed = stdout.getvalue()
    recorded = json.loads(config_path.read_text(encoding="utf-8"))

    assert code == 0
    assert f"sha256={hashlib.sha256(output.read_bytes()).hexdigest()}" in printed
    assert "remains null" in printed
    assert recorded["evidence"]["tape_sha256"] is None
    assert recorded["comparable_performance_claim"] is False
    assert main(["--from-csv", str(csv_path), "--provider", "tiingo"]) == 2


def test_raw_csv_records_unadjusted_bars_and_missing_adjclose_is_refused(
    tmp_path: Path,
) -> None:
    raw_path = _write_csv(tmp_path, _csv_bytes(_tickers(), include_adj=False), "raw.csv")
    vendor_path = _write_csv(tmp_path, _csv_bytes(_tickers(), include_adj=True), "vendor.csv")

    with pytest.raises(ValueError, match="Silent adjustment is refused"):
        read_ohlcv_csv(raw_path, adjustment="auto")
    rows, note = read_ohlcv_csv(raw_path, adjustment="raw")
    assert note.startswith("raw OHLC")
    assert all(row["adjclose"] == row["close"] for row in rows)
    with pytest.raises(ValueError, match="already has adjclose"):
        read_ohlcv_csv(vendor_path, adjustment="raw")


def test_ingest_fails_closed_on_a_missing_name_and_refuses_software_marks(
    tmp_path: Path,
) -> None:
    config_path = _config_copy(tmp_path)
    csv_path = _write_csv(
        tmp_path, _csv_bytes(_tickers(), include_adj=True, drop="QQQ")
    )

    with pytest.raises(ValueError, match="missing ticker"):
        ingest_barebone_tape(
            config_path=config_path,
            source="local-csv",
            csv_path=csv_path,
            root=tmp_path,
        )
    with pytest.raises(ValueError, match="software-mark"):
        refuse_software_mark_source("software-marks")
    with pytest.raises(ValueError, match="software-mark"):
        ingest_barebone_tape(
            config_path=config_path,
            source="software-marks",
            csv_path=csv_path,
            root=tmp_path,
        )
    assert not (tmp_path / BAREBONE_OHLCV).exists()


def test_ingest_refuses_the_fair_race_extract_as_a_source(tmp_path: Path) -> None:
    fair = _root() / "evidence" / "market" / "locked_ohlcv.json"
    body = fair.read_bytes()
    config_path = _config_copy(tmp_path)

    with pytest.raises(ValueError, match="fair-race OHLCV"):
        read_ohlcv_csv(fair, adjustment="auto")
    with pytest.raises(ValueError, match="fair-race OHLCV"):
        refuse_fair_race_tape_bytes(body)
    with pytest.raises(ValueError, match="fair-race"):
        write_locked_tape_sha256(config_path, FAIR_RACE_TAPE_HASH)
    with pytest.raises(ValueError, match="fair-race"):
        write_locked_tape_sha256(config_path, hashlib.sha256(body).hexdigest())
    assert json.loads(config_path.read_text(encoding="utf-8"))["evidence"]["tape_sha256"] is None


def test_provider_fetch_requires_an_env_key_and_parses_fixture_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    config_path = _config_copy(tmp_path)

    def boom(request):
        raise AssertionError(request.full_url)

    with pytest.raises(ValueError, match="TIINGO_API_KEY"):
        ingest_barebone_tape(
            config_path=config_path,
            source="tiingo",
            opener=boom,
            root=tmp_path,
        )

    def tiingo(request):
        assert request.get_header("Authorization") == "Token test-key"
        ticker = request.full_url.split("/daily/")[1].split("/prices")[0].upper()
        index = _tickers().index(ticker)
        base = 50.0 + index
        payload = []
        for day in _DATES:
            payload.append(
                {
                    "date": f"{day}T00:00:00.000Z",
                    "open": base,
                    "high": base + 1,
                    "low": base - 1,
                    "close": base,
                    "volume": 1000 + index,
                    "adjClose": base - 0.25,
                }
            )
        return json.dumps(payload).encode("utf-8")

    digest = ingest_barebone_tape(
        config_path=config_path,
        source="tiingo",
        api_key="test-key",
        opener=tiingo,
        now=datetime(2026, 9, 30, 4, 0, tzinfo=timezone.utc),
        root=tmp_path,
    )
    document = json.loads((tmp_path / BAREBONE_OHLCV).read_text(encoding="utf-8"))
    assert hashlib.sha256((tmp_path / BAREBONE_OHLCV).read_bytes()).hexdigest() == digest
    assert document["vendor"] == "Tiingo daily prices"
    assert "Tiingo adjClose" in document["adjustment"]

    def polygon(request):
        assert "apiKey=secret-key" in request.full_url
        symbol = request.full_url.split("/ticker/")[1].split("/range")[0]
        ticker = symbol.replace(".", "-")
        index = _tickers().index(ticker)
        base = 80.0 + index
        results = []
        for day in _DATES:
            stamp = int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp() * 1000)
            results.append(
                {"o": base, "h": base + 1, "l": base - 1, "c": base, "v": 10, "t": stamp}
            )
        return json.dumps({"status": "OK", "results": results}).encode("utf-8")

    def broken(request):
        raise RuntimeError(f"down secret-key {request.full_url}")

    with pytest.raises(ValueError, match="\\*\\*\\*") as caught:
        fetch_provider_rows(
            "polygon",
            ("AAPL",),
            api_key="secret-key",
            opener=broken,
        )
    assert "secret-key" not in str(caught.value)
    rows, note = fetch_provider_rows(
        "polygon", ("AAPL",), api_key="secret-key", opener=polygon
    )
    assert note.startswith("Polygon v2 aggs")
    assert rows[0]["adjclose"] == rows[0]["close"]
    assert rows[0]["ticker"] == "AAPL"
