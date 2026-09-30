#!/usr/bin/env python3
"""Bind SEC EDGAR submissions metadata for the locked Barebone window.

Reads ``data.sec.gov`` submissions JSON for the frozen CIK map. Forms are
exactly 8-K, 10-Q, and 10-K. Amendments are excluded. ``available_ts`` is
``acceptanceDateTime``. The events file is gitignored. ``--lock-config``
records ``edgar_sha256`` and does not change ``tape_sha256`` or
``narrative_sha256``.
"""

from __future__ import annotations

from demo.barebone_edgar import main


if __name__ == "__main__":
    raise SystemExit(main())
