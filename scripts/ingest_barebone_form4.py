#!/usr/bin/env python3
"""Bind SEC Form 4 insider filings for the locked Barebone window.

Reads ``data.sec.gov`` submissions JSON and the raw ownership XML for exact
Form 4 filings. Amendments are excluded. ``available_ts`` is
``acceptanceDateTime``. The events file is gitignored. ``--lock-config``
records ``edgar_form4_sha256`` and does not change ``tape_sha256``,
``narrative_sha256``, or ``edgar_sha256``.
"""

from __future__ import annotations

from demo.barebone_form4 import main


if __name__ == "__main__":
    raise SystemExit(main())
