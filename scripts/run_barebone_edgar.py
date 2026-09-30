#!/usr/bin/env python3
"""Size the Barebone EDGAR filing-count arm.

Uses the locked submissions events, a 63-session log1p count, and the same
expanding Spearman gate as the momentum book. This does not rewrite the
momentum ledger or download filing HTML.
"""

from __future__ import annotations

from demo.barebone_edgar_run import main


if __name__ == "__main__":
    raise SystemExit(main())
