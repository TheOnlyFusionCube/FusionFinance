#!/usr/bin/env python3
"""Bind FINRA consolidated short interest for the locked Barebone window.

Reads ``api.finra.org`` ``consolidatedShortInterest``. ``available_ts`` is the
publication date, not the settlement date. The events file is gitignored.
``--lock-config`` records ``short_interest_sha256`` and does not change
``tape_sha256``, ``narrative_sha256``, ``edgar_sha256``, or
``edgar_form4_sha256``.
"""

from __future__ import annotations

from demo.barebone_short_interest import main


if __name__ == "__main__":
    raise SystemExit(main())
