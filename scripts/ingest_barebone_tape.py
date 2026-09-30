#!/usr/bin/env python3
"""Bind a local Barebone-window OHLCV extract without redistributing it.

Reads ``configs/barebone-comparison-v1.json``. ``--provider tiingo`` uses
``TIINGO_API_KEY``. ``--provider polygon`` uses ``POLYGON_API_KEY``.
``--from-csv`` reads a local dump and does not call a vendor. The OHLCV file
is gitignored. The provenance sidecar and, with ``--lock-config``, the
config's ``tape_sha256`` may be committed. Neither stores an API key or a
performance claim.

``tape_sha256`` stays null until ``--lock-config`` records the digest of the
file just written. This script does not invent that digest.
"""

from __future__ import annotations

from demo.barebone_tape import main


if __name__ == "__main__":
    raise SystemExit(main())
