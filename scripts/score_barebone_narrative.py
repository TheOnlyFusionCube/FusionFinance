#!/usr/bin/env python3
"""Score the locked Barebone narrative events with the frozen lexicon.

Reads ``configs/barebone-comparison-v1.json`` and the gitignored events file.
It does not call a live model, does not rewrite the events file, and does not
change ``narrative_sha256``. ``--lock-config`` records ``polarity_sha256`` as
the digest of the scores file just written.
"""

from __future__ import annotations

from demo.barebone_polarity import main


if __name__ == "__main__":
    raise SystemExit(main())
