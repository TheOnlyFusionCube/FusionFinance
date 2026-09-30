#!/usr/bin/env python3
"""Count locked Barebone Hacker News events into an attention scorebook.

Reads ``configs/barebone-comparison-v1.json`` and the gitignored events file.
The score is ``log1p`` of the mapped count in a 21-session window ending at
the decision session, minus the cross-sectional median. It does not read the
polarity lexicon, does not rewrite the events file, and does not change
``narrative_sha256`` or ``tape_sha256``. ``--lock-config`` records
``attention_sha256`` as the digest of the scores file just written.
"""

from __future__ import annotations

from demo.barebone_attention import main


if __name__ == "__main__":
    raise SystemExit(main())
