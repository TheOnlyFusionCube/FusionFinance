#!/usr/bin/env python3
"""Size the Barebone narrative arm from the locked polarity scores.

Uses the frozen lexicon scores, the shared rebalance clock, and the same
0.10 / 1.0 risk and out-of-sample skill gate as the momentum book. Skill that
is not strictly positive stays in cash. This does not rewrite the momentum
ledger, the events file, or the scores file.
"""

from __future__ import annotations

from demo.barebone_narrative_run import main


if __name__ == "__main__":
    raise SystemExit(main())
