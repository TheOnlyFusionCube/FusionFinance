#!/usr/bin/env python3
"""Size the Barebone attention arm and the momentum-attention hybrid.

Attention uses demeaned event counts and the expanding Spearman gate. The
hybrid arm is cash unless the momentum skill gate and the attention skill
gate both pass; the weights are then the momentum scores. The polarity
lexicon is not used. This does not rewrite the momentum ledger, the events
file, or the lexicon.
"""

from __future__ import annotations

from demo.barebone_attention_run import main


if __name__ == "__main__":
    raise SystemExit(main())
