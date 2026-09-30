#!/usr/bin/env python3
"""Bind a local Barebone-window narrative extract without redistributing it.

Reads ``configs/barebone-comparison-v1.json``. ``--provider hn`` reads Hacker
News stories from the Algolia ``search_by_date`` API and does not use a key.
``--provider reddit`` requires ``REDDIT_CLIENT_ID``, ``REDDIT_CLIENT_SECRET``,
and ``REDDIT_USER_AGENT``. ``--provider x`` requires ``X_BEARER_TOKEN``. Missing
credentials skip that provider and invent nothing. This build does not call
Reddit or X even when credentials are present, and it does not label an X
recent search as a full-window archive.

The events file is gitignored. The provenance sidecar and, with
``--lock-config``, the config's ``narrative_sha256`` may be committed. Neither
stores event text as a redistributed dump, an API key, or a performance claim.
``narrative_sha256`` stays null until ``--lock-config`` records the digest of
a non-empty file just written.
"""

from __future__ import annotations

from demo.barebone_narrative import main


if __name__ == "__main__":
    raise SystemExit(main())
