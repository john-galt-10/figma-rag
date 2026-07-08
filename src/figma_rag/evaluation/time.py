"""Timezone helpers for evaluation artifacts."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

EVALUATION_TIMEZONE = ZoneInfo("Europe/Rome")


def evaluation_now() -> datetime:
    """Return the current evaluation timestamp in the Europe/Rome timezone."""

    return datetime.now(EVALUATION_TIMEZONE)
