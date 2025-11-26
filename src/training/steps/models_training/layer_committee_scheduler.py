"""Calendar-based retrain scheduler for committee layers.

This module provides a small, standalone helper that decides which
layers should be retrained as of a given date, based on simple
calendar rules:

- If day-of-month == 1  → retrain Layer 1 (base specialists)
- If weekday == Monday  → retrain Layer 3 (meta layer)

Layer 2 can be treated as a more frequent retrain (e.g., whenever this
scheduler is invoked).

The actual training logic is implemented separately in
`src/training/utils/layer2_training.py` and related utilities. This
module only answers the question "what should we retrain today?".
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict

from src.utils.logger import system_logger


logger = system_logger.getChild("LayerCommitteeScheduler")


def get_retrain_plan(as_of: datetime | None = None) -> Dict[str, object]:
    """Return a simple retrain plan for Layer 1/2/3.

    Args:
        as_of: Reference datetime (UTC). If None, uses current UTC time.

    Returns:
        Dict with keys:
            - "as_of": datetime
            - "layer1": bool (retrain base layer?)
            - "layer2": bool (retrain committee / mid layer?)
            - "layer3": bool (retrain meta layer?)
    """
    if as_of is None:
        as_of = datetime.utcnow()

    is_first_day = as_of.day == 1
    is_monday = as_of.weekday() == 0  # Monday == 0

    plan: Dict[str, object] = {
        "as_of": as_of,
        "layer1": is_first_day,
        # Default: allow callers to retrain Layer 2 whenever this scheduler
        # runs. This can be tightened later (e.g., only on certain weekdays).
        "layer2": True,
        "layer3": is_monday,
    }

    logger.info(
        "[LayerCommitteeScheduler] Retrain plan for %s → L1=%s, L2=%s, L3=%s",
        as_of.date(),
        plan["layer1"],
        plan["layer2"],
        plan["layer3"],
    )

    return plan


if __name__ == "__main__":  # pragma: no cover - convenience entrypoint
    current_plan = get_retrain_plan()
    logger.info("Current retrain plan: %s", current_plan)
