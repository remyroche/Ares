from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_febapr2025_historical_catboost_six_class_gate import CLASS_ORDER, _six


def test_historical_six_class_mapping_is_fixed_and_does_not_touch_production_contract() -> None:
    labels = pd.Series([
        "fast_clean_winner", "fast_winner_early_drawdown",
        "early_mfe_full_reversal", "noisy_timeout_usable_mfe", "dead_timeout",
    ])
    assert _six(labels).tolist() == [
        "fast_realization_winner", "fast_realization_winner",
        "mfe_reversal_or_timeout", "mfe_reversal_or_timeout", "dead_timeout",
    ]
    assert len(CLASS_ORDER) == 6
