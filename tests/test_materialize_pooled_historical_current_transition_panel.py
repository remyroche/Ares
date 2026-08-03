from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.materialize_historical_current_common_transition_geometry import CANONICAL_FEATURES
from scripts.materialize_pooled_historical_current_transition_panel import HISTORICAL_SOURCE, build_panel


def _labels(start: str, rows: int = 20) -> pd.DataFrame:
    stamps = pd.date_range(start, periods=rows, freq="h", tz="UTC")
    frame = pd.DataFrame({
        "cohort_anchor_utc": stamps, "horizon_hours": 12, "horizon_role": "fixture", "book_fraction": 0.10,
        "before_global_hour_complete_flag": True, "after_global_hour_complete_flag": True,
        "before_selected_candidate_support": 10, "after_selected_candidate_support": 10,
        "before_target_available_utc": stamps, "after_target_available_utc": stamps + pd.Timedelta(hours=12),
        "outcome_only_not_model_feature": True, "before_direct_mean_net": 0.01, "after_direct_mean_net": -0.01,
        "delta_direct_mean_net": -0.02, "delta_mean_conversion_residual": -0.02,
        "after_mean_conversion_residual": -0.01, "delta_positive_net_contribution": -0.01,
        "delta_loss_net_contribution": 0.01, "delta_opportunity_probability_0bps": -0.2,
    })
    return frame


def _geometry(labels: pd.DataFrame) -> pd.DataFrame:
    stamps = pd.to_datetime(labels["cohort_anchor_utc"], utc=True) - pd.Timedelta(hours=1)
    frame = pd.DataFrame({"signal_context_utc": stamps, "common_transition_context_available": True})
    for index, column in enumerate(CANONICAL_FEATURES):
        frame[column] = float(index + 1)
    return frame


def test_panel_uses_exact_common_geometry_and_preserves_non_oof_source() -> None:
    historical = _labels("2023-01-01")
    current_labels = _labels("2026-06-01")
    current = current_labels.assign(
        source_family="current_exact_spread_mayjul2026", economics_tier="exact", policy_cost_contract="exact",
        path_frequency="1m", promotion_use="research", mapping_provenance_role="strict_oof",
        provenance_oof_share=1.0, provenance_forward_oos_share=0.0,
    )
    panel = build_panel(historical, current, _geometry(historical), _geometry(current_labels))
    assert set(CANONICAL_FEATURES).issubset(panel.columns)
    assert len(CANONICAL_FEATURES) == 90
    old = panel.loc[panel["source_family"].eq(HISTORICAL_SOURCE)]
    assert old["mapping_provenance_role"].eq("historical_non_oof_backcast").all()
    assert old["source_domain"].eq("historical_non_oof_backcast").all()
    assert old["context_available"].all()
    resolved = old["target__active_adverse"].notna()
    assert resolved.any()
    assert old.loc[resolved, "target__active_adverse_available_utc"].notna().all()
    assert not any(token in name.lower() for name in CANONICAL_FEATURES for token in ("target", "future", "outcome"))
