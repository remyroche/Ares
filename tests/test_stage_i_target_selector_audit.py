from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_i_target_selector_audit import (
    StageITargetSelectorAuditError,
    audit_stage_i_target_selector_information,
)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    features = []
    for side_index, side in enumerate(("long", "short")):
        for index in range(12):
            signal = pd.Timestamp("2023-01-01", tz="UTC") + pd.Timedelta(hours=index)
            rows.append({
                "side_name": side,
                "__ts__": signal,
                "decision_ts": signal + pd.Timedelta(hours=1),
                "label_available_ts": signal + pd.Timedelta(hours=13),
                "r3_class": index % 3,
                "r3_metric_target": float(index),
                "robust_clear_soft_b25_t50": float(index) / 11.0,
                "t2_tp6_sl4_event": index % 3,
                "exact_net_bps": float(index * 10 - 50 + side_index),
            })
            features.append({"signal": float(index), "noise": float((-1) ** index)})
    return pd.DataFrame(rows), pd.DataFrame(features)


def test_audit_distinguishes_stable_target_information_and_policy_alignment() -> None:
    ledger, features = _inputs()
    audit, composition, summary = audit_stage_i_target_selector_information(
        ledger,
        features,
        side_feature_universes={"long": ["signal", "noise"], "short": ["signal", "noise"]},
    )
    signal = audit.loc[audit.feature.eq("signal")]
    assert signal["spearman_r3_metric_target"].eq(1.0).all()
    assert signal["all_era_target_sign_consistent"].all()
    assert set(composition["slice"]) == {"r3_class", "first_touch_event"}
    assert summary["long"]["soft_target_oracle"]["top_10"]["rows"] == 2


def test_audit_rejects_row_misalignment(tmp_path) -> None:
    ledger, features = _inputs()
    features.index = np.arange(len(features)) + 1
    with pytest.raises(StageITargetSelectorAuditError, match="identical row order"):
        audit_stage_i_target_selector_information(
            ledger,
            features,
            side_feature_universes={"long": ["signal"], "short": ["signal"]},
        )
