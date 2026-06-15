from __future__ import annotations

import pandas as pd

from extreme_price_movements.rule_mask_features import (
    RULE_MASK_FEATURE_PREFIX,
    append_rule_mask_features,
    rule_mask_feature_source_keys,
)


def test_rule_mask_features_materialize_from_diversified_registry(
    tmp_path, monkeypatch
) -> None:
    registry = tmp_path / "diversified_final_selection.csv"
    pd.DataFrame(
        [
            {
                "canonical_key": "(*)|(*)|(a>0.5&b<=1.0)",
                "source_target": "returns_target",
                "source_horizon": 5,
                "side": "short",
            },
            {
                "canonical_key": "(*)|(c>=2.0)|(*)",
                "source_target": "returns_target",
                "source_horizon": 10,
                "side": "long",
            },
        ]
    ).to_csv(registry, index=False)
    monkeypatch.setenv("EPM_LGBM_RULE_MASK_FEATURES_ENABLED", "1")
    monkeypatch.setenv("EPM_LGBM_RULE_MASK_FEATURES_CSV", str(registry))

    assert set(rule_mask_feature_source_keys({})) == {"a", "b", "c"}

    frame = pd.DataFrame(
        {
            "a": [0.6, 0.4, 0.8],
            "b": [0.9, 0.5, 1.2],
            "c": [1.0, 2.0, 3.0],
        }
    )
    out, diag = append_rule_mask_features(frame, {}, side="short", context="test")
    cols = [c for c in out.columns if c.startswith(RULE_MASK_FEATURE_PREFIX)]
    assert len(cols) == 2
    assert diag["enabled"] is True
    assert diag["missing_source_keys"] == 0
    assert out[cols[0]].tolist() == [1.0, 0.0, 0.0]
    assert out[cols[1]].tolist() == [0.0, 1.0, 1.0]


def test_rule_mask_features_skip_malformed_rules(tmp_path, monkeypatch) -> None:
    registry = tmp_path / "diversified_final_selection.csv"
    pd.DataFrame(
        [
            {
                "canonical_key": "(*)|(*)|(a>0.5)",
                "source_target": "returns_target",
                "source_horizon": 5,
                "side": "short",
            },
            {
                "canonical_key": "broken_one_slot_rule",
                "source_target": "returns_target",
                "source_horizon": 5,
                "side": "short",
            },
        ]
    ).to_csv(registry, index=False)
    monkeypatch.setenv("EPM_LGBM_RULE_MASK_FEATURES_ENABLED", "1")
    monkeypatch.setenv("EPM_LGBM_RULE_MASK_FEATURES_CSV", str(registry))

    frame = pd.DataFrame({"a": [0.6, 0.4, 0.8]})
    out, diag = append_rule_mask_features(frame, {}, side="short", context="test")

    cols = [c for c in out.columns if c.startswith(RULE_MASK_FEATURE_PREFIX)]
    assert len(cols) == 1
    assert diag["invalid_rule_count"] == 1
    assert len(diag["added_feature_names"]) == 1
    assert out[cols[0]].tolist() == [1.0, 0.0, 1.0]
