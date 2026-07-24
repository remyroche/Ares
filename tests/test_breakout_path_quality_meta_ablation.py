from __future__ import annotations

import pandas as pd
import pytest

import scripts.run_breakout_path_quality_meta_ablation as ablation
from scripts.run_breakout_path_quality_meta_ablation import (
    ARMS,
    BreakoutPathContext,
    RAW_PATH_FIELDS,
    _fixed_features,
    _local_short_breakout_override,
)


def test_path_context_only_populates_the_target_short_archetype(tmp_path) -> None:
    context = pd.DataFrame(
        {
            "__ts__": [pd.Timestamp("2026-04-01", tz="UTC")],
            "__symbol__": ["BTC/USD:USD"],
            "side_name": ["short"],
            "__archetype_policy_key__": ["short_breakout_precision"],
            "breakout_rapid_reversal_probability_ebm": [0.7],
            "breakout_rapid_reversal_probability_reliability": [0.8],
            "breakout_severe_retention_probability_ebm": [0.3],
            "breakout_severe_retention_probability_reliability": [0.9],
        }
    )
    path = tmp_path / "context.parquet"
    context.to_parquet(path, index=False)
    builder = BreakoutPathContext(path)
    frame = pd.DataFrame(
        {
            "__ts__": [pd.Timestamp("2026-04-01", tz="UTC")] * 3,
            "__symbol__": ["BTC/USD:USD", "ETH/USD:USD", "SOL/USD:USD"],
            "side_name": ["short", "short", "long"],
            "archetype_policy_key": [
                "short_breakout_precision",
                "short_default_clean_path",
                "short_breakout_precision",
            ],
        }
    )

    out, coverage = builder.attach(frame, ARMS["full_path_context"].fields)

    assert coverage["active_rows"] == 1
    assert coverage["matched_rows"] == 1
    assert coverage["inactive_non_null_rows"] == 0
    assert out.loc[0, "breakout_rapid_reversal_reliable_risk"] == pytest.approx(0.56)
    assert out.loc[0, "breakout_severe_retention_uncertain_risk"] == pytest.approx(0.03)
    assert out.loc[1:, list(RAW_PATH_FIELDS)].isna().all().all()


def test_fixed_feature_contract_only_expands_short_side() -> None:
    parent = {
        "selected_features_by_side": {
            "long": ["score", "long_feature"],
            "short": ["score", "short_feature"],
        }
    }
    features = _fixed_features(parent, ARMS["raw_path_fields"])
    assert features["long"] == ["score", "long_feature"]
    assert features["short"][:2] == ["score", "short_feature"]
    assert set(RAW_PATH_FIELDS).issubset(features["short"])


def test_local_override_replaces_only_target_archetype_scores(monkeypatch) -> None:
    rows = 110
    train = pd.DataFrame(
        {
            "side_name": ["short"] * 100 + ["long"] * 10,
            "archetype_policy_key": ["short_breakout_precision"] * 100
            + ["long_vol_compression"] * 10,
        }
    )
    target = pd.Series([0.8] * 100 + [0.2] * 10)
    x_train = pd.DataFrame({"score": [0.1] * rows, "path": [0.2] * rows})
    x_valid = pd.DataFrame({"score": [0.1, 0.2], "path": [0.2, 0.3]})
    scored = pd.DataFrame(
        {
            "side_name": ["short", "short"],
            "archetype_policy_key": ["short_breakout_precision", "short_default_clean_path"],
            "score_meta_base_soft_label": [0.4, 0.5],
        }
    )
    monkeypatch.setattr(ablation, "_fit_base_soft_label_model", lambda *args, **kwargs: "model")
    monkeypatch.setattr(
        ablation,
        "_predict",
        lambda model, matrix, classifier: pd.Series([0.75] * len(matrix), index=matrix.index),
    )

    override = _local_short_breakout_override(ARMS["local_short_breakout_path"])
    out, models, feature_names, metadata = override(
        x_train=x_train,
        train=train,
        x_valid=x_valid,
        scored=scored,
        base_target=target,
        feature_names_by_side={"long": ["score"], "short": ["score", "path"]},
        classifier_params={},
        fold="2026-04",
        seed=1,
    )

    assert out.loc[0, "score_meta_base_soft_label"] == pytest.approx(0.75)
    assert out.loc[1, "score_meta_base_soft_label"] == pytest.approx(0.5)
    assert out.loc[0, "score_meta_base_soft_label_parent"] == pytest.approx(0.4)
    assert pd.isna(out.loc[1, "score_meta_base_soft_label_local_short_breakout"])
    assert set(models) == {"base_soft_label_short_breakout_local"}
    assert feature_names["base_soft_label_short_breakout_local"] == ["score", "path"]
    assert metadata["train_rows"] == 100
