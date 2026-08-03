from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_ii_meta_archetypes import (
    META_ARCHETYPE_PREFIX,
    UNKNOWN_MEMBERSHIP_COLUMN,
    SideLocalMetaArchetypeState,
    StageIIMetaArchetypeConfig,
    strict_oof_meta_archetype_features,
)
from extreme_price_movements.config import CFG


def _frame(rows: int = 1600) -> pd.DataFrame:
    rng = np.random.default_rng(8)
    decision = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    side = np.where(np.arange(rows) % 2, "long", "short")
    regime = rng.normal(size=rows).astype(np.float32)
    trust = rng.normal(size=rows).astype(np.float32)
    base = (15.0 * regime + 5.0 * trust).astype(np.float32)
    # Three conversion modes with a genuine side difference.  These are only
    # present on the fitting view, never supplied to transform.
    mode = np.where(regime > 0.8, 1, np.where(regime < -0.8, -1, 0))
    net = (base + 90.0 * mode + np.where(side == "long", 12.0, -9.0) + rng.normal(0, 7, rows)).astype(np.float32)
    return pd.DataFrame(
        {
            "decision_ts": decision,
            "label_available_ts": decision + pd.Timedelta(hours=13),
            "side_name": side,
            "exact_net_bps": net,
            "prequential_base_expected_net_bps": base,
            "realised_mfe_atr": (mode + rng.normal(0, 0.2, rows)).astype(np.float32),
            "regime": regime,
            "trust": trust,
        }
    )


def _config() -> StageIIMetaArchetypeConfig:
    return StageIIMetaArchetypeConfig(
        path_descriptor_cols=("realised_mfe_atr",),
        min_side_rows=120,
        min_component_rows=25,
        min_train_rows=300,
        components=3,
        oof_folds=3,
        random_state=9,
    )


def test_side_local_transform_is_soft_and_rejects_path_values() -> None:
    frame = _frame()
    cfg = _config()
    state = SideLocalMetaArchetypeState(cfg, ["regime", "trust", "prequential_base_expected_net_bps"]).fit(frame.iloc[:1000])
    assert set(state.side_models) == {"long", "short"}
    safe = frame.iloc[1000:].drop(columns=["exact_net_bps", "realised_mfe_atr"])
    out = state.transform(safe)
    probabilities = out.filter(like=f"{META_ARCHETYPE_PREFIX}prob__")
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert out[f"{META_ARCHETYPE_PREFIX}available"].eq(1.0).all()
    assert state.manifest()["hard_routing"] is False
    assert state.manifest()["local_trading_experts"] is False
    with pytest.raises(ValueError, match="realised path"):
        state.transform(frame.iloc[1000:])


def test_strict_oof_uses_only_prior_resolved_rows_and_marks_burnin_unknown() -> None:
    frame = _frame()
    result = strict_oof_meta_archetype_features(
        frame,
        config=_config(),
        causal_feature_cols=["regime", "trust", "prequential_base_expected_net_bps"],
    )
    scored = result.fold_audit.loc[result.fold_audit.status.eq("scored")]
    assert not scored.empty
    assert (pd.to_datetime(scored.train_max_label_available_ts, utc=True) < pd.to_datetime(scored.valid_start, utc=True)).all()
    assert result.manifest["strict_oof"] is True
    assert result.manifest["side_local_construction"] is True
    assert result.features[UNKNOWN_MEMBERSHIP_COLUMN].gt(0.5).any()
    assert result.features[f"{META_ARCHETYPE_PREFIX}available"].eq(1.0).any()
    assert result.catalog["side"].isin(["long", "short"]).all()
    assert result.diagnostic_truth_memberships.notna().any().any()
    assert scored["causal_membership_log_loss"].notna().all()


def test_outcome_columns_cannot_be_pretended_to_be_causal_inputs() -> None:
    frame = _frame()
    with pytest.raises(ValueError, match="Realised path"):
        strict_oof_meta_archetype_features(
            frame,
            config=_config(),
            causal_feature_cols=["regime", "exact_net_bps"],
        )


def test_label_availability_must_be_after_the_decision() -> None:
    frame = _frame(800)
    frame.loc[0, "label_available_ts"] = frame.loc[0, "decision_ts"]
    with pytest.raises(ValueError, match="resolve strictly after"):
        strict_oof_meta_archetype_features(
            frame,
            config=_config(),
            causal_feature_cols=["regime", "trust", "prequential_base_expected_net_bps"],
        )


def test_stage_ii_rejects_non_path_defined_conversion_clusters() -> None:
    frame = _frame()
    with pytest.raises(ValueError, match="path_descriptor"):
        SideLocalMetaArchetypeState(
            StageIIMetaArchetypeConfig(min_side_rows=20, min_component_rows=5),
            ["regime", "trust", "prequential_base_expected_net_bps"],
        ).fit(frame)


def test_stage_ii_features_are_registered_for_meta_not_base() -> None:
    names = set(CFG["STAGE_II_META_CONVERSION_ARCHETYPE_FEATURE_KEYS"])
    assert f"{META_ARCHETYPE_PREFIX}prior_residual_bps" in names
    assert "STAGE_II_META_CONVERSION_ARCHETYPE_FEATURE_KEYS" in CFG["meta_shared_feature_keys"]
    assert "STAGE_II_META_CONVERSION_ARCHETYPE_FEATURE_KEYS" not in CFG["base_shared_feature_keys"]
