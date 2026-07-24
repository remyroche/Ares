import numpy as np
import pandas as pd

from extreme_price_movements.market_state_archetype_overlay import (
    MarketStateOverlayConfig,
    fit_market_state_archetype_overlay,
    select_market_state_columns,
)
from extreme_price_movements.regime_ev_calibration import apply_regime_ev_calibration


class _ToyRiskModel:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        values = pd.to_numeric(frame["ctx_market_stress"], errors="coerce").fillna(0.0)
        return np.where(values.gt(0.5), 0.02, -0.01).astype(np.float32)


class _ToyInternallyDerivedRiskModel:
    internally_derived_feature_columns = frozenset({"local_margin"})

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        work = frame.copy()
        work["local_margin"] = work["rank_pct"] - 0.75
        return np.asarray(work["local_margin"], dtype=np.float32)


def _panel() -> pd.DataFrame:
    n = 240
    ts = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    stress = np.linspace(0.0, 1.0, n, dtype=np.float32)
    ret = (0.018 - 0.040 * stress).astype(np.float32)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": "AAA/USD:USD",
            "side_name": "long",
            "policy_archetype": "long__breakout",
            "rank_pct": np.full(n, 0.90, dtype=np.float32),
            "ctx_market_stress": stress,
            "gmm_cluster_posterior_0": (1.0 - stress).astype(np.float32),
            "ret_net_notional": ret,
            "full_sl": (ret < -0.01).astype(np.int8),
        }
    )


def test_market_state_overlay_learns_unfavorable_bucket_and_adjusts_score() -> None:
    frame = _panel()
    result = fit_market_state_archetype_overlay(
        frame,
        feature_columns=["ctx_market_stress"],
        outcome_col="ret_net_notional",
        config=MarketStateOverlayConfig(
            min_group_rows=50,
            min_bucket_rows=20,
            max_features_per_group=2,
            min_abs_effect=0.0001,
            effect_scale=2.5,
        ),
    )
    assert result.artifact["effects"]
    effect = result.artifact["effects"][0]
    effects = {int(k): float(v) for k, v in effect["params"]["effects"].items()}
    assert effects[max(effects)] > 0.0

    scored = apply_regime_ev_calibration(
        frame,
        result.artifact,
        source_score_col="rank_pct",
        side_col="side_name",
        archetype_col="policy_archetype",
    )
    high = scored.loc[
        scored["ctx_market_stress"] > 0.90, "score_market_state_calibrated"
    ].mean()
    low = scored.loc[
        scored["ctx_market_stress"] < 0.10, "score_market_state_calibrated"
    ].mean()
    assert float(high) < float(low)


def test_market_state_feature_selector_excludes_realized_outcome_columns() -> None:
    frame = _panel()
    frame["ctx_target_like"] = np.arange(len(frame), dtype=np.float32)
    frame["ctx_clean_signal"] = np.arange(len(frame), dtype=np.float32)
    frame["ret_net_context"] = np.arange(len(frame), dtype=np.float32)
    selected = select_market_state_columns(
        frame,
        include_prefixes=("ctx_", "gmm_"),
        required_columns=["timestamp", "side_name", "policy_archetype", "rank_pct"],
    )
    assert "ctx_clean_signal" in selected
    assert "gmm_cluster_posterior_0" in selected
    assert "ctx_target_like" not in selected
    assert "ret_net_context" not in selected


def test_regime_ev_calibration_accepts_shallow_lgbm_and_score_application(
    tmp_path,
) -> None:
    import joblib

    model_path = tmp_path / "lgbm_shallow_long_arch.joblib"
    joblib.dump(_ToyRiskModel(), model_path)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
            "side_name": ["long", "long"],
            "policy_archetype": ["arch", "arch"],
            "rank_pct": [0.50, 0.50],
            "ctx_market_stress": [0.1, 0.9],
        }
    )
    artifact = {
        "_artifact_base_dir": str(tmp_path),
        "source_score_col": "rank_pct",
        "adjusted_score_col": "score_regime_calibrated",
        "risk_score_col": "regime_ev_risk_score",
        "effect_count_col": "regime_ev_effect_count",
        "risk_cap_negative": 0.08,
        "risk_cap_positive": 0.08,
        "score_application": {
            "mode": "additive",
            "scale": 2.0,
            "max_upscore": 0.015,
            "max_downscore": 0.030,
        },
        "effects": [
            {
                "side_name": "long",
                "archetype_policy_key": "arch",
                "shape": "lgbm_shallow_pickle",
                "model_path": model_path.name,
                "feature_cols": ["ctx_market_stress"],
                "fill_values": {"ctx_market_stress": 0.0},
            }
        ],
    }

    scored = apply_regime_ev_calibration(
        frame,
        artifact,
        source_score_col="rank_pct",
        side_col="side_name",
        archetype_col="policy_archetype",
    )

    assert np.allclose(scored["regime_ev_risk_score"], [-0.01, 0.02])
    assert np.allclose(scored["score_regime_calibrated"], [0.515, 0.470])


def test_strict_calibration_allows_model_internally_derived_features(tmp_path) -> None:
    import joblib

    model_path = tmp_path / "derived.joblib"
    joblib.dump(_ToyInternallyDerivedRiskModel(), model_path)
    frame = pd.DataFrame(
        {
            "side_name": ["long"],
            "archetype_policy_key": ["arch"],
            "rank_pct": [0.80],
        }
    )
    artifact = {
        "_artifact_base_dir": str(tmp_path),
        "strict_required_features": True,
        "source_score_col": "rank_pct",
        "effects": [
            {
                "side_name": "long",
                "archetype_policy_key": "arch",
                "shape": "sklearn_pickle",
                "model_path": model_path.name,
                "feature_cols": ["rank_pct", "local_margin"],
            }
        ],
    }

    scored = apply_regime_ev_calibration(frame, artifact)

    assert scored.loc[0, "regime_ev_risk_score"] == np.float32(0.05)


def test_common_ev_mapping_and_dirty_avoid_blacklist_are_shared_contract() -> None:
    frame = pd.DataFrame(
        {
            "side_name": ["long", "short"],
            "archetype_policy_key": [
                "long_dirtyavoid_sparse_questionable",
                "short_breakout",
            ],
            "calibrated_score": [0.95, 0.95],
        }
    )
    artifact = {
        "source_score_col": "calibrated_score",
        "adjusted_score_col": "score_regime_calibrated",
        "effects": [],
        "expected_ev_mapping": {
            "global": {"x": [0.0, 1.0], "y": [-0.01, 0.02]},
            "local": {
                "short||short_breakout": {
                    "x": [0.0, 1.0],
                    "y": [0.0, 0.04],
                    "weight": 0.5,
                    "support": 1000,
                }
            },
            "rank_reference": [-0.01, 0.0, 0.01, 0.02, 0.03],
        },
        "blacklisted_side_archetypes": [
            "long||long_dirtyavoid_sparse_questionable"
        ],
    }
    scored = apply_regime_ev_calibration(frame, artifact)
    assert bool(scored.loc[0, "regime_ev_blacklisted"])
    assert scored.loc[0, "expected_net_ev_after_1pct"] == -1.0
    assert scored.loc[0, "expected_ev_rank_score"] == 0.0
    assert scored.loc[1, "expected_net_ev_after_1pct"] > 0.02
    assert scored.loc[1, "expected_ev_rank_score"] > 0.5
