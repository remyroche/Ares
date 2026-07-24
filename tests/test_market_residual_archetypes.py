from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.market_residual_archetypes import (
    MarketResidualConfig,
    MarketResidualStateRecognizer,
    PerArchetypeMarketAdverseConfig,
    PerArchetypeMarketAdverseRecognizer,
    adverse_episode_ranking_metrics,
    market_archetype_adverse_feature_names,
    market_residual_feature_names,
)
from extreme_price_movements.meta_residual_archetypes import (
    inference_feature_columns,
    strip_outcomes_for_oos,
)


def _frame(timestamps: int = 1000, assets: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(77)
    ts = pd.date_range("2025-01-01", periods=timestamps, freq="h", tz="UTC")
    phase = np.sin(np.arange(timestamps) / 24.0 * np.pi / 2.0).astype(np.float32)
    shock = np.repeat(phase, assets)
    rows = timestamps * assets
    score = np.clip(0.65 + 0.12 * shock + rng.normal(0.0, 0.03, rows), 0.01, 0.99)
    clean_probability = np.clip(score - 0.45 * np.maximum(shock, 0.0), 0.02, 0.98)
    clean = (rng.random(rows) < clean_probability).astype(np.float32)
    ev = (0.012 * clean - 0.016 * (1.0 - clean)).astype(np.float32)
    return pd.DataFrame(
        {
            "__ts__": ts.repeat(assets),
            "__symbol__": [f"S{i % assets}" for i in range(rows)],
            "side_name": np.where(np.arange(rows) % 2, "short", "long"),
            "archetype_policy_key": np.where(
                np.arange(rows) % 4 < 2, "continuation", "compression"
            ),
            "score_regime_calibrated": score.astype(np.float32),
            "clean_exec": clean,
            "dirty_positive": ((ev > 0.0) & (clean < 0.5)).astype(np.float32),
            "full_path_bad_mae_1r": (clean < 0.5).astype(np.float32),
            "timeout": np.zeros(rows, dtype=np.float32),
            "ev_after_1pct": ev,
            "mkt_shock": shock,
            "market_breadth_chg_1h": (-shock).astype(np.float32),
            "cross_asset_downside_corr_1h": np.abs(shock).astype(np.float32),
            "asset_minus_mkt_oi_chg_1h_rz": rng.normal(size=rows).astype(np.float32),
        }
    )


def test_market_residual_recognizer_is_market_only_and_oos_safe() -> None:
    frame = _frame()
    train = frame.iloc[:7200].copy()
    valid = frame.iloc[7200:].copy()
    recognizer = MarketResidualStateRecognizer(
        config=MarketResidualConfig(
            score_col="score_regime_calibrated",
            min_rows=100,
            max_fit_rows=1000,
            max_features=6,
            random_state=11,
        ),
        candidate_features=[
            "mkt_shock",
            "market_breadth_chg_1h",
            "cross_asset_downside_corr_1h",
            "asset_minus_mkt_oi_chg_1h_rz",
        ],
    ).fit(train)
    assert "asset_minus_mkt_oi_chg_1h_rz" not in recognizer.feature_columns
    output = recognizer.transform_oos(strip_outcomes_for_oos(valid))
    assert set(market_residual_feature_names()).issubset(output.columns)
    assert np.isfinite(output.to_numpy(dtype=np.float32)).all()
    np.testing.assert_allclose(output.filter(like="prob__").sum(axis=1), 1.0, atol=1e-5)
    assert recognizer.manifest()["leakage_contract"]["recent_hit_rate_inputs"] is False
    diagnostics = recognizer.manifest()["label_diagnostics"]
    assert diagnostics["label_basis"].startswith("train_relative")
    assert diagnostics["final_label_counts"][1] > 0
    assert diagnostics["final_label_counts"][2] > 0


def test_per_archetype_market_states_emit_continuous_failure_posteriors() -> None:
    frame = _frame(timestamps=1200, assets=8)
    train = frame.iloc[:8000].copy()
    train["threshold_basis_selected"] = (
        pd.to_numeric(train["score_regime_calibrated"], errors="coerce") >= 0.70
    )
    assert "threshold_basis_selected" not in inference_feature_columns(
        train, train.columns
    )
    valid = frame.iloc[8000:].copy()
    recognizer = PerArchetypeMarketAdverseRecognizer(
        config=PerArchetypeMarketAdverseConfig(
            score_col="score_regime_calibrated",
            min_archetype_rows=200,
            min_selected_rows=30,
            min_adverse_days=2,
            max_fit_rows=800,
            max_features=5,
            ae_gmm_max_rows=300,
            ae_max_iter=12,
            cluster_candidates=(2, 3),
            random_state=19,
        ),
        candidate_features=[
            "mkt_shock",
            "market_breadth_chg_1h",
            "cross_asset_downside_corr_1h",
            "asset_minus_mkt_oi_chg_1h_rz",
        ],
    ).fit(train)
    assert set(recognizer.models) == {"compression", "continuation"}
    safe_valid = strip_outcomes_for_oos(valid)
    assert "threshold_basis_selected" not in strip_outcomes_for_oos(train).columns
    generated = recognizer.transform_oos(safe_valid)
    assert set(market_archetype_adverse_feature_names()).issubset(generated.columns)
    assert not any("cluster_id" in name for name in generated.columns)
    assert generated.filter(like="state_prob__").to_numpy().min() >= 0.0
    assert generated.filter(like="state_prob__").to_numpy().max() <= 1.0
    targets = recognizer.prepare_evaluation_targets(valid)
    assessed = pd.concat([valid.reset_index(drop=True), generated, targets], axis=1)
    metrics = adverse_episode_ranking_metrics(assessed)
    assert {"aegmm", "mlp", "lgbm", "ensemble"}.issubset(set(metrics["model"]))
    manifest = recognizer.manifest()
    assert manifest["selection_contract"] == "static_train_ev_equivalent_threshold"
    assert manifest["leakage_contract"]["hard_cluster_ids_exposed"] is False
    assert manifest["leakage_contract"]["continuous_probabilities"] is True
    assert manifest["leakage_contract"]["global_market_state_model"] is False
