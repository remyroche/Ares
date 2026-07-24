import numpy as np
import pandas as pd

from extreme_price_movements.unsupervised_regime_learning.economic_relevance import (
    EconomicRegimeRelevanceConfig,
    add_global_topk_surprise_targets,
    materialize_composite_features,
    score_side_archetype_economic_relevance,
)


def _panel() -> pd.DataFrame:
    n = 600
    idx = np.arange(n)
    side = np.where(idx % 2 == 0, "long", "short")
    arch = np.where(side == "long", "long__breakout", "short__pullback")
    stress = ((idx % 100) / 99.0).astype(np.float32)
    opportunity = (1.0 - stress).astype(np.float32)
    score = np.linspace(0.0, 1.0, n, dtype=np.float32)
    ev = np.where(stress > 0.75, -0.015, 0.006).astype(np.float32)
    # Make near-threshold low-stress rows profitable so the positive surprise
    # task has a learnable state in the global top20 excluding top10 bucket.
    ev[(score >= 0.70) & (score < 0.90) & (opportunity > 0.75)] = 0.018
    clean = (ev > 0.0).astype(np.int8)
    return pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC"),
            "month": "2026-01",
            "__symbol__": "AAA/USD:USD",
            "side_name": side,
            "archetype_policy_key": arch,
            "score_meta_base_soft_label": score,
            "state_stress": stress,
            "state_opportunity": opportunity,
            "ev_after_1pct": ev,
            "clean_exec": clean,
            "dirty_positive": ((ev > 0.0) & (stress > 0.8)).astype(np.int8),
            "full_path_bad_mae_1r": (stress > 0.75).astype(np.int8),
            "timeout": np.zeros(n, dtype=np.int8),
            "stop_or_adverse": (ev < 0.0).astype(np.int8),
        }
    )


def test_global_topk_surprise_targets_use_global_denominator() -> None:
    frame = add_global_topk_surprise_targets(_panel())
    assert 0.09 <= float(frame["url_trade_top10_population"].mean()) <= 0.11
    assert 0.09 <= float(frame["url_demote_top10_population"].mean()) <= 0.11
    assert 0.09 <= float(frame["url_promote_top20_not_top10_population"].mean()) <= 0.11
    assert 0.14 <= float(frame["url_negative_top15_population"].mean()) <= 0.16
    assert 0.19 <= float(frame["url_positive_mid30_population"].mean()) <= 0.21
    assert frame.loc[frame["url_demote_top10_population"].eq(1), "url_global_score_rank_pct"].min() >= 0.90


def test_global_topk_can_disable_monthwise_rank_normalization() -> None:
    frame = _panel()
    frame.loc[:299, "month"] = "2026-01"
    frame.loc[300:, "month"] = "2026-02"
    # The second month contains all of the globally highest scores.  A global
    # overlay population must therefore admit no January rows at top 10%.
    frame.loc[:299, "score_meta_base_soft_label"] = np.linspace(
        0.0, 0.4, 300, dtype=np.float32
    )
    frame.loc[300:, "score_meta_base_soft_label"] = np.linspace(
        0.6, 1.0, 300, dtype=np.float32
    )
    global_rank = add_global_topk_surprise_targets(
        frame,
        config=EconomicRegimeRelevanceConfig(month_col=""),
    )
    assert int(global_rank.loc[:299, "url_trade_top10_population"].sum()) == 0
    assert 0.09 <= float(global_rank["url_trade_top10_population"].mean()) <= 0.11


def test_global_topk_preserves_frozen_parent_percentile_after_filtering() -> None:
    frame = _panel().iloc[:100].copy()
    # This mirrors a parent-score top-20 prefilter.  The score must remain its
    # original global percentile: re-ranking these rows would turn 0.80 into
    # the bottom of the local sample and manufacture a different top-10 set.
    frame["parent_rank_v9"] = np.linspace(0.80, 0.99, len(frame), dtype=np.float32)
    scored = add_global_topk_surprise_targets(
        frame,
        config=EconomicRegimeRelevanceConfig(
            score_col="parent_rank_v9",
            score_is_percentile=True,
            month_col="",
        ),
    )
    expected = frame["parent_rank_v9"].ge(0.90).to_numpy()
    actual = scored["url_trade_top10_population"].astype(bool).to_numpy()
    np.testing.assert_array_equal(actual, expected)


def test_side_archetype_relevance_finds_negative_and_positive_states() -> None:
    frame = _panel()
    cfg = EconomicRegimeRelevanceConfig(
        min_group_rows=100,
        min_population_rows=20,
        min_state_rows=10,
        max_features_per_group=4,
        max_features_for_composites=2,
        min_candidate_score=0.01,
        lgbm_enabled=False,
    )
    feature_metrics, composite_metrics, definitions = score_side_archetype_economic_relevance(
        frame,
        ["state_stress", "state_opportunity"],
        config=cfg,
    )
    assert not feature_metrics.empty
    neg = feature_metrics.loc[
        feature_metrics["task"].eq("demote_top10")
        & feature_metrics["feature"].eq("state_stress")
        & feature_metrics["feature_bin"].eq("high")
    ]
    assert not neg.empty
    assert float(neg["economic_relevance_score"].max()) > 0.0
    pos = feature_metrics.loc[
        feature_metrics["task"].eq("promote_top20_not_top10")
        & feature_metrics["feature"].eq("state_opportunity")
        & feature_metrics["feature_bin"].eq("high")
    ]
    assert not pos.empty
    assert float(pos["economic_relevance_score"].max()) > 0.0
    assert {"day_temporal_score", "week_temporal_score", "temporal_relevance_score"}.issubset(feature_metrics.columns)
    assert {
        "nonlinear_relevance_score",
        "nonlinear_target_rate_span",
        "nonlinear_ev_span",
    }.issubset(feature_metrics.columns)
    assert float(feature_metrics["nonlinear_relevance_score"].max()) >= 0.0
    if definitions:
        mat = materialize_composite_features(frame, definitions[:2])
        assert len(mat) == len(frame)
        assert all(str(col).startswith("url_cmp__") for col in mat.columns)
        assert any(str(col).endswith("__intensity") for col in mat.columns)
    assert isinstance(composite_metrics, pd.DataFrame)


def test_shock_entropy_composites_do_not_pair_with_asset_residuals() -> None:
    frame = _panel()
    frame["market_shock_iqr"] = np.linspace(0.0, 1.0, len(frame), dtype=np.float32)
    frame["symbol_minus_mkt_ret_1h"] = np.linspace(1.0, 0.0, len(frame), dtype=np.float32)
    frame["xs_dispersion__asset_minus_mkt_oi_1d"] = np.linspace(0.1, 0.9, len(frame), dtype=np.float32)
    frame["market_breadth_24h"] = np.sin(np.arange(len(frame)) / 13.0).astype(np.float32)
    cfg = EconomicRegimeRelevanceConfig(
        min_group_rows=100,
        min_population_rows=20,
        min_state_rows=10,
        max_features_per_group=5,
        max_features_for_composites=5,
        min_candidate_score=0.0,
        lgbm_enabled=False,
    )
    _features, _composites, definitions = score_side_archetype_economic_relevance(
        frame,
        [
            "market_shock_iqr",
            "symbol_minus_mkt_ret_1h",
            "xs_dispersion__asset_minus_mkt_oi_1d",
            "market_breadth_24h",
        ],
        config=cfg,
    )
    names = [str(item.get("name", "")) for item in definitions]
    assert not any("market_shock_iqr" in name and "symbol_minus_mkt_ret_1h" in name for name in names)
    assert not any("market_shock_iqr" in name and "xs_dispersion_asset_minus_mkt_oi_1d" in name for name in names)
