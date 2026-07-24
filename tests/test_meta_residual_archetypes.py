from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import extreme_price_movements.meta_residual_archetypes as residual_module

from extreme_price_movements.meta_residual_archetypes import (
    ResidualArchetypeConfig,
    ResidualArchetypeRecognizer,
    _select_recognizer_features,
    add_reference_surprise_targets,
    residual_feature_names,
    strip_outcomes_for_oos,
)
from scripts import run_meta_residual_archetype_discovery as discovery_runner


def _frame(rows: int = 4800) -> pd.DataFrame:
    rng = np.random.default_rng(52)
    ts = pd.date_range("2025-01-01", periods=rows // 8, freq="h", tz="UTC").repeat(8)
    side = np.where(np.arange(rows) % 2 == 0, "long", "short")
    archetype = np.where(np.arange(rows) % 4 < 2, "continuation", "compression")
    shock = rng.normal(0.0, 1.0, rows).astype(np.float32)
    breadth = rng.normal(0.0, 1.0, rows).astype(np.float32)
    score = np.clip(
        0.55 + 0.16 * shock - 0.08 * breadth + rng.normal(0.0, 0.05, rows), 0.01, 0.99
    )
    clean_prob = np.clip(
        score - 0.25 * (shock > 1.0) + 0.20 * (breadth > 1.0), 0.02, 0.98
    )
    clean = (rng.random(rows) < clean_prob).astype(np.float32)
    bad_mae = ((shock > 0.8) & (clean < 0.5)).astype(np.float32)
    timeout = ((breadth < -1.3) & (clean < 0.5)).astype(np.float32)
    ev = (0.012 * clean - 0.016 * (1.0 - clean) + 0.003 * breadth).astype(np.float32)
    return pd.DataFrame(
        {
            "__ts__": ts,
            "__symbol__": [f"S{i % 8}" for i in range(rows)],
            "side_name": side,
            "archetype_policy_key": archetype,
            "oos_fold": ts.to_period("M").astype(str),
            "score_meta_base_soft_label": score.astype(np.float32),
            "clean_exec": clean,
            "dirty_positive": ((ev > 0.0) & (bad_mae > 0.5)).astype(np.float32),
            "full_path_bad_mae_1r": bad_mae,
            "timeout": timeout,
            "ev_after_1pct": ev,
            "mkt_shock": shock,
            "market_breadth": breadth,
            "oi_flush": (shock - breadth).astype(np.float32),
            "base_score": score.astype(np.float32),
        }
    )


def test_reference_surprise_targets_are_globally_ranked_over_train_history() -> None:
    frame = add_reference_surprise_targets(_frame(), ResidualArchetypeConfig())
    assert frame["reference_rank_pct"].between(0.0, 1.0).all()
    assert set(frame["reference_rank_band"].unique()) <= {
        "below_top30",
        "top20_30",
        "top10_20",
        "top10",
    }
    expected = frame["clean_exec"] - frame["score_meta_base_soft_label"]
    np.testing.assert_allclose(frame["hit_surprise"], expected, atol=1e-7)
    expected_rank = frame["score_meta_base_soft_label"].rank(method="average", pct=True)
    np.testing.assert_allclose(frame["reference_rank_pct"], expected_rank, atol=1e-7)


def test_recognizer_oos_contract_and_fixed_semantics() -> None:
    frame = _frame()
    cfg = ResidualArchetypeConfig(
        min_side_rows=500,
        min_local_rows=350,
        min_cluster_rows=30,
        max_cluster_fit_rows=2500,
        max_recognizer_fit_rows=2500,
        max_recognizer_features=8,
        mutual_info_rows=1500,
        cluster_candidates=(3,),
        allow_side_fallback=True,
        random_state=7,
    )
    recognizer = ResidualArchetypeRecognizer(
        config=cfg,
        candidate_features=["mkt_shock", "market_breadth", "oi_flush", "base_score"],
    ).fit(frame.iloc[:4000].copy())
    assert recognizer.side_models
    safe = strip_outcomes_for_oos(frame.iloc[4000:].copy())
    out = recognizer.transform_oos(safe)
    expected = residual_feature_names(include_ae_gmm=False)
    assert set(expected).issubset(out.columns)
    assert np.isfinite(out[expected].to_numpy(dtype=np.float32)).all()
    assert (out.filter(like="prob__").sum(axis=1) > 0.99).all()
    assert recognizer.manifest()["leakage_contract"]["raw_cluster_ids_exposed"] is False


def test_recognizer_rejects_oos_outcomes() -> None:
    frame = _frame(2400)
    cfg = ResidualArchetypeConfig(
        min_side_rows=300,
        min_local_rows=250,
        min_cluster_rows=20,
        max_cluster_fit_rows=1200,
        max_recognizer_fit_rows=1200,
        cluster_candidates=(3,),
    )
    recognizer = ResidualArchetypeRecognizer(
        config=cfg,
        candidate_features=["mkt_shock", "market_breadth", "oi_flush", "base_score"],
    ).fit(frame.iloc[:2000].copy())
    with pytest.raises(ValueError, match="received outcomes"):
        recognizer.transform_oos(frame.iloc[2000:].copy())


def test_economic_semantic_labels_are_temporally_stable_and_oos_safe() -> None:
    frame = _frame(6000)
    cfg = ResidualArchetypeConfig(
        min_side_rows=500,
        min_local_rows=350,
        min_cluster_rows=20,
        max_recognizer_fit_rows=3000,
        max_recognizer_features=8,
        mutual_info_rows=1800,
        label_mode="economic_semantic",
        semantic_min_temporal_segments=2,
        semantic_min_segment_rows=5,
        random_state=17,
        allow_side_fallback=True,
    )
    recognizer = ResidualArchetypeRecognizer(
        config=cfg,
        candidate_features=["mkt_shock", "market_breadth", "oi_flush", "base_score"],
    ).fit(frame.iloc[:5000].copy())
    assert recognizer.side_models
    assert recognizer.local_models
    assert recognizer.manifest()["label_mode"] == "economic_semantic"
    assert recognizer.catalog_["label_mode"].eq("economic_semantic").all()
    assert recognizer.catalog_["stable_temporal_segments"].ge(2).all()
    safe = strip_outcomes_for_oos(frame.iloc[5000:].copy())
    out = recognizer.transform_oos(safe)
    probabilities = out.filter(like="meta_resid_arch_prob__")
    assert np.isfinite(probabilities.to_numpy(dtype=np.float32)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)


def test_binned_mi_screen_recovers_nonlinear_large_surprise_signal() -> None:
    rng = np.random.default_rng(123)
    rows = 3000
    nonlinear = rng.normal(size=rows).astype(np.float32)
    event = (np.abs(nonlinear) > 1.25).astype(np.int8)
    frame = pd.DataFrame(
        {
            "nonlinear_state": nonlinear,
            "linear_noise": rng.normal(size=rows).astype(np.float32),
            "large_negative_surprise_label": event,
            "negative_autocorr_label": event,
        }
    )
    selected, relevance = _select_recognizer_features(
        frame,
        np.zeros(rows, dtype=np.int32),
        ["linear_noise", "nonlinear_state"],
        ResidualArchetypeConfig(
            max_recognizer_features=1,
            mutual_info_rows=3000,
            feature_screen_bins=8,
            feature_screen_lgbm_rounds=20,
        ),
        9,
    )
    assert selected == ["nonlinear_state"]
    assert relevance[0]["binned_mi"] > 0.0


def test_recognizer_uses_explicit_uncertain_fallback_for_unknown_side() -> None:
    frame = _frame(2400)
    cfg = ResidualArchetypeConfig(
        min_side_rows=300,
        min_local_rows=250,
        min_cluster_rows=20,
        max_cluster_fit_rows=1200,
        max_recognizer_fit_rows=1200,
        cluster_candidates=(3,),
        allow_side_fallback=True,
    )
    recognizer = ResidualArchetypeRecognizer(
        config=cfg,
        candidate_features=["mkt_shock", "market_breadth", "oi_flush", "base_score"],
    ).fit(frame.iloc[:2000].copy())
    safe = strip_outcomes_for_oos(frame.iloc[2000:2010].copy())
    safe["side_name"] = "unknown"
    out = recognizer.transform_oos(safe)
    assert out["meta_resid_arch_prob__base_low_edge_noise"].eq(1.0).all()
    assert out["meta_resid_arch_confidence"].eq(0.0).all()
    assert out["meta_resid_arch_entropy"].eq(1.0).all()


def test_residual_gmm_defaults_cover_validation_component_sweep() -> None:
    assert ResidualArchetypeConfig().ae_gmm_cluster_candidates == (4, 6, 8, 10, 12)


def test_recognizer_can_disable_local_models() -> None:
    frame = _frame(5000)
    cfg = ResidualArchetypeConfig(
        min_side_rows=500,
        min_local_rows=250,
        min_cluster_rows=10,
        max_cluster_fit_rows=2200,
        max_recognizer_fit_rows=2200,
        cluster_candidates=(3,),
        fit_local_models=False,
        allow_side_fallback=True,
    )
    recognizer = ResidualArchetypeRecognizer(
        config=cfg,
        candidate_features=["mkt_shock", "market_breadth", "oi_flush", "base_score"],
    ).fit(frame.iloc[:4500].copy())
    assert recognizer.side_models
    assert not recognizer.local_models
    assert recognizer.manifest()["fit_local_models"] is False


def test_oos_evaluation_reuses_train_score_reference_and_local_thresholds() -> None:
    frame = _frame(10000)
    cfg = ResidualArchetypeConfig(
        score_col="score_meta_base_soft_label",
        rank_scope="global",
        min_side_rows=500,
        min_local_rows=350,
        min_cluster_rows=10,
        max_recognizer_fit_rows=2500,
        max_recognizer_features=8,
        mutual_info_rows=1500,
        cluster_candidates=(3,),
        allow_side_fallback=False,
        label_mode="economic_semantic",
        semantic_min_temporal_segments=1,
        semantic_min_segment_rows=3,
        random_state=31,
    )
    train = frame.iloc[:8000].copy()
    valid = frame.iloc[8000:].copy()
    recognizer = ResidualArchetypeRecognizer(
        config=cfg,
        candidate_features=["mkt_shock", "market_breadth", "oi_flush", "base_score"],
    ).fit(train)
    prepared = recognizer.prepare_evaluation_targets(valid)
    expected_rank = np.searchsorted(
        recognizer.score_reference_values_,
        valid["score_meta_base_soft_label"].to_numpy(dtype=np.float32),
        side="right",
    ) / len(recognizer.score_reference_values_)
    np.testing.assert_allclose(prepared["reference_rank_pct"], expected_rank, atol=1e-7)
    assert (
        prepared.attrs["ev_equivalent_thresholds"]
        == recognizer.ev_equivalent_thresholds_
    )
    assert recognizer.side_models == {}
    assert recognizer.local_models


def test_discovery_appends_and_hydrates_ledger_only_rows(
    tmp_path, monkeypatch
) -> None:
    data = pd.DataFrame(
        {
            "row_id": ["old"],
            "__ts__": pd.to_datetime(["2026-06-30T23:00:00Z"]),
            "__symbol__": ["BTC/USD:USD"],
            "side_name": ["long"],
            "archetype_policy_key": ["long_a"],
            "score": [0.7],
            "ev_after_1pct": [0.01],
            "clean_exec": [1.0],
            "feat_x": np.array([1.0], dtype=np.float32),
        }
    )
    ledger_path = tmp_path / "ledger.parquet"
    ledger = pd.concat(
        [
            data.drop(columns="feat_x"),
            pd.DataFrame(
                {
                    "row_id": ["new"],
                    "__ts__": pd.to_datetime(["2026-07-01T00:00:00Z"]),
                    "__symbol__": ["BTC/USD:USD"],
                    "side_name": ["long"],
                    "archetype_policy_key": ["long_a"],
                    "score": [0.8],
                    "ev_after_1pct": [0.02],
                    "clean_exec": [1.0],
                }
            ),
        ],
        ignore_index=True,
    )
    pq.write_table(pa.Table.from_pandas(ledger), ledger_path)
    (tmp_path / "symbol=BTC_USD:USD.parquet").touch()

    def fake_read(*_args, **_kwargs):
        return pd.DataFrame(
            {"feat_x": np.array([3.0], dtype=np.float32)},
            index=pd.to_datetime(["2026-07-01T00:00:00Z"]),
        )

    monkeypatch.setattr(discovery_runner, "read_symbol_features", fake_read)
    result, manifest = discovery_runner._append_feature_hydrated_ledger_rows(
        data, ledger_path, tmp_path
    )
    assert len(result) == 2
    assert manifest["appended_rows"] == 1
    assert manifest["rows_with_observable_features"] == 1
    assert float(result.loc[result["row_id"].eq("new"), "feat_x"].iloc[0]) == 3.0


def test_discovery_state_inputs_exclude_post_meta_outputs() -> None:
    frame = pd.DataFrame(
        {
            "score": [0.6],
            "score_regime_calibrated": [0.7],
            "score_meta_uncalibrated": [0.6],
            "hit_probability": [0.8],
            "market_state_feature": [1.0],
        }
    )
    features = discovery_runner._residual_candidate_features(
        frame, "score_regime_calibrated"
    )
    assert "score" in features
    assert "market_state_feature" in features
    assert "score_regime_calibrated" not in features
    assert "score_meta_uncalibrated" not in features
    assert "hit_probability" not in features


def test_local_aegmm_state_can_be_frozen_across_growing_folds(monkeypatch) -> None:
    frame = _frame(2400)
    frame["side_name"] = "long"
    frame["archetype_policy_key"] = "long_a"
    cfg = ResidualArchetypeConfig(
        min_local_rows=200,
        min_cluster_rows=10,
        max_recognizer_fit_rows=800,
        max_recognizer_features=4,
        mutual_info_rows=600,
        use_residual_ae_gmm=True,
        ae_gmm_max_rows=250,
        ae_gmm_max_iter=5,
        label_mode="economic_semantic",
        semantic_min_temporal_segments=1,
        semantic_min_segment_rows=2,
    )
    first = ResidualArchetypeRecognizer(
        cfg, ["mkt_shock", "market_breadth", "oi_flush", "base_score"]
    ).fit(frame.iloc[:1800])
    model = first.local_models[("long", "long_a")]
    frozen = {
        ("long", "long_a"): (
            model.ae_gmm_state,
            model.ae_gmm_input_features,
            model.ae_gmm_output_features,
        )
    }

    def unexpected_refit(*_args, **_kwargs):
        raise AssertionError("frozen AE/GMM must not be refit")

    monkeypatch.setattr(residual_module, "fit_ae_gmm_state", unexpected_refit)
    growing = ResidualArchetypeRecognizer(
        cfg, ["mkt_shock", "market_breadth", "oi_flush", "base_score"]
    )
    growing.frozen_ae_gmm_by_local = frozen
    growing.fit(frame)
    assert growing.local_models[("long", "long_a")].ae_gmm_state is model.ae_gmm_state
