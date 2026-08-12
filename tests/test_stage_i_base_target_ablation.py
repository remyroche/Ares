from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_i_base_target_ablation import (
    BaseTargetAblationError,
    BarrierGeometry,
    DevelopmentModelCache,
    Round1Gates,
    causal_map_oof_scores,
    chronological_fold_vector,
    development_holdout_split,
    cumulative_ordinal_targets,
    geometry_grid,
    materialize_geometry_labels,
    materialize_geometry_labels_from_traversal,
    materialize_h12_geometry_traversal,
    materialize_h12_path_primitives,
    ordinal_o_target,
    pooled_global_tail_metrics,
    recover_ordinal_simplex,
    require_selected_feature_contract,
    round1_screen,
    robust_top10_lift_score,
    scalar_s_target,
    run_development_holdout_arm,
    target_arm_grid,
    training_weights,
    validate_entry_timing,
    verify_completed_manifest,
    file_sha256,
)
from extreme_price_movements.stage_i_base_target_ablation import (
    _fit_cached_development_model,
    fit_predict_target_arm,
)


def _paths(rows: int = 3) -> tuple[np.ndarray, ...]:
    high = np.full((rows, 720), 100.0)
    low = np.full((rows, 720), 100.0)
    close = np.full((rows, 720), 100.0)
    return high, low, close


def test_grid_is_15_geometries_60_arms_with_only_strict_cells_promotable() -> None:
    geometries = geometry_grid()
    arms = target_arm_grid()
    assert len(geometries) == 15
    assert sum(item.promotion_eligible_geometry for item in geometries) == 12
    assert len(arms) == 60
    assert sum(item.promotion_eligible for item in arms) == 48
    diagnostic = {item.key for item in geometries if not item.promotion_eligible_geometry}
    assert diagnostic == {"sl3_tp3", "sl4_tp3", "sl4_tp4"}


def test_geometry_uses_floor_cap_and_adverse_same_minute_precedence() -> None:
    high, low, close = _paths()
    # ATR fractions .1%, 1%, 2%; unclipped upper 0.3%, 3%, 6% -> 1.5%,3%,4%.
    atr = np.array([0.1, 1.0, 2.0])
    # First row touches both clipped TP and SL on minute zero: adverse wins.
    high[0, 0], low[0, 0] = 102.0, 99.7
    # Long upper, short upper.
    high[1, 1] = 103.1
    low[2, 2] = 95.5
    result = materialize_geometry_labels(
        entry_price=np.full(3, 100.0), atr=atr, side_sign=np.array([1, 1, -1]),
        high=high, low=low, close=close, path_complete=np.ones(3, bool),
        geometry=BarrierGeometry(2, 3),
    )
    assert result.event.tolist() == [0, 2, 2]
    np.testing.assert_allclose(result.upper_fraction, [.015, .03, .04])
    assert result.upper_floor_bound.tolist() == [True, False, False]
    assert result.upper_cap_bound.tolist() == [False, False, True]
    # Physical loss is 2 ATR; cost is subtracted exactly once only in net economics.
    assert result.gross_bps[0] == pytest.approx(-20.0)
    assert result.net_bps[0] == pytest.approx(-120.0)


def test_geometry_distinguishes_timeout_sentinel_from_real_same_bar_tie() -> None:
    high, low, close = _paths(2)
    # Row zero never touches either barrier. Row one touches both on minute 7.
    high[1, 7], low[1, 7] = 104.0, 97.0
    result = materialize_geometry_labels(
        entry_price=np.full(2, 100.0),
        atr=np.ones(2),
        side_sign=np.ones(2, dtype=np.int8),
        high=high,
        low=low,
        close=close,
        path_complete=np.ones(2, dtype=bool),
        geometry=BarrierGeometry(2, 3),
    )
    assert result.event.tolist() == [1, 0]
    assert result.event_minute.tolist() == [-1, 7]


def test_round1_rejects_nominal_soft_families_collapsed_to_hard_extremes() -> None:
    rows: list[dict[str, object]] = []
    for geometry in geometry_grid():
        for i, event in enumerate((0, 2)):
            rows.append(
                {
                    "candidate_id": f"{geometry.key}-{i}",
                    "__ts__": pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(hours=i),
                    "__symbol__": "X",
                    "decision_ts": pd.Timestamp("2025-01-02", tz="UTC") + pd.Timedelta(hours=i),
                    "side_name": "long",
                    "causal_regime": "stable",
                    "geometry": geometry.key,
                    "event": event,
                    "gross_bps": -200.0 if event == 0 else 300.0,
                    "net_bps": -300.0 if event == 0 else 200.0,
                    "target_valid": True,
                    "upper_floor_bound": False,
                    "upper_cap_bound": False,
                    "S_target": float(event == 2),
                    "O_a0p25_target": 4 if event == 2 else 0,
                    "O_a0p33_target": 4 if event == 2 else 0,
                    "O_a0p5_target": 4 if event == 2 else 0,
                }
            )
    metrics, gates = round1_screen(
        pd.DataFrame(rows),
        gates=Round1Gates(
            min_upper_support_rows=1,
            max_timeout_prevalence=0.9,
            min_worst_regime_upper_rate=0.0,
            min_oracle_top10_net_bps=-1_000.0,
        ),
        regime_column="causal_regime",
    )
    assert not metrics["family_support_ok"].any()
    scalar_reasons = gates.loc[gates.arm.str.startswith("S__"), "rejection_reasons"]
    ordinal_reasons = gates.loc[gates.arm.str.startswith("O_"), "rejection_reasons"]
    assert scalar_reasons.str.contains("degenerate_hard_first_touch_scalar").all()
    assert ordinal_reasons.str.contains("degenerate_hard_first_touch_ordinal").all()


def test_one_primitive_traversal_is_exactly_equal_for_all_15_geometries() -> None:
    rng = np.random.default_rng(771)
    rows = 17
    entry = rng.uniform(50, 250, rows)
    atr = rng.uniform(.05, 5, rows)
    side = rng.choice(np.array([-1, 1], dtype=np.int8), rows)
    increments = rng.normal(0, .001, size=(rows, 720))
    middle = entry[:, None] * np.exp(np.cumsum(increments, axis=1))
    spread = rng.uniform(0, .003, size=(rows, 720))
    high = middle * (1 + spread)
    low = middle * (1 - spread)
    close = middle.copy()
    complete = np.ones(rows, dtype=bool)
    primitive = materialize_h12_path_primitives(
        entry_price=entry, atr=atr, side_sign=side, high=high, low=low,
        close=close, path_complete=complete,
    )
    traversal = materialize_h12_geometry_traversal(primitive)
    assert len(traversal.upper_first) == 5
    assert len(traversal.lower_first) == 3
    for geometry in geometry_grid():
        cached = materialize_geometry_labels_from_traversal(traversal, geometry)
        direct = materialize_geometry_labels(
            entry_price=entry, atr=atr, side_sign=side, high=high, low=low,
            close=close, path_complete=complete, geometry=geometry,
        )
        for field in cached.__dataclass_fields__:
            np.testing.assert_array_equal(
                getattr(cached, field), getattr(direct, field),
                err_msg=f"geometry parity failed for {geometry.key}/{field}",
            )


def test_invalid_or_incomplete_paths_never_receive_targets() -> None:
    high, low, close = _paths(2)
    high[0, 0] = np.nan
    result = materialize_geometry_labels(
        entry_price=np.full(2, 100.0), atr=np.ones(2), side_sign=np.ones(2, dtype=np.int8),
        high=high, low=low, close=close, path_complete=np.array([True, False]),
        geometry=BarrierGeometry(2, 3),
    )
    assert not result.valid.any()
    assert result.event.tolist() == [-1, -1]
    assert np.isnan(scalar_s_target(result.event, result.dominance)).all()
    assert ordinal_o_target(result.event, result.dominance, .25).tolist() == [-1, -1]


def test_missing_path_complete_flag_fails_closed() -> None:
    high, low, close = _paths(2)
    result = materialize_geometry_labels(
        entry_price=np.full(2, 100.0),
        atr=np.ones(2),
        side_sign=np.ones(2, dtype=np.int8),
        high=high,
        low=low,
        close=close,
        path_complete=np.asarray([1.0, np.nan]),
        geometry=BarrierGeometry(2, 3),
    )
    assert result.valid.tolist() == [True, False]
    assert result.event.tolist() == [1, -1]


def test_scalar_and_ordinal_label_math() -> None:
    event = np.array([0, 2, 1, 1, 1], dtype=np.int8)
    dominance = np.array([0, 0, -.6, 0, .6], dtype=float)
    scalar = scalar_s_target(event, dominance)
    assert scalar[0] == 0 and scalar[1] == 1
    assert .35 < scalar[2] < scalar[3] < scalar[4] < .65
    assert ordinal_o_target(event, dominance, .25).tolist() == [0, 4, 1, 2, 3]
    cumulative = cumulative_ordinal_targets(np.array([0, 1, 2, 3, 4]))
    assert cumulative.tolist() == [
        [0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0],
        [1, 1, 1, 0], [1, 1, 1, 1],
    ]


def test_ordinal_recovery_is_monotone_and_a_probability_simplex() -> None:
    raw = np.array([[.2, .9, .4, .7], [.9, .8, .3, .1]])
    simplex = recover_ordinal_simplex(raw)
    np.testing.assert_allclose(simplex.sum(axis=1), 1.0)
    assert (simplex >= 0).all()
    survival = 1 - np.cumsum(simplex, axis=1)[:, :-1]
    assert np.all(np.diff(survival, axis=1) <= 1e-8)


def test_exact_next_minute_entry_contract() -> None:
    signal = pd.to_datetime(["2026-01-01T00:00:00Z"])
    validate_entry_timing(signal, signal + pd.Timedelta(hours=1), signal + pd.Timedelta(hours=1))
    with pytest.raises(BaseTargetAblationError, match=r"signal \+1h"):
        validate_entry_timing(
            signal, signal + pd.Timedelta(hours=1), signal + pd.Timedelta(hours=1, minutes=1)
        )


def test_robust_score_formula_is_exact() -> None:
    eras = np.array([2.0, 1.0, -0.5])
    # median=1, MAD=1, worst penalty=.5
    assert robust_top10_lift_score(3.0, eras) == pytest.approx(1.0)


def _mapping_frame() -> pd.DataFrame:
    rows = []
    for side in ("long", "short"):
        for day in range(1, 9):
            rows.append({
                "candidate_id": f"{side}-{day}", "__ts__": pd.Timestamp(f"2026-01-{day:02}T00:00Z"),
                "__symbol__": "X", "side_name": side,
                "decision_ts": pd.Timestamp(f"2026-01-{day:02}T01:00Z"),
                "label_available_ts": pd.Timestamp(f"2026-01-{day:02}T13:00Z"),
                "fold": 0 if day <= 4 else 1, "raw": float(day),
                "net_bps": float(day * 10 + (100 if side == "short" else 0)),
            })
    return pd.DataFrame(rows)


def test_causal_map_uses_prior_resolved_rows_and_global_rank_is_after_mapping() -> None:
    frame = _mapping_frame()
    mapped = causal_map_oof_scores(frame, score_column="raw", fold_column="fold", min_rows=4)
    # First fold has no prior labels and is unmapped; second is causally mapped.
    assert mapped.loc[mapped.fold.eq(0), "expected_net_bps"].isna().all()
    assert mapped.loc[mapped.fold.eq(1), "expected_net_bps"].notna().all()
    metrics = pooled_global_tail_metrics(mapped, fractions=(.50,))
    assert metrics.iloc[0].ranking_policy.startswith("pooled_global_common_bps")
    # Common-bps conversion makes the short side dominate the pooled book.
    assert metrics.iloc[0].short_rows == metrics.iloc[0].selected_rows


def test_chronological_folds_purge_unresolved_and_hybrid_weights_are_bounded() -> None:
    decision = pd.date_range("2026-01-01", periods=12, freq="D", tz="UTC")
    available = decision + pd.Timedelta(hours=12)
    fold = chronological_fold_vector(decision, available, folds=3, min_train_rows=3)
    assert (fold[:3] == -1).all()
    for fold_id in range(3):
        held = np.flatnonzero(fold == fold_id)
        assert len(held)
        assert (available < decision[held].min()).sum() >= 3
    frame = pd.DataFrame({
        "decision_ts": decision,
        "contract_certainty": np.linspace(0, 1, len(decision)),
        "causal_regime": ["a"] * 9 + ["b"] * 3,
    })
    weight = training_weights(
        frame, target=np.array([0, 1, 2] * 4), mode="hybrid", regime_column="causal_regime"
    )
    assert weight.min() >= .25 and weight.max() <= 4
    assert weight.mean() == pytest.approx(1.0)


def test_single_development_holdout_is_whole_timestamp_and_purged() -> None:
    decision = pd.date_range("2025-01-01", periods=100, freq="h", tz="UTC")
    available = decision + pd.Timedelta(hours=12)
    split = development_holdout_split(
        decision, available, evaluation_fraction=.25, min_train_rows=50,
    )
    assert split.evaluation_mask.sum() == 25
    assert split.train_mask.sum() == 63
    assert split.purged_pre_evaluation_rows == 12
    assert (available[split.train_mask] < split.evaluation_start).all()
    assert (decision[split.evaluation_mask] >= split.evaluation_start).all()


@pytest.mark.parametrize(
    ("family", "target"),
    [
        ("scalar_S", np.linspace(0, 1, 160, dtype=np.float32)),
        ("ordinal_O", np.tile(np.arange(5, dtype=np.int8), 32)),
        ("R3_control", np.tile(np.arange(3, dtype=np.int8), 54)[:160]),
    ],
)
def test_cached_lightgbm_bins_and_batched_heads_match_reference_wrapper(
    family: str, target: np.ndarray,
) -> None:
    rng = np.random.default_rng(19)
    rows = 160
    decision = pd.date_range("2025-01-01", periods=rows // 2, freq="h", tz="UTC").repeat(2)
    frame = pd.DataFrame({
        "candidate_id": [f"c-{item}" for item in range(rows)],
        "__ts__": decision - pd.Timedelta(hours=1), "__symbol__": ["X"] * rows,
        "side_name": np.tile(["long", "short"], rows // 2),
        "decision_ts": decision, "label_available_ts": decision + pd.Timedelta(hours=12),
        "causal_regime": np.tile(["a", "b"], rows // 2),
        "f0": rng.normal(size=rows), "f1": rng.normal(size=rows),
    })
    params = {
        side: {"num_leaves": 7, "n_estimators": 15, "learning_rate": .05,
               "min_child_samples": 5, "max_bin": 31}
        for side in ("long", "short")
    }
    cache = DevelopmentModelCache(
        frame, selected_features={"long": ["f0", "f1"], "short": ["f0", "f1"]},
        fixed_params=params, evaluation_fraction=.25, min_train_rows=20, seed=11,
    )
    prepared = cache.sides["long"]
    train = frame.iloc[prepared.train_positions].copy()
    held = frame.iloc[prepared.evaluation_positions].copy()
    local_target = target[prepared.train_positions]
    train["target"] = local_target
    held["target"] = target[prepared.evaluation_positions]
    weights = np.ones(len(train), dtype=np.float32)
    cached_train, cached_eval, audit = _fit_cached_development_model(
        prepared, target=local_target, weight=weights, family=family,
        fixed_params=params["long"], seed=11,
    )
    reference_eval, _ = fit_predict_target_arm(
        train, held, features=["f0", "f1"], target_column="target", family=family,
        fixed_params=params["long"], seed=11, weight_mode="uniform",
        regime_column="causal_regime",
    )
    reference_train, _ = fit_predict_target_arm(
        train, train, features=["f0", "f1"], target_column="target", family=family,
        fixed_params=params["long"], seed=11, weight_mode="uniform",
        regime_column="causal_regime",
    )
    np.testing.assert_allclose(cached_eval, reference_eval, rtol=0, atol=1e-7)
    np.testing.assert_allclose(cached_train, reference_train, rtol=0, atol=1e-7)
    assert audit["models"] == (4 if family == "ordinal_O" else 1)


def test_development_runner_maps_holdout_only_from_prior_resolved_training_rows() -> None:
    rng = np.random.default_rng(31)
    rows = 240
    decision = pd.date_range("2025-01-01", periods=rows // 2, freq="h", tz="UTC").repeat(2)
    side = np.tile(["long", "short"], rows // 2)
    signal = rng.normal(size=rows)
    frame = pd.DataFrame({
        "candidate_id": [f"d-{item}" for item in range(rows)],
        "__ts__": decision - pd.Timedelta(hours=1), "__symbol__": ["X"] * rows,
        "side_name": side, "decision_ts": decision,
        "label_available_ts": decision + pd.Timedelta(hours=12),
        "causal_regime": np.where(signal > 0, "trend", "chop"),
        "net_bps": 50 * signal + rng.normal(0, 10, rows),
        "target": 1 / (1 + np.exp(-signal)), "f0": signal, "f1": rng.normal(size=rows),
    })
    params = {
        name: {"num_leaves": 7, "n_estimators": 12, "learning_rate": .05,
               "min_child_samples": 5, "max_bin": 31}
        for name in ("long", "short")
    }
    prediction, reference, audit, cache = run_development_holdout_arm(
        frame, arm=None, target_column="target", family="scalar_S",
        selected_features={"long": ["f0", "f1"], "short": ["f0", "f1"]},
        fixed_params=params, seed=11, min_train_rows=30, weight_mode="uniform",
        regime_column="causal_regime", evaluation_fraction=.25,
    )
    assert prediction.split_role.eq("held_out_development_evaluation").all()
    assert reference.split_role.eq("prior_resolved_training_reference").all()
    assert pd.to_datetime(reference.label_available_ts, utc=True).lt(cache.split.evaluation_start).all()
    assert pd.to_datetime(prediction.decision_ts, utc=True).ge(cache.split.evaluation_start).all()
    assert prediction.expected_net_bps.notna().all()
    assert all(item["strict_prior_resolved"] for item in audit)


def test_selected_feature_lineage_and_resume_fail_closed(tmp_path: Path) -> None:
    selector = tmp_path / "selector"
    selection = tmp_path / "selection"
    selector.mkdir()
    pd.DataFrame({"candidate_id": ["x"]}).to_parquet(selector / "selector_ledger.parquet")
    pd.DataFrame({"candidate_id": ["x"], "f": [1.]}).to_parquet(selector / "selector_features.parquet")
    integrity = {
        "schema": "stage_i_selector_artifact_integrity_v1",
        "selector_ledger_sha256": file_sha256(selector / "selector_ledger.parquet"),
        "selector_features_sha256": file_sha256(selector / "selector_features.parquet"),
    }
    (selector / "manifest.json").write_text(json.dumps({
        "status": "complete", "feature_cap_policy": "uncapped", "artifact_integrity": integrity,
    }))
    (selector / "selector_feature_contract.json").write_text(json.dumps({
        "max_feature_columns": 0, "feature_columns": ["f"],
    }))
    selector_sha = file_sha256(selector / "manifest.json")
    for side in ("long", "short"):
        root = selection / side
        root.mkdir(parents=True)
        (root / "manifest.json").write_text(json.dumps({
            "schema": "stage_i_base_feature_selection_v1", "status": "complete", "side": side,
            "selector_sample_manifest_sha256": selector_sha,
            "selected_features": ["f"], "selected_feature_contract": ["f"],
            "input_feature_contract": ["f"], "selector_artifact_integrity": integrity,
            "best_params": {"num_leaves": 7},
        }))
    contract = require_selected_feature_contract(selector_dir=selector, base_selection_dir=selection)
    assert contract["sides"]["long"]["selected_features"] == ["f"]
    out = tmp_path / "out"; out.mkdir()
    artifact = out / "metrics.parquet"; pd.DataFrame({"x": [1]}).to_parquet(artifact)
    request = "abc"
    (out / "manifest.json").write_text(json.dumps({
        "status": "complete", "request_sha256": request,
        "artifact_sha256": {"metrics.parquet": file_sha256(artifact)},
    }))
    assert verify_completed_manifest(out, request) is not None
    pd.DataFrame({"x": [2]}).to_parquet(artifact)
    with pytest.raises(BaseTargetAblationError, match="artifact drift"):
        verify_completed_manifest(out, request)
