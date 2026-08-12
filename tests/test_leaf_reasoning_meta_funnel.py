from __future__ import annotations

import hashlib
import json
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

import extreme_price_movements.leaf_reasoning_meta_funnel as meta_funnel
from extreme_price_movements.leaf_reasoning_meta_funnel import (
    ClusterTaxonomyContract,
    FROZEN_META_CONTROL_FEATURES,
    FrozenMetaModelSpec,
    MetaFunnelConfig,
    MetaFunnelError,
    MetaTransportGateConfig,
    NestedPredecessorOOFContract,
    build_sequential_arms,
    compare_successor_meta_generations,
    run_leaf_reasoning_meta_funnel,
    select_h6_features_train_only,
    write_immutable_meta_funnel_output,
)


def _ledger(rows: int = 48) -> pd.DataFrame:
    decision = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    side = np.where(np.arange(rows) % 2, "short", "long")
    base = np.linspace(-5.0, 5.0, rows)
    signal = np.linspace(-1.0, 1.0, rows)
    net = base + 18.0 * signal
    frame = pd.DataFrame(
        {
            "candidate_id": [f"c{i:03d}" for i in range(rows)],
            "side_name": side,
            "decision_ts": decision,
            "label_available_ts": decision + pd.Timedelta(hours=2),
            "base_oof_fit_end_ts": decision - pd.Timedelta(hours=1),
            "base_oof_generated_ts": decision,
            "base_same_side_strict_oof": True,
            "base_expected_bps": base,
            "realized_gross_bps": net + 2.0,
            "realized_cost_bps": 2.0,
            "realized_net_bps": net,
            "reasoning_a": signal,
            "reasoning_b": np.cos(np.arange(rows)),
        }
    )
    # The frozen L0 control consumes every declared base-output/context field,
    # including names that cannot be discovered by a prefix heuristic.
    for index, feature in enumerate(FROZEN_META_CONTROL_FEATURES):
        if feature not in frame:
            frame[feature] = np.float64(index + 1) + signal * np.float64(index + 1) / 100.0
    frame["p_adverse"] = 0.2 + 0.05 * signal
    frame["p_weak"] = 0.3 - 0.02 * signal
    frame["p_clear"] = 1.0 - frame["p_adverse"] - frame["p_weak"]
    return frame


def _groups() -> dict[str, tuple[str, ...]]:
    return {"L0": FROZEN_META_CONTROL_FEATURES, "L1": ("reasoning_a",), "H1": ("reasoning_b",)}


def _model_spec() -> FrozenMetaModelSpec:
    return FrozenMetaModelSpec(
        family="lightgbm_lgbmregressor",
        params={"objective": "huber", "n_estimators": 9},
        contract_id="test_frozen_huber_control_v1",
    )


def _lightweight_factory(_: FrozenMetaModelSpec) -> Ridge:
    """Unit-test stand-in; production CLI always instantiates frozen LightGBM."""
    return Ridge(alpha=1.0)


def test_meta_provenance_uses_only_prior_resolved_labels() -> None:
    frame = _ledger()
    result = run_leaf_reasoning_meta_funnel(
        frame, feature_groups=_groups(), model_spec=_model_spec(), model_factory=_lightweight_factory,
        config=MetaFunnelConfig(min_train_rows=4)
    )
    assert result.provenance.strict_prior_resolved.all()
    populated = result.provenance.loc[result.provenance.train_rows.gt(0)]
    assert (pd.to_datetime(populated.max_label_available_used, utc=True) < pd.to_datetime(populated.fit_reference_ts, utc=True)).all()
    predicted = result.predictions
    assert np.allclose(predicted.common_bps_score, predicted.base_expected_bps + predicted.predicted_residual_bps)


def test_top_rank_is_global_not_per_timestamp_or_side() -> None:
    frame = _ledger(20)
    # All high scores live on the long side; a side quota would select short rows.
    frame.loc[frame.side_name.eq("long"), "base_expected_bps"] = 100.0
    frame.loc[frame.side_name.eq("short"), "base_expected_bps"] = -100.0
    result = run_leaf_reasoning_meta_funnel(
        frame, feature_groups={"L0": FROZEN_META_CONTROL_FEATURES}, model_spec=_model_spec(),
        model_factory=_lightweight_factory, config=MetaFunnelConfig(min_train_rows=4),
    )
    selected = result.predictions.loc[result.predictions.arm.eq("L0")].sort_values("common_bps_score", ascending=False).head(1)
    assert selected.side_name.tolist() == ["long"]
    metric = result.metrics.loc[(result.metrics.arm.eq("L0")) & (result.metrics.top_fraction.eq(.01))].iloc[0]
    assert metric.selection_scope == "one_pooled_global_post_common_bps_top_k_per_transport"


def test_strict_identity_allows_reused_candidate_ids_without_collapsing_global_tails() -> None:
    """A generator ID may recur across strict transport/partition rows."""

    frame = _ledger(80)
    frame["candidate_id"] = "reused-generator-candidate"
    frame["transport"] = np.where(np.arange(len(frame)) < 40, "transport_a", "transport_b")
    frame["meta_partition"] = "inner_oof"
    frame.loc[(np.arange(len(frame)) >= 20) & (np.arange(len(frame)) < 40), "meta_partition"] = "outer_test"
    frame.loc[np.arange(len(frame)) >= 60, "meta_partition"] = "outer_test"
    frame["fold_id"] = np.where(frame["meta_partition"].eq("inner_oof"), "inner_fold", "outer_fold")
    result = run_leaf_reasoning_meta_funnel(
        frame, feature_groups={"L0": FROZEN_META_CONTROL_FEATURES}, model_spec=_model_spec(),
        model_factory=_lightweight_factory,
        config=MetaFunnelConfig(min_train_rows=4, fit_protocol="transport_outer_frozen"),
    )
    predicted = result.predictions.loc[result.predictions.arm.eq("L0")]
    assert len(predicted) == 40
    assert predicted.candidate_id.nunique() == 1
    strict_key = [
        "candidate_id", "decision_ts", "side_name", "__strict_fold_id__",
        "__strict_transport__", "__strict_meta_partition__",
    ]
    assert not predicted.duplicated(strict_key).any()
    top10 = result.metrics.loc[(result.metrics.arm.eq("L0")) & result.metrics.top_fraction.eq(.10)]
    assert top10.set_index("transport_id")["population_rows"].to_dict() == {"transport_a": 20, "transport_b": 20}

    duplicated = pd.concat((frame, frame.iloc[[0]]), ignore_index=True)
    with pytest.raises(MetaFunnelError, match="full strict candidate identity"):
        run_leaf_reasoning_meta_funnel(
            duplicated, feature_groups={"L0": FROZEN_META_CONTROL_FEATURES}, model_spec=_model_spec(),
            model_factory=_lightweight_factory,
            config=MetaFunnelConfig(min_train_rows=4, fit_protocol="transport_outer_frozen"),
        )


def test_h6_phantom_selection_is_train_only_and_never_selects_test_only_signal() -> None:
    train = _ledger(60)
    # A feature which is only useful in the final (would-be test) region cannot
    # be selected by the train-only H6 selector.
    train["test_only"] = 0.0
    train.loc[50:, "test_only"] = np.linspace(0.0, 100.0, 10)
    selected, audit = select_h6_features_train_only(
        train.iloc[:50].copy(), ("reasoning_a", "test_only"), config=MetaFunnelConfig(min_train_rows=8, h6_min_holdout_rows=8)
    )
    assert "test_only" not in selected
    assert audit.selection_scope.eq("chronological_train_only_internal_holdout").all()
    assert audit.phantom_q95.notna().all()


def _h6_parity_frame(rows: int = 240) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Exercise missingness, constants, and a near-correlation gate boundary."""
    frame = _ledger(rows)
    rng = np.random.default_rng(20260804)
    primary = rng.normal(size=rows)
    secondary = .79 * primary + rng.normal(scale=.61, size=rows)
    target = 17.0 * primary - 4.0 * secondary + rng.normal(scale=.8, size=rows)
    frame["realized_net_bps"] = frame["base_expected_bps"] + target
    fields = []
    for index in range(13):
        name = f"h6_parity_{index:02d}"
        frame[name] = rng.normal(size=rows)
        fields.append(name)
    frame[fields[0]] = primary
    frame[fields[1]] = secondary
    frame[fields[2]] = 0.0
    frame.loc[::17, fields[3]] = np.nan
    frame.loc[::29, fields[4]] = np.nan
    return frame, tuple(fields)


def test_h6_batched_selector_matches_reference_scores_threshold_and_schema() -> None:
    frame, fields = _h6_parity_frame()
    config = MetaFunnelConfig(
        min_train_rows=8,
        h6_min_holdout_rows=32,
        h6_holdout_fraction=.25,
        random_seed=20260805,
    )
    selected, audit = select_h6_features_train_only(frame, fields, config=config)
    reference_selected, reference_audit = meta_funnel._select_h6_features_train_only_reference(
        frame,
        fields,
        config=config,
    )
    assert selected == reference_selected
    assert list(audit.columns) == list(reference_audit.columns)
    selection_columns = [
        name for name in audit.columns
        if name not in {
            "h6_batched_attempted",
            "h6_batched_fallback",
            "h6_batched_fallback_exception_type",
            "h6_batched_fallback_reason",
            "h6_mda_backend",
        }
    ]
    pd.testing.assert_frame_equal(
        audit.loc[:, selection_columns].sort_values("feature", kind="mergesort").reset_index(drop=True),
        reference_audit.loc[:, selection_columns].sort_values("feature", kind="mergesort").reset_index(drop=True),
        check_exact=False,
        rtol=1e-10,
        atol=1e-10,
    )
    assert audit.selection_scope.eq("chronological_train_only_internal_holdout").all()


def test_h6_batched_selector_is_block_size_invariant() -> None:
    frame, fields = _h6_parity_frame()
    config = MetaFunnelConfig(min_train_rows=8, h6_min_holdout_rows=32, random_seed=23)
    selected, audit = select_h6_features_train_only(frame, fields, config=config)
    original_real = meta_funnel._H6_REAL_FEATURE_BLOCK
    original_phantom = meta_funnel._H6_PHANTOM_BLOCK
    try:
        meta_funnel._H6_REAL_FEATURE_BLOCK = 1
        meta_funnel._H6_PHANTOM_BLOCK = 1
        single_selected, single_audit = select_h6_features_train_only(frame, fields, config=config)
    finally:
        meta_funnel._H6_REAL_FEATURE_BLOCK = original_real
        meta_funnel._H6_PHANTOM_BLOCK = original_phantom
    assert selected == single_selected
    pd.testing.assert_frame_equal(
        audit.sort_values("feature", kind="mergesort").reset_index(drop=True),
        single_audit.sort_values("feature", kind="mergesort").reset_index(drop=True),
        check_exact=False,
        rtol=1e-10,
        atol=1e-10,
    )


def test_h6_batched_selector_falls_back_to_reference_on_linear_algebra_failure(monkeypatch) -> None:
    frame, fields = _h6_parity_frame()
    config = MetaFunnelConfig(min_train_rows=8, h6_min_holdout_rows=32, random_seed=71)
    expected_selected, expected_audit = meta_funnel._select_h6_features_train_only_reference(
        frame,
        fields,
        config=config,
    )

    def fail_batched(*_args, **_kwargs):
        raise np.linalg.LinAlgError("simulated factorisation failure")

    monkeypatch.setattr(meta_funnel, "_h6_batched_phantom_mda", fail_batched)
    selected, audit = select_h6_features_train_only(frame, fields, config=config)
    assert selected == expected_selected
    # The fallback must retain the reference selection calculation exactly;
    # its additional columns are observational audit telemetry only.
    shared = [
        name for name in expected_audit.columns
        if name not in {
            "h6_batched_attempted",
            "h6_batched_fallback",
            "h6_batched_fallback_exception_type",
            "h6_batched_fallback_reason",
            "h6_mda_backend",
        }
    ]
    pd.testing.assert_frame_equal(audit.loc[:, shared], expected_audit.loc[:, shared])
    assert audit["h6_batched_attempted"].eq(True).all()
    assert audit["h6_batched_fallback"].eq(True).all()
    assert audit["h6_batched_fallback_exception_type"].eq("LinAlgError").all()
    assert audit["h6_batched_fallback_reason"].eq("simulated factorisation failure").all()
    assert audit["h6_mda_backend"].eq(
        "reference_after_batched_linear_algebra_failure"
    ).all()
    holdout_rows = max(
        config.h6_min_holdout_rows,
        int(np.ceil(len(frame) * config.h6_holdout_fraction)),
    )
    assert audit["h6_fit_rows"].eq(len(frame) - holdout_rows).all()
    assert audit["h6_valid_rows"].eq(holdout_rows).all()
    assert audit["h6_fit_feature_count"].eq(len(fields)).all()
    assert audit["h6_fit_condition_diagnostic"].eq(
        "not_computed_to_preserve_selector_runtime"
    ).all()


def test_h6_batched_selector_emits_nonfallback_scale_telemetry() -> None:
    frame, fields = _h6_parity_frame()
    config = MetaFunnelConfig(min_train_rows=8, h6_min_holdout_rows=32, random_seed=29)
    _selected, audit = select_h6_features_train_only(frame, fields, config=config)
    assert audit["h6_batched_attempted"].eq(True).all()
    assert audit["h6_batched_fallback"].eq(False).all()
    assert audit["h6_batched_fallback_exception_type"].isna().all()
    assert audit["h6_batched_fallback_reason"].isna().all()
    assert audit["h6_mda_backend"].eq("batched_linear").all()
    assert audit["h6_fit_scale_min"].gt(0.0).all()
    assert audit["h6_fit_scale_median"].ge(audit["h6_fit_scale_min"]).all()
    assert audit["h6_fit_scale_max"].ge(audit["h6_fit_scale_median"]).all()
    assert audit["h6_fit_condition_number"].isna().all()


def test_raw_leaf_columns_are_rejected() -> None:
    frame = _ledger().assign(raw_leaf_id=7)
    with pytest.raises(MetaFunnelError, match="raw fold-local leaf identifiers"):
        run_leaf_reasoning_meta_funnel(frame, feature_groups=_groups(), model_spec=_model_spec(), model_factory=_lightweight_factory)


def test_s2_is_fail_closed_without_valid_nested_predecessor_oof_contract() -> None:
    frame = _ledger()
    with pytest.raises(MetaFunnelError, match="S2 requires"):
        run_leaf_reasoning_meta_funnel(frame, feature_groups=_groups(), successor="S2", model_spec=_model_spec(), model_factory=_lightweight_factory)
    frame["predecessor_feature"] = 1.0
    contract = NestedPredecessorOOFContract(("predecessor_feature",))
    with pytest.raises(MetaFunnelError, match="missing columns"):
        run_leaf_reasoning_meta_funnel(frame, feature_groups=_groups(), successor="S2", predecessor_contract=contract, model_spec=_model_spec(), model_factory=_lightweight_factory)


def test_transport_outer_protocol_fits_only_prior_resolved_inner_rows_and_scores_outer() -> None:
    frame = _ledger(80)
    frame["transport"] = "A_2023q4_to_2024h1"
    frame["meta_partition"] = "inner_oof"
    frame.loc[64:, "meta_partition"] = "outer_test"
    outer_start = frame.loc[frame.meta_partition.eq("outer_test"), "decision_ts"].min()
    # Outcome changes on the untouched outer rows cannot change frozen scores.
    changed = frame.copy()
    changed.loc[changed.meta_partition.eq("outer_test"), "realized_gross_bps"] += 10_000.0
    changed.loc[changed.meta_partition.eq("outer_test"), "realized_net_bps"] += 10_000.0
    config = MetaFunnelConfig(min_train_rows=8, fit_protocol="transport_outer_frozen")
    first = run_leaf_reasoning_meta_funnel(frame, feature_groups=_groups(), model_spec=_model_spec(), model_factory=_lightweight_factory, config=config)
    second = run_leaf_reasoning_meta_funnel(changed, feature_groups=_groups(), model_spec=_model_spec(), model_factory=_lightweight_factory, config=config)
    assert set(first.predictions.candidate_id) == set(frame.loc[64:, "candidate_id"])
    assert len(first.predictions) == len(frame.loc[64:]) * len(first.arms)
    assert first.predictions.meta_fit_protocol.eq("transport_outer_frozen").all()
    assert pd.to_datetime(first.provenance.max_label_available_used, utc=True).lt(outer_start).all()
    assert np.allclose(first.predictions.predicted_residual_bps, second.predictions.predicted_residual_bps)
    assert first.provenance.fit_protocol.eq("transport_outer_frozen").all()
    assert first.complexity.fit_count.eq(2).all()  # one frozen fit per side, not per timestamp
    assert {"incremental_global_top_k_net_bps_vs_l0", "positive_month_count", "worst_month_net_bps"}.issubset(first.metrics.columns)
    assert {"within_side_rank_ic", "false_positive_net_bps", "within_side_decile_monotonic_nonincreasing_fraction"}.issubset(first.side_metrics.columns)
    assert {"score_decile", "net_bps", "gross_bps"}.issubset(first.side_decile_metrics.columns)


def test_l0_requires_frozen_huber_control_and_nonempty_control_feature_subset() -> None:
    frame = _ledger()
    with pytest.raises(MetaFunnelError, match="frozen LightGBM Huber"):
        run_leaf_reasoning_meta_funnel(frame, feature_groups=_groups())
    with pytest.raises(MetaFunnelError, match="L0 must declare"):
        run_leaf_reasoning_meta_funnel(
            frame, feature_groups={"L0": ()}, model_spec=_model_spec(), model_factory=_lightweight_factory,
        )
    incomplete = tuple(feature for feature in FROZEN_META_CONTROL_FEATURES if feature != "negative_breadth_pct")
    with pytest.raises(MetaFunnelError, match="negative_breadth_pct"):
        run_leaf_reasoning_meta_funnel(
            frame, feature_groups={"L0": incomplete}, model_spec=_model_spec(), model_factory=_lightweight_factory,
        )


def test_sparse_frozen_context_is_retained_and_train_only_imputation_is_audited() -> None:
    frame = _ledger(80)
    frame["transport"] = "A_2023q4_to_2024h1"
    frame["meta_partition"] = "inner_oof"
    frame.loc[64:, "meta_partition"] = "outer_test"
    sparse = "market_state_transition_entropy_5d"
    frame[sparse] = np.nan
    frame.loc[5, sparse] = 0.75  # only a prior inner observation; outer stays missing
    result = run_leaf_reasoning_meta_funnel(
        frame, feature_groups=_groups(), model_spec=_model_spec(), model_factory=_lightweight_factory,
        config=MetaFunnelConfig(min_train_rows=8, fit_protocol="transport_outer_frozen"),
    )
    l0_provenance = result.provenance.loc[result.provenance.arm.eq("L0")]
    assert l0_provenance.missing_value_handling.eq("train_only_median_for_injected_factory").all()
    assert l0_provenance.prediction_missing_cells.gt(0).all()
    assert l0_provenance.all_missing_features.gt(0).any()
    assert result.predictions.loc[result.predictions.arm.eq("L0"), "selected_features_json"].str.contains(sparse, regex=False).all()


def test_immutable_output_records_frozen_model_and_rich_metric_tables(tmp_path) -> None:
    result = run_leaf_reasoning_meta_funnel(
        _ledger(), feature_groups=_groups(), model_spec=_model_spec(), model_factory=_lightweight_factory,
        config=MetaFunnelConfig(min_train_rows=4),
    )
    output = write_immutable_meta_funnel_output(result, tmp_path)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["frozen_meta_model"]["contract_id"] == _model_spec().contract_id
    assert manifest["frozen_meta_model"]["params_hash"] == _model_spec().params_hash
    assert manifest["artifact_state"] == "COMPLETE"
    assert manifest["table_format"] == "parquet_zstd"
    expected_tables = {
        "predictions": result.predictions,
        "metrics": result.metrics,
        "side_metrics": result.side_metrics,
        "side_decile_metrics": result.side_decile_metrics,
        "month_metrics": result.month_metrics,
        "transport_metrics": result.transport_metrics,
        "complexity": result.complexity,
        "h6_train_only_selection": result.h6_selection,
        "provenance": result.provenance,
        "ablation_results": result.ablation_results,
        "transport_gates": result.transport_gates,
    }
    assert set(manifest["sha256"]) == {f"{name}.parquet" for name in expected_tables}
    assert not list(output.glob("*.csv"))
    for name, expected in expected_tables.items():
        path = output / f"{name}.parquet"
        assert path.is_file()
        assert manifest["sha256"][path.name] == hashlib.sha256(path.read_bytes()).hexdigest()
        observed = pd.read_parquet(path)
        assert list(observed.columns) == list(expected.columns)
        assert len(observed) == len(expected)
    assert not list(tmp_path.glob(".leaf_reasoning_meta_funnel_*.tmp-*"))


def test_immutable_output_cleans_failed_staging_directory_without_publishing_manifest(tmp_path, monkeypatch) -> None:
    result = run_leaf_reasoning_meta_funnel(
        _ledger(), feature_groups=_groups(), model_spec=_model_spec(), model_factory=_lightweight_factory,
        config=MetaFunnelConfig(min_train_rows=4),
    )
    original = pd.DataFrame.to_parquet
    calls = {"count": 0}

    def fail_after_first_table(self, path, *args, **kwargs):
        calls["count"] += 1
        # While the writer is still staging, no final immutable directory or
        # manifest may be observable by an external consumer.
        assert not list(tmp_path.glob("leaf_reasoning_meta_funnel_*"))
        if calls["count"] == 2:
            raise OSError("simulated disk exhaustion")
        return original(self, path, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_after_first_table)
    with pytest.raises(OSError, match="simulated disk exhaustion"):
        write_immutable_meta_funnel_output(result, tmp_path)
    assert not list(tmp_path.glob("leaf_reasoning_meta_funnel_*"))
    assert not list(tmp_path.glob(".leaf_reasoning_meta_funnel_*.tmp-*"))


def test_bounded_prediction_cache_matches_in_memory_h_stage_and_is_arm_incremental(tmp_path, monkeypatch) -> None:
    """H metrics/audit are identical without retaining the multi-arm panel."""

    frame = _ledger(160)
    frame["transport"] = "A_2023q4_to_2024h1"
    frame["meta_partition"] = "inner_oof"
    frame.loc[128:, "meta_partition"] = "outer_test"
    frame["rule_family_aggregate"] = np.linspace(-1.0, 1.0, len(frame))
    frame["contribution_bundle"] = np.cos(np.arange(len(frame), dtype=float) / 3.0)
    for number in range(1, 6):
        frame[f"health_{number}"] = np.sin(np.arange(len(frame), dtype=float) / (number + 1.0))
    groups = {
        "L0": FROZEN_META_CONTROL_FEATURES,
        "L2": ("rule_family_aggregate",),
        "L3": ("contribution_bundle",),
        "H1": ("health_1",),
        "H2": ("health_2",),
        "H3": ("health_3",),
        "H4": ("health_4",),
        "H5": ("health_5",),
    }
    config = MetaFunnelConfig(
        min_train_rows=8,
        fit_protocol="transport_outer_frozen",
        h6_min_holdout_rows=8,
    )
    memory = run_leaf_reasoning_meta_funnel(
        frame,
        feature_groups=groups,
        model_spec=_model_spec(),
        model_factory=_lightweight_factory,
        config=config,
        stages=("L", "H"),
    )
    cache = tmp_path / "bounded_predictions.parquet"
    appended_rows: list[int] = []
    original_append = meta_funnel._IncrementalPredictionParquet.append

    def record_append(self, chunk):
        appended_rows.append(len(chunk))
        return original_append(self, chunk)

    monkeypatch.setattr(meta_funnel._IncrementalPredictionParquet, "append", record_append)
    bounded = run_leaf_reasoning_meta_funnel(
        frame,
        feature_groups=groups,
        model_spec=_model_spec(),
        model_factory=_lightweight_factory,
        config=config,
        stages=("L", "H"),
        prediction_cache_path=cache,
    )
    assert bounded.predictions.empty
    assert bounded.prediction_cache_path == cache
    assert bounded.prediction_rows == len(memory.predictions)
    # There is exactly one compact evaluated arm panel per append; no H-wide
    # all-arm prediction table is retained in the streaming path.
    assert len(appended_rows) == len(memory.arms)
    assert max(appended_rows) == int(frame.meta_partition.eq("outer_test").sum())
    cached = pd.read_parquet(cache)
    tie = ["arm", "candidate_id", "decision_ts", "side_name", "__strict_fold_id__", "__strict_transport__", "__strict_meta_partition__"]
    pd.testing.assert_frame_equal(
        memory.predictions.sort_values(tie, kind="mergesort").reset_index(drop=True),
        cached.sort_values(tie, kind="mergesort").reset_index(drop=True),
        check_dtype=False,
    )
    for name in ("metrics", "side_metrics", "side_decile_metrics", "month_metrics", "transport_metrics", "complexity", "h6_selection", "provenance", "ablation_results", "transport_gates"):
        pd.testing.assert_frame_equal(getattr(memory, name), getattr(bounded, name), check_dtype=False)

    output = write_immutable_meta_funnel_output(
        bounded,
        tmp_path / "immutable",
        config=config,
        consume_prediction_cache=True,
    )
    assert not cache.exists()
    assert len(pd.read_parquet(output / "predictions.parquet")) == bounded.prediction_rows
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["prediction_materialization"] == "bounded_incremental_parquet_cache"
    assert manifest["prediction_rows"] == bounded.prediction_rows


def test_l_and_h_arms_follow_the_linked_sequential_contract() -> None:
    groups = {
        "L0": FROZEN_META_CONTROL_FEATURES,
        "L1": ("individual_leaf_aggregate",),
        "L2": ("rule_family_aggregate",),
        "L3": ("contribution_bundle",),
        "H1": ("support_posterior",),
        "H2": ("portability_excess_variance",),
        "H3": ("regime_compatibility",),
        "H4": ("covariance_compatibility",),
        "H5": ("relationship_break",),
    }
    arms = {spec.arm: spec for spec in build_sequential_arms(groups)}
    assert "individual_leaf_aggregate" in arms["L1"].features
    assert "individual_leaf_aggregate" not in arms["L2"].features
    assert "individual_leaf_aggregate" not in arms["L3"].features
    assert "individual_leaf_aggregate" not in arms["L4"].features
    assert {"rule_family_aggregate", "contribution_bundle"}.issubset(arms["L4"].features)
    assert arms["H0"].features == arms["L4"].features
    assert "support_posterior" in arms["H1"].features
    assert "support_posterior" in arms["H2"].features
    assert "portability_excess_variance" in arms["H2"].features
    assert arms["H6"].h6_fixed_features == arms["H0"].features
    assert set(arms["H6"].h6_candidate_features) == {
        "support_posterior", "portability_excess_variance", "regime_compatibility",
        "covariance_compatibility", "relationship_break",
    }


def test_s2_reasoning_entropy_group_is_reserved_and_never_an_arm_input() -> None:
    """Only S2 may carry the ledger's entropy hand-off declaration.

    The source field is deliberately retained outside the L/H/C arm graph: it
    is consumed by the nested predecessor materialiser and must not become a
    direct successor feature merely because the complete new ledger is passed
    to the funnel runner.
    """
    groups = {
        "L0": FROZEN_META_CONTROL_FEATURES,
        "S2_reasoning_entropy": ("base_reasoning__family_contribution_entropy",),
    }
    arms = build_sequential_arms(groups, successor="S2")
    assert all(
        "base_reasoning__family_contribution_entropy" not in spec.features
        for spec in arms
    )

    for successor in ("S0", "S1"):
        with pytest.raises(MetaFunnelError, match="only S2 may additionally"):
            build_sequential_arms(groups, successor=successor)

    with pytest.raises(MetaFunnelError, match="reserved for the exact causal ledger field"):
        build_sequential_arms(
            {"L0": FROZEN_META_CONTROL_FEATURES, "S2_reasoning_entropy": ("arbitrary_field",)},
            successor="S2",
        )
    with pytest.raises(MetaFunnelError, match="only S2 may additionally"):
        build_sequential_arms(
            {"L0": FROZEN_META_CONTROL_FEATURES, "S3_arbitrary": ("anything",)},
            successor="S2",
        )


def test_h6_keeps_frozen_l4_control_when_train_only_mda_selects_no_health_field() -> None:
    frame = _ledger(80)
    frame["transport"] = "A_2023q4_to_2024h1"
    frame["meta_partition"] = "inner_oof"
    frame.loc[64:, "meta_partition"] = "outer_test"
    frame["rule_family_aggregate"] = np.arange(len(frame), dtype=float)
    # H1 is deliberately constant, so a train-only selector may retain no
    # health field.  H6 must still contain every frozen L4 control field.
    frame["support_posterior"] = 0.0
    frame["contribution_bundle"] = 0.0
    for name in ("portability_excess", "regime_compatibility", "covariance_compatibility", "relationship_break"):
        frame[name] = 0.0
    groups = {
        "L0": FROZEN_META_CONTROL_FEATURES,
        "L2": ("rule_family_aggregate",),
        "L3": ("contribution_bundle",),
        "H1": ("support_posterior",),
        "H2": ("portability_excess",),
        "H3": ("regime_compatibility",),
        "H4": ("covariance_compatibility",),
        "H5": ("relationship_break",),
    }
    result = run_leaf_reasoning_meta_funnel(
        frame,
        feature_groups=groups,
        model_spec=_model_spec(),
        model_factory=_lightweight_factory,
        config=MetaFunnelConfig(min_train_rows=8, fit_protocol="transport_outer_frozen", h6_min_holdout_rows=8),
        stages=("L", "H"),
    )
    h6 = result.predictions.loc[result.predictions.arm.eq("H6")]
    for selected in h6["selected_features_json"]:
        assert set(build_sequential_arms(groups)[4].features).issubset(set(json.loads(selected)))
    assert result.h6_selection.selection_role.eq("H1_H5_train_only_candidate").all()


def _cluster_contract() -> ClusterTaxonomyContract:
    return ClusterTaxonomyContract(
        linkage="complete",
        cluster_ids_by_arm={
            "C1": ("c1_a", "c1_b"),
            "C2": ("c2_a",),
            "C3": ("c3_a", "c3_b"),
            "C4": ("c4_a",),
            "C5": ("c1_a",),
            "C6": ("c1_a",),
        },
        c5_source_arm="C1",
        top_decile_coverage_by_arm={"C5": 0.95},
        portable_top_decile_coverage_by_arm={"C5": 0.82},
        c6_best_cross_era_score=0.10,
        c6_best_cross_era_standard_error=0.02,
        c6_compact_cross_era_score=0.09,
    )


def test_c_taxonomy_is_threshold_specific_and_rejects_single_linkage() -> None:
    groups = {"L0": FROZEN_META_CONTROL_FEATURES}
    cluster_groups = {
        "C0": ("frozen_h6_feature",),
        "C1": ("cluster_at_060",),
        "C2": ("cluster_at_070",),
        "C3": ("cluster_at_080",),
        "C4": ("cluster_at_090",),
        "C5": ("cluster_coverage_95",),
        "C6": ("cluster_one_se",),
    }
    arms = {spec.arm: spec for spec in build_sequential_arms(groups, cluster_groups, cluster_taxonomy=_cluster_contract())}
    assert arms["C1"].features == ("frozen_h6_feature", "cluster_at_060")
    assert arms["C2"].features == ("frozen_h6_feature", "cluster_at_070")
    assert arms["C5"].features == ("frozen_h6_feature", "cluster_coverage_95")
    assert arms["C1"].cluster_similarity_threshold == pytest.approx(0.60)
    with pytest.raises(MetaFunnelError, match="single linkage"):
        ClusterTaxonomyContract(
            linkage="single",
            cluster_ids_by_arm={
                "C1": (), "C2": (), "C3": (), "C4": (), "C5": (), "C6": (),
            },
        )
    with pytest.raises(MetaFunnelError, match="one-SE"):
        ClusterTaxonomyContract(
            linkage="average",
            cluster_ids_by_arm={
                "C1": ("c1",), "C2": (), "C3": (), "C4": (), "C5": ("c1",), "C6": ("c1",),
            },
            c5_source_arm="C1",
            top_decile_coverage_by_arm={"C5": 0.95},
        )


def test_c_stage_threshold_sweep_selects_only_available_c0_to_c4_arms() -> None:
    """C5/C6 are unavailable until their post-sweep immutable overlay exists."""
    groups = {"L0": FROZEN_META_CONTROL_FEATURES}
    cluster_groups = {
        "C0": FROZEN_META_CONTROL_FEATURES,
        "C1": ("cluster_at_060",),
        "C2": ("cluster_at_070",),
        "C3": ("cluster_at_080",),
        "C4": ("cluster_at_090",),
    }
    taxonomy = ClusterTaxonomyContract(
        linkage="complete",
        selection_phase="threshold_sweep",
        cluster_ids_by_arm={
            "C1": ("c1",),
            "C2": ("c2",),
            "C3": ("c3",),
            "C4": ("c4",),
        },
    )
    all_arms = build_sequential_arms(groups, cluster_groups, cluster_taxonomy=taxonomy)

    selected = meta_funnel._select_stage_arms(
        all_arms,
        stages=("C",),
        feature_groups=groups,
        cluster_groups=cluster_groups,
    )

    assert tuple(spec.arm for spec in selected) == ("L0", "C0", "C1", "C2", "C3", "C4")
    assert {spec.arm for spec in selected}.isdisjoint({"C5", "C6"})


def test_stage_controls_and_transport_gate_artifacts_are_explicit() -> None:
    frame = _ledger(80)
    frame["transport"] = "A_2023q4_to_2024h1"
    frame["meta_partition"] = "inner_oof"
    frame.loc[64:, "meta_partition"] = "outer_test"
    frame["reasoning_rule"] = np.linspace(-1.0, 1.0, len(frame))
    frame["contribution_bundle"] = np.linspace(1.0, 0.0, len(frame))
    frame["portability_excess"] = np.cos(np.arange(len(frame)))
    frame["regime_compatibility"] = np.sin(np.arange(len(frame)))
    frame["covariance_compatibility"] = np.linspace(0.0, 1.0, len(frame))
    frame["relationship_break"] = np.linspace(1.0, 0.0, len(frame))
    result = run_leaf_reasoning_meta_funnel(
        frame,
        feature_groups={"L0": FROZEN_META_CONTROL_FEATURES, "L2": ("reasoning_rule",), "L3": ("contribution_bundle",), "H1": ("reasoning_b",), "H2": ("portability_excess",), "H3": ("regime_compatibility",), "H4": ("covariance_compatibility",), "H5": ("relationship_break",)},
        model_spec=_model_spec(),
        model_factory=_lightweight_factory,
        config=MetaFunnelConfig(min_train_rows=8, fit_protocol="transport_outer_frozen"),
        stages=("L", "H"),
    )
    controls = result.metrics.groupby("arm", observed=True)["control_arm"].first().to_dict()
    assert controls["L2"] == "L0"
    assert controls["H0"] == "L4"
    assert controls["H1"] == "H0"
    assert result.transport_gates.grouped_transport_mda_evidence_present.eq(False).all()
    assert result.transport_gates.passes_all_advancement_gates.eq(False).all()


def test_s2_rejects_predecessor_noop_and_requires_compact_base_reasoning() -> None:
    frame = _ledger(80)
    frame["predecessor_feature"] = 1.0
    frame["predecessor_oof_fit_end_ts"] = frame["decision_ts"] - pd.Timedelta(hours=2)
    frame["predecessor_oof_generated_ts"] = frame["decision_ts"] - pd.Timedelta(hours=1)
    frame["predecessor_oof_available_ts"] = frame["decision_ts"]
    frame["predecessor_same_side_strict_oof"] = True
    contract = NestedPredecessorOOFContract(("predecessor_feature",))
    with pytest.raises(MetaFunnelError, match="compact base reasoning"):
        run_leaf_reasoning_meta_funnel(
            frame,
            feature_groups={"L0": FROZEN_META_CONTROL_FEATURES, "L3": ("predecessor_feature",)},
            successor="S2",
            predecessor_contract=contract,
            model_spec=_model_spec(),
            model_factory=_lightweight_factory,
            config=MetaFunnelConfig(min_train_rows=8),
        )


def test_s1_requires_a_base_reasoning_representation() -> None:
    with pytest.raises(MetaFunnelError, match="S1 requires compact base reasoning"):
        run_leaf_reasoning_meta_funnel(
            _ledger(),
            feature_groups={"L0": FROZEN_META_CONTROL_FEATURES},
            successor="S1",
            model_spec=_model_spec(),
            model_factory=_lightweight_factory,
            config=MetaFunnelConfig(min_train_rows=4),
        )


def test_health_stage_refuses_empty_proxy_groups() -> None:
    with pytest.raises(MetaFunnelError, match="materialised causal H1--H5"):
        run_leaf_reasoning_meta_funnel(
            _ledger(),
            feature_groups={"L0": FROZEN_META_CONTROL_FEATURES, "L2": ("reasoning_a",), "L3": ("reasoning_b",), "H1": ("reasoning_a",)},
            model_spec=_model_spec(),
            model_factory=_lightweight_factory,
            config=MetaFunnelConfig(min_train_rows=4),
            stages=("H",),
        )


def test_s2_requires_a_matched_positive_lift_over_s1_in_both_transports() -> None:
    def metrics(arm: str, value: float) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"arm": arm, "transport_id": transport, "top_fraction": fraction, "net_bps": value, "gross_bps": value + 100.0, "cost_bps": 100.0}
                for transport in ("A", "B")
                for fraction in (0.05, 0.10)
            ]
        )
    s2 = metrics("winner", 12.0)
    s2.loc[(s2.transport_id.eq("B")) & (s2.top_fraction.eq(0.10)), "net_bps"] = 9.0
    rejected = compare_successor_meta_generations(
        {"S0": metrics("control", 8.0), "S1": metrics("reasoning", 10.0), "S2": s2},
        selected_arm_by_generation={"S0": "control", "S1": "reasoning", "S2": "winner"},
        gate_config=MetaTransportGateConfig(required_transport_count=2),
    )
    assert rejected.terminal_decision.eq("PREDECESSOR_META_RECURSION_REJECTED").all()
    s2.loc[(s2.transport_id.eq("B")) & (s2.top_fraction.eq(0.10)), "net_bps"] = 12.0
    advanced = compare_successor_meta_generations(
        {"S0": metrics("control", 8.0), "S1": metrics("reasoning", 10.0), "S2": s2},
        selected_arm_by_generation={"S0": "control", "S1": "reasoning", "S2": "winner"},
        gate_config=MetaTransportGateConfig(required_transport_count=2),
    )
    assert advanced.terminal_decision.eq("PREDECESSOR_META_REASONING_ADDS_VALUE").all()
