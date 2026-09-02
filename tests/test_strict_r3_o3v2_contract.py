from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from extreme_price_movements.strict_r3_rich_policy import RichPolicyParams


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SEMANTICS = _load("strict_r3_o3v2_semantics_test", "materialize_strict_r3_o3v2_semantics.py")
FUNNEL = _load("strict_r3_o3v2_funnel_test", "run_strict_r3_o3v2_target_funnel.py")
SUPPORT = _load("strict_r3_o3v2_support_test", "run_strict_r3_o3v2_support_funnel.py")
SUPPORT_V2 = _load("strict_r3_o3v2_support_v2_test", "run_strict_r3_o3v2_support_funnel_v2.py")
SUPPORT_V3 = _load("strict_r3_o3v2_support_v3_test", "run_strict_r3_o3v2_support_funnel_v3.py")
FEATURE_SCREEN = _load("strict_r3_o3v2_feature_screen_test", "run_strict_r3_o3v2_feature_screen.py")
SPECIALISTS = _load("strict_r3_o3v2_specialists_test", "run_strict_r3_o3v2_specialist_funnel.py")
T2_ADAPTER = _load("strict_r3_o3v2_t2_adapter_test", "materialize_strict_r3_o3v2_t2_f4f5_adapter.py")
TARGET_SELECTION = _load("strict_r3_o3v2_target_selection_test", "select_strict_r3_o3v2_targets.py")
SUPPORT_SELECTION = _load("strict_r3_o3v2_support_selection_test", "select_strict_r3_o3v2_support.py")
QUERY_SELECTION = _load("strict_r3_o3v2_query_selection_test", "select_strict_r3_o3v2_query.py")
GREEDY = _load("strict_r3_o3v2_greedy_test", "run_strict_r3_o3v2_greedy_features.py")
QUERY_SCREEN = _load("strict_r3_o3v2_query_screen_test", "run_strict_r3_o3v2_fixed_contract_query_screen.py")
PHYSICAL_SLOTS = _load("strict_r3_o3v2_physical_slots_test", "select_strict_r3_o3v2_physical_slots.py")
AUXILIARY_PATH = _load("strict_r3_o3v2_auxiliary_path_test", "materialize_strict_r3_o3v2_path_auxiliary_labels.py")
PATH_AUXILIARY_FUNNEL = _load("strict_r3_o3v2_path_auxiliary_funnel_test", "run_strict_r3_o3v2_path_auxiliary_funnel.py")
MARKET_DYNAMICS_INPUTS = _load("strict_r3_o3v2_market_dynamics_inputs_test", "audit_strict_r3_o3v2_market_dynamics_inputs.py")
MARKET_CONTEXT = _load("strict_r3_o3v2_market_context_test", "run_strict_r3_o3v2_market_context_funnel.py")
MARKET_EXTENDED = _load("strict_r3_o3v2_market_extended_labels_test", "materialize_strict_r3_o3v2_market_dynamics_extended_labels.py")
DERIVATIVE_LABELS = _load("strict_r3_o3v2_derivatives_labels_test", "materialize_strict_r3_o3v2_derivatives_positioning_labels.py")
HISTORY_PARENT_ADAPTER = _load("strict_r3_o3v2_history_parent_adapter_test", "materialize_strict_r3_o3v2_history_parent_adapter.py")
O3_AUDIT = _load("strict_r3_o3v2_audit_test", "audit_strict_r3_o3v2.py")
MC1_PORTFOLIO = _load("strict_r3_o3v2_mc1_portfolio_test", "run_strict_r3_o3v2_mc1_portfolio.py")
G3_FORWARD = _load("strict_r3_o3v2_g3_forward_test", "score_strict_r3_o3v2_g3_forward.py")


def test_neighbourhood_is_small_fixed_and_deterministic() -> None:
    first = SEMANTICS._fixed_neighbourhood(RichPolicyParams())
    second = SEMANTICS._fixed_neighbourhood(RichPolicyParams())
    assert [name for name, _ in first] == [name for name, _ in second]
    assert len(first) == 9
    assert first[0][0] == "canonical"


def test_invalid_sidecar_row_has_no_economic_semantic() -> None:
    frame = pd.DataFrame({
        "candidate_id": ["invalid"],
        "__decision_ts__": pd.to_datetime(["2026-01-01T00:00:00Z"]),
        "__symbol__": ["TEST/USD:USD"],
        "side_name": ["long"],
    })
    sidecar = SEMANTICS._empty_output(frame)
    assert not bool(sidecar.loc[0, "semantic_path_valid"])
    assert pd.isna(sidecar.loc[0, "semantic_axis_a_sequence"])
    assert pd.isna(sidecar.loc[0, "semantic_archetype"])


def test_auxiliary_path_labels_keep_invalid_rows_out_of_supervision() -> None:
    frame = pd.DataFrame({
        "candidate_id": ["invalid"],
        "__ts__": pd.to_datetime(["2026-01-01T00:00:00Z"]),
        "__decision_ts__": pd.to_datetime(["2026-01-01T01:00:00Z"]),
        "__symbol__": ["TEST/USD:USD"],
        "side_name": ["long"],
    })
    sidecar = AUXILIARY_PATH._empty_output(frame)
    assert not bool(sidecar.loc[0, "aux_path_valid"])
    assert sidecar.loc[0, "aux_label_available_ts"].tzinfo is None or str(sidecar["aux_label_available_ts"].dtype) == "datetime64[ns, UTC]"
    target_columns = [
        column for column in AUXILIARY_PATH.TARGET_COLUMNS
        if column not in {"aux_path_valid", "aux_path_complete", "aux_label_available_ts"}
    ]
    assert sidecar.loc[0, target_columns].isna().all()


def test_auxiliary_path_threshold_times_are_censored_at_h12() -> None:
    assert AUXILIARY_PATH._time_censored(np.array([0, 1, 4, 48], dtype=np.int16)).tolist() == [12.0, .25, 1.0, 12.0]
    assert "aux_mae_before_100bps_atr" in AUXILIARY_PATH.TARGET_COLUMNS


def test_history_parent_adapter_rejects_outcome_schema() -> None:
    columns = [*HISTORY_PARENT_ADAPTER.IDENTITY, *HISTORY_PARENT_ADAPTER.SOURCE_FIELDS, "policy_net_bps"]
    try:
        HISTORY_PARENT_ADAPTER._assert_target_free_schema(columns)
    except AssertionError as error:
        assert "policy" in str(error)
    else:
        raise AssertionError("outcome-bearing history schema was accepted")


def test_history_parent_adapter_route_is_exact_timestamp_top30() -> None:
    stamp = pd.Timestamp("2026-01-01T00:00:00Z")
    frame = pd.DataFrame({
        "candidate_id": [f"c{i:02d}" for i in range(10)],
        "__decision_ts__": [stamp] * 10,
        "enhanced_base_bps": list(range(10, 0, -1)),
    })
    assert int(HISTORY_PARENT_ADAPTER._route(frame).sum()) == 3


def test_market_dynamics_blocks_are_bounded_and_distinct() -> None:
    assert MARKET_DYNAMICS_INPUTS.MAX_PER_FAMILY == 10
    assert MARKET_DYNAMICS_INPUTS.MIN_COVERAGE == .90
    assert len(MARKET_DYNAMICS_INPUTS.FAMILY_CANDIDATES) == 11
    for family, fields in MARKET_DYNAMICS_INPUTS.FAMILY_CANDIDATES.items():
        assert 5 <= len(fields) <= MARKET_DYNAMICS_INPUTS.MAX_PER_FAMILY, family
        assert len(fields) == len(set(fields)), family
    assert "aux_time_to_trailing_activation_h" in AUXILIARY_PATH.TARGET_COLUMNS


def test_market_context_is_timestamp_level_and_never_asset_tie_breaks() -> None:
    stamp = pd.Timestamp("2026-01-01T00:00:00Z")
    frame = pd.DataFrame({
        "__decision_ts__": [stamp] * 10,
        "candidate_id": [f"a{i:02d}" for i in range(10)],
        "prequential_base_score": list(range(10, 0, -1)),
    })
    assert int(MARKET_CONTEXT._route_top30(frame).sum()) == 3
    for group, block in MARKET_CONTEXT.BLOCK.items():
        assert block in MARKET_DYNAMICS_INPUTS.FAMILY_CANDIDATES, group
        assert 5 <= len(MARKET_DYNAMICS_INPUTS.FAMILY_CANDIDATES[block]) <= 10
    assert "market_anchor_reversion_fraction_12h" in {
        target.column for target in MARKET_CONTEXT.GROUPS["stretch"]
    }


def test_market_context_receipt_is_label_and_outcome_free(tmp_path: Path) -> None:
    held = pd.DataFrame({"__decision_ts__": pd.to_datetime(["2026-01-01T00:00:00Z"])})
    root = tmp_path / "context"
    root.mkdir()
    (root / "audit_parts").mkdir()
    target = MARKET_CONTEXT.GROUPS["trend"][0]
    fold = MARKET_CONTEXT.FOLDS[0]
    MARKET_CONTEXT._write_score_receipt(
        root, fold=fold, target=target, held=held,
        raw=np.array([.2]), expected=np.array([42.0]), audit={"status": "ok"},
    )
    columns = pd.read_parquet(root / "target_free_scores" / target.name / f"fold={fold.name}.parquet").columns
    assert not any(column.startswith("market_") or "policy_net" in column for column in columns)


def test_market_extended_labels_censor_missing_events_at_h12() -> None:
    values = MARKET_EXTENDED._first_hit(np.array([[False, True, False], [False, False, False]]))
    assert np.allclose(values, [.5, 12.0])


def test_derivatives_positioning_labels_are_h12_resolved_and_proxy_is_explicit() -> None:
    # A compact common 50-asset panel is sufficient to exercise the declared
    # source-support gates.  The final H12 rows cannot be labelled because
    # their future window is unresolved.
    rows = DERIVATIVE_LABELS.HOURS + 1
    oi = np.full((rows, DERIVATIVE_LABELS.MIN_OI_ASSETS), 100.0)
    oi[-1] = 110.0
    funding = np.full_like(oi, 1e-5)
    labels = DERIVATIVE_LABELS._build(oi, funding)
    assert np.isfinite(labels.loc[0, "market_open_interest_change_12h"])
    assert not bool(labels.loc[rows - 1, "derivatives_label_valid"])
    assert "proxy" in MARKET_CONTEXT.GROUPS["leverage"][2].name


def test_path_auxiliary_funnel_routes_exactly_top30_and_keeps_scores_target_free() -> None:
    stamp = pd.Timestamp("2026-01-01T00:00:00Z")
    frame = pd.DataFrame({
        "__decision_ts__": [stamp] * 10,
        "candidate_id": [f"a{i:02d}" for i in range(10)],
        "base": list(range(10, 0, -1)),
    })
    routed = PATH_AUXILIARY_FUNNEL._exact_timestamp_top_fraction(frame, "base", .30)
    assert int(routed.sum()) == 3
    assert PATH_AUXILIARY_FUNNEL._score_leak_columns([
        "candidate_id", "predicted_policy_net_bps", "policy_net_bps", "aux_mfe_bps_1h",
    ]) == ["policy_net_bps", "aux_mfe_bps_1h"]
    assert {spec.family for spec in PATH_AUXILIARY_FUNNEL.TARGETS} == {"path_order", "magnitude", "timing"}


def test_economic_residual_keeps_declared_seven_classes() -> None:
    timestamps = pd.to_datetime([
        "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z",
        "2026-01-01T01:00:00Z", "2026-01-01T01:00:00Z",
    ])
    train = pd.DataFrame({
        "__decision_ts__": timestamps,
        "base_rank_ts": [.1, .9, .2, .8],
        "semantic_policy_net_bps": [-400., 400., -80., 180.],
        "semantic_axis_f_exit4": ["stop", "trailing", "timeout", "smooth_protection"],
        "semantic_axis_f_exit5": ["stop", "large_trailing", "timeout", "smooth_protection"],
    })
    value, grade, objective, _ = FUNNEL._anchor_and_targets(train, "T1_economic_residual_lambdarank")
    assert objective == "ordinal_lambdarank"
    assert grade is not None
    assert value.shape == grade.shape == (4,)
    assert grade.min() >= 0 and grade.max() <= 6


def test_ordinal_targets_fit_ordered_states_not_raw_residuals() -> None:
    train = pd.DataFrame({
        "__decision_ts__": pd.to_datetime(["2026-01-01T00:00:00Z"] * 7),
        "base_rank_ts": [.1, .2, .3, .4, .5, .6, .7],
        "semantic_policy_net_bps": [-500., -180., -60., 0., 60., 160., 500.],
        "semantic_axis_f_exit4": ["stop"] * 7,
        "semantic_axis_f_exit5": ["stop"] * 7,
    })
    value, grade, objective, mode = FUNNEL._anchor_and_targets(train, "T2_economic_residual_ordinal")
    assert objective == "l2_regression" and mode == "economic_residual_ordinal"
    assert grade is not None and np.array_equal(value, grade.astype(np.float32))
    assert grade.min() >= 0 and grade.max() <= 6

    value, grade, objective, mode = FUNNEL._anchor_and_targets(train, "T6_rank_error_ordinal")
    assert objective == "l2_regression" and mode == "rank_error_ordinal"
    assert grade is not None and np.array_equal(value, grade.astype(np.float32))
    assert grade.min() >= 0 and grade.max() <= 4


def test_target_free_projection_rejects_outcome_columns() -> None:
    score = pd.DataFrame({
        "candidate_id": ["a", "b"],
        "__decision_ts__": pd.to_datetime(["2026-01-01T00:00:00Z"] * 2),
        "side_name": ["long", "long"],
        "enhanced_base_routed": [True, True],
        "enhanced_base_bps": [10., 20.],
        "base_rank_ts": [.4, .8],
        "conditional_consensus_rank": [.3, .9],
        "ordinary_shadow_consensus_rank": [.3, .9],
        "head_agreement_std": [.1, .2],
        "head__one__rank": [.2, .8],
        "semantic_axis_a_sequence": ["never", "never"],
        "policy_net_bps": [100., -100.],
    })
    projected = FUNNEL._score_columns(score)
    assert not (set(projected.columns) & FUNNEL.PROHIBITED_SCORE_COLUMNS)
    assert np.isfinite(projected["o3v2_rank_75_25"]).all()


def test_support_weights_are_bounded_and_training_only() -> None:
    train = pd.DataFrame({
        "candidate_id": ["a", "b", "c", "d"],
        "__decision_ts__": pd.to_datetime([
            "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z",
            "2026-01-01T01:00:00Z", "2026-01-01T01:00:00Z",
        ]),
        "semantic_archetype": ["timeout", "clean_fast_persistent_trailing", "timeout", "adverse_first_stop"],
        "semantic_tbm_event": ["vertical", "upper_first", "ambiguous", "lower_first"],
        "semantic_axis_k_policy_robustness": ["policy_sensitive", "mixed_robustness", "policy_sensitive", "mixed_robustness"],
        "semantic_policy_net_bps": [-80.0, 180.0, -20.0, 40.0],
        "base_rank_ts": [.8, .2, .9, .1],
    })
    for arm in SUPPORT.SUPPORT_ARMS:
        weights = SUPPORT._weights(train, arm)
        assert weights.shape == (4,)
        assert np.isfinite(weights).all()
        assert (weights >= .25).all() and (weights <= 4.0).all()


def test_support_selector_compares_each_weight_to_matching_uniform_target(tmp_path: Path) -> None:
    """A target's unweighted receipt must be eligible as S0.

    Comparing a weighting arm only with broad B0 could retain a configuration
    that is worse than the same target trained uniformly.  The selector must
    preserve the target-funnel S0 control without rerunning it.
    """
    rows = []
    for tail in (.01, .02, .05):
        rows.extend((
            {"arm": "T0_current_o3_control", "score": "o3v2_rank_75_25", "month": "2025-11", "tail": tail, "net_ev_bps_per_trade": 0.0, "policy_rank_ic": 0.0},
            {"arm": "T1_economic_residual_lambdarank", "score": "o3v2_rank_75_25", "month": "2025-11", "tail": tail, "net_ev_bps_per_trade": 100.0, "policy_rank_ic": .1},
        ))
    target = tmp_path / "target.parquet"
    pd.DataFrame(rows).to_parquet(target, index=False)
    weighted = []
    for tail in (.01, .02, .05):
        weighted.append({
            "arm": "T1_economic_residual_lambdarank__S1_archetype_balance",
            "score": "o3v2_rank_75_25", "month": "2025-11", "tail": tail,
            "net_ev_bps_per_trade": 50.0, "policy_rank_ic": .05,
        })
    support = tmp_path / "support.parquet"
    pd.DataFrame(weighted).to_parquet(support, index=False)
    out = tmp_path / "selection"
    SUPPORT_SELECTION.run(
        target_metrics=target, support_metrics=(support,), out=out, months=("2025-11",),
    )
    selected = pd.read_parquet(out / "support_development_selection.parquet")
    winner = selected.loc[selected["target_arm"].eq("T1_economic_residual_lambdarank")].iloc[0]
    assert winner["support_arm"] == "S0_uniform"


def test_query_preselection_core_contract_has_no_later_g3_additions() -> None:
    """Query geometry must be selected before any outcome-selected G3 block."""
    fields = QUERY_SCREEN._load_contract(
        ROOT / "does_not_need_to_exist.json",
        preselection_core_score_only=True,
    )
    assert fields == QUERY_SCREEN.PRESELECTION_CORE_SCORE_FIELDS
    assert len(fields) == 9
    assert not any(field.startswith(("f2_", "f3_", "f4_", "f5_", "f6_")) for field in fields)


def test_g3_requires_the_sealed_query_geometry(tmp_path: Path) -> None:
    contract = tmp_path / "query.json"
    contract.write_text('{"schema":"strict_r3_o3v2_query_selection_v1","selected_query_mode":"cycle_4h_side","development_months":["2026-02"]}')
    accepted = GREEDY._load_query_contract(contract, "cycle_4h_side")
    assert accepted is not None
    try:
        GREEDY._load_query_contract(contract, "exact_timestamp_side")
    except AssertionError as error:
        assert "differs" in str(error)
    else:
        raise AssertionError("G3 accepted a query mode different from the sealed selector")


def test_physical_slot_selection_must_follow_query_selection() -> None:
    PHYSICAL_SLOTS._require_later(("2026-03", "2026-04"), ("2026-02",), name="slot development")
    try:
        PHYSICAL_SLOTS._require_later(("2026-02",), ("2026-02",), name="slot development")
    except AssertionError as error:
        assert "strictly after" in str(error)
    else:
        raise AssertionError("physical-slot selection accepted an overlapping query-development month")


def test_query_selector_uses_only_declared_development_metrics(tmp_path: Path) -> None:
    metrics = pd.DataFrame([
        {"query_mode": "exact_timestamp_side", "month": "2026-02", "utility": 100.0, "rank_ic": .1, "top1": 100.0, "top2": 100.0, "top5": 100.0},
        {"query_mode": "cycle_4h_side", "month": "2026-02", "utility": 120.0, "rank_ic": .1, "top1": 90.0, "top2": 90.0, "top5": 90.0},
        # A later month must not influence the February selection.
        {"query_mode": "exact_timestamp_side", "month": "2026-05", "utility": 1000.0, "rank_ic": .2, "top1": 1000.0, "top2": 1000.0, "top5": 1000.0},
        {"query_mode": "cycle_4h_side", "month": "2026-05", "utility": -1000.0, "rank_ic": -.2, "top1": -1000.0, "top2": -1000.0, "top5": -1000.0},
    ])
    source = tmp_path / "metrics.parquet"
    metrics.to_parquet(source, index=False)
    out = tmp_path / "selection"
    QUERY_SELECTION.run(metrics_path=source, out=out, development_months=("2026-02",))
    import json
    contract = json.loads((out / "selected_query_contract.json").read_text())
    assert contract["selected_query_mode"] == "cycle_4h_side"


def test_v2_support_weights_keep_uniform_control_and_use_no_score_field() -> None:
    train = pd.DataFrame({
        "__decision_ts__": pd.to_datetime([
            "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z",
            "2026-01-01T01:00:00Z", "2026-01-01T01:00:00Z",
        ]),
        "semantic_archetype": ["timeout", "clean", "timeout", "adverse"],
        "semantic_tbm_event": ["vertical", "upper_first", "ambiguous", "lower_first"],
        "semantic_axis_a_sequence": ["clear", "clear", "same_bar", "clear"],
        "semantic_axis_c_persistence": ["persistent", "persistent", "mixed", "persistent"],
        "semantic_axis_f_exit5": ["timeout", "large_trailing", "timeout", "stop"],
        "semantic_policy_net_bps": [-150.0, 150.0, -20.0, 80.0],
        "base_rank_ts": [.8, .2, .9, .1],
    })
    uniform = SUPPORT_V2._weights(train, "S0_uniform")
    assert np.allclose(uniform, 1.0)
    for arm in SUPPORT_V2.SUPPORT_ARMS:
        weights = SUPPORT_V2._weights(train, arm)
        assert weights.shape == (4,)
        assert np.isfinite(weights).all()
        assert (weights >= .25).all() and (weights <= 4.0).all()


def test_support_screen_retains_pairwise_and_ordinal_selected_target_concepts() -> None:
    assert SUPPORT.TARGET_ARMS == (
        "T3_pair_residual_lambdarank",
        "T6_rank_error_ordinal",
    )


def test_exact_timestamp_baseband_query_is_causal_and_fixed() -> None:
    frame = pd.DataFrame({
        "__decision_ts__": pd.to_datetime(["2026-01-01T00:00:00Z"] * 5),
        "side_name": ["long"] * 5,
        "base_rank_ts": [.75, .85, .92, .96, .99],
    })
    query = FUNNEL.parent._query(frame, "exact_timestamp_baseband_side")
    assert query.str.endswith("|long").all()
    assert {"70_80", "80_90", "90_95", "95_98", "98_100"} == {
        token.split("|")[-2] for token in query
    }


def test_meta_route_is_exact_timestamp_local_top30_with_stable_ties() -> None:
    stamp = pd.Timestamp("2026-01-01T00:00:00Z")
    frame = pd.DataFrame({
        "__decision_ts__": [stamp] * 10 + [stamp + pd.Timedelta(hours=1)] * 7,
        "candidate_id": [f"a{i:02d}" for i in range(10)] + [f"b{i:02d}" for i in range(7)],
        # The first cross-section deliberately has a tied cutoff.  Candidate
        # IDs make the selected three reproducible rather than retaining every
        # tied row through a percentile threshold.
        "score": [10., 9., 8., 8., 7., 6., 5., 4., 3., 2.] + list(range(7, 0, -1)),
    })
    routed = FUNNEL.parent._exact_timestamp_top_fraction(frame, "score", .30)
    assert int(routed.iloc[:10].sum()) == 3
    assert int(routed.iloc[10:].sum()) == 3
    assert routed.iloc[:10].to_list() == [True, True, True, False, False, False, False, False, False, False]


def test_mc1_recomputes_canonical_base_route_before_consuming_o3_scores(tmp_path: Path) -> None:
    """MC1 must not rely on an upstream persisted routed flag."""
    stamp = pd.Timestamp("2026-05-01T00:00:00Z")
    month = pd.Timestamp("2026-05-01T00:00:00Z")
    candidate_ids = [f"c{index:02d}" for index in range(10)]
    p2_root = tmp_path / "p2"
    p2_path = p2_root / "target_free_scores" / "current" / "month=2026-05.parquet"
    p2_path.parent.mkdir(parents=True)
    parent_frame = pd.DataFrame({
        "candidate_id": candidate_ids,
        "__decision_ts__": [stamp] * len(candidate_ids),
        "side_name": ["long"] * len(candidate_ids),
        # Deliberately false: the consumer must derive the route itself.
        "enhanced_base_routed": [False] * len(candidate_ids),
        "enhanced_base_bps": list(range(10, 0, -1)),
        "base_anchor_bps": np.linspace(10., 90., len(candidate_ids)),
        **{field: np.linspace(.1, .9, len(candidate_ids)) for field in MC1_PORTFOLIO.parent.MC1_FEATURES},
    })
    parent_frame.to_parquet(p2_path, index=False)

    arm_root = tmp_path / "arm"
    arm_path = arm_root / "target_free_scores" / "arm" / "month=2026-05.parquet"
    arm_path.parent.mkdir(parents=True)
    # The fourth row is deliberately outside the canonical top-three route.
    arm_ids = [*candidate_ids[:3], candidate_ids[-1]]
    arm_frame = pd.DataFrame({
        "candidate_id": arm_ids,
        "base_rank_ts": [.9, .8, .7, .1],
        **{f"head__{head}__rank": [.9, .8, .7, .1] for head in MC1_PORTFOLIO.HEADS},
    })
    arm_frame.to_parquet(arm_path, index=False)

    result = MC1_PORTFOLIO._load_family(
        p2_root, {"arm": (arm_root,)}, "current", (month,), "additive", {"arm": "cap100_ordinary"},
    )
    assert result["candidate_id"].to_list() == candidate_ids[:3]
    assert result["enhanced_base_routed"].all()
    replacement = MC1_PORTFOLIO._load_family(
        p2_root, {"arm": (arm_root,)}, "current", (month,), "replace_correction", {"arm": "cap100_ordinary"},
    )
    assert replacement["candidate_id"].to_list() == candidate_ids[:3]
    assert "o3__arm__delta_parent_consensus" not in replacement
    assert np.allclose(replacement["final_score"], [.9, .8, .7])


def test_mc1_replacement_mode_excludes_incumbent_correction_inputs() -> None:
    fields = MC1_PORTFOLIO._feature_names(
        ("T1", "T6"), "aggregate", "replace_correction", {"T1": "cap100_ordinary", "T6": "cap80_ordinary"},
    )
    assert {"base_rank42", "base_anchor_bps", "correctness_rank"}.issubset(fields)
    assert not {
        "final_score", "upstream", "conditional_consensus_rank", "ordinary_shadow_consensus_rank",
    }.intersection(fields)
    assert not any(field.endswith("delta_parent_consensus") for field in fields)
    assert {"o3__t1__consensus_rank", "o3__t6__consensus_rank"}.issubset(fields)


def test_mc1_single_physical_slot_contract_removes_five_slot_ensemble_inputs() -> None:
    fields = MC1_PORTFOLIO._feature_names(
        ("T1", "T6"), "full", "replace_correction", {"T1": "cap120_equal_month", "T6": "cap80_ordinary"},
    )
    assert "o3__t1__cap120_equal_month_rank" in fields
    assert "o3__t6__cap80_ordinary_rank" in fields
    assert not any(field.endswith("head_rank_std") for field in fields)
    assert not any(field.endswith("cap100_ordinary_rank") and not field.startswith("o3__t1__") for field in fields)
    try:
        MC1_PORTFOLIO._load_physical_slots(None, ("T1",))
    except ValueError as error:
        assert "physical-slot" in str(error)
    else:
        raise AssertionError("MC1 accepted an O3 challenger without a physical-slot contract")


def test_target_funnel_physical_slot_contract_must_match_query_mode(tmp_path: Path) -> None:
    contract = tmp_path / "slots.json"
    contract.write_text(
        '{"schema":"strict_r3_o3v2_physical_slot_selection_v1",'
        '"query_mode":"cycle_4h_side",'
        '"selected_slots":{"T1_economic_residual_lambdarank":"cap100_ordinary"}}'
    )
    accepted = FUNNEL._load_physical_slot_contract(
        contract, ("T1_economic_residual_lambdarank",), query_mode="cycle_4h_side",
    )
    assert accepted == {"T1_economic_residual_lambdarank": "cap100_ordinary"}
    try:
        FUNNEL._load_physical_slot_contract(
            contract, ("T1_economic_residual_lambdarank",), query_mode="exact_timestamp_side",
        )
    except AssertionError as error:
        assert "differs" in str(error)
    else:
        raise AssertionError("target funnel accepted a physical-slot query mismatch")


def test_g3_uses_the_sealed_single_physical_slot(tmp_path: Path) -> None:
    contract = tmp_path / "slots.json"
    contract.write_text(
        '{"schema":"strict_r3_o3v2_physical_slot_selection_v1",'
        '"query_mode":"cycle_4h_side",'
        '"selected_slots":{"T6_rank_error_ordinal":"cap80_ordinary"}}'
    )
    slot, cap, weight = GREEDY._load_physical_slot_contract(
        contract, "T6_rank_error_ordinal", query_mode="cycle_4h_side",
    )
    assert (slot, cap, weight) == ("cap80_ordinary", 80, "ordinary")
    try:
        GREEDY._load_physical_slot_contract(
            None, "T6_rank_error_ordinal", query_mode="cycle_4h_side",
        )
    except ValueError as error:
        assert "physical-slot" in str(error)
    else:
        raise AssertionError("post-selector G3 accepted no physical-slot contract")


def test_g3_forward_projection_is_target_free_and_single_slot() -> None:
    held = pd.DataFrame({
        "candidate_id": ["a", "b"],
        "__decision_ts__": pd.to_datetime(["2026-05-01T00:00:00Z"] * 2),
        "side_name": ["long", "long"],
        "f1_base_rank_ts": [.9, .8],
        "policy_net_bps": [100.0, -100.0],
    })
    projection = G3_FORWARD._project(
        held, slot="cap80_ordinary", raw=np.array([.2, .1]), rank=np.array([.8, .7]),
    )
    assert "policy_net_bps" not in projection
    assert set(projection.columns) == {
        "candidate_id", "__decision_ts__", "side_name", "base_rank_ts",
        "head__cap80_ordinary__raw", "head__cap80_ordinary__rank",
    }


def test_single_equal_month_head_has_no_synthetic_ordinary_ensemble() -> None:
    class Head:
        spec = type("Spec", (), {"name": "cap120_equal_month"})()

        def predict_rank(self, frame: pd.DataFrame):
            values = np.linspace(.2, .8, len(frame), dtype=np.float32)
            return values, values

    frame = pd.DataFrame({"candidate_id": ["a", "b"], "__decision_ts__": pd.to_datetime(["2026-01-01T00:00:00Z"] * 2)})
    scored = FUNNEL.parent._score_heads(frame, (Head(),))
    assert np.allclose(scored["conditional_consensus_rank"], scored["ordinary_shadow_consensus_rank"])
    assert np.allclose(scored["head__cap120_equal_month__rank"], scored["ordinary_shadow_consensus_rank"])


def test_g3_family_additions_are_bounded_and_cross_family_gate_is_incremental() -> None:
    assert GREEDY.MAX_ADDITIONS == 4
    current = pd.DataFrame({"month": ["2026-01", "2026-02", "2026-03"], "utility": [10.0, 11.0, 12.0]})
    candidate = pd.DataFrame({"month": ["2026-01", "2026-02", "2026-03"], "utility": [11.0, 10.0, 13.0]})
    accepted, improvement, positive, required = GREEDY._acceptance(
        candidate_score=12.0, current_score=11.0,
        candidate_metrics=candidate, baseline_metrics=current,
    )
    assert accepted
    assert improvement == 1.0
    assert positive == required == 2


def test_g3_rejects_a_partial_declared_six_month_training_window() -> None:
    history = pd.DataFrame({
        "candidate_id": ["a", "b", "c"],
        "__decision_ts__": pd.to_datetime([
            "2025-08-01T00:00:00Z", "2025-08-01T00:00:00Z", "2025-08-01T01:00:00Z",
        ]),
        "f1_enhanced_base_bps": [3.0, 2.0, 1.0],
    })
    policy = pd.DataFrame({
        "candidate_id": ["a", "b", "c"],
        "policy_path_valid": [True, True, True],
        "policy_label_available_ts": pd.to_datetime([
            "2025-08-02T00:00:00Z", "2025-08-02T00:00:00Z", "2025-08-02T00:00:00Z",
        ]),
        "policy_net_bps": [10.0, 20.0, 30.0],
    })
    try:
        GREEDY._prepare_folds(history, policy, (pd.Timestamp("2025-09-01T00:00:00Z"),))
    except AssertionError as error:
        assert "incomplete six-month G3 training history" in str(error)
    else:
        raise AssertionError("partial G3 training window was accepted")


def test_g3_six_month_fit_starts_before_the_reserve_not_the_held_month() -> None:
    # For a September fold the 28-day reserve begins on August 4.  A history
    # starting March 1 satisfies the old ``held_month - six months`` check,
    # but is short of the required six months before that reserve (Feb 4).
    history = pd.DataFrame({
        "candidate_id": ["a", "b"],
        "__decision_ts__": pd.to_datetime(["2025-03-01T00:00:00Z", "2025-03-01T01:00:00Z"]),
        "f1_enhanced_base_bps": [2.0, 1.0],
    })
    policy = pd.DataFrame({
        "candidate_id": ["a", "b"], "policy_path_valid": [True, True],
        "policy_label_available_ts": pd.to_datetime(["2025-03-02T00:00:00Z", "2025-03-02T00:00:00Z"]),
        "policy_net_bps": [10.0, 20.0],
    })
    try:
        GREEDY._prepare_folds(history, policy, (pd.Timestamp("2025-09-01T00:00:00Z"),))
    except AssertionError as error:
        assert "incomplete six-month G3 training history" in str(error)
    else:
        raise AssertionError("reserve-shortened G3 training window was accepted")


def test_target_funnel_uses_six_full_months_before_its_reserve() -> None:
    month = pd.Timestamp("2026-02-01T00:00:00Z")
    assert FUNNEL._strict_train_start(month) == pd.Timestamp("2025-07-04T00:00:00Z")


def test_g3_audit_rejects_manifest_with_an_underwarmed_window(tmp_path: Path) -> None:
    root = tmp_path / "g3"
    root.mkdir()
    (root / "target_free_scores").mkdir()
    (root / "g3_feature_contracts.json").write_text('{"schema":"strict_r3_o3v2_greedy_features_v1","contracts":{}}')
    pd.DataFrame({"tag": ["core"], "month": ["2025-09"]}).to_parquet(root / "g3_strict_oof_trace.parquet", index=False)
    (root / "run_manifest.json").write_text(
        '{"schema":"strict_r3_o3v2_greedy_features_v1",'
        '"history_start":"2025-08-01T00:00:00Z",'
        '"development_months":["2025-09"],'
        '"training":{"full_window_required":true,"calendar_months":6,"reserve_days":28}}'
    )
    evidence: dict[str, object] = {}
    failures: list[str] = []
    O3_AUDIT._audit_g3(root, evidence, failures)
    assert any("history begins after a required training window" in failure for failure in failures)


def test_v3_policy_state_supports_are_bounded_and_factorised() -> None:
    train = pd.DataFrame({
        "__decision_ts__": pd.to_datetime(["2026-01-01T00:00:00Z"] * 4),
        "semantic_archetype": ["timeout", "clean", "timeout", "adverse"],
        "semantic_tbm_event": ["vertical", "upper_first", "vertical", "lower_first"],
        "semantic_axis_a_sequence": ["clear"] * 4,
        "semantic_axis_c_persistence": ["persistent"] * 4,
        "semantic_axis_f_exit4": ["timeout", "trailing", "timeout", "stop"],
        "semantic_axis_f_exit5": ["timeout", "large_trailing", "regular_trailing", "stop"],
        "semantic_policy_net_bps": [-50., 150., 50., -120.],
        "base_rank_ts": [.8, .2, .9, .1],
    })
    for arm in SUPPORT_V3.SUPPORT_ARMS:
        weights = SUPPORT_V3._weights(train, arm)
        assert weights.shape == (4,)
        assert np.isfinite(weights).all()
        assert (weights >= .25).all() and (weights <= 4.0).all()


def test_support_lambdarank_reuses_seven_grade_parent_gains() -> None:
    spec = SUPPORT.parent.ConsensusHeadSpec(
        name="test", cap=100, weight_mode="ordinary", query="exact_timestamp_side",
        fields=("f1_base_bps",), target_edges_bps=(-100., -30., 30., 90.),
        params={"objective": "lambdarank", "label_gain": [0, .25, 1., 3., 7.]},
    )
    fitted = SUPPORT._label_specs_for_target((spec,), np.asarray([0, 6], dtype=np.int32))
    assert fitted[0].params["label_gain"] == [0, 1, 2, 4, 7, 12, 20]


def test_recent_error_telemetry_is_availability_causal() -> None:
    stamps = pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="h")
    frame = pd.DataFrame({
        "__decision_ts__": stamps.repeat(2),
        "base_rank_ts": [.1, .9] * 4,
        "base_anchor_bps": [0.0] * 8,
        "policy_path_valid": [True] * 8,
        "policy_net_bps": [-30.0, 30.0] * 4,
        "policy_label_available_ts": (stamps + pd.Timedelta(hours=1)).repeat(2),
    })
    first = FEATURE_SCREEN._recent_error_features(frame)
    changed = frame.copy()
    changed.loc[changed.index[-2:], "policy_net_bps"] = [9999.0, -9999.0]
    second = FEATURE_SCREEN._recent_error_features(changed)
    # The final outcome becomes available after the earlier decisions, so it
    # cannot alter their causal historical-error features.
    pd.testing.assert_frame_equal(first.iloc[:-2], second.iloc[:-2])


def test_feature_screen_allows_parent_only_provenance_without_legacy_o3() -> None:
    stamp = pd.Timestamp("2026-01-01T00:00:00Z")
    identity = {
        "candidate_id": ["a", "b"],
        "__decision_ts__": [stamp, stamp],
        "side_name": ["long", "long"],
    }
    base = pd.DataFrame({
        **identity,
        "base_bps": [10.0, 20.0], "efficiency_bps": [11.0, 21.0],
        "timing_bps": [12.0, 22.0], "enhanced_base_bps": [13.0, 23.0],
        "base_rank_ts": [.2, .8], "e_minus_t": [-1.0, -1.0],
        "e_minus_b0": [1.0, 1.0], "t_minus_b0": [2.0, 2.0],
        "base_component_std": [1.0, 1.0], "enhanced_base_routed": [True, True],
        "mkt_rv_4h": [1.0, 2.0],
    })
    parent = pd.DataFrame({
        **identity,
        "base_rank42": [.3, .9], "base_anchor_bps": [5.0, 25.0],
        "conditional_consensus_rank": [.4, .8], "ordinary_shadow_consensus_rank": [.5, .7],
        "upstream": [.4, .8], "correctness_rank": [.6, .9],
        "head_agreement_std": [.1, .2], "final_score": [.4, .9],
    })
    output = FEATURE_SCREEN._add_families(base, parent, parent)
    assert len(output) == 2
    assert "f5_current_final_score" in output
    assert "f5_bcf_final_score" in output
    assert not any(field.startswith("f5_o3_") for field in output.columns)


def test_specialist_target_free_panel_rejects_policy_outcomes(tmp_path: Path) -> None:
    source = tmp_path / "panel.parquet"
    pd.DataFrame({
        "candidate_id": ["a"], "__decision_ts__": pd.to_datetime(["2026-01-01T00:00:00Z"]),
        "side_name": ["long"], "f1_base_rank_ts": [.8], "f1_base_bps": [10.0],
        "policy_net_bps": [100.0],
    }).to_parquet(source, index=False)
    try:
        SPECIALISTS._read_history(source, ["f1_base_bps"])
    except AssertionError as error:
        assert "outcome" in str(error)
    else:
        raise AssertionError("target-free specialist reader accepted an outcome field")


def test_t2_adapter_is_exactly_two_head_consensus() -> None:
    f4 = np.asarray([.2, .8])
    f5 = np.asarray([.4, .6])
    consensus = .5 * (f4 + f5)
    expected = np.asarray([.3, .7])
    assert np.allclose(consensus, expected)
    assert np.allclose(np.abs(f4 - f5) / np.sqrt(2.0), [np.sqrt(.02), np.sqrt(.02)])
    assert T2_ADAPTER.ARM_BY_MODE["F4F5"] == "T2_F4_F5_selected"
    assert T2_ADAPTER.ARM_BY_MODE["H3"] == "T2_H3_F4_F5_selected"
    assert T2_ADAPTER.ARM_BY_MODE["H2EA"] == "T2_H2_EQUAL_ARCHETYPE_selected"


def test_h3_specialist_keeps_five_frozen_semantic_roles() -> None:
    selection = {family: [f"{family}_a"] for family in ("f1", "f2", "f3", "f4", "f5", "f6")}
    selection["f4"] = ["f4_a", "f4_b"]
    selection["f5"] = ["f5_a"]
    heads = SPECIALISTS._head_definitions(selection, "H3_hybrid_f4_f5", "SB3_error_semantic")
    assert [head[0] for head in heads] == [
        "h3_context_f5", "h3_context_f6", "h3_base_query", "h3_recent_error", "h3_state_support",
    ]
    core = {
        "f1_enhanced_base_bps", "f1_base_rank_ts", "f1_base_bps", "f1_efficiency_bps", "f1_timing_bps",
        "f1_e_minus_t", "f1_e_minus_b0", "f1_t_minus_b0", "f1_base_component_std",
    }
    assert core.issubset(set(heads[0][1]))
    assert "f5_a" in heads[0][1]
    assert "f6_a" in heads[1][1]
    assert {"f1_a", "f2_a"}.issubset(set(heads[2][1]))
    assert "f3_a" in heads[3][1]
    assert {"f4_a", "f4_b"}.issubset(set(heads[4][1]))


def test_h2_population_preserves_its_predeclared_weighting_contract() -> None:
    selection = {family: [f"{family}_a"] for family in ("f1", "f2", "f3", "f4", "f5", "f6")}
    heads = SPECIALISTS._head_definitions(selection, "H2_population", "SB3_error_semantic")
    assert [head[0] for head in heads] == ["h2_ordinary", "h2_equal_month", "h2_equal_archetype", "h2_hard_base_error", "h2_policy_state"]
    assert heads[1][3] is True
    assert SPECIALISTS.TRAIN_MONTHS == 6


def test_target_selection_is_development_only_and_one_per_concept() -> None:
    rows = []
    for arm, concept, delta in (
        ("T1_economic_residual_lambdarank", "economic_residual", 20.0),
        ("T3_pair_residual_lambdarank", "economic_residual", 30.0),
        ("T6_rank_error_ordinal", "rank_error", 25.0),
    ):
        for month in ("2025-10", "2025-11", "2025-12"):
            for tail in (.01, .02, .05):
                rows.append({
                    "arm": arm, "concept": concept, "month": month, "tail": tail,
                    "delta_net_ev_bps_per_trade": delta, "policy_rank_ic": .1,
                    "control_rank_ic": .05,
                })
    table = TARGET_SELECTION._development_table(pd.DataFrame(rows), ("2025-10", "2025-11", "2025-12"))
    winners = (
        table.loc[table["eligible"]].sort_values("selection_score_bps", ascending=False)
        .groupby("concept", sort=False).first().reset_index()
    )
    assert winners.loc[winners["concept"].eq("economic_residual"), "arm"].item() == "T3_pair_residual_lambdarank"
    assert set(winners["concept"]) == {"economic_residual", "rank_error"}


def test_target_selection_uses_only_declared_primary_score() -> None:
    rows = []
    for score, delta in (("conditional_consensus_rank", -1000.0), ("o3v2_rank_75_25", 20.0)):
        for arm, ev in (("T0_current_o3_control", 100.0), ("T3_pair_residual_lambdarank", 100.0 + delta)):
            for tail in (.01, .02, .05):
                rows.append({"arm": arm, "score": score, "month": "2025-10", "tail": tail, "net_ev_bps_per_trade": ev, "policy_rank_ic": .1})
    result = TARGET_SELECTION._delta(pd.DataFrame(rows), ("2025-10",))
    assert set(result["score"]) == {"o3v2_rank_75_25"}
    assert np.allclose(result.loc[result["arm"].eq("T3_pair_residual_lambdarank"), "delta_net_ev_bps_per_trade"], 20.0)


def test_target_selection_candidate_set_is_explicitly_bounded() -> None:
    declared = TARGET_SELECTION._arms(
        "T1_economic_residual_lambdarank,T2_economic_residual_ordinal,"
        "T4_hard_inversion_lambdarank,T6_rank_error_ordinal,"
        "T8_exit5_lambdarank,T9_exit5_ordinal"
    )
    assert declared == TARGET_SELECTION.DEFAULT_CANDIDATE_ARMS
    try:
        TARGET_SELECTION._arms("T1_economic_residual_lambdarank,unknown")
    except ValueError as error:
        assert "unknown candidate arms" in str(error)
    else:
        raise AssertionError("target selection accepted an undeclared arm")
