from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from extreme_price_movements.causal_leaf_health_prerequisites import (
    DEFAULT_HEALTH_CONTEXT_COLUMNS,
    FAMILY_SELECTION_ROOT_STATUS,
    PredecessorFamilySelectionConfig,
    STRICT_CONTEXT_END_EXCLUSIVE_UTC,
    STRICT_CONTEXT_START_UTC,
    CausalLeafHealthPrerequisiteError,
    load_frozen_family_selection,
    load_strict_fold_context,
    materialize_strict_fold_causal_context,
    materialize_strict_predecessor_family_selections,
    _selection_audit,
    _streaming_selection_audit,
    validate_selection_application,
)
from extreme_price_movements.causal_leaf_health_artifacts import (
    collect_completed_strict_oof_family_inputs,
    spool_completed_strict_oof_family_inputs,
)
from extreme_price_movements.strict_event_store import build_strict_event_store


def _empty_shard() -> pd.DataFrame:
    return pd.DataFrame({
        "candidate_id": pd.Series(dtype="string"),
        "decision_ts": pd.Series(dtype="datetime64[ns, UTC]"),
        "label_available_ts": pd.Series(dtype="datetime64[ns, UTC]"),
        "side_name": pd.Series(dtype="string"), "fold_id": pd.Series(dtype="string"),
        "feature_generation_ts": pd.Series(dtype="datetime64[ns, UTC]"),
        "feature_contract_sha256": pd.Series(dtype="string"), "base_expected_bps": pd.Series(dtype="float64"),
        "asset": pd.Series(dtype="string"), "r3_class": pd.Series(dtype="int8"),
    })


def _strict_root(tmp_path: Path) -> Path:
    root = tmp_path / "strict"
    root.mkdir()
    transport = "transport_a"
    (root / "strict_oof_reasoning_manifest.json").write_text(json.dumps({
        "status": "STRICT_OOF_BASE_REASONING_MATERIALIZED", "transports": [transport],
    }))
    timestamp = pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T03:00:00Z"], utc=True)
    inner = pd.DataFrame({
        "candidate_id": ["a", "b"], "decision_ts": timestamp,
        "label_available_ts": timestamp + pd.to_timedelta([1, 3], unit="h"),
        "side_name": ["long", "long"], "fold_id": ["fold-1", "fold-1"],
        "feature_generation_ts": timestamp, "feature_contract_sha256": ["f" * 64] * 2,
        "base_expected_bps": [5.0, 5.0], "asset": ["A", "B"], "r3_class": [2, 1],
    })
    outer = pd.DataFrame({
        "candidate_id": ["outer"], "decision_ts": [pd.Timestamp("2024-01-01T00:30:00Z")],
        "label_available_ts": [pd.Timestamp("2024-01-01T01:00:00Z")],
        "side_name": ["long"], "fold_id": ["fold-1"],
        "feature_generation_ts": [pd.Timestamp("2024-01-01T00:30:00Z")],
        "feature_contract_sha256": ["f" * 64], "base_expected_bps": [5.0], "asset": ["X"], "r3_class": [2],
    })
    for side, in_frame, out_frame in (("long", inner, outer), ("short", _empty_shard(), _empty_shard())):
        directory = root / "base_prediction_shards" / transport / side
        directory.mkdir(parents=True)
        in_frame.to_parquet(directory / "strict_oof_predictions.parquet", index=False)
        out_frame.to_parquet(directory / "outer_predictions.parquet", index=False)

    artifact = root / "strict_oof_base_reasoning" / transport / "folds" / "long" / "fold-1" / "p_clear"
    artifact.mkdir(parents=True)
    head_timestamp = pd.to_datetime([
        "2024-01-01T00:00:00Z", "2024-01-01T03:00:00Z", "2024-01-01T00:30:00Z",
    ], utc=True)
    assignment = pd.DataFrame({
        "candidate_id": ["a", "b", "outer"], "__ts__": head_timestamp, "side_name": ["long"] * 3,
        "head_name": ["p_clear"] * 3, "fold_id": ["fold-1"] * 3,
        "leaf_assignment__model_00_head_tree_000": np.array([10, 20, 30], dtype=np.uint64),
    })
    assignment.to_parquet(artifact / "leaf_assignments.parquet", index=False)
    pd.DataFrame({
        "head_name": ["p_clear"] * 3, "side_name": ["long"] * 3, "fold_id": ["fold-1"] * 3,
        "model_slot": [0] * 3, "head_tree_slot": [0] * 3, "leaf_token": np.array([10, 20, 30], dtype=np.uint64),
        "rule_signature": ["early_family", "late_family", "outer_family"], "tree_leaf_value": [1.0] * 3,
        "ensemble_tree_contribution": [1.0] * 3,
    }).to_parquet(artifact / "leaf_rule_catalog.parquet", index=False)
    pd.DataFrame({
        "candidate_id": ["a", "b", "outer"], "__ts__": head_timestamp, "side_name": ["long"] * 3,
        "head_name": ["p_clear"] * 3, "fold_id": ["fold-1"] * 3, "base_prediction": [.7, .6, .8],
    }).to_parquet(artifact / "base_reasoning_predictions.parquet", index=False)
    pd.DataFrame({
        "candidate_id": ["a", "b", "outer"], "__ts__": head_timestamp, "side_name": ["long"] * 3,
        "head_name": ["p_clear"] * 3, "fold_id": ["fold-1"] * 3,
        "label__r3_class": [2, 1, 2], "label__net_bps": [50.0, -50.0, 500.0],
        "label__label_available_ts": pd.to_datetime([
            "2024-01-01T01:00:00Z", "2024-01-01T06:00:00Z", "2024-01-01T01:00:00Z",
        ], utc=True),
    }).to_parquet(artifact / "base_reasoning_labels.parquet", index=False)
    (artifact / "base_reasoning_manifest.json").write_text(json.dumps({
        "status": "MATERIALIZED_STRICT_OOF", "head_name": "p_clear", "side_name": "long", "fold_id": "fold-1",
        "provenance": {"model_hashes": ["m"], "feature_contract_sha256": "f" * 64, "class_index": 2},
    }))
    return root


def test_predecessor_selection_excludes_unresolved_and_outer_evaluation_rows(tmp_path: Path) -> None:
    root = _strict_root(tmp_path)
    output = materialize_strict_predecessor_family_selections(
        [root], tmp_path / "selections", selection_cutoff_utc="2024-01-01T02:00:00Z",
        config=PredecessorFamilySelectionConfig(
            min_rows=1, min_independent_timestamps=1, min_trading_days=1, min_symbols=1,
            max_context_families_per_scope=5, max_covariance_families_per_scope=5,
            max_relationship_families_per_scope=5,
        ),
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == FAMILY_SELECTION_ROOT_STATUS
    selected = load_frozen_family_selection(output / "h3_context_family_selection.json", expected_kind="context")
    assert {item[3] for item in selected.selected_families} == {"early_family"}
    audit = pd.read_parquet(output / "predecessor_family_selection_audit.parquet")
    assert set(audit["rule_signature"]) == {"early_family"}
    assert audit["latest_label_available_utc"].lt(pd.Timestamp("2024-01-01T02:00:00Z")).all()
    # The family selector has no realised-economics input at all.  Changing a
    # permitted predecessor label must therefore not change its frozen family
    # identity; the label availability cutoff is provenance, not a target.
    labels = root / "strict_oof_base_reasoning" / "transport_a" / "folds" / "long" / "fold-1" / "p_clear" / "base_reasoning_labels.parquet"
    mutated = pd.read_parquet(labels)
    mutated.loc[mutated["candidate_id"].eq("a"), "label__net_bps"] = -9_999.0
    mutated.to_parquet(labels, index=False)
    repeated = materialize_strict_predecessor_family_selections(
        [root], tmp_path / "selections_mutated", selection_cutoff_utc="2024-01-01T02:00:00Z",
        config=PredecessorFamilySelectionConfig(
            min_rows=1, min_independent_timestamps=1, min_trading_days=1, min_symbols=1,
            max_context_families_per_scope=5, max_covariance_families_per_scope=5,
            max_relationship_families_per_scope=5,
        ),
    )
    assert load_frozen_family_selection(
        repeated / "h3_context_family_selection.json", expected_kind="context",
    ).selected_families == selected.selected_families
    # The root retains candidate a as pre-cutoff state history, but the
    # returned cutoff becomes a hard H3/H4/H5 emission boundary.
    assert validate_selection_application([selected], strict_roots=[root]) == pd.Timestamp("2024-01-01T02:00:00Z")


def test_bounded_predecessor_selector_matches_legacy_support_audit(tmp_path: Path) -> None:
    """The disk-spilling path is a refactor, not a semantic change.

    This deliberately compares against the previous in-memory collector on a
    tiny root.  Production never invokes that collector because it builds the
    full candidate/contribution merge that can exceed available memory.
    """

    root = _strict_root(tmp_path)
    config = PredecessorFamilySelectionConfig(
        min_rows=1, min_independent_timestamps=1, min_trading_days=1, min_symbols=1,
        max_context_families_per_scope=5, max_covariance_families_per_scope=5,
        max_relationship_families_per_scope=5,
    )
    cutoff = pd.Timestamp("2024-01-01T02:00:00Z")
    legacy = _selection_audit(
        collect_completed_strict_oof_family_inputs([root]), cutoff=cutoff, config=config,
    )
    spill = tmp_path / "stream_spill"
    try:
        streamed, source = _streaming_selection_audit(
            [root], cutoff=cutoff, config=config, spill_directory=spill,
        )
    finally:
        shutil.rmtree(spill, ignore_errors=True)
    columns = [
        "feature_contract_sha256", "side_name", "head_name", "rule_signature",
        "contribution_direction", "predecessor_rows", "predecessor_timestamps",
        "predecessor_days", "predecessor_symbols", "contribution_abs_mass",
        "contribution_abs_mean", "latest_label_available_utc", "eligible_support",
        "selection_support_score",
    ]
    pd.testing.assert_frame_equal(
        legacy.loc[:, columns].reset_index(drop=True), streamed.loc[:, columns].reset_index(drop=True),
        check_dtype=False,
    )
    assert source.strict_roots == (str(root),)
    assert not any("leaf" in column.lower() for column in streamed.columns)


def test_event_store_predecessor_selector_matches_legacy_and_seals_lineage(tmp_path: Path) -> None:
    root = _strict_root(tmp_path)
    config = PredecessorFamilySelectionConfig(
        min_rows=1, min_independent_timestamps=1, min_trading_days=1, min_symbols=1,
        max_context_families_per_scope=5, max_covariance_families_per_scope=5,
        max_relationship_families_per_scope=5,
    )
    spool = spool_completed_strict_oof_family_inputs([root], tmp_path / "spool")
    store = build_strict_event_store(spool.root, tmp_path / "event_store")
    output = materialize_strict_predecessor_family_selections(
        None, tmp_path / "event_selections", event_store=store.root,
        selection_cutoff_utc="2024-01-01T02:00:00Z", config=config,
    )
    selected = load_frozen_family_selection(output / "h3_context_family_selection.json", expected_kind="context")
    assert {item[3] for item in selected.selected_families} == {"early_family"}
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["event_store_root"] == str(store.root)
    assert isinstance(manifest["event_store_manifest_sha256"], str)
    selection_payload = json.loads((output / "h3_context_family_selection.json").read_text())
    assert selection_payload["source"]["event_store_root"] == str(store.root)


def test_strict_context_wrapper_reuses_exact_candidate_identity_and_includes_november(monkeypatch, tmp_path: Path) -> None:
    root = _strict_root(tmp_path)
    calls: list[dict[str, object]] = []

    def fake_materialize(**kwargs):
        calls.append(kwargs)
        output = Path(kwargs["output_dir"])
        output.mkdir()
        candidates = pd.read_parquet(kwargs["candidate_path"])
        timeline = pd.DataFrame({
            "source_utc": pd.date_range(STRICT_CONTEXT_START_UTC, STRICT_CONTEXT_END_EXCLUSIVE_UTC, freq="MS", inclusive="left"),
        })
        for column in DEFAULT_HEALTH_CONTEXT_COLUMNS:
            timeline[column] = np.float32(0.25)
        timeline.to_parquet(output / "hourly_oof_market_regimes.parquet", index=False)
        candidate = candidates.copy()
        candidate["regime_available_utc"] = candidate["__ts__"]
        for column in DEFAULT_HEALTH_CONTEXT_COLUMNS:
            candidate[column] = np.float32(0.25)
        candidate.to_parquet(output / "candidate_oof_market_regimes.parquet", index=False)
        (output / "manifest.json").write_text(json.dumps({"status": "MATERIALIZED_CAUSAL_FROZEN_OOF"}))
        return output

    import scripts.materialize_oof_market_regime_systems as regime_script
    monkeypatch.setattr(regime_script, "materialize", fake_materialize)
    panel = tmp_path / "panel.parquet"
    pd.DataFrame({"source_utc": [pd.Timestamp("2022-01-01T00:00:00Z")]}).to_parquet(panel, index=False)
    output = materialize_strict_fold_causal_context([root], tmp_path / "context", panel_path=panel)
    assert calls[0]["evaluation_start"] == STRICT_CONTEXT_START_UTC.isoformat()
    assert calls[0]["evaluation_end"] == STRICT_CONTEXT_END_EXCLUSIVE_UTC.isoformat()
    context, columns, manifest = load_strict_fold_context(output)
    assert columns == DEFAULT_HEALTH_CONTEXT_COLUMNS
    assert len(context) == 17
    assert manifest["window"]["includes_untouched_november_2024_context"] is True
