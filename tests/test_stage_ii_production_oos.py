from __future__ import annotations

from hashlib import sha256

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_ii_production_oos import (
    StageIIFoldLineage,
    StageIILockedOOSRequest,
    StageIILockedOOSScoringRequest,
    StageIILockedOOSScoringResult,
    StageIIProductionError,
    StageIIWindowContract,
    StageIIWinnerManifest,
    build_stage_ii_locked_oos_report,
    load_stage_ii_winner_bundle,
    publish_stage_ii_locked_oos_bundle,
    publish_stage_ii_winner_bundle,
    run_stage_ii_locked_oos_scoring,
    run_and_publish_stage_ii_locked_oos_scoring,
    validate_stage_ii_locked_oos,
)
from extreme_price_movements.stage_ii_production_oos import _fold_lineage_hash


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _window() -> StageIIWindowContract:
    return StageIIWindowContract(
        "2024-01-01T00:00:00Z", "2024-02-01T00:00:00Z",
        "2024-02-01T00:00:00Z", "2024-03-01T00:00:00Z",
        "2024-04-01T00:00:00Z", "2024-05-01T00:00:00Z",
    )


def _dev_identity() -> pd.DataFrame:
    decision = pd.date_range("2024-02-02", periods=6, freq="h", tz="UTC")
    return pd.DataFrame({
        "candidate_id": [f"dev-{i}" for i in range(len(decision))],
        "symbol": ["BTC" if i % 2 else "ETH" for i in range(len(decision))],
        "signal_close_ts": decision - pd.Timedelta(hours=1),
        "decision_ts": decision,
        "label_available_ts": decision + pd.Timedelta(hours=12),
        "side_name": ["long" if i % 2 else "short" for i in range(len(decision))],
    })


def _manifest() -> StageIIWinnerManifest:
    identity = _dev_identity()
    from extreme_price_movements.stage_ii_production_oos import _identity_digest

    return StageIIWinnerManifest(
        run_id="stage-ii-dev-001", dataset_id="frozen_panel", dataset_sha256=_digest("dataset"),
        label_manifest_id="tp6_sl4_h12", label_manifest_sha256=_digest("labels"),
        universe_id="common30", universe_sha256=_digest("universe"), code_revision="abc1234",
        stage_i_base_winner_artifact_id="stage-i-base-001",
        stage_i_base_winner_artifact_sha256=_digest("base-artifact"),
        stage_i_base_oof_ledger_sha256=_digest("base-ledger"),
        selected_discovery_candidate_id="mfe_k3", selected_control_arm="both",
        selected_config={"components": 3, "arm": "both"},
        ordered_meta_features=("prequential_base_expected_net_bps", "r3_p_adverse", "r3_p_weak", "r3_p_clear", "meta_signal"),
        ordered_archetype_features=("meta_conversion_arch_prob__0", "meta_conversion_arch_prob__1", "meta_conversion_arch_prob__2", "meta_conversion_arch_prob__unknown", "meta_conversion_arch_prior_residual_bps"),
        development_identity_sha256=_identity_digest(identity, columns=("candidate_id", "symbol", "signal_close_ts", "decision_ts", "label_available_ts", "side_name")),
        window=_window(),
    )


def _publish_winner(tmp_path) -> str:
    return str(publish_stage_ii_winner_bundle(
        tmp_path / "winner", manifest=_manifest(), development_identity=_dev_identity(),
        development_metrics=pd.DataFrame({"net": [1.0]}), candidate_audit=pd.DataFrame({"candidate": ["mfe_k3"]}),
        control_metrics=pd.DataFrame({"arm": ["both"]}),
    ))


def _ledger(rows: int = 24) -> pd.DataFrame:
    decision = pd.date_range("2024-04-02", periods=rows, freq="12h", tz="UTC")
    side = np.where(np.arange(rows) % 2, "long", "short")
    clear = np.where(np.arange(rows) % 3 == 0, .8, .35).astype("float32")
    adverse = np.where(np.arange(rows) % 3 == 0, .1, .3).astype("float32")
    weak = 1 - clear - adverse
    base = 100 * (clear - adverse)
    residual = np.where(np.arange(rows) % 3 == 0, 10.0, -5.0)
    net = base + residual + np.where(np.arange(rows) % 3 == 0, 20.0, -10.0)
    candidate = [f"eval-{i}" for i in range(rows)]
    symbol = np.where(np.arange(rows) % 3, "BTC", "ETH")
    frame = pd.DataFrame({
        "candidate_id": candidate, "symbol": symbol,
        "signal_close_ts": decision - pd.Timedelta(hours=1), "decision_ts": decision,
        "label_available_ts": decision + pd.Timedelta(hours=12), "side_name": side,
        "exact_net_bps": net, "exact_gross_bps": net + 100.0, "total_cost_bps": 100.0,
        "prequential_base_expected_net_bps": base,
        "r3_p_adverse": adverse, "r3_p_weak": weak, "r3_p_clear": clear,
        "r3_raw_clear_minus_adverse": clear-adverse,
        "base_is_strict_oof": True, "base_source_side": side,
        "base_score_semantics": "same_side_direct_strict_oof_probabilities_without_conversion",
        "base_oof_fold_id": 0, "base_train_max_label_available_ts": pd.Timestamp("2024-03-31T00:00:00Z"),
        "base_map_is_prequential": True, "base_map_source_side": side,
        "base_map_max_label_available_ts": pd.Timestamp("2024-03-31T00:00:00Z"),
        "meta_raw_predicted_residual_bps": residual, "meta_reconstructed_expected_net_bps": base+residual,
        "meta_signal": np.linspace(-1., 1., rows),
        "meta_is_strict_oof": True, "meta_source_side": side,
        "meta_score_semantics": "raw_predicted_residual_bps", "meta_oof_fold_id": 0,
        "meta_train_max_label_available_ts": pd.Timestamp("2024-03-31T00:00:00Z"),
        "base_candidate_id": candidate, "base_symbol": symbol, "base_decision_ts": decision, "base_side_name": side,
        "meta_candidate_id": candidate, "meta_symbol": symbol, "meta_decision_ts": decision, "meta_side_name": side,
        "base_causal_21d_side_expected_net_bps": base, "base_causal_21d_side_admitted_ge_50bps": base >= 50,
        "meta_causal_21d_side_expected_net_bps": base+residual, "meta_causal_21d_side_admitted_ge_50bps": base+residual >= 50,
        "base_causal_21d_admission_source_side": side, "base_causal_21d_admission_is_prequential": True,
        "base_causal_21d_admission_max_label_available_ts": pd.Timestamp("2024-03-31T00:00:00Z"), "base_causal_21d_admission_window_days": 21,
        "meta_causal_21d_admission_source_side": side, "meta_causal_21d_admission_is_prequential": True,
        "meta_causal_21d_admission_max_label_available_ts": pd.Timestamp("2024-03-31T00:00:00Z"), "meta_causal_21d_admission_window_days": 21,
        "meta_conversion_arch_prob__0": .4, "meta_conversion_arch_prob__1": .3,
        "meta_conversion_arch_prob__2": .3, "meta_conversion_arch_prob__unknown": 0.,
        "meta_conversion_arch_prior_residual_bps": 1.,
    })
    return frame


def _request(winner: str, ledger: pd.DataFrame | None = None) -> StageIILockedOOSRequest:
    fold = StageIIFoldLineage(0, "2024-03-31T00:00:00Z", "2024-04-01T00:00:00Z", "2024-05-01T00:00:00Z")
    manifest = _manifest()
    return StageIILockedOOSRequest(winner, _ledger() if ledger is None else ledger, (fold,), (fold,), manifest.stage_i_base_winner_artifact_sha256, manifest.stage_i_base_oof_ledger_sha256)


def test_winner_bundle_is_atomic_checksummed_and_immutable(tmp_path) -> None:
    path = _publish_winner(tmp_path)
    loaded = load_stage_ii_winner_bundle(path)
    assert loaded.selected_control_arm == "both"
    with pytest.raises(StageIIProductionError, match="already exists"):
        _publish_winner(tmp_path)
    # A mutation after publication is detected before an OOS score can use it.
    from pathlib import Path
    (Path(path) / "feature_contract.json").write_text("{}", encoding="utf-8")
    with pytest.raises(StageIIProductionError, match="checksum"):
        load_stage_ii_winner_bundle(path)


def test_locked_oos_refuses_eval_window_leakage_and_identity_or_lineage_spoofing(tmp_path) -> None:
    winner = _publish_winner(tmp_path)
    leaking = _ledger()
    leaked_decision = pd.Timestamp("2024-02-15T00:00:00Z")
    leaking.loc[0, ["decision_ts", "base_decision_ts", "meta_decision_ts"]] = leaked_decision
    leaking.loc[0, "signal_close_ts"] = leaked_decision - pd.Timedelta(hours=1)
    leaking.loc[0, "label_available_ts"] = leaked_decision + pd.Timedelta(hours=12)
    with pytest.raises(StageIIProductionError, match="development/history"):
        validate_stage_ii_locked_oos(_request(winner, leaking))
    spoofed = _ledger()
    spoofed.loc[0, "meta_symbol"] = "NOT_BTC"
    with pytest.raises(StageIIProductionError, match="identity join"):
        validate_stage_ii_locked_oos(_request(winner, spoofed))
    cutoff = _ledger()
    cutoff.loc[0, "meta_train_max_label_available_ts"] = pd.Timestamp("2024-04-02T00:00:00Z")
    with pytest.raises(StageIIProductionError, match="strict prior-resolved fold cutoff"):
        validate_stage_ii_locked_oos(_request(winner, cutoff))
    map_spoof = _ledger()
    map_spoof.loc[0, "base_map_is_prequential"] = False
    with pytest.raises(StageIIProductionError, match="expected-net map"):
        validate_stage_ii_locked_oos(_request(winner, map_spoof))
    admission_spoof = _ledger()
    admission_spoof.loc[0, "meta_causal_21d_side_admitted_ge_50bps"] = False
    with pytest.raises(StageIIProductionError, match="admission flag"):
        validate_stage_ii_locked_oos(_request(winner, admission_spoof))


def test_report_ranks_one_pooled_global_book_and_handles_empty_admission(tmp_path) -> None:
    winner = _publish_winner(tmp_path)
    frame = _ledger()
    # The top base score is a short in this construction; a side-quota ranker
    # would also include long rows at top 1%, while the global book selects one.
    summary, contribution = build_stage_ii_locked_oos_report(_request(winner, frame))
    top = summary.loc[(summary.layer.eq("base")) & summary.admission_scope.eq("without_21d_admission") & np.isclose(summary.top_fraction, .01)].iloc[0]
    assert top.selected_rows == 1
    selected_side = contribution.loc[(contribution.layer.eq("base")) & contribution.admission_scope.eq("without_21d_admission") & np.isclose(contribution.top_fraction, .01) & contribution.scope.eq("side")]
    assert len(selected_side) == 1
    assert {"candidate_population", "selected_tail"}.issubset(set(summary.record_type))
    assert {"worst_month_net_bps_per_trade", "worst_week_net_bps_per_trade", "worst_month_side_net_bps_per_trade", "worst_week_side_net_bps_per_trade"}.issubset(summary.columns)
    empty = _ledger()
    empty["base_causal_21d_side_admitted_ge_50bps"] = False
    empty["meta_causal_21d_side_admitted_ge_50bps"] = False
    empty["base_causal_21d_side_expected_net_bps"] = np.nan
    empty["meta_causal_21d_side_expected_net_bps"] = np.nan
    summary, _ = build_stage_ii_locked_oos_report(_request(winner, empty))
    admitted = summary.loc[summary.admission_scope.eq("with_21d_side_local_admission")]
    assert admitted.selected_rows.eq(0).all()
    assert admitted.net_bps_per_trade.isna().all()


def test_locked_oos_publication_writes_one_final_ledger_and_does_not_reselect(tmp_path) -> None:
    winner = _publish_winner(tmp_path)
    output = publish_stage_ii_locked_oos_bundle(
        tmp_path / "oos", request=_request(winner), scorer_model_sha256=_digest("model"),
    )
    assert (output / "locked_oos_ledger.parquet").is_file()
    manifest = (output / "run_manifest.json").read_text(encoding="utf-8")
    assert '"selection_forbidden": true' in manifest
    assert '"oos_content_sha256"' in manifest
    assert '"frozen_feature_contract_sha256"' in manifest
    assert '"scorer_model_sha256"' in manifest
    with pytest.raises(StageIIProductionError, match="new path"):
        publish_stage_ii_locked_oos_bundle(
            output, request=_request(winner), scorer_model_sha256=_digest("model"),
        )


def test_code_revision_and_one_shot_locked_scorer_contract(tmp_path) -> None:
    invalid = _manifest()
    with pytest.raises(StageIIProductionError, match="code_revision"):
        StageIIWinnerManifest(**{**invalid.__dict__, "code_revision": "version-one"}).validate()
    winner = _publish_winner(tmp_path)
    history_decision = pd.date_range("2024-01-05", periods=4, freq="h", tz="UTC")
    development_decision = pd.date_range("2024-02-05", periods=4, freq="h", tz="UTC")
    history = pd.DataFrame({"decision_ts": history_decision, "label_available_ts": history_decision + pd.Timedelta(hours=13)})
    development = pd.DataFrame({"decision_ts": development_decision, "label_available_ts": development_decision + pd.Timedelta(hours=13)})
    ledger = _ledger()
    evaluation_identity = ledger[["candidate_id", "symbol", "signal_close_ts", "decision_ts", "side_name"]].copy()
    manifest = _manifest()
    fold = StageIIFoldLineage(0, "2024-03-31T00:00:00Z", "2024-04-01T00:00:00Z", "2024-05-01T00:00:00Z")
    request = StageIILockedOOSScoringRequest(winner, history, development, evaluation_identity, (fold,), (fold,), manifest.stage_i_base_winner_artifact_sha256, manifest.stage_i_base_oof_ledger_sha256)
    calls: list[object] = []
    from pathlib import Path
    provenance = {
        "winner_manifest_sha256": sha256((Path(winner) / "winner_manifest.json").read_bytes()).hexdigest(),
        "feature_contract_sha256": sha256(b'{"meta":["prequential_base_expected_net_bps","r3_p_adverse","r3_p_weak","r3_p_clear","meta_signal"],"archetype":["meta_conversion_arch_prob__0","meta_conversion_arch_prob__1","meta_conversion_arch_prob__2","meta_conversion_arch_prob__unknown","meta_conversion_arch_prior_residual_bps"]}').hexdigest(),
        "label_manifest_sha256": manifest.label_manifest_sha256,
        "stage_i_base_winner_artifact_sha256": manifest.stage_i_base_winner_artifact_sha256,
        "stage_i_base_oof_ledger_sha256": manifest.stage_i_base_oof_ledger_sha256,
        "base_fold_lineage_sha256": _fold_lineage_hash((fold,)),
        "meta_fold_lineage_sha256": _fold_lineage_hash((fold,)),
        "model_sha256": _digest("model"), "reselection_forbidden": True, "hpo_forbidden": True,
        "selected_discovery_candidate_id": manifest.selected_discovery_candidate_id,
        "selected_control_arm": manifest.selected_control_arm,
    }
    def scorer(context):
        calls.append(context)
        assert context["reselection_forbidden"] and context["hpo_forbidden"]
        return StageIILockedOOSScoringResult(ledger, provenance)
    _, scored = run_stage_ii_locked_oos_scoring(request, scorer=scorer)
    assert len(calls) == 1 and len(scored) == len(ledger)
    calls.clear()
    output = run_and_publish_stage_ii_locked_oos_scoring(tmp_path / "scored_oos", request, scorer=scorer)
    assert len(calls) == 1 and (output / "locked_oos_ledger.parquet").is_file()
    bad_development = development.copy()
    bad_development["label_available_ts"] = pd.Timestamp("2024-04-01T00:00:00Z")
    with pytest.raises(StageIIProductionError, match="unresolved"):
        run_stage_ii_locked_oos_scoring(
            StageIILockedOOSScoringRequest(winner, history, bad_development, evaluation_identity, (fold,), (fold,), manifest.stage_i_base_winner_artifact_sha256, manifest.stage_i_base_oof_ledger_sha256),
            scorer=scorer,
        )
