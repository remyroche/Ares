from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_i_target_adapter import (
    LEGACY_R3_MULTICLASS3,
    SOFT_SCALAR_S,
)
from scripts.run_stage_i_direct_fq3_meta_feature_selection import _bounded_relief_checkpoint_root, _load_feature_sidecar, _native_handoff
from scripts.run_stage_i_direct_fq3_meta_feature_selection import _validate_pristine_orchestrator_directory
from scripts.run_stage_i_direct_fq3_meta_feature_selection import (
    _bootstrap_direct_runner_receipt, _prepare_clean_resume_attempt,
    _publish_completed_resume_attempt, _validate_orchestrator_receipt,
)


def test_native_r3_handoff_preserves_contrast_and_three_state_simplex() -> None:
    frame = pd.DataFrame({
        "r3_p_adverse": [0.6, 0.1], "r3_p_weak": [0.3, 0.2], "r3_p_clear": [0.1, 0.7],
        "base_raw_score": [-0.5, 0.6],
    })
    handoff, states, domain = _native_handoff(frame, LEGACY_R3_MULTICLASS3)
    assert states == ("base_state_p0", "base_state_p1", "base_state_p2")
    assert domain == (-1.0, 1.0)
    np.testing.assert_allclose(handoff.base_raw_score, frame.r3_p_clear - frame.r3_p_adverse)
    assert {"base_output_entropy", "base_output_top2_margin", "base_output_max_probability"}.issubset(handoff)


def test_sliced_relief_checkpoint_uses_new_namespace_without_mutating_v1(tmp_path) -> None:
    root = tmp_path / "long"
    old = root / "_bounded_relief"
    old.mkdir(parents=True)
    (old / "state.json").write_text('{"schema":"stage_i_bounded_relief_state_v1"}')
    a32 = _bounded_relief_checkpoint_root(root, feature_chunk_size=64, anchor_chunk_size=32)
    a128 = _bounded_relief_checkpoint_root(root, feature_chunk_size=64, anchor_chunk_size=128)
    assert a32 == root / "_bounded_relief_v2_f64_a32"
    assert a128 == root / "_bounded_relief_v2_f64_a128"
    assert a32 != a128
    assert (old / "state.json").is_file()


def test_native_scalar_handoff_requires_exact_two_state_contract() -> None:
    valid = pd.DataFrame({
        "base_raw_score": [0.2, 0.8], "base_state_p0": [0.8, 0.2], "base_state_p1": [0.2, 0.8],
    })
    handoff, states, domain = _native_handoff(valid, SOFT_SCALAR_S)
    assert states == ("base_state_p0", "base_state_p1") and domain == (0.0, 1.0)
    with pytest.raises(ValueError, match="state-width drift"):
        _native_handoff(valid.drop(columns="base_state_p1"), SOFT_SCALAR_S)


def test_pristine_orchestrator_directory_is_accepted_but_partial_artifacts_fail(tmp_path) -> None:
    import json

    root = tmp_path / "long"
    root.mkdir()
    receipt = {
        "schema": "stage_i_bounded_side_orchestrator_v2", "side": "long",
        "request_sha256": "a" * 64,
        "command": ["python3", "runner.py", "--side", "long", "--resume"],
    }
    (root / "orchestrator_request.json").write_text(json.dumps(receipt), encoding="utf-8")
    assert _validate_pristine_orchestrator_directory(root, side="long") == receipt
    (root / "partial_checkpoint.parquet").write_bytes(b"partial")
    with pytest.raises(FileExistsError, match="partial direct-FQ3 selector artifacts"):
        _validate_pristine_orchestrator_directory(root, side="long")


def test_pristine_orchestrator_directory_rejects_cross_side_or_unbound_receipt(tmp_path) -> None:
    import json

    root = tmp_path / "long"
    root.mkdir()
    (root / "orchestrator_request.json").write_text(json.dumps({
        "schema": "stage_i_bounded_side_orchestrator_v2", "side": "short",
        "request_sha256": "a" * 64, "command": ["python3", "runner.py", "--resume"],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="cross-side"):
        _validate_pristine_orchestrator_directory(root, side="long")


def test_direct_runner_bootstraps_only_its_known_bounded_root_layout(tmp_path) -> None:
    root = tmp_path / "short"
    root.mkdir()
    (root / "base_candidate_handoff.parquet").write_bytes(b"identity-bound-audit")
    (root / "_bounded_univariate").mkdir()
    lineage = {
        "selector_sample_manifest_sha256": "a" * 64,
        "selector_feature_contract_sha256": "b" * 64,
        "base_selector_manifest_sha256": "c" * 64,
        "target_contract_sha256": "d" * 64,
        "correlation_policy": "grouped-preserve",
        "hpo_trials": 60,
        "hpo_patience": 15,
        "base_candidate_fraction": 1.0,
        "mda_support_mode": "full",
    }
    receipt = _bootstrap_direct_runner_receipt(root, side="short", lineage=lineage)
    assert receipt["origin"] == "direct_runner_self_bootstrap_v1"
    assert _validate_orchestrator_receipt(root, side="short") == receipt

    unsafe = tmp_path / "unsafe"
    unsafe.mkdir()
    (unsafe / "unknown.bin").write_bytes(b"not-a-runner-artifact")
    with pytest.raises(FileExistsError, match="unknown artifacts"):
        _bootstrap_direct_runner_receipt(unsafe, side="short", lineage=lineage)


def test_clean_resume_attempt_preserves_partial_evidence_and_reuses_only_complete_attempt(tmp_path) -> None:
    import json
    from extreme_price_movements.stage_i_target_adapter import file_sha256

    root = tmp_path / "long"
    root.mkdir()
    (root / "orchestrator_request.json").write_text(json.dumps({
        "schema": "stage_i_bounded_side_orchestrator_v2", "side": "long",
        "request_sha256": "a" * 64, "command": ["python3", "runner.py", "--resume"],
    }))
    partial = root / "mda" / "round_1.json"
    partial.parent.mkdir()
    partial.write_text('{"incomplete": true}')
    lineage = {
        "selector_sample_manifest_sha256": "b" * 64,
        "selector_feature_contract_sha256": "c" * 64,
        "base_selector_manifest_sha256": "d" * 64,
        "target_contract_sha256": "e" * 64,
        "correlation_policy": "grouped-preserve", "hpo_trials": 60,
        "hpo_patience": 15, "base_candidate_fraction": 1.0,
        "mda_support_mode": "full",
    }
    attempt, complete = _prepare_clean_resume_attempt(root, side="long", lineage=lineage)
    assert not complete and attempt.name == "attempt-0001"
    assert partial.read_text() == '{"incomplete": true}'
    request = json.loads((attempt / "attempt_request.json").read_text())
    assert request["resume_policy"] == "clean_restart_no_partial_mda_checkpoint_reuse"
    assert "mda/round_1.json" in request["prior_evidence_inventory"]

    # An incomplete attempt is preserved, not resumed in place.
    (attempt / "mda_partial.bin").write_bytes(b"partial")
    next_attempt, next_complete = _prepare_clean_resume_attempt(root, side="long", lineage=lineage)
    assert not next_complete and next_attempt.name == "attempt-0002"

    oof = next_attempt / "selector_meta_oof.parquet"
    oof.write_bytes(b"complete-oof")
    manifest = {
        "status": "complete", "side": "long",
        "artifact_sha256": {oof.name: file_sha256(oof)},
    }
    (next_attempt / "manifest.json").write_text(json.dumps(manifest))
    _publish_completed_resume_attempt(root, side="long", attempt=next_attempt, lineage=lineage)
    resolved, is_complete = _prepare_clean_resume_attempt(root, side="long", lineage=lineage)
    assert is_complete and resolved == next_attempt

    # Pointer contract is immutable: altered source lineage cannot adopt it.
    changed = dict(lineage)
    changed["hpo_trials"] = 61
    with pytest.raises(ValueError, match="lineage drift"):
        _prepare_clean_resume_attempt(root, side="long", lineage=changed)


def test_feature_sidecar_is_exactly_identity_bound_and_coverage_gated(tmp_path) -> None:
    ts = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    selector = pd.DataFrame({
        "candidate_id": ["a", "b", "c"], "__ts__": ts, "__symbol__": ["BTC", "ETH", "SOL"],
    })
    sidecar = selector.iloc[[2, 0, 1]].copy()
    sidecar["meta_aegmm_gmm_ood_score"] = [0.3, 0.1, 0.2]
    path = tmp_path / "sidecar.parquet"
    sidecar.to_parquet(path, index=False)
    values, lineage = _load_feature_sidecar(
        path, fields=("meta_aegmm_gmm_ood_score",), selector_features=selector, min_coverage=0.90,
    )
    np.testing.assert_allclose(values["meta_aegmm_gmm_ood_score"], [0.1, 0.2, 0.3])
    assert lineage["enabled"] is True and lineage["coverage"]["meta_aegmm_gmm_ood_score"] == 1.0

    sidecar.loc[0, "meta_aegmm_gmm_ood_score"] = np.nan
    sidecar.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="below 0.9000"):
        _load_feature_sidecar(
            path, fields=("meta_aegmm_gmm_ood_score",), selector_features=selector, min_coverage=0.90,
        )


def test_runner_smoke_reaches_selector_with_train_meta_after_pristine_orchestrator_dir(tmp_path, monkeypatch) -> None:
    import json
    from extreme_price_movements.stage_i_target_adapter import bind_target_contract, file_sha256
    import scripts.run_stage_i_direct_fq3_meta_feature_selection as runner

    n = 600
    signal = pd.date_range("2022-01-01", periods=n, freq="h", tz="UTC")
    decision = signal + pd.Timedelta(hours=1)
    available = decision + pd.Timedelta(hours=12)
    ids = [f"long-{i}" for i in range(n)]
    symbols = np.resize(np.asarray(["BTC", "ETH"]), n)
    net = np.sin(np.arange(n) / 11.0) * 250.0
    ledger = pd.DataFrame({
        "candidate_id": ids, "__ts__": signal, "__symbol__": symbols, "side_name": "long",
        "decision_ts": decision, "label_available_ts": available,
        "r3_class": np.resize(np.asarray([0, 1, 2]), n),
        "robust_clear_soft_b25_t50": np.resize(np.asarray([0.1, 0.5, 0.9]), n),
        "t2_tp6_sl4_event": np.resize(np.asarray([0, 1, 2]), n), "exact_net_bps": net,
    })
    features = ledger.loc[:, ["candidate_id", "__ts__", "__symbol__"]].copy()
    features["regime_transition_entropy_12h"] = np.sin(np.arange(n) / 7.0)
    features["chop_score"] = np.cos(np.arange(n) / 9.0)
    selector = tmp_path / "selector"
    selector.mkdir()
    ledger_path, features_path = selector / "selector_ledger.parquet", selector / "selector_features.parquet"
    ledger.to_parquet(ledger_path, index=False)
    features.to_parquet(features_path, index=False)
    (selector / "selector_feature_contract.json").write_text(json.dumps({"features": list(features.columns[3:])}))
    (selector / "manifest.json").write_text(json.dumps({
        "status": "complete", "artifact_integrity": {
            "schema": "stage_i_selector_artifact_integrity_v1",
            "selector_ledger_sha256": file_sha256(ledger_path),
            "selector_features_sha256": file_sha256(features_path),
        },
    }))
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "manifest.json").write_text(json.dumps({"request_sha256": "c" * 64}))
    base_root = tmp_path / "base"
    side_root = base_root / "long"
    side_root.mkdir(parents=True)
    contract_frame = pd.DataFrame({
        "candidate_id": ids, "__ts__": signal, "__symbol__": symbols, "side_name": "long",
        "base_target": np.clip((np.sin(np.arange(n) / 13.0) + 1.0) / 2.0, 0.0, 1.0),
        "gross_bps": net + 100.0, "net_bps": net, "target_valid": True, "sample_weight": 1.0,
    })
    base_contract = bind_target_contract(
        contract_frame, family=SOFT_SCALAR_S, layer="base", target_name="S", geometry="TP7_SL2_H12",
        target_columns=("base_target",),
    )
    score = contract_frame.base_target.to_numpy(np.float32)
    oof = ledger.loc[:, ["candidate_id", "__ts__", "__symbol__", "side_name", "decision_ts", "label_available_ts"]].copy()
    oof["exact_net_bps"], oof["exact_gross_bps"] = net, net + 100.0
    oof["base_raw_score"], oof["base_state_p0"], oof["base_state_p1"] = score, 1.0 - score, score
    oof_path = side_root / "selector_base_oof.parquet"
    oof.to_parquet(oof_path, index=False)
    (side_root / "manifest.json").write_text(json.dumps({
        "status": "complete", "side": "long", "selector_base_oof_sha256": file_sha256(oof_path),
        "hpo_oof_regeneration_fold_audit": [{"strict_prior_resolved": True}],
        "target_contract": base_contract.to_dict(),
    }))
    output = tmp_path / "output"
    pristine = output / "long"
    pristine.mkdir(parents=True)
    (pristine / "orchestrator_request.json").write_text(json.dumps({
        "schema": "stage_i_bounded_side_orchestrator_v2", "side": "long",
        "request_sha256": "d" * 64, "command": ["python3", "runner.py", "--resume"],
    }))

    class ReachedSelector(RuntimeError):
        pass

    def fake_selector(*_args, **kwargs):
        assert kwargs["candidate_kwargs"]["hpo_objective_mode"] == "train_meta"
        assert kwargs["candidate_kwargs"]["frozen_base_direct_score_units"] == "native_score"
        assert kwargs["candidate_kwargs"]["candidate_ids"].shape == (n,)
        assert set(kwargs["candidate_kwargs"]["label_context"]) == {
            "side_name", "valid_resolved_support",
        }
        assert kwargs["candidate_kwargs"]["cfg"]["mda_config"]["archetype_conditioned_enabled"] is False
        assert kwargs["candidate_kwargs"]["mda_reference"]["archetype_labels"] is None
        support = kwargs["candidate_kwargs"]["mda_reference"]["archetype_label_audit"]
        assert support["training_only"] is True
        assert support["inference_feature"] is False
        assert support["side"] == "long"
        raise ReachedSelector

    monkeypatch.setattr(runner, "run_stage_i_head_selection", fake_selector)
    with pytest.raises(ReachedSelector):
        runner.main([
            "--selector-dir", str(selector), "--base-selection-dir", str(base_root),
            "--output-dir", str(output), "--side", "long", "--resume",
            "--target-neutral-cache-dir", str(cache),
            "--required-regime-feature", "regime_transition_entropy_12h",
            "--required-context-feature", "chop_score",
            "--mda-support-mode", "target-only",
        ])
