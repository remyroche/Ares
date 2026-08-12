from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_ii_production_oos import (
    StageIIWindowContract,
    StageIIWinnerManifest,
    _identity_digest,
    publish_stage_ii_winner_bundle,
)
from extreme_price_movements.stage_iii_shared_expert_runner import (
    StageIIIInputLineageContract,
    StageIIISequentialRunnerError,
    stage_iii_feature_contract_sha256,
)


def _load_cli():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_stage_iii_shared_expert_funnel.py"
    spec = importlib.util.spec_from_file_location("stage_iii_shared_expert_cli", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _digest_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _dev_identity() -> pd.DataFrame:
    decision = pd.date_range("2024-02-02", periods=4, freq="h", tz="UTC")
    return pd.DataFrame({
        "candidate_id": [f"dev-{i}" for i in range(len(decision))],
        "symbol": ["BTC", "ETH", "BTC", "ETH"],
        "signal_close_ts": decision - pd.Timedelta(hours=1),
        "decision_ts": decision,
        "label_available_ts": decision + pd.Timedelta(hours=12),
        "side_name": ["long", "short", "long", "short"],
    })


def _winner(tmp_path: Path) -> Path:
    dev = _dev_identity()
    manifest = StageIIWinnerManifest(
        run_id="stage-ii-test", dataset_id="panel", dataset_sha256=_sha("panel"),
        label_manifest_id="tp6", label_manifest_sha256=_sha("labels"),
        universe_id="common", universe_sha256=_sha("universe"), code_revision="deadbee",
        stage_i_base_winner_artifact_id="stage-i", stage_i_base_winner_artifact_sha256=_sha("base"),
        stage_i_base_oof_ledger_sha256=_sha("base-oof"), selected_discovery_candidate_id="arch",
        selected_control_arm="both", selected_config={"components": 3},
        ordered_meta_features=("market_confirmation",),
        ordered_archetype_features=("arch_prob",),
        development_identity_sha256=_identity_digest(
            dev, columns=("candidate_id", "symbol", "signal_close_ts", "decision_ts", "label_available_ts", "side_name")
        ),
        window=StageIIWindowContract(
            "2024-01-01T00:00:00Z", "2024-02-01T00:00:00Z",
            "2024-02-01T00:00:00Z", "2024-03-01T00:00:00Z",
            "2024-04-01T00:00:00Z", "2025-01-01T00:00:00Z",
        ),
    )
    return publish_stage_ii_winner_bundle(
        tmp_path / "winner", manifest=manifest, development_identity=dev,
        development_metrics=pd.DataFrame({"metric": [1.0]}),
        candidate_audit=pd.DataFrame({"candidate": ["arch"]}),
        control_metrics=pd.DataFrame({"arm": ["both"]}),
    )


def _locked_ledger() -> pd.DataFrame:
    # Six contiguous environments are enough to test the preflight fold contract.
    rows_per_environment = 4
    rows = rows_per_environment * 6
    decision = pd.date_range("2024-04-02", periods=rows, freq="24h", tz="UTC")
    index = np.arange(rows)
    side = np.where(index % 2 == 0, "long", "short")
    clear = np.full(rows, 0.6)
    adverse = np.full(rows, 0.2)
    base = 100.0 * (clear - adverse)
    net = base + np.where(index % 3 == 0, 20.0, -10.0)
    return pd.DataFrame({
        "candidate_id": [f"c-{i}" for i in index], "symbol": np.where(index % 3, "BTC", "ETH"),
        "signal_close_ts": decision - pd.Timedelta(hours=1), "decision_ts": decision,
        "label_available_ts": decision + pd.Timedelta(hours=12), "side_name": side,
        "exact_gross_bps": net + 100.0, "exact_net_bps": net, "total_cost_bps": 100.0,
        "prequential_base_expected_net_bps": base, "r3_p_adverse": adverse,
        "r3_p_weak": 1.0 - clear - adverse, "r3_p_clear": clear,
        "base_is_strict_oof": True, "base_source_side": side,
        "base_train_max_label_available_ts": decision - pd.Timedelta(hours=13),
        "base_score_semantics": "same_side_direct_strict_oof_probabilities_without_conversion",
        "base_map_is_prequential": True, "base_map_source_side": side,
        "base_map_max_label_available_ts": decision - pd.Timedelta(hours=1),
        "meta_causal_21d_side_admitted_ge_50bps": index % 3 != 0,
        "meta_causal_21d_admission_is_prequential": True,
        "meta_causal_21d_admission_source_side": side,
        "meta_causal_21d_admission_max_label_available_ts": decision - pd.Timedelta(hours=1),
        "meta_causal_21d_admission_window_days": 21,
    })


def _stage_ii_oos(tmp_path: Path, winner: Path, ledger: pd.DataFrame) -> Path:
    output = tmp_path / "oos"
    output.mkdir()
    ledger.to_parquet(output / "locked_oos_ledger.parquet", index=False)
    run = {
        "winner_manifest_sha256": _digest_file(winner / "winner_manifest.json"),
        "stage_i_base_winner_artifact_sha256": _sha("base"),
        "stage_i_base_oof_ledger_sha256": _sha("base-oof"),
        "selection_forbidden": True, "reselection_forbidden": True,
        "oos_content_sha256": _sha("content"),
    }
    (output / "run_manifest.json").write_text(json.dumps(run), encoding="utf-8")
    checksums = {name: _digest_file(output / name) for name in ("locked_oos_ledger.parquet", "run_manifest.json")}
    (output / "checksums.json").write_text(json.dumps(checksums), encoding="utf-8")
    return output


def _inputs(tmp_path: Path, ledger: pd.DataFrame) -> tuple[Path, Path, Path, Path]:
    enriched = ledger.copy()
    phase = np.arange(len(enriched)) / 4.0
    enriched["p_regime_calm"] = np.clip(0.5 + 0.2 * np.sin(phase), 0.05, 0.95)
    enriched["p_regime_stress"] = 1.0 - enriched["p_regime_calm"]
    enriched["market_confirmation"] = np.sin(phase)
    enriched["relationship_break_score"] = np.abs(np.sin(phase))
    enriched["contribution_ood_score"] = np.abs(np.cos(phase))
    enriched["active_failure_probability"] = 0.2
    enriched["cost_to_atr"] = 1.2
    enriched["cost_atr_is_causal"] = True
    enriched["soft_regime_is_causal_prequential"] = True
    enriched["soft_regime_fit_end_ts"] = enriched["decision_ts"] - pd.Timedelta(hours=1)
    enriched["environment"] = np.repeat([f"era_{i}" for i in range(6)], 4)
    enriched["broad_regime"] = np.where(enriched.p_regime_calm >= .5, "calm", "stress")
    ledger_path = tmp_path / "enriched.parquet"
    enriched.to_parquet(ledger_path, index=False)
    names = [
        "r3_p_clear", "r3_p_adverse", "r3_p_weak", "market_confirmation",
        "p_regime_calm", "p_regime_stress", "relationship_break_score",
        "contribution_ood_score", "active_failure_probability",
    ]
    feature = {
        "schema": "stage_iii_feature_admission_v1", "config": {"min_coverage": 0.9},
        "admitted_ordered_features": names,
        "feature_audit": [{"feature_name": name, "classification": "INVARIANT_CORE",
                            "admitted": True, "coverage": 1.0, "null_fraction": 0.0,
                            "finite_fraction": 1.0, "live_parity": True, "meta_allowed_key": True}
                          for name in names],
    }
    paths: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for name in ("r3", "base_map", "soft_regime", "label", "admission"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"source": name}), encoding="utf-8")
        paths[f"{name}_artifact_path"] = str(path)
        hashes[f"{name}_artifact_sha256"] = _digest_file(path)
    feature_path = tmp_path / "features.json"
    feature_path.write_text(json.dumps(feature), encoding="utf-8")
    lineage = StageIIIInputLineageContract(
        **paths, **hashes, feature_contract_artifact_path=str(feature_path),
        feature_contract_artifact_sha256=_digest_file(feature_path),
        feature_contract_sha256=stage_iii_feature_contract_sha256(names),
    )
    lineage_path = tmp_path / "lineage.json"
    lineage_path.write_text(lineage.to_json(), encoding="utf-8")
    groups_path = tmp_path / "groups.json"
    groups_path.write_text(json.dumps({
        "soft_regime_columns": ["p_regime_calm", "p_regime_stress"],
        "invariant_features": ["r3_p_clear", "r3_p_adverse", "r3_p_weak", "market_confirmation"],
        "regime_relative_features": ["market_confirmation"],
        "restricted_interaction_features": ["r3_p_clear", "r3_p_adverse"],
        "validity_feature_groups": {"relationship_breaks": ["relationship_break_score"],
                                    "contribution_ood": ["contribution_ood_score"],
                                    "active_failure_probability": ["active_failure_probability"]},
    }), encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "hard_regime_column": "broad_regime", "min_train_environments": 2,
        "min_train_rows": 1, "min_rows_per_side": 1, "top_fractions": [0.1],
        "primary_top_fraction": 0.1,
    }), encoding="utf-8")
    reproducibility_path = tmp_path / "repro.json"
    reproducibility_path.write_text(json.dumps({
        "run_id": "stage3", "dataset_id": "dataset", "dataset_sha256": _sha("dataset"),
        "label_manifest_id": "labels", "label_manifest_sha256": _sha("labels"),
        "feature_contract_sha256": lineage.feature_contract_sha256,
        "input_lineage_contract_sha256": lineage.contract_sha256, "code_revision": "deadbee",
        "split_definition": {"kind": "expanding"}, "model_configuration": {"routing": "shared"},
        "random_seeds": [17],
    }), encoding="utf-8")
    return ledger_path, lineage_path, groups_path, config_path, reproducibility_path


def test_cli_preflight_binds_enriched_input_to_frozen_stage_ii_oos(tmp_path: Path, capsys) -> None:
    cli = _load_cli()
    winner = _winner(tmp_path)
    source = _locked_ledger()
    oos = _stage_ii_oos(tmp_path, winner, source)
    ledger, lineage, groups, config, reproducibility = _inputs(tmp_path, source)
    assert cli.main([
        "--stage-ii-winner-bundle", str(winner), "--stage-ii-oos-bundle", str(oos),
        "--enriched-ledger", str(ledger), "--input-lineage", str(lineage),
        "--feature-groups", str(groups), "--runner-config", str(config),
        "--reproducibility", str(reproducibility), "--preflight",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "preflight_complete"


def test_cli_rejects_enriched_ledger_that_changes_frozen_stage_ii_economics(tmp_path: Path) -> None:
    cli = _load_cli()
    winner = _winner(tmp_path)
    source = _locked_ledger()
    oos = _stage_ii_oos(tmp_path, winner, source)
    ledger, lineage, groups, config, reproducibility = _inputs(tmp_path, source)
    altered = pd.read_parquet(ledger)
    altered.loc[0, "exact_net_bps"] += 1.0
    altered.to_parquet(ledger, index=False)
    with pytest.raises(cli.StageIIICommandError, match="changes frozen Stage-II economics"):
        cli.main([
            "--stage-ii-winner-bundle", str(winner), "--stage-ii-oos-bundle", str(oos),
            "--enriched-ledger", str(ledger), "--input-lineage", str(lineage),
            "--feature-groups", str(groups), "--runner-config", str(config),
            "--reproducibility", str(reproducibility), "--preflight",
        ])


def test_cli_can_bind_and_verify_a_direct_fq3_stage_ii_predecessor(tmp_path: Path) -> None:
    cli = _load_cli()
    winner = _winner(tmp_path)
    source = _locked_ledger()
    # The locked predecessor must carry the actual direct-FQ3 result, rather
    # than letting Stage III relabel a base-only bps field after publication.
    source["meta_is_strict_oof"] = True
    source["meta_source_side"] = source.side_name
    source["meta_score_semantics"] = "same_side_direct_base_output_correctness_q33_v1"
    source["meta_direct_score"] = np.linspace(-.2, .2, len(source))
    source["meta_p_error_tercile_0"] = .2
    source["meta_p_error_tercile_1"] = .5
    source["meta_p_error_tercile_2"] = .3
    source["prequential_joint_expected_net_bps"] = source.prequential_base_expected_net_bps + 1.0
    source["joint_expected_net_bps_semantics"] = "direct_fq3_reconstructed_causal_21d_common_bps_v1"
    source["joint_map_is_prequential"] = True
    source["joint_map_source_side"] = source.side_name
    source["joint_map_max_label_available_ts"] = source.decision_ts - pd.Timedelta(hours=1)
    oos = _stage_ii_oos(tmp_path, winner, source)
    ledger, lineage_path, groups, config, reproducibility_path = _inputs(tmp_path, source)
    enriched = pd.read_parquet(ledger)
    enriched.to_parquet(ledger, index=False)
    lineage = StageIIIInputLineageContract.from_dict(json.loads(lineage_path.read_text()))
    direct_lineage = StageIIIInputLineageContract(**{
        **lineage.to_dict(), "require_direct_fq3_meta_lineage": True,
    })
    lineage_path.write_text(direct_lineage.to_json(), encoding="utf-8")
    reproducibility = json.loads(reproducibility_path.read_text())
    reproducibility["input_lineage_contract_sha256"] = direct_lineage.contract_sha256
    reproducibility_path.write_text(json.dumps(reproducibility), encoding="utf-8")
    # The default legacy base alias cannot silently become the Stage-III
    # upstream score merely because a direct FQ3 correction is present.
    with pytest.raises(StageIIISequentialRunnerError, match="canonical joint expected-net column"):
        cli.main([
            "--stage-ii-winner-bundle", str(winner), "--stage-ii-oos-bundle", str(oos),
            "--enriched-ledger", str(ledger), "--input-lineage", str(lineage_path),
            "--feature-groups", str(groups), "--runner-config", str(config),
            "--reproducibility", str(reproducibility_path), "--preflight",
        ])
    config_value = json.loads(config.read_text())
    config_value["base_expected_net_column"] = "prequential_joint_expected_net_bps"
    config.write_text(json.dumps(config_value), encoding="utf-8")
    assert cli.main([
        "--stage-ii-winner-bundle", str(winner), "--stage-ii-oos-bundle", str(oos),
        "--enriched-ledger", str(ledger), "--input-lineage", str(lineage_path),
        "--feature-groups", str(groups), "--runner-config", str(config),
        "--reproducibility", str(reproducibility_path), "--preflight",
    ]) == 0
    broken = pd.read_parquet(ledger)
    broken.loc[0, "meta_score_semantics"] = "premapped_bps_residual"
    broken.to_parquet(ledger, index=False)
    with pytest.raises(StageIIISequentialRunnerError, match="direct FQ3 correction semantics"):
        cli.main([
            "--stage-ii-winner-bundle", str(winner), "--stage-ii-oos-bundle", str(oos),
            "--enriched-ledger", str(ledger), "--input-lineage", str(lineage_path),
            "--feature-groups", str(groups), "--runner-config", str(config),
            "--reproducibility", str(reproducibility_path), "--preflight",
        ])
