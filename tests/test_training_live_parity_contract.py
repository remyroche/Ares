import pytest
import pandas as pd

from extreme_price_movements.inference.run_inference import (
    _resolve_live_feature_source_run_id,
)
from extreme_price_movements.inference.training_live_parity_contract import (
    build_training_live_parity_contract,
    parity_contract_output_paths,
    validate_training_live_parity_contract,
)


def _write_runtime_artifacts(run_root):
    (run_root / "models" / "native" / "demo_H5").mkdir(parents=True)
    (run_root / "models" / "native" / "demo_H5" / "model.joblib").write_bytes(
        b"native-model"
    )
    (run_root / "models" / "native" / "demo_H5" / "sidecar.pkl").write_bytes(
        b"native-sidecar"
    )
    (run_root / "models" / "trained_state.pkl").write_bytes(b"trained")
    (run_root / "models" / "model_state_meta.pkl").write_bytes(b"meta")
    (run_root / "base_models_intermediate.pkl").write_bytes(b"base")
    (run_root / "base_meta_contract.json").write_text("{}", encoding="utf-8")
    (run_root / "meta_oof").mkdir()
    (run_root / "meta_oof" / "meta_feature_contract.json").write_text(
        "{}", encoding="utf-8"
    )
    (run_root / "simple_policy_optimiser" / "deployment").mkdir(parents=True)
    (
        run_root
        / "simple_policy_optimiser"
        / "deployment"
        / "best_policy_params.json"
    ).write_text("{}", encoding="utf-8")
    (run_root / "simple_policy_optimiser" / "rank_reference").mkdir(parents=True)
    (
        run_root
        / "simple_policy_optimiser"
        / "rank_reference"
        / "manifest.json"
    ).write_text("{}", encoding="utf-8")
    (
        run_root
        / "simple_policy_optimiser"
        / "rank_reference"
        / "cross_strategy_auction.parquet"
    ).write_bytes(b"auction")
    (run_root / "policy_params").mkdir()
    (run_root / "policy_params" / "optimized_portfolio_policy_config.json").write_text(
        "{}", encoding="utf-8"
    )


def test_training_live_parity_contract_records_base_and_native_artifacts(tmp_path):
    run_root = tmp_path / "artifacts" / "run_a"
    run_root.mkdir(parents=True)
    _write_runtime_artifacts(run_root)

    contract = build_training_live_parity_contract(
        data_root=str(tmp_path),
        run_id="run_a",
        market_mode="perps",
        model_bundle={"bundle": {}},
        strategy_ids=["long_demo"],
    )

    artifact_hashes = contract["artifact_hashes"]
    assert artifact_hashes["base_models_intermediate"]["exists"] is True
    assert artifact_hashes["base_models_intermediate"]["artifact_type"] == "file"
    assert artifact_hashes["native_model_dir"]["exists"] is True
    assert artifact_hashes["native_model_dir"]["artifact_type"] == "directory_tree"
    assert validate_training_live_parity_contract(
        contract,
        active_strategy_ids=["long_demo"],
        data_root=str(tmp_path),
        run_id="run_a",
    )


def test_training_live_parity_contract_fails_on_native_artifact_drift(tmp_path):
    run_root = tmp_path / "artifacts" / "run_a"
    run_root.mkdir(parents=True)
    _write_runtime_artifacts(run_root)
    contract = build_training_live_parity_contract(
        data_root=str(tmp_path),
        run_id="run_a",
        market_mode="perps",
        model_bundle={"bundle": {}},
        strategy_ids=["long_demo"],
    )

    (run_root / "models" / "native" / "demo_H5" / "model.joblib").write_bytes(
        b"native-model-updated"
    )

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        validate_training_live_parity_contract(
            contract,
            active_strategy_ids=["long_demo"],
            data_root=str(tmp_path),
            run_id="run_a",
        )


def test_training_live_parity_contract_requires_policy_oos_artifact_entries(tmp_path):
    contract = {
        "strategy_contract": {"strategy_ids": ["long_demo"]},
        "artifact_hashes": {
            "trained_state": {"exists": True, "sha256": "abc"},
            "meta_state": {"exists": True, "sha256": "def"},
        },
    }

    with pytest.raises(ValueError, match="missing required hash entries"):
        validate_training_live_parity_contract(
            contract,
            active_strategy_ids=["long_demo"],
            data_root=str(tmp_path),
            run_id="run_a",
        )


def test_training_live_parity_contract_rejects_rank_reference_outside_trained_universe(
    tmp_path,
    monkeypatch,
):
    run_root = tmp_path / "artifacts" / "run_a"
    run_root.mkdir(parents=True)
    _write_runtime_artifacts(run_root)
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01T00:00:00Z"] * 2),
            "symbol": ["AAA/USD:USD", "ZZZ/USD:USD"],
            "calibrated_score": [0.9, 0.8],
        }
    ).to_parquet(
        run_root
        / "simple_policy_optimiser"
        / "rank_reference"
        / "long_demo.parquet",
        index=False,
    )

    def fake_universe(data_root, run_id):
        return {"AAA/USD:USD"}

    monkeypatch.setattr(
        "extreme_price_movements.inference.config.load_trained_symbol_universe",
        fake_universe,
    )
    contract = build_training_live_parity_contract(
        data_root=str(tmp_path),
        run_id="run_a",
        market_mode="perps",
        model_bundle={"bundle": {}},
        strategy_ids=["long_demo"],
    )

    with pytest.raises(ValueError, match="rank-reference universe mismatch"):
        validate_training_live_parity_contract(
            contract,
            active_strategy_ids=["long_demo"],
            data_root=str(tmp_path),
            run_id="run_a",
        )


def test_training_live_parity_contract_honors_policy_artifact_root_override(
    tmp_path,
    monkeypatch,
):
    run_root = tmp_path / "artifacts" / "run_a"
    run_root.mkdir(parents=True)
    _write_runtime_artifacts(run_root)

    policy_root = tmp_path / "policy_override"
    _write_runtime_artifacts(policy_root)
    (
        policy_root
        / "simple_policy_optimiser"
        / "rank_reference"
        / "manifest.json"
    ).write_text('{"override": true}', encoding="utf-8")
    (
        policy_root
        / "policy_params"
        / "optimized_portfolio_policy_config.json"
    ).write_text('{"override": true}', encoding="utf-8")

    monkeypatch.setenv("EPM_INFERENCE_POLICY_ARTIFACT_ROOT", str(policy_root))
    contract = build_training_live_parity_contract(
        data_root=str(tmp_path),
        run_id="run_a",
        market_mode="perps",
        model_bundle={"bundle": {}},
        strategy_ids=["long_demo"],
    )

    hashes = contract["artifact_hashes"]
    assert hashes["base_models_intermediate"]["path"].endswith(
        "artifacts/run_a/base_models_intermediate.pkl"
    )
    assert hashes["simple_policy_rank_manifest"]["path"].endswith(
        "policy_override/simple_policy_optimiser/rank_reference/manifest.json"
    )
    assert hashes["optimized_portfolio_policy"]["path"].endswith(
        "policy_override/policy_params/optimized_portfolio_policy_config.json"
    )
    assert parity_contract_output_paths(str(tmp_path), "run_a")[0] == (
        policy_root / "policy_params" / "training_live_parity_contract.json"
    )
    assert validate_training_live_parity_contract(
        contract,
        active_strategy_ids=["long_demo"],
        data_root=str(tmp_path),
        run_id="run_a",
    )


def test_training_live_parity_contract_records_feature_source_run_id(
    tmp_path,
    monkeypatch,
):
    run_root = tmp_path / "artifacts" / "run_a"
    run_root.mkdir(parents=True)
    _write_runtime_artifacts(run_root)

    monkeypatch.setenv("EPM_LIVE_FEATURE_SOURCE_RUN_ID", "feature_run_a")
    contract = build_training_live_parity_contract(
        data_root=str(tmp_path),
        run_id="run_a",
        market_mode="perps",
        model_bundle={"bundle": {}},
        strategy_ids=["long_demo"],
    )
    assert contract["feature_source"]["run_id"] == "feature_run_a"

    monkeypatch.delenv("EPM_LIVE_FEATURE_SOURCE_RUN_ID")
    assert (
        _resolve_live_feature_source_run_id(
            {"training_live_parity_contract": contract}
        )
        == "feature_run_a"
    )


def test_training_live_parity_contract_rejects_rank_reference_without_policy_oos_contract(
    tmp_path,
):
    run_root = tmp_path / "artifacts" / "run_a"
    run_root.mkdir(parents=True)
    _write_runtime_artifacts(run_root)
    manifest_path = (
        run_root
        / "simple_policy_optimiser"
        / "rank_reference"
        / "manifest.json"
    )
    manifest_path.write_text(
        '{"schema_version":"policy_rank_reference_v1",'
        '"generated_by":"simple_policy_optimiser",'
        '"strategies":{"long_demo":{"path":"long_demo.parquet",'
        '"score_col":"calibrated_score","rank_col":"rank_pct","n_rows":3}}}',
        encoding="utf-8",
    )
    contract = build_training_live_parity_contract(
        data_root=str(tmp_path),
        run_id="run_a",
        market_mode="perps",
        model_bundle={"bundle": {}},
        strategy_ids=["long_demo"],
    )

    with pytest.raises(ValueError, match="rank-reference provenance mismatch"):
        validate_training_live_parity_contract(
            contract,
            active_strategy_ids=["long_demo"],
            data_root=str(tmp_path),
            run_id="run_a",
        )
