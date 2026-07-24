from __future__ import annotations

from scripts.package_s52_base_oos_checkpoint import (
    BASE_INPUT_NUMERIC_CONTRACT,
    _ae_gmm_source_manifest_path,
    _validate_ae_gmm_source_contract,
    _install_base_checkpoint,
    _install_meta_checkpoint,
    _install_meta_reliability_priors,
    _install_trained_symbol_summary,
    _install_v9_predecessor,
    _physical_feature_store_run_id,
    _rebind_local_policy_artifact_paths,
    _write_meta_feature_contract,
)


class _FrozenPredecessor:
    def required_input_features(self) -> list[str]:
        return ["market_state", "base_score"]

    def predict(self, frame):
        return frame


def test_physical_feature_store_run_id_strips_descriptive_alias() -> None:
    assert _physical_feature_store_run_id(
        "20260711_070000_shared_static_feature_store"
    ) == "20260711_070000"
    assert _physical_feature_store_run_id("20260711_070000") == "20260711_070000"


def test_install_trained_symbol_summary_uses_base_oos_universe(tmp_path) -> None:
    import pandas as pd

    ledger = tmp_path / "best_oos_scored_ledger.parquet"
    pd.DataFrame(
        {"__symbol__": ["ETH/USD:USD", "BTC/USD:USD", "ETH/USD:USD"]}
    ).to_parquet(ledger, index=False)
    bundle = tmp_path / "bundle"

    result = _install_trained_symbol_summary(
        bundle, base_oos_ledger=ledger
    )

    assert result is not None
    assert result["symbol_count"] == 2
    installed = pd.read_csv(
        bundle / "features/feature_health_symbol_summary.csv"
    )
    assert installed["symbol"].tolist() == ["BTC/USD:USD", "ETH/USD:USD"]


def test_ae_gmm_source_manifest_path_resolves_cycle_state_manifest(tmp_path) -> None:
    state = tmp_path / "cycle__global_state.pkl"
    manifest = tmp_path / "cycle__global_manifest.json"
    state.write_bytes(b"state")
    manifest.write_text("{}")
    assert _ae_gmm_source_manifest_path(state) == manifest


def test_validate_ae_gmm_source_contract_accepts_bare_hex_hashes(
    tmp_path, monkeypatch
) -> None:
    from extreme_price_movements.feature_transform_contract import ordered_names_hash
    from extreme_price_movements.features_gmm_ae import ae_gmm_learned_transform_hash
    import scripts.package_s52_base_oos_checkpoint as packager

    path = tmp_path / "cycle__global_state.pkl"
    path.write_bytes(b"state")
    state = {
        "enabled": True,
        "feature_columns": ["a", "b"],
        "center": [0.0, 0.0],
        "scale": [1.0, 1.0],
    }
    monkeypatch.setattr(packager, "load_ae_gmm_state_artifact", lambda _: state)
    transform_hash = ae_gmm_learned_transform_hash(state)
    result = _validate_ae_gmm_source_contract(
        path,
        {
            "input_feature_order_hash": ordered_names_hash(
                state["feature_columns"]
            ).removeprefix("sha256:"),
            "learned_transform_hash": transform_hash,
            "cycle_state_hash": transform_hash,
            "materialized_transform_rules": {"clip": [-8.0, 8.0]},
        },
    )
    assert result["input_feature_count"] == 2


def test_validate_ae_gmm_source_contract_rejects_transform_mismatch(
    tmp_path, monkeypatch
) -> None:
    import pytest
    import scripts.package_s52_base_oos_checkpoint as packager

    path = tmp_path / "cycle__global_state.pkl"
    path.write_bytes(b"state")
    state = {
        "enabled": True,
        "feature_columns": ["a", "b"],
        "center": [0.0, 0.0],
        "scale": [1.0, 1.0],
    }
    monkeypatch.setattr(packager, "load_ae_gmm_state_artifact", lambda _: state)
    from extreme_price_movements.feature_transform_contract import ordered_names_hash

    with pytest.raises(ValueError, match="learned transform"):
        _validate_ae_gmm_source_contract(
            path,
            {
                "input_feature_order_hash": ordered_names_hash(
                    state["feature_columns"]
                ),
                "learned_transform_hash": "wrong",
                "materialized_transform_rules": {"clip": [-8.0, 8.0]},
            },
        )


def test_install_base_checkpoint_replaces_both_side_and_horizon_models() -> None:
    class Model:
        pass

    old = object()
    new = Model()
    state = {
        "bundle": {
            "alpha_models": {
                side: {
                    "model": old,
                    "feat_cols": ["old"],
                    "H": 5,
                    "models_by_h": {5: {"model": old, "feat_cols": ["old"]}},
                }
                for side in ("long_key", "short_key")
            }
        }
    }
    _install_base_checkpoint(state, model=new, feature_names=["side", "signal"])
    for side_state in state["bundle"]["alpha_models"].values():
        assert side_state["model"] is new
        assert side_state["feat_cols"] == ["side", "signal"]
        assert side_state["models_by_h"][5]["model"] is new
        assert side_state["models_by_h"][5]["feat_cols"] == ["side", "signal"]
        assert side_state["input_numeric_contract"] == BASE_INPUT_NUMERIC_CONTRACT
        assert (
            side_state["models_by_h"][5]["input_numeric_contract"]
            == BASE_INPUT_NUMERIC_CONTRACT
        )
        assert side_state["input_numeric_contract_payload"]["name"] == (
            BASE_INPUT_NUMERIC_CONTRACT
        )
        assert side_state["models_by_h"][5]["input_numeric_contract_payload"] == (
            side_state["input_numeric_contract_payload"]
        )
    assert new.epm_input_numeric_contract_ == BASE_INPUT_NUMERIC_CONTRACT
    assert new.epm_input_numeric_contract_payload_["name"] == (
        BASE_INPUT_NUMERIC_CONTRACT
    )


def test_install_meta_checkpoint_uses_oos_domain_without_refit_alignment() -> None:
    class Model:
        pass

    old = Model()
    old.s52_meta_ood_reference_ = {"enabled": True}
    old.s52_meta_score_alignment_ = {"enabled": True}
    new = Model()
    state = {
        "bundle": {
            "meta_models": {"long_key": old, "short_key": old},
        }
    }
    installed = _install_meta_checkpoint(
        state, model=new, feature_names=["score_base", "score"]
    )
    assert installed["long_key"] is new
    assert installed["short_key"] is new
    assert new.feature_columns == ["score_base", "score"]
    assert new.s52_meta_ood_reference_ == {"enabled": True}
    assert not hasattr(new, "s52_meta_score_alignment_")


def test_write_meta_feature_contract_includes_score_base(tmp_path) -> None:
    _write_meta_feature_contract(
        tmp_path, model_keys=["long_key", "short_key"], feature_names=["score_base", "score"]
    )
    import json

    payload = json.loads((tmp_path / "meta_oof/meta_feature_contract.json").read_text())
    assert payload["meta_models"]["long_key"]["feature_columns"] == [
        "score_base",
        "score",
    ]


def test_install_meta_reliability_priors_requires_pre_oos_cutoff(tmp_path) -> None:
    import json

    source = tmp_path / "priors.json"
    source.write_text(
        json.dumps(
            {
                "rows": 123,
                "groups": {"long|a|1|1": {}},
                "global_stats": {"clean_rate": 0.5},
                "exact_groups_only": True,
                "source": {"train_end_exclusive": "2026-07-01T00:00:00Z"},
            }
        )
    )
    output = tmp_path / "bundle"
    result = _install_meta_reliability_priors(
        output,
        source=source,
        valid_start="2026-07-01T00:00:00Z",
    )
    assert result["rows"] == 123
    assert result["train_end_exclusive"] == "2026-07-01T00:00:00Z"
    installed = output / "policy_params/meta_reliability_priors.json"
    assert json.loads(installed.read_text())["rows"] == 123


def test_install_meta_reliability_priors_rejects_future_rows(tmp_path) -> None:
    import json
    import pytest

    source = tmp_path / "priors.json"
    source.write_text(
        json.dumps(
            {
                "rows": 123,
                "global_stats": {"clean_rate": 0.5},
                "source": {"train_end_exclusive": "2026-07-02T00:00:00Z"},
            }
        )
    )
    with pytest.raises(ValueError, match="extend into the OOS checkpoint"):
        _install_meta_reliability_priors(
            tmp_path / "bundle",
            source=source,
            valid_start="2026-07-01T00:00:00Z",
        )


def test_install_v9_predecessor_records_exact_model_contract(tmp_path) -> None:
    import joblib

    source = tmp_path / "historical_v9.joblib"
    joblib.dump(_FrozenPredecessor(), source)
    bundle = tmp_path / "bundle"
    result = _install_v9_predecessor(bundle, source=source)
    installed = bundle / "policy_params/v9_tail95_predecessor_bundle.joblib"
    assert installed.is_file()
    assert result["required_input_feature_count"] == 2
    assert result["class"].endswith("._FrozenPredecessor")
    assert result["contract"] == "exact_frozen_historical_policy_predecessor_v1"


def test_rebind_local_policy_artifact_paths_replaces_runtime_keys(tmp_path) -> None:
    import json

    policy = tmp_path / "policy_params"
    policy.mkdir()
    for name in (
        "composite_policy_regime_ev_calibration.json",
        "v9_tail95_predecessor_bundle.joblib",
        "residual_event_state.joblib",
        "threshold_basis_policy_sidearch_ev70_trim10_21d.json",
    ):
        (policy / name).write_text("{}")
    config = tmp_path / "runtime.json"
    config.write_text(
        json.dumps(
            {
                "predecessor_bundle_path": "/stale/v9.joblib",
                "nested": {
                    "threshold_basis_policy_path": "/stale/threshold.json",
                    "source_threshold_basis_policy": "/provenance/unchanged.json",
                },
            }
        )
    )
    updated = _rebind_local_policy_artifact_paths(tmp_path)
    payload = json.loads(config.read_text())
    assert str(config) in updated
    assert payload["predecessor_bundle_path"] == str(
        (policy / "v9_tail95_predecessor_bundle.joblib").resolve()
    )
    assert payload["nested"]["threshold_basis_policy_path"] == str(
        (policy / "threshold_basis_policy_sidearch_ev70_trim10_21d.json").resolve()
    )
    assert payload["nested"]["source_threshold_basis_policy"] == (
        "/provenance/unchanged.json"
    )
