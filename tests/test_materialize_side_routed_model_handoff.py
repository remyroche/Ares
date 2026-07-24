from __future__ import annotations

import json
from pathlib import Path

import joblib

from scripts.materialize_side_routed_model_handoff import materialize


class _Model:
    pass


def _state(path: Path, *, components: int) -> None:
    joblib.dump(
        {
            "enabled": True,
            "feature_columns": ["x"],
            "latent_columns": ["z"],
            "input_feature_order_hash": f"hash-{components}",
            "gmm_n_components": components,
            "gmm_reg_covar": 0.01,
            "ae_fit_rows": 150_000,
            "gmm_fit_rows": 150_000,
        },
        path,
    )


def _expert(path: Path) -> None:
    joblib.dump(
        {
            "schema": "side_base_residual_expert_inference_v2",
            "backbone_score": "base",
            "backbone_score_col": "score_base",
            "feature_contract": {"long": ["score_base"], "short": ["score_base"]},
            "residual_models": {"long": _Model(), "short": _Model()},
            "alpha_by_side": {"long": 0.8, "short": 0.6},
            "round_trip_cost": 0.01,
        },
        path,
    )


def test_side_routed_handoff_keeps_distinct_ae_gmm_states(tmp_path: Path) -> None:
    long_model = tmp_path / "long.joblib"
    joblib.dump(_Model(), long_model)
    long_columns = tmp_path / "long_columns.json"
    long_columns.write_text(json.dumps({"feature_names": ["x"]}))
    long_expert = tmp_path / "long_expert.joblib"
    _expert(long_expert)
    long_state = tmp_path / "long_state.pkl"
    _state(long_state, components=12)

    short_base = tmp_path / "short_base"
    short_base.mkdir()
    joblib.dump({"long": _Model(), "short": _Model()}, short_base / "base_model.joblib")
    (short_base / "columns.json").write_text(
        json.dumps({"feature_names_by_side": {"long": ["a"], "short": ["b"]}})
    )
    short_meta = tmp_path / "short_meta"
    short_meta.mkdir()
    joblib.dump(_Model(), short_meta / "base_soft_label_short.joblib")
    (short_meta / "columns.json").write_text(
        json.dumps({"feature_names_by_model": {"base_soft_label_short": ["m"]}})
    )
    short_expert = tmp_path / "short_expert.joblib"
    _expert(short_expert)
    short_state = tmp_path / "short_state.pkl"
    _state(short_state, components=6)
    policy = tmp_path / "policy"
    policy.mkdir()
    (policy / "optimized_portfolio_policy_config.json").write_text(
        json.dumps(
            {
                "canonical_meta_postprocessor_sides": [],
                "side_residual_expert_enabled": True,
                "side_residual_expert_artifact_path": str(
                    policy / "side_residual_expert.joblib"
                ),
            }
        )
    )
    (policy / "training_live_parity_contract.json").write_text("{}")
    short_manifest = tmp_path / "short_manifest.json"
    short_manifest.write_text(
        json.dumps(
            {
                "base_model_dir": str(short_base),
                "meta_model_dir": str(short_meta),
                "side_residual_expert": str(short_expert),
                "ae_gmm_state": str(short_state),
                "policy_root": str(policy),
            }
        )
    )

    result = materialize(
        output_root=tmp_path / "out",
        long_base_model=long_model,
        long_base_columns=long_columns,
        long_residual_expert=long_expert,
        long_ae_gmm_state=long_state,
        short_bundle_manifest=short_manifest,
    )
    assert result["routes"]["long"]["ae_gmm"]["gmm_n_components"] == 12
    assert result["routes"]["short"]["ae_gmm"]["gmm_n_components"] == 6
    assert result["routing_contract"]["shared_ae_gmm_state"] is False
    assert result["status"]["native_inference_ready"] is False
    assert result["status"]["side_routed_model_bundle_ready"] is True
    assert result["status"]["model_replay_handoff_ready"] is True
    assert result["status"]["full_policy_replay_ready"] is False
    assert (tmp_path / "out" / "models" / "trained_state.pkl").exists()
    assert (tmp_path / "out" / "policy_params" / "side_residual_expert.joblib").exists()
    assert not (
        tmp_path / "out" / "policy_params" / "training_live_parity_contract.json"
    ).exists()
    policy_payload = json.loads(
        (tmp_path / "out" / "policy_params" / "optimized_portfolio_policy_config.json")
        .read_text()
    )
    assert policy_payload["canonical_meta_postprocessor_sides"] == ["short"]
    assert policy_payload["side_residual_expert_artifact_path"] == str(
        (tmp_path / "out" / "policy_params" / "side_residual_expert.joblib").resolve()
    )
    contract = json.loads(
        (tmp_path / "out" / "meta_oof" / "meta_feature_contract.json").read_text()
    )
    long_row = contract["meta_models"]["long_s52_meta_threshold_handoff"]
    short_row = contract["meta_models"]["short_s52_meta_threshold_handoff"]
    assert long_row["positional_feature_mapping"] == {"f0": "base_score_raw"}
    assert short_row["positional_feature_mapping"] == {"f0": "m"}
    assert short_row["n_features"] == 1
