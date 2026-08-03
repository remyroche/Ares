from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import materialize_frozen_entry_action_handoff as handoff


def _seal(root: Path, schema: str, outputs: dict[str, pd.DataFrame | dict]) -> Path:
    root.mkdir()
    hashes = {}
    for name, value in outputs.items():
        path = root / name
        if isinstance(value, pd.DataFrame):
            value.to_parquet(path, index=False)
        else:
            path.write_text(json.dumps(value))
        hashes[name] = handoff.sha256(path)
    manifest = {"schema": schema, "outputs_sha256": hashes}
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "manifest.sha256").write_text(
        f"{handoff.sha256(root / 'manifest.json')}  manifest.json\n"
    )
    return root


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    ts = pd.date_range("2025-03-22", periods=2, freq="h", tz="UTC")
    identity = pd.DataFrame(
        {
            "candidate_id": ["a", "b"],
            "side_name": ["long", "short"],
            "__symbol__": ["BTC_USD:USD", "ETH_USD:USD"],
            "__ts__": ts,
        }
    )
    roles = {
        "default_ev_inputs": [
            "raw_score",
            "score_residual_expected_ev",
            "base_rank_pct_timestamp_global",
        ]
    }
    panel = identity.copy()
    panel["execution_decision_utc"] = ts
    panel["execution_label_end_utc"] = ts + pd.Timedelta(hours=12)
    panel["candidate_month"] = "2025-03"
    panel["score_available_utc"] = ts
    panel["fold_train_cutoff_utc"] = ts - pd.Timedelta(days=1)
    panel["training_label_resolved_max_utc"] = ts - pd.Timedelta(hours=1)
    panel["residual_fold_x"] = "fold"
    panel["residual_is_oof"] = True
    panel["upstream_scores_are_outer_oof"] = True
    panel["candidate_score_is_oof"] = True
    panel["candidate_score_is_forward_oos"] = False
    panel["support_fold"] = "support-fold"
    panel["support_train_decision_max_utc"] = ts - pd.Timedelta(days=1)
    panel["support_train_label_end_max_utc"] = ts - pd.Timedelta(hours=1)
    panel["support_scores_are_chronological_oof"] = True
    panel["support_scores_are_frozen_forward"] = False
    panel["raw_score"] = [0.2, 0.1]
    panel["score_residual_expected_ev"] = [0.01, -0.01]
    panel["base_rank_pct_timestamp_global"] = [1.0, 0.5]
    panel_root = _seal(
        tmp_path / "panel",
        "canonical_execution_reliability_input_v4",
        {"panel.parquet": panel, "feature_roles.json": roles},
    )

    targets = identity.copy()
    targets["canonical_cost_return"] = 0.01
    targets["target_fixed_12h_net_return"] = [0.02, -0.03]
    target_root = _seal(
        tmp_path / "targets",
        "execution_action_target_pack_v2",
        {"labels.parquet": targets},
    )

    paths = identity.copy()
    paths["__symbol__"] = ["BTC/USD:USD", "ETH/USD:USD"]
    paths["execution_future_path"] = ["path-a", "path-b"]
    paths["atr_1h"] = 1.0
    paths["decision_price"] = 100.0
    paths["fee"] = 0.001
    paths["entry_spread"] = 10.0
    paths["exit_spread"] = 10.0
    path_root = _seal(
        tmp_path / "paths",
        "execution_entry_timing_1m_paths_v1",
        {"paths.parquet": paths},
    )

    book = identity.copy()
    book["execution_decision_utc"] = ts
    book["candidate_month"] = "2025-03"
    book["mapped_score"] = [0.9, 0.8]
    book["mapped_eligible"] = True
    book["gross__deployed"] = [0.03, -0.02]
    book["net__deployed"] = [0.02, -0.03]
    book["cost__deployed"] = 0.01
    book["execution_exit_reason"] = ["trailing", "full_stop"]
    for name in handoff.WEIGHTS:
        book[name] = [1.0, 0.5]
    book_root = _seal(
        tmp_path / "book",
        "frozen_exit_state_action_ablation_v4",
        {"paired_candidates.parquet": book},
    )
    policy_root = tmp_path / "policy"
    policy_root.mkdir()
    context = identity.copy()
    context["policy_archetype"] = ["long_raw", "short_raw"]
    context.to_parquet(policy_root / "context.parquet", index=False)
    policy_targets = identity.copy()
    policy_targets["__barrier_pct__"] = 0.01
    policy_targets["__path_auxiliary_atr_fraction__"] = 0.02
    policy_targets.to_parquet(policy_root / "path_targets.parquet", index=False)
    candidates = identity.copy()
    candidates.to_parquet(policy_root / "candidates.parquet", index=False)
    policy_manifest = {
        "schema": "historical_execution_ev_deployed_policy_inputs_v1",
        "outputs": {
            name: {
                "path": name,
                "sha256": handoff.sha256(policy_root / name),
            }
            for name in ("candidates.parquet", "context.parquet", "path_targets.parquet")
        },
    }
    (policy_root / "manifest.json").write_text(json.dumps(policy_manifest))
    return panel_root, target_root, path_root, book_root, policy_root


def test_materializes_exact_normalized_path_join_and_role_boundary(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    output = tmp_path / "output"
    manifest = handoff.materialize(*inputs, output)
    frame = pd.read_parquet(output / "handoff.parquet")
    roles = json.loads((output / "feature_roles.json").read_text())

    assert manifest["rows"] == 2
    assert frame["path_symbol"].tolist() == ["BTC/USD:USD", "ETH/USD:USD"]
    assert frame["weight_top_10"].tolist() == [1.0, 0.5]
    assert set(roles["model_inputs"]) == {
        "raw_score",
        "score_residual_expected_ev",
        "base_rank_pct_timestamp_global",
    }
    assert "execution_future_path" in roles["target_only_never_model_inputs"]
    assert "mapped_score" in roles["selection_only_never_model_inputs"]
    assert "__barrier_pct__" in roles["execution_only_never_model_inputs"]
    assert frame["gross__deployed"].tolist() == [0.03, -0.02]
    assert handoff.verify_seal(output, handoff.SCHEMA)["rows"] == 2


def test_rejects_normalized_symbol_mismatch(tmp_path: Path) -> None:
    panel_root, target_root, path_root, book_root, policy_root = _inputs(tmp_path)
    paths = pd.read_parquet(path_root / "paths.parquet")
    paths.loc[0, "__symbol__"] = "SOL/USD:USD"
    paths.to_parquet(path_root / "paths.parquet", index=False)
    manifest = json.loads((path_root / "manifest.json").read_text())
    manifest["outputs_sha256"]["paths.parquet"] = handoff.sha256(
        path_root / "paths.parquet"
    )
    (path_root / "manifest.json").write_text(json.dumps(manifest))
    (path_root / "manifest.sha256").write_text(
        f"{handoff.sha256(path_root / 'manifest.json')}  manifest.json\n"
    )

    with pytest.raises(handoff.ContractError, match="normalized path symbol"):
        handoff.materialize(
            panel_root,
            target_root,
            path_root,
            book_root,
            policy_root,
            tmp_path / "output",
        )


def test_rejects_future_target_in_authorized_features(tmp_path: Path) -> None:
    panel_root, target_root, path_root, book_root, policy_root = _inputs(tmp_path)
    roles = json.loads((panel_root / "feature_roles.json").read_text())
    roles["default_ev_inputs"].append("target_future_gain")
    (panel_root / "feature_roles.json").write_text(json.dumps(roles))
    manifest = json.loads((panel_root / "manifest.json").read_text())
    manifest["outputs_sha256"]["feature_roles.json"] = handoff.sha256(
        panel_root / "feature_roles.json"
    )
    (panel_root / "manifest.json").write_text(json.dumps(manifest))
    (panel_root / "manifest.sha256").write_text(
        f"{handoff.sha256(panel_root / 'manifest.json')}  manifest.json\n"
    )

    with pytest.raises(handoff.ContractError, match="future/target"):
        handoff.materialize(
            panel_root,
            target_root,
            path_root,
            book_root,
            policy_root,
            tmp_path / "output",
        )


def test_accepts_historical_signed_path_manifest_without_seal(tmp_path: Path) -> None:
    panel_root, target_root, path_root, book_root, policy_root = _inputs(tmp_path)
    (path_root / "manifest.sha256").unlink()
    manifest = json.loads((path_root / "manifest.json").read_text())
    manifest["source_artifact_sha256"] = manifest["outputs_sha256"].pop("paths.parquet")
    manifest.pop("outputs_sha256")
    canonical = {
        key: handoff.safe(value)
        for key, value in manifest.items()
        if key != "prediction_role_manifest_sha256"
    }
    manifest["prediction_role_manifest_sha256"] = handoff.hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (path_root / "manifest.json").write_text(json.dumps(manifest))

    result = handoff.materialize(
        panel_root,
        target_root,
        path_root,
        book_root,
        policy_root,
        tmp_path / "output",
    )
    assert result["rows"] == 2
