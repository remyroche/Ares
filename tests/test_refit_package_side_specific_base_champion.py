from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import refit_package_side_specific_base_champion as runner


class _Model:
    pass


def _params() -> dict[str, object]:
    return {
        "n_estimators": 5, "learning_rate": 0.1, "num_leaves": 7, "max_depth": 3,
        "min_child_samples": 2, "subsample": 1.0, "colsample_bytree": 1.0,
        "reg_alpha": 0.0, "reg_lambda": 1.0, "loss_function": "regression",
        "min_split_gain": 0.0, "target_mode": "target_soft", "weight_arm": "W0_base",
    }


def _write_fixture(tmp_path: Path) -> Path:
    report = tmp_path / "report"
    fold = report / "_fold_cache" / "old"
    fold.mkdir(parents=True)
    state_dir = report / "state"
    state_dir.mkdir()
    state_path = state_dir / "cycle__global_state.pkl"
    state = {"feature_columns": ["raw_a", "raw_b"], "input_feature_order_hash": runner._feature_contract_hash(["raw_a", "raw_b"])}
    joblib.dump(state, state_path)
    (state_dir / "cycle__global_manifest.json").write_text(json.dumps({"input_feature_columns": ["raw_a", "raw_b"], "input_feature_order_hash": runner._feature_contract_hash(["raw_a", "raw_b"])}))
    sidecar = report / "sidecar.parquet"
    sidecar.touch()
    (report / "sidecar.manifest.json").write_text(json.dumps({"state_sha256": runner._sha256_file(state_path), "output_feature_hash": runner._feature_contract_hash(["dae_b16_06", "gmm_ood_score"]), "output_features": ["dae_b16_06", "gmm_ood_score"]}))
    params = _params()
    (report / "params.json").write_text(json.dumps({"params": params}))
    (report / "best.json").write_text(json.dumps(params))
    features = {"long": ["f1", "dae_b16_06"], "short": ["f2", "gmm_ood_score"]}
    x_train = pd.DataFrame({"dae_b16_06": [1.0, 2.0, 3.0, 4.0], "f1": [4.0, 3.0, 2.0, 1.0], "f2": [0.1, 0.2, 0.3, 0.4], "gmm_ood_score": [0.4, 0.3, 0.2, 0.1]})
    x_valid = x_train.iloc[:2].add(10.0)
    x_train.to_parquet(fold / "x_train.parquet", index=False)
    x_valid.to_parquet(fold / "x_valid.parquet", index=False)
    pd.DataFrame({"target_soft": [0.1, 0.2, 0.3, 0.4]}).to_parquet(fold / "train_target.parquet", index=False)
    pd.DataFrame({"sample_weight": [1.0, 1.1, 1.2, 1.3]}).to_parquet(fold / "train_weight.parquet", index=False)
    pd.DataFrame({"side_name": ["long", "short", "long", "short"]}).to_parquet(fold / "train_side.parquet", index=False)
    train = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                [
                    "2025-07-11T00:00:00Z",
                    "2025-07-12T00:00:00Z",
                    "2025-07-13T00:00:00Z",
                    "2025-07-14T00:00:00Z",
                ]
            )
        }
    )
    train.to_parquet(fold / "train.parquet", index=False)
    valid = pd.DataFrame({"side_name": ["long", "short"], "__symbol__": ["L", "S"], "__ts__": pd.to_datetime(["2026-07-10T00:00:00Z", "2026-07-10T01:00:00Z"]), "__first_touch_target_soft__": [0.8, 0.7], "__u_policy_net__": [0.01, 0.02], "__first_touch_policy_soft__": [0.8, 0.7]})
    valid.to_parquet(fold / "valid.parquet", index=False)
    pd.DataFrame({"u_policy_net": [0.01, 0.02], "mae_norm": [0.2, 0.3], "mfe_norm": [1.1, 1.2], "bars_to_mfe": [2.0, 3.0], "barrier": [0.01, 0.01], "is_timeout": [False, False], "first_touch_net": [0.01, 0.02], "clean_first_touch_exec": [1.0, 1.0], "first_touch_hit": [1.0, 1.0], "first_touch_stop": [0.0, 0.0], "first_touch_timeout": [0.0, 0.0], "first_touch_same_bar": [0.0, 0.0], "first_touch_mae_to_sl": [0.0, 0.0], "first_touch_bar": [1.0, 1.0]}).to_parquet(fold / "valid_metrics.parquet", index=False)
    paths = {name.removesuffix(".parquet"): name for name in ("x_train.parquet", "x_valid.parquet", "train_target.parquet", "train_weight.parquet", "train_side.parquet", "train.parquet", "valid.parquet", "valid_metrics.parquet")}
    (fold / "fold_manifest.json").write_text(json.dumps({"fold": "latest", "valid_end": "2026-07-30T00:00:00Z", "fixed_training_target_mode": params["target_mode"], "fixed_training_weight_arm": params["weight_arm"], "selected_features_by_side": features, "payload_paths": paths}))
    manifest = {"model_side_scope": "per_side", "train_window_days": 365, "compact_fixed_training_payload": True, "fold_cache_dir": "_fold_cache", "selected_features_by_side": features, "fixed_params_json": "params.json", "outputs": {"best_params": "best.json"}, "fixed_ae_gmm_state_pkl": "state/cycle__global_state.pkl", "ae_gmm_state_reference_state_path": "state/cycle__global_state.pkl", "ae_gmm_input_features": ["raw_a", "raw_b"], "frozen_ae_gmm_output_sidecar_path": "sidecar.parquet"}
    (report / "manifest.json").write_text(json.dumps(manifest))
    return report


def test_refit_packages_per_side_models_and_marks_non_oos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = _write_fixture(tmp_path)
    seen: list[tuple[list[str], int]] = []

    def fake_fit(*, matrix, target, weight, params, seed):
        seen.append((list(matrix.columns), len(matrix)))
        assert len(target) == len(weight) == len(matrix)
        return _Model()

    monkeypatch.setattr(runner, "_fit_side_model", fake_fit)
    output = tmp_path / "package"
    manifest = runner.run_final_refit(report_dir=report, output_dir=output, seed=11)
    assert seen == [(["f1", "dae_b16_06"], 3), (["f2", "gmm_ood_score"], 3)]
    assert manifest["excluded_from_oos_metrics"] is True
    assert manifest["all_permitted_labelled_rows_refit"] is True
    assert manifest["permitted_row_accounting"]["permitted_rows"] == 6
    assert manifest["permitted_row_accounting"]["train_window_days"] == 365
    assert manifest["permitted_row_accounting"]["excluded_outside_trailing_window_rows"] == 0
    assert (output / "ae_gmm_state/ae_gmm_state.pkl").is_file()
    for side in ("long", "short"):
        metadata = json.loads((output / side / "metadata.json").read_text())
        assert metadata["leakage_contract"]["excluded_from_oos_metrics"] is True
        assert metadata["feature_contract_hash"] == runner._feature_contract_hash(manifest["selected_features_by_side"][side])


def test_refit_trims_appended_rows_to_exact_trailing_365_days(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _write_fixture(tmp_path)
    train_path = report / "_fold_cache/old/train.parquet"
    train = pd.read_parquet(train_path)
    train.loc[0, "__ts__"] = pd.Timestamp("2025-07-01T00:00:00Z")
    train.to_parquet(train_path, index=False)
    seen: list[tuple[str, int]] = []

    def fake_fit(*, matrix, target, weight, params, seed):
        seen.append((str(matrix.columns[0]), len(matrix)))
        return _Model()

    monkeypatch.setattr(runner, "_fit_side_model", fake_fit)
    manifest = runner.run_final_refit(
        report_dir=report,
        output_dir=tmp_path / "package",
        seed=11,
    )

    assert manifest["permitted_row_accounting"]["permitted_rows"] == 5
    assert manifest["permitted_row_accounting"]["excluded_outside_trailing_window_rows"] == 1
    assert seen == [("f1", 2), ("f2", 3)]


def _compact_recovery_args(report: Path, labels: Path) -> dict[str, Path]:
    return {
        "labels_path": labels,
        "fixed_params_json": report / "params.json",
        "fixed_ae_gmm_state_pkl": report / "state/cycle__global_state.pkl",
        "frozen_ae_gmm_output_sidecar_path": report / "sidecar.parquet",
    }


def _write_recovery_sidecar(report: Path, labels: Path) -> None:
    frame = pd.read_parquet(labels / "labels.parquet")
    pd.DataFrame(
        {
            "__ts__": frame["__ts__"],
            "__symbol__": frame["__symbol__"],
            "side": np.where(frame["side_name"].eq("short"), -1, 1).astype(np.int8),
            "dae_b16_06": 1.0,
            "gmm_ood_score": 1.0,
        }
    ).to_parquet(report / "sidecar.parquet", index=False)


def test_compact_recovery_packages_without_report_manifest_or_train_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _write_fixture(tmp_path)
    params_path = report / "params.json"
    params_payload = json.loads(params_path.read_text())
    params_payload["params"]["weight_arm"] = "W7_timestamp_balanced"
    params_path.write_text(json.dumps(params_payload))
    (report / "best.json").write_text(json.dumps(params_payload["params"]))
    fold_dir = report / "_fold_cache/old"
    fold_manifest_path = fold_dir / "fold_manifest.json"
    fold_manifest = json.loads(fold_manifest_path.read_text())
    fold_manifest["fixed_training_weight_arm"] = "W7_timestamp_balanced"
    fold_manifest["compact_fixed_training_payload"] = True
    fold_manifest["payload_train_sampling"] = "full_train_rows"
    fold_manifest["train_start"] = "2025-07-10T00:00:00Z"
    fold_manifest["valid_start"] = "2026-07-10T00:00:00Z"
    fold_manifest["payload_paths"].pop("train")
    fold_manifest_path.write_text(json.dumps(fold_manifest))
    (fold_dir / "train.parquet").unlink()
    (report / "manifest.json").unlink()

    labels = tmp_path / "labels"
    labels.mkdir()
    pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                [
                    "2025-07-01T00:00:00Z",
                    "2025-07-12T00:00:00Z",
                    "2025-07-13T00:00:00Z",
                    "2025-07-14T00:00:00Z",
                    "2026-07-10T00:00:00Z",
                    "2026-07-10T01:00:00Z",
                ]
            ),
            "__symbol__": ["old", "L1", "S1", "L2", "L", "S"],
            "side_name": ["long", "long", "short", "long", "long", "short"],
            "__first_touch_target_soft__": [9.0, 0.2, 0.3, 0.4, 0.8, 0.7],
        }
    ).to_parquet(labels / "labels.parquet", index=False)

    # The compact train payload is exactly the latter three train identity rows.
    train_target = pd.read_parquet(fold_dir / "train_target.parquet")
    train_target["target_soft"] = [0.2, 0.3, 0.4, 0.0]
    train_target.to_parquet(fold_dir / "train_target.parquet", index=False)
    train_side = pd.read_parquet(fold_dir / "train_side.parquet")
    train_side["side_name"] = ["long", "short", "long", "short"]
    train_side.to_parquet(fold_dir / "train_side.parquet", index=False)
    # Make the fourth cached row the old row so it is deterministically trimmed.
    labels_frame = pd.read_parquet(labels / "labels.parquet")
    labels_frame.loc[0, "__first_touch_target_soft__"] = 0.0
    labels_frame.loc[0, "side_name"] = "short"
    labels_frame.to_parquet(labels / "labels.parquet", index=False)
    labels_frame = pd.read_parquet(labels / "labels.parquet")
    labels_frame.loc[0, "__ts__"] = pd.Timestamp("2025-07-15T00:00:00Z")
    labels_frame.loc[0, "__symbol__"] = "S2"
    labels_frame = pd.concat(
        [
            labels_frame,
            pd.DataFrame(
                {
                    "__ts__": [pd.Timestamp("2026-07-09T23:00:00Z")],
                    "__symbol__": ["gap"],
                    "side_name": ["long"],
                    "__first_touch_target_soft__": [0.8],
                }
            ),
        ],
        ignore_index=True,
    )
    labels_frame.to_parquet(labels / "labels.parquet", index=False)
    _write_recovery_sidecar(report, labels)

    # The latest compact train cache purges the final pre-validation row.  The
    # preceding fold's validation cache is the authoritative compact recovery
    # source for that one row.
    previous = report / "_fold_cache/previous"
    previous.mkdir()
    current_valid = pd.read_parquet(fold_dir / "valid.parquet").iloc[[0]].copy()
    current_valid.loc[:, "__ts__"] = pd.Timestamp("2026-07-09T23:00:00Z")
    current_valid.loc[:, "__symbol__"] = "gap"
    current_valid.loc[:, "side_name"] = "long"
    current_valid.loc[:, "__first_touch_target_soft__"] = 0.8
    current_valid.to_parquet(previous / "valid.parquet", index=False)
    pd.read_parquet(fold_dir / "valid_metrics.parquet").iloc[[0]].to_parquet(
        previous / "valid_metrics.parquet", index=False
    )
    pd.read_parquet(fold_dir / "x_train.parquet").iloc[[0]].to_parquet(
        previous / "x_valid.parquet", index=False
    )
    (previous / "fold_manifest.json").write_text(
        json.dumps(
            {
                "fold": "previous",
                "valid_end": "2026-07-10T00:00:00Z",
                "payload_paths": {
                    "valid": "valid.parquet",
                    "valid_metrics": "valid_metrics.parquet",
                    "x_valid": "x_valid.parquet",
                },
            }
        )
    )

    seen: list[tuple[str, int]] = []

    def fake_fit(*, matrix, target, weight, params, seed):
        seen.append((str(matrix.columns[0]), len(matrix)))
        return _Model()

    monkeypatch.setattr(runner, "_fit_side_model", fake_fit)
    manifest = runner.run_final_refit(
        report_dir=report,
        output_dir=tmp_path / "package",
        seed=11,
        compact_recovery=_compact_recovery_args(report, labels),
    )

    assert manifest["compact_recovery"]["schema"] == "compact_fixed_base_refit_recovery_v1"
    assert manifest["permitted_row_accounting"]["compact_recovery_identity_validation"]["train"]["rows"] == 4
    assert manifest["permitted_row_accounting"]["permitted_rows"] == 7
    assert manifest["permitted_row_accounting"]["recovered_purged_gap_rows"] == 1
    assert manifest["permitted_row_accounting"]["compact_recovery_identity_validation"]["purged_gap"]["source_folds"] == ["previous"]
    assert seen == [("f1", 4), ("f2", 3)]


def test_compact_recovery_fails_closed_when_labels_do_not_match_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _write_fixture(tmp_path)
    fold_dir = report / "_fold_cache/old"
    fold_manifest_path = fold_dir / "fold_manifest.json"
    fold_manifest = json.loads(fold_manifest_path.read_text())
    fold_manifest["payload_paths"].pop("train")
    fold_manifest["compact_fixed_training_payload"] = True
    fold_manifest["payload_train_sampling"] = "full_train_rows"
    fold_manifest["train_start"] = "2025-07-10T00:00:00Z"
    fold_manifest["valid_start"] = "2026-07-10T00:00:00Z"
    fold_manifest_path.write_text(json.dumps(fold_manifest))
    (fold_dir / "train.parquet").unlink()
    (report / "manifest.json").unlink()
    params = json.loads((report / "params.json").read_text())
    params["params"]["weight_arm"] = "W7_timestamp_balanced"
    (report / "params.json").write_text(json.dumps(params))
    (report / "best.json").write_text(json.dumps(params["params"]))
    fold_manifest["fixed_training_weight_arm"] = "W7_timestamp_balanced"
    fold_manifest_path.write_text(json.dumps(fold_manifest))
    labels = tmp_path / "labels"
    labels.mkdir()
    pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2025-07-11T00:00:00Z"]),
            "__symbol__": ["only"],
            "side_name": ["long"],
            "__first_touch_target_soft__": [0.9],
        }
    ).to_parquet(labels / "labels.parquet", index=False)
    _write_recovery_sidecar(report, labels)
    monkeypatch.setattr(runner, "_fit_side_model", lambda **_: _Model())
    with pytest.raises(ValueError, match="Compact recovery has fewer immutable label identities"):
        runner.run_final_refit(
            report_dir=report,
            output_dir=tmp_path / "bad-package",
            compact_recovery=_compact_recovery_args(report, labels),
        )


def test_refit_fails_closed_on_state_feature_and_target_contract_mismatches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = _write_fixture(tmp_path)
    monkeypatch.setattr(runner, "_fit_side_model", lambda **_: _Model())
    sidecar = report / "sidecar.manifest.json"
    state = json.loads(sidecar.read_text())
    state["state_sha256"] = "bad"
    sidecar.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="Serialized AE/GMM state hash mismatch"):
        runner.run_final_refit(report_dir=report, output_dir=tmp_path / "bad-state")
    _write_fixture(tmp_path / "feature")
    feature_report = tmp_path / "feature/report"
    fold = feature_report / "_fold_cache/old/fold_manifest.json"
    payload = json.loads(fold.read_text())
    payload["selected_features_by_side"]["long"] = ["f2"]
    fold.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="long feature contract hash mismatch"):
        runner.run_final_refit(report_dir=feature_report, output_dir=tmp_path / "bad-feature")
    _write_fixture(tmp_path / "target")
    target_report = tmp_path / "target/report"
    fold = target_report / "_fold_cache/old/fold_manifest.json"
    payload = json.loads(fold.read_text())
    payload["fixed_training_target_mode"] = "policy_soft"
    fold.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Target contract hash mismatch"):
        runner.run_final_refit(report_dir=target_report, output_dir=tmp_path / "bad-target")
