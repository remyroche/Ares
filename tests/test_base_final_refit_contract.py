from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import run_materialized_trailing_label_topk_lgbm_hpo as runner


def test_final_refit_uses_exact_trailing_window_plus_latest_labelled_oos_and_is_excluded(
    tmp_path: Path, monkeypatch
) -> None:
    train = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"], utc=True
            )
        }
    )
    valid = pd.DataFrame(
        {"__ts__": pd.to_datetime(["2026-01-03T00:00:00Z"], utc=True)}
    )
    payload = {
        "compact_fixed_training_payload": True,
        "fixed_training_target_mode": "target_soft",
        "fixed_training_weight_arm": "W0_base",
        "x_train": pd.DataFrame({"f1": [1.0, 2.0]}, dtype=np.float32),
        "x_valid": pd.DataFrame({"f1": [3.0]}, dtype=np.float32),
        "train_target": pd.DataFrame({"target_soft": [0.1, 0.2]}),
        "train_weight": pd.DataFrame({"sample_weight": [1.0, 1.0]}),
        "train": train,
        "valid": valid,
        "train_metrics": pd.DataFrame(index=train.index),
        "valid_metrics": pd.DataFrame(index=valid.index),
    }
    monkeypatch.setattr(runner, "_load_fold_payload_keys", lambda *_a, **_k: payload)
    monkeypatch.setattr(
        runner,
        "_target_from_frame",
        lambda *_a, **_k: pd.DataFrame({"target_soft": [0.3]}),
    )
    monkeypatch.setattr(
        runner,
        "_weight_series",
        lambda **_k: pd.Series([1.5], name="sample_weight"),
    )
    captured: dict[str, object] = {}

    def fake_fit(*, x_train, y_train, w_train, params, seed):
        captured.update(
            rows=len(x_train),
            target=y_train.tolist(),
            weight=w_train.tolist(),
            seed=seed,
        )
        return object()

    monkeypatch.setattr(runner, "_fit_lgbm_model", fake_fit)

    def fake_save(*, model_dir, **kwargs):
        final_dir = model_dir / "final_all_rows"
        final_dir.mkdir(parents=True, exist_ok=True)
        return {
            "model_path": str(final_dir / "base_model.joblib"),
            "manifest_path": str(final_dir / "manifest.json"),
            "feature_count": len(kwargs["feature_names"]),
        }

    monkeypatch.setattr(runner, "_save_base_fold_model", fake_save)
    folds = [
        {
            "fold": "2026-01-01_2026-02-01",
            "month": "2026-01",
            "valid_start": pd.Timestamp("2026-01-01", tz="UTC"),
            "valid_end": pd.Timestamp("2026-02-01", tz="UTC"),
        }
    ]
    result = runner._fit_final_all_rows_base_model(
        folds=folds,
        params={"target_mode": "target_soft", "weight_arm": "W0_base"},
        trial_number=7,
        max_train_rows=0,
        train_window_days=1,
        seed=42,
        model_dir=tmp_path / "models",
    )

    assert captured["rows"] == 2
    assert captured["target"] == [0.2, 0.3]
    assert captured["weight"] == [1.0, 1.5]
    assert result["excluded_from_oos_metrics"] is True
    assert result["source_oos_fold"] == "2026-01-01_2026-02-01"
    assert result["train_start"] == pd.Timestamp("2026-01-02T00:00:00Z")
    assert result["train_end"] == pd.Timestamp("2026-01-03T00:00:00Z")
    assert result["train_window_days"] == 1
    assert result["excluded_outside_train_window_rows"] == 1
    persisted = json.loads(
        (tmp_path / "models/final_all_rows/manifest.json").read_text(encoding="utf-8")
    )
    assert persisted["excluded_from_oos_metrics"] is True
    assert persisted["leakage_contract"]["oos_metrics"].startswith("excluded")


def test_final_refit_packages_exact_frozen_ae_gmm_contract(tmp_path: Path) -> None:
    source = tmp_path / "cycle__global_state.pkl"
    source.write_bytes(b"frozen-state-bytes")
    source.with_name("cycle__global_manifest.json").write_text(
        json.dumps({"schema": "source-state"}), encoding="utf-8"
    )

    result = runner._package_final_ae_gmm_contract(
        final_dir=tmp_path / "models/final_all_rows",
        state_path=source,
        input_features=["f2", "f1", "f2"],
    )

    packaged = tmp_path / "models/final_all_rows/ae_gmm_state/ae_gmm_state.pkl"
    assert packaged.read_bytes() == source.read_bytes()
    assert result["status"] == "packaged"
    assert result["state_sha256"] == runner._sha256_file(source)
    inputs = json.loads(
        (tmp_path / "models/final_all_rows/ae_gmm_state/input_features.json").read_text(
            encoding="utf-8"
        )
    )
    assert inputs["ordered_input_features"] == ["f2", "f1"]
    assert inputs["input_feature_count"] == 2
    assert inputs["input_feature_order_hash"] == runner._feature_contract_hash(
        ["f2", "f1"]
    )
    assert (
        tmp_path
        / "models/final_all_rows/ae_gmm_state/source_state_manifest.json"
    ).is_file()


def test_compact_final_refit_defers_without_full_row_payloads(tmp_path: Path) -> None:
    folds = [
        {
            "fold": "2026-06-30_2026-07-30",
            "valid_start": pd.Timestamp("2026-06-30", tz="UTC"),
            "valid_end": pd.Timestamp("2026-07-30", tz="UTC"),
            "compact_fixed_training_payload": True,
            "payload_paths": {
                "x_train": "x_train.parquet",
                "x_valid": "x_valid.parquet",
                "train_target": "train_target.parquet",
                "train_weight": "train_weight.parquet",
                "train_side": "train_side.parquet",
                "valid": "valid.parquet",
                "valid_metrics": "valid_metrics.parquet",
            },
        }
    ]

    result = runner._fit_final_all_rows_base_model(
        folds=folds,
        params={"target_mode": "target_soft", "weight_arm": "W7_timestamp_balanced"},
        trial_number=135,
        max_train_rows=0,
        train_window_days=365,
        seed=42,
        model_dir=tmp_path / "models",
    )

    assert result["status"] == "deferred_to_compact_side_specific_packager"
    assert "train" in result["missing_payloads"]
    assert result["excluded_from_oos_metrics"] is True
