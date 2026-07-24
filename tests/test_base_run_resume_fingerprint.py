from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts import run_materialized_trailing_label_topk_lgbm_hpo as runner


class _ConstantModel:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), 0.5, dtype=np.float32)


def _compact_fold() -> dict[str, object]:
    return {
        "fold": "2026-04",
        "month": "2026-04",
        "valid_start": pd.Timestamp("2026-04-01", tz="UTC"),
        "valid_end": pd.Timestamp("2026-05-01", tz="UTC"),
        "max_oos_model_age_days": 0,
        "train_rows_uncapped": 3,
        "train_rows_payload": 2,
        "payload_train_sampling": "beginning_middle_end_time_spread",
        "fixed_training_target_mode": "target_soft",
        "fixed_training_weight_arm": "W0_base",
        "compact_fixed_training_payload": True,
        "reuse_fingerprint": "fold-input-v1",
        "x_train": pd.DataFrame({"x": [0.1, 0.2]}, dtype=np.float32),
        "x_valid": pd.DataFrame({"x": [0.3, 0.4]}, dtype=np.float32),
        "train_median_imputation_values": {"x": 0.15},
        "train_side": pd.DataFrame({"side_name": ["long", "short"]}),
        "ae_gmm_context_valid": pd.DataFrame({"gmm_entropy": [0.1, 0.2]}),
        "train_target": pd.DataFrame(
            {"target_soft": [0.2, 0.8], "target_hard": [0.0, 1.0]}
        ),
        "train_weight": pd.DataFrame({"sample_weight": [1.0, 1.0]}),
        "valid": pd.DataFrame(
            {
                "__ts__": pd.date_range("2026-04-01", periods=2, freq="h", tz="UTC"),
                "__symbol__": ["BTC", "ETH"],
                "side": [1, -1],
                "candidate_id": ["BTC|2026-04-01T00:00:00Z|long", "ETH|2026-04-01T01:00:00Z|short"],
                "__u_policy_net__": [0.01, -0.01],
            }
        ),
        "train_provenance": pd.DataFrame(
            {
                "candidate_id": ["BTC|2026-03-31T21:00:00Z|long", "ETH|2026-03-31T22:00:00Z|short"],
                "__decision_ts__": pd.to_datetime(
                    ["2026-03-31T22:00:00Z", "2026-03-31T23:00:00Z"], utc=True
                ),
                "__label_resolution_ts__": pd.to_datetime(
                    ["2026-03-31T22:15:00Z", "2026-03-31T23:30:00Z"], utc=True
                ),
            }
        ),
        "valid_metrics": pd.DataFrame(
            {
                "ret_net": [0.01, -0.01],
                "first_touch_hit": [1.0, 0.0],
                "first_touch_stop": [0.0, 1.0],
                "first_touch_timeout": [0.0, 0.0],
                "first_touch_same_bar": [0.0, 0.0],
                "first_touch_mae_to_sl": [0.5, 1.5],
                "first_touch_bar": [4.0, 18.0],
                "first_touch_net": [0.01, -0.01],
                "clean_first_touch_exec": [1.0, 0.0],
            }
        ),
    }


def test_fold_payload_reuse_requires_exact_fingerprint(tmp_path) -> None:
    fold = _compact_fold()
    runner._write_fold_payload(fold, tmp_path / "cache")
    window = {
        "fold": "2026-04",
        "month": "2026-04",
        "valid_start": pd.Timestamp("2026-04-01", tz="UTC"),
        "valid_end": pd.Timestamp("2026-05-01", tz="UTC"),
        "train_rows_estimate": 3,
        "valid_rows_estimate": 2,
        "valid_rows_raw_estimate": 2,
    }
    contract = {"target_mode": "target_soft", "weight_arm": "W0_base"}

    reused = runner._reuse_complete_fold_payload(
        cache_dir=tmp_path / "cache",
        window=window,
        selected_features=["x"],
        fixed_training_contract=contract,
        expected_reuse_fingerprint="fold-input-v1",
    )
    assert reused is not None
    assert reused["reuse_fingerprint"] == "fold-input-v1"

    stale = runner._reuse_complete_fold_payload(
        cache_dir=tmp_path / "cache",
        window=window,
        selected_features=["x"],
        fixed_training_contract=contract,
        expected_reuse_fingerprint="fold-input-v2",
    )
    assert stale is None


def test_base_oof_provenance_derives_resolution_from_first_touch_path() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-04-01T00:00:00Z"], utc=True),
            "__symbol__": ["BTC"],
            "side": [1],
            "candidate_id": ["btc-candidate"],
            "__first_path_ts__": pd.to_datetime(["2026-04-01T01:00:00Z"], utc=True),
            "__first_touch_bar__": [2.0],
        }
    )

    out, contract = runner._base_oof_provenance_columns(frame)

    assert out.loc[0, "candidate_id"] == "btc-candidate"
    assert out.loc[0, "__label_resolution_ts__"] == pd.Timestamp(
        "2026-04-01T01:30:00Z"
    )
    assert contract["label_resolution_column"] == "__label_resolution_ts__"
    assert contract["label_resolution_source_column"] == ""
    assert contract["label_resolution_derivation"] == (
        "first_path_plus_first_touch_bars_x_15m"
    )


def test_scored_and_model_reuse_require_matching_fingerprint(tmp_path, monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def _fit(**kwargs):
        calls.append(dict(kwargs["params"]))
        return _ConstantModel()

    monkeypatch.setattr(runner, "_fit_lgbm_model", _fit)
    fold = _compact_fold()
    output_path = tmp_path / "best.parquet"
    model_dir = tmp_path / "models"
    params = {
        "n_estimators": 10,
        "learning_rate": 0.03,
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "loss_function": "regression",
        "min_split_gain": 0.0,
        "target_mode": "target_soft",
        "weight_arm": "W0_base",
    }

    runner._score_best_oos_ledger(
        folds=[fold],
        params=params,
        trial_number=1,
        max_train_rows=0,
        seed=7,
        save_fold_models_dir=model_dir,
        output_path=output_path,
    )
    assert len(calls) == 1
    ledger = pd.read_parquet(output_path)
    assert ledger["candidate_id"].tolist() == [
        "BTC|2026-04-01T00:00:00Z|long",
        "ETH|2026-04-01T01:00:00Z|short",
    ]
    assert ledger["fold_validation_start"].eq(fold["valid_start"]).all()
    assert ledger["fold_validation_end"].eq(fold["valid_end"]).all()
    assert ledger["latest_train_decision_cutoff"].eq(
        pd.Timestamp("2026-03-31T23:00:00Z")
    ).all()
    assert ledger["latest_train_decision_timestamp"].eq(
        pd.Timestamp("2026-03-31T23:00:00Z")
    ).all()
    assert ledger["latest_train_resolved_label_timestamp"].eq(
        pd.Timestamp("2026-03-31T23:30:00Z")
    ).all()
    assert ledger["label_resolution_column"].eq("__label_resolution_ts__").all()
    scored_manifest = json.loads(
        (tmp_path / "_scored_fold_cache" / "2026-04.manifest.json").read_text()
    )
    assert scored_manifest["base_oof_provenance"]["latest_train_decision_cutoff"] == "2026-03-31T23:00:00+00:00"
    model_manifest = json.loads((model_dir / "2026-04" / "manifest.json").read_text())
    assert model_manifest["base_oof_provenance"] == scored_manifest["base_oof_provenance"]

    runner._score_best_oos_ledger(
        folds=[fold],
        params=params,
        trial_number=1,
        max_train_rows=0,
        seed=7,
        save_fold_models_dir=model_dir,
        output_path=output_path,
    )
    assert len(calls) == 1

    scored_cache = tmp_path / "_scored_fold_cache"
    (scored_cache / "2026-04.parquet").unlink()
    (scored_cache / "2026-04.manifest.json").unlink()
    runner._score_best_oos_ledger(
        folds=[fold],
        params=params,
        trial_number=1,
        max_train_rows=0,
        seed=7,
        save_fold_models_dir=model_dir,
        output_path=output_path,
    )
    assert len(calls) == 1

    changed_params = {**params, "n_estimators": 11}
    runner._score_best_oos_ledger(
        folds=[fold],
        params=changed_params,
        trial_number=1,
        max_train_rows=0,
        seed=7,
        save_fold_models_dir=model_dir,
        output_path=output_path,
    )
    assert len(calls) == 2
