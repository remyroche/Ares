from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.materialize_strict_oos_proxy_predictions import (
    compact_policy_oos_frame,
    materialize_predictions,
    month_to_fold_suffix,
)


STRATEGY_ID = "long_demo_strategy"


def _write_policy_oos(root: Path, experiment_id: str, period: str, frame: pd.DataFrame) -> Path:
    suffix = month_to_fold_suffix(period)
    out_dir = (
        root
        / "artifacts"
        / f"{experiment_id}_{suffix}"
        / "policy_oos_predictions"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"policy_oos_{STRATEGY_ID}_clf.parquet"
    frame.to_parquet(path, index=False)
    return path


def _frame(period: str, *, symbol: str = "BTC/USD:USD", score: float = 0.7) -> pd.DataFrame:
    timestamp = pd.Timestamp(f"{period}-03 00:00:00", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": [timestamp],
            "symbol": [symbol],
            "oof_pred": [score],
            "oof_base_clf": [score + 0.01],
            "oof_meta_clf": [score + 0.02],
            "base_rank_pct": [0.8],
            "base_model_score_pct": [0.75],
            "mr_tf_policy_score_source": ["general"],
            "pred_H10_pred_mean": [score + 0.03],
            "base_H10_pred_mean": [score + 0.04],
        }
    )


def test_month_to_fold_suffix_generalizes_to_july() -> None:
    assert month_to_fold_suffix("2026-04") == "train_march_score_april"
    assert month_to_fold_suffix("2026-07") == "train_june_score_july"


def test_compact_policy_oos_frame_uses_expected_schema(tmp_path: Path) -> None:
    path = tmp_path / "policy_oos_demo_clf.parquet"
    _frame("2026-04").to_parquet(path, index=False)

    compact = compact_policy_oos_frame(path, period="2026-04")

    assert list(compact.columns) == [
        "timestamp",
        "symbol",
        "oof_pred",
        "oof_base_clf",
        "oof_meta_clf",
        "base_rank_pct",
        "base_model_score_pct",
        "mr_tf_policy_score_source",
        "pred_H10_pred_mean",
        "base_H10_pred_mean",
        "prediction_source_path",
    ]
    assert compact.loc[0, "symbol"] == "BTC/USD:USD"
    assert compact.loc[0, "prediction_source_path"] == str(path)


def test_compact_policy_oos_frame_falls_back_from_clf(tmp_path: Path) -> None:
    path = tmp_path / "policy_oos_demo_clf.parquet"
    pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-04-03", tz="UTC")],
            "symbol": ["ETH/USD:USD"],
            "clf": [0.61],
        }
    ).to_parquet(path, index=False)

    compact = compact_policy_oos_frame(path, period="2026-04")

    assert compact.loc[0, "oof_pred"] == pytest.approx(0.61)
    assert compact.loc[0, "oof_meta_clf"] == pytest.approx(0.61)
    assert compact.loc[0, "mr_tf_policy_score_source"] == "unknown"


def test_materialize_predictions_writes_parquet_and_manifest(tmp_path: Path) -> None:
    data_root = tmp_path / "data_perp"
    experiment_id = "exp"
    _write_policy_oos(data_root, experiment_id, "2026-04", _frame("2026-04", score=0.7))
    _write_policy_oos(data_root, experiment_id, "2026-05", _frame("2026-05", score=0.8))
    output_path = tmp_path / "out" / "strict_proxy.parquet"

    manifest = materialize_predictions(
        data_root=data_root,
        experiment_id=experiment_id,
        months=["2026-04", "2026-05"],
        output_path=output_path,
        strategy_id=STRATEGY_ID,
    )

    assert output_path.exists()
    out = pd.read_parquet(output_path)
    assert len(out) == 2
    assert manifest["period_counts"] == {"2026-04": 1, "2026-05": 1}
    manifest_path = Path(manifest["manifest_path"])
    assert manifest_path.exists()
    loaded = json.loads(manifest_path.read_text())
    assert loaded["rows"] == 2


def test_materialize_predictions_rejects_duplicate_timestamp_symbol(tmp_path: Path) -> None:
    data_root = tmp_path / "data_perp"
    experiment_id = "exp"
    _write_policy_oos(data_root, experiment_id, "2026-04", _frame("2026-04", score=0.7))
    _write_policy_oos(data_root, experiment_id, "2026-05", _frame("2026-05", score=0.8))

    # Force the May fixture into April's timestamp/symbol after month filtering
    # would normally separate it.
    may_path = (
        data_root
        / "artifacts"
        / f"{experiment_id}_train_april_score_may"
        / "policy_oos_predictions"
        / f"policy_oos_{STRATEGY_ID}_clf.parquet"
    )
    dup = _frame("2026-05", score=0.8)
    dup["timestamp"] = pd.Timestamp("2026-04-03 00:00:00", tz="UTC")
    dup.to_parquet(may_path, index=False)

    with pytest.raises(RuntimeError, match="duplicate timestamp/symbol"):
        materialize_predictions(
            data_root=data_root,
            experiment_id=experiment_id,
            months=["2026-04", "2026-04"],
            output_path=tmp_path / "strict_proxy.parquet",
            strategy_id=STRATEGY_ID,
        )
