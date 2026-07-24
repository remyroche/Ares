from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "materialize_execution_ev_auxiliary_oof",
    ROOT / "scripts" / "materialize_execution_ev_auxiliary_oof.py",
)
assert SPEC and SPEC.loader
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def _args(head_dir: Path, output: Path, **overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "head_dir": head_dir,
        "target_kind": "timing",
        "output": output,
        "manifest": output.with_suffix(".manifest.json"),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _write_metrics(head_dir: Path) -> None:
    payload = {
        "fold_metrics": {
            "long": [
                {
                    "fold": 3,
                    "train_end": "2025-12-31T11:00:00Z",
                    "valid_start": "2026-01-01T00:00:00Z",
                    "valid_end": "2026-01-01T01:00:00Z",
                },
                {
                    "fold": 7,
                    "train_end": "2025-12-31T13:00:00Z",
                    "valid_start": "2026-01-01T02:00:00Z",
                    "valid_end": "2026-01-01T03:00:00Z",
                },
            ],
            "short": [
                {
                    "fold": 2,
                    "train_end": "2025-12-31T11:00:00Z",
                    "valid_start": "2026-01-01T00:00:00Z",
                    "valid_end": "2026-01-01T01:00:00Z",
                },
                {
                    "fold": 5,
                    "train_end": "2025-12-31T13:00:00Z",
                    "valid_start": "2026-01-01T02:00:00Z",
                    "valid_end": "2026-01-01T03:00:00Z",
                },
            ],
        }
    }
    (head_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def _old_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T01:00:00Z",
                    "2026-01-01T02:00:00Z",
                    "2026-01-01T03:00:00Z",
                    "2025-12-31T23:00:00Z",
                ]
            ),
            "__symbol__": ["BTC/USD:USD", "ETH/USD:USD", "BTC/USD:USD", "ETH/USD:USD", "SOL/USD:USD"],
            "side": ["long", "short", "long", "short", "long"],
            "target": [1.0, 2.0, 3.0, 4.0, 5.0],
            "oof_prediction": np.array(
                [np.log1p(1.0), np.log1p(2.0), np.log1p(20.0), np.log1p(-0.5), np.nan]
            ),
        }
    )
    finite = frame["oof_prediction"].notna()
    frame["candidate_id"] = [f"candidate-{index}" for index in range(len(frame))]
    frame["oof_fold"] = [3, 2, 7, 5, -1]
    frame["validation_start"] = pd.to_datetime(
        ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "2026-01-01T02:00:00Z", "2026-01-01T02:00:00Z", "2025-12-31T23:00:00Z"]
    )
    frame["train_decision_cutoff"] = frame["validation_start"] - pd.Timedelta(hours=1)
    frame["label_resolution_available_at"] = frame["train_decision_cutoff"]
    frame["available_at"] = frame["__ts__"]
    frame.loc[~finite, "oof_fold"] = -1
    return frame


def test_strict_schema_preserves_identity_fold_evidence_clips_and_excludes_target(tmp_path: Path) -> None:
    head_dir = tmp_path / "time_to_peak_mfe"
    head_dir.mkdir()
    _old_frame().to_parquet(head_dir / "oof_predictions.parquet", index=False)

    output = tmp_path / "timing.parquet"
    paths = adapter.run(_args(head_dir, output))
    result = pd.read_parquet(paths["output"])
    assert list(result.columns) == list(adapter.OUTPUT_COLUMNS)
    assert result["prediction"].tolist() == pytest.approx([1.0, 2.0, 12.0, 0.0])
    assert result["oof_fold"].tolist() == [3, 2, 7, 5]
    assert result["candidate_id"].tolist() == ["candidate-0", "candidate-1", "candidate-2", "candidate-3"]
    assert result["available_at"].equals(result["__ts__"])
    assert "target" not in result

    manifest = json.loads(paths["manifest"].read_text())
    assert manifest["oof_fold"]["mode"] == "source_row_level_actual_fitted_fold_evidence"
    assert manifest["source"]["dropped_unavailable_rows"] == 1
    assert len(manifest["output"]["sha256"]) == 64
    assert manifest["source_artifact_sha256"] == adapter._sha256(paths["output"])
    assert manifest["prediction_role"] == "time_to_mfe_oof"
    assert manifest["prediction_role_manifest_sha256"] == adapter._canonical_json_hash(
        manifest, excluded=("prediction_role_manifest_sha256",)
    )


def test_new_schema_uses_source_fold_and_availability_and_converts_log1p(tmp_path: Path) -> None:
    head_dir = tmp_path / "peak_mfe"
    head_dir.mkdir()
    times = pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"])
    pd.DataFrame(
        {
            "timestamp": times,
            "symbol": ["BTC/USD:USD", "ETH/USD:USD"],
            "side_name": ["buy", "sell"],
            "oof_prediction_log1p": np.log1p([0.5, 12.0]),
            "oof_fold": [4, 4],
            "available_at": times - pd.Timedelta(minutes=5),
            "candidate_id": ["btc-0", "eth-1"],
            "validation_start": times.floor("h"),
            "train_decision_cutoff": times - pd.Timedelta(hours=1),
            "label_resolution_available_at": times - pd.Timedelta(hours=2),
            "target": [0.5, 12.0],
        }
    ).to_parquet(head_dir / "oof_predictions.parquet", index=False)

    output = tmp_path / "peak.parquet"
    paths = adapter.run(_args(head_dir, output, target_kind="peak_mfe"))
    result = pd.read_parquet(paths["output"])
    assert result["side_name"].tolist() == ["long", "short"]
    assert result["prediction"].tolist() == pytest.approx([0.5, 10.0])
    assert result["available_at"].tolist() == list(times - pd.Timedelta(minutes=5))
    manifest = json.loads(paths["manifest"].read_text())
    assert manifest["oof_fold"]["mode"] == "source_row_level_actual_fitted_fold_evidence"


@pytest.mark.parametrize(
    ("target_kind", "natural_column", "values", "expected"),
    [
        (
            "mae_before_meaningful_mfe_atr",
            "pred_mae_before_meaningful_mfe_atr_12h",
            [0.5, 12.0],
            [0.5, 10.0],
        ),
        (
            "bars_before_price_stops_decreasing",
            "pred_bars_before_price_stops_decreasing_12h",
            [2.0, 20.0],
            [2.0, 12.0],
        ),
        (
            "future_slope_atr_per_hour",
            "pred_future_slope_atr_per_hour_12h",
            [0.75, 15.0],
            [0.75, 10.0],
        ),
    ],
)
def test_adapter_supports_additional_path_heads(
    tmp_path: Path,
    target_kind: str,
    natural_column: str,
    values: list[float],
    expected: list[float],
) -> None:
    head_dir = tmp_path / target_kind
    head_dir.mkdir()
    times = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    pd.DataFrame(
        {
            "__ts__": times,
            "__symbol__": ["BTC/USD:USD", "ETH/USD:USD"],
            "side": ["long", "short"],
            natural_column: values,
            "oof_fold": [0, 0],
            "available_at": times,
            "candidate_id": ["btc-0", "eth-1"],
            "validation_start": times,
            "train_decision_cutoff": times - pd.Timedelta(hours=1),
            "label_resolution_available_at": times - pd.Timedelta(hours=2),
        }
    ).to_parquet(head_dir / "oof_predictions.parquet", index=False)
    output = tmp_path / f"{target_kind}.parquet"

    adapter.run(_args(head_dir, output, target_kind=target_kind))

    result = pd.read_parquet(output)
    assert result["prediction"].tolist() == pytest.approx(expected)


def test_rejects_row_level_evidence_that_places_prediction_before_validation(tmp_path: Path) -> None:
    head_dir = tmp_path / "timing"
    head_dir.mkdir()
    frame = _old_frame().iloc[:4].copy()
    frame["oof_prediction_log1p"] = frame.pop("oof_prediction")
    frame.loc[0, "validation_start"] = frame.loc[0, "__ts__"] + pd.Timedelta(hours=1)
    frame.to_parquet(head_dir / "oof_predictions.parquet", index=False)
    with pytest.raises(ValueError, match="validation start is after"):
        adapter.run(_args(head_dir, tmp_path / "mismatch.parquet"))


@pytest.mark.parametrize("missing_column", ["candidate_id", "validation_start"])
def test_rejects_legacy_source_without_row_level_fold_evidence(
    tmp_path: Path, missing_column: str
) -> None:
    head_dir = tmp_path / "time_to_peak_mfe"
    head_dir.mkdir()
    frame = _old_frame().iloc[:1].copy()
    frame = frame.drop(columns=[missing_column])
    frame.to_parquet(head_dir / "oof_predictions.parquet", index=False)
    with pytest.raises(ValueError, match="do not infer"):
        adapter.run(_args(head_dir, tmp_path / "out.parquet"))


def test_rejects_duplicate_identity_nonfinite_only_and_late_availability(tmp_path: Path) -> None:
    head_dir = tmp_path / "peak_mfe"
    head_dir.mkdir()
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-01-01T00:00:00Z"]),
            "__symbol__": ["BTC/USD:USD"],
            "side": ["long"],
            "oof_prediction_log1p": [np.nan],
            "oof_fold": [0],
            "candidate_id": ["btc-0"],
            "validation_start": [pd.Timestamp("2026-01-01T00:00:00Z")],
            "train_decision_cutoff": [pd.Timestamp("2025-12-31T23:00:00Z")],
            "label_resolution_available_at": [pd.Timestamp("2025-12-31T22:00:00Z")],
            "available_at": [pd.Timestamp("2026-01-01T00:00:00Z")],
        }
    )
    frame.to_parquet(head_dir / "oof_predictions.parquet", index=False)
    with pytest.raises(ValueError, match="no finite"):
        adapter.run(_args(head_dir, tmp_path / "nonfinite.parquet", target_kind="peak"))

    frame.loc[0, "oof_prediction_log1p"] = np.log1p(1.0)
    duplicate = pd.concat([frame, frame], ignore_index=True)
    duplicate.to_parquet(head_dir / "oof_predictions.parquet", index=False)
    with pytest.raises(ValueError, match="duplicate rows"):
        adapter.run(_args(head_dir, tmp_path / "duplicate.parquet", target_kind="peak"))

    frame["available_at"] = frame["__ts__"] + pd.Timedelta(seconds=1)
    frame.to_parquet(head_dir / "oof_predictions.parquet", index=False)
    with pytest.raises(ValueError, match="availability is after"):
        adapter.run(_args(head_dir, tmp_path / "late.parquet", target_kind="peak"))
