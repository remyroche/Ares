from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "report_catboost_path_archetype_oof.py"
_SPEC = importlib.util.spec_from_file_location("report_catboost_path_archetype_oof", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
run_report = _MODULE.run_report
PATH_SHAPE_TYPES = _MODULE.PATH_SHAPE_TYPES


def _oof_frame() -> pd.DataFrame:
    probabilities = [(0.89, 0.05), (0.05, 0.89), (0.59, 0.35), (0.65, 0.29)]
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                [
                    "2026-01-31T23:00:00Z",
                    "2026-02-01T00:00:00Z",
                    "2026-02-02T00:00:00Z",
                    "2026-02-08T00:00:00Z",
                ],
                utc=True,
            ),
            "side": ["long", "long", "short", "short"],
            "path_archetype": [PATH_SHAPE_TYPES[0], PATH_SHAPE_TYPES[1], PATH_SHAPE_TYPES[0], PATH_SHAPE_TYPES[1]],
            "predicted_path_archetype": [PATH_SHAPE_TYPES[0], PATH_SHAPE_TYPES[1], PATH_SHAPE_TYPES[0], PATH_SHAPE_TYPES[0]],
            "oof_fold_id": [0, 0, 1, 1],
        }
    )
    frame[f"probability__{PATH_SHAPE_TYPES[0]}"] = [value[0] for value in probabilities]
    frame[f"probability__{PATH_SHAPE_TYPES[1]}"] = [value[1] for value in probabilities]
    for shape in PATH_SHAPE_TYPES[2:]:
        frame[f"probability__{shape}"] = 0.01
    return frame


def test_report_writes_oof_metrics_and_utc_calendar_groups(tmp_path) -> None:
    input_path = tmp_path / "oof_probabilities.parquet"
    output_dir = tmp_path / "report"
    _oof_frame().to_parquet(input_path, index=False)

    result = run_report(input_path, output_dir)

    metrics = json.loads((output_dir / "oof_metrics.json").read_text())
    manifest = json.loads((output_dir / "oof_manifest.json").read_text())
    csv = pd.read_csv(output_dir / "oof_metrics.csv")
    assert result["metrics"]["overall"]["all"]["rows"] == 4
    assert metrics["overall"]["all"]["accuracy"] == pytest.approx(0.75)
    assert set(metrics) >= {
        "month",
        "week",
        "fold",
        "side",
        "true_path_archetype",
        "side_x_true_path_archetype",
    }
    assert set(metrics["month"]) == {"2026-01", "2026-02"}
    assert "2026-W05" in metrics["week"]
    assert metrics["side"]["long"]["rows"] == 2
    assert metrics["true_path_archetype"][PATH_SHAPE_TYPES[0]]["classwise"][PATH_SHAPE_TYPES[0]]["recall"] == 1.0
    assert manifest["source"]["sha256"]
    assert "OOF-only" in manifest["claim"]
    assert {"classwise_json", "confusion_matrix_json"}.issubset(csv.columns)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        (f"probability__{PATH_SHAPE_TYPES[0]}", 1.2, r"within \[0, 1\]"),
        (f"probability__{PATH_SHAPE_TYPES[1]}", 0.3, "sum to 1"),
    ],
)
def test_report_rejects_invalid_probabilities(tmp_path, column, value, message) -> None:
    frame = _oof_frame()
    frame.loc[0, column] = value
    input_path = tmp_path / "invalid.parquet"
    frame.to_parquet(input_path, index=False)

    with pytest.raises(ValueError, match=message):
        run_report(input_path, tmp_path / "report")


def test_report_rejects_non_utc_timestamps(tmp_path) -> None:
    frame = _oof_frame()
    frame["__ts__"] = frame["__ts__"].dt.tz_convert("Europe/Paris")
    input_path = tmp_path / "non_utc.parquet"
    frame.to_parquet(input_path, index=False)

    with pytest.raises(ValueError, match="must use UTC"):
        run_report(input_path, tmp_path / "report")
