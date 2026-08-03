from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_execution_ev_regime_diagnosis",
    ROOT / "scripts" / "run_execution_ev_regime_diagnosis.py",
)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _frame() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for month in pd.period_range("2026-01", "2026-05", freq="M"):
        times = month.start_time.tz_localize("UTC") + pd.to_timedelta(
            np.arange(12), unit="h"
        )
        score = np.linspace(-1.0, 1.0, len(times))
        frames.append(
            pd.DataFrame(
                {
                    "execution_decision_utc": times,
                    "execution_label_end_utc": times + pd.Timedelta(hours=6),
                    "execution_net_ev_12h": score * 0.01,
                    "score": score,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_cli_defaults_to_bounded_split_only_dry_run(tmp_path: Path) -> None:
    input_path = tmp_path / "joined.parquet"
    _frame().to_parquet(input_path, index=False)
    paths = runner.run(
        runner._parser().parse_args(
            [
                "--input",
                str(input_path),
                "--output-dir",
                str(tmp_path / "diagnosis"),
                "--feature-cols",
                "score",
                "--train-window-months",
                "2",
                "--min-train-rows",
                "8",
                "--max-periods",
                "3",
            ]
        )
    )
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    split_rows = pd.read_csv(paths["splits"])
    assert manifest["status"] == "dry_run_planned"
    assert manifest["execution"]["requested"] is False
    assert manifest["selection_basis"] == "global_topk_across_full_evaluation_period"
    assert "metrics" not in paths
    assert split_rows["promotion_eligible"].eq(False).all()
    assert split_rows.loc[
        split_rows["mode"].eq("reversed_month_diagnostic"), "evaluation_status"
    ].eq("diagnostic_non_oos_reversed_training").all()


def test_fixed_catboost_winner_fits_residual_and_restores_baseline(
    monkeypatch,
) -> None:
    observed: dict[str, np.ndarray] = {}

    class FakeCatBoost:
        def __init__(self, **kwargs):
            observed["iterations"] = np.asarray([kwargs["iterations"]])

        def fit(self, x, y, sample_weight=None):
            observed["target"] = np.asarray(y, dtype=float)
            observed["weight"] = np.asarray(sample_weight, dtype=float)

        def predict(self, x):
            return np.full(len(x), 0.25, dtype=float)

    monkeypatch.setitem(
        sys.modules,
        "catboost",
        types.SimpleNamespace(CatBoostRegressor=FakeCatBoost),
    )
    hook = runner._fixed_catboost_residual_hook(
        baseline_column="existing_alpha_ev",
        n_estimators=17,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=6.0,
        random_strength=0.5,
        bagging_temperature=1.0,
        random_state=42,
        n_jobs=1,
    )
    train_x = pd.DataFrame(
        {"existing_alpha_ev": [0.1, 0.2], "feature": [1.0, 2.0]}
    )
    evaluation_x = pd.DataFrame(
        {"existing_alpha_ev": [0.3, 0.4], "feature": [3.0, 4.0]}
    )
    prediction = hook(
        train_x,
        np.asarray([0.5, 0.7]),
        evaluation_x,
        np.asarray([1.0, 2.0]),
    )
    np.testing.assert_allclose(observed["target"], [0.4, 0.5])
    np.testing.assert_allclose(observed["weight"], [1.0, 2.0])
    np.testing.assert_allclose(prediction, [0.55, 0.65])
    assert observed["iterations"][0] == 17


def test_execute_validates_input_only_once(tmp_path: Path) -> None:
    input_path = tmp_path / "joined.parquet"
    _frame().to_parquet(input_path, index=False)
    paths = runner.run(
        runner._parser().parse_args(
            [
                "--input",
                str(input_path),
                "--output-dir",
                str(tmp_path / "executed"),
                "--feature-cols",
                "score",
                "--train-window-months",
                "2",
                "--min-train-rows",
                "8",
                "--max-periods",
                "5",
                "--n-estimators",
                "5",
                "--n-jobs",
                "1",
                "--execute",
            ]
        )
    )
    metrics = pd.read_csv(paths["metrics"])
    assert not metrics.empty
    assert metrics["selection_basis"].eq(
        "global_topk_across_full_evaluation_period"
    ).all()


def test_recency_weights_are_training_only_and_halve_by_age() -> None:
    train = pd.DataFrame(
        {
            "execution_decision_utc": pd.to_datetime(
                ["2026-06-01T00:00:00Z", "2026-06-11T00:00:00Z"], utc=True
            )
        }
    )
    weights = runner._recency_sample_weight_hook(
        "execution_decision_utc", 10.0
    )(train)
    np.testing.assert_allclose(weights, [0.5, 1.0])


def test_side_local_dry_run_filters_before_building_splits(tmp_path: Path) -> None:
    input_path = tmp_path / "joined.parquet"
    frame = pd.concat(
        [
            _frame().assign(side_name="long"),
            _frame().assign(side_name="short"),
        ],
        ignore_index=True,
    )
    frame.to_parquet(input_path, index=False)
    paths = runner.run(
        runner._parser().parse_args(
            [
                "--input", str(input_path),
                "--output-dir", str(tmp_path / "short_diagnosis"),
                "--feature-cols", "score",
                "--side", "short",
                "--train-window-months", "2",
                "--min-train-rows", "8",
                "--max-periods", "3",
            ]
        )
    )
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["side"] == "short"
    assert manifest["input"]["rows"] == len(_frame())
