from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/run_packb_side_local_outer_oof.py"
SPEC = importlib.util.spec_from_file_location("packb_outer_runner", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _ledger(side: str = "long") -> pd.DataFrame:
    signal = pd.to_datetime(["2026-04-01T00:00:00Z", "2026-04-01T01:00:00Z"], utc=True)
    return pd.DataFrame(
        {
            "candidate_id": [f"{side}-a", f"{side}-b"],
            "side_name": side,
            "__ts__": signal,
            "__decision_ts__": signal + pd.Timedelta(hours=1),
            "__label_resolution_ts__": signal + pd.Timedelta(hours=25),
            "__symbol__": ["BTCUSDT", "ETHUSDT"],
        }
    )


def test_complete_case_gate_never_imputes() -> None:
    ledger = _ledger()
    features = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 3.0]})
    labels = pd.DataFrame(
        {
            runner.TARGET_COLUMN: [0.4, 0.5],
            runner.WEIGHT_COLUMN: [1.0, 1.0],
            runner.ECONOMIC_COLUMN: [0.01, 0.02],
        }
    )
    admitted_ledger, admitted_x, admitted_labels, evidence = runner._admit_complete(
        ledger, features, labels
    )
    assert admitted_ledger["candidate_id"].tolist() == ["long-a"]
    assert admitted_x.to_dict(orient="records") == [{"a": 1.0, "b": 2.0}]
    assert len(admitted_labels) == 1
    assert evidence["joint_complete_fraction"] == 0.5
    assert evidence["policy"].startswith("no_imputation")


def test_complete_case_gate_rejects_alignment_and_zero_weight() -> None:
    ledger = _ledger()
    labels = pd.DataFrame(
        {
            runner.TARGET_COLUMN: [0.4, 0.5],
            runner.WEIGHT_COLUMN: [0.0, 0.0],
            runner.ECONOMIC_COLUMN: [0.01, 0.02],
        }
    )
    with pytest.raises(runner.PackBOuterOOFRunnerError, match="positive-weight"):
        runner._admit_complete(ledger, pd.DataFrame({"a": [1.0, 2.0]}), labels)
    with pytest.raises(runner.PackBOuterOOFRunnerError, match="alignment"):
        runner._admit_complete(ledger.iloc[:1], pd.DataFrame({"a": [1.0, 2.0]}), labels)


def test_native_missing_admission_keeps_nan_rows_without_imputation() -> None:
    ledger = _ledger()
    features = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 3.0]})
    labels = pd.DataFrame(
        {
            runner.TARGET_COLUMN: [0.4, 0.5],
            runner.WEIGHT_COLUMN: [1.0, 1.0],
            runner.ECONOMIC_COLUMN: [0.01, 0.02],
        }
    )
    admitted_ledger, admitted_x, admitted_labels, evidence = (
        runner._admit_native_missing(
            ledger,
            features,
            labels,
            min_per_feature_finite_fraction=0.5,
        )
    )
    assert admitted_ledger["candidate_id"].tolist() == ["long-a", "long-b"]
    assert np.isnan(admitted_x.loc[1, "a"])
    assert len(admitted_labels) == 2
    assert evidence["minimum_per_feature_finite_fraction"] == 0.5
    assert evidence["policy"] == "lightgbm_native_nan_no_imputation_label_complete_rows"


def test_native_missing_admission_rejects_feature_below_floor() -> None:
    ledger = _ledger()
    features = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 3.0]})
    labels = pd.DataFrame(
        {
            runner.TARGET_COLUMN: [0.4, 0.5],
            runner.WEIGHT_COLUMN: [1.0, 1.0],
            runner.ECONOMIC_COLUMN: [0.01, 0.02],
        }
    )
    with pytest.raises(runner.PackBOuterOOFRunnerError, match="coverage fell below"):
        runner._admit_native_missing(
            ledger,
            features,
            labels,
            min_per_feature_finite_fraction=0.95,
        )


def test_outer_representation_union_is_loaded_in_bounded_batches() -> None:
    ledger = pd.concat([_ledger(), _ledger()], ignore_index=True)
    ledger["candidate_id"] = [f"id-{index}" for index in range(len(ledger))]
    calls: list[int] = []

    def loader(batch: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
        calls.append(len(batch))
        assert features == ("dae_b16_06",)
        return pd.DataFrame({"dae_b16_06": np.arange(len(batch), dtype=np.float32)})

    cache, evidence, union = runner._precompute_outer_representations(
        loader,
        [ledger],
        ["dae_b16_06"],
        batch_rows=2,
    )

    assert calls == [2, 2]
    assert evidence["batch_count"] == 2
    assert cache(union, ["dae_b16_06"]).shape == (4, 1)


def test_metrics_are_cost_aware_and_timestamp_ranked() -> None:
    ledger = _ledger()
    ledger.loc[1, "__ts__"] = ledger.loc[0, "__ts__"]
    labels = pd.DataFrame(
        {
            runner.TARGET_COLUMN: [1.0, 0.0],
            runner.WEIGHT_COLUMN: [1.0, 1.0],
            runner.ECONOMIC_COLUMN: [0.02, -0.01],
        }
    )
    result = runner._metrics(np.array([0.9, 0.1]), ledger, labels)
    assert result["top10_mean_net_return"] == pytest.approx(0.02)
    assert result["ranking_scope"] == "within_utc_timestamp_and_side"
