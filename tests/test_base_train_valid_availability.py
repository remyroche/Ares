import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_materialized_trailing_label_topk_lgbm_hpo import (
    _cycle_reference_input_survivors,
    _restore_cycle_input_columns,
    _train_valid_availability_survivors,
)


def test_train_valid_availability_drops_collapsed_tail_feature() -> None:
    train = pd.DataFrame(
        {
            "healthy": np.linspace(-1.0, 1.0, 100),
            "tail_store_outage": np.tile([-1.0, 1.0], 50),
            "legitimate_zero": np.zeros(100),
        }
    )
    valid = pd.DataFrame(
        {
            "healthy": np.linspace(-0.5, 0.5, 40),
            "tail_store_outage": np.zeros(40),
            "legitimate_zero": np.zeros(40),
        }
    )

    survivors, report = _train_valid_availability_survivors(train, valid)

    assert survivors == ["healthy", "legitimate_zero"]
    assert report["collapsed_tail_features"] == ["tail_store_outage"]


def test_train_valid_availability_rejects_missing_validation_column() -> None:
    train = pd.DataFrame({"present": [1.0, 2.0], "missing": [1.0, 2.0]})
    valid = pd.DataFrame({"present": [1.0, 2.0]})

    survivors, report = _train_valid_availability_survivors(train, valid)

    assert survivors == ["present"]
    assert report["checked_features"] == 1


def test_cycle_preflight_runs_before_frozen_state_fit() -> None:
    timestamps = pd.date_range("2025-01-01", periods=1_200, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "__ts__": timestamps,
            "healthy": np.linspace(-1.0, 1.0, len(timestamps)),
            "tail_store_outage": np.concatenate(
                [np.tile([-1.0, 1.0], 500), np.zeros(200)]
            ),
            "legitimate_zero": np.zeros(len(timestamps)),
        }
    )
    reference_window = {
        "valid_start": timestamps[1_000],
        "valid_end": timestamps[-1] + pd.Timedelta(hours=1),
        "train_start": None,
    }

    survivors, report = _cycle_reference_input_survivors(
        frame=frame,
        ts_utc=pd.Series(timestamps),
        reference_window=reference_window,
        candidate_features=["healthy", "tail_store_outage", "legitimate_zero"],
        payload_max_train_rows=0,
    )

    assert "healthy" in survivors
    assert "legitimate_zero" in survivors
    assert "tail_store_outage" not in survivors
    assert report["enabled"] is True
    assert report["availability"]["collapsed_tail_features"] == [
        "tail_store_outage"
    ]


def test_cycle_inputs_are_restored_outside_the_model_candidate_matrix() -> None:
    x_train = pd.DataFrame({"selected": [1.0, 2.0]})
    x_valid = pd.DataFrame({"selected": [3.0]})
    train_source = pd.DataFrame(
        {"selected": [1.0, 2.0], "state_only": [4.0, 5.0]}
    )
    valid_source = pd.DataFrame({"selected": [3.0], "state_only": [6.0]})

    train_out, valid_out, restored = _restore_cycle_input_columns(
        x_train,
        x_valid,
        train_source=train_source,
        valid_source=valid_source,
        required_columns=["selected", "state_only"],
    )

    assert restored == ["state_only"]
    assert train_out["state_only"].tolist() == [4.0, 5.0]
    assert valid_out["state_only"].tolist() == [6.0]
