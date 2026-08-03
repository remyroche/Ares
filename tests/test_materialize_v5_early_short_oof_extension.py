from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

RUNNER = (
    Path(__file__).resolve().parents[1]
    / "scripts/materialize_v5_early_short_oof_extension.py"
)
SPEC = importlib.util.spec_from_file_location("early_short_extension", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_frozen_contract_requires_exact_selected_arm_and_order() -> None:
    payload = {
        "frozen_winner": {
            "arm": "B_peak_slope",
            "key": "B_peak_slope__tail_2",
            "short_tail_weight": 2.0,
        },
        "contract": {
            "arms": {
                "B_peak_slope": [
                    *MODULE.base.F,
                    "peak_contribution",
                    "pred_future_slope_atr_per_hour__diagnostic",
                ]
            }
        },
    }
    features, tail = MODULE.frozen_contract(payload)
    assert features[-2:] == [
        "peak_contribution",
        "pred_future_slope_atr_per_hour__diagnostic",
    ]
    assert tail == 2.0


def test_early_short_rows_are_purged_and_cover_march_13_19(monkeypatch) -> None:
    timestamps = pd.date_range(
        "2025-03-01T00:00:00Z",
        "2025-03-19T23:00:00Z",
        freq="h",
    )
    rows = []
    for timestamp in timestamps:
        for index in range(48):
            rows.append(
                {
                    "candidate_id": f"{timestamp}-{index}",
                    "side_name": "short",
                    "__symbol__": str(index),
                    "__ts__": timestamp - pd.Timedelta(hours=1),
                    MODULE.TIME: timestamp,
                    MODULE.END: timestamp + pd.Timedelta(hours=12),
                    **{field: float(index) for field in MODULE.base.F},
                    "peak_contribution": float(index),
                    "pred_future_slope_atr_per_hour__diagnostic": float(index),
                }
            )
    frame = pd.DataFrame(rows)

    def fake_fit(train, valid, features, tail):
        assert train[MODULE.END].max() < valid[MODULE.TIME].min()
        assert tail == 2.0
        return (
            np.full(len(valid), 0.5),
            np.full(len(valid), 0.02),
            np.full(len(valid), 0.01),
            np.linspace(-0.01, 0.01, len(valid)),
        )

    monkeypatch.setattr(MODULE.short, "fit_decomp", fake_fit)
    result, audit = MODULE.early_short_rows(
        frame,
        features=[
            *MODULE.base.F,
            "peak_contribution",
            "pred_future_slope_atr_per_hour__diagnostic",
        ],
        tail_weight=2.0,
    )
    assert len(result) == 8_064
    assert result[MODULE.TIME].min() == pd.Timestamp("2025-03-13T00:00:00Z")
    assert result[MODULE.TIME].max() == pd.Timestamp("2025-03-19T23:00:00Z")
    assert result.candidate_score_is_oof.all()
    assert audit.train_label_end_max_utc.lt(audit.validation_start_utc).all()
