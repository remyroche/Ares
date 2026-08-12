import numpy as np
import pandas as pd

from scripts.run_strict_r3_current_exact_b5_fold import (
    _eligible_fields,
    _timestamp_top30,
    _train_selection_mask,
)


def test_expected_map_sidecar_fields_never_enter_n5_features() -> None:
    rows = 128
    frame = pd.DataFrame({
        **{f"market_feature_{index}": np.linspace(index, index + 1, rows)
           for index in range(12)},
        "__n5_raw_expected_map_bps": np.linspace(-100.0, 200.0, rows),
        "__n5_raw_expected_map_admitted": np.arange(rows) % 2,
        "__n5_map_decision_ts": pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC"),
    })

    selected = _eligible_fields(frame)

    assert len(selected) == 12
    assert not any(field.startswith("__n5_") for field in selected)


def test_mixed_domain_contract_retains_lower_score_reference_rows() -> None:
    frame = pd.DataFrame({
        "candidate_id": [f"c{index}" for index in range(20)],
        "__decision_ts__": np.repeat(
            pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"), 10,
        ),
        "final_score": np.tile(np.arange(10, dtype=float), 2),
    })
    top = _timestamp_top30(frame)
    mask, definition = _train_selection_mask(frame, "mixed_top30_reference")
    assert top.sum() == 6
    assert mask.all()
    assert definition == "75pct_timestamp_top30_plus_25pct_lower_reference"
