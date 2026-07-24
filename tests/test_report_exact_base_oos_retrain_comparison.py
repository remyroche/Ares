from __future__ import annotations

import pandas as pd

from scripts.report_exact_base_oos_retrain_comparison import run


def _ledger(*, aware: bool) -> pd.DataFrame:
    ts = pd.date_range("2026-03-28", periods=4, freq="h", tz="UTC")
    if not aware:
        ts = ts.tz_localize(None)
    return pd.DataFrame(
        {
            "__ts__": ts,
            "__symbol__": ["BTC/USD:USD"] * 4,
            "side": [1, -1, 1, -1],
            "side_name": ["long", "short", "long", "short"],
            "__archetype_label_family__": ["mixed"] * 4,
            "oos_fold": ["2026-04"] * 4,
            "score": [0.1, 0.2, 0.3, 0.4],
            "selected_top10": [False, False, False, True],
            "selected_top20": [False, False, True, True],
            "selected_top30": [False, True, True, True],
            "__first_touch_capture_net__": [0.01, -0.01, 0.02, 0.03],
            "__first_touch_hit__": [1, 0, 1, 1],
            "__first_touch_stop__": [0, 1, 0, 0],
            "__first_touch_timeout__": [0, 0, 0, 0],
            "__first_touch_mae_to_sl__": [0.2, 1.1, 0.3, 0.1],
        }
    )


def test_comparison_joins_utc_aware_and_utc_naive_as_same_instants(tmp_path) -> None:
    new_path = tmp_path / "new.parquet"
    incumbent_path = tmp_path / "incumbent.parquet"
    _ledger(aware=False).to_parquet(new_path, index=False)
    _ledger(aware=True).drop(
        columns=["side_name", "__archetype_label_family__", "oos_fold"]
    ).to_parquet(incumbent_path, index=False)

    manifest = run(
        new_ledger=new_path,
        incumbent_ledger=incumbent_path,
        output_dir=tmp_path / "report",
    )

    assert manifest["row_contract"]["overlap_rows"] == 4
    assert manifest["label_and_score_parity"]["label_event_mismatch_rows"] == 0
    assert manifest["label_and_score_parity"]["max_label_net_abs_diff"] == 0.0
    assert manifest["timestamp_contract"].startswith("join_on_utc_epoch_ns")
