from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_exact_base_oos_retrain_comparison import run


def _ledger(*, scores: list[float], selected: list[bool]) -> pd.DataFrame:
    ts = pd.date_range("2026-04-01", periods=4, freq="7D", tz="UTC")
    return pd.DataFrame(
        {
            "__ts__": ts,
            "__symbol__": ["BTC/USD:USD", "ETH/USD:USD", "BTC/USD:USD", "ETH/USD:USD"],
            "side": [1, -1, 1, -1],
            "side_name": ["long", "short", "long", "short"],
            "__archetype_label_family__": ["trend", "breakout", "trend", "breakout"],
            "oos_fold": ["f1", "f1", "f2", "f2"],
            "score": scores,
            "selected_top10": selected,
            "selected_top20": selected,
            "selected_top30": selected,
            "__first_touch_capture_net__": [0.02, -0.01, 0.03, 0.01],
            "__first_touch_hit__": [1, 0, 1, 1],
            "__first_touch_stop__": [0, 1, 0, 0],
            "__first_touch_timeout__": [0, 0, 0, 0],
            "__first_touch_mae_to_sl__": [0.2, 1.2, 0.1, 0.4],
        }
    )


def test_exact_oos_comparison_uses_shared_rows_and_label_contract(tmp_path: Path) -> None:
    new_path = tmp_path / "new.parquet"
    old_path = tmp_path / "old.parquet"
    out_dir = tmp_path / "report"
    _ledger(scores=[0.9, 0.2, 0.8, 0.1], selected=[True, False, True, False]).to_parquet(new_path)
    _ledger(scores=[0.8, 0.3, 0.7, 0.2], selected=[True, False, False, True]).to_parquet(old_path)

    manifest = run(new_ledger=new_path, incumbent_ledger=old_path, output_dir=out_dir)

    assert manifest["row_contract"]["overlap_rows"] == 4
    assert manifest["label_and_score_parity"]["max_label_net_abs_diff"] == 0.0
    delta = pd.read_csv(out_dir / "overall_delta.csv")
    top10 = delta.loc[delta["top_k"].eq(10)].iloc[0]
    assert top10["net_ev_per_trade_new"] == 0.025
    assert top10["net_ev_per_trade_incumbent"] == 0.015
    assert top10["net_ev_per_trade_delta"] == 0.01
