from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.report_oos_layer_metrics import LayerSpec, generate_report, prepare_layer


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [
                "2026-07-06 00:00:00",
                "2026-07-06 01:00:00",
                "2026-07-13 00:00:00",
                "2026-07-13 01:00:00",
            ],
            "symbol": ["aaa/usd:usd", "BBB/USD:USD", "AAA/USD:USD", "BBB/USD:USD"],
            "side": ["long", "short", "long", "short"],
            "archetype_label_family": ["trend", "mean_reversion", "trend", "mean_reversion"],
            "score": [0.99, 0.90, 0.80, 0.70],
            "net_return": [0.02, -0.01, 0.03, -0.02],
            "clean_exec": [1, 0, 1, 0],
            "dirty_positive": [0, 1, 0, 1],
            "first_touch_bad_mae_1r": [0, 1, 0, 1],
            "timeout": [0, 0, 1, 0],
        }
    )


def test_generate_report_uses_utc_key_intersection_and_emits_all_group_tables(tmp_path: Path) -> None:
    base = _rows()
    base_path = tmp_path / "base.parquet"
    base.to_parquet(base_path, index=False)
    meta = base.iloc[:3].copy()
    meta["accepted"] = [True, True, False]
    meta["net_return"] = [0.03, -0.005, 0.04]
    meta_path = tmp_path / "meta.parquet"
    meta.to_parquet(meta_path, index=False)

    manifest = generate_report(
        [
            LayerSpec("base", base_path, score_col="score", top_frac=0.75, cost_provenance="fee30bps_spread_p90"),
            LayerSpec("meta", meta_path, selected_col="accepted", cost_provenance="fee30bps_spread_p90"),
        ],
        tmp_path / "report",
    )

    assert manifest["canonical_key"] == ["UTC timestamp", "symbol", "side"]
    assert set(manifest["metric_tables"]) == {
        "overall", "month", "week", "side", "archetype", "week_side_archetype"
    }
    overall = pd.read_csv(tmp_path / "report" / "oos_layer_metrics_overall.csv")
    assert overall.set_index("layer").loc["base", "selected_rows"] == 3
    assert overall.set_index("layer").loc["meta", "selected_rows"] == 2
    assert overall.set_index("layer").loc["base", "trades_per_day"] == pytest.approx(1.5)
    deltas = pd.read_csv(tmp_path / "report" / "oos_layer_deltas_overall.csv")
    assert deltas.loc[0, "overlap_rows"] == 2
    assert bool(deltas.loc[0, "cost_comparable"])
    # Meta's same-key mean is (3.0 - .5) / 2 percent; base's is (2.0 - 1.0) / 2.
    assert deltas.loc[0, "delta_notional_net_return_per_trade"] == pytest.approx(0.0075)
    weekly = pd.read_csv(tmp_path / "report" / "oos_layer_metrics_week_side_archetype.csv")
    assert {"week_start", "side", "archetype"}.issubset(weekly.columns)


def test_duplicate_canonical_keys_are_rejected(tmp_path: Path) -> None:
    duplicate = _rows().iloc[:2].copy()
    duplicate.loc[1, ["timestamp", "symbol", "side"]] = duplicate.loc[0, ["timestamp", "symbol", "side"]].to_numpy()
    path = tmp_path / "duplicate.parquet"
    duplicate.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="duplicate canonical keys"):
        prepare_layer(LayerSpec("base", path, allow_all_rows=True, cost_provenance="fee30bps"))


def test_cost_mismatch_is_explicit_and_suppresses_financial_delta(tmp_path: Path) -> None:
    base = _rows().iloc[:2].copy()
    base_path = tmp_path / "base.parquet"
    base.to_parquet(base_path, index=False)
    meta = base.copy()
    meta["accepted"] = True
    meta_path = tmp_path / "meta.parquet"
    meta.to_parquet(meta_path, index=False)
    generate_report(
        [
            LayerSpec("base", base_path, allow_all_rows=True, cost_provenance="fee100bps"),
            LayerSpec("meta", meta_path, selected_col="accepted", cost_provenance="fee30bps_spread_p90"),
        ],
        tmp_path / "report",
    )
    delta = pd.read_csv(tmp_path / "report" / "oos_layer_deltas_overall.csv").iloc[0]
    assert not bool(delta["cost_comparable"])
    assert np.isnan(delta["delta_notional_net_return_per_trade"])
    assert delta["delta_clean_rate"] == pytest.approx(0.0)
