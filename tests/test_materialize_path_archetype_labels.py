from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_path_archetype_labels import materialize


def test_script_materializes_manifest_and_support_table(tmp_path) -> None:
    candidates = pd.DataFrame(
        {
            "__ts__": [pd.Timestamp("2026-01-01", tz="UTC")],
            "__symbol__": ["BTC"],
            "side": ["long"],
            "entry_price": [100.0],
            "risk_distance": [10.0],
            "atr_fraction": [0.02],
        }
    )
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01 01:00:00", periods=24, freq="h", tz="UTC"
            ),
            "symbol": ["BTC"] * 24,
            "high": [101.0 + i for i in range(24)],
            "low": [99.0 + i for i in range(24)],
            "close": [100.5 + i for i in range(24)],
        }
    )
    candidate_path, bars_path, output_dir = (
        tmp_path / "candidates.parquet",
        tmp_path / "bars.parquet",
        tmp_path / "out",
    )
    candidates.to_parquet(candidate_path, index=False)
    bars.to_parquet(bars_path, index=False)
    manifest = materialize(candidate_path, bars_path, output_dir, batch_rows=1)
    written = pd.read_parquet(output_dir / "path_archetype_labels.parquet")
    assert manifest["complete_24h_rows"] == 1
    assert manifest["realization_strength"]["atr_thresholds"] == [1.5, 2.0, 3.0, 5.0]
    assert written.loc[0, "path_archetype"]
    support = pd.read_csv(output_dir / "path_archetype_support_summary.csv")
    assert {
        "mean_path_arch_time_to_first_meaningful_mfe_h",
        "mean_path_arch_time_to_90pct_peak_mfe_h",
        "mean_path_arch_time_to_150atr_h",
        "mean_path_arch_time_to_200atr_h",
        "mean_path_arch_time_to_300atr_h",
        "mean_path_arch_time_to_500atr_h",
    }.issubset(support.columns)
    assert json.loads((output_dir / "manifest.json").read_text())["utc_key"] == [
        "__ts__",
        "__symbol__",
        "side",
    ]
