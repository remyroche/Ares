from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def test_source_index_range_materialises_complete_target_free_population(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    symbols = tuple(f"S{index:03d}/USD:USD" for index in range(160))
    index = pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="h")
    source = tmp_path / "source.joblib"
    joblib.dump(
        {
            "symbols": symbols,
            "panel": {"close": pd.DataFrame(np.ones((3, 160)), index=index, columns=symbols)},
        },
        source,
    )
    output = tmp_path / "candidates"
    runner = root / "scripts" / "materialize_strict_r3_p8u_target_free_candidates.py"
    subprocess.run(
        [
            sys.executable,
            str(runner),
            "--source-state",
            str(source),
            "--start",
            "2026-01-01T00:00:00Z",
            "--end-exclusive",
            "2026-01-01T02:00:00Z",
            "--out-dir",
            str(output),
        ],
        cwd=root,
        check=True,
    )
    candidates = pd.read_parquet(output / "candidates.parquet")
    receipt = json.loads((output / "receipt.json").read_text())
    assert len(candidates) == 320
    assert candidates["__ts__"].nunique() == 2
    assert candidates.groupby("__ts__")["__symbol__"].nunique().eq(160).all()
    assert receipt["selection_mode"] == "source_index_range"
    assert receipt["selected_timestamps"] is None
    assert receipt["selected_timestamps_count"] == 2
    assert receipt["first_selected_timestamp"] == "2026-01-01T00:00:00+00:00"
    assert receipt["last_selected_timestamp"] == "2026-01-01T01:00:00+00:00"
    assert receipt["future_path_or_outcome_filter_applied"] is False
