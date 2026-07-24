from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from scripts.extend_frozen_ae_gmm_output_sidecar import (
    _restore_case_sensitive_output_names,
)


def test_restore_case_sensitive_output_names_repairs_duckdb_suffix(tmp_path: Path) -> None:
    path = tmp_path / "outputs.parquet"
    pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
            "__symbol__": ["BTC/USD:USD", "ETH/USD:USD"],
            "side": [1, -1],
            "AE_reconstruction_error": [0.1, 0.2],
            "ae_reconstruction_error_1": [0.3, 0.4],
        }
    ).to_parquet(path, index=False)

    _restore_case_sensitive_output_names(
        path,
        output_features=("AE_reconstruction_error", "ae_reconstruction_error"),
    )

    assert pq.ParquetFile(path).schema_arrow.names == [
        "__ts__",
        "__symbol__",
        "side",
        "AE_reconstruction_error",
        "ae_reconstruction_error",
    ]
    repaired = pd.read_parquet(path)
    assert repaired["ae_reconstruction_error"].tolist() == [0.3, 0.4]
