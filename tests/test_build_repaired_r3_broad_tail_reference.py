from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.build_repaired_r3_broad_tail_reference import _validate_part


def _expected() -> pd.DataFrame:
    return pd.DataFrame({
        "candidate_id": ["candidate"],
        "__ts__": [pd.Timestamp("2023-04-01T00:00:00Z")],
        "__symbol__": ["BTC/USD:USD"],
        "side_name": ["long"],
    })


def test_repaired_reference_rejects_a_legacy_part_without_exact_t3_fields(tmp_path: Path) -> None:
    part = _expected().assign(
        label_valid=True,
        target_invalid=False,
        t4_tp6_sl4_gross_bps=100.0,
        t4_tp6_sl4_net_bps=0.0,
    )
    path = tmp_path / "legacy.parquet"
    part.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="first_tp4_minute"):
        _validate_part(path, _expected(), month="2023-04", side="long")
