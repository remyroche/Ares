from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scripts.run_strict_r3_meta_t6t9_incumbent_selection_v2 import _counterpart_window


def _write_rank(root: Path, arm: str, month: str, candidate_id: str, timestamp: str) -> None:
    target = root / "target_free_scores" / arm
    target.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "candidate_id": [candidate_id],
        "__decision_ts__": pd.to_datetime([timestamp], utc=True),
        "side_name": ["long"],
        "head__cap80_ordinary__rank": [0.75],
    }).to_parquet(target / f"month={month}.parquet", index=False)


def test_counterpart_window_uses_only_strict_monthly_oof_receipts(tmp_path: Path) -> None:
    arm = "T6_rank_error_ordinal"
    _write_rank(tmp_path, arm, "2026-01", "a", "2026-01-31T23:00:00Z")
    _write_rank(tmp_path, arm, "2026-02", "b", "2026-02-01T00:00:00Z")

    result = _counterpart_window(
        tmp_path,
        pd.Timestamp("2026-01-15T00:00:00Z"),
        pd.Timestamp("2026-02-15T00:00:00Z"),
        arm,
    )

    assert result.candidate_id.tolist() == ["a", "b"]
    assert arm in result.columns


def test_counterpart_window_fails_closed_when_a_strict_oof_month_is_missing(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="missing strict-OOF incumbent geometry receipt"):
        _counterpart_window(
            tmp_path,
            pd.Timestamp("2026-01-01T00:00:00Z"),
            pd.Timestamp("2026-02-01T00:00:00Z"),
            "T9_exit5_ordinal",
        )
