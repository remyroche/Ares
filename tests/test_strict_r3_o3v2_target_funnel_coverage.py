from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scripts.run_strict_r3_o3v2_target_funnel import _require_complete_feature_months


def _write_month(root: Path, month: str) -> None:
    target = root / f"month={month}"
    target.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"candidate_id": ["a"]}).to_parquet(target / "scores_features.parquet", index=False)


def test_complete_feature_month_guard_accepts_all_declared_months(tmp_path: Path) -> None:
    _write_month(tmp_path, "2025-04")
    _write_month(tmp_path, "2025-05")
    _require_complete_feature_months(
        tmp_path,
        pd.Timestamp("2025-04-03T00:00:00Z"),
        pd.Timestamp("2025-06-01T00:00:00Z"),
        context="test",
    )


def test_complete_feature_month_guard_rejects_silently_shortened_fit_window(tmp_path: Path) -> None:
    _write_month(tmp_path, "2025-04")
    with pytest.raises(AssertionError, match="missing source months"):
        _require_complete_feature_months(
            tmp_path,
            pd.Timestamp("2025-03-03T00:00:00Z"),
            pd.Timestamp("2025-05-01T00:00:00Z"),
            context="test",
        )
