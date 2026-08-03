from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from scripts.score_frozen_path_head_forward_oos import classify_forward_rows


def test_forward_classifier_preserves_final_refit_information_boundary() -> None:
    timestamps = pd.Series(pd.to_datetime([
        "2026-07-11T00:00:00Z", "2026-07-11T12:00:00Z", "2026-07-11T13:00:00Z",
    ], utc=True))
    result = classify_forward_rows(
        timestamps,
        oof_last_input=pd.Timestamp("2026-07-10T23:00:00Z"),
        final_refit_available_at=pd.Timestamp("2026-07-11T12:00:00Z"),
    )
    assert result["prediction_origin"].tolist() == [
        "retrospective_post_oof_overlap",
        "forward_frozen_final_refit",
        "forward_frozen_final_refit",
    ]
    assert not result["is_oof"].any()
    assert result["is_forward_oos"].tolist() == [False, True, True]
    assert not result["promotion_eligible"].any()


def test_forward_classifier_rejects_historical_oof_rows() -> None:
    with pytest.raises(ValueError, match="historical OOF"):
        classify_forward_rows(
            pd.Series(pd.to_datetime(["2026-07-10T23:00:00Z"], utc=True)),
            oof_last_input=pd.Timestamp("2026-07-10T23:00:00Z"),
            final_refit_available_at=pd.Timestamp("2026-07-11T12:00:00Z"),
        )


def test_cli_help_imports_from_the_scripts_directory() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/score_frozen_path_head_forward_oos.py", "--help"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "frozen Peak-MFE and CatBoost" in result.stdout
