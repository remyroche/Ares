from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_source_gated_vanilla_diagnostic import run_report  # noqa: E402


def _frame() -> pd.DataFrame:
    rows = []
    for month, day in [("2026-04", "2026-04-01"), ("2026-05", "2026-05-01")]:
        ts = pd.date_range(day, periods=4, freq="h", tz="UTC")
        rows.extend(
            [
                {
                    "__ts__": ts[0],
                    "__symbol__": "BTC/USD:USD",
                    "__barrier_pct__": 0.020,
                    "__mfe_ret__": 0.050,
                    "__mae_ret__": 0.006,
                    "__u_policy_net__": 0.030,
                    "__y_ret__": 0.031,
                    "__y_bin__": 1,
                    "__is_timeout__": 0,
                    "oof_pred": 0.90,
                    "train_include_risk_adjusted_capture_candidate_v4": True,
                    "train_include_compression_capture_candidate_v3": False,
                    "train_include_dirty_excluded_v0": True,
                    "month": month,
                },
                {
                    "__ts__": ts[1],
                    "__symbol__": "ETH/USD:USD",
                    "__barrier_pct__": 0.019,
                    "__mfe_ret__": 0.040,
                    "__mae_ret__": 0.004,
                    "__u_policy_net__": 0.020,
                    "__y_ret__": 0.021,
                    "__y_bin__": 1,
                    "__is_timeout__": 0,
                    "oof_pred": 0.80,
                    "train_include_risk_adjusted_capture_candidate_v4": True,
                    "train_include_compression_capture_candidate_v3": True,
                    "train_include_dirty_excluded_v0": True,
                    "month": month,
                },
                {
                    "__ts__": ts[2],
                    "__symbol__": "SOL/USD:USD",
                    "__barrier_pct__": 0.040,
                    "__mfe_ret__": 0.010,
                    "__mae_ret__": 0.060,
                    "__u_policy_net__": -0.030,
                    "__y_ret__": -0.029,
                    "__y_bin__": 0,
                    "__is_timeout__": 1,
                    "oof_pred": 0.95,
                    "train_include_risk_adjusted_capture_candidate_v4": False,
                    "train_include_compression_capture_candidate_v3": False,
                    "train_include_dirty_excluded_v0": False,
                    "month": month,
                },
                {
                    "__ts__": ts[3],
                    "__symbol__": "XRP/USD:USD",
                    "__barrier_pct__": 0.018,
                    "__mfe_ret__": 0.005,
                    "__mae_ret__": 0.015,
                    "__u_policy_net__": -0.010,
                    "__y_ret__": -0.009,
                    "__y_bin__": 0,
                    "__is_timeout__": 0,
                    "oof_pred": 0.10,
                    "train_include_risk_adjusted_capture_candidate_v4": False,
                    "train_include_compression_capture_candidate_v3": False,
                    "train_include_dirty_excluded_v0": True,
                    "month": month,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_source_gated_vanilla_report_writes_outputs_and_deltas(tmp_path: Path) -> None:
    input_path = tmp_path / "joined.parquet"
    output_dir = tmp_path / "out"
    _frame().to_parquet(input_path, index=False)

    manifest = run_report(
        joined_subset_path=input_path,
        output_dir=output_dir,
        score_cols=["oof_pred"],
        gate_columns=["train_include_risk_adjusted_capture_candidate_v4"],
        dirty_excluded_column="train_include_dirty_excluded_v0",
        barrier_guards=[0.025],
        months=["2026-04", "2026-05"],
        top_fracs=[0.5],
        min_score_rows=1,
    )

    assert manifest["rows"] == 8
    assert Path(manifest["outputs"]["report"]).exists()
    monthly = pd.read_csv(output_dir / "source_gated_vanilla_monthly.csv")
    aggregate = pd.read_csv(output_dir / "source_gated_vanilla_aggregate.csv")
    profile = pd.read_csv(output_dir / "source_gated_vanilla_gate_profile_by_period.csv")

    gated = aggregate[aggregate["gate"].eq("risk_adjusted_capture_candidate")]
    assert not gated.empty
    assert gated["delta_mean_u_vs_all_rows"].iloc[0] > 0.0
    assert set(monthly["period"]).issuperset({"2026-04", "2026-05"})
    assert "gate_profile_month" in set(profile["selector"])


def test_source_gated_vanilla_report_requires_available_score(tmp_path: Path) -> None:
    input_path = tmp_path / "joined.parquet"
    _frame().drop(columns=["oof_pred"]).to_parquet(input_path, index=False)

    with pytest.raises(ValueError, match="No requested score columns"):
        run_report(
            joined_subset_path=input_path,
            output_dir=tmp_path / "out",
            score_cols=["oof_pred"],
            gate_columns=["train_include_risk_adjusted_capture_candidate_v4"],
            dirty_excluded_column="train_include_dirty_excluded_v0",
            barrier_guards=[0.025],
            months=["2026-04", "2026-05"],
            top_fracs=[0.5],
            min_score_rows=1,
        )
