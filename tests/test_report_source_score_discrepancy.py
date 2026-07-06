from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_source_score_discrepancy import run_report  # noqa: E402


def _frame() -> pd.DataFrame:
    rows = []
    for month, start in [
        ("2026-03", "2026-03-01"),
        ("2026-04", "2026-04-01"),
        ("2026-05", "2026-05-01"),
    ]:
        for i, ts in enumerate(pd.date_range(start, periods=6, freq="h", tz="UTC")):
            good = i < 3
            gate = i in {0, 1, 3}
            rows.append(
                {
                    "__ts__": ts,
                    "__symbol__": f"S{i % 3}/USD:USD",
                    "candidate_id": f"{month}-{i}",
                    "side": 1,
                    "__barrier_pct__": 0.020 if good else 0.030,
                    "__mfe_ret__": 0.050 if good else 0.010,
                    "__mae_ret__": 0.004 if good else 0.045,
                    "__bars_to_mfe__": 2 if good else 12,
                    "__bars_policy__": 5 if good else 24,
                    "__y_ret__": 0.020 if good else -0.020,
                    "__y_bin__": 1 if good else 0,
                    "__is_timeout__": 0 if good else 1,
                    "__u_policy_net__": 0.025 if good else -0.020,
                    "__y_outcome__": 1 if good else 0,
                    "f0": float(i),
                    "f1": float(good),
                    "oof_pred": 1.0 - (i / 10.0),
                    "train_include_risk_adjusted_capture_candidate_v4": gate,
                    "train_include_compression_capture_candidate_v3": i in {1, 2},
                    "train_include_dirty_excluded_v0": i != 3,
                    "primary_source_tag": "risk_adjusted_capture_candidate" if gate else "ambiguous_none",
                    "tag_risk_adjusted_capture_candidate": gate,
                }
            )
    return pd.DataFrame(rows)


def test_source_score_discrepancy_report_writes_overlap_and_ledgers(tmp_path: Path) -> None:
    input_path = tmp_path / "joined.parquet"
    output_dir = tmp_path / "out"
    _frame().to_parquet(input_path, index=False)

    manifest = run_report(
        joined_subset_path=input_path,
        output_dir=output_dir,
        feature_dir=None,
        feature_list_csv=None,
        feature_cols=["f0", "f1"],
        max_feature_store_features=None,
        existing_score_cols=["oof_pred"],
        gate_columns=["train_include_risk_adjusted_capture_candidate_v4"],
        dirty_excluded_column="train_include_dirty_excluded_v0",
        barrier_guards=[0.025],
        months=["2026-04", "2026-05"],
        top_fracs=[0.5],
        seeds=[42],
        train_lookback_months=None,
        min_train_rows=4,
        min_valid_rows=1,
    )

    assert manifest["rows"] == 18
    assert Path(manifest["outputs"]["report"]).exists()
    selection = pd.read_csv(output_dir / "source_score_discrepancy_selection_summary.csv")
    overlap = pd.read_csv(output_dir / "source_score_discrepancy_overlap.csv")
    ledger = pd.read_csv(output_dir / "source_score_discrepancy_selected_ledger.csv")

    assert {"extratrees_s10_policy_net_soft", "existing_score"}.issubset(set(selection["selector"]))
    assert not overlap.empty
    assert overlap["other_score_col"].eq("oof_pred").any()
    assert {"candidate_id", "selector", "score_col", "u_policy_net"}.issubset(ledger.columns)


def test_source_score_discrepancy_requires_existing_score(tmp_path: Path) -> None:
    input_path = tmp_path / "joined.parquet"
    frame = _frame().drop(columns=["oof_pred"])
    frame.to_parquet(input_path, index=False)

    with pytest.raises(ValueError, match="No existing score columns"):
        run_report(
            joined_subset_path=input_path,
            output_dir=tmp_path / "out",
            feature_dir=None,
            feature_list_csv=None,
            feature_cols=["f0", "f1"],
            max_feature_store_features=None,
            existing_score_cols=["oof_pred"],
            gate_columns=["train_include_risk_adjusted_capture_candidate_v4"],
            dirty_excluded_column="train_include_dirty_excluded_v0",
            barrier_guards=[0.025],
            months=["2026-04", "2026-05"],
            top_fracs=[0.5],
            seeds=[42],
            train_lookback_months=None,
            min_train_rows=4,
            min_valid_rows=1,
        )
