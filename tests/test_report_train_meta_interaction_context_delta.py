from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_train_meta_interaction_context_delta import run


def _write_smoke_dir(path: Path, *, ev_shift: float, full_sl_shift: float) -> None:
    path.mkdir(parents=True)
    aggregate_rows = []
    cell_rows = []
    for month in ("2026-01", "2026-02"):
        for selector in ("s0_ev_only", "s12_ev_clean_strong_risk"):
            aggregate_rows.append(
                {
                    "month": month,
                    "selector": selector,
                    "top_frac": 0.10,
                    "selected_rows": 10,
                    "precision_positive_ev": 0.4 + ev_shift,
                    "ev_weighted_precision": 0.7 + ev_shift,
                    "mean_ev_after_1pct": 0.01 + ev_shift,
                    "full_sl_rate": 0.3 + full_sl_shift,
                    "timeout_rate": 0.2,
                    "clean_exec_proxy_rate": 0.25 + ev_shift,
                }
            )
            for side, arch in (("long", "long_bars"), ("short", "short_bollinger")):
                cell_rows.append(
                    {
                        "month": month,
                        "side_name": side,
                        "source_archetype": arch,
                        "selector": selector,
                        "top_frac": 0.10,
                        "rows": 100,
                        "eligible_rows": 100,
                        "guard_pass_rate": 1.0,
                        "selected_rows": 10,
                        "selected_share": 0.1,
                        "precision_positive_ev": 0.4 + ev_shift,
                        "ev_weighted_precision": 0.7 + ev_shift,
                        "mean_ev_after_1pct": 0.01 + ev_shift,
                        "sum_ev_after_1pct": 0.1,
                        "full_sl_rate": 0.3 + full_sl_shift,
                        "timeout_rate": 0.2,
                        "clean_exec_proxy_rate": 0.25 + ev_shift,
                    }
                )
    pd.DataFrame(aggregate_rows).to_csv(path / "risk_aware_train_meta_aggregate.csv", index=False)
    pd.DataFrame(cell_rows).to_csv(path / "risk_aware_train_meta_by_cell.csv", index=False)
    pd.DataFrame(
        {
            "month": ["2026-01"],
            "status": ["fit"],
            "feature_count": [12],
            "all_null_feature_count": [1],
            "constant_feature_count": [2],
        }
    ).to_csv(path / "risk_aware_train_meta_fit_events.csv", index=False)


def test_report_train_meta_interaction_context_delta(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    feature_set = tmp_path / "feature_set"
    feature_set.mkdir()
    _write_smoke_dir(baseline, ev_shift=0.0, full_sl_shift=0.0)
    _write_smoke_dir(candidate, ev_shift=0.02, full_sl_shift=-0.05)
    pd.DataFrame(
        {
            "feature": ["ctx_a"],
            "min_monthly_non_null_share": [0.0],
            "mean_monthly_non_null_share": [0.5],
            "fully_missing_months": [1],
            "usable_months": [1],
        }
    ).to_csv(feature_set / "train_meta_interaction_context_feature_availability_summary.csv", index=False)

    manifest = run(
        baseline_dir=baseline,
        candidate_dir=candidate,
        feature_set_dir=feature_set,
        output_dir=tmp_path / "out",
        selectors=("s12_ev_clean_strong_risk",),
        reference_selector="s12_ev_clean_strong_risk",
    )

    aggregate_delta = pd.read_csv(manifest["outputs"]["aggregate_delta"])
    assert round(float(aggregate_delta["mean_delta_mean_ev_after_1pct"].iloc[0]), 8) == 0.02
    assert round(float(aggregate_delta["mean_delta_full_sl_rate"].iloc[0]), 8) == -0.05

    cell_summary = pd.read_csv(manifest["outputs"]["cell_delta_summary"])
    assert set(cell_summary["side_name"]) == {"long", "short"}
    assert "mean_delta_precision" in cell_summary.columns
    assert Path(manifest["outputs"]["selector_delta_vs_reference_summary"]).exists()
    assert Path(manifest["outputs"]["report"]).exists()
