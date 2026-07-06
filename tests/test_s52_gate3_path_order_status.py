from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_s52_gate3_path_order_status import build_report


def test_s52_gate3_status_flags_long_side_failure_and_warnings(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    pd.DataFrame(
        [
            {
                "variant": "ranker_side_specific_timestamp",
                "objective": 0.7,
                "mean_top10_ev_weighted_first_touch_precision": 0.76,
                "mean_top20_ev_weighted_first_touch_precision": 0.71,
                "mean_top30_ev_weighted_first_touch_precision": 0.66,
                "mean_top10_first_touch_bad_mae_1r_rate": 0.21,
                "mean_top10_first_touch_full_path_bad_mae_1r_rate": 0.62,
                "mean_top10_timeout_rate": 0.04,
                "mean_top10_mean_first_touch_mae_norm": 0.84,
                "mean_top10_p90_first_touch_mae_norm": 1.98,
                "mean_top10_p90_first_touch_full_path_mae_norm": 7.25,
                "mean_top10_mean_max_adverse_before_mfe_1r": 1.18,
                "mean_top10_mean_underwater_fraction_before_mfe": 0.32,
                "mean_top10_mean_underwater_bars_before_mfe": 7.0,
                "mean_top10_mfe_1r_before_mae_1r_rate": 0.71,
                "mean_top10_mae_1r_before_mfe_1r_rate": 0.26,
                "mean_top10_mean_u": 0.001,
                "mean_top10_bad_mae_rate": 0.69,
                "mean_long_top10_mae_1r_before_mfe_1r_rate": 0.39,
                "mean_short_top10_mae_1r_before_mfe_1r_rate": 0.23,
            }
        ]
    ).to_csv(input_dir / "s52_ranker_smoke_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "variant": "ranker_side_specific_timestamp",
                "month": "2026-04",
                "top10_p90_first_touch_mae_norm": 3.04,
                "top10_first_touch_full_path_bad_mae_1r_rate": 0.66,
                "top10_p90_first_touch_full_path_mae_norm": 8.5,
                "top10_mean_max_adverse_before_mfe_1r": 1.65,
                "top10_mean_underwater_fraction_before_mfe": 0.49,
                "top10_mean_underwater_bars_before_mfe": 11.2,
                "top10_mae_1r_before_mfe_1r_rate": 0.36,
                "top10_mean_u": 0.0002,
            },
            {
                "variant": "pointwise_lgbm",
                "month": "2026-04",
                "top10_p90_first_touch_mae_norm": 9.0,
                "top10_first_touch_full_path_bad_mae_1r_rate": 0.95,
                "top10_p90_first_touch_full_path_mae_norm": 20.0,
                "top10_mean_max_adverse_before_mfe_1r": 6.0,
                "top10_mean_underwater_fraction_before_mfe": 0.90,
                "top10_mean_underwater_bars_before_mfe": 40.0,
                "top10_mae_1r_before_mfe_1r_rate": 0.80,
                "top10_mean_u": -0.20,
            }
        ]
    ).to_csv(input_dir / "s52_ranker_smoke_folds.csv", index=False)
    pd.DataFrame(
        [
            {
                "variant": "ranker_side_specific_timestamp",
                "state_feature": "gmm_cluster_id",
                "bucket": "0",
                "side": "long",
                "rows": 100,
                "selected_rows": 40,
                "selected_mfe_before_mae_1r_rate": 0.36,
                "selected_mae_before_mfe_1r_rate": 0.61,
                "selected_mean_first_touch_mae_norm": 1.3,
                "selected_p90_first_touch_mae_norm": 3.8,
                "selected_first_touch_full_path_bad_mae_1r_rate": 0.75,
                "selected_p90_first_touch_full_path_mae_norm": 9.1,
                "selected_mean_u": -0.007,
            },
            {
                "variant": "ranker_side_specific_timestamp",
                "state_feature": "gmm_cluster_id",
                "bucket": "2",
                "side": "long",
                "rows": 100,
                "selected_rows": 32,
                "selected_mfe_before_mae_1r_rate": 0.75,
                "selected_mae_before_mfe_1r_rate": 0.25,
                "selected_mean_first_touch_mae_norm": 0.65,
                "selected_p90_first_touch_mae_norm": 1.8,
                "selected_first_touch_full_path_bad_mae_1r_rate": 0.32,
                "selected_p90_first_touch_full_path_mae_norm": 2.6,
                "selected_mean_u": 0.04,
            },
        ]
    ).to_csv(input_dir / "s52_ranker_smoke_archetype_path_diagnostics.csv", index=False)
    (input_dir / "manifest.json").write_text(json.dumps({"scope": "s52_timestamp_side_ranker_smoke"}))

    pipeline = tmp_path / "lgbm_pipeline.py"
    pipeline.write_text("# no native ranker here\n")

    decision = build_report(input_dir=input_dir, output_dir=output_dir, lgbm_pipeline_path=pipeline)

    assert decision["status"] == "fail"
    assert decision["production_ranker_materialization"]["status"] == "blocked"
    checks = pd.read_csv(output_dir / "s52_gate3_path_order_checks.csv")
    long_check = checks[checks["metric"].eq("long_top10_mae_1r_before_mfe_1r_rate")].iloc[0]
    assert long_check["status"] == "fail"
    full_path_check = checks[checks["metric"].eq("top10_post_exit_full_path_bad_mae_1r_rate")].iloc[0]
    assert full_path_check["status"] == "warn"
    assert not checks["scope"].astype(str).str.contains("pointwise_lgbm").any()
    assert not checks["value"].eq(-0.20).any()
    assert (checks["status"] == "warn").any()
    assert (output_dir / "s52_gate3_path_order_status.md").exists()
