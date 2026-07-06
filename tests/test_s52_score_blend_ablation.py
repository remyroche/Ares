from __future__ import annotations

import pandas as pd

from scripts import report_s52_score_blend_ablation as blend


def test_score_blend_report_runs_on_minimal_ledger(tmp_path):
    rows = []
    variants = (
        "pointwise_lgbm",
        "ranker_timestamp_side_fullpath_evpath",
        "ranker_timestamp_side_soft_ordered_ev",
    )
    for month in ("2026-04", "2026-05"):
        for i in range(80):
            side = 1 if i % 2 == 0 else -1
            good = int(i % 5 == 0)
            bad = int(not good and i % 7 == 0)
            for variant in variants:
                score = float(i)
                if variant == "ranker_timestamp_side_fullpath_evpath":
                    score = float(80 - i)
                elif variant == "ranker_timestamp_side_soft_ordered_ev":
                    score = float(i % 11)
                rows.append(
                    {
                        "variant": variant,
                        "month": month,
                        "__ts__": pd.Timestamp(f"{month}-01") + pd.Timedelta(hours=i),
                        "__symbol__": f"SYM{i % 10}/USD:USD",
                        "score": score,
                        "target_soft": float(good),
                        "target_hard": good,
                        "first_pass_good": good,
                        "first_pass_bad": bad,
                        "side_name": "long" if side > 0 else "short",
                        "u_policy_net": 0.02 if good else -0.01,
                        "ret_net": 0.02 if good else -0.01,
                        "side": side,
                        "is_timeout": 0,
                        "mae_norm": 0.4 if good else 1.2,
                        "mfe_norm": 1.4 if good else 0.2,
                        "first_touch_net": 0.01 if good else -0.01,
                        "first_touch_mae_norm": 0.4 if good else 1.2,
                        "first_touch_mfe_norm": 1.4 if good else 0.2,
                        "first_touch_full_path_mae_norm": 0.6 if good else 1.4,
                        "mfe_1r_before_mae_1r": good,
                        "mae_1r_before_mfe_1r": bad,
                        "max_adverse_before_mfe_1r": 0.4 if good else 1.4,
                        "underwater_bars_before_mfe_1r": 1 if good else 8,
                        "underwater_fraction_before_mfe_1r": 0.1 if good else 0.8,
                    }
                )
    ledger_path = tmp_path / "ledger.parquet"
    pd.DataFrame(rows).to_parquet(ledger_path)
    output_dir = tmp_path / "out"

    blend.run(
        ledger_path=ledger_path,
        output_dir=output_dir,
        round_trip_cost=0.01,
        weights=[0.0, 0.5, 1.0],
        normalization="global",
    )

    summary = pd.read_csv(output_dir / "s52_score_blend_summary.csv")
    assert not summary.empty
    assert "blend3::w050_pointwise_lgbm+path_heads" in set(summary["variant"])
    assert (output_dir / "s52_score_blend_ablation.md").exists()
