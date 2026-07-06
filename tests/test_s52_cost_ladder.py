from __future__ import annotations

import pandas as pd

from scripts import report_s52_cost_ladder as ladder


def test_cost_ladder_adjusts_original_net_columns():
    base = pd.DataFrame(
        {
            "first_touch_net": [-0.002, 0.003],
            "u_policy_net": [-0.004, 0.002],
            "ret_net": [-0.005, 0.001],
        }
    )
    adjusted = ladder._adjust_cost(base, original_cost=0.01, new_cost=0.0025)

    assert adjusted["first_touch_net"].round(6).tolist() == [0.0055, 0.0105]
    assert adjusted["u_policy_net"].round(6).tolist() == [0.0035, 0.0095]
    assert adjusted["ret_net"].round(6).tolist() == [0.0025, 0.0085]


def test_cost_ladder_report_runs_on_minimal_ledger(tmp_path):
    rows = []
    variants = (
        "pointwise_lgbm",
        "ranker_timestamp_side_fullpath_evpath",
        "ranker_timestamp_side_soft_ordered_ev",
    )
    for month in ("2026-04", "2026-05"):
        for i in range(70):
            side = 1 if i % 2 == 0 else -1
            good = int(i % 4 == 0)
            bad = int(not good and i % 6 == 0)
            for variant in variants:
                rows.append(
                    {
                        "variant": variant,
                        "month": month,
                        "__ts__": pd.Timestamp(f"{month}-01") + pd.Timedelta(hours=i),
                        "__symbol__": f"SYM{i % 8}/USD:USD",
                        "score": float(i if variant == "pointwise_lgbm" else 70 - i),
                        "target_soft": float(good),
                        "target_hard": good,
                        "first_pass_good": good,
                        "first_pass_bad": bad,
                        "side_name": "long" if side > 0 else "short",
                        "u_policy_net": 0.005 if good else -0.015,
                        "ret_net": 0.005 if good else -0.015,
                        "side": side,
                        "is_timeout": 0,
                        "mae_norm": 0.4 if good else 1.2,
                        "mfe_norm": 1.4 if good else 0.2,
                        "first_touch_net": 0.004 if good else -0.012,
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

    ladder.run(
        ledger_path=ledger_path,
        output_dir=output_dir,
        original_cost=0.01,
        costs=[0.0, 0.005, 0.01],
        weights=[0.0, 0.5, 1.0],
        normalization="global",
        max_variants_per_cost=4,
        selection_metric="objective",
    )

    summary = pd.read_csv(output_dir / "s52_cost_ladder_summary.csv")
    assert sorted(summary["cost_bps"].unique().tolist()) == [0.0, 50.0, 100.0]
    assert (output_dir / "s52_cost_ladder.md").exists()


def test_cost_ladder_can_shortlist_by_ev_weighted_precision():
    summary = pd.DataFrame(
        [
            {
                "variant": "objective_best",
                "objective": 10.0,
                "mean_top10_ev_weighted_first_touch_precision": 0.10,
            },
            {
                "variant": "ev_precision_best",
                "objective": -1.0,
                "mean_top10_ev_weighted_first_touch_precision": 0.90,
            },
        ]
    )

    selected = ladder._best_variants(
        summary,
        max_variants=1,
        selection_metric="mean_top10_ev_weighted_first_touch_precision",
    )

    assert selected == ["ev_precision_best"]
