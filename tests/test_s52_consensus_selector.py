from __future__ import annotations

import pandas as pd

from scripts import report_s52_consensus_selector as consensus


def test_top_fraction_mask_keeps_requested_fraction_by_group():
    score = pd.Series([10.0, 9.0, 1.0, 8.0, 7.0, 0.0])
    groups = pd.Series(["a", "a", "a", "b", "b", "b"])

    mask = consensus._top_fraction_mask(score, 0.5, groups)

    assert mask.tolist() == [True, True, False, True, True, False]


def test_admitted_score_keeps_rejected_rows_finite_but_lower():
    primary = pd.Series([1.0, 2.0, 3.0, 4.0])
    gate = pd.Series([4.0, 1.0, 3.0, 2.0])

    score, admit = consensus._admitted_score(primary, gate, gate_fraction=0.5, groups=None)

    assert admit.tolist() == [True, False, True, False]
    assert score.notna().all()
    assert score.iloc[1] < score.iloc[0]
    assert score.iloc[3] < score.iloc[2]


def test_consensus_selector_report_runs_on_minimal_ledger(tmp_path):
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
                rows.append(
                    {
                        "variant": variant,
                        "month": month,
                        "__ts__": pd.Timestamp(f"{month}-01") + pd.Timedelta(hours=i),
                        "__symbol__": f"SYM{i % 10}/USD:USD",
                        "score": float(i if variant == "pointwise_lgbm" else 80 - i),
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

    consensus.run(
        ledger_path=ledger_path,
        output_dir=output_dir,
        round_trip_cost=0.01,
        weights=[0.0, 0.5, 1.0],
        normalization="global",
        gate_fractions=[0.5],
        gate_scores=["ranker_timestamp_side_fullpath_evpath"],
        max_primary_variants=2,
    )

    summary = pd.read_csv(output_dir / "s52_consensus_selector_summary.csv")
    assert not summary.empty
    assert (output_dir / "s52_consensus_selector_admission.csv").exists()
    assert (output_dir / "s52_consensus_selector.md").exists()
