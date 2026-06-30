from pathlib import Path

import pandas as pd

from scripts.diagnose_c3el_interventions import _load_exact_support, main


def test_diagnose_c3el_interventions_reports_loss_avoided(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ts = pd.Timestamp("2026-06-15 12:00:00", tz="UTC")
    strategy = "short_asset_example"

    pd.DataFrame(
        {
            "timestamp": [ts],
            "strategy_id": [strategy],
            "head": ["short_asset"],
            "multiplier": [0.0],
        }
    ).to_csv(run_dir / "head_native_size_schedule.csv", index=False)
    pd.DataFrame(
        {
            "timestamp": [ts],
            "strategy_id": [strategy],
            "head": ["short_asset"],
            "p_intervene": [0.91],
            "pred_action_delta_J": [400.0],
            "selected_multiplier": [0.0],
            "gate_keep": [True],
            "week_start": [ts],
        }
    ).to_csv(run_dir / "head_native_group_scores.csv", index=False)
    pd.DataFrame(
        {
            "arm": ["C0_baseline", "C3el_head_native"],
            "timestamp": [ts, ts],
            "strategy_id": [strategy, strategy],
            "symbol": ["BTC/USD:USD", "BTC/USD:USD"],
            "net_pnl": [-100.0, -25.0],
            "gross_pnl": [-80.0, -20.0],
            "cost_pnl": [20.0, 5.0],
            "position_size": [1000.0, 250.0],
            "net_win": [0, 0],
            "full_sl": [1, 1],
            "timeout": [0, 0],
        }
    ).to_csv(run_dir / "accepted_trades.csv", index=False)
    exact_panel = tmp_path / "exact.csv"
    pd.DataFrame(
        {
            "timestamp": [ts],
            "strategy_id": [strategy],
            "multiplier": [0.0],
            "action_binds": [1.0],
            "delta_full_J": [75.0],
            "delta_immediate_J": [60.0],
        }
    ).to_csv(exact_panel, index=False)
    out_dir = tmp_path / "out"

    monkeypatch.setattr(
        "sys.argv",
        [
            "diagnose_c3el_interventions.py",
            "--run-dir",
            str(run_dir),
            "--exact-action-panels",
            str(exact_panel),
            "--out-dir",
            str(out_dir),
        ],
    )

    main()

    detail = pd.read_csv(out_dir / "intervention_diagnostics.csv")
    assert len(detail) == 1
    row = detail.iloc[0]
    assert row["direct_delta_net_pnl"] == 75.0
    assert row["loss_avoided"] == 75.0
    assert row["winner_pnl_sacrificed"] == 0.0
    assert row["defensive_success"] == 75.0
    assert row["exact_positive_e50_rows"] == 1

    by_head = pd.read_csv(out_dir / "intervention_summary_by_head.csv")
    assert by_head.loc[0, "positive_direct_delta_rate"] == 1.0
    assert by_head.loc[0, "delta_full_sl"] == 0


def test_load_exact_support_accepts_action_value_alias(tmp_path: Path) -> None:
    ts = pd.Timestamp("2026-06-15 12:00:00", tz="UTC")
    exact_panel = tmp_path / "exact_action_value.csv"
    pd.DataFrame(
        {
            "timestamp": [ts],
            "strategy_id": ["short_asset_example"],
            "action_value": [0.0],
            "action_binds": [1.0],
            "delta_full_J": [75.0],
            "delta_immediate_J": [60.0],
        }
    ).to_csv(exact_panel, index=False)

    out = _load_exact_support([exact_panel])

    assert len(out) == 1
    assert out.loc[0, "exact_positive_e50_rows"] == 1
    assert out.loc[0, "exact_best_delta_full_J"] == 75.0
