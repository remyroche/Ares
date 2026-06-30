from pathlib import Path

import pandas as pd

from scripts.analyze_size_action_missed_oracle import (
    DEFAULT_ARM,
    analyze_missed_oracle,
    live_feature_columns,
)


def test_live_feature_columns_exclude_counterfactual_labels() -> None:
    frame = pd.DataFrame(
        {
            "fold_id": [0],
            "timestamp": ["2026-06-01"],
            "strategy_id": ["short_asset"],
            "multiplier": [1.0],
            "wallet": [1000.0],
            "strategy_rank_q90": [0.9],
            "best_gain": [10.0],
            "group_can_bind": [1],
            "delta_full_J": [5.0],
            "y_intervene": [1],
            "oracle_best_delta_full_J": [5.0],
            "selected_delta_full_J": [5.0],
        }
    )

    columns = live_feature_columns(frame)

    assert "wallet" in columns
    assert "strategy_rank_q90" in columns
    assert "best_gain" not in columns
    assert "group_can_bind" not in columns
    assert "delta_full_J" not in columns
    assert "y_intervene" not in columns
    assert "oracle_best_delta_full_J" not in columns
    assert "selected_delta_full_J" not in columns


def test_analyze_missed_oracle_outputs_population_and_auc(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    run_dir.mkdir()

    panel_rows = []
    diag_rows = []
    for idx in range(12):
        if idx < 4:
            population = "selected"
            selected = True
            selected_delta = 10.0
            missed = False
            oracle_delta = 10.0
            oracle_multiplier = 0.5
            live_signal = 0.8 + idx * 0.01
        elif idx < 8:
            population = "missed"
            selected = False
            selected_delta = 0.0
            missed = True
            oracle_delta = 8.0
            oracle_multiplier = 0.5
            live_signal = 0.9 + idx * 0.01
        else:
            population = "noop"
            selected = False
            selected_delta = 0.0
            missed = False
            oracle_delta = 0.0
            oracle_multiplier = 1.0
            live_signal = 0.1 + idx * 0.01
        timestamp = f"2026-06-{idx + 1:02d}"
        strategy = "short_asset" if idx % 2 == 0 else "short_boll"
        panel_rows.append(
            {
                "fold_id": idx % 3,
                "timestamp": timestamp,
                "strategy_id": strategy,
                "multiplier": 1.0,
                "wallet": 1000.0,
                "live_signal": live_signal,
                "best_gain": 999.0,
                "delta_full_J": 999.0,
                "group_can_bind": 1,
                "y_intervene": 1 if population != "noop" else 0,
            }
        )
        diag_rows.append(
            {
                "arm": DEFAULT_ARM,
                "fold_id": idx % 3,
                "timestamp": timestamp,
                "strategy_id": strategy,
                "selected": selected,
                "selected_delta_full_J": selected_delta,
                "missed_positive_oracle": missed,
                "oracle_best_delta_full_J": oracle_delta,
                "oracle_best_multiplier": oracle_multiplier,
                "p_intervene": live_signal,
                "pred_delta_J": live_signal * 10.0,
            }
        )

    pd.DataFrame(panel_rows).to_csv(run_dir / "size_action_exact_panel.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(run_dir / "size_action_selector_transfer_diagnostics.csv", index=False)

    payload = analyze_missed_oracle(run_dir=run_dir, arm=DEFAULT_ARM, out_dir=out_dir)

    counts = {row["population"]: row["groups"] for row in payload["population_counts"]}
    assert counts == {
        "missed_oracle_positive": 4,
        "non_actionable": 4,
        "selected_positive": 4,
    }
    auc = pd.read_csv(out_dir / "missed_oracle_feature_auc.csv")
    live_signal = auc.loc[auc["feature"] == "live_signal"].iloc[0]
    assert live_signal["separation_auc"] > 0.99
    diffs = pd.read_csv(out_dir / "missed_oracle_feature_differences.csv")
    assert "best_gain" not in set(diffs["feature"])
    assert (out_dir / "missed_oracle_diagnostic.md").exists()
