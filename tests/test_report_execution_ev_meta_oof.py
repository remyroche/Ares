from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "report_execution_ev_meta_oof", ROOT / "scripts" / "report_execution_ev_meta_oof.py"
)
assert SPEC and SPEC.loader
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


def _ledger() -> pd.DataFrame:
    rows = 40
    ts = pd.date_range("2026-01-01", periods=rows, freq="12h", tz="UTC")
    net = np.linspace(-0.02, 0.03, rows)
    return pd.DataFrame(
        {
            "__ts__": ts,
            "__symbol__": np.where(np.arange(rows) % 2, "ETH/USD:USD", "BTC/USD:USD"),
            "side_name": np.where(np.arange(rows) % 2, "long", "short"),
            "candidate_id": [f"c-{i}" for i in range(rows)],
            "catboost_archetype": np.where(np.arange(rows) % 3, "trend", "reversal"),
            "execution_net_ev_12h": net,
            "execution_gross_ev_12h": net + 0.002,
            "execution_exit_reason": np.where(np.arange(rows) % 7 == 0, "stop", np.where(np.arange(rows) % 5 == 0, "timeout", "take_profit")),
            "clean": np.arange(rows) % 2 == 0,
            "dirty": np.arange(rows) % 2 == 1,
            "signed_hit_rate_surprise": np.linspace(-0.2, 0.2, rows),
            "portfolio_pnl": net * 100.0,
            "position_size": np.full(rows, 1000.0),
            "execution_ev_oof_fold": np.where(np.arange(rows) < 4, -1, 0),
            "direct__all_features": net + 0.001,
            "residual__all_features": net - 0.001,
            "direct__all_features__is_oof": np.arange(rows) >= 4,
            "residual__all_features__is_oof": np.arange(rows) >= 4,
        }
    )


def test_wide_ledger_reports_required_oof_tail_and_execution_metrics(tmp_path: Path) -> None:
    path = tmp_path / "ledger.parquet"
    _ledger().to_parquet(path, index=False)
    result = reporter.run_report([path], tmp_path / "report")
    direct = result["metrics"]["direct__all_features"]["overall"]["all"]
    assert direct["rows"] == 36
    assert direct["top_10pct_rows"] == 4
    assert direct["top_10pct_mean_net_ev_per_trade"] > 0.0
    assert np.isfinite(direct["signed_residual_mean"])
    assert direct["hit_rate_surprise"]["available"] is True
    assert direct["hit_rate_surprise"]["signed_rows"] == 36
    assert direct["hit_rate_surprise"]["positive_component_rows"] > 0
    assert direct["hit_rate_surprise"]["negative_component_rows"] > 0
    assert np.isclose(direct["hit_rate_surprise"]["signed_lag1_autocorrelation"], 1.0)
    assert direct["hit_rate_surprise"]["positive_component_lag1_autocorrelation"] > 0.9
    assert direct["hit_rate_surprise"]["negative_component_lag1_autocorrelation"] > 0.9
    assert direct["hit_rate_surprise"]["positive_component_support_rows"] == 36
    assert direct["hit_rate_surprise"]["negative_component_support_rows"] == 36
    assert direct["hit_rate_surprise"]["negative_component_mean"] < 0.0
    assert direct["bankroll"]["available"] is True
    assert direct["stop_rate"] > 0.0
    assert direct["timeout_rate"] > 0.0
    assert "worst_week_top_10pct_mean_net_ev" in direct
    assert "worst_month_top_30pct_mean_net_ev" in direct
    assert set(result["metrics"]["direct__all_features"]) >= {"month", "week", "side", "base_archetype"}
    global_top10 = result["metrics"]["direct__all_features"]["global_top_tail_breakdown"]["top_10pct"]
    assert global_top10["selection_basis"]["selection_scope"] == "global_oof_rows_within_arm"
    assert global_top10["selection_basis"]["ordering"] == "prediction_descending"
    assert global_top10["overall"]["selected_rows"] == 4
    assert sum(item["selected_rows"] for item in global_top10["month"].values()) == 4
    assert sum(item["selected_rows"] for item in global_top10["week"].values()) == 4
    assert sum(item["selected_rows"] for item in global_top10["side"].values()) == 4
    assert sum(item["selected_rows"] for item in global_top10["base_archetype"].values()) == 4
    assert result["arm_comparisons"]
    assert result["paths"]["json"].is_file()
    payload = json.loads(result["paths"]["json"].read_text())
    assert payload["manifest"]["status"] == "evaluation_only_not_policy_selection"


def test_rank_and_long_arm_input_are_supported_and_unsized_bankroll_is_explicit(tmp_path: Path) -> None:
    frame = _ledger().iloc[4:].copy()
    frame = frame.loc[:, ["__ts__", "side_name", "catboost_archetype", "execution_net_ev_12h", "execution_gross_ev_12h", "execution_ev_oof_fold"]]
    left = frame.assign(arm="direct", prediction=np.linspace(0.0, 1.0, len(frame)), rank=np.arange(len(frame), 0, -1))
    right = frame.assign(arm="residual", prediction=np.linspace(1.0, 0.0, len(frame)), rank=np.arange(len(frame)))
    path = tmp_path / "long.csv"
    pd.concat([left, right], ignore_index=True).to_csv(path, index=False)
    result = reporter.run_report(
        [path], tmp_path / "long-report", arm_col="arm", prediction_col="prediction", rank_col="rank"
    )
    direct = result["metrics"]["direct"]["overall"]["all"]
    assert direct["top_1pct_rows"] == 1
    global_top1 = result["metrics"]["direct"]["global_top_tail_breakdown"]["top_1pct"]
    assert global_top1["selection_basis"]["ordering"] == "rank_ascending"
    assert global_top1["overall"]["selected_rows"] == 1
    assert direct["bankroll"]["available"] is False
    assert "requires both" in direct["bankroll"]["reason"]
    assert {row["left_arm"] for row in result["arm_comparisons"]} == {"direct"}


def test_rejects_ledger_without_explicit_oof_evidence(tmp_path: Path) -> None:
    frame = _ledger().drop(columns=["execution_ev_oof_fold", "direct__all_features__is_oof", "residual__all_features__is_oof"])
    path = tmp_path / "not-oof.parquet"
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="no explicit OOF indicator"):
        reporter.run_report([path], tmp_path / "invalid")
