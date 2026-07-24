from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.report_weighted_packb_july_replay_layers import (
    EV_LAYER,
    POLICY_LAYER,
    PORTFOLIO_LAYER,
    RAW_LAYER,
    build_report,
    main,
    parse_args,
)


def _candidates(score_column: str, outcomes: list[float] | None) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "timestamp": ["2026-07-01T00:00:00Z"] * 10 + ["2026-07-02T00:00:00Z"] * 10,
            "side_name": ["long", "short"] * 10,
            "archetype_policy_key": ["a", "b"] * 10,
            score_column: list(range(10)) * 2,
        }
    )
    if outcomes is not None:
        frame["net_return"] = outcomes
    return frame


def _args(tmp_path: Path, raw: Path, ev: Path, policy: Path, portfolio: Path):
    return parse_args(
        [
            "--raw-meta", str(raw), "--ev-mapped", str(ev), "--policy-execution", str(policy),
            "--portfolio-decisions", str(portfolio), "--output-dir", str(tmp_path / "output"),
        ]
    )


def test_layer_tables_select_top10_and_filter_portfolio(tmp_path: Path) -> None:
    raw, ev, policy, portfolio = (tmp_path / name for name in ("raw.csv", "ev.csv", "policy.csv", "portfolio.csv"))
    _candidates("meta_score", [0.0] * 9 + [0.10] + [0.0] * 9 + [-0.20]).to_csv(raw, index=False)
    _candidates("expected_ev_rank_score", [0.0] * 9 + [0.05] + [0.0] * 9 + [0.15]).to_csv(ev, index=False)
    pd.DataFrame({
        "timestamp": ["2026-07-01T00:00:00Z", "2026-07-02T00:00:00Z"],
        "side_name": ["long", "short"], "archetype_policy_key": ["a", "b"],
        "net_return": [0.10, -0.05], "simple_policy_exit_reason": ["trailing", "timeout"],
    }).to_csv(policy, index=False)
    pd.DataFrame({
        "timestamp": ["2026-07-01T00:00:00Z", "2026-07-01T02:00:00Z", "2026-07-02T00:00:00Z"],
        "side": [1, -1, 1], "policy_archetype": ["a", "b", "a"], "accepted": [True, False, True],
        "position_net_return": [0.10, 9.0, -0.20], "position_exit_reason": ["full_sl", "timeout", "trailing"],
    }).to_csv(portfolio, index=False)

    args = _args(tmp_path, raw, ev, policy, portfolio)
    tables = build_report(args)
    overall = tables["overall"].set_index("layer")
    assert overall.loc[RAW_LAYER, "rows"] == 2
    assert overall.loc[RAW_LAYER, "trades_per_day"] == pytest.approx(1.0)
    assert overall.loc[RAW_LAYER, "sum_net_return"] == pytest.approx(-0.10)
    assert overall.loc[EV_LAYER, "sum_net_return"] == pytest.approx(0.20)
    assert overall.loc[POLICY_LAYER, "timeout_rate"] == pytest.approx(0.5)
    assert overall.loc[PORTFOLIO_LAYER, "rows"] == 2
    assert overall.loc[PORTFOLIO_LAYER, "stop_trades"] == pytest.approx(1.0)
    assert overall.loc[PORTFOLIO_LAYER, "worst_day_net_return"] == pytest.approx(-0.20)
    assert {"utc_week_start", "archetype"}.issubset(tables["by_utc_week_archetype"].columns)
    assert set(tables["by_side"]["side"]) == {"long", "short"}
    assert tables["field_mappings"].set_index("layer").loc[PORTFOLIO_LAYER, "include"] == "accepted"

    assert main([
        "--raw-meta", str(raw), "--ev-mapped", str(ev), "--policy-execution", str(policy),
        "--portfolio-decisions", str(portfolio), "--output-dir", str(tmp_path / "written"),
    ]) == 0
    for name in ("overall", "by_utc_week", "by_archetype", "by_utc_week_archetype", "by_side", "field_mappings"):
        assert (tmp_path / "written" / f"weighted_packb_july_replay_layers_{name}.csv").exists()


def test_missing_outcomes_are_nan_and_bad_override_fails(tmp_path: Path) -> None:
    raw, ev, policy, portfolio = (tmp_path / name for name in ("raw.csv", "ev.csv", "policy.csv", "portfolio.csv"))
    _candidates("meta_score", None).to_csv(raw, index=False)
    _candidates("expected_ev_rank_score", [0.0] * 9 + [0.10] + [0.0] * 9 + [0.20]).to_csv(ev, index=False)
    pd.DataFrame({"timestamp": ["2026-07-01T00:00:00Z"], "side_name": ["long"], "archetype_policy_key": ["a"]}).to_csv(policy, index=False)
    pd.DataFrame({"timestamp": ["2026-07-01T00:00:00Z"], "side_name": ["long"], "archetype_policy_key": ["a"], "accepted": [True]}).to_csv(portfolio, index=False)
    args = _args(tmp_path, raw, ev, policy, portfolio)
    raw_row = build_report(args)["overall"].set_index("layer").loc[RAW_LAYER]
    assert raw_row["trades"] == 2
    assert pd.isna(raw_row["mean_net_return"])
    assert pd.isna(raw_row["hit_rate"])
    assert pd.isna(raw_row["stop_rate"])

    bad = parse_args([
        "--raw-meta", str(raw), "--ev-mapped", str(ev), "--policy-execution", str(policy),
        "--portfolio-decisions", str(portfolio), "--output-dir", str(tmp_path / "bad"),
        "--raw-meta-score-col", "not_a_column",
    ])
    with pytest.raises(ValueError, match="not_a_column"):
        build_report(bad)
