#!/usr/bin/env python3
"""Matched T6/T9 router-input meta comparison, research only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))


START = pd.Timestamp("2026-04-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
FRACTIONS = (.01, .02, .05, .10)


def _score_metrics(frame: pd.DataFrame, score: str) -> dict[str, float | int]:
    values: dict[str, float | int] = {}
    top2: list[float] = []
    for fraction in FRACTIONS:
        ev, precision, rows = [], [], 0
        for _, part in frame.groupby("__decision_ts__", sort=False):
            selected = part.nlargest(max(1, int(np.ceil(len(part) * fraction))), score, keep="first")
            ev.append(float(selected["policy_net_bps"].mean()))
            precision.append(float(selected["policy_net_bps"].gt(50.0).mean()))
            rows += len(selected)
        key = f"top{int(fraction * 100):02d}"
        values[f"{key}_timestamp_net_bps"] = float(np.mean(ev))
        values[f"{key}_precision_gt50"] = float(np.mean(precision))
        values[f"{key}_rows"] = int(rows)
    for _, part in frame.groupby("__decision_ts__", sort=False):
        top2.append(float(part.nlargest(min(2, len(part)), score, keep="first")["policy_net_bps"].mean()))
    values["top2_timestamp_net_bps"] = float(np.mean(top2))
    values["timestamps"] = int(frame["__decision_ts__"].nunique())
    return values


def _daily_sortino(equity: pd.DataFrame) -> float:
    if not {"timestamp", "wallet"}.issubset(equity.columns):
        return float("nan")
    work = equity.loc[:, ["timestamp", "wallet"]].copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work["wallet"] = pd.to_numeric(work["wallet"], errors="coerce")
    work = work.dropna().sort_values("timestamp", kind="stable")
    work["day"] = work["timestamp"].dt.floor("D")
    ret = work.groupby("day", sort=True)["wallet"].last().pct_change().dropna().to_numpy(float)
    downside = float(np.sqrt(np.mean(np.minimum(ret, 0.0) ** 2))) if len(ret) else np.nan
    return float(np.sqrt(365.0) * np.mean(ret) / downside) if np.isfinite(downside) and downside > 0 else float("nan")


def _load_panel(root: Path, policy: pd.DataFrame, *, family: str) -> pd.DataFrame:
    parts = [pd.read_parquet(path) for path in sorted((root / "target_free_scores" / family).glob("month=*.parquet"))]
    result = pd.concat(parts, ignore_index=True)
    result["__decision_ts__"] = pd.to_datetime(result["__decision_ts__"], utc=True, errors="raise")
    result = result.loc[result["__decision_ts__"].ge(START) & result["__decision_ts__"].lt(END)].copy()
    result = result.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    valid = result["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(pd.to_numeric(result["policy_net_bps"], errors="coerce"))
    return result.loc[valid].copy()


def _read_arm(label: str, root: Path, policy: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    bcf = _load_panel(root, policy, family="bcf")
    current = _load_panel(root, policy, family="current")
    row: dict[str, object] = {"arm": label, "rows": int(len(bcf)), **_score_metrics(bcf, "final_score")}
    row.update({f"current_{key}": value for key, value in _score_metrics(current, "final_score").items()})
    stored = pd.read_parquet(root / "portfolio_metrics.parquet")
    metric = stored.loc[pd.to_numeric(stored["threshold_bps"], errors="coerce").eq(50.0)].iloc[0].to_dict()
    row.update({f"portfolio_{key}": value for key, value in metric.items()})
    equity = pd.read_parquet(root / "routed_base_dual_50_2026_marjul_equity.parquet")
    row["portfolio_sortino_daily_annualized"] = _daily_sortino(equity)
    heads = [field for field in bcf.columns if field.startswith("head__") and field.endswith("__rank")]
    head_rows = []
    for head in ["final_score", *sorted(heads)]:
        head_rows.append({"arm": label, "head": head, **_score_metrics(bcf, head)})
    return row, pd.DataFrame(head_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-router-root", type=Path, required=True)
    parser.add_argument("--meta-router-root", type=Path, required=True)
    parser.add_argument("--old-control-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    policy = pd.read_parquet(args.policy_path, columns=["candidate_id", "policy_path_valid", "policy_net_bps"])
    # The reporting utility is used for both the historical full-base study
    # and the routed-base study.  Keep the arm labels neutral so a routed-only
    # result cannot be misread as the older full-base experiment.
    primary = (
        ("T6T9_without_router_inputs", args.no_router_root),
        ("T6T9_with_router_inputs", args.meta_router_root),
    )
    rows, heads = [], []
    for label, root in (*primary, ("diagnostic_old_routedbase_T6T9", args.old_control_root)):
        row, per_head = _read_arm(label, root, policy)
        rows.append(row)
        heads.append(per_head)
    table = pd.DataFrame(rows)
    per_head = pd.concat(heads, ignore_index=True)
    table.to_parquet(args.out / "meta_variant_metrics.parquet", index=False, compression="zstd")
    per_head.to_parquet(args.out / "meta_per_head_metrics.parquet", index=False, compression="zstd")
    # Selection is intentionally limited to the exact-base primary pair.  A
    # raw score gain is not sufficient: this stack is consumed by a causal
    # mapper and a capacity-constrained portfolio.  Retain only a candidate
    # that keeps top-10 precision within one point *and* does not lower either
    # constrained EV/trade or total realised contribution versus the matched
    # no-router control.  Rank the surviving arms by top-two score quality,
    # then Sortino and drawdown.
    primary_table = table.loc[table["arm"].isin([item[0] for item in primary])].copy()
    control_precision = float(primary_table.loc[
        primary_table["arm"].eq("T6T9_without_router_inputs"), "top10_precision_gt50"
    ].iloc[0])
    control_row = primary_table.loc[
        primary_table["arm"].eq("T6T9_without_router_inputs")
    ].iloc[0]
    eligible = primary_table.loc[
        primary_table["top10_precision_gt50"].ge(control_precision - .01)
        & primary_table["portfolio_net_ev_bps_per_realised_trade"].ge(
            float(control_row["portfolio_net_ev_bps_per_realised_trade"])
        )
        & primary_table["portfolio_net_sum_bps_realised"].ge(
            float(control_row["portfolio_net_sum_bps_realised"])
        )
    ].copy()
    if eligible.empty:
        raise AssertionError("no meta arm retained the one-point top10 precision floor")
    winner = eligible.sort_values(
        ["top2_timestamp_net_bps", "portfolio_sortino_daily_annualized", "portfolio_max_drawdown", "portfolio_net_ev_bps_per_realised_trade"],
        ascending=[False, False, False, False], kind="stable",
    ).iloc[0]
    selection = {
        "selection_rule": "top10 >50-bps precision within 1pp plus no degradation of constrained EV/trade or total realised bps versus no-router control; then BCF top2 EV, daily Sortino, lower drawdown",
        "winner": str(winner["arm"]),
        "control_top10_precision_gt50": control_precision,
        "winner_metrics": winner.to_dict(),
        "period": "2026-04 through 2026-07; caller-defined matched base and exact router top-50% source",
        "old_control_note": "third arm is descriptive only and not eligible for selection",
    }
    args.out.joinpath("meta_selection.json").write_text(json.dumps(selection, indent=2, default=str) + "\n")
    args.out.joinpath("run_manifest.json").write_text(json.dumps({
        "scope": "offline research only", "policy": str(args.policy_path), "selection": selection,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
