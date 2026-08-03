#!/usr/bin/env python3
"""Diagnose the frozen selected P30 residual/reliability stack.

This is deliberately an *audit*, not an experiment runner.  It never refits a
model, changes a score, or uses outcomes to select a configuration.  It
describes the existing globally pooled score after its causal high-base-30%
admission gate.  In particular, all tail selections are one global OOS book;
monthly, weekly, side, day, and symbol tables are decompositions of that
already-selected book, not independently re-ranked sub-books.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SCORED = Path(
    "data_perp/artifacts/full_universe_round_final_selected_p30_audit_20260803_v1/"
    "scored_predictions.parquet"
)
DEFAULT_BASE_STATE = Path(
    "data_perp/artifacts/full_universe_base_state_diagnostics_20260803_v1/"
    "base_state_diagnostics.parquet"
)
TAILS = (0.001, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.20)
BUCKETS = ((0.0, 0.01), (0.01, 0.02), (0.02, 0.03), (0.03, 0.05), (0.05, 0.10))
DIST_COLUMNS = (
    "base_score",
    "meta_score",
    "residual_correction_bps",
    "p_upper",
    "p_lower",
    "p_timeout",
    "base_entropy_normalised",
    "base_top2_probability_margin",
    "base_payoff_mixture_sd_bps",
)


def _tail_name(value: float) -> str:
    return f"top_{value * 100:g}pct"


def _select(frame: pd.DataFrame, score: str, fraction: float) -> pd.DataFrame:
    """Deterministic global top-k; NaN scores sort below admissible scores."""
    n = int(np.ceil(len(frame) * fraction))
    return frame.sort_values([score, "candidate_id"], ascending=[False, True], na_position="last").head(n).copy()


def _metric(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "n": int(len(frame)),
        "gross_bps": float(frame.gross_bps.mean()),
        "net_bps": float(frame.net_bps.mean()),
        "net_pnl_bps_sum": float(frame.net_bps.sum()),
        "long_n": int(frame.side_name.eq("long").sum()),
        "short_n": int(frame.side_name.eq("short").sum()),
        "long_share": float(frame.side_name.eq("long").mean()),
    }


def _week_bootstrap(
    winner: pd.DataFrame, base: pd.DataFrame, draws: int, seed: int
) -> dict[str, float | int]:
    """Weekly-block paired bootstrap of fixed global winner/base selections."""
    winner_week = winner.groupby("week", observed=True).agg(net_sum=("net_bps", "sum"), n=("net_bps", "size"))
    base_week = base.groupby("week", observed=True).agg(net_sum=("net_bps", "sum"), n=("net_bps", "size"))
    weeks = winner_week.index.intersection(base_week.index).to_numpy()
    if len(weeks) < 2:
        return {"n_weeks": int(len(weeks))}
    w_sum = winner_week.loc[weeks, "net_sum"].to_numpy(float)
    w_n = winner_week.loc[weeks, "n"].to_numpy(float)
    b_sum = base_week.loc[weeks, "net_sum"].to_numpy(float)
    b_n = base_week.loc[weeks, "n"].to_numpy(float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(weeks), size=(draws, len(weeks)))
    w = w_sum[indices].sum(axis=1) / w_n[indices].sum(axis=1)
    b = b_sum[indices].sum(axis=1) / b_n[indices].sum(axis=1)
    lift = w - b
    return {
        "n_weeks": int(len(weeks)),
        "draws": int(draws),
        "winner_net_bps_p2_5": float(np.quantile(w, .025)),
        "winner_net_bps_p50": float(np.quantile(w, .50)),
        "winner_net_bps_p97_5": float(np.quantile(w, .975)),
        "base_net_bps_p2_5": float(np.quantile(b, .025)),
        "base_net_bps_p50": float(np.quantile(b, .50)),
        "base_net_bps_p97_5": float(np.quantile(b, .975)),
        "paired_lift_bps_p2_5": float(np.quantile(lift, .025)),
        "paired_lift_bps_p50": float(np.quantile(lift, .50)),
        "paired_lift_bps_p97_5": float(np.quantile(lift, .975)),
        "paired_probability_winner_improves": float((lift > 0.0).mean()),
    }


def _concentration(frame: pd.DataFrame, column: str) -> dict[str, float | int | str]:
    counts = frame[column].value_counts(dropna=False)
    shares = counts / counts.sum()
    return {
        f"unique_{column}": int(len(counts)),
        f"top_{column}": str(counts.index[0]),
        f"top_{column}_share": float(shares.iloc[0]),
        f"{column}_count_hhi": float(np.square(shares).sum()),
    }


def _quantile_bins(values: pd.Series, n: int = 10) -> pd.Series:
    """Tie-safe equal-count diagnostic bins.  Only used after OOS scoring."""
    ranks = values.rank(method="first", pct=True)
    return np.minimum((ranks * n).astype(int), n - 1) + 1


def _write_report(out: Path, summary: dict, tail: pd.DataFrame, buckets: pd.DataFrame) -> None:
    top10 = tail.loc[tail["tail"] == "top_10pct"].iloc[0]
    top1 = tail.loc[tail["tail"] == "top_1pct"].iloc[0]
    top01 = tail.loc[tail["tail"] == "top_0.1pct"].iloc[0]
    def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
        header = "| " + " | ".join(columns) + " |"
        divider = "|" + "|".join("---" for _ in columns) + "|"
        body = []
        for row in frame[columns].itertuples(index=False, name=None):
            values = []
            for value in row:
                if isinstance(value, (float, np.floating)):
                    values.append(f"{value:.2f}")
                else:
                    values.append(f"{value:,}" if isinstance(value, (int, np.integer)) else str(value))
            body.append("| " + " | ".join(values) + " |")
        return "\n".join([header, divider, *body])

    bucket_view = markdown_table(buckets, ["bucket", "n", "gross_bps", "net_bps", "long_share"])
    curve_view = markdown_table(tail, ["tail", "n", "gross_bps", "net_bps", "long_share", "unique_day", "unique_symbol"])
    b = summary["weekly_block_bootstrap"]["top_10pct"]
    report = f"""# Selected P30 trust diagnostics

## Contract

- Frozen selected P30 stack: residualised value plus shared cost-clear reliability rank blend.
- OOS rows: {summary['row_count']:,}, {summary['oos_start']} through {summary['oos_end']}.
- Admission: causal high-base P30; final winner score is undefined outside it.  Every tail below is still selected as a **single pooled global book** over the full OOS candidate population (NaN/ineligible scores sort last).  No table re-ranks by month, week, side, day, or symbol.
- This is a descriptive audit.  No output is used for model fitting or configuration selection.

## Readout

- Top 0.1% net: {top01['net_bps']:.2f} bps ({int(top01['n']):,} trades); top 1% net: {top1['net_bps']:.2f} bps ({int(top1['n']):,} trades).
- Top 10% net: {top10['net_bps']:.2f} bps ({int(top10['n']):,} trades); weekly-block 95% interval: [{b.get('winner_net_bps_p2_5', float('nan')):.2f}, {b.get('winner_net_bps_p97_5', float('nan')):.2f}] bps.
- Paired versus the frozen B2 mapped-base global top-10 book: median lift {b.get('paired_lift_bps_p50', float('nan')):.2f} bps, 95% interval [{b.get('paired_lift_bps_p2_5', float('nan')):.2f}, {b.get('paired_lift_bps_p97_5', float('nan')):.2f}], P(lift > 0) {b.get('paired_probability_winner_improves', float('nan')):.3f}.

## Global marginal tail curve

{curve_view}

## Marginal global rank buckets

{bucket_view}

## Files

- `tail_curve.parquet`: global tail outcomes, concentration and weekly intervals.
- `marginal_buckets.parquet`: mutually exclusive global-score buckets.
- `tail_month_side.parquet`, `tail_day.parquet`, `tail_symbol.parquet`: decompositions of fixed global selections.
- `tail_distributions.parquet`: requested base/reliability/residual/event-probability distributions.
- `side_reliability_calibration.parquet`, `side_base_value_calibration.parquet`, and `side_value_reliability_grid.parquet`: side-conditioned calibration diagnostics within the P30 reliability population.
"""
    (out / "TRUST_DIAGNOSTICS_REPORT.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored", type=Path, default=DEFAULT_SCORED)
    parser.add_argument("--base-state", type=Path, default=DEFAULT_BASE_STATE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()

    scored = pd.read_parquet(args.scored)
    state_columns = ["candidate_id", "p_upper", "p_lower", "p_timeout", "base_entropy_normalised", "base_top2_probability_margin"]
    state = pd.read_parquet(args.base_state, columns=state_columns)
    if scored.candidate_id.duplicated().any() or state.candidate_id.duplicated().any():
        raise ValueError("candidate_id must be unique in both frozen artifacts")
    data = scored.merge(state, on="candidate_id", how="left", validate="one_to_one")
    if data[state_columns[1:]].isna().any().any():
        raise ValueError("base-state join is incomplete; do not silently omit requested distributions")
    data["__ts__"] = pd.to_datetime(data["__ts__"], utc=True)
    data["day"] = data["__ts__"].dt.strftime("%Y-%m-%d")
    data["symbol"] = data.candidate_id.str.split("|", n=1).str[0]
    data["residual_correction_bps"] = data.value_score - data.base_score
    if not data.loc[data.high_base_eligible, "meta_score"].notna().all():
        raise ValueError("P30-admitted candidates lack reliability scores")
    if data.loc[~data.high_base_eligible, "winner_score"].notna().any():
        raise ValueError("winner score contract unexpectedly changed outside P30 admission")

    tail_rows: list[dict] = []
    bucket_rows: list[dict] = []
    month_side_rows: list[pd.DataFrame] = []
    day_rows: list[pd.DataFrame] = []
    symbol_rows: list[pd.DataFrame] = []
    dist_rows: list[dict] = []
    bootstrap: dict[str, dict] = {}
    # Sorting the 360k-row OOS book once per score keeps this diagnostic cheap
    # enough to rerun routinely; each tail is just a prefix of these orders.
    ranked = data.sort_values(["winner_score", "candidate_id"], ascending=[False, True], na_position="last")
    base_ranked = data.sort_values(["base_score", "candidate_id"], ascending=[False, True], na_position="last")
    for fraction in TAILS:
        name = _tail_name(fraction)
        n = int(np.ceil(len(data) * fraction))
        selected = ranked.iloc[:n].copy()
        base = base_ranked.iloc[:n].copy()
        row = {"tail": name, "fraction": fraction, **_metric(selected)}
        row.update(_concentration(selected, "month"))
        row.update(_concentration(selected, "day"))
        row.update(_concentration(selected, "symbol"))
        tail_rows.append(row)
        bootstrap[name] = _week_bootstrap(selected, base, args.bootstrap_draws, args.seed + int(fraction * 1_000_000))

        month_side = selected.groupby(["month", "side_name"], observed=True).agg(
            n=("net_bps", "size"), gross_bps=("gross_bps", "mean"), net_bps=("net_bps", "mean"), net_pnl_bps_sum=("net_bps", "sum")
        ).reset_index()
        month_side.insert(0, "tail", name)
        month_side_rows.append(month_side)
        day = selected.groupby("day", observed=True).agg(n=("net_bps", "size"), gross_bps=("gross_bps", "mean"), net_bps=("net_bps", "mean"), long_share=("side_name", lambda s: float(s.eq("long").mean()))).reset_index()
        day.insert(0, "tail", name)
        day_rows.append(day)
        symbol = selected.groupby(["symbol", "side_name"], observed=True).agg(n=("net_bps", "size"), gross_bps=("gross_bps", "mean"), net_bps=("net_bps", "mean"), net_pnl_bps_sum=("net_bps", "sum")).reset_index()
        symbol.insert(0, "tail", name)
        symbol_rows.append(symbol)
        for side, part in selected.groupby("side_name", observed=True):
            for column in DIST_COLUMNS:
                values = part[column].dropna().to_numpy(float)
                dist_rows.append({"tail": name, "side_name": side, "feature": column, "n": int(len(values)), "mean": float(values.mean()), "std": float(values.std(ddof=0)), "p05": float(np.quantile(values, .05)), "p25": float(np.quantile(values, .25)), "p50": float(np.quantile(values, .50)), "p75": float(np.quantile(values, .75)), "p95": float(np.quantile(values, .95))})

    for lo, hi in BUCKETS:
        left = int(np.floor(len(ranked) * lo))
        right = int(np.ceil(len(ranked) * hi))
        part = ranked.iloc[left:right]
        bucket_rows.append({"bucket": f"{lo * 100:g}-{hi * 100:g}%", "lower_fraction": lo, "upper_fraction": hi, **_metric(part)})

    high = data.loc[data.high_base_eligible].copy()
    calibration_rows: list[pd.DataFrame] = []
    value_rows: list[pd.DataFrame] = []
    grid_rows: list[pd.DataFrame] = []
    calibration_summary_rows: list[dict] = []
    for side, part in high.groupby("side_name", observed=True):
        part = part.copy()
        part["reliability_decile"] = _quantile_bins(part.meta_score)
        part["base_value_decile"] = _quantile_bins(part.base_score)
        reliability = part.groupby("reliability_decile", observed=True).agg(
            n=("net_bps", "size"), predicted_cost_clear=("meta_score", "mean"), realised_cost_clear=("net_bps", lambda x: float((x > 0.0).mean())), gross_bps=("gross_bps", "mean"), net_bps=("net_bps", "mean"), base_expected_net_bps=("base_score", "mean")
        ).reset_index()
        reliability.insert(0, "side_name", side)
        calibration_rows.append(reliability)
        value = part.groupby("base_value_decile", observed=True).agg(
            n=("net_bps", "size"), base_expected_net_bps=("base_score", "mean"), realised_cost_clear=("net_bps", lambda x: float((x > 0.0).mean())), gross_bps=("gross_bps", "mean"), net_bps=("net_bps", "mean"), reliability_probability=("meta_score", "mean")
        ).reset_index()
        value.insert(0, "side_name", side)
        value_rows.append(value)
        grid = part.groupby(["base_value_decile", "reliability_decile"], observed=True).agg(
            n=("net_bps", "size"), gross_bps=("gross_bps", "mean"), net_bps=("net_bps", "mean"), realised_cost_clear=("net_bps", lambda x: float((x > 0.0).mean())), base_expected_net_bps=("base_score", "mean"), reliability_probability=("meta_score", "mean"), residual_correction_bps=("residual_correction_bps", "mean")
        ).reset_index()
        grid.insert(0, "side_name", side)
        grid_rows.append(grid)
        calibration_summary_rows.append({
            "side_name": side,
            "n": int(len(part)),
            "reliability_predicted_cost_clear": float(part.meta_score.mean()),
            "reliability_realised_cost_clear": float((part.net_bps > 0.0).mean()),
            "reliability_brier": float(np.mean(np.square(part.meta_score - (part.net_bps > 0.0).astype(float)))),
            "reliability_ece_decile": float(
                (reliability.n * (reliability.predicted_cost_clear - reliability.realised_cost_clear).abs()).sum() / reliability.n.sum()
            ),
            "base_expected_net_bps": float(part.base_score.mean()),
            "realised_net_bps": float(part.net_bps.mean()),
            "base_value_mean_error_bps": float((part.net_bps - part.base_score).mean()),
            "base_value_mae_bps": float((part.net_bps - part.base_score).abs().mean()),
        })

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    tail = pd.DataFrame(tail_rows)
    buckets = pd.DataFrame(bucket_rows)
    tail.to_parquet(out / "tail_curve.parquet", index=False)
    buckets.to_parquet(out / "marginal_buckets.parquet", index=False)
    pd.concat(month_side_rows, ignore_index=True).to_parquet(out / "tail_month_side.parquet", index=False)
    pd.concat(day_rows, ignore_index=True).to_parquet(out / "tail_day.parquet", index=False)
    pd.concat(symbol_rows, ignore_index=True).to_parquet(out / "tail_symbol.parquet", index=False)
    pd.DataFrame(dist_rows).to_parquet(out / "tail_distributions.parquet", index=False)
    pd.concat(calibration_rows, ignore_index=True).to_parquet(out / "side_reliability_calibration.parquet", index=False)
    pd.concat(value_rows, ignore_index=True).to_parquet(out / "side_base_value_calibration.parquet", index=False)
    pd.concat(grid_rows, ignore_index=True).to_parquet(out / "side_value_reliability_grid.parquet", index=False)
    pd.DataFrame(calibration_summary_rows).to_parquet(out / "side_calibration_summary.parquet", index=False)
    summary = {
        "schema": "full_universe_selected_p30_trust_diagnostics_v1",
        "scored_input": str(args.scored), "base_state_input": str(args.base_state),
        "row_count": int(len(data)), "oos_start": str(data["__ts__"].min()), "oos_end": str(data["__ts__"].max()),
        "high_base_eligible_rows": int(data.high_base_eligible.sum()),
        "high_base_eligible_fraction": float(data.high_base_eligible.mean()),
        "selection_contract": "global pooled top-k across full OOS candidate population; P30 admission makes winner_score undefined outside causal high-base population; no side/timestamp quota",
        "weekly_block_bootstrap": bootstrap,
    }
    (out / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2))
    _write_report(out, summary, tail, buckets)
    print(json.dumps({"out": str(out), "top10": tail.loc[tail["tail"].eq("top_10pct")].to_dict(orient="records")[0], "bootstrap_top10": bootstrap["top_10pct"]}, indent=2))


if __name__ == "__main__":
    main()
