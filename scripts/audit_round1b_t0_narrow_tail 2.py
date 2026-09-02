#!/usr/bin/env python3
"""Independent narrow-tail audit for Round-1B T0.

This is intentionally a read-only postmortem of the frozen Round-1 meta-OOS
predictions.  It never re-trains or re-ranks within timestamps: every book is
the top fraction of the one pooled global score universe specified by Round 1.

The paired UTC-day bootstrap compares the *already selected* base+meta book
with the respective base-only book.  It resamples days jointly, so the
interval measures whether the meta increment is broad across days rather than
being driven by an isolated episode.  It is not an estimate for a retuned
threshold or a portfolio backtest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FRACTIONS = (0.0025, 0.005, 0.01, 0.02, 0.05)
ARM = "T0_reconstructed_control"
VARIANTS = ("base_only", "meta_only", "base_plus_meta")


def _bps(series: pd.Series) -> pd.Series:
    return series.astype(float) * 10_000.0


def _top_book(frame: pd.DataFrame, fraction: float) -> pd.DataFrame:
    """One global (not per-side or per-timestamp) deterministic top-k book."""
    count = int(round(len(frame) * fraction))
    if count <= 0:
        raise ValueError(f"empty selection for fraction={fraction}")
    # Stable candidate-id tie breaking makes membership auditable.
    return frame.sort_values(["score_bps", "candidate_id"], ascending=[False, True], kind="mergesort").head(count).copy()


def _book_summary(book: pd.DataFrame) -> dict[str, object]:
    gross = _bps(book["execution_gross_ev_12h"])
    cost = _bps(book["execution_cost_return"])
    net = _bps(book["execution_net_ev_12h"])
    return {
        "selected_rows": int(len(book)),
        "gross_bps_per_trade": float(gross.mean()),
        "cost_bps_per_trade": float(cost.mean()),
        "net_bps_per_trade": float(net.mean()),
        "gross_sum_bps": float(gross.sum()),
        "net_sum_bps": float(net.sum()),
        "gross_win_rate": float((gross > 0.0).mean()),
        "utc_days": int(book["utc_day"].nunique()),
        "symbols": int(book["__symbol__"].nunique()),
    }


def _side_summary(book: pd.DataFrame) -> pd.DataFrame:
    overall = len(book)
    rows = []
    for side, group in book.groupby("side_name", sort=True):
        values = _book_summary(group)
        values.update({"side": side, "selected_share": len(group) / overall})
        rows.append(values)
    return pd.DataFrame(rows)


def _month_summary(book: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for month, group in book.groupby("utc_month", sort=True):
        values = _book_summary(group)
        values["utc_month"] = month
        rows.append(values)
    return pd.DataFrame(rows).sort_values("utc_month")


def _symbol_summary(book: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    gross_all = _bps(book["execution_gross_ev_12h"]).sum()
    for symbol, group in book.groupby("__symbol__", sort=True):
        n = len(group)
        gross_sum = _bps(group["execution_gross_ev_12h"]).sum()
        rows.append(
            {
                "symbol": symbol,
                "selected_rows": n,
                "row_share": n / len(book),
                "gross_bps_per_trade": float(_bps(group["execution_gross_ev_12h"]).mean()),
                "gross_sum_bps": float(gross_sum),
                "gross_sum_share": float(gross_sum / gross_all) if gross_all else np.nan,
            }
        )
    table = pd.DataFrame(rows).sort_values(["selected_rows", "symbol"], ascending=[False, True])
    shares = table["row_share"].to_numpy()
    concentration = {
        "symbol_hhi_rows": float(np.square(shares).sum()),
        "top_symbol_row_share": float(shares[0]) if len(shares) else np.nan,
        "top5_symbol_row_share": float(shares[:5].sum()),
    }
    return table, concentration


def _day_contribution(book: pd.DataFrame) -> pd.DataFrame:
    values = book.assign(
        gross_bps=_bps(book["execution_gross_ev_12h"]),
        cost_bps=_bps(book["execution_cost_return"]),
        net_bps=_bps(book["execution_net_ev_12h"]),
    )
    return (
        values.groupby("utc_day", as_index=False)
        .agg(rows=("candidate_id", "size"), gross_sum_bps=("gross_bps", "sum"), cost_sum_bps=("cost_bps", "sum"), net_sum_bps=("net_bps", "sum"))
        .sort_values("utc_day")
    )


def _paired_day_bootstrap(combined: pd.DataFrame, base: pd.DataFrame, reps: int, seed: int) -> dict[str, float | int]:
    """Paired UTC-day resampling of selected-book mean returns and increment."""
    c = _day_contribution(combined).set_index("utc_day")
    b = _day_contribution(base).set_index("utc_day")
    days = c.index.union(b.index).sort_values()
    c = c.reindex(days, fill_value=0.0)
    b = b.reindex(days, fill_value=0.0)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(days), size=(reps, len(days)))
    c_rows = c["rows"].to_numpy(float)
    b_rows = b["rows"].to_numpy(float)
    c_gross = c["gross_sum_bps"].to_numpy(float)
    b_gross = b["gross_sum_bps"].to_numpy(float)
    c_net = c["net_sum_bps"].to_numpy(float)
    b_net = b["net_sum_bps"].to_numpy(float)
    # Ratio of summed returns to summed selected rows preserves daily
    # candidate counts while jointly resampling the selected books.
    c_mean_gross = c_gross[draws].sum(axis=1) / c_rows[draws].sum(axis=1)
    b_mean_gross = b_gross[draws].sum(axis=1) / b_rows[draws].sum(axis=1)
    c_mean_net = c_net[draws].sum(axis=1) / c_rows[draws].sum(axis=1)
    b_mean_net = b_net[draws].sum(axis=1) / b_rows[draws].sum(axis=1)
    delta_gross = c_mean_gross - b_mean_gross
    delta_net = c_mean_net - b_mean_net
    return {
        "paired_days": int(len(days)),
        "bootstrap_reps": int(reps),
        "combined_gross_bootstrap_p05": float(np.quantile(c_mean_gross, 0.05)),
        "combined_gross_bootstrap_p50": float(np.quantile(c_mean_gross, 0.50)),
        "combined_gross_bootstrap_p95": float(np.quantile(c_mean_gross, 0.95)),
        "delta_combined_minus_base_gross_bps": float(_bps(combined["execution_gross_ev_12h"]).mean() - _bps(base["execution_gross_ev_12h"]).mean()),
        "delta_gross_bootstrap_p05": float(np.quantile(delta_gross, 0.05)),
        "delta_gross_bootstrap_p50": float(np.quantile(delta_gross, 0.50)),
        "delta_gross_bootstrap_p95": float(np.quantile(delta_gross, 0.95)),
        "delta_combined_minus_base_net_bps": float(_bps(combined["execution_net_ev_12h"]).mean() - _bps(base["execution_net_ev_12h"]).mean()),
        "delta_net_bootstrap_p05": float(np.quantile(delta_net, 0.05)),
        "delta_net_bootstrap_p50": float(np.quantile(delta_net, 0.50)),
        "delta_net_bootstrap_p95": float(np.quantile(delta_net, 0.95)),
        "delta_gross_probability_positive": float((delta_gross > 0.0).mean()),
    }


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _md_table(table: pd.DataFrame) -> str:
    """Small dependency-free Markdown table renderer (no optional tabulate)."""
    columns = list(table.columns)
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join(["---"] * len(columns)) + " |"
    values = table.copy()
    for column in columns:
        values[column] = values[column].map(_fmt)
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in values.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *body])


def _markdown(summary: pd.DataFrame, side: pd.DataFrame, month: pd.DataFrame, symbols: pd.DataFrame, bootstrap: pd.DataFrame, overlap: pd.DataFrame, source: Path) -> str:
    lines = [
        "# Round-1B T0 narrow-tail audit",
        "",
        "Independent read-only audit of frozen Round-1 `meta_oos` predictions. "
        "All selections are a single pooled-global ranking across sides and timestamps; no per-timestamp or per-side quotas were applied.",
        "",
        f"Source: `{source}`.",
        "",
        "## Pooled narrow tails",
        "",
        _md_table(summary),
        "",
        "## Side contribution",
        "",
        _md_table(side),
        "",
        "## Month contribution",
        "",
        _md_table(month),
        "",
        "## Symbol concentration (top five per book)",
        "",
        _md_table(symbols),
        "",
        "## Paired UTC-day bootstrap: base+meta versus base-only",
        "",
        _md_table(bootstrap),
        "",
        "## Base / base+meta membership overlap",
        "",
        _md_table(overlap),
        "",
        "The bootstrap reuses the already selected global book for each model, jointly resampling UTC days. It therefore tests the stability of the observed meta increment, not a re-optimised trading policy.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data_perp/artifacts/sequential_funnel_round1_g0_tau025_20260801_v6/base_meta_stack_predictions.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_perp/artifacts/round1b_t0_narrow_tail_audit_20260801_v2"))
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_801)
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    frame = frame.loc[(frame["target_arm"] == ARM) & (frame["prediction_fold_id"] == "meta_oos") & frame["model_variant"].isin(VARIANTS)].copy()
    if frame.empty or set(frame["model_variant"].unique()) != set(VARIANTS):
        raise RuntimeError("Frozen T0 meta-OOS output does not contain all three model variants")
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
    frame["utc_day"] = frame["__decision_ts__"].dt.strftime("%Y-%m-%d")
    frame["utc_month"] = frame["__decision_ts__"].dt.strftime("%Y-%m")
    counts = frame.groupby("model_variant")["candidate_id"].nunique()
    if counts.nunique() != 1 or not all(frame.groupby("model_variant")["candidate_id"].size() == counts):
        raise RuntimeError("T0 variants do not share one unique candidate universe")

    summary_rows: list[dict[str, object]] = []
    side_rows: list[pd.DataFrame] = []
    month_rows: list[pd.DataFrame] = []
    symbol_rows: list[pd.DataFrame] = []
    bootstrap_rows: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []
    books: dict[tuple[str, float], pd.DataFrame] = {}
    for variant in VARIANTS:
        variant_frame = frame.loc[frame["model_variant"] == variant]
        for fraction in FRACTIONS:
            book = _top_book(variant_frame, fraction)
            books[(variant, fraction)] = book
            row = _book_summary(book)
            month_for_worst = _month_summary(book)
            worst = month_for_worst.loc[month_for_worst["gross_bps_per_trade"].idxmin()]
            row.update(
                {
                    "worst_month": str(worst["utc_month"]),
                    "worst_month_gross_bps_per_trade": float(worst["gross_bps_per_trade"]),
                    "worst_month_net_bps_per_trade": float(worst["net_bps_per_trade"]),
                }
            )
            row.update({"model_variant": variant, "top_fraction_pct": fraction * 100.0})
            summary_rows.append(row)
            by_side = _side_summary(book)
            by_side.insert(0, "top_fraction_pct", fraction * 100.0)
            by_side.insert(0, "model_variant", variant)
            side_rows.append(by_side)
            by_month = month_for_worst
            by_month.insert(0, "top_fraction_pct", fraction * 100.0)
            by_month.insert(0, "model_variant", variant)
            month_rows.append(by_month)
            by_symbol, concentration = _symbol_summary(book)
            by_symbol = by_symbol.head(5)
            by_symbol.insert(0, "top_fraction_pct", fraction * 100.0)
            by_symbol.insert(0, "model_variant", variant)
            for key, value in concentration.items():
                by_symbol[key] = value
            symbol_rows.append(by_symbol)
    for fraction in FRACTIONS:
        result = _paired_day_bootstrap(books[("base_plus_meta", fraction)], books[("base_only", fraction)], args.bootstrap_reps, args.seed + int(fraction * 1_000_000))
        result["top_fraction_pct"] = fraction * 100.0
        bootstrap_rows.append(result)
        base_ids = set(books[("base_only", fraction)]["candidate_id"])
        combined_ids = set(books[("base_plus_meta", fraction)]["candidate_id"])
        intersection = len(base_ids & combined_ids)
        union = len(base_ids | combined_ids)
        overlap_rows.append(
            {
                "top_fraction_pct": fraction * 100.0,
                "base_selected_rows": len(base_ids),
                "base_plus_meta_selected_rows": len(combined_ids),
                "common_selected_rows": intersection,
                "base_membership_retained_share": intersection / len(base_ids),
                "jaccard_share": intersection / union,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(["top_fraction_pct", "model_variant"])
    side = pd.concat(side_rows, ignore_index=True).sort_values(["top_fraction_pct", "model_variant", "side"])
    month = pd.concat(month_rows, ignore_index=True).sort_values(["top_fraction_pct", "model_variant", "utc_month"])
    symbols = pd.concat(symbol_rows, ignore_index=True).sort_values(["top_fraction_pct", "model_variant", "selected_rows"], ascending=[True, True, False])
    bootstrap = pd.DataFrame(bootstrap_rows).sort_values("top_fraction_pct")
    overlap = pd.DataFrame(overlap_rows).sort_values("top_fraction_pct")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary.to_csv(args.output_dir / "t0_narrow_tail_summary.csv", index=False)
    side.to_csv(args.output_dir / "t0_narrow_tail_by_side.csv", index=False)
    month.to_csv(args.output_dir / "t0_narrow_tail_by_month.csv", index=False)
    symbols.to_csv(args.output_dir / "t0_narrow_tail_top_symbols.csv", index=False)
    bootstrap.to_csv(args.output_dir / "t0_narrow_tail_paired_day_bootstrap.csv", index=False)
    overlap.to_csv(args.output_dir / "t0_narrow_tail_base_combined_overlap.csv", index=False)
    manifest = {
        "source": str(args.input),
        "target_arm": ARM,
        "variants": list(VARIANTS),
        "top_fractions": list(FRACTIONS),
        "selection": "single pooled global score ranking across all sides/timestamps; deterministic candidate_id tie-break",
        "bootstrap": "paired UTC-day resampling of already selected base+meta/base-only books",
        "bootstrap_reps": args.bootstrap_reps,
        "seed": args.seed,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "report.md").write_text(_markdown(summary, side, month, symbols, bootstrap, overlap, args.input))


if __name__ == "__main__":
    main()
