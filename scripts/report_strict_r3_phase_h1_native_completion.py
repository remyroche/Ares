#!/usr/bin/env python3
"""Render the offline phase-native Strict-R3 replay receipt as Markdown.

The detailed ``phase_hourly_admissions.parquet`` remains the authoritative
per-decision record.  This renderer writes a concise companion report that
makes the phase-local admission funnel and the only shared component (the
chronological portfolio auction) explicit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _fmt(value: object, digits: int = 2) -> str:
    if pd.isna(value):
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _table(frame: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    if frame.empty:
        return "_No rows._\n"
    headers = [label for _, label in columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in frame.loc[:, [name for name, _ in columns]].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_fmt(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.chain_root
    pooled = root / "pooled_four_phase_native"
    required = [
        pooled / "phase_hourly_admissions.parquet",
        pooled / "phase_hourly_admission_summary.parquet",
        pooled / "portfolio_decisions.parquet",
        pooled / "portfolio_metrics.parquet",
        pooled / "run_manifest.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"phase-native report inputs missing: {missing}")

    hourly = pd.read_parquet(pooled / "phase_hourly_admissions.parquet")
    summary = pd.read_parquet(pooled / "phase_hourly_admission_summary.parquet")
    decisions = pd.read_parquet(pooled / "portfolio_decisions.parquet")
    portfolio = pd.read_parquet(pooled / "portfolio_metrics.parquet")
    accepted_flag = decisions.get("accepted", pd.Series(False, index=decisions.index))
    accepted = decisions.loc[accepted_flag.fillna(False).astype(bool)].copy()
    if not accepted.empty:
        accepted["timestamp"] = pd.to_datetime(accepted["timestamp"], utc=True)
        accepted["net_bps"] = pd.to_numeric(accepted["position_net_return"], errors="coerce") * 10_000.0
        accepted_phase = accepted.groupby("phase_minutes", as_index=False).agg(
            entries=("candidate_id", "size"),
            accepted_net_ev_bps=("net_bps", "mean"),
            accepted_net_sum_bps=("net_bps", "sum"),
            first_entry=("timestamp", "min"),
            last_entry=("timestamp", "max"),
        )
    else:
        accepted_phase = pd.DataFrame(columns=[
            "phase_minutes", "entries", "accepted_net_ev_bps", "accepted_net_sum_bps", "first_entry", "last_entry",
        ])
    hourly_shape = hourly.groupby("phase_minutes", as_index=False).agg(
        decision_hours=("hour", "size"),
        hours_with_current_mc1_pass=(
            "current_mapper_pass_rows",
            lambda x: int(pd.to_numeric(x, errors="coerce").gt(0).sum()),
        ),
        hours_with_bcf_mc1_pass=(
            "bcf_mapper_pass_rows",
            lambda x: int(pd.to_numeric(x, errors="coerce").gt(0).sum()),
        ),
        hours_with_dual_admission=("dual_admitted_target_free_rows", lambda x: int(pd.to_numeric(x).gt(0).sum())),
        hours_with_portfolio_entry=("portfolio_accepted_rows", lambda x: int(pd.to_numeric(x).gt(0).sum())),
        peak_dual_admitted_per_hour=("dual_admitted_target_free_rows", "max"),
        peak_portfolio_entries_per_hour=("portfolio_accepted_rows", "max"),
    )
    # These are admission-time counts, not an after-the-fact global-tail
    # statistic.  Keep their distribution beside the totals so a phase cannot
    # appear attractive merely because a small number of hours created all of
    # its admitted population.
    hourly_distribution = (
        hourly.groupby("phase_minutes", as_index=False, sort=True)
        .agg(
            base_routed_per_hour=("current_routed_rows", "mean"),
            current_pass_per_hour=("current_mapper_pass_rows", "mean"),
            bcf_pass_per_hour=("bcf_mapper_pass_rows", "mean"),
            current_pass_p50_per_hour=("current_mapper_pass_rows", "median"),
            current_pass_p90_per_hour=(
                "current_mapper_pass_rows",
                lambda x: float(pd.to_numeric(x, errors="coerce").quantile(.90)),
            ),
            bcf_pass_p50_per_hour=("bcf_mapper_pass_rows", "median"),
            bcf_pass_p90_per_hour=(
                "bcf_mapper_pass_rows",
                lambda x: float(pd.to_numeric(x, errors="coerce").quantile(.90)),
            ),
            dual_admitted_p50_per_hour=("dual_admitted_target_free_rows", "median"),
            dual_admitted_p90_per_hour=(
                "dual_admitted_target_free_rows",
                lambda x: float(pd.to_numeric(x, errors="coerce").quantile(.90)),
            ),
            accepted_p50_per_hour=("portfolio_accepted_rows", "median"),
            accepted_p90_per_hour=(
                "portfolio_accepted_rows",
                lambda x: float(pd.to_numeric(x, errors="coerce").quantile(.90)),
            ),
        )
    )
    # ``phase_hourly_admission_summary`` is authoritative for the decision-hour
    # denominator.  Do not merge the independently recomputed denominator from
    # ``hourly_shape`` as well: pandas would suffix both columns, which in turn
    # makes the admission-rate calculations below depend on a non-existent
    # unsuffixed name.  The remaining fields are reporting diagnostics only.
    phase = summary.merge(
        hourly_shape.drop(columns=["decision_hours"]),
        on="phase_minutes",
        how="left",
        validate="one_to_one",
    )
    phase = phase.merge(hourly_distribution, on="phase_minutes", how="left", validate="one_to_one")
    phase = phase.merge(accepted_phase, on="phase_minutes", how="left", suffixes=("", "_decision"))
    phase["dual_admissions_per_hour"] = (
        pd.to_numeric(phase["dual_admitted_target_free_rows"], errors="coerce")
        / pd.to_numeric(phase["decision_hours"], errors="coerce").replace(0, pd.NA)
    )
    phase["accepted_entries_per_hour"] = (
        pd.to_numeric(phase["portfolio_accepted_rows"], errors="coerce")
        / pd.to_numeric(phase["decision_hours"], errors="coerce").replace(0, pd.NA)
    )
    phase["base_to_dual_admission_rate"] = (
        pd.to_numeric(phase["dual_admitted_target_free_rows"], errors="coerce")
        / pd.to_numeric(phase["current_routed_rows"], errors="coerce").replace(0, pd.NA)
    )
    phase["dual_to_portfolio_rate"] = (
        pd.to_numeric(phase["portfolio_accepted_rows"], errors="coerce")
        / pd.to_numeric(phase["dual_admitted_target_free_rows"], errors="coerce").replace(0, pd.NA)
    )
    phase = phase.sort_values("phase_minutes", kind="stable")
    overall = portfolio.iloc[0].to_dict() if not portfolio.empty else {}

    lines = [
        "# Strict-R3 phase-native four-decision replay",
        "",
        "This is an offline research receipt.  :15, :30 and :45 were independently materialised, scored, and mapped before one chronological portfolio auction combined their already-admitted rows.  It is not a live-trading result.",
        "",
        "## Admission and portfolio funnel by phase",
        "",
        _table(phase, [
            ("phase_minutes", "Phase"),
            ("decision_hours", "Decision hours"),
            ("current_routed_rows", "Base routed"),
            ("current_mapper_pass_rows", "Current MC1 pass"),
            ("current_pass_per_hour", "Current pass/hour"),
            ("current_pass_p50_per_hour", "Current pass p50/hour"),
            ("current_pass_p90_per_hour", "Current pass p90/hour"),
            ("bcf_mapper_pass_rows", "BCF MC1 pass"),
            ("bcf_pass_per_hour", "BCF pass/hour"),
            ("bcf_pass_p50_per_hour", "BCF pass p50/hour"),
            ("bcf_pass_p90_per_hour", "BCF pass p90/hour"),
            ("dual_admitted_target_free_rows", "Dual admitted"),
            ("dual_admissions_per_hour", "Dual/hour"),
            ("dual_admitted_p50_per_hour", "Dual p50/hour"),
            ("dual_admitted_p90_per_hour", "Dual p90/hour"),
            ("base_to_dual_admission_rate", "Base→dual rate"),
            ("portfolio_accepted_rows", "Portfolio entries"),
            ("accepted_entries_per_hour", "Entries/hour"),
            ("dual_to_portfolio_rate", "Dual→entry rate"),
            ("accepted_net_ev_bps", "Accepted net bps/trade"),
            ("accepted_net_sum_bps", "Accepted net bps sum"),
        ]),
        "## Hourly admissions",
        "",
        "The complete target-free decision-level ledger is `pooled_four_phase_native/phase_hourly_admissions.parquet`.  It distinguishes base route, each native MC1 pass, dual admission before outcome join, valid-outcome support, and final portfolio acceptance.",
        "",
        _table(phase, [
            ("phase_minutes", "Phase"),
            ("hours_with_current_mc1_pass", "Hours with current pass"),
            ("hours_with_bcf_mc1_pass", "Hours with BCF pass"),
            ("hours_with_dual_admission", "Hours with dual admission"),
            ("hours_with_portfolio_entry", "Hours with entry"),
            ("peak_dual_admitted_per_hour", "Peak dual/hour"),
            ("peak_portfolio_entries_per_hour", "Peak entries/hour"),
            ("first_entry", "First entry"),
            ("last_entry", "Last entry"),
        ]),
        "## Shared portfolio result",
        "",
        _table(pd.DataFrame([overall]), [
            ("accepted_rows", "Accepted rows"),
            ("realised_rows", "Resolved trades"),
            ("net_ev_bps_per_realised_trade", "Net bps/trade"),
            ("net_sum_bps_realised", "Net bps sum"),
            ("max_drawdown", "Max drawdown"),
            ("worst_month_bps", "Worst month bps/trade"),
            ("worst_week_bps", "Worst week bps/trade"),
        ]),
        "## Contract checks",
        "",
        "- phase feature schemas are target/outcome-free before raw scoring;",
        "- each raw scoring receipt uses the declared phase stream and same-phase reserve;",
        "- current and BCF maps remain native to their own raw score families;",
        "- policy outcomes join only after target-free dual admission;",
        "- only the global chronological auction shares portfolio state.",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))
    print(f"wrote {root / 'REPORT.md'}")


if __name__ == "__main__":
    main()
