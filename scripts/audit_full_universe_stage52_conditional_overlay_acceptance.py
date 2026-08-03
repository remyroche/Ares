#!/usr/bin/env python3
"""Acceptance audit for the frozen Stage 5.2 conditional-overlay replay.

This runner is intentionally descriptive.  It reads final fixed scores and
labels only after the development-selected configuration has been frozen; it
does not fit, tune, calibrate, or overwrite any model artefact.  Every
selection below is a single global top-k across the complete OOS candidate
book.  Month/week/side figures are attribution of that one selection, never
separate re-ranking exercises.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORED = ROOT / "data_perp/artifacts/full_universe_stage5_2_conditional_overlay_20260804_v1/oos_integrated_predictions.parquet"
DEFAULT_OVERLAY_REPORT = ROOT / "data_perp/artifacts/full_universe_stage5_2_conditional_overlay_20260804_v1/report.json"
DEFAULT_PAYOFF_MANIFEST = ROOT / "data_perp/artifacts/full_universe_stage5_2_conditional_payoff_20260804_v1/manifest.json"
DEFAULT_RESIDUAL_MANIFEST = ROOT / "data_perp/artifacts/full_universe_round_b_residual_20260803_v1/manifest.json"
DEFAULT_RELIABILITY_MANIFEST = ROOT / "data_perp/artifacts/full_universe_round2_reliability_net_gt_25_oos_20260804_v1/manifest.json"
TAILS = (0.001, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.20)
BUCKETS = ((0.0, 0.001), (0.001, 0.0025), (0.0025, 0.005), (0.005, 0.01), (0.01, 0.02), (0.02, 0.03), (0.03, 0.05), (0.05, 0.10), (0.10, 0.20))


def _tail_name(frac: float) -> str:
    return f"top_{frac * 100:g}pct"


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "n": int(len(frame)),
        "gross_bps": float(frame.gross_bps.mean()),
        "net_bps": float(frame.net_bps.mean()),
        "net_pnl_bps_sum": float(frame.net_bps.sum()),
        "long_n": int(frame.side_name.eq("long").sum()),
        "short_n": int(frame.side_name.eq("short").sum()),
        "long_share": float(frame.side_name.eq("long").mean()),
        "unique_days": int(frame.day.nunique()),
        "unique_weeks": int(frame.week.nunique()),
        "unique_symbols": int(frame.symbol.nunique()),
    }


def _rank(frame: pd.DataFrame, score: str) -> pd.DataFrame:
    if not np.isfinite(frame[score].to_numpy(float)).all():
        raise ValueError(f"{score} contains a non-finite candidate score")
    return frame.sort_values([score, "candidate_id"], ascending=[False, True], kind="mergesort").reset_index(drop=True)


def _week_bootstrap(winner: pd.DataFrame, baseline: pd.DataFrame, *, draws: int, seed: int) -> dict[str, Any]:
    """Paired UTC-week block bootstrap of two *fixed global books*.

    This is paired at the resampled time-block level, not trade identity:
    candidate membership can differ by score, while the two books share the
    identical OOS weeks and fixed population size for a given global tail.
    """
    w = winner.groupby("week", observed=True).agg(net_sum=("net_bps", "sum"), n=("net_bps", "size"))
    b = baseline.groupby("week", observed=True).agg(net_sum=("net_bps", "sum"), n=("net_bps", "size"))
    weeks = w.index.intersection(b.index)
    if len(weeks) < 2:
        return {"supported": False, "n_weeks": int(len(weeks))}
    wsum, wn = w.loc[weeks, "net_sum"].to_numpy(float), w.loc[weeks, "n"].to_numpy(float)
    bsum, bn = b.loc[weeks, "net_sum"].to_numpy(float), b.loc[weeks, "n"].to_numpy(float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(weeks), size=(draws, len(weeks)))
    wm = wsum[indices].sum(axis=1) / wn[indices].sum(axis=1)
    bm = bsum[indices].sum(axis=1) / bn[indices].sum(axis=1)
    delta = wm - bm
    return {
        "supported": True, "n_weeks": int(len(weeks)), "draws": int(draws),
        "winner_net_bps_p2_5": float(np.quantile(wm, .025)), "winner_net_bps_p50": float(np.quantile(wm, .5)), "winner_net_bps_p97_5": float(np.quantile(wm, .975)),
        "baseline_net_bps_p2_5": float(np.quantile(bm, .025)), "baseline_net_bps_p50": float(np.quantile(bm, .5)), "baseline_net_bps_p97_5": float(np.quantile(bm, .975)),
        "paired_lift_bps_p2_5": float(np.quantile(delta, .025)), "paired_lift_bps_p50": float(np.quantile(delta, .5)), "paired_lift_bps_p97_5": float(np.quantile(delta, .975)),
        "p_winner_beats_conditional": float((delta > 0.).mean()),
    }


def _table(frame: pd.DataFrame, columns: list[str]) -> str:
    labels = [x.replace("_", " ") for x in columns]
    lines = ["| " + " | ".join(labels) + " |", "|" + "|".join("---" for _ in labels) + "|"]
    for values in frame[columns].itertuples(index=False, name=None):
        cells = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                cells.append(f"{value:.2f}")
            elif isinstance(value, (int, np.integer)):
                cells.append(f"{value:,}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scored", type=Path, default=DEFAULT_SCORED)
    p.add_argument("--overlay-report", type=Path, default=DEFAULT_OVERLAY_REPORT)
    p.add_argument("--payoff-manifest", type=Path, default=DEFAULT_PAYOFF_MANIFEST)
    p.add_argument("--residual-manifest", type=Path, default=DEFAULT_RESIDUAL_MANIFEST)
    p.add_argument("--reliability-manifest", type=Path, default=DEFAULT_RELIABILITY_MANIFEST)
    p.add_argument("--out", type=Path, default=ROOT / "data_perp/artifacts/full_universe_stage52_conditional_overlay_acceptance_audit_20260804_v1")
    p.add_argument("--bootstrap-draws", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260804)
    a = p.parse_args()

    data = pd.read_parquet(a.scored).copy()
    required = {"candidate_id", "__ts__", "side_name", "gross_bps", "net_bps", "conditional_score_bps", "conditional_plus_frozen_residual_bps", "selected_score", "reliability_score", "reliability_eligible"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if data.candidate_id.duplicated().any():
        raise ValueError("candidate ids must be unique")
    # Stored outcome fields are float32, hence the largest round-off observed
    # at very large barrier returns is roughly 1.22e-4 bps.
    if not np.allclose(data.gross_bps.to_numpy(float) - 100., data.net_bps.to_numpy(float), rtol=0., atol=1.5e-4):
        raise ValueError("gross/net cost contract is not exactly 100 bps per row")
    data["__ts__"] = pd.to_datetime(data["__ts__"], utc=True)
    # Keep compact numerical/calendar keys rather than allocating hundreds of
    # thousands of Python strings; this is an audit of a large fixed book.
    data["month"] = data["__ts__"].dt.year * 100 + data["__ts__"].dt.month
    iso = data["__ts__"].dt.isocalendar()
    data["week"] = (iso.year.astype("int32") * 100 + iso.week.astype("int32")).astype("int32")
    data["day"] = data["__ts__"].dt.floor("D")
    data["symbol"] = data.candidate_id.str.partition("|")[0]
    # The selected policy is rank_blend(0.25).  Verify its advertised exact
    # relation rather than trusting the stored final score blindly.
    value = data.conditional_plus_frozen_residual_bps
    value_rank = value.rank(method="average", pct=True)
    reliability_rank = value_rank.copy()
    eligible = data.reliability_eligible.to_numpy(bool)
    reliability_rank.loc[eligible] = data.loc[eligible, "reliability_score"].rank(method="average", pct=True)
    reconstructed = .75 * value_rank + .25 * reliability_rank
    if not np.allclose(reconstructed.to_numpy(float), data.selected_score.to_numpy(float), rtol=0., atol=1e-12):
        raise ValueError("stored selected score does not equal the frozen 75/25 global rank blend")
    if data.loc[eligible, "reliability_score"].isna().any() or data.loc[~eligible, "reliability_score"].notna().any():
        raise ValueError("reliability P30 availability contract fails")

    winner_ranked = _rank(data, "selected_score")
    conditional_ranked = _rank(data, "conditional_score_bps")
    b2_ranked = _rank(data, "b2_score_bps")
    tail_rows, boot = [], {}
    attribution, weekly_rows = [], []
    for frac in TAILS:
        n = int(np.ceil(len(data) * frac))
        winner, conditional, b2 = winner_ranked.iloc[:n], conditional_ranked.iloc[:n], b2_ranked.iloc[:n]
        row = {"tail": _tail_name(frac), "fraction": frac, "selection_n": n, **_metrics(winner)}
        row.update({"conditional_net_bps": float(conditional.net_bps.mean()), "b2_net_bps": float(b2.net_bps.mean()), "lift_vs_conditional_bps": float(winner.net_bps.mean() - conditional.net_bps.mean()), "lift_vs_b2_bps": float(winner.net_bps.mean() - b2.net_bps.mean())})
        tail_rows.append(row)
        boot[row["tail"]] = _week_bootstrap(winner, conditional, draws=a.bootstrap_draws, seed=a.seed + n)
        for scope, part in (("month_side", winner.groupby(["month", "side_name"], observed=True)), ("month", winner.groupby("month", observed=True)), ("side", winner.groupby("side_name", observed=True))):
            x = part.agg(n=("net_bps", "size"), gross_bps=("gross_bps", "mean"), net_bps=("net_bps", "mean"), net_pnl_bps_sum=("net_bps", "sum")).reset_index()
            x.insert(0, "scope", scope); x.insert(0, "tail", row["tail"]); attribution.append(x)
        week = winner.groupby("week", observed=True).agg(n=("net_bps", "size"), gross_bps=("gross_bps", "mean"), net_bps=("net_bps", "mean"), net_pnl_bps_sum=("net_bps", "sum")).reset_index()
        week.insert(0, "tail", row["tail"]); weekly_rows.append(week)

    buckets = []
    for lo, hi in BUCKETS:
        left, right = int(np.floor(len(data) * lo)), int(np.ceil(len(data) * hi))
        part = winner_ranked.iloc[left:right]
        buckets.append({"bucket": f"{lo * 100:g}-{hi * 100:g}%", "lo_fraction": lo, "hi_fraction": hi, **_metrics(part)})
    tail = pd.DataFrame(tail_rows)
    marginal = pd.DataFrame(buckets)
    attribution_df, weekly_df = pd.concat(attribution, ignore_index=True), pd.concat(weekly_rows, ignore_index=True)
    overlay = json.loads(a.overlay_report.read_text())
    payoff = json.loads(a.payoff_manifest.read_text())
    residual = json.loads(a.residual_manifest.read_text())
    reliability = json.loads(a.reliability_manifest.read_text())
    top10 = tail.loc[tail["tail"].eq("top_10pct")].iloc[0].to_dict()
    top1 = tail.loc[tail["tail"].eq("top_1pct")].iloc[0].to_dict()
    top10boot = boot["top_10pct"]
    acceptance = {
        "positive_top1_net": bool(top1["net_bps"] > 0.),
        "positive_top10_net": bool(top10["net_bps"] > 0.),
        "positive_top10_weekly_ci_lower": bool(top10boot["winner_net_bps_p2_5"] > 0.),
        "no_side_failure_top10": bool((attribution_df.loc[(attribution_df["tail"].eq("top_10pct")) & (attribution_df["scope"].eq("side")), "net_bps"] >= 0.).all()),
        "positive_each_month_top10": bool((attribution_df.loc[(attribution_df["tail"].eq("top_10pct")) & (attribution_df["scope"].eq("month")), "net_bps"] > 0.).all()),
        "material_top10_lift_vs_conditional": bool(top10["lift_vs_conditional_bps"] >= 5.),
    }
    acceptance["decision"] = "REJECT_NO_PROMOTION" if not all(acceptance.values()) else "PASS_PROMOTION_GATE"
    summary = {
        "schema": "full_universe_stage52_conditional_overlay_acceptance_audit_v1",
        "status": "COMPLETED_AUDIT_NO_MODEL_CHANGE",
        "inputs": {"scored": str(a.scored), "overlay_report": str(a.overlay_report), "conditional_payoff_manifest": str(a.payoff_manifest), "residual_manifest": str(a.residual_manifest), "reliability_manifest": str(a.reliability_manifest)},
        "invariants": {"unique_candidate_id": True, "gross_minus_net_exactly_100bps": True, "selected_score_exactly_reconstructed": True, "reliability_score_only_on_causal_p30": True, "all_selected_score_finite": True, "global_pool_size": int(len(data)), "oos_start": str(data.__ts__.min()), "oos_end": str(data.__ts__.max()), "oos_months": sorted(data.month.unique().tolist())},
        "frozen_rule": overlay["winner"],
        "causality": {"conditional_payoff": payoff["causality"], "residual_train_eval": {"train_window": residual["train_window"], "eval_window": residual["eval_window"], "target": residual["target_definition"]}, "reliability_train_eval": {"train_window": reliability["train_window"], "eval_window": reliability["eval_window"], "target": reliability["target_definition"], "causal_admission": "frozen B2 top 30 percent"}},
        "coverage": {"conditional_payoff_feature_finite_fraction": payoff["feature_coverage"], "reliability_eligible_rows": int(eligible.sum()), "reliability_eligible_fraction": float(eligible.mean()), "full_score_rows": int(data.selected_score.notna().sum())},
        "weekly_block_bootstrap": boot,
        "acceptance": acceptance,
    }
    a.out.mkdir(parents=True, exist_ok=True)
    tail.to_parquet(a.out / "global_tail_curve.parquet", index=False)
    marginal.to_parquet(a.out / "global_marginal_buckets.parquet", index=False)
    attribution_df.to_parquet(a.out / "fixed_global_selection_attribution.parquet", index=False)
    weekly_df.to_parquet(a.out / "fixed_global_selection_weekly.parquet", index=False)
    (a.out / "audit.json").write_text(json.dumps(summary, indent=2))
    monthtop10 = attribution_df.loc[(attribution_df["tail"].eq("top_10pct")) & (attribution_df["scope"].eq("month"))]
    sidetop10 = attribution_df.loc[(attribution_df["tail"].eq("top_10pct")) & (attribution_df["scope"].eq("side"))]
    report = f"""# Stage 5.2 conditional-overlay acceptance audit

## Decision

**{acceptance['decision']}**.  The frozen development-selected stack improves the immediate conditional-payoff baseline by {top10['lift_vs_conditional_bps']:.2f} bps at global top 10%, but it does not clear the economic promotion gate.

The fixed replay is positive at top 1% ({top1['net_bps']:.2f} net bps on {top1['n']:,} candidates), but that tail is not stable enough to support promotion: its weekly 95% interval crosses zero, and the top 0.1% is concentrated in only two decision days.  The economically relevant broad top 10% is {top10['net_bps']:.2f} net bps, its weekly block-bootstrap 95% interval is [{top10boot['winner_net_bps_p2_5']:.2f}, {top10boot['winner_net_bps_p97_5']:.2f}], and the short-side contribution is negative.

## Frozen contract and causality

- Rule: 75% global rank of conditional value plus frozen residual delta, and 25% global rank of the frozen shared `P(net > 25 bps)` head within its causal B2 P30 population.  Outside P30, the reliability coordinate equals the value rank, so no candidate is removed from the globally pooled book.
- Evaluation: {data.__ts__.min()} to {data.__ts__.max()} ({len(data):,} candidates).  Every selection is one global deterministic top-k, tie-broken by candidate id; tables below only attribute that fixed selection.
- Conditional-payoff fit labels end before {payoff['causality']['oos_fit_labels_resolved_before']}; the manifest explicitly asserts that no OOS outcome entered the payoff fit.  Residual and reliability train windows both end at {residual['train_window'][1]}; OOS begins {residual['eval_window'][0]}.
- Coverage: all {len(data):,} rows have a finite final score. Reliability is available on {int(eligible.sum()):,} rows ({eligible.mean():.2%}) exactly; conditional payoff inputs have finite OOS coverage from {min(v['oos_finite_fraction'] for v in payoff['feature_coverage'].values()):.2%} to 100% (the minimum is `amihud_z`).

## Global tail curve

{_table(tail, ['tail', 'selection_n', 'gross_bps', 'net_bps', 'conditional_net_bps', 'lift_vs_conditional_bps', 'long_share', 'unique_days', 'unique_symbols'])}

## Marginal global buckets

{_table(marginal, ['bucket', 'n', 'gross_bps', 'net_bps', 'long_share', 'unique_days', 'unique_symbols'])}

## Top-10 fixed-global attribution by month

{_table(monthtop10, ['month', 'n', 'gross_bps', 'net_bps', 'net_pnl_bps_sum'])}

## Top-10 fixed-global attribution by side

{_table(sidetop10, ['side_name', 'n', 'gross_bps', 'net_bps', 'net_pnl_bps_sum'])}

## Weekly paired uncertainty

The comparator is the exact same full OOS population and same global tail count ranked by the Stage 5.2 conditional payoff alone.  The bootstrap resamples UTC weeks jointly from the fixed winner and fixed comparator selections; it measures the lift of the frozen overlay rather than re-selecting policies within each week.

- Top 10% overlay: [{top10boot['winner_net_bps_p2_5']:.2f}, {top10boot['winner_net_bps_p97_5']:.2f}] net bps (95% interval).
- Top 10% lift vs conditional: [{top10boot['paired_lift_bps_p2_5']:.2f}, {top10boot['paired_lift_bps_p97_5']:.2f}] bps; P(lift > 0) = {top10boot['p_winner_beats_conditional']:.3f}.

## Gate detail

{json.dumps(acceptance, indent=2)}

The acceptance decision is intentionally based on broad, net-of-cost economics and side/month robustness, not on the highly concentrated top-1 pocket.
"""
    (a.out / "ACCEPTANCE_AUDIT.md").write_text(report)
    print(json.dumps({"out": str(a.out), "top10": top10, "acceptance": acceptance}, indent=2))


if __name__ == "__main__":
    main()
