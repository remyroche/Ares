#!/usr/bin/env python3
"""Matched constrained replay of MC1 admission with Adaptive Exit V1.

The admission set is frozen MC1_d2 (absolute EV >= 50 bps) and the auction
order is the frozen strict-R3 ``final_score``.  The baseline arm uses the
SimplePolicyOptimiser outcome.  The challenger changes only the trailing
activation on candidates with an existing OOF Adaptive Exit V1 replay; rows
without such historical states retain the optimiser exit exactly.  This is a
conservative source-aligned comparison, not an imputation of adaptive uplift.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_strict_r3_six_mapper_families import replay
from extreme_price_movements.strict_r3_inference_bundle import StrictR3InferenceBundle

DEFAULT_LEDGER = ROOT / "data_perp/artifacts/strict_r3_lockstep_history_long_2024apr_jul2026_strictfull_prior28_optimizedpolicy_20260812_v1/walkforward_scored_label_ledger.parquet"
DEFAULT_MAPPERS = ROOT / "data_perp/artifacts/strict_r3_six_mapper_families_long_2025_2026_20260813_v4/finalist_causal_predictions.parquet"
DEFAULT_ADAPTIVE = ROOT / "data_perp/artifacts/canonical_a5_source_aligned_hybrid_adaptive_exit_funnel_20260813_v4"
DEFAULT_BUNDLE = ROOT / "data_perp/artifacts/adaptive_exit_v1_canonical_long_20260801_v1"
DEFAULT_POLICY = ROOT / "data_perp/artifacts/strict_r3_schema_v2_simple_policy_targetfree_long_pre2025_20260809_v3/winner.json"
DEFAULT_INFERENCE_BUNDLE = ROOT / "config/strict_r3_inference_bundle_long_20260801_v6_robust21_mc1_d2_adaptive_exit_v1.json"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/strict_r3_mc1_adaptive_exit_matched_2025_2026_20260813_v1"
WINNER = "F4_disagreement_abstain_p80"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _risk(equity: pd.DataFrame, decisions: pd.DataFrame) -> dict[str, float]:
    e = equity.copy()
    e["timestamp"] = pd.to_datetime(e.timestamp, utc=True)
    value = pd.to_numeric(e.mtm_equity, errors="coerce").dropna()
    if len(value):
        peak = value.cummax()
        drawdown = value / peak - 1.0
        max_drawdown = float(drawdown.min())
        ulcer = float(np.sqrt(np.mean(np.square(100.0 * drawdown))))
        growth = float(value.iloc[-1] / value.iloc[0]) if value.iloc[0] else np.nan
    else:
        max_drawdown = ulcer = np.nan
        growth = 1.0
    d = decisions.loc[decisions.accepted.fillna(False)].copy()
    net = pd.to_numeric(d.position_net_return, errors="coerce").dropna()
    downside = net[net < 0]
    sortino = float(net.mean() / downside.std(ddof=0)) if len(downside) > 1 and downside.std(ddof=0) > 0 else np.nan
    log_growth = math.log(growth) if np.isfinite(growth) and growth > 0 else np.nan
    return {
        "portfolio_growth_multiple": growth,
        "portfolio_growth_pct": (growth - 1.0) * 100.0,
        "max_drawdown": max_drawdown,
        "growth_to_drawdown": log_growth / abs(max_drawdown) if max_drawdown < 0 else np.nan,
        "sortino_trade": sortino,
        "ulcer_index_pct": ulcer,
    }


def _period_rows(
    arm: str,
    decisions: pd.DataFrame,
    equity: pd.DataFrame,
    *,
    start: str | pd.Timestamp = "2025-01-01",
    end_exclusive: str | pd.Timestamp = "2026-08-01",
) -> list[dict[str, object]]:
    accepted = decisions.loc[decisions.accepted.fillna(False)].copy()
    accepted["timestamp"] = pd.to_datetime(accepted.timestamp, utc=True)
    accepted["net_bps"] = pd.to_numeric(accepted.position_net_return, errors="coerce") * 10_000.0
    report_start = pd.Timestamp(start)
    report_end = pd.Timestamp(end_exclusive)
    report_start = report_start.tz_localize("UTC") if report_start.tzinfo is None else report_start.tz_convert("UTC")
    report_end = report_end.tz_localize("UTC") if report_end.tzinfo is None else report_end.tz_convert("UTC")
    periods: list[tuple[str, pd.Timestamp, pd.Timestamp]] = [
        ("global", report_start, report_end),
    ]
    for year in range(report_start.year, report_end.year + 1):
        lo = max(report_start, pd.Timestamp(year, 1, 1, tz="UTC"))
        hi = min(report_end, pd.Timestamp(year + 1, 1, 1, tz="UTC"))
        if lo < hi:
            periods.append((str(year), lo, hi))
    for month_start in pd.date_range(
        report_start.normalize().replace(day=1), report_end, freq="MS",
    ):
        lo = max(report_start, month_start)
        hi = min(report_end, month_start + pd.offsets.MonthBegin())
        if lo < hi:
            periods.append((month_start.strftime("%Y-%m"), lo, hi))
    out: list[dict[str, object]] = []
    for label, lo, hi in periods:
        rows = accepted[accepted.timestamp.ge(lo) & accepted.timestamp.lt(hi)]
        if label not in {"global", "2025", "2026"} and rows.empty:
            continue
        days = max(1, (hi - lo).days)
        weeks = rows.assign(week=rows.timestamp.dt.strftime("%G-W%V")).groupby("week").net_bps.agg(["size", "mean", "sum"])
        months = rows.assign(month=rows.timestamp.dt.strftime("%Y-%m")).groupby("month").net_bps.agg(["size", "mean", "sum"])
        daily_counts = rows.groupby(rows.timestamp.dt.normalize()).size().reindex(pd.date_range(lo, hi - pd.Timedelta(days=1), freq="D"), fill_value=0)
        period_equity = e = equity.copy()
        e["timestamp"] = pd.to_datetime(e.timestamp, utc=True)
        period_equity = e[e.timestamp.ge(lo) & e.timestamp.lt(hi)]
        period_decisions = decisions.copy()
        period_decisions["timestamp"] = pd.to_datetime(period_decisions.timestamp, utc=True)
        period_decisions = period_decisions[
            period_decisions.timestamp.ge(lo) & period_decisions.timestamp.lt(hi)
        ]
        row = {
            "arm": arm, "period": label, "start": lo, "end_exclusive": hi,
            "trades": int(len(rows)), "trades_per_day": len(rows) / days,
            "net_ev_per_trade_bps": float(rows.net_bps.mean()) if len(rows) else np.nan,
            "net_ev_per_day_bps": float(rows.net_bps.sum() / days),
            "net_sum_bps": float(rows.net_bps.sum()),
            "positive_trade_fraction": float(rows.net_bps.gt(0).mean()) if len(rows) else np.nan,
            "worst_week_ev_bps": float(weeks["mean"].min()) if len(weeks) else np.nan,
            "worst_week_sum_bps": float(weeks["sum"].min()) if len(weeks) else np.nan,
            "worst_month_ev_bps": float(months["mean"].min()) if len(months) else np.nan,
            "worst_month_sum_bps": float(months["sum"].min()) if len(months) else np.nan,
            "days_without_trades": int((daily_counts == 0).sum()),
            "days_with_lt5_trades": int((daily_counts < 5).sum()),
        }
        row.update(_risk(period_equity, period_decisions))
        out.append(row)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    p.add_argument("--mappers", type=Path, default=DEFAULT_MAPPERS)
    p.add_argument("--adaptive-dir", type=Path, default=DEFAULT_ADAPTIVE)
    p.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE)
    p.add_argument("--policy-json", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--inference-bundle", type=Path, default=DEFAULT_INFERENCE_BUNDLE)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--start", default="2025-01-01T00:00:00Z")
    p.add_argument("--end-exclusive", default="2026-08-01T00:00:00Z")
    p.add_argument(
        "--enforce-timestamp-top20-base-route",
        action="store_true",
        help="Replay the executable inference compute gate before MC1 admission.",
    )
    args = p.parse_args()
    inference_bundle_path = args.inference_bundle.resolve()
    inference_bundle = StrictR3InferenceBundle.load(inference_bundle_path, root=ROOT)
    inference_audit = inference_bundle.validate(decision_ts="2026-08-14T00:00:00Z")
    sealed = inference_bundle.payload
    expected = sealed["sha256"]
    replay_hashes = {
        "adaptive_exit_v1_model": _sha(args.bundle_dir / "adaptive_exit_v1.joblib"),
        "adaptive_exit_v1_manifest": _sha(args.bundle_dir / "run_manifest.json"),
        "exit_policy": _sha(args.policy_json),
    }
    for key, actual in replay_hashes.items():
        if actual != expected[key]:
            raise ValueError(
                f"Replay input {key} does not match sealed inference bundle: "
                f"expected {expected[key]}, got {actual}"
            )
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)

    con = duckdb.connect()
    con.execute("set memory_limit='8GB'")
    start = pd.Timestamp(args.start)
    end_exclusive = pd.Timestamp(args.end_exclusive)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end_exclusive = (
        end_exclusive.tz_localize("UTC")
        if end_exclusive.tzinfo is None else end_exclusive.tz_convert("UTC")
    )
    if start >= end_exclusive:
        raise ValueError("start must precede end-exclusive")
    sql = f"""
      select l.*, p.mapped_ev
      from read_parquet('{args.ledger.as_posix()}') l
      join read_parquet('{args.mappers.as_posix()}') p using(candidate_id)
      where p.arm='MC1_d2'
        and l.__decision_ts__ >= timestamptz '{start.isoformat()}'
        and l.__decision_ts__ < timestamptz '{end_exclusive.isoformat()}'
    """
    base = con.execute(sql).fetchdf()
    base["__decision_ts__"] = pd.to_datetime(base.__decision_ts__, utc=True)
    route_rows = len(base)
    if args.enforce_timestamp_top20_base_route:
        ordered = base.sort_values(
            ["__decision_ts__", "base_score", "candidate_id"],
            ascending=[True, False, True],
            kind="stable",
        )
        group_size = ordered.groupby("__decision_ts__")["candidate_id"].transform("size")
        route_count = np.maximum(1, np.ceil(0.20 * group_size).astype(int))
        route_position = ordered.groupby("__decision_ts__").cumcount()
        routed_ids = ordered.loc[route_position.lt(route_count), "candidate_id"]
        base = base.loc[base.candidate_id.isin(routed_ids)].copy()
        if len(base) != int(route_count.groupby(ordered["__decision_ts__"]).first().sum()):
            raise AssertionError("timestamp-local top-20 route changed candidate identity")
    replay_rows = pd.read_parquet(args.adaptive_dir / "oof_replay.parquet", filters=[("arm", "=", WINNER)])
    replay_rows = replay_rows.drop_duplicates("candidate_id").set_index("candidate_id")
    valid_source = base.policy_outcome_source.astype(str).eq("existing_15m_or_exact")
    supported = base.candidate_id.astype(str).isin(replay_rows.index) & valid_source
    adaptive = base.copy()
    ids = adaptive.loc[supported, "candidate_id"].astype(str)
    adaptive.loc[supported, "policy_net_bps"] = ids.map(replay_rows.adaptive_net_bps).to_numpy(float)
    adaptive.loc[supported, "policy_gross_bps"] = ids.map(replay_rows.adaptive_gross_bps).to_numpy(float)
    adaptive.loc[supported, "policy_exit_bar_15m"] = ids.map(replay_rows.adaptive_exit_bar).to_numpy(float)
    adaptive.loc[supported, "policy_exit_reason"] = ids.map(replay_rows.adaptive_exit_reason).astype(str).to_numpy()
    adaptive.loc[supported, "policy_exit_price"] = pd.to_numeric(adaptive.loc[supported, "policy_entry_price"], errors="raise").to_numpy(float) * (1.0 + pd.to_numeric(adaptive.loc[supported, "policy_gross_bps"], errors="raise").to_numpy(float) / 10_000.0)
    adaptive["adaptive_exit_historical_supported"] = supported

    baseline_summary, _, baseline_decisions, baseline_equity = replay(base, "MC1_SIMPLE_POLICY", 0)
    adaptive_summary, _, adaptive_decisions, adaptive_equity = replay(adaptive, "MC1_ADAPTIVE_EXIT_V1", 0)
    baseline_decisions.to_parquet(args.out_dir / "baseline_portfolio_decisions.parquet", index=False)
    adaptive_decisions.to_parquet(args.out_dir / "adaptive_portfolio_decisions.parquet", index=False)
    baseline_equity.to_parquet(args.out_dir / "baseline_equity.parquet", index=False)
    adaptive_equity.to_parquet(args.out_dir / "adaptive_equity.parquet", index=False)
    supported_table = adaptive.loc[:, ["candidate_id", "__decision_ts__", "mapped_ev", "adaptive_exit_historical_supported"]]
    supported_table.to_parquet(args.out_dir / "adaptive_support_audit.parquet", index=False)

    # The joined ledger is large.  Release it before building reporting tables
    # so the deterministic report phase does not exceed a live host's memory.
    adaptive_candidate_support_rows = int(supported.sum())
    adaptive_candidate_population_rows = int(len(base))
    adaptive_candidate_support_fraction = float(supported.mean())
    del base, adaptive, replay_rows, supported_table
    con.close()
    gc.collect()

    metrics = pd.DataFrame(
        _period_rows(
            "SimplePolicyOptimiser baseline", baseline_decisions, baseline_equity,
            start=start, end_exclusive=end_exclusive,
        )
        + _period_rows(
            "SimplePolicyOptimiser + Adaptive Exit V1", adaptive_decisions, adaptive_equity,
            start=start, end_exclusive=end_exclusive,
        )
    )
    controls = metrics[metrics.arm.eq("SimplePolicyOptimiser baseline")].drop(columns="arm")
    delta = metrics.merge(controls, on=["period", "start", "end_exclusive"], suffixes=("", "__baseline"), validate="many_to_one")
    for field in ("trades", "trades_per_day", "net_ev_per_trade_bps", "net_ev_per_day_bps", "net_sum_bps", "positive_trade_fraction", "worst_week_ev_bps", "worst_week_sum_bps", "worst_month_ev_bps", "worst_month_sum_bps", "days_without_trades", "days_with_lt5_trades", "portfolio_growth_multiple", "portfolio_growth_pct", "max_drawdown", "growth_to_drawdown", "sortino_trade", "ulcer_index_pct"):
        if field in delta:
            delta[f"delta_{field}"] = delta[field] - delta[f"{field}__baseline"]
    metrics.to_parquet(args.out_dir / "period_metrics.parquet", index=False)
    metrics.to_csv(args.out_dir / "period_metrics.csv", index=False)
    delta.to_parquet(args.out_dir / "period_metrics_with_delta.parquet", index=False)
    delta.to_csv(args.out_dir / "period_metrics_with_delta.csv", index=False)

    manifest = {
        "schema": "strict_r3_mc1_adaptive_exit_source_aligned_replay_v1",
        "admission": "frozen MC1_d2 expected net >= +50 bps",
        "auction": "frozen strict-R3 final_score",
        "exit_baseline": "frozen SimplePolicyOptimiser",
        "adaptive_exit_role": "activation_only_overlay_on_simple_policy_optimiser",
        "adaptive_controller": WINNER,
        "start": start.isoformat(),
        "end_exclusive": end_exclusive.isoformat(),
        "base_route": (
            "decision_timestamp_local_top20_base_score_candidate_id_tiebreak"
            if args.enforce_timestamp_top20_base_route else "legacy_all_candidates"
        ),
        "pre_route_candidate_rows": int(route_rows),
        "post_route_candidate_rows": int(adaptive_candidate_population_rows),
        "unsupported_historical_action": "preserve SimplePolicyOptimiser outcome exactly",
        "adaptive_candidate_support_rows": adaptive_candidate_support_rows,
        "adaptive_candidate_population_rows": adaptive_candidate_population_rows,
        "adaptive_candidate_support_fraction": adaptive_candidate_support_fraction,
        "adaptive_bundle_manifest_sha256": _sha(args.bundle_dir / "run_manifest.json"),
        "adaptive_model_sha256": _sha(args.bundle_dir / "adaptive_exit_v1.joblib"),
        "policy_sha256": _sha(args.policy_json),
        "mapper_source_sha256": _sha(args.mappers),
        "inference_bundle": str(inference_bundle_path.relative_to(ROOT)),
        "inference_bundle_sha256": _sha(inference_bundle_path),
        "inference_bundle_audit": inference_audit,
        "sealed_replay_input_hashes": replay_hashes,
        "calibration_contract": {
            "upstream_same_model_reserve_days": 28,
            "conversion_same_model_reserve_days": 28,
            "robust21_history_days": int(sealed["reference_window_days"]),
            "mc1_dynamic_shift_days": 21,
            "mc1_day_tail_trim_each_side": 0.10,
            "resolved_outcomes_only": True,
            "held_window_percentiles": False,
        },
        "baseline_summary": baseline_summary,
        "adaptive_summary": adaptive_summary,
        "cost_bps": 100.0,
        "portfolio": "long-only; 7x leverage; 10% margin slots; 80% margin cap; 8 concurrent; 2 new entries/hour",
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(json.dumps(manifest, default=str), flush=True)


if __name__ == "__main__":
    main()
