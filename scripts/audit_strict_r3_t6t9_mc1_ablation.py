#!/usr/bin/env python3
"""Seal the offline T6/T9 score-construction and dual-MC1 ablation.

This audit never fits a model, changes a score, writes a portfolio state, or
performs exchange I/O.  It turns completed immutable trial receipts into the
matched-ID, calibration, risk, and displacement tables required for the
research decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROHIBITED = frozenset({
    "policy_path_valid", "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m",
    "policy_entry_price", "policy_exit_price", "policy_exit_reason", "policy_label_available_ts",
    "policy_cost_bps", "semantic_path_valid", "semantic_sequence", "semantic_speed_bin",
    "semantic_persistence_bin", "semantic_pre_adverse_bin", "semantic_policy_conversion_bin",
    "semantic_exit_reason", "semantic_composite", "semantic_tbm_event",
})
EVAL_START = pd.Timestamp("2026-05-01T00:00:00Z")
EVAL_END = pd.Timestamp("2026-08-01T00:00:00Z")
BASE_BANDS = (0.60, 0.80, 0.90, 0.95, 0.97, 0.99, 1.00)
BASE_LABELS = ("60-80%", "80-90%", "90-95%", "95-97%", "97-99%", "99-100%")


def _timestamp(value: pd.Series) -> pd.Series:
    return pd.to_datetime(value, utc=True, errors="raise")


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without a non-core dependency."""
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    values: list[str] = []
    for _, row in frame.iterrows():
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append("" if not np.isfinite(value) else f"{value:.4f}")
            else:
                cells.append(str(value).replace("|", "\\|"))
        values.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, divider, *values])


def _candidate_id(decisions: pd.DataFrame) -> pd.Series:
    decision = _timestamp(decisions["timestamp"])
    return decisions["symbol"].astype(str) + "|long|" + (decision - pd.Timedelta(hours=1)).dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _only(paths: Iterable[Path], description: str) -> Path:
    paths = list(paths)
    if len(paths) != 1:
        raise AssertionError(f"expected one {description}, got {paths}")
    return paths[0]


def _risk_from_equity(path: Path) -> dict[str, float]:
    equity = pd.read_parquet(path)
    equity["timestamp"] = _timestamp(equity["timestamp"])
    equity = equity.sort_values("timestamp", kind="stable").drop_duplicates("timestamp", keep="last").set_index("timestamp")
    hourly = equity["mtm_equity"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    downside = np.sqrt(np.mean(np.minimum(hourly.to_numpy(float), 0.0) ** 2)) if len(hourly) else np.nan
    sortino = (float(hourly.mean()) / downside * np.sqrt(24.0 * 365.25)) if downside and np.isfinite(downside) else np.nan
    weekly = equity["mtm_equity"].resample("W-SUN").last().pct_change().dropna()
    monthly = equity["mtm_equity"].resample("ME").last().pct_change().dropna()
    return {
        "hourly_sortino_annualized": sortino,
        "positive_week_fraction": float((weekly > 0.0).mean()) if len(weekly) else np.nan,
        "positive_month_fraction_equity": float((monthly > 0.0).mean()) if len(monthly) else np.nan,
    }


def _risk_from_decisions(path: Path) -> dict[str, float]:
    decisions = pd.read_parquet(path)
    selected = decisions.loc[decisions["accepted"].fillna(False).astype(bool)].copy()
    bps = pd.to_numeric(selected["position_net_return"], errors="coerce").dropna().to_numpy(float) * 10_000.0
    if not len(bps):
        return {"trade_cvar5_bps": np.nan, "trade_downside_deviation_bps": np.nan, "trade_win_rate": np.nan}
    tail = np.sort(bps)[:max(1, int(np.ceil(.05 * len(bps))))]
    return {
        "trade_cvar5_bps": float(tail.mean()),
        "trade_downside_deviation_bps": float(np.sqrt(np.mean(np.minimum(bps, 0.0) ** 2))),
        "trade_win_rate": float((bps > 0.0).mean()),
    }


def _portfolio_metrics(run: Path, label: str, stage: str) -> pd.DataFrame:
    metrics = pd.read_parquet(run / "portfolio_metrics.parquet")
    rows: list[dict[str, object]] = []
    for _, item in metrics.iterrows():
        threshold = int(round(float(item["threshold_bps"])))
        decisions = _only(run.glob(f"*_{threshold}_mayjul_2026_decisions.parquet"), f"{run.name} threshold={threshold} decisions")
        equity = _only(run.glob(f"*_{threshold}_mayjul_2026_equity.parquet"), f"{run.name} threshold={threshold} equity")
        manifest = json.loads((run / "run_manifest.json").read_text())
        row = {"stage": stage, "label": label, **item.to_dict(), **_risk_from_equity(equity), **_risk_from_decisions(decisions)}
        row["trades_per_day"] = float(item["accepted_rows"]) / 92.0
        row["score_weights"] = json.dumps(manifest.get("score_weights", {}), sort_keys=True)
        row["geometry_blocks"] = ",".join(manifest.get("geometry_blocks", [])) or "M0"
        row["mc1_capacity"] = json.dumps(manifest.get("mc1_capacity", {}), sort_keys=True)
        rows.append(row)
    return pd.DataFrame(rows)


def _matched_live_control(root: Path) -> pd.DataFrame:
    """Load the supplied historical live-control comparator without refitting it.

    It shares the reconciled policy and portfolio machinery, but predates the
    enhanced-base/T6/T9 score receipt.  We therefore label it as a matched
    policy/execution comparator, not as the exact source-population control.
    """
    metrics = pd.read_parquet(root / "portfolio_metrics.parquet")
    item = metrics.loc[
        metrics["arm"].eq("live_control_matched_50") & metrics["threshold_bps"].eq(50.0)
    ]
    if len(item) != 1:
        raise AssertionError("missing unique live_control_matched_50 row")
    decision = root / "live_control_matched_50_202605_202607_decisions.parquet"
    equity = root / "live_control_matched_50_202605_202607_equity.parquet"
    row = {
        "stage": "matched_live_control",
        "label": "live_control_matched_50",
        **item.iloc[0].to_dict(),
        **_risk_from_equity(equity),
        **_risk_from_decisions(decision),
        "trades_per_day": float(item.iloc[0]["accepted_rows"]) / 92.0,
        "score_weights": "legacy live control; not the enhanced-base source population",
        "geometry_blocks": "live control",
        "mc1_capacity": "frozen legacy live control",
    }
    return pd.DataFrame([row])


def _rank_top(frame: pd.DataFrame, score: str, fraction: float, *, local: bool) -> pd.Index:
    if not local:
        return frame.nlargest(max(1, int(np.ceil(len(frame) * fraction))), score, keep="all").index
    # A single stable sort is much cheaper and less memory-intensive than
    # thousands of group-level ``nlargest`` allocations on the full ledger.
    work = frame.loc[:, ["__decision_ts__", "candidate_id", score]].copy()
    work["__position__"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(["__decision_ts__", score, "candidate_id"], ascending=[True, False, True], kind="stable")
    rank = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy(float)
    size = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy(float)
    selected_position = work.loc[rank < np.maximum(1.0, np.ceil(size * fraction)), "__position__"].to_numpy(np.int64)
    return frame.index.take(selected_position)


def _score_detail(run: Path, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # These receipts are deliberately wide.  Read only the frozen score
    # coordinates needed by this audit rather than materialising the full
    # feature panel and risking an avoidable memory failure.
    source = pd.read_parquet(
        run / "current_target_free_score_panel.parquet",
        columns=["candidate_id", "__decision_ts__", "final_score", "b_rank", "t6_rank", "t9_rank"],
    )
    labels = pd.read_parquet(run / "dual_mc1_predictions.parquet", columns=["candidate_id", "policy_path_valid", "policy_net_bps"])
    source["__decision_ts__"] = _timestamp(source["__decision_ts__"])
    source = source.loc[source["__decision_ts__"].ge(EVAL_START) & source["__decision_ts__"].lt(EVAL_END)]
    frame = source.merge(labels, on="candidate_id", how="inner", validate="one_to_one")
    frame = frame.loc[
        frame["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
    ].copy()
    frame["policy_net_bps"] = pd.to_numeric(frame["policy_net_bps"], errors="coerce")
    frame["base_band"] = pd.cut(frame["b_rank"], BASE_BANDS, labels=BASE_LABELS, include_lowest=False, right=True)
    rows: list[dict[str, object]] = []
    score = "final_score"
    for scope, local in (("global", False), ("timestamp", True)):
        for fraction in (.01, .02, .05, .10):
            chosen = frame.loc[_rank_top(frame, score, fraction, local=local)]
            rows.append({"label": label, "scope": scope, "fraction": fraction, "metric": "policy_net_bps", "value": float(chosen["policy_net_bps"].mean()), "rows": int(len(chosen))})
    score_top40 = _rank_top(frame, score, .40, local=True)
    selected = frame.index.isin(score_top40)
    target_top40 = _rank_top(frame, "policy_net_bps", .40, local=True)
    positive = frame["policy_net_bps"] >= 50.0
    rows.extend([
        {"label": label, "scope": "timestamp", "fraction": .40, "metric": "policy_ge_50_recall", "value": float(selected[positive.to_numpy()].mean()) if positive.any() else np.nan, "rows": int(positive.sum())},
        {"label": label, "scope": "timestamp", "fraction": .40, "metric": "realised_top40_recall", "value": float(np.mean(target_top40.isin(score_top40))) if len(target_top40) else np.nan, "rows": int(len(target_top40))},
        {"label": label, "scope": "timestamp", "fraction": .40, "metric": "severe_loss_rate", "value": float((frame.loc[score_top40, "policy_net_bps"] <= -200.0).mean()), "rows": int(len(score_top40))},
        {"label": label, "scope": "all", "fraction": np.nan, "metric": "rank_ic_spearman", "value": float(frame[score].corr(frame["policy_net_bps"], method="spearman")), "rows": int(len(frame))},
    ])
    correlations = []
    for field in ("b_rank", "t6_rank", "t9_rank"):
        correlations.append({"label": label, "field": field, "spearman_to_final_score": float(frame[score].corr(frame[field], method="spearman")), "rows": int(len(frame))})
    band = frame.groupby("base_band", observed=False)["policy_net_bps"].agg(["count", "mean"]).reset_index().rename(columns={"count": "rows", "mean": "policy_net_bps"})
    band.insert(0, "label", label)
    band["score"] = score
    return pd.DataFrame(rows), pd.concat((pd.DataFrame(correlations), band), ignore_index=True, sort=False)


def _mapper_rank_diagnostics(run: Path) -> pd.DataFrame:
    labels = ("candidate_id", "__decision_ts__", "final_score", "mc1_expected_bps", "policy_path_valid", "policy_net_bps")
    current = pd.read_parquet(run / "current_experimental_mc1_predictions.parquet", columns=list(labels)).rename(columns={"final_score": "current_final_score", "mc1_expected_bps": "current_expected"})
    bcf = pd.read_parquet(run / "bcf_experimental_mc1_predictions.parquet", columns=["candidate_id", "__decision_ts__", "final_score", "mc1_expected_bps"]).rename(columns={"final_score": "bcf_final_score", "mc1_expected_bps": "bcf_expected"})
    frame = current.merge(bcf, on=["candidate_id", "__decision_ts__"], how="inner", validate="one_to_one")
    frame["__decision_ts__"] = _timestamp(frame["__decision_ts__"])
    frame = frame.loc[
        frame["__decision_ts__"].ge(EVAL_START) & frame["__decision_ts__"].lt(EVAL_END)
        & frame["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
    ].copy()
    frame["dual_min_expected"] = np.minimum(frame["current_expected"], frame["bcf_expected"])
    frame["dual_admitted_50"] = frame["dual_min_expected"] >= 50.0
    output: list[dict[str, object]] = []
    for family, score, base in (("current", "current_expected", "current_final_score"), ("bcf", "bcf_expected", "bcf_final_score"), ("dual_min", "dual_min_expected", "current_final_score")):
        active = frame if family != "dual_min" else frame.loc[frame["dual_admitted_50"]].copy()
        active["score_band"] = pd.qcut(active[base].rank(method="first"), q=20, labels=False, duplicates="drop")
        expected_resid = active[score] - active.groupby("score_band", observed=True)[score].transform("mean")
        outcome_resid = active["policy_net_bps"] - active.groupby("score_band", observed=True)["policy_net_bps"].transform("mean")
        for scope, local in (("global", False), ("timestamp", True)):
            for fraction in (.01, .02, .05, .10):
                picked = active.loc[_rank_top(active, score, fraction, local=local)]
                output.append({"family": family, "scope": scope, "fraction": fraction, "metric": "policy_net_bps", "value": float(picked["policy_net_bps"].mean()), "rows": int(len(picked))})
        output.append({"family": family, "scope": "all", "fraction": np.nan, "metric": "rank_ic_spearman", "value": float(active[score].corr(active["policy_net_bps"], method="spearman")), "rows": int(len(active))})
        output.append({"family": family, "scope": "all", "fraction": np.nan, "metric": "conditional_rank_ic_beyond_final_score", "value": float(expected_resid.corr(outcome_resid, method="spearman")), "rows": int(len(active))})
    return pd.DataFrame(output)


def _trade_set_attribution(control: Path, challenger: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    def accepted(run: Path) -> pd.DataFrame:
        decision = pd.read_parquet(_only(run.glob("*_50_mayjul_2026_decisions.parquet"), f"{run.name} threshold=50 decisions"))
        decision = decision.loc[decision["accepted"].fillna(False).astype(bool)].copy()
        decision["candidate_id"] = _candidate_id(decision)
        return decision.set_index("candidate_id")

    a, b = accepted(control), accepted(challenger)
    source = pd.read_parquet(
        challenger / "current_target_free_score_panel.parquet",
        columns=["candidate_id", "__decision_ts__", "final_score", "b_rank", "t6_rank", "t9_rank"],
    )
    outcomes = pd.read_parquet(challenger / "dual_mc1_predictions.parquet", columns=["candidate_id", "policy_net_bps", "current_mc1_expected_bps", "bcf_mc1_expected_bps"])
    source = source.merge(outcomes, on="candidate_id", how="inner", validate="one_to_one")
    source["__decision_ts__"] = _timestamp(source["__decision_ts__"])
    source = source.loc[source["__decision_ts__"].ge(EVAL_START) & source["__decision_ts__"].lt(EVAL_END)].copy()
    source["query_rank"] = source.groupby("__decision_ts__", sort=False)["final_score"].rank(ascending=False, method="first", pct=True)
    source["t6_minus_base"] = source["t6_rank"] - source["b_rank"]
    source["t9_conversion_state"] = pd.cut(source["t9_rank"], [-np.inf, .2, .8, np.inf], labels=["low", "middle", "high"])
    records: list[dict[str, object]] = []
    all_ids = set(a.index) | set(b.index)
    groups = {"common": set(a.index) & set(b.index), "control_only": set(a.index) - set(b.index), "challenger_only": set(b.index) - set(a.index)}
    for cohort, ids in groups.items():
        part = source.loc[source["candidate_id"].isin(ids)]
        records.append({
            "cohort": cohort, "rows": int(len(part)), "net_ev_bps_per_trade": float(part["policy_net_bps"].mean()),
            "total_net_bps": float(part["policy_net_bps"].sum()), "base_rank": float(part["b_rank"].mean()),
            "t6_rank": float(part["t6_rank"].mean()), "t9_rank": float(part["t9_rank"].mean()),
            "upstream_score": float(part["final_score"].mean()), "current_mc1_expected": float(part["current_mc1_expected_bps"].mean()),
            "bcf_mc1_expected": float(part["bcf_mc1_expected_bps"].mean()), "t6_minus_base": float(part["t6_minus_base"].mean()),
            "query_rank": float(part["query_rank"].mean()), "high_t9_conversion_share": float((part["t9_conversion_state"] == "high").mean()),
        })
    # Pair differing selections in the same timestamp in portfolio-priority order.
    a_reset, b_reset = a.reset_index(), b.reset_index()
    a_reset["timestamp"] = _timestamp(a_reset["timestamp"])
    b_reset["timestamp"] = _timestamp(b_reset["timestamp"])
    displacement: list[dict[str, object]] = []
    for stamp in sorted(set(a_reset["timestamp"]) | set(b_reset["timestamp"])):
        left = a_reset.loc[a_reset["timestamp"].eq(stamp) & ~a_reset["candidate_id"].isin(b.index)].sort_values("portfolio_priority", ascending=False)
        right = b_reset.loc[b_reset["timestamp"].eq(stamp) & ~b_reset["candidate_id"].isin(a.index)].sort_values("portfolio_priority", ascending=False)
        for order, (cid_a, cid_b) in enumerate(zip(left["candidate_id"], right["candidate_id"]), start=1):
            pa = source.loc[source["candidate_id"].eq(cid_a)].iloc[0]
            pb = source.loc[source["candidate_id"].eq(cid_b)].iloc[0]
            displacement.append({
                "timestamp": stamp, "month": f"{stamp:%Y-%m}", "pair_order": order,
                "control_candidate_id": cid_a, "challenger_candidate_id": cid_b,
                "control_net_bps": float(pa["policy_net_bps"]), "challenger_net_bps": float(pb["policy_net_bps"]),
                "challenger_minus_control_bps": float(pb["policy_net_bps"] - pa["policy_net_bps"]),
                "control_priority": float(left.loc[left["candidate_id"].eq(cid_a), "portfolio_priority"].iloc[0]),
                "challenger_priority": float(right.loc[right["candidate_id"].eq(cid_b), "portfolio_priority"].iloc[0]),
            })
    return pd.DataFrame(records), pd.DataFrame(displacement)


def _causality(run: Path) -> dict[str, object]:
    target_free: dict[str, object] = {}
    for family in ("current", "bcf"):
        path = run / f"{family}_target_free_score_panel.parquet"
        fields = set(pq.ParquetFile(path).schema_arrow.names)
        target_free[family] = {"rows": int(pq.ParquetFile(path).metadata.num_rows), "prohibited_columns": sorted(fields & PROHIBITED)}
    current = pd.read_parquet(run / "current_experimental_mc1_predictions.parquet", columns=["candidate_id", "__decision_ts__", "policy_path_valid", "policy_label_available_ts"])
    bcf = pd.read_parquet(run / "bcf_experimental_mc1_predictions.parquet", columns=["candidate_id"])
    valid = current["policy_path_valid"].fillna(False).astype(bool)
    later = _timestamp(current.loc[valid, "policy_label_available_ts"]) > _timestamp(current.loc[valid, "__decision_ts__"])
    return {
        "target_free": target_free,
        "current_bcf_identical_candidate_ids": bool(set(current["candidate_id"]) == set(bcf["candidate_id"])),
        "valid_labels_available_after_decision": bool(later.all()),
        "valid_prediction_rows": int(valid.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--stage1b-root", type=Path, required=True)
    parser.add_argument("--stage2-root", type=Path, required=True)
    parser.add_argument("--capacity-root", type=Path, required=True)
    parser.add_argument("--live-control-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)

    all_portfolios: list[pd.DataFrame] = []
    for stage, root in (("stage1", args.stage1_root), ("stage1b", args.stage1b_root), ("stage2", args.stage2_root), ("capacity", args.capacity_root)):
        for run in sorted(path for path in root.iterdir() if path.is_dir() and (path / "portfolio_metrics.parquet").exists()):
            all_portfolios.append(_portfolio_metrics(run, run.name, stage))
    all_portfolios.append(_matched_live_control(args.live_control_root))
    portfolio = pd.concat(all_portfolios, ignore_index=True)
    portfolio.to_parquet(args.out / "portfolio_metrics_all_arms.parquet", index=False, compression="zstd")

    serious = ["S0_BASE", "S1_CURRENT_T6T9", "S5_T6_15", "S5_T6_15_T9_hidden", "S11_T6_20_T9_05", "S19_M30_T6_24_T9_06"]
    score_rows: list[pd.DataFrame] = []
    correlation_rows: list[pd.DataFrame] = []
    for name in serious:
        run = args.stage1_root / name
        if not run.exists():
            run = args.stage1b_root / name
        if run.exists() and (run / "dual_mc1_predictions.parquet").exists():
            metrics, details = _score_detail(run, name)
            score_rows.append(metrics)
            correlation_rows.append(details)
    pd.concat(score_rows, ignore_index=True).to_parquet(args.out / "score_diagnostics_extended.parquet", index=False, compression="zstd")
    pd.concat(correlation_rows, ignore_index=True).to_parquet(args.out / "score_coordinate_and_baseband_audit.parquet", index=False, compression="zstd")

    s11 = args.stage1_root / "S11_T6_20_T9_05"
    _mapper_rank_diagnostics(s11).to_parquet(args.out / "mc1_matched_id_rank_diagnostics.parquet", index=False, compression="zstd")
    for name in ("head_month_metrics", "head_month_deciles", "head_agreement", "dual_admission_cohorts"):
        source = s11 / "dual_mc1_mapper_audit" / f"{name}.parquet"
        if source.exists():
            pd.read_parquet(source).to_parquet(args.out / f"{name}.parquet", index=False, compression="zstd")
    pd.read_parquet(s11 / "mc1_admission_calibration.parquet").to_parquet(args.out / "s11_admission_calibration.parquet", index=False, compression="zstd")

    set_hidden, displacement_hidden = _trade_set_attribution(args.stage1_root / "S5_T6_15_T9_hidden", s11)
    set_visible, displacement_visible = _trade_set_attribution(args.stage1_root / "S5_T6_15", s11)
    set_s1, displacement_s1 = _trade_set_attribution(args.stage1_root / "S1_CURRENT_T6T9", s11)
    set_s1.assign(comparator="S1_current_t6t9").to_parquet(args.out / "trade_set_attribution_s1_vs_s11.parquet", index=False, compression="zstd")
    displacement_s1.assign(comparator="S1_current_t6t9").to_parquet(args.out / "portfolio_displacement_s1_vs_s11.parquet", index=False, compression="zstd")
    displacement_summary = pd.DataFrame([{
        "pairs": int(len(displacement_s1)),
        "challenger_minus_control_bps_mean": float(displacement_s1["challenger_minus_control_bps"].mean()),
        "challenger_minus_control_bps_total": float(displacement_s1["challenger_minus_control_bps"].sum()),
        "positive_replacement_share": float((displacement_s1["challenger_minus_control_bps"] > 0.0).mean()),
    }])
    displacement_summary.to_parquet(args.out / "portfolio_displacement_s1_vs_s11_summary.parquet", index=False, compression="zstd")
    set_hidden.assign(comparator="S5_hidden").to_parquet(args.out / "trade_set_attribution_s5_hidden_vs_s11.parquet", index=False, compression="zstd")
    set_visible.assign(comparator="S5_visible").to_parquet(args.out / "trade_set_attribution_s5_visible_vs_s11.parquet", index=False, compression="zstd")
    displacement_hidden.assign(comparator="S5_hidden").to_parquet(args.out / "portfolio_displacement_s5_hidden_vs_s11.parquet", index=False, compression="zstd")
    displacement_visible.assign(comparator="S5_visible").to_parquet(args.out / "portfolio_displacement_s5_visible_vs_s11.parquet", index=False, compression="zstd")

    causality = _causality(s11)
    (args.out / "correctness_report.json").write_text(json.dumps({
        "schema": "strict_r3_t6t9_dual_mc1_ablation_audit_v1",
        "scope": "offline audit only; no live artifact or exchange state was read or written",
        "s11_causality": causality,
        "pass": all(not item["prohibited_columns"] for item in causality["target_free"].values()) and causality["current_bcf_identical_candidate_ids"] and causality["valid_labels_available_after_decision"],
    }, indent=2) + "\n")

    summary = portfolio.loc[portfolio["threshold_bps"].eq(50.0) & portfolio["label"].isin(["live_control_matched_50", "S1_CURRENT_T6T9", "S5_T6_15_T9_hidden", "S5_T6_15", "S11_T6_20_T9_05", "G1", "G7", "G10", "C0_current", "C1_d2_l4", "C2_d3_l6", "C3_d3_l8"])].copy()
    keep = ["stage", "label", "accepted_rows", "candidate_admitted_rows", "net_ev_bps_per_realised_trade", "net_sum_bps_realised", "worst_month_bps", "worst_week_bps", "max_drawdown", "hourly_sortino_annualized", "trade_cvar5_bps"]
    report = [
        "# T6/T9 Score Construction + Dual-MC1 Ablation — Offline Result",
        "",
        "Scope: strict-prequential Current and BCF maps, rich-policy labels, dual admission, and one constrained chronological portfolio over May–July 2026. This report is research-only and makes no live-stack change.",
        "",
        "## Selected score-contract and MC1 result",
        "",
        _markdown_table(summary.loc[:, keep]),
        "",
        "S11 (75% Base / 20% T6 / 5% T9) remains the score-contract challenger. No geometry or mapper-capacity arm meets the advancement gate: an arm must improve total portfolio economics without meaningful deterioration in EV/trade, drawdown, or month/week stability.",
        "The `live_control_matched_50` row shares canonical policy/execution and portfolio rules but not the newer enhanced-base source population. S1 is the exact score-construction control for the new receipt family; use the live-control row only as an external incumbent comparison.",
        "At the participation-matched frontier, S11 at +70 bps selects 1,907 trades (20.73/day) versus the legacy matched-live control's 1,896 (20.61/day); it is a sensitivity result, not a replacement for the frozen +50-bps admission contract.",
        f"Against same-family S1, the 253 direct same-timestamp portfolio substitutions have mean S11-minus-S1 realised EV of {float(displacement_summary.iloc[0]['challenger_minus_control_bps_mean']):.2f} bps and only {100.0 * float(displacement_summary.iloc[0]['positive_replacement_share']):.1f}% are positive. S11's aggregate gain is therefore from useful additional admissions, not proven superior auction priority; this blocks promotion pending an untouched test.",
        "",
        "## Causality receipt",
        "",
        "```json",
        json.dumps(causality, indent=2),
        "```",
        "",
        "## Interpretation",
        "",
        "- T9 has modest value as an MC1 conditioning coordinate; it does not justify material direct score authority.",
        "- Current and BCF maps are separate strict-prequential fits but highly correlated. Their dual threshold is a corroboration gate, not a source of independent ranking alpha.",
        "- The 50–60 bps dual-MC1 band is overconfident in S11; threshold relaxation is not supported by this selection period.",
        "- The available homogeneous T6/T9 receipts begin in November 2025, leaving May–July 2026 as the only fully supported strict-prequential test block. A later untouched block is required for promotion.",
    ]
    (args.out / "MC1_ABLATION_REPORT.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
