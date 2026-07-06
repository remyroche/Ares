#!/usr/bin/env python3
"""Summarize why strong four-head OOF metrics do not translate to OOS PnL."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN_ID = "20260629_050000_lgbm_mda"
MONTHLY_WF_ID = "20260701_130000_single_head_monthly_walkforward_oos"
SOURCE_ROOT = ROOT / "data_perp" / "artifacts" / SOURCE_RUN_ID
MONTHLY_ROOT = ROOT / "data_perp" / "reports" / MONTHLY_WF_ID
OUT_DIR = ROOT / "data_perp" / "reports" / f"{SOURCE_RUN_ID}_four_head_oof_policy_gap"

EXACT_OOS_WINDOWS = {
    "2026-04_exact_oos": ("2026-04-16T00:00:00Z", "2026-05-01T00:00:00Z"),
    "2026-05_exact_oos": ("2026-05-16T00:00:00Z", "2026-06-01T00:00:00Z"),
    "2026-06_exact_oos": ("2026-06-16T00:00:00Z", "2026-07-01T00:00:00Z"),
}

RANK_SLICES = {
    "all": 0.0,
    "top_15": 0.85,
    "top_5": 0.95,
    "top_1": 0.99,
}


def _head_name(strategy_id: str) -> str:
    if strategy_id.startswith("long_dist"):
        return "long_dist"
    if strategy_id.startswith("long_bars"):
        return "long_bars"
    if strategy_id.startswith("short_boll"):
        return "short_boll"
    if strategy_id.startswith("short_asset"):
        return "short_asset"
    return strategy_id[:40]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_num(series: pd.Series, op: str) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return float("nan")
    if op == "sum":
        return float(vals.sum())
    if op == "mean":
        return float(vals.mean())
    if op == "median":
        return float(vals.median())
    raise ValueError(op)


def _summarize_returns(frame: pd.DataFrame, value_col: str, hit_col: str | None = None) -> dict[str, Any]:
    if frame.empty or value_col not in frame.columns:
        return {
            "rows": int(len(frame)),
            "sum_return": 0.0,
            "mean_return": float("nan"),
            "median_return": float("nan"),
            "hit_rate": float("nan"),
        }
    vals = pd.to_numeric(frame[value_col], errors="coerce")
    hit = pd.to_numeric(frame[hit_col], errors="coerce") > 0 if hit_col and hit_col in frame.columns else vals > 0
    return {
        "rows": int(len(frame)),
        "sum_return": float(vals.fillna(0.0).sum()),
        "mean_return": float(vals.mean()) if vals.notna().any() else float("nan"),
        "median_return": float(vals.median()) if vals.notna().any() else float("nan"),
        "hit_rate": float(hit.mean()) if len(hit) else float("nan"),
    }


def build_oof_metrics(registry: pd.DataFrame) -> pd.DataFrame:
    metrics = _read_json(SOURCE_ROOT / "meta_oof" / "meta_head_metrics.json")
    rows: list[dict[str, Any]] = []
    for rec in registry.to_dict("records"):
        sid = rec["strategy_id"]
        m = metrics.get(sid) or metrics.get(f"{sid}_tbm_clf") or {}
        rows.append(
            {
                "head": _head_name(sid),
                "strategy_id": sid,
                "side": rec.get("side") or rec.get("trade_side"),
                "n_samples": m.get("n_samples"),
                "base_rate": m.get("base_rate", m.get("hit_rate")),
                "precision_20": m.get("precision_20"),
                "precision_10": m.get("precision_10"),
                "precision_5": m.get("precision_5"),
                "precision_1": m.get("precision_1"),
                "auc": m.get("auc"),
                "pr_auc": m.get("pr_auc"),
                "ic": m.get("ic"),
            }
        )
    return pd.DataFrame(rows)


def build_oof_window_slices(registry: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sid in registry["strategy_id"].astype(str):
        path = SOURCE_ROOT / "meta_oof" / f"meta_oof_{sid}_tbm_clf.parquet"
        if not path.exists():
            rows.append({"head": _head_name(sid), "strategy_id": sid, "window": "missing_oof_file"})
            continue
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        score_col = "oof_meta_clf" if "oof_meta_clf" in df.columns else "oof_pred"
        df["_score"] = pd.to_numeric(df[score_col], errors="coerce")
        df["_rank_pct"] = df["_score"].rank(pct=True, method="average")
        windows = {"full_oof": (None, None), **EXACT_OOS_WINDOWS}
        for window, (start, end) in windows.items():
            if start is None:
                sub = df
            else:
                sub = df[
                    (df["timestamp"] >= pd.Timestamp(start))
                    & (df["timestamp"] < pd.Timestamp(end))
                ]
            for rank_slice, threshold in RANK_SLICES.items():
                ss = sub[sub["_rank_pct"] >= float(threshold)]
                summary = _summarize_returns(ss, "return", hit_col="y_bin")
                rows.append(
                    {
                        "head": _head_name(sid),
                        "strategy_id": sid,
                        "window": window,
                        "rank_slice": rank_slice,
                        "rank_threshold": threshold,
                        "score_col": score_col,
                        "period_start": ss["timestamp"].min() if len(ss) else pd.NaT,
                        "period_end": ss["timestamp"].max() if len(ss) else pd.NaT,
                        "mean_score": _metric_num(ss["_score"], "mean") if len(ss) else float("nan"),
                        **summary,
                    }
                )
    return pd.DataFrame(rows)


def build_source_candidate_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = SOURCE_ROOT / "simple_policy_optimiser" / "simple_policy_candidates_broad.parquet"
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["head"] = df["strategy_id"].astype(str).map(_head_name)
    windows = {
        "candidate_artifact_all": (df["timestamp"].min(), df["timestamp"].max() + pd.Timedelta(seconds=1)),
        **{k: (pd.Timestamp(s), pd.Timestamp(e)) for k, (s, e) in EXACT_OOS_WINDOWS.items()},
    }
    rows: list[dict[str, Any]] = []
    reason_rows: list[dict[str, Any]] = []
    for window, (start, end) in windows.items():
        sub = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
        for head in ["ALL", *sorted(sub["head"].dropna().unique())]:
            g = sub if head == "ALL" else sub[sub["head"] == head]
            for rank_slice, threshold in RANK_SLICES.items():
                gg = g[pd.to_numeric(g["rank_pct"], errors="coerce") >= float(threshold)]
                rows.append(
                    {
                        "scope": "source_candidates_broad",
                        "window": window,
                        "head": head,
                        "rank_slice": rank_slice,
                        "rank_threshold": threshold,
                        "rows": int(len(gg)),
                        "net_sum": _metric_num(gg["net_return"], "sum") if len(gg) else 0.0,
                        "net_mean": _metric_num(gg["net_return"], "mean") if len(gg) else float("nan"),
                        "gross_sum": _metric_num(gg["gross_return"], "sum") if len(gg) else 0.0,
                        "gross_mean": _metric_num(gg["gross_return"], "mean") if len(gg) else float("nan"),
                        "hit_rate": float((pd.to_numeric(gg["net_return"], errors="coerce") > 0).mean()) if len(gg) else float("nan"),
                        "cost_drag_sum": (
                            _metric_num(gg["gross_return"], "sum") - _metric_num(gg["net_return"], "sum")
                            if len(gg)
                            else 0.0
                        ),
                        "mean_expected_spread_bps": _metric_num(gg["expected_spread_bps"], "mean") if "expected_spread_bps" in gg.columns and len(gg) else float("nan"),
                        "mean_exit_spread_cost_bps": _metric_num(gg["exit_spread_cost_bps"], "mean") if "exit_spread_cost_bps" in gg.columns and len(gg) else float("nan"),
                    }
                )
            if "simple_policy_exit_reason" in g.columns and len(g):
                counts = g["simple_policy_exit_reason"].astype(str).value_counts(normalize=False)
                for reason, count in counts.items():
                    reason_rows.append(
                        {
                            "scope": "source_candidates_broad",
                            "window": window,
                            "head": head,
                            "exit_reason": reason,
                            "count": int(count),
                            "rate": float(count / len(g)),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(reason_rows)


def build_monthly_policy_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparison = pd.read_csv(MONTHLY_ROOT / "policy_comparison" / "monthly_oos_policy_comparison.csv")
    exact_monthly = pd.read_csv(MONTHLY_ROOT / "policy_baseline_investigation" / "policy_baseline_monthly.csv")
    aggregate = pd.read_csv(MONTHLY_ROOT / "policy_baseline_investigation" / "policy_baseline_aggregate.csv")
    return comparison, exact_monthly, aggregate


def build_source_validation_summary() -> dict[str, Any]:
    summary_path = MONTHLY_ROOT / "summary.json"
    summary = _read_json(summary_path)
    metrics_prefix = "\n".join(
        (SOURCE_ROOT / "policy_optimisation_oos_metrics_perps.json")
        .read_text(encoding="utf-8")
        .splitlines()[:42]
    )
    return {
        "source_run_id": SOURCE_RUN_ID,
        "monthly_walkforward_experiment_id": MONTHLY_WF_ID,
        "monthly_walkforward_strategy_id": summary.get("strategy_id"),
        "monthly_walkforward_head": _head_name(str(summary.get("strategy_id", ""))),
        "monthly_walkforward_contract": summary.get("contract", {}),
        "policy_optimisation_oos_metrics_prefix": metrics_prefix,
    }


def _fmt_float(value: Any, digits: int = 6) -> str:
    try:
        val = float(value)
    except Exception:
        return ""
    if not math.isfinite(val):
        return ""
    return f"{val:.{digits}f}"


def write_report(
    oof_metrics: pd.DataFrame,
    oof_windows: pd.DataFrame,
    candidates: pd.DataFrame,
    candidate_exit_reasons: pd.DataFrame,
    monthly_comparison: pd.DataFrame,
    exact_monthly: pd.DataFrame,
    aggregate: pd.DataFrame,
    validation: dict[str, Any],
) -> None:
    source_candidate_focus = candidates[
        (candidates["head"] == "ALL")
        & (candidates["rank_slice"].isin(["all", "top_15"]))
        & (candidates["window"].isin(["candidate_artifact_all", "2026-05_exact_oos", "2026-06_exact_oos"]))
    ].copy()
    source_candidate_focus = source_candidate_focus[
        [
            "window",
            "rank_slice",
            "rows",
            "net_sum",
            "net_mean",
            "hit_rate",
            "gross_sum",
            "cost_drag_sum",
            "mean_expected_spread_bps",
            "mean_exit_spread_cost_bps",
        ]
    ]

    oof_focus = oof_windows[
        (oof_windows["rank_slice"] == "top_15")
        & (oof_windows["window"].isin(["full_oof", "2026-04_exact_oos", "2026-05_exact_oos", "2026-06_exact_oos"]))
    ][
        ["head", "window", "rows", "hit_rate", "mean_return", "sum_return", "mean_score"]
    ].copy()

    single_head_focus = monthly_comparison[
        monthly_comparison["scope"].eq("single_head_monthly_walkforward")
    ][
        [
            "eval_month",
            "policy",
            "oos_window_start",
            "oos_window_end",
            "n_trades",
            "net_pnl",
            "mean_net_trade",
            "hit_rate",
            "oof_top15_hit_rate",
            "oof_top15_mean_return",
        ]
    ].copy()

    full_month_overlay = monthly_comparison[
        monthly_comparison["scope"].eq("source_run_four_head_overlay")
    ][
        [
            "eval_month",
            "policy",
            "oos_window_start",
            "oos_window_end",
            "n_trades",
            "net_pnl",
            "mean_net_trade",
            "hit_rate",
        ]
    ].copy()

    exact_overlay = exact_monthly[
        exact_monthly["scope"].eq("source_run_selected_rows_exact_oos_window")
    ].copy()
    exact_overlay_focus = (
        exact_overlay.sort_values(["eval_month", "net_pnl"], ascending=[True, False])
        .groupby("eval_month", observed=True)
        .head(5)[
            [
                "eval_month",
                "policy",
                "n_trades",
                "net_pnl",
                "mean_net_trade",
                "hit_rate",
                "mean_expected_spread_bps",
                "mean_exit_spread_cost_bps",
            ]
        ]
    )

    exact_overlay_month_stats = exact_overlay.groupby("eval_month", observed=True).agg(
        policies=("policy", "nunique"),
        positive_policies=("net_pnl", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
        best_net_pnl=("net_pnl", "max"),
        worst_net_pnl=("net_pnl", "min"),
    ).reset_index()

    aggregate_focus = aggregate[
        aggregate["scope"].isin(["single_head_monthly_walkforward", "source_run_selected_rows_exact_oos_window"])
    ].sort_values("net_pnl", ascending=False).head(12)

    report = [
        "# Four-Head OOF vs Policy-OOS Gap Investigation",
        "",
        f"Source run: `{SOURCE_RUN_ID}`.",
        f"Monthly walk-forward report: `{MONTHLY_WF_ID}`.",
        "",
        "## Executive Diagnosis",
        "",
        "The gap is not caused by the four classifiers suddenly having poor OOF ranking. The gap is caused by comparing different layers of evidence:",
        "",
        "1. The source-run policy metrics were optimized/reported from `meta_oof` training-window predictions, not untouched policy-OOS predictions.",
        "2. The Apr-May-Jun walk-forward policy-OOS experiment only retrained/scored one selected head (`long_dist`), not the full four-head portfolio.",
        "3. OOF label metrics remain very strong, but executable policy-OOS PnL is negative after delayed entry, spread, stop fills, concurrency, and exit geometry.",
        "4. Positive source-run overlay numbers are date-window sensitive: full-month May/June overlays are positive, while the exact June held-out window (`2026-06-16..2026-07-01`) is negative for every selected-row overlay scanned.",
        "5. The selected-row overlay artifacts are also not fully comparable to the recomputed simple-policy simulator: their exact-window rows report `exit_spread_cost_bps=0`, while source candidate rows include exit spread costs.",
        "",
        "## Four OOF Heads",
        "",
        oof_metrics.drop(columns=["strategy_id"]).to_markdown(index=False),
        "",
        "## OOF Top-15 By Window",
        "",
        "These are label/OOF returns, not executable replay PnL.",
        "",
        oof_focus.to_markdown(index=False),
        "",
        "## Single-Head Monthly Policy-OOS",
        "",
        f"The monthly walk-forward artifact selected `{validation['monthly_walkforward_head']}` only. It is useful as clean policy-OOS evidence for that head, but it is not a four-head portfolio test.",
        "",
        single_head_focus.to_markdown(index=False),
        "",
        "## Source Candidate Economics",
        "",
        "These are source-run candidate rows from `simple_policy_candidates_broad.parquet`, not the monthly single-head walk-forward rows.",
        "",
        source_candidate_focus.to_markdown(index=False),
        "",
        "## Full-Month Overlay vs Exact Held-Out Overlay",
        "",
        "Full-month overlay rows can be positive because they include earlier June. The exact held-out June window starts on 2026-06-16.",
        "",
        "### Full-Month Overlay Rows",
        "",
        full_month_overlay.to_markdown(index=False),
        "",
        "### Exact-Window Overlay Positivity",
        "",
        exact_overlay_month_stats.to_markdown(index=False),
        "",
        "### Best Exact-Window Overlay Rows Per Month",
        "",
        exact_overlay_focus.to_markdown(index=False),
        "",
        "## Aggregate Comparison Rows",
        "",
        aggregate_focus.to_markdown(index=False),
        "",
        "## Candidate Exit-Reason Mix",
        "",
        candidate_exit_reasons[
            candidate_exit_reasons["window"].isin(["2026-05_exact_oos", "2026-06_exact_oos"])
            & candidate_exit_reasons["head"].eq("ALL")
        ].to_markdown(index=False),
        "",
        "## Source Validation Evidence",
        "",
        "The top-level source-validation block from `policy_optimisation_oos_metrics_perps.json` states the policy metrics used meta-OOF predictions, not policy-OOS predictions:",
        "",
        "```text",
        validation["policy_optimisation_oos_metrics_prefix"],
        "```",
        "",
        "## Conclusion",
        "",
        "The best-supported explanation is a protocol and objective mismatch, plus execution friction. The OOF classifiers rank label outcomes well, but the reported policy layer was selected on training-window OOF evidence and then evaluated under different executable assumptions and date windows. The strongest clean policy-OOS evidence currently available is the one-head monthly walk-forward, and that evidence is negative despite high OOF top-15 hit rates. To close this gap, the next evaluation should produce a true four-head Apr-May-Jun policy-OOS replay with the same date windows, the same exit-cost model, and no source-run selected-row reuse.",
        "",
    ]
    (OUT_DIR / "gap_diagnosis_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    registry = pd.read_csv(SOURCE_ROOT / "strategy_registry" / "deployed_four_heads_perps.csv")
    oof_metrics = build_oof_metrics(registry)
    oof_windows = build_oof_window_slices(registry)
    candidates, candidate_exit_reasons = build_source_candidate_tables()
    monthly_comparison, exact_monthly, aggregate = build_monthly_policy_tables()
    validation = build_source_validation_summary()

    outputs = {
        "oof_metrics": OUT_DIR / "four_head_oof_metrics.csv",
        "oof_window_slices": OUT_DIR / "four_head_oof_window_slices.csv",
        "source_candidates": OUT_DIR / "source_candidate_window_economics.csv",
        "source_candidate_exit_reasons": OUT_DIR / "source_candidate_exit_reason_mix.csv",
        "monthly_policy_comparison": OUT_DIR / "monthly_policy_comparison_snapshot.csv",
        "exact_policy_monthly": OUT_DIR / "exact_window_policy_monthly_snapshot.csv",
        "exact_policy_aggregate": OUT_DIR / "exact_window_policy_aggregate_snapshot.csv",
        "source_validation": OUT_DIR / "source_validation_summary.json",
        "report": OUT_DIR / "gap_diagnosis_report.md",
    }
    oof_metrics.to_csv(outputs["oof_metrics"], index=False)
    oof_windows.to_csv(outputs["oof_window_slices"], index=False)
    candidates.to_csv(outputs["source_candidates"], index=False)
    candidate_exit_reasons.to_csv(outputs["source_candidate_exit_reasons"], index=False)
    monthly_comparison.to_csv(outputs["monthly_policy_comparison"], index=False)
    exact_monthly.to_csv(outputs["exact_policy_monthly"], index=False)
    aggregate.to_csv(outputs["exact_policy_aggregate"], index=False)
    outputs["source_validation"].write_text(json.dumps(_json_safe(validation), indent=2), encoding="utf-8")
    write_report(
        oof_metrics=oof_metrics,
        oof_windows=oof_windows,
        candidates=candidates,
        candidate_exit_reasons=candidate_exit_reasons,
        monthly_comparison=monthly_comparison,
        exact_monthly=exact_monthly,
        aggregate=aggregate,
        validation=validation,
    )
    manifest = {
        "generated_by": Path(__file__).name,
        "source_run_id": SOURCE_RUN_ID,
        "monthly_walkforward_id": MONTHLY_WF_ID,
        "outputs": {k: str(v) for k, v in outputs.items()},
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    print(outputs["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
