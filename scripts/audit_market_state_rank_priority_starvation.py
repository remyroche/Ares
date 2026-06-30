#!/usr/bin/env python3
"""Audit why market-state head priority can or cannot rescue global ranking.

This is a diagnostic companion to the market-state head-priority learner.  It
does not train models or replay trades.  It compares the current timestamp-rank
T1 contract against the causal global-over-time rank challenger and separates:

* pre-auction rank/threshold eligibility;
* path-dependent portfolio rejection reasons;
* accepted trade concentration by head;
* whether learned priority schedules changed accepted trades.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_TIMESTAMP_DIR = Path(
    "data_perp/artifacts/reliability_blend_T1_repaired_static_baseline_20260625_jun15_22"
)
DEFAULT_GLOBAL_DIR = Path(
    "data_perp/artifacts/reliability_blend_T1_global_rank_challenger_20260626_jun15_22_v1"
)
DEFAULT_TIMESTAMP_PRIORITY_DIR = Path(
    "data_perp/reports/market_state_head_priority_learning_actionaware_20260626_jun15_22"
)
DEFAULT_GLOBAL_PRIORITY_DIR = Path(
    "data_perp/reports/market_state_head_priority_learning_actionaware_globalrank_20260626_jun15_22_v2"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/market_state_rank_priority_starvation_audit_20260626_jun15_22"
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _head_from_strategy(strategy_id: Any) -> str:
    text = str(strategy_id)
    if text.startswith("short_boll"):
        return "short_boll"
    if text.startswith("short_asset"):
        return "short_asset"
    if text.startswith("long_bars"):
        return "long_bars"
    if text.startswith("long_dist"):
        return "long_dist"
    return text.split("_", 2)[0] if text else "unknown"


def _ensure_head(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "head" not in out.columns and "strategy_id" in out.columns:
        out["head"] = out["strategy_id"].map(_head_from_strategy)
    return out


def _decision_key(frame: pd.DataFrame) -> pd.Series:
    cols = [c for c in ("timestamp", "symbol", "strategy_id", "side", "head") if c in frame.columns]
    if not cols:
        return pd.Series(np.arange(len(frame)), index=frame.index).astype(str)
    values: list[pd.Series] = []
    for col in cols:
        if col == "timestamp":
            values.append(pd.to_datetime(frame[col], utc=True, errors="coerce").astype(str))
        else:
            values.append(frame[col].astype(str))
    out = values[0]
    for value in values[1:]:
        out = out.str.cat(value, sep="|")
    return out


def _rank_stats(frame: pd.DataFrame, prefix: str) -> dict[str, Any]:
    rank = pd.to_numeric(frame.get("normalized_rank_score"), errors="coerce")
    out: dict[str, Any] = {
        f"{prefix}_rows": int(len(frame)),
        f"{prefix}_rank_mean": float(rank.mean()) if len(frame) else float("nan"),
        f"{prefix}_rank_p50": float(rank.quantile(0.50)) if len(frame) else float("nan"),
        f"{prefix}_rank_p90": float(rank.quantile(0.90)) if len(frame) else float("nan"),
        f"{prefix}_rank_max": float(rank.max()) if len(frame) else float("nan"),
    }
    for cutoff in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        out[f"{prefix}_share_rank_ge_{int(cutoff * 100)}"] = float((rank >= cutoff).mean()) if len(frame) else float("nan")
    if "net_return" in frame.columns:
        returns = pd.to_numeric(frame["net_return"], errors="coerce")
        out[f"{prefix}_mean_net_return"] = float(returns.mean()) if len(frame) else float("nan")
        out[f"{prefix}_q05_net_return"] = float(returns.quantile(0.05)) if len(frame) else float("nan")
    return out


def candidate_starvation_stats(contract_name: str, artifact_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    simple_dir = artifact_dir / "simple_policy_optimiser"
    broad = _ensure_head(_load_parquet(simple_dir / "simple_policy_candidates_broad.parquet"))
    deployable = _ensure_head(_load_parquet(simple_dir / "simple_policy_candidates.parquet"))
    decisions = _ensure_head(_load_parquet(simple_dir / "portfolio_decisions.parquet"))
    accepted = _ensure_head(_load_parquet(simple_dir / "accepted_trades.parquet"))

    rows: list[dict[str, Any]] = []
    heads = sorted(
        set(broad.get("head", pd.Series(dtype=str)).dropna().astype(str))
        | set(deployable.get("head", pd.Series(dtype=str)).dropna().astype(str))
        | set(decisions.get("head", pd.Series(dtype=str)).dropna().astype(str))
        | set(accepted.get("head", pd.Series(dtype=str)).dropna().astype(str))
    )
    for head in heads:
        broad_h = broad.loc[broad["head"].astype(str).eq(head)].copy() if not broad.empty else pd.DataFrame()
        dep_h = deployable.loc[deployable["head"].astype(str).eq(head)].copy() if not deployable.empty else pd.DataFrame()
        dec_h = decisions.loc[decisions["head"].astype(str).eq(head)].copy() if not decisions.empty else pd.DataFrame()
        acc_h = accepted.loc[accepted["head"].astype(str).eq(head)].copy() if not accepted.empty else pd.DataFrame()
        if not broad_h.empty:
            rank = pd.to_numeric(broad_h.get("normalized_rank_score"), errors="coerce")
            base_threshold = pd.to_numeric(broad_h.get("base_strategy_threshold"), errors="coerce").fillna(0.70)
            threshold_pass = broad_h.loc[(rank >= base_threshold).fillna(False)].copy()
        else:
            threshold_pass = pd.DataFrame()
        rec: dict[str, Any] = {
            "contract_name": contract_name,
            "head": head,
            "broad_rows": int(len(broad_h)),
            "base_threshold_pass_rows": int(len(threshold_pass)),
            "base_threshold_pass_share": float(len(threshold_pass) / max(len(broad_h), 1)),
            "deployable_rows": int(len(dep_h)),
            "decision_rows": int(len(dec_h)),
            "accepted_rows": int(len(acc_h)),
            "accepted_per_broad": float(len(acc_h) / max(len(broad_h), 1)),
            "accepted_per_deployable": float(len(acc_h) / max(len(dep_h), 1)),
        }
        rec.update(_rank_stats(broad_h, "broad"))
        rec.update(_rank_stats(threshold_pass, "threshold_pass"))
        rec.update(_rank_stats(dep_h, "deployable"))
        if not acc_h.empty:
            returns = pd.to_numeric(acc_h.get("net_return"), errors="coerce")
            rec.update(
                {
                    "accepted_mean_net_return": float(returns.mean()),
                    "accepted_q05_net_return": float(returns.quantile(0.05)),
                    "accepted_full_sl_rate": float(
                        acc_h.get("simple_policy_exit_reason", pd.Series("", index=acc_h.index))
                        .astype(str)
                        .eq("full_sl")
                        .mean()
                    ),
                }
            )
        rows.append(rec)

    reason_rows: list[dict[str, Any]] = []
    if not decisions.empty and {"head", "rejection_reason", "accepted"}.issubset(decisions.columns):
        decisions = decisions.copy()
        decisions["accepted"] = decisions["accepted"].astype(bool)
        reason_counts = (
            decisions.groupby(["head", "rejection_reason", "accepted"], observed=True)
            .size()
            .reset_index(name="rows")
        )
        for _, row in reason_counts.iterrows():
            reason_rows.append(
                {
                    "contract_name": contract_name,
                    "head": str(row["head"]),
                    "rejection_reason": str(row["rejection_reason"]),
                    "accepted": bool(row["accepted"]),
                    "rows": int(row["rows"]),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(reason_rows)


def priority_replay_stats(label: str, priority_dir: Path) -> dict[str, Any]:
    summary = _load_csv(priority_dir / "head_priority_learning_replay_summary.csv")
    by_head = _load_csv(priority_dir / "head_priority_learning_by_head.csv")
    overlap = _load_csv(priority_dir / "head_priority_learning_accepted_overlap.csv")
    if summary.empty:
        return {"label": label, "available": False, "artifact_dir": str(priority_dir)}
    base_arm = "P0_static_priority"
    candidates = summary.loc[~summary["arm"].astype(str).eq(base_arm)].copy()
    candidate_arm = str(candidates.iloc[0]["arm"]) if not candidates.empty else None
    base = summary.loc[summary["arm"].astype(str).eq(base_arm)].iloc[0].to_dict()
    cand = (
        summary.loc[summary["arm"].astype(str).eq(candidate_arm)].iloc[0].to_dict()
        if candidate_arm
        else {}
    )
    overlap_row = (
        overlap.loc[overlap["arm"].astype(str).eq(candidate_arm)].iloc[0].to_dict()
        if candidate_arm and not overlap.empty and not overlap.loc[overlap["arm"].astype(str).eq(candidate_arm)].empty
        else {}
    )
    base_net = float(pd.to_numeric(pd.Series([base.get("net_pnl")]), errors="coerce").iloc[0])
    cand_net = float(pd.to_numeric(pd.Series([cand.get("net_pnl")]), errors="coerce").iloc[0]) if cand else float("nan")
    base_full_sl = float(pd.to_numeric(pd.Series([base.get("full_sl_rate")]), errors="coerce").iloc[0])
    cand_full_sl = float(pd.to_numeric(pd.Series([cand.get("full_sl_rate")]), errors="coerce").iloc[0]) if cand else float("nan")
    return {
        "label": label,
        "available": True,
        "artifact_dir": str(priority_dir),
        "candidate_arm": candidate_arm,
        "static_trades": int(base.get("trade_count", 0)),
        "priority_trades": int(cand.get("trade_count", 0)) if cand else 0,
        "static_net_pnl": base_net,
        "priority_net_pnl": cand_net,
        "delta_net_pnl": cand_net - base_net if np.isfinite(cand_net) else float("nan"),
        "static_full_sl_rate": base_full_sl,
        "priority_full_sl_rate": cand_full_sl,
        "delta_full_sl_rate": cand_full_sl - base_full_sl if np.isfinite(cand_full_sl) else float("nan"),
        "accepted_jaccard": float(pd.to_numeric(pd.Series([overlap_row.get("jaccard_vs_baseline")]), errors="coerce").iloc[0])
        if overlap_row
        else float("nan"),
        "baseline_only": int(overlap_row.get("baseline_only", 0)) if overlap_row else 0,
        "priority_only": int(overlap_row.get("arm_only", 0)) if overlap_row else 0,
        "by_head": by_head.to_dict("records"),
    }


def _comparison_rows(starvation: pd.DataFrame) -> pd.DataFrame:
    if starvation.empty:
        return pd.DataFrame()
    rows = []
    heads = sorted(starvation["head"].dropna().astype(str).unique())
    contracts = sorted(starvation["contract_name"].dropna().astype(str).unique())
    if len(contracts) < 2:
        return pd.DataFrame()
    base_name = "timestamp_rank_t1" if "timestamp_rank_t1" in contracts else contracts[0]
    for challenger in [c for c in contracts if c != base_name]:
        for head in heads:
            base = starvation.loc[
                starvation["contract_name"].eq(base_name) & starvation["head"].eq(head)
            ]
            comp = starvation.loc[
                starvation["contract_name"].eq(challenger) & starvation["head"].eq(head)
            ]
            if base.empty or comp.empty:
                continue
            b = base.iloc[0]
            c = comp.iloc[0]
            rows.append(
                {
                    "base_contract": base_name,
                    "challenger_contract": challenger,
                    "head": head,
                    "delta_base_threshold_pass_rows": int(c["base_threshold_pass_rows"] - b["base_threshold_pass_rows"]),
                    "delta_deployable_rows": int(c["deployable_rows"] - b["deployable_rows"]),
                    "delta_accepted_rows": int(c["accepted_rows"] - b["accepted_rows"]),
                    "delta_deployable_rank_max": float(c["deployable_rank_max"] - b["deployable_rank_max"]),
                    "delta_deployable_rank_mean": float(c["deployable_rank_mean"] - b["deployable_rank_mean"]),
                    "delta_accepted_mean_net_return": float(
                        c.get("accepted_mean_net_return", np.nan)
                        - b.get("accepted_mean_net_return", np.nan)
                    ),
                    "delta_accepted_full_sl_rate": float(
                        c.get("accepted_full_sl_rate", np.nan)
                        - b.get("accepted_full_sl_rate", np.nan)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _render_report(
    *,
    starvation: pd.DataFrame,
    comparison: pd.DataFrame,
    reason_counts: pd.DataFrame,
    priority: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> str:
    lines = [
        "# Market-State Rank/Priority Starvation Audit",
        "",
        "This audit separates rank-threshold eligibility from global auction ordering.",
        "A market-state `portfolio_priority_adjustment` can only reorder candidates that are already viable; it cannot rescue rows that fail the deployed rank threshold.",
        "",
        "## Inputs",
        "",
        f"- Timestamp T1 artifact: `{manifest['inputs']['timestamp_artifact_dir']}`",
        f"- Global-rank challenger artifact: `{manifest['inputs']['global_artifact_dir']}`",
        f"- Timestamp priority report: `{manifest['inputs']['timestamp_priority_dir']}`",
        f"- Global priority report: `{manifest['inputs']['global_priority_dir']}`",
        "",
        "## Candidate Starvation By Head",
        "",
    ]
    view_cols = [
        "contract_name",
        "head",
        "broad_rows",
        "base_threshold_pass_rows",
        "base_threshold_pass_share",
        "deployable_rows",
        "deployable_rank_mean",
        "deployable_rank_max",
        "accepted_rows",
        "accepted_per_deployable",
        "accepted_mean_net_return",
        "accepted_full_sl_rate",
    ]
    lines.append(starvation[[c for c in view_cols if c in starvation.columns]].to_markdown(index=False))
    lines.extend(["", "## Challenger Minus Timestamp T1", ""])
    lines.append(comparison.to_markdown(index=False) if not comparison.empty else "_No comparison rows._")
    lines.extend(["", "## Learned Priority Replays", ""])
    priority_view = pd.DataFrame(
        [
            {
                "label": row.get("label"),
                "candidate_arm": row.get("candidate_arm"),
                "static_trades": row.get("static_trades"),
                "priority_trades": row.get("priority_trades"),
                "delta_net_pnl": row.get("delta_net_pnl"),
                "delta_full_sl_rate": row.get("delta_full_sl_rate"),
                "accepted_jaccard": row.get("accepted_jaccard"),
                "baseline_only": row.get("baseline_only"),
                "priority_only": row.get("priority_only"),
            }
            for row in priority
        ]
    )
    lines.append(priority_view.to_markdown(index=False) if not priority_view.empty else "_No priority rows._")
    lines.extend(["", "## Interpretation", ""])
    short_boll_delta = comparison.loc[comparison["head"].eq("short_boll")] if not comparison.empty else pd.DataFrame()
    if not short_boll_delta.empty:
        row = short_boll_delta.iloc[0]
        lines.append(
            f"- Under the global-rank challenger, short_boll deployable rows change by `{int(row['delta_deployable_rows'])}` and accepted trades change by `{int(row['delta_accepted_rows'])}` versus timestamp T1."
        )
        lines.append(
            f"- Short_boll max deployable rank changes by `{float(row['delta_deployable_rank_max']):.6f}`. If this is strongly negative, the problem is rank-threshold eligibility, not auction ordering."
        )
    global_priority = next((row for row in priority if row.get("label") == "global_rank_priority"), None)
    if global_priority:
        lines.append(
            f"- Global-rank learned priority accepted-set Jaccard is `{float(global_priority.get('accepted_jaccard', np.nan)):.6f}` with delta net PnL `{float(global_priority.get('delta_net_pnl', np.nan)):.6f}`."
        )
    lines.append(
        "- If global-rank priority Jaccard is 1.0, market-state priority modulation cannot fix the current global-rank starvation without also changing threshold eligibility or the rank contract."
    )
    lines.extend(["", "## Rejection Reasons", ""])
    if reason_counts.empty:
        lines.append("_No replay decisions found._")
    else:
        top_reasons = reason_counts.sort_values("rows", ascending=False).groupby(
            ["contract_name", "head"], observed=True
        ).head(8)
        lines.append(top_reasons.to_markdown(index=False))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp-artifact-dir", type=Path, default=DEFAULT_TIMESTAMP_DIR)
    parser.add_argument("--global-artifact-dir", type=Path, default=DEFAULT_GLOBAL_DIR)
    parser.add_argument("--timestamp-priority-dir", type=Path, default=DEFAULT_TIMESTAMP_PRIORITY_DIR)
    parser.add_argument("--global-priority-dir", type=Path, default=DEFAULT_GLOBAL_PRIORITY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    starvation_frames: list[pd.DataFrame] = []
    reason_frames: list[pd.DataFrame] = []
    for name, artifact_dir in [
        ("timestamp_rank_t1", args.timestamp_artifact_dir),
        ("global_rank_challenger", args.global_artifact_dir),
    ]:
        stats, reasons = candidate_starvation_stats(name, artifact_dir)
        starvation_frames.append(stats)
        reason_frames.append(reasons)
    starvation = pd.concat(starvation_frames, ignore_index=True) if starvation_frames else pd.DataFrame()
    reasons = pd.concat(reason_frames, ignore_index=True) if reason_frames else pd.DataFrame()
    comparison = _comparison_rows(starvation)
    priority = [
        priority_replay_stats("timestamp_rank_priority", args.timestamp_priority_dir),
        priority_replay_stats("global_rank_priority", args.global_priority_dir),
    ]

    manifest = {
        "generated_by": "audit_market_state_rank_priority_starvation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "diagnose_whether_market_state_priority_can_rescue_global_rank_short_boll",
        "inputs": {
            "timestamp_artifact_dir": str(args.timestamp_artifact_dir),
            "global_artifact_dir": str(args.global_artifact_dir),
            "timestamp_priority_dir": str(args.timestamp_priority_dir),
            "global_priority_dir": str(args.global_priority_dir),
        },
        "contract": {
            "trains_models": False,
            "replays_trades": False,
            "changes_active_stack": False,
            "active_baseline_remains": "T1_repaired_static_baseline",
        },
        "outputs": {
            "manifest": str(args.output_dir / "rank_priority_starvation_manifest.json"),
            "report": str(args.output_dir / "rank_priority_starvation_report.md"),
            "starvation_by_head": str(args.output_dir / "rank_priority_starvation_by_head.csv"),
            "challenger_delta": str(args.output_dir / "rank_priority_starvation_delta.csv"),
            "rejection_reasons": str(args.output_dir / "rank_priority_rejection_reasons.csv"),
            "priority_replays": str(args.output_dir / "rank_priority_learned_priority_replays.json"),
        },
        "starvation_by_head": starvation.to_dict("records"),
        "challenger_delta": comparison.to_dict("records"),
        "priority_replays": priority,
    }

    starvation.to_csv(args.output_dir / "rank_priority_starvation_by_head.csv", index=False)
    comparison.to_csv(args.output_dir / "rank_priority_starvation_delta.csv", index=False)
    reasons.to_csv(args.output_dir / "rank_priority_rejection_reasons.csv", index=False)
    (args.output_dir / "rank_priority_learned_priority_replays.json").write_text(
        json.dumps(_json_safe(priority), indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "rank_priority_starvation_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2) + "\n",
        encoding="utf-8",
    )
    report = _render_report(
        starvation=starvation,
        comparison=comparison,
        reason_counts=reasons,
        priority=priority,
        manifest=manifest,
    )
    (args.output_dir / "rank_priority_starvation_report.md").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            _json_safe(
                {
                    "output_dir": str(args.output_dir),
                    "short_boll_delta": comparison.loc[
                        comparison["head"].eq("short_boll")
                    ].to_dict("records")
                    if not comparison.empty
                    else [],
                    "priority_replays": priority,
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
