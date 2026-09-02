#!/usr/bin/env python3
"""Policy-conditioned q-fail diagnostics and constrained ablations.

The native q-fail blend improved broad top-tail hit rate but damaged the fixed
portfolio replay.  This script tests q-fail only where it is actually used:
``short_asset`` rows that are anchor-eligible under the frozen rank reference.

It also evaluates constrained alternatives that cannot rebuild the portfolio:

* C1: veto-only, no backfill, applied to the exact A0 accepted set;
* C2: size-only, applied to the exact A0 accepted set;
* C3: period-gated versions of C1/C2.

No model is trained here.  All calculations use already materialized June
15-22 candidate, component-score, and replay-decision artifacts.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_COMPONENT_SCORES = Path(
    "data_perp/reports/native_reliability_blend_scores_20260625_jun15_22_fullfit"
    "/native_reliability_blend_scores.parquet"
)
DEFAULT_TRANSITION_DIR = Path(
    "data_perp/reports/reliability_blend_policy_transition_diagnostics_20260625_jun15_22"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/qfail_policy_conditioned_ablation_20260625_jun15_22"
)
ARM_CANDIDATES = {
    "A0": Path(
        "data_perp/artifacts/reliability_blend_arm_A0_anchor_only_20260625_jun15_22"
        "/simple_policy_optimiser/simple_policy_candidates_broad.parquet"
    ),
    "A1": Path(
        "data_perp/artifacts/reliability_blend_arm_A1_anchor_qfail_20260625_jun15_22"
        "/simple_policy_optimiser/simple_policy_candidates_broad.parquet"
    ),
    "B0": Path(
        "data_perp/artifacts/reliability_blend_arm_B0_full_native_blend_20260625_jun15_22"
        "/simple_policy_optimiser/simple_policy_candidates_broad.parquet"
    ),
}
KEY_COLS = ["timestamp", "symbol", "side", "strategy_id"]
COMPONENT_COLS = [
    "anchor_score",
    "anchor_component_rank",
    "period_component_score",
    "period_component_rank",
    "qfail_component_score",
    "qfail_component_rank",
    "reliability_anchor_only_score",
    "reliability_anchor_qfail_score",
    "reliability_anchor_period_score",
    "reliability_blend_score",
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _infer_head(strategy_id: Any) -> str | None:
    if not isinstance(strategy_id, str):
        return None
    for head in ("long_bars", "long_dist", "short_asset", "short_boll"):
        if strategy_id.startswith(head):
            return head
    return None


def _canonicalise(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if "head" not in out.columns:
        out["head"] = out["strategy_id"].map(_infer_head)
    for col in ("symbol", "side", "strategy_id", "head"):
        if col in out.columns:
            out[col] = out[col].astype(str)
    return out


def _load_components(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    keep = ["timestamp", "symbol", "strategy_id"] + [c for c in COMPONENT_COLS if c in raw.columns]
    raw = _canonicalise(raw[keep])
    raw = raw[["timestamp", "symbol", "strategy_id"] + [c for c in COMPONENT_COLS if c in raw.columns]]
    return raw.rename(columns={c: f"component_{c}" for c in COMPONENT_COLS if c in raw.columns})


def _load_arm_candidates(path: Path, components: pd.DataFrame) -> pd.DataFrame:
    raw = _canonicalise(pd.read_parquet(path))
    raw = raw.merge(
        components,
        on=["timestamp", "symbol", "strategy_id"],
        how="left",
        validate="many_to_one",
    )
    if "deployment_rank_threshold" not in raw.columns:
        raw["deployment_rank_threshold"] = raw.get("base_strategy_threshold", 0.70)
    return raw


def _decision_accept_flags(transition_dir: Path, arm_file_stem: str, prefix: str) -> pd.DataFrame:
    decisions = pd.read_parquet(transition_dir / f"{arm_file_stem}_decisions.parquet")
    accepted = decisions[["timestamp", "symbol", "side", "strategy_id", "accepted", "rejection_reason"]].copy()
    accepted = _canonicalise(accepted)
    accepted = accepted.rename(
        columns={
            "accepted": f"{prefix}_accepted",
            "rejection_reason": f"{prefix}_rejection_reason",
        }
    )
    return accepted[KEY_COLS + [f"{prefix}_accepted", f"{prefix}_rejection_reason"]]


def _accepted_trades(transition_dir: Path) -> pd.DataFrame:
    accepted = pd.read_parquet(transition_dir / "A0_anchor_only_accepted_trades.parquet")
    return _canonicalise(accepted)


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _rate(mask: pd.Series) -> float:
    if len(mask) == 0:
        return np.nan
    return float(mask.fillna(False).mean())


def _cell_metrics(group: pd.DataFrame) -> dict[str, Any]:
    net = _numeric(group, "net_return")
    gross = _numeric(group, "gross_return")
    exit_reason = group.get("simple_policy_exit_reason", pd.Series(index=group.index, dtype=object)).astype(str).str.lower()
    return {
        "rows": int(len(group)),
        "mean_net_return": float(net.mean()) if net.notna().any() else np.nan,
        "median_net_return": float(net.median()) if net.notna().any() else np.nan,
        "q05_net_return": float(net.quantile(0.05)) if net.notna().any() else np.nan,
        "mean_gross_return": float(gross.mean()) if gross.notna().any() else np.nan,
        "full_sl_rate": _rate(exit_reason.isin(["sl", "full_sl"])),
        "timeout_rate": _rate(exit_reason.eq("timeout")),
        "mean_qfail_score": float(_numeric(group, "component_qfail_component_score").mean()),
        "mean_qfail_rank": float(_numeric(group, "component_qfail_component_rank").mean()),
        "mean_period_score": float(_numeric(group, "component_period_component_score").mean()),
        "mean_period_rank": float(_numeric(group, "component_period_component_rank").mean()),
        "mean_anchor_rank": float(_numeric(group, "policy_rank_pct").mean()),
        "a0_accept_rate": _rate(group.get("a0_accepted", pd.Series(False, index=group.index)).astype(bool)),
        "a1_accept_rate": _rate(group.get("a1_accepted", pd.Series(False, index=group.index)).astype(bool)),
        "b0_accept_rate": _rate(group.get("b0_accepted", pd.Series(False, index=group.index)).astype(bool)),
    }


def _add_bins(eligible: pd.DataFrame) -> pd.DataFrame:
    out = eligible.copy()
    out["anchor_rank_band"] = pd.cut(
        _numeric(out, "policy_rank_pct"),
        bins=[0.70, 0.80, 0.90, 0.95, np.inf],
        labels=["0.70_0.80", "0.80_0.90", "0.90_0.95", "0.95_1.00"],
        include_lowest=True,
        right=False,
    )
    qfail_rank = _numeric(out, "component_qfail_component_rank")
    out["qfail_decile_high"] = pd.qcut(
        qfail_rank.rank(method="first"),
        q=10,
        labels=[f"D{i}" for i in range(1, 11)],
    )
    out["qfail_tail_high"] = pd.qcut(
        qfail_rank.rank(method="first"),
        q=10,
        labels=[f"D{i}" for i in range(1, 11)],
    )
    return out


def _qfail_conditioning_tables(eligible: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    binned = _add_bins(eligible)
    rows: list[dict[str, Any]] = []
    for (band, decile), group in binned.groupby(["anchor_rank_band", "qfail_decile_high"], observed=True):
        rec = {
            "anchor_rank_band": str(band),
            "qfail_decile_high": str(decile),
        }
        rec.update(_cell_metrics(group))
        rows.append(rec)
    cell_table = pd.DataFrame(rows)

    decile_rows = []
    for decile, group in binned.groupby("qfail_decile_high", observed=True):
        rec = {"qfail_decile_high": str(decile)}
        rec.update(_cell_metrics(group))
        decile_rows.append(rec)
    decile_table = pd.DataFrame(decile_rows)

    mono_rows = []
    for band, group in binned.groupby("anchor_rank_band", observed=True):
        qfail = _numeric(group, "component_qfail_component_rank")
        net = _numeric(group, "net_return")
        full_sl = group["simple_policy_exit_reason"].astype(str).str.lower().isin(["sl", "full_sl"]).astype(float)
        mono_rows.append(
            {
                "anchor_rank_band": str(band),
                "rows": int(len(group)),
                "spearman_qfail_rank_vs_net_return": float(qfail.corr(net, method="spearman")) if len(group) > 2 else np.nan,
                "spearman_qfail_rank_vs_full_sl": float(qfail.corr(full_sl, method="spearman")) if len(group) > 2 else np.nan,
                "low_decile_mean_net_return": float(
                    net.loc[qfail <= qfail.quantile(0.10)].mean()
                ),
                "high_decile_mean_net_return": float(
                    net.loc[qfail >= qfail.quantile(0.90)].mean()
                ),
                "low_decile_full_sl_rate": _rate(full_sl.loc[qfail <= qfail.quantile(0.10)] > 0.5),
                "high_decile_full_sl_rate": _rate(full_sl.loc[qfail >= qfail.quantile(0.90)] > 0.5),
            }
        )
    monotonicity = pd.DataFrame(mono_rows)
    return cell_table, decile_table, monotonicity


def _tail_risk(row: pd.DataFrame, direction: str) -> pd.Series:
    rank = _numeric(row, "component_qfail_component_rank").clip(0.0, 1.0)
    if direction == "high_qfail":
        return rank
    if direction == "low_qfail":
        return 1.0 - rank
    raise ValueError(f"unknown direction: {direction}")


def _period_gate(row: pd.DataFrame, gate: str, eligible: pd.DataFrame) -> pd.Series:
    period = _numeric(row, "component_period_component_rank")
    ref = _numeric(eligible, "component_period_component_rank")
    if gate == "none":
        return pd.Series(True, index=row.index)
    if gate == "high_period_half":
        return period >= ref.quantile(0.50)
    if gate == "low_period_half":
        return period <= ref.quantile(0.50)
    if gate == "high_period_quartile":
        return period >= ref.quantile(0.75)
    if gate == "low_period_quartile":
        return period <= ref.quantile(0.25)
    raise ValueError(f"unknown period gate: {gate}")


def _accepted_metrics(rows: pd.DataFrame, *, label: str, variant: str) -> dict[str, Any]:
    net_pnl = _numeric(rows, "net_pnl")
    gross_pnl = _numeric(rows, "gross_pnl")
    net_return = _numeric(rows, "net_return")
    exit_reason = rows.get("simple_policy_exit_reason", pd.Series(index=rows.index, dtype=object)).astype(str).str.lower()
    return {
        "family": label,
        "variant": variant,
        "trade_count": int(len(rows)),
        "net_pnl": float(net_pnl.sum()),
        "gross_pnl": float(gross_pnl.sum()),
        "mean_net_return": float(net_return.mean()) if net_return.notna().any() else np.nan,
        "win_rate": _rate(net_pnl > 0.0),
        "full_sl_rate": _rate(exit_reason.isin(["sl", "full_sl"])),
        "timeout_rate": _rate(exit_reason.eq("timeout")),
    }


def _defensive_success(original: pd.DataFrame, multiplier: pd.Series) -> dict[str, float]:
    net = _numeric(original, "net_pnl").fillna(0.0)
    mult = pd.to_numeric(multiplier, errors="coerce").fillna(1.0).clip(0.0, 1.0)
    reduction = 1.0 - mult
    avoided_loss = float((-net.clip(upper=0.0) * reduction).sum())
    winner_sacrificed = float((net.clip(lower=0.0) * reduction).sum())
    return {
        "loss_avoided": avoided_loss,
        "winner_pnl_sacrificed": winner_sacrificed,
        "defensive_success": avoided_loss - winner_sacrificed,
    }


def _veto_ablation(a0_accepted: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline = _accepted_metrics(a0_accepted, label="baseline", variant="A0_exact")
    baseline.update({"veto_share": 0.0, "risk_direction": "none", "period_gate": "none"})
    baseline.update(_defensive_success(a0_accepted, pd.Series(1.0, index=a0_accepted.index)))
    rows.append(baseline)

    for direction in ("high_qfail", "low_qfail"):
        ref_risk = _tail_risk(eligible, direction)
        for share in (0.05, 0.10, 0.15):
            threshold = float(ref_risk.quantile(1.0 - share))
            for gate in ("none", "high_period_half", "low_period_half", "high_period_quartile", "low_period_quartile"):
                risk = _tail_risk(a0_accepted, direction)
                gate_mask = _period_gate(a0_accepted, gate, eligible)
                veto = (risk >= threshold) & gate_mask
                kept = a0_accepted.loc[~veto].copy()
                mult = pd.Series(np.where(veto, 0.0, 1.0), index=a0_accepted.index)
                rec = _accepted_metrics(kept, label="C1_veto_no_backfill", variant=f"{direction}_{gate}_top{int(share*100)}")
                rec.update(
                    {
                        "risk_direction": direction,
                        "period_gate": gate,
                        "veto_share": share,
                        "threshold": threshold,
                        "vetoed_trades": int(veto.sum()),
                        "net_pnl_delta_vs_A0": rec["net_pnl"] - baseline["net_pnl"],
                    }
                )
                rec.update(_defensive_success(a0_accepted, mult))
                rows.append(rec)
    return pd.DataFrame(rows)


def _size_ablation(a0_accepted: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline_pnl = float(_numeric(a0_accepted, "net_pnl").sum())
    for direction in ("high_qfail", "low_qfail"):
        for lam in (0.25, 0.50, 0.75, 1.00):
            for gate in ("none", "high_period_half", "low_period_half", "high_period_quartile", "low_period_quartile"):
                risk = _tail_risk(a0_accepted, direction)
                gate_mask = _period_gate(a0_accepted, gate, eligible)
                effective_risk = risk.where(gate_mask, 0.0)
                multiplier = (1.0 - lam * effective_risk).clip(0.0, 1.0)
                rows_scaled = a0_accepted.copy()
                rows_scaled["net_pnl"] = _numeric(rows_scaled, "net_pnl") * multiplier
                rows_scaled["gross_pnl"] = _numeric(rows_scaled, "gross_pnl") * multiplier
                rows_scaled["position_size"] = _numeric(rows_scaled, "position_size") * multiplier
                rec = _accepted_metrics(rows_scaled, label="C2_size_only", variant=f"{direction}_{gate}_lambda{lam:.2f}")
                rec.update(
                    {
                        "risk_direction": direction,
                        "period_gate": gate,
                        "lambda": lam,
                        "mean_size_multiplier": float(multiplier.mean()),
                        "net_pnl_delta_vs_A0": rec["net_pnl"] - baseline_pnl,
                    }
                )
                rec.update(_defensive_success(a0_accepted, multiplier))
                rows.append(rec)
    return pd.DataFrame(rows)


def _load_a0_state_pool(
    transition_dir: Path,
    a0_candidates: pd.DataFrame,
) -> pd.DataFrame:
    decisions = pd.read_parquet(transition_dir / "A0_anchor_only_decisions.parquet")
    decisions = decisions[
        [
            "candidate_index",
            "timestamp",
            "symbol",
            "side",
            "strategy_id",
            "accepted",
            "rejection_reason",
            "position_size",
            "normalized_rank_score",
            "portfolio_priority",
        ]
    ].copy()
    decisions["candidate_index"] = pd.to_numeric(decisions["candidate_index"], errors="coerce").astype("Int64")
    decisions = decisions[decisions["candidate_index"].notna()].copy()
    candidates = a0_candidates.reset_index(drop=True).copy()
    candidates["candidate_index"] = np.arange(len(candidates), dtype=np.int64)
    merged = candidates.merge(
        decisions.drop(columns=["timestamp", "symbol", "side", "strategy_id"]),
        on="candidate_index",
        how="left",
        validate="one_to_one",
        suffixes=("", "_a0_decision"),
    )
    return _canonicalise(merged)


def _arm_score_lookup(candidates_by_arm: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = candidates_by_arm["A0"][KEY_COLS].copy()
    for arm, frame in candidates_by_arm.items():
        score_cols = KEY_COLS + ["policy_rank_pct", "reliability_blend_score"]
        tmp = frame[score_cols].copy().rename(
            columns={
                "policy_rank_pct": f"{arm}_policy_rank_pct",
                "reliability_blend_score": f"{arm}_score",
            }
        )
        base = base.merge(tmp, on=KEY_COLS, how="left", validate="one_to_one")
    return base


def _fixed_count_counterfactual(
    *,
    transition_dir: Path,
    candidates_by_arm: dict[str, pd.DataFrame],
    a0_accepted: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select each timestamp's A0 entry count using each arm's ordering.

    This is a deliberately local counterfactual.  It keeps A0's timestamp-level
    entry count and uses A0's available candidate pool after state constraints
    that are visible in the A0 decision table.  It does not let earlier
    challenger choices change future cooldowns, wallet, or open positions.
    """
    pool = _load_a0_state_pool(transition_dir, candidates_by_arm["A0"])
    scores = _arm_score_lookup(candidates_by_arm)
    pool = pool.merge(scores, on=KEY_COLS, how="left", validate="one_to_one")
    allowed_reasons = {
        "accepted",
        "max_new_entries_per_strategy_per_bar_reached",
        "max_concurrent_per_strategy_reached",
    }
    pool = pool[
        pool["head"].eq("short_asset")
        & pool["rejection_reason"].astype(str).isin(allowed_reasons)
    ].copy()
    a0_accepted = a0_accepted.copy()
    a0_accepted["timestamp"] = pd.to_datetime(a0_accepted["timestamp"], utc=True, errors="coerce")
    size_schedule = {}
    for ts, group in a0_accepted.groupby("timestamp", sort=True):
        ordered = group.sort_values("normalized_rank_score", ascending=False)
        size_schedule[ts] = pd.to_numeric(ordered["position_size"], errors="coerce").fillna(0.0).to_numpy()

    selected_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for arm in ("A0", "A1", "B0"):
        if arm == "A0":
            arm_df = a0_accepted.copy()
            arm_df["counterfactual_arm"] = arm
            arm_df["counterfactual_position_size"] = _numeric(arm_df, "position_size").fillna(0.0)
            arm_df["counterfactual_net_pnl"] = _numeric(arm_df, "net_pnl").fillna(0.0)
            arm_df["counterfactual_gross_pnl"] = _numeric(arm_df, "gross_pnl").fillna(0.0)
            selected_frames.append(arm_df)
            exit_reason = arm_df.get("simple_policy_exit_reason", pd.Series(index=arm_df.index, dtype=object)).astype(str).str.lower()
            summary_rows.append(
                {
                    "arm": arm,
                    "selected_rows": int(len(arm_df)),
                    "timestamp_count": int(arm_df["timestamp"].nunique()) if not arm_df.empty else 0,
                    "net_pnl": float(_numeric(arm_df, "counterfactual_net_pnl").sum()) if not arm_df.empty else 0.0,
                    "gross_pnl": float(_numeric(arm_df, "counterfactual_gross_pnl").sum()) if not arm_df.empty else 0.0,
                    "mean_net_return": float(_numeric(arm_df, "net_return").mean()) if not arm_df.empty else np.nan,
                    "win_rate": _rate(_numeric(arm_df, "counterfactual_net_pnl") > 0.0) if not arm_df.empty else np.nan,
                    "full_sl_rate": _rate(exit_reason.isin(["sl", "full_sl"])) if not arm_df.empty else np.nan,
                    "timeout_rate": _rate(exit_reason.eq("timeout")) if not arm_df.empty else np.nan,
                }
            )
            continue
        score_col = f"{arm}_policy_rank_pct"
        arm_rows: list[pd.DataFrame] = []
        for ts, sizes in size_schedule.items():
            if len(sizes) == 0:
                continue
            group = pool[pool["timestamp"].eq(ts)].copy()
            if group.empty:
                continue
            group["_arm_order_score"] = _numeric(group, score_col)
            group = group[np.isfinite(group["_arm_order_score"])]
            if group.empty:
                continue
            chosen = group.sort_values("_arm_order_score", ascending=False).head(len(sizes)).copy()
            assigned_sizes = np.resize(sizes, len(chosen)) if len(chosen) > len(sizes) else sizes[: len(chosen)]
            chosen["counterfactual_arm"] = arm
            chosen["counterfactual_position_size"] = assigned_sizes
            chosen["counterfactual_net_pnl"] = _numeric(chosen, "net_return").fillna(0.0) * chosen[
                "counterfactual_position_size"
            ]
            chosen["counterfactual_gross_pnl"] = _numeric(chosen, "gross_return").fillna(0.0) * chosen[
                "counterfactual_position_size"
            ]
            arm_rows.append(chosen)
        arm_df = pd.concat(arm_rows, ignore_index=True) if arm_rows else pd.DataFrame()
        if not arm_df.empty:
            selected_frames.append(arm_df)
        exit_reason = arm_df.get("simple_policy_exit_reason", pd.Series(index=arm_df.index, dtype=object)).astype(str).str.lower()
        summary_rows.append(
            {
                "arm": arm,
                "selected_rows": int(len(arm_df)),
                "timestamp_count": int(arm_df["timestamp"].nunique()) if not arm_df.empty else 0,
                "net_pnl": float(_numeric(arm_df, "counterfactual_net_pnl").sum()) if not arm_df.empty else 0.0,
                "gross_pnl": float(_numeric(arm_df, "counterfactual_gross_pnl").sum()) if not arm_df.empty else 0.0,
                "mean_net_return": float(_numeric(arm_df, "net_return").mean()) if not arm_df.empty else np.nan,
                "win_rate": _rate(_numeric(arm_df, "counterfactual_net_pnl") > 0.0) if not arm_df.empty else np.nan,
                "full_sl_rate": _rate(exit_reason.isin(["sl", "full_sl"])) if not arm_df.empty else np.nan,
                "timeout_rate": _rate(exit_reason.eq("timeout")) if not arm_df.empty else np.nan,
            }
        )
    selected = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    a0_net = float(summary.loc[summary["arm"].eq("A0"), "net_pnl"].iloc[0]) if "A0" in set(summary["arm"]) else 0.0
    summary["net_pnl_delta_vs_A0_counterfactual"] = summary["net_pnl"] - a0_net

    # Accepted-set overlap under the same timestamp counts.
    if not selected.empty:
        a0_keys = set(map(tuple, selected[selected["counterfactual_arm"].eq("A0")][KEY_COLS].to_numpy()))
        overlap_rows = []
        for arm in ("A1", "B0"):
            arm_keys = set(map(tuple, selected[selected["counterfactual_arm"].eq(arm)][KEY_COLS].to_numpy()))
            inter = len(a0_keys & arm_keys)
            union = len(a0_keys | arm_keys)
            overlap_rows.append(
                {
                    "arm": arm,
                    "top_count_jaccard_vs_A0": inter / union if union else np.nan,
                    "common": inter,
                    "added": len(arm_keys - a0_keys),
                    "removed": len(a0_keys - arm_keys),
                }
            )
        overlap = pd.DataFrame(overlap_rows)
        summary = summary.merge(overlap, on="arm", how="left")
    return summary, selected


def _transition_incremental_table(
    transition_details: pd.DataFrame,
    eligible: pd.DataFrame,
) -> pd.DataFrame:
    if transition_details.empty:
        return pd.DataFrame()
    work = _canonicalise(transition_details)
    comp = eligible[["timestamp", "symbol", "side", "strategy_id", "component_qfail_component_rank", "policy_rank_pct"]].copy()
    comp = comp.rename(
        columns={
            "component_qfail_component_rank": "qfail_rank",
            "policy_rank_pct": "anchor_rank",
        }
    )
    work = work.merge(comp, on=KEY_COLS, how="left", validate="many_to_one")
    work["anchor_rank_band"] = pd.cut(
        _numeric(work, "anchor_rank"),
        bins=[0.70, 0.80, 0.90, 0.95, np.inf],
        labels=["0.70_0.80", "0.80_0.90", "0.90_0.95", "0.95_1.00"],
        include_lowest=True,
        right=False,
    )
    work["qfail_decile_high"] = pd.qcut(
        _numeric(work, "qfail_rank").rank(method="first"),
        q=10,
        labels=[f"D{i}" for i in range(1, 11)],
    )
    rows = []
    for keys, group in work.groupby(["comparison", "cohort", "anchor_rank_band", "qfail_decile_high"], observed=True):
        comparison, cohort, band, decile = keys
        rows.append(
            {
                "comparison": comparison,
                "cohort": cohort,
                "anchor_rank_band": str(band),
                "qfail_decile_high": str(decile),
                "rows": int(len(group)),
                "base_net_pnl": float(_numeric(group, "base_net_pnl").fillna(0.0).sum()),
                "challenger_net_pnl": float(_numeric(group, "challenger_net_pnl").fillna(0.0).sum()),
                "delta_net_pnl": float(_numeric(group, "delta_net_pnl").fillna(0.0).sum()),
            }
        )
    return pd.DataFrame(rows)


def _render_report(
    *,
    deciles: pd.DataFrame,
    monotonicity: pd.DataFrame,
    veto: pd.DataFrame,
    size: pd.DataFrame,
    counterfactual: pd.DataFrame,
    manifest: dict[str, Any],
) -> str:
    lines = ["# Q-Fail Policy-Conditioned Diagnostics", ""]
    lines.append(f"Generated: {manifest['generated_at_utc']}")
    lines.append("")
    lines.append("## Q-Fail Deciles On Anchor-Eligible Short Asset")
    lines.extend(
        _markdown_table(
            deciles,
            [
                "qfail_decile_high",
                "rows",
                "mean_net_return",
                "full_sl_rate",
                "timeout_rate",
                "mean_qfail_rank",
                "mean_period_rank",
                "a0_accept_rate",
                "a1_accept_rate",
                "b0_accept_rate",
            ],
        )
    )
    lines.append("")
    lines.append("## Within Anchor-Rank Band Monotonicity")
    lines.extend(
        _markdown_table(
            monotonicity,
            [
                "anchor_rank_band",
                "rows",
                "spearman_qfail_rank_vs_net_return",
                "spearman_qfail_rank_vs_full_sl",
                "low_decile_mean_net_return",
                "high_decile_mean_net_return",
                "low_decile_full_sl_rate",
                "high_decile_full_sl_rate",
            ],
        )
    )
    lines.append("")
    lines.append("## Best Veto-Only No-Backfill Arms")
    top_veto = veto.sort_values("net_pnl", ascending=False).head(12)
    lines.extend(
        _markdown_table(
            top_veto,
            [
                "family",
                "variant",
                "trade_count",
                "net_pnl",
                "net_pnl_delta_vs_A0",
                "full_sl_rate",
                "loss_avoided",
                "winner_pnl_sacrificed",
                "defensive_success",
            ],
        )
    )
    lines.append("")
    lines.append("## Best Size-Only Arms")
    top_size = size.sort_values("net_pnl", ascending=False).head(12)
    lines.extend(
        _markdown_table(
            top_size,
            [
                "family",
                "variant",
                "trade_count",
                "net_pnl",
                "net_pnl_delta_vs_A0",
                "mean_size_multiplier",
                "full_sl_rate",
                "loss_avoided",
                "winner_pnl_sacrificed",
                "defensive_success",
            ],
        )
    )
    lines.append("")
    lines.append("## Fixed-Count One-Bar Counterfactual")
    lines.extend(
        _markdown_table(
            counterfactual,
            [
                "arm",
                "selected_rows",
                "net_pnl",
                "net_pnl_delta_vs_A0_counterfactual",
                "full_sl_rate",
                "timeout_rate",
                "top_count_jaccard_vs_A0",
                "common",
                "added",
                "removed",
            ],
        )
    )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- These are cost-inclusive June 15-22 fixed-policy replay diagnostics.")
    lines.append("- C1 and C2 keep the A0 accepted set as the starting point, so any improvement must come from removing or shrinking A0 trades, not from adding replacements.")
    lines.append("- Both q-fail risk orientations are reported because economic orientation must be proven inside the anchor-eligible short_asset frontier.")
    lines.append("- The fixed-count counterfactual keeps A0's entry count per timestamp and uses A0's state-filtered candidate pool; it estimates immediate swap value before path-dependent cooldown, wallet and capacity cascades.")
    return "\n".join(lines) + "\n"


def _markdown_table(df: pd.DataFrame, cols: list[str]) -> list[str]:
    present = [c for c in cols if c in df.columns]
    if not present:
        return ["No columns available."]
    lines = ["|" + "|".join(present) + "|", "|" + "|".join(["---"] * len(present)) + "|"]
    for _, row in df[present].iterrows():
        vals = []
        for col in present:
            val = row[col]
            if pd.isna(val):
                vals.append("")
            elif isinstance(val, float):
                vals.append(f"{val:.6f}")
            else:
                vals.append(str(val))
        lines.append("|" + "|".join(vals) + "|")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-scores", type=Path, default=DEFAULT_COMPONENT_SCORES)
    parser.add_argument("--transition-dir", type=Path, default=DEFAULT_TRANSITION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    components = _load_components(args.component_scores)
    candidates = {arm: _load_arm_candidates(path, components) for arm, path in ARM_CANDIDATES.items()}
    for prefix, stem in (
        ("a0", "A0_anchor_only"),
        ("a1", "A1_anchor_qfail"),
        ("b0", "B0_full_native_blend"),
    ):
        flags = _decision_accept_flags(args.transition_dir, stem, prefix)
        candidates["A0"] = candidates["A0"].merge(flags, on=KEY_COLS, how="left", validate="one_to_one")
    for col in ("a0_accepted", "a1_accepted", "b0_accepted"):
        candidates["A0"][col] = candidates["A0"][col].fillna(False).astype(bool)

    a0 = candidates["A0"]
    eligible = a0[
        (a0["head"].eq("short_asset"))
        & (_numeric(a0, "policy_rank_pct") >= _numeric(a0, "deployment_rank_threshold"))
    ].copy()
    cell_table, decile_table, monotonicity = _qfail_conditioning_tables(eligible)
    a0_accepted = _accepted_trades(args.transition_dir)
    # Accepted trades already contain component score columns from the transition diagnostic.
    transition_details = pd.read_csv(args.transition_dir / "accepted_trade_transition_details.csv")
    transition_incremental = _transition_incremental_table(transition_details, eligible)
    veto = _veto_ablation(a0_accepted, eligible)
    size = _size_ablation(a0_accepted, eligible)
    counterfactual, counterfactual_selected = _fixed_count_counterfactual(
        transition_dir=args.transition_dir,
        candidates_by_arm=candidates,
        a0_accepted=a0_accepted,
    )

    cell_table.to_csv(args.output_dir / "qfail_anchor_band_decile_cells.csv", index=False)
    decile_table.to_csv(args.output_dir / "qfail_decile_summary.csv", index=False)
    monotonicity.to_csv(args.output_dir / "qfail_within_anchor_band_monotonicity.csv", index=False)
    transition_incremental.to_csv(args.output_dir / "transition_incremental_by_qfail_cell.csv", index=False)
    veto.to_csv(args.output_dir / "c1_veto_no_backfill_ablation.csv", index=False)
    size.to_csv(args.output_dir / "c2_size_only_ablation.csv", index=False)
    counterfactual.to_csv(args.output_dir / "one_bar_fixed_count_counterfactual_summary.csv", index=False)
    counterfactual_selected.to_csv(args.output_dir / "one_bar_fixed_count_counterfactual_selected.csv", index=False)

    manifest = {
        "generated_by": "analyze_qfail_policy_conditioned_ablation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "component_scores": str(args.component_scores),
        "transition_dir": str(args.transition_dir),
        "candidate_paths": {arm: str(path) for arm, path in ARM_CANDIDATES.items()},
        "population": "short_asset rows where A0 frozen policy_rank_pct >= deployment_rank_threshold",
        "eligible_rows": int(len(eligible)),
        "a0_accepted_rows": int(len(a0_accepted)),
        "outputs": {
            "cell_table": str(args.output_dir / "qfail_anchor_band_decile_cells.csv"),
            "decile_summary": str(args.output_dir / "qfail_decile_summary.csv"),
            "monotonicity": str(args.output_dir / "qfail_within_anchor_band_monotonicity.csv"),
            "transition_incremental": str(args.output_dir / "transition_incremental_by_qfail_cell.csv"),
            "veto_ablation": str(args.output_dir / "c1_veto_no_backfill_ablation.csv"),
            "size_ablation": str(args.output_dir / "c2_size_only_ablation.csv"),
            "one_bar_counterfactual": str(args.output_dir / "one_bar_fixed_count_counterfactual_summary.csv"),
            "one_bar_counterfactual_selected": str(args.output_dir / "one_bar_fixed_count_counterfactual_selected.csv"),
            "report": str(args.output_dir / "qfail_policy_conditioned_ablation_report.md"),
        },
    }
    (args.output_dir / "qfail_policy_conditioned_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2) + "\n",
        encoding="utf-8",
    )
    report = _render_report(
        deciles=decile_table,
        monotonicity=monotonicity,
        veto=veto,
        size=size,
        counterfactual=counterfactual,
        manifest=manifest,
    )
    (args.output_dir / "qfail_policy_conditioned_ablation_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(_json_safe(manifest), indent=2))


if __name__ == "__main__":
    main()
