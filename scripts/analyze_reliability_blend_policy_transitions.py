#!/usr/bin/env python3
"""Diagnose reliability-blend score changes at the portfolio decision boundary.

This script replays the same candidate universe through the fixed
``refit_bar4_strategy_bar2`` portfolio policy and decomposes how challenger arms
change accepted trades versus the anchor.  It is intentionally diagnostic: no
models are trained and no thresholds are selected here.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extreme_price_movements.portfolio_policy_replay import (
    fit_hierarchical_ev_curves,
    normalise_candidate_table,
    portfolio_policy_params_from_live_config,
    replay_candidates,
)


DEFAULT_POLICY_MANIFEST = Path(
    "data_perp/reports/reliability_blend_portfolio_policy_ablation_20260624"
    "/portfolio_policy_ablation_manifest.json"
)
DEFAULT_TRAIN_CANDIDATES = Path(
    "data_perp/artifacts/reliability_blend_native_simple_policy_replay_20260624_floor070"
    "/simple_policy_optimiser/simple_policy_candidates.parquet"
)
DEFAULT_COMPONENT_SCORES = Path(
    "data_perp/reports/native_reliability_blend_scores_20260625_jun15_22_fullfit"
    "/native_reliability_blend_scores.parquet"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/reliability_blend_policy_transition_diagnostics_20260625_jun15_22"
)

ARM_CANDIDATE_PATHS = {
    "A0_anchor_only": Path(
        "data_perp/artifacts/reliability_blend_arm_A0_anchor_only_20260625_jun15_22"
        "/simple_policy_optimiser/simple_policy_candidates_broad.parquet"
    ),
    "A1_anchor_qfail": Path(
        "data_perp/artifacts/reliability_blend_arm_A1_anchor_qfail_20260625_jun15_22"
        "/simple_policy_optimiser/simple_policy_candidates_broad.parquet"
    ),
    "B0_full_native_blend": Path(
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


def _canonicalise_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "head" not in out.columns:
        out["head"] = out["strategy_id"].map(_infer_head)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    for col in ("symbol", "side", "strategy_id", "head"):
        if col in out.columns:
            out[col] = out[col].astype(str)
    return out


def _load_components_fast(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "symbol", "strategy_id"])
    meta = pd.read_parquet(path)
    keep = ["timestamp", "symbol", "strategy_id"] + [c for c in COMPONENT_COLS if c in meta.columns]
    meta = _canonicalise_keys(meta[keep])
    return meta.rename(columns={c: f"component_{c}" for c in COMPONENT_COLS if c in meta.columns})


def _enrich_candidates(path: Path, components: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing candidate table: {path}")
    df = _canonicalise_keys(pd.read_parquet(path))
    if not components.empty:
        df = df.merge(
            components,
            on=["timestamp", "symbol", "strategy_id"],
            how="left",
            validate="many_to_one",
        )
    if "deployment_rank_threshold" not in df.columns:
        df["deployment_rank_threshold"] = df.get("base_strategy_threshold", np.nan)
    out = normalise_candidate_table(df)
    if "head" not in out.columns:
        out["head"] = out["strategy_id"].map(_infer_head)
    return _canonicalise_keys(out)


def _load_policy_params(manifest_path: Path, variant: str):
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    variant_params = payload.get("variant_params", {}).get(variant)
    if not isinstance(variant_params, dict):
        raise KeyError(f"Missing variant_params[{variant!r}] in {manifest_path}")
    return portfolio_policy_params_from_live_config(variant_params), payload


def _accepted_trades(candidates: pd.DataFrame, decisions: pd.DataFrame, arm: str) -> pd.DataFrame:
    accepted = decisions.loc[decisions["accepted"].astype(bool)].copy()
    if accepted.empty:
        return pd.DataFrame(columns=["arm", *KEY_COLS])
    idx = pd.to_numeric(accepted["candidate_index"], errors="coerce").astype("Int64")
    accepted = accepted.loc[idx.notna()].copy()
    idx = idx.loc[idx.notna()].astype(int)
    cand = candidates.reset_index(drop=True).iloc[idx.to_numpy()].reset_index(drop=True)
    accepted = accepted.reset_index(drop=True)
    out = accepted.copy()
    out["arm"] = arm
    copy_cols = [
        "head",
        "net_return",
        "gross_return",
        "simple_policy_exit_reason",
        "reliability_blend_score",
        "calibrated_score",
        "policy_rank_pct",
        "rank_pct",
        "deployment_rank_threshold",
        "base_strategy_threshold",
        "score_source",
        "component_anchor_score",
        "component_anchor_component_rank",
        "component_period_component_score",
        "component_period_component_rank",
        "component_qfail_component_score",
        "component_qfail_component_rank",
        "component_reliability_anchor_only_score",
        "component_reliability_anchor_qfail_score",
        "component_reliability_anchor_period_score",
        "component_reliability_blend_score",
    ]
    for col in copy_cols:
        if col in cand.columns:
            out[col] = cand[col].to_numpy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    for col in ("symbol", "side", "strategy_id", "head"):
        if col in out.columns:
            out[col] = out[col].astype(str)
    out["net_pnl"] = (
        pd.to_numeric(out["position_size"], errors="coerce").fillna(0.0)
        * pd.to_numeric(out["net_return"], errors="coerce").fillna(0.0)
    )
    out["gross_pnl"] = (
        pd.to_numeric(out["position_size"], errors="coerce").fillna(0.0)
        * pd.to_numeric(out.get("gross_return", out["net_return"]), errors="coerce").fillna(0.0)
    )
    return out


def _make_key_index(df: pd.DataFrame) -> pd.Index:
    keys = _canonicalise_keys(df)[KEY_COLS]
    return pd.MultiIndex.from_frame(keys)


def _ensure_unique_keys(df: pd.DataFrame, label: str) -> int:
    idx = _make_key_index(df)
    dupes = int(idx.duplicated().sum())
    if dupes:
        raise ValueError(f"{label} has {dupes} duplicate decision keys")
    return dupes


def _safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _rate(mask: pd.Series) -> float:
    if len(mask) == 0:
        return np.nan
    return float(mask.fillna(False).mean())


def _cohort_metrics(
    *,
    comparison: str,
    cohort: str,
    base_rows: pd.DataFrame,
    challenger_rows: pd.DataFrame,
    base_candidates: pd.DataFrame,
    challenger_candidates: pd.DataFrame,
) -> dict[str, Any]:
    rows = challenger_rows if not challenger_rows.empty else base_rows
    if rows.empty:
        rows = pd.DataFrame(columns=KEY_COLS)
    base_pnl = float(_safe_numeric(base_rows, "net_pnl").sum()) if not base_rows.empty else 0.0
    challenger_pnl = (
        float(_safe_numeric(challenger_rows, "net_pnl").sum()) if not challenger_rows.empty else 0.0
    )
    base_gross = float(_safe_numeric(base_rows, "gross_pnl").sum()) if not base_rows.empty else 0.0
    challenger_gross = (
        float(_safe_numeric(challenger_rows, "gross_pnl").sum()) if not challenger_rows.empty else 0.0
    )
    net_return = _safe_numeric(rows, "net_return")
    exit_reason = rows.get("simple_policy_exit_reason", pd.Series(index=rows.index, dtype=object)).astype(str)
    out = {
        "comparison": comparison,
        "cohort": cohort,
        "trade_count": int(len(rows)),
        "timestamp_count": int(pd.to_datetime(rows.get("timestamp"), utc=True, errors="coerce").nunique())
        if "timestamp" in rows
        else 0,
        "symbol_count": int(rows.get("symbol", pd.Series(dtype=object)).astype(str).nunique())
        if "symbol" in rows
        else 0,
        "head_count": int(rows.get("head", pd.Series(dtype=object)).astype(str).nunique())
        if "head" in rows
        else 0,
        "base_net_pnl": base_pnl,
        "challenger_net_pnl": challenger_pnl,
        "delta_net_pnl": challenger_pnl - base_pnl,
        "base_gross_pnl": base_gross,
        "challenger_gross_pnl": challenger_gross,
        "delta_gross_pnl": challenger_gross - base_gross,
        "win_rate": _rate(net_return > 0.0),
        "mean_net_return": float(net_return.mean()) if net_return.notna().any() else np.nan,
        "median_net_return": float(net_return.median()) if net_return.notna().any() else np.nan,
        "q05_net_return": float(net_return.quantile(0.05)) if net_return.notna().any() else np.nan,
        "full_sl_rate": _rate(exit_reason.str.lower().isin(["sl", "full_sl"])),
        "timeout_rate": _rate(exit_reason.str.lower().eq("timeout")),
        "mean_anchor_score": float(_safe_numeric(rows, "component_anchor_score").mean()),
        "mean_anchor_rank": float(_safe_numeric(rows, "component_anchor_component_rank").mean()),
        "mean_qfail_score": float(_safe_numeric(rows, "component_qfail_component_score").mean()),
        "mean_qfail_rank": float(_safe_numeric(rows, "component_qfail_component_rank").mean()),
        "mean_period_score": float(_safe_numeric(rows, "component_period_component_score").mean()),
        "mean_period_rank": float(_safe_numeric(rows, "component_period_component_rank").mean()),
        "mean_final_score": float(_safe_numeric(rows, "reliability_blend_score").mean()),
        "mean_final_frozen_rank": float(_safe_numeric(rows, "policy_rank_pct").mean()),
        "mean_dynamic_threshold": float(_safe_numeric(rows, "dynamic_threshold").mean()),
        "mean_portfolio_priority": float(_safe_numeric(rows, "portfolio_priority").mean()),
        "mean_position_size": float(_safe_numeric(rows, "position_size").mean()),
    }
    # For added/removed cohorts, test whether the opposite arm's candidate row was
    # already eligible under the anchor-like frozen rank threshold.
    if cohort == "added":
        candidate_frame = base_candidates
    elif cohort == "removed":
        candidate_frame = challenger_candidates
    else:
        candidate_frame = pd.DataFrame()
    if not candidate_frame.empty and not rows.empty:
        candidate_lookup = candidate_frame.set_index(_make_key_index(candidate_frame), drop=False)
        row_idx = _make_key_index(rows)
        matched = candidate_lookup.reindex(row_idx)
        rank = _safe_numeric(matched, "policy_rank_pct")
        threshold = _safe_numeric(matched, "deployment_rank_threshold")
        out["opposite_arm_candidate_match_rate"] = float(matched["timestamp"].notna().mean())
        out["opposite_arm_rank_ge_070_rate"] = _rate(rank >= 0.70)
        out["opposite_arm_rank_ge_threshold_rate"] = _rate(rank >= threshold)
        out["opposite_arm_mean_rank"] = float(rank.mean()) if rank.notna().any() else np.nan
        out["opposite_arm_mean_threshold"] = float(threshold.mean()) if threshold.notna().any() else np.nan
    return out


def _transition_rows_for_comparison(
    *,
    comparison: str,
    base: pd.DataFrame,
    challenger: pd.DataFrame,
    base_candidates: pd.DataFrame,
    challenger_candidates: pd.DataFrame,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    _ensure_unique_keys(base, f"{comparison}:base accepted")
    _ensure_unique_keys(challenger, f"{comparison}:challenger accepted")
    base_idx = _make_key_index(base)
    challenger_idx = _make_key_index(challenger)
    base_by_key = base.set_index(base_idx, drop=False)
    challenger_by_key = challenger.set_index(challenger_idx, drop=False)
    common = base_idx.intersection(challenger_idx)
    added = challenger_idx.difference(base_idx)
    removed = base_idx.difference(challenger_idx)

    cohort_pairs = {
        "common": (base_by_key.loc[common].reset_index(drop=True), challenger_by_key.loc[common].reset_index(drop=True)),
        "added": (pd.DataFrame(columns=base.columns), challenger_by_key.loc[added].reset_index(drop=True)),
        "removed": (base_by_key.loc[removed].reset_index(drop=True), pd.DataFrame(columns=challenger.columns)),
    }
    summary_rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []
    for cohort, (base_rows, challenger_rows) in cohort_pairs.items():
        summary_rows.append(
            _cohort_metrics(
                comparison=comparison,
                cohort=cohort,
                base_rows=base_rows,
                challenger_rows=challenger_rows,
                base_candidates=base_candidates,
                challenger_candidates=challenger_candidates,
            )
        )
        detail_source = challenger_rows if not challenger_rows.empty else base_rows
        if detail_source.empty:
            continue
        detail = detail_source[KEY_COLS].copy()
        detail["comparison"] = comparison
        detail["cohort"] = cohort
        for prefix, frame in (("base", base_rows), ("challenger", challenger_rows)):
            if frame.empty:
                for col in (
                    "net_pnl",
                    "gross_pnl",
                    "net_return",
                    "simple_policy_exit_reason",
                    "position_size",
                    "portfolio_priority",
                    "dynamic_threshold",
                    "policy_rank_pct",
                    "reliability_blend_score",
                    "component_anchor_component_rank",
                    "component_qfail_component_rank",
                    "component_period_component_rank",
                ):
                    detail[f"{prefix}_{col}"] = np.nan
                continue
            aligned = frame.reset_index(drop=True)
            for col in (
                "net_pnl",
                "gross_pnl",
                "net_return",
                "simple_policy_exit_reason",
                "position_size",
                "portfolio_priority",
                "dynamic_threshold",
                "policy_rank_pct",
                "reliability_blend_score",
                "component_anchor_component_rank",
                "component_qfail_component_rank",
                "component_period_component_rank",
            ):
                detail[f"{prefix}_{col}"] = aligned[col].to_numpy() if col in aligned.columns else np.nan
        detail["delta_net_pnl"] = (
            pd.to_numeric(detail["challenger_net_pnl"], errors="coerce").fillna(0.0)
            - pd.to_numeric(detail["base_net_pnl"], errors="coerce").fillna(0.0)
        )
        detail_frames.append(detail)

    common_base = cohort_pairs["common"][0]
    common_ch = cohort_pairs["common"][1]
    if not common_base.empty and not common_ch.empty:
        base_size = _safe_numeric(common_base, "position_size")
        ch_size = _safe_numeric(common_ch, "position_size")
        rel_diff = (ch_size - base_size).abs() / base_size.abs().clip(lower=1e-9)
        resized_mask = rel_diff > 0.05
        resized_base = common_base.loc[resized_mask].reset_index(drop=True)
        resized_ch = common_ch.loc[resized_mask].reset_index(drop=True)
        summary_rows.append(
            _cohort_metrics(
                comparison=comparison,
                cohort="resized_common_gt5pct",
                base_rows=resized_base,
                challenger_rows=resized_ch,
                base_candidates=base_candidates,
                challenger_candidates=challenger_candidates,
            )
        )
    details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    return summary_rows, details


def _quantile_metrics(values: pd.Series, prefix: str) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return {
        f"{prefix}_q01": float(clean.quantile(0.01)) if clean.notna().any() else np.nan,
        f"{prefix}_q05": float(clean.quantile(0.05)) if clean.notna().any() else np.nan,
        f"{prefix}_q25": float(clean.quantile(0.25)) if clean.notna().any() else np.nan,
        f"{prefix}_q50": float(clean.quantile(0.50)) if clean.notna().any() else np.nan,
        f"{prefix}_q75": float(clean.quantile(0.75)) if clean.notna().any() else np.nan,
        f"{prefix}_q95": float(clean.quantile(0.95)) if clean.notna().any() else np.nan,
        f"{prefix}_q99": float(clean.quantile(0.99)) if clean.notna().any() else np.nan,
    }


def _rank_reference_audit(
    *,
    arm: str,
    candidates: pd.DataFrame,
    decisions: pd.DataFrame,
) -> list[dict[str, Any]]:
    accepted_idx = set(pd.to_numeric(decisions.loc[decisions["accepted"], "candidate_index"], errors="coerce").dropna().astype(int))
    candidates = _canonicalise_keys(candidates.reset_index(drop=True))
    candidates["_accepted"] = [i in accepted_idx for i in range(len(candidates))]
    rows: list[dict[str, Any]] = []
    for head, group in [("ALL", candidates)] + list(candidates.groupby("head", sort=True)):
        score = _safe_numeric(group, "reliability_blend_score")
        rank = _safe_numeric(group, "policy_rank_pct")
        threshold = _safe_numeric(group, "deployment_rank_threshold")
        rec = {
            "arm": arm,
            "head": head,
            "candidate_rows": int(len(group)),
            "accepted_count": int(group["_accepted"].sum()),
            "portfolio_acceptance_rate": float(group["_accepted"].mean()) if len(group) else np.nan,
            "score_nonfinite_rate": float((~np.isfinite(score)).mean()) if len(score) else np.nan,
            "rank_nonfinite_rate": float((~np.isfinite(rank)).mean()) if len(rank) else np.nan,
            "frac_rank_ge_070": _rate(rank >= 0.70),
            "frac_rank_ge_080": _rate(rank >= 0.80),
            "frac_rank_ge_090": _rate(rank >= 0.90),
            "frac_rank_ge_threshold": _rate(rank >= threshold),
            "mean_threshold": float(threshold.mean()) if threshold.notna().any() else np.nan,
        }
        rec.update(_quantile_metrics(score, "raw_score"))
        rec.update(_quantile_metrics(rank, "frozen_rank"))
        rows.append(rec)
    return rows


def _rejection_audit(arm: str, decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()
    work = decisions.copy()
    work["head"] = work["strategy_id"].map(_infer_head)
    rows = []
    for head, group in [("ALL", work)] + list(work.groupby("head", sort=True)):
        counts = group["rejection_reason"].astype(str).value_counts(dropna=False)
        for reason, count in counts.items():
            rows.append(
                {
                    "arm": arm,
                    "head": head,
                    "rejection_reason": reason,
                    "count": int(count),
                    "share": float(count / max(len(group), 1)),
                }
            )
    return pd.DataFrame(rows)


def _render_report(
    *,
    summary: pd.DataFrame,
    rank_audit: pd.DataFrame,
    rejections: pd.DataFrame,
    manifest: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# Reliability Blend Policy Transition Diagnostics")
    lines.append("")
    lines.append(f"Generated: {manifest['generated_at_utc']}")
    lines.append("")
    lines.append("## Accepted-Trade Transitions")
    display_cols = [
        "comparison",
        "cohort",
        "trade_count",
        "base_net_pnl",
        "challenger_net_pnl",
        "delta_net_pnl",
        "win_rate",
        "full_sl_rate",
        "timeout_rate",
        "mean_anchor_rank",
        "mean_qfail_rank",
        "mean_period_rank",
        "mean_final_frozen_rank",
        "opposite_arm_rank_ge_threshold_rate",
    ]
    lines.extend(_markdown_table(summary, display_cols))
    lines.append("")
    lines.append("## Rank-Reference Audit")
    rank_cols = [
        "arm",
        "head",
        "candidate_rows",
        "accepted_count",
        "portfolio_acceptance_rate",
        "frac_rank_ge_070",
        "frac_rank_ge_080",
        "frac_rank_ge_090",
        "frac_rank_ge_threshold",
        "raw_score_q50",
        "raw_score_q95",
        "frozen_rank_q50",
        "frozen_rank_q95",
    ]
    lines.extend(_markdown_table(rank_audit, rank_cols))
    lines.append("")
    lines.append("## Dominant Rejection Reasons")
    if not rejections.empty:
        top_rej = (
            rejections.sort_values(["arm", "head", "count"], ascending=[True, True, False])
            .groupby(["arm", "head"], as_index=False)
            .head(5)
        )
        lines.extend(_markdown_table(top_rej, ["arm", "head", "rejection_reason", "count", "share"]))
    else:
        lines.append("No rejection audit rows.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- `added` rows are trades accepted only by the challenger arm; these directly explain participation expansion.")
    lines.append("- `removed` rows are anchor trades not accepted by the challenger; their PnL is the opportunity cost of the overlay.")
    lines.append("- `opposite_arm_rank_ge_threshold_rate` tests whether added rows were already anchor-eligible, or whether the challenger expanded into below-anchor-rank rows.")
    lines.append("- Rank-reference rates use the frozen policy rank already materialized in each candidate table; no whole-window rank normalization is used.")
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
    parser.add_argument("--policy-manifest", type=Path, default=DEFAULT_POLICY_MANIFEST)
    parser.add_argument("--policy-variant", default="refit_bar4_strategy_bar2")
    parser.add_argument("--train-candidates", type=Path, default=DEFAULT_TRAIN_CANDIDATES)
    parser.add_argument("--component-scores", type=Path, default=DEFAULT_COMPONENT_SCORES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--market-mode", default="perps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    params, policy_manifest = _load_policy_params(args.policy_manifest, args.policy_variant)
    train = normalise_candidate_table(pd.read_parquet(args.train_candidates))
    ev_curve = fit_hierarchical_ev_curves(train)
    components = _load_components_fast(args.component_scores)

    candidates_by_arm: dict[str, pd.DataFrame] = {}
    decisions_by_arm: dict[str, pd.DataFrame] = {}
    accepted_by_arm: dict[str, pd.DataFrame] = {}
    metrics_by_arm: dict[str, dict[str, Any]] = {}
    rank_rows: list[dict[str, Any]] = []
    rejection_frames: list[pd.DataFrame] = []

    for arm, path in ARM_CANDIDATE_PATHS.items():
        candidates = _enrich_candidates(path, components)
        decisions, equity, metrics = replay_candidates(
            candidates,
            params,
            mode="global_auction",
            ev_curve=ev_curve,
            market_mode=args.market_mode,
        )
        accepted = _accepted_trades(candidates, decisions, arm)
        candidates_by_arm[arm] = candidates
        decisions_by_arm[arm] = decisions
        accepted_by_arm[arm] = accepted
        metrics_by_arm[arm] = metrics
        rank_rows.extend(_rank_reference_audit(arm=arm, candidates=candidates, decisions=decisions))
        rejection_frames.append(_rejection_audit(arm, decisions))
        decisions.to_parquet(args.output_dir / f"{arm}_decisions.parquet", index=False)
        accepted.to_parquet(args.output_dir / f"{arm}_accepted_trades.parquet", index=False)
        equity.to_parquet(args.output_dir / f"{arm}_equity_curve.parquet", index=False)

    summary_rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []
    for challenger in ("A1_anchor_qfail", "B0_full_native_blend"):
        rows, details = _transition_rows_for_comparison(
            comparison=f"{challenger}_vs_A0_anchor_only",
            base=accepted_by_arm["A0_anchor_only"],
            challenger=accepted_by_arm[challenger],
            base_candidates=candidates_by_arm["A0_anchor_only"],
            challenger_candidates=candidates_by_arm[challenger],
        )
        summary_rows.extend(rows)
        detail_frames.append(details)

    summary = pd.DataFrame(summary_rows)
    details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    rank_audit = pd.DataFrame(rank_rows)
    rejections = pd.concat(rejection_frames, ignore_index=True) if rejection_frames else pd.DataFrame()
    metrics = pd.DataFrame(
        [
            {"arm": arm, **{k: v for k, v in metric.items() if isinstance(v, (int, float, str, bool)) or v is None}}
            for arm, metric in metrics_by_arm.items()
        ]
    )

    summary.to_csv(args.output_dir / "accepted_trade_transition_summary.csv", index=False)
    details.to_csv(args.output_dir / "accepted_trade_transition_details.csv", index=False)
    rank_audit.to_csv(args.output_dir / "rank_reference_audit.csv", index=False)
    rejections.to_csv(args.output_dir / "rejection_reason_audit.csv", index=False)
    metrics.to_csv(args.output_dir / "replayed_portfolio_metrics.csv", index=False)

    manifest = {
        "generated_by": "analyze_reliability_blend_policy_transitions",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy_manifest": str(args.policy_manifest),
        "policy_variant": args.policy_variant,
        "train_candidates": str(args.train_candidates),
        "component_scores": str(args.component_scores),
        "arm_candidate_paths": {k: str(v) for k, v in ARM_CANDIDATE_PATHS.items()},
        "policy_params": params.to_live_config(),
        "policy_manifest_train_candidates": policy_manifest.get("train_candidates"),
        "outputs": {
            "transition_summary": str(args.output_dir / "accepted_trade_transition_summary.csv"),
            "transition_details": str(args.output_dir / "accepted_trade_transition_details.csv"),
            "rank_reference_audit": str(args.output_dir / "rank_reference_audit.csv"),
            "rejection_reason_audit": str(args.output_dir / "rejection_reason_audit.csv"),
            "replayed_portfolio_metrics": str(args.output_dir / "replayed_portfolio_metrics.csv"),
            "report": str(args.output_dir / "reliability_blend_policy_transition_diagnostics.md"),
        },
    }
    (args.output_dir / "transition_diagnostics_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2) + "\n",
        encoding="utf-8",
    )
    report = _render_report(
        summary=summary,
        rank_audit=rank_audit,
        rejections=rejections,
        manifest=manifest,
    )
    (args.output_dir / "reliability_blend_policy_transition_diagnostics.md").write_text(
        report,
        encoding="utf-8",
    )
    print(json.dumps(_json_safe({"outputs": manifest["outputs"], "metrics": metrics_by_arm}), indent=2)[:8000])


if __name__ == "__main__":
    main()
