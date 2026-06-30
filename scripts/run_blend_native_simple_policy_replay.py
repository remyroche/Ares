#!/usr/bin/env python3
"""Replay native simple-policy candidates after re-ranking with reliability blend scores.

The existing simple_policy_optimiser candidate tables already contain the native
entry/exit replay, including fees, spread, slippage, delayed entry, and realized
``net_return``.  This script does not recompute TP/SL outcomes.  It re-keys those
native candidate rows to reliability-blend component scores, replaces the ranking
columns with the selected blend score, and selects per-strategy thresholds from
native replay economics.

Rows without a materialized blend score are dropped by default.  Falling back to
the original meta score would make the resulting artifact ambiguous and defeat the
purpose of this audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_fixed_tpsl_blend_simple_policy_optimiser import (
    SIDES,
    STRATEGY_IDS,
    _file_sha256,
    _json_safe,
)
from scripts.reliability_blend_rank_reference import apply_frozen_policy_rank_reference


DEFAULT_NATIVE_CANDIDATES = Path(
    "data_perp/artifacts/20260620_185313_no_mkt4_evband002_policy_uncertainty_ev"
    "/simple_policy_optimiser/simple_policy_candidates_broad.parquet"
)
DEFAULT_NATIVE_METADATA = Path(
    "data_perp/artifacts/20260620_185313_no_mkt4_evband002_policy_uncertainty_ev"
    "/simple_policy_optimiser/simple_policy_candidates_broad_metadata.json"
)
DEFAULT_BLEND_SCORES = Path(
    "data_perp/reports/reliability_blend_optuna_20260623_full"
    "/reliability_blend_component_scores.parquet"
)
DEFAULT_BLEND_CONFIG = Path("config/reliability_blend_default_configs.json")
DEFAULT_OUTPUT_RUN_ID = "reliability_blend_native_simple_policy_replay_20260624"

HEAD_BY_STRATEGY_ID = {strategy_id: head for head, strategy_id in STRATEGY_IDS.items()}


def _rank_pct(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.rank(method="average", pct=True)


def _load_default_variants(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text())
    configs = payload.get("configs") or {}
    variants: dict[str, str] = {}
    for head, cfg in configs.items():
        variant = str((cfg or {}).get("variant") or "").strip()
        if variant:
            variants[str(head)] = variant
    return variants


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"read_error": f"failed_to_parse:{path}"}


def _candidate_week(values: pd.Series) -> pd.Series:
    ts = pd.to_datetime(values, utc=True, errors="coerce").dt.tz_convert(None)
    return ts.dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d")


def _metric_block(rows: pd.DataFrame) -> dict[str, Any]:
    n = int(len(rows))
    if n == 0:
        return {
            "trades": 0,
            "timestamps": 0,
            "symbols": 0,
            "win_rate": np.nan,
            "mean_net": np.nan,
            "median_net": np.nan,
            "q05_net": np.nan,
            "q25_net": np.nan,
            "net_pnl": 0.0,
            "gross_pnl": 0.0,
            "cost_pnl": 0.0,
        }
    net = pd.to_numeric(rows["net_return"], errors="coerce").fillna(0.0)
    gross = pd.to_numeric(rows.get("gross_return", net), errors="coerce").fillna(0.0)
    return {
        "trades": n,
        "timestamps": int(pd.to_datetime(rows["timestamp"], utc=True, errors="coerce").nunique()),
        "symbols": int(rows["symbol"].astype(str).nunique()),
        "win_rate": float((net > 0.0).mean()),
        "mean_net": float(net.mean()),
        "median_net": float(net.median()),
        "q05_net": float(net.quantile(0.05)),
        "q25_net": float(net.quantile(0.25)),
        "net_pnl": float(net.sum()),
        "gross_pnl": float(gross.sum()),
        "cost_pnl": float((gross - net).sum()),
    }


def _threshold_metrics(rows: pd.DataFrame) -> dict[str, Any]:
    out = _metric_block(rows)
    n = int(out["trades"])
    if n:
        reasons = rows.get("simple_policy_exit_reason", pd.Series("", index=rows.index)).astype(str)
        out.update(
            {
                "trailing_rate": float((reasons == "trailing").mean()),
                "full_sl_rate": float((reasons == "full_sl").mean()),
                "timeout_rate": float((reasons == "timeout").mean()),
                "adverse_rate": float(reasons.str.contains("adverse", case=False, na=False).mean()),
            }
        )
    else:
        out.update(
            {
                "trailing_rate": np.nan,
                "full_sl_rate": np.nan,
                "timeout_rate": np.nan,
                "adverse_rate": np.nan,
            }
        )
    return out


def _load_and_join(
    *,
    native_candidates_path: Path,
    blend_scores_path: Path,
    config_path: Path,
    start: str | None,
    end: str | None,
    keep_missing_blend: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    variants = _load_default_variants(config_path)
    candidates = pd.read_parquet(native_candidates_path)
    candidates["timestamp"] = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce")
    candidates["head"] = candidates["strategy_id"].astype(str).map(HEAD_BY_STRATEGY_ID)
    if candidates["head"].isna().any():
        missing = sorted(candidates.loc[candidates["head"].isna(), "strategy_id"].astype(str).unique())
        raise RuntimeError(f"Unknown strategy_id values in native candidates: {missing[:10]}")
    if start:
        candidates = candidates[candidates["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        candidates = candidates[candidates["timestamp"] <= pd.Timestamp(end, tz="UTC")]
    if candidates.empty:
        raise RuntimeError("No native candidates after date filtering.")
    dupes = int(candidates.duplicated(["head", "timestamp", "symbol"]).sum())
    if dupes:
        raise RuntimeError(
            "Native candidate join key is not unique: "
            f"{dupes} duplicate head/timestamp/symbol rows."
        )

    score_cols = ["head", "timestamp", "symbol", "anchor_score"]
    selected_cols: set[str] = set(score_cols)
    for variant in variants.values():
        selected_cols.add(f"blend_{variant}_score")
        selected_cols.add(f"blend_{variant}_rank")
    available = pd.read_parquet(blend_scores_path)
    available["timestamp"] = pd.to_datetime(available["timestamp"], utc=True, errors="coerce")
    keep_cols = [col for col in selected_cols if col in available.columns]
    scores = available[keep_cols].copy()
    if scores.duplicated(["head", "timestamp", "symbol"]).any():
        raise RuntimeError("Blend score table contains duplicate head/timestamp/symbol keys.")

    joined = candidates.merge(scores, on=["head", "timestamp", "symbol"], how="left", indicator=True)
    coverage_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    selected = np.full(len(joined), np.nan, dtype=np.float64)
    selected_variant: list[str | None] = [None] * len(joined)
    for head, idx in joined.groupby("head", sort=True).groups.items():
        variant = variants.get(str(head))
        col = f"blend_{variant}_score" if variant else ""
        positions = np.asarray(idx, dtype=np.int64)
        if not variant or col not in joined.columns:
            missing_rows.append(
                {
                    "head": head,
                    "variant": variant,
                    "missing_score_column": col,
                    "rows": int(len(positions)),
                }
            )
            continue
        selected[positions] = pd.to_numeric(joined.iloc[positions][col], errors="coerce").to_numpy(dtype=np.float64)
        for pos in positions:
            selected_variant[int(pos)] = variant

    joined["reliability_blend_score"] = selected
    joined["reliability_blend_variant"] = selected_variant
    joined["blend_score_available"] = np.isfinite(selected)
    joined["_source_join_status"] = joined["_merge"].astype(str)

    for head, group in joined.groupby("head", sort=True):
        coverage_rows.append(
            {
                "head": head,
                "native_rows": int(len(group)),
                "matched_blend_key_rows": int((group["_merge"] == "both").sum()),
                "blend_score_rows": int(group["blend_score_available"].sum()),
                "blend_score_coverage": float(group["blend_score_available"].mean()),
                "timestamp_min": group["timestamp"].min().isoformat(),
                "timestamp_max": group["timestamp"].max().isoformat(),
                "variant": variants.get(str(head)),
            }
        )

    joined = joined.drop(columns=["_merge"])
    if not keep_missing_blend:
        joined = joined[joined["blend_score_available"]].copy()
    if joined.empty:
        raise RuntimeError("No rows with materialized reliability-blend scores.")
    return joined.reset_index(drop=True), pd.DataFrame(coverage_rows), missing_rows


def _materialise_blend_ranked_candidates(
    joined: pd.DataFrame,
    *,
    data_root: Path,
    rank_reference_run_id: str | None,
    allow_window_rank_debug: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = joined.copy()
    out["anchor_calibrated_score"] = pd.to_numeric(out["calibrated_score"], errors="coerce")
    if "normalized_rank_score" in out.columns:
        out["anchor_normalized_rank_score"] = pd.to_numeric(out["normalized_rank_score"], errors="coerce")
    if "strategy_rank_pct" in out.columns:
        out["anchor_strategy_rank_pct"] = pd.to_numeric(out["strategy_rank_pct"], errors="coerce")
    if "auction_rank_score" in out.columns:
        out["anchor_auction_rank_score"] = pd.to_numeric(out["auction_rank_score"], errors="coerce")
    if "policy_rank_pct" in out.columns:
        out["anchor_policy_rank_pct"] = pd.to_numeric(out["policy_rank_pct"], errors="coerce")

    out["calibrated_score"] = pd.to_numeric(out["reliability_blend_score"], errors="coerce")
    out["blend_window_strategy_rank_pct_debug"] = out.groupby("strategy_id", group_keys=False)["calibrated_score"].apply(_rank_pct)
    out["blend_window_auction_rank_pct_debug"] = _rank_pct(out["calibrated_score"])
    out["score_source"] = "reliability_blend_default_variant"
    out["blend_native_replay_key"] = (
        out["head"].astype(str)
        + "|"
        + out["timestamp"].astype(str)
        + "|"
        + out["symbol"].astype(str)
    )
    out, rank_diag = apply_frozen_policy_rank_reference(
        out,
        data_root=data_root,
        run_id=rank_reference_run_id,
        score_col="calibrated_score",
        allow_window_rank_debug=bool(allow_window_rank_debug),
    )
    return out, rank_diag


def _select_thresholds(
    candidates: pd.DataFrame,
    *,
    threshold_lo: float,
    threshold_hi: float,
    threshold_step: float,
    local_band_width: float,
    min_trades: int,
    min_mean_net: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    thresholds = np.unique(
        np.round(
            np.arange(
                float(threshold_lo),
                float(threshold_hi) + float(threshold_step) / 2.0,
                float(threshold_step),
            ),
            4,
        )
    )
    rows: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for (head, strategy_id), group in candidates.groupby(["head", "strategy_id"], sort=True):
        rank = pd.to_numeric(group["policy_rank_pct"], errors="coerce").fillna(-np.inf)
        strategy_rows: list[dict[str, Any]] = []
        for threshold in thresholds:
            cumulative = group.loc[rank >= threshold]
            local = group.loc[(rank >= threshold) & (rank < min(1.0 + 1e-9, threshold + local_band_width))]
            rec = {
                "head": head,
                "strategy_id": strategy_id,
                "deployment_rank_threshold": float(threshold),
                "rank_col": "policy_rank_pct",
                "score_source": "reliability_blend_default_variant",
                **{f"cumulative_{k}": v for k, v in _threshold_metrics(cumulative).items()},
                **{f"local_{k}": v for k, v in _threshold_metrics(local).items()},
            }
            rec["selection_eligible"] = bool(
                int(rec["cumulative_trades"]) >= int(min_trades)
                and float(rec["cumulative_mean_net"]) >= float(min_mean_net)
            )
            rec["objective"] = float(rec["cumulative_net_pnl"])
            strategy_rows.append(rec)
            rows.append(rec)
        eligible = [r for r in strategy_rows if r["selection_eligible"]]
        pool = eligible or strategy_rows
        best = max(
            pool,
            key=lambda r: (
                float(r["objective"]),
                float(r["cumulative_mean_net"]),
                float(r["cumulative_win_rate"]),
                int(r["cumulative_trades"]),
            ),
        )
        selected.append(
            {
                **best,
                "selection_reason": (
                    "max_native_net_pnl_with_min_trades_and_mean_net"
                    if eligible
                    else "fallback_max_native_net_pnl_no_eligible_threshold"
                ),
                "eligible_threshold_count": int(len(eligible)),
                "threshold_lo": float(threshold_lo),
                "threshold_hi": float(threshold_hi),
                "threshold_step": float(threshold_step),
                "local_band_width": float(local_band_width),
                "min_trades": int(min_trades),
                "min_mean_net": float(min_mean_net),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(selected)


def _apply_thresholds(candidates: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    thresholds = selected.set_index("strategy_id")["deployment_rank_threshold"].astype(float).to_dict()
    out = candidates.copy()
    out["base_strategy_threshold"] = out["strategy_id"].astype(str).map(thresholds)
    out["deployment_rank_threshold"] = out["base_strategy_threshold"]
    keep = pd.to_numeric(out["policy_rank_pct"], errors="coerce") >= pd.to_numeric(
        out["deployment_rank_threshold"], errors="coerce"
    )
    return out.loc[keep.fillna(False)].reset_index(drop=True)


def _parse_forced_thresholds(values: list[str] | None) -> dict[str, float]:
    forced: dict[str, float] = {}
    for raw in values or []:
        if "=" not in raw:
            raise ValueError(f"Invalid --force-threshold value {raw!r}; expected HEAD=THRESHOLD.")
        head, value = raw.split("=", 1)
        head = head.strip()
        if not head:
            raise ValueError(f"Invalid --force-threshold value {raw!r}; head is empty.")
        threshold = float(value)
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"Invalid threshold for {head}: {threshold}; expected 0..1.")
        forced[head] = threshold
    return forced


def _apply_forced_thresholds(
    grid: pd.DataFrame,
    selected: pd.DataFrame,
    forced: dict[str, float],
) -> pd.DataFrame:
    if not forced:
        return selected
    out = selected.copy()
    grid_thresholds = pd.to_numeric(grid["deployment_rank_threshold"], errors="coerce")
    for head, threshold in forced.items():
        head_mask = grid["head"].astype(str).eq(head)
        if not bool(head_mask.any()):
            raise ValueError(f"Cannot force threshold for unknown head {head!r}.")
        distances = (grid_thresholds - float(threshold)).abs()
        candidate = grid.loc[head_mask].iloc[[int(distances.loc[head_mask].argmin())]].copy()
        if candidate.empty:
            raise ValueError(f"No threshold grid row found for forced head {head!r}.")
        forced_row = candidate.iloc[0].to_dict()
        forced_row.update(
            {
                "selection_reason": f"forced_threshold_override:{threshold:.4f}",
                "eligible_threshold_count": int(
                    selected.loc[selected["head"].astype(str).eq(head), "eligible_threshold_count"].iloc[0]
                )
                if bool(selected["head"].astype(str).eq(head).any())
                else 0,
                "threshold_lo": float(selected["threshold_lo"].iloc[0]) if "threshold_lo" in selected else np.nan,
                "threshold_hi": float(selected["threshold_hi"].iloc[0]) if "threshold_hi" in selected else np.nan,
                "threshold_step": float(selected["threshold_step"].iloc[0]) if "threshold_step" in selected else np.nan,
                "local_band_width": float(selected["local_band_width"].iloc[0])
                if "local_band_width" in selected
                else np.nan,
                "min_trades": int(selected["min_trades"].iloc[0]) if "min_trades" in selected else 0,
                "min_mean_net": float(selected["min_mean_net"].iloc[0]) if "min_mean_net" in selected else np.nan,
            }
        )
        out = out.loc[~out["head"].astype(str).eq(head)].copy()
        out = pd.concat([out, pd.DataFrame([forced_row])], ignore_index=True, sort=False)
    return out.sort_values(["head", "strategy_id"]).reset_index(drop=True)


def _weekly_metrics(rows: pd.DataFrame, *, group_cols: list[str]) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    work = rows.copy()
    work["week"] = _candidate_week(work["timestamp"])
    records: list[dict[str, Any]] = []
    for keys, group in work.groupby(["week"] + group_cols, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = {"week": keys[0]}
        for col, val in zip(group_cols, keys[1:]):
            rec[col] = val
        rec.update(_metric_block(group))
        records.append(rec)
    return pd.DataFrame(records)


def _write_report(
    *,
    report_path: Path,
    output_root: Path,
    candidates: pd.DataFrame,
    deployable: pd.DataFrame,
    coverage: pd.DataFrame,
    selected: pd.DataFrame,
    source_native_path: Path,
    source_blend_path: Path,
    start: str | None,
    end: str | None,
) -> None:
    src_ts = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce")
    dep_ts = pd.to_datetime(deployable["timestamp"], utc=True, errors="coerce")
    lines = [
        "# Reliability-Blend Native Simple-Policy Replay",
        "",
        "Metric type: native `simple_policy_optimiser` candidate replay re-ranked by reliability-blend score.",
        "",
        f"Native source: `{source_native_path}`",
        f"Blend score source: `{source_blend_path}`",
        f"Requested date filter: `{start or ''}` to `{end or ''}`",
        f"Candidate replay range after blend-score filtering: {src_ts.min()} to {src_ts.max()}",
        f"Deployable replay range: {dep_ts.min()} to {dep_ts.max()}",
        f"Blend-ranked candidate rows: {len(candidates):,}",
        f"Deployable rows after selected thresholds: {len(deployable):,}",
        "",
        "## Blend Score Coverage",
        "",
        "| head | variant | native_rows | blend_score_rows | coverage | source_range |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in coverage.to_dict("records"):
        lines.append(
            "| {head} | {variant} | {native_rows} | {blend_score_rows} | {coverage:.2%} | {lo} -> {hi} |".format(
                head=row["head"],
                variant=row.get("variant", ""),
                native_rows=int(row["native_rows"]),
                blend_score_rows=int(row["blend_score_rows"]),
                coverage=float(row["blend_score_coverage"]),
                lo=row["timestamp_min"],
                hi=row["timestamp_max"],
            )
        )
    lines.extend(
        [
            "",
            "## Selected Thresholds",
            "",
            "| head | threshold | trades | win_rate | mean_net | net_pnl | q05 | reason |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in selected.to_dict("records"):
        lines.append(
            "| {head} | {thr:.3f} | {n} | {win:.3f} | {mean:.5f} | {pnl:.5f} | {q05:.5f} | {reason} |".format(
                head=row["head"],
                thr=float(row["deployment_rank_threshold"]),
                n=int(row["cumulative_trades"]),
                win=float(row["cumulative_win_rate"]),
                mean=float(row["cumulative_mean_net"]),
                pnl=float(row["cumulative_net_pnl"]),
                q05=float(row["cumulative_q05_net"]),
                reason=row["selection_reason"],
            )
        )
    portfolio = _metric_block(deployable)
    lines.extend(
        [
            "",
            "## Portfolio Candidate Summary",
            "",
            "| trades | timestamps | win_rate | mean_net | median_net | q05_net | net_pnl | gross_pnl | costs |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            "| {trades} | {timestamps} | {win_rate:.3f} | {mean_net:.5f} | {median_net:.5f} | {q05_net:.5f} | {net_pnl:.5f} | {gross_pnl:.5f} | {cost_pnl:.5f} |".format(
                **portfolio
            ),
            "",
            "Artifacts are under:",
            "",
            f"`{output_root}`",
            "",
            "Important limitation: this script can only re-rank rows that already have both native simple-policy replay outcomes and materialized reliability-blend component scores. It does not create deployable q_fail/new_period scorers for new post-boundary rows.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-candidates", type=Path, default=DEFAULT_NATIVE_CANDIDATES)
    parser.add_argument("--native-metadata", type=Path, default=DEFAULT_NATIVE_METADATA)
    parser.add_argument("--blend-scores", type=Path, default=DEFAULT_BLEND_SCORES)
    parser.add_argument("--blend-config", type=Path, default=DEFAULT_BLEND_CONFIG)
    parser.add_argument("--data-root", type=Path, default=Path("data_perp"))
    parser.add_argument("--output-run-id", type=str, default=DEFAULT_OUTPUT_RUN_ID)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--keep-missing-blend", action="store_true")
    parser.add_argument("--threshold-lo", type=float, default=0.50)
    parser.add_argument("--threshold-hi", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--local-band-width", type=float, default=0.05)
    parser.add_argument("--min-trades", type=int, default=10)
    parser.add_argument("--min-mean-net", type=float, default=0.0)
    parser.add_argument("--rank-reference-run-id", type=str, default=None)
    parser.add_argument(
        "--allow-window-rank-debug",
        action="store_true",
        help="Allow non-deployable current-window rank fallback for audit runs only.",
    )
    parser.add_argument(
        "--force-threshold",
        action="append",
        default=[],
        help="Override selected threshold for a head, e.g. --force-threshold long_dist=0.50.",
    )
    args = parser.parse_args()

    joined, coverage, missing = _load_and_join(
        native_candidates_path=args.native_candidates,
        blend_scores_path=args.blend_scores,
        config_path=args.blend_config,
        start=args.start,
        end=args.end,
        keep_missing_blend=bool(args.keep_missing_blend),
    )
    if bool(args.keep_missing_blend) and joined["reliability_blend_score"].isna().any():
        raise RuntimeError("--keep-missing-blend is for audit only; ranking missing blend rows is unsupported.")
    candidates, rank_reference_diag = _materialise_blend_ranked_candidates(
        joined,
        data_root=args.data_root,
        rank_reference_run_id=args.rank_reference_run_id,
        allow_window_rank_debug=bool(args.allow_window_rank_debug),
    )
    grid, selected = _select_thresholds(
        candidates,
        threshold_lo=float(args.threshold_lo),
        threshold_hi=float(args.threshold_hi),
        threshold_step=float(args.threshold_step),
        local_band_width=float(args.local_band_width),
        min_trades=int(args.min_trades),
        min_mean_net=float(args.min_mean_net),
    )
    selected = _apply_forced_thresholds(grid, selected, _parse_forced_thresholds(args.force_threshold))
    deployable = _apply_thresholds(candidates, selected)

    output_root = args.data_root / "artifacts" / args.output_run_id
    policy_dir = output_root / "simple_policy_optimiser"
    deployment_dir = policy_dir / "deployment"
    policy_params_dir = output_root / "policy_params"
    for path in (policy_dir, deployment_dir, policy_params_dir):
        path.mkdir(parents=True, exist_ok=True)

    broad_path = policy_dir / "simple_policy_candidates_broad.parquet"
    candidate_path = policy_dir / "simple_policy_candidates.parquet"
    deployable_path = policy_dir / "simple_policy_candidates_deployable.parquet"
    candidates.to_parquet(broad_path, index=False)
    deployable.to_parquet(candidate_path, index=False)
    deployable.to_parquet(deployable_path, index=False)
    grid_path = policy_dir / "blend_native_threshold_sensitivity.csv"
    selected_path = policy_dir / "blend_native_selected_thresholds.csv"
    coverage_path = policy_dir / "blend_native_score_coverage.csv"
    weekly_global_path = policy_dir / "weekly_native_blend_replay_global.csv"
    weekly_strategy_path = policy_dir / "weekly_native_blend_replay_by_strategy.csv"
    grid.to_csv(grid_path, index=False)
    selected.to_csv(selected_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    _weekly_metrics(deployable, group_cols=[]).to_csv(weekly_global_path, index=False)
    _weekly_metrics(deployable, group_cols=["head", "strategy_id"]).to_csv(weekly_strategy_path, index=False)

    strategies: list[dict[str, Any]] = []
    for row in selected.to_dict("records"):
        head_rank_sources = (
            candidates.loc[candidates["head"].eq(row["head"]), "threshold_rank_score_source"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
            if "threshold_rank_score_source" in candidates.columns
            else []
        )
        strategies.append(
            {
                "strategy_id": row["strategy_id"],
                "head": row["head"],
                "side": SIDES.get(str(row["head"]), "unknown"),
                "deployment_rank_threshold": float(row["deployment_rank_threshold"]),
                "threshold_rank_score_source": head_rank_sources[0]
                if len(head_rank_sources) == 1
                else "policy_rank_reference_percentile_mixed",
                "score_source": "reliability_blend_default_variant",
                "blend_variant": candidates.loc[candidates["head"].eq(row["head"]), "reliability_blend_variant"].dropna().astype(str).head(1).squeeze(),
                "deployment_threshold_metrics": _json_safe(row),
            }
        )
    deployment_payload = {
        "schema_version": "simple_policy_v1",
        "generated_by": "run_blend_native_simple_policy_replay",
        "run_id": args.output_run_id,
        "market_mode": "perps",
        "selection_rules": {
            "metric_type": "native_simple_policy_candidate_replay",
            "score_source": "reliability_blend_default_variant",
            "threshold_selection_objective": "max_native_net_pnl",
            "threshold_space": "per_strategy_rank_percentile",
            "threshold_rank_score_source": str(candidates["threshold_rank_score_source"].dropna().astype(str).iloc[0])
            if "threshold_rank_score_source" in candidates.columns and candidates["threshold_rank_score_source"].notna().any()
            else "policy_rank_pct",
            "rank_reference": rank_reference_diag,
            "costs_included": True,
            "fees_spread_slippage_source": "native_simple_policy_optimiser_candidate_table",
            "threshold_lo": float(args.threshold_lo),
            "threshold_hi": float(args.threshold_hi),
            "threshold_step": float(args.threshold_step),
            "min_trades": int(args.min_trades),
            "min_mean_net": float(args.min_mean_net),
            "requires_blend_score": not bool(args.keep_missing_blend),
        },
        "strategies": strategies,
        "rejected_strategies": [],
        "source": {
            "native_candidates_path": str(args.native_candidates),
            "native_candidates_sha256": _file_sha256(args.native_candidates),
            "native_metadata_path": str(args.native_metadata),
            "native_metadata_sha256": _file_sha256(args.native_metadata) if args.native_metadata.exists() else None,
            "native_metadata": _read_optional_json(args.native_metadata),
            "blend_scores_path": str(args.blend_scores),
            "blend_scores_sha256": _file_sha256(args.blend_scores),
            "blend_config_path": str(args.blend_config),
            "blend_config_sha256": _file_sha256(args.blend_config),
            "missing_blend_score_columns": missing,
            "blend_score_coverage": coverage.to_dict("records"),
            "rank_reference": rank_reference_diag,
        },
        "candidate_artifacts": {
            "broad_candidates": str(broad_path),
            "deployable_candidates": str(candidate_path),
            "threshold_sensitivity": str(grid_path),
            "selected_thresholds": str(selected_path),
            "coverage": str(coverage_path),
            "weekly_global": str(weekly_global_path),
            "weekly_by_strategy": str(weekly_strategy_path),
        },
    }
    for path in (
        deployment_dir / "best_policy_params.json",
        policy_params_dir / "best_policy_params.json",
        output_root / "best_policy_params.json",
        output_root / "strategy_for_inference.json",
    ):
        path.write_text(json.dumps(_json_safe(deployment_payload), indent=2) + "\n")

    optimisation = {
        "schema_version": "blend_native_simple_policy_replay_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric_type": "native_simple_policy_candidate_replay",
        "costs_included": True,
        "source_native_candidates": str(args.native_candidates),
        "source_blend_scores": str(args.blend_scores),
        "candidate_rows": int(len(candidates)),
        "deployable_rows": int(len(deployable)),
        "candidate_timestamp_min": pd.to_datetime(candidates["timestamp"], utc=True).min().isoformat(),
        "candidate_timestamp_max": pd.to_datetime(candidates["timestamp"], utc=True).max().isoformat(),
        "deployable_timestamp_min": pd.to_datetime(deployable["timestamp"], utc=True).min().isoformat() if len(deployable) else None,
        "deployable_timestamp_max": pd.to_datetime(deployable["timestamp"], utc=True).max().isoformat() if len(deployable) else None,
        "portfolio_candidate_summary": _metric_block(deployable),
        "selected_thresholds": selected.to_dict("records"),
        "coverage": coverage.to_dict("records"),
        "rank_reference": rank_reference_diag,
        "output_root": str(output_root),
    }
    (output_root / "policy_optimisation.json").write_text(
        json.dumps(_json_safe(optimisation), indent=2) + "\n"
    )
    (output_root / "policy_optimisation_oos_metrics.json").write_text(
        json.dumps(
            _json_safe(
                {
                    **optimisation,
                    "oos_status": "not_fresh_oos",
                    "note": (
                        "This is a native simple_policy_optimiser candidate replay re-ranked by "
                        "OOF reliability-blend scores. It is not a fresh deployable auxiliary-scorer OOS run."
                    ),
                }
            ),
            indent=2,
        )
        + "\n"
    )
    _write_report(
        report_path=policy_dir / "blend_native_simple_policy_replay_report.md",
        output_root=output_root,
        candidates=candidates,
        deployable=deployable,
        coverage=coverage,
        selected=selected,
        source_native_path=args.native_candidates,
        source_blend_path=args.blend_scores,
        start=args.start,
        end=args.end,
    )
    print(json.dumps(_json_safe(optimisation), indent=2)[:6000])
    print(f"\nWrote {output_root}")


if __name__ == "__main__":
    main()
