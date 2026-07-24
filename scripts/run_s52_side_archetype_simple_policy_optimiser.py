#!/usr/bin/env python3
"""Run simple-policy exit optimisation on S52 replay-ready candidates.

This runner bridges the S52 meta handoff candidate parquet into the main
``simple_policy_optimiser`` internals. It optimises side-parent policies first,
then side x archetype policies with the shrinkage estimator implemented in the
main optimiser.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# This runner exists specifically to run side x archetype execution optimisation.
# The optimiser keeps that path behind environment flags for the generic CLI, so
# set safe runner-local defaults before importing the module-level constants.
os.environ.setdefault("EPM_SIMPLE_POLICY_PER_ARCHETYPE_OPTIMISATION", "1")
os.environ.setdefault("EPM_SIMPLE_POLICY_PER_ARCHETYPE_FULL_OPTIMISATION", "1")
os.environ.setdefault("EPM_SIMPLE_POLICY_PER_ARCHETYPE_MAX_TRIALS", "96")
# This production runner is Kraken-perps specific.  An unset exchange used to
# resolve the local replay cache through the legacy Binance default even though
# missing 15m bars were fetched from Kraken Futures.
os.environ.setdefault("EPM_EXCHANGE", "krakenfutures")

from scripts.ablate_simple_policy_exit_geometry import (  # noqa: E402
    _load_bundles,
    _prepare_rows,
)
from extreme_price_movements.regime_ev_calibration import (  # noqa: E402
    default_regime_ev_calibration_artifact,
    default_regime_ev_feature_handoff,
)
from extreme_price_movements.simple_policy_optimiser import (  # noqa: E402
    DEFAULT_CV_FOLDS,
    _attach_policy_archetype_column,
    _evaluate_policy_subsets,
    _json_safe,
    _optimise_policy_by_archetype,
    _optimise_policy_on_rows,
    _stable_fold_policy_objective,
)


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _extract_stage_fold_diag(fit_summary: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for stage_key in ("trailing_stage", "stage2"):
        stage = fit_summary.get(stage_key)
        if not isinstance(stage, dict):
            continue
        selected = stage.get("selection", {}).get("selected_trial")
        if not isinstance(selected, dict):
            selected = {}
        out[f"{stage_key}_objective_source"] = stage.get("objective_source")
        out[f"{stage_key}_selected_objective"] = stage.get("selected_medoid_objective")
        out[f"{stage_key}_fold_mean_score"] = selected.get("stable_fold_mean_score")
        out[f"{stage_key}_fold_std_score"] = selected.get("stable_fold_std_score")
        out[f"{stage_key}_fold_worst_score"] = selected.get("stable_fold_worst_score")
        out[f"{stage_key}_objective_formula"] = selected.get(
            "stable_fold_objective_formula",
            stage.get("objective_formula"),
        )
    return out


def _month_week_metrics(
    rows: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    group_cols: list[str],
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    work = rows.copy()
    ts = pd.to_datetime(work[timestamp_col], utc=True, errors="coerce")
    work["month"] = ts.dt.strftime("%Y-%m")
    work["week"] = ts.dt.to_period("W").astype(str)
    metric_cols = [c for c in ("rank_pct", "calibrated_score", "base_score_oof", "meta_score_oof") if c in work.columns]
    rows_out: list[dict[str, Any]] = []
    for keys, group in work.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = {col: val for col, val in zip(group_cols, keys, strict=False)}
        rec["rows"] = int(len(group))
        rec["symbols"] = int(group["symbol"].astype(str).nunique()) if "symbol" in group else 0
        rec["long_share"] = float((pd.to_numeric(group.get("side"), errors="coerce") > 0.0).mean()) if "side" in group else np.nan
        for col in metric_cols:
            rec[f"mean_{col}"] = _safe_float(pd.to_numeric(group[col], errors="coerce").mean())
        rows_out.append(rec)
    return pd.DataFrame(rows_out)


def _cost_accounting_audit(rows: pd.DataFrame, *, round_trip_cost_pct: float) -> dict[str, Any]:
    lower_to_col = {str(col).lower(): str(col) for col in rows.columns}
    precomputed_return_cols = [
        lower_to_col[name]
        for name in ("ret_net", "net_return", "fees_bps", "round_trip_cost_floor")
        if name in lower_to_col
    ]
    fee_like_cols = [
        str(col)
        for col in rows.columns
        if "fee" in str(col).lower() and str(col) not in precomputed_return_cols
    ]
    spread_cols = [str(col) for col in rows.columns if "spread" in str(col).lower()]

    def _mean_col(name: str) -> float:
        if name not in rows.columns:
            return float("nan")
        return _safe_float(pd.to_numeric(rows[name], errors="coerce").mean())

    return {
        "round_trip_cost_pct_requested": float(round_trip_cost_pct),
        "cost_pct_per_side_used_by_optimizer": float(round_trip_cost_pct) / 2.0,
        "precomputed_net_or_fee_columns_in_candidates": precomputed_return_cols,
        "other_fee_like_columns_in_candidates": fee_like_cols,
        "spread_columns_in_candidates": spread_cols,
        "mean_entry_half_spread_bps": _mean_col("spread_cost_bps"),
        "mean_exit_half_spread_bps": _mean_col("exit_spread_cost_bps"),
        "fee_double_count_guard_pass": not precomputed_return_cols and not fee_like_cols,
        "accounting_note": (
            "The runner passes the requested round-trip cost once as per-side "
            "cost_pct into simple_policy_optimiser. Candidate rows must not carry "
            "precomputed net/fee returns. Entry/exit spread bps are separate "
            "execution frictions and are reported separately from the 1% fee."
        ),
    }


def _write_policy_checkpoints(
    out_dir: Path,
    *,
    parent_rows: list[dict[str, Any]],
    archetype_frames: list[pd.DataFrame],
    completed_strategy_ids: list[str],
) -> None:
    if parent_rows:
        pd.DataFrame(parent_rows).to_csv(
            out_dir / "side_parent_policy_summary.partial.csv",
            index=False,
        )
    if archetype_frames:
        pd.concat(archetype_frames, ignore_index=True).to_csv(
            out_dir / "side_archetype_policy_summary.partial.csv",
            index=False,
        )
    (out_dir / "progress.partial.json").write_text(
        json.dumps(
            _json_safe(
                {
                    "completed_strategy_ids": completed_strategy_ids,
                    "side_parent_rows": len(parent_rows),
                    "side_archetype_frames": len(archetype_frames),
                }
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _split_optimisation_holdout_rows(
    rows: pd.DataFrame,
    *,
    holdout_start: str | None,
    holdout_end: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not holdout_start and not holdout_end:
        return rows, pd.DataFrame(columns=rows.columns), {
            "enabled": False,
            "reason": "no_holdout_interval_requested",
            "optimisation_rows": int(len(rows)),
            "holdout_rows": 0,
        }
    ts = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
    start = pd.Timestamp("1900-01-01", tz="UTC")
    end = pd.Timestamp("2100-01-01", tz="UTC")
    if holdout_start:
        start = pd.Timestamp(holdout_start, tz="UTC")
    if holdout_end:
        end = pd.Timestamp(holdout_end, tz="UTC")
    holdout_mask = ts.ge(start) & ts.lt(end)
    holdout = rows.loc[holdout_mask].copy().reset_index(drop=True)
    optimise = rows.loc[~holdout_mask].copy().reset_index(drop=True)
    if optimise.empty:
        raise ValueError("Holdout interval removed all optimisation rows")
    diag = {
        "enabled": True,
        "holdout_start": str(start),
        "holdout_end": str(end),
        "optimisation_rows": int(len(optimise)),
        "holdout_rows": int(len(holdout)),
        "holdout_fraction": float(len(holdout) / max(len(rows), 1)),
        "optimisation_min_ts": str(pd.to_datetime(optimise["timestamp"], utc=True).min()),
        "optimisation_max_ts": str(pd.to_datetime(optimise["timestamp"], utc=True).max()),
        "holdout_min_ts": str(pd.to_datetime(holdout["timestamp"], utc=True).min())
        if not holdout.empty
        else None,
        "holdout_max_ts": str(pd.to_datetime(holdout["timestamp"], utc=True).max())
        if not holdout.empty
        else None,
    }
    return optimise, holdout, diag


def _flatten_eval_metrics(
    eval_metrics: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for top_key, metrics in eval_metrics.items():
        if not isinstance(metrics, Mapping):
            continue
        key_prefix = f"{prefix}{top_key}"
        for metric_key in (
            "n_trades",
            "candidate_count",
            "avg_pnl_bankroll",
            "avg_pnl_sized",
            "avg_pnl_notional",
            "avg_gross_pnl_per_trade",
            "avg_gross_return_per_trade",
            "hit_rate",
            "pnl_positive_rate",
            "full_sl_exit_rate",
            "adverse_fast_exit_rate",
            "timeout_exit_rate",
            "worst_week",
            "max_dd",
            "w_sortino",
            "m_sortino",
        ):
            if metric_key in metrics:
                out[f"{key_prefix}_{metric_key}"] = metrics.get(metric_key)
    return out


def _load_parent_summary(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "strategy_id" not in df.columns:
        raise ValueError(f"Parent policy summary missing strategy_id: {path}")
    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        strategy_id = str(row.get("strategy_id"))
        if strategy_id and strategy_id != "nan":
            out[strategy_id] = row.to_dict()
    return out


def _params_from_parent_summary_row(row: Mapping[str, Any]) -> tuple[dict[str, Any], float]:
    params: dict[str, Any] = {}
    for key, value in row.items():
        key = str(key)
        if not key.startswith("param_"):
            continue
        param_key = key[len("param_") :]
        if value is None:
            continue
        if isinstance(value, float) and not np.isfinite(value):
            continue
        if isinstance(value, str) and value.lower() == "nan":
            continue
        if isinstance(value, np.generic):
            value = value.item()
        params[param_key] = value
    size_power = _safe_float(row.get("best_size_power"), 1.0)
    if not np.isfinite(size_power) or size_power <= 0.0:
        size_power = 1.0
    return params, float(size_power)


def _geometry_params_from_archetype_row(
    row: Mapping[str, Any],
    *,
    parent_params: Mapping[str, Any],
    parent_size_power: float,
) -> tuple[dict[str, Any], float]:
    params = dict(parent_params)
    raw_geometry = row.get("shrinkage_final_geometry")
    if isinstance(raw_geometry, str):
        try:
            raw_geometry = json.loads(raw_geometry)
        except Exception:
            raw_geometry = {}
    geometry = raw_geometry if isinstance(raw_geometry, Mapping) else {}
    for key, value in geometry.items():
        if key == "size_power":
            continue
        if isinstance(value, (int, float, bool, str)) or value is None:
            params[str(key)] = value
    size_power = _safe_float(geometry.get("size_power"), parent_size_power)
    return params, float(size_power)


def _evaluate_holdout_policies(
    holdout_bundles: list[Any],
    *,
    parent_params_by_strategy: Mapping[str, Any],
    parent_size_by_strategy: Mapping[str, float],
    archetype_df: pd.DataFrame,
    cost_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not holdout_bundles:
        return pd.DataFrame(), pd.DataFrame()
    parent_rows: list[dict[str, Any]] = []
    archetype_rows: list[dict[str, Any]] = []
    archetype_lookup: dict[tuple[str, str], Mapping[str, Any]] = {}
    if not archetype_df.empty:
        for _, row in archetype_df.iterrows():
            strategy = str(row.get("strategy_id", ""))
            archetype = str(row.get("policy_archetype", ""))
            if strategy and archetype:
                archetype_lookup[(strategy, archetype)] = row.to_dict()

    for bundle in holdout_bundles:
        strategy_id = str(bundle.strategy_id)
        parent_params = dict(parent_params_by_strategy.get(strategy_id) or bundle.base_params)
        parent_size = float(parent_size_by_strategy.get(strategy_id, bundle.best_size_power))
        parent_eval = _evaluate_policy_subsets(
            strategy_id,
            "holdout_side_parent",
            bundle.rows,
            bundle.paths,
            cost_pct=cost_pct,
            best_params=parent_params,
            best_size_power=parent_size,
            log_details=False,
        )
        parent_rows.append(
            {
                "strategy_id": strategy_id,
                "side": "short" if strategy_id.startswith("short") else "long",
                "holdout_rows": int(len(bundle.rows)),
                "holdout_symbols": int(bundle.rows["symbol"].astype(str).nunique())
                if "symbol" in bundle.rows
                else 0,
                **_flatten_eval_metrics(parent_eval),
            }
        )

        # Use the same canonical side/archetype key contract used during local
        # policy fitting.  Raw handoff values are intentionally prefixed by
        # their source column, so grouping the unnormalised holdout values
        # would silently fall back to the side-parent geometry.
        work = _attach_policy_archetype_column(
            bundle.rows.copy(), strategy_id=strategy_id
        )
        if "policy_archetype" not in work.columns:
            continue
        for archetype, idx in work.groupby("policy_archetype", sort=True).groups.items():
            idx_arr = np.asarray(list(idx), dtype=np.int64)
            if idx_arr.size < 5:
                continue
            sub_df = work.iloc[idx_arr].copy().reset_index(drop=True)
            sub_paths = tuple(arr[idx_arr] for arr in bundle.paths)
            arch_row = archetype_lookup.get((strategy_id, str(archetype)))
            if arch_row:
                params, size_power = _geometry_params_from_archetype_row(
                    arch_row,
                    parent_params=parent_params,
                    parent_size_power=parent_size,
                )
                policy_source = "side_archetype_shrunk_geometry"
            else:
                params, size_power = parent_params, parent_size
                policy_source = "side_parent_fallback"
            eval_metrics = _evaluate_policy_subsets(
                strategy_id,
                f"holdout_side_archetype_{archetype}",
                sub_df,
                sub_paths,
                cost_pct=cost_pct,
                best_params=params,
                best_size_power=float(size_power),
                log_details=False,
            )
            archetype_rows.append(
                {
                    "strategy_id": strategy_id,
                    "side": "short" if strategy_id.startswith("short") else "long",
                    "policy_archetype": str(archetype),
                    "policy_source": policy_source,
                    "holdout_rows": int(len(sub_df)),
                    "holdout_symbols": int(sub_df["symbol"].astype(str).nunique())
                    if "symbol" in sub_df
                    else 0,
                    **_flatten_eval_metrics(eval_metrics),
                }
            )
    return pd.DataFrame(parent_rows), pd.DataFrame(archetype_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--market-mode", choices=["spot", "perps"], default="perps")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-rank", type=float, default=0.70)
    parser.add_argument(
        "--rank-score-col",
        default="rank_pct",
        help="Candidate column to rank into rank_pct before min-rank filtering.",
    )
    parser.add_argument(
        "--rank-scope",
        choices=["per_strategy", "side", "global", "timestamp_side"],
        default="per_strategy",
        help="Scope used when --rank-score-col is not rank_pct.",
    )
    parser.add_argument(
        "--regime-ev-calibration-artifact",
        type=Path,
        default=default_regime_ev_calibration_artifact(),
        help="Frozen regime EV calibration JSON to apply before rank filtering.",
    )
    parser.add_argument(
        "--regime-ev-feature-handoff",
        type=Path,
        default=default_regime_ev_feature_handoff(),
        help="Optional handoff parquet used to join missing regime feature columns.",
    )
    parser.add_argument(
        "--disable-regime-ev-calibration",
        action="store_true",
        help=(
            "Do not apply the legacy regime-EV calibrator before geometry fitting. "
            "Use this when candidates already contain the canonical V9/MLP/EV-admission chain."
        ),
    )
    parser.add_argument("--path-len", type=int, default=96)
    parser.add_argument("--min-rows-per-strategy", type=int, default=100)
    parser.add_argument("--n-trials", type=int, default=24)
    parser.add_argument("--round-trip-cost-pct", type=float, default=0.003)
    parser.add_argument(
        "--holdout-start",
        default=None,
        help="UTC timestamp/date to exclude from optimisation and replay only after policy fit.",
    )
    parser.add_argument(
        "--holdout-end",
        default=None,
        help="UTC exclusive timestamp/date ending the replay-only holdout interval.",
    )
    parser.add_argument(
        "--parent-policy-summary",
        type=Path,
        default=None,
        help="Existing side_parent_policy_summary.csv to reuse parent geometry and skip parent HPO.",
    )
    parser.add_argument(
        "--allow-empty-archetype-policy",
        action="store_true",
        help="Write parent/fallback outputs even if no side x archetype policy rows are produced.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("EPM_SIMPLE_POLICY_PER_ARCHETYPE_OPTIMISATION", "1")
    os.environ.setdefault("EPM_SIMPLE_POLICY_PER_ARCHETYPE_FULL_OPTIMISATION", "1")
    os.environ.setdefault("EPM_SIMPLE_POLICY_STABLE_FOLD_OBJECTIVE", "1")
    os.environ.setdefault("EPM_SIMPLE_POLICY_STABLE_FOLD_OBJECTIVE_STD_WEIGHT", "0.5")
    os.environ.setdefault("EPM_SIMPLE_POLICY_STABLE_FOLD_OBJECTIVE_WORST_WEIGHT", "0.25")
    # Policy replay and production inference use 15-minute paths. Do not
    # trigger the legacy minute-bar backfill from this standalone runner.
    os.environ.setdefault("EPM_SIMPLE_POLICY_1M_DOWNLOAD", "0")
    os.environ.setdefault("EPM_SIMPLE_POLICY_15M_DOWNLOAD", "1")

    cost_pct = float(args.round_trip_cost_pct) / 2.0
    rows = _prepare_rows(
        args.candidates,
        min_rank=float(args.min_rank),
        rank_score_col=str(args.rank_score_col),
        rank_scope=str(args.rank_scope),
        regime_ev_calibration_artifact=args.regime_ev_calibration_artifact,
        regime_ev_feature_handoff=args.regime_ev_feature_handoff,
        apply_regime_ev_calibration_artifact=not bool(args.disable_regime_ev_calibration),
    )
    optimisation_rows, holdout_rows, holdout_diag = _split_optimisation_holdout_rows(
        rows,
        holdout_start=args.holdout_start,
        holdout_end=args.holdout_end,
    )
    cost_audit = _cost_accounting_audit(
        rows,
        round_trip_cost_pct=float(args.round_trip_cost_pct),
    )
    bundles = _load_bundles(
        optimisation_rows,
        data_root=str(args.data_root),
        market_mode=str(args.market_mode),
        path_len=int(args.path_len),
        min_rows_per_strategy=int(args.min_rows_per_strategy),
    )

    parent_rows: list[dict[str, Any]] = []
    archetype_frames: list[pd.DataFrame] = []
    subset_metrics: dict[str, Any] = {}
    parent_params_by_strategy: dict[str, Any] = {}
    parent_size_by_strategy: dict[str, float] = {}
    completed_strategy_ids: list[str] = []
    parent_summary_by_strategy = _load_parent_summary(args.parent_policy_summary)

    for bundle in bundles:
        loaded_parent_row = parent_summary_by_strategy.get(str(bundle.strategy_id))
        if loaded_parent_row is not None:
            best_params, best_size_power = _params_from_parent_summary_row(loaded_parent_row)
            best_metrics: dict[str, Any] = {}
            fit_summary: dict[str, Any] = {}
        else:
            best_params, best_size_power, best_metrics, fit_summary = _optimise_policy_on_rows(
                bundle.rows,
                bundle.paths,
                cost_pct=cost_pct,
                n_trials=int(args.n_trials),
            )
        parent_params_by_strategy[bundle.strategy_id] = dict(best_params)
        parent_size_by_strategy[bundle.strategy_id] = float(best_size_power)
        if loaded_parent_row is not None:
            stable_value = _safe_float(loaded_parent_row.get("stable_fold_objective"), np.nan)
            stable_diag = {
                "mean_score": loaded_parent_row.get("stable_fold_mean_score"),
                "std_score": loaded_parent_row.get("stable_fold_std_score"),
                "worst_score": loaded_parent_row.get("stable_fold_worst_score"),
                "objective_formula": loaded_parent_row.get("stable_fold_objective_formula"),
            }
        else:
            stable_value, stable_diag = _stable_fold_policy_objective(
                bundle.rows,
                bundle.paths,
                params=best_params,
                cost_pct=cost_pct,
                size_power=float(best_size_power),
                n_folds=DEFAULT_CV_FOLDS,
            )
        eval_metrics = _evaluate_policy_subsets(
            bundle.strategy_id,
            "side_parent",
            bundle.rows,
            bundle.paths,
            cost_pct=cost_pct,
            best_params=best_params,
            best_size_power=float(best_size_power),
            log_details=False,
        )
        subset_metrics[bundle.strategy_id] = eval_metrics
        rec = dict(loaded_parent_row) if loaded_parent_row is not None else {}
        rec.update({
            "strategy_id": bundle.strategy_id,
            "side": "short" if str(bundle.strategy_id).startswith("short") else "long",
            "rows": int(len(bundle.rows)),
            "symbols": int(bundle.rows["symbol"].astype(str).nunique()) if "symbol" in bundle.rows else 0,
            "round_trip_cost_pct": float(args.round_trip_cost_pct),
            "cost_pct_per_side": float(cost_pct),
            "n_trials": int(args.n_trials),
            "parent_policy_source": "loaded_summary" if loaded_parent_row is not None else "optimised",
            "best_size_power": float(best_size_power),
            "stable_fold_objective": float(stable_value),
            "stable_fold_mean_score": _safe_float(stable_diag.get("mean_score")),
            "stable_fold_std_score": _safe_float(stable_diag.get("std_score")),
            "stable_fold_worst_score": _safe_float(stable_diag.get("worst_score")),
            "stable_fold_objective_formula": stable_diag.get("objective_formula"),
            "best_metrics_net_pnl": _safe_float(best_metrics.get("net_pnl") if isinstance(best_metrics, dict) else np.nan),
            **{f"param_{k}": v for k, v in best_params.items() if isinstance(v, (int, float, str, bool))},
            **_extract_stage_fold_diag(dict(fit_summary or {})),
        })
        for top_key, metrics in eval_metrics.items():
            if not isinstance(metrics, dict):
                continue
            rec[f"{top_key}_n_trades"] = metrics.get("n_trades")
            rec[f"{top_key}_avg_pnl_bankroll"] = metrics.get("avg_pnl_bankroll")
            rec[f"{top_key}_avg_pnl_sized"] = metrics.get("avg_pnl_sized")
            rec[f"{top_key}_avg_pnl_notional"] = metrics.get(
                "avg_pnl_notional",
                metrics.get("avg_pnl_sized"),
            )
            rec[f"{top_key}_avg_gross_return_per_trade"] = metrics.get("avg_gross_return_per_trade")
            rec[f"{top_key}_hit_rate"] = metrics.get("hit_rate")
            rec[f"{top_key}_pnl_positive_rate"] = metrics.get("pnl_positive_rate")
            rec[f"{top_key}_full_sl_exit_rate"] = metrics.get("full_sl_exit_rate")
            rec[f"{top_key}_timeout_exit_rate"] = metrics.get("timeout_exit_rate")
            rec[f"{top_key}_worst_week"] = metrics.get("worst_week")
            rec[f"{top_key}_max_dd"] = metrics.get("max_dd")
        parent_rows.append(rec)

        arch_report, _arch_summary = _optimise_policy_by_archetype(
            strategy_id=bundle.strategy_id,
            df_top=bundle.rows,
            paths=bundle.paths,
            cost_pct=cost_pct,
            stage_a_cost_pct=cost_pct,
            n_trials=int(args.n_trials),
            market_mode=str(args.market_mode),
            parent_params=best_params,
            parent_size_power=float(best_size_power),
        )
        if not arch_report.empty:
            archetype_frames.append(arch_report)
        completed_strategy_ids.append(str(bundle.strategy_id))
        _write_policy_checkpoints(
            args.out_dir,
            parent_rows=parent_rows,
            archetype_frames=archetype_frames,
            completed_strategy_ids=completed_strategy_ids,
        )

    parent_df = pd.DataFrame(parent_rows)
    archetype_df = pd.concat(archetype_frames, ignore_index=True) if archetype_frames else pd.DataFrame()
    month_df = _month_week_metrics(rows, group_cols=["month", "side_name", "policy_archetype"])
    week_df = _month_week_metrics(rows, group_cols=["week", "side_name", "policy_archetype"])
    holdout_parent_df = pd.DataFrame()
    holdout_archetype_df = pd.DataFrame()
    holdout_month_df = pd.DataFrame()
    holdout_week_df = pd.DataFrame()
    if not holdout_rows.empty:
        try:
            holdout_bundles = _load_bundles(
                holdout_rows,
                data_root=str(args.data_root),
                market_mode=str(args.market_mode),
                path_len=int(args.path_len),
                min_rows_per_strategy=5,
            )
            holdout_parent_df, holdout_archetype_df = _evaluate_holdout_policies(
                holdout_bundles,
                parent_params_by_strategy=parent_params_by_strategy,
                parent_size_by_strategy=parent_size_by_strategy,
                archetype_df=archetype_df,
                cost_pct=cost_pct,
            )
            holdout_month_df = _month_week_metrics(
                holdout_rows,
                group_cols=["month", "side_name", "policy_archetype"],
            )
            holdout_week_df = _month_week_metrics(
                holdout_rows,
                group_cols=["week", "side_name", "policy_archetype"],
            )
            holdout_diag["evaluation_status"] = "ok"
            holdout_diag["holdout_parent_policy_rows"] = int(len(holdout_parent_df))
            holdout_diag["holdout_archetype_policy_rows"] = int(len(holdout_archetype_df))
        except Exception as exc:
            holdout_diag["evaluation_status"] = "error"
            holdout_diag["evaluation_error"] = f"{type(exc).__name__}: {exc}"

    parent_df.to_csv(args.out_dir / "side_parent_policy_summary.csv", index=False)
    archetype_df.to_csv(args.out_dir / "side_archetype_policy_summary.csv", index=False)
    month_df.to_csv(args.out_dir / "candidate_month_side_archetype_metrics.csv", index=False)
    week_df.to_csv(args.out_dir / "candidate_week_side_archetype_metrics.csv", index=False)
    if not holdout_parent_df.empty:
        holdout_parent_df.to_csv(args.out_dir / "holdout_side_parent_policy_metrics.csv", index=False)
    if not holdout_archetype_df.empty:
        holdout_archetype_df.to_csv(args.out_dir / "holdout_side_archetype_policy_metrics.csv", index=False)
    if not holdout_month_df.empty:
        holdout_month_df.to_csv(args.out_dir / "holdout_candidate_month_side_archetype_metrics.csv", index=False)
    if not holdout_week_df.empty:
        holdout_week_df.to_csv(args.out_dir / "holdout_candidate_week_side_archetype_metrics.csv", index=False)
    (args.out_dir / "side_parent_subset_metrics.json").write_text(
        json.dumps(_json_safe(subset_metrics), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest = {
        "generated_by": "run_s52_side_archetype_simple_policy_optimiser",
        "candidate_path": str(args.candidates),
        "parent_policy_summary": str(args.parent_policy_summary)
        if args.parent_policy_summary is not None
        else None,
        "data_root": str(args.data_root),
        "market_mode": str(args.market_mode),
        "min_rank": float(args.min_rank),
        "path_len": int(args.path_len),
        "n_trials": int(args.n_trials),
        "round_trip_cost_pct": float(args.round_trip_cost_pct),
        "legacy_regime_ev_calibration_applied": not bool(
            args.disable_regime_ev_calibration
        ),
        "cost_accounting_audit": cost_audit,
        "holdout": holdout_diag,
        "fold_objective": "mean_score - 0.5 * std_score + 0.25 * worst_score",
        "metrics_source": "non-training validation folds inside each Optuna trial; final subset metrics are replay diagnostics",
        "side_parent_rows": int(len(parent_df)),
        "side_archetype_rows": int(len(archetype_df)),
        "side_archetype_status": "ok" if len(archetype_df) > 0 else "empty",
        "outputs": {
            "side_parent_policy_summary": str(args.out_dir / "side_parent_policy_summary.csv"),
            "side_archetype_policy_summary": str(args.out_dir / "side_archetype_policy_summary.csv"),
            "candidate_month_side_archetype_metrics": str(args.out_dir / "candidate_month_side_archetype_metrics.csv"),
            "candidate_week_side_archetype_metrics": str(args.out_dir / "candidate_week_side_archetype_metrics.csv"),
            "holdout_side_parent_policy_metrics": str(args.out_dir / "holdout_side_parent_policy_metrics.csv"),
            "holdout_side_archetype_policy_metrics": str(args.out_dir / "holdout_side_archetype_policy_metrics.csv"),
            "holdout_candidate_month_side_archetype_metrics": str(args.out_dir / "holdout_candidate_month_side_archetype_metrics.csv"),
            "holdout_candidate_week_side_archetype_metrics": str(args.out_dir / "holdout_candidate_week_side_archetype_metrics.csv"),
            "side_parent_subset_metrics": str(args.out_dir / "side_parent_subset_metrics.json"),
        },
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe({"event": "s52_side_archetype_policy_done", **manifest}), sort_keys=True))
    if archetype_df.empty and not bool(args.allow_empty_archetype_policy):
        raise SystemExit(
            "side x archetype optimisation produced zero rows; "
            "rerun with --allow-empty-archetype-policy only for parent-fallback diagnostics"
        )


if __name__ == "__main__":
    main()
