#!/usr/bin/env python3
"""Portfolio-level ablation for simple-policy exit geometry changes.

The runner applies cumulative Optuna stages:

1. risk-time geometry envelope
2. decoupled capital-protection lock
3. time-decaying trailing activation

Each trial rebuilds strategy candidate rows from OHLCV paths, then replays the
whole portfolio so altered holding times can change later accepted assets.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    PortfolioPolicyParams,
    fit_hierarchical_ev_curves,
    replay_candidates,
)
from extreme_price_movements.simple_policy_optimiser import (  # noqa: E402
    DEFAULT_POLICY_PER_SIDE_COST_PCT,
    CAPITAL_PROTECT_LOCK_FRAC_GRID,
    CAPITAL_PROTECT_MIN_LOCK_BPS_GRID,
    GEOMETRY_SL_ABS_CAP_PCT_GRID,
    GEOMETRY_TRAILING_ACTIVATION_CAP_PCT_GRID,
    TRAILING_ACTIVATION_DECAY_HALF_LIFE_BARS_GRID,
    TRAILING_ACTIVATION_DECAY_START_BARS_GRID,
    TRAILING_ACTIVATION_MIN_MULT_GRID,
    _apply_delayed_entry_execution_model,
    _build_simple_policy_candidate_rows,
    _fetch_policy_paths,
    _json_safe,
    _make_policy_replay_store,
    _median_policy_barrier_frac,
    _path_take,
    _policy_path_finite_mask,
)


DEFAULT_CANDIDATES = Path(
    "data_perp/artifacts/20260629_050000_lgbm_mda/"
    "simple_policy_optimiser/simple_policy_candidates_broad.parquet"
)
DEFAULT_OUT_DIR = Path("data_perp/reports/simple_policy_exit_geometry_ablation")
DEFAULT_PATH_LEN = 96
ROW_LEVEL_POLICY_OVERRIDE_COLUMNS = {
    "portfolio_max_new_entries_per_bar",
    "portfolio_max_new_entries_per_strategy_per_bar",
    "portfolio_max_concurrent_per_strategy",
    "portfolio_wallet_cap_multiplier",
    "portfolio_size_multiplier",
    "portfolio_priority_multiplier",
    "portfolio_priority_adjustment",
    "portfolio_rank_adjustment",
    "portfolio_fixed_position_size",
}
LOCAL_POLICY_OVERRIDE_KEYS = {
    "best_size_power",
    "size_power",
    "base_strategy_threshold",
    "deployment_rank_threshold",
    *ROW_LEVEL_POLICY_OVERRIDE_COLUMNS,
}


@dataclass
class StrategyBundle:
    strategy_id: str
    rows: pd.DataFrame
    paths: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    base_params: Dict[str, Any]
    base_threshold: float
    best_size_power: float


def _side_code(value: Any) -> float:
    try:
        numeric = float(value)
        if np.isfinite(numeric):
            return -1.0 if numeric < 0.0 else 1.0
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower()
    if text in {"-1", "short", "sell"} or text.startswith("short"):
        return -1.0
    return 1.0


def _first_finite(rows: pd.DataFrame, column: str, default: float) -> float:
    if column not in rows.columns:
        return float(default)
    values = pd.to_numeric(rows[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    finite = values.dropna()
    if finite.empty:
        return float(default)
    return float(finite.iloc[0])


def _first_bool(rows: pd.DataFrame, column: str, default: bool) -> bool:
    if column not in rows.columns:
        return bool(default)
    value = rows[column].dropna()
    if value.empty:
        return bool(default)
    raw = value.iloc[0]
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(raw)


def _strategy_base_params(rows: pd.DataFrame) -> Dict[str, Any]:
    median_barrier = _first_finite(
        rows,
        "policy_median_barrier_frac",
        _median_policy_barrier_frac(rows),
    )
    return {
        "sl_mult": _first_finite(rows, "policy_sl_mult", 1.0),
        "sl_abs_cap_pct": _first_finite(rows, "policy_sl_abs_cap_pct", 0.0),
        "trailing_activation_mult": _first_finite(
            rows, "policy_trailing_activation_mult", 1.0
        ),
        "trailing_activation_cap_pct": _first_finite(
            rows, "policy_trailing_activation_cap_pct", 0.0
        ),
        "trailing_activation_decay_half_life_bars": _first_finite(
            rows, "policy_trailing_activation_decay_half_life_bars", 0.0
        ),
        "trailing_activation_decay_start_bars": int(
            _first_finite(rows, "policy_trailing_activation_decay_start_bars", 0.0)
        ),
        "trailing_activation_min_mult": _first_finite(
            rows, "policy_trailing_activation_min_mult", 1.0
        ),
        "trailing_power": _first_finite(rows, "policy_trailing_power", 1.5),
        "trailing_squash_divisor": _first_finite(
            rows, "policy_trailing_squash_divisor", 2.0
        ),
        "giveback_beta": _first_finite(rows, "policy_giveback_beta", 0.5),
        "capital_protect_mfe_mult": _first_finite(
            rows, "policy_capital_protect_mfe_mult", 0.0
        ),
        "capital_protect_regression_frac": _first_finite(
            rows, "policy_capital_protect_regression_frac", 0.45
        ),
        "capital_protect_lock_frac": (
            _first_finite(rows, "policy_capital_protect_lock_frac", np.nan)
        ),
        "capital_protect_min_lock_bps": _first_finite(
            rows, "policy_capital_protect_min_lock_bps", 0.0
        ),
        "atr_power": _first_finite(rows, "policy_atr_power", 1.0),
        "atr_multiplier": _first_finite(rows, "policy_atr_multiplier", 1.0),
        "hard_tp_abs_pct": _first_finite(rows, "policy_hard_tp_abs_pct", 0.0),
        "exit_pressure_enabled": _first_bool(rows, "policy_exit_pressure_enabled", False),
        "exit_pressure_alpha": _first_finite(rows, "policy_exit_pressure_alpha", 1.0),
        "exit_pressure_beta": _first_finite(rows, "policy_exit_pressure_beta", 0.0),
        "exit_pressure_delta": _first_finite(rows, "policy_exit_pressure_delta", 1.0),
        "exit_pressure_kappa": _first_finite(rows, "policy_exit_pressure_kappa", 0.0),
        "exit_pressure_psi": _first_finite(rows, "policy_exit_pressure_psi", 0.7),
        "exit_pressure_omega": _first_finite(rows, "policy_exit_pressure_omega", 1.0),
        "exit_pressure_min_multiplier": _first_finite(
            rows, "policy_exit_pressure_min_multiplier", 1.0
        ),
        "redeploy_scale_bps": _first_finite(rows, "policy_redeploy_scale_bps", 100.0),
        "target_holding_hours": _first_finite(rows, "policy_target_holding_hours", 0.0),
        "churn_penalty_bps": _first_finite(rows, "policy_churn_penalty_bps", 100.0),
        "median_barrier_frac": median_barrier,
        "policy_median_barrier_frac": median_barrier,
    }


def _prepare_rows(path: Path, *, min_rank: float) -> pd.DataFrame:
    rows = pd.read_parquet(path)
    required = {"timestamp", "symbol", "strategy_id", "rank_pct", "barrier_pct"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    rows = rows.copy()
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
    rows = rows.dropna(subset=["timestamp", "symbol", "strategy_id"]).copy()
    rows["rank_pct"] = pd.to_numeric(rows["rank_pct"], errors="coerce")
    rows = rows.loc[rows["rank_pct"].ge(float(min_rank))].copy()
    rows["side"] = [_side_code(v) for v in rows.get("side", 1.0)]
    rows["symbol"] = rows["symbol"].astype(str)
    rows["strategy_id"] = rows["strategy_id"].astype(str)
    rows = rows.sort_values(["strategy_id", "timestamp", "symbol"]).reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"No rows left after rank_pct >= {min_rank}")
    return rows


def _load_bundles(
    rows: pd.DataFrame,
    *,
    data_root: str,
    market_mode: str,
    path_len: int,
    min_rows_per_strategy: int,
) -> List[StrategyBundle]:
    store = _make_policy_replay_store(data_root, market_mode)
    bundles: List[StrategyBundle] = []
    for strategy_id, group in rows.groupby("strategy_id", sort=True):
        group = group.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        if len(group) < int(min_rows_per_strategy):
            continue
        paths = _fetch_policy_paths(group, store, path_len=int(path_len))
        group, paths = _apply_delayed_entry_execution_model(
            group,
            paths,
            data_root=data_root,
            market_mode=market_mode,
        )
        finite = _policy_path_finite_mask(paths)
        group = group.loc[finite].reset_index(drop=True)
        paths = _path_take(paths, np.flatnonzero(finite))
        if len(group) < int(min_rows_per_strategy):
            continue
        bundles.append(
            StrategyBundle(
                strategy_id=str(strategy_id),
                rows=group,
                paths=paths,
                base_params=_strategy_base_params(group),
                base_threshold=_first_finite(group, "base_strategy_threshold", 0.70),
                best_size_power=_first_finite(group, "best_size_power", 1.0),
            )
        )
    if not bundles:
        raise ValueError("No strategy bundles available after path loading")
    return bundles


def _clean_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(params)
    try:
        lock_frac = float(out.get("capital_protect_lock_frac", np.nan))
    except (TypeError, ValueError):
        lock_frac = np.nan
    if not np.isfinite(lock_frac):
        out["capital_protect_lock_frac"] = None
    return out


def _candidate_table_for_overrides(
    bundles: Iterable[StrategyBundle],
    *,
    overrides: Mapping[str, Any],
    cost_pct: float,
    market_mode: str,
    arm: str,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for bundle in bundles:
        overrides = dict(overrides)
        simulator_overrides = {
            key: value
            for key, value in overrides.items()
            if key not in LOCAL_POLICY_OVERRIDE_KEYS
        }
        params = _clean_params({**bundle.base_params, **simulator_overrides})
        size_power = float(
            overrides.get(
                "best_size_power",
                overrides.get("size_power", bundle.best_size_power),
            )
        )
        base_threshold = float(
            overrides.get(
                "base_strategy_threshold",
                overrides.get("deployment_rank_threshold", bundle.base_threshold),
            )
        )
        frame = _build_simple_policy_candidate_rows(
            strategy_id=bundle.strategy_id,
            df_top=bundle.rows.copy(),
            paths=bundle.paths,
            cost_pct=float(cost_pct),
            best_params=params,
            best_size_power=size_power,
            base_strategy_threshold=base_threshold,
            market_mode=market_mode,
        )
        if frame.empty:
            continue
        for col in sorted(ROW_LEVEL_POLICY_OVERRIDE_COLUMNS):
            if col in overrides:
                frame[col] = overrides[col]
        frame["policy_override_best_size_power"] = size_power
        frame["policy_override_base_strategy_threshold"] = base_threshold
        frame["exit_geometry_ablation_arm"] = str(arm)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0].reset_index(drop=True)
    return (
        pd.concat(frames, ignore_index=True, copy=False)
        .sort_values(["timestamp", "strategy_id", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )


def _accepted_rows(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty or "accepted" not in decisions.columns:
        return pd.DataFrame()
    return decisions.loc[decisions["accepted"].astype(bool)].copy()


def _score_replay(
    candidates: pd.DataFrame,
    *,
    market_mode: str,
    global_threshold_floor: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame(), {"objective": float("-inf"), "trade_count": 0}
    ev_curve = fit_hierarchical_ev_curves(candidates)
    params = PortfolioPolicyParams(global_threshold_floor=float(global_threshold_floor))
    decisions, equity, metrics = replay_candidates(
        candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=market_mode,
    )
    return decisions, equity, dict(metrics)


def _metrics_row(
    *,
    arm: str,
    stage: str,
    overrides: Mapping[str, Any],
    candidates: pd.DataFrame,
    decisions: pd.DataFrame,
    metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    accepted = _accepted_rows(decisions)
    if not accepted.empty and {"timestamp", "position_exit_timestamp"}.issubset(accepted.columns):
        start = pd.to_datetime(accepted["timestamp"], utc=True, errors="coerce")
        end = pd.to_datetime(accepted["position_exit_timestamp"], utc=True, errors="coerce")
        hold_hours = (end - start).dt.total_seconds() / 3600.0
    else:
        hold_hours = pd.Series(dtype=float)
    row: Dict[str, Any] = {
        "arm": arm,
        "stage": stage,
        "candidate_rows": int(len(candidates)),
        "accepted_trades": int(len(accepted)),
        "avg_accepted_holding_hours": float(hold_hours.mean()) if len(hold_hours) else 0.0,
        "p75_accepted_holding_hours": float(hold_hours.quantile(0.75)) if len(hold_hours) else 0.0,
    }
    for key, value in metrics.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            row[f"portfolio_{key}"] = value
    for key, value in overrides.items():
        row[f"param_{key}"] = value
    return row


def _suggest_geometry(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "sl_abs_cap_pct": trial.suggest_categorical(
            "sl_abs_cap_pct", list(GEOMETRY_SL_ABS_CAP_PCT_GRID)
        ),
        "trailing_activation_cap_pct": trial.suggest_categorical(
            "trailing_activation_cap_pct",
            list(GEOMETRY_TRAILING_ACTIVATION_CAP_PCT_GRID),
        ),
        "hard_tp_abs_pct": trial.suggest_categorical(
            "hard_tp_abs_pct", [0.0, 0.006, 0.008, 0.010, 0.0125, 0.015, 0.020]
        ),
    }


def _suggest_capital_lock(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "capital_protect_mfe_mult": trial.suggest_categorical(
            "capital_protect_mfe_mult", [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        ),
        "capital_protect_lock_frac": trial.suggest_categorical(
            "capital_protect_lock_frac", list(CAPITAL_PROTECT_LOCK_FRAC_GRID)
        ),
        "capital_protect_min_lock_bps": trial.suggest_categorical(
            "capital_protect_min_lock_bps", list(CAPITAL_PROTECT_MIN_LOCK_BPS_GRID)
        ),
    }


def _suggest_time_decay(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "trailing_activation_decay_half_life_bars": trial.suggest_categorical(
            "trailing_activation_decay_half_life_bars",
            list(TRAILING_ACTIVATION_DECAY_HALF_LIFE_BARS_GRID),
        ),
        "trailing_activation_decay_start_bars": trial.suggest_categorical(
            "trailing_activation_decay_start_bars",
            list(TRAILING_ACTIVATION_DECAY_START_BARS_GRID),
        ),
        "trailing_activation_min_mult": trial.suggest_categorical(
            "trailing_activation_min_mult", list(TRAILING_ACTIVATION_MIN_MULT_GRID)
        ),
    }


def _optimise_stage(
    *,
    stage: str,
    base_overrides: Mapping[str, Any],
    suggest: Callable[[optuna.Trial], Dict[str, Any]],
    bundles: List[StrategyBundle],
    n_trials: int,
    seed: int,
    cost_pct: float,
    market_mode: str,
    global_threshold_floor: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    sampler = optuna.samplers.TPESampler(seed=int(seed))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    trial_rows: List[Dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        trial_overrides = {**dict(base_overrides), **suggest(trial)}
        candidates = _candidate_table_for_overrides(
            bundles,
            overrides=trial_overrides,
            cost_pct=cost_pct,
            market_mode=market_mode,
            arm=f"{stage}_trial_{trial.number}",
        )
        decisions, _equity, metrics = _score_replay(
            candidates,
            market_mode=market_mode,
            global_threshold_floor=global_threshold_floor,
        )
        value = float(metrics.get("objective", -np.inf))
        row = _metrics_row(
            arm=f"{stage}_trial_{trial.number}",
            stage=stage,
            overrides=trial_overrides,
            candidates=candidates,
            decisions=decisions,
            metrics=metrics,
        )
        row["trial_number"] = int(trial.number)
        row["objective"] = value
        trial_rows.append(row)
        trial.set_user_attr("metrics", row)
        return value if np.isfinite(value) else -1.0e12

    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    best = study.best_trial
    best_stage_params = {k: v for k, v in best.params.items()}
    return {**dict(base_overrides), **best_stage_params}, trial_rows


def _write_arm_outputs(
    *,
    out_dir: Path,
    arm: str,
    stage: str,
    overrides: Mapping[str, Any],
    candidates: pd.DataFrame,
    decisions: pd.DataFrame,
    equity: pd.DataFrame,
    metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    arm_dir = out_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(arm_dir / "candidates.parquet", index=False)
    decisions.to_parquet(arm_dir / "decisions.parquet", index=False)
    equity.to_parquet(arm_dir / "equity.parquet", index=False)
    row = _metrics_row(
        arm=arm,
        stage=stage,
        overrides=overrides,
        candidates=candidates,
        decisions=decisions,
        metrics=metrics,
    )
    (arm_dir / "metrics.json").write_text(json.dumps(_json_safe(row), indent=2))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--market-mode", default="perps", choices=["spot", "perps"])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-rank", type=float, default=0.70)
    parser.add_argument("--path-len", type=int, default=DEFAULT_PATH_LEN)
    parser.add_argument("--min-rows-per-strategy", type=int, default=50)
    parser.add_argument("--strategy-ids", default="")
    parser.add_argument("--n-trials", type=int, default=8)
    parser.add_argument("--seed", type=int, default=104729)
    parser.add_argument("--global-threshold-floor", type=float, default=0.0)
    parser.add_argument("--cost-pct", type=float, default=DEFAULT_POLICY_PER_SIDE_COST_PCT)
    parser.add_argument(
        "--enable-geometry-overrides",
        action="store_true",
        help=(
            "Opt in to the experimental exit-geometry override stages. "
            "By default this script writes the baseline replay only."
        ),
    )
    parser.add_argument(
        "--download-missing-1m",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Opt in to downloading/materializing missing 1m delayed-entry execution "
            "candles. Also enables missing 15m chart fallback for true 15m replay."
        ),
    )
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["EPM_SIMPLE_POLICY_1M_DOWNLOAD"] = (
        "1" if bool(args.download_missing_1m) else "0"
    )
    os.environ["EPM_SIMPLE_POLICY_15M_DOWNLOAD"] = (
        "1" if bool(args.download_missing_1m) else "0"
    )

    rows = _prepare_rows(args.candidates, min_rank=float(args.min_rank))
    if args.strategy_ids.strip():
        allowed = {s.strip() for s in args.strategy_ids.split(",") if s.strip()}
        rows = rows.loc[rows["strategy_id"].isin(allowed)].copy()
    bundles = _load_bundles(
        rows,
        data_root=str(args.data_root),
        market_mode=str(args.market_mode),
        path_len=int(args.path_len),
        min_rows_per_strategy=int(args.min_rows_per_strategy),
    )

    summary_rows: List[Dict[str, Any]] = []
    trial_rows: List[Dict[str, Any]] = []
    stage_overrides: Dict[str, Any] = {}

    stage_plan: List[Tuple[str, str, Optional[Callable[[optuna.Trial], Dict[str, Any]]]]] = [
        ("A0_baseline", "baseline", None),
    ]
    if bool(args.enable_geometry_overrides):
        stage_plan.extend(
            [
                ("A1_geometry_envelope", "geometry_envelope", _suggest_geometry),
                ("A2_capital_lock", "capital_lock", _suggest_capital_lock),
                ("A3_time_decay", "time_decay", _suggest_time_decay),
            ]
        )

    manifest: Dict[str, Any] = {
        "generated_by": "ablate_simple_policy_exit_geometry",
        "candidate_path": str(args.candidates),
        "data_root": str(args.data_root),
        "market_mode": str(args.market_mode),
        "min_rank": float(args.min_rank),
        "path_len": int(args.path_len),
        "n_trials_per_stage": int(args.n_trials),
        "geometry_overrides_enabled": bool(args.enable_geometry_overrides),
        "download_missing_1m": bool(args.download_missing_1m),
        "global_threshold_floor": float(args.global_threshold_floor),
        "cost_pct": float(args.cost_pct),
        "strategy_count": int(len(bundles)),
        "strategies": [
            {
                "strategy_id": b.strategy_id,
                "rows": int(len(b.rows)),
                "base_threshold": float(b.base_threshold),
            }
            for b in bundles
        ],
        "arms": [],
    }

    for arm, stage, suggest in stage_plan:
        if suggest is not None:
            stage_overrides, rows_for_stage = _optimise_stage(
                stage=stage,
                base_overrides=stage_overrides,
                suggest=suggest,
                bundles=bundles,
                n_trials=int(args.n_trials),
                seed=int(args.seed) + len(trial_rows) * 17 + len(summary_rows) * 101,
                cost_pct=float(args.cost_pct),
                market_mode=str(args.market_mode),
                global_threshold_floor=float(args.global_threshold_floor),
            )
            trial_rows.extend(rows_for_stage)

        candidates = _candidate_table_for_overrides(
            bundles,
            overrides=stage_overrides,
            cost_pct=float(args.cost_pct),
            market_mode=str(args.market_mode),
            arm=arm,
        )
        decisions, equity, metrics = _score_replay(
            candidates,
            market_mode=str(args.market_mode),
            global_threshold_floor=float(args.global_threshold_floor),
        )
        row = _write_arm_outputs(
            out_dir=args.out_dir,
            arm=arm,
            stage=stage,
            overrides=stage_overrides,
            candidates=candidates,
            decisions=decisions,
            equity=equity,
            metrics=metrics,
        )
        summary_rows.append(row)
        manifest["arms"].append(row)

    summary = pd.DataFrame(summary_rows)
    trials = pd.DataFrame(trial_rows)
    summary.to_csv(args.out_dir / "portfolio_ablation_summary.csv", index=False)
    trials.to_csv(args.out_dir / "portfolio_ablation_trials.csv", index=False)
    (args.out_dir / "ablation_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2)
    )


if __name__ == "__main__":
    main()
