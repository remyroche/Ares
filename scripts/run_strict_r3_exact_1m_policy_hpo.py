#!/usr/bin/env python3
"""Sequential, exact-one-minute HPO for the live parent policy.

The runner consumes only an immutable dataset made by
``materialize_strict_r3_exact_1m_policy_hpo_dataset.py``.  It is offline
research: it cannot load a live state, contact Kraken, amend a policy bundle,
or submit orders.

All selection is limited to 2024.  The sequential funnel deliberately keeps
the search interpretable:

The default ``live_parent`` surface is intentionally limited to the three
parameters that the deployed minute state machine actually consumes:

1. stop ATR;
2. trailing-activation ATR; and
3. trailing-giveback ATR.

It still performs a sequential broad → refine → polish funnel over all three
deployable parameters.  ``rich_research`` retains the additional experimental
stages (time controls, geometry transforms, and protection exits), but those
results are explicitly non-deployable until a separately validated live-state
implementation exists.

Finalists are replayed through the normal portfolio auction using the exact
one-minute exit timestamps, never ``holding_bars * 15 minutes``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.exact_1m_policy_contract import (  # noqa: E402
    EXACT_1M_POLICY_SCHEMA,
    Exact1mExecutionContract,
    Exact1mPolicyParams,
    exact_policy_monthly_metrics,
    exact_policy_objective,
    fit_adverse_theta_exact_1m,
    simulate_exact_1m_parent_policy,
)
from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    normalise_candidate_table,
    replay_candidates,
)
from scripts.report_strict_r3_mc1_d2_controlled_portfolio import (  # noqa: E402
    CAUSAL_AUCTION_CURVE,
    _params as canonical_portfolio_params,
)


DEFAULT_DATASET = ROOT / "data_perp/artifacts/strict_r3_exact_1m_policy_hpo_dataset_202402_20260817_v1"
DEFAULT_OUT = ROOT / "data_perp/artifacts/strict_r3_exact_1m_policy_hpo_long_20260817_v1"
SEED = 20260817
HPO_YEAR = 2024
HPO_START = pd.Timestamp("2024-02-01", tz="UTC")
CALIBRATION_MONTHS = pd.period_range("2024-02", "2024-06", freq="M")
SELECTION_MONTHS = pd.period_range("2024-07", "2024-12", freq="M")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_dataset(
    root: Path, *, side: str
) -> tuple[pd.DataFrame, dict[str, np.ndarray], Exact1mExecutionContract, dict[str, Any]]:
    manifest_path = root / "dataset_manifest.json"
    rows_path = root / "training_rows.parquet"
    paths_path = root / "exact_paths.npz"
    if not (manifest_path.exists() and rows_path.exists() and paths_path.exists()):
        raise FileNotFoundError("exact-1m HPO dataset requires manifest, training_rows, and exact_paths")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "strict_r3_exact_1m_policy_hpo_dataset_v1":
        raise ValueError("unrecognised exact-1m HPO dataset schema")
    contract = Exact1mExecutionContract(**dict(manifest.get("contract") or {}))
    contract.validate()
    if str(manifest.get("contract_hash")) != contract.hash:
        raise AssertionError("dataset execution contract hash mismatch")
    rows = pd.read_parquet(rows_path)
    required = {"candidate_id", "timestamp", "symbol", "score", "entry_ts", "entry_price", "signal_atr", "path_valid"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"exact-1m HPO training rows miss columns: {missing}")
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True, errors="raise")
    rows["entry_ts"] = pd.to_datetime(rows["entry_ts"], utc=True, errors="raise")
    if not rows.get("side_name", pd.Series(dtype=str)).astype(str).str.lower().eq(side).all():
        raise AssertionError("exact-1m HPO dataset is not side-local to the requested run")
    manifest_side = str(manifest.get("side") or "long").strip().lower()
    if manifest_side != side:
        raise AssertionError("exact-1m HPO dataset manifest side mismatch")
    if not rows["path_valid"].fillna(False).astype(bool).all():
        raise AssertionError("training_rows must contain valid complete paths only")
    raw = np.load(paths_path, allow_pickle=False)
    paths = {key: np.asarray(raw[key]) for key in ("entry", "atr", "high", "low", "close")}
    ids = np.asarray(raw["candidate_id"]).astype(str)
    if len(rows) != len(ids) or not np.array_equal(rows["candidate_id"].astype(str).to_numpy(), ids):
        raise AssertionError("HPO path identity order does not match training rows")
    if any(len(value) != len(rows) for value in paths.values()):
        raise AssertionError("HPO path arrays do not match training rows")
    if paths["high"].shape[1] != contract.horizon_minutes:
        raise AssertionError("HPO paths must contain exactly 720 minutes")
    return rows, paths, contract, manifest


def _assert_dataset_gates(
    rows: pd.DataFrame,
    paths: Mapping[str, np.ndarray],
    contract: Exact1mExecutionContract,
    manifest: Mapping[str, Any],
    *,
    min_monthly_support: int,
    min_path_coverage: float,
) -> dict[str, Any]:
    """Fail closed on a weak or semantically mismatched HPO substrate."""
    if rows["candidate_id"].astype(str).duplicated().any():
        raise AssertionError("exact-1m HPO training rows contain duplicate candidate IDs")
    expected_entry = rows["timestamp"] + pd.Timedelta(minutes=contract.entry_delay_minutes)
    entry_ns = rows["entry_ts"].astype("int64").to_numpy()
    expected_entry_ns = expected_entry.astype("int64").to_numpy()
    if not np.array_equal(entry_ns, expected_entry_ns):
        raise AssertionError("all HPO rows must use the one uniform decision+delay entry convention")
    if not rows["timestamp"].dt.year.eq(HPO_YEAR).all() or not rows["timestamp"].ge(HPO_START).all():
        raise AssertionError("exact-1m policy HPO may consume only the compatible Feb-Dec 2024 rows")
    for field in ("entry", "atr", "high", "low", "close"):
        value = np.asarray(paths[field])
        if not np.isfinite(value).all():
            raise AssertionError(f"exact-1m HPO paths contain non-finite {field}")
    candidate_rows = int(manifest.get("candidate_rows", len(rows)))
    valid_rows = int(manifest.get("valid_training_rows", len(rows)))
    coverage = float(valid_rows / max(candidate_rows, 1))
    if coverage < float(min_path_coverage):
        raise RuntimeError(
            f"exact-1m HPO path coverage {coverage:.2%} is below required {min_path_coverage:.2%}"
        )
    months = rows["timestamp"].dt.tz_localize(None).dt.to_period("M")
    counts = months.value_counts().sort_index()
    required = list(CALIBRATION_MONTHS) + list(SELECTION_MONTHS)
    insufficient = {
        str(month): int(counts.get(month, 0))
        for month in required
        if int(counts.get(month, 0)) < int(min_monthly_support)
    }
    if insufficient:
        raise RuntimeError(
            "exact-1m policy HPO lacks balanced 2024 monthly support: "
            f"required >= {min_monthly_support}, observed {insufficient}"
        )
    return {
        "candidate_rows": candidate_rows,
        "valid_training_rows": valid_rows,
        "path_coverage": coverage,
        "monthly_rows": {str(month): int(counts.get(month, 0)) for month in required},
        "entry_delay_minutes": int(contract.entry_delay_minutes),
    }


def _take(paths: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    indices = np.flatnonzero(np.asarray(mask, dtype=bool))
    return {key: np.asarray(value)[indices] for key, value in paths.items()}


def _stable_month_sample(
    rows: pd.DataFrame,
    paths: Mapping[str, np.ndarray],
    *,
    rows_per_month: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, int]]:
    """Take a reproducible, outcome-free equal-month HPO sample.

    Full paths remain reserved for finalist portfolio selection. The funnel
    needs only a stable search substrate; candidate-ID selection is independent
    of path outcomes, exit geometry, and future returns.
    """
    if int(rows_per_month) <= 0:
        months = rows["timestamp"].dt.tz_localize(None).dt.to_period("M").astype(str)
        return rows.reset_index(drop=True), {key: np.asarray(value) for key, value in paths.items()}, {
            str(month): int(size) for month, size in months.value_counts().sort_index().items()
        }
    months = rows["timestamp"].dt.tz_localize(None).dt.to_period("M").astype(str)
    chosen: list[np.ndarray] = []
    counts: dict[str, int] = {}
    for month in sorted(months.unique()):
        positions = np.flatnonzero(months.to_numpy() == month)
        if len(positions) > int(rows_per_month):
            hashes = pd.util.hash_pandas_object(
                rows.iloc[positions]["candidate_id"].astype(str), index=False
            ).to_numpy(np.uint64)
            positions = positions[np.argsort(hashes, kind="stable")[: int(rows_per_month)]]
        chosen.append(positions)
        counts[str(month)] = int(len(positions))
    indices = np.sort(np.concatenate(chosen)) if chosen else np.empty(0, dtype=np.int64)
    mask = np.zeros(len(rows), dtype=bool)
    mask[indices] = True
    return rows.iloc[indices].reset_index(drop=True), _take(paths, mask), counts


def _median_atr_fraction(paths: Mapping[str, np.ndarray]) -> float:
    ratio = np.asarray(paths["atr"], float) / np.maximum(np.asarray(paths["entry"], float), 1e-12)
    ratio = ratio[np.isfinite(ratio) & (ratio > 0.0)]
    if not len(ratio):
        raise ValueError("no finite ATR fractions")
    return float(np.median(ratio))


def _with_theta(
    params: Exact1mPolicyParams,
    calibration_paths: Mapping[str, np.ndarray],
    *,
    median_atr: float,
    side: str,
) -> Exact1mPolicyParams:
    if not params.adverse_exit_enabled:
        return params
    theta = fit_adverse_theta_exact_1m(
        entry=np.asarray(calibration_paths["entry"]), atr=np.asarray(calibration_paths["atr"]),
        highs=np.asarray(calibration_paths["high"]), lows=np.asarray(calibration_paths["low"]),
        params=params, median_atr_fraction=median_atr, side=side,
    )
    return Exact1mPolicyParams.from_mapping({**params.to_dict(), "adverse_exit_theta": theta})


def _evaluate(
    rows: pd.DataFrame,
    paths: Mapping[str, np.ndarray],
    calibration_paths: Mapping[str, np.ndarray],
    *,
    params: Exact1mPolicyParams,
    contract: Exact1mExecutionContract,
    median_atr: float,
    side: str,
) -> tuple[float, dict[str, np.ndarray], Exact1mPolicyParams, pd.DataFrame]:
    fitted = _with_theta(params, calibration_paths, median_atr=median_atr, side=side)
    replay = simulate_exact_1m_parent_policy(
        entry=np.asarray(paths["entry"]), atr=np.asarray(paths["atr"]),
        highs=np.asarray(paths["high"]), lows=np.asarray(paths["low"]), closes=np.asarray(paths["close"]),
        entry_timestamps=rows["entry_ts"], params=fitted, contract=contract,
        median_atr_fraction=median_atr, side=side,
    )
    score = exact_policy_objective(rows, replay, params=fitted)
    monthly = exact_policy_monthly_metrics(rows, replay)
    return score, replay, fitted, monthly


def _core(trial: optuna.Trial, parent: Exact1mPolicyParams) -> Exact1mPolicyParams:
    return Exact1mPolicyParams.from_mapping({
        **parent.to_dict(),
        "sl_mult": trial.suggest_float("sl_mult", 1.0, 8.0),
        "trailing_activation_mult": trial.suggest_float("trailing_activation_mult", 0.25, 5.0),
        "fixed_trailing_gap_mult": trial.suggest_float("fixed_trailing_gap_mult", 0.02, 0.75),
    })


def _bounded_product(
    trial: optuna.Trial,
    *,
    name: str,
    parent: float,
    low_scale: float,
    high_scale: float,
    lower: float,
    upper: float,
) -> float:
    value = float(parent) * trial.suggest_float(name, low_scale, high_scale)
    return float(np.clip(value, lower, upper))


def _core_refine(trial: optuna.Trial, parent: Exact1mPolicyParams) -> Exact1mPolicyParams:
    """Second funnel pass, local to a broad live-parent candidate."""
    return Exact1mPolicyParams.from_mapping({
        **parent.to_dict(),
        "sl_mult": _bounded_product(
            trial, name="sl_scale", parent=parent.sl_mult,
            low_scale=0.65, high_scale=1.35, lower=0.75, upper=10.0,
        ),
        "trailing_activation_mult": _bounded_product(
            trial, name="activation_scale", parent=parent.trailing_activation_mult,
            low_scale=0.65, high_scale=1.35, lower=0.10, upper=7.0,
        ),
        "fixed_trailing_gap_mult": _bounded_product(
            trial, name="giveback_scale", parent=parent.fixed_trailing_gap_mult,
            low_scale=0.50, high_scale=1.50, lower=0.01, upper=1.0,
        ),
    })


def _core_polish(trial: optuna.Trial, parent: Exact1mPolicyParams) -> Exact1mPolicyParams:
    """Final narrow pass over the complete, deployable parent surface."""
    return Exact1mPolicyParams.from_mapping({
        **parent.to_dict(),
        "sl_mult": _bounded_product(
            trial, name="sl_scale", parent=parent.sl_mult,
            low_scale=0.85, high_scale=1.15, lower=0.75, upper=10.0,
        ),
        "trailing_activation_mult": _bounded_product(
            trial, name="activation_scale", parent=parent.trailing_activation_mult,
            low_scale=0.85, high_scale=1.15, lower=0.10, upper=7.0,
        ),
        "fixed_trailing_gap_mult": _bounded_product(
            trial, name="giveback_scale", parent=parent.fixed_trailing_gap_mult,
            low_scale=0.75, high_scale=1.25, lower=0.01, upper=1.0,
        ),
    })


def _time_controls(trial: optuna.Trial, parent: Exact1mPolicyParams) -> Exact1mPolicyParams:
    return Exact1mPolicyParams.from_mapping({
        **parent.to_dict(),
        "trailing_activation_decay_half_life_minutes": trial.suggest_categorical("activation_half_life_minutes", [0.0, 15.0, 30.0, 60.0, 120.0, 240.0]),
        "trailing_activation_decay_start_minutes": trial.suggest_categorical("activation_decay_start_minutes", [0, 15, 30, 60, 120]),
        "trailing_activation_min_mult": trial.suggest_categorical("activation_min_mult", [0.35, 0.50, 0.70, 0.85, 1.0]),
        "holding_rent_bps_per_hour": trial.suggest_categorical("holding_rent_bps_per_hour", [0.0, 0.10, 0.25, 0.50]),
        "risk_time_weight": trial.suggest_categorical("risk_time_weight", [0.0, 0.025, 0.05, 0.10]),
    })


def _geometry(trial: optuna.Trial, parent: Exact1mPolicyParams) -> Exact1mPolicyParams:
    mode = trial.suggest_categorical("trailing_mode", ["fixed", "dynamic"])
    values: dict[str, Any] = {
        **parent.to_dict(),
        "sl_abs_floor_pct": trial.suggest_categorical("sl_abs_floor_pct", [0.0, 0.004, 0.006, 0.010, 0.015]),
        "sl_abs_cap_pct": trial.suggest_categorical("sl_abs_cap_pct", [0.0, 0.010, 0.015, 0.020, 0.030, 0.050]),
        "trailing_activation_min_pct": trial.suggest_categorical("activation_min_pct", [0.0, 0.003, 0.005, 0.0075, 0.010]),
        "trailing_activation_cap_pct": trial.suggest_categorical("activation_cap_pct", [0.0, 0.010, 0.015, 0.020, 0.030, 0.050]),
        "sl_atr_power": trial.suggest_categorical("sl_atr_power", [0.70, 0.85, 1.0, 1.15, 1.30]),
        "sl_atr_multiplier": trial.suggest_categorical("sl_atr_multiplier", [0.75, 1.0, 1.25]),
        "tp_atr_power": trial.suggest_categorical("tp_atr_power", [0.70, 0.85, 1.0, 1.15, 1.30]),
        "tp_atr_multiplier": trial.suggest_categorical("tp_atr_multiplier", [0.75, 1.0, 1.25]),
    }
    if values["sl_abs_cap_pct"] and values["sl_abs_floor_pct"] > values["sl_abs_cap_pct"]:
        values["sl_abs_cap_pct"] = values["sl_abs_floor_pct"]
    if values["trailing_activation_cap_pct"] and values["trailing_activation_min_pct"] > values["trailing_activation_cap_pct"]:
        values["trailing_activation_cap_pct"] = values["trailing_activation_min_pct"]
    if mode == "fixed":
        values["fixed_trailing_gap_mult"] = trial.suggest_float("fixed_trailing_gap_mult", 0.02, 0.60)
    else:
        values.update(
            fixed_trailing_gap_mult=0.0,
            trailing_power=trial.suggest_float("trailing_power", 0.60, 3.0),
            trailing_squash_divisor=trial.suggest_float("trailing_squash_divisor", 0.50, 5.0),
            giveback_beta=trial.suggest_float("giveback_beta", 0.10, 0.90),
        )
    return Exact1mPolicyParams.from_mapping(values)


def _protection(trial: optuna.Trial, parent: Exact1mPolicyParams) -> Exact1mPolicyParams:
    return Exact1mPolicyParams.from_mapping({
        **parent.to_dict(),
        "capital_protect_mfe_mult": trial.suggest_categorical("capital_protect_mfe_mult", [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]),
        "capital_protect_regression_frac": trial.suggest_float("capital_protect_regression_frac", 0.10, 0.80),
        "capital_protect_lock_frac": trial.suggest_categorical("capital_protect_lock_frac", [None, 0.0, 0.20, 0.40, 0.60, 0.80]),
        "capital_protect_min_lock_bps": trial.suggest_categorical("capital_protect_min_lock_bps", [0.0, 10.0, 25.0, 50.0, 75.0]),
        "adverse_exit_enabled": trial.suggest_categorical("adverse_exit_enabled", [False, True]),
        "adverse_exit_min_mae_atr": trial.suggest_categorical("adverse_exit_min_mae_atr", [0.50, 0.75, 1.0, 1.25, 1.50]),
        "adverse_exit_min_speed_per_hour": trial.suggest_categorical("adverse_exit_min_speed_per_hour", [0.10, 0.20, 0.30, 0.45, 0.60]),
        "adverse_exit_fast_minutes": trial.suggest_categorical("adverse_exit_fast_minutes", [15, 30, 45, 60, 90, 120]),
        "adverse_exit_max_mfe_atr": trial.suggest_categorical("adverse_exit_max_mfe_atr", [0.10, 0.20, 0.25, 0.35, 0.50]),
        "adverse_exit_severity_quantile": trial.suggest_categorical("adverse_exit_severity_quantile", [0.55, 0.65, 0.75, 0.85, 0.90]),
    })


def _run_stage(
    *,
    name: str,
    parents: list[Exact1mPolicyParams],
    trials_per_parent: int,
    suggest: Callable[[optuna.Trial, Exact1mPolicyParams], Exact1mPolicyParams],
    selection_rows: pd.DataFrame,
    selection_paths: Mapping[str, np.ndarray],
    calibration_paths: Mapping[str, np.ndarray],
    contract: Exact1mExecutionContract,
    median_atr: float,
    side: str,
    seed: int,
) -> tuple[list[dict[str, Any]], list[Exact1mPolicyParams]]:
    records: list[dict[str, Any]] = []
    for parent_id, parent in enumerate(parents):
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed + parent_id, multivariate=True, group=True, n_startup_trials=min(12, trials_per_parent)),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=12),
        )
        def objective(trial: optuna.Trial) -> float:
            params = suggest(trial, parent)
            score, replay, fitted, monthly = _evaluate(
                selection_rows, selection_paths, calibration_paths,
                params=params, contract=contract, median_atr=median_atr, side=side,
            )
            valid = np.asarray(replay["path_valid"], dtype=bool)
            record = {
                "stage": name, "parent_id": parent_id, "trial": trial.number,
                "objective": score, "valid_rows": int(valid.sum()),
                "worst_month_bps": float(monthly["net_bps_per_trade"].min()) if not monthly.empty else float("nan"),
                "median_month_bps": float(monthly["net_bps_per_trade"].median()) if not monthly.empty else float("nan"),
                "exit_trailing": int((np.asarray(replay["exit_reason"], dtype=object) == "trailing").sum()),
                "exit_stop": int((np.asarray(replay["exit_reason"], dtype=object) == "stop_loss").sum()),
                "exit_timeout": int((np.asarray(replay["exit_reason"], dtype=object) == "timeout_h12").sum()),
                **fitted.to_dict(),
            }
            records.append(record)
            trial.set_user_attr("params", fitted.to_dict())
            trial.set_user_attr("monthly", monthly.to_dict(orient="records"))
            return score
        study.optimize(objective, n_trials=trials_per_parent, n_jobs=1, show_progress_bar=False)
    unique: dict[str, Exact1mPolicyParams] = {}
    for record in sorted(records, key=lambda item: (-float(item["objective"]), int(item["trial"]))):
        params = Exact1mPolicyParams.from_mapping(record)
        key = json.dumps(params.to_dict(), sort_keys=True, default=str)
        unique.setdefault(key, params)
    return records, list(unique.values())


def _portfolio_candidates(rows: pd.DataFrame, replay: Mapping[str, np.ndarray], *, strategy_id: str) -> pd.DataFrame:
    valid = np.asarray(replay["path_valid"], dtype=bool) & np.isfinite(np.asarray(replay["net_bps"], float))
    work = rows.loc[valid].copy().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    score = pd.to_numeric(work["score"], errors="raise")
    rank = score.groupby(work["entry_ts"], sort=False).rank(pct=True, method="average")
    positions = np.flatnonzero(valid)
    result = pd.DataFrame({
        "timestamp": work["entry_ts"], "symbol": work["symbol"].astype(str),
        "side": work["side_name"].astype(str).str.lower(),
        "strategy_id": strategy_id, "policy_archetype": strategy_id,
        "normalized_rank_score": rank.to_numpy(float), "strategy_rank_pct": rank.to_numpy(float),
        "base_strategy_threshold": 0.0, "calibrated_score": score.to_numpy(float),
        "entry_price": pd.to_numeric(work["entry_price"], errors="raise"),
        "exit_timestamp": pd.to_datetime(np.asarray(replay["exit_timestamp"])[positions], utc=True),
        "exit_price": np.asarray(replay["exit_price"], float)[positions],
        "net_return": np.asarray(replay["net_bps"], float)[positions] / 10_000.0,
        "gross_return": np.asarray(replay["gross_bps"], float)[positions] / 10_000.0,
        "holding_bars": np.asarray(replay["exit_bar"], float)[positions] + 1.0,
        "simple_policy_exit_reason": np.asarray(replay["exit_reason"], object)[positions],
        "fees_bps": 100.0, "slippage_bps": 0.0, "expected_friction_bps": 100.0,
        "price_gap_bps": 0.0, "liquidity_capacity_weight": 1.0,
        "candidate_id": work["candidate_id"].astype(str), "policy_outcome_available": True,
    })
    return normalise_candidate_table(result)


def _portfolio_selection_metrics(decisions: pd.DataFrame) -> dict[str, Any]:
    """Robust finalist score from the actual constrained portfolio entries."""
    accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)].copy()
    if accepted.empty:
        return {
            "portfolio_entries": 0,
            "portfolio_net_bps_per_trade": float("nan"),
            "portfolio_total_net_bps": 0.0,
            "portfolio_median_month_bps": float("nan"),
            "portfolio_worst_month_bps": float("nan"),
            "portfolio_selection_score": float("-inf"),
            "portfolio_monthly": [],
        }
    accepted["timestamp"] = pd.to_datetime(accepted["timestamp"], utc=True, errors="raise")
    accepted["net_bps"] = pd.to_numeric(accepted["position_net_return"], errors="raise") * 10_000.0
    accepted["month"] = accepted["timestamp"].dt.tz_localize(None).dt.to_period("M").astype(str)
    monthly = accepted.groupby("month", as_index=False).agg(
        trades=("net_bps", "size"),
        net_bps_per_trade=("net_bps", "mean"),
        total_net_bps=("net_bps", "sum"),
    )
    values = monthly["net_bps_per_trade"].to_numpy(float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    worst = float(np.min(values))
    # Final policy selection is portfolio-aware.  It shares the HPO objective's
    # portability preference but derives every term from actual accepted rows.
    selection = median - 0.5 * mad - max(0.0, -worst)
    return {
        "portfolio_entries": int(len(accepted)),
        "portfolio_net_bps_per_trade": float(accepted["net_bps"].mean()),
        "portfolio_total_net_bps": float(accepted["net_bps"].sum()),
        "portfolio_median_month_bps": median,
        "portfolio_worst_month_bps": worst,
        "portfolio_selection_score": float(selection),
        "portfolio_monthly": monthly.to_dict(orient="records"),
    }


def _is_live_parent_compatible(params: Exact1mPolicyParams) -> bool:
    """Whether a contract can be applied by ``advance_shadow_state`` today."""
    return bool(
        float(params.fixed_trailing_gap_mult) > 0.0
        and np.isclose(float(params.sl_abs_floor_pct), 0.0)
        and np.isclose(float(params.sl_abs_cap_pct), 0.0)
        and np.isclose(float(params.trailing_activation_min_pct), 0.0)
        and np.isclose(float(params.trailing_activation_cap_pct), 0.0)
        and np.isclose(float(params.trailing_activation_decay_half_life_minutes), 0.0)
        and int(params.trailing_activation_decay_start_minutes) == 0
        and np.isclose(float(params.capital_protect_mfe_mult), 0.0)
        and not bool(params.adverse_exit_enabled)
        and np.isclose(float(params.holding_rent_bps_per_hour), 0.0)
        and np.isclose(float(params.risk_time_weight), 0.0)
        and params.sl_atr_power is None
        and params.sl_atr_multiplier is None
        and params.tp_atr_power is None
        and params.tp_atr_multiplier is None
    )


def run(args: argparse.Namespace) -> Path:
    dataset = Path(args.dataset_dir).resolve()
    output = Path(args.out_dir).resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite immutable HPO output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    side = str(args.side).strip().lower()
    if side not in {"long", "short"}:
        raise ValueError("exact-1m policy HPO requires side=long or short")
    rows, paths, contract, dataset_manifest = _read_dataset(dataset, side=side)
    # Cast once at dataset load. The simulator can then reuse the matrices for
    # each trial rather than allocating float64 copies repeatedly.
    paths = {key: np.asarray(value, dtype=float) for key, value in paths.items()}
    gates = _assert_dataset_gates(
        rows, paths, contract, dataset_manifest,
        min_monthly_support=int(args.min_monthly_support),
        min_path_coverage=float(args.min_path_coverage),
    )
    calibration_mask = rows["timestamp"].lt(pd.Timestamp("2024-07-01", tz="UTC")).to_numpy()
    selection_mask = ~calibration_mask
    if calibration_mask.sum() < 500 or selection_mask.sum() < 500:
        raise RuntimeError("exact-1m HPO requires substantial pre-July calibration and Jul-Dec selection support")
    calibration_paths = _take(paths, calibration_mask)
    full_selection_rows = rows.loc[selection_mask].reset_index(drop=True)
    full_selection_paths = _take(paths, selection_mask)
    selection_rows, selection_paths, hpo_sample_counts = _stable_month_sample(
        full_selection_rows,
        full_selection_paths,
        rows_per_month=int(args.hpo_rows_per_month),
    )
    median_atr = _median_atr_fraction(calibration_paths)
    incumbent = Exact1mPolicyParams()
    incumbent_score, incumbent_replay, incumbent_fitted, incumbent_monthly = _evaluate(
        selection_rows, selection_paths, calibration_paths,
        params=incumbent, contract=contract, median_atr=median_atr, side=side,
    )
    records: list[dict[str, Any]] = [{
        "stage": "incumbent_control", "parent_id": -1, "trial": -1,
        "objective": incumbent_score, "valid_rows": int(np.asarray(incumbent_replay["path_valid"], bool).sum()),
        "median_month_bps": float(incumbent_monthly["net_bps_per_trade"].median()),
        "worst_month_bps": float(incumbent_monthly["net_bps_per_trade"].min()),
        **incumbent_fitted.to_dict(),
    }]
    if args.execution_surface == "live_parent":
        stages: list[tuple[str, Callable[[optuna.Trial, Exact1mPolicyParams], Exact1mPolicyParams], int, int]] = [
            ("core_broad", _core, int(args.core_trials), int(args.live_broad_parents)),
            ("core_refine", _core_refine, int(args.refine_trials), int(args.live_refine_parents)),
            ("core_polish", _core_polish, int(args.polish_trials), int(args.finalists)),
        ]
    else:
        stages = [
            ("core_geometry", _core, int(args.core_trials), int(args.stage_parents)),
            ("time_controls", _time_controls, int(args.time_trials), int(args.stage_parents)),
            ("full_geometry", _geometry, int(args.geometry_trials), int(args.stage_parents)),
            ("protection", _protection, int(args.protection_trials), int(args.stage_parents)),
        ]
    parents = [incumbent]
    for index, (name, suggest, trials, keep) in enumerate(stages):
        stage_records, candidates = _run_stage(
            name=name, parents=parents, trials_per_parent=trials, suggest=suggest,
            selection_rows=selection_rows, selection_paths=selection_paths,
            calibration_paths=calibration_paths, contract=contract, median_atr=median_atr,
            side=side,
            seed=int(args.seed) + 10_000 * (index + 1),
        )
        records.extend(stage_records)
        parents = candidates[:keep]
        if not parents:
            raise RuntimeError(f"{name} produced no valid candidates")
    trials = pd.DataFrame(records).sort_values(["objective", "stage", "trial"], ascending=[False, True, True], kind="stable")
    trials.to_parquet(output / "sequential_trials.parquet", index=False)
    # A sequential challenger may improve the per-row HPO objective yet lose
    # after exit timestamps interact with the portfolio auction.  The frozen
    # live-parent control must therefore participate in the *same* final
    # constrained tournament; never promote a challenger merely because the
    # control was omitted from the last selection step.
    finalists = [incumbent]
    seen_finalists = {json.dumps(incumbent.to_dict(), sort_keys=True, default=str)}
    for candidate in parents:
        key = json.dumps(candidate.to_dict(), sort_keys=True, default=str)
        if key not in seen_finalists:
            finalists.append(candidate)
            seen_finalists.add(key)
        if len(finalists) >= int(args.finalists):
            break
    final_rows: list[dict[str, Any]] = []
    for rank, params in enumerate(finalists, start=1):
        score, replay, fitted, monthly = _evaluate(
            full_selection_rows, full_selection_paths, calibration_paths,
            params=params, contract=contract, median_atr=median_atr, side=side,
        )
        candidates = _portfolio_candidates(full_selection_rows, replay, strategy_id=f"strict_r3_exact_1m_hpo_finalist_{rank}")
        decisions, equity, _ = replay_candidates(
            candidates, canonical_portfolio_params(), mode="global_auction",
            ev_curve=CAUSAL_AUCTION_CURVE, market_mode="perps", initial_wallet=1000.0,
        )
        portfolio = _portfolio_selection_metrics(decisions)
        final_rows.append({
            "rank": rank, "is_incumbent_control": bool(rank == 1), "objective": score,
            "portfolio_uses_exact_exit_timestamp": bool(pd.to_datetime(candidates["exit_timestamp"], utc=True).notna().all()),
            "monthly": monthly.to_dict(orient="records"), **portfolio, **fitted.to_dict(),
        })
        candidates.to_parquet(output / f"finalist_{rank}_portfolio_candidates.parquet", index=False)
        decisions.to_parquet(output / f"finalist_{rank}_portfolio_decisions.parquet", index=False)
        equity.to_parquet(output / f"finalist_{rank}_portfolio_equity.parquet", index=False)
    final = pd.DataFrame(final_rows).sort_values(
        ["portfolio_selection_score", "portfolio_net_bps_per_trade", "objective", "rank"],
        ascending=[False, False, False, True],
        kind="stable",
    )
    final.to_parquet(output / "finalist_portfolio_metrics.parquet", index=False)
    winner = final.iloc[0].to_dict()
    winner_params = Exact1mPolicyParams.from_mapping(winner)
    live_parent_compatible = _is_live_parent_compatible(winner_params)
    if args.execution_surface == "live_parent" and not live_parent_compatible:
        raise AssertionError("live_parent HPO produced an unsupported policy parameter")
    (output / "winner.json").write_text(json.dumps({
        "schema": EXACT_1M_POLICY_SCHEMA, "research_only": True,
        "side": side,
        "contract": contract.to_dict(), "contract_hash": contract.hash,
        "dataset_manifest_sha256": _sha256(dataset / "dataset_manifest.json"),
        "selection": "calibration Feb-Jun 2024; HPO/final constrained portfolio selection Jul-Dec 2024; 2025+ untouched",
        "execution_surface": str(args.execution_surface),
        "live_parent_compatible": bool(live_parent_compatible),
        "dataset_gates": gates,
        "hpo_sample": {
            "rows_per_month": int(args.hpo_rows_per_month),
            "rows": int(len(selection_rows)),
            "monthly_rows": hpo_sample_counts,
            "selection": "deterministic candidate-ID hash; outcome-free; full selection rows retained for finalist portfolio replay",
        },
        "winner": {key: value for key, value in winner.items() if key in Exact1mPolicyParams.__dataclass_fields__},
        "winner_objective": float(winner["objective"]),
        "portfolio": {key: winner[key] for key in winner if key.startswith("portfolio_")},
        "entry": "uniform decision + 5m for every HPO row",
        "exit": "completed-1m parent-policy threshold; trailing closes at threshold bar close",
        "cost": "100 bps exactly once", "promotion": "not authorised",
    }, indent=2, default=str) + "\n", encoding="utf-8")
    correctness = {
        "schema": "strict_r3_exact_1m_policy_hpo_correctness_v1", "status": "passed",
        "side": side,
        "contract_hash": contract.hash,
        "candidate_routing": dataset_manifest["routing"],
        "dataset_gates": gates,
        "entry": "all selected HPO rows use one fixed decision+5m convention; no rank-dependent entry materialisation",
        "paths": "complete 720x1m only; invalid rows excluded from HPO",
        "trailing": "prior completed MFE arms following-bar action; exit uses threshold-bar close",
        "time_units": "all new decay, fast-adverse and holding-rent inputs are minutes/hours",
        "cost": "gross minus net equals 100 bps once for every valid replay row",
        "portfolio": "finalists are selected by robust constrained-portfolio monthly economics using explicit exact one-minute exit timestamps",
        "prohibitions": ["no Kraken IO", "no live state", "no 2025+ HPO outcomes", "no policy promotion"],
    }
    (output / "correctness_report.json").write_text(json.dumps(correctness, indent=2) + "\n", encoding="utf-8")
    (output / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_exact_1m_policy_hpo_run_v1", "dataset": str(dataset),
        "side": side,
        "dataset_manifest_sha256": _sha256(dataset / "dataset_manifest.json"),
        "output": str(output), "seed": int(args.seed), "contract_hash": contract.hash,
        "execution_surface": str(args.execution_surface),
        "stages": [value[0] for value in stages], "finalists": int(args.finalists),
        "hpo_rows_per_month": int(args.hpo_rows_per_month),
        "hpo_sample_monthly_rows": hpo_sample_counts,
    }, indent=2) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--side", choices=["long", "short"], default="long")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--execution-surface",
        choices=["live_parent", "rich_research"],
        default="live_parent",
        help="live_parent exactly matches the deployed stop/activation/giveback state machine; rich_research is non-deployable.",
    )
    parser.add_argument("--core-trials", type=int, default=48)
    parser.add_argument("--refine-trials", type=int, default=24)
    parser.add_argument("--polish-trials", type=int, default=16)
    parser.add_argument("--live-broad-parents", type=int, default=6)
    parser.add_argument("--live-refine-parents", type=int, default=4)
    parser.add_argument("--time-trials", type=int, default=24)
    parser.add_argument("--geometry-trials", type=int, default=36)
    parser.add_argument("--protection-trials", type=int, default=36)
    parser.add_argument("--stage-parents", type=int, default=4)
    parser.add_argument("--finalists", type=int, default=3)
    parser.add_argument("--min-monthly-support", type=int, default=75)
    parser.add_argument("--min-path-coverage", type=float, default=0.90)
    parser.add_argument(
        "--hpo-rows-per-month", type=int, default=500,
        help="Deterministic outcome-free HPO sample cap per month; finalists replay full Jul-Dec paths.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
