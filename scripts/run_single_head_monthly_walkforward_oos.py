#!/usr/bin/env python3
"""Run a single-head monthly train-base/train-meta policy-OOS walk-forward.

This runner is intentionally narrow:

* selects the June-best simple-policy head from a source candidate table;
* reuses the source run's native LGBM selected features and best params;
* trains only that head through March, April, and May cutoffs by default;
  an explicit opt-in flag can add a train-through-June / score-July fold;
* generates policy-OOS predictions for the following calendar month;
* runs simple_policy_optimiser with portfolio replay disabled; and
* summarizes validation-only OOS metrics from rows not used for policy tuning.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data_perp"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

DEFAULT_SOURCE_RUN_ID = "20260629_050000_lgbm_mda"
DEFAULT_FEATURE_SOURCE_RUN_ID = "20260629_050000"
DEFAULT_EXPERIMENT_ID = "20260701_130000_single_head_monthly_walkforward_oos"
LABEL_EMBARGO_HOURS = 13
SOURCE_REGISTRY_REL = Path("strategy_registry") / "deployed_four_heads_perps.csv"
SOURCE_CANDIDATES_REL = (
    Path("simple_policy_optimiser") / "simple_policy_candidates_broad.parquet"
)
REQUIRED_BASE_ERROR_OOF_COLUMNS = [
    "oof_base_error_archetype_id",
    "oof_base_error_archetype_is_bad",
    "oof_base_error_archetype_is_good",
    "oof_base_error_archetype_is_neutral",
    "oof_base_error_distance_to_archetype_centroid",
    "oof_base_error_distance_to_nearest_bad_archetype",
    "oof_base_error_archetype_oof_bad_rate_lift",
    "oof_base_error_distance_to_bad_archetype",
    "oof_base_error_distance_to_good_archetype",
]


@dataclass(frozen=True)
class FoldSpec:
    name: str
    run_id: str
    train_end: pd.Timestamp
    policy_start: pd.Timestamp
    policy_split: pd.Timestamp
    policy_end: pd.Timestamp


def _utc(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def _coerce_utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _parse_csv_arg(value: str | None) -> list[str]:
    if value is None or str(value).strip() == "":
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _strategy_ids_csv(strategy_ids: Sequence[str] | str) -> str:
    if isinstance(strategy_ids, str):
        return ",".join(_parse_csv_arg(strategy_ids))
    return ",".join(str(s).strip() for s in strategy_ids if str(s).strip())


def _strategy_id_list(strategy_ids: Sequence[str] | str) -> list[str]:
    if isinstance(strategy_ids, str):
        return _parse_csv_arg(strategy_ids)
    return [str(s).strip() for s in strategy_ids if str(s).strip()]


def _normalise_side_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"long", "buy", "+1", "1"}:
        return "long"
    if text in {"short", "sell", "-1"}:
        return "short"
    try:
        numeric = float(text)
    except Exception:
        return ""
    if numeric > 0:
        return "long"
    if numeric < 0:
        return "short"
    return ""


def _strategy_side_from_id(strategy_id: Any) -> str:
    sid = str(strategy_id or "").strip().lower()
    if sid.startswith("long_"):
        return "long"
    if sid.startswith("short_"):
        return "short"
    return ""


def _fold_aliases(fold: FoldSpec) -> set[str]:
    aliases = {fold.name, fold.run_id}
    if fold.name == "train_through_june_score_july":
        aliases.update({"july", "2026-07", "train_june_score_july"})
    return aliases


def _filter_folds(folds: list[FoldSpec], only_folds: list[str]) -> list[FoldSpec]:
    requested = [str(value).strip() for value in only_folds if str(value).strip()]
    if not requested:
        return list(folds)
    selected: list[FoldSpec] = []
    unmatched: list[str] = []
    for value in requested:
        match = next((fold for fold in folds if value in _fold_aliases(fold)), None)
        if match is None:
            unmatched.append(value)
            continue
        if match not in selected:
            selected.append(match)
    if unmatched:
        available = sorted(alias for fold in folds for alias in _fold_aliases(fold))
        raise ValueError(
            "Unknown fold filter(s): "
            + ", ".join(unmatched)
            + ". Available folds/aliases: "
            + ", ".join(available)
        )
    if not selected:
        raise ValueError("Fold filter selected no folds.")
    return selected


def _bounded_policy_split(
    *,
    policy_start: pd.Timestamp,
    policy_end: pd.Timestamp,
    preferred_split: pd.Timestamp,
) -> pd.Timestamp:
    if policy_end <= policy_start:
        raise ValueError(
            f"policy_end must be after policy_start: {policy_start} >= {policy_end}"
        )
    if policy_start < preferred_split < policy_end:
        return preferred_split
    midpoint_ns = int(policy_start.value + (policy_end.value - policy_start.value) // 2)
    split = pd.Timestamp(midpoint_ns, tz="UTC")
    if not policy_start < split < policy_end:
        raise ValueError(
            f"Unable to build non-empty optimise/validation split for {policy_start}..{policy_end}"
        )
    return split


def _july_fold(
    experiment_id: str,
    *,
    policy_end: str | pd.Timestamp | None = None,
    policy_split: str | pd.Timestamp | None = None,
) -> FoldSpec:
    policy_start = _utc("2026-07-01 00:00:00")
    end = _coerce_utc(policy_end or "2026-08-01 00:00:00")
    preferred_split = _coerce_utc(policy_split or "2026-07-16 00:00:00")
    split = _bounded_policy_split(
        policy_start=policy_start,
        policy_end=end,
        preferred_split=preferred_split,
    )
    return FoldSpec(
        name="train_through_june_score_july",
        run_id=f"{experiment_id}_train_june_score_july",
        train_end=policy_start - pd.Timedelta(hours=LABEL_EMBARGO_HOURS),
        policy_start=policy_start,
        policy_split=split,
        policy_end=end,
    )


def _folds(
    experiment_id: str,
    *,
    include_july_fold: bool = False,
    july_policy_end: str | pd.Timestamp | None = None,
    july_policy_split: str | pd.Timestamp | None = None,
) -> list[FoldSpec]:
    def train_cutoff(policy_start: str) -> pd.Timestamp:
        return _utc(policy_start) - pd.Timedelta(hours=LABEL_EMBARGO_HOURS)

    folds = [
        FoldSpec(
            name="train_through_march_score_april",
            run_id=f"{experiment_id}_train_march_score_april",
            train_end=train_cutoff("2026-04-01 00:00:00"),
            policy_start=_utc("2026-04-01 00:00:00"),
            policy_split=_utc("2026-04-16 00:00:00"),
            policy_end=_utc("2026-05-01 00:00:00"),
        ),
        FoldSpec(
            name="train_through_april_score_may",
            run_id=f"{experiment_id}_train_april_score_may",
            train_end=train_cutoff("2026-05-01 00:00:00"),
            policy_start=_utc("2026-05-01 00:00:00"),
            policy_split=_utc("2026-05-16 00:00:00"),
            policy_end=_utc("2026-06-01 00:00:00"),
        ),
        FoldSpec(
            name="train_through_may_score_june",
            run_id=f"{experiment_id}_train_may_score_june",
            train_end=train_cutoff("2026-06-01 00:00:00"),
            policy_start=_utc("2026-06-01 00:00:00"),
            policy_split=_utc("2026-06-16 00:00:00"),
            policy_end=_utc("2026-07-01 00:00:00"),
        ),
    ]
    if include_july_fold:
        folds.append(
            _july_fold(
                experiment_id,
                policy_end=july_policy_end,
                policy_split=july_policy_split,
            )
        )
    return folds


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _timestamp(value: Any) -> str:
    return pd.Timestamp(value).tz_convert("UTC").isoformat()


def _load_source_symbols(source_run_id: str) -> list[str]:
    source_slice = DATA_ROOT / "artifacts" / source_run_id / "slices" / "slice_plan.json"
    payload = json.loads(source_slice.read_text(encoding="utf-8"))
    views = payload.get("materialized_views") or {}
    symbols: set[str] = set()
    for key in ("train_base", "train_meta", "policy_optimiser"):
        view = views.get(key) or {}
        symbols.update(str(s) for s in view.get("symbols") or [] if str(s))
    if not symbols:
        raise RuntimeError(f"No symbols found in source slice plan: {source_slice}")
    return sorted(symbols)


def _source_registry_rows_by_id(source_run_id: str) -> dict[str, dict[str, str]]:
    source = DATA_ROOT / "artifacts" / source_run_id / SOURCE_REGISTRY_REL
    if not source.exists():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        sid = str(row.get("strategy_id", "")).strip()
        if sid:
            out[sid] = row
    return out


def _source_registry_side_map(source_run_id: str) -> dict[str, str]:
    rows = _source_registry_rows_by_id(source_run_id)
    out: dict[str, str] = {}
    for sid, row in rows.items():
        side = (
            _normalise_side_text(row.get("trade_side"))
            or _normalise_side_text(row.get("side"))
            or _strategy_side_from_id(sid)
        )
        if side:
            out[sid] = side
    return out


def _read_candidate_table_for_selection(candidate_path: Path) -> pd.DataFrame:
    if not candidate_path.exists():
        raise FileNotFoundError(candidate_path)
    schema_names = set(pq.read_schema(candidate_path).names)
    cols = [
        col
        for col in ("timestamp", "strategy_id", "net_return", "rank_pct", "side", "trade_side")
        if col in schema_names
    ]
    missing = sorted({"timestamp", "strategy_id", "net_return", "rank_pct"}.difference(cols))
    if missing:
        raise RuntimeError(f"{candidate_path}: missing required columns {missing}")
    return pd.read_parquet(candidate_path, columns=cols)


def _attach_candidate_side(df: pd.DataFrame, source_run_id: str) -> pd.DataFrame:
    out = df.copy()
    side = pd.Series("", index=out.index, dtype=object)
    for col in ("trade_side", "side"):
        if col not in out.columns:
            continue
        current = out[col].map(_normalise_side_text)
        side = side.where(side.astype(str).ne(""), current)
    missing = side.astype(str).eq("")
    if missing.any():
        side_map = _source_registry_side_map(source_run_id)
        mapped = out.loc[missing, "strategy_id"].astype(str).map(side_map).fillna("")
        side.loc[missing] = mapped
    missing = side.astype(str).eq("")
    if missing.any():
        side.loc[missing] = out.loc[missing, "strategy_id"].map(_strategy_side_from_id)
    out["_trade_side"] = side.astype(str)
    return out


def _june_strategy_summary(source_run_id: str) -> pd.DataFrame:
    candidate_path = DATA_ROOT / "artifacts" / source_run_id / SOURCE_CANDIDATES_REL
    df = _read_candidate_table_for_selection(candidate_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    june = df[
        (df["timestamp"] >= pd.Timestamp("2026-06-01", tz="UTC"))
        & (df["timestamp"] < pd.Timestamp("2026-07-01", tz="UTC"))
    ].copy()
    if june.empty:
        raise RuntimeError(f"No June rows in {candidate_path}")
    june = _attach_candidate_side(june, source_run_id)
    grouped = (
        june.groupby(["strategy_id", "_trade_side"], dropna=False)
        .agg(
            rows=("net_return", "size"),
            net_return_sum=("net_return", "sum"),
            mean_net_return=("net_return", "mean"),
            win_rate=("net_return", lambda s: float((s > 0).mean())),
            avg_rank=("rank_pct", "mean"),
        )
        .reset_index()
        .sort_values(["net_return_sum", "mean_net_return"], ascending=False)
    )
    grouped = grouped.rename(columns={"_trade_side": "side"})
    grouped["selection_source"] = str(candidate_path)
    grouped["selection_window_start"] = "2026-06-01T00:00:00+00:00"
    grouped["selection_window_end"] = "2026-07-01T00:00:00+00:00"
    grouped["selection_metric"] = "max_june_total_net_return"
    return grouped


def _select_june_best_strategy(source_run_id: str) -> dict[str, Any]:
    grouped = _june_strategy_summary(source_run_id)
    best = grouped.iloc[0].to_dict()
    best["all_june_strategies"] = grouped.to_dict(orient="records")
    best["selection_mode"] = "overall"
    best["strategy_ids"] = [str(best["strategy_id"])]
    best["selected_strategies"] = [dict(best)]
    return best


def _select_june_strategy_set(
    source_run_id: str,
    *,
    selection_mode: str,
    sides: Sequence[str] = ("long", "short"),
) -> dict[str, Any]:
    mode = str(selection_mode or "overall").strip().lower().replace("-", "_")
    grouped = _june_strategy_summary(source_run_id)
    if mode in {"overall", "single", "best"}:
        return _select_june_best_strategy(source_run_id)
    if mode not in {"best_per_side", "per_side", "bidirectional"}:
        raise ValueError(f"Unknown selection mode: {selection_mode}")

    requested_sides = [_normalise_side_text(side) for side in sides]
    requested_sides = [side for side in requested_sides if side in {"long", "short"}]
    if not requested_sides:
        raise ValueError("best_per_side selection requires at least one side.")
    selected: list[dict[str, Any]] = []
    missing_sides: list[str] = []
    for side in requested_sides:
        side_rows = grouped[grouped["side"].astype(str).eq(side)]
        if side_rows.empty:
            missing_sides.append(side)
            continue
        selected.append(side_rows.iloc[0].to_dict())
    if missing_sides:
        raise RuntimeError(
            "No June candidate strategies found for side(s): " + ", ".join(missing_sides)
        )
    strategy_ids = [str(row["strategy_id"]) for row in selected]
    primary = dict(selected[0])
    primary.update(
        {
            "selection_mode": "best_per_side",
            "selection_sides": requested_sides,
            "strategy_id": strategy_ids[0],
            "strategy_ids": strategy_ids,
            "selected_strategies": selected,
            "all_june_strategies": grouped.to_dict(orient="records"),
        }
    )
    return primary


def _source_strategy_row(source_run_id: str, strategy_id: str) -> dict[str, str]:
    rows = _source_registry_rows_by_id(source_run_id)
    row = rows.get(str(strategy_id).strip())
    if row is None:
        source = DATA_ROOT / "artifacts" / source_run_id / SOURCE_REGISTRY_REL
        raise RuntimeError(f"Strategy {strategy_id} not found in {source}")
    return row


def _source_strategy_rows(source_run_id: str, strategy_ids: Sequence[str]) -> list[dict[str, str]]:
    rows = _source_registry_rows_by_id(source_run_id)
    selected: list[dict[str, str]] = []
    missing: list[str] = []
    for strategy_id in strategy_ids:
        row = rows.get(str(strategy_id).strip())
        if row is None:
            missing.append(str(strategy_id))
        else:
            selected.append(row)
    if missing:
        source = DATA_ROOT / "artifacts" / source_run_id / SOURCE_REGISTRY_REL
        raise RuntimeError(f"Strategy ids not found in {source}: {missing}")
    return selected


def _write_strategy_registry(run_id: str, rows: Sequence[dict[str, str]]) -> Path:
    out = DATA_ROOT / "artifacts" / run_id / "strategy_registry"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "selected_single_head_strategy_registry.csv"
    rows = [dict(row) for row in rows]
    if not rows:
        raise ValueError("Cannot write an empty strategy registry")
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_one_row_registry(run_id: str, row: dict[str, str]) -> Path:
    return _write_strategy_registry(run_id, [row])


def _plan(
    *,
    run_id: str,
    fold: FoldSpec,
    symbols: list[str],
    source_run_id: str,
    feature_source_run_id: str,
    strategy_id: str,
) -> dict[str, Any]:
    train_start = "2022-11-15T00:00:00+00:00"
    train_end = _timestamp(fold.train_end)
    policy_start = _timestamp(fold.policy_start)
    policy_split = _timestamp(fold.policy_split)
    policy_end = _timestamp(fold.policy_end)

    def training_plan(role: str, tag: str) -> dict[str, Any]:
        return {
            "fit_idx": [0],
            "val_idx": [],
            "predict_idx": [0],
            "symbols_fit": symbols,
            "symbols_predict": symbols,
            "tag": tag,
            "metadata": {
                "preset": "monthly_calendar_walkforward",
                "plan_kind": "fit_all_before_policy_month",
                "fit_role": "training_rows_before_policy_month",
                "predict_role": "training_rows_before_policy_month",
                "n_fit": 1,
                "n_predict": 1,
                "n_symbols_fit": len(symbols),
                "n_symbols_predict": len(symbols),
                "fit_start": train_start,
                "fit_end": train_end,
                "predict_start": train_start,
                "predict_end": train_end,
                "fit_actual_start": train_start,
                "fit_actual_end": train_end,
                "predict_actual_start": train_start,
                "predict_actual_end": train_end,
                "fit_window_start": train_start,
                "fit_window_end": train_end,
                "predict_window_start": train_start,
                "predict_window_end": train_end,
                "consumer_role": role,
            },
        }

    def policy_plan(tag: str, predict_role: str, start: str, end: str, idx: int) -> dict[str, Any]:
        return {
            "fit_idx": [0],
            "val_idx": [],
            "predict_idx": [idx],
            "symbols_fit": symbols,
            "symbols_predict": symbols,
            "tag": tag,
            "metadata": {
                "preset": "monthly_calendar_policy_oos",
                "plan_kind": "fit_predict",
                "fit_role": "rows_before_policy_month",
                "predict_role": predict_role,
                "policy_optimiser_predict_scope": tag,
                "policy_optimiser_recent_weeks_enable": True,
                "policy_optimiser_all_symbols": True,
                "policy_optimiser_sample_fraction_cap_applied": False,
                "policy_optimiser_tail_months": 1,
                "n_fit": 1,
                "n_predict": 1,
                "n_total_valid": 2,
                "predict_fraction": 0.5,
                "n_symbols_fit": len(symbols),
                "n_symbols_predict": len(symbols),
                "fit_start": train_start,
                "fit_end": train_end,
                "predict_start": start,
                "predict_end": end,
                "fit_actual_start": train_start,
                "fit_actual_end": train_end,
                "predict_actual_start": start,
                "predict_actual_end": end,
                "fit_window_start": train_start,
                "fit_window_end": train_end,
                "predict_window_start": start,
                "predict_window_end": end,
            },
        }

    consumer_plans = {
        "base_model_fit": [
            training_plan("base_model_fit", "train_base_monthly_walkforward")
        ],
        "meta_model_fit": [
            training_plan("meta_model_fit", "train_meta_monthly_walkforward")
        ],
        "policy_optimiser": [
            policy_plan(
                "policy_recent_optimise",
                "policy_holdout_recent_optimise",
                policy_start,
                policy_split,
                1,
            ),
            policy_plan(
                "policy_recent_validation",
                "policy_holdout_recent_validation",
                policy_split,
                policy_end,
                2,
            ),
        ],
    }
    materialized_views = {
        "train_base": {
            "stage_name": "train_base",
            "allocation_target": 1.0,
            "source_roles": ["base_model_fit"],
            "symbols": symbols,
            "allowed_start_ts": train_start,
            "allowed_end_ts": train_end,
            "n_plans": 1,
        },
        "train_meta": {
            "stage_name": "train_meta",
            "allocation_target": 1.0,
            "source_roles": ["meta_model_fit"],
            "symbols": symbols,
            "allowed_start_ts": train_start,
            "allowed_end_ts": train_end,
            "n_plans": 1,
        },
        "policy_optimiser": {
            "stage_name": "policy_optimiser",
            "allocation_target": 1.0,
            "source_roles": ["policy_optimiser"],
            "symbols": symbols,
            "allowed_start_ts": policy_start,
            "allowed_end_ts": policy_end,
            "allowed_periods": [
                {
                    "start_ts": policy_start,
                    "end_ts": policy_split,
                    "predict_role": "policy_holdout_recent_optimise",
                    "tag": "policy_recent_optimise",
                },
                {
                    "start_ts": policy_split,
                    "end_ts": policy_end,
                    "predict_role": "policy_holdout_recent_validation",
                    "tag": "policy_recent_validation",
                },
            ],
            "n_plans": 2,
        },
    }
    return {
        "version": 4,
        "run_id": run_id,
        "ts_sig": _timestamp(fold.train_end),
        "exchange_context": {"market_mode": "perps", "exchange": "krakenfutures"},
        "planner": {
            "preset": "manual_monthly_calendar_walkforward",
            "symbol_policy_mode": "all_symbols",
            "symbol_fraction": 1.0,
        },
        "policy_optimiser_tail_months": 1,
        "policy_optimiser_max_sample_fraction": 1.0,
        "policy_optimiser_recent_weeks_enable": True,
        "allocation_targets": {
            "train_base": 1.0,
            "train_meta": 1.0,
            "policy_optimiser": 1.0,
        },
        "consumer_plans": consumer_plans,
        "materialized_views": materialized_views,
        "allocation_diagnostics": {},
        "event_fingerprint": {
            "source_run_id": source_run_id,
            "feature_source_run_id": feature_source_run_id,
            "strategy_id": strategy_id,
            "fold": fold.name,
        },
    }


def _write_slice_plan(
    *,
    run_id: str,
    fold: FoldSpec,
    symbols: list[str],
    source_run_id: str,
    feature_source_run_id: str,
    strategy_id: str,
) -> Path:
    path = DATA_ROOT / "artifacts" / run_id / "slices" / "slice_plan.json"
    payload = _plan(
        run_id=run_id,
        fold=fold,
        symbols=symbols,
        source_run_id=source_run_id,
        feature_source_run_id=feature_source_run_id,
        strategy_id=strategy_id,
    )
    _json_dump(path, payload)
    return path


def _base_env(
    *,
    run_id: str,
    source_run_id: str,
    feature_source_run_id: str,
    label_run_id: str,
    label_ablation_mode: str,
    label_weight_mode: str,
    label_weight_recipe: str,
    true_soft_labels: bool,
    disable_self_distillation: bool,
    registry_path: Path,
    slice_plan_path: Path,
    strategy_id: str,
    policy_trials: int | None,
) -> dict[str, str]:
    env = os.environ.copy()
    strategy_ids_csv = _strategy_ids_csv(strategy_id)
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(ROOT),
            "MPLCONFIGDIR": str(ROOT / ".mplconfig"),
            "EPM_OUTPUT_RUN_ID": run_id,
            "EPM_EXCHANGE": "krakenfutures",
            "EPM_ARTIFACT_SOURCE_RUN_ID": source_run_id,
            "EPM_LABEL_SOURCE_RUN_ID": label_run_id,
            "EPM_LABEL_ARTIFACT_RUN_ID": label_run_id,
            "EPM_LABEL_ABLATION_MODE": label_ablation_mode,
            "EPM_LGBM_TRUE_SOFT_LABELS": "1" if true_soft_labels else "0",
            "EPM_FEATURE_SOURCE_RUN_ID": feature_source_run_id,
            "EPM_POLICY_FEATURE_SOURCE_RUN_ID": feature_source_run_id,
            "EPM_TRAIN_SLICE_PLAN_PATH": str(slice_plan_path),
            "EPM_TRAIN_SLICE_PLAN_EVENT_RUN_ID": source_run_id,
            "EPM_TRAIN_EXTEND_TO_LATEST": "0",
            "EPM_MASK_STRATEGY_SOURCE_CSV": str(registry_path),
            "EPM_MASK_STRATEGY_TOP_N": "1",
            "EPM_MASK_STRATEGY_RANKING_METRIC": "score_for_best_params",
            "EPM_BASE_STRATEGY_IDS": strategy_ids_csv,
            "EPM_META_STRATEGY_IDS": strategy_ids_csv,
            "EPM_POLICY_STRATEGY_IDS": strategy_ids_csv,
            "EPM_LABEL_STRATEGY_IDS": strategy_ids_csv,
            "EPM_REQUIRE_STRATEGY_ALLOWLIST": "1",
            "EPM_MODEL_BACKEND": "lgbm_pipeline",
            "EPM_TRAINING_MODEL_BACKEND": "lgbm_pipeline",
            "EPM_LGBM_USE_NATIVE_PRESET": "1",
            "EPM_LGBM_REQUIRE_NATIVE_PRESET": "1",
            "EPM_LGBM_NATIVE_PRESET_SOURCE_RUN_ID": source_run_id,
            "EPM_LGBM_NATIVE_PRESET_PARAMS_ONLY": "0",
            "EPM_LGBM_CV_MODE": "forward_burnin",
            "EPM_LGBM_BASE_FORWARD_BURN_IN_DAYS": "365",
            "EPM_LGBM_META_FORWARD_VALIDATION_MONTHS": "6",
            "EPM_LGBM_FORWARD_ALLOW_SHORT_HISTORY_FALLBACK": "1",
            "EPM_LGBM_FORWARD_SHORT_HISTORY_FALLBACK_FRAC": "0.70",
            "EPM_LGBM_FORWARD_MIN_TRAIN_ROWS": "50",
            "EPM_LGBM_FORWARD_MIN_VALID_ROWS": "10",
            "EPM_LGBM_TIME_SPREAD_HPO_SELECTION": "1",
            "EPM_LGBM_HPO_TRIALS": "0",
            "EPM_BASE_HPO_TRIALS": "0",
            "EPM_META_HPO_TRIALS": "0",
            "EPM_LGBM_BASE_LABEL_WEIGHT_HPO": "0",
            "EPM_LGBM_LABEL_WEIGHT_HPO_LAYER1_TRIALS": "0",
            "EPM_LGBM_LABEL_WEIGHT_HPO_LAYER2_TRIALS": "0",
            "EPM_META_BASE_QUALITY_GATE_ENABLE": "0",
            "EPM_LGBM_ARCHETYPE_FEATURES": "1",
            "EPM_LGBM_RAW_CONTRIB_OOF_EXPORT": "0",
            "EPM_LGBM_META_DRIFT_FEATURES": "1",
            "EPM_LGBM_META_DRIFT_MAX_FEATURES": "32",
            "EPM_LGBM_FINAL_OOF_CONTEXT_FEATURES": "0",
            "EPM_LGBM_FINAL_OOF_RAW_STATE_CONTEXT_FEATURES": "1",
            "EPM_LGBM_FINAL_OOF_CONTRIB_CONTEXT_FEATURES": "0",
            "EPM_LGBM_FINAL_OOF_BASE_ERROR_CONTEXT_FEATURES": "1",
            "EPM_LGBM_REGIME_SPECIALIST_ENABLED": "0",
            "EPM_LGBM_REGIME_SPECIALIST_FEATURE_ENGINEERING_DIAGNOSTICS_ENABLED": "0",
            "EPM_LGBM_REGIME_SCORE_FEATURES_ENABLED": "0",
            "EPM_SIMPLE_POLICY_USE_POLICY_OOS_PREDICTIONS": "1",
            "EPM_SIMPLE_POLICY_USE_PRECOMPUTED_META_OOF": "0",
            "EPM_SIMPLE_POLICY_ALLOW_FINAL_FIT_POLICY_GENERATION": "0",
            "EPM_SIMPLE_POLICY_RUN_PORTFOLIO_REPLAY": "0",
            "EPM_POLICY_REPLAY_THRESHOLD_SELECTOR_ENABLED": "0",
            "EPM_SIMPLE_POLICY_WRITE_TRAINING_LIVE_PARITY_CONTRACT": "0",
            "EPM_SIMPLE_POLICY_WRITE_DRIFT_BENCHMARKS": "0",
            "EPM_SIMPLE_POLICY_15M_DOWNLOAD": "1",
            "EPM_SIMPLE_POLICY_ALLOW_LGBM_NATIVE_MISSING": "1",
        }
    )
    mode = str(label_weight_mode or "legacy").strip().lower()
    if mode == "w0":
        env.update(
            {
                "EPM_LABEL_WEIGHT_DISABLE": "1",
                "EPM_LABEL_WEIGHT_USE_BEST_DEFAULT": "0",
                "EPM_LABEL_WEIGHT_BYPASS_BEST_DEFAULT": "1",
                "EPM_LABEL_WEIGHT_RECIPE": "",
                "EPM_LABEL_WEIGHT_BASE_RECIPE": "",
                "EPM_LABEL_WEIGHT_META_RECIPE": "",
                "EPM_LGBM_BASE_LABEL_WEIGHT_HPO": "0",
                "EPM_LGBM_LABEL_WEIGHT_HPO_LAYER1_TRIALS": "0",
                "EPM_LGBM_LABEL_WEIGHT_HPO_LAYER2_TRIALS": "0",
            }
        )
    elif mode == "recipe":
        recipe = str(label_weight_recipe or "").strip()
        if not recipe:
            raise ValueError("--label-weight-mode recipe requires --label-weight-recipe")
        env.update(
            {
                "EPM_LABEL_WEIGHT_DISABLE": "0",
                "EPM_LABEL_WEIGHT_USE_BEST_DEFAULT": "0",
                "EPM_LABEL_WEIGHT_BYPASS_BEST_DEFAULT": "0",
                "EPM_LABEL_WEIGHT_RECIPE": recipe,
                "EPM_LGBM_BASE_LABEL_WEIGHT_HPO": "0",
                "EPM_LGBM_LABEL_WEIGHT_HPO_LAYER1_TRIALS": "0",
                "EPM_LGBM_LABEL_WEIGHT_HPO_LAYER2_TRIALS": "0",
            }
        )
    elif mode != "legacy":
        raise ValueError(f"Unknown label weight mode: {label_weight_mode}")
    if disable_self_distillation:
        env.update(
            {
                "EPM_LGBM_DISABLE_SELF_DISTILLATION": "1",
                "EPM_LGBM_OOF_DISTILLATION_PASSES": "0",
                "EPM_LGBM_MIN_OOF_DISTILLATION_PASSES": "0",
                "EPM_LGBM_META_MIN_OOF_DISTILLATION_PASSES": "0",
                "EPM_META_REQUIRE_DISTILLED_BASE_OOF": "0",
                "EPM_META_MIN_BASE_OOF_DISTILLATION_PASSES": "0",
            }
        )
    if policy_trials is not None:
        env["SIMPLE_POLICY_N_TRIALS"] = str(int(policy_trials))
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    return env


def _labels_env(
    *,
    label_run_id: str,
    source_run_id: str,
    feature_source_run_id: str,
    label_ablation_mode: str,
    label_policy_net_replay: bool,
    label_policy_net_replay_min_coverage: float | None,
    registry_path: Path,
    strategy_id: str,
) -> dict[str, str]:
    env = os.environ.copy()
    strategy_ids_csv = _strategy_ids_csv(strategy_id)
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(ROOT),
            "MPLCONFIGDIR": str(ROOT / ".mplconfig"),
            "EPM_OUTPUT_RUN_ID": label_run_id,
            "EPM_LABEL_ARTIFACT_RUN_ID": label_run_id,
            "EPM_LABEL_ABLATION_MODE": label_ablation_mode,
            "EPM_ARTIFACT_SOURCE_RUN_ID": source_run_id,
            "EPM_FEATURE_SOURCE_RUN_ID": feature_source_run_id,
            "EPM_MASK_STRATEGY_SOURCE_CSV": str(registry_path),
            "EPM_MASK_STRATEGY_TOP_N": "1",
            "EPM_MASK_STRATEGY_RANKING_METRIC": "score_for_best_params",
            "EPM_LABEL_STRATEGY_IDS": strategy_ids_csv,
            "EPM_BASE_STRATEGY_IDS": strategy_ids_csv,
            "EPM_META_STRATEGY_IDS": strategy_ids_csv,
            "EPM_REQUIRE_STRATEGY_ALLOWLIST": "1",
            "EPM_MODEL_BACKEND": "lgbm_pipeline",
            "EPM_TRAINING_MODEL_BACKEND": "lgbm_pipeline",
        }
    )
    if label_policy_net_replay:
        env["EPM_LABEL_POLICY_NET_REPLAY_ENABLED"] = "1"
        if label_policy_net_replay_min_coverage is not None:
            env["EPM_LABEL_POLICY_NET_REPLAY_MIN_COVERAGE"] = str(
                float(label_policy_net_replay_min_coverage)
            )
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    return env


def _pipeline_cmd(stage: str, run_id: str, feature_source_run_id: str) -> list[str]:
    return [
        sys.executable,
        "-u",
        "extreme_price_movements/run_pipeline.py",
        stage,
        "--perps",
        "--exchange",
        "krakenfutures",
        "--model-backend",
        "lgbm_pipeline",
        "--ts",
        feature_source_run_id,
        "--run-id",
        run_id,
    ]


def _data_root_cli_arg() -> str:
    try:
        return str(DATA_ROOT.relative_to(ROOT))
    except ValueError:
        return str(DATA_ROOT)


def _policy_oos_cmd(run_id: str, strategy_id: Sequence[str] | str) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        "scripts/generate_policy_oos_predictions.py",
        "--data-root",
        _data_root_cli_arg(),
        "--run-id",
        run_id,
        "--market-mode",
        "perps",
    ]
    for sid in _strategy_id_list(strategy_id):
        cmd.extend(["--strategy-id", sid])
    return cmd


def _simple_policy_cmd(run_id: str, strategy_id: Sequence[str] | str) -> list[str]:
    return [
        sys.executable,
        "-u",
        "extreme_price_movements/simple_policy_optimiser.py",
        "--data_root",
        _data_root_cli_arg(),
        "--run_id",
        run_id,
        "--market-mode",
        "perps",
        "--strategy-ids",
        _strategy_ids_csv(strategy_id),
    ]


def _append(log_path: Path, message: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _run_step(name: str, cmd: list[str], env: dict[str, str], log_path: Path) -> None:
    _append(log_path, f"START {name}: {' '.join(cmd)}")
    with log_path.open("ab", buffering=0) as log_fp:
        log_fp.write(f"\n=== START {name} ===\n".encode())
        log_fp.write(("CMD " + " ".join(cmd) + "\n").encode())
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            env=env,
        )
        ret = proc.wait()
        log_fp.write(f"\n=== END {name} ret={ret} ===\n".encode())
    _append(log_path, f"END {name}: ret={ret}")
    if ret != 0:
        raise SystemExit(ret)


def _strategy_core(strategy_id: str) -> str:
    raw = str(strategy_id)
    if raw.startswith("long_"):
        return raw[len("long_") :]
    if raw.startswith("short_"):
        return raw[len("short_") :]
    return raw


def _parquet_has_columns(path: Path, columns: list[str]) -> bool:
    try:
        present = set(pq.read_schema(path).names)
    except Exception:
        return False
    return set(columns).issubset(present)


def _label_key(strategy_id: str) -> str:
    return f"train_{strategy_id}_5"


def _label_file(label_run_id: str, strategy_id: str) -> Path:
    return DATA_ROOT / "artifacts" / label_run_id / "labels" / f"{_label_key(strategy_id)}.parquet"


def _label_timestamp_max(label_run_id: str, strategy_id: str) -> pd.Timestamp | None:
    label_file = _label_file(label_run_id, strategy_id)
    if not label_file.exists() or label_file.stat().st_size <= 0:
        return None
    if not _parquet_has_columns(label_file, ["__ts__"]):
        return None
    try:
        frame = pd.read_parquet(label_file, columns=["__ts__"])
    except Exception:
        return None
    if frame.empty:
        return None
    ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return None
    return pd.Timestamp(ts.max()).tz_convert("UTC")


def _labels_ready(
    label_run_id: str,
    strategy_id: str,
    *,
    min_label_max_ts: pd.Timestamp | None = None,
) -> bool:
    run_root = DATA_ROOT / "artifacts" / label_run_id
    manifest = run_root / "labels" / "labels_manifest.json"
    label_key = _label_key(strategy_id)
    label_file = _label_file(label_run_id, strategy_id)
    if not manifest.exists() or not label_file.exists():
        return False
    if label_file.stat().st_size <= 0:
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return False
    if label_key not in (payload.get("datasets") or {}):
        return False
    if min_label_max_ts is not None:
        actual_max = _label_timestamp_max(label_run_id, strategy_id)
        if actual_max is None:
            return False
        if actual_max < _coerce_utc(min_label_max_ts):
            return False
    return True


def _labels_ready_all(
    label_run_id: str,
    strategy_ids: Sequence[str],
    *,
    min_label_max_ts: pd.Timestamp | None = None,
) -> bool:
    return all(
        _labels_ready(
            label_run_id,
            strategy_id,
            min_label_max_ts=min_label_max_ts,
        )
        for strategy_id in strategy_ids
    )


def _missing_label_ids(
    label_run_id: str,
    strategy_ids: Sequence[str],
    *,
    min_label_max_ts: pd.Timestamp | None = None,
) -> list[str]:
    return [
        strategy_id
        for strategy_id in strategy_ids
        if not _labels_ready(
            label_run_id,
            strategy_id,
            min_label_max_ts=min_label_max_ts,
        )
    ]


def _stage_done_marker(run_id: str, stage: str) -> Path:
    return DATA_ROOT / "artifacts" / run_id / "stage_markers" / f"{stage}.done"


def _mark_stage_done(run_id: str, stage: str) -> None:
    marker = _stage_done_marker(run_id, stage)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "stage": stage,
                "completed_at_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _stage_success_logged(run_id: str, stage: str, log_path: Path | None) -> bool:
    if log_path is None or not log_path.exists():
        return False
    needle = f"END {run_id}_{stage}: ret=0"
    try:
        return needle in log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False


def _stage_ready(
    run_id: str,
    stage: str,
    strategy_id: str,
    log_path: Path | None = None,
) -> bool:
    run_root = DATA_ROOT / "artifacts" / run_id
    core = _strategy_core(strategy_id)
    if stage == "train_base":
        oof_path = run_root / "oof" / f"oof_{strategy_id}_H5.parquet"
        return (
            (run_root / "base_models_intermediate.pkl").exists()
            and (run_root / "base_models_intermediate.manifest.json").exists()
            and oof_path.exists()
            and _parquet_has_columns(oof_path, REQUIRED_BASE_ERROR_OOF_COLUMNS)
        )
    if stage == "train_meta":
        return (
            (run_root / "models" / "model_state_meta.pkl").exists()
            and (run_root / "models" / "model_state_meta.manifest.json").exists()
            and (
                run_root
                / "meta_oof"
                / f"meta_oof_{strategy_id}_tbm_clf.parquet"
            ).exists()
        )
    if stage == "policy_oos":
        return (
            run_root
            / "policy_oos_predictions"
            / f"policy_oos_{core}_clf.parquet"
        ).exists() or (
            run_root
            / "policy_oos_predictions"
            / f"policy_oos_{strategy_id}_clf.parquet"
        ).exists()
    if stage == "simple_policy":
        metrics = run_root / "policy_optimisation_oos_metrics.json"
        candidates = (
            run_root
            / "simple_policy_optimiser"
            / "simple_policy_candidates_broad.parquet"
        )
        return (
            metrics.exists()
            and candidates.exists()
            and (
                _stage_done_marker(run_id, stage).exists()
                or _stage_success_logged(run_id, stage, log_path)
            )
        )
    return False


def _require_stage_ready(
    run_id: str,
    stage: str,
    strategy_id: str,
    log_path: Path | None = None,
) -> None:
    if not _stage_ready(run_id, stage, strategy_id, log_path=log_path):
        raise RuntimeError(
            f"{run_id} stage {stage} finished without required artifacts."
        )


def _stage_ready_all(
    run_id: str,
    stage: str,
    strategy_ids: Sequence[str],
    log_path: Path | None = None,
) -> bool:
    return all(_stage_ready(run_id, stage, sid, log_path=log_path) for sid in strategy_ids)


def _require_stage_ready_all(
    run_id: str,
    stage: str,
    strategy_ids: Sequence[str],
    log_path: Path | None = None,
) -> None:
    missing = [
        sid
        for sid in strategy_ids
        if not _stage_ready(run_id, stage, sid, log_path=log_path)
    ]
    if missing:
        raise RuntimeError(
            f"{run_id} stage {stage} finished without required artifacts for {missing}."
        )


def _validation_candidate_summary(run_id: str, fold: FoldSpec, strategy_id: str) -> dict[str, Any]:
    path = (
        DATA_ROOT
        / "artifacts"
        / run_id
        / "simple_policy_optimiser"
        / "simple_policy_candidates_broad.parquet"
    )
    if not path.exists():
        return {"candidate_path": str(path), "present": False}
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns:
        return {"candidate_path": str(path), "present": True, "reason": "missing_timestamp"}
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    rows = df[
        (df["strategy_id"].astype(str) == _strategy_core(strategy_id))
        & (df["timestamp"] >= fold.policy_split)
        & (df["timestamp"] < fold.policy_end)
    ].copy()
    if rows.empty:
        return {"candidate_path": str(path), "present": True, "rows": 0}
    net = pd.to_numeric(rows.get("net_return"), errors="coerce")
    gross = pd.to_numeric(rows.get("gross_return"), errors="coerce")
    return {
        "candidate_path": str(path),
        "present": True,
        "rows": int(len(rows)),
        "net_return_sum": float(net.sum()),
        "mean_net_return": float(net.mean()),
        "gross_return_sum": float(gross.sum()),
        "win_rate": float((net > 0).mean()),
        "min_timestamp": rows["timestamp"].min().isoformat(),
        "max_timestamp": rows["timestamp"].max().isoformat(),
    }


def _summarise_fold(run_id: str, fold: FoldSpec, strategy_id: str) -> dict[str, Any]:
    metrics_path = DATA_ROOT / "artifacts" / run_id / "policy_optimisation_oos_metrics.json"
    out: dict[str, Any] = {
        "fold": fold.name,
        "run_id": run_id,
        "strategy_id": strategy_id,
        "train_end": _timestamp(fold.train_end),
        "policy_optimise_start": _timestamp(fold.policy_start),
        "policy_optimise_end": _timestamp(fold.policy_split),
        "policy_validation_start": _timestamp(fold.policy_split),
        "policy_validation_end": _timestamp(fold.policy_end),
        "metrics_path": str(metrics_path),
    }
    if not metrics_path.exists():
        out["status"] = "missing_metrics"
        return out
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    strategies = payload.get("strategies") or {}
    row = strategies.get(_strategy_core(strategy_id)) or strategies.get(strategy_id)
    if not isinstance(row, dict):
        out["status"] = "missing_strategy_metrics"
        out["available_strategies"] = sorted(str(k) for k in strategies)
        return out
    out["status"] = "ok"
    out["prediction_source"] = payload.get("prediction_source", {})
    out["policy_outer_split"] = row.get("policy_outer_split", {})
    out["outer_policy_validation_deployment_metrics"] = row.get(
        "outer_policy_validation_deployment_metrics", {}
    )
    out["final_policy_deployment_metrics"] = row.get(
        "final_policy_deployment_metrics", {}
    )
    out["deployment_threshold_metrics"] = row.get("deployment_threshold_metrics", {})
    out["validation_candidate_summary"] = _validation_candidate_summary(
        run_id, fold, strategy_id
    )
    return out


def _write_summary(
    *,
    experiment_id: str,
    source_run_id: str,
    feature_source_run_id: str,
    label_run_id: str,
    label_ablation_mode: str,
    label_weight_mode: str,
    label_weight_recipe: str,
    true_soft_labels: bool,
    disable_self_distillation: bool,
    label_policy_net_replay: bool,
    label_policy_net_replay_min_coverage: float | None,
    strategy_selection: dict[str, Any],
    folds: list[FoldSpec],
    strategy_id: Sequence[str] | str,
) -> Path:
    out_dir = DATA_ROOT / "reports" / experiment_id
    strategy_ids = _strategy_id_list(strategy_id)
    rows = [
        _summarise_fold(fold.run_id, fold, sid)
        for fold in folds
        for sid in strategy_ids
    ]
    summary = {
        "experiment_id": experiment_id,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "source_run_id": source_run_id,
        "feature_source_run_id": feature_source_run_id,
        "strategy_id": strategy_ids[0] if strategy_ids else "",
        "strategy_ids": strategy_ids,
        "strategy_selection": strategy_selection,
        "contract": {
            "model_pipeline": "train_base -> train_meta",
            "native_preset_source_run_id": source_run_id,
            "native_preset_params_only": False,
            "label_run_id": label_run_id,
            "label_ablation_mode": label_ablation_mode,
            "true_soft_labels": bool(true_soft_labels),
            "label_weight_mode": str(label_weight_mode),
            "label_weight_recipe": str(label_weight_recipe),
            "self_distillation_disabled": bool(disable_self_distillation),
            "label_policy_net_replay": bool(label_policy_net_replay),
            "label_policy_net_replay_min_coverage": (
                float(label_policy_net_replay_min_coverage)
                if label_policy_net_replay_min_coverage is not None
                else None
            ),
            "label_embargo_hours": LABEL_EMBARGO_HOURS,
            "portfolio_replay": False,
            "policy_metric_source": "outer_policy_validation_deployment_metrics",
            "precomputed_meta_oof_for_policy": False,
            "final_fit_policy_generation": False,
        },
        "folds": rows,
    }
    _json_dump(out_dir / "summary.json", summary)
    flat_rows = []
    for row in rows:
        metrics = row.get("outer_policy_validation_deployment_metrics") or {}
        cand = row.get("validation_candidate_summary") or {}
        flat_rows.append(
            {
                "fold": row.get("fold"),
                "run_id": row.get("run_id"),
                "strategy_id": row.get("strategy_id"),
                "status": row.get("status"),
                "train_end": row.get("train_end"),
                "validation_start": row.get("policy_validation_start"),
                "validation_end": row.get("policy_validation_end"),
                "oos_net_pnl": metrics.get("net_pnl"),
                "oos_n_trades": metrics.get("n_trades"),
                "oos_hit_rate": metrics.get("hit_rate"),
                "oos_mean_net_trade": metrics.get("mean_net_trade"),
                "oos_max_drawdown": metrics.get("max_drawdown"),
                "oos_sortino": metrics.get("sortino"),
                "candidate_validation_rows": cand.get("rows"),
                "candidate_validation_net_return_sum": cand.get("net_return_sum"),
                "candidate_validation_mean_net_return": cand.get("mean_net_return"),
                "candidate_validation_win_rate": cand.get("win_rate"),
            }
        )
    pd.DataFrame(flat_rows).to_csv(out_dir / "summary.csv", index=False)
    return out_dir / "summary.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--source-run-id", default=DEFAULT_SOURCE_RUN_ID)
    parser.add_argument("--feature-source-run-id", default=DEFAULT_FEATURE_SOURCE_RUN_ID)
    parser.add_argument(
        "--label-run-id",
        default="",
        help="Existing label artifact run id to train from. Defaults to <experiment-id>_labels.",
    )
    parser.add_argument(
        "--label-ablation-mode",
        default="",
        help="Optional label target mode passed to training, e.g. s10.",
    )
    parser.add_argument(
        "--label-weight-mode",
        choices=["legacy", "w0", "recipe"],
        default="legacy",
        help=(
            "Training label-weight recipe mode. Use 'w0' to disable label-weight "
            "recipes/HPO explicitly; use 'recipe' with --label-weight-recipe."
        ),
    )
    parser.add_argument(
        "--label-weight-recipe",
        default="",
        help="Recipe path used when --label-weight-mode=recipe.",
    )
    parser.add_argument(
        "--true-soft-labels",
        action="store_true",
        help="Train LGBM classifiers on soft targets instead of thresholded hard labels.",
    )
    parser.add_argument(
        "--disable-self-distillation",
        action="store_true",
        help="Disable LGBM OOF and sequential self-distillation for a cleaner label/weight ablation.",
    )
    parser.add_argument(
        "--label-policy-net-replay",
        action="store_true",
        help="Materialize __u_policy_net__/__r_policy_net__ during label generation.",
    )
    parser.add_argument(
        "--label-policy-net-replay-min-coverage",
        type=float,
        default=None,
        help="Optional minimum finite coverage for --label-policy-net-replay.",
    )
    parser.add_argument("--strategy-id", default="")
    parser.add_argument(
        "--selection-mode",
        choices=("overall", "best_per_side", "bidirectional"),
        default="overall",
        help=(
            "Strategy selection mode when --strategy-id is omitted. `overall` "
            "preserves the legacy one-head winner; `best_per_side` selects the "
            "best real June candidate for each requested side."
        ),
    )
    parser.add_argument(
        "--selection-sides",
        default="long,short",
        help="Comma-separated sides used by --selection-mode best_per_side/bidirectional.",
    )
    parser.add_argument("--policy-trials", type=int, default=None)
    parser.add_argument(
        "--include-july-fold",
        action="store_true",
        help=(
            "Add an explicit train-through-June / score-July fold. Disabled by "
            "default so existing Apr-Jun reports remain unchanged."
        ),
    )
    parser.add_argument(
        "--july-policy-end",
        default="",
        help=(
            "Optional UTC policy end for the July fold. Defaults to 2026-08-01 "
            "for a full July fold; use an earlier timestamp only for explicitly "
            "partial July diagnostics."
        ),
    )
    parser.add_argument(
        "--july-policy-split",
        default="",
        help=(
            "Optional UTC optimise/validation split for the July fold. Defaults "
            "to 2026-07-16, or to a bounded midpoint when the end is earlier."
        ),
    )
    parser.add_argument(
        "--only-folds",
        default="",
        help=(
            "Comma-separated fold names or aliases to run/prepare from the planned "
            "fold set. Example: --include-july-fold --only-folds july."
        ),
    )
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help=(
            "Generate/validate the shared label artifact, then stop before model "
            "training. Useful when preparing a new policy month."
        ),
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    source_run_id = str(args.source_run_id).strip()
    feature_source_run_id = str(args.feature_source_run_id).strip()
    experiment_id = str(args.experiment_id).strip()
    label_run_id = str(args.label_run_id or f"{experiment_id}_labels").strip()
    label_ablation_mode = str(args.label_ablation_mode or "").strip()
    label_weight_mode = str(args.label_weight_mode or "legacy").strip().lower()
    label_weight_recipe = str(args.label_weight_recipe or "").strip()
    explicit_strategy_ids = _parse_csv_arg(args.strategy_id)
    if explicit_strategy_ids:
        strategy_ids = explicit_strategy_ids
        selection = {
            "selection_mode": "explicit",
            "strategy_id": strategy_ids[0],
            "strategy_ids": strategy_ids,
            "selected_strategies": [],
        }
    else:
        selection = _select_june_strategy_set(
            source_run_id,
            selection_mode=str(args.selection_mode),
            sides=_parse_csv_arg(args.selection_sides) or ["long", "short"],
        )
        strategy_ids = [str(sid) for sid in selection.get("strategy_ids", []) if str(sid)]
    if not strategy_ids:
        raise RuntimeError("No strategy id selected.")
    strategy_id = strategy_ids[0]
    strategy_ids_csv = _strategy_ids_csv(strategy_ids)
    source_rows = _source_strategy_rows(source_run_id, strategy_ids)
    source_row = source_rows[0]
    symbols = _load_source_symbols(source_run_id)
    planned_folds = _folds(
        experiment_id,
        include_july_fold=bool(args.include_july_fold),
        july_policy_end=args.july_policy_end or None,
        july_policy_split=args.july_policy_split or None,
    )
    only_folds = _parse_csv_arg(args.only_folds)
    folds = _filter_folds(planned_folds, only_folds)
    required_label_max_ts = max(fold.policy_start for fold in folds)

    report_dir = DATA_ROOT / "reports" / experiment_id
    _json_dump(
        report_dir / "selection_manifest.json",
        {
            "experiment_id": experiment_id,
            "source_run_id": source_run_id,
            "feature_source_run_id": feature_source_run_id,
            "strategy_id": strategy_id,
            "strategy_ids": strategy_ids,
            "strategy_selection": selection,
            "source_strategy_registry_row": source_row,
            "source_strategy_registry_rows": source_rows,
            "symbol_count": len(symbols),
            "label_run_id": label_run_id,
            "label_ablation_mode": label_ablation_mode,
            "label_weight_mode": label_weight_mode,
            "label_weight_recipe": label_weight_recipe,
            "true_soft_labels": bool(args.true_soft_labels),
            "self_distillation_disabled": bool(args.disable_self_distillation),
            "label_policy_net_replay": bool(args.label_policy_net_replay),
            "label_policy_net_replay_min_coverage": (
                float(args.label_policy_net_replay_min_coverage)
                if args.label_policy_net_replay_min_coverage is not None
                else None
            ),
            "label_embargo_hours": LABEL_EMBARGO_HOURS,
            "include_july_fold": bool(args.include_july_fold),
            "july_policy_end": args.july_policy_end or None,
            "july_policy_split": args.july_policy_split or None,
            "only_folds": only_folds,
            "labels_only": bool(args.labels_only),
            "required_label_max_ts": _timestamp(required_label_max_ts),
            "planned_folds": [fold.__dict__ for fold in planned_folds],
            "folds": [fold.__dict__ for fold in folds],
        },
    )

    label_registry_path = _write_strategy_registry(label_run_id, source_rows)
    for fold in folds:
        registry_path = _write_strategy_registry(fold.run_id, source_rows)
        slice_plan_path = _write_slice_plan(
            run_id=fold.run_id,
            fold=fold,
            symbols=symbols,
            source_run_id=source_run_id,
            feature_source_run_id=feature_source_run_id,
            strategy_id=strategy_ids_csv,
        )
        _append(
            LOG_DIR / f"{experiment_id}.log",
            f"prepared {fold.run_id}: registry={registry_path} slice={slice_plan_path}",
        )

    if args.summary_only or args.prepare_only:
        summary_path = _write_summary(
            experiment_id=experiment_id,
            source_run_id=source_run_id,
            feature_source_run_id=feature_source_run_id,
            label_run_id=label_run_id,
            label_ablation_mode=label_ablation_mode,
            label_weight_mode=label_weight_mode,
            label_weight_recipe=label_weight_recipe,
            true_soft_labels=bool(args.true_soft_labels),
            disable_self_distillation=bool(args.disable_self_distillation),
            label_policy_net_replay=bool(args.label_policy_net_replay),
            label_policy_net_replay_min_coverage=args.label_policy_net_replay_min_coverage,
            strategy_selection=selection,
            folds=folds,
            strategy_id=strategy_ids,
        )
        print(summary_path)
        return 0

    labels_log_path = LOG_DIR / f"{experiment_id}_labels.log"
    if not _labels_ready_all(
        label_run_id,
        strategy_ids,
        min_label_max_ts=required_label_max_ts,
    ):
        _run_step(
            f"{label_run_id}_labels",
            _pipeline_cmd("labels", label_run_id, feature_source_run_id),
            _labels_env(
                label_run_id=label_run_id,
                source_run_id=source_run_id,
                feature_source_run_id=feature_source_run_id,
                label_ablation_mode=label_ablation_mode,
                label_policy_net_replay=bool(args.label_policy_net_replay),
                label_policy_net_replay_min_coverage=args.label_policy_net_replay_min_coverage,
                registry_path=label_registry_path,
                strategy_id=strategy_ids_csv,
            ),
            labels_log_path,
        )
        if not _labels_ready_all(
            label_run_id,
            strategy_ids,
            min_label_max_ts=required_label_max_ts,
        ):
            missing_labels = _missing_label_ids(
                label_run_id,
                strategy_ids,
                min_label_max_ts=required_label_max_ts,
            )
            raise RuntimeError(
                f"{label_run_id} labels finished without required train_*_5 "
                f"datasets covering {_timestamp(required_label_max_ts)} for {missing_labels}."
            )
    else:
        _append(
            labels_log_path,
            f"SKIP {label_run_id}_labels already ready through "
            f"{_timestamp(required_label_max_ts)}",
        )

    if args.labels_only:
        summary_path = _write_summary(
            experiment_id=experiment_id,
            source_run_id=source_run_id,
            feature_source_run_id=feature_source_run_id,
            label_run_id=label_run_id,
            label_ablation_mode=label_ablation_mode,
            label_weight_mode=label_weight_mode,
            label_weight_recipe=label_weight_recipe,
            true_soft_labels=bool(args.true_soft_labels),
            disable_self_distillation=bool(args.disable_self_distillation),
            label_policy_net_replay=bool(args.label_policy_net_replay),
            label_policy_net_replay_min_coverage=args.label_policy_net_replay_min_coverage,
            strategy_selection=selection,
            folds=folds,
            strategy_id=strategy_ids,
        )
        print(summary_path)
        return 0

    for fold in folds:
        log_path = LOG_DIR / f"{experiment_id}_{fold.name}.log"
        registry_path = DATA_ROOT / "artifacts" / fold.run_id / "strategy_registry" / "selected_single_head_strategy_registry.csv"
        slice_plan_path = DATA_ROOT / "artifacts" / fold.run_id / "slices" / "slice_plan.json"
        env = _base_env(
            run_id=fold.run_id,
            source_run_id=source_run_id,
            feature_source_run_id=feature_source_run_id,
            label_run_id=label_run_id,
            label_ablation_mode=label_ablation_mode,
            label_weight_mode=label_weight_mode,
            label_weight_recipe=label_weight_recipe,
            true_soft_labels=bool(args.true_soft_labels),
            disable_self_distillation=bool(args.disable_self_distillation),
            registry_path=registry_path,
            slice_plan_path=slice_plan_path,
            strategy_id=strategy_ids_csv,
            policy_trials=args.policy_trials,
        )
        if not _stage_ready_all(fold.run_id, "train_base", strategy_ids):
            _run_step(
                f"{fold.run_id}_train_base",
                _pipeline_cmd("train_base", fold.run_id, feature_source_run_id),
                env,
                log_path,
            )
            _mark_stage_done(fold.run_id, "train_base")
            _require_stage_ready_all(fold.run_id, "train_base", strategy_ids)
        else:
            _append(log_path, f"SKIP {fold.run_id}_train_base already ready")
        if not _stage_ready_all(fold.run_id, "train_meta", strategy_ids):
            _run_step(
                f"{fold.run_id}_train_meta",
                _pipeline_cmd("train_meta", fold.run_id, feature_source_run_id),
                env,
                log_path,
            )
            _mark_stage_done(fold.run_id, "train_meta")
            _require_stage_ready_all(fold.run_id, "train_meta", strategy_ids)
        else:
            _append(log_path, f"SKIP {fold.run_id}_train_meta already ready")
        if not _stage_ready_all(fold.run_id, "policy_oos", strategy_ids):
            _run_step(
                f"{fold.run_id}_policy_oos",
                _policy_oos_cmd(fold.run_id, strategy_ids),
                env,
                log_path,
            )
            _mark_stage_done(fold.run_id, "policy_oos")
            _require_stage_ready_all(fold.run_id, "policy_oos", strategy_ids)
        else:
            _append(log_path, f"SKIP {fold.run_id}_policy_oos already ready")
        if not _stage_ready_all(
            fold.run_id, "simple_policy", strategy_ids, log_path=log_path
        ):
            _run_step(
                f"{fold.run_id}_simple_policy",
                _simple_policy_cmd(fold.run_id, strategy_ids),
                env,
                log_path,
            )
            _mark_stage_done(fold.run_id, "simple_policy")
            _require_stage_ready_all(
                fold.run_id, "simple_policy", strategy_ids, log_path=log_path
            )
        else:
            _append(log_path, f"SKIP {fold.run_id}_simple_policy already ready")

    summary_path = _write_summary(
        experiment_id=experiment_id,
        source_run_id=source_run_id,
        feature_source_run_id=feature_source_run_id,
        label_run_id=label_run_id,
        label_ablation_mode=label_ablation_mode,
        label_weight_mode=label_weight_mode,
        label_weight_recipe=label_weight_recipe,
        true_soft_labels=bool(args.true_soft_labels),
        disable_self_distillation=bool(args.disable_self_distillation),
        label_policy_net_replay=bool(args.label_policy_net_replay),
        label_policy_net_replay_min_coverage=args.label_policy_net_replay_min_coverage,
        strategy_selection=selection,
        folds=folds,
        strategy_id=strategy_ids,
    )
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
