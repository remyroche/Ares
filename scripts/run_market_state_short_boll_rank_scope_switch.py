#!/usr/bin/env python3
"""Replay a shadow market-state short_boll rank-scope switch.

This is deliberately not a production controller.  It tests the hypothesis that
June's issue is short_boll rank-scope eligibility rather than auction ordering:

* short_asset always uses the current T1 contract;
* short_boll can use either the current T1 within-timestamp rank or the causal
  global-over-time challenger rank;
* the switch is driven only by the already-fitted market-state head-priority
  schedule.

No q-fail, HeadHealth, threshold lowering, sizing, labels, or policy parameters
are changed.
"""

from __future__ import annotations

import argparse
import hashlib
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

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    fit_hierarchical_ev_curves,
    replay_candidates,
)
from scripts import run_market_state_threshold_controller as mstc  # noqa: E402
from scripts.run_market_state_head_priority_modulation import apply_head_priority_schedule  # noqa: E402
from scripts.run_market_state_head_priority_learning import DEFAULT_TRAIN_DEPLOYABLE  # noqa: E402


DEFAULT_T1_DIR = Path(
    "data_perp/artifacts/reliability_blend_T1_repaired_static_baseline_20260625_jun15_22"
)
DEFAULT_GLOBAL_DIR = Path(
    "data_perp/artifacts/reliability_blend_T1_global_rank_challenger_20260626_jun15_22_v1"
)
DEFAULT_PRIORITY_DIR = Path(
    "data_perp/reports/market_state_head_priority_learning_actionaware_20260626_jun15_22"
)
DEFAULT_POLICY_MANIFEST = Path(
    "data_perp/reports/reliability_blend_component_arm_portfolio_ablation_20260625/"
    "A0_anchor_only/portfolio_policy_ablation_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/market_state_short_boll_rank_scope_switch_20260626_jun15_22"
)
DEFAULT_CANDIDATE_FILE_NAME = "simple_policy_candidates_broad.parquet"
RANK_CONTRACT_MUTABLE_COLUMNS = {
    "normalized_rank_score",
    "strategy_rank_pct",
    "policy_rank_pct",
    "rank_pct",
    "rank_contract_source",
    "rank_scope",
    "short_boll_rank_scope",
    "rank_scope_arm",
    "deployment_rank_threshold",
}
BLENDABLE_RANK_COLUMNS = (
    "normalized_rank_score",
    "strategy_rank_pct",
    "policy_rank_pct",
    "rank_pct",
)
CRITICAL_PARITY_COLUMNS = (
    "timestamp",
    "symbol",
    "side",
    "strategy_id",
    "head",
    "calibrated_score",
    "entry_price",
    "exit_timestamp",
    "exit_price",
    "net_return",
    "gross_return",
    "holding_bars",
    "simple_policy_exit_reason",
    "base_strategy_threshold",
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


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or path.is_dir():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _load_candidates(path: Path) -> pd.DataFrame:
    out = pd.read_parquet(path)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if "head" not in out.columns:
        out["head"] = out["strategy_id"].map(_head_from_strategy)
    return mstc.normalise_candidate_table(out)


def _candidate_file(simple_policy_dir: Path, file_name: str = DEFAULT_CANDIDATE_FILE_NAME) -> Path:
    """Resolve the candidate ledger used for rank-contract switching.

    Rank-scope comparisons must start from the broad candidate universe.  A
    deployable-only ledger has already been threshold-filtered under its own
    rank contract, so using it would bias the switch by silently removing rows
    that only become eligible under the alternate contract.
    """

    requested = simple_policy_dir / str(file_name)
    if requested.exists():
        return requested
    broad = simple_policy_dir / DEFAULT_CANDIDATE_FILE_NAME
    if broad.exists():
        return broad
    fallback = simple_policy_dir / "simple_policy_candidates.parquet"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(requested)


def _decision_key(df: pd.DataFrame) -> pd.Series:
    cols = [col for col in ("timestamp", "symbol", "strategy_id", "side", "head") if col in df.columns]
    if not cols:
        return pd.Series(np.arange(len(df)), index=df.index).astype(str)
    values: list[pd.Series] = []
    for col in cols:
        if col == "timestamp":
            values.append(pd.to_datetime(df[col], utc=True, errors="coerce").astype(str))
        else:
            values.append(df[col].astype(str))
    out = values[0]
    for value in values[1:]:
        out = out.str.cat(value, sep="|")
    return out


def _rank_contract_parity_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["_decision_key"] = _decision_key(out)
    if out["_decision_key"].duplicated().any():
        dupes = int(out["_decision_key"].duplicated().sum())
        raise ValueError(f"candidate table has duplicate decision keys: {dupes}")
    return out.set_index("_decision_key", drop=False)


def assert_rank_contract_candidate_parity(
    t1_candidates: pd.DataFrame,
    global_candidates: pd.DataFrame,
    *,
    head: str = "short_boll",
    fail_closed: bool = True,
) -> dict[str, Any]:
    """Verify that rank-contract arms differ only by rank-like columns.

    The shadow switch is interpretable only if both ledgers contain the same
    candidate keys and the same deployable economics for the switched head.
    """

    t1 = t1_candidates.loc[t1_candidates["head"].astype(str).eq(str(head))].copy()
    glob = global_candidates.loc[global_candidates["head"].astype(str).eq(str(head))].copy()
    t1_idx = _rank_contract_parity_frame(t1)
    glob_idx = _rank_contract_parity_frame(glob)
    t1_keys = set(t1_idx.index)
    global_keys = set(glob_idx.index)
    missing_in_global = sorted(t1_keys.difference(global_keys))
    missing_in_t1 = sorted(global_keys.difference(t1_keys))
    common = sorted(t1_keys.intersection(global_keys))
    failures: list[str] = []
    if missing_in_global:
        failures.append(f"{len(missing_in_global)} {head} T1 candidate keys missing from global ledger")
    if missing_in_t1:
        failures.append(f"{len(missing_in_t1)} {head} global candidate keys missing from T1 ledger")

    compared_columns: list[str] = []
    mismatch_counts: dict[str, int] = {}
    if common:
        for col in CRITICAL_PARITY_COLUMNS:
            if col in RANK_CONTRACT_MUTABLE_COLUMNS:
                continue
            if col not in t1_idx.columns or col not in glob_idx.columns:
                continue
            lhs = t1_idx.loc[common, col]
            rhs = glob_idx.loc[common, col]
            compared_columns.append(col)
            if pd.api.types.is_numeric_dtype(lhs) or pd.api.types.is_numeric_dtype(rhs):
                lhs_num = pd.to_numeric(lhs, errors="coerce")
                rhs_num = pd.to_numeric(rhs, errors="coerce")
                same = np.isclose(
                    lhs_num.to_numpy(dtype=float),
                    rhs_num.to_numpy(dtype=float),
                    rtol=1e-10,
                    atol=1e-12,
                    equal_nan=True,
                )
                mismatch = int((~same).sum())
            else:
                mismatch = int((lhs.astype(str).fillna("<NA>") != rhs.astype(str).fillna("<NA>")).sum())
            if mismatch:
                mismatch_counts[col] = mismatch
                failures.append(f"{head} non-rank column mismatch: {col} ({mismatch} rows)")

    report = {
        "head": str(head),
        "t1_candidate_rows": int(len(t1)),
        "global_candidate_rows": int(len(glob)),
        "common_candidate_rows": int(len(common)),
        "missing_in_global": int(len(missing_in_global)),
        "missing_in_t1": int(len(missing_in_t1)),
        "compared_columns": compared_columns,
        "mismatch_counts": mismatch_counts,
        "passed": not failures,
        "failures": failures,
    }
    if failures and fail_closed:
        raise ValueError("; ".join(failures[:5]))
    return report


def load_priority_switch_schedule(
    schedule_path: Path,
    *,
    margin: float,
    blend_scale: float | None = None,
) -> pd.DataFrame:
    schedule = pd.read_parquet(schedule_path)
    schedule["timestamp"] = pd.to_datetime(schedule["timestamp"], utc=True, errors="coerce")
    required = {"timestamp", "head", "portfolio_priority_adjustment"}
    missing = required.difference(schedule.columns)
    if missing:
        raise ValueError(f"priority schedule missing columns: {sorted(missing)}")
    if schedule.duplicated(["timestamp", "head"]).any():
        raise ValueError("priority schedule has duplicate timestamp/head rows")
    pivot = schedule.pivot(
        index="timestamp",
        columns="head",
        values="portfolio_priority_adjustment",
    )
    for head in ("short_asset", "short_boll"):
        if head not in pivot.columns:
            raise ValueError(f"priority schedule missing head: {head}")
    out = pivot.reset_index().rename_axis(None, axis=1)
    out["short_boll_minus_short_asset_priority"] = (
        pd.to_numeric(out["short_boll"], errors="coerce")
        - pd.to_numeric(out["short_asset"], errors="coerce")
    )
    out["short_boll_rank_scope"] = np.where(
        out["short_boll_minus_short_asset_priority"] > float(margin),
        "timestamp_rank",
        "global_rank",
    )
    if blend_scale is None:
        out["short_boll_timestamp_weight"] = np.where(
            out["short_boll_rank_scope"].astype(str).eq("timestamp_rank"),
            1.0,
            0.0,
        )
    else:
        scale = float(blend_scale)
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError("blend_scale must be non-negative")
        diff = pd.to_numeric(out["short_boll_minus_short_asset_priority"], errors="coerce")
        if scale <= 1e-12:
            out["short_boll_timestamp_weight"] = np.where(diff > float(margin), 1.0, 0.0)
        else:
            z = ((diff - float(margin)) / scale).clip(lower=-30.0, upper=30.0)
            out["short_boll_timestamp_weight"] = 1.0 / (1.0 + np.exp(-z))
    return out[
        [
            "timestamp",
            "short_asset",
            "short_boll",
            "short_boll_minus_short_asset_priority",
            "short_boll_rank_scope",
            "short_boll_timestamp_weight",
        ]
    ].reset_index(drop=True)


def load_rank_reference_router_schedule(schedule_path: Path) -> pd.DataFrame:
    """Load a formal rank-reference router schedule.

    The router schedule is produced by
    build_market_state_rank_reference_router.py.  It is intentionally a
    timestamp-level artifact so replay can distinguish state scoring from
    portfolio replay.
    """

    schedule = pd.read_parquet(schedule_path)
    required = {"timestamp", "short_boll_rank_scope", "short_boll_timestamp_weight"}
    missing = sorted(required.difference(schedule.columns))
    if missing:
        raise ValueError(f"rank-reference router schedule missing columns: {missing}")
    out = schedule.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if out["timestamp"].isna().any():
        raise ValueError("rank-reference router schedule contains nonparseable timestamps")
    if out["timestamp"].duplicated().any():
        raise ValueError("rank-reference router schedule has duplicate timestamps")
    out["short_boll_timestamp_weight"] = pd.to_numeric(
        out["short_boll_timestamp_weight"],
        errors="coerce",
    )
    if out["short_boll_timestamp_weight"].isna().any():
        raise ValueError("rank-reference router schedule has nonfinite timestamp weights")
    if not out["short_boll_timestamp_weight"].between(0.0, 1.0).all():
        raise ValueError("rank-reference router schedule weights must be in [0, 1]")
    out["short_boll_rank_scope"] = out["short_boll_rank_scope"].astype(str)
    valid_scopes = {"timestamp_rank", "global_rank", "state_blend"}
    unknown = sorted(set(out["short_boll_rank_scope"]).difference(valid_scopes))
    if unknown:
        raise ValueError(f"rank-reference router schedule has unknown scopes: {unknown}")
    return out.reset_index(drop=True)


def load_priority_action_schedule(schedule_path: Path) -> pd.DataFrame:
    schedule = pd.read_parquet(schedule_path)
    required = {"timestamp", "head", "portfolio_priority_adjustment"}
    missing = sorted(required.difference(schedule.columns))
    if missing:
        raise ValueError(f"priority action schedule missing columns: {missing}")
    out = schedule.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if out["timestamp"].isna().any():
        raise ValueError("priority action schedule contains nonparseable timestamps")
    if out.duplicated(["timestamp", "head"]).any():
        raise ValueError("priority action schedule has duplicate timestamp/head rows")
    if "portfolio_priority_multiplier" not in out.columns:
        out["portfolio_priority_multiplier"] = 1.0
    return out.reset_index(drop=True)


def _blend_short_boll_rank_columns(
    timestamp_rank: pd.DataFrame,
    global_rank: pd.DataFrame,
    schedule_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Blend global and timestamp rank columns for short_boll by timestamp."""

    t1 = _rank_contract_parity_frame(timestamp_rank)
    glob = _rank_contract_parity_frame(global_rank)
    common = sorted(set(t1.index).intersection(set(glob.index)))
    if len(common) != len(t1) or len(common) != len(glob):
        raise ValueError("cannot blend rank columns without exact short_boll key parity")

    weights = schedule_rows.copy()
    weights["timestamp"] = pd.to_datetime(weights["timestamp"], utc=True, errors="coerce")
    weight_map = (
        weights.dropna(subset=["timestamp"])
        .drop_duplicates("timestamp")
        .set_index("timestamp")["short_boll_timestamp_weight"]
        .to_dict()
    )
    out = glob.loc[common].copy()
    ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    weight_series = ts.map(weight_map)
    if weight_series.isna().any():
        raise ValueError(f"missing short_boll blend weights for {int(weight_series.isna().sum())} rows")
    weight = pd.to_numeric(weight_series, errors="coerce").clip(lower=0.0, upper=1.0).to_numpy(dtype=float)

    for col in BLENDABLE_RANK_COLUMNS:
        if col not in t1.columns or col not in glob.columns:
            continue
        timestamp_values = pd.to_numeric(t1.loc[common, col], errors="coerce").to_numpy(dtype=float)
        global_values = pd.to_numeric(glob.loc[common, col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(timestamp_values) & np.isfinite(global_values)
        blended = global_values.copy()
        blended[finite] = global_values[finite] + weight[finite] * (
            timestamp_values[finite] - global_values[finite]
        )
        out[col] = blended

    out["short_boll_rank_scope"] = "state_blend"
    out["rank_scope"] = "state_blend"
    out["rank_contract_source"] = "state_blend_global_to_timestamp"
    out["short_boll_timestamp_weight"] = weight
    return out.reset_index(drop=True)


def build_rank_scope_candidates(
    t1_candidates: pd.DataFrame,
    global_candidates: pd.DataFrame,
    switch_schedule: pd.DataFrame | None,
    *,
    arm: str,
    fail_closed: bool = True,
    validate_candidate_parity: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a candidate table where only short_boll rank scope may change."""
    t1 = t1_candidates.copy()
    glob = global_candidates.copy()
    t1["timestamp"] = pd.to_datetime(t1["timestamp"], utc=True, errors="coerce")
    glob["timestamp"] = pd.to_datetime(glob["timestamp"], utc=True, errors="coerce")
    t1_short_asset = t1.loc[t1["head"].astype(str).eq("short_asset")].copy()
    t1_short_boll = t1.loc[t1["head"].astype(str).eq("short_boll")].copy()
    global_short_boll = glob.loc[glob["head"].astype(str).eq("short_boll")].copy()
    if validate_candidate_parity and arm in {
        "R1_global_short_boll",
        "R2_state_switch_short_boll",
        "R3_state_blended_short_boll",
    }:
        assert_rank_contract_candidate_parity(
            t1,
            glob,
            head="short_boll",
            fail_closed=fail_closed,
        )

    if arm == "R0_t1_timestamp_short_boll":
        short_boll = t1_short_boll
        schedule_rows = pd.DataFrame(
            {
                "timestamp": sorted(t1["timestamp"].dropna().unique()),
                "short_boll_rank_scope": "timestamp_rank",
            }
        )
    elif arm == "R1_global_short_boll":
        short_boll = global_short_boll
        schedule_rows = pd.DataFrame(
            {
                "timestamp": sorted(t1["timestamp"].dropna().unique()),
                "short_boll_rank_scope": "global_rank",
            }
        )
    elif arm == "R2_state_switch_short_boll":
        if switch_schedule is None or switch_schedule.empty:
            raise ValueError("state switch arm requires a non-empty switch schedule")
        sched = switch_schedule.copy()
        sched["timestamp"] = pd.to_datetime(sched["timestamp"], utc=True, errors="coerce")
        needed_ts = set(t1["timestamp"].dropna().unique())
        available_ts = set(sched["timestamp"].dropna().unique())
        missing_ts = sorted(needed_ts.difference(available_ts))
        if missing_ts and fail_closed:
            raise ValueError(f"switch schedule missing {len(missing_ts)} candidate timestamps")
        sched = sched.loc[sched["timestamp"].isin(needed_ts)].copy()
        timestamp_scope = set(
            sched.loc[
                sched["short_boll_rank_scope"].astype(str).eq("timestamp_rank"),
                "timestamp",
            ]
        )
        global_scope = set(
            sched.loc[
                sched["short_boll_rank_scope"].astype(str).eq("global_rank"),
                "timestamp",
            ]
        )
        short_boll = pd.concat(
            [
                t1_short_boll.loc[t1_short_boll["timestamp"].isin(timestamp_scope)],
                global_short_boll.loc[global_short_boll["timestamp"].isin(global_scope)],
            ],
            ignore_index=True,
        )
        schedule_rows = sched[["timestamp", "short_boll_rank_scope"]].copy()
    elif arm == "R3_state_blended_short_boll":
        if switch_schedule is None or switch_schedule.empty:
            raise ValueError("state blend arm requires a non-empty switch schedule")
        sched = switch_schedule.copy()
        sched["timestamp"] = pd.to_datetime(sched["timestamp"], utc=True, errors="coerce")
        if "short_boll_timestamp_weight" not in sched.columns:
            raise ValueError("state blend arm requires short_boll_timestamp_weight")
        needed_ts = set(t1["timestamp"].dropna().unique())
        available_ts = set(sched["timestamp"].dropna().unique())
        missing_ts = sorted(needed_ts.difference(available_ts))
        if missing_ts and fail_closed:
            raise ValueError(f"blend schedule missing {len(missing_ts)} candidate timestamps")
        sched = sched.loc[sched["timestamp"].isin(needed_ts)].copy()
        short_boll = _blend_short_boll_rank_columns(
            t1_short_boll,
            global_short_boll,
            sched[["timestamp", "short_boll_timestamp_weight"]].copy(),
        )
        schedule_rows = sched[["timestamp", "short_boll_rank_scope", "short_boll_timestamp_weight"]].copy()
        schedule_rows["short_boll_rank_scope"] = "state_blend"
    else:
        raise ValueError(f"unknown rank-scope arm: {arm}")

    out = pd.concat([t1_short_asset, short_boll], ignore_index=True)
    out["rank_scope_arm"] = arm
    if arm == "R0_t1_timestamp_short_boll":
        out["short_boll_rank_scope"] = np.where(out["head"].eq("short_boll"), "timestamp_rank", "t1_short_asset")
    elif arm == "R1_global_short_boll":
        out["short_boll_rank_scope"] = np.where(out["head"].eq("short_boll"), "global_rank", "t1_short_asset")
    elif arm == "R2_state_switch_short_boll":
        scope_map = schedule_rows.set_index("timestamp")["short_boll_rank_scope"].to_dict()
        out["short_boll_rank_scope"] = np.where(
            out["head"].eq("short_boll"),
            out["timestamp"].map(scope_map).fillna("missing"),
            "t1_short_asset",
        )
    else:
        out["short_boll_rank_scope"] = np.where(out["head"].eq("short_boll"), "state_blend", "t1_short_asset")
        weight_map = schedule_rows.set_index("timestamp")["short_boll_timestamp_weight"].to_dict()
        out["short_boll_timestamp_weight"] = np.where(
            out["head"].eq("short_boll"),
            out["timestamp"].map(weight_map),
            0.0,
        )
    decision_keys = _decision_key(out)
    if decision_keys.duplicated().any():
        dupes = int(decision_keys.duplicated().sum())
        raise ValueError(f"rank-scope candidate table has duplicate decision keys: {dupes}")
    out = out.sort_values(["timestamp", "head", "normalized_rank_score"], ascending=[True, True, False])
    return out.reset_index(drop=True), schedule_rows.reset_index(drop=True)


def build_rank_scope_priority_candidates(
    t1_candidates: pd.DataFrame,
    global_candidates: pd.DataFrame,
    switch_schedule: pd.DataFrame,
    priority_schedule: pd.DataFrame,
    *,
    rank_arm: str,
    combo_arm: str,
    fail_closed: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    candidates, rank_schedule = build_rank_scope_candidates(
        t1_candidates,
        global_candidates,
        switch_schedule,
        arm=rank_arm,
        fail_closed=fail_closed,
        validate_candidate_parity=False,
    )
    out, coverage = apply_head_priority_schedule(
        candidates,
        priority_schedule,
        fail_closed=fail_closed,
    )
    out["rank_scope_arm"] = combo_arm
    out["priority_actions_applied"] = True
    rank_schedule = rank_schedule.copy()
    rank_schedule["priority_actions_applied"] = True
    rank_schedule["source_rank_arm"] = rank_arm
    return out, rank_schedule, coverage


def _replay(
    *,
    arm: str,
    candidates: pd.DataFrame,
    train_deployable: pd.DataFrame,
    params: Any,
    market_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ev_curve = fit_hierarchical_ev_curves(train_deployable)
    decisions, equity, metrics = replay_candidates(
        candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=market_mode,
    )
    accepted = mstc._accepted_trades(candidates, decisions)
    if not accepted.empty:
        accepted["arm"] = arm
    summary = pd.DataFrame([mstc._metrics_row(arm, metrics, accepted, schedule=None)])
    by_head = mstc._by_head(arm, accepted)
    return decisions, equity, accepted, summary, by_head


def _accepted_overlap(accepted_by_arm: dict[str, pd.DataFrame], baseline_arm: str) -> pd.DataFrame:
    base = accepted_by_arm.get(baseline_arm, pd.DataFrame())
    base_keys = set(_decision_key(base)) if not base.empty else set()
    rows = []
    for arm, frame in accepted_by_arm.items():
        keys = set(_decision_key(frame)) if not frame.empty else set()
        union = base_keys | keys
        inter = base_keys & keys
        rows.append(
            {
                "arm": arm,
                "baseline_accepted": int(len(base_keys)),
                "arm_accepted": int(len(keys)),
                "intersection": int(len(inter)),
                "union": int(len(union)),
                "jaccard_vs_baseline": float(len(inter) / len(union)) if union else 1.0,
                "baseline_only": int(len(base_keys - keys)),
                "arm_only": int(len(keys - base_keys)),
            }
        )
    return pd.DataFrame(rows)


def _schedule_summary(schedule: pd.DataFrame) -> pd.DataFrame:
    if schedule.empty or "short_boll_rank_scope" not in schedule.columns:
        return pd.DataFrame()
    out = (
        schedule.groupby("short_boll_rank_scope", observed=True)
        .agg(timestamp_count=("timestamp", "nunique"))
        .reset_index()
    )
    out["timestamp_share"] = out["timestamp_count"] / max(float(out["timestamp_count"].sum()), 1.0)
    return out


def _render_report(
    *,
    manifest: dict[str, Any],
    summary: pd.DataFrame,
    by_head: pd.DataFrame,
    overlap: pd.DataFrame,
    schedule_summary: pd.DataFrame,
) -> str:
    lines = [
        "# Market-State Short-Boll Rank-Scope Switch",
        "",
        "This is a shadow replay. It keeps short_asset on the current T1 contract and only tests whether market-state priority can choose short_boll timestamp-rank versus global-rank eligibility.",
        "",
        "## Contract",
        "",
        "- q-fail: disabled",
        "- HeadHealth: disabled",
        "- Threshold lowering: disabled",
        "- Portfolio params: unchanged",
        "- short_asset rank contract: current T1",
        "",
        "## Replay Summary",
        "",
        "| arm | trades | net_pnl | gross_pnl | cost_pnl | full_sl_rate | timeout_rate | worst_24h_net_pnl |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['arm']} | {int(row['trade_count'])} | "
            f"{float(row['net_pnl']):.6f} | {float(row['gross_pnl']):.6f} | "
            f"{float(row['cost_pnl']):.6f} | {float(row['full_sl_rate']):.6f} | "
            f"{float(row['timeout_rate']):.6f} | {float(row['worst_24h_net_pnl']):.6f} |"
        )
    lines.extend(["", "## By Head", ""])
    lines.append(by_head.to_markdown(index=False) if not by_head.empty else "_No by-head rows._")
    lines.extend(["", "## Accepted Overlap Vs T1", ""])
    lines.append(overlap.to_markdown(index=False) if not overlap.empty else "_No overlap rows._")
    lines.extend(["", "## State Switch Schedule", ""])
    lines.append(schedule_summary.to_markdown(index=False) if not schedule_summary.empty else "_No switch schedule rows._")
    lines.extend(["", "## Interpretation", ""])
    lines.append(
        "If the state switch mostly chooses timestamp_rank and reproduces T1, then the market-state signal is not yet a better general rank contract; it is just rediscovering the June short_boll repair."
    )
    lines.append(
        "If the state switch or state blend improves over T1 while retaining material global-rank weight, it becomes a valid shadow candidate for a future rank-contract controller."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1-artifact-dir", type=Path, default=DEFAULT_T1_DIR)
    parser.add_argument("--global-artifact-dir", type=Path, default=DEFAULT_GLOBAL_DIR)
    parser.add_argument("--priority-dir", type=Path, default=DEFAULT_PRIORITY_DIR)
    parser.add_argument(
        "--router-schedule",
        type=Path,
        help=(
            "Optional formal rank-reference router schedule produced by "
            "build_market_state_rank_reference_router.py. When provided, "
            "--priority-dir/head_priority_learned_schedule.parquet is not "
            "used for routing."
        ),
    )
    parser.add_argument("--candidate-file-name", default=DEFAULT_CANDIDATE_FILE_NAME)
    parser.add_argument("--train-deployable-candidates", type=Path, default=DEFAULT_TRAIN_DEPLOYABLE)
    parser.add_argument("--policy-manifest", type=Path, default=DEFAULT_POLICY_MANIFEST)
    parser.add_argument("--policy-variant", default="refit_bar4_strategy_bar2")
    parser.add_argument(
        "--apply-priority-actions",
        action="store_true",
        help=(
            "Also replay composed shadow arms that apply the learned bounded "
            "auction-priority schedule after rank-reference resolution."
        ),
    )
    parser.add_argument(
        "--priority-action-schedule",
        type=Path,
        help=(
            "Optional priority action schedule. Defaults to "
            "--priority-dir/head_priority_learned_schedule.parquet when "
            "--apply-priority-actions is set."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--switch-margin", type=float, default=0.0)
    parser.add_argument(
        "--blend-scale",
        type=float,
        default=0.05,
        help=(
            "Sigmoid scale for R3 state-blended short_boll rank. "
            "Use 0 for a hard 0/1 blend around --switch-margin."
        ),
    )
    parser.add_argument("--market-mode", default="perps")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t1_path = _candidate_file(args.t1_artifact_dir / "simple_policy_optimiser", args.candidate_file_name)
    global_path = _candidate_file(args.global_artifact_dir / "simple_policy_optimiser", args.candidate_file_name)
    schedule_path = (
        args.router_schedule
        if args.router_schedule is not None
        else args.priority_dir / "head_priority_learned_schedule.parquet"
    )
    t1_candidates = _load_candidates(t1_path)
    global_candidates = _load_candidates(global_path)
    train_deployable = _load_candidates(args.train_deployable_candidates)
    params, policy_payload = mstc._load_policy_params(args.policy_manifest, args.policy_variant)
    if args.router_schedule is not None:
        switch_schedule = load_rank_reference_router_schedule(schedule_path)
        schedule_source = "rank_reference_router_schedule"
    else:
        switch_schedule = load_priority_switch_schedule(
            schedule_path,
            margin=float(args.switch_margin),
            blend_scale=float(args.blend_scale),
        )
        schedule_source = "head_priority_learned_schedule"
    priority_action_path = (
        args.priority_action_schedule
        if args.priority_action_schedule is not None
        else args.priority_dir / "head_priority_learned_schedule.parquet"
    )
    priority_action_schedule = (
        load_priority_action_schedule(priority_action_path)
        if bool(args.apply_priority_actions)
        else pd.DataFrame()
    )
    parity_report = assert_rank_contract_candidate_parity(
        t1_candidates,
        global_candidates,
        head="short_boll",
        fail_closed=True,
    )

    arms = [
        "R0_t1_timestamp_short_boll",
        "R1_global_short_boll",
        "R2_state_switch_short_boll",
        "R3_state_blended_short_boll",
    ]
    if bool(args.apply_priority_actions):
        arms.extend(
            [
                "R4_t1_timestamp_plus_priority",
                "R5_state_blended_plus_priority",
            ]
        )
    accepted_by_arm: dict[str, pd.DataFrame] = {}
    summary_frames: list[pd.DataFrame] = []
    by_head_frames: list[pd.DataFrame] = []
    schedule_frames: list[pd.DataFrame] = []

    for arm in arms:
        priority_coverage: dict[str, Any] | None = None
        if arm == "R4_t1_timestamp_plus_priority":
            candidates, arm_schedule, priority_coverage = build_rank_scope_priority_candidates(
                t1_candidates,
                global_candidates,
                switch_schedule,
                priority_action_schedule,
                rank_arm="R0_t1_timestamp_short_boll",
                combo_arm=arm,
            )
        elif arm == "R5_state_blended_plus_priority":
            candidates, arm_schedule, priority_coverage = build_rank_scope_priority_candidates(
                t1_candidates,
                global_candidates,
                switch_schedule,
                priority_action_schedule,
                rank_arm="R3_state_blended_short_boll",
                combo_arm=arm,
            )
        else:
            candidates, arm_schedule = build_rank_scope_candidates(
                t1_candidates,
                global_candidates,
                switch_schedule,
                arm=arm,
                validate_candidate_parity=False,
            )
        decisions, equity, accepted, summary, by_head = _replay(
            arm=arm,
            candidates=candidates,
            train_deployable=train_deployable,
            params=params,
            market_mode=str(args.market_mode),
        )
        accepted_by_arm[arm] = accepted
        summary_frames.append(summary)
        by_head_frames.append(by_head)
        arm_schedule["arm"] = arm
        if priority_coverage is not None:
            for key, value in priority_coverage.items():
                arm_schedule[f"priority_{key}"] = value
        schedule_frames.append(arm_schedule)
        candidates.to_parquet(args.output_dir / f"{arm}_candidates.parquet", index=False)
        decisions.to_parquet(args.output_dir / f"{arm}_decisions.parquet", index=False)
        equity.to_parquet(args.output_dir / f"{arm}_equity.parquet", index=False)
        accepted.to_parquet(args.output_dir / f"{arm}_accepted_trades.parquet", index=False)

    summary = pd.concat(summary_frames, ignore_index=True)
    by_head = pd.concat(by_head_frames, ignore_index=True)
    schedule = pd.concat(schedule_frames, ignore_index=True)
    overlap = _accepted_overlap(accepted_by_arm, "R0_t1_timestamp_short_boll")
    sched_summary = (
        schedule.groupby(["arm", "short_boll_rank_scope"], observed=True)
        .agg(timestamp_count=("timestamp", "nunique"))
        .reset_index()
    )
    sched_summary["timestamp_share"] = sched_summary.groupby("arm", observed=True)[
        "timestamp_count"
    ].transform(lambda x: x / max(float(x.sum()), 1.0))

    summary.to_csv(args.output_dir / "rank_scope_switch_summary.csv", index=False)
    by_head.to_csv(args.output_dir / "rank_scope_switch_by_head.csv", index=False)
    overlap.to_csv(args.output_dir / "rank_scope_switch_accepted_overlap.csv", index=False)
    schedule.to_csv(args.output_dir / "rank_scope_switch_schedule.csv", index=False)
    sched_summary.to_csv(args.output_dir / "rank_scope_switch_schedule_summary.csv", index=False)

    manifest = {
        "generated_by": "run_market_state_short_boll_rank_scope_switch",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "shadow_market_state_rank_scope_switch_for_short_boll",
        "contract": {
            "changes_active_stack": False,
            "short_asset_rank_contract": "current_T1",
            "short_boll_rank_contract": "state_switch_or_continuous_blend_between_timestamp_and_global",
            "candidate_universe": "broad_before_rank_threshold_filtering",
            "rank_contract_candidate_parity_passed": bool(parity_report["passed"]),
            "qfail_active": False,
            "head_health_active": False,
            "threshold_lowering": False,
            "changes_policy_params": False,
            "promotion_status": "shadow_only",
            "production_eligible": False,
            "threshold_controller_must_promote_first": True,
        },
        "params": {
            "switch_margin": float(args.switch_margin),
            "blend_scale": float(args.blend_scale),
            "policy_variant": str(args.policy_variant),
            "candidate_file_name": str(args.candidate_file_name),
            "apply_priority_actions": bool(args.apply_priority_actions),
        },
        "inputs": {
            "t1_candidates": str(t1_path),
            "t1_candidates_sha256": _sha256(t1_path),
            "global_candidates": str(global_path),
            "global_candidates_sha256": _sha256(global_path),
            "priority_schedule": (
                str(schedule_path) if args.router_schedule is None else None
            ),
            "priority_schedule_sha256": (
                _sha256(schedule_path) if args.router_schedule is None else None
            ),
            "router_schedule": (
                str(schedule_path) if args.router_schedule is not None else None
            ),
            "router_schedule_sha256": (
                _sha256(schedule_path) if args.router_schedule is not None else None
            ),
            "schedule_source": schedule_source,
            "priority_action_schedule": (
                str(priority_action_path) if bool(args.apply_priority_actions) else None
            ),
            "priority_action_schedule_sha256": (
                _sha256(priority_action_path) if bool(args.apply_priority_actions) else None
            ),
            "train_deployable_candidates": str(args.train_deployable_candidates),
            "train_deployable_candidates_sha256": _sha256(args.train_deployable_candidates),
            "policy_manifest": str(args.policy_manifest),
            "policy_manifest_sha256": _sha256(args.policy_manifest),
            "policy_manifest_run_id": policy_payload.get("run_id"),
        },
        "summary": summary.to_dict("records"),
        "by_head": by_head.to_dict("records"),
        "overlap": overlap.to_dict("records"),
        "schedule_summary": sched_summary.to_dict("records"),
        "rank_contract_candidate_parity": parity_report,
        "outputs": {
            "manifest": str(args.output_dir / "rank_scope_switch_manifest.json"),
            "report": str(args.output_dir / "rank_scope_switch_report.md"),
            "summary": str(args.output_dir / "rank_scope_switch_summary.csv"),
            "by_head": str(args.output_dir / "rank_scope_switch_by_head.csv"),
            "overlap": str(args.output_dir / "rank_scope_switch_accepted_overlap.csv"),
            "schedule": str(args.output_dir / "rank_scope_switch_schedule.csv"),
        },
    }
    report = _render_report(
        manifest=manifest,
        summary=summary,
        by_head=by_head,
        overlap=overlap,
        schedule_summary=sched_summary,
    )
    (args.output_dir / "rank_scope_switch_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "rank_scope_switch_report.md").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            _json_safe({"output_dir": str(args.output_dir), "summary": summary.to_dict("records")}),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
