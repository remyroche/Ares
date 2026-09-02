#!/usr/bin/env python3
"""Sequential exact-1m research HPO for the frozen rich-exit extensions.

This is intentionally an offline-only producer.  It starts from the frozen
rich SimplePolicy parameters and explores only ``RichExitExtensions`` over
complete one-minute paths.  The score/admission route is immutable and target
free: BCF MC1 >= 30, current-v5 MC1 >= 30, with BCF MC1 expected bps as the
only auction priority.  No live configuration, exchange client, executor, or
runtime state is imported or amended.

Temporal protocol
-----------------
* HPO: February--August 2025 exact-one-minute decision-entry paths only,
  ranked by the normal constrained portfolio auction, not row-level return.
* Final constrained tournament: September--December 2025 only.
* Frozen evaluation: 2026 only.  It has no authority in parameter selection.
* Optional +5m result: a post-selection sensitivity only; it is not HPO data.

The funnel is deliberately sequential rather than factorial:
``soft_trailing -> no_progress -> local_peak_velocity -> smooth_protection``.
Each stage evaluates a small predeclared grid against the previous stage's
best extensions, preserves every trial receipt, and passes at most three
parents forward.  A plain frozen-rich no-extension control remains in every
final tournament.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.exact_1m_rich_policy_contract import (  # noqa: E402
    Exact1mRichV2ExecutionContract,
    RichExitExtensions,
    exact_1m_rich_v2_receipt,
    replay_exact_1m_rich_policy_v2,
)
from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    compute_replay_metrics,
    normalise_candidate_table,
    replay_candidates,
)
from extreme_price_movements.strict_r3_rich_policy import RichPolicyParams  # noqa: E402
from scripts.report_strict_r3_mc1_d2_controlled_portfolio import (  # noqa: E402
    CAUSAL_AUCTION_CURVE,
    _params as canonical_portfolio_params,
)


DEFAULT_DECISION_DATASET = ROOT / (
    "data_perp/artifacts/strict_r3_exact_1m_dual30_decision_dataset_"
    "2025_2026_20260817_v1"
)
DEFAULT_PLUS5_DATASET = ROOT / (
    "data_perp/artifacts/strict_r3_exact_1m_dual30_plus5m_dataset_"
    "2025_2026_20260817_v1"
)
DEFAULT_FROZEN_POLICY = ROOT / (
    "data_perp/artifacts/strict_r3_rich_policy_hpo_long_20260817_v1/"
    "frozen_challenger.json"
)
DEFAULT_OUT = ROOT / (
    "data_perp/artifacts/strict_r3_exact_1m_rich_extensions_hpo_"
    "decision2025_frozen2026_20260817_v1"
)

SCHEMA = "strict_r3_exact_1m_rich_extensions_hpo_v1"
SEED = 20260817
TUNE_START = pd.Timestamp("2025-02-01", tz="UTC")
TUNE_END = pd.Timestamp("2025-09-01", tz="UTC")
SELECT_START = pd.Timestamp("2025-09-01", tz="UTC")
SELECT_END = pd.Timestamp("2026-01-01", tz="UTC")
FROZEN_START = pd.Timestamp("2026-01-01", tz="UTC")
FROZEN_END = pd.Timestamp("2027-01-01", tz="UTC")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(values: object) -> object:
    if isinstance(values, pd.Series):
        return pd.to_datetime(values, utc=True, errors="raise")
    value = pd.Timestamp(values)
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


@dataclass(frozen=True)
class ExactPaths:
    rows: pd.DataFrame
    entry: np.ndarray
    atr: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    manifest: dict[str, Any]
    audit: dict[str, Any]

    def take(self, mask: np.ndarray) -> "ExactPaths":
        mask = np.asarray(mask, dtype=bool)
        return ExactPaths(
            rows=self.rows.loc[mask].reset_index(drop=True),
            entry=np.ascontiguousarray(self.entry[mask]),
            atr=np.ascontiguousarray(self.atr[mask]),
            high=np.ascontiguousarray(self.high[mask]),
            low=np.ascontiguousarray(self.low[mask]),
            close=np.ascontiguousarray(self.close[mask]),
            manifest=self.manifest,
            audit=self.audit,
        )


def _load_frozen_policy(
    path: Path, *, expected_side: str,
) -> tuple[RichPolicyParams, float, dict[str, Any]]:
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Schema-v1 long challengers predate explicit side lineage. Retain that
    # historical read path only for the canonical long default; a newly built
    # short policy must always declare its side rather than inheriting it.
    policy_side = str(
        payload.get("side") or ("long" if expected_side == "long" else "")
    ).strip().lower()
    if policy_side != expected_side:
        raise AssertionError(
            "frozen rich policy must carry an explicit matching side; "
            f"expected={expected_side}, observed={policy_side or 'missing'}"
        )
    params = RichPolicyParams.from_mapping(dict(payload.get("params") or {}))
    median_atr = float(payload.get("median_atr_fraction_fitted_on_complete_2024_development"))
    if not np.isfinite(median_atr) or median_atr <= 0.0:
        raise AssertionError("frozen rich policy has no valid development ATR anchor")
    if not np.isclose(float(payload.get("cost_bps")), 100.0):
        raise AssertionError("frozen rich policy does not bind 100-bps cost exactly once")
    return params, median_atr, {
        "path": str(path), "sha256": _sha256(path), "schema": payload.get("schema"),
        "side": policy_side,
        "params": params.to_dict(), "median_atr_fraction": median_atr,
    }


def _load_dataset(
    path: Path, *, expected_delay: int, expected_side: str,
) -> ExactPaths:
    root = Path(path).resolve()
    manifest_path = root / "dataset_manifest.json"
    rows_path = root / "training_rows.parquet"
    paths_path = root / "exact_paths.npz"
    audit_path = root / "candidate_path_audit.parquet"
    for item in (manifest_path, rows_path, paths_path, audit_path):
        if not item.is_file():
            raise FileNotFoundError(f"missing exact-1m HPO input: {item}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "strict_r3_exact_1m_policy_hpo_dataset_v1":
        raise AssertionError("unsupported exact-1m dataset schema")
    contract = dict(manifest.get("contract") or {})
    required_contract = {
        "bar_minutes": 1, "horizon_minutes": 720, "policy_cost_bps_once": 100.0,
        "entry_delay_minutes": int(expected_delay), "same_bar_activation_allowed": False,
    }
    for key, expected in required_contract.items():
        value = contract.get(key)
        if isinstance(expected, float):
            if not np.isclose(float(value), expected):
                raise AssertionError(f"exact-1m dataset has incompatible {key}")
        elif value != expected:
            raise AssertionError(f"exact-1m dataset has incompatible {key}")
    source = dict(manifest.get("candidate_source") or {})
    if source.get("target_free") is not True:
        raise AssertionError("HPO dataset candidate route is not explicitly target-free")
    if source.get("score_column") != "priority_bps":
        raise AssertionError("HPO dataset does not use BCF MC1 expected bps priority")
    forbidden = set(source.get("forbidden_selection_inputs") or [])
    if not {"label", "outcome", "policy_net_bps", "policy_path_valid"}.issubset(forbidden):
        raise AssertionError("target-free HPO route has incomplete forbidden-input audit")
    rows = pd.read_parquet(rows_path).copy()
    required_columns = {
        "candidate_id", "timestamp", "entry_ts", "symbol", "side_name", "score",
        "priority_bps", "entry_price", "signal_atr", "path_valid",
    }
    missing = sorted(required_columns.difference(rows.columns))
    if missing:
        raise AssertionError(f"exact-1m rows miss required fields: {missing}")
    rows["candidate_id"] = rows["candidate_id"].astype(str)
    rows["timestamp"] = _utc(rows["timestamp"])
    rows["entry_ts"] = _utc(rows["entry_ts"])
    if rows["candidate_id"].duplicated().any():
        raise AssertionError("exact-1m rows have duplicate candidate IDs")
    observed_side = rows["side_name"].astype(str).str.strip().str.lower()
    if not observed_side.eq(expected_side).all():
        raise AssertionError(
            "rich extension dataset is not side-local; "
            f"expected={expected_side}, observed="
            f"{observed_side.value_counts(dropna=False).to_dict()}"
        )
    expected_entry = rows["timestamp"] + pd.Timedelta(minutes=int(expected_delay))
    if not rows["entry_ts"].equals(expected_entry):
        raise AssertionError("exact-1m entries differ from declared decision-delay contract")
    if not rows["path_valid"].fillna(False).astype(bool).all():
        raise AssertionError("training rows must exclude invalid paths, not encode them as outcomes")
    if not np.isclose(
        pd.to_numeric(rows["score"], errors="raise"),
        pd.to_numeric(rows["priority_bps"], errors="raise"), rtol=0.0, atol=1e-12,
    ).all():
        raise AssertionError("HPO priority is not equal to frozen BCF MC1 expected bps")
    raw = np.load(paths_path, allow_pickle=False)
    for key in ("candidate_id", "entry", "atr", "high", "low", "close"):
        if key not in raw:
            raise AssertionError(f"exact path archive misses {key}")
    ids = raw["candidate_id"].astype(str)
    if not np.array_equal(ids, rows["candidate_id"].to_numpy()):
        raise AssertionError("exact paths do not align to the target-free row identities")
    arrays = {key: np.asarray(raw[key], dtype=np.float32) for key in ("entry", "atr", "high", "low", "close")}
    if arrays["high"].shape != arrays["low"].shape or arrays["high"].shape != arrays["close"].shape:
        raise AssertionError("exact path OHLC matrices are misaligned")
    if arrays["high"].shape != (len(rows), 720):
        raise AssertionError("exact paths must have 720 complete one-minute bars")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise AssertionError("complete exact HPO paths contain a non-finite value")
    if not np.isclose(arrays["entry"], pd.to_numeric(rows["entry_price"], errors="raise"), rtol=0.0, atol=1e-12).all():
        raise AssertionError("row entry price and exact path entry differ")
    audit = {
        "dataset_dir": str(root), "manifest_sha256": _sha256(manifest_path),
        "training_rows_sha256": _sha256(rows_path), "paths_sha256": _sha256(paths_path),
        "path_audit_sha256": _sha256(audit_path), "entry_delay_minutes": int(expected_delay),
        "valid_rows": int(len(rows)), "candidate_rows_before_outcome_join": int(manifest.get("candidate_rows", len(rows))),
        "invalid_rows_excluded_after_route": int(manifest.get("invalid_rows", 0)),
        "target_free_route": source,
    }
    # Preserve archive identity order here.  ``_resort`` performs one explicit
    # joint chronological reordering of rows *and* every path matrix.  Sorting
    # only the DataFrame here would silently detach candidate identity from
    # its OHLC path.
    return ExactPaths(rows=rows.reset_index(drop=True),
                      entry=arrays["entry"], atr=arrays["atr"], high=arrays["high"], low=arrays["low"], close=arrays["close"],
                      manifest=manifest, audit=audit)


def _resort(paths: ExactPaths) -> ExactPaths:
    """The archive is identity-ordered; make chronological slices explicit."""
    order = np.argsort(paths.rows["timestamp"].astype("int64").to_numpy(), kind="stable")
    if np.array_equal(order, np.arange(len(order))):
        return paths
    return ExactPaths(
        rows=paths.rows.iloc[order].reset_index(drop=True), entry=np.ascontiguousarray(paths.entry[order]),
        atr=np.ascontiguousarray(paths.atr[order]), high=np.ascontiguousarray(paths.high[order]),
        low=np.ascontiguousarray(paths.low[order]), close=np.ascontiguousarray(paths.close[order]),
        manifest=paths.manifest, audit=paths.audit,
    )


def _stable_month_sample(paths: ExactPaths, *, rows_per_month: int, seed: int) -> ExactPaths:
    """Outcome-free, timestamp-complete optional acceleration sample.

    A candidate-hash sample would arbitrarily remove a candidate's competitors
    from its decision auction.  This sample hashes decision timestamps and
    takes every candidate at each chosen timestamp.  It preserves within-hour
    competition and does not inspect a path or an outcome.  The default is
    ``rows_per_month=0``—the full Feb--Aug 2025 portfolio timeline—which is
    the authoritative HPO setting because it also preserves prior-position
    capacity interactions across every timestamp.
    """
    if rows_per_month <= 0:
        return paths
    months = paths.rows["timestamp"].dt.tz_localize(None).dt.to_period("M")
    timestamps = paths.rows["timestamp"]
    hashes = pd.util.hash_pandas_object(timestamps, index=False).to_numpy(np.uint64) ^ np.uint64(seed)
    keep = np.zeros(len(paths.rows), dtype=bool)
    positions_by_month = pd.Series(np.arange(len(paths.rows))).groupby(months, sort=True)
    for _, positions in positions_by_month:
        taken = positions.to_numpy(dtype=int)
        # Each timestamp's first occurrence holds its deterministic selector.
        unique_times, first = np.unique(timestamps.iloc[taken].astype("int64").to_numpy(), return_index=True)
        ordered_times = unique_times[np.argsort(hashes[taken[first]], kind="stable")]
        selected = 0
        for stamp in ordered_times:
            members = taken[timestamps.iloc[taken].astype("int64").to_numpy() == stamp]
            keep[members] = True
            selected += len(members)
            if selected >= int(rows_per_month):
                break
    return paths.take(keep)


def _monthly_objective(rows: pd.DataFrame, net_bps: np.ndarray) -> dict[str, float]:
    work = pd.DataFrame({
        "month": rows["timestamp"].dt.tz_localize(None).dt.to_period("M").astype(str),
        "net_bps": np.asarray(net_bps, dtype=float),
    })
    monthly = work.groupby("month", sort=True)["net_bps"].mean().to_numpy(float)
    if not len(monthly) or not np.isfinite(monthly).all():
        return {"objective": float("-inf"), "median_month_bps": float("nan"), "worst_month_bps": float("nan"), "month_mad_bps": float("nan")}
    median = float(np.median(monthly))
    mad = float(np.median(np.abs(monthly - median)))
    worst = float(np.min(monthly))
    return {
        "objective": median - 0.5 * mad - max(0.0, -worst),
        "median_month_bps": median, "worst_month_bps": worst, "month_mad_bps": mad,
    }


def _replay(paths: ExactPaths, *, params: RichPolicyParams, median_atr: float, extensions: RichExitExtensions, delay: int) -> dict[str, np.ndarray]:
    result = replay_exact_1m_rich_policy_v2(
        entry=paths.entry, atr=paths.atr, highs=paths.high, lows=paths.low, closes=paths.close,
        entry_timestamps=paths.rows["entry_ts"], params=params, median_atr_fraction=median_atr,
        extensions=extensions, contract=Exact1mRichV2ExecutionContract(entry_delay_minutes=int(delay)),
    )
    valid = np.asarray(result["path_valid"], dtype=bool)
    if not valid.all() or not np.isfinite(np.asarray(result["net_bps"], dtype=float)[valid]).all():
        raise AssertionError("complete exact paths did not yield complete rich-policy outcomes")
    return result


def _trial_record(stage: str, parent_id: int, trial_id: int, extensions: RichExitExtensions, paths: ExactPaths, replay: Mapping[str, np.ndarray], *, portfolio: Mapping[str, Any] | None = None) -> dict[str, Any]:
    metrics = _monthly_objective(paths.rows, np.asarray(replay["net_bps"], dtype=float))
    reasons = pd.Series(np.asarray(replay["exit_reason"], dtype=object)).value_counts()
    result = {
        "stage": stage, "parent_id": int(parent_id), "trial": int(trial_id),
        "rows": int(len(paths.rows)), "net_ev_bps_per_trade": float(np.mean(np.asarray(replay["net_bps"], dtype=float)),),
        "gross_ev_bps_per_trade": float(np.mean(np.asarray(replay["gross_bps"], dtype=float))),
        "mean_exit_minute": float(np.mean(np.asarray(replay["exit_minute"], dtype=float))),
        "trailing_exits": int(reasons.get("trailing", 0)), "stop_exits": int(reasons.get("stop_loss", 0)),
        "timeout_exits": int(reasons.get("timeout_h12", 0)), "no_progress_exits": int(reasons.get("no_progress", 0)),
        "peak_stall_exits": int(reasons.get("peak_stall", 0)), "velocity_exits": int(reasons.get("giveback_velocity", 0)),
        "smooth_capital_exits": int(reasons.get("smooth_capital_protect", 0)),
        # Keep row-level economics diagnostic-only.  The ``objective`` used
        # for HPO ordering is set below from the constrained portfolio.
        "row_monthly_objective": metrics.pop("objective"), **metrics, **asdict(extensions),
    }
    if portfolio is not None:
        for key, value in portfolio.items():
            if key != "portfolio_monthly":
                result[key] = value
        result["objective"] = float(portfolio["portfolio_selection_score"])
    else:
        result["objective"] = float(result["row_monthly_objective"])
    return result


def _extension_key(value: RichExitExtensions) -> str:
    return json.dumps(asdict(value), sort_keys=True, separators=(",", ":"))


def _unique(items: Iterable[RichExitExtensions]) -> list[RichExitExtensions]:
    out: dict[str, RichExitExtensions] = {}
    for item in items:
        item.validate()
        out.setdefault(_extension_key(item), item)
    return list(out.values())


def _stage_options(stage: str, parent: RichExitExtensions) -> list[RichExitExtensions]:
    """Small predeclared sequential grids; no 2026-driven branching."""
    if stage == "soft_trailing":
        updates = (
            {},
            {"giveback_confirmation_window_minutes": 2},
            {"giveback_confirmation_window_minutes": 3},
            {"giveback_confirmation_window_minutes": 3, "giveback_confirmation_fraction": 2.0 / 3.0},
            {"trail_hysteresis_atr": 0.05},
            {"trail_hysteresis_atr": 0.10},
            {"minute_noise_scale": 0.20, "minute_noise_floor_atr": 0.02, "minute_noise_cap_atr": 0.12, "minute_noise_ewma_minutes": 5.0},
            {"minute_noise_scale": 0.20, "minute_noise_floor_atr": 0.02, "minute_noise_cap_atr": 0.12, "minute_noise_ewma_minutes": 15.0},
            {"minute_noise_scale": 0.35, "minute_noise_floor_atr": 0.03, "minute_noise_cap_atr": 0.18, "minute_noise_ewma_minutes": 30.0},
            {"giveback_confirmation_window_minutes": 2, "trail_hysteresis_atr": 0.05},
            {"giveback_confirmation_window_minutes": 2, "minute_noise_scale": 0.20, "minute_noise_floor_atr": 0.02, "minute_noise_cap_atr": 0.12, "minute_noise_ewma_minutes": 15.0},
            {"trailing_ratchet_step_atr": 0.10},
            {"trailing_ratchet_step_atr": 0.20},
            {"giveback_confirmation_window_minutes": 2, "trailing_ratchet_step_atr": 0.10},
            {"trail_hysteresis_atr": 0.05, "trailing_ratchet_step_atr": 0.10},
        )
    elif stage == "no_progress":
        updates = (
            {},
            {"no_progress_start_minutes": 45, "no_progress_required_mfe_atr": 0.25},
            {"no_progress_start_minutes": 60, "no_progress_required_mfe_atr": 0.25},
            {"no_progress_start_minutes": 60, "no_progress_required_mfe_atr": 0.50},
            {"no_progress_start_minutes": 90, "no_progress_required_mfe_atr": 0.50},
            {"no_progress_start_minutes": 60, "no_progress_min_mfe_slope_atr_per_hour": 0.10},
            {"no_progress_start_minutes": 90, "no_progress_min_mfe_slope_atr_per_hour": 0.10},
            {"no_progress_start_minutes": 60, "no_progress_required_mfe_atr": 0.25, "no_progress_min_mfe_slope_atr_per_hour": 0.10},
            {"no_progress_origin": "mae", "no_progress_start_minutes": 45, "no_progress_required_mfe_atr": 0.25},
            {"no_progress_origin": "mae", "no_progress_start_minutes": 60, "no_progress_required_mfe_atr": 0.50},
            {"no_progress_origin": "mae", "no_progress_start_minutes": 90, "no_progress_required_mfe_atr": 0.50},
            {"no_progress_origin": "mae", "no_progress_start_minutes": 60, "no_progress_min_mfe_slope_atr_per_hour": 0.10},
        )
    elif stage == "local_peak_velocity":
        updates = (
            {},
            {"stalled_peak_minutes": 10, "stalled_peak_drawdown_atr": 0.25},
            {"stalled_peak_minutes": 15, "stalled_peak_drawdown_atr": 0.25},
            {"stalled_peak_minutes": 20, "stalled_peak_drawdown_atr": 0.50},
            {"stalled_peak_minutes": 30, "stalled_peak_drawdown_atr": 0.50},
            {"giveback_velocity_atr_per_hour": 0.50},
            {"giveback_velocity_atr_per_hour": 1.0},
            {"giveback_velocity_atr_per_hour": 1.5},
            {"stalled_peak_minutes": 15, "stalled_peak_drawdown_atr": 0.25, "giveback_velocity_atr_per_hour": 1.0},
        )
    elif stage == "smooth_protection":
        updates = (
            {},
            {"protection_activation_atr": 1.0, "protection_strength": 0.25},
            {"protection_activation_atr": 1.0, "protection_strength": 0.50},
            {"protection_activation_atr": 1.25, "protection_strength": 0.50},
            {"protection_activation_atr": 1.5, "protection_strength": 0.25},
            {"protection_activation_atr": 1.5, "protection_strength": 0.50},
            {"protection_activation_atr": 1.75, "protection_strength": 0.50},
            {"protection_activation_atr": 2.0, "protection_strength": 0.50},
            {"protection_activation_atr": 2.5, "protection_strength": 0.50},
            {"protection_activation_atr": 1.5, "protection_strength": 0.50, "protection_power": 0.75},
            {"protection_activation_atr": 1.5, "protection_strength": 0.50, "protection_power": 1.0},
            {"protection_activation_atr": 1.5, "protection_strength": 0.50, "protection_power": 1.25},
            {"protection_activation_atr": 1.5, "protection_strength": 0.50, "protection_power": 1.5},
            {"protection_activation_atr": 1.5, "protection_strength": 0.50, "protection_power": 1.75},
            {"protection_activation_atr": 1.5, "protection_strength": 0.50, "protection_power": 2.0},
        )
    else:
        raise ValueError(f"unknown extension stage: {stage}")
    return _unique(replace(parent, **update) for update in updates)


def _top_extensions(records: list[dict[str, Any]], *, keep: int) -> list[RichExitExtensions]:
    ordered = sorted(
        records,
        key=lambda item: (
            -float(item["portfolio_selection_score"]),
            -float(item["portfolio_net_ev_bps_per_trade"]),
            -float(item["portfolio_total_net_bps"]),
            int(item["trial"]),
        ),
    )
    selected: list[RichExitExtensions] = []
    seen: set[str] = set()
    for item in ordered:
        payload = {key: item[key] for key in RichExitExtensions.__dataclass_fields__}
        ext = RichExitExtensions(**payload)
        key = _extension_key(ext)
        if key not in seen:
            selected.append(ext)
            seen.add(key)
        if len(selected) >= int(keep):
            break
    return selected


def _portfolio_candidates(
    paths: ExactPaths, replay: Mapping[str, np.ndarray], *, arm: str, side: str,
) -> pd.DataFrame:
    net = np.asarray(replay["net_bps"], dtype=float)
    gross = np.asarray(replay["gross_bps"], dtype=float)
    exit_ts = pd.to_datetime(np.asarray(replay["exit_timestamp"]), utc=True)
    if not np.isfinite(net).all() or exit_ts.isna().any():
        raise AssertionError("portfolio input has unresolved exact-1m outcomes")
    exit_minutes = np.asarray(replay["exit_minute"], dtype=float)
    rows = paths.rows
    table = pd.DataFrame({
        "timestamp": rows["entry_ts"], "decision_timestamp": rows["timestamp"],
        "candidate_id": rows["candidate_id"].astype(str), "symbol": rows["symbol"].astype(str),
        "side": side, "strategy_id": arm, "policy_archetype": arm,
        # The route's BCF MC1 expected bps is the frozen auction priority.  It
        # is not re-ranked within timestamp and it is not outcome dependent.
        "normalized_rank_score": 1.0, "strategy_rank_pct": 1.0, "base_strategy_threshold": 0.0,
        "calibrated_score": pd.to_numeric(rows["score"], errors="raise"),
        "portfolio_priority_adjustment": pd.to_numeric(rows["score"], errors="raise"),
        "entry_price": np.asarray(paths.entry, dtype=float), "exit_timestamp": exit_ts,
        "exit_price": np.asarray(replay["exit_price"], dtype=float),
        "net_return": net / 10_000.0, "gross_return": gross / 10_000.0,
        "holding_bars": np.maximum(1.0, np.ceil((exit_minutes + 1.0) / 15.0)),
        "simple_policy_exit_reason": np.asarray(replay["exit_reason"], dtype=object),
        # Net already includes the one frozen 100-bps policy cost.  The
        # auction must never debit it a second time.
        "fees_bps": 100.0, "expected_friction_bps": 0.0, "price_gap_bps": 0.0,
        "liquidity_capacity_weight": 1.0, "policy_outcome_available": True,
    })
    return normalise_candidate_table(table)


def _portfolio_metrics(
    paths: ExactPaths, replay: Mapping[str, np.ndarray], *, arm: str,
    include_frames: bool, side: str,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    candidates = _portfolio_candidates(paths, replay, arm=arm, side=side)
    decisions, equity, _ = replay_candidates(
        candidates, canonical_portfolio_params(), mode="global_auction", ev_curve=CAUSAL_AUCTION_CURVE,
        market_mode="perp", initial_wallet=1000.0,
    )
    indices = pd.to_numeric(decisions["candidate_index"], errors="raise").astype(int).to_numpy()
    decisions = decisions.copy()
    decisions["candidate_id"] = candidates.iloc[indices]["candidate_id"].to_numpy()
    decisions["decision_timestamp"] = candidates.iloc[indices]["decision_timestamp"].to_numpy()
    accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)].copy()
    accepted["net_bps"] = pd.to_numeric(accepted["position_net_return"], errors="raise") * 10_000.0
    accepted["month"] = pd.to_datetime(accepted["decision_timestamp"], utc=True).dt.strftime("%Y-%m")
    monthly = accepted.groupby("month", sort=True).agg(
        trades=("candidate_id", "size"), net_ev_bps_per_trade=("net_bps", "mean"), net_sum_bps=("net_bps", "sum"),
    ).reset_index()
    values = monthly["net_ev_bps_per_trade"].to_numpy(float)
    median = float(np.median(values)) if len(values) else float("nan")
    mad = float(np.median(np.abs(values - median))) if len(values) else float("nan")
    worst = float(np.min(values)) if len(values) else float("nan")
    raw = compute_replay_metrics(candidates, decisions, equity, params=canonical_portfolio_params())
    result = {
        "portfolio_entries": int(len(accepted)),
        "portfolio_net_ev_bps_per_trade": float(accepted["net_bps"].mean()) if len(accepted) else float("nan"),
        "portfolio_total_net_bps": float(accepted["net_bps"].sum()) if len(accepted) else 0.0,
        "portfolio_median_month_bps": median, "portfolio_month_mad_bps": mad,
        "portfolio_worst_month_bps": worst,
        "portfolio_selection_score": median - 0.5 * mad - max(0.0, -worst) if len(values) else float("-inf"),
        "portfolio_max_drawdown": float(raw.get("max_drawdown", np.nan)),
        "portfolio_sortino": float(raw.get("sortino", np.nan)),
        "portfolio_worst_week_return": float(raw.get("worst_week", np.nan)),
        "portfolio_monthly": monthly.to_dict(orient="records"),
    }
    return result, ({"candidates": candidates, "decisions": decisions, "equity": equity, "accepted": accepted, "monthly": monthly} if include_frames else {})


def _write_frames(out: Path, stem: str, frames: Mapping[str, pd.DataFrame]) -> None:
    for name, frame in frames.items():
        frame.to_parquet(out / f"{stem}_{name}.parquet", index=False, compression="zstd")


def _window(paths: ExactPaths, start: pd.Timestamp, end: pd.Timestamp) -> ExactPaths:
    time = paths.rows["timestamp"]
    mask = time.ge(start) & time.lt(end)
    if not mask.any():
        raise RuntimeError(f"no exact-path rows in required window {start}..{end}")
    return paths.take(mask.to_numpy())


def _stage_hpo(
    paths: ExactPaths, *, params: RichPolicyParams, median_atr: float,
    parents: list[RichExitExtensions], stage: str, keep: int, trial_offset: int,
    side: str,
) -> tuple[list[dict[str, Any]], list[RichExitExtensions]]:
    records: list[dict[str, Any]] = []
    counter = int(trial_offset)
    for parent_id, parent in enumerate(parents):
        for extensions in _stage_options(stage, parent):
            replay = _replay(paths, params=params, median_atr=median_atr, extensions=extensions, delay=0)
            portfolio, _ = _portfolio_metrics(
                paths, replay, arm=f"hpo_{stage}_{parent_id}_{counter}",
                include_frames=False, side=side,
            )
            records.append(_trial_record(
                stage, parent_id, counter, extensions, paths, replay, portfolio=portfolio,
            ))
            counter += 1
    return records, _top_extensions(records, keep=keep)


def _final_tournament(
    paths: ExactPaths, *, params: RichPolicyParams, median_atr: float,
    finalists: list[tuple[str, RichExitExtensions]], out: Path, side: str,
) -> tuple[pd.DataFrame, RichExitExtensions]:
    results: list[dict[str, Any]] = []
    for order, (name, extensions) in enumerate(finalists):
        replay = _replay(paths, params=params, median_atr=median_atr, extensions=extensions, delay=0)
        portfolio, frames = _portfolio_metrics(
            paths, replay, arm=f"selection_{order:02d}_{name}",
            include_frames=True, side=side,
        )
        _write_frames(out, f"selection_{order:02d}_{name}", frames)
        results.append({"arm": name, "order": order, **_trial_record("final_2025_selection", -1, order, extensions, paths, replay), **portfolio})
    table = pd.DataFrame(results).sort_values(
        ["portfolio_selection_score", "portfolio_net_ev_bps_per_trade", "portfolio_total_net_bps", "order"],
        ascending=[False, False, False, True], kind="stable",
    ).reset_index(drop=True)
    winner_payload = {key: table.iloc[0][key] for key in RichExitExtensions.__dataclass_fields__}
    return table, RichExitExtensions(**winner_payload)


def _evaluate_frozen(
    paths: ExactPaths, *, params: RichPolicyParams, median_atr: float,
    extensions: RichExitExtensions, delay: int, label: str, out: Path, side: str,
) -> dict[str, Any]:
    replay = _replay(paths, params=params, median_atr=median_atr, extensions=extensions, delay=delay)
    portfolio, frames = _portfolio_metrics(
        paths, replay, arm=label, include_frames=True, side=side,
    )
    _write_frames(out, label, frames)
    return {
        "arm": label, "entry_delay_minutes": int(delay), "rows": int(len(paths.rows)),
        **_trial_record("frozen_2026_evaluation", -1, -1, extensions, paths, replay), **portfolio,
        "v2_receipt": exact_1m_rich_v2_receipt(
            params=params, extensions=extensions, replay=replay,
            contract=Exact1mRichV2ExecutionContract(entry_delay_minutes=delay),
        ),
    }


def _assert_output_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite immutable output: {path}")
    path.mkdir(parents=True, exist_ok=False)


def run(args: argparse.Namespace) -> Path:
    if args.side not in {"long", "short"}:
        raise ValueError("rich extension HPO side must be long or short")
    output = Path(args.out_dir).resolve()
    _assert_output_empty(output)
    params, median_atr, policy_audit = _load_frozen_policy(
        Path(args.frozen_policy), expected_side=args.side,
    )
    decision = _resort(_load_dataset(
        Path(args.decision_dataset), expected_delay=0, expected_side=args.side,
    ))
    # 2025 tuning receives a deterministic identity-only equal-month sample.
    tune_full = _window(decision, TUNE_START, TUNE_END)
    tuning = _stable_month_sample(tune_full, rows_per_month=int(args.hpo_rows_per_month), seed=int(args.seed))
    if len(tuning.rows) < int(args.min_tuning_rows):
        raise RuntimeError("insufficient exact-1m 2025 tuning rows after deterministic sampling")
    select = _window(decision, SELECT_START, SELECT_END)
    frozen_2026 = _window(decision, FROZEN_START, FROZEN_END)
    stages = ("soft_trailing", "no_progress", "local_peak_velocity", "smooth_protection")
    base = RichExitExtensions()
    records: list[dict[str, Any]] = []
    base_replay = _replay(tuning, params=params, median_atr=median_atr, extensions=base, delay=0)
    base_portfolio, _ = _portfolio_metrics(
        tuning, base_replay, arm="hpo_frozen_rich_control",
        include_frames=False, side=args.side,
    )
    records.append(_trial_record("frozen_rich_control", -1, -1, base, tuning, base_replay, portfolio=base_portfolio))
    parents = [base]
    offset = 0
    stage_winners: dict[str, list[dict[str, Any]]] = {}
    for stage in stages:
        trial_rows, parents = _stage_hpo(
            tuning, params=params, median_atr=median_atr, parents=parents, stage=stage,
            keep=int(args.parents_per_stage), trial_offset=offset, side=args.side,
        )
        records.extend(trial_rows)
        offset += len(trial_rows)
        stage_winners[stage] = [asdict(item) for item in parents]
        if not parents:
            raise RuntimeError(f"{stage} produced no valid extension candidate")
    trials = pd.DataFrame(records).sort_values(
        ["stage", "objective", "net_ev_bps_per_trade", "trial"], ascending=[True, False, False, True], kind="stable",
    )
    trials.to_parquet(output / "sequential_extension_trials_2025_tuning.parquet", index=False, compression="zstd")
    finalists = [("frozen_rich_control", base)]
    seen = {_extension_key(base)}
    for index, extensions in enumerate(parents, start=1):
        if _extension_key(extensions) not in seen:
            finalists.append((f"combined_{index:02d}", extensions))
            seen.add(_extension_key(extensions))
    # Keep the best standalone winner of each stage as an interpretable
    # ablation, even if a later combined parent displaced it.
    for stage, values in stage_winners.items():
        if values:
            extensions = RichExitExtensions(**values[0])
            if _extension_key(extensions) not in seen:
                finalists.append((f"{stage}_standalone", extensions))
                seen.add(_extension_key(extensions))
    finalists = finalists[: int(args.max_finalists)]
    tournament, winner = _final_tournament(
        select, params=params, median_atr=median_atr, finalists=finalists,
        out=output, side=args.side,
    )
    tournament.to_parquet(output / "final_2025_constrained_tournament.parquet", index=False, compression="zstd")
    winner_index = int(tournament.iloc[0]["order"])
    winner_name = str(tournament.iloc[0]["arm"])
    # This section is strictly evaluation: 2026 is never passed to HPO,
    # tournament ordering, or extension construction.
    frozen_default = _evaluate_frozen(
        frozen_2026, params=params, median_atr=median_atr, extensions=base,
        delay=0, label="frozen_rich_control_2026_decision", out=output,
        side=args.side,
    )
    frozen_winner = _evaluate_frozen(
        frozen_2026, params=params, median_atr=median_atr, extensions=winner,
        delay=0, label="frozen_extensions_winner_2026_decision", out=output,
        side=args.side,
    )
    evaluations = [frozen_default, frozen_winner]
    if args.plus5_dataset is not None:
        plus5 = _resort(_load_dataset(
            Path(args.plus5_dataset), expected_delay=5, expected_side=args.side,
        ))
        plus5_2026 = _window(plus5, FROZEN_START, FROZEN_END)
        evaluations.append(_evaluate_frozen(
            plus5_2026, params=params, median_atr=median_atr,
            extensions=winner, delay=5,
            label="frozen_extensions_winner_2026_plus5_sensitivity", out=output,
            side=args.side,
        ))
    evaluation_table = pd.DataFrame(evaluations)
    evaluation_table.to_parquet(output / "frozen_2026_evaluation.parquet", index=False, compression="zstd")
    winner_payload = {
        "schema": SCHEMA, "research_only": True, "promotion": "not authorised",
        "side": args.side, "seed": int(args.seed),
        "frozen_base_policy": policy_audit, "extensions": asdict(winner),
        "winner_name": winner_name, "winner_finalist_order": winner_index,
        "selection_protocol": {
            "hpo": (
                "2025-02 through 2025-08 exact-1m decision-entry only; "
                "normal constrained BCF-priority portfolio score; "
                + (
                    "complete Feb-Aug portfolio timeline"
                    if int(args.hpo_rows_per_month) <= 0
                    else "outcome-free timestamp-complete deterministic acceleration sample"
                )
            ),
            "final_tournament": "2025-09 through 2025-12 only; full exact-path normal constrained portfolio",
            "frozen_evaluation": "2026 only; no HPO or winner selection authority",
            "plus5": "post-selection sensitivity only; never parameter-selection input",
        },
        "route": decision.audit["target_free_route"], "decision_dataset": decision.audit,
        "final_tournament": tournament.to_dict(orient="records"),
        "frozen_2026": evaluation_table.to_dict(orient="records"),
        "portfolio_contract": {
            "source": "scripts.report_strict_r3_mc1_d2_controlled_portfolio._params",
            "priority": "BCF MC1 expected bps only", "max_new_entries_per_decision": 2,
            "margin_budget_pct": 0.80, "margin_slot_pct": 0.10, "leverage": 7,
        },
        "cost": "100 bps is included exactly once in rich policy net_bps; auction recharges zero",
    }
    (output / "frozen_extensions_winner.json").write_text(json.dumps(_json_safe(winner_payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    correctness = {
        "schema": f"{SCHEMA}_correctness", "status": "passed",
        "target_free_route": decision.audit["target_free_route"],
        "candidate_paths": "complete 720x1m rows only; invalid paths excluded after target-free route",
        "side": args.side,
        "entry": "decision timestamp for primary HPO/selection/evaluation; +5 is sensitivity only",
        "exit": "exact rich v2 state machine; hard stop/capital precedence and completed-minute soft actions",
        "selection": "2025 only", "frozen_evaluation": "2026 only", "no_live_imports_or_writes": True,
        "cost": "100 bps once", "contract": Exact1mRichV2ExecutionContract(entry_delay_minutes=0).to_dict(),
    }
    (output / "correctness_report.json").write_text(json.dumps(_json_safe(correctness), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema": SCHEMA, "side": args.side,
        "code_sha256": _sha256(Path(__file__).resolve()), "winner": winner_payload,
        "correctness": correctness, "trials": int(len(trials)), "stage_winners": stage_winners,
    }
    (output / "run_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-dataset", type=Path, default=DEFAULT_DECISION_DATASET)
    parser.add_argument("--plus5-dataset", type=Path, default=DEFAULT_PLUS5_DATASET)
    parser.add_argument("--frozen-policy", type=Path, default=DEFAULT_FROZEN_POLICY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--hpo-rows-per-month", type=int, default=0,
                        help="0 (default) uses the full Feb-Aug portfolio timeline; positive values use an explicitly documented timestamp-complete acceleration sample")
    parser.add_argument("--min-tuning-rows", type=int, default=1000)
    parser.add_argument("--parents-per-stage", type=int, default=3)
    parser.add_argument("--max-finalists", type=int, default=8)
    parser.add_argument("--side", choices=("long", "short"), default="long")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    print(run(parse_args()))


if __name__ == "__main__":
    main()
