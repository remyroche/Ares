#!/usr/bin/env python3
"""Exact-one-minute SimplePolicyOptimiser for the routed P8U live contract.

This is the strict replacement for a 15-minute exit-proxy policy search.  It
reads only score columns to construct the candidate route, then joins complete
one-minute paths, evaluates the same rich state machine used by live exits,
and applies the normal chronological eight-slot portfolio auction.

Temporal protocol for a requested May--August window:

* May is an adverse-threshold calibration reserve;
* June--July are the HPO and full-tournament development window;
* August is a frozen confirmation only.

The run is offline research.  It never loads live account state, contacts an
exchange, changes a frozen bundle, or submits an order.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.data_store import (  # noqa: E402
    _execution_1m_part_bounds_seconds,
    _execution_1m_part_paths_for_range,
    _execution_1m_rows_equal,
    _normalise_execution_1m_frame,
    canonical_kraken_execution_1m_root,
)
from extreme_price_movements.exact_1m_rich_policy_contract import (  # noqa: E402
    Exact1mRichV2ExecutionContract,
    RichExitExtensions,
    exact_1m_rich_v2_receipt,
    fit_adverse_theta_exact_1m,
    replay_exact_1m_rich_policy_v2,
)
from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    compute_replay_metrics,
    normalise_candidate_table,
    replay_candidates,
)
from extreme_price_movements.strict_r3_rich_policy import RichPolicyParams  # noqa: E402
from scripts.materialize_strict_r3_exact_1m_policy_hpo_dataset import (  # noqa: E402
    ATR_SOURCE_LOOKBACK_HOURS,
    _causal_atr,
    _clean_minute,
)
from scripts.report_strict_r3_mc1_d2_controlled_portfolio import (  # noqa: E402
    CAUSAL_AUCTION_CURVE,
    _params as canonical_portfolio_params,
)
from scripts.run_strict_r3_exact_1m_rich_entry_delay_ladder import (  # noqa: E402
    HORIZON_MINUTES,
    _complete_mask,
)


DEFAULT_DUAL = ROOT / (
    "data_perp/artifacts/strict_r3_p8u_f72_underf120_dual_mc1_nov25_aug27_"
    "20260828_v1/dual_predictions.parquet"
)
DEFAULT_POLICY = ROOT / (
    "data_perp/artifacts/strict_r3_rich_policy_smooth_protection_long_"
    "20260817_v1/frozen_policy.json"
)
DEFAULT_MINUTE_ROOT = ROOT / "data_perp/exchanges/krakenfutures/execution_1m"
DEFAULT_EXACT_PATH_SOURCE = ROOT / (
    "data_perp/artifacts/strict_r3_p8u_exact1m_simple_policy_optimiser_"
    "mayjul_frozenaug2026_20260829_v1"
)
DEFAULT_OUT = ROOT / (
    "data_perp/artifacts/strict_r3_p8u_exact1m_simple_policy_optimiser_"
    "mayjul2026_20260829_v1"
)
SEED = 20260829
OPTIONAL_POLICY_FIELDS = frozenset({
    "sl_atr_power", "sl_atr_multiplier", "tp_atr_power", "tp_atr_multiplier",
    "capital_protect_lock_frac", "adverse_exit_theta",
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: Any) -> pd.Series | pd.Timestamp:
    if isinstance(value, pd.Series):
        return pd.to_datetime(value, utc=True, errors="raise")
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


@dataclass(frozen=True)
class ExactPaths:
    rows: pd.DataFrame
    entry: np.ndarray
    atr: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray

    def take(self, mask: np.ndarray) -> "ExactPaths":
        take = np.asarray(mask, dtype=bool)
        return ExactPaths(
            rows=self.rows.loc[take].reset_index(drop=True),
            entry=np.ascontiguousarray(self.entry[take]),
            atr=np.ascontiguousarray(self.atr[take]),
            high=np.ascontiguousarray(self.high[take]),
            low=np.ascontiguousarray(self.low[take]),
            close=np.ascontiguousarray(self.close[take]),
        )


def _load_frozen_exact_paths(
    source_root: Path,
    *,
    population: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[ExactPaths, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load a target-free exact-path source panel frozen before policy HPO.

    Reusing a hash-bound raw path panel avoids a second historical source
    retrieval.  The panel is acceptable only when its full score-only route is
    identity-equal to the current route and every stored path array is aligned
    to its candidate row.  It contains no policy outcome or optimization
    result: one-minute high/low/close paths are replayed below for each trial.
    """
    root = Path(source_root).resolve()
    candidates_path = root / "target_free_candidates.parquet"
    rows_path = root / "valid_exact_paths_rows.parquet"
    arrays_path = root / "exact_paths.npz"
    coverage_path = root / "source_coverage.parquet"
    invalid_path = root / "invalid_outcomes_after_route.parquet"
    required = [candidates_path, rows_path, arrays_path, coverage_path, invalid_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"frozen exact path source is incomplete: {missing}")
    source_candidates = pd.read_parquet(candidates_path)
    source_candidates["candidate_id"] = source_candidates["candidate_id"].astype(str)
    expected_ids = set(population["candidate_id"].astype(str))
    if set(source_candidates["candidate_id"]) != expected_ids:
        raise AssertionError("frozen exact path source route differs from the score-only route")
    rows = pd.read_parquet(rows_path).copy()
    rows["candidate_id"] = rows["candidate_id"].astype(str)
    rows["timestamp"] = _utc(rows["timestamp"])
    rows["entry_ts"] = _utc(rows["entry_ts"])
    packed = np.load(arrays_path, allow_pickle=True)
    expected_arrays = {"candidate_id", "entry", "atr", "high", "low", "close"}
    if not expected_arrays.issubset(set(packed.files)):
        raise AssertionError("frozen exact path archive is missing a required array")
    array_ids = packed["candidate_id"].astype(str)
    if len(rows) != len(array_ids) or not np.array_equal(rows["candidate_id"].to_numpy(str), array_ids):
        raise AssertionError("frozen exact path archive is not identity-aligned to its row panel")
    arrays = {key: np.asarray(packed[key]) for key in ("entry", "atr", "high", "low", "close")}
    if any(len(value) != len(rows) for value in arrays.values()):
        raise AssertionError("frozen exact path archive has a mismatched row count")
    if arrays["high"].ndim != 2 or arrays["high"].shape[1] != HORIZON_MINUTES:
        raise AssertionError("frozen exact path archive does not contain 720-minute paths")
    if any(not np.isfinite(np.asarray(value, dtype=float)).all() for value in arrays.values()):
        raise AssertionError("frozen exact path archive contains non-finite values")
    keep = rows["timestamp"].ge(start) & rows["timestamp"].lt(end)
    kept_rows = rows.loc[keep].reset_index(drop=True)
    kept_arrays = {key: np.ascontiguousarray(value[keep.to_numpy()]) for key, value in arrays.items()}
    valid_ids = set(kept_rows["candidate_id"])
    invalid = pd.read_parquet(invalid_path).copy()
    if not invalid.empty:
        invalid["candidate_id"] = invalid["candidate_id"].astype(str)
        invalid["timestamp"] = _utc(invalid["timestamp"])
        invalid = invalid.loc[invalid["timestamp"].ge(start) & invalid["timestamp"].lt(end)].reset_index(drop=True)
    invalid_ids = set(invalid["candidate_id"]) if not invalid.empty else set()
    if valid_ids.intersection(invalid_ids) or valid_ids.union(invalid_ids) != expected_ids:
        raise AssertionError("frozen exact path source has invalid route/path identity coverage")
    coverage = pd.read_parquet(coverage_path).copy()
    audit = {
        "root": str(root),
        "target_free_candidates_sha256": _sha256(candidates_path),
        "valid_rows_sha256": _sha256(rows_path),
        "exact_paths_sha256": _sha256(arrays_path),
        "source_coverage_sha256": _sha256(coverage_path),
        "invalid_outcomes_sha256": _sha256(invalid_path),
        "stored_route_rows": int(len(source_candidates)),
        "replay_route_rows": int(len(population)),
        "valid_exact_paths": int(len(kept_rows)),
        "invalid_paths_after_route": int(len(invalid)),
        "identity_equal": True,
        "path_shape": [int(value) for value in kept_arrays["high"].shape],
    }
    return (
        ExactPaths(rows=kept_rows, **kept_arrays),
        coverage,
        invalid,
        audit,
    )


def _target_free_population(
    source: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    threshold_bps: float,
) -> pd.DataFrame:
    """Build the dual-MC1 route using score columns only.

    ``dual_predictions`` also stores old policy columns.  Projection is
    deliberately restricted to identity and MC1 score columns so neither the
    route nor its portfolio priority can see a realised outcome.
    """
    columns = [
        "candidate_id", "__decision_ts__", "__symbol__", "side_name",
        "bcf_mc1_expected_bps", "current_mc1_expected_bps",
    ]
    frame = pd.read_parquet(source, columns=columns).copy()
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["timestamp"] = _utc(frame.pop("__decision_ts__"))
    frame["symbol"] = frame.pop("__symbol__").astype(str)
    frame["side_name"] = frame["side_name"].astype(str).str.strip().str.lower()
    frame["bcf_mc1_expected_bps"] = pd.to_numeric(frame["bcf_mc1_expected_bps"], errors="coerce")
    frame["current_mc1_expected_bps"] = pd.to_numeric(frame["current_mc1_expected_bps"], errors="coerce")
    frame = frame.loc[
        frame["side_name"].eq("long")
        & frame["timestamp"].ge(start)
        & frame["timestamp"].lt(end)
        & frame["bcf_mc1_expected_bps"].ge(float(threshold_bps))
        & frame["current_mc1_expected_bps"].ge(float(threshold_bps))
    ].copy()
    frame["entry_ts"] = frame["timestamp"] + pd.Timedelta(minutes=5)
    frame["priority_bps"] = frame["bcf_mc1_expected_bps"]
    if frame.empty or frame["candidate_id"].duplicated().any():
        raise AssertionError("target-free P8U policy route is empty or has duplicate identities")
    return frame.sort_values(["timestamp", "candidate_id"], kind="stable").reset_index(drop=True)


def _read_exact_ohlc(
    minute_root: Path,
    symbol: str,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    prefer_covering_part: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Read canonical 1m OHLC, recording only safe legacy-compact exclusions.

    A historical compaction can be a zero-byte/cloud-placeholder file even
    though an immutable append part covers its complete named interval.  That
    compact cannot carry unique source data.  It may be excluded *only* when a
    valid ``part-*`` file spans the whole named interval; all individual policy
    paths are still subsequently required to have 720 observed rows.  Every
    exclusion is returned for the artifact-level source audit.
    """
    root = canonical_kraken_execution_1m_root(minute_root)
    symbol_dir = root / "ohlcv" / f"symbol={str(symbol).replace('/', '_')}"
    if not symbol_dir.exists():
        return pd.DataFrame(columns=["open", "high", "low", "close"], index=pd.DatetimeIndex([], tz="UTC")), []
    parts = _execution_1m_part_paths_for_range(symbol_dir, start=start, end=end)
    all_bounds = {part: _execution_1m_part_bounds_seconds(part) for part in parts}
    rows: dict[int, dict[str, Any]] = {}
    exclusions: list[dict[str, str]] = []
    if prefer_covering_part:
        # Recovery created many overlapping cumulative copies.  Build a
        # minimal payload-verified cover of the requested *minute grid* rather
        # than trusting broad filename bounds.  This remains research-only:
        # every selected duplicate is conflict-checked, every requested minute
        # must be present, and promotion validation still uses the all-parts
        # scan below.
        minute_ns = 60 * 10**9
        required = set(range(int(start.value), int(end.value) + minute_ns, minute_ns))
        selected_rows: dict[int, dict[str, Any]] = {}
        selected_count = 0
        for part in sorted(parts, key=lambda path: (-path.stat().st_size, str(path))):
            if not required:
                break
            bounds = all_bounds.get(part)
            if bounds is not None:
                lower = min(required) // 10**9
                upper = max(required) // 10**9
                if bounds[1] < lower or bounds[0] > upper:
                    continue
            try:
                frame = _normalise_execution_1m_frame(pd.read_parquet(part))
            except Exception as exc:
                exclusions.append({
                    "path": str(part),
                    "reason": f"unreadable_fast_cover_candidate:{type(exc).__name__}",
                })
                continue
            keys = pd.to_datetime(frame["ts"], utc=True, errors="raise").astype("int64", copy=False)
            wanted = np.fromiter((int(key) in required for key in keys), dtype=bool, count=len(keys))
            if not bool(wanted.any()):
                continue
            selected_count += 1
            for row in frame.loc[wanted].to_dict(orient="records"):
                key = int(pd.Timestamp(row["ts"]).value)
                prior = selected_rows.get(key)
                if prior is not None and not _execution_1m_rows_equal(prior, row):
                    raise ValueError(
                        f"conflicting selected execution_1m rows for {symbol} at "
                        f"{pd.Timestamp(key, tz='UTC').isoformat()}"
                    )
                selected_rows[key] = row
                required.discard(key)
        if not required:
            exclusions.append({
                "path": str(symbol_dir),
                "reason": (
                    "research_fast_payload_verified_minimal_cover;"
                    f"selected_parts={selected_count};bypassed_parts={len(parts) - selected_count}"
                ),
            })
            fast_frame = pd.DataFrame(selected_rows.values()).sort_values("ts", kind="stable")
            return fast_frame.set_index("ts")[["open", "high", "low", "close"]], exclusions
        exclusions.append({
            "path": str(symbol_dir),
            "reason": "research_fast_payload_cover_incomplete_fell_back_to_full_conflict_scan",
        })
    for part in parts:
        try:
            frame = _normalise_execution_1m_frame(pd.read_parquet(part))
        except Exception as exc:
            bounds = all_bounds.get(part)
            requested_start_s = int(pd.Timestamp(start).value // 10**9)
            requested_end_s = int(pd.Timestamp(end).value // 10**9)
            superseded = bool(bounds) and any(
                other.name.startswith("part-")
                and other != part
                and other_bounds is not None
                # We only need a source covering the requested range, not
                # the corrupt file's unrelated earlier/later tail.
                and other_bounds[0] <= requested_start_s
                and other_bounds[1] >= requested_end_s
                for other, other_bounds in all_bounds.items()
            )
            if not superseded:
                raise ValueError(
                    f"unreadable required execution_1m source {part}: {exc}"
                ) from exc
            exclusions.append({
                "path": str(part),
                "reason": "unreadable_immutable_part_fully_superseded_by_part",
            })
            continue
        for row in frame.to_dict(orient="records"):
            key = int(pd.Timestamp(row["ts"]).value)
            prior = rows.get(key)
            if prior is not None and not _execution_1m_rows_equal(prior, row):
                raise ValueError(
                    f"conflicting execution_1m rows for {symbol} at "
                    f"{pd.Timestamp(key, tz='UTC').isoformat()}"
                )
            rows[key] = row
    selected = [
        row for key, row in rows.items()
        if pd.Timestamp(key, tz="UTC") >= start and pd.Timestamp(key, tz="UTC") <= end
    ]
    if not selected:
        return pd.DataFrame(columns=["open", "high", "low", "close"], index=pd.DatetimeIndex([], tz="UTC")), exclusions
    frame = pd.DataFrame(selected).sort_values("ts", kind="stable")
    return frame.set_index("ts")[["open", "high", "low", "close"]], exclusions


def _materialize_exact_paths(
    population: pd.DataFrame,
    *,
    minute_root: Path,
    prefer_covering_part: bool = False,
) -> tuple[ExactPaths, pd.DataFrame, pd.DataFrame]:
    """Join complete 720x1m paths only after score-only routing."""
    valid_parts: list[pd.DataFrame] = []
    entry_parts: list[np.ndarray] = []
    atr_parts: list[np.ndarray] = []
    high_parts: list[np.ndarray] = []
    low_parts: list[np.ndarray] = []
    close_parts: list[np.ndarray] = []
    coverage: list[dict[str, Any]] = []
    invalid_parts: list[pd.DataFrame] = []
    for symbol, group in population.groupby("symbol", sort=True):
        group = group.reset_index(drop=True)
        earliest = group["timestamp"].min() - pd.Timedelta(hours=ATR_SOURCE_LOOKBACK_HOURS, minutes=1)
        latest = group["entry_ts"].max() + pd.Timedelta(minutes=HORIZON_MINUTES - 1)
        # Read the immutable execution contract directly.  The generic OHLCV
        # store overlays an optional volume sidecar; that auxiliary source is
        # neither an input to the exact policy state machine nor part of the
        # source contract.  Going direct prevents a missing volume sidecar
        # from turning a valid OHLC path into an HPO failure.
        raw_minute, source_exclusions = _read_exact_ohlc(
            minute_root,
            str(symbol),
            start=earliest,
            end=latest,
            prefer_covering_part=prefer_covering_part,
        )
        minute = _clean_minute(raw_minute)
        if minute.empty:
            bad = group.copy()
            bad["outcome_invalid_reason"] = "missing_minute_source"
            invalid_parts.append(bad)
            coverage.append({
                "symbol": str(symbol), "candidate_rows": len(group), "valid_rows": 0,
                "reason": "missing_minute_source",
                "source_exclusions": json.dumps(source_exclusions, sort_keys=True),
            })
            continue
        atr = _causal_atr(minute)
        valid, locations, atr_values, reasons = _complete_mask(
            minute, atr, pd.DatetimeIndex(group["entry_ts"]),
        )
        bad = group.loc[~valid].copy()
        if not bad.empty:
            bad["outcome_invalid_reason"] = reasons[~valid]
            invalid_parts.append(bad)
        coverage.append({
            "symbol": str(symbol), "candidate_rows": int(len(group)), "valid_rows": int(valid.sum()),
            "reason": "ok" if bool(valid.all()) else "partial_outcome_coverage",
            "source_exclusions": json.dumps(source_exclusions, sort_keys=True),
        })
        if not valid.any():
            continue
        offsets = np.arange(HORIZON_MINUTES, dtype=np.int64)
        selected = locations[valid, None] + offsets[None, :]
        valid_parts.append(group.loc[valid].copy())
        entry_parts.append(minute["open"].to_numpy(float)[selected[:, 0]])
        atr_parts.append(atr_values[valid])
        high_parts.append(minute["high"].to_numpy(float)[selected].astype(np.float32, copy=False))
        low_parts.append(minute["low"].to_numpy(float)[selected].astype(np.float32, copy=False))
        close_parts.append(minute["close"].to_numpy(float)[selected].astype(np.float32, copy=False))
    if not valid_parts:
        raise RuntimeError("no complete one-minute policy paths after target-free routing")
    rows = pd.concat(valid_parts, ignore_index=True)
    paths = ExactPaths(
        rows=rows,
        entry=np.concatenate(entry_parts).astype(float, copy=False),
        atr=np.concatenate(atr_parts).astype(float, copy=False),
        high=np.concatenate(high_parts).astype(np.float32, copy=False),
        low=np.concatenate(low_parts).astype(np.float32, copy=False),
        close=np.concatenate(close_parts).astype(np.float32, copy=False),
    )
    if len(paths.rows) != len(paths.entry):
        raise AssertionError("exact one-minute path arrays lost candidate identity alignment")
    invalid = pd.concat(invalid_parts, ignore_index=True) if invalid_parts else pd.DataFrame(columns=list(population.columns) + ["outcome_invalid_reason"])
    return paths, pd.DataFrame(coverage), invalid


def _median_atr_fraction(paths: ExactPaths) -> float:
    values = paths.atr / np.maximum(paths.entry, 1e-12)
    values = values[np.isfinite(values) & (values > 0.0)]
    if not len(values):
        raise ValueError("no finite ATR fractions")
    return float(np.median(values))


def _load_policy(path: Path) -> tuple[RichPolicyParams, float, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = RichPolicyParams.from_mapping(dict(payload.get("params") or {}))
    median = float(payload.get("median_atr_fraction_fitted_on_complete_2024_development"))
    if not np.isfinite(median) or median <= 0.0:
        raise AssertionError("frozen rich policy lacks a valid ATR anchor")
    if not np.isclose(float(payload.get("cost_bps")), 100.0):
        raise AssertionError("frozen rich policy does not bind the 100-bps cost exactly once")
    return params, median, {"path": str(path), "sha256": _sha256(path), "params": params.to_dict(), "median_atr_fraction": median}


def _params_from_record(record: Mapping[str, Any]) -> RichPolicyParams:
    """Restore a policy row without converting optional ``None`` to ``NaN``.

    Parquet/DataFrame round trips represent optional numeric fields as NaN.
    In the rich policy contract, however, ``None`` means to use the shared ATR
    geometry while NaN produces an invalid distance.  Canonicalize only those
    declared optional fields at the HPO-to-tournament boundary.
    """
    payload: dict[str, Any] = {}
    for field in RichPolicyParams.__dataclass_fields__:
        if field not in record:
            continue
        value = record[field]
        if field in OPTIONAL_POLICY_FIELDS and pd.isna(value):
            value = None
        payload[field] = value
    return RichPolicyParams.from_mapping(payload)


def _hpo_branch_specs(
    total_trials: int,
    *,
    trials_per_branch: int | None = None,
) -> tuple[tuple[str, int], ...]:
    """Allocate an explicit search budget to both trailing implementations.

    ``fixed_trailing_gap_mult`` and ``giveback_beta`` drive mutually exclusive
    branches in the exact state machine. A single categorical TPE study can
    collapse into fixed gap before it has learned whether dynamic giveback is
    useful. Each branch therefore receives half of the development-only HPO
    budget; the frozen evaluation window never allocates trials.
    """
    if trials_per_branch is not None:
        if int(trials_per_branch) < 1:
            raise ValueError("trials_per_branch must be positive")
        return (
            ("fixed_gap", int(trials_per_branch)),
            ("dynamic_giveback", int(trials_per_branch)),
        )
    if total_trials < 2:
        raise ValueError("exact rich-policy HPO requires at least two trials to cover both trailing branches")
    fixed_gap_trials = (int(total_trials) + 1) // 2
    dynamic_giveback_trials = int(total_trials) - fixed_gap_trials
    return (
        ("fixed_gap", fixed_gap_trials),
        ("dynamic_giveback", dynamic_giveback_trials),
    )


def _early_stop_reached(*, trial_number: int, best_trial_number: int, patience: int) -> bool:
    """Return whether a branch has completed ``patience`` non-improving trials."""
    return patience > 0 and best_trial_number >= 0 and (trial_number - best_trial_number) >= patience


def _candidate_params(
    trial: optuna.Trial,
    incumbent: RichPolicyParams,
    *,
    trailing_family: str,
) -> RichPolicyParams:
    """Bounded rich surface with every active exact-v2 policy field covered.

    The fixed-gap and dynamic-giveback branches are mutually exclusive in the
    state machine.  Sampling a zero fixed gap explicitly activates the latter,
    so trailing power, squash and giveback beta are genuinely evaluated rather
    than merely written into an inactive policy object.  Likewise, the shared
    ATR branch clears the explicit SL/TP overrides so its base power and
    multiplier actually define both geometries.  Legacy capital-protection
    fields remain fixed because smooth capital protection is always enabled
    and those fields are not consulted by the exact-v2 smooth branch.
    """
    values = incumbent.to_dict()
    values.update({
        "sl_mult": trial.suggest_float("sl_mult", 2.5, 6.0),
        "trailing_activation_mult": trial.suggest_float("trailing_activation_mult", 0.5, 4.0),
        "fixed_trailing_gap_mult": (
            trial.suggest_categorical(
                "fixed_trailing_gap_mult", [0.04, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.35],
            )
            if trailing_family == "fixed_gap"
            else 0.0
        ),
        "trailing_power": trial.suggest_categorical(
            "trailing_power", [0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
        ),
        "trailing_squash_divisor": trial.suggest_categorical(
            "trailing_squash_divisor", [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
        ),
        "giveback_beta": trial.suggest_categorical(
            "giveback_beta", [0.20, 0.35, 0.50, 0.65, 0.80, 1.00],
        ),
        "sl_abs_floor_pct": trial.suggest_categorical("sl_abs_floor_pct", [0.0, 0.004, 0.006, 0.008, 0.010]),
        "sl_abs_cap_pct": trial.suggest_categorical("sl_abs_cap_pct", [0.015, 0.020, 0.030, 0.050, 0.060]),
        "trailing_activation_min_pct": trial.suggest_categorical("trailing_activation_min_pct", [0.0, 0.003, 0.005, 0.0075, 0.010]),
        "trailing_activation_cap_pct": trial.suggest_categorical("trailing_activation_cap_pct", [0.010, 0.015, 0.020, 0.030, 0.050]),
        "trailing_activation_decay_half_life_bars": trial.suggest_categorical("trailing_activation_decay_half_life_bars", [0.0, 8.0, 16.0, 24.0, 32.0]),
        "trailing_activation_decay_start_bars": trial.suggest_categorical("trailing_activation_decay_start_bars", [0, 4, 8, 16]),
        "trailing_activation_min_mult": trial.suggest_categorical("trailing_activation_min_mult", [0.35, 0.50, 0.70, 0.85, 1.0]),
        "smooth_capital_protection_enabled": True,
        "protection_activation_atr": trial.suggest_categorical("protection_activation_atr", [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]),
        "protection_strength": trial.suggest_categorical("protection_strength", [0.25, 0.5, 0.75]),
        "protection_power": trial.suggest_categorical("protection_power", [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]),
        "adverse_exit_theta": None,
    })
    if trailing_family not in {"fixed_gap", "dynamic_giveback"}:
        raise ValueError(f"unknown trailing family: {trailing_family}")
    geometry_family = trial.suggest_categorical("geometry_family", ["separate_sl_tp", "shared_atr"])
    if geometry_family == "separate_sl_tp":
        values.update({
            "sl_atr_power": trial.suggest_categorical("sl_atr_power", [0.70, 0.85, 1.0, 1.15, 1.30]),
            "sl_atr_multiplier": trial.suggest_categorical("sl_atr_multiplier", [0.75, 1.0, 1.25]),
            "tp_atr_power": trial.suggest_categorical("tp_atr_power", [0.70, 0.85, 1.0, 1.15, 1.30]),
            "tp_atr_multiplier": trial.suggest_categorical("tp_atr_multiplier", [0.75, 1.0, 1.25]),
        })
    else:
        values.update({
            "atr_power": trial.suggest_categorical("atr_power", [0.70, 0.85, 1.0, 1.15, 1.30]),
            "atr_multiplier": trial.suggest_categorical("atr_multiplier", [0.75, 1.0, 1.25]),
            "sl_atr_power": None, "sl_atr_multiplier": None,
            "tp_atr_power": None, "tp_atr_multiplier": None,
        })
    if values["sl_abs_floor_pct"] > values["sl_abs_cap_pct"]:
        values["sl_abs_cap_pct"] = values["sl_abs_floor_pct"]
    if values["trailing_activation_min_pct"] > values["trailing_activation_cap_pct"]:
        values["trailing_activation_cap_pct"] = values["trailing_activation_min_pct"]
    return RichPolicyParams.from_mapping(values)


def _fitted_params(params: RichPolicyParams, calibration: ExactPaths, *, median_atr: float) -> RichPolicyParams:
    if not params.adverse_exit_enabled:
        return params
    theta = fit_adverse_theta_exact_1m(
        entry=calibration.entry, atr=calibration.atr, highs=calibration.high, lows=calibration.low,
        params=params, median_atr_fraction=median_atr, side="long",
    )
    return RichPolicyParams.from_mapping({**params.to_dict(), "adverse_exit_theta": theta})


def _replay(paths: ExactPaths, params: RichPolicyParams, *, median_atr: float) -> dict[str, np.ndarray]:
    result = replay_exact_1m_rich_policy_v2(
        entry=paths.entry, atr=paths.atr, highs=paths.high, lows=paths.low, closes=paths.close,
        entry_timestamps=paths.rows["entry_ts"], params=params, median_atr_fraction=median_atr,
        extensions=RichExitExtensions(), contract=Exact1mRichV2ExecutionContract(entry_delay_minutes=5),
    )
    valid = np.asarray(result["path_valid"], dtype=bool)
    if not valid.all() or not np.isfinite(np.asarray(result["net_bps"], dtype=float)).all():
        raise AssertionError("complete exact paths produced an incomplete rich-policy outcome")
    if not np.allclose(np.asarray(result["gross_bps"], float) - np.asarray(result["net_bps"], float), 100.0, atol=1e-8, rtol=0.0):
        raise AssertionError("exact rich policy must apply the 100-bps cost exactly once")
    return result


def _portfolio(paths: ExactPaths, replay: Mapping[str, np.ndarray], *, arm: str) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    exit_ts = pd.to_datetime(np.asarray(replay["exit_timestamp"]), utc=True, errors="raise")
    holding = np.maximum(
        1,
        np.ceil((exit_ts - paths.rows["entry_ts"]).dt.total_seconds() / 900.0),
    ).astype(int)
    candidates = pd.DataFrame({
        "timestamp": paths.rows["entry_ts"], "decision_timestamp": paths.rows["timestamp"],
        "candidate_id": paths.rows["candidate_id"].astype(str), "symbol": paths.rows["symbol"].astype(str),
        "side": "long", "strategy_id": arm, "policy_archetype": arm,
        "normalized_rank_score": 1.0, "strategy_rank_pct": 1.0, "base_strategy_threshold": 0.0,
        "calibrated_score": 1.0, "portfolio_priority_adjustment": pd.to_numeric(paths.rows["priority_bps"], errors="raise"),
        "entry_price": paths.entry, "exit_timestamp": exit_ts, "exit_price": np.asarray(replay["exit_price"], dtype=float),
        "net_return": np.asarray(replay["net_bps"], dtype=float) / 10_000.0,
        "gross_return": np.asarray(replay["gross_bps"], dtype=float) / 10_000.0,
        "holding_bars": holding, "simple_policy_exit_reason": np.asarray(replay["exit_reason"], dtype=object),
        "fees_bps": 100.0, "expected_friction_bps": 0.0, "price_gap_bps": 0.0,
        "liquidity_capacity_weight": 1.0, "policy_outcome_available": True,
    })
    candidates = normalise_candidate_table(candidates)
    decisions, equity, _ = replay_candidates(
        candidates, canonical_portfolio_params(), mode="global_auction", ev_curve=CAUSAL_AUCTION_CURVE, market_mode="perp",
    )
    positions = pd.to_numeric(decisions["candidate_index"], errors="raise").astype(int).to_numpy()
    decisions = decisions.copy()
    decisions["candidate_id"] = candidates.iloc[positions]["candidate_id"].to_numpy()
    decisions["decision_timestamp"] = candidates.iloc[positions]["decision_timestamp"].to_numpy()
    accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)].copy()
    accepted["net_bps"] = pd.to_numeric(accepted["position_net_return"], errors="raise") * 10_000.0
    accepted["month"] = pd.to_datetime(accepted["decision_timestamp"], utc=True).dt.strftime("%Y-%m")
    accepted["week"] = pd.to_datetime(accepted["decision_timestamp"], utc=True).dt.strftime("%G-W%V")
    monthly = accepted.groupby("month", sort=True).agg(trades=("candidate_id", "size"), net_bps_per_trade=("net_bps", "mean"), total_net_bps=("net_bps", "sum")).reset_index()
    weekly = accepted.groupby("week", sort=True).agg(trades=("candidate_id", "size"), net_bps_per_trade=("net_bps", "mean"), total_net_bps=("net_bps", "sum")).reset_index()
    values = monthly["net_bps_per_trade"].to_numpy(float)
    median = float(np.median(values)) if len(values) else float("nan")
    mad = float(np.median(np.abs(values - median))) if len(values) else float("nan")
    worst = float(np.min(values)) if len(values) else float("nan")
    metrics = compute_replay_metrics(candidates, decisions, equity, params=canonical_portfolio_params())
    result = {
        "portfolio_entries": int(len(accepted)),
        "portfolio_net_bps_per_trade": float(accepted["net_bps"].mean()) if len(accepted) else float("nan"),
        "portfolio_total_net_bps": float(accepted["net_bps"].sum()) if len(accepted) else 0.0,
        "portfolio_median_month_bps": median, "portfolio_month_mad_bps": mad,
        "portfolio_worst_month_bps": worst,
        "portfolio_selection_score": median - 0.5 * mad - max(0.0, -worst) if len(values) else float("-inf"),
        "portfolio_max_drawdown": float(metrics.get("max_drawdown", np.nan)),
        "portfolio_sortino": float(metrics.get("sortino", np.nan)),
        "portfolio_worst_week_return": float(metrics.get("worst_week", np.nan)),
    }
    return result, {"candidates": candidates, "decisions": decisions, "equity": equity, "accepted": accepted, "monthly": monthly, "weekly": weekly}


def _sample(paths: ExactPaths, *, rows_per_month: int) -> ExactPaths:
    if rows_per_month <= 0:
        return paths
    months = pd.to_datetime(paths.rows["timestamp"], utc=True).dt.strftime("%Y-%m").to_numpy()
    selected: list[np.ndarray] = []
    for month in sorted(set(months)):
        positions = np.flatnonzero(months == month)
        if len(positions) > rows_per_month:
            hashes = pd.util.hash_pandas_object(paths.rows.iloc[positions]["candidate_id"].astype(str), index=False).to_numpy(np.uint64)
            positions = positions[np.argsort(hashes, kind="stable")[:rows_per_month]]
        selected.append(positions)
    mask = np.zeros(len(paths.rows), dtype=bool)
    mask[np.concatenate(selected)] = True
    return paths.take(mask)


def _window(paths: ExactPaths, start: pd.Timestamp, end: pd.Timestamp) -> ExactPaths:
    mask = paths.rows["timestamp"].ge(start) & paths.rows["timestamp"].lt(end)
    if not mask.any():
        raise RuntimeError(f"no exact paths in window {start}..{end}")
    return paths.take(mask.to_numpy())


def run(args: argparse.Namespace) -> Path:
    output = Path(args.out_dir).resolve()
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    start, end = _utc(args.start), _utc(args.end)
    calibration_end, selection_end = _utc(args.calibration_end), _utc(args.selection_end)
    if not (start < calibration_end < selection_end < end):
        raise ValueError("require start < calibration_end < selection_end < end")
    output.mkdir(parents=True, exist_ok=False)
    source = Path(args.dual_predictions).resolve()
    population = _target_free_population(source, start=start, end=end, threshold_bps=float(args.dual_threshold_bps))
    population.to_parquet(output / "target_free_candidates.parquet", index=False, compression="zstd")
    candidate_manifest = {
        "schema": "strict_r3_p8u_exact_1m_simple_policy_candidate_v1", "target_free": True,
        "rows": int(len(population)), "candidate_sha256": _sha256(output / "target_free_candidates.parquet"),
        "selection": {"bcf_mc1_expected_bps_gte": float(args.dual_threshold_bps), "current_mc1_expected_bps_gte": float(args.dual_threshold_bps), "priority": "bcf_mc1_expected_bps"},
        "selection_inputs": ["candidate_id", "__decision_ts__", "__symbol__", "side_name", "bcf_mc1_expected_bps", "current_mc1_expected_bps"],
        "forbidden_selection_inputs": ["policy_path_valid", "policy_net_bps", "policy_gross_bps", "outcome", "label", "exit"],
        "entry": "decision + five completed minutes", "source": str(source), "source_sha256": _sha256(source),
    }
    _write_once(output / "target_free_candidate_manifest.json", candidate_manifest)
    if args.exact_path_source_root is not None:
        paths, coverage, invalid, path_source_audit = _load_frozen_exact_paths(
            Path(args.exact_path_source_root), population=population, start=start, end=end,
        )
    else:
        paths, coverage, invalid = _materialize_exact_paths(population, minute_root=Path(args.minute_root).resolve())
        path_source_audit = {
            "root": str(Path(args.minute_root).resolve()), "materialized_for_this_run": True,
            "valid_exact_paths": int(len(paths.rows)), "invalid_paths_after_route": int(len(invalid)),
        }
    coverage.to_parquet(output / "source_coverage.parquet", index=False, compression="zstd")
    invalid.to_parquet(output / "invalid_outcomes_after_route.parquet", index=False, compression="zstd")
    paths.rows.to_parquet(output / "valid_exact_paths_rows.parquet", index=False, compression="zstd")
    _write_once(output / "exact_path_source_manifest.json", path_source_audit)
    incumbent, frozen_median, policy_audit = _load_policy(Path(args.frozen_policy).resolve())
    calibration = _window(paths, start, calibration_end)
    selection = _window(paths, calibration_end, selection_end)
    evaluation = _window(paths, selection_end, end)
    median_atr = _median_atr_fraction(calibration)
    sample = _sample(selection, rows_per_month=int(args.hpo_rows_per_month))
    contract = Exact1mRichV2ExecutionContract(entry_delay_minutes=5)
    literal_replay = _replay(selection, incumbent, median_atr=frozen_median)
    literal_metrics, _ = _portfolio(selection, literal_replay, arm="literal_current_frozen")
    incumbent_recalibrated = _fitted_params(incumbent, calibration, median_atr=median_atr)
    control_replay = _replay(selection, incumbent_recalibrated, median_atr=median_atr)
    control_metrics, _ = _portfolio(selection, control_replay, arm="current_geometry_recalibrated")
    branch_specs = _hpo_branch_specs(
        int(args.trials),
        trials_per_branch=args.trials_per_trailing_branch,
    )
    records: list[dict[str, Any]] = []
    for branch_index, (trailing_family, branch_trials) in enumerate(branch_specs):
        sampler = optuna.samplers.TPESampler(
            seed=int(args.seed) + branch_index,
            multivariate=True,
            group=True,
            n_startup_trials=min(12, branch_trials),
        )
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=min(12, branch_trials)),
        )
        best_objective = float("-inf")
        best_trial_number = -1

        def objective(trial: optuna.Trial, *, _family: str = trailing_family) -> float:
            nonlocal best_objective, best_trial_number
            params = _candidate_params(trial, incumbent, trailing_family=_family)
            fitted = _fitted_params(params, calibration, median_atr=median_atr)
            replay = _replay(sample, fitted, median_atr=median_atr)
            trial_id = f"{_family}_{trial.number:03d}"
            metrics, _ = _portfolio(sample, replay, arm=f"hpo_{trial_id}")
            value = float(metrics["portfolio_selection_score"])
            records.append({
                "trial": int(trial.number),
                "trial_id": trial_id,
                "trailing_family": _family,
                "objective": value,
                "geometry_family": trial.params["geometry_family"],
                **metrics,
                **fitted.to_dict(),
            })
            trial.set_user_attr("params", fitted.to_dict())
            if value > best_objective + 1e-12:
                best_objective = value
                best_trial_number = int(trial.number)
            elif _early_stop_reached(
                trial_number=int(trial.number),
                best_trial_number=best_trial_number,
                patience=int(args.early_stop_patience),
            ):
                study.stop()
            return value

        study.optimize(objective, n_trials=branch_trials, n_jobs=1, show_progress_bar=False)
    trials = pd.DataFrame(records).sort_values(["objective", "trial_id"], ascending=[False, True], kind="stable")
    trials.to_parquet(output / "hpo_trials.parquet", index=False, compression="zstd")
    finalists: list[tuple[str, RichPolicyParams]] = [("current_geometry_recalibrated", incumbent_recalibrated)]
    seen = {json.dumps(incumbent_recalibrated.to_dict(), sort_keys=True, default=str)}
    for row in trials.head(int(args.finalists)).itertuples(index=False):
        params = _params_from_record(row._asdict())
        key = json.dumps(params.to_dict(), sort_keys=True, default=str)
        if key not in seen:
            finalists.append((f"hpo_{str(row.trial_id)}", params))
            seen.add(key)
    tournament: list[dict[str, Any]] = []
    for name, params in finalists:
        replay = _replay(selection, params, median_atr=median_atr)
        metrics, _ = _portfolio(selection, replay, arm=f"selection_{name}")
        tournament.append({"arm": name, **metrics, **params.to_dict()})
    tournament_frame = pd.DataFrame(tournament).sort_values(
        ["portfolio_selection_score", "portfolio_net_bps_per_trade", "portfolio_total_net_bps", "arm"],
        ascending=[False, False, False, True], kind="stable",
    ).reset_index(drop=True)
    tournament_frame.to_parquet(output / "june_full_tournament.parquet", index=False, compression="zstd")
    winner = _params_from_record(tournament_frame.iloc[0].to_dict())
    calibration_plus_selection = paths.take(paths.rows["timestamp"].lt(selection_end).to_numpy())
    winner_for_july = _fitted_params(RichPolicyParams.from_mapping({**winner.to_dict(), "adverse_exit_theta": None}), calibration_plus_selection, median_atr=_median_atr_fraction(calibration_plus_selection))
    evaluation_rows: list[dict[str, Any]] = []
    for name, params, anchor in (
        ("literal_current_frozen", incumbent, frozen_median),
        ("current_geometry_recalibrated", incumbent_recalibrated, median_atr),
        ("june_selected_challenger", winner_for_july, _median_atr_fraction(calibration_plus_selection)),
    ):
        replay = _replay(evaluation, params, median_atr=anchor)
        metrics, frames = _portfolio(evaluation, replay, arm=f"frozen_{name}")
        for frame_name, frame in frames.items():
            frame.to_parquet(output / f"frozen_{name}_{frame_name}.parquet", index=False, compression="zstd")
        evaluation_rows.append({"arm": name, **metrics, **params.to_dict(), "exact_1m_receipt": json.dumps(exact_1m_rich_v2_receipt(params=params, extensions=RichExitExtensions(), replay=replay, contract=contract), sort_keys=True)})
    evaluation_frame = pd.DataFrame(evaluation_rows)
    evaluation_frame.to_parquet(output / "frozen_evaluation.parquet", index=False, compression="zstd")
    _write_once(output / "run_manifest.json", {
        "schema": "strict_r3_p8u_exact_1m_simple_policy_optimiser_v1", "status": "complete", "research_only": True,
        "temporal_protocol": {"calibration": [str(start), str(calibration_end)], "hpo_and_tournament": [str(calibration_end), str(selection_end)], "frozen_confirmation": [str(selection_end), str(end)]},
        "one_minute_contract": contract.to_dict(), "one_minute_contract_hash": contract.hash,
        "policy_cost": "100 bps exactly once in the exact outcome; no second auction debit",
        "candidate_route": candidate_manifest, "exact_path_source": path_source_audit,
        "valid_exact_paths": int(len(paths.rows)), "invalid_paths_after_route": int(len(invalid)),
        "frozen_policy": policy_audit, "literal_current_selection_metrics": literal_metrics,
        "current_geometry_recalibrated_selection_metrics": control_metrics,
        "hpo_trials": int(len(records)),
        "hpo_branch_trials_planned": {name: int(count) for name, count in branch_specs},
        "hpo_branch_trials_completed": {
            name: int((trials["trailing_family"] == name).sum())
            for name, _ in branch_specs
        },
        "hpo_early_stop_patience": int(args.early_stop_patience),
        "hpo_rows_per_month": int(args.hpo_rows_per_month),
        "development_selected_arm": str(tournament_frame.iloc[0]["arm"]), "final_window_is_evaluation_only": True,
        "prohibitions": ["no_15m_exit_proxy", "no_exchange_io", "no_live_state", "no_policy_promotion"],
    })
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dual-predictions", type=Path, default=DEFAULT_DUAL)
    parser.add_argument("--frozen-policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--minute-root", type=Path, default=DEFAULT_MINUTE_ROOT)
    parser.add_argument(
        "--exact-path-source-root", type=Path, default=DEFAULT_EXACT_PATH_SOURCE,
        help=(
            "Immutable target-free exact-1m path panel to hash-bind and reuse. "
            "Pass an empty value is not supported; use a distinct source panel or "
            "modify the research contract to materialize local paths anew."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start", default="2026-05-01T00:00:00Z")
    parser.add_argument("--calibration-end", default="2026-06-01T00:00:00Z")
    parser.add_argument("--selection-end", default="2026-08-01T00:00:00Z")
    parser.add_argument("--end", default="2026-08-28T00:00:00Z")
    parser.add_argument("--dual-threshold-bps", type=float, default=50.0)
    parser.add_argument("--trials", type=int, default=36)
    parser.add_argument(
        "--trials-per-trailing-branch",
        type=int,
        default=None,
        help="Run this many development HPO trials in each mutually exclusive trailing branch; overrides --trials.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=30,
        help="Stop each branch after this many non-improving trials; set 0 to disable.",
    )
    parser.add_argument("--finalists", type=int, default=5)
    parser.add_argument("--hpo-rows-per-month", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
