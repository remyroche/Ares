#!/usr/bin/env python3
"""Materialise training-only policy/path semantics for orthogonal meta research.

The input path panels are already complete H12 outcome labels.  This producer
never changes candidates, features, or target-free scores.  Its output is a
separate post-resolution label relation keyed by ``candidate_id``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_rich_policy import (  # noqa: E402
    RichPolicyParams,
    _activation_distance,
    _barrier_distances,
    _stop_distance,
)


SCHEMA = "strict_r3_orthogonal_meta_semantics_v1"
HORIZON_HOURS = 12
COST_FLOOR_BPS = 100.0
BAR_MINUTES = 15
HORIZON_BARS = HORIZON_HOURS * 60 // BAR_MINUTES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*.parquet")) if path.is_dir() else [path]:
        digest.update(str(child).encode())
        with child.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _first_true(mask: np.ndarray) -> np.ndarray:
    """One-based first index, zero when no event occurs."""
    found = mask.any(axis=1)
    out = np.zeros(mask.shape[0], dtype=np.int16)
    out[found] = np.argmax(mask[found], axis=1).astype(np.int16) + 1
    return out


def _load_15m_bars(bars_root: Path, symbol: str) -> pd.DataFrame:
    path = bars_root / f"{str(symbol).lower().replace('/', '')}_15m.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["high", "low"])
    bars = pd.read_parquet(path, columns=["high", "low"])
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError(f"15m source index is not datetime: {path}")
    index = pd.DatetimeIndex(bars.index)
    bars.index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    return bars.loc[~bars.index.duplicated(keep="last")].sort_index()


def _complete_windows(
    bars: pd.DataFrame,
    decisions: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return only full post-decision 15m paths; no bar is forward-filled."""
    decisions = pd.DatetimeIndex(pd.to_datetime(decisions, utc=True, errors="raise"))
    empty = np.full((len(decisions), HORIZON_BARS), np.nan, dtype=np.float32)
    if bars.empty:
        return np.zeros(len(decisions), dtype=bool), empty, empty.copy()
    start = min(pd.Timestamp(decisions.min()), pd.Timestamp(bars.index.min())).floor("15min")
    end = max(
        pd.Timestamp(decisions.max()) + pd.Timedelta(hours=HORIZON_HOURS),
        pd.Timestamp(bars.index.max()),
    ).ceil("15min")
    grid = pd.date_range(start, end, freq="15min", inclusive="left", tz="UTC")
    values = bars.reindex(grid).loc[:, ["high", "low"]].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    offsets = ((decisions - start) / pd.Timedelta(minutes=BAR_MINUTES)).astype(np.int64)
    high, low = empty.copy(), empty.copy()
    usable = (offsets >= 0) & (offsets + HORIZON_BARS <= len(grid))
    for local in np.flatnonzero(usable):
        path = values[offsets[local] : offsets[local] + HORIZON_BARS]
        high[local], low[local] = path[:, 0], path[:, 1]
    complete = (
        np.isfinite(high).all(axis=1)
        & np.isfinite(low).all(axis=1)
        & (high > 0.0).all(axis=1)
        & (high >= low).all(axis=1)
    )
    return complete, high, low


def _policy_tbm(
    frame: pd.DataFrame,
    *,
    bars_root: Path,
    params: RichPolicyParams,
    median_atr_fraction: float,
) -> pd.DataFrame:
    """Build frozen-policy TBM descriptors from the actual complete 15m path.

    The initial lower barrier is exactly the frozen RichPolicy stop.  The upper
    barrier is the frozen initial trailing activation floored at the declared
    100-bps policy cost.  Same-bar high/low crossings are persisted as
    ambiguous instead of granting an arbitrary favorable ordering.
    """
    result = frame.loc[:, ["candidate_id"]].copy()
    result["semantic_tbm_path_complete"] = False
    result["semantic_tbm_event"] = pd.Series(pd.NA, index=result.index, dtype="string")
    for field in (
        "semantic_upper_distance_atr", "semantic_lower_distance_atr",
        "semantic_upper_bar", "semantic_lower_bar", "semantic_time_to_event_h",
    ):
        result[field] = np.nan
    required = (
        np.isfinite(pd.to_numeric(frame["entry_price"], errors="coerce").to_numpy(float))
        & (pd.to_numeric(frame["entry_price"], errors="coerce").to_numpy(float) > 0.0)
        & np.isfinite(pd.to_numeric(frame["path_arch_atr_fraction"], errors="coerce").to_numpy(float))
        & (pd.to_numeric(frame["path_arch_atr_fraction"], errors="coerce").to_numpy(float) > 0.0)
    )
    for symbol, positions in frame.loc[required].groupby("__symbol__", sort=True).groups.items():
        index = np.asarray(list(positions), dtype=np.int64)
        rows = frame.loc[index].reset_index(drop=True)
        bars = _load_15m_bars(bars_root, str(symbol))
        complete, high, low = _complete_windows(bars, rows["__decision_ts__"])
        if not complete.any():
            continue
        local = np.flatnonzero(complete)
        entry = pd.to_numeric(rows.loc[local, "entry_price"], errors="coerce").to_numpy(float)
        atr_fraction = pd.to_numeric(rows.loc[local, "path_arch_atr_fraction"], errors="coerce").to_numpy(float)
        sl_raw, tp_raw = _barrier_distances(
            entry,
            entry * atr_fraction,
            params,
            median_atr_fraction=median_atr_fraction,
        )
        lower = _stop_distance(sl_raw, entry, params)
        upper = np.maximum(
            _activation_distance(tp_raw, entry, params, bar=0),
            entry * (COST_FLOOR_BPS / 10_000.0),
        )
        upper_bar = _first_true(high[local] >= entry[:, None] + upper[:, None])
        lower_bar = _first_true(low[local] <= entry[:, None] - lower[:, None])
        upper_hit, lower_hit = upper_bar > 0, lower_bar > 0
        same = upper_hit & lower_hit & (upper_bar == lower_bar)
        upper_first = upper_hit & (~lower_hit | (upper_bar < lower_bar))
        lower_first = lower_hit & (~upper_hit | (lower_bar < upper_bar))
        event = np.full(len(local), "vertical", dtype=object)
        event[upper_first] = "upper_first"
        event[lower_first] = "lower_first"
        event[same] = "ambiguous"
        target_index = index[local]
        result.loc[target_index, "semantic_tbm_path_complete"] = True
        result.loc[target_index, "semantic_tbm_event"] = event
        result.loc[target_index, "semantic_upper_distance_atr"] = upper / (entry * atr_fraction)
        result.loc[target_index, "semantic_lower_distance_atr"] = lower / (entry * atr_fraction)
        result.loc[target_index, "semantic_upper_bar"] = upper_bar
        result.loc[target_index, "semantic_lower_bar"] = lower_bar
        first_bar = np.where(upper_first, upper_bar, np.where(lower_first, lower_bar, np.nan))
        result.loc[target_index, "semantic_time_to_event_h"] = first_bar * (BAR_MINUTES / 60.0)
    return result


def _path_axes(
    frame: pd.DataFrame,
    tbm: pd.DataFrame,
) -> pd.DataFrame:
    """Derive fixed outcome descriptors.  They are never inference inputs."""
    out = frame.loc[:, ["candidate_id", "__decision_ts__", "supportive_label_available_ts"]].copy()
    valid = (
        pd.to_numeric(frame["supportive_path_valid"], errors="coerce").fillna(0).astype(bool)
        & frame["policy_path_valid"].fillna(False).astype(bool)
        & tbm["semantic_tbm_path_complete"].to_numpy(bool)
    ).to_numpy(bool)
    out["semantic_path_valid"] = valid
    out["semantic_label_available_ts"] = pd.to_datetime(
        frame["supportive_label_available_ts"], utc=True, errors="coerce"
    )
    # All semantic values deliberately remain missing for invalid paths.  An
    # unavailable path is not an economic failure and cannot enter fitting.
    for field in (
        "semantic_sequence", "semantic_speed", "semantic_persistence",
        "semantic_preop_adversity", "semantic_conversion", "semantic_exit",
        "semantic_composite", "semantic_tbm_event",
    ):
        out[field] = pd.Series(pd.NA, index=out.index, dtype="string")
    for field in (
        "semantic_post_upper_giveback_atr", "semantic_policy_capture_bps",
    ):
        out[field] = np.nan

    if not valid.any():
        return out
    idx = np.flatnonzero(valid)
    atr_fraction = pd.to_numeric(frame.loc[valid, "path_arch_atr_fraction"], errors="coerce").to_numpy(float)
    tbm_valid = tbm.loc[valid].reset_index(drop=True)
    event = tbm_valid["semantic_tbm_event"].astype(str).to_numpy(object)
    upper_bar = pd.to_numeric(tbm_valid["semantic_upper_bar"], errors="coerce").to_numpy(float)
    lower_bar = pd.to_numeric(tbm_valid["semantic_lower_bar"], errors="coerce").to_numpy(float)
    upper_hit = event == "upper_first"
    lower_first = event == "lower_first"
    upper_first = event == "upper_first"
    upper_h = pd.to_numeric(tbm_valid["semantic_time_to_event_h"], errors="coerce").to_numpy(float)

    peak = pd.to_numeric(frame.loc[valid, "path_arch_peak_mfe_atr"], errors="coerce").to_numpy(float)
    pre_mae = pd.to_numeric(frame.loc[valid, "path_arch_mae_before_meaningful_mfe_r"], errors="coerce").to_numpy(float)
    # The source expresses MAE in side-normalised return units; divide by ATR
    # so semantics remain portable across symbols and volatility.
    pre_mae_atr = np.divide(pre_mae, atr_fraction, out=np.full_like(pre_mae, np.nan), where=atr_fraction > 0.0)
    retention = pd.to_numeric(frame.loc[valid, "path_arch_peak_retention_ratio"], errors="coerce").to_numpy(float)
    final_return = pd.to_numeric(frame.loc[valid, "path_arch_final_return_r"], errors="coerce").to_numpy(float)
    policy_net = pd.to_numeric(frame.loc[valid, "policy_net_bps"], errors="coerce").to_numpy(float)
    exit_reason = frame.loc[valid, "policy_exit_reason"].astype(str).to_numpy(object)

    sequence = np.full(len(idx), "no_opportunity", dtype=object)
    sequence[upper_first & (pre_mae_atr <= 0.5)] = "favourable_first_clean"
    sequence[upper_first & (pre_mae_atr > 0.5)] = "favourable_first_after_mild_adversity"
    sequence[lower_first & upper_hit] = "adverse_first_recovery"
    sequence[lower_first & ~upper_hit] = "adverse_first_failure"
    speed = np.full(len(idx), "never", dtype=object)
    speed[upper_first & (upper_h <= 2.0)] = "fast"
    speed[upper_first & (upper_h > 2.0) & (upper_h <= 6.0)] = "normal"
    speed[upper_first & (upper_h > 6.0)] = "slow"
    persistence = np.full(len(idx), "large_giveback", dtype=object)
    persistence[retention >= 0.75] = "persistent"
    persistence[(retention >= 0.50) & (retention < 0.75)] = "partial_giveback"
    persistence[final_return <= 0.0] = "full_reversal"
    adversity = np.full(len(idx), "severe", dtype=object)
    adversity[pre_mae_atr <= 0.25] = "clean"
    adversity[(pre_mae_atr > 0.25) & (pre_mae_atr <= 0.75)] = "brief"
    adversity[(pre_mae_atr > 0.75) & (pre_mae_atr <= 1.50)] = "sustained"
    conversion = np.where(upper_hit, np.where(policy_net >= 50.0, "high_mfe_good_capture", "high_mfe_poor_capture"), np.where(policy_net >= 50.0, "low_mfe_good_capture", "low_mfe_poor_capture"))
    composite = np.full(len(idx), "no_opportunity_timeout", dtype=object)
    composite[(sequence == "favourable_first_clean") & (speed == "fast") & (persistence == "persistent")] = "clean_fast_persistent_winner"
    composite[(sequence == "favourable_first_clean") & (speed != "fast") & (persistence == "persistent")] = "clean_slow_persistent_winner"
    composite[sequence == "adverse_first_recovery"] = "adverse_first_recovery"
    composite[upper_hit & (persistence == "large_giveback")] = "transient_favourable_spike_giveback"
    composite[upper_hit & (adversity == "sustained")] = "choppy_eventually_favourable"
    composite[upper_hit & (policy_net < 50.0)] = "opportunity_policy_capture_fails"
    composite[sequence == "adverse_first_failure"] = "early_adverse_failure"

    giveback_atr = peak * np.maximum(1.0 - retention, 0.0)
    for field in (
        "semantic_upper_distance_atr", "semantic_lower_distance_atr",
        "semantic_upper_bar", "semantic_lower_bar", "semantic_time_to_event_h",
    ):
        out.loc[idx, field] = pd.to_numeric(tbm_valid[field], errors="coerce").to_numpy(float)
    out.loc[idx, "semantic_post_upper_giveback_atr"] = giveback_atr
    out.loc[idx, "semantic_policy_capture_bps"] = policy_net
    out.loc[idx, "semantic_sequence"] = sequence
    out.loc[idx, "semantic_speed"] = speed
    out.loc[idx, "semantic_persistence"] = persistence
    out.loc[idx, "semantic_preop_adversity"] = adversity
    out.loc[idx, "semantic_conversion"] = conversion
    out.loc[idx, "semantic_exit"] = exit_reason
    out.loc[idx, "semantic_composite"] = composite
    out.loc[idx, "semantic_tbm_event"] = event
    return out


def _months(root: Path) -> list[str]:
    return sorted(part.name.split("=", 1)[1] for part in root.glob("month=*") if part.is_dir())


def _write_manifest(out: Path, payload: dict[str, object]) -> None:
    target = out / "run_manifest.json"
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-root", type=Path, required=True)
    parser.add_argument("--policy-json", type=Path, required=True)
    parser.add_argument(
        "--canonical-policy-labels",
        type=Path,
        required=True,
        help="reconciled rich-policy labels; replaces stale convenience fields in path panels",
    )
    parser.add_argument("--bars-root", type=Path, default=Path("15m_ohlcv_perp"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--months", help="optional comma-separated YYYY-MM subset")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"immutable output already exists: {args.out}")
    policy = json.loads(args.policy_json.read_text())
    params = RichPolicyParams.from_mapping(policy["params"])
    median_atr_fraction = float(
        policy.get(
            "median_atr_fraction",
            policy["median_atr_fraction_fitted_on_complete_2024_development"],
        )
    )
    policy_labels = pd.read_parquet(
        args.canonical_policy_labels,
        columns=(
            "candidate_id", "__decision_ts__", "policy_path_valid", "policy_net_bps",
            "policy_exit_reason", "policy_label_available_ts",
        ),
    )
    if policy_labels["candidate_id"].duplicated().any():
        raise AssertionError("canonical rich-policy labels have duplicate candidate IDs")
    policy_labels["__decision_ts__"] = pd.to_datetime(
        policy_labels["__decision_ts__"], utc=True, errors="raise"
    )
    available_months = set(policy_labels["__decision_ts__"].dt.strftime("%Y-%m"))
    requested = (
        args.months.split(",")
        if args.months
        else [month for month in _months(args.path_root) if month in available_months]
    )
    if not requested:
        raise AssertionError("no path-label months overlap canonical rich-policy labels")
    args.out.mkdir(parents=True)
    coverage: list[dict[str, object]] = []
    for month in requested:
        source = args.path_root / f"month={month}" / "side=long.parquet"
        if not source.exists():
            raise FileNotFoundError(source)
        labels = pd.read_parquet(source)
        stale_policy = [
            field for field in (
                "policy_path_valid", "policy_net_bps", "policy_exit_reason",
                "policy_label_available_ts",
            ) if field in labels
        ]
        labels = labels.drop(columns=stale_policy).merge(
            policy_labels.drop(columns="__decision_ts__"),
            on="candidate_id",
            how="left",
            validate="one_to_one",
        )
        if labels["policy_path_valid"].isna().any():
            raise AssertionError(f"{month}: canonical policy labels missed path identities")
        tbm = _policy_tbm(
            labels,
            bars_root=args.bars_root,
            params=params,
            median_atr_fraction=median_atr_fraction,
        )
        result = _path_axes(labels, tbm)
        target = args.out / "parts" / f"month={month}"
        target.mkdir(parents=True)
        result.to_parquet(target / "semantics.parquet", index=False, compression="zstd")
        coverage.append({
            "month": month,
            "rows": int(len(result)),
            "valid_rows": int(result["semantic_path_valid"].sum()),
            "valid_fraction": float(result["semantic_path_valid"].mean()),
            "unique_composites": int(result.loc[result["semantic_path_valid"], "semantic_composite"].nunique()),
        })
        if coverage[-1]["valid_fraction"] < 0.90:
            raise AssertionError(
                f"{month}: semantic path coverage below 90%; inspect source availability before fitting"
            )
    _write_manifest(args.out, {
        "schema": SCHEMA,
        "scope": "training-only outcome semantics; prohibited from target-free/inference score panels",
        "path_root": str(args.path_root),
        "path_root_sha256": _sha256(args.path_root),
        "policy_json": str(args.policy_json),
        "policy_json_sha256": _sha256(args.policy_json),
        "canonical_policy_labels": str(args.canonical_policy_labels),
        "canonical_policy_labels_sha256": _sha256(args.canonical_policy_labels),
        "bars_root": str(args.bars_root),
        "tbm": {
            "lower": "frozen RichPolicy initial stop",
            "upper": "max(frozen initial trailing activation, 100-bps cost floor)",
            "same_bar": "ambiguous",
            "horizon_bars": HORIZON_BARS,
        },
        "horizon_hours": HORIZON_HOURS,
        "cost_floor_bps": COST_FLOOR_BPS,
        "months": requested,
        "coverage": coverage,
    })


if __name__ == "__main__":
    main()
