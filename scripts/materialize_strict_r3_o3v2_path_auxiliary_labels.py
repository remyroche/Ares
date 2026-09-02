#!/usr/bin/env python3
"""Materialise extra resolved H12 path targets for O3-v2 meta research.

This producer deliberately writes *labels only*.  It extends the existing
strict-R3 long supportive path panel with the few quantities that cannot be
recovered from its scalar summaries: bps-threshold reach/timing, MAE before a
threshold is first reached, and timing under the frozen rich-policy geometry.
It must never be joined to target-free scoring or inference inputs.

The source path is the next 48 observed 15-minute bars from the frozen
decision open.  Missing opens, ATR warm-up, or any incomplete H12 path leave
all supervised values null; they are not encoded as adverse outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for candidate in (ROOT, SCRIPTS):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import materialize_strict_r3_o3v2_semantics as semantics  # noqa: E402
from extreme_price_movements.strict_r3_rich_policy import (  # noqa: E402
    RichPolicyParams,
    _activation_distance,
    _barrier_distances,
    _stop_distance,
)


SCHEMA = "strict_r3_o3v2_path_auxiliary_labels_v2"
SIDE = "long"
HORIZON_HOURS = 12.0
HORIZON_BARS = 48
BAR_HOURS = 0.25
MFE_HOURS = (1, 3, 6, 9, 12)
MFE_BPS = (50, 100, 200, 300, 500)
MAE_BEFORE_BPS = (100, 200, 250)
# This is intentionally a label-only contract.  The wider values are required
# by the recall-router auxiliary-label study, not by live inference.
# Eight and ten ATR are required solely for the predeclared TBM-B/TBM-C
# recall-router targets.  They remain outcome-only sidecar labels and are
# never copied to score-time or inference panels.
MFE_ATR = (1, 2, 3, 4, 6, 8, 10)
ADVERSE_ATR = (3, 4)
IDENTITY = ("candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name")
SOURCE_COLUMNS = (
    *IDENTITY,
    "supportive_path_valid",
    "supportive_label_available_ts",
    "entry_price",
    "path_arch_atr_fraction",
    "path_arch_mfe_before_mae",
    "path_arch_mae_before_mfe",
    "path_arch_efficiency",
    "path_arch_time_to_first_meaningful_mfe_h",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files: Iterable[Path] = sorted(path.rglob("*.parquet")) if path.is_dir() else (path,)
    for item in files:
        digest.update(str(item.relative_to(path) if path.is_dir() else item.name).encode())
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _months(path_root: Path) -> tuple[str, ...]:
    return tuple(sorted(part.name.split("=", 1)[1] for part in path_root.glob("month=*") if part.is_dir()))


def _first_hit(mask: np.ndarray) -> np.ndarray:
    """Return a one-based bar index; zero denotes no event during H12."""
    return semantics._first_true(mask)


def _time_censored(bar: np.ndarray) -> np.ndarray:
    return np.where(bar > 0, bar.astype(float) * BAR_HOURS, HORIZON_HOURS).astype(np.float32)


def _target_columns() -> tuple[str, ...]:
    columns: list[str] = [
        "aux_label_available_ts",
        "aux_path_valid",
        "aux_path_complete",
        "aux_path_mfe_before_mae",
        "aux_path_mae_before_mfe",
        "aux_path_efficiency",
        "aux_time_to_first_meaningful_mfe_h",
    ]
    for hour in MFE_HOURS:
        columns.extend((f"aux_mfe_bps_{hour}h", f"aux_mae_bps_{hour}h", f"aux_mfe_atr_{hour}h", f"aux_mae_atr_{hour}h"))
    for bps in MFE_BPS:
        columns.extend((f"aux_reached_{bps}bps", f"aux_time_to_{bps}bps_h"))
    for bps in MAE_BEFORE_BPS:
        columns.append(f"aux_mae_before_{bps}bps_atr")
    for atr in MFE_ATR:
        columns.extend((f"aux_reached_{atr}atr", f"aux_time_to_{atr}atr_h"))
    for atr in ADVERSE_ATR:
        columns.extend((f"aux_reached_adverse_{atr}atr", f"aux_time_to_adverse_{atr}atr_h"))
    columns.extend((
        "aux_peak_mfe_atr_h12",
        "aux_peak_mfe_bps_h12",
        "aux_path_length_bps_h12",
        "aux_mfe_over_path_length_h12",
        "aux_mfe_over_abs_mae_h12",
        "aux_reached_trailing_activation",
        "aux_time_to_trailing_activation_h",
        "aux_reached_stop_loss",
        "aux_time_to_stop_loss_h",
        "aux_first_policy_hit_interval_h",
    ))
    return tuple(columns)


TARGET_COLUMNS = _target_columns()


def _empty_output(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.loc[:, list(IDENTITY)].copy()
    # Keep a timezone-aware dtype from the start.  Assigning UTC values into
    # a naive all-NaT column silently coerces it to object under pandas 2.3,
    # which weakens the availability lineage and emits a future warning.
    output["aux_label_available_ts"] = pd.Series(pd.NaT, index=output.index, dtype="datetime64[ns, UTC]")
    output["aux_path_valid"] = False
    output["aux_path_complete"] = False
    for column in TARGET_COLUMNS:
        if column not in output:
            output[column] = np.nan
    return output


def _read_month(path_root: Path, token: str) -> pd.DataFrame:
    source = path_root / f"month={token}" / "side=long.parquet"
    if not source.exists():
        raise FileNotFoundError(source)
    frame = pd.read_parquet(source, columns=list(SOURCE_COLUMNS)).copy()
    for column in ("__ts__", "__decision_ts__", "supportive_label_available_ts"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    if frame["candidate_id"].duplicated().any():
        raise AssertionError(f"{token}: duplicate auxiliary-label identities")
    if not frame["side_name"].astype(str).str.lower().eq(SIDE).all():
        raise AssertionError(f"{token}: source must be long-only")
    return frame.reset_index(drop=True)


def _write_json_exclusive(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _assign_symbol(
    output: pd.DataFrame,
    group: pd.DataFrame,
    *,
    bars_root: Path,
    params: RichPolicyParams,
    median_atr_fraction: float,
) -> None:
    """Assign all target-only values for one symbol without cross-symbol state."""
    positions = group.index.to_numpy(np.int64)
    entry = pd.to_numeric(group["entry_price"], errors="coerce").to_numpy(float)
    atr_fraction = pd.to_numeric(group["path_arch_atr_fraction"], errors="coerce").to_numpy(float)
    required = (
        group["supportive_path_valid"].fillna(False).astype(bool).to_numpy()
        & np.isfinite(entry) & (entry > 0.0)
        & np.isfinite(atr_fraction) & (atr_fraction > 0.0)
    )
    if not required.any():
        return
    bars = semantics._load_bars(bars_root, str(group["__symbol__"].iloc[0]))
    complete, high, low, close = semantics._windows(bars, group["__decision_ts__"])
    valid = required & complete
    if not valid.any():
        return
    local = np.flatnonzero(valid)
    target = positions[local]
    local_entry = entry[local]
    local_atr_fraction = atr_fraction[local]
    atr = local_entry * local_atr_fraction
    local_high = high[local].astype(float)
    local_low = low[local].astype(float)
    favourable_bps = np.maximum((local_high / local_entry[:, None] - 1.0) * 10_000.0, 0.0)
    adverse_bps = np.maximum((1.0 - local_low / local_entry[:, None]) * 10_000.0, 0.0)
    favourable_atr = favourable_bps / np.maximum(local_atr_fraction[:, None] * 10_000.0, 1e-12)
    adverse_atr = adverse_bps / np.maximum(local_atr_fraction[:, None] * 10_000.0, 1e-12)
    peak_mfe_bps = favourable_bps.max(axis=1)
    peak_mfe_atr = favourable_atr.max(axis=1)
    peak_mae_atr = adverse_atr.max(axis=1)
    # Use the executable 15-minute close path to measure travelled distance.
    # The entry is prepended so the first post-entry move is included.  A
    # small ATR-relative denominator floor avoids an unbounded ratio for a
    # perfectly monotone, near-zero-adverse path while retaining its ordering.
    close_bps = (close[local].astype(float) / local_entry[:, None] - 1.0) * 10_000.0
    close_path = np.column_stack((np.zeros(len(local), dtype=float), close_bps))
    path_length_bps = np.abs(np.diff(close_path, axis=1)).sum(axis=1)
    mae_floor_atr = np.maximum(peak_mae_atr, 0.10)
    # MAE prior to the first reach deliberately excludes the ambiguous same
    # 15-minute bar.  It is a conservative, reproducible convention; no
    # intrabar ordering is claimed where the historical source cannot prove it.
    prior_adverse = np.concatenate((np.zeros((len(local), 1)), np.maximum.accumulate(adverse_atr, axis=1)[:, :-1]), axis=1)
    output.loc[target, "aux_path_complete"] = True
    output.loc[target, "aux_path_valid"] = True
    output.loc[target, "aux_label_available_ts"] = group.iloc[local]["__decision_ts__"].to_numpy() + pd.Timedelta(hours=HORIZON_HOURS)
    for source, destination in (
        ("path_arch_mfe_before_mae", "aux_path_mfe_before_mae"),
        ("path_arch_mae_before_mfe", "aux_path_mae_before_mfe"),
        ("path_arch_efficiency", "aux_path_efficiency"),
        ("path_arch_time_to_first_meaningful_mfe_h", "aux_time_to_first_meaningful_mfe_h"),
    ):
        output.loc[target, destination] = pd.to_numeric(group.iloc[local][source], errors="coerce").to_numpy(float)
    for hour in MFE_HOURS:
        bar_count = int(hour / BAR_HOURS)
        output.loc[target, f"aux_mfe_bps_{hour}h"] = favourable_bps[:, :bar_count].max(axis=1)
        output.loc[target, f"aux_mae_bps_{hour}h"] = adverse_bps[:, :bar_count].max(axis=1)
        output.loc[target, f"aux_mfe_atr_{hour}h"] = favourable_atr[:, :bar_count].max(axis=1)
        output.loc[target, f"aux_mae_atr_{hour}h"] = adverse_atr[:, :bar_count].max(axis=1)
    for bps in MFE_BPS:
        hit = _first_hit(favourable_bps >= float(bps))
        output.loc[target, f"aux_reached_{bps}bps"] = (hit > 0).astype(np.float32)
        output.loc[target, f"aux_time_to_{bps}bps_h"] = _time_censored(hit)
    for bps in MAE_BEFORE_BPS:
        hit = _first_hit(favourable_bps >= float(bps))
        values = np.full(len(local), np.nan, dtype=np.float32)
        reached = hit > 0
        positions_before = np.maximum(hit[reached] - 1, 0)
        values[reached] = prior_adverse[np.flatnonzero(reached), positions_before].astype(np.float32)
        output.loc[target, f"aux_mae_before_{bps}bps_atr"] = values
    for multiple in MFE_ATR:
        hit = _first_hit(favourable_atr >= float(multiple))
        output.loc[target, f"aux_reached_{multiple}atr"] = (hit > 0).astype(np.float32)
        output.loc[target, f"aux_time_to_{multiple}atr_h"] = _time_censored(hit)
    for multiple in ADVERSE_ATR:
        hit = _first_hit(adverse_atr >= float(multiple))
        output.loc[target, f"aux_reached_adverse_{multiple}atr"] = (hit > 0).astype(np.float32)
        output.loc[target, f"aux_time_to_adverse_{multiple}atr_h"] = _time_censored(hit)
    output.loc[target, "aux_peak_mfe_atr_h12"] = peak_mfe_atr.astype(np.float32)
    output.loc[target, "aux_peak_mfe_bps_h12"] = peak_mfe_bps.astype(np.float32)
    output.loc[target, "aux_path_length_bps_h12"] = path_length_bps.astype(np.float32)
    output.loc[target, "aux_mfe_over_path_length_h12"] = np.divide(
        peak_mfe_bps,
        np.maximum(path_length_bps, 1.0),
    ).astype(np.float32)
    output.loc[target, "aux_mfe_over_abs_mae_h12"] = np.divide(
        peak_mfe_atr,
        mae_floor_atr,
    ).astype(np.float32)
    sl_raw, tp_raw = _barrier_distances(local_entry, atr, params, median_atr_fraction=median_atr_fraction)
    stop = _stop_distance(sl_raw, local_entry, params)
    activation = _activation_distance(tp_raw, local_entry, params, bar=0)
    activation = np.maximum(activation, local_entry * semantics.COST_BPS / 10_000.0)
    trailing_hit = _first_hit(local_high >= local_entry[:, None] + activation[:, None])
    stop_hit = _first_hit(local_low <= local_entry[:, None] - stop[:, None])
    output.loc[target, "aux_reached_trailing_activation"] = (trailing_hit > 0).astype(np.float32)
    output.loc[target, "aux_time_to_trailing_activation_h"] = _time_censored(trailing_hit)
    output.loc[target, "aux_reached_stop_loss"] = (stop_hit > 0).astype(np.float32)
    output.loc[target, "aux_time_to_stop_loss_h"] = _time_censored(stop_hit)
    both = np.column_stack((_time_censored(trailing_hit), _time_censored(stop_hit)))
    output.loc[target, "aux_first_policy_hit_interval_h"] = np.minimum(both[:, 0], both[:, 1])


def _materialize_month(
    frame: pd.DataFrame,
    *,
    bars_root: Path,
    params: RichPolicyParams,
    median_atr_fraction: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    output = _empty_output(frame)
    for _symbol, group in frame.groupby("__symbol__", sort=True, observed=True):
        _assign_symbol(output, group, bars_root=bars_root, params=params, median_atr_fraction=median_atr_fraction)
    valid = output["aux_path_valid"].fillna(False).astype(bool)
    output.loc[~valid, [column for column in TARGET_COLUMNS if column not in {"aux_path_valid", "aux_path_complete", "aux_label_available_ts"}]] = np.nan
    output.loc[~valid, "aux_label_available_ts"] = pd.NaT
    if output.loc[valid, "aux_label_available_ts"].lt(output.loc[valid, "__decision_ts__"] + pd.Timedelta(hours=HORIZON_HOURS)).any():
        raise AssertionError("auxiliary labels available before their complete H12 path")
    target_values = [column for column in TARGET_COLUMNS if column not in {"aux_path_valid", "aux_path_complete", "aux_label_available_ts"}]
    if output.loc[~valid, target_values].notna().any().any():
        raise AssertionError("invalid auxiliary paths were encoded as ordinary labels")
    return output.loc[:, [*IDENTITY, *TARGET_COLUMNS]], {
        "rows": int(len(output)),
        "source_path_valid_rows": int(frame["supportive_path_valid"].fillna(False).astype(bool).sum()),
        "aux_path_valid_rows": int(valid.sum()),
        "aux_path_valid_fraction": float(valid.mean()),
        "complete_rows": int(output["aux_path_complete"].fillna(False).astype(bool).sum()),
    }


def run(*, path_root: Path, bars_root: Path, policy_json: Path, out: Path, months: Sequence[str]) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    payload = json.loads(policy_json.read_text())
    params = RichPolicyParams.from_mapping(payload["params"])
    median_atr_fraction = float(payload.get("median_atr_fraction", payload["median_atr_fraction_fitted_on_complete_2024_development"]))
    out.mkdir(parents=True, exist_ok=False)
    coverage: list[dict[str, object]] = []
    for token in months:
        frame = _read_month(path_root, token)
        output, record = _materialize_month(frame, bars_root=bars_root, params=params, median_atr_fraction=median_atr_fraction)
        record["month"] = token
        destination = out / "parts" / f"month={token}"
        destination.mkdir(parents=True, exist_ok=False)
        output.to_parquet(destination / "auxiliary_path_labels.parquet", index=False, compression="zstd")
        coverage.append(record)
        print(json.dumps({"event": "materialized", **record}, sort_keys=True), flush=True)
    pd.DataFrame(coverage).to_parquet(out / "coverage_by_month.parquet", index=False, compression="zstd")
    _write_json_exclusive(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "resolved H12 path labels only; prohibited from target-free scoring and inference",
        "path_root": str(path_root.resolve()),
        "path_root_sha256": _sha256(path_root),
        "bars_root": str(bars_root.resolve()),
        "policy_json": str(policy_json.resolve()),
        "policy_json_sha256": _sha256(policy_json),
        "months": list(months),
        "identity": list(IDENTITY),
        "labels": list(TARGET_COLUMNS),
        "bar_contract": "next 48 observed 15-minute bars from frozen decision open",
        "availability": "decision + 12 hours",
        "intrabar_convention": "MAE-before-bps excludes the same 15-minute hit bar because ordering is not observed",
        "invalidity": "invalid/incomplete paths retain null target values and are excluded from supervised fitting",
        "coverage": coverage,
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-root", type=Path, required=True)
    parser.add_argument("--bars-root", type=Path, default=ROOT / "15m_ohlcv_perp")
    parser.add_argument("--policy-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--months", help="comma-separated YYYY-MM; defaults to all path-root partitions")
    args = parser.parse_args()
    months = tuple(args.months.split(",")) if args.months else _months(args.path_root)
    if not months:
        parser.error("no monthly path partitions found")
    print(run(path_root=args.path_root.resolve(), bars_root=args.bars_root.resolve(), policy_json=args.policy_json.resolve(), out=args.out.resolve(), months=months))


if __name__ == "__main__":
    main()
