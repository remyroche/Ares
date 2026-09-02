#!/usr/bin/env python3
"""Materialize exact-policy pre-exit capture and give-back labels.

The source outcome is the canonical deployed-policy one-minute replay.  This
script re-reads the same immutable one-minute paths only to describe what
happened before the policy exit.  It does not simulate a different exit and it
does not create decision-time features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_execution_ev_policy_labels import (  # noqa: E402
    _load_symbol_bars,
)

IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
HORIZON_MINUTES = 720
SCHEMA = "exact_policy_pre_exit_capture_labels_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_crossing(values: np.ndarray, threshold: float) -> int:
    positions = np.flatnonzero(values >= threshold)
    return int(positions[0]) if len(positions) else -1


def _giveback_after_fraction(
    favorable: np.ndarray,
    signed_close: np.ndarray,
    peak: float,
    fraction: float,
) -> tuple[float, float]:
    if not np.isfinite(peak) or peak <= 1e-12:
        return 0.0, 0.0
    reached = _first_crossing(favorable, fraction * peak)
    if reached < 0:
        return 0.0, 0.0
    reference_peak = float(np.max(favorable[: reached + 1]))
    trough = float(np.min(signed_close[reached:]))
    giveback = max(reference_peak - trough, 0.0)
    return giveback, giveback / peak


def compute_capture_labels(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    *,
    entry: np.ndarray,
    side: np.ndarray,
    exit_bar: np.ndarray,
    gross: np.ndarray,
    cost: np.ndarray,
    atr_fraction: np.ndarray,
) -> pd.DataFrame:
    """Compute labels through the exact deployed-policy exit, inclusive."""
    arrays = (highs, lows, closes)
    if any(array.ndim != 2 for array in arrays):
        raise ValueError("path arrays must be two-dimensional")
    if not (highs.shape == lows.shape == closes.shape):
        raise ValueError("path arrays must share a shape")
    rows, horizon = highs.shape
    vectors = (entry, side, exit_bar, gross, cost, atr_fraction)
    if any(np.asarray(vector).shape != (rows,) for vector in vectors):
        raise ValueError("label vectors must match path rows")
    if horizon <= 0:
        raise ValueError("path horizon must be positive")
    if not all(np.isfinite(np.asarray(value, dtype=float)).all() for value in (*arrays, *vectors)):
        raise ValueError("capture-label inputs must be finite")
    if (entry <= 0).any() or (atr_fraction <= 0).any():
        raise ValueError("entry and ATR fraction must be strictly positive")
    if not np.isin(side, (-1.0, 1.0)).all():
        raise ValueError("side must be canonical -1/+1")

    records: list[dict[str, Any]] = []
    for row in range(rows):
        end = int(np.clip(np.rint(exit_bar[row]), 0, horizon - 1))
        high = highs[row, : end + 1] / entry[row] - 1.0
        low = lows[row, : end + 1] / entry[row] - 1.0
        close = closes[row, : end + 1] / entry[row] - 1.0
        if side[row] > 0:
            favorable = high
            adverse = -low
            signed_close = close
        else:
            favorable = -low
            adverse = high
            signed_close = -close
        mfe = float(max(np.max(favorable), 0.0))
        mae = float(max(np.max(adverse), 0.0))
        peak_minute = int(np.argmax(favorable))
        post_peak = signed_close[min(peak_minute + 1, end) : end + 1]
        post_peak_trough = float(np.min(post_peak)) if len(post_peak) else float(signed_close[end])
        post_peak_giveback = max(mfe - post_peak_trough, 0.0)

        threshold_cost = float(cost[row])
        threshold_half_atr = 0.5 * float(atr_fraction[row])
        cost_fav = _first_crossing(favorable, threshold_cost)
        cost_adv = _first_crossing(adverse, threshold_cost)
        atr_fav = _first_crossing(favorable, threshold_half_atr)
        atr_adv = _first_crossing(adverse, threshold_half_atr)

        def order(first_favorable: int, first_adverse: int) -> tuple[bool, bool, bool]:
            favorable_first = first_favorable >= 0 and (
                first_adverse < 0 or first_favorable < first_adverse
            )
            adverse_first = first_adverse >= 0 and (
                first_favorable < 0 or first_adverse < first_favorable
            )
            same_minute = first_favorable >= 0 and first_favorable == first_adverse
            return favorable_first, adverse_first, same_minute

        cost_order = order(cost_fav, cost_adv)
        atr_order = order(atr_fav, atr_adv)
        giveback_50, giveback_50_ratio = _giveback_after_fraction(
            favorable, signed_close, mfe, 0.5
        )
        giveback_80, giveback_80_ratio = _giveback_after_fraction(
            favorable, signed_close, mfe, 0.8
        )
        capture_gap = mfe - float(gross[row])
        capture_ratio = max(float(gross[row]), 0.0) / max(mfe, 1e-4)
        net = float(gross[row] - cost[row])
        records.append(
            {
                "policy_exit_bar_1m": end,
                "pre_exit_mfe_return": mfe,
                "pre_exit_mae_return": mae,
                "pre_exit_peak_minute": peak_minute,
                "pre_exit_close_return": float(signed_close[end]),
                "pre_exit_mfe_to_gross_gap": capture_gap,
                "pre_exit_gross_capture_ratio": float(np.clip(capture_ratio, 0.0, 1.0)),
                "post_peak_close_giveback_return": post_peak_giveback,
                "post_peak_close_giveback_ratio": post_peak_giveback / max(mfe, 1e-4),
                "giveback_after_50pct_mfe_return": giveback_50,
                "giveback_after_50pct_mfe_ratio": giveback_50_ratio,
                "giveback_after_80pct_mfe_return": giveback_80,
                "giveback_after_80pct_mfe_ratio": giveback_80_ratio,
                "first_favorable_cost_minute": cost_fav,
                "first_adverse_cost_minute": cost_adv,
                "favorable_before_adverse_at_cost": cost_order[0],
                "adverse_before_favorable_at_cost": cost_order[1],
                "cost_barriers_same_minute": cost_order[2],
                "first_favorable_half_atr_minute": atr_fav,
                "first_adverse_half_atr_minute": atr_adv,
                "favorable_before_adverse_at_half_atr": atr_order[0],
                "adverse_before_favorable_at_half_atr": atr_order[1],
                "half_atr_barriers_same_minute": atr_order[2],
                "exact_gross_positive": bool(gross[row] > 0.0),
                "exact_net_positive": bool(net > 0.0),
                "exact_net_loss_worse_one_cost": bool(net <= -cost[row]),
                "exact_net_loss_worse_two_costs": bool(net <= -2.0 * cost[row]),
            }
        )
    return pd.DataFrame.from_records(records)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise ValueError("refusing to overwrite output directory")
    canonical = pd.read_parquet(args.canonical_input)
    policy = pd.read_parquet(
        args.policy_labels,
        columns=[
            *IDENTITY,
            "execution_decision_utc",
            "execution_exit_hour",
            "execution_entry_price",
            "execution_gross_ev_12h",
            "execution_cost_return",
            "execution_label_end_utc",
        ],
    )
    for name, frame in (("canonical", canonical), ("policy", policy)):
        if frame.duplicated(list(IDENTITY)).any():
            raise ValueError(f"{name} input contains duplicate identities")
    keep = [
        *IDENTITY,
        "execution_decision_utc",
        "execution_exit_hour",
        "execution_entry_price",
        "execution_gross_ev_12h",
        "execution_cost_return",
        "execution_label_end_utc",
    ]
    frame = canonical.loc[:, [*IDENTITY, "oof_entry_atr_fraction"]].merge(
        policy.loc[:, keep],
        on=list(IDENTITY),
        how="left",
        validate="one_to_one",
    )
    if frame[keep[len(IDENTITY) :]].isna().any().any():
        raise ValueError("canonical-to-policy join is incomplete")
    frame["execution_decision_utc"] = pd.to_datetime(
        frame["execution_decision_utc"], utc=True, errors="raise"
    )
    frame["execution_label_end_utc"] = pd.to_datetime(
        frame["execution_label_end_utc"], utc=True, errors="raise"
    )
    frame["__row__"] = np.arange(len(frame), dtype=np.int64)
    parts: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []
    for symbol, local in frame.groupby("__symbol__", sort=True):
        local = local.sort_values("execution_decision_utc", kind="stable")
        start = local["execution_decision_utc"].min()
        end = local["execution_decision_utc"].max() + pd.Timedelta(
            minutes=HORIZON_MINUTES
        )
        bars = _load_symbol_bars(args.data_root, str(symbol), start, end)
        grid = pd.date_range(start, end, freq="min", inclusive="left", tz="UTC")
        dense = bars.reindex(grid).loc[:, ["high", "low", "close"]]
        values = dense.to_numpy(dtype=np.float64)
        offsets = (
            (local["execution_decision_utc"] - start) / pd.Timedelta(minutes=1)
        ).astype(np.int64).to_numpy()
        complete = np.zeros(len(local), dtype=bool)
        local_parts: list[pd.DataFrame] = []
        for batch_start in range(0, len(local), int(args.batch_rows)):
            batch_end = min(batch_start + int(args.batch_rows), len(local))
            batch = local.iloc[batch_start:batch_end]
            positions = offsets[batch_start:batch_end, None] + np.arange(
                HORIZON_MINUTES, dtype=np.int64
            )[None, :]
            matrices = tuple(values[positions, column] for column in range(3))
            valid = np.isfinite(np.stack(matrices, axis=2)).all(axis=(1, 2))
            complete[batch_start:batch_end] = valid
            if not valid.all():
                continue
            exit_bar = np.rint(
                batch["execution_exit_hour"].to_numpy(dtype=float) * 60.0
            )
            labels = compute_capture_labels(
                *matrices,
                entry=batch["execution_entry_price"].to_numpy(dtype=float),
                side=np.where(batch["side_name"].eq("long"), 1.0, -1.0),
                exit_bar=exit_bar,
                gross=batch["execution_gross_ev_12h"].to_numpy(dtype=float),
                cost=batch["execution_cost_return"].to_numpy(dtype=float),
                atr_fraction=batch["oof_entry_atr_fraction"].to_numpy(dtype=float),
            )
            labels["__row__"] = batch["__row__"].to_numpy()
            local_parts.append(labels)
        coverage_rows.append(
            {
                "__symbol__": str(symbol),
                "rows": int(len(local)),
                "complete_rows": int(complete.sum()),
                "coverage": float(complete.mean()),
            }
        )
        if local_parts:
            parts.append(pd.concat(local_parts, ignore_index=True))
    if not parts:
        raise ValueError("no exact capture labels were materialized")
    labels = pd.concat(parts, ignore_index=True)
    result = frame.merge(labels, on="__row__", how="left", validate="one_to_one")
    label_columns = [column for column in labels if column != "__row__"]
    missing = result[label_columns].isna().any(axis=1)
    if missing.any():
        raise ValueError(
            f"exact one-minute capture path coverage is incomplete: {int(missing.sum())}"
        )
    result["label_resolution_utc"] = result["execution_label_end_utc"]
    result = result.drop(columns="__row__").sort_values(
        list(IDENTITY), kind="stable"
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "exact_policy_capture_labels.parquet"
    coverage_output = args.output_dir / "coverage.csv"
    manifest_output = args.output_dir / "manifest.json"
    result.to_parquet(output, index=False, compression="zstd")
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(coverage_output, index=False)
    manifest = {
        "schema": SCHEMA,
        "status": "completed_exact_policy_supporting_labels_not_decision_features",
        "rows": int(len(result)),
        "identity_unique": bool(not result.duplicated(list(IDENTITY)).any()),
        "coverage": {
            "complete_rows": int(coverage["complete_rows"].sum()),
            "rows": int(coverage["rows"].sum()),
            "rate": float(coverage["complete_rows"].sum() / coverage["rows"].sum()),
        },
        "lineage": {
            "canonical_input": str(args.canonical_input),
            "canonical_input_sha256": _sha256(args.canonical_input),
            "policy_labels": str(args.policy_labels),
            "policy_labels_sha256": _sha256(args.policy_labels),
            "immutable_execution_1m_root": str(args.data_root),
        },
        "contract": {
            "path_window": "decision through deployed-policy exit, inclusive",
            "exit_policy_changed": False,
            "cadence": "exact 1m",
            "label_availability": "conservative full 12h replay horizon",
            "spread": "embedded in exact gross and executable entry price",
            "fee": "exact row execution_cost_return",
            "ohlc_same_minute_order": "barrier ties are explicit and not assigned to either order",
            "whole_horizon_mfe_used": False,
        },
        "outputs": {
            "labels": {"path": str(output), "sha256": _sha256(output)},
            "coverage": {
                "path": str(coverage_output),
                "sha256": _sha256(coverage_output),
            },
        },
    }
    manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize deployed-policy pre-exit capture labels."
    )
    parser.add_argument(
        "--canonical-input",
        type=Path,
        default=Path(
            "data_perp/artifacts/execution_ev_canonical_exact_policy_regime_input_20260727_v3/joined.parquet"
        ),
    )
    parser.add_argument(
        "--policy-labels",
        type=Path,
        default=Path(
            "data_perp/artifacts/execution_ev_policy_labels_12h_july20_20260726_v1/execution_ev_policy_labels.parquet"
        ),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data_perp"))
    parser.add_argument("--batch-rows", type=int, default=2048)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(_parser().parse_args()), indent=2))
