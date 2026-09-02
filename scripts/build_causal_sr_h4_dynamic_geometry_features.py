#!/usr/bin/env python3
"""Materialise target-free dynamic local-boundary geometry for H4 research.

It represents causal price acceptance/rejection around *prior* local one-hour
boundaries, boundary-cross density, extension, balance density and range
compression.  These are local post-fill path features, intentionally separate
from the existing static S/R distance fields and from any future outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts import run_causal_sr_h4_actuator_counterfactual_ablation as base
except ModuleNotFoundError:
    import run_causal_sr_h4_actuator_counterfactual_ablation as base


KEY = ("candidate_id", "state_decision_ts")
FEATURE_NAMES = (
    "dynamic_geometry_prior_range_1h_atr", "dynamic_geometry_position_in_prior_range_1h",
    "dynamic_geometry_favorable_extension_atr_15m", "dynamic_geometry_adverse_extension_atr_15m",
    "dynamic_geometry_favorable_boundary_acceptance_15m", "dynamic_geometry_adverse_boundary_acceptance_15m",
    "dynamic_geometry_favorable_rejection_15m", "dynamic_geometry_adverse_rejection_15m",
    "dynamic_geometry_boundary_cross_density_15m", "dynamic_geometry_local_balance_density_15m",
    "dynamic_geometry_range_compression_15m", "dynamic_geometry_breakout_efficiency_15m",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _features(*, entry: float, atr: float, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, state_minute: int, side: str) -> dict[str, float]:
    end = min(max(int(state_minute), 0), len(closes) - 1)
    if end < 15 or not np.isfinite([entry, atr]).all() or entry <= 0.0 or atr <= 0.0:
        return {name: np.nan for name in FEATURE_NAMES}
    direction = 1.0 if str(side).lower() == "long" else -1.0
    current_start = max(0, end - 14)
    prior_start = max(0, current_start - 45)
    prior_end = current_start - 1
    prior_high = float(np.max(highs[prior_start : prior_end + 1]))
    prior_low = float(np.min(lows[prior_start : prior_end + 1]))
    prior_close = np.asarray(closes[prior_start : prior_end + 1], dtype=float)
    current_close = np.asarray(closes[current_start : end + 1], dtype=float)
    current_high = np.asarray(highs[current_start : end + 1], dtype=float)
    current_low = np.asarray(lows[current_start : end + 1], dtype=float)
    # Directional boundaries retain the exact same physical prices while
    # expressing features as favourable/adverse for either side.
    favorable_boundary, adverse_boundary = (prior_high, prior_low) if direction > 0 else (prior_low, prior_high)
    aligned_close = direction * (current_close - entry)
    aligned_high = direction * (current_high - entry)
    aligned_low = direction * (current_low - entry)
    favorable_extension = max(float(np.max(direction * (current_high - favorable_boundary))), 0.0)
    adverse_extension = max(float(np.max(-direction * (current_low - adverse_boundary))), 0.0)
    favorable_beyond = direction * (current_close - favorable_boundary) > 0.0
    adverse_beyond = direction * (current_close - adverse_boundary) < 0.0
    # A rejected breakout requires an intrabar breach then close back inside
    # the preceding local boundary.  It is entirely known at state time.
    favorable_rejection = float(favorable_extension > 0.0 and not bool(favorable_beyond[-1]))
    adverse_rejection = float(adverse_extension > 0.0 and not bool(adverse_beyond[-1]))
    midpoint = .5 * (prior_high + prior_low)
    relation = np.sign(current_close - midpoint)
    cross_count = float(np.count_nonzero(relation[1:] != relation[:-1])) if len(relation) > 1 else 0.0
    q25, q75 = np.quantile(prior_close, [.25, .75])
    balance_density = float(((current_close >= q25) & (current_close <= q75)).mean())
    prior_range = max(prior_high - prior_low, 1e-12)
    current_range = float(np.max(current_high) - np.min(current_low))
    displacement = float(abs(aligned_close[-1] - aligned_close[0]))
    travel = float(np.abs(np.diff(np.concatenate(([aligned_close[0]], aligned_close)))).sum())
    return {
        "dynamic_geometry_prior_range_1h_atr": prior_range / atr,
        "dynamic_geometry_position_in_prior_range_1h": float((current_close[-1] - prior_low) / prior_range),
        "dynamic_geometry_favorable_extension_atr_15m": favorable_extension / atr,
        "dynamic_geometry_adverse_extension_atr_15m": adverse_extension / atr,
        "dynamic_geometry_favorable_boundary_acceptance_15m": float(favorable_beyond.mean()),
        "dynamic_geometry_adverse_boundary_acceptance_15m": float(adverse_beyond.mean()),
        "dynamic_geometry_favorable_rejection_15m": favorable_rejection,
        "dynamic_geometry_adverse_rejection_15m": adverse_rejection,
        "dynamic_geometry_boundary_cross_density_15m": cross_count / max(len(current_close) - 1, 1),
        "dynamic_geometry_local_balance_density_15m": balance_density,
        "dynamic_geometry_range_compression_15m": current_range / prior_range,
        "dynamic_geometry_breakout_efficiency_15m": displacement / travel if travel > 1e-12 else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, default=base.DEFAULT_PARENT)
    parser.add_argument("--state-root", type=Path, default=base.DEFAULT_STATES)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    parent_root, state_root = args.parent_root.resolve(), args.state_root.resolve()
    _, rows, _, arrays, states = base._load_parent(parent_root, state_root)
    positions = pd.Series(np.arange(len(rows), dtype=np.int64), index=rows["candidate_id"].astype(str))
    sides = rows.set_index("candidate_id")["side_name"].astype(str)
    records: list[dict[str, object]] = []
    for ordinal, state in enumerate(states.loc[:, [*KEY, "state_minute"]].itertuples(index=False), 1):
        candidate = str(state.candidate_id)
        position = int(positions.loc[candidate])
        record: dict[str, object] = {"candidate_id": candidate, "state_decision_ts": pd.Timestamp(state.state_decision_ts)}
        record.update(_features(entry=float(arrays["entry"][position]), atr=float(arrays["atr"][position]), closes=arrays["close"][position], highs=arrays["high"][position], lows=arrays["low"][position], state_minute=int(state.state_minute), side=sides.loc[candidate]))
        records.append(record)
        if ordinal == 1 or ordinal % 100_000 == 0 or ordinal == len(states):
            print(f"dynamic geometry states {ordinal}/{len(states)}", flush=True)
    panel = pd.DataFrame(records)
    if len(panel) != len(states) or panel.duplicated(list(KEY)).any():
        raise AssertionError("dynamic geometry feature panel lost state identity")
    out.mkdir(parents=True, exist_ok=False)
    panel.to_parquet(out / "h4_dynamic_geometry_target_free.parquet", index=False, compression="zstd")
    pd.DataFrame({"position": range(len(FEATURE_NAMES)), "feature": FEATURE_NAMES}).to_parquet(out / "feature_contract.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "causal-sr-h4-dynamic-geometry-v1",
        "scope": "offline target-free research feature block only; no labels, live policy, admission, portfolio, MC1, C1 S/R, Geometry/K9, or exchange mutation",
        "causality": "each state uses only post-fill exact 1m high/low/close bars through state_minute; boundaries are formed from prior bars only and values are normalized by immutable entry-time ATR",
        "source": {"parent_root": str(parent_root), "parent_manifest_sha256": _sha256(parent_root / "run_manifest.json"), "state_root": str(state_root), "state_manifest_sha256": _sha256(state_root / "run_manifest.json")},
        "features": list(FEATURE_NAMES), "row_count": int(len(panel)), "no_exchange_calls": True,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
