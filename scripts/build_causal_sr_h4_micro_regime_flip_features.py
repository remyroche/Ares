#!/usr/bin/env python3
"""Materialise a causal 15-minute micro-regime-flip H4 feature block.

The block describes whether the recent post-fill minute path has turned
against the longer one-hour path: adverse acceleration, persistent adverse
closes, local structural failure, failed favourable reclaim, and a fixed
target-free composite.  It is an offline research panel only.
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
    "micro_trend_return_15m_atr", "micro_trend_return_30m_atr", "micro_trend_return_60m_atr",
    "micro_trend_flip_vs_1h", "micro_adverse_acceleration_15m", "micro_adverse_close_streak_1m",
    "micro_adverse_efficiency_15m", "micro_structure_break_1h", "micro_failed_favorable_reclaim_15m",
    "micro_range_position_aligned_1h", "micro_regime_flip_score_15m",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _returns(closes: np.ndarray, entry: float, atr: float, direction: float, end: int, width: int) -> np.ndarray:
    start = max(0, end - width + 1)
    prior = entry if start == 0 else float(closes[start - 1])
    values = np.asarray(closes[start : end + 1], dtype=float)
    return direction * np.diff(np.concatenate(([prior], values))) / atr


def _features(*, entry: float, atr: float, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, state_minute: int, side: str) -> dict[str, float]:
    end = min(max(int(state_minute), 0), len(closes) - 1)
    if end < 1 or not np.isfinite([entry, atr]).all() or entry <= 0 or atr <= 0:
        return {name: np.nan for name in FEATURE_NAMES}
    direction = 1.0 if str(side).lower() == "long" else -1.0
    r15, r30, r60 = (_returns(closes, entry, atr, direction, end, width) for width in (15, 30, 60))
    net15, net30, net60 = (float(value.sum()) for value in (r15, r30, r60))
    adverse = r15[r15 < 0]
    travelled15 = float(np.abs(r15).sum())
    adverse_efficiency = float(max(-net15, 0.0) / travelled15) if travelled15 > 1e-12 else np.nan
    streak = 0
    for value in r15[::-1]:
        if value < 0:
            streak += 1
        else:
            break
    one_hour_start = max(0, end - 59)
    prior_end = max(one_hour_start, end - 15)
    prior_close = np.asarray(closes[one_hour_start : prior_end + 1], dtype=float)
    current_close = float(closes[end])
    aligned_prior = direction * (prior_close - entry)
    # Current close breaking the prior one-hour adverse boundary is known at
    # the completed state, as are all values defining the boundary.
    structure_break = float(direction * (current_close - entry) < float(np.min(aligned_prior))) if len(prior_close) else np.nan
    last_15_start = max(0, end - 14)
    recent_high = float(np.max(direction * (np.asarray(highs[last_15_start : end + 1]) - entry)))
    prior_high = float(np.max(aligned_prior)) if len(prior_close) else np.nan
    failed_reclaim = float(recent_high > prior_high and direction * (current_close - entry) < prior_high) if np.isfinite(prior_high) else np.nan
    range_high = float(np.max(direction * (np.asarray(highs[one_hour_start : end + 1]) - entry)))
    range_low = float(np.min(direction * (np.asarray(lows[one_hour_start : end + 1]) - entry)))
    range_position = (direction * (current_close - entry) - range_low) / (range_high - range_low) if range_high - range_low > 1e-12 else np.nan
    prior_15 = float(r30[:-15].sum()) if len(r30) > 15 else np.nan
    adverse_acceleration = max(-net15, 0.0) - max(-prior_15, 0.0) if np.isfinite(prior_15) else np.nan
    # Fixed, causal composition in ATR units—not learned from outcomes.  It
    # intentionally weights adverse impulse, speed, inefficient retreat and
    # structural failure, leaving any relevance judgement to the later OOF
    # model screen.
    flip_score = (
        max(-net15, 0.0)
        + .50 * max(-net30 / 2.0, 0.0)
        + .50 * float(streak / 15.0)
        + .50 * (adverse_efficiency if np.isfinite(adverse_efficiency) else 0.0)
        + .75 * (structure_break if np.isfinite(structure_break) else 0.0)
        + .50 * (failed_reclaim if np.isfinite(failed_reclaim) else 0.0)
    )
    return {
        "micro_trend_return_15m_atr": net15,
        "micro_trend_return_30m_atr": net30,
        "micro_trend_return_60m_atr": net60,
        "micro_trend_flip_vs_1h": float(np.sign(net15) != np.sign(net60) and abs(net15) > 1e-12 and abs(net60) > 1e-12),
        "micro_adverse_acceleration_15m": adverse_acceleration,
        "micro_adverse_close_streak_1m": float(streak),
        "micro_adverse_efficiency_15m": adverse_efficiency,
        "micro_structure_break_1h": structure_break,
        "micro_failed_favorable_reclaim_15m": failed_reclaim,
        "micro_range_position_aligned_1h": range_position,
        "micro_regime_flip_score_15m": float(flip_score),
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
            print(f"micro-regime states {ordinal}/{len(states)}", flush=True)
    panel = pd.DataFrame(records)
    if len(panel) != len(states) or panel.duplicated(list(KEY)).any():
        raise AssertionError("micro-regime feature panel lost state identity")
    out.mkdir(parents=True, exist_ok=False)
    panel.to_parquet(out / "h4_micro_regime_flip_target_free.parquet", index=False, compression="zstd")
    pd.DataFrame({"position": range(len(FEATURE_NAMES)), "feature": FEATURE_NAMES}).to_parquet(out / "feature_contract.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "causal-sr-h4-micro-regime-flip-v1",
        "scope": "offline target-free research feature block only; no labels, live policy, admission, portfolio, MC1, C1 S/R, Geometry/K9, or exchange mutation",
        "causality": "each state uses only post-fill exact 1m high/low/close through state_minute, with immutable entry-time ATR normalization",
        "source": {"parent_root": str(parent_root), "parent_manifest_sha256": _sha256(parent_root / "run_manifest.json"), "state_root": str(state_root), "state_manifest_sha256": _sha256(state_root / "run_manifest.json")},
        "features": list(FEATURE_NAMES), "row_count": int(len(panel)), "no_exchange_calls": True,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
