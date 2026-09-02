#!/usr/bin/env python3
"""Materialise target-free directional-versus-chop volatility state for H4.

The existing H4 contract already contains broad RV and volume levels.  This
separate offline block measures the *realised post-fill minute-path shape* at
each completed state: directional expansion, chop, volatility acceleration,
and favourable/adverse semivolatility.  It consumes no future bars or labels.
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
    "path_realized_volatility_15m_atr", "path_realized_volatility_30m_atr", "path_realized_volatility_60m_atr",
    "path_volatility_acceleration_15m", "path_volatility_decay_15m",
    "path_directional_expansion_15m", "path_directional_expansion_30m",
    "path_chop_ratio_15m", "path_chop_ratio_30m",
    "path_favorable_semivol_15m", "path_adverse_semivol_15m", "path_adverse_volatility_share_15m",
    "path_favorable_range_share_15m", "path_adverse_range_share_15m",
    "path_volatility_of_volatility_15m",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _segment(closes: np.ndarray, entry: float, atr: float, direction: float, end: int, length: int) -> np.ndarray:
    start = max(0, end - int(length) + 1)
    prior = float(entry) if start == 0 else float(closes[start - 1])
    values = np.asarray(closes[start : end + 1], dtype=float)
    return direction * np.diff(np.concatenate(([prior], values))) / atr


def _rv(returns: np.ndarray) -> float:
    return float(np.sqrt(np.square(returns).sum())) if len(returns) else np.nan


def _features(*, entry: float, atr: float, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, state_minute: int, side: str) -> dict[str, float]:
    end = min(max(int(state_minute), 0), len(closes) - 1)
    if end < 1 or not np.isfinite([entry, atr]).all() or entry <= 0.0 or atr <= 0.0:
        return {name: np.nan for name in FEATURE_NAMES}
    direction = 1.0 if str(side).lower() == "long" else -1.0
    r15 = _segment(closes, entry, atr, direction, end, 15)
    r30 = _segment(closes, entry, atr, direction, end, 30)
    r60 = _segment(closes, entry, atr, direction, end, 60)
    rv15, rv30, rv60 = _rv(r15), _rv(r30), _rv(r60)
    def efficiency(values: np.ndarray) -> float:
        travelled = float(np.abs(values).sum())
        return float(abs(values.sum()) / travelled) if travelled > 1e-12 else np.nan
    def chop(values: np.ndarray) -> float:
        net = float(abs(values.sum()))
        return float(np.abs(values).sum() / net) if net > 1e-12 else np.nan
    adverse = r15[r15 < 0.0]
    favourable = r15[r15 > 0.0]
    adverse_sq = float(np.square(adverse).sum())
    total_sq = float(np.square(r15).sum())
    start = max(0, end - 14)
    aligned_high = direction * (np.asarray(highs[start : end + 1], dtype=float) - entry) / atr
    aligned_low = direction * (np.asarray(lows[start : end + 1], dtype=float) - entry) / atr
    favorable_range = float(np.maximum(aligned_high, 0.0).max()) if len(aligned_high) else np.nan
    adverse_range = float(max(-aligned_low.min(), 0.0)) if len(aligned_low) else np.nan
    minute_rv = np.sqrt(np.square(r15))
    return {
        "path_realized_volatility_15m_atr": rv15,
        "path_realized_volatility_30m_atr": rv30,
        "path_realized_volatility_60m_atr": rv60,
        "path_volatility_acceleration_15m": rv15 / rv30 if np.isfinite(rv30) and rv30 > 1e-12 else np.nan,
        "path_volatility_decay_15m": rv30 / rv60 if np.isfinite(rv60) and rv60 > 1e-12 else np.nan,
        "path_directional_expansion_15m": efficiency(r15),
        "path_directional_expansion_30m": efficiency(r30),
        "path_chop_ratio_15m": chop(r15),
        "path_chop_ratio_30m": chop(r30),
        "path_favorable_semivol_15m": float(np.sqrt(np.square(favourable).sum())),
        "path_adverse_semivol_15m": float(np.sqrt(adverse_sq)),
        "path_adverse_volatility_share_15m": adverse_sq / total_sq if total_sq > 1e-12 else np.nan,
        "path_favorable_range_share_15m": favorable_range / (favorable_range + adverse_range) if favorable_range + adverse_range > 1e-12 else np.nan,
        "path_adverse_range_share_15m": adverse_range / (favorable_range + adverse_range) if favorable_range + adverse_range > 1e-12 else np.nan,
        "path_volatility_of_volatility_15m": float(np.std(minute_rv)) if len(minute_rv) > 1 else np.nan,
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
        record.update(_features(
            entry=float(arrays["entry"][position]), atr=float(arrays["atr"][position]), closes=arrays["close"][position],
            highs=arrays["high"][position], lows=arrays["low"][position], state_minute=int(state.state_minute), side=sides.loc[candidate],
        ))
        records.append(record)
        if ordinal == 1 or ordinal % 100_000 == 0 or ordinal == len(states):
            print(f"volatility states {ordinal}/{len(states)}", flush=True)
    panel = pd.DataFrame(records)
    if len(panel) != len(states) or panel.duplicated(list(KEY)).any():
        raise AssertionError("volatility feature panel lost state identity")
    out.mkdir(parents=True, exist_ok=False)
    panel.to_parquet(out / "h4_directional_chop_volatility_target_free.parquet", index=False, compression="zstd")
    pd.DataFrame({"position": range(len(FEATURE_NAMES)), "feature": FEATURE_NAMES}).to_parquet(out / "feature_contract.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "causal-sr-h4-directional-chop-volatility-v1",
        "scope": "offline target-free research feature block only; no labels, live policy, admission, portfolio, MC1, C1 S/R, Geometry/K9, or exchange mutation",
        "causality": "each state uses only post-fill exact 1m high/low/close bars through state_minute, normalized by immutable entry-time ATR",
        "source": {"parent_root": str(parent_root), "parent_manifest_sha256": _sha256(parent_root / "run_manifest.json"), "state_root": str(state_root), "state_manifest_sha256": _sha256(state_root / "run_manifest.json")},
        "features": list(FEATURE_NAMES), "row_count": int(len(panel)), "no_exchange_calls": True,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
