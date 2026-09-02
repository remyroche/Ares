#!/usr/bin/env python3
"""Materialise target-free 15-minute path deterioration/recovery features.

This is an offline research-only feature builder.  It operates on the same
exact post-fill one-minute paths used by the H4 actuator study, and emits one
row for every pre-existing H4 decision state.  Each feature uses only minute
bars ending no later than that state decision; it never reads a policy label,
subsequent state, exit, or realised outcome.

The block intentionally complements rather than duplicates the 91-field H4
contract: it represents the *sequence* of favourable/adverse impulses,
recoveries, pullbacks, and failed breaks.  It does not modify the live feature
contract or any execution component.
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 1e-12:
        return np.nan
    return float(numerator / denominator)


def _runs(values: np.ndarray, positive: bool) -> list[tuple[float, int]]:
    """Contiguous signed 15m-close-return impulses (size, number of bars)."""
    active = 0.0
    count = 0
    result: list[tuple[float, int]] = []
    for value in values:
        take = value > 0.0 if positive else value < 0.0
        if take:
            active += float(value if positive else -value)
            count += 1
        elif count:
            result.append((float(active), int(count)))
            active = 0.0
            count = 0
    if count:
        result.append((float(active), int(count)))
    return result


def _path_features(
    *,
    entry: float,
    atr: float,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    state_minute: int,
    side: str,
) -> dict[str, float]:
    """Compute a causal impulse/path state through one completed H4 state."""
    direction = 1.0 if str(side).lower() == "long" else -1.0
    end = min(max(int(state_minute), 0), len(closes) - 1)
    close = np.asarray(closes[: end + 1], dtype=float)
    high = np.asarray(highs[: end + 1], dtype=float)
    low = np.asarray(lows[: end + 1], dtype=float)
    if len(close) < 2 or not np.isfinite([entry, atr]).all() or entry <= 0.0 or atr <= 0.0:
        return {name: np.nan for name in FEATURE_NAMES}

    # State boundaries correspond to completed calendar 15-minute intervals
    # after an x:05 entry: the first has ten post-fill minutes, then 15m each.
    # The 1m arrays are post-fill only, so every endpoint is causal.
    endpoints = np.arange(9, end + 1, 15, dtype=int)
    if len(endpoints) == 0 or endpoints[-1] != end:
        endpoints = np.append(endpoints, end)
    endpoints = np.unique(endpoints)
    block_close = close[endpoints]
    prior = np.concatenate(([float(entry)], block_close[:-1]))
    returns = direction * (block_close - prior) / atr
    # Exclude the partial duplicate endpoint if the state does not land on a
    # boundary; it is still a valid latest bar but has fewer minutes.  Its
    # magnitude remains normalized by the immutable entry ATR.
    favorable_runs = _runs(returns, positive=True)
    adverse_runs = _runs(returns, positive=False)
    last_fav = favorable_runs[-1] if favorable_runs else (0.0, 0)
    prev_fav = favorable_runs[-2] if len(favorable_runs) > 1 else (np.nan, 0)
    last_adv = adverse_runs[-1] if adverse_runs else (0.0, 0)
    prev_adv = adverse_runs[-2] if len(adverse_runs) > 1 else (np.nan, 0)

    # Short windows operate on the latest four completed/partial 15m bars;
    # long states therefore do not receive accidental infinite history.
    recent_start = max(0, len(endpoints) - 4)
    recent_start_minute = 0 if recent_start == 0 else int(endpoints[recent_start - 1] + 1)
    recent_close = close[recent_start_minute : end + 1]
    recent_high = high[recent_start_minute : end + 1]
    recent_low = low[recent_start_minute : end + 1]
    aligned_close = direction * (recent_close - float(entry))
    aligned_high = direction * (recent_high - float(entry))
    aligned_low = direction * (recent_low - float(entry))
    peak_index = int(np.argmax(aligned_high))
    trough_index = int(np.argmin(aligned_low))
    current = float(aligned_close[-1])
    recent_peak = float(aligned_high[peak_index])
    recent_trough = float(aligned_low[trough_index])
    pullback = max(recent_peak - current, 0.0)
    recovery = max(current - recent_trough, 0.0)
    recovery_bars = max(len(aligned_close) - 1 - trough_index, 1)
    pullback_bars = max(len(aligned_close) - 1 - peak_index, 1)

    # Directional efficiency is a ratio of signed displacement to travelled
    # distance.  The decay compares the latest two 15m blocks, not future
    # values.  Adverse efficiency is specifically the cleanness of the path
    # opposite to the trade direction.
    block_efficiencies: list[float] = []
    adverse_efficiencies: list[float] = []
    for idx, endpoint in enumerate(endpoints):
        begin = 0 if idx == 0 else int(endpoints[idx - 1] + 1)
        segment = direction * np.diff(np.concatenate(([float(entry) if idx == 0 else close[begin - 1]], close[begin : endpoint + 1]))) / atr
        travelled = float(np.abs(segment).sum())
        net = float(segment.sum())
        block_efficiencies.append(_safe_ratio(abs(net), travelled))
        adverse_efficiencies.append(_safe_ratio(max(-net, 0.0), travelled))
    latest_eff = block_efficiencies[-1] if block_efficiencies else np.nan
    previous_eff = block_efficiencies[-2] if len(block_efficiencies) > 1 else np.nan
    latest_adverse_eff = adverse_efficiencies[-1] if adverse_efficiencies else np.nan

    # A failed favourable break: most recent block made a new local aligned
    # high but closed back through the preceding local high.  This is a
    # target-free price-structure condition, not a realised-exit feature.
    failed_break = 0.0
    if len(endpoints) >= 3:
        current_begin = int(endpoints[-2] + 1)
        before_begin = 0 if len(endpoints) <= 3 else int(endpoints[-4] + 1)
        prior_peak = float(np.max(direction * (high[before_begin : endpoints[-2] + 1] - float(entry))))
        current_peak = float(np.max(direction * (high[current_begin : end + 1] - float(entry))))
        failed_break = float(current_peak > prior_peak and current < prior_peak)

    prior_pulls: list[float] = []
    if len(endpoints) >= 2:
        for idx, endpoint in enumerate(endpoints[-3:]):
            start_idx = max(0, len(endpoints) - 3) + idx
            begin = 0 if start_idx == 0 else int(endpoints[start_idx - 1] + 1)
            segment_high = float(np.max(direction * (high[begin : endpoint + 1] - float(entry))))
            segment_close = float(direction * (close[endpoint] - float(entry)))
            prior_pulls.append(max(segment_high - segment_close, 0.0) / atr)
    current_pull_atr = pullback / atr
    prior_pull = prior_pulls[-2] if len(prior_pulls) > 1 else np.nan

    return {
        "path_favorable_impulse_strength_15m": float(last_fav[0]),
        "path_adverse_impulse_strength_15m": float(last_adv[0]),
        "path_favorable_vs_adverse_impulse_ratio_15m": _safe_ratio(float(last_fav[0]), float(last_adv[0])),
        "path_favorable_impulse_decay_15m": _safe_ratio(float(last_fav[0]), float(prev_fav[0])),
        "path_adverse_impulse_acceleration_15m": _safe_ratio(float(last_adv[0]), float(prev_adv[0])),
        "path_directional_efficiency_decay_15m": float(latest_eff - previous_eff) if np.isfinite(latest_eff) and np.isfinite(previous_eff) else np.nan,
        "path_adverse_efficiency_15m": float(latest_adverse_eff),
        "path_recovery_strength_atr_15m": float(recovery / atr),
        "path_recovery_speed_atr_per_15m": float((recovery / atr) * 15.0 / recovery_bars),
        "path_recovery_strength_decay_15m": _safe_ratio(float(recovery / atr), float(max(current_pull_atr, 1e-12))),
        "path_pullback_depth_atr_15m": float(current_pull_atr),
        "path_pullback_duration_15m": float(pullback_bars / 15.0),
        "path_pullback_speed_atr_per_15m": float((pullback / atr) * 15.0 / pullback_bars),
        "path_pullback_severity_trend_15m": float(current_pull_atr - prior_pull) if np.isfinite(prior_pull) else np.nan,
        "path_failed_favorable_break_15m": float(failed_break),
        "path_favorable_impulse_count_15m": float(len(favorable_runs)),
        "path_adverse_impulse_count_15m": float(len(adverse_runs)),
    }


FEATURE_NAMES = tuple(_path_features(
    entry=1.0, atr=1.0, closes=np.array([1.0, 1.0]), highs=np.array([1.0, 1.0]),
    lows=np.array([1.0, 1.0]), state_minute=1, side="long",
).keys())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, default=base.DEFAULT_PARENT)
    parser.add_argument("--state-root", type=Path, default=base.DEFAULT_STATES)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    parent_root = args.parent_root.resolve()
    state_root = args.state_root.resolve()
    _, rows, _, arrays, states = base._load_parent(parent_root, state_root)
    position = pd.Series(np.arange(len(rows), dtype=np.int64), index=rows["candidate_id"].astype(str))
    side = rows.set_index("candidate_id")["side_name"].astype(str)
    records: list[dict[str, object]] = []
    for ordinal, state in enumerate(states.loc[:, [*KEY, "state_minute"]].itertuples(index=False), 1):
        candidate = str(state.candidate_id)
        index = int(position.loc[candidate])
        record: dict[str, object] = {
            "candidate_id": candidate,
            "state_decision_ts": pd.Timestamp(state.state_decision_ts),
        }
        record.update(_path_features(
            entry=float(arrays["entry"][index]), atr=float(arrays["atr"][index]),
            closes=arrays["close"][index], highs=arrays["high"][index], lows=arrays["low"][index],
            state_minute=int(state.state_minute), side=str(side.loc[candidate]),
        ))
        records.append(record)
        if ordinal == 1 or ordinal % 100_000 == 0 or ordinal == len(states):
            print(f"path states {ordinal}/{len(states)}", flush=True)
    panel = pd.DataFrame(records)
    if panel.duplicated(list(KEY)).any() or len(panel) != len(states):
        raise AssertionError("path feature panel lost state identity")
    if not panel.loc[:, FEATURE_NAMES].apply(lambda col: np.isfinite(col).any()).all():
        missing = [name for name in FEATURE_NAMES if not np.isfinite(panel[name]).any()]
        raise AssertionError(f"path feature block has all-missing fields: {missing}")
    out.mkdir(parents=True, exist_ok=False)
    panel.to_parquet(out / "h4_path_deterioration_target_free.parquet", index=False, compression="zstd")
    pd.DataFrame({"position": range(len(FEATURE_NAMES)), "feature": FEATURE_NAMES}).to_parquet(
        out / "feature_contract.parquet", index=False, compression="zstd",
    )
    manifest = {
        "schema": "causal-sr-h4-path-deterioration-v1",
        "scope": "offline target-free H4 research feature block only; no labels, live policy, admission, portfolio, MC1, C1 S/R, Geometry/K9, or exchange mutation",
        "causality": "each state uses only that candidate's post-fill exact 1m high/low/close values through state_minute; all features are normalized by immutable entry-time signal ATR",
        "source": {"parent_root": str(parent_root), "parent_manifest_sha256": _sha256(parent_root / "run_manifest.json"), "state_root": str(state_root), "state_manifest_sha256": _sha256(state_root / "run_manifest.json")},
        "features": list(FEATURE_NAMES),
        "row_count": int(len(panel)),
        "no_exchange_calls": True,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
