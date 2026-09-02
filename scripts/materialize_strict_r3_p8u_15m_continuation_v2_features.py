#!/usr/bin/env python3
"""Materialise the causal v2 continuation-state feature panel, offline only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.p8u_continuation_v2_features import (
    EXTENDED_STATE_FEATURE_KEYS,
    add_causal_age_expectations,
    materialize_extended_state_features,
)
from scripts import run_strict_r3_p8u_15m_continuation_walkforward as base


DEFAULT_STATE_ROOT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_continuation_predictive_observed25h_20260830_v3/target_free_continuation_state_parts"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_continuation_v2_features_20260830_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_parts(root: Path) -> list[Path]:
    parts = sorted(root.glob("symbol=*/states.parquet"))
    if not parts:
        raise FileNotFoundError(f"no source-aligned state parts under {root}")
    return parts


def _symbol_from_part(part: Path) -> str:
    payload = pd.read_parquet(part, columns=["__symbol__"])
    values = payload["__symbol__"].dropna().astype(str).unique().tolist()
    if len(values) != 1:
        raise AssertionError(f"state part has non-unique symbol: {part}")
    return values[0]


def _future_mutation_probe(states: pd.DataFrame, bars: pd.DataFrame) -> dict[str, object]:
    """Assert a later bar does not alter an already completed feature row."""
    if states.empty:
        return {"checked": 0, "pass": True}
    sample = states.sort_values("state_decision_ts", kind="stable").head(1)
    first = materialize_extended_state_features(sample, bars)
    latest = bars.index.max()
    mutated = bars.copy()
    # Strictly later than every state in the probe.  This value is intentionally
    # extreme so any accidental future dependency is observable.
    ts = latest + pd.Timedelta(minutes=15)
    reference = mutated.iloc[-1].copy()
    reference[["open", "high", "low", "close"]] = [1.0e9, 1.1e9, 0.9e9, 1.05e9]
    reference["volume"] = np.float32(1.0e12)
    mutated.loc[ts] = reference
    second = materialize_extended_state_features(sample, mutated)
    columns = list(EXTENDED_STATE_FEATURE_KEYS)
    a = first.loc[:, columns].to_numpy(float)
    b = second.loc[:, columns].to_numpy(float)
    equal = np.allclose(a, b, rtol=0.0, atol=0.0, equal_nan=True)
    return {"checked": 1, "pass": bool(equal), "candidate_id": str(sample.iloc[0]["candidate_id"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--symbol", action="append", help="optional repeatable symbol subset for a smoke run")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    state_root = args.state_root.resolve()
    parts = _state_parts(state_root)
    if args.symbol:
        selected = {str(value) for value in args.symbol}
        parts = [part for part in parts if _symbol_from_part(part) in selected]
    if not parts:
        raise RuntimeError("no selected continuation state parts")
    frames: list[pd.DataFrame] = []
    coverage: list[dict[str, object]] = []
    probes: list[dict[str, object]] = []
    for part in parts:
        states = pd.read_parquet(part)
        symbol = _symbol_from_part(part)
        bar_path = base.BARS_ROOT / base._symbol_filename(symbol)
        if not bar_path.is_file():
            coverage.append({"symbol": symbol, "source_states": len(states), "v2_states": 0, "reason": "missing_15m_source"})
            continue
        bars = pd.read_parquet(bar_path, columns=["open", "high", "low", "close", "volume"])
        bars.index = pd.to_datetime(bars.index, utc=True, errors="raise")
        panel = materialize_extended_state_features(states, bars)
        frames.append(panel)
        coverage.append({
            "symbol": symbol, "source_states": len(states), "v2_states": len(panel),
            "reason": "ok", "finite_extended_feature_mean": float(panel.loc[:, EXTENDED_STATE_FEATURE_KEYS].notna().sum(axis=1).mean()),
        })
        probes.append(_future_mutation_probe(states, bars))
    if not frames:
        raise RuntimeError("no v2 state frames were materialised")
    raw = pd.concat(frames, ignore_index=True)
    panel = add_causal_age_expectations(raw)
    if len(panel) != len(raw) or panel.duplicated(["candidate_id", "state_decision_ts", "state_bar_15m"]).any():
        raise AssertionError("v2 feature materialisation changed state identity")
    if not all(bool(item["pass"]) for item in probes):
        raise AssertionError("future-mutation probe changed a completed feature row")
    if not panel["state_source_end_ts"].lt(panel["state_decision_ts"]).all():
        raise AssertionError("state source includes the decision timestamp or later")
    output.mkdir(parents=True, exist_ok=False)
    panel.to_parquet(output / "continuation_v2_state_features.parquet", index=False, compression="zstd")
    pd.DataFrame(coverage).to_parquet(output / "source_coverage.parquet", index=False)
    pd.DataFrame(probes).to_parquet(output / "future_mutation_probe.parquet", index=False)
    feature_coverage = pd.DataFrame({
        "feature": EXTENDED_STATE_FEATURE_KEYS,
        "finite_fraction": [float(panel[name].notna().mean()) for name in EXTENDED_STATE_FEATURE_KEYS],
    })
    feature_coverage["available"] = feature_coverage["finite_fraction"].gt(0.50)
    feature_coverage.to_parquet(output / "feature_coverage.parquet", index=False)
    manifest = {
        "schema": "strict-r3-p8u-15m-continuation-v2-causal-features-v1",
        "scope": "offline research only; no model fit, policy change, exchange IO, or order submission",
        "state_source": str(state_root),
        "state_part_count": len(parts),
        "state_part_sha256": {str(path): _sha256(path) for path in parts},
        "bars": "retained 15m OHLCV source used by the parent continuation state; bars strictly before each state_decision_ts",
        "extended_features": list(EXTENDED_STATE_FEATURE_KEYS),
        "direction": "trade-direction signed features; long in this study",
        "expectations": "90-day trailing prior-only age x fixed initial-ATR-fraction band means; simultaneous rows are excluded until after feature evaluation",
        "oi": "no retained causal 15m OI source was joined; OI-specific requested fields are explicitly NaN and unavailable for selection",
        "future_mutation_probe": "one extreme later 15m bar per symbol; exact equality required for all deterministic v2 features",
        "rows": len(panel),
        "candidates": int(panel["candidate_id"].nunique()),
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
