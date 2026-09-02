#!/usr/bin/env python3
"""Build a full-population exact +5m H4 state ledger for offline promotion replay.

The source is the sealed May--July v58-style dual-MC1 candidate/path ledger.
Each state is emitted only after a completed UTC 15-minute bar while the
unchanged one-minute rich parent position remains open.  Dynamic MFE/MAE are
strictly post-fill; context features use completed 15-minute market bars.
This is offline research only and contains no exchange or live execution path.
"""

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

from extreme_price_movements.exact_1m_h4_overlay_research import replay_exact_1m_h4_giveback20
from extreme_price_movements.p8u_continuation_v2_features import (
    add_causal_age_expectations,
    materialize_extended_state_features,
)
from scripts.run_strict_r3_p8u_v58_e2_50_exact_matched import (
    DEFAULT_PATH_ROOT,
    DEFAULT_POLICY,
    DEFAULT_SCORE_LEDGER,
    _load_policy,
    _load_route,
)
from scripts.run_strict_r3_p8u_15m_continuation_walkforward import BARS_ROOT, _symbol_filename


DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/strict_r3_p8u_v58_exact_h4_states_20260830_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_bars(symbol: str) -> pd.DataFrame | None:
    path = BARS_ROOT / _symbol_filename(symbol)
    if not path.is_file():
        return None
    try:
        bars = pd.read_parquet(path, columns=["open", "high", "low", "close", "volume"])
        bars.index = pd.to_datetime(bars.index, utc=True, errors="raise")
        bars = bars.loc[~bars.index.duplicated(keep="last")].sort_index()
        return bars
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-ledger", type=Path, default=DEFAULT_SCORE_LEDGER)
    parser.add_argument("--path-root", type=Path, default=DEFAULT_PATH_ROOT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")

    route = _load_route(args.score_ledger.resolve(), args.path_root.resolve())
    rows = pd.read_parquet(args.path_root.resolve() / "valid_exact_paths_rows.parquet")
    rows["candidate_id"] = rows["candidate_id"].astype(str)
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True, errors="raise")
    rows["entry_ts"] = pd.to_datetime(rows["entry_ts"], utc=True, errors="raise")
    route = route.merge(
        rows.loc[:, ["candidate_id", "timestamp", "entry_ts", "symbol"]],
        on=["candidate_id", "timestamp", "entry_ts", "symbol"], how="inner", validate="one_to_one",
    )
    if route.empty:
        raise RuntimeError("sealed route has no exact-path overlap")
    row_index = pd.Series(np.arange(len(rows), dtype=np.int64), index=rows.candidate_id.astype(str))
    route["__path_index__"] = route.candidate_id.map(row_index)
    if route["__path_index__"].isna().any():
        raise AssertionError("route/path identity mismatch")
    route["__path_index__"] = route["__path_index__"].astype(int)
    archive = np.load(args.path_root.resolve() / "exact_paths.npz", allow_pickle=False)
    entry = np.asarray(archive["entry"], dtype=float)
    atr = np.asarray(archive["atr"], dtype=float)
    high = np.asarray(archive["high"], dtype=np.float32)
    low = np.asarray(archive["low"], dtype=np.float32)
    close = np.asarray(archive["close"], dtype=np.float32)
    params, median, policy_receipt = _load_policy(args.policy.resolve())

    state_parts: list[pd.DataFrame] = []
    outcomes: list[dict[str, object]] = []
    for symbol, group in route.groupby("symbol", sort=True):
        for _, row in group.sort_values(["timestamp", "candidate_id"], kind="stable").iterrows():
            idx = int(row["__path_index__"])
            trace = replay_exact_1m_h4_giveback20(
                entry_price=float(entry[idx]), signal_atr=float(atr[idx]), entry_ts=row["entry_ts"],
                highs=high[idx], lows=low[idx], closes=close[idx], params=params,
                median_atr_fraction=float(median), mc1_expected_bps=float(row["bcf_mc1_expected_bps"]),
                state_decider=None, emit_states=True,
            )
            outcomes.append({
                "candidate_id": str(row["candidate_id"]), "parent_exact_net_bps": float(trace["net_bps"]),
                "parent_exact_gross_bps": float(trace["gross_bps"]), "parent_exit_minute": int(trace["exit_minute"]),
                "parent_exit_reason": str(trace["exit_reason"]), "parent_exit_ts": pd.Timestamp(trace["exit_timestamp"]),
            })
            states = trace["states"]
            if not states:
                continue
            part = pd.DataFrame(states)
            part["candidate_id"] = str(row["candidate_id"])
            part["__symbol__"] = str(symbol)
            part["entry_decision_ts"] = pd.Timestamp(row["timestamp"])
            part["entry_price"] = float(entry[idx])
            part["signal_atr"] = float(atr[idx])
            part["state_bar_15m"] = np.arange(len(part), dtype=np.int16)
            # The v2 feature producer expects the legacy bps field.  Preserve
            # the raw dynamic ATR value separately and overwrite the required
            # H4 fields after feature materialisation below.
            part["current_PnL"] = (
                part["current_pnl_atr"].to_numpy(float) * float(atr[idx]) / float(entry[idx]) * 10_000.0 - 100.0
            )
            state_parts.append(part)

    parent = pd.DataFrame(outcomes)
    if parent.candidate_id.duplicated().any() or len(parent) != len(route):
        raise AssertionError("exact parent outcome identity was not one-to-one")
    raw_states = pd.concat(state_parts, ignore_index=True) if state_parts else pd.DataFrame()
    if raw_states.empty:
        raise RuntimeError("no completed exact H4 states")
    keys = ["candidate_id", "state_decision_ts", "state_bar_15m"]
    if raw_states.duplicated(keys).any():
        raise AssertionError("exact H4 state identity is not unique")

    # Contextual C4 fields are deterministic and strictly prior to each state.
    expanded: list[pd.DataFrame] = []
    coverage: list[dict[str, object]] = []
    for symbol, group in raw_states.groupby("__symbol__", sort=True):
        bars = _read_bars(str(symbol))
        if bars is None:
            coverage.append({"symbol": symbol, "state_rows": len(group), "materialised": 0, "reason": "missing_or_unreadable_15m_source"})
            continue
        try:
            # The helper derives a convenient ``current_pnl_atr`` from its
            # legacy bps input.  Keep the exact dynamic version out of the
            # join, then restore it below from the raw state identity.
            features = materialize_extended_state_features(group.drop(columns=["current_pnl_atr"]), bars, side="long")
        except Exception as exc:
            coverage.append({"symbol": symbol, "state_rows": len(group), "materialised": 0, "reason": f"feature_error:{type(exc).__name__}"})
            continue
        expanded.append(features)
        coverage.append({"symbol": symbol, "state_rows": len(group), "materialised": len(features), "reason": "ok"})
    if not expanded:
        raise RuntimeError("no exact H4 state feature rows were materialised")
    panel = pd.concat(expanded, ignore_index=True)
    exact_dynamic = raw_states.loc[:, [*keys, "current_pnl_atr"]]
    panel = panel.merge(exact_dynamic, on=keys, how="left", validate="one_to_one", suffixes=("", "__exact"))
    if "current_pnl_atr__exact" not in panel:
        raise AssertionError("exact H4 dynamic PnL did not survive state expansion")
    panel["current_pnl_atr"] = panel.pop("current_pnl_atr__exact")
    # Preserve the live-exact dynamic state rather than the 15m helper's
    # convenient reconstructive aliases.
    for name in (
        "current_pnl_atr", "current_MFE_ATR", "current_MAE_ATR",
        "giveback_from_MFE_ATR", "distance_to_current_SL_ATR",
        "is_trailing_active", "current_protection_state", "MC1_expected_bps",
    ):
        panel[name] = pd.to_numeric(panel[name], errors="coerce")
    panel = add_causal_age_expectations(panel)
    if panel.duplicated(keys).any():
        raise AssertionError("H4 state feature expansion changed identity")

    output.mkdir(parents=True, exist_ok=False)
    route.to_parquet(output / "target_free_route.parquet", index=False, compression="zstd")
    raw_states.to_parquet(output / "exact_parent_h4_states_target_free.parquet", index=False, compression="zstd")
    panel.to_parquet(output / "exact_parent_h4_state_features_target_free.parquet", index=False, compression="zstd")
    parent.to_parquet(output / "exact_parent_outcomes.parquet", index=False, compression="zstd")
    pd.DataFrame(coverage).to_parquet(output / "state_feature_coverage.parquet", index=False)
    manifest = {
        "schema": "strict-r3-p8u-v58-exact-h4-states-v1",
        "scope": "offline promotion research only; no live config, exchange IO, or order submission",
        "route": "sealed Router50, dual BCF/current MC1 >=50 source route",
        "entry": "exact +5m one-minute path entry",
        "state": "post-fill dynamic MFE/MAE; completed UTC 15-minute state; action may only affect following interval",
        "parent_parity": "H4-disabled adapter is separately tested against frozen exact one-minute parent policy",
        "score_ledger": str(args.score_ledger.resolve()), "score_ledger_sha256": _sha256(args.score_ledger.resolve()),
        "paths": str(args.path_root.resolve()), "policy": str(args.policy.resolve()),
        "policy_sha256": _sha256(args.policy.resolve()), "policy_receipt": policy_receipt,
        "route_rows": len(route), "parent_rows": len(parent), "state_rows": len(panel),
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
