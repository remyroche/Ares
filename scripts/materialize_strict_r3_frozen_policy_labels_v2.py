#!/usr/bin/env python3
"""Materialise schema-v2 frozen-policy labels from complete 15-minute bars."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from pyarrow.lib import ArrowInvalid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_frozen_policy_labels import (  # noqa: E402
    replay_frozen_policy_15m,
)
from scripts.materialize_packb_tp6_sl4_h12_labels import (  # noqa: E402
    _minute_path_pruned,
    _packb_to_kraken_symbol,
)


def _symbol_path(root: Path, symbol: str) -> Path:
    stem = symbol.lower().replace("/", "").replace("_", "")
    return root / f"{stem}_15m.parquet"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    paths = [path] if path.is_file() else sorted(value for value in path.rglob("*") if value.is_file())
    for value in paths:
        # File hashes must be the ordinary raw-byte SHA-256 used by inference
        # seal validation. Directory hashes additionally bind relative names.
        if path.is_dir():
            digest.update(str(value.relative_to(path)).encode())
        with value.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _complete_15m_from_minute(
    minute_root: Path, symbol: str, start: pd.Timestamp, end: pd.Timestamp,
) -> pd.DataFrame:
    minute = _minute_path_pruned(
        minute_root,
        _packb_to_kraken_symbol(symbol),
        start.floor("15min"),
        end.ceil("15min"),
    )
    finite = minute[["open", "high", "low", "close"]].notna().all(axis=1)
    bars = minute.resample("15min", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
    )
    complete = finite.resample("15min", label="left", closed="left").sum().eq(15)
    bars.loc[~complete, :] = float("nan")
    return bars


def _causal_hourly_atr_from_15m(bars: pd.DataFrame) -> pd.Series:
    """Wilder ATR(14) available at each signal close from complete prior bars.

    A value indexed at ``t`` uses only the four 15-minute candles in
    ``[t-1h, t)`` plus earlier history.  It is therefore a decision-time input,
    not an execution-path proxy.
    """
    ohlc = bars.loc[:, ["open", "high", "low", "close"]].apply(
        pd.to_numeric, errors="coerce"
    )
    finite = pd.Series(
        np.isfinite(ohlc.to_numpy(dtype=np.float64)).all(axis=1), index=ohlc.index
    )
    hourly = ohlc.resample("1h", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
    )
    complete = finite.resample("1h", label="left", closed="left").sum().eq(4)
    hourly.loc[~complete, :] = np.nan
    previous = hourly["close"].shift(1)
    true_range = pd.concat(
        [
            hourly["high"] - hourly["low"],
            (hourly["high"] - previous).abs(),
            (hourly["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
    atr = atr.where(complete.rolling(14, min_periods=14).sum().eq(14))
    atr.index = atr.index + pd.Timedelta(hours=1)
    return atr


def _require_side(frame: pd.DataFrame, side: str, *, purpose: str) -> pd.DataFrame:
    """Return one canonical side without silently mixing economic orientation."""
    canonical = str(side).strip().lower()
    if canonical not in {"long", "short"}:
        raise ValueError(f"{purpose} has a noncanonical requested side: {side!r}")
    if "side_name" not in frame:
        raise ValueError(f"{purpose} lacks side_name")
    observed = frame["side_name"].astype(str).str.lower()
    invalid = ~observed.isin(("long", "short"))
    if invalid.any():
        raise ValueError(f"{purpose} contains noncanonical side values")
    if not observed.eq(canonical).all():
        counts = observed.value_counts(dropna=False).to_dict()
        raise ValueError(
            f"{purpose} must be side-local ({canonical}); observed={counts}",
        )
    result = frame.copy()
    result["side_name"] = canonical
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument(
        "--side", choices=("long", "short"), default="long",
        help=(
            "Economic side to materialise. A run is deliberately side-local; "
            "mixed candidate populations are rejected rather than pooled."
        ),
    )
    parser.add_argument(
        "--atr-context", type=Path,
        help=(
            "Optional candidate_id/atr_1h source.  When omitted, ATR(14) is "
            "derived causally from complete prior 15-minute bars."
        ),
    )
    parser.add_argument("--bar-root", type=Path, required=True)
    parser.add_argument("--minute-root", type=Path)
    parser.add_argument("--policy-json", type=Path, required=True)
    parser.add_argument(
        "--label-available-before",
        help=(
            "Optional strict UTC cutoff. Only candidates whose H12 label is "
            "available before this timestamp are replayed; unresolved/current "
            "paths are never opened."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    policy_payload = json.loads(args.policy_json.read_text())
    policy_side = str(policy_payload.get("side") or "").strip().lower()
    if policy_side != args.side:
        raise ValueError(
            "policy JSON side must match --side; a missing side is not valid "
            "for a short frozen-policy materialisation"
        )
    policy = policy_payload.get("winner", policy_payload)
    policy_keys = ("sl_mult", "trailing_activation_mult", "fixed_trailing_gap_mult")
    missing_policy = [key for key in policy_keys if key not in policy]
    if missing_policy:
        raise ValueError(f"policy JSON lacks {missing_policy}")
    policy = {key: float(policy[key]) for key in policy_keys}
    if args.out_dir.exists() and not args.resume:
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    if args.candidates.is_dir():
        candidate_parts = sorted(
            args.candidates.glob(f"parts/month=*/side={args.side}.parquet"),
        )
        if not candidate_parts:
            raise FileNotFoundError(
                f"no {args.side} candidate parts under {args.candidates}",
            )
        candidates = pd.concat(
            [
                pd.read_parquet(
                    path,
                    columns=[
                        "candidate_id", "__ts__", "__decision_ts__",
                        "__symbol__", "side_name",
                    ],
                )
                for path in candidate_parts
            ],
            ignore_index=True,
        )
    else:
        candidates = pd.read_parquet(args.candidates)
    if "__decision_ts__" not in candidates and "decision_ts" in candidates:
        candidates = candidates.rename(columns={"decision_ts": "__decision_ts__"})
    # Outcome/path columns in a historical path pack are intentionally not
    # propagated into this independent policy-label materialisation.
    candidates = candidates.loc[:, [
        "candidate_id", "__ts__", "__symbol__", "side_name", "__decision_ts__",
    ]].copy()
    candidates = _require_side(
        candidates, args.side, purpose="frozen-policy candidate population",
    )
    label_cutoff = None
    if args.label_available_before is not None:
        label_cutoff = pd.Timestamp(args.label_available_before)
        label_cutoff = (
            label_cutoff.tz_localize("UTC")
            if label_cutoff.tzinfo is None else label_cutoff.tz_convert("UTC")
        )
        decision = pd.to_datetime(candidates["__decision_ts__"], utc=True, errors="raise")
        candidates = candidates.loc[
            (decision + pd.Timedelta(hours=12)).lt(label_cutoff)
        ].copy()
        if candidates.empty:
            raise ValueError("no policy labels are resolved before the declared cutoff")
    if args.atr_context is None:
        atr = candidates.loc[:, ["candidate_id"]].copy()
        atr["atr_1h"] = np.nan
    elif args.atr_context.is_dir():
        atr_parts = sorted(
            args.atr_context.glob(f"parts/month=*/side={args.side}.parquet")
        )
        if not atr_parts:
            raise FileNotFoundError(f"no exact-label ATR parts under {args.atr_context}")
        atr = pd.concat(
            [pd.read_parquet(path, columns=["candidate_id", "atr_1h"]) for path in atr_parts],
            ignore_index=True,
        )
    else:
        atr = pd.read_parquet(args.atr_context, columns=["candidate_id", "atr_1h"])
    frame = candidates.merge(atr, on="candidate_id", how="left", validate="one_to_one")
    labels_path = args.out_dir / "frozen_policy_labels.parquet"
    coverage_path = args.out_dir / "policy_path_coverage.parquet"
    if args.resume and labels_path.exists() and coverage_path.exists():
        labels = pd.read_parquet(labels_path)
        if len(labels) != len(frame) or set(labels["candidate_id"]) != set(frame["candidate_id"]):
            raise ValueError("incomplete policy artifact does not match candidate identity/cardinality")
    else:
        parts: list[pd.DataFrame] = []
        coverage: list[dict[str, object]] = []
        for symbol, block in frame.groupby("__symbol__", sort=True):
            path = _symbol_path(args.bar_root, str(symbol))
            bars = None
            bar_source = "missing"
            local_15m_error = None
            if path.exists():
                try:
                    candidate_bars = pd.read_parquet(
                        path, columns=["open", "high", "low", "close"]
                    )
                    candidate_bars.index = pd.to_datetime(candidate_bars.index, utc=True)
                    required_start = pd.to_datetime(block["__decision_ts__"], utc=True).min()
                    required_end = pd.to_datetime(block["__decision_ts__"], utc=True).max() + pd.Timedelta(hours=12)
                    if (
                        len(candidate_bars)
                        and candidate_bars.index.min() <= required_start
                        and candidate_bars.index.max() >= required_end - pd.Timedelta(minutes=15)
                    ):
                        bars = candidate_bars
                        bar_source = "local_15m"
                except (ArrowInvalid, OSError, ValueError) as exc:
                    # A corrupt local cache must not abort every unrelated
                    # symbol.  Fall back only to the existing local one-minute
                    # archive; no download, interpolation, or future fill is
                    # permitted.
                    local_15m_error = type(exc).__name__
            if bars is None and args.minute_root is not None:
                required_start = pd.to_datetime(block["__decision_ts__"], utc=True).min()
                required_end = pd.to_datetime(block["__decision_ts__"], utc=True).max() + pd.Timedelta(hours=12)
                bars = _complete_15m_from_minute(
                    args.minute_root, str(symbol), required_start, required_end
                )
                bar_source = "local_1m_aggregate" if not bars.empty else "missing"
            if bars is None:
                invalid = block.copy()
                invalid["policy_path_valid"] = False
                invalid["policy_exit_reason"] = "missing_15m_source"
                for column in (
                    "policy_entry_price", "policy_exit_price", "policy_gross_bps",
                    "policy_net_bps", "policy_exit_bar_15m", "policy_cost_bps",
                ):
                    invalid[column] = float("nan")
                invalid["policy_label_available_ts"] = pd.to_datetime(
                    invalid["__decision_ts__"], utc=True,
                ) + pd.Timedelta(hours=12)
                result = invalid
            else:
                # Exact-label ATR coverage may be sparse even where the
                # executable 15-minute path is complete.  Repair only the
                # decision-time ATR input from completed prior 15-minute bars;
                # never inspect the candidate's forward path for this value.
                atr_fallback = _causal_hourly_atr_from_15m(bars)
                signal_ts = pd.to_datetime(block["__ts__"], utc=True)
                stored_atr = pd.to_numeric(block["atr_1h"], errors="coerce")
                fallback_values = signal_ts.map(atr_fallback)
                use_fallback = ~np.isfinite(stored_atr) | stored_atr.le(0.0)
                block = block.copy()
                block["atr_1h"] = stored_atr.where(~use_fallback, fallback_values)
                result = replay_frozen_policy_15m(
                    block, bars,
                    stop_loss_atr=policy["sl_mult"],
                    trailing_activation_atr=policy["trailing_activation_mult"],
                    trailing_giveback_atr=policy["fixed_trailing_gap_mult"],
                )
            parts.append(result)
            coverage.append({
                "symbol": str(symbol), "rows": len(result),
                "side": args.side,
                "valid_rows": int(result["policy_path_valid"].fillna(False).sum()),
                "source_exists": path.exists(),
                "minute_fallback_enabled": args.minute_root is not None,
                "bar_source": bar_source,
                "local_15m_error": local_15m_error,
            })
        labels = pd.concat(parts, ignore_index=True).sort_values(
            ["__decision_ts__", "__symbol__", "side_name"], kind="stable",
        )
        args.out_dir.mkdir(parents=True, exist_ok=args.resume)
        labels.to_parquet(labels_path, index=False, compression="zstd")
        pd.DataFrame(coverage).to_parquet(coverage_path, index=False)
    manifest = {
        "schema": "strict_r3_frozen_policy_15m_labels_v2",
        "side": args.side,
        "candidate_source_sha256": _sha(args.candidates),
        "atr_context_sha256": _sha(args.atr_context) if args.atr_context else None,
        "atr_context": str(args.atr_context) if args.atr_context else None,
        "atr_definition": (
            "stored candidate ATR with causal prior-15m Wilder ATR(14) fallback"
            if args.atr_context else
            "causal prior-15m Wilder ATR(14); no forward execution bars"
        ),
        "bar_root": str(args.bar_root), "rows": len(labels),
        "minute_fallback_root": str(args.minute_root) if args.minute_root else None,
        "label_available_before": (
            label_cutoff.isoformat() if label_cutoff is not None else None
        ),
        "unresolved_or_current_paths_opened": False if label_cutoff is not None else None,
        "policy_json": str(args.policy_json),
        "policy_json_sha256": _sha(args.policy_json),
        "policy": policy,
        "valid_rows": int(labels["policy_path_valid"].sum()),
        "entry": "first 15-minute open at signal close + one hour",
        "exit": (
            f"SL {policy['sl_mult']} ATR; activate trailing at "
            f"{policy['trailing_activation_mult']} ATR; giveback "
            f"{policy['fixed_trailing_gap_mult']} ATR; H12 timeout"
        ),
        "ordering": "stop; prior-bar trailing; current-bar MFE update",
        "cost_bps_once": 100.0,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"event": "complete", "output": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
