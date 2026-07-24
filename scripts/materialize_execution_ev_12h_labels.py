#!/usr/bin/env python3
"""Materialize causal 12-hour execution-EV labels from canonical hourly bars."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.data_store import PartitionedOHLCVStore  # noqa: E402
from extreme_price_movements.execution_ev_labels import (  # noqa: E402
    policy_geometry_from_manifest,
    reason_names,
    simulate_execution_ev_12h,
)

SCHEMA = "execution_ev_12h_hourly_policy_labels_v2"
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _signed_manifest_hash(payload: Mapping[str, Any]) -> str:
    canonical = {
        str(key): _json_safe(value)
        for key, value in payload.items()
        if key != "prediction_role_manifest_sha256"
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_candidates(
    path: Path,
    *,
    fee_round_trip_return: float | None = None,
    spread_return_col: str | None = None,
    spread_map_csv: Path | None = None,
    spread_map_symbol_col: str = "symbol",
    spread_map_bps_col: str = "p90_spread_bps",
    missing_spread_policy: str = "reject",
) -> pd.DataFrame:
    if spread_return_col is not None and spread_map_csv is not None:
        raise ValueError(
            "use either spread_return_col or spread_map_csv, not both"
        )
    if missing_spread_policy not in {"reject", "drop"}:
        raise ValueError("missing_spread_policy must be 'reject' or 'drop'")
    columns = [
        "__ts__",
        "__symbol__",
        "candidate_id",
        "side",
        "side_name",
        "__path_auxiliary_atr_fraction__",
        "path_cost_return",
        *( [spread_return_col] if spread_return_col else [] ),
    ]
    try:
        import pyarrow.parquet as pq

        available = set(pq.read_schema(path).names)
    except ImportError:  # pragma: no cover
        available = set(pd.read_parquet(path).columns)
    selected = [column for column in columns if column in available]
    frame = pd.read_parquet(path, columns=selected)
    source_rows = len(frame)
    missing = {
        "__ts__",
        "__symbol__",
        "candidate_id",
        "__path_auxiliary_atr_fraction__",
    } - set(frame)
    if missing:
        raise ValueError(f"candidate source is missing {sorted(missing)}")
    if "side_name" not in frame and "side" not in frame:
        raise ValueError("candidate source requires side_name or side")
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    if frame["__ts__"].isna().any():
        raise ValueError("candidate source contains invalid UTC timestamps")
    frame["__symbol__"] = frame["__symbol__"].astype("string").str.strip()
    if frame["__symbol__"].isna().any() or frame["__symbol__"].eq("").any():
        raise ValueError("candidate source contains blank symbols")
    frame["candidate_id"] = frame["candidate_id"].astype("string").str.strip()
    if frame["candidate_id"].isna().any() or frame["candidate_id"].eq("").any():
        raise ValueError("candidate source contains blank candidate IDs")
    raw_side = frame.get("side_name", frame.get("side")).astype(str).str.lower()
    frame["side_name"] = np.where(raw_side.isin(("short", "sell", "-1", "-1.0")), "short", "long")
    frame["side_sign"] = np.where(frame["side_name"].eq("short"), -1.0, 1.0).astype(np.float32)
    frame["atr_fraction"] = pd.to_numeric(
        frame["__path_auxiliary_atr_fraction__"], errors="coerce"
    ).astype(np.float32)
    missing_symbols: list[str] = []
    missing_spread_rows = 0
    if spread_map_csv is not None:
        spread_map = pd.read_csv(
            spread_map_csv,
            usecols=[spread_map_symbol_col, spread_map_bps_col],
        )
        spread_map[spread_map_symbol_col] = (
            spread_map[spread_map_symbol_col].astype("string").str.strip()
        )
        spread_map[spread_map_bps_col] = pd.to_numeric(
            spread_map[spread_map_bps_col], errors="coerce"
        )
        if (
            spread_map[spread_map_symbol_col].isna().any()
            or spread_map[spread_map_symbol_col].eq("").any()
            or spread_map[spread_map_symbol_col].duplicated().any()
            or spread_map[spread_map_bps_col].isna().any()
            or (spread_map[spread_map_bps_col] < 0).any()
        ):
            raise ValueError("spread map must have one finite non-negative row per symbol")
        mapped = frame["__symbol__"].map(
            spread_map.set_index(spread_map_symbol_col)[spread_map_bps_col]
        )
        if mapped.isna().any():
            missing_symbols = sorted(frame.loc[mapped.isna(), "__symbol__"].unique())
            if missing_spread_policy == "reject":
                raise ValueError(
                    "spread map is missing candidate symbols: "
                    + ", ".join(missing_symbols[:20])
                )
            missing_spread_rows = int(mapped.isna().sum())
            frame = frame.loc[mapped.notna()].reset_index(drop=True)
            mapped = mapped.loc[mapped.notna()].reset_index(drop=True)
        else:
            missing_symbols = []
            missing_spread_rows = 0
        spread_return_col = "__p90_full_spread_return__"
        frame[spread_return_col] = (mapped / 10_000.0).astype(np.float32)

    if spread_return_col is not None:
        if fee_round_trip_return is None:
            raise ValueError(
                "fee_round_trip_return is required with spread_return_col"
            )
        if not np.isfinite(fee_round_trip_return) or fee_round_trip_return < 0:
            raise ValueError("fee_round_trip_return must be finite and non-negative")
        if spread_return_col not in frame:
            raise ValueError(
                f"candidate source is missing spread return column {spread_return_col!r}"
            )
        spread = pd.to_numeric(frame[spread_return_col], errors="coerce")
        if spread.isna().any() or (spread < 0).any():
            raise ValueError(
                f"candidate source has invalid spread returns in {spread_return_col!r}"
            )
        frame["fee_return"] = np.float32(fee_round_trip_return)
        frame["spread_return"] = spread.astype(np.float32)
        frame["cost_return"] = (
            frame["fee_return"] + frame["spread_return"]
        ).astype(np.float32)
        cost_contract = "explicit_fee_plus_full_p90_spread"
    else:
        frame["cost_return"] = pd.to_numeric(
            frame.get("path_cost_return", pd.Series(0.003, index=frame.index)),
            errors="coerce",
        ).fillna(0.003).clip(lower=0.0).astype(np.float32)
        frame["fee_return"] = frame["cost_return"]
        frame["spread_return"] = np.float32(0.0)
        cost_contract = "legacy_candidate_total_cost"
    invalid_atr = ~np.isfinite(frame["atr_fraction"]) | (frame["atr_fraction"] <= 0.0)
    invalid_atr_rows = int(invalid_atr.sum())
    if invalid_atr_rows:
        frame = frame.loc[~invalid_atr].reset_index(drop=True)
    if frame.empty:
        raise ValueError("candidate source has no row with a valid ATR fraction")
    if frame.duplicated(list(IDENTITY)).any():
        raise ValueError("candidate source violates unique timestamp/symbol/side identity")
    frame.attrs.update(
        {
            "source_rows": int(source_rows),
            "invalid_atr_rows_excluded": invalid_atr_rows,
            "cost_contract": cost_contract,
            "fee_round_trip_return": (
                float(fee_round_trip_return)
                if fee_round_trip_return is not None
                else None
            ),
            "spread_return_column": spread_return_col,
            "spread_map_csv": (
                str(spread_map_csv.resolve())
                if spread_map_csv is not None
                else None
            ),
            "spread_map_sha256": (
                _sha256(spread_map_csv) if spread_map_csv is not None else None
            ),
            "missing_spread_policy": missing_spread_policy,
            "missing_spread_rows_excluded": int(missing_spread_rows),
            "missing_spread_symbols": list(missing_symbols),
        }
    )
    return frame


def _paths_for_symbol(
    store: PartitionedOHLCVStore,
    symbol: str,
    decision_ts: pd.Series,
    *,
    horizon_hours: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    start, end = decision_ts.min(), decision_ts.max() + pd.Timedelta(hours=horizon_hours)
    bars = store.load(symbol, columns=["open", "high", "low", "close"], start_ts=start, end_ts=end)
    shape = (len(decision_ts), horizon_hours)
    output = [np.full(shape, np.nan, dtype=np.float32) for _ in range(4)]
    valid = np.zeros(len(decision_ts), dtype=bool)
    if bars is None or bars.empty or not isinstance(bars.index, pd.DatetimeIndex):
        return (*output, valid)
    bars = bars[~bars.index.duplicated(keep="last")].sort_index()
    index = bars.index.tz_localize("UTC") if bars.index.tz is None else bars.index.tz_convert("UTC")
    index_ns = index.astype("int64").to_numpy(dtype=np.int64)
    decision_ns = decision_ts.astype("int64").to_numpy(dtype=np.int64)
    starts = np.searchsorted(index_ns, decision_ns)
    offsets = np.arange(horizon_hours, dtype=np.int64)
    positions = starts[:, None] + offsets[None, :]
    in_bounds = positions[:, -1] < len(index_ns)
    local = np.flatnonzero(in_bounds)
    if len(local):
        expected = decision_ns[local, None] + offsets[None, :] * 3_600_000_000_000
        contiguous = np.all(index_ns[positions[local]] == expected, axis=1)
        local = local[contiguous]
    if not len(local):
        return (*output, valid)
    for target, column in zip(output, ("open", "high", "low", "close")):
        values = pd.to_numeric(bars[column], errors="coerce").to_numpy(dtype=np.float32)
        target[local] = values[positions[local]]
    valid[local] = np.logical_and.reduce(
        [np.isfinite(values[local]).all(axis=1) for values in output]
    )
    return (*output, valid)


def materialize(
    candidates_path: Path,
    ohlcv_root: Path,
    policy_manifest_path: Path,
    output: Path,
    manifest_path: Path,
    *,
    decision_delay_hours: int = 1,
    horizon_hours: int = 12,
    fee_round_trip_return: float | None = None,
    spread_return_col: str | None = None,
    spread_map_csv: Path | None = None,
    spread_map_symbol_col: str = "symbol",
    spread_map_bps_col: str = "p90_spread_bps",
    missing_spread_policy: str = "reject",
) -> dict[str, Path]:
    if output.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite execution-EV label artifacts")
    candidates = _canonical_candidates(
        candidates_path,
        fee_round_trip_return=fee_round_trip_return,
        spread_return_col=spread_return_col,
        spread_map_csv=spread_map_csv,
        spread_map_symbol_col=spread_map_symbol_col,
        spread_map_bps_col=spread_map_bps_col,
        missing_spread_policy=missing_spread_policy,
    )
    source_rows = int(candidates.attrs.get("source_rows", len(candidates)))
    invalid_atr_rows = int(candidates.attrs.get("invalid_atr_rows_excluded", 0))
    policy_payload = json.loads(policy_manifest_path.read_text(encoding="utf-8"))
    long_geometry = policy_geometry_from_manifest(policy_payload, "long")
    short_geometry = policy_geometry_from_manifest(policy_payload, "short")
    candidates["__decision_ts__"] = candidates["__ts__"] + pd.Timedelta(
        hours=decision_delay_hours
    )
    candidates["execution_label_end_utc"] = candidates["__decision_ts__"] + pd.Timedelta(
        hours=horizon_hours
    )
    n_rows = len(candidates)
    gross = np.full(n_rows, np.nan, dtype=np.float64)
    net = np.full(n_rows, np.nan, dtype=np.float64)
    reason = np.full(n_rows, -1, dtype=np.int8)
    exit_bar = np.full(n_rows, -1, dtype=np.int16)
    mfe = np.full(n_rows, np.nan, dtype=np.float64)
    mae = np.full(n_rows, np.nan, dtype=np.float64)
    store = PartitionedOHLCVStore(str(ohlcv_root), timeframe="1h")
    groups = candidates.groupby("__symbol__", sort=True).groups
    for number, (symbol, index) in enumerate(groups.items(), start=1):
        positions = np.asarray(list(index), dtype=np.int64)
        opens, highs, lows, closes, valid = _paths_for_symbol(
            store,
            str(symbol),
            candidates.loc[positions, "__decision_ts__"],
            horizon_hours=horizon_hours,
        )
        if valid.any():
            local = np.flatnonzero(valid)
            result = simulate_execution_ev_12h(
                opens[local],
                highs[local],
                lows[local],
                closes[local],
                candidates.loc[positions[local], "side_sign"].to_numpy(dtype=np.float64),
                candidates.loc[positions[local], "atr_fraction"].to_numpy(dtype=np.float64),
                candidates.loc[positions[local], "cost_return"].to_numpy(dtype=np.float64),
                long_geometry.vector(),
                short_geometry.vector(),
                60,
            )
            target = positions[local]
            gross[target], net[target], reason[target], exit_bar[target], mfe[target], mae[target] = result
        if number == 1 or number % 25 == 0 or number == len(groups):
            print(f"[execution-ev-labels] {number}/{len(groups)} {symbol}", flush=True)
    valid = np.isfinite(net) & (exit_bar >= 0) & (reason >= 0)
    result = candidates.loc[valid, [*IDENTITY, "__decision_ts__", "execution_label_end_utc"]].copy()
    result["execution_label_available_at"] = result["execution_label_end_utc"]
    result["execution_gross_ev_12h"] = gross[valid].astype(np.float32)
    result["execution_fee_return"] = candidates.loc[
        valid, "fee_return"
    ].to_numpy(dtype=np.float32)
    result["execution_spread_return"] = candidates.loc[
        valid, "spread_return"
    ].to_numpy(dtype=np.float32)
    result["execution_cost_return"] = candidates.loc[valid, "cost_return"].to_numpy(dtype=np.float32)
    result["execution_net_ev_12h"] = net[valid].astype(np.float32)
    result["execution_exit_reason"] = reason_names(reason[valid])
    result["execution_exit_hour"] = (exit_bar[valid] + 1).astype(np.int16)
    result["execution_mfe_return_12h"] = mfe[valid].astype(np.float32)
    result["execution_mae_return_12h"] = mae[valid].astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False, compression="zstd")
    cost_contract = str(candidates.attrs.get("cost_contract", ""))
    decomposed_cost = cost_contract == "explicit_fee_plus_full_p90_spread"
    effective_spread_column = candidates.attrs.get("spread_return_column")
    payload = {
        "schema": SCHEMA,
        "prediction_role": "execution_ev_12h_labels",
        "source_artifact_sha256": _sha256(output),
        "source": {"candidates": str(candidates_path), "sha256": _sha256(candidates_path)},
        "ohlcv": {"root": str(ohlcv_root), "timeframe": "1h"},
        "policy": {
            "manifest": str(policy_manifest_path),
            "sha256": _sha256(policy_manifest_path),
            "long_geometry": long_geometry.__dict__,
            "short_geometry": short_geometry.__dict__,
        },
        "timing": {
            "signal_timestamp": "__ts__",
            "decision_delay_hours": decision_delay_hours,
            "first_path_timestamp": "__decision_ts__",
            "horizon_hours": horizon_hours,
            "label_end": "__decision_ts__ + 12h",
            "label_resolution_available_at": "execution_label_available_at = execution_label_end_utc",
        },
        "accounting": {
            "gross_return": "side * (exit / executable decision-bar open - 1)",
            "cost": (
                "execution_fee_return plus execution_spread_return deducted "
                "exactly once after gross return"
            ),
            "fee": (
                "explicit round-trip fee deducted once"
                if decomposed_cost
                else "legacy all-in candidate cost; fee/spread not decomposed"
            ),
            "spread": (
                f"full per-symbol p90 spread from {effective_spread_column!r} deducted once"
                if decomposed_cost
                else "zero in decomposed output; included in legacy cost only if source supplied it"
            ),
            "cost_contract": cost_contract,
            "fee_round_trip_return": candidates.attrs.get(
                "fee_round_trip_return"
            ),
            "spread_return_column": candidates.attrs.get(
                "spread_return_column"
            ),
            "spread_map_csv": candidates.attrs.get("spread_map_csv"),
            "spread_map_sha256": candidates.attrs.get("spread_map_sha256"),
            "missing_spread_policy": candidates.attrs.get(
                "missing_spread_policy"
            ),
            "timeout": "exit at final available close, never treated as a full loss",
            "collision": "pessimistic stop before favorable excursion within each hourly candle",
        },
        "resolution_limitation": "training target uses canonical 1h OHLC because complete historical 1m paths do not exist for the full candidate universe; it is not claimed to be a 1m replay",
        "rows": {
            "source": source_rows,
            "valid_atr_input": n_rows,
            "invalid_atr_excluded": invalid_atr_rows,
            "missing_spread_excluded": int(
                candidates.attrs.get("missing_spread_rows_excluded", 0)
            ),
            "output": int(len(result)),
            "path_coverage_on_valid_atr": float(valid.mean()),
            "total_coverage": float(len(result) / max(source_rows, 1)),
        },
        "output": {"path": str(output), "sha256": _sha256(output)},
    }
    payload["prediction_role_manifest_sha256"] = _signed_manifest_hash(payload)
    manifest_path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"output": output, "manifest": manifest_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--ohlcv-root", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--decision-delay-hours", type=int, default=1)
    parser.add_argument("--horizon-hours", type=int, default=12)
    parser.add_argument(
        "--fee-round-trip-return",
        type=float,
        help=(
            "Explicit round-trip fee return. Required with --spread-return-col; "
            "0.003 means 0.30%%."
        ),
    )
    parser.add_argument(
        "--spread-return-col",
        help=(
            "Candidate column containing the full per-symbol spread as a return. "
            "When supplied, total cost is fee + spread exactly once."
        ),
    )
    parser.add_argument(
        "--spread-map-csv",
        type=Path,
        help=(
            "CSV with one pooled per-symbol p90 full spread. This avoids "
            "materializing a wide enriched candidate copy."
        ),
    )
    parser.add_argument("--spread-map-symbol-col", default="symbol")
    parser.add_argument("--spread-map-bps-col", default="p90_spread_bps")
    parser.add_argument(
        "--missing-spread-policy",
        choices=("reject", "drop"),
        default="reject",
        help=(
            "Reject missing spread mappings, or explicitly drop symbols outside "
            "the frozen eligible-spread universe."
        ),
    )
    args = parser.parse_args()
    if args.horizon_hours != 12:
        raise SystemExit("execution-EV target contract requires exactly 12 hours")
    paths = materialize(
        args.candidates,
        args.ohlcv_root,
        args.policy_manifest,
        args.output,
        args.manifest or args.output.with_suffix(".manifest.json"),
        decision_delay_hours=args.decision_delay_hours,
        horizon_hours=args.horizon_hours,
        fee_round_trip_return=args.fee_round_trip_return,
        spread_return_col=args.spread_return_col,
        spread_map_csv=args.spread_map_csv,
        spread_map_symbol_col=args.spread_map_symbol_col,
        spread_map_bps_col=args.spread_map_bps_col,
        missing_spread_policy=args.missing_spread_policy,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
