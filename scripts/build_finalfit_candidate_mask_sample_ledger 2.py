#!/usr/bin/env python3
"""Build a final-fit sample ledger from strategy-mask candidate rows.

This differs from ``build_finalfit_broad_sample_ledger.py``: it does not emit
the full timestamp/symbol/head universe.  It reconstructs the four final-fit
strategy masks from the encoded strategy IDs, infers each comparison direction
from the persisted OOF candidate universe, and scans the feature store for
matching timestamp-symbol candidates.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


NUMERIC_TOKEN_RE = re.compile(r"-?\d+(?:_\d+)?")


@dataclass(frozen=True)
class StrategyCondition:
    feature: str
    threshold: float
    operator: str
    inferred_gt_share: float
    inferred_le_share: float
    inferred_rows: int


def _utc_timestamp(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    return pd.Timestamp(value).tz_convert("UTC") if pd.Timestamp(value).tzinfo else pd.Timestamp(value, tz="UTC")


def _symbol_from_path(path: Path) -> str:
    name = path.name
    if not (name.startswith("symbol=") and name.endswith(".parquet")):
        raise ValueError(f"Unexpected feature-store file name: {path}")
    return name[len("symbol=") : -len(".parquet")].replace("_", "/", 1)


def _symbol_to_path(feature_store: Path, symbol: str) -> Path:
    return feature_store / f"symbol={symbol.replace('/', '_', 1)}.parquet"


def _head_from_strategy(strategy_id: str) -> str:
    if strategy_id.startswith("long_bars_"):
        return "long_bars"
    if strategy_id.startswith("long_dist_"):
        return "long_dist"
    if strategy_id.startswith("short_asset_"):
        return "short_asset"
    if strategy_id.startswith("short_bollinger_"):
        return "short_boll"
    return strategy_id.split("_", 1)[0]


def _side_from_strategy(strategy_id: str) -> str:
    if strategy_id.startswith("long_"):
        return "long"
    if strategy_id.startswith("short_"):
        return "short"
    return ""


def _feature_schema(feature_store: Path) -> list[str]:
    first = next(iter(sorted(feature_store.glob("symbol=*.parquet"))), None)
    if first is None:
        raise FileNotFoundError(f"No symbol parquet files under {feature_store}")
    return list(pq.read_schema(first).names)


def _parse_strategy_conditions(strategy_id: str, feature_names: Iterable[str]) -> list[tuple[str, float]]:
    remaining = strategy_id
    if remaining.startswith("long_"):
        remaining = remaining[len("long_") :]
    elif remaining.startswith("short_"):
        remaining = remaining[len("short_") :]

    features_by_len = sorted(set(map(str, feature_names)), key=len, reverse=True)
    parsed: list[tuple[str, float]] = []
    while remaining:
        matched = False
        for feature in features_by_len:
            prefix = f"{feature}_"
            if not remaining.startswith(prefix):
                continue
            tail = remaining[len(prefix) :]
            token = NUMERIC_TOKEN_RE.match(tail)
            if token is None:
                continue
            threshold = float(token.group(0).replace("_", "."))
            parsed.append((feature, threshold))
            remaining = tail[token.end() :]
            if remaining.startswith("_"):
                remaining = remaining[1:]
            matched = True
            break
        if not matched:
            raise ValueError(
                f"Could not parse strategy condition from {strategy_id!r}; "
                f"stopped at {remaining[:96]!r}"
            )
    return parsed


def _load_strategy_ids(manifest_path: Path) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    strategy_ids = [str(x) for x in manifest.get("strategy_ids", []) if str(x)]
    if not strategy_ids:
        raise ValueError(f"No strategy_ids found in {manifest_path}")
    return strategy_ids


def _read_symbol_feature_slice(
    path: Path,
    columns: list[str],
    *,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=columns)
    if frame.empty:
        return frame
    frame.index = pd.to_datetime(frame.index, utc=True)
    if start is not None:
        frame = frame.loc[frame.index >= start]
    if end is not None:
        frame = frame.loc[frame.index <= end]
    return frame


def _sample_feature_values(
    *,
    rows: pd.DataFrame,
    feature_store: Path,
    columns: list[str],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    work = rows.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.dropna(subset=["timestamp", "symbol"])
    for symbol, group in work.groupby("symbol", sort=False):
        path = _symbol_to_path(feature_store, str(symbol))
        if not path.exists():
            continue
        ts = pd.DatetimeIndex(group["timestamp"])
        if ts.empty:
            continue
        frame = _read_symbol_feature_slice(path, columns, start=ts.min(), end=ts.max())
        if frame.empty:
            continue
        selected = frame.reindex(ts)
        parts.append(selected.reset_index(drop=True))
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True, copy=False)


def _infer_conditions(
    *,
    strategy_id: str,
    parsed_conditions: list[tuple[str, float]],
    feature_store: Path,
    oof_dir: Path,
    min_timestamp: pd.Timestamp | None,
    max_sample_rows: int,
) -> list[StrategyCondition]:
    oof_path = oof_dir / f"oof_{strategy_id}_H5.parquet"
    if not oof_path.exists():
        raise FileNotFoundError(f"Missing OOF candidate universe for {strategy_id}: {oof_path}")
    rows = pd.read_parquet(oof_path, columns=["timestamp", "symbol"])
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
    rows = rows.dropna(subset=["timestamp", "symbol"])
    if min_timestamp is not None:
        rows = rows.loc[rows["timestamp"] >= min_timestamp]
    if len(rows) > max_sample_rows:
        rows = rows.sample(max_sample_rows, random_state=42).sort_values("timestamp")
    columns = [feature for feature, _ in parsed_conditions]
    values = _sample_feature_values(rows=rows, feature_store=feature_store, columns=columns)
    inferred: list[StrategyCondition] = []
    for feature, threshold in parsed_conditions:
        x = pd.to_numeric(values.get(feature, pd.Series(dtype=float)), errors="coerce")
        x = x.replace([np.inf, -np.inf], np.nan).dropna()
        if x.empty:
            raise ValueError(f"No feature values available to infer {strategy_id} condition {feature}")
        gt_share = float((x > threshold).mean())
        le_share = float((x <= threshold).mean())
        operator = ">" if gt_share >= le_share else "<="
        inferred.append(
            StrategyCondition(
                feature=feature,
                threshold=float(threshold),
                operator=operator,
                inferred_gt_share=gt_share,
                inferred_le_share=le_share,
                inferred_rows=int(len(x)),
            )
        )
    return inferred


def _mask_for_conditions(frame: pd.DataFrame, conditions: list[StrategyCondition]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for condition in conditions:
        values = pd.to_numeric(frame[condition.feature], errors="coerce")
        if condition.operator == ">":
            mask &= values > condition.threshold
        elif condition.operator == "<=":
            mask &= values <= condition.threshold
        else:
            raise ValueError(f"Unsupported operator: {condition.operator}")
    return mask.fillna(False)


def _write_frame(frame: pd.DataFrame, output_path: Path, *, write_csv: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    if write_csv:
        frame.to_csv(output_path.with_suffix(".csv"), index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-store", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--oof-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--operator-inference-start", default="2025-12-25T00:00:00Z")
    parser.add_argument("--operator-inference-sample-rows", type=int, default=5000)
    parser.add_argument("--write-csv", action="store_true")
    args = parser.parse_args()

    start = _utc_timestamp(args.start)
    end = _utc_timestamp(args.end)
    inference_start = _utc_timestamp(args.operator_inference_start)
    if start is None or end is None:
        raise ValueError("--start and --end are required")
    if end < start:
        raise ValueError("--end must be >= --start")

    feature_names = _feature_schema(args.feature_store)
    strategy_ids = _load_strategy_ids(args.manifest)
    conditions_by_strategy: dict[str, list[StrategyCondition]] = {}
    for strategy_id in strategy_ids:
        parsed = _parse_strategy_conditions(strategy_id, feature_names)
        conditions_by_strategy[strategy_id] = _infer_conditions(
            strategy_id=strategy_id,
            parsed_conditions=parsed,
            feature_store=args.feature_store,
            oof_dir=args.oof_dir,
            min_timestamp=inference_start,
            max_sample_rows=max(1, int(args.operator_inference_sample_rows)),
        )

    rows: list[pd.DataFrame] = []
    symbol_files = sorted(args.feature_store.glob("symbol=*.parquet"))
    for strategy_id, conditions in conditions_by_strategy.items():
        needed = [condition.feature for condition in conditions]
        head = _head_from_strategy(strategy_id)
        side = _side_from_strategy(strategy_id)
        strategy_parts: list[pd.DataFrame] = []
        for i, path in enumerate(symbol_files, start=1):
            frame = _read_symbol_feature_slice(path, needed, start=start, end=end)
            if frame.empty:
                continue
            active = _mask_for_conditions(frame, conditions)
            if not bool(active.any()):
                continue
            timestamps = frame.index[active.to_numpy()]
            part = pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "symbol": _symbol_from_path(path),
                    "head": head,
                    "side": side,
                    "strategy_id": strategy_id,
                }
            )
            strategy_parts.append(part)
            if i == 1 or i % 50 == 0 or i == len(symbol_files):
                print(
                    f"{strategy_id}: scanned {i}/{len(symbol_files)} symbols, "
                    f"rows={sum(len(x) for x in strategy_parts)}",
                    flush=True,
                )
        if strategy_parts:
            strategy_frame = pd.concat(strategy_parts, ignore_index=True, copy=False)
        else:
            strategy_frame = pd.DataFrame(columns=["timestamp", "symbol", "head", "side", "strategy_id"])
        print(
            f"{strategy_id}: generated {len(strategy_frame)} candidate-mask sample rows",
            flush=True,
        )
        rows.append(strategy_frame)

    out = pd.concat(rows, ignore_index=True, copy=False) if rows else pd.DataFrame()
    if not out.empty:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
        out = out.sort_values(["timestamp", "head", "symbol"], kind="mergesort").reset_index(drop=True)
    _write_frame(out, args.output_path, write_csv=bool(args.write_csv))

    diagnostics_rows = []
    for strategy_id, conditions in conditions_by_strategy.items():
        for condition in conditions:
            row = asdict(condition)
            row["strategy_id"] = strategy_id
            row["head"] = _head_from_strategy(strategy_id)
            diagnostics_rows.append(row)
    diagnostics = pd.DataFrame(diagnostics_rows)
    diagnostics_path = args.output_path.with_name(args.output_path.stem + "_condition_diagnostics.parquet")
    _write_frame(diagnostics, diagnostics_path, write_csv=bool(args.write_csv))

    manifest = {
        "generated_by": "build_finalfit_candidate_mask_sample_ledger",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_store": str(args.feature_store),
        "finalfit_manifest": str(args.manifest),
        "oof_dir": str(args.oof_dir),
        "output_path": str(args.output_path),
        "condition_diagnostics_path": str(diagnostics_path),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "operator_inference_start": inference_start.isoformat() if inference_start is not None else None,
        "operator_inference_sample_rows": int(args.operator_inference_sample_rows),
        "feature_symbol_files": len(symbol_files),
        "rows": int(len(out)),
        "timestamp_min": out["timestamp"].min().isoformat() if not out.empty else None,
        "timestamp_max": out["timestamp"].max().isoformat() if not out.empty else None,
        "timestamp_count": int(out["timestamp"].nunique()) if not out.empty else 0,
        "symbol_count": int(out["symbol"].nunique()) if not out.empty else 0,
        "head_count": int(out["head"].nunique()) if not out.empty else 0,
        "rows_by_head": {str(k): int(v) for k, v in out["head"].value_counts().sort_index().items()}
        if not out.empty
        else {},
        "conditions_by_strategy": {
            strategy_id: [asdict(condition) for condition in conditions]
            for strategy_id, conditions in conditions_by_strategy.items()
        },
    }
    manifest_path = args.output_path.with_name(args.output_path.stem + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
