#!/usr/bin/env python3
"""Materialize causal, deterministic economic path-archetype training targets."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.path_archetype_labels import (  # noqa: E402
    ATR_REALIZATION_THRESHOLDS,
    CATBOOST_ARCHETYPE_ATR_FLOOR,
    CATBOOST_ARCHETYPE_COST_RETURN,
    CATBOOST_ARCHETYPE_NET_MARGIN_ATR,
    PATH_ARCHETYPE_RULE_VERSION,
    PathArchetypeLabelConfig,
    materialize_path_archetypes,
    path_archetype_support_table,
)
from extreme_price_movements.path_auxiliary_targets import (  # noqa: E402
    MIN_USABLE_MFE_ATR,
    MIN_USABLE_MFE_RETURN,
)

SCHEMA_VERSION = "materialize_path_archetype_labels_v7_costaware_geometry_search_dense12h"


def _json_safe(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, pd.Timedelta, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _iter_parquet(
    path: Path, *, batch_rows: int, columns: Sequence[str] | None = None
) -> Iterator[pd.DataFrame]:
    """Read parquet in bounded row batches, retaining source columns unchanged."""
    try:
        import pyarrow.parquet as pq
    except (
        ImportError
    ):  # pragma: no cover - project environments normally have pyarrow.
        yield pd.read_parquet(path, columns=list(columns) if columns else None)
        return
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=int(batch_rows), columns=list(columns) if columns else None
    ):
        yield batch.to_pandas()


def _candidate_columns(path: Path) -> list[str]:
    """Return the narrow target-materialization projection present in parquet."""
    try:
        import pyarrow.parquet as pq

        available = set(pq.ParquetFile(path).schema.names)
    except ImportError:  # pragma: no cover
        available = set(pd.read_parquet(path).columns)
    requested = (
        "__ts__",
        "__symbol__",
        "side",
        "side_name",
        "__side__",
        "entry_price",
        "__entry_price__",
        "stop_price",
        "__stop_price__",
        "risk_distance",
        "__risk_distance__",
        "__barrier_pct__",
        "barrier_pct",
        "atr_fraction",
        "__atr_fraction__",
        "__path_auxiliary_atr_fraction__",
        "atr_pct",
        "atr_pct_base",
        "take_profit_price",
        "tp_price",
        "take_profit_r",
        "tp_r",
        "trailing_activation_price",
        "trail_activation_price",
        "trailing_trigger_r",
        "trailing_activation_r",
        "stop_r",
        "sl_r",
        "path_cost_return",
        "round_trip_cost_return",
        "execution_cost_return",
        "cost_return",
        "activation_distance_return",
        "trailing_activation_distance_return",
        "candidate_id",
    )
    columns = [column for column in requested if column in available]
    required = {"__ts__", "__symbol__"}
    if not required.issubset(columns):
        raise ValueError(f"candidate parquet is missing {sorted(required - set(columns))}")
    if not any(column in columns for column in ("side", "side_name", "__side__")):
        raise ValueError("candidate parquet needs side, side_name, or __side__")
    return columns


def _read_canonical_bars(path: Path, *, batch_rows: int) -> pd.DataFrame:
    """Chunk-read only OHLC keys/columns and downcast prices before joining."""
    parts: list[pd.DataFrame] = []
    for part in _iter_parquet(path, batch_rows=batch_rows):
        timestamp = (
            "timestamp" if "timestamp" in part else "ts" if "ts" in part else None
        )
        symbol = (
            "symbol"
            if "symbol" in part
            else "__symbol__"
            if "__symbol__" in part
            else None
        )
        if timestamp is None or symbol is None:
            raise ValueError(
                "canonical bars need timestamp/ts and symbol/__symbol__ columns"
            )
        missing = {"high", "low", "close"}.difference(part.columns)
        if missing:
            raise ValueError(f"canonical bars missing {sorted(missing)}")
        columns = [timestamp, symbol, "high", "low", "close"]
        if "open" in part:
            columns.append("open")
        small = part.loc[:, columns].rename(
            columns={timestamp: "timestamp", symbol: "symbol"}
        )
        for column in ("high", "low", "close", "open"):
            if column not in small:
                continue
            small[column] = pd.to_numeric(small[column], errors="coerce").astype(
                np.float32
            )
        parts.append(small)
    if not parts:
        return pd.DataFrame(columns=["timestamp", "symbol", "high", "low", "close"])
    return pd.concat(parts, ignore_index=True)


def _read_partitioned_ohlcv(
    root: Path, candidates: pd.DataFrame, *, decision_delay_hours: int
) -> pd.DataFrame:
    """Read only required symbols/time span through the canonical OHLCV store."""
    from extreme_price_movements.data_store import PartitionedOHLCVStore

    timestamps = pd.to_datetime(candidates["__ts__"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("candidate parquet contains invalid UTC timestamps")
    start = timestamps.min() + pd.Timedelta(hours=int(decision_delay_hours))
    end = timestamps.max() + pd.Timedelta(hours=int(decision_delay_hours) + 24)
    store = PartitionedOHLCVStore(root_dir=str(root), timeframe="1h")
    parts: list[pd.DataFrame] = []
    for symbol in candidates["__symbol__"].dropna().astype(str).unique():
        frame = store.load(
            symbol,
            columns=["open", "high", "low", "close"],
            start_ts=start,
            end_ts=end,
        )
        if frame.empty:
            continue
        small = frame.reset_index().rename(columns={frame.index.name or "index": "timestamp"})
        if "timestamp" not in small:
            small = small.rename(columns={small.columns[0]: "timestamp"})
        small["symbol"] = symbol
        parts.append(small.loc[:, ["timestamp", "symbol", "open", "high", "low", "close"]])
    if not parts:
        raise ValueError(f"no canonical OHLCV rows loaded from {root}")
    return pd.concat(parts, ignore_index=True)


def materialize(
    candidates_path: Path,
    canonical_bars_path: Path | None,
    output_dir: Path,
    *,
    ohlcv_root: Path | None = None,
    batch_rows: int = 100_000,
    decision_delay_hours: int = 1,
    bar_hours: float = 1.0,
    default_cost_return: float = CATBOOST_ARCHETYPE_COST_RETURN,
    default_activation_r: float = 1.0,
) -> dict[str, Any]:
    """Write materialized rows, deterministic support table, and UTC manifest."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if (canonical_bars_path is None) == (ohlcv_root is None):
        raise ValueError("provide exactly one of canonical_bars_path or ohlcv_root")
    columns = _candidate_columns(candidates_path)
    candidates = pd.concat(
        list(_iter_parquet(candidates_path, batch_rows=batch_rows, columns=columns)),
        ignore_index=True,
    )
    bars = (
        _read_canonical_bars(canonical_bars_path, batch_rows=batch_rows)
        if canonical_bars_path is not None
        else _read_partitioned_ohlcv(
            Path(ohlcv_root), candidates, decision_delay_hours=decision_delay_hours
        )
    )
    config = PathArchetypeLabelConfig(
        decision_delay_hours=decision_delay_hours,
        bar_hours=bar_hours,
        default_cost_return=default_cost_return,
        default_activation_r=default_activation_r,
    )
    if candidates.empty:
        raise ValueError("candidate parquet contains no rows")
    result = materialize_path_archetypes(candidates, bars, config=config)
    output_path = output_dir / "path_archetype_labels.parquet"
    result.to_parquet(output_path, index=False, compression="zstd")
    support = path_archetype_support_table(result)
    support_path = output_dir / "path_archetype_support_summary.csv"
    support.to_csv(support_path, index=False)
    try:
        code_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        code_revision = "unknown"
    source_start = pd.to_datetime(result["__ts__"], utc=True, errors="coerce").min()
    source_end = pd.to_datetime(result["__ts__"], utc=True, errors="coerce").max()
    manifest = {
        "schema": SCHEMA_VERSION,
        "candidate_source": str(candidates_path),
        "canonical_bars_source": (
            str(canonical_bars_path)
            if canonical_bars_path is not None
            else f"PartitionedOHLCVStore:{ohlcv_root}"
        ),
        "output": str(output_path),
        "utc_key": ["__ts__", "__symbol__", "side"],
        "decision_timestamp_column": "__decision_ts__",
        "label_end_timestamp_column": "__label_end_ts__",
        "decision_delay_hours": decision_delay_hours,
        "bar_hours": bar_hours,
        "horizons_hours": list(config.horizons_hours),
        "rule_version": PATH_ARCHETYPE_RULE_VERSION,
        "usable_mfe_floor": {
            "atr_multiple": float(MIN_USABLE_MFE_ATR),
            "minimum_return": float(MIN_USABLE_MFE_RETURN),
            "contract": "max(atr_multiple * atr_fraction, minimum_return)",
        },
        "realization_strength": {
            "atr_thresholds": list(ATR_REALIZATION_THRESHOLDS),
            "peak_mfe_atr_cap": 10.0,
            "cost_return_fallback": float(default_cost_return),
            "activation_r_fallback": float(default_activation_r),
            "ratio_numerator": "raw non-negative peak MFE return for every complete path",
        },
        "geometry_search_support": {
            "fixed_total_cost_return": float(CATBOOST_ARCHETYPE_COST_RETURN),
            "default_atr_floor": float(CATBOOST_ARCHETYPE_ATR_FLOOR),
            "default_net_margin_atr": float(CATBOOST_ARCHETYPE_NET_MARGIN_ATR),
            "meaningful_mfe_contract": (
                "max(atr_floor, fixed_total_cost_return / atr_fraction "
                "+ net_margin_atr)"
            ),
            "dense_path_primitives": {
                "hours": list(range(1, 13)),
                "families": [
                    "raw_mfe_r",
                    "raw_mfe_atr",
                    "raw_mae_r",
                    "close_return_r",
                    "cumulative_variation_r",
                ],
            },
            "support_labels_are_realized_targets_only": True,
        },
        "deterministic_target_column": "path_archetype",
        "deterministic_shape_column": "path_shape_archetype",
        "realization_strength_column": "path_realization_strength",
        "discovery_cluster_id": "nullable diagnostic-only field; never used by rules",
        "no_preentry_leakage": "path fields and path_archetype are realised targets only",
        "rows": int(len(result)),
        "columns": int(len(result.columns)),
        "run_id": output_dir.name,
        "source_period_utc": {
            "start": source_start.isoformat() if pd.notna(source_start) else None,
            "end": source_end.isoformat() if pd.notna(source_end) else None,
        },
        "bar_frequency": f"{bar_hours:g}h",
        "feature_contract": "realized_path_targets_only; excluded from pre-entry model inputs",
        "universe_contract": "exact candidate identities supplied by candidate_source",
        "code_revision": code_revision,
        "complete_24h_rows": int(result["path_arch_complete_24h"].sum()),
        "type_support": support.to_dict(orient="records"),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--canonical-bars", type=Path)
    sources.add_argument("--ohlcv-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=100_000)
    parser.add_argument("--decision-delay-hours", type=int, default=1)
    parser.add_argument("--bar-hours", type=float, default=1.0)
    parser.add_argument(
        "--default-cost-return",
        type=float,
        default=CATBOOST_ARCHETYPE_COST_RETURN,
    )
    parser.add_argument("--default-activation-r", type=float, default=1.0)
    args = parser.parse_args()
    manifest = materialize(
        args.candidates,
        args.canonical_bars,
        args.output_dir,
        ohlcv_root=args.ohlcv_root,
        batch_rows=args.batch_rows,
        decision_delay_hours=args.decision_delay_hours,
        bar_hours=args.bar_hours,
        default_cost_return=args.default_cost_return,
        default_activation_r=args.default_activation_r,
    )
    print(json.dumps(_json_safe(manifest), sort_keys=True))


if __name__ == "__main__":
    main()
