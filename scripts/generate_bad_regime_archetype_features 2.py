#!/usr/bin/env python3
"""Generate causal bad-regime archetype scores from a feature-store snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from extreme_price_movements.unsupervised_regime_learning.bad_regime_archetypes import (
    BadRegimeArchetypeFeatureConfig,
    build_bad_regime_archetype_feature_frame,
    load_bad_regime_archetype_definitions,
)


def _feature_aliases(name: str) -> list[str]:
    raw = str(name)
    aliases = [raw]
    for prefix in ("export__", "pred_H5_", "base_H5_", "oof_"):
        if raw.startswith(prefix):
            aliases.append(raw[len(prefix) :])
    if "_H5_" in raw:
        aliases.append(raw.split("_H5_", 1)[1])
    if raw.startswith("pred_") and "_" in raw:
        aliases.append(raw.rsplit("_", 1)[-1])
    return list(dict.fromkeys(alias for alias in aliases if alias))


def _requested_features(definitions: dict[str, dict[str, object]]) -> list[str]:
    out: list[str] = []
    for payload in definitions.values():
        for feature in payload.get("top_features", []) or []:
            out.extend(_feature_aliases(str(feature)))
    return list(dict.fromkeys(out))


def _parquet_columns(path: Path) -> list[str]:
    return list(pq.ParquetFile(path).schema.names)


def _symbol_from_path(path: Path) -> str:
    return path.name.removeprefix("symbol=").removesuffix(".parquet").replace("_", "/", 1)


def _read_symbol_frame(path: Path, requested: Iterable[str], *, max_rows: int = 0) -> pd.DataFrame:
    schema_cols = set(_parquet_columns(path))
    present = [col for col in requested if col in schema_cols]
    key_cols = ["ts"] if "ts" in schema_cols else ["timestamp"] if "timestamp" in schema_cols else []
    if not key_cols:
        # Feature snapshots usually store time in the parquet index, so this
        # fallback is still useful for older artifacts.
        read_cols = present
    else:
        read_cols = [*key_cols, *present]
    if not read_cols:
        return pd.DataFrame()
    table = pq.read_table(path, columns=read_cols)
    frame = table.to_pandas()
    if key_cols and key_cols[0] in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame.pop(key_cols[0]), utc=True, errors="coerce")
    else:
        frame["timestamp"] = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame["symbol"] = _symbol_from_path(path)
    if max_rows > 0 and len(frame) > max_rows:
        frame = frame.tail(int(max_rows)).copy()
    return frame[["timestamp", "symbol", *present]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", default="data_perp/features/20260605_070000")
    parser.add_argument(
        "--definitions",
        default="data_perp/reports/meta_recent_failure_diagnostics_20260622_archetype_synthesis_clean_contract_v1/soft_archetype_definitions.json",
    )
    parser.add_argument(
        "--output-parquet",
        default="data_perp/reports/meta_recent_failure_diagnostics_20260622_archetype_synthesis_clean_contract_v1/bad_regime_archetype_features_smoke.parquet",
    )
    parser.add_argument("--diagnostics-json", default="")
    parser.add_argument("--symbols", nargs="*", default=[])
    parser.add_argument("--max-symbols", type=int, default=20)
    parser.add_argument("--max-rows-per-symbol", type=int, default=3000)
    parser.add_argument("--trailing-window", type=int, default=24 * 28)
    parser.add_argument("--min-periods", type=int, default=24 * 7)
    parser.add_argument("--min-resolved-features", type=int, default=2)
    parser.add_argument("--clip-z", type=float, default=6.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feature_dir = Path(args.feature_dir)
    definitions = load_bad_regime_archetype_definitions(args.definitions)
    requested = _requested_features(definitions)
    if not definitions:
        raise SystemExit("No archetype definitions loaded")
    if not requested:
        raise SystemExit("No requested features resolved from definitions")

    paths = sorted(feature_dir.glob("symbol=*.parquet"))
    if args.symbols:
        wanted = {str(symbol).replace("/", "_") for symbol in args.symbols}
        paths = [path for path in paths if path.name.removeprefix("symbol=").removesuffix(".parquet") in wanted]
    if int(args.max_symbols) > 0:
        paths = paths[: int(args.max_symbols)]
    frames: list[pd.DataFrame] = []
    availability: dict[str, int] = {feature: 0 for feature in requested}
    for path in paths:
        frame = _read_symbol_frame(path, requested, max_rows=int(args.max_rows_per_symbol))
        if frame.empty:
            continue
        for col in frame.columns:
            if col not in {"timestamp", "symbol"}:
                availability[col] = availability.get(col, 0) + 1
                frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(np.float32, copy=False)
        frames.append(frame)
    if not frames:
        raise SystemExit("No symbol frames could be read")
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)
    config = BadRegimeArchetypeFeatureConfig(
        trailing_window=int(args.trailing_window),
        min_periods=int(args.min_periods),
        min_resolved_features=int(args.min_resolved_features),
        clip_z=float(args.clip_z),
    )
    features, diagnostics = build_bad_regime_archetype_feature_frame(panel, definitions, config=config)
    out = pd.concat([panel[["timestamp", "symbol"]].reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    output_path = Path(args.output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False, compression="zstd")
    diagnostics.update(
        {
            "feature_dir": str(feature_dir),
            "definition_path": str(args.definitions),
            "symbols_read": int(len(frames)),
            "rows": int(len(panel)),
            "requested_feature_alias_count": int(len(requested)),
            "available_requested_features": {k: v for k, v in availability.items() if v > 0},
            "output_parquet": str(output_path),
        }
    )
    diagnostics_path = Path(args.diagnostics_json) if args.diagnostics_json else output_path.with_suffix(".diagnostics.json")
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True, default=str))
    print(f"Wrote {features.shape[1]} archetype features for {len(panel)} rows to {output_path}")
    print(f"Diagnostics: {diagnostics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
