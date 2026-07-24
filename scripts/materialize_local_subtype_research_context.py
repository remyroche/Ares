#!/usr/bin/env python3
"""Materialize leakage-safe historical state context for local MLP subtypes.

The historical meta handoff contains side, archetype and resolved outcomes but
not the full modern market-state universe.  The feature store does.  This
script joins only decision-time state features onto the historical candidate
stream, preserving its original resolved outcomes exclusively for train-time
state selection.  The output is intended for fitting frozen AE/GMM subtype
encoders through a research cutoff, never for OOS transformation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


KEYS = ["__ts__", "__symbol__", "side_name", "archetype_policy_key"]
OUTCOME_COLUMNS = [
    "ev_after_1pct", "clean_exec", "dirty_positive", "full_path_bad_mae_1r",
    "timeout", "score_base",
]
STATE_TOKENS = (
    "mkt_", "market_", "xasset", "xs_", "oi_", "funding", "breadth",
    "shock", "entropy", "vol", "volume", "range", "recovery", "liquidation",
    "delever", "gmm", "aegmm", "mahal", "reconstruction", "cluster",
)
FORBIDDEN_TOKENS = ("target", "label", "future", "oracle", "realized_", "outcome")
STATE_FAMILIES = {
    "market": ("mkt_", "market_", "xasset", "xs_"),
    "oi": ("oi_", "open_interest", "leverage", "delever"),
    "funding": ("fund", "carry"),
    "breadth": ("breadth", "cross_asset", "corr", "dispersion"),
    "shock": ("shock", "climax", "liquidation", "flush"),
    "entropy": ("entropy", "complex", "chop"),
    "volatility": ("vol", "rv_", "atr", "range"),
    "recovery": ("recovery", "wick", "decel", "rebound"),
    "latent": ("aegmm", "gmm", "mahal", "reconstruction", "cluster"),
    "liquidity": ("volume", "spread", "orderbook", "ob_", "impact"),
}


def _state_columns(feature_dir: Path, ceiling: int) -> list[str]:
    sample = next(feature_dir.glob("symbol=*.parquet"), None)
    if sample is None:
        raise FileNotFoundError(f"no symbol parquet found in {feature_dir}")
    names = pq.ParquetFile(sample).schema.names
    eligible = [
        name for name in names
        if name != "ts"
        and any(token in name.lower() for token in STATE_TOKENS)
        and not any(token in name.lower() for token in FORBIDDEN_TOKENS)
        # Avoid model-version-specific historical base artifacts.  The frozen
        # subtype state must be computable from the same raw feature family at
        # current inference, not from an older base model's diagnostics.
        and not name.lower().startswith(("base_lgbm_", "meta_"))
    ]
    # Deterministic broad basket with family coverage. Economic relevance is
    # decided later per side/archetype through causal residual screening, not
    # by this registry-level prefilter.
    per_family = max(1, ceiling // len(STATE_FAMILIES))
    selected: list[str] = []
    for tokens in STATE_FAMILIES.values():
        selected.extend(sorted(name for name in eligible if any(token in name.lower() for token in tokens))[:per_family])
    selected = list(dict.fromkeys(selected))
    if len(selected) < ceiling:
        selected.extend(name for name in sorted(eligible) if name not in selected)
    return selected[:ceiling]


def _symbol_path(feature_dir: Path, symbol: str) -> Path:
    return feature_dir / f"symbol={symbol.replace('/', '_')}.parquet"


def _read_source(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    available = set(pq.ParquetFile(path).schema.names)
    score_column = next((name for name in ("score_base", "policy_parent_rank", "score") if name in available), None)
    needed = list(dict.fromkeys(
        column for column in [*KEYS, *OUTCOME_COLUMNS, score_column]
        if column is not None and column in available
    ))
    frame = pd.read_parquet(path, columns=needed)
    frame = frame.loc[:, ~frame.columns.duplicated(keep="first")]
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    frame = frame.loc[frame["__ts__"].between(start, end, inclusive="left")].copy()
    frame = frame.dropna(subset=["__ts__", "__symbol__", "side_name", "archetype_policy_key"])
    frame["side_name"] = frame["side_name"].astype(str)
    frame["archetype_policy_key"] = frame["archetype_policy_key"].astype(str)
    # Global timestamp rank is the historical analogue of the parent stream's
    # admission ordering. It is only used to focus state selection on the
    # candidate tail during research.
    if "policy_parent_rank" not in frame:
        if score_column is None:
            frame["policy_parent_rank"] = np.float32(0.5)
        else:
            frame["policy_parent_rank"] = frame.groupby("__ts__", observed=True)[score_column].rank(pct=True, method="average").astype(np.float32)
    return frame.sort_values("__ts__", kind="stable").reset_index(drop=True)


def _attach_state_features(frame: pd.DataFrame, feature_dir: Path, columns: list[str]) -> tuple[pd.DataFrame, dict[str, object]]:
    pieces: list[pd.DataFrame] = []
    coverage: dict[str, float] = {}
    for symbol, group in frame.groupby("__symbol__", observed=True, sort=False):
        path = _symbol_path(feature_dir, str(symbol))
        if not path.exists():
            piece = group.copy()
            for name in columns:
                piece[f"state__{name}"] = np.float32(np.nan)
            pieces.append(piece)
            continue
        available = set(pq.ParquetFile(path).schema.names)
        source_columns = ["ts", *[name for name in columns if name in available]]
        state = pd.read_parquet(path, columns=source_columns)
        state = state.reset_index() if "ts" not in state.columns else state
        state["ts"] = pd.to_datetime(state["ts"], utc=True, errors="coerce")
        state = state.rename(columns={"ts": "__ts__"})
        state = state.loc[state["__ts__"].between(group["__ts__"].min(), group["__ts__"].max(), inclusive="both")]
        state = state.drop_duplicates("__ts__", keep="last")
        state = state.rename(columns={name: f"state__{name}" for name in state.columns if name != "__ts__"})
        piece = group.merge(state, on="__ts__", how="left", validate="many_to_one", copy=False)
        pieces.append(piece)
    result = pd.concat(pieces, ignore_index=True)
    state_cols = [f"state__{name}" for name in columns]
    for column in state_cols:
        if column not in result:
            result[column] = np.float32(np.nan)
        result[column] = pd.to_numeric(result[column], errors="coerce").astype(np.float32)
        coverage[column] = float(result[column].notna().mean())
    return result, {"state_columns": state_cols, "coverage": coverage}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data_perp/artifacts/20260713_meta_fullhistory_old55_expandedpool/s52_train_meta_regime_handoff_smoke_predictions.parquet"))
    parser.add_argument("--feature-dir", type=Path, default=Path("data_perp/features/20260710_170000"))
    parser.add_argument("--start", default="2025-04-01")
    parser.add_argument("--end", default="2026-04-01")
    parser.add_argument("--state-feature-ceiling", type=int, default=240)
    parser.add_argument("--output", type=Path, default=Path("data_perp/reports/meta_v9_recovery_20260714/local_subtype_research_context_202504_202603.parquet"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    state_cols = _state_columns(args.feature_dir, args.state_feature_ceiling)
    if args.dry_run:
        print(json.dumps({"state_features": len(state_cols), "sample": state_cols[:20]}, indent=2))
        return
    source = _read_source(args.source, start, end)
    result, report = _attach_state_features(source, args.feature_dir, state_cols)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False, compression="zstd")
    manifest = {
        "source": str(args.source), "feature_dir": str(args.feature_dir),
        "research_start": str(start), "research_end_exclusive": str(end),
        "rows": int(len(result)), "symbols": int(result["__symbol__"].nunique()),
        "state_feature_ceiling": args.state_feature_ceiling,
        "state_feature_count": len(state_cols), **report,
        "leakage_contract": "state__ columns are decision-time feature-store values; resolved outcomes remain train-only research fields",
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"rows": len(result), "state_columns": len(report["state_columns"]), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
