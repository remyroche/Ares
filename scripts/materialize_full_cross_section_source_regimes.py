#!/usr/bin/env python3
"""Materialize causal source-regime scores from a full static universe.

Candidate ledgers are intentionally sparse.  Their rows cannot be used as the
cross-section for ``__regime_source_*`` features because several source scores
contain timestamp-wise percentile ranks.  This utility reads the same static
feature-store view used by training and inference, computes source scores on
the complete available universe, then attaches only the requested source
columns to the candidate rows.

Outcome columns are never read or materialized.  A short causal history is
included for the source run-entry/late-run calculation, but only rows at the
requested candidate timestamps are written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.live_meta_feature_overlays import (  # noqa: E402
    SOURCE_REGIME_PREFIX,
    SOURCE_REGIME_SUFFIX,
    materialize_live_source_regime_features,
)
from extreme_price_movements.static_feature_store import read_static_features  # noqa: E402
from scripts.materialize_candidate_source_tags import DEFAULT_CONFIG, load_config  # noqa: E402


DEFAULT_MIN_SOURCE_REGIME_SYMBOLS = 32
DEFAULT_HISTORY_HOURS = 6
DEFAULT_SYMBOL_BATCH_SIZE = 24


def _is_market_wide_feature(column: str) -> bool:
    """Return whether a feature must be aggregated before candidate joining."""

    name = str(column)
    return name.startswith("mkt_") or name.startswith("market_")


def _utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _store_timestamp(features_dir: Path) -> pd.Timestamp:
    """Parse the canonical ``YYYYMMDD_HHMMSS`` store identifier."""

    try:
        return pd.to_datetime(features_dir.name, format="%Y%m%d_%H%M%S", utc=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "--features-dir must be a canonical static feature-store directory "
            "named YYYYMMDD_HHMMSS"
        ) from exc


def configured_causal_source_columns(config_path: Path = DEFAULT_CONFIG) -> list[str]:
    """Return only raw, configured source inputs; never proxy/outcome columns."""

    config = load_config(config_path)
    groups = config.get("allowed_causal_feature_groups") or {}
    return sorted(
        {
            str(column)
            for columns in groups.values()
            for column in (columns or [])
            if str(column)
        }
    )


def required_source_columns(state: dict[str, Any]) -> list[str]:
    return sorted(
        {
            str(column)
            for column in state.get("feature_columns", [])
            if str(column).startswith(SOURCE_REGIME_PREFIX)
            and str(column).endswith(SOURCE_REGIME_SUFFIX)
        }
    )


def _discover_static_symbols(features_dir: Path) -> list[str]:
    """Discover static-store keys without constructing a feature matrix.

    ``read_static_features(symbols=None)`` still creates raw buffers for every
    symbol before it can report the universe.  That defeats bounded reads on a
    long historical range.  Feature-file names are store metadata, not model
    values, so scanning them is safe; all actual values remain read through
    ``read_static_features`` below.
    """

    symbols: set[str] = set()
    for path in features_dir.glob("symbol=*.parquet"):
        raw = path.stem.removeprefix("symbol=")
        if raw:
            symbols.add(raw.replace("_", "/"))
    if not symbols:
        raise RuntimeError(f"No static symbol parquet files found in {features_dir}")
    return sorted(symbols)


def full_static_source_panel(
    *,
    data_root: Path,
    features_dir: Path,
    source_columns: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    symbol_batch_size: int = DEFAULT_SYMBOL_BATCH_SIZE,
) -> pd.DataFrame:
    """Read a compact symbol x time x source-input frame through the shared API."""

    symbols = _discover_static_symbols(features_dir)
    parts: list[pd.DataFrame] = []
    batch_size = max(1, int(symbol_batch_size))
    for offset in range(0, len(symbols), batch_size):
        batch = symbols[offset : offset + batch_size]
        view = read_static_features(
            feature_store_ts=_store_timestamp(features_dir),
            data_root=data_root,
            feature_keys=source_columns,
            symbols=batch,
            start_ts=start,
            end_ts=end,
        )
        if view is None:
            continue
        for symbol in batch:
            frame = view.symbol_frame(symbol, keys=source_columns)
            if frame.empty:
                continue
            frame = frame.loc[(frame.index >= start) & (frame.index <= end)].copy()
            if frame.empty:
                continue
            frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
            frame = frame.loc[frame.index.notna()]
            frame["__ts__"] = frame.index
            frame["__symbol__"] = str(symbol)
            parts.append(frame.reset_index(drop=True))
    if not parts:
        raise RuntimeError("Static feature reader yielded no source rows")
    return pd.concat(parts, ignore_index=True, copy=False)


def materialize_source_regimes(
    full_panel: pd.DataFrame,
    *,
    required_columns: list[str],
    min_timestamp_symbols: int = DEFAULT_MIN_SOURCE_REGIME_SYMBOLS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score a full universe and return exactly the frozen contract outputs."""

    if full_panel.empty:
        return full_panel.copy(), {"rows": 0, "timestamps": 0, "source_columns": 0}
    if "__ts__" not in full_panel.columns or "__symbol__" not in full_panel.columns:
        raise ValueError("full source panel requires __ts__ and __symbol__")
    work = full_panel.copy()
    work["__ts__"] = pd.to_datetime(work["__ts__"], utc=True, errors="coerce")
    work = work.dropna(subset=["__ts__", "__symbol__"])
    support = work.groupby("__ts__", observed=True)["__symbol__"].nunique()
    undersupported = support.loc[support < int(min_timestamp_symbols)]
    if len(undersupported):
        sample = ", ".join(
            f"{timestamp.isoformat()}:{count}" for timestamp, count in undersupported.head(5).items()
        )
        raise ValueError(
            "Full source panel lacks adequate cross-sectional coverage: "
            f"{len(undersupported)}/{len(support)} timestamps below "
            f"{int(min_timestamp_symbols)} symbols ({sample})"
        )
    enriched = materialize_live_source_regime_features(
        work.sort_values(["__symbol__", "__ts__"], kind="stable").reset_index(drop=True),
        side="long",
        signal_bar_ts=None,
        required_columns=required_columns,
        overwrite_existing=True,
    )
    source_columns = [column for column in required_columns if column in enriched.columns]
    return enriched.loc[:, ["__ts__", "__symbol__", *source_columns]], {
        "rows": int(len(enriched)),
        "timestamps": int(enriched["__ts__"].nunique()),
        "symbols": int(enriched["__symbol__"].nunique()),
        "min_symbols_per_timestamp": int(support.min()),
        "median_symbols_per_timestamp": float(support.median()),
        "max_symbols_per_timestamp": int(support.max()),
        "source_columns": int(len(source_columns)),
    }


def materialize_full_universe_passthrough(
    full_panel: pd.DataFrame,
    *,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Materialize causal phase inputs without candidate-universe distortion.

    Market-wide fields are stored redundantly on the symbol panels.  Collapse
    them to a cross-sectional median before joining candidates, both to avoid
    picking an arbitrary source symbol and to keep the result invariant to the
    candidate population.  Asset-level phase fields remain symbol-specific.
    """

    requested = [column for column in feature_columns if column in full_panel.columns]
    if not requested:
        return full_panel.loc[:, ["__ts__", "__symbol__"]].drop_duplicates().copy()
    asset_columns = [column for column in requested if not _is_market_wide_feature(column)]
    market_columns = [column for column in requested if _is_market_wide_feature(column)]
    output = full_panel.loc[:, ["__ts__", "__symbol__", *asset_columns]].copy()
    if market_columns:
        market = (
            full_panel.loc[:, ["__ts__", *market_columns]]
            .groupby("__ts__", observed=True, as_index=False)
            .median(numeric_only=True)
        )
        output = output.merge(market, on="__ts__", how="left", validate="many_to_one")
    return output.drop_duplicates(["__ts__", "__symbol__"], keep="last")


def retain_candidate_fraction(
    candidates: pd.DataFrame,
    *,
    fraction: float,
) -> pd.DataFrame:
    """Retain a timestamp top fraction without losing its full-stream rank.

    Source-regime materialization can safely trim a wide candidate ledger after
    calculating the full cross-sectional source state.  Downstream meta-state
    training still needs the *original* base score percentile: recomputing it
    on a top-20-only ledger would make every retained row appear high ranked.
    """

    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("candidate_top_fraction must be in (0, 1]")
    if "score" not in candidates.columns:
        raise ValueError("--candidate-top-fraction requires a score column")
    ranked = candidates.copy()
    full_rank = ranked.groupby(
        "__ts__", observed=True
    )["score"].rank(method="average", pct=True)
    ranked["base_rank_pct_by_timestamp"] = full_rank.astype(np.float32)
    return ranked.loc[
        full_rank.gt(1.0 - float(fraction))
    ].copy()


def materialize_candidate_source_regimes(
    *,
    candidates_path: Path,
    features_dir: Path,
    ae_gmm_state_path: Path,
    out_path: Path,
    data_root: Path,
    history_hours: int = DEFAULT_HISTORY_HOURS,
    min_timestamp_symbols: int = DEFAULT_MIN_SOURCE_REGIME_SYMBOLS,
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
    symbol_batch_size: int = DEFAULT_SYMBOL_BATCH_SIZE,
    candidate_top_fraction: float = 0.0,
    passthrough_columns: list[str] | None = None,
) -> dict[str, Any]:
    from extreme_price_movements.features_gmm_ae import load_ae_gmm_state_artifact

    candidates = pd.read_parquet(candidates_path).copy()
    timestamp_column = "timestamp" if "timestamp" in candidates.columns else "__ts__"
    symbol_column = "symbol" if "symbol" in candidates.columns else "__symbol__"
    candidates["__ts__"] = pd.to_datetime(candidates[timestamp_column], utc=True, errors="coerce")
    candidates["__symbol__"] = candidates[symbol_column].astype(str)
    candidates = candidates.dropna(subset=["__ts__", "__symbol__"])
    if start_ts is not None:
        candidates = candidates.loc[candidates["__ts__"] >= _utc(start_ts)].copy()
    if end_ts is not None:
        candidates = candidates.loc[candidates["__ts__"] <= _utc(end_ts)].copy()
    if candidates.empty:
        raise ValueError("No candidate rows remain after requested date bounds")
    original_candidate_rows = int(len(candidates))
    if float(candidate_top_fraction) > 0.0:
        candidates = retain_candidate_fraction(
            candidates,
            fraction=float(candidate_top_fraction),
        )
        if candidates.empty:
            raise ValueError("No candidate rows remain after top-fraction filter")
    state = load_ae_gmm_state_artifact(ae_gmm_state_path)
    requested = required_source_columns(state)
    if not requested:
        raise ValueError("Frozen AE/GMM state does not request __regime_source_* features")

    passthrough_columns = list(dict.fromkeys(map(str, passthrough_columns or [])))
    source_columns = list(
        dict.fromkeys([*configured_causal_source_columns(), *passthrough_columns])
    )
    candidates["__month__"] = candidates["__ts__"].dt.to_period("M").astype(str)
    outputs: list[pd.DataFrame] = []
    month_reports: dict[str, Any] = {}
    for month, candidate_month in candidates.groupby("__month__", sort=True, observed=True):
        target_start = _utc(candidate_month["__ts__"].min())
        target_end = _utc(candidate_month["__ts__"].max())
        panel = full_static_source_panel(
            data_root=data_root,
            features_dir=features_dir,
            source_columns=source_columns,
            start=target_start - pd.Timedelta(hours=int(history_hours)),
            end=target_end,
            symbol_batch_size=int(symbol_batch_size),
        )
        source, report = materialize_source_regimes(
            panel,
            required_columns=requested,
            min_timestamp_symbols=int(min_timestamp_symbols),
        )
        passthrough = materialize_full_universe_passthrough(
            panel,
            feature_columns=passthrough_columns,
        )
        source = source.merge(
            passthrough,
            on=["__ts__", "__symbol__"],
            how="left",
            validate="one_to_one",
        )
        source = source.loc[(source["__ts__"] >= target_start) & (source["__ts__"] <= target_end)]
        merged = candidate_month.merge(
            source,
            on=["__ts__", "__symbol__"],
            how="left",
            validate="many_to_one",
            sort=False,
        )
        coverage = merged[requested].notna().all(axis=1) if requested else pd.Series(True, index=merged.index)
        report["candidate_rows"] = int(len(merged))
        report["candidate_source_complete_rate"] = float(coverage.mean()) if len(coverage) else float("nan")
        report["passthrough_columns_requested"] = passthrough_columns
        report["passthrough_columns_present"] = [
            column for column in passthrough_columns if column in merged.columns
        ]
        report["passthrough_complete_rate"] = (
            float(merged[passthrough_columns].notna().all(axis=1).mean())
            if passthrough_columns and all(column in merged.columns for column in passthrough_columns)
            else None
        )
        month_reports[str(month)] = report
        outputs.append(merged)

    out = pd.concat(outputs, ignore_index=True, copy=False)
    out = out.drop(columns=["__month__"], errors="ignore")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    manifest = {
        "generated_by": "materialize_full_cross_section_source_regimes",
        "candidates": str(candidates_path),
        "features_dir": str(features_dir),
        "ae_gmm_state_path": str(ae_gmm_state_path),
        "source_input_columns": source_columns,
        "requested_source_columns": requested,
        "passthrough_columns": passthrough_columns,
        "history_hours": int(history_hours),
        "min_timestamp_symbols": int(min_timestamp_symbols),
        "symbol_batch_size": int(symbol_batch_size),
        "input_rows": int(len(candidates)),
        "input_rows_before_candidate_filter": original_candidate_rows,
        "candidate_top_fraction": float(candidate_top_fraction),
        "output_rows": int(len(out)),
        "month_reports": month_reports,
    }
    out_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--ae-gmm-state", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data_perp")
    parser.add_argument("--history-hours", type=int, default=DEFAULT_HISTORY_HOURS)
    parser.add_argument("--min-source-regime-symbols", type=int, default=DEFAULT_MIN_SOURCE_REGIME_SYMBOLS)
    parser.add_argument("--symbol-batch-size", type=int, default=DEFAULT_SYMBOL_BATCH_SIZE)
    parser.add_argument(
        "--candidate-top-fraction",
        type=float,
        default=0.0,
        help="Optional global per-timestamp source-candidate fraction to retain after full-universe scoring.",
    )
    parser.add_argument(
        "--passthrough-feature",
        action="append",
        default=[],
        help=(
            "Repeatable causal static feature carried into the candidate output. "
            "Market-wide names are cross-sectionally median-aggregated."
        ),
    )
    parser.add_argument("--start-ts", type=str, default=None, help="Inclusive UTC bound for a materialization slice.")
    parser.add_argument("--end-ts", type=str, default=None, help="Inclusive UTC bound for a materialization slice.")
    args = parser.parse_args()
    result = materialize_candidate_source_regimes(
        candidates_path=args.candidates,
        features_dir=args.features_dir,
        ae_gmm_state_path=args.ae_gmm_state,
        out_path=args.out,
        data_root=args.data_root,
        history_hours=args.history_hours,
        min_timestamp_symbols=args.min_source_regime_symbols,
        start_ts=pd.Timestamp(args.start_ts) if args.start_ts else None,
        end_ts=pd.Timestamp(args.end_ts) if args.end_ts else None,
        symbol_batch_size=args.symbol_batch_size,
        candidate_top_fraction=args.candidate_top_fraction,
        passthrough_columns=args.passthrough_feature,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
