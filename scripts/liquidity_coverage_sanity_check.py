#!/usr/bin/env python3
"""Liquidity-specific specialist coverage sanity check.

This script inspects the labeled meta-labeling dataset for a given
symbol/exchange/timeframe/direction and reports per-specialist non-NaN
coverage over a recent lookback window (default: 365 days).

It is intended as an independent sanity check confirming that Liquidity
specialist features have comparable coverage to other specialists on the
effective training index used by meta-labeling.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import sys

import pandas as pd

# Ensure repository root is on sys.path so that 'src' can be imported when this
# script is executed directly (e.g., `python scripts/liquidity_coverage_sanity_check.py`).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.versioned_artifacts import VersionedArtifactStore


@dataclass
class SpecialistCoverage:
    name: str
    n_columns: int
    n_rows_nonnull: int
    n_rows_total: int

    @property
    def coverage_ratio(self) -> float:
        if self.n_rows_total <= 0:
            return 0.0
        return self.n_rows_nonnull / float(self.n_rows_total)


def _find_latest_labeled_data_version(
    store: VersionedArtifactStore,
    artifact_name: str,
) -> str:
    """Return the version name of the latest labeled_data artifact.

    We prefer the most recent `created_at` timestamp; if unavailable,
    we fall back to the version with the largest num_rows.
    """

    best_version: Optional[str] = None
    best_created: Optional[pd.Timestamp] = None
    best_rows: int = 0

    for version in store.list_versions():
        info: Dict = store.get_version_info(version)  # type: ignore[assignment]
        if info.get("artifact_name") != artifact_name:
            continue

        created_raw = info.get("created_at")
        created = None
        if created_raw is not None:
            try:
                created = pd.to_datetime(created_raw)
            except Exception:
                created = None

        num_rows = int(info.get("num_rows", 0) or 0)

        if best_version is None:
            best_version = version
            best_created = created
            best_rows = num_rows
            continue

        # Prefer by created_at when both are available; otherwise, by num_rows
        if created is not None and best_created is not None:
            if created > best_created:
                best_version = version
                best_created = created
                best_rows = num_rows
        elif created is not None and best_created is None:
            best_version = version
            best_created = created
            best_rows = num_rows
        elif created is None and best_created is None and num_rows > best_rows:
            best_version = version
            best_rows = num_rows

    if best_version is None:
        raise RuntimeError(f"No versions found for artifact_name={artifact_name!r}")

    return best_version


def _load_labeled_data(
    root: Path,
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str,
) -> pd.DataFrame:
    """Load the latest labeled_data DataFrame from the analyst store."""

    store_name = f"{symbol}_{exchange}_{timeframe}_{direction}_analyst"
    store_path = root / store_name
    if not store_path.exists():
        raise FileNotFoundError(f"VersionedArtifactStore path not found: {store_path}")

    store = VersionedArtifactStore(store_path)
    artifact_name = f"labeled_data_{symbol}_{timeframe}"
    version = _find_latest_labeled_data_version(store, artifact_name)

    view = store.get_view(version)
    df = view.materialize()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    # Prefer DatetimeIndex if already present; otherwise, try to build one
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
    elif "timestamp" in df.columns:
        idx = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        raise RuntimeError(
            "Labeled data has neither DatetimeIndex nor 'timestamp' column; "
            "cannot construct temporal index for coverage analysis."
        )

    if not isinstance(idx, pd.DatetimeIndex) or idx.isna().all():
        raise RuntimeError("Failed to construct valid DatetimeIndex for labeled_data")

    df = df.copy()
    df.index = idx
    return df


def _classify_specialist_column(col: str) -> Optional[str]:
    """Heuristic mapping from column name to specialist group.

    This is intentionally conservative and only classifies columns that
    clearly originate from known specialist steps.
    """

    # Liquidity regime probabilities (raw or prefixed with 'liquidity_')
    if "liquidity_regime_" in col:
        return "liquidity"

    # SMC scalar / probabilities
    if col.startswith("smc_") or col == "smc_predicted":
        return "smc"

    # ML Risk probabilities / scalar scores
    if col.startswith("risk_") or col.startswith("risk_regime_"):
        return "risk"

    # Macro trend scalar
    if "macro_trend" in col:
        return "macro_trend"

    # Mean-reversion / MR trend
    if col.startswith("mr_") or col.startswith("mr_trend_"):
        return "mean_reversion_or_mr_trend"

    # Path risk scalar
    if col.startswith("path_risk") or col == "path_risk_score":
        return "path_risk"

    # Breakout/Bounce regime outputs & scalars
    if col.startswith("breakout_") or col in (
        "support_scalar",
        "resistance_scalar",
        "breakout_success_prob",
        "breakout_high_conf_signal",
        "is_support",
        "is_resistance",
    ):
        return "breakout_bounce"

    # Volume Force scalar outputs
    if col.startswith("vol_force_") or col.startswith("volume_force"):
        return "volume_force"

    return None


def compute_specialist_coverage(
    df: pd.DataFrame,
    lookback_days: int,
) -> List[SpecialistCoverage]:
    """Compute per-specialist non-NaN coverage over a recent lookback window.

    The effective target index is defined as rows within the last
    `lookback_days` days of the labeled_data index, additionally
    restricted to rows where `binary_label` is non-null when available.
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("labeled_data must have a DatetimeIndex for coverage analysis")

    idx = df.index.sort_values()
    end = idx.max()
    start = end - pd.Timedelta(days=lookback_days)

    window_mask = (idx >= start) & (idx <= end)
    df_win = df.loc[window_mask]

    if "binary_label" in df_win.columns:
        target_mask = df_win["binary_label"].notna()
        df_target = df_win.loc[target_mask]
    else:
        df_target = df_win

    n_rows_total = len(df_target)
    if n_rows_total == 0:
        raise RuntimeError("No rows in target window after applying binary_label mask (if present)")

    # Build mapping from specialist -> list of columns
    groups: Dict[str, List[str]] = {}
    for col in df_target.columns:
        spec = _classify_specialist_column(col)
        if spec is None:
            continue
        groups.setdefault(spec, []).append(col)

    coverages: List[SpecialistCoverage] = []
    for spec, cols in sorted(groups.items()):
        block = df_target[cols]
        # Row is considered covered if any of the group's columns is non-NaN
        n_nonnull_rows = int(block.notna().any(axis=1).sum())
        coverages.append(
            SpecialistCoverage(
                name=spec,
                n_columns=len(cols),
                n_rows_nonnull=n_nonnull_rows,
                n_rows_total=n_rows_total,
            )
        )

    return coverages


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Liquidity-specific coverage sanity check over labeled meta-labeling "
            "data, reporting per-specialist non-NaN coverage over a recent "
            "lookback window."
        )
    )
    parser.add_argument("--symbol", default="ETHUSDT", help="Trading symbol (default: ETHUSDT)")
    parser.add_argument("--exchange", default="binance", help="Exchange (default: binance)")
    parser.add_argument("--timeframe", default="15m", help="Base timeframe (default: 15m)")
    parser.add_argument("--direction", default="long", help="Direction (default: long)")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=365,
        help="Lookback window in days for coverage analysis (default: 365)",
    )
    parser.add_argument(
        "--store-root",
        type=str,
        default="versioned_artifacts",
        help="Root directory for VersionedArtifactStore (default: versioned_artifacts)",
    )

    args = parser.parse_args()

    root = Path(args.store_root)
    df = _load_labeled_data(
        root=root,
        symbol=args.symbol,
        exchange=args.exchange,
        timeframe=args.timeframe,
        direction=args.direction,
    )

    idx = df.index.sort_values()
    end = idx.max()
    start = end - pd.Timedelta(days=args.lookback_days)

    print("=" * 80)
    print("Liquidity & specialist coverage sanity check over labeled_data")
    print("- Symbol      :", args.symbol)
    print("- Exchange    :", args.exchange)
    print("- Timeframe   :", args.timeframe)
    print("- Direction   :", args.direction)
    print("- Store root  :", root)
    print("- Index range :", idx.min(), "→", idx.max(), f"(n={len(idx)})")
    print("- Window      :", start, "→", end, f"(lookback_days={args.lookback_days})")

    coverages = compute_specialist_coverage(df, lookback_days=args.lookback_days)

    print("\nPer-specialist non-NaN coverage over target window (binary_label mask applied if present):")
    print("\n{:<28} {:>8} {:>10} {:>9}".format("Specialist", "n_cols", "rows_ok", "cover%"))
    print("-" * 60)
    for cov in coverages:
        print(
            "{:<28} {:>8d} {:>10d} {:>8.1f}".format(
                cov.name,
                cov.n_columns,
                cov.n_rows_nonnull,
                cov.coverage_ratio * 100.0,
            )
        )

    print("\nNOTE: Liquidity should now exhibit coverage comparable to other specialists "
          "on this effective training index. If Liquidity coverage here is ~100% "
          "while the specialist_feature_diagnostics report still shows lower "
          "coverage, the discrepancy is in diagnostics reporting, not in the "
          "underlying Liquidity features used for meta-label training.")
    print("=" * 80)


if __name__ == "__main__":
    main()
