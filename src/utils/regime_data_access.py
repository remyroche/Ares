from __future__ import annotations

"""
Unified helpers to access and operate on per-HMM regime data consistently across steps.

Primary responsibilities:
- Detect the regime label column in a DataFrame (prefers 'composite_cluster_id')
- Ensure regime labels by merging HMM composite clusters on timestamp when missing
- Iterate per-regime subsets with minimum sample thresholds
- Provide consistent per-regime train/val/test splits
- Load the unified regime dataset created in step04
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple

import os
import pandas as pd

try:
    from src.utils.logger import system_logger
except Exception:  # pragma: no cover - fallback
    import logging
    system_logger = logging.getLogger(__name__)

try:
    from src.utils.hmm_composite_manager import get_hmm_composite_manager
except Exception:  # pragma: no cover - fallback
    get_hmm_composite_manager = None  # type: ignore


REGIME_COLUMN_CANDIDATES: list[str] = [
    "composite_cluster_id",
    "regime",
    "hmm_regime",
    "market_regime",
    "cluster_id",
    "regime_id",
]


def get_regime_column(df: pd.DataFrame) -> Optional[str]:
    """Return the name of the regime column if present, else None.

    Prefers 'composite_cluster_id' for HMM composite regimes.
    """
    for candidate in REGIME_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


def ensure_regime_labels(
    df: pd.DataFrame,
    exchange: str,
    symbol: str,
    timeframe: str,
    data_dir: str = "data/training",
) -> pd.DataFrame:
    """Ensure DataFrame has a regime column; if missing, try to merge HMM clusters.

    If HMM composite clusters can be loaded, merge their 'composite_cluster_id' by 'timestamp'.
    Returns a new DataFrame (does not mutate input).
    """
    logger = system_logger.getChild("RegimeDataAccess")
    if get_regime_column(df) is not None:
        return df
    try:
        hmm_df: Optional[pd.DataFrame] = None
        # Try manager-backed load first
        if get_hmm_composite_manager is not None:
            manager = get_hmm_composite_manager()
            hmm_df = manager.load_composite_clusters(exchange, symbol, timeframe, data_dir)
        # Fallback to step04-style file layout if manager path not found
        if hmm_df is None or hmm_df.empty:
            try:
                fallback_path = os.path.join(
                    data_dir,
                    "hmm_regimes",
                    f"{exchange}_{symbol}_{timeframe}_composite_clusters.parquet",
                )
                if os.path.exists(fallback_path):
                    hmm_df = pd.read_parquet(fallback_path)
                    logger.info(
                        "🟡 Loaded HMM clusters from fallback path: %s", fallback_path
                    )
            except Exception:
                pass
        if hmm_df is None or hmm_df.empty:
            logger.warning(
                "No HMM composite clusters available for merge; leaving df unchanged"
            )
            return df
        # Standardize timestamp dtype to enable safe merge
        left = df.copy()
        right = hmm_df[["timestamp", "composite_cluster_id"]].copy()
        # Align timestamp dtype
        try:
            left_ts = left["timestamp"].dtype
            if right["timestamp"].dtype != left_ts:
                right["timestamp"] = right["timestamp"].astype(left_ts, copy=False)
        except Exception:
            pass
        merged = pd.merge(left, right, on="timestamp", how="left")
        if "composite_cluster_id" in merged.columns:
            logger.info(
                "✅ Added 'composite_cluster_id' to DataFrame via HMM merge (%d rows)",
                len(merged),
            )
        return merged
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Failed to ensure regime labels: {e}")
        return df


def get_regime_ids(df: pd.DataFrame, regime_column: Optional[str] = None) -> List[Any]:
    """Return sorted unique regime identifiers from DataFrame."""
    col = regime_column or get_regime_column(df)
    if not col or col not in df.columns:
        return []
    series = pd.Series(df[col]).dropna()
    try:
        return sorted(series.unique().tolist())
    except Exception:
        return sorted(series.astype(str).unique().tolist())


def iter_regimes(
    df: pd.DataFrame,
    regime_column: Optional[str] = None,
    min_samples: int = 10,
) -> Iterator[Tuple[Any, pd.DataFrame]]:
    """Yield (regime_id, regime_df) for each regime with at least min_samples rows."""
    col = regime_column or get_regime_column(df)
    if not col or col not in df.columns:
        return
    for regime_id, regime_df in df.groupby(col, sort=True):
        if len(regime_df) >= min_samples:
            yield regime_id, regime_df


def split_train_val_test_by_regime(
    df: pd.DataFrame,
    regime_column: Optional[str] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    min_samples_per_split: int = 10,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Create consistent per-regime chronological splits for train/val/test.

    Returns dict: {str(regime_id): {"train": df, "validation": df, "test": df}}
    """
    col = regime_column or get_regime_column(df)
    if not col or col not in df.columns:
        return {}
    results: Dict[str, Dict[str, pd.DataFrame]] = {}
    for regime_id, regime_df in df.sort_values("timestamp").groupby(col, sort=True):
        n = len(regime_df)
        if n < max(min_samples_per_split, 3):
            continue
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        # Ensure at least one sample per split when possible
        train_end = max(train_end, 1)
        val_end = max(val_end, train_end + 1) if n >= 2 else train_end
        val_end = min(val_end, n - 1)
        train_df = regime_df.iloc[:train_end].copy()
        val_df = regime_df.iloc[train_end:val_end].copy()
        test_df = regime_df.iloc[val_end:].copy()
        if len(test_df) == 0 and len(val_df) > 1:
            test_df = val_df.tail(1)
            val_df = val_df.iloc[:-1]
        results[str(regime_id)] = {
            "train": train_df,
            "validation": val_df,
            "test": test_df,
        }
    return results


def _candidate_unified_paths(
    exchange: str, symbol: str, timeframe: str, data_dir: str
) -> List[str]:
    base_name = f"{exchange}_{symbol}_{timeframe}_unified_regime_data.parquet"
    return [
        os.path.join(data_dir, "training", base_name),
        os.path.join("data", "training", base_name),
        os.path.join("data_cache", "training", base_name),
    ]


def load_unified_regime_dataset(
    exchange: str, symbol: str, timeframe: str, data_dir: str = "data/training"
) -> Optional[pd.DataFrame]:
    """Load the unified regime dataset produced by step04 if available."""
    logger = system_logger.getChild("RegimeDataAccess")
    for path in _candidate_unified_paths(exchange, symbol, timeframe, data_dir):
        try:
            if os.path.exists(path):
                df = pd.read_parquet(path)
                logger.info(
                    "✅ Loaded unified regime dataset (%d rows) from %s", len(df), path
                )
                return df
        except Exception:
            continue
    logger.warning("Unified regime dataset not found for %s_%s_%s", exchange, symbol, timeframe)
    return None

