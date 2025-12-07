"""Unified exchange-level OHLCV standardizer stubs.

This module provides a minimal implementation of the interface expected by the
*klines_adapter* modules for different exchanges. It is deliberately lightweight
and focuses on:

- Exposing a `UnifiedExchangeStandardizer` class with a `.standardize(df, ...)`
  method that returns a cleaned/normalized OHLCV DataFrame.
- Providing a `DataQualityLevel` enum with at least a `STANDARD` level so that
  adapters can pass a quality hint into higher-level components.

For the purposes of public klines download + processing, the implementation can
safely be a thin wrapper around the existing `UnifiedOHLCVStandardizer` stub.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd

try:  # Prefer the dedicated OHLCV standardizer when available
    from exchanges.shared.unified_ohlcv_standardizer import UnifiedOHLCVStandardizer
except Exception:  # pragma: no cover - defensive fallback
    class UnifiedOHLCVStandardizer:  # type: ignore[no-redef]
        """Fallback no-op OHLCV standardizer.

        If the real implementation is not present, we simply return the input
        DataFrame unchanged. This keeps the data pipeline functional without
        enforcing any strict schema transformations.
        """

        def standardize(self, df: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
            return df


class DataQualityLevel(Enum):
    """Data quality level hint for unified exchange adapters.

    The Binance/BingX/OKX klines adapters only ever use `STANDARD` when
    constructing enhanced unified adapters. Additional levels can be added
    later if needed without breaking the existing interface.
    """

    STANDARD = "standard"


@dataclass
class UnifiedExchangeStandardizer:
    """Thin wrapper around `UnifiedOHLCVStandardizer`.

    This class mirrors the name and basic interface expected by the various
    *klines_adapter* modules (Binance, BingX, OKX, etc.). It delegates actual
    work to `UnifiedOHLCVStandardizer`, which is itself a lightweight stub in
    this repository.
    """

    _ohlcv_standardizer: UnifiedOHLCVStandardizer = UnifiedOHLCVStandardizer()

    def standardize(self, df: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Standardize an OHLCV DataFrame.

        Args:
            df: Input DataFrame containing OHLCV data.

        Returns:
            Standardized DataFrame. In this stub implementation, this is
            effectively the same as the input, possibly after any light
            normalization performed by `UnifiedOHLCVStandardizer`.
        """

        if df is None or df.empty:
            return df

        return self._ohlcv_standardizer.standardize(df, *args, **kwargs)
