from collections.abc import Callable

import pandas as pd

from src.utils.centralized_decorators import (
    guard_dataframe_nulls,
    validate_call_or_runtime_types,
    with_tracing_span,
)


class FeatureGenerator:
    def __init__(
        self, custom_features: list[Callable[[pd.DataFrame], pd.DataFrame]] | None = None,
    ) -> None:
        self.feature_functions = [
            self.price_features,
            self.moving_averages,
            self.volatility_features,
            self.volume_features,
            self.technical_indicators,
        ]
        if custom_features:
            self.feature_functions.extend(custom_features)

    @validate_call_or_runtime_types
    @guard_dataframe_nulls(mode="warn", arg_index=1)
    @with_tracing_span("FeatureGenerator.generate", log_args=False)
    @validate_call_or_runtime_types
    @guard_dataframe_nulls(mode="warn", arg_index=1)
    @with_tracing_span("FeatureGenerator.price_features", log_args=False)
    @validate_call_or_runtime_types
    @guard_dataframe_nulls(mode="warn", arg_index=1)
    @with_tracing_span("FeatureGenerator.moving_averages", log_args=False)
    @validate_call_or_runtime_types
    @guard_dataframe_nulls(mode="warn", arg_index=1)
    @with_tracing_span("FeatureGenerator.volatility_features", log_args=False)
    @validate_call_or_runtime_types
    @guard_dataframe_nulls(mode="warn", arg_index=1)
    @with_tracing_span("FeatureGenerator.volume_features", log_args=False)
    @validate_call_or_runtime_types
    @guard_dataframe_nulls(mode="warn", arg_index=1)
    @with_tracing_span("FeatureGenerator.technical_indicators", log_args=False)
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, 1e-9)
        return 100 - (100 / (1 + rs))

    def _calculate_macd(
        self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9,
    ) -> pd.Series:
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line - signal_line
