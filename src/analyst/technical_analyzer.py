import pandas as pd
import pandas_ta as ta
from loguru import logger

class TechnicalAnalyzer:
    """
    A specialized component for calculating a wide range of technical indicators.
    It takes raw kline data and enriches it with technical analysis features.
    """
    def __init__(self):
        self.logger = logger
        self.logger.info("TechnicalAnalyzer initialized.")

    def calculate_indicators(self, kline_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates technical indicators and appends them to the DataFrame.

        Args:
            kline_df: A pandas DataFrame with columns ['open', 'high', 'low', 'close', 'volume'].

        Returns:
            The original DataFrame with added columns for each indicator.
        """
        if kline_df.empty:
            self.logger.warning("Cannot calculate indicators on an empty DataFrame.")
            return kline_df

        try:
            # Use the pandas_ta library's "Strategy" feature for clean indicator management
            # You can customize this strategy with any indicators from the library.
            custom_strategy = ta.Strategy(
                name="Ares_TA_Strategy",
                description="A collection of standard technical indicators for Ares.",
                ta=[
                    # Trend Indicators
                    {"kind": "sma", "length": 20},
                    {"kind": "sma", "length": 50},
                    {"kind": "ema", "length": 200},
                    {"kind": "adx", "length": 14},
                    {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
                    
                    # Momentum Indicators
                    {"kind": "rsi", "length": 14},
                    {"kind": "stoch", "k": 14, "d": 3, "smooth_k": 3},

                    # Volatility Indicators
                    {"kind": "bbands", "length": 20, "std": 2},
                    {"kind": "atr", "length": 14},

                    # Volume Indicators
                    {"kind": "obv"},
                ]
            )
            
            # Append the indicators to the DataFrame
            kline_df.ta.strategy(custom_strategy)
            
            self.logger.debug(f"Calculated {len(custom_strategy.ta)} indicators.")
            return kline_df

        except Exception as e:
            self.logger.error(f"An error occurred during indicator calculation: {e}", exc_info=True)
            # Return the original dataframe on error
            return kline_df
