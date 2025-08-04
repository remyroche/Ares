# src/analyst/advanced_feature_engineering.py

import pandas as pd
import numpy as np
import talib
from scipy.signal import find_peaks
from scipy import stats
from typing import Any, Dict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analyst.feature_engineering import FeatureEngineeringEngine
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger

import pandas as pd
import numpy as np
import talib
from scipy.signal import find_peaks
from scipy import stats
from typing import Any, Dict

# Mock implementations for standalone execution and clarity
class MockLogger:
    def getChild(self, name):
        return self
    def info(self, msg):
        print(f"INFO: {msg}")
    def error(self, msg):
        print(f"ERROR: {msg}")

system_logger = MockLogger()

def handle_errors(exceptions, default_return, context):
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            try:
                # The first argument is 'self', so we get the logger from there
                logger = args[0].logger if args else system_logger
                return await func(*args, **kwargs)
            except exceptions as e:
                logger.error(f"Error in {context}: {e}")
                return default_return
        
        def sync_wrapper(*args, **kwargs):
            try:
                logger = args[0].logger if args else system_logger
                return func(*args, **kwargs)
            except exceptions as e:
                logger.error(f"Error in {context}: {e}")
                return default_return

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


class AdvancedFeatureEngineering:
    """
    A comprehensive feature engineering system that generates a wide array of features,
    from standard technical indicators to advanced market microstructure and pattern analysis.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the advanced feature engineering system.
        Args:
            config: Configuration dictionary, which can contain parameters for feature calculation.
        """
        self.config = config
        self.logger = system_logger.getChild("AdvancedFeatureEngineering")
        self.advanced_config = config.get("advanced_feature_engineering", {})
        
        # Add new config flags for better control
        self.enable_divergence_detection = self.advanced_config.get("enable_divergence_detection", True)
        self.enable_pattern_recognition = self.advanced_config.get("enable_pattern_recognition", True)
        self.enable_volume_profile = self.advanced_config.get("enable_volume_profile", True)
        self.enable_market_microstructure = self.advanced_config.get("enable_market_microstructure", True)
        self.enable_volatility_targeting = self.advanced_config.get("enable_volatility_targeting", True)
        
        self.logger.info("🚀 Initialized Comprehensive AdvancedFeatureEngineering")

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError),
        default_return=pd.DataFrame(),
        context="advanced feature generation",
    )
    def generate_features(
        self,
        klines_df: pd.DataFrame,
        agg_trades_df: pd.DataFrame | None = None,
        order_book_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Generate a comprehensive set of features from various data sources.
        """
        self.logger.info("🎯 Generating a full spectrum of features...")
        features = klines_df.copy()

        # --- Feature Calculation Pipeline ---
        features = self._calculate_standard_indicators(features)
        features = self._calculate_advanced_momentum_volume_features(features, agg_trades_df)
        features = self._calculate_volatility_risk_features(features)
        features = self._calculate_sr_channel_features(features)
        features = self._calculate_pivot_points(features) # New

        if self.enable_divergence_detection:
            features = self._calculate_divergence_features(features)
        if self.enable_pattern_recognition:
            features = self._calculate_pattern_recognition_features(features)
        if self.enable_volume_profile and agg_trades_df is not None:
            features = self._calculate_volume_profile_features(features, agg_trades_df)
        if self.enable_market_microstructure and order_book_df is not None:
            features = self._calculate_market_microstructure_features(features, order_book_df)
        
        features = self._calculate_advanced_order_flow_funding_features(features, agg_trades_df)

        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.logger.info(f"✅ Generated {len(features.columns)} total features")
        return features

    # --- Category 1: Standard Technical Indicators ---
    def _calculate_standard_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Calculating Standard Technical Indicators...")
        df['RSI'] = talib.RSI(df['close'])
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['close'])
        df['MOM'] = talib.MOM(df['close'])
        df['ADX'] = talib.ADX(df['high'], df['low'], df['close'])
        df['SMA_50'] = talib.SMA(df['close'], timeperiod=50)
        df['SMA_200'] = talib.SMA(df['close'], timeperiod=200)
        df['EMA_20'] = talib.EMA(df['close'], timeperiod=20)
        # New: Stochastic RSI
        stoch_rsi_k, stoch_rsi_d = talib.STOCHRSI(df['close'])
        df['STOCHRSI_k'] = stoch_rsi_k
        df['STOCHRSI_d'] = stoch_rsi_d
        return df

    # --- Category 2: Advanced Momentum & Volume Indicators ---
    def _calculate_advanced_momentum_volume_features(self, df: pd.DataFrame, agg_trades_df: pd.DataFrame | None) -> pd.DataFrame:
        self.logger.info("Calculating Advanced Momentum & Volume Indicators...")
        df['Williams_R'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)
        df['MFI'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
        df['OBV'] = talib.OBV(df['close'], df['volume'])
        df['volume_change'] = df['volume'].pct_change()
        df['volume_MA_10'] = df['volume'].rolling(window=10).mean()
        df['VROC'] = df['volume'].pct_change(periods=14) * 100
        
        # New: Chaikin Money Flow (CMF)
        money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-9)
        money_flow_volume = money_flow_multiplier * df['volume']
        df['CMF_20'] = money_flow_volume.rolling(window=20).sum() / df['volume'].rolling(window=20).sum()

        # VWAP Calculation
        vwap_data = df.copy()
        vwap_data['tpv'] = ((vwap_data['high'] + vwap_data['low'] + vwap_data['close']) / 3) * vwap_data['volume']
        vwap_data['cum_tpv'] = vwap_data['tpv'].cumsum()
        vwap_data['cum_volume'] = vwap_data['volume'].cumsum()
        df['VWAP'] = vwap_data['cum_tpv'] / vwap_data['cum_volume']
        
        # Estimated Order Flow from OHLCV
        buying_pressure = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-9)
        selling_pressure = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)
        df['buy_order_flow'] = buying_pressure * df['volume']
        df['sell_order_flow'] = selling_pressure * df['volume']
        df['order_flow_imbalance'] = df['buy_order_flow'] - df['sell_order_flow']
        return df

    # --- Category 3: Volatility & Risk Features ---
    def _calculate_volatility_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Calculating Volatility & Risk Features...")
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['ATR_normalized'] = df['ATR'] / df['close']
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['realized_volatility_30d'] = df['log_return'].rolling(window=30).std() * np.sqrt(365)
        target_vol = self.advanced_config.get("target_volatility", 0.15)
        df['volatility_position_size'] = target_vol / (df['realized_volatility_30d'] + 1e-9)
        df['volatility_regime'] = pd.cut(df['realized_volatility_30d'], bins=[0, target_vol*0.75, target_vol*1.5, np.inf], labels=['low', 'medium', 'high'])
        
        # New: Ulcer Index
        period = self.advanced_config.get("ulcer_period", 14)
        highest_close = df['close'].rolling(window=period).max()
        percentage_drawdown = ((df['close'] - highest_close) / highest_close) * 100
        squared_drawdown = percentage_drawdown ** 2
        df['ulcer_index'] = np.sqrt(squared_drawdown.rolling(window=period).mean())
        return df

    # --- Category 4: Support, Resistance, and Channel Features ---
    def _calculate_sr_channel_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Calculating S/R and Channel Features...")
        # Bollinger Bands
        bb_period = self.advanced_config.get("bb_period", 20)
        bbands = talib.BBANDS(df['close'], timeperiod=bb_period, nbdevup=2, nbdevdn=2, matype=0)
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = bbands
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / (df['BB_middle'] + 1e-9)
        # Keltner Channels
        kc_period = self.advanced_config.get("kc_period", 20)
        ema = talib.EMA(df['close'], timeperiod=kc_period)
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=kc_period)
        df['KC_upper'] = ema + (atr * 2)
        df['KC_middle'] = ema
        df['KC_lower'] = ema - (atr * 2)
        # VWAP Standard Deviation Bands
        if 'VWAP' in df.columns:
            vwap_std = df['close'].rolling(window=20).std()
            df["vwap_upper_band"] = df["VWAP"] + (vwap_std * 2)
            df["vwap_lower_band"] = df["VWAP"] - (vwap_std * 2)
        # Distance to nearest S/R (from peaks/troughs)
        peaks, _ = find_peaks(df["high"].values, distance=15, prominence=df['ATR'].mean()*0.5 if 'ATR' in df.columns and not df['ATR'].isnull().all() else 1)
        troughs, _ = find_peaks(-df["low"].values, distance=15, prominence=df['ATR'].mean()*0.5 if 'ATR' in df.columns and not df['ATR'].isnull().all() else 1)
        if len(peaks) > 0:
            df["distance_to_resistance"] = df["close"].apply(lambda x: min([abs(x - df["high"].iloc[p]) for p in peaks]) / x)
        if len(troughs) > 0:
            df["distance_to_support"] = df["close"].apply(lambda x: min([abs(x - df["low"].iloc[t]) for t in troughs]) / x)
        return df
        
    # --- New Category: Pivot Points ---
    def _calculate_pivot_points(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Calculating Pivot Points...")
        prev_high = df['high'].shift(1)
        prev_low = df['low'].shift(1)
        prev_close = df['close'].shift(1)
        
        pivot = (prev_high + prev_low + prev_close) / 3
        df['pivot'] = pivot
        df['pivot_s1'] = (pivot * 2) - prev_high
        df['pivot_r1'] = (pivot * 2) - prev_low
        df['pivot_s2'] = pivot - (prev_high - prev_low)
        df['pivot_r2'] = pivot + (prev_high - prev_low)
        df['pivot_s3'] = prev_low - 2 * (prev_high - pivot)
        df['pivot_r3'] = prev_high + 2 * (pivot - prev_low)
        return df

    # --- Category 5: Divergence Detection ---
    def _calculate_divergence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Calculating Divergence Features...")
        df = self._detect_divergence(df, 'close', 'RSI', 'rsi_divergence')
        df = self._detect_divergence(df, 'close', 'MACD_hist', 'macd_divergence')
        df = self._detect_divergence(df, 'close', 'OBV', 'obv_divergence')
        df = self._detect_divergence(df, 'close', 'MFI', 'mfi_divergence')
        df = self._detect_divergence(df, 'close', 'Williams_R', 'willr_divergence')
        return df

    def _detect_divergence(self, df: pd.DataFrame, price_col: str, ind_col: str, new_col: str) -> pd.DataFrame:
        # Find peaks and troughs
        price_peaks, _ = find_peaks(df[price_col], distance=10)
        price_troughs, _ = find_peaks(-df[price_col], distance=10)
        ind_peaks, _ = find_peaks(df[ind_col].dropna(), distance=10)
        ind_troughs, _ = find_peaks(-df[ind_col].dropna(), distance=10)
        df[f'bullish_{new_col}'] = 0
        df[f'bearish_{new_col}'] = 0
        if len(price_troughs) >= 2 and len(ind_troughs) >= 2:
            if (df[price_col].iloc[price_troughs[-1]] < df[price_col].iloc[price_troughs[-2]]) and (df[ind_col].iloc[ind_troughs[-1]] > df[ind_col].iloc[ind_troughs[-2]]):
                df.loc[df.index[price_troughs[-1]], f'bullish_{new_col}'] = 1
        if len(price_peaks) >= 2 and len(ind_peaks) >= 2:
            if (df[price_col].iloc[price_peaks[-1]] > df[price_col].iloc[price_peaks[-2]]) and (df[ind_col].iloc[ind_peaks[-1]] < df[ind_col].iloc[ind_peaks[-2]]):
                df.loc[df.index[price_peaks[-1]], f'bearish_{new_col}'] = 1
        return df

    # --- Category 6: Chart Pattern Recognition ---
    def _calculate_pattern_recognition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Calculating Pattern Recognition Features...")
        # Candlestick Patterns
        df['CDLDOJI'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
        df['CDLENGULFING'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
        df['CDLHAMMER'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
        # New Candlestick Patterns
        df['CDLMORNINGSTAR'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
        df['CDLEVENINGSTAR'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])
        df['CDL3WHITESOLDIERS'] = talib.CDL3WHITESOLDIERS(df['open'], df['high'], df['low'], df['close'])
        df['CDL3BLACKCROWS'] = talib.CDL3BLACKCROWS(df['open'], df['high'], df['low'], df['close'])
        # Chart Patterns
        df = self._calculate_double_patterns(df)
        df = self._calculate_head_shoulders(df)
        df = self._calculate_triangle_patterns(df)
        return df

    def _calculate_double_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        peaks, _ = find_peaks(df["high"].values, distance=20)
        troughs, _ = find_peaks(-df["low"].values, distance=20)
        df["double_top"] = 0
        df["double_bottom"] = 0
        if len(peaks) >= 2 and abs(df["high"].iloc[peaks[-1]] - df["high"].iloc[peaks[-2]]) / df["high"].iloc[peaks[-2]] < 0.02:
            df.loc[df.index[peaks[-1]], "double_top"] = 1
        if len(troughs) >= 2 and abs(df["low"].iloc[troughs[-1]] - df["low"].iloc[troughs[-2]]) / df["low"].iloc[troughs[-2]] < 0.02:
            df.loc[df.index[troughs[-1]], "double_bottom"] = 1
        return df

    def _calculate_head_shoulders(self, df: pd.DataFrame) -> pd.DataFrame:
        peaks, _ = find_peaks(df["high"].values, distance=15)
        df["head_shoulders"] = 0
        if len(peaks) >= 3:
            left_s, head, right_s = peaks[-3], peaks[-2], peaks[-1]
            if (df["high"].iloc[head] > df["high"].iloc[left_s]) and (df["high"].iloc[head] > df["high"].iloc[right_s]):
                df.loc[df.index[right_s], "head_shoulders"] = 1
        return df

    def _calculate_triangle_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        window = 30
        df["ascending_triangle"] = 0
        df["descending_triangle"] = 0
        for i in range(window, len(df)):
            recent_data = df.iloc[i - window : i + 1]
            highs, lows = recent_data["high"].values, recent_data["low"].values
            x = np.arange(len(highs))
            high_slope, _, _, _, _ = stats.linregress(x, highs)
            low_slope, _, _, _, _ = stats.linregress(x, lows)
            if abs(high_slope) < 0.001 and low_slope > 0.001:
                df.loc[df.index[i], "ascending_triangle"] = 1
            if high_slope < -0.001 and abs(low_slope) < 0.001:
                df.loc[df.index[i], "descending_triangle"] = 1
        return df

    # --- Category 7: Volume Profile & CVD ---
    def _calculate_volume_profile_features(self, df: pd.DataFrame, agg_trades_df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Calculating Volume Profile and CVD Features...")
        # Calculate Volume Profile (POC, HVN, LVN)
        price_volume = agg_trades_df.groupby('price')['quantity'].sum()
        df['poc'] = price_volume.idxmax()
        volume_threshold_high = price_volume.quantile(0.80)
        volume_threshold_low = price_volume.quantile(0.20)
        hvn_prices = price_volume[price_volume > volume_threshold_high].index
        lvn_prices = price_volume[price_volume < volume_threshold_low].index
        df['distance_to_poc'] = (df['close'] - df['poc']).abs() / df['poc']
        if not hvn_prices.empty:
            df['distance_to_hvn'] = df['close'].apply(lambda x: min([abs(x - p) for p in hvn_prices]) / x)
        if not lvn_prices.empty:
            df['distance_to_lvn'] = df['close'].apply(lambda x: min([abs(x - p) for p in lvn_prices]) / x)
        # Calculate Cumulative Volume Delta (CVD)
        agg_trades_df['delta'] = agg_trades_df['quantity'] * np.where(agg_trades_df['is_buyer_maker'], -1, 1)
        cvd = agg_trades_df.set_index('timestamp')['delta'].cumsum()
        df['CVD'] = cvd.reindex(df.index, method='ffill')
        df['CVD_change'] = df['CVD'].diff()
        return df

    # --- Category 8: Market Microstructure ---
    def _calculate_market_microstructure_features(self, df: pd.DataFrame, order_book_df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Calculating Market Microstructure Features...")
        if not isinstance(order_book_df.index, pd.DatetimeIndex):
            order_book_df['timestamp'] = pd.to_datetime(order_book_df['timestamp'])
            order_book_df.set_index('timestamp', inplace=True)
        # Bid-Ask Spread
        order_book_df["spread"] = order_book_df["ask_price"] - order_book_df["bid_price"]
        df['bid_ask_spread'] = order_book_df["spread"].resample(df.index.freq).mean().reindex(df.index, method='ffill')
        # Order Imbalance
        order_book_df["imbalance"] = (order_book_df["bid_quantity"] - order_book_df["ask_quantity"]) / (order_book_df["bid_quantity"] + order_book_df["ask_quantity"] + 1e-9)
        df['order_imbalance'] = order_book_df["imbalance"].resample(df.index.freq).mean().reindex(df.index, method='ffill')
        # Market Depth
        order_book_df["market_depth"] = order_book_df["bid_quantity"] + order_book_df["ask_quantity"]
        df['market_depth'] = order_book_df["market_depth"].resample(df.index.freq).sum().reindex(df.index, method='ffill')
        return df

    # --- Category 9: Advanced Order Flow & Funding ---
    def _calculate_advanced_order_flow_funding_features(self, df: pd.DataFrame, agg_trades_df: pd.DataFrame | None) -> pd.DataFrame:
        """Wrapper for advanced order flow and funding features."""
        if agg_trades_df is not None:
            df = self._calculate_order_flow_indicators(df, agg_trades_df)
        if 'funding_rate' in df.columns:
            df = self._calculate_enhanced_funding_features(df)
        return df

    def _calculate_enhanced_funding_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate enhanced funding rate features."""
        self.logger.info("Calculating enhanced funding rate features...")
        if "funding_rate" not in df.columns:
            self.logger.warning("`funding_rate` column not found. Skipping.")
            return df
        try:
            funding_momentum_period = self.config.get("funding_momentum_period", 3)
            df["Funding_Momentum"] = df["funding_rate"].diff(funding_momentum_period).fillna(0.0)
            price_change = df["close"].pct_change(funding_momentum_period).fillna(0.0)
            df["Funding_Divergence"] = df["Funding_Momentum"] - price_change
            funding_window = self.config.get("funding_window", 24)
            funding_mean = df["funding_rate"].rolling(funding_window, min_periods=1).mean()
            funding_std = df["funding_rate"].rolling(funding_window, min_periods=1).std().replace(0, 1e-8)
            df["Funding_Extreme"] = ((df["funding_rate"] - funding_mean) / funding_std).fillna(0.0)
        except Exception as e:
            self.logger.warning(f"Failed to calculate enhanced funding features: {e}")
            df["Funding_Momentum"], df["Funding_Divergence"], df["Funding_Extreme"] = 0.0, 0.0, 0.0
        return df

    def _calculate_order_flow_indicators(self, df: pd.DataFrame, agg_trades_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate order flow and liquidity indicators from aggregated trades."""
        self.logger.info("Calculating order flow and liquidity indicators...")
        df["Buy_Sell_Pressure_Ratio"], df["Order_Flow_Imbalance"], df["Large_Order_Count"], df["Liquidity_Score"] = 0.5, 0.0, 0, 0.0
        if agg_trades_df.empty or "is_buyer_maker" not in agg_trades_df.columns:
            return df
        try:
            if not isinstance(agg_trades_df.index, pd.DatetimeIndex):
                agg_trades_df['timestamp'] = pd.to_datetime(agg_trades_df['timestamp'])
                agg_trades_df = agg_trades_df.set_index('timestamp')

            resample_interval = self.config.get("resample_interval", "1T")
            agg_trades_df["buy_volume"] = np.where(~agg_trades_df["is_buyer_maker"], agg_trades_df["quantity"] * agg_trades_df["price"], 0)
            agg_trades_df["sell_volume"] = np.where(agg_trades_df["is_buyer_maker"], agg_trades_df["quantity"] * agg_trades_df["price"], 0)
            flow_data = agg_trades_df.resample(resample_interval).agg({"buy_volume": "sum", "sell_volume": "sum"}).reindex(df.index, method="ffill").fillna(0)
            
            total_volume = flow_data["buy_volume"] + flow_data["sell_volume"]
            df["Buy_Sell_Pressure_Ratio"] = np.where(total_volume > 0, flow_data["buy_volume"] / total_volume, 0.5)
            df["Order_Flow_Imbalance"] = np.where(total_volume > 0, (flow_data["buy_volume"] - flow_data["sell_volume"]) / total_volume, 0.0)

            large_order_threshold = agg_trades_df["quantity"].quantile(0.95)
            agg_trades_df["is_large_order"] = agg_trades_df["quantity"] > large_order_threshold
            large_order_counts = agg_trades_df["is_large_order"].resample(resample_interval).sum().reindex(df.index, fill_value=0)
            df["Large_Order_Count"] = large_order_counts
            
            trade_counts = agg_trades_df['quantity'].resample(resample_interval).count().reindex(df.index, method="ffill").fillna(0)
            df["Liquidity_Score"] = (trade_counts * total_volume).fillna(0)
        except Exception as e:
            self.logger.error(f"Error calculating order flow indicators: {e}")
        return df

    # --- Utility Methods ---
    def get_feature_statistics(self) -> dict[str, Any]:
        """
        Get statistics about the advanced feature engineering system configuration.
        Returns:
            Dictionary with system statistics
        """
        return {
            "enable_divergence_detection": self.enable_divergence_detection,
            "enable_pattern_recognition": self.enable_pattern_recognition,
            "enable_volume_profile": self.enable_volume_profile,
            "enable_market_microstructure": self.enable_market_microstructure,
            "enable_volatility_targeting": self.enable_volatility_targeting,
            "advanced_config": self.advanced_config,
        }
