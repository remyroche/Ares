# src/analyst/hmm_regime_classifier.py
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from hmmlearn import hmm
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler

from src.config import CONFIG
from src.utils.logger import system_logger


class HMMRegimeClassifier:
    """
    Hidden Markov Model-based Market Regime Classifier for 1-hour timeframe.
    
    This classifier uses HMM to identify underlying market states using:
    1. log_returns - for capturing price movement patterns
    2. volatility_20 - for capturing volatility regimes (adapted for 1h timeframe)
    
    The HMM identifies latent states which are then interpreted and mapped to
    market regimes (BULL, BEAR, SIDEWAYS) with additional technical analysis
            for HUGE_CANDLE and SR_ZONE_ACTION classifications.
    
    IMPORTANT: This classifier is designed to work with 1-hour timeframe data
    for regime classification.
    """

    def __init__(self, config):
        self.config = config.get("analyst", {}).get("hmm_regime_classifier", {})
        self.global_config = config
        self.logger = system_logger.getChild("HMMRegimeClassifier")

        # HMM parameters
        self.n_states = self.config.get("n_states", 4)  # Number of hidden states
        self.n_iter = self.config.get(
            "n_iter",
            100,
        )  # Maximum iterations for HMM training
        self.random_state = self.config.get("random_state", 42)

        # Timeframe-specific parameters for 15m-1h data
        self.target_timeframe = self.config.get("target_timeframe", "1h")  # Use 1h data for regime classification
        self.volatility_period = self.config.get(
            "volatility_period",
            10,
        )  # 10 periods for 15m-1h timeframe (more responsive)
        self.min_data_points = self.config.get(
            "min_data_points",
            1000,  # General minimum data points requirement
        )  # Minimum data points

        # Models
        self.hmm_model = None
        self.scaler = None
        self.lgbm_classifier = None
        self.state_to_regime_map = {}

        # Training status
        self.trained = False
        self.last_training_time = None

        # Model paths
        self.default_model_path = os.path.join(
            CONFIG["CHECKPOINT_DIR"],
            "analyst_models",
            f"hmm_regime_classifier_{self.target_timeframe}.joblib",
        )
        os.makedirs(os.path.dirname(self.default_model_path), exist_ok=True)

    def _validate_timeframe(self, klines_df: pd.DataFrame) -> bool:
        """
        Validate that the data is from the correct timeframe (15m-1h).

        Args:
            klines_df: DataFrame with OHLCV data

        Returns:
            bool: True if timeframe is valid, False otherwise
        """
        if klines_df.empty:
            self.logger.warning("Empty dataframe provided")
            return False

        # Check if we have timestamp index
        if not isinstance(klines_df.index, pd.DatetimeIndex):
            self.logger.warning("DataFrame must have DatetimeIndex")
            return False

        # Estimate timeframe from data
        if len(klines_df) < 2:
            self.logger.warning("Insufficient data to estimate timeframe")
            return False

        # Calculate time difference between consecutive rows
        time_diff = klines_df.index[1] - klines_df.index[0]
        minutes_diff = time_diff.total_seconds() / 60

        # Allow 15m-1h timeframes (13 to 62 minutes)
        if not (13 <= minutes_diff <= 62):
            self.logger.warning(
                f"Data appears to be {minutes_diff:.2f}m timeframe, but HMM classifier requires 15m-1h data",
            )
            return False

        self.logger.info(f"✅ Validated {self.target_timeframe} timeframe data (actual: {minutes_diff:.2f}m)")
        return True

    def _calculate_features(self, klines_df: pd.DataFrame, min_data_points: int = None) -> pd.DataFrame:
        """
        Calculate comprehensive features for regime classification with enhanced NaN handling.
        
        Args:
            klines_df: DataFrame with OHLCV data
            min_data_points: Override minimum data points requirement (default: None, uses general requirement)
        """
        try:
            if klines_df.empty:
                self.logger.warning("⚠️ Empty klines data provided")
                return pd.DataFrame()

            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in klines_df.columns]
            if missing_cols:
                self.logger.warning(f"⚠️ Missing required columns: {missing_cols}")
                return pd.DataFrame()

            # Clean input data - remove rows with NaN in critical columns
            initial_rows = len(klines_df)
            klines_df = klines_df.dropna(subset=required_cols)
            if len(klines_df) < initial_rows:
                self.logger.warning(f"⚠️ Removed {initial_rows - len(klines_df)} rows with NaN in critical columns")

            # Use provided minimum data points or general requirement
            effective_min_data_points = min_data_points if min_data_points is not None else self.min_data_points
            
            if len(klines_df) < effective_min_data_points:
                self.logger.warning(f"Insufficient data after cleaning. Got {len(klines_df)}, need {effective_min_data_points}")
                return pd.DataFrame()

            features = pd.DataFrame(index=klines_df.index)

            # Basic price features with safe calculations
            features["log_returns"] = np.log(klines_df["close"] / klines_df["close"].shift(1))
            features["log_returns"] = features["log_returns"].replace([np.inf, -np.inf], np.nan)
            features["log_returns"] = features["log_returns"].fillna(0)

            # Calculate volatility with better NaN handling
            features["volatility_20"] = features["log_returns"].rolling(window=20, min_periods=1).std()
            features["volatility_20"] = features["volatility_20"].replace([np.inf, -np.inf], 0).fillna(0)

            # Candle size features with safe division
            features["candle_total_size"] = klines_df["high"] - klines_df["low"]
            features["candle_body_size"] = abs(klines_df["close"] - klines_df["open"])
            
            # Safe division for ratios
            total_size_safe = features["candle_total_size"].replace(0, np.nan)
            features["candle_body_ratio"] = features["candle_body_size"] / total_size_safe
            features["candle_body_ratio"] = features["candle_body_ratio"].fillna(0.5)  # Default to 50% if no range

            # Average candle size for comparison
            avg_candle_size = features["candle_total_size"].rolling(window=20, min_periods=1).mean()
            avg_size_safe = avg_candle_size.replace(0, np.nan)
            features["candle_size_ratio"] = features["candle_total_size"] / avg_size_safe
            features["candle_size_ratio"] = features["candle_size_ratio"].fillna(1.0)

            # Price gap features
            features["price_gap"] = abs(klines_df["open"] - klines_df["close"].shift(1))
            features["price_gap"] = features["price_gap"].fillna(0)

            # Volume features
            features["volume_ratio"] = klines_df["volume"] / klines_df["volume"].rolling(window=20, min_periods=1).mean()
            features["volume_ratio"] = features["volume_ratio"].fillna(1.0)

            # Technical indicators with enhanced error handling
            try:
                # RSI
                features["rsi"] = self._calculate_rsi(klines_df["close"])
            except Exception as e:
                self.logger.warning(f"Error calculating RSI: {e}")
                features["rsi"] = pd.Series(45.0, index=klines_df.index)

            try:
                # MACD
                macd_data = self._calculate_macd(klines_df["close"])
                features["macd"] = macd_data["macd"]
                features["macd_signal"] = macd_data["signal"]
                features["macd_histogram"] = macd_data["histogram"]
            except Exception as e:
                self.logger.warning(f"Error calculating MACD: {e}")
                features["macd"] = pd.Series(0.0, index=klines_df.index)
                features["macd_signal"] = pd.Series(0.0, index=klines_df.index)
                features["macd_histogram"] = pd.Series(0.0, index=klines_df.index)

            try:
                # Bollinger Bands
                bb_data = self._calculate_bollinger_bands(klines_df["close"])
                features["bb_position"] = bb_data["position"]
                features["bb_width"] = bb_data["width"]
            except Exception as e:
                self.logger.warning(f"Error calculating Bollinger Bands: {e}")
                features["bb_position"] = pd.Series(0.5, index=klines_df.index)
                features["bb_width"] = pd.Series(0.0, index=klines_df.index)

            try:
                # ATR
                features["atr"] = self._calculate_atr(klines_df)
            except Exception as e:
                self.logger.warning(f"Error calculating ATR: {e}")
                features["atr"] = pd.Series(0.0, index=klines_df.index)

            try:
                # ADX
                adx_data = self._calculate_adx(klines_df)
                features["adx"] = adx_data["adx"]
                features["plus_di"] = adx_data["plus_di"]
                features["minus_di"] = adx_data["minus_di"]
            except Exception as e:
                self.logger.warning(f"Error calculating ADX: {e}")
                features["adx"] = pd.Series(0.0, index=klines_df.index)
                features["plus_di"] = pd.Series(0.0, index=klines_df.index)
                features["minus_di"] = pd.Series(0.0, index=klines_df.index)

            # Candle pattern features
            try:
                pattern_features = self._detect_enhanced_candle_patterns(klines_df)
                for key, value in pattern_features.items():
                    features[key] = value.fillna(0)
            except Exception as e:
                self.logger.warning(f"Error calculating candle patterns: {e}")
                # Add default pattern features
                pattern_cols = ['hammer_ratio', 'shooting_star_ratio', 'engulfing_bull', 'engulfing_bear', 'is_doji']
                for col in pattern_cols:
                    features[col] = pd.Series(0, index=klines_df.index)

            # Multi-timeframe features
            try:
                mtf_features = self._calculate_multi_timeframe_features(klines_df)
                for key, value in mtf_features.items():
                    features[key] = value.fillna(0)
            except Exception as e:
                self.logger.warning(f"Error calculating multi-timeframe features: {e}")

            # Volume profile features
            try:
                vp_features = self._calculate_volume_profile_features(klines_df)
                for key, value in vp_features.items():
                    features[key] = value.fillna(0)
            except Exception as e:
                self.logger.warning(f"Error calculating volume profile features: {e}")

            # Microstructure features
            try:
                micro_features = self._calculate_microstructure_features(klines_df)
                for key, value in micro_features.items():
                    features[key] = value.fillna(0)
            except Exception as e:
                self.logger.warning(f"Error calculating microstructure features: {e}")

            # Volatility regime classification
            features["volatility_regime"] = self._classify_volatility_regime(features["volatility_20"])
            
            # Intelligent NaN/Inf handling - replace with reasonable defaults instead of dropping
            self.logger.info(f"🔧 Cleaning features - initial shape: {features.shape}")
            
            # Replace infinite values with reasonable defaults
            features = features.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN values with reasonable defaults based on feature type
            for col in features.columns:
                if col in ['log_returns', 'volatility_20', 'atr', 'price_gap']:
                    # For volatility/return features, use 0 as default
                    features[col] = features[col].fillna(0)
                elif col in ['rsi']:
                    # For RSI, use 45 as default (slightly bearish to help detect sideways)
                    features[col] = features[col].fillna(45)
                elif col in ['bb_position', 'price_position']:
                    # For position features, use 0.5 as default (middle)
                    features[col] = features[col].fillna(0.5)
                elif col in ['bb_width', 'volume_ratio', 'high_low_ratio', 'open_close_ratio', 
                            'candle_size_ratio', 'candle_body_ratio', 'hammer_ratio', 'shooting_star_ratio']:
                    # For ratio/width features, use 1 as default
                    features[col] = features[col].fillna(1)
                elif col in ['volatility_regime', 'volume_spike', 'is_doji', 'engulfing_bull', 'engulfing_bear'] or col.startswith('pattern_'):
                    # For categorical/binary features and pattern features, use 0 as default
                    features[col] = features[col].fillna(0)
                elif col.startswith('mtf_') or col.startswith('vp_') or col.startswith('micro_'):
                    # For multi-timeframe, volume profile, and microstructure features, use 0 as default
                    features[col] = features[col].fillna(0)
                elif col in ['candle_body_size', 'candle_total_size']:
                    # For size features, use 0 as default
                    features[col] = features[col].fillna(0)
                else:
                    # For any other features, use 0 as default
                    features[col] = features[col].fillna(0)
            
            # Final validation - only drop rows if they still have NaN values after filling
            initial_rows = len(features)
            features = features.dropna()
            dropped_rows = initial_rows - len(features)
            
            if dropped_rows > 0:
                self.logger.warning(f"⚠️ Dropped {dropped_rows} rows with persistent NaN values after filling")

            if len(features) < self.min_data_points:
                self.logger.warning(
                    f"Insufficient data after feature calculation. Got {len(features)}, need {self.min_data_points}",
                )
                return pd.DataFrame()

            self.logger.info(f"✅ Calculated comprehensive features for {len(features)} periods (dropped {dropped_rows} rows)")
            return features

        except Exception as e:
            self.logger.error(f"Error in feature calculation: {e}")
            return pd.DataFrame()

    def _calculate_rsi(self, prices: pd.Series, period: int = 8) -> pd.Series:
        """Calculate RSI (Relative Strength Index)."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            # Handle division by zero and infinite values more robustly
            rs = gain / loss
            rs = rs.replace([np.inf, -np.inf], 0)
            rs = rs.fillna(0)  # Fill NaN with 0 for neutral RSI
            
            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.replace([np.inf, -np.inf], 45)
            rsi = rsi.fillna(45)  # Fill any remaining NaN with slightly bearish RSI
            
            return rsi
        except Exception as e:
            self.logger.warning(f"Error calculating RSI: {e}")
            return pd.Series(45.0, index=prices.index)

    def _calculate_macd(self, prices: pd.Series, fast: int = 6, slow: int = 13, signal: int = 4) -> dict:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal).mean()
            histogram = macd - signal_line
            
            return {
                "macd": macd.fillna(0),
                "signal": signal_line.fillna(0),
                "histogram": histogram.fillna(0)
            }
        except Exception as e:
            self.logger.warning(f"Error calculating MACD: {e}")
            return {
                "macd": pd.Series(0.0, index=prices.index),
                "signal": pd.Series(0.0, index=prices.index),
                "histogram": pd.Series(0.0, index=prices.index)
            }

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 10, std_dev: float = 2) -> dict:
        """Calculate Bollinger Bands."""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            # Calculate position within bands (0 = at lower band, 1 = at upper band)
            band_range = upper_band - lower_band
            band_range = band_range.replace(0, np.nan)  # Avoid division by zero
            position = (prices - lower_band) / band_range
            position = position.replace([np.inf, -np.inf], 0.5)
            position = position.fillna(0.5)
            
            # Calculate band width as percentage of SMA
            width = band_range / sma
            width = width.replace([np.inf, -np.inf], 0)
            width = width.fillna(0)
            
            return {
                "position": position,
                "width": width
            }
        except Exception as e:
            self.logger.warning(f"Error calculating Bollinger Bands: {e}")
            return {
                "position": pd.Series(0.5, index=prices.index),
                "width": pd.Series(0.0, index=prices.index)
            }

    def _calculate_atr(self, df: pd.DataFrame, period: int = 8) -> pd.Series:
        """Calculate ATR (Average True Range)."""
        try:
            high = df["high"]
            low = df["low"]
            close = df["close"]
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            return atr.fillna(0)
        except Exception as e:
            self.logger.warning(f"Error calculating ATR: {e}")
            return pd.Series(0.0, index=df.index)

    def _calculate_adx(self, df: pd.DataFrame, period: int = 8) -> dict:
        """Calculate ADX (Average Directional Index)."""
        try:
            high = df["high"]
            low = df["low"]
            close = df["close"]
            
            # Calculate +DM and -DM
            high_diff = high - high.shift(1)
            low_diff = low.shift(1) - low
            
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            
            # Calculate TR
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Smooth the values with better NaN handling
            tr_rolling = tr.rolling(period).mean()
            plus_dm_rolling = pd.Series(plus_dm, index=df.index).rolling(period).mean()
            minus_dm_rolling = pd.Series(minus_dm, index=df.index).rolling(period).mean()
            
            # Handle division by zero
            plus_di = 100 * plus_dm_rolling / tr_rolling
            minus_di = 100 * minus_dm_rolling / tr_rolling
            
            plus_di = plus_di.replace([np.inf, -np.inf], 0).fillna(0)
            minus_di = minus_di.replace([np.inf, -np.inf], 0).fillna(0)
            
            # Calculate ADX with better division handling
            di_sum = plus_di + minus_di
            di_sum = di_sum.replace(0, np.nan)  # Avoid division by zero
            dx = 100 * abs(plus_di - minus_di) / di_sum
            dx = dx.replace([np.inf, -np.inf], 0).fillna(0)
            adx = dx.rolling(period).mean().fillna(0)
            
            return {
                "adx": adx,
                "dmp": plus_di,
                "dmn": minus_di
            }
        except Exception as e:
            self.logger.warning(f"Error calculating ADX: {e}")
            return {
                "adx": pd.Series(0.0, index=df.index),
                "dmp": pd.Series(0.0, index=df.index),
                "dmn": pd.Series(0.0, index=df.index)
            }

    def _detect_enhanced_candle_patterns(self, df: pd.DataFrame) -> dict:
        """
        Enhanced candle pattern detection for sudden events.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with enhanced candle pattern features
        """
        try:
            patterns = {}
            
            # Calculate basic candle metrics
            body_size = abs(df['close'] - df['open'])
            total_size = df['high'] - df['low']
            body_ratio = body_size / total_size.replace(0, np.nan)
            body_ratio = body_ratio.fillna(0.5)
            
            # Upper and lower shadows
            upper_shadow = np.maximum(df['open'], df['close']) - df['high']
            lower_shadow = df['low'] - np.minimum(df['open'], df['close'])
            
            # 1. Marubozu (strong directional candle with minimal shadows)
            marubozu_bull = ((body_ratio > 0.8) & (df['close'] > df['open']) & 
                            (upper_shadow < total_size * 0.1) & (lower_shadow < total_size * 0.1))
            marubozu_bear = ((body_ratio > 0.8) & (df['close'] < df['open']) & 
                            (upper_shadow < total_size * 0.1) & (lower_shadow < total_size * 0.1))
            
            # 2. Hammer/Shooting Star detection
            hammer = ((lower_shadow > body_size * 2) & (upper_shadow < body_size * 0.5) & 
                     (body_ratio < 0.3))
            shooting_star = ((upper_shadow > body_size * 2) & (lower_shadow < body_size * 0.5) & 
                           (body_ratio < 0.3))
            
            # 3. Doji patterns (indecision)
            doji = body_ratio < 0.1
            
            # 4. Engulfing patterns (sudden reversals)
            prev_body = body_size.shift(1)
            engulfing_bull = ((body_size > prev_body * 1.5) & (df['close'] > df['open']) & 
                             (df['open'].shift(1) > df['close'].shift(1)))
            engulfing_bear = ((body_size > prev_body * 1.5) & (df['close'] < df['open']) & 
                             (df['open'].shift(1) < df['close'].shift(1)))
            
            # 5. Gap detection
            gap_up = df['low'] > df['high'].shift(1)
            gap_down = df['high'] < df['low'].shift(1)
            
            # 6. Volume confirmation for sudden events
            if 'volume' in df.columns:
                avg_volume = df['volume'].rolling(20).mean()
                volume_spike = df['volume'] > avg_volume * 2.0
            else:
                volume_spike = pd.Series(False, index=df.index)
            
            # 7. Price momentum confirmation
            price_change = df['close'].pct_change()
            strong_move = abs(price_change) > 0.02  # 2% move
            
            # 8. Volatility spike detection
            atr = self._calculate_atr(df)
            avg_atr = atr.rolling(20).mean()
            volatility_spike = atr > avg_atr * 2.0
            
            patterns = {
                'marubozu_bull': marubozu_bull.astype(int),
                'marubozu_bear': marubozu_bear.astype(int),
                'hammer': hammer.astype(int),
                'shooting_star': shooting_star.astype(int),
                'doji': doji.astype(int),
                'engulfing_bull': engulfing_bull.astype(int),
                'engulfing_bear': engulfing_bear.astype(int),
                'gap_up': gap_up.astype(int),
                'gap_down': gap_down.astype(int),
                'volume_spike': volume_spike.astype(int),
                'strong_move': strong_move.astype(int),
                'volatility_spike': volatility_spike.astype(int),
                'body_ratio': body_ratio,
                'upper_shadow_ratio': upper_shadow / total_size.replace(0, np.nan),
                'lower_shadow_ratio': lower_shadow / total_size.replace(0, np.nan)
            }
            
            # Fill NaN values
            for key, value in patterns.items():
                if isinstance(value, pd.Series):
                    patterns[key] = value.fillna(0)
            
            return patterns
            
        except Exception as e:
            self.logger.warning(f"Error in enhanced candle pattern detection: {e}")
            return {}

    def _analyze_candle_trading_action(self, current_candle: pd.Series, recent_context: pd.DataFrame, 
                                     rsi: float, atr: float, gap_size: float, body_ratio: float) -> str:
        """
        Analyze candle patterns and provide specific trading recommendations.
        
        Args:
            current_candle: Current candle data
            recent_context: Recent candles for context
            rsi: Current RSI value
            atr: Current ATR value
            gap_size: Gap size from previous candle
            body_ratio: Body to total size ratio
            
        Returns:
            Trading action: 'RIDE', 'AVOID', 'WAIT_FALLBACK'
        """
        try:
            # Extract current candle characteristics
            open_price = current_candle['open']
            close_price = current_candle['close']
            high_price = current_candle['high']
            low_price = current_candle['low']
            volume = current_candle.get('volume', 0)
            
            # Determine candle direction
            is_bullish = close_price > open_price
            is_bearish = close_price < open_price
            
            # Calculate additional metrics
            candle_size = high_price - low_price
            body_size = abs(close_price - open_price)
            
            # Get recent context metrics
            if len(recent_context) > 0:
                avg_recent_volume = recent_context['volume'].mean() if 'volume' in recent_context.columns else volume
                avg_recent_size = (recent_context['high'] - recent_context['low']).mean()
                recent_trend = recent_context['close'].iloc[-1] - recent_context['close'].iloc[0] if len(recent_context) > 1 else 0
            else:
                avg_recent_volume = volume
                avg_recent_size = candle_size
                recent_trend = 0
            
            # Volume analysis
            volume_spike = volume > avg_recent_volume * 1.5 if avg_recent_volume > 0 else False
            volume_suppression = volume < avg_recent_volume * 0.5 if avg_recent_volume > 0 else False
            
            # RSI analysis
            rsi_overbought = rsi > 70
            rsi_oversold = rsi < 30
            rsi_neutral = 30 <= rsi <= 70
            
            # Gap analysis
            significant_gap = gap_size > 0.02  # 2% gap
            moderate_gap = 0.005 < gap_size <= 0.02  # 0.5-2% gap
            
            # Body strength analysis
            strong_body = body_ratio > 0.7
            moderate_body = 0.4 <= body_ratio <= 0.7
            weak_body = body_ratio < 0.4
            
            # ATR analysis (volatility context)
            high_volatility = atr > recent_context['close'].std() * 2 if len(recent_context) > 0 else False
            low_volatility = atr < recent_context['close'].std() * 0.5 if len(recent_context) > 0 else False
            
            # Multi-timeframe confirmation analysis
            mtf_consensus = recent_context.get('mtf_consensus', 0) if hasattr(recent_context, 'get') else 0
            mtf_consensus_strength = recent_context.get('mtf_consensus_strength', 0) if hasattr(recent_context, 'get') else 0
            
            # Volume profile analysis
            vp_current_vs_poc = recent_context.get('vp_current_vs_poc', 0) if hasattr(recent_context, 'get') else 0
            vp_volume_trend = recent_context.get('vp_volume_trend', 0) if hasattr(recent_context, 'get') else 0
            
            # Market microstructure analysis
            micro_market_efficiency = recent_context.get('micro_market_efficiency', 0) if hasattr(recent_context, 'get') else 0
            micro_order_imbalance = recent_context.get('micro_order_imbalance', 0) if hasattr(recent_context, 'get') else 0
            
            # Decision Matrix for Trading Actions
            
            # 1. RIDE Conditions (Strong bullish/bearish momentum with confirmation)
            # Enhanced with multi-timeframe, volume profile, and microstructure confirmation
            if (is_bullish and strong_body and volume_spike and rsi_neutral and not rsi_overbought and 
                mtf_consensus > 0.3 and vp_current_vs_poc == 1):
                return "RIDE"  # Strong bullish with multi-timeframe and POC confirmation
            elif (is_bearish and strong_body and volume_spike and rsi_neutral and not rsi_oversold and 
                  mtf_consensus < -0.3 and vp_current_vs_poc == 1):
                return "RIDE"  # Strong bearish with multi-timeframe and POC confirmation
            elif (is_bullish and significant_gap and volume_spike and rsi < 60 and 
                  mtf_consensus > 0.2 and micro_market_efficiency > 0.6):
                return "RIDE"  # Gap up with efficiency and consensus
            elif (is_bearish and significant_gap and volume_spike and rsi > 40 and 
                  mtf_consensus < -0.2 and micro_market_efficiency > 0.6):
                return "RIDE"  # Gap down with efficiency and consensus
            elif (is_bullish and strong_body and high_volatility and volume_spike and 
                  mtf_consensus > 0.1 and vp_volume_trend > 0):
                return "RIDE"  # Volatility spike with volume trend confirmation
            elif (is_bearish and strong_body and high_volatility and volume_spike and 
                  mtf_consensus < -0.1 and vp_volume_trend < 0):
                return "RIDE"  # Volatility spike with volume trend confirmation
            
            # 2. AVOID Conditions (High risk or conflicting signals)
            # Enhanced with multi-timeframe, volume profile, and microstructure analysis
            elif (rsi_overbought and is_bullish and mtf_consensus < 0):
                return "AVOID"  # Overbought bullish candle with conflicting timeframes
            elif (rsi_oversold and is_bearish and mtf_consensus > 0):
                return "AVOID"  # Oversold bearish candle with conflicting timeframes
            elif (volume_suppression and strong_body and vp_volume_trend < -0.1):
                return "AVOID"  # Strong move without volume confirmation and declining volume
            elif (moderate_gap and weak_body and micro_market_efficiency < 0.4):
                return "AVOID"  # Gap without strong follow-through and low efficiency
            elif (is_bullish and rsi_overbought and volume_spike and mtf_consensus < -0.2):
                return "AVOID"  # Overbought with volume spike and bearish consensus
            elif (is_bearish and rsi_oversold and volume_spike and mtf_consensus > 0.2):
                return "AVOID"  # Oversold with volume spike and bullish consensus
            elif (abs(recent_trend) > candle_size * 2 and is_bullish and recent_trend < 0 and mtf_consensus < -0.3):
                return "AVOID"  # Bullish candle against strong downtrend with bearish consensus
            elif (abs(recent_trend) > candle_size * 2 and is_bearish and recent_trend > 0 and mtf_consensus > 0.3):
                return "AVOID"  # Bearish candle against strong uptrend with bullish consensus
            elif (is_bullish and vp_current_vs_poc == 0 and micro_order_imbalance < -0.3):
                return "AVOID"  # Bullish candle away from POC with bearish order flow
            elif (is_bearish and vp_current_vs_poc == 0 and micro_order_imbalance > 0.3):
                return "AVOID"  # Bearish candle away from POC with bullish order flow
            
            # 3. WAIT_FALLBACK Conditions (Uncertain or mixed signals)
            # Enhanced with multi-timeframe, volume profile, and microstructure analysis
            elif (moderate_body and volume_spike and rsi_neutral and abs(mtf_consensus) < 0.2):
                return "WAIT_FALLBACK"  # Moderate strength with volume but weak consensus
            elif (strong_body and not volume_spike and rsi_neutral and vp_current_vs_poc == 1):
                return "WAIT_FALLBACK"  # Strong body but no volume confirmation at POC
            elif (significant_gap and weak_body and micro_market_efficiency > 0.5):
                return "WAIT_FALLBACK"  # Gap but weak follow-through with decent efficiency
            elif (moderate_gap and moderate_body and abs(mtf_consensus) < 0.3):
                return "WAIT_FALLBACK"  # Moderate gap and body with mixed consensus
            elif (low_volatility and strong_body and vp_volume_trend > 0):
                return "WAIT_FALLBACK"  # Strong move in low volatility with volume trend
            elif (rsi_neutral and moderate_body and not volume_spike and abs(micro_order_imbalance) < 0.2):
                return "WAIT_FALLBACK"  # Neutral conditions with balanced order flow
            elif (is_bullish and mtf_consensus > 0.1 and vp_current_vs_poc == 0):
                return "WAIT_FALLBACK"  # Bullish with consensus but away from POC
            elif (is_bearish and mtf_consensus < -0.1 and vp_current_vs_poc == 0):
                return "WAIT_FALLBACK"  # Bearish with consensus but away from POC
            
            # Default fallback
            else:
                return "WAIT_FALLBACK"
                
        except Exception as e:
            self.logger.warning(f"Error in candle trading action analysis: {e}")
            return "WAIT_FALLBACK"

    def _calculate_multi_timeframe_features(self, df: pd.DataFrame) -> dict:
        """
        Calculate multi-timeframe confirmation features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with multi-timeframe features
        """
        try:
            features = {}
            
            # Resample to different timeframes for confirmation
            timeframes = {
                '15m': '15T',
                '30m': '30T', 
                '4h': '4H',
                '1d': 'D'
            }
            
            for tf_name, tf_code in timeframes.items():
                try:
                    # Resample data to target timeframe
                    resampled = df.resample(tf_code).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()
                    
                    if len(resampled) > 10:
                        # Calculate trend alignment
                        current_trend = resampled['close'].iloc[-1] - resampled['close'].iloc[-5] if len(resampled) >= 5 else 0
                        longer_trend = resampled['close'].iloc[-1] - resampled['close'].iloc[-10] if len(resampled) >= 10 else 0
                        
                        # Trend strength
                        trend_strength = abs(current_trend) / resampled['close'].std() if resampled['close'].std() > 0 else 0
                        
                        # Volume trend alignment
                        volume_trend = resampled['volume'].iloc[-5:].mean() - resampled['volume'].iloc[-10:-5].mean() if len(resampled) >= 10 else 0
                        
                        features[f'mtf_{tf_name}_trend_alignment'] = np.sign(current_trend) * np.sign(longer_trend)
                        features[f'mtf_{tf_name}_trend_strength'] = trend_strength
                        features[f'mtf_{tf_name}_volume_trend'] = volume_trend
                        features[f'mtf_{tf_name}_price_momentum'] = current_trend / resampled['close'].iloc[-5] if resampled['close'].iloc[-5] > 0 else 0
                        
                    else:
                        # Default values if insufficient data
                        features[f'mtf_{tf_name}_trend_alignment'] = 0
                        features[f'mtf_{tf_name}_trend_strength'] = 0
                        features[f'mtf_{tf_name}_volume_trend'] = 0
                        features[f'mtf_{tf_name}_price_momentum'] = 0
                        
                except Exception as e:
                    self.logger.warning(f"Error calculating {tf_name} timeframe features: {e}")
                    features[f'mtf_{tf_name}_trend_alignment'] = 0
                    features[f'mtf_{tf_name}_trend_strength'] = 0
                    features[f'mtf_{tf_name}_volume_trend'] = 0
                    features[f'mtf_{tf_name}_price_momentum'] = 0
            
            # Multi-timeframe consensus
            trend_alignments = [features.get(f'mtf_{tf}_trend_alignment', 0) for tf in timeframes.keys()]
            features['mtf_consensus'] = np.mean(trend_alignments) if trend_alignments else 0
            features['mtf_consensus_strength'] = np.std(trend_alignments) if len(trend_alignments) > 1 else 0
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Error in multi-timeframe feature calculation: {e}")
            return {}

    def _calculate_volume_profile_features(self, df: pd.DataFrame) -> dict:
        """
        Calculate volume profile analysis features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with volume profile features
        """
        try:
            features = {}
            
            if 'volume' not in df.columns:
                return features
            
            # Calculate volume profile metrics
            price_levels = np.linspace(df['low'].min(), df['high'].max(), 50)
            volume_profile = np.zeros(len(price_levels))
            
            for i in range(len(df)):
                price = df['close'].iloc[i]
                volume = df['volume'].iloc[i]
                
                # Find closest price level
                closest_level = np.argmin(np.abs(price_levels - price))
                volume_profile[closest_level] += volume
            
            # Volume profile analysis
            if np.sum(volume_profile) > 0:
                # High Volume Nodes (HVNs)
                mean_volume = np.mean(volume_profile)
                hvn_threshold = mean_volume * 1.5
                hvn_count = np.sum(volume_profile > hvn_threshold)
                
                # Point of Control (POC) - price level with highest volume
                poc_index = np.argmax(volume_profile)
                poc_price = price_levels[poc_index]
                
                # Current price position relative to POC
                current_price = df['close'].iloc[-1]
                poc_distance = (current_price - poc_price) / poc_price if poc_price > 0 else 0
                
                # Volume distribution skewness
                volume_skew = np.sum((volume_profile - mean_volume) ** 3) / (len(volume_profile) * np.std(volume_profile) ** 3) if np.std(volume_profile) > 0 else 0
                
                # Value Area (70% of volume)
                total_volume = np.sum(volume_profile)
                target_volume = total_volume * 0.7
                sorted_indices = np.argsort(volume_profile)[::-1]
                cumulative_volume = 0
                value_area_count = 0
                
                for idx in sorted_indices:
                    cumulative_volume += volume_profile[idx]
                    value_area_count += 1
                    if cumulative_volume >= target_volume:
                        break
                
                features['vp_hvn_count'] = hvn_count
                features['vp_poc_distance'] = poc_distance
                features['vp_volume_skew'] = volume_skew
                features['vp_value_area_ratio'] = value_area_count / len(volume_profile)
                features['vp_current_vs_poc'] = 1 if abs(poc_distance) < 0.02 else 0  # Within 2% of POC
                
            else:
                features['vp_hvn_count'] = 0
                features['vp_poc_distance'] = 0
                features['vp_volume_skew'] = 0
                features['vp_value_area_ratio'] = 0
                features['vp_current_vs_poc'] = 0
            
            # Volume trend analysis
            recent_volume = df['volume'].iloc[-20:].mean() if len(df) >= 20 else df['volume'].mean()
            older_volume = df['volume'].iloc[-40:-20].mean() if len(df) >= 40 else recent_volume
            
            features['vp_volume_trend'] = (recent_volume - older_volume) / older_volume if older_volume > 0 else 0
            features['vp_volume_consistency'] = 1 - (df['volume'].iloc[-10:].std() / df['volume'].iloc[-10:].mean()) if df['volume'].iloc[-10:].mean() > 0 else 0
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Error in volume profile feature calculation: {e}")
            return {}

    def _calculate_microstructure_features(self, df: pd.DataFrame) -> dict:
        """
        Calculate market microstructure features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with microstructure features
        """
        try:
            features = {}
            
            # Bid-ask spread approximation (using high-low as proxy)
            spread_ratio = (df['high'] - df['low']) / df['close']
            features['micro_avg_spread'] = spread_ratio.mean()
            features['micro_spread_volatility'] = spread_ratio.std()
            features['micro_spread_trend'] = spread_ratio.iloc[-5:].mean() - spread_ratio.iloc[-10:-5].mean() if len(spread_ratio) >= 10 else 0
            
            # Price impact analysis
            if 'volume' in df.columns:
                # Price change per unit volume
                price_changes = df['close'].diff().abs()
                volume_impact = price_changes / df['volume'].replace(0, 1)
                features['micro_price_impact'] = volume_impact.mean()
                features['micro_impact_volatility'] = volume_impact.std()
                
                # Large trade detection (high volume periods)
                volume_threshold = df['volume'].quantile(0.8)
                large_trade_ratio = (df['volume'] > volume_threshold).mean()
                features['micro_large_trade_ratio'] = large_trade_ratio
                
                # Volume clustering
                volume_clusters = (df['volume'] > df['volume'].rolling(5).mean()).rolling(3).sum()
                features['micro_volume_clustering'] = volume_clusters.mean()
            
            # Liquidity analysis
            # High-low range as liquidity proxy
            liquidity_ratio = (df['high'] - df['low']) / df['close']
            features['micro_liquidity_avg'] = liquidity_ratio.mean()
            features['micro_liquidity_trend'] = liquidity_ratio.iloc[-5:].mean() - liquidity_ratio.iloc[-10:-5].mean() if len(liquidity_ratio) >= 10 else 0
            
            # Market efficiency (price efficiency ratio)
            price_changes = df['close'].diff()
            price_efficiency = abs(price_changes.sum()) / price_changes.abs().sum() if price_changes.abs().sum() > 0 else 0
            features['micro_market_efficiency'] = price_efficiency
            
            # Volatility clustering
            returns = df['close'].pct_change()
            volatility_clustering = returns.rolling(5).std().rolling(3).std()
            features['micro_volatility_clustering'] = volatility_clustering.mean()
            
            # Order flow imbalance (approximation using volume and price direction)
            if 'volume' in df.columns:
                bullish_volume = df['volume'].where(df['close'] > df['open'], 0)
                bearish_volume = df['volume'].where(df['close'] < df['open'], 0)
                order_imbalance = (bullish_volume - bearish_volume) / (bullish_volume + bearish_volume).replace(0, 1)
                features['micro_order_imbalance'] = order_imbalance.mean()
                features['micro_imbalance_volatility'] = order_imbalance.std()
            
            # Market depth approximation (using volume and price range)
            if 'volume' in df.columns:
                depth_ratio = df['volume'] / (df['high'] - df['low']).replace(0, 1)
                features['micro_market_depth'] = depth_ratio.mean()
                features['micro_depth_trend'] = depth_ratio.iloc[-5:].mean() - depth_ratio.iloc[-10:-5].mean() if len(depth_ratio) >= 10 else 0
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Error in microstructure feature calculation: {e}")
            return {}

    def _get_candle_trading_explanation(self, action: str, candle_data: dict) -> str:
        """
        Provide detailed explanation for trading action recommendations.
        
        Args:
            action: Trading action ('RIDE', 'AVOID', 'WAIT_FALLBACK')
            candle_data: Dictionary with candle analysis data
            
        Returns:
            Detailed explanation string
        """
        explanations = {
            "RIDE": {
                "bullish_strong": "Strong bullish momentum with volume confirmation and neutral RSI - ideal for riding the trend",
                "bearish_strong": "Strong bearish momentum with volume confirmation and neutral RSI - ideal for riding the trend",
                "bullish_gap": "Significant bullish gap with volume spike and favorable RSI - strong breakout signal",
                "bearish_gap": "Significant bearish gap with volume spike and favorable RSI - strong breakdown signal",
                "volatility_spike": "High volatility move with strong body and volume - momentum continuation likely"
            },
            "AVOID": {
                "overbought_bullish": "Bullish candle in overbought conditions - high reversal risk",
                "oversold_bearish": "Bearish candle in oversold conditions - high reversal risk",
                "no_volume": "Strong move without volume confirmation - weak signal",
                "gap_no_follow": "Gap without strong follow-through - potential false signal",
                "trend_conflict": "Candle direction conflicts with strong trend - likely to fail",
                "reversal_signal": "Volume spike in extreme RSI conditions - potential reversal"
            },
            "WAIT_FALLBACK": {
                "moderate_strength": "Moderate strength with mixed signals - wait for confirmation",
                "no_volume": "Strong body but no volume confirmation - wait for volume",
                "weak_follow": "Gap with weak follow-through - wait for stronger signal",
                "low_volatility": "Strong move in low volatility - wait for volatility increase",
                "neutral_conditions": "Neutral conditions with moderate signals - wait for clearer direction"
            }
        }
        
        # Determine specific reason based on candle characteristics
        if action == "RIDE":
            if candle_data.get('is_bullish', False) and candle_data.get('strong_body', False):
                return explanations["RIDE"]["bullish_strong"]
            elif candle_data.get('is_bearish', False) and candle_data.get('strong_body', False):
                return explanations["RIDE"]["bearish_strong"]
            elif candle_data.get('significant_gap', False) and candle_data.get('is_bullish', False):
                return explanations["RIDE"]["bullish_gap"]
            elif candle_data.get('significant_gap', False) and candle_data.get('is_bearish', False):
                return explanations["RIDE"]["bearish_gap"]
            else:
                return explanations["RIDE"]["volatility_spike"]
                
        elif action == "AVOID":
            if candle_data.get('rsi_overbought', False) and candle_data.get('is_bullish', False):
                return explanations["AVOID"]["overbought_bullish"]
            elif candle_data.get('rsi_oversold', False) and candle_data.get('is_bearish', False):
                return explanations["AVOID"]["oversold_bearish"]
            elif candle_data.get('volume_suppression', False):
                return explanations["AVOID"]["no_volume"]
            elif candle_data.get('moderate_gap', False) and candle_data.get('weak_body', False):
                return explanations["AVOID"]["gap_no_follow"]
            elif candle_data.get('trend_conflict', False):
                return explanations["AVOID"]["trend_conflict"]
            else:
                return explanations["AVOID"]["reversal_signal"]
                
        else:  # WAIT_FALLBACK
            if candle_data.get('moderate_body', False) and candle_data.get('volume_spike', False):
                return explanations["WAIT_FALLBACK"]["moderate_strength"]
            elif candle_data.get('strong_body', False) and not candle_data.get('volume_spike', False):
                return explanations["WAIT_FALLBACK"]["no_volume"]
            elif candle_data.get('significant_gap', False) and candle_data.get('weak_body', False):
                return explanations["WAIT_FALLBACK"]["weak_follow"]
            elif candle_data.get('low_volatility', False):
                return explanations["WAIT_FALLBACK"]["low_volatility"]
            else:
                return explanations["WAIT_FALLBACK"]["neutral_conditions"]

    def _classify_volatility_regime(self, volatility: pd.Series) -> pd.Series:
        """Classify volatility into regimes (low, medium, high)."""
        try:
            # Define volatility thresholds
            low_threshold = volatility.quantile(0.33)
            high_threshold = volatility.quantile(0.67)
            
            regime = pd.Series(index=volatility.index, dtype='object')
            regime[volatility <= low_threshold] = 0  # Low volatility
            regime[(volatility > low_threshold) & (volatility <= high_threshold)] = 1  # Medium volatility
            regime[volatility > high_threshold] = 2  # High volatility
            
            return regime.fillna(1)  # Default to medium volatility
        except Exception as e:
            self.logger.warning(f"Error classifying volatility regime: {e}")
            return pd.Series(1, index=volatility.index)  # Default to medium volatility

    def _interpret_hmm_states(
        self,
        features_df: pd.DataFrame,
        state_sequence: np.ndarray,
    ) -> dict:
        """
        Interpret HMM states and map them to market regimes.

        Args:
            features_df: DataFrame with log_returns and volatility_20
            state_sequence: Array of predicted state labels

        Returns:
            Dictionary mapping state indices to regime labels
        """
        # Create DataFrame with states
        analysis_df = features_df.copy()
        analysis_df["state"] = state_sequence

        # Analyze each state's characteristics
        state_analysis = {}

        for state in range(self.n_states):
            state_data = analysis_df[analysis_df["state"] == state]

            if len(state_data) == 0:
                continue

            # Calculate state characteristics
            mean_return = state_data["log_returns"].mean()
            mean_volatility = state_data["volatility_20"].mean()
            std_return = state_data["log_returns"].std()
            std_volatility = state_data["volatility_20"].std()

            # Determine regime based on characteristics (BULL, SIDEWAYS, or BEAR)
            regime = self._classify_state_to_regime(
                mean_return,
                mean_volatility,
                std_return,
                std_volatility,
            )
            


            state_analysis[state] = {
                "regime": regime,
                "mean_return": mean_return,
                "mean_volatility": mean_volatility,
                "std_return": std_return,
                "std_volatility": std_volatility,
                "count": len(state_data),
            }

            self.logger.info(
                f"State {state}: {regime} "
                f"(mean_return={mean_return:.4f}, mean_vol={mean_volatility:.4f})",
            )

        # Create state_to_regime_map for easy lookup
        state_to_regime_map = {state: data["regime"] for state, data in state_analysis.items()}
        state_analysis["state_to_regime_map"] = state_to_regime_map

        return state_analysis

    def _classify_state_to_regime(
        self,
        mean_return: float,
        mean_volatility: float,
        std_return: float,
        std_volatility: float,
    ) -> str:
        """
        Classify a state's characteristics into basic market regimes:
        BULL, BEAR, or SIDEWAYS.

        Args:
            mean_return: Mean log return for the state
            mean_volatility: Mean volatility for the state
            std_return: Standard deviation of returns for the state
            std_volatility: Standard deviation of volatility for the state

        Returns:
            Market regime label (BULL, BEAR, or SIDEWAYS)
        """
        # Define thresholds for hourly returns (much smaller than daily)
        # For hourly data, typical returns are 0.1-0.5% per hour
        return_threshold = self.config.get(
            "return_threshold",
            0.0001,  # 0.01% hourly return - very small threshold to capture more regimes
        )
        
        # Also consider volatility for classification
        volatility_threshold = self.config.get(
            "volatility_threshold",
            0.5,  # Medium volatility threshold
        )

        # Classify based on mean return and volatility
        # BULL: Positive returns with moderate volatility
        if mean_return > return_threshold and mean_volatility < volatility_threshold:
            return "BULL"
        
        # BEAR: Negative returns with moderate volatility
        if mean_return < -return_threshold and mean_volatility < volatility_threshold:
            return "BEAR"
        
        # HIGH_VOLATILITY: High volatility periods (could be either bull or bear)
        if mean_volatility > volatility_threshold:
            if mean_return > return_threshold:
                return "BULL"
            elif mean_return < -return_threshold:
                return "BEAR"
            else:
                return "SIDEWAYS"

        # SIDEWAYS: Neutral conditions (weak trending, low volatility)
        return "SIDEWAYS"

    def train_classifier(
        self,
        historical_klines: pd.DataFrame,
        model_path: str | None = None,
    ) -> bool:
        """
        Train the HMM-based regime classifier.

        Args:
            historical_klines: Historical OHLCV data
            model_path: Optional path to save the trained model

        Returns:
            True if training successful, False otherwise
        """
        self.logger.info("Training HMM-based Market Regime Classifier...")

        if historical_klines.empty:
            self.logger.warning("No historical klines provided for training")
            return False

        # Calculate features
        features_df = self._calculate_features(historical_klines)

        if features_df.empty:
            self.logger.warning("No features could be calculated from historical data")
            return False

        # Prepare features for HMM (only log_returns and volatility_20)
        hmm_features = features_df[["log_returns", "volatility_20"]].values

        # Remove any infinite values
        hmm_features = hmm_features[np.isfinite(hmm_features).all(axis=1)]

        if len(hmm_features) < self.n_states * 10:  # Need sufficient data
            self.logger.warning(
                f"Insufficient data for HMM training. Need at least {self.n_states * 10} samples",
            )
            return False

        # Scale features for HMM
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(hmm_features)

        # Train HMM
        try:
            self.hmm_model = hmm.GaussianHMM(
                n_components=self.n_states,
                n_iter=self.n_iter,
                random_state=self.random_state,
                covariance_type="full",
            )

            # Fit the HMM
            self.hmm_model.fit(scaled_features)

            # Get state sequence for the training data
            state_sequence = self.hmm_model.predict(scaled_features)

            # Interpret states and create regime mapping
            state_analysis = self._interpret_hmm_states(
                features_df.iloc[: len(scaled_features)],
                state_sequence,
            )

            # Create state to regime mapping
            self.state_to_regime_map = {}
            for state, analysis in state_analysis.items():
                if isinstance(analysis, dict) and "regime" in analysis:
                    self.state_to_regime_map[state] = analysis["regime"]
                else:
                    # Fallback: use state number as regime
                    self.state_to_regime_map[state] = f"STATE_{state}"

            # Create superior simulated labels using HMM states
            superior_labels = []
            for state in state_sequence:
                superior_labels.append(
                    self.state_to_regime_map.get(state, "SIDEWAYS"),
                )

            # Train LightGBM classifier on HMM-derived labels
            # Ensure the features DataFrame has the same length as the labels
            features_for_lgbm = features_df.iloc[: len(scaled_features)].copy()
            self._train_lgbm_classifier(
                features_for_lgbm,
                superior_labels,
            )

            self.trained = True
            self.last_training_time = datetime.now()
            self.logger.info(
                "HMM-based regime classifier training completed successfully",
            )

            # Save model if path provided
            if model_path:
                self.save_model(model_path)

            return True

        except Exception as e:
            self.logger.error(f"Error training HMM model: {e}", exc_info=True)
            self.trained = False
            return False

    def _train_lgbm_classifier(
        self,
        features_df: pd.DataFrame,
        labels: list[str],
    ) -> None:
        """
        Train LightGBM classifier using HMM-derived labels with comprehensive features.

        Args:
            features_df: Features DataFrame with comprehensive technical indicators
            labels: List of regime labels from HMM interpretation
        """
        try:
            # Prepare features for LightGBM with comprehensive feature set
            lgbm_features = features_df.copy()

            # Define comprehensive feature set
            comprehensive_features = [
                # Basic price features
                "log_returns", "volatility_20",
                
                # Price momentum features
                "price_momentum_5", "price_momentum_10", "price_momentum_20",
                
                # Volume features
                "volume_ratio", "volume_momentum",
                
                # Technical indicators
                "rsi", "macd", "macd_signal", "macd_histogram",
                "bb_position", "bb_width", "atr", "adx", "dmp", "dmn",
                
                # Market structure features
                "high_low_ratio", "open_close_ratio", "price_position",
                
                # Volatility regime
                "volatility_regime"
            ]

            # Filter available features
            available_features = [
                col for col in comprehensive_features if col in lgbm_features.columns
            ]

            if len(available_features) < 2:
                self.logger.warning("Insufficient features for LightGBM training")
                return

            self.logger.info(f"Using {len(available_features)} features for LightGBM training: {available_features}")

            # Train LightGBM with comprehensive features
            self.lgbm_classifier = LGBMClassifier(
                random_state=42,
                verbose=-1,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                num_leaves=31,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
            )

            # Use comprehensive features
            X = lgbm_features[available_features].values
            y = np.array(labels)

            # Remove any rows with NaN values
            valid_mask = ~np.isnan(X).any(axis=1)
            X = X[valid_mask]
            y = y[valid_mask]

            if len(X) > 0:
                self.lgbm_classifier.fit(X, y)
                self.logger.info(f"LightGBM classifier trained on {len(X)} samples with {len(available_features)} features")
                
                # Log feature importance
                feature_importance = dict(zip(available_features, self.lgbm_classifier.feature_importances_))
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                self.logger.info(f"Top 5 feature importances: {top_features}")
            else:
                self.logger.warning("No valid data for LightGBM training")
                self.lgbm_classifier = None

        except Exception as e:
            self.logger.error(f"Error training LightGBM classifier: {e}", exc_info=True)
            self.lgbm_classifier = None

    def predict_regime(self, current_klines: pd.DataFrame) -> tuple[str, float, dict]:
        """
        Predict the current market regime using HMM and LightGBM.

        Args:
            current_klines: Current OHLCV data

        Returns:
            Tuple of (regime, confidence, additional_info)
        """
        if not self.trained:
            self.logger.warning(
                "HMM classifier not trained. Attempting to load model...",
            )
            if not self.load_model():
                return "UNKNOWN", 0.0, {}

        # Ensure HMM model is available
        if self.hmm_model is None:
            self.logger.warning("HMM model not available for prediction")
            return "UNKNOWN", 0.0, {}

        # Calculate current features
        current_features = self._calculate_features(current_klines)

        if current_features.empty:
            return "UNKNOWN", 0.0, {}

        # Get the most recent data point with comprehensive features
        comprehensive_features = [
            "log_returns", "volatility_20", "price_momentum_5", "price_momentum_10", 
            "price_momentum_20", "volume_ratio", "volume_momentum", "rsi", "macd", 
            "macd_signal", "macd_histogram", "bb_position", "bb_width", "atr", 
            "adx", "dmp", "dmn", "high_low_ratio", "open_close_ratio", 
            "price_position", "volatility_regime"
        ]
        
        # Filter available features
        available_features = [
            col for col in comprehensive_features if col in current_features.columns
        ]
        
        if len(available_features) < 2:
            self.logger.warning("Insufficient features for prediction")
            return "UNKNOWN", 0.0, {}
        
        latest_features = current_features[available_features].tail(1).values

        # Remove any infinite values
        if not np.isfinite(latest_features).all():
            return "UNKNOWN", 0.0, {}

        # Ensure we have valid features
        if latest_features.size == 0 or np.isnan(latest_features).any():
            return "UNKNOWN", 0.0, {}

        # Scale features (use only basic features for HMM)
        if self.scaler is None:
            self.logger.warning("Scaler not available for prediction")
            return "UNKNOWN", 0.0, {}

        # For HMM, use only basic features (log_returns, volatility_20)
        basic_features = ["log_returns", "volatility_20"]
        basic_available = [col for col in basic_features if col in available_features]
        
        if len(basic_available) < 2:
            self.logger.warning("Insufficient basic features for HMM prediction")
            return "UNKNOWN", 0.0, {}
        
        hmm_features = current_features[basic_available].tail(1).values
        scaled_features = self.scaler.transform(hmm_features)

        # Predict HMM state
        predicted_state = self.hmm_model.predict(scaled_features)[0]

        # Get regime from state mapping
        if not self.state_to_regime_map:
            self.logger.warning("State to regime mapping not available")
            return "UNKNOWN", 0.0, {}

        hmm_regime = self.state_to_regime_map.get(predicted_state, "SIDEWAYS")

        # Get LightGBM prediction if available
        lgbm_regime = "UNKNOWN"
        confidence = 0.0

        if self.lgbm_classifier is not None and hasattr(
            self.lgbm_classifier,
            "predict",
        ):
            try:
                # Prepare comprehensive features for LightGBM
                lgbm_features = current_features[available_features].tail(1).values

                # Ensure we have valid features for LightGBM
                if lgbm_features.size > 0 and not np.isnan(lgbm_features).any():
                    try:
                        lgbm_regime = self.lgbm_classifier.predict(lgbm_features)[0]

                        # Get prediction probabilities
                        probabilities = self.lgbm_classifier.predict_proba(
                            lgbm_features,
                        )[0]
                        confidence = np.max(probabilities)
                        
                        self.logger.info(f"LightGBM prediction: {lgbm_regime} (confidence: {confidence:.3f})")
                    except Exception as e:
                        self.logger.error(f"Error in LightGBM prediction: {e}")
                        lgbm_regime = "UNKNOWN"
                        confidence = 0.0
                else:
                    self.logger.warning("Invalid features for LightGBM prediction")

            except Exception as e:
                self.logger.error(f"Error in LightGBM prediction: {e}")

        # Combine HMM and LightGBM predictions
        final_regime = lgbm_regime if lgbm_regime != "UNKNOWN" else hmm_regime

        # Enhanced candle analysis for ML model features
        candle_analysis = {}
        
        if not current_klines.empty:
            try:
                # Get the latest candle
                latest_candle = current_klines.iloc[-1]
                
                # Calculate candle metrics
                current_body_size = abs(latest_candle['close'] - latest_candle['open'])
                current_total_size = latest_candle['high'] - latest_candle['low']
                body_ratio = current_body_size / current_total_size if current_total_size > 0 else 0
                
                # Calculate gap from previous candle
                if len(current_klines) > 1:
                    prev_close = current_klines.iloc[-2]['close']
                    current_open = latest_candle['open']
                    gap_size = abs(current_open - prev_close) / prev_close if prev_close > 0 else 0
                else:
                    gap_size = 0
                
                # Get technical indicators
                rsi = current_features['rsi'].iloc[-1] if 'rsi' in current_features.columns else 50
                atr = current_features['atr'].iloc[-1] if 'atr' in current_features.columns else 0
                
                candle_analysis = {
                    "body_ratio": body_ratio,
                    "gap_size": gap_size,
                    "rsi": rsi,
                    "atr": atr,
                    "candle_direction": "BULLISH" if latest_candle['close'] > latest_candle['open'] else "BEARISH",
                    "volume_analysis": "HIGH" if latest_candle.get('volume', 0) > current_klines['volume'].iloc[-10:].mean() * 1.5 else "LOW" if latest_candle.get('volume', 0) < current_klines['volume'].iloc[-10:].mean() * 0.5 else "NORMAL"
                }
                    
            except Exception as e:
                self.logger.warning(f"Error in candle analysis: {e}")
                candle_analysis = {"error": str(e)}

        additional_info = {
            "hmm_state": predicted_state,
            "hmm_regime": hmm_regime,
            "lgbm_regime": lgbm_regime,
            "confidence": confidence,
            "features_used": len(available_features),
            "log_return": float(latest_features[0, 0]) if len(latest_features) > 0 else 0.0,
            "volatility": float(latest_features[0, 1]) if len(latest_features) > 1 else 0.0,
            "feature_names": available_features,
            "candle_analysis": candle_analysis
        }

        return final_regime, confidence, additional_info

    def save_model(self, model_path: str | None = None) -> None:
        """Save the trained HMM model and related components."""
        path_to_save = model_path if model_path else self.default_model_path

        if not self.trained or self.hmm_model is None:
            self.logger.warning("Cannot save untrained HMM model")
            return

        try:
            model_data = {
                "hmm_model": self.hmm_model,
                "scaler": self.scaler,
                "lgbm_classifier": self.lgbm_classifier,
                "state_to_regime_map": self.state_to_regime_map,
                "n_states": self.n_states,
                "config": self.config,
                "last_training_time": self.last_training_time,
            }

            joblib.dump(model_data, path_to_save)
            self.logger.info(f"HMM regime classifier saved to {path_to_save}")

        except Exception as e:
            self.logger.error(f"Error saving HMM model: {e}", exc_info=True)

    def load_model(self, model_path: str | None = None) -> bool:
        """Load the trained HMM model and related components."""
        path_to_load = model_path if model_path else self.default_model_path

        if not os.path.exists(path_to_load):
            self.logger.warning(f"HMM model file not found at {path_to_load}")
            return False

        try:
            model_data = joblib.load(path_to_load)

            # Validate loaded model components
            if "hmm_model" not in model_data or model_data["hmm_model"] is None:
                self.logger.error("Invalid HMM model in saved file")
                return False

            if "scaler" not in model_data or model_data["scaler"] is None:
                self.logger.error("Invalid scaler in saved file")
                return False

            self.hmm_model = model_data["hmm_model"]
            self.scaler = model_data["scaler"]
            self.lgbm_classifier = model_data.get("lgbm_classifier")
            self.state_to_regime_map = model_data.get("state_to_regime_map", {})
            self.n_states = model_data.get("n_states", 4)
            self.last_training_time = model_data.get("last_training_time")

            self.trained = True
            self.logger.info(f"HMM regime classifier loaded from {path_to_load}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading HMM model: {e}", exc_info=True)
            self.trained = False
            return False

    def get_state_statistics(self) -> dict:
        """Get statistics about the HMM states."""
        if not self.trained or self.hmm_model is None:
            return {}

        try:
            stats = {
                "n_states": self.n_states,
                "state_to_regime_map": self.state_to_regime_map,
                "transition_matrix": self.hmm_model.transmat_.tolist(),
                "start_probabilities": self.hmm_model.startprob_.tolist(),
            }

            return stats
        except Exception as e:
            self.logger.error(f"Error getting state statistics: {e}")
            return {}

    def _load_and_parse_klines_file(self, file_path: str, start_date, end_date):
        """
        Load and parse a klines CSV file with robust timestamp handling.
        
        Args:
            file_path: Path to the CSV file
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            DataFrame or None if parsing failed
        """
        try:
            # Load CSV directly with pandas
            df = pd.read_csv(file_path)
            if df.empty:
                return None
                
            # Convert timestamp to datetime with error handling
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                # Remove rows with invalid timestamps
                df = df.dropna(subset=['timestamp'])
            except Exception as e:
                self.logger.warning(f"⚠️ Could not parse timestamps in {file_path}: {e}")
                return None
            
            if df.empty:
                return None
                
            # Filter by date range
            df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
            
            if df.empty:
                return None
                
            # Set timestamp as index
            df = df.set_index('timestamp')
            
            # Ensure we have the required OHLCV columns
            if all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                return df
            else:
                self.logger.warning(f"⚠️ Missing required columns in {file_path}")
                return None
                
        except Exception as e:
            self.logger.warning(f"⚠️ Could not load {file_path}: {e}")
            return None

    def _load_and_parse_pickle_file(self, file_path: str, start_date, end_date):
        """
        Load and parse a pickle file with 1h klines data.
        
        Args:
            file_path: Path to the pickle file
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            DataFrame or None if loading failed
        """
        try:
            import pickle
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Convert to DataFrame if it's not already
            if isinstance(data, dict):
                # Handle the standard cached data structure
                if 'klines' in data:
                    df = data['klines']  # Already a DataFrame
                elif 'data' in data:
                    df = pd.DataFrame(data['data'])
                else:
                    # Assume the dict contains OHLCV data directly
                    df = pd.DataFrame(data)
            else:
                df = pd.DataFrame(data)
            
            if df.empty: return None
            
            # Ensure we have the required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                self.logger.warning(f"⚠️ Missing required columns in {file_path}")
                return None
            
            # Filter by date range if timestamp column exists
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df = df.dropna(subset=['timestamp'])
                if df.empty: return None
                df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
                if df.empty: return None
                df = df.set_index('timestamp')
            
            return df
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not load {file_path}: {e}")
            return None

    def _resample_15m_to_1h(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample 15-minute data to 1-hour data using OHLCV aggregation.
        
        Args:
            df: DataFrame with 15-minute OHLCV data
            
        Returns:
            DataFrame with 1-hour OHLCV data
        """
        if df.empty:
            return df
            
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            self.logger.warning("⚠️ Missing required OHLCV columns for resampling")
            return df
            
        # Resample to 1-hour data
        # OHLCV aggregation: first open, max high, min low, last close, sum volume
        resampled = df.resample('1H').agg({
            'open': 'first',
            'high': 'max', 
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        self.logger.info(f"✅ Resampled {len(df)} 15m records to {len(resampled)} 1h records")
        return resampled

    def _resample_to_1h(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample 1-minute data to 1-hour data using OHLCV aggregation.
        
        Args:
            df: DataFrame with 1-minute OHLCV data
            
        Returns:
            DataFrame with 1-hour OHLCV data
        """
        if df.empty:
            return df
            
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            self.logger.warning("⚠️ Missing required OHLCV columns for resampling")
            return df
            
        # Resample to 1-hour data
        # OHLCV aggregation: first open, max high, min low, last close, sum volume
        resampled = df.resample('1H').agg({
            'open': 'first',
            'high': 'max', 
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        self.logger.info(f"✅ Resampled {len(df)} 1m records to {len(resampled)} 1h records")
        return resampled

    async def label_historical_data(self, symbol: str, exchange: str = "BINANCE", lookback_years: int = 2) -> bool:
        """
        Label historical data using HMM regime classification.
        
        Args:
            symbol: Trading symbol (e.g., ETHUSDT)
            exchange: Exchange name (default: BINANCE)
            lookback_years: Number of years of data to use (default: 2)
            
        Returns:
            bool: True if labeling was successful, False otherwise
        """
        try:
            self.logger.info(f"🧠 Starting HMM labeling for {symbol} on {exchange} (last {lookback_years} years)")
            
            # Import utilities
            from src.config import CONFIG
            import glob
            import os
            from datetime import datetime, timedelta
            
            # Check for existing labeled data first
            labeled_data_path = os.path.join(
                CONFIG.get("DATA_DIR", "data"),
                f"{exchange}_{symbol}_labeled_regimes.csv"
            )
            
            if os.path.exists(labeled_data_path):
                self.logger.info(f"✅ Found existing labeled data at {labeled_data_path}")
                return True
            
            # Find existing 1h klines data
            self.logger.info("📊 Looking for existing historical data...")
            
            # First try to find consolidated 15m CSV data
            consolidated_pattern = f"data_cache/klines_{exchange}_{symbol}_15m_consolidated.csv"
            if os.path.exists(consolidated_pattern):
                self.logger.info(f"✅ Found consolidated 15m data: {consolidated_pattern}")
                # Load the consolidated CSV data
                df = pd.read_csv(consolidated_pattern)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
                
                # Filter by date range
                df = df[(df.index >= start_date) & (df.index <= end_date)]
                
                if len(df) > 0:
                    self.logger.info(f"✅ Loaded {len(df)} 15m records from consolidated file")
                    # Resample 15m to 1h
                    resampled_df = self._resample_15m_to_1h(df)
                    if len(resampled_df) >= self.min_data_points:
                        self.logger.info(f"✅ Resampled to {len(resampled_df)} 1h records")
                        return await self._train_with_data(resampled_df, symbol, exchange)
                    else:
                        self.logger.warning(f"⚠️ Insufficient data after resampling: {len(resampled_df)} records")
                else:
                    self.logger.warning(f"⚠️ No data in date range for consolidated file")
            
            # Fallback to pickle files
            pattern = f"data_cache/{symbol}_1h_*_cached_data.pkl"
            klines_files = glob.glob(pattern)
            
            if not klines_files:
                self.logger.error(f"❌ No historical data found for {symbol} on {exchange}")
                self.logger.info("💡 Please ensure you have downloaded data first using: python ares_launcher.py load --symbol ETHUSDT --exchange BINANCE")
                return False
            
            # Calculate date range for last N years
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_years * 365)
            
            self.logger.info(f"📅 Using data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Load and combine all 1h pickle files within date range
            klines_dfs = []
            for file in sorted(klines_files):
                df = self._load_and_parse_pickle_file(file, start_date, end_date)
                if df is not None:
                    klines_dfs.append(df)
            
            if not klines_dfs:
                self.logger.error("❌ Failed to load any 1h klines data in the specified date range")
                return False
            
            # Combine all dataframes
            klines_df = pd.concat(klines_dfs, ignore_index=False)
            klines_df = klines_df.sort_index()
            
            # Remove duplicates
            klines_df = klines_df[~klines_df.index.duplicated(keep='first')]
            
            self.logger.info(f"✅ Loaded {len(klines_df)} 1m klines records from last {lookback_years} years")
            
            # Resample 1-minute data to 1-hour data
            self.logger.info("🔄 Resampling 1m data to 1h data...")
            klines_df = self._resample_to_1h(klines_df)
            
            if klines_df.empty:
                self.logger.error("❌ Failed to resample data to 1h timeframe")
                return False
            
            self.logger.info(f"✅ Resampled to {len(klines_df)} 1h klines records")
            
            # Validate timeframe (must be 1h)
            if not self._validate_timeframe(klines_df):
                self.logger.error("❌ Data is not from 1h timeframe")
                return False
            
            # Train the HMM classifier
            self.logger.info("🎯 Training HMM classifier...")
            success = self.train_classifier(klines_df)
            
            if not success:
                self.logger.error("❌ Failed to train HMM classifier")
                return False
            
            # Calculate features and get state sequence
            features_df = self._calculate_features(klines_df)
            if features_df.empty:
                self.logger.error("❌ Failed to calculate features")
                return False
            
            # Get state sequence from trained HMM
            state_sequence = self.hmm_model.predict(features_df.values)
            
            # Interpret states and create regime labels
            state_interpretation = self._interpret_hmm_states(features_df, state_sequence)
            
            # Create labeled dataframe - use the same index as features_df to avoid length mismatch
            labeled_df = klines_df.loc[features_df.index].copy()
            labeled_df['hmm_state'] = state_sequence
            
            # Get regime labels from state interpretation
            if 'state_to_regime_map' in state_interpretation:
                labeled_df['hmm_regime'] = [state_interpretation['state_to_regime_map'].get(state, 'UNKNOWN') 
                                           for state in state_sequence]
            else:
                # Fallback: use state numbers as regime labels
                labeled_df['hmm_regime'] = [f"STATE_{state}" for state in state_sequence]
            
            # Add additional regime classifications based on technical analysis
            labeled_df['market_regime'] = await self._add_advanced_regime_classifications(labeled_df)
            
            # Save labeled data
            labeled_data_path = os.path.join(
                CONFIG.get("DATA_DIR", "data"),
                f"{exchange}_{symbol}_labeled_regimes.csv"
            )
            os.makedirs(os.path.dirname(labeled_data_path), exist_ok=True)
            
            labeled_df.to_csv(labeled_data_path)
            self.logger.info(f"✅ Labeled data saved to {labeled_data_path}")
            
            # Print statistics
            regime_counts = labeled_df['market_regime'].value_counts()
            self.logger.info("📊 Regime distribution:")
            for regime, count in regime_counts.items():
                percentage = (count / len(labeled_df)) * 100
                self.logger.info(f"   {regime}: {count} ({percentage:.1f}%)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during HMM labeling: {e}", exc_info=True)
            return False

    async def train_ml_model(self, symbol: str, exchange: str = "BINANCE", lookback_years: int = 2, training_min_data_points: int = None) -> bool:
        """
        Train ML model based on HMM-labeled historical data.
        
        Args:
            symbol: Trading symbol (e.g., ETHUSDT)
            exchange: Exchange name (default: BINANCE)
            lookback_years: Number of years of data to use (default: 2)
            training_min_data_points: Override minimum data points for training (default: None, uses general requirement)
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        try:
            # Use training-specific minimum data points if provided, otherwise use general requirement
            effective_min_data_points = training_min_data_points if training_min_data_points is not None else self.min_data_points
            
            self.logger.info(f"🤖 Starting ML model training for {symbol} on {exchange} (last {lookback_years} years)")
            self.logger.info(f"📊 Configuration:")
            self.logger.info(f"   - Target timeframe: {self.target_timeframe}")
            self.logger.info(f"   - Volatility period: {self.volatility_period}")
            self.logger.info(f"   - Min data points: {effective_min_data_points} (training override: {training_min_data_points if training_min_data_points is not None else 'None'})")
            self.logger.info(f"   - HMM states: {self.n_states}")
            self.logger.info(f"   - HMM iterations: {self.n_iter}")
            
            # Import required modules
            from backtesting.ares_data_preparer import calculate_features_and_score, get_sr_levels
            from src.config import CONFIG
            import glob
            import os
            from datetime import datetime, timedelta
            
            # Load existing 1m consolidated data and resample to 1h
            self.logger.info("📊 Loading existing 1m consolidated data...")
            
            # Look for consolidated 1m CSV file
            consolidated_file = f"data_cache/klines_{exchange}_{symbol}_1m_consolidated.csv"
            
            if not os.path.exists(consolidated_file):
                self.logger.error(f"❌ No consolidated 1m data found for {symbol} on {exchange}")
                self.logger.info("💡 Please ensure you have downloaded and consolidated data first using: python ares_launcher.py load --symbol ETHUSDT --exchange BINANCE")
                return False
            
            self.logger.info(f"✅ Found consolidated data: {consolidated_file}")
            
            # Calculate date range for last N years
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_years * 365)
            
            self.logger.info(f"📅 Using data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Load the consolidated CSV data
            klines_df = pd.read_csv(consolidated_file)
            klines_df['timestamp'] = pd.to_datetime(klines_df['timestamp'])
            klines_df = klines_df.set_index('timestamp')
            
            # Filter by date range
            klines_df = klines_df[(klines_df.index >= start_date) & (klines_df.index <= end_date)]
            
            if klines_df.empty:
                self.logger.error("❌ No data found in the specified date range")
                return False
            
            self.logger.info(f"✅ Loaded {len(klines_df)} 1m klines records from last {lookback_years} years")
            
            # Resample 1-minute data to 1-hour data
            self.logger.info("🔄 Resampling 1m data to 1h data...")
            klines_df = self._resample_to_1h(klines_df)
            
            if klines_df.empty:
                self.logger.error("❌ Failed to resample data to 1h timeframe")
                return False
            
            self.logger.info(f"✅ Resampled to {len(klines_df)} 1h klines records")
            
            # Get SR levels using proper SR analyzer
            self.logger.info("🔍 Calculating support and resistance levels...")
            sr_levels = await self._calculate_sr_levels(klines_df)
            
            # Calculate features and prepare data
            self.logger.info("🔧 Calculating features and preparing data...")
            
            # For now, use the HMM features directly instead of the complex feature calculation
            # This avoids the need for agg_trades_df and futures_df
            prepared_df = self._calculate_features(klines_df, effective_min_data_points)
            
            if prepared_df.empty:
                self.logger.error("❌ Failed to calculate features")
                return False
            
            if prepared_df.empty:
                self.logger.error("❌ Failed to prepare data")
                return False
            
            # Train HMM classifier first if not already trained
            if not self.trained:
                self.logger.info("🎯 Training HMM classifier...")
                success = self.train_classifier(klines_df)
                if not success:
                    self.logger.error("❌ Failed to train HMM classifier")
                    return False
            
            # Get HMM features and state sequence (use only basic features for HMM)
            hmm_features = self._calculate_features(klines_df, effective_min_data_points)
            if hmm_features.empty:
                self.logger.error("❌ Failed to calculate HMM features")
                return False
            
            # Use only basic features for HMM prediction
            basic_features = ["log_returns", "volatility_20"]
            hmm_basic_features = hmm_features[basic_features].values
            
            # Remove any infinite values
            hmm_basic_features = hmm_basic_features[np.isfinite(hmm_basic_features).all(axis=1)]
            
            if len(hmm_basic_features) < self.n_states * 10:
                self.logger.error("❌ Insufficient valid data for HMM prediction")
                return False
            
            # Scale features for HMM prediction
            if self.scaler is None:
                self.logger.error("❌ Scaler not available for HMM prediction")
                return False
            
            scaled_hmm_features = self.scaler.transform(hmm_basic_features)
            state_sequence = self.hmm_model.predict(scaled_hmm_features)
            state_interpretation = self._interpret_hmm_states(hmm_features, state_sequence)
            
            # Add HMM regime labels to prepared data
            prepared_df['hmm_regime'] = [state_interpretation['state_to_regime_map'].get(state, 'UNKNOWN') 
                                       for state in state_sequence[:len(prepared_df)]]
            
            # Add advanced regime classifications (HUGE_CANDLE, MOMENTUM, SR_ZONE_ACTION)
            # Use original klines_df for advanced classification since it has OHLCV data
            labeled_df = klines_df.loc[prepared_df.index].copy()
            labeled_df['hmm_regime'] = [state_interpretation['state_to_regime_map'].get(state, 'UNKNOWN') 
                                       for state in state_sequence[:len(labeled_df)]]
            prepared_df['market_regime'] = await self._add_advanced_regime_classifications(labeled_df)
            
            # Prepare features for ML training
            feature_columns = [col for col in prepared_df.columns 
                             if col not in ['hmm_regime', 'Market_Regime_Label', 'Is_Strong_Trend']]
            
            X = prepared_df[feature_columns].dropna()
            y = prepared_df.loc[X.index, 'market_regime']  # Use advanced market_regime instead of basic hmm_regime
            
            if len(X) < 100:
                self.logger.error("❌ Insufficient data for ML training")
                return False
            
            # Train LightGBM classifier
            self.logger.info("🚀 Training LightGBM classifier...")
            self._train_lgbm_classifier(X, y.tolist())
            
            # Save the complete model with exchange and symbol in filename
            custom_model_path = os.path.join(
                CONFIG["CHECKPOINT_DIR"],
                "analyst_models",
                f"hmm_regime_classifier_{exchange}_{symbol}_{self.target_timeframe}.joblib",
            )
            self.save_model(custom_model_path)
            
            self.logger.info("✅ ML model training completed successfully")
            
            # Print training statistics
            self.logger.info(f"📊 Training statistics:")
            self.logger.info(f"   Total samples: {len(X)}")
            self.logger.info(f"   Features: {len(feature_columns)}")
            self.logger.info(f"   Regime distribution: {y.value_counts().to_dict()}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during ML model training: {e}", exc_info=True)
            return False

    async def _add_advanced_regime_classifications(self, labeled_df: pd.DataFrame) -> list[str]:
        """
        Add advanced regime classifications (HUGE_CANDLE, SR_ZONE_ACTION) based on technical analysis.
        S/R classification uses sr_analyzer for proper support/resistance detection.
        
        Args:
            labeled_df: DataFrame with HMM regime classifications
            
        Returns:
            List of advanced regime classifications
        """
        try:
            # Import sr_analyzer
            from src.analyst.sr_analyzer import SRLevelAnalyzer
            
            # Calculate technical indicators for advanced classification
            close_prices = labeled_df['close']
            
            # Calculate moving averages
            ma_20 = close_prices.rolling(window=20).mean()
            ma_50 = close_prices.rolling(window=50).mean()
            
            # Calculate RSI
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Calculate volatility (ATR-like)
            high_low = labeled_df['high'] - labeled_df['low']
            high_close = np.abs(labeled_df['high'] - close_prices.shift())
            low_close = np.abs(labeled_df['low'] - close_prices.shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window=14).mean()
            
            # Calculate basic indicators
            
            # Initialize SR analyzer
            sr_analyzer = SRLevelAnalyzer(self.config)
            
            # Analyze SR levels using historical data
            sr_analysis = await sr_analyzer.analyze(labeled_df)
            
            # Initialize regime list
            advanced_regimes = []
            
            for i in range(len(labeled_df)):
                if i < 50:  # Not enough data for indicators
                    advanced_regimes.append(labeled_df.iloc[i]['hmm_regime'])
                    continue
                
                current_rsi = rsi.iloc[i]
                current_atr = atr.iloc[i]
                current_ma_20 = ma_20.iloc[i]
                current_ma_50 = ma_50.iloc[i]
                current_price = close_prices.iloc[i]
                hmm_regime = labeled_df.iloc[i]['hmm_regime']
                
                # Get basic indicators
                current_rsi = rsi.iloc[i]
                current_atr = atr.iloc[i]
                
                # ENHANCED HUGE CANDLES detection - Focus on sudden events, not trends
                current_candle_size = abs(labeled_df.iloc[i]['high'] - labeled_df.iloc[i]['low'])
                current_body_size = abs(labeled_df.iloc[i]['close'] - labeled_df.iloc[i]['open'])
                
                # Calculate multiple thresholds for better detection
                atr_value = atr.iloc[i] if not pd.isna(atr.iloc[i]) else 0
                
                # Threshold 1: Candle size relative to ATR (sudden volatility)
                atr_threshold = 2.5 if atr_value > 0 else 0
                
                # Threshold 2: Candle size relative to recent average (sudden size increase)
                recent_candles = labeled_df.iloc[max(0, i-20):i]
                if len(recent_candles) > 0:
                    avg_recent_size = (recent_candles['high'] - recent_candles['low']).mean()
                    size_threshold = avg_recent_size * 2.0 if avg_recent_size > 0 else 0
                else:
                    size_threshold = 0
                
                # Threshold 3: Body size relative to total size (strong directional move)
                body_ratio = current_body_size / current_candle_size if current_candle_size > 0 else 0
                body_threshold = 0.6  # At least 60% body
                
                # Threshold 4: Price gap from previous candle (sudden gap)
                if i > 0:
                    prev_close = labeled_df.iloc[i-1]['close']
                    current_open = labeled_df.iloc[i]['open']
                    gap_size = abs(current_open - prev_close) / prev_close if prev_close > 0 else 0
                    gap_threshold = 0.02  # 2% gap
                else:
                    gap_size = 0
                    gap_threshold = 0
                
                # Enhanced detection criteria
                is_huge_candle = False
                
                # Criterion 1: Large candle with strong body (sudden directional move)
                if (current_candle_size > atr_threshold and 
                    current_candle_size > size_threshold and 
                    body_ratio > body_threshold):
                    is_huge_candle = True
                
                # Criterion 2: Large gap with volume spike (sudden news/event)
                elif (gap_size > gap_threshold and 
                      labeled_df.iloc[i].get('volume', 0) > labeled_df.iloc[max(0, i-20):i]['volume'].mean() * 1.5):
                    is_huge_candle = True
                
                # Criterion 3: Extremely large candle (5x average or 4x ATR)
                elif (current_candle_size > avg_recent_size * 5.0 or 
                      current_candle_size > atr_value * 4.0):
                    is_huge_candle = True
                
                if is_huge_candle:
                    # Simple HUGE_CANDLE detection - let ML model handle trading decisions
                    advanced_regimes.append("HUGE_CANDLE")
                # S/R classification using sr_analyzer - More aggressive
                else:
                    # Check if price is near S/R levels using sr_analyzer
                    sr_proximity = sr_analyzer.detect_sr_zone_proximity(
                        current_price, 
                        tolerance_percent=0.5  # Reduced to 0.5% tolerance
                    )
                    
                    if sr_proximity['in_zone']:
                        # More restrictive conditions for S/R zone action
                        level_strength = sr_proximity.get('level_strength', 0.0)
                        touch_count = sr_proximity.get('touch_count', 0)
                        
                        # Classify as SR_ZONE_ACTION with slightly more lenient conditions
                        if (level_strength > 0.8 and 
                            touch_count >= 2 and 
                            sr_proximity['distance_percent'] < 0.75):  # Slightly more lenient distance
                            advanced_regimes.append("SR_ZONE_ACTION")
                        else:
                            advanced_regimes.append(hmm_regime)
                    else:
                        # Keep the HMM classification
                        advanced_regimes.append(hmm_regime)
            
            return advanced_regimes
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error in advanced regime classification: {e}")
            # Fallback to HMM regimes
            return labeled_df['hmm_regime'].tolist()

    async def _calculate_sr_levels(self, klines_df: pd.DataFrame) -> dict:
        """
        Calculate support and resistance levels using the existing SR analyzer.
        
        Args:
            klines_df: DataFrame with OHLCV data
            
        Returns:
            dict: Dictionary containing support and resistance levels
        """
        try:
            # Import SR analyzer
            from src.analyst.sr_analyzer import SRLevelAnalyzer
            
            # Initialize SR analyzer with proper config
            sr_config = {
                "sr_analyzer": {
                    "min_touch_count": 2,
                    "lookback_period": 100,
                    "strength_weights": {"touches": 0.6, "recency": 0.4},
                    "consolidation_tolerance": 0.0075,
                    "use_time_decay": True
                }
            }
            
            sr_analyzer = SRLevelAnalyzer(sr_config)
            
            # Initialize the SR analyzer
            await sr_analyzer.initialize()
            
            # Analyze SR levels using the existing analyzer
            sr_result = await sr_analyzer.analyze(klines_df)
            
            if sr_result is None:
                self.logger.warning("SR analysis returned None, using fallback levels")
                return self._calculate_fallback_sr_levels(klines_df)
            
            # Extract levels from result
            support_levels = sr_result.get('support_levels', [])
            resistance_levels = sr_result.get('resistance_levels', [])
            
            # Convert to simple format for compatibility
            sr_levels = {
                'support': [level.get('price', 0) for level in support_levels],
                'resistance': [level.get('price', 0) for level in resistance_levels],
                'support_strength': [level.get('strength', 0) for level in support_levels],
                'resistance_strength': [level.get('strength', 0) for level in resistance_levels],
                'analysis_time': sr_result.get('analysis_time'),
                'confidence': {
                    'support': sr_result.get('support_confidence', 0),
                    'resistance': sr_result.get('resistance_confidence', 0)
                }
            }
            
            self.logger.info(f"SR: Found {len(support_levels)} support and {len(resistance_levels)} resistance levels")
            return sr_levels
            
        except Exception as e:
            self.logger.warning(f"SR analysis failed: {e}, using fallback method")
            return self._calculate_fallback_sr_levels(klines_df)
    
    def _calculate_fallback_sr_levels(self, klines_df: pd.DataFrame) -> dict:
        """
        Fallback SR level calculation using simple pivot points.
        
        Args:
            klines_df: DataFrame with OHLCV data
            
        Returns:
            dict: Simple SR levels using pivot points
        """
        try:
            if klines_df.empty:
                return {'support': [], 'resistance': [], 'support_strength': [], 'resistance_strength': []}
            
            # Calculate pivot points
            high = klines_df['high'].max()
            low = klines_df['low'].min()
            close = klines_df['close'].iloc[-1]
            
            # Simple pivot point calculation
            pivot = (high + low + close) / 3
            r1 = 2 * pivot - low
            s1 = 2 * pivot - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            
            sr_levels = {
                'support': [s2, s1, low],
                'resistance': [high, r1, r2],
                'support_strength': [0.5, 0.7, 0.9],
                'resistance_strength': [0.9, 0.7, 0.5],
                'analysis_time': datetime.now(),
                'confidence': {'support': 0.6, 'resistance': 0.6}
            }
            
            self.logger.info(f"SR: Fallback calculation found {len(sr_levels['support'])} support and {len(sr_levels['resistance'])} resistance levels")
            return sr_levels
            
        except Exception as e:
            self.logger.error(f"Fallback SR calculation failed: {e}")
            return {'support': [], 'resistance': [], 'support_strength': [], 'resistance_strength': []}


# Example usage and testing
if __name__ == "__main__":
    from src.analyst.data_utils import create_dummy_data, load_klines_data
    from src.config import CONFIG

    # Create dummy data for testing
    klines_filename = CONFIG["KLINES_FILENAME"]
    create_dummy_data(klines_filename, "klines", num_records=1000)

    # Load data
    klines_df = load_klines_data(klines_filename)

    if klines_df.empty:
        print("Failed to load klines data for testing")
        sys.exit(1)

    # Initialize and train HMM classifier
    hmm_classifier = HMMRegimeClassifier(CONFIG)

    # Train the classifier
    success = hmm_classifier.train_classifier(klines_df)

    if success:
        print("HMM classifier trained successfully!")

        # Test prediction
        test_klines = klines_df.tail(10)
        regime, confidence, info = hmm_classifier.predict_regime(test_klines)

        print(f"Predicted regime: {regime}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Additional info: {info}")

        # Get state statistics
        stats = hmm_classifier.get_state_statistics()
        print(f"State statistics: {stats}")

    else:
        print("Failed to train HMM classifier")
