# src/analyst/unified_regime_classifier.py
import os
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from hmmlearn import hmm
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import CONFIG
from src.utils.logger import system_logger


class UnifiedRegimeClassifier:
    """
    Unified Market Regime Classifier with HMM-based labeling and ensemble prediction.

    Approach:
    1. HMM-based labeling for basic regimes (BULL, BEAR, SIDEWAYS, VOLATILE)
    2. Ensemble prediction with majority voting for basic regimes
    3. Location classification (SUPPORT, RESISTANCE, OPEN_RANGE)
    """
    
    def __init__(self, config: dict[str, Any], exchange: str = "UNKNOWN", symbol: str = "UNKNOWN"):
        self.config = config.get("analyst", {}).get("unified_regime_classifier", {})
        self.global_config = config
        self.logger = system_logger.getChild("UnifiedRegimeClassifier")
        self.exchange = exchange
        self.symbol = symbol
        
        # HMM Configuration - now 4 states for 4 regimes
        self.n_states = self.config.get("n_states", 4)  # BULL, BEAR, SIDEWAYS, VOLATILE
        self.n_iter = self.config.get("n_iter", 100)
        self.random_state = self.config.get("random_state", 42)
        self.target_timeframe = self.config.get(
            "target_timeframe",
            "1h",
        )  # Strategist works on 1h timeframe
        self.volatility_period = self.config.get("volatility_period", 10)
        
        # Detect BLANK mode and adjust minimum data points accordingly
        import os
        blank_mode = os.environ.get("BLANK_TRAINING_MODE", "0") == "1"
        if blank_mode:
            self.min_data_points = self.config.get("min_data_points", 750)  # Reduced for BLANK mode
            self.logger.info("🔧 BLANK MODE DETECTED: Using reduced minimum data points (750)")
        else:
            self.min_data_points = self.config.get("min_data_points", 1000)  # Default for full mode

        # Models
        self.hmm_model = None
        self.scaler = None
        self.state_to_regime_map = {}

        # Ensemble Models for Basic Regimes
        self.basic_ensemble = None

        # Location Classifier
        self.location_classifier = None
        self.location_label_encoder = None

        # Legacy S/R/Candle code removed
        self.enable_sr_integration = self.config.get("enable_sr_integration", True)
        self.basic_label_encoder = None

        # Training Status
        self.trained = False
        self.last_training_time = None

        # Model Paths
        self.model_dir = os.path.join(CONFIG["CHECKPOINT_DIR"], "analyst_models")
        os.makedirs(self.model_dir, exist_ok=True)

        self.hmm_model_path = os.path.join(
            self.model_dir,
            f"unified_hmm_model_{self.exchange}_{self.symbol}_{self.target_timeframe}.joblib",
        )
        self.ensemble_model_path = os.path.join(
            self.model_dir,
            f"unified_ensemble_model_{self.exchange}_{self.symbol}_{self.target_timeframe}.joblib",
        )
        self.location_model_path = os.path.join(
            self.model_dir,
            f"unified_location_model_{self.exchange}_{self.symbol}_{self.target_timeframe}.joblib",
        )

    async def initialize(self) -> bool:
        """
        Initialize the UnifiedRegimeClassifier.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info(f"Initializing UnifiedRegimeClassifier for {self.exchange}_{self.symbol}")
            
            # Create model directory if it doesn't exist
            os.makedirs(self.model_dir, exist_ok=True)
            
            # Try to load existing models
            if self.load_models():
                self.logger.info("✅ Loaded existing models successfully")
                self.trained = True
            else:
                self.logger.info("ℹ️  No existing models found, will train new models")
                self.trained = False
            
            self.logger.info("✅ UnifiedRegimeClassifier initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize UnifiedRegimeClassifier: {e}")
            return False

    def _calculate_features(
        self,
        klines_df: pd.DataFrame,
        min_data_points: int = None,
    ) -> pd.DataFrame:
        """
        Calculate comprehensive features for regime and location classification.

        Args:
            klines_df: DataFrame with OHLCV data
            min_data_points: Minimum data points required

        Returns:
            DataFrame with calculated features
        """
        if min_data_points is None:
            min_data_points = self.min_data_points

        if len(klines_df) < min_data_points:
            self.logger.warning(
                f"Insufficient data: {len(klines_df)} < {min_data_points}. Consider reducing min_data_points or collecting more data.",
            )
            # Try with a lower threshold if possible
            if len(klines_df) >= 200:  # Minimum viable amount
                self.logger.info(f"Proceeding with {len(klines_df)} data points (reduced from {min_data_points})")
                min_data_points = len(klines_df)
            else:
                self.logger.error(f"Data too small: {len(klines_df)} < 200 minimum required")
                return pd.DataFrame()

        self.logger.info(f"🔧 Calculating features for {len(klines_df)} periods...")

        # Create features DataFrame
        features_df = klines_df.copy()

        # Basic price features
        features_df["log_returns"] = np.log(
            features_df["close"] / features_df["close"].shift(1),
        )
        features_df["price_change"] = features_df["close"].pct_change()
        features_df["high_low_ratio"] = features_df["high"] / features_df["low"]
        features_df["close_open_ratio"] = features_df["close"] / features_df["open"]

        # Volatility features
        features_df["volatility_20"] = features_df["log_returns"].rolling(20).std()
        features_df["volatility_10"] = features_df["log_returns"].rolling(10).std()
        features_df["volatility_5"] = features_df["log_returns"].rolling(5).std()

        # Volume features
        features_df["volume_ratio"] = (
            features_df["volume"] / features_df["volume"].rolling(20).mean()
        )
        features_df["volume_change"] = features_df["volume"].pct_change()

        # Technical indicators
        features_df = self._calculate_rsi(features_df)
        features_df = self._calculate_macd(features_df)
        features_df = self._calculate_bollinger_bands(features_df)
        features_df = self._calculate_atr(features_df)
        features_df['atr_normalized'] = features_df['atr'] / features_df['close']
        features_df = self._calculate_adx(features_df)
    
        # Enhanced volatility features for VOLATILE regime detection
        features_df["volatility_regime"] = self._calculate_volatility_regime(
            features_df,
        )
        features_df["volatility_acceleration"] = features_df["volatility_20"].diff()
        features_df["volatility_momentum"] = features_df["volatility_20"] - features_df[
            "volatility_20"
        ].shift(5)
        
        # Improved NaN handling: use forward fill for technical indicators
        # This preserves more data points while maintaining feature quality
        technical_columns = [
            "rsi", "macd", "macd_signal", "macd_histogram", 
            "bb_upper", "bb_middle", "bb_lower", "bb_position", "bb_width",
            "atr", "atr_normalized", "adx", "volatility_regime"
        ]
        
        for col in technical_columns:
            if col in features_df.columns:
                # Forward fill NaN values for technical indicators
                features_df[col] = features_df[col].ffill()
                # Fill any remaining NaN values with 0
                features_df[col] = features_df[col].fillna(0)

        # For log_returns and other price-based features, use 0 for NaN
        price_features = ["log_returns", "price_change", "volume_change", "volatility_acceleration", "volatility_momentum"]
        for col in price_features:
            if col in features_df.columns:
                features_df[col] = features_df[col].fillna(0)

        # For ratio features, use 1 for NaN (neutral ratio)
        ratio_features = ["high_low_ratio", "close_open_ratio", "volume_ratio"]
        for col in ratio_features:
            if col in features_df.columns:
                features_df[col] = features_df[col].fillna(1)

        # For volatility features, use 0 for NaN
        vol_features = ["volatility_20", "volatility_10", "volatility_5"]
        for col in vol_features:
            if col in features_df.columns:
                features_df[col] = features_df[col].fillna(0)

        # Only drop rows that still have NaN values after all the filling
        # This should be minimal now
        initial_length = len(features_df)
        features_df = features_df.dropna()
        dropped_rows = initial_length - len(features_df)

        self.logger.info(
            f"✅ Calculated comprehensive features for {len(features_df)} periods (dropped {dropped_rows} rows due to NaN)",
        )

        return features_df

    def _calculate_volatility_regime(self, features_df: pd.DataFrame) -> pd.Series:
        """
        Calculate volatility regime for VOLATILE classification.
        """
        # Calculate rolling volatility percentiles
        vol_20_percentile = features_df["volatility_20"].rolling(100).rank(pct=True)
        vol_10_percentile = features_df["volatility_10"].rolling(100).rank(pct=True)

        # High volatility regime (top 20% of volatility)
        high_vol = (vol_20_percentile > 0.8) & (vol_10_percentile > 0.8)

        # Volatility acceleration (increasing volatility)
        vol_accel = features_df["volatility_20"].diff() > 0

        # Combine conditions for VOLATILE regime
        volatile_regime = high_vol | vol_accel

        return volatile_regime.astype(int)

    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate RSI indicator."""
        try:
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df["rsi"] = 100 - (100 / (1 + rs))
        except Exception as e:
            self.logger.warning(f"Error calculating RSI: {e}")
            df["rsi"] = 50  # Default value
        return df

    def _calculate_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """Calculate MACD indicator."""
        try:
            exp1 = df["close"].ewm(span=fast).mean()
            exp2 = df["close"].ewm(span=slow).mean()
            df["macd"] = exp1 - exp2
            df["macd_signal"] = df["macd"].ewm(span=signal).mean()
            df["macd_histogram"] = df["macd"] - df["macd_signal"]
        except Exception as e:
            self.logger.warning(f"Error calculating MACD: {e}")
            df["macd"] = 0
            df["macd_signal"] = 0
            df["macd_histogram"] = 0
        return df

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
            """Calculate the Average Directional Index (ADX)."""
            try:
                high = df["high"]
                low = df["low"]
                close = df["close"]
    
                # Calculate True Range (TR)
                tr1 = high - low
                tr2 = abs(high - close.shift(1))
                tr3 = abs(low - close.shift(1))
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.ewm(alpha=1/period, adjust=False).mean()
    
                # Calculate Directional Movement (+DM, -DM)
                move_up = high.diff()
                move_down = low.diff()
                plus_dm = ((move_up > move_down) & (move_up > 0)) * move_up
                minus_dm = ((move_down > move_up) & (move_down > 0)) * move_down
                
                plus_dm = plus_dm.ewm(alpha=1/period, adjust=False).mean()
                minus_dm = minus_dm.ewm(alpha=1/period, adjust=False).mean()
    
                # Calculate Directional Index (+DI, -DI)
                plus_di = 100 * (plus_dm / atr)
                minus_di = 100 * (minus_dm / atr)
    
                # Calculate Directional Movement Index (DX) and ADX
                dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
                df["adx"] = dx.ewm(alpha=1/period, adjust=False).mean()
                df["adx"] = df["adx"].fillna(25) # Fill initial NaNs with a neutral value
            
            except Exception as e:
                self.logger.warning(f"Error calculating ADX: {e}")
                df["adx"] = 25  # Default neutral value on error
            
            return df

    def _calculate_bollinger_bands(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2,
    ) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        try:
            sma = df["close"].rolling(window=period).mean()
            std = df["close"].rolling(window=period).std()
            df["bb_upper"] = sma + (std * std_dev)
            df["bb_lower"] = sma - (std * std_dev)
            df["bb_position"] = (df["close"] - df["bb_lower"]) / (
                df["bb_upper"] - df["bb_lower"]
            )
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma
        except Exception as e:
            self.logger.warning(f"Error calculating Bollinger Bands: {e}")
            df["bb_position"] = 0.5
            df["bb_width"] = 0.1
        return df

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average True Range."""
        try:
            high_low = df["high"] - df["low"]
            high_close = np.abs(df["high"] - df["close"].shift())
            low_close = np.abs(df["low"] - df["close"].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            df["atr"] = true_range.rolling(window=period).mean()
        except Exception as e:
            self.logger.warning(f"Error calculating ATR: {e}")
            df["atr"] = df["close"] * 0.02  # Default to 2% of close
        return df


    def _interpret_hmm_states(
            self,
            features_df: pd.DataFrame,
            state_sequence: np.ndarray,
        ) -> dict:
            """
            Interpret HMM states and map them to basic market regimes.
            Now uses ADX for robust SIDEWAYS detection.
            """
            analysis_df = features_df.copy()
            analysis_df["state"] = state_sequence
            state_analysis = {}
            
            # Define the thresholds for regime classification
            adx_sideways_threshold = self.config.get("adx_sideways_threshold", 25)
            volatility_threshold = self.config.get("volatility_threshold", 0.015)
    
            for state in range(self.n_states):
                state_data = analysis_df[analysis_df["state"] == state]
    
                if len(state_data) == 0:
                    continue
    
                # Calculate state characteristics
                mean_return = state_data["log_returns"].mean()
                mean_volatility = state_data["volatility_20"].mean()
                mean_adx = state_data["adx"].mean() # Calculate the mean ADX for the state
                mean_atr_norm = state_data["atr_normalized"].mean()
    
                # New regime classification logic with configurable thresholds
                if mean_volatility > volatility_threshold or mean_atr_norm > 0.02:
                    regime = "VOLATILE"
                elif mean_adx < adx_sideways_threshold:
                    regime = "SIDEWAYS"
                elif mean_return > 0: 
                    regime = "BULL"
                else:
                    regime = "BEAR"
    
                state_analysis[state] = {
                    "regime": regime,
                    "mean_return": mean_return,
                    "mean_volatility": mean_volatility,
                    "mean_adx": mean_adx, # Store for analysis
                    "count": len(state_data),
                }
    
                self.logger.info(
                    f"State {state}: {regime} "
                    f"(mean_return={mean_return:.4f}, mean_vol={mean_volatility:.4f}, mean_adx={mean_adx:.2f})",
                )
    
            # Create state to regime mapping
            state_to_regime_map = {
                state: data["regime"] for state, data in state_analysis.items()
            }
            state_analysis["state_to_regime_map"] = state_to_regime_map
    
            return state_analysis

    def _calculate_rolling_pivots(self, df_window: pd.DataFrame) -> dict:
        """
        Calculate rolling pivot points for dynamic support and resistance with strength metrics.
        
        Args:
            df_window: DataFrame window for pivot calculation
            
        Returns:
            Dict containing pivot levels with strength metrics
        """
        try:
            if len(df_window) < 5:
                return {
                    "s1": 0, "s2": 0, "r1": 0, "r2": 0,
                    "pivot": 0,
                    "strengths": {
                        "s1": {"strength": 0.0, "touches": 0, "volume": 0.0, "age": 0},
                        "s2": {"strength": 0.0, "touches": 0, "volume": 0.0, "age": 0},
                        "r1": {"strength": 0.0, "touches": 0, "volume": 0.0, "age": 0},
                        "r2": {"strength": 0.0, "touches": 0, "volume": 0.0, "age": 0}
                    }
                }
            
            # Calculate pivot point
            high = df_window["high"].max()
            low = df_window["low"].min()
            close = df_window["close"].iloc[-1]
            
            pivot = (high + low + close) / 3
            
            # Calculate support and resistance levels
            r1 = 2 * pivot - low
            r2 = pivot + (high - low)
            s1 = 2 * pivot - high
            s2 = pivot - (high - low)
            
            # Calculate strength metrics for each level
            levels = {"s1": s1, "s2": s2, "r1": r1, "r2": r2}
            strengths = {}
            
            for level_name, level_price in levels.items():
                if level_price <= 0:
                    strengths[level_name] = {"strength": 0.0, "touches": 0, "volume": 0.0, "age": 0}
                    continue
                
                # Calculate touches (how many times price approached this level)
                touches = 0
                tolerance = df_window["close"].std() * 0.1  # 10% of price volatility
                
                for i in range(1, len(df_window)):
                    prev_close = df_window["close"].iloc[i-1]
                    curr_close = df_window["close"].iloc[i]
                    
                    # Check if price approached the level
                    if abs(curr_close - level_price) <= tolerance:
                        touches += 1
                
                # Calculate volume near this level
                volume_near_level = 0.0
                for i in range(len(df_window)):
                    if abs(df_window["close"].iloc[i] - level_price) <= tolerance:
                        volume_near_level += df_window["volume"].iloc[i]
                
                # Calculate age (how long ago this level was first established)
                age = 0
                for i in range(len(df_window)):
                    if abs(df_window["close"].iloc[i] - level_price) <= tolerance:
                        age = len(df_window) - i
                        break
                
                # Calculate overall strength (0.0 to 1.0)
                # Factors: touches (30%), volume (40%), age (30%)
                touch_strength = min(touches / 5.0, 1.0)  # Normalize touches
                volume_strength = min(volume_near_level / df_window["volume"].sum(), 1.0)  # Normalize volume
                age_strength = min(age / len(df_window), 1.0)  # Normalize age
                
                overall_strength = (touch_strength * 0.3 + volume_strength * 0.4 + age_strength * 0.3)
                
                strengths[level_name] = {
                    "strength": overall_strength,
                    "touches": touches,
                    "volume": volume_near_level,
                    "age": age
                }
            
            return {
                "s1": s1,
                "s2": s2,
                "r1": r1,
                "r2": r2,
                "pivot": pivot,
                "strengths": strengths
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating rolling pivots: {e}")
            return {
                "s1": 0, "s2": 0, "r1": 0, "r2": 0,
                "pivot": 0,
                "strengths": {
                    "s1": {"strength": 0.0, "touches": 0, "volume": 0.0, "age": 0},
                    "s2": {"strength": 0.0, "touches": 0, "volume": 0.0, "age": 0},
                    "r1": {"strength": 0.0, "touches": 0, "volume": 0.0, "age": 0},
                    "r2": {"strength": 0.0, "touches": 0, "volume": 0.0, "age": 0}
                }
            }

    def _analyze_volume_levels(self, df_window: pd.DataFrame) -> dict | None:
        """
        Analyzes the volume profile to find the two most significant High Volume Nodes (HVNs),
        their age, and the number of times they've been tested.
        """
        if df_window.empty or len(df_window) < 20:
            return None
    
        # --- 1. ATR-Dynamic Binning (same as before) ---
        high_low = df_window['high'] - df_window['low']
        high_close = abs(df_window['high'] - df_window['close'].shift())
        low_close = abs(df_window['low'] - df_window['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        avg_atr = tr.mean()
        bin_size = max(avg_atr * 0.25, 1e-6)
        min_price = df_window['low'].min()
        max_price = df_window['high'].max()
        bins = np.arange(min_price, max_price, bin_size)
    
        # --- 2. Find Top 2 HVNs ---
        price_bins = pd.cut(df_window['close'], bins=bins, right=False)
        volume_by_bin = df_window.groupby(price_bins, observed=False)['volume'].sum()
        if volume_by_bin.empty:
            return None
            
        # Get the top 2 HVNs by volume
        top_hvns = volume_by_bin.nlargest(2)
        if top_hvns.empty:
            return None
    
        # --- 3. Analyze Each HVN ---
        analyzed_levels = {}
        for i, (level_bin, level_volume) in enumerate(top_hvns.items()):
            level_price = level_bin.mid
            
            # Find first time price entered this bin to determine age
            level_indices = df_window.index[df_window['close'].between(level_bin.left, level_bin.right)]
            if len(level_indices) == 0:
                continue
                
            first_touch_index = level_indices[0]
            age = len(df_window) - df_window.index.get_loc(first_touch_index)
    
            # Count touches after formation
            touches = 0
            data_since_formation = df_window.loc[first_touch_index:]
            for k in range(1, len(data_since_formation)):
                prev_high = data_since_formation['high'].iloc[k-1]
                prev_low = data_since_formation['low'].iloc[k-1]
                curr_high = data_since_formation['high'].iloc[k]
                curr_low = data_since_formation['low'].iloc[k]
                
                # A "touch" is when price crosses the level
                if (prev_low < level_price < curr_high) or (prev_high > level_price > curr_low):
                    touches += 1
            
            # Calculate additional strength metrics
            # Volume strength (normalized)
            total_volume = df_window['volume'].sum()
            volume_strength = min(level_volume / total_volume, 1.0) if total_volume > 0 else 0.0
            
            # Touch strength (normalized)
            touch_strength = min(touches / 10.0, 1.0)  # Normalize touches
            
            # Age strength (normalized)
            age_strength = min(age / len(df_window), 1.0)  # Normalize age
            
            # Calculate overall strength (0.0 to 1.0)
            # Factors: volume (50%), touches (30%), age (20%)
            overall_strength = (volume_strength * 0.5 + touch_strength * 0.3 + age_strength * 0.2)
            
            level_name = "poc" if i == 0 else "hvn_secondary"
            analyzed_levels[level_name] = {
                "price": level_price,
                "volume": level_volume,
                "age": age, # in number of candles
                "touches": touches,
                "strength": overall_strength,
                "volume_strength": volume_strength,
                "touch_strength": touch_strength,
                "age_strength": age_strength
            }
            
        return analyzed_levels
            
    def _classify_location(self, features_df: pd.DataFrame) -> list[str]:
        """
        Classifies location using a multi-layered context of short-term Dynamic Pivots (tactical)
        and long-term High Volume Nodes (strategic).
        """
        self.logger.info("Classifying location with tactical pivots and strategic volume levels...")
    
        # --- Configuration for dual-timeframe analysis ---
        long_term_hvn_period = self.config.get("long_term_hvn_period", 720)        # 30 days on a 1h chart
        short_term_pivot_period = self.config.get("short_term_pivot_period", 24)   # 1 day on a 1h chart
        tolerance = self.config.get("level_tolerance", 0.01)                       # 1% proximity tolerance
        max_level_age_pct = self.config.get("max_level_age_pct", 0.9)              # Allow older levels for long-term analysis
        min_level_touches = self.config.get("min_level_touches", 1)                # Must have at least 1 re-test
    
        locations = []
        
        # Start loop after the longest period to ensure enough data for all calculations
        start_index = long_term_hvn_period
        for i in range(start_index, len(features_df)):
            current_price = features_df["close"].iloc[i]
            
            # --- 1. Tactical Pivot Analysis (Short-Term) ---
            pivot_window = features_df.iloc[i - short_term_pivot_period : i]
            pivots = self._calculate_rolling_pivots(pivot_window)
            pivot_supports = [pivots["s1"], pivots["s2"]]
            pivot_resistances = [pivots["r1"], pivots["r2"]]
            
            # --- 2. Strategic Volume Level Analysis (Long-Term) ---
            hvn_window = features_df.iloc[i - long_term_hvn_period : i]
            volume_levels = self._analyze_volume_levels(hvn_window)
            
            # --- 3. Classification Logic ---
            loc_pivot = None
            loc_hvn = None
    
            # Check for Pivot proximity
            for p_sup in pivot_supports:
                if abs(current_price - p_sup) / current_price <= tolerance:
                    loc_pivot = "PIVOT_S"
                    break
            if not loc_pivot:
                for p_res in pivot_resistances:
                    if abs(current_price - p_res) / current_price <= tolerance:
                        loc_pivot = "PIVOT_R"
                        break
    
            # Check for HVN proximity using richer data
            if volume_levels:
                for level_data in volume_levels.values():
                    # Intelligence Rule: Filter out untested levels
                    if level_data["touches"] < min_level_touches:
                        continue
    
                    if abs(current_price - level_data["price"]) / current_price <= tolerance:
                        hvn_type = "SUPPORT" if current_price > level_data["price"] else "RESISTANCE"
                        loc_hvn = f"HVN_{hvn_type}"
                        break # Stop at the first significant HVN found
    
            # --- 4. Final Label Assignment ---
            if loc_pivot and loc_hvn:
                # A tactical pivot aligns with a strategic volume level - high confluence
                if "S" in loc_pivot and "SUPPORT" in loc_hvn:
                    locations.append("CONFLUENCE_S")
                elif "R" in loc_pivot and "RESISTANCE" in loc_hvn:
                    locations.append("CONFLUENCE_R")
                else:
                    locations.append(loc_pivot)
            elif loc_pivot:
                locations.append(loc_pivot)
            elif loc_hvn:
                locations.append(loc_hvn)
            else:
                locations.append("OPEN_RANGE")
        
        # Pad the beginning of the list for alignment
        padding = ["OPEN_RANGE"] * start_index
        final_locations = padding + locations
        
        self.logger.info(f"Finished classifying locations. Found: {pd.Series(final_locations).value_counts().to_dict()}")
        return final_locations
        

    async def train_hmm_labeler(self, historical_klines: pd.DataFrame) -> bool:
        """
        Train HMM-based labeler for basic regimes (BULL, BEAR, SIDEWAYS, VOLATILE).
        """
        try:
            self.logger.info("🎓 Training HMM-based Market Regime Classifier...")

            # Calculate features
            features_df = self._calculate_features(historical_klines)
            if features_df.empty:
                self.logger.error("No features available for HMM training")
                return False

            # Prepare features for HMM
            hmm_features = features_df[
                [
                    "log_returns",
                    "volatility_20",
                    "volume_ratio",
                    "rsi",
                    "macd",
                    "macd_signal",
                    "macd_histogram",
                    "bb_position",
                    "bb_width",
                    "atr",
                    "adx",
                    "volatility_regime",
                    "volatility_acceleration",
                ]
            ].fillna(0)

            # Scale features
            self.scaler = StandardScaler()
            hmm_features_scaled = self.scaler.fit_transform(hmm_features)

            # Train HMM model
            self.hmm_model = hmm.GaussianHMM(
                n_components=self.n_states,
                n_iter=self.n_iter,
                random_state=self.random_state,
                covariance_type="full",
            )

            self.hmm_model.fit(hmm_features_scaled)

            # Get state sequence
            state_sequence = self.hmm_model.predict(hmm_features_scaled)

            # Interpret states and create regime mapping
            state_analysis = self._interpret_hmm_states(features_df, state_sequence)
            self.state_to_regime_map = state_analysis["state_to_regime_map"]

            self.logger.info("✅ HMM-based regime classifier trained successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to train HMM regime classifier: {e}")
            return False

    async def train_location_classifier(self, historical_klines: pd.DataFrame) -> bool:
        """
        Train location classifier for OPEN_RANGE, PIVOT_S, PIVOT_R, HVN_SUPPORT, HVN_RESISTANCE, CONFLUENCE_S, CONFLUENCE_R.
        """
        try:
            self.logger.info("🎓 Training Location Classifier...")

            # Calculate features
            features_df = self._calculate_features(historical_klines)
            if features_df.empty:
                self.logger.error("No features available for location classifier training")
                return False

            # Get location labels using the new _classify_location method
            location_labels = self._classify_location(features_df)

            # Encode location labels
            self.location_label_encoder = LabelEncoder()
            location_encoded = self.location_label_encoder.fit_transform(
                location_labels,
            )

            # Prepare features for location classification
            location_features = features_df[
                ["close", "volume", "volatility_20", "rsi", "bb_position", "atr"]
            ].fillna(0)

            # Train location classifier
            self.location_classifier = LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbose=-1,
            )

            self.location_classifier.fit(location_features, location_encoded)

            self.logger.info("✅ Location classifier trained successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to train location classifier: {e}")
            return False

    async def train_basic_ensemble(self, historical_klines: pd.DataFrame) -> bool:
        """
        Train ensemble for basic regime classification (BULL, BEAR, SIDEWAYS, VOLATILE).
        """
        try:
            self.logger.info("🎓 Training Basic Regime Ensemble...")

            # Calculate features
            features_df = self._calculate_features(historical_klines)
            if features_df.empty:
                self.logger.error("No features available for ensemble training")
                return False

            # Get HMM-based labels
            hmm_features = features_df[
                [
                    "log_returns",
                    "volatility_20",
                    "volume_ratio",
                    "rsi",
                    "macd",
                    "macd_signal",
                    "macd_histogram",
                    "bb_position",
                    "bb_width",
                    "atr",
                    "adx",
                    "volatility_regime",
                    "volatility_acceleration",
                ]
            ].fillna(0)
            hmm_features_scaled = self.scaler.transform(hmm_features)
            state_sequence = self.hmm_model.predict(hmm_features_scaled)

            # Map states to regimes
            regime_labels = [
                self.state_to_regime_map.get(state, "SIDEWAYS")
                for state in state_sequence
            ]

            # Encode regime labels
            self.basic_label_encoder = LabelEncoder()
            regime_encoded = self.basic_label_encoder.fit_transform(regime_labels)

            # Prepare features for ensemble
            ensemble_features = features_df[
                [
                    "log_returns",
                    "volatility_20",
                    "volume_ratio",
                    "rsi",
                    "macd",
                    "macd_signal",
                    "macd_histogram",
                    "bb_position",
                    "bb_width",
                    "atr",
                    "adx",
                    "volatility_regime",
                    "volatility_acceleration",
                ]
            ].fillna(0)

            # Train ensemble
            self.basic_ensemble = LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbose=-1,
            )

            self.basic_ensemble.fit(ensemble_features, regime_encoded)

            self.logger.info("✅ Basic regime ensemble trained successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to train basic ensemble: {e}")
            return False

    async def train_complete_system(self, historical_klines: pd.DataFrame) -> bool:
        """
        Train the complete regime and location classification system.
        """
        try:
            self.logger.info("🎓 Training Complete Regime Classification System...")

            # Initialize SR analyzer
            # Legacy S/R/Candle code removed

            # Train HMM labeler
            if not await self.train_hmm_labeler(historical_klines):
                return False

            # Train basic ensemble
            if not await self.train_basic_ensemble(historical_klines):
                return False

            # Train location classifier
            if not await self.train_location_classifier(historical_klines):
                return False

            self.trained = True
            self.last_training_time = datetime.now()

            self.logger.info(
                "✅ Complete regime classification system trained successfully",
            )
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to train complete system: {e}")
            return False

    def predict_regime(
        self,
        current_klines: pd.DataFrame,
    ) -> tuple[str, float, dict]:
        """
        Predict only the regime (for backward compatibility).

        Args:
            current_klines: Current market data

        Returns:
            Tuple of (regime, confidence, additional_info)
        """
        try:
            if not self.trained:
                self.logger.warning("Models not trained, returning default prediction")
                return "SIDEWAYS", 0.5, {}

            # Calculate features
            features_df = self._calculate_features(current_klines)
            if features_df.empty:
                return "SIDEWAYS", 0.5, {}

            current_features = features_df.iloc[-1] if len(features_df) > 0 else None
            if current_features is None:
                return "SIDEWAYS", 0.5, {}

            # Predict regime
            regime_features = features_df[
                [
                    "log_returns",
                    "volatility_20",
                    "volume_ratio",
                    "rsi",
                    "macd",
                    "macd_signal",
                    "macd_histogram",
                    "bb_position",
                    "bb_width",
                    "atr",
                    "volatility_regime",
                    "volatility_acceleration",
                ]
            ].fillna(0)

            if self.basic_ensemble:
                regime_proba = self.basic_ensemble.predict_proba(
                    regime_features.iloc[-1:],
                )
                regime_pred = self.basic_ensemble.predict(regime_features.iloc[-1:])[0]
                regime = self.basic_label_encoder.inverse_transform([regime_pred])[0]
                regime_confidence = np.max(regime_proba)
            else:
                regime = "SIDEWAYS"
                regime_confidence = 0.5

            additional_info = {
                "regime_confidence": regime_confidence,
                "features_used": list(features_df.columns),
                "prediction_time": datetime.now().isoformat(),
            }

            return regime, regime_confidence, additional_info

        except Exception as e:
            self.logger.error(f"❌ Error in regime prediction: {e}")
            return "SIDEWAYS", 0.5, {"error": str(e)}

    def predict_regime_and_location(
        self,
        current_klines: pd.DataFrame,
    ) -> tuple[str, str, float, dict]:
        """
        Predict both regime and location.

        Args:
            current_klines: Current market data

        Returns:
            Tuple of (regime, location, confidence, additional_info)
        """
        try:
            if not self.trained:
                self.logger.warning("Models not trained, returning default predictions")
                return "SIDEWAYS", "OPEN_RANGE", 0.5, {}

            # Calculate features
            features_df = self._calculate_features(current_klines)
            if features_df.empty:
                return "SIDEWAYS", "OPEN_RANGE", 0.5, {}

            current_features = features_df.iloc[-1] if len(features_df) > 0 else None
            if current_features is None:
                return "SIDEWAYS", "OPEN_RANGE", 0.5, {}

            # Predict regime
            regime_features = features_df[
                [
                    "log_returns",
                    "volatility_20",
                    "volume_ratio",
                    "rsi",
                    "macd",
                    "macd_signal",
                    "macd_histogram",
                    "bb_position",
                    "bb_width",
                    "atr",
                    "volatility_regime",
                    "volatility_acceleration",
                ]
            ].fillna(0)

            if self.basic_ensemble:
                regime_proba = self.basic_ensemble.predict_proba(
                    regime_features.iloc[-1:],
                )
                regime_pred = self.basic_ensemble.predict(regime_features.iloc[-1:])[0]
                regime = self.basic_label_encoder.inverse_transform([regime_pred])[0]
                regime_confidence = np.max(regime_proba)
            else:
                regime = "SIDEWAYS"
                regime_confidence = 0.5

            # Predict location
            location_features = features_df[
                ["close", "volume", "volatility_20", "rsi", "bb_position", "atr"]
            ].fillna(0)

            if self.location_classifier:
                location_proba = self.location_classifier.predict_proba(
                    location_features.iloc[-1:],
                )
                location_pred = self.location_classifier.predict(
                    location_features.iloc[-1:],
                )[0]
                location = self.location_label_encoder.inverse_transform(
                    [location_pred],
                )[0]
                location_confidence = np.max(location_proba)
            else:
                # Fallback to rule-based location classification
                location_labels = self._classify_location(features_df)
                location = location_labels[-1] if location_labels else "OPEN_RANGE"
                location_confidence = 0.7

            # Calculate overall confidence
            overall_confidence = (regime_confidence + location_confidence) / 2

            additional_info = {
                "regime_confidence": regime_confidence,
                "location_confidence": location_confidence,
                "features_used": list(features_df.columns),
                "prediction_time": datetime.now().isoformat(),
            }

            return regime, location, overall_confidence, additional_info

        except Exception as e:
            self.logger.error(f"❌ Error in regime/location prediction: {e}")
            return "SIDEWAYS", "OPEN_RANGE", 0.5, {"error": str(e)}

    def save_models(self) -> None:
        """Save all trained models."""
        try:
            if self.hmm_model:
                joblib.dump(self.hmm_model, self.hmm_model_path)
                self.logger.info(f"✅ HMM model saved to {self.hmm_model_path}")

            if self.basic_ensemble:
                joblib.dump(self.basic_ensemble, self.ensemble_model_path)
                self.logger.info(
                    f"✅ Basic ensemble saved to {self.ensemble_model_path}",
                )

            if self.location_classifier:
                joblib.dump(self.location_classifier, self.location_model_path)
                self.logger.info(
                    f"✅ Location classifier saved to {self.location_model_path}",
                )

            # Save label encoders
            if self.basic_label_encoder:
                joblib.dump(
                    self.basic_label_encoder,
                    self.ensemble_model_path.replace(".joblib", "_encoder.joblib"),
                )

            if self.location_label_encoder:
                joblib.dump(
                    self.location_label_encoder,
                    self.location_model_path.replace(".joblib", "_encoder.joblib"),
                )

        except Exception as e:
            self.logger.error(f"❌ Error saving models: {e}")

    def load_models(self) -> bool:
        """Load all trained models."""
        try:
            # Load HMM model
            if os.path.exists(self.hmm_model_path):
                self.hmm_model = joblib.load(self.hmm_model_path)
                self.logger.info("✅ Loaded existing HMM model")

            # Load basic ensemble
            if os.path.exists(self.ensemble_model_path):
                self.basic_ensemble = joblib.load(self.ensemble_model_path)
                self.logger.info("✅ Loaded existing basic ensemble")

            # Load location classifier
            if os.path.exists(self.location_model_path):
                self.location_classifier = joblib.load(self.location_model_path)
                self.logger.info("✅ Loaded existing location classifier")

            # Load label encoders
            encoder_path = self.ensemble_model_path.replace(
                ".joblib",
                "_encoder.joblib",
            )
            if os.path.exists(encoder_path):
                self.basic_label_encoder = joblib.load(encoder_path)

            location_encoder_path = self.location_model_path.replace(
                ".joblib",
                "_encoder.joblib",
            )
            if os.path.exists(location_encoder_path):
                self.location_label_encoder = joblib.load(location_encoder_path)

            self.trained = True
            return True

        except Exception as e:
            self.logger.error(f"❌ Error loading models: {e}")
            return False

    async def classify_regimes(self, historical_klines: pd.DataFrame) -> dict[str, Any]:
        """
        Classify regimes for historical data (for training purposes).
        
        Args:
            historical_klines: Historical market data
            
        Returns:
            Dict containing regime classification results
        """
        try:
            if not self.trained:
                self.logger.warning("Models not trained, training now...")
                await self.train_complete_system(historical_klines)
            
            # Calculate features
            features_df = self._calculate_features(historical_klines)
            if features_df.empty:
                return {"error": "No features available for classification"}
            
            # Get regime predictions
            regime_features = features_df[
                [
                    "log_returns",
                    "volatility_20",
                    "volume_ratio",
                    "rsi",
                    "macd",
                    "macd_signal",
                    "macd_histogram",
                    "bb_position",
                    "bb_width",
                    "atr",
                    "adx",
                    "volatility_regime",
                    "volatility_acceleration",
                ]
            ].fillna(0)
            
            regimes = []
            if self.basic_ensemble:
                regime_predictions = self.basic_ensemble.predict(regime_features)
                regimes = self.basic_label_encoder.inverse_transform(regime_predictions).tolist()
            else:
                # Fallback to HMM states
                if self.hmm_model and self.scaler:
                    hmm_features_scaled = self.scaler.transform(regime_features)
                    state_sequence = self.hmm_model.predict(hmm_features_scaled)
                    regimes = [self.state_to_regime_map.get(state, "SIDEWAYS") for state in state_sequence]
                else:
                    regimes = ["SIDEWAYS"] * len(features_df)
            
            # Get location predictions
            location_labels = self._classify_location(features_df)
            
            return {
                "regimes": regimes,
                "locations": location_labels,
                "total_records": len(features_df),
                "regime_distribution": dict(pd.Series(regimes).value_counts()),
                "location_distribution": dict(pd.Series(location_labels).value_counts()),
            }
            
        except Exception as e:
            self.logger.error(f"Error in regime classification: {e}")
            return {"error": str(e)}

    def get_system_status(self) -> dict[str, Any]:
        """Get system status and statistics."""
        return {
            "trained": self.trained,
            "last_training_time": self.last_training_time.isoformat()
            if self.last_training_time
            else None,
            "hmm_model_loaded": self.hmm_model is not None,
            "basic_ensemble_loaded": self.basic_ensemble is not None,
            "location_classifier_loaded": self.location_classifier is not None,
            # Legacy S/R code removed
            "n_states": self.n_states,
            "target_timeframe": self.target_timeframe,
            "state_to_regime_map": self.state_to_regime_map,
        }
