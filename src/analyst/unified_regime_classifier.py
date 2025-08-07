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

    def __init__(self, config: dict[str, Any]):
        self.config = config.get("analyst", {}).get("unified_regime_classifier", {})
        self.global_config = config
        self.logger = system_logger.getChild("UnifiedRegimeClassifier")

        # HMM Configuration - now 4 states for 4 regimes
        self.n_states = self.config.get("n_states", 4)  # BULL, BEAR, SIDEWAYS, VOLATILE
        self.n_iter = self.config.get("n_iter", 100)
        self.random_state = self.config.get("random_state", 42)
        self.target_timeframe = self.config.get(
            "target_timeframe",
            "1h",
        )  # Strategist works on 1h timeframe
        self.volatility_period = self.config.get("volatility_period", 10)
        self.min_data_points = self.config.get("min_data_points", 1000)

        # Models
        self.hmm_model = None
        self.scaler = None
        self.state_to_regime_map = {}

        # Ensemble Models for Basic Regimes
        self.basic_ensemble = None

        # Location Classifier
        self.location_classifier = None
        self.location_label_encoder = None

        # Legacy S/R/Candle code removed integration for location classification
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
            f"unified_hmm_model_{self.target_timeframe}.joblib",
        )
        self.ensemble_model_path = os.path.join(
            self.model_dir,
            f"unified_ensemble_model_{self.target_timeframe}.joblib",
        )
        self.location_model_path = os.path.join(
            self.model_dir,
            f"unified_location_model_{self.target_timeframe}.joblib",
        )


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
                f"Insufficient data: {len(klines_df)} < {min_data_points}",
            )
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
        

        # Clean features
        features_df = features_df.dropna()

        self.logger.info(
            f"✅ Calculated comprehensive features for {len(features_df)} periods (dropped {len(klines_df) - len(features_df)} rows)",
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
            
            # Define the ADX threshold for a sideways market
            adx_sideways_threshold = self.config.get("adx_sideways_threshold", 25)
    
            for state in range(self.n_states):
                state_data = analysis_df[analysis_df["state"] == state]
    
                if len(state_data) == 0:
                    continue
    
                # Calculate state characteristics
                mean_return = state_data["log_returns"].mean()
                mean_volatility = state_data["volatility_20"].mean()
                mean_adx = state_data["adx"].mean() # Calculate the mean ADX for the state
                mean_atr_norm = state_data["atr_normalized"].mean()
    
                # New regime classification logic with ADX
                if mean_volatility > 0.015 or mean_atr_norm > 0.02:
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

    def _classify_location(
        self,
        current_price: float,
        sr_levels: dict[str, Any],
    ) -> str:
        """
        Classify current location relative to S/R levels.

        Args:
            current_price: Current market price
            sr_levels: Support and resistance levels from SR analyzer

        Returns:
            Location classification: SUPPORT, RESISTANCE, or OPEN_RANGE
        """
        try:
            if sr_levels is None or not sr_levels:
                return "OPEN_RANGE"

            support_levels = sr_levels.get("support_levels", [])
            resistance_levels = sr_levels.get("resistance_levels", [])

            if not support_levels and not resistance_levels:
                return "OPEN_RANGE"

            # Tolerance for proximity (2% of current price)
            tolerance = current_price * 0.02

            # Check proximity to support levels
            for level in support_levels:
                if isinstance(level, dict):
                    level_price = level.get("price", 0)
                elif isinstance(level, (int, float)):
                    level_price = float(level)
                else:
                    continue

                if abs(current_price - level_price) <= tolerance:
                    return "SUPPORT"

            # Check proximity to resistance levels
            for level in resistance_levels:
                if isinstance(level, dict):
                    level_price = level.get("price", 0)
                elif isinstance(level, (int, float)):
                    level_price = float(level)
                else:
                    continue

                if abs(current_price - level_price) <= tolerance:
                    return "RESISTANCE"

            return "OPEN_RANGE"

        except Exception as e:
            self.logger.error(f"Error in location classification: {e}")
            return "OPEN_RANGE"

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
        Train location classifier for SUPPORT, RESISTANCE, OPEN_RANGE.
        """
        try:
            self.logger.info("🎓 Training Location Classifier...")

            # Legacy S/R code removed
            return False

            for i, row in features_df.iterrows():
                current_price = row["close"]
                location = self._classify_location(current_price, sr_levels)
                location_labels.append(location)

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
                current_price = current_features["close"]
                sr_levels = {
                    # Legacy S/R code removed,
                    # Legacy S/R code removed
                }
                location = self._classify_location(current_price, sr_levels)
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
