# src/analyst/unified_regime_classifier.py
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from hmmlearn import hmm
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC

from src.config import CONFIG
from src.utils.logger import system_logger


class UnifiedRegimeClassifier:
    """
    Unified Market Regime Classifier with HMM-based labeling and ensemble prediction.
    
    Approach:
    1. HMM-based labeling for basic regimes (BULL, BEAR, SIDEWAYS)
    2. Ensemble prediction with majority voting for basic regimes
    3. Advanced regime classification for special cases (HUGE_CANDLE, SR_ZONE_ACTION)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("analyst", {}).get("unified_regime_classifier", {})
        self.global_config = config
        self.logger = system_logger.getChild("UnifiedRegimeClassifier")

        # HMM Configuration
        self.n_states = self.config.get("n_states", 4)
        self.n_iter = self.config.get("n_iter", 100)
        self.random_state = self.config.get("random_state", 42)
        self.target_timeframe = self.config.get("target_timeframe", "1h")
        self.volatility_period = self.config.get("volatility_period", 10)
        self.min_data_points = self.config.get("min_data_points", 1000)

        # Models
        self.hmm_model = None
        self.scaler = None
        self.state_to_regime_map = {}
        
        # Ensemble Models for Basic Regimes
        self.basic_ensemble = None
        
        # SR Analyzer integration
        self.sr_analyzer = None
        self.enable_sr_integration = self.config.get("enable_sr_integration", True)
        self.basic_label_encoder = None
        
        # Advanced Classification Models
        self.advanced_classifier = None
        self.advanced_label_encoder = None
        
        # Training Status
        self.trained = False
        self.last_training_time = None

        # SR Analyzer integration
        self.sr_analyzer = None
        self.enable_sr_integration = self.config.get("enable_sr_integration", True)
        
        # Candle Analyzer integration
        self.candle_analyzer = None
        self.enable_candle_integration = self.config.get("enable_candle_integration", True)

        # Model Paths
        self.model_dir = os.path.join(CONFIG["CHECKPOINT_DIR"], "analyst_models")
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.hmm_model_path = os.path.join(
            self.model_dir, 
            f"unified_hmm_model_{self.target_timeframe}.joblib"
        )
        self.ensemble_model_path = os.path.join(
            self.model_dir, 
            f"unified_ensemble_model_{self.target_timeframe}.joblib"
        )
        self.advanced_model_path = os.path.join(
            self.model_dir, 
            f"unified_advanced_model_{self.target_timeframe}.joblib"
        )

    async def _initialize_sr_analyzer(self) -> None:
        """
        Initialize SR analyzer for enhanced regime classification.
        """
        if not self.enable_sr_integration:
            self.logger.info("SR integration disabled")
            return
            
        try:
            from src.analyst.sr_analyzer import SRLevelAnalyzer
            
            self.sr_analyzer = SRLevelAnalyzer(self.global_config)
            await self.sr_analyzer.initialize()
            self.logger.info("✅ SR analyzer initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing SR analyzer: {e}")
            self.sr_analyzer = None

    async def _analyze_sr_levels(self, historical_klines: pd.DataFrame) -> None:
        """
        Analyze historical data to populate SR levels for detection.
        """
        if not self.sr_analyzer:
            self.logger.warning("SR analyzer not available for analysis")
            return
            
        try:
            self.logger.info("🔍 Analyzing historical data for SR levels...")
            analysis_result = await self.sr_analyzer.analyze(historical_klines)
            
            if analysis_result:
                support_count = len(analysis_result.get('support_levels', []))
                resistance_count = len(analysis_result.get('resistance_levels', []))
                self.logger.info(f"📊 SR Analysis completed: {support_count} support levels, {resistance_count} resistance levels")
            else:
                self.logger.warning("⚠️ SR analysis returned no results")
                
        except Exception as e:
            self.logger.error(f"Error analyzing SR levels: {e}")
            self.sr_analyzer = None

    async def _initialize_candle_analyzer(self) -> None:
        """
        Initialize candle analyzer for enhanced regime classification.
        """
        if not self.enable_candle_integration:
            self.logger.info("Candle integration disabled")
            return
            
        try:
            from src.analyst.candle_analyzer import CandleAnalyzer
            
            self.candle_analyzer = CandleAnalyzer(self.global_config)
            await self.candle_analyzer.initialize()
            self.logger.info("✅ Candle analyzer initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing candle analyzer: {e}")
            self.candle_analyzer = None

    def _calculate_features(self, klines_df: pd.DataFrame, min_data_points: int = None) -> pd.DataFrame:
        """
        Calculate comprehensive features for regime classification.
        
        Args:
            klines_df: DataFrame with OHLCV data
            min_data_points: Override minimum data points requirement
            
        Returns:
            DataFrame with calculated features
        """
        try:
            if klines_df.empty:
                self.logger.warning("Empty klines data provided")
                return pd.DataFrame()

            # Ensure required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in klines_df.columns]
            if missing_cols:
                self.logger.warning(f"Missing required columns: {missing_cols}")
                return pd.DataFrame()

            # Clean data
            initial_rows = len(klines_df)
            klines_df = klines_df.dropna(subset=required_cols)
            if len(klines_df) < initial_rows:
                self.logger.warning(f"Removed {initial_rows - len(klines_df)} rows with NaN")

            # Use provided minimum or default
            effective_min_data_points = min_data_points if min_data_points is not None else self.min_data_points
            
            if len(klines_df) < effective_min_data_points:
                self.logger.warning(f"Insufficient data: {len(klines_df)} < {effective_min_data_points}")
                return pd.DataFrame()

            features = pd.DataFrame(index=klines_df.index)

            # Basic price features
            features["log_returns"] = np.log(klines_df["close"] / klines_df["close"].shift(1))
            features["log_returns"] = features["log_returns"].replace([np.inf, -np.inf], np.nan).fillna(0)
            
            features["volatility_20"] = features["log_returns"].rolling(window=20, min_periods=1).std()
            features["volatility_20"] = features["volatility_20"].fillna(0)

            # Volume features
            features["volume_ratio"] = klines_df["volume"] / klines_df["volume"].rolling(window=20, min_periods=1).mean()
            features["volume_ratio"] = features["volume_ratio"].fillna(1.0)

            # Technical indicators
            features["rsi"] = self._calculate_rsi(klines_df["close"])
            macd_data = self._calculate_macd(klines_df["close"])
            features["macd"] = macd_data["macd"]
            features["macd_signal"] = macd_data["signal"]
            features["macd_histogram"] = macd_data["histogram"]

            bb_data = self._calculate_bollinger_bands(klines_df["close"])
            features["bb_position"] = bb_data["position"]
            features["bb_width"] = bb_data["width"]

            features["atr"] = self._calculate_atr(klines_df)

            # Candle features
            features["candle_body_size"] = abs(klines_df["close"] - klines_df["open"])
            features["candle_total_size"] = klines_df["high"] - klines_df["low"]
            
            total_size_safe = features["candle_total_size"].replace(0, np.nan)
            features["candle_body_ratio"] = features["candle_body_size"] / total_size_safe
            features["candle_body_ratio"] = features["candle_body_ratio"].fillna(0.5)

            # Price momentum
            features["price_momentum_5"] = klines_df["close"].pct_change(5)
            features["price_momentum_10"] = klines_df["close"].pct_change(10)
            features["price_momentum_20"] = klines_df["close"].pct_change(20)

            # Clean features
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.fillna(0)

            self.logger.info(f"Calculated {len(features.columns)} features for {len(features)} periods")
            return features

        except Exception as e:
            self.logger.error(f"Error calculating features: {e}")
            return pd.DataFrame()

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rs = rs.replace([np.inf, -np.inf], 0).fillna(0)
            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.replace([np.inf, -np.inf], 45).fillna(45)
            return rsi
        except Exception as e:
            self.logger.warning(f"Error calculating RSI: {e}")
            return pd.Series(45.0, index=prices.index)

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD indicator."""
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

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            band_range = upper_band - lower_band
            band_range = band_range.replace(0, np.nan)
            position = (prices - lower_band) / band_range
            position = position.replace([np.inf, -np.inf], 0.5).fillna(0.5)
            
            width = band_range / sma
            width = width.replace([np.inf, -np.inf], 0).fillna(0)
            
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

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
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

    def _interpret_hmm_states(self, features_df: pd.DataFrame, state_sequence: np.ndarray) -> Dict:
        """
        Interpret HMM states and map them to basic market regimes.
        
        Args:
            features_df: DataFrame with features
            state_sequence: Array of predicted state labels
            
        Returns:
            Dictionary mapping state indices to regime labels
        """
        analysis_df = features_df.copy()
        analysis_df["state"] = state_sequence

        state_analysis = {}

        for state in range(self.n_states):
            state_data = analysis_df[analysis_df["state"] == state]

            if len(state_data) == 0:
                continue

            # Calculate state characteristics
            mean_return = state_data["log_returns"].mean()
            mean_volatility = state_data["volatility_20"].mean()

            # Classify into basic regimes
            if mean_return > 0.0001 and mean_volatility < 0.5:
                regime = "BULL"
            elif mean_return < -0.0001 and mean_volatility < 0.5:
                regime = "BEAR"
            else:
                regime = "SIDEWAYS"

            state_analysis[state] = {
                "regime": regime,
                "mean_return": mean_return,
                "mean_volatility": mean_volatility,
                "count": len(state_data),
            }

            self.logger.info(
                f"State {state}: {regime} "
                f"(mean_return={mean_return:.4f}, mean_vol={mean_volatility:.4f})"
            )

        # Create state to regime mapping
        state_to_regime_map = {state: data["regime"] for state, data in state_analysis.items()}
        state_analysis["state_to_regime_map"] = state_to_regime_map

        return state_analysis

    def _detect_advanced_regimes(self, klines_df: pd.DataFrame, features_df: pd.DataFrame) -> List[str]:
        """
        Detect advanced regimes: HUGE_CANDLE, SR_ZONE_ACTION.
        
        Args:
            klines_df: DataFrame with OHLCV data
            features_df: DataFrame with calculated features
            
        Returns:
            List of advanced regime labels
        """
        self.logger.info(f"🔍 Starting advanced regime detection for {len(klines_df)} periods...")
        self.logger.info(f"📊 Candle analyzer available: {self.candle_analyzer is not None}")
        self.logger.info(f"📊 SR analyzer available: {self.sr_analyzer is not None}")
        
        advanced_regimes = []
        huge_candle_count = 0
        sr_zone_count = 0
        normal_count = 0
        
        for i in range(len(klines_df)):
            if i < 20:  # Not enough data for indicators
                advanced_regimes.append("NORMAL")
                normal_count += 1
                continue
                
            current_candle = klines_df.iloc[i]
            current_features = features_df.iloc[i]
            
            # HUGE_CANDLE detection with candle analyzer
            if self.candle_analyzer:
                try:
                    # Get current candle data
                    current_candle_data = {
                        'open': current_candle['open'],
                        'high': current_candle['high'],
                        'low': current_candle['low'],
                        'close': current_candle['close'],
                        'volume': current_candle.get('volume', 0)
                    }
                    
                    # Use candle analyzer to detect large candles
                    candle_analysis = self.candle_analyzer.detect_large_candle(current_candle_data)
                    
                    if candle_analysis.get('is_large', False) and candle_analysis.get('size_class', '') in ['huge', 'extreme']:
                        advanced_regimes.append("HUGE_CANDLE")
                        huge_candle_count += 1
                        continue
                except Exception as e:
                    self.logger.warning(f"Error in candle analysis: {e}")
                    # Fallback to simplified detection with more lenient threshold
                    candle_size = current_candle['high'] - current_candle['low']
                    avg_candle_size = (klines_df['high'] - klines_df['low']).iloc[i-20:i].mean()
                    
                    # More lenient threshold for 1-hour data: 2x average size instead of 3x
                    if candle_size > avg_candle_size * 2:
                        advanced_regimes.append("HUGE_CANDLE")
                        huge_candle_count += 1
                        continue
            else:
                # Fallback to simplified detection with more lenient threshold
                candle_size = current_candle['high'] - current_candle['low']
                avg_candle_size = (klines_df['high'] - klines_df['low']).iloc[i-20:i].mean()
                
                # More lenient threshold for 1-hour data: 2x average size instead of 3x
                if candle_size > avg_candle_size * 2:
                    advanced_regimes.append("HUGE_CANDLE")
                    huge_candle_count += 1
                    continue
                
            # SR_ZONE_ACTION detection with SR analyzer
            if self.sr_analyzer:
                try:
                    # Get current price
                    current_price = current_candle['close']
                    
                    # Check SR zone proximity
                    sr_proximity = self.sr_analyzer.detect_sr_zone_proximity(current_price)
                    
                    if sr_proximity.get('in_zone', False):
                        advanced_regimes.append("SR_ZONE_ACTION")
                        sr_zone_count += 1
                        continue
                except Exception as e:
                    self.logger.warning(f"Error in SR zone detection: {e}")
                    # Fallback to simplified detection with more lenient threshold
                    price_position = current_features.get('bb_position', 0.5)
                    # More lenient threshold: 0.2 or 0.8 instead of 0.1 or 0.9
                    if price_position < 0.2 or price_position > 0.8:
                        advanced_regimes.append("SR_ZONE_ACTION")
                        sr_zone_count += 1
                        continue
            else:
                # Fallback to simplified detection with more lenient threshold
                price_position = current_features.get('bb_position', 0.5)
                # More lenient threshold: 0.2 or 0.8 instead of 0.1 or 0.9
                if price_position < 0.2 or price_position > 0.8:
                    advanced_regimes.append("SR_ZONE_ACTION")
                    sr_zone_count += 1
                    continue
                
            advanced_regimes.append("NORMAL")
            normal_count += 1
            
        self.logger.info(f"📊 Advanced regime detection results:")
        self.logger.info(f"   - HUGE_CANDLE: {huge_candle_count} periods")
        self.logger.info(f"   - SR_ZONE_ACTION: {sr_zone_count} periods")
        self.logger.info(f"   - NORMAL: {normal_count} periods")
        self.logger.info(f"   - Total special cases: {huge_candle_count + sr_zone_count}")
        
        return advanced_regimes

    async def train_hmm_labeler(self, historical_klines: pd.DataFrame) -> bool:
        """
        Step 1: Train HMM-based labeler for basic regimes.
        
        Args:
            historical_klines: Historical OHLCV data
            
        Returns:
            True if training successful, False otherwise
        """
        self.logger.info("🎯 Step 1: Training HMM-based labeler...")

        if historical_klines.empty:
            self.logger.error("No historical data provided")
            return False

        # Calculate features
        features_df = self._calculate_features(historical_klines)
        if features_df.empty:
            self.logger.error("Failed to calculate features")
            return False

        # Prepare HMM features (log_returns and volatility only)
        hmm_features = features_df[["log_returns", "volatility_20"]].values
        hmm_features = hmm_features[np.isfinite(hmm_features).all(axis=1)]

        if len(hmm_features) < self.n_states * 10:
            self.logger.error(f"Insufficient data for HMM: {len(hmm_features)} < {self.n_states * 10}")
            return False

        # Scale features
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

            self.hmm_model.fit(scaled_features)
            state_sequence = self.hmm_model.predict(scaled_features)

            # Interpret states
            state_analysis = self._interpret_hmm_states(
                features_df.iloc[:len(scaled_features)],
                state_sequence
            )

            self.state_to_regime_map = state_analysis["state_to_regime_map"]
            
            self.logger.info("✅ HMM labeler trained successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error training HMM: {e}")
            return False

    async def train_basic_ensemble(self, historical_klines: pd.DataFrame) -> bool:
        """
        Step 2: Train ensemble for basic regime prediction (BULL, BEAR, SIDEWAYS).
        
        Args:
            historical_klines: Historical OHLCV data
            
        Returns:
            True if training successful, False otherwise
        """
        self.logger.info("🎯 Step 2: Training basic regime ensemble...")

        if self.hmm_model is None:
            self.logger.error("HMM model not trained. Run train_hmm_labeler first.")
            return False

        # Calculate features
        features_df = self._calculate_features(historical_klines)
        if features_df.empty:
            self.logger.error("Failed to calculate features")
            return False

        # Generate HMM-based labels
        hmm_features = features_df[["log_returns", "volatility_20"]].values
        hmm_features = hmm_features[np.isfinite(hmm_features).all(axis=1)]
        scaled_features = self.scaler.transform(hmm_features)
        state_sequence = self.hmm_model.predict(scaled_features)

        # Create basic regime labels
        basic_labels = []
        for state in state_sequence:
            basic_labels.append(self.state_to_regime_map.get(state, "SIDEWAYS"))

        # Prepare features for ensemble
        feature_columns = [
            "log_returns", "volatility_20", "volume_ratio", "rsi", 
            "macd", "macd_signal", "macd_histogram", "bb_position", 
            "bb_width", "atr", "price_momentum_5", "price_momentum_10", 
            "price_momentum_20"
        ]
        
        available_features = [col for col in feature_columns if col in features_df.columns]
        
        if len(available_features) < 5:
            self.logger.error("Insufficient features for ensemble training")
            return False

        X = features_df[available_features].iloc[:len(basic_labels)]
        y = basic_labels

        # Remove any rows with NaN
        valid_mask = ~X.isna().any(axis=1)
        X = X[valid_mask]
        y = [y[i] for i in range(len(y)) if valid_mask.iloc[i]]

        if len(X) < 100:
            self.logger.error("Insufficient valid data for ensemble training")
            return False

        # Encode labels
        self.basic_label_encoder = LabelEncoder()
        y_encoded = self.basic_label_encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Create ensemble
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('lgbm', LGBMClassifier(random_state=42, verbose=-1)),
            ('svm', SVC(probability=True, random_state=42))
        ]

        self.basic_ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft'
        )

        # Train ensemble
        self.basic_ensemble.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.basic_ensemble.score(X_train, y_train)
        test_score = self.basic_ensemble.score(X_test, y_test)
        
        self.logger.info(f"✅ Basic ensemble trained - Train: {train_score:.3f}, Test: {test_score:.3f}")
        return True

    async def train_advanced_classifier(self, historical_klines: pd.DataFrame) -> bool:
        """
        Step 3: Train advanced classifier for special regimes (HUGE_CANDLE, SR_ZONE_ACTION).
        
        Args:
            historical_klines: Historical OHLCV data
            
        Returns:
            True if training successful, False otherwise
        """
        self.logger.info("🎯 Step 3: Training advanced regime classifier...")

        # Calculate features
        features_df = self._calculate_features(historical_klines)
        if features_df.empty:
            self.logger.error("Failed to calculate features")
            return False

        # Detect advanced regimes
        advanced_labels = self._detect_advanced_regimes(historical_klines, features_df)
        
        # Filter to only special cases
        special_cases = []
        special_features = []
        
        for i, label in enumerate(advanced_labels):
            if label in ["HUGE_CANDLE", "SR_ZONE_ACTION"]:
                special_cases.append(label)
                special_features.append(features_df.iloc[i])

        self.logger.info(f"📊 Special cases analysis:")
        self.logger.info(f"   - Total periods analyzed: {len(advanced_labels)}")
        self.logger.info(f"   - Special cases found: {len(special_cases)}")
        self.logger.info(f"   - Special case types: {set(special_cases)}")
        
        if len(special_cases) < 50:
            self.logger.warning(f"Only {len(special_cases)} special cases found. Using all data.")
            special_cases = advanced_labels
            special_features = [features_df.iloc[i] for i in range(len(features_df))]
            self.logger.info(f"📊 Expanded to use all {len(special_cases)} cases for training")

        # Prepare features for advanced classification
        feature_columns = [
            "log_returns", "volatility_20", "volume_ratio", "rsi", 
            "macd", "bb_position", "bb_width", "atr", 
            "candle_body_ratio", "price_momentum_5"
        ]
        
        available_features = [col for col in feature_columns if col in features_df.columns]
        self.logger.info(f"📊 Using {len(available_features)} features: {available_features}")
        
        X_advanced = pd.DataFrame(special_features)[available_features]
        y_advanced = special_cases

        # Encode labels
        self.advanced_label_encoder = LabelEncoder()
        y_advanced_encoded = self.advanced_label_encoder.fit_transform(y_advanced)
        
        unique_labels = self.advanced_label_encoder.classes_
        self.logger.info(f"📊 Encoded labels: {unique_labels}")
        self.logger.info(f"📊 Label distribution: {dict(zip(unique_labels, np.bincount(y_advanced_encoded)))}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_advanced, y_advanced_encoded, test_size=0.2, random_state=42, stratify=y_advanced_encoded
        )
        
        self.logger.info(f"📊 Training set size: {len(X_train)}")
        self.logger.info(f"📊 Test set size: {len(X_test)}")

        # Train advanced classifier
        self.advanced_classifier = LGBMClassifier(
            random_state=42,
            verbose=-1,
            n_estimators=100,
            learning_rate=0.1
        )

        self.advanced_classifier.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.advanced_classifier.score(X_train, y_train)
        test_score = self.advanced_classifier.score(X_test, y_test)
        
        self.logger.info(f"✅ Advanced classifier trained - Train: {train_score:.3f}, Test: {test_score:.3f}")
        return True

    async def train_complete_system(self, historical_klines: pd.DataFrame) -> bool:
        """
        Train the complete unified regime classification system.
        
        Args:
            historical_klines: Historical OHLCV data
            
        Returns:
            True if training successful, False otherwise
        """
        self.logger.info("🚀 Training complete unified regime classification system...")

        # Initialize SR analyzer
        await self._initialize_sr_analyzer()
        
        # Analyze SR levels for detection
        await self._analyze_sr_levels(historical_klines)

        # Initialize candle analyzer
        await self._initialize_candle_analyzer()

        # Step 1: Train HMM labeler
        if not await self.train_hmm_labeler(historical_klines):
            return False

        # Step 2: Train basic ensemble
        if not await self.train_basic_ensemble(historical_klines):
            return False

        # Step 3: Train advanced classifier
        if not await self.train_advanced_classifier(historical_klines):
            return False

        # Save models
        self.save_models()
        
        self.trained = True
        self.last_training_time = datetime.now()
        
        self.logger.info("✅ Complete unified regime classification system trained successfully")
        return True

    def predict_regime(self, current_klines: pd.DataFrame) -> Tuple[str, float, Dict]:
        """
        Predict market regime using the unified system.
        
        Args:
            current_klines: Current OHLCV data
            
        Returns:
            Tuple of (regime, confidence, additional_info)
        """
        if not self.trained:
            self.logger.error("System not trained")
            return "UNKNOWN", 0.0, {}

        # Calculate features (use smaller minimum for prediction)
        features_df = self._calculate_features(current_klines, min_data_points=50)
        if features_df.empty:
            return "UNKNOWN", 0.0, {}

        # Get latest features
        latest_features = features_df.iloc[-1:]

        # Step 1: Get basic regime prediction from ensemble
        basic_regime = "SIDEWAYS"
        basic_confidence = 0.0
        
        if self.basic_ensemble is not None:
            try:
                feature_columns = [
                    "log_returns", "volatility_20", "volume_ratio", "rsi", 
                    "macd", "macd_signal", "macd_histogram", "bb_position", 
                    "bb_width", "atr", "price_momentum_5", "price_momentum_10", 
                    "price_momentum_20"
                ]
                
                available_features = [col for col in feature_columns if col in latest_features.columns]
                X_basic = latest_features[available_features]
                
                if not X_basic.isna().any().any():
                    basic_pred = self.basic_ensemble.predict(X_basic)[0]
                    basic_proba = self.basic_ensemble.predict_proba(X_basic)[0]
                    basic_regime = self.basic_label_encoder.inverse_transform([basic_pred])[0]
                    basic_confidence = np.max(basic_proba)
                    
            except Exception as e:
                self.logger.warning(f"Error in basic ensemble prediction: {e}")

        # Step 2: Check for advanced regimes
        advanced_regime = "NORMAL"
        advanced_confidence = 0.0
        
        if self.advanced_classifier is not None:
            try:
                feature_columns = [
                    "log_returns", "volatility_20", "volume_ratio", "rsi", 
                    "macd", "bb_position", "bb_width", "atr", 
                    "candle_body_ratio", "price_momentum_5"
                ]
                
                available_features = [col for col in feature_columns if col in latest_features.columns]
                X_advanced = latest_features[available_features]
                
                if not X_advanced.isna().any().any():
                    advanced_pred = self.advanced_classifier.predict(X_advanced)[0]
                    advanced_proba = self.advanced_classifier.predict_proba(X_advanced)[0]
                    advanced_regime = self.advanced_label_encoder.inverse_transform([advanced_pred])[0]
                    advanced_confidence = np.max(advanced_proba)
                    
            except Exception as e:
                self.logger.warning(f"Error in advanced classifier prediction: {e}")

        # Step 3: Enhanced HUGE_CANDLE detection with candle analyzer
        if self.candle_analyzer:
            try:
                current_candle = {
                    'open': current_klines['open'].iloc[-1],
                    'high': current_klines['high'].iloc[-1],
                    'low': current_klines['low'].iloc[-1],
                    'close': current_klines['close'].iloc[-1],
                    'volume': current_klines['volume'].iloc[-1] if 'volume' in current_klines.columns else 0
                }
                
                candle_analysis = self.candle_analyzer.detect_large_candle(current_candle)
                
                if candle_analysis.get('is_large', False) and candle_analysis.get('size_class', '') in ['huge', 'extreme']:
                    additional_info["candle_analysis"] = candle_analysis
                    return "HUGE_CANDLE", 0.8, additional_info
            except Exception as e:
                self.logger.warning(f"Candle analysis failed: {e}")
        
        # Step 4: Enhanced SR_ZONE_ACTION detection with SR analyzer
        if self.sr_analyzer:
            try:
                current_price = current_klines['close'].iloc[-1]
                sr_proximity = self.sr_analyzer.detect_sr_zone_proximity(current_price)
                
                if sr_proximity.get('in_zone', False):
                    additional_info["sr_zone_info"] = sr_proximity
                    return "SR_ZONE_ACTION", 0.8, additional_info
            except Exception as e:
                self.logger.warning(f"SR zone detection failed: {e}")
        
        # Step 5: Combine predictions
        if advanced_regime in ["HUGE_CANDLE", "SR_ZONE_ACTION"] and advanced_confidence > 0.7:
            final_regime = advanced_regime
            final_confidence = advanced_confidence
        else:
            final_regime = basic_regime
            final_confidence = basic_confidence

        additional_info = {
            "basic_regime": basic_regime,
            "basic_confidence": basic_confidence,
            "advanced_regime": advanced_regime,
            "advanced_confidence": advanced_confidence,
            "features_used": len(latest_features.columns),
            "timestamp": datetime.now()
        }

        return final_regime, final_confidence, additional_info

    def save_models(self) -> None:
        """Save all trained models."""
        try:
            # Save HMM model
            hmm_data = {
                "hmm_model": self.hmm_model,
                "scaler": self.scaler,
                "state_to_regime_map": self.state_to_regime_map,
                "n_states": self.n_states,
                "config": self.config
            }
            joblib.dump(hmm_data, self.hmm_model_path)
            self.logger.info(f"HMM model saved to {self.hmm_model_path}")

            # Save basic ensemble
            ensemble_data = {
                "ensemble": self.basic_ensemble,
                "label_encoder": self.basic_label_encoder,
                "config": self.config
            }
            joblib.dump(ensemble_data, self.ensemble_model_path)
            self.logger.info(f"Basic ensemble saved to {self.ensemble_model_path}")

            # Save advanced classifier
            advanced_data = {
                "classifier": self.advanced_classifier,
                "label_encoder": self.advanced_label_encoder,
                "config": self.config
            }
            joblib.dump(advanced_data, self.advanced_model_path)
            self.logger.info(f"Advanced classifier saved to {self.advanced_model_path}")

        except Exception as e:
            self.logger.error(f"Error saving models: {e}")

    def load_models(self) -> bool:
        """Load all trained models."""
        try:
            # Load HMM model
            if os.path.exists(self.hmm_model_path):
                hmm_data = joblib.load(self.hmm_model_path)
                self.hmm_model = hmm_data["hmm_model"]
                self.scaler = hmm_data["scaler"]
                self.state_to_regime_map = hmm_data["state_to_regime_map"]
                self.logger.info("HMM model loaded successfully")
            else:
                self.logger.warning("HMM model file not found")
                return False

            # Load basic ensemble
            if os.path.exists(self.ensemble_model_path):
                ensemble_data = joblib.load(self.ensemble_model_path)
                self.basic_ensemble = ensemble_data["ensemble"]
                self.basic_label_encoder = ensemble_data["label_encoder"]
                self.logger.info("Basic ensemble loaded successfully")
            else:
                self.logger.warning("Basic ensemble file not found")
                return False

            # Load advanced classifier
            if os.path.exists(self.advanced_model_path):
                advanced_data = joblib.load(self.advanced_model_path)
                self.advanced_classifier = advanced_data["classifier"]
                self.advanced_label_encoder = advanced_data["label_encoder"]
                self.logger.info("Advanced classifier loaded successfully")
            else:
                self.logger.warning("Advanced classifier file not found")
                return False

            self.trained = True
            self.logger.info("All models loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return False

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and statistics."""
        return {
            "trained": self.trained,
            "last_training_time": self.last_training_time,
            "hmm_states": self.n_states,
            "state_mapping": self.state_to_regime_map,
            "models_loaded": {
                "hmm_model": self.hmm_model is not None,
                "basic_ensemble": self.basic_ensemble is not None,
                "advanced_classifier": self.advanced_classifier is not None
            }
        } 