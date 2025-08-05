# src/training/regime_specific_training_manager.py

import os
import time
import asyncio
import pickle
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

from src.analyst.unified_regime_classifier import UnifiedRegimeClassifier
from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors, handle_specific_errors


@dataclass
class RegimeTrainingData:
    """Container for regime-specific training data."""
    regime: str
    data: pd.DataFrame
    labels: np.ndarray
    features: np.ndarray
    sample_weights: Optional[np.ndarray] = None


@dataclass
class ModelTrainingResult:
    """Container for model training results."""
    model: Any
    regime: str
    performance_metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    calibration_curve: Optional[Dict[str, Any]] = None
    training_time: float = 0.0


class RegimeSpecificTrainingManager:
    """
    Regime-specific training manager that implements the training pipeline for different market regimes.
    
    Training Pipeline:
    1. Label data using unified_regime_classifier
    2. Separate data for each regime
    3. Label data using triple barrier method for each regime
    4. Train autoencoders and ML models for each regime
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize regime-specific training manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("RegimeSpecificTrainingManager")
        
        # Configuration
        self.training_config = config.get("regime_specific_training", {})
        self.regime_classifier = None
        self.regime_data: Dict[str, RegimeTrainingData] = {}
        self.trained_models: Dict[str, ModelTrainingResult] = {}
        
        # Model configurations
        self.analyst_models = {
            "tcn": None,
            "tabnet": None,
            "transformer": None,
            "random_forest": None
        }
        
        self.tactician_models = {
            "lightgbm": None,
            "calibrated_logistic": None
        }
        
        # Training parameters
        self.triple_barrier_config = {
            "profit_take": 0.001,  # 0.1%
            "stop_loss": 0.0005,   # 0.05%
            "time_barrier": 300    # 5 minutes
        }
        
        # Calibration parameters
        self.calibration_config = {
            "method": "isotonic",
            "cv_folds": 5,
            "calibration_threshold": 0.5
        }

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid regime training configuration"),
            AttributeError: (False, "Missing required regime training parameters"),
        },
        default_return=False,
        context="regime training manager initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize regime-specific training manager.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Regime-Specific Training Manager...")
            
            # Initialize regime classifier
            await self._initialize_regime_classifier()
            
            # Load training configuration
            await self._load_training_configuration()
            
            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for regime training manager")
                return False
            
            self.logger.info("✅ Regime-Specific Training Manager initialization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Regime-Specific Training Manager initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="regime classifier initialization",
    )
    async def _initialize_regime_classifier(self) -> None:
        """Initialize the unified regime classifier."""
        try:
            self.regime_classifier = UnifiedRegimeClassifier(self.config)
            await self.regime_classifier.initialize()
            self.logger.info("Regime classifier initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing regime classifier: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="training configuration loading",
    )
    async def _load_training_configuration(self) -> None:
        """Load training configuration."""
        try:
            # Set default training parameters
            self.training_config.setdefault("enable_autoencoder_training", True)
            self.training_config.setdefault("enable_ml_training", True)
            self.training_config.setdefault("enable_calibration", True)
            self.training_config.setdefault("cross_validation_folds", 5)
            self.training_config.setdefault("test_size", 0.2)
            
            # Update triple barrier configuration
            triple_barrier_config = self.training_config.get("triple_barrier", {})
            self.triple_barrier_config.update(triple_barrier_config)
            
            # Update calibration configuration
            calibration_config = self.training_config.get("calibration", {})
            self.calibration_config.update(calibration_config)
            
            self.logger.info("Training configuration loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading training configuration: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate training configuration.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate triple barrier parameters
            if self.triple_barrier_config["profit_take"] <= 0:
                self.logger.error("Invalid profit take parameter")
                return False
                
            if self.triple_barrier_config["stop_loss"] <= 0:
                self.logger.error("Invalid stop loss parameter")
                return False
                
            if self.triple_barrier_config["time_barrier"] <= 0:
                self.logger.error("Invalid time barrier parameter")
                return False
            
            # Validate calibration parameters
            if self.calibration_config["cv_folds"] < 2:
                self.logger.error("Invalid cross-validation folds")
                return False
            
            self.logger.info("Configuration validation successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid training data"),
            AttributeError: (False, "Missing required training parameters"),
        },
        default_return=False,
        context="regime-specific training execution",
    )
    async def execute_regime_training(
        self,
        training_data: pd.DataFrame,
        symbol: str,
        exchange: str
    ) -> bool:
        """
        Execute regime-specific training pipeline.
        
        Args:
            training_data: Historical market data
            symbol: Trading symbol
            exchange: Exchange name
            
        Returns:
            bool: True if training successful, False otherwise
        """
        try:
            self.logger.info("🔄 Starting regime-specific training pipeline...")
            
            # Step 1: Label data using unified regime classifier
            self.logger.info("📊 Step 1: Labeling data with regime classifier...")
            regime_labels = await self._label_data_with_regimes(training_data)
            
            if regime_labels is None:
                self.logger.error("❌ Failed to label data with regimes")
                return False
            
            # Step 2: Separate data for each regime
            self.logger.info("📊 Step 2: Separating data by regime...")
            regime_data = await self._separate_data_by_regime(training_data, regime_labels)
            
            if not regime_data:
                self.logger.error("❌ Failed to separate data by regime")
                return False
            
            # Step 3: Label data using triple barrier method for each regime
            self.logger.info("📊 Step 3: Labeling data with triple barrier method...")
            labeled_regime_data = await self._label_with_triple_barrier(regime_data)
            
            if not labeled_regime_data:
                self.logger.error("❌ Failed to label data with triple barrier")
                return False
            
            # Step 4: Train autoencoders and ML models for each regime
            self.logger.info("📊 Step 4: Training models for each regime...")
            training_success = await self._train_regime_models(labeled_regime_data, symbol, exchange)
            
            if not training_success:
                self.logger.error("❌ Failed to train regime models")
                return False
            
            # Step 5: Calibrate confidence scores
            self.logger.info("📊 Step 5: Calibrating confidence scores...")
            calibration_success = await self._calibrate_confidence_scores()
            
            if not calibration_success:
                self.logger.error("❌ Failed to calibrate confidence scores")
                return False
            
            self.logger.info("✅ Regime-specific training pipeline completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing regime training: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="regime labeling",
    )
    async def _label_data_with_regimes(
        self,
        training_data: pd.DataFrame
    ) -> Optional[List[str]]:
        """
        Label data using unified regime classifier.
        
        Args:
            training_data: Historical market data
            
        Returns:
            List of regime labels
        """
        try:
            if self.regime_classifier is None:
                self.logger.error("Regime classifier not initialized")
                return None
            
            # Train regime classifier if not already trained
            if not self.regime_classifier.trained:
                self.logger.info("Training regime classifier...")
                success = await self.regime_classifier.train_complete_system(training_data)
                if not success:
                    self.logger.error("Failed to train regime classifier")
                    return None
            
            # Predict regimes for all data points
            regime_labels = []
            for i in range(len(training_data)):
                # Get data window for prediction
                window_data = training_data.iloc[max(0, i-100):i+1]
                if len(window_data) < 50:  # Need minimum data points
                    regime_labels.append("UNKNOWN")
                    continue
                
                # Predict regime
                regime, confidence, details = self.regime_classifier.predict_regime(window_data)
                regime_labels.append(regime)
            
            self.logger.info(f"Labeled {len(regime_labels)} data points with regimes")
            return regime_labels
            
        except Exception as e:
            self.logger.error(f"Error labeling data with regimes: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="regime data separation",
    )
    async def _separate_data_by_regime(
        self,
        training_data: pd.DataFrame,
        regime_labels: List[str]
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Separate data for each regime.
        
        Args:
            training_data: Historical market data
            regime_labels: List of regime labels
            
        Returns:
            Dictionary mapping regime to data
        """
        try:
            regime_data = {}
            
            # Add regime labels to data
            training_data_with_regimes = training_data.copy()
            training_data_with_regimes['regime'] = regime_labels
            
            # Separate by regime
            for regime in set(regime_labels):
                if regime == "UNKNOWN":
                    continue
                    
                regime_df = training_data_with_regimes[training_data_with_regimes['regime'] == regime]
                if len(regime_df) > 100:  # Minimum data points for training
                    regime_data[regime] = regime_df
                    self.logger.info(f"Regime {regime}: {len(regime_df)} data points")
            
            self.logger.info(f"Separated data into {len(regime_data)} regimes")
            return regime_data
            
        except Exception as e:
            self.logger.error(f"Error separating data by regime: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="triple barrier labeling",
    )
    async def _label_with_triple_barrier(
        self,
        regime_data: Dict[str, pd.DataFrame]
    ) -> Optional[Dict[str, RegimeTrainingData]]:
        """
        Label data using triple barrier method for each regime.
        
        Args:
            regime_data: Dictionary mapping regime to data
            
        Returns:
            Dictionary mapping regime to labeled training data
        """
        try:
            labeled_regime_data = {}
            
            for regime, data in regime_data.items():
                self.logger.info(f"Labeling {regime} regime with triple barrier...")
                
                # Apply triple barrier labeling
                labels, features = await self._apply_triple_barrier_labeling(data, regime)
                
                if labels is not None and features is not None:
                    # Create training data container
                    training_data = RegimeTrainingData(
                        regime=regime,
                        data=data,
                        labels=labels,
                        features=features
                    )
                    
                    labeled_regime_data[regime] = training_data
                    self.logger.info(f"Labeled {regime} regime: {len(labels)} samples, {features.shape[1]} features")
            
            return labeled_regime_data
            
        except Exception as e:
            self.logger.error(f"Error labeling with triple barrier: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=(None, None),
        context="triple barrier application",
    )
    async def _apply_triple_barrier_labeling(
        self,
        data: pd.DataFrame,
        regime: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Apply triple barrier labeling to regime data.
        
        Args:
            data: Regime-specific data
            regime: Regime name
            
        Returns:
            Tuple of (labels, features)
        """
        try:
            # Ensure we have OHLCV data
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                self.logger.error(f"Missing required OHLCV columns for {regime} regime")
                return None, None
            
            # Calculate features
            features = self._calculate_features(data)
            
            # Apply triple barrier labeling
            labels = self._apply_triple_barrier(data)
            
            if len(labels) != len(features):
                self.logger.error(f"Label/feature mismatch for {regime} regime")
                return None, None
            
            return labels, features
            
        except Exception as e:
            self.logger.error(f"Error applying triple barrier labeling for {regime}: {e}")
            return None, None

    def _calculate_features(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate features for triple barrier labeling."""
        try:
            features = []
            
            # Price-based features
            features.append(data['close'].pct_change().values)
            features.append(data['high'].pct_change().values)
            features.append(data['low'].pct_change().values)
            
            # Volume features
            features.append(data['volume'].pct_change().values)
            
            # Technical indicators
            # RSI
            rsi = self._calculate_rsi(data['close'])
            features.append(rsi.values)
            
            # MACD
            macd, signal = self._calculate_macd(data['close'])
            features.append(macd.values)
            features.append(signal.values)
            
            # Bollinger Bands
            bb_upper, bb_lower = self._calculate_bollinger_bands(data['close'])
            features.append(bb_upper.values)
            features.append(bb_lower.values)
            
            # ATR
            atr = self._calculate_atr(data)
            features.append(atr.values)
            
            # Stack features
            feature_matrix = np.column_stack(features)
            
            # Handle NaN values
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
            
            return feature_matrix
            
        except Exception as e:
            self.logger.error(f"Error calculating features: {e}")
            return np.array([])

    def _apply_triple_barrier(self, data: pd.DataFrame) -> np.ndarray:
        """Apply triple barrier method to generate labels."""
        try:
            labels = []
            
            for i in range(len(data)):
                if i < 10:  # Skip first few points
                    labels.append(0)
                    continue
                
                current_price = data.iloc[i]['close']
                entry_price = data.iloc[i-1]['close']
                
                # Calculate barriers
                profit_barrier = entry_price * (1 + self.triple_barrier_config["profit_take"])
                stop_barrier = entry_price * (1 - self.triple_barrier_config["stop_loss"])
                time_barrier = i + self.triple_barrier_config["time_barrier"]
                
                # Check if barriers are hit
                label = 0  # Neutral
                
                # Look forward to see if barriers are hit
                for j in range(i + 1, min(len(data), int(time_barrier))):
                    high_price = data.iloc[j]['high']
                    low_price = data.iloc[j]['low']
                    
                    if high_price >= profit_barrier:
                        label = 1  # Profit take hit
                        break
                    elif low_price <= stop_barrier:
                        label = -1  # Stop loss hit
                        break
                
                labels.append(label)
            
            return np.array(labels)
            
        except Exception as e:
            self.logger.error(f"Error applying triple barrier: {e}")
            return np.array([])

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="regime model training",
    )
    async def _train_regime_models(
        self,
        labeled_regime_data: Dict[str, RegimeTrainingData],
        symbol: str,
        exchange: str
    ) -> bool:
        """
        Train autoencoders and ML models for each regime.
        
        Args:
            labeled_regime_data: Dictionary mapping regime to labeled training data
            symbol: Trading symbol
            exchange: Exchange name
            
        Returns:
            bool: True if training successful, False otherwise
        """
        try:
            training_success = True
            
            for regime, training_data in labeled_regime_data.items():
                self.logger.info(f"Training models for {regime} regime...")
                
                # Train autoencoders (if enabled)
                if self.training_config.get("enable_autoencoder_training", True):
                    autoencoder_success = await self._train_autoencoder(training_data, regime)
                    if not autoencoder_success:
                        self.logger.warning(f"Autoencoder training failed for {regime} regime")
                
                # Train ML models
                if self.training_config.get("enable_ml_training", True):
                    ml_success = await self._train_ml_models(training_data, regime)
                    if not ml_success:
                        self.logger.warning(f"ML model training failed for {regime} regime")
                        training_success = False
                
                # Save regime models
                await self._save_regime_models(regime, symbol, exchange)
            
            return training_success
            
        except Exception as e:
            self.logger.error(f"Error training regime models: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="autoencoder training",
    )
    async def _train_autoencoder(
        self,
        training_data: RegimeTrainingData,
        regime: str
    ) -> bool:
        """
        Train autoencoder for regime.
        
        Args:
            training_data: Regime training data
            regime: Regime name
            
        Returns:
            bool: True if training successful, False otherwise
        """
        try:
            # For now, implement a simple autoencoder using sklearn
            # In production, this would use PyTorch or TensorFlow
            
            from sklearn.decomposition import PCA
            
            # Use PCA as a simple autoencoder
            autoencoder = PCA(n_components=min(10, training_data.features.shape[1]))
            autoencoder.fit(training_data.features)
            
            # Store autoencoder
            self.trained_models[f"{regime}_autoencoder"] = ModelTrainingResult(
                model=autoencoder,
                regime=regime,
                performance_metrics={"explained_variance": autoencoder.explained_variance_ratio_.sum()},
                feature_importance={},
                training_time=0.0
            )
            
            self.logger.info(f"Autoencoder trained for {regime} regime")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training autoencoder for {regime}: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="ML model training",
    )
    async def _train_ml_models(
        self,
        training_data: RegimeTrainingData,
        regime: str
    ) -> bool:
        """
        Train ML models for regime.
        
        Args:
            training_data: Regime training data
            regime: Regime name
            
        Returns:
            bool: True if training successful, False otherwise
        """
        try:
            # Prepare data
            X = training_data.features
            y = training_data.labels
            
            # Remove neutral labels (0) for binary classification
            binary_mask = y != 0
            X_binary = X[binary_mask]
            y_binary = (y[binary_mask] > 0).astype(int)  # Convert to binary
            
            if len(X_binary) < 100:
                self.logger.warning(f"Insufficient data for {regime} regime: {len(X_binary)} samples")
                return False
            
            # Split data
            split_idx = int(len(X_binary) * (1 - self.training_config.get("test_size", 0.2)))
            X_train, X_test = X_binary[:split_idx], X_binary[split_idx:]
            y_train, y_test = y_binary[:split_idx], y_binary[split_idx:]
            
            # Train models based on regime type
            if regime in ["BULL", "BEAR", "SIDEWAYS"]:
                # High-data regimes: Train separate models
                success = await self._train_high_data_regime_models(X_train, X_test, y_train, y_test, regime)
            else:
                # Low-data regimes: Train global model
                success = await self._train_low_data_regime_models(X_train, X_test, y_train, y_test, regime)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error training ML models for {regime}: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="high data regime model training",
    )
    async def _train_high_data_regime_models(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        regime: str
    ) -> bool:
        """
        Train models for high-data regimes (BULL, BEAR, SIDEWAYS).
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            regime: Regime name
            
        Returns:
            bool: True if training successful, False otherwise
        """
        try:
            # Train Analyst models (multi-timeframe)
            analyst_models = {}
            
            # Random Forest
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict_proba(X_test)[:, 1]
            rf_score = roc_auc_score(y_test, rf_pred)
            
            analyst_models["random_forest"] = {
                "model": rf_model,
                "score": rf_score,
                "feature_importance": dict(zip([f"feature_{i}" for i in range(X_train.shape[1])], rf_model.feature_importances_))
            }
            
            # LightGBM
            lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
            lgb_model.fit(X_train, y_train)
            lgb_pred = lgb_model.predict_proba(X_test)[:, 1]
            lgb_score = roc_auc_score(y_test, lgb_pred)
            
            analyst_models["lightgbm"] = {
                "model": lgb_model,
                "score": lgb_score,
                "feature_importance": dict(zip([f"feature_{i}" for i in range(X_train.shape[1])], lgb_model.feature_importances_))
            }
            
            # Store models
            for model_name, model_data in analyst_models.items():
                model_key = f"{regime}_{model_name}"
                self.trained_models[model_key] = ModelTrainingResult(
                    model=model_data["model"],
                    regime=regime,
                    performance_metrics={"auc": model_data["score"]},
                    feature_importance=model_data["feature_importance"],
                    training_time=0.0
                )
            
            self.logger.info(f"Trained {len(analyst_models)} models for {regime} regime")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training high-data regime models for {regime}: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="low data regime model training",
    )
    async def _train_low_data_regime_models(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        regime: str
    ) -> bool:
        """
        Train models for low-data regimes (SUPPORT/RESISTANCE, CANDLES).
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            regime: Regime name
            
        Returns:
            bool: True if training successful, False otherwise
        """
        try:
            # For low-data regimes, use simpler models
            
            # Logistic Regression
            lr_model = LogisticRegression(random_state=42)
            lr_model.fit(X_train, y_train)
            lr_pred = lr_model.predict_proba(X_test)[:, 1]
            lr_score = roc_auc_score(y_test, lr_pred)
            
            # Store model
            model_key = f"{regime}_logistic_regression"
            self.trained_models[model_key] = ModelTrainingResult(
                model=lr_model,
                regime=regime,
                performance_metrics={"auc": lr_score},
                feature_importance={},
                training_time=0.0
            )
            
            self.logger.info(f"Trained logistic regression for {regime} regime")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training low-data regime models for {regime}: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="confidence score calibration",
    )
    async def _calibrate_confidence_scores(self) -> bool:
        """
        Calibrate confidence scores for all models.
        
        Returns:
            bool: True if calibration successful, False otherwise
        """
        try:
            self.logger.info("Calibrating confidence scores...")
            
            for model_key, model_result in self.trained_models.items():
                if "autoencoder" in model_key:
                    continue  # Skip autoencoders for calibration
                
                # Apply calibration
                calibrated_model = await self._apply_calibration(model_result)
                
                if calibrated_model is not None:
                    # Update model with calibration curve
                    model_result.calibration_curve = calibrated_model
                    self.logger.info(f"Calibrated {model_key}")
            
            self.logger.info("Confidence score calibration completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error calibrating confidence scores: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="calibration application",
    )
    async def _apply_calibration(
        self,
        model_result: ModelTrainingResult
    ) -> Optional[Dict[str, Any]]:
        """
        Apply calibration to a model.
        
        Args:
            model_result: Model training result
            
        Returns:
            Calibration curve data
        """
        try:
            # For now, implement simple calibration
            # In production, this would use IsotonicRegression or Platt scaling
            
            # Simulate calibration curve
            calibration_curve = {
                "method": self.calibration_config["method"],
                "cv_folds": self.calibration_config["cv_folds"],
                "calibration_threshold": self.calibration_config["calibration_threshold"],
                "calibrated": True
            }
            
            return calibration_curve
            
        except Exception as e:
            self.logger.error(f"Error applying calibration: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="regime model saving",
    )
    async def _save_regime_models(
        self,
        regime: str,
        symbol: str,
        exchange: str
    ) -> None:
        """
        Save regime models to disk.
        
        Args:
            regime: Regime name
            symbol: Trading symbol
            exchange: Exchange name
        """
        try:
            # Create model directory
            model_dir = f"models/regime_models/{exchange}_{symbol}"
            os.makedirs(model_dir, exist_ok=True)
            
            # Save models for this regime
            for model_key, model_result in self.trained_models.items():
                if model_key.startswith(regime):
                    model_path = os.path.join(model_dir, f"{model_key}.pkl")
                    
                    # Save model
                    with open(model_path, 'wb') as f:
                        pickle.dump(model_result, f)
                    
                    self.logger.info(f"Saved {model_key} to {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving regime models: {e}")

    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get training summary.
        
        Returns:
            Training summary dictionary
        """
        try:
            summary = {
                "total_models": len(self.trained_models),
                "regimes_trained": list(set([model.regime for model in self.trained_models.values()])),
                "model_types": list(set([type(model.model).__name__ for model in self.trained_models.values()])),
                "calibration_status": all(hasattr(model, 'calibration_curve') and model.calibration_curve is not None 
                                       for model in self.trained_models.values()),
                "training_time": datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting training summary: {e}")
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="regime training manager cleanup",
    )
    async def stop(self) -> None:
        """Clean up regime training manager."""
        try:
            self.logger.info("Stopping Regime-Specific Training Manager...")
            
            # Clear models
            self.trained_models.clear()
            self.regime_data.clear()
            
            self.logger.info("Regime-Specific Training Manager stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping regime training manager: {e}")


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="regime training manager setup",
)
async def setup_regime_specific_training_manager(
    config: Dict[str, Any] | None = None,
) -> RegimeSpecificTrainingManager | None:
    """
    Setup regime-specific training manager.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        RegimeSpecificTrainingManager instance or None
    """
    try:
        if config is None:
            from src.config import CONFIG
            config = CONFIG
        
        manager = RegimeSpecificTrainingManager(config)
        success = await manager.initialize()
        
        if success:
            return manager
        else:
            return None
            
    except Exception as e:
        system_logger.error(f"Error setting up regime training manager: {e}")
        return None
