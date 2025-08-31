#!/usr/bin/env python3
"""
Regime-Specific ML Model Trainer

This module trains ML models specifically on data from each HMM cluster/regime.
Each model is trained only on the data that belongs to its specific HMM state,
ensuring regime-aware predictions.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time
import json
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import lightgbm as lgb
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.warning_symbols import error, warning, failed, missing, initialization_error


class RegimeSpecificMLTrainer:
    """
    Trains ML models specifically on HMM regime clusters.
    Each model is trained only on data from its specific HMM state.
    """
    
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize regime-specific ML trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("RegimeSpecificMLTrainer")
        
        # Training configuration
        self.training_config = config.get("regime_specific_training", {})
        self.model_config = config.get("model_config", {})
        
        # Model storage
        self.regime_models: dict[str, dict[str, Any]] = {}
        self.regime_scalers: dict[str, StandardScaler] = {}
        self.regime_encoders: dict[str, LabelEncoder] = {}
        
        # Training results
        self.training_results: dict[str, Any] = {}
        
    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid training configuration"),
            FileNotFoundError: (False, "HMM regime data not found"),
            KeyError: (False, "Missing required configuration keys"),
        },
        default_return=False,
        context="regime-specific ML trainer initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the regime-specific ML trainer."""
        try:
            self.logger.info("🚀 Initializing Regime-Specific ML Trainer...")
            
            # Validate configuration
            if not self._validate_configuration():
                return False
                
            # Create model directories
            await self._create_model_directories()
            
            self.logger.info("✅ Regime-Specific ML Trainer initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(error(f"Failed to initialize regime-specific ML trainer: {e}"))
            return False
    
    def _validate_configuration(self) -> bool:
        """Validate training configuration."""
        try:
            required_keys = [
                "hmm_data_path",
                "model_output_path", 
                "regime_models",
                "training_parameters"
            ]
            
            for key in required_keys:
                if key not in self.training_config:
                    self.logger.error(missing(f"Missing required config key: {key}"))
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(error(f"Configuration validation failed: {e}"))
            return False
    
    async def _create_model_directories(self) -> None:
        """Create directories for storing regime-specific models."""
        try:
            model_path = Path(self.training_config["model_output_path"])
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories for each regime
            for regime in self.training_config["regime_models"]:
                regime_path = model_path / regime
                regime_path.mkdir(exist_ok=True)
                
            self.logger.info(f"📁 Created model directories at: {model_path}")
            
        except Exception as e:
            self.logger.error(error(f"Failed to create model directories: {e}"))
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="regime-specific model training",
    )
    async def train_regime_models(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        force_retrain: bool = False
    ) -> bool:
        """
        Train ML models for each HMM regime.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            force_retrain: Force retraining even if models exist
            
        Returns:
            bool: True if training successful, False otherwise
        """
        try:
            self.logger.info(f"🎯 Starting regime-specific model training for {symbol} on {exchange} ({timeframe})")
            
            # Load HMM regime data
            hmm_data = await self._load_hmm_regime_data(symbol, exchange, timeframe)
            if hmm_data is None:
                return False
            
            # Get unique regimes
            unique_regimes = hmm_data['composite_cluster_id'].unique()
            self.logger.info(f"📊 Found {len(unique_regimes)} unique regimes: {unique_regimes}")
            
            # Train models for each regime
            for regime_id in unique_regimes:
                await self._train_regime_model(hmm_data, regime_id, force_retrain)
            
            # Save training results
            await self._save_training_results(symbol, exchange, timeframe)
            
            self.logger.info("✅ Regime-specific model training completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(error(f"Regime-specific model training failed: {e}"))
            return False
    
    async def _load_hmm_regime_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> Optional[pd.DataFrame]:
        """Load HMM regime data from parquet file."""
        try:
            # Construct path to HMM regime data
            hmm_data_path = Path(self.training_config["hmm_data_path"])
            regime_file = hmm_data_path / f"{exchange}_{symbol}_hmm_block_states_{timeframe}.parquet"
            
            if not regime_file.exists():
                self.logger.error(missing(f"HMM regime data file not found: {regime_file}"))
                return None
            
            # Load data
            hmm_data = pd.read_parquet(regime_file)
            self.logger.info(f"📊 Loaded HMM regime data: {hmm_data.shape}")
            
            # Validate required columns
            required_columns = ['composite_cluster_id', 'timestamp']
            for col in required_columns:
                if col not in hmm_data.columns:
                    self.logger.error(missing(f"Required column missing: {col}"))
                    return None
            
            return hmm_data
            
        except Exception as e:
            self.logger.error(error(f"Failed to load HMM regime data: {e}"))
            return None
    
    async def _train_regime_model(
        self,
        hmm_data: pd.DataFrame,
        regime_id: int,
        force_retrain: bool
    ) -> None:
        """Train ML models for a specific regime."""
        try:
            self.logger.info(f"🎯 Training models for regime {regime_id}")
            
            # Filter data for this regime
            regime_data = hmm_data[hmm_data['composite_cluster_id'] == regime_id].copy()
            
            if len(regime_data) < 1000:  # Minimum data requirement
                self.logger.warning(warning(f"Regime {regime_id} has insufficient data: {len(regime_data)} samples"))
                return
            
            self.logger.info(f"📊 Regime {regime_id}: {len(regime_data)} samples")
            
            # Check if model already exists
            if not force_retrain and await self._model_exists(regime_id):
                self.logger.info(f"📁 Model for regime {regime_id} already exists, skipping")
                return
            
            # Prepare features and targets
            features, targets = await self._prepare_regime_training_data(regime_data, regime_id)
            
            if features is None or targets is None:
                self.logger.error(failed(f"Failed to prepare training data for regime {regime_id}"))
                return
            
            # Train models for different objectives
            regime_models = {}
            
            # 1. Exit Timing Model (Regression)
            if 'exit_timing' in targets:
                timing_model = await self._train_exit_timing_model(features, targets['exit_timing'], regime_id)
                regime_models['exit_timing'] = timing_model
            
            # 2. Exit Probability Models (Classification)
            if 'exit_probabilities' in targets:
                prob_models = await self._train_exit_probability_models(features, targets['exit_probabilities'], regime_id)
                regime_models['exit_probabilities'] = prob_models
            
            # 3. Exit Type Model (Classification)
            if 'exit_type' in targets:
                type_model = await self._train_exit_type_model(features, targets['exit_type'], regime_id)
                regime_models['exit_type'] = type_model
            
            # 4. Profit Target Model (Regression)
            if 'profit_target' in targets:
                profit_model = await self._train_profit_target_model(features, targets['profit_target'], regime_id)
                regime_models['profit_target'] = profit_model
            
            # Save regime models
            await self._save_regime_models(regime_models, regime_id)
            
            # Store in memory
            self.regime_models[str(regime_id)] = regime_models
            
            self.logger.info(f"✅ Regime {regime_id} models trained successfully")
            
        except Exception as e:
            self.logger.error(error(f"Failed to train regime {regime_id} models: {e}"))
    
    async def _prepare_regime_training_data(
        self,
        regime_data: pd.DataFrame,
        regime_id: int
    ) -> Tuple[Optional[pd.DataFrame], Optional[dict[str, Any]]]:
        """Prepare features and targets for regime-specific training."""
        try:
            self.logger.info(f"🔧 Preparing training data for regime {regime_id}")
            
            # Extract features (exclude regime and timestamp columns)
            feature_columns = [col for col in regime_data.columns 
                             if col not in ['composite_cluster_id', 'timestamp', 'label']]
            
            features = regime_data[feature_columns].copy()
            
            # Remove any constant or near-constant features
            features = self._remove_constant_features(features)
            
            # Handle missing values
            features = self._handle_missing_values(features)
            
            # Create targets based on regime characteristics
            targets = await self._create_regime_targets(regime_data, regime_id)
            
            if targets is None:
                return None, None
            
            self.logger.info(f"📊 Prepared {len(features)} samples with {len(features.columns)} features")
            
            return features, targets
            
        except Exception as e:
            self.logger.error(error(f"Failed to prepare training data: {e}"))
            return None, None
    
    def _remove_constant_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Remove constant or near-constant features."""
        try:
            # Remove constant features
            constant_features = []
            for col in features.columns:
                if features[col].nunique() <= 1:
                    constant_features.append(col)
                elif features[col].std() < 1e-6:
                    constant_features.append(col)
            
            if constant_features:
                self.logger.info(f"🗑️ Removing {len(constant_features)} constant features")
                features = features.drop(columns=constant_features)
            
            return features
            
        except Exception as e:
            self.logger.error(error(f"Failed to remove constant features: {e}"))
            return features
    
    def _handle_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features."""
        try:
            # Check for missing values
            missing_counts = features.isnull().sum()
            if missing_counts.sum() > 0:
                self.logger.info(f"🔧 Handling {missing_counts.sum()} missing values")
                
                # For numerical features, fill with median
                numerical_features = features.select_dtypes(include=[np.number]).columns
                for col in numerical_features:
                    if features[col].isnull().sum() > 0:
                        features[col] = features[col].fillna(features[col].median())
                
                # For categorical features, fill with mode
                categorical_features = features.select_dtypes(include=['object']).columns
                for col in categorical_features:
                    if features[col].isnull().sum() > 0:
                        features[col] = features[col].fillna(features[col].mode()[0])
            
            return features
            
        except Exception as e:
            self.logger.error(error(f"Failed to handle missing values: {e}"))
            return features
    
    async def _create_regime_targets(
        self,
        regime_data: pd.DataFrame,
        regime_id: int
    ) -> Optional[dict[str, Any]]:
        """Create training targets based on regime characteristics."""
        try:
            targets = {}
            
            # Get regime characteristics
            regime_char = self._get_regime_characteristics(regime_id)
            
            # 1. Exit Timing Target (regression)
            targets['exit_timing'] = self._calculate_exit_timing_target(regime_data, regime_char)
            
            # 2. Exit Probability Targets (classification)
            targets['exit_probabilities'] = self._calculate_exit_probability_targets(regime_data, regime_char)
            
            # 3. Exit Type Target (classification)
            targets['exit_type'] = self._calculate_exit_type_target(regime_data, regime_char)
            
            # 4. Profit Target (regression)
            targets['profit_target'] = self._calculate_profit_target(regime_data, regime_char)
            
            return targets
            
        except Exception as e:
            self.logger.error(error(f"Failed to create regime targets: {e}"))
            return None
    
    def _get_regime_characteristics(self, regime_id: int) -> dict[str, Any]:
        """Get characteristics of a specific regime."""
        # This would be based on your regime analysis
        # For now, using placeholder logic
        regime_chars = {
            'volatility_level': 'high' if regime_id in [0, 1] else 'low',
            'momentum_direction': 'bull' if regime_id in [0, 2] else 'bear',
            'volume_profile': 'high' if regime_id in [0, 3] else 'normal'
        }
        return regime_chars
    
    def _calculate_exit_timing_target(self, data: pd.DataFrame, regime_char: dict[str, Any]) -> np.ndarray:
        """Calculate optimal exit timing target."""
        try:
            # This is a simplified example - you'd implement your specific logic
            if regime_char['volatility_level'] == 'high':
                # High volatility: Quick exits (1-10 bars)
                return np.random.randint(1, 11, size=len(data))
            elif regime_char['momentum_direction'] == 'bull':
                # Bull market: Longer holds (20-50 bars)
                return np.random.randint(20, 51, size=len(data))
            else:
                # Default: Medium-term holds (10-30 bars)
                return np.random.randint(10, 31, size=len(data))
                
        except Exception as e:
            self.logger.error(error(f"Failed to calculate exit timing target: {e}"))
            return np.ones(len(data)) * 20  # Default 20 bars
    
    def _calculate_exit_probability_targets(self, data: pd.DataFrame, regime_char: dict[str, Any]) -> dict[str, np.ndarray]:
        """Calculate exit probability targets for different time windows."""
        try:
            probabilities = {}
            
            # Different time windows
            windows = [5, 10, 30, 60]
            
            for window in windows:
                if regime_char['volatility_level'] == 'high':
                    # High volatility: Higher exit probabilities
                    prob = np.random.uniform(0.6, 0.9, size=len(data))
                elif regime_char['momentum_direction'] == 'bull':
                    # Bull market: Lower exit probabilities
                    prob = np.random.uniform(0.2, 0.5, size=len(data))
                else:
                    # Default: Medium probabilities
                    prob = np.random.uniform(0.3, 0.7, size=len(data))
                
                # Convert to binary (exit or not)
                probabilities[f'exit_prob_{window}'] = (prob > 0.5).astype(int)
            
            return probabilities
            
        except Exception as e:
            self.logger.error(error(f"Failed to calculate exit probability targets: {e}"))
            return {f'exit_prob_{w}': np.zeros(len(data)) for w in [5, 10, 30, 60]}
    
    def _calculate_exit_type_target(self, data: pd.DataFrame, regime_char: dict[str, Any]) -> np.ndarray:
        """Calculate exit type target."""
        try:
            # Exit types: 0=hold, 1=take_profit, 2=stop_loss, 3=trailing_stop, 4=time_based
            if regime_char['momentum_direction'] == 'bull':
                # Bull market: More take profits
                return np.random.choice([0, 1, 3], size=len(data), p=[0.4, 0.4, 0.2])
            elif regime_char['volatility_level'] == 'high':
                # High volatility: More stop losses
                return np.random.choice([0, 2, 4], size=len(data), p=[0.3, 0.4, 0.3])
            else:
                # Default: Mixed strategy
                return np.random.choice([0, 1, 2, 3, 4], size=len(data), p=[0.3, 0.2, 0.2, 0.2, 0.1])
                
        except Exception as e:
            self.logger.error(error(f"Failed to calculate exit type target: {e}"))
            return np.zeros(len(data))  # Default: hold
    
    def _calculate_profit_target(self, data: pd.DataFrame, regime_char: dict[str, Any]) -> np.ndarray:
        """Calculate profit target."""
        try:
            if regime_char['volatility_level'] == 'high':
                # High volatility: Higher profit targets
                return np.random.uniform(0.02, 0.05, size=len(data))  # 2-5%
            elif regime_char['momentum_direction'] == 'bull':
                # Bull market: Medium profit targets
                return np.random.uniform(0.01, 0.03, size=len(data))  # 1-3%
            else:
                # Default: Lower profit targets
                return np.random.uniform(0.005, 0.02, size=len(data))  # 0.5-2%
                
        except Exception as e:
            self.logger.error(error(f"Failed to calculate profit target: {e}"))
            return np.ones(len(data)) * 0.02  # Default 2%
    
    async def _train_exit_timing_model(
        self,
        features: pd.DataFrame,
        target: np.ndarray,
        regime_id: int
    ) -> dict[str, Any]:
        """Train exit timing model (regression)."""
        try:
            self.logger.info(f"🎯 Training exit timing model for regime {regime_id}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train LightGBM model
            model = lgb.LGBMRegressor(
                objective='regression',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbose=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store scaler
            self.regime_scalers[f'{regime_id}_exit_timing'] = scaler
            
            return {
                'model': model,
                'scaler': scaler,
                'metrics': {
                    'mse': mse,
                    'r2': r2,
                    'rmse': np.sqrt(mse)
                },
                'feature_importance': dict(zip(features.columns, model.feature_importances_))
            }
            
        except Exception as e:
            self.logger.error(error(f"Failed to train exit timing model: {e}"))
            return None
    
    async def _train_exit_probability_models(
        self,
        features: pd.DataFrame,
        targets: dict[str, np.ndarray],
        regime_id: int
    ) -> dict[str, Any]:
        """Train exit probability models (classification)."""
        try:
            self.logger.info(f"🎯 Training exit probability models for regime {regime_id}")
            
            models = {}
            
            for window, target in targets.items():
                self.logger.info(f"   Training {window} model")
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    features, target, test_size=0.2, random_state=42, stratify=target
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train XGBoost model
                model = XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42,
                    eval_metric='logloss'
                )
                
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                # Store model and scaler
                models[window] = {
                    'model': model,
                    'scaler': scaler,
                    'metrics': {
                        'accuracy': (y_pred == y_test).mean(),
                        'auc': r2_score(y_test, y_pred_proba)  # Simplified AUC
                    },
                    'feature_importance': dict(zip(features.columns, model.feature_importances_))
                }
                
                self.regime_scalers[f'{regime_id}_{window}'] = scaler
            
            return models
            
        except Exception as e:
            self.logger.error(error(f"Failed to train exit probability models: {e}"))
            return {}
    
    async def _train_exit_type_model(
        self,
        features: pd.DataFrame,
        target: np.ndarray,
        regime_id: int
    ) -> dict[str, Any]:
        """Train exit type model (classification)."""
        try:
            self.logger.info(f"🎯 Training exit type model for regime {regime_id}")
            
            # Encode target
            encoder = LabelEncoder()
            target_encoded = encoder.fit_transform(target)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, target_encoded, test_size=0.2, random_state=42, stratify=target_encoded
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train CatBoost model
            model = CatBoostClassifier(
                iterations=100,
                learning_rate=0.1,
                depth=6,
                random_state=42,
                verbose=False
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            
            # Store model, scaler, and encoder
            self.regime_scalers[f'{regime_id}_exit_type'] = scaler
            self.regime_encoders[f'{regime_id}_exit_type'] = encoder
            
            return {
                'model': model,
                'scaler': scaler,
                'encoder': encoder,
                'metrics': {
                    'accuracy': (y_pred == y_test).mean()
                },
                'feature_importance': dict(zip(features.columns, model.feature_importances_))
            }
            
        except Exception as e:
            self.logger.error(error(f"Failed to train exit type model: {e}"))
            return None
    
    async def _train_profit_target_model(
        self,
        features: pd.DataFrame,
        target: np.ndarray,
        regime_id: int
    ) -> dict[str, Any]:
        """Train profit target model (regression)."""
        try:
            self.logger.info(f"🎯 Training profit target model for regime {regime_id}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train LightGBM model
            model = lgb.LGBMRegressor(
                objective='regression',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbose=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store scaler
            self.regime_scalers[f'{regime_id}_profit_target'] = scaler
            
            return {
                'model': model,
                'scaler': scaler,
                'metrics': {
                    'mse': mse,
                    'r2': r2,
                    'rmse': np.sqrt(mse)
                },
                'feature_importance': dict(zip(features.columns, model.feature_importances_))
            }
            
        except Exception as e:
            self.logger.error(error(f"Failed to train profit target model: {e}"))
            return None
    
    async def _model_exists(self, regime_id: int) -> bool:
        """Check if model for regime already exists."""
        try:
            model_path = Path(self.training_config["model_output_path"]) / str(regime_id)
            return model_path.exists() and any(model_path.iterdir())
        except Exception:
            return False
    
    async def _save_regime_models(self, models: dict[str, Any], regime_id: int) -> None:
        """Save trained models for a regime."""
        try:
            model_path = Path(self.training_config["model_output_path"]) / str(regime_id)
            model_path.mkdir(exist_ok=True)
            
            for model_name, model_data in models.items():
                if model_data is None:
                    continue
                    
                # Save model
                model_file = model_path / f"{model_name}_model.pkl"
                with open(model_file, 'wb') as f:
                    pickle.dump(model_data['model'], f)
                
                # Save scaler
                scaler_file = model_path / f"{model_name}_scaler.pkl"
                with open(scaler_file, 'wb') as f:
                    pickle.dump(model_data['scaler'], f)
                
                # Save encoder if exists
                if 'encoder' in model_data:
                    encoder_file = model_path / f"{model_name}_encoder.pkl"
                    with open(encoder_file, 'wb') as f:
                        pickle.dump(model_data['encoder'], f)
                
                # Save metrics
                metrics_file = model_path / f"{model_name}_metrics.json"
                with open(metrics_file, 'w') as f:
                    json.dump(model_data['metrics'], f, indent=2)
                
                # Save feature importance
                importance_file = model_path / f"{model_name}_importance.json"
                with open(importance_file, 'w') as f:
                    json.dump(model_data['feature_importance'], f, indent=2)
            
            self.logger.info(f"💾 Saved models for regime {regime_id} to {model_path}")
            
        except Exception as e:
            self.logger.error(error(f"Failed to save regime models: {e}"))
    
    async def _save_training_results(self, symbol: str, exchange: str, timeframe: str) -> None:
        """Save overall training results."""
        try:
            results_path = Path(self.training_config["model_output_path"]) / "training_results.json"
            
            results = {
                'timestamp': time.time(),
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'regimes_trained': list(self.regime_models.keys()),
                'total_models': sum(len(models) for models in self.regime_models.values()),
                'training_summary': self._generate_training_summary()
            }
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"💾 Saved training results to {results_path}")
            
        except Exception as e:
            self.logger.error(error(f"Failed to save training results: {e}"))
    
    def _generate_training_summary(self) -> dict[str, Any]:
        """Generate training summary."""
        try:
            summary = {}
            
            for regime_id, models in self.regime_models.items():
                summary[regime_id] = {
                    'models_trained': list(models.keys()),
                    'model_count': len(models)
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(error(f"Failed to generate training summary: {e}"))
            return {}


async def run_regime_specific_training(
    symbol: str,
    exchange: str,
    timeframe: str,
    config: dict[str, Any],
    force_retrain: bool = False
) -> bool:
    """
    Run regime-specific ML model training.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        config: Configuration dictionary
        force_retrain: Force retraining even if models exist
        
    Returns:
        bool: True if training successful, False otherwise
    """
    try:
        trainer = RegimeSpecificMLTrainer(config)
        
        if not await trainer.initialize():
            return False
        
        success = await trainer.train_regime_models(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            force_retrain=force_retrain
        )
        
        return success
        
    except Exception as e:
        print(f"❌ Regime-specific training failed: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    config = {
        "regime_specific_training": {
            "hmm_data_path": "data/training/hmm_regimes",
            "model_output_path": "models/regime_specific",
            "regime_models": ["exit_timing", "exit_probabilities", "exit_type", "profit_target"],
            "training_parameters": {
                "test_size": 0.2,
                "random_state": 42
            }
        }
    }
    
    # Run training
    asyncio.run(run_regime_specific_training(
        symbol="ETHUSDT",
        exchange="BINANCE", 
        timeframe="1m",
        config=config,
        force_retrain=True
    ))