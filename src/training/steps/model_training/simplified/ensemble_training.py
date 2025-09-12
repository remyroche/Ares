"""
Ensemble Training Pipeline

This module provides ensemble model training that combines multiple models for better predictions.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

from src.utils.logger import system_logger
from src.utils.data.real_data_loader import real_data_loader
from .general_model_training import GeneralModelTrainer, ModelTrainingConfig, ModelType, TaskType

logger = logging.getLogger(__name__)

class EnsembleTrainingPipeline:
    """Ensemble model training pipeline."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ensemble training pipeline.
        
        Args:
            config: Training configuration
        """
        self.config = config or {}
        self.logger = system_logger.getChild('EnsembleTrainingPipeline')
        self.general_trainer = GeneralModelTrainer()
        
    async def train_ensemble_models(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        force_rerun: bool = False
    ) -> Dict[str, Any]:
        """
        Train ensemble models that combine multiple algorithms.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            force_rerun: Force retraining
            
        Returns:
            Training results dictionary
        """
        try:
            self.logger.info(f"🎯 Starting ensemble training for {symbol}/{exchange}/{timeframe}")
            
            # Load real market data
            market_data = await real_data_loader.load_market_data(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                force_download=force_rerun
            )
            
            if market_data is None or len(market_data) == 0:
                raise RuntimeError("No market data available for ensemble training")
            
            # Process and validate data
            processed_data = real_data_loader.process_and_validate_data(
                market_data, symbol, exchange, timeframe
            )
            
            # Create ensemble features
            ensemble_features = self._create_ensemble_features(processed_data)
            
            # Train individual models
            individual_models = await self._train_individual_models(ensemble_features)
            
            # Train ensemble meta-model
            ensemble_model = await self._train_ensemble_meta_model(
                ensemble_features, individual_models
            )
            
            # Calculate ensemble weights
            ensemble_weights = await self._calculate_ensemble_weights(
                ensemble_features, individual_models
            )
            
            # Save models
            model_paths = await self._save_ensemble_models(
                individual_models, ensemble_model, ensemble_weights,
                symbol, exchange, timeframe, data_dir
            )
            
            # Calculate performance metrics
            metrics = await self._calculate_ensemble_metrics(
                ensemble_features, individual_models, ensemble_model
            )
            
            self.logger.info("✅ Ensemble training completed successfully")
            
            return {
                'models': model_paths,
                'metrics': metrics,
                'weights': ensemble_weights,
                'performance': {
                    'ensemble_accuracy': metrics.get('ensemble_accuracy', 0.0),
                    'individual_accuracies': metrics.get('individual_accuracies', {}),
                    'data_points': len(processed_data)
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Ensemble training failed: {e}")
            raise
    
    def _create_ensemble_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features suitable for ensemble training."""
        try:
            features = data.copy()
            
            # Add technical indicators
            features['returns'] = features['close'].pct_change()
            features['volatility'] = features['returns'].rolling(window=20).std()
            features['volume_ma'] = features['volume'].rolling(window=20).mean()
            features['price_ma'] = features['close'].rolling(window=20).mean()
            features['rsi'] = self._calculate_rsi(features['close'])
            features['macd'] = self._calculate_macd(features['close'])
            
            # Add price-based features
            features['high_low_ratio'] = features['high'] / features['low']
            features['close_open_ratio'] = features['close'] / features['open']
            features['volume_ratio'] = features['volume'] / features['volume_ma']
            
            # Add lagged features
            for lag in [1, 2, 3, 5, 10]:
                features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
                features[f'volume_lag_{lag}'] = features['volume'].shift(lag)
            
            # Remove NaN values
            features = features.dropna()
            
            self.logger.info(f"✅ Created ensemble features: {features.shape}")
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Error creating ensemble features: {e}")
            return data
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD indicator."""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            return macd
        except:
            return pd.Series([0] * len(prices), index=prices.index)
    
    async def _train_individual_models(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Train individual models for the ensemble."""
        try:
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import SVC
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            
            # Prepare features and targets
            feature_columns = [col for col in features.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            X = features[feature_columns].values
            y = (features['close'].shift(-1) > features['close']).astype(int).values[:-1]
            X = X[:-1]  # Remove last row to match y
            
            if len(X) < 10:  # Need minimum samples
                raise ValueError("Insufficient data for training")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            models = {}
            
            # Random Forest
            try:
                rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_model.fit(X_train, y_train)
                rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
                models['random_forest'] = {
                    'model': rf_model,
                    'accuracy': rf_accuracy
                }
                self.logger.info(f"✅ Random Forest: {rf_accuracy:.3f} accuracy")
            except Exception as e:
                self.logger.warning(f"⚠️ Random Forest training failed: {e}")
            
            # Gradient Boosting
            try:
                gb_model = GradientBoostingClassifier(random_state=42)
                gb_model.fit(X_train, y_train)
                gb_accuracy = accuracy_score(y_test, gb_model.predict(X_test))
                models['gradient_boosting'] = {
                    'model': gb_model,
                    'accuracy': gb_accuracy
                }
                self.logger.info(f"✅ Gradient Boosting: {gb_accuracy:.3f} accuracy")
            except Exception as e:
                self.logger.warning(f"⚠️ Gradient Boosting training failed: {e}")
            
            # Logistic Regression
            try:
                lr_model = LogisticRegression(random_state=42, max_iter=1000)
                lr_model.fit(X_train, y_train)
                lr_accuracy = accuracy_score(y_test, lr_model.predict(X_test))
                models['logistic_regression'] = {
                    'model': lr_model,
                    'accuracy': lr_accuracy
                }
                self.logger.info(f"✅ Logistic Regression: {lr_accuracy:.3f} accuracy")
            except Exception as e:
                self.logger.warning(f"⚠️ Logistic Regression training failed: {e}")
            
            return models
            
        except Exception as e:
            self.logger.error(f"❌ Error training individual models: {e}")
            raise
    
    async def _train_ensemble_meta_model(
        self, 
        features: pd.DataFrame, 
        individual_models: Dict[str, Any]
    ) -> Any:
        """Train ensemble meta-model."""
        try:
            from sklearn.ensemble import VotingClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            
            if not individual_models:
                raise ValueError("No individual models available for ensemble")
            
            # Prepare features and targets
            feature_columns = [col for col in features.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            X = features[feature_columns].values
            y = (features['close'].shift(-1) > features['close']).astype(int).values[:-1]
            X = X[:-1]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Create voting classifier
            estimators = [(name, model_data['model']) for name, model_data in individual_models.items()]
            ensemble_model = VotingClassifier(estimators=estimators, voting='soft')
            ensemble_model.fit(X_train, y_train)
            
            # Calculate accuracy
            ensemble_accuracy = accuracy_score(y_test, ensemble_model.predict(X_test))
            
            self.logger.info(f"✅ Ensemble meta-model: {ensemble_accuracy:.3f} accuracy")
            
            return ensemble_model
            
        except Exception as e:
            self.logger.error(f"❌ Error training ensemble meta-model: {e}")
            raise
    
    async def _calculate_ensemble_weights(
        self, 
        features: pd.DataFrame, 
        individual_models: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate ensemble weights based on individual model performance."""
        try:
            weights = {}
            total_accuracy = sum(model_data['accuracy'] for model_data in individual_models.values())
            
            if total_accuracy > 0:
                for name, model_data in individual_models.items():
                    weights[name] = model_data['accuracy'] / total_accuracy
            else:
                # Equal weights if no accuracy data
                equal_weight = 1.0 / len(individual_models) if individual_models else 0.0
                weights = {name: equal_weight for name in individual_models.keys()}
            
            self.logger.info(f"✅ Calculated ensemble weights: {weights}")
            return weights
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating ensemble weights: {e}")
            return {}
    
    async def _save_ensemble_models(
        self,
        individual_models: Dict[str, Any],
        ensemble_model: Any,
        ensemble_weights: Dict[str, float],
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str
    ) -> List[str]:
        """Save trained ensemble models."""
        try:
            import pickle
            from pathlib import Path
            
            models_dir = Path(data_dir) / 'models' / 'ensemble'
            models_dir.mkdir(parents=True, exist_ok=True)
            
            model_paths = []
            
            # Save individual models
            for name, model_data in individual_models.items():
                model_path = models_dir / f'ensemble_{name}_{symbol}_{exchange}_{timeframe}.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump(model_data['model'], f)
                model_paths.append(str(model_path))
            
            # Save ensemble meta-model
            ensemble_path = models_dir / f'ensemble_meta_{symbol}_{exchange}_{timeframe}.pkl'
            with open(ensemble_path, 'wb') as f:
                pickle.dump(ensemble_model, f)
            model_paths.append(str(ensemble_path))
            
            # Save ensemble weights
            weights_path = models_dir / f'ensemble_weights_{symbol}_{exchange}_{timeframe}.pkl'
            with open(weights_path, 'wb') as f:
                pickle.dump(ensemble_weights, f)
            model_paths.append(str(weights_path))
            
            self.logger.info(f"✅ Saved {len(model_paths)} ensemble models")
            return model_paths
            
        except Exception as e:
            self.logger.error(f"❌ Error saving ensemble models: {e}")
            raise
    
    async def _calculate_ensemble_metrics(
        self,
        features: pd.DataFrame,
        individual_models: Dict[str, Any],
        ensemble_model: Any
    ) -> Dict[str, Any]:
        """Calculate ensemble training metrics."""
        try:
            metrics = {
                'ensemble_accuracy': 0.0,
                'individual_accuracies': {},
                'n_models': len(individual_models),
                'total_samples': len(features)
            }
            
            # Individual model accuracies
            for name, model_data in individual_models.items():
                metrics['individual_accuracies'][name] = model_data['accuracy']
            
            # Calculate average individual accuracy
            if individual_models:
                avg_individual_accuracy = np.mean([model_data['accuracy'] for model_data in individual_models.values()])
                metrics['average_individual_accuracy'] = avg_individual_accuracy
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating ensemble metrics: {e}")
            return {'error': str(e)}

# Global instance for convenience
ensemble_training_pipeline = EnsembleTrainingPipeline()