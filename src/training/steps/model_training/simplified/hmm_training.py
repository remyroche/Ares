"""
HMM Training Pipeline

This module provides HMM-based model training with regime detection and optimization.
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

class HMMTrainingPipeline:
    """HMM-based model training pipeline."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize HMM training pipeline.
        
        Args:
            config: Training configuration
        """
        self.config = config or {}
        self.logger = system_logger.getChild('HMMTrainingPipeline')
        self.general_trainer = GeneralModelTrainer()
        
    async def train_hmm_models(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        force_rerun: bool = False
    ) -> Dict[str, Any]:
        """
        Train HMM-based models for regime detection and prediction.
        
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
            self.logger.info(f"🔄 Starting HMM training for {symbol}/{exchange}/{timeframe}")
            
            # Load real market data
            market_data = await real_data_loader.load_market_data(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                force_download=force_rerun
            )
            
            if market_data is None or len(market_data) == 0:
                raise RuntimeError("No market data available for HMM training")
            
            # Process and validate data
            processed_data = real_data_loader.process_and_validate_data(
                market_data, symbol, exchange, timeframe
            )
            
            # Create HMM-specific features
            hmm_features = self._create_hmm_features(processed_data)
            
            # Train regime detection models
            regime_models = await self._train_regime_models(hmm_features)
            
            # Train prediction models for each regime
            prediction_models = await self._train_regime_prediction_models(
                hmm_features, regime_models
            )
            
            # Save models
            model_paths = await self._save_hmm_models(
                regime_models, prediction_models, symbol, exchange, timeframe, data_dir
            )
            
            # Calculate performance metrics
            metrics = await self._calculate_hmm_metrics(
                hmm_features, regime_models, prediction_models
            )
            
            self.logger.info("✅ HMM training completed successfully")
            
            return {
                'models': model_paths,
                'metrics': metrics,
                'performance': {
                    'regime_accuracy': metrics.get('regime_accuracy', 0.0),
                    'prediction_accuracy': metrics.get('prediction_accuracy', 0.0),
                    'data_points': len(processed_data)
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ HMM training failed: {e}")
            raise
    
    def _create_hmm_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features suitable for HMM training."""
        try:
            features = data.copy()
            
            # Add technical indicators
            features['returns'] = features['close'].pct_change()
            features['volatility'] = features['returns'].rolling(window=20).std()
            features['volume_ma'] = features['volume'].rolling(window=20).mean()
            features['price_ma'] = features['close'].rolling(window=20).mean()
            
            # Add regime indicators
            features['high_volatility'] = (features['volatility'] > features['volatility'].quantile(0.8)).astype(int)
            features['high_volume'] = (features['volume'] > features['volume_ma'] * 1.5).astype(int)
            features['trend_up'] = (features['close'] > features['price_ma']).astype(int)
            
            # Remove NaN values
            features = features.dropna()
            
            self.logger.info(f"✅ Created HMM features: {features.shape}")
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Error creating HMM features: {e}")
            return data
    
    async def _train_regime_models(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Train regime detection models."""
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # Prepare features for regime detection
            regime_features = features[['returns', 'volatility', 'high_volatility', 'high_volume', 'trend_up']].values
            
            # Standardize features
            scaler = StandardScaler()
            regime_features_scaled = scaler.fit_transform(regime_features)
            
            # Train K-means for regime detection
            n_regimes = 3  # Bull, Bear, Sideways
            kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
            regime_labels = kmeans.fit_predict(regime_features_scaled)
            
            # Add regime labels to features
            features['regime'] = regime_labels
            
            self.logger.info(f"✅ Trained regime detection: {n_regimes} regimes identified")
            
            return {
                'kmeans_model': kmeans,
                'scaler': scaler,
                'regime_labels': regime_labels,
                'n_regimes': n_regimes
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error training regime models: {e}")
            raise
    
    async def _train_regime_prediction_models(
        self, 
        features: pd.DataFrame, 
        regime_models: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train prediction models for each regime."""
        try:
            prediction_models = {}
            
            for regime in range(regime_models['n_regimes']):
                regime_data = features[features['regime'] == regime]
                
                if len(regime_data) < 10:  # Need minimum data points
                    continue
                
                # Prepare features and targets
                X = regime_data[['returns', 'volatility', 'high_volatility', 'high_volume', 'trend_up']].values
                y = (regime_data['close'].shift(-1) > regime_data['close']).astype(int).values[:-1]
                X = X[:-1]  # Remove last row to match y
                
                if len(X) < 5:  # Need minimum samples
                    continue
                
                # Train Random Forest for this regime
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.model_selection import train_test_split
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                rf_model = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=10,
                    random_state=42
                )
                rf_model.fit(X_train, y_train)
                
                # Calculate accuracy
                accuracy = rf_model.score(X_test, y_test)
                
                prediction_models[f'regime_{regime}'] = {
                    'model': rf_model,
                    'accuracy': accuracy,
                    'n_samples': len(X)
                }
                
                self.logger.info(f"✅ Trained prediction model for regime {regime}: {accuracy:.3f} accuracy")
            
            return prediction_models
            
        except Exception as e:
            self.logger.error(f"❌ Error training prediction models: {e}")
            raise
    
    async def _save_hmm_models(
        self,
        regime_models: Dict[str, Any],
        prediction_models: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str
    ) -> List[str]:
        """Save trained HMM models."""
        try:
            import pickle
            from pathlib import Path
            
            models_dir = Path(data_dir) / 'models' / 'hmm'
            models_dir.mkdir(parents=True, exist_ok=True)
            
            model_paths = []
            
            # Save regime detection model
            regime_path = models_dir / f'hmm_regime_{symbol}_{exchange}_{timeframe}.pkl'
            with open(regime_path, 'wb') as f:
                pickle.dump(regime_models, f)
            model_paths.append(str(regime_path))
            
            # Save prediction models
            for regime_name, model_data in prediction_models.items():
                pred_path = models_dir / f'hmm_prediction_{regime_name}_{symbol}_{exchange}_{timeframe}.pkl'
                with open(pred_path, 'wb') as f:
                    pickle.dump(model_data, f)
                model_paths.append(str(pred_path))
            
            self.logger.info(f"✅ Saved {len(model_paths)} HMM models")
            return model_paths
            
        except Exception as e:
            self.logger.error(f"❌ Error saving HMM models: {e}")
            raise
    
    async def _calculate_hmm_metrics(
        self,
        features: pd.DataFrame,
        regime_models: Dict[str, Any],
        prediction_models: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate HMM training metrics."""
        try:
            metrics = {
                'regime_accuracy': 0.0,
                'prediction_accuracy': 0.0,
                'n_regimes': regime_models.get('n_regimes', 0),
                'n_prediction_models': len(prediction_models),
                'total_samples': len(features)
            }
            
            # Calculate average prediction accuracy
            if prediction_models:
                accuracies = [model_data['accuracy'] for model_data in prediction_models.values()]
                metrics['prediction_accuracy'] = np.mean(accuracies)
            
            # Regime distribution
            regime_counts = features['regime'].value_counts()
            metrics['regime_distribution'] = regime_counts.to_dict()
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating HMM metrics: {e}")
            return {'error': str(e)}

# Global instance for convenience
hmm_training_pipeline = HMMTrainingPipeline()