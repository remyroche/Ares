"""
HMM Training Pipeline

This module provides HMM-based model training that consumes regime discovery results
from market_analysis/ and creates a single model that predicts regime membership probabilities.
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

logger = logging.getLogger(__name__)

class HMMTrainingPipeline:
    """HMM-based model training pipeline that uses market analysis results."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize HMM training pipeline.
        
        Args:
            config: Training configuration
        """
        self.config = config or {}
        self.logger = system_logger.getChild('HMMTrainingPipeline')
        
    async def train_hmm_models(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        pipeline_state: Dict[str, Any],
        force_rerun: bool = False
    ) -> Dict[str, Any]:
        """
        Train HMM-based models using regime discovery results from market analysis.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            pipeline_state: Pipeline state containing regime discovery results
            force_rerun: Force retraining
            
        Returns:
            Training results dictionary with regime probabilities and confidence scores
        """
        try:
            self.logger.info(f"🔄 Starting HMM training for {symbol}/{exchange}/{timeframe}")
            
            # Check if regime discovery results are available
            if not self._validate_regime_discovery_results(pipeline_state):
                raise RuntimeError("Missing regime discovery results from market analysis")
            
            # Extract regime discovery results from pipeline state
            regime_data = self._extract_regime_data(pipeline_state)
            
            # Load market data for training
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
            
            # Train single HMM model that predicts regime membership probabilities
            hmm_model_results = await self._train_regime_probability_model(
                processed_data, regime_data
            )
            
            # Calculate regime confidence scores
            confidence_scores = await self._calculate_regime_confidence(
                processed_data, hmm_model_results
            )
            
            # Save models and results
            model_paths = await self._save_hmm_models(
                hmm_model_results, symbol, exchange, timeframe, data_dir
            )
            
            # Calculate performance metrics
            metrics = await self._calculate_hmm_metrics(
                processed_data, hmm_model_results, confidence_scores
            )
            
            self.logger.info("✅ HMM training completed successfully")
            
            return {
                'models': model_paths,
                'metrics': metrics,
                'regime_probabilities': hmm_model_results['regime_probabilities'],
                'regime_confidence': confidence_scores,
                'hmm_state_probs': hmm_model_results['hmm_state_probs'],
                'performance': {
                    'regime_accuracy': metrics.get('regime_accuracy', 0.0),
                    'prediction_accuracy': metrics.get('prediction_accuracy', 0.0),
                    'data_points': len(processed_data),
                    'n_regimes': hmm_model_results['n_regimes']
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ HMM training failed: {e}")
            raise
    
    def _validate_regime_discovery_results(self, pipeline_state: Dict[str, Any]) -> bool:
        """Validate that regime discovery results are available."""
        required_keys = [
            'hmm_regime_discovery_completed',
            'step03_hmm_regime_discovery_completed',
            'regime_states',
            'regime_probabilities'
        ]
        
        for key in required_keys:
            if key not in pipeline_state:
                self.logger.error(f"Missing required regime discovery result: {key}")
                return False
        
        return pipeline_state.get('hmm_regime_discovery_completed', False)
    
    def _extract_regime_data(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract regime discovery data from pipeline state."""
        return {
            'regime_states': pipeline_state.get('regime_states', []),
            'regime_probabilities': pipeline_state.get('regime_probabilities', []),
            'regime_confidence': pipeline_state.get('regime_confidence', []),
            'hmm_state_sequence': pipeline_state.get('hmm_state_sequence', []),
            'hmm_state_probs': pipeline_state.get('hmm_state_probs', []),
            'n_regimes': len(pipeline_state.get('regime_states', [])),
            'regime_characteristics': pipeline_state.get('regime_characteristics', {}),
            'transition_matrix': pipeline_state.get('transition_matrix', None)
        }
    
    async def _train_regime_probability_model(
        self, 
        market_data: pd.DataFrame, 
        regime_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train a single HMM model that predicts regime membership probabilities."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import accuracy_score, classification_report
            
            # Prepare features from market data
            features = self._prepare_training_features(market_data)
            
            # Get regime labels and probabilities from market analysis
            regime_labels = np.array(regime_data['regime_states'])
            regime_probabilities = np.array(regime_data['regime_probabilities'])
            
            if len(regime_labels) != len(features):
                self.logger.warning(f"Length mismatch: features={len(features)}, regimes={len(regime_labels)}")
                min_len = min(len(features), len(regime_labels))
                features = features.iloc[:min_len]
                regime_labels = regime_labels[:min_len]
                regime_probabilities = regime_probabilities[:min_len]
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Split data for training and validation
            X_train, X_val, y_train, y_val = train_test_split(
                features_scaled, regime_labels, test_size=0.2, random_state=42, stratify=regime_labels
            )
            
            # Train Random Forest for regime classification
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            
            # Get predictions and probabilities
            y_pred = rf_model.predict(X_val)
            y_pred_proba = rf_model.predict_proba(X_val)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_val, y_pred)
            
            # Get full dataset predictions and probabilities
            full_predictions = rf_model.predict(features_scaled)
            full_probabilities = rf_model.predict_proba(features_scaled)
            
            self.logger.info(f"✅ Trained regime probability model: {accuracy:.3f} accuracy")
            self.logger.info(f"   - Number of regimes: {rf_model.n_classes_}")
            self.logger.info(f"   - Feature importance shape: {rf_model.feature_importances_.shape}")
            
            return {
                'model': rf_model,
                'scaler': scaler,
                'regime_predictions': full_predictions,
                'regime_probabilities': full_probabilities,
                'hmm_state_probs': full_probabilities,  # Alias for compatibility
                'n_regimes': rf_model.n_classes_,
                'accuracy': accuracy,
                'feature_names': features.columns.tolist(),
                'regime_labels': regime_labels
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error training regime probability model: {e}")
            raise
    
    def _prepare_training_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for HMM training from market data."""
        try:
            features = market_data.copy()
            
            # Basic price features
            features['returns'] = features['close'].pct_change()
            features['log_returns'] = np.log(features['close'] / features['close'].shift(1))
            
            # Volatility features
            features['volatility_20'] = features['returns'].rolling(window=20).std()
            features['volatility_5'] = features['returns'].rolling(window=5).std()
            features['volatility_ratio'] = features['volatility_5'] / features['volatility_20']
            
            # Volume features
            features['volume_ma_20'] = features['volume'].rolling(window=20).mean()
            features['volume_ratio'] = features['volume'] / features['volume_ma_20']
            features['volume_price_trend'] = features['volume'] * features['returns']
            
            # Price momentum features
            features['price_ma_20'] = features['close'].rolling(window=20).mean()
            features['price_ma_5'] = features['close'].rolling(window=5).mean()
            features['momentum_20'] = features['close'] / features['close'].shift(20) - 1
            features['momentum_5'] = features['close'] / features['close'].shift(5) - 1
            
            # Technical indicators
            features['rsi_14'] = self._calculate_rsi(features['close'], 14)
            features['macd'] = self._calculate_macd(features['close'])
            features['bollinger_position'] = self._calculate_bollinger_position(features['close'])
            
            # High-frequency features
            features['high_low_ratio'] = features['high'] / features['low']
            features['close_position'] = (features['close'] - features['low']) / (features['high'] - features['low'])
            
            # Remove NaN values
            features = features.dropna()
            
            # Select only numeric columns for training
            numeric_columns = features.select_dtypes(include=[np.number]).columns
            features = features[numeric_columns]
            
            self.logger.info(f"✅ Prepared HMM training features: {features.shape}")
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Error preparing training features: {e}")
            return market_data
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        return ema_fast - ema_slow
    
    def _calculate_bollinger_position(self, prices: pd.Series, window: int = 20, std_dev: int = 2) -> pd.Series:
        """Calculate Bollinger Bands position."""
        ma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = ma + (std * std_dev)
        lower = ma - (std * std_dev)
        return (prices - lower) / (upper - lower)
    
    async def _calculate_regime_confidence(
        self, 
        market_data: pd.DataFrame, 
        hmm_model_results: Dict[str, Any]
    ) -> np.ndarray:
        """Calculate confidence scores for regime assignments."""
        try:
            regime_probabilities = hmm_model_results['regime_probabilities']
            
            # Calculate confidence as the maximum probability for each time point
            confidence_scores = np.max(regime_probabilities, axis=1)
            
            # Calculate entropy-based confidence (lower entropy = higher confidence)
            entropy = -np.sum(regime_probabilities * np.log(regime_probabilities + 1e-10), axis=1)
            max_entropy = np.log(regime_probabilities.shape[1])
            entropy_confidence = 1 - (entropy / max_entropy)
            
            # Combine both confidence measures
            combined_confidence = (confidence_scores + entropy_confidence) / 2
            
            self.logger.info(f"✅ Calculated regime confidence scores: mean={np.mean(combined_confidence):.3f}")
            return combined_confidence
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating regime confidence: {e}")
            return np.ones(len(market_data)) * 0.5  # Default confidence
    
    async def _save_hmm_models(
        self,
        hmm_model_results: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str
    ) -> List[str]:
        """Save trained HMM models and results."""
        try:
            import pickle
            from pathlib import Path
            
            models_dir = Path(data_dir) / 'models' / 'hmm'
            models_dir.mkdir(parents=True, exist_ok=True)
            
            model_paths = []
            
            # Save the main HMM model
            model_path = models_dir / f'hmm_regime_model_{symbol}_{exchange}_{timeframe}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': hmm_model_results['model'],
                    'scaler': hmm_model_results['scaler'],
                    'feature_names': hmm_model_results['feature_names'],
                    'n_regimes': hmm_model_results['n_regimes']
                }, f)
            model_paths.append(str(model_path))
            
            # Save regime probabilities
            probs_path = models_dir / f'hmm_regime_probabilities_{symbol}_{exchange}_{timeframe}.pkl'
            with open(probs_path, 'wb') as f:
                pickle.dump({
                    'regime_probabilities': hmm_model_results['regime_probabilities'],
                    'regime_predictions': hmm_model_results['regime_predictions'],
                    'regime_labels': hmm_model_results['regime_labels']
                }, f)
            model_paths.append(str(probs_path))
            
            self.logger.info(f"✅ Saved {len(model_paths)} HMM model files")
            return model_paths
            
        except Exception as e:
            self.logger.error(f"❌ Error saving HMM models: {e}")
            raise
    
    async def _calculate_hmm_metrics(
        self,
        market_data: pd.DataFrame,
        hmm_model_results: Dict[str, Any],
        confidence_scores: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate HMM training metrics."""
        try:
            metrics = {
                'regime_accuracy': hmm_model_results.get('accuracy', 0.0),
                'prediction_accuracy': hmm_model_results.get('accuracy', 0.0),
                'n_regimes': hmm_model_results.get('n_regimes', 0),
                'total_samples': len(market_data),
                'mean_confidence': float(np.mean(confidence_scores)),
                'confidence_std': float(np.std(confidence_scores)),
                'min_confidence': float(np.min(confidence_scores)),
                'max_confidence': float(np.max(confidence_scores))
            }
            
            # Regime distribution
            regime_predictions = hmm_model_results.get('regime_predictions', [])
            if len(regime_predictions) > 0:
                unique_regimes, counts = np.unique(regime_predictions, return_counts=True)
                regime_distribution = {f'regime_{regime}': int(count) for regime, count in zip(unique_regimes, counts)}
                metrics['regime_distribution'] = regime_distribution
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating HMM metrics: {e}")
            return {'error': str(e)}

# Global instance for convenience
hmm_training_pipeline = HMMTrainingPipeline()