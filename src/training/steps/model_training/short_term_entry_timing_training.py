"""
Short-Term Entry Timing Training Step

This step handles training of the short-term entry timing model using the triple barrier method
for predicting optimal entry timing based on expected short-term price movements (0.1% to 0.5%).

Key Features:
- Triple barrier method integration
- Multi-output prediction for 5 different target percentages
- Direction-aware predictions (up/down movement)
- Adverse movement protection
- Integration with existing Tactician architecture
- Comprehensive evaluation metrics
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time
from datetime import datetime

from src.utils.logger import system_logger
from src.utils.ml_common.config import PerRegimeTrainingConfig
from src.utils.ml_common.training import PerRegimeTrainingStep
from src.tactician.short_term_entry_timing_model import (
    ShortTermEntryTimingModel, ShortTermEntryTimingConfig
)
from src.tactician.short_term_target_generator import (
    ShortTermTargetGenerator, TripleBarrierConfig
)

# Import vectorized training manager for enhanced capabilities
try:
    from src.utils.ml_common.training.vectorized_training_manager import VectorizedTrainingManager
    VECTORIZED_TRAINING_AVAILABLE = True
except ImportError:
    VECTORIZED_TRAINING_AVAILABLE = False

logger = system_logger.getChild('ShortTermEntryTimingTraining')


class ShortTermEntryTimingTrainingStep(PerRegimeTrainingStep):
    """
    Training step for short-term entry timing model with triple barrier method.
    
    This step trains models to predict optimal entry timing for 0.1% to 0.5% price movements
    using the triple barrier method to ensure expected price direction without adverse movement.
    """
    
    def __init__(self, config: Optional[PerRegimeTrainingConfig] = None, enable_vectorization: bool = True):
        """
        Initialize short-term entry timing training step.

        Args:
            config: Per-regime training configuration
            enable_vectorization: Whether to enable vectorized training
        """
        # Set default configuration for short-term entry timing models
        if config is None:
            config = PerRegimeTrainingConfig(
                model_name="short_term_entry_timing",
                timeframe="1m",
                model_types=["ShortTermEntryTimingModel"],
                hpo_n_trials=100,
                hpo_timeout_seconds=3600,
                min_samples_per_regime=1000,
                enable_data_augmentation=True,
                augmentation_method="smote",
                model_save_path="./models/short_term_entry_timing",
                evaluation_metrics=["hit_rate", "timing_accuracy", "profit_factor", "sharpe_ratio", "mse", "mae", "r2"]
            )

        super().__init__(config)
        self.logger = logger.getChild('ShortTermEntryTimingTrainingStep')

        # Vectorization support
        self.enable_vectorization = enable_vectorization and VECTORIZED_TRAINING_AVAILABLE
        if self.enable_vectorization:
            self.logger.info("🚀 Short-Term Entry Timing Training Step initialized with vectorization")
        else:
            self.logger.info("✅ Short-Term Entry Timing Training Step initialized (standard mode)")
        
        # Short-term specific configuration
        self.short_term_config = ShortTermEntryTimingConfig(
            model_name="short_term_entry_timing",
            timeframe="1m",
            target_percentages=[0.001, 0.002, 0.003, 0.004, 0.005],  # 0.1% to 0.5%
            triple_barrier=TripleBarrierConfig(
                target_percentages=[0.001, 0.002, 0.003, 0.004, 0.005],
                max_hold_time_minutes=15,
                max_adverse_movement=0.002,
                min_risk_reward_ratio=1.5
            )
        )
    
    def execute(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        hmm_states: Optional[np.ndarray] = None,
        analyst_signals: Optional[np.ndarray] = None,
        analyst_model_outputs: Optional[np.ndarray] = None,
        hmm_regime_features: Optional[np.ndarray] = None,
        all_analyst_models_outputs: Optional[Dict[str, np.ndarray]] = None,
        price_data: Optional[pd.DataFrame] = None,
        symbol: str = "UNKNOWN"
    ) -> Dict[str, Any]:
        """
        Execute short-term entry timing training step.
        
        Args:
            X: Input features (1m timeframe with cross-timeframe features)
            y: Target values (not used for short-term model - generated from price data)
            regime_labels: Regime labels for each sample
            feature_names: Names of input features
            hmm_states: HMM cluster/regime states
            analyst_signals: Binary signals from Analyst (green light indicators)
            analyst_model_outputs: Analyst model predictions used as features
            hmm_regime_features: HMM regime features (probabilities, characteristics)
            all_analyst_models_outputs: All individual analyst ML model outputs
            price_data: Price data for target generation (required for short-term model)
            symbol: Trading symbol
            
        Returns:
            Dictionary containing training results and metadata
        """
        self.logger.info("🚀 Starting short-term entry timing training step")
        
        # Validate required inputs
        if price_data is None:
            self.logger.error("❌ Price data is required for short-term entry timing training")
            return {'error': 'Price data is required for short-term entry timing training'}
        
        if len(price_data) < 100:
            self.logger.error("❌ Insufficient price data for training")
            return {'error': 'Insufficient price data for training'}
        
        # Filter data to only include periods where Analyst gives green light
        if analyst_signals is not None:
            green_light_mask = analyst_signals == 1
            self.logger.info(f"📊 Filtering to {np.sum(green_light_mask)} samples with Analyst green light signals")
            
            X = X[green_light_mask]
            regime_labels = regime_labels[green_light_mask]
            if hmm_states is not None:
                hmm_states = hmm_states[green_light_mask]
        
        # Combine all features: base features + HMM regime features + all analyst model outputs
        additional_features = []
        additional_feature_names = []
        
        # Add HMM regime features if provided
        if hmm_regime_features is not None:
            if analyst_signals is not None:
                hmm_regime_features = hmm_regime_features[green_light_mask]
            additional_features.append(hmm_regime_features)
            additional_feature_names.extend([f"hmm_regime_{i}" for i in range(hmm_regime_features.shape[1])])
            self.logger.info(f"📊 Added {hmm_regime_features.shape[1]} HMM regime features")
        
        # Add all individual analyst model outputs if provided
        if all_analyst_models_outputs is not None:
            for model_name, model_outputs in all_analyst_models_outputs.items():
                if analyst_signals is not None:
                    model_outputs = model_outputs[green_light_mask]
                additional_features.append(model_outputs)
                additional_feature_names.extend([f"analyst_{model_name}_{i}" for i in range(model_outputs.shape[1])])
            self.logger.info(f"📊 Added outputs from {len(all_analyst_models_outputs)} analyst models")
        
        # Add legacy analyst model outputs for backward compatibility
        if analyst_model_outputs is not None:
            if analyst_signals is not None:
                analyst_model_outputs = analyst_model_outputs[green_light_mask]
            additional_features.append(analyst_model_outputs)
            additional_feature_names.extend([f"analyst_legacy_{i}" for i in range(analyst_model_outputs.shape[1])])
            self.logger.info(f"📊 Added {analyst_model_outputs.shape[1]} legacy analyst outputs")
        
        # Concatenate all additional features
        if additional_features:
            X = np.column_stack([X] + additional_features)
            
            # Update feature names
            if feature_names is not None:
                feature_names = feature_names + additional_feature_names
            else:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
            self.logger.info(f"📊 Total features: {X.shape[1]} (base + HMM + all analyst models)")
        
        # Train short-term entry timing models
        try:
            results = self._train_short_term_models(
                X, regime_labels, feature_names, hmm_states, price_data, symbol
            )
            
            if 'error' not in results:
                self.logger.info("✅ Short-term entry timing training completed successfully")
            else:
                self.logger.error(f"❌ Short-term entry timing training failed: {results['error']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Short-term entry timing training failed: {e}")
            return {'error': str(e)}
    
    def _train_short_term_models(
        self,
        X: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]],
        hmm_states: Optional[np.ndarray],
        price_data: pd.DataFrame,
        symbol: str
    ) -> Dict[str, Any]:
        """Train short-term entry timing models."""
        
        self.logger.info("🔄 Training short-term entry timing models")
        start_time = time.time()
        
        try:
            # Get unique regimes
            unique_regimes = np.unique(regime_labels)
            self.logger.info(f"📊 Training models for {len(unique_regimes)} regimes: {unique_regimes}")
            
            # Initialize results
            results = {
                'model_name': 'short_term_entry_timing',
                'timeframe': '1m',
                'n_regimes': len(unique_regimes),
                'regime_analysis': {},
                'evaluation_results': {},
                'training_time': 0.0,
                'models_trained': 0,
                'successful_regimes': 0,
                'failed_regimes': 0
            }
            
            # Train models for each regime
            for regime in unique_regimes:
                regime_mask = regime_labels == regime
                regime_X = X[regime_mask]
                regime_price_data = price_data.iloc[regime_mask]
                
                self.logger.info(f"🔄 Training model for regime {regime} ({np.sum(regime_mask)} samples)")
                
                try:
                    # Create and train model for this regime
                    model = ShortTermEntryTimingModel(self.short_term_config)
                    
                    # Train the model
                    success = model.fit(regime_X, regime_price_data, symbol, "1m")
                    
                    if success:
                        # Evaluate the model
                        evaluation = self._evaluate_short_term_model(model, regime_X, regime_price_data, symbol)
                        
                        # Store results
                        results['regime_analysis'][f'regime_{regime}'] = {
                            'n_samples': np.sum(regime_mask),
                            'training_successful': True,
                            'model_type': 'ShortTermEntryTimingModel',
                            'evaluation': evaluation
                        }
                        
                        results['evaluation_results'][f'regime_{regime}'] = evaluation
                        results['successful_regimes'] += 1
                        results['models_trained'] += 1
                        
                        self.logger.info(f"✅ Regime {regime} model trained successfully")
                        
                    else:
                        results['regime_analysis'][f'regime_{regime}'] = {
                            'n_samples': np.sum(regime_mask),
                            'training_successful': False,
                            'error': 'Model training failed'
                        }
                        results['failed_regimes'] += 1
                        self.logger.warning(f"⚠️ Regime {regime} model training failed")
                        
                except Exception as e:
                    self.logger.error(f"❌ Error training regime {regime} model: {e}")
                    results['regime_analysis'][f'regime_{regime}'] = {
                        'n_samples': np.sum(regime_mask),
                        'training_successful': False,
                        'error': str(e)
                    }
                    results['failed_regimes'] += 1
            
            # Calculate training time
            results['training_time'] = time.time() - start_time
            
            # Add short-term specific metadata
            results = self._add_short_term_metadata(results, symbol, price_data)
            
            self.logger.info(f"✅ Short-term entry timing training completed in {results['training_time']:.3f}s")
            self.logger.info(f"📊 Successful regimes: {results['successful_regimes']}/{len(unique_regimes)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error in short-term model training: {e}")
            return {'error': str(e)}
    
    def _evaluate_short_term_model(
        self,
        model: ShortTermEntryTimingModel,
        X: np.ndarray,
        price_data: pd.DataFrame,
        symbol: str
    ) -> Dict[str, Any]:
        """Evaluate short-term entry timing model."""
        
        try:
            # Make predictions
            predictions = model.predict(X, price_data, symbol, "1m")
            
            if predictions is None:
                return {'error': 'Failed to make predictions'}
            
            # Calculate evaluation metrics
            evaluation = {
                'hit_rate': 0.0,
                'timing_accuracy': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'overall_confidence': predictions.overall_confidence,
                'risk_score': predictions.risk_score,
                'entry_recommendation': predictions.entry_recommendation,
                'valid_predictions': predictions.valid_predictions,
                'total_predictions': predictions.n_predictions,
                'model_confidence': predictions.model_confidence
            }
            
            # Calculate hit rate (percentage of valid predictions)
            if predictions.n_predictions > 0:
                evaluation['hit_rate'] = predictions.valid_predictions / predictions.n_predictions
            
            # Calculate timing accuracy (average timing prediction accuracy)
            if predictions.predictions:
                timings = [p.timing_minutes for p in predictions.predictions if p.is_valid]
                if timings:
                    evaluation['timing_accuracy'] = 1.0 / (1.0 + np.std(timings))  # Lower std = higher accuracy
            
            # Calculate profit factor (simplified)
            if predictions.predictions:
                probabilities = [p.probability for p in predictions.predictions if p.is_valid]
                risk_rewards = [p.risk_reward_ratio for p in predictions.predictions if p.is_valid]
                
                if probabilities and risk_rewards:
                    avg_prob = np.mean(probabilities)
                    avg_risk_reward = np.mean(risk_rewards)
                    evaluation['profit_factor'] = avg_prob * avg_risk_reward
            
            # Calculate Sharpe ratio (simplified)
            if predictions.overall_confidence > 0 and predictions.risk_score > 0:
                evaluation['sharpe_ratio'] = predictions.overall_confidence / predictions.risk_score
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"❌ Error evaluating short-term model: {e}")
            return {'error': str(e)}
    
    def _add_short_term_metadata(self, results: Dict[str, Any], symbol: str, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Add short-term specific metadata to results."""
        
        try:
            # Add short-term specific analysis
            short_term_analysis = {
                'target_percentages': self.short_term_config.target_percentages,
                'target_names': [f"{p*100:.1f}pct" for p in self.short_term_config.target_percentages],
                'triple_barrier_config': {
                    'max_hold_time_minutes': self.short_term_config.triple_barrier.max_hold_time_minutes,
                    'max_adverse_movement': self.short_term_config.triple_barrier.max_adverse_movement,
                    'min_risk_reward_ratio': self.short_term_config.triple_barrier.min_risk_reward_ratio
                },
                'timeframe': self.config.timeframe,
                'model_types': ['ShortTermEntryTimingModel'],
                'symbol': symbol,
                'price_data_samples': len(price_data),
                'feature_engineering': 'ultra_short_term_optimized'
            }
            
            results['short_term_analysis'] = short_term_analysis
            
            # Add model performance summary
            if 'evaluation_results' in results:
                evaluation_results = results['evaluation_results']
                
                # Calculate best performing regime
                best_regime = None
                best_hit_rate = -1
                
                for regime, metrics in evaluation_results.items():
                    if isinstance(metrics, dict) and 'error' not in metrics:
                        hit_rate = metrics.get('hit_rate', 0)
                        if hit_rate > best_hit_rate:
                            best_hit_rate = hit_rate
                            best_regime = regime
                
                if best_regime:
                    results['best_regime'] = {
                        'regime': best_regime,
                        'hit_rate': best_hit_rate,
                        'overall_confidence': evaluation_results[best_regime].get('overall_confidence', 0),
                        'risk_score': evaluation_results[best_regime].get('risk_score', 0)
                    }
            
            # Add timing-specific analysis
            timing_analysis = {
                'base_timeframe': self.config.timeframe,
                'ultra_short_term_features': True,
                'triple_barrier_method': True,
                'direction_aware_predictions': True,
                'adverse_movement_protection': True,
                'analyst_dependency': True,
                'multi_output_architecture': True
            }
            results['timing_analysis'] = timing_analysis
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error adding short-term metadata: {e}")
            return results


# Convenience functions for backward compatibility
def create_short_term_entry_timing_training_step(
    config: Optional[PerRegimeTrainingConfig] = None
) -> ShortTermEntryTimingTrainingStep:
    """Create short-term entry timing training step."""
    return ShortTermEntryTimingTrainingStep(config)


def execute_short_term_entry_timing_training(
    X: np.ndarray,
    y: np.ndarray,
    regime_labels: np.ndarray,
    price_data: pd.DataFrame,
    config: Optional[PerRegimeTrainingConfig] = None,
    feature_names: Optional[List[str]] = None,
    hmm_states: Optional[np.ndarray] = None,
    analyst_signals: Optional[np.ndarray] = None,
    analyst_model_outputs: Optional[np.ndarray] = None,
    hmm_regime_features: Optional[np.ndarray] = None,
    all_analyst_models_outputs: Optional[Dict[str, np.ndarray]] = None,
    symbol: str = "UNKNOWN"
) -> Dict[str, Any]:
    """Execute short-term entry timing training step."""
    step = create_short_term_entry_timing_training_step(config)
    return step.execute(
        X, y, regime_labels, feature_names, hmm_states, analyst_signals,
        analyst_model_outputs, hmm_regime_features, all_analyst_models_outputs,
        price_data, symbol
    )


# Example usage
if __name__ == "__main__":
    # Example of how to use the short-term entry timing training step
    print("Short-Term Entry Timing Training Step")
    print("=" * 45)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    # Generate sample features
    X = np.random.randn(n_samples, n_features)
    
    # Generate sample regime labels
    regime_labels = np.random.randint(0, 3, n_samples)
    
    # Generate sample analyst signals
    analyst_signals = np.random.randint(0, 2, n_samples)
    
    # Generate sample price data
    base_price = 100.0
    price_changes = np.random.normal(0, 0.001, n_samples)
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, 0.0005)))
        low = price * (1 - abs(np.random.normal(0, 0.0005)))
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'open': price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    price_data = pd.DataFrame(data)
    
    # Create training step
    training_step = create_short_term_entry_timing_training_step()
    
    print(f"✅ Created training step for short-term entry timing")
    print(f"📊 Target percentages: {[f'{p*100:.1f}%' for p in training_step.short_term_config.target_percentages]}")
    print(f"⏰ Timeframe: {training_step.config.timeframe}")
    print(f"🎯 Triple barrier method: {training_step.short_term_config.triple_barrier.max_hold_time_minutes} min max hold")
    
    # Execute training
    results = training_step.execute(
        X, np.zeros(n_samples), regime_labels, None, None, analyst_signals,
        None, None, None, price_data, "BTCUSDT"
    )
    
    if 'error' not in results:
        print(f"✅ Training completed successfully in {results['training_time']:.3f}s")
        print(f"📊 Successful regimes: {results['successful_regimes']}/{results['n_regimes']}")
        print(f"🎯 Models trained: {results['models_trained']}")
        
        if 'best_regime' in results:
            best = results['best_regime']
            print(f"🏆 Best regime: {best['regime']} - Hit rate: {best['hit_rate']:.3f}")
        
        print("\n📋 Regime Analysis:")
        for regime, analysis in results['regime_analysis'].items():
            status = "✅" if analysis['training_successful'] else "❌"
            print(f"{status} {regime}: {analysis['n_samples']} samples")
    else:
        print(f"❌ Training failed: {results['error']}")