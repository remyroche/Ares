"""
Regime HPO Integration for Hybrid NAS-TAS Regime Training

This module integrates hyperparameter optimization with the regime training pipeline,
providing automated optimization of regime detection and prediction models.

Key Features:
- Integration with existing regime training pipeline
- Hierarchical optimization (base models → meta model → meta features)
- Support for all regime model types (CatBoost, ExtraTrees, LightGBM, Bayesian Rules)
- OOF validation and time-series CV
- Meta-feature optimization for enhanced regime detection
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time
from datetime import datetime
from pathlib import Path
import yaml
import json

# Import regime HPO wrapper
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))
from src.utils.ml_common.optimization.regime_hpo_wrapper import (
    RegimeHPOWrapper,
    RegimeHPOConfig,
    RegimeHPOResult,
    optimize_regime_models,
    create_regime_hpo_config
)

# Import existing regime training components
from .regime_training_pipeline import RegimeTrainingPipeline
from ..config.hybrid_regime_config import HybridRegimeConfig

logger = logging.getLogger(__name__)

class RegimeHPOIntegration:
    """
    Integration class for regime-specific hyperparameter optimization.

    This class bridges the gap between the regime training pipeline and
    the HPO infrastructure, providing seamless optimization capabilities.
    """

    def __init__(self,
                 regime_config: Optional[HybridRegimeConfig] = None,
                 hpo_config: Optional[RegimeHPOConfig] = None):
        """
        Initialize the regime HPO integration.

        Args:
            regime_config: Regime training configuration
            hpo_config: HPO configuration
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 Initializing RegimeHPOIntegration...")

        # Initialize configurations
        self.regime_config = regime_config or HybridRegimeConfig()
        self.hpo_config = hpo_config or RegimeHPOConfig()

        # Initialize regime training pipeline
        self.regime_pipeline = RegimeTrainingPipeline(self.regime_config)

        # Initialize HPO wrapper
        self.hpo_wrapper = RegimeHPOWrapper(hpo_config=self.hpo_config)

        # Results storage
        self.optimization_results: Optional[RegimeHPOResult] = None
        self.optimization_history: List[Dict[str, Any]] = []

        self.logger.info("✅ RegimeHPOIntegration initialized successfully")

    def optimize_regime_detection_models(self,
                                       market_data: pd.DataFrame,
                                       regime_labels: Optional[np.ndarray] = None,
                                       features: Optional[List[str]] = None) -> RegimeHPOResult:
        """
        Optimize regime detection models using HPO.

        Args:
            market_data: Market data DataFrame
            regime_labels: Pre-computed regime labels (optional)
            features: Feature names to use (optional)

        Returns:
            Optimization results
        """
        self.logger.info("🎯 Starting regime detection model optimization")
        start_time = time.time()

        try:
            # Prepare data for optimization
            X, y = self._prepare_optimization_data(market_data, regime_labels, features)

            # Perform hierarchical optimization
            self.optimization_results = self.hpo_wrapper.hierarchical_optimization(X, y)

            # Store optimization history
            self.optimization_history.append({
                'timestamp': datetime.now().isoformat(),
                'data_shape': X.shape,
                'n_regimes': len(np.unique(y)),
                'optimization_time': time.time() - start_time,
                'strategy': self.hpo_config.optimization_strategy
            })

            optimization_time = time.time() - start_time
            self.logger.info(f"✅ Regime detection optimization completed in {optimization_time:.2f}s")

            return self.optimization_results

        except Exception as e:
            self.logger.error(f"❌ Regime detection optimization failed: {e}")
            raise

    def optimize_regime_prediction_models(self,
                                        market_data: pd.DataFrame,
                                        regime_labels: np.ndarray,
                                        prediction_horizon: int = 1) -> Dict[str, Any]:
        """
        Optimize regime prediction models using HPO.

        Args:
            market_data: Market data DataFrame
            regime_labels: Regime labels
            prediction_horizon: Prediction horizon in time steps

        Returns:
            Optimization results for prediction models
        """
        self.logger.info(f"🎯 Starting regime prediction model optimization (horizon={prediction_horizon})")
        start_time = time.time()

        try:
            # Prepare prediction data
            X, y = self._prepare_prediction_data(market_data, regime_labels, prediction_horizon)

            # Optimize base models for prediction
            base_results = self.hpo_wrapper.optimize_regime_base_models(X, y, 'prediction')

            # Optimize meta model for prediction
            meta_results = self.hpo_wrapper.optimize_regime_meta_model(X, y)

            optimization_time = time.time() - start_time

            results = {
                'base_model_results': base_results,
                'meta_model_results': meta_results,
                'optimization_time': optimization_time,
                'prediction_horizon': prediction_horizon
            }

            self.logger.info(f"✅ Regime prediction optimization completed in {optimization_time:.2f}s")
            return results

        except Exception as e:
            self.logger.error(f"❌ Regime prediction optimization failed: {e}")
            raise

    def optimize_meta_features(self,
                              market_data: pd.DataFrame,
                              regime_labels: np.ndarray,
                              base_model_predictions: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Optimize meta-features for enhanced regime detection.

        Args:
            market_data: Market data DataFrame
            regime_labels: Regime labels
            base_model_predictions: Base model predictions (optional)

        Returns:
            Meta-feature optimization results
        """
        self.logger.info("🎯 Starting meta-feature optimization")
        start_time = time.time()

        try:
            # Prepare data
            X, y = self._prepare_optimization_data(market_data, regime_labels)

            # Generate base model predictions if not provided
            if base_model_predictions is None:
                base_model_predictions = self._generate_base_model_predictions(X, y)

            # Optimize meta-features
            meta_feature_results = self.hpo_wrapper.optimize_meta_features(X, y, base_model_predictions)

            optimization_time = time.time() - start_time

            results = {
                'meta_feature_results': meta_feature_results,
                'optimization_time': optimization_time,
                'n_base_models': base_model_predictions.shape[1] if base_model_predictions is not None else 0
            }

            self.logger.info(f"✅ Meta-feature optimization completed in {optimization_time:.2f}s")
            return results

        except Exception as e:
            self.logger.error(f"❌ Meta-feature optimization failed: {e}")
            raise

    def run_complete_optimization(self,
                                 market_data: pd.DataFrame,
                                 regime_labels: Optional[np.ndarray] = None,
                                 features: Optional[List[str]] = None,
                                 save_results: bool = True) -> Dict[str, Any]:
        """
        Run complete regime optimization pipeline.

        Args:
            market_data: Market data DataFrame
            regime_labels: Pre-computed regime labels (optional)
            features: Feature names to use (optional)
            save_results: Whether to save results to file

        Returns:
            Complete optimization results
        """
        self.logger.info("🏗️ Starting complete regime optimization pipeline")
        total_start_time = time.time()

        try:
            # Step 1: Optimize regime detection models
            detection_results = self.optimize_regime_detection_models(
                market_data, regime_labels, features
            )

            # Step 2: Optimize regime prediction models
            prediction_results = self.optimize_regime_prediction_models(
                market_data, detection_results.base_model_results.get('regime_labels', np.array([]))
            )

            # Step 3: Optimize meta-features
            meta_feature_results = self.optimize_meta_features(
                market_data, detection_results.base_model_results.get('regime_labels', np.array([]))
            )

            # Compile complete results
            complete_results = {
                'detection_results': detection_results,
                'prediction_results': prediction_results,
                'meta_feature_results': meta_feature_results,
                'total_optimization_time': time.time() - total_start_time,
                'timestamp': datetime.now().isoformat(),
                'data_info': {
                    'n_samples': len(market_data),
                    'n_features': market_data.shape[1],
                    'n_regimes': len(np.unique(regime_labels)) if regime_labels is not None else 'unknown'
                }
            }

            # Save results if requested
            if save_results:
                self._save_optimization_results(complete_results)

            self.logger.info(f"🏆 Complete optimization pipeline finished in {complete_results['total_optimization_time']:.2f}s")
            return complete_results

        except Exception as e:
            self.logger.error(f"❌ Complete optimization pipeline failed: {e}")
            raise

    def _prepare_optimization_data(self,
                                  market_data: pd.DataFrame,
                                  regime_labels: Optional[np.ndarray] = None,
                                  features: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for optimization."""
        try:
            # Select features
            if features is None:
                # Use all numeric columns
                numeric_cols = market_data.select_dtypes(include=[np.number]).columns
                X = market_data[numeric_cols].values
            else:
                X = market_data[features].values

            # Generate regime labels if not provided
            if regime_labels is None:
                # Use regime detection from pipeline
                regime_labels = self.regime_pipeline.detect_regimes(market_data)

            y = regime_labels

            self.logger.info(f"📊 Prepared optimization data: {X.shape} features, {len(np.unique(y))} regimes")
            return X, y

        except Exception as e:
            self.logger.error(f"❌ Data preparation failed: {e}")
            raise

    def _prepare_prediction_data(self,
                                market_data: pd.DataFrame,
                                regime_labels: np.ndarray,
                                prediction_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for prediction optimization."""
        try:
            # Create lagged features for prediction
            lagged_data = self._create_lagged_features(market_data, prediction_horizon)

            # Align with regime labels
            X = lagged_data[:-prediction_horizon]  # Features
            y = regime_labels[prediction_horizon:]  # Future regime labels

            self.logger.info(f"📊 Prepared prediction data: {X.shape} features, horizon={prediction_horizon}")
            return X, y

        except Exception as e:
            self.logger.error(f"❌ Prediction data preparation failed: {e}")
            raise

    def _create_lagged_features(self, data: pd.DataFrame, max_lags: int) -> np.ndarray:
        """Create lagged features for time series prediction."""
        try:
            lagged_features = []

            for col in data.select_dtypes(include=[np.number]).columns:
                for lag in range(1, max_lags + 1):
                    lagged_features.append(data[col].shift(lag))

            # Combine all lagged features
            lagged_df = pd.concat(lagged_features, axis=1)
            lagged_df = lagged_df.dropna()

            return lagged_df.values

        except Exception as e:
            self.logger.error(f"❌ Lagged feature creation failed: {e}")
            raise

    def _generate_base_model_predictions(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate base model predictions for meta-feature optimization."""
        try:
            # This would train base models and generate predictions
            # For now, return random predictions as placeholder
            n_regimes = len(np.unique(y))
            n_models = 5  # Number of base models

            predictions = np.random.rand(len(X), n_models * n_regimes)

            self.logger.info(f"📊 Generated base model predictions: {predictions.shape}")
            return predictions

        except Exception as e:
            self.logger.error(f"❌ Base model prediction generation failed: {e}")
            raise

    def _save_optimization_results(self, results: Dict[str, Any], filepath: Optional[str] = None):
        """Save optimization results to file."""
        try:
            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"regime_optimization_results_{timestamp}.json"

            # Convert results to serializable format
            serializable_results = self._make_serializable(results)

            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)

            self.logger.info(f"💾 Optimization results saved to {filepath}")

        except Exception as e:
            self.logger.error(f"❌ Failed to save optimization results: {e}")

    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results."""
        if self.optimization_results is None:
            return {'status': 'no_optimization_performed'}

        return {
            'status': 'optimization_completed',
            'total_time': self.optimization_results.total_optimization_time,
            'strategy': self.optimization_results.optimization_strategy,
            'n_trials': self.optimization_results.n_total_trials,
            'base_models_optimized': list(self.optimization_results.base_model_best_params.keys()),
            'best_base_scores': self.optimization_results.base_model_best_scores,
            'meta_model_score': self.optimization_results.meta_model_best_score,
            'convergence_info': self.optimization_results.convergence_info
        }

# Convenience functions for easy integration
def run_regime_optimization(market_data: pd.DataFrame,
                           regime_labels: Optional[np.ndarray] = None,
                           features: Optional[List[str]] = None,
                           hpo_config: Optional[RegimeHPOConfig] = None) -> Dict[str, Any]:
    """
    Convenience function to run regime optimization.

    Args:
        market_data: Market data DataFrame
        regime_labels: Pre-computed regime labels (optional)
        features: Feature names to use (optional)
        hpo_config: HPO configuration

    Returns:
        Optimization results
    """
    integration = RegimeHPOIntegration(hpo_config=hpo_config)
    return integration.run_complete_optimization(market_data, regime_labels, features)

def create_regime_hpo_integration_config(**kwargs) -> Dict[str, Any]:
    """Create configuration for regime HPO integration."""
    default_config = {
        'optimization_strategy': 'hierarchical',
        'base_model_n_trials': 100,
        'meta_model_n_trials': 50,
        'enable_meta_feature_optimization': True,
        'enable_parallel': True,
        'max_workers': 4,
        'enable_oof_validation': True,
        'enable_time_series_cv': True
    }

    default_config.update(kwargs)
    return default_config
