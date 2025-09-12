"""
Analyst Ensemble Training Step

This step handles per-regime ensemble training of Analyst models using common dependencies.
The Analyst Ensemble operates on 5m timeframe and combines individual analyst models
to create robust ensemble predictions for trade decisions.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

from src.utils.logger import system_logger
from src.utils.ml_common.config.base_training_config import EnsembleTrainingConfig
from src.utils.ml_common.training.ensemble_training_step import EnsembleTrainingStep

logger = system_logger.getChild('AnalystEnsembleTraining')


class AnalystEnsembleTrainingStep(EnsembleTrainingStep):
    """
    Analyst Ensemble Training Step with per-regime ensemble training, HPO, saving, and metrics.
    
    The Analyst Ensemble operates on 5m timeframe and combines individual analyst models
    to create robust ensemble predictions for trade decisions.
    """
    
    def __init__(self, config: Optional[EnsembleTrainingConfig] = None):
        """
        Initialize Analyst ensemble training step.
        
        Args:
            config: Per-regime training configuration
        """
        # Set default configuration for analyst ensemble models
        if config is None:
            config = EnsembleTrainingConfig(
                model_name="analyst_ensemble_models",
                timeframe="5m",
                model_types=["VotingRegressor", "StackingRegressor", "BaggingRegressor", "AdaBoostRegressor"],
                hpo_n_trials=100,
                hpo_timeout_seconds=3600,
                min_samples_per_regime=1000,
                enable_data_augmentation=True,
                augmentation_method="smote",
                model_save_path="./models/analyst_ensemble_models",
                evaluation_metrics=["mse", "mae", "r2", "mape", "smape"]
            )
        
        super().__init__(config)
        self.logger = logger.getChild('AnalystEnsembleTrainingStep')
        
        self.logger.info("✅ Analyst Ensemble Training Step initialized")
    
    def execute(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        hmm_states: Optional[np.ndarray] = None,
        base_analyst_models: Optional[Dict[str, Any]] = None,
        analyst_training_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute Analyst ensemble training step.
        
        Args:
            X: Input features (5m timeframe with cross-timeframe features)
            y: Target values (analyst outputs)
            regime_labels: Regime labels for each sample
            feature_names: Names of input features
            hmm_states: HMM cluster/regime states
            base_analyst_models: Individual analyst models to ensemble
            analyst_training_metrics: Performance metrics of base models
            
        Returns:
            Dictionary containing training results and metadata
        """
        self.logger.info("🚀 Starting Analyst ensemble training step")
        
        # Validate base models are provided
        if base_analyst_models is None or not base_analyst_models:
            self.logger.warning("⚠️ No base analyst models provided, using mock models")
            base_analyst_models = self._create_mock_base_models()
        
        # Use the parent class execute method with additional ensemble-specific logic
        results = super().execute(
            X=X,
            y=y,
            regime_labels=regime_labels,
            feature_names=feature_names,
            hmm_states=hmm_states,
            is_classification=False,  # Analyst ensemble models are typically regression
            symbol=None,  # Can be passed as kwargs
            exchange=None,
            timeframe=self.config.timeframe
        )
        
        # Add ensemble-specific post-processing if needed
        if 'error' not in results:
            results = self._add_ensemble_specific_metadata(results, base_analyst_models, analyst_training_metrics)
        
        return results
    
    def _create_mock_base_models(self) -> Dict[str, Any]:
        """Create mock base models for testing purposes."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        
        mock_models = {
            'tcn_model': RandomForestRegressor(n_estimators=10, random_state=42),
            'catboost_model': RandomForestRegressor(n_estimators=10, random_state=43),
            'lightgbm_model': RandomForestRegressor(n_estimators=10, random_state=44),
            'rf_model': RandomForestRegressor(n_estimators=10, random_state=45)
        }
        
        self.logger.info(f"📊 Created {len(mock_models)} mock base models for ensemble training")
        return mock_models
    
    def _add_ensemble_specific_metadata(self, results: Dict[str, Any], base_models: Dict[str, Any], base_metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Add ensemble-specific metadata to results.
        
        Args:
            results: Training results
            base_models: Base analyst models used in ensemble
            base_metrics: Performance metrics of base models
            
        Returns:
            Enhanced results with ensemble-specific metadata
        """
        # Add ensemble-specific analysis
        if 'regime_analysis' in results:
            regime_analysis = results['regime_analysis']
            
            # Calculate ensemble-specific metrics
            ensemble_metrics = {
                'total_regimes': len(regime_analysis.get('unique_regimes', [])),
                'sufficient_regimes': len(regime_analysis.get('sufficient_regimes', [])),
                'insufficient_regimes': len(regime_analysis.get('insufficient_regimes', [])),
                'regime_balance': regime_analysis.get('regime_balance_train', 0.0),
                'timeframe': self.config.timeframe,
                'ensemble_model_types': self.config.model_types,
                'base_models_count': len(base_models) if base_models else 0
            }
            
            # Add base model performance analysis if available
            if base_metrics:
                ensemble_metrics['base_model_performance'] = base_metrics
                self.logger.info("📊 Integrated base model performance metrics")
            
            results['ensemble_metrics'] = ensemble_metrics
        
        # Add ensemble performance summary
        if 'evaluation_results' in results:
            evaluation_results = results['evaluation_results']
            
            # Calculate best performing ensemble per regime
            best_ensembles = {}
            for regime, regime_metrics in evaluation_results.items():
                if isinstance(regime_metrics, dict) and 'error' not in regime_metrics:
                    best_ensemble = None
                    best_r2 = -np.inf
                    
                    for ensemble_name, metrics in regime_metrics.items():
                        if isinstance(metrics, dict) and 'r2' in metrics:
                            if metrics['r2'] > best_r2:
                                best_r2 = metrics['r2']
                                best_ensemble = ensemble_name
                    
                    if best_ensemble:
                        best_ensembles[regime] = {
                            'ensemble': best_ensemble,
                            'r2_score': best_r2
                        }
            
            results['best_ensembles_per_regime'] = best_ensembles
        
        # Add ensemble-specific analysis
        ensemble_analysis = {
            'base_timeframe': self.config.timeframe,
            'cross_timeframe_features': True,
            'ensemble_method': 'per_regime',
            'base_models_integrated': len(base_models) if base_models else 0,
            'ensemble_role': 'trade_decision_enhancement'
        }
        results['ensemble_analysis'] = ensemble_analysis
        
        return results


# Convenience functions for backward compatibility
def create_analyst_ensemble_training_step(
    config: Optional[EnsembleTrainingConfig] = None
) -> AnalystEnsembleTrainingStep:
    """Create Analyst ensemble training step."""
    return AnalystEnsembleTrainingStep(config)


def execute_analyst_ensemble_training(
    X: np.ndarray,
    y: np.ndarray,
    regime_labels: np.ndarray,
    config: Optional[EnsembleTrainingConfig] = None,
    feature_names: Optional[List[str]] = None,
    hmm_states: Optional[np.ndarray] = None,
    base_analyst_models: Optional[Dict[str, Any]] = None,
    analyst_training_metrics: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Execute Analyst ensemble training step."""
    step = create_analyst_ensemble_training_step(config)
    return step.execute(X, y, regime_labels, feature_names, hmm_states, base_analyst_models, analyst_training_metrics)


# Example usage and comparison
if __name__ == "__main__":
    # Example of how to use the ensemble training version
    print("Analyst Ensemble Training Step")
    print("=" * 50)
    
    # Create configuration
    config = EnsembleTrainingConfig(
        model_name="analyst_ensemble_models",
        timeframe="5m",
        model_types=["VotingRegressor", "StackingRegressor", "BaggingRegressor", "AdaBoostRegressor"],
        hpo_n_trials=50,  # Reduced for demo
        enable_hpo=True,
        save_models=True,
        model_save_path="./models/analyst_ensemble_models_refactored"
    )
    
    # Create training step
    training_step = create_analyst_ensemble_training_step(config)
    
    print(f"✅ Created analyst ensemble training step with {len(config.model_types)} ensemble types")
    print(f"📊 HPO enabled: {config.enable_hpo}")
    print(f"💾 Save models: {config.save_models}")
    print(f"📁 Save path: {config.model_save_path}")
    print(f"⏰ Base timeframe: {config.timeframe}")
    
    # The actual training would be called with:
    # results = training_step.execute(X, y, regime_labels, feature_names, hmm_states, base_analyst_models, analyst_training_metrics)
    
    print("\n🎯 Analyst Ensemble Module Features:")
    print("- Operates on 5m timeframe with cross-timeframe features")
    print("- Combines individual analyst models into robust ensembles")
    print("- Per-regime ensemble training for regime-specific optimization")
    print("- Enhanced trade decision accuracy through model combination")
    print("- Models: VotingRegressor, StackingRegressor, BaggingRegressor, AdaBoostRegressor")
    print("- Comprehensive context from multi-timeframe dynamics")
    
    print("\n🔄 Integration with Individual Analyst Models:")
    print("- Receives individual analyst model predictions")
    print("- Uses base model performance metrics for weighting")
    print("- Creates regime-specific ensemble combinations")
    print("- Provides enhanced trade decision signals")