"""
Tactician Models Training Step - Refactored

This step handles per-regime training of individual Tactician models using common dependencies.
The Tactician operates on 1m timeframe and decides WHEN to trade based on Analyst's green light signals.
This is a refactored version that demonstrates the use of common utilities.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

from src.utils.logger import system_logger
from src.utils.ml_common.config import PerRegimeTrainingConfig
from src.utils.ml_common.training import PerRegimeTrainingStep

logger = system_logger.getChild('TacticianModelsTrainingRefactored')


class TacticianModelsTrainingStepRefactored(PerRegimeTrainingStep):
    """
    Tactician Models Training Step with per-regime training, HPO, saving, and metrics.
    
    The Tactician operates on 1m timeframe and is trained on:
    1. Only periods where the Analyst gives a green light
    2. Using the Analyst's model outputs as input features
    
    This is a refactored version that uses common dependencies to reduce code duplication.
    """
    
    def __init__(self, config: Optional[PerRegimeTrainingConfig] = None):
        """
        Initialize Tactician models training step.
        
        Args:
            config: Per-regime training configuration
        """
        # Set default configuration for tactician models
        if config is None:
            config = PerRegimeTrainingConfig(
                model_name="tactician_models",
                timeframe="1m",
                model_types=["NeuralObliviousDecisionEnsembles", "CatBoostRegressor", "LGBMRegressor", "Ridge"],
                hpo_n_trials=100,
                hpo_timeout_seconds=3600,
                min_samples_per_regime=1000,
                enable_data_augmentation=True,
                augmentation_method="smote",
                model_save_path="./models/tactician_models",
                evaluation_metrics=["mse", "mae", "r2", "mape", "smape"]
            )
        
        super().__init__(config)
        self.logger = logger.getChild('TacticianModelsTrainingStepRefactored')
        
        self.logger.info("✅ Tactician Models Training Step (Refactored) initialized")
    
    def execute(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        hmm_states: Optional[np.ndarray] = None,
        analyst_signals: Optional[np.ndarray] = None,
        analyst_model_outputs: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Execute Tactician models training step.
        
        Args:
            X: Input features (1m timeframe with cross-timeframe features)
            y: Target values (tactician outputs - timing decisions)
            regime_labels: Regime labels for each sample
            feature_names: Names of input features
            hmm_states: HMM cluster/regime states
            analyst_signals: Binary signals from Analyst (green light indicators)
            analyst_model_outputs: Analyst model predictions used as features
            
        Returns:
            Dictionary containing training results and metadata
        """
        self.logger.info("🚀 Starting Tactician models training step (refactored)")
        
        # Filter data to only include periods where Analyst gives green light
        if analyst_signals is not None:
            green_light_mask = analyst_signals == 1
            self.logger.info(f"📊 Filtering to {np.sum(green_light_mask)} samples with Analyst green light signals")
            
            X = X[green_light_mask]
            y = y[green_light_mask]
            regime_labels = regime_labels[green_light_mask]
            if hmm_states is not None:
                hmm_states = hmm_states[green_light_mask]
        
        # Add analyst model outputs as features if provided
        if analyst_model_outputs is not None:
            if analyst_signals is not None:
                analyst_model_outputs = analyst_model_outputs[green_light_mask]
            
            # Concatenate analyst outputs as additional features
            X = np.column_stack([X, analyst_model_outputs])
            
            # Update feature names
            if feature_names is not None:
                analyst_feature_names = [f"analyst_output_{i}" for i in range(analyst_model_outputs.shape[1])]
                feature_names = feature_names + analyst_feature_names
            else:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Use the parent class execute method with additional tactician-specific logic
        results = super().execute(
            X=X,
            y=y,
            regime_labels=regime_labels,
            feature_names=feature_names,
            hmm_states=hmm_states,
            is_classification=False,  # Tactician models are typically regression
            symbol=None,  # Can be passed as kwargs
            exchange=None,
            timeframe=self.config.timeframe
        )
        
        # Add tactician-specific post-processing if needed
        if 'error' not in results:
            results = self._add_tactician_specific_metadata(results, analyst_signals)
        
        return results
    
    def _add_tactician_specific_metadata(self, results: Dict[str, Any], analyst_signals: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Add tactician-specific metadata to results.
        
        Args:
            results: Training results
            analyst_signals: Analyst green light signals for analysis
            
        Returns:
            Enhanced results with tactician-specific metadata
        """
        # Add tactician-specific analysis
        if 'regime_analysis' in results:
            regime_analysis = results['regime_analysis']
            
            # Calculate tactician-specific metrics
            tactician_metrics = {
                'total_regimes': len(regime_analysis.get('unique_regimes', [])),
                'sufficient_regimes': len(regime_analysis.get('sufficient_regimes', [])),
                'insufficient_regimes': len(regime_analysis.get('insufficient_regimes', [])),
                'regime_balance': regime_analysis.get('regime_balance_train', 0.0),
                'timeframe': self.config.timeframe,
                'model_types': self.config.model_types
            }
            
            # Add analyst signal analysis if available
            if analyst_signals is not None:
                green_light_rate = np.mean(analyst_signals)
                tactician_metrics.update({
                    'analyst_green_light_rate': green_light_rate,
                    'total_samples_with_green_light': int(np.sum(analyst_signals)),
                    'total_samples_analyzed': len(analyst_signals)
                })
            
            results['tactician_metrics'] = tactician_metrics
        
        # Add model performance summary
        if 'evaluation_results' in results:
            evaluation_results = results['evaluation_results']
            
            # Calculate best performing model per regime
            best_models = {}
            for regime, regime_metrics in evaluation_results.items():
                if isinstance(regime_metrics, dict) and 'error' not in regime_metrics:
                    best_model = None
                    best_r2 = -np.inf
                    
                    for model_name, metrics in regime_metrics.items():
                        if isinstance(metrics, dict) and 'r2' in metrics:
                            if metrics['r2'] > best_r2:
                                best_r2 = metrics['r2']
                                best_model = model_name
                    
                    if best_model:
                        best_models[regime] = {
                            'model': best_model,
                            'r2_score': best_r2
                        }
            
            results['best_models_per_regime'] = best_models
        
        # Add timing-specific analysis
        timing_analysis = {
            'base_timeframe': self.config.timeframe,
            'cross_timeframe_features': True,
            'analyst_dependency': True,
            'timing_decision_role': True
        }
        results['timing_analysis'] = timing_analysis
        
        return results


# Convenience functions for backward compatibility
def create_tactician_models_training_step_refactored(
    config: Optional[PerRegimeTrainingConfig] = None
) -> TacticianModelsTrainingStepRefactored:
    """Create Tactician models training step (refactored)."""
    return TacticianModelsTrainingStepRefactored(config)


def execute_tactician_models_training_refactored(
    X: np.ndarray,
    y: np.ndarray,
    regime_labels: np.ndarray,
    config: Optional[PerRegimeTrainingConfig] = None,
    feature_names: Optional[List[str]] = None,
    hmm_states: Optional[np.ndarray] = None,
    analyst_signals: Optional[np.ndarray] = None,
    analyst_model_outputs: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Execute Tactician models training step (refactored)."""
    step = create_tactician_models_training_step_refactored(config)
    return step.execute(X, y, regime_labels, feature_names, hmm_states, analyst_signals, analyst_model_outputs)


# Example usage and comparison
if __name__ == "__main__":
    # Example of how to use the refactored version
    print("Tactician Models Training Step - Refactored Version")
    print("=" * 50)
    
    # Create configuration
    config = PerRegimeTrainingConfig(
        model_name="tactician_models",
        timeframe="1m",
        model_types=["NeuralObliviousDecisionEnsembles", "CatBoostRegressor", "LGBMRegressor", "Ridge"],
        hpo_n_trials=50,  # Reduced for demo
        enable_hpo=True,
        save_models=True,
        model_save_path="./models/tactician_models_refactored"
    )
    
    # Create training step
    training_step = create_tactician_models_training_step_refactored(config)
    
    print(f"✅ Created tactician training step with {len(config.model_types)} model types")
    print(f"📊 HPO enabled: {config.enable_hpo}")
    print(f"💾 Save models: {config.save_models}")
    print(f"📁 Save path: {config.model_save_path}")
    print(f"⏰ Base timeframe: {config.timeframe}")
    
    # The actual training would be called with:
    # results = training_step.execute(X, y, regime_labels, feature_names, hmm_states, analyst_signals, analyst_model_outputs)
    
    print("\n🎯 Tactician Module Features:")
    print("- Operates on 1m timeframe with cross-timeframe features")
    print("- Trained only on Analyst green light periods")
    print("- Uses Analyst model outputs as input features")
    print("- Decides WHEN to trade (timing decisions)")
    print("- Models: NODE, CatBoost, LightGBM, Ridge")
    print("- Comprehensive context from multi-timeframe dynamics")
    
    print("\n🔄 Integration with Analyst:")
    print("- Receives green light signals from Analyst")
    print("- Uses Analyst predictions as additional features")
    print("- Focuses on timing rather than trade decision")
    print("- Operates on higher frequency (1m vs 5m)")