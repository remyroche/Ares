"""
Analyst Models Training Step - Refactored

This step handles per-regime training of individual Analyst models using common dependencies.
This is a refactored version that demonstrates the use of common utilities.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

from src.utils.logger import system_logger
from src.utils.ml_common.config import PerRegimeTrainingConfig
from src.utils.ml_common.training import PerRegimeTrainingStep

logger = system_logger.getChild('AnalystModelsTrainingRefactored')


class AnalystModelsTrainingStepRefactored(PerRegimeTrainingStep):
    """
    Analyst Models Training Step with per-regime training, HPO, saving, and metrics.
    
    This is a refactored version that uses common dependencies to reduce code duplication.
    """
    
    def __init__(self, config: Optional[PerRegimeTrainingConfig] = None):
        """
        Initialize Analyst models training step.
        
        Args:
            config: Per-regime training configuration
        """
        # Set default configuration for analyst models
        if config is None:
            config = PerRegimeTrainingConfig(
                model_name="analyst_models",
                timeframe="5m",
                model_types=["TEMPORAL_FUSION_TRANSFORMER", "TABNET", "HIST_GRADIENT_BOOSTING", "EXTRA_TREES"],
                hpo_n_trials=100,
                hpo_timeout_seconds=3600,
                min_samples_per_regime=1000,
                enable_data_augmentation=True,
                augmentation_method="smote",
                model_save_path="./models/analyst_models",
                evaluation_metrics=["mse", "mae", "r2", "mape", "smape"]
            )
        
        super().__init__(config)
        self.logger = logger.getChild('AnalystModelsTrainingStepRefactored')
        
        self.logger.info("✅ Analyst Models Training Step (Refactored) initialized")
    
    def execute(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        hmm_states: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Execute Analyst models training step.
        
        Args:
            X: Input features
            y: Target values (analyst outputs)
            regime_labels: Regime labels for each sample
            feature_names: Names of input features
            hmm_states: HMM cluster/regime states
            
        Returns:
            Dictionary containing training results and metadata
        """
        self.logger.info("🚀 Starting Analyst models training step (refactored)")
        
        # Use the parent class execute method with additional analyst-specific logic
        results = super().execute(
            X=X,
            y=y,
            regime_labels=regime_labels,
            feature_names=feature_names,
            hmm_states=hmm_states,
            is_classification=False,  # Analyst models are typically regression
            symbol=None,  # Can be passed as kwargs
            exchange=None,
            timeframe=self.config.timeframe
        )
        
        # Add analyst-specific post-processing if needed
        if 'error' not in results:
            results = self._add_analyst_specific_metadata(results)
        
        return results
    
    def _add_analyst_specific_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add analyst-specific metadata to results.
        
        Args:
            results: Training results
            
        Returns:
            Enhanced results with analyst-specific metadata
        """
        # Add analyst-specific analysis
        if 'regime_analysis' in results:
            regime_analysis = results['regime_analysis']
            
            # Calculate analyst-specific metrics
            analyst_metrics = {
                'total_regimes': len(regime_analysis.get('unique_regimes', [])),
                'sufficient_regimes': len(regime_analysis.get('sufficient_regimes', [])),
                'insufficient_regimes': len(regime_analysis.get('insufficient_regimes', [])),
                'regime_balance': regime_analysis.get('regime_balance_train', 0.0)
            }
            
            results['analyst_metrics'] = analyst_metrics
        
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
        
        return results


# Convenience functions for backward compatibility
def create_analyst_models_training_step_refactored(
    config: Optional[PerRegimeTrainingConfig] = None
) -> AnalystModelsTrainingStepRefactored:
    """Create Analyst models training step (refactored)."""
    return AnalystModelsTrainingStepRefactored(config)


def execute_analyst_models_training_refactored(
    X: np.ndarray,
    y: np.ndarray,
    regime_labels: np.ndarray,
    config: Optional[PerRegimeTrainingConfig] = None,
    feature_names: Optional[List[str]] = None,
    hmm_states: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Execute Analyst models training step (refactored)."""
    step = create_analyst_models_training_step_refactored(config)
    return step.execute(X, y, regime_labels, feature_names, hmm_states)


# Example usage and comparison
if __name__ == "__main__":
    # Example of how to use the refactored version
    print("Analyst Models Training Step - Refactored Version")
    print("=" * 50)
    
    # Create configuration
    config = PerRegimeTrainingConfig(
        model_name="analyst_models",
        timeframe="5m",
        model_types=["TEMPORAL_FUSION_TRANSFORMER", "TABNET", "HIST_GRADIENT_BOOSTING"],
        hpo_n_trials=50,  # Reduced for demo
        enable_hpo=True,
        save_models=True,
        model_save_path="./models/analyst_models_refactored"
    )
    
    # Create training step
    training_step = create_analyst_models_training_step_refactored(config)
    
    print(f"✅ Created training step with {len(config.model_types)} model types")
    print(f"📊 HPO enabled: {config.enable_hpo}")
    print(f"💾 Save models: {config.save_models}")
    print(f"📁 Save path: {config.model_save_path}")
    
    # The actual training would be called with:
    # results = training_step.execute(X, y, regime_labels, feature_names, hmm_states)
    
    print("\n🎯 Benefits of refactored version:")
    print("- Reduced from ~600 lines to ~150 lines (75% reduction)")
    print("- Uses common dependencies for consistency")
    print("- Easier to maintain and extend")
    print("- Standardized error handling and logging")
    print("- Reusable components across all training modules")