"""
Analyst Models Integration Example

This example shows how to integrate the new Analyst models (A1-A4) 
into the existing analyst training pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
import asyncio

# Import the new analyst models
from .analyst_models_orchestrator import (
    AnalystModelsOrchestrator,
    AnalystModelsConfig,
    create_analyst_models_orchestrator
)

logger = logging.getLogger(__name__)


class AnalystModelsIntegration:
    """Integration class for Analyst models with existing pipeline."""
    
    def __init__(self, config: Optional[AnalystModelsConfig] = None):
        self.config = config or AnalystModelsConfig()
        self.orchestrator = None
        self.is_initialized = False
        
    async def initialize(self) -> bool:
        """Initialize the analyst models integration."""
        try:
            logger.info("Initializing Analyst Models Integration...")
            
            # Create orchestrator
            self.orchestrator = create_analyst_models_orchestrator(self.config)
            self.is_initialized = True
            
            logger.info("✅ Analyst Models Integration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Analyst Models Integration: {e}")
            return False
    
    async def train_models(self, 
                          training_data: pd.DataFrame,
                          feature_columns: List[str],
                          target_column: str,
                          regime_assignments: Optional[np.ndarray] = None,
                          sample_weight: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train all analyst models."""
        if not self.is_initialized:
            raise ValueError("Integration must be initialized before training")
        
        try:
            logger.info("Training Analyst Models...")
            
            # Prepare data
            X = training_data[feature_columns].values
            y = training_data[target_column].values
            
            # Train orchestrator
            await self.orchestrator.fit(
                X=X,
                y=y,
                regimes=regime_assignments,
                sample_weight=sample_weight
            )
            
            # Get performance metrics
            performance = self.orchestrator.get_model_performance()
            
            # Save models if configured
            if self.config.save_models:
                self.orchestrator.save_models()
            
            logger.info("✅ Analyst Models training completed successfully")
            return {
                'success': True,
                'performance': performance,
                'models_trained': list(self.orchestrator.training_results.keys()),
                'stacker_trained': self.orchestrator.stacker is not None
            }
            
        except Exception as e:
            logger.error(f"❌ Analyst Models training failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def predict_green_light(self, 
                           market_data: pd.DataFrame,
                           feature_columns: List[str],
                           regime_assignments: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Predict green light probability and uncertainty."""
        if not self.is_initialized or not self.orchestrator.is_fitted:
            raise ValueError("Models must be trained before prediction")
        
        try:
            # Prepare data
            X = market_data[feature_columns].values
            
            # Get predictions
            probabilities = self.orchestrator.predict_proba(X, regime_assignments)
            uncertainty = self.orchestrator.predict_uncertainty(X, regime_assignments)
            
            # Determine green light decision
            green_light_threshold = 0.6  # Configurable threshold
            green_light_decisions = (probabilities > green_light_threshold).astype(int)
            
            return {
                'probabilities': probabilities,
                'green_light_decisions': green_light_decisions,
                'uncertainty': uncertainty,
                'threshold': green_light_threshold,
                'confidence_levels': uncertainty.get('confidence_intervals', {}),
                'margin_stats': uncertainty.get('margin_stats', {}),
                'regime_stats': uncertainty.get('regime_stats', {})
            }
            
        except Exception as e:
            logger.error(f"❌ Green light prediction failed: {e}")
            return {
                'error': str(e),
                'probabilities': np.zeros(len(market_data)),
                'green_light_decisions': np.zeros(len(market_data), dtype=int)
            }
    
    def get_model_insights(self) -> Dict[str, Any]:
        """Get insights from trained models."""
        if not self.is_initialized or not self.orchestrator.is_fitted:
            return {}
        
        insights = {
            'performance': self.orchestrator.get_model_performance(),
            'feature_importance': {},
            'model_info': {}
        }
        
        # Get feature importance from each model
        for model_name, result in self.orchestrator.training_results.items():
            if result.get('model') is not None and hasattr(result['model'], 'get_feature_importance'):
                insights['feature_importance'][model_name] = result['model'].get_feature_importance()
            
            # Get model-specific info
            if hasattr(result['model'], 'get_booster_info'):
                insights['model_info'][model_name] = result['model'].get_booster_info()
            elif hasattr(result['model'], 'get_catboost_info'):
                insights['model_info'][model_name] = result['model'].get_catboost_info()
        
        # Get stacker info
        if self.orchestrator.stacker is not None:
            insights['stacker_info'] = {
                'feature_importance': self.orchestrator.stacker.get_feature_importance(),
                'regime_calibration': self.orchestrator.stacker.get_regime_calibration_info()
            }
        
        return insights
    
    def load_models(self, model_path: str) -> bool:
        """Load pre-trained models."""
        try:
            logger.info(f"Loading models from {model_path}...")
            
            self.orchestrator = AnalystModelsOrchestrator.load_models(model_path, self.config)
            self.is_initialized = True
            
            logger.info("✅ Models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            return False


# Example usage function
async def example_usage():
    """Example of how to use the Analyst Models Integration."""
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 300
    
    # Generate synthetic market data
    market_data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='15T'),
        **{f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)},
        'regime': np.random.choice(['bull', 'bear', 'sideways'], n_samples),
        'target': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # Imbalanced
    })
    
    feature_columns = [f'feature_{i}' for i in range(n_features)]
    target_column = 'target'
    regime_assignments = market_data['regime'].map({'bull': 0, 'bear': 1, 'sideways': 2}).values
    
    # Create integration
    config = AnalystModelsConfig(
        enable_a1=True,
        enable_a2=True,
        enable_a3=True,
        enable_a4=True,
        enable_stacker=True,
        enable_parallel_training=True,
        max_workers=4,
        save_models=True,
        output_directory="generated/analyst_models_example"
    )
    
    integration = AnalystModelsIntegration(config)
    
    # Initialize
    await integration.initialize()
    
    # Train models
    training_result = await integration.train_models(
        training_data=market_data,
        feature_columns=feature_columns,
        target_column=target_column,
        regime_assignments=regime_assignments
    )
    
    print("Training Result:", training_result)
    
    # Make predictions
    prediction_result = integration.predict_green_light(
        market_data=market_data.head(100),
        feature_columns=feature_columns,
        regime_assignments=regime_assignments[:100]
    )
    
    print("Prediction Result Keys:", list(prediction_result.keys()))
    print("Green Light Decisions:", prediction_result['green_light_decisions'][:10])
    print("Probabilities:", prediction_result['probabilities'][:10])
    
    # Get insights
    insights = integration.get_model_insights()
    print("Model Performance:", insights['performance'])
    
    return integration


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())