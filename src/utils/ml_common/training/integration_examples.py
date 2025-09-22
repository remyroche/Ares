"""
Integration Examples for Enhanced Training Utilities

This module provides concrete examples of how to integrate the enhanced training
utilities into existing Analyst and Tactician training steps.

Examples:
1. Analyst Models Training Integration
2. Tactician Models Training Integration  
3. Ensemble Training Integration
4. Cross-Validation Integration
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
import time

# Import enhanced training utilities
from .enhanced_training_utils import (
    EnhancedTrainingUtils,
    EarlyStoppingConfig,
    PurgedCVConfig,
    OverfittingMonitorConfig,
    RegularizationConfig
)

from .training_integration import (
    enhanced_training,
    enhanced_ensemble_training,
    enhanced_cross_validation,
    TrainingStepEnhancer,
    TrainingIntegrationConfig
)

# Import existing training utilities
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint_info(*args, **kwargs): print(f"INFO: {args[0] if args else ''}")
    def tprint_success(*args, **kwargs): print(f"SUCCESS: {args[0] if args else ''}")
    def tprint_warning(*args, **kwargs): print(f"WARNING: {args[0] if args else ''}")
    def tprint_error(*args, **kwargs): print(f"ERROR: {args[0] if args else ''}")


class AnalystTrainingIntegration:
    """
    Example integration for Analyst models training with enhanced utilities.
    """
    
    def __init__(self, config: Optional[TrainingIntegrationConfig] = None):
        """Initialize Analyst training integration."""
        self.config = config or TrainingIntegrationConfig(
            enable_early_stopping=True,
            enable_purged_cv=True,
            enable_lookahead_detection=True,
            enable_temporal_splits=True,
            enable_regularization=True,
            enable_overfitting_monitoring=True,
            model_type='auto'
        )
        
        self.enhancer = TrainingStepEnhancer(self.config)
        tprint_success("✅ Analyst Training Integration initialized")
    
    def train_analyst_models_enhanced(self, 
                                     X: np.ndarray, 
                                     y: np.ndarray, 
                                     models: Dict[str, Any],
                                     timestamps: Optional[np.ndarray] = None,
                                     regime_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train Analyst models with enhanced overfitting prevention and lookahead bias detection.
        
        Args:
            X: Feature matrix
            y: Target array
            models: Dictionary of models to train
            timestamps: Timestamp array (optional)
            regime_labels: Regime labels (optional)
            
        Returns:
            Training results with enhanced metadata
        """
        tprint_info("🚀 Starting enhanced Analyst models training...")
        
        results = {
            'models': {},
            'training_metadata': {},
            'overfitting_warnings': [],
            'ensemble_diversity': None,
            'success': True
        }
        
        try:
            # Validate temporal data
            if self.config.enable_lookahead_detection:
                tprint_info("🔍 Validating temporal data for lookahead bias...")
                is_valid, warnings = self.enhancer.enhanced_utils.validate_temporal_data(
                    X, y, timestamps, strict_mode=True
                )
                results['overfitting_warnings'].extend(warnings)
                
                if not is_valid:
                    tprint_error("❌ Temporal data validation failed")
                    results['success'] = False
                    return results
            
            # Train each model with enhancements
            trained_models = []
            for model_name, model in models.items():
                tprint_info(f"🚀 Training Analyst model: {model_name}")
                
                trained_model, model_metadata = self.enhancer.enhance_training_step(
                    X, y, model, timestamps, f"analyst_{model_name}"
                )
                
                results['models'][model_name] = {
                    'model': trained_model,
                    'metadata': model_metadata
                }
                trained_models.append(trained_model)
                
                if model_metadata.get('overfitting_detected', False):
                    results['overfitting_warnings'].append(f"Overfitting detected in {model_name}")
            
            # Calculate ensemble diversity if multiple models
            if len(trained_models) > 1 and self.config.enable_ensemble_diversity:
                tprint_info("📊 Calculating Analyst ensemble diversity...")
                diversity_metrics = self.enhancer.enhanced_utils.calculate_ensemble_diversity(
                    trained_models, X, y
                )
                results['ensemble_diversity'] = diversity_metrics
                
                if diversity_metrics.get('diversity_score', 0) < 0.1:
                    tprint_warning("⚠️ Low Analyst ensemble diversity detected")
                else:
                    tprint_success("✅ Good Analyst ensemble diversity")
            
            # Add comprehensive metadata
            results['training_metadata'] = {
                'total_models': len(models),
                'enhancements_applied': [
                    'lookahead_bias_detection',
                    'enhanced_regularization',
                    'early_stopping',
                    'overfitting_monitoring',
                    'ensemble_diversity_monitoring'
                ],
                'config': self.config.__dict__
            }
            
            tprint_success("✅ Enhanced Analyst models training completed")
            return results
            
        except Exception as e:
            tprint_error(f"❌ Enhanced Analyst training failed: {e}")
            results['success'] = False
            results['error'] = str(e)
            return results


class TacticianTrainingIntegration:
    """
    Example integration for Tactician models training with enhanced utilities.
    """
    
    def __init__(self, config: Optional[TrainingIntegrationConfig] = None):
        """Initialize Tactician training integration."""
        self.config = config or TrainingIntegrationConfig(
            enable_early_stopping=True,
            enable_purged_cv=True,
            enable_lookahead_detection=True,
            enable_temporal_splits=True,
            enable_regularization=True,
            enable_overfitting_monitoring=True,
            enable_walk_forward=True,  # Enable for Tactician
            model_type='auto'
        )
        
        self.enhancer = TrainingStepEnhancer(self.config)
        tprint_success("✅ Tactician Training Integration initialized")
    
    def train_tactician_models_enhanced(self, 
                                       X: np.ndarray, 
                                       y: np.ndarray, 
                                       models: Dict[str, Any],
                                       timestamps: Optional[np.ndarray] = None,
                                       regime_labels: Optional[np.ndarray] = None,
                                       analyst_green_light_periods: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train Tactician models with enhanced overfitting prevention and lookahead bias detection.
        
        Args:
            X: Feature matrix
            y: Target array
            models: Dictionary of models to train
            timestamps: Timestamp array (optional)
            regime_labels: Regime labels (optional)
            analyst_green_light_periods: Analyst green light periods (optional)
            
        Returns:
            Training results with enhanced metadata
        """
        tprint_info("🚀 Starting enhanced Tactician models training...")
        
        results = {
            'models': {},
            'training_metadata': {},
            'overfitting_warnings': [],
            'ensemble_diversity': None,
            'walk_forward_validation': None,
            'success': True
        }
        
        try:
            # Filter for Analyst green light periods if provided
            if analyst_green_light_periods is not None:
                tprint_info("🔍 Filtering for Analyst green light periods...")
                green_light_mask = analyst_green_light_periods
                X_filtered = X[green_light_mask]
                y_filtered = y[green_light_mask]
                timestamps_filtered = timestamps[green_light_mask] if timestamps is not None else None
                
                tprint_info(f"📊 Filtered to {len(X_filtered)} samples ({np.mean(green_light_mask):.2%} green light ratio)")
            else:
                X_filtered, y_filtered, timestamps_filtered = X, y, timestamps
                tprint_warning("⚠️ No Analyst green light periods provided, using all data")
            
            # Validate temporal data
            if self.config.enable_lookahead_detection:
                tprint_info("🔍 Validating temporal data for lookahead bias...")
                is_valid, warnings = self.enhancer.enhanced_utils.validate_temporal_data(
                    X_filtered, y_filtered, timestamps_filtered, strict_mode=True
                )
                results['overfitting_warnings'].extend(warnings)
                
                if not is_valid:
                    tprint_error("❌ Temporal data validation failed")
                    results['success'] = False
                    return results
            
            # Train each model with enhancements
            trained_models = []
            for model_name, model in models.items():
                tprint_info(f"🚀 Training Tactician model: {model_name}")
                
                trained_model, model_metadata = self.enhancer.enhance_training_step(
                    X_filtered, y_filtered, model, timestamps_filtered, f"tactician_{model_name}"
                )
                
                results['models'][model_name] = {
                    'model': trained_model,
                    'metadata': model_metadata
                }
                trained_models.append(trained_model)
                
                if model_metadata.get('overfitting_detected', False):
                    results['overfitting_warnings'].append(f"Overfitting detected in {model_name}")
            
            # Calculate ensemble diversity if multiple models
            if len(trained_models) > 1 and self.config.enable_ensemble_diversity:
                tprint_info("📊 Calculating Tactician ensemble diversity...")
                diversity_metrics = self.enhancer.enhanced_utils.calculate_ensemble_diversity(
                    trained_models, X_filtered, y_filtered
                )
                results['ensemble_diversity'] = diversity_metrics
                
                if diversity_metrics.get('diversity_score', 0) < 0.1:
                    tprint_warning("⚠️ Low Tactician ensemble diversity detected")
                else:
                    tprint_success("✅ Good Tactician ensemble diversity")
            
            # Perform walk-forward validation if enabled
            if self.config.enable_walk_forward and len(trained_models) > 0:
                tprint_info("🚶 Performing walk-forward validation...")
                wfv_results = self.enhancer.enhanced_utils.perform_walk_forward_validation(
                    trained_models[0], X_filtered, y_filtered,
                    initial_train_size=1000, test_size=100, step_size=50
                )
                results['walk_forward_validation'] = wfv_results
                
                if wfv_results.get('performance_trend', {}).get('trend') == 'declining':
                    tprint_warning("⚠️ Declining performance trend detected in walk-forward validation")
                else:
                    tprint_success("✅ Stable performance trend in walk-forward validation")
            
            # Add comprehensive metadata
            results['training_metadata'] = {
                'total_models': len(models),
                'green_light_filtering': analyst_green_light_periods is not None,
                'filtered_samples': len(X_filtered) if analyst_green_light_periods is not None else len(X),
                'enhancements_applied': [
                    'lookahead_bias_detection',
                    'enhanced_regularization',
                    'early_stopping',
                    'overfitting_monitoring',
                    'ensemble_diversity_monitoring',
                    'walk_forward_validation'
                ],
                'config': self.config.__dict__
            }
            
            tprint_success("✅ Enhanced Tactician models training completed")
            return results
            
        except Exception as e:
            tprint_error(f"❌ Enhanced Tactician training failed: {e}")
            results['success'] = False
            results['error'] = str(e)
            return results


class EnsembleTrainingIntegration:
    """
    Example integration for ensemble training with enhanced utilities.
    """
    
    def __init__(self, config: Optional[TrainingIntegrationConfig] = None):
        """Initialize ensemble training integration."""
        self.config = config or TrainingIntegrationConfig(
            enable_early_stopping=True,
            enable_purged_cv=True,
            enable_lookahead_detection=True,
            enable_temporal_splits=True,
            enable_regularization=True,
            enable_overfitting_monitoring=True,
            enable_ensemble_diversity=True,
            model_type='auto'
        )
        
        self.enhancer = TrainingStepEnhancer(self.config)
        tprint_success("✅ Ensemble Training Integration initialized")
    
    def train_ensemble_enhanced(self, 
                               X: np.ndarray, 
                               y: np.ndarray, 
                               base_models: Dict[str, Any],
                               meta_model: Any,
                               timestamps: Optional[np.ndarray] = None,
                               regime_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train ensemble with enhanced overfitting prevention and diversity monitoring.
        
        Args:
            X: Feature matrix
            y: Target array
            base_models: Dictionary of base models
            meta_model: Meta-learner model
            timestamps: Timestamp array (optional)
            regime_labels: Regime labels (optional)
            
        Returns:
            Training results with enhanced metadata
        """
        tprint_info("🚀 Starting enhanced ensemble training...")
        
        results = {
            'base_models': {},
            'meta_model': None,
            'training_metadata': {},
            'overfitting_warnings': [],
            'ensemble_diversity': None,
            'success': True
        }
        
        try:
            # Validate temporal data
            if self.config.enable_lookahead_detection:
                tprint_info("🔍 Validating temporal data for lookahead bias...")
                is_valid, warnings = self.enhancer.enhanced_utils.validate_temporal_data(
                    X, y, timestamps, strict_mode=True
                )
                results['overfitting_warnings'].extend(warnings)
                
                if not is_valid:
                    tprint_error("❌ Temporal data validation failed")
                    results['success'] = False
                    return results
            
            # Train base models with enhancements
            trained_base_models = []
            for model_name, model in base_models.items():
                tprint_info(f"🚀 Training base model: {model_name}")
                
                trained_model, model_metadata = self.enhancer.enhance_training_step(
                    X, y, model, timestamps, f"base_{model_name}"
                )
                
                results['base_models'][model_name] = {
                    'model': trained_model,
                    'metadata': model_metadata
                }
                trained_base_models.append(trained_model)
                
                if model_metadata.get('overfitting_detected', False):
                    results['overfitting_warnings'].append(f"Overfitting detected in base model {model_name}")
            
            # Calculate base model diversity
            if len(trained_base_models) > 1:
                tprint_info("📊 Calculating base model diversity...")
                diversity_metrics = self.enhancer.enhanced_utils.calculate_ensemble_diversity(
                    trained_base_models, X, y
                )
                results['ensemble_diversity'] = diversity_metrics
                
                if diversity_metrics.get('diversity_score', 0) < 0.1:
                    tprint_warning("⚠️ Low base model diversity detected")
                else:
                    tprint_success("✅ Good base model diversity")
            
            # Train meta-model with enhanced utilities
            tprint_info("🚀 Training meta-model...")
            meta_model_trained, meta_metadata = self.enhancer.enhance_training_step(
                X, y, meta_model, timestamps, "meta_model"
            )
            
            results['meta_model'] = {
                'model': meta_model_trained,
                'metadata': meta_metadata
            }
            
            if meta_metadata.get('overfitting_detected', False):
                results['overfitting_warnings'].append("Overfitting detected in meta-model")
            
            # Add comprehensive metadata
            results['training_metadata'] = {
                'base_models_count': len(base_models),
                'enhancements_applied': [
                    'lookahead_bias_detection',
                    'enhanced_regularization',
                    'early_stopping',
                    'overfitting_monitoring',
                    'ensemble_diversity_monitoring'
                ],
                'config': self.config.__dict__
            }
            
            tprint_success("✅ Enhanced ensemble training completed")
            return results
            
        except Exception as e:
            tprint_error(f"❌ Enhanced ensemble training failed: {e}")
            results['success'] = False
            results['error'] = str(e)
            return results


# Example usage functions
def example_analyst_integration():
    """Example of how to integrate enhanced training into Analyst models."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import ElasticNet
    from sklearn.ensemble import GradientBoostingRegressor
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(1000, 20)
    y = np.random.randn(1000)
    timestamps = pd.date_range('2023-01-01', periods=1000, freq='1H')
    
    # Create models
    models = {
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'elastic_net': ElasticNet(alpha=0.1, random_state=42),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    # Initialize enhanced training
    analyst_integration = AnalystTrainingIntegration()
    
    # Train with enhancements
    results = analyst_integration.train_analyst_models_enhanced(
        X, y, models, timestamps
    )
    
    print("Analyst Training Results:")
    print(f"Success: {results['success']}")
    print(f"Models trained: {len(results['models'])}")
    print(f"Overfitting warnings: {len(results['overfitting_warnings'])}")
    if results['ensemble_diversity']:
        print(f"Ensemble diversity score: {results['ensemble_diversity']['diversity_score']:.3f}")


def example_tactician_integration():
    """Example of how to integrate enhanced training into Tactician models."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import ElasticNet
    from sklearn.ensemble import GradientBoostingRegressor
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(1000, 20)
    y = np.random.randn(1000)
    timestamps = pd.date_range('2023-01-01', periods=1000, freq='1H')
    analyst_green_light = np.random.choice([True, False], 1000, p=[0.3, 0.7])
    
    # Create models
    models = {
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'elastic_net': ElasticNet(alpha=0.1, random_state=42),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    # Initialize enhanced training
    tactician_integration = TacticianTrainingIntegration()
    
    # Train with enhancements
    results = tactician_integration.train_tactician_models_enhanced(
        X, y, models, timestamps, analyst_green_light_periods=analyst_green_light
    )
    
    print("Tactician Training Results:")
    print(f"Success: {results['success']}")
    print(f"Models trained: {len(results['models'])}")
    print(f"Overfitting warnings: {len(results['overfitting_warnings'])}")
    if results['ensemble_diversity']:
        print(f"Ensemble diversity score: {results['ensemble_diversity']['diversity_score']:.3f}")
    if results['walk_forward_validation']:
        print(f"Walk-forward validation trend: {results['walk_forward_validation']['performance_trend']['trend']}")


def example_ensemble_integration():
    """Example of how to integrate enhanced training into ensemble training."""
    from sklearn.ensemble import RandomForestRegressor, VotingRegressor
    from sklearn.linear_model import ElasticNet, Ridge
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(1000, 20)
    y = np.random.randn(1000)
    timestamps = pd.date_range('2023-01-01', periods=1000, freq='1H')
    
    # Create base models
    base_models = {
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'elastic_net': ElasticNet(alpha=0.1, random_state=42),
        'ridge': Ridge(alpha=1.0, random_state=42)
    }
    
    # Create meta-model
    meta_model = VotingRegressor([
        ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
        ('en', ElasticNet(alpha=0.1, random_state=42))
    ])
    
    # Initialize enhanced training
    ensemble_integration = EnsembleTrainingIntegration()
    
    # Train with enhancements
    results = ensemble_integration.train_ensemble_enhanced(
        X, y, base_models, meta_model, timestamps
    )
    
    print("Ensemble Training Results:")
    print(f"Success: {results['success']}")
    print(f"Base models trained: {len(results['base_models'])}")
    print(f"Meta model trained: {results['meta_model'] is not None}")
    print(f"Overfitting warnings: {len(results['overfitting_warnings'])}")
    if results['ensemble_diversity']:
        print(f"Ensemble diversity score: {results['ensemble_diversity']['diversity_score']:.3f}")


if __name__ == "__main__":
    print("Enhanced Training Integration Examples")
    print("=" * 50)
    
    print("\n1. Analyst Integration Example:")
    example_analyst_integration()
    
    print("\n2. Tactician Integration Example:")
    example_tactician_integration()
    
    print("\n3. Ensemble Integration Example:")
    example_ensemble_integration()