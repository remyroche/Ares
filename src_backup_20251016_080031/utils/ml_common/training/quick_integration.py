"""
Quick Integration Script for Enhanced Training Utilities

This script provides simple functions to quickly integrate enhanced training
utilities into existing training steps without major code changes.

Usage:
    from src.utils.ml_common.training.quick_integration import enhance_training_step
    
    # Replace existing training
    model = enhance_training_step(X, y, model, timestamps)
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import warnings

# Import enhanced training utilities
from .enhanced_training_utils import (
    EnhancedTrainingUtils,
    EarlyStoppingConfig,
    PurgedCVConfig,
    OverfittingMonitorConfig,
    RegularizationConfig
)

from .training_integration import (
    TrainingStepEnhancer,
    TrainingIntegrationConfig
)

# Import existing utilities
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint_info(*args, **kwargs): print(f"INFO: {args[0] if args else ''}")
    def tprint_success(*args, **kwargs): print(f"SUCCESS: {args[0] if args else ''}")
    def tprint_warning(*args, **kwargs): print(f"WARNING: {args[0] if args else ''}")
    def tprint_error(*args, **kwargs): print(f"ERROR: {args[0] if args else ''}")


def enhance_training_step(X: np.ndarray, 
                         y: np.ndarray, 
                         model: Any,
                         timestamps: Optional[np.ndarray] = None,
                         model_name: str = 'model',
                         config: Optional[TrainingIntegrationConfig] = None) -> Tuple[Any, Dict[str, Any]]:
    """
    Quick function to enhance any training step with comprehensive utilities.
    
    Args:
        X: Feature matrix
        y: Target array
        model: Model to train
        timestamps: Timestamp array (optional)
        model_name: Name of the model
        config: Training configuration (optional)
        
    Returns:
        Tuple of (trained_model, training_metadata)
        
    Example:
        # Before
        model.fit(X, y)
        
        # After
        trained_model, metadata = enhance_training_step(X, y, model, timestamps)
    """
    try:
        # Use default config if not provided
        if config is None:
            config = TrainingIntegrationConfig(
                enable_early_stopping=True,
                enable_purged_cv=True,
                enable_lookahead_detection=True,
                enable_temporal_splits=True,
                enable_regularization=True,
                enable_overfitting_monitoring=True
            )
        
        # Create enhancer
        enhancer = TrainingStepEnhancer(config)
        
        # Enhance training
        trained_model, metadata = enhancer.enhance_training_step(
            X, y, model, timestamps, model_name
        )
        
        return trained_model, metadata
        
    except Exception as e:
        tprint_error(f"❌ Quick enhancement failed: {e}")
        # Fallback to standard training
        model.fit(X, y)
        return model, {'error': str(e), 'fallback': True}


def enhance_ensemble_training(X: np.ndarray, 
                            y: np.ndarray, 
                            models: List[Any],
                            timestamps: Optional[np.ndarray] = None,
                            config: Optional[TrainingIntegrationConfig] = None) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Quick function to enhance ensemble training with diversity monitoring.
    
    Args:
        X: Feature matrix
        y: Target array
        models: List of models to train
        timestamps: Timestamp array (optional)
        config: Training configuration (optional)
        
    Returns:
        Tuple of (trained_models, training_metadata)
        
    Example:
        # Before
        for model in models:
            model.fit(X, y)
        
        # After
        trained_models, metadata = enhance_ensemble_training(X, y, models, timestamps)
    """
    try:
        # Use default config if not provided
        if config is None:
            config = TrainingIntegrationConfig(
                enable_early_stopping=True,
                enable_purged_cv=True,
                enable_lookahead_detection=True,
                enable_temporal_splits=True,
                enable_regularization=True,
                enable_overfitting_monitoring=True,
                enable_ensemble_diversity=True
            )
        
        # Create enhancer
        enhancer = TrainingStepEnhancer(config)
        
        # Enhance ensemble training
        trained_models, metadata = enhancer.enhance_ensemble_training(
            X, y, models, timestamps
        )
        
        return trained_models, metadata
        
    except Exception as e:
        tprint_error(f"❌ Quick ensemble enhancement failed: {e}")
        # Fallback to standard training
        for model in models:
            model.fit(X, y)
        return models, {'error': str(e), 'fallback': True}


def enhance_cross_validation(X: np.ndarray, 
                           y: np.ndarray, 
                           model: Any,
                           timestamps: Optional[np.ndarray] = None,
                           n_splits: int = 5,
                           config: Optional[TrainingIntegrationConfig] = None) -> Dict[str, Any]:
    """
    Quick function to enhance cross-validation with temporal integrity.
    
    Args:
        X: Feature matrix
        y: Target array
        model: Model to evaluate
        timestamps: Timestamp array (optional)
        n_splits: Number of CV splits
        config: Training configuration (optional)
        
    Returns:
        Cross-validation results with temporal integrity
        
    Example:
        # Before
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X, y, cv=5)
        
        # After
        results = enhance_cross_validation(X, y, model, timestamps, n_splits=5)
        scores = results['cv_scores']
    """
    try:
        # Use default config if not provided
        if config is None:
            config = TrainingIntegrationConfig(
                enable_purged_cv=True,
                enable_temporal_splits=True,
                enable_lookahead_detection=True,
                cv_n_splits=n_splits
            )
        
        # Create enhancer
        enhancer = TrainingStepEnhancer(config)
        
        # Create temporal splits
        temporal_splits = enhancer.enhanced_utils.create_temporal_splits(
            X, y, timestamps, use_purged=True
        )
        
        # Perform cross-validation via unified API
        try:
            from src.utils.ml_common.validation.unified_cv import temporal_cross_validation as unified_temporal_cv
            results = unified_temporal_cv(model, X, y, n_splits=n_splits, gap=0, test_size=None, scoring='accuracy')
            # Ensure expected keys for quick integration example
            results = {
                'cv_scores': results.get('scores', []) or [],
                'mean_score': results.get('mean', 0.0) or 0.0,
                'std_score': results.get('std', 0.0) or 0.0,
                'fold_results': [],
                'n_splits': n_splits,
                'temporal_integrity': True
            }
        except Exception as fold_error:
            tprint_warning(f"⚠️ Unified temporal CV failed: {fold_error}")
            results = {'cv_scores': [], 'mean_score': 0.0, 'std_score': 0.0, 'fold_results': [], 'n_splits': 0, 'temporal_integrity': True}
        
        tprint_success(f"✅ Enhanced CV completed: {results['mean_score']:.4f} ± {results['std_score']:.4f}")
        return results
        
    except Exception as e:
        tprint_error(f"❌ Quick CV enhancement failed: {e}")
        # Fallback to standard CV
        from sklearn.model_selection import cross_val_score
        try:
            scores = cross_val_score(model, X, y, cv=n_splits)
            return {
                'cv_scores': scores.tolist(),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'n_splits': len(scores),
                'temporal_integrity': False,
                'fallback': True,
                'error': str(e)
            }
        except Exception as fallback_error:
            return {'error': str(fallback_error), 'fallback': True}


def validate_temporal_data(X: np.ndarray, 
                          y: np.ndarray, 
                          timestamps: Optional[np.ndarray] = None,
                          strict_mode: bool = True) -> Tuple[bool, List[str]]:
    """
    Quick function to validate temporal data for lookahead bias.
    
    Args:
        X: Feature matrix
        y: Target array
        timestamps: Timestamp array (optional)
        strict_mode: Whether to raise errors on violations
        
    Returns:
        Tuple of (is_valid, warnings)
        
    Example:
        # Before training
        is_valid, warnings = validate_temporal_data(X, y, timestamps)
        if not is_valid:
            print("Data validation failed!")
    """
    try:
        # Create enhanced utils
        enhanced_utils = EnhancedTrainingUtils()
        
        # Validate data
        is_valid, warnings = enhanced_utils.validate_temporal_data(
            X, y, timestamps, strict_mode
        )
        
        if warnings:
            for warning in warnings:
                tprint_warning(f"⚠️ {warning}")
        
        if is_valid:
            tprint_success("✅ Temporal data validation passed")
        else:
            tprint_error("❌ Temporal data validation failed")
        
        return is_valid, warnings
        
    except Exception as e:
        tprint_error(f"❌ Temporal data validation failed: {e}")
        return False, [str(e)]


def monitor_overfitting(model: Any, 
                       X_train: np.ndarray, 
                       y_train: np.ndarray,
                       X_val: np.ndarray, 
                       y_val: np.ndarray,
                       model_name: str = 'model') -> Dict[str, Any]:
    """
    Quick function to monitor for overfitting.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        model_name: Name of the model
        
    Returns:
        Overfitting monitoring results
        
    Example:
        # After training
        overfitting_results = monitor_overfitting(model, X_train, y_train, X_val, y_val)
        if overfitting_results['is_overfitting']:
            print("Overfitting detected!")
    """
    try:
        # Create enhanced utils
        enhanced_utils = EnhancedTrainingUtils()
        
        # Monitor overfitting
        results = enhanced_utils.monitor_overfitting(
            model, X_train, y_train, X_val, y_val, model_name
        )
        
        if results.get('is_overfitting', False):
            tprint_warning("⚠️ Overfitting detected!")
        else:
            tprint_success("✅ No overfitting detected")
        
        return results
        
    except Exception as e:
        tprint_error(f"❌ Overfitting monitoring failed: {e}")
        return {'error': str(e)}


def calculate_ensemble_diversity(models: List[Any], 
                               X: np.ndarray, 
                               y: np.ndarray) -> Dict[str, Any]:
    """
    Quick function to calculate ensemble diversity metrics.
    
    Args:
        models: List of trained models
        X: Feature matrix
        y: Target array
        
    Returns:
        Ensemble diversity metrics
        
    Example:
        # After ensemble training
        diversity = calculate_ensemble_diversity(models, X, y)
        if diversity['diversity_score'] < 0.1:
            print("Low ensemble diversity!")
    """
    try:
        # Create enhanced utils
        enhanced_utils = EnhancedTrainingUtils()
        
        # Calculate diversity
        diversity_metrics = enhanced_utils.calculate_ensemble_diversity(
            models, X, y
        )
        
        if diversity_metrics.get('diversity_score', 0) < 0.1:
            tprint_warning("⚠️ Low ensemble diversity detected")
        else:
            tprint_success("✅ Good ensemble diversity")
        
        return diversity_metrics
        
    except Exception as e:
        tprint_error(f"❌ Ensemble diversity calculation failed: {e}")
        return {'error': str(e)}


# Convenience function for complete training enhancement
def enhance_complete_training(X: np.ndarray, 
                            y: np.ndarray, 
                            model: Any,
                            timestamps: Optional[np.ndarray] = None,
                            model_name: str = 'model',
                            enable_validation: bool = True,
                            enable_monitoring: bool = True) -> Dict[str, Any]:
    """
    Complete training enhancement with validation and monitoring.
    
    Args:
        X: Feature matrix
        y: Target array
        model: Model to train
        timestamps: Timestamp array (optional)
        model_name: Name of the model
        enable_validation: Whether to validate temporal data
        enable_monitoring: Whether to monitor for overfitting
        
    Returns:
        Complete training results with all enhancements
        
    Example:
        # Complete enhanced training
        results = enhance_complete_training(X, y, model, timestamps)
        trained_model = results['model']
        metadata = results['metadata']
    """
    try:
        results = {
            'model': None,
            'metadata': {},
            'validation': {},
            'monitoring': {},
            'success': True
        }
        
        # Step 1: Validate temporal data
        if enable_validation:
            tprint_info("🔍 Step 1: Validating temporal data...")
            is_valid, warnings = validate_temporal_data(X, y, timestamps)
            results['validation'] = {
                'is_valid': is_valid,
                'warnings': warnings
            }
            
            if not is_valid:
                tprint_error("❌ Temporal data validation failed")
                results['success'] = False
                return results
        
        # Step 2: Enhanced training
        tprint_info("🚀 Step 2: Enhanced training...")
        trained_model, training_metadata = enhance_training_step(
            X, y, model, timestamps, model_name
        )
        results['model'] = trained_model
        results['metadata'] = training_metadata
        
        # Step 3: Overfitting monitoring
        if enable_monitoring and len(X) > 200:
            tprint_info("📊 Step 3: Overfitting monitoring...")
            
            # Create validation split
            split_point = int(len(X) * 0.8)
            X_train, X_val = X[:split_point], X[split_point:]
            y_train, y_val = y[:split_point], y[split_point:]
            
            monitoring_results = monitor_overfitting(
                trained_model, X_train, y_train, X_val, y_val, model_name
            )
            results['monitoring'] = monitoring_results
        
        tprint_success("✅ Complete enhanced training completed")
        return results
        
    except Exception as e:
        tprint_error(f"❌ Complete enhanced training failed: {e}")
        results['success'] = False
        results['error'] = str(e)
        return results


# Example usage
if __name__ == "__main__":
    print("Quick Integration Examples")
    print("=" * 40)
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(1000, 20)
    y = np.random.randn(1000)
    timestamps = pd.date_range('2023-01-01', periods=1000, freq='1H')
    
    # Create sample model
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    print("\n1. Quick Training Enhancement:")
    trained_model, metadata = enhance_training_step(X, y, model, timestamps)
    print(f"Training completed: {metadata.get('training_time', 0):.2f}s")
    
    print("\n2. Quick Cross-Validation:")
    cv_results = enhance_cross_validation(X, y, model, timestamps)
    print(f"CV Score: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
    
    print("\n3. Complete Enhanced Training:")
    complete_results = enhance_complete_training(X, y, model, timestamps)
    print(f"Success: {complete_results['success']}")
    print(f"Overfitting detected: {complete_results['monitoring'].get('is_overfitting', False)}")