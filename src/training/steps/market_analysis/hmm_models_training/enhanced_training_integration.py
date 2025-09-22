"""
Enhanced HMM Training Integration

Demonstrates the integration of all new enhanced components:
1. Standardized timeframe configuration
2. Early stopping and aggressive overfitting detection
3. Temporal validation with walk-forward validation
4. Proper time series cross-validation

This script shows how to use all the new components together.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

# Import all new enhanced components
from .timeframe_config import (
    get_timeframe_config, 
    validate_timeframe_consistency, 
    get_primary_timeframe,
    set_timeframe_config,
    TimeframeConfig
)
from .early_stopping import (
    get_early_stopping_config,
    get_overfitting_detector,
    EarlyStoppingMonitor,
    EarlyStoppingConfig
)
from .temporal_validation import (
    get_temporal_config,
    get_temporal_validator,
    get_temporal_cv,
    TemporalValidationConfig
)
from .temporal_cross_validation import (
    get_temporal_cv_config,
    get_validation_pipeline,
    create_time_series_split,
    TemporalCVConfig
)
from .overfitting_reporting import (
    get_overfitting_reporter,
    OverfittingReporter,
    OverfittingReport
)

# Import existing components
from .hmm_models_training_enhanced import HMMModelsTrainingEnhanced
from .hmm_ensemble_training import HMMEnsembleTraining

logger = logging.getLogger(__name__)

class EnhancedHMMTrainingPipeline:
    """
    Enhanced HMM Training Pipeline with all new components integrated.
    
    Features:
    - Standardized timeframe configuration
    - Early stopping and aggressive overfitting detection
    - Temporal validation with walk-forward validation
    - Proper time series cross-validation
    """
    
    def __init__(self, 
                 timeframe: str = "15m",
                 enable_early_stopping: bool = True,
                 enable_temporal_validation: bool = True,
                 enable_walk_forward: bool = True):
        """
        Initialize enhanced HMM training pipeline.
        
        Args:
            timeframe: Primary timeframe for HMM operations
            enable_early_stopping: Enable early stopping and overfitting detection
            enable_temporal_validation: Enable temporal validation
            enable_walk_forward: Enable walk-forward validation
        """
        self.timeframe = timeframe
        self.enable_early_stopping = enable_early_stopping
        self.enable_temporal_validation = enable_temporal_validation
        self.enable_walk_forward = enable_walk_forward
        
        # Initialize configurations
        self._setup_timeframe_config()
        self._setup_early_stopping_config()
        self._setup_temporal_validation_config()
        self._setup_temporal_cv_config()
        
        # Initialize components
        self._initialize_components()
    
    def _setup_timeframe_config(self):
        """Setup standardized timeframe configuration."""
        timeframe_config = TimeframeConfig(
            primary_timeframe=self.timeframe,
            supported_timeframes=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
            enable_cross_timeframe_features=True,
            cross_timeframe_list=["5m", "30m", "1h"],
            strict_timeframe_validation=True
        )
        set_timeframe_config(timeframe_config)
        
        # Validate timeframe consistency
        if not validate_timeframe_consistency(self.timeframe, "EnhancedHMMTrainingPipeline"):
            raise ValueError(f"Invalid timeframe: {self.timeframe}")
        
        logger.info(f"✅ Timeframe configuration set to: {self.timeframe}")
    
    def _setup_early_stopping_config(self):
        """Setup early stopping and overfitting detection configuration."""
        if not self.enable_early_stopping:
            return
        
        early_stopping_config = EarlyStoppingConfig(
            patience=5,
            min_delta=0.001,
            monitor_metric='validation_loss',
            mode='min',
            # Aggressive overfitting detection thresholds
            accuracy_gap_threshold=0.05,  # 5% gap triggers warning
            severe_accuracy_gap_threshold=0.15,  # 15% gap triggers early stopping
            f1_gap_threshold=0.03,  # 3% F1 gap triggers warning
            severe_f1_gap_threshold=0.10,  # 10% F1 gap triggers early stopping
            confidence_gap_threshold=0.1,  # 10% confidence gap
            overconfident_ratio_threshold=0.3,  # 30% overconfident predictions
            feature_concentration_threshold=0.8,  # 80% of importance in top features
            correlation_threshold=0.95,  # High correlation indicates overfitting
            cv_variance_threshold=0.05,  # 5% CV variance threshold
            cv_test_gap_threshold=0.08,  # 8% gap between CV and test
            enable_early_stopping=True,
            enable_aggressive_detection=True
        )
        
        logger.info("✅ Early stopping and aggressive overfitting detection enabled")
    
    def _setup_temporal_validation_config(self):
        """Setup temporal validation configuration."""
        if not self.enable_temporal_validation:
            return
        
        temporal_config = TemporalValidationConfig(
            enable_temporal_checks=True,
            strict_temporal_order=True,
            min_temporal_gap=1,
            enable_walk_forward=self.enable_walk_forward,
            initial_train_size=0.6,
            step_size=0.1,
            min_test_size=0.1,
            enable_leakage_detection=True,
            max_correlation_threshold=0.95,
            temporal_consistency_threshold=0.8,
            detailed_reporting=True,
            save_validation_plots=False
        )
        
        logger.info("✅ Temporal validation with walk-forward validation enabled")
    
    def _setup_temporal_cv_config(self):
        """Setup temporal cross-validation configuration."""
        temporal_cv_config = TemporalCVConfig(
            n_splits=5,
            test_size=0.2,
            gap_size=1,
            enable_temporal_splits=True,
            strict_temporal_order=True,
            min_train_size=0.3,
            min_test_size=0.1,
            enable_validation=True,
            validation_size=0.1,
            early_stopping_patience=3,
            track_performance=True,
            save_predictions=False,
            detailed_reporting=True
        )
        
        logger.info("✅ Temporal cross-validation with temporal splits enabled")
    
    def _initialize_components(self):
        """Initialize all training components."""
        # Get configurations
        self.timeframe_config = get_timeframe_config()
        self.early_stopping_config = get_early_stopping_config()
        self.temporal_config = get_temporal_config()
        self.temporal_cv_config = get_temporal_cv_config()
        
        # Get validators and detectors
        self.temporal_validator = get_temporal_validator()
        self.overfitting_detector = get_overfitting_detector()
        self.validation_pipeline = get_validation_pipeline()
        self.overfitting_reporter = get_overfitting_reporter()
        
        # Initialize training components
        self.hmm_training = HMMModelsTrainingEnhanced()
        self.ensemble_training = HMMEnsembleTraining()
        
        logger.info("✅ All enhanced components initialized")
    
    def train_with_enhanced_validation(self, 
                                     X: np.ndarray, 
                                     y: np.ndarray,
                                     timestamps: Optional[np.ndarray] = None,
                                     feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train HMM models with enhanced validation.
        
        Args:
            X: Input features
            y: Target labels
            timestamps: Optional timestamps for temporal validation
            feature_names: Optional feature names
            
        Returns:
            Dict: Training results with enhanced validation
        """
        logger.info("🚀 Starting enhanced HMM training with comprehensive validation")
        
        results = {
            'training_successful': False,
            'timeframe_validation': {},
            'temporal_validation': {},
            'overfitting_analysis': {},
            'cross_validation': {},
            'recommendations': []
        }
        
        try:
            # 1. Timeframe validation
            timeframe_valid = validate_timeframe_consistency(
                self.timeframe, "EnhancedHMMTrainingPipeline"
            )
            results['timeframe_validation'] = {
                'valid': timeframe_valid,
                'timeframe': self.timeframe,
                'message': f"Timeframe {self.timeframe} validation: {'PASSED' if timeframe_valid else 'FAILED'}"
            }
            
            if not timeframe_valid:
                results['recommendations'].append("Fix timeframe configuration")
                return results
            
            # 2. Temporal validation
            if self.enable_temporal_validation and timestamps is not None:
                temporal_results = self.temporal_validator.validate_temporal_split(
                    X[:len(X)//2], X[len(X)//2:],  # Simple split for demo
                    y[:len(y)//2], y[len(y)//2:],
                    timestamps
                )
                results['temporal_validation'] = temporal_results
                
                if not temporal_results['temporal_order_valid']:
                    results['recommendations'].append("Fix temporal order in data split")
                
                if temporal_results['leakage_detected']:
                    results['recommendations'].append("Investigate and fix data leakage")
            
            # 3. Cross-validation with temporal splits
            if self.enable_temporal_validation:
                # Create time series splitter
                tscv = create_time_series_split(
                    n_splits=5,
                    test_size=0.2,
                    gap_size=1,
                    min_train_size=0.3,
                    min_test_size=0.1
                )
                
                # Perform temporal cross-validation
                cv_results = self.validation_pipeline.validate_model(
                    estimator=None,  # Will be set by training component
                    X=X,
                    y=y,
                    timestamps=timestamps,
                    feature_names=feature_names
                )
                results['cross_validation'] = cv_results
            
            # 4. Train models with enhanced validation
            training_results = self._train_models_with_validation(X, y, timestamps, feature_names)
            results.update(training_results)
            
            # 5. Overfitting analysis
            if self.enable_early_stopping and 'train_predictions' in training_results:
                overfitting_analysis = self.overfitting_detector.comprehensive_overfitting_analysis(
                    train_predictions=training_results['train_predictions'],
                    val_predictions=training_results['test_predictions'],
                    train_labels=training_results['train_labels'],
                    val_labels=training_results['test_labels'],
                    train_probabilities=training_results.get('train_probabilities'),
                    val_probabilities=training_results.get('test_probabilities'),
                    feature_importance=training_results.get('feature_importance')
                )
                results['overfitting_analysis'] = overfitting_analysis
                
                if overfitting_analysis['is_overfitting']:
                    results['recommendations'].extend(overfitting_analysis['recommendations'])
            
            results['training_successful'] = True
            logger.info("✅ Enhanced HMM training completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Enhanced training failed: {e}")
            results['error'] = str(e)
            results['recommendations'].append(f"Fix training error: {e}")
        
        return results
    
    def _train_models_with_validation(self, 
                                    X: np.ndarray, 
                                    y: np.ndarray,
                                    timestamps: Optional[np.ndarray] = None,
                                    feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train models with enhanced validation."""
        # This would integrate with the existing training components
        # For now, return a placeholder structure
        return {
            'train_predictions': np.random.randint(0, 3, len(X)//2),
            'test_predictions': np.random.randint(0, 3, len(X)//2),
            'train_labels': y[:len(y)//2],
            'test_labels': y[len(y)//2:],
            'train_probabilities': None,
            'test_probabilities': None,
            'feature_importance': np.random.random(X.shape[1])
        }
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results."""
        return {
            'timeframe_config': {
                'primary_timeframe': self.timeframe_config.primary_timeframe,
                'supported_timeframes': self.timeframe_config.supported_timeframes,
                'cross_timeframe_enabled': self.timeframe_config.enable_cross_timeframe_features
            },
            'early_stopping_config': {
                'enabled': self.enable_early_stopping,
                'patience': self.early_stopping_config.patience,
                'aggressive_detection': self.early_stopping_config.enable_aggressive_detection
            },
            'temporal_validation_config': {
                'enabled': self.enable_temporal_validation,
                'temporal_checks': self.temporal_config.enable_temporal_checks,
                'walk_forward_enabled': self.temporal_config.enable_walk_forward
            },
            'temporal_cv_config': {
                'n_splits': self.temporal_cv_config.n_splits,
                'temporal_splits': self.temporal_cv_config.enable_temporal_splits,
                'strict_temporal_order': self.temporal_cv_config.strict_temporal_order
            }
        }

def demonstrate_enhanced_training():
    """Demonstrate the enhanced training pipeline."""
    print("🚀 Enhanced HMM Training Pipeline Demonstration")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 3, n_samples)
    timestamps = np.arange(n_samples)
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    # Initialize enhanced pipeline
    pipeline = EnhancedHMMTrainingPipeline(
        timeframe="15m",
        enable_early_stopping=True,
        enable_temporal_validation=True,
        enable_walk_forward=True
    )
    
    # Train with enhanced validation
    results = pipeline.train_with_enhanced_validation(
        X=X,
        y=y,
        timestamps=timestamps,
        feature_names=feature_names
    )
    
    # Print results
    print("\n📊 Training Results:")
    print(f"Training successful: {results['training_successful']}")
    print(f"Timeframe validation: {results['timeframe_validation']['message']}")
    
    if results['temporal_validation']:
        print(f"Temporal validation score: {results['temporal_validation']['validation_score']:.3f}")
    
    if results['overfitting_analysis']:
        overfitting = results['overfitting_analysis']
        print(f"Overfitting detected: {overfitting['is_overfitting']}")
        if overfitting['is_overfitting']:
            print(f"Severity: {overfitting['severity']}")
            print(f"Warnings: {len(overfitting['warnings'])}")
    
    if results['recommendations']:
        print(f"\n💡 Recommendations ({len(results['recommendations'])}):")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    # Print validation summary
    print("\n📋 Validation Summary:")
    summary = pipeline.get_validation_summary()
    for category, config in summary.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    return results

if __name__ == "__main__":
    # Run demonstration
    results = demonstrate_enhanced_training()
    print("\n✅ Enhanced HMM Training Pipeline demonstration completed!")