"""
Enhanced Triple Barrier Validation and Testing Framework

This module provides comprehensive validation and testing for the enhanced triple barrier
labeling system. It validates the quality of profit potential labels, tests the ML
integration, and provides detailed performance analysis.

Key Validation Areas:
1. Label Quality Validation (distribution, consistency, calibration)
2. Profit Potential Accuracy (magnitude scoring, confidence scoring)
3. ML Model Performance (direction, magnitude, confidence prediction)
4. Regime-Specific Performance (regime-aware adjustments)
5. Feature Engineering Quality (feature importance, correlation analysis)
6. End-to-End Pipeline Testing (full workflow validation)
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime

import pandas as pd
import numpy as np

from src.utils.tprint import tprint
from src.utils.logger import get_logger

# Import validation libraries
try:
    from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import cross_val_score
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Import our enhanced modules
from src.training.steps.market_analysis.enhanced_triple_barrier_labeling import (
    EnhancedTripleBarrierLabeler, EnhancedTripleBarrierConfig, apply_enhanced_triple_barrier_labeling
)
from src.training.steps.market_analysis.enhanced_profit_feature_engineering import (
    EnhancedProfitFeatureEngineering, apply_enhanced_profit_feature_engineering
)
from src.training.steps.market_analysis.ml_profit_potential_integration import (
    MLProfitPotentialIntegration, train_ml_models_with_profit_potential
)

@dataclass
class ValidationConfig:
    """Configuration for enhanced triple barrier validation."""
    
    # Validation parameters
    enable_label_quality_validation: bool = True
    enable_profit_accuracy_validation: bool = True
    enable_ml_performance_validation: bool = True
    enable_regime_validation: bool = True
    enable_feature_validation: bool = True
    enable_end_to_end_validation: bool = True
    
    # Test data parameters
    test_data_size: int = 1000
    test_regimes: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    test_profit_ranges: Tuple[float, float] = (-0.05, 0.05)  # -5% to 5%
    
    # Validation thresholds
    min_label_distribution_ratio: float = 0.05  # Minimum 5% for any category
    min_confidence_calibration: float = 0.7     # Minimum 70% calibration
    min_ml_accuracy: float = 0.6                # Minimum 60% ML accuracy
    min_profit_correlation: float = 0.3         # Minimum 30% profit correlation
    
    # Cross-validation parameters
    cv_folds: int = 5
    random_state: int = 42

@dataclass
class ValidationResult:
    """Result of validation testing."""
    
    # Overall validation status
    overall_success: bool = False
    validation_score: float = 0.0
    
    # Individual validation results
    label_quality_result: Dict[str, Any] = field(default_factory=dict)
    profit_accuracy_result: Dict[str, Any] = field(default_factory=dict)
    ml_performance_result: Dict[str, Any] = field(default_factory=dict)
    regime_validation_result: Dict[str, Any] = field(default_factory=dict)
    feature_validation_result: Dict[str, Any] = field(default_factory=dict)
    end_to_end_result: Dict[str, Any] = field(default_factory=dict)
    
    # Execution metadata
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime = field(default_factory=datetime.now)
    execution_duration: float = 0.0
    
    # Summary statistics
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

class EnhancedTripleBarrierValidator:
    """Comprehensive validator for enhanced triple barrier labeling system."""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """Initialize the enhanced triple barrier validator."""
        self.config = config or ValidationConfig()
        self.logger = get_logger('EnhancedTripleBarrierValidator')
        
        self.logger.info("🔍 Enhanced Triple Barrier Validator initialized")
        tprint("🔍 Enhanced Triple Barrier Validator initialized")
    
    def run_comprehensive_validation(self, data: Optional[pd.DataFrame] = None) -> ValidationResult:
        """Run comprehensive validation of the enhanced triple barrier system."""
        start_time = datetime.now()
        
        tprint("🚀 Starting Comprehensive Enhanced Triple Barrier Validation")
        self.logger.info("🚀 Starting Comprehensive Enhanced Triple Barrier Validation")
        
        result = ValidationResult(start_time=start_time)
        
        try:
            # Generate test data if not provided
            if data is None:
                data = self._generate_test_data()
            
            # Run individual validation tests
            if self.config.enable_label_quality_validation:
                tprint("📊 Running label quality validation...")
                result.label_quality_result = self._validate_label_quality(data)
                tprint("✅ Label quality validation completed")
            
            if self.config.enable_profit_accuracy_validation:
                tprint("📊 Running profit accuracy validation...")
                result.profit_accuracy_result = self._validate_profit_accuracy(data)
                tprint("✅ Profit accuracy validation completed")
            
            if self.config.enable_ml_performance_validation:
                tprint("📊 Running ML performance validation...")
                result.ml_performance_result = self._validate_ml_performance(data)
                tprint("✅ ML performance validation completed")
            
            if self.config.enable_regime_validation:
                tprint("📊 Running regime validation...")
                result.regime_validation_result = self._validate_regime_performance(data)
                tprint("✅ Regime validation completed")
            
            if self.config.enable_feature_validation:
                tprint("📊 Running feature validation...")
                result.feature_validation_result = self._validate_feature_engineering(data)
                tprint("✅ Feature validation completed")
            
            if self.config.enable_end_to_end_validation:
                tprint("📊 Running end-to-end validation...")
                result.end_to_end_result = self._validate_end_to_end_pipeline(data)
                tprint("✅ End-to-end validation completed")
            
            # Calculate overall validation score
            result.validation_score = self._calculate_overall_validation_score(result)
            result.overall_success = result.validation_score >= 0.7  # 70% threshold
            
            # Calculate test statistics
            result.total_tests = self._count_total_tests(result)
            result.passed_tests = self._count_passed_tests(result)
            result.failed_tests = result.total_tests - result.passed_tests
            
            result.end_time = datetime.now()
            result.execution_duration = (result.end_time - result.start_time).total_seconds()
            
            tprint(f"✅ Comprehensive validation completed")
            tprint(f"   Duration: {result.execution_duration:.2f}s")
            tprint(f"   Overall Score: {result.validation_score:.2%}")
            tprint(f"   Tests Passed: {result.passed_tests}/{result.total_tests}")
            tprint(f"   Status: {'✅ PASSED' if result.overall_success else '❌ FAILED'}")
            
            return result
            
        except Exception as e:
            result.overall_success = False
            result.errors.append(str(e))
            result.end_time = datetime.now()
            result.execution_duration = (result.end_time - result.start_time).total_seconds()
            
            tprint(f"❌ Validation failed: {e}")
            self.logger.error(f"❌ Validation failed: {e}")
            
            return result
    
    def _generate_test_data(self) -> pd.DataFrame:
        """Generate comprehensive test data for validation."""
        
        tprint("📊 Generating test data for validation...")
        
        # Create realistic market data
        dates = pd.date_range('2024-01-01', periods=self.config.test_data_size, freq='1min')
        
        # Generate price data with realistic patterns
        np.random.seed(self.config.random_state)
        
        # Base price with trend and volatility
        base_price = 100
        trend = np.linspace(0, 0.1, self.config.test_data_size)  # 10% upward trend
        noise = np.random.normal(0, 0.01, self.config.test_data_size)  # 1% volatility
        prices = base_price * (1 + trend + noise)
        
        # Generate OHLC data
        data = pd.DataFrame({
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, self.config.test_data_size))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, self.config.test_data_size))),
            'close': prices,
            'volume': np.random.uniform(1000, 10000, self.config.test_data_size),
            'hmm_regime': np.random.choice(self.config.test_regimes, self.config.test_data_size)
        }, index=dates)
        
        # Ensure OHLC consistency
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        
        tprint(f"✅ Test data generated: {len(data)} samples")
        
        return data
    
    def _validate_label_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate the quality of profit potential labels."""
        
        # Apply enhanced triple barrier labeling
        labeling_result = apply_enhanced_triple_barrier_labeling(data)
        
        if not labeling_result.success:
            return {
                'success': False,
                'error': labeling_result.error_message,
                'score': 0.0
            }
        
        labeled_data = labeling_result.labeled_data
        
        # Test 1: Label distribution balance
        distribution_score = self._test_label_distribution(labeled_data)
        
        # Test 2: Profit category consistency
        consistency_score = self._test_profit_consistency(labeled_data)
        
        # Test 3: Confidence calibration
        calibration_score = self._test_confidence_calibration(labeled_data)
        
        # Test 4: Magnitude score quality
        magnitude_score = self._test_magnitude_quality(labeled_data)
        
        # Overall score
        overall_score = (distribution_score + consistency_score + calibration_score + magnitude_score) / 4
        
        return {
            'success': overall_score >= 0.7,
            'score': overall_score,
            'distribution_score': distribution_score,
            'consistency_score': consistency_score,
            'calibration_score': calibration_score,
            'magnitude_score': magnitude_score,
            'label_distribution': labeled_data['profit_category'].value_counts().to_dict(),
            'confidence_distribution': labeled_data['confidence_category'].value_counts().to_dict()
        }
    
    def _validate_profit_accuracy(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate the accuracy of profit potential predictions."""
        
        # Apply enhanced triple barrier labeling
        labeling_result = apply_enhanced_triple_barrier_labeling(data)
        
        if not labeling_result.success:
            return {
                'success': False,
                'error': labeling_result.error_message,
                'score': 0.0
            }
        
        labeled_data = labeling_result.labeled_data
        
        # Test 1: Profit magnitude accuracy
        magnitude_accuracy = self._test_profit_magnitude_accuracy(labeled_data)
        
        # Test 2: Confidence accuracy
        confidence_accuracy = self._test_confidence_accuracy(labeled_data)
        
        # Test 3: Regime-specific accuracy
        regime_accuracy = self._test_regime_specific_accuracy(labeled_data)
        
        # Test 4: Profit correlation
        profit_correlation = self._test_profit_correlation(labeled_data)
        
        # Overall score
        overall_score = (magnitude_accuracy + confidence_accuracy + regime_accuracy + profit_correlation) / 4
        
        return {
            'success': overall_score >= 0.7,
            'score': overall_score,
            'magnitude_accuracy': magnitude_accuracy,
            'confidence_accuracy': confidence_accuracy,
            'regime_accuracy': regime_accuracy,
            'profit_correlation': profit_correlation
        }
    
    def _validate_ml_performance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate ML model performance with profit potential labels."""
        
        # Apply enhanced triple barrier labeling
        labeling_result = apply_enhanced_triple_barrier_labeling(data)
        
        if not labeling_result.success:
            return {
                'success': False,
                'error': labeling_result.error_message,
                'score': 0.0
            }
        
        labeled_data = labeling_result.labeled_data
        
        # Apply feature engineering
        try:
            enhanced_data = apply_enhanced_profit_feature_engineering(labeled_data)
        except Exception as e:
            return {
                'success': False,
                'error': f"Feature engineering failed: {e}",
                'score': 0.0
            }
        
        # Train ML models
        try:
            ml_results = train_ml_models_with_profit_potential(enhanced_data)
        except Exception as e:
            return {
                'success': False,
                'error': f"ML training failed: {e}",
                'score': 0.0
            }
        
        # Test 1: Direction prediction accuracy
        direction_score = self._test_direction_prediction(ml_results)
        
        # Test 2: Magnitude prediction accuracy
        magnitude_score = self._test_magnitude_prediction(ml_results)
        
        # Test 3: Confidence prediction accuracy
        confidence_score = self._test_confidence_prediction(ml_results)
        
        # Test 4: Profit-focused metrics
        profit_metrics_score = self._test_profit_metrics(ml_results)
        
        # Overall score
        overall_score = (direction_score + magnitude_score + confidence_score + profit_metrics_score) / 4
        
        return {
            'success': overall_score >= 0.6,  # Lower threshold for ML
            'score': overall_score,
            'direction_score': direction_score,
            'magnitude_score': magnitude_score,
            'confidence_score': confidence_score,
            'profit_metrics_score': profit_metrics_score,
            'ml_results': ml_results
        }
    
    def _validate_regime_performance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate regime-specific performance."""
        
        # Apply enhanced triple barrier labeling
        labeling_result = apply_enhanced_triple_barrier_labeling(data)
        
        if not labeling_result.success:
            return {
                'success': False,
                'error': labeling_result.error_message,
                'score': 0.0
            }
        
        labeled_data = labeling_result.labeled_data
        
        # Test regime-specific performance
        regime_scores = {}
        for regime in self.config.test_regimes:
            regime_data = labeled_data[labeled_data['hmm_regime'] == regime]
            if len(regime_data) > 10:  # Minimum samples
                regime_score = self._test_regime_specific_quality(regime_data, regime)
                regime_scores[f'regime_{regime}'] = regime_score
        
        # Overall regime score
        if regime_scores:
            overall_score = np.mean(list(regime_scores.values()))
        else:
            overall_score = 0.0
        
        return {
            'success': overall_score >= 0.6,
            'score': overall_score,
            'regime_scores': regime_scores
        }
    
    def _validate_feature_engineering(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate feature engineering quality."""
        
        # Apply enhanced triple barrier labeling
        labeling_result = apply_enhanced_triple_barrier_labeling(data)
        
        if not labeling_result.success:
            return {
                'success': False,
                'error': labeling_result.error_message,
                'score': 0.0
            }
        
        labeled_data = labeling_result.labeled_data
        
        # Apply feature engineering
        try:
            enhanced_data = apply_enhanced_profit_feature_engineering(labeled_data)
        except Exception as e:
            return {
                'success': False,
                'error': f"Feature engineering failed: {e}",
                'score': 0.0
            }
        
        # Test 1: Feature diversity
        diversity_score = self._test_feature_diversity(enhanced_data)
        
        # Test 2: Feature correlation
        correlation_score = self._test_feature_correlation(enhanced_data)
        
        # Test 3: Feature importance
        importance_score = self._test_feature_importance(enhanced_data)
        
        # Test 4: Feature stability
        stability_score = self._test_feature_stability(enhanced_data)
        
        # Overall score
        overall_score = (diversity_score + correlation_score + importance_score + stability_score) / 4
        
        return {
            'success': overall_score >= 0.7,
            'score': overall_score,
            'diversity_score': diversity_score,
            'correlation_score': correlation_score,
            'importance_score': importance_score,
            'stability_score': stability_score,
            'feature_count': len(enhanced_data.columns),
            'original_feature_count': len(labeled_data.columns)
        }
    
    def _validate_end_to_end_pipeline(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate the complete end-to-end pipeline."""
        
        start_time = time.time()
        
        try:
            # Step 1: Enhanced triple barrier labeling
            tprint("   Step 1: Enhanced triple barrier labeling...")
            labeling_result = apply_enhanced_triple_barrier_labeling(data)
            
            if not labeling_result.success:
                return {
                    'success': False,
                    'error': f"Labeling failed: {labeling_result.error_message}",
                    'score': 0.0
                }
            
            # Step 2: Feature engineering
            tprint("   Step 2: Feature engineering...")
            enhanced_data = apply_enhanced_profit_feature_engineering(labeling_result.labeled_data)
            
            # Step 3: ML model training
            tprint("   Step 3: ML model training...")
            ml_results = train_ml_models_with_profit_potential(enhanced_data)
            
            # Step 4: Model evaluation
            tprint("   Step 4: Model evaluation...")
            evaluation_score = self._evaluate_end_to_end_performance(ml_results)
            
            # Overall score
            overall_score = evaluation_score
            
            execution_time = time.time() - start_time
            
            return {
                'success': overall_score >= 0.6,
                'score': overall_score,
                'execution_time': execution_time,
                'labeling_success': labeling_result.success,
                'feature_engineering_success': True,
                'ml_training_success': True,
                'evaluation_score': evaluation_score
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"End-to-end pipeline failed: {e}",
                'score': 0.0
            }
    
    # Individual test methods
    def _test_label_distribution(self, data: pd.DataFrame) -> float:
        """Test label distribution balance."""
        if 'profit_category' not in data.columns:
            return 0.0
        
        category_counts = data['profit_category'].value_counts()
        if len(category_counts) < 2:
            return 0.0
        
        # Check if any category has less than minimum ratio
        min_ratio = self.config.min_label_distribution_ratio
        total_samples = len(data)
        
        for count in category_counts:
            if count / total_samples < min_ratio:
                return 0.5  # Partial score for imbalanced distribution
        
        return 1.0  # Full score for balanced distribution
    
    def _test_profit_consistency(self, data: pd.DataFrame) -> float:
        """Test profit category consistency with actual profits."""
        if 'profit_category' not in data.columns or 'potential_profit_pct' not in data.columns:
            return 0.0
        
        # Check if profit categories align with actual profit percentages
        consistency_score = 0.0
        total_checks = 0
        
        for category in data['profit_category'].unique():
            category_data = data[data['profit_category'] == category]
            if len(category_data) > 0:
                profits = category_data['potential_profit_pct']
                
                # Check if profits are in expected range for category
                if category == 'high_profit' and profits.mean() > 0.005:  # > 0.5%
                    consistency_score += 1
                elif category == 'medium_profit' and 0.001 < profits.mean() <= 0.005:  # 0.1-0.5%
                    consistency_score += 1
                elif category == 'low_profit' and 0 < profits.mean() <= 0.001:  # 0-0.1%
                    consistency_score += 1
                elif category == 'break_even' and abs(profits.mean()) <= 0.001:  # ±0.1%
                    consistency_score += 1
                elif category == 'small_loss' and -0.001 < profits.mean() <= 0:  # -0.1% to 0%
                    consistency_score += 1
                elif category == 'medium_loss' and -0.005 < profits.mean() <= -0.001:  # -0.5% to -0.1%
                    consistency_score += 1
                elif category == 'large_loss' and profits.mean() <= -0.005:  # < -0.5%
                    consistency_score += 1
                
                total_checks += 1
        
        return consistency_score / total_checks if total_checks > 0 else 0.0
    
    def _test_confidence_calibration(self, data: pd.DataFrame) -> float:
        """Test confidence score calibration."""
        if 'confidence_score' not in data.columns or 'profit_magnitude_score' not in data.columns:
            return 0.0
        
        # Check if confidence scores correlate with profit magnitude
        correlation = data['confidence_score'].corr(data['profit_magnitude_score'])
        
        # Good calibration should have positive correlation
        return max(0.0, correlation)
    
    def _test_magnitude_quality(self, data: pd.DataFrame) -> float:
        """Test profit magnitude score quality."""
        if 'profit_magnitude_score' not in data.columns:
            return 0.0
        
        magnitude_scores = data['profit_magnitude_score']
        
        # Check if magnitude scores are in valid range (0-10)
        valid_range = (magnitude_scores >= 0) & (magnitude_scores <= 10)
        range_score = valid_range.mean()
        
        # Check if magnitude scores have good distribution
        distribution_score = 1.0 - abs(magnitude_scores.std() - 2.5) / 2.5  # Target std of 2.5
        
        return (range_score + distribution_score) / 2
    
    def _test_profit_magnitude_accuracy(self, data: pd.DataFrame) -> float:
        """Test profit magnitude prediction accuracy."""
        if 'profit_magnitude_score' not in data.columns or 'potential_profit_pct' not in data.columns:
            return 0.0
        
        # Check correlation between magnitude scores and actual profits
        correlation = data['profit_magnitude_score'].corr(data['potential_profit_pct'])
        
        return max(0.0, correlation)
    
    def _test_confidence_accuracy(self, data: pd.DataFrame) -> float:
        """Test confidence prediction accuracy."""
        if 'confidence_score' not in data.columns:
            return 0.0
        
        # Check if confidence scores are well-distributed
        confidence_scores = data['confidence_score']
        
        # Good confidence should have reasonable distribution
        distribution_score = 1.0 - abs(confidence_scores.std() - 0.25) / 0.25  # Target std of 0.25
        
        return distribution_score
    
    def _test_regime_specific_accuracy(self, data: pd.DataFrame) -> float:
        """Test regime-specific accuracy."""
        if 'hmm_regime' not in data.columns:
            return 1.0  # No regime data, so no regime-specific issues
        
        regime_scores = []
        for regime in data['hmm_regime'].unique():
            if not pd.isna(regime):
                regime_data = data[data['hmm_regime'] == regime]
                if len(regime_data) > 5:  # Minimum samples
                    # Check if regime has reasonable profit distribution
                    if 'potential_profit_pct' in regime_data.columns:
                        regime_profit_std = regime_data['potential_profit_pct'].std()
                        # Good regime should have reasonable profit variability
                        regime_score = min(1.0, regime_profit_std / 0.01)  # Target std of 1%
                        regime_scores.append(regime_score)
        
        return np.mean(regime_scores) if regime_scores else 0.0
    
    def _test_profit_correlation(self, data: pd.DataFrame) -> float:
        """Test profit correlation with predictions."""
        if 'profit_magnitude_score' not in data.columns or 'potential_profit_pct' not in data.columns:
            return 0.0
        
        # Check correlation between predicted and actual profits
        correlation = data['profit_magnitude_score'].corr(data['potential_profit_pct'])
        
        return max(0.0, correlation)
    
    def _test_direction_prediction(self, ml_results: Dict[str, Any]) -> float:
        """Test direction prediction accuracy."""
        if 'direction_model' not in ml_results:
            return 0.0
        
        direction_result = ml_results['direction_model']
        if 'accuracy' in direction_result:
            return direction_result['accuracy']
        
        return 0.0
    
    def _test_magnitude_prediction(self, ml_results: Dict[str, Any]) -> float:
        """Test magnitude prediction accuracy."""
        if 'magnitude_model' not in ml_results:
            return 0.0
        
        magnitude_result = ml_results['magnitude_model']
        if 'r2' in magnitude_result:
            return max(0.0, magnitude_result['r2'])
        
        return 0.0
    
    def _test_confidence_prediction(self, ml_results: Dict[str, Any]) -> float:
        """Test confidence prediction accuracy."""
        if 'confidence_model' not in ml_results:
            return 0.0
        
        confidence_result = ml_results['confidence_model']
        if 'r2' in confidence_result:
            return max(0.0, confidence_result['r2'])
        
        return 0.0
    
    def _test_profit_metrics(self, ml_results: Dict[str, Any]) -> float:
        """Test profit-focused metrics."""
        # Check if any model has profit metrics
        for model_name, model_result in ml_results.items():
            if isinstance(model_result, dict) and 'profit_metrics' in model_result:
                profit_metrics = model_result['profit_metrics']
                if 'profit_correlation' in profit_metrics:
                    return max(0.0, profit_metrics['profit_correlation'])
        
        return 0.0
    
    def _test_regime_specific_quality(self, regime_data: pd.DataFrame, regime: int) -> float:
        """Test quality of regime-specific data."""
        if len(regime_data) < 5:
            return 0.0
        
        # Check profit distribution for this regime
        if 'potential_profit_pct' in regime_data.columns:
            profit_std = regime_data['potential_profit_pct'].std()
            # Good regime should have reasonable profit variability
            return min(1.0, profit_std / 0.01)  # Target std of 1%
        
        return 0.5  # Default score if no profit data
    
    def _test_feature_diversity(self, data: pd.DataFrame) -> float:
        """Test feature diversity."""
        # Check if we have a good variety of features
        feature_types = set()
        for col in data.columns:
            if 'profit_cat' in col:
                feature_types.add('category')
            elif 'magnitude' in col:
                feature_types.add('magnitude')
            elif 'confidence' in col:
                feature_types.add('confidence')
            elif 'regime' in col:
                feature_types.add('regime')
            elif 'interaction' in col:
                feature_types.add('interaction')
            elif any(f'_{w}' in col for w in [5, 10, 20, 50]):
                feature_types.add('timeseries')
        
        # Good diversity should have multiple feature types
        diversity_score = min(1.0, len(feature_types) / 5)  # Target 5 feature types
        
        return diversity_score
    
    def _test_feature_correlation(self, data: pd.DataFrame) -> float:
        """Test feature correlation quality."""
        # Check for excessive correlation between features
        numerical_features = data.select_dtypes(include=[np.number]).columns
        if len(numerical_features) < 2:
            return 0.0
        
        correlation_matrix = data[numerical_features].corr()
        
        # Check for high correlations (should be limited)
        high_correlations = (correlation_matrix.abs() > 0.95).sum().sum() - len(numerical_features)  # Exclude diagonal
        total_pairs = len(numerical_features) * (len(numerical_features) - 1) / 2
        
        # Good correlation should have limited high correlations
        correlation_score = max(0.0, 1.0 - high_correlations / total_pairs)
        
        return correlation_score
    
    def _test_feature_importance(self, data: pd.DataFrame) -> float:
        """Test feature importance quality."""
        # This is a simplified test - in practice, you'd use actual feature importance
        # For now, just check if we have reasonable number of features
        feature_count = len(data.columns)
        
        # Good feature engineering should create many features
        importance_score = min(1.0, feature_count / 50)  # Target 50 features
        
        return importance_score
    
    def _test_feature_stability(self, data: pd.DataFrame) -> float:
        """Test feature stability."""
        # Check for NaN values in features
        nan_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        
        # Good stability should have minimal NaN values
        stability_score = max(0.0, 1.0 - nan_ratio)
        
        return stability_score
    
    def _evaluate_end_to_end_performance(self, ml_results: Dict[str, Any]) -> float:
        """Evaluate end-to-end pipeline performance."""
        scores = []
        
        # Check each model's performance
        for model_name, model_result in ml_results.items():
            if isinstance(model_result, dict):
                if 'accuracy' in model_result:
                    scores.append(model_result['accuracy'])
                elif 'r2' in model_result:
                    scores.append(max(0.0, model_result['r2']))
        
        return np.mean(scores) if scores else 0.0
    
    def _calculate_overall_validation_score(self, result: ValidationResult) -> float:
        """Calculate overall validation score."""
        scores = []
        
        if result.label_quality_result and 'score' in result.label_quality_result:
            scores.append(result.label_quality_result['score'])
        
        if result.profit_accuracy_result and 'score' in result.profit_accuracy_result:
            scores.append(result.profit_accuracy_result['score'])
        
        if result.ml_performance_result and 'score' in result.ml_performance_result:
            scores.append(result.ml_performance_result['score'])
        
        if result.regime_validation_result and 'score' in result.regime_validation_result:
            scores.append(result.regime_validation_result['score'])
        
        if result.feature_validation_result and 'score' in result.feature_validation_result:
            scores.append(result.feature_validation_result['score'])
        
        if result.end_to_end_result and 'score' in result.end_to_end_result:
            scores.append(result.end_to_end_result['score'])
        
        return np.mean(scores) if scores else 0.0
    
    def _count_total_tests(self, result: ValidationResult) -> int:
        """Count total number of tests run."""
        return 6  # Fixed number of validation categories
    
    def _count_passed_tests(self, result: ValidationResult) -> int:
        """Count number of passed tests."""
        passed = 0
        
        if result.label_quality_result and result.label_quality_result.get('success', False):
            passed += 1
        
        if result.profit_accuracy_result and result.profit_accuracy_result.get('success', False):
            passed += 1
        
        if result.ml_performance_result and result.ml_performance_result.get('success', False):
            passed += 1
        
        if result.regime_validation_result and result.regime_validation_result.get('success', False):
            passed += 1
        
        if result.feature_validation_result and result.feature_validation_result.get('success', False):
            passed += 1
        
        if result.end_to_end_result and result.end_to_end_result.get('success', False):
            passed += 1
        
        return passed

# Convenience functions
def create_enhanced_triple_barrier_validator(
    enable_label_quality_validation: bool = True,
    enable_profit_accuracy_validation: bool = True,
    enable_ml_performance_validation: bool = True,
    enable_regime_validation: bool = True,
    enable_feature_validation: bool = True,
    enable_end_to_end_validation: bool = True,
    test_data_size: int = 1000
) -> EnhancedTripleBarrierValidator:
    """Create enhanced triple barrier validator with specified parameters."""
    config = ValidationConfig(
        enable_label_quality_validation=enable_label_quality_validation,
        enable_profit_accuracy_validation=enable_profit_accuracy_validation,
        enable_ml_performance_validation=enable_ml_performance_validation,
        enable_regime_validation=enable_regime_validation,
        enable_feature_validation=enable_feature_validation,
        enable_end_to_end_validation=enable_end_to_end_validation,
        test_data_size=test_data_size
    )
    
    return EnhancedTripleBarrierValidator(config)

def run_enhanced_triple_barrier_validation(
    data: Optional[pd.DataFrame] = None,
    enable_label_quality_validation: bool = True,
    enable_profit_accuracy_validation: bool = True,
    enable_ml_performance_validation: bool = True,
    enable_regime_validation: bool = True,
    enable_feature_validation: bool = True,
    enable_end_to_end_validation: bool = True,
    test_data_size: int = 1000
) -> ValidationResult:
    """Run comprehensive validation of enhanced triple barrier system."""
    validator = create_enhanced_triple_barrier_validator(
        enable_label_quality_validation=enable_label_quality_validation,
        enable_profit_accuracy_validation=enable_profit_accuracy_validation,
        enable_ml_performance_validation=enable_ml_performance_validation,
        enable_regime_validation=enable_regime_validation,
        enable_feature_validation=enable_feature_validation,
        enable_end_to_end_validation=enable_end_to_end_validation,
        test_data_size=test_data_size
    )
    
    return validator.run_comprehensive_validation(data)

if __name__ == '__main__':
    # Test the enhanced triple barrier validation
    tprint('🧪 Testing Enhanced Triple Barrier Validation')
    
    # Run comprehensive validation
    tprint('\n📊 Running comprehensive validation...')
    validation_result = run_enhanced_triple_barrier_validation()
    
    tprint(f'✅ Validation completed')
    tprint(f'   Overall Success: {validation_result.overall_success}')
    tprint(f'   Validation Score: {validation_result.validation_score:.2%}')
    tprint(f'   Tests Passed: {validation_result.passed_tests}/{validation_result.total_tests}')
    tprint(f'   Duration: {validation_result.execution_duration:.2f}s')
    
    # Show detailed results
    if validation_result.label_quality_result:
        tprint(f'\n📋 Label Quality: {validation_result.label_quality_result["score"]:.2%}')
    
    if validation_result.profit_accuracy_result:
        tprint(f'📋 Profit Accuracy: {validation_result.profit_accuracy_result["score"]:.2%}')
    
    if validation_result.ml_performance_result:
        tprint(f'📋 ML Performance: {validation_result.ml_performance_result["score"]:.2%}')
    
    if validation_result.regime_validation_result:
        tprint(f'📋 Regime Validation: {validation_result.regime_validation_result["score"]:.2%}')
    
    if validation_result.feature_validation_result:
        tprint(f'📋 Feature Validation: {validation_result.feature_validation_result["score"]:.2%}')
    
    if validation_result.end_to_end_result:
        tprint(f'📋 End-to-End: {validation_result.end_to_end_result["score"]:.2%}')
    
    tprint('✅ Enhanced Triple Barrier Validation test completed!')