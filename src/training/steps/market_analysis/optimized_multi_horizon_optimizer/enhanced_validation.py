"""
Enhanced Validation Framework using ml_commons utilities

This module provides comprehensive validation of optimized timeframes
using ml_commons validation utilities extensively.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

# Import ml_commons validation utilities
from src.utils.ml_common.validation.unified_validation_system import UnifiedValidationSystem
from src.utils.ml_common.validation.temporal_cross_validation import TemporalCrossValidator
from src.utils.ml_common.validation.cv_utils import CrossValidationUtilities
from src.utils.ml_common.validation.enhanced_overfitting_detection import EnhancedOverfittingDetector
from src.utils.ml_common.validation.data_leakage_prevention import DataLeakagePrevention
from src.utils.ml_common.validation.stability import StabilityValidator
from src.utils.ml_common.validation.model_complexity_analysis import ModelComplexityAnalyzer

# Import optimization configuration
from .optimization_config import ValidationConfig, ValidationLevel, OptimizationResult

# Import multi-horizon components
from src.training.steps.pre_training.multi_horizon_profit_labeler import MultiHorizonConfig

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of validation process."""
    validation_score: float
    statistical_metrics: Dict[str, float]
    economic_metrics: Dict[str, float]
    microstructure_metrics: Dict[str, float]
    cross_validation_metrics: Dict[str, float]
    stability_metrics: Dict[str, float]
    overall_quality: str
    recommendations: List[str]
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'validation_score': self.validation_score,
            'statistical_metrics': self.statistical_metrics,
            'economic_metrics': self.economic_metrics,
            'microstructure_metrics': self.microstructure_metrics,
            'cross_validation_metrics': self.cross_validation_metrics,
            'stability_metrics': self.stability_metrics,
            'overall_quality': self.overall_quality,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp.isoformat()
        }

class EnhancedValidationFramework:
    """
    Enhanced validation framework using ml_commons utilities extensively.
    """

    def __init__(self, config: ValidationConfig):
        """Initialize enhanced validation framework."""
        self.config = config
        self.logger = logging.getLogger('EnhancedValidationFramework')

        # Initialize ml_commons validation utilities
        self._initialize_ml_commons_utilities()

        self.logger.info(f'🔧 Enhanced validation framework initialized with {config.validation_level.value} level')

    def _initialize_ml_commons_utilities(self):
        """Initialize ml_commons validation utilities."""
        try:
            # Initialize unified validation system
            self.unified_validator = UnifiedValidationSystem()

            # Initialize temporal cross-validator
            self.temporal_cv = TemporalCrossValidator(
                n_splits=self.config.cv_folds,
                gap=1  # 1 period gap to prevent lookahead
            )

            # Initialize cross-validation utilities
            cv_config = {
                'initial_train_size': 0.6,
                'step_size': 0.1,
                'min_test_size': 0.1
            }
            self.cv_utilities = CrossValidationUtilities(cv_config)

            # Initialize overfitting detector
            self.overfitting_detector = EnhancedOverfittingDetector()

            # Initialize data leakage prevention
            self.leakage_prevention = DataLeakagePrevention()

            # Initialize stability validator
            self.stability_validator = StabilityValidator()

            # Initialize model complexity analyzer
            self.complexity_analyzer = ModelComplexityAnalyzer()

            self.logger.info('✅ ml_commons validation utilities initialized successfully')

        except Exception as e:
            self.logger.error(f'❌ Failed to initialize ml_commons validation utilities: {e}')
            raise RuntimeError(f"Failed to initialize ml_commons validation utilities: {e}")

    def validate_optimized_configuration(self,
                                       config: MultiHorizonConfig,
                                       market_data: pd.DataFrame,
                                       model_type: str = "analyst") -> ValidationResult:
        """
        Comprehensive validation of optimized configuration using ml_commons utilities.

        Args:
            config: Optimized configuration to validate
            market_data: Market data for validation
            model_type: Type of model (analyst/tactician)

        Returns:
            ValidationResult with comprehensive validation metrics
        """
        self.logger.info(f'🔍 Starting comprehensive validation for {model_type} model')

        try:
            # Generate labels using configuration
            from src.training.steps.pre_training.multi_horizon_profit_labeler import MultiHorizonProfitLabeler
            labeler = MultiHorizonProfitLabeler(config)
            labeled_data = labeler.generate_labels(market_data.copy())

            # Statistical validation using ml_commons
            statistical_metrics = self._statistical_validation(labeled_data, market_data)

            # Economic validation using ml_commons
            economic_metrics = self._economic_validation(labeled_data, market_data)

            # Market microstructure validation using ml_commons
            microstructure_metrics = self._microstructure_validation(labeled_data, market_data)

            # Cross-validation using ml_commons
            cross_validation_metrics = self._cross_validation(labeled_data, market_data)

            # Stability validation using ml_commons
            stability_metrics = self._stability_validation(labeled_data, market_data)

            # Calculate overall validation score
            validation_score = self._calculate_overall_score(
                statistical_metrics, economic_metrics,
                microstructure_metrics, cross_validation_metrics, stability_metrics
            )

            # Determine overall quality
            overall_quality = self._determine_quality(validation_score)

            # Generate recommendations
            recommendations = self._generate_recommendations(
                statistical_metrics, economic_metrics,
                microstructure_metrics, cross_validation_metrics, stability_metrics
            )

            result = ValidationResult(
                validation_score=validation_score,
                statistical_metrics=statistical_metrics,
                economic_metrics=economic_metrics,
                microstructure_metrics=microstructure_metrics,
                cross_validation_metrics=cross_validation_metrics,
                stability_metrics=stability_metrics,
                overall_quality=overall_quality,
                recommendations=recommendations,
                timestamp=datetime.now()
            )

            self.logger.info(f'✅ Validation completed - Score: {validation_score:.3f}, Quality: {overall_quality}')
            return result

        except Exception as e:
            self.logger.error(f'❌ Validation failed: {e}')
            raise RuntimeError(f"Validation failed: {e}")

    def _statistical_validation(self, labeled_data: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, float]:
        """Perform statistical validation using ml_commons utilities."""
        self.logger.info('   → Running statistical validation...')

        try:
            # Use ml_commons unified validation system
            validation_result = self.unified_validator.validate_model_performance(
                model=None,  # No model needed for configuration validation
                X=market_data,
                y=labeled_data,
                validation_type='statistical_validation'
            )

            # Extract statistical metrics
            statistical_metrics = {
                'information_coefficient': validation_result.get('information_coefficient', 0.0),
                'signal_to_noise_ratio': validation_result.get('signal_to_noise_ratio', 0.0),
                'hit_rate': validation_result.get('hit_rate', 0.5),
                'statistical_significance': validation_result.get('statistical_significance', 0.0),
                'correlation_strength': validation_result.get('correlation_strength', 0.0),
                'overall_statistical_score': validation_result.get('overall_statistical_score', 0.5)
            }

            return statistical_metrics

        except Exception as e:
            self.logger.warning(f'⚠️ Statistical validation error: {e}')
            return {
                'information_coefficient': 0.0,
                'signal_to_noise_ratio': 0.0,
                'hit_rate': 0.5,
                'statistical_significance': 0.0,
                'correlation_strength': 0.0,
                'overall_statistical_score': 0.5
            }

    def _economic_validation(self, labeled_data: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, float]:
        """Perform economic validation using ml_commons utilities."""
        self.logger.info('   → Running economic validation...')

        try:
            # Use ml_commons unified validation system
            validation_result = self.unified_validator.validate_model_performance(
                model=None,
                X=market_data,
                y=labeled_data,
                validation_type='economic_validation'
            )

            # Extract economic metrics
            economic_metrics = {
                'transaction_cost_ratio': validation_result.get('transaction_cost_ratio', 0.0),
                'sharpe_ratio': validation_result.get('sharpe_ratio', 0.0),
                'max_drawdown': validation_result.get('max_drawdown', 0.1),
                'information_ratio': validation_result.get('information_ratio', 0.0),
                'economic_significance': validation_result.get('economic_significance', 0.0),
                'overall_economic_score': validation_result.get('overall_economic_score', 0.5)
            }

            return economic_metrics

        except Exception as e:
            self.logger.warning(f'⚠️ Economic validation error: {e}')
            return {
                'transaction_cost_ratio': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.1,
                'information_ratio': 0.0,
                'economic_significance': 0.0,
                'overall_economic_score': 0.5
            }

    def _microstructure_validation(self, labeled_data: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, float]:
        """Perform market microstructure validation using ml_commons utilities."""
        self.logger.info('   → Running microstructure validation...')

        try:
            # Use ml_commons unified validation system
            validation_result = self.unified_validator.validate_model_performance(
                model=None,
                X=market_data,
                y=labeled_data,
                validation_type='microstructure_validation'
            )

            # Extract microstructure metrics
            microstructure_metrics = {
                'liquidity_score': validation_result.get('liquidity_score', 0.0),
                'volatility_stability': validation_result.get('volatility_stability', 0.0),
                'market_depth_score': validation_result.get('market_depth_score', 0.0),
                'spread_impact': validation_result.get('spread_impact', 0.0),
                'overall_microstructure_score': validation_result.get('overall_microstructure_score', 0.5)
            }

            return microstructure_metrics

        except Exception as e:
            self.logger.warning(f'⚠️ Microstructure validation error: {e}')
            return {
                'liquidity_score': 0.0,
                'volatility_stability': 0.0,
                'market_depth_score': 0.0,
                'spread_impact': 0.0,
                'overall_microstructure_score': 0.5
            }

    def _cross_validation(self, labeled_data: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, float]:
        """Perform cross-validation using ml_commons utilities."""
        self.logger.info('   → Running cross-validation...')

        try:
            # Use ml_commons temporal cross-validation
            cv_scores = []

            # Perform temporal cross-validation
            for train_idx, test_idx in self.temporal_cv.split(market_data):
                train_data = market_data.iloc[train_idx]
                test_data = market_data.iloc[test_idx]
                train_labels = labeled_data.iloc[train_idx]
                test_labels = labeled_data.iloc[test_idx]

                # Calculate performance on test set
                performance = self._calculate_cv_performance(train_labels, test_labels)
                cv_scores.append(performance)

            # Calculate cross-validation metrics
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            stability_score = 1.0 - (cv_std / (cv_mean + 1e-9))

            cross_validation_metrics = {
                'cross_validation_score': cv_mean,
                'cross_validation_std': cv_std,
                'stability_score': stability_score,
                'overall_cv_score': (cv_mean + stability_score) / 2
            }

            return cross_validation_metrics

        except Exception as e:
            self.logger.warning(f'⚠️ Cross-validation error: {e}')
            return {
                'cross_validation_score': 0.5,
                'cross_validation_std': 0.1,
                'stability_score': 0.5,
                'overall_cv_score': 0.5
            }

    def _stability_validation(self, labeled_data: pd.DataFrame, market_data: pd.DataFrame) -> Dict[str, float]:
        """Perform stability validation using ml_commons utilities."""
        self.logger.info('   → Running stability validation...')

        try:
            # Use ml_commons stability validator
            stability_result = self.stability_validator.validate_stability(
                data=market_data,
                labels=labeled_data,
                validation_type='configuration_stability'
            )

            # Extract stability metrics
            stability_metrics = {
                'temporal_stability': stability_result.get('temporal_stability', 0.5),
                'regime_stability': stability_result.get('regime_stability', 0.5),
                'parameter_stability': stability_result.get('parameter_stability', 0.5),
                'overall_stability_score': stability_result.get('overall_stability_score', 0.5)
            }

            return stability_metrics

        except Exception as e:
            self.logger.warning(f'⚠️ Stability validation error: {e}')
            return {
                'temporal_stability': 0.5,
                'regime_stability': 0.5,
                'parameter_stability': 0.5,
                'overall_stability_score': 0.5
            }

    def _calculate_cv_performance(self, train_labels: pd.DataFrame, test_labels: pd.DataFrame) -> float:
        """Calculate cross-validation performance using ml_commons utilities."""
        try:
            # Use ml_commons cross-validation utilities
            performance = self.cv_utilities.calculate_performance_metric(
                train_labels=train_labels,
                test_labels=test_labels,
                metric_type='configuration_performance'
            )

            return performance

        except Exception as e:
            self.logger.warning(f'⚠️ CV performance calculation error: {e}')
            return 0.5

    def _calculate_overall_score(self, statistical_metrics: Dict[str, float],
                               economic_metrics: Dict[str, float],
                               microstructure_metrics: Dict[str, float],
                               cross_validation_metrics: Dict[str, float],
                               stability_metrics: Dict[str, float]) -> float:
        """Calculate overall validation score using ml_commons utilities."""
        try:
            # Use ml_commons unified validation system for overall scoring
            overall_score = self.unified_validator.calculate_overall_validation_score(
                statistical_metrics=statistical_metrics,
                economic_metrics=economic_metrics,
                microstructure_metrics=microstructure_metrics,
                cross_validation_metrics=cross_validation_metrics,
                stability_metrics=stability_metrics
            )

            return max(0.0, min(1.0, overall_score))

        except Exception as e:
            self.logger.warning(f'⚠️ Overall score calculation error: {e}')
            # Fallback to simple weighted average
            weights = {
                'statistical': 0.25,
                'economic': 0.25,
                'microstructure': 0.15,
                'cross_validation': 0.20,
                'stability': 0.15
            }

            score = (
                weights['statistical'] * statistical_metrics.get('overall_statistical_score', 0.5) +
                weights['economic'] * economic_metrics.get('overall_economic_score', 0.5) +
                weights['microstructure'] * microstructure_metrics.get('overall_microstructure_score', 0.5) +
                weights['cross_validation'] * cross_validation_metrics.get('overall_cv_score', 0.5) +
                weights['stability'] * stability_metrics.get('overall_stability_score', 0.5)
            )

            return max(0.0, min(1.0, score))

    def _determine_quality(self, validation_score: float) -> str:
        """Determine overall quality based on validation score."""
        if validation_score >= 0.8:
            return "Excellent"
        elif validation_score >= 0.7:
            return "Good"
        elif validation_score >= 0.6:
            return "Fair"
        elif validation_score >= 0.5:
            return "Poor"
        else:
            return "Very Poor"

    def _generate_recommendations(self, statistical_metrics: Dict[str, float],
                                economic_metrics: Dict[str, float],
                                microstructure_metrics: Dict[str, float],
                                cross_validation_metrics: Dict[str, float],
                                stability_metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations using ml_commons utilities."""
        recommendations = []

        try:
            # Use ml_commons unified validation system for recommendations
            recommendations = self.unified_validator.generate_validation_recommendations(
                statistical_metrics=statistical_metrics,
                economic_metrics=economic_metrics,
                microstructure_metrics=microstructure_metrics,
                cross_validation_metrics=cross_validation_metrics,
                stability_metrics=stability_metrics
            )

            return recommendations

        except Exception as e:
            self.logger.warning(f'⚠️ Recommendation generation error: {e}')
            # Fallback to basic recommendations
            recommendations = []

            # Statistical recommendations
            if statistical_metrics.get('information_coefficient', 0) < 0.05:
                recommendations.append("Consider improving feature engineering to increase information coefficient")

            if statistical_metrics.get('hit_rate', 0) < 0.55:
                recommendations.append("Optimize profit targets to improve hit rate")

            # Economic recommendations
            if economic_metrics.get('transaction_cost_ratio', 0) > 0.1:
                recommendations.append("Reduce transaction frequency or improve profit targets to lower cost ratio")

            if economic_metrics.get('sharpe_ratio', 0) < 0.5:
                recommendations.append("Improve risk-adjusted returns by optimizing timeframes")

            # Microstructure recommendations
            if microstructure_metrics.get('liquidity_score', 0) < 0.7:
                recommendations.append("Consider market liquidity when selecting timeframes")

            if microstructure_metrics.get('volatility_stability', 0) < 0.6:
                recommendations.append("Optimize for more stable volatility patterns")

            # Cross-validation recommendations
            if cross_validation_metrics.get('stability_score', 0) < 0.8:
                recommendations.append("Improve model stability across different time periods")

            # Stability recommendations
            if stability_metrics.get('overall_stability_score', 0) < 0.7:
                recommendations.append("Improve configuration stability across different market conditions")

            return recommendations
