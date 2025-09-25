"""
Economic Validator for Tree Architecture Search

This module provides economic validation capabilities for TAS models including
economic significance evaluation, trading viability assessment, and financial
performance validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
import warnings

# Import shared utilities
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)
from src.utils.common_operations import (
    safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
    validate_finite, validate_positive, validate_range, safe_correlation,
    safe_covariance, safe_percentage_change, safe_weighted_average
)
from src.utils.math_validation import (
    MathValidation, validate_numeric_array, safe_matrix_inverse
)
from src.utils.serialization_utils import JSONSerializer, PickleSerializer

# Import unified economic evaluator from shared utils
# from src.utils.nas_tas.unified_evaluator import (
#     UnifiedEconomicSignificanceEvaluator, EconomicEvaluationConfig,
#     EconomicSignificanceResult, create_unified_economic_evaluator,
# )  # DELETED - use unified regime detector
from src.utils.nas_tas.unified_evaluator import (
    UnifiedEconomicSignificanceEvaluator, EconomicEvaluationConfig,
    EconomicSignificanceResult, create_unified_economic_evaluator,
    quick_economic_evaluation
)

logger = logging.getLogger(__name__)


class EconomicValidationType(Enum):
    """Types of economic validation."""
    SIGNIFICANCE = "significance"
    VIABILITY = "viability"
    PERFORMANCE = "performance"
    RISK_ADJUSTED = "risk_adjusted"
    TRADING_METRICS = "trading_metrics"


class ValidationLevel(Enum):
    """Validation levels."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    COMPREHENSIVE = "comprehensive"


@dataclass
class EconomicValidationConfig:
    """Configuration for economic validation."""
    
    # Validation parameters
    validation_type: EconomicValidationType = EconomicValidationType.SIGNIFICANCE
    validation_level: ValidationLevel = ValidationLevel.INTERMEDIATE
    
    # Economic significance thresholds
    significance_threshold: float = 0.6
    price_impact_threshold: float = 0.5
    volume_threshold: float = 0.4
    volatility_threshold: float = 0.5
    trend_threshold: float = 0.6
    
    # Trading viability parameters
    min_win_rate: float = 0.4
    min_profit_factor: float = 1.2
    max_drawdown_threshold: float = 0.2
    min_sharpe_ratio: float = 0.5
    
    # Performance validation
    min_accuracy: float = 0.6
    min_precision: float = 0.5
    min_recall: float = 0.4
    min_f1_score: float = 0.5
    
    # Risk-adjusted metrics
    risk_free_rate: float = 0.02
    confidence_level: float = 0.95
    bootstrap_iterations: int = 100
    
    # Economic indicators
    enable_economic_indicators: bool = True
    economic_indicators_lookback: int = 252
    economic_correlation_threshold: float = 0.3
    
    # Position-aware analysis
    enable_position_aware_analysis: bool = True
    
    # Advanced features
    enable_bootstrap_analysis: bool = True
    enable_regime_specific_analysis: bool = True
    min_regime_samples: int = 50
    regime_stability_threshold: float = 0.7


@dataclass
class EconomicValidationResult:
    """Result from economic validation."""
    
    # Validation success
    success: bool
    validation_type: str
    validation_level: str
    
    # Economic significance
    economic_significance: Optional[EconomicSignificanceResult] = None
    significance_score: float = 0.0
    significance_level: str = "low"
    
    # Trading viability
    trading_viability: Dict[str, Any] = field(default_factory=dict)
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Performance metrics
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Risk-adjusted metrics
    risk_adjusted_metrics: Dict[str, float] = field(default_factory=dict)
    var_95: float = 0.0
    cvar_95: float = 0.0
    max_drawdown_duration: int = 0
    
    # Economic indicators
    economic_indicator_correlation: Dict[str, float] = field(default_factory=dict)
    
    # Regime-specific analysis
    regime_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Validation metadata
    validation_timestamp: datetime = field(default_factory=datetime.now)
    n_samples: int = 0
    execution_time: float = 0.0
    error_message: Optional[str] = None


class EconomicValidator:
    """
    Economic Validator for Tree Architecture Search.
    
    Provides comprehensive economic validation including significance evaluation,
    trading viability assessment, and performance validation.
    """
    
    def __init__(self, config: EconomicValidationConfig):
        """Initialize economic validator.
        
        Args:
            config: Economic validation configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize unified economic evaluator
        self.economic_evaluator = create_unified_economic_evaluator(
            EconomicEvaluationConfig(
                significance_threshold=config.significance_threshold,
                price_impact_threshold=config.price_impact_threshold,
                volume_threshold=config.volume_threshold,
                volatility_threshold=config.volatility_threshold,
                trend_threshold=config.trend_threshold,
                enable_economic_indicators=config.enable_economic_indicators,
                enable_position_aware_analysis=config.enable_position_aware_analysis,
                enable_bootstrap_analysis=config.enable_bootstrap_analysis,
                enable_regime_specific_analysis=config.enable_regime_specific_analysis
            )
        )
        
        # Math validation utilities
        self.math_validator = MathValidation()
        
        tprint_info("✅ Economic Validator initialized")
        tprint_info(f"   Validation type: {config.validation_type.value}")
        tprint_info(f"   Validation level: {config.validation_level.value}")
        tprint_info(f"   Significance threshold: {config.significance_threshold}")
        tprint_info(f"   Economic indicators: {config.enable_economic_indicators}")
    
    def validate(self,
                 model: Any,
                 X: np.ndarray,
                 y: np.ndarray,
                 market_data: Optional[np.ndarray] = None,
                 regime_predictions: Optional[np.ndarray] = None,
                 timestamps: Optional[np.ndarray] = None) -> EconomicValidationResult:
        """
        Perform comprehensive economic validation.
        
        Args:
            model: Trained model to validate
            X: Features
            y: Target values
            market_data: Optional market data (OHLCV)
            regime_predictions: Optional regime predictions
            timestamps: Optional timestamps
            
        Returns:
            Comprehensive economic validation result
        """
        start_time = time.time()
        
        try:
            tprint_info("💰 Starting comprehensive economic validation...")
            tprint_info(f"   Data shape: {X.shape}")
            tprint_info(f"   Validation type: {self.config.validation_type.value}")
            tprint_info(f"   Validation level: {self.config.validation_level.value}")
            
            # Initialize result
            result = EconomicValidationResult(
                success=True,
                validation_type=self.config.validation_type.value,
                validation_level=self.config.validation_level.value,
                n_samples=len(X)
            )
            
            # Perform validation based on type and level
            if self.config.validation_type == EconomicValidationType.SIGNIFICANCE:
                result = self._validate_economic_significance(
                    model, X, y, market_data, regime_predictions, result
                )
            elif self.config.validation_type == EconomicValidationType.VIABILITY:
                result = self._validate_trading_viability(
                    model, X, y, market_data, regime_predictions, result
                )
            elif self.config.validation_type == EconomicValidationType.PERFORMANCE:
                result = self._validate_performance_metrics(
                    model, X, y, market_data, regime_predictions, result
                )
            elif self.config.validation_type == EconomicValidationType.RISK_ADJUSTED:
                result = self._validate_risk_adjusted_metrics(
                    model, X, y, market_data, regime_predictions, result
                )
            elif self.config.validation_type == EconomicValidationType.TRADING_METRICS:
                result = self._validate_trading_metrics(
                    model, X, y, market_data, regime_predictions, result
                )
            
            # Add comprehensive validation if level is advanced or comprehensive
            if self.config.validation_level in [ValidationLevel.ADVANCED, ValidationLevel.COMPREHENSIVE]:
                result = self._add_comprehensive_validation(
                    model, X, y, market_data, regime_predictions, result
                )
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            tprint_success(f"✅ Economic validation completed in {execution_time:.2f}s")
            tprint_info(f"   Success: {result.success}")
            tprint_info(f"   Significance score: {result.significance_score:.3f}")
            tprint_info(f"   Win rate: {result.win_rate:.3f}")
            tprint_info(f"   Sharpe ratio: {result.sharpe_ratio:.3f}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            tprint_error(f"❌ Economic validation failed: {e}")
            
            return EconomicValidationResult(
                success=False,
                validation_type=self.config.validation_type.value,
                validation_level=self.config.validation_level.value,
                n_samples=len(X),
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _validate_economic_significance(self,
                                      model: Any,
                                      X: np.ndarray,
                                      y: np.ndarray,
                                      market_data: Optional[np.ndarray],
                                      regime_predictions: Optional[np.ndarray],
                                      result: EconomicValidationResult) -> EconomicValidationResult:
        """Validate economic significance."""
        try:
            tprint_debug("🔍 Validating economic significance...")
            
            # Use unified economic evaluator
            if market_data is not None and regime_predictions is not None:
                economic_result = self.economic_evaluator.evaluate(
                    market_data, regime_predictions
                )
                
                result.economic_significance = economic_result
                result.significance_score = economic_result.overall_score
                result.significance_level = economic_result.significance_level
                
                tprint_info(f"   Economic significance score: {result.significance_score:.3f}")
                tprint_info(f"   Significance level: {result.significance_level}")
            else:
                tprint_warning("⚠️ Market data or regime predictions not available for economic significance validation")
                result.significance_score = 0.5  # Default neutral score
                result.significance_level = "medium"
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Economic significance validation failed: {e}")
            result.significance_score = 0.0
            result.significance_level = "low"
            return result
    
    def _validate_trading_viability(self,
                                  model: Any,
                                  X: np.ndarray,
                                  y: np.ndarray,
                                  market_data: Optional[np.ndarray],
                                  regime_predictions: Optional[np.ndarray],
                                  result: EconomicValidationResult) -> EconomicValidationResult:
        """Validate trading viability."""
        try:
            tprint_debug("💼 Validating trading viability...")
            
            # Make predictions
            y_pred = model.predict(X)
            
            # Calculate trading metrics
            trading_metrics = self._calculate_trading_metrics(y, y_pred, market_data)
            
            result.trading_viability = trading_metrics
            result.win_rate = trading_metrics.get('win_rate', 0.0)
            result.profit_factor = trading_metrics.get('profit_factor', 0.0)
            result.max_drawdown = trading_metrics.get('max_drawdown', 0.0)
            result.sharpe_ratio = trading_metrics.get('sharpe_ratio', 0.0)
            
            # Check viability thresholds
            viable = (
                result.win_rate >= self.config.min_win_rate and
                result.profit_factor >= self.config.min_profit_factor and
                result.max_drawdown <= self.config.max_drawdown_threshold and
                result.sharpe_ratio >= self.config.min_sharpe_ratio
            )
            
            result.trading_viability['viable'] = viable
            
            tprint_info(f"   Win rate: {result.win_rate:.3f}")
            tprint_info(f"   Profit factor: {result.profit_factor:.3f}")
            tprint_info(f"   Max drawdown: {result.max_drawdown:.3f}")
            tprint_info(f"   Sharpe ratio: {result.sharpe_ratio:.3f}")
            tprint_info(f"   Trading viable: {viable}")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Trading viability validation failed: {e}")
            result.win_rate = 0.0
            result.profit_factor = 0.0
            result.max_drawdown = 1.0
            result.sharpe_ratio = 0.0
            return result
    
    def _validate_performance_metrics(self,
                                    model: Any,
                                    X: np.ndarray,
                                    y: np.ndarray,
                                    market_data: Optional[np.ndarray],
                                    regime_predictions: Optional[np.ndarray],
                                    result: EconomicValidationResult) -> EconomicValidationResult:
        """Validate performance metrics."""
        try:
            tprint_debug("📊 Validating performance metrics...")
            
            # Make predictions
            y_pred = model.predict(X)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(y, y_pred)
            
            result.performance_metrics = performance_metrics
            result.accuracy = performance_metrics.get('accuracy', 0.0)
            result.precision = performance_metrics.get('precision', 0.0)
            result.recall = performance_metrics.get('recall', 0.0)
            result.f1_score = performance_metrics.get('f1_score', 0.0)
            
            # Check performance thresholds
            performance_adequate = (
                result.accuracy >= self.config.min_accuracy and
                result.precision >= self.config.min_precision and
                result.recall >= self.config.min_recall and
                result.f1_score >= self.config.min_f1_score
            )
            
            result.performance_metrics['adequate'] = performance_adequate
            
            tprint_info(f"   Accuracy: {result.accuracy:.3f}")
            tprint_info(f"   Precision: {result.precision:.3f}")
            tprint_info(f"   Recall: {result.recall:.3f}")
            tprint_info(f"   F1 Score: {result.f1_score:.3f}")
            tprint_info(f"   Performance adequate: {performance_adequate}")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Performance metrics validation failed: {e}")
            result.accuracy = 0.0
            result.precision = 0.0
            result.recall = 0.0
            result.f1_score = 0.0
            return result
    
    def _validate_risk_adjusted_metrics(self,
                                      model: Any,
                                      X: np.ndarray,
                                      y: np.ndarray,
                                      market_data: Optional[np.ndarray],
                                      regime_predictions: Optional[np.ndarray],
                                      result: EconomicValidationResult) -> EconomicValidationResult:
        """Validate risk-adjusted metrics."""
        try:
            tprint_debug("⚠️ Validating risk-adjusted metrics...")
            
            # Make predictions
            y_pred = model.predict(X)
            
            # Calculate risk-adjusted metrics
            risk_metrics = self._calculate_risk_adjusted_metrics(y, y_pred, market_data)
            
            result.risk_adjusted_metrics = risk_metrics
            result.var_95 = risk_metrics.get('var_95', 0.0)
            result.cvar_95 = risk_metrics.get('cvar_95', 0.0)
            result.max_drawdown_duration = risk_metrics.get('max_drawdown_duration', 0)
            
            tprint_info(f"   VaR 95%: {result.var_95:.3f}")
            tprint_info(f"   CVaR 95%: {result.cvar_95:.3f}")
            tprint_info(f"   Max drawdown duration: {result.max_drawdown_duration}")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Risk-adjusted metrics validation failed: {e}")
            result.var_95 = 0.0
            result.cvar_95 = 0.0
            result.max_drawdown_duration = 0
            return result
    
    def _validate_trading_metrics(self,
                                model: Any,
                                X: np.ndarray,
                                y: np.ndarray,
                                market_data: Optional[np.ndarray],
                                regime_predictions: Optional[np.ndarray],
                                result: EconomicValidationResult) -> EconomicValidationResult:
        """Validate trading metrics."""
        try:
            tprint_debug("📈 Validating trading metrics...")
            
            # Make predictions
            y_pred = model.predict(X)
            
            # Calculate comprehensive trading metrics
            trading_metrics = self._calculate_comprehensive_trading_metrics(
                y, y_pred, market_data, regime_predictions
            )
            
            result.trading_viability.update(trading_metrics)
            
            tprint_info(f"   Total return: {trading_metrics.get('total_return', 0.0):.3f}")
            tprint_info(f"   Volatility: {trading_metrics.get('volatility', 0.0):.3f}")
            tprint_info(f"   Calmar ratio: {trading_metrics.get('calmar_ratio', 0.0):.3f}")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Trading metrics validation failed: {e}")
            return result
    
    def _add_comprehensive_validation(self,
                                    model: Any,
                                    X: np.ndarray,
                                    y: np.ndarray,
                                    market_data: Optional[np.ndarray],
                                    regime_predictions: Optional[np.ndarray],
                                    result: EconomicValidationResult) -> EconomicValidationResult:
        """Add comprehensive validation for advanced/comprehensive levels."""
        try:
            tprint_debug("🔬 Adding comprehensive validation...")
            
            # Economic indicator correlation
            if self.config.enable_economic_indicators and market_data is not None:
                result.economic_indicator_correlation = self._calculate_economic_indicator_correlation(
                    market_data, regime_predictions
                )
            
            # Regime-specific analysis
            if self.config.enable_regime_specific_analysis and regime_predictions is not None:
                result.regime_analysis = self._analyze_regime_specific_performance(
                    y, model.predict(X), regime_predictions
                )
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Comprehensive validation failed: {e}")
            return result
    
    def _calculate_trading_metrics(self,
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 market_data: Optional[np.ndarray]) -> Dict[str, float]:
        """Calculate trading metrics."""
        try:
            # Simulate returns based on predictions
            if market_data is not None and market_data.shape[1] >= 4:
                returns = self._simulate_trading_returns(market_data, y_pred)
            else:
                # Use prediction accuracy as proxy for returns
                returns = y_pred - y_true
            
            if len(returns) == 0:
                return {
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'max_drawdown': 1.0,
                    'sharpe_ratio': 0.0,
                    'total_return': 0.0,
                    'volatility': 0.0
                }
            
            # Calculate metrics
            win_rate = np.mean(returns > 0)
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            
            profit_factor = (
                np.sum(positive_returns) / abs(np.sum(negative_returns))
                if len(negative_returns) > 0 and np.sum(negative_returns) != 0
                else float('inf') if len(positive_returns) > 0 else 0.0
            )
            
            max_drawdown = self._calculate_max_drawdown(returns)
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            total_return = np.prod(1 + returns) - 1
            volatility = np.std(returns)
            
            return {
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'total_return': total_return,
                'volatility': volatility
            }
            
        except Exception as e:
            tprint_warning(f"⚠️ Trading metrics calculation failed: {e}")
            return {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 1.0,
                'sharpe_ratio': 0.0,
                'total_return': 0.0,
                'volatility': 0.0
            }
    
    def _calculate_performance_metrics(self,
                                     y_true: np.ndarray,
                                     y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics."""
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
        except Exception as e:
            tprint_warning(f"⚠️ Performance metrics calculation failed: {e}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
    
    def _calculate_risk_adjusted_metrics(self,
                                       y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       market_data: Optional[np.ndarray]) -> Dict[str, float]:
        """Calculate risk-adjusted metrics."""
        try:
            # Simulate returns
            if market_data is not None and market_data.shape[1] >= 4:
                returns = self._simulate_trading_returns(market_data, y_pred)
            else:
                returns = y_pred - y_true
            
            if len(returns) == 0:
                return {
                    'var_95': 0.0,
                    'cvar_95': 0.0,
                    'max_drawdown_duration': 0
                }
            
            # Calculate VaR and CVaR
            var_95 = np.percentile(returns, 5)  # 5th percentile for 95% VaR
            cvar_95 = np.mean(returns[returns <= var_95])
            
            # Calculate max drawdown duration
            max_dd_duration = self._calculate_max_drawdown_duration(returns)
            
            return {
                'var_95': var_95,
                'cvar_95': cvar_95,
                'max_drawdown_duration': max_dd_duration
            }
            
        except Exception as e:
            tprint_warning(f"⚠️ Risk-adjusted metrics calculation failed: {e}")
            return {
                'var_95': 0.0,
                'cvar_95': 0.0,
                'max_drawdown_duration': 0
            }
    
    def _calculate_comprehensive_trading_metrics(self,
                                               y_true: np.ndarray,
                                               y_pred: np.ndarray,
                                               market_data: Optional[np.ndarray],
                                               regime_predictions: Optional[np.ndarray]) -> Dict[str, float]:
        """Calculate comprehensive trading metrics."""
        try:
            # Basic trading metrics
            basic_metrics = self._calculate_trading_metrics(y_true, y_pred, market_data)
            
            # Additional metrics
            if market_data is not None and market_data.shape[1] >= 4:
                returns = self._simulate_trading_returns(market_data, y_pred)
                
                if len(returns) > 0:
                    # Calmar ratio
                    calmar_ratio = (
                        np.mean(returns) / basic_metrics['max_drawdown']
                        if basic_metrics['max_drawdown'] > 0
                        else 0.0
                    )
                    
                    # Sortino ratio
                    downside_returns = returns[returns < 0]
                    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
                    sortino_ratio = (
                        np.mean(returns) / downside_std
                        if downside_std > 0
                        else 0.0
                    )
                    
                    # Information ratio
                    benchmark_returns = np.random.normal(0, 0.01, len(returns))  # Placeholder benchmark
                    excess_returns = returns - benchmark_returns
                    information_ratio = (
                        np.mean(excess_returns) / np.std(excess_returns)
                        if np.std(excess_returns) > 0
                        else 0.0
                    )
                    
                    basic_metrics.update({
                        'calmar_ratio': calmar_ratio,
                        'sortino_ratio': sortino_ratio,
                        'information_ratio': information_ratio
                    })
            
            return basic_metrics
            
        except Exception as e:
            tprint_warning(f"⚠️ Comprehensive trading metrics calculation failed: {e}")
            return {}
    
    def _calculate_economic_indicator_correlation(self,
                                                market_data: np.ndarray,
                                                regime_predictions: Optional[np.ndarray]) -> Dict[str, float]:
        """Calculate correlation with economic indicators."""
        try:
            # Placeholder implementation - would integrate with actual economic data
            correlations = {
                'gdp_correlation': np.random.uniform(-0.3, 0.3),
                'inflation_correlation': np.random.uniform(-0.2, 0.2),
                'interest_rate_correlation': np.random.uniform(-0.4, 0.4),
                'vix_correlation': np.random.uniform(-0.5, 0.1)
            }
            
            return correlations
            
        except Exception as e:
            tprint_warning(f"⚠️ Economic indicator correlation calculation failed: {e}")
            return {}
    
    def _analyze_regime_specific_performance(self,
                                           y_true: np.ndarray,
                                           y_pred: np.ndarray,
                                           regime_predictions: np.ndarray) -> Dict[str, Any]:
        """Analyze performance by regime."""
        try:
            unique_regimes = np.unique(regime_predictions)
            regime_analysis = {}
            
            for regime in unique_regimes:
                regime_mask = regime_predictions == regime
                regime_y_true = y_true[regime_mask]
                regime_y_pred = y_pred[regime_mask]
                
                if len(regime_y_true) > 0:
                    regime_accuracy = np.mean(regime_y_true == regime_y_pred)
                    regime_analysis[f'regime_{regime}'] = {
                        'accuracy': regime_accuracy,
                        'n_samples': len(regime_y_true),
                        'performance': 'good' if regime_accuracy > 0.6 else 'poor'
                    }
            
            return regime_analysis
            
        except Exception as e:
            tprint_warning(f"⚠️ Regime-specific analysis failed: {e}")
            return {}
    
    def _simulate_trading_returns(self,
                                market_data: np.ndarray,
                                predictions: np.ndarray) -> np.ndarray:
        """Simulate trading returns based on predictions."""
        try:
            if market_data.shape[1] < 4:
                return np.array([])
            
            close_prices = market_data[:, 3]
            returns = []
            
            for i in range(1, len(predictions)):
                if predictions[i] != predictions[i-1]:
                    # Regime change - simulate trade
                    trade_return = (close_prices[i] - close_prices[i-1]) / close_prices[i-1]
                    returns.append(trade_return)
            
            return np.array(returns)
            
        except Exception:
            return np.array([])
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        try:
            if len(returns) == 0:
                return 0.0
            
            cumulative = np.cumprod(1 + returns)
            peak = cumulative[0]
            max_dd = 0.0
            
            for value in cumulative:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                max_dd = max(max_dd, dd)
            
            return max_dd
            
        except Exception:
            return 0.0
    
    def _calculate_max_drawdown_duration(self, returns: np.ndarray) -> int:
        """Calculate maximum drawdown duration."""
        try:
            if len(returns) == 0:
                return 0
            
            cumulative = np.cumprod(1 + returns)
            peak = cumulative[0]
            current_dd_duration = 0
            max_dd_duration = 0
            
            for value in cumulative:
                if value > peak:
                    peak = value
                    current_dd_duration = 0
                else:
                    current_dd_duration += 1
                    max_dd_duration = max(max_dd_duration, current_dd_duration)
            
            return max_dd_duration
            
        except Exception:
            return 0
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        try:
            if len(returns) == 0 or np.std(returns) == 0:
                return 0.0
            
            excess_returns = returns - self.config.risk_free_rate / 252
            return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            
        except Exception:
            return 0.0


# Convenience functions
def create_economic_validator(config: Optional[EconomicValidationConfig] = None) -> EconomicValidator:
    """Create an economic validator with default configuration."""
    if config is None:
        config = EconomicValidationConfig()
    return EconomicValidator(config)


def quick_economic_validation(model: Any,
                            X: np.ndarray,
                            y: np.ndarray,
                            market_data: Optional[np.ndarray] = None,
                            config: Optional[EconomicValidationConfig] = None) -> EconomicValidationResult:
    """Quick economic validation with default settings."""
    validator = create_economic_validator(config)
    return validator.validate(model, X, y, market_data)


def validate_economic_significance(model: Any,
                                 X: np.ndarray,
                                 y: np.ndarray,
                                 market_data: np.ndarray,
                                 regime_predictions: np.ndarray) -> EconomicValidationResult:
    """Validate economic significance specifically."""
    config = EconomicValidationConfig(
        validation_type=EconomicValidationType.SIGNIFICANCE,
        validation_level=ValidationLevel.INTERMEDIATE
    )
    validator = EconomicValidator(config)
    return validator.validate(model, X, y, market_data, regime_predictions)


def validate_trading_viability(model: Any,
                              X: np.ndarray,
                              y: np.ndarray,
                              market_data: np.ndarray) -> EconomicValidationResult:
    """Validate trading viability specifically."""
    config = EconomicValidationConfig(
        validation_type=EconomicValidationType.VIABILITY,
        validation_level=ValidationLevel.INTERMEDIATE
    )
    validator = EconomicValidator(config)
    return validator.validate(model, X, y, market_data)