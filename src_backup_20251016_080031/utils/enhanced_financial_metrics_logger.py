"""
Enhanced Financial Metrics Logger with Per-HMM Regime Logging and Fail-Fast Validation

This module extends the existing financial_metrics_logger to provide:
1. Enhanced per-HMM regime logging for steps after HMM-based data splitting
2. Fail-fast validation to prevent empty running or important degradation
3. Comprehensive regime-specific metrics tracking
4. Automatic regime detection and validation
"""

import logging


from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import threading
from contextlib import contextmanager

# Import the base financial metrics logger
try:
    from src.utils.financial_metrics_logger import (
        FinancialMetricsLogger, 
        FinancialMetric, 
        TradingPerformanceMetrics,
        get_financial_metrics_logger
    )
    BASE_LOGGER_AVAILABLE = True
except ImportError:
    BASE_LOGGER_AVAILABLE = False
    FinancialMetricsLogger = None
    FinancialMetric = None
    TradingPerformanceMetrics = None
    get_financial_metrics_logger = None

# Import the main logger for fallback
try:
    from src.utils.logger import system_logger, get_logger
    import time

except ImportError:
    system_logger = None
    get_logger = lambda name: logging.getLogger(name)

@dataclass
class RegimeValidationResult:
    """Result of regime validation checks."""
    is_valid: bool
    regime_count: int
    regime_ids: List[str]
    missing_regimes: List[str]
    empty_regimes: List[str]
    validation_errors: List[str]
    quality_score: float

@dataclass
class FailFastValidationResult:
    """Result of comprehensive fail-fast validation checks."""
    should_fail: bool
    failure_reason: Optional[str]
    warnings: List[str]
    critical_issues: List[str]
    degradation_detected: bool
    empty_running_detected: bool
    validation_categories: Dict[str, bool] = None  # Category-wise validation results
    data_quality_score: float = 0.0
    performance_score: float = 0.0
    model_quality_score: float = 0.0
    feature_quality_score: float = 0.0
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.validation_categories is None:
            self.validation_categories = {}
        if self.recommendations is None:
            self.recommendations = []

class EnhancedFinancialMetricsLogger:
    """
    Enhanced financial metrics logger with per-HMM regime logging and fail-fast validation.
    
    Features:
    - Per-HMM regime logging for steps after HMM-based data splitting
    - Fail-fast validation to prevent empty running or important degradation
    - Automatic regime detection and validation
    - Comprehensive regime-specific metrics tracking
    - Integration with existing financial_metrics_logger
    """
    
    def __init__(self,
                 log_dir: Optional[str] = None,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 enable_csv: bool = True,
                 enable_json: bool = True,
                 fail_fast_enabled: bool = True,
                 regime_validation_enabled: bool = True,
                 min_regime_samples: int = 100,
                 max_regime_imbalance: float = 0.8):
        """
        Initialize the enhanced financial metrics logger.
        
        Args:
            log_dir: Directory for financial metrics logs
            enable_console: Enable console output
            enable_file: Enable file logging
            enable_csv: Enable CSV export
            enable_json: Enable JSON export
            fail_fast_enabled: Enable fail-fast validation
            regime_validation_enabled: Enable regime validation
            min_regime_samples: Minimum samples required per regime
            max_regime_imbalance: Maximum allowed regime imbalance ratio
        """
        if log_dir is None:
            # Use absolute path based on project root
            project_root = Path(__file__).parent.parent.parent  # src/utils -> src -> project root
            log_dir = str(project_root / "logs" / "financial_metrics")

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_csv = enable_csv
        self.enable_json = enable_json
        self.fail_fast_enabled = fail_fast_enabled
        self.regime_validation_enabled = regime_validation_enabled
        self.min_regime_samples = min_regime_samples
        self.max_regime_imbalance = max_regime_imbalance
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize base logger if available
        if BASE_LOGGER_AVAILABLE:
            self.base_logger = get_financial_metrics_logger()
        else:
            self.base_logger = None
        
        # Initialize enhanced logger
        self._setup_enhanced_logger()
        
        # Regime tracking
        self.regime_registry = {}
        self.regime_validation_history = []
        self.fail_fast_history = []
        
        # Fallback to main logger if available
        self.fallback_logger = system_logger.getChild('EnhancedFinancialMetrics') if system_logger else None
    
    def _setup_enhanced_logger(self):
        """Setup the enhanced financial metrics logger."""
        # Enhanced financial metrics logger via central system
        self.logger = get_logger('EnhancedFinancialMetrics')
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter for file outputs (console handled by central logger)
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler with timestamp
        if self.enable_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = self.log_dir / f'enhanced_financial_metrics_{timestamp}.log'
            
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_file, 
                maxBytes=50 * 1024 * 1024,  # 50MB
                backupCount=10
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Propagate to central logger handlers for console output
        self.logger.propagate = True
    
    def validate_regime_data(self, 
                           data: pd.DataFrame, 
                           regime_column: str = 'composite_cluster_id',
                           step_name: str = 'unknown') -> RegimeValidationResult:
        """
        Validate regime data for fail-fast behavior.
        
        Args:
            data: DataFrame containing regime data
            regime_column: Name of the regime column
            step_name: Name of the current step
            
        Returns:
            RegimeValidationResult with validation details
        """
        validation_errors = []
        warnings = []
        
        try:
            # Check if regime column exists
            if regime_column not in data.columns:
                validation_errors.append(f"Regime column '{regime_column}' not found in data")
                return RegimeValidationResult(
                    is_valid=False,
                    regime_count=0,
                    regime_ids=[],
                    missing_regimes=[],
                    empty_regimes=[],
                    validation_errors=validation_errors,
                    quality_score=0.0
                )
            
            # Get regime data
            regime_data = data[regime_column].dropna()
            
            if regime_data.empty:
                validation_errors.append("No valid regime data found")
                return RegimeValidationResult(
                    is_valid=False,
                    regime_count=0,
                    regime_ids=[],
                    missing_regimes=[],
                    empty_regimes=[],
                    validation_errors=validation_errors,
                    quality_score=0.0
                )
            
            # Get unique regimes
            unique_regimes = regime_data.unique()
            regime_ids = [str(regime) for regime in unique_regimes]
            regime_count = len(unique_regimes)
            
            # Check minimum regime count
            if regime_count < 2:
                validation_errors.append(f"Insufficient regime diversity: only {regime_count} regimes found")
            
            # Check regime sample sizes
            regime_counts = regime_data.value_counts()
            empty_regimes = []
            small_regimes = []
            
            for regime_id in regime_ids:
                count = regime_counts.get(regime_id, 0)
                if count == 0:
                    empty_regimes.append(regime_id)
                elif count < self.min_regime_samples:
                    small_regimes.append(regime_id)
                    warnings.append(f"Regime {regime_id} has only {count} samples (minimum: {self.min_regime_samples})")
            
            # Check regime imbalance
            if len(regime_counts) > 0:
                max_count = regime_counts.max()
                min_count = regime_counts.min()
                imbalance_ratio = min_count / max_count if max_count > 0 else 0
                
                if imbalance_ratio < (1 - self.max_regime_imbalance):
                    warnings.append(f"Severe regime imbalance detected: ratio {imbalance_ratio:.3f} (max allowed: {1 - self.max_regime_imbalance:.3f})")
            
            # Calculate quality score
            quality_score = 1.0
            if validation_errors:
                quality_score -= len(validation_errors) * 0.3
            if warnings:
                quality_score -= len(warnings) * 0.1
            if empty_regimes:
                quality_score -= len(empty_regimes) * 0.2
            if small_regimes:
                quality_score -= len(small_regimes) * 0.1
            
            quality_score = max(0.0, quality_score)
            
            # Determine if valid
            is_valid = len(validation_errors) == 0 and len(empty_regimes) == 0
            
            result = RegimeValidationResult(
                is_valid=is_valid,
                regime_count=regime_count,
                regime_ids=regime_ids,
                missing_regimes=[],
                empty_regimes=empty_regimes,
                validation_errors=validation_errors,
                quality_score=quality_score
            )
            
            # Store validation history
            self.regime_validation_history.append({
                'timestamp': datetime.now().isoformat(),
                'step_name': step_name,
                'result': result,
                'warnings': warnings
            })
            
            return result
            
        except Exception as e:
            validation_errors.append(f"Regime validation failed: {str(e)}")
            return RegimeValidationResult(
                is_valid=False,
                regime_count=0,
                regime_ids=[],
                missing_regimes=[],
                empty_regimes=[],
                validation_errors=validation_errors,
                quality_score=0.0
            )
    
    def validate_fail_fast_conditions(self, 
                                    data: pd.DataFrame,
                                    step_name: str,
                                    expected_regimes: Optional[List[str]] = None,
                                    min_data_quality: float = 0.7,
                                    check_empty_running: bool = True,
                                    check_degradation: bool = True,
                                    additional_context: Optional[Dict[str, Any]] = None) -> FailFastValidationResult:
        """
        Perform comprehensive fail-fast validation covering all important aspects.
        
        Args:
            data: DataFrame to validate
            step_name: Name of the current step
            expected_regimes: List of expected regime IDs
            min_data_quality: Minimum required data quality score
            check_empty_running: Whether to check for empty running conditions
            check_degradation: Whether to check for performance degradation
            additional_context: Additional context for validation (model metrics, performance data, etc.)
            
        Returns:
            FailFastValidationResult with comprehensive validation details
        """
        warnings = []
        critical_issues = []
        should_fail = False
        failure_reason = None
        empty_running_detected = False
        degradation_detected = False
        validation_categories = {}
        recommendations = []
        
        try:
            # 1. DATA QUALITY VALIDATION
            data_quality_score = self._validate_data_quality_comprehensive(data, warnings, critical_issues)
            validation_categories['data_quality'] = data_quality_score >= 0.7
            
            # 2. REGIME VALIDATION (for post-HMM steps)
            regime_quality_score = 1.0
            if step_name and self._is_post_hmm_step(step_name):
                regime_validation = self.validate_regime_data(data, step_name=step_name)
                regime_quality_score = self._validate_regime_quality_comprehensive(
                    regime_validation, expected_regimes, warnings, critical_issues
                )
                validation_categories['regime_quality'] = regime_validation.is_valid and regime_quality_score >= 0.5
            else:
                validation_categories['regime_quality'] = True  # Not applicable for pre-HMM steps
            
            # 3. PERFORMANCE VALIDATION
            performance_score = self._validate_performance_comprehensive(
                additional_context, check_degradation, warnings, critical_issues
            )
            validation_categories['performance'] = performance_score >= 0.6
            
            # 4. MODEL QUALITY VALIDATION
            model_quality_score = self._validate_model_quality_comprehensive(
                additional_context, warnings, critical_issues
            )
            validation_categories['model_quality'] = model_quality_score >= 0.6
            
            # 5. FEATURE QUALITY VALIDATION
            feature_quality_score = self._validate_feature_quality_comprehensive(
                data, additional_context, warnings, critical_issues
            )
            validation_categories['feature_quality'] = feature_quality_score >= 0.6
            
            # 6. EXECUTION ENVIRONMENT VALIDATION
            execution_score = self._validate_execution_environment_comprehensive(
                additional_context, warnings, critical_issues
            )
            validation_categories['execution_environment'] = execution_score >= 0.7
            
            # 7. BUSINESS LOGIC VALIDATION
            business_logic_score = self._validate_business_logic_comprehensive(
                data, step_name, additional_context, warnings, critical_issues
            )
            validation_categories['business_logic'] = business_logic_score >= 0.7
            
            # 8. EMPTY RUNNING DETECTION
            if check_empty_running:
                empty_running_detected = self._detect_empty_running_comprehensive(
                    data, warnings, critical_issues
                )
            
            # Calculate overall quality score
            applicable_scores = [score for score in [
                data_quality_score, regime_quality_score, performance_score,
                model_quality_score, feature_quality_score, execution_score, business_logic_score
            ] if score is not None]
            
            overall_score = sum(applicable_scores) / len(applicable_scores) if applicable_scores else 0.0
            
            # Determine degradation
            if overall_score < 0.5:
                degradation_detected = True
                critical_issues.append(f"Overall system quality below threshold: {overall_score:.2f}")
            
            # Generate recommendations
            recommendations = self._generate_recommendations_comprehensive(
                validation_categories, overall_score, warnings, critical_issues
            )
            
            # Determine if we should fail fast
            should_fail = (
                empty_running_detected or 
                (degradation_detected and self.fail_fast_enabled) or
                len(critical_issues) > 0 or
                overall_score < 0.4 or
                sum(validation_categories.values()) < len(validation_categories) * 0.6
            )
            
            # Set failure reason
            if should_fail:
                if empty_running_detected:
                    failure_reason = "Empty running detected - insufficient data"
                elif degradation_detected:
                    failure_reason = "Performance degradation detected"
                elif critical_issues:
                    failure_reason = f"Critical issues: {', '.join(critical_issues[:2])}"
                elif overall_score < 0.4:
                    failure_reason = f"Overall system quality critically low: {overall_score:.2f}"
                else:
                    failure_reason = "Multiple validation categories failed"
            
            result = FailFastValidationResult(
                should_fail=should_fail,
                failure_reason=failure_reason,
                warnings=warnings,
                critical_issues=critical_issues,
                degradation_detected=degradation_detected,
                empty_running_detected=empty_running_detected,
                validation_categories=validation_categories,
                data_quality_score=data_quality_score,
                performance_score=performance_score,
                model_quality_score=model_quality_score,
                feature_quality_score=feature_quality_score,
                recommendations=recommendations
            )
            
            # Store fail-fast history
            self.fail_fast_history.append({
                'timestamp': datetime.now().isoformat(),
                'step_name': step_name,
                'result': result,
                'overall_score': overall_score,
                'validation_categories': validation_categories
            })
            
            return result
            
        except Exception as e:
            should_fail = True
            failure_reason = f"Fail-fast validation error: {str(e)}"
            critical_issues.append(f"Validation error: {str(e)}")
            
            return FailFastValidationResult(
                should_fail=should_fail,
                failure_reason=failure_reason,
                warnings=[],
                critical_issues=critical_issues,
                degradation_detected=True,
                empty_running_detected=True,
                validation_categories={},
                recommendations=[f"Fix validation error: {str(e)}"]
            )
    
    def _is_post_hmm_step(self, step_name: str) -> bool:
        """Check if this is a post-HMM step (step number > 8)."""
        try:
            if 'step' in step_name.lower():
                step_num_str = step_name.lower().split('step')[1]
                # Extract numeric part
                step_num = int(''.join(filter(str.isdigit, step_num_str)))
                return step_num > 8
        except:
            pass
        return False
    
    def _validate_data_quality_comprehensive(self, data: pd.DataFrame, warnings: List[str], critical_issues: List[str]) -> float:
        """Comprehensive data quality validation."""
        if data is None or data.empty:
            critical_issues.append("Data is None or empty")
            return 0.0
        
        score = 1.0
        
        # Check for excessive NaN values
        nan_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        if nan_ratio > 0.5:
            critical_issues.append(f"Excessive NaN values: {nan_ratio:.3f}")
            score -= 0.5
        elif nan_ratio > 0.2:
            warnings.append(f"High NaN ratio: {nan_ratio:.3f}")
            score -= 0.2
        
        # Check for constant columns
        constant_columns = []
        for col in data.columns:
            if data[col].nunique() <= 1:
                constant_columns.append(col)
        
        if len(constant_columns) > len(data.columns) * 0.3:
            critical_issues.append(f"Too many constant columns: {len(constant_columns)}/{len(data.columns)}")
            score -= 0.4
        elif constant_columns:
            warnings.append(f"Constant columns detected: {constant_columns}")
            score -= 0.1
        
        # Check for data types
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < len(data.columns) * 0.5:
            warnings.append(f"Low ratio of numeric columns: {len(numeric_cols)}/{len(data.columns)}")
            score -= 0.1
        
        # Check for outliers (basic check)
        if len(numeric_cols) > 0:
            outlier_ratio = 0
            for col in numeric_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
                outlier_ratio += outliers / len(data)
            
            outlier_ratio /= len(numeric_cols)
            if outlier_ratio > 0.3:
                warnings.append(f"High outlier ratio: {outlier_ratio:.3f}")
                score -= 0.1
        
        return max(0.0, score)
    
    def _validate_regime_quality_comprehensive(self, regime_validation: RegimeValidationResult, 
                                             expected_regimes: Optional[List[str]], 
                                             warnings: List[str], critical_issues: List[str]) -> float:
        """Comprehensive regime quality validation."""
        score = regime_validation.quality_score
        
        if not regime_validation.is_valid:
            critical_issues.extend(regime_validation.validation_errors)
            score = 0.0
        
        if regime_validation.regime_count < 2:
            critical_issues.append("Insufficient regime diversity")
            score = 0.0
        
        if expected_regimes:
            missing_regimes = set(expected_regimes) - set(regime_validation.regime_ids)
            if missing_regimes:
                critical_issues.append(f"Missing expected regimes: {list(missing_regimes)}")
                score -= 0.3
        
        if regime_validation.empty_regimes:
            warnings.append(f"Empty regimes detected: {regime_validation.empty_regimes}")
            score -= 0.2
        
        return max(0.0, score)
    
    def _validate_performance_comprehensive(self, additional_context: Optional[Dict[str, Any]], 
                                          check_degradation: bool, warnings: List[str], critical_issues: List[str]) -> float:
        """Comprehensive performance validation."""
        score = 1.0
        
        if not additional_context:
            return score
        
        # Check for performance degradation patterns
        if check_degradation and len(self.fail_fast_history) > 0:
            recent_failures = [h for h in self.fail_fast_history[-5:] if h.get('result', {}).get('should_fail', False)]
            if len(recent_failures) >= 3:
                critical_issues.append("Performance degradation: multiple recent failures")
                score = 0.0
            elif len(recent_failures) >= 2:
                warnings.append("Performance degradation: multiple recent failures")
                score -= 0.3
        
        # Check model performance metrics
        if 'model_performance' in additional_context:
            perf = additional_context['model_performance']
            accuracy = perf.get('accuracy', 0)
            if accuracy < 0.5:
                critical_issues.append(f"Model accuracy too low: {accuracy:.3f}")
                score -= 0.5
            elif accuracy < 0.7:
                warnings.append(f"Model accuracy below threshold: {accuracy:.3f}")
                score -= 0.2
        
        # Check execution time
        if 'execution_time' in additional_context:
            exec_time = additional_context['execution_time']
            if exec_time > 3600:  # 1 hour
                warnings.append(f"Long execution time: {exec_time:.1f}s")
                score -= 0.1
        
        return max(0.0, score)
    
    def _validate_model_quality_comprehensive(self, additional_context: Optional[Dict[str, Any]], 
                                            warnings: List[str], critical_issues: List[str]) -> float:
        """Comprehensive model quality validation."""
        score = 1.0
        
        if not additional_context:
            return score
        
        # Check model convergence
        if 'model_convergence' in additional_context:
            convergence = additional_context['model_convergence']
            if not convergence:
                critical_issues.append("Model did not converge")
                score = 0.0
        
        # Check model metrics
        if 'model_metrics' in additional_context:
            metrics = additional_context['model_metrics']
            loss = metrics.get('loss', float('inf'))
            if loss > 10.0:
                critical_issues.append(f"Model loss too high: {loss:.3f}")
                score -= 0.5
            elif loss > 5.0:
                warnings.append(f"Model loss high: {loss:.3f}")
                score -= 0.2
        
        # Check for overfitting
        if 'training_accuracy' in additional_context and 'validation_accuracy' in additional_context:
            train_acc = additional_context['training_accuracy']
            val_acc = additional_context['validation_accuracy']
            if train_acc - val_acc > 0.2:
                warnings.append(f"Potential overfitting: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")
                score -= 0.2
        
        return max(0.0, score)
    
    def _validate_feature_quality_comprehensive(self, data: pd.DataFrame, additional_context: Optional[Dict[str, Any]], 
                                              warnings: List[str], critical_issues: List[str]) -> float:
        """Comprehensive feature quality validation."""
        score = 1.0
        
        if data is None or data.empty:
            return 0.0
        
        # Check feature count
        if len(data.columns) < 5:
            warnings.append(f"Low feature count: {len(data.columns)}")
            score -= 0.2
        
        # Check feature correlation
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.95:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            if len(high_corr_pairs) > len(numeric_cols) * 0.3:
                warnings.append(f"High feature correlation detected: {len(high_corr_pairs)} pairs")
                score -= 0.2
        
        # Check feature importance if available
        if additional_context and 'feature_importance' in additional_context:
            importance = additional_context['feature_importance']
            if isinstance(importance, dict):
                # Check if any feature has extremely low importance
                min_importance = min(importance.values())
                if min_importance < 0.01:
                    warnings.append("Some features have very low importance")
                    score -= 0.1
        
        return max(0.0, score)
    
    def _validate_execution_environment_comprehensive(self, additional_context: Optional[Dict[str, Any]], 
                                                    warnings: List[str], critical_issues: List[str]) -> float:
        """Comprehensive execution environment validation."""
        score = 1.0
        
        if not additional_context:
            return score
        
        # Check memory usage
        if 'memory_usage_mb' in additional_context:
            memory = additional_context['memory_usage_mb']
            if memory > 8000:  # 8GB
                warnings.append(f"High memory usage: {memory:.1f}MB")
                score -= 0.2
        
        # Check CPU usage
        if 'cpu_usage_percent' in additional_context:
            cpu = additional_context['cpu_usage_percent']
            if cpu > 90:
                warnings.append(f"High CPU usage: {cpu:.1f}%")
                score -= 0.1
        
        # Check disk space
        if 'disk_usage_percent' in additional_context:
            disk = additional_context['disk_usage_percent']
            if disk > 90:
                critical_issues.append(f"Low disk space: {disk:.1f}%")
                score -= 0.5
        
        # Check for errors in context
        if 'errors' in additional_context:
            errors = additional_context['errors']
            if len(errors) > 0:
                critical_issues.append(f"Execution errors detected: {len(errors)}")
                score -= 0.3
        
        return max(0.0, score)
    
    def _validate_business_logic_comprehensive(self, data: pd.DataFrame, step_name: str, 
                                             additional_context: Optional[Dict[str, Any]], 
                                             warnings: List[str], critical_issues: List[str]) -> float:
        """Comprehensive business logic validation."""
        score = 1.0
        
        # Check for required columns based on step
        if data is not None and not data.empty:
            required_columns = self._get_required_columns_for_step(step_name)
            missing_columns = set(required_columns) - set(data.columns)
            if missing_columns:
                critical_issues.append(f"Missing required columns for {step_name}: {list(missing_columns)}")
                score -= 0.5
        
        # Check for business rule violations
        if additional_context and 'business_rules' in additional_context:
            rules = additional_context['business_rules']
            violations = rules.get('violations', [])
            if violations:
                critical_issues.append(f"Business rule violations: {violations}")
                score -= 0.3
        
        # Check for data consistency
        if data is not None and not data.empty:
            # Check for negative prices (if price columns exist)
            price_columns = [col for col in data.columns if 'price' in col.lower()]
            for col in price_columns:
                if (data[col] < 0).any():
                    warnings.append(f"Negative prices detected in {col}")
                    score -= 0.1
        
        return max(0.0, score)
    
    def _detect_empty_running_comprehensive(self, data: pd.DataFrame, warnings: List[str], critical_issues: List[str]) -> bool:
        """Comprehensive empty running detection."""
        if data is None or data.empty:
            critical_issues.append("Data is None or empty")
            return True
        
        if len(data) < 10:
            critical_issues.append(f"Dataset too small: {len(data)} samples")
            return True
        
        # Check if all values are the same
        if data.nunique().sum() <= len(data.columns):
            critical_issues.append("Empty running: insufficient data variation")
            return True
        
        # Check for suspicious patterns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Check if all numeric values are the same
            for col in numeric_cols:
                if data[col].nunique() == 1:
                    warnings.append(f"Column {col} has no variation")
        
        return False
    
    def _get_required_columns_for_step(self, step_name: str) -> List[str]:
        """Get required columns for a specific step."""
        step_requirements = {
            'step09': ['composite_cluster_id', 'features'],
            'step10': ['composite_cluster_id', 'features'],
            'step11': ['composite_cluster_id', 'features'],
            'step12': ['composite_cluster_id', 'features'],
            'step13': ['composite_cluster_id', 'features'],
            'step14': ['composite_cluster_id', 'features'],
            'step15': ['composite_cluster_id', 'features'],
            'step16': ['composite_cluster_id', 'features'],
            'step17': ['composite_cluster_id', 'features'],
            'step18': ['composite_cluster_id', 'features'],
            'step19': ['composite_cluster_id', 'features'],
            'step20': ['composite_cluster_id', 'features'],
        }
        
        for step_key, columns in step_requirements.items():
            if step_key in step_name.lower():
                return columns
        
        return []
    
    def _generate_recommendations_comprehensive(self, validation_categories: Dict[str, bool], 
                                              overall_score: float, warnings: List[str], 
                                              critical_issues: List[str]) -> List[str]:
        """Generate comprehensive recommendations based on validation results."""
        recommendations = []
        
        if overall_score < 0.5:
            recommendations.append("Overall system quality is low - consider data preprocessing and feature engineering")
        
        if not validation_categories.get('data_quality', True):
            recommendations.append("Improve data quality by handling missing values and outliers")
        
        if not validation_categories.get('regime_quality', True):
            recommendations.append("Check regime data integrity and ensure proper regime labeling")
        
        if not validation_categories.get('performance', True):
            recommendations.append("Investigate performance degradation and optimize model parameters")
        
        if not validation_categories.get('model_quality', True):
            recommendations.append("Review model training process and ensure proper convergence")
        
        if not validation_categories.get('feature_quality', True):
            recommendations.append("Analyze feature importance and consider feature selection")
        
        if not validation_categories.get('execution_environment', True):
            recommendations.append("Monitor system resources and optimize execution environment")
        
        if not validation_categories.get('business_logic', True):
            recommendations.append("Review business rules and data consistency requirements")
        
        if len(critical_issues) > 0:
            recommendations.append("Address critical issues before proceeding with training")
        
        if len(warnings) > 0:
            recommendations.append("Review warnings and consider preventive measures")
        
        return recommendations
    
    def log_financial_metric_with_regime_validation(self,
                                                  symbol: str,
                                                  exchange: str,
                                                  timeframe: str,
                                                  metric_name: str,
                                                  metric_value: float,
                                                  metric_type: str,
                                                  step_name: str,
                                                  regime_id: Optional[str] = None,
                                                  additional_data: Optional[Dict[str, Any]] = None,
                                                  data: Optional[pd.DataFrame] = None,
                                                  expected_regimes: Optional[List[str]] = None) -> bool:
        """
        Log financial metric with regime validation and fail-fast checks.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            metric_name: Name of the metric
            metric_value: Value of the metric
            metric_type: Type of metric
            step_name: Training step name
            regime_id: Market regime identifier
            additional_data: Additional context data
            data: DataFrame for validation (optional)
            expected_regimes: Expected regime IDs (optional)
            
        Returns:
            True if logging succeeded, False if fail-fast conditions triggered
        """
        with self._lock:
            try:
                # Perform fail-fast validation if enabled
                if self.fail_fast_enabled and data is not None:
                    fail_fast_result = self.validate_fail_fast_conditions(
                        data=data,
                        step_name=step_name,
                        expected_regimes=expected_regimes
                    )
                    
                    if fail_fast_result.should_fail:
                        self.logger.error(f"🚨 FAIL-FAST TRIGGERED for {step_name}")
                        self.logger.error(f"   Reason: {fail_fast_result.failure_reason}")
                        for issue in fail_fast_result.critical_issues:
                            self.logger.error(f"   Critical Issue: {issue}")
                        
                        # Log the failure as a financial metric
                        if self.base_logger:
                            self.base_logger.log_financial_metric(
                                symbol=symbol,
                                exchange=exchange,
                                timeframe=timeframe,
                                metric_name="fail_fast_triggered",
                                metric_value=1.0,
                                metric_type="risk",
                                step_name=step_name,
                                regime_id=regime_id,
                                additional_data={
                                    'failure_reason': fail_fast_result.failure_reason,
                                    'critical_issues': fail_fast_result.critical_issues,
                                    'degradation_detected': fail_fast_result.degradation_detected,
                                    'empty_running_detected': fail_fast_result.empty_running_detected
                                }
                            )
                        
                        return False
                
                # Log the metric using base logger
                if self.base_logger:
                    self.base_logger.log_financial_metric(
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe,
                        metric_name=metric_name,
                        metric_value=metric_value,
                        metric_type=metric_type,
                        step_name=step_name,
                        regime_id=regime_id,
                        additional_data=additional_data
                    )
                
                # Log regime-specific metrics if regime_id is provided
                if regime_id and self.regime_validation_enabled:
                    self._log_regime_specific_metrics(
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe,
                        step_name=step_name,
                        regime_id=regime_id,
                        metric_name=metric_name,
                        metric_value=metric_value,
                        metric_type=metric_type
                    )
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to log financial metric with regime validation: {e}")
                return False
    
    def _log_regime_specific_metrics(self,
                                   symbol: str,
                                   exchange: str,
                                   timeframe: str,
                                   step_name: str,
                                   regime_id: str,
                                   metric_name: str,
                                   metric_value: float,
                                   metric_type: str) -> None:
        """Log regime-specific metrics and tracking."""
        try:
            # Track regime usage
            if regime_id not in self.regime_registry:
                self.regime_registry[regime_id] = {
                    'first_seen': datetime.now().isoformat(),
                    'last_seen': datetime.now().isoformat(),
                    'metric_count': 0,
                    'steps_used': set()
                }
            
            self.regime_registry[regime_id]['last_seen'] = datetime.now().isoformat()
            self.regime_registry[regime_id]['metric_count'] += 1
            self.regime_registry[regime_id]['steps_used'].add(step_name)
            
            # Log regime tracking metric
            if self.base_logger:
                self.base_logger.log_financial_metric(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    metric_name=f"regime_{regime_id}_usage_count",
                    metric_value=float(self.regime_registry[regime_id]['metric_count']),
                    metric_type="regime",
                    step_name=step_name,
                    regime_id=regime_id,
                    additional_data={
                        'regime_tracking': True,
                        'steps_used': list(self.regime_registry[regime_id]['steps_used'])
                    }
                )
            
        except Exception as e:
            self.logger.warning(f"Failed to log regime-specific metrics: {e}")
    
    def log_per_regime_metrics(self,
                              symbol: str,
                              exchange: str,
                              timeframe: str,
                              step_name: str,
                              regime_metrics: Dict[str, Dict[str, Any]],
                              data: Optional[pd.DataFrame] = None) -> bool:
        """
        Log metrics for multiple regimes with validation.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            step_name: Training step name
            regime_metrics: Dictionary of regime_id -> metrics
            data: DataFrame for validation (optional)
            
        Returns:
            True if all logging succeeded, False if any fail-fast conditions triggered
        """
        success = True
        
        try:
            # Validate regime data if provided
            if data is not None and self.regime_validation_enabled:
                regime_validation = self.validate_regime_data(data, step_name=step_name)
                
                if not regime_validation.is_valid:
                    self.logger.error(f"🚨 Regime validation failed for {step_name}")
                    for error in regime_validation.validation_errors:
                        self.logger.error(f"   Error: {error}")
                    
                    if self.fail_fast_enabled:
                        return False
            
            # Log metrics for each regime
            for regime_id, metrics in regime_metrics.items():
                for metric_name, metric_value in metrics.items():
                    metric_success = self.log_financial_metric_with_regime_validation(
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe,
                        metric_name=f"regime_{regime_id}_{metric_name}",
                        metric_value=float(metric_value),
                        metric_type="regime",
                        step_name=step_name,
                        regime_id=str(regime_id),
                        data=data
                    )
                    
                    if not metric_success:
                        success = False
            
            # Log regime summary metrics
            if self.base_logger:
                self.base_logger.log_financial_metric(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    metric_name="total_regimes_processed",
                    metric_value=float(len(regime_metrics)),
                    metric_type="regime",
                    step_name=step_name
                )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to log per-regime metrics: {e}")
            return False
    
    def get_regime_summary(self) -> Dict[str, Any]:
        """Get summary of regime usage and validation history."""
        try:
            return {
                'regime_registry': self.regime_registry,
                'validation_history': self.regime_validation_history[-10:],  # Last 10 validations
                'fail_fast_history': self.fail_fast_history[-10:],  # Last 10 fail-fast checks
                'total_regimes_tracked': len(self.regime_registry),
                'total_validations': len(self.regime_validation_history),
                'total_fail_fast_checks': len(self.fail_fast_history)
            }
        except Exception as e:
            self.logger.error(f"Failed to get regime summary: {e}")
            return {}
    
    def close(self) -> None:
        """Close the enhanced financial metrics logger and clean up resources."""
        with self._lock:
            try:
                # Close base logger if available
                if self.base_logger and hasattr(self.base_logger, 'close'):
                    self.base_logger.close()
                
                # Clear handlers
                for handler in self.logger.handlers[:]:
                    handler.close()
                    self.logger.removeHandler(handler)
                
                self.logger.info("🔒 Enhanced financial metrics logger closed successfully")
                
            except Exception as e:
                if self.fallback_logger:
                    self.fallback_logger.error(f"Error closing enhanced financial metrics logger: {e}")

# Global instance
_enhanced_financial_metrics_logger: Optional[EnhancedFinancialMetricsLogger] = None

def get_enhanced_financial_metrics_logger() -> EnhancedFinancialMetricsLogger:
    """Get the global enhanced financial metrics logger instance."""
    global _enhanced_financial_metrics_logger
    if _enhanced_financial_metrics_logger is None:
        _enhanced_financial_metrics_logger = EnhancedFinancialMetricsLogger()
    return _enhanced_financial_metrics_logger

def setup_enhanced_financial_metrics_logging(log_dir: Optional[str] = None, **kwargs) -> EnhancedFinancialMetricsLogger:
    """Setup the global enhanced financial metrics logger."""
    global _enhanced_financial_metrics_logger
    _enhanced_financial_metrics_logger = EnhancedFinancialMetricsLogger(log_dir=log_dir, **kwargs)
    return _enhanced_financial_metrics_logger

@contextmanager
def enhanced_financial_metrics_context(step_name: str, symbol: str, exchange: str, timeframe: str, 
                                     data: Optional[pd.DataFrame] = None, expected_regimes: Optional[List[str]] = None):
    """Context manager for enhanced financial metrics logging within a training step."""
    logger = get_enhanced_financial_metrics_logger()
    
    try:
        # Perform initial validation
        if data is not None and logger.fail_fast_enabled:
            fail_fast_result = logger.validate_fail_fast_conditions(
                data=data,
                step_name=step_name,
                expected_regimes=expected_regimes
            )
            
            if fail_fast_result.should_fail:
                logger.logger.error(f"🚨 FAIL-FAST TRIGGERED at step start for {step_name}")
                logger.logger.error(f"   Reason: {fail_fast_result.failure_reason}")
                raise RuntimeError(f"Fail-fast validation failed: {fail_fast_result.failure_reason}")
        
        # Log step start
        if logger.base_logger:
            logger.base_logger.log_step_start(step_name, symbol, exchange, timeframe)
        
        yield logger
        
        # Log step end
        if logger.base_logger:
            logger.base_logger.log_step_end(step_name, symbol, exchange, timeframe, success=True)
            
    except Exception as e:
        # Log step end with error
        if logger.base_logger:
            logger.base_logger.log_step_end(step_name, symbol, exchange, timeframe, success=False, error_message=str(e))
        raise

# Convenience functions for enhanced operations
def log_regime_metric_with_validation(symbol: str, exchange: str, timeframe: str, step_name: str, 
                                    regime_id: str, metric_name: str, metric_value: float, 
                                    metric_type: str = "regime", data: Optional[pd.DataFrame] = None) -> bool:
    """Log a regime-specific metric with validation."""
    logger = get_enhanced_financial_metrics_logger()
    return logger.log_financial_metric_with_regime_validation(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        metric_name=metric_name,
        metric_value=metric_value,
        metric_type=metric_type,
        step_name=step_name,
        regime_id=regime_id,
        data=data
    )

def validate_and_log_regime_data(symbol: str, exchange: str, timeframe: str, step_name: str,
                               data: pd.DataFrame, regime_column: str = 'composite_cluster_id') -> bool:
    """Validate regime data and log validation results."""
    logger = get_enhanced_financial_metrics_logger()
    
    # Perform validation
    validation_result = logger.validate_regime_data(data, regime_column, step_name)
    
    # Log validation results
    logger.log_financial_metric_with_regime_validation(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        metric_name="regime_validation_quality_score",
        metric_value=validation_result.quality_score,
        metric_type="quality",
        step_name=step_name,
        data=data
    )
    
    logger.log_financial_metric_with_regime_validation(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        metric_name="regime_count",
        metric_value=float(validation_result.regime_count),
        metric_type="regime",
        step_name=step_name,
        data=data
    )
    
    return validation_result.is_valid

# Export main classes and functions
__all__ = [
    'EnhancedFinancialMetricsLogger',
    'RegimeValidationResult',
    'FailFastValidationResult',
    'get_enhanced_financial_metrics_logger',
    'setup_enhanced_financial_metrics_logging',
    'enhanced_financial_metrics_context',
    'log_regime_metric_with_validation',
    'validate_and_log_regime_data'
]