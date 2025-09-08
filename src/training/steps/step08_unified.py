"""
Unified Step08: Advanced Feature Selection with Regime Data Splitting and Financial Risk Assessment

This consolidated module combines:
- Regime data splitting with HMM composite clusters
- Advanced feature selection with bias prevention
- Financial metrics calculation (returns, volatility, Sharpe ratio, VaR)
- Regime balance handling for imbalanced distributions
- Comprehensive risk assessment with explicit risk metrics

Author: AI Assistant
Date: 2024-01-XX
Version: 2.0.0
"""

import json
import os
import warnings
from datetime import datetime, timedelta

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

# Core imports

from src.utils.comprehensive_function_logger import (
    log_step_functions, log_important_calls, log_all_calls, 
    log_internal_call, log_step_progress, log_data_operation
)
from src.utils.common_operations import create_fallback_logger, create_fallback_decorator
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_kelly_calculation,
    validate_positive, validate_range, MathValidationError
)
from src.utils.lookahead_bias_detector import (
    get_global_detector, validate_no_future_data, LookaheadBiasError
)

# Enhanced optimization imports
try:
    from src.utils.m1_gpu_utils import get_m1_gpu_manager, M1GPUManager
    from src.utils.m1_memory_optimizer import get_m1_memory_optimizer, M1MemoryOptimizer
    from src.utils.m1_cpu_optimizer import get_m1_cpu_optimizer, M1CPUOptimizer
    from src.utils.vectorized_processing_core import OptimizedPipelineExecutor, PipelineStage, PipelineExecutionMode
    from src.utils.enhanced_matrix_operations import EnhancedMatrixOperations, ErrorHandler
    from src.utils.enhanced_step_optimizations import IntelligentOptimizationSelector, OptimizationStrategy, WorkloadType, OptimizationProfile
    from src.utils.optimized_data_manager import OptimizedDataManager, DataMetadata
    ENHANCED_OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    ENHANCED_OPTIMIZATIONS_AVAILABLE = False

# Machine learning imports
try:
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

# Optional dependencies with graceful fallbacks
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from boruta import BorutaPy
    BORUTA_AVAILABLE = True
except ImportError:
    BORUTA_AVAILABLE = False

try:
    import lime
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

# Pipeline standards and utilities
try:
    from src.utils.pipeline_standards import PipelineStandards, pipeline_standards
    from src.utils.common_operations import ensure_directory, safe_json_dump
except ImportError:
    def ensure_directory(path: str) -> str:
        os.makedirs(path, exist_ok=True)
        return path

    def safe_json_dump(data: Any, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    class MockPipelineStandards:
        def __init__(self):
            self.feature_selection_output_dir = "data/selected_features"

    pipeline_standards = MockPipelineStandards()

# Centralized decorators
try:
    from src.utils.centralized_decorators import (
        auto_fix_data_quality_issues, artifact_versioning, artifact_write_lock,
        circuit_breaker_protection, debug_training_step, deterministic_seed,
        handle_errors, idempotent_step, memory_efficient, nan_inf_and_constant_guard,
        prevent_data_leakage, quality_gate, resource_monitor, secure_data_processing,
        time_budget_watchdog, validate_step_output, validate_step_prerequisites,
        with_tracing_span
    )
    CENTRALIZED_DECORATORS_AVAILABLE = True
except ImportError:
    CENTRALIZED_DECORATORS_AVAILABLE = False
    # Create fallback decorators
    auto_fix_data_quality_issues = create_fallback_decorator(lambda x: x)
    artifact_versioning = create_fallback_decorator(lambda x: x)
    artifact_write_lock = create_fallback_decorator(lambda x: x)
    circuit_breaker_protection = create_fallback_decorator(lambda x: x)
    debug_training_step = create_fallback_decorator(lambda x: x)
    deterministic_seed = create_fallback_decorator(lambda x: x)
    handle_errors = create_fallback_decorator(lambda x: x)
    idempotent_step = create_fallback_decorator(lambda x: x)
    memory_efficient = create_fallback_decorator(lambda x: x)
    nan_inf_and_constant_guard = create_fallback_decorator(lambda x: x)
    prevent_data_leakage = create_fallback_decorator(lambda x: x)
    quality_gate = create_fallback_decorator(lambda x: x)
    resource_monitor = create_fallback_decorator(lambda x: x)
    secure_data_processing = create_fallback_decorator(lambda x: x)
    time_budget_watchdog = create_fallback_decorator(lambda x: x)
    validate_step_output = create_fallback_decorator(lambda x: x)
    validate_step_prerequisites = create_fallback_decorator(lambda x: x)
    with_tracing_span = create_fallback_decorator(lambda x: x)

# Enhanced MLflow integration
try:
    from src.utils.enhanced_mlflow_integration import (
        with_enhanced_mlflow_logging, log_step_report, create_detailed_step_report,
        log_step_metrics, log_step_dataframe_with_standardized_name,
        log_step_artifact_with_standardized_name
    )
    ENHANCED_MLFLOW_AVAILABLE = True
except ImportError:
    ENHANCED_MLFLOW_AVAILABLE = False
    with_enhanced_mlflow_logging = create_fallback_decorator(lambda x: x)
    log_step_report = lambda *args, **kwargs: 'fallback_report'
    create_detailed_step_report = lambda *args, **kwargs: {}
    log_step_metrics = lambda *args, **kwargs: None
    log_step_dataframe_with_standardized_name = lambda *args, **kwargs: 'fallback_dataframe'
    log_step_artifact_with_standardized_name = lambda *args, **kwargs: 'fallback_artifact'

# Unified data loader
try:
    from src.training.steps.data_collection.unified_data_loader import UnifiedDataLoader
    UNIFIED_DATA_LOADER_AVAILABLE = True
except ImportError:
    UNIFIED_DATA_LOADER_AVAILABLE = False

# Numba optimizations
if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True)
    def fast_correlation_matrix(X: np.ndarray) -> np.ndarray:
        """Compute correlation matrix using Numba for speed."""
        n_features = X.shape[1]
        corr_matrix = np.zeros((n_features, n_features))
        X_std = np.zeros_like(X)
        
        for i in prange(n_features):
            mean = np.mean(X[:, i])
            std = np.std(X[:, i])
            if std > 0:
                X_std[:, i] = (X[:, i] - mean) / std
            else:
                X_std[:, i] = 0
        
        n_samples = X.shape[0]
        for i in prange(n_features):
            for j in range(i, n_features):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    corr = np.sum(X_std[:, i] * X_std[:, j]) / (n_samples - 1)
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr
        
        return corr_matrix

    @jit(nopython=True)
    def fast_mutual_info_discrete(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fast mutual information calculation for discrete targets."""
        n_features = X.shape[1]
        mi_scores = np.zeros(n_features)
        
        for i in range(n_features):
            x_bins = np.percentile(X[:, i], np.linspace(0, 100, 11))
            x_discrete = np.searchsorted(x_bins[1:-1], X[:, i])
            mi_scores[i] = _calculate_mi_discrete(x_discrete, y)
        
        return mi_scores

    @jit(nopython=True)
    def _calculate_mi_discrete(x: np.ndarray, y: np.ndarray) -> float:
        """Calculate MI between two discrete variables."""
        xy_counts = np.zeros((10, 2))
        for i in range(len(x)):
            if y[i] < 2:
                xy_counts[min(x[i], 9), int(y[i])] += 1
        
        n = len(x)
        mi = 0.0
        for i in range(10):
            for j in range(2):
                pxy = xy_counts[i, j] / n
                if pxy > 0:
                    px = np.sum(xy_counts[i, :]) / n
                    py = np.sum(xy_counts[:, j]) / n
                    if px > 0 and py > 0:
                        mi += pxy * np.log(pxy / (px * py))
        
        return mi
else:
    def fast_correlation_matrix(X: np.ndarray) -> np.ndarray:
        return np.corrcoef(X.T)

    def fast_mutual_info_discrete(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return mutual_info_classif(X, y, random_state=42)

# Financial metrics dataclasses
@dataclass
class FinancialMetrics:
    """Comprehensive financial metrics for trading analysis."""
    returns: Dict[str, float] = field(default_factory=dict)
    volatility: Dict[str, float] = field(default_factory=dict)
    sharpe_ratio: Dict[str, float] = field(default_factory=dict)
    var_95: Dict[str, float] = field(default_factory=dict)
    var_99: Dict[str, float] = field(default_factory=dict)
    max_drawdown: Dict[str, float] = field(default_factory=dict)
    calmar_ratio: Dict[str, float] = field(default_factory=dict)
    sortino_ratio: Dict[str, float] = field(default_factory=dict)
    information_ratio: Dict[str, float] = field(default_factory=dict)
    beta: Dict[str, float] = field(default_factory=dict)
    alpha: Dict[str, float] = field(default_factory=dict)

@dataclass
class RiskMetrics:
    """Comprehensive risk assessment metrics."""
    portfolio_var: float = 0.0
    portfolio_es: float = 0.0
    concentration_risk: float = 0.0
    liquidity_risk: float = 0.0
    model_risk: float = 0.0
    regime_risk: float = 0.0
    feature_stability_risk: float = 0.0
    overfitting_risk: float = 0.0
    data_quality_risk: float = 0.0
    operational_risk: float = 0.0
    overall_risk_score: float = 0.0

@dataclass
class RegimeBalanceMetrics:
    """Metrics for regime balance assessment and handling."""
    regime_counts: Dict[str, int] = field(default_factory=dict)
    regime_percentages: Dict[str, float] = field(default_factory=dict)
    balance_score: float = 0.0
    imbalance_severity: str = "none"
    rebalancing_applied: bool = False
    rebalancing_method: str = ""
    min_samples_per_regime: int = 100
    target_balance_ratio: float = 0.8

@dataclass
class FeatureSelectionValidation:
    """Validation metrics for feature selection bias prevention."""
    selection_bias_score: float = 0.0
    temporal_stability: float = 0.0
    regime_consistency: float = 0.0
    correlation_stability: float = 0.0
    importance_stability: float = 0.0
    overfitting_indicators: Dict[str, float] = field(default_factory=dict)
    validation_passed: bool = False
    warnings: List[str] = field(default_factory=list)

@dataclass
class Step08Results:
    """Comprehensive results from unified Step08 execution."""
    regime_data: pd.DataFrame = None
    selected_features: Dict[str, List[str]] = field(default_factory=dict)
    financial_metrics: FinancialMetrics = field(default_factory=FinancialMetrics)
    risk_metrics: RiskMetrics = field(default_factory=RiskMetrics)
    regime_balance: RegimeBalanceMetrics = field(default_factory=RegimeBalanceMetrics)
    feature_validation: FeatureSelectionValidation = field(default_factory=FeatureSelectionValidation)
    execution_metadata: Dict[str, Any] = field(default_factory=dict)
    artifacts_generated: List[str] = field(default_factory=list)
    success: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)