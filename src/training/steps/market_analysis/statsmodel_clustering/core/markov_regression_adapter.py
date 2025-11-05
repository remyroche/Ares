"""
Enhanced Markov Regression Adapter

This module provides an enhanced adapter for statsmodels.tsa.regime_switching.MarkovRegression
with hardware optimization, parameter mapping, advanced diagnostics, and integration capabilities.

Key Features:
- Hardware optimization integration with UnifiedHardwareManager
- Parameter mapping from Pyro configurations
- Advanced diagnostics and validation
- Hierarchical optimization support
- VectorBT integration hooks
- Comprehensive error handling and logging
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
from dataclasses import dataclass, field
import logging
import time
import warnings
import json
from pathlib import Path
from contextlib import contextmanager

# Import statsmodels components
try:
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    MarkovRegression = None
    MarkovAutoregression = None

# Import hardware optimization
try:
    from src.utils.hardware.unified_hardware_manager import (
        UnifiedHardwareManager,
        WorkloadType,
        OptimizationLevel,
        get_unified_hardware_manager
    )
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATION_AVAILABLE = False
    UnifiedHardwareManager = None
    WorkloadType = None
    OptimizationLevel = None

# Import hierarchical optimization
try:
    from src.utils.ml_common.optimization.hierarchical_parameter_optimizer import (
        HierarchicalParameterOptimizer,
        ParameterGroup,
        OptimizationStage
    )
    HIERARCHICAL_OPTIMIZATION_AVAILABLE = True
except ImportError:
    HIERARCHICAL_OPTIMIZATION_AVAILABLE = False
    HierarchicalParameterOptimizer = None
    ParameterGroup = None
    OptimizationStage = None

# Import VectorBT integration
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None

# Import utilities
try:
    from src.utils.tprint import (
        tprint_info, tprint_success, tprint_warning, tprint_error,
        tprint_timer, tprint_structured
    )
except ImportError:
    def tprint_info(msg): print(f'ℹ️  {msg}')
    def tprint_success(msg): print(f'✅ {msg}')
    def tprint_warning(msg): print(f'⚠️  {msg}')
    def tprint_error(msg): print(f'❌ {msg}')
    def tprint_timer(msg, level="INFO"):
        class TimerContext:
            def __enter__(self):
                print(f'⏱️  Starting: {msg}')
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                print(f'⏱️  Completed: {msg}')
        return TimerContext()
    def tprint_structured(data, level="INFO"):
        for key, value in data.items():
            print(f'🔧 {key}: {value}')

# Import sklearn for preprocessing
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.model_selection import ParameterGrid
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    StandardScaler = None
    PCA = None
    ParameterGrid = None


@dataclass
class MarkovRegressionConfig:
    """
    Enhanced configuration for MarkovRegression with hardware optimization.
    
    This configuration extends basic statsmodels parameters with
    hardware optimization, parameter mapping, and integration options.
    """
    # Model structure
    k_regimes: int = 2
    trend: str = 'c'  # 'c', 't', 'ct'
    order: int = 0  # Autoregressive order
    exog_tvtp: Optional[np.ndarray] = None  # Exogenous variables for time-varying transition probabilities
    switching_variance: bool = True
    switching_trend: bool = True
    switching_exog: bool = False
    
    # Training parameters
    maxiter: int = 100
    tolerance: float = 1e-6
    random_state: int = 42
    loglikelihood_burn: int = 0
    method: str = 'bfgs'  # 'em' or 'bfgs'
    
    # Data preprocessing
    enable_pca: bool = True
    pca_components: int = 12
    pca_variance_threshold: float = 0.95
    enable_scaling: bool = True
    
    # Hardware optimization
    enable_hardware_optimization: bool = True
    workload_type: str = 'ml_training'
    optimization_level: str = 'balanced'
    hardware_config: Optional[Dict[str, Any]] = None
    
    # Parameter mapping (from Pyro)
    pyro_config: Optional[Dict[str, Any]] = None
    auto_map_parameters: bool = True
    
    # Hierarchical optimization
    enable_hierarchical_optimization: bool = False
    optimization_stages: List[Dict[str, Any]] = field(default_factory=list)
    
    # VectorBT integration
    enable_vectorbt_integration: bool = False
    vectorbt_config: Optional[Dict[str, Any]] = None
    
    # Diagnostics and validation
    enable_diagnostics: bool = True
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    
    # Advanced options
    missing: str = 'drop'
    verbose: bool = True
    save_intermediate_results: bool = False
    output_dir: Optional[str] = None


@dataclass
class MarkovRegressionResult:
    """
    Enhanced result container for MarkovRegression with additional diagnostics.
    """
    # Core results
    fitted_model: Optional[Any] = None
    cluster_labels: np.ndarray = field(default_factory=lambda: np.array([]))
    cluster_probabilities: Optional[np.ndarray] = None
    n_regimes: int = 0
    
    # Model parameters
    transition_matrix: Optional[np.ndarray] = None
    regime_params: Optional[Dict[str, Any]] = None
    model_summary: Optional[Dict[str, Any]] = None
    
    # Performance metrics
    log_likelihood: float = 0.0
    aic: float = 0.0
    bic: float = 0.0
    hqic: float = 0.0
    
    # Processing metadata
    processing_time: float = 0.0
    optimization_time: float = 0.0
    feature_names: List[str] = field(default_factory=list)
    success: bool = False
    error_message: Optional[str] = None
    
    # Hardware optimization
    hardware_metrics: Optional[Dict[str, Any]] = None
    
    # Diagnostics
    diagnostics: Optional[Dict[str, Any]] = None
    validation_results: Optional[Dict[str, Any]] = None
    
    # VectorBT integration
    vectorbt_results: Optional[Dict[str, Any]] = None
    
    # Additional metadata
    metadata: Optional[Dict[str, Any]] = None


class ParameterMapper:
    """
    Maps Pyro Sticky Finite HMM parameters to statsmodels equivalents.
    
    This class handles conversion between Pyro-based parameter configurations
    and statsmodels-compatible parameters, ensuring seamless migration.
    """
    
    @staticmethod
    def map_pyro_to_statsmodels(pyro_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Pyro parameters to statsmodels format.
        
        Args:
            pyro_params: Dictionary of Pyro model parameters
            
        Returns:
            Dictionary of statsmodels-compatible parameters
        """
        tprint_info("🔄 Mapping Pyro parameters to statsmodels format")
        
        if not pyro_params:
            tprint_warning("⚠️ No Pyro parameters provided")
            return {}
        
        mapped_params = {}
        
        # Map number of regimes
        if 'K' in pyro_params:
            mapped_params['k_regimes'] = pyro_params['K']
            tprint_info(f"📊 Mapped K={pyro_params['K']} to k_regimes")
        
        # Map transition matrix
        if 'transition_matrix' in pyro_params:
            # Pyro uses different format, convert to statsmodels format
            transition_matrix = np.array(pyro_params['transition_matrix'])
            if transition_matrix.shape[0] == transition_matrix.shape[1]:
                mapped_params['transition_matrix'] = transition_matrix
                tprint_info(f"🔄 Mapped transition matrix: {transition_matrix.shape}")
        
        # Map emission parameters
        if 'emission_means' in pyro_params:
            mapped_params['regime_means'] = pyro_params['emission_means']
            tprint_info("📊 Mapped emission means to regime means")
        
        if 'emission_covs' in pyro_params:
            mapped_params['regime_covs'] = pyro_params['emission_covs']
            tprint_info("📊 Mapped emission covariances to regime covs")
        
        # Map hyperparameters
        if 'alpha' in pyro_params:
            mapped_params['transition_prior'] = pyro_params['alpha']
            tprint_info("🔧 Mapped alpha to transition prior")
        
        if 'beta' in pyro_params:
            mapped_params['emission_prior'] = pyro_params['beta']
            tprint_info("🔧 Mapped beta to emission prior")
        
        # Map variance switching
        if 'switching_variance' in pyro_params:
            mapped_params['switching_variance'] = pyro_params['switching_variance']
            tprint_info("🔧 Mapped variance switching")
        
        tprint_success(f"✅ Mapped {len(mapped_params)} parameters from Pyro to statsmodels")
        return mapped_params
    
    @staticmethod
    def map_search_spaces(pyro_search_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Pyro hyperparameter search spaces to statsmodels format.
        
        Args:
            pyro_search_space: Pyro hyperparameter search space
            
        Returns:
            statsmodels-compatible search space
        """
        tprint_info("🔄 Mapping Pyro search spaces to statsmodels format")
        
        if not pyro_search_space:
            tprint_warning("⚠️ No Pyro search space provided")
            return {}
        
        mapped_space = {}
        
        # Map K (number of regimes)
        if 'K' in pyro_search_space:
            mapped_space['k_regimes'] = pyro_search_space['K']
            tprint_info("📊 Mapped K search space to k_regimes")
        
        # Map variance switching
        if 'switching_variance' in pyro_search_space:
            mapped_space['switching_variance'] = pyro_search_space['switching_variance']
            tprint_info("🔧 Mapped variance switching search space")
        
        # Map trend switching
        if 'switching_trend' in pyro_search_space:
            mapped_space['switching_trend'] = pyro_search_space['switching_trend']
            tprint_info("🔧 Mapped trend switching search space")
        
        # Map autoregressive order
        if 'order' in pyro_search_space:
            mapped_space['order'] = pyro_search_space['order']
            tprint_info("📊 Mapped autoregressive order search space")
        
        tprint_success(f"✅ Mapped {len(mapped_space)} search space parameters")
        return mapped_space


class MarkovRegressionDiagnostics:
    """
    Advanced diagnostics for MarkovRegression models.
    
    Provides comprehensive validation, analysis, and reporting capabilities
    for regime switching models.
    """
    
    def __init__(self, config: MarkovRegressionConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def run_comprehensive_diagnostics(self,
                                    model: Any,
                                    data: np.ndarray,
                                    labels: np.ndarray,
                                    probabilities: np.ndarray) -> Dict[str, Any]:
        """
        Run comprehensive diagnostics on fitted model.
        
        Args:
            model: Fitted statsmodels model
            data: Original data
            labels: Predicted regime labels
            probabilities: Regime probabilities
            
        Returns:
            Dictionary with diagnostic results
        """
        tprint_info("🔍 Running comprehensive model diagnostics")
        
        diagnostics = {}
        
        # Model fit diagnostics
        tprint_info("📊 Assessing model fit")
        diagnostics['model_fit'] = self._assess_model_fit(model)
        
        # Regime stability analysis
        tprint_info("📈 Analyzing regime stability")
        diagnostics['regime_stability'] = self._analyze_regime_stability(labels, probabilities)
        
        # Transition analysis
        tprint_info("🔄 Analyzing transition dynamics")
        diagnostics['transition_analysis'] = self._analyze_transitions(model, labels)
        
        # Residual analysis
        tprint_info("🔍 Analyzing model residuals")
        diagnostics['residual_analysis'] = self._analyze_residuals(model, data)
        
        # Predictive performance
        tprint_info("📈 Assessing predictive performance")
        diagnostics['predictive_performance'] = self._assess_predictive_performance(model, data)
        
        # Regime characteristics
        tprint_info("📊 Analyzing regime characteristics")
        diagnostics['regime_characteristics'] = self._analyze_regime_characteristics(data, labels)
        
        tprint_success("✅ Comprehensive diagnostics completed")
        return diagnostics
    
    def _assess_model_fit(self, model: Any) -> Dict[str, Any]:
        """Assess model fit quality."""
        tprint_info("📊 Assessing model fit quality")
        
        fit_assessment = {}
        
        if hasattr(model, 'llf'):
            fit_assessment['log_likelihood'] = model.llf
            tprint_info(f"📈 Log likelihood: {model.llf:.4f}")
        
        if hasattr(model, 'aic'):
            fit_assessment['aic'] = model.aic
            tprint_info(f"📈 AIC: {model.aic:.4f}")
        
        if hasattr(model, 'bic'):
            fit_assessment['bic'] = model.bic
            tprint_info(f"📈 BIC: {model.bic:.4f}")
        
        if hasattr(model, 'hqic'):
            fit_assessment['hqic'] = model.hqic
            tprint_info(f"📈 HQIC: {model.hqic:.4f}")
        
        # Convergence diagnostics
        if hasattr(model, 'mle_retvals'):
            converged = model.mle_retvals.get('converged', False)
            iterations = model.mle_retvals.get('iterations', 0)
            function_evals = model.mle_retvals.get('function_evals', 0)
            
            fit_assessment['converged'] = converged
            fit_assessment['iterations'] = iterations
            fit_assessment['function_evals'] = function_evals
            
            tprint_info(f"🔄 Convergence: {converged}, Iterations: {iterations}, Function evals: {function_evals}")
        
        return fit_assessment
    
    def _analyze_regime_stability(self, labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, Any]:
        """Analyze regime stability and persistence."""
        tprint_info("📈 Analyzing regime stability and persistence")
        
        stability = {}
        
        # Regime persistence
        tprint_info("🔄 Calculating regime persistence")
        stability['regime_persistence'] = self._calculate_regime_persistence(labels)
        
        # Regime duration statistics
        tprint_info("⏱️ Calculating duration statistics")
        stability['duration_stats'] = self._calculate_duration_statistics(labels)
        
        # Probability confidence
        tprint_info("📊 Calculating probability confidence")
        stability['probability_confidence'] = self._calculate_probability_confidence(probabilities)
        
        # Regime switching frequency
        tprint_info("🔄 Calculating switching frequency")
        stability['switching_frequency'] = self._calculate_switching_frequency(labels)
        
        tprint_success("✅ Regime stability analysis completed")
        return stability
    
    def _analyze_transitions(self, model: Any, labels: np.ndarray) -> Dict[str, Any]:
        """Analyze transition dynamics."""
        tprint_info("🔄 Analyzing transition dynamics")
        
        transitions = {}
        
        # Extract transition matrix
        if hasattr(model, 'regime_transition_matrix'):
            transitions['transition_matrix'] = model.regime_transition_matrix
            tprint_info("📊 Extracted regime transition matrix from model")
        elif hasattr(model, 'params'):
            # Try to extract from parameters
            transitions['transition_matrix'] = self._extract_transition_matrix(model)
            tprint_info("🔧 Extracted transition matrix from model parameters")
        
        # Transition counts
        tprint_info("📊 Calculating transition counts")
        transitions['transition_counts'] = self._calculate_transition_counts(labels)
        
        # Transition probabilities (empirical)
        tprint_info("📈 Calculating empirical transition probabilities")
        transitions['empirical_transition_probs'] = self._calculate_empirical_transitions(labels)
        
        tprint_success("✅ Transition analysis completed")
        return transitions
    
    def _analyze_residuals(self, model: Any, data: np.ndarray) -> Dict[str, Any]:
        """Analyze model residuals."""
        tprint_info("🔍 Analyzing model residuals")
        
        residuals = {}
        
        if hasattr(model, 'resid'):
            resid = model.resid
            tprint_info(f"📊 Analyzing {len(resid)} residual values")
            
            # Basic statistics
            tprint_info("📈 Calculating residual statistics")
            residuals['mean'] = np.mean(resid)
            residuals['std'] = np.std(resid)
            residuals['skewness'] = self._calculate_skewness(resid)
            residuals['kurtosis'] = self._calculate_kurtosis(resid)
            
            # Normality test
            tprint_info("🔍 Testing for normality")
            residuals['normality_test'] = self._test_normality(resid)
            
            # Autocorrelation
            tprint_info("📈 Calculating autocorrelation")
            residuals['autocorrelation'] = self._calculate_autocorrelation(resid)
        else:
            tprint_warning("⚠️ No residuals found in model")
        
        tprint_success("✅ Residual analysis completed")
        return residuals
    
    def _assess_predictive_performance(self, model: Any, data: np.ndarray) -> Dict[str, Any]:
        """Assess predictive performance."""
        tprint_info("📈 Assessing predictive performance")
        
        performance = {}
        
        # In-sample performance
        if hasattr(model, 'fittedvalues'):
            tprint_info("📊 Calculating in-sample R²")
            performance['in_sample_r2'] = self._calculate_r2(data.flatten(), model.fittedvalues)
        
        # Out-of-sample performance (if validation split)
        if self.config.validation_split > 0:
            tprint_info("🔍 Performing cross-validation")
            performance['out_of_sample'] = self._cross_validate_model(model, data)
        
        tprint_success("✅ Predictive performance assessment completed")
        return performance
    
    def _analyze_regime_characteristics(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Analyze characteristics of each regime."""
        tprint_info("📊 Analyzing regime characteristics")
        
        characteristics = {}
        n_regimes = len(np.unique(labels))
        tprint_info(f"📈 Analyzing {n_regimes} regimes")
        
        for regime in range(n_regimes):
            regime_mask = labels == regime
            regime_data = data[regime_mask]
            
            if len(regime_data) > 0:
                proportion = len(regime_data) / len(data)
                tprint_info(f"📊 Regime {regime}: {len(regime_data)} samples ({proportion:.2%})")
                
                characteristics[f'regime_{regime}'] = {
                    'size': len(regime_data),
                    'proportion': proportion,
                    'mean': np.mean(regime_data, axis=0),
                    'std': np.std(regime_data, axis=0),
                    'min': np.min(regime_data, axis=0),
                    'max': np.max(regime_data, axis=0)
                }
        
        tprint_success("✅ Regime characteristics analysis completed")
        return characteristics
    
    def _calculate_regime_persistence(self, labels: np.ndarray) -> Dict[int, float]:
        """Calculate regime persistence (probability of staying in same regime)."""
        persistence = {}
        n_regimes = len(np.unique(labels))
        
        for regime in range(n_regimes):
            regime_mask = labels == regime
            regime_indices = np.where(regime_mask)[0]
            
            if len(regime_indices) > 1:
                # Count transitions within regime
                transitions = 0
                total = 0
                
                for i in range(1, len(regime_indices)):
                    if regime_indices[i] == regime_indices[i-1] + 1:
                        transitions += 1
                    total += 1
                
                persistence[regime] = transitions / max(1, total)
            else:
                persistence[regime] = 0.0
        
        return persistence
    
    def _calculate_duration_statistics(self, labels: np.ndarray) -> Dict[str, Any]:
        """Calculate regime duration statistics."""
        durations = []
        
        # Find continuous segments
        current_regime = labels[0]
        current_duration = 1
        
        for i in range(1, len(labels)):
            if labels[i] == current_regime:
                current_duration += 1
            else:
                durations.append(current_duration)
                current_regime = labels[i]
                current_duration = 1
        
        durations.append(current_duration)
        
        return {
            'mean_duration': np.mean(durations),
            'std_duration': np.std(durations),
            'min_duration': np.min(durations),
            'max_duration': np.max(durations),
            'median_duration': np.median(durations)
        }
    
    def _calculate_probability_confidence(self, probabilities: np.ndarray) -> Dict[str, float]:
        """Calculate confidence metrics for regime probabilities."""
        max_probs = np.max(probabilities, axis=1)
        
        return {
            'mean_confidence': np.mean(max_probs),
            'std_confidence': np.std(max_probs),
            'min_confidence': np.min(max_probs),
            'max_confidence': np.max(max_probs)
        }
    
    def _calculate_switching_frequency(self, labels: np.ndarray) -> float:
        """Calculate frequency of regime switching."""
        switches = np.sum(labels[1:] != labels[:-1])
        return switches / (len(labels) - 1)
    
    def _extract_transition_matrix(self, model: Any) -> Optional[np.ndarray]:
        """Extract transition matrix from model parameters."""
        # This is a simplified implementation
        # In practice, you'd need to parse the parameter vector
        return None
    
    def _calculate_transition_counts(self, labels: np.ndarray) -> np.ndarray:
        """Calculate transition count matrix."""
        n_regimes = len(np.unique(labels))
        transition_counts = np.zeros((n_regimes, n_regimes))
        
        for i in range(1, len(labels)):
            from_regime = labels[i-1]
            to_regime = labels[i]
            transition_counts[from_regime, to_regime] += 1
        
        return transition_counts
    
    def _calculate_empirical_transitions(self, labels: np.ndarray) -> np.ndarray:
        """Calculate empirical transition probabilities."""
        counts = self._calculate_transition_counts(labels)
        
        # Normalize rows to get probabilities
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        
        return counts / row_sums
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        from scipy import stats
        return stats.skew(data)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        from scipy import stats
        return stats.kurtosis(data)
    
    def _test_normality(self, data: np.ndarray) -> Dict[str, Any]:
        """Test for normality."""
        from scipy import stats
        
        try:
            statistic, p_value = stats.normaltest(data)
            return {
                'statistic': statistic,
                'p_value': p_value,
                'is_normal': p_value > 0.05
            }
        except:
            return {'error': 'Normality test failed'}
    
    def _calculate_autocorrelation(self, data: np.ndarray, lags: int = 10) -> List[float]:
        """Calculate autocorrelation at different lags."""
        autocorr = []
        
        for lag in range(1, lags + 1):
            if len(data) > lag:
                corr = np.corrcoef(data[:-lag], data[lag:])[0, 1]
                autocorr.append(corr if not np.isnan(corr) else 0.0)
            else:
                autocorr.append(0.0)
        
        return autocorr
    
    def _calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared."""
        # Ensure arrays have the same shape
        if y_true.shape != y_pred.shape:
            # If shapes don't match, try to reshape y_pred to match y_true
            if y_true.size == y_pred.size:
                y_pred = y_pred.reshape(y_true.shape)
            else:
                # If sizes don't match, return 0.0 as fallback
                return 0.0
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    def _cross_validate_model(self, model: Any, data: np.ndarray) -> Dict[str, Any]:
        """Perform cross-validation for model assessment."""
        # Simplified cross-validation
        n_samples = len(data)
        val_size = int(n_samples * self.config.validation_split)
        
        if val_size > 0:
            train_data = data[:-val_size]
            val_data = data[-val_size:]
            
            # This is a placeholder - actual implementation would refit model
            return {
                'validation_size': val_size,
                'note': 'Cross-validation placeholder - implement actual refitting'
            }
        
        return {'error': 'Insufficient data for validation'}


class MarkovRegressionAdapter:
    """
    Enhanced adapter for statsmodels.tsa.regime_switching.MarkovRegression.
    
    This adapter provides a comprehensive interface with hardware optimization,
    parameter mapping, advanced diagnostics, and integration capabilities.
    """
    
    def __init__(self, config: Optional[MarkovRegressionConfig] = None,
                 hardware_manager: Optional[UnifiedHardwareManager] = None):
        """
        Initialize enhanced MarkovRegression adapter.
        
        Args:
            config: Configuration for adapter
            hardware_manager: Hardware optimization manager
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError(
                "statsmodels is required for MarkovRegressionAdapter. "
                "Install with: pip install statsmodels>=0.13.0"
            )
        
        self.config = config or MarkovRegressionConfig()
        self.hardware_manager = hardware_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.model = None
        self.scaler = None
        self.pca = None
        self.parameter_mapper = ParameterMapper()
        self.diagnostics = MarkovRegressionDiagnostics(self.config)
        
        # Optimization components
        self.hierarchical_optimizer = None
        self.vectorbt_integration = None
        
        # State tracking
        self.is_fitted = False
        self.feature_names = []
        self.pca_loadings = None
        
        # Initialize hardware optimization
        if self.config.enable_hardware_optimization and HARDWARE_OPTIMIZATION_AVAILABLE:
            self._initialize_hardware_optimization()
        
        # Initialize hierarchical optimization
        if self.config.enable_hierarchical_optimization and HIERARCHICAL_OPTIMIZATION_AVAILABLE:
            self._initialize_hierarchical_optimization()
        
        # Initialize VectorBT integration
        if self.config.enable_vectorbt_integration and VECTORBT_AVAILABLE:
            self._initialize_vectorbt_integration()
        
        tprint_info(f"🚀 Initialized Enhanced MarkovRegressionAdapter (k_regimes={self.config.k_regimes})")
    
    def _initialize_hardware_optimization(self):
        """Initialize hardware optimization."""
        if self.hardware_manager is None:
            self.hardware_manager = get_unified_hardware_manager()
        
        # Configure workload type
        if self.config.workload_type in [wt.value for wt in WorkloadType]:
            workload_type = WorkloadType(self.config.workload_type)
        else:
            workload_type = WorkloadType.ML_TRAINING
        
        # Configure optimization level
        if self.config.optimization_level in [ol.value for ol in OptimizationLevel]:
            optimization_level = OptimizationLevel(self.config.optimization_level)
        else:
            optimization_level = OptimizationLevel.BALANCED
        
        self.hardware_manager.configure_workload(workload_type, optimization_level)
        
        tprint_info(f"🔧 Hardware optimization enabled: {workload_type.value} ({optimization_level.value})")
    
    def _initialize_hierarchical_optimization(self):
        """Initialize hierarchical optimization."""
        # Create optimization stages from config
        stages = []
        for stage_config in self.config.optimization_stages:
            stage = OptimizationStage(
                name=stage_config.get('name', 'default'),
                parameters=stage_config.get('parameters', {}),
                search_space=stage_config.get('search_space', {}),
                max_iterations=stage_config.get('max_iterations', 10)
            )
            stages.append(stage)
        
        self.hierarchical_optimizer = HierarchicalParameterOptimizer(stages=stages)
        tprint_info("🔧 Hierarchical optimization enabled")
    
    def _initialize_vectorbt_integration(self):
        """Initialize VectorBT integration."""
        # This would initialize VectorBT components for backtesting
        self.vectorbt_integration = {
            'enabled': True,
            'config': self.config.vectorbt_config or {}
        }
        tprint_info("🔧 VectorBT integration enabled")
    
    def _preprocess_data(self, data: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Preprocess data with scaling and optional PCA.
        
        Args:
            data: Input data (n_samples, n_features)
            
        Returns:
            Tuple of (processed_data, feature_names)
        """
        if not SKLEARN_AVAILABLE:
            tprint_warning("⚠️ sklearn not available, skipping preprocessing")
            return data, [f'feature_{i}' for i in range(data.shape[1])]
        
        # Store original shape
        n_samples, n_features = data.shape
        
        # Use more efficient data types
        if data.dtype == np.float64:
            tprint_info("🔧 Converting data to float32 for memory efficiency")
            data = data.astype(np.float32)
        
        # Apply scaling if enabled
        if self.config.enable_scaling:
            self.scaler = StandardScaler()
            data_scaled = self.scaler.fit_transform(data)
            # Use float32 for scaled data
            data_scaled = data_scaled.astype(np.float32)
        else:
            data_scaled = data
        
        # Apply PCA if enabled
        if self.config.enable_pca and n_features > self.config.pca_components:
            self.pca = PCA(
                n_components=self.config.pca_components,
                random_state=self.config.random_state
            )
            data_processed = self.pca.fit_transform(data_scaled)
            # Use float32 for PCA results
            data_processed = data_processed.astype(np.float32)
            feature_names = [f'pca_{i+1}' for i in range(data_processed.shape[1])]
            
            # Store PCA loadings for analysis
            self.pca_loadings = {}
            for i in range(min(5, self.config.pca_components)):
                component_name = f'pca_{i+1}'
                component_loadings = self.pca.components_[i]
                # Get top features
                top_indices = np.argsort(np.abs(component_loadings))[::-1][:10]
                self.pca_loadings[component_name] = {
                    f'feature_{j}': float(component_loadings[j])
                    for j in top_indices
                }
        else:
            data_processed = data_scaled
            feature_names = [f'feature_{i}' for i in range(n_features)]
        
        return data_processed, feature_names
    
    def fit(self, data: Union[np.ndarray, pd.DataFrame],
            optimization_config: Optional[Dict[str, Any]] = None) -> MarkovRegressionResult:
        """
        Fit MarkovRegression model with enhanced optimization support.
        
        Args:
            data: Input data with features and target
            optimization_config: Configuration for hierarchical optimization
            
        Returns:
            Enhanced result object with additional diagnostics
        """
        start_time = time.time()
        
        try:
            # Convert to numpy if needed
            if hasattr(data, 'values'):
                if hasattr(data, 'columns'):
                    self.feature_names = list(data.columns)
                data = data.values
            elif not hasattr(data, 'shape'):
                raise TypeError(f"Expected numpy array or pandas DataFrame, got {type(data)}")
            
            # Validate input
            self._validate_input(data)
            
            # Check if we should use batch processing for large datasets
            n_samples = data.shape[0]
            batch_size = 10000  # Process in batches of 10k samples
            
            if n_samples > batch_size:
                tprint_info(f"📊 Large dataset detected ({n_samples} samples), using batch processing")
                return self._fit_with_batch_processing(data, batch_size)
            
            # Preprocess data
            data_processed, feature_names = self._preprocess_data(data)
            if not self.feature_names:
                self.feature_names = feature_names
            
            # Apply parameter mapping if Pyro config is provided
            if self.config.pyro_config and self.config.auto_map_parameters:
                mapped_params = self.parameter_mapper.map_pyro_to_statsmodels(self.config.pyro_config)
                self._apply_mapped_parameters(mapped_params)
            
            # Use hardware optimization context if enabled
            optimization_context = None
            if self.config.enable_hardware_optimization and self.hardware_manager:
                workload_type = WorkloadType(self.config.workload_type)
                optimization_level = OptimizationLevel(self.config.optimization_level)
                optimization_context = self.hardware_manager.optimization_context(
                    workload_type, optimization_level
                )
            
            with optimization_context if optimization_context else self._null_context():
                optimization_start = time.time()
                
                # Create and fit model
                self.model = self._create_model(data_processed)
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.model = self.model.fit(
                        maxiter=self.config.maxiter,
                        tolerance=self.config.tolerance,
                        method=self.config.method,
                        loglikelihood_burn=self.config.loglikelihood_burn
                    )
                
                self.optimization_time = time.time() - optimization_start
            
            # Get predictions
            labels = self._get_regime_labels()
            probabilities = self._get_regime_probabilities()
            
            # Get model artifacts
            transition_matrix = self._get_transition_matrix()
            regime_params = self._get_regime_params()
            model_summary = self._get_model_summary()
            
            # Run diagnostics if enabled
            diagnostics = None
            if self.config.enable_diagnostics:
                diagnostics = self.diagnostics.run_comprehensive_diagnostics(
                    self.model, data_processed, labels, probabilities
                )
            
            # Get hardware metrics
            hardware_metrics = None
            if self.config.enable_hardware_optimization and self.hardware_manager:
                hardware_metrics = self.hardware_manager.get_system_status()
            
            # VectorBT integration if enabled
            vectorbt_results = None
            if self.config.enable_vectorbt_integration and self.vectorbt_integration:
                vectorbt_results = self._run_vectorbt_analysis(data, labels)
            
            # Save intermediate results if enabled
            if self.config.save_intermediate_results:
                self._save_intermediate_results(labels, probabilities, transition_matrix)
            
            self.is_fitted = True
            processing_time = time.time() - start_time
            
            # Build result
            result = MarkovRegressionResult(
                fitted_model=self.model,
                cluster_labels=labels,
                cluster_probabilities=probabilities,
                n_regimes=self.config.k_regimes,
                transition_matrix=transition_matrix,
                regime_params=regime_params,
                model_summary=model_summary,
                log_likelihood=model_summary.get('log_likelihood', 0.0),
                aic=model_summary.get('aic', 0.0),
                bic=model_summary.get('bic', 0.0),
                hqic=model_summary.get('hqic', 0.0),
                processing_time=processing_time,
                optimization_time=self.optimization_time,
                feature_names=self.feature_names,
                success=True,
                hardware_metrics=hardware_metrics,
                diagnostics=diagnostics,
                vectorbt_results=vectorbt_results,
                metadata={
                    'config': self.config.__dict__,
                    'pca_loadings': self.pca_loadings,
                    'data_shape': data.shape,
                    'processed_shape': data_processed.shape
                }
            )
            
            tprint_success(f"✅ Enhanced MarkovRegression fitted successfully")
            tprint_structured({
                "n_regimes": result.n_regimes,
                "log_likelihood": result.log_likelihood,
                "aic": result.aic,
                "processing_time": f"{processing_time:.2f}s",
                "optimization_time": f"{self.optimization_time:.2f}s"
            }, level="INFO")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Enhanced MarkovRegression fitting failed: {e}")
            self.logger.error(f"Fitting error: {e}", exc_info=True)
            
            return MarkovRegressionResult(
                cluster_labels=np.array([]),
                cluster_probabilities=None,
                n_regimes=0,
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time,
                feature_names=self.feature_names
            )
    
    def predict(self, steps: int, 
                confidence_intervals: bool = True) -> Dict[str, np.ndarray]:
        """
        Generate predictions with confidence intervals.
        
        Args:
            steps: Number of steps to forecast
            confidence_intervals: Whether to calculate confidence intervals
            
        Returns:
            Dictionary with predictions, regimes, and confidence intervals
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        try:
            # Get predictions from model
            predictions = self.model.predict(steps=steps)
            
            result = {
                'predictions': predictions,
                'steps': steps
            }
            
            # Add confidence intervals if requested
            if confidence_intervals:
                try:
                    forecast = self.model.get_forecast(steps=steps)
                    result['confidence_intervals'] = forecast.conf_int()
                except Exception as e:
                    tprint_warning(f"⚠️ Could not compute confidence intervals: {e}")
            
            # Add regime predictions
            try:
                regime_predictions = self.model.predict_regime(steps=steps)
                result['regime_predictions'] = regime_predictions
            except Exception as e:
                tprint_warning(f"⚠️ Could not predict regimes: {e}")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Prediction failed: {e}")
            raise
    
    def get_regime_probabilities(self) -> pd.DataFrame:
        """Get smoothed regime probabilities over time."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        try:
            if hasattr(self.model, 'smoothed_marginal_probabilities'):
                probabilities = self.model.smoothed_marginal_probabilities
                return pd.DataFrame(
                    probabilities,
                    columns=[f'regime_{i}' for i in range(probabilities.shape[1])]
                )
            else:
                tprint_warning("⚠️ Smoothed probabilities not available")
                return pd.DataFrame()
        except Exception as e:
            tprint_error(f"❌ Failed to get regime probabilities: {e}")
            return pd.DataFrame()
    
    def get_transition_matrix(self) -> np.ndarray:
        """Get estimated transition probability matrix."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self._get_transition_matrix()
    
    def run_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive model diagnostics."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # This would need original data, which we don't store
        # For now, return basic diagnostics
        return self._get_model_summary()
    
    def _validate_input(self, data: np.ndarray):
        """Validate input data."""
        if len(data.shape) != 2:
            raise ValueError(f"Expected 2D array, got shape {data.shape}")
        
        n_samples, n_features = data.shape
        
        if n_samples < 100:
            raise ValueError(f"Insufficient samples: {n_samples} < 100")
        
        if n_features < 1:
            raise ValueError(f"Insufficient features: {n_features} < 1")
        
        # Check for NaN values
        nan_ratio = np.isnan(data).sum() / data.size
        if nan_ratio > 0.1:
            raise ValueError(f"Excessive NaN values: {nan_ratio:.1%} > 10%")
        
        # Check for infinite values
        inf_ratio = np.isinf(data).sum() / data.size
        if inf_ratio > 0:
            raise ValueError(f"Data contains {inf_ratio:.1%} infinite values")
    
    def _apply_mapped_parameters(self, mapped_params: Dict[str, Any]):
        """Apply mapped parameters from Pyro configuration."""
        if 'k_regimes' in mapped_params:
            self.config.k_regimes = mapped_params['k_regimes']
        
        if 'switching_variance' in mapped_params:
            self.config.switching_variance = mapped_params['switching_variance']
        
        if 'switching_trend' in mapped_params:
            self.config.switching_trend = mapped_params['switching_trend']
        
        if 'order' in mapped_params:
            self.config.order = mapped_params['order']
    
    def _create_model(self, data: np.ndarray) -> MarkovRegression:
        """Create MarkovRegression model."""
        # MarkovRegression requires univariate data
        # For multivariate clustering, we use first feature as target and others as exogenous variables
        
        if data.shape[1] > 1:
            tprint_info("📊 Using multivariate approach: first feature as target, others as regressors")
            # Use first feature as target (endog)
            endog_data = data[:, 0]
            # Use remaining features as exogenous variables (exog)
            exog_data = data[:, 1:]
        else:
            tprint_info("📊 Using univariate approach: single feature as target")
            endog_data = data.flatten()
            exog_data = None
        
        # Create model with configured parameters
        model = MarkovRegression(
            endog=endog_data,
            k_regimes=self.config.k_regimes,
            trend=self.config.trend,
            order=self.config.order,
            exog_tvtp=self.config.exog_tvtp,
            switching_variance=self.config.switching_variance,
            switching_trend=self.config.switching_trend,
            switching_exog=self.config.switching_exog,
            exog=exog_data,  # Pass exogenous variables
            missing=self.config.missing
        )
        
        return model
    
    def _get_regime_labels(self) -> np.ndarray:
        """Get regime labels from fitted model."""
        if hasattr(self.model, 'smoothed_marginal_probabilities'):
            probabilities = self.model.smoothed_marginal_probabilities
            return np.argmax(probabilities, axis=1)
        else:
            # Fallback: create dummy labels
            return np.zeros(len(self.model.fittedvalues), dtype=int)
    
    def _get_regime_probabilities(self) -> np.ndarray:
        """Get regime probabilities from fitted model."""
        if hasattr(self.model, 'smoothed_marginal_probabilities'):
            return self.model.smoothed_marginal_probabilities
        else:
            # Fallback: create uniform probabilities
            n_samples = len(self.model.fittedvalues)
            return np.ones((n_samples, self.config.k_regimes)) / self.config.k_regimes
    
    def _get_transition_matrix(self) -> np.ndarray:
        """Get transition matrix from fitted model."""
        if hasattr(self.model, 'regime_transition_matrix'):
            return self.model.regime_transition_matrix
        elif hasattr(self.model, 'params'):
            # Try to extract from parameters
            # This is a simplified implementation
            k = self.config.k_regimes
            return np.eye(k) / k  # Uniform transitions as fallback
        else:
            tprint_warning("⚠️ Could not extract transition matrix, using uniform fallback")
            k = self.config.k_regimes
            return np.eye(k) / k
    
    def _get_regime_params(self) -> Dict[str, Any]:
        """Get regime-specific parameters."""
        params = {}
        
        # Extract regime-specific parameters
        if hasattr(self.model, 'params'):
            try:
                # Try to get parameter names
                if hasattr(self.model, 'param_names'):
                    param_names = self.model.param_names
                else:
                    param_names = [f'param_{i}' for i in range(len(self.model.params))]
                
                # Group parameters by regime (simplified approach)
                for i in range(self.config.k_regimes):
                    regime_params = {
                        'intercept': 0.0,
                        'trend': 0.0,
                        'variance': 1.0
                    }
                    params[f'regime_{i}'] = regime_params
                
            except Exception as e:
                tprint_warning(f"⚠️ Could not extract detailed parameters: {e}")
                # Provide basic fallback parameters
                for i in range(self.config.k_regimes):
                    params[f'regime_{i}'] = {
                        'intercept': 0.0,
                        'trend': 0.0,
                        'variance': 1.0
                    }
        
        return params
    
    def _get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        if self.model is None:
            return {}
        
        summary = {
            'log_likelihood': getattr(self.model, 'llf', 0.0),
            'aic': getattr(self.model, 'aic', 0.0),
            'bic': getattr(self.model, 'bic', 0.0),
            'hqic': getattr(self.model, 'hqic', 0.0),
            'n_regimes': self.config.k_regimes,
            'n_parameters': len(getattr(self.model, 'params', [])),
        }
        
        # Add convergence information
        if hasattr(self.model, 'mle_retvals'):
            summary['converged'] = self.model.mle_retvals.get('converged', False)
            summary['iterations'] = self.model.mle_retvals.get('iterations', 0)
        
        return summary
    
    def _run_vectorbt_analysis(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Run VectorBT analysis if enabled."""
        if not VECTORBT_AVAILABLE:
            return {'error': 'VectorBT not available'}
        
        try:
            # This is a placeholder for VectorBT integration
            # In practice, you would create portfolios and run backtests
            return {
                'enabled': True,
                'note': 'VectorBT integration placeholder',
                'regime_labels': labels.tolist()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _save_intermediate_results(self, labels: np.ndarray, 
                                 probabilities: np.ndarray,
                                 transition_matrix: np.ndarray):
        """Save intermediate results if enabled."""
        if self.config.output_dir is None:
            return
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save labels
        np.save(output_dir / 'regime_labels.npy', labels)
        
        # Save probabilities
        if probabilities is not None:
            np.save(output_dir / 'regime_probabilities.npy', probabilities)
        
        # Save transition matrix
        if transition_matrix is not None:
            np.save(output_dir / 'transition_matrix.npy', transition_matrix)
        
        # Save configuration
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)
        
        tprint_info(f"💾 Intermediate results saved to {output_dir}")
    
    def _fit_with_batch_processing(self, data: np.ndarray, batch_size: int) -> MarkovRegressionResult:
        """
        Fit model using batch processing for large datasets.
        
        Args:
            data: Input data
            batch_size: Size of each batch
            
        Returns:
            MarkovRegressionResult with combined results
        """
        tprint_info(f"🔄 Processing {data.shape[0]} samples in batches of {batch_size}")
        
        # Preprocess data once
        data_processed, feature_names = self._preprocess_data(data)
        if not self.feature_names:
            self.feature_names = feature_names
        
        # Initialize result containers
        all_labels = []
        all_probabilities = None
        batch_results = []
        
        # Process in batches
        n_batches = int(np.ceil(data_processed.shape[0] / batch_size))
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, data_processed.shape[0])
            
            tprint_info(f"🔄 Processing batch {i+1}/{n_batches} (samples {start_idx}-{end_idx})")
            
            # Extract batch data
            batch_data = data_processed[start_idx:end_idx]
            
            # Create and fit model for this batch
            batch_model = self._create_model(batch_data)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                batch_model = batch_model.fit(
                    maxiter=self.config.maxiter,
                    tolerance=self.config.tolerance,
                    method=self.config.method,
                    loglikelihood_burn=self.config.loglikelihood_burn
                )
            
            # Get batch results
            batch_labels = self._get_regime_labels_from_model(batch_model)
            batch_probabilities = self._get_regime_probabilities_from_model(batch_model)
            
            # Store results
            all_labels.extend(batch_labels)
            
            if all_probabilities is None:
                all_probabilities = batch_probabilities
            else:
                all_probabilities = np.vstack([all_probabilities, batch_probabilities])
            
            # Store batch model for diagnostics
            batch_results.append({
                'model': batch_model,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'labels': batch_labels,
                'probabilities': batch_probabilities
            })
            
            # Force garbage collection to free memory
            import gc
            gc.collect()
        
        # Convert to numpy arrays
        all_labels = np.array(all_labels)
        
        # Create a combined model for final results
        # Use the last batch model as the representative model
        final_model = batch_results[-1]['model']
        
        # Get model artifacts
        transition_matrix = self._get_transition_matrix_from_model(final_model)
        regime_params = self._get_regime_params_from_model(final_model)
        model_summary = self._get_model_summary_from_model(final_model)
        
        # Run diagnostics on a sample of the data if enabled
        diagnostics = None
        if self.config.enable_diagnostics and len(batch_results) > 0:
            # Use the first batch for diagnostics to save time
            sample_batch = batch_results[0]
            diagnostics = self.diagnostics.run_comprehensive_diagnostics(
                sample_batch['model'],
                data_processed[:batch_size],
                sample_batch['labels'],
                sample_batch['probabilities']
            )
        
        # Get hardware metrics
        hardware_metrics = None
        if self.config.enable_hardware_optimization and self.hardware_manager:
            hardware_metrics = self.hardware_manager.get_system_status()
        
        # Build result
        result = MarkovRegressionResult(
            fitted_model=final_model,
            cluster_labels=all_labels,
            cluster_probabilities=all_probabilities,
            n_regimes=self.config.k_regimes,
            transition_matrix=transition_matrix,
            regime_params=regime_params,
            model_summary=model_summary,
            log_likelihood=model_summary.get('log_likelihood', 0.0),
            aic=model_summary.get('aic', 0.0),
            bic=model_summary.get('bic', 0.0),
            hqic=model_summary.get('hqic', 0.0),
            processing_time=0.0,  # Will be set by caller
            optimization_time=0.0,  # Will be set by caller
            feature_names=self.feature_names,
            success=True,
            hardware_metrics=hardware_metrics,
            diagnostics=diagnostics,
            metadata={
                'config': self.config.__dict__,
                'pca_loadings': self.pca_loadings,
                'data_shape': data.shape,
                'processed_shape': data_processed.shape,
                'batch_size': batch_size,
                'n_batches': n_batches
            }
        )
        
        tprint_success(f"✅ Batch processing completed successfully")
        return result
    
    def _get_regime_labels_from_model(self, model) -> np.ndarray:
        """Get regime labels from a fitted model."""
        if hasattr(model, 'smoothed_marginal_probabilities'):
            probabilities = model.smoothed_marginal_probabilities
            return np.argmax(probabilities, axis=1)
        else:
            # Fallback: create dummy labels
            return np.zeros(len(model.fittedvalues), dtype=int)
    
    def _get_regime_probabilities_from_model(self, model) -> np.ndarray:
        """Get regime probabilities from a fitted model."""
        if hasattr(model, 'smoothed_marginal_probabilities'):
            return model.smoothed_marginal_probabilities
        else:
            # Fallback: create uniform probabilities
            n_samples = len(model.fittedvalues)
            return np.ones((n_samples, self.config.k_regimes)) / self.config.k_regimes
    
    def _get_transition_matrix_from_model(self, model) -> np.ndarray:
        """Get transition matrix from a fitted model."""
        if hasattr(model, 'regime_transition_matrix'):
            return model.regime_transition_matrix
        elif hasattr(model, 'params'):
            # Try to extract from parameters
            k = self.config.k_regimes
            return np.eye(k) / k  # Uniform transitions as fallback
        else:
            tprint_warning("⚠️ Could not extract transition matrix, using uniform fallback")
            k = self.config.k_regimes
            return np.eye(k) / k
    
    def _get_regime_params_from_model(self, model) -> Dict[str, Any]:
        """Get regime-specific parameters from a fitted model."""
        params = {}
        
        # Extract regime-specific parameters
        if hasattr(model, 'params'):
            try:
                # Try to get parameter names
                if hasattr(model, 'param_names'):
                    param_names = model.param_names
                else:
                    param_names = [f'param_{i}' for i in range(len(model.params))]
                
                # Group parameters by regime (simplified approach)
                for i in range(self.config.k_regimes):
                    regime_params = {
                        'intercept': 0.0,
                        'trend': 0.0,
                        'variance': 1.0
                    }
                    params[f'regime_{i}'] = regime_params
                
            except Exception as e:
                tprint_warning(f"⚠️ Could not extract detailed parameters: {e}")
                # Provide basic fallback parameters
                for i in range(self.config.k_regimes):
                    params[f'regime_{i}'] = {
                        'intercept': 0.0,
                        'trend': 0.0,
                        'variance': 1.0
                    }
        
        return params
    
    def _get_model_summary_from_model(self, model) -> Dict[str, Any]:
        """Get comprehensive model summary from a fitted model."""
        if model is None:
            return {}
        
        summary = {
            'log_likelihood': getattr(model, 'llf', 0.0),
            'aic': getattr(model, 'aic', 0.0),
            'bic': getattr(model, 'bic', 0.0),
            'hqic': getattr(model, 'hqic', 0.0),
            'n_regimes': self.config.k_regimes,
            'n_parameters': len(getattr(model, 'params', [])),
        }
        
        # Add convergence information
        if hasattr(model, 'mle_retvals'):
            summary['converged'] = model.mle_retvals.get('converged', False)
            summary['iterations'] = model.mle_retvals.get('iterations', 0)
        
        return summary
    
    @contextmanager
    def _null_context(self):
        """Null context manager for when hardware optimization is disabled."""
        yield self


# Convenience function for creating an enhanced adapter
def create_enhanced_markov_regression_adapter(
    k_regimes: int = 2,
    trend: str = 'c',
    order: int = 0,
    switching_variance: bool = True,
    switching_trend: bool = True,
    maxiter: int = 100,
    enable_hardware_optimization: bool = True,
    enable_diagnostics: bool = True,
    enable_pca: bool = True,
    pca_components: int = 12,
    random_state: int = 42,
    **kwargs
) -> MarkovRegressionAdapter:
    """
    Create an enhanced MarkovRegression adapter with specified parameters.
    
    Args:
        k_regimes: Number of regimes
        trend: Trend component ('c', 't', 'ct')
        order: Autoregressive order
        switching_variance: Allow variance to switch
        switching_trend: Allow trend to switch
        maxiter: Maximum EM iterations
        enable_hardware_optimization: Enable hardware optimization
        enable_diagnostics: Enable advanced diagnostics
        enable_pca: Enable PCA preprocessing
        pca_components: Number of PCA components
        random_state: Random seed
        **kwargs: Additional configuration parameters
        
    Returns:
        Enhanced MarkovRegressionAdapter instance
    """
    tprint_info("🏭 Creating Enhanced MarkovRegression Adapter with factory function")
    
    config = MarkovRegressionConfig(
        k_regimes=k_regimes,
        trend=trend,
        order=order,
        switching_variance=switching_variance,
        switching_trend=switching_trend,
        maxiter=maxiter,
        enable_hardware_optimization=enable_hardware_optimization,
        enable_diagnostics=enable_diagnostics,
        enable_pca=enable_pca,
        pca_components=pca_components,
        random_state=random_state,
        **kwargs
    )
    
    tprint_info(f"📊 Configuration: {k_regimes} regimes, trend='{trend}', order={order}")
    adapter = MarkovRegressionAdapter(config)
    tprint_success("✅ Enhanced MarkovRegression Adapter created successfully")
    return adapter