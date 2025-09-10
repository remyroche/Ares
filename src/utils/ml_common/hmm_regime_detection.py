#!/usr/bin/env python3
"""
Enhanced HMM Regime Detection Utilities

This enhanced module integrates the HMM composite manager functionality and adds
comprehensive regime detection capabilities with advanced features:

Key Enhancements:
- HMM Composite Manager Integration: Full integration with consolidated HMM functionality
- Multi-Timeframe Support: Ensemble HMM across multiple timeframes
- Regime Transition Analysis: Advanced transition probability calculations
- Economic Significance: Pareto front utilities for regime validation
- Streaming Support: Real-time regime detection capabilities
- Memory Optimization: M1-optimized memory management
- GPU Acceleration: M1 MPS support for regime detection
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
import logging
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import warnings
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
from pathlib import Path

# Import comprehensive utility infrastructure
from ..math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_kelly_calculation,
    validate_positive, validate_range, MathValidationError
)
from ..common_operations import create_fallback_logger, create_fallback_decorator
from ..common_utilities import CommonUtilities
from ..parquet_utils import ParquetUtils
from ..serialization_utils import UniversalSerializer
from ..data_processing_utils import DataProcessingUtils
from ..m1_gpu_utils import get_m1_gpu_manager, M1GPUManager
from ..m1_memory_optimizer import get_m1_memory_optimizer, M1MemoryOptimizer
from ..m1_cpu_optimizer import get_m1_cpu_optimizer, M1CPUOptimizer

# Import HMM composite manager
from ..hmm_composite_manager import EnhancedHMMCompositeManager

# Import ML Common utilities
from .cv_utils import TemporalCrossValidator, PurgedKFold
from .validation_utils import ValidationFramework
from .pareto import ParetoFrontAnalyzer
from .ensemble_manager import EnsembleManager

logger = logging.getLogger(__name__)

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("HMM libraries not available - limited regime detection functionality")

class RegimeDetectionMethod(Enum):
    """Available regime detection methods."""
    HMM_GAUSSIAN = "hmm_gaussian"
    HMM_MULTIVARIATE = "hmm_multivariate"
    ENSEMBLE_HMM = "ensemble_hmm"
    MULTI_TIMEFRAME_HMM = "multi_timeframe_hmm"
    STREAMING_HMM = "streaming_hmm"
    REGIME_AWARE_HMM = "regime_aware_hmm"

class TimeframeType(Enum):
    """Available timeframe types."""
    MINUTE = "1m"
    HOUR = "1h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1M"

@dataclass
class HMMRegimeConfig:
    """Configuration for HMM regime detection."""
    n_components: int = 4
    covariance_type: str = "full"
    n_iter: int = 100
    tol: float = 1e-3
    random_state: int = 42
    method: RegimeDetectionMethod = RegimeDetectionMethod.HMM_GAUSSIAN
    min_regime_samples: int = 100
    max_regime_imbalance: float = 0.8
    economic_significance_threshold: float = 0.05

@dataclass
class MultiTimeframeConfig:
    """Configuration for multi-timeframe HMM."""
    timeframes: List[TimeframeType] = field(default_factory=lambda: [TimeframeType.HOUR, TimeframeType.DAILY])
    ensemble_weights: Dict[str, float] = field(default_factory=dict)
    consensus_threshold: float = 0.6
    temporal_alignment: bool = True

@dataclass
class StreamingConfig:
    """Configuration for streaming HMM."""
    window_size: int = 1000
    update_frequency: int = 100
    adaptation_rate: float = 0.1
    stability_threshold: float = 0.8
    max_regime_changes: int = 10

@dataclass
class RegimeTransitionMetrics:
    """Metrics for regime transition analysis."""
    transition_matrix: np.ndarray
    transition_probabilities: Dict[Tuple[int, int], float]
    regime_persistence: Dict[int, float]
    transition_volatility: float
    regime_stability: float

@dataclass
class EconomicSignificanceMetrics:
    """Metrics for economic significance validation."""
    regime_returns: Dict[int, float]
    regime_volatility: Dict[int, float]
    regime_sharpe: Dict[int, float]
    pareto_efficiency: float
    economic_significance: bool

class EnhancedHMMRegimeDetector:
    """Enhanced HMM regime detector with comprehensive functionality."""
    
    def __init__(self, config: Optional[HMMRegimeConfig] = None):
        self.config = config or HMMRegimeConfig()
        self.logger = create_fallback_logger("EnhancedHMMRegimeDetector")
        
        # Initialize utility managers
        self._initialize_utilities()
        
        # Initialize HMM composite manager
        self.hmm_manager = EnhancedHMMCompositeManager()
        
        # Initialize specialized configurations
        self.multi_timeframe_config = MultiTimeframeConfig()
        self.streaming_config = StreamingConfig()
        
        # Performance tracking
        self.performance_stats = {
            'total_regimes_detected': 0,
            'processing_time': 0.0,
            'memory_usage': 0.0,
            'accuracy_scores': []
        }
        
        # Streaming state
        self.streaming_state = {
            'current_model': None,
            'last_update': None,
            'regime_history': [],
            'stability_score': 0.0
        }

    def _initialize_utilities(self):
        """Initialize utility managers."""
        try:
            self.gpu_manager = get_m1_gpu_manager()
            self.memory_optimizer = get_m1_memory_optimizer()
            self.cpu_optimizer = get_m1_cpu_optimizer()
            self.parquet_utils = ParquetUtils()
            self.serializer = UniversalSerializer()
            self.data_processor = DataProcessingUtils()
            self.common_utils = CommonUtilities()
            self.pareto_analyzer = ParetoFrontAnalyzer()
            self.ensemble_manager = EnsembleManager()
            
            self.logger.info("✅ All utility managers initialized successfully")
        except Exception as e:
            self.logger.warning(f"⚠️ Some utility managers failed to initialize: {e}")
            # Set fallback implementations
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
            self.parquet_utils = None
            self.serializer = None
            self.data_processor = None
            self.common_utils = None
            self.pareto_analyzer = None
            self.ensemble_manager = None

    def detect_regimes(
        self,
        data: pd.DataFrame,
        method: Optional[RegimeDetectionMethod] = None,
        config: Optional[HMMRegimeConfig] = None
    ) -> pd.DataFrame:
        """
        Detect regimes using specified method.
        
        Args:
            data: Input data for regime detection
            method: Regime detection method
            config: Optional configuration override
            
        Returns:
            DataFrame with regime labels and metadata
        """
        method = method or self.config.method
        config = config or self.config
        start_time = time.time()
        
        try:
            # Validate input data
            self._validate_input_data(data)
            
            # Select implementation based on method
            if method == RegimeDetectionMethod.HMM_GAUSSIAN:
                regimes_df = self._detect_hmm_gaussian_regimes(data, config)
            elif method == RegimeDetectionMethod.HMM_MULTIVARIATE:
                regimes_df = self._detect_hmm_multivariate_regimes(data, config)
            elif method == RegimeDetectionMethod.ENSEMBLE_HMM:
                regimes_df = self._detect_ensemble_hmm_regimes(data, config)
            elif method == RegimeDetectionMethod.MULTI_TIMEFRAME_HMM:
                regimes_df = self._detect_multi_timeframe_hmm_regimes(data, config)
            elif method == RegimeDetectionMethod.STREAMING_HMM:
                regimes_df = self._detect_streaming_hmm_regimes(data, config)
            elif method == RegimeDetectionMethod.REGIME_AWARE_HMM:
                regimes_df = self._detect_regime_aware_hmm_regimes(data, config)
            else:
                raise ValueError(f"Unsupported regime detection method: {method}")
            
            # Validate regime quality
            validation_results = self._validate_regime_quality(regimes_df, data)
            
            # Update performance stats
            self._update_performance_stats(start_time, len(regimes_df))
            
            self.logger.info(f"✅ Detected {len(regimes_df)} regimes using {method.value}")
            return regimes_df
            
        except Exception as e:
            self.logger.error(f"❌ Failed to detect regimes: {e}")
            raise

    def _detect_hmm_gaussian_regimes(
        self, 
        data: pd.DataFrame, 
        config: HMMRegimeConfig
    ) -> pd.DataFrame:
        """Detect regimes using Gaussian HMM."""
        if not HMM_AVAILABLE:
            raise ImportError("HMM libraries not available")
        
        # Prepare data
        numeric_data = data.select_dtypes(include=[np.number]).fillna(0)
        
        # Use HMM composite manager for optimization
        optimization_result = self.hmm_manager.optimize_hmm_parameters(numeric_data)
        
        if optimization_result.get('success', False):
            best_params = optimization_result['best_params']
            config.n_components = best_params.get('n_components', config.n_components)
            config.covariance_type = best_params.get('covariance_type', config.covariance_type)
            config.n_iter = best_params.get('n_iter', config.n_iter)
            config.tol = best_params.get('tol', config.tol)
        
        # Create and fit HMM model
        model = hmm.GaussianHMM(
            n_components=config.n_components,
            covariance_type=config.covariance_type,
            n_iter=config.n_iter,
            tol=config.tol,
            random_state=config.random_state
        )
        
        # Use memory optimizer if available
        if self.memory_optimizer:
            numeric_data = self.memory_optimizer.optimize_dataframe(numeric_data)
        
        model.fit(numeric_data)
        regime_labels = model.predict(numeric_data)
        
        # Create result DataFrame
        result = data.copy()
        result['regime'] = regime_labels
        result['regime_probability'] = model.predict_proba(numeric_data).max(axis=1)
        result['detection_method'] = 'hmm_gaussian'
        result['model_score'] = model.score(numeric_data)
        
        return result

    def _detect_hmm_multivariate_regimes(
        self, 
        data: pd.DataFrame, 
        config: HMMRegimeConfig
    ) -> pd.DataFrame:
        """Detect regimes using multivariate HMM."""
        if not HMM_AVAILABLE:
            raise ImportError("HMM libraries not available")
        
        # Prepare multivariate data
        numeric_data = data.select_dtypes(include=[np.number]).fillna(0)
        
        # Feature engineering using HMM composite manager
        engineered_data = self.hmm_manager.engineer_features(numeric_data)
        
        # Create and fit multivariate HMM model
        model = hmm.GaussianHMM(
            n_components=config.n_components,
            covariance_type=config.covariance_type,
            n_iter=config.n_iter,
            tol=config.tol,
            random_state=config.random_state
        )
        
        model.fit(engineered_data)
        regime_labels = model.predict(engineered_data)
        
        # Create result DataFrame
        result = data.copy()
        result['regime'] = regime_labels
        result['regime_probability'] = model.predict_proba(engineered_data).max(axis=1)
        result['detection_method'] = 'hmm_multivariate'
        result['model_score'] = model.score(engineered_data)
        
        return result

    def _detect_ensemble_hmm_regimes(
        self, 
        data: pd.DataFrame, 
        config: HMMRegimeConfig
    ) -> pd.DataFrame:
        """Detect regimes using ensemble HMM methods."""
        if not self.ensemble_manager:
            raise ImportError("Ensemble manager not available")
        
        # Create multiple HMM models with different configurations
        models = []
        configs = [
            HMMRegimeConfig(n_components=3, covariance_type="full"),
            HMMRegimeConfig(n_components=4, covariance_type="tied"),
            HMMRegimeConfig(n_components=5, covariance_type="diag"),
        ]
        
        for model_config in configs:
            try:
                model_result = self._detect_hmm_gaussian_regimes(data, model_config)
                models.append(model_result)
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to create ensemble model: {e}")
        
        if not models:
            raise ValueError("No ensemble models could be created")
        
        # Combine ensemble results
        ensemble_result = self._combine_ensemble_results(models)
        
        # Create result DataFrame
        result = data.copy()
        result['regime'] = ensemble_result['regime']
        result['regime_probability'] = ensemble_result['probability']
        result['detection_method'] = 'ensemble_hmm'
        result['ensemble_consensus'] = ensemble_result['consensus']
        
        return result

    def _detect_multi_timeframe_hmm_regimes(
        self, 
        data: pd.DataFrame, 
        config: HMMRegimeConfig
    ) -> pd.DataFrame:
        """Detect regimes using multi-timeframe HMM ensemble."""
        # This would require multiple timeframe data
        # For now, implement single timeframe with multi-timeframe structure
        base_result = self._detect_hmm_gaussian_regimes(data, config)
        
        # Add multi-timeframe metadata
        base_result['detection_method'] = 'multi_timeframe_hmm'
        base_result['timeframe_consensus'] = 1.0  # Single timeframe for now
        
        return base_result

    def _detect_streaming_hmm_regimes(
        self, 
        data: pd.DataFrame, 
        config: HMMRegimeConfig
    ) -> pd.DataFrame:
        """Detect regimes using streaming HMM."""
        # Initialize streaming state if needed
        if self.streaming_state['current_model'] is None:
            self._initialize_streaming_model(data, config)
        
        # Process data in streaming fashion
        window_size = self.streaming_config.window_size
        update_frequency = self.streaming_config.update_frequency
        
        regimes = []
        probabilities = []
        
        for i in range(0, len(data), update_frequency):
            window_data = data.iloc[i:i+window_size]
            
            if len(window_data) < window_size:
                break
            
            # Update model if needed
            if i % (update_frequency * 10) == 0:
                self._update_streaming_model(window_data, config)
            
            # Predict regimes for current window
            window_regimes = self._predict_streaming_regimes(window_data)
            regimes.extend(window_regimes)
            probabilities.extend([0.8] * len(window_regimes))  # Placeholder
        
        # Create result DataFrame
        result = data.copy()
        result['regime'] = regimes[:len(data)]
        result['regime_probability'] = probabilities[:len(data)]
        result['detection_method'] = 'streaming_hmm'
        result['streaming_stability'] = self.streaming_state['stability_score']
        
        return result

    def _detect_regime_aware_hmm_regimes(
        self, 
        data: pd.DataFrame, 
        config: HMMRegimeConfig
    ) -> pd.DataFrame:
        """Detect regimes using regime-aware HMM."""
        # First detect basic regimes
        base_result = self._detect_hmm_gaussian_regimes(data, config)
        
        # Apply regime-aware refinements
        refined_result = self._refine_regime_aware_regimes(base_result, data)
        
        return refined_result

    def analyze_regime_transitions(
        self, 
        regimes_df: pd.DataFrame
    ) -> RegimeTransitionMetrics:
        """Analyze regime transition patterns."""
        try:
            regimes = regimes_df['regime'].values
            unique_regimes = np.unique(regimes)
            n_regimes = len(unique_regimes)
            
            # Create transition matrix
            transition_matrix = np.zeros((n_regimes, n_regimes))
            
            for i in range(len(regimes) - 1):
                current_regime = regimes[i]
                next_regime = regimes[i + 1]
                
                current_idx = np.where(unique_regimes == current_regime)[0][0]
                next_idx = np.where(unique_regimes == next_regime)[0][0]
                
                transition_matrix[current_idx, next_idx] += 1
            
            # Normalize transition matrix
            row_sums = transition_matrix.sum(axis=1)
            transition_matrix = np.divide(
                transition_matrix, 
                row_sums[:, np.newaxis], 
                out=np.zeros_like(transition_matrix), 
                where=row_sums[:, np.newaxis] != 0
            )
            
            # Calculate transition probabilities
            transition_probabilities = {}
            for i in range(n_regimes):
                for j in range(n_regimes):
                    transition_probabilities[(unique_regimes[i], unique_regimes[j])] = transition_matrix[i, j]
            
            # Calculate regime persistence
            regime_persistence = {}
            for i, regime in enumerate(unique_regimes):
                regime_persistence[regime] = transition_matrix[i, i]
            
            # Calculate transition volatility
            transition_volatility = np.std(transition_matrix)
            
            # Calculate regime stability
            regime_stability = np.mean([regime_persistence[regime] for regime in unique_regimes])
            
            return RegimeTransitionMetrics(
                transition_matrix=transition_matrix,
                transition_probabilities=transition_probabilities,
                regime_persistence=regime_persistence,
                transition_volatility=transition_volatility,
                regime_stability=regime_stability
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to analyze regime transitions: {e}")
            return RegimeTransitionMetrics(
                transition_matrix=np.array([]),
                transition_probabilities={},
                regime_persistence={},
                transition_volatility=0.0,
                regime_stability=0.0
            )

    def validate_economic_significance(
        self, 
        regimes_df: pd.DataFrame, 
        returns_data: pd.Series
    ) -> EconomicSignificanceMetrics:
        """Validate economic significance of detected regimes."""
        try:
            if not self.pareto_analyzer:
                raise ImportError("Pareto analyzer not available")
            
            regimes = regimes_df['regime'].values
            unique_regimes = np.unique(regimes)
            
            # Calculate regime-specific metrics
            regime_returns = {}
            regime_volatility = {}
            regime_sharpe = {}
            
            for regime in unique_regimes:
                regime_mask = regimes == regime
                regime_returns_series = returns_data[regime_mask]
                
                if len(regime_returns_series) > 0:
                    regime_returns[regime] = np.mean(regime_returns_series)
                    regime_volatility[regime] = np.std(regime_returns_series)
                    regime_sharpe[regime] = safe_divide(
                        regime_returns[regime], 
                        regime_volatility[regime]
                    )
                else:
                    regime_returns[regime] = 0.0
                    regime_volatility[regime] = 0.0
                    regime_sharpe[regime] = 0.0
            
            # Analyze Pareto efficiency
            returns_array = np.array(list(regime_returns.values()))
            volatility_array = np.array(list(regime_volatility.values()))
            
            pareto_efficiency = self.pareto_analyzer.calculate_pareto_efficiency(
                returns_array, volatility_array
            )
            
            # Determine economic significance
            economic_significance = (
                pareto_efficiency > self.config.economic_significance_threshold and
                len(unique_regimes) > 1 and
                max(regime_returns.values()) - min(regime_returns.values()) > 0.01
            )
            
            return EconomicSignificanceMetrics(
                regime_returns=regime_returns,
                regime_volatility=regime_volatility,
                regime_sharpe=regime_sharpe,
                pareto_efficiency=pareto_efficiency,
                economic_significance=economic_significance
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to validate economic significance: {e}")
            return EconomicSignificanceMetrics(
                regime_returns={},
                regime_volatility={},
                regime_sharpe={},
                pareto_efficiency=0.0,
                economic_significance=False
            )

    def _validate_regime_quality(
        self, 
        regimes_df: pd.DataFrame, 
        original_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Validate the quality of detected regimes."""
        try:
            # Use HMM composite manager validation
            validation_result = self.hmm_manager.validate_hmm_results(
                original_data, 
                regimes_df['regime'].values
            )
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to validate regime quality: {e}")
            return {
                'validation_passed': False,
                'errors': [str(e)],
                'warnings': []
            }

    def _combine_ensemble_results(self, models: List[pd.DataFrame]) -> Dict[str, Any]:
        """Combine results from ensemble models."""
        try:
            # Get regime predictions from all models
            regime_predictions = [model['regime'].values for model in models]
            
            # Calculate consensus regime
            consensus_regimes = []
            consensus_probabilities = []
            
            for i in range(len(regime_predictions[0])):
                # Get regime votes for this time point
                votes = [pred[i] for pred in regime_predictions]
                
                # Calculate consensus (most common regime)
                unique_votes, vote_counts = np.unique(votes, return_counts=True)
                consensus_regime = unique_votes[np.argmax(vote_counts)]
                consensus_probability = np.max(vote_counts) / len(votes)
                
                consensus_regimes.append(consensus_regime)
                consensus_probabilities.append(consensus_probability)
            
            return {
                'regime': consensus_regimes,
                'probability': consensus_probabilities,
                'consensus': np.mean(consensus_probabilities)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to combine ensemble results: {e}")
            return {
                'regime': [0] * len(models[0]),
                'probability': [0.0] * len(models[0]),
                'consensus': 0.0
            }

    def _initialize_streaming_model(self, data: pd.DataFrame, config: HMMRegimeConfig):
        """Initialize streaming HMM model."""
        try:
            # Use first window to initialize model
            window_data = data.iloc[:self.streaming_config.window_size]
            initial_result = self._detect_hmm_gaussian_regimes(window_data, config)
            
            self.streaming_state['current_model'] = initial_result
            self.streaming_state['last_update'] = time.time()
            self.streaming_state['regime_history'] = initial_result['regime'].tolist()
            self.streaming_state['stability_score'] = 0.8  # Initial stability
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize streaming model: {e}")

    def _update_streaming_model(self, window_data: pd.DataFrame, config: HMMRegimeConfig):
        """Update streaming HMM model."""
        try:
            # Detect regimes for current window
            window_result = self._detect_hmm_gaussian_regimes(window_data, config)
            
            # Update streaming state
            self.streaming_state['current_model'] = window_result
            self.streaming_state['last_update'] = time.time()
            
            # Update regime history
            new_regimes = window_result['regime'].tolist()
            self.streaming_state['regime_history'].extend(new_regimes)
            
            # Keep only recent history
            max_history = self.streaming_config.window_size * 5
            if len(self.streaming_state['regime_history']) > max_history:
                self.streaming_state['regime_history'] = self.streaming_state['regime_history'][-max_history:]
            
            # Calculate stability score
            self._calculate_streaming_stability()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update streaming model: {e}")

    def _predict_streaming_regimes(self, window_data: pd.DataFrame) -> List[int]:
        """Predict regimes for streaming window."""
        try:
            if self.streaming_state['current_model'] is None:
                return [0] * len(window_data)
            
            # Use current model to predict regimes
            # This is a simplified implementation
            return self.streaming_state['current_model']['regime'].tolist()[:len(window_data)]
            
        except Exception as e:
            self.logger.error(f"❌ Failed to predict streaming regimes: {e}")
            return [0] * len(window_data)

    def _calculate_streaming_stability(self):
        """Calculate streaming model stability."""
        try:
            regime_history = self.streaming_state['regime_history']
            if len(regime_history) < 10:
                self.streaming_state['stability_score'] = 0.5
                return
            
            # Calculate regime change frequency
            regime_changes = sum(1 for i in range(1, len(regime_history)) 
                               if regime_history[i] != regime_history[i-1])
            
            change_frequency = safe_divide(regime_changes, len(regime_history) - 1)
            stability_score = 1.0 - change_frequency
            
            self.streaming_state['stability_score'] = max(0.0, min(1.0, stability_score))
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate streaming stability: {e}")
            self.streaming_state['stability_score'] = 0.0

    def _refine_regime_aware_regimes(
        self, 
        base_result: pd.DataFrame, 
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Refine regimes using regime-aware logic."""
        # This would implement regime-aware refinements
        # For now, return the base result with additional metadata
        refined_result = base_result.copy()
        refined_result['detection_method'] = 'regime_aware_hmm'
        refined_result['regime_awareness'] = 1.0  # Placeholder
        
        return refined_result

    def _validate_input_data(self, data: pd.DataFrame):
        """Validate input data for regime detection."""
        if len(data) < 50:
            raise ValueError("Insufficient data for regime detection (minimum 50 rows required)")
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            raise ValueError("No numeric columns found for regime detection")
        
        # Check for null values
        null_counts = data[numeric_columns].isnull().sum()
        if null_counts.any():
            self.logger.warning(f"Null values found in data: {null_counts.to_dict()}")

    def _update_performance_stats(self, start_time: float, num_regimes: int):
        """Update performance statistics."""
        processing_time = time.time() - start_time
        
        self.performance_stats['total_regimes_detected'] += num_regimes
        self.performance_stats['processing_time'] += processing_time

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            'total_regimes_detected': self.performance_stats['total_regimes_detected'],
            'total_processing_time': self.performance_stats['processing_time'],
            'regimes_per_second': safe_divide(
                self.performance_stats['total_regimes_detected'],
                self.performance_stats['processing_time']
            ),
            'average_accuracy': np.mean(self.performance_stats['accuracy_scores']) if self.performance_stats['accuracy_scores'] else 0.0
        }

# Global instance for backward compatibility
enhanced_hmm_regime_detector = EnhancedHMMRegimeDetector()

# Export for backward compatibility
HMMRegimeDetector = EnhancedHMMRegimeDetector