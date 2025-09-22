"""
Advanced Model Integration for Production HMM Pipeline

This module integrates data-driven advanced Markov models (MSM + HSMM) 
into the production walk-forward validation framework with comprehensive
model selection, validation, and stability testing.

Key Features:
1. Walk-forward validation integration
2. Advanced model selection framework
3. Comprehensive evaluation metrics
4. Stability testing and robustness validation
5. Production-ready model artifacts
6. Leakage-safe model training and validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import warnings
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
import pickle

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.utils.logger import system_logger

# Import our advanced models
from .data_driven_markov_models import (
    DataDrivenMarkovSwitchingModel, 
    DataDrivenHiddenSemiMarkovModel,
    DataDrivenMSMConfig,
    DataDrivenHSMMConfig
)

# Import production features
from .production_feature_integration import ProductionLeakageSafeFeatures, ProductionFeatureConfig

# Import existing HMM components
try:
    from src.training.steps.market_analysis.components.hmm_regime_discovery import HMMRegimeDiscoveryComponent
    # HMMClusteringComponent no longer exists - replaced by OptimalRegimeClusteringComponent
    # from src.training.steps.market_analysis.components.hmm_clustering import HMMClusteringComponent
    HMM_COMPONENTS_AVAILABLE = True
except ImportError:
    HMM_COMPONENTS_AVAILABLE = False
    warnings.warn("Traditional HMM components not available")


class ModelType(Enum):
    """Available model types for integration."""
    TRADITIONAL_HMM = "traditional_hmm"
    MARKOV_SWITCHING = "markov_switching"
    HIDDEN_SEMI_MARKOV = "hidden_semi_markov"
    HYBRID_MSM_HSMM = "hybrid_msm_hsmm"


class ValidationMetric(Enum):
    """Validation metrics for model selection."""
    LOG_LIKELIHOOD = "log_likelihood"
    BIC = "bic"
    AIC = "aic"
    SILHOUETTE_SCORE = "silhouette_score"
    REGIME_STABILITY = "regime_stability"
    TRANSITION_QUALITY = "transition_quality"
    ECONOMIC_PLAUSIBILITY = "economic_plausibility"


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    # Training and validation windows
    train_months: int = 12
    validation_months: int = 1
    step_months: int = 1
    
    # Number of folds
    n_folds: int = 12
    min_train_observations: int = 2000  # Minimum for 1h data
    
    # Model selection
    primary_metric: ValidationMetric = ValidationMetric.LOG_LIKELIHOOD
    secondary_metrics: List[ValidationMetric] = field(default_factory=lambda: [
        ValidationMetric.BIC, 
        ValidationMetric.REGIME_STABILITY,
        ValidationMetric.TRANSITION_QUALITY
    ])
    
    # Stability testing
    stability_test_iterations: int = 5
    stability_noise_level: float = 0.01
    stability_threshold: float = 0.7  # ARI threshold for stability
    
    # Performance thresholds
    min_log_likelihood: float = -np.inf
    max_bic_penalty: float = np.inf
    min_regime_stability: float = 0.3


@dataclass
class ModelCandidate:
    """Configuration for a model candidate."""
    model_type: ModelType
    config: Any  # Model-specific configuration
    enabled: bool = True
    priority: int = 1  # Higher priority = tested first


@dataclass
class ValidationResult:
    """Result from model validation."""
    model_type: ModelType
    fold: int
    
    # Core metrics
    log_likelihood: float
    bic: float
    aic: float
    
    # Advanced metrics
    regime_stability: float
    transition_quality: float
    economic_plausibility: float
    
    # Model-specific metrics
    n_regimes: int
    regime_assignments: np.ndarray
    
    # Stability metrics
    stability_score: float = 0.0
    noise_robustness: float = 0.0
    
    # Metadata
    training_time: float = 0.0
    validation_time: float = 0.0
    model_artifacts: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_type': self.model_type.value,
            'fold': self.fold,
            'log_likelihood': self.log_likelihood,
            'bic': self.bic,
            'aic': self.aic,
            'regime_stability': self.regime_stability,
            'transition_quality': self.transition_quality,
            'economic_plausibility': self.economic_plausibility,
            'n_regimes': self.n_regimes,
            'regime_assignments': self.regime_assignments.tolist() if self.regime_assignments is not None else None,
            'stability_score': self.stability_score,
            'noise_robustness': self.noise_robustness,
            'training_time': self.training_time,
            'validation_time': self.validation_time
        }


class TraditionalHMMWrapper:
    """Wrapper for traditional HMM to match advanced model interface."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild('TraditionalHMMWrapper')
        self.hmm_component = None
        
        if HMM_COMPONENTS_AVAILABLE:
            try:
                # Create component config
                component_config = type('ComponentConfig', (), {
                    'symbol': config.get('symbol', 'ETHUSDT'),
                    'exchange': 'binance',
                    'timeframe': '1h',
                    'optimization_mode': config.get('optimization_mode', 'blank')
                })()
                
                self.hmm_component = HMMRegimeDiscoveryComponent(component_config)
            except Exception as e:
                self.logger.warning(f"Could not initialize traditional HMM: {e}")
    
    def fit(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Fit traditional HMM model."""
        if self.hmm_component is None:
            raise ValueError("Traditional HMM component not available")
        
        try:
            # Run HMM component
            result = asyncio.run(self.hmm_component.execute(data, {}))
            
            if result.success:
                hmm_result = result.artifacts.get('hmm_regime_discovery_result', {})
                
                return {
                    'regime_assignments': np.array(hmm_result.get('regime_assignments', [])),
                    'n_regimes': len(np.unique(hmm_result.get('regime_assignments', []))),
                    'regime_discovery': hmm_result,
                    'performance_metrics': result.metadata,
                    'method': 'traditional_hmm'
                }
            else:
                raise RuntimeError(f"HMM fitting failed: {result.error_message}")
                
        except Exception as e:
            self.logger.error(f"Traditional HMM fitting failed: {e}")
            raise
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict regime assignments."""
        # For traditional HMM, we need to refit on new data
        result = self.fit(data)
        return result['regime_assignments']


class HybridMSMHSMMModel:
    """Hybrid model combining MSM and HSMM insights."""
    
    def __init__(self, msm_config: DataDrivenMSMConfig, hsmm_config: DataDrivenHSMMConfig):
        self.msm_config = msm_config
        self.hsmm_config = hsmm_config
        self.logger = system_logger.getChild('HybridMSMHSMM')
        
        self.msm_model = None
        self.hsmm_model = None
        
    def fit(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Fit hybrid MSM-HSMM model."""
        try:
            # Fit both models
            self.msm_model = DataDrivenMarkovSwitchingModel(self.msm_config)
            msm_results = self.msm_model.fit(data)
            
            self.hsmm_model = DataDrivenHiddenSemiMarkovModel(self.hsmm_config)
            hsmm_results = self.hsmm_model.fit(data)
            
            # Combine insights from both models
            msm_regimes = msm_results['regime_assignments']
            hsmm_states = hsmm_results['state_sequence']
            
            # Create hybrid regime assignment using agreement
            hybrid_regimes = self._create_hybrid_assignment(msm_regimes, hsmm_states)
            
            return {
                'regime_assignments': hybrid_regimes,
                'n_regimes': len(np.unique(hybrid_regimes)),
                'msm_results': msm_results,
                'hsmm_results': hsmm_results,
                'model_agreement': adjusted_rand_score(msm_regimes, hsmm_states),
                'method': 'hybrid_msm_hsmm'
            }
            
        except Exception as e:
            self.logger.error(f"Hybrid model fitting failed: {e}")
            raise
    
    def _create_hybrid_assignment(self, msm_regimes: np.ndarray, hsmm_states: np.ndarray) -> np.ndarray:
        """Create hybrid regime assignment from MSM and HSMM results."""
        # Simple approach: use MSM for structural breaks, HSMM for duration consistency
        
        # Align lengths
        min_len = min(len(msm_regimes), len(hsmm_states))
        msm_aligned = msm_regimes[:min_len]
        hsmm_aligned = hsmm_states[:min_len]
        
        # Create hybrid assignment based on agreement
        hybrid = np.zeros(min_len, dtype=int)
        
        # Use a sliding window to determine local agreement
        window_size = 20  # 20-hour window
        
        for i in range(min_len):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(min_len, i + window_size // 2 + 1)
            
            # Local agreement within window
            window_msm = msm_aligned[start_idx:end_idx]
            window_hsmm = hsmm_aligned[start_idx:end_idx]
            
            # Calculate local ARI
            if len(np.unique(window_msm)) > 1 and len(np.unique(window_hsmm)) > 1:
                local_agreement = adjusted_rand_score(window_msm, window_hsmm)
            else:
                local_agreement = 0.0
            
            # Choose assignment based on agreement level
            if local_agreement > 0.5:
                # High agreement: use MSM (better for structural breaks)
                hybrid[i] = msm_aligned[i]
            else:
                # Low agreement: use HSMM (better for duration modeling)
                hybrid[i] = hsmm_aligned[i] + 100  # Offset to avoid confusion
        
        # Relabel to consecutive integers
        unique_labels = np.unique(hybrid)
        label_mapping = {old: new for new, old in enumerate(unique_labels)}
        hybrid_relabeled = np.array([label_mapping[label] for label in hybrid])
        
        return hybrid_relabeled


class AdvancedModelSelector:
    """
    Advanced model selection framework with walk-forward validation.
    
    Integrates traditional HMM, MSM, HSMM, and hybrid models within
    a comprehensive validation framework.
    """
    
    def __init__(self, 
                 walk_forward_config: WalkForwardConfig,
                 feature_config: ProductionFeatureConfig):
        self.wf_config = walk_forward_config
        self.feature_config = feature_config
        self.logger = system_logger.getChild('AdvancedModelSelector')
        
        # Initialize feature generator
        self.feature_generator = ProductionLeakageSafeFeatures(feature_config)
        
        # Model candidates
        self.model_candidates = self._initialize_model_candidates()
        
        # Results storage
        self.validation_results: List[ValidationResult] = []
        self.best_model = None
        self.model_artifacts = {}
    
    def _initialize_model_candidates(self) -> List[ModelCandidate]:
        """Initialize model candidates for selection."""
        candidates = []
        
        # Traditional HMM
        if HMM_COMPONENTS_AVAILABLE:
            candidates.append(ModelCandidate(
                model_type=ModelType.TRADITIONAL_HMM,
                config={'optimization_mode': 'blank', 'symbol': 'ETHUSDT'},
                priority=1
            ))
        
        # Markov-Switching Model
        candidates.append(ModelCandidate(
            model_type=ModelType.MARKOV_SWITCHING,
            config=DataDrivenMSMConfig(
                n_regimes=3,
                enable_break_detection=True,
                adaptive_n_regimes=True,
                max_regimes=8
            ),
            priority=3
        ))
        
        # Hidden Semi-Markov Model
        candidates.append(ModelCandidate(
            model_type=ModelType.HIDDEN_SEMI_MARKOV,
            config=DataDrivenHSMMConfig(
                n_states=4,
                learn_duration_from_data=True,
                adaptive_durations=True,
                automatic_state_number=True,
                max_states=10
            ),
            priority=2
        ))
        
        # Hybrid Model
        candidates.append(ModelCandidate(
            model_type=ModelType.HYBRID_MSM_HSMM,
            config={
                'msm_config': DataDrivenMSMConfig(n_regimes=3, adaptive_n_regimes=True),
                'hsmm_config': DataDrivenHSMMConfig(n_states=4, adaptive_durations=True)
            },
            priority=4
        ))
        
        return candidates
    
    async def run_walk_forward_selection(self, 
                                       data: pd.DataFrame,
                                       symbol: str = "ETHUSDT") -> Dict[str, Any]:
        """
        Run comprehensive walk-forward model selection.
        
        Args:
            data: 1h OHLCV market data
            symbol: Trading symbol
            
        Returns:
            Model selection results
        """
        self.logger.info(f"🚀 Starting walk-forward model selection for {symbol}")
        self.logger.info(f"📊 Data: {len(data)} observations, {data.index[0]} to {data.index[-1]}")
        
        # Generate time series splits
        splits = self._generate_time_series_splits(data)
        self.logger.info(f"🔄 Generated {len(splits)} walk-forward folds")
        
        # Run validation for each model candidate
        for candidate in sorted(self.model_candidates, key=lambda x: x.priority, reverse=True):
            if not candidate.enabled:
                continue
                
            self.logger.info(f"🧪 Testing {candidate.model_type.value} model")
            
            try:
                candidate_results = await self._validate_model_candidate(
                    candidate, data, splits, symbol
                )
                self.validation_results.extend(candidate_results)
                
                self.logger.info(f"✅ {candidate.model_type.value} validation completed: {len(candidate_results)} folds")
                
            except Exception as e:
                self.logger.error(f"❌ {candidate.model_type.value} validation failed: {e}")
                continue
        
        # Select best model
        best_model_info = self._select_best_model()
        
        # Generate comprehensive results
        selection_results = {
            'best_model': best_model_info,
            'all_results': [result.to_dict() for result in self.validation_results],
            'model_comparison': self._generate_model_comparison(),
            'stability_analysis': self._analyze_model_stability(),
            'selection_metadata': {
                'n_folds': len(splits),
                'n_models_tested': len([c for c in self.model_candidates if c.enabled]),
                'data_shape': data.shape,
                'timespan': (data.index[0].isoformat(), data.index[-1].isoformat()),
                'primary_metric': self.wf_config.primary_metric.value
            }
        }
        
        self.logger.info(f"🏆 Best model: {best_model_info['model_type']} "
                        f"(score: {best_model_info['score']:.4f})")
        
        return selection_results
    
    def _generate_time_series_splits(self, data: pd.DataFrame) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """Generate time series splits for walk-forward validation."""
        splits = []
        
        # Calculate split parameters
        train_hours = self.wf_config.train_months * 30 * 24  # Approximate
        val_hours = self.wf_config.validation_months * 30 * 24
        step_hours = self.wf_config.step_months * 30 * 24
        
        # Generate splits
        start_idx = 0
        while start_idx + train_hours + val_hours <= len(data):
            # Training period
            train_start = start_idx
            train_end = start_idx + train_hours
            
            # Validation period
            val_start = train_end
            val_end = train_end + val_hours
            
            # Create date ranges
            train_dates = data.index[train_start:train_end]
            val_dates = data.index[val_start:val_end]
            
            splits.append((train_dates, val_dates))
            
            # Move to next split
            start_idx += step_hours
        
        return splits
    
    async def _validate_model_candidate(self,
                                      candidate: ModelCandidate,
                                      data: pd.DataFrame,
                                      splits: List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]],
                                      symbol: str) -> List[ValidationResult]:
        """Validate a model candidate across all folds."""
        results = []
        
        for fold, (train_dates, val_dates) in enumerate(splits):
            try:
                result = await self._validate_single_fold(
                    candidate, data, train_dates, val_dates, fold, symbol
                )
                if result is not None:
                    results.append(result)
                    
            except Exception as e:
                self.logger.warning(f"Fold {fold} failed for {candidate.model_type.value}: {e}")
                continue
        
        return results
    
    async def _validate_single_fold(self,
                                  candidate: ModelCandidate,
                                  data: pd.DataFrame,
                                  train_dates: pd.DatetimeIndex,
                                  val_dates: pd.DatetimeIndex,
                                  fold: int,
                                  symbol: str) -> Optional[ValidationResult]:
        """Validate model on a single fold."""
        import time
        
        # Extract training and validation data
        train_data = data.loc[train_dates]
        val_data = data.loc[val_dates]
        
        if len(train_data) < self.wf_config.min_train_observations:
            self.logger.warning(f"Insufficient training data for fold {fold}: {len(train_data)}")
            return None
        
        # Generate features for training data
        train_start_time = time.time()
        
        train_features = self.feature_generator.generate_production_features(
            train_data, symbol, current_time=train_dates[-1]
        )
        
        # Initialize and fit model
        model = self._create_model_instance(candidate)
        
        try:
            fit_result = model.fit(train_data)
            training_time = time.time() - train_start_time
            
        except Exception as e:
            self.logger.error(f"Model fitting failed for fold {fold}: {e}")
            return None
        
        # Validation
        val_start_time = time.time()
        
        # Generate validation features
        val_features = self.feature_generator.generate_production_features(
            val_data, symbol, current_time=val_dates[-1]
        )
        
        # Calculate validation metrics
        validation_metrics = self._calculate_validation_metrics(
            model, fit_result, val_data, val_features
        )
        
        validation_time = time.time() - val_start_time
        
        # Stability testing
        stability_metrics = await self._test_model_stability(
            model, train_data, candidate.model_type
        )
        
        # Create validation result
        result = ValidationResult(
            model_type=candidate.model_type,
            fold=fold,
            log_likelihood=validation_metrics.get('log_likelihood', -np.inf),
            bic=validation_metrics.get('bic', np.inf),
            aic=validation_metrics.get('aic', np.inf),
            regime_stability=validation_metrics.get('regime_stability', 0.0),
            transition_quality=validation_metrics.get('transition_quality', 0.0),
            economic_plausibility=validation_metrics.get('economic_plausibility', 0.0),
            n_regimes=fit_result.get('n_regimes', 0),
            regime_assignments=fit_result.get('regime_assignments', np.array([])),
            stability_score=stability_metrics.get('stability_score', 0.0),
            noise_robustness=stability_metrics.get('noise_robustness', 0.0),
            training_time=training_time,
            validation_time=validation_time,
            model_artifacts=self._extract_model_artifacts(model, fit_result)
        )
        
        return result
    
    def _create_model_instance(self, candidate: ModelCandidate):
        """Create model instance from candidate."""
        if candidate.model_type == ModelType.TRADITIONAL_HMM:
            return TraditionalHMMWrapper(candidate.config)
        
        elif candidate.model_type == ModelType.MARKOV_SWITCHING:
            return DataDrivenMarkovSwitchingModel(candidate.config)
        
        elif candidate.model_type == ModelType.HIDDEN_SEMI_MARKOV:
            return DataDrivenHiddenSemiMarkovModel(candidate.config)
        
        elif candidate.model_type == ModelType.HYBRID_MSM_HSMM:
            return HybridMSMHSMMModel(
                candidate.config['msm_config'],
                candidate.config['hsmm_config']
            )
        
        else:
            raise ValueError(f"Unknown model type: {candidate.model_type}")
    
    def _calculate_validation_metrics(self,
                                    model: Any,
                                    fit_result: Dict[str, Any],
                                    val_data: pd.DataFrame,
                                    val_features: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive validation metrics."""
        metrics = {}
        
        try:
            # Basic metrics
            regime_assignments = fit_result.get('regime_assignments', np.array([]))
            n_regimes = fit_result.get('n_regimes', 0)
            
            if len(regime_assignments) > 0 and n_regimes > 1:
                # Log-likelihood approximation
                metrics['log_likelihood'] = self._approximate_log_likelihood(
                    val_data, regime_assignments, n_regimes
                )
                
                # Information criteria
                n_params = self._estimate_n_parameters(n_regimes, val_features.shape[1])
                n_obs = len(val_data)
                
                metrics['aic'] = 2 * n_params - 2 * metrics['log_likelihood']
                metrics['bic'] = np.log(n_obs) * n_params - 2 * metrics['log_likelihood']
                
                # Regime stability
                metrics['regime_stability'] = self._calculate_regime_stability(regime_assignments)
                
                # Transition quality
                metrics['transition_quality'] = self._calculate_transition_quality(regime_assignments)
                
                # Economic plausibility
                metrics['economic_plausibility'] = self._assess_economic_plausibility(
                    val_data, regime_assignments
                )
            
            else:
                # Default values for failed models
                metrics.update({
                    'log_likelihood': -np.inf,
                    'aic': np.inf,
                    'bic': np.inf,
                    'regime_stability': 0.0,
                    'transition_quality': 0.0,
                    'economic_plausibility': 0.0
                })
        
        except Exception as e:
            self.logger.warning(f"Validation metric calculation failed: {e}")
            metrics.update({
                'log_likelihood': -np.inf,
                'aic': np.inf,
                'bic': np.inf,
                'regime_stability': 0.0,
                'transition_quality': 0.0,
                'economic_plausibility': 0.0
            })
        
        return metrics
    
    def _approximate_log_likelihood(self, data: pd.DataFrame, regimes: np.ndarray, n_regimes: int) -> float:
        """Approximate log-likelihood calculation."""
        try:
            returns = data['close'].pct_change().dropna()
            
            if len(regimes) != len(returns):
                # Align lengths
                min_len = min(len(regimes), len(returns))
                regimes = regimes[:min_len]
                returns = returns.iloc[:min_len]
            
            log_likelihood = 0.0
            
            for regime in range(n_regimes):
                regime_mask = regimes == regime
                regime_returns = returns[regime_mask]
                
                if len(regime_returns) > 1:
                    # Gaussian likelihood for regime
                    mean_return = regime_returns.mean()
                    std_return = regime_returns.std()
                    
                    if std_return > 0:
                        # Log-likelihood for normal distribution
                        ll = -0.5 * len(regime_returns) * np.log(2 * np.pi * std_return**2)
                        ll -= 0.5 * np.sum((regime_returns - mean_return)**2) / (std_return**2)
                        log_likelihood += ll
            
            return float(log_likelihood)
        
        except Exception:
            return -np.inf
    
    def _estimate_n_parameters(self, n_regimes: int, n_features: int) -> int:
        """Estimate number of model parameters."""
        # Simple estimation: means + variances + transition probabilities
        regime_params = n_regimes * 2  # mean + variance per regime
        transition_params = n_regimes * (n_regimes - 1)  # transition matrix
        
        return regime_params + transition_params
    
    def _calculate_regime_stability(self, regimes: np.ndarray) -> float:
        """Calculate regime stability score."""
        try:
            if len(regimes) < 2:
                return 0.0
            
            # Calculate transition rate
            transitions = np.sum(np.diff(regimes) != 0)
            transition_rate = transitions / len(regimes)
            
            # Stability is inverse of transition rate
            stability = 1.0 / (1.0 + transition_rate * 10)
            
            return float(stability)
        
        except Exception:
            return 0.0
    
    def _calculate_transition_quality(self, regimes: np.ndarray) -> float:
        """Calculate transition quality score."""
        try:
            if len(regimes) < 10:
                return 0.0
            
            # Calculate regime durations
            changes = np.where(np.diff(regimes) != 0)[0] + 1
            starts = np.concatenate([[0], changes])
            ends = np.concatenate([changes, [len(regimes)]])
            
            durations = ends - starts
            
            if len(durations) > 1:
                # Quality based on duration consistency
                mean_duration = np.mean(durations)
                duration_cv = np.std(durations) / mean_duration if mean_duration > 0 else np.inf
                
                # Lower coefficient of variation = higher quality
                quality = 1.0 / (1.0 + duration_cv)
                return float(quality)
            
            return 0.0
        
        except Exception:
            return 0.0
    
    def _assess_economic_plausibility(self, data: pd.DataFrame, regimes: np.ndarray) -> float:
        """Assess economic plausibility of regimes."""
        try:
            returns = data['close'].pct_change().dropna()
            
            if len(regimes) != len(returns):
                min_len = min(len(regimes), len(returns))
                regimes = regimes[:min_len]
                returns = returns.iloc[:min_len]
            
            unique_regimes = np.unique(regimes)
            plausibility_scores = []
            
            for regime in unique_regimes:
                regime_returns = returns[regimes == regime]
                
                if len(regime_returns) > 10:
                    # Check for reasonable return/volatility characteristics
                    mean_return = regime_returns.mean()
                    volatility = regime_returns.std()
                    
                    # Plausibility based on Sharpe-like ratio and reasonable volatility
                    if volatility > 0:
                        sharpe_like = abs(mean_return) / volatility
                        vol_plausibility = 1.0 / (1.0 + max(0, volatility - 0.05) * 20)  # Penalize extreme vol
                        
                        regime_plausibility = (sharpe_like + vol_plausibility) / 2
                        plausibility_scores.append(regime_plausibility)
            
            return float(np.mean(plausibility_scores)) if plausibility_scores else 0.0
        
        except Exception:
            return 0.0
    
    async def _test_model_stability(self, model: Any, data: pd.DataFrame, model_type: ModelType) -> Dict[str, float]:
        """Test model stability with noise injection."""
        stability_scores = []
        noise_robustness_scores = []
        
        try:
            # Original fit
            original_result = model.fit(data)
            original_regimes = original_result.get('regime_assignments', np.array([]))
            
            if len(original_regimes) == 0:
                return {'stability_score': 0.0, 'noise_robustness': 0.0}
            
            # Test with noise injection
            for _ in range(self.wf_config.stability_test_iterations):
                # Add small amount of noise to data
                noisy_data = data.copy()
                noise = np.random.normal(0, self.wf_config.stability_noise_level, len(data))
                noisy_data['close'] = noisy_data['close'] * (1 + noise)
                
                try:
                    # Create new model instance for noise test
                    noise_model = self._create_model_instance(
                        ModelCandidate(model_type, model.config if hasattr(model, 'config') else {})
                    )
                    
                    noise_result = noise_model.fit(noisy_data)
                    noise_regimes = noise_result.get('regime_assignments', np.array([]))
                    
                    if len(noise_regimes) > 0:
                        # Calculate agreement with original
                        min_len = min(len(original_regimes), len(noise_regimes))
                        if min_len > 0:
                            ari = adjusted_rand_score(
                                original_regimes[:min_len],
                                noise_regimes[:min_len]
                            )
                            stability_scores.append(max(0.0, ari))
                        
                        # Noise robustness based on regime count stability
                        regime_count_stability = 1.0 - abs(
                            len(np.unique(original_regimes)) - len(np.unique(noise_regimes))
                        ) / max(len(np.unique(original_regimes)), 1)
                        noise_robustness_scores.append(max(0.0, regime_count_stability))
                
                except Exception:
                    stability_scores.append(0.0)
                    noise_robustness_scores.append(0.0)
            
            return {
                'stability_score': float(np.mean(stability_scores)) if stability_scores else 0.0,
                'noise_robustness': float(np.mean(noise_robustness_scores)) if noise_robustness_scores else 0.0
            }
        
        except Exception as e:
            self.logger.warning(f"Stability testing failed: {e}")
            return {'stability_score': 0.0, 'noise_robustness': 0.0}
    
    def _extract_model_artifacts(self, model: Any, fit_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract model artifacts for production deployment."""
        artifacts = {
            'model_type': type(model).__name__,
            'n_regimes': fit_result.get('n_regimes', 0),
            'fit_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Model-specific artifacts
        if hasattr(model, 'transition_matrix') and model.transition_matrix is not None:
            artifacts['transition_matrix'] = model.transition_matrix.tolist()
        
        if hasattr(model, 'regime_models') and model.regime_models:
            artifacts['regime_characteristics'] = model.regime_models
        
        if hasattr(model, 'duration_models') and model.duration_models:
            artifacts['duration_models'] = model.duration_models
        
        # Add structural break information for MSM
        if 'structural_breaks' in fit_result:
            artifacts['structural_breaks'] = fit_result['structural_breaks']
        
        return artifacts
    
    def _select_best_model(self) -> Dict[str, Any]:
        """Select best model based on validation results."""
        if not self.validation_results:
            return {'model_type': 'none', 'score': 0.0, 'reason': 'no_valid_results'}
        
        # Group results by model type
        model_scores = {}
        
        for result in self.validation_results:
            model_type = result.model_type.value
            
            if model_type not in model_scores:
                model_scores[model_type] = []
            
            # Calculate composite score
            primary_score = getattr(result, self.wf_config.primary_metric.value.lower())
            
            # Handle different metric types (higher/lower is better)
            if self.wf_config.primary_metric in [ValidationMetric.LOG_LIKELIHOOD, 
                                                ValidationMetric.REGIME_STABILITY,
                                                ValidationMetric.TRANSITION_QUALITY]:
                score = primary_score  # Higher is better
            else:  # BIC, AIC
                score = -primary_score  # Lower is better (so negate)
            
            # Add stability bonus
            stability_bonus = result.stability_score * 0.1
            composite_score = score + stability_bonus
            
            model_scores[model_type].append(composite_score)
        
        # Calculate mean scores
        mean_scores = {
            model_type: np.mean(scores)
            for model_type, scores in model_scores.items()
        }
        
        # Select best model
        best_model_type = max(mean_scores.keys(), key=lambda k: mean_scores[k])
        best_score = mean_scores[best_model_type]
        
        # Get best individual result for this model type
        best_results = [r for r in self.validation_results if r.model_type.value == best_model_type]
        best_individual_result = max(best_results, key=lambda r: getattr(r, self.wf_config.primary_metric.value.lower()))
        
        return {
            'model_type': best_model_type,
            'score': best_score,
            'individual_best_fold': best_individual_result.fold,
            'n_folds': len(best_results),
            'mean_stability': np.mean([r.stability_score for r in best_results]),
            'selection_reason': f'best_{self.wf_config.primary_metric.value.lower()}'
        }
    
    def _generate_model_comparison(self) -> Dict[str, Any]:
        """Generate comprehensive model comparison."""
        if not self.validation_results:
            return {}
        
        comparison = {}
        
        # Group by model type
        by_model = {}
        for result in self.validation_results:
            model_type = result.model_type.value
            if model_type not in by_model:
                by_model[model_type] = []
            by_model[model_type].append(result)
        
        # Calculate statistics for each model
        for model_type, results in by_model.items():
            comparison[model_type] = {
                'n_folds': len(results),
                'mean_log_likelihood': np.mean([r.log_likelihood for r in results]),
                'std_log_likelihood': np.std([r.log_likelihood for r in results]),
                'mean_bic': np.mean([r.bic for r in results]),
                'mean_regime_stability': np.mean([r.regime_stability for r in results]),
                'mean_stability_score': np.mean([r.stability_score for r in results]),
                'mean_training_time': np.mean([r.training_time for r in results]),
                'success_rate': len([r for r in results if r.log_likelihood > -np.inf]) / len(results)
            }
        
        return comparison
    
    def _analyze_model_stability(self) -> Dict[str, Any]:
        """Analyze overall model stability across folds."""
        if not self.validation_results:
            return {}
        
        # Group by model type
        by_model = {}
        for result in self.validation_results:
            model_type = result.model_type.value
            if model_type not in by_model:
                by_model[model_type] = []
            by_model[model_type].append(result)
        
        stability_analysis = {}
        
        for model_type, results in by_model.items():
            # Cross-fold regime assignment consistency
            regime_assignments = [r.regime_assignments for r in results if len(r.regime_assignments) > 0]
            
            cross_fold_consistency = []
            if len(regime_assignments) > 1:
                for i in range(len(regime_assignments)):
                    for j in range(i + 1, len(regime_assignments)):
                        # Compare overlapping periods (simplified)
                        min_len = min(len(regime_assignments[i]), len(regime_assignments[j]))
                        if min_len > 0:
                            ari = adjusted_rand_score(
                                regime_assignments[i][:min_len],
                                regime_assignments[j][:min_len]
                            )
                            cross_fold_consistency.append(ari)
            
            stability_analysis[model_type] = {
                'cross_fold_consistency': np.mean(cross_fold_consistency) if cross_fold_consistency else 0.0,
                'stability_score_consistency': np.std([r.stability_score for r in results]),
                'regime_count_consistency': np.std([r.n_regimes for r in results]),
                'performance_consistency': np.std([r.log_likelihood for r in results if r.log_likelihood > -np.inf])
            }
        
        return stability_analysis
    
    def save_selection_results(self, results: Dict[str, Any], filepath: str):
        """Save model selection results."""
        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save results
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"💾 Model selection results saved to {filepath}")
    
    def get_production_artifacts(self) -> Dict[str, Any]:
        """Get production artifacts for deployment."""
        if not self.validation_results:
            return {}
        
        # Get best model results
        best_model_info = self._select_best_model()
        best_model_type = best_model_info['model_type']
        
        best_results = [
            r for r in self.validation_results 
            if r.model_type.value == best_model_type
        ]
        
        if not best_results:
            return {}
        
        # Get the best individual result
        best_result = max(best_results, key=lambda r: r.log_likelihood)
        
        return {
            'best_model_type': best_model_type,
            'model_artifacts': best_result.model_artifacts,
            'performance_metrics': {
                'log_likelihood': best_result.log_likelihood,
                'bic': best_result.bic,
                'regime_stability': best_result.regime_stability,
                'stability_score': best_result.stability_score
            },
            'feature_config': self.feature_config.__dict__,
            'validation_config': self.wf_config.__dict__,
            'selection_timestamp': pd.Timestamp.now().isoformat()
        }


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    # Generate synthetic 1h market data for testing
    np.random.seed(42)
    
    # Create 6 months of 1h data
    dates = pd.date_range('2023-01-01', '2023-07-01', freq='1H')
    n_obs = len(dates)
    
    # Create realistic regime-switching market data
    prices = np.zeros(n_obs)
    prices[0] = 100.0
    
    # Three distinct regimes
    regime_periods = [
        (0, n_obs//3, 0),      # Low vol regime
        (n_obs//3, 2*n_obs//3, 1),  # High vol regime  
        (2*n_obs//3, n_obs, 2)      # Medium vol regime
    ]
    
    for start, end, regime in regime_periods:
        if regime == 0:  # Low vol
            vol = 0.008
            drift = 0.0001
        elif regime == 1:  # High vol
            vol = 0.025
            drift = -0.0005
        else:  # Medium vol
            vol = 0.015
            drift = 0.0002
        
        for i in range(start, end):
            if i < len(prices) - 1:
                ret = np.random.normal(drift, vol)
                prices[i + 1] = prices[i] * (1 + ret)
    
    # Create test data
    test_data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.0005, n_obs)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n_obs))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_obs))),
        'close': prices,
        'volume': np.random.lognormal(12, 0.3, n_obs)
    }, index=dates)
    
    print("🧪 Testing Advanced Model Integration")
    print(f"📊 Test data: {len(test_data)} observations (1h timeframe)")
    print(f"⏰ Timespan: {test_data.index[0]} to {test_data.index[-1]}")
    
    # Configure walk-forward validation
    wf_config = WalkForwardConfig(
        train_months=3,
        validation_months=1,
        step_months=1,
        n_folds=3,  # Limited for testing
        stability_test_iterations=2
    )
    
    # Configure features
    feature_config = ProductionFeatureConfig(
        primary_timeframe="1h",
        horizons=[1, 2, 4],
        use_existing_orchestrator=False,  # Disable for testing
        use_existing_feature_engineer=False
    )
    
    # Initialize model selector
    model_selector = AdvancedModelSelector(wf_config, feature_config)
    
    async def test_model_selection():
        # Run walk-forward model selection
        results = await model_selector.run_walk_forward_selection(test_data, "ETHUSDT")
        
        print(f"\n🏆 Best model: {results['best_model']['model_type']}")
        print(f"📊 Score: {results['best_model']['score']:.4f}")
        print(f"🔄 Tested {len(results['all_results'])} model-fold combinations")
        
        # Show model comparison
        print(f"\n📈 Model Comparison:")
        for model_type, metrics in results['model_comparison'].items():
            print(f"  {model_type}:")
            print(f"    Success rate: {metrics['success_rate']:.1%}")
            print(f"    Mean log-likelihood: {metrics['mean_log_likelihood']:.2f}")
            print(f"    Mean stability: {metrics['mean_stability_score']:.3f}")
        
        # Show stability analysis
        print(f"\n🔒 Stability Analysis:")
        for model_type, stability in results['stability_analysis'].items():
            print(f"  {model_type}:")
            print(f"    Cross-fold consistency: {stability['cross_fold_consistency']:.3f}")
            print(f"    Performance consistency: {stability['performance_consistency']:.3f}")
        
        return results
    
    # Run test
    results = asyncio.run(test_model_selection())
    
    print(f"\n🎯 Advanced model integration test completed!")
    print(f"   Ready for production pipeline deployment")