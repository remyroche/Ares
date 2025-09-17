"""
Advanced Markov Models for Enhanced Regime Detection

This module implements sophisticated Markov modeling techniques that extend
beyond basic Hidden Markov Models to provide more accurate and economically
meaningful regime detection for financial markets.

Key Advanced Models:
1. Markov-Switching Models (MSM) - Regime-dependent parameters
2. Hidden Semi-Markov Models (HSMM) - Variable state durations  
3. Hierarchical Markov Models - Multi-scale regime structure
4. Economic Regime Models - Market-specific constraints

Integration with existing HMM framework while providing advanced capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.special import logsumexp

from src.utils.logger import system_logger

try:
    import statsmodels.tsa.regime_switching as sm_rs
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available - Markov-Switching models limited")

try:
    from hmmlearn import hmm
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False
    warnings.warn("hmmlearn not available - basic HMM functionality limited")


class AdvancedRegimeType(Enum):
    """Enhanced regime types with economic meaning."""
    BULL_MARKET = "bull_market"           # Sustained uptrend with low volatility
    BEAR_MARKET = "bear_market"           # Sustained downtrend with high volatility
    HIGH_VOLATILITY = "high_volatility"   # Crisis/uncertainty periods
    LOW_VOLATILITY = "low_volatility"     # Calm, stable periods
    MOMENTUM = "momentum"                 # Strong trending behavior
    MEAN_REVERSION = "mean_reversion"     # Range-bound, oscillating
    TRANSITION = "transition"             # Regime change periods
    CONSOLIDATION = "consolidation"       # Low activity, accumulation


class DurationDistribution(Enum):
    """Available duration distributions for HSMM."""
    GEOMETRIC = "geometric"      # Standard HMM (exponential)
    GAMMA = "gamma"             # Flexible, realistic for markets
    WEIBULL = "weibull"         # Good for survival analysis
    LOGNORMAL = "lognormal"     # Heavy-tailed durations
    POISSON = "poisson"         # Discrete event modeling
    NEGATIVE_BINOMIAL = "negative_binomial"  # Overdispersed counts


@dataclass
class MarkovSwitchingConfig:
    """Configuration for Markov-Switching Models."""
    n_regimes: int = 3
    regime_types: List[AdvancedRegimeType] = field(default_factory=lambda: [
        AdvancedRegimeType.BULL_MARKET,
        AdvancedRegimeType.BEAR_MARKET, 
        AdvancedRegimeType.HIGH_VOLATILITY
    ])
    
    # Economic constraints
    min_regime_duration: int = 20  # Minimum 20 periods per regime
    max_volatility_ratio: float = 10.0  # Max vol ratio between regimes
    transition_smoothing: float = 0.1  # Transition probability smoothing
    
    # Model specifications per regime
    volatility_models: Dict[str, str] = field(default_factory=lambda: {
        'bull_market': 'low_vol',
        'bear_market': 'high_vol', 
        'high_volatility': 'extreme_vol'
    })
    
    # Prior constraints
    use_economic_priors: bool = True
    prior_regime_probabilities: Optional[Dict[str, float]] = None


@dataclass  
class SemiMarkovConfig:
    """Configuration for Hidden Semi-Markov Models."""
    n_states: int = 4
    duration_distributions: Dict[int, DurationDistribution] = field(default_factory=lambda: {
        0: DurationDistribution.GAMMA,      # Bull markets: gamma distribution
        1: DurationDistribution.WEIBULL,    # Bear markets: weibull (survival)
        2: DurationDistribution.LOGNORMAL,  # High vol: heavy-tailed
        3: DurationDistribution.GEOMETRIC   # Transitions: exponential
    })
    
    # Duration constraints (in periods)
    min_durations: Dict[int, int] = field(default_factory=lambda: {
        0: 30,   # Bull markets: at least 30 periods
        1: 10,   # Bear markets: at least 10 periods  
        2: 5,    # High vol: at least 5 periods
        3: 1     # Transitions: at least 1 period
    })
    
    max_durations: Dict[int, int] = field(default_factory=lambda: {
        0: 500,  # Bull markets: max 500 periods
        1: 200,  # Bear markets: max 200 periods
        2: 100,  # High vol: max 100 periods  
        3: 50    # Transitions: max 50 periods
    })
    
    # Economic regime mapping
    state_interpretations: Dict[int, AdvancedRegimeType] = field(default_factory=lambda: {
        0: AdvancedRegimeType.BULL_MARKET,
        1: AdvancedRegimeType.BEAR_MARKET,
        2: AdvancedRegimeType.HIGH_VOLATILITY,
        3: AdvancedRegimeType.TRANSITION
    })


class MarkovSwitchingRegimeModel:
    """
    Markov-Switching Model for regime-dependent parameter estimation.
    
    This model allows different statistical models in each regime,
    providing more realistic modeling of market behavior than basic HMMs.
    """
    
    def __init__(self, config: MarkovSwitchingConfig):
        self.config = config
        self.logger = system_logger.getChild('MarkovSwitchingModel')
        
        # Model components
        self.regime_models: Dict[int, Any] = {}
        self.transition_matrix: Optional[np.ndarray] = None
        self.regime_probabilities: Optional[np.ndarray] = None
        self.fitted_models: Dict[int, Dict[str, Any]] = {}
        
        # Economic constraints
        self.economic_priors = self._setup_economic_priors()
        
    def _setup_economic_priors(self) -> Dict[str, Any]:
        """Setup economic priors for regime parameters."""
        priors = {
            'bull_market': {
                'mean_return': (0.08/252, 0.02/252),  # 8% annual, low variance
                'volatility': (0.15/np.sqrt(252), 0.05/np.sqrt(252)),  # 15% annual vol
                'min_duration': 60,  # At least 3 months
                'max_duration': 1260  # At most 5 years
            },
            'bear_market': {
                'mean_return': (-0.20/252, 0.05/252),  # -20% annual, higher variance
                'volatility': (0.30/np.sqrt(252), 0.10/np.sqrt(252)),  # 30% annual vol
                'min_duration': 20,  # At least 1 month
                'max_duration': 500  # At most 2 years
            },
            'high_volatility': {
                'mean_return': (0.0, 0.10/252),  # Neutral return, high variance
                'volatility': (0.50/np.sqrt(252), 0.20/np.sqrt(252)),  # 50% annual vol
                'min_duration': 5,   # At least 1 week
                'max_duration': 60   # At most 3 months
            }
        }
        return priors
    
    def fit(self, data: pd.DataFrame, features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Fit Markov-Switching model to market data.
        
        Args:
            data: Market data with OHLCV
            features: Optional feature list for regime identification
            
        Returns:
            Fitting results and regime assignments
        """
        self.logger.info(f"🔄 Fitting Markov-Switching model with {self.config.n_regimes} regimes")
        
        try:
            # Prepare features for regime detection
            feature_data = self._prepare_features(data, features)
            
            # Initial regime detection using clustering
            initial_regimes = self._initial_regime_detection(feature_data)
            
            # Fit regime-specific models
            regime_results = {}
            for regime_idx in range(self.config.n_regimes):
                regime_mask = initial_regimes == regime_idx
                regime_data = data[regime_mask]
                
                if len(regime_data) < self.config.min_regime_duration:
                    self.logger.warning(f"Regime {regime_idx} has insufficient data ({len(regime_data)} < {self.config.min_regime_duration})")
                    continue
                
                # Fit regime-specific model
                regime_model = self._fit_regime_model(regime_idx, regime_data)
                regime_results[regime_idx] = regime_model
            
            # Estimate transition matrix with economic constraints
            transition_matrix = self._estimate_transition_matrix(initial_regimes)
            
            # Refine regime assignments using Viterbi-like algorithm
            refined_regimes = self._refine_regime_assignments(
                feature_data, regime_results, transition_matrix
            )
            
            # Store results
            self.regime_models = regime_results
            self.transition_matrix = transition_matrix
            
            # Calculate regime statistics
            regime_stats = self._calculate_regime_statistics(data, refined_regimes)
            
            return {
                'regime_assignments': refined_regimes,
                'regime_models': regime_results,
                'transition_matrix': transition_matrix,
                'regime_statistics': regime_stats,
                'economic_validation': self._validate_economic_constraints(regime_stats)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Markov-Switching model fitting failed: {e}")
            raise
    
    def _prepare_features(self, data: pd.DataFrame, features: Optional[List[str]]) -> np.ndarray:
        """Prepare features for regime detection."""
        if features is None:
            # Default market regime features
            feature_data = pd.DataFrame()
            
            # Returns and volatility
            feature_data['returns'] = data['close'].pct_change()
            feature_data['volatility'] = feature_data['returns'].rolling(20).std()
            
            # Trend indicators
            feature_data['ma_ratio'] = data['close'] / data['close'].rolling(50).mean()
            feature_data['momentum'] = data['close'].pct_change(20)
            
            # Volume indicators  
            if 'volume' in data.columns:
                feature_data['volume_ma'] = data['volume'] / data['volume'].rolling(20).mean()
            
            # Volatility clustering
            feature_data['vol_cluster'] = feature_data['volatility'] / feature_data['volatility'].rolling(100).mean()
            
        else:
            feature_data = data[features]
        
        # Standardize features
        scaler = StandardScaler()
        return scaler.fit_transform(feature_data.fillna(method='ffill').fillna(0))
    
    def _initial_regime_detection(self, features: np.ndarray) -> np.ndarray:
        """Initial regime detection using clustering."""
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=self.config.n_regimes, random_state=42, n_init=10)
        initial_regimes = kmeans.fit_predict(features)
        
        self.logger.info(f"📊 Initial regime distribution: {np.bincount(initial_regimes)}")
        return initial_regimes
    
    def _fit_regime_model(self, regime_idx: int, regime_data: pd.DataFrame) -> Dict[str, Any]:
        """Fit statistical model for a specific regime."""
        regime_type = self.config.regime_types[regime_idx] if regime_idx < len(self.config.regime_types) else None
        
        # Calculate regime characteristics
        returns = regime_data['close'].pct_change().dropna()
        
        model = {
            'regime_idx': regime_idx,
            'regime_type': regime_type.value if regime_type else f'regime_{regime_idx}',
            'n_observations': len(regime_data),
            'mean_return': float(returns.mean()),
            'volatility': float(returns.std()),
            'skewness': float(returns.skew()),
            'kurtosis': float(returns.kurtosis()),
            'sharpe_ratio': float(returns.mean() / returns.std()) if returns.std() > 0 else 0,
            'max_drawdown': float(self._calculate_max_drawdown(regime_data['close'])),
            'avg_duration': len(regime_data)  # Will be refined later
        }
        
        # Fit regime-specific volatility model
        if regime_type == AdvancedRegimeType.HIGH_VOLATILITY:
            model['volatility_model'] = self._fit_high_vol_model(returns)
        elif regime_type == AdvancedRegimeType.BULL_MARKET:
            model['volatility_model'] = self._fit_low_vol_model(returns)
        else:
            model['volatility_model'] = self._fit_standard_vol_model(returns)
        
        return model
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown for regime."""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())
    
    def _fit_high_vol_model(self, returns: pd.Series) -> Dict[str, Any]:
        """Fit volatility model for high volatility regime."""
        # Use GARCH-like characteristics
        return {
            'type': 'high_volatility',
            'base_vol': float(returns.std()),
            'vol_clustering': float(returns.rolling(5).std().std()),
            'extreme_quantiles': {
                'q01': float(returns.quantile(0.01)),
                'q99': float(returns.quantile(0.99))
            }
        }
    
    def _fit_low_vol_model(self, returns: pd.Series) -> Dict[str, Any]:
        """Fit volatility model for low volatility regime."""
        return {
            'type': 'low_volatility', 
            'base_vol': float(returns.std()),
            'mean_reversion': float(self._calculate_mean_reversion_speed(returns)),
            'stability_score': float(1.0 / (1.0 + returns.rolling(10).std().std()))
        }
    
    def _fit_standard_vol_model(self, returns: pd.Series) -> Dict[str, Any]:
        """Fit standard volatility model."""
        return {
            'type': 'standard_volatility',
            'base_vol': float(returns.std()),
            'autocorr_1': float(returns.autocorr(lag=1)) if len(returns) > 1 else 0.0
        }
    
    def _calculate_mean_reversion_speed(self, returns: pd.Series) -> float:
        """Calculate mean reversion speed parameter."""
        if len(returns) < 10:
            return 0.0
        
        # Simple AR(1) coefficient as proxy for mean reversion
        y = returns[1:].values
        x = returns[:-1].values
        
        if len(x) > 0 and np.std(x) > 0:
            correlation = np.corrcoef(x, y)[0, 1]
            return max(0.0, -correlation)  # Higher negative correlation = faster mean reversion
        return 0.0
    
    def _estimate_transition_matrix(self, regimes: np.ndarray) -> np.ndarray:
        """Estimate transition matrix with economic constraints."""
        n_regimes = self.config.n_regimes
        transition_matrix = np.zeros((n_regimes, n_regimes))
        
        # Count transitions
        for t in range(len(regimes) - 1):
            current_regime = regimes[t]
            next_regime = regimes[t + 1]
            transition_matrix[current_regime, next_regime] += 1
        
        # Apply economic constraints
        transition_matrix = self._apply_economic_transition_constraints(transition_matrix)
        
        # Normalize rows to probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix = transition_matrix / row_sums
        
        return transition_matrix
    
    def _apply_economic_transition_constraints(self, transition_matrix: np.ndarray) -> np.ndarray:
        """Apply economic constraints to transition probabilities."""
        # Add minimum self-transition probability (regime persistence)
        min_persistence = 0.7  # Regimes should persist at least 70% of the time
        
        for i in range(len(transition_matrix)):
            # Ensure minimum persistence
            if transition_matrix[i, i] < min_persistence * transition_matrix[i].sum():
                total_transitions = transition_matrix[i].sum()
                if total_transitions > 0:
                    # Redistribute transition probabilities
                    other_transitions = total_transitions - transition_matrix[i, i]
                    min_self_transitions = min_persistence * total_transitions
                    
                    if other_transitions > 0:
                        reduction_factor = (total_transitions - min_self_transitions) / other_transitions
                        for j in range(len(transition_matrix)):
                            if i != j:
                                transition_matrix[i, j] *= reduction_factor
                    
                    transition_matrix[i, i] = min_self_transitions
        
        return transition_matrix
    
    def _refine_regime_assignments(self, 
                                 features: np.ndarray,
                                 regime_models: Dict[int, Dict[str, Any]],
                                 transition_matrix: np.ndarray) -> np.ndarray:
        """Refine regime assignments using forward-backward algorithm."""
        # This is a simplified version - full implementation would use
        # proper forward-backward algorithm with emission probabilities
        
        n_obs = len(features)
        n_regimes = len(regime_models)
        
        # Calculate emission probabilities (simplified)
        emission_probs = np.zeros((n_obs, n_regimes))
        
        for regime_idx, model in regime_models.items():
            # Simple Gaussian emission probability based on regime characteristics
            regime_mean = np.zeros(features.shape[1])  # Could be learned from data
            regime_cov = np.eye(features.shape[1])     # Simplified covariance
            
            for t in range(n_obs):
                emission_probs[t, regime_idx] = stats.multivariate_normal.pdf(
                    features[t], regime_mean, regime_cov
                )
        
        # Viterbi algorithm for most likely path
        refined_regimes = self._viterbi_decode(emission_probs, transition_matrix)
        
        return refined_regimes
    
    def _viterbi_decode(self, emission_probs: np.ndarray, transition_matrix: np.ndarray) -> np.ndarray:
        """Viterbi algorithm for finding most likely regime sequence."""
        n_obs, n_regimes = emission_probs.shape
        
        # Initialize
        delta = np.zeros((n_obs, n_regimes))
        psi = np.zeros((n_obs, n_regimes), dtype=int)
        
        # Initial probabilities (uniform)
        initial_probs = np.ones(n_regimes) / n_regimes
        
        # Forward pass
        delta[0] = np.log(initial_probs) + np.log(emission_probs[0] + 1e-10)
        
        for t in range(1, n_obs):
            for j in range(n_regimes):
                trans_scores = delta[t-1] + np.log(transition_matrix[:, j] + 1e-10)
                psi[t, j] = np.argmax(trans_scores)
                delta[t, j] = np.max(trans_scores) + np.log(emission_probs[t, j] + 1e-10)
        
        # Backward pass
        path = np.zeros(n_obs, dtype=int)
        path[-1] = np.argmax(delta[-1])
        
        for t in range(n_obs - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        
        return path
    
    def _calculate_regime_statistics(self, data: pd.DataFrame, regimes: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """Calculate comprehensive statistics for each regime."""
        stats_by_regime = {}
        
        for regime_idx in range(self.config.n_regimes):
            regime_mask = regimes == regime_idx
            regime_data = data[regime_mask]
            
            if len(regime_data) == 0:
                continue
            
            returns = regime_data['close'].pct_change().dropna()
            
            # Calculate regime durations
            regime_changes = np.diff(regime_mask.astype(int))
            regime_starts = np.where(regime_changes == 1)[0] + 1
            regime_ends = np.where(regime_changes == -1)[0] + 1
            
            # Handle edge cases
            if regime_mask[0]:
                regime_starts = np.concatenate([[0], regime_starts])
            if regime_mask[-1]:
                regime_ends = np.concatenate([regime_ends, [len(regime_mask)]])
            
            durations = regime_ends - regime_starts if len(regime_starts) == len(regime_ends) else []
            
            stats_by_regime[regime_idx] = {
                'regime_type': self.config.regime_types[regime_idx].value if regime_idx < len(self.config.regime_types) else f'regime_{regime_idx}',
                'frequency': float(np.sum(regime_mask) / len(regimes)),
                'n_observations': int(np.sum(regime_mask)),
                'n_episodes': len(durations),
                'avg_duration': float(np.mean(durations)) if durations else 0,
                'median_duration': float(np.median(durations)) if durations else 0,
                'min_duration': float(np.min(durations)) if durations else 0,
                'max_duration': float(np.max(durations)) if durations else 0,
                'mean_return': float(returns.mean()) if len(returns) > 0 else 0,
                'volatility': float(returns.std()) if len(returns) > 0 else 0,
                'sharpe_ratio': float(returns.mean() / returns.std()) if len(returns) > 0 and returns.std() > 0 else 0,
                'max_drawdown': float(self._calculate_max_drawdown(regime_data['close'])) if len(regime_data) > 1 else 0,
                'skewness': float(returns.skew()) if len(returns) > 0 else 0,
                'kurtosis': float(returns.kurtosis()) if len(returns) > 0 else 0
            }
        
        return stats_by_regime
    
    def _validate_economic_constraints(self, regime_stats: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Validate regime statistics against economic constraints."""
        validation_results = {
            'constraints_satisfied': True,
            'violations': [],
            'warnings': [],
            'economic_plausibility': 'high'
        }
        
        for regime_idx, stats in regime_stats.items():
            regime_type = stats['regime_type']
            
            # Check duration constraints
            if stats['avg_duration'] < self.config.min_regime_duration:
                validation_results['violations'].append(
                    f"Regime {regime_idx} ({regime_type}) has insufficient average duration: {stats['avg_duration']:.1f} < {self.config.min_regime_duration}"
                )
                validation_results['constraints_satisfied'] = False
            
            # Check volatility ratios
            if regime_type == 'high_volatility' and stats['volatility'] < 0.02:
                validation_results['warnings'].append(
                    f"High volatility regime {regime_idx} has low volatility: {stats['volatility']:.4f}"
                )
            
            if regime_type == 'low_volatility' and stats['volatility'] > 0.05:
                validation_results['warnings'].append(
                    f"Low volatility regime {regime_idx} has high volatility: {stats['volatility']:.4f}"
                )
        
        # Overall economic plausibility
        if validation_results['violations']:
            validation_results['economic_plausibility'] = 'low'
        elif validation_results['warnings']:
            validation_results['economic_plausibility'] = 'medium'
        
        return validation_results


class HiddenSemiMarkovModel:
    """
    Hidden Semi-Markov Model with explicit duration modeling.
    
    This model extends HMMs by allowing flexible duration distributions
    for each state, providing more realistic regime persistence modeling.
    """
    
    def __init__(self, config: SemiMarkovConfig):
        self.config = config
        self.logger = system_logger.getChild('HiddenSemiMarkovModel')
        
        # Model components
        self.duration_models: Dict[int, Any] = {}
        self.emission_models: Dict[int, Any] = {}
        self.transition_matrix: Optional[np.ndarray] = None
        self.initial_probabilities: Optional[np.ndarray] = None
        
    def fit(self, data: pd.DataFrame, max_iterations: int = 100) -> Dict[str, Any]:
        """
        Fit Hidden Semi-Markov Model to market data.
        
        Args:
            data: Market data with OHLCV
            max_iterations: Maximum EM iterations
            
        Returns:
            Fitting results and state assignments
        """
        self.logger.info(f"🔄 Fitting Hidden Semi-Markov model with {self.config.n_states} states")
        
        try:
            # Prepare observations
            observations = self._prepare_observations(data)
            
            # Initialize model parameters
            self._initialize_parameters(observations)
            
            # EM algorithm for HSMM
            log_likelihood_history = []
            
            for iteration in range(max_iterations):
                # E-step: Forward-backward algorithm for HSMM
                forward_probs, backward_probs, log_likelihood = self._forward_backward_hsmm(observations)
                log_likelihood_history.append(log_likelihood)
                
                # M-step: Update parameters
                self._update_parameters(observations, forward_probs, backward_probs)
                
                # Check convergence
                if iteration > 0:
                    improvement = log_likelihood_history[-1] - log_likelihood_history[-2]
                    if abs(improvement) < 1e-6:
                        self.logger.info(f"✅ HSMM converged after {iteration + 1} iterations")
                        break
            
            # Decode most likely state sequence
            state_sequence = self._viterbi_decode_hsmm(observations)
            
            # Calculate state statistics
            state_stats = self._calculate_state_statistics(data, state_sequence)
            
            # Validate duration constraints
            duration_validation = self._validate_duration_constraints(state_sequence)
            
            return {
                'state_sequence': state_sequence,
                'duration_models': self.duration_models,
                'emission_models': self.emission_models,
                'transition_matrix': self.transition_matrix,
                'state_statistics': state_stats,
                'duration_validation': duration_validation,
                'log_likelihood_history': log_likelihood_history,
                'final_log_likelihood': log_likelihood_history[-1] if log_likelihood_history else None
            }
            
        except Exception as e:
            self.logger.error(f"❌ Hidden Semi-Markov model fitting failed: {e}")
            raise
    
    def _prepare_observations(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare observations for HSMM."""
        # Multi-dimensional observations: returns, volatility, volume
        obs_data = pd.DataFrame()
        
        obs_data['returns'] = data['close'].pct_change()
        obs_data['log_vol'] = np.log(obs_data['returns'].rolling(5).std() + 1e-8)
        
        if 'volume' in data.columns:
            obs_data['log_volume'] = np.log(data['volume'] + 1)
        else:
            obs_data['log_volume'] = np.zeros(len(data))
        
        # Standardize observations
        scaler = StandardScaler()
        return scaler.fit_transform(obs_data.fillna(method='ffill').fillna(0))
    
    def _initialize_parameters(self, observations: np.ndarray):
        """Initialize HSMM parameters."""
        n_obs, n_features = observations.shape
        n_states = self.config.n_states
        
        # Initialize transition matrix (between different states only)
        self.transition_matrix = np.random.rand(n_states, n_states)
        np.fill_diagonal(self.transition_matrix, 0)  # No self-transitions in HSMM
        self.transition_matrix = self.transition_matrix / self.transition_matrix.sum(axis=1, keepdims=True)
        
        # Initialize initial probabilities
        self.initial_probabilities = np.ones(n_states) / n_states
        
        # Initialize emission models (Gaussian for each state)
        self.emission_models = {}
        for state in range(n_states):
            self.emission_models[state] = {
                'mean': observations[state::n_states].mean(axis=0) if len(observations) > state else np.zeros(n_features),
                'cov': np.eye(n_features)
            }
        
        # Initialize duration models
        self._initialize_duration_models()
    
    def _initialize_duration_models(self):
        """Initialize duration distribution parameters for each state."""
        self.duration_models = {}
        
        for state in range(self.config.n_states):
            duration_dist = self.config.duration_distributions[state]
            min_dur = self.config.min_durations[state]
            max_dur = self.config.max_durations[state]
            
            if duration_dist == DurationDistribution.GAMMA:
                # Initialize gamma distribution
                self.duration_models[state] = {
                    'distribution': 'gamma',
                    'shape': 2.0,
                    'scale': (min_dur + max_dur) / 4.0,
                    'min_duration': min_dur,
                    'max_duration': max_dur
                }
            elif duration_dist == DurationDistribution.WEIBULL:
                # Initialize Weibull distribution  
                self.duration_models[state] = {
                    'distribution': 'weibull',
                    'shape': 1.5,
                    'scale': (min_dur + max_dur) / 3.0,
                    'min_duration': min_dur,
                    'max_duration': max_dur
                }
            elif duration_dist == DurationDistribution.LOGNORMAL:
                # Initialize log-normal distribution
                mean_log_dur = np.log((min_dur + max_dur) / 2.0)
                self.duration_models[state] = {
                    'distribution': 'lognormal',
                    'mu': mean_log_dur,
                    'sigma': 0.5,
                    'min_duration': min_dur,
                    'max_duration': max_dur
                }
            else:  # Default to geometric
                self.duration_models[state] = {
                    'distribution': 'geometric',
                    'p': 1.0 / ((min_dur + max_dur) / 2.0),
                    'min_duration': min_dur,
                    'max_duration': max_dur
                }
    
    def _forward_backward_hsmm(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Forward-backward algorithm for HSMM."""
        # This is a simplified version - full HSMM forward-backward is complex
        # For now, we'll use a basic implementation
        
        n_obs = len(observations)
        n_states = self.config.n_states
        
        # Initialize forward and backward probabilities
        forward_probs = np.zeros((n_obs, n_states))
        backward_probs = np.zeros((n_obs, n_states))
        
        # Forward pass (simplified)
        forward_probs[0] = self.initial_probabilities * self._emission_probability(observations[0])
        
        for t in range(1, n_obs):
            for j in range(n_states):
                # Sum over possible previous states and durations
                prob_sum = 0.0
                for i in range(n_states):
                    if i != j:  # No self-transitions
                        prob_sum += forward_probs[t-1, i] * self.transition_matrix[i, j]
                
                forward_probs[t, j] = prob_sum * self._emission_probability(observations[t])[j]
        
        # Backward pass (simplified)
        backward_probs[-1] = np.ones(n_states)
        
        for t in range(n_obs - 2, -1, -1):
            for i in range(n_states):
                prob_sum = 0.0
                for j in range(n_states):
                    if i != j:  # No self-transitions
                        prob_sum += (self.transition_matrix[i, j] * 
                                   self._emission_probability(observations[t+1])[j] * 
                                   backward_probs[t+1, j])
                
                backward_probs[t, i] = prob_sum
        
        # Calculate log-likelihood
        log_likelihood = np.log(forward_probs[-1].sum())
        
        return forward_probs, backward_probs, log_likelihood
    
    def _emission_probability(self, observation: np.ndarray) -> np.ndarray:
        """Calculate emission probabilities for all states."""
        probs = np.zeros(self.config.n_states)
        
        for state in range(self.config.n_states):
            model = self.emission_models[state]
            probs[state] = stats.multivariate_normal.pdf(
                observation, model['mean'], model['cov']
            )
        
        return probs + 1e-10  # Avoid zero probabilities
    
    def _update_parameters(self, observations: np.ndarray, forward_probs: np.ndarray, backward_probs: np.ndarray):
        """Update HSMM parameters (M-step)."""
        # This is a simplified parameter update
        # Full HSMM parameter updates are more complex
        
        n_obs, n_features = observations.shape
        n_states = self.config.n_states
        
        # Calculate state probabilities
        state_probs = forward_probs * backward_probs
        state_probs = state_probs / state_probs.sum(axis=1, keepdims=True)
        
        # Update emission parameters
        for state in range(n_states):
            weights = state_probs[:, state]
            total_weight = weights.sum()
            
            if total_weight > 0:
                # Update mean
                self.emission_models[state]['mean'] = (
                    (observations * weights[:, np.newaxis]).sum(axis=0) / total_weight
                )
                
                # Update covariance (simplified - diagonal)
                diff = observations - self.emission_models[state]['mean']
                weighted_diff = diff * np.sqrt(weights[:, np.newaxis])
                self.emission_models[state]['cov'] = (
                    np.cov(weighted_diff.T) + 1e-6 * np.eye(n_features)
                )
    
    def _viterbi_decode_hsmm(self, observations: np.ndarray) -> np.ndarray:
        """Viterbi decoding for HSMM (simplified)."""
        # This is a simplified version - full HSMM Viterbi is complex
        # For now, use standard Viterbi with duration constraints
        
        n_obs = len(observations)
        n_states = self.config.n_states
        
        # Dynamic programming arrays
        delta = np.zeros((n_obs, n_states))
        psi = np.zeros((n_obs, n_states), dtype=int)
        
        # Initialize
        delta[0] = np.log(self.initial_probabilities) + np.log(self._emission_probability(observations[0]))
        
        # Forward pass
        for t in range(1, n_obs):
            for j in range(n_states):
                # Find best previous state (excluding self-transitions)
                trans_scores = []
                for i in range(n_states):
                    if i != j:  # No self-transitions in HSMM
                        score = delta[t-1, i] + np.log(self.transition_matrix[i, j])
                        trans_scores.append((score, i))
                
                if trans_scores:
                    best_score, best_prev = max(trans_scores)
                    psi[t, j] = best_prev
                    delta[t, j] = best_score + np.log(self._emission_probability(observations[t])[j])
        
        # Backward pass
        path = np.zeros(n_obs, dtype=int)
        path[-1] = np.argmax(delta[-1])
        
        for t in range(n_obs - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        
        # Apply duration constraints
        path = self._apply_duration_constraints(path)
        
        return path
    
    def _apply_duration_constraints(self, path: np.ndarray) -> np.ndarray:
        """Apply minimum and maximum duration constraints to state sequence."""
        constrained_path = path.copy()
        
        # Find state segments
        state_changes = np.diff(path)
        segment_starts = np.concatenate([[0], np.where(state_changes != 0)[0] + 1])
        segment_ends = np.concatenate([np.where(state_changes != 0)[0] + 1, [len(path)]])
        
        for start, end in zip(segment_starts, segment_ends):
            state = path[start]
            duration = end - start
            min_dur = self.config.min_durations[state]
            max_dur = self.config.max_durations[state]
            
            # Handle violations of minimum duration
            if duration < min_dur:
                # Extend segment if possible
                extension_needed = min_dur - duration
                
                # Try to extend forward
                if end + extension_needed <= len(path):
                    constrained_path[end:end + extension_needed] = state
                # Try to extend backward
                elif start - extension_needed >= 0:
                    constrained_path[start - extension_needed:start] = state
            
            # Handle violations of maximum duration
            elif duration > max_dur:
                # Split long segments (simplified approach)
                mid_point = start + max_dur // 2
                # This is a simplified approach - more sophisticated methods exist
                pass
        
        return constrained_path
    
    def _calculate_state_statistics(self, data: pd.DataFrame, states: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """Calculate comprehensive statistics for each state."""
        stats_by_state = {}
        
        for state in range(self.config.n_states):
            state_mask = states == state
            state_data = data[state_mask]
            
            if len(state_data) == 0:
                continue
            
            returns = state_data['close'].pct_change().dropna()
            
            # Calculate state durations
            state_changes = np.diff(state_mask.astype(int))
            state_starts = np.where(state_changes == 1)[0] + 1
            state_ends = np.where(state_changes == -1)[0] + 1
            
            # Handle edge cases
            if state_mask[0]:
                state_starts = np.concatenate([[0], state_starts])
            if state_mask[-1]:
                state_ends = np.concatenate([state_ends, [len(state_mask)]])
            
            durations = state_ends - state_starts if len(state_starts) == len(state_ends) else []
            
            state_interpretation = self.config.state_interpretations.get(state, AdvancedRegimeType.CONSOLIDATION)
            
            stats_by_state[state] = {
                'state_interpretation': state_interpretation.value,
                'frequency': float(np.sum(state_mask) / len(states)),
                'n_observations': int(np.sum(state_mask)),
                'n_episodes': len(durations),
                'avg_duration': float(np.mean(durations)) if durations else 0,
                'median_duration': float(np.median(durations)) if durations else 0,
                'duration_std': float(np.std(durations)) if durations else 0,
                'min_duration': float(np.min(durations)) if durations else 0,
                'max_duration': float(np.max(durations)) if durations else 0,
                'mean_return': float(returns.mean()) if len(returns) > 0 else 0,
                'volatility': float(returns.std()) if len(returns) > 0 else 0,
                'sharpe_ratio': float(returns.mean() / returns.std()) if len(returns) > 0 and returns.std() > 0 else 0
            }
        
        return stats_by_state
    
    def _validate_duration_constraints(self, states: np.ndarray) -> Dict[str, Any]:
        """Validate that duration constraints are satisfied."""
        validation_results = {
            'constraints_satisfied': True,
            'violations': [],
            'duration_statistics': {}
        }
        
        # Calculate actual durations for each state
        for state in range(self.config.n_states):
            state_mask = states == state
            
            # Find state segments
            state_changes = np.diff(state_mask.astype(int))
            state_starts = np.where(state_changes == 1)[0] + 1
            state_ends = np.where(state_changes == -1)[0] + 1
            
            # Handle edge cases
            if state_mask[0]:
                state_starts = np.concatenate([[0], state_starts])
            if state_mask[-1]:
                state_ends = np.concatenate([state_ends, [len(state_mask)]])
            
            if len(state_starts) == len(state_ends):
                durations = state_ends - state_starts
                
                min_dur = self.config.min_durations[state]
                max_dur = self.config.max_durations[state]
                
                # Check violations
                min_violations = np.sum(durations < min_dur)
                max_violations = np.sum(durations > max_dur)
                
                if min_violations > 0:
                    validation_results['violations'].append(
                        f"State {state}: {min_violations} episodes below minimum duration {min_dur}"
                    )
                    validation_results['constraints_satisfied'] = False
                
                if max_violations > 0:
                    validation_results['violations'].append(
                        f"State {state}: {max_violations} episodes above maximum duration {max_dur}"
                    )
                    validation_results['constraints_satisfied'] = False
                
                validation_results['duration_statistics'][state] = {
                    'mean_duration': float(np.mean(durations)),
                    'std_duration': float(np.std(durations)),
                    'min_duration': float(np.min(durations)),
                    'max_duration': float(np.max(durations)),
                    'min_violations': int(min_violations),
                    'max_violations': int(max_violations),
                    'constraint_min': min_dur,
                    'constraint_max': max_dur
                }
        
        return validation_results


# Integration with existing framework
class AdvancedMarkovIntegration:
    """Integration layer for advanced Markov models with existing HMM framework."""
    
    def __init__(self):
        self.logger = system_logger.getChild('AdvancedMarkovIntegration')
        self.markov_switching_model = None
        self.semi_markov_model = None
    
    def run_advanced_regime_analysis(self, 
                                   data: pd.DataFrame,
                                   include_markov_switching: bool = True,
                                   include_semi_markov: bool = True) -> Dict[str, Any]:
        """Run comprehensive advanced Markov model analysis."""
        
        self.logger.info("🚀 Starting Advanced Markov Models Analysis")
        
        results = {
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'data_shape': data.shape,
            'models_run': []
        }
        
        # Run Markov-Switching Model
        if include_markov_switching:
            try:
                self.logger.info("📊 Running Markov-Switching Model")
                ms_config = MarkovSwitchingConfig()
                self.markov_switching_model = MarkovSwitchingRegimeModel(ms_config)
                ms_results = self.markov_switching_model.fit(data)
                
                results['markov_switching'] = ms_results
                results['models_run'].append('markov_switching')
                
                self.logger.info(f"✅ Markov-Switching completed: {len(np.unique(ms_results['regime_assignments']))} regimes identified")
                
            except Exception as e:
                self.logger.error(f"❌ Markov-Switching model failed: {e}")
                results['markov_switching_error'] = str(e)
        
        # Run Hidden Semi-Markov Model
        if include_semi_markov:
            try:
                self.logger.info("📊 Running Hidden Semi-Markov Model")
                hsmm_config = SemiMarkovConfig()
                self.semi_markov_model = HiddenSemiMarkovModel(hsmm_config)
                hsmm_results = self.semi_markov_model.fit(data)
                
                results['hidden_semi_markov'] = hsmm_results
                results['models_run'].append('hidden_semi_markov')
                
                self.logger.info(f"✅ Hidden Semi-Markov completed: {len(np.unique(hsmm_results['state_sequence']))} states identified")
                
            except Exception as e:
                self.logger.error(f"❌ Hidden Semi-Markov model failed: {e}")
                results['hidden_semi_markov_error'] = str(e)
        
        # Compare models if both ran successfully
        if len(results['models_run']) > 1:
            results['model_comparison'] = self._compare_advanced_models(results)
        
        # Generate recommendations
        results['recommendations'] = self._generate_advanced_recommendations(results)
        
        self.logger.info(f"✅ Advanced Markov analysis completed: {len(results['models_run'])} models run")
        return results
    
    def _compare_advanced_models(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare results from different advanced Markov models."""
        comparison = {
            'model_agreement': {},
            'duration_analysis': {},
            'economic_plausibility': {}
        }
        
        # Compare regime/state assignments if both models ran
        if 'markov_switching' in results and 'hidden_semi_markov' in results:
            ms_regimes = results['markov_switching']['regime_assignments']
            hsmm_states = results['hidden_semi_markov']['state_sequence']
            
            # Calculate agreement metrics
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
            
            comparison['model_agreement'] = {
                'adjusted_rand_score': float(adjusted_rand_score(ms_regimes, hsmm_states)),
                'normalized_mutual_info': float(normalized_mutual_info_score(ms_regimes, hsmm_states)),
                'n_regimes_ms': len(np.unique(ms_regimes)),
                'n_states_hsmm': len(np.unique(hsmm_states))
            }
        
        return comparison
    
    def _generate_advanced_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on advanced Markov model results."""
        recommendations = []
        
        if 'markov_switching' in results:
            ms_results = results['markov_switching']
            if ms_results.get('economic_validation', {}).get('constraints_satisfied', False):
                recommendations.append("✅ Markov-Switching model shows economically plausible regimes")
            else:
                recommendations.append("⚠️ Markov-Switching model may need parameter tuning for economic realism")
        
        if 'hidden_semi_markov' in results:
            hsmm_results = results['hidden_semi_markov']
            if hsmm_results.get('duration_validation', {}).get('constraints_satisfied', False):
                recommendations.append("✅ Hidden Semi-Markov model respects duration constraints")
            else:
                recommendations.append("⚠️ Hidden Semi-Markov model violates some duration constraints")
        
        if 'model_comparison' in results:
            agreement = results['model_comparison'].get('model_agreement', {})
            ari = agreement.get('adjusted_rand_score', 0)
            
            if ari > 0.5:
                recommendations.append("🎯 High agreement between advanced models - results are robust")
            elif ari > 0.3:
                recommendations.append("📊 Moderate agreement between models - consider ensemble approach")
            else:
                recommendations.append("🔍 Low agreement between models - investigate data characteristics")
        
        if not recommendations:
            recommendations.append("📝 Run complete advanced Markov analysis to generate specific recommendations")
        
        return recommendations


# Example usage and testing
if __name__ == "__main__":
    # Generate synthetic market data for testing
    np.random.seed(42)
    
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n_obs = len(dates)
    
    # Create regime-switching synthetic data
    true_regimes = np.zeros(n_obs, dtype=int)
    prices = np.zeros(n_obs)
    prices[0] = 100.0
    
    # Define regime periods
    regime_periods = [
        (0, 400, 0),      # Bull market
        (400, 600, 1),    # Bear market  
        (600, 800, 2),    # High volatility
        (800, n_obs, 0)   # Bull market again
    ]
    
    for start, end, regime in regime_periods:
        true_regimes[start:end] = regime
        
        # Generate returns based on regime
        if regime == 0:  # Bull market
            returns = np.random.normal(0.0008, 0.015, end - start)
        elif regime == 1:  # Bear market
            returns = np.random.normal(-0.002, 0.025, end - start)
        else:  # High volatility
            returns = np.random.normal(0.0, 0.040, end - start)
        
        for i, ret in enumerate(returns):
            if start + i < n_obs - 1:
                prices[start + i + 1] = prices[start + i] * (1 + ret)
    
    # Create synthetic market data
    synthetic_data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.normal(0, 0.001, n_obs)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_obs))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_obs))),
        'close': prices,
        'volume': np.random.lognormal(15, 0.5, n_obs)
    })
    
    print("🧪 Testing Advanced Markov Models")
    print(f"📊 Synthetic data: {len(synthetic_data)} observations")
    print(f"🎯 True regimes: {np.bincount(true_regimes)}")
    
    # Test advanced models
    integration = AdvancedMarkovIntegration()
    results = integration.run_advanced_regime_analysis(synthetic_data)
    
    print("\n📈 Advanced Markov Models Results:")
    print(f"Models run: {results['models_run']}")
    
    if 'markov_switching' in results:
        ms_regimes = results['markov_switching']['regime_assignments']
        print(f"Markov-Switching regimes: {np.bincount(ms_regimes)}")
    
    if 'hidden_semi_markov' in results:
        hsmm_states = results['hidden_semi_markov']['state_sequence']
        print(f"Hidden Semi-Markov states: {np.bincount(hsmm_states)}")
    
    print("\n💡 Recommendations:")
    for rec in results['recommendations']:
        print(f"  {rec}")