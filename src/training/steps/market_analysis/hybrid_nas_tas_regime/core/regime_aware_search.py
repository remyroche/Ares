"""
Regime-Aware Search System

This module provides regime-aware search capabilities that integrate market regime
information into the architecture search process for both NAS and TAS systems.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings

from .financial_architecture_primitives import RegimeType, FinancialActivationType
from .dynamic_search_space import MarketCondition, DynamicSearchSpace

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

logger = logging.getLogger(__name__)


class RegimeDetectionMethod(Enum):
    """Methods for regime detection."""
    KMEANS = "kmeans"
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    HIDDEN_MARKOV = "hidden_markov"
    CHANGEPOINT = "changepoint"
    WAVELET = "wavelet"
    ADAPTIVE = "adaptive"


class RegimeTransitionType(Enum):
    """Types of regime transitions."""
    GRADUAL = "gradual"
    SUDDEN = "sudden"
    CYCLICAL = "cyclical"
    RANDOM = "random"
    TREND_BASED = "trend_based"


@dataclass
class RegimeAwareSearchConfig:
    """Configuration for regime-aware search."""
    # Regime detection
    regime_detection_method: RegimeDetectionMethod = RegimeDetectionMethod.ADAPTIVE
    n_regimes: int = 4
    regime_window: int = 50
    regime_stability_threshold: float = 0.7
    regime_transition_threshold: float = 0.3
    
    # Regime-aware search parameters
    enable_regime_awareness: bool = True
    regime_weight_decay: float = 0.95
    regime_memory_size: int = 1000
    regime_adaptation_rate: float = 0.1
    
    # Regime-specific architecture constraints
    regime_specific_constraints: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'bull_market': {'max_layers': 8, 'min_layers': 3, 'preferred_activations': ['momentum_based', 'sharpe_optimized']},
        'bear_market': {'max_layers': 6, 'min_layers': 2, 'preferred_activations': ['drawdown_aware', 'volatility_sensitive']},
        'sideways': {'max_layers': 5, 'min_layers': 2, 'preferred_activations': ['mean_reversion', 'regime_aware']},
        'high_volatility': {'max_layers': 4, 'min_layers': 2, 'preferred_activations': ['volatility_sensitive', 'drawdown_aware']}
    })
    
    # Regime transition handling
    enable_transition_detection: bool = True
    transition_window: int = 20
    transition_sensitivity: float = 0.5
    
    # Regime performance tracking
    enable_regime_performance_tracking: bool = True
    regime_performance_window: int = 100
    regime_performance_threshold: float = 0.6


@dataclass
class RegimeInfo:
    """Information about a detected regime."""
    regime_id: int
    regime_type: RegimeType
    start_time: datetime
    end_time: Optional[datetime]
    duration: int
    stability: float
    characteristics: Dict[str, Any]
    performance_metrics: Dict[str, float]
    transition_probability: float


@dataclass
class RegimeTransition:
    """Information about regime transitions."""
    from_regime: int
    to_regime: int
    transition_time: datetime
    transition_type: RegimeTransitionType
    transition_strength: float
    transition_duration: int
    transition_indicators: Dict[str, float]


@dataclass
class RegimeAwareSearchResult:
    """Result from regime-aware search."""
    best_architecture: Dict[str, Any]
    best_score: float
    regime_analysis: Dict[str, Any]
    regime_transitions: List[RegimeTransition]
    regime_performance: Dict[int, Dict[str, float]]
    regime_specific_architectures: Dict[int, Dict[str, Any]]
    search_history: List[Dict[str, Any]]
    convergence_info: Dict[str, Any]
    execution_time: float
    n_evaluations: int


class RegimeDetector:
    """Detects market regimes from financial data."""
    
    def __init__(self, config: RegimeAwareSearchConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Regime detection components
        self.regime_detector = None
        self.regime_scaler = StandardScaler()
        self.regime_history = []
        self.regime_labels = []
        
        # Regime tracking
        self.current_regime = None
        self.regime_start_time = None
        self.regime_stability = 0.0
        
        self._initialize_regime_detector()
    
    def _initialize_regime_detector(self):
        """Initialize regime detection method."""
        if self.config.regime_detection_method == RegimeDetectionMethod.KMEANS:
            self.regime_detector = KMeans(
                n_clusters=self.config.n_regimes,
                random_state=42,
                n_init=10
            )
        elif self.config.regime_detection_method == RegimeDetectionMethod.GAUSSIAN_MIXTURE:
            self.regime_detector = GaussianMixture(
                n_components=self.config.n_regimes,
                random_state=42
            )
        else:  # ADAPTIVE
            self.regime_detector = self._create_adaptive_detector()
    
    def _create_adaptive_detector(self):
        """Create adaptive regime detector."""
        # This would be a more sophisticated adaptive detector
        # For now, use Gaussian Mixture as default
        return GaussianMixture(
            n_components=self.config.n_regimes,
            random_state=42
        )
    
    def detect_regimes(self, market_data: pd.DataFrame, 
                      features: Optional[np.ndarray] = None) -> List[RegimeInfo]:
        """Detect regimes from market data."""
        try:
            # Extract features if not provided
            if features is None:
                features = self._extract_regime_features(market_data)
            
            # Scale features
            features_scaled = self.regime_scaler.fit_transform(features)
            
            # Detect regimes
            regime_labels = self.regime_detector.fit_predict(features_scaled)
            
            # Create regime information
            regime_info = self._create_regime_info(regime_labels, market_data)
            
            # Update regime history
            self.regime_history = regime_info
            self.regime_labels = regime_labels
            
            self.logger.info(f"Detected {len(regime_info)} regimes")
            return regime_info
            
        except Exception as e:
            self.logger.error(f"Regime detection failed: {e}")
            return []
    
    def _extract_regime_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Extract features for regime detection."""
        features = []
        
        # Price-based features
        if 'close' in market_data.columns:
            prices = market_data['close'].values
            returns = np.diff(prices) / prices[:-1]
            
            # Volatility features
            volatility = pd.Series(returns).rolling(window=20).std().values
            features.append(volatility)
            
            # Trend features
            trend = pd.Series(prices).rolling(window=20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0]).values
            features.append(trend)
            
            # Momentum features
            momentum = pd.Series(returns).rolling(window=10).mean().values
            features.append(momentum)
        
        # Volume features
        if 'volume' in market_data.columns:
            volume = market_data['volume'].values
            volume_ma = pd.Series(volume).rolling(window=20).mean().values
            volume_ratio = volume / (volume_ma + 1e-8)
            features.append(volume_ratio)
        
        # Combine features
        if features:
            # Pad features to same length
            max_length = max(len(f) for f in features)
            padded_features = []
            for f in features:
                if len(f) < max_length:
                    padded_f = np.pad(f, (0, max_length - len(f)), mode='edge')
                else:
                    padded_f = f[:max_length]
                padded_features.append(padded_f)
            
            return np.column_stack(padded_features)
        else:
            # Return dummy features if no data
            return np.random.randn(len(market_data), 5)
    
    def _create_regime_info(self, regime_labels: np.ndarray, market_data: pd.DataFrame) -> List[RegimeInfo]:
        """Create regime information from labels."""
        regime_info = []
        unique_regimes = np.unique(regime_labels)
        
        for regime_id in unique_regimes:
            regime_mask = regime_labels == regime_id
            regime_data = market_data[regime_mask]
            
            if len(regime_data) == 0:
                continue
            
            # Calculate regime characteristics
            characteristics = self._calculate_regime_characteristics(regime_data)
            
            # Calculate regime stability
            stability = self._calculate_regime_stability(regime_labels, regime_id)
            
            # Determine regime type
            regime_type = self._determine_regime_type(characteristics)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_regime_performance(regime_data)
            
            # Calculate transition probability
            transition_prob = self._calculate_transition_probability(regime_labels, regime_id)
            
            regime_info.append(RegimeInfo(
                regime_id=regime_id,
                regime_type=regime_type,
                start_time=regime_data.index[0] if hasattr(regime_data.index, 'to_pydatetime') else datetime.now(),
                end_time=regime_data.index[-1] if hasattr(regime_data.index, 'to_pydatetime') else datetime.now(),
                duration=len(regime_data),
                stability=stability,
                characteristics=characteristics,
                performance_metrics=performance_metrics,
                transition_probability=transition_prob
            ))
        
        return regime_info
    
    def _calculate_regime_characteristics(self, regime_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate characteristics of a regime."""
        characteristics = {}
        
        if 'close' in regime_data.columns:
            prices = regime_data['close'].values
            returns = np.diff(prices) / prices[:-1]
            
            characteristics.update({
                'volatility': np.std(returns),
                'mean_return': np.mean(returns),
                'trend_strength': self._calculate_trend_strength(prices),
                'momentum': np.mean(returns[-10:]) if len(returns) >= 10 else np.mean(returns),
                'price_range': (np.max(prices) - np.min(prices)) / np.mean(prices)
            })
        
        if 'volume' in regime_data.columns:
            volume = regime_data['volume'].values
            characteristics['volume_ratio'] = np.mean(volume) / (np.std(volume) + 1e-8)
        
        return characteristics
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate trend strength."""
        if len(prices) < 3:
            return 0.0
        
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        
        # Normalize slope
        price_range = np.max(prices) - np.min(prices)
        trend_strength = abs(slope) / (price_range / len(prices))
        
        return min(trend_strength, 1.0)
    
    def _calculate_regime_stability(self, regime_labels: np.ndarray, regime_id: int) -> float:
        """Calculate stability of a regime."""
        regime_mask = regime_labels == regime_id
        regime_indices = np.where(regime_mask)[0]
        
        if len(regime_indices) < 2:
            return 0.0
        
        # Calculate consecutive periods
        consecutive_periods = []
        current_period = 1
        
        for i in range(1, len(regime_indices)):
            if regime_indices[i] == regime_indices[i-1] + 1:
                current_period += 1
            else:
                consecutive_periods.append(current_period)
                current_period = 1
        
        consecutive_periods.append(current_period)
        
        # Stability is the ratio of longest consecutive period to total periods
        max_consecutive = max(consecutive_periods)
        total_periods = len(regime_indices)
        
        return max_consecutive / total_periods if total_periods > 0 else 0.0
    
    def _determine_regime_type(self, characteristics: Dict[str, Any]) -> RegimeType:
        """Determine regime type from characteristics."""
        volatility = characteristics.get('volatility', 0.02)
        trend_strength = characteristics.get('trend_strength', 0.5)
        mean_return = characteristics.get('mean_return', 0.0)
        
        if volatility > 0.03:
            return RegimeType.HIGH_VOLATILITY
        elif volatility < 0.01:
            return RegimeType.LOW_VOLATILITY
        elif trend_strength > 0.7 and mean_return > 0:
            return RegimeType.BULL
        elif trend_strength > 0.7 and mean_return < 0:
            return RegimeType.BEAR
        elif trend_strength < 0.3:
            return RegimeType.SIDEWAYS
        else:
            return RegimeType.TRENDING
    
    def _calculate_regime_performance(self, regime_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics for a regime."""
        if 'close' not in regime_data.columns:
            return {}
        
        prices = regime_data['close'].values
        returns = np.diff(prices) / prices[:-1]
        
        if len(returns) == 0:
            return {}
        
        return {
            'total_return': (prices[-1] - prices[0]) / prices[0],
            'mean_return': np.mean(returns),
            'volatility': np.std(returns),
            'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-8),
            'max_drawdown': self._calculate_max_drawdown(prices),
            'win_rate': np.sum(returns > 0) / len(returns)
        }
    
    def _calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        peak = np.maximum.accumulate(prices)
        drawdown = (peak - prices) / peak
        return np.max(drawdown)
    
    def _calculate_transition_probability(self, regime_labels: np.ndarray, regime_id: int) -> float:
        """Calculate transition probability for a regime."""
        regime_mask = regime_labels == regime_id
        regime_indices = np.where(regime_mask)[0]
        
        if len(regime_indices) < 2:
            return 0.0
        
        # Count transitions from this regime
        transitions = 0
        for i in range(len(regime_indices) - 1):
            if regime_indices[i+1] - regime_indices[i] > 1:
                transitions += 1
        
        return transitions / len(regime_indices)
    
    def detect_transitions(self, regime_labels: np.ndarray) -> List[RegimeTransition]:
        """Detect regime transitions."""
        transitions = []
        
        for i in range(1, len(regime_labels)):
            if regime_labels[i] != regime_labels[i-1]:
                # Calculate transition characteristics
                transition_strength = self._calculate_transition_strength(regime_labels, i)
                transition_type = self._determine_transition_type(regime_labels, i)
                transition_duration = self._calculate_transition_duration(regime_labels, i)
                
                transition = RegimeTransition(
                    from_regime=regime_labels[i-1],
                    to_regime=regime_labels[i],
                    transition_time=datetime.now(),  # Simplified
                    transition_type=transition_type,
                    transition_strength=transition_strength,
                    transition_duration=transition_duration,
                    transition_indicators=self._calculate_transition_indicators(regime_labels, i)
                )
                transitions.append(transition)
        
        return transitions
    
    def _calculate_transition_strength(self, regime_labels: np.ndarray, transition_idx: int) -> float:
        """Calculate strength of a regime transition."""
        # Simplified transition strength calculation
        return np.random.uniform(0.3, 0.9)
    
    def _determine_transition_type(self, regime_labels: np.ndarray, transition_idx: int) -> RegimeTransitionType:
        """Determine type of regime transition."""
        # Simplified transition type determination
        return np.random.choice(list(RegimeTransitionType))
    
    def _calculate_transition_duration(self, regime_labels: np.ndarray, transition_idx: int) -> int:
        """Calculate duration of regime transition."""
        # Simplified transition duration calculation
        return np.random.randint(1, 10)
    
    def _calculate_transition_indicators(self, regime_labels: np.ndarray, transition_idx: int) -> Dict[str, float]:
        """Calculate transition indicators."""
        return {
            'volatility_change': np.random.uniform(-0.5, 0.5),
            'trend_change': np.random.uniform(-0.8, 0.8),
            'volume_change': np.random.uniform(-0.3, 0.3),
            'momentum_change': np.random.uniform(-0.6, 0.6)
        }


class RegimeAwareSearch:
    """Regime-aware search system for financial architectures."""
    
    def __init__(self, config: RegimeAwareSearchConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.regime_detector = RegimeDetector(config)
        self.regime_performance_tracker = RegimePerformanceTracker(config)
        self.regime_architecture_mapper = RegimeArchitectureMapper(config)
        
        # Search state
        self.current_regime = None
        self.regime_history = []
        self.regime_transitions = []
        self.regime_specific_architectures = {}
        
        # Performance tracking
        self.regime_performance = {}
        self.architecture_performance_by_regime = {}
        
        self.logger.info("✅ Regime-Aware Search initialized")
        self.logger.info(f"   Regime Detection Method: {config.regime_detection_method.value}")
        self.logger.info(f"   Number of Regimes: {config.n_regimes}")
        self.logger.info(f"   Regime Awareness: {config.enable_regime_awareness}")
    
    def search(self, architecture_generator: Callable, performance_evaluator: Callable,
               constraint_validator: Callable, n_iterations: int,
               market_data: Optional[pd.DataFrame] = None) -> RegimeAwareSearchResult:
        """Perform regime-aware search."""
        start_time = time.time()
        self.logger.info("🔍 Starting Regime-Aware Search...")
        
        try:
            # Detect initial regimes if market data provided
            if market_data is not None:
                regime_info = self.regime_detector.detect_regimes(market_data)
                self.regime_history = regime_info
                self.logger.info(f"Detected {len(regime_info)} initial regimes")
            
            # Initialize search
            search_history = []
            best_architecture = None
            best_score = -np.inf
            
            # Regime-specific architecture storage
            regime_architectures = {}
            
            # Search loop
            for iteration in range(n_iterations):
                # Update current regime
                current_regime = self._get_current_regime(iteration)
                
                # Generate regime-aware architecture
                architecture = self._generate_regime_aware_architecture(
                    architecture_generator, current_regime
                )
                
                # Validate architecture
                if not constraint_validator(architecture).is_valid:
                    continue
                
                # Evaluate architecture
                performance = performance_evaluator(architecture)
                
                # Update regime-specific performance
                self._update_regime_performance(current_regime, performance)
                
                # Store regime-specific architecture
                if current_regime not in regime_architectures:
                    regime_architectures[current_regime] = []
                regime_architectures[current_regime].append((architecture, performance))
                
                # Update best architecture
                if performance > best_score:
                    best_score = performance
                    best_architecture = architecture
                
                # Detect regime transitions
                if self.config.enable_transition_detection:
                    transitions = self._detect_regime_transitions(iteration)
                    self.regime_transitions.extend(transitions)
                
                # Store search history
                search_history.append({
                    'iteration': iteration,
                    'architecture': architecture,
                    'performance': performance,
                    'regime': current_regime,
                    'regime_stability': self._get_regime_stability(current_regime),
                    'timestamp': datetime.now()
                })
                
                # Log progress
                if iteration % 100 == 0:
                    self.logger.debug(f"Iteration {iteration}: Performance = {performance:.4f}, Regime = {current_regime}")
            
            execution_time = time.time() - start_time
            
            # Analyze regime performance
            regime_analysis = self._analyze_regime_performance()
            regime_performance = self._calculate_regime_performance_metrics()
            
            # Get best architectures by regime
            regime_specific_architectures = self._get_best_architectures_by_regime(regime_architectures)
            
            return RegimeAwareSearchResult(
                best_architecture=best_architecture,
                best_score=best_score,
                regime_analysis=regime_analysis,
                regime_transitions=self.regime_transitions,
                regime_performance=regime_performance,
                regime_specific_architectures=regime_specific_architectures,
                search_history=search_history,
                convergence_info=self._analyze_convergence(search_history),
                execution_time=execution_time,
                n_evaluations=len(search_history)
            )
            
        except Exception as e:
            self.logger.error(f"Regime-aware search failed: {e}")
            return self._create_error_result(str(e), time.time() - start_time)
    
    def _get_current_regime(self, iteration: int) -> int:
        """Get current regime for iteration."""
        if not self.regime_history:
            return 0
        
        # Simplified regime selection
        # In practice, this would use actual regime detection
        regime_cycle = iteration // 50  # Change regime every 50 iterations
        return regime_cycle % len(self.regime_history)
    
    def _generate_regime_aware_architecture(self, architecture_generator: Callable, 
                                         current_regime: int) -> Dict[str, Any]:
        """Generate architecture aware of current regime."""
        # Get base architecture
        architecture = architecture_generator()
        
        # Apply regime-specific modifications
        if current_regime < len(self.regime_history):
            regime_info = self.regime_history[current_regime]
            architecture = self._apply_regime_specific_modifications(architecture, regime_info)
        
        return architecture
    
    def _apply_regime_specific_modifications(self, architecture: Dict[str, Any], 
                                           regime_info: RegimeInfo) -> Dict[str, Any]:
        """Apply regime-specific modifications to architecture."""
        modified_architecture = architecture.copy()
        
        # Get regime-specific constraints
        regime_type = regime_info.regime_type.value
        if regime_type in self.config.regime_specific_constraints:
            constraints = self.config.regime_specific_constraints[regime_type]
            
            # Apply layer constraints
            if 'max_layers' in constraints:
                max_layers = constraints['max_layers']
                if 'layers' in modified_architecture:
                    modified_architecture['layers'] = modified_architecture['layers'][:max_layers]
            
            # Apply activation preferences
            if 'preferred_activations' in constraints:
                preferred_activations = constraints['preferred_activations']
                if 'activation' in modified_architecture:
                    # Select from preferred activations
                    modified_architecture['activation'] = np.random.choice(preferred_activations)
        
        # Apply regime characteristics
        if regime_info.characteristics:
            characteristics = regime_info.characteristics
            
            # Adjust for volatility
            if 'volatility' in characteristics:
                volatility = characteristics['volatility']
                if volatility > 0.03:  # High volatility
                    modified_architecture['volatility_sensitive'] = True
                    modified_architecture['activation'] = 'volatility_sensitive'
                elif volatility < 0.01:  # Low volatility
                    modified_architecture['activation'] = 'momentum_based'
            
            # Adjust for trend strength
            if 'trend_strength' in characteristics:
                trend_strength = characteristics['trend_strength']
                if trend_strength > 0.7:  # Strong trend
                    modified_architecture['activation'] = 'sharpe_optimized'
                elif trend_strength < 0.3:  # Weak trend
                    modified_architecture['activation'] = 'mean_reversion'
        
        return modified_architecture
    
    def _update_regime_performance(self, regime: int, performance: float):
        """Update performance tracking for regime."""
        if regime not in self.regime_performance:
            self.regime_performance[regime] = []
        
        self.regime_performance[regime].append(performance)
        
        # Keep only recent performance
        if len(self.regime_performance[regime]) > self.config.regime_performance_window:
            self.regime_performance[regime].pop(0)
    
    def _detect_regime_transitions(self, iteration: int) -> List[RegimeTransition]:
        """Detect regime transitions."""
        # Simplified transition detection
        # In practice, this would use actual regime transition detection
        transitions = []
        
        if iteration > 0 and iteration % 100 == 0:
            # Simulate regime transition
            from_regime = (iteration - 100) // 50 % len(self.regime_history)
            to_regime = iteration // 50 % len(self.regime_history)
            
            if from_regime != to_regime:
                transition = RegimeTransition(
                    from_regime=from_regime,
                    to_regime=to_regime,
                    transition_time=datetime.now(),
                    transition_type=RegimeTransitionType.GRADUAL,
                    transition_strength=np.random.uniform(0.3, 0.8),
                    transition_duration=np.random.randint(5, 20),
                    transition_indicators={
                        'volatility_change': np.random.uniform(-0.3, 0.3),
                        'trend_change': np.random.uniform(-0.5, 0.5),
                        'volume_change': np.random.uniform(-0.2, 0.2)
                    }
                )
                transitions.append(transition)
        
        return transitions
    
    def _get_regime_stability(self, regime: int) -> float:
        """Get stability of current regime."""
        if regime < len(self.regime_history):
            return self.regime_history[regime].stability
        return 0.5
    
    def _analyze_regime_performance(self) -> Dict[str, Any]:
        """Analyze performance across regimes."""
        analysis = {}
        
        for regime, performances in self.regime_performance.items():
            if performances:
                analysis[f'regime_{regime}'] = {
                    'mean_performance': np.mean(performances),
                    'std_performance': np.std(performances),
                    'max_performance': np.max(performances),
                    'min_performance': np.min(performances),
                    'count': len(performances)
                }
        
        return analysis
    
    def _calculate_regime_performance_metrics(self) -> Dict[int, Dict[str, float]]:
        """Calculate performance metrics by regime."""
        metrics = {}
        
        for regime, performances in self.regime_performance.items():
            if performances:
                metrics[regime] = {
                    'mean_performance': np.mean(performances),
                    'std_performance': np.std(performances),
                    'sharpe_ratio': np.mean(performances) / (np.std(performances) + 1e-8),
                    'max_drawdown': np.max(performances) - np.min(performances),
                    'win_rate': np.sum(np.array(performances) > np.mean(performances)) / len(performances)
                }
        
        return metrics
    
    def _get_best_architectures_by_regime(self, regime_architectures: Dict[int, List[Tuple[Dict[str, Any], float]]]) -> Dict[int, Dict[str, Any]]:
        """Get best architectures by regime."""
        best_architectures = {}
        
        for regime, architectures in regime_architectures.items():
            if architectures:
                # Find best architecture for this regime
                best_arch, best_perf = max(architectures, key=lambda x: x[1])
                best_architectures[regime] = {
                    'architecture': best_arch,
                    'performance': best_perf,
                    'count': len(architectures)
                }
        
        return best_architectures
    
    def _analyze_convergence(self, search_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze search convergence."""
        if len(search_history) < 10:
            return {'converged': False, 'reason': 'insufficient_data'}
        
        performances = [entry['performance'] for entry in search_history]
        recent_performances = performances[-10:]
        
        performance_std = np.std(recent_performances)
        performance_trend = np.mean(recent_performances[-5:]) - np.mean(recent_performances[:5])
        
        return {
            'converged': performance_std < 0.01 and abs(performance_trend) < 0.001,
            'performance_std': performance_std,
            'performance_trend': performance_trend,
            'regime_diversity': len(set(entry['regime'] for entry in search_history))
        }
    
    def _create_error_result(self, error_message: str, execution_time: float) -> RegimeAwareSearchResult:
        """Create error result."""
        return RegimeAwareSearchResult(
            best_architecture={},
            best_score=0.0,
            regime_analysis={},
            regime_transitions=[],
            regime_performance={},
            regime_specific_architectures={},
            search_history=[],
            convergence_info={'error': error_message},
            execution_time=execution_time,
            n_evaluations=0
        )


class RegimePerformanceTracker:
    """Tracks performance by regime."""
    
    def __init__(self, config: RegimeAwareSearchConfig):
        self.config = config
        self.regime_performance = {}
        self.regime_architecture_performance = {}
    
    def update(self, regime: int, architecture: Dict[str, Any], performance: float):
        """Update performance tracking."""
        if regime not in self.regime_performance:
            self.regime_performance[regime] = []
            self.regime_architecture_performance[regime] = {}
        
        self.regime_performance[regime].append(performance)
        
        # Track by architecture type
        arch_type = architecture.get('type', 'unknown')
        if arch_type not in self.regime_architecture_performance[regime]:
            self.regime_architecture_performance[regime][arch_type] = []
        self.regime_architecture_performance[regime][arch_type].append(performance)


class RegimeArchitectureMapper:
    """Maps regimes to optimal architectures."""
    
    def __init__(self, config: RegimeAwareSearchConfig):
        self.config = config
        self.regime_architecture_map = {}
    
    def map_regime_to_architecture(self, regime: int, regime_info: RegimeInfo) -> Dict[str, Any]:
        """Map regime to optimal architecture."""
        # Simplified mapping
        # In practice, this would use learned mappings
        return {
            'type': 'neural',
            'layers': [{'hidden_size': 64, 'dropout': 0.2}],
            'activation': 'regime_aware',
            'regime_aware': True,
            'volatility_sensitive': True
        }


def create_regime_aware_search(config: RegimeAwareSearchConfig) -> RegimeAwareSearch:
    """Create regime-aware search instance."""
    return RegimeAwareSearch(config)

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
