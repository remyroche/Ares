"""
NAS Regime Optimizer

Implementation for NAS regime optimization.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time


class RegimeType(Enum):
    """Types of market regimes."""
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    STABLE = "stable"


@dataclass
class RegimeConfig:
    """Configuration for regime optimization."""
    regime_types: List[RegimeType]
    optimization_objectives: List[str]
    optimization_weights: List[float]
    max_iterations: int = 100
    convergence_threshold: float = 1e-4
    regime_detection_window: int = 100


class NASRegimeOptimizer:
    """NAS Regime Optimizer for regime-aware architecture search."""
    
    def __init__(self, config: RegimeConfig):
        """Initialize NAS regime optimizer.
        
        Args:
            config: Regime optimization configuration
        """
        self.config = config
        self.regime_models = {}
        self.optimization_history = []
        self.best_architectures = {}
        self.regime_detector = None
        
    def optimize_regimes(self, data: np.ndarray, target: np.ndarray, 
                        architectures: List[Dict]) -> Dict:
        """Optimize architectures for different regimes.
        
        Args:
            data: Input data
            target: Target data
            architectures: List of architecture specifications
            
        Returns:
            Dictionary containing optimization results
        """
        start_time = time.time()
        
        try:
            # Detect regimes in data
            regimes = self._detect_regimes(data)
            
            # Optimize for each regime
            regime_results = {}
            for regime_type in self.config.regime_types:
                regime_data, regime_target = self._extract_regime_data(
                    data, target, regimes, regime_type
                )
                
                if len(regime_data) > 0:
                    regime_result = self._optimize_for_regime(
                        regime_data, regime_target, architectures, regime_type
                    )
                    regime_results[regime_type.value] = regime_result
                    self.best_architectures[regime_type.value] = regime_result.get('best_architecture')
            
            # Record optimization
            optimization_record = {
                'regime_results': regime_results,
                'optimization_time': time.time() - start_time,
                'timestamp': time.time()
            }
            self.optimization_history.append(optimization_record)
            
            return {
                'regime_results': regime_results,
                'best_architectures': self.best_architectures,
                'optimization_time': time.time() - start_time
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'optimization_time': time.time() - start_time
            }
    
    def _detect_regimes(self, data: np.ndarray) -> np.ndarray:
        """Detect market regimes in data."""
        # Simple regime detection based on volatility and trend
        regimes = np.zeros(len(data))
        
        window_size = self.config.regime_detection_window
        
        for i in range(window_size, len(data)):
            window_data = data[i-window_size:i]
            
            # Calculate regime indicators
            volatility = np.std(window_data)
            trend = np.mean(np.diff(window_data))
            
            # Classify regime
            if volatility > np.percentile(data, 75):
                regimes[i] = RegimeType.VOLATILE.value
            elif abs(trend) > np.percentile(np.abs(np.diff(data)), 75):
                regimes[i] = RegimeType.TRENDING.value
            elif volatility < np.percentile(data, 25):
                regimes[i] = RegimeType.STABLE.value
            else:
                regimes[i] = RegimeType.RANGING.value
        
        return regimes
    
    def _extract_regime_data(self, data: np.ndarray, target: np.ndarray, 
                           regimes: np.ndarray, regime_type: RegimeType) -> Tuple[np.ndarray, np.ndarray]:
        """Extract data for specific regime."""
        regime_mask = regimes == regime_type.value
        regime_data = data[regime_mask]
        regime_target = target[regime_mask]
        
        return regime_data, regime_target
    
    def _optimize_for_regime(self, data: np.ndarray, target: np.ndarray, 
                           architectures: List[Dict], regime_type: RegimeType) -> Dict:
        """Optimize architectures for specific regime."""
        regime_architectures = architectures.copy()
        best_architecture = None
        best_score = float('-inf')
        
        # Evaluate each architecture for this regime
        for architecture in regime_architectures:
            try:
                score = self._evaluate_architecture_for_regime(
                    architecture, data, target, regime_type
                )
                
                if score > best_score:
                    best_score = score
                    best_architecture = architecture.copy()
                
            except Exception as e:
                continue
        
        return {
            'regime_type': regime_type.value,
            'best_architecture': best_architecture,
            'best_score': best_score,
            'data_size': len(data)
        }
    
    def _evaluate_architecture_for_regime(self, architecture: Dict, 
                                         data: np.ndarray, target: np.ndarray, 
                                         regime_type: RegimeType) -> float:
        """Evaluate architecture performance for specific regime."""
        # Simulate regime-specific evaluation
        base_score = np.random.random()
        
        # Adjust score based on regime type
        if regime_type == RegimeType.TRENDING:
            # Trending regimes favor architectures with good trend following
            trend_score = np.random.random() * 0.3
            base_score += trend_score
        elif regime_type == RegimeType.RANGING:
            # Ranging regimes favor architectures with good mean reversion
            range_score = np.random.random() * 0.2
            base_score += range_score
        elif regime_type == RegimeType.VOLATILE:
            # Volatile regimes favor robust architectures
            volatility_score = np.random.random() * 0.4
            base_score += volatility_score
        elif regime_type == RegimeType.STABLE:
            # Stable regimes favor efficient architectures
            efficiency_score = np.random.random() * 0.1
            base_score += efficiency_score
        
        return base_score
    
    def get_best_architecture_for_regime(self, regime_type: RegimeType) -> Optional[Dict]:
        """Get best architecture for specific regime."""
        return self.best_architectures.get(regime_type.value)
    
    def get_all_best_architectures(self) -> Dict:
        """Get best architectures for all regimes."""
        return self.best_architectures
    
    def get_optimization_history(self) -> List[Dict]:
        """Get optimization history."""
        return self.optimization_history
    
    def predict_regime(self, data: np.ndarray) -> RegimeType:
        """Predict regime for new data."""
        if len(data) < self.config.regime_detection_window:
            return RegimeType.STABLE
        
        # Use recent data for regime prediction
        recent_data = data[-self.config.regime_detection_window:]
        
        # Calculate regime indicators
        volatility = np.std(recent_data)
        trend = np.mean(np.diff(recent_data))
        
        # Classify regime
        if volatility > np.percentile(data, 75):
            return RegimeType.VOLATILE
        elif abs(trend) > np.percentile(np.abs(np.diff(data)), 75):
            return RegimeType.TRENDING
        elif volatility < np.percentile(data, 25):
            return RegimeType.STABLE
        else:
            return RegimeType.RANGING
    
    def get_regime_statistics(self, data: np.ndarray) -> Dict:
        """Get statistics about regimes in data."""
        regimes = self._detect_regimes(data)
        
        regime_counts = {}
        for regime_type in self.config.regime_types:
            count = np.sum(regimes == regime_type.value)
            regime_counts[regime_type.value] = count
        
        return {
            'regime_counts': regime_counts,
            'total_samples': len(data),
            'regime_percentages': {
                regime: count / len(data) * 100 
                for regime, count in regime_counts.items()
            }
        }
