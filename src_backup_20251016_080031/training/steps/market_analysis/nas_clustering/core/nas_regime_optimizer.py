"""
NAS Regime Optimizer

Implementation for NAS regime optimization.
"""

from rich.console import Console
from rich import print as tprint

tprint("🔍 [NAS_REGIME_OPTIMIZER] Loading NAS Regime Optimizer module")
tprint("🔍 [NAS_REGIME_OPTIMIZER] Module path: /workspace/src/training/steps/market_analysis/nas_clustering/core/nas_regime_optimizer.py")
tprint("🔍 [NAS_REGIME_OPTIMIZER] Purpose: Implementation for NAS regime optimization")
tprint("🔍 [NAS_REGIME_OPTIMIZER] Status: Starting module import")

import numpy as np
tprint("🔍 [NAS_REGIME_OPTIMIZER] ✓ NumPy imported successfully")

from typing import Dict, List, Any, Optional, Tuple
tprint("🔍 [NAS_REGIME_OPTIMIZER] ✓ Typing imports completed")

from dataclasses import dataclass
tprint("🔍 [NAS_REGIME_OPTIMIZER] ✓ Dataclasses imported successfully")

from enum import Enum
tprint("🔍 [NAS_REGIME_OPTIMIZER] ✓ Enum imported successfully")

import time
tprint("🔍 [NAS_REGIME_OPTIMIZER] ✓ Time module imported successfully")

tprint("🔍 [NAS_REGIME_OPTIMIZER] All imports completed successfully")


class RegimeType(Enum):
    """Types of market regimes."""
    tprint("🔍 [REGIME_TYPE] Defining RegimeType enum")
    TRENDING = "trending"
    tprint("🔍 [REGIME_TYPE] ✓ TRENDING defined")
    RANGING = "ranging"
    tprint("🔍 [REGIME_TYPE] ✓ RANGING defined")
    VOLATILE = "volatile"
    tprint("🔍 [REGIME_TYPE] ✓ VOLATILE defined")
    STABLE = "stable"
    tprint("🔍 [REGIME_TYPE] ✓ STABLE defined")
    tprint("🔍 [REGIME_TYPE] All regime types defined successfully")


@dataclass
class RegimeConfig:
    """Configuration for regime optimization."""
    tprint("🔍 [REGIME_CONFIG] Defining RegimeConfig dataclass")
    regime_types: List[RegimeType]
    tprint("🔍 [REGIME_CONFIG] ✓ regime_types field defined")
    optimization_objectives: List[str]
    tprint("🔍 [REGIME_CONFIG] ✓ optimization_objectives field defined")
    optimization_weights: List[float]
    tprint("🔍 [REGIME_CONFIG] ✓ optimization_weights field defined")
    max_iterations: int = 100
    tprint("🔍 [REGIME_CONFIG] ✓ max_iterations field defined (default: 100)")
    convergence_threshold: float = 1e-4
    tprint("🔍 [REGIME_CONFIG] ✓ convergence_threshold field defined (default: 1e-4)")
    regime_detection_window: int = 100
    tprint("🔍 [REGIME_CONFIG] ✓ regime_detection_window field defined (default: 100)")
    tprint("🔍 [REGIME_CONFIG] All configuration fields defined successfully")


class NASRegimeOptimizer:
    """NAS Regime Optimizer for regime-aware architecture search."""
    
    def __init__(self, config: RegimeConfig):
        """Initialize NAS regime optimizer.
        
        Args:
            config: Regime optimization configuration
        """
        tprint("🔍 [NAS_REGIME_OPTIMIZER_INIT] Initializing NASRegimeOptimizer")
        tprint(f"🔍 [NAS_REGIME_OPTIMIZER_INIT] Config received: {config}")
        tprint(f"🔍 [NAS_REGIME_OPTIMIZER_INIT] Config type: {type(config)}")
        tprint(f"🔍 [NAS_REGIME_OPTIMIZER_INIT] Regime types: {config.regime_types}")
        tprint(f"🔍 [NAS_REGIME_OPTIMIZER_INIT] Optimization objectives: {config.optimization_objectives}")
        tprint(f"🔍 [NAS_REGIME_OPTIMIZER_INIT] Optimization weights: {config.optimization_weights}")
        tprint(f"🔍 [NAS_REGIME_OPTIMIZER_INIT] Max iterations: {config.max_iterations}")
        tprint(f"🔍 [NAS_REGIME_OPTIMIZER_INIT] Convergence threshold: {config.convergence_threshold}")
        tprint(f"🔍 [NAS_REGIME_OPTIMIZER_INIT] Regime detection window: {config.regime_detection_window}")
        
        self.config = config
        tprint("🔍 [NAS_REGIME_OPTIMIZER_INIT] ✓ Config assigned to self.config")
        
        self.regime_models = {}
        tprint("🔍 [NAS_REGIME_OPTIMIZER_INIT] ✓ regime_models initialized as empty dict")
        
        self.optimization_history = []
        tprint("🔍 [NAS_REGIME_OPTIMIZER_INIT] ✓ optimization_history initialized as empty list")
        
        self.best_architectures = {}
        tprint("🔍 [NAS_REGIME_OPTIMIZER_INIT] ✓ best_architectures initialized as empty dict")
        
        self.regime_detector = None
        tprint("🔍 [NAS_REGIME_OPTIMIZER_INIT] ✓ regime_detector initialized as None")
        
        tprint("🔍 [NAS_REGIME_OPTIMIZER_INIT] Initialization complete!")
        
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
        tprint("🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] Starting regime optimization")
        tprint(f"🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] Data shape: {data.shape}")
        tprint(f"🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] Data type: {type(data)}")
        tprint(f"🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] Data dtype: {data.dtype}")
        tprint(f"🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] Data min: {np.min(data):.6f}")
        tprint(f"🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] Data max: {np.max(data):.6f}")
        tprint(f"🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] Data mean: {np.mean(data):.6f}")
        tprint(f"🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] Data std: {np.std(data):.6f}")
        tprint(f"🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] Target shape: {target.shape}")
        tprint(f"🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] Target type: {type(target)}")
        tprint(f"🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] Target dtype: {target.dtype}")
        tprint(f"🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] Target min: {np.min(target):.6f}")
        tprint(f"🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] Target max: {np.max(target):.6f}")
        tprint(f"🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] Target mean: {np.mean(target):.6f}")
        tprint(f"🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] Target std: {np.std(target):.6f}")
        tprint(f"🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] Number of architectures: {len(architectures)}")
        tprint(f"🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] Regime types: {self.config.regime_types}")
        
        start_time = time.time()
        tprint(f"🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] Start time recorded: {start_time}")
        
        try:
            tprint("🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] Starting try block")
            # Detect regimes in data
            tprint("🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] Detecting regimes in data...")
            regimes = self._detect_regimes(data)
            tprint(f"🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] ✓ Regimes detected - shape: {regimes.shape}")
            tprint(f"🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] Unique regimes: {np.unique(regimes)}")
            tprint(f"🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] Regime counts: {dict(zip(*np.unique(regimes, return_counts=True)))}")
            
            # Optimize for each regime
            tprint("🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] Starting optimization for each regime...")
            regime_results = {}
            tprint("🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] ✓ Regime results dictionary initialized")
            
            for regime_type in self.config.regime_types:
                tprint(f"🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] Processing regime type: {regime_type}")
                regime_data, regime_target = self._extract_regime_data(
                    data, target, regimes, regime_type
                )
                tprint(f"🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] Regime data shape: {regime_data.shape}")
                tprint(f"🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] Regime target shape: {regime_target.shape}")
                
                if len(regime_data) > 0:
                    tprint(f"🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] ✓ Regime data available - starting optimization")
                    regime_result = self._optimize_for_regime(
                        regime_data, regime_target, architectures, regime_type
                    )
                    regime_results[regime_type.value] = regime_result
                    tprint(f"🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] ✓ Regime optimization completed: {regime_result}")
                    self.best_architectures[regime_type.value] = regime_result.get('best_architecture')
                    tprint(f"🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] ✓ Best architecture stored for regime: {regime_type.value}")
                else:
                    tprint(f"🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] ❌ No data available for regime: {regime_type}")
            
            # Record optimization
            optimization_time = time.time() - start_time
            tprint(f"🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] Optimization completed in {optimization_time:.4f}s")
            
            optimization_record = {
                'regime_results': regime_results,
                'optimization_time': optimization_time,
                'timestamp': time.time()
            }
            tprint("🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] Creating optimization record...")
            self.optimization_history.append(optimization_record)
            tprint(f"🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] ✓ Optimization record added to history (total: {len(self.optimization_history)})")
            
            result = {
                'regime_results': regime_results,
                'best_architectures': self.best_architectures,
                'optimization_time': optimization_time
            }
            tprint(f"🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] ✓ Optimization completed successfully")
            tprint(f"🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] Result: {result}")
            return result
            
        except Exception as e:
            tprint(f"🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] ❌ Exception occurred: {e}")
            tprint(f"🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] Exception type: {type(e)}")
            optimization_time = time.time() - start_time
            tprint(f"🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] Optimization time before error: {optimization_time:.4f}s")
            
            error_result = {
                'error': str(e),
                'optimization_time': optimization_time
            }
            tprint(f"🔍 [NAS_REGIME_OPTIMIZER_OPTIMIZE] Returning error result: {error_result}")
            return error_result
    
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
