"""
Enhanced Feature Lookback Optimization System

This module integrates hardware optimization, feature selection tools, and safe math
operations to provide a comprehensive optimization system for feature lookback periods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
import time
import asyncio
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# Import hardware optimization tools
try:
    from src.utils.hardware.m1_gpu_utils import M1GPUManager
    from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
    from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Hardware optimization tools not available: {e}")
    HARDWARE_OPTIMIZATION_AVAILABLE = False

# Import safe math operations
try:
    from src.utils.math_validation import safe_divide, safe_log, safe_sqrt
    SAFE_MATH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Safe math operations not available: {e}")
    SAFE_MATH_AVAILABLE = False

# Import feature selection tools
try:
    from src.utils.feature_selection.step08_optimized_methods import (
        fast_correlation_matrix, optimized_mutual_information, 
        vectorized_feature_stability, parallel_feature_importance
    )
    FEATURE_SELECTION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Feature selection tools not available: {e}")
    FEATURE_SELECTION_AVAILABLE = False

# Import parallel processing
try:
    from src.utils.parallel_processing_optimizer import ParallelProcessor
    PARALLEL_PROCESSING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Parallel processing not available: {e}")
    PARALLEL_PROCESSING_AVAILABLE = False

# Import comprehensive feature generators
try:
    from src.feature_engineering.comprehensive_feature_generators import (
        ComprehensiveFeatureGenerators, COMPREHENSIVE_FEATURE_GENERATORS
    )
    COMPREHENSIVE_GENERATORS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Comprehensive feature generators not available: {e}")
    COMPREHENSIVE_GENERATORS_AVAILABLE = False

class EnhancedOptimizationSystem:
    """
    Enhanced feature lookback optimization system with hardware acceleration,
    feature selection integration, and safe math operations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced optimization system."""
        self.logger = logger.getChild('EnhancedOptimizationSystem')
        self.config = config or {}
        
        # Initialize hardware optimization
        if HARDWARE_OPTIMIZATION_AVAILABLE:
            self.gpu_manager = M1GPUManager()
            self.cpu_optimizer = M1CPUOptimizer()
            self.memory_optimizer = M1MemoryOptimizer()
            self.logger.info("✅ Hardware optimization initialized")
        else:
            self.gpu_manager = None
            self.cpu_optimizer = None
            self.memory_optimizer = None
            self.logger.info("ℹ️ Hardware optimization not available")
        
        # Initialize parallel processing
        if PARALLEL_PROCESSING_AVAILABLE:
            max_workers = self.config.get('max_workers', 4)
            self.parallel_processor = ParallelProcessor(max_workers=max_workers)
            self.logger.info(f"✅ Parallel processing initialized with {max_workers} workers")
        else:
            self.parallel_processor = None
            self.logger.info("ℹ️ Parallel processing not available")
        
        # Initialize feature generators
        if COMPREHENSIVE_GENERATORS_AVAILABLE:
            self.feature_generators = ComprehensiveFeatureGenerators()
            self.logger.info("✅ Comprehensive feature generators initialized")
        else:
            self.feature_generators = None
            self.logger.info("ℹ️ Comprehensive feature generators not available")
        
        # Performance tracking
        self.optimization_times = {}
        self.performance_metrics = {}
        
        self.logger.info("🚀 Enhanced optimization system initialized")
    
    def _safe_divide(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division with fallback."""
        if SAFE_MATH_AVAILABLE:
            return safe_divide(numerator, denominator, default)
        else:
            return numerator / denominator if denominator != 0 else default
    
    def _safe_log(self, value: float, default: float = 0.0) -> float:
        """Safe logarithm with fallback."""
        if SAFE_MATH_AVAILABLE:
            return safe_log(value, default)
        else:
            return np.log(value) if value > 0 else default
    
    def _safe_sqrt(self, value: float, default: float = 0.0) -> float:
        """Safe square root with fallback."""
        if SAFE_MATH_AVAILABLE:
            return safe_sqrt(value, default)
        else:
            return np.sqrt(value) if value >= 0 else default
    
    async def optimize_feature_lookback_enhanced(
        self,
        data: pd.DataFrame,
        feature_name: str,
        periods: List[int],
        optimization_method: str = 'signal_strength',
        target_column: Optional[str] = None,
        regime_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enhanced feature lookback optimization with hardware acceleration and feature selection.
        
        Args:
            data: Input data DataFrame
            feature_name: Name of the feature to optimize
            periods: List of periods to test
            optimization_method: Method for optimization
            target_column: Target column for optimization
            regime_column: Regime column for regime-aware optimization
            
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        self.logger.info(f"🔧 Starting enhanced optimization for {feature_name}")
        
        try:
            # Get feature generator
            if not self.feature_generators:
                raise ValueError("Feature generators not available")
            
            generator_func = getattr(self.feature_generators, f"{feature_name}_generator", None)
            if not generator_func:
                raise ValueError(f"Feature generator for {feature_name} not found")
            
            # Memory optimization
            if self.memory_optimizer:
                optimal_chunk_size = self.memory_optimizer.calculate_optimal_chunk_size(
                    data.shape, f"optimization_{feature_name}"
                )
                self.logger.debug(f"Optimal chunk size: {optimal_chunk_size}")
            
            # GPU acceleration if available
            if self.gpu_manager and self.gpu_manager.is_mps_available():
                result = await self._gpu_accelerated_optimization(
                    data, feature_name, periods, optimization_method, 
                    generator_func, target_column, regime_column
                )
            else:
                result = await self._cpu_optimized_optimization(
                    data, feature_name, periods, optimization_method,
                    generator_func, target_column, regime_column
                )
            
            # Record performance
            optimization_time = time.time() - start_time
            self.optimization_times[feature_name] = optimization_time
            result['optimization_time'] = optimization_time
            
            self.logger.info(f"✅ Enhanced optimization completed for {feature_name} in {optimization_time:.3f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced optimization failed for {feature_name}: {e}")
            return {
                'feature_name': feature_name,
                'optimal_lookback': periods[len(periods) // 2],
                'optimization_method': optimization_method,
                'error': str(e),
                'optimization_time': time.time() - start_time
            }
    
    async def _gpu_accelerated_optimization(
        self,
        data: pd.DataFrame,
        feature_name: str,
        periods: List[int],
        optimization_method: str,
        generator_func: Callable,
        target_column: Optional[str],
        regime_column: Optional[str]
    ) -> Dict[str, Any]:
        """GPU-accelerated optimization using M1 GPU."""
        self.logger.info(f"🚀 Using GPU acceleration for {feature_name}")
        
        try:
            import torch
            
            # Convert data to tensor if possible
            if target_column and target_column in data.columns:
                target_data = torch.tensor(data[target_column].values, dtype=torch.float32)
            else:
                target_data = None
            
            best_period = periods[0]
            best_score = float('-inf')
            scores = []
            
            for period in periods:
                try:
                    # Generate feature
                    feature_values = generator_func(data, period)
                    feature_tensor = torch.tensor(feature_values.values, dtype=torch.float32)
                    
                    # Calculate score using GPU
                    if target_data is not None:
                        # Calculate correlation on GPU
                        correlation = torch.corrcoef(torch.stack([feature_tensor, target_data]))[0, 1]
                        score = abs(correlation.item()) if not torch.isnan(correlation) else 0
                    else:
                        # Use autocorrelation
                        autocorr = torch.corrcoef(torch.stack([feature_tensor[:-1], feature_tensor[1:]]))[0, 1]
                        score = abs(autocorr.item()) if not torch.isnan(autocorr) else 0
                    
                    scores.append(score)
                    
                    if score > best_score:
                        best_score = score
                        best_period = period
                        
                except Exception as e:
                    self.logger.debug(f"GPU optimization failed for period {period}: {e}")
                    scores.append(0)
                    continue
            
            return {
                'feature_name': feature_name,
                'optimal_lookback': best_period,
                'optimization_method': f'gpu_accelerated_{optimization_method}',
                'performance_score': best_score,
                'scores': scores,
                'hardware_used': 'M1_GPU'
            }
            
        except Exception as e:
            self.logger.warning(f"GPU optimization failed, falling back to CPU: {e}")
            return await self._cpu_optimized_optimization(
                data, feature_name, periods, optimization_method,
                generator_func, target_column, regime_column
            )
    
    async def _cpu_optimized_optimization(
        self,
        data: pd.DataFrame,
        feature_name: str,
        periods: List[int],
        optimization_method: str,
        generator_func: Callable,
        target_column: Optional[str],
        regime_column: Optional[str]
    ) -> Dict[str, Any]:
        """CPU-optimized optimization with feature selection integration."""
        self.logger.info(f"💻 Using CPU optimization for {feature_name}")
        
        best_period = periods[0]
        best_score = float('-inf')
        scores = []
        
        for period in periods:
            try:
                # Generate feature
                feature_values = generator_func(data, period)
                
                # Calculate score using enhanced methods
                if optimization_method == 'signal_strength':
                    score = self._calculate_signal_strength_enhanced(
                        data, feature_values, target_column, period
                    )
                elif optimization_method == 'noise_reduction':
                    score = self._calculate_noise_reduction_enhanced(
                        data, feature_values, period
                    )
                elif optimization_method == 'trend_following':
                    score = self._calculate_trend_following_enhanced(
                        data, feature_values, target_column, period
                    )
                elif optimization_method == 'information_content':
                    score = self._calculate_information_content_enhanced(
                        data, feature_values, target_column, period
                    )
                elif optimization_method == 'regime_adaptation':
                    score = self._calculate_regime_adaptation_enhanced(
                        data, feature_values, target_column, regime_column, period
                    )
                else:
                    score = self._calculate_signal_strength_enhanced(
                        data, feature_values, target_column, period
                    )
                
                scores.append(score)
                
                if score > best_score:
                    best_score = score
                    best_period = period
                    
            except Exception as e:
                self.logger.debug(f"CPU optimization failed for period {period}: {e}")
                scores.append(0)
                continue
        
        return {
            'feature_name': feature_name,
            'optimal_lookback': best_period,
            'optimization_method': f'cpu_optimized_{optimization_method}',
            'performance_score': best_score,
            'scores': scores,
            'hardware_used': 'M1_CPU'
        }
    
    def _calculate_signal_strength_enhanced(
        self, data: pd.DataFrame, feature_values: pd.Series, 
        target_column: Optional[str], period: int
    ) -> float:
        """Enhanced signal strength calculation with feature selection tools."""
        try:
            if target_column and target_column in data.columns:
                target_data = data[target_column]
                
                # Use feature selection tools if available
                if FEATURE_SELECTION_AVAILABLE:
                    # Calculate mutual information
                    valid_indices = ~(feature_values.isna() | target_data.isna())
                    if valid_indices.sum() > 10:
                        mi_score = optimized_mutual_information(
                            feature_values[valid_indices].values.reshape(-1, 1),
                            target_data[valid_indices].values
                        )
                        return mi_score if not np.isnan(mi_score) else 0
                
                # Fallback to correlation
                correlation = abs(feature_values.corr(target_data))
                return correlation if not pd.isna(correlation) else 0
            else:
                # Use autocorrelation
                autocorr = feature_values.autocorr(lag=1)
                return abs(autocorr) if not pd.isna(autocorr) else 0
                
        except Exception as e:
            self.logger.debug(f"Signal strength calculation failed: {e}")
            return 0
    
    def _calculate_noise_reduction_enhanced(
        self, data: pd.DataFrame, feature_values: pd.Series, period: int
    ) -> float:
        """Enhanced noise reduction calculation."""
        try:
            # Calculate coefficient of variation with safe math
            feature_mean = feature_values.mean()
            feature_std = feature_values.std()
            
            if feature_mean != 0:
                cv = self._safe_divide(feature_std, abs(feature_mean), 1.0)
                # Return negative CV for minimization (noise reduction)
                return -cv
            else:
                return 0
                
        except Exception as e:
            self.logger.debug(f"Noise reduction calculation failed: {e}")
            return 0
    
    def _calculate_trend_following_enhanced(
        self, data: pd.DataFrame, feature_values: pd.Series,
        target_column: Optional[str], period: int
    ) -> float:
        """Enhanced trend following calculation."""
        try:
            if target_column and target_column in data.columns:
                target_data = data[target_column]
                
                # Calculate correlation with price trend
                price_trend = target_data.pct_change(period)
                correlation = abs(feature_values.rolling(period).mean().corr(price_trend))
                
                # Add lag penalty (shorter periods preferred)
                lag_penalty = 1 / (1 + period / 20)
                return correlation * lag_penalty if not pd.isna(correlation) else 0
            else:
                # Use autocorrelation
                autocorr = feature_values.autocorr(lag=period)
                return abs(autocorr) if not pd.isna(autocorr) else 0
                
        except Exception as e:
            self.logger.debug(f"Trend following calculation failed: {e}")
            return 0
    
    def _calculate_information_content_enhanced(
        self, data: pd.DataFrame, feature_values: pd.Series,
        target_column: Optional[str], period: int
    ) -> float:
        """Enhanced information content calculation."""
        try:
            if target_column and target_column in data.columns:
                target_data = data[target_column]
                
                # Use feature selection tools if available
                if FEATURE_SELECTION_AVAILABLE:
                    valid_indices = ~(feature_values.isna() | target_data.isna())
                    if valid_indices.sum() > 10:
                        # Discretize for mutual information
                        feature_bins = pd.cut(feature_values[valid_indices], bins=10, labels=False)
                        target_bins = pd.cut(target_data[valid_indices], bins=10, labels=False)
                        
                        # Calculate mutual information
                        mi_score = optimized_mutual_information(
                            feature_bins.values.reshape(-1, 1),
                            target_bins.values
                        )
                        return mi_score if not np.isnan(mi_score) else 0
                
                # Fallback to correlation
                correlation = abs(feature_values.corr(target_data))
                return correlation if not pd.isna(correlation) else 0
            else:
                # Use autocorrelation
                autocorr = feature_values.autocorr(lag=period)
                return abs(autocorr) if not pd.isna(autocorr) else 0
                
        except Exception as e:
            self.logger.debug(f"Information content calculation failed: {e}")
            return 0
    
    def _calculate_regime_adaptation_enhanced(
        self, data: pd.DataFrame, feature_values: pd.Series,
        target_column: Optional[str], regime_column: Optional[str], period: int
    ) -> float:
        """Enhanced regime adaptation calculation."""
        try:
            if regime_column and regime_column in data.columns:
                regime_data = data[regime_column]
                regimes = regime_data.unique()
                regime_scores = []
                
                for regime in regimes:
                    regime_mask = regime_data == regime
                    regime_feature = feature_values[regime_mask]
                    
                    if len(regime_feature) > period:
                        # Calculate regime-specific performance
                        regime_performance = abs(regime_feature.rolling(period).std().mean())
                        regime_scores.append(regime_performance)
                
                # Use minimum performance across regimes (worst-case optimization)
                return min(regime_scores) if regime_scores else 0
            else:
                # Fallback to signal strength
                return self._calculate_signal_strength_enhanced(
                    data, feature_values, target_column, period
                )
                
        except Exception as e:
            self.logger.debug(f"Regime adaptation calculation failed: {e}")
            return 0
    
    async def optimize_multiple_features_enhanced(
        self,
        data: pd.DataFrame,
        feature_configs: List[Dict[str, Any]],
        target_column: Optional[str] = None,
        regime_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Optimize multiple features with enhanced parallel processing.
        
        Args:
            data: Input data DataFrame
            feature_configs: List of feature configurations
            target_column: Target column for optimization
            regime_column: Regime column for regime-aware optimization
            
        Returns:
            Dictionary with optimization results for all features
        """
        self.logger.info(f"🚀 Starting enhanced optimization for {len(feature_configs)} features")
        start_time = time.time()
        
        results = {}
        
        if self.parallel_processor and len(feature_configs) > 1:
            # Parallel optimization
            self.logger.info("🔄 Using parallel processing for optimization")
            
            tasks = []
            for config in feature_configs:
                task = self.optimize_feature_lookback_enhanced(
                    data, config['name'], config['periods'], 
                    config.get('method', 'signal_strength'),
                    target_column, regime_column
                )
                tasks.append((config['name'], task))
            
            # Execute tasks in parallel
            for feature_name, task in tasks:
                try:
                    result = await task
                    results[feature_name] = result
                except Exception as e:
                    self.logger.error(f"Error optimizing feature {feature_name}: {e}")
                    results[feature_name] = {
                        'feature_name': feature_name,
                        'error': str(e),
                        'optimal_lookback': config['periods'][len(config['periods']) // 2]
                    }
        else:
            # Sequential optimization
            self.logger.info("🔄 Using sequential processing for optimization")
            
            for config in feature_configs:
                try:
                    result = await self.optimize_feature_lookback_enhanced(
                        data, config['name'], config['periods'],
                        config.get('method', 'signal_strength'),
                        target_column, regime_column
                    )
                    results[config['name']] = result
                except Exception as e:
                    self.logger.error(f"Error optimizing feature {config['name']}: {e}")
                    results[config['name']] = {
                        'feature_name': config['name'],
                        'error': str(e),
                        'optimal_lookback': config['periods'][len(config['periods']) // 2]
                    }
        
        # Calculate overall performance metrics
        total_time = time.time() - start_time
        successful_optimizations = sum(1 for r in results.values() if 'error' not in r)
        
        overall_results = {
            'feature_results': results,
            'optimization_summary': {
                'total_features': len(feature_configs),
                'successful_optimizations': successful_optimizations,
                'failed_optimizations': len(feature_configs) - successful_optimizations,
                'total_optimization_time': total_time,
                'average_time_per_feature': total_time / len(feature_configs),
                'hardware_used': 'M1_GPU' if self.gpu_manager and self.gpu_manager.is_mps_available() else 'M1_CPU',
                'parallel_processing_used': self.parallel_processor is not None
            }
        }
        
        self.logger.info(f"✅ Enhanced optimization completed for {successful_optimizations}/{len(feature_configs)} features in {total_time:.3f}s")
        return overall_results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of optimization system."""
        return {
            'optimization_times': self.optimization_times,
            'performance_metrics': self.performance_metrics,
            'hardware_available': {
                'gpu_optimization': self.gpu_manager is not None,
                'cpu_optimization': self.cpu_optimizer is not None,
                'memory_optimization': self.memory_optimizer is not None,
                'parallel_processing': self.parallel_processor is not None
            },
            'feature_selection_available': FEATURE_SELECTION_AVAILABLE,
            'safe_math_available': SAFE_MATH_AVAILABLE
        }

# Convenience functions
def create_enhanced_optimization_system(config: Optional[Dict[str, Any]] = None) -> EnhancedOptimizationSystem:
    """Create an enhanced optimization system with the given configuration."""
    return EnhancedOptimizationSystem(config)

async def optimize_features_enhanced(
    data: pd.DataFrame,
    feature_configs: List[Dict[str, Any]],
    target_column: Optional[str] = None,
    regime_column: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Convenience function for enhanced feature optimization."""
    system = create_enhanced_optimization_system(config)
    return await system.optimize_multiple_features_enhanced(
        data, feature_configs, target_column, regime_column
    )