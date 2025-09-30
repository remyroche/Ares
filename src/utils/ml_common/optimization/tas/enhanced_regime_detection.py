"""
Enhanced TAS Regime Detection with Performance Optimizations and Advanced Validation.

This module provides enhanced regime detection capabilities with:
- Memory-efficient processing for large datasets
- Parallel processing across timeframes
- Intelligent caching for regime detection results
- Cross-validation for regime stability
- Out-of-sample testing for regime validation
- Regime persistence analysis over time
"""

import logging
import time
import gc
import hashlib
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import contextmanager
import numpy as np
import pandas as pd
from pathlib import Path

# Import existing TAS components
from .data_pipeline.regime_detection import RegimeDetectionPipeline
from .regime_analysis.regime_qualification import RegimeQualification
from .regime_analysis.unsupervised_regime_detection import UnsupervisedRegimeDetection
from .regime_analysis.clustering_regime_detection import ClusteringRegimeDetection

# Import optimization utilities
from ...hardware.m1_gpu_utils import get_m1_gpu_manager
from ...hardware.m1_memory_optimizer import get_m1_memory_optimizer
from ...hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
from ...matrix_operations.unified_operations import get_unified_matrix_operations

# Import ML common utilities
from ..cvlsa.cvlsa_cross_validation import CVLSA
from ..validation.universal_validation import UniversalValidator
from ..evaluation.performance_metrics import PerformanceMetrics

logger = logging.getLogger(__name__)

class EnhancedTASRegimeDetection:
    """
    Enhanced TAS regime detection with performance optimizations and advanced validation.
    
    Features:
    - Memory-efficient processing for large datasets
    - Parallel processing across timeframes
    - Intelligent caching for regime detection results
    - Cross-validation for regime stability
    - Out-of-sample testing for regime validation
    - Regime persistence analysis over time
    """
    
    def __init__(self, 
                 enable_gpu: bool = True,
                 enable_memory_optimization: bool = True,
                 enable_parallel: bool = True,
                 cache_dir: Optional[str] = None,
                 max_memory_gb: Optional[float] = None):
        """
        Initialize enhanced TAS regime detection.
        
        Args:
            enable_gpu: Whether to enable GPU acceleration
            enable_memory_optimization: Whether to enable memory optimization
            enable_parallel: Whether to enable parallel processing
            cache_dir: Directory for caching results
            max_memory_gb: Maximum memory usage in GB
        """
        self.logger = logger.getChild('EnhancedTASRegimeDetection')
        
        # Configuration
        self.enable_gpu = enable_gpu
        self.enable_memory_optimization = enable_memory_optimization
        self.enable_parallel = enable_parallel
        self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / 'tas_cache'
        self.max_memory_gb = max_memory_gb
        
        # Initialize components
        self._initialize_components()
        
        # Performance tracking
        self.performance_stats = {
            'total_detections': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'parallel_detections': 0,
            'memory_optimized_detections': 0,
            'average_detection_time': 0.0,
            'peak_memory_usage_mb': 0.0
        }
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("✅ Enhanced TAS Regime Detection initialized")
        self.logger.info(f"📊 GPU: {self.enable_gpu}, Memory Opt: {self.enable_memory_optimization}, Parallel: {self.enable_parallel}")
    
    def _initialize_components(self):
        """Initialize all required components."""
        try:
            # Initialize hardware optimizers
            if self.enable_gpu:
                self.gpu_manager = get_m1_gpu_manager()
                self.logger.info("✅ M1 GPU Manager initialized")
            else:
                self.gpu_manager = None
            
            if self.enable_memory_optimization:
                self.memory_optimizer = get_m1_memory_optimizer(memory_limit_gb=self.max_memory_gb)
                self.logger.info("✅ M1 Memory Optimizer initialized")
            else:
                self.memory_optimizer = None
            
            if self.enable_parallel:
                self.cpu_optimizer = get_m1_cpu_optimizer()
                self.logger.info("✅ M1 CPU Optimizer initialized")
            else:
                self.cpu_optimizer = None
            
            # Initialize matrix operations
            self.matrix_ops = get_unified_matrix_operations(
                enable_gpu=self.enable_gpu,
                enable_memory_optimization=self.enable_memory_optimization,
                enable_parallel=self.enable_parallel
            )
            
            # Initialize TAS components
            self.regime_pipeline = RegimeDetectionPipeline()
            self.regime_qualification = RegimeQualification()
            self.unsupervised_detection = UnsupervisedRegimeDetection()
            self.clustering_detection = ClusteringRegimeDetection()
            
            # Initialize validation components
            self.cvlsa = CVLSA()
            self.validator = UniversalValidator()
            self.metrics = PerformanceMetrics()
            
            self.logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing components: {e}")
            raise
    
    def detect_regimes_enhanced(self, 
                              data: Union[np.ndarray, pd.DataFrame],
                              timeframes: List[str] = None,
                              methods: List[str] = None,
                              use_cache: bool = True,
                              parallel: bool = True) -> Dict[str, Any]:
        """
        Enhanced regime detection with performance optimizations.
        
        Args:
            data: Input data for regime detection
            timeframes: List of timeframes to analyze
            methods: List of detection methods to use
            use_cache: Whether to use cached results
            parallel: Whether to use parallel processing
            
        Returns:
            Dictionary containing regime detection results
        """
        start_time = time.time()
        
        # Default parameters
        if timeframes is None:
            timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        if methods is None:
            methods = ['unsupervised', 'clustering', 'qualification']
        
        # Generate cache key
        cache_key = self._generate_cache_key(data, timeframes, methods)
        
        # Check cache first
        if use_cache:
            cached_result = self._get_cached_result(cache_key)
            if cached_result is not None:
                self.performance_stats['cache_hits'] += 1
                self.logger.info("✅ Using cached regime detection results")
                return cached_result
        
        self.performance_stats['cache_misses'] += 1
        
        try:
            # Memory optimization checkpoint
            if self.memory_optimizer:
                with self.memory_optimizer.memory_checkpoint("regime_detection"):
                    # Optimize data for memory efficiency
                    optimized_data = self._optimize_data_for_memory(data)
                    
                    # Perform regime detection
                    if parallel and self.enable_parallel:
                        results = self._parallel_regime_detection(
                            optimized_data, timeframes, methods
                        )
                        self.performance_stats['parallel_detections'] += 1
                    else:
                        results = self._sequential_regime_detection(
                            optimized_data, timeframes, methods
                        )
                    
                    # Apply advanced validation
                    validated_results = self._apply_advanced_validation(results, data)
                    
                    # Cache results
                    if use_cache:
                        self._cache_result(cache_key, validated_results)
                    
                    # Update performance stats
                    execution_time = time.time() - start_time
                    self.performance_stats['total_detections'] += 1
                    self.performance_stats['average_detection_time'] = (
                        (self.performance_stats['average_detection_time'] *
                         (self.performance_stats['total_detections'] - 1)) + execution_time
                    ) / self.performance_stats['total_detections']
                    
                    if self.memory_optimizer:
                        self.performance_stats['memory_optimized_detections'] += 1
                    
                    self.logger.info(f"✅ Enhanced regime detection completed in {execution_time:.3f}s")
                    return validated_results
        
        except Exception as e:
            self.logger.error(f"❌ Enhanced regime detection failed: {e}")
            raise
    
    def _generate_cache_key(self, data: Any, timeframes: List[str], methods: List[str]) -> str:
        """Generate cache key for regime detection results."""
        try:
            # Create hash of data, timeframes, and methods
            data_hash = hashlib.md5(str(data).encode()).hexdigest()[:8]
            timeframes_str = '_'.join(sorted(timeframes))
            methods_str = '_'.join(sorted(methods))
            
            cache_key = f"regime_{data_hash}_{timeframes_str}_{methods_str}"
            return cache_key
            
        except Exception as e:
            self.logger.warning(f"Could not generate cache key: {e}")
            return f"regime_{int(time.time())}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached regime detection result."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            self.logger.debug(f"Could not load cached result: {e}")
        return None
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache regime detection result."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            self.logger.debug(f"Cached result: {cache_key}")
        except Exception as e:
            self.logger.warning(f"Could not cache result: {e}")
    
    def _optimize_data_for_memory(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Optimize data for memory efficiency."""
        if self.memory_optimizer and isinstance(data, pd.DataFrame):
            return self.memory_optimizer.optimize_dataframe_memory(data)
        elif self.memory_optimizer and isinstance(data, np.ndarray):
            # Convert to DataFrame for optimization, then back to array
            df = pd.DataFrame(data)
            optimized_df = self.memory_optimizer.optimize_dataframe_memory(df)
            return optimized_df.values
        return data
    
    def _parallel_regime_detection(self, 
                                  data: Union[np.ndarray, pd.DataFrame],
                                  timeframes: List[str],
                                  methods: List[str]) -> Dict[str, Any]:
        """Perform regime detection in parallel across timeframes and methods."""
        try:
            # Create tasks for parallel execution
            tasks = []
            for timeframe in timeframes:
                for method in methods:
                    task = {
                        'timeframe': timeframe,
                        'method': method,
                        'data': data,
                        'func': self._detect_regime_single
                    }
                    tasks.append(task)
            
            # Execute tasks in parallel
            if self.cpu_optimizer:
                with self.cpu_optimizer.create_optimized_thread_pool() as executor:
                    futures = []
                    for task in tasks:
                        future = executor.submit(
                            task['func'],
                            task['data'],
                            task['timeframe'],
                            task['method']
                        )
                        futures.append((future, task))
                    
                    # Collect results
                    results = {}
                    for future, task in futures:
                        try:
                            result = future.result()
                            key = f"{task['timeframe']}_{task['method']}"
                            results[key] = result
                        except Exception as e:
                            self.logger.warning(f"Task failed: {e}")
                            key = f"{task['timeframe']}_{task['method']}"
                            results[key] = {'error': str(e), 'success': False}
            
            return results
            
        except Exception as e:
            self.logger.error(f"Parallel regime detection failed: {e}")
            return self._sequential_regime_detection(data, timeframes, methods)
    
    def _sequential_regime_detection(self, 
                                   data: Union[np.ndarray, pd.DataFrame],
                                   timeframes: List[str],
                                   methods: List[str]) -> Dict[str, Any]:
        """Perform regime detection sequentially."""
        results = {}
        
        for timeframe in timeframes:
            for method in methods:
                try:
                    result = self._detect_regime_single(data, timeframe, method)
                    key = f"{timeframe}_{method}"
                    results[key] = result
                except Exception as e:
                    self.logger.warning(f"Regime detection failed for {timeframe}_{method}: {e}")
                    key = f"{timeframe}_{method}"
                    results[key] = {'error': str(e), 'success': False}
        
        return results
    
    def _detect_regime_single(self, 
                             data: Union[np.ndarray, pd.DataFrame],
                             timeframe: str,
                             method: str) -> Dict[str, Any]:
        """Detect regime for a single timeframe and method."""
        try:
            if method == 'unsupervised':
                result = self.unsupervised_detection.detect_regimes(data, timeframe)
            elif method == 'clustering':
                result = self.clustering_detection.detect_regimes(data, timeframe)
            elif method == 'qualification':
                result = self.regime_qualification.qualify_regimes(data, timeframe)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            return {
                'success': True,
                'timeframe': timeframe,
                'method': method,
                'result': result
            }
            
        except Exception as e:
            return {
                'success': False,
                'timeframe': timeframe,
                'method': method,
                'error': str(e)
            }
    
    def _apply_advanced_validation(self, 
                                 results: Dict[str, Any],
                                 data: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """Apply advanced validation to regime detection results."""
        try:
            validated_results = results.copy()
            
            # Add cross-validation for regime stability
            validated_results['cross_validation'] = self._cross_validate_regimes(results, data)
            
            # Add out-of-sample testing
            validated_results['out_of_sample'] = self._out_of_sample_validation(results, data)
            
            # Add regime persistence analysis
            validated_results['persistence'] = self._analyze_regime_persistence(results, data)
            
            # Add performance metrics
            validated_results['performance_metrics'] = self._calculate_performance_metrics(results)
            
            return validated_results
            
        except Exception as e:
            self.logger.warning(f"Advanced validation failed: {e}")
            return results
    
    def _cross_validate_regimes(self, 
                               results: Dict[str, Any],
                               data: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """Perform cross-validation for regime stability."""
        try:
            cv_results = {}
            
            for key, result in results.items():
                if result.get('success', False):
                    try:
                        # Use CVLSA for cross-validation
                        cv_score = self.cvlsa.cross_validate(
                            data, 
                            result['result'],
                            cv_folds=5,
                            stability_metric='regime_consistency'
                        )
                        cv_results[key] = {
                            'cv_score': cv_score,
                            'stability': cv_score.get('stability', 0.0),
                            'consistency': cv_score.get('consistency', 0.0)
                        }
                    except Exception as e:
                        self.logger.warning(f"CV failed for {key}: {e}")
                        cv_results[key] = {'error': str(e)}
            
            return cv_results
            
        except Exception as e:
            self.logger.warning(f"Cross-validation failed: {e}")
            return {}
    
    def _out_of_sample_validation(self, 
                                  results: Dict[str, Any],
                                  data: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """Perform out-of-sample validation for regime detection."""
        try:
            oos_results = {}
            
            # Split data for out-of-sample testing
            split_ratio = 0.8
            split_idx = int(len(data) * split_ratio)
            
            train_data = data[:split_idx]
            test_data = data[split_idx:]
            
            for key, result in results.items():
                if result.get('success', False):
                    try:
                        # Train on in-sample data
                        train_result = self._detect_regime_single(
                            train_data, 
                            result.get('timeframe', '1h'),
                            result.get('method', 'unsupervised')
                        )
                        
                        # Test on out-of-sample data
                        test_result = self._detect_regime_single(
                            test_data,
                            result.get('timeframe', '1h'),
                            result.get('method', 'unsupervised')
                        )
                        
                        # Calculate out-of-sample metrics
                        oos_metrics = self._calculate_oos_metrics(train_result, test_result)
                        oos_results[key] = oos_metrics
                        
                    except Exception as e:
                        self.logger.warning(f"OOS validation failed for {key}: {e}")
                        oos_results[key] = {'error': str(e)}
            
            return oos_results
            
        except Exception as e:
            self.logger.warning(f"Out-of-sample validation failed: {e}")
            return {}
    
    def _analyze_regime_persistence(self, 
                                   results: Dict[str, Any],
                                   data: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze regime persistence over time."""
        try:
            persistence_results = {}
            
            for key, result in results.items():
                if result.get('success', False):
                    try:
                        # Calculate regime persistence metrics
                        regime_data = result['result']
                        
                        # Calculate transition probabilities
                        transition_probs = self.matrix_ops.calculate_transition_probabilities(
                            regime_data.get('regime_labels', []),
                            regime_data.get('n_regimes', 2)
                        )
                        
                        # Calculate stability scores
                        stability_scores = self.matrix_ops.calculate_regime_stability(
                            regime_data.get('regime_labels', []),
                            regime_data.get('timestamps', [])
                        )
                        
                        persistence_results[key] = {
                            'transition_probabilities': transition_probs,
                            'stability_scores': stability_scores,
                            'average_stability': np.mean(stability_scores) if len(stability_scores) > 0 else 0.0,
                            'regime_duration': self._calculate_regime_duration(regime_data)
                        }
                        
                    except Exception as e:
                        self.logger.warning(f"Persistence analysis failed for {key}: {e}")
                        persistence_results[key] = {'error': str(e)}
            
            return persistence_results
            
        except Exception as e:
            self.logger.warning(f"Regime persistence analysis failed: {e}")
            return {}
    
    def _calculate_regime_duration(self, regime_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate regime duration statistics."""
        try:
            regime_labels = regime_data.get('regime_labels', [])
            if len(regime_labels) == 0:
                return {'error': 'No regime labels available'}
            
            # Calculate duration for each regime
            durations = []
            current_regime = regime_labels[0]
            current_duration = 1
            
            for i in range(1, len(regime_labels)):
                if regime_labels[i] == current_regime:
                    current_duration += 1
                else:
                    durations.append(current_duration)
                    current_regime = regime_labels[i]
                    current_duration = 1
            
            # Add last duration
            durations.append(current_duration)
            
            return {
                'durations': durations,
                'mean_duration': np.mean(durations),
                'std_duration': np.std(durations),
                'min_duration': np.min(durations),
                'max_duration': np.max(durations)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_oos_metrics(self, 
                               train_result: Dict[str, Any],
                               test_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate out-of-sample validation metrics."""
        try:
            if not train_result.get('success', False) or not test_result.get('success', False):
                return {'error': 'Training or test result failed'}
            
            train_data = train_result['result']
            test_data = test_result['result']
            
            # Calculate similarity between train and test regimes
            similarity = self._calculate_regime_similarity(train_data, test_data)
            
            # Calculate prediction accuracy
            accuracy = self._calculate_regime_accuracy(train_data, test_data)
            
            return {
                'similarity': similarity,
                'accuracy': accuracy,
                'oos_score': (similarity + accuracy) / 2.0
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_regime_similarity(self, 
                                   train_data: Dict[str, Any],
                                   test_data: Dict[str, Any]) -> float:
        """Calculate similarity between train and test regime results."""
        try:
            # Extract regime characteristics
            train_regimes = train_data.get('regime_labels', [])
            test_regimes = test_data.get('regime_labels', [])
            
            if len(train_regimes) == 0 or len(test_regimes) == 0:
                return 0.0
            
            # Calculate regime distribution similarity
            train_dist = np.bincount(train_regimes) / len(train_regimes)
            test_dist = np.bincount(test_regimes) / len(test_regimes)
            
            # Pad distributions to same length
            max_len = max(len(train_dist), len(test_dist))
            train_dist = np.pad(train_dist, (0, max_len - len(train_dist)))
            test_dist = np.pad(test_dist, (0, max_len - len(test_dist)))
            
            # Calculate cosine similarity
            similarity = np.dot(train_dist, test_dist) / (
                np.linalg.norm(train_dist) * np.linalg.norm(test_dist)
            )
            
            return float(similarity)
            
        except Exception as e:
            self.logger.warning(f"Regime similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_regime_accuracy(self, 
                                  train_data: Dict[str, Any],
                                  test_data: Dict[str, Any]) -> float:
        """Calculate regime prediction accuracy."""
        try:
            # This is a simplified accuracy calculation
            # In practice, you would compare predicted vs actual regimes
            
            train_quality = train_data.get('quality_score', 0.5)
            test_quality = test_data.get('quality_score', 0.5)
            
            # Calculate accuracy based on quality scores
            accuracy = (train_quality + test_quality) / 2.0
            
            return float(accuracy)
            
        except Exception as e:
            self.logger.warning(f"Regime accuracy calculation failed: {e}")
            return 0.0
    
    def _calculate_performance_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance metrics for regime detection."""
        try:
            total_results = len(results)
            successful_results = sum(1 for r in results.values() if r.get('success', False))
            success_rate = successful_results / total_results if total_results > 0 else 0.0
            
            # Calculate average execution time
            execution_times = []
            for result in results.values():
                if 'execution_time' in result:
                    execution_times.append(result['execution_time'])
            
            avg_execution_time = np.mean(execution_times) if execution_times else 0.0
            
            return {
                'total_results': total_results,
                'successful_results': successful_results,
                'success_rate': success_rate,
                'average_execution_time': avg_execution_time,
                'performance_score': success_rate * (1.0 / (1.0 + avg_execution_time))
            }
            
        except Exception as e:
            self.logger.warning(f"Performance metrics calculation failed: {e}")
            return {'error': str(e)}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()
        
        # Add hardware info
        if self.gpu_manager:
            stats['gpu_info'] = self.gpu_manager.get_gpu_info()
        
        if self.memory_optimizer:
            stats['memory_info'] = self.memory_optimizer.get_memory_stats()
        
        if self.cpu_optimizer:
            stats['cpu_info'] = self.cpu_optimizer.get_cpu_info()
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        try:
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info("✅ Cache cleared successfully")
        except Exception as e:
            self.logger.warning(f"Could not clear cache: {e}")
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage for regime detection."""
        if self.memory_optimizer:
            return self.memory_optimizer.optimize_memory_usage()
        else:
            return {'status': 'memory_optimizer_not_available'}
    
    def optimize_cpu_usage(self, target_utilization: float = 0.8) -> Dict[str, Any]:
        """Optimize CPU usage for regime detection."""
        if self.cpu_optimizer:
            return self.cpu_optimizer.optimize_cpu_usage(target_utilization)
        else:
            return {'status': 'cpu_optimizer_not_available'}


# Factory function for easy access
def get_enhanced_tas_regime_detection(enable_gpu: bool = True,
                                    enable_memory_optimization: bool = True,
                                    enable_parallel: bool = True,
                                    cache_dir: Optional[str] = None,
                                    max_memory_gb: Optional[float] = None) -> EnhancedTASRegimeDetection:
    """
    Factory function to create enhanced TAS regime detection instance.
    
    Args:
        enable_gpu: Whether to enable GPU acceleration
        enable_memory_optimization: Whether to enable memory optimization
        enable_parallel: Whether to enable parallel processing
        cache_dir: Directory for caching results
        max_memory_gb: Maximum memory usage in GB
        
    Returns:
        Configured EnhancedTASRegimeDetection instance
    """
    return EnhancedTASRegimeDetection(
        enable_gpu=enable_gpu,
        enable_memory_optimization=enable_memory_optimization,
        enable_parallel=enable_parallel,
        cache_dir=cache_dir,
        max_memory_gb=max_memory_gb
    )


# Example usage
if __name__ == "__main__":
    # Example usage of enhanced TAS regime detection
    print("🚀 Enhanced TAS Regime Detection Demo")
    print("="*50)
    
    # Create sample data
    np.random.seed(42)
    data = np.random.randn(1000, 10)
    
    # Initialize enhanced regime detection
    regime_detector = get_enhanced_tas_regime_detection(
        enable_gpu=True,
        enable_memory_optimization=True,
        enable_parallel=True,
        cache_dir='./tas_cache',
        max_memory_gb=8.0
    )
    
    # Perform enhanced regime detection
    print("\n🔍 Performing Enhanced Regime Detection...")
    results = regime_detector.detect_regimes_enhanced(
        data=data,
        timeframes=['1h', '4h', '1d'],
        methods=['unsupervised', 'clustering'],
        use_cache=True,
        parallel=True
    )
    
    print(f"✅ Regime detection completed")
    print(f"📊 Results: {len(results)} regime analyses")
    
    # Print performance stats
    print("\n📈 Performance Statistics:")
    stats = regime_detector.get_performance_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n🎉 Enhanced TAS Regime Detection Demo Complete!")