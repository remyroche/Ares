"""
Performance Benchmark Suite for Regime Detection Systems

This module provides comprehensive performance benchmarking for all regime detection
systems including unified, TAS, NAS, and hybrid approaches.
"""

import numpy as np
import pandas as pd
import time
import psutil
import gc
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

# Import tprint for logging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

# Import all regime detection systems
try:
    from src.utils.ml_common.nas_tas_unified import (
        UnifiedRegimeDetector, UnifiedRegimeConfig, 
        PerformanceOptimizer, get_performance_optimizer
    )
    UNIFIED_AVAILABLE = True
except ImportError:
    UNIFIED_AVAILABLE = False

try:
    from src.training.steps.market_analysis.tas_regime.core.tas_regime_detector import TASRegimeDetector
    from src.training.steps.market_analysis.tas_regime.core.tas_regime_config import TASRegimeConfig
    TAS_AVAILABLE = True
except ImportError:
    TAS_AVAILABLE = False

try:
    from src.training.steps.market_analysis.nas_regime.core.perfect_nas_regime_detector import PerfectNASRegimeDetector
    from src.training.steps.market_analysis.nas_regime.core.perfect_nas_config import PerfectNASConfig
    NAS_AVAILABLE = True
except ImportError:
    NAS_AVAILABLE = False

try:
    from src.training.steps.market_analysis.hybrid_nas_tas_regime.unified_regime_integration import UnifiedRegimeIntegration
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False

logger = logging.getLogger(__name__)

class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite for regime detection systems."""
    
    def __init__(self, results_dir: str = "benchmark_results"):
        """Initialize performance benchmark suite."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.benchmark_results = {}
        self.system_availability = {
            'unified': UNIFIED_AVAILABLE,
            'tas': TAS_AVAILABLE,
            'nas': NAS_AVAILABLE,
            'hybrid': HYBRID_AVAILABLE
        }
        
        tprint_info("📊 Performance Benchmark Suite initialized")
        logger.info("Performance Benchmark Suite initialized")
    
    def generate_test_data(self, n_samples: int = 1000, n_features: int = 5, 
                          data_type: str = "realistic") -> pd.DataFrame:
        """Generate test data for benchmarking."""
        np.random.seed(42)
        
        if data_type == "realistic":
            # Generate realistic OHLCV data
            base_price = 100.0
            prices = [base_price]
            
            for i in range(n_samples - 1):
                # Random walk with some trend and volatility clustering
                volatility = 0.02 + 0.01 * np.random.random()
                change = np.random.normal(0, volatility) + 0.0001
                new_price = prices[-1] * (1 + change)
                prices.append(max(new_price, 0.01))
            
            # Create OHLCV data
            data = []
            for i, close in enumerate(prices):
                open_price = prices[i-1] if i > 0 else close
                high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.01)))
                low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.01)))
                volume = np.random.lognormal(10, 0.5)
                
                data.append([open_price, high, low, close, volume])
            
            df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])
            
        elif data_type == "synthetic":
            # Generate synthetic data with known regimes
            regime_length = n_samples // 4
            data = []
            
            for regime in range(4):
                if regime == 0:  # Bull market
                    trend = 0.001
                    volatility = 0.015
                elif regime == 1:  # Bear market
                    trend = -0.001
                    volatility = 0.025
                elif regime == 2:  # Sideways market
                    trend = 0.0
                    volatility = 0.01
                else:  # High volatility
                    trend = 0.0
                    volatility = 0.05
                
                regime_data = np.random.normal(trend, volatility, regime_length)
                data.extend(regime_data)
            
            # Convert to OHLCV format
            prices = np.cumsum(data) + 100
            df = pd.DataFrame({
                'open': prices,
                'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(prices)))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(prices)))),
                'close': prices,
                'volume': np.random.lognormal(10, 0.5, len(prices))
            })
        
        else:  # Random data
            df = pd.DataFrame(np.random.randn(n_samples, n_features))
            df.columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Add timestamps
        df['timestamp'] = pd.date_range('2023-01-01', periods=len(df), freq='15T')
        
        return df
    
    def measure_memory_usage(self, func: callable, *args, **kwargs) -> Dict[str, float]:
        """Measure memory usage during function execution."""
        process = psutil.Process()
        
        # Measure memory before
        memory_before = process.memory_info().rss / 1024**3  # GB
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Measure memory after
        memory_after = process.memory_info().rss / 1024**3  # GB
        
        # Force garbage collection
        gc.collect()
        
        # Measure memory after GC
        memory_after_gc = process.memory_info().rss / 1024**3  # GB
        
        return {
            'memory_before_gb': memory_before,
            'memory_after_gb': memory_after,
            'memory_peak_gb': memory_after,
            'memory_after_gc_gb': memory_after_gc,
            'memory_delta_gb': memory_after - memory_before,
            'memory_cleanup_gb': memory_after - memory_after_gc
        }
    
    def benchmark_system(self, system_name: str, detector, test_data: pd.DataFrame, 
                        timestamps: Optional[np.ndarray] = None, 
                        iterations: int = 3) -> Dict[str, Any]:
        """Benchmark a single regime detection system."""
        tprint_info(f"🔬 Benchmarking {system_name.upper()} system")
        
        execution_times = []
        memory_usage_stats = []
        success_count = 0
        results = []
        
        for i in range(iterations):
            tprint_debug(f"   Iteration {i+1}/{iterations}")
            
            try:
                # Measure execution time
                start_time = time.time()
                
                # Measure memory usage
                memory_stats = self.measure_memory_usage(
                    detector.detect_regimes, test_data, timestamps
                )
                
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
                memory_usage_stats.append(memory_stats)
                success_count += 1
                
                # Store result for analysis
                result = detector.detect_regimes(test_data, timestamps)
                results.append(result)
                
                tprint_debug(f"   ✅ Iteration {i+1}: {execution_time:.3f}s, {memory_stats['memory_delta_gb']:+.2f}GB")
                
            except Exception as e:
                tprint_error(f"   ❌ Iteration {i+1} failed: {e}")
                execution_times.append(float('inf'))
                memory_usage_stats.append({
                    'memory_before_gb': 0,
                    'memory_after_gb': 0,
                    'memory_delta_gb': 0,
                    'memory_cleanup_gb': 0
                })
        
        # Calculate statistics
        valid_times = [t for t in execution_times if t != float('inf')]
        
        if valid_times:
            avg_time = np.mean(valid_times)
            min_time = np.min(valid_times)
            max_time = np.max(valid_times)
            std_time = np.std(valid_times)
        else:
            avg_time = min_time = max_time = std_time = float('inf')
        
        # Calculate memory statistics
        avg_memory_delta = np.mean([s['memory_delta_gb'] for s in memory_usage_stats])
        avg_memory_cleanup = np.mean([s['memory_cleanup_gb'] for s in memory_usage_stats])
        
        benchmark_result = {
            'system_name': system_name,
            'iterations': iterations,
            'success_count': success_count,
            'success_rate': success_count / iterations,
            'execution_time': {
                'average_s': avg_time,
                'minimum_s': min_time,
                'maximum_s': max_time,
                'std_dev_s': std_time
            },
            'memory_usage': {
                'average_delta_gb': avg_memory_delta,
                'average_cleanup_gb': avg_memory_cleanup
            },
            'performance_score': self._calculate_performance_score(
                avg_time, avg_memory_delta, success_count / iterations
            )
        }
        
        # Add result analysis if successful
        if results and results[0].success:
            result = results[0]
            benchmark_result['result_analysis'] = {
                'regimes_detected': len(np.unique(result.regime_predictions)),
                'avg_economic_significance': float(np.mean(result.economic_significance_scores)),
                'avg_trading_viability': float(np.mean(result.trading_viability_scores)),
                'avg_regime_stability': float(np.mean(result.regime_stability_scores))
            }
        
        tprint_success(f"✅ {system_name.upper()} benchmark completed: {avg_time:.3f}s avg, {success_count}/{iterations} successful")
        
        return benchmark_result
    
    def _calculate_performance_score(self, avg_time: float, memory_delta: float, success_rate: float) -> float:
        """Calculate overall performance score."""
        if avg_time == float('inf') or success_rate == 0:
            return 0.0
        
        # Normalize metrics (lower is better for time and memory)
        time_score = max(0, 1 - (avg_time / 60))  # Normalize to 60s
        memory_score = max(0, 1 - (memory_delta / 2))  # Normalize to 2GB
        success_score = success_rate
        
        # Weighted combination
        return (0.4 * time_score + 0.3 * memory_score + 0.3 * success_score)
    
    def run_comprehensive_benchmark(self, data_sizes: List[int] = [500, 1000, 2000], 
                                   iterations: int = 3) -> Dict[str, Any]:
        """Run comprehensive benchmark across all systems and data sizes."""
        tprint_info("🚀 Starting comprehensive performance benchmark")
        
        all_results = {}
        
        for data_size in data_sizes:
            tprint_info(f"📊 Testing with {data_size} samples")
            
            # Generate test data
            test_data = self.generate_test_data(data_size)
            timestamps = test_data['timestamp'].values
            
            size_results = {}
            
            # Benchmark Unified System
            if self.system_availability['unified']:
                try:
                    config = UnifiedRegimeConfig.create_production_config()
                    detector = UnifiedRegimeDetector(config)
                    result = self.benchmark_system('unified', detector, test_data, timestamps, iterations)
                    size_results['unified'] = result
                except Exception as e:
                    tprint_error(f"❌ Unified system benchmark failed: {e}")
            
            # Benchmark TAS System
            if self.system_availability['tas']:
                try:
                    config = TASRegimeConfig.create_production_config()
                    detector = TASRegimeDetector(config)
                    result = self.benchmark_system('tas', detector, test_data, timestamps, iterations)
                    size_results['tas'] = result
                except Exception as e:
                    tprint_error(f"❌ TAS system benchmark failed: {e}")
            
            # Benchmark NAS System
            if self.system_availability['nas']:
                try:
                    config = PerfectNASConfig.create_production_config()
                    detector = PerfectNASRegimeDetector(config)
                    result = self.benchmark_system('nas', detector, test_data, timestamps, iterations)
                    size_results['nas'] = result
                except Exception as e:
                    tprint_error(f"❌ NAS system benchmark failed: {e}")
            
            # Benchmark Hybrid Integration
            if self.system_availability['hybrid']:
                try:
                    integration = UnifiedRegimeIntegration()
                    result = self.benchmark_system('hybrid', integration, test_data, timestamps, iterations)
                    size_results['hybrid'] = result
                except Exception as e:
                    tprint_error(f"❌ Hybrid system benchmark failed: {e}")
            
            all_results[f'data_size_{data_size}'] = size_results
        
        # Generate summary
        summary = self._generate_benchmark_summary(all_results)
        
        # Save results
        self._save_benchmark_results(all_results, summary)
        
        return {
            'detailed_results': all_results,
            'summary': summary
        }
    
    def _generate_benchmark_summary(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate benchmark summary with rankings."""
        summary = {
            'overall_rankings': {},
            'performance_by_system': {},
            'performance_by_data_size': {},
            'recommendations': []
        }
        
        # Collect all system results
        system_performance = {}
        
        for size_key, size_results in all_results.items():
            for system_name, result in size_results.items():
                if system_name not in system_performance:
                    system_performance[system_name] = []
                
                system_performance[system_name].append({
                    'data_size': size_key,
                    'performance_score': result['performance_score'],
                    'avg_execution_time': result['execution_time']['average_s'],
                    'success_rate': result['success_rate']
                })
        
        # Calculate average performance scores
        avg_scores = {}
        for system_name, results in system_performance.items():
            scores = [r['performance_score'] for r in results]
            avg_scores[system_name] = np.mean(scores)
        
        # Rank systems
        sorted_systems = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        summary['overall_rankings'] = {rank+1: system for rank, system in enumerate(sorted_systems)}
        
        # Performance by system
        for system_name, results in system_performance.items():
            summary['performance_by_system'][system_name] = {
                'average_score': avg_scores[system_name],
                'best_data_size': max(results, key=lambda x: x['performance_score'])['data_size'],
                'consistency': 1 - np.std([r['performance_score'] for r in results])
            }
        
        # Generate recommendations
        best_system = sorted_systems[0][0] if sorted_systems else None
        if best_system:
            summary['recommendations'].append(f"Best overall performance: {best_system.upper()}")
        
        fastest_system = min(system_performance.items(), 
                           key=lambda x: np.mean([r['avg_execution_time'] for r in x[1]]))[0]
        summary['recommendations'].append(f"Fastest execution: {fastest_system.upper()}")
        
        most_reliable = max(system_performance.items(),
                          key=lambda x: np.mean([r['success_rate'] for r in x[1]]))[0]
        summary['recommendations'].append(f"Most reliable: {most_reliable.upper()}")
        
        return summary
    
    def _save_benchmark_results(self, results: Dict[str, Any], summary: Dict[str, Any]):
        """Save benchmark results to file."""
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"benchmark_results_{timestamp}.json"
        
        full_results = {
            'timestamp': timestamp,
            'system_availability': self.system_availability,
            'detailed_results': results,
            'summary': summary
        }
        
        with open(results_file, 'w') as f:
            json.dump(full_results, f, indent=2, default=str)
        
        tprint_success(f"📊 Benchmark results saved to {results_file}")
    
    def print_benchmark_summary(self, results: Dict[str, Any]):
        """Print formatted benchmark summary."""
        summary = results['summary']
        
        tprint("📊 PERFORMANCE BENCHMARK SUMMARY", color="cyan", bold=True)
        tprint("=" * 50, color="cyan")
        
        # Overall rankings
        tprint("\n🏆 OVERALL RANKINGS:", color="blue", bold=True)
        for rank, (system, score) in summary['overall_rankings'].items():
            tprint(f"   {rank}. {system.upper()}: {score:.3f}", color="green" if rank == 1 else "white")
        
        # Performance by system
        tprint("\n📈 PERFORMANCE BY SYSTEM:", color="blue", bold=True)
        for system, perf in summary['performance_by_system'].items():
            tprint(f"   {system.upper()}:", color="white")
            tprint(f"     Average Score: {perf['average_score']:.3f}", color="white")
            tprint(f"     Best Data Size: {perf['best_data_size']}", color="white")
            tprint(f"     Consistency: {perf['consistency']:.3f}", color="white")
        
        # Recommendations
        tprint("\n💡 RECOMMENDATIONS:", color="blue", bold=True)
        for rec in summary['recommendations']:
            tprint(f"   • {rec}", color="yellow")

def main():
    """Main benchmark execution function."""
    tprint("🚀 REGIME DETECTION PERFORMANCE BENCHMARK", color="cyan", bold=True)
    tprint("=" * 60, color="cyan")
    
    # Initialize benchmark suite
    benchmark = PerformanceBenchmark()
    
    # Check system availability
    tprint("\n🔍 SYSTEM AVAILABILITY:", color="blue", bold=True)
    for system, available in benchmark.system_availability.items():
        status = "✅ Available" if available else "❌ Not Available"
        color = "green" if available else "red"
        tprint(f"   {system.upper()}: {status}", color=color)
    
    # Run comprehensive benchmark
    tprint("\n🏃 RUNNING COMPREHENSIVE BENCHMARK...", color="blue", bold=True)
    results = benchmark.run_comprehensive_benchmark(
        data_sizes=[500, 1000, 2000],
        iterations=3
    )
    
    # Print summary
    benchmark.print_benchmark_summary(results)
    
    tprint("\n✅ BENCHMARK COMPLETED SUCCESSFULLY", color="green", bold=True)
    tprint("=" * 60, color="green")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run benchmark
    main()