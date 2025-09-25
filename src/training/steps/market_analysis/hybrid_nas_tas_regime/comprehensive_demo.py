"""
Comprehensive Demo of Unified Regime Detection System

This script demonstrates all the features of the unified regime detection system
including performance optimization, real-time monitoring, and benchmarking.
"""

import numpy as np
import pandas as pd
import time
import asyncio
from typing import Dict, List, Any
from datetime import datetime, timedelta

# Import tprint for visual output
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

# Import unified regime detection system
from src.utils.nas_tas import (
    UnifiedRegimeDetector, UnifiedRegimeConfig, RegimeDetectionMethod,
    OptimizationStrategy, EconomicEvaluationMode,
    PerformanceOptimizer, get_performance_optimizer,
    RealTimeRegimeMonitor, RegimeChangeEvent, create_real_time_monitor
)

# Import benchmark suite
try:
    from .performance_benchmark import PerformanceBenchmark
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False

class ComprehensiveDemo:
    """Comprehensive demonstration of the unified regime detection system."""
    
    def __init__(self):
        """Initialize the comprehensive demo."""
        self.demo_results = {}
        
        tprint("🚀 COMPREHENSIVE REGIME DETECTION DEMO", color="cyan", bold=True)
        tprint("=" * 60, color="cyan")
    
    def generate_demo_data(self, n_samples: int = 2000) -> pd.DataFrame:
        """Generate realistic market data for demonstration."""
        tprint_info("📊 Generating demo market data...")
        
        np.random.seed(42)
        
        # Create realistic market data with multiple regimes
        data = []
        base_price = 100.0
        current_price = base_price
        
        for i in range(n_samples):
            # Simulate different market regimes
            if i < n_samples // 4:
                # Bull market regime
                trend = 0.0005
                volatility = 0.015
                regime = 0
            elif i < n_samples // 2:
                # Bear market regime
                trend = -0.0003
                volatility = 0.020
                regime = 1
            elif i < 3 * n_samples // 4:
                # Sideways market regime
                trend = 0.0
                volatility = 0.008
                regime = 2
            else:
                # High volatility regime
                trend = 0.0001
                volatility = 0.035
                regime = 3
            
            # Generate price movement
            price_change = np.random.normal(trend, volatility)
            current_price *= (1 + price_change)
            
            # Create OHLCV data
            open_price = current_price
            close_price = current_price * (1 + np.random.normal(0, volatility/2))
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, volatility/4)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, volatility/4)))
            volume = np.random.lognormal(10, 0.5)
            
            data.append({
                'timestamp': datetime.now() - timedelta(minutes=n_samples-i),
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
                'true_regime': regime
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        tprint_success(f"✅ Generated {len(df)} samples with 4 distinct regimes")
        return df
    
    def demo_basic_functionality(self):
        """Demonstrate basic unified regime detection functionality."""
        tprint("\n🔬 BASIC FUNCTIONALITY DEMO", color="blue", bold=True)
        tprint("-" * 40, color="blue")
        
        # Generate demo data
        demo_data = self.generate_demo_data(1000)
        market_data = demo_data[['open', 'high', 'low', 'close', 'volume']]
        timestamps = demo_data.index.values
        
        # Test different detection methods
        methods = [
            RegimeDetectionMethod.TAS_ONLY,
            RegimeDetectionMethod.NAS_ONLY,
            RegimeDetectionMethod.HYBRID,
            RegimeDetectionMethod.ADAPTIVE
        ]
        
        results = {}
        
        for method in methods:
            tprint_info(f"🧠 Testing {method.value} detection...")
            
            try:
                # Create configuration
                config = UnifiedRegimeConfig(
                    detection_method=method,
                    n_regimes=4,
                    primary_timeframe="15m",
                    optimization_strategy=OptimizationStrategy.BALANCED,
                    economic_evaluation=EconomicEvaluationMode.ADVANCED
                )
                
                # Initialize detector
                detector = UnifiedRegimeDetector(config)
                
                # Detect regimes
                start_time = time.time()
                result = detector.detect_regimes(market_data, timestamps)
                execution_time = time.time() - start_time
                
                if result.success:
                    unique_regimes = len(np.unique(result.regime_predictions))
                    avg_confidence = np.mean([np.max(probs) for probs in result.regime_probabilities])
                    avg_economic = np.mean(result.economic_significance_scores)
                    avg_trading = np.mean(result.trading_viability_scores)
                    
                    results[method.value] = {
                        'execution_time': execution_time,
                        'regimes_detected': unique_regimes,
                        'avg_confidence': avg_confidence,
                        'avg_economic_significance': avg_economic,
                        'avg_trading_viability': avg_trading,
                        'success': True
                    }
                    
                    tprint_success(f"✅ {method.value}: {execution_time:.3f}s, {unique_regimes} regimes, "
                                  f"confidence: {avg_confidence:.3f}")
                else:
                    results[method.value] = {
                        'execution_time': execution_time,
                        'success': False,
                        'error': result.error_message
                    }
                    tprint_error(f"❌ {method.value} failed: {result.error_message}")
                    
            except Exception as e:
                tprint_error(f"❌ {method.value} error: {e}")
                results[method.value] = {'success': False, 'error': str(e)}
        
        self.demo_results['basic_functionality'] = results
        return results
    
    def demo_performance_optimization(self):
        """Demonstrate performance optimization features."""
        tprint("\n⚡ PERFORMANCE OPTIMIZATION DEMO", color="blue", bold=True)
        tprint("-" * 40, color="blue")
        
        # Generate larger dataset for performance testing
        demo_data = self.generate_demo_data(2000)
        market_data = demo_data[['open', 'high', 'low', 'close', 'volume']]
        timestamps = demo_data.index.values
        
        # Test with and without optimization
        config = UnifiedRegimeConfig(
            detection_method=RegimeDetectionMethod.HYBRID,
            n_regimes=4,
            enable_hardware_optimization=True,
            enable_gpu_acceleration=True,
            enable_memory_optimization=True
        )
        
        detector = UnifiedRegimeDetector(config)
        
        # Test performance optimizer
        optimizer = get_performance_optimizer()
        
        tprint_info("📊 Testing performance optimization...")
        
        # Multiple runs to get average performance
        execution_times = []
        memory_usage = []
        
        for i in range(3):
            tprint_debug(f"   Run {i+1}/3")
            
            start_time = time.time()
            result = detector.detect_regimes(market_data, timestamps)
            execution_time = time.time() - start_time
            
            execution_times.append(execution_time)
            
            # Get memory usage (simplified)
            try:
                import psutil
                process = psutil.Process()
                memory_usage.append(process.memory_info().rss / 1024**2)  # MB
            except ImportError:
                memory_usage.append(0)
        
        avg_time = np.mean(execution_times)
        avg_memory = np.mean(memory_usage)
        
        # Get optimizer statistics
        optimizer_stats = optimizer.get_performance_stats()
        
        results = {
            'average_execution_time': avg_time,
            'average_memory_usage_mb': avg_memory,
            'optimizer_stats': optimizer_stats,
            'execution_times': execution_times,
            'memory_usage': memory_usage
        }
        
        tprint_success(f"✅ Performance optimization: {avg_time:.3f}s avg, {avg_memory:.1f}MB avg")
        tprint_info(f"📊 Optimizer stats: {optimizer_stats}")
        
        self.demo_results['performance_optimization'] = results
        return results
    
    def demo_real_time_monitoring(self):
        """Demonstrate real-time monitoring capabilities."""
        tprint("\n🔄 REAL-TIME MONITORING DEMO", color="blue", bold=True)
        tprint("-" * 40, color="blue")
        
        # Create real-time monitor
        config = UnifiedRegimeConfig.create_production_config()
        monitor = create_real_time_monitor(config)
        
        # Add event callback
        def regime_change_callback(event: RegimeChangeEvent):
            tprint(f"🔄 Regime Change: {event.from_regime} → {event.to_regime} "
                  f"(confidence: {event.confidence:.3f})", color="yellow")
        
        monitor.add_event_callback(regime_change_callback)
        
        # Start monitoring
        monitor.start_monitoring()
        
        tprint_info("🚀 Real-time monitoring started. Simulating market data...")
        
        # Simulate real-time data for 30 seconds
        start_time = time.time()
        data_count = 0
        
        while time.time() - start_time < 30:
            # Generate realistic market data
            price = 100 + np.sin(data_count * 0.1) * 5 + np.random.normal(0, 1)
            volume = 1000 + np.random.exponential(500)
            
            data_point = {
                'open': price,
                'high': price + abs(np.random.normal(0, 0.5)),
                'low': price - abs(np.random.normal(0, 0.5)),
                'close': price + np.random.normal(0, 0.3),
                'volume': volume
            }
            
            monitor.add_market_data(data_point)
            data_count += 1
            
            # Print status every 5 seconds
            if int(time.time() - start_time) % 5 == 0:
                status = monitor.get_current_status()
                tprint(f"📊 Status: Regime {status['current_regime']}, "
                      f"Events: {status['total_events']}, "
                      f"Queue: {status['data_queue_size']}")
            
            time.sleep(0.1)  # 100ms intervals
        
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Get final results
        final_status = monitor.get_current_status()
        regime_history = monitor.get_regime_history(10)
        change_events = monitor.get_change_events(5)
        
        results = {
            'total_events': final_status['total_events'],
            'final_regime': final_status['current_regime'],
            'data_processed': data_count,
            'performance_summary': final_status['performance_summary'],
            'regime_history': len(regime_history),
            'change_events': len(change_events)
        }
        
        tprint_success(f"✅ Real-time monitoring: {data_count} data points, "
                      f"{final_status['total_events']} regime changes")
        
        self.demo_results['real_time_monitoring'] = results
        return results
    
    def demo_benchmarking(self):
        """Demonstrate performance benchmarking."""
        if not BENCHMARK_AVAILABLE:
            tprint_warning("⚠️ Benchmark suite not available, skipping benchmarking demo")
            return {}
        
        tprint("\n📊 BENCHMARKING DEMO", color="blue", bold=True)
        tprint("-" * 40, color="blue")
        
        # Create benchmark suite
        benchmark = PerformanceBenchmark("demo_benchmark_results")
        
        # Run quick benchmark
        tprint_info("🏃 Running performance benchmark...")
        
        try:
            results = benchmark.run_comprehensive_benchmark(
                data_sizes=[500, 1000],  # Smaller sizes for demo
                iterations=2  # Fewer iterations for demo
            )
            
            # Print summary
            benchmark.print_benchmark_summary(results)
            
            self.demo_results['benchmarking'] = results
            return results
            
        except Exception as e:
            tprint_error(f"❌ Benchmarking failed: {e}")
            return {'error': str(e)}
    
    def demo_configuration_options(self):
        """Demonstrate different configuration options."""
        tprint("\n⚙️ CONFIGURATION OPTIONS DEMO", color="blue", bold=True)
        tprint("-" * 40, color="blue")
        
        # Generate small dataset for quick testing
        demo_data = self.generate_demo_data(500)
        market_data = demo_data[['open', 'high', 'low', 'close', 'volume']]
        timestamps = demo_data.index.values
        
        configurations = [
            {
                'name': 'Production Config',
                'config': UnifiedRegimeConfig.create_production_config()
            },
            {
                'name': 'Development Config',
                'config': UnifiedRegimeConfig.create_development_config()
            },
            {
                'name': 'Performance Config',
                'config': UnifiedRegimeConfig.create_performance_config()
            },
            {
                'name': 'Accuracy Config',
                'config': UnifiedRegimeConfig.create_accuracy_config()
            },
            {
                'name': 'Economic Config',
                'config': UnifiedRegimeConfig.create_economic_config()
            }
        ]
        
        results = {}
        
        for config_info in configurations:
            name = config_info['name']
            config = config_info['config']
            
            tprint_info(f"⚙️ Testing {name}...")
            
            try:
                detector = UnifiedRegimeDetector(config)
                
                start_time = time.time()
                result = detector.detect_regimes(market_data, timestamps)
                execution_time = time.time() - start_time
                
                if result.success:
                    results[name] = {
                        'execution_time': execution_time,
                        'detection_method': config.detection_method.value,
                        'optimization_strategy': config.optimization_strategy.value,
                        'economic_evaluation': config.economic_evaluation.value,
                        'regimes_detected': len(np.unique(result.regime_predictions)),
                        'avg_economic_significance': np.mean(result.economic_significance_scores),
                        'success': True
                    }
                    
                    tprint_success(f"✅ {name}: {execution_time:.3f}s, "
                                  f"{config.detection_method.value}")
                else:
                    results[name] = {
                        'execution_time': execution_time,
                        'success': False,
                        'error': result.error_message
                    }
                    tprint_error(f"❌ {name} failed: {result.error_message}")
                    
            except Exception as e:
                tprint_error(f"❌ {name} error: {e}")
                results[name] = {'success': False, 'error': str(e)}
        
        self.demo_results['configuration_options'] = results
        return results
    
    def run_comprehensive_demo(self):
        """Run the complete comprehensive demonstration."""
        tprint("🚀 STARTING COMPREHENSIVE DEMONSTRATION", color="cyan", bold=True)
        tprint("=" * 60, color="cyan")
        
        start_time = time.time()
        
        try:
            # Run all demo components
            self.demo_basic_functionality()
            self.demo_performance_optimization()
            self.demo_real_time_monitoring()
            self.demo_benchmarking()
            self.demo_configuration_options()
            
            total_time = time.time() - start_time
            
            # Print final summary
            self.print_demo_summary(total_time)
            
        except Exception as e:
            tprint_error(f"❌ Demo failed: {e}")
        
        finally:
            tprint("\n✅ COMPREHENSIVE DEMO COMPLETED", color="green", bold=True)
            tprint("=" * 60, color="green")
    
    def print_demo_summary(self, total_time: float):
        """Print comprehensive demo summary."""
        tprint("\n📊 DEMO SUMMARY", color="cyan", bold=True)
        tprint("=" * 40, color="cyan")
        
        tprint(f"⏱️ Total Demo Time: {total_time:.2f} seconds", color="white")
        
        for demo_name, results in self.demo_results.items():
            tprint(f"\n📋 {demo_name.upper().replace('_', ' ')}:", color="blue")
            
            if demo_name == 'basic_functionality':
                successful_methods = [k for k, v in results.items() if v.get('success', False)]
                tprint(f"   ✅ Successful methods: {', '.join(successful_methods)}", color="green")
                
                if successful_methods:
                    fastest = min(successful_methods, 
                                key=lambda x: results[x].get('execution_time', float('inf')))
                    tprint(f"   ⚡ Fastest: {fastest}", color="yellow")
            
            elif demo_name == 'performance_optimization':
                avg_time = results.get('average_execution_time', 0)
                avg_memory = results.get('average_memory_usage_mb', 0)
                tprint(f"   ⚡ Average execution time: {avg_time:.3f}s", color="white")
                tprint(f"   💾 Average memory usage: {avg_memory:.1f}MB", color="white")
            
            elif demo_name == 'real_time_monitoring':
                total_events = results.get('total_events', 0)
                data_processed = results.get('data_processed', 0)
                tprint(f"   📊 Data points processed: {data_processed}", color="white")
                tprint(f"   🔄 Regime changes detected: {total_events}", color="white")
            
            elif demo_name == 'configuration_options':
                successful_configs = [k for k, v in results.items() if v.get('success', False)]
                tprint(f"   ✅ Successful configurations: {len(successful_configs)}/{len(results)}", color="green")

def main():
    """Main demo execution function."""
    demo = ComprehensiveDemo()
    demo.run_comprehensive_demo()

if __name__ == "__main__":
    # Setup logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run comprehensive demo
    main()