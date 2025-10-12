"""
Test script for Enhanced Data-Driven Interaction Generator

This script demonstrates the improvements and performance gains achieved
by integrating VectorBT utilities into the DataDrivenInteractionGenerator.
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import List, Dict, Any
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the enhanced generator
try:
    from src.feature_generation.utils.enhanced_data_driven_interaction_generator import (
        EnhancedDataDrivenInteractionGenerator, 
        EnhancedInteractionConfig,
        InteractionResult
    )
    ENHANCED_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Enhanced generator not available: {e}")
    ENHANCED_AVAILABLE = False

# Import the original generator for comparison
try:
    from src.feature_generation.utils.data_driven_interaction_generator import (
        DataDrivenInteractionGenerator
    )
    ORIGINAL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Original generator not available: {e}")
    ORIGINAL_AVAILABLE = False


def create_test_data(n_samples: int = 5000, n_features: int = 10) -> pd.DataFrame:
    """Create test data for interaction generation."""
    np.random.seed(42)
    
    # Generate realistic financial data
    returns = np.random.normal(0.001, 0.02, n_samples)
    prices = 100 * (1 + returns).cumprod()
    
    data = {
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples))
    }
    
    # Add additional features
    for i in range(n_features - 5):
        data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
    
    df = pd.DataFrame(data)
    
    # Add some realistic correlations
    df['rsi'] = 50 + 20 * np.sin(np.arange(n_samples) * 0.1) + np.random.normal(0, 5, n_samples)
    df['macd'] = df['close'].rolling(12).mean() - df['close'].rolling(26).mean()
    df['bb_upper'] = df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()
    df['bb_lower'] = df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std()
    
    return df


def create_targets(data: pd.DataFrame) -> pd.Series:
    """Create target variable for utility scoring."""
    # Next period returns
    targets = data['close'].pct_change().shift(-1)
    return targets


def benchmark_generators(data: pd.DataFrame, targets: pd.Series) -> Dict[str, Any]:
    """Benchmark both generators and compare performance."""
    results = {}
    
    # Test Enhanced Generator
    if ENHANCED_AVAILABLE:
        logger.info("Testing Enhanced DataDrivenInteractionGenerator...")
        
        # Test with different configurations
        configs = [
            {
                'name': 'Enhanced - Basic',
                'config': EnhancedInteractionConfig(
                    max_interactions=50,
                    enable_vectorbt=True,
                    enable_gpu=False,
                    enable_parallel=True,
                    memory_efficient=True,
                    enable_batch_processing=False,
                    enable_monitoring=True
                )
            },
            {
                'name': 'Enhanced - Batch Processing',
                'config': EnhancedInteractionConfig(
                    max_interactions=50,
                    enable_vectorbt=True,
                    enable_gpu=False,
                    enable_parallel=True,
                    memory_efficient=True,
                    enable_batch_processing=True,
                    enable_monitoring=True
                )
            },
            {
                'name': 'Enhanced - Memory Optimized',
                'config': EnhancedInteractionConfig(
                    max_interactions=50,
                    enable_vectorbt=True,
                    enable_gpu=False,
                    enable_parallel=True,
                    memory_efficient=True,
                    enable_batch_processing=True,
                    enable_monitoring=True,
                    chunk_size=500,
                    cache_size=500
                )
            }
        ]
        
        for config_info in configs:
            config_name = config_info['name']
            config = config_info['config']
            
            logger.info(f"Testing {config_name}...")
            
            try:
                generator = EnhancedDataDrivenInteractionGenerator(config)
                
                start_time = time.time()
                interactions = generator.generate_interactions(data, targets)
                end_time = time.time()
                
                processing_time = end_time - start_time
                stats = generator.get_performance_stats()
                
                results[config_name] = {
                    'interactions_generated': len(interactions),
                    'processing_time': processing_time,
                    'average_utility_score': np.mean([i.utility_score for i in interactions]) if interactions else 0,
                    'max_utility_score': max([i.utility_score for i in interactions]) if interactions else 0,
                    'vectorbt_operations': stats.get('vectorbt_operations', 0),
                    'pandas_fallbacks': stats.get('pandas_fallbacks', 0),
                    'cached_operations': stats.get('cached_operations', 0),
                    'memory_optimizations': stats.get('memory_optimizations', 0),
                    'cache_hit_rate': stats.get('cache_hit_rate', 0),
                    'top_interactions': [
                        {
                            'name': i.feature_name,
                            'utility_score': i.utility_score,
                            'optimization_method': i.optimization_method,
                            'processing_time': i.processing_time
                        } for i in interactions[:5]
                    ]
                }
                
                logger.info(f"✅ {config_name}: {len(interactions)} interactions in {processing_time:.2f}s")
                
            except Exception as e:
                logger.error(f"❌ {config_name} failed: {e}")
                results[config_name] = {'error': str(e)}
    
    # Test Original Generator
    if ORIGINAL_AVAILABLE:
        logger.info("Testing Original DataDrivenInteractionGenerator...")
        
        try:
            generator = DataDrivenInteractionGenerator(
                max_interactions=50,
                utility_threshold=0.1,
                correlation_threshold=0.95,
                enable_vectorbt=True
            )
            
            start_time = time.time()
            interactions = generator.generate_interactions(data, targets)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            results['Original'] = {
                'interactions_generated': len(interactions),
                'processing_time': processing_time,
                'average_utility_score': np.mean([i.utility_score for i in interactions]) if interactions else 0,
                'max_utility_score': max([i.utility_score for i in interactions]) if interactions else 0,
                'top_interactions': [
                    {
                        'name': i.feature_name,
                        'utility_score': i.utility_score,
                        'interaction_type': i.interaction_type
                    } for i in interactions[:5]
                ]
            }
            
            logger.info(f"✅ Original: {len(interactions)} interactions in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Original failed: {e}")
            results['Original'] = {'error': str(e)}
    
    return results


def print_comparison_results(results: Dict[str, Any]):
    """Print comparison results in a formatted table."""
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON RESULTS")
    print("="*80)
    
    # Create comparison table
    print(f"{'Generator':<30} {'Interactions':<12} {'Time (s)':<10} {'Avg Utility':<12} {'Max Utility':<12} {'VectorBT Ops':<12}")
    print("-" * 80)
    
    for name, result in results.items():
        if 'error' in result:
            print(f"{name:<30} {'ERROR':<12} {'-':<10} {'-':<12} {'-':<12} {'-':<12}")
            continue
        
        interactions = result.get('interactions_generated', 0)
        time_taken = result.get('processing_time', 0)
        avg_utility = result.get('average_utility_score', 0)
        max_utility = result.get('max_utility_score', 0)
        vectorbt_ops = result.get('vectorbt_operations', 0)
        
        print(f"{name:<30} {interactions:<12} {time_taken:<10.2f} {avg_utility:<12.3f} {max_utility:<12.3f} {vectorbt_ops:<12}")
    
    print("\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80)
    
    for name, result in results.items():
        if 'error' in result:
            print(f"\n❌ {name}: {result['error']}")
            continue
        
        print(f"\n📊 {name}:")
        print(f"   • Interactions generated: {result.get('interactions_generated', 0)}")
        print(f"   • Processing time: {result.get('processing_time', 0):.2f}s")
        print(f"   • Average utility score: {result.get('average_utility_score', 0):.3f}")
        print(f"   • Max utility score: {result.get('max_utility_score', 0):.3f}")
        
        if 'vectorbt_operations' in result:
            print(f"   • VectorBT operations: {result.get('vectorbt_operations', 0)}")
            print(f"   • Pandas fallbacks: {result.get('pandas_fallbacks', 0)}")
            print(f"   • Cached operations: {result.get('cached_operations', 0)}")
            print(f"   • Memory optimizations: {result.get('memory_optimizations', 0)}")
            print(f"   • Cache hit rate: {result.get('cache_hit_rate', 0):.1f}%")
        
        print(f"   • Top interactions:")
        for i, interaction in enumerate(result.get('top_interactions', [])[:3]):
            print(f"     {i+1}. {interaction['name']}: {interaction['utility_score']:.3f}")


def test_memory_efficiency():
    """Test memory efficiency with large datasets."""
    logger.info("Testing memory efficiency with large dataset...")
    
    if not ENHANCED_AVAILABLE:
        logger.warning("Enhanced generator not available for memory test")
        return
    
    # Create large dataset
    large_data = create_test_data(n_samples=20000, n_features=15)
    large_targets = create_targets(large_data)
    
    # Test with memory optimization
    config = EnhancedInteractionConfig(
        max_interactions=100,
        enable_vectorbt=True,
        enable_parallel=True,
        memory_efficient=True,
        enable_batch_processing=True,
        chunk_size=1000,
        enable_monitoring=True
    )
    
    generator = EnhancedDataDrivenInteractionGenerator(config)
    
    start_time = time.time()
    interactions = generator.generate_interactions(large_data, large_targets)
    end_time = time.time()
    
    stats = generator.get_performance_stats()
    
    print(f"\n📈 Memory Efficiency Test:")
    print(f"   • Dataset size: {large_data.shape}")
    print(f"   • Processing time: {end_time - start_time:.2f}s")
    print(f"   • Interactions generated: {len(interactions)}")
    print(f"   • Memory optimizations: {stats.get('memory_optimizations', 0)}")
    print(f"   • Memory savings: {stats.get('memory_savings', 0):.1f}%")


def test_different_data_sizes():
    """Test performance across different data sizes."""
    logger.info("Testing performance across different data sizes...")
    
    if not ENHANCED_AVAILABLE:
        logger.warning("Enhanced generator not available for size test")
        return
    
    sizes = [1000, 5000, 10000, 20000]
    results = {}
    
    for size in sizes:
        logger.info(f"Testing with {size} samples...")
        
        data = create_test_data(n_samples=size, n_features=8)
        targets = create_targets(data)
        
        config = EnhancedInteractionConfig(
            max_interactions=30,
            enable_vectorbt=True,
            enable_parallel=True,
            memory_efficient=True,
            enable_batch_processing=True,
            enable_monitoring=True
        )
        
        generator = EnhancedDataDrivenInteractionGenerator(config)
        
        start_time = time.time()
        interactions = generator.generate_interactions(data, targets)
        end_time = time.time()
        
        results[size] = {
            'samples': size,
            'interactions': len(interactions),
            'time': end_time - start_time,
            'time_per_interaction': (end_time - start_time) / max(len(interactions), 1),
            'avg_utility': np.mean([i.utility_score for i in interactions]) if interactions else 0
        }
    
    print(f"\n📊 Performance by Data Size:")
    print(f"{'Samples':<10} {'Interactions':<12} {'Time (s)':<10} {'Time/Int (s)':<12} {'Avg Utility':<12}")
    print("-" * 60)
    
    for size, result in results.items():
        print(f"{result['samples']:<10} {result['interactions']:<12} {result['time']:<10.2f} "
              f"{result['time_per_interaction']:<12.4f} {result['avg_utility']:<12.3f}")


def main():
    """Main test function."""
    print("🚀 Enhanced Data-Driven Interaction Generator Test Suite")
    print("="*60)
    
    # Create test data
    logger.info("Creating test data...")
    data = create_test_data(n_samples=5000, n_features=10)
    targets = create_targets(data)
    
    print(f"📊 Test data: {data.shape[0]} samples, {data.shape[1]} features")
    print(f"📊 Target variable: {targets.notna().sum()} valid values")
    
    # Benchmark generators
    logger.info("Starting benchmark tests...")
    results = benchmark_generators(data, targets)
    
    # Print results
    print_comparison_results(results)
    
    # Test memory efficiency
    test_memory_efficiency()
    
    # Test different data sizes
    test_different_data_sizes()
    
    print("\n✅ Test suite completed successfully!")
    
    # Performance improvement summary
    if 'Original' in results and 'Enhanced - Basic' in results:
        orig_time = results['Original'].get('processing_time', 0)
        enh_time = results['Enhanced - Basic'].get('processing_time', 0)
        
        if orig_time > 0 and enh_time > 0:
            improvement = ((orig_time - enh_time) / orig_time) * 100
            print(f"\n🎯 Performance Improvement: {improvement:.1f}% faster than original")


if __name__ == "__main__":
    main()