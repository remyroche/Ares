#!/usr/bin/env python3
"""
Demo Script for Enhanced Profit Labeling System

This script demonstrates the key features and capabilities of the enhanced
profit labeling system with a simple, interactive demo.

Author: AI Assistant
Date: 2025-01-10
"""

import sys
import time
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from src.training.steps.pre_training.profit_labeling.enhanced_profit_labeling_system import (
    EnhancedProfitLabelingSystem, ProfitLabelingConfig
)
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error


def print_banner():
    """Print demo banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    Enhanced Profit Labeling System Demo                     ║
║                                                                              ║
║  🚀 Comprehensive profit labeling with integrated tools                     ║
║  📊 VectorBT optimization and feature generation                            ║
║  🎯 Advanced feature selection and hyperparameter optimization              ║
║  ⚡ Hardware acceleration and memory optimization                           ║
║  📈 Quality assessment and evaluation metrics                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def demo_basic_usage():
    """Demo 1: Basic usage with simple configuration."""
    tprint_info("🔧 Demo 1: Basic Usage")
    tprint_info("=" * 50)
    
    # Create simple configuration
    config = ProfitLabelingConfig(
        symbols=["BTCUSDT"],
        timeframes=["1h"],
        max_features=50,
        enable_bayesian_optimization=False
    )
    
    tprint_info("Configuration:")
    tprint_info(f"  Symbols: {config.symbols}")
    tprint_info(f"  Timeframes: {config.timeframes}")
    tprint_info(f"  Max Features: {config.max_features}")
    
    # Initialize system
    tprint_info("\n🚀 Initializing system...")
    system = EnhancedProfitLabelingSystem(config)
    
    # Run pipeline
    tprint_info("🏃 Running pipeline...")
    start_time = time.time()
    
    try:
        results = system.run_full_pipeline()
        end_time = time.time()
        
        tprint_success(f"✅ Pipeline completed in {end_time - start_time:.2f} seconds")
        
        # Print results summary
        tprint_info("\n📊 Results Summary:")
        tprint_info(f"  Datasets: {len(results['data'])}")
        tprint_info(f"  Features: {sum(results['features'].values())}")
        tprint_info(f"  Labels: {sum(results['labels'].values())}")
        
        return results
        
    except Exception as e:
        tprint_error(f"❌ Demo failed: {e}")
        return None


def demo_advanced_usage():
    """Demo 2: Advanced usage with optimization."""
    tprint_info("\n🔧 Demo 2: Advanced Usage with Optimization")
    tprint_info("=" * 50)
    
    # Create advanced configuration
    config = ProfitLabelingConfig(
        symbols=["BTCUSDT", "ETHUSDT"],
        timeframes=["1h", "4h"],
        feature_categories=["volatility", "momentum", "volume"],
        max_features=100,
        feature_selection_method="ensemble",
        enable_bayesian_optimization=True,
        n_trials=25,  # Reduced for demo
        enable_parallel=True,
        memory_efficient=True
    )
    
    tprint_info("Advanced Configuration:")
    tprint_info(f"  Symbols: {config.symbols}")
    tprint_info(f"  Timeframes: {config.timeframes}")
    tprint_info(f"  Feature Categories: {config.feature_categories}")
    tprint_info(f"  Feature Selection: {config.feature_selection_method}")
    tprint_info(f"  Bayesian Optimization: {config.enable_bayesian_optimization}")
    tprint_info(f"  Trials: {config.n_trials}")
    
    # Initialize system
    tprint_info("\n🚀 Initializing advanced system...")
    system = EnhancedProfitLabelingSystem(config)
    
    # Run pipeline
    tprint_info("🏃 Running advanced pipeline...")
    start_time = time.time()
    
    try:
        results = system.run_full_pipeline()
        end_time = time.time()
        
        tprint_success(f"✅ Advanced pipeline completed in {end_time - start_time:.2f} seconds")
        
        # Print detailed results
        tprint_info("\n📊 Detailed Results:")
        for dataset, feature_count in results['features'].items():
            label_count = results['labels'].get(dataset, 0)
            selected_count = results['selected_features'].get(dataset, 0)
            tprint_info(f"  {dataset}:")
            tprint_info(f"    Features Generated: {feature_count}")
            tprint_info(f"    Labels Generated: {label_count}")
            tprint_info(f"    Features Selected: {selected_count}")
        
        if results.get('optimization'):
            tprint_info(f"\n🔧 Optimization Results:")
            for param, value in results['optimization'].items():
                tprint_info(f"  {param}: {value:.4f}")
        
        return results
        
    except Exception as e:
        tprint_error(f"❌ Advanced demo failed: {e}")
        return None


def demo_step_by_step():
    """Demo 3: Step-by-step pipeline execution."""
    tprint_info("\n🔧 Demo 3: Step-by-Step Pipeline Execution")
    tprint_info("=" * 50)
    
    # Create configuration
    config = ProfitLabelingConfig(
        symbols=["BTCUSDT"],
        timeframes=["1h"],
        max_features=75
    )
    
    # Initialize system
    system = EnhancedProfitLabelingSystem(config)
    
    try:
        # Step 1: Load data
        tprint_info("📊 Step 1: Loading data...")
        data = system.load_data()
        tprint_success(f"✅ Loaded {len(data)} datasets")
        
        # Step 2: Generate features
        tprint_info("\n🔧 Step 2: Generating features...")
        features = system.generate_features(data)
        tprint_success(f"✅ Generated features for {len(features)} datasets")
        
        # Step 3: Generate labels
        tprint_info("\n🏷️ Step 3: Generating labels...")
        labels = system.generate_labels(data)
        tprint_success(f"✅ Generated labels for {len(labels)} datasets")
        
        # Step 4: Select features
        tprint_info("\n🎯 Step 4: Selecting features...")
        selected_features = system.select_features(features, labels)
        tprint_success(f"✅ Selected features for {len(selected_features)} datasets")
        
        # Step 5: Evaluate labels
        tprint_info("\n📊 Step 5: Evaluating labels...")
        evaluation = system.evaluate_labels(features, labels)
        tprint_success(f"✅ Evaluated {len(evaluation)} datasets")
        
        # Print evaluation results
        tprint_info("\n📈 Evaluation Results:")
        for dataset, metrics in evaluation.items():
            tprint_info(f"  {dataset}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    tprint_info(f"    {metric}: {value:.4f}")
                else:
                    tprint_info(f"    {metric}: {value}")
        
        return {
            'data': data,
            'features': features,
            'labels': labels,
            'selected_features': selected_features,
            'evaluation': evaluation
        }
        
    except Exception as e:
        tprint_error(f"❌ Step-by-step demo failed: {e}")
        return None


def demo_performance_comparison():
    """Demo 4: Performance comparison between configurations."""
    tprint_info("\n🔧 Demo 4: Performance Comparison")
    tprint_info("=" * 50)
    
    configurations = [
        {
            "name": "Basic",
            "config": ProfitLabelingConfig(
                symbols=["BTCUSDT"],
                timeframes=["1h"],
                max_features=50,
                enable_bayesian_optimization=False
            )
        },
        {
            "name": "Optimized",
            "config": ProfitLabelingConfig(
                symbols=["BTCUSDT"],
                timeframes=["1h"],
                max_features=50,
                enable_bayesian_optimization=True,
                n_trials=10,  # Small number for demo
                enable_parallel=True
            )
        }
    ]
    
    results = {}
    
    for config_info in configurations:
        name = config_info["name"]
        config = config_info["config"]
        
        tprint_info(f"\n🏃 Running {name} configuration...")
        
        try:
            system = EnhancedProfitLabelingSystem(config)
            
            start_time = time.time()
            result = system.run_full_pipeline()
            end_time = time.time()
            
            execution_time = end_time - start_time
            results[name] = {
                'execution_time': execution_time,
                'datasets': len(result['data']),
                'features': sum(result['features'].values()),
                'labels': sum(result['labels'].values())
            }
            
            tprint_success(f"✅ {name} completed in {execution_time:.2f} seconds")
            
        except Exception as e:
            tprint_error(f"❌ {name} failed: {e}")
            results[name] = {'error': str(e)}
    
    # Print comparison
    tprint_info("\n📊 Performance Comparison:")
    for name, result in results.items():
        if 'error' in result:
            tprint_warning(f"  {name}: Failed - {result['error']}")
        else:
            tprint_info(f"  {name}:")
            tprint_info(f"    Execution Time: {result['execution_time']:.2f}s")
            tprint_info(f"    Datasets: {result['datasets']}")
            tprint_info(f"    Features: {result['features']}")
            tprint_info(f"    Labels: {result['labels']}")
    
    return results


def interactive_demo():
    """Interactive demo with user choices."""
    tprint_info("\n🔧 Interactive Demo")
    tprint_info("=" * 50)
    
    print("\nAvailable demos:")
    print("1. Basic Usage")
    print("2. Advanced Usage with Optimization")
    print("3. Step-by-Step Pipeline")
    print("4. Performance Comparison")
    print("5. All Demos")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nSelect demo (0-5): ").strip()
            
            if choice == "0":
                tprint_info("👋 Goodbye!")
                break
            elif choice == "1":
                demo_basic_usage()
            elif choice == "2":
                demo_advanced_usage()
            elif choice == "3":
                demo_step_by_step()
            elif choice == "4":
                demo_performance_comparison()
            elif choice == "5":
                run_all_demos()
            else:
                tprint_warning("Invalid choice. Please select 0-5.")
                
        except KeyboardInterrupt:
            tprint_info("\n👋 Goodbye!")
            break
        except Exception as e:
            tprint_error(f"Error: {e}")


def run_all_demos():
    """Run all demos in sequence."""
    tprint_info("\n🚀 Running All Demos")
    tprint_info("=" * 50)
    
    demos = [
        ("Basic Usage", demo_basic_usage),
        ("Advanced Usage", demo_advanced_usage),
        ("Step-by-Step", demo_step_by_step),
        ("Performance Comparison", demo_performance_comparison)
    ]
    
    results = {}
    
    for name, demo_func in demos:
        tprint_info(f"\n{'='*60}")
        tprint_info(f"Running: {name}")
        tprint_info(f"{'='*60}")
        
        try:
            result = demo_func()
            results[name] = "Success" if result else "Failed"
            tprint_success(f"✅ {name} completed")
        except Exception as e:
            tprint_error(f"❌ {name} failed: {e}")
            results[name] = f"Failed: {e}"
    
    # Summary
    tprint_info(f"\n{'='*60}")
    tprint_info("Demo Summary")
    tprint_info(f"{'='*60}")
    
    for name, status in results.items():
        if "Success" in status:
            tprint_success(f"✅ {name}: {status}")
        else:
            tprint_error(f"❌ {name}: {status}")


def main():
    """Main demo function."""
    print_banner()
    
    if len(sys.argv) > 1:
        # Command line argument provided
        demo_type = sys.argv[1].lower()
        
        if demo_type == "basic":
            demo_basic_usage()
        elif demo_type == "advanced":
            demo_advanced_usage()
        elif demo_type == "step":
            demo_step_by_step()
        elif demo_type == "performance":
            demo_performance_comparison()
        elif demo_type == "all":
            run_all_demos()
        else:
            tprint_warning(f"Unknown demo type: {demo_type}")
            tprint_info("Available types: basic, advanced, step, performance, all")
    else:
        # Interactive mode
        interactive_demo()


if __name__ == "__main__":
    main()