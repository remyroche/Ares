# src/training/examples/optimized_training_example.py

"""
Example script demonstrating how to use the optimized enhanced training manager
with all computational optimization strategies.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.computational_optimization_config import (
    get_optimization_config,
    get_performance_expectations,
    validate_optimization_config,
)
from src.training.factory import (
    OptimizedTrainingFactory,
    create_optimized_training_system,
    get_optimization_recommendations,
)
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    failed,
    warning,
)
from src.training.memory_profiler import profile_memory_usage, MemoryProfiler


async def main():
    """Main example function demonstrating optimized training."""
    logger = system_logger.getChild("OptimizedTrainingExample")
    logger.info("🚀 Starting Optimized Training Example")

    # 1. Load base configuration (you would load from your config system)
    base_config = {
        "training": {
            "n_trials": 200,
            "max_trials": 500,
            "lookback_days": 30,
            "blank_training_mode": False,
        },
        "model": {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 100},
    }

    # 2. Get optimization configuration with custom overrides
    custom_optimization_config = {
        "parallelization": {
            "max_workers": min(os.cpu_count(), 12),  # Use more workers if available
        },
        "memory_management": {
            "memory_threshold": 0.75,  # More aggressive memory management
        },
        "monitoring": {
            "monitoring_interval": 15,  # More frequent monitoring
        },
    }

    optimization_config = get_optimization_config(custom_optimization_config)

    # 3. Validate configuration
    validation_results = validate_optimization_config(optimization_config)
    if not validation_results["valid"]:
        print(failed("Configuration validation failed:"))
        for error in validation_results["errors"]:
            print(error("  - {error}"))
        return

    if validation_results["warnings"]:
        print(warning("Configuration warnings:"))
        for warning in validation_results["warnings"]:
            print(warning("  - {warning}"))

    # 4. Get optimization recommendations
    recommendations = get_optimization_recommendations(base_config)
    logger.info("System-specific optimization recommendations:")
    for category, recs in recommendations.items():
        if recs:
            logger.info(f"  {category.replace('_', ' ').title()}:")
            for rec in recs:
                logger.info(f"    - {rec}")

    # 5. Create optimized training system
    logger.info("Creating optimized training system...")
    complete_config = base_config.copy()
    complete_config["computational_optimization"] = optimization_config

    training_system = create_optimized_training_system(complete_config)

    # 6. Get optimization summary
    factory = OptimizedTrainingFactory(complete_config)
    optimization_summary = factory.get_optimization_summary()

    logger.info("Optimization Summary:")
    logger.info(
        f"  Enabled Optimizations: {optimization_summary['optimizations_enabled']}",
    )
    logger.info(f"  Configuration: {optimization_summary['configuration']}")

    # 7. Show expected performance improvements
    performance_expectations = get_performance_expectations()
    logger.info("Expected Performance Improvements:")
    for category, improvements in performance_expectations[
        "computational_time_reduction"
    ].items():
        logger.info(
            f"  {category}: {improvements['min']}-{improvements['max']}% reduction",
        )

    # 8. Initialize components
    logger.info("Initializing optimized training components...")

    training_manager = training_system["training_manager"]
    memory_profiler = training_system["memory_profiler"]
    leak_detector = training_system["leak_detector"]
    step_executor = training_system["step_executor"]

    # Initialize training manager
    if not await training_manager.initialize():
        print(failed("Failed to initialize training manager"))
        return

    # 9. Take initial memory snapshot
    initial_snapshot = memory_profiler.take_snapshot("initialization")
    logger.info(
        f"Initial memory usage: {initial_snapshot['process_memory']['rss_mb']:.1f}MB",
    )

    # 10. Example: Execute optimized training pipeline
    logger.info("Executing optimized training pipeline...")

    symbol = "ETHUSDT"
    exchange = "BINANCE"
    timeframe = "1h"

    try:
        # Option A: Use the enhanced training manager directly
        training_results = await training_manager.execute_optimized_training(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
        )

        logger.info("Training completed via enhanced training manager")
        logger.info(f"Training results keys: {list(training_results.keys())}")

        # Option B: Use the step executor directly for more control
        # step_results = await step_executor.execute_optimized_pipeline(
        #     symbol=symbol,
        #     exchange=exchange,
        #     timeframe=timeframe
        # )
        # logger.info("Training completed via step executor")

    except Exception:
        print(failed("Training failed: {e}"))
        training_results = {}

    # 11. Memory analysis after training
    final_snapshot = memory_profiler.take_snapshot("training_completed")
    logger.info(
        f"Final memory usage: {final_snapshot['process_memory']['rss_mb']:.1f}MB",
    )

    # Check for memory leaks
    leak_results = leak_detector.check_for_leaks()
    if leak_results["leak_detected"]:
        print(warning("Memory leak detected!"))
        for _indicator in leak_results.get("indicators", []):
            print(warning("  - {indicator}"))
        print(warning("Recommendations:"))
        for rec in leak_results.get("recommendations", []):
            print(warning("  - {rec}"))
    else:
        logger.info("No memory leaks detected")

    # 12. Generate memory usage trends
    memory_trends = memory_profiler.analyze_memory_trends()
    if memory_trends["status"] == "success":
        rss_stats = memory_trends["rss_stats"]
        logger.info(
            f"Memory usage trends - Mean: {rss_stats['mean']:.1f}MB, "
            f"Max: {rss_stats['max']:.1f}MB, "
            f"Trend: {rss_stats['trend_mb_per_snapshot']:.2f}MB/snapshot",
        )

    # 13. Get execution statistics
    if hasattr(step_executor, "get_execution_stats"):
        exec_stats = step_executor.get_execution_stats()
        logger.info("Execution Statistics:")
        logger.info(f"  Cache hit ratio: {exec_stats.get('cache_hit_ratio', 0):.2%}")
        logger.info(
            f"  Parallel execution: {exec_stats.get('parallel_execution_enabled', False)}",
        )
        logger.info(f"  Max workers: {exec_stats.get('max_workers', 1)}")

    optimization_stats = training_manager.get_optimization_stats()
    logger.info(f"Optimization Statistics: {optimization_stats}")

    # 14. Generate comprehensive memory report
    memory_report = memory_profiler.generate_memory_report()
    if memory_report["status"] == "success":
        logger.info("Memory Report Generated:")
        for rec in memory_report.get("recommendations", []):
            logger.info(f"  - {rec}")

    # 15. Force memory optimization and cleanup
    logger.info("Performing final memory optimization...")
    optimization_results = memory_profiler.optimize_memory_usage()
    memory_freed = optimization_results["garbage_collection"]["memory_freed_mb"]
    logger.info(f"Memory optimization freed {memory_freed:.1f}MB")

    # 16. Cleanup
    await training_manager.cleanup()
    memory_profiler.stop_continuous_monitoring()

    logger.info("✅ Optimized training example completed successfully")

    # 17. Summary of results
    logger.info("\n📊 TRAINING SUMMARY:")
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Exchange: {exchange}")
    logger.info(f"Timeframe: {timeframe}")
    logger.info(
        f"Optimizations Used: {list(optimization_summary['optimizations_enabled'].keys())}",
    )
    logger.info(f"Memory Freed: {memory_freed:.1f}MB")
    logger.info(
        f"Memory Leak Status: {'Detected' if leak_results['leak_detected'] else 'Clean'}",
    )

    if training_results.get("execution_stats"):
        stats = training_results["execution_stats"]
        logger.info(f"Total Execution Time: {stats.get('total_time_seconds', 0):.2f}s")
        logger.info(f"Cache Hit Ratio: {stats.get('cache_hit_ratio', 0):.2%}")
        logger.info(f"Parallel Workers Used: {stats.get('parallel_workers_used', 1)}")


def demonstrate_individual_components():
    """Demonstrate individual optimization components."""
    logger = system_logger.getChild("ComponentDemo")
    logger.info("🔧 Demonstrating Individual Optimization Components")

    # 1. Memory Profiler Demo
    logger.info("1. Memory Profiler Demo")
    profiler = MemoryProfiler(enable_continuous_monitoring=False)

    snapshot1 = profiler.take_snapshot("demo_start")
    logger.info(f"Demo start memory: {snapshot1['process_memory']['rss_mb']:.1f}MB")

    # Simulate some memory usage
    large_data = [list(range(10000)) for _ in range(100)]

    snapshot2 = profiler.take_snapshot("after_allocation")
    logger.info(f"After allocation: {snapshot2['process_memory']['rss_mb']:.1f}MB")

    # Clean up
    del large_data
    optimization_results = profiler.optimize_memory_usage()
    logger.info(
        f"Memory freed: {optimization_results['garbage_collection']['memory_freed_mb']:.1f}MB",
    )

    # 2. Configuration Demo
    logger.info("2. Configuration Demo")
    config = get_optimization_config()
    logger.info(f"Default max workers: {config['parallelization']['max_workers']}")
    logger.info(f"Memory threshold: {config['memory_management']['memory_threshold']}")
    logger.info(f"Caching enabled: {config['caching']['enabled']}")

    # 3. Factory Demo
    logger.info("3. Factory Demo")
    factory = OptimizedTrainingFactory({"training": {}})
    summary = factory.get_optimization_summary()
    logger.info(f"Factory optimization summary: {summary['optimizations_enabled']}")


if __name__ == "__main__":
    # Run the main example
    asyncio.run(main())

    # Run component demonstrations
    print("\n" + "=" * 50)
    demonstrate_individual_components()
