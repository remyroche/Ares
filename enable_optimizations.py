#!/usr/bin/env python3
"""
Enable Optimizations Script

This script demonstrates how to enable the new computational optimization features
in the Ares training system.

Usage:
    python enable_optimizations.py [command] [options]
    
Examples:
    # Enable optimizations for blank training
    python enable_optimizations.py blank --symbol ETHUSDT --exchange BINANCE
    
    # Enable optimizations for backtesting  
    python enable_optimizations.py backtest --symbol ETHUSDT --exchange BINANCE
    
    # Check optimization status
    python enable_optimizations.py status
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config.computational_optimization_config import (
    get_optimization_config,
    get_performance_expectations,
    is_optimization_enabled
)
from src.training.factory import (
    create_optimized_training_system,
    get_optimization_recommendations
)
from src.training.enhanced_training_manager_optimized import EnhancedTrainingManagerOptimized
from src.utils.logger import system_logger


class OptimizedAresRunner:
    """Runner that uses the new optimization features."""
    
    def __init__(self):
        self.logger = system_logger.getChild("OptimizedAresRunner")
        
    async def run_optimized_training(self, command: str, symbol: str, exchange: str):
        """Run training with optimizations enabled."""
        self.logger.info(f"🚀 Running optimized {command} for {symbol} on {exchange}")
        
        # Create optimized configuration
        base_config = {
            "training": {
                "n_trials": 8 if command == "blank" else 100,
                "max_trials": 5 if command == "blank" else 200,
                "lookback_days": 45 if command == "blank" else 730
            },
            "model": {
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 100
            }
        }
        
        # Get system-specific recommendations
        recommendations = get_optimization_recommendations(base_config)
        self.logger.info("💡 System-specific optimization recommendations:")
        for category, recs in recommendations.items():
            if recs:
                self.logger.info(f"  {category.replace('_', ' ').title()}:")
                for rec in recs:
                    self.logger.info(f"    - {rec}")
        
        # Create optimized training system
        training_system = create_optimized_training_system(base_config)
        
        # Show what optimizations are enabled
        optimization_stats = training_system["training_manager"].get_optimization_stats()
        self.logger.info("🔧 Enabled optimizations:")
        for opt_name, enabled in optimization_stats.items():
            if isinstance(enabled, bool):
                status = "✅" if enabled else "❌"
                self.logger.info(f"  {opt_name}: {status}")
        
        # Execute optimized training
        try:
            results = await training_system["training_manager"].execute_optimized_training(
                symbol=symbol,
                exchange=exchange,
                timeframe="1h"
            )
            
            # Show results
            if results:
                self.logger.info("✅ Optimized training completed successfully!")
                
                # Show optimization statistics
                if "execution_stats" in results:
                    stats = results["execution_stats"]
                    self.logger.info("📊 Performance Statistics:")
                    self.logger.info(f"  Total time: {stats.get('total_time_seconds', 0):.2f}s")
                    self.logger.info(f"  Cache hit ratio: {stats.get('cache_hit_ratio', 0):.2%}")
                    self.logger.info(f"  Parallel workers: {stats.get('parallel_workers_used', 1)}")
                
                # Show memory profile
                memory_profile = training_system["memory_profiler"].get_memory_profile()
                self.logger.info("💾 Memory Profile:")
                self.logger.info(f"  Memory usage: {memory_profile.get('percentage', 0):.1f}%")
                self.logger.info(f"  Used memory: {memory_profile.get('used_gb', 0):.1f}GB")
                
                return True
            else:
                self.logger.error("❌ Training failed - no results returned")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Optimized training failed: {e}")
            return False
        
        finally:
            # Cleanup
            await training_system["training_manager"].cleanup()
            training_system["memory_profiler"].stop_continuous_monitoring()
    
    def show_optimization_status(self):
        """Show the status of optimization features."""
        self.logger.info("🔍 Optimization Status Check")
        self.logger.info("=" * 50)
        
        # Check which optimizations are available
        optimizations = [
            "caching", "parallelization", "early_stopping", 
            "memory_management", "data_streaming", "adaptive_sampling"
        ]
        
        for opt in optimizations:
            enabled = is_optimization_enabled(opt)
            status = "✅ ENABLED" if enabled else "❌ DISABLED"
            self.logger.info(f"{opt.replace('_', ' ').title()}: {status}")
        
        # Show expected performance improvements
        self.logger.info("\n📈 Expected Performance Improvements:")
        expectations = get_performance_expectations()
        for category, improvements in expectations["computational_time_reduction"].items():
            self.logger.info(f"  {category}: {improvements['min']}-{improvements['max']}% faster")
        
        # Show system recommendations
        base_config = {"training": {"n_trials": 100}}
        recommendations = get_optimization_recommendations(base_config)
        self.logger.info("\n💡 System Recommendations:")
        for category, recs in recommendations.items():
            if recs:
                self.logger.info(f"  {category.replace('_', ' ').title()}:")
                for rec in recs[:2]:  # Show top 2 recommendations
                    self.logger.info(f"    - {rec}")
    
    def compare_with_original(self):
        """Show comparison between original and optimized approaches."""
        self.logger.info("🔄 Original vs Optimized Comparison")
        self.logger.info("=" * 50)
        
        comparisons = [
            ("Training Manager", "EnhancedTrainingManager", "EnhancedTrainingManagerOptimized"),
            ("Backtesting", "Sequential evaluation", "Cached + Parallel evaluation"),
            ("Memory Usage", "Standard pandas operations", "Optimized dtypes + Parquet"),
            ("Parameter Search", "Random sampling", "Adaptive sampling"),
            ("Data Loading", "Full reload each time", "Streaming + Incremental"),
            ("Memory Monitoring", "Manual checking", "Continuous profiling + Leak detection"),
            ("Step Execution", "Sequential steps", "Parallel + Cached steps"),
            ("Model Training", "Full training each trial", "Incremental + State reuse"),
        ]
        
        for feature, original, optimized in comparisons:
            self.logger.info(f"\n{feature}:")
            self.logger.info(f"  📜 Original: {original}")
            self.logger.info(f"  🚀 Optimized: {optimized}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enable and test Ares optimization features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enable_optimizations.py blank --symbol ETHUSDT --exchange BINANCE
  python enable_optimizations.py backtest --symbol ETHUSDT --exchange BINANCE  
  python enable_optimizations.py status
  python enable_optimizations.py compare
        """
    )
    
    parser.add_argument(
        "command",
        choices=["blank", "backtest", "status", "compare"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--symbol",
        type=str,
        default="ETHUSDT",
        help="Trading symbol (default: ETHUSDT)"
    )
    
    parser.add_argument(
        "--exchange", 
        type=str,
        default="BINANCE",
        help="Exchange (default: BINANCE)"
    )
    
    args = parser.parse_args()
    
    runner = OptimizedAresRunner()
    
    if args.command == "status":
        runner.show_optimization_status()
    elif args.command == "compare":
        runner.compare_with_original()
    elif args.command in ["blank", "backtest"]:
        success = asyncio.run(runner.run_optimized_training(
            args.command, args.symbol, args.exchange
        ))
        if not success:
            sys.exit(1)
    
    print("\n🎉 Optimization demonstration completed!")
    print("📚 For more details, see: src/training/README_OPTIMIZATIONS.md")
    print("🔧 Integration guide: src/training/integration_guide.py")


if __name__ == "__main__":
    main()