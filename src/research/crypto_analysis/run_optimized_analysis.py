#!/usr/bin/env python3
"""
Optimized Cryptocurrency Analysis Launcher

This script uses the advanced Ares utility framework for maximum performance:
- Hardware acceleration (M1 GPU/CPU/Memory optimization)
- Parallel processing and vectorized operations
- Advanced data validation and error handling
- Async file operations and memory management
"""

import asyncio
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from optimized_crypto_processor import OptimizedCryptoProcessor
from config import (
    ASSETS, DATA_CONFIG, API_CONFIG, HARDWARE_CONFIG,
    create_directories, validate_config, get_config_summary
)

def print_optimized_banner():
    """Print the optimized analysis banner."""
    print("=" * 80)
    print("🚀 OPTIMIZED CRYPTOCURRENCY ANALYSIS PIPELINE")
    print("   Powered by Ares Advanced Utilities Framework")
    print("=" * 80)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def print_optimization_status():
    """Print optimization capabilities."""
    print("🔧 OPTIMIZATION CAPABILITIES")
    print("-" * 40)

    # Check available optimizations
    optimizations = []

    try:
        from src.utils.hardware import get_integrated_hardware_manager
        optimizations.append("✅ M1 GPU Acceleration")
    except ImportError:
        optimizations.append("❌ M1 GPU Acceleration")

    try:
        from src.utils.hardware import get_integrated_hardware_manager
        optimizations.append("✅ M1 Memory Optimization")
    except ImportError:
        optimizations.append("❌ M1 Memory Optimization")

    try:
        from src.utils.hardware import get_comprehensive_optimizer
        optimizations.append("✅ M1 CPU Optimization")
    except ImportError:
        optimizations.append("❌ M1 CPU Optimization")

    try:
        from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations
        optimizations.append("✅ Matrix Operations")
    except ImportError:
        optimizations.append("❌ Matrix Operations")

    try:
        from src.utils.parquet_utils import ParquetUtils
        optimizations.append("✅ Advanced Parquet Utils")
    except ImportError:
        optimizations.append("❌ Advanced Parquet Utils")

    for opt in optimizations:
        print(f"  {opt}")
    print()

def main():
    """Main function for optimized analysis."""
    parser = argparse.ArgumentParser(description="Run optimized cryptocurrency analysis")
    parser.add_argument("--years", type=int, default=DATA_CONFIG["years"],
                       help="Number of years of data to analyze (default: 2)")
    parser.add_argument("--use-existing", action="store_true", default=True,
                       help="Use existing data files if available (default: True)")
    parser.add_argument("--force-download", action="store_true",
                       help="Force fresh data download")
    parser.add_argument("--assets", nargs="+", default=ASSETS,
                       help="List of assets to analyze")
    parser.add_argument("--output-dir", type=str, default=DATA_CONFIG["output_dir"],
                       help="Output directory for results")
    parser.add_argument("--data-dir", type=str, default=DATA_CONFIG["data_dir"],
                       help="Data directory for historical data")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate configuration and exit")
    parser.add_argument("--optimization-status", action="store_true",
                       help="Show optimization status and exit")

    args = parser.parse_args()

    # Print banner
    print_optimized_banner()

    # Show optimization status if requested
    if args.optimization_status:
        print_optimization_status()
        return 0

    # Validate configuration
    errors = validate_config()
    if errors:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"  • {error}")
        return 1

    # Validate only if requested
    if args.validate_only:
        print("✅ Configuration is valid")
        print_optimization_status()
        return 0

    # Show optimization capabilities
    print_optimization_status()

    # Create directories
    create_directories()

    # Update configuration
    use_existing = args.use_existing and not args.force_download

    # Run the optimized analysis
    async def run_optimized_analysis():
        """Run the optimized analysis pipeline."""
        try:
            # Initialize optimized processor
            processor = OptimizedCryptoProcessor(
                data_dir=args.data_dir,
                output_dir=args.output_dir
            )

            # Update assets if specified
            if args.assets != ASSETS:
                processor.assets = args.assets

            # Run the optimized pipeline
            results = await processor.process_all_assets_optimized(
                years=args.years,
                use_existing=use_existing
            )

            # Print final results
            print("\n" + "=" * 80)
            print("🎉 OPTIMIZED ANALYSIS COMPLETED")
            print("=" * 80)

            if results["success"]:
                print(f"✅ Successfully processed: {results['summary']['successfully_processed']}/{results['summary']['total_assets']} assets")
                print(f"📊 Success rate: {results['summary']['success_rate']:.1f}%")
                print(f"⚡ Processing time: {results['processing_time']:.1f}s")
                print(f"🚀 Optimization enabled: {results['optimization_enabled']}")

                # Show optimization details
                opts = results['optimizations_used']
                print(f"\n🔧 Optimizations used:")
                print(f"   GPU Acceleration: {'✅' if opts['gpu_acceleration'] else '❌'}")
                print(f"   Memory Optimization: {'✅' if opts['memory_optimization'] else '❌'}")
                print(f"   CPU Optimization: {'✅' if opts['cpu_optimization'] else '❌'}")
                print(f"   Matrix Operations: {'✅' if opts['matrix_operations'] else '❌'}")
                print(f"   Parallel Processing: {'✅' if opts['parallel_processing'] else '❌'}")
                print(f"   Data Validation: {'✅' if opts['data_validation'] else '❌'}")
                print(f"   Async File Operations: {'✅' if opts['async_file_operations'] else '❌'}")

                # Show top performers with composite scores
                if "summary_metrics" in results and "composite_scores" in results["summary_metrics"]:
                    print(f"\n🏆 TOP PERFORMERS (Composite Score):")
                    composite_scores = results["summary_metrics"]["composite_scores"]
                    sorted_performers = sorted(composite_scores.items(),
                                             key=lambda x: x[1]['composite_score'], reverse=True)

                    for i, (asset, scores) in enumerate(sorted_performers[:5], 1):
                        print(f"   {i}. {asset}: Score {scores['composite_score']:.3f} "
                              f"(Profit: {scores['avg_profit']*100:.2f}%, "
                              f"Frequency: {scores['avg_frequency']*100:.1f}%)")

                print(f"\n📁 Results saved to: {args.output_dir}")
                print(f"  • {args.output_dir}/reports/ - Enhanced reports with methodology")
                print(f"  • {args.output_dir}/csv/ - Structured data for analysis")
                print(f"  • Optimized JSON results with complete metadata")

            else:
                print(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
                return 1

            return 0

        except KeyboardInterrupt:
            print("\n⚠️ Analysis interrupted by user")
            return 1
        except Exception as e:
            print(f"\n❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
        finally:
            # Cleanup
            try:
                processor.cleanup()
            except:
                pass

    # Run the async optimized analysis
    return asyncio.run(run_optimized_analysis())

if __name__ == "__main__":
    sys.exit(main())
