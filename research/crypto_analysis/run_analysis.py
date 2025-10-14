#!/usr/bin/env python3
"""
Launcher script for Automated Cryptocurrency Analysis

This script provides a simple interface to run the automated crypto analysis pipeline.
"""

import asyncio
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from automated_crypto_processor import AutomatedCryptoProcessor
from config import (
    ASSETS, DATA_CONFIG, API_CONFIG, HARDWARE_CONFIG,
    create_directories, validate_config, get_config_summary
)

def print_banner():
    """Print the analysis banner."""
    print("=" * 80)
    print("🚀 AUTOMATED CRYPTOCURRENCY ANALYSIS PIPELINE")
    print("=" * 80)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def print_configuration():
    """Print current configuration."""
    print("📋 CONFIGURATION")
    print("-" * 40)
    summary = get_config_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run automated cryptocurrency analysis")
    parser.add_argument("--years", type=int, default=DATA_CONFIG["years"],
                       help="Number of years of data to download (default: 2)")
    parser.add_argument("--assets", nargs="+", default=ASSETS,
                       help="List of assets to analyze (default: all configured assets)")
    parser.add_argument("--api-key", type=str, default=API_CONFIG["binance_api_key"],
                       help="Binance API key (optional)")
    parser.add_argument("--api-secret", type=str, default=API_CONFIG["binance_api_secret"],
                       help="Binance API secret (optional)")
    parser.add_argument("--output-dir", type=str, default=DATA_CONFIG["output_dir"],
                       help="Output directory for results")
    parser.add_argument("--data-dir", type=str, default=DATA_CONFIG["data_dir"],
                       help="Data directory for historical data")
    parser.add_argument("--no-hardware-optimization", action="store_true",
                       help="Disable hardware optimizations")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate configuration and exit")
    parser.add_argument("--config-summary", action="store_true",
                       help="Show configuration summary and exit")
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Validate configuration
    errors = validate_config()
    if errors:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"  • {error}")
        return 1
    
    # Show configuration summary if requested
    if args.config_summary:
        print_configuration()
        return 0
    
    # Validate only if requested
    if args.validate_only:
        print("✅ Configuration is valid")
        return 0
    
    # Create directories
    create_directories()
    
    # Update configuration with command line arguments
    DATA_CONFIG["years"] = args.years
    DATA_CONFIG["output_dir"] = args.output_dir
    DATA_CONFIG["data_dir"] = args.data_dir
    API_CONFIG["binance_api_key"] = args.api_key
    API_CONFIG["binance_api_secret"] = args.api_secret
    HARDWARE_CONFIG["use_m1_optimizations"] = not args.no_hardware_optimization
    
    # Print configuration
    print_configuration()
    
    # Run the analysis
    async def run_analysis():
        """Run the analysis pipeline."""
        try:
            # Initialize processor
            processor = AutomatedCryptoProcessor(
                data_dir=args.data_dir,
                output_dir=args.output_dir
            )
            
            # Update assets if specified
            if args.assets != ASSETS:
                processor.assets = args.assets
            
            # Run the pipeline
            results = await processor.process_all_assets(
                years=args.years,
                api_key=args.api_key,
                api_secret=args.api_secret
            )
            
            # Print final results
            print("\n" + "=" * 80)
            print("🎉 ENHANCED ANALYSIS COMPLETED")
            print("=" * 80)
            print(f"✅ Successfully processed: {results['summary']['successfully_processed']}/{results['summary']['total_assets']} assets")
            print(f"📊 Success rate: {results['summary']['success_rate']:.1f}%")
            print(f"📁 Results saved to: {args.output_dir}")
            
            # Show optimization status if available
            if "optimization_status" in results:
                opt_status = results["optimization_status"]
                print(f"\n🔧 OPTIMIZATION STATUS:")
                print(f"   Ares Utilities: {'✅' if opt_status['ares_utilities_available'] else '❌'}")
                print(f"   Hardware Optimizations: {'✅' if opt_status['hardware_optimizations_available'] else '❌'}")
                print(f"   Parquet Utils: {'✅' if opt_status['parquet_utils_enabled'] else '❌'}")
                print(f"   Memory Optimization: {'✅' if opt_status['memory_optimization_enabled'] else '❌'}")
                print(f"   GPU Acceleration: {'✅' if opt_status['gpu_acceleration_enabled'] else '❌'}")
                print(f"   CPU Optimization: {'✅' if opt_status['cpu_optimization_enabled'] else '❌'}")
                print(f"   Enhanced Processing: {'✅' if results.get('enhanced_processing', False) else '❌'}")
            
            if results['assets_processed']:
                print("\n📈 Processed assets:")
                for asset in results['assets_processed']:
                    metrics = results['all_metrics'][asset]
                    print(f"  • {asset}: {metrics['price_metrics']['total_return']*100:.2f}% return, "
                          f"{metrics['price_metrics']['price_volatility']*100:.2f}% volatility")
            
            if results['assets_failed']:
                print("\n❌ Failed assets:")
                for failed in results['assets_failed']:
                    print(f"  • {failed['asset']}: {failed['error']}")
            
            print(f"\n📊 Check the following directories:")
            print(f"  • {args.output_dir}/reports/ - Human-readable reports and detailed metrics")
            print(f"  • {args.output_dir}/csv/ - CSV files for further analysis")
            print(f"  • {args.output_dir}/charts/ - Visualization charts")
            
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
    
    # Run the async analysis
    return asyncio.run(run_analysis())

if __name__ == "__main__":
    sys.exit(main())