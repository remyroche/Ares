#!/usr/bin/env python3
"""
Example script showing how to call Sticky Finite HMM Regime Discovery Step directly.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.steps.market_analysis.sticky_finite_hmm_clustering.sticky_finite_hmm_regime_discovery_step import run_sticky_finite_hmm_step


async def main():
    """Main function to demonstrate direct step execution."""
    
    print("🚀 Calling Sticky Finite HMM Regime Discovery Step Directly")
    print("=" * 80)
    
    # Configuration for the step
    config = {
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'regime_timeframe': '1h',
        'execution_mode': 'light',  # Use light mode for faster execution
        'enable_auto_tuning': True,
        'direction': 'long',
        'interaction_generation_mode': 'analyst'
    }
    
    print(f"📊 Configuration:")
    print(f"   • Symbol: {config['symbol']}")
    print(f"   • Exchange: {config['exchange']}")
    print(f"   • Timeframe: {config['regime_timeframe']}")
    print(f"   • Execution Mode: {config['execution_mode']}")
    print(f"   • Auto-Tuning: {config['enable_auto_tuning']}")
    print()
    
    try:
        # Run the step directly
        print("🔧 Executing Sticky Finite HMM Regime Discovery Step...")
        results = await run_sticky_finite_hmm_step(config)
        
        # Report results
        if results.get('success', False):
            print("✅ Sticky Finite HMM Regime Discovery completed successfully!")
            print()
            print("📈 Results Summary:")
            
            # Key metrics
            n_regimes = results.get('n_regimes', 0)
            execution_time = results.get('execution_time', 0)
            
            print(f"   • Number of Regimes: {n_regimes}")
            print(f"   • Execution Time: {execution_time:.2f}s")
            
            # Quality metrics if available
            quality_metrics = results.get('quality_metrics', {})
            if quality_metrics:
                print(f"   • Composite Quality Score: {quality_metrics.get('composite_score', 0):.4f}")
                print(f"   • Silhouette Score: {quality_metrics.get('silhouette_score', 0):.4f}")
                print(f"   • Temporal Smoothness: {quality_metrics.get('temporal_smoothness', 0):.4f}")
            
            # Artifacts
            if 'artifacts' in results:
                print("💾 Generated Artifacts:")
                for artifact_name, artifact_path in results['artifacts'].items():
                    print(f"   • {artifact_name}: {artifact_path}")
            
            print()
            print("🎉 Complete pipeline executed successfully!")
            
        else:
            print(f"❌ Sticky Finite HMM Regime Discovery failed:")
            print(f"   Error: {results.get('error', 'Unknown error')}")
            return 1
            
    except KeyboardInterrupt:
        print("⚠️ Execution interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    # Run the async main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
