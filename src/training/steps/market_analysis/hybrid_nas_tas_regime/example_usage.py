"""
Example usage of the Hybrid NAS-TAS Regime Detection system.

This script demonstrates how to use the hybrid orchestrator to perform
comprehensive regime detection using both NAS and TAS approaches.

The system uses the same data source as hmm_regime_discovery.py (klines_parquet)
but operates independently, and delivers similar outputs to hmm_clustering
but with enhanced hybrid metrics.
"""

import asyncio
import logging
from datetime import datetime

from .hybrid_orchestrator import HybridOrchestrator, HybridOrchestratorConfig
from .shared_utils.metrics_reporting import MetricsReporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main example function demonstrating hybrid NAS-TAS regime detection."""
    
    print("🚀 Hybrid NAS-TAS Regime Detection Example")
    print("=" * 50)
    
    # Configuration
    config = HybridOrchestratorConfig(
        symbol="BTCUSDT",
        timeframe="15m",
        start_date="2023-01-01",
        end_date="2023-12-31",
        
        # Feature collection settings
        use_standardized_features=True,
        feature_categories=['momentum', 'volatility', 'volume', 'trend'],
        
        # Economic significance settings
        significance_threshold=0.5,
        min_regime_duration=10,
        
        # Trading viability settings
        viability_threshold=0.5,
        minimum_regime_duration=5,
        
        # Search strategy settings
        max_iterations=50,
        use_bayesian_optimization=True,
        
        # Evolutionary algorithm settings
        population_size=50,
        max_generations=25,
        use_nsga2=True,
        use_spea2=True,
        
        # Hardware optimization settings
        use_gpu_acceleration=True,
        memory_limit_gb=8.0,
        
        # Metrics reporting settings
        include_detailed_metrics=True,
        save_to_file=True
    )
    
    try:
        # Create hybrid orchestrator
        print("🔧 Initializing Hybrid NAS-TAS Orchestrator...")
        orchestrator = HybridOrchestrator(config)
        
        # Check pipeline status
        status = orchestrator.get_pipeline_status()
        print(f"✅ Pipeline Status: {status['orchestrator_active']}")
        
        # Execute hybrid pipeline
        print("\n🚀 Executing Hybrid NAS-TAS Pipeline...")
        print("-" * 40)
        
        consolidated_report = await orchestrator.execute_hybrid_pipeline()
        
        if consolidated_report.success:
            print("\n✅ Pipeline Execution Completed Successfully!")
            print("=" * 50)
            
            # Display summary
            print(f"📊 Execution Time: {consolidated_report.execution_time:.2f} seconds")
            print(f"🧠 NAS Regime Count: {consolidated_report.nas_metrics.get('regime_count', 0)}")
            print(f"🌳 TAS Regime Count: {consolidated_report.tas_metrics.get('regime_count', 0)}")
            print(f"🔄 Hybrid Regime Count: {consolidated_report.hybrid_metrics.get('consolidated_regime_count', 0)}")
            
            # Display clustering quality
            nas_silhouette = consolidated_report.nas_metrics.get('clustering_quality', {}).get('silhouette_score', 0.0)
            tas_silhouette = consolidated_report.tas_metrics.get('clustering_quality', {}).get('silhouette_score', 0.0)
            hybrid_silhouette = consolidated_report.hybrid_metrics.get('consolidation_quality', {}).get('silhouette_score', 0.0)
            
            print(f"\n📈 Clustering Quality (Silhouette Score):")
            print(f"   NAS: {nas_silhouette:.3f}")
            print(f"   TAS: {tas_silhouette:.3f}")
            print(f"   Hybrid: {hybrid_silhouette:.3f}")
            
            # Display economic significance
            nas_economic = consolidated_report.nas_metrics.get('economic_significance', {}).get('overall_score', 0.0)
            tas_economic = consolidated_report.tas_metrics.get('economic_significance', {}).get('overall_score', 0.0)
            hybrid_economic = consolidated_report.hybrid_metrics.get('consensus_metrics', {}).get('economic_consensus_score', 0.0)
            
            print(f"\n💰 Economic Significance:")
            print(f"   NAS: {nas_economic:.3f}")
            print(f"   TAS: {tas_economic:.3f}")
            print(f"   Hybrid: {hybrid_economic:.3f}")
            
            # Display trading viability
            nas_trading = consolidated_report.nas_metrics.get('trading_viability', {}).get('overall_score', 0.0)
            tas_trading = consolidated_report.tas_metrics.get('trading_viability', {}).get('overall_score', 0.0)
            hybrid_trading = consolidated_report.hybrid_metrics.get('consensus_metrics', {}).get('trading_consensus_score', 0.0)
            
            print(f"\n📈 Trading Viability:")
            print(f"   NAS: {nas_trading:.3f}")
            print(f"   TAS: {tas_trading:.3f}")
            print(f"   Hybrid: {hybrid_trading:.3f}")
            
            # Display recommendations
            economic_recommendation = consolidated_report.economic_summary.get('recommended_system', 'Unknown')
            trading_recommendation = consolidated_report.trading_summary.get('recommended_system', 'Unknown')
            
            print(f"\n🎯 Recommendations:")
            print(f"   Economic Analysis: {economic_recommendation}")
            print(f"   Trading Decisions: {trading_recommendation}")
            
            # Generate and display summary report
            print(f"\n📄 Generating Summary Report...")
            metrics_reporter = MetricsReporter(consolidated_report.report_metadata.get('config', {}))
            summary_report = metrics_reporter.generate_summary_report(consolidated_report)
            print(summary_report)
            
        else:
            print(f"\n❌ Pipeline Execution Failed!")
            print(f"Error: {consolidated_report.error_message}")
            
    except Exception as e:
        logger.error(f"❌ Example execution failed: {e}")
        print(f"\n❌ Error: {e}")


def run_example():
    """Run the example with proper async handling."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ Example interrupted by user")
    except Exception as e:
        print(f"\n❌ Example failed: {e}")


if __name__ == "__main__":
    run_example()