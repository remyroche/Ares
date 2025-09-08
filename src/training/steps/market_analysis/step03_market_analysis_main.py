#!/usr/bin/env python3
"""Step 3: Market Analysis Pipeline.

This module provides the main interface for market analysis with:
1. HMM regime discovery and clustering
2. Regime data splitting and labeling
3. Feature engineering and selection
4. Advanced matrix operations
"""

import asyncio
import sys
from pathlib import Path
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path (workspace root so `src.*` imports resolve)
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.steps.market_analysis.enhanced_market_analysis_orchestrator import (
    run_enhanced_market_analysis_pipeline,
    MarketAnalysisPipelineOrchestrator,
)
from src.training.steps.market_analysis.enhanced_logging_metrics import enhanced_logger
from src.training.steps.market_analysis.progress_monitor import progress_monitor

from src.training.reports import save_training_report

def analyze_hmm_clustering_results(symbol: str, exchange: str, timeframe: str) -> dict:
    """Analyze HMM clustering results and return comprehensive summary."""
    try:
        # Load HMM composite metadata
        meta_file = Path("data/training") / f"BINANCE_{symbol}_hmm_composite_meta_{timeframe}.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"HMM metadata file not found: {meta_file}")

        with open(meta_file, 'r') as f:
            meta_data = json.load(f)

        # Load HMM block states
        block_states_file = Path("data/training") / f"BINANCE_{symbol}_hmm_block_states_{timeframe}.parquet"
        if block_states_file.exists():
            block_states_df = pd.read_parquet(block_states_file)
        else:
            block_states_df = None

        # Load HMM composite clusters
        clusters_file = Path("data/training") / f"BINANCE_{symbol}_hmm_composite_clusters_{timeframe}.parquet"
        if clusters_file.exists():
            clusters_df = pd.read_parquet(clusters_file)
        else:
            clusters_df = None

        # Analyze HMM blocks
        blocks_analysis = {}
        for block in meta_data.get("blocks", []):
            block_name = block["name"]
            n_states = block["n_states"]
            blocks_analysis[block_name] = {
                "n_states": n_states,
                "state_medians": meta_data.get("state_feature_medians", {}).get(block_name, {}),
                "state_names": meta_data.get("state_names", {}).get(block_name, {})
            }

        # Analyze cluster centroids
        centroids = meta_data.get("cluster_centroids", {})
        cluster_analysis = {
            "n_clusters": len(centroids),
            "cluster_sizes": {f"cluster_{i}": len(centroids.get(str(i), [])) for i in range(len(centroids))},
            "centroids_summary": {
                f"cluster_{i}": {
                    "size": len(centroids.get(str(i), [])),
                    "mean_value": np.mean(centroids.get(str(i), [0])) if centroids.get(str(i)) else 0
                } for i in range(len(centroids))
            }
        }

        # Analyze regime combinations
        combinations = meta_data.get("combination_counts", {})
        top_combinations = sorted(combinations.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "blocks_analysis": blocks_analysis,
            "cluster_analysis": cluster_analysis,
            "regime_combinations": {
                "total_combinations": len(combinations),
                "top_combinations": top_combinations,
                "most_common_regime": top_combinations[0][0] if top_combinations else None
            },
            "data_availability": {
                "block_states_available": block_states_df is not None,
                "clusters_available": clusters_df is not None,
                "metadata_available": True
            },
            "summary": {
                "total_regime_blocks": len(meta_data.get("blocks", [])),
                "total_clusters": len(centroids),
                "total_regime_combinations": len(combinations)
            }
        }

    except Exception as e:
        raise RuntimeError(f"Failed to analyze HMM clustering results: {str(e)}") from e

def analyze_regime_discovery_statistics(symbol: str, exchange: str, timeframe: str) -> dict:
    """Analyze regime discovery statistics."""
    try:
        # Load HMM metadata
        meta_file = Path("data/training") / f"BINANCE_{symbol}_hmm_composite_meta_{timeframe}.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"HMM metadata file not found: {meta_file}")

        with open(meta_file, 'r') as f:
            meta_data = json.load(f)

        # Load labeled data for regime statistics
        labeled_files = list(Path("data/training").glob(f"BINANCE_{symbol}_labeled_{timeframe}_*.parquet"))
        regime_stats = {}

        for file_path in labeled_files:
            try:
                df = pd.read_parquet(file_path)
                if 'regime' in df.columns:
                    regime_counts = df['regime'].value_counts().to_dict()
                    regime_percentages = (df['regime'].value_counts(normalize=True) * 100).to_dict()

                    # Calculate regime persistence
                    regime_changes = (df['regime'] != df['regime'].shift(1)).sum()
                    total_periods = len(df)
                    persistence_rate = (total_periods - regime_changes) / total_periods * 100

                    regime_stats[str(file_path.stem)] = {
                        "total_samples": len(df),
                        "unique_regimes": len(regime_counts),
                        "regime_distribution": regime_counts,
                        "regime_percentages": regime_percentages,
                        "persistence_rate": persistence_rate,
                        "avg_regime_duration": total_periods / regime_changes if regime_changes > 0 else total_periods
                    }
            except Exception as e:
                raise RuntimeError(f"Failed to process labeled data file {file_path}: {str(e)}") from e

        # Analyze regime transitions
        combinations = meta_data.get("combination_counts", {})
        total_combinations = sum(combinations.values())

        return {
            "regime_statistics": regime_stats,
            "regime_transition_analysis": {
                "total_regime_combinations": len(combinations),
                "total_transition_events": total_combinations,
                "most_stable_regime": max(combinations.items(), key=lambda x: x[1])[0] if combinations else None,
                "regime_stability_score": len(combinations) / total_combinations if total_combinations > 0 else 0
            },
            "summary": {
                "datasets_analyzed": len(regime_stats),
                "total_regime_types": len(meta_data.get("state_names", {}).get("momentum", {})),
                "regime_detection_success": len(regime_stats) > 0
            }
        }

    except Exception as e:
        raise RuntimeError(f"Failed to analyze regime discovery statistics: {str(e)}") from e

def analyze_feature_engineering_metrics(symbol: str, exchange: str, timeframe: str) -> dict:
    """Analyze feature engineering metrics."""
    try:
        # Load vectorized features
        vectorized_file = Path("data/training") / f"BINANCE_{symbol}_{timeframe}_vectorized_feature_pre_optimization.json"
        if not vectorized_file.exists():
            raise FileNotFoundError(f"Vectorized features file not found: {vectorized_file}")

        with open(vectorized_file, 'r') as f:
            features_data = json.load(f)

        # Analyze feature statistics
        feature_stats = {}
        if "features" in features_data:
            features = features_data["features"]
            feature_stats = {
                "total_features": len(features),
                "feature_types": {},
                "feature_categories": set()
            }

            for feature_name in features:
                # Categorize features
                if "_momentum" in feature_name or "_trend" in feature_name:
                    category = "momentum"
                elif "_volatility" in feature_name or "_std" in feature_name:
                    category = "volatility"
                elif "_volume" in feature_name or "_liquidity" in feature_name:
                    category = "volume"
                elif "_rsi" in feature_name or "_macd" in feature_name:
                    category = "technical"
                else:
                    category = "other"

                feature_stats["feature_categories"].add(category)
                if category not in feature_stats["feature_types"]:
                    feature_stats["feature_types"][category] = 0
                feature_stats["feature_types"][category] += 1

            feature_stats["feature_categories"] = list(feature_stats["feature_categories"])

        # Load precomputed features metadata
        precomputed_dir = Path("data/precomputed_features")
        precomputed_stats = {}
        if precomputed_dir.exists():
            json_files = list(precomputed_dir.glob("*.json"))
            precomputed_stats = {
                "total_precomputed_files": len(json_files),
                "feature_computation_status": "available" if json_files else "not_available"
            }

        return {
            "feature_statistics": feature_stats,
            "precomputed_features": precomputed_stats,
            "feature_engineering_summary": {
                "feature_generation_success": len(feature_stats) > 0,
                "precomputed_available": precomputed_stats.get("total_precomputed_files", 0) > 0,
                "feature_diversity_score": len(feature_stats.get("feature_types", {}))
            }
        }

    except Exception as e:
        raise RuntimeError(f"Failed to analyze feature engineering metrics: {str(e)}") from e

def analyze_matrix_operations_performance(symbol: str, exchange: str, timeframe: str) -> dict:
    """Analyze matrix operations performance."""
    try:
        # Check for matrix operation artifacts
        matrix_artifacts = list(Path("data/training").glob(f"**/*matrix*.json"))
        matrix_artifacts += list(Path("data/training").glob(f"**/*matrix*.parquet"))

        # Analyze wavelet cache for matrix operations
        wavelet_dir = Path("data/wavelet_cache")
        wavelet_stats = {}
        if wavelet_dir.exists():
            wavelet_stats = {
                "wavelet_cache_available": True,
                "cache_files": len(list(wavelet_dir.rglob("*.json"))),
                "feature_files": len(list(wavelet_dir.glob("features/**/*.json"))),
                "metadata_files": len(list(wavelet_dir.glob("metadata/**/*.json")))
            }

        # Check for optimized features
        optimized_features = list(Path("data/training").glob(f"**/*optimized*.json"))
        optimized_features += list(Path("data/training").glob(f"**/*optimized*.parquet"))

        return {
            "matrix_operations_artifacts": {
                "total_artifacts": len(matrix_artifacts),
                "artifact_types": [str(f.suffix) for f in matrix_artifacts],
                "available": len(matrix_artifacts) > 0
            },
            "wavelet_transformations": wavelet_stats,
            "optimized_features": {
                "total_optimized_files": len(optimized_features),
                "optimization_available": len(optimized_features) > 0
            },
            "performance_summary": {
                "matrix_operations_completed": len(matrix_artifacts) > 0,
                "wavelet_processing_available": wavelet_stats.get("wavelet_cache_available", False),
                "optimization_performed": len(optimized_features) > 0
            }
        }

    except Exception as e:
        raise RuntimeError(f"Failed to analyze matrix operations performance: {str(e)}") from e

def generate_comprehensive_report(symbol: str, exchange: str, timeframe: str, execution_time: float, correlation_id: str) -> dict:
    """Generate comprehensive market analysis report."""
    try:
        # Collect all analysis results
        hmm_results = analyze_hmm_clustering_results(symbol, exchange, timeframe)
        regime_results = analyze_regime_discovery_statistics(symbol, exchange, timeframe)
        feature_results = analyze_feature_engineering_metrics(symbol, exchange, timeframe)
        matrix_results = analyze_matrix_operations_performance(symbol, exchange, timeframe)

        # Generate summary statistics
        summary = {
            "execution_info": {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "execution_time_seconds": execution_time,
                "correlation_id": correlation_id,
                "timestamp": datetime.now().isoformat()
            },
            "pipeline_components": {
                "hmm_clustering": "completed" if not hmm_results.get("error") else "failed",
                "regime_discovery": "completed" if not regime_results.get("error") else "failed",
                "feature_engineering": "completed" if not feature_results.get("error") else "failed",
                "matrix_operations": "completed" if not matrix_results.get("error") else "failed"
            },
            "data_quality_metrics": {
                "hmm_data_available": hmm_results.get("data_availability", {}).get("metadata_available", False),
                "regime_data_processed": regime_results.get("summary", {}).get("datasets_analyzed", 0),
                "features_generated": feature_results.get("feature_statistics", {}).get("total_features", 0),
                "matrix_operations_performed": matrix_results.get("performance_summary", {}).get("matrix_operations_completed", False)
            }
        }

        return {
            "summary": summary,
            "hmm_clustering_analysis": hmm_results,
            "regime_discovery_analysis": regime_results,
            "feature_engineering_analysis": feature_results,
            "matrix_operations_analysis": matrix_results,
            "recommendations": generate_recommendations(hmm_results, regime_results, feature_results, matrix_results)
        }

    except Exception as e:
        raise RuntimeError(f"Failed to generate comprehensive report: {str(e)}") from e

def generate_recommendations(hmm_results, regime_results, feature_results, matrix_results) -> list:
    """Generate actionable recommendations based on analysis results."""
    recommendations = []

    # HMM Clustering recommendations
    n_clusters = hmm_results.get("cluster_analysis", {}).get("n_clusters", 0)
    if n_clusters > 0:
        recommendations.append(f"✅ HMM clustering successful with {n_clusters} clusters identified")
        if n_clusters < 5:
            recommendations.append("⚠️ Consider increasing number of clusters for better regime granularity")
    else:
        recommendations.append("⚠️ No clusters found - review HMM parameters")

    # Regime discovery recommendations
    datasets = regime_results.get("summary", {}).get("datasets_analyzed", 0)
    if datasets > 0:
        recommendations.append(f"✅ Regime discovery successful - {datasets} datasets processed")
    else:
        recommendations.append("⚠️ No regime datasets found - verify data splitting")

    # Feature engineering recommendations
    total_features = feature_results.get("feature_statistics", {}).get("total_features", 0)
    if total_features > 0:
        recommendations.append(f"✅ Feature engineering successful - {total_features} features generated")
        if total_features < 50:
            recommendations.append("⚠️ Limited feature set - consider adding more technical indicators")
    else:
        recommendations.append("⚠️ No features generated - review feature engineering configuration")

    # Matrix operations recommendations
    if matrix_results.get("performance_summary", {}).get("matrix_operations_completed", False):
        recommendations.append("✅ Matrix operations completed successfully")
    else:
        recommendations.append("⚠️ Matrix operations not fully completed - review wavelet processing")

    return recommendations

async def main():
    """Main function to run market analysis pipeline with enhanced logging."""
    # Configuration
    symbol = "ETHUSDT"
    exchange = "BINANCE"
    timeframe = "1m"
    data_dir = "data_cache"
    
    # Market analysis parameters
    config = {
        'force_rerun': True,
        'hmm_clustering': True,
        'regime_splitting': True,
        'feature_engineering': True,
        'matrix_operations': True,
        'feature_selection': True,
        'random_state': 42,
    }
    
    # Start enhanced logging
    correlation_id = f"market_analysis_{symbol}_{exchange}_{int(time.time())}"
    enhanced_logger.start_pipeline(symbol, exchange, correlation_id)
    
    enhanced_logger.logger.info("▶️ Step 3: Market Analysis Pipeline")
    enhanced_logger.logger.info("=" * 80)
    enhanced_logger.logger.info(f"⚙️ Configuration:")
    enhanced_logger.logger.info(f"   Symbol: {symbol}")
    enhanced_logger.logger.info(f"   Exchange: {exchange}")
    enhanced_logger.logger.info(f"   Timeframe: {timeframe}")
    enhanced_logger.logger.info(f"   Data directory: {data_dir}")
    enhanced_logger.logger.info(f"   HMM clustering: {config['hmm_clustering']}")
    enhanced_logger.logger.info(f"   Regime splitting: {config['regime_splitting']}")
    enhanced_logger.logger.info(f"   Feature engineering: {config['feature_engineering']}")
    enhanced_logger.logger.info("=" * 80)
    
    # Run market analysis pipeline
    start_time = time.time()
    
    try:
        # Use enhanced market analysis pipeline with comprehensive validation
        success = await run_enhanced_market_analysis_pipeline(
            symbol = symbol,
            exchange = exchange,
            timeframe = timeframe,
            data_dir = data_dir,
            **config
        )
        
        total_time = time.time() - start_time
        
        if success:
            enhanced_logger.logger.info("\n✓ MARKET ANALYSIS COMPLETED")
            enhanced_logger.logger.info("=" * 80)
            enhanced_logger.logger.info("✓ All market analysis steps completed:")
            enhanced_logger.logger.info("   ✓ HMM regime discovery and clustering")
            enhanced_logger.logger.info("   ✓ Regime data splitting and labeling")
            enhanced_logger.logger.info("   ✓ Feature engineering and selection")
            enhanced_logger.logger.info("   ✓ Advanced matrix operations")
            enhanced_logger.logger.info(f"⏱️ Total execution time: {total_time:.2f} seconds")
            enhanced_logger.logger.info("=" * 80)

            # Generate comprehensive analysis reports
            enhanced_logger.logger.info("📊 Generating comprehensive analysis reports...")
            comprehensive_report = generate_comprehensive_report(symbol, exchange, timeframe, total_time, correlation_id)

            if not comprehensive_report.get("error"):
                # Save comprehensive report using centralized system
                report_file = save_training_report(
                    data=comprehensive_report,
                    step_name="step03_market_analysis",
                    report_type="comprehensive_report",
                    symbol=symbol,
                    timeframe=timeframe,
                    file_format="json"
                )
                enhanced_logger.logger.info(f"💾 Comprehensive report saved to: {report_file}")

                # Generate summary report for console
                summary = comprehensive_report.get("summary", {})
                recommendations = comprehensive_report.get("recommendations", [])

                enhanced_logger.logger.info("\n📈 MARKET ANALYSIS SUMMARY")
                enhanced_logger.logger.info("=" * 80)
                enhanced_logger.logger.info(f"🎯 Symbol: {summary.get('execution_info', {}).get('symbol', 'N/A')}")
                enhanced_logger.logger.info(f"🏢 Exchange: {summary.get('execution_info', {}).get('exchange', 'N/A')}")
                enhanced_logger.logger.info(f"📊 Timeframe: {summary.get('execution_info', {}).get('timeframe', 'N/A')}")
                enhanced_logger.logger.info(f"⏱️ Execution Time: {total_time:.2f} seconds")
                enhanced_logger.logger.info("=" * 80)

                # Pipeline components status
                pipeline_components = summary.get("pipeline_components", {})
                enhanced_logger.logger.info("🔧 Pipeline Components Status:")
                for component, status in pipeline_components.items():
                    status_icon = "✅" if status == "completed" else "❌"
                    enhanced_logger.logger.info(f"   {status_icon} {component.replace('_', ' ').title()}: {status}")

                # Data quality metrics
                data_metrics = summary.get("data_quality_metrics", {})
                enhanced_logger.logger.info("\n📊 Data Quality Metrics:")
                enhanced_logger.logger.info(f"   📈 HMM Data Available: {data_metrics.get('hmm_data_available', False)}")
                enhanced_logger.logger.info(f"   🎯 Regime Datasets Processed: {data_metrics.get('regime_data_processed', 0)}")
                enhanced_logger.logger.info(f"   🔧 Features Generated: {data_metrics.get('features_generated', 0)}")
                enhanced_logger.logger.info(f"   ⚡ Matrix Operations: {data_metrics.get('matrix_operations_performed', False)}")

                # Recommendations
                if recommendations:
                    enhanced_logger.logger.info("\n💡 Recommendations:")
                    for rec in recommendations:
                        enhanced_logger.logger.info(f"   {rec}")

                enhanced_logger.logger.info("=" * 80)

            else:
                raise RuntimeError(f"Could not generate comprehensive report: {comprehensive_report.get('error')}")

            # Save configuration for future reference
            config_file = Path(data_dir) / f"market_analysis_config_{symbol}_{timeframe}.json"
            with open(config_file, 'w') as f:
                json.dump({
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'config': config,
                    'execution_time': total_time,
                    'success': True,
                    'correlation_id': correlation_id,
                    'comprehensive_report_generated': not comprehensive_report.get("error", False)
                }, f, indent = 2)

            enhanced_logger.logger.info(f"💾 Configuration saved to: {config_file}")

        else:
            enhanced_logger.logger.error("\n❌ MARKET ANALYSIS FAILED!")
            enhanced_logger.logger.error("=" * 80)
            enhanced_logger.logger.error("❌ Please check the logs for error details")
            enhanced_logger.logger.error(f"⏱️ Total execution time: {total_time:.2f} seconds")
            enhanced_logger.logger.error("=" * 80)
            
    except Exception as e:
        total_time = time.time() - start_time
        error_message = str(e)
        enhanced_logger.logger.error(f"\n❌ MARKET ANALYSIS FAILED WITH EXCEPTION: {error_message}")
        enhanced_logger.logger.error("=" * 80)
        enhanced_logger.logger.error(f"⏱️ Total execution time: {total_time:.2f} seconds")
        enhanced_logger.logger.error("=" * 80)
        
        # End enhanced logging and progress monitoring with failure
        progress_monitor.stop_monitoring()
        enhanced_logger.end_pipeline(success = False, error_message = error_message)
        raise

if __name__ == "__main__":
    # Run the market analysis pipeline
    asyncio.run(main())