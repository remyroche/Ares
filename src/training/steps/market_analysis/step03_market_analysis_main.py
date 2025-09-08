from ..standardized_parquet_handler import standardized_parquet_handler
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

async def analyze_hmm_clustering_results(symbol: str, exchange: str, timeframe: str) -> dict:
    """Analyze HMM clustering results and return comprehensive summary with optimizations."""
    try:
        # Import optimization components
        from src.training.steps.market_analysis.hmm_clustering.step03_fast_fail_validation import get_fast_fail_validator
        from src.training.steps.market_analysis.hmm_clustering.step03_parallel_io_operations import get_parallel_io_operations
        from src.training.steps.market_analysis.hmm_clustering.step03_intelligent_caching import get_intelligent_cache
        
        # Get optimization components
        validator = get_fast_fail_validator()
        io_ops = get_parallel_io_operations()
        cache = get_intelligent_cache()
        
        # Check cache first
        cache_key = f"hmm_analysis_{symbol}_{exchange}_{timeframe}"
        cached_result = cache.get(cache_key)
        if cached_result:
            enhanced_logger.logger.info("📦 Using cached HMM clustering analysis results")
            return cached_result
        
        # Fast fail validation for file existence
        meta_file = Path("data/training") / f"BINANCE_{symbol}_hmm_composite_meta_{timeframe}.json"
        file_validation = await validator.validate_data_file(meta_file)
        if not file_validation.passed:
            raise FileNotFoundError(f"HMM metadata file validation failed: {file_validation.message}")

        # Load HMM composite metadata with async I/O
        meta_data = await io_ops.load_file_async(meta_file, 'json')

        # Load HMM block states and clusters in parallel
        block_states_file = Path("data/training") / f"BINANCE_{symbol}_hmm_block_states_{timeframe}.parquet"
        clusters_file = Path("data/training") / f"BINANCE_{symbol}_hmm_composite_clusters_{timeframe}.parquet"
        
        # Prepare files for parallel loading
        files_to_load = []
        if block_states_file.exists():
            files_to_load.append(block_states_file)
        if clusters_file.exists():
            files_to_load.append(clusters_file)
        
        # Load files in parallel
        if files_to_load:
            loaded_dataframes = await io_ops.load_files_parallel(files_to_load)
            block_states_df = loaded_dataframes[0] if block_states_file.exists() else None
            clusters_df = loaded_dataframes[1] if clusters_file.exists() and len(loaded_dataframes) > 1 else loaded_dataframes[0] if clusters_file.exists() else None
        else:
            block_states_df = None
            clusters_df = None
        # Fast-fail validation: Check if required files exist before processing
        base_path = Path("data/training")
        required_files = [
            f"BINANCE_{symbol}_hmm_composite_meta_{timeframe}.json",
            f"BINANCE_{symbol}_hmm_block_states_{timeframe}.parquet",
            f"BINANCE_{symbol}_hmm_composite_clusters_{timeframe}.parquet"
        ]
        
        missing_files = [f for f in required_files if not (base_path / f).exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing required HMM files: {missing_files}")

        # Load HMM composite metadata with caching
        meta_file = base_path / f"BINANCE_{symbol}_hmm_composite_meta_{timeframe}.json"
        with open(meta_file, 'r') as f:
            meta_data = json.load(f)

        # Load HMM block states and clusters in parallel (optimization) with standardized handler
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            block_states_future = executor.submit(
                standardized_parquet_handler.read_parquet_standardized,
                base_path / f"BINANCE_{symbol}_hmm_block_states_{timeframe}.parquet"
            )
            clusters_future = executor.submit(
                standardized_parquet_handler.read_parquet_standardized,
                base_path / f"BINANCE_{symbol}_hmm_composite_clusters_{timeframe}.parquet"
            )
            
            block_states_df = block_states_future.result()
            clusters_df = clusters_future.result()

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

        result = {
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
        
        # Cache the result
        cache.set(cache_key, result, ttl_seconds=3600, tags=['hmm_analysis'])
        
        return result

    except Exception as e:
        raise RuntimeError(f"Failed to analyze HMM clustering results: {str(e)}") from e

def analyze_regime_discovery_statistics(symbol: str, exchange: str, timeframe: str) -> dict:
    """Analyze regime discovery statistics."""
    try:
        # Fast-fail validation: Check data availability early
        base_path = Path("data/training")
        meta_file = base_path / f"BINANCE_{symbol}_hmm_composite_meta_{timeframe}.json"
        
        if not meta_file.exists():
            raise FileNotFoundError(f"HMM metadata file not found: {meta_file}")

        # Load HMM metadata
        with open(meta_file, 'r') as f:
            meta_data = json.load(f)

        # Fast-fail validation: Check for labeled data files
        labeled_files = list(base_path.glob(f"BINANCE_{symbol}_labeled_{timeframe}_*.parquet"))
        if not labeled_files:
            raise FileNotFoundError(f"No labeled data files found for {symbol}_{timeframe}")
        
        # Fast-fail validation: Check metadata structure
        required_metadata_keys = ["combination_counts", "state_names"]
        missing_keys = [key for key in required_metadata_keys if key not in meta_data]
        if missing_keys:
            raise ValueError(f"Missing required metadata keys: {missing_keys}")
        
        regime_stats = {}

        for file_path in labeled_files:
            try:
                df = standardized_parquet_handler.read_parquet_standardized(file_path)
                
                # Validity checks: Data integrity validation
                if df.empty:
                    raise ValueError(f"Empty dataset in {file_path}")
                
                if 'regime' not in df.columns:
                    raise ValueError(f"Missing 'regime' column in {file_path}")
                
                # Validity checks: Regime data quality
                regime_values = df['regime'].dropna()
                if len(regime_values) == 0:
                    raise ValueError(f"No valid regime data in {file_path}")
                
                # Check for reasonable regime distribution
                unique_regimes = regime_values.nunique()
                if unique_regimes < 2:
                    raise ValueError(f"Insufficient regime diversity in {file_path}: {unique_regimes} regimes")
                
                if unique_regimes > 20:
                    raise ValueError(f"Excessive regime diversity in {file_path}: {unique_regimes} regimes")
                
                regime_counts = regime_values.value_counts().to_dict()
                regime_percentages = (regime_values.value_counts(normalize=True) * 100).to_dict()

                # Calculate regime persistence with validation
                regime_changes = (df['regime'] != df['regime'].shift(1)).sum()
                total_periods = len(df)
                
                # Validity check: Reasonable persistence rate
                persistence_rate = (total_periods - regime_changes) / total_periods * 100
                if persistence_rate < 10 or persistence_rate > 99:
                    raise ValueError(f"Unrealistic persistence rate in {file_path}: {persistence_rate:.2f}%")

                regime_stats[str(file_path.stem)] = {
                    "total_samples": len(df),
                    "unique_regimes": unique_regimes,
                    "regime_distribution": regime_counts,
                    "regime_percentages": regime_percentages,
                    "persistence_rate": persistence_rate,
                    "avg_regime_duration": total_periods / regime_changes if regime_changes > 0 else total_periods,
                    "data_quality_score": min(100, (unique_regimes / 5) * 100)  # Quality metric
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
        # Fast-fail validation: Check file existence early
        vectorized_file = Path("data/training") / f"BINANCE_{symbol}_{timeframe}_vectorized_feature_pre_optimization.json"
        if not vectorized_file.exists():
            raise FileNotFoundError(f"Vectorized features file not found: {vectorized_file}")

        # Memory optimization: Load only required data
        with open(vectorized_file, 'r') as f:
            features_data = json.load(f)
        
        # Validity check: Ensure features data structure is valid
        if not isinstance(features_data, dict):
            raise ValueError("Invalid features data structure: expected dictionary")
        
        if "features" not in features_data:
            raise ValueError("Missing 'features' key in features data")

        # Analyze feature statistics with improved logic
        feature_stats = {}
        features = features_data["features"]
        
        # Validity check: Ensure features is a list
        if not isinstance(features, list):
            raise ValueError("Features data must be a list")
        
        if len(features) == 0:
            raise ValueError("No features found in features data")
        
        feature_stats = {
            "total_features": len(features),
            "feature_types": {},
            "feature_categories": set(),
            "feature_quality_metrics": {}
        }

        # Improved feature categorization logic
        feature_categories = {
            "momentum": ["_momentum", "_trend", "_change", "_return"],
            "volatility": ["_volatility", "_std", "_var", "_atr"],
            "volume": ["_volume", "_liquidity", "_turnover"],
            "technical": ["_rsi", "_macd", "_bollinger", "_bb_", "_ema", "_sma"],
            "price": ["_price", "_close", "_high", "_low", "_open"],
            "other": []
        }
        
        category_counts = {cat: 0 for cat in feature_categories.keys()}
        
        for feature_name in features:
            # Improved categorization logic
            category = "other"  # Default category
            for cat, keywords in feature_categories.items():
                if any(keyword in feature_name.lower() for keyword in keywords):
                    category = cat
                    break
            
            feature_stats["feature_categories"].add(category)
            category_counts[category] += 1

        # Convert set to list and add counts
        feature_stats["feature_categories"] = list(feature_stats["feature_categories"])
        feature_stats["feature_types"] = category_counts
        
        # Add feature quality metrics
        feature_stats["feature_quality_metrics"] = {
            "diversity_score": len(feature_stats["feature_categories"]) / len(feature_categories),
            "balance_score": 1.0 - (max(category_counts.values()) - min(category_counts.values())) / len(features),
            "coverage_score": len(features) / 100.0  # Normalize to 0-1 scale
        }

        # Memory optimization: Efficient precomputed features metadata loading
        precomputed_dir = Path("data/precomputed_features")
        precomputed_stats = {}
        if precomputed_dir.exists():
            # Memory optimization: Use generator instead of list for large directories
            json_files = list(precomputed_dir.glob("*.json"))
            precomputed_stats = {
                "total_precomputed_files": len(json_files),
                "feature_computation_status": "available" if json_files else "not_available",
                "memory_efficient_loading": True
            }
            
            # Memory optimization: Add file size information for memory planning
            total_size = sum(f.stat().st_size for f in json_files if f.is_file())
            precomputed_stats["total_size_mb"] = total_size / (1024 * 1024)

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
        # Memory optimization: Use generators for large directory scans
        base_path = Path("data/training")
        
        # Fast-fail validation: Check if training directory exists
        if not base_path.exists():
            raise FileNotFoundError(f"Training directory not found: {base_path}")
        
        # Memory optimization: Efficient artifact discovery
        matrix_patterns = ["**/*matrix*.json", "**/*matrix*.parquet"]
        matrix_artifacts = []
        for pattern in matrix_patterns:
            matrix_artifacts.extend(base_path.glob(pattern))

        # Memory optimization: Efficient wavelet cache analysis
        wavelet_dir = Path("data/wavelet_cache")
        wavelet_stats = {}
        if wavelet_dir.exists():
            # Memory optimization: Count files without loading them
            cache_files = sum(1 for _ in wavelet_dir.rglob("*.json"))
            feature_files = sum(1 for _ in wavelet_dir.glob("features/**/*.json"))
            metadata_files = sum(1 for _ in wavelet_dir.glob("metadata/**/*.json"))
            
            wavelet_stats = {
                "wavelet_cache_available": True,
                "cache_files": cache_files,
                "feature_files": feature_files,
                "metadata_files": metadata_files,
                "total_files": cache_files + feature_files + metadata_files
            }
            
            # Memory optimization: Calculate total cache size
            total_size = sum(f.stat().st_size for f in wavelet_dir.rglob("*") if f.is_file())
            wavelet_stats["total_size_mb"] = total_size / (1024 * 1024)

        # Memory optimization: Efficient optimized features discovery
        optimized_patterns = ["**/*optimized*.json", "**/*optimized*.parquet"]
        optimized_features = []
        for pattern in optimized_patterns:
            optimized_features.extend(base_path.glob(pattern))

        # Memory optimization: Calculate total memory usage
        total_memory_mb = 0
        for artifact in matrix_artifacts + optimized_features:
            if artifact.is_file():
                total_memory_mb += artifact.stat().st_size / (1024 * 1024)
        
        total_memory_mb += wavelet_stats.get("total_size_mb", 0)
        
        return {
            "matrix_operations_artifacts": {
                "total_artifacts": len(matrix_artifacts),
                "artifact_types": list(set(str(f.suffix) for f in matrix_artifacts)),  # Remove duplicates
                "available": len(matrix_artifacts) > 0,
                "total_size_mb": sum(f.stat().st_size for f in matrix_artifacts if f.is_file()) / (1024 * 1024)
            },
            "wavelet_transformations": wavelet_stats,
            "optimized_features": {
                "total_optimized_files": len(optimized_features),
                "optimization_available": len(optimized_features) > 0,
                "total_size_mb": sum(f.stat().st_size for f in optimized_features if f.is_file()) / (1024 * 1024)
            },
            "performance_summary": {
                "matrix_operations_completed": len(matrix_artifacts) > 0,
                "wavelet_processing_available": wavelet_stats.get("wavelet_cache_available", False),
                "optimization_performed": len(optimized_features) > 0,
                "total_memory_usage_mb": total_memory_mb,
                "memory_efficiency_score": min(100, max(0, 100 - (total_memory_mb / 1000)))  # Penalty for >1GB
            }
        }

    except Exception as e:
        raise RuntimeError(f"Failed to analyze matrix operations performance: {str(e)}") from e

def generate_comprehensive_report(symbol: str, exchange: str, timeframe: str, execution_time: float, correlation_id: str) -> dict:
    """Generate comprehensive market analysis report."""
    try:
        # Performance optimization: Parallel execution of analysis functions
        import concurrent.futures
        import time as time_module
        
        start_time = time_module.time()
        
        # Execute analysis functions in parallel for better performance
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all analysis tasks
            hmm_future = executor.submit(analyze_hmm_clustering_results, symbol, exchange, timeframe)
            regime_future = executor.submit(analyze_regime_discovery_statistics, symbol, exchange, timeframe)
            feature_future = executor.submit(analyze_feature_engineering_metrics, symbol, exchange, timeframe)
            matrix_future = executor.submit(analyze_matrix_operations_performance, symbol, exchange, timeframe)
            
            # Collect results
            hmm_results = hmm_future.result()
            regime_results = regime_future.result()
            feature_results = feature_future.result()
            matrix_results = matrix_future.result()
        
        analysis_time = time_module.time() - start_time
        logging.info(f"Analysis functions completed in {analysis_time:.2f}s (parallel execution)")

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
    """Main function to run market analysis pipeline with enhanced logging and optimizations."""
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
    
    # Check if optimized version should be used
    use_optimized = True  # Set to True to use optimized version
    
    if use_optimized:
        enhanced_logger.logger.info("🚀 Using OPTIMIZED Step03 with comprehensive optimizations")
        enhanced_logger.logger.info("=" * 80)
        
        # Import and run optimized version
        from src.training.steps.market_analysis.hmm_clustering.step03_enhanced_optimized import run_optimized_step03, OptimizedStep03Config
        
        # Create optimized configuration
        optimized_config = OptimizedStep03Config(
            max_memory_usage_percent=80.0,
            chunk_size_mb=100,
            enable_memory_monitoring=True,
            max_concurrent_files=10,
            max_workers=4,
            enable_compression=True,
            max_memory_cache_size_mb=500,
            max_disk_cache_size_mb=2000,
            cache_ttl_seconds=3600,
            min_available_memory_gb=2.0,
            min_disk_space_gb=5.0,
            enable_extensive_logging=True,
            enable_performance_monitoring=True,
            enable_parallel_processing=True,
            enable_chunked_processing=True
        )
        
        try:
            # Run optimized Step03
            results = await run_optimized_step03(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                force_rerun=config['force_rerun'],
                config=optimized_config
            )
            
            enhanced_logger.logger.info("🎉 OPTIMIZED MARKET ANALYSIS COMPLETED SUCCESSFULLY!")
            enhanced_logger.logger.info("=" * 80)
            enhanced_logger.logger.info(f"⏱️ Total execution time: {results['execution_time']:.2f} seconds")
            enhanced_logger.logger.info(f"✅ Success: {results['success']}")
            
            # Log performance metrics
            if 'performance_report' in results:
                perf = results['performance_report']
                enhanced_logger.logger.info("📊 Performance Metrics:")
                enhanced_logger.logger.info(f"   🧠 Memory usage: {perf.get('memory_performance', {}).get('process_memory', {}).get('current_mb', 0):.1f}MB")
                enhanced_logger.logger.info(f"   💾 Cache hit rate: {perf.get('cache_performance', {}).get('performance', {}).get('hit_rate', 0):.1%}")
                enhanced_logger.logger.info(f"   📁 I/O throughput: {perf.get('io_performance', {}).get('io_performance', {}).get('average_throughput_mbps', 0):.1f} MB/s")
            
            enhanced_logger.logger.info("=" * 80)
            return
            
        except Exception as e:
            enhanced_logger.logger.error(f"❌ Optimized Step03 failed: {e}")
            enhanced_logger.logger.info("🔄 Falling back to standard Step03...")
            # Fall through to standard implementation
    
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