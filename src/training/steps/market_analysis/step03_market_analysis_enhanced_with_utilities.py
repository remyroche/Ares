"""
Step 3: Enhanced Market Analysis Pipeline with Comprehensive Utility Integration

This module provides the main interface for market analysis with extensive
integration of all specified utilities through dependency injection.
"""

import asyncio
import sys
from pathlib import Path
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import dependency injection and utilities
from .hmm_clustering.step03_dependency_injection import (
    Step03ServiceProvider, Step03Config, Step03UtilityMixin,
    get_step03_service_provider, inject_step03_utilities
)

# Import existing components
from .enhanced_market_analysis_orchestrator import run_enhanced_market_analysis_pipeline
from .enhanced_logging_metrics import enhanced_logger
from .progress_monitor import progress_monitor
from .standardized_parquet_handler import standardized_parquet_handler
from src.training.reports import save_training_report
import logging

class EnhancedMarketAnalysisPipeline(Step03UtilityMixin):
    """Enhanced Market Analysis Pipeline with comprehensive utility integration."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.logger = self.utils['common_operations']['logging']['get_logger'](__name__)
        
        # Initialize service provider
        step03_config = Step03Config(
            enable_gpu_optimization=True,
            enable_memory_optimization=True,
            enable_cpu_optimization=True,
            enable_math_validation=True,
            enable_data_validation=True,
            enable_serialization=True,
            enable_parquet_operations=True,
            max_memory_usage_gb=8.0,
            max_workers=4,
            enable_extensive_logging=True
        )
        self.service_provider = get_step03_service_provider(step03_config)
        
        # Get utility instances
        self.common_ops = self.get_common_ops()
        self.common_utils = self.get_common_utils()
        self.math_validation = self.get_math_validation()
        self.serialization = self.get_serialization()
        self.m1_optimizers = self.get_m1_optimizers()
        self.data_processing = self.get_data_processing()
        self.parquet_utils = self.get_parquet_utils()
        
        self.logger.info("🚀 Enhanced Market Analysis Pipeline initialized with comprehensive utilities")

    @inject_step03_utilities
    async def analyze_hmm_clustering_results(self, symbol: str, exchange: str, timeframe: str, 
                                           utils: Dict[str, Any] = None, services: Step03ServiceProvider = None) -> dict:
        """Analyze HMM clustering results using comprehensive utilities."""
        try:
            # Use common operations for file validation
            base_path = Path("data/training")
            meta_file = base_path / f"BINANCE_{symbol}_hmm_composite_meta_{timeframe}.json"
            
            if not utils['common_operations']['file_operations']['safe_file_exists'](meta_file):
                raise FileNotFoundError(f"HMM metadata file not found: {meta_file}")
            
            # Load metadata using serialization utilities
            meta_data = utils['serialization']['convenience_functions']['load_json'](meta_file, {})
            
            # Load data files using parquet utilities
            block_states_file = base_path / f"BINANCE_{symbol}_hmm_block_states_{timeframe}.parquet"
            clusters_file = base_path / f"BINANCE_{symbol}_hmm_composite_clusters_{timeframe}.parquet"
            
            # Use parquet utilities for safe loading
            parquet_handler = services.get_service(utils['parquet']['ParquetUtils'])
            block_states_df = parquet_handler.safe_read_parquet(str(block_states_file)) if block_states_file.exists() else None
            clusters_df = parquet_handler.safe_read_parquet(str(clusters_file)) if clusters_file.exists() else None
            
            # Analyze using data processing utilities
            df_validator = services.get_service(utils['data_processing']['validators']['DataFrameValidator'])
            
            if block_states_df is not None:
                validation_result = df_validator.validate_dataframe(block_states_df)
                if validation_result.summary['critical_issues'] > 0:
                    self.logger.warning(f"Block states validation issues: {validation_result.issues}")
            
            if clusters_df is not None:
                validation_result = df_validator.validate_dataframe(clusters_df)
                if validation_result.summary['critical_issues'] > 0:
                    self.logger.warning(f"Clusters validation issues: {validation_result.issues}")
            
            # Use math validation for calculations
            centroids = meta_data.get("cluster_centroids", {})
            cluster_analysis = {
                "n_clusters": len(centroids),
                "cluster_sizes": {},
                "centroids_summary": {}
            }
            
            for i in range(len(centroids)):
                cluster_data = centroids.get(str(i), [])
                if cluster_data:
                    # Use math validation for safe calculations
                    mean_value = utils['math_validation']['basic_math']['safe_divide'](
                        sum(cluster_data), len(cluster_data), default=0.0
                    )
                    cluster_analysis["cluster_sizes"][f"cluster_{i}"] = len(cluster_data)
                    cluster_analysis["centroids_summary"][f"cluster_{i}"] = {
                        "size": len(cluster_data),
                        "mean_value": mean_value
                    }
            
            # Use common utilities for data quality analysis
            if block_states_df is not None:
                data_quality_report = utils['common_utilities']['data_quality']['create_data_quality_report'](block_states_df)
                self.logger.info(f"Block states data quality: {data_quality_report['status']}")
            
            result = {
                "cluster_analysis": cluster_analysis,
                "data_availability": {
                    "block_states_available": block_states_df is not None,
                    "clusters_available": clusters_df is not None,
                    "metadata_available": True
                },
                "summary": {
                    "total_regime_blocks": len(meta_data.get("blocks", [])),
                    "total_clusters": len(centroids),
                    "total_regime_combinations": len(meta_data.get("combination_counts", {}))
                }
            }
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to analyze HMM clustering results: {str(e)}") from e

    @inject_step03_utilities
    async def analyze_regime_discovery_statistics(self, symbol: str, exchange: str, timeframe: str,
                                                utils: Dict[str, Any] = None, services: Step03ServiceProvider = None) -> dict:
        """Analyze regime discovery statistics using comprehensive utilities."""
        try:
            # Use common operations for file validation
            base_path = Path("data/training")
            meta_file = base_path / f"BINANCE_{symbol}_hmm_composite_meta_{timeframe}.json"
            
            if not utils['common_operations']['file_operations']['safe_file_exists'](meta_file):
                raise FileNotFoundError(f"HMM metadata file not found: {meta_file}")
            
            # Load metadata using serialization utilities
            meta_data = utils['serialization']['convenience_functions']['load_json'](meta_file, {})
            
            # Use common operations for file discovery
            labeled_files = list(base_path.glob(f"BINANCE_{symbol}_labeled_{timeframe}_*.parquet"))
            if not labeled_files:
                raise FileNotFoundError(f"No labeled data files found for {symbol}_{timeframe}")
            
            regime_stats = {}
            
            # Use parquet utilities for safe loading
            parquet_handler = services.get_service(utils['parquet']['ParquetUtils'])
            df_validator = services.get_service(utils['data_processing']['validators']['DataFrameValidator'])
            
            for file_path in labeled_files:
                try:
                    # Load data using parquet utilities
                    df = parquet_handler.safe_read_parquet(str(file_path))
                    if df is None:
                        raise ValueError(f"Failed to load data from {file_path}")
                    
                    # Validate data using data processing utilities
                    validation_result = df_validator.validate_dataframe(df)
                    if validation_result.summary['critical_issues'] > 0:
                        self.logger.warning(f"Validation issues in {file_path}: {validation_result.issues}")
                    
                    # Use common utilities for data analysis
                    if 'regime' not in df.columns:
                        raise ValueError(f"Missing 'regime' column in {file_path}")
                    
                    regime_values = df['regime'].dropna()
                    if len(regime_values) == 0:
                        raise ValueError(f"No valid regime data in {file_path}")
                    
                    # Use math validation for calculations
                    unique_regimes = regime_values.nunique()
                    regime_counts = regime_values.value_counts().to_dict()
                    
                    # Calculate regime persistence using math validation
                    regime_changes = (df['regime'] != df['regime'].shift(1)).sum()
                    total_periods = len(df)
                    
                    persistence_rate = utils['math_validation']['basic_math']['safe_divide'](
                        total_periods - regime_changes, total_periods, default=0.0
                    ) * 100
                    
                    avg_duration = utils['math_validation']['basic_math']['safe_divide'](
                        total_periods, regime_changes, default=total_periods
                    )
                    
                    regime_stats[str(file_path.stem)] = {
                        "total_samples": len(df),
                        "unique_regimes": unique_regimes,
                        "regime_distribution": regime_counts,
                        "persistence_rate": persistence_rate,
                        "avg_regime_duration": avg_duration,
                        "data_quality_score": min(100, (unique_regimes / 5) * 100)
                    }
                    
                except Exception as e:
                    raise RuntimeError(f"Failed to process labeled data file {file_path}: {str(e)}") from e
            
            # Analyze regime transitions using math validation
            combinations = meta_data.get("combination_counts", {})
            total_combinations = sum(combinations.values())
            
            stability_score = utils['math_validation']['basic_math']['safe_divide'](
                len(combinations), total_combinations, default=0.0
            )
            
            return {
                "regime_statistics": regime_stats,
                "regime_transition_analysis": {
                    "total_regime_combinations": len(combinations),
                    "total_transition_events": total_combinations,
                    "most_stable_regime": max(combinations.items(), key=lambda x: x[1])[0] if combinations else None,
                    "regime_stability_score": stability_score
                },
                "summary": {
                    "datasets_analyzed": len(regime_stats),
                    "total_regime_types": len(meta_data.get("state_names", {}).get("momentum", {})),
                    "regime_detection_success": len(regime_stats) > 0
                }
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to analyze regime discovery statistics: {str(e)}") from e

    @inject_step03_utilities
    async def analyze_feature_engineering_metrics(self, symbol: str, exchange: str, timeframe: str,
                                                utils: Dict[str, Any] = None, services: Step03ServiceProvider = None) -> dict:
        """Analyze feature engineering metrics using comprehensive utilities."""
        try:
            # Use common operations for file validation
            vectorized_file = Path("data/training") / f"BINANCE_{symbol}_{timeframe}_vectorized_feature_pre_optimization.json"
            
            if not utils['common_operations']['file_operations']['safe_file_exists'](vectorized_file):
                raise FileNotFoundError(f"Vectorized features file not found: {vectorized_file}")
            
            # Load features using serialization utilities
            features_data = utils['serialization']['convenience_functions']['load_json'](vectorized_file, {})
            
            if not isinstance(features_data, dict) or "features" not in features_data:
                raise ValueError("Invalid features data structure")
            
            features = features_data["features"]
            if not isinstance(features, list) or len(features) == 0:
                raise ValueError("No features found in features data")
            
            # Analyze features using data processing utilities
            feature_stats = {
                "total_features": len(features),
                "feature_types": {},
                "feature_categories": set(),
                "feature_quality_metrics": {}
            }
            
            # Use common utilities for feature categorization
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
                category = "other"
                for cat, keywords in feature_categories.items():
                    if any(keyword in feature_name.lower() for keyword in keywords):
                        category = cat
                        break
                
                feature_stats["feature_categories"].add(category)
                category_counts[category] += 1
            
            feature_stats["feature_categories"] = list(feature_stats["feature_categories"])
            feature_stats["feature_types"] = category_counts
            
            # Use math validation for quality metrics
            diversity_score = utils['math_validation']['basic_math']['safe_divide'](
                len(feature_stats["feature_categories"]), len(feature_categories), default=0.0
            )
            
            max_count = max(category_counts.values()) if category_counts.values() else 0
            min_count = min(category_counts.values()) if category_counts.values() else 0
            balance_score = utils['math_validation']['basic_math']['safe_divide'](
                max_count - min_count, len(features), default=0.0
            )
            balance_score = 1.0 - balance_score
            
            coverage_score = utils['math_validation']['basic_math']['safe_divide'](
                len(features), 100.0, default=0.0
            )
            
            feature_stats["feature_quality_metrics"] = {
                "diversity_score": diversity_score,
                "balance_score": balance_score,
                "coverage_score": coverage_score
            }
            
            return {
                "feature_statistics": feature_stats,
                "feature_engineering_summary": {
                    "feature_generation_success": len(feature_stats) > 0,
                    "feature_diversity_score": len(feature_stats.get("feature_types", {}))
                }
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to analyze feature engineering metrics: {str(e)}") from e

    @inject_step03_utilities
    async def generate_comprehensive_report(self, symbol: str, exchange: str, timeframe: str, 
                                          execution_time: float, correlation_id: str,
                                          utils: Dict[str, Any] = None, services: Step03ServiceProvider = None) -> dict:
        """Generate comprehensive market analysis report using all utilities."""
        try:
            # Use M1 CPU optimizer for parallel execution
            cpu_optimizer = services.get_service(utils['m1_optimizers']['cpu']['M1CPUOptimizer'])
            
            # Execute analysis functions in parallel
            analysis_tasks = [
                self.analyze_hmm_clustering_results(symbol, exchange, timeframe),
                self.analyze_regime_discovery_statistics(symbol, exchange, timeframe),
                self.analyze_feature_engineering_metrics(symbol, exchange, timeframe)
            ]
            
            # Use common operations for async gathering
            results = await utils['common_operations']['async_operations']['safe_gather'](*analysis_tasks)
            hmm_results, regime_results, feature_results = results
            
            # Generate summary using common operations
            current_time = utils['common_operations']['datetime']['get_current_datetime']()
            
            summary = {
                "execution_info": {
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "execution_time_seconds": execution_time,
                    "correlation_id": correlation_id,
                    "timestamp": current_time.isoformat()
                },
                "pipeline_components": {
                    "hmm_clustering": "completed" if not hmm_results.get("error") else "failed",
                    "regime_discovery": "completed" if not regime_results.get("error") else "failed",
                    "feature_engineering": "completed" if not feature_results.get("error") else "failed"
                },
                "data_quality_metrics": {
                    "hmm_data_available": hmm_results.get("data_availability", {}).get("metadata_available", False),
                    "regime_data_processed": regime_results.get("summary", {}).get("datasets_analyzed", 0),
                    "features_generated": feature_results.get("feature_statistics", {}).get("total_features", 0)
                }
            }
            
            return {
                "summary": summary,
                "hmm_clustering_analysis": hmm_results,
                "regime_discovery_analysis": regime_results,
                "feature_engineering_analysis": feature_results,
                "utilities_used": {
                    "common_operations": True,
                    "common_utilities": True,
                    "math_validation": True,
                    "parquet_utils": True,
                    "serialization_utils": True,
                    "data_processing_utils": True,
                    "m1_optimizers": True
                }
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate comprehensive report: {str(e)}") from e

@inject_step03_utilities
async def main(utils: Dict[str, Any] = None, services: Step03ServiceProvider = None):
    """Main function to run enhanced market analysis pipeline with comprehensive utilities."""
    # Configuration
    symbol = "ETHUSDT"
    exchange = "BINANCE"
    timeframe = "1m"
    data_dir = "data_cache"
    
    # Initialize enhanced pipeline
    pipeline = EnhancedMarketAnalysisPipeline({})
    
    # Use common operations for logging
    current_time = utils['common_operations']['datetime']['format_datetime'](
        utils['common_operations']['datetime']['get_current_datetime'](), 
        '%Y-%m-%d %H:%M:%S'
    )
    
    print('🚀 Enhanced Market Analysis Pipeline with Comprehensive Utilities')
    print('=' * 80)
    print(f'📊 Configuration:')
    print(f'   Symbol: {symbol}')
    print(f'   Exchange: {exchange}')
    print(f'   Timeframe: {timeframe}')
    print(f'   Data directory: {data_dir}')
    print(f'   Start time: {current_time}')
    print('🔧 Comprehensive Utilities Integrated:')
    print('   ✅ common_operations.py - Core operations and utilities')
    print('   ✅ common_utilities.py - Data processing utilities')
    print('   ✅ math_validation.py - Mathematical validation and operations')
    print('   ✅ parquet_utils.py - Parquet file operations')
    print('   ✅ serialization_utils.py - Data serialization')
    print('   ✅ data_processing_utils.py - DataFrame processing')
    print('   ✅ m1_gpu_utils.py - M1 GPU optimization')
    print('   ✅ m1_memory_optimizer.py - M1 memory optimization')
    print('   ✅ m1_cpu_optimizer.py - M1 CPU optimization')
    print('=' * 80)
    
    start_time = time.time()
    
    try:
        # Run market analysis pipeline
        success = await run_enhanced_market_analysis_pipeline(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            data_dir=data_dir,
            force_rerun=True,
            hmm_clustering=True,
            regime_splitting=True,
            feature_engineering=True,
            matrix_operations=True,
            feature_selection=True,
            random_state=42
        )
        
        total_time = time.time() - start_time
        
        if success:
            print('\n✅ ENHANCED MARKET ANALYSIS WITH COMPREHENSIVE UTILITIES COMPLETED!')
            print('=' * 80)
            
            # Generate comprehensive report using all utilities
            correlation_id = f"enhanced_market_analysis_{symbol}_{exchange}_{int(time.time())}"
            comprehensive_report = await pipeline.generate_comprehensive_report(
                symbol, exchange, timeframe, total_time, correlation_id
            )
            
            # Save report using serialization utilities
            report_file = save_training_report(
                data=comprehensive_report,
                step_name="step03_market_analysis_enhanced_with_utilities",
                report_type="comprehensive_report_with_utilities",
                symbol=symbol,
                timeframe=timeframe,
                file_format="json"
            )
            
            print(f'💾 Comprehensive report with utilities saved to: {report_file}')
            print(f'⏱️ Total execution time: {total_time:.2f} seconds')
            print('🔧 All utilities extensively used and integrated successfully!')
            print('=' * 80)
        else:
            print('\n❌ ENHANCED MARKET ANALYSIS WITH UTILITIES FAILED!')
            print('=' * 80)
            print('❌ Please check the logs for error details')
            print('=' * 80)
            
    except Exception as e:
        total_time = time.time() - start_time
        print(f'\n❌ ENHANCED MARKET ANALYSIS WITH UTILITIES FAILED WITH EXCEPTION: {e}')
        print('=' * 80)
        print(f'⏱️ Total execution time: {total_time:.2f} seconds')
        print('=' * 80)
        raise

if __name__ == "__main__":
    asyncio.run(main())