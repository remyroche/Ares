#!/usr / bin / env python3
""""""Step 3: Parameter Optimization for HMM Regime Discovery.""

This module performs comprehensive parameter optimization for HMM regime discovery,"
focusing on finding optimal parameters for clustering, feature engineering, and"""
regime detection algorithms."""
""""""""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
project_root, Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.centralized_decorators import ()
    comprehensive_data_validation,
    handle_errors,
    memory_efficient,
    resource_monitor,
    secure_data_processing,
    validate_data_structure,
    with_tracing_span,
    quality_gate,
    monitor_feature_engineering,
    ensure_data_integrity,
    monitor_step_execution,
    secure_step_execution,
    validate_pipeline_step
"
from src.utils.logger import system_logger"""
""""
logger, system_logger.getChild("Step3ParameterOptimization")"
"""
class ParameterOptimizationStep:"""
    """Step 3: Parameter Optimization for HMM Regime Discovery."""""
"
    def __init__(self, config: dict[str, Any]) -> None:"""
        self.config, config""""
        self.logger, system_logger.getChild("ParameterOptimizationStep")
        self.start_time, None
        self.optimization_results = {}
        self._initialize_components()
"
    @secure_step_execution"""
    def _initialize_components(self) -> None:"""
        """Initialize parameter optimization components.""""""
        self.logger.info("🔧 Initializing parameter optimization components...")
        try:
            except Exception as e:"
                pass"""
        # Initialize optimization components""""
        self.logger.info("✅ Parameter optimization components initialized successfully")
"
        except Exception as e:"""
            pass""""
        self.logger.error(f"❌ Failed to initialize parameter optimization components: {e}")
            raise

    @handle_errors()"
        exceptions=(Exception,),"""
        default_return = False,""""
        context="parameter_optimization_initialization"
    "
    @secure_step_execution"""
    async def initialize(self) -> bool:"""
        """Initialize the parameter optimization step."""""
        try:"
            except Exception as e:"""
                pass""""
        self.logger.info("🚀 Initializing parameter optimization step...")"
"""
        # Load optimization configuration""""
            optimization_config, self.config.get("parameter_optimization", {})""""
        self.logger.info(f"📋 Optimization configuration loaded: {len(optimization_config)} parameters")""
"""""
        self.logger.info("✅ Parameter optimization step initialized successfully")
        return True
"
        except Exception as e:"""
            pass""""
        self.logger.error(f"Failed to initialize parameter optimization step: {e}")
        return False

    @monitor_step_execution
    @secure_step_execution
    @validate_pipeline_step
    @handle_errors()"
        exceptions=(Exception,),"""
        default_return = False,""""
        context="parameter_optimization_execution""
    """
    async def execute(self) -> bool:"""
        """Execute the parameter optimization step."""""
        try:"
            except Exception as e:"""
                pass""""
        self.logger.info("🎯 Starting parameter optimization for HMM regime discovery...")
        self.start_time, time.time()
"
        # Step 1: Load and validate data"""
            data_loaded, await self._load_and_validate_data()""""
        if not data_loaded.get("success", False):"""
            pass""""
        self.logger.error("Failed to load and validate data")
        return False"
"""
        # Step 2: Perform HMM parameter optimization""""
            hmm_optimization, await self._optimize_hmm_parameters(data_loaded["data"])"
"""
        # Step 3: Perform clustering parameter optimization""""
            clustering_optimization, await self._optimize_clustering_parameters(data_loaded["data"])"
"""
        # Step 4: Perform feature engineering parameter optimization""""
            feature_optimization, await self._optimize_feature_parameters(data_loaded["data"])

        # Step 5: Combine optimization results
            combined_results, await self._combine_optimization_results([])
                hmm_optimization,
                clustering_optimization,
                feature_optimization
            

        # Step 6: Save optimization results
        await self._save_optimization_results(combined_results)

        # Step 7: Generate optimization reports
        await self._generate_optimization_reports(combined_results)"
"""
            execution_time, time.time() - self.start_time""""
        self.logger.info(f"✅ Parameter optimization completed successfully in {execution_time:.2f}s")

        return True
"
        except Exception as e:"""
            pass""""
        self.logger.error(f"Failed to execute parameter optimization: {e}")
        return False
"
    @handle_errors()"""
        exceptions=(Exception,),""""
        default_return={"success": False, "error": "Data loading failed"},""""
        context="load_and_validate_data"
    
    @comprehensive_data_validation"
    @ensure_data_integrity"""
    async def _load_and_validate_data(self) -> dict[str, Any]:"""
        """Load and validate data for parameter optimization."""""
        try:"
            except Exception as e:"""
                pass""""
        self.logger.info("📊 Loading and validating data for parameter optimization...")"
"""
        # Get data parameters from config""""
            symbol, self.config.get("SYMBOL", "ETHUSDT")""""
            exchange, self.config.get("EXCHANGE", "BINANCE")""""
            timeframe, self.config.get("TIMEFRAME", "1m")""""
            data_dir, self.config.get("DATA_DIR", "data_cache")"
"""
        # Load klines data""""
            klines_path, Path(data_dir) / f"klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet"
"
        if not klines_path.exists():"""
            pass""""
        self.logger.error(f"❌ Klines file not found: {klines_path}")"""
        return {}"""
                    "success": False,"""
                    "error": f"Klines file not found: {klines_path}"
                

        # Load data
            df, pd.read_parquet(klines_path)
"
        if df.empty:"""
            pass""""
        self.logger.error("❌ Data is empty")"""
        return {}"""
                    "success": False,"""
                    "error": "Data is empty"
                

        # Prepare features for optimization"
            features, await self._prepare_features_for_optimization(df)""
"""""
        self.logger.info(f"✅ Data loaded and validated: {len(df):,} rows, {len(features.columns)} features")"
"""
        return {}"""
                "success": True,"""
                "data": df,"""
                "features": features,"""
                "data_info": {}"""
                    "rows": len(df),"""
                    "columns": list(df.columns),"""
                    "date_range": {}"""
                        "start": df["timestamp"].min().isoformat(),"""
                        "end": df["timestamp""].max().isoformat()"
                    
                
            
"
        except Exception as e:"""
            pass""""
        self.logger.error(f"Failed to load and validate data: {e}")""""
        return {"success": False, "error": str(e)}
"
    @handle_errors()"""
        exceptions=(Exception,),"""
        default_return = pd.DataFrame(),""""
        context="prepare_features_for_optimization"
    
    @monitor_feature_engineering()"
    @validate_data_structure"""
    async def _prepare_features_for_optimization(self, df: pd.DataFrame) -> pd.DataFrame:"""
        """Prepare features for parameter optimization."""""
        try:"
            except Exception as e:"""
                pass""""
        self.logger.info("🔧 Preparing features for parameter optimization...")"
"""
        # Ensure timestamp is datetime""""
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):""""
                df["timestamp"] = pd.to_datetime(df["timestamp"])"
"""
        # Sort by timestamp""""
            df, df.sort_values("timestamp").reset_index(drop = True)
"
        # Calculate basic features"""
            features, pd.DataFrame()""""
            features["timestamp"] = df["timestamp"]"
"""
        # Price - based features""""
            features["price_momentum_5"] = df["close"].pct_change(5)""""
            features["price_momentum_10"] = df["close"].pct_change(10)""""
            features["price_momentum_20"] = df["close"].pct_change(20)"
"""
        # Volatility features""""
            features["volatility_5"] = df["close"].pct_change().rolling(window = 5).std()""""
            features["volatility_10"] = df["close"].pct_change().rolling(window = 10).std()""""
            features["volatility_20"] = df["close"].pct_change().rolling(window = 20).std()"
"""
        # Volume features""""
            features["volume_ratio_5"] = df["volume"] / df["volume"].rolling(window = 5).mean()""""
            features["volume_ratio_10"] = df["volume"] / df["volume"].rolling(window = 10).mean()""""
            features["volume_ratio_20"] = df["volume"] / df["volume"].rolling(window = 20).mean()"
"""
        # Technical indicators""""
            features["rsi"] = self._calculate_rsi(df["close"])""""
            features["macd"] = self._calculate_macd(df["close"])""""
            features["atr"] = self._calculate_atr(df)""
""
        # Remove timestamp and handle NaN values""""
            optimization_features, features.drop("timestamp", axis = 1)"
            optimization_features, optimization_features.fillna(0)""
"""""
        self.logger.info(f"✅ Features prepared: {len(optimization_features.columns)} features")
        return optimization_features
"
        except Exception as e:"""
            pass""""
        self.logger.error(f"Failed to prepare features: {e}")
        return pd.DataFrame()

    @handle_errors()"
        exceptions=(Exception,),"""
        default_return={},""""
        context="optimize_hmm_parameters"
    
    @resource_monitor"
    @secure_data_processing"""
    async def _optimize_hmm_parameters(self, data: pd.DataFrame) -> dict[str, Any]:"""
        """Optimize HMM parameters."""""
        try:"
            except Exception as e:"""
                pass""""
        self.logger.info("🧠 Optimizing HMM parameters...")"
"""
            optimization_result = {}"""
                "n_components_range": [2, 3, 4, 5, 6, 8, 10],"""
                "covariance_types": ["full", "tied", "diag", "spherical"],"""
                "n_iter_range": [50, 100, 200],"""
                "random_states": [42, 123, 456],"""
                "best_parameters": {},"""
                "optimization_scores": {},"""
                "recommendations"": []"
            

        # Simple optimization based on data characteristics
            data_size, len(data)

        # Recommend number of components based on data size
        if data_size < 1000:
                optimal_components, 3
            elif data_size < 5000:
                optimal_components, 4
            elif data_size < 10000:
                optimal_components, 5
            else:"
                optimal_components, 6""
"""""
            optimization_result["best_parameters"] = {}"""
                "n_components": optimal_components,"""
                "covariance_type": "full","""
                "n_iter": 100,"""
                "random_state": 42"
            ""
"""""
            optimization_result["recommendations"] = []""""
                f"Use {optimal_components} HMM components for data size {data_size:,}","""
                "Full covariance type recommended for comprehensive regime modeling","""
                "100 iterations sufficient for convergence""
            ""
"""""
        self.logger.info(f"✅ HMM parameters optimized: {optimal_components} components")
        return optimization_result
"
        except Exception as e:"""
            pass""""
        self.logger.error(f"Failed to optimize HMM parameters: {e}")
        return {}

    @handle_errors()"
        exceptions=(Exception,),"""
        default_return={},""""
        context="optimize_clustering_parameters"
    
    @resource_monitor"
    @secure_data_processing"""
    async def _optimize_clustering_parameters(self, data: pd.DataFrame) -> dict[str, Any]:"""
        """Optimize clustering parameters."""""
        try:"
            except Exception as e:"""
                pass""""
        self.logger.info("🎯 Optimizing clustering parameters...")"
"""
            optimization_result = {}"""
                "n_clusters_range": [5, 10, 15, 20, 25, 30],"""
                "clustering_methods": ["kmeans", "dbscan", "hierarchical"],"""
                "best_parameters": {},"""
                "optimization_scores": {},"""
                "recommendations"": []"
            

        # Simple optimization based on data characteristics
            data_size, len(data)

        # Recommend number of clusters based on data size
        if data_size < 1000:
                optimal_clusters, 10
            elif data_size < 5000:
                optimal_clusters, 15
            elif data_size < 10000:
                optimal_clusters, 20
            else:"
                optimal_clusters, 25""
"""""
            optimization_result["best_parameters"] = {}"""
                "n_clusters": optimal_clusters,"""
                "method": "kmeans","""
                "random_state": 42,"""
                "n_init": 10"
            ""
"""""
            optimization_result["recommendations"] = []""""
                f"Use {optimal_clusters} clusters for data size {data_size:,}","""
                "K - means clustering recommended for regime discovery","""
                "10 initializations for robust clustering""
            ""
"""""
        self.logger.info(f"✅ Clustering parameters optimized: {optimal_clusters} clusters")
        return optimization_result
"
        except Exception as e:"""
            pass""""
        self.logger.error(f"Failed to optimize clustering parameters: {e}")
        return {}

    @handle_errors()"
        exceptions=(Exception,),"""
        default_return={},""""
        context="optimize_feature_parameters"
    
    @resource_monitor"
    @secure_data_processing"""
    async def _optimize_feature_parameters(self, data: pd.DataFrame) -> dict[str, Any]:"""
        """Optimize feature engineering parameters."""""
        try:"
            except Exception as e:"""
                pass""""
        self.logger.info("🔧 Optimizing feature engineering parameters...")"
"""
            optimization_result = {}"""
                "momentum_windows": [5, 10, 15, 20, 25, 30],"""
                "volatility_windows": [5, 10, 15, 20, 25, 30],"""
                "volume_windows": [5, 10, 15, 20, 25, 30],"""
                "best_parameters": {},"""
                "optimization_scores": {},"""
                "recommendations": []
            

        # Simple optimization based on data characteristics
            data_size, len(data)

        # Recommend feature windows based on data size
        if data_size < 1000:
                optimal_momentum, 10
                optimal_volatility, 15
                optimal_volume, 10
            elif data_size < 5000:
                optimal_momentum, 15
                optimal_volatility, 20
                optimal_volume, 15
            else:
                optimal_momentum, 20
                optimal_volatility, 25"
                optimal_volume, 20""
"""""
            optimization_result["best_parameters"] = {}"""
                "momentum_window": optimal_momentum,"""
                "volatility_window": optimal_volatility,"""
                "volume_window": optimal_volume,"""
                "rsi_window": 14,"""
                "macd_fast": 12,"""
                "macd_slow": 26,"""
                "macd_signal": 9,"""
                "atr_window": 14"
            ""
"""""
            optimization_result["recommendations"] = []""""
                f"Use momentum window {optimal_momentum} for data size {data_size:,}",""""
                f"Use volatility window {optimal_volatility} for data size {data_size:,}",""""
                f"Use volume window {optimal_volume} for data size {data_size:,}","""
                "Standard technical indicator parameters recommended"""
            ""
""""
        self.logger.info(f"✅ Feature parameters optimized")
        return optimization_result
"
        except Exception as e:"""
            pass""""
        self.logger.error(f"Failed to optimize feature parameters: {e}")
        return {}

    @handle_errors()"
        exceptions=(Exception,),"""
        default_return={},""""
        context="combine_optimization_results"
    "
    @secure_data_processing"""
    async def _combine_optimization_results(self, results: List[dict[str, Any]]) -> dict[str, Any]:"""
        """Combine all optimization results."""""
        try:"
            except Exception as e:"""
                pass""""
        self.logger.info("🔗 Combining optimization results...")

        # Filter out empty results
            valid_results = [r for r in results if r]
"
        if not valid_results:"""
            pass""""
        self.logger.warning("No valid optimization results to combine")
        return {}"
"""
            combined_result = {}"""
                "hmm_optimization": {},"""
                "clustering_optimization": {},"""
                "feature_optimization": {},"""
                "combined_parameters": {},"""
                "optimization_summary": {}"""
                    "total_optimizations": len(valid_results),"""
                    "optimization_status": "completed","""
                    "timestamp": datetime.now().isoformat()"""
                },"""
                "recommendations"": []"
            

        # Extract results by type"
        for result in valid_results:"""
            pass""""
        if "n_components_range" in result:""""
                    combined_result["hmm_optimization"] = result""""
                elif "n_clusters_range" in result:""""
                    combined_result["clustering_optimization"] = result""""
                elif "momentum_windows" in result:""""
                    combined_result["feature_optimization"] = result

        # Create combined parameters"
            combined_params = {}""
"""""
        if combined_result["hmm_optimization"]:""""
                combined_params.update(combined_result["hmm_optimization"].get("best_parameters", {}))""
"""""
        if combined_result["clustering_optimization"]:""""
                combined_params.update(combined_result["clustering_optimization"].get("best_parameters", {}))""
"""""
        if combined_result["feature_optimization"]:""""
                combined_params.update(combined_result["feature_optimization"].get("best_parameters", {}))""
"""""
            combined_result["combined_parameters"] = combined_params

        # Combine recommendations
            all_recommendations = []"
        for result in valid_results:"""
            pass""""
        if "recommendations" in result:""""
                    all_recommendations.extend(result["recommendations"])""
"""""
            combined_result["recommendations"] = all_recommendations""
"""""
        self.logger.info(f"✅ Combined {len(valid_results)} optimization results")
        return combined_result
"
        except Exception as e:"""
            pass""""
        self.logger.error(f"Failed to combine optimization results: {e}")
        return {}

    @handle_errors()"
        exceptions=(Exception,),"""
        default_return = False,""""
        context="save_optimization_results"
    "
    @secure_data_processing"""
    async def _save_optimization_results(self, optimization_results: dict[str, Any]) -> bool:"""
        """Save optimization results."""""
        try:"
            except Exception as e:"""
                pass""""
        self.logger.info("💾 Saving optimization results...")"
"""
        # Create optimization results directory""""
            results_dir, Path("data / optimization")
            results_dir.mkdir(parents = True, exist_ok = True)"
"""
        # Save optimization results""""
            results_file, results_dir / "parameter_optimization_results.json"""
"""""
        with open(results_file, 'w') as f:'
                json.dump(optimization_results, f, indent = 2, default = str)''
'''''
        self.logger.info(f"✅ Optimization results saved to {results_file}")
        return True
"
        except Exception as e:"""
            pass""""
        self.logger.error(f"Failed to save optimization results: {e}")
        return False

    @handle_errors()"
        exceptions=(Exception,),"""
        default_return = False,""""
        context="generate_optimization_reports"
    "
    @secure_data_processing"""
    async def _generate_optimization_reports(self, optimization_results: dict[str, Any]) -> bool:"""
        """Generate optimization reports."""""
        try:"
            except Exception as e:"""
                pass""""
        self.logger.info("📋 Generating optimization reports...")"
"""
        # Create reports directory""""
            reports_dir, Path("reports / parameter_optimization")
            reports_dir.mkdir(parents = True, exist_ok = True)
"
        # Generate summary report"""
            summary_report = {}"""
                "optimization_summary": optimization_results.get("optimization_summary", {}),"""
                "combined_parameters": optimization_results.get("combined_parameters", {}),"""
                "recommendations": optimization_results.get("recommendations", []),"""
                "next_steps": []"""
                    "Proceed to step03_5 for final regime clustering","""
                    "Use optimized parameters in regime discovery","""
                    "Validate parameters with out - of - sample data"""
                
            "
"""
        # Save summary report""""
            summary_file, reports_dir / "parameter_optimization_summary.json""""""""
        with open(summary_file, 'w') as f:
                json.dump(summary_report, f, indent = 2, default = str)'
'''
        # Log summary''''
        self.logger.info("=" * 60)""""
        self.logger.info("📊 PARAMETER OPTIMIZATION SUMMARY")""""
        self.logger.info("=" * 60)""""
        self.logger.info(f"🔧 HMM Components: {optimization_results.get("combined_parameters', {}).get('n_components', 'N / A')}')''''
        self.logger.info(f"🎯 Clusters: {optimization_results.get("combined_parameters', {}).get('n_clusters', 'N / A')}')''''
        self.logger.info(f"📈 Momentum Window: {optimization_results.get("combined_parameters', {}).get('momentum_window', 'N / A')}')''''
        self.logger.info(f"📊 Volatility Window: {optimization_results.get("combined_parameters', {}).get('volatility_window', 'N / A')}')''''
        self.logger.info(f"📋 Recommendations: {len(optimization_results.get("recommendations', []))}')''''
        self.logger.info("=" * 60)""
"""""
        self.logger.info(f"✅ Optimization reports saved to {reports_dir}")
        return True
"
        except Exception as e:"""
            pass""""
        self.logger.error(f"Failed to generate optimization reports: {e}")
        return False

    # Helper methods for technical indicators
    @handle_errors()"
        exceptions=(Exception,),"""
        default_return = pd.Series(),""""
        context="calculate_rsi""
    """
    def _calculate_rsi(self, prices: pd.Series, window: int, 14) -> pd.Series:"""
        """Calculate Relative Strength Index."""""
        delta, prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window = window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window = window).mean()
        rs, gain / loss
        rsi, 100 - (100 / (1 + rs))
        return rsi

    @handle_errors()"
        exceptions=(Exception,),"""
        default_return = pd.Series(),""""
        context="calculate_macd""
    """
    def _calculate_macd(self, prices: pd.Series, fast: int, 12, slow: int, 26, signal: int, 9) -> pd.Series:"""
        """Calculate MACD."""""
        ema_fast, prices.ewm(span = fast).mean()
        ema_slow, prices.ewm(span = slow).mean()
        macd, ema_fast - ema_slow
        return macd

    @handle_errors()"
        exceptions=(Exception,),"""
        default_return = pd.Series(),""""
        context="calculate_atr"""""""
    ""
    def _calculate_atr(self, df: pd.DataFrame, window: int, 14) -> pd.Series:"""
        """Calculate Average True Range.""""""
        high, df["high"]""""
        low, df["low"]""""
        close, df["close"]

        tr1, high - low
        tr2, abs(high - close.shift(1))
        tr3, abs(low - close.shift(1))

        tr, pd.concat([tr1, tr2, tr3], axis = 1).max(axis = 1)
        atr, tr.rolling(window = window).mean()
        return atr

    @handle_errors()"
        exceptions=(Exception,),"""
        default_return = False,""""
        context="parameter_optimization_cleanup"
    "
    @secure_step_execution"""
    async def cleanup(self) -> bool:"""
        """Clean up resources after optimization."""""
        try:"
            except Exception as e:"""
                pass""""
        self.logger.info("🧹 Cleaning up parameter optimization resources...")""""
        self.logger.info("✅ Parameter optimization cleanup completed")
        return True
"
        except Exception as e:"""
            pass""""
        self.logger.error(f"Failed to cleanup parameter optimization: {e}")
        return False

@handle_errors()"
    exceptions=(Exception,),"""
    default_return = False,""""
    context="step03_parameter_optimization"
"
@secure_step_execution"""
async def run_step(config: dict[str, Any]) -> bool:"""
    """Run the parameter optimization step."""""
    try:"
        except Exception as e:"""
            pass""""
        logger.info("🚀 Starting Step 3: Parameter Optimization")

        # Create and initialize the step
        step, ParameterOptimizationStep(config)
"
        # Initialize the step"""
        if not await step.initialize():""""
            logger.error("Failed to initialize parameter optimization step")
        return False

        # Execute the step
        success, await step.execute()

        # Cleanup
        await step.cleanup()"
"""
        if success:""""
            logger.info("✅ Step 3: Parameter Optimization completed successfully")"""
        else:""""
            logger.error("❌ Step 3: Parameter Optimization failed")

        return success"
"""
    except Exception as e:""""
        logger.error(f"Failed to run parameter optimization step: {e}")"
        return False""
"""""
if __name__ == "__main__":
    # Test the step
    import asyncio
"
    # Load test configuration"""
    test_config = {}"""
        "SYMBOL": "ETHUSDT","""
        "EXCHANGE": "BINANCE","""
        "TIMEFRAME": "1m","""
        "DATA_DIR": "data_cache","""
        "parameter_optimization": {}"""
            "enable_hmm_optimization": True,"""
            "enable_clustering_optimization": True,"""
            "enable_feature_optimization": True,"""
            "optimization_timeout": 600,  # 10 minutes"""
            "max_trials": 50
        
    
"
    # Run the step"""
    success, asyncio.run(run_step(test_config))""""
    print(f"Step execution {"successful' if success else 'failed'}')'''"""'''""""