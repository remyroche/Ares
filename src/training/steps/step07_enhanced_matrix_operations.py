# src/training/steps/step7_enhanced_matrix_operations.py

"""Step 7: Enhanced Matrix Operations with Standardized Data Quality Management.
This step performs advanced matrix operations for comprehensive data analysis after feature engineering.
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
import sys
sys.path.insert(0, str(project_root))

# Import pipeline standards
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards

# Standardized import management
REQUIRED_MODULES = [
    "pandas",
    "numpy",
    "src.training.enhanced_matrix_operations",
    "src.utils.error_handler",
    "src.utils.logger",
    "src.training.feature_engineering_optimizer",
    "src.training.timeframe_relevance_analyzer",
    "src.utils.training_pipeline_decorators",
    "src.utils.enhanced_mlflow_integration"
]

# Validate environment dependencies
dependency_status = PipelineStandards.validate_environment_dependencies(REQUIRED_MODULES)

# Safe imports with fallbacks
enhanced_matrix_operations = PipelineStandards.safe_import("src.training.enhanced_matrix_operations", None)
error_handler = PipelineStandards.safe_import("src.utils.error_handler", None)
system_logger = PipelineStandards.safe_import("src.utils.logger", None)
feature_engineering_optimizer = PipelineStandards.safe_import("src.training.feature_engineering_optimizer", None)
timeframe_relevance_analyzer = PipelineStandards.safe_import("src.training.timeframe_relevance_analyzer", None)
training_pipeline_decorators = PipelineStandards.safe_import("src.utils.training_pipeline_decorators", None)
enhanced_mlflow = PipelineStandards.safe_import("src.utils.enhanced_mlflow_integration", None)
numpy = PipelineStandards.safe_import("numpy", None)
pandas = PipelineStandards.safe_import("pandas", None)

# Fallback functions if imports fail
def create_fallback_logger():
    import logging
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

def create_fallback_decorator():
    def decorator(func):
        return func
    return decorator

# Initialize fallbacks
if system_logger is None:
    system_logger = create_fallback_logger()

if training_pipeline_decorators is None:
    circuit_breaker_protection = create_fallback_decorator()
    debug_training_step = create_fallback_decorator()
    memory_efficient = create_fallback_decorator()
    prevent_data_leakage = create_fallback_decorator()
    quality_gate = create_fallback_decorator()
    resource_monitor = create_fallback_decorator()
    secure_data_processing = create_fallback_decorator()
    validate_step_output = create_fallback_decorator()
else:
    circuit_breaker_protection = training_pipeline_decorators.circuit_breaker_protection
    debug_training_step = training_pipeline_decorators.debug_training_step
    memory_efficient = training_pipeline_decorators.memory_efficient
    prevent_data_leakage = training_pipeline_decorators.prevent_data_leakage
    quality_gate = training_pipeline_decorators.quality_gate
    resource_monitor = training_pipeline_decorators.resource_monitor
    secure_data_processing = training_pipeline_decorators.secure_data_processing
    validate_step_output = training_pipeline_decorators.validate_step_output

if error_handler is None:
    handle_errors = create_fallback_decorator()
else:
    handle_errors = error_handler.handle_errors

if enhanced_mlflow is None:
    with_enhanced_mlflow_logging = create_fallback_decorator()
    log_step_report = lambda *args, **kwargs: "fallback_report"
    create_detailed_step_report = lambda *args, **kwargs: {}
    log_step_metrics = lambda *args, **kwargs: None
    log_step_dataframe_with_standardized_name = lambda *args, **kwargs: "fallback_dataframe"
    log_step_artifact_with_standardized_name = lambda *args, **kwargs: "fallback_artifact"
else:
    with_enhanced_mlflow_logging = enhanced_mlflow.with_enhanced_mlflow_logging
    log_step_report = enhanced_mlflow.log_step_report
    create_detailed_step_report = enhanced_mlflow.create_detailed_step_report
    log_step_metrics = enhanced_mlflow.log_step_metrics
    log_step_dataframe_with_standardized_name = enhanced_mlflow.log_step_dataframe_with_standardized_name
    log_step_artifact_with_standardized_name = enhanced_mlflow.log_step_artifact_with_standardized_name


class Step7EnhancedMatrixOperations:
    """Step 7: Enhanced Matrix Operations with standardized data quality management."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize Step 7 Enhanced Matrix Operations."""
        self.config = config
        self.logger = system_logger.getChild("Step7EnhancedMatrixOperations")
        self.standards = pipeline_standards
        
        # Validate environment on initialization
        self._validate_environment()
        
        # Initialize enhanced matrix operations if available
        if enhanced_matrix_operations is not None:
            self.matrix_ops = enhanced_matrix_operations.EnhancedMatrixOperations(config)
        else:
            self.logger.warning("⚠️ EnhancedMatrixOperations not available")
            self.matrix_ops = None
        
        # Step-specific configuration
        self.step_config = config.get("step7_enhanced_matrix_operations", {})
        self.output_dir = Path(self.step_config.get("output_dir", "data/matrix_operations"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _validate_environment(self) -> None:
        """Validate environment dependencies."""
        self.logger.info("🔍 Validating environment dependencies...")
        
        missing_modules = [module for module, available in dependency_status.items() if not available]
        if missing_modules:
            self.logger.warning(f"⚠️ Missing optional modules: {missing_modules}")
            self.logger.info("📝 Pipeline will continue with fallback implementations")
        else:
            self.logger.info("✅ All required dependencies available")

    @secure_data_processing(encryption_level="high", data_validation=True)
    @prevent_data_leakage(validate_inputs=True, sanitize_outputs=True)
    @resource_monitor(cpu_threshold_percent=90.0, memory_threshold_gb=16.0)
    @memory_efficient(chunk_size=5000, streaming_processing=True)
    @debug_training_step(log_intermediate_results=True, save_debug_artifacts=True)
    @circuit_breaker_protection(failure_threshold=3, recovery_timeout=300.0)
    @validate_step_output(
        required_files=["matrix_operations_config.json"],
        data_quality_checks={"min_operations": 1}
    )
    @quality_gate(
        model_performance_thresholds={},
        data_quality_metrics={"completeness": 0.95}
    )
    @with_enhanced_mlflow_logging("step7_enhanced_matrix_operations")
    @handle_errors(exceptions=(ValueError, RuntimeError), default_return=False)
    async def execute(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Execute Step 7: Enhanced Matrix Operations.
        
        Args:
            training_input: Input data from previous steps
            pipeline_state: Current pipeline state
            
        Returns:
            Updated pipeline state with matrix operations results
        """
        try:
            start_time = datetime.now()
            self.logger.info("🚀 Starting Step 7: Enhanced Matrix Operations...")
            
            # Extract parameters
            symbol = training_input.get("symbol", "UNKNOWN")
            exchange = training_input.get("exchange", "UNKNOWN")
            timeframe = training_input.get("timeframe", "1m")
            
            # Load engineered features from step6
            features_train_path = f"data/training/{exchange}_{symbol}_{timeframe}_features_train.parquet"
            features_val_path = f"data/training/{exchange}_{symbol}_{timeframe}_features_val.parquet"
            
            if not os.path.exists(features_train_path):
                raise ValueError(f"Features train file not found: {features_train_path}")
            
            if not os.path.exists(features_val_path):
                raise ValueError(f"Features validation file not found: {features_val_path}")
            
            self.logger.info(f"📊 Loading engineered features from: {features_train_path}")
            
            # Load the engineered features (combine train and validation)
            df_train = pd.read_parquet(features_train_path)
            df_val = pd.read_parquet(features_val_path)
            df = pd.concat([df_train, df_val], ignore_index=True)
            
            self.logger.info(f"📈 Loaded {len(df)} rows of engineered features")
            self.logger.info(f"🔢 Features: {len(df.columns)} columns")

            
            # Initialize feature engineering optimization
            feature_optimizer = FeatureEngineeringOptimizer(self.config)
            timeframe_analyzer = TimeframeRelevanceAnalyzer(self.config)
            
            # Load HMM regime data if available
            hmm_regimes = None
            hmm_path = f"data/hmm_regimes/{exchange}_{symbol}_{timeframe}_hmm_regimes.parquet"
            if os.path.exists(hmm_path):
                self.logger.info(f"🎭 Loading HMM regimes from: {hmm_path}")
                hmm_data = pd.read_parquet(hmm_path)
                if 'regime' in hmm_data.columns:
                    hmm_regimes = hmm_data['regime']
            
            # Prepare target variable for optimization (use returns if available)
            target = None
            if 'returns' in df.columns:
                target = df['returns']
            elif 'close' in df.columns:
                target = df['close'].pct_change().dropna()
                df = df.loc[target.index]  # Align data
            else:
                self.logger.warning("⚠️ No target variable found for feature optimization")
            
            # 1. Optimize feature engineering parameters
            if target is not None:
                self.logger.info("🔧 Starting feature engineering parameter optimization...")
                feature_optimization_results = await feature_optimizer.optimize_feature_parameters(
                    data=df,
                    target=target,
                    regimes=hmm_regimes,
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe
                )
                
                # Store optimization results in pipeline state
                pipeline_state["feature_engineering_optimization"] = feature_optimization_results
                
                self.logger.info("✅ Feature engineering parameter optimization completed")
            else:
                self.logger.warning("⚠️ Skipping feature engineering optimization - no target variable")
                feature_optimization_results = {}
            
            # 2. Analyze timeframe relevance for high leverage trading
            self.logger.info("⏰ Starting timeframe relevance analysis...")
            
            # Load multi-timeframe data if available
            timeframe_data = {}
            for tf in ['1m', '5m', '15m', '30m', '1h']:
                tf_path = f"data/training/{exchange}_{symbol}_{tf}_features_train.parquet"
                if os.path.exists(tf_path):
                    tf_data = pd.read_parquet(tf_path)
                    timeframe_data[tf] = tf_data
            
            if timeframe_data:
                timeframe_analysis_results = await timeframe_analyzer.analyze_timeframe_relevance(
                    data_dict=timeframe_data,
                    symbol=symbol,
                    exchange=exchange,
                    leverage_range=(10, 100)  # 10x to 100x leverage
                )
                
                # Store timeframe analysis results
                pipeline_state["timeframe_relevance_analysis"] = timeframe_analysis_results
                
                self.logger.info("✅ Timeframe relevance analysis completed")
            else:
                self.logger.warning("⚠️ Skipping timeframe analysis - insufficient multi-timeframe data")
                timeframe_analysis_results = {}
            
            # Prepare matrix operations configuration
            matrix_config = self._prepare_matrix_operations_config(df, symbol, exchange, timeframe)
            
            # Execute matrix operations
            matrix_results = await self._execute_matrix_operations(df, matrix_config)
            
            # Execute enhanced stability analysis
            self.logger.info("🔍 Starting enhanced stability analysis...")
            
            # 1. Time-based stability analysis
            time_stability_results = self._analyze_feature_stability_over_time(df)
            matrix_results["time_based_stability"] = time_stability_results
            
            # 2. Distribution stability analysis
            distribution_stability_results = self._analyze_distribution_stability(df)
            matrix_results["distribution_stability"] = distribution_stability_results
            
            # 3. Feature importance stability analysis
            target_column = 'returns' if 'returns' in df.columns else 'close' if 'close' in df.columns else None
            importance_stability_results = self._analyze_feature_importance_stability(df, target_column)
            matrix_results["feature_importance_stability"] = importance_stability_results
            
            self.logger.info("✅ Enhanced stability analysis completed")
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(df, matrix_results)
            
            # Save results
            output_files = await self._save_matrix_operations_results(
                matrix_results, matrix_config, quality_metrics, symbol, exchange, timeframe
            )
            
            # Update pipeline state
            pipeline_state["step7_enhanced_matrix_operations"] = {
                "status": "completed",
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "output_files": output_files,
                "matrix_config": matrix_config,
                "matrix_results": matrix_results,
                "quality_metrics": quality_metrics,
                "data_shape": df.shape,
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "feature_engineering_optimization": feature_optimization_results,
                "timeframe_relevance_analysis": timeframe_analysis_results,
                "enhanced_stability_analysis": {
                    "time_based_stability": time_stability_results,
                    "distribution_stability": distribution_stability_results,
                    "feature_importance_stability": importance_stability_results
                }
            }
            
            self.logger.info("✅ Step 7: Enhanced Matrix Operations completed successfully")
            
            # Log artifacts and create detailed report
            await self._log_step7_artifacts_and_report(
            # Standardized naming pattern: {exchange}_{symbol}_{timestamp}_{step_num}_{artifact_type}
                training_input, pipeline_state, matrix_results, output_files, quality_metrics
            )
            
            return pipeline_state
            
        except Exception as e:
            self.logger.error(f"❌ Step 7 failed: {str(e)}")
            pipeline_state["step7_enhanced_matrix_operations"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            return pipeline_state

    async def _log_step7_artifacts_and_report(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
        matrix_results: dict[str, Any],
        output_files: dict[str, str],
        quality_metrics: dict[str, Any]
    ) -> None:
        """Log step 7 artifacts and create detailed report."""
        try:
            symbol = training_input.get("symbol", "UNKNOWN")
            exchange = training_input.get("exchange", "UNKNOWN")
            timeframe = training_input.get("timeframe", "1m")
            
            # Collect execution metadata
            execution_metadata = {
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": 0.0,  # Will be calculated if available
                "memory_usage_mb": 0.0,  # Will be calculated if available
                "cpu_usage_percent": 0.0,  # Will be calculated if available
                "data_quality_score": quality_metrics.get("overall_quality", 0.0),
                "processing_efficiency": 1.0 if pipeline_state.get("step7_enhanced_matrix_operations", {}).get("status") == "completed" else 0.0,
            }
            
            # Collect artifacts generated
            artifacts_generated = list(output_files.values()) if output_files else []
            
            # Collect metrics
            metrics_calculated = {
                "matrix_operations_success": 1.0 if pipeline_state.get("step7_enhanced_matrix_operations", {}).get("status") == "completed" else 0.0,
                "matrix_operations_count": len(matrix_results) if matrix_results else 0,
                "output_files_count": len(output_files) if output_files else 0,
                "overall_quality_score": quality_metrics.get("overall_quality", 0.0),
                "data_completeness": quality_metrics.get("data_completeness", 0.0),
                "feature_quality": quality_metrics.get("feature_quality", 0.0),
            }
            
            # Create step data for report
            step_data = {
                "matrix_results": matrix_results,
                "output_files": output_files,
                "quality_metrics": quality_metrics,
                "matrix_config": pipeline_state.get("step7_enhanced_matrix_operations", {}).get("matrix_config", {}),
            }
            
            # Create detailed report
            report_data = create_detailed_step_report(
                step_name="step7_enhanced_matrix_operations",
                step_data=step_data,
                training_input=training_input,
                execution_metadata=execution_metadata,
                artifacts_generated=artifacts_generated,
                metrics_calculated=metrics_calculated,
                errors_encountered=[] if pipeline_state.get("step7_enhanced_matrix_operations", {}).get("status") == "completed" else ["Matrix operations failed"]
            )
            
            # Log the report
            report_name = log_step_report(
                config=self.config,
                step_name="step7_enhanced_matrix_operations",
                report_data=report_data,
                report_type="matrix_operations_report",
                additional_metadata={
                    "matrix_operations_success": pipeline_state.get("step7_enhanced_matrix_operations", {,
                    "asset": symbol,
                    "lookback_period": self.config.get("lookback_days", 1095),
                    "project_version": self.config.get("project_version", "1.0.0"),
                }).get("status") == "completed",
                    "matrix_operations_count": len(matrix_results) if matrix_results else 0,
                    "timeframe": timeframe,
                }
            )
            self.logger.info(f"✅ Logged matrix operations report: {report_name}")
            
            # Log matrix results
            if matrix_results:
                matrix_report_name = log_step_report(
                    config=self.config,
                    step_name="step7_enhanced_matrix_operations",
                    report_data=matrix_results,
                    report_type="matrix_results",
                    additional_metadata={
                        "matrix_operations_count": len(matrix_results),
                        "timeframe": timeframe,
                    ,
                    "asset": symbol,
                    "lookback_period": self.config.get("lookback_days", 1095),
                    "project_version": self.config.get("project_version", "1.0.0"),
                }
                )
                self.logger.info(f"✅ Logged matrix results: {matrix_report_name}")
            
            # Log quality metrics
            if quality_metrics:
                quality_report_name = log_step_report(
                    config=self.config,
                    step_name="step7_enhanced_matrix_operations",
                    report_data=quality_metrics,
                    report_type="quality_metrics",
                    additional_metadata={
                        "overall_quality_score": quality_metrics.get("overall_quality", 0.0),
                        "timeframe": timeframe,
                    ,
                    "asset": symbol,
                    "lookback_period": self.config.get("lookback_days", 1095),
                    "project_version": self.config.get("project_version", "1.0.0"),
                }
                )
                self.logger.info(f"✅ Logged quality metrics: {quality_report_name}")
            
            # Log metrics
            log_step_metrics(
                config=self.config,
                step_name="step7_enhanced_matrix_operations",
                metrics=metrics_calculated,
                additional_metadata={
                    "metrics_type": "matrix_operations_performance",
                    "timeframe": timeframe,
                ,
                    "asset": symbol,
                    "lookback_period": self.config.get("lookback_days", 1095),
                    "project_version": self.config.get("project_version", "1.0.0"),
                }
            )
            
            self.logger.info("✅ Step 7 artifacts and reports logged successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log step 7 artifacts and reports: {e}")
            # Don't fail the step if MLflow logging fails

    def _prepare_matrix_operations_config(
        self, 
        df: pd.DataFrame, 
        symbol: str, 
        exchange: str, 
        timeframe: str
    ) -> dict[str, Any]:
        """Prepare configuration for matrix operations."""
        
        # Identify SR features for specialized analysis (comprehensive list)
        sr_features = [col for col in df.columns if any(keyword in col.lower() for keyword in [
            # Basic SR features
            "sr_", "support", "resistance", "proximity", "sr_distance",
            "sr_proximity", "sr_outcome", "normalized_distance", "sr_proximity_score",
            "strength_score", "clarity_factor", "directional_pressure", "sr_score",
            "delta_sr_score", "isolation_score", "sr_level", "sr_multi_timeframe", 
            "support_", "resistance_",
            
            # Enhanced SR features from SR breakout predictor
            "sr_enhanced_support_strength", "sr_enhanced_resistance_strength",
            "sr_clusters_detected", "sr_noise_points", "sr_clustering_quality",
            "sr_fibonacci_levels", "sr_elliott_waves", "sr_order_flow_poc",
            "sr_order_flow_hvns", "sr_order_flow_imbalances",
            "sr_pivot_level_pct", "sr_support_1_pct", "sr_support_2_pct", "sr_resistance_1_pct", "sr_resistance_2_pct",
            
            # SR optimization features from SR detection optimization
            "sr_optimized_method_weights", "sr_optimized_strength_weights",
            "sr_optimized_dbscan_eps", "sr_optimized_dbscan_min_samples",
            "sr_optimized_fibonacci_sensitivity", "sr_optimized_elliott_confidence",
            "sr_optimized_order_flow_threshold", "sr_optimized_tf_",
            "sr_optimization_score",
            
            # Additional SR features
            "sr_distance", "sr_proximity", "sr_zone_width", "sr_nearest_support",
            "sr_nearest_resistance", "sr_total_support_levels", "sr_total_resistance_levels",
            "sr_zone_position_pct", "sr_momentum_pct", "sr_volatility_pct", "sr_trend_pct"
        ])]
        
        # Basic matrix operations configuration
        config = {
            "enable_gpu_acceleration": self.step_config.get("enable_gpu_acceleration", False),
            "enable_sparse_optimizations": self.step_config.get("enable_sparse_optimizations", True),
            "enable_memory_optimization": self.step_config.get("enable_memory_optimization", True),
            "enable_parallel_processing": self.step_config.get("enable_parallel_processing", True),
            
            # Quality thresholds
            "condition_number_threshold": self.step_config.get("condition_number_threshold", 1e12),
            "min_eigenvalue_threshold": self.step_config.get("min_eigenvalue_threshold", 1e-10),
            "correlation_threshold": self.step_config.get("correlation_threshold", 0.8),
            "memory_threshold_gb": self.step_config.get("memory_threshold_gb", 8.0),
            
            # Performance settings
            "batch_size": self.step_config.get("batch_size", 1000),
            "max_iterations": self.step_config.get("max_iterations", 1000),
            "tolerance": self.step_config.get("tolerance", 1e-6),
            
            # Data-specific settings
            "data_shape": df.shape,
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            
            # SR-specific settings
            "sr_features": sr_features,
            "sr_feature_count": len(sr_features),
            "enable_sr_analysis": len(sr_features) > 0,
            "sr_correlation_threshold": self.step_config.get("sr_correlation_threshold", 0.7),
            "sr_condition_number_threshold": self.step_config.get("sr_condition_number_threshold", 1e10),
        }
        
        self.logger.info(f"🔧 Matrix operations configuration prepared:")
        self.logger.info(f"   - Total features: {len(df.columns)}")
        self.logger.info(f"   - SR features: {len(sr_features)}")
        self.logger.info(f"   - Numeric features: {len(config['numeric_columns'])}")
        
        return config

    async def _execute_matrix_operations(
        self, 
        df: pd.DataFrame, 
        config: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute matrix operations on the data."""
        
        results = {}
        
        # Get numeric columns for matrix operations
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) == 0:
            self.logger.warning("⚠️ No numeric columns found for matrix operations")
            return {"error": "No numeric columns available"}
        
        self.logger.info(f"🔢 Performing matrix operations on {len(numeric_df.columns)} numeric columns")
        
        # Standard matrix operations
        results.update(await self._execute_standard_matrix_operations(numeric_df, config))
        
        # SR-specific matrix operations
        if config.get("enable_sr_analysis", False) and config.get("sr_features"):
            self.logger.info("🎯 Performing SR-specific matrix operations...")
            results["sr_analysis"] = await self._execute_sr_matrix_operations(df, config)
            
            # Enhanced SR analysis using SR breakout predictor features
            results["sr_enhanced_analysis"] = await self._execute_enhanced_sr_analysis(df, config)
            
            # SR optimization analysis
            results["sr_optimization_analysis"] = await self._execute_sr_optimization_analysis(df, config)
        
        return results

    async def _execute_standard_matrix_operations(
        self, 
        numeric_df: pd.DataFrame, 
        config: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute standard matrix operations."""
        results = {}
        
        # 1. Correlation Analysis
        self.logger.info("📊 Performing correlation analysis...")
        correlation_matrix = numeric_df.corr()
        results["correlation_analysis"] = {
            "correlation_matrix": correlation_matrix.to_dict(),
            "high_correlations": self._find_high_correlations(correlation_matrix, config["correlation_threshold"])
        }
        
        # 2. Condition Number Check
        self.logger.info("🔍 Checking condition number...")
        condition_number = np.linalg.cond(numeric_df.values)
        results["condition_number_check"] = {
            "condition_number": float(condition_number),
            "is_well_conditioned": condition_number < config["condition_number_threshold"]
        }
        
        # 3. Eigenvalue Analysis
        self.logger.info("📈 Performing eigenvalue analysis...")
        eigenvalues = np.linalg.eigvals(numeric_df.values)
        results["eigenvalue_analysis"] = {
            "eigenvalues": eigenvalues.tolist(),
            "min_eigenvalue": float(np.min(eigenvalues)),
            "max_eigenvalue": float(np.max(eigenvalues)),
            "eigenvalue_ratio": float(np.max(eigenvalues) / np.min(eigenvalues)),
            "small_eigenvalues": int(np.sum(np.abs(eigenvalues) < config["min_eigenvalue_threshold"]))
        }
        
        # 4. Singular Value Decomposition
        self.logger.info("🔧 Performing SVD analysis...")
        try:
            U, s, Vt = np.linalg.svd(numeric_df.values, full_matrices=False)
            results["singular_value_decomposition"] = {
                "singular_values": s.tolist(),
                "rank": int(np.sum(s > config["min_eigenvalue_threshold"])),
                "condition_number_svd": float(s[0] / s[-1]) if len(s) > 1 else float('inf')
            }
        except Exception as e:
            self.logger.warning(f"⚠️ SVD failed: {str(e)}")
            results["singular_value_decomposition"] = {"error": str(e)}
        
        # 5. Matrix Rank Analysis
        self.logger.info("📊 Analyzing matrix rank...")
        try:
            rank = np.linalg.matrix_rank(numeric_df.values)
            results["matrix_rank_analysis"] = {
                "rank": int(rank),
                "full_rank": rank == min(numeric_df.shape),
                "rank_deficiency": min(numeric_df.shape) - rank
            }
        except Exception as e:
            self.logger.warning(f"⚠️ Rank analysis failed: {str(e)}")
            results["matrix_rank_analysis"] = {"error": str(e)}
        
        return results

    async def _execute_sr_matrix_operations(
        self, 
        df: pd.DataFrame, 
        config: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute SR-specific matrix operations."""
        try:
            sr_features = config.get("sr_features", [])
            if not sr_features:
                return {"error": "No SR features found"}
            
            # Get SR feature columns
            sr_df = df[sr_features].select_dtypes(include=[np.number])
            
            if len(sr_df.columns) == 0:
                return {"error": "No numeric SR features found"}
            
            self.logger.info(f"🎯 Analyzing {len(sr_df.columns)} SR features")
            
            results = {}
            
            # 1. SR Feature Correlation Analysis
            self.logger.info("📊 Performing SR feature correlation analysis...")
            sr_correlation_matrix = sr_df.corr()
            results["sr_correlation_analysis"] = {
                "correlation_matrix": sr_correlation_matrix.to_dict(),
                "high_correlations": self._find_high_correlations(sr_correlation_matrix, config["sr_correlation_threshold"]),
                "sr_feature_count": len(sr_df.columns)
            }
            
            # 2. SR Feature Condition Number
            self.logger.info("🔍 Checking SR feature condition number...")
            sr_condition_number = np.linalg.cond(sr_df.values)
            results["sr_condition_number"] = {
                "condition_number": float(sr_condition_number),
                "is_well_conditioned": sr_condition_number < config["sr_condition_number_threshold"]
            }
            
            # 3. SR Feature Eigenvalue Analysis
            self.logger.info("📈 Performing SR feature eigenvalue analysis...")
            sr_eigenvalues = np.linalg.eigvals(sr_df.values)
            results["sr_eigenvalue_analysis"] = {
                "eigenvalues": sr_eigenvalues.tolist(),
                "min_eigenvalue": float(np.min(sr_eigenvalues)),
                "max_eigenvalue": float(np.max(sr_eigenvalues)),
                "eigenvalue_ratio": float(np.max(sr_eigenvalues) / np.min(sr_eigenvalues)),
                "small_eigenvalues": int(np.sum(np.abs(sr_eigenvalues) < config["min_eigenvalue_threshold"]))
            }
            
            # 4. SR Feature Clustering Analysis
            self.logger.info("🔧 Performing SR feature clustering analysis...")
            results["sr_clustering_analysis"] = self._analyze_sr_feature_clusters(sr_df)
            
            # 5. SR Feature Stability Analysis
            self.logger.info("📊 Analyzing SR feature stability...")
            results["sr_stability_analysis"] = self._analyze_sr_feature_stability(sr_df)
            
            # 6. SR Feature Importance Analysis
            self.logger.info("🎯 Analyzing SR feature importance...")
            results["sr_importance_analysis"] = self._analyze_sr_feature_importance(sr_df)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in SR matrix operations: {e}")
            return {"error": str(e)}

    async def _execute_enhanced_sr_analysis(
        self, 
        df: pd.DataFrame, 
        config: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute enhanced SR analysis using SR breakout predictor features."""
        try:
            # Identify enhanced SR features
            enhanced_sr_features = [col for col in df.columns if any(keyword in col.lower() for keyword in [
                "sr_enhanced_", "sr_clusters_", "sr_fibonacci_", "sr_elliott_", "sr_order_flow_",
                "sr_pivot_", "sr_support_1_pct", "sr_support_2_pct", "sr_resistance_1_pct", "sr_resistance_2_pct"
            ])]
            
            if not enhanced_sr_features:
                return {"error": "No enhanced SR features found"}
            
            enhanced_sr_df = df[enhanced_sr_features].select_dtypes(include=[np.number])
            
            if len(enhanced_sr_df.columns) == 0:
                return {"error": "No numeric enhanced SR features found"}
            
            self.logger.info(f"🎯 Analyzing {len(enhanced_sr_df.columns)} enhanced SR features")
            
            results = {}
            
            # 1. Enhanced SR Feature Correlation Analysis
            self.logger.info("📊 Performing enhanced SR feature correlation analysis...")
            enhanced_correlation_matrix = enhanced_sr_df.corr()
            results["enhanced_sr_correlation_analysis"] = {
                "correlation_matrix": enhanced_correlation_matrix.to_dict(),
                "high_correlations": self._find_high_correlations(enhanced_correlation_matrix, config["sr_correlation_threshold"]),
                "enhanced_sr_feature_count": len(enhanced_sr_df.columns)
            }
            
            # 2. Enhanced SR Feature Clustering Analysis
            self.logger.info("🔧 Performing enhanced SR feature clustering analysis...")
            results["enhanced_sr_clustering_analysis"] = self._analyze_enhanced_sr_feature_clusters(enhanced_sr_df)
            
            # 3. Enhanced SR Feature Stability Analysis
            self.logger.info("📊 Analyzing enhanced SR feature stability...")
            results["enhanced_sr_stability_analysis"] = self._analyze_enhanced_sr_feature_stability(enhanced_sr_df)
            
            # 4. Enhanced SR Feature Importance Analysis
            self.logger.info("🎯 Analyzing enhanced SR feature importance...")
            results["enhanced_sr_importance_analysis"] = self._analyze_enhanced_sr_feature_importance(enhanced_sr_df)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in enhanced SR analysis: {e}")
            return {"error": str(e)}

    async def _execute_sr_optimization_analysis(
        self, 
        df: pd.DataFrame, 
        config: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute SR optimization analysis using optimization features."""
        try:
            # Identify SR optimization features
            optimization_features = [col for col in df.columns if any(keyword in col.lower() for keyword in [
                "sr_optimized_", "sr_optimization_"
            ])]
            
            if not optimization_features:
                return {"error": "No SR optimization features found"}
            
            optimization_df = df[optimization_features].select_dtypes(include=[np.number])
            
            if len(optimization_df.columns) == 0:
                return {"error": "No numeric SR optimization features found"}
            
            self.logger.info(f"🎯 Analyzing {len(optimization_df.columns)} SR optimization features")
            
            results = {}
            
            # 1. SR Optimization Feature Correlation Analysis
            self.logger.info("📊 Performing SR optimization feature correlation analysis...")
            optimization_correlation_matrix = optimization_df.corr()
            results["sr_optimization_correlation_analysis"] = {
                "correlation_matrix": optimization_correlation_matrix.to_dict(),
                "high_correlations": self._find_high_correlations(optimization_correlation_matrix, config["sr_correlation_threshold"]),
                "optimization_feature_count": len(optimization_df.columns)
            }
            
            # 2. SR Optimization Parameter Analysis
            self.logger.info("🔧 Analyzing SR optimization parameters...")
            results["sr_optimization_parameter_analysis"] = self._analyze_sr_optimization_parameters(optimization_df)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in SR optimization analysis: {e}")
            return {"error": str(e)}

    def _analyze_enhanced_sr_feature_clusters(self, enhanced_sr_df: pd.DataFrame) -> dict[str, Any]:
        """Analyze enhanced SR feature clusters."""
        try:
            # Group enhanced SR features by type
            feature_groups = {
                "enhanced_strength": [col for col in enhanced_sr_df.columns if "enhanced_strength" in col],
                "clustering": [col for col in enhanced_sr_df.columns if "clusters" in col or "noise" in col],
                "fibonacci": [col for col in enhanced_sr_df.columns if "fibonacci" in col],
                "elliott": [col for col in enhanced_sr_df.columns if "elliott" in col],
                "order_flow": [col for col in enhanced_sr_df.columns if "order_flow" in col],
                "pivot": [col for col in enhanced_sr_df.columns if "pivot" in col or "support_1" in col or "resistance_1" in col]
            }
            
            # Calculate group statistics
            group_stats = {}
            for group_name, group_features in feature_groups.items():
                if group_features:
                    group_data = enhanced_sr_df[group_features]
                    group_stats[group_name] = {
                        "feature_count": len(group_features),
                        "mean_correlation": group_data.corr().abs().mean().mean(),
                        "mean_variance": group_data.var().mean(),
                        "features": group_features
                    }
            
            return {
                "feature_groups": group_stats,
                "total_groups": len([g for g in group_stats.values() if g["feature_count"] > 0]),
                "group_correlations": self._calculate_group_correlations(enhanced_sr_df, feature_groups)
            }
            
        except Exception as e:
            return {"error": str(e)}

    def _analyze_enhanced_sr_feature_stability(self, enhanced_sr_df: pd.DataFrame) -> dict[str, Any]:
        """Analyze enhanced SR feature stability."""
        try:
            stability_metrics = {}
            
            for column in enhanced_sr_df.columns:
                values = enhanced_sr_df[column].dropna()
                if len(values) > 1:
                    # Coefficient of variation
                    cv = values.std() / abs(values.mean()) if values.mean() != 0 else float('inf')
                    
                    # Feature type classification
                    feature_type = "unknown"
                    if "enhanced_strength" in column:
                        feature_type = "enhanced_strength"
                    elif "clusters" in column or "noise" in column:
                        feature_type = "clustering"
                    elif "fibonacci" in column:
                        feature_type = "fibonacci"
                    elif "elliott" in column:
                        feature_type = "elliott"
                    elif "order_flow" in column:
                        feature_type = "order_flow"
                    elif "pivot" in column or "support_" in column or "resistance_" in column:
                        feature_type = "pivot"
                    elif "momentum_pct" in column or "volatility_pct" in column or "trend_pct" in column:
                        feature_type = "momentum"
                    
                    stability_metrics[column] = {
                        "coefficient_of_variation": float(cv),
                        "feature_type": feature_type,
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                        "stability_score": 1.0 / (1.0 + cv) if cv != float('inf') else 0.0
                    }
            
            # Group stability by feature type
            type_stability = {}
            for metrics in stability_metrics.values():
                feature_type = metrics["feature_type"]
                if feature_type not in type_stability:
                    type_stability[feature_type] = []
                type_stability[feature_type].append(metrics["stability_score"])
            
            # Calculate average stability by type
            for feature_type, scores in type_stability.items():
                type_stability[feature_type] = {
                    "average_stability": np.mean(scores),
                    "stability_count": len(scores)
                }
            
            return {
                "feature_stability": stability_metrics,
                "type_stability": type_stability,
                "overall_stability": np.mean([m["stability_score"] for m in stability_metrics.values()])
            }
            
        except Exception as e:
            return {"error": str(e)}

    def _analyze_enhanced_sr_feature_importance(self, enhanced_sr_df: pd.DataFrame) -> dict[str, Any]:
        """Analyze enhanced SR feature importance."""
        try:
            # Calculate variance-based importance
            variances = enhanced_sr_df.var()
            variance_importance = variances.sort_values(ascending=False)
            
            # Calculate correlation-based importance
            correlation_matrix = enhanced_sr_df.corr()
            avg_correlations = correlation_matrix.abs().mean()
            correlation_importance = (1.0 / (1.0 + avg_correlations)).sort_values(ascending=False)
            
            # Combined importance score
            combined_importance = (variance_importance + correlation_importance) / 2
            combined_importance = combined_importance.sort_values(ascending=False)
            
            # Group importance by feature type
            feature_importance_by_type = {
                "enhanced_strength": [],
                "clustering": [],
                "fibonacci": [],
                "elliott": [],
                "order_flow": [],
                "pivot": [],
                "momentum": []
            }
            
            for feature, importance in combined_importance.items():
                if "enhanced_strength" in feature:
                    feature_importance_by_type["enhanced_strength"].append((feature, importance))
                elif "clusters" in feature or "noise" in feature:
                    feature_importance_by_type["clustering"].append((feature, importance))
                elif "fibonacci" in feature:
                    feature_importance_by_type["fibonacci"].append((feature, importance))
                elif "elliott" in feature:
                    feature_importance_by_type["elliott"].append((feature, importance))
                elif "order_flow" in feature:
                    feature_importance_by_type["order_flow"].append((feature, importance))
                elif "pivot" in feature or "support_" in feature or "resistance_" in feature:
                    feature_importance_by_type["pivot"].append((feature, importance))
                elif "momentum_pct" in feature or "volatility_pct" in feature or "trend_pct" in feature:
                    feature_importance_by_type["momentum"].append((feature, importance))
            
            # Sort each group by importance
            for feature_type in feature_importance_by_type:
                feature_importance_by_type[feature_type].sort(key=lambda x: x[1], reverse=True)
            
            return {
                "variance_importance": variance_importance.to_dict(),
                "correlation_importance": correlation_importance.to_dict(),
                "combined_importance": combined_importance.to_dict(),
                "importance_by_type": feature_importance_by_type,
                "top_features": combined_importance.head(10).index.tolist()
            }
            
        except Exception as e:
            return {"error": str(e)}



    def _analyze_sr_optimization_parameters(self, optimization_df: pd.DataFrame) -> dict[str, Any]:
        """Analyze SR optimization parameters."""
        try:
            # Identify parameter features
            parameter_features = [col for col in optimization_df.columns if "sr_optimized_" in col and any(param in col for param in [
                "method_weights", "strength_weights", "dbscan", "fibonacci", "elliott", "order_flow", "tf_"
            ])]
            
            if not parameter_features:
                return {"error": "No parameter features found"}
            
            parameter_data = optimization_df[parameter_features]
            
            # Calculate parameter statistics
            parameter_stats = {}
            for col in parameter_data.columns:
                values = parameter_data[col].dropna()
                if len(values) > 0:
                    parameter_stats[col] = {
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "median": float(values.median())
                    }
            
            # Group parameters by type
            parameter_groups = {
                "weights": [col for col in parameter_features if "weights" in col],
                "dbscan": [col for col in parameter_features if "dbscan" in col],
                "advanced": [col for col in parameter_features if any(adv in col for adv in ["fibonacci", "elliott", "order_flow"])],
                "timeframe": [col for col in parameter_features if "tf_" in col]
            }
            
            return {
                "parameter_features": parameter_features,
                "parameter_statistics": parameter_stats,
                "parameter_groups": parameter_groups,
                "parameter_correlations": parameter_data.corr().to_dict()
            }
            
        except Exception as e:
            return {"error": str(e)}

    def _calculate_group_correlations(self, df: pd.DataFrame, feature_groups: dict[str, list]) -> dict[str, float]:
        """Calculate correlations between feature groups."""
        try:
            group_correlations = {}
            
            for group1_name, group1_features in feature_groups.items():
                for group2_name, group2_features in feature_groups.items():
                    if group1_name < group2_name and group1_features and group2_features:
                        # Calculate average correlation between groups
                        group1_data = df[group1_features]
                        group2_data = df[group2_features]
                        
                        # Calculate cross-correlations
                        cross_corr = group1_data.corrwith(group2_data, axis=0)
                        avg_correlation = cross_corr.abs().mean()
                        
                        group_correlations[f"{group1_name}_vs_{group2_name}"] = float(avg_correlation)
            
            return group_correlations
            
        except Exception as e:
            return {"error": str(e)}

    def _analyze_sr_feature_clusters(self, sr_df: pd.DataFrame) -> dict[str, Any]:
        """Analyze SR feature clusters."""
        try:
            # Simple clustering analysis based on correlation
            correlation_matrix = sr_df.corr()
            
            # Find feature groups with high correlation
            high_corr_groups = []
            processed_features = set()
            
            for i, feature1 in enumerate(sr_df.columns):
                if feature1 in processed_features:
                    continue
                    
                group = [feature1]
                processed_features.add(feature1)
                
                for feature2 in sr_df.columns[i+1:]:
                    if feature2 not in processed_features:
                        corr = abs(correlation_matrix.loc[feature1, feature2])
                        if corr > 0.8:  # High correlation threshold
                            group.append(feature2)
                            processed_features.add(feature2)
                
                if len(group) > 1:
                    high_corr_groups.append(group)
            
            return {
                "high_correlation_groups": high_corr_groups,
                "group_count": len(high_corr_groups),
                "total_grouped_features": sum(len(group) for group in high_corr_groups)
            }
            
        except Exception as e:
            return {"error": str(e)}

    def _analyze_sr_feature_stability(self, sr_df: pd.DataFrame) -> dict[str, Any]:
        """Analyze SR feature stability over time."""
        try:
            # Calculate stability metrics for each SR feature
            stability_metrics = {}
            
            for column in sr_df.columns:
                values = sr_df[column].dropna()
                if len(values) > 1:
                    # Coefficient of variation (lower = more stable)
                    cv = values.std() / abs(values.mean()) if values.mean() != 0 else float('inf')
                    
                    # Range stability
                    range_stability = 1.0 / (1.0 + (values.max() - values.min()))
                    
                    # Entropy-based stability
                    entropy_stability = self._calculate_entropy_stability(values)
                    
                    stability_metrics[column] = {
                        "coefficient_of_variation": float(cv),
                        "range_stability": float(range_stability),
                        "entropy_stability": float(entropy_stability),
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                        "min": float(values.min()),
                        "max": float(values.max())
                    }
            
            # Overall stability metrics
            overall_stability = {
                "mean_cv": np.mean([metrics["coefficient_of_variation"] for metrics in stability_metrics.values()]),
                "mean_range_stability": np.mean([metrics["range_stability"] for metrics in stability_metrics.values()]),
                "mean_entropy_stability": np.mean([metrics["entropy_stability"] for metrics in stability_metrics.values()]),
                "stable_features": len([cv for cv in [metrics["coefficient_of_variation"] for metrics in stability_metrics.values()] if cv < 0.5]),
                "unstable_features": len([cv for cv in [metrics["coefficient_of_variation"] for metrics in stability_metrics.values()] if cv > 1.0])
            }
            
            return {
                "feature_stability": stability_metrics,
                "overall_stability": overall_stability
            }
            
        except Exception as e:
            return {"error": str(e)}

    def _analyze_feature_stability_over_time(self, df: pd.DataFrame, window_sizes: list[int] = None) -> dict[str, Any]:
        """Analyze feature stability over different time windows."""
        try:
            if window_sizes is None:
                window_sizes = [100, 500, 1000]  # Default window sizes
            
            stability_over_time = {}
            
            for column in df.select_dtypes(include=[np.number]).columns:
                values = df[column].dropna()
                if len(values) < min(window_sizes):
                    continue
                
                column_stability = {}
                
                for window_size in window_sizes:
                    if len(values) < window_size:
                        continue
                    
                    # Rolling statistics
                    rolling_mean = values.rolling(window=window_size, min_periods=window_size//2).mean()
                    rolling_std = values.rolling(window=window_size, min_periods=window_size//2).std()
                    
                    # Stability metrics for this window
                    mean_stability = 1.0 / (1.0 + rolling_std.std())  # Lower std of rolling std = more stable
                    variance_stability = 1.0 / (1.0 + rolling_std.var())  # Lower variance of rolling std = more stable
                    
                    # Entropy stability over time
                    entropy_stability = self._calculate_rolling_entropy_stability(values, window_size)
                    
                    column_stability[f"window_{window_size}"] = {
                        "mean_stability": float(mean_stability),
                        "variance_stability": float(variance_stability),
                        "entropy_stability": float(entropy_stability),
                        "rolling_mean_std": float(rolling_mean.std()),
                        "rolling_std_std": float(rolling_std.std())
                    }
                
                if column_stability:
                    stability_over_time[column] = column_stability
            
            # Overall time-based stability metrics
            overall_time_stability = {}
            for window_size in window_sizes:
                window_stabilities = []
                for column_data in stability_over_time.values():
                    if f"window_{window_size}" in column_data:
                        window_stabilities.append(column_data[f"window_{window_size}"])
                
                if window_stabilities:
                    overall_time_stability[f"window_{window_size}"] = {
                        "mean_mean_stability": np.mean([w["mean_stability"] for w in window_stabilities]),
                        "mean_variance_stability": np.mean([w["variance_stability"] for w in window_stabilities]),
                        "mean_entropy_stability": np.mean([w["entropy_stability"] for w in window_stabilities]),
                        "stable_features_count": len([w for w in window_stabilities if w["mean_stability"] > 0.7])
                    }
            
            return {
                "feature_stability_over_time": stability_over_time,
                "overall_time_stability": overall_time_stability
            }
            
        except Exception as e:
            return {"error": str(e)}

    def _analyze_distribution_stability(self, df: pd.DataFrame, reference_period: int = 1000) -> dict[str, Any]:
        """Analyze distribution stability using PSI and other distribution metrics."""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            distribution_stability = {}
            
            for column in numeric_df.columns:
                values = numeric_df[column].dropna()
                if len(values) < reference_period * 2:
                    continue
                
                # Split data into reference and current periods
                reference_data = values.iloc[:reference_period]
                current_data = values.iloc[reference_period:]
                
                # Calculate Population Stability Index (PSI)
                psi = self._calculate_psi(reference_data, current_data)
                
                # Calculate Kolmogorov-Smirnov test
                ks_stat, ks_pvalue = self._calculate_ks_test(reference_data, current_data)
                
                # Calculate distribution moments stability
                moment_stability = self._calculate_moment_stability(reference_data, current_data)
                
                # Calculate entropy-based distribution stability
                entropy_stability = self._calculate_entropy_distribution_stability(reference_data, current_data)
                
                distribution_stability[column] = {
                    "psi": float(psi),
                    "ks_statistic": float(ks_stat),
                    "ks_pvalue": float(ks_pvalue),
                    "moment_stability": moment_stability,
                    "entropy_stability": float(entropy_stability),
                    "distribution_shift": "significant" if psi > 0.25 else "moderate" if psi > 0.1 else "stable"
                }
            
            # Overall distribution stability metrics
            overall_distribution_stability = {
                "mean_psi": np.mean([metrics["psi"] for metrics in distribution_stability.values()]),
                "stable_distributions": len([metrics for metrics in distribution_stability.values() if metrics["psi"] < 0.1]),
                "moderate_shifts": len([metrics for metrics in distribution_stability.values() if 0.1 <= metrics["psi"] <= 0.25]),
                "significant_shifts": len([metrics for metrics in distribution_stability.values() if metrics["psi"] > 0.25]),
                "mean_entropy_stability": np.mean([metrics["entropy_stability"] for metrics in distribution_stability.values()])
            }
            
            return {
                "feature_distribution_stability": distribution_stability,
                "overall_distribution_stability": overall_distribution_stability
            }
            
        except Exception as e:
            return {"error": str(e)}

    def _analyze_feature_importance_stability(self, df: pd.DataFrame, target_column: str = None, 
                                           window_sizes: list[int] = None) -> dict[str, Any]:
        """Analyze stability of feature importance over time."""
        try:
            if window_sizes is None:
                window_sizes = [500, 1000, 2000]
            
            numeric_df = df.select_dtypes(include=[np.number])
            if target_column and target_column in numeric_df.columns:
                target = numeric_df[target_column]
                features_df = numeric_df.drop(columns=[target_column])
            else:
                # Use first column as target if none specified
                target = numeric_df.iloc[:, 0]
                features_df = numeric_df.iloc[:, 1:]
            
            importance_stability = {}
            
            for column in features_df.columns:
                values = features_df[column].dropna()
                target_values = target.loc[values.index].dropna()
                
                if len(values) < min(window_sizes) or len(target_values) < min(window_sizes):
                    continue
                
                column_importance_stability = {}
                
                for window_size in window_sizes:
                    if len(values) < window_size:
                        continue
                    
                    # Rolling correlation importance
                    rolling_corr = self._calculate_rolling_correlation(values, target_values, window_size)
                    corr_stability = 1.0 / (1.0 + rolling_corr.std())
                    
                    # Rolling mutual information importance
                    rolling_mi = self._calculate_rolling_mutual_information(values, target_values, window_size)
                    mi_stability = 1.0 / (1.0 + rolling_mi.std()) if rolling_mi.std() > 0 else 1.0
                    
                    # Rolling variance importance
                    rolling_var = values.rolling(window=window_size, min_periods=window_size//2).var()
                    var_stability = 1.0 / (1.0 + rolling_var.std())
                    
                    # Entropy-based importance stability
                    entropy_importance_stability = self._calculate_entropy_importance_stability(values, target_values, window_size)
                    
                    column_importance_stability[f"window_{window_size}"] = {
                        "correlation_stability": float(corr_stability),
                        "mutual_info_stability": float(mi_stability),
                        "variance_stability": float(var_stability),
                        "entropy_importance_stability": float(entropy_importance_stability),
                        "overall_importance_stability": float((corr_stability + mi_stability + var_stability + entropy_importance_stability) / 4)
                    }
                
                if column_importance_stability:
                    importance_stability[column] = column_importance_stability
            
            # Overall importance stability metrics
            overall_importance_stability = {}
            for window_size in window_sizes:
                window_stabilities = []
                for column_data in importance_stability.values():
                    if f"window_{window_size}" in column_data:
                        window_stabilities.append(column_data[f"window_{window_size}"])
                
                if window_stabilities:
                    overall_importance_stability[f"window_{window_size}"] = {
                        "mean_correlation_stability": np.mean([w["correlation_stability"] for w in window_stabilities]),
                        "mean_mutual_info_stability": np.mean([w["mutual_info_stability"] for w in window_stabilities]),
                        "mean_variance_stability": np.mean([w["variance_stability"] for w in window_stabilities]),
                        "mean_entropy_importance_stability": np.mean([w["entropy_importance_stability"] for w in window_stabilities]),
                        "mean_overall_stability": np.mean([w["overall_importance_stability"] for w in window_stabilities]),
                        "stable_features_count": len([w for w in window_stabilities if w["overall_importance_stability"] > 0.7])
                    }
            
            return {
                "feature_importance_stability": importance_stability,
                "overall_importance_stability": overall_importance_stability
            }
            
        except Exception as e:
            return {"error": str(e)}

    def _analyze_sr_feature_importance(self, sr_df: pd.DataFrame) -> dict[str, Any]:
        """Analyze SR feature importance based on variance and correlation."""
        try:
            # Calculate variance-based importance
            variances = sr_df.var()
            variance_importance = variances.sort_values(ascending=False)
            
            # Calculate correlation-based importance (inverse of average correlation)
            correlation_matrix = sr_df.corr()
            avg_correlations = correlation_matrix.abs().mean()
            correlation_importance = (1.0 / (1.0 + avg_correlations)).sort_values(ascending=False)
            
            # Combined importance score
            combined_importance = (variance_importance + correlation_importance) / 2
            combined_importance = combined_importance.sort_values(ascending=False)
            
            return {
                "variance_importance": variance_importance.to_dict(),
                "correlation_importance": correlation_importance.to_dict(),
                "combined_importance": combined_importance.to_dict(),
                "top_features": combined_importance.head(10).index.tolist()
            }
            
        except Exception as e:
            return {"error": str(e)}

    def _calculate_quality_metrics(self, df: pd.DataFrame, matrix_results: dict[str, Any]) -> dict[str, Any]:
        """Calculate comprehensive quality metrics for the feature matrix."""
        try:
            self.logger.info("📊 Calculating quality metrics...")
            
            numeric_df = df.select_dtypes(include=[np.number])
            quality_metrics = {}
            
            # 1. Data Completeness Metrics
            quality_metrics["completeness"] = {
                "total_cells": numeric_df.size,
                "missing_cells": numeric_df.isnull().sum().sum(),
                "missing_ratio": float(numeric_df.isnull().sum().sum() / numeric_df.size),
                "complete_rows": int(numeric_df.dropna().shape[0]),
                "complete_columns": int(numeric_df.dropna(axis=1).shape[1])
            }
            
            # 2. Feature Variance Metrics
            variances = numeric_df.var()
            quality_metrics["variance"] = {
                "mean_variance": float(variances.mean()),
                "median_variance": float(variances.median()),
                "min_variance": float(variances.min()),
                "max_variance": float(variances.max()),
                "low_variance_features": int((variances < 1e-6).sum()),
                "zero_variance_features": int((variances == 0).sum())
            }
            
            # 3. Feature Correlation Metrics
            if "correlation_analysis" in matrix_results:
                corr_matrix = pd.DataFrame(matrix_results["correlation_analysis"]["correlation_matrix"])
                high_corrs = matrix_results["correlation_analysis"]["high_correlations"]
                
                quality_metrics["correlation"] = {
                    "mean_correlation": float(corr_matrix.abs().mean().mean()),
                    "max_correlation": float(corr_matrix.abs().max().max()),
                    "high_correlation_pairs": len(high_corrs),
                    "correlation_threshold": 0.8
                }
            
            # 4. Numerical Stability Metrics
            if "condition_number_check" in matrix_results:
                quality_metrics["numerical_stability"] = {
                    "condition_number": matrix_results["condition_number_check"]["condition_number"],
                    "is_well_conditioned": matrix_results["condition_number_check"]["is_well_conditioned"],
                    "condition_threshold": 1e12
                }
            
            # 5. Dimensionality Metrics
            if "matrix_rank_analysis" in matrix_results:
                quality_metrics["dimensionality"] = {
                    "matrix_rank": matrix_results["matrix_rank_analysis"]["rank"],
                    "full_rank": matrix_results["matrix_rank_analysis"]["full_rank"],
                    "rank_deficiency": matrix_results["matrix_rank_analysis"]["rank_deficiency"],
                    "effective_dimensions": matrix_results["matrix_rank_analysis"]["rank"]
                }
            
            # 6. Feature Distribution Metrics
            quality_metrics["distribution"] = {
                "skewness_mean": float(numeric_df.skew().mean()),
                "skewness_std": float(numeric_df.skew().std()),
                "kurtosis_mean": float(numeric_df.kurtosis().mean()),
                "kurtosis_std": float(numeric_df.kurtosis().std()),
                "high_skew_features": int((abs(numeric_df.skew()) > 3).sum()),
                "high_kurtosis_features": int((numeric_df.kurtosis() > 10).sum())
            }
            
            # 7. Outlier Metrics
            quality_metrics["outliers"] = self._calculate_outlier_metrics(numeric_df)
            
            # 8. Memory Usage Metrics
            quality_metrics["memory"] = {
                "memory_usage_mb": float(numeric_df.memory_usage(deep=True).sum() / 1024 / 1024),
                "memory_per_feature_kb": float(numeric_df.memory_usage(deep=True).sum() / len(numeric_df.columns) / 1024),
                "data_types": numeric_df.dtypes.value_counts().to_dict()
            }
            
            # 9. Stability Metrics
            quality_metrics["stability"] = self._calculate_stability_metrics(matrix_results)
            
            # 10. Overall Quality Score
            quality_metrics["overall_score"] = self._calculate_overall_quality_score(quality_metrics)
            
            self.logger.info(f"✅ Quality metrics calculated. Overall score: {quality_metrics['overall_score']:.2f}")
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating quality metrics: {str(e)}")
            return {"error": str(e)}

    def _calculate_outlier_metrics(self, df: pd.DataFrame) -> dict[str, Any]:
        """Calculate outlier metrics for features."""
        outlier_metrics = {}
        
        try:
            # IQR-based outlier detection
            outlier_counts = []
            outlier_ratios = []
            
            for col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outlier_counts.append(outliers)
                outlier_ratios.append(outliers / len(df))
            
            outlier_metrics = {
                "total_outliers": sum(outlier_counts),
                "mean_outliers_per_feature": float(np.mean(outlier_counts)),
                "max_outliers_in_feature": max(outlier_counts),
                "mean_outlier_ratio": float(np.mean(outlier_ratios)),
                "high_outlier_features": int(sum(1 for ratio in outlier_ratios if ratio > 0.1))
            }
            
        except Exception as e:
            outlier_metrics = {"error": str(e)}
        
        return outlier_metrics

    def _calculate_overall_quality_score(self, quality_metrics: dict[str, Any]) -> float:
        """Calculate overall quality score from individual metrics."""
        try:
            score = 0.0
            max_score = 0.0
            
            # Completeness score (0-25 points)
            completeness = quality_metrics.get("completeness", {})
            if "missing_ratio" in completeness:
                completeness_score = max(0, 25 * (1 - completeness["missing_ratio"]))
                score += completeness_score
                max_score += 25
            
            # Variance score (0-20 points)
            variance = quality_metrics.get("variance", {})
            if "zero_variance_features" in variance:
                zero_var_ratio = variance["zero_variance_features"] / len(quality_metrics.get("completeness", {}).get("total_cells", 1))
                variance_score = max(0, 20 * (1 - zero_var_ratio))
                score += variance_score
                max_score += 20
            
            # Correlation score (0-20 points)
            correlation = quality_metrics.get("correlation", {})
            if "high_correlation_pairs" in correlation:
                corr_score = max(0, 20 * (1 - correlation["high_correlation_pairs"] / 100))  # Penalize high correlations
                score += corr_score
                max_score += 20
            
            # Numerical stability score (0-15 points)
            stability = quality_metrics.get("numerical_stability", {})
            if "is_well_conditioned" in stability:
                stability_score = 15 if stability["is_well_conditioned"] else 5
                score += stability_score
                max_score += 15
            
            # Dimensionality score (0-10 points)
            dimensionality = quality_metrics.get("dimensionality", {})
            if "rank_deficiency" in dimensionality:
                rank_score = max(0, 10 * (1 - dimensionality["rank_deficiency"] / 100))
                score += rank_score
                max_score += 10
            
            # Distribution score (0-10 points)
            distribution = quality_metrics.get("distribution", {})
            if "high_skew_features" in distribution:
                skew_penalty = min(10, distribution["high_skew_features"] / 10)
                distribution_score = max(0, 10 - skew_penalty)
                score += distribution_score
                max_score += 10
            
            return score / max_score if max_score > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating overall quality score: {str(e)}")
            return 0.0

    def _calculate_stability_metrics(self, matrix_results: dict[str, Any]) -> dict[str, Any]:
        """Calculate comprehensive stability metrics from matrix results."""
        try:
            stability_metrics = {}
            
            # Time-based stability metrics
            if "time_based_stability" in matrix_results:
                time_stability = matrix_results["time_based_stability"]
                if "overall_time_stability" in time_stability:
                    overall_time = time_stability["overall_time_stability"]
                    stability_metrics["time_based"] = {
                        "mean_stability_score": np.mean([
                            overall_time.get(f"window_{w}", {}).get("mean_mean_stability", 0.0)
                            for w in [100, 500, 1000]
                            if f"window_{w}" in overall_time
                        ]),
                        "variance_stability_score": np.mean([
                            overall_time.get(f"window_{w}", {}).get("mean_variance_stability", 0.0)
                            for w in [100, 500, 1000]
                            if f"window_{w}" in overall_time
                        ]),
                        "entropy_stability_score": np.mean([
                            overall_time.get(f"window_{w}", {}).get("mean_entropy_stability", 0.0)
                            for w in [100, 500, 1000]
                            if f"window_{w}" in overall_time
                        ]),
                        "stable_features_count": sum([
                            overall_time.get(f"window_{w}", {}).get("stable_features_count", 0)
                            for w in [100, 500, 1000]
                            if f"window_{w}" in overall_time
                        ])
                    }
            
            # Distribution stability metrics
            if "distribution_stability" in matrix_results:
                dist_stability = matrix_results["distribution_stability"]
                if "overall_distribution_stability" in dist_stability:
                    overall_dist = dist_stability["overall_distribution_stability"]
                    stability_metrics["distribution"] = {
                        "mean_psi": overall_dist.get("mean_psi", 0.0),
                        "stable_distributions_count": overall_dist.get("stable_distributions", 0),
                        "moderate_shifts_count": overall_dist.get("moderate_shifts", 0),
                        "significant_shifts_count": overall_dist.get("significant_shifts", 0),
                        "mean_entropy_stability": overall_dist.get("mean_entropy_stability", 0.0),
                        "distribution_stability_score": 1.0 / (1.0 + overall_dist.get("mean_psi", 0.0))
                    }
            
            # Feature importance stability metrics
            if "feature_importance_stability" in matrix_results:
                imp_stability = matrix_results["feature_importance_stability"]
                if "overall_importance_stability" in imp_stability:
                    overall_imp = imp_stability["overall_importance_stability"]
                    stability_metrics["importance"] = {
                        "mean_correlation_stability": np.mean([
                            overall_imp.get(f"window_{w}", {}).get("mean_correlation_stability", 0.0)
                            for w in [500, 1000, 2000]
                            if f"window_{w}" in overall_imp
                        ]),
                        "mean_mutual_info_stability": np.mean([
                            overall_imp.get(f"window_{w}", {}).get("mean_mutual_info_stability", 0.0)
                            for w in [500, 1000, 2000]
                            if f"window_{w}" in overall_imp
                        ]),
                        "mean_variance_stability": np.mean([
                            overall_imp.get(f"window_{w}", {}).get("mean_variance_stability", 0.0)
                            for w in [500, 1000, 2000]
                            if f"window_{w}" in overall_imp
                        ]),
                        "mean_entropy_importance_stability": np.mean([
                            overall_imp.get(f"window_{w}", {}).get("mean_entropy_importance_stability", 0.0)
                            for w in [500, 1000, 2000]
                            if f"window_{w}" in overall_imp
                        ]),
                        "mean_overall_stability": np.mean([
                            overall_imp.get(f"window_{w}", {}).get("mean_overall_stability", 0.0)
                            for w in [500, 1000, 2000]
                            if f"window_{w}" in overall_imp
                        ]),
                        "stable_features_count": sum([
                            overall_imp.get(f"window_{w}", {}).get("stable_features_count", 0)
                            for w in [500, 1000, 2000]
                            if f"window_{w}" in overall_imp
                        ])
                    }
            
            # Overall stability score
            if stability_metrics:
                overall_stability_score = np.mean([
                    stability_metrics.get("time_based", {}).get("mean_stability_score", 0.0),
                    stability_metrics.get("distribution", {}).get("distribution_stability_score", 0.0),
                    stability_metrics.get("importance", {}).get("mean_overall_stability", 0.0)
                ])
                stability_metrics["overall_stability_score"] = float(overall_stability_score)
            else:
                stability_metrics["overall_stability_score"] = 0.0
            
            return stability_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating stability metrics: {str(e)}")
            return {"overall_stability_score": 0.0}

    def _generate_detailed_quality_report(self, quality_metrics: dict[str, Any]) -> str:
        """Generate detailed quality report with recommendations."""
        try:
            report = []
            report.append("=" * 80)
            report.append("📊 DETAILED FEATURE MATRIX QUALITY REPORT")
            report.append("=" * 80)
            
            # Overall Score
            overall_score = quality_metrics.get("overall_score", 0.0)
            report.append(f"🎯 OVERALL QUALITY SCORE: {overall_score:.2f}/1.00")
            
            # Score interpretation
            if overall_score >= 0.9:
                report.append("✅ EXCELLENT - Feature matrix is of very high quality")
            elif overall_score >= 0.8:
                report.append("🟢 GOOD - Feature matrix is of good quality with minor issues")
            elif overall_score >= 0.7:
                report.append("🟡 ACCEPTABLE - Feature matrix has some quality issues")
            elif overall_score >= 0.6:
                report.append("🟠 POOR - Feature matrix has significant quality issues")
            else:
                report.append("🔴 CRITICAL - Feature matrix has severe quality issues")
            
            report.append("")
            
            # 1. Completeness Analysis
            completeness = quality_metrics.get("completeness", {})
            report.append("📋 1. DATA COMPLETENESS ANALYSIS")
            report.append("-" * 40)
            report.append(f"   Total cells: {completeness.get('total_cells', 0):,}")
            report.append(f"   Missing cells: {completeness.get('missing_cells', 0):,}")
            report.append(f"   Missing ratio: {completeness.get('missing_ratio', 0):.2%}")
            report.append(f"   Complete rows: {completeness.get('complete_rows', 0):,}")
            report.append(f"   Complete columns: {completeness.get('complete_columns', 0):,}")
            
            if completeness.get('missing_ratio', 0) > 0.05:
                report.append("   ⚠️  RECOMMENDATION: High missing data ratio - consider imputation")
            else:
                report.append("   ✅ Data completeness is acceptable")
            report.append("")
            
            # 2. Variance Analysis
            variance = quality_metrics.get("variance", {})
            report.append("📊 2. FEATURE VARIANCE ANALYSIS")
            report.append("-" * 40)
            report.append(f"   Mean variance: {variance.get('mean_variance', 0):.6f}")
            report.append(f"   Median variance: {variance.get('median_variance', 0):.6f}")
            report.append(f"   Min variance: {variance.get('min_variance', 0):.6f}")
            report.append(f"   Max variance: {variance.get('max_variance', 0):.6f}")
            report.append(f"   Low variance features: {variance.get('low_variance_features', 0)}")
            report.append(f"   Zero variance features: {variance.get('zero_variance_features', 0)}")
            
            if variance.get('zero_variance_features', 0) > 0:
                report.append("   ⚠️  RECOMMENDATION: Remove zero-variance features")
            else:
                report.append("   ✅ Feature variance is acceptable")
            report.append("")
            
            # 3. Correlation Analysis
            correlation = quality_metrics.get("correlation", {})
            report.append("🔗 3. FEATURE CORRELATION ANALYSIS")
            report.append("-" * 40)
            report.append(f"   Mean correlation: {correlation.get('mean_correlation', 0):.4f}")
            report.append(f"   Max correlation: {correlation.get('max_correlation', 0):.4f}")
            report.append(f"   High correlation pairs: {correlation.get('high_correlation_pairs', 0)}")
            report.append(f"   Correlation threshold: {correlation.get('correlation_threshold', 0.8)}")
            
            if correlation.get('high_correlation_pairs', 0) > 10:
                report.append("   ⚠️  RECOMMENDATION: Many highly correlated features - consider feature selection")
            elif correlation.get('high_correlation_pairs', 0) > 0:
                report.append("   ⚠️  RECOMMENDATION: Some highly correlated features - review for redundancy")
            else:
                report.append("   ✅ Feature correlations are acceptable")
            report.append("")
            
            # 4. Numerical Stability Analysis
            stability = quality_metrics.get("numerical_stability", {})
            report.append("🔢 4. NUMERICAL STABILITY ANALYSIS")
            report.append("-" * 40)
            report.append(f"   Condition number: {stability.get('condition_number', 0):.2e}")
            report.append(f"   Well-conditioned: {stability.get('is_well_conditioned', False)}")
            report.append(f"   Condition threshold: {stability.get('condition_threshold', 1e12):.2e}")
            
            if not stability.get('is_well_conditioned', False):
                report.append("   ⚠️  RECOMMENDATION: Matrix is ill-conditioned - consider regularization or feature scaling")
            else:
                report.append("   ✅ Numerical stability is good")
            report.append("")
            
            # 5. Dimensionality Analysis
            dimensionality = quality_metrics.get("dimensionality", {})
            report.append("📐 5. DIMENSIONALITY ANALYSIS")
            report.append("-" * 40)
            report.append(f"   Matrix rank: {dimensionality.get('matrix_rank', 0)}")
            report.append(f"   Full rank: {dimensionality.get('full_rank', False)}")
            report.append(f"   Rank deficiency: {dimensionality.get('rank_deficiency', 0)}")
            report.append(f"   Effective dimensions: {dimensionality.get('effective_dimensions', 0)}")
            
            if dimensionality.get('rank_deficiency', 0) > 0:
                report.append("   ⚠️  RECOMMENDATION: Rank-deficient matrix - consider dimensionality reduction")
            else:
                report.append("   ✅ Matrix has full rank")
            report.append("")
            
            # 6. Distribution Analysis
            distribution = quality_metrics.get("distribution", {})
            report.append("📈 6. FEATURE DISTRIBUTION ANALYSIS")
            report.append("-" * 40)
            report.append(f"   Mean skewness: {distribution.get('skewness_mean', 0):.4f}")
            report.append(f"   Skewness std: {distribution.get('skewness_std', 0):.4f}")
            report.append(f"   Mean kurtosis: {distribution.get('kurtosis_mean', 0):.4f}")
            report.append(f"   Kurtosis std: {distribution.get('kurtosis_std', 0):.4f}")
            report.append(f"   High skew features: {distribution.get('high_skew_features', 0)}")
            report.append(f"   High kurtosis features: {distribution.get('high_kurtosis_features', 0)}")
            
            if distribution.get('high_skew_features', 0) > 10:
                report.append("   ⚠️  RECOMMENDATION: Many skewed features - consider transformations")
            else:
                report.append("   ✅ Feature distributions are generally acceptable")
            report.append("")
            
            # 7. Outlier Analysis
            outliers = quality_metrics.get("outliers", {})
            report.append("🎯 7. OUTLIER ANALYSIS")
            report.append("-" * 40)
            report.append(f"   Total outliers: {outliers.get('total_outliers', 0):,}")
            report.append(f"   Mean outliers per feature: {outliers.get('mean_outliers_per_feature', 0):.1f}")
            report.append(f"   Max outliers in feature: {outliers.get('max_outliers_in_feature', 0)}")
            report.append(f"   Mean outlier ratio: {outliers.get('mean_outlier_ratio', 0):.2%}")
            report.append(f"   High outlier features: {outliers.get('high_outlier_features', 0)}")
            
            if outliers.get('high_outlier_features', 0) > 5:
                report.append("   ⚠️  RECOMMENDATION: Many features with high outlier ratios - consider outlier handling")
            else:
                report.append("   ✅ Outlier levels are acceptable")
            report.append("")
            
            # 8. Memory Usage Analysis
            memory = quality_metrics.get("memory", {})
            report.append("💾 8. MEMORY USAGE ANALYSIS")
            report.append("-" * 40)
            report.append(f"   Total memory usage: {memory.get('memory_usage_mb', 0):.1f} MB")
            report.append(f"   Memory per feature: {memory.get('memory_per_feature_kb', 0):.1f} KB")
            report.append(f"   Data types: {memory.get('data_types', {})}")
            
            if memory.get('memory_usage_mb', 0) > 1000:
                report.append("   ⚠️  RECOMMENDATION: High memory usage - consider data type optimization")
            else:
                report.append("   ✅ Memory usage is reasonable")
            report.append("")
            
            # 9. Stability Analysis
            stability = quality_metrics.get("stability", {})
            report.append("🔄 9. STABILITY ANALYSIS")
            report.append("-" * 40)
            
            # Time-based stability
            time_stability = stability.get("time_based", {})
            if time_stability:
                report.append(f"   Time-based stability score: {time_stability.get('mean_stability_score', 0):.3f}")
                report.append(f"   Variance stability score: {time_stability.get('variance_stability_score', 0):.3f}")
                report.append(f"   Entropy stability score: {time_stability.get('entropy_stability_score', 0):.3f}")
                report.append(f"   Stable features count: {time_stability.get('stable_features_count', 0)}")
            
            # Distribution stability
            dist_stability = stability.get("distribution", {})
            if dist_stability:
                report.append(f"   Distribution stability score: {dist_stability.get('distribution_stability_score', 0):.3f}")
                report.append(f"   Mean PSI: {dist_stability.get('mean_psi', 0):.3f}")
                report.append(f"   Stable distributions: {dist_stability.get('stable_distributions_count', 0)}")
                report.append(f"   Moderate shifts: {dist_stability.get('moderate_shifts_count', 0)}")
                report.append(f"   Significant shifts: {dist_stability.get('significant_shifts_count', 0)}")
                report.append(f"   Mean entropy stability: {dist_stability.get('mean_entropy_stability', 0):.3f}")
            
            # Feature importance stability
            imp_stability = stability.get("importance", {})
            if imp_stability:
                report.append(f"   Importance stability score: {imp_stability.get('mean_overall_stability', 0):.3f}")
                report.append(f"   Correlation stability: {imp_stability.get('mean_correlation_stability', 0):.3f}")
                report.append(f"   Mutual info stability: {imp_stability.get('mean_mutual_info_stability', 0):.3f}")
                report.append(f"   Variance stability: {imp_stability.get('mean_variance_stability', 0):.3f}")
                report.append(f"   Entropy importance stability: {imp_stability.get('mean_entropy_importance_stability', 0):.3f}")
                report.append(f"   Stable importance features: {imp_stability.get('stable_features_count', 0)}")
            
            # Overall stability score
            overall_stability = stability.get("overall_stability_score", 0.0)
            report.append(f"   Overall stability score: {overall_stability:.3f}")
            
            if overall_stability >= 0.8:
                report.append("   ✅ EXCELLENT - Features are very stable over time")
            elif overall_stability >= 0.6:
                report.append("   🟢 GOOD - Features are generally stable")
            elif overall_stability >= 0.4:
                report.append("   🟡 MODERATE - Some features show instability")
            else:
                report.append("   🔴 POOR - Many features are unstable")
            report.append("")
            
            # 10. SR-Specific Analysis (if available)
            if "sr_analysis" in matrix_results or "sr_enhanced_analysis" in matrix_results or "sr_optimization_analysis" in matrix_results:
                report.append("🎯 10. SR-SPECIFIC ANALYSIS")
                report.append("-" * 40)
                
                # Basic SR analysis
                if "sr_analysis" in matrix_results:
                    sr_analysis = matrix_results["sr_analysis"]
                    if "sr_feature_count" in sr_analysis:
                        report.append(f"   SR Features: {sr_analysis['sr_feature_count']}")
                    if "sr_correlation_analysis" in sr_analysis:
                        high_corrs = sr_analysis["sr_correlation_analysis"].get("high_correlations", [])
                        report.append(f"   SR High Correlations: {len(high_corrs)}")
                
                # Enhanced SR analysis
                if "sr_enhanced_analysis" in matrix_results:
                    enhanced_analysis = matrix_results["sr_enhanced_analysis"]
                    if "enhanced_sr_feature_count" in enhanced_analysis:
                        report.append(f"   Enhanced SR Features: {enhanced_analysis['enhanced_sr_feature_count']}")
                    if "enhanced_sr_importance_analysis" in enhanced_analysis:
                        importance = enhanced_analysis["enhanced_sr_importance_analysis"]
                        if "top_features" in importance:
                            report.append(f"   Top Enhanced SR Features: {len(importance['top_features'])}")
                
                # SR optimization analysis
                if "sr_optimization_analysis" in matrix_results:
                    opt_analysis = matrix_results["sr_optimization_analysis"]
                    if "optimization_feature_count" in opt_analysis:
                        report.append(f"   SR Optimization Features: {opt_analysis['optimization_feature_count']}")
                
                report.append("")
            
            # 10. Actionable Recommendations
            report.append("🚀 10. ACTIONABLE RECOMMENDATIONS")
            report.append("-" * 40)
            
            recommendations = []
            
            if completeness.get('missing_ratio', 0) > 0.05:
                recommendations.append("• Implement data imputation for missing values")
            
            if variance.get('zero_variance_features', 0) > 0:
                recommendations.append("• Remove zero-variance features")
            
            if correlation.get('high_correlation_pairs', 0) > 5:
                recommendations.append("• Apply feature selection to reduce multicollinearity")
            
            if not stability.get('is_well_conditioned', False):
                recommendations.append("• Apply feature scaling or regularization")
            
            if dimensionality.get('rank_deficiency', 0) > 0:
                recommendations.append("• Consider PCA or other dimensionality reduction techniques")
            
            if distribution.get('high_skew_features', 0) > 10:
                recommendations.append("• Apply log or power transformations to skewed features")
            
            if outliers.get('high_outlier_features', 0) > 5:
                recommendations.append("• Implement outlier detection and handling strategies")
            
            if memory.get('memory_usage_mb', 0) > 1000:
                recommendations.append("• Optimize data types to reduce memory usage")
            
            # SR-specific recommendations
            if "sr_analysis" in matrix_results or "sr_enhanced_analysis" in matrix_results:
                recommendations.append("• Review SR feature correlations and consider feature selection")
                recommendations.append("• Validate SR feature stability across different market conditions")
                recommendations.append("• Consider SR feature importance for model training prioritization")
            

            
            if not recommendations:
                recommendations.append("• No immediate actions required - feature matrix is in good condition")
            
            for rec in recommendations:
                report.append(f"   {rec}")
            
            report.append("")
            
            # 11. Summary
            report.append("📋 11. SUMMARY")
            report.append("-" * 40)
            report.append(f"   Overall Quality Score: {overall_score:.2f}/1.00")
            
            # Stability summary
            stability = quality_metrics.get("stability", {})
            if stability:
                overall_stability = stability.get("overall_stability_score", 0.0)
                report.append(f"   Overall Stability Score: {overall_stability:.3f}/1.00")
                
                if overall_stability >= 0.8:
                    report.append("   Stability Status: ✅ EXCELLENT - Features are very stable")
                elif overall_stability >= 0.6:
                    report.append("   Stability Status: 🟢 GOOD - Features are generally stable")
                elif overall_stability >= 0.4:
                    report.append("   Stability Status: 🟡 MODERATE - Some features show instability")
                else:
                    report.append("   Stability Status: 🔴 POOR - Many features are unstable")
            else:
                report.append("   Stability Status: ⚠️  NO STABILITY DATA AVAILABLE")
            
            # SR-specific summary
            if "sr_analysis" in matrix_results or "sr_enhanced_analysis" in matrix_results or "sr_optimization_analysis" in matrix_results:
                report.append("   SR Analysis: ✅ COMPREHENSIVE SR FEATURES ANALYZED")
                
                total_sr_features = 0
                if "sr_analysis" in matrix_results:
                    total_sr_features += matrix_results["sr_analysis"].get("sr_feature_count", 0)
                if "sr_enhanced_analysis" in matrix_results:
                    total_sr_features += matrix_results["sr_enhanced_analysis"].get("enhanced_sr_feature_count", 0)
                if "sr_optimization_analysis" in matrix_results:
                    total_sr_features += matrix_results["sr_optimization_analysis"].get("optimization_feature_count", 0)
                
                report.append(f"   Total SR Features: {total_sr_features}")
                
                # SR optimization status
                if "sr_optimization_analysis" in matrix_results:
                    opt_analysis = matrix_results["sr_optimization_analysis"]
                    if "sr_optimization_performance_analysis" in opt_analysis:
                        perf_score = opt_analysis["sr_optimization_performance_analysis"].get("overall_performance_score", 0)
                        if perf_score >= 0.7:
                            report.append("   SR Optimization: ✅ HIGH PERFORMANCE")
                        elif perf_score >= 0.5:
                            report.append("   SR Optimization: ⚠️  MODERATE PERFORMANCE")
                        else:
                            report.append("   SR Optimization: 🔴 LOW PERFORMANCE")
            else:
                report.append("   SR Analysis: ⚠️  NO SR FEATURES DETECTED")
            
            if overall_score >= 0.8:
                report.append("   Status: ✅ READY FOR MODEL TRAINING")
            elif overall_score >= 0.6:
                report.append("   Status: ⚠️  NEEDS IMPROVEMENT BEFORE TRAINING")
            else:
                report.append("   Status: 🔴 REQUIRES SIGNIFICANT IMPROVEMENT")
            
            report.append("=" * 80)
            
            return "\n".join(report)
            
        except Exception as e:
            self.logger.error(f"Error generating detailed quality report: {str(e)}")
            return f"Error generating report: {str(e)}"

    def _find_high_correlations(
        self, 
        correlation_matrix: pd.DataFrame, 
        threshold: float
    ) -> list[dict[str, Any]]:
        """Find high correlation pairs."""
        high_correlations = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    high_correlations.append({
                        "column1": correlation_matrix.columns[i],
                        "column2": correlation_matrix.columns[j],
                        "correlation": float(corr_value)
                    })
        
        return high_correlations

    async def _save_matrix_operations_results(
        self,
        results: dict[str, Any],
        config: dict[str, Any],
        quality_metrics: dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> dict[str, str]:
        """Save matrix operations results to files."""
        
        output_files = {}
        
        # Save configuration
        config_file = self.output_dir / f"{exchange}_{symbol}_{timeframe}_matrix_operations_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        output_files["config"] = str(config_file)
        
        # Save results
        results_file = self.output_dir / f"{exchange}_{symbol}_{timeframe}_matrix_operations_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        output_files["results"] = str(results_file)
        
        # Save quality metrics
        quality_file = self.output_dir / f"{exchange}_{symbol}_{timeframe}_quality_metrics.json"
        with open(quality_file, 'w') as f:
            json.dump(quality_metrics, f, indent=2, default=str)
        output_files["quality_metrics"] = str(quality_file)
        
        # Generate and save detailed quality report
        detailed_report = self._generate_detailed_quality_report(quality_metrics)
        report_file = self.output_dir / f"{exchange}_{symbol}_{timeframe}_quality_report.txt"
        with open(report_file, 'w') as f:
            f.write(detailed_report)
        output_files["quality_report"] = str(report_file)
        
        # Log the detailed report
        self.logger.info("\n" + detailed_report)
        
        # Save summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            "operations_performed": list(results.keys()),
            "data_shape": config["data_shape"],
            "numeric_columns": len(config["numeric_columns"]),
            "overall_quality_score": quality_metrics.get("overall_score", 0.0),
            "quality_summary": {
                "completeness_ratio": quality_metrics.get("completeness", {}).get("missing_ratio", 1.0),
                "zero_variance_features": quality_metrics.get("variance", {}).get("zero_variance_features", 0),
                "high_correlations": quality_metrics.get("correlation", {}).get("high_correlation_pairs", 0),
                "is_well_conditioned": quality_metrics.get("numerical_stability", {}).get("is_well_conditioned", False)
            }
        }
        
        summary_file = self.output_dir / f"{exchange}_{symbol}_{timeframe}_matrix_operations_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        output_files["summary"] = str(summary_file)
        
        self.logger.info(f"💾 Saved matrix operations results to {self.output_dir}")
        return output_files

    # ============================================================================
    # ENTROPY AND STABILITY CALCULATION METHODS
    # ============================================================================

    def _calculate_entropy_stability(self, values: pd.Series) -> float:
        """Calculate entropy-based stability measure."""
        try:
            if len(values) < 2:
                return 0.0
            
            # Calculate Shannon entropy
            hist, _ = np.histogram(values, bins=min(20, len(values)//10), density=True)
            hist = hist[hist > 0]  # Remove zero bins
            entropy = -np.sum(hist * np.log2(hist))
            
            # Normalize entropy (0 = no uncertainty, 1 = maximum uncertainty)
            max_entropy = np.log2(len(hist))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            # Stability is inverse of normalized entropy (lower entropy = more stable)
            stability = 1.0 - normalized_entropy
            
            return max(0.0, min(1.0, stability))
            
        except Exception:
            return 0.0

    def _calculate_rolling_entropy_stability(self, values: pd.Series, window_size: int) -> float:
        """Calculate rolling entropy stability over time."""
        try:
            if len(values) < window_size:
                return 0.0
            
            # Calculate rolling entropy
            rolling_entropy = []
            for i in range(window_size, len(values)):
                window_values = values.iloc[i-window_size:i]
                hist, _ = np.histogram(window_values, bins=min(10, window_size//5), density=True)
                hist = hist[hist > 0]
                if len(hist) > 1:
                    entropy = -np.sum(hist * np.log2(hist))
                    max_entropy = np.log2(len(hist))
                    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                    rolling_entropy.append(normalized_entropy)
            
            if not rolling_entropy:
                return 0.0
            
            # Stability is inverse of entropy variance (lower variance = more stable)
            entropy_std = np.std(rolling_entropy)
            stability = 1.0 / (1.0 + entropy_std)
            
            return max(0.0, min(1.0, stability))
            
        except Exception:
            return 0.0

    def _calculate_entropy_distribution_stability(self, reference: pd.Series, current: pd.Series) -> float:
        """Calculate entropy-based distribution stability between reference and current data."""
        try:
            if len(reference) < 2 or len(current) < 2:
                return 0.0
            
            # Calculate entropy for both distributions
            ref_hist, _ = np.histogram(reference, bins=min(20, len(reference)//10), density=True)
            curr_hist, _ = np.histogram(current, bins=min(20, len(current)//10), density=True)
            
            ref_hist = ref_hist[ref_hist > 0]
            curr_hist = curr_hist[curr_hist > 0]
            
            if len(ref_hist) < 2 or len(curr_hist) < 2:
                return 0.0
            
            ref_entropy = -np.sum(ref_hist * np.log2(ref_hist))
            curr_entropy = -np.sum(curr_hist * np.log2(curr_hist))
            
            # Calculate entropy difference
            entropy_diff = abs(curr_entropy - ref_entropy)
            max_entropy = max(ref_entropy, curr_entropy)
            
            # Stability is inverse of relative entropy difference
            if max_entropy > 0:
                relative_diff = entropy_diff / max_entropy
                stability = 1.0 - relative_diff
            else:
                stability = 1.0
            
            return max(0.0, min(1.0, stability))
            
        except Exception:
            return 0.0

    def _calculate_entropy_importance_stability(self, feature: pd.Series, target: pd.Series, window_size: int) -> float:
        """Calculate entropy-based importance stability."""
        try:
            if len(feature) < window_size or len(target) < window_size:
                return 0.0
            
            # Calculate rolling mutual information
            rolling_mi = []
            for i in range(window_size, len(feature)):
                f_window = feature.iloc[i-window_size:i]
                t_window = target.iloc[i-window_size:i]
                
                # Calculate mutual information for this window
                mi = self._calculate_mutual_information(f_window, t_window)
                rolling_mi.append(mi)
            
            if not rolling_mi:
                return 0.0
            
            # Stability is inverse of mutual information variance
            mi_std = np.std(rolling_mi)
            stability = 1.0 / (1.0 + mi_std)
            
            return max(0.0, min(1.0, stability))
            
        except Exception:
            return 0.0

    def _calculate_mutual_information(self, x: pd.Series, y: pd.Series) -> float:
        """Calculate mutual information between two series."""
        try:
            if len(x) < 2 or len(y) < 2:
                return 0.0
            
            # Create 2D histogram
            hist_2d, _, _ = np.histogram2d(x, y, bins=min(10, len(x)//10))
            hist_2d = hist_2d.flatten()
            hist_2d = hist_2d[hist_2d > 0]
            
            if len(hist_2d) < 2:
                return 0.0
            
            # Normalize to probabilities
            p_xy = hist_2d / hist_2d.sum()
            
            # Calculate mutual information
            mi = -np.sum(p_xy * np.log2(p_xy))
            
            return max(0.0, mi)
            
        except Exception:
            return 0.0

    def _calculate_psi(self, reference: pd.Series, current: pd.Series) -> float:
        """Calculate Population Stability Index."""
        try:
            if len(reference) < 2 or len(current) < 2:
                return 0.0
            
            # Create bins for both distributions
            combined = pd.concat([reference, current])
            bins = pd.cut(combined, bins=10, duplicates='drop')
            
            # Calculate bin counts
            ref_counts = reference.groupby(pd.cut(reference, bins=bins.cat.categories)).count()
            curr_counts = current.groupby(pd.cut(current, bins=bins.cat.categories)).count()
            
            # Normalize to probabilities
            ref_probs = ref_counts / ref_counts.sum()
            curr_probs = curr_counts / curr_counts.sum()
            
            # Calculate PSI
            psi = 0
            for bin_name in ref_probs.index:
                if bin_name in curr_probs.index:
                    ref_p = ref_probs[bin_name]
                    curr_p = curr_probs[bin_name]
                    
                    if ref_p > 0 and curr_p > 0:
                        psi += (curr_p - ref_p) * np.log(curr_p / ref_p)
            
            return max(0.0, psi)
            
        except Exception:
            return 0.0

    def _calculate_ks_test(self, reference: pd.Series, current: pd.Series) -> tuple[float, float]:
        """Calculate Kolmogorov-Smirnov test statistic and p-value."""
        try:
            from scipy import stats
            
            ref_clean = reference.dropna()
            curr_clean = current.dropna()
            
            if len(ref_clean) > 0 and len(curr_clean) > 0:
                ks_stat, p_value = stats.ks_2samp(ref_clean, curr_clean)
                return float(ks_stat), float(p_value)
            else:
                return 0.0, 1.0
                
        except Exception:
            return 0.0, 1.0

    def _calculate_moment_stability(self, reference: pd.Series, current: pd.Series) -> dict[str, float]:
        """Calculate stability of distribution moments."""
        try:
            ref_mean = reference.mean()
            ref_std = reference.std()
            ref_skew = reference.skew()
            ref_kurt = reference.kurtosis()
            
            curr_mean = current.mean()
            curr_std = current.std()
            curr_skew = current.skew()
            curr_kurt = current.kurtosis()
            
            # Calculate relative differences
            mean_stability = 1.0 / (1.0 + abs(curr_mean - ref_mean) / (abs(ref_mean) + 1e-8))
            std_stability = 1.0 / (1.0 + abs(curr_std - ref_std) / (ref_std + 1e-8))
            skew_stability = 1.0 / (1.0 + abs(curr_skew - ref_skew) / (abs(ref_skew) + 1e-8))
            kurt_stability = 1.0 / (1.0 + abs(curr_kurt - ref_kurt) / (abs(ref_kurt) + 1e-8))
            
            return {
                "mean_stability": float(max(0.0, min(1.0, mean_stability))),
                "std_stability": float(max(0.0, min(1.0, std_stability))),
                "skew_stability": float(max(0.0, min(1.0, skew_stability))),
                "kurt_stability": float(max(0.0, min(1.0, kurt_stability)))
            }
            
        except Exception:
            return {
                "mean_stability": 0.0,
                "std_stability": 0.0,
                "skew_stability": 0.0,
                "kurt_stability": 0.0
            }

    def _calculate_rolling_correlation(self, x: pd.Series, y: pd.Series, window_size: int) -> pd.Series:
        """Calculate rolling correlation between two series."""
        try:
            if len(x) < window_size or len(y) < window_size:
                return pd.Series(dtype=float)
            
            # Align series
            aligned_data = pd.DataFrame({'x': x, 'y': y}).dropna()
            
            if len(aligned_data) < window_size:
                return pd.Series(dtype=float)
            
            # Calculate rolling correlation
            rolling_corr = aligned_data['x'].rolling(window=window_size, min_periods=window_size//2).corr(aligned_data['y'])
            
            return rolling_corr
            
        except Exception:
            return pd.Series(dtype=float)

    def _calculate_rolling_mutual_information(self, x: pd.Series, y: pd.Series, window_size: int) -> pd.Series:
        """Calculate rolling mutual information between two series."""
        try:
            if len(x) < window_size or len(y) < window_size:
                return pd.Series(dtype=float)
            
            # Align series
            aligned_data = pd.DataFrame({'x': x, 'y': y}).dropna()
            
            if len(aligned_data) < window_size:
                return pd.Series(dtype=float)
            
            # Calculate rolling mutual information
            rolling_mi = []
            for i in range(window_size, len(aligned_data)):
                x_window = aligned_data['x'].iloc[i-window_size:i]
                y_window = aligned_data['y'].iloc[i-window_size:i]
                
                mi = self._calculate_mutual_information(x_window, y_window)
                rolling_mi.append(mi)
            
            # Create series with proper index
            result = pd.Series(rolling_mi, index=aligned_data.index[window_size:])
            return result
            
        except Exception:
            return pd.Series(dtype=float)


# Step execution function
async def run_step(
    symbol: str,
    exchange: str,
    timeframe: str = "1m",
    data_dir: str = None,
    force_rerun: bool = False,
    **kwargs: Any,
) -> bool:
    """
    Run Step 7: Enhanced Matrix Operations with standardized data quality management.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        data_dir: Data directory (will use standardized path if None)
        force_rerun: Force rerun the step
        **kwargs: Additional arguments
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Use standardized path construction
        if data_dir is None:
            data_dir = pipeline_standards.build_path("processed_data", exchange, symbol)
        
        # Load configuration
        from src.config.training import get_training_config
        config = get_training_config()
        
        # Create step instance
        step = Step7EnhancedMatrixOperations(config)
        
        # Prepare training input
        training_input = {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            "data_dir": data_dir,
            "force_rerun": force_rerun,
            "asset": symbol,  # Use symbol as asset
            "lookback_period": config.get("lookback_days", 1095),  # Default to 3 years
            "project_version": config.get("project_version", "1.0.0"),  # Default version
            **kwargs
        }
        
        # Execute step
        pipeline_state = {}
        result = await step.execute(training_input, pipeline_state)
        
        # Check if step was successful
        step_result = result.get("step7_enhanced_matrix_operations", {})
        return step_result.get("status") == "completed"
        
    except Exception as e:
        system_logger.error(f"❌ Step 7 failed: {str(e)}")
        return False


# Export the main class for external use
__all__ = ["Step7EnhancedMatrixOperations", "run_step"]