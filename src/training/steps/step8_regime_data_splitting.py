# src/training/steps/step8_regime_data_splitting.py

import asyncio
import json
import os
from datetime import datetime
from typing import Any
from pathlib import Path

# Add project root to path
import pandas as pd

project_root = Path(__file__).parent.parent.parent
import sys
sys.path.insert(0, str(project_root))

# Import pipeline standards
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards

# Standardized import management
REQUIRED_MODULES = [
    "pandas",
    "src.utils.centralized_decorators",
    "src.training.steps.unified_data_loader",
    "src.utils.logger",
    "src.utils.enhanced_mlflow_integration"
]

# Validate environment dependencies
dependency_status = PipelineStandards.validate_environment_dependencies(REQUIRED_MODULES)

# Safe imports with fallbacks
centralized_decorators = PipelineStandards.safe_import("src.utils.centralized_decorators", None)
unified_data_loader = PipelineStandards.safe_import("src.training.steps.unified_data_loader", None)
system_logger = PipelineStandards.safe_import("src.utils.logger", None)
enhanced_mlflow = PipelineStandards.safe_import("src.utils.enhanced_mlflow_integration", None)
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

if centralized_decorators is None:
    auto_fix_data_quality_issues = create_fallback_decorator()
    artifact_versioning = create_fallback_decorator()
    artifact_write_lock = create_fallback_decorator()
    circuit_breaker_protection = create_fallback_decorator()
    debug_training_step = create_fallback_decorator()
    deterministic_seed = create_fallback_decorator()
    handle_errors = create_fallback_decorator()
    idempotent_step = create_fallback_decorator()
    memory_efficient = create_fallback_decorator()
    nan_inf_and_constant_guard = create_fallback_decorator()
    prevent_data_leakage = create_fallback_decorator()
    quality_gate = create_fallback_decorator()
    resource_monitor = create_fallback_decorator()
    secure_data_processing = create_fallback_decorator()
    time_budget_watchdog = create_fallback_decorator()
    validate_step_output = create_fallback_decorator()
    validate_step_prerequisites = create_fallback_decorator()
    with_tracing_span = create_fallback_decorator()
else:
    auto_fix_data_quality_issues = centralized_decorators.auto_fix_data_quality_issues
    artifact_versioning = centralized_decorators.artifact_versioning
    artifact_write_lock = centralized_decorators.artifact_write_lock
    circuit_breaker_protection = centralized_decorators.circuit_breaker_protection
    debug_training_step = centralized_decorators.debug_training_step
    deterministic_seed = centralized_decorators.deterministic_seed
    handle_errors = centralized_decorators.handle_errors
    idempotent_step = centralized_decorators.idempotent_step
    memory_efficient = centralized_decorators.memory_efficient
    nan_inf_and_constant_guard = centralized_decorators.nan_inf_and_constant_guard
    prevent_data_leakage = centralized_decorators.prevent_data_leakage
    quality_gate = centralized_decorators.quality_gate
    resource_monitor = centralized_decorators.resource_monitor
    secure_data_processing = centralized_decorators.secure_data_processing
    time_budget_watchdog = centralized_decorators.time_budget_watchdog
    validate_step_output = centralized_decorators.validate_step_output
    validate_step_prerequisites = centralized_decorators.validate_step_prerequisites
    with_tracing_span = centralized_decorators.with_tracing_span

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

class RegimeDataSplittingStep:
    """Step 8: Unified Regime Data Creation with standardized data quality management."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("Step8.RegimeSplit")
        self.standards = pipeline_standards
        
        # Validate environment on initialization
        self._validate_environment()

    def _validate_environment(self) -> None:
        """Validate environment dependencies."""
        self.logger.info("🔍 Validating environment dependencies...")
        
        missing_modules = [module for module, available in dependency_status.items() if not available]
        if missing_modules:
            self.logger.warning(f"⚠️ Missing optional modules: {missing_modules}")
            self.logger.info("📝 Pipeline will continue with fallback implementations")
        else:
            self.logger.info("✅ All required dependencies available")

    @with_tracing_span("step8_regime_splitting.initialize", log_args=False)
    @handles_errors(fallback=None)
    async def initialize(self) -> None:
        """Initialize the regime data splitting step."""
        self.logger.info("🚀 Initializing Step 8: Unified HMM Composite Regime Data Creation...")
        self.logger.info("📋 Step 8 Configuration:")
        self.logger.info(f"   - Unified dataset approach: Enabled")
        self.logger.info(f"   - Regime labels: composite_cluster_id")
        self.logger.info(f"   - Maintains temporal continuity: Yes")
        self.logger.info("✅ Unified HMM Composite Regime Data Creation initialized successfully")

    @with_enhanced_mlflow_logging("step08")
    @with_tracing_span("step8_regime_splitting.execute", log_args=False)
    @handles_errors, default_return={"success": False, "error": "Execution failed"}, context="step8_execution")
    async def execute(self) -> dict[str, Any]:
        """Execute the unified regime data creation step."""
        try:
            self.logger.info("🔄 Loading unified data for HMM composite regime data creation...")
            data_loader = get_unified_data_loader(self.config)
            from src.config.constants import (
import numpy as np
import os.path
from src.core.decorators import handles_errors

                BLANK_TRAINING_LOOKBACK_DAYS,
            )

            # Use lookback_days from config (should be passed from enhanced training manager)
            config_lookback = self.config.get(
                "lookback_days", BLANK_TRAINING_LOOKBACK_DAYS,
            )
            unified_data = await data_loader.load_unified_data(
                symbol=self.config.get("symbol", "ETHUSDT"),
                exchange=self.config.get("exchange", "BINANCE"),
                timeframe=self.config.get("timeframe", "1m"),
                lookback_days=config_lookback,
            )

            self.logger.info(f"✅ Loaded unified data: {len(unified_data)} rows")
            self.logger.info(
                f"   Date range: {unified_data.index.min()} to {unified_data.index.max()}",
            )

            # HMM COMPOSITE CLUSTERS ONLY - NO FALLBACKS
            self.logger.info("🎯 Using HMM composite clusters for regime labeling (PARAMOUNT)")

            # Check for HMM composite cluster data
            if "composite_cluster_id" not in unified_data.columns:
                self.logger.error("🚨 HMM composite_cluster_id column is missing from unified data")
                self.logger.error("   This is a critical failure - HMM composite clusters are paramount")
                self.logger.error("   Please ensure step3_hmm_regime_discovery completed successfully")
                return {"success": False, "error": "Missing HMM composite_cluster_id - paramount requirement"}

            # Verify HMM composite clusters are not all null
            composite_clusters = unified_data["composite_cluster_id"].dropna()
            if composite_clusters.empty:
                self.logger.error("🚨 HMM composite_cluster_id column contains only null values")
                self.logger.error("   This indicates step3_hmm_regime_discovery failed to generate valid clusters")
                return {"success": False, "error": "HMM composite_cluster_id contains only null values"}

            # Get unique HMM composite clusters
            unique_clusters = composite_clusters.unique()
            self.logger.info(f"📊 Found {len(unique_clusters)} unique HMM composite clusters: {sorted(unique_clusters)}")

            # Ensure data is sorted by timestamp for proper lookback periods
            unified_data = unified_data.sort_index()
            
            # Create unified dataset with regime labels (no splitting into separate files)
            self.logger.info("🔀 Creating unified dataset with regime labels...")
            
            # Save unified dataset with regime labels
            success = self._save_unified_regime_dataset(unified_data, unique_clusters)
            
            if not success:
                self.logger.error("🚨 Failed to save unified regime dataset")
                return {"success": False, "error": "Failed to save unified regime dataset"}

            self.logger.info(f"✅ Successfully created unified dataset with {len(unique_clusters)} HMM composite regime labels")

            # Create regime summary
            summary = self._create_regime_summary(unified_data, unique_clusters)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f"log/step8_regime_unified_{ts}.json", "w") as f:
                json.dump(summary, f, indent=2)

            self.logger.info("✅ Unified HMM composite regime data creation completed successfully")
            
            # Log artifacts and create detailed report
            await self._log_step8_artifacts_and_report(unified_data, summary)
            
            return {"success": True, "regime_summary": summary}
        except Exception as e:
            self.logger.exception(f"❌ Unified HMM composite regime data creation failed: {e}")
            return {"success": False, "error": str(e)}

    async def _log_step8_artifacts_and_report(
        self,
        unified_data: pd.DataFrame,
        summary: dict[str, Any]
    ) -> None:
        """Log step 8 artifacts and create detailed report."""
        try:
            symbol = self.config.get("symbol", "ETHUSDT")
            exchange = self.config.get("exchange", "BINANCE")
            timeframe = self.config.get("timeframe", "1m")
            
            # Collect execution metadata
            execution_metadata = {
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": 0.0,  # Will be calculated if available
                "memory_usage_mb": 0.0,  # Will be calculated if available
                "cpu_usage_percent": 0.0,  # Will be calculated if available
                "data_quality_score": 1.0,
                "processing_efficiency": 1.0,
            }
            
            # Collect artifacts generated
            artifacts_generated = [
                f"{exchange}_{symbol}_{timeframe}_unified_regime_data.parquet",
                f"{exchange}_{symbol}_{timeframe}_regime_labels.json",
                f"{exchange}_{symbol}_{timeframe}_regime_statistics.json"
            ]
            
            # Collect metrics
            metrics_calculated = {
                "regime_creation_success": 1.0,
                "total_regimes": summary.get("total_regimes", 0),
                "total_samples": len(unified_data),
                "regime_ids": summary.get("regime_ids", []),
            }
            
            # Create training input for report
            training_input = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "lookback_days": self.config.get("lookback_days", 1095),
                "asset": symbol,  # Use symbol as asset
                "lookback_period": self.config.get("lookback_days", 1095),
                "project_version": self.config.get("project_version", "1.0.0"),
            }
            
            # Create step data for report
            step_data = {
                "regime_summary": summary,
                "regime_count": summary.get("total_regimes", 0),
                "regime_ids": summary.get("regime_ids", []),
                "approach": "unified_dataset_with_labels",
            }
            
            # Create detailed report
            report_data = create_detailed_step_report(
                step_name="step8_regime_data_splitting",
                step_data=step_data,
                training_input=training_input,
                execution_metadata=execution_metadata,
                artifacts_generated=artifacts_generated,
                metrics_calculated=metrics_calculated,
                errors_encountered=[]
            )
            
            # Log the report
            report_name = log_step_report(
                config=self.config,
                step_name="step8_regime_data_splitting",
                report_data=report_data,
                report_type="unified_regime_data_creation_report",
                additional_metadata={
                    "regime_creation_success": True,
                    "total_regimes": summary.get("total_regimes", 0),
                    "timeframe": timeframe,
                    "asset": symbol,
                    "lookback_period": self.config.get("lookback_days", 1095),
                    "project_version": self.config.get("project_version", "1.0.0"),
                    "approach": "unified_dataset_with_labels",
                }
            )
            self.logger.info(f"✅ Logged unified regime data creation report: {report_name}")
            
            # Log regime summary
            if summary:
                summary_report_name = log_step_report(
                    config=self.config,
                    step_name="step8_regime_data_splitting",
                    report_data=summary,
                    report_type="unified_regime_summary",
                    additional_metadata={
                        "total_regimes": summary.get("total_regimes", 0),
                        "timeframe": timeframe,
                        "asset": symbol,
                        "lookback_period": self.config.get("lookback_days", 1095),
                        "project_version": self.config.get("project_version", "1.0.0"),
                        "approach": "unified_dataset_with_labels",
                    }
                )
                self.logger.info(f"✅ Logged unified regime summary: {summary_report_name}")
            
            # Log metrics
            log_step_metrics(
                config=self.config,
                step_name="step8_regime_data_splitting",
                metrics=metrics_calculated,
                additional_metadata={
                    "metrics_type": "unified_regime_creation_performance",
                    "timeframe": timeframe,
                    "asset": symbol,
                    "lookback_period": self.config.get("lookback_days", 1095),
                    "project_version": self.config.get("project_version", "1.0.0"),
                    "approach": "unified_dataset_with_labels",
                }
            )
            
            self.logger.info("✅ Step 8 artifacts and reports logged successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log step 8 artifacts and reports: {e}")
            # Don't fail the step if MLflow logging fails

    @with_tracing_span("step8_regime_splitting._save_unified_regime_dataset", log_args=False)
    @handles_errors(fallback=False)
    def _save_unified_regime_dataset(self, unified_data: pd.DataFrame, unique_clusters: list) -> bool:
        """Save unified dataset with regime labels."""
        try:
            data_dir = self.config.get("data_dir", "data/training")
            os.makedirs(data_dir, exist_ok=True)
            
            symbol = self.config.get("symbol", "ETHUSDT")
            exchange = self.config.get("exchange", "BINANCE")
            timeframe = self.config.get("timeframe", "1m")
            
            # Save unified dataset with regime labels
            unified_file = os.path.join(data_dir, f"{exchange}_{symbol}_{timeframe}_unified_regime_data.parquet")
            unified_data.to_parquet(unified_file, index=True)
            self.logger.info(f"✅ Saved unified regime dataset: {len(unified_data)} rows -> {unified_file}")
            
            # Create regime labels mapping
            regime_labels = {
                "regime_column": "composite_cluster_id",
                "regime_ids": sorted(unique_clusters),
                "total_regimes": len(unique_clusters),
                "data_shape": unified_data.shape,
                "timestamp_range": {
                    "start": unified_data.index.min().isoformat(),
                    "end": unified_data.index.max().isoformat()
                },
                "usage_instructions": {
                    "description": "Load the unified dataset and filter by composite_cluster_id for regime-specific processing",
                    "example": "regime_data = data[data['composite_cluster_id'] == regime_id]",
                    "benefits": [
                        "Maintains temporal continuity for trading indicators",
                        "Preserves lookback periods",
                        "Eliminates need for multiple file management",
                        "Enables regime-aware processing with single dataset"
                    ]
                }
            }
            
            labels_file = os.path.join(data_dir, f"{exchange}_{symbol}_{timeframe}_regime_labels.json")
            with open(labels_file, 'w') as f:
                json.dump(regime_labels, f, indent=2)
            self.logger.info(f"✅ Saved regime labels mapping: {labels_file}")
            
            # Create regime statistics
            regime_stats = self._create_regime_statistics(unified_data, unique_clusters)
            stats_file = os.path.join(data_dir, f"{exchange}_{symbol}_{timeframe}_regime_statistics.json")
            with open(stats_file, 'w') as f:
                json.dump(regime_stats, f, indent=2)
            self.logger.info(f"✅ Saved regime statistics: {stats_file}")
            
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Failed to save unified regime dataset: {e}")
            return False

    def _create_regime_statistics(self, unified_data: pd.DataFrame, unique_clusters: list) -> dict[str, Any]:
        """Create statistics for the unified regime dataset."""
        try:
            stats = {
                "approach": "unified_dataset_with_labels",
                "total_regimes": len(unique_clusters),
                "total_data_points": len(unified_data),
                "regime_details": {},
                "overall_statistics": {
                    "date_range": {
                        "start": unified_data.index.min().isoformat(),
                        "end": unified_data.index.max().isoformat()
                    }
                }
            }
            
            # Calculate statistics for each regime
            for cluster_id in unique_clusters:
                regime_data = unified_data[unified_data["composite_cluster_id"] == cluster_id]
                
                if len(regime_data) > 0:
                    regime_stats = {
                        "data_points": len(regime_data),
                        "percentage": len(regime_data) / len(unified_data) * 100,
                        "date_range": {
                            "start": regime_data.index.min().isoformat(),
                            "end": regime_data.index.max().isoformat()
                        }
                    }
                    
                    # Add price statistics if available
                    if 'close' in regime_data.columns:
                        regime_stats["price_stats"] = {
                            "mean": float(regime_data['close'].mean()),
                            "std": float(regime_data['close'].std()),
                            "min": float(regime_data['close'].min()),
                            "max": float(regime_data['close'].max())
                        }
                    
                    stats["regime_details"][f"regime_{cluster_id}"] = regime_stats
            
            return stats
            
        except Exception as e:
            self.logger.exception(f"❌ Error creating regime statistics: {e}")
            return {}

    @with_tracing_span("step8_regime_splitting._create_regime_summary", log_args=False)
    @handles_errors(fallback={})
    def _create_regime_summary(self, unified_data: pd.DataFrame, unique_clusters: list) -> dict[str, Any]:
        """Create a summary of the unified regime dataset."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "approach": "unified_dataset_with_labels",
            "regime_basis": "hmm_composite_clusters_only",
            "total_regimes": len(unique_clusters),
            "regime_ids": sorted(unique_clusters),
            "total_rows": len(unified_data),
            "data_shape": unified_data.shape,
            "timestamp_range": {
                "start": unified_data.index.min().isoformat(),
                "end": unified_data.index.max().isoformat()
            },
            "regime_column": "composite_cluster_id",
            "usage_instructions": {
                "description": "Load the unified dataset and filter by composite_cluster_id for regime-specific processing",
                "example": "regime_data = data[data['composite_cluster_id'] == regime_id]",
                "benefits": [
                    "Maintains temporal continuity for trading indicators",
                    "Preserves lookback periods",
                    "Eliminates need for multiple file management",
                    "Enables regime-aware processing with single dataset"
                ]
            }
        }

        return summary

@deterministic_seed(42)
@idempotent_step(step_key="step8_regime_data_splitting")
@artifact_write_lock()
@nan_inf_and_constant_guard()
@artifact_versioning("1.0")
@time_budget_watchdog(soft_timeout_seconds=1800.0)
@validate_step_prerequisites(
    required_directories=["data/training"],
    min_memory_gb=4.0,
    min_disk_gb=3.0,
    required_packages=["pandas", "numpy"],
    data_quality_checks={
        "min_rows": 1000,
        "required_columns": ["timestamp", "composite_cluster_id"],
    },
    context="Unified Regime Data Creation",
)
@secure_data_processing(
    backup_before=True, 
    integrity_checks=True, 
    memory_cleanup=True, 
    data_validation=True,
)
@prevent_data_leakage(
    temporal_validation=True,
    feature_leakage_detection=False,
    lookahead_bias_prevention=True,
)
@resource_monitor(
    memory_threshold_gb=8.0,
    cpu_threshold_percent=70.0,
    disk_threshold_gb=5.0,
    monitor_interval=30.0,
    auto_cleanup=True,
)
@memory_efficient(
    chunk_size=20000, 
    streaming_processing=True, 
    memory_pool=True, 
    cleanup_frequency=40,
)
@debug_training_step(
    log_intermediate_results=True,
    save_debug_artifacts=True,
    performance_profiling=True,
    error_context_preservation=True,
)
@circuit_breaker_protection(
    failure_threshold=3,
    recovery_timeout=90.0,
    expected_exception=Exception,
    monitor_interval=30.0,
)
@validate_step_output(
    required_files=["data/training/*_unified_regime_data.parquet"],
    data_quality_checks={"min_rows": 100, "required_columns": ["timestamp", "composite_cluster_id"]},
    performance_thresholds={"creation_time_minutes": 30.0},
    format_validation=True,
)
@quality_gate(
    data_quality_metrics={"completeness": 0.9, "consistency": 0.8},
    validation_score_requirements={"creation_accuracy": 0.8},
)
@auto_fix_data_quality_issues
@handles_errors(fallback=False)
async def run_step(
    symbol: str, 
    exchange: str, 
    data_dir: str = None, 
    timeframe: str = "1m", 
    force_rerun: bool = False,
    **kwargs,
) -> bool:
    """Run the unified HMM composite regime data creation step with standardized data quality management."""
    
    # Use standardized path construction
    if data_dir is None:
        data_dir = pipeline_standards.build_path("processed_data", exchange, symbol)
    
    config = {
        "symbol": symbol,
        "exchange": exchange,
        "data_dir": data_dir,
        "timeframe": timeframe,
        "force_rerun": force_rerun,
        **kwargs,
    }

    step = RegimeDataSplittingStep(config)
    await step.initialize()
    result = await step.execute()
    return result.get("success", False)

if __name__ == "__main__":
    async def await _test() -> None:
        await run_step("ETHUSDT", "BINANCE", "data/training")

    asyncio.run(_test())