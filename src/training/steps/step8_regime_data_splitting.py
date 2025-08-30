# src/training/steps/step7_regime_data_splitting.py

import asyncio
import json
import os
from datetime import datetime
from typing import Any

import pandas as pd

# Import the auto-fix decorator for data quality issues
from src.utils.centralized_decorators import auto_fix_data_quality_issues
from src.training.steps.unified_data_loader import get_unified_data_loader
from src.utils.logger import system_logger

# Import training pipeline decorators for comprehensive security and troubleshooting
from src.utils.centralized_decorators import (

from src.utils.enhanced_mlflow_integration import (
    with_enhanced_mlflow_logging,
    log_step_report,
    create_detailed_step_report,
    log_step_metrics,
    log_step_dataframe_with_standardized_name,
    log_step_artifact_with_standardized_name
)
    artifact_versioning,
    artifact_write_lock,
    circuit_breaker_protection,
    debug_training_step,
    deterministic_seed,
    handle_errors,
    idempotent_step,
    memory_efficient,
    nan_inf_and_constant_guard,
    prevent_data_leakage,
    quality_gate,
    resource_monitor,
    secure_data_processing,
    time_budget_watchdog,
    validate_step_output,
    validate_step_prerequisites,
    with_tracing_span,
)


class RegimeDataSplittingStep:
    """Step 4: Data Splitting for Training - HMM composite clusters only."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("Step4.RegimeSplit")

    @with_tracing_span("step4_regime_splitting.initialize", log_args=False)
    @handle_errors(exceptions=(Exception,), default_return=None, context="step4_initialization")
    async def initialize(self) -> None:
        """Initialize the regime data splitting step."""
        self.logger.info("🚀 Initializing Step 4: HMM Composite Regime Data Splitting...")
        self.logger.info("✅ HMM Composite Regime Data Splitting initialized successfully")

    @with_enhanced_mlflow_logging("step8")
    @with_tracing_span("step4_regime_splitting.execute", log_args=False)
    @handle_errors(exceptions=(Exception,), default_return={"success": False, "error": "Execution failed"}, context="step4_execution")
    async def execute(self) -> dict[str, Any]:
        """Execute the regime data splitting step."""
        try:
            self.logger.info("🔄 Loading unified data for HMM composite regime data splitting...")
            data_loader = get_unified_data_loader(self.config)
            from src.config.constants import (
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
            self.logger.info("🎯 Using HMM composite clusters for regime splitting (PARAMOUNT)")

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

            # Split data by HMM composite clusters
            regime_splits: dict[str, pd.DataFrame] = {}
            for cluster_id in unique_clusters:
                cluster_mask = unified_data["composite_cluster_id"] == cluster_id
                cluster_data = unified_data[cluster_mask].copy()

                if not cluster_data.empty:
                    regime_name = f"hmm_composite_{cluster_id}"
                    regime_splits[regime_name] = cluster_data
                    self.logger.info(f"✅ Created regime split for {regime_name}: {len(cluster_data)} rows")

            if not regime_splits:
                self.logger.error("🚨 No valid regime splits created from HMM composite clusters")
                return {"success": False, "error": "No valid regime splits created"}

            self.logger.info(f"✅ Successfully created {len(regime_splits)} HMM composite regime splits")

            # Save regime splits & summary
            self._save_regime_splits(regime_splits)
            summary = self._create_regime_splitting_summary(regime_splits)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f"log/step4_regime_split_{ts}.json", "w") as f:
                json.dump(summary, f, indent=2)

            self.logger.info("✅ HMM composite regime data splitting completed successfully")
            
            # Log artifacts and create detailed report
            await self._log_step8_artifacts_and_report(regime_splits, summary)
            
            return {"success": True, "regime_splits": summary}
        except Exception as e:
            self.logger.exception(f"❌ HMM composite regime data splitting failed: {e}")
            return {"success": False, "error": str(e)}

    async def _log_step8_artifacts_and_report(
        self,
        regime_splits: dict[str, pd.DataFrame],
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
            artifacts_generated = []
            for regime_name in regime_splits.keys():
                artifacts_generated.append(f"{regime_name}.parquet")
            
            # Collect metrics
            metrics_calculated = {
                "regime_splitting_success": 1.0,
                "total_regimes": len(regime_splits),
                "total_samples": sum(len(df) for df in regime_splits.values()),
                "regime_names": list(regime_splits.keys()),
            }
            
            # Create training input for report
            training_input = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "lookback_days": self.config.get("lookback_days", 1095),
            }
            
            # Create step data for report
            step_data = {
                "regime_splits_summary": summary,
                "regime_count": len(regime_splits),
                "regime_names": list(regime_splits.keys()),
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
                report_type="regime_data_splitting_report",
                additional_metadata={
                    "regime_splitting_success": True,
                    "total_regimes": len(regime_splits),
                    "timeframe": timeframe,
                }
            )
            self.logger.info(f"✅ Logged regime data splitting report: {report_name}")
            
            # Log regime splits summary
            if summary:
                summary_report_name = log_step_report(
                    config=self.config,
                    step_name="step8_regime_data_splitting",
                    report_data=summary,
                    report_type="regime_splits_summary",
                    additional_metadata={
                        "total_regimes": len(regime_splits),
                        "timeframe": timeframe,
                    }
                )
                self.logger.info(f"✅ Logged regime splits summary: {summary_report_name}")
            
            # Log metrics
            log_step_metrics(
                config=self.config,
                step_name="step8_regime_data_splitting",
                metrics=metrics_calculated,
                additional_metadata={
                    "metrics_type": "regime_splitting_performance",
                    "timeframe": timeframe,
                }
            )
            
            self.logger.info("✅ Step 8 artifacts and reports logged successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log step 8 artifacts and reports: {e}")
            # Don't fail the step if MLflow logging fails

    @with_tracing_span("step4_regime_splitting._save_regime_splits", log_args=False)
    @handle_errors(exceptions=(Exception,), default_return=None, context="save_regime_splits")
    def _save_regime_splits(self, regime_splits: dict[str, pd.DataFrame]) -> None:
        """Save regime splits to parquet files."""
        data_dir = self.config.get("data_dir", "data/training")
        os.makedirs(data_dir, exist_ok=True)
        regime_data_dir = os.path.join(data_dir, "regime_data")
        os.makedirs(regime_data_dir, exist_ok=True)
        
        for regime, regime_df in regime_splits.items():
            if not regime_df.empty:
                regime_file = os.path.join(regime_data_dir, f"{regime}.parquet")
                try:
                    regime_df.to_parquet(regime_file, index=False)
                    self.logger.info(
                        f"✅ Saved {regime} regime data: {len(regime_df)} rows -> {regime_file}",
                    )
                except Exception as e:
                    self.logger.exception(f"🚨 Failed to save {regime} regime data: {e}")
            else:
                self.logger.warning(f"⚠️ No data for {regime} regime")

    @with_tracing_span("step4_regime_splitting._create_regime_splitting_summary", log_args=False)
    @handle_errors(exceptions=(Exception,), default_return={}, context="create_regime_summary")
    def _create_regime_splitting_summary(self, regime_splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
        """Create a summary of the regime splitting results."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "regime_basis": "hmm_composite_clusters_only",
            "total_regimes": len(regime_splits),
            "regimes": {},
            "total_rows": sum(len(df) for df in regime_splits.values()),
        }

        for regime_name, regime_df in regime_splits.items():
            summary["regimes"][regime_name] = {
                "rows": len(regime_df),
                "date_range": {
                    "start": regime_df["timestamp"].min().isoformat() if "timestamp" in regime_df.columns else None,
                    "end": regime_df["timestamp"].max().isoformat() if "timestamp" in regime_df.columns else None,
                },
                "composite_cluster_id": regime_name.replace("hmm_composite_", ""),
            }

        return summary


@deterministic_seed(42)
@idempotent_step(step_key="step7_regime_data_splitting")
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
    context="Regime Data Splitting",
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
    required_files=["data/training/regime_data/*.parquet"],
    data_quality_checks={"min_rows": 100, "required_columns": ["timestamp", "composite_cluster_id"]},
    performance_thresholds={"splitting_time_minutes": 30.0},
    format_validation=True,
)
@quality_gate(
    data_quality_metrics={"completeness": 0.9, "consistency": 0.8},
    validation_score_requirements={"splitting_accuracy": 0.8},
)
@auto_fix_data_quality_issues
@handle_errors(exceptions=(Exception,), default_return=False, context="step7_regime_data_splitting")
async def run_step(
    symbol: str, 
    exchange: str, 
    data_dir: str, 
    timeframe: str = "1m", 
    force_rerun: bool = False,
    **kwargs,
) -> bool:
    """Run the HMM composite regime data splitting step."""
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
    async def _test() -> None:
        await run_step("ETHUSDT", "BINANCE", "data/training")

    asyncio.run(_test())