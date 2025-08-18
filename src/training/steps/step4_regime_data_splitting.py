# src/training/steps/step4_regime_data_splitting.py

import asyncio
import os
import json
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.utils.warning_symbols import failed
from src.training.steps.unified_data_loader import get_unified_data_loader

# Import the auto-fix decorator for data quality issues
from src.training.steps.raw_data_quality_checker import auto_fix_data_quality_issues


class RegimeDataSplittingStep:
    """Step 4: Data Splitting for Training - HMM composite clusters only."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("Step4.RegimeSplit")

    async def initialize(self) -> None:
        self.logger.info("🚀 Initializing Step 4: HMM Composite Regime Data Splitting...")
        print("Step4Split ▶ init")
        self.logger.info("✅ HMM Composite Regime Data Splitting initialized successfully")

    async def execute(self) -> dict[str, Any]:
        try:
            self.logger.info("🔄 Loading unified data for HMM composite regime data splitting...")
            print("Step4Split ▶ load_unified_data")
            data_loader = get_unified_data_loader(self.config)
            from src.config.constants import (
                BLANK_TRAINING_LOOKBACK_DAYS,
                FULL_TRAINING_LOOKBACK_DAYS,
                SHORT_BLANK_LOOKBACK_DAYS,
            )

            # Use lookback_days from config (should be passed from enhanced training manager)
            config_lookback = self.config.get(
                "lookback_days", BLANK_TRAINING_LOOKBACK_DAYS
            )
            unified_data = await data_loader.load_unified_data(
                symbol=self.config.get("symbol", "ETHUSDT"),
                exchange=self.config.get("exchange", "BINANCE"),
                timeframe=self.config.get("timeframe", "1m"),
                lookback_days=config_lookback,
            )

            self.logger.info(f"✅ Loaded unified data: {len(unified_data)} rows")
            self.logger.info(
                f"   Date range: {unified_data.index.min()} to {unified_data.index.max()}"
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
            print(f"Step4Split ▶ hmm_composite_splits={len(regime_splits)}")

            # Save regime splits & summary
            self._save_regime_splits(regime_splits)
            summary = self._create_regime_splitting_summary(regime_splits)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f"log/step4_regime_split_{ts}.json", "w") as f:
                json.dump(summary, f, indent=2)

            self.logger.info("✅ HMM composite regime data splitting completed successfully")
            print("Step4Split ▶ done")
            return {"success": True, "regime_splits": summary}
        except Exception as e:
            self.logger.error(f"❌ HMM composite regime data splitting failed: {e}")
            return {"success": False, "error": str(e)}

    def _save_regime_splits(self, regime_splits: dict[str, pd.DataFrame]):
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
                        f"✅ Saved {regime} regime data: {len(regime_df)} rows -> {regime_file}"
                    )
                except Exception as e:
                    self.logger.error(f"🚨 Failed to save {regime} regime data: {e}")
            else:
                self.logger.warning(f"⚠️ No data for {regime} regime")

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


# Import training pipeline decorators for comprehensive security and troubleshooting
from src.utils.training_pipeline_decorators import (
    validate_step_prerequisites,
    secure_data_processing,
    prevent_data_leakage,
    resource_monitor,
    memory_efficient,
    debug_training_step,
    circuit_breaker_protection,
    validate_step_output,
    quality_gate,
    deterministic_seed,
    idempotent_step,
    artifact_write_lock,
    nan_inf_and_constant_guard,
    artifact_versioning,
    time_budget_watchdog,
)


@deterministic_seed(42)
@idempotent_step(step_key="step5_regime_data_splitting")
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
        "required_columns": ["timestamp", "regime", "confidence"],
    },
    context="Regime Data Splitting",
)
@secure_data_processing(
    backup_before=True, integrity_checks=True, memory_cleanup=True, data_validation=True
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
    chunk_size=20000, streaming_processing=True, memory_pool=True, cleanup_frequency=40
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
    data_quality_checks={"min_rows": 100, "required_columns": ["timestamp", "regime"]},
    performance_thresholds={"splitting_time_minutes": 30.0},
    format_validation=True,
)
@quality_gate(
    data_quality_metrics={"completeness": 0.9, "consistency": 0.8},
    validation_score_requirements={"splitting_accuracy": 0.8},
)
@auto_fix_data_quality_issues
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
    async def _test():
        ok = await run_step("ETHUSDT", "BINANCE", "data/training")
        print(f"Step 4 test result: {ok}")

    asyncio.run(_test())
