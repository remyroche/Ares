#!/usr/bin/env python3
"""Step 5: Labeling with Standardized Data Quality Management.

This module creates comprehensive labels for the training data, combining triple barrier
labels with additional labeling strategies and meta-labeling features.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import time
from datetime import datetime

import pandas as pd
import numpy as np
from src.training.hmm_regime_barrier_optimizer import HMMRegimeBarrierOptimizer
from src.training.steps.step4_analyst_labeling_feature_engineering_components.regime_aware_triple_barrier_labeling import apply_regime_aware_triple_barrier_labeling_with_barriers
from src.training.steps.step4_analyst_labeling_feature_engineering_components.optimized_triple_barrier_labeling import OptimizedTripleBarrierLabeling

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import pipeline standards
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards

# Standardized import management
REQUIRED_MODULES = [
    "pandas",
    "numpy",
    "psutil",
    "src.utils.centralized_decorators",
    "src.utils.logger",
    "src.utils.enhanced_mlflow_integration",
    "src.analyst.meta_labeling_system"
]

# Validate environment dependencies
dependency_status = PipelineStandards.validate_environment_dependencies(REQUIRED_MODULES)

# Safe imports with fallbacks
centralized_decorators = PipelineStandards.safe_import("src.utils.centralized_decorators", None)
system_logger = PipelineStandards.safe_import("src.utils.logger", None)
enhanced_mlflow = PipelineStandards.safe_import("src.utils.enhanced_mlflow_integration", None)
meta_labeling_system = PipelineStandards.safe_import("src.analyst.meta_labeling_system", None)
psutil = PipelineStandards.safe_import("psutil", None)
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

if centralized_decorators is None:
    comprehensive_data_validation = create_fallback_decorator()
    handle_errors = create_fallback_decorator()
    memory_efficient = create_fallback_decorator()
    resource_monitor = create_fallback_decorator()
    secure_data_processing = create_fallback_decorator()
    validate_data_structure = create_fallback_decorator()
    with_tracing_span = create_fallback_decorator()
    quality_gate = create_fallback_decorator()
    monitor_feature_engineering = create_fallback_decorator()
else:
    comprehensive_data_validation = centralized_decorators.comprehensive_data_validation
    handle_errors = centralized_decorators.handle_errors
    memory_efficient = centralized_decorators.memory_efficient
    resource_monitor = centralized_decorators.resource_monitor
    secure_data_processing = centralized_decorators.secure_data_processing
    validate_data_structure = centralized_decorators.validate_data_structure
    with_tracing_span = centralized_decorators.with_tracing_span
    quality_gate = centralized_decorators.quality_gate
    monitor_feature_engineering = centralized_decorators.monitor_feature_engineering

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

logger = system_logger.getChild("Step5Labeling")


class LabelingStep:
    """Step 5: Labeling with standardized data quality management."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("LabelingStep")
        self.standards = pipeline_standards
        self.start_time = None
        self.step_timings = {}
        
        # Validate environment on initialization
        self._validate_environment()
        self._initialize_components()

    def _validate_environment(self) -> None:
        """Validate environment dependencies."""
        self.logger.info("🔍 Validating environment dependencies...")
        
        missing_modules = [module for module, available in dependency_status.items() if not available]
        if missing_modules:
            self.logger.warning(f"⚠️ Missing optional modules: {missing_modules}")
            self.logger.info("📝 Pipeline will continue with fallback implementations")
        else:
            self.logger.info("✅ All required dependencies available")

    def _initialize_components(self) -> None:
        """Initialize labeling components."""
        self.logger.info("🔧 Initializing labeling components...")
        
        # Initialize meta-labeling system if available
        if meta_labeling_system is not None:
            try:
                self.meta_labeling_system = meta_labeling_system.MetaLabelingSystem(self.config)
                self.logger.info("✅ Meta-labeling system initialized successfully")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not initialize MetaLabelingSystem: {e}")
                self.meta_labeling_system = None
        else:
            self.logger.warning("⚠️ Meta-labeling system not available")
            self.meta_labeling_system = None

    async def initialize(self) -> None:
        """Initialize the labeling step."""
        self.start_time = time.time()
        self.logger.info("🚀 Initializing Labeling Step...")
        self.logger.info("📋 Step 5 Configuration:")
        self.logger.info(f"   - Symbol: {self.config.get('SYMBOL', 'N/A')}")
        self.logger.info(f"   - Exchange: {self.config.get('EXCHANGE', 'N/A')}")
        self.logger.info(f"   - Timeframe: {self.config.get('TIMEFRAME', 'N/A')}")
        self.logger.info(f"   - Data Directory: {self.config.get('DATA_DIR', 'N/A')}")
        self.logger.info("✅ Labeling Step initialized successfully")

    def _log_step_timing(self, step_name: str, start_time: float) -> None:
        """Log timing information for a step."""
        elapsed = time.time() - start_time
        self.step_timings[step_name] = elapsed
        self.logger.info(f"⏱️ {step_name} completed in {elapsed:.2f} seconds")

    @with_tracing_span("execute_labeling")
    @quality_gate(
        min_quality_score=0.7,
        max_correlation=0.95,
        required_grade="C"
    )
    @with_enhanced_mlflow_logging("step5_labeling")
    @comprehensive_data_validation
    @handle_errors
    @memory_efficient
    @resource_monitor
    @secure_data_processing
    @validate_data_structure
    async def execute_labeling(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str = "data_cache",
        force_rerun: bool = False,
    ) -> bool:
        """Execute the labeling step.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe for data
            data_dir: Data directory
            force_rerun: Force rerun the step

        Returns:
            True if successful, False otherwise
        """
        step_start = time.time()
        self.logger.info(f"🚀 Executing Labeling for {symbol} on {exchange}")

        try:
            # Configuration toggles
            labeling_cfg = self.config.get("vectorized_labelling_orchestrator", {})
            auto_calc = bool(labeling_cfg.get("auto_recalculate_hmm_barriers", True))
            regime_col = str(labeling_cfg.get("hmm_barrier_regime_column", "hmm_regime"))
            time_barrier_minutes = int(labeling_cfg.get("time_barrier_minutes", 30))
            max_lookahead = int(labeling_cfg.get("max_lookahead", 100))

            # Preferred: apply labels now from unified data
            # Fallback: use step4 output if present
            triple_barrier_path = Path(data_dir) / "training" / f"{exchange}_{symbol}_{timeframe}_triple_barrier_labels.parquet"

            labeled_input_df: Optional[pd.DataFrame] = None

            # If we auto-apply labels in step5 or step4 output missing, compute labels here
            do_in_step5 = auto_calc or not triple_barrier_path.exists()
            if do_in_step5:
                self.logger.info("🔁 Applying triple barrier labels within step5 (regime-aware if possible)...")
                # Load unified data
                unified_path = Path(data_dir) / "unified" / exchange / symbol / timeframe
                if not unified_path.exists():
                    self.logger.error(f"❌ Unified data not found at {unified_path}")
                    return False
                data_files = list(unified_path.glob("*.parquet"))
                if not data_files:
                    self.logger.error(f"❌ No parquet files found in {unified_path}")
                    return False
                latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
                self.logger.info(f"📁 Loading unified data from {latest_file}")
                base_df = pd.read_parquet(latest_file)
                self.logger.info(f"✅ Loaded unified data shape: {base_df.shape}")

                # Try regime-aware labeling via HMM barrier optimizer
                try:
                    if auto_calc and regime_col in base_df.columns:
                        hmm_optimizer = HMMRegimeBarrierOptimizer(
                            self.config.get("hmm_regime_barrier_optimizer", {})
                        )
                        _ = await hmm_optimizer.optimize_regime_barriers(
                            base_df, regime_column=regime_col
                        )
                        barriers_path = hmm_optimizer.export_barrier_map()
                        labeled_input_df = apply_regime_aware_triple_barrier_labeling_with_barriers(
                            data=base_df.copy(),
                            barrier_map_or_path=barriers_path,
                            regime_column=regime_col,
                            binary_classification=True,
                            default_time_barrier_minutes=time_barrier_minutes,
                            default_max_lookahead=max_lookahead,
                        )
                        self.logger.info("✅ Regime-aware labels applied in step5")
                    else:
                        self.logger.warning(
                            f"⚠️ Regime-aware labeling not available (auto={auto_calc}, column_present={regime_col in base_df.columns}); using default labeler."
                        )
                        default_labeler = OptimizedTripleBarrierLabeling(
                            profit_take_multiplier=self.config.get("profit_take_multiplier", 0.002),
                            stop_loss_multiplier=self.config.get("stop_loss_multiplier", 0.001),
                            time_barrier_minutes=time_barrier_minutes,
                            max_lookahead=max_lookahead,
                            binary_classification=True,
                        )
                        labeled_input_df = default_labeler.apply_triple_barrier_labeling_vectorized(base_df.copy())
                except Exception as e:
                    self.logger.warning(f"⚠️ In-step regime-aware/default labeling failed: {e}")
                    labeled_input_df = None

                if labeled_input_df is None or labeled_input_df.empty:
                    self.logger.error("❌ Failed to compute labels in step5")
                    return False

                # Prepare data for comprehensive labeling: expect 'triple_barrier_label' present
                data = base_df.copy()
                data["triple_barrier_label"] = labeled_input_df["label"].astype(int)
                if "potential_profit_pct" in labeled_input_df.columns:
                    data["potential_profit_pct"] = labeled_input_df["potential_profit_pct"].astype(float)
                else:
                    data["potential_profit_pct"] = 0.0
                self.logger.info("✅ Prepared base data with triple_barrier_label for comprehensive labeling")
            else:
                # Fallback to step4 result
                self.logger.info(f"📁 Loading triple barrier labels from {triple_barrier_path}")
                data = pd.read_parquet(triple_barrier_path)
                self.logger.info(f"✅ Loaded data with shape: {data.shape}")

            # Generate comprehensive labels
            labeled_data = await self._generate_comprehensive_labels(data, symbol, exchange, timeframe)

            if labeled_data is None:
                self.logger.error("❌ Failed to generate comprehensive labels")
                return False

            # Save results
            output_path = Path(data_dir) / "training" / f"{exchange}_{symbol}_{timeframe}_labeled_data.parquet"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            labeled_data.to_parquet(output_path)
            self.logger.info(f"✅ Labeled data saved to {output_path}")

            # Save labeling metadata
            metadata_path = Path(data_dir) / "training" / f"{exchange}_{symbol}_{timeframe}_labeling_metadata.json"
            metadata = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "total_samples": len(labeled_data),
                "label_distribution": labeled_data['label'].value_counts().to_dict() if 'label' in labeled_data.columns else {},
                "triple_barrier_distribution": labeled_data['triple_barrier_label'].value_counts().to_dict() if 'triple_barrier_label' in labeled_data.columns else {},
                "created_at": pd.Timestamp.now().isoformat(),
                "labeling_config": self.config.get("labeling", {}),
                "regime_column": regime_col if do_in_step5 else None,
                "auto_recalculated_barriers": bool(do_in_step5 and auto_calc),
            }
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            self.logger.info(f"✅ Labeling metadata saved to {metadata_path}")

            self._log_step_timing("Labeling", step_start)

            # Log artifacts and create detailed report
            await self._log_step5_artifacts_and_report(
                symbol, exchange, timeframe, data_dir, labeled_data, output_path, metadata_path
            )

            return True

        except Exception as e:
            self.logger.exception(f"❌ Error in labeling: {e}")
            return False

    async def _log_step5_artifacts_and_report(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        labeled_data: pd.DataFrame,
        output_path: Path,
        metadata_path: Path
    ) -> None:
        """Log step 5 artifacts and create detailed report."""
        try:
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
                str(output_path),
                str(metadata_path),
                f"{exchange}_{symbol}_{timeframe}_labeling_metrics.json",
            ]
            
            # Collect metrics
            metrics_calculated = {
                "labeling_success": 1.0,
                "total_samples": len(labeled_data) if labeled_data is not None else 0,
                "labeled_samples": len(labeled_data[labeled_data['label'].notna()]) if labeled_data is not None else 0,
                "label_distribution": labeled_data['label'].value_counts().to_dict() if labeled_data is not None and 'label' in labeled_data.columns else {},
                "triple_barrier_distribution": labeled_data['triple_barrier_label'].value_counts().to_dict() if labeled_data is not None and 'triple_barrier_label' in labeled_data.columns else {},
            }
            
            # Create training input for report
            training_input = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "data_dir": data_dir,
            ,
                "asset": symbol,  # Use symbol as asset
                "lookback_period": self.config.get("lookback_days", 1095),  # Default to 3 years
                "project_version": self.config.get("project_version", "1.0.0"),  # Default version
            }
            
            # Create step data for report
            step_data = {
                "output_path": str(output_path),
                "metadata_path": str(metadata_path),
                "data_shape": list(labeled_data.shape) if labeled_data is not None else [],
                "label_columns": list(labeled_data.columns) if labeled_data is not None else [],
            }
            
            # Create detailed report
            report_data = create_detailed_step_report(
                step_name="step5_labeling",
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
                step_name="step5_labeling",
                report_data=report_data,
                report_type="labeling_report",
                additional_metadata={
                    "labeling_success": True,
                    "timeframe": timeframe,
                ,
                    "asset": symbol,
                    "lookback_period": self.config.get("lookback_days", 1095),
                    "project_version": self.config.get("project_version", "1.0.0"),
                }
            )
            self.logger.info(f"✅ Logged labeling report: {report_name}")
            
            # Log labeled data DataFrame
            if labeled_data is not None:
                artifact_name = log_step_dataframe_with_standardized_name(
                    config=self.config,
                    step_name="step5_labeling",
                    df=labeled_data,
                    artifact_type="labeled_data",
                    additional_metadata={
                        "artifact_type": "labeled_data",
                        "dataframe_shape": list(labeled_data.shape),
                        "label_distribution": labeled_data['label'].value_counts().to_dict() if 'label' in labeled_data.columns else {,
                    "asset": symbol,
                    "lookback_period": self.config.get("lookback_days", 1095),
                    "project_version": self.config.get("project_version", "1.0.0"),
                },
                        "timeframe": timeframe,
                    }
                )
                self.logger.info(f"✅ Logged labeled data: {artifact_name}")
            
            # Log labeling metadata
            if metadata_path.exists():
                metadata_artifact_name = log_step_artifact_with_standardized_name(
                    config=self.config,
                    step_name="step5_labeling",
                    artifact_path=str(metadata_path),
                    artifact_type="labeling_metadata",
                    additional_metadata={
                        "metadata_type": "labeling_metadata",
                        "timeframe": timeframe,
                    ,
                    "asset": symbol,
                    "lookback_period": self.config.get("lookback_days", 1095),
                    "project_version": self.config.get("project_version", "1.0.0"),
                }
                )
                self.logger.info(f"✅ Logged labeling metadata: {metadata_artifact_name}")
            
            # Log metrics
            log_step_metrics(
                config=self.config,
                step_name="step5_labeling",
                metrics=metrics_calculated,
                additional_metadata={
                    "metrics_type": "labeling_performance",
                    "timeframe": timeframe,
                ,
                    "asset": symbol,
                    "lookback_period": self.config.get("lookback_days", 1095),
                    "project_version": self.config.get("project_version", "1.0.0"),
                }
            )
            
            self.logger.info("✅ Step 5 artifacts and reports logged successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log step 5 artifacts and reports: {e}")
            # Don't fail the step if MLflow logging fails

    async def _generate_comprehensive_labels(self, data: pd.DataFrame, symbol: str, exchange: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Generate comprehensive labels combining multiple labeling strategies."""
        try:
            result_data = data.copy()
            
            # 1. Triple barrier labels (already present)
            if 'triple_barrier_label' not in result_data.columns:
                self.logger.error("❌ Triple barrier labels not found in data")
                return None
            
            # 2. Generate meta-labels if meta-labeling system is available
            if self.meta_labeling_system:
                try:
                    await self.meta_labeling_system.initialize()
                    
                    # Generate analyst labels
                    analyst_labels = await self.meta_labeling_system._generate_analyst_labels(
                        data, symbol, exchange, timeframe
                    )
                    if analyst_labels is not None:
                        result_data['analyst_label'] = analyst_labels
                        self.logger.info("✅ Generated analyst labels")
                    
                    # Generate tactician labels
                    tactician_labels = await self.meta_labeling_system._generate_tactician_labels(
                        data, symbol, exchange, timeframe
                    )
                    if tactician_labels is not None:
                        result_data['tactician_label'] = tactician_labels
                        self.logger.info("✅ Generated tactician labels")
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Meta-labeling failed: {e}")
            
            # 3. Create composite label (primary label for training)
            composite_label = await self._create_composite_label(result_data)
            result_data['label'] = composite_label
            
            # 6. Add label metadata
            result_data['label_confidence'] = await self._calculate_label_confidence(result_data)
            result_data['label_source'] = await self._determine_label_source(result_data)
            
            self.logger.info(f"✅ Generated comprehensive labels with {len(result_data.columns)} columns")
            self.logger.info(f"   - Label distribution: {result_data['label'].value_counts().to_dict()}")
            
            return result_data

        except Exception as e:
            self.logger.exception(f"❌ Error generating comprehensive labels: {e}")
            return None



    async def _create_composite_label(self, data: pd.DataFrame) -> pd.Series:
        """Create composite label from multiple labeling strategies."""
        try:
            # Start with triple barrier labels as base
            composite_label = data['triple_barrier_label'].copy()
            
            # If we have analyst labels, use them to enhance the composite
            if 'analyst_label' in data.columns:
                # Combine triple barrier with analyst labels
                # Analyst labels can override triple barrier in certain conditions
                analyst_override_mask = (
                    (data['analyst_label'] != 0) & 
                    (data['triple_barrier_label'] == 0)
                )
                composite_label[analyst_override_mask] = data['analyst_label'][analyst_override_mask]
            
            return composite_label

        except Exception as e:
            self.logger.warning(f"⚠️ Error creating composite label: {e}")
            # Fallback to triple barrier labels
            return data['triple_barrier_label']

    async def _calculate_label_confidence(self, data: pd.DataFrame) -> pd.Series:
        """Calculate confidence scores for labels."""
        try:
            confidence = np.ones(len(data), dtype=np.float32)
            
            # Higher confidence when multiple labeling strategies agree
            if 'analyst_label' in data.columns:
                agreement_mask = (data['label'] == data['analyst_label']) & (data['analyst_label'] != 0)
                confidence[agreement_mask] += 0.2
            
            # Cap confidence at 1.0
            confidence = np.minimum(confidence, 1.0)
            
            return pd.Series(confidence, index=data.index)

        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating label confidence: {e}")
            return pd.Series(1.0, index=data.index)

    async def _determine_label_source(self, data: pd.DataFrame) -> pd.Series:
        """Determine the source of each label."""
        try:
            sources = []
            
            for idx in range(len(data)):
                if data['label'].iloc[idx] == data['triple_barrier_label'].iloc[idx]:
                    if 'analyst_label' in data.columns and data['label'].iloc[idx] == data['analyst_label'].iloc[idx]:
                        sources.append("triple_barrier+analyst")
                    else:
                        sources.append("triple_barrier")
                elif 'analyst_label' in data.columns and data['label'].iloc[idx] == data['analyst_label'].iloc[idx]:
                    sources.append("analyst")
                else:
                    sources.append("composite")
            
            return pd.Series(sources, index=data.index)

        except Exception as e:
            self.logger.warning(f"⚠️ Error determining label source: {e}")
            return pd.Series("unknown", index=data.index)


async def run_step(
    symbol: str,
    exchange: str,
    timeframe: str,
    data_dir: str = None,
    force_rerun: bool = False,
    config: Optional[Dict[str, Any]] = None,
) -> bool:
    """Run the labeling step with standardized data quality management.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe for data
        data_dir: Data directory (will use standardized path if None)
        force_rerun: Force rerun the step
        config: Configuration dictionary

    Returns:
        True if successful, False otherwise
    """
    if config is None:
        config = {}

    # Use standardized path construction
    if data_dir is None:
        data_dir = pipeline_standards.build_path("processed_data", exchange, symbol)

    # Add step-specific configuration
    step_config = {
        "SYMBOL": symbol,
        "EXCHANGE": exchange,
        "TIMEFRAME": timeframe,
        "DATA_DIR": data_dir,
        "labeling": {
            "enable_meta_labeling": True,
            "enable_trend_labels": True,
            "enable_volatility_labels": True,
            "composite_label_strategy": "weighted_combination",
        },
        **config
    }

    step = LabelingStep(step_config)
    await step.initialize()
    
    return await step.execute_labeling(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        force_rerun=force_rerun,
    )


if __name__ == "__main__":
    # Test the step
    async def test():
        success = await run_step(
            symbol="ETHUSDT",
            exchange="BINANCE",
            timeframe="1m",
            data_dir="data_cache"
        )
        print(f"Step 5 result: {success}")

    asyncio.run(test())