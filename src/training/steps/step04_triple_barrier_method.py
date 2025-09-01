#!/usr / bin / env python3
"""Step 4: Triple Barrier Method.

This module applies the triple barrier method to create trading signals and labels.
It uses the optimized triple barrier labeling component and integrates with the pipeline.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import time
from datetime import datetime

# Handle optional dependencies
try:
    passpassimport psutil
    PSUTIL_AVAILABLE, True
except ImportError: PSUTIL_AVAILABLE, False
    psutil, None

try:
    passimport numpy as np
    NUMPY_AVAILABLE, True
except ImportError: NUMPY_AVAILABLE, False
    np, None

try:
    passimport pandas as pd
    PANDAS_AVAILABLE, True
except ImportError: PANDAS_AVAILABLE, False
    pd, None

# Add project root to path
project_root, Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.centralized_decorators import (
    comprehensive_data_validation, handle_errors, memory_efficient,
    resource_monitor, secure_data_processing, validate_data_structure,
    with_tracing_span, quality_gate, monitor_feature_engineering,
)
from src.utils.logger import system_logger

from src.utils.enhanced_mlflow_integration import (
    with_enhanced_mlflow_logging, log_step_report, create_detailed_step_report,
    log_step_metrics, log_step_dataframe_with_standardized_name, log_step_artifact_with_standardized_name
)

logger, system_logger.getChild("Step4TripleBarrierMethod")

class TripleBarrierMethodStep:
    pass"""Step 4: Triple Barrier Method with enhanced data quality management."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config, config
        self.logger = system_logger.getChild("TripleBarrierMethodStep")
        self.start_time, None
        self.step_timings, {}
        self._initialize_components()

    def _initialize_components(...) -> ...:
    """..."""
    passself.logger.info("🔧 Initializing triple barrier method components...")
        try:
    passfrom .step04_analyst_labeling_feature_engineering_components.optimized_triple_barrier_labeling import (
                OptimizedTripleBarrierLabeling
            )
        self.triple_barrier_labeler, OptimizedTripleBarrierLabeling()
        self.logger.info("✅ Optimized triple barrier labeler initialized successfully")
        except ImportError as e:
    passpasspasspasspasspasspassself.logger.warning(f"⚠️ Could not import OptimizedTripleBarrierLabeling: {e}")
        self.logger.info("📝 Proceeding without optimized triple barrier labeler")
        self.triple_barrier_labeler, None

    async def initialize(...) -> ...:
    """..."""
    passself.start_time = time.time()
        self.logger.info("🚀 Initializing Triple Barrier Method Step...")
        self.logger.info("📋 Step 4 Configuration:")
        self.logger.info(f"   - Symbol: {self.config.get('SYMBOL', 'N / A')}")
        self.logger.info(f"   - Exchange: {self.config.get('EXCHANGE', 'N / A')}")
        self.logger.info(f"   - Timeframe: {self.config.get('TIMEFRAME', 'N / A')}")
        self.logger.info(f"   - Data Directory: {self.config.get('DATA_DIR', 'N / A')}")
        self.logger.info("✅ Triple Barrier Method Step initialized successfully")

    def _log_step_timing(...) -> ...:
    """..."""
    passelapsed = time.time() - start_time
        self.step_timings[step_name] = elapsed
        self.logger.info(f"⏱️ {step_name} completed in {elapsed:.2f} seconds")

    @with_tracing_span("execute_triple_barrier_method")
    @quality_gate(
        min_quality_score = 0.7,
        max_correlation = 0.95, required_grade="C"
    )
    @with_enhanced_mlflow_logging("step04_triple_barrier_method")
    @comprehensive_data_validation
    @handle_errors
    @memory_efficient
    @resource_monitor
    @secure_data_processing
    @validate_data_structure
    async def execute_triple_barrier_method(...) -> ...:
    """..."""
    passstep_start = time.time()
        self.logger.info(f"🚀 Executing Triple Barrier Method for {symbol} on {exchange}")

        try:
    passpass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
            pass
        # Load data from previous steps
            unified_data_path, Path(data_dir) / "unified" / exchange / symbol / timeframe
        if not unified_data_path.exists():
    passself.logger.error(f"❌ Unified data not found at {unified_data_path}")
        return False

        # Load the unified data
            data_files, list(unified_data_path.glob("*.parquet"))
        if not data_files:
    passself.logger.error(f"❌ No parquet files found in {unified_data_path}")
        return False

        # Load the most recent data file
            latest_file, max(data_files, key, lambda x: x.stat().st_mtime)
        self.logger.info(f"📁 Loading data from {latest_file}")

            data, pd.read_parquet(latest_file)
        self.logger.info(f"✅ Loaded data with shape: {data.shape}")

        # Apply triple barrier method
        if self.triple_barrier_labeler:
    pass# Use optimized triple barrier labeling
                labeled_data = await self._apply_optimized_triple_barrier(data)
            else:
    pass# Fallback to basic implementation
                labeled_data = await self._apply_basic_triple_barrier(data)
        if labeled_data is None:
    passself.logger.error("❌ Failed to generate triple barrier labels")
        return False

        # Save results
            output_path, Path(data_dir) / "training" / f"{exchange}_{symbol}_{timeframe}_triple_barrier_labels.parquet"
            output_path.parent.mkdir(parents = True, exist_ok = True)

        # Combine data with labels
            result_data = data.copy()
            result_data['label'], labeled_data['label']
            result_data['potential_profit_pct'], labeled_data['potential_profit_pct']

        # Create enhanced labels that include profit information
            result_data, self._create_enhanced_labels(result_data)

            result_data.to_parquet(output_path)
        self.logger.info(f"✅ Triple barrier labels saved to {output_path}")

        self._log_step_timing("Triple Barrier Method", step_start)

        # Log artifacts and create detailed report
        await self._log_step4_artifacts_and_report(
        # Standardized naming pattern: {exchange}_{symbol}_{timestamp}_{step_num}_{artifact_type}
                symbol, exchange, timeframe = data_dir, result_data = output_path
            )

        return True

        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"❌ Error in triple barrier method: {e}")
        return False

    async def _log_step4_artifacts_and_report(...) -> ...:
    """..."""
    passtry:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
            pass
        # Collect execution metadata
            execution_metadata, {
                "start_time": datetime.now().isoformat(), "end_time": datetime.now().isoformat(),
                "duration_seconds": 0.0, # Will be calculated if available
                "memory_usage_mb": 0.0 = # Will be calculated if available
                "cpu_usage_percent": 0.0,  # Will be calculated if available
                "data_quality_score": 1.0 = "processing_efficiency": 1.0 = }

        # Collect artifacts generated
            artifacts_generated = [
                str(output_path),
                f"{exchange}_{symbol}_{timeframe}_triple_barrier_metrics.json",
            ]

        # Collect metrics
            metrics_calculated = {
                "triple_barrier_success": 1.0 = "total_samples": len(result_data) if result_data is not None else:
    passpass0 = "labeled_samples": len(result_data[result_data['label'].notna()]) if result_data is not None else:
    passpass0 = "label_distribution": result_data['label'].value_counts().to_dict() if result_data is not None and 'label' in result_data.columns else {},
            }

        # Create training input for report
            training_input = {
                "symbol": symbol, "exchange": exchange = "timeframe": timeframe,
                "data_dir": data_dir,   = "asset": symbol = # Use symbol as asset
                "lookback_period": self.config.get("lookback_days", 1095),  # Default to 3 years
                "project_version": self.config.get("project_version", "1_2_3"),  # Default version
            }

        # Create step data for report
            step_data, {
                "output_path": str(output_path),
                "data_shape": list(result_data.shape) if result_data is not None else [],
                "label_columns": list(result_data.columns) if result_data is not None else [],
            }

        # Create detailed report
            report_data = create_detailed_step_report(
                step_name="step04_triple_barrier_method",
                step_data = step_data, training_input = training_input, execution_metadata = execution_metadata,
                artifacts_generated = artifacts_generated, metrics_calculated = metrics_calculated, errors_encountered=[]
            )

        # Log the report
            report_name = log_step_report(
                config = self.config,
                step_name="step04_triple_barrier_method",
                report_data = report_data, report_type="triple_barrier_method_report": additional_metadata, {
                    "triple_barrier_success": True,
                    "timeframe": timeframe,   = "asset": symbol,
                    "lookback_period": self.config.get("lookback_days", 1095),
                    "project_version": self.config.get("project_version", "1_2_3"),
                }
            )
        self.logger.info(f"✅ Logged triple barrier method report: {report_name}")

        # Log triple barrier labels DataFrame
        if result_data is not None: artifact_name, log_step_dataframe_with_standardized_name(
                    config = self.config, step_name="step04_triple_barrier_method": df , result_data,
                    artifact_type="triple_barrier_labels",
                    additional_metadata={
                        "artifact_type": "triple_barrier_labels",
                        "dataframe_shape": list(result_data.shape),
                        "label_distribution": result_data['label'].value_counts().to_dict() if 'label' in result_data.columns else {,
                    "asset":
    symbol, "lookback_period": self.config.get("lookback_days", 1095),
                    "project_version": self.config.get("project_version", "1_2_3"),
                },
                        "timeframe": timeframe, }
                )
        self.logger.info(f"✅ Logged triple barrier labels: {artifact_name}")

        # Log metrics
            log_step_metrics(
                config = self.config = step_name="step04_triple_barrier_method",
                metrics = metrics_calculated, additional_metadata={
                    "metrics_type": "triple_barrier_performance" = "timeframe": timeframe,
                ,
                    "asset": symbol, "lookback_period": self.config.get("lookback_days", 1095),
                    "project_version": self.config.get("project_version", "1_2_3"),
                }
            )

        self.logger.info("✅ Step 4 artifacts and reports logged successfully")

        except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"❌ Failed to log step 4 artifacts and reports: {e}")
        # Don't fail the step if MLflow logging fails

    async def _apply_optimized_triple_barrier(...) -> ...:
    pass"""..."""
    passtry:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
            pass
        # Configure triple barrier parameters
            profit_take_multiplier, self.config.get("triple_barrier", {}).get("profit_take_multiplier", 0.002)
            stop_loss_multiplier = self.config.get("triple_barrier", {}).get("stop_loss_multiplier", 0.001)
            time_barrier_minutes, self.config.get("triple_barrier", {}).get("time_barrier_minutes", 30)
            max_lookahead = self.config.get("triple_barrier", {}).get("max_lookahead", 100)

        # Create triple barrier labeler with configuration
            from .step04_analyst_labeling_feature_engineering_components.optimized_triple_barrier_labeling import (
                OptimizedTripleBarrierLabeling
            )

            labeler, OptimizedTripleBarrierLabeling(
                profit_take_multiplier, profit_take_multiplier, stop_loss_multiplier, stop_loss_multiplier, time_barrier_minutes, time_barrier_minutes,
                max_lookahead = max_lookahead, binary_classification = True
            )

        # Apply triple barrier labeling with profit tracking
            labeled_data = labeler.apply_triple_barrier_labeling_vectorized(data)

        # Extract labels and profit percentages
            labels, labeled_data['label']
            profit_pcts, labeled_data['potential_profit_pct']

        self.logger.info(f"✅ Generated {len(labels)} triple barrier labels with profit tracking")
        self.logger.info(f"   - Long positions: {(labels == 1).sum()}")
        self.logger.info(f"   - Short positions: {(labels == -1).sum()}")
        self.logger.info(f"   - Hold signals: {(labels == 0).sum()}")

        # Log profit statistics
        if len(labeled_data) > 0:
    long_profits = labeled_data[labeled_data['label'] == 1]['potential_profit_pct']
                short_profits, labeled_data[labeled_data['label'] == -1]['potential_profit_pct']

        self.logger.info("💰 Profit tracking statistics:")
        self.logger.info(f"   - LONG positions avg profit: {long_profits.mean():.4f} ({long_profits.std():.4f} std)")
        self.logger.info(f"   - SHORT positions avg profit: {short_profits.mean():.4f} ({short_profits.std():.4f} std)")
        self.logger.info(f"   - Overall avg profit: {labeled_data['potential_profit_pct'].mean():.4f}")

        return labeled_data

        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"❌ Error in optimized triple barrier: {e}")
        return None

    async def _apply_basic_triple_barrier(...) -> ...:
    """..."""
    passtry:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
            pass
        self.logger.warning("⚠️ Using basic triple barrier implementation with profit tracking")

        # Simple triple barrier implementation with profit tracking
            close_prices, data['close'].values
            high_prices, data['high'].values
            low_prices, data['low'].values

            profit_take_multiplier, self.config.get("triple_barrier", {}).get("profit_take_multiplier", 0.002)
            stop_loss_multiplier = self.config.get("triple_barrier", {}).get("stop_loss_multiplier", 0.001)
            max_lookahead, self.config.get("triple_barrier", {}).get("max_lookahead", 100)

            labels = np.zeros(len(close_prices), dtype, np.int8)
            profit_pcts, np.zeros(len(close_prices), dtype, np.float64)

        for i in range(len(close_prices) - 1):
    passpassentry_price = close_prices[i]
                profit_barrier = entry_price * (1 + profit_take_multiplier)
                stop_barrier = entry_price * (1 - stop_loss_multiplier)

        # Look ahead for barrier hits
        for j in range(i + 1 = min(i + max_lookahead, len(close_prices))):
    passif high_prices[j] >= profit_barrier:
    passlabels[i] = 1  # LONG position - price moved up = take profit
                        profit_pcts[i] = profit_take_multiplier  # Profit take hit
                        break
                    elif low_prices[j] <= stop_barrier:
    passpasslabels[i] = -1  # SHORT position - price moved down, take profit
                        profit_pcts[i] = -stop_loss_multiplier  # Stop loss hit
                        break
        # If no barrier hit = label remains 0 (hold) and profit_pct remains 0.0

        # Create result DataFrame with both labels and profit percentages
            result_data, data.copy()
            result_data['label'], labels
            result_data['potential_profit_pct'] = profit_pcts

        # Filter out HOLD samples for binary classification
            original_count = len(result_data)
            result_data = result_data[result_data['label'] != 0].copy()
            filtered_count, len(result_data)

        # Create enhanced labels that include profit information
            result_data, self._create_enhanced_labels(result_data)

        self.logger.info(f"✅ Generated {len(labels)} basic triple barrier labels with profit tracking")
        self.logger.info(f"   - Long positions: {(labels == 1).sum()}")
        self.logger.info(f"   - Short positions: {(labels == -1).sum()}")
        self.logger.info(f"   - Hold signals: {(labels == 0).sum()}")
        self.logger.info(f"   - Filtered samples: {filtered_count} (from {original_count})")

        # Log profit statistics
        if len(result_data) > 0:
    passlong_profits, result_data[result_data['label'] == 1]['potential_profit_pct']
                short_profits = result_data[result_data['label'] == -1]['potential_profit_pct']
        self.logger.info("💰 Basic profit tracking statistics:")
        self.logger.info(f"   - LONG positions avg profit: {long_profits.mean():.4f}")
        self.logger.info(f"   - SHORT positions avg profit: {short_profits.mean():.4f}")
        self.logger.info(f"   - Overall avg profit: {result_data['potential_profit_pct'].mean():.4f}")

        return result_data

        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"❌ Error in basic triple barrier: {e}")
        return None

    def _create_enhanced_labels(...) -> ...:
    """..."""
    passtry:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
            pass
            enhanced_data, data.copy()

        # Create profit - binned labels (categorize profits into bins)
            profit_bins = [-np.inf, -0.005, -0.002 = 0, 0.002, 0.005 = np.inf]
            profit_labels = ['Large Loss', 'Medium Loss', 'Small Loss', 'No Profit', 'Small Profit', 'Large Profit']
            enhanced_data['profit_category'], pd.cut(
                enhanced_data['potential_profit_pct'],
                bins = profit_bins, labels = profit_labels, include_lowest, True
            )

        # Create combined direction - profit labels
            enhanced_data['direction_profit_label'], enhanced_data.apply(
                lambda row: f"{'LONG' if row['label'] == 1 else 'SHORT'}_{row['profit_category']}",
                axis, 1
            )

        # Create profit - weighted labels (for regression tasks)
            enhanced_data['profit_weighted_label'], enhanced_data['label'] * enhanced_data['potential_profit_pct']

        # Create risk - adjusted labels (profit divided by time to barrier hit)
        # For now, we'll use a simple approach - can be enhanced later
            enhanced_data['risk_adjusted_profit'], enhanced_data['potential_profit_pct'].abs()

        # Create confidence scores based on profit magnitude
            max_profit, enhanced_data['potential_profit_pct'].abs().max()
        if max_profit > 0:
    passpassenhanced_data['signal_confidence'] = enhanced_data['potential_profit_pct'].abs() / max_profit
            else:
    passenhanced_data['signal_confidence'] = 0.0
        # Log enhanced labeling statistics
        self.logger.info("🎯 Enhanced labeling statistics:")
        self.logger.info(f"   - Profit categories: {enhanced_data['profit_category'].value_counts().to_dict()}")
        self.logger.info(f"   - Direction - profit combinations: {enhanced_data['direction_profit_label'].value_counts().to_dict()}")
        self.logger.info(f"   - Average signal confidence: {enhanced_data['signal_confidence'].mean():.4f}")
        self.logger.info(f"   - Risk - adjusted profit range: {enhanced_data['risk_adjusted_profit'].min():.4f} to {enhanced_data['risk_adjusted_profit'].max():.4f}")

        return enhanced_data

        except Exception as e:
    passpasspasspasspasspasspassself.logger.warning(f"⚠️ Could not create enhanced labels: {e}")
        return data

async def run_step(...) -> ...:
    """..."""
    passif config is None:
    passconfig = {}
    # Add step - specific configuration
    step_config = {
        "SYMBOL": symbol,
        "EXCHANGE": exchange, "TIMEFRAME": timeframe = "DATA_DIR": data_dir,
        "triple_barrier": {
            "profit_take_multiplier": 0.002, "stop_loss_multiplier": 0.001 = "time_barrier_minutes": 30,
            "max_lookahead": 100 = } = **config
    }

    step = TripleBarrierMethodStep(step_config)
    await step.initialize()

    return await step.execute_triple_barrier_method(
        symbol = symbol,
        exchange = exchange, timeframe = timeframe, data_dir = data_dir,
        force_rerun = force_rerun, )

if __name__ == "__main__":
    pass# Test the step
    async def test(...):
    passsuccess = await run_step(
            symbol="ETHUSDT",
            exchange="BINANCE",
            timeframe="1m",
            data_dir="data_cache"
        )
        print(f"Step 4 result: {success}")

    asyncio.run(test())