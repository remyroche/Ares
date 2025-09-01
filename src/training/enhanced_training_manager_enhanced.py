"""
Enhanced Training Manager with Existing Decorators Integration
Provides thorough decorators, detailed reports, and consistent storage for all pipeline steps.
"""

import json
import time
from datetime import datetime
from pathlib import Path

from src.training.enhanced_training_manager import EnhancedTrainingManager
from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors
from src.utils.training_pipeline_decorators import (
import monitor_pipeline_step,
    monitor_pipeline_step,
    validate_pipeline_input,
    monitor_pipeline_performance,
    PipelineStage,
    PipelineValidationLevel
)


class EnhancedTrainingManagerWithReporting(EnhancedTrainingManager):
    """
    Enhanced Training Manager with comprehensive decorators and detailed reporting.

    This class extends the base EnhancedTrainingManager to provide:
    1. Thorough decorators for each pipeline step using existing decorators
    2. Detailed reports upon completion
    3. Consistent storage of all reports in a centralized location
    """

    def __init__(self, config: Dict[str, Any]):
    pass
    pass
        super().__init__(config)
        self.logger = system_logger.getChild("EnhancedTrainingManagerWithReporting")
        self.pipeline_reports_dir = Path("reports/enhanced_training_pipeline")
        self.pipeline_reports_dir.mkdir(parents=True, exist_ok=True)

        # Initialize reporting configuration
        self.reporting_config = config.get("enhanced_reporting", {})
        self.enable_detailed_reporting = self.reporting_config.get("enable_detailed_reporting", True)
        self.auto_cleanup_reports = self.reporting_config.get("auto_cleanup_reports", True)
        self.reports_retention_days = self.reporting_config.get("reports_retention_days", 30)

        # Track step execution for reporting
        self.current_pipeline_execution_id = None
        self.step_reports = {}

        self.logger.info(f"🚀 Enhanced Training Manager with Reporting initialized")
        self.logger.info(f"   📁 Reports Directory: {self.pipeline_reports_dir}")
        self.logger.info(f"   🧹 Auto Cleanup: {self.auto_cleanup_reports}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="enhanced_training_execution"
    )
    @monitor_pipeline_step(
        stage=PipelineStage.MODEL_TRAINING,
        validation_level=PipelineValidationLevel.WARNING,
        enable_data_quality=True,
        memory_threshold=80.0,
        duration_threshold=3600.0  # 1 hour
    )
    async def execute_enhanced_training(
        self,
        enhanced_training_input: dict[str, Any],
    ) -> bool:
        """Execute the comprehensive enhanced training pipeline with detailed reporting."""

        # Generate unique execution ID for this pipeline run
        self.current_pipeline_execution_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{enhanced_training_input.get('symbol', 'unknown')}_{enhanced_training_input.get('exchange', 'unknown')}"
        self.step_reports = {}

        # Initialize pipeline reporting
        pipeline_report = {
            "pipeline_execution_id": self.current_pipeline_execution_id,
            "pipeline_start_time": datetime.now().isoformat(),
            "training_input": enhanced_training_input,
            "steps": {},
            "overall_metrics": {},
            "artifacts": {},
            "validation_results": {},
            "errors": [],
            "warnings": [],
            "recommendations": []
        }

        try:
            # Call the parent method with enhanced monitoring
    except Exception as e:
        pass
    except Exception as e:
        pass
            result = await super().execute_enhanced_training(enhanced_training_input)

            # Generate comprehensive pipeline report
            pipeline_report["pipeline_end_time"] = datetime.now().isoformat()
            pipeline_report["overall_success"] = result
            pipeline_report["steps"] = self.step_reports

            # Generate and store pipeline report
            await self._generate_pipeline_report(pipeline_report)

            return result

        except Exception as e:
            pipeline_report["errors"].append({
                "type": type(e).__name__,
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            })
            pipeline_report["steps"] = self.step_reports
            await self._generate_pipeline_report(pipeline_report)
            raise

    async def _generate_step_report(self, step_name: str, step_result: Any, step_start_time: float, step_success: bool, step_errors: List[str] = None, step_warnings: List[str] = None):
        """Generate and append step information to shared pipeline report."""

        if not self.enable_detailed_reporting:
    pass
    pass
            return

        try:
            step_end_time = time.time()
    except Exception as e:
        pass
    except Exception as e:
        pass
            execution_duration = step_end_time - step_start_time

            # Get step-specific quality metrics
            step_quality_metrics = await self._get_step_quality_metrics(step_name, step_result)

            # Create step report section
            step_report_section = {
                "step_name": step_name,
                "pipeline_execution_id": self.current_pipeline_execution_id,
                "execution_start_time": datetime.fromtimestamp(step_start_time).isoformat(),
                "execution_end_time": datetime.fromtimestamp(step_end_time).isoformat(),
                "execution_duration_seconds": execution_duration,
                "execution_duration_formatted": f"{execution_duration:.2f}s",
                "success": step_success,
                "result_type": type(step_result).__name__,
                "result_summary": self._summarize_result(step_result),
                "step_quality_metrics": step_quality_metrics,
                "errors": step_errors or [],
                "warnings": step_warnings or [],
                "system_resources": await self._get_system_resources(),
                "timestamp": datetime.now().isoformat()
            }

            # Load existing shared report or create new one
            shared_report_path = self.pipeline_reports_dir / f"{self.current_pipeline_execution_id}_shared_report.json"

            if shared_report_path.exists():
    pass
    pass
                with open(shared_report_path, 'r', encoding='utf-8') as f:
                    shared_report = json.load(f)
            else:
                shared_report = {
                    "pipeline_execution_id": self.current_pipeline_execution_id,
                    "pipeline_start_time": datetime.fromtimestamp(step_start_time).isoformat(),
                    "pipeline_config": self.config,
                    "steps": {},
                    "pipeline_summary": {
                        "total_steps": len(self.STEP_ORDER),
                        "completed_steps": 0,
                        "failed_steps": 0,
                        "total_duration": 0,
                        "overall_success": True
                    }
                }

            # Append step information to shared report
            shared_report["steps"][step_name] = step_report_section
            shared_report["pipeline_summary"]["completed_steps"] = len(shared_report["steps"])
            shared_report["pipeline_summary"]["failed_steps"] = sum(1 for step in shared_report["steps"].values() if not step["success"])
            shared_report["pipeline_summary"]["overall_success"] = shared_report["pipeline_summary"]["failed_steps"] == 0
            shared_report["pipeline_summary"]["total_duration"] = sum(step["execution_duration_seconds"] for step in shared_report["steps"].values())

            # Save updated shared report
            with open(shared_report_path, 'w', encoding='utf-8') as f:
                json.dump(shared_report, f, indent=2, ensure_ascii=False, default=str)

            # Generate step summary
            summary_report = self._generate_step_summary(step_report_section)
            summary_filename = f"{step_name}_{self.current_pipeline_execution_id}_summary.txt"
            summary_path = self.pipeline_reports_dir / summary_filename

            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary_report)

            # Store in memory for pipeline summary
            self.step_reports[step_name] = step_report_section

            # Log completion
            status_emoji = "✅" if step_success else "❌"
            self.logger.info(f"{status_emoji} [STEP REPORT] {step_name} appended to shared report: {shared_report_path}")

        except Exception as e:
            self.logger.error(f"❌ Failed to generate step report for {step_name}: {e}")

    def _summarize_result(self, result: Any) -> Dict[str, Any]:
    pass
    pass
        """Create a summary of the step result."""

        try:
            if hasattr(result, 'shape'):  # DataFrame
                return {
                    "type": "DataFrame",
                    "shape": result.shape,
                    "columns_count": len(result.columns),
                    "memory_usage_mb": result.memory_usage(deep=True).sum() / (1024**2) if hasattr(result, 'memory_usage') else None
                }
    except Exception as e:
        pass
    except Exception as e:
        pass
            elif isinstance(result, dict):
                return {
                    "type": "dict",
                    "keys_count": len(result),
                    "keys": list(result.keys())[:10]  # First 10 keys
                }
            elif isinstance(result, (list, tuple)):
                return {
                    "type": type(result).__name__,
                    "length": len(result),
                    "element_types": [type(item).__name__ for item in result[:5]]  # First 5 elements
                }
            elif isinstance(result, bool):
                return {
                    "type": "boolean",
                    "value": result
                }
            else:
                return {
                    "type": type(result).__name__,
                    "value_preview": str(result)[:100]  # First 100 characters
                }
        except Exception:
            return {
                "type": "unknown",
                "error": "Could not summarize result"
            }

    async def _get_system_resources(self) -> Dict[str, Any]:
        """Get current system resource usage."""

        try:
            import psutil

    except Exception as e:
        pass
    except Exception as e:
        pass
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent()
            disk = psutil.disk_usage('/')

            return {
                "memory_usage_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "cpu_usage_percent": cpu,
                "disk_usage_percent": disk.percent,
                "disk_available_gb": disk.free / (1024**3)
            }
        except Exception:
            return {
                "error": "Could not retrieve system resources"
            }

    def _generate_step_summary(self, step_report: Dict[str, Any]) -> str:
    pass
    pass
        """Generate a human-readable summary for a step report."""

        summary = []
        summary.append("=" * 80)
        summary.append(f"STEP EXECUTION REPORT: {step_report['step_name']}")
        summary.append("=" * 80)
        summary.append(f"Pipeline Execution ID: {step_report['pipeline_execution_id']}")
        summary.append(f"Execution Start: {step_report['execution_start_time']}")
        summary.append(f"Execution End: {step_report['execution_end_time']}")
        summary.append(f"Duration: {step_report['execution_duration_formatted']}")
        summary.append(f"Success: {step_report['success']}")
        summary.append(f"Result Type: {step_report['result_type']}")
        summary.append("")

        # Result summary
        if step_report.get("result_summary"):
    pass
    pass
            result_summary = step_report["result_summary"]
            summary.append("RESULT SUMMARY:")
            summary.append("-" * 40)
            for key, value in result_summary.items():
    pass
    pass
                summary.append(f"  {key}: {value}")
            summary.append("")

        # Step-specific quality metrics
        if step_report.get("step_quality_metrics"):
    pass
    pass
            quality_metrics = step_report["step_quality_metrics"]
            summary.append("STEP QUALITY METRICS:")
            summary.append("-" * 40)

            # Handle different types of quality metrics based on step
            step_name = step_report['step_name']

            if step_name in ["step01_data_collection", "step01_5_data_converter"]:
    pass
    pass
                if "data_quality" in quality_metrics:
    pass
    pass
                    data_quality = quality_metrics["data_quality"]
                    summary.append("  Data Quality:")
                    summary.append(f"    Total Rows: {data_quality.get('total_rows', 'N/A')}")
                    summary.append(f"    Total Columns: {data_quality.get('total_columns', 'N/A')}")
                    summary.append(f"    Memory Usage: {data_quality.get('memory_usage_mb', 'N/A'):.2f} MB")

                    if "null_percentage" in data_quality:
    pass
    pass
                        max_null = max(data_quality["null_percentage"].values()) if data_quality["null_percentage"] else 0
                        summary.append(f"    Max Null Percentage: {max_null:.2f}%")

                    if "duplicate_percentage" in data_quality:
    pass
    pass
                        summary.append(f"    Duplicate Rows: {data_quality['duplicate_percentage']:.2f}%")

                if "data_validation" in quality_metrics:
    pass
    pass
                    validation = quality_metrics["data_validation"]
                    summary.append("  Data Validation:")
                    summary.append(f"    Has Required Columns: {validation.get('has_required_columns', 'N/A')}")

                    if "price_consistency" in validation:
    pass
    pass
                        price_check = validation["price_consistency"]
                        summary.append(f"    Price Consistency: {'❌ Issues' if price_check.get('has_issues') else '✅ OK'}")
                        if price_check.get('issues'):
    pass
    pass
                            for issue in price_check['issues']:
    pass
    pass
                                summary.append(f"      - {issue}")

                    if "volume_consistency" in validation:
    pass
    pass
                        volume_check = validation["volume_consistency"]
                        summary.append(f"    Volume Consistency: {'❌ Issues' if volume_check.get('has_issues') else '✅ OK'}")
                        if volume_check.get('issues'):
    pass
    pass
                            for issue in volume_check['issues']:
    pass
    pass
                                summary.append(f"      - {issue}")

            elif step_name == "step02_feature_engineering":
                if "feature_quality" in quality_metrics:
    pass
    pass
                    feature_quality = quality_metrics["feature_quality"]
                    summary.append("  Feature Quality:")
                    summary.append(f"    Total Features: {feature_quality.get('total_features', 'N/A')}")
                    summary.append(f"    Numeric Features: {feature_quality.get('numeric_features', 'N/A')}")
                    summary.append(f"    Categorical Features: {feature_quality.get('categorical_features', 'N/A')}")
                    summary.append(f"    Memory Usage: {feature_quality.get('memory_usage_mb', 'N/A'):.2f} MB")

                if "multicollinearity_analysis" in quality_metrics:
    pass
    pass
                    multicollinearity = quality_metrics["multicollinearity_analysis"]
                    summary.append("  Multicollinearity Analysis:")
                    summary.append(f"    High Correlation Pairs: {multicollinearity.get('high_correlation_count', 'N/A')}")

                    if "high_correlation_pairs" in multicollinearity and multicollinearity["high_correlation_pairs"]:
    pass
    pass
                        summary.append("    High Correlation Pairs Details:")
                        for pair in multicollinearity["high_correlation_pairs"][:5]:  # Show first 5
                            summary.append(f"      - {pair['feature1']} ↔ {pair['feature2']} (r={pair['correlation']:.3f})")
                        if len(multicollinearity["high_correlation_pairs"]) > 5:
    pass
    pass
                            summary.append(f"      ... and {len(multicollinearity['high_correlation_pairs']) - 5} more pairs")

                    if "high_vif_features" in multicollinearity:
    pass
    pass
                        high_vif = multicollinearity["high_vif_features"]
                        summary.append(f"    High VIF Features ({len(high_vif)}):")
                        if high_vif:
    pass
    pass
                            for feature in high_vif[:10]:  # Show first 10
                                vif_score = multicollinearity.get("vif_scores", {}).get(feature, "N/A")
                                summary.append(f"      - {feature} (VIF: {vif_score})")
                            if len(high_vif) > 10:
    pass
    pass
                                summary.append(f"      ... and {len(high_vif) - 10} more features")

                if "feature_statistics" in quality_metrics:
    pass
    pass
                    feature_stats = quality_metrics["feature_statistics"]
                    summary.append("  Feature Statistics:")

                    constant_features = feature_stats.get('constant_features', [])
                    summary.append(f"    Constant Features ({len(constant_features)}):")
                    if constant_features:
    pass
    pass
                        for feature in constant_features[:10]:
    pass
    pass
                            summary.append(f"      - {feature}")
                        if len(constant_features) > 10:
    pass
    pass
                            summary.append(f"      ... and {len(constant_features) - 10} more")

                    low_var_features = feature_stats.get('low_variance_features', [])
                    summary.append(f"    Low Variance Features ({len(low_var_features)}):")
                    if low_var_features:
    pass
    pass
                        for feature in low_var_features[:10]:
    pass
    pass
                            summary.append(f"      - {feature}")
                        if len(low_var_features) > 10:
    pass
    pass
                            summary.append(f"      ... and {len(low_var_features) - 10} more")

                    high_card_features = feature_stats.get('high_cardinality_features', [])
                    summary.append(f"    High Cardinality Features ({len(high_card_features)}):")
                    if high_card_features:
    pass
    pass
                        for feature in high_card_features[:10]:
    pass
    pass
                            summary.append(f"      - {feature}")
                        if len(high_card_features) > 10:
    pass
    pass
                            summary.append(f"      ... and {len(high_card_features) - 10} more")

                if "data_quality_issues" in quality_metrics:
    pass
    pass
                    quality_issues = quality_metrics["data_quality_issues"]
                    summary.append("  Data Quality Issues:")

                    nan_features = quality_issues.get('nan_features', [])
                    summary.append(f"    NaN Features ({len(nan_features)}):")
                    if nan_features:
    pass
    pass
                        for feature in nan_features[:10]:
    pass
    pass
                            summary.append(f"      - {feature}")
                        if len(nan_features) > 10:
    pass
    pass
                            summary.append(f"      ... and {len(nan_features) - 10} more")

                    inf_features = quality_issues.get('inf_features', [])
                    summary.append(f"    Inf Features ({len(inf_features)}):")
                    if inf_features:
    pass
    pass
                        for feature in inf_features[:10]:
    pass
    pass
                            summary.append(f"      - {feature}")
                        if len(inf_features) > 10:
    pass
    pass
                            summary.append(f"      ... and {len(inf_features) - 10} more")

                    zero_var_features = quality_issues.get('zero_variance_features', [])
                    summary.append(f"    Zero Variance Features ({len(zero_var_features)}):")
                    if zero_var_features:
    pass
    pass
                        for feature in zero_var_features[:10]:
    pass
    pass
                            summary.append(f"      - {feature}")
                        if len(zero_var_features) > 10:
    pass
    pass
                            summary.append(f"      ... and {len(zero_var_features) - 10} more")

            elif step_name == "step03_hmm_regime_discovery":
                if "regime_analysis" in quality_metrics:
    pass
    pass
                    regime_analysis = quality_metrics["regime_analysis"]
                    summary.append("  Regime Analysis:")
                    summary.append(f"    Number of Regimes: {regime_analysis.get('number_of_regimes', 'N/A')}")
                    summary.append(f"    Convergence Status: {regime_analysis.get('convergence_status', 'N/A')}")
                    summary.append(f"    Log Likelihood: {regime_analysis.get('log_likelihood', 'N/A')}")

                if "validation_metrics" in quality_metrics:
    pass
    pass
                    validation_metrics = quality_metrics["validation_metrics"]
                    summary.append("  Validation Metrics:")
                    summary.append(f"    AIC Score: {validation_metrics.get('aic_score', 'N/A')}")
                    summary.append(f"    BIC Score: {validation_metrics.get('bic_score', 'N/A')}")
                    summary.append(f"    Model Complexity: {validation_metrics.get('model_complexity', 'N/A')}")

            elif step_name == "step04_regime_data_splitting":
                if "splitting_analysis" in quality_metrics:
    pass
    pass
                    splitting = quality_metrics["splitting_analysis"]
                    summary.append("  Splitting Analysis:")
                    summary.append(f"    Total Regimes: {splitting.get('total_regimes', 'N/A')}")
                    summary.append(f"    Train/Test Split: {splitting.get('train_test_split_ratio', 'N/A')}")
                    summary.append(f"    Validation Split: {splitting.get('validation_split_ratio', 'N/A')}")
                    summary.append(f"    Stratified Splitting: {splitting.get('stratified_splitting', 'N/A')}")

                if "data_distribution" in quality_metrics:
    pass
    pass
                    distribution = quality_metrics["data_distribution"]
                    summary.append("  Data Distribution:")
                    summary.append(f"    Total Samples: {distribution.get('total_samples', 'N/A')}")
                    summary.append(f"    Train Samples: {distribution.get('train_samples', 'N/A')}")
                    summary.append(f"    Test Samples: {distribution.get('test_samples', 'N/A')}")
                    summary.append(f"    Validation Samples: {distribution.get('validation_samples', 'N/A')}")

                if "time_distribution" in quality_metrics:
    pass
    pass
                    time_dist = quality_metrics["time_distribution"]
                    summary.append("  Time Distribution:")
                    summary.append(f"    Regime Time Periods: {time_dist.get('regime_time_periods', 'N/A')}")
                    summary.append(f"    Regime Transition Frequency: {time_dist.get('regime_transition_frequency', 'N/A')}")

                    if "regime_duration_stats" in time_dist:
    pass
    pass
                        duration_stats = time_dist["regime_duration_stats"]
                        summary.append("    Regime Duration Statistics:")
                        for regime, stats in duration_stats.items():
    pass
    pass
                            if isinstance(stats, dict):
    pass
    pass
                                summary.append(f"      {regime}: Mean={stats.get('mean_duration', 'N/A'):.2f}s, Min={stats.get('min_duration', 'N/A')}s, Max={stats.get('max_duration', 'N/A')}s")

                if "quality_validation" in quality_metrics:
    pass
    pass
                    validation = quality_metrics["quality_validation"]
                    summary.append("  Quality Validation:")
                    summary.append(f"    No Data Leakage: {validation.get('no_data_leakage', 'N/A')}")
                    summary.append(f"    Temporal Consistency: {validation.get('temporal_consistency', 'N/A')}")

                    if "regime_representation" in validation:
    pass
    pass
                        regime_rep = validation["regime_representation"]
                        summary.append(f"    All Regimes Represented: {regime_rep.get('all_regimes_represented', 'N/A')}")
                        if not regime_rep.get('all_regimes_represented', True):
    pass
    pass
                            missing = regime_rep.get('missing_regimes_in_test', [])
                            summary.append(f"    Missing Regimes in Test: {missing}")

            elif step_name == "step05_triple_barrier_method":
                if "barrier_analysis" in quality_metrics:
    pass
    pass
                    barrier = quality_metrics["barrier_analysis"]
                    summary.append("  Barrier Analysis:")
                    summary.append(f"    Total Labels: {barrier.get('total_labels', 'N/A')}")

                    if "barrier_parameters" in barrier:
    pass
    pass
                        params = barrier["barrier_parameters"]
                        summary.append(f"    Upper Barrier: {params.get('upper_barrier', 'N/A')}")
                        summary.append(f"    Lower Barrier: {params.get('lower_barrier', 'N/A')}")
                        summary.append(f"    Time Horizon: {params.get('time_horizon', 'N/A')}")

                if "daily_statistics" in quality_metrics:
    pass
    pass
                    daily_stats = quality_metrics["daily_statistics"]
                    summary.append("  Daily Statistics:")
                    summary.append(f"    Average Barriers per Day: {daily_stats.get('average_barriers_per_day', 'N/A'):.2f}" if isinstance(daily_stats.get('average_barriers_per_day'), (int, float)) else f"    Average Barriers per Day: {daily_stats.get('average_barriers_per_day', 'N/A')}")
                    summary.append(f"    Total Trading Days: {daily_stats.get('total_trading_days', 'N/A')}")
                    summary.append(f"    Days with Barriers: {daily_stats.get('days_with_barriers', 'N/A')}")
                    summary.append(f"    Barrier Density: {daily_stats.get('barrier_density', 'N/A'):.4f}" if isinstance(daily_stats.get('barrier_density'), (int, float)) else f"    Barrier Density: {daily_stats.get('barrier_density', 'N/A')}")

                if "barrier_values" in quality_metrics:
    pass
    pass
                    barrier_vals = quality_metrics["barrier_values"]
                    summary.append("  Barrier Values:")
                    summary.append(f"    Upper Barrier Value: {barrier_vals.get('upper_barrier_value', 'N/A')}")
                    summary.append(f"    Lower Barrier Value: {barrier_vals.get('lower_barrier_value', 'N/A')}")
                    summary.append(f"    Barrier Spread: {barrier_vals.get('barrier_spread', 'N/A')}")
                    summary.append(f"    Barrier Volatility: {barrier_vals.get('barrier_volatility', 'N/A')}")

                if "position_ratios" in quality_metrics:
    pass
    pass
                    position_ratios = quality_metrics["position_ratios"]
                    summary.append("  Position Ratios:")
                    summary.append(f"    Long/Short Ratio: {position_ratios.get('long_short_ratio', 'N/A'):.3f}" if isinstance(position_ratios.get('long_short_ratio'), (int, float)) else f"    Long/Short Ratio: {position_ratios.get('long_short_ratio', 'N/A')}")
                    summary.append(f"    Long Positions: {position_ratios.get('long_positions', 'N/A')}")
                    summary.append(f"    Short Positions: {position_ratios.get('short_positions', 'N/A')}")
                    summary.append(f"    Hold Positions: {position_ratios.get('hold_positions', 'N/A')}")

                if "triple_barrier_captured_changes" in quality_metrics:
    pass
    pass
                    captured_changes = quality_metrics["triple_barrier_captured_changes"]
                    summary.append("  Triple Barrier Captured Changes:")

                    if "barrier_hit_analysis" in captured_changes:
    pass
    pass
                        hit_analysis = captured_changes["barrier_hit_analysis"]

                        # Upper barrier hits without lower barrier hits first
                        upper_first = hit_analysis.get("upper_hits_without_lower_first", {})
                        summary.append("    Upper Barrier Hits (Without Lower First):")
                        summary.append(f"      Total Count: {upper_first.get('total_count', 'N/A')}")
                        summary.append(f"      Long Positions: {upper_first.get('long_positions', 'N/A')}")
                        summary.append(f"      Short Positions: {upper_first.get('short_positions', 'N/A')}")
                        summary.append(f"      Average Post-Hit Movement: {upper_first.get('average_post_hit_movement', 'N/A'):.4f}" if isinstance(upper_first.get('average_post_hit_movement'), (int, float)) else f"      Average Post-Hit Movement: {upper_first.get('average_post_hit_movement', 'N/A')}")
                        summary.append(f"      Max Post-Hit Movement: {upper_first.get('max_post_hit_movement', 'N/A'):.4f}" if isinstance(upper_first.get('max_post_hit_movement'), (int, float)) else f"      Max Post-Hit Movement: {upper_first.get('max_post_hit_movement', 'N/A')}")

                        # Lower barrier hits without upper barrier hits first
                        lower_first = hit_analysis.get("lower_hits_without_upper_first", {})
                        summary.append("    Lower Barrier Hits (Without Upper First):")
                        summary.append(f"      Total Count: {lower_first.get('total_count', 'N/A')}")
                        summary.append(f"      Long Positions: {lower_first.get('long_positions', 'N/A')}")
                        summary.append(f"      Short Positions: {lower_first.get('short_positions', 'N/A')}")

                    if "upper_barrier_post_hit_analysis" in captured_changes:
    pass
    pass
                        post_hit_analysis = captured_changes["upper_barrier_post_hit_analysis"]
                        summary.append("    Upper Barrier Post-Hit Analysis:")
                        summary.append(f"      Total Post-Hit Movements: {post_hit_analysis.get('total_post_hit_movements', 'N/A')}")
                        summary.append(f"      Mean Post-Hit Movement: {post_hit_analysis.get('mean_post_hit_movement', 'N/A'):.4f}" if isinstance(post_hit_analysis.get('mean_post_hit_movement'), (int, float)) else f"      Mean Post-Hit Movement: {post_hit_analysis.get('mean_post_hit_movement', 'N/A')}")
                        summary.append(f"      Max Post-Hit Movement: {post_hit_analysis.get('max_post_hit_movement', 'N/A'):.4f}" if isinstance(post_hit_analysis.get('max_post_hit_movement'), (int, float)) else f"      Max Post-Hit Movement: {post_hit_analysis.get('max_post_hit_movement', 'N/A')}")

                        movement_dist = post_hit_analysis.get("post_hit_movement_distribution", {})
                        summary.append("      Post-Hit Movement Distribution:")
                        summary.append(f"        Small (≤1%): {movement_dist.get('small_movements', 'N/A')}")
                        summary.append(f"        Medium (1-5%): {movement_dist.get('medium_movements', 'N/A')}")
                        summary.append(f"        Large (>5%): {movement_dist.get('large_movements', 'N/A')}")

                    if "summary_statistics" in captured_changes:
    pass
    pass
                        summary_stats = captured_changes["summary_statistics"]
                        summary.append("    Summary Statistics:")
                        summary.append(f"      Total Barrier Hits: {summary_stats.get('total_barrier_hits', 'N/A')}")
                        summary.append(f"      Upper First Hits: {summary_stats.get('upper_first_hits', 'N/A')}")
                        summary.append(f"      Lower First Hits: {summary_stats.get('lower_first_hits', 'N/A')}")
                        summary.append(f"      Both Barriers Hit: {summary_stats.get('both_barriers_hit', 'N/A')}")
                        summary.append(f"      Upper First Ratio: {summary_stats.get('upper_first_ratio', 'N/A'):.4f}" if isinstance(summary_stats.get('upper_first_ratio'), (int, float)) else f"      Upper First Ratio: {summary_stats.get('upper_first_ratio', 'N/A')}")
                        summary.append(f"      Lower First Ratio: {summary_stats.get('lower_first_ratio', 'N/A'):.4f}" if isinstance(summary_stats.get('lower_first_ratio'), (int, float)) else f"      Lower First Ratio: {summary_stats.get('lower_first_ratio', 'N/A')}")

                if "label_quality" in quality_metrics:
    pass
    pass
                    label_quality = quality_metrics["label_quality"]
                    summary.append("  Label Quality:")
                    summary.append(f"    Label Consistency: {label_quality.get('label_consistency', 'N/A')}")
                    summary.append(f"    No Label Leakage: {label_quality.get('no_label_leakage', 'N/A')}")

                    if "balanced_labels" in label_quality:
    pass
    pass
                        balance = label_quality["balanced_labels"]
                        summary.append(f"    Labels Balanced: {balance.get('is_balanced', 'N/A')}")
                        if not balance.get('is_balanced', True):
    pass
    pass
                            summary.append(f"    Majority Class: {balance.get('majority_class', 'N/A')}")
                            summary.append(f"    Minority Class: {balance.get('minority_class', 'N/A')}")

            elif step_name == "step06_feature_generation":
                if "feature_generation_analysis" in quality_metrics:
    pass
    pass
                    generation = quality_metrics["feature_generation_analysis"]
                    summary.append("  Feature Generation Analysis:")
                    summary.append(f"    Generation Type: {generation.get('generation_type', 'N/A')}")
                    summary.append(f"    Original Features: {generation.get('original_features', 'N/A')}")
                    summary.append(f"    Generated Features: {generation.get('generated_features', 'N/A')}")
                    summary.append(f"    Feature Increase: {generation.get('feature_increase', 'N/A')}")
                    summary.append(f"    Generation Methods: {generation.get('generation_methods', 'N/A')}")

                if "feature_quality" in quality_metrics:
    pass
    pass
                    quality = quality_metrics["feature_quality"]
                    summary.append("  Feature Quality:")
                    summary.append(f"    Feature Relevance: {quality.get('feature_relevance', 'N/A'):.4f}" if isinstance(quality.get('feature_relevance'), (int, float)) else f"    Feature Relevance: {quality.get('feature_relevance', 'N/A')}")
                    summary.append(f"    Information Gain: {quality.get('information_gain', 'N/A'):.4f}" if isinstance(quality.get('information_gain'), (int, float)) else f"    Information Gain: {quality.get('information_gain', 'N/A')}")
                    summary.append(f"    Feature Diversity: {quality.get('feature_diversity', 'N/A'):.4f}" if isinstance(quality.get('feature_diversity'), (int, float)) else f"    Feature Diversity: {quality.get('feature_diversity', 'N/A')}")
                    summary.append(f"    Feature Stability: {quality.get('feature_stability', 'N/A'):.4f}" if isinstance(quality.get('feature_stability'), (int, float)) else f"    Feature Stability: {quality.get('feature_stability', 'N/A')}")

                if "generation_performance" in quality_metrics:
    pass
    pass
                    performance = quality_metrics["generation_performance"]
                    summary.append("  Generation Performance:")
                    summary.append(f"    Generation Time: {performance.get('generation_time', 'N/A')}")
                    summary.append(f"    Memory Usage: {performance.get('memory_usage', 'N/A')}")
                    summary.append(f"    Computational Efficiency: {performance.get('computational_efficiency', 'N/A'):.4f}" if isinstance(performance.get('computational_efficiency'), (int, float)) else f"    Computational Efficiency: {performance.get('computational_efficiency', 'N/A')}")
                    summary.append(f"    Parallel Processing: {performance.get('parallel_processing', 'N/A')}")

                if "feature_statistics" in quality_metrics:
    pass
    pass
                    stats = quality_metrics["feature_statistics"]
                    summary.append("  Feature Statistics:")
                    summary.append(f"    Feature Types: {stats.get('feature_types', 'N/A')}")
                    summary.append(f"    Feature Complexity: {stats.get('feature_complexity', 'N/A')}")
                    summary.append(f"    Feature Correlations: {stats.get('feature_correlations', 'N/A')}")
                    summary.append(f"    Feature Redundancy: {stats.get('feature_redundancy', 'N/A'):.4f}" if isinstance(stats.get('feature_redundancy'), (int, float)) else f"    Feature Redundancy: {stats.get('feature_redundancy', 'N/A')}")

            elif step_name == "step07_matrix_feature_selection":
                if "selection_analysis" in quality_metrics:
    pass
    pass
                    selection = quality_metrics["selection_analysis"]
                    summary.append("  Selection Analysis:")
                    summary.append(f"    Selection Method: {selection.get('selection_method', 'N/A')}")
                    summary.append(f"    Original Features: {selection.get('original_features', 'N/A')}")
                    summary.append(f"    Selected Features: {selection.get('selected_features', 'N/A')}")
                    summary.append(f"    Reduction Ratio: {selection.get('reduction_ratio', 'N/A'):.4f}" if isinstance(selection.get('reduction_ratio'), (int, float)) else f"    Reduction Ratio: {selection.get('reduction_ratio', 'N/A')}")
                    summary.append(f"    Selection Criteria: {selection.get('selection_criteria', 'N/A')}")

                if "selection_quality" in quality_metrics:
    pass
    pass
                    quality = quality_metrics["selection_quality"]
                    summary.append("  Selection Quality:")
                    summary.append(f"    Feature Importance: {quality.get('feature_importance', 'N/A'):.4f}" if isinstance(quality.get('feature_importance'), (int, float)) else f"    Feature Importance: {quality.get('feature_importance', 'N/A')}")
                    summary.append(f"    Information Preservation: {quality.get('information_preservation', 'N/A'):.4f}" if isinstance(quality.get('information_preservation'), (int, float)) else f"    Information Preservation: {quality.get('information_preservation', 'N/A')}")
                    summary.append(f"    Selection Stability: {quality.get('selection_stability', 'N/A'):.4f}" if isinstance(quality.get('selection_stability'), (int, float)) else f"    Selection Stability: {quality.get('selection_stability', 'N/A')}")
                    summary.append(f"    Cross Validation Score: {quality.get('cross_validation_score', 'N/A'):.4f}" if isinstance(quality.get('cross_validation_score'), (int, float)) else f"    Cross Validation Score: {quality.get('cross_validation_score', 'N/A')}")

                if "matrix_analysis" in quality_metrics:
    pass
    pass
                    matrix = quality_metrics["matrix_analysis"]
                    summary.append("  Matrix Analysis:")
                    summary.append(f"    Correlation Matrix: {matrix.get('correlation_matrix', 'N/A')}")
                    summary.append(f"    Variance Explained: {matrix.get('variance_explained', 'N/A'):.4f}" if isinstance(matrix.get('variance_explained'), (int, float)) else f"    Variance Explained: {matrix.get('variance_explained', 'N/A')}")
                    summary.append(f"    Eigenvalue Distribution: {matrix.get('eigenvalue_distribution', 'N/A')}")
                    summary.append(f"    Multicollinearity Reduction: {matrix.get('multicollinearity_reduction', 'N/A'):.4f}" if isinstance(matrix.get('multicollinearity_reduction'), (int, float)) else f"    Multicollinearity Reduction: {matrix.get('multicollinearity_reduction', 'N/A')}")

                if "performance_impact" in quality_metrics:
    pass
    pass
                    impact = quality_metrics["performance_impact"]
                    summary.append("  Performance Impact:")
                    summary.append(f"    Pre-Selection Accuracy: {impact.get('pre_selection_accuracy', 'N/A'):.4f}" if isinstance(impact.get('pre_selection_accuracy'), (int, float)) else f"    Pre-Selection Accuracy: {impact.get('pre_selection_accuracy', 'N/A')}")
                    summary.append(f"    Post-Selection Accuracy: {impact.get('post_selection_accuracy', 'N/A'):.4f}" if isinstance(impact.get('post_selection_accuracy'), (int, float)) else f"    Post-Selection Accuracy: {impact.get('post_selection_accuracy', 'N/A')}")
                    summary.append(f"    Accuracy Change: {impact.get('accuracy_change', 'N/A'):.4f}" if isinstance(impact.get('accuracy_change'), (int, float)) else f"    Accuracy Change: {impact.get('accuracy_change', 'N/A')}")
                    summary.append(f"    Computational Savings: {impact.get('computational_savings', 'N/A')}")

                if "selected_features_analysis" in quality_metrics:
    pass
    pass
                    features = quality_metrics["selected_features_analysis"]
                    summary.append("  Selected Features Analysis:")
                    summary.append(f"    Top Features: {features.get('top_features', 'N/A')}")
                    summary.append(f"    Feature Categories: {features.get('feature_categories', 'N/A')}")
                    summary.append(f"    Feature Rankings: {features.get('feature_rankings', 'N/A')}")
                    summary.append(f"    Selection Confidence: {features.get('selection_confidence', 'N/A'):.4f}" if isinstance(features.get('selection_confidence'), (int, float)) else f"    Selection Confidence: {features.get('selection_confidence', 'N/A')}")

            # Add warnings from quality metrics
            if "warnings" in quality_metrics and quality_metrics["warnings"]:
    pass
    pass
                summary.append("  Quality Warnings:")
                for warning in quality_metrics["warnings"]:
    pass
    pass
                    summary.append(f"    ⚠️ {warning}")

            summary.append("")

        # System resources
        if step_report.get("system_resources"):
    pass
    pass
            resources = step_report["system_resources"]
            summary.append("SYSTEM RESOURCES:")
            summary.append("-" * 40)
            for key, value in resources.items():
    pass
    pass
                if isinstance(value, float):
    pass
    pass
                    summary.append(f"  {key}: {value:.2f}")
                else:
                    summary.append(f"  {key}: {value}")
            summary.append("")

        # Errors and warnings
        if step_report.get("errors"):
    pass
    pass
            summary.append("ERRORS:")
            summary.append("-" * 40)
            for error in step_report["errors"]:
    pass
    pass
                summary.append(f"❌ {error}")
            summary.append("")

        if step_report.get("warnings"):
    pass
    pass
            summary.append("WARNINGS:")
            summary.append("-" * 40)
            for warning in step_report["warnings"]:
    pass
    pass
                summary.append(f"⚠️ {warning}")
            summary.append("")

        summary.append("=" * 80)
        summary.append("End of Step Report")
        summary.append("=" * 80)

        return "\\\n".join(summary)

    # Enhanced step execution methods with report generation
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step01_data_collection"
    )
    @monitor_pipeline_step(
        stage=PipelineStage.DATA_COLLECTION,
        validation_level=PipelineValidationLevel.WARNING,
        enable_data_quality=True
    )
    @validate_pipeline_input(
        required_params=["symbol", "exchange", "timeframe", "data_dir"],
        required_directories=["data_cache"],
        min_memory_gb=4.0,
        min_disk_gb=2.0
    )
    async def _execute_step1_enhanced(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        force_rerun: bool,
    ) -> bool:
        """Execute Step 1: Data Collection with enhanced reporting."""

        step_start_time = time.time()
        step_errors = []
        step_warnings = []

        try:
            from src.training.steps import step01_data_collection

    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
import result = await step01_data_collection.run_step
            result = await step01_data_collection.run_step(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                force_rerun=force_rerun,
            )

            # Generate step report
            await self._generate_step_report(
                "step01_data_collection",
                result,
                step_start_time,
                bool(result),
                step_errors,
                step_warnings
            )

            return result

        except Exception as e:
            step_errors.append(str(e))
            self.logger.error(f"❌ Step 1 failed: {e}")

            # Generate step report even on failure
            await self._generate_step_report(
                "step01_data_collection",
                None,
                step_start_time,
                False,
                step_errors,
                step_warnings
            )
            raise

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step01_5_data_converter"
    )
    @monitor_pipeline_step(
        stage=PipelineStage.DATA_PREPROCESSING,
        validation_level=PipelineValidationLevel.WARNING,
        enable_data_quality=True
    )
    @validate_pipeline_input(
        required_params=["symbol", "exchange", "timeframe", "data_dir"],
        required_directories=["data_cache"],
        min_memory_gb=4.0,
        min_disk_gb=2.0
    )
    async def _execute_step1_5_enhanced(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        force_rerun: bool,
    ) -> bool:
        """Execute Step 1.5: Data Converter with enhanced reporting."""

        step_start_time = time.time()
        step_errors = []
        step_warnings = []

        try:
            from src.training.steps.step01_5_data_converter import run_step as step01_5_run_step

    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
import result = await step01_5_run_step
            result = await step01_5_run_step(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                force_rerun=force_rerun,
            )

            # Generate step report
            await self._generate_step_report(
                "step01_5_data_converter",
                result,
                step_start_time,
                bool(result),
                step_errors,
                step_warnings
            )

            return result

        except Exception as e:
            step_errors.append(str(e))
            self.logger.error(f"❌ Step 1.5 failed: {e}")

            # Generate step report even on failure
            await self._generate_step_report(
                "step01_5_data_converter",
                None,
                step_start_time,
                False,
                step_errors,
                step_warnings
            )
            raise

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step02_feature_engineering"
    )
    @monitor_pipeline_step(
        stage=PipelineStage.FEATURE_ENGINEERING,
        validation_level=PipelineValidationLevel.WARNING,
        enable_data_quality=True
    )
    @monitor_pipeline_performance(
        enable_memory_tracking=True,
        enable_cpu_tracking=True,
        memory_threshold_gb=16.0,
        cpu_threshold_percent=90.0
    )
    async def _execute_step2_enhanced(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
        timeframe: str,
        force_rerun: bool,
        feature_config: dict,
    ) -> bool:
        """Execute Step 2: Feature Engineering with enhanced reporting."""

        step_start_time = time.time()
        step_errors = []
        step_warnings = []

        try:
            from src.training.steps import step02_feature_engineering

    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
import result = await step02_feature_engineering.run_step
            result = await step02_feature_engineering.run_step(
                symbol=symbol,
                exchange=exchange,
                data_dir=data_dir,
                timeframe=timeframe,
                force_rerun=force_rerun,
                feature_config=feature_config,
            )

            # Generate step report
            await self._generate_step_report(
                "step02_feature_engineering",
                result,
                step_start_time,
                bool(result),
                step_errors,
                step_warnings
            )

            return result

        except Exception as e:
            step_errors.append(str(e))
            self.logger.error(f"❌ Step 2 failed: {e}")

            # Generate step report even on failure
            await self._generate_step_report(
                "step02_feature_engineering",
                None,
                step_start_time,
                False,
                step_errors,
                step_warnings
            )
            raise

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step03_hmm_regime_discovery"
    )
    @monitor_pipeline_step(
        stage=PipelineStage.MODEL_TRAINING,
        validation_level=PipelineValidationLevel.STRICT,
        enable_data_quality=True
    )
    @monitor_pipeline_performance(
        enable_memory_tracking=True,
        enable_cpu_tracking=True,
        memory_threshold_gb=32.0,
        cpu_threshold_percent=95.0
    )
    async def _execute_step3_enhanced(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
        timeframe: str,
        lookback_days: int,
        force_rerun: bool,
    ) -> bool:
        """Execute Step 3: HMM Regime Discovery with comprehensive reporting."""

        step_start_time = time.time()
        step_errors = []
        step_warnings = []

        try:
            from src.training.steps import step03_hmm_regime_discovery as _step3

    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
import result = await _step3.run_step_enhanced
            result = await _step3.run_step_enhanced(
                symbol=symbol,
                exchange=exchange,
                data_dir=data_dir,
                timeframe=timeframe,
                lookback_days=lookback_days,
                force_rerun=force_rerun,
            )

            # Generate step report
            await self._generate_step_report(
                "step03_hmm_regime_discovery",
                result,
                step_start_time,
                bool(result),
                step_errors,
                step_warnings
            )

            return result

        except Exception as e:
            step_errors.append(str(e))
            self.logger.error(f"❌ Step 3 failed: {e}")

            # Generate step report even on failure
            await self._generate_step_report(
                "step03_hmm_regime_discovery",
                None,
                step_start_time,
                False,
                step_errors,
                step_warnings
            )
            raise

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step04_regime_data_splitting"
    )
    @monitor_pipeline_step(
        stage=PipelineStage.DATA_PREPROCESSING,
        validation_level=PipelineValidationLevel.WARNING,
        enable_data_quality=True
    )
    async def _execute_step4_enhanced(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        force_rerun: bool,
    ) -> bool:
        """Execute Step 4: Regime Data Splitting with enhanced reporting."""

        step_start_time = time.time()
        step_errors = []
        step_warnings = []

        try:
            from src.training.steps import step04_regime_data_splitting

    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
import result = await step04_regime_data_splitting.run_step
            result = await step04_regime_data_splitting.run_step(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                force_rerun=force_rerun,
                config=self.config,
            )

            # Generate step report
            await self._generate_step_report(
                "step04_regime_data_splitting",
                result,
                step_start_time,
                bool(result),
                step_errors,
                step_warnings
            )

            return result

        except Exception as e:
            step_errors.append(str(e))
            self.logger.error(f"❌ Step 4 failed: {e}")

            # Generate step report even on failure
            await self._generate_step_report(
                "step04_regime_data_splitting",
                None,
                step_start_time,
                False,
                step_errors,
                step_warnings
            )
            raise

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step05_triple_barrier_method"
    )
    @monitor_pipeline_step(
        stage=PipelineStage.DATA_PREPROCESSING,
        validation_level=PipelineValidationLevel.WARNING,
        enable_data_quality=True
    )
    async def _execute_step5_enhanced(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        force_rerun: bool,
    ) -> bool:
        """Execute Step 5: Triple Barrier Method with enhanced reporting."""

        step_start_time = time.time()
        step_errors = []
        step_warnings = []

        try:
            from src.training.steps import step05_triple_barrier_method

    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
import result = await step05_triple_barrier_method.run_step
            result = await step05_triple_barrier_method.run_step(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                force_rerun=force_rerun,
                config=self.config,
            )

            # Generate step report
            await self._generate_step_report(
                "step05_triple_barrier_method",
                result,
                step_start_time,
                bool(result),
                step_errors,
                step_warnings
            )

            return result

        except Exception as e:
            step_errors.append(str(e))
            self.logger.error(f"❌ Step 5 failed: {e}")

            # Generate step report even on failure
            await self._generate_step_report(
                "step05_triple_barrier_method",
                None,
                step_start_time,
                False,
                step_errors,
                step_warnings
            )
            raise

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step06_hmm_based_training"
    )
    @monitor_pipeline_step(
        stage=PipelineStage.MODEL_TRAINING,
        validation_level=PipelineValidationLevel.STRICT,
        enable_data_quality=True
    )
    @monitor_pipeline_performance(
        enable_memory_tracking=True,
        enable_cpu_tracking=True,
        memory_threshold_gb=32.0,
        cpu_threshold_percent=95.0
    )
    async def _execute_step6_enhanced(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        force_rerun: bool,
    ) -> bool:
        """Execute Step 6: HMM-Based Training with comprehensive reporting."""

        step_start_time = time.time()
        step_errors = []
        step_warnings = []

        try:
            from src.training.steps import step06_hmm_based_training

    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
import result = await step06_hmm_based_training.run_step
            result = await step06_hmm_based_training.run_step(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                force_rerun=force_rerun,
                config=self.config,
            )

            # Generate step report
            await self._generate_step_report(
                "step06_hmm_based_training",
                result,
                step_start_time,
                bool(result),
                step_errors,
                step_warnings
            )

            return result

        except Exception as e:
            step_errors.append(str(e))
            self.logger.error(f"❌ Step 6 failed: {e}")

            # Generate step report even on failure
            await self._generate_step_report(
                "step06_hmm_based_training",
                None,
                step_start_time,
                False,
                step_errors,
                step_warnings
            )
            raise

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step07_analyst_enhancement"
    )
    @monitor_pipeline_step(
        stage=PipelineStage.MODEL_TRAINING,
        validation_level=PipelineValidationLevel.WARNING,
        enable_data_quality=True
    )
    async def _execute_step7_enhanced(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        force_rerun: bool,
    ) -> bool:
        """Execute Step 7: Analyst Enhancement with enhanced reporting."""

        step_start_time = time.time()
        step_errors = []
        step_warnings = []

        try:
            from src.training.steps import step07_analyst_enhancement

    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
import result = await step07_analyst_enhancement.run_step
            result = await step07_analyst_enhancement.run_step(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                force_rerun=force_rerun,
                config=self.config,
            )

            # Generate step report
            await self._generate_step_report(
                "step07_analyst_enhancement",
                result,
                step_start_time,
                bool(result),
                step_errors,
                step_warnings
            )

            return result

        except Exception as e:
            step_errors.append(str(e))
            self.logger.error(f"❌ Step 7 failed: {e}")

            # Generate step report even on failure
            await self._generate_step_report(
                "step07_analyst_enhancement",
                None,
                step_start_time,
                False,
                step_errors,
                step_warnings
            )
            raise

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step08_tactician_labeling"
    )
    @monitor_pipeline_step(
        stage=PipelineStage.DATA_PREPROCESSING,
        validation_level=PipelineValidationLevel.WARNING,
        enable_data_quality=True
    )
    async def _execute_step8_enhanced(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        force_rerun: bool,
    ) -> bool:
        """Execute Step 8: Tactician Labeling with enhanced reporting."""

        step_start_time = time.time()
        step_errors = []
        step_warnings = []

        try:
            from src.training.steps import step08_tactician_labeling

    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
import result = await step08_tactician_labeling.run_step
            result = await step08_tactician_labeling.run_step(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                force_rerun=force_rerun,
                config=self.config,
            )

            # Generate step report
            await self._generate_step_report(
                "step08_tactician_labeling",
                result,
                step_start_time,
                bool(result),
                step_errors,
                step_warnings
            )

            return result

        except Exception as e:
            step_errors.append(str(e))
            self.logger.error(f"❌ Step 8 failed: {e}")

            # Generate step report even on failure
            await self._generate_step_report(
                "step08_tactician_labeling",
                None,
                step_start_time,
                False,
                step_errors,
                step_warnings
            )
            raise

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step09_tactician_specialist_training"
    )
    @monitor_pipeline_step(
        stage=PipelineStage.MODEL_TRAINING,
        validation_level=PipelineValidationLevel.STRICT,
        enable_data_quality=True
    )
    @monitor_pipeline_performance(
        enable_memory_tracking=True,
        enable_cpu_tracking=True,
        memory_threshold_gb=32.0,
        cpu_threshold_percent=95.0
    )
    async def _execute_step9_enhanced(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        force_rerun: bool,
    ) -> bool:
        """Execute Step 9: Tactician Specialist Training with comprehensive reporting."""

        step_start_time = time.time()
        step_errors = []
        step_warnings = []

        try:
            from src.training.steps import step09_tactician_specialist_training

    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
import result = await step09_tactician_specialist_training.run_step
            result = await step09_tactician_specialist_training.run_step(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                force_rerun=force_rerun,
                config=self.config,
            )

            # Generate step report
            await self._generate_step_report(
                "step09_tactician_specialist_training",
                result,
                step_start_time,
                bool(result),
                step_errors,
                step_warnings
            )

            return result

        except Exception as e:
            step_errors.append(str(e))
            self.logger.error(f"❌ Step 9 failed: {e}")

            # Generate step report even on failure
            await self._generate_step_report(
                "step09_tactician_specialist_training",
                None,
                step_start_time,
                False,
                step_errors,
                step_warnings
            )
            raise

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step10_confidence_calibration"
    )
    @monitor_pipeline_step(
        stage=PipelineStage.VALIDATION,
        validation_level=PipelineValidationLevel.WARNING,
        enable_data_quality=True
    )
    async def _execute_step10_enhanced(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        force_rerun: bool,
    ) -> bool:
        """Execute Step 10: Confidence Calibration with enhanced reporting."""

        step_start_time = time.time()
        step_errors = []
        step_warnings = []

        try:
            from src.training.steps import step10_confidence_calibration

    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
import result = await step10_confidence_calibration.run_step
            result = await step10_confidence_calibration.run_step(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                force_rerun=force_rerun,
                config=self.config,
            )

            # Generate step report
            await self._generate_step_report(
                "step10_confidence_calibration",
                result,
                step_start_time,
                bool(result),
                step_errors,
                step_warnings
            )

            return result

        except Exception as e:
            step_errors.append(str(e))
            self.logger.error(f"❌ Step 10 failed: {e}")

            # Generate step report even on failure
            await self._generate_step_report(
                "step10_confidence_calibration",
                None,
                step_start_time,
                False,
                step_errors,
                step_warnings
            )
            raise

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step11_final_parameters_optimization"
    )
    @monitor_pipeline_step(
        stage=PipelineStage.OPTIMIZATION,
        validation_level=PipelineValidationLevel.WARNING,
        enable_data_quality=True
    )
    async def _execute_step11_enhanced(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        force_rerun: bool,
    ) -> bool:
        """Execute Step 11: Final Parameters Optimization with enhanced reporting."""

        step_start_time = time.time()
        step_errors = []
        step_warnings = []

        try:
            from src.training.steps import step11_final_parameters_optimization

    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
import result = await step11_final_parameters_optimization.run_step
            result = await step11_final_parameters_optimization.run_step(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                force_rerun=force_rerun,
                config=self.config,
            )

            # Generate step report
            await self._generate_step_report(
                "step11_final_parameters_optimization",
                result,
                step_start_time,
                bool(result),
                step_errors,
                step_warnings
            )

            return result

        except Exception as e:
            step_errors.append(str(e))
            self.logger.error(f"❌ Step 11 failed: {e}")

            # Generate step report even on failure
            await self._generate_step_report(
                "step11_final_parameters_optimization",
                None,
                step_start_time,
                False,
                step_errors,
                step_warnings
            )
            raise

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step12_walk_forward_validation"
    )
    @monitor_pipeline_step(
        stage=PipelineStage.VALIDATION,
        validation_level=PipelineValidationLevel.STRICT,
        enable_data_quality=True
    )
    @monitor_pipeline_performance(
        enable_memory_tracking=True,
        enable_cpu_tracking=True,
        memory_threshold_gb=32.0,
        cpu_threshold_percent=95.0
    )
    async def _execute_step12_enhanced(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        force_rerun: bool,
    ) -> bool:
        """Execute Step 12: Walk Forward Validation with comprehensive reporting."""

        step_start_time = time.time()
        step_errors = []
        step_warnings = []

        try:
            from src.training.steps import step12_walk_forward_validation

    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
import result = await step12_walk_forward_validation.run_step
            result = await step12_walk_forward_validation.run_step(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                force_rerun=force_rerun,
                config=self.config,
            )

            # Generate step report
            await self._generate_step_report(
                "step12_walk_forward_validation",
                result,
                step_start_time,
                bool(result),
                step_errors,
                step_warnings
            )

            return result

        except Exception as e:
            step_errors.append(str(e))
            self.logger.error(f"❌ Step 12 failed: {e}")

            # Generate step report even on failure
            await self._generate_step_report(
                "step12_walk_forward_validation",
                None,
                step_start_time,
                False,
                step_errors,
                step_warnings
            )
            raise

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step13_monte_carlo_validation"
    )
    @monitor_pipeline_step(
        stage=PipelineStage.VALIDATION,
        validation_level=PipelineValidationLevel.STRICT,
        enable_data_quality=True
    )
    @monitor_pipeline_performance(
        enable_memory_tracking=True,
        enable_cpu_tracking=True,
        memory_threshold_gb=32.0,
        cpu_threshold_percent=95.0
    )
    async def _execute_step13_enhanced(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        force_rerun: bool,
    ) -> bool:
        """Execute Step 13: Monte Carlo Validation with comprehensive reporting."""

        step_start_time = time.time()
        step_errors = []
        step_warnings = []

        try:
            from src.training.steps import step13_monte_carlo_validation

    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
import result = await step13_monte_carlo_validation.run_step
            result = await step13_monte_carlo_validation.run_step(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                force_rerun=force_rerun,
                config=self.config,
            )

            # Generate step report
            await self._generate_step_report(
                "step13_monte_carlo_validation",
                result,
                step_start_time,
                bool(result),
                step_errors,
                step_warnings
            )

            return result

        except Exception as e:
            step_errors.append(str(e))
            self.logger.error(f"❌ Step 13 failed: {e}")

            # Generate step report even on failure
            await self._generate_step_report(
                "step13_monte_carlo_validation",
                None,
                step_start_time,
                False,
                step_errors,
                step_warnings
            )
            raise

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step14_ab_testing"
    )
    @monitor_pipeline_step(
        stage=PipelineStage.VALIDATION,
        validation_level=PipelineValidationLevel.WARNING,
        enable_data_quality=True
    )
    async def _execute_step14_enhanced(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        force_rerun: bool,
    ) -> bool:
        """Execute Step 14: A/B Testing with enhanced reporting."""

        step_start_time = time.time()
        step_errors = []
        step_warnings = []

        try:
            from src.training.steps import step14_ab_testing

    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
import result = await step14_ab_testing.run_step
            result = await step14_ab_testing.run_step(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                force_rerun=force_rerun,
                config=self.config,
            )

            # Generate step report
            await self._generate_step_report(
                "step14_ab_testing",
                result,
                step_start_time,
                bool(result),
                step_errors,
                step_warnings
            )

            return result

        except Exception as e:
            step_errors.append(str(e))
            self.logger.error(f"❌ Step 14 failed: {e}")

            # Generate step report even on failure
            await self._generate_step_report(
                "step14_ab_testing",
                None,
                step_start_time,
                False,
                step_errors,
                step_warnings
            )
            raise

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step15_saving"
    )
    @monitor_pipeline_step(
        stage=PipelineStage.DEPLOYMENT,
        validation_level=PipelineValidationLevel.WARNING,
        enable_data_quality=True
    )
    async def _execute_step15_enhanced(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        force_rerun: bool,
    ) -> bool:
        """Execute Step 15: Saving Results with enhanced reporting."""

        step_start_time = time.time()
        step_errors = []
        step_warnings = []

        try:
            from src.training.steps import step15_saving

    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
import result = await step15_saving.run_step
            result = await step15_saving.run_step(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                force_rerun=force_rerun,
                config=self.config,
            )

            # Generate step report
            await self._generate_step_report(
                "step15_saving",
                result,
                step_start_time,
                bool(result),
                step_errors,
                step_warnings
            )

            return result

        except Exception as e:
            step_errors.append(str(e))
            self.logger.error(f"❌ Step 15 failed: {e}")

            # Generate step report even on failure
            await self._generate_step_report(
                "step15_saving",
                None,
                step_start_time,
                False,
                step_errors,
                step_warnings
            )
            raise

    async def _generate_pipeline_report(self, pipeline_report: Dict[str, Any]):
        """Generate and store the comprehensive pipeline report."""

        try:
            # Generate report filename
    except Exception as e:
        pass
    except Exception as e:
        pass
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            symbol = pipeline_report.get("training_input", {}).get("symbol", "unknown")
            exchange = pipeline_report.get("training_input", {}).get("exchange", "unknown")

            filename = f"pipeline_{symbol}_{exchange}_{timestamp}.json"
            report_path = self.pipeline_reports_dir / filename

            # Save detailed JSON report
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(pipeline_report, f, indent=2, ensure_ascii=False, default=str)

            # Generate summary report
            summary_report = self._generate_pipeline_summary(pipeline_report)
            summary_filename = f"pipeline_{symbol}_{exchange}_{timestamp}_summary.txt"
            summary_path = self.pipeline_reports_dir / summary_filename

            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary_report)

            self.logger.info(f"📊 Pipeline report saved to {report_path}")
            self.logger.info(f"📋 Pipeline summary saved to {summary_path}")

        except Exception as e:
            self.logger.error(f"❌ Failed to generate pipeline report: {e}")

    def _generate_pipeline_summary(self, pipeline_report: Dict[str, Any]) -> str:
    pass
    pass
        """Generate a human-readable pipeline summary report."""

        summary = []
        summary.append("=" * 100)
        summary.append("ENHANCED TRAINING PIPELINE COMPREHENSIVE REPORT")
        summary.append("=" * 100)

        # Pipeline overview
        training_input = pipeline_report.get("training_input", {})
        summary.append(f"Pipeline Overview:")
        summary.append(f"  Execution ID: {pipeline_report.get('pipeline_execution_id', 'N/A')}")
        summary.append(f"  Symbol: {training_input.get('symbol', 'N/A')}")
        summary.append(f"  Exchange: {training_input.get('exchange', 'N/A')}")
        summary.append(f"  Timeframe: {training_input.get('timeframe', 'N/A')}")
        summary.append(f"  Start Time: {pipeline_report.get('pipeline_start_time', 'N/A')}")
        summary.append(f"  End Time: {pipeline_report.get('pipeline_end_time', 'N/A')}")
        summary.append(f"  Overall Success: {pipeline_report.get('overall_success', 'N/A')}")
        summary.append("")

        # Step summary
        steps = pipeline_report.get("steps", {})
        if steps:
    pass
    pass
            summary.append("Step Execution Summary:")
            summary.append("-" * 50)

            step_statuses = {
                "success": [],
                "failed": [],
                "skipped": []
            }

            for step_name, step_report in steps.items():
    pass
    pass
                status = step_report.get("success", False)
                if status:
    pass
    pass
                    step_statuses["success"].append(step_name)
                else:
                    step_statuses["failed"].append(step_name)

            summary.append(f"✅ Successful Steps ({len(step_statuses['success'])}):")
            for step in step_statuses["success"]:
    pass
    pass
                duration = steps[step].get("execution_duration_formatted", "N/A")
                summary.append(f"  - {step}: {duration}")

            if step_statuses["failed"]:
    pass
    pass
                summary.append(f"❌ Failed Steps ({len(step_statuses['failed'])}):")
                for step in step_statuses["failed"]:
    pass
    pass
                    summary.append(f"  - {step}")

            summary.append("")

        # Errors and warnings
        if pipeline_report.get("errors"):
    pass
    pass
            summary.append("Pipeline Errors:")
            summary.append("-" * 50)
            for error in pipeline_report["errors"]:
    pass
    pass
                summary.append(f"❌ {error.get('type', 'Unknown')}: {error.get('message', 'No message')}")
            summary.append("")

        if pipeline_report.get("warnings"):
    pass
    pass
            summary.append("Pipeline Warnings:")
            summary.append("-" * 50)
            for warning in pipeline_report["warnings"]:
    pass
    pass
                summary.append(f"⚠️ {warning}")
            summary.append("")

        # Recommendations
        if pipeline_report.get("recommendations"):
    pass
    pass
            summary.append("Recommendations:")
            summary.append("-" * 50)
            for rec in pipeline_report["recommendations"]:
    pass
    pass
                summary.append(f"💡 {rec}")
            summary.append("")

        summary.append("=" * 100)
        summary.append("End of Pipeline Report")
        summary.append("=" * 100)

        return "\\\n".join(summary)

    async def _get_step_quality_metrics(self, step_name: str, step_result: Any) -> Dict[str, Any]:
        """Get step-specific quality metrics and validation information."""

        try:
            if step_name == "step01_data_collection":
    pass
    except Exception as e:
        pass
    pass
                return await self._get_data_collection_metrics(step_result)
    except Exception as e:
        pass
            elif step_name == "step01_5_data_converter":
                return await self._get_data_converter_metrics(step_result)
            elif step_name == "step02_feature_engineering":
                return await self._get_feature_engineering_metrics(step_result)
            elif step_name == "step03_hmm_regime_discovery":
                return await self._get_hmm_regime_metrics(step_result)
            elif step_name == "step04_regime_data_splitting":
                return await self._get_regime_splitting_metrics(step_result)
            elif step_name == "step05_triple_barrier_method":
                return await self._get_triple_barrier_metrics(step_result)
            elif step_name == "step06_feature_generation":
                return await self._get_feature_generation_metrics(step_result)
            elif step_name == "step07_matrix_feature_selection":
                return await self._get_matrix_feature_selection_metrics(step_result)
            elif step_name == "step08_tactician_labeling":
                return await self._get_tactician_labeling_metrics(step_result)
            elif step_name == "step09_tactician_specialist_training":
                return await self._get_tactician_training_metrics(step_result)
            elif step_name == "step10_confidence_calibration":
                return await self._get_confidence_calibration_metrics(step_result)
            elif step_name == "step11_final_parameters_optimization":
                return await self._get_optimization_metrics(step_result)
            elif step_name == "step12_walk_forward_validation":
                return await self._get_walk_forward_metrics(step_result)
            elif step_name == "step13_monte_carlo_validation":
                return await self._get_monte_carlo_metrics(step_result)
            elif step_name == "step14_ab_testing":
                return await self._get_ab_testing_metrics(step_result)
            elif step_name == "step15_saving":
                return await self._get_saving_metrics(step_result)
            else:
                return {"error": f"Unknown step: {step_name}"}

        except Exception as e:
            return {"error": f"Failed to get quality metrics: {str(e)}"}

    async def _get_data_collection_metrics(self, result: Any) -> Dict[str, Any]:
        """Get data collection quality metrics."""

        try:
            import pandas as pd

    except Exception as e:
        pass
    except Exception as e:
        pass
            if isinstance(result, pd.DataFrame) and not result.empty:
    pass
    pass
                return {
                    "data_quality": {
                        "total_rows": len(result),
                        "total_columns": len(result.columns),
                        "null_counts": result.isnull().sum().to_dict(),
                        "null_percentage": (result.isnull().sum() / len(result) * 100).to_dict(),
                        "duplicate_rows": result.duplicated().sum(),
                        "duplicate_percentage": (result.duplicated().sum() / len(result) * 100),
                        "data_types": result.dtypes.to_dict(),
                        "memory_usage_mb": result.memory_usage(deep=True).sum() / (1024**2),
                        "date_range": {
                            "start": str(result.index.min()) if hasattr(result.index, 'min') else None,
                            "end": str(result.index.max()) if hasattr(result.index, 'max') else None
                        }
                    },
                    "data_validation": {
                        "has_required_columns": all(col in result.columns for col in ['open', 'high', 'low', 'close', 'volume']),
                        "price_consistency": self._check_price_consistency(result),
                        "volume_consistency": self._check_volume_consistency(result),
                        "timestamp_consistency": self._check_timestamp_consistency(result)
                    },
                    "warnings": self._generate_data_collection_warnings(result)
                }
            else:
                return {"error": "No DataFrame result available"}

        except Exception as e:
            return {"error": f"Failed to analyze data collection metrics: {str(e)}"}

    async def _get_data_converter_metrics(self, result: Any) -> Dict[str, Any]:
        """Get data converter quality metrics."""

        try:
            import pandas as pd

    except Exception as e:
        pass
    except Exception as e:
        pass
            if isinstance(result, pd.DataFrame) and not result.empty:
    pass
    pass
                return {
                    "conversion_quality": {
                        "total_rows": len(result),
                        "total_columns": len(result.columns),
                        "converted_columns": list(result.columns),
                        "null_counts": result.isnull().sum().to_dict(),
                        "null_percentage": (result.isnull().sum() / len(result) * 100).to_dict(),
                        "data_types": result.dtypes.to_dict(),
                        "memory_usage_mb": result.memory_usage(deep=True).sum() / (1024**2)
                    },
                    "format_validation": {
                        "has_ohlcv": all(col in result.columns for col in ['open', 'high', 'low', 'close', 'volume']),
                        "has_timestamp": 'timestamp' in result.columns or result.index.name == 'timestamp',
                        "numeric_columns": result.select_dtypes(include=['number']).columns.tolist(),
                        "datetime_columns": result.select_dtypes(include=['datetime']).columns.tolist()
                    },
                    "warnings": self._generate_data_converter_warnings(result)
                }
            else:
                return {"error": "No DataFrame result available"}

        except Exception as e:
            return {"error": f"Failed to analyze data converter metrics: {str(e)}"}

    async def _get_feature_engineering_metrics(self, result: Any) -> Dict[str, Any]:
        """Get feature engineering quality metrics."""

        try:
            import pandas as pd
    except Exception as e:
        pass
    except Exception as e:
        pass
            import numpy as np

            if isinstance(result, pd.DataFrame) and not result.empty:
    pass
    pass
                # Calculate multicollinearity
                numeric_cols = result.select_dtypes(include=[np.number]).columns
                correlation_matrix = result[numeric_cols].corr()
                high_correlation_pairs = []

                for i in range(len(correlation_matrix.columns)):
    pass
    pass
                    for j in range(i+1, len(correlation_matrix.columns)):
    pass
    pass
                        corr_value = correlation_matrix.iloc[i, j]
                        if abs(corr_value) > 0.95:  # High correlation threshold
                            high_correlation_pairs.append({
                                "feature1": correlation_matrix.columns[i],
                                "feature2": correlation_matrix.columns[j],
                                "correlation": corr_value
                            })

                # Calculate VIF for multicollinearity
                vif_scores = {}
                try:
                    from statsmodels.stats.outliers_influence import variance_inflation_factor
    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
import for col in numeric_cols:
                    for col in numeric_cols:
    pass
    pass
                        if len(numeric_cols) > 1:
    pass
    pass
                            other_cols = [c for c in numeric_cols if c != col]
                            if len(other_cols) > 0:
    pass
    pass
                                vif_scores[col] = variance_inflation_factor(result[other_cols + [col]], len(other_cols))
                except ImportError:
                    vif_scores = {"error": "statsmodels not available for VIF calculation"}

                return {
                    "feature_quality": {
                        "total_features": len(result.columns),
                        "numeric_features": len(numeric_cols),
                        "categorical_features": len(result.select_dtypes(include=['object', 'category']).columns),
                        "null_counts": result.isnull().sum().to_dict(),
                        "null_percentage": (result.isnull().sum() / len(result) * 100).to_dict(),
                        "memory_usage_mb": result.memory_usage(deep=True).sum() / (1024**2)
                    },
                    "multicollinearity_analysis": {
                        "high_correlation_pairs": high_correlation_pairs,
                        "high_correlation_count": len(high_correlation_pairs),
                        "vif_scores": vif_scores,
                        "high_vif_features": [col for col, vif in vif_scores.items() if isinstance(vif, (int, float)) and vif > 10]
                    },
                    "feature_statistics": {
                        "constant_features": result.columns[result.nunique() == 1].tolist(),
                        "low_variance_features": result.columns[result.var() < 0.01].tolist(),
                        "high_cardinality_features": result.columns[result.nunique() > len(result) * 0.5].tolist()
                    },
                    "data_quality_issues": {
                        "nan_features": result.columns[result.isnull().any()].tolist(),
                        "inf_features": result.columns[np.isinf(result.select_dtypes(include=[np.number])).any()].tolist(),
                        "zero_variance_features": result.columns[result.var() == 0].tolist()
                    },
                    "warnings": self._generate_feature_engineering_warnings(result, high_correlation_pairs, vif_scores)
                }
            else:
                return {"error": "No DataFrame result available"}

        except Exception as e:
            return {"error": f"Failed to analyze feature engineering metrics: {str(e)}"}

    async def _get_hmm_regime_metrics(self, result: Any) -> Dict[str, Any]:
        """Get HMM regime discovery quality metrics."""

        try:
            if isinstance(result, dict):
    pass
    except Exception as e:
        pass
    pass
                return {
                    "regime_analysis": {
                        "number_of_regimes": result.get("n_regimes", "Unknown"),
                        "regime_transitions": result.get("transition_matrix", "Unknown"),
                        "regime_probabilities": result.get("regime_probs", "Unknown"),
                        "convergence_status": result.get("converged", "Unknown"),
                        "log_likelihood": result.get("log_likelihood", "Unknown")
                    },
                    "regime_quality": {
                        "regime_separation": self._calculate_regime_separation(result),
                        "regime_stability": self._calculate_regime_stability(result),
                        "regime_duration": self._calculate_regime_duration(result)
                    },
                    "validation_metrics": {
                        "aic_score": result.get("aic", "Unknown"),
                        "bic_score": result.get("bic", "Unknown"),
                        "model_complexity": result.get("n_parameters", "Unknown")
                    }
                }
    except Exception as e:
        pass
            else:
                return {"error": "No HMM result available"}

        except Exception as e:
            return {"error": f"Failed to analyze HMM regime metrics: {str(e)}"}

    # Helper methods for quality checks
    def _check_price_consistency(self, df) -> Dict[str, Any]:
    pass
    pass
        """Check price data consistency."""
        try:
            issues = []
    except Exception as e:
        pass
    except Exception as e:
        pass
            if 'high' in df.columns and 'low' in df.columns:
    pass
    pass
                invalid_high_low = (df['high'] < df['low']).sum()
                if invalid_high_low > 0:
    pass
    pass
                    issues.append(f"High < Low: {invalid_high_low} rows")

            if 'open' in df.columns and 'close' in df.columns:
    pass
    pass
                zero_prices = ((df['open'] == 0) | (df['close'] == 0)).sum()
                if zero_prices > 0:
    pass
    pass
                    issues.append(f"Zero prices: {zero_prices} rows")

            return {
                "has_issues": len(issues) > 0,
                "issues": issues
            }
        except Exception:
            return {"error": "Could not check price consistency"}

    def _check_volume_consistency(self, df) -> Dict[str, Any]:
    pass
    pass
        """Check volume data consistency."""
        try:
            issues = []
    except Exception as e:
        pass
    except Exception as e:
        pass
            if 'volume' in df.columns:
    pass
    pass
                negative_volume = (df['volume'] < 0).sum()
                if negative_volume > 0:
    pass
    pass
                    issues.append(f"Negative volume: {negative_volume} rows")

                zero_volume = (df['volume'] == 0).sum()
                if zero_volume > 0:
    pass
    pass
                    issues.append(f"Zero volume: {zero_volume} rows")

            return {
                "has_issues": len(issues) > 0,
                "issues": issues
            }
        except Exception:
            return {"error": "Could not check volume consistency"}

    def _check_timestamp_consistency(self, df) -> Dict[str, Any]:
    pass
    pass
        """Check timestamp consistency."""
        try:
            issues = []
    except Exception as e:
        pass
    except Exception as e:
        pass
            if hasattr(df.index, 'is_monotonic_increasing'):
    pass
    pass
                if not df.index.is_monotonic_increasing:
    pass
    pass
                    issues.append("Timestamps not in ascending order")

            if hasattr(df.index, 'duplicated'):
    pass
    pass
                duplicates = df.index.duplicated().sum()
                if duplicates > 0:
    pass
    pass
                    issues.append(f"Duplicate timestamps: {duplicates}")

            return {
                "has_issues": len(issues) > 0,
                "issues": issues
            }
        except Exception:
            return {"error": "Could not check timestamp consistency"}

    def _generate_data_collection_warnings(self, df) -> List[str]:
    pass
    pass
        """Generate warnings for data collection."""
        warnings = []

        try:
            if df.isnull().any().any():
    pass
    except Exception as e:
        pass
    pass
                null_percentage = (df.isnull().sum() / len(df) * 100).max()
                if null_percentage > 10:
    pass
    pass
                    warnings.append(f"High null percentage: {null_percentage:.2f}%")

    except Exception as e:
        pass
            if len(df) < 1000:
    pass
    pass
                warnings.append(f"Low data volume: {len(df)} rows")

            if 'volume' in df.columns and (df['volume'] == 0).sum() > len(df) * 0.5:
    pass
    pass
                warnings.append("High percentage of zero volume data")

        except Exception:
            warnings.append("Could not generate data collection warnings")

        return warnings

    def _generate_data_converter_warnings(self, df) -> List[str]:
    pass
    pass
        """Generate warnings for data converter."""
        warnings = []

        try:
            if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
    pass
    except Exception as e:
        pass
    pass
                warnings.append("Missing required OHLCV columns")

    except Exception as e:
        pass
            if df.isnull().any().any():
    pass
    pass
                warnings.append("Data contains null values after conversion")

        except Exception:
            warnings.append("Could not generate data converter warnings")

        return warnings

    def _generate_feature_engineering_warnings(self, df, high_correlation_pairs, vif_scores) -> List[str]:
    pass
    pass
        """Generate warnings for feature engineering."""
        warnings = []

        try:
            if len(high_correlation_pairs) > 10:
    pass
    except Exception as e:
        pass
    pass
                warnings.append(f"High multicollinearity: {len(high_correlation_pairs)} highly correlated feature pairs")

    except Exception as e:
        pass
            high_vif_features = [col for col, vif in vif_scores.items() if isinstance(vif, (int, float)) and vif > 10]
            if high_vif_features:
    pass
    pass
                warnings.append(f"High VIF features: {high_vif_features}")

            if df.isnull().any().any():
    pass
    pass
                warnings.append("Features contain null values")

        except Exception:
            warnings.append("Could not generate feature engineering warnings")

        return warnings

    async def _get_regime_splitting_metrics(self, result: Any) -> Dict[str, Any]:
        """Get regime data splitting quality metrics."""

        try:
            if isinstance(result, dict):
    pass
    except Exception as e:
        pass
    pass
                return {
                    "splitting_analysis": {
                        "total_regimes": result.get("n_regimes", "Unknown"),
                        "regime_distributions": result.get("regime_counts", "Unknown"),
                        "train_test_split_ratio": result.get("split_ratio", "Unknown"),
                        "validation_split_ratio": result.get("val_split_ratio", "Unknown"),
                        "stratified_splitting": result.get("stratified", "Unknown")
                    },
                    "data_distribution": {
                        "total_samples": result.get("total_samples", "Unknown"),
                        "train_samples": result.get("train_samples", "Unknown"),
                        "test_samples": result.get("test_samples", "Unknown"),
                        "validation_samples": result.get("val_samples", "Unknown"),
                        "regime_balance": self._calculate_regime_balance(result)
                    },
                    "time_distribution": {
                        "regime_time_periods": result.get("regime_time_periods", "Unknown"),
                        "regime_duration_stats": self._calculate_regime_duration_stats(result),
                        "regime_transition_frequency": result.get("regime_transitions", "Unknown"),
                        "temporal_regime_distribution": self._analyze_temporal_regime_distribution(result)
                    },
                    "quality_validation": {
                        "no_data_leakage": result.get("no_leakage", "Unknown"),
                        "regime_representation": self._validate_regime_representation(result),
                        "temporal_consistency": result.get("temporal_consistent", "Unknown")
                    },
                    "warnings": self._generate_regime_splitting_warnings(result)
                }
    except Exception as e:
        pass
            else:
                return {"error": "No regime splitting result available"}

        except Exception as e:
            return {"error": f"Failed to analyze regime splitting metrics: {str(e)}"}

    async def _get_triple_barrier_metrics(self, result: Any) -> Dict[str, Any]:
        """Get triple barrier method quality metrics."""

        try:
            if isinstance(result, dict):
    pass
    except Exception as e:
        pass
    pass
                return {
                    "barrier_analysis": {
                        "total_labels": result.get("total_labels", "Unknown"),
                        "label_distribution": result.get("label_counts", "Unknown"),
                        "barrier_parameters": {
                            "upper_barrier": result.get("upper_barrier", "Unknown"),
                            "lower_barrier": result.get("lower_barrier", "Unknown"),
                            "time_horizon": result.get("time_horizon", "Unknown")
                        }
                    },
                    "daily_statistics": {
                        "average_barriers_per_day": result.get("avg_barriers_per_day", "Unknown"),
                        "total_trading_days": result.get("total_trading_days", "Unknown"),
                        "days_with_barriers": result.get("days_with_barriers", "Unknown"),
                        "barrier_density": result.get("barrier_density", "Unknown"),
                        "daily_barrier_counts": result.get("daily_barrier_counts", "Unknown")
                    },
                    "barrier_values": {
                        "upper_barrier_value": result.get("upper_barrier_value", "Unknown"),
                        "lower_barrier_value": result.get("lower_barrier_value", "Unknown"),
                        "barrier_spread": result.get("barrier_spread", "Unknown"),
                        "barrier_volatility": result.get("barrier_volatility", "Unknown"),
                        "barrier_distribution": result.get("barrier_distribution", "Unknown")
                    },
                    "position_ratios": {
                        "long_short_ratio": result.get("long_short_ratio", "Unknown"),
                        "long_positions": result.get("long_positions", "Unknown"),
                        "short_positions": result.get("short_positions", "Unknown"),
                        "hold_positions": result.get("hold_positions", "Unknown"),
                        "position_distribution": result.get("position_distribution", "Unknown")
                    },
                    "triple_barrier_captured_changes": {
                        "upper_barrier_captures": result.get("upper_barrier_captures", "Unknown"),
                        "lower_barrier_captures": result.get("lower_barrier_captures", "Unknown"),
                        "captured_price_changes": self._analyze_triple_barrier_captured_changes(result),
                        "capture_efficiency": result.get("capture_efficiency", "Unknown"),
                        "capture_distribution": result.get("capture_distribution", "Unknown")
                    },
                    "label_quality": {
                        "balanced_labels": self._check_label_balance(result),
                        "label_consistency": result.get("label_consistent", "Unknown"),
                        "no_label_leakage": result.get("no_label_leakage", "Unknown"),
                        "label_validation": self._validate_triple_barrier_labels(result)
                    },
                    "performance_metrics": {
                        "label_generation_time": result.get("generation_time", "Unknown"),
                        "memory_usage": result.get("memory_usage", "Unknown"),
                        "efficiency_score": result.get("efficiency", "Unknown")
                    },
                    "warnings": self._generate_triple_barrier_warnings(result)
                }
    except Exception as e:
        pass
            else:
                return {"error": "No triple barrier result available"}

        except Exception as e:
            return {"error": f"Failed to analyze triple barrier metrics: {str(e)}"}

    async def _get_feature_generation_metrics(self, result: Any) -> Dict[str, Any]:
        """Get feature generation quality metrics (Step 6)."""

        try:
            if isinstance(result, dict):
    pass
    except Exception as e:
        pass
    pass
                return {
                    "feature_generation_analysis": {
                        "generation_type": result.get("generation_type", "Unknown"),
                        "original_features": result.get("original_feature_count", "Unknown"),
                        "generated_features": result.get("generated_feature_count", "Unknown"),
                        "feature_increase": result.get("feature_increase", "Unknown"),
                        "generation_methods": result.get("generation_methods", "Unknown")
                    },
                    "feature_quality": {
                        "feature_relevance": result.get("feature_relevance_score", "Unknown"),
                        "information_gain": result.get("information_gain", "Unknown"),
                        "feature_diversity": result.get("feature_diversity", "Unknown"),
                        "feature_stability": result.get("feature_stability", "Unknown")
                    },
                    "generation_performance": {
                        "generation_time": result.get("generation_time", "Unknown"),
                        "memory_usage": result.get("memory_usage", "Unknown"),
                        "computational_efficiency": result.get("efficiency_score", "Unknown"),
                        "parallel_processing": result.get("parallel_processing", "Unknown")
                    },
                    "feature_statistics": {
                        "feature_types": result.get("feature_types", "Unknown"),
                        "feature_complexity": result.get("feature_complexity", "Unknown"),
                        "feature_correlations": result.get("feature_correlations", "Unknown"),
                        "feature_redundancy": result.get("feature_redundancy", "Unknown")
                    },
                    "warnings": self._generate_feature_generation_warnings(result)
                }
    except Exception as e:
        pass
            else:
                return {"error": "No feature generation result available"}

        except Exception as e:
            return {"error": f"Failed to analyze feature generation metrics: {str(e)}"}

    async def _get_matrix_feature_selection_metrics(self, result: Any) -> Dict[str, Any]:
        """Get matrix feature selection quality metrics (Step 7)."""

        try:
            if isinstance(result, dict):
    pass
    except Exception as e:
        pass
    pass
                return {
                    "selection_analysis": {
                        "selection_method": result.get("selection_method", "Unknown"),
                        "original_features": result.get("original_feature_count", "Unknown"),
                        "selected_features": result.get("selected_feature_count", "Unknown"),
                        "reduction_ratio": result.get("reduction_ratio", "Unknown"),
                        "selection_criteria": result.get("selection_criteria", "Unknown")
                    },
                    "selection_quality": {
                        "feature_importance": result.get("feature_importance", "Unknown"),
                        "information_preservation": result.get("information_preservation", "Unknown"),
                        "selection_stability": result.get("selection_stability", "Unknown"),
                        "cross_validation_score": result.get("cv_score", "Unknown")
                    },
                    "matrix_analysis": {
                        "correlation_matrix": result.get("correlation_matrix_stats", "Unknown"),
                        "variance_explained": result.get("variance_explained", "Unknown"),
                        "eigenvalue_distribution": result.get("eigenvalue_dist", "Unknown"),
                        "multicollinearity_reduction": result.get("multicollinearity_reduction", "Unknown")
                    },
                    "performance_impact": {
                        "pre_selection_accuracy": result.get("pre_selection_accuracy", "Unknown"),
                        "post_selection_accuracy": result.get("post_selection_accuracy", "Unknown"),
                        "accuracy_change": result.get("accuracy_change", "Unknown"),
                        "computational_savings": result.get("computational_savings", "Unknown")
                    },
                    "selected_features_analysis": {
                        "top_features": result.get("top_features", "Unknown"),
                        "feature_categories": result.get("feature_categories", "Unknown"),
                        "feature_rankings": result.get("feature_rankings", "Unknown"),
                        "selection_confidence": result.get("selection_confidence", "Unknown")
                    },
                    "warnings": self._generate_matrix_selection_warnings(result)
                }
    except Exception as e:
        pass
            else:
                return {"error": "No matrix feature selection result available"}

        except Exception as e:
            return {"error": f"Failed to analyze matrix feature selection metrics: {str(e)}"}

    # Helper methods for step-specific analysis
    def _calculate_regime_balance(self, result: Any) -> Dict[str, Any]:
    pass
    pass
        """Calculate regime balance in the split data."""
        try:
            regime_counts = result.get("regime_counts", {})
    except Exception as e:
        pass
    except Exception as e:
        pass
            if regime_counts:
    pass
    pass
                total = sum(regime_counts.values())
                balance_scores = {regime: count/total for regime, count in regime_counts.items()}
                return {
                    "regime_balance_scores": balance_scores,
                    "is_balanced": all(0.1 <= score <= 0.9 for score in balance_scores.values()),
                    "imbalance_score": max(balance_scores.values()) - min(balance_scores.values())
                }
            return {"error": "No regime counts available"}
        except Exception:
            return {"error": "Could not calculate regime balance"}

    def _validate_regime_representation(self, result: Any) -> Dict[str, Any]:
    pass
    pass
        """Validate regime representation across splits."""
        try:
            train_regimes = result.get("train_regime_counts", {})
    except Exception as e:
        pass
    except Exception as e:
        pass
            test_regimes = result.get("test_regime_counts", {})

            if train_regimes and test_regimes:
    pass
    pass
                missing_in_test = set(train_regimes.keys()) - set(test_regimes.keys())
                return {
                    "all_regimes_represented": len(missing_in_test) == 0,
                    "missing_regimes_in_test": list(missing_in_test),
                    "representation_consistency": "Good" if len(missing_in_test) == 0 else "Poor"
                }
            return {"error": "No regime counts available"}
        except Exception:
            return {"error": "Could not validate regime representation"}

    def _check_label_balance(self, result: Any) -> Dict[str, Any]:
    pass
    pass
        """Check label balance in triple barrier method."""
        try:
            label_counts = result.get("label_counts", {})
    except Exception as e:
        pass
    except Exception as e:
        pass
            if label_counts:
    pass
    pass
                total = sum(label_counts.values())
                balance_scores = {label: count/total for label, count in label_counts.items()}
                return {
                    "label_distribution": balance_scores,
                    "is_balanced": all(0.2 <= score <= 0.8 for score in balance_scores.values()),
                    "majority_class": max(label_counts, key=label_counts.get),
                    "minority_class": min(label_counts, key=label_counts.get)
                }
            return {"error": "No label counts available"}
        except Exception:
            return {"error": "Could not check label balance"}

    def _validate_triple_barrier_labels(self, result: Any) -> Dict[str, Any]:
    pass
    pass
        """Validate triple barrier labels."""
        try:
            return {
                "no_future_leakage": result.get("no_future_leakage", "Unknown"),
                "barrier_constraints_satisfied": result.get("constraints_satisfied", "Unknown"),
                "label_consistency": result.get("label_consistency", "Unknown"),
                "temporal_validity": result.get("temporal_valid", "Unknown")
    except Exception as e:
        pass
    except Exception as e:
        pass
            }
        except Exception:
            return {"error": "Could not validate triple barrier labels"}

    def _calculate_overfitting_score(self, result: Any) -> Dict[str, Any]:
    pass
    pass
        """Calculate overfitting score."""
        try:
            train_acc = result.get("train_accuracy", 0)
    except Exception as e:
        pass
    except Exception as e:
        pass
            val_acc = result.get("val_accuracy", 0)

            if train_acc > 0 and val_acc > 0:
    pass
    pass
                overfitting_gap = train_acc - val_acc
                return {
                    "overfitting_gap": overfitting_gap,
                    "overfitting_severity": "High" if overfitting_gap > 0.1 else "Medium" if overfitting_gap > 0.05 else "Low",
                    "is_overfitting": overfitting_gap > 0.05
                }
            return {"error": "No accuracy metrics available"}
        except Exception:
            return {"error": "Could not calculate overfitting score"}

    def _assess_model_stability(self, result: Any) -> Dict[str, Any]:
    pass
    pass
        """Assess model stability."""
        try:
            return {
                "convergence_stability": result.get("convergence_stable", "Unknown"),
                "loss_stability": result.get("loss_stable", "Unknown"),
                "parameter_stability": result.get("param_stable", "Unknown"),
                "regime_stability": result.get("regime_stable", "Unknown")
    except Exception as e:
        pass
    except Exception as e:
        pass
            }
        except Exception:
            return {"error": "Could not assess model stability"}

    def _calculate_regime_duration_stats(self, result: Any) -> Dict[str, Any]:
    pass
    pass
        """Calculate regime duration statistics."""
        try:
            regime_durations = result.get("regime_durations", {})
    except Exception as e:
        pass
    except Exception as e:
        pass
            if regime_durations:
    pass
    pass
                stats = {}
                for regime, durations in regime_durations.items():
    pass
    pass
                    if durations:
    pass
    pass
                        stats[regime] = {
                            "mean_duration": sum(durations) / len(durations),
                            "min_duration": min(durations),
                            "max_duration": max(durations),
                            "std_duration": (sum((x - sum(durations)/len(durations))**2 for x in durations) / len(durations))**0.5,
                            "total_periods": len(durations)
                        }
                return stats
            return {"error": "No regime durations available"}
        except Exception:
            return {"error": "Could not calculate regime duration stats"}

    def _analyze_temporal_regime_distribution(self, result: Any) -> Dict[str, Any]:
    pass
    pass
        """Analyze temporal distribution of regimes."""
        try:
            temporal_data = result.get("temporal_regime_data", {})
    except Exception as e:
        pass
    except Exception as e:
        pass
            if temporal_data:
    pass
    pass
                return {
                    "regime_time_periods": temporal_data.get("time_periods", "Unknown"),
                    "regime_seasonality": temporal_data.get("seasonality", "Unknown"),
                    "regime_trends": temporal_data.get("trends", "Unknown"),
                    "regime_volatility": temporal_data.get("volatility", "Unknown")
                }
            return {"error": "No temporal regime data available"}
        except Exception:
            return {"error": "Could not analyze temporal regime distribution"}

    def _analyze_triple_barrier_captured_changes(self, result: Any) -> Dict[str, Any]:
    pass
    pass
        """Analyze price changes specifically captured by triple barrier method."""
        try:
            # Get the triple barrier results
    except Exception as e:
        pass
    except Exception as e:
        pass
            barrier_results = result.get("triple_barrier_results", {})
            if not barrier_results:
    pass
    pass
                return {"error": "No triple barrier results available"}

            # Extract barrier hit information
            barrier_hits = barrier_results.get("barrier_hits", [])
            price_movements = barrier_results.get("price_movements", [])

            # Analyze upper barrier hits without lower barrier hits first
            upper_hits_without_lower_first = []
            upper_hits_with_lower_first = []

            # Analyze lower barrier hits without upper barrier hits first
            lower_hits_without_upper_first = []
            lower_hits_with_upper_first = []

            # Analyze price movement AFTER upper barrier is hit
            upper_barrier_post_hit_movements = []

            for hit in barrier_hits:
    pass
    pass
                hit_type = hit.get("hit_type")  # "upper", "lower", or "both"
                hit_order = hit.get("hit_order")  # Which barrier was hit first
                price_deviation = hit.get("price_deviation", 0)  # How much further price moved
                position_type = hit.get("position_type")  # "long" or "short"

                if hit_type == "upper":
    pass
    pass
                    # Get price movement AFTER the upper barrier was hit
                    post_hit_movement = hit.get("post_hit_price_movement", 0)  # Price movement after barrier hit

                    if hit_order == "upper_first":
    pass
    pass
                        upper_hits_without_lower_first.append({
                            "position_type": position_type,
                            "post_hit_movement": post_hit_movement,
                            "timestamp": hit.get("timestamp")
                        })
                        upper_barrier_post_hit_movements.append(post_hit_movement)
                    else:
                        upper_hits_with_lower_first.append({
                            "position_type": position_type,
                            "post_hit_movement": post_hit_movement,
                            "timestamp": hit.get("timestamp")
                        })

                elif hit_type == "lower":
                    if hit_order == "lower_first":
    pass
    pass
                        lower_hits_without_upper_first.append({
                            "position_type": position_type,
                            "timestamp": hit.get("timestamp")
                        })
                    else:
                        lower_hits_with_upper_first.append({
                            "position_type": position_type,
                            "timestamp": hit.get("timestamp")
                        })

            return {
                "barrier_hit_analysis": {
                    "upper_hits_without_lower_first": {
                        "total_count": len(upper_hits_without_lower_first),
                        "long_positions": len([h for h in upper_hits_without_lower_first if h["position_type"] == "long"]),
                        "short_positions": len([h for h in upper_hits_without_lower_first if h["position_type"] == "short"]),
                        "average_post_hit_movement": sum(h["post_hit_movement"] for h in upper_hits_without_lower_first) / len(upper_hits_without_lower_first) if upper_hits_without_lower_first else 0,
                        "max_post_hit_movement": max(h["post_hit_movement"] for h in upper_hits_without_lower_first) if upper_hits_without_lower_first else 0,
                        "post_hit_movement_percentiles": self._calculate_percentiles([h["post_hit_movement"] for h in upper_hits_without_lower_first])
                    },
                    "lower_hits_without_upper_first": {
                        "total_count": len(lower_hits_without_upper_first),
                        "long_positions": len([h for h in lower_hits_without_upper_first if h["position_type"] == "long"]),
                        "short_positions": len([h for h in lower_hits_without_upper_first if h["position_type"] == "short"])
                    },
                    "upper_hits_with_lower_first": {
                        "total_count": len(upper_hits_with_lower_first),
                        "long_positions": len([h for h in upper_hits_with_lower_first if h["position_type"] == "long"]),
                        "short_positions": len([h for h in upper_hits_with_lower_first if h["position_type"] == "short"])
                    },
                    "lower_hits_with_upper_first": {
                        "total_count": len(lower_hits_with_upper_first),
                        "long_positions": len([h for h in lower_hits_with_upper_first if h["position_type"] == "long"]),
                        "short_positions": len([h for h in lower_hits_with_upper_first if h["position_type"] == "short"])
                    }
                },
                "price_deviation_analysis": {
                    "upper_barrier_deviations": {
                        "total_deviations": len(upper_barrier_price_deviations),
                        "mean_deviation": sum(upper_barrier_price_deviations) / len(upper_barrier_price_deviations) if upper_barrier_price_deviations else 0,
                        "max_deviation": max(upper_barrier_price_deviations) if upper_barrier_price_deviations else 0,
                        "min_deviation": min(upper_barrier_price_deviations) if upper_barrier_price_deviations else 0,
                        "deviation_percentiles": self._calculate_percentiles(upper_barrier_price_deviations),
                        "deviation_distribution": {
                            "small_deviations": len([d for d in upper_barrier_price_deviations if d <= 0.01]),  # <= 1%
                            "medium_deviations": len([d for d in upper_barrier_price_deviations if 0.01 < d <= 0.05]),  # 1-5%
                            "large_deviations": len([d for d in upper_barrier_price_deviations if d > 0.05])  # > 5%
                        }
                    },
                    "lower_barrier_deviations": {
                        "total_deviations": len(lower_barrier_price_deviations),
                        "mean_deviation": sum(lower_barrier_price_deviations) / len(lower_barrier_price_deviations) if lower_barrier_price_deviations else 0,
                        "max_deviation": max(lower_barrier_price_deviations) if lower_barrier_price_deviations else 0,
                        "min_deviation": min(lower_barrier_price_deviations) if lower_barrier_price_deviations else 0,
                        "deviation_percentiles": self._calculate_percentiles(lower_barrier_price_deviations),
                        "deviation_distribution": {
                            "small_deviations": len([d for d in lower_barrier_price_deviations if d <= 0.01]),  # <= 1%
                            "medium_deviations": len([d for d in lower_barrier_price_deviations if 0.01 < d <= 0.05]),  # 1-5%
                            "large_deviations": len([d for d in lower_barrier_price_deviations if d > 0.05])  # > 5%
                        }
                    }
                },
                "summary_statistics": {
                    "total_barrier_hits": len(barrier_hits),
                    "upper_first_hits": len(upper_hits_without_lower_first),
                    "lower_first_hits": len(lower_hits_without_upper_first),
                    "both_barriers_hit": len(upper_hits_with_lower_first) + len(lower_hits_with_upper_first),
                    "upper_first_ratio": len(upper_hits_without_lower_first) / len(barrier_hits) if barrier_hits else 0,
                    "lower_first_ratio": len(lower_hits_without_upper_first) / len(barrier_hits) if barrier_hits else 0
                }
            }

        except Exception as e:
            return {"error": f"Could not analyze triple barrier captured changes: {str(e)}"}

    def _calculate_percentiles(self, data: List[float]) -> Dict[str, float]:
    pass
    pass
        """Calculate percentiles for price change data."""
        try:
            if not data:
    pass
    except Exception as e:
        pass
    pass
                return {}
    except Exception as e:
        pass
            sorted_data = sorted(data)
            return {
                "p10": sorted_data[int(0.1 * len(sorted_data))],
                "p25": sorted_data[int(0.25 * len(sorted_data))],
                "p50": sorted_data[int(0.5 * len(sorted_data))],
                "p75": sorted_data[int(0.75 * len(sorted_data))],
                "p90": sorted_data[int(0.9 * len(sorted_data))]
            }
        except Exception:
            return {}

    # Warning generation methods
    def _generate_regime_splitting_warnings(self, result: Any) -> List[str]:
    pass
    pass
        """Generate warnings for regime splitting."""
        warnings = []

        try:
            if result.get("regime_balance", {}).get("is_balanced") == False:
    pass
    except Exception as e:
        pass
    pass
                warnings.append("Regime imbalance detected in data splits")

    except Exception as e:
        pass
            if result.get("quality_validation", {}).get("all_regimes_represented") == False:
    pass
    pass
                warnings.append("Not all regimes represented in test set")

            if result.get("data_distribution", {}).get("train_samples", 0) < 1000:
    pass
    pass
                warnings.append("Small training set size")

        except Exception:
            warnings.append("Could not generate regime splitting warnings")

        return warnings

    def _generate_triple_barrier_warnings(self, result: Any) -> List[str]:
    pass
    pass
        """Generate warnings for triple barrier method."""
        warnings = []

        try:
            if result.get("label_quality", {}).get("balanced_labels", {}).get("is_balanced") == False:
    pass
    except Exception as e:
        pass
    pass
                warnings.append("Label imbalance detected")

    except Exception as e:
        pass
            if result.get("label_quality", {}).get("no_label_leakage") == False:
    pass
    pass
                warnings.append("Potential label leakage detected")

            if result.get("barrier_analysis", {}).get("total_labels", 0) < 1000:
    pass
    pass
                warnings.append("Low number of generated labels")

        except Exception:
            warnings.append("Could not generate triple barrier warnings")

        return warnings

    def _generate_hmm_training_warnings(self, result: Any) -> List[str]:
    pass
    pass
        """Generate warnings for HMM training."""
        warnings = []

        try:
            if result.get("model_performance", {}).get("overfitting_score", {}).get("is_overfitting") == True:
    pass
    except Exception as e:
        pass
    pass
                warnings.append("Model shows signs of overfitting")

    except Exception as e:
        pass
            if result.get("training_analysis", {}).get("convergence_status") == False:
    pass
    pass
                warnings.append("Model did not converge")

            if result.get("model_performance", {}).get("validation_accuracy", 0) < 0.5:
    pass
    pass
                warnings.append("Low validation accuracy")

        except Exception:
            warnings.append("Could not generate HMM training warnings")

        return warnings

    def _generate_analyst_enhancement_warnings(self, result: Any) -> List[str]:
    pass
    pass
        """Generate warnings for analyst enhancement."""
        warnings = []

        try:
            if result.get("performance_impact", {}).get("accuracy_improvement", 0) < 0.01:
    pass
    except Exception as e:
        pass
    pass
                warnings.append("Minimal accuracy improvement from enhancement")

    except Exception as e:
        pass
            if result.get("enhancement_quality", {}).get("feature_relevance", 0) < 0.5:
    pass
    pass
                warnings.append("Low feature relevance in enhancements")

            if result.get("enhancement_analysis", {}).get("feature_increase", 0) > 100:
    pass
    pass
                warnings.append("Large increase in feature count may cause overfitting")

        except Exception:
            warnings.append("Could not generate analyst enhancement warnings")

        return warnings

    def _generate_feature_generation_warnings(self, result: Any) -> List[str]:
    pass
    pass
        """Generate warnings for feature generation (Step 6)."""
        warnings = []

        try:
            if result.get("feature_generation_analysis", {}).get("feature_increase", 0) > 200:
    pass
    except Exception as e:
        pass
    pass
                warnings.append("Large increase in feature count may cause overfitting")

    except Exception as e:
        pass
            if result.get("feature_quality", {}).get("feature_relevance", 0) < 0.5:
    pass
    pass
                warnings.append("Low feature relevance in generated features")

            if result.get("generation_performance", {}).get("generation_time", 0) > 300:
    pass
    pass
                warnings.append("Feature generation took longer than expected")

        except Exception:
            warnings.append("Could not generate feature generation warnings")

        return warnings

    def _generate_matrix_selection_warnings(self, result: Any) -> List[str]:
    pass
    pass
        """Generate warnings for matrix feature selection (Step 7)."""
        warnings = []

        try:
            if result.get("selection_analysis", {}).get("reduction_ratio", 0) > 0.8:
    pass
    except Exception as e:
        pass
    pass
                warnings.append("High feature reduction may lose important information")

    except Exception as e:
        pass
            if result.get("performance_impact", {}).get("accuracy_change", 0) < -0.02:
    pass
    pass
                warnings.append("Feature selection caused significant accuracy drop")

            if result.get("selection_quality", {}).get("selection_stability", 0) < 0.7:
    pass
    pass
                warnings.append("Low selection stability across different samples")

        except Exception:
            warnings.append("Could not generate matrix selection warnings")

        return warnings

    # Placeholder methods for other step metrics (to be implemented based on actual step outputs)
    async def _get_tactician_labeling_metrics(self, result: Any) -> Dict[str, Any]:
        return {"status": "Metrics calculation not implemented yet"}

    async def _get_tactician_training_metrics(self, result: Any) -> Dict[str, Any]:
        return {"status": "Metrics calculation not implemented yet"}

    async def _get_confidence_calibration_metrics(self, result: Any) -> Dict[str, Any]:
        return {"status": "Metrics calculation not implemented yet"}

    async def _get_optimization_metrics(self, result: Any) -> Dict[str, Any]:
        return {"status": "Metrics calculation not implemented yet"}

    async def _get_walk_forward_metrics(self, result: Any) -> Dict[str, Any]:
        return {"status": "Metrics calculation not implemented yet"}

    async def _get_monte_carlo_metrics(self, result: Any) -> Dict[str, Any]:
        return {"status": "Metrics calculation not implemented yet"}

    async def _get_ab_testing_metrics(self, result: Any) -> Dict[str, Any]:
        return {"status": "Metrics calculation not implemented yet"}

    async def _get_saving_metrics(self, result: Any) -> Dict[str, Any]:
        return {"status": "Metrics calculation not implemented yet"}

    # Helper methods for HMM analysis
    def _calculate_regime_separation(self, result: Any) -> Dict[str, Any]:
    pass
    pass
        return {"status": "Not implemented yet"}

    def _calculate_regime_stability(self, result: Any) -> Dict[str, Any]:
    pass
    pass
        return {"status": "Not implemented yet"}

    def _calculate_regime_duration(self, result: Any) -> Dict[str, Any]:
    pass
    pass
        return {"status": "Not implemented yet"}


# Convenience function to create enhanced training manager
async def create_enhanced_training_manager_with_reporting(config: Dict[str, Any]) -> EnhancedTrainingManagerWithReporting:
    """Create an enhanced training manager with comprehensive reporting."""

    manager = EnhancedTrainingManagerWithReporting(config)
    await manager.initialize()
    return manager


# Export the main class and convenience function
__all__ = [
    "EnhancedTrainingManagerWithReporting",
    "create_enhanced_training_manager_with_reporting"
]