"""
Enhanced Training Manager with Existing Decorators Integration
Provides thorough decorators, detailed reports, and consistent storage for all pipeline steps.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.training.enhanced_training_manager import EnhancedTrainingManager
from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors
from src.utils.training_pipeline_decorators import (
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
        """Generate and save a detailed report for a specific step."""
        
        if not self.enable_detailed_reporting:
            return
        
        try:
            step_end_time = time.time()
            execution_duration = step_end_time - step_start_time
            
            # Get step-specific quality metrics
            step_quality_metrics = await self._get_step_quality_metrics(step_name, step_result)
            
            # Create step report
            step_report = {
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
            
            # Save step report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{step_name}_{timestamp}_{self.current_pipeline_execution_id}.json"
            report_path = self.pipeline_reports_dir / filename
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(step_report, f, indent=2, ensure_ascii=False, default=str)
            
            # Generate summary report
            summary_report = self._generate_step_summary(step_report)
            summary_filename = f"{step_name}_{timestamp}_{self.current_pipeline_execution_id}_summary.txt"
            summary_path = self.pipeline_reports_dir / summary_filename
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary_report)
            
            # Store in pipeline report
            self.step_reports[step_name] = step_report
            
            # Log completion
            status_emoji = "✅" if step_success else "❌"
            self.logger.info(f"{status_emoji} [STEP REPORT] {step_name} report saved to {report_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate step report for {step_name}: {e}")
    
    def _summarize_result(self, result: Any) -> Dict[str, Any]:
        """Create a summary of the step result."""
        
        try:
            if hasattr(result, 'shape'):  # DataFrame
                return {
                    "type": "DataFrame",
                    "shape": result.shape,
                    "columns_count": len(result.columns),
                    "memory_usage_mb": result.memory_usage(deep=True).sum() / (1024**2) if hasattr(result, 'memory_usage') else None
                }
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
            result_summary = step_report["result_summary"]
            summary.append("RESULT SUMMARY:")
            summary.append("-" * 40)
            for key, value in result_summary.items():
                summary.append(f"  {key}: {value}")
            summary.append("")
        
        # Step-specific quality metrics
        if step_report.get("step_quality_metrics"):
            quality_metrics = step_report["step_quality_metrics"]
            summary.append("STEP QUALITY METRICS:")
            summary.append("-" * 40)
            
            # Handle different types of quality metrics based on step
            step_name = step_report['step_name']
            
            if step_name in ["step1_data_collection", "step1_5_data_converter"]:
                if "data_quality" in quality_metrics:
                    data_quality = quality_metrics["data_quality"]
                    summary.append("  Data Quality:")
                    summary.append(f"    Total Rows: {data_quality.get('total_rows', 'N/A')}")
                    summary.append(f"    Total Columns: {data_quality.get('total_columns', 'N/A')}")
                    summary.append(f"    Memory Usage: {data_quality.get('memory_usage_mb', 'N/A'):.2f} MB")
                    
                    if "null_percentage" in data_quality:
                        max_null = max(data_quality["null_percentage"].values()) if data_quality["null_percentage"] else 0
                        summary.append(f"    Max Null Percentage: {max_null:.2f}%")
                    
                    if "duplicate_percentage" in data_quality:
                        summary.append(f"    Duplicate Rows: {data_quality['duplicate_percentage']:.2f}%")
                
                if "data_validation" in quality_metrics:
                    validation = quality_metrics["data_validation"]
                    summary.append("  Data Validation:")
                    summary.append(f"    Has Required Columns: {validation.get('has_required_columns', 'N/A')}")
                    
                    if "price_consistency" in validation:
                        price_check = validation["price_consistency"]
                        summary.append(f"    Price Consistency: {'❌ Issues' if price_check.get('has_issues') else '✅ OK'}")
                        if price_check.get('issues'):
                            for issue in price_check['issues']:
                                summary.append(f"      - {issue}")
                    
                    if "volume_consistency" in validation:
                        volume_check = validation["volume_consistency"]
                        summary.append(f"    Volume Consistency: {'❌ Issues' if volume_check.get('has_issues') else '✅ OK'}")
                        if volume_check.get('issues'):
                            for issue in volume_check['issues']:
                                summary.append(f"      - {issue}")
            
            elif step_name == "step2_feature_engineering":
                if "feature_quality" in quality_metrics:
                    feature_quality = quality_metrics["feature_quality"]
                    summary.append("  Feature Quality:")
                    summary.append(f"    Total Features: {feature_quality.get('total_features', 'N/A')}")
                    summary.append(f"    Numeric Features: {feature_quality.get('numeric_features', 'N/A')}")
                    summary.append(f"    Categorical Features: {feature_quality.get('categorical_features', 'N/A')}")
                    summary.append(f"    Memory Usage: {feature_quality.get('memory_usage_mb', 'N/A'):.2f} MB")
                
                if "multicollinearity_analysis" in quality_metrics:
                    multicollinearity = quality_metrics["multicollinearity_analysis"]
                    summary.append("  Multicollinearity Analysis:")
                    summary.append(f"    High Correlation Pairs: {multicollinearity.get('high_correlation_count', 'N/A')}")
                    
                    if "high_vif_features" in multicollinearity:
                        high_vif = multicollinearity["high_vif_features"]
                        summary.append(f"    High VIF Features: {len(high_vif)}")
                        if high_vif:
                            summary.append(f"      - {', '.join(high_vif[:5])}{'...' if len(high_vif) > 5 else ''}")
                
                if "feature_statistics" in quality_metrics:
                    feature_stats = quality_metrics["feature_statistics"]
                    summary.append("  Feature Statistics:")
                    summary.append(f"    Constant Features: {len(feature_stats.get('constant_features', []))}")
                    summary.append(f"    Low Variance Features: {len(feature_stats.get('low_variance_features', []))}")
                    summary.append(f"    High Cardinality Features: {len(feature_stats.get('high_cardinality_features', []))}")
                
                if "data_quality_issues" in quality_metrics:
                    quality_issues = quality_metrics["data_quality_issues"]
                    summary.append("  Data Quality Issues:")
                    summary.append(f"    NaN Features: {len(quality_issues.get('nan_features', []))}")
                    summary.append(f"    Inf Features: {len(quality_issues.get('inf_features', []))}")
                    summary.append(f"    Zero Variance Features: {len(quality_issues.get('zero_variance_features', []))}")
            
            elif step_name == "step3_hmm_regime_discovery":
                if "regime_analysis" in quality_metrics:
                    regime_analysis = quality_metrics["regime_analysis"]
                    summary.append("  Regime Analysis:")
                    summary.append(f"    Number of Regimes: {regime_analysis.get('number_of_regimes', 'N/A')}")
                    summary.append(f"    Convergence Status: {regime_analysis.get('convergence_status', 'N/A')}")
                    summary.append(f"    Log Likelihood: {regime_analysis.get('log_likelihood', 'N/A')}")
                
                if "validation_metrics" in quality_metrics:
                    validation_metrics = quality_metrics["validation_metrics"]
                    summary.append("  Validation Metrics:")
                    summary.append(f"    AIC Score: {validation_metrics.get('aic_score', 'N/A')}")
                    summary.append(f"    BIC Score: {validation_metrics.get('bic_score', 'N/A')}")
                    summary.append(f"    Model Complexity: {validation_metrics.get('model_complexity', 'N/A')}")
            
            # Add warnings from quality metrics
            if "warnings" in quality_metrics and quality_metrics["warnings"]:
                summary.append("  Quality Warnings:")
                for warning in quality_metrics["warnings"]:
                    summary.append(f"    ⚠️ {warning}")
            
            summary.append("")
        
        # System resources
        if step_report.get("system_resources"):
            resources = step_report["system_resources"]
            summary.append("SYSTEM RESOURCES:")
            summary.append("-" * 40)
            for key, value in resources.items():
                if isinstance(value, float):
                    summary.append(f"  {key}: {value:.2f}")
                else:
                    summary.append(f"  {key}: {value}")
            summary.append("")
        
        # Errors and warnings
        if step_report.get("errors"):
            summary.append("ERRORS:")
            summary.append("-" * 40)
            for error in step_report["errors"]:
                summary.append(f"❌ {error}")
            summary.append("")
        
        if step_report.get("warnings"):
            summary.append("WARNINGS:")
            summary.append("-" * 40)
            for warning in step_report["warnings"]:
                summary.append(f"⚠️ {warning}")
            summary.append("")
        
        summary.append("=" * 80)
        summary.append("End of Step Report")
        summary.append("=" * 80)
        
        return "\n".join(summary)
    
    # Enhanced step execution methods with report generation
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step1_data_collection"
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
            from src.training.steps import step1_data_collection
            
            result = await step1_data_collection.run_step(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                force_rerun=force_rerun,
            )
            
            # Generate step report
            await self._generate_step_report(
                "step1_data_collection",
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
                "step1_data_collection",
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
        context="step1_5_data_converter"
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
            from src.training.steps.step1_5_data_converter import run_step as step1_5_run_step
            
            result = await step1_5_run_step(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                force_rerun=force_rerun,
            )
            
            # Generate step report
            await self._generate_step_report(
                "step1_5_data_converter",
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
                "step1_5_data_converter",
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
        context="step2_feature_engineering"
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
            from src.training.steps import step2_feature_engineering
            
            result = await step2_feature_engineering.run_step(
                symbol=symbol,
                exchange=exchange,
                data_dir=data_dir,
                timeframe=timeframe,
                force_rerun=force_rerun,
                feature_config=feature_config,
            )
            
            # Generate step report
            await self._generate_step_report(
                "step2_feature_engineering",
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
                "step2_feature_engineering",
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
        context="step3_hmm_regime_discovery"
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
            from src.training.steps import step3_hmm_regime_discovery as _step3
            
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
                "step3_hmm_regime_discovery",
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
                "step3_hmm_regime_discovery",
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
        context="step4_regime_data_splitting"
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
            from src.training.steps import step4_regime_data_splitting
            
            result = await step4_regime_data_splitting.run_step(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                force_rerun=force_rerun,
                config=self.config,
            )
            
            # Generate step report
            await self._generate_step_report(
                "step4_regime_data_splitting",
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
                "step4_regime_data_splitting",
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
        context="step5_triple_barrier_method"
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
            from src.training.steps import step5_triple_barrier_method
            
            result = await step5_triple_barrier_method.run_step(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                force_rerun=force_rerun,
                config=self.config,
            )
            
            # Generate step report
            await self._generate_step_report(
                "step5_triple_barrier_method",
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
                "step5_triple_barrier_method",
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
        context="step6_hmm_based_training"
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
            from src.training.steps import step6_hmm_based_training
            
            result = await step6_hmm_based_training.run_step(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                force_rerun=force_rerun,
                config=self.config,
            )
            
            # Generate step report
            await self._generate_step_report(
                "step6_hmm_based_training",
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
                "step6_hmm_based_training",
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
        context="step7_analyst_enhancement"
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
            from src.training.steps import step7_analyst_enhancement
            
            result = await step7_analyst_enhancement.run_step(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                force_rerun=force_rerun,
                config=self.config,
            )
            
            # Generate step report
            await self._generate_step_report(
                "step7_analyst_enhancement",
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
                "step7_analyst_enhancement",
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
        context="step8_tactician_labeling"
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
            from src.training.steps import step8_tactician_labeling
            
            result = await step8_tactician_labeling.run_step(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                force_rerun=force_rerun,
                config=self.config,
            )
            
            # Generate step report
            await self._generate_step_report(
                "step8_tactician_labeling",
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
                "step8_tactician_labeling",
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
        context="step9_tactician_specialist_training"
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
            from src.training.steps import step9_tactician_specialist_training
            
            result = await step9_tactician_specialist_training.run_step(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                force_rerun=force_rerun,
                config=self.config,
            )
            
            # Generate step report
            await self._generate_step_report(
                "step9_tactician_specialist_training",
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
                "step9_tactician_specialist_training",
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
            summary.append("Step Execution Summary:")
            summary.append("-" * 50)
            
            step_statuses = {
                "success": [],
                "failed": [],
                "skipped": []
            }
            
            for step_name, step_report in steps.items():
                status = step_report.get("success", False)
                if status:
                    step_statuses["success"].append(step_name)
                else:
                    step_statuses["failed"].append(step_name)
            
            summary.append(f"✅ Successful Steps ({len(step_statuses['success'])}):")
            for step in step_statuses["success"]:
                duration = steps[step].get("execution_duration_formatted", "N/A")
                summary.append(f"  - {step}: {duration}")
            
            if step_statuses["failed"]:
                summary.append(f"❌ Failed Steps ({len(step_statuses['failed'])}):")
                for step in step_statuses["failed"]:
                    summary.append(f"  - {step}")
            
            summary.append("")
        
        # Errors and warnings
        if pipeline_report.get("errors"):
            summary.append("Pipeline Errors:")
            summary.append("-" * 50)
            for error in pipeline_report["errors"]:
                summary.append(f"❌ {error.get('type', 'Unknown')}: {error.get('message', 'No message')}")
            summary.append("")
        
        if pipeline_report.get("warnings"):
            summary.append("Pipeline Warnings:")
            summary.append("-" * 50)
            for warning in pipeline_report["warnings"]:
                summary.append(f"⚠️ {warning}")
            summary.append("")
        
        # Recommendations
        if pipeline_report.get("recommendations"):
            summary.append("Recommendations:")
            summary.append("-" * 50)
            for rec in pipeline_report["recommendations"]:
                summary.append(f"💡 {rec}")
            summary.append("")
        
        summary.append("=" * 100)
        summary.append("End of Pipeline Report")
        summary.append("=" * 100)
        
        return "\n".join(summary)

    async def _get_step_quality_metrics(self, step_name: str, step_result: Any) -> Dict[str, Any]:
        """Get step-specific quality metrics and validation information."""
        
        try:
            if step_name == "step1_data_collection":
                return await self._get_data_collection_metrics(step_result)
            elif step_name == "step1_5_data_converter":
                return await self._get_data_converter_metrics(step_result)
            elif step_name == "step2_feature_engineering":
                return await self._get_feature_engineering_metrics(step_result)
            elif step_name == "step3_hmm_regime_discovery":
                return await self._get_hmm_regime_metrics(step_result)
            elif step_name == "step4_regime_data_splitting":
                return await self._get_regime_splitting_metrics(step_result)
            elif step_name == "step5_triple_barrier_method":
                return await self._get_triple_barrier_metrics(step_result)
            elif step_name == "step6_hmm_based_training":
                return await self._get_hmm_training_metrics(step_result)
            elif step_name == "step7_analyst_enhancement":
                return await self._get_analyst_enhancement_metrics(step_result)
            elif step_name == "step8_tactician_labeling":
                return await self._get_tactician_labeling_metrics(step_result)
            elif step_name == "step9_tactician_specialist_training":
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
            
            if isinstance(result, pd.DataFrame) and not result.empty:
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
            
            if isinstance(result, pd.DataFrame) and not result.empty:
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
            import numpy as np
            
            if isinstance(result, pd.DataFrame) and not result.empty:
                # Calculate multicollinearity
                numeric_cols = result.select_dtypes(include=[np.number]).columns
                correlation_matrix = result[numeric_cols].corr()
                high_correlation_pairs = []
                
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
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
                    for col in numeric_cols:
                        if len(numeric_cols) > 1:
                            other_cols = [c for c in numeric_cols if c != col]
                            if len(other_cols) > 0:
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
            else:
                return {"error": "No HMM result available"}
                
        except Exception as e:
            return {"error": f"Failed to analyze HMM regime metrics: {str(e)}"}
    
    # Helper methods for quality checks
    def _check_price_consistency(self, df) -> Dict[str, Any]:
        """Check price data consistency."""
        try:
            issues = []
            if 'high' in df.columns and 'low' in df.columns:
                invalid_high_low = (df['high'] < df['low']).sum()
                if invalid_high_low > 0:
                    issues.append(f"High < Low: {invalid_high_low} rows")
            
            if 'open' in df.columns and 'close' in df.columns:
                zero_prices = ((df['open'] == 0) | (df['close'] == 0)).sum()
                if zero_prices > 0:
                    issues.append(f"Zero prices: {zero_prices} rows")
            
            return {
                "has_issues": len(issues) > 0,
                "issues": issues
            }
        except Exception:
            return {"error": "Could not check price consistency"}
    
    def _check_volume_consistency(self, df) -> Dict[str, Any]:
        """Check volume data consistency."""
        try:
            issues = []
            if 'volume' in df.columns:
                negative_volume = (df['volume'] < 0).sum()
                if negative_volume > 0:
                    issues.append(f"Negative volume: {negative_volume} rows")
                
                zero_volume = (df['volume'] == 0).sum()
                if zero_volume > 0:
                    issues.append(f"Zero volume: {zero_volume} rows")
            
            return {
                "has_issues": len(issues) > 0,
                "issues": issues
            }
        except Exception:
            return {"error": "Could not check volume consistency"}
    
    def _check_timestamp_consistency(self, df) -> Dict[str, Any]:
        """Check timestamp consistency."""
        try:
            issues = []
            if hasattr(df.index, 'is_monotonic_increasing'):
                if not df.index.is_monotonic_increasing:
                    issues.append("Timestamps not in ascending order")
            
            if hasattr(df.index, 'duplicated'):
                duplicates = df.index.duplicated().sum()
                if duplicates > 0:
                    issues.append(f"Duplicate timestamps: {duplicates}")
            
            return {
                "has_issues": len(issues) > 0,
                "issues": issues
            }
        except Exception:
            return {"error": "Could not check timestamp consistency"}
    
    def _generate_data_collection_warnings(self, df) -> List[str]:
        """Generate warnings for data collection."""
        warnings = []
        
        try:
            if df.isnull().any().any():
                null_percentage = (df.isnull().sum() / len(df) * 100).max()
                if null_percentage > 10:
                    warnings.append(f"High null percentage: {null_percentage:.2f}%")
            
            if len(df) < 1000:
                warnings.append(f"Low data volume: {len(df)} rows")
            
            if 'volume' in df.columns and (df['volume'] == 0).sum() > len(df) * 0.5:
                warnings.append("High percentage of zero volume data")
                
        except Exception:
            warnings.append("Could not generate data collection warnings")
        
        return warnings
    
    def _generate_data_converter_warnings(self, df) -> List[str]:
        """Generate warnings for data converter."""
        warnings = []
        
        try:
            if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                warnings.append("Missing required OHLCV columns")
            
            if df.isnull().any().any():
                warnings.append("Data contains null values after conversion")
                
        except Exception:
            warnings.append("Could not generate data converter warnings")
        
        return warnings
    
    def _generate_feature_engineering_warnings(self, df, high_correlation_pairs, vif_scores) -> List[str]:
        """Generate warnings for feature engineering."""
        warnings = []
        
        try:
            if len(high_correlation_pairs) > 10:
                warnings.append(f"High multicollinearity: {len(high_correlation_pairs)} highly correlated feature pairs")
            
            high_vif_features = [col for col, vif in vif_scores.items() if isinstance(vif, (int, float)) and vif > 10]
            if high_vif_features:
                warnings.append(f"High VIF features: {high_vif_features}")
            
            if df.isnull().any().any():
                warnings.append("Features contain null values")
                
        except Exception:
            warnings.append("Could not generate feature engineering warnings")
        
        return warnings
    
    # Placeholder methods for other step metrics (to be implemented based on actual step outputs)
    async def _get_regime_splitting_metrics(self, result: Any) -> Dict[str, Any]:
        return {"status": "Metrics calculation not implemented yet"}
    
    async def _get_triple_barrier_metrics(self, result: Any) -> Dict[str, Any]:
        return {"status": "Metrics calculation not implemented yet"}
    
    async def _get_hmm_training_metrics(self, result: Any) -> Dict[str, Any]:
        return {"status": "Metrics calculation not implemented yet"}
    
    async def _get_analyst_enhancement_metrics(self, result: Any) -> Dict[str, Any]:
        return {"status": "Metrics calculation not implemented yet"}
    
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
        return {"status": "Not implemented yet"}
    
    def _calculate_regime_stability(self, result: Any) -> Dict[str, Any]:
        return {"status": "Not implemented yet"}
    
    def _calculate_regime_duration(self, result: Any) -> Dict[str, Any]:
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