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
        
        # Initialize pipeline reporting
        pipeline_report = {
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
            
            # Generate and store pipeline report
            await self._generate_pipeline_report(pipeline_report)
            
            return result
            
        except Exception as e:
            pipeline_report["errors"].append({
                "type": type(e).__name__,
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            })
            await self._generate_pipeline_report(pipeline_report)
            raise
    
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
        
        try:
            from src.training.steps import step1_data_collection
            
            result = await step1_data_collection.run_step(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                force_rerun=force_rerun,
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 1 failed: {e}")
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
        
        try:
            from src.training.steps.step1_5_data_converter import run_step as step1_5_run_step
            
            result = await step1_5_run_step(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                force_rerun=force_rerun,
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 1.5 failed: {e}")
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
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 2 failed: {e}")
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
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 3 failed: {e}")
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
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 4 failed: {e}")
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
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 5 failed: {e}")
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
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 6 failed: {e}")
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
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 7 failed: {e}")
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
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 8 failed: {e}")
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
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 9 failed: {e}")
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
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 10 failed: {e}")
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
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 11 failed: {e}")
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
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 12 failed: {e}")
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
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 13 failed: {e}")
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
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 14 failed: {e}")
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
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 15 failed: {e}")
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
        summary.append(f"  Symbol: {training_input.get('symbol', 'N/A')}")
        summary.append(f"  Exchange: {training_input.get('exchange', 'N/A')}")
        summary.append(f"  Timeframe: {training_input.get('timeframe', 'N/A')}")
        summary.append(f"  Start Time: {pipeline_report.get('pipeline_start_time', 'N/A')}")
        summary.append(f"  End Time: {pipeline_report.get('pipeline_end_time', 'N/A')}")
        summary.append(f"  Overall Success: {pipeline_report.get('overall_success', 'N/A')}")
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