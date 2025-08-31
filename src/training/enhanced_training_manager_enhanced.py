"""
Enhanced Training Manager with Comprehensive Decorators and Reporting
Provides thorough decorators, detailed reports, and consistent storage for all pipeline steps.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.training.enhanced_training_manager import EnhancedTrainingManager
from src.utils.enhanced_pipeline_decorators import (
    enhanced_pipeline_step,
    detailed_pipeline_step,
    comprehensive_pipeline_step,
    get_step_reports,
    get_latest_step_report,
    cleanup_old_reports,
    ReportLevel
)
from src.utils.logger import system_logger


class EnhancedTrainingManagerWithReporting(EnhancedTrainingManager):
    """
    Enhanced Training Manager with comprehensive decorators and detailed reporting.
    
    This class extends the base EnhancedTrainingManager to provide:
    1. Thorough decorators for each pipeline step
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
        self.report_level = ReportLevel(self.reporting_config.get("report_level", "detailed"))
        self.auto_cleanup_reports = self.reporting_config.get("auto_cleanup_reports", True)
        self.reports_retention_days = self.reporting_config.get("reports_retention_days", 30)
        
        self.logger.info(f"🚀 Enhanced Training Manager with Reporting initialized")
        self.logger.info(f"   📊 Report Level: {self.report_level.value}")
        self.logger.info(f"   📁 Reports Directory: {self.pipeline_reports_dir}")
        self.logger.info(f"   🧹 Auto Cleanup: {self.auto_cleanup_reports}")
    
    @enhanced_pipeline_step("execute_enhanced_training", ReportLevel.COMPREHENSIVE)
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
            
            # Collect step reports
            await self._collect_step_reports(pipeline_report)
            
            # Generate and store pipeline report
            await self._generate_pipeline_report(pipeline_report)
            
            # Cleanup old reports if enabled
            if self.auto_cleanup_reports:
                await cleanup_old_reports(self.reports_retention_days)
            
            return result
            
        except Exception as e:
            pipeline_report["errors"].append({
                "type": type(e).__name__,
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            })
            await self._generate_pipeline_report(pipeline_report)
            raise
    
    @detailed_pipeline_step("step1_data_collection")
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
            
            # Track data collection metrics
            if result:
                await self._track_data_collection_metrics(symbol, exchange, timeframe)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 1 failed: {e}")
            raise
    
    @detailed_pipeline_step("step1_5_data_converter")
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
            
            # Track data conversion metrics
            if result:
                await self._track_data_conversion_metrics(symbol, exchange, timeframe)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 1.5 failed: {e}")
            raise
    
    @detailed_pipeline_step("step2_feature_engineering")
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
            
            # Track feature engineering metrics
            if result:
                await self._track_feature_engineering_metrics(symbol, exchange, timeframe)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 2 failed: {e}")
            raise
    
    @detailed_pipeline_step("step2_5_sr_optimization")
    async def _execute_step2_5_enhanced(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        force_rerun: bool,
    ) -> bool:
        """Execute Step 2.5: S/R Optimization with enhanced reporting."""
        
        try:
            from src.training.steps import step2_5_sr_optimization
            
            result = await step2_5_sr_optimization.run_step(
                config=self.config,
            )
            
            # Track S/R optimization metrics
            if result:
                await self._track_sr_optimization_metrics(symbol, exchange, timeframe)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 2.5 failed: {e}")
            raise
    
    @comprehensive_pipeline_step("step3_hmm_regime_discovery")
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
            
            # Track HMM regime discovery metrics
            if result:
                await self._track_hmm_regime_metrics(symbol, exchange, timeframe)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 3 failed: {e}")
            raise
    
    @detailed_pipeline_step("step4_regime_data_splitting")
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
            
            # Track regime data splitting metrics
            if result:
                await self._track_regime_splitting_metrics(symbol, exchange, timeframe)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 4 failed: {e}")
            raise
    
    @detailed_pipeline_step("step5_triple_barrier_method")
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
            
            # Track triple barrier metrics
            if result:
                await self._track_triple_barrier_metrics(symbol, exchange, timeframe)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 5 failed: {e}")
            raise
    
    @comprehensive_pipeline_step("step6_hmm_based_training")
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
            
            # Track HMM training metrics
            if result:
                await self._track_hmm_training_metrics(symbol, exchange, timeframe)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 6 failed: {e}")
            raise
    
    @detailed_pipeline_step("step7_analyst_enhancement")
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
            
            # Track analyst enhancement metrics
            if result:
                await self._track_analyst_enhancement_metrics(symbol, exchange, timeframe)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 7 failed: {e}")
            raise
    
    @detailed_pipeline_step("step8_tactician_labeling")
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
            
            # Track tactician labeling metrics
            if result:
                await self._track_tactician_labeling_metrics(symbol, exchange, timeframe)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 8 failed: {e}")
            raise
    
    @comprehensive_pipeline_step("step9_tactician_specialist_training")
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
            
            # Track tactician training metrics
            if result:
                await self._track_tactician_training_metrics(symbol, exchange, timeframe)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 9 failed: {e}")
            raise
    
    @detailed_pipeline_step("step10_confidence_calibration")
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
            
            # Track confidence calibration metrics
            if result:
                await self._track_confidence_calibration_metrics(symbol, exchange, timeframe)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 10 failed: {e}")
            raise
    
    @detailed_pipeline_step("step11_final_parameters_optimization")
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
            
            # Track optimization metrics
            if result:
                await self._track_optimization_metrics(symbol, exchange, timeframe)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 11 failed: {e}")
            raise
    
    @comprehensive_pipeline_step("step12_walk_forward_validation")
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
            
            # Track walk forward validation metrics
            if result:
                await self._track_walk_forward_metrics(symbol, exchange, timeframe)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 12 failed: {e}")
            raise
    
    @comprehensive_pipeline_step("step13_monte_carlo_validation")
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
            
            # Track Monte Carlo validation metrics
            if result:
                await self._track_monte_carlo_metrics(symbol, exchange, timeframe)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 13 failed: {e}")
            raise
    
    @detailed_pipeline_step("step14_ab_testing")
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
            
            # Track A/B testing metrics
            if result:
                await self._track_ab_testing_metrics(symbol, exchange, timeframe)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 14 failed: {e}")
            raise
    
    @detailed_pipeline_step("step15_saving")
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
            
            # Track saving metrics
            if result:
                await self._track_saving_metrics(symbol, exchange, timeframe)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Step 15 failed: {e}")
            raise
    
    # Metric tracking methods for each step
    async def _track_data_collection_metrics(self, symbol: str, exchange: str, timeframe: str):
        """Track data collection metrics."""
        # Implementation for data collection metrics
        pass
    
    async def _track_data_conversion_metrics(self, symbol: str, exchange: str, timeframe: str):
        """Track data conversion metrics."""
        # Implementation for data conversion metrics
        pass
    
    async def _track_feature_engineering_metrics(self, symbol: str, exchange: str, timeframe: str):
        """Track feature engineering metrics."""
        # Implementation for feature engineering metrics
        pass
    
    async def _track_sr_optimization_metrics(self, symbol: str, exchange: str, timeframe: str):
        """Track S/R optimization metrics."""
        # Implementation for S/R optimization metrics
        pass
    
    async def _track_hmm_regime_metrics(self, symbol: str, exchange: str, timeframe: str):
        """Track HMM regime discovery metrics."""
        # Implementation for HMM regime metrics
        pass
    
    async def _track_regime_splitting_metrics(self, symbol: str, exchange: str, timeframe: str):
        """Track regime data splitting metrics."""
        # Implementation for regime splitting metrics
        pass
    
    async def _track_triple_barrier_metrics(self, symbol: str, exchange: str, timeframe: str):
        """Track triple barrier method metrics."""
        # Implementation for triple barrier metrics
        pass
    
    async def _track_hmm_training_metrics(self, symbol: str, exchange: str, timeframe: str):
        """Track HMM training metrics."""
        # Implementation for HMM training metrics
        pass
    
    async def _track_analyst_enhancement_metrics(self, symbol: str, exchange: str, timeframe: str):
        """Track analyst enhancement metrics."""
        # Implementation for analyst enhancement metrics
        pass
    
    async def _track_tactician_labeling_metrics(self, symbol: str, exchange: str, timeframe: str):
        """Track tactician labeling metrics."""
        # Implementation for tactician labeling metrics
        pass
    
    async def _track_tactician_training_metrics(self, symbol: str, exchange: str, timeframe: str):
        """Track tactician training metrics."""
        # Implementation for tactician training metrics
        pass
    
    async def _track_confidence_calibration_metrics(self, symbol: str, exchange: str, timeframe: str):
        """Track confidence calibration metrics."""
        # Implementation for confidence calibration metrics
        pass
    
    async def _track_optimization_metrics(self, symbol: str, exchange: str, timeframe: str):
        """Track optimization metrics."""
        # Implementation for optimization metrics
        pass
    
    async def _track_walk_forward_metrics(self, symbol: str, exchange: str, timeframe: str):
        """Track walk forward validation metrics."""
        # Implementation for walk forward metrics
        pass
    
    async def _track_monte_carlo_metrics(self, symbol: str, exchange: str, timeframe: str):
        """Track Monte Carlo validation metrics."""
        # Implementation for Monte Carlo metrics
        pass
    
    async def _track_ab_testing_metrics(self, symbol: str, exchange: str, timeframe: str):
        """Track A/B testing metrics."""
        # Implementation for A/B testing metrics
        pass
    
    async def _track_saving_metrics(self, symbol: str, exchange: str, timeframe: str):
        """Track saving metrics."""
        # Implementation for saving metrics
        pass
    
    async def _collect_step_reports(self, pipeline_report: Dict[str, Any]):
        """Collect reports from all executed steps."""
        
        step_names = [
            "step1_data_collection",
            "step1_5_data_converter", 
            "step2_feature_engineering",
            "step2_5_sr_optimization",
            "step3_hmm_regime_discovery",
            "step4_regime_data_splitting",
            "step5_triple_barrier_method",
            "step6_hmm_based_training",
            "step7_analyst_enhancement",
            "step8_tactician_labeling",
            "step9_tactician_specialist_training",
            "step10_confidence_calibration",
            "step11_final_parameters_optimization",
            "step12_walk_forward_validation",
            "step13_monte_carlo_validation",
            "step14_ab_testing",
            "step15_saving"
        ]
        
        for step_name in step_names:
            try:
                latest_report = await get_latest_step_report(step_name)
                if latest_report:
                    pipeline_report["steps"][step_name] = latest_report
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to collect report for {step_name}: {e}")
    
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
        
        # Step summary
        steps = pipeline_report.get("steps", {})
        summary.append("Step Execution Summary:")
        summary.append("-" * 50)
        
        step_statuses = {
            "success": [],
            "failed": [],
            "skipped": []
        }
        
        for step_name, step_report in steps.items():
            status = step_report.get("status", "unknown")
            if status == "success":
                step_statuses["success"].append(step_name)
            elif status == "failed":
                step_statuses["failed"].append(step_name)
            else:
                step_statuses["skipped"].append(step_name)
        
        summary.append(f"✅ Successful Steps ({len(step_statuses['success'])}):")
        for step in step_statuses["success"]:
            summary.append(f"  - {step}")
        
        if step_statuses["failed"]:
            summary.append(f"❌ Failed Steps ({len(step_statuses['failed'])}):")
            for step in step_statuses["failed"]:
                summary.append(f"  - {step}")
        
        if step_statuses["skipped"]:
            summary.append(f"⏭️ Skipped Steps ({len(step_statuses['skipped'])}):")
            for step in step_statuses["skipped"]:
                summary.append(f"  - {step}")
        
        summary.append("")
        
        # Performance summary
        summary.append("Performance Summary:")
        summary.append("-" * 50)
        
        total_duration = 0
        for step_name, step_report in steps.items():
            if step_report.get("performance_metrics"):
                duration = step_report["performance_metrics"].get("execution_time_seconds", 0)
                total_duration += duration
                summary.append(f"  {step_name}: {duration:.2f}s")
        
        summary.append(f"  Total Pipeline Duration: {total_duration:.2f}s")
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
    
    async def get_pipeline_reports(self, symbol: str = None, exchange: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve pipeline reports with optional filtering."""
        
        try:
            # Get all step reports
            all_reports = await get_step_reports(limit=limit * 10)  # Get more to filter
            
            # Filter by symbol and exchange if provided
            if symbol or exchange:
                filtered_reports = []
                for report in all_reports:
                    # Extract symbol and exchange from step name or report data
                    # This is a simplified implementation
                    if symbol and symbol.lower() in report.get("step_name", "").lower():
                        filtered_reports.append(report)
                    elif exchange and exchange.lower() in report.get("step_name", "").lower():
                        filtered_reports.append(report)
                    else:
                        filtered_reports.append(report)
                
                return filtered_reports[:limit]
            
            return all_reports[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve pipeline reports: {e}")
            return []
    
    async def get_step_performance_summary(self, step_name: str = None) -> Dict[str, Any]:
        """Get performance summary for a specific step or all steps."""
        
        try:
            reports = await get_step_reports(step_name, limit=100)
            
            if not reports:
                return {"error": "No reports found"}
            
            # Calculate performance statistics
            durations = []
            success_count = 0
            failure_count = 0
            
            for report in reports:
                if report.get("performance_metrics"):
                    duration = report["performance_metrics"].get("execution_time_seconds", 0)
                    durations.append(duration)
                
                if report.get("status") == "success":
                    success_count += 1
                elif report.get("status") == "failed":
                    failure_count += 1
            
            summary = {
                "total_executions": len(reports),
                "success_count": success_count,
                "failure_count": failure_count,
                "success_rate": success_count / len(reports) if reports else 0,
                "average_duration": sum(durations) / len(durations) if durations else 0,
                "min_duration": min(durations) if durations else 0,
                "max_duration": max(durations) if durations else 0,
                "latest_execution": reports[0] if reports else None
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get performance summary: {e}")
            return {"error": str(e)}


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