#!/usr/bin/env python3
"""
Ares Launcher - Granular Sub-Pipeline Control

This launcher provides granular control over training pipeline execution,
allowing users to execute specific sub-pipelines at different stages with
full, light, or blank execution modes.

Key Features:
- Granular sub-pipeline control
- Multiple execution modes (full, light, blank)
- Stage-specific execution
- Sub-pipeline-specific execution
- Comprehensive monitoring and reporting
- Mid-function artifact creation
- Real-time progress tracking
"""

import asyncio
import json
import logging
import os
import sys
import argparse
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path
from enum import Enum

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from src.training.steps.main_training_pipeline import (
    MainTrainingPipeline, MainPipelineConfig, MainPipelineResult,
    PipelineStage, ExecutionMode, get_full_pipeline_config, 
    get_light_pipeline_config, get_blank_pipeline_config
)

logger = system_logger.getChild('AresLauncher')

class LauncherMode(Enum):
    """Launcher execution modes."""
    FULL = "full"          # Complete pipeline execution
    LIGHT = "light"        # Lightweight execution
    BLANK = "blank"        # Minimal execution for testing
    STAGE = "stage"        # Execute specific stage
    SUB_PIPELINE = "sub_pipeline"  # Execute specific sub-pipeline

class AresLauncher:
    """
    Ares Launcher with Granular Sub-Pipeline Control.
    
    Provides comprehensive control over training pipeline execution with
    granular sub-pipeline management and real-time monitoring.
    """
    
    def __init__(self):
        """Initialize the Ares launcher."""
        self.logger = logger.getChild('AresLauncher')
        self.pipeline = MainTrainingPipeline()
        self.current_execution: Optional[MainPipelineResult] = None
        self.execution_history: List[MainPipelineResult] = []
        
        # Initialize monitoring
        self._setup_logging()
        self._setup_monitoring()
    
    def _setup_logging(self):
        """Setup comprehensive logging."""
        self.logger.info("🚀 Ares Launcher Initialized")
        self.logger.info("=" * 80)
        self.logger.info("🎯 Granular Sub-Pipeline Control Enabled")
        self.logger.info("=" * 80)
    
    def _setup_monitoring(self):
        """Setup monitoring and progress tracking."""
        self.monitoring_enabled = True
        self.progress_callbacks: List[callable] = []
        
        # Register default progress callback
        self.register_progress_callback(self._default_progress_callback)
    
    def register_progress_callback(self, callback: callable):
        """Register a progress callback function."""
        self.progress_callbacks.append(callback)
    
    def _default_progress_callback(self, progress_data: Dict[str, Any]):
        """Default progress callback for monitoring."""
        self.logger.info(f"📊 Progress: {progress_data.get('message', 'Unknown')}")
    
    async def execute_pipeline(
        self,
        mode: LauncherMode = LauncherMode.FULL,
        symbol: str = "BTCUSDT",
        exchange: str = "binance",
        timeframe: str = "1m",
        data_dir: str = "data/training",
        stage: Optional[PipelineStage] = None,
        sub_pipeline: Optional[str] = None,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> MainPipelineResult:
        """
        Execute the training pipeline with granular control.
        
        Args:
            mode: Execution mode
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe
            data_dir: Data directory
            stage: Specific stage to execute (for STAGE mode)
            sub_pipeline: Specific sub-pipeline to execute (for SUB_PIPELINE mode)
            custom_config: Custom configuration parameters
            
        Returns:
            MainPipelineResult with execution details
        """
        self.logger.info(f"🚀 Starting pipeline execution: {mode.value}")
        
        # Create configuration based on mode
        config = self._create_config(
            mode, symbol, exchange, timeframe, data_dir, 
            stage, sub_pipeline, custom_config
        )
        
        # Execute based on mode
        if mode == LauncherMode.SUB_PIPELINE and sub_pipeline:
            return await self._execute_sub_pipeline(sub_pipeline, config)
        elif mode == LauncherMode.STAGE and stage:
            return await self._execute_stage(stage, config)
        else:
            return await self._execute_full_pipeline(config)
    
    def _create_config(
        self,
        mode: LauncherMode,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        stage: Optional[PipelineStage],
        sub_pipeline: Optional[str],
        custom_config: Optional[Dict[str, Any]]
    ) -> MainPipelineConfig:
        """Create pipeline configuration based on mode and parameters."""
        
        # Base configuration
        base_config = {
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': timeframe,
            'data_dir': data_dir,
            'custom_params': custom_config or {}
        }
        
        # Mode-specific configuration
        if mode == LauncherMode.FULL:
            config = get_full_pipeline_config(**base_config)
        elif mode == LauncherMode.LIGHT:
            config = get_light_pipeline_config(**base_config)
        elif mode == LauncherMode.BLANK:
            config = get_blank_pipeline_config(**base_config)
        elif mode == LauncherMode.STAGE and stage:
            config = self._create_stage_config(stage, base_config)
        elif mode == LauncherMode.SUB_PIPELINE and sub_pipeline:
            config = self._create_sub_pipeline_config(sub_pipeline, base_config)
        else:
            # Default to full configuration
            config = get_full_pipeline_config(**base_config)
        
        return config
    
    def _create_stage_config(self, stage: PipelineStage, base_config: Dict[str, Any]) -> MainPipelineConfig:
        """Create configuration for a specific stage."""
        config = get_full_pipeline_config(**base_config)
        
        # Enable only the specified stage
        config.enabled_stages = [stage]
        
        # Get all available sub-pipelines for the stage
        available_sub_pipelines = self.pipeline.get_available_sub_pipelines(stage)
        config.enabled_sub_pipelines[stage] = available_sub_pipelines
        
        return config
    
    def _create_sub_pipeline_config(self, sub_pipeline: str, base_config: Dict[str, Any]) -> MainPipelineConfig:
        """Create configuration for a specific sub-pipeline."""
        config = get_full_pipeline_config(**base_config)
        
        # Find which stage contains the sub-pipeline
        target_stage = None
        for stage in PipelineStage:
            available_sub_pipelines = self.pipeline.get_available_sub_pipelines(stage)
            if sub_pipeline in available_sub_pipelines:
                target_stage = stage
                break
        
        if not target_stage:
            raise ValueError(f"Sub-pipeline '{sub_pipeline}' not found in any stage")
        
        # Enable only the target stage and sub-pipeline
        config.enabled_stages = [target_stage]
        config.enabled_sub_pipelines[target_stage] = [sub_pipeline]
        
        return config
    
    async def _execute_full_pipeline(self, config: MainPipelineConfig) -> MainPipelineResult:
        """Execute the full pipeline."""
        self.logger.info("🎯 Executing full pipeline")
        
        # Create mid-function artifacts
        artifacts = await self._create_mid_function_artifacts(config)
        
        # Execute pipeline
        result = await self.pipeline.execute_pipeline(config)
        
        # Store execution
        self.current_execution = result
        self.execution_history.append(result)
        
        # Log results
        self._log_execution_results(result)
        
        return result
    
    async def _execute_stage(self, stage: PipelineStage, config: MainPipelineConfig) -> MainPipelineResult:
        """Execute a specific stage."""
        self.logger.info(f"🎯 Executing stage: {stage.value}")
        
        # Create mid-function artifacts for the stage
        artifacts = await self._create_stage_artifacts(stage, config)
        
        # Execute only the specified stage
        result = await self.pipeline.execute_pipeline(config)
        
        # Store execution
        self.current_execution = result
        self.execution_history.append(result)
        
        # Log results
        self._log_execution_results(result)
        
        return result
    
    async def _execute_sub_pipeline(self, sub_pipeline: str, config: MainPipelineConfig) -> MainPipelineResult:
        """Execute a specific sub-pipeline."""
        self.logger.info(f"🎯 Executing sub-pipeline: {sub_pipeline}")
        
        # Create mid-function artifacts for the sub-pipeline
        artifacts = await self._create_sub_pipeline_artifacts(sub_pipeline, config)
        
        # Execute only the specified sub-pipeline
        result = await self.pipeline.execute_pipeline(config)
        
        # Store execution
        self.current_execution = result
        self.execution_history.append(result)
        
        # Log results
        self._log_execution_results(result)
        
        return result
    
    async def _create_mid_function_artifacts(self, config: MainPipelineConfig) -> Dict[str, Any]:
        """Create mid-function artifacts for full pipeline execution."""
        self.logger.info("🔧 Creating mid-function artifacts for full pipeline")
        
        artifacts = {
            'pipeline_config': {
                'mode': config.mode.value,
                'symbol': config.symbol,
                'exchange': config.exchange,
                'timeframe': config.timeframe,
                'enabled_stages': [stage.value for stage in config.enabled_stages],
                'enabled_sub_pipelines': {
                    stage.value: sub_pipelines 
                    for stage, sub_pipelines in config.enabled_sub_pipelines.items()
                }
            },
            'execution_metadata': {
                'start_time': datetime.now().isoformat(),
                'pipeline_id': f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'execution_mode': 'full_pipeline'
            },
            'monitoring_config': {
                'progress_tracking': True,
                'artifact_creation': True,
                'real_time_monitoring': True
            }
        }
        
        # Save artifacts
        await self._save_artifacts(artifacts, 'full_pipeline_artifacts.json')
        
        return artifacts
    
    async def _create_stage_artifacts(self, stage: PipelineStage, config: MainPipelineConfig) -> Dict[str, Any]:
        """Create mid-function artifacts for stage execution."""
        self.logger.info(f"🔧 Creating mid-function artifacts for stage: {stage.value}")
        
        artifacts = {
            'stage_config': {
                'stage': stage.value,
                'mode': config.mode.value,
                'symbol': config.symbol,
                'exchange': config.exchange,
                'timeframe': config.timeframe,
                'enabled_sub_pipelines': config.enabled_sub_pipelines.get(stage, [])
            },
            'execution_metadata': {
                'start_time': datetime.now().isoformat(),
                'pipeline_id': f"stage_{stage.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'execution_mode': 'stage_execution'
            },
            'stage_info': {
                'available_sub_pipelines': self.pipeline.get_available_sub_pipelines(stage),
                'stage_description': self._get_stage_description(stage)
            }
        }
        
        # Save artifacts
        await self._save_artifacts(artifacts, f'{stage.value}_artifacts.json')
        
        return artifacts
    
    async def _create_sub_pipeline_artifacts(self, sub_pipeline: str, config: MainPipelineConfig) -> Dict[str, Any]:
        """Create mid-function artifacts for sub-pipeline execution."""
        self.logger.info(f"🔧 Creating mid-function artifacts for sub-pipeline: {sub_pipeline}")
        
        # Find the stage containing this sub-pipeline
        target_stage = None
        for stage in PipelineStage:
            if sub_pipeline in config.enabled_sub_pipelines.get(stage, []):
                target_stage = stage
                break
        
        artifacts = {
            'sub_pipeline_config': {
                'sub_pipeline': sub_pipeline,
                'stage': target_stage.value if target_stage else 'unknown',
                'mode': config.mode.value,
                'symbol': config.symbol,
                'exchange': config.exchange,
                'timeframe': config.timeframe
            },
            'execution_metadata': {
                'start_time': datetime.now().isoformat(),
                'pipeline_id': f"sub_pipeline_{sub_pipeline}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'execution_mode': 'sub_pipeline_execution'
            },
            'sub_pipeline_info': {
                'description': self._get_sub_pipeline_description(sub_pipeline),
                'dependencies': self._get_sub_pipeline_dependencies(sub_pipeline),
                'outputs': self._get_sub_pipeline_outputs(sub_pipeline)
            }
        }
        
        # Save artifacts
        await self._save_artifacts(artifacts, f'{sub_pipeline}_artifacts.json')
        
        return artifacts
    
    async def _save_artifacts(self, artifacts: Dict[str, Any], filename: str):
        """Save artifacts to file."""
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        
        artifacts_file = artifacts_dir / filename
        with open(artifacts_file, 'w') as f:
            json.dump(artifacts, f, indent=2, default=str)
        
        self.logger.info(f"💾 Artifacts saved to: {artifacts_file}")
    
    def _get_stage_description(self, stage: PipelineStage) -> str:
        """Get description for a pipeline stage."""
        descriptions = {
            PipelineStage.DATA_COLLECTION: "Data collection and preparation stage",
            PipelineStage.MARKET_ANALYSIS: "Market analysis and regime detection stage",
            PipelineStage.MODEL_TRAINING: "Model training and validation stage",
            PipelineStage.BACKTESTING: "Backtesting and optimization stage"
        }
        return descriptions.get(stage, "Unknown stage")
    
    def _get_sub_pipeline_description(self, sub_pipeline: str) -> str:
        """Get description for a sub-pipeline."""
        descriptions = {
            # Data Collection
            'data_download': "Download raw data from exchanges",
            'data_conversion': "Convert data formats and standardize",
            'data_validation': "Validate data quality and integrity",
            'data_preparation': "Prepare data for further processing",
            'feature_engineering': "Basic feature engineering",
            'data_quality_check': "Comprehensive quality assessment",
            'data_storage': "Store processed data",
            'data_monitoring': "Monitor data collection process",
            'data_integration': "Integrate multiple data sources",
            'data_export': "Export data in various formats",
            
            # Market Analysis
            'sr_detection': "Detect Support/Resistance levels",
            'sr_clustering': "Generate SR clusters",
            'sr_ml_learning': "ML-based learning for SR clusters",
            'hmm_clustering': "HMM-based regime clustering",
            'hmm_regime_discovery': "Discover market regimes",
            'regime_data_splitting': "Split data by regimes",
            'triple_barrier_labeling': "Apply triple barrier method",
            'feature_lookback_optimization': "Optimize feature lookback periods",
            'fractional_differentiation': "Apply fractional differentiation",
            'cross_timeframe_analysis': "Cross timeframe interaction features",
            
            # Model Training
            'general_model_training': "Train general ML models",
            'analyst_model_training': "Train analyst-specific models",
            'tactician_model_training': "Train tactician-specific models",
            'hmm_training': "HMM-based model training",
            'ensemble_training': "Ensemble model training",
            'multi_timeframe_training': "Multi-timeframe model training",
            'regime_specific_training': "Regime-specific model training",
            'model_validation': "Model validation and testing",
            'model_persistence': "Save and load models",
            'model_evaluation': "Comprehensive model evaluation",
            
            # Backtesting
            'walk_forward_validation': "Walk-forward backtesting",
            'monte_carlo_simulation': "Monte Carlo backtesting",
            'ab_testing': "A/B testing for strategies",
            'model_persistence': "Save and load models",
            'final_parameters_optimization': "System-wide parameter optimization",
            'performance_analytics': "Performance analysis and reporting",
            'risk_analysis': "Risk metrics and analysis",
            'trade_analysis': "Trade-level analysis",
            'portfolio_analysis': "Portfolio-level analysis",
            'reporting': "Comprehensive reporting"
        }
        return descriptions.get(sub_pipeline, "Unknown sub-pipeline")
    
    def _get_sub_pipeline_dependencies(self, sub_pipeline: str) -> List[str]:
        """Get dependencies for a sub-pipeline."""
        dependencies = {
            'data_conversion': ['data_download'],
            'data_validation': ['data_download', 'data_conversion'],
            'data_preparation': ['data_validation'],
            'feature_engineering': ['data_preparation'],
            'data_quality_check': ['feature_engineering'],
            'sr_clustering': ['sr_detection'],
            'sr_ml_learning': ['sr_clustering'],
            'hmm_regime_discovery': ['hmm_clustering'],
            'regime_data_splitting': ['hmm_regime_discovery'],
            'triple_barrier_labeling': ['regime_data_splitting'],
            'feature_lookback_optimization': ['triple_barrier_labeling'],
            'model_validation': ['general_model_training', 'analyst_model_training', 'tactician_model_training'],
            'model_persistence': ['model_validation'],
            'walk_forward_validation': ['model_persistence'],
            'monte_carlo_simulation': ['walk_forward_validation'],
            'final_parameters_optimization': ['monte_carlo_simulation'],
            'performance_analytics': ['final_parameters_optimization'],
            'reporting': ['performance_analytics']
        }
        return dependencies.get(sub_pipeline, [])
    
    def _get_sub_pipeline_outputs(self, sub_pipeline: str) -> List[str]:
        """Get expected outputs for a sub-pipeline."""
        outputs = {
            'data_download': ['raw_data.parquet'],
            'data_conversion': ['converted_data.parquet'],
            'data_validation': ['validation_report.json'],
            'data_preparation': ['prepared_data.parquet'],
            'feature_engineering': ['features.parquet'],
            'data_quality_check': ['quality_report.json'],
            'sr_detection': ['sr_levels.json'],
            'sr_clustering': ['sr_clusters.json'],
            'hmm_regime_discovery': ['regime_assignments.parquet'],
            'triple_barrier_labeling': ['labels.parquet'],
            'general_model_training': ['general_model.pkl'],
            'analyst_model_training': ['analyst_model.pkl'],
            'tactician_model_training': ['tactician_model.pkl'],
            'model_validation': ['validation_results.json'],
            'walk_forward_validation': ['backtest_results.json'],
            'monte_carlo_simulation': ['mc_results.json'],
            'final_parameters_optimization': ['optimized_parameters.json'],
            'performance_analytics': ['performance_report.json'],
            'reporting': ['comprehensive_report.pdf']
        }
        return outputs.get(sub_pipeline, [])
    
    def _log_execution_results(self, result: MainPipelineResult):
        """Log execution results."""
        self.logger.info("=" * 80)
        self.logger.info("📊 PIPELINE EXECUTION RESULTS")
        self.logger.info("=" * 80)
        self.logger.info(f"Pipeline ID: {result.pipeline_id}")
        self.logger.info(f"Status: {result.status.value}")
        self.logger.info(f"Duration: {result.duration_seconds:.2f}s")
        self.logger.info(f"Total Sub-pipelines: {result.total_sub_pipelines}")
        self.logger.info(f"Completed: {result.completed_sub_pipelines}")
        self.logger.info(f"Failed: {result.failed_sub_pipelines}")
        self.logger.info(f"Success Rate: {result.success_rate:.2%}")
        
        if result.failed_stages:
            self.logger.warning(f"Failed Stages: {[stage.value for stage in result.failed_stages]}")
        
        if result.error_message:
            self.logger.error(f"Error: {result.error_message}")
        
        self.logger.info("=" * 80)
    
    def get_execution_history(self) -> List[MainPipelineResult]:
        """Get execution history."""
        return self.execution_history
    
    def get_current_execution(self) -> Optional[MainPipelineResult]:
        """Get current execution status."""
        return self.current_execution
    
    def get_available_stages(self) -> List[str]:
        """Get list of available pipeline stages."""
        return [stage.value for stage in PipelineStage]
    
    def get_available_sub_pipelines(self, stage: Optional[str] = None) -> Dict[str, List[str]]:
        """Get available sub-pipelines for stages."""
        if stage:
            stage_enum = PipelineStage(stage)
            return {stage: self.pipeline.get_available_sub_pipelines(stage_enum)}
        else:
            return {
                stage.value: self.pipeline.get_available_sub_pipelines(stage)
                for stage in PipelineStage
            }
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all executions."""
        return self.pipeline.get_execution_summary()

# CLI Interface
def create_cli_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Ares Launcher - Granular Sub-Pipeline Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline execution
  python ares_launcher.py --mode full --symbol BTCUSDT --exchange binance

  # Light pipeline execution
  python ares_launcher.py --mode light --symbol ETHUSDT

  # Execute specific stage
  python ares_launcher.py --mode stage --stage data_collection --symbol BTCUSDT

  # Execute specific sub-pipeline
  python ares_launcher.py --mode sub_pipeline --sub_pipeline sr_detection --symbol BTCUSDT

  # Blank mode for testing
  python ares_launcher.py --mode blank --symbol BTCUSDT
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['full', 'light', 'blank', 'stage', 'sub_pipeline'],
        default='full',
        help='Execution mode (default: full)'
    )
    
    parser.add_argument(
        '--symbol',
        default='BTCUSDT',
        help='Trading symbol (default: BTCUSDT)'
    )
    
    parser.add_argument(
        '--exchange',
        default='binance',
        help='Exchange name (default: binance)'
    )
    
    parser.add_argument(
        '--timeframe',
        default='1m',
        help='Data timeframe (default: 1m)'
    )
    
    parser.add_argument(
        '--data-dir',
        default='data/training',
        help='Data directory (default: data/training)'
    )
    
    parser.add_argument(
        '--stage',
        choices=['data_collection', 'market_analysis', 'model_training', 'backtesting'],
        help='Specific stage to execute (for stage mode)'
    )
    
    parser.add_argument(
        '--sub-pipeline',
        help='Specific sub-pipeline to execute (for sub_pipeline mode)'
    )
    
    parser.add_argument(
        '--config',
        help='Path to custom configuration file (JSON)'
    )
    
    parser.add_argument(
        '--list-stages',
        action='store_true',
        help='List available pipeline stages'
    )
    
    parser.add_argument(
        '--list-sub-pipelines',
        help='List available sub-pipelines for a stage'
    )
    
    return parser

async def main():
    """Main entry point."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Initialize launcher
    launcher = AresLauncher()
    
    # Handle list commands
    if args.list_stages:
        stages = launcher.get_available_stages()
        print("Available Pipeline Stages:")
        for stage in stages:
            print(f"  - {stage}")
        return
    
    if args.list_sub_pipelines:
        sub_pipelines = launcher.get_available_sub_pipelines(args.list_sub_pipelines)
        print(f"Available Sub-pipelines for {args.list_sub_pipelines}:")
        for stage, pipelines in sub_pipelines.items():
            print(f"  {stage}:")
            for pipeline in pipelines:
                print(f"    - {pipeline}")
        return
    
    # Load custom configuration if provided
    custom_config = None
    if args.config:
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
    
    # Convert string mode to enum
    mode_map = {
        'full': LauncherMode.FULL,
        'light': LauncherMode.LIGHT,
        'blank': LauncherMode.BLANK,
        'stage': LauncherMode.STAGE,
        'sub_pipeline': LauncherMode.SUB_PIPELINE
    }
    mode = mode_map[args.mode]
    
    # Convert string stage to enum if provided
    stage = None
    if args.stage:
        stage_map = {
            'data_collection': PipelineStage.DATA_COLLECTION,
            'market_analysis': PipelineStage.MARKET_ANALYSIS,
            'model_training': PipelineStage.MODEL_TRAINING,
            'backtesting': PipelineStage.BACKTESTING
        }
        stage = stage_map[args.stage]
    
    # Execute pipeline
    try:
        result = await launcher.execute_pipeline(
            mode=mode,
            symbol=args.symbol,
            exchange=args.exchange,
            timeframe=args.timeframe,
            data_dir=args.data_dir,
            stage=stage,
            sub_pipeline=args.sub_pipeline,
            custom_config=custom_config
        )
        
        # Print final results
        print("\n" + "=" * 80)
        print("🎯 EXECUTION COMPLETED")
        print("=" * 80)
        print(f"Status: {result.status.value}")
        print(f"Duration: {result.duration_seconds:.2f}s")
        print(f"Success Rate: {result.success_rate:.2%}")
        print("=" * 80)
        
        if result.status.value == 'failed':
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())