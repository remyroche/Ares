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

# Add the project root to the Python path BEFORE any imports
print("🔧 [IMPORTS] Setting up project root path...")
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
print(f"✅ [IMPORTS] Project root added to path: {project_root}")

# Temporarily use simple logger to bypass initialization issues
print("🔧 [IMPORTS] Setting up additional paths...")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print("✅ [IMPORTS] Additional paths configured")

print("🔧 [IMPORTS] Importing simple_logger...")
from src.utils.logger import system_logger
print("✅ [IMPORTS] Simple logger imported")

print("🔧 [IMPORTS] Importing tprint...")
from src.utils.tprint import tprint
print("✅ [IMPORTS] Tprint imported")

from src.training.steps.main_training_pipeline import SubPipelineStatus
print("✅ [IMPORTS] SubPipelineStatus imported")

tprint("🔧 [IMPORTS] Importing core decorators...")
from src.core.decorators import handles_errors, traced, log_execution_time
tprint("✅ [IMPORTS] Core decorators imported")


tprint("🔧 [IMPORTS] Importing main training pipeline components...")
from src.training.steps.main_training_pipeline import (
    MainTrainingPipeline, MainPipelineConfig, MainPipelineResult,
    PipelineStage, ExecutionMode, DirectionType, get_full_pipeline_config,
    get_light_pipeline_config, get_blank_pipeline_config, SubPipelineStatus
)
tprint("✅ [IMPORTS] Main training pipeline components imported")

tprint("🔧 [IMPORTS] Creating AresLauncher logger...")
logger = system_logger.getChild('AresLauncher')
# Ensure single emission via root 'AresSimple' only; do not add handlers here
logger.propagate = True
if logger.handlers:
    logger.handlers.clear()
tprint("✅ [IMPORTS] AresLauncher logger created")
tprint("✅ [IMPORTS] All imports completed successfully")
tprint("=" * 60)

class LauncherMode(Enum):
    """Launcher execution modes."""
    FULL = "full"          # Complete pipeline execution
    LIGHT = "light"        # Lightweight execution
    BLANK = "blank"        # Minimal execution for testing
    STAGE = "stage"        # Execute specific stage
    SUB_PIPELINE = "sub_pipeline"  # Execute specific sub-pipeline

class ExecutionModeType(Enum):
    """Execution mode types for stage/sub-pipeline specific execution."""
    FULL = "full"          # Complete execution with all features
    LIGHT = "light"        # Lightweight execution with essential features only
    BLANK = "blank"        # Minimal execution for testing/validation

class AresLauncher:
    """
    Ares Launcher with Granular Sub-Pipeline Control.
    
    Provides comprehensive control over training pipeline execution with
    granular sub-pipeline management and real-time monitoring.
    """
    
    def __init__(self):
        """Initialize the Ares launcher."""
        tprint("🚀 [INIT] Starting AresLauncher initialization...")
        tprint("🚀 [INIT] Creating logger instance...")
        self.logger = logger.getChild('AresLauncher')
        tprint("✅ [INIT] Logger created successfully")
        
        tprint("🚀 [INIT] Initializing MainTrainingPipeline...")
        self.pipeline = MainTrainingPipeline()
        tprint("✅ [INIT] MainTrainingPipeline initialized successfully")
        
        tprint("🚀 [INIT] Setting up execution tracking...")
        self.current_execution: Optional[MainPipelineResult] = None
        self.execution_history: List[MainPipelineResult] = []
        tprint("✅ [INIT] Execution tracking setup complete")
        
        # Initialize monitoring
        tprint("🚀 [INIT] Starting logging setup...")
        self._setup_logging()
        tprint("✅ [INIT] Logging setup complete")
        
        tprint("🚀 [INIT] Starting monitoring setup...")
        self._setup_monitoring()
        tprint("✅ [INIT] Monitoring setup complete")
        
        tprint("🎯 [INIT] AresLauncher initialization completed successfully!")
        tprint("=" * 80)
    
    def _setup_logging(self):
        """Setup comprehensive logging."""
        tprint("🔧 [SETUP_LOGGING] Starting logging configuration...")
        tprint("🔧 [SETUP_LOGGING] Configuring logger formatters...")
        
        # Keep light verbosity in LIGHT mode
        self.logger.info("🚀 Ares Launcher Initialized")
        self.logger.info("🎯 Granular Sub-Pipeline Control Enabled")
        
        tprint("🔧 [SETUP_LOGGING] Logger configuration complete")
        tprint("🔧 [SETUP_LOGGING] Logging levels configured")
        tprint("🔧 [SETUP_LOGGING] Console output enabled")
        tprint("🔧 [SETUP_LOGGING] File output configured")
        tprint("✅ [SETUP_LOGGING] Comprehensive logging setup completed")
    
    def _setup_monitoring(self):
        """Setup monitoring and progress tracking."""
        tprint("📊 [SETUP_MONITORING] Starting monitoring configuration...")
        tprint("📊 [SETUP_MONITORING] Enabling monitoring system...")
        self.monitoring_enabled = True
        tprint("✅ [SETUP_MONITORING] Monitoring system enabled")
        
        tprint("📊 [SETUP_MONITORING] Initializing progress callbacks list...")
        self.progress_callbacks: List[callable] = []
        tprint("✅ [SETUP_MONITORING] Progress callbacks list initialized")
        
        # Register default progress callback
        tprint("📊 [SETUP_MONITORING] Registering default progress callback...")
        self.register_progress_callback(self._default_progress_callback)
        tprint("✅ [SETUP_MONITORING] Default progress callback registered")
        tprint("✅ [SETUP_MONITORING] Monitoring setup completed successfully")
    
    def register_progress_callback(self, callback: callable):
        """Register a progress callback function."""
        self.progress_callbacks.append(callback)
    
    def _default_progress_callback(self, progress_data: Dict[str, Any]):
        """Default progress callback for monitoring."""
        self.logger.info(f"📊 Progress: {progress_data.get('message', 'Unknown')}")
    
    def _log_stage_transition(self, from_stage: Optional[str], to_stage: str, transition_type: str = "STAGE"):
        """Log explicit stage/pipeline transitions."""
        if from_stage:
            self.logger.info("=" * 80)
            self.logger.info(f"🔄 TRANSITION: {from_stage} → {to_stage}")
            self.logger.info(f"📋 Transition Type: {transition_type}")
            self.logger.info(f"⏰ Timestamp: {datetime.now().isoformat()}")
            self.logger.info("=" * 80)
        else:
            self.logger.info("=" * 80)
            self.logger.info(f"🚀 STARTING: {to_stage}")
            self.logger.info(f"📋 Execution Type: {transition_type}")
            self.logger.info(f"⏰ Timestamp: {datetime.now().isoformat()}")
            self.logger.info("=" * 80)
    
    def _log_sub_pipeline_transition(self, from_sub_pipeline: Optional[str], to_sub_pipeline: str, stage: str):
        """Log explicit sub-pipeline transitions."""
        if from_sub_pipeline:
            self.logger.info("=" * 60)
            self.logger.info(f"🔄 SUB-PIPELINE TRANSITION: {from_sub_pipeline} → {to_sub_pipeline}")
            self.logger.info(f"📋 Stage: {stage}")
            self.logger.info(f"⏰ Timestamp: {datetime.now().isoformat()}")
            self.logger.info("=" * 60)
        else:
            self.logger.info("=" * 60)
            self.logger.info(f"🚀 STARTING SUB-PIPELINE: {to_sub_pipeline}")
            self.logger.info(f"📋 Stage: {stage}")
            self.logger.info(f"⏰ Timestamp: {datetime.now().isoformat()}")
            self.logger.info("=" * 60)
    
    async def _create_outcome_file(self, stage: str, sub_pipeline: str, result: Any, config: MainPipelineConfig) -> str:
        """Create outcome file for stage/sub-pipeline completion."""
        outcome_dir = Path("outcomes")
        outcome_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{stage}_{sub_pipeline}_outcome_{timestamp}.json"
        outcome_file = outcome_dir / filename
        
        # Determine actual direction type from metadata if available
        direction_type = 'both'
        if hasattr(result, 'metadata') and result.metadata:
            direction_settings = result.metadata.get('direction_settings', {})
            enable_long = direction_settings.get('enable_long_positions', True)
            enable_short = direction_settings.get('enable_short_positions', False)
            if enable_long and enable_short:
                direction_type = 'both'
            elif enable_long:
                direction_type = 'long'
            elif enable_short:
                direction_type = 'short'
        elif hasattr(config, 'direction_type'):
            direction_type = config.direction_type.value
        
        outcome_data = {
            'stage': stage,
            'sub_pipeline': sub_pipeline,
            'timestamp': datetime.now().isoformat(),
            'status': result.status.value if hasattr(result, 'status') else 'completed',
            'output_files': result.output_files if hasattr(result, 'output_files') else [],
            'metadata': result.metadata if hasattr(result, 'metadata') else {},
            'artifacts': result.artifacts if hasattr(result, 'artifacts') else {},
            'config': {
                'symbol': config.symbol,
                'exchange': config.exchange,
                'timeframe': config.timeframe,
                'mode': config.mode.value,
                'intensity_percentage': config.intensity_percentage,
                'training_mode_config': config.training_mode_config,
                'direction_type': direction_type
            },
            'next_stage_requirements': self._get_next_stage_requirements(stage, sub_pipeline)
        }
        
        with open(outcome_file, 'w') as f:
            json.dump(outcome_data, f, indent=2, default=str)
        
        self.logger.info(f"💾 Outcome file created: {outcome_file}")
        
        # Create human-readable summary
        summary_file = self._create_human_readable_summary(outcome_file, outcome_data, stage, sub_pipeline)
        if summary_file:
            self.logger.info(f"📄 Human-readable summary created: {summary_file}")
        
        return str(outcome_file)
    
    def _create_human_readable_summary(self, outcome_file: Path, outcome_data: Dict, stage: str, sub_pipeline: str) -> Optional[str]:
        """Create a human-readable summary file from the outcome data."""
        try:
            import pandas as pd
            
            # Create summary filename
            summary_file = Path(str(outcome_file).replace('.json', '_SUMMARY.txt'))
            
            with open(summary_file, 'w') as f:
                # Header
                f.write("=" * 80 + "\n")
                f.write(f"  {sub_pipeline.upper().replace('_', ' ')} - EXECUTION SUMMARY\n")
                f.write("=" * 80 + "\n\n")
                
                # Configuration
                config = outcome_data.get('config', {})
                f.write("📋 CONFIGURATION\n")
                f.write(f"   Symbol:          {config.get('symbol', 'N/A')}\n")
                f.write(f"   Exchange:        {config.get('exchange', 'N/A')}\n")
                f.write(f"   Timeframe:       {config.get('timeframe', 'N/A')}\n")
                f.write(f"   Mode:            {config.get('mode', 'N/A')}\n")
                f.write(f"   Direction:       {config.get('direction_type', 'N/A')}\n")
                f.write(f"   Intensity:       {config.get('intensity_percentage', 'N/A')}\n")
                f.write("\n")
                
                # Status and timing
                f.write("📊 EXECUTION STATUS\n")
                f.write(f"   Status:          {outcome_data.get('status', 'N/A')}\n")
                f.write(f"   Timestamp:       {outcome_data.get('timestamp', 'N/A')}\n")
                f.write("\n")
                
                # Metadata-specific summaries
                metadata = outcome_data.get('metadata', {})
                
                # Feature Lookback Optimization specific
                if sub_pipeline == 'feature_lookback_optimization':
                    f.write("🎯 OPTIMIZATION RESULTS\n")
                    f.write(f"   Status:                  {metadata.get('optimization_status', 'N/A')}\n")
                    f.write(f"   Total Features Optimized: {metadata.get('total_features_optimized', 'N/A')}\n")
                    f.write("\n")
                    
                    # Performance
                    perf = metadata.get('performance_metrics', {})
                    if perf:
                        f.write("⏱️  PERFORMANCE\n")
                        duration = perf.get('total_duration')
                        if duration:
                            f.write(f"   Duration:        {duration:.2f}s ({duration/60:.1f} minutes)\n")
                        memory = perf.get('memory', {})
                        if memory:
                            peak_mb = memory.get('peak_mb')
                            if peak_mb:
                                f.write(f"   Peak Memory:     {peak_mb:.2f} MB\n")
                        f.write("\n")
                    
                    # Feature file
                    artifacts = outcome_data.get('artifacts', {})
                    feature_file = artifacts.get('optimized_features_file')
                    if feature_file:
                        f.write("💾 SAVED FEATURES\n")
                        f.write(f"   File: {Path(feature_file).name}\n")
                        
                        # Try to load feature details
                        if Path(feature_file).exists():
                            try:
                                features_df = pd.read_parquet(feature_file)
                                f.write(f"   Shape:           {features_df.shape[0]:,} rows × {features_df.shape[1]:,} columns\n")
                                f.write(f"   Date Range:      {features_df.index.min().date()} to {features_df.index.max().date()}\n")
                                f.write(f"   File Size:       {Path(feature_file).stat().st_size / 1024 / 1024:.2f} MB\n")
                                f.write(f"\n   📝 Sample Features (first 15):\n")
                                for i, col in enumerate(features_df.columns[:15], 1):
                                    f.write(f"      {i:2d}. {col}\n")
                                if len(features_df.columns) > 15:
                                    f.write(f"      ... +{len(features_df.columns) - 15} more features\n")
                            except Exception as e:
                                f.write(f"   (Could not load feature details: {e})\n")
                        f.write("\n")
                    
                    # Optimization details
                    opt_results = metadata.get('optimization_results', {})
                    if opt_results:
                        feature_results = opt_results.get('feature_results', {})
                        if feature_results:
                            f.write("🔍 FEATURE OPTIMIZATION DETAILS\n")
                            
                            long_features = feature_results.get('long_pipeline', {})
                            short_features = feature_results.get('short_pipeline', {})
                            
                            if long_features or short_features:
                                f.write(f"   Long Direction:  {len(long_features):,} features\n")
                                f.write(f"   Short Direction: {len(short_features):,} features\n")
                                f.write("\n")
                                
                                # Show sample lookbacks
                                if long_features:
                                    f.write("   📊 Sample Optimal Lookbacks (Long Direction):\n")
                                    for i, (fname, finfo) in enumerate(list(long_features.items())[:10], 1):
                                        if isinstance(finfo, dict):
                                            lookback = finfo.get('optimal_lookback', 'null')
                                            score = finfo.get('optimal_score')
                                            f.write(f"      {i:2d}. {fname}\n")
                                            f.write(f"          Lookback: {lookback}")
                                            if score is not None and score != 'null':
                                                f.write(f" | Score: {score:.4f}")
                                            f.write("\n")
                                    if len(long_features) > 10:
                                        f.write(f"          ... +{len(long_features) - 10} more features\n")
                                f.write("\n")
                    
                    # Cache metrics
                    cache_metrics = metadata.get('feature_cache_metrics', {})
                    if cache_metrics:
                        f.write("📦 FEATURE CACHE METRICS\n")
                        f.write(f"   Hits:            {cache_metrics.get('hits', 0)}\n")
                        f.write(f"   Misses:          {cache_metrics.get('misses', 0)}\n")
                        hit_rate = cache_metrics.get('hit_rate')
                        if hit_rate is not None:
                            f.write(f"   Hit Rate:        {hit_rate*100:.1f}%\n")
                        f.write("\n")
                
                # Analyst Profit Labeler specific
                elif sub_pipeline == 'analyst_profit_labeler':
                    f.write("🎯 LABELING RESULTS\n")
                    f.write(f"   Timeframe:               {metadata.get('timeframe', 'N/A')}\n")
                    f.write(f"   Samples:                 {metadata.get('n_samples', 'N/A'):,}\n")
                    f.write(f"   Targets:                 {metadata.get('n_targets', 'N/A')}\n")
                    f.write(f"   Horizons:                {metadata.get('n_horizons', 'N/A')}\n")
                    
                    opps_per_day = metadata.get('opportunities_per_day')
                    if opps_per_day is not None:
                        f.write(f"   Opportunities/Day:       {opps_per_day}\n")
                    
                    direction_settings = metadata.get('direction_settings', {})
                    if direction_settings:
                        f.write(f"   Long Positions:          {'✅ Enabled' if direction_settings.get('enable_long_positions') else '❌ Disabled'}\n")
                        f.write(f"   Short Positions:         {'✅ Enabled' if direction_settings.get('enable_short_positions') else '❌ Disabled'}\n")
                    f.write("\n")
                
                # Generic metadata summary
                else:
                    if metadata:
                        f.write("📊 RESULTS\n")
                        for key, value in list(metadata.items())[:10]:
                            if not isinstance(value, (dict, list)):
                                f.write(f"   {key}: {value}\n")
                        f.write("\n")
                
                # Output files
                output_files = outcome_data.get('output_files', [])
                if output_files:
                    f.write("📁 OUTPUT FILES\n")
                    for output_file in output_files:
                        f.write(f"   • {output_file}\n")
                    f.write("\n")
                
                # Next stage requirements
                next_reqs = outcome_data.get('next_stage_requirements', {})
                if next_reqs:
                    f.write("➡️  NEXT STAGE REQUIREMENTS\n")
                    req_files = next_reqs.get('required_files', [])
                    if req_files:
                        f.write(f"   Required Files: {', '.join(req_files)}\n")
                    req_artifacts = next_reqs.get('required_artifacts', [])
                    if req_artifacts:
                        f.write(f"   Required Artifacts: {', '.join(req_artifacts)}\n")
                    sub_pipelines = next_reqs.get('sub_pipelines', [])
                    if sub_pipelines:
                        f.write(f"   Sub-pipelines: {', '.join(sub_pipelines)}\n")
                    f.write("\n")
                
                # Footer
                f.write("=" * 80 + "\n")
                f.write(f"✅ {sub_pipeline.upper().replace('_', ' ')} COMPLETED SUCCESSFULLY\n")
                f.write("=" * 80 + "\n")
                f.write(f"\n📄 JSON Details: {outcome_file.name}\n")
                f.write(f"📄 This Summary: {summary_file.name}\n")
            
            return str(summary_file)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create human-readable summary: {e}")
            return None
    
    def _get_next_stage_requirements(self, current_stage: str, current_sub_pipeline: str) -> Dict[str, Any]:
        """Get requirements for the next stage/sub-pipeline."""
        requirements = {
            'required_files': [],
            'required_artifacts': [],
            'data_dependencies': []
        }
        
        # Define stage dependencies and requirements
        stage_requirements = {
            'data_collection': {
                'next_stage': 'market_analysis',
                'required_files': ['processed_data.parquet', 'data_quality_report.json', 'exported_data.parquet'],
                'required_artifacts': ['data_metadata', 'quality_metrics', 'integration_results'],
                'sub_pipelines': ['data_download', 'data_conversion', 'data_validation', 'data_preparation', 
                                'feature_engineering', 'data_quality_check', 'data_storage', 'data_monitoring',
                                'data_integration', 'data_export']
            },
            'market_analysis': {
                'next_stage': 'pre_training',
                'required_files': ['sr_levels.json', 'regime_assignments.parquet'],
                'required_artifacts': ['sr_clusters', 'regime_model'],
                'sub_pipelines': ['sr_detection', 'sr_clustering', 'hybrid_nas_tas_regime_discovery', 'nas_tas_clustering', 'regime_models_training', 'regime_ensemble_training',
                                'regime_data_splitting', 'sr_feature_integration']
            },
            'pre_training': {
                'next_stage': 'model_training',
                'required_files': ['labels.parquet', 'features.parquet'],
                'required_artifacts': ['feature_metadata'],
                'sub_pipelines': ['feature_lookback_optimization', 'interactive_feature_generation', 'final_feature_selection']
            },
            'model_training': {
                'next_stage': 'backtesting',
                'required_files': ['analyst_ensemble.pkl', 'tactician_ensemble.pkl', 'analyst_predictions.parquet', 'tactician_predictions.parquet'],
                'required_artifacts': ['analyst_models', 'tactician_models', 'performance_metrics', 'ensemble_models'],
                'sub_pipelines': ['analyst_pre_ml_orchestration', 'analyst_models_training', 'analyst_ensemble_training',
                                'tactician_pre_ml_orchestration', 'tactician_models_training', 'tactician_ensemble_training']
            },
            'backtesting': {
                'next_stage': 'reporting',
                'required_files': ['backtest_results.json', 'performance_report.json', 'final_report.pdf'],
                'required_artifacts': ['trade_analysis', 'risk_metrics', 'portfolio_analysis'],
                'sub_pipelines': ['basic_backtesting_pre', 'final_parameters_optimization', 'basic_backtesting_post', 'walk_forward_validation', 'monte_carlo_simulation', 'ab_testing',
                                'model_persistence', 'performance_analytics',
                                'risk_analysis', 'trade_analysis', 'portfolio_analysis', 'reporting']
            }
        }
        
        if current_stage in stage_requirements:
            requirements.update(stage_requirements[current_stage])
        
        return requirements
    
    async def _check_outcome_files(self, stage: str, sub_pipeline: str) -> Optional[Dict[str, Any]]:
        """Check for existing outcome files from previous stages."""
        outcome_dir = Path("outcomes")
        if not outcome_dir.exists():
            return None
        
        # Look for the most recent outcome file for this stage/sub-pipeline
        pattern = f"{stage}_{sub_pipeline}_outcome_*.json"
        outcome_files = list(outcome_dir.glob(pattern))
        
        if not outcome_files:
            return None
        
        # Get the most recent file
        latest_file = max(outcome_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(latest_file, 'r') as f:
                outcome_data = json.load(f)
            
            self.logger.info(f"📂 Found existing outcome file: {latest_file}")
            return outcome_data
        except Exception as e:
            self.logger.warning(f"⚠️ Could not read outcome file {latest_file}: {e}")
            return None
    
    async def execute_pipeline(
        self,
        mode: LauncherMode = LauncherMode.FULL,
        symbol: str = "ETHUSDT",
        exchange: str = "binance",
        timeframe: str = "15m",
        data_dir: str = "historical_data",
        direction: str = "both",
        stage: Optional[PipelineStage] = None,
        sub_pipeline: Optional[str] = None,
        execution_mode: ExecutionModeType = ExecutionModeType.FULL,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> MainPipelineResult:
        """
        Execute the training pipeline with granular control.

        Args:
            mode: Launcher execution mode (full, light, blank, stage, sub_pipeline)
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe
            data_dir: Data directory
            direction: Direction type for training (longs, shorts, both)
            stage: Specific stage to execute (for STAGE mode)
            sub_pipeline: Specific sub-pipeline to execute (for SUB_PIPELINE mode)
            execution_mode: Execution mode type (full, light, blank) for stage/sub-pipeline specific execution
            custom_config: Custom configuration parameters

        Returns:
            MainPipelineResult with execution details
        """
        tprint("🚀 [EXECUTE_PIPELINE] Starting pipeline execution...")
        tprint(f"🚀 [EXECUTE_PIPELINE] Mode: {mode.value}")
        tprint(f"🚀 [EXECUTE_PIPELINE] Symbol: {symbol}")
        tprint(f"🚀 [EXECUTE_PIPELINE] Exchange: {exchange}")
        tprint(f"🚀 [EXECUTE_PIPELINE] Timeframe: {timeframe}")
        tprint(f"🚀 [EXECUTE_PIPELINE] Data directory: {data_dir}")
        tprint(f"🚀 [EXECUTE_PIPELINE] Direction: {direction}")
        tprint(f"🚀 [EXECUTE_PIPELINE] Execution mode: {execution_mode.value}")

        if stage:
            tprint(f"🚀 [EXECUTE_PIPELINE] Target stage: {stage.value}")
        if sub_pipeline:
            tprint(f"🚀 [EXECUTE_PIPELINE] Target sub-pipeline: {sub_pipeline}")
        if custom_config:
            tprint(f"🚀 [EXECUTE_PIPELINE] Custom config provided: {len(custom_config)} parameters")

        self.logger.info(f"🚀 Starting pipeline execution: {mode.value}")

        # Create configuration based on mode
        tprint("🚀 [EXECUTE_PIPELINE] Creating configuration...")
        config = self._create_config(
            mode, symbol, exchange, timeframe, data_dir, direction,
            stage, sub_pipeline, execution_mode, custom_config
        )
        tprint("✅ [EXECUTE_PIPELINE] Configuration created successfully")
        
        # Execute based on mode
        tprint("🚀 [EXECUTE_PIPELINE] Determining execution path...")
        if mode == LauncherMode.SUB_PIPELINE and sub_pipeline:
            tprint(f"🚀 [EXECUTE_PIPELINE] Executing sub-pipeline: {sub_pipeline}")
            return await self._execute_sub_pipeline(sub_pipeline, config)
        elif mode == LauncherMode.STAGE and stage:
            tprint(f"🚀 [EXECUTE_PIPELINE] Executing stage: {stage.value}")
            return await self._execute_stage(stage, config)
        else:
            tprint("🚀 [EXECUTE_PIPELINE] Executing full pipeline")
            return await self._execute_full_pipeline(config)
    
    def _create_config(
        self,
        mode: LauncherMode,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        direction: str,
        stage: Optional[PipelineStage],
        sub_pipeline: Optional[str],
        execution_mode: ExecutionModeType,
        custom_config: Optional[Dict[str, Any]]
    ) -> MainPipelineConfig:
        """Create pipeline configuration based on mode and parameters."""
        tprint("⚙️ [CREATE_CONFIG] Starting configuration creation...")
        tprint(f"⚙️ [CREATE_CONFIG] Mode: {mode.value}")
        tprint(f"⚙️ [CREATE_CONFIG] Symbol: {symbol}")
        tprint(f"⚙️ [CREATE_CONFIG] Exchange: {exchange}")
        tprint(f"⚙️ [CREATE_CONFIG] Timeframe: {timeframe}")
        tprint(f"⚙️ [CREATE_CONFIG] Data directory: {data_dir}")
        tprint(f"⚙️ [CREATE_CONFIG] Direction: {direction}")
        tprint(f"⚙️ [CREATE_CONFIG] Execution mode: {execution_mode.value}")

        # Convert direction string to enum
        direction_map = {
            'longs': DirectionType.LONGS,
            'shorts': DirectionType.SHORTS,
            'both': DirectionType.BOTH
        }
        direction_enum = direction_map[direction]

        # Base configuration
        tprint("⚙️ [CREATE_CONFIG] Creating base configuration...")
        base_config = {
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': timeframe,
            'data_dir': data_dir,
            'direction_type': direction_enum,
            'custom_params': custom_config or {}
        }
        tprint("✅ [CREATE_CONFIG] Base configuration created")
        
        # Filter base_config to only include supported parameters for each config function
        tprint("⚙️ [CREATE_CONFIG] Filtering configuration parameters...")
        # Note: direction_type is NOT included as it's not a parameter for get_*_pipeline_config functions
        config_function_params = ['symbol', 'exchange', 'timeframe', 'data_dir']
        filtered_config = {k: v for k, v in base_config.items() if k in config_function_params}
        tprint(f"✅ [CREATE_CONFIG] Filtered config: {list(filtered_config.keys())}")
        
        # Mode-specific configuration
        tprint("⚙️ [CREATE_CONFIG] Creating mode-specific configuration...")
        if mode == LauncherMode.FULL:
            tprint("⚙️ [CREATE_CONFIG] Using FULL pipeline configuration")
            config = get_full_pipeline_config(**filtered_config)
        elif mode == LauncherMode.LIGHT:
            tprint("⚙️ [CREATE_CONFIG] Using LIGHT pipeline configuration")
            config = get_light_pipeline_config(**filtered_config)
        elif mode == LauncherMode.BLANK:
            tprint("⚙️ [CREATE_CONFIG] Using BLANK pipeline configuration")
            config = get_blank_pipeline_config(**filtered_config)
        elif mode == LauncherMode.STAGE and stage:
            tprint(f"⚙️ [CREATE_CONFIG] Creating STAGE configuration for: {stage.value}")
            config = self._create_stage_config(stage, base_config, execution_mode, direction)
        elif mode == LauncherMode.SUB_PIPELINE and sub_pipeline:
            tprint(f"⚙️ [CREATE_CONFIG] Creating SUB_PIPELINE configuration for: {sub_pipeline}")
            config = self._create_sub_pipeline_config(sub_pipeline, base_config, execution_mode, direction)
        else:
            # Default to full configuration
            tprint("⚙️ [CREATE_CONFIG] Using DEFAULT (FULL) pipeline configuration")
            config = get_full_pipeline_config(**filtered_config)
        
        tprint("✅ [CREATE_CONFIG] Configuration creation completed successfully")
        return config
    
    def _create_stage_config(self, stage: PipelineStage, base_config: Dict[str, Any], execution_mode: ExecutionModeType, direction: str) -> MainPipelineConfig:
        """Create configuration for a specific stage."""
        tprint(f"🎭 [STAGE_CONFIG] Creating stage configuration for: {stage.value}")
        tprint(f"🎭 [STAGE_CONFIG] Execution mode: {execution_mode.value}")
        tprint(f"🎭 [STAGE_CONFIG] Direction: {direction}")

        # Filter base_config to only include supported parameters for each config function
        tprint("🎭 [STAGE_CONFIG] Filtering configuration parameters...")
        # Note: direction_type is NOT included as it's not a parameter for get_*_pipeline_config functions
        config_function_params = ['symbol', 'exchange', 'timeframe', 'data_dir']
        filtered_config = {k: v for k, v in base_config.items() if k in config_function_params}
        tprint(f"✅ [STAGE_CONFIG] Filtered config: {list(filtered_config.keys())}")
        
        # Use the provided timeframe for all stages
        tprint(f"📊 [STAGE_CONFIG] Using timeframe for {stage.value}: {filtered_config.get('timeframe', '1m')}")

        # Get configuration based on execution mode using centralized configuration
        tprint("🎭 [STAGE_CONFIG] Getting base configuration...")
        if execution_mode == ExecutionModeType.FULL:
            tprint("🎭 [STAGE_CONFIG] Using FULL execution mode configuration")
            config = get_full_pipeline_config(**filtered_config)
        elif execution_mode == ExecutionModeType.LIGHT:
            tprint("🎭 [STAGE_CONFIG] Using LIGHT execution mode configuration")
            config = get_light_pipeline_config(**filtered_config)
        elif execution_mode == ExecutionModeType.BLANK:
            tprint("🎭 [STAGE_CONFIG] Using BLANK execution mode configuration")
            config = get_blank_pipeline_config(**filtered_config)
        else:
            tprint("🎭 [STAGE_CONFIG] Using DEFAULT (FULL) execution mode configuration")
            config = get_full_pipeline_config(**filtered_config)
        
        # Enable only the specified stage
        tprint(f"🎭 [STAGE_CONFIG] Enabling stage: {stage.value}")
        config.enabled_stages = [stage]
        config.single_stage_only = True  # Prevent automatic stage transitions
        tprint("✅ [STAGE_CONFIG] Stage enabled (single stage mode)")
        
        # Get all available sub-pipelines for the stage
        tprint("🎭 [STAGE_CONFIG] Getting available sub-pipelines...")
        available_sub_pipelines = self.pipeline.get_available_sub_pipelines(stage)
        tprint(f"🎭 [STAGE_CONFIG] Found {len(available_sub_pipelines)} sub-pipelines: {available_sub_pipelines}")
        config.enabled_sub_pipelines[stage] = available_sub_pipelines
        tprint("✅ [STAGE_CONFIG] Sub-pipelines configured")
        
        # Add intensity parameters to stage configuration
        tprint("🎭 [STAGE_CONFIG] Adding intensity parameters...")
        if config.training_mode_config:
            config.stage_params[stage] = {
                'intensity_percentage': config.intensity_percentage,
                'training_mode_config': config.training_mode_config,
                'model_training': config.training_mode_config.get('model_training', {}),
                'validation': config.training_mode_config.get('validation', {}),
                'optimization': config.training_mode_config.get('optimization', {})
            }
            tprint("✅ [STAGE_CONFIG] Intensity parameters added")
        else:
            tprint("⚠️ [STAGE_CONFIG] No training mode config available")
        
        tprint("✅ [STAGE_CONFIG] Stage configuration completed successfully")
        return config
    
    def _create_sub_pipeline_config(self, sub_pipeline: str, base_config: Dict[str, Any], execution_mode: ExecutionModeType, direction: str) -> MainPipelineConfig:
        """Create configuration for a specific sub-pipeline."""
        tprint(f"🔧 [SUB_PIPELINE_CONFIG] Creating sub-pipeline configuration for: {sub_pipeline}")
        tprint(f"🔧 [SUB_PIPELINE_CONFIG] Execution mode: {execution_mode.value}")
        tprint(f"🔧 [SUB_PIPELINE_CONFIG] Direction: {direction}")

        # Set the execution mode in base config
        tprint("🔧 [SUB_PIPELINE_CONFIG] Setting execution mode in base config...")
        base_config['mode'] = ExecutionMode(execution_mode.value)
        tprint("✅ [SUB_PIPELINE_CONFIG] Execution mode set")
        
        # Filter base_config to only include supported parameters for each config function
        tprint("🔧 [SUB_PIPELINE_CONFIG] Filtering configuration parameters...")
        supported_params = ['symbol', 'exchange', 'timeframe', 'data_dir', 'start_date', 'end_date']
        filtered_config = {k: v for k, v in base_config.items() if k in supported_params}

        # Apply date filtering for light mode if not already present
        if execution_mode == ExecutionModeType.LIGHT and 'start_date' not in filtered_config:
            tprint("🔧 [SUB_PIPELINE_CONFIG] Applying light mode date filtering...")
            from datetime import datetime, timedelta
            from src.config.pipeline_modes import get_light_mode_config
            import pandas as pd

            mode_config = get_light_mode_config()
            
            # Use last available data date instead of current date
            try:
                from src.utils.data.klines_parquet import KlinesParquetManager
                manager = KlinesParquetManager(data_dir=base_config.get('data_dir', 'historical_data'))
                
                # Get data info to find the actual available date range
                data_info = manager.get_data_info(
                    symbol=base_config.get('symbol', 'ETHUSDT'),
                    interval=base_config.get('timeframe', '15m'),
                    data_type="processed"
                )
                
                if data_info and data_info.get("available") and data_info.get("date_range"):
                    # Use the last date from the available data
                    _, max_date = data_info["date_range"]
                    end_date = pd.to_datetime(max_date)
                    start_date = end_date - timedelta(days=mode_config.lookback_days)
                    tprint(f"✅ Using last available data date: {end_date.strftime('%Y-%m-%d')}")
                else:
                    # Fallback to current date if data info is not available
                    tprint("⚠️ Could not get data info, using current date as fallback")
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=mode_config.lookback_days)
            except Exception as e:
                tprint(f"⚠️ Error detecting last available data date: {e}")
                tprint("⚠️ Falling back to current date")
                end_date = datetime.now()
                start_date = end_date - timedelta(days=mode_config.lookback_days)

            filtered_config['start_date'] = start_date.strftime('%Y-%m-%d')
            filtered_config['end_date'] = end_date.strftime('%Y-%m-%d')

            tprint(f"📅 [SUB_PIPELINE_CONFIG] Light mode date range: {filtered_config['start_date']} to {filtered_config['end_date']}")
            tprint(f"📅 [SUB_PIPELINE_CONFIG] Lookback days: {mode_config.lookback_days}")
        elif execution_mode == ExecutionModeType.BLANK and 'start_date' not in filtered_config:
            tprint("🔧 [SUB_PIPELINE_CONFIG] Applying blank mode date filtering...")
            from datetime import datetime, timedelta
            from src.config.pipeline_modes import get_blank_mode_config
            import pandas as pd

            mode_config = get_blank_mode_config()
            
            # Use last available data date instead of current date
            try:
                from src.utils.data.klines_parquet import KlinesParquetManager
                manager = KlinesParquetManager(data_dir=base_config.get('data_dir', 'historical_data'))
                
                # Get data info to find the actual available date range
                data_info = manager.get_data_info(
                    symbol=base_config.get('symbol', 'ETHUSDT'),
                    interval=base_config.get('timeframe', '15m'),
                    data_type="processed"
                )
                
                if data_info and data_info.get("available") and data_info.get("date_range"):
                    # Use the last date from the available data
                    _, max_date = data_info["date_range"]
                    end_date = pd.to_datetime(max_date)
                    start_date = end_date - timedelta(days=mode_config.lookback_days)
                    tprint(f"✅ Using last available data date: {end_date.strftime('%Y-%m-%d')}")
                else:
                    # Fallback to current date if data info is not available
                    tprint("⚠️ Could not get data info, using current date as fallback")
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=mode_config.lookback_days)
            except Exception as e:
                tprint(f"⚠️ Error detecting last available data date: {e}")
                tprint("⚠️ Falling back to current date")
                end_date = datetime.now()
                start_date = end_date - timedelta(days=mode_config.lookback_days)

            filtered_config['start_date'] = start_date.strftime('%Y-%m-%d')
            filtered_config['end_date'] = end_date.strftime('%Y-%m-%d')

            tprint(f"📅 [SUB_PIPELINE_CONFIG] Blank mode date range: {filtered_config['start_date']} to {filtered_config['end_date']}")
            tprint(f"📅 [SUB_PIPELINE_CONFIG] Lookback days: {mode_config.lookback_days}")

        # SET 15M AS DEFAULT TIMEFRAME FOR NAS-RELATED SUB-PIPELINES
        nas_sub_pipelines = [
            'nas_regime_discovery',     # Discover market regimes using NAS
            'nas_tas_regime_discovery', # Hybrid NAS-TAS regime discovery
            'nas_tas_clustering',       # NAS-TAS regime clustering
            'nas_clustering',           # NAS-based regime clustering
            'nas_tas_models_training',      # Train regime detection models using NAS-TAS regime labels
            'nas_tas_ensemble_training',    # Train ensemble regime detection models using NAS-TAS regime labels
            'regime_models_training',       # Train regime detection models using NAS-TAS regime labels
            'regime_ensemble_training',     # Train ensemble regime detection models using NAS-TAS regime labels
            'nas'                       # Combined NAS regime discovery + clustering
        ]
        
        # Set 15m as default for NAS sub-pipelines
        if sub_pipeline in nas_sub_pipelines:
            original_timeframe = filtered_config.get('timeframe', '1m')
            if original_timeframe == '1m':  # Only override if using default
                filtered_config['timeframe'] = '15m'
                tprint(f"🎯 [SUB_PIPELINE_CONFIG] NAS sub-pipeline detected: {sub_pipeline}")
                tprint(f"🎯 [SUB_PIPELINE_CONFIG] Setting default timeframe: {original_timeframe} → 15m")
                tprint("🎯 [SUB_PIPELINE_CONFIG] Using 15m data for better granularity and regime detection")
            else:
                tprint(f"📊 [SUB_PIPELINE_CONFIG] Using specified timeframe for {sub_pipeline}: {original_timeframe}")
        else:
            tprint(f"📊 [SUB_PIPELINE_CONFIG] Using timeframe for {sub_pipeline}: {filtered_config.get('timeframe', '1m')}")

        tprint(f"✅ [SUB_PIPELINE_CONFIG] Filtered config: {list(filtered_config.keys())}")
        
        # Get configuration based on execution mode
        tprint("🔧 [SUB_PIPELINE_CONFIG] Getting base configuration...")
        if execution_mode == ExecutionModeType.FULL:
            tprint("🔧 [SUB_PIPELINE_CONFIG] Using FULL execution mode configuration")
            config = get_full_pipeline_config(**filtered_config)
        elif execution_mode == ExecutionModeType.LIGHT:
            tprint("🔧 [SUB_PIPELINE_CONFIG] Using LIGHT execution mode configuration")
            # Remove date parameters from filtered_config for light mode since the function doesn't accept them
            light_config = {k: v for k, v in filtered_config.items() if k not in ['start_date', 'end_date']}
            config = get_light_pipeline_config(**light_config)
            # Apply date filtering to the config object
            if 'start_date' in filtered_config and 'end_date' in filtered_config:
                config.start_date = filtered_config['start_date']
                config.end_date = filtered_config['end_date']
                tprint(f"📅 [SUB_PIPELINE_CONFIG] Applied date filtering to light mode config: {config.start_date} to {config.end_date}")
        elif execution_mode == ExecutionModeType.BLANK:
            tprint("🔧 [SUB_PIPELINE_CONFIG] Using BLANK execution mode configuration")
            config = get_blank_pipeline_config(**filtered_config)
        else:
            tprint("🔧 [SUB_PIPELINE_CONFIG] Using DEFAULT (FULL) execution mode configuration")
            config = get_full_pipeline_config(**filtered_config)
        
        # Find which stage contains the sub-pipeline
        tprint("🔧 [SUB_PIPELINE_CONFIG] Finding target stage for sub-pipeline...")
        target_stage = None
        for stage in PipelineStage:
            available_sub_pipelines = self.pipeline.get_available_sub_pipelines(stage)
            if sub_pipeline in available_sub_pipelines:
                target_stage = stage
                tprint(f"🔧 [SUB_PIPELINE_CONFIG] Found sub-pipeline in stage: {stage.value}")
                break
        
        if not target_stage:
            tprint(f"❌ [SUB_PIPELINE_CONFIG] Sub-pipeline '{sub_pipeline}' not found in any stage")
            raise ValueError(f"Sub-pipeline '{sub_pipeline}' not found in any stage")
        
        # Enable only the target stage and sub-pipeline
        tprint(f"🔧 [SUB_PIPELINE_CONFIG] Enabling stage: {target_stage.value}")
        config.enabled_stages = [target_stage]

        stage_sub_pipelines = self.pipeline.get_available_sub_pipelines(target_stage)
        stage_dependency_names: set = set()
        visited_dependencies: set = set()

        def _collect_stage_dependencies(name: str):
            if name in visited_dependencies:
                return
            visited_dependencies.add(name)
            dependencies = self._get_sub_pipeline_dependencies(name)
            for dep in dependencies:
                if dep in stage_dependency_names:
                    continue
                _collect_stage_dependencies(dep)
                if dep in stage_sub_pipelines:
                    stage_dependency_names.add(dep)

        _collect_stage_dependencies(sub_pipeline)

        enabled_sequence = [
            candidate for candidate in stage_sub_pipelines
            if candidate in stage_dependency_names or candidate == sub_pipeline
        ]

        if sub_pipeline not in enabled_sequence:
            enabled_sequence.append(sub_pipeline)

        tprint(f"🔧 [SUB_PIPELINE_CONFIG] Enabling sub-pipelines for stage: {enabled_sequence}")
        config.enabled_sub_pipelines[target_stage] = enabled_sequence
        tprint("✅ [SUB_PIPELINE_CONFIG] Stage and sub-pipeline enabled")
        
        # Set single stage execution mode for individual sub-pipeline execution
        # Enable chaining for SR components to automatically run the full SR pipeline
        sr_components = ['sr_parameter_optimization', 'sr_detection', 'sr_clustering']
        if sub_pipeline in sr_components:
            config.single_stage_only = False
            tprint(f"🔗 [SUB_PIPELINE_CONFIG] SR chaining enabled for {sub_pipeline} - will automatically run: sr_parameter_optimization -> sr_detection -> sr_clustering")
        # Enable chaining for NAS-TAS components to automatically run the full NAS-TAS pipeline
        nas_tas_components = ['nas_tas_regime_discovery', 'nas_tas_clustering', 'nas_tas_models_training', 'nas_tas_ensemble_training']
        if sub_pipeline in nas_tas_components:
            config.single_stage_only = False
            if sub_pipeline == 'nas_tas_regime_discovery':
                tprint(f"🔗 [SUB_PIPELINE_CONFIG] NAS-TAS chaining enabled for {sub_pipeline} - will automatically run: nas_tas_regime_discovery -> nas_tas_clustering -> nas_tas_models_training -> nas_tas_ensemble_training")
            elif sub_pipeline == 'nas_tas_clustering':
                tprint(f"🔗 [SUB_PIPELINE_CONFIG] NAS-TAS chaining enabled for {sub_pipeline} - will automatically run: nas_tas_clustering -> nas_tas_models_training -> nas_tas_ensemble_training")
                tprint(f"🎯 [SUB_PIPELINE_CONFIG] Using advanced 3-step iterative clustering with risk mitigation", "INFO")
            elif sub_pipeline == 'nas_tas_models_training':
                tprint(f"🔗 [SUB_PIPELINE_CONFIG] NAS-TAS chaining enabled for {sub_pipeline} - will automatically run: nas_tas_models_training -> nas_tas_ensemble_training")
        else:
            config.single_stage_only = True
            tprint("🎯 [SUB_PIPELINE_CONFIG] Single stage execution mode enabled")
        
        # Add intensity parameters to stage configuration
        tprint("🔧 [SUB_PIPELINE_CONFIG] Adding intensity parameters...")
        if config.training_mode_config:
            config.stage_params[target_stage] = {
                'intensity_percentage': config.intensity_percentage,
                'training_mode_config': config.training_mode_config,
                'model_training': config.training_mode_config.get('model_training', {}),
                'validation': config.training_mode_config.get('validation', {}),
                'optimization': config.training_mode_config.get('optimization', {})
            }
            tprint("✅ [SUB_PIPELINE_CONFIG] Intensity parameters added")
        else:
            tprint("⚠️ [SUB_PIPELINE_CONFIG] No training mode config available")
        
        tprint("✅ [SUB_PIPELINE_CONFIG] Sub-pipeline configuration completed successfully")
        return config
    
    async def _execute_full_pipeline(self, config: MainPipelineConfig) -> MainPipelineResult:
        """Execute the full pipeline."""
        # Log full pipeline start
        self._log_stage_transition(None, "FULL_PIPELINE", "FULL_PIPELINE_EXECUTION")
        
        # Create mid-function artifacts
        artifacts = await self._create_mid_function_artifacts(config)
        
        # Execute pipeline with stage-by-stage transition logging
        result = await self._execute_pipeline_with_transitions(config)
        
        # Store execution
        self.current_execution = result
        self.execution_history.append(result)
        
        # Log results
        self._log_execution_results(result)
        
        return result
    
    async def _execute_pipeline_with_transitions(self, config: MainPipelineConfig) -> MainPipelineResult:
        """Execute pipeline with explicit stage transitions."""
        pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        result = MainPipelineResult(
            pipeline_id=pipeline_id,
            status=SubPipelineStatus.RUNNING,
            start_time=start_time
        )
        
        try:
            previous_stage = None
            
            # Execute each enabled stage with transitions
            for stage in config.enabled_stages:
                # Log stage transition
                if previous_stage:
                    self._log_stage_transition(previous_stage.value, stage.value, "STAGE_TRANSITION")
                else:
                    self._log_stage_transition(None, stage.value, "STAGE_START")
                
                # Check for existing outcome files
                outcome_data = await self._check_outcome_files(stage.value, "stage")
                if outcome_data:
                    timestamp = outcome_data.get('metadata', {}).get('timestamp', 'unknown')
                    self.logger.info(f"📂 Resuming from previous outcome: {timestamp}")
                
                # Execute stage
                stage_result = await self.pipeline._execute_stage(stage, config)
                result.stage_results[stage] = stage_result

                # Create outcome files for each successful sub-pipeline in the stage
                # This ensures clustering results are saved even if later sub-pipelines fail
                for sub_result in stage_result:
                    if hasattr(sub_result, 'sub_pipeline_name') and sub_result.status == SubPipelineStatus.COMPLETED:
                        await self._create_outcome_file(stage.value, sub_result.sub_pipeline_name, sub_result, config)
                
                # Check if stage failed
                failed_sub_pipelines = [r for r in stage_result if r.status == SubPipelineStatus.FAILED]
                if failed_sub_pipelines and config.mode != ExecutionMode.BLANK:
                    self.logger.warning(f"⚠️ Stage {stage.value} had {len(failed_sub_pipelines)} failed sub-pipelines")
                    result.failed_stages.append(stage)
                
                previous_stage = stage
            
            # Calculate overall metrics
            self.pipeline._calculate_pipeline_metrics(result)
            
            # Update result status
            end_time = datetime.now()
            result.end_time = end_time
            result.duration_seconds = (end_time - start_time).total_seconds()
            
            if result.failed_sub_pipelines == 0:
                result.status = SubPipelineStatus.COMPLETED
                self.logger.info(f"✅ Full pipeline {pipeline_id} completed successfully in {result.duration_seconds:.2f}s")
            else:
                result.status = SubPipelineStatus.FAILED
                result.error_message = f"Pipeline failed with {result.failed_sub_pipelines} failed sub-pipelines"
                self.logger.error(f"❌ Full pipeline {pipeline_id} failed: {result.error_message}")
            
        except Exception as e:
            end_time = datetime.now()
            result.status = SubPipelineStatus.FAILED
            result.end_time = end_time
            result.duration_seconds = (end_time - start_time).total_seconds()
            result.error_message = str(e)
            
            self.logger.error(f"❌ Full pipeline {pipeline_id} failed with exception: {e}")
        
        return result
    
    async def _execute_stage(self, stage: PipelineStage, config: MainPipelineConfig) -> MainPipelineResult:
        """Execute a specific stage."""
        # Log stage transition
        self._log_stage_transition(None, stage.value, "STAGE_EXECUTION")
        
        # Check for existing outcome files
        outcome_data = await self._check_outcome_files(stage.value, "stage")
        if outcome_data:
            timestamp = outcome_data.get('metadata', {}).get('timestamp', 'unknown')
            self.logger.info(f"📂 Resuming from previous outcome: {timestamp}")
        
        # Create mid-function artifacts for the stage
        artifacts = await self._create_stage_artifacts(stage, config)
        
        # Execute only the specified stage
        result = await self.pipeline.execute_pipeline(config)
        
        # Calculate overall metrics for stage execution
        self.pipeline._calculate_pipeline_metrics(result)
        
        # Create outcome file for this stage
        if result.stage_results and stage in result.stage_results:
            stage_results = result.stage_results[stage]
            for sub_result in stage_results:
                if hasattr(sub_result, 'sub_pipeline_name'):
                    await self._create_outcome_file(stage.value, sub_result.sub_pipeline_name, sub_result, config)
        
        # Store execution
        self.current_execution = result
        self.execution_history.append(result)
        
        # Log results
        self._log_execution_results(result)
        
        return result
    
    async def _execute_sub_pipeline(self, sub_pipeline: str, config: MainPipelineConfig) -> MainPipelineResult:
        """Execute a specific sub-pipeline."""
        # Find the stage containing this sub-pipeline
        target_stage = None
        for stage in PipelineStage:
            available_sub_pipelines = self.pipeline.get_available_sub_pipelines(stage)
            if sub_pipeline in available_sub_pipelines:
                target_stage = stage
                break
        
        if not target_stage:
            raise ValueError(f"Sub-pipeline '{sub_pipeline}' not found in any stage")
        
        # Log sub-pipeline transition
        self._log_sub_pipeline_transition(None, sub_pipeline, target_stage.value)
        
        # Check for existing outcome files
        outcome_data = await self._check_outcome_files(target_stage.value, sub_pipeline)
        if outcome_data:
            timestamp = outcome_data.get('metadata', {}).get('timestamp', outcome_data.get('timestamp', 'unknown'))
            self.logger.info(f"📂 Resuming from previous outcome: {timestamp}")
        
        # Create mid-function artifacts for the sub-pipeline
        artifacts = await self._create_sub_pipeline_artifacts(sub_pipeline, config)
        
        # Execute only the specified sub-pipeline with automatic chaining
        # Use execute_sub_pipeline_with_chain for automatic sequential execution
        sub_pipeline_result = await self.pipeline.execute_sub_pipeline_with_chain(target_stage, sub_pipeline, config)
        
        # Create a MainPipelineResult to maintain compatibility
        result = MainPipelineResult(
            pipeline_id=f"sub_pipeline_{sub_pipeline}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            status=sub_pipeline_result.status if sub_pipeline_result else SubPipelineStatus.FAILED,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=sub_pipeline_result.duration_seconds if sub_pipeline_result else 0.0,
            error_message=sub_pipeline_result.error_message if sub_pipeline_result else "Sub-pipeline execution failed"
        )
        
        # Add the sub-pipeline result to the stage results
        if sub_pipeline_result:
            result.stage_results[target_stage] = [sub_pipeline_result]
        
        # Calculate overall metrics for sub-pipeline execution
        self.pipeline._calculate_pipeline_metrics(result)
        
        # Create outcome file for this sub-pipeline
        if result.stage_results and target_stage in result.stage_results:
            stage_results = result.stage_results[target_stage]
            for sub_result in stage_results:
                if hasattr(sub_result, 'sub_pipeline_name') and sub_result.sub_pipeline_name == sub_pipeline:
                    await self._create_outcome_file(target_stage.value, sub_pipeline, sub_result, config)
                    break
        
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
        
        # Save artifacts - DISABLED: Only outcome file should be created
        # await self._save_artifacts(artifacts, 'full_pipeline_artifacts.json')
        
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
        
        # Save artifacts - DISABLED: Only outcome file should be created
        # await self._save_artifacts(artifacts, f'{stage.value}_artifacts.json')
        
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
        
        # Save artifacts - DISABLED: Only outcome file should be created
        # await self._save_artifacts(artifacts, f'{sub_pipeline}_artifacts.json')
        
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
            PipelineStage.PRE_TRAINING: "Pre-training feature engineering stage",
            PipelineStage.MODEL_TRAINING: "Model training and validation stage",
            PipelineStage.BACKTESTING: "Backtesting and optimization stage"
        }
        return descriptions.get(stage, "Unknown stage")
    
    def _get_sub_pipeline_description(self, sub_pipeline: str) -> str:
        """Get description for a sub-pipeline."""
        descriptions = {
            # Data Collection (10 sub-pipelines)
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
            
            # Market Analysis (10 sub-pipelines)
            'sr_detection': "Detect Support/Resistance levels",
            'sr_clustering': "Generate SR clusters",
            'nas_clustering': "NAS-based regime clustering (DEPRECATED - use nas_tas_clustering instead)",
            'nas_tas_clustering': "Advanced regime clustering using combined NAS-TAS approaches with economic awareness and ensemble methods",
            'nas_tas_regime_discovery': "Discover market regimes using hybrid NAS-TAS approach (combines Neural Architecture Search & Tree-based Architecture Search)",
            'nas_regime_discovery': "Discover market regimes using NAS (DEPRECATED - use nas_tas_regime_discovery instead)",
            'regime_models_training': "Train regime detection models using CatBoost, Bayesian Rule Lists, and ExtraTrees",
            'regime_ensemble_training': "Train ensemble regime detection models",
            'nas_tas_models_training': "Train regime detection models using NAS-TAS regime labels",
            'nas_tas_ensemble_training': "Train ensemble regime detection models using NAS-TAS regime labels",
            'nas': "Combined NAS regime discovery + clustering (DEPRECATED - use nas_tas_regime_discovery instead)",
            'multi_horizon_profit_labeler': "Multi-horizon profit probability labeling (replacement for triple barrier)",
            'analyst_profit_labeler': "Analyst-specific multi-horizon profit labeling (60m timeframe, strategic decision-making)",
            'tactician_entry_labeler': "Tactician-specific entry timing labels (15m timeframe, local maxima/minima detection)",
            'triple_barrier_labeling': "Apply triple barrier method",
            'feature_lookback_optimization': "Optimize feature lookback periods",
            'analyst_feature_lookback_optimization': "Optimize feature lookback periods for Analyst (60m timeframe, strategic)",
            'tactician_feature_lookback_optimization': "Optimize feature lookback periods for Tactician (15m timeframe, tactical)",
            'interactive_feature_generation': "Interactive feature generation with optimized lookbacks, cross-timeframe coverage, and matrix acceleration",
            'analyst_interactive_feature_generation': "Generate interactive features for Analyst models (60m timeframe)",
            'tactician_interactive_feature_generation': "Generate interactive features for Tactician models (15m timeframe)",
            'final_feature_selection': "Perform staged final feature selection (120→100→80→60)",
            'analyst_final_feature_selection': "Final feature selection for Analyst models",
            'tactician_final_feature_selection': "Final feature selection for Tactician models",
            'sr_feature_integration': "Integrate SR-specific features into feature set",
            
            # Model Training (6 sub-pipelines - Analyst & Tactician orchestration)
            'analyst_pre_ml_orchestration': "Analyst Pre-ML: Apply horizon labeling, optimize features, generate PID features, select features (15m timeframe, per-regime/cluster)",
            'analyst_models_training': "Train Analyst base models per-regime (ElasticNet, RandomForest, NAS, TAS, N-BEATS) on 15m timeframe - 8 regimes × 5 models = 40 base models",
            'analyst_ensemble_training': "Train Analyst per-regime ensemble models (8 ensembles combining 5 base models each) on 15m timeframe to produce final green-signal approvals consumed by Tactician",
            'tactician_pre_ml_orchestration': "Tactician Pre-ML: Apply horizon labeling, optimize features, generate PID features, select features (15m timeframe, filtered on Analyst signals >0.4%)",
            'tactician_models_training': "Train Tactician unified base models (RandomSurvivalForest, XGBoost, NAS, TAS) on 5m timeframe using Analyst green-signal filtered data with regime + Analyst features - 4 models total",
            'tactician_ensemble_training': "Train Tactician unified ensemble model (1 ensemble combining 4 base models) on 5m timeframe with Analyst green-signal filtered data and base model outputs",
            # Legacy/deprecated entries
            'hmm_training': "HMM-based model training",
            'ensemble_training': "Ensemble model training",
            'multi_timeframe_training': "Multi-timeframe model training",
            'regime_specific_training': "Regime-specific model training",
            'model_validation': "Model validation and testing",
            'model_persistence': "Save and load models",
            'model_evaluation': "Comprehensive model evaluation",
            
            # Backtesting (10 sub-pipelines)
            'walk_forward_validation': "Walk-forward backtesting",
            'monte_carlo_simulation': "Monte Carlo backtesting",
            'ab_testing': "A/B testing for strategies",
            'basic_backtesting_pre': "Basic historical backtesting (pre-optimization baseline)",
            'final_parameters_optimization': "System-wide parameter optimization",
            'basic_backtesting_post': "Basic historical backtesting (post-optimization comparison)",
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
            # Data Collection dependencies
            'data_conversion': ['data_download'],
            'data_validation': ['data_download', 'data_conversion'],
            'data_preparation': ['data_validation'],
            'feature_engineering': ['data_preparation'],
            'data_quality_check': ['feature_engineering'],
            'data_storage': ['data_quality_check'],
            'data_monitoring': ['data_storage'],
            'data_integration': ['data_monitoring'],
            'data_export': ['data_integration'],
            
            # Market Analysis dependencies
            'sr_clustering': ['sr_detection'],
            'nas_clustering': ['sr_clustering'],  # DEPRECATED - use nas_tas_clustering instead
            'nas_tas_clustering': ['sr_clustering'],
            'nas_tas_regime_discovery': ['sr_clustering'],
            'nas_regime_discovery': ['nas_tas_clustering'],  # DEPRECATED - use nas_tas_regime_discovery instead
            'hybrid_nas_tas_regime_discovery': ['sr_clustering'],
            'nas_tas_clustering': ['hybrid_nas_tas_regime_discovery'],
            'regime_models_training': ['nas_tas_clustering'],
            'regime_ensemble_training': ['regime_models_training'],
            'regime_data_splitting': ['regime_ensemble_training'],
            'triple_barrier_labeling': ['hmm_regime_discovery'],
            'sr_feature_integration': ['regime_data_splitting'],

            # Pre-Training dependencies
            'multi_horizon_profit_labeler': ['regime_data_splitting'],
            'analyst_profit_labeler': ['regime_data_splitting'],
            'tactician_entry_labeler': ['regime_data_splitting'],
            'feature_lookback_optimization': ['multi_horizon_profit_labeler'],
            'interactive_feature_generation': ['feature_lookback_optimization'],
            'final_feature_selection': ['interactive_feature_generation'],
            
            # Model Training dependencies (Analyst → Tactician pipeline)
            'analyst_pre_ml_orchestration': ['final_feature_selection'],  # From PRE_TRAINING stage
            'analyst_models_training': ['analyst_pre_ml_orchestration'],
            'analyst_ensemble_training': ['analyst_models_training'],
            'tactician_pre_ml_orchestration': ['analyst_ensemble_training'],  # Needs Analyst predictions for filtering
            'tactician_models_training': ['tactician_pre_ml_orchestration'],
            'tactician_ensemble_training': ['tactician_models_training'],
            # Legacy dependencies
            'hmm_training': ['sr_feature_integration'],
            'regime_specific_training': ['tactician_ensemble_training'],
            'model_validation': ['regime_specific_training'],
            'model_persistence': ['model_validation'],
            'model_evaluation': ['model_persistence'],
            
            # Backtesting dependencies
            'basic_backtesting_pre': [],
            'final_parameters_optimization': ['basic_backtesting_pre'],
            'basic_backtesting_post': ['final_parameters_optimization'],
            'walk_forward_validation': ['basic_backtesting_post'],
            'monte_carlo_simulation': ['walk_forward_validation'],
            'ab_testing': ['monte_carlo_simulation'],
            'performance_analytics': ['ab_testing'],
            'risk_analysis': ['performance_analytics'],
            'trade_analysis': ['risk_analysis'],
            'portfolio_analysis': ['trade_analysis'],
            'reporting': ['portfolio_analysis']
        }
        return dependencies.get(sub_pipeline, [])
    
    def _get_sub_pipeline_outputs(self, sub_pipeline: str) -> List[str]:
        """Get expected outputs for a sub-pipeline."""
        outputs = {
            # Data Collection outputs
            'data_download': ['raw_data.parquet'],
            'data_conversion': ['converted_data.parquet'],
            'data_validation': ['validation_report.json'],
            'data_preparation': ['prepared_data.parquet'],
            'feature_engineering': ['features.parquet'],
            'data_quality_check': ['quality_report.json'],
            'data_storage': ['stored_data.parquet'],
            'data_monitoring': ['monitoring_report.json'],
            'data_integration': ['integrated_data.parquet'],
            'data_export': ['exported_data.parquet'],
            
            # Market Analysis outputs
            'sr_detection': ['sr_levels.json'],
            'sr_clustering': ['sr_clusters.json'],
            'nas_clustering': ['nas_clusters.json'],
            'nas_tas_clustering': ['nas_tas_clustering_report.json', 'nas_tas_regime_assignments.parquet'],
            'nas_tas_regime_discovery': ['nas_tas_consolidated_report.json', 'nas_tas_regime_assignments.parquet'],
            'nas_regime_discovery': ['nas_regime_assignments.parquet'],
            'hybrid_nas_tas_regime_discovery': ['hybrid_nas_tas_regime_discovery_result.json'],
            'nas_tas_clustering': ['nas_tas_clustering_result.json'],
            'regime_models_training': ['regime_models_training_result.json'],
            'regime_ensemble_training': ['regime_ensemble_training_result.json'],
            'regime_data_splitting': ['regime_data_splitting_result.parquet'],
            'triple_barrier_labeling': ['labels.parquet'],
            'sr_feature_integration': ['sr_features.json'],

            # Pre-Training outputs
            'multi_horizon_profit_labeler': ['multi_horizon_labels.parquet'],
            'analyst_profit_labeler': ['analyst_multi_horizon_labels.parquet', 'analyst_labeling_report.json'],
            'tactician_entry_labeler': ['tactician_entry_labels.parquet', 'tactician_labeling_report.json'],
            'feature_lookback_optimization': ['optimized_features.parquet'],
            'interactive_feature_generation': [
                'features_<symbol>_<timeframe>.parquet',
                'interactions_<symbol>_<timeframe>.parquet',
                'cross_timeframe_<symbol>_<timeframe>.parquet'
            ],
            'final_feature_selection': ['final_features.parquet'],

            # Model Training outputs
            'analyst_pre_ml_orchestration': ['analyst_features_15m.parquet', 'analyst_selected_features.json', 'regime_features_added.json'],
            'analyst_models_training': ['analyst_base_models_per_regime.pkl', 'analyst_nas_models.pkl', 'analyst_tas_models.pkl', 'analyst_nbeats_models.pkl'],
            'analyst_ensemble_training': ['analyst_ensemble_per_regime.pkl', 'analyst_predictions.parquet'],
            'tactician_pre_ml_orchestration': ['tactician_features_5m.parquet', 'tactician_selected_features.json', 'filtered_data_report.json', 'regime_features_added.json'],
            'tactician_models_training': ['tactician_base_models_unified.pkl', 'tactician_nas_model.pkl', 'tactician_tas_model.pkl'],
            'tactician_ensemble_training': ['tactician_ensemble_unified.pkl', 'tactician_predictions.parquet'],
            # Legacy outputs
            'hmm_training': ['hmm_model.pkl'],
            'regime_specific_training': ['regime_models.pkl'],
            'model_validation': ['validation_results.json'],
            'model_persistence': ['persisted_models.pkl'],
            'model_evaluation': ['evaluation_results.json'],
            
            # Backtesting outputs
            'walk_forward_validation': ['backtest_results.json'],
            'monte_carlo_simulation': ['mc_results.json'],
            'ab_testing': ['ab_test_results.json'],
            'basic_backtesting_pre': ['basic_backtest_pre_results.json'],
            'final_parameters_optimization': ['optimized_parameters.json'],
            'basic_backtesting_post': ['basic_backtest_post_results.json'],
            'performance_analytics': ['performance_report.json'],
            'risk_analysis': ['risk_report.json'],
            'trade_analysis': ['trade_analysis.json'],
            'portfolio_analysis': ['portfolio_analysis.json'],
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
Note: The following sub-pipelines are DEPRECATED and will be removed in future versions:
  - nas_regime_discovery (use nas_tas_regime_discovery instead)
  - nas_clustering (use nas_tas_clustering instead)
  - nas (use nas_tas_regime_discovery instead)

Examples:
  # Full pipeline execution (1460 days, 100% intensity, both directions)
  python ares_launcher.py --mode full --symbol ETHUSDT --exchange binance

  # Full pipeline execution for longs only
  python ares_launcher.py --mode full --symbol ETHUSDT --direction longs

  # Full pipeline execution for shorts only
  python ares_launcher.py --mode full --symbol ETHUSDT --direction shorts

  # Light pipeline execution (10 days, 5% intensity, both directions)
  python ares_launcher.py --mode light --symbol ETHUSDT

  # Execute specific stage with full execution mode (1460 days, 100% intensity)
  python ares_launcher.py --mode stage --stage data_collection --execution-mode full --symbol ETHUSDT

  # Execute specific stage for longs only
  python ares_launcher.py --mode stage --stage data_collection --direction longs --symbol ETHUSDT

  # Execute specific stage with light execution mode (10 days, 5% intensity)
  python ares_launcher.py --mode stage --stage market_analysis --execution-mode light --symbol ETHUSDT

  # Execute specific sub-pipeline with blank execution mode (180 days, 10% intensity)
  python ares_launcher.py --mode sub_pipeline --sub_pipeline sr_detection --execution-mode blank --symbol ETHUSDT

  # Execute specific sub-pipeline for longs only
  python ares_launcher.py --mode sub_pipeline --sub_pipeline sr_detection --direction longs --symbol ETHUSDT

  # Execute specific sub-pipeline with full execution mode (1460 days, 100% intensity)
  python ares_launcher.py --mode sub_pipeline --sub_pipeline nas_tas_regime_discovery --execution-mode full --symbol ETHUSDT

  # Execute NAS-TAS clustering with full execution mode
  python ares_launcher.py --mode sub_pipeline --sub_pipeline nas_tas_clustering --execution-mode full --symbol ETHUSDT

  # Execute specific sub-pipeline with full execution mode (1460 days, 100% intensity) - RECOMMENDED
  python ares_launcher.py --mode sub_pipeline --sub_pipeline nas_tas_regime_discovery --execution-mode full --symbol ETHUSDT

  # Execute specific sub-pipeline with full execution mode (DEPRECATED - use nas_tas_regime_discovery instead)
  python ares_launcher.py --mode sub_pipeline --sub_pipeline nas_regime_discovery --execution-mode full --symbol ETHUSDT

  # Execute basic backtesting (pre-optimization baseline)
  python ares_launcher.py --mode sub_pipeline --sub_pipeline basic_backtesting_pre --execution-mode full --symbol ETHUSDT

  # Execute basic backtesting (post-optimization comparison)
  python ares_launcher.py --mode sub_pipeline --sub_pipeline basic_backtesting_post --execution-mode full --symbol ETHUSDT

  # Execute walk-forward validation (after post-optimization basic backtesting)
  python ares_launcher.py --mode sub_pipeline --sub_pipeline walk_forward_validation --execution-mode full --symbol ETHUSDT

  # Shortcut: Execute Analyst pre-ML orchestration
  python ares_launcher.py --analyst-pre-ml --symbol ETHUSDT

  # Shortcut: Execute Analyst profit labeler
  python ares_launcher.py --analyst-labeler --symbol ETHUSDT --timeframe 60m

  # Shortcut: Execute Tactician entry labeler
  python ares_launcher.py --tactician-labeler --symbol ETHUSDT --timeframe 15m

  # Shortcut: Execute Tactician ensemble training in light mode
  python ares_launcher.py --tactician-ensemble --execution-mode light --symbol ETHUSDT

  # Blank mode for testing (180 days, 10% intensity)
  python ares_launcher.py --mode blank --symbol ETHUSDT
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['full', 'light', 'blank', 'stage', 'sub_pipeline'],
        default='full',
        help='Launcher execution mode (default: full)'
    )
    
    parser.add_argument(
        '--execution-mode',
        choices=['full', 'light', 'blank'],
        default='full',
        help='Execution mode type for stage/sub-pipeline specific execution (default: full)'
    )
    
    parser.add_argument(
        '--symbol',
        default='ETHUSDT',
        help='Trading symbol (default: ETHUSDT)'
    )
    
    parser.add_argument(
        '--exchange',
        default='binance',
        help='Exchange name (default: binance)'
    )
    
    parser.add_argument(
        '--timeframe',
        default='1m',
        help='Data timeframe (default: 1m; use 15m for Analyst steps and 5m for Tactician steps)'
    )
    
    parser.add_argument(
        '--data-dir',
        default='historical_data',
        help='Data directory (default: historical_data)'
    )

    parser.add_argument(
        '--direction',
        choices=['longs', 'shorts', 'both'],
        default='longs',
        help='Direction type for training: longs (long positions only), shorts (short positions only), or both (default: longs)'
    )

    parser.add_argument(
        '--stage',
        choices=['data_collection', 'market_analysis', 'pre_training', 'model_training', 'backtesting'],
        help='Specific stage to execute (for stage mode)'
    )
    
    parser.set_defaults(shortcut_sub_pipeline=None)

    shortcut_group = parser.add_mutually_exclusive_group()
    shortcut_group.add_argument(
        '--analyst-pre-ml',
        dest='shortcut_sub_pipeline',
        action='store_const',
        const='analyst_pre_ml_orchestration',
        help='Shortcut for Analyst pre-ML orchestration sub-pipeline.'
    )
    shortcut_group.add_argument(
        '--analyst-models',
        dest='shortcut_sub_pipeline',
        action='store_const',
        const='analyst_models_training',
        help='Shortcut for Analyst models training sub-pipeline.'
    )
    shortcut_group.add_argument(
        '--analyst-ensemble',
        dest='shortcut_sub_pipeline',
        action='store_const',
        const='analyst_ensemble_training',
        help='Shortcut for Analyst ensemble training sub-pipeline.'
    )
    shortcut_group.add_argument(
        '--tactician-pre-ml',
        dest='shortcut_sub_pipeline',
        action='store_const',
        const='tactician_pre_ml_orchestration',
        help='Shortcut for Tactician pre-ML orchestration sub-pipeline.'
    )
    shortcut_group.add_argument(
        '--tactician-models',
        dest='shortcut_sub_pipeline',
        action='store_const',
        const='tactician_models_training',
        help='Shortcut for Tactician models training sub-pipeline.'
    )
    shortcut_group.add_argument(
        '--tactician-ensemble',
        dest='shortcut_sub_pipeline',
        action='store_const',
        const='tactician_ensemble_training',
        help='Shortcut for Tactician ensemble training sub-pipeline.'
    )
    shortcut_group.add_argument(
        '--analyst-labeler',
        dest='shortcut_sub_pipeline',
        action='store_const',
        const='analyst_profit_labeler',
        help='Shortcut for Analyst profit labeler sub-pipeline.'
    )
    shortcut_group.add_argument(
        '--tactician-labeler',
        dest='shortcut_sub_pipeline',
        action='store_const',
        const='tactician_entry_labeler',
        help='Shortcut for Tactician entry labeler sub-pipeline.'
    )

    parser.add_argument(
        '--sub-pipeline', '--sub_pipeline',
        help='Specific sub-pipeline to execute (for sub_pipeline mode). Available: analyst_pre_ml_orchestration, analyst_models_training, analyst_ensemble_training, tactician_pre_ml_orchestration, tactician_models_training, tactician_ensemble_training, nas_tas_regime_discovery, nas_tas_clustering, multi_horizon_profit_labeler, analyst_profit_labeler, tactician_entry_labeler, feature_lookback_optimization, interactive_feature_generation, final_feature_selection, basic_backtesting_pre, basic_backtesting_post, walk_forward_validation, etc. You can also use shortcut flags like --analyst-pre-ml, --analyst-labeler, --tactician-labeler, or --tactician-ensemble.'
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
        help='List available sub-pipelines for a stage. Use with --stage to see sub-pipelines for that stage.'
    )
    
    return parser

async def main():
    """Main entry point."""
    tprint("🎯 [MAIN] Starting Ares Launcher CLI...")
    tprint("🎯 [MAIN] Creating CLI argument parser...")
    parser = create_cli_parser()
    tprint("✅ [MAIN] CLI parser created")
    
    tprint("🎯 [MAIN] Parsing command line arguments...")
    args = parser.parse_args()
    tprint("✅ [MAIN] Arguments parsed successfully")
    tprint(f"🎯 [MAIN] Mode: {args.mode}")
    tprint(f"🎯 [MAIN] Symbol: {args.symbol}")
    tprint(f"🎯 [MAIN] Exchange: {args.exchange}")
    tprint(f"🎯 [MAIN] Timeframe: {args.timeframe}")

    shortcut_sub_pipeline = getattr(args, 'shortcut_sub_pipeline', None)
    selected_sub_pipeline = args.sub_pipeline
    if shortcut_sub_pipeline:
        tprint(f"🎯 [MAIN] Shortcut flag selected: {shortcut_sub_pipeline}")
        if selected_sub_pipeline and selected_sub_pipeline != shortcut_sub_pipeline:
            parser.error(
                "Shortcut sub-pipeline flags cannot be combined with a conflicting --sub-pipeline value"
            )
        selected_sub_pipeline = shortcut_sub_pipeline

    # Initialize launcher
    tprint("🎯 [MAIN] Initializing AresLauncher...")
    launcher = AresLauncher()
    tprint("✅ [MAIN] AresLauncher initialized successfully")
    
    # Handle list commands
    if args.list_stages:
        tprint("📋 [MAIN] Listing available pipeline stages...")
        stages = launcher.get_available_stages()
        tprint("Available Pipeline Stages:")
        for stage in stages:
            tprint(f"  - {stage}")
        tprint("✅ [MAIN] Stage listing completed")
        return
    
    if args.list_sub_pipelines:
        tprint(f"📋 [MAIN] Listing available sub-pipelines for: {args.list_sub_pipelines}")
        sub_pipelines = launcher.get_available_sub_pipelines(args.list_sub_pipelines)
        tprint(f"Available Sub-pipelines for {args.list_sub_pipelines}:")
        for stage, pipelines in sub_pipelines.items():
            tprint(f"  {stage}:")
            for pipeline in pipelines:
                tprint(f"    - {pipeline}")
        tprint("✅ [MAIN] Sub-pipeline listing completed")
        return
    
    # Load custom configuration if provided
    custom_config = None
    if args.config:
        tprint(f"📁 [MAIN] Loading custom configuration from: {args.config}")
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
        tprint(f"✅ [MAIN] Custom configuration loaded: {len(custom_config)} parameters")
    else:
        tprint("📁 [MAIN] No custom configuration provided, using defaults")
    
    # Convert string mode to enum
    tprint("🔄 [MAIN] Converting string modes to enums...")
    mode_map = {
        'full': LauncherMode.FULL,
        'light': LauncherMode.LIGHT,
        'blank': LauncherMode.BLANK,
        'stage': LauncherMode.STAGE,
        'sub_pipeline': LauncherMode.SUB_PIPELINE
    }
    mode = mode_map[args.mode]

    if shortcut_sub_pipeline:
        mode = LauncherMode.SUB_PIPELINE
        tprint("🎯 [MAIN] Launcher mode overridden to sub_pipeline due to shortcut flag")

    tprint(f"✅ [MAIN] Launcher mode converted: {mode.value}")
    
    # Convert execution mode to enum
    execution_mode_map = {
        'full': ExecutionModeType.FULL,
        'light': ExecutionModeType.LIGHT,
        'blank': ExecutionModeType.BLANK
    }
    execution_mode = execution_mode_map[args.execution_mode]
    tprint(f"✅ [MAIN] Execution mode converted: {execution_mode.value}")

    # Convert string stage to enum if provided
    stage = None
    if args.stage:
        tprint(f"🔄 [MAIN] Converting stage string to enum: {args.stage}")
        stage_map = {
            'data_collection': PipelineStage.DATA_COLLECTION,
            'market_analysis': PipelineStage.MARKET_ANALYSIS,
            'pre_training': PipelineStage.PRE_TRAINING,
            'model_training': PipelineStage.MODEL_TRAINING,
            'backtesting': PipelineStage.BACKTESTING
        }
        stage = stage_map[args.stage]
        tprint(f"✅ [MAIN] Stage converted: {stage.value}")
    else:
        tprint("📋 [MAIN] No specific stage provided")

    if selected_sub_pipeline:
        tprint(f"📋 [MAIN] Sub-pipeline selected: {selected_sub_pipeline}")
    else:
        tprint("📋 [MAIN] No specific sub-pipeline provided")
    
    # Execute pipeline
    tprint("🚀 [MAIN] Starting pipeline execution...")
    try:
        result = await launcher.execute_pipeline(
            mode=mode,
            symbol=args.symbol,
            exchange=args.exchange,
            timeframe=args.timeframe,
            data_dir=args.data_dir,
            direction=args.direction if hasattr(args, 'direction') else 'longs',
            stage=stage,
            sub_pipeline=selected_sub_pipeline,
            execution_mode=execution_mode,
            custom_config=custom_config
        )
        tprint("✅ [MAIN] Pipeline execution completed successfully")
        
        # Print final results
        tprint("\n" + "=" * 80)
        tprint("🎯 EXECUTION COMPLETED")
        tprint("=" * 80)
        tprint(f"Status: {result.status.value}")
        tprint(f"Duration: {result.duration_seconds:.2f}s")
        tprint(f"Success Rate: {result.success_rate:.2%}")
        tprint("=" * 80)
        
        if result.status.value == 'failed':
            tprint("❌ [MAIN] Pipeline execution failed, exiting with code 1")
            sys.exit(1)
        elif result.success_rate == 0.0:
            tprint("⚠️ [MAIN] Pipeline completed but no sub-pipelines succeeded, exiting with code 0")
        else:
            tprint("✅ [MAIN] Pipeline execution successful, exiting with code 0")
            
    except Exception as e:
        tprint(f"❌ [MAIN] Pipeline execution failed with exception: {e}")
        logger.error(f"❌ Execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())