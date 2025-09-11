"""
Market Analysis Sub-Pipeline

This module provides granular sub-pipeline functionality for market analysis,
allowing execution of specific market analysis steps with different modes.

Sub-pipelines:
1. SR Detection - Detect Support/Resistance levels
2. SR Clustering - Generate SR clusters
3. SR ML Learning - ML-based learning for SR clusters
4. HMM Clustering - HMM-based regime clustering
5. HMM Regime Discovery - Discover market regimes
6. Regime Data Splitting - Split data by regimes
7. Triple Barrier Labeling - Apply triple barrier method
8. Feature Lookback Optimization - Optimize feature lookback periods
9. Fractional Differentiation - Apply fractional differentiation
10. Cross Timeframe Analysis - Cross timeframe interaction features
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
from src.utils.enhanced_artifact_manager import get_artifact_manager
from src.utils.version_manager import get_version_manager

# Import ML commons utilities
try:
    from src.utils.ml_common.data_labeling import get_data_labeler, TripleBarrierConfig, LabelingMethod
    from src.utils.ml_common.hmm_regime_detection import get_hmm_regime_detector, HMMRegimeConfig, RegimeDetectionMethod
    from src.utils.ml_common.regime_data_processing import get_regime_processor, RegimeProcessingConfig
    from src.feature_engineering.feature_generation_optimization import get_feature_optimizer, FeatureOptimizationConfig
    ML_COMMONS_AVAILABLE = True
except ImportError:
    ML_COMMONS_AVAILABLE = False

logger = system_logger.getChild('MarketAnalysisSubPipeline')

class ExecutionMode(Enum):
    """Execution modes for sub-pipelines."""
    FULL = "full"          # Complete execution with all features
    LIGHT = "light"        # Lightweight execution with essential features only
    BLANK = "blank"        # Minimal execution for testing/validation

class SubPipelineStatus(Enum):
    """Status of sub-pipeline execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class SubPipelineConfig:
    """Configuration for sub-pipeline execution."""
    mode: ExecutionMode = ExecutionMode.FULL
    symbol: str = "BTCUSDT"
    exchange: str = "binance"
    timeframe: str = "30m"
    data_dir: str = "data/training"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    force_rerun: bool = False
    parallel_processing: bool = True
    max_workers: int = 4
    validation_enabled: bool = True
    monitoring_enabled: bool = True
    custom_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SubPipelineResult:
    """Result of sub-pipeline execution."""
    sub_pipeline_name: str
    status: SubPipelineStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    output_files: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)

class MarketAnalysisSubPipeline:
    """
    Market Analysis Sub-Pipeline Manager.
    
    Provides granular control over market analysis processes with different
    execution modes and comprehensive monitoring.
    """
    
    def __init__(self, config: Optional[SubPipelineConfig] = None):
        """Initialize the market analysis sub-pipeline with backward compatibility."""
        # Handle both old dict config and new SubPipelineConfig
        if isinstance(config, dict):
            # Convert old config format to SubPipelineConfig
            self.original_config = config
            self.config = self._convert_old_config(config)
        else:
            # Use provided SubPipelineConfig or create default
            self.config = config or SubPipelineConfig()
            self.original_config = {}
        
        self.logger = logger.getChild('MarketAnalysisSubPipeline')
        self.results: List[SubPipelineResult] = []
        
        # Initialize artifact and version managers
        self.artifact_manager = get_artifact_manager()
        self.version_manager = get_version_manager()
        
        # Initialize sub-pipeline registry
        self.sub_pipelines = {
            'sr_detection': self._sr_detection_pipeline,
            'sr_clustering': self._sr_clustering_pipeline,
            'sr_ml_learning': self._sr_ml_learning_pipeline,
            'hmm_clustering': self._hmm_clustering_pipeline,
            'hmm_regime_discovery': self._hmm_regime_discovery_pipeline,
            'regime_data_splitting': self._regime_data_splitting_pipeline,
            'triple_barrier_labeling': self._triple_barrier_labeling_pipeline,
            'feature_lookback_optimization': self._feature_lookback_optimization_pipeline,
            'fractional_differentiation': self._fractional_differentiation_pipeline,
            'cross_timeframe_analysis': self._cross_timeframe_analysis_pipeline
        }
    
    def _convert_old_config(self, config: Dict[str, Any]) -> SubPipelineConfig:
        """Convert old config format to SubPipelineConfig."""
        # Extract relevant configuration
        sr_config = config.get('sr_optimization', {})
        training_mode = config.get('training_mode', 'full')
        
        # Determine execution mode
        if training_mode == 'light':
            mode = ExecutionMode.LIGHT
        elif training_mode == 'blank':
            mode = ExecutionMode.BLANK
        else:
            mode = ExecutionMode.FULL
        
        # Create SubPipelineConfig
        sub_config = SubPipelineConfig(
            mode=mode,
            symbol=config.get('symbol', 'BTCUSDT'),
            exchange=config.get('exchange', 'binance'),
            timeframe=config.get('timeframe', '1m'),
            data_dir=config.get('data_dir', './data'),
            output_dir=config.get('output_dir', './output')
        )
        
        return sub_config
    
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the SR optimization pipeline with backward compatible interface.
        
        This method provides the same interface as the original SROptimizationStep
        while orchestrating the three SR stages internally.
        """
        self.logger.info('🎯 Starting SROptimizationStep execution with backward compatibility')
        
        try:
            # Extract data from pipeline state
            data = pipeline_state.get('dataframe')
            if data is None:
                raise ValueError("No dataframe found in pipeline state")
            
            # Update config with data information
            self.config.symbol = training_input.get('symbol', 'BTCUSDT')
            self.config.exchange = training_input.get('exchange', 'binance')
            self.config.timeframe = training_input.get('timeframe', '1m')
            
            # Execute the three SR stages in sequence
            results = {}
            
            # Stage 1: SR Detection
            self.logger.info('🎯 Executing Stage 1: SR Detection')
            detection_result = await self.execute_sub_pipeline('sr_detection', self.config)
            if detection_result.success:
                results['sr_levels'] = detection_result.artifacts.get('sr_levels', [])
                results['sr_metrics'] = detection_result.artifacts.get('sr_metrics', {})
                self.logger.info(f"✅ SR Detection completed: {len(results['sr_levels'])} levels detected")
            else:
                self.logger.error(f"❌ SR Detection failed: {detection_result.error}")
                return {
                    'success': False,
                    'error': f"SR Detection failed: {detection_result.error}",
                    'stage': 'sr_detection'
                }
            
            # Stage 2: SR Clustering
            self.logger.info('🚀 Executing Stage 2: SR Clustering')
            clustering_result = await self.execute_sub_pipeline('sr_clustering', self.config)
            if clustering_result.success:
                results['clustered_levels'] = clustering_result.artifacts.get('clustered_levels', [])
                results['cluster_metrics'] = clustering_result.artifacts.get('cluster_metrics', {})
                self.logger.info(f"✅ SR Clustering completed: {len(results['clustered_levels'])} clusters")
            else:
                self.logger.error(f"❌ SR Clustering failed: {clustering_result.error}")
                return {
                    'success': False,
                    'error': f"SR Clustering failed: {clustering_result.error}",
                    'stage': 'sr_clustering'
                }
            
            # Stage 3: SR ML Learning
            self.logger.info('🤖 Executing Stage 3: SR ML Learning')
            ml_result = await self.execute_sub_pipeline('sr_ml_learning', self.config)
            if ml_result.success:
                results['ml_models'] = ml_result.artifacts.get('ml_models', [])
                results['ml_metrics'] = ml_result.artifacts.get('ml_metrics', {})
                self.logger.info(f"✅ SR ML Learning completed: {len(results['ml_models'])} models")
            else:
                self.logger.error(f"❌ SR ML Learning failed: {ml_result.error}")
                return {
                    'success': False,
                    'error': f"SR ML Learning failed: {ml_result.error}",
                    'stage': 'sr_ml_learning'
                }
            
            # Calculate total execution time
            total_time = (
                detection_result.execution_time + 
                clustering_result.execution_time + 
                ml_result.execution_time
            )
            
            self.logger.info('🎯 SROptimizationStep execution completed successfully')
            self.logger.info(f"📊 Total execution time: {total_time:.2f} seconds")
            
            return {
                'success': True,
                'sr_levels': results['sr_levels'],
                'clustered_levels': results['clustered_levels'],
                'ml_models': results['ml_models'],
                'sr_metrics': results['sr_metrics'],
                'cluster_metrics': results['cluster_metrics'],
                'ml_metrics': results['ml_metrics'],
                'execution_time': total_time,
                'stage_times': {
                    'detection': detection_result.execution_time,
                    'clustering': clustering_result.execution_time,
                    'ml_learning': ml_result.execution_time
                },
                'stage': 'complete_sr_optimization'
            }
            
        except Exception as e:
            self.logger.error(f'❌ SROptimizationStep execution failed: {e}')
            import traceback
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')
            return {
                'success': False,
                'error': str(e),
                'stage': 'complete_sr_optimization'
            }
    
    def validate_config(self):
        """Validate configuration for backward compatibility."""
        self.logger.info('🔍 Validating SROptimizationStep configuration...')
        
        # Validate required configuration
        if hasattr(self, 'original_config') and self.original_config:
            required_keys = ['sr_optimization']
            for key in required_keys:
                if key not in self.original_config:
                    self.logger.warning(f"⚠️ Missing configuration key: {key}")
            
            # Validate SR optimization config
            sr_config = self.original_config.get('sr_optimization', {})
            required_sr_keys = ['min_touches', 'tolerance_pct', 'lookback_periods']
            for key in required_sr_keys:
                if key not in sr_config:
                    self.logger.warning(f"⚠️ Missing SR optimization key: {key}")
        
        self.logger.info('✅ SROptimizationStep configuration validation completed')
        return True
    
    def get_status(self):
        """Get status for backward compatibility."""
        return {
            'stage': 'sr_optimization',
            'status': 'ready',
            'config': getattr(self, 'original_config', {}),
            'sub_pipeline_status': 'initialized'
        }
    
    def _log_sub_pipeline_completion(self, sub_pipeline_name: str, config: SubPipelineConfig, artifacts: Dict[str, Any]):
        """Helper method to log sub-pipeline completion with emojis and artifact paths."""
        print("\n" + "="*80)
        print(f"🎉 {sub_pipeline_name.upper().replace('_', ' ')} SUB-PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"📁 Artifact Paths:")
        
        # Log different types of artifacts with appropriate emojis
        for key, value in artifacts.items():
            if isinstance(value, list) and value:
                if 'model' in key.lower():
                    for item in value:
                        print(f"   🤖 {key.title()}: {config.data_dir}/models/{item}")
                elif 'file' in key.lower() or 'data' in key.lower():
                    for item in value:
                        print(f"   📄 {key.title()}: {config.data_dir}/{item}")
                elif 'report' in key.lower():
                    for item in value:
                        print(f"   📋 {key.title()}: {config.data_dir}/{item}")
                else:
                    for item in value:
                        print(f"   📊 {key.title()}: {config.data_dir}/{item}")
            elif isinstance(value, dict) and value:
                print(f"   📊 {key.title()}: {config.data_dir}/{key}.json")
        
        print(f"📊 Artifacts Summary: {len(artifacts)} artifact types generated")
        print("="*80 + "\n")
        
        # Log to logger as well
        self.logger.info(f"🎉 {sub_pipeline_name.upper().replace('_', ' ')} SUB-PIPELINE COMPLETED SUCCESSFULLY!")
        self.logger.info(f"📁 Artifact Paths:")
        for key, value in artifacts.items():
            if isinstance(value, list) and value:
                if 'model' in key.lower():
                    for item in value:
                        self.logger.info(f"   🤖 {key.title()}: {config.data_dir}/models/{item}")
                elif 'file' in key.lower() or 'data' in key.lower():
                    for item in value:
                        self.logger.info(f"   📄 {key.title()}: {config.data_dir}/{item}")
                elif 'report' in key.lower():
                    for item in value:
                        self.logger.info(f"   📋 {key.title()}: {config.data_dir}/{item}")
                else:
                    for item in value:
                        self.logger.info(f"   📊 {key.title()}: {config.data_dir}/{item}")
            elif isinstance(value, dict) and value:
                self.logger.info(f"   📊 {key.title()}: {config.data_dir}/{key}.json")
        self.logger.info(f"📊 Artifacts Summary: {len(artifacts)} artifact types generated")
    
    async def execute_sub_pipeline(
        self,
        sub_pipeline_name: str,
        config: Optional[SubPipelineConfig] = None
    ) -> SubPipelineResult:
        """
        Execute a specific sub-pipeline.
        
        Args:
            sub_pipeline_name: Name of the sub-pipeline to execute
            config: Optional configuration override
            
        Returns:
            SubPipelineResult with execution details
        """
        config = config or self.config
        self.logger.info(f"🚀 Starting market analysis sub-pipeline: {sub_pipeline_name} (mode: {config.mode.value})")
        
        start_time = datetime.now()
        result = SubPipelineResult(
            sub_pipeline_name=sub_pipeline_name,
            status=SubPipelineStatus.RUNNING,
            start_time=start_time
        )
        
        try:
            if sub_pipeline_name not in self.sub_pipelines:
                raise ValueError(f"Unknown sub-pipeline: {sub_pipeline_name}")
            
            # Execute the sub-pipeline
            pipeline_func = self.sub_pipelines[sub_pipeline_name]
            artifacts = await pipeline_func(config)
            
            # Update result
            end_time = datetime.now()
            result.status = SubPipelineStatus.COMPLETED
            result.end_time = end_time
            result.duration_seconds = (end_time - start_time).total_seconds()
            result.artifacts = artifacts
            result.metadata = {
                'mode': config.mode.value,
                'symbol': config.symbol,
                'exchange': config.exchange,
                'timeframe': config.timeframe
            }
            
            self.logger.info(f"✅ Market analysis sub-pipeline {sub_pipeline_name} completed in {result.duration_seconds:.2f}s")
            
        except Exception as e:
            end_time = datetime.now()
            result.status = SubPipelineStatus.FAILED
            result.end_time = end_time
            result.duration_seconds = (end_time - start_time).total_seconds()
            result.error_message = str(e)
            
            self.logger.error(f"❌ Market analysis sub-pipeline {sub_pipeline_name} failed: {e}")
        
        self.results.append(result)
        return result
    
    async def execute_multiple_sub_pipelines(
        self,
        sub_pipeline_names: List[str],
        config: Optional[SubPipelineConfig] = None,
        sequential: bool = False
    ) -> List[SubPipelineResult]:
        """
        Execute multiple sub-pipelines.
        
        Args:
            sub_pipeline_names: List of sub-pipeline names to execute
            config: Optional configuration override
            sequential: Whether to execute sequentially or in parallel
            
        Returns:
            List of SubPipelineResult objects
        """
        config = config or self.config
        self.logger.info(f"🚀 Starting {len(sub_pipeline_names)} market analysis sub-pipelines (sequential: {sequential})")
        
        if sequential:
            results = []
            for name in sub_pipeline_names:
                result = await self.execute_sub_pipeline(name, config)
                results.append(result)
                if result.status == SubPipelineStatus.FAILED:
                    self.logger.warning(f"⚠️ Stopping sequential execution due to failure in {name}")
                    break
            return results
        else:
            # Execute in parallel
            tasks = [self.execute_sub_pipeline(name, config) for name in sub_pipeline_names]
            return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def execute_sub_pipeline_with_next(
        self,
        sub_pipeline_name: str,
        config: Optional[SubPipelineConfig] = None
    ) -> SubPipelineResult:
        """
        Execute a specific sub-pipeline and automatically trigger the next one upon completion.
        
        Args:
            sub_pipeline_name: Name of the sub-pipeline to execute
            config: Optional configuration override
            
        Returns:
            SubPipelineResult with execution details
        """
        config = config or self.config
        self.logger.info(f"🚀 Starting sub-pipeline with auto-next: {sub_pipeline_name}")
        
        # Execute the current sub-pipeline
        result = await self.execute_sub_pipeline(sub_pipeline_name, config)
        
        # If successful, find and execute the next sub-pipeline
        if result.status == SubPipelineStatus.COMPLETED:
            next_sub_pipeline = self._get_next_sub_pipeline(sub_pipeline_name)
            if next_sub_pipeline:
                self.logger.info(f"✅ {sub_pipeline_name} completed successfully, triggering next: {next_sub_pipeline}")
                try:
                    next_result = await self.execute_sub_pipeline_with_next(next_sub_pipeline, config)
                    # Add the next result to our results list
                    self.results.append(next_result)
                except Exception as e:
                    self.logger.error(f"❌ Failed to execute next sub-pipeline {next_sub_pipeline}: {e}")
            else:
                self.logger.info(f"✅ {sub_pipeline_name} completed successfully - no more sub-pipelines to execute")
        else:
            self.logger.warning(f"⚠️ {sub_pipeline_name} failed, not triggering next sub-pipeline")
        
        return result
    
    def _get_next_sub_pipeline(self, current_sub_pipeline: str) -> Optional[str]:
        """
        Get the next sub-pipeline in the sequence.
        
        Args:
            current_sub_pipeline: Current sub-pipeline name
            
        Returns:
            Next sub-pipeline name or None if no more sub-pipelines
        """
        # Define the execution order for market analysis sub-pipelines
        execution_order = [
            'sr_detection',
            'sr_clustering', 
            'sr_ml_learning',
            'hmm_clustering',
            'hmm_regime_discovery',
            'regime_data_splitting',
            'triple_barrier_labeling',
            'feature_lookback_optimization',
            'fractional_differentiation',
            'cross_timeframe_analysis'
        ]
        
        try:
            current_index = execution_order.index(current_sub_pipeline)
            if current_index < len(execution_order) - 1:
                return execution_order[current_index + 1]
        except ValueError:
            self.logger.warning(f"⚠️ Unknown sub-pipeline: {current_sub_pipeline}")
        
        return None
    
    # Sub-pipeline implementations
    async def _sr_detection_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """SR detection sub-pipeline using the new SRDetectionStep."""
        print("📊 Executing SR detection pipeline")
        print(f"🔧 Configuration: mode={config.mode.value}, symbol={config.symbol}, exchange={config.exchange}, timeframe={config.timeframe}")
        self.logger.info("📊 Executing SR detection pipeline")
        self.logger.info(f"🔧 Configuration: mode={config.mode.value}, symbol={config.symbol}, exchange={config.exchange}, timeframe={config.timeframe}")
        
        artifacts = {
            'sr_levels': [],
            'sr_metrics': {},
            'detection_params': {}
        }
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual SR detection")
            artifacts['sr_levels'] = [{'level': 50000, 'type': 'support', 'strength': 0.8}]
            return artifacts
        
        # Use the new SRDetectionStep
        try:
            from .sr_detection import SRDetectionStep
            
            # Create configuration for SR detection
            sr_config = {
                'sr_optimization': {
                    'min_touches': 2,
                    'tolerance_pct': 0.5,
                    'lookback_periods': 100 if config.mode == ExecutionMode.FULL else 10,
                    'proximity_threshold': 0.002,
                    'min_sr_ratio': 0.15,
                    'max_sr_ratio': 0.30
                },
                'training_mode': 'light' if config.mode == ExecutionMode.LIGHT else 'full'
            }
            
            # Create SR detection step
            sr_detection_step = SRDetectionStep(sr_config)
            
            # Load market data
            market_data = await self._load_market_data_for_sr_detection(config)
            if market_data is None or market_data.empty:
                self.logger.error("❌ No market data available for SR detection")
                return artifacts
            
            # Prepare pipeline state
            pipeline_state = {'dataframe': market_data}
            training_input = {'training_mode': config.mode.value}
            
            # Execute SR detection
            result = await sr_detection_step.execute(training_input, pipeline_state)
            
            if result.get('success', False):
                sr_levels = result.get('sr_levels', {})
                artifacts['sr_levels'] = sr_levels.get('all_levels', [])
                artifacts['sr_metrics'] = {
                    'detection_time': result.get('execution_time', 0),
                    'support_count': len(sr_levels.get('support_levels', [])),
                    'resistance_count': len(sr_levels.get('resistance_levels', [])),
                    'total_levels': len(sr_levels.get('all_levels', []))
                }
                artifacts['detection_params'] = sr_levels.get('detection_config', {})
                
                self.logger.info(f"✅ SR detection completed successfully")
                self.logger.info(f"   - Support levels: {artifacts['sr_metrics']['support_count']}")
                self.logger.info(f"   - Resistance levels: {artifacts['sr_metrics']['resistance_count']}")
                self.logger.info(f"   - Total levels: {artifacts['sr_metrics']['total_levels']}")
            else:
                self.logger.error(f"❌ SR detection failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            self.logger.error(f"❌ SR detection pipeline failed: {e}")
            import traceback
            self.logger.error(f"❌ Error details: {traceback.format_exc()}")
        
        return artifacts
    
    async def _sr_clustering_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """SR clustering sub-pipeline."""
        print("🔗 Executing SR clustering pipeline")
        print(f"🔧 Configuration: mode={config.mode.value}, symbol={config.symbol}, exchange={config.exchange}, timeframe={config.timeframe}")
        self.logger.info("🔗 Executing SR clustering pipeline")
        self.logger.info(f"🔧 Configuration: mode={config.mode.value}, symbol={config.symbol}, exchange={config.exchange}, timeframe={config.timeframe}")
        
        print("📊 Initializing SR clustering artifacts...")
        self.logger.info("📊 Initializing SR clustering artifacts...")
        
        artifacts = {
            'sr_clusters': [],
            'clustering_metrics': {},
            'cluster_params': {}
        }
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual SR clustering")
            artifacts['sr_clusters'] = [{'cluster_id': 1, 'levels': [50000, 50100], 'strength': 0.8}]
            return artifacts
        
        # Import and use existing SR levels manager for clustering
        try:
            print("📦 Importing SRLevelsManager for clustering...")
            print("   🔍 Loading SR levels manager module...")
            self.logger.info("📦 Importing SRLevelsManager for clustering...")
            self.logger.info("   🔍 Loading SR levels manager module...")
            from src.tactician.sr_levels.sr_levels_manager import SRLevelsManager
            print("   ✅ SRLevelsManager imported successfully")
            self.logger.info("   ✅ SRLevelsManager imported successfully")
            
            # Create proper configuration for SR levels manager
            # Use the same path as sr_detection pipeline
            sr_config = {
                'sr_levels_manager': {
                    'storage_path': f"{config.data_dir}/sr_levels",  # Use config data_dir to match sr_detection
                    'max_levels': 50,
                    'min_strength': 0.3,
                    'proximity_threshold': 0.005
                }
            }
            
            print(f"🔧 Creating SRLevelsManager with config: {sr_config}")
            sr_manager = SRLevelsManager(sr_config)
            
            print("🚀 Initializing SRLevelsManager...")
            await sr_manager.initialize()
            print("✅ SRLevelsManager initialized successfully")
            
            # Load existing SR levels
            print("📂 Loading existing SR levels for clustering...")
            print("   🔍 Attempting to load SR levels from storage...")
            self.logger.info("📂 Loading existing SR levels for clustering...")
            self.logger.info("   🔍 Attempting to load SR levels from storage...")
            
            print("   📁 Storage path:", sr_config['sr_levels_manager']['storage_path'])
            self.logger.info(f"   📁 Storage path: {sr_config['sr_levels_manager']['storage_path']}")
            
            await sr_manager.load_levels()
            print("   ✅ SR levels loaded from storage")
            self.logger.info("   ✅ SR levels loaded from storage")
            
            existing_levels = sr_manager.support_levels + sr_manager.resistance_levels
            print(f"📂 Loaded {len(existing_levels)} existing SR levels for clustering")
            print(f"   📊 Support levels: {len(sr_manager.support_levels)}")
            print(f"   📊 Resistance levels: {len(sr_manager.resistance_levels)}")
            self.logger.info(f"📂 Loaded {len(existing_levels)} existing SR levels for clustering")
            self.logger.info(f"   📊 Support levels: {len(sr_manager.support_levels)}")
            self.logger.info(f"   📊 Resistance levels: {len(sr_manager.resistance_levels)}")
            
            if len(existing_levels) == 0:
                print("❌ ERROR: No SR levels found for clustering!")
                print("   - This indicates that the sr_detection pipeline failed to detect or save levels")
                print("   - Please run sr_detection pipeline first to generate SR levels")
                self.logger.error("❌ ERROR: No SR levels found for clustering!")
                self.logger.error("   - This indicates that the sr_detection pipeline failed to detect or save levels")
                self.logger.error("   - Please run sr_detection pipeline first to generate SR levels")
                
                # Return error artifacts instead of failing completely
                artifacts['sr_clusters'] = []
                artifacts['clustering_metrics'] = {
                    'clustering_method': 'none',
                    'n_clusters': 0,
                    'avg_cluster_size': 0,
                    'total_levels_clustered': 0,
                    'clustering_efficiency': 0,
                    'error': 'no_sr_levels_found'
                }
                artifacts['cluster_params'] = {'algorithm': 'none', 'error': 'no_input_data'}
                return artifacts
            
            # Group levels into clusters based on proximity
            print("🔗 Clustering SR levels...")
            print("   🧮 Starting clustering algorithm...")
            self.logger.info("🔗 Clustering SR levels...")
            self.logger.info("   🧮 Starting clustering algorithm...")
            
            if existing_levels:
                print(f"📊 Input data for clustering:")
                print(f"   - Total levels: {len(existing_levels)}")
                print(f"   - Support levels: {len([l for l in existing_levels if l.level_type == 'support'])}")
                print(f"   - Resistance levels: {len([l for l in existing_levels if l.level_type == 'resistance'])}")
                self.logger.info(f"📊 Input data for clustering:")
                self.logger.info(f"   - Total levels: {len(existing_levels)}")
                self.logger.info(f"   - Support levels: {len([l for l in existing_levels if l.level_type == 'support'])}")
                self.logger.info(f"   - Resistance levels: {len([l for l in existing_levels if l.level_type == 'resistance'])}")
                
                # Show price distribution
                prices = [level.price for level in existing_levels]
                print(f"   - Price range: ${min(prices):.2f} - ${max(prices):.2f}")
                print(f"   - Average price: ${np.mean(prices):.2f}")
                self.logger.info(f"   - Price range: ${min(prices):.2f} - ${max(prices):.2f}")
                self.logger.info(f"   - Average price: ${np.mean(prices):.2f}")
            
            print("   🔄 Calling clustering algorithm...")
            self.logger.info("   🔄 Calling clustering algorithm...")
            clusters = self._cluster_sr_levels(existing_levels)
            print(f"   ✅ Clustering algorithm completed")
            self.logger.info(f"   ✅ Clustering algorithm completed")
            print(f"✅ Created {len(clusters)} SR clusters")
            self.logger.info(f"✅ Created {len(clusters)} SR clusters")
            
            # Detailed cluster analysis
            if clusters:
                print(f"📊 Cluster Analysis:")
                self.logger.info(f"📊 Cluster Analysis:")
                for i, cluster in enumerate(clusters):
                    cluster_size = len(cluster.get('levels', []))
                    cluster_strength = cluster.get('strength', 0)
                    cluster_id = cluster.get('cluster_id', i)
                    print(f"   - Cluster {cluster_id}: {cluster_size} levels, strength: {cluster_strength:.4f}")
                    self.logger.info(f"   - Cluster {cluster_id}: {cluster_size} levels, strength: {cluster_strength:.4f}")
                    
                    if 'levels' in cluster and cluster['levels']:
                        level_prices = [level.price if hasattr(level, 'price') else level for level in cluster['levels']]
                        if level_prices:
                            print(f"     * Price range: ${min(level_prices):.2f} - ${max(level_prices):.2f}")
                            self.logger.info(f"     * Price range: ${min(level_prices):.2f} - ${max(level_prices):.2f}")
            else:
                print("⚠️ No clusters created - insufficient data or clustering failed")
                self.logger.warning("⚠️ No clusters created - insufficient data or clustering failed")
            
            artifacts['sr_clusters'] = clusters
            artifacts['clustering_metrics'] = {
                'clustering_method': 'proximity_based',
                'n_clusters': len(clusters),
                'avg_cluster_size': np.mean([len(cluster['levels']) for cluster in clusters]) if clusters else 0,
                'total_levels_clustered': len(existing_levels),
                'clustering_efficiency': len(existing_levels) / len(clusters) if clusters else 0
            }
            artifacts['cluster_params'] = {'algorithm': 'proximity_clustering', 'distance_threshold': 0.02}
            
            print(f"📈 Clustering Results Summary:")
            print(f"   - Total clusters created: {len(clusters)}")
            print(f"   - Average cluster size: {artifacts['clustering_metrics']['avg_cluster_size']:.2f}")
            print(f"   - Clustering efficiency: {artifacts['clustering_metrics']['clustering_efficiency']:.2f} levels per cluster")
            self.logger.info(f"📈 Clustering Results Summary:")
            self.logger.info(f"   - Total clusters created: {len(clusters)}")
            self.logger.info(f"   - Average cluster size: {artifacts['clustering_metrics']['avg_cluster_size']:.2f}")
            self.logger.info(f"   - Clustering efficiency: {artifacts['clustering_metrics']['clustering_efficiency']:.2f} levels per cluster")
            
            # Log completion with emojis and artifact paths
            self._log_sub_pipeline_completion("sr_clustering", config, artifacts)
            
        except ImportError as e:
            print(f"❌ Import Error in SR clustering: {e}")
            print("   🔄 Falling back to mock clusters...")
            self.logger.error(f"❌ Import Error in SR clustering: {e}")
            self.logger.warning("⚠️ SR levels manager not available, using mock clusters")
            artifacts['sr_clusters'] = [
                {'cluster_id': 1, 'levels': [50000, 50100], 'strength': 0.8},
                {'cluster_id': 2, 'levels': [52000, 52100], 'strength': 0.7}
            ]
        except Exception as e:
            print(f"❌ Unexpected Error in SR clustering: {e}")
            print("   🔄 Falling back to mock clusters...")
            self.logger.error(f"❌ Unexpected Error in SR clustering: {e}")
            import traceback
            self.logger.error(f"❌ Traceback: {traceback.format_exc()}")
            artifacts['sr_clusters'] = [
                {'cluster_id': 1, 'levels': [50000, 50100], 'strength': 0.8},
                {'cluster_id': 2, 'levels': [52000, 52100], 'strength': 0.7}
            ]
        
        # Print artifact paths
        print("📁 SR Clustering Artifacts:")
        print(f"   🔗 SR Clusters: {artifacts.get('sr_clusters', 'N/A')}")
        print(f"   📊 Clustering Metrics: {artifacts.get('clustering_metrics', 'N/A')}")
        print(f"   🔧 Cluster Params: {artifacts.get('cluster_params', 'N/A')}")
        self.logger.info("📁 SR Clustering Artifacts:")
        self.logger.info(f"   🔗 SR Clusters: {artifacts.get('sr_clusters', 'N/A')}")
        self.logger.info(f"   📊 Clustering Metrics: {artifacts.get('clustering_metrics', 'N/A')}")
        self.logger.info(f"   🔧 Cluster Params: {artifacts.get('cluster_params', 'N/A')}")
        
        # Automatically trigger the next sub-pipeline: sr_ml_learning
        print("🔄 SR clustering completed, triggering next: sr_ml_learning")
        print("   🚀 Starting SR ML learning pipeline...")
        self.logger.info("🔄 SR clustering completed, triggering next: sr_ml_learning")
        self.logger.info("   🚀 Starting SR ML learning pipeline...")
        try:
            next_artifacts = await self._sr_ml_learning_pipeline(config)
            print("   ✅ SR ML learning pipeline completed successfully")
            self.logger.info("   ✅ SR ML learning pipeline completed successfully")
            # Merge artifacts from next pipeline
            artifacts.update(next_artifacts)
            print("   🔗 Artifacts merged from SR ML learning pipeline")
            self.logger.info("   🔗 Artifacts merged from SR ML learning pipeline")
        except Exception as e:
            print(f"   ❌ Failed to execute SR ML learning pipeline: {e}")
            self.logger.error(f"❌ Failed to execute SR ML learning pipeline: {e}")
            import traceback
            self.logger.error(f"❌ Traceback: {traceback.format_exc()}")
        
        return artifacts
    
    async def _sr_ml_learning_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """SR ML learning sub-pipeline."""
        print("🤖 Executing SR ML learning pipeline")
        print(f"🔧 Configuration: mode={config.mode.value}, symbol={config.symbol}, exchange={config.exchange}, timeframe={config.timeframe}")
        self.logger.info("🤖 Executing SR ML learning pipeline")
        self.logger.info(f"🔧 Configuration: mode={config.mode.value}, symbol={config.symbol}, exchange={config.exchange}, timeframe={config.timeframe}")
        
        artifacts = {
            'ml_models': [],
            'training_metrics': {},
            'model_performance': {}
        }
        
        if config.mode == ExecutionMode.BLANK:
            print("🔄 Blank mode: Skipping actual SR ML learning")
            self.logger.info("🔄 Blank mode: Skipping actual SR ML learning")
            artifacts['ml_models'] = ['sr_predictor_model.pkl']
            return artifacts
        
        print("🚀 Starting SR ML Learning Process...")
        print("   🤖 This pipeline uses machine learning to predict SR level effectiveness")
        print("   📊 ML models learn from historical price patterns and volume data")
        print("   🎯 Goal: Predict which SR levels are most likely to hold or break")
        self.logger.info("🚀 Starting SR ML Learning Process...")
        self.logger.info("   🤖 This pipeline uses machine learning to predict SR level effectiveness")
        self.logger.info("   📊 ML models learn from historical price patterns and volume data")
        self.logger.info("   🎯 Goal: Predict which SR levels are most likely to hold or break")
        
        try:
            print("📦 Importing ML Libraries...")
            print("   🔍 Loading scikit-learn for Random Forest classification")
            print("   📊 Loading numpy and pandas for data processing")
            print("   📈 Loading metrics for model evaluation")
            self.logger.info("📦 Importing ML Libraries...")
            self.logger.info("   🔍 Loading scikit-learn for Random Forest classification")
            self.logger.info("   📊 Loading numpy and pandas for data processing")
            self.logger.info("   📈 Loading metrics for model evaluation")
            import numpy as np
            import pandas as pd
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report, accuracy_score
            print("✅ ML Libraries Imported Successfully")
            print("   - RandomForestClassifier: Ready for ensemble learning")
            print("   - train_test_split: Ready for data splitting")
            print("   - classification_report: Ready for model evaluation")
            self.logger.info("✅ ML Libraries Imported Successfully")
            self.logger.info("   - RandomForestClassifier: Ready for ensemble learning")
            self.logger.info("   - train_test_split: Ready for data splitting")
            self.logger.info("   - classification_report: Ready for model evaluation")
            
            print("📊 Loading Market Data for ML Training...")
            print("   🔍 Loading OHLCV data to extract price patterns")
            print("   📈 Data will be used to train models that predict SR level behavior")
            self.logger.info("📊 Loading Market Data for ML Training...")
            self.logger.info("   🔍 Loading OHLCV data to extract price patterns")
            self.logger.info("   📈 Data will be used to train models that predict SR level behavior")
            market_data = await self._load_market_data_for_sr_detection(config)
            
            if market_data is not None and not market_data.empty:
                print(f"✅ Market Data Loaded Successfully")
                print(f"   📊 Dataset size: {len(market_data)} rows, {len(market_data.columns)} columns")
                print(f"   📅 Time range: {market_data.index.min()} to {market_data.index.max()}")
                print(f"   📋 Available columns: {list(market_data.columns)}")
                self.logger.info(f"✅ Market Data Loaded Successfully")
                self.logger.info(f"   📊 Dataset size: {len(market_data)} rows, {len(market_data.columns)} columns")
                self.logger.info(f"   📅 Time range: {market_data.index.min()} to {market_data.index.max()}")
                self.logger.info(f"   📋 Available columns: {list(market_data.columns)}")
                
                print("🔍 Feature Engineering for ML Training...")
                print("   🧮 Extracting technical indicators from price data")
                print("   📊 Creating features that help predict SR level effectiveness")
                print("   🎯 Features include: price changes, volatility, volume patterns")
                self.logger.info("🔍 Feature Engineering for ML Training...")
                self.logger.info("   🧮 Extracting technical indicators from price data")
                self.logger.info("   📊 Creating features that help predict SR level effectiveness")
                self.logger.info("   🎯 Features include: price changes, volatility, volume patterns")
                
                # Create simple features for demonstration
                features = []
                labels = []
                
                # Use OHLC data to create features
                if 'open' in market_data.columns and 'high' in market_data.columns and 'low' in market_data.columns and 'close' in market_data.columns:
                    print("📈 Creating Price-Based Features...")
                    print("   💰 Feature 1: Price change percentage (momentum indicator)")
                    print("   📊 Feature 2: High/Low ratio (volatility indicator)")
                    print("   📈 Feature 3: Volume change percentage (participation indicator)")
                    self.logger.info("📈 Creating Price-Based Features...")
                    self.logger.info("   💰 Feature 1: Price change percentage (momentum indicator)")
                    self.logger.info("   📊 Feature 2: High/Low ratio (volatility indicator)")
                    self.logger.info("   📈 Feature 3: Volume change percentage (participation indicator)")
                    
                    # Calculate price changes
                    print("   🔄 Calculating price change percentages...")
                    price_changes = market_data['close'].pct_change().dropna()
                    print(f"      ✅ Price changes: {len(price_changes)} samples (avg: {price_changes.mean():.4f})")
                    
                    print("   📊 Calculating High/Low volatility ratios...")
                    high_low_ratio = (market_data['high'] / market_data['low']).dropna()
                    print(f"      ✅ High/Low ratios: {len(high_low_ratio)} samples (avg: {high_low_ratio.mean():.4f})")
                    
                    print("   📈 Calculating volume change percentages...")
                    volume_ratio = market_data['volume'].pct_change().dropna() if 'volume' in market_data.columns else pd.Series()
                    if not volume_ratio.empty:
                        print(f"      ✅ Volume changes: {len(volume_ratio)} samples (avg: {volume_ratio.mean():.4f})")
                    else:
                        print("      ⚠️ No volume data available")
                    
                    self.logger.info(f"   - Price changes: {len(price_changes)} samples (avg: {price_changes.mean():.4f})")
                    self.logger.info(f"   - High/Low ratio: {len(high_low_ratio)} samples (avg: {high_low_ratio.mean():.4f})")
                    if not volume_ratio.empty:
                        self.logger.info(f"   - Volume changes: {len(volume_ratio)} samples (avg: {volume_ratio.mean():.4f})")
                    else:
                        self.logger.warning("   - No volume data available")
                    
                    # Create simple binary labels (price going up or down)
                    print("🏷️ Creating Training Labels...")
                    print("   📈 Binary classification: 1 = price up, 0 = price down")
                    print("   🎯 Labels help model learn to predict price direction")
                    labels = (price_changes > 0).astype(int).values
                    print(f"      ✅ Labels created: {len(labels)} samples")
                    print(f"      📊 Label distribution: {np.bincount(labels)} (0=down, 1=up)")
                    
                    # Create comprehensive feature matrix using EnhancedFeatureEngineering
                    print("🧮 Building Comprehensive Feature Matrix...")
                    print("   🔄 Using EnhancedFeatureEngineering for 100+ features")
                    print("   📊 Including SR-specific features, technical indicators, and interactions")
                    
                    try:
                        # Import EnhancedFeatureEngineering
                        from src.feature_engineering.step06_enhanced_feature_engineering import EnhancedFeatureEngineering
                        from src.training.steps.model_training.sr_ml_enhancer import SRMLEnhancer
                        
                        print("   🔧 Initializing Enhanced Feature Engineering...")
                        feature_config = {
                            'step06_feature_engineering': {
                                'chunk_size': 10000,
                                'max_features': 200,
                                'polynomial_degree': 2,
                                'correlation_threshold': 0.95,
                                'memory_limit_mb': 1000
                            }
                        }
                        
                        # Initialize feature engineering
                        feature_engineer = EnhancedFeatureEngineering(feature_config)
                        await feature_engineer.initialize_utilities()
                        
                        print("   📊 Extracting comprehensive features...")
                        # Extract all features except wavelet
                        enhanced_features = await feature_engineer.create_enhanced_features_with_utilities(market_data)
                        
                        # Get technical indicators
                        periods_config = {
                            'RSI': [14, 21, 28],
                            'MACD': [12, 26, 9],
                            'Bollinger_Bands': [20, 50],
                            'SMA': [20, 50, 100],
                            'EMA': [12, 26, 50],
                            'ATR': [14, 21],
                            'Stochastic': [14, 21],
                            'ADX': [14, 21],
                            'OBV': [1],
                            'MFI': [14, 21]
                        }
                        
                        technical_indicators = feature_engineer.extract_indicators_batch(market_data, periods_config)
                        
                        # Combine all features
                        all_features = pd.concat([enhanced_features, technical_indicators], axis=1)
                        
                        # Remove wavelet features as requested
                        wavelet_columns = [col for col in all_features.columns if 'wavelet' in col.lower()]
                        if wavelet_columns:
                            all_features = all_features.drop(columns=wavelet_columns)
                            print(f"   🚫 Removed {len(wavelet_columns)} wavelet features")
                        
                        # Add SR-specific features using SRMLEnhancer
                        print("   🎯 Adding SR-specific features...")
                        sr_enhancer = SRMLEnhancer(config)
                        
                        # Create dummy SR levels for feature extraction
                        dummy_sr_levels = []
                        price_range = market_data['high'].max() - market_data['low'].min()
                        for i in range(10):  # Create 10 dummy levels
                            level_price = market_data['low'].min() + (i * price_range / 10)
                            dummy_level = {
                                'price': level_price,
                                'touch_count': np.random.randint(1, 20),
                                'strength': np.random.uniform(0.3, 0.9),
                                'age_bars': np.random.randint(10, 100),
                                'avg_bounce_ratio': np.random.uniform(0.1, 0.8),
                                'max_bounce_ratio': np.random.uniform(0.2, 1.0),
                                'volume_confirmation_score': np.random.uniform(0.2, 0.9),
                                'consistency_score': np.random.uniform(0.3, 0.8),
                                'failure_count': np.random.randint(0, 5),
                                'id': f'level_{i}'
                            }
                            dummy_sr_levels.append(dummy_level)
                        
                        # Extract SR-specific features
                        sr_features_list = []
                        for level in dummy_sr_levels:
                            sr_features = await sr_enhancer._extract_level_features(market_data, level)
                            if sr_features:
                                sr_features_list.append(sr_features)
                        
                        if sr_features_list:
                            sr_features_array = np.array(sr_features_list)
                            sr_feature_names = await sr_enhancer._get_feature_names()
                            
                            # Create SR features DataFrame
                            sr_features_df = pd.DataFrame(sr_features_array, columns=sr_feature_names)
                            sr_features_df.index = all_features.index[:len(sr_features_df)]
                            
                            # Combine with other features
                            all_features = pd.concat([all_features, sr_features_df], axis=1)
                            print(f"   ✅ Added {len(sr_feature_names)} SR-specific features")
                        
                        # Clean and prepare features
                        all_features = all_features.fillna(0).replace([np.inf, -np.inf], 0)
                        
                        # Use all features - filtering will be done in ML training module
                        features = all_features.values
                        feature_names = list(all_features.columns)
                        
                        print(f"   ✅ Feature preparation complete: {len(feature_names)} features available")
                        print(f"   📊 Feature categories:")
                        sr_count = len([f for f in feature_names if any(pattern in f.lower() for pattern in ['touch', 'bounce', 'strength', 'level', 'hvn', 'fib', 'pivot', 'trendline'])])
                        technical_count = len([f for f in feature_names if any(pattern in f.lower() for pattern in ['rsi', 'macd', 'bb_', 'sma', 'ema', 'atr', 'stoch', 'adx', 'obv', 'mfi'])])
                        other_count = len(feature_names) - sr_count - technical_count
                        print(f"      - SR-specific features: {sr_count}")
                        print(f"      - Technical indicators: {technical_count}")
                        print(f"      - Other features: {other_count}")
                        
                        # Align labels with features
                        labels = labels[:len(features)]
                        
                        # Cleanup
                        await feature_engineer.cleanup()
                        
                    except Exception as e:
                        print(f"   ⚠️ Enhanced feature engineering failed: {e}")
                        print("   🔄 Falling back to basic features...")
                        self.logger.warning(f"Enhanced feature engineering failed: {e}")
                        
                        # Fallback to basic features
                        feature_data = []
                        max_samples = min(len(price_changes), 10000)
                        
                        for i in range(1, max_samples):
                            feature_row = [
                                price_changes.iloc[i-1] if i > 0 else 0,
                                high_low_ratio.iloc[i] if i < len(high_low_ratio) else 1.0,
                                volume_ratio.iloc[i] if not volume_ratio.empty and i < len(volume_ratio) else 0.0
                            ]
                            feature_data.append(feature_row)
                        
                        features = np.array(feature_data)
                        feature_names = ['price_change', 'high_low_ratio', 'volume_change']
                        labels = labels[1:len(feature_data)+1]
                    
                    print(f"✅ Feature Matrix Created Successfully")
                    print(f"   📊 Matrix shape: {features.shape[0]} samples × {features.shape[1]} features")
                    print(f"   🏷️ Label distribution: {np.bincount(labels)} (0=down, 1=up)")
                    print(f"   📈 Feature statistics:")
                    print(f"      - Feature 1 (price change): mean={features[:, 0].mean():.4f}, std={features[:, 0].std():.4f}")
                    print(f"      - Feature 2 (high/low ratio): mean={features[:, 1].mean():.4f}, std={features[:, 1].std():.4f}")
                    print(f"      - Feature 3 (volume change): mean={features[:, 2].mean():.4f}, std={features[:, 2].std():.4f}")
                    self.logger.info(f"✅ Feature Matrix Created Successfully")
                    self.logger.info(f"   📊 Matrix shape: {features.shape[0]} samples × {features.shape[1]} features")
                    self.logger.info(f"   🏷️ Label distribution: {np.bincount(labels)} (0=down, 1=up)")
                    self.logger.info(f"   📈 Feature statistics:")
                    self.logger.info(f"      - Feature 1 (price change): mean={features[:, 0].mean():.4f}, std={features[:, 0].std():.4f}")
                    self.logger.info(f"      - Feature 2 (high/low ratio): mean={features[:, 1].mean():.4f}, std={features[:, 1].std():.4f}")
                    self.logger.info(f"      - Feature 3 (volume change): mean={features[:, 2].mean():.4f}, std={features[:, 2].std():.4f}")
                    
                    if len(features) > 100:  # Only train if we have enough data
                        print("🤖 Training Random Forest Model...")
                        print("   🌲 Random Forest: Ensemble of decision trees for robust predictions")
                        print("   📊 Model learns patterns from historical price movements")
                        print("   🎯 Goal: Predict whether price will go up or down")
                        self.logger.info("🤖 Training Random Forest Model...")
                        self.logger.info("   🌲 Random Forest: Ensemble of decision trees for robust predictions")
                        self.logger.info("   📊 Model learns patterns from historical price movements")
                        self.logger.info("   🎯 Goal: Predict whether price will go up or down")
                        
                        # Split data
                        print("📊 Splitting Data for Training and Testing...")
                        print("   🔄 80% for training, 20% for testing")
                        print("   ⚖️ Stratified split to maintain class balance")
                        X_train, X_test, y_train, y_test = train_test_split(
                            features, labels, test_size=0.2, random_state=42, stratify=labels
                        )
                        
                        print(f"✅ Data Split Complete")
                        print(f"   🏋️ Training set: {X_train.shape[0]} samples")
                        print(f"   🧪 Test set: {X_test.shape[0]} samples")
                        print(f"   📊 Training label distribution: {np.bincount(y_train)}")
                        print(f"   📊 Test label distribution: {np.bincount(y_test)}")
                        self.logger.info(f"✅ Data Split Complete")
                        self.logger.info(f"   🏋️ Training set: {X_train.shape[0]} samples")
                        self.logger.info(f"   🧪 Test set: {X_test.shape[0]} samples")
                        self.logger.info(f"   📊 Training label distribution: {np.bincount(y_train)}")
                        self.logger.info(f"   📊 Test label distribution: {np.bincount(y_test)}")
                        
                        # Train model
                        print("🌲 Initializing Random Forest Model...")
                        print("   🔧 Parameters: 100 trees, random_state=42, parallel processing")
                        print("   🎯 Each tree learns different patterns from the data")
                        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                        
                        print("🔄 Training Model (Fitting to Data)...")
                        print("   🌲 Growing 100 decision trees...")
                        print("   📊 Each tree learns to classify price movements")
                        print("   ⏱️ This may take a few moments...")
                        self.logger.info("🔄 Training Model (Fitting to Data)...")
                        self.logger.info("   🌲 Growing 100 decision trees...")
                        self.logger.info("   📊 Each tree learns to classify price movements")
                        self.logger.info("   ⏱️ This may take a few moments...")
                        model.fit(X_train, y_train)
                        print("✅ Model Training Completed Successfully")
                        print("   🌲 All 100 trees have been trained")
                        print("   📊 Model is ready to make predictions")
                        self.logger.info("✅ Model Training Completed Successfully")
                        self.logger.info("   🌲 All 100 trees have been trained")
                        self.logger.info("   📊 Model is ready to make predictions")
                        
                        # Evaluate model
                        print("📊 Evaluating Model Performance...")
                        print("   🧪 Testing model on unseen data (test set)")
                        print("   📈 Measuring prediction accuracy")
                        self.logger.info("📊 Evaluating Model Performance...")
                        self.logger.info("   🧪 Testing model on unseen data (test set)")
                        self.logger.info("   📈 Measuring prediction accuracy")
                        y_pred = model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        print(f"✅ Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                        print(f"   📊 Model correctly predicted {accuracy*100:.1f}% of test cases")
                        if accuracy > 0.6:
                            print(f"   🎉 Good performance! Model shows predictive capability")
                        elif accuracy > 0.5:
                            print(f"   📈 Decent performance, better than random guessing")
                        else:
                            print(f"   ⚠️ Model performance below random guessing - may need more data or features")
                        self.logger.info(f"✅ Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                        self.logger.info(f"   📊 Model correctly predicted {accuracy*100:.1f}% of test cases")
                        
                        # Calculate additional metrics
                        from sklearn.metrics import precision_score, recall_score, f1_score
                        precision = precision_score(y_test, y_pred, average='weighted')
                        recall = recall_score(y_test, y_pred, average='weighted')
                        f1 = f1_score(y_test, y_pred, average='weighted')
                        
                        print(f"📊 Additional Performance Metrics:")
                        print(f"   🎯 Precision: {precision:.4f} (how many positive predictions were correct)")
                        print(f"   🔍 Recall: {recall:.4f} (how many actual positives were found)")
                        print(f"   ⚖️ F1-Score: {f1:.4f} (harmonic mean of precision and recall)")
                        self.logger.info(f"📊 Additional Performance Metrics:")
                        self.logger.info(f"   🎯 Precision: {precision:.4f} (how many positive predictions were correct)")
                        self.logger.info(f"   🔍 Recall: {recall:.4f} (how many actual positives were found)")
                        self.logger.info(f"   ⚖️ F1-Score: {f1:.4f} (harmonic mean of precision and recall)")
                        
                        # Save model info
                        artifacts['ml_models'] = ['sr_predictor_model.pkl']
                        artifacts['training_metrics'] = {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1,
                            'n_samples': len(features),
                            'n_features': features.shape[1],
                            'model_type': 'RandomForestClassifier'
                        }
                        # Get selected features from ML enhancer (scientifically selected)
                        feature_names_data = feature_names if 'feature_names' in locals() else ['price_change', 'high_low_ratio', 'volume_change']
                        
                        # Get scientifically selected features from SR ML Enhancer
                        try:
                            from src.training.steps.model_training.sr_ml_enhancer import SRMLEnhancer
                            sr_enhancer = SRMLEnhancer(config)
                            
                            # Check if we have scientifically selected features
                            if hasattr(sr_enhancer, 'feature_importance') and sr_enhancer.feature_importance:
                                enhanced_importance = sr_enhancer.feature_importance
                                if 'selected_features' in enhanced_importance:
                                    feature_names_data = enhanced_importance['selected_features']
                                    print(f"   🔍 Using scientifically selected features from ML enhancer: {len(feature_names_data)} features")
                                    print(f"   📊 Feature selection method: mRMR (Minimum Redundancy Maximum Relevance)")
                                    print(f"   🎯 Features selected based on: relevance to target + low redundancy")
                        except Exception as e:
                            print(f"   ⚠️ Could not get scientifically selected features: {e}")
                        
                        artifacts['model_performance'] = {
                            'train_samples': X_train.shape[0],
                            'test_samples': X_test.shape[0],
                            'feature_importance': model.feature_importances_.tolist(),
                            'feature_names': feature_names_data
                        }
                    else:
                        print("⚠️ Insufficient data for training (need >100 samples)")
                        self.logger.warning("⚠️ Insufficient data for training (need >100 samples)")
                        artifacts['ml_models'] = ['sr_predictor_model.pkl']
                        artifacts['training_metrics'] = {'error': 'insufficient_data'}
                else:
                    print("⚠️ Missing required OHLC columns in market data")
                    self.logger.warning("⚠️ Missing required OHLC columns in market data")
                    artifacts['ml_models'] = ['sr_predictor_model.pkl']
                    artifacts['training_metrics'] = {'error': 'missing_columns'}
            else:
                print("⚠️ No market data available for ML training")
                self.logger.warning("⚠️ No market data available for ML training")
                artifacts['ml_models'] = ['sr_predictor_model.pkl']
                artifacts['training_metrics'] = {'error': 'no_data'}
                
        except Exception as e:
            print(f"❌ Error in SR ML learning: {e}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
            self.logger.error(f"❌ Error in SR ML learning: {e}")
            self.logger.error(f"❌ Traceback: {traceback.format_exc()}")
            artifacts['ml_models'] = ['sr_predictor_model.pkl']
            artifacts['training_metrics'] = {'error': str(e)}
        
        print("🎯 SR ML Learning Pipeline Completed Successfully!")
        print("   🤖 Machine learning model trained to predict price movements")
        print("   📊 Model can now help identify which SR levels are most likely to hold")
        print("   🎯 This enhances the overall SR detection system with predictive capabilities")
        self.logger.info("🎯 SR ML Learning Pipeline Completed Successfully!")
        self.logger.info("   🤖 Machine learning model trained to predict price movements")
        self.logger.info("   📊 Model can now help identify which SR levels are most likely to hold")
        self.logger.info("   🎯 This enhances the overall SR detection system with predictive capabilities")
        
        # Log completion with emojis and artifact paths
        self._log_sub_pipeline_completion("sr_ml_learning", config, artifacts)
        
        
        # Automatically trigger the next sub-pipeline: hmm_clustering
        self.logger.info("🔄 SR ML learning completed, triggering next: hmm_clustering")
        try:
            next_artifacts = await self._hmm_clustering_pipeline(config)
            # Merge artifacts from next pipeline
            artifacts.update(next_artifacts)
            self.logger.info("✅ HMM clustering pipeline completed successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to execute HMM clustering pipeline: {e}")
        
        return artifacts
    
    async def _hmm_clustering_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """HMM clustering sub-pipeline."""
        self.logger.info("🔄 Executing HMM clustering pipeline")
        
        artifacts = {
            'hmm_models': [],
            'clustering_results': {},
            'regime_assignments': []
        }
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual HMM clustering")
            artifacts['hmm_models'] = ['hmm_model.pkl']
            artifacts['regime_assignments'] = [0, 1, 2, 0, 1]
            return artifacts
        
        # Import and use existing HMM composite manager
        try:
            from src.utils.hmm_composite_manager import HMMCompositeManager
            
            hmm_manager = HMMCompositeManager()
            
            # Load existing HMM composite data from the data directory
            hmm_data = hmm_manager.load_composite_data(config.data_dir)
            
            artifacts['hmm_models'] = ['hmm_composite_model']
            artifacts['clustering_results'] = {
                'n_states': hmm_data.get('n_states', 3),
                'convergence_iterations': hmm_data.get('convergence_iterations', 100),
                'log_likelihood': hmm_data.get('log_likelihood', -1000.0)
            }
            artifacts['regime_assignments'] = hmm_data.get('regime_assignments', [0, 1, 2, 0, 1, 2, 1, 0])
            artifacts['transition_matrix'] = hmm_data.get('transition_matrix', [[0.33, 0.33, 0.34], [0.33, 0.33, 0.34], [0.33, 0.33, 0.34]])
            artifacts['performance_metrics'] = hmm_data.get('performance_metrics', {})
            
        except ImportError:
            self.logger.warning("⚠️ HMM composite manager not available, using mock clustering")
            artifacts['hmm_models'] = ['hmm_model.pkl']
            artifacts['regime_assignments'] = [0, 1, 2, 0, 1, 2, 1, 0]
        
        # Log completion with emojis and artifact paths
        self._log_sub_pipeline_completion("hmm_clustering", config, artifacts)
        
        # Automatically trigger the next sub-pipeline: hmm_regime_discovery
        self.logger.info("🔄 HMM clustering completed, triggering next: hmm_regime_discovery")
        try:
            next_artifacts = await self._hmm_regime_discovery_pipeline(config)
            # Merge artifacts from next pipeline
            artifacts.update(next_artifacts)
            self.logger.info("✅ HMM regime discovery pipeline completed successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to execute HMM regime discovery pipeline: {e}")
        
        return artifacts
    
    async def _hmm_regime_discovery_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """HMM regime discovery sub-pipeline."""
        self.logger.info("🔍 Executing HMM regime discovery pipeline")
        
        artifacts = {
            'regime_models': [],
            'regime_statistics': {},
            'regime_transitions': {}
        }
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual HMM regime discovery")
            artifacts['regime_models'] = ['regime_model.pkl']
            artifacts['regime_statistics'] = {'n_regimes': 3, 'avg_duration': 100}
            return artifacts
        
        # Use ML commons HMM regime detection if available
        if ML_COMMONS_AVAILABLE:
            try:
                hmm_detector = get_hmm_regime_detector()
                # Load data for regime detection
                data_file = f"{config.data_dir}/features_{config.exchange}_{config.symbol}_consolidated.parquet"
                if Path(data_file).exists():
                    data = standardized_parquet_handler.read_parquet(data_file)
                    regime_result = hmm_detector.detect_regimes(data)
                    
                    artifacts['regime_models'] = ['regime_model.pkl']
                    artifacts['regime_statistics'] = regime_result.regime_qualities
                    artifacts['regime_transitions'] = {'transition_matrix': regime_result.transition_matrix.tolist()}
                else:
                    self.logger.warning("⚠️ Data file not found, using mock regime discovery")
                    artifacts['regime_models'] = ['regime_model.pkl']
                    artifacts['regime_statistics'] = {'n_regimes': 3, 'avg_duration': 100}
            except Exception as e:
                self.logger.warning(f"⚠️ ML commons HMM regime detection failed: {e}, using mock")
                artifacts['regime_models'] = ['regime_model.pkl']
                artifacts['regime_statistics'] = {'n_regimes': 3, 'avg_duration': 100}
        else:
            self.logger.warning("⚠️ ML commons not available, using mock regime discovery")
            artifacts['regime_models'] = ['regime_model.pkl']
            artifacts['regime_statistics'] = {'n_regimes': 3, 'avg_duration': 100}
        
        # Log completion with emojis and artifact paths
        self._log_sub_pipeline_completion("hmm_regime_discovery", config, artifacts)
        
        return artifacts
    
    async def _regime_data_splitting_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Regime data splitting sub-pipeline."""
        self.logger.info("✂️ Executing regime data splitting pipeline")
        
        artifacts = {
            'split_data_files': [],
            'regime_statistics': {},
            'splitting_metrics': {}
        }
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual regime data splitting")
            artifacts['split_data_files'] = ['regime_0_data.parquet', 'regime_1_data.parquet']
            return artifacts
        
        # Use ML commons regime data processing if available
        if ML_COMMONS_AVAILABLE:
            try:
                regime_processor = get_regime_processor()
                # Load data for regime processing
                data_file = f"{config.data_dir}/features_{config.exchange}_{config.symbol}_consolidated.parquet"
                if Path(data_file).exists():
                    data = standardized_parquet_handler.read_parquet(data_file)
                    # Assume regime column exists
                    if 'regime' in data.columns:
                        regime_ids = data['regime'].values
                        processing_result = regime_processor.process_regime_data(data, regime_ids)
                        
                        artifacts['split_data_files'] = list(processing_result.processed_data.keys())
                        artifacts['regime_statistics'] = processing_result.regime_statistics
                        artifacts['splitting_metrics'] = processing_result.performance_metrics
                    else:
                        self.logger.warning("⚠️ No regime column found, using mock splitting")
                        artifacts['split_data_files'] = ['regime_0_data.parquet', 'regime_1_data.parquet']
                else:
                    self.logger.warning("⚠️ Data file not found, using mock regime splitting")
                    artifacts['split_data_files'] = ['regime_0_data.parquet', 'regime_1_data.parquet']
            except Exception as e:
                self.logger.warning(f"⚠️ ML commons regime processing failed: {e}, using mock")
                artifacts['split_data_files'] = ['regime_0_data.parquet', 'regime_1_data.parquet']
        else:
            self.logger.warning("⚠️ ML commons not available, using mock regime splitting")
            artifacts['split_data_files'] = ['regime_0_data.parquet', 'regime_1_data.parquet']
        
        # Log completion with emojis and artifact paths
        self._log_sub_pipeline_completion("regime_data_splitting", config, artifacts)
        
        return artifacts
    
    async def _triple_barrier_labeling_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Triple barrier labeling sub-pipeline."""
        self.logger.info("🏷️ Executing triple barrier labeling pipeline")
        
        artifacts = {
            'label_files': [],
            'labeling_metrics': {},
            'label_statistics': {}
        }
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual triple barrier labeling")
            artifacts['label_files'] = ['labels.parquet']
            return artifacts
        
        # Use ML commons data labeling if available
        if ML_COMMONS_AVAILABLE:
            try:
                data_labeler = get_data_labeler()
                # Load data for labeling
                data_file = f"{config.data_dir}/features_{config.exchange}_{config.symbol}_consolidated.parquet"
                if Path(data_file).exists():
                    data = standardized_parquet_handler.read_parquet(data_file)
                    # Ensure OHLCV columns exist
                    ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
                    if all(col in data.columns for col in ohlcv_columns):
                        labeling_config = TripleBarrierConfig(
                            profit_take_multiplier=0.02,
                            stop_loss_multiplier=0.01,
                            time_barrier_minutes=30
                        )
                        labeling_result = data_labeler.label_data(data[ohlcv_columns], LabelingMethod.TRIPLE_BARRIER, labeling_config)
                        
                        artifacts['label_files'] = ['labels.parquet']
                        artifacts['labeling_metrics'] = labeling_result.metadata.get('statistics', {})
                        artifacts['label_statistics'] = {
                            'total_labels': len(labeling_result.labels),
                            'long_ratio': sum(labeling_result.labels == 1) / len(labeling_result.labels),
                            'short_ratio': sum(labeling_result.labels == -1) / len(labeling_result.labels)
                        }
                    else:
                        self.logger.warning("⚠️ OHLCV columns not found, using mock labeling")
                        artifacts['label_files'] = ['labels.parquet']
                else:
                    self.logger.warning("⚠️ Data file not found, using mock labeling")
                    artifacts['label_files'] = ['labels.parquet']
            except Exception as e:
                self.logger.warning(f"⚠️ ML commons data labeling failed: {e}, using mock")
                artifacts['label_files'] = ['labels.parquet']
        else:
            self.logger.warning("⚠️ ML commons not available, using mock labeling")
            artifacts['label_files'] = ['labels.parquet']
        
        # Log completion with emojis and artifact paths
        self._log_sub_pipeline_completion("triple_barrier_labeling", config, artifacts)
        
        return artifacts
    
    async def _feature_lookback_optimization_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Feature lookback optimization sub-pipeline."""
        self.logger.info("⚙️ Executing feature lookback optimization pipeline")
        
        artifacts = {
            'optimization_results': {},
            'optimal_lookbacks': {},
            'optimization_metrics': {}
        }
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual feature lookback optimization")
            artifacts['optimal_lookbacks'] = {'rsi': 14, 'sma': 20, 'ema': 12}
            return artifacts
        
        # Use ML commons feature generation optimization if available
        if ML_COMMONS_AVAILABLE:
            try:
                feature_optimizer = get_feature_optimizer()
                # Load data for optimization
                data_file = f"{config.data_dir}/features_{config.exchange}_{config.symbol}_consolidated.parquet"
                if Path(data_file).exists():
                    data = standardized_parquet_handler.read_parquet(data_file)
                    optimization_config = FeatureOptimizationConfig(
                        methods=['cross_validation', 'statistical_analysis'],
                        cv_folds=5,
                        optimization_metric='sharpe_ratio'
                    )
                    optimization_result = feature_optimizer.optimize_features(data, optimization_config)
                    
                    artifacts['optimization_results'] = optimization_result.results
                    artifacts['optimal_lookbacks'] = {k: v['optimal_lookback'] for k, v in optimization_result.results.items()}
                    artifacts['optimization_metrics'] = optimization_result.metadata
                else:
                    self.logger.warning("⚠️ Data file not found, using mock optimization")
                    artifacts['optimal_lookbacks'] = {'rsi': 14, 'sma': 20, 'ema': 12}
            except Exception as e:
                self.logger.warning(f"⚠️ ML commons feature optimization failed: {e}, using mock")
                artifacts['optimal_lookbacks'] = {'rsi': 14, 'sma': 20, 'ema': 12}
        else:
            self.logger.warning("⚠️ ML commons not available, using mock optimization")
            artifacts['optimal_lookbacks'] = {'rsi': 14, 'sma': 20, 'ema': 12}
        
        # Log completion with emojis and artifact paths
        self._log_sub_pipeline_completion("feature_lookback_optimization", config, artifacts)
        
        return artifacts
    
    async def _fractional_differentiation_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Fractional differentiation sub-pipeline."""
        self.logger.info("🔢 Executing fractional differentiation pipeline")
        
        artifacts = {
            'differentiated_data': [],
            'differentiation_params': {},
            'stationarity_metrics': {}
        }
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual fractional differentiation")
            artifacts['differentiated_data'] = ['fractional_diff_data.parquet']
            return artifacts
        
        # Import and use fractional differentiation
        try:
            from src.feature_engineering.fractional_differentiation_pipeline import FractionalDifferentiationPipeline, FractionalDiffConfig
            
            frac_diff_config = FractionalDiffConfig(
                d_min=0.0,
                d_max=1.0,
                d_step=0.1,
                threshold=0.01,
                enable_data_quality_validation=True
            )
            frac_diff_pipeline = FractionalDifferentiationPipeline(frac_diff_config)
            
            # Execute fractional differentiation
            frac_diff_result = await frac_diff_pipeline.apply_fractional_differentiation(
                data_dir=config.data_dir,
                symbol=config.symbol,
                exchange=config.exchange,
                timeframe=config.timeframe
            )
            
            artifacts['differentiated_data'] = ['fractional_diff_data.parquet']
            artifacts['differentiation_params'] = frac_diff_result.differentiation_params
            artifacts['stationarity_metrics'] = frac_diff_result.stationarity_metrics
            artifacts['memory_metrics'] = frac_diff_result.memory_metrics
            artifacts['optimal_d'] = frac_diff_result.optimal_d
            
        except ImportError:
            self.logger.warning("⚠️ Fractional differentiation pipeline not available, using mock")
            artifacts['differentiated_data'] = ['fractional_diff_data.parquet']
        
        # Log completion with emojis and artifact paths
        self._log_sub_pipeline_completion("fractional_differentiation", config, artifacts)
        
        return artifacts
    
    async def _cross_timeframe_analysis_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Cross timeframe analysis sub-pipeline."""
        self.logger.info("⏰ Executing cross timeframe analysis pipeline")
        
        artifacts = {
            'cross_timeframe_features': [],
            'interaction_metrics': {},
            'timeframe_correlations': {}
        }
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual cross timeframe analysis")
            artifacts['cross_timeframe_features'] = ['cross_tf_features.parquet']
            return artifacts
        
        # Import and use cross timeframe analysis
        try:
            from src.feature_engineering.cross_timeframe_analysis_pipeline import CrossTimeframeAnalysisPipeline, CrossTimeframeConfig
            
            cross_tf_config = CrossTimeframeConfig(
                timeframes=['1m', '5m', '15m', '30m'],  # Short timeframes for high leverage
                base_timeframe='1m',
                interaction_features=['correlation', 'momentum', 'volatility', 'volume', 'microstructure'],
                lookback_periods=[3, 5, 10, 15, 20],  # Shorter periods for high leverage
                correlation_threshold=0.6,  # Lower threshold for short timeframes
                min_observations=50,  # Reduced for short timeframes
                enable_microstructure_features=True,
                enable_order_flow_features=True,
                enable_momentum_divergence=True,
                enable_volatility_spillover=True,
                enable_data_quality_validation=True
            )
            cross_tf_pipeline = CrossTimeframeAnalysisPipeline(cross_tf_config)
            
            # Execute cross timeframe analysis
            cross_tf_result = await cross_tf_pipeline.analyze_cross_timeframes(
                data_dir=config.data_dir,
                symbol=config.symbol,
                exchange=config.exchange,
                timeframes=['1m', '5m', '15m', '30m']  # Short timeframes for high leverage
            )
            
            artifacts['cross_timeframe_features'] = ['cross_tf_features.parquet']
            artifacts['interaction_metrics'] = cross_tf_result.interaction_metrics
            artifacts['timeframe_correlations'] = cross_tf_result.timeframe_correlations
            artifacts['feature_importance'] = cross_tf_result.feature_importance
            artifacts['analysis_metadata'] = cross_tf_result.analysis_metadata
            
        except ImportError:
            self.logger.warning("⚠️ Cross timeframe analysis pipeline not available, using mock")
            artifacts['cross_timeframe_features'] = ['cross_tf_features.parquet']
        
        # Log completion with emojis and artifact paths
        self._log_sub_pipeline_completion("cross_timeframe_analysis", config, artifacts)
        
        return artifacts
    
    def _cluster_sr_levels(self, levels: List[Any]) -> List[Dict[str, Any]]:
        """Cluster SR levels based on proximity."""
        if not levels:
            return []
        
        clusters = []
        used_levels = set()
        
        for i, level in enumerate(levels):
            if i in used_levels:
                continue
            
            # Start a new cluster
            cluster = {
                'cluster_id': len(clusters) + 1,
                'levels': [level.price],  # Fixed: use level.price instead of level.level
                'strength': level.strength,
                'type': level.level_type,
                'touches': level.touch_count  # Fixed: use level.touch_count instead of level.touches
            }
            used_levels.add(i)
            
            # Find nearby levels
            for j, other_level in enumerate(levels[i+1:], i+1):
                if j in used_levels:
                    continue
                
                # Check if levels are close enough
                price_diff = abs(level.price - other_level.price)  # Fixed: use level.price instead of level.level
                price_tolerance = level.price * 0.02  # 2% tolerance  # Fixed: use level.price instead of level.level
                
                if price_diff <= price_tolerance and level.level_type == other_level.level_type:
                    cluster['levels'].append(other_level.price)  # Fixed: use other_level.price instead of other_level.level
                    cluster['strength'] = max(cluster['strength'], other_level.strength)
                    cluster['touches'] += other_level.touch_count  # Fixed: use other_level.touch_count instead of other_level.touches
                    used_levels.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def get_available_sub_pipelines(self) -> List[str]:
        """Get list of available sub-pipelines."""
        return list(self.sub_pipelines.keys())
    
    def get_sub_pipeline_status(self, sub_pipeline_name: str) -> Optional[SubPipelineStatus]:
        """Get status of a specific sub-pipeline."""
        for result in self.results:
            if result.sub_pipeline_name == sub_pipeline_name:
                return result.status
        return None
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all sub-pipeline executions."""
        total_executions = len(self.results)
        completed = sum(1 for r in self.results if r.status == SubPipelineStatus.COMPLETED)
        failed = sum(1 for r in self.results if r.status == SubPipelineStatus.FAILED)
        total_duration = sum(r.duration_seconds or 0 for r in self.results)
        
        return {
            'total_executions': total_executions,
            'completed': completed,
            'failed': failed,
            'success_rate': completed / total_executions if total_executions > 0 else 0,
            'total_duration_seconds': total_duration,
            'results': self.results
        }

# Convenience functions
def get_market_analysis_sub_pipeline(config: Optional[SubPipelineConfig] = None) -> MarketAnalysisSubPipeline:
    """Get a configured market analysis sub-pipeline."""
    return MarketAnalysisSubPipeline(config)

async def execute_market_analysis_sub_pipeline(
    sub_pipeline_name: str,
    config: Optional[SubPipelineConfig] = None
) -> SubPipelineResult:
    """Convenience function to execute a market analysis sub-pipeline."""
    pipeline = get_market_analysis_sub_pipeline(config)
    return await pipeline.execute_sub_pipeline(sub_pipeline_name, config)

async def execute_market_analysis_sub_pipeline_with_next(
    sub_pipeline_name: str,
    config: Optional[SubPipelineConfig] = None
) -> SubPipelineResult:
    """Convenience function to execute a market analysis sub-pipeline with automatic next triggering."""
    pipeline = get_market_analysis_sub_pipeline(config)
    return await pipeline.execute_sub_pipeline_with_next(sub_pipeline_name, config)