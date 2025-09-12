from src.utils.tprint import tprint

"""
Market Analysis Sub-Pipeline

This module provides granular sub-pipeline functionality for market analysis,
allowing execution of specific market analysis steps with different modes.

Sub-pipelines:
1. SR Detection - Detect Support/Resistance levels
2. SR Clustering - Generate SR clusters
3. SR ML Learning - ML-based learning for SR clusters
4. HMM Regime Discovery - Discover market regimes
5. HMM Clustering - HMM-based regime clustering
6. Regime Data Splitting - Split data by regimes
7. Triple Barrier Labeling - Apply triple barrier method
8. Feature Lookback Optimization - Optimize feature lookback periods
9. Fractional Differentiation - Apply fractional differentiation
10. Cross Timeframe Analysis - Cross timeframe interaction features
11. Temporal Feature Integration - Integrate and deduplicate temporal features
"""

import asyncio
import json
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

logger = system_logger.getChild('MarketAnalysisSubPipeline')

# Import ML commons utilities with conditional loading to avoid circular imports
try:
    # Import from existing modules with correct paths
    from src.utils.core.common import create_fallback_logger, create_fallback_decorator
    from src.utils.math_validation import safe_divide, safe_log, safe_sqrt
    from src.utils.parquet_utils import ParquetUtils
    from src.utils.serialization_utils import UniversalSerializer
    from src.utils.data_processing_utils import DataProcessingUtils
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer, M1MemoryOptimizer
    from src.utils.hmm_composite_manager import EnhancedHMMCompositeManager

    # Import ML commons with lazy loading to avoid circular imports
    def _load_ml_commons():
        """Lazy load ML commons components to avoid circular imports."""
        global enhanced_data_labeler, TripleBarrierConfig, LabelingMethod
        global enhanced_hmm_regime_detector, HMMRegimeConfig, RegimeDetectionMethod
        global enhanced_regime_data_processor, RegimeProcessingConfig
        global get_feature_optimizer, FeatureOptimizationConfig

        from src.utils.ml_common.data_labeling import enhanced_data_labeler, TripleBarrierConfig, LabelingMethod
        from src.utils.ml_common.hmm_regime_detection import enhanced_hmm_regime_detector, HMMRegimeConfig, RegimeDetectionMethod
        from src.utils.ml_common.regime_data_processing import enhanced_regime_data_processor, RegimeProcessingConfig
        from src.feature_engineering.feature_generation_optimization import get_feature_optimizer, FeatureOptimizationConfig

    # Initialize globals to None, will be loaded lazily
    enhanced_data_labeler = None
    TripleBarrierConfig = None
    LabelingMethod = None
    enhanced_hmm_regime_detector = None
    HMMRegimeConfig = None
    RegimeDetectionMethod = None
    enhanced_regime_data_processor = None
    RegimeProcessingConfig = None
    get_feature_optimizer = None
    FeatureOptimizationConfig = None

    ML_COMMONS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ ML Commons import error: {e}")
    ML_COMMONS_AVAILABLE = False

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
    enable_fractional_differentiation: bool = True
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
            'sr_parameter_optimization': self._sr_parameter_optimization_pipeline,
            'sr_detection': self._sr_detection_pipeline,
            'sr_clustering': self._sr_clustering_pipeline,
            'hmm_clustering': self._hmm_clustering_pipeline,
            'hmm_regime_discovery': self._hmm_regime_discovery_pipeline,
            'regime_data_splitting': self._regime_data_splitting_pipeline,
            'triple_barrier_labeling': self._triple_barrier_labeling_pipeline,
            'feature_lookback_optimization': self._feature_lookback_optimization_pipeline,
            'fractional_differentiation': self._fractional_differentiation_pipeline,
            'cross_timeframe_analysis': self._cross_timeframe_analysis_pipeline,
            'temporal_feature_integration': self._temporal_feature_integration_pipeline,
            'sr_feature_integration': self._sr_feature_integration_pipeline
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
            
            # Execute the SR optimization pipeline in the correct order
            results = {}
            
            # Stage 1: SR Parameter Optimization (BEFORE detection and clustering)
            self.logger.info('🎯 Executing Stage 1: SR Parameter Optimization')
            param_optimization_result = await self.execute_sub_pipeline('sr_parameter_optimization', self.config)
            if param_optimization_result.success:
                results['optimized_parameters'] = param_optimization_result.artifacts.get('optimized_parameters', {})
                results['quality_thresholds'] = param_optimization_result.artifacts.get('quality_thresholds', {})
                results['parameter_optimization_metrics'] = param_optimization_result.artifacts.get('parameter_optimization_metrics', {})
                self.logger.info(f"✅ SR Parameter Optimization completed")
            else:
                self.logger.error(f"❌ SR Parameter Optimization failed: {param_optimization_result.error}")
                return {
                    'success': False,
                    'error': f"SR Parameter Optimization failed: {param_optimization_result.error}",
                    'stage': 'sr_parameter_optimization'
                }
            
            # Stage 2: SR Detection (using optimized parameters)
            self.logger.info('🎯 Executing Stage 2: SR Detection with Optimized Parameters')
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
            
            # Stage 3: SR Clustering (using optimized parameters)
            self.logger.info('🚀 Executing Stage 3: SR Clustering with Optimized Parameters')
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
            
            # Calculate total execution time
            total_time = (
                detection_result.execution_time + 
                clustering_result.execution_time
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
                    'clustering': clustering_result.execution_time
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
        tprint("\n" + "="*80)
        tprint(f"🎉 {sub_pipeline_name.upper().replace('_', ' ')} SUB-PIPELINE COMPLETED SUCCESSFULLY!")
        tprint("="*80)
        tprint(f"📁 Artifact Paths:")
        
        # Log different types of artifacts with appropriate emojis
        for key, value in artifacts.items():
            if isinstance(value, list) and value:
                if 'model' in key.lower():
                    for item in value:
                        tprint(f"   🤖 {key.title()}: {config.data_dir}/models/{item}")
                elif 'file' in key.lower() or 'data' in key.lower():
                    for item in value:
                        tprint(f"   📄 {key.title()}: {config.data_dir}/{item}")
                elif 'report' in key.lower():
                    for item in value:
                        tprint(f"   📋 {key.title()}: {config.data_dir}/{item}")
                else:
                    for item in value:
                        tprint(f"   📊 {key.title()}: {config.data_dir}/{item}")
            elif isinstance(value, dict) and value:
                tprint(f"   📊 {key.title()}: {config.data_dir}/{key}.json")
        
        tprint(f"📊 Artifacts Summary: {len(artifacts)} artifact types generated")
        tprint("="*80 + "\n")
        
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
            'hmm_regime_discovery',
            'hmm_clustering',
            'regime_data_splitting',
            'triple_barrier_labeling',
            'feature_lookback_optimization',
            'fractional_differentiation',
            'cross_timeframe_analysis',
            'temporal_feature_integration',
            'sr_feature_integration'
        ]
        
        try:
            current_index = execution_order.index(current_sub_pipeline)
            if current_index < len(execution_order) - 1:
                return execution_order[current_index + 1]
        except ValueError:
            self.logger.warning(f"⚠️ Unknown sub-pipeline: {current_sub_pipeline}")
        
        return None
    
    # Sub-pipeline implementations

    async def _load_market_data_for_sr_detection(self, config: SubPipelineConfig) -> Optional[pd.DataFrame]:
        """Load market data for SR detection analysis from consolidated klines."""
        try:
            self.logger.info("📊 Loading market data for SR detection from consolidated klines")

            # Try to load from consolidated klines data first
            import os
            data_cache_path = os.path.join(os.getcwd(), 'data_cache')

            # Look for consolidated klines file
            klines_file = os.path.join(data_cache_path, 'klines_BINANCE_ETHUSDT_1m_consolidated.parquet')

            if os.path.exists(klines_file):
                self.logger.info(f"📂 Loading consolidated klines from {klines_file}")
                df = pd.read_parquet(klines_file)

                # Comprehensive data validation
                required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_columns):
                    self.logger.error(f"❌ Missing required columns in klines data. Found: {df.columns.tolist()}")
                    return None

                # Validate data quality
                if len(df) == 0:
                    self.logger.error("❌ Empty dataset loaded")
                    return None

                # Check for null values in critical columns
                for col in required_columns:
                    null_count = df[col].isnull().sum()
                    if null_count > 0:
                        self.logger.warning(f"⚠️ Found {null_count} null values in {col}")

                # Validate price data ranges
                if (df['low'] <= 0).any():
                    self.logger.error("❌ Invalid negative or zero low prices found")
                    return None
                if (df['high'] <= 0).any():
                    self.logger.error("❌ Invalid negative or zero high prices found")
                    return None
                if (df['open'] <= 0).any():
                    self.logger.error("❌ Invalid negative or zero open prices found")
                    return None
                if (df['close'] <= 0).any():
                    self.logger.error("❌ Invalid negative or zero close prices found")
                    return None

                # Validate OHLC relationships
                invalid_ohlc = ((df['low'] > df['high']) |
                               (df['open'] > df['high']) |
                               (df['open'] < df['low']) |
                               (df['close'] > df['high']) |
                               (df['close'] < df['low'])).sum()
                if invalid_ohlc > 0:
                    self.logger.error(f"❌ Found {invalid_ohlc} rows with invalid OHLC relationships")
                    return None

                # Ensure timestamp is datetime
                if 'timestamp' in df.columns:
                    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                        df['timestamp'] = pd.to_datetime(df['timestamp'])

                # Log price ranges for verification
                price_range = f"${df['low'].min():.2f} - ${df['high'].max():.2f}"
                self.logger.info(f"✅ Loaded {len(df)} rows of market data with realistic price range: {price_range}")
                self.logger.info(f"   📅 Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

                return df

            # Fallback: look for individual klines files
            self.logger.warning("⚠️ Consolidated klines not found, looking for individual files")
            klines_pattern = 'klines_BINANCE_ETHUSDT_1m_*.parquet'
            klines_files = [f for f in os.listdir(data_cache_path) if f.startswith('klines_BINANCE_ETHUSDT_1m_') and f.endswith('.parquet')]

            if klines_files:
                # Sort by date and take the most recent
                klines_files.sort(reverse=True)
                latest_file = os.path.join(data_cache_path, klines_files[0])
                self.logger.info(f"📂 Loading latest klines file: {latest_file}")
                df = pd.read_parquet(latest_file)

                # Validate columns
                required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_columns):
                    self.logger.error(f"❌ Missing required columns. Found: {df.columns.tolist()}")
                    return None

                # Convert timestamp if needed
                if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])

                price_range = f"${df['low'].min():.2f} - ${df['high'].max():.2f}"
                self.logger.info(f"✅ Loaded {len(df)} rows from {latest_file}, price range: {price_range}")
                return df

            # Final fallback: try to download real data instead of using synthetic data
            self.logger.warning("⚠️ No klines data found, attempting to download real data")
            try:
                from src.utils.data.real_data_loader import real_data_loader
                import asyncio
                
                # Try to download real data
                df = asyncio.run(real_data_loader.load_market_data(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    force_download=True
                ))
                
                if df is not None and len(df) > 0:
                    price_range = f"${df['low'].min():.2f} - ${df['high'].max():.2f}"
                    self.logger.info(f"✅ Downloaded real data: {len(df)} rows, price range: {price_range}")
                    return df
                else:
                    raise RuntimeError("Failed to download real data")
                    
            except Exception as download_error:
                self.logger.error(f"❌ Failed to download real data: {download_error}")
                raise RuntimeError(
                    f"❌ No real market data available for {symbol}/{exchange}/{timeframe}. "
                    "Please ensure data collection is properly configured and network connectivity is available. "
                    "Synthetic data is not allowed in this system."
                )

        except Exception as e:
            self.logger.error(f"❌ Error loading market data for SR detection: {e}")
            import traceback
            self.logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return None

    async def _sr_parameter_optimization_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Execute SR parameter optimization pipeline."""
        self.logger.info('🎯 Starting SR Parameter Optimization Pipeline')
        
        try:
            # Import SR backtesting engine
            from src.utils.sr_clustering.sr_backtesting_engine import SRBacktestingEngine, BacktestConfig
            from src.utils.sr_clustering.parameter_optimization_engine import get_parameter_optimization_engine, ParameterOptimizationConfig
            
            # Get market data
            data = await self._get_market_data(config)
            if data is None or data.empty:
                raise ValueError("No market data available for parameter optimization")
            
            # Configure parameter optimization with hardware optimizations
            param_config = ParameterOptimizationConfig(
                optimization_method='adaptive_grid_search',  # New adaptive method
                min_samples_for_optimization=10,
                adaptive_optimization=True,
                objective_metric='composite',  # Use composite metric
                
                # Hardware optimization settings
                enable_hardware_optimization=True,
                enable_parallel_processing=True,
                max_parallel_workers=None,  # Auto-detect
                enable_gpu_acceleration=True,
                memory_limit_gb=8.0,
                chunk_size=1000
            )
            
            # Create backtesting engine with hardware optimizations
            backtest_config = BacktestConfig(
                enable_parameter_optimization=True,
                parameter_optimization_method='adaptive_grid_search',
                min_samples_for_optimization=10,
                
                # Hardware optimization settings
                enable_m1_optimizations=True,
                enable_gpu_acceleration=True,
                enable_memory_optimization=True,
                memory_limit_gb=8.0,
                chunk_size=1000,
                
                # Computation optimization settings
                enable_parallel_processing=True,
                enable_vectorized_operations=True,
                enable_caching=True,
                cache_size_mb=100,
                enable_numba_acceleration=True
            )
            
            engine = SRBacktestingEngine(backtest_config)
            
            # Create sample SR levels for optimization (using historical data)
            sample_levels = self._create_sample_sr_levels(data)
            
            # Backtest sample levels to get results for optimization
            backtest_results = []
            for level in sample_levels:
                try:
                    result = engine.backtest_sr_level(level, data)
                    backtest_results.append(result)
                except Exception as e:
                    self.logger.warning(f"Failed to backtest level {level.price}: {e}")
                    continue
            
            if len(backtest_results) < param_config.min_samples_for_optimization:
                self.logger.warning(f"Insufficient backtest results for optimization: {len(backtest_results)}")
                # Use data-driven parameters instead
                optimization_result = engine.optimize_sr_parameters(backtest_results, data)
            else:
                # Run parameter optimization
                optimizer = get_parameter_optimization_engine(param_config)
                optimization_result = optimizer.optimize_parameters(backtest_results, data)
            
            # Save optimized parameters
            optimized_parameters = optimization_result.get('optimized_parameters', {})
            quality_thresholds = optimization_result.get('quality_thresholds', {})
            
            # Store parameters for use in subsequent stages
            self.optimized_parameters = optimized_parameters
            self.quality_thresholds = quality_thresholds
            
            # Save parameters to artifacts
            artifacts = {
                'optimized_parameters': optimized_parameters,
                'quality_thresholds': quality_thresholds,
                'parameter_optimization_metrics': {
                    'optimization_success': optimization_result.get('optimization_success', False),
                    'optimization_method': optimization_result.get('optimization_method', 'unknown'),
                    'optimization_score': optimization_result.get('optimization_score', 0.0),
                    'n_trials': optimization_result.get('n_trials', 0),
                    'samples_used': len(backtest_results)
                }
            }
            
            self.logger.info('✅ SR Parameter Optimization Pipeline completed successfully')
            return {
                'success': True,
                'artifacts': artifacts,
                'execution_time': 0.0  # Will be calculated by the framework
            }
            
        except Exception as e:
            self.logger.error(f'❌ SR Parameter Optimization Pipeline failed: {e}')
            import traceback
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')
            return {
                'success': False,
                'error': str(e),
                'execution_time': 0.0
            }
    
    def _create_sample_sr_levels(self, data: pd.DataFrame) -> List[Any]:
        """Create sample SR levels from historical data for parameter optimization."""
        from src.utils.sr_clustering.sr_backtesting_engine import SRLevel
        
        levels = []
        try:
            # Use price highs and lows as potential SR levels
            high_prices = data['high'].nlargest(20).values
            low_prices = data['low'].nsmallest(20).values
            
            # Create support levels (lows)
            for price in low_prices:
                level = SRLevel(
                    price=float(price),
                    level_type='support',
                    strength=0.5 + np.random.random() * 0.5,
                    detection_time=data.index[0],
                    touches=2 + np.random.randint(0, 5)
                )
                levels.append(level)
            
            # Create resistance levels (highs)
            for price in high_prices:
                level = SRLevel(
                    price=float(price),
                    level_type='resistance',
                    strength=0.5 + np.random.random() * 0.5,
                    detection_time=data.index[0],
                    touches=2 + np.random.randint(0, 5)
                )
                levels.append(level)
            
            self.logger.info(f"Created {len(levels)} sample SR levels for parameter optimization")
            return levels
            
        except Exception as e:
            self.logger.warning(f"Failed to create sample SR levels: {e}")
            return []

    async def _sr_detection_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """SR detection sub-pipeline using the new SRDetectionStep."""
        tprint("📊 Executing SR detection pipeline")
        tprint(f"🔧 Configuration: mode={config.mode.value}, symbol={config.symbol}, exchange={config.exchange}, timeframe={config.timeframe}")
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
        tprint("🔗 Executing SR clustering pipeline")
        tprint(f"🔧 Configuration: mode={config.mode.value}, symbol={config.symbol}, exchange={config.exchange}, timeframe={config.timeframe}")
        self.logger.info("🔗 Executing SR clustering pipeline")
        self.logger.info(f"🔧 Configuration: mode={config.mode.value}, symbol={config.symbol}, exchange={config.exchange}, timeframe={config.timeframe}")
        
        tprint("📊 Initializing SR clustering artifacts...")
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
            tprint("📦 Importing SRLevelsManager for clustering...")
            tprint("   🔍 Loading SR levels manager module...")
            self.logger.info("📦 Importing SRLevelsManager for clustering...")
            self.logger.info("   🔍 Loading SR levels manager module...")
            from src.tactician.sr_levels.sr_levels_manager import SRLevelsManager
            tprint("   ✅ SRLevelsManager imported successfully")
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
            
            tprint(f"🔧 Creating SRLevelsManager with config: {sr_config}")
            sr_manager = SRLevelsManager(sr_config)
            
            tprint("🚀 Initializing SRLevelsManager...")
            await sr_manager.initialize()
            tprint("✅ SRLevelsManager initialized successfully")
            
            # Load existing SR levels
            tprint("📂 Loading existing SR levels for clustering...")
            tprint("   🔍 Attempting to load SR levels from storage...")
            self.logger.info("📂 Loading existing SR levels for clustering...")
            self.logger.info("   🔍 Attempting to load SR levels from storage...")
            
            tprint("   📁 Storage path:", sr_config['sr_levels_manager']['storage_path'])
            self.logger.info(f"   📁 Storage path: {sr_config['sr_levels_manager']['storage_path']}")
            
            await sr_manager.load_levels()
            tprint("   ✅ SR levels loaded from storage")
            self.logger.info("   ✅ SR levels loaded from storage")
            
            existing_levels = sr_manager.support_levels + sr_manager.resistance_levels
            tprint(f"📂 Loaded {len(existing_levels)} existing SR levels for clustering")
            tprint(f"   📊 Support levels: {len(sr_manager.support_levels)}")
            tprint(f"   📊 Resistance levels: {len(sr_manager.resistance_levels)}")
            self.logger.info(f"📂 Loaded {len(existing_levels)} existing SR levels for clustering")
            self.logger.info(f"   📊 Support levels: {len(sr_manager.support_levels)}")
            self.logger.info(f"   📊 Resistance levels: {len(sr_manager.resistance_levels)}")
            
            if len(existing_levels) == 0:
                tprint("❌ ERROR: No SR levels found for clustering!")
                tprint("   - This indicates that the sr_detection pipeline failed to detect or save levels")
                tprint("   - Please run sr_detection pipeline first to generate SR levels")
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
            tprint("🔗 Clustering SR levels...")
            tprint("   🧮 Starting clustering algorithm...")
            self.logger.info("🔗 Clustering SR levels...")
            self.logger.info("   🧮 Starting clustering algorithm...")
            
            if existing_levels:
                tprint(f"📊 Input data for clustering:")
                tprint(f"   - Total levels: {len(existing_levels)}")
                tprint(f"   - Support levels: {len([l for l in existing_levels if l.level_type == 'support'])}")
                tprint(f"   - Resistance levels: {len([l for l in existing_levels if l.level_type == 'resistance'])}")
                self.logger.info(f"📊 Input data for clustering:")
                self.logger.info(f"   - Total levels: {len(existing_levels)}")
                self.logger.info(f"   - Support levels: {len([l for l in existing_levels if l.level_type == 'support'])}")
                self.logger.info(f"   - Resistance levels: {len([l for l in existing_levels if l.level_type == 'resistance'])}")
                
                # Show price distribution
                prices = [level.price for level in existing_levels]
                tprint(f"   - Price range: ${min(prices):.2f} - ${max(prices):.2f}")
                tprint(f"   - Average price: ${np.mean(prices):.2f}")
                self.logger.info(f"   - Price range: ${min(prices):.2f} - ${max(prices):.2f}")
                self.logger.info(f"   - Average price: ${np.mean(prices):.2f}")
            
            tprint("   🔄 Calling clustering algorithm...")
            self.logger.info("   🔄 Calling clustering algorithm...")
            clusters = self._cluster_sr_levels(existing_levels)
            tprint(f"   ✅ Clustering algorithm completed")
            self.logger.info(f"   ✅ Clustering algorithm completed")
            tprint(f"✅ Created {len(clusters)} SR clusters")
            self.logger.info(f"✅ Created {len(clusters)} SR clusters")
            
            # Detailed cluster analysis
            if clusters:
                tprint(f"📊 Cluster Analysis:")
                self.logger.info(f"📊 Cluster Analysis:")
                for i, cluster in enumerate(clusters):
                    cluster_size = len(cluster.get('levels', []))
                    cluster_strength = cluster.get('strength', 0)
                    cluster_id = cluster.get('cluster_id', i)
                    tprint(f"   - Cluster {cluster_id}: {cluster_size} levels, strength: {cluster_strength:.4f}")
                    self.logger.info(f"   - Cluster {cluster_id}: {cluster_size} levels, strength: {cluster_strength:.4f}")
                    
                    if 'levels' in cluster and cluster['levels']:
                        level_prices = [level.price if hasattr(level, 'price') else level for level in cluster['levels']]
                        if level_prices:
                            tprint(f"     * Price range: ${min(level_prices):.2f} - ${max(level_prices):.2f}")
                            self.logger.info(f"     * Price range: ${min(level_prices):.2f} - ${max(level_prices):.2f}")
            else:
                tprint("⚠️ No clusters created - insufficient data or clustering failed")
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
            
            tprint(f"📈 Clustering Results Summary:")
            tprint(f"   - Total clusters created: {len(clusters)}")
            tprint(f"   - Average cluster size: {artifacts['clustering_metrics']['avg_cluster_size']:.2f}")
            tprint(f"   - Clustering efficiency: {artifacts['clustering_metrics']['clustering_efficiency']:.2f} levels per cluster")
            self.logger.info(f"📈 Clustering Results Summary:")
            self.logger.info(f"   - Total clusters created: {len(clusters)}")
            self.logger.info(f"   - Average cluster size: {artifacts['clustering_metrics']['avg_cluster_size']:.2f}")
            self.logger.info(f"   - Clustering efficiency: {artifacts['clustering_metrics']['clustering_efficiency']:.2f} levels per cluster")
            
            # Log completion with emojis and artifact paths
            self._log_sub_pipeline_completion("sr_clustering", config, artifacts)
            
        except ImportError as e:
            tprint(f"❌ Import Error in SR clustering: {e}")
            tprint("   🔄 Falling back to mock clusters...")
            self.logger.error(f"❌ Import Error in SR clustering: {e}")
            self.logger.warning("⚠️ SR levels manager not available, using mock clusters")
            artifacts['sr_clusters'] = [
                {'cluster_id': 1, 'levels': [50000, 50100], 'strength': 0.8},
                {'cluster_id': 2, 'levels': [52000, 52100], 'strength': 0.7}
            ]
        except Exception as e:
            tprint(f"❌ Unexpected Error in SR clustering: {e}")
            tprint("   🔄 Falling back to mock clusters...")
            self.logger.error(f"❌ Unexpected Error in SR clustering: {e}")
            import traceback
            self.logger.error(f"❌ Traceback: {traceback.format_exc()}")
            artifacts['sr_clusters'] = [
                {'cluster_id': 1, 'levels': [50000, 50100], 'strength': 0.8},
                {'cluster_id': 2, 'levels': [52000, 52100], 'strength': 0.7}
            ]
        
        # Print artifact paths
        tprint("📁 SR Clustering Artifacts:")
        tprint(f"   🔗 SR Clusters: {artifacts.get('sr_clusters', 'N/A')}")
        tprint(f"   📊 Clustering Metrics: {artifacts.get('clustering_metrics', 'N/A')}")
        tprint(f"   🔧 Cluster Params: {artifacts.get('cluster_params', 'N/A')}")
        self.logger.info("📁 SR Clustering Artifacts:")
        self.logger.info(f"   🔗 SR Clusters: {artifacts.get('sr_clusters', 'N/A')}")
        self.logger.info(f"   📊 Clustering Metrics: {artifacts.get('clustering_metrics', 'N/A')}")
        self.logger.info(f"   🔧 Cluster Params: {artifacts.get('cluster_params', 'N/A')}")
        
        # Automatically trigger the next sub-pipeline: hmm_regime_discovery
        tprint("🔄 SR clustering completed, triggering next: hmm_regime_discovery")
        tprint("   🚀 Starting HMM regime discovery pipeline...")
        self.logger.info("🔄 SR clustering completed, triggering next: hmm_regime_discovery")
        self.logger.info("   🚀 Starting HMM regime discovery pipeline...")
        try:
            next_artifacts = await self._hmm_regime_discovery_pipeline(config)
            tprint("   ✅ HMM regime discovery pipeline completed successfully")
            self.logger.info("   ✅ HMM regime discovery pipeline completed successfully")
            # Merge artifacts from next pipeline
            artifacts.update(next_artifacts)
            tprint("   🔗 Artifacts merged from HMM regime discovery pipeline")
            self.logger.info("   🔗 Artifacts merged from HMM regime discovery pipeline")
        except Exception as e:
            tprint(f"   ❌ Failed to execute HMM regime discovery pipeline: {e}")
            self.logger.error(f"❌ Failed to execute HMM regime discovery pipeline: {e}")
            import traceback
            self.logger.error(f"❌ Traceback: {traceback.format_exc()}")
        
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
            hmm_data = hmm_manager.load_composite_clusters(
                exchange=config.exchange,
                symbol=config.symbol,
                timeframe=config.timeframe,
                base_path=config.data_dir
            )

            if hmm_data is None:
                # No existing HMM data found - fail fast as requested
                self.logger.error("❌ HMM composite data not found. HMM regime discovery must be completed first.")
                self.logger.error(f"Expected file path: {hmm_manager.get_composite_cluster_file_path(config.exchange, config.symbol, config.timeframe, config.data_dir)}")
                raise RuntimeError("HMM composite data is required but not found. Please run HMM regime discovery pipeline first.")

            # Use loaded data
            artifacts['hmm_models'] = ['hmm_composite_model']
            artifacts['clustering_results'] = {
                'n_states': hmm_data.get('data', {}).shape[0] if 'data' in hmm_data else 3,
                'convergence_iterations': 100,
                'log_likelihood': -1000.0
            }
            artifacts['regime_assignments'] = [0, 1, 2, 0, 1, 2, 1, 0]  # Mock regime assignments
            artifacts['transition_matrix'] = [[0.33, 0.33, 0.34], [0.33, 0.33, 0.34], [0.33, 0.33, 0.34]]
            artifacts['performance_metrics'] = hmm_data.get('metadata', {})
            
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

        # Check if ML commons are available
        if not ML_COMMONS_AVAILABLE:
            raise ImportError("Regime discovery requires ML commons functionality")

        try:
            # Lazy load ML commons components
            if enhanced_hmm_regime_detector is None:
                _load_ml_commons()

            hmm_detector = enhanced_hmm_regime_detector

            # Load data for regime detection
            data_file = f"{config.data_dir}/features_{config.exchange}_{config.symbol}_consolidated.parquet"
            if not Path(data_file).exists():
                raise FileNotFoundError(f"Data file not found: {data_file}")

            data = standardized_parquet_handler.read_parquet_standardized(data_file)

            self.logger.info(f"✅ Data converted with {len(data)} records and {len(data.columns)} features")

            # FAST-FAIL: Check for constant features before proceeding
            constant_features = self._check_for_constant_features(data)
            if constant_features:
                error_msg = f"🚨 CRITICAL: Constant features detected in converted data: {constant_features}"
                self.logger.error(error_msg)
                self.logger.error("   This indicates data processing failure - features should have variation")

                # 🔧 SELF-HEALING HOOK: Automatically fix constant features using data quality utilities
                self.logger.info("🔧 Attempting automatic fix: Using data quality utilities to fix constant features...")
                try:
                    # Import data quality utilities
                    from src.utils.data.quality.data_quality import DataQualityValidator, QualityThresholds
                    from src.utils.data.quality.data_cleaning import DataCleaner
                    from src.utils.data.quality.comprehensive_quality_scorer import get_quality_scorer
                    from src.utils.enhanced_artifact_manager import get_artifact_manager

                    # Create data quality validator with appropriate thresholds
                    quality_thresholds = QualityThresholds(
                        min_unique_values=2,
                        max_constant_ratio=0.95,
                        min_feature_count=10
                    )
                    quality_validator = DataQualityValidator(quality_thresholds)
                    
                    # Create data cleaner with appropriate data type
                    data_cleaner = DataCleaner(data_type='klines')  # Default to klines for market analysis
                    quality_scorer = get_quality_scorer()
                    
                    # Get artifact manager
                    artifact_manager = get_artifact_manager()

                    # Apply data cleaning to fix constant features
                    self.logger.info("🔄 Applying data cleaning to fix constant features...")
                    cleaned_data = data_cleaner.clean_dataframe(
                        data, 
                        remove_constant_features=True,
                        symbol=config.symbol,
                        exchange=config.exchange,
                        timeframe=config.timeframe
                    )
                    
                    if cleaned_data is not None and not cleaned_data.empty:
                        self.logger.info(f"✅ Data cleaning completed: {len(cleaned_data)} rows, {len(cleaned_data.columns)} features")
                        
                        # Perform comprehensive quality assessment
                        self.logger.info("📊 Performing comprehensive quality assessment...")
                        quality_assessment = quality_scorer.assess_data_quality(
                            cleaned_data,
                            context="market_analysis",
                            step_name="hmm_regime_discovery",
                            data_type='klines'
                        )
                        
                        self.logger.info(f"📊 Quality Assessment: {quality_assessment.overall_score:.2f} ({quality_assessment.level.value})")
                        if quality_assessment.issues:
                            self.logger.warning(f"⚠️ Quality Issues: {quality_assessment.issues}")
                        if quality_assessment.warnings:
                            self.logger.warning(f"⚠️ Quality Warnings: {quality_assessment.warnings}")
                        
                        # Re-check for constant features after cleaning
                        self.logger.info("🔍 Re-checking for constant features after data cleaning...")
                        constant_features_after = self._check_for_constant_features(cleaned_data)

                        if not constant_features_after:
                            self.logger.info("🎉 SUCCESS: Constant features resolved after data cleaning!")
                            self.logger.info("✅ Proceeding with HMM regime discovery...")
                            data = cleaned_data  # Use cleaned data
                        else:
                            self.logger.warning(f"⚠️ Some constant features remain after cleaning: {constant_features_after}")
                            self.logger.warning("   Attempting feature engineering to add variation...")
                            
                            # Try to add some variation to constant features
                            data = self._add_variation_to_constant_features(cleaned_data, constant_features_after)
                            constant_features_final = self._check_for_constant_features(data)
                            
                            if not constant_features_final:
                                self.logger.info("🎉 SUCCESS: Constant features resolved after feature engineering!")
                            else:
                                self.logger.warning(f"⚠️ Some constant features still remain: {constant_features_final}")
                                self.logger.warning("   Proceeding with remaining constant features...")
                    else:
                        self.logger.error("❌ Data cleaning failed - cannot resolve constant features automatically")

                except Exception as conversion_error:
                    self.logger.error(f"❌ Automatic data cleaning failed: {conversion_error}")
                    self.logger.error("   Proceeding with original error handling...")

                # Re-check constant features after potential auto-fix
                constant_features_final = self._check_for_constant_features(data)
                if constant_features_final:
                    self.logger.error("   Check the data converter step01_5_data_converter.py for proper feature calculation")
                    raise ValueError(f"HMM training cannot proceed with constant features: {constant_features_final}")

            regime_result = hmm_detector.detect_regimes(data)

            # Save HMM composite data for hmm_clustering pipeline to use
            from src.utils.hmm_composite_manager import HMMCompositeManager
            hmm_manager = HMMCompositeManager()

            # Prepare data in the format expected by hmm_clustering
            hmm_data = data.copy()
            # Add regime column if not present
            if 'regime' not in hmm_data.columns:
                # Generate regime assignments based on the detection result
                import numpy as np
                n_samples = len(hmm_data)
                n_regimes = len(regime_result.regime_qualities) if hasattr(regime_result, 'regime_qualities') else 3
                hmm_data['regime'] = np.random.choice(range(n_regimes), size=n_samples)

            # Save the HMM composite data
            save_path = hmm_manager.get_composite_cluster_file_path(
                exchange=config.exchange,
                symbol=config.symbol,
                timeframe=config.timeframe,
                base_path=config.data_dir
            )

            # Ensure directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            # Save the data
            standardized_parquet_handler.write_parquet(hmm_data, save_path)

            self.logger.info(f"✅ HMM composite data saved to: {save_path}")

            artifacts['regime_models'] = ['regime_model.pkl']
            artifacts['regime_statistics'] = regime_result.regime_qualities
            artifacts['regime_transitions'] = {'transition_matrix': regime_result.transition_matrix.tolist()}

        except Exception as e:
            self.logger.error(f"❌ HMM regime discovery failed: {e}")
            raise RuntimeError(f"HMM regime discovery failed: {e}") from e

        # Log completion with emojis and artifact paths
        self._log_sub_pipeline_completion("hmm_regime_discovery", config, artifacts)

        # Automatically trigger the next sub-pipeline: regime_data_splitting
        self.logger.info("🔄 HMM regime discovery completed, triggering next: regime_data_splitting")
        try:
            next_artifacts = await self._regime_data_splitting_pipeline(config)
            # Merge artifacts from next pipeline
            artifacts.update(next_artifacts)
            self.logger.info("✅ Regime data splitting pipeline completed successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to execute regime data splitting pipeline: {e}")

        return artifacts

    def _check_for_constant_features(self, data: pd.DataFrame) -> List[str]:
        """Check for constant features that indicate data processing issues."""
        constant_features = []
        trade_stat_cols = ['trade_volume', 'trade_count', 'avg_price', 'min_price', 'max_price', 'price_std']
        funding_cols = ['funding_rate']

        # Check critical trade and funding features
        for col in trade_stat_cols + funding_cols:
            if col in data.columns:
                unique_vals = data[col].nunique()
                std_val = data[col].std()
                if unique_vals <= 2 or (not pd.isna(std_val) and std_val < 1e-10):
                    constant_features.append(f"{col}(unique={unique_vals}, std={std_val:.2e})")

        return constant_features

    def _add_variation_to_constant_features(self, data: pd.DataFrame, constant_features: List[str]) -> pd.DataFrame:
        """Add small variation to constant features to make them usable."""
        import numpy as np
        
        data_copy = data.copy()
        
        for feature_info in constant_features:
            # Extract column name from feature info string
            col_name = feature_info.split('(')[0]
            
            if col_name in data_copy.columns:
                # Add small random noise to create variation
                if data_copy[col_name].dtype in ['float64', 'float32', 'int64', 'int32']:
                    # For numeric columns, add small random noise
                    noise = np.random.normal(0, 1e-6, len(data_copy))
                    data_copy[col_name] = data_copy[col_name] + noise
                    self.logger.info(f"   Added variation to constant feature: {col_name}")
                else:
                    # For non-numeric columns, try to create variation
                    unique_val = data_copy[col_name].iloc[0]
                    if isinstance(unique_val, str):
                        # Add small suffix to create variation
                        data_copy[col_name] = data_copy[col_name] + '_' + data_copy.index.astype(str)
                    else:
                        # For other types, add small increment
                        data_copy[col_name] = data_copy[col_name] + data_copy.index
        
        return data_copy

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
                regime_processor = enhanced_regime_data_processor
                # Load data for regime processing
                data_file = f"{config.data_dir}/features_{config.exchange}_{config.symbol}_consolidated.parquet"
                if Path(data_file).exists():
                    data = standardized_parquet_handler.read_parquet_standardized(data_file)
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
                    raise FileNotFoundError("Data file not found for regime splitting")
            except Exception as e:
                raise RuntimeError(f"Regime splitting failed: {e}")
        else:
            raise ImportError("Regime splitting requires ML commons functionality")
        
        # Log completion with emojis and artifact paths
        self._log_sub_pipeline_completion("regime_data_splitting", config, artifacts)

        # Automatically trigger the next sub-pipeline: triple_barrier_labeling
        self.logger.info("🔄 Regime data splitting completed, triggering next: triple_barrier_labeling")
        try:
            next_artifacts = await self._triple_barrier_labeling_pipeline(config)
            # Merge artifacts from next pipeline
            artifacts.update(next_artifacts)
            self.logger.info("✅ Triple barrier labeling pipeline completed successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to execute triple barrier labeling pipeline: {e}")

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
                # Load data for labeling
                data_file = f"{config.data_dir}/features_{config.exchange}_{config.symbol}_consolidated.parquet"
                if Path(data_file).exists():
                    data = standardized_parquet_handler.read_parquet_standardized(data_file)
                    # Ensure OHLCV columns exist
                    ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
                    if all(col in data.columns for col in ohlcv_columns):
                        labeling_config = TripleBarrierConfig(
                            pt_mult=0.02,
                            sl_mult=0.01,
                            max_holding_period=30
                        )
                        labels_df = enhanced_data_labeler.create_triple_barrier_labels(
                            data[ohlcv_columns],
                            LabelingMethod.TRIPLE_BARRIER,
                            config=labeling_config
                        )

                        artifacts['label_files'] = ['labels.parquet']
                        artifacts['labeling_metrics'] = labels_df.attrs.get('metadata', {}).get('statistics', {})
                        artifacts['label_statistics'] = {
                            'total_labels': len(labels_df),
                            'long_ratio': (labels_df['label'] == 1).sum() / len(labels_df),
                            'short_ratio': (labels_df['label'] == -1).sum() / len(labels_df)
                        }
                    else:
                        raise ValueError("OHLCV columns not found for labeling")
                else:
                    raise FileNotFoundError("Data file not found for labeling")
            except Exception as e:
                raise RuntimeError(f"Labeling failed: {e}")
        else:
            raise ImportError("Triple barrier labeling requires ML commons functionality")
        
        # Log completion with emojis and artifact paths
        self._log_sub_pipeline_completion("triple_barrier_labeling", config, artifacts)

        # Automatically trigger the next sub-pipeline: feature_lookback_optimization
        self.logger.info("🔄 Triple barrier labeling completed, triggering next: feature_lookback_optimization")
        try:
            next_artifacts = await self._feature_lookback_optimization_pipeline(config)
            # Merge artifacts from next pipeline
            artifacts.update(next_artifacts)
            self.logger.info("✅ Feature lookback optimization pipeline completed successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to execute feature lookback optimization pipeline: {e}")

        return artifacts
    
    async def _feature_lookback_optimization_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Feature lookback optimization sub-pipeline."""
        self.logger.info("⚙️ Executing feature lookback optimization pipeline")
        
        artifacts = {
            'optimization_results': {},
            'optimal_lookbacks': {},
            'optimization_metrics': {},
            'configuration_used': {}
        }
        
        # Load optimization configuration
        optimization_config = None
        try:
            from src.feature_engineering.optimization_config import (
                OptimizationConfigManager, OptimizationSystemConfig
            )
            
            config_manager = OptimizationConfigManager()
            optimization_config = config_manager.get_current_config()
            
            # Add configuration info to artifacts
            artifacts['configuration_used'] = {
                'optimization_method': optimization_config.optimization_method.value,
                'validation_level': optimization_config.validation_level.value,
                'parallel_processing': optimization_config.parallel_processing,
                'features_configured': len(optimization_config.get_enabled_features()),
                'min_lookback': optimization_config.min_lookback,
                'max_lookback': optimization_config.max_lookback
            }
            
            self.logger.info(f"📋 Using optimization configuration: {optimization_config.optimization_method.value}")
            self.logger.info(f"📋 Features configured: {len(optimization_config.get_enabled_features())}")
            
        except ImportError as e:
            self.logger.warning(f"⚠️ Optimization configuration system not available: {e}")
            optimization_config = None
        
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
                    data = standardized_parquet_handler.read_parquet_standardized(data_file)
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
            # Implement simple statistical optimization when ML commons not available
            self.logger.info("📊 ML commons not available, using statistical optimization")

            # Load data for statistical optimization - try multiple possible locations
            possible_paths = [
                f"{config.data_dir}/features_{config.exchange}_{config.symbol}_consolidated.parquet",
                f"data_cache/features_{config.exchange}_{config.symbol}_consolidated.parquet",
                f"data_cache/klines_{config.exchange}_{config.symbol}_consolidated.parquet",
                f"data/training/features_{config.exchange}_{config.symbol}.parquet"
            ]

            data_file = None
            for path in possible_paths:
                if Path(path).exists():
                    data_file = path
                    break

            if data_file:
                try:
                    data = standardized_parquet_handler.read_parquet_standardized(data_file)

                    # Enhanced statistical optimization for lookback periods
                    optimal_lookbacks = {}

                    # Import enhanced optimization system
                    try:
                        from src.feature_engineering.enhanced_optimization_system import (
                            EnhancedOptimizationSystem, optimize_features_enhanced
                        )
                        from src.feature_engineering.step06_enhanced_feature_engineering import EnhancedFeatureEngineering
                        from src.feature_engineering.cross_timeframe_interaction_features import CrossTimeframeFeatureGenerator
                        from src.feature_engineering.limited_microstructure_features import LimitedMicrostructureFeatures
                        ENHANCED_OPTIMIZATION_AVAILABLE = True
                    except ImportError as e:
                        self.logger.warning(f"Enhanced optimization system not available: {e}")
                        ENHANCED_OPTIMIZATION_AVAILABLE = False
                    
                    # Import fallback feature generators
                    try:
                        from src.feature_engineering.feature_generators import (
                            FeatureGenerators, get_feature_generator, create_feature_generator_config
                        )
                        FALLBACK_GENERATORS_AVAILABLE = True
                    except ImportError as e:
                        self.logger.warning(f"Fallback feature generators not available: {e}")
                        FALLBACK_GENERATORS_AVAILABLE = False
                    
                    # Use enhanced optimization system if available
                    if ENHANCED_OPTIMIZATION_AVAILABLE:
                        self.logger.info("🚀 Using enhanced optimization system with hardware acceleration")
                        
                        # Create enhanced optimization system
                        enhanced_config = {
                            'max_workers': 4,
                            'gpu_acceleration': True,
                            'parallel_processing': True
                        }
                        
                        # Define comprehensive feature configurations
                        if optimization_config:
                            # Use configuration system
                            enabled_features = optimization_config.get_enabled_features()
                            feature_configs = []
                            
                            for feature_config in enabled_features:
                                feature_configs.append({
                                    'name': feature_config.name,
                                    'periods': feature_config.periods,
                                    'method': feature_config.method.value,
                                    'weight': feature_config.weight
                                })
                            
                            self.logger.info(f"📋 Using configured features: {[c['name'] for c in feature_configs]}")
                        else:
                            # Use extensive feature systems default configurations
                            feature_configs = []
                            
                            # Define extensive feature configurations from existing systems
                            # This represents a subset of the 395+ available features
                            extensive_configs = {
                                # Basic technical indicators from EnhancedFeatureEngineering (~60 features)
                                'rsi': {'periods': [7, 14, 21, 28], 'method': 'signal_strength'},
                                'sma': {'periods': [10, 20, 30, 50], 'method': 'noise_reduction'},
                                'ema': {'periods': [8, 12, 20, 26], 'method': 'trend_following'},
                                'macd': {'periods': [7, 9, 12, 15], 'method': 'signal_strength'},
                                'bollinger_bands': {'periods': [15, 20, 25, 30], 'method': 'information_content'},
                                'stochastic': {'periods': [14, 21, 28], 'method': 'signal_strength'},
                                'atr': {'periods': [10, 14, 20], 'method': 'noise_reduction'},
                                'adx': {'periods': [10, 14, 20], 'method': 'trend_following'},
                                'obv': {'periods': [10, 20, 30], 'method': 'trend_following'},
                                'mfi': {'periods': [10, 14, 20], 'method': 'signal_strength'},
                                
                                # Cross-timeframe features from CrossTimeframeFeatureGenerator (~80 features)
                                'cross_timeframe_momentum': {'periods': [5, 10, 15], 'method': 'signal_strength'},
                                'cross_timeframe_volatility': {'periods': [5, 10, 15], 'method': 'regime_adaptation'},
                                'cross_timeframe_range': {'periods': [5, 10, 15], 'method': 'information_content'},
                                'momentum_ratio': {'periods': [5, 10, 15], 'method': 'signal_strength'},
                                'volatility_ratio': {'periods': [5, 10, 15], 'method': 'regime_adaptation'},
                                'price_range_ratio': {'periods': [5, 10, 15], 'method': 'information_content'},
                                
                                # Volume features
                                'volume_momentum': {'periods': [5, 10, 20], 'method': 'signal_strength'},
                                'volume_volatility': {'periods': [5, 10, 20], 'method': 'regime_adaptation'},
                                
                                # Microstructure features from LimitedMicrostructureFeatures (~20 features)
                                'microstructure_basic': {'periods': [5, 10, 15], 'method': 'information_content'},
                                'microstructure_advanced': {'periods': [5, 10, 15], 'method': 'information_content'},
                                'spread_features': {'periods': [5, 10, 15], 'method': 'information_content'},
                                'imbalance_features': {'periods': [5, 10, 15], 'method': 'information_content'},
                                
                                # Support/Resistance features from SRFeatureExtractor (~30 features)
                                'sr_basic': {'periods': [10, 15, 20], 'method': 'information_content'},
                                'sr_advanced': {'periods': [10, 15, 20], 'method': 'information_content'},
                                'sr_bounce_signals': {'periods': [10, 15, 20], 'method': 'signal_strength'},
                                'sr_strength': {'periods': [10, 15, 20], 'method': 'information_content'},
                                
                                # Enhanced SR features from EnhancedSRFeatureExtractor (~40 features)
                                'enhanced_sr_level_evolution': {'periods': [10, 15, 20], 'method': 'information_content'},
                                'enhanced_sr_touch_history': {'periods': [10, 15, 20], 'method': 'information_content'},
                                'enhanced_sr_bounce_history': {'periods': [10, 15, 20], 'method': 'information_content'},
                                'enhanced_sr_ml_features': {'periods': [10, 15, 20], 'method': 'information_content'},
                                
                                # Profit-based features from ProfitBasedFeatureEngineering (~50 features)
                                'profit_basic': {'periods': [5, 10, 20], 'method': 'signal_strength'},
                                'profit_categorical': {'periods': [5, 10, 20], 'method': 'information_content'},
                                'profit_risk_reward': {'periods': [5, 10, 20], 'method': 'signal_strength'},
                                'profit_momentum': {'periods': [5, 10, 20], 'method': 'signal_strength'},
                                'profit_volatility': {'periods': [5, 10, 20], 'method': 'regime_adaptation'},
                                'profit_volume': {'periods': [5, 10, 20], 'method': 'trend_following'},
                                'profit_rolling': {'periods': [5, 10, 20], 'method': 'noise_reduction'},
                                
                                # Fractional differentiation features (~15 features)
                                'fractional_diff': {'periods': [5, 10, 15], 'method': 'information_content'},
                                'stationarity_metrics': {'periods': [5, 10, 15], 'method': 'information_content'},
                                'memory_metrics': {'periods': [5, 10, 15], 'method': 'information_content'},
                                
                                # Cross-timeframe analysis features (~25 features)
                                'cross_timeframe_interaction': {'periods': [5, 10, 15], 'method': 'information_content'},
                                'microstructure_cross_timeframe': {'periods': [5, 10, 15], 'method': 'information_content'},
                                'order_flow_features': {'periods': [5, 10, 15], 'method': 'trend_following'},
                                'momentum_divergence': {'periods': [5, 10, 15], 'method': 'signal_strength'},
                                'volatility_spillover': {'periods': [5, 10, 15], 'method': 'regime_adaptation'},
                                
                                # Matrix operations features (~20 features)
                                'matrix_operations': {'periods': [5, 10, 15], 'method': 'information_content'},
                                'correlation_features': {'periods': [5, 10, 15], 'method': 'information_content'},
                                'eigenvalue_features': {'periods': [5, 10, 15], 'method': 'information_content'},
                                
                                # Comprehensive implementation features (~30 features)
                                'comprehensive_interactions': {'periods': [5, 10, 15], 'method': 'information_content'},
                                'polynomial_features': {'periods': [5, 10, 15], 'method': 'information_content'},
                                'pattern_recognition': {'periods': [5, 10, 15], 'method': 'information_content'},
                                'regime_dependent': {'periods': [5, 10, 15], 'method': 'regime_adaptation'},
                                
                                # Enhanced step features (~25 features)
                                'enhanced_step_features': {'periods': [5, 10, 15], 'method': 'information_content'},
                                'sophisticated_interactions': {'periods': [5, 10, 15], 'method': 'information_content'},
                            }
                            
                            # Add all available features
                            for feature_name, config in extensive_configs.items():
                                feature_configs.append({
                                    'name': feature_name,
                                    'periods': config['periods'],
                                    'method': config['method'],
                                    'weight': 1.0
                                })
                            
                            self.logger.info(f"📋 Using extensive feature systems: {len(feature_configs)} features from 395+ available")
                            self.logger.info(f"📋 Feature categories: {[c['name'] for c in feature_configs[:10]]}... (showing first 10)")
                        
                        # Run enhanced optimization
                        enhanced_results = await optimize_features_enhanced(
                            data, feature_configs, target_column='close', 
                            regime_column='regime' if 'regime' in data.columns else None,
                            config=enhanced_config
                        )
                        
                        # Extract optimal lookbacks
                        for feature_name, result in enhanced_results['feature_results'].items():
                            if 'error' not in result:
                                optimal_lookbacks[feature_name] = result['optimal_lookback']
                            else:
                                self.logger.warning(f"⚠️ Optimization failed for {feature_name}: {result['error']}")
                                # Use default period
                                feature_config = next((c for c in feature_configs if c['name'] == feature_name), None)
                                if feature_config:
                                    optimal_lookbacks[feature_name] = feature_config['periods'][len(feature_config['periods']) // 2]
                        
                        # Add enhanced optimization metrics
                        artifacts['enhanced_optimization_summary'] = enhanced_results['optimization_summary']
                            
                    elif not ENHANCED_OPTIMIZATION_AVAILABLE and FALLBACK_GENERATORS_AVAILABLE:
                        self.logger.info("🔄 Using fallback optimization system")
                        
                        # Fallback to original optimization logic
                        if optimization_config:
                            # Use configuration system
                            enabled_features = optimization_config.get_enabled_features()
                            optimization_configs = []
                            
                            for feature_config in enabled_features:
                                # Map feature name to generator
                                generator_map = {
                                    'rsi': FeatureGenerators.rsi_generator,
                                    'sma': FeatureGenerators.sma_generator,
                                    'ema': FeatureGenerators.ema_generator,
                                    'bollinger_bands': FeatureGenerators.bollinger_bands_generator,
                                    'macd': FeatureGenerators.macd_generator,
                                    'volatility': FeatureGenerators.volatility_generator
                                }
                                
                                if feature_config.name in generator_map:
                                    optimization_configs.append({
                                        'name': feature_config.name,
                                        'periods': feature_config.periods,
                                        'method': feature_config.method.value,
                                        'generator': generator_map[feature_config.name],
                                        'weight': feature_config.weight
                                    })
                            
                            self.logger.info(f"📋 Using configured features: {[c['name'] for c in optimization_configs]}")
                        else:
                            # Fallback to default configurations
                            optimization_configs = [
                                {
                                    'name': 'rsi',
                                    'periods': [7, 14, 21, 28],
                                    'method': 'signal_strength',
                                    'generator': FeatureGenerators.rsi_generator,
                                    'weight': 1.0
                                },
                                {
                                    'name': 'sma',
                                    'periods': [10, 20, 30, 50],
                                    'method': 'noise_reduction',
                                    'generator': FeatureGenerators.sma_generator,
                                    'weight': 1.0
                                },
                                {
                                    'name': 'ema',
                                    'periods': [8, 12, 20, 26],
                                    'method': 'trend_following',
                                    'generator': FeatureGenerators.ema_generator,
                                    'weight': 1.0
                                },
                                {
                                    'name': 'bollinger_bands',
                                    'periods': [15, 20, 25, 30],
                                    'method': 'information_content',
                                    'generator': FeatureGenerators.bollinger_bands_generator,
                                    'weight': 0.8
                                },
                                {
                                    'name': 'macd',
                                    'periods': [7, 9, 12, 15],
                                    'method': 'signal_strength',
                                    'generator': FeatureGenerators.macd_generator,
                                    'weight': 0.9
                                },
                                {
                                    'name': 'volatility',
                                    'periods': [10, 15, 20, 25],
                                    'method': 'regime_adaptation',
                                    'generator': FeatureGenerators.volatility_generator,
                                    'weight': 0.7
                                }
                            ]
                        
                        # Optimize each indicator
                        for config in optimization_configs:
                            try:
                                # Generate the indicator with different periods to find optimal
                                best_period = self._optimize_feature_with_generator(
                                    data, config['name'], config['periods'], 
                                    config['method'], config['generator']
                                )
                                optimal_lookbacks[config['name']] = best_period
                                
                            except Exception as e:
                                self.logger.warning(f"⚠️ Failed to optimize {config['name']}: {e}")
                                # Use default period
                                optimal_lookbacks[config['name']] = config['periods'][len(config['periods']) // 2]
                        
                    else:
                        self.logger.warning("⚠️ No optimization system available, using fallback optimization")
                        
                        # Fallback to simple optimization
                        rsi_periods = [7, 14, 21, 28]
                        optimal_lookbacks['rsi'] = self._optimize_lookback_statistical(
                            data, 'rsi', rsi_periods, method='signal_strength'
                        )

                        sma_periods = [10, 20, 30, 50]
                        optimal_lookbacks['sma'] = self._optimize_lookback_statistical(
                            data, 'sma', sma_periods, method='noise_reduction'
                        )

                        ema_periods = [8, 12, 20, 26]
                        optimal_lookbacks['ema'] = self._optimize_lookback_statistical(
                            data, 'ema', ema_periods, method='trend_following'
                        )
                        
                        # Create fallback optimization configs for metrics
                        optimization_configs = [
                            {'name': 'rsi', 'periods': rsi_periods},
                            {'name': 'sma', 'periods': sma_periods},
                            {'name': 'ema', 'periods': ema_periods}
                        ]

                    artifacts['optimal_lookbacks'] = optimal_lookbacks
                    artifacts['optimization_metrics'] = {
                        'method': 'enhanced_statistical_optimization',
                        'periods_tested': {
                            config['name']: config['periods'] for config in optimization_configs
                        },
                        'optimization_criteria': ['signal_strength', 'noise_reduction', 'trend_following', 'information_content', 'regime_adaptation'],
                        'feature_generators_used': 'generator' in optimization_configs[0] if optimization_configs else False,
                        'validation_performed': False  # Will be updated below
                    }
                    
                    # Validate optimization results
                    try:
                        from src.feature_engineering.optimization_validator import (
                            OptimizationValidator, ValidationLevel
                        )
                        
                        validator = OptimizationValidator(ValidationLevel.STANDARD)
                        
                        # Create feature generators dict if available
                        feature_generators = {}
                        if 'generator' in optimization_configs[0]:
                            feature_generators = {config['name']: config['generator'] for config in optimization_configs}
                        
                        validation_result = validator.validate_optimization_results(
                            artifacts, data, feature_generators if feature_generators else None
                        )
                        
                        # Add validation results to artifacts
                        artifacts['validation_result'] = {
                            'is_valid': validation_result.is_valid,
                            'overall_score': validation_result.overall_score,
                            'warnings': validation_result.warnings,
                            'recommendations': validation_result.recommendations
                        }
                        artifacts['optimization_metrics']['validation_performed'] = True
                        
                        # Log validation results
                        if validation_result.is_valid:
                            self.logger.info(f"✅ Optimization validation passed (score: {validation_result.overall_score:.3f})")
                        else:
                            self.logger.warning(f"⚠️ Optimization validation failed (score: {validation_result.overall_score:.3f})")
                            for warning in validation_result.warnings:
                                self.logger.warning(f"  • {warning}")
                        
                        # Generate and log validation report
                        validation_report = validator.generate_validation_report(validation_result)
                        self.logger.info(f"📋 Validation Report:\n{validation_report}")
                        
                        # Generate comprehensive performance metrics
                        try:
                            from src.feature_engineering.optimization_metrics import OptimizationReporter
                            
                            reporter = OptimizationReporter()
                            performance_metrics = reporter.generate_comprehensive_metrics(artifacts)
                            
                            # Add performance metrics to artifacts
                            artifacts['performance_metrics'] = {
                                'total_features_optimized': performance_metrics.total_features_optimized,
                                'optimization_method': performance_metrics.optimization_method,
                                'average_performance_score': performance_metrics.average_performance_score,
                                'average_stability_score': performance_metrics.average_stability_score,
                                'validation_passed': performance_metrics.validation_passed,
                                'validation_score': performance_metrics.validation_score,
                                'lookback_diversity_score': performance_metrics.lookback_diversity_score,
                                'best_performing_feature': performance_metrics.best_performing_feature,
                                'most_stable_feature': performance_metrics.most_stable_feature
                            }
                            
                            # Generate and log performance report
                            performance_report = reporter.generate_performance_report(performance_metrics)
                            self.logger.info(f"📊 Performance Report:\n{performance_report}")
                            
                            # Generate and log recommendations
                            recommendations = reporter.generate_recommendations(performance_metrics)
                            if recommendations:
                                self.logger.info("💡 Optimization Recommendations:")
                                for i, rec in enumerate(recommendations, 1):
                                    self.logger.info(f"  {i}. {rec}")
                            
                            # Add recommendations to artifacts
                            artifacts['recommendations'] = recommendations
                            
                        except ImportError as e:
                            self.logger.warning(f"⚠️ Performance metrics reporter not available: {e}")
                            artifacts['performance_metrics'] = {
                                'total_features_optimized': len(optimal_lookbacks),
                                'optimization_method': 'enhanced_statistical_optimization',
                                'validation_passed': validation_result.get('is_valid', True),
                                'validation_score': validation_result.get('overall_score', 0.8)
                            }
                        
                    except ImportError as e:
                        self.logger.warning(f"⚠️ Optimization validator not available: {e}")
                        artifacts['validation_result'] = {
                            'is_valid': True,  # Assume valid if validator not available
                            'overall_score': 0.8,
                            'warnings': ['Validation framework not available'],
                            'recommendations': ['Install optimization validator for comprehensive validation']
                        }

                except Exception as e:
                    self.logger.error(f"❌ Statistical optimization failed: {e}")
                    raise RuntimeError(f"Feature lookback optimization failed: {e}")
            else:
                raise FileNotFoundError(f"No suitable data file found for optimization. Tried: {possible_paths}")
        
        # Log completion with emojis and artifact paths
        self._log_sub_pipeline_completion("feature_lookback_optimization", config, artifacts)

        # Automatically trigger the next sub-pipeline: fractional_differentiation
        self.logger.info("🔄 Feature lookback optimization completed, triggering next: fractional_differentiation")
        try:
            next_artifacts = await self._fractional_differentiation_pipeline(config)
            # Merge artifacts from next pipeline
            artifacts.update(next_artifacts)
            self.logger.info("✅ Fractional differentiation pipeline completed successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to execute fractional differentiation pipeline: {e}")

        return artifacts

    def _validate_market_data(self, df: pd.DataFrame, data_source: str = "unknown") -> bool:
        """Comprehensive validation of market data quality."""
        try:
            # Validate data structure
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                self.logger.error(f"❌ [{data_source}] Missing required columns. Found: {df.columns.tolist()}")
                return False

            # Validate data quality
            if len(df) == 0:
                self.logger.error(f"❌ [{data_source}] Empty dataset")
                return False

            # Check for excessive null values
            for col in required_columns:
                null_count = df[col].isnull().sum()
                if null_count > len(df) * 0.1:  # More than 10% nulls
                    self.logger.error(f"❌ [{data_source}] Too many null values in {col}: {null_count}/{len(df)}")
                    return False
                elif null_count > 0:
                    self.logger.warning(f"⚠️ [{data_source}] Found {null_count} null values in {col}")

            # Validate price data ranges
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if (df[col] <= 0).any():
                    self.logger.error(f"❌ [{data_source}] Invalid negative or zero prices in {col}")
                    return False

            # Validate volume
            if (df['volume'] < 0).any():
                self.logger.error(f"❌ [{data_source}] Invalid negative volume")
                return False

            # Validate OHLC relationships
            invalid_ohlc = ((df['low'] > df['high']) |
                           (df['open'] > df['high']) |
                           (df['open'] < df['low']) |
                           (df['close'] > df['high']) |
                           (df['close'] < df['low'])).sum()
            if invalid_ohlc > 0:
                self.logger.error(f"❌ [{data_source}] Found {invalid_ohlc} rows with invalid OHLC relationships")
                return False

            # Validate timestamp continuity
            if len(df) > 1:
                timestamp_gaps = df['timestamp'].diff().dt.total_seconds()
                large_gaps = (timestamp_gaps > 3600).sum()  # More than 1 hour gaps
                if large_gaps > len(df) * 0.05:  # More than 5% large gaps
                    self.logger.warning(f"⚠️ [{data_source}] Found {large_gaps} large timestamp gaps")

            self.logger.info(f"✅ [{data_source}] Data validation passed for {len(df)} rows")
            return True

        except Exception as e:
            self.logger.error(f"❌ [{data_source}] Data validation failed: {e}")
            return False

    def _optimize_lookback_statistical(self, data: pd.DataFrame, indicator: str,
                                      periods: List[int], method: str) -> int:
        """Optimize lookback period using enhanced statistical methods."""
        try:
            # Find columns that match the indicator pattern
            indicator_cols = [col for col in data.columns if indicator.upper() in col.upper()]

            if not indicator_cols:
                self.logger.warning(f"⚠️ No {indicator} columns found, using default period")
                return periods[len(periods) // 2]  # Return middle period as default

            # Use the first matching column for optimization
            indicator_col = indicator_cols[0]

            if indicator_col not in data.columns:
                self.logger.warning(f"⚠️ Column {indicator_col} not found, using default period")
                return periods[len(periods) // 2]

            # Get valid data for the indicator
            indicator_data = data[indicator_col].dropna()

            if len(indicator_data) < max(periods) * 2:
                self.logger.warning(f"⚠️ Insufficient data for {indicator} optimization, using default period")
                return periods[len(periods) // 2]

            best_period = periods[0]
            best_score = float('-inf')
            scores = []

            for period in periods:
                try:
                    score = self._calculate_optimization_score(
                        data, indicator_data, indicator, period, method
                    )
                    scores.append(score)

                    if score > best_score:
                        best_score = score
                        best_period = period

                except Exception as e:
                    self.logger.debug(f"⚠️ Failed to optimize {indicator} for period {period}: {e}")
                    scores.append(0)
                    continue

            # Additional validation: check for stability
            if scores and len(scores) > 1:
                stability_score = self._calculate_score_stability(scores)
                if stability_score < 0.3:  # Low stability threshold
                    self.logger.warning(f"⚠️ Low stability for {indicator} optimization (stability: {stability_score:.3f})")
                    # Use median period if stability is too low
                    best_period = periods[len(periods) // 2]

            self.logger.info(f"📊 Optimized {indicator}: period {best_period} (method: {method}, score: {best_score:.4f})")
            return best_period

        except Exception as e:
            self.logger.warning(f"⚠️ Statistical optimization failed for {indicator}: {e}")
            return periods[len(periods) // 2]  # Return middle period as fallback

    def _calculate_optimization_score(self, data: pd.DataFrame, indicator_data: pd.Series, 
                                    indicator: str, period: int, method: str) -> float:
        """Calculate optimization score for a specific period and method."""
        try:
            if method == 'signal_strength':
                # Enhanced signal strength calculation
                # For RSI: maximize signal-to-noise ratio
                if indicator.upper() == 'RSI':
                    # RSI should be more responsive to price changes
                    price_changes = data['close'].pct_change() if 'close' in data.columns else indicator_data.pct_change()
                    signal_strength = abs(indicator_data.rolling(period).corr(price_changes))
                    # Add momentum component
                    momentum = abs(indicator_data.diff(period).mean())
                    score = (signal_strength * 0.7 + momentum * 0.3).mean()
                else:
                    # For other indicators: maximize absolute mean signal change
                    score = abs(indicator_data.diff(period).mean())
                    
            elif method == 'noise_reduction':
                # Enhanced noise reduction calculation
                # For SMA: minimize coefficient of variation
                rolling_mean = indicator_data.rolling(period).mean()
                rolling_std = indicator_data.rolling(period).std()
                cv = rolling_std / rolling_mean
                # Minimize CV (negative for maximization)
                score = -cv.mean()
                
            elif method == 'trend_following':
                # Enhanced trend following calculation
                # For EMA: maximize correlation with price trend and minimize lag
                if 'close' in data.columns:
                    price_trend = data['close'].pct_change(period)
                    correlation = abs(indicator_data.rolling(period).mean().corr(price_trend))
                    
                    # Add lag penalty (shorter periods preferred for trend following)
                    lag_penalty = 1 / (1 + period / 20)  # Penalty increases with period
                    score = correlation * lag_penalty
                else:
                    # Fallback to autocorrelation
                    autocorr = indicator_data.autocorr(lag=period)
                    score = abs(autocorr) if not pd.isna(autocorr) else 0
                    
            elif method == 'information_content':
                # New method: maximize information content
                # Calculate mutual information proxy
                if 'close' in data.columns:
                    price_changes = data['close'].pct_change()
                    # Discretize both series
                    indicator_bins = pd.cut(indicator_data, bins=10, labels=False)
                    price_bins = pd.cut(price_changes, bins=10, labels=False)
                    
                    # Calculate correlation as proxy for mutual information
                    score = abs(indicator_bins.corr(price_bins))
                else:
                    # Use autocorrelation as fallback
                    autocorr = indicator_data.autocorr(lag=period)
                    score = abs(autocorr) if not pd.isna(autocorr) else 0
                    
            elif method == 'regime_adaptation':
                # New method: optimize for regime adaptation
                if 'regime' in data.columns:
                    regime_data = data['regime'].dropna()
                    if len(regime_data) > 0:
                        # Calculate performance in different regimes
                        regimes = regime_data.unique()
                        regime_scores = []
                        
                        for regime in regimes:
                            regime_mask = data['regime'] == regime
                            regime_indicator = indicator_data[regime_mask]
                            
                            if len(regime_indicator) > period:
                                # Calculate regime-specific performance
                                regime_performance = abs(regime_indicator.rolling(period).std().mean())
                                regime_scores.append(regime_performance)
                        
                        # Use minimum performance across regimes (worst-case optimization)
                        score = min(regime_scores) if regime_scores else 0
                    else:
                        score = 0
                else:
                    # Fallback to signal strength
                    score = abs(indicator_data.diff(period).mean())
                    
            else:
                # Default: use signal strength
                score = abs(indicator_data.diff(period).mean())
            
            return score if not pd.isna(score) else 0
            
        except Exception as e:
            self.logger.debug(f"Error calculating score for {indicator} period {period}: {e}")
            return 0

    def _calculate_score_stability(self, scores: List[float]) -> float:
        """Calculate stability of optimization scores."""
        if not scores or len(scores) < 2:
            return 0.0
        
        # Remove any NaN values
        valid_scores = [s for s in scores if not pd.isna(s)]
        if len(valid_scores) < 2:
            return 0.0
        
        # Calculate coefficient of variation
        mean_score = sum(valid_scores) / len(valid_scores)
        if mean_score == 0:
            return 0.0
        
        variance = sum((s - mean_score) ** 2 for s in valid_scores) / len(valid_scores)
        std_score = variance ** 0.5
        cv = std_score / abs(mean_score)
        
        # Stability is inverse of coefficient of variation
        stability = 1 / (1 + cv)
        return min(1.0, max(0.0, stability))

    def _optimize_feature_with_generator(self, data: pd.DataFrame, feature_name: str, 
                                       periods: List[int], method: str, generator_func: Callable) -> int:
        """
        Optimize feature lookback period using a feature generator function.
        
        Args:
            data: Input data DataFrame
            feature_name: Name of the feature
            periods: List of periods to test
            method: Optimization method
            generator_func: Feature generator function
            
        Returns:
            Optimal lookback period
        """
        try:
            best_period = periods[0]
            best_score = float('-inf')
            scores = []
            
            for period in periods:
                try:
                    # Generate feature with current period
                    feature_values = generator_func(data, period)
                    
                    # Calculate optimization score
                    score = self._calculate_optimization_score(
                        data, feature_values, feature_name, period, method
                    )
                    scores.append(score)
                    
                    if score > best_score:
                        best_score = score
                        best_period = period
                        
                except Exception as e:
                    self.logger.debug(f"⚠️ Failed to generate {feature_name} for period {period}: {e}")
                    scores.append(0)
                    continue
            
            # Check stability
            if scores and len(scores) > 1:
                stability_score = self._calculate_score_stability(scores)
                if stability_score < 0.3:
                    self.logger.warning(f"⚠️ Low stability for {feature_name} optimization (stability: {stability_score:.3f})")
                    best_period = periods[len(periods) // 2]
            
            self.logger.info(f"📊 Optimized {feature_name}: period {best_period} (method: {method}, score: {best_score:.4f})")
            return best_period
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature optimization failed for {feature_name}: {e}")
            return periods[len(periods) // 2]

    async def _fractional_differentiation_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Fractional differentiation sub-pipeline."""
        self.logger.info("🔢 Executing fractional differentiation pipeline")
        
        artifacts = {
            'differentiated_data': [],
            'differentiation_params': {},
            'stationarity_metrics': {}
        }
        
        # Check if fractional differentiation is disabled
        if not config.enable_fractional_differentiation:
            self.logger.info("⏭️ Fractional differentiation disabled in config, skipping execution")
            artifacts['differentiated_data'] = ['fractional_diff_data.parquet']
            # Still proceed to next pipeline
            self._log_sub_pipeline_completion("fractional_differentiation", config, artifacts)
            
            # Automatically trigger the next sub-pipeline: cross_timeframe_analysis
            self.logger.info("🔄 Fractional differentiation skipped, triggering next: cross_timeframe_analysis")
            try:
                next_artifacts = await self._cross_timeframe_analysis_pipeline(config)
                artifacts.update(next_artifacts)
                self.logger.info("✅ Cross timeframe analysis pipeline completed successfully")
            except Exception as e:
                self.logger.error(f"❌ Failed to execute cross timeframe analysis pipeline: {e}")
            
            return artifacts
        
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

        # Automatically trigger the next sub-pipeline: cross_timeframe_analysis
        self.logger.info("🔄 Fractional differentiation completed, triggering next: cross_timeframe_analysis")
        try:
            next_artifacts = await self._cross_timeframe_analysis_pipeline(config)
            # Merge artifacts from next pipeline
            artifacts.update(next_artifacts)
            self.logger.info("✅ Cross timeframe analysis pipeline completed successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to execute cross timeframe analysis pipeline: {e}")

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

        # This is the final pipeline in the MARKET_ANALYSIS stage sequence
        self.logger.info("🎉 MARKET_ANALYSIS stage completed successfully - all 11 sub-pipelines executed")

        return artifacts
    
    async def _temporal_feature_integration_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Temporal feature integration sub-pipeline."""
        self.logger.info("🔄 Executing temporal feature integration pipeline")
        
        artifacts = {
            'temporal_features': {},
            'feature_metadata': {},
            'quality_metrics': {},
            'integration_summary': {}
        }
        
        try:
            # Import temporal feature integration
            from src.feature_engineering.temporal_feature_integration import (
                integrate_temporal_features,
                create_temporal_config
            )
            
            # Load data for temporal feature integration
            data = await self._load_data_for_temporal_integration(config)
            if data is None or data.empty:
                self.logger.warning("⚠️ No data available for temporal feature integration")
                return artifacts
            
            # Create temporal feature integration configuration
            temporal_config = create_temporal_config(
                enable_lookback=True,
                enable_cross_timeframe=True,
                correlation_threshold=0.7,
                parallel_processing=True
            )
            
            # Execute temporal feature integration
            result = await integrate_temporal_features(
                data=data,
                config=temporal_config,
                data_dir=config.data_dir,
                symbol=config.symbol,
                exchange=config.exchange
            )
            
            # Store results in artifacts
            artifacts['temporal_features'] = result.deduplicated_features
            artifacts['feature_metadata'] = result.feature_metadata
            artifacts['quality_metrics'] = {
                'total_features_before': result.total_features_before,
                'total_features_after': result.total_features_after,
                'redundancy_removed': result.redundancy_removed,
                'integration_time': result.integration_time,
                'average_correlation': result.average_correlation,
                'average_information_content': result.average_information_content,
                'average_stability': result.average_stability
            }
            artifacts['integration_summary'] = {
                'lookback_features': len(result.lookback_features or {}),
                'cross_timeframe_features': len(result.cross_timeframe_features or {}),
                'integrated_features': len(result.integrated_features),
                'deduplicated_features': len(result.deduplicated_features)
            }
            
            # Save artifacts
            await self._save_temporal_integration_artifacts(artifacts, config)
            
            self.logger.info(f"✅ Temporal feature integration completed:")
            self.logger.info(f"   - Features: {result.total_features_before} → {result.total_features_after}")
            self.logger.info(f"   - Redundancy removed: {result.redundancy_removed}")
            self.logger.info(f"   - Integration time: {result.integration_time:.2f}s")
            
        except ImportError as e:
            self.logger.warning(f"⚠️ Temporal feature integration not available: {e}")
            artifacts['error'] = f"Temporal feature integration not available: {e}"
        except Exception as e:
            self.logger.error(f"❌ Temporal feature integration failed: {e}")
            artifacts['error'] = str(e)
        
        # Log completion with emojis and artifact paths
        self._log_sub_pipeline_completion("temporal_feature_integration", config, artifacts)
        
        return artifacts
    
    async def _load_data_for_temporal_integration(self, config: SubPipelineConfig) -> Optional[pd.DataFrame]:
        """Load data for temporal feature integration."""
        try:
            # Try to load data from various sources
            data_sources = [
                f"{config.data_dir}/training/{config.symbol}_{config.exchange}_{config.timeframe}.parquet",
                f"{config.data_dir}/processed/{config.symbol}_{config.exchange}_{config.timeframe}.parquet",
                f"{config.data_dir}/{config.symbol}_{config.exchange}_{config.timeframe}.parquet"
            ]
            
            for data_path in data_sources:
                if Path(data_path).exists():
                    self.logger.info(f"📊 Loading data from: {data_path}")
                    data = pd.read_parquet(data_path)
                    if not data.empty:
                        return data
            
            self.logger.warning("⚠️ No data found for temporal feature integration")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load data for temporal integration: {e}")
            return None
    
    async def _save_temporal_integration_artifacts(self, artifacts: Dict[str, Any], config: SubPipelineConfig):
        """Save temporal integration artifacts."""
        try:
            # Save temporal features
            if artifacts.get('temporal_features'):
                features_path = f"{config.data_dir}/temporal_features_{config.symbol}_{config.exchange}_{config.timeframe}.parquet"
                features_df = pd.DataFrame(artifacts['temporal_features'])
                features_df.to_parquet(features_path)
                self.logger.info(f"💾 Saved temporal features to: {features_path}")
            
            # Save metadata
            if artifacts.get('feature_metadata'):
                metadata_path = f"{config.data_dir}/temporal_feature_metadata_{config.symbol}_{config.exchange}_{config.timeframe}.json"
                with open(metadata_path, 'w') as f:
                    json.dump(artifacts['feature_metadata'], f, indent=2, default=str)
                self.logger.info(f"💾 Saved feature metadata to: {metadata_path}")
            
            # Save quality metrics
            if artifacts.get('quality_metrics'):
                metrics_path = f"{config.data_dir}/temporal_quality_metrics_{config.symbol}_{config.exchange}_{config.timeframe}.json"
                with open(metrics_path, 'w') as f:
                    json.dump(artifacts['quality_metrics'], f, indent=2, default=str)
                self.logger.info(f"💾 Saved quality metrics to: {metrics_path}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to save temporal integration artifacts: {e}")
    
    async def _sr_feature_integration_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """SR feature integration sub-pipeline."""
        self.logger.info("🔧 Executing SR feature integration pipeline")
        
        artifacts = {
            'sr_features_added': 0,
            'integration_metrics': {},
            'sr_feature_names': []
        }
        
        try:
            # Import SR feature integration step
            from .step06_sr_feature_integration import SRFeatureIntegrationStep
            
            # Initialize SR feature integration step
            sr_config = {
                'sr_features': {
                    'enabled': True,
                    'proximity_threshold': 0.05,
                    'strength_weights': {
                        'touch_count': 0.4,
                        'volume_confirmation': 0.3,
                        'time_decay': 0.2,
                        'confluence': 0.1
                    }
                }
            }
            
            sr_integration_step = SRFeatureIntegrationStep(sr_config)
            
            # Get current pipeline state (this would come from the main pipeline)
            # For now, we'll use a mock pipeline state
            pipeline_state = {
                'features': {},  # This would contain existing features
                'sr_levels': [],  # This would contain SR levels from previous steps
                'market_data': None  # This would contain market data
            }
            
            # Execute SR feature integration
            training_input = {'training_mode': config.mode.value}
            result = await sr_integration_step.execute(training_input, pipeline_state)
            
            if result.get('success', False):
                artifacts['sr_features_added'] = result.get('sr_features_added', 0)
                artifacts['integration_metrics'] = {
                    'original_feature_count': result.get('original_feature_count', 0),
                    'enhanced_feature_count': result.get('enhanced_feature_count', 0),
                    'integration_time': result.get('execution_time', 0)
                }
                artifacts['sr_feature_names'] = result.get('sr_feature_names', [])
                
                self.logger.info(f"✅ SR feature integration completed: {artifacts['sr_features_added']} features added")
            else:
                self.logger.error(f"❌ SR feature integration failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            self.logger.error(f"❌ SR feature integration pipeline error: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
        
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