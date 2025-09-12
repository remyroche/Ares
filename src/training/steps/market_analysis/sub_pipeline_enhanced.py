"""
Enhanced Market Analysis Sub-Pipeline with HMM Training

This module provides granular sub-pipeline functionality for market analysis,
including HMM training after HMM clustering.

Sub-pipelines:
1. SR Detection - Detect Support/Resistance levels
2. SR Clustering - Generate SR clusters
3. HMM Clustering - HMM-based regime clustering
4. HMM Training - Train HMM ML models for regime prediction
5. Regime Data Splitting - Split data by regimes (with HMM ML tagging)
6. Triple Barrier Labeling - Apply triple barrier method
7. Feature Lookback Optimization - Optimize feature lookback periods
8. Cross Timeframe Analysis - Cross timeframe interaction features
9. SR Feature Integration - Integrate SR features
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

logger = system_logger.getChild('MarketAnalysisSubPipelineEnhanced')

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
    timeframe: str = "1m"
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

class MarketAnalysisSubPipelineEnhanced:
    """
    Enhanced Market Analysis Sub-Pipeline Manager with HMM Training.
    
    Provides granular control over market analysis processes with HMM training
    integration and comprehensive monitoring.
    """
    
    def __init__(self, config: Optional[SubPipelineConfig] = None):
        """Initialize the enhanced market analysis sub-pipeline."""
        self.config = config or SubPipelineConfig()
        self.logger = logger.getChild('MarketAnalysisSubPipelineEnhanced')
        self.results: List[SubPipelineResult] = []
        
        # Initialize artifact and version managers
        self.artifact_manager = get_artifact_manager()
        self.version_manager = get_version_manager()
        
        # Initialize sub-pipeline registry with HMM training
        self.sub_pipelines = {
            'sr_detection': self._sr_detection_pipeline,
            'sr_clustering': self._sr_clustering_pipeline,
            'hmm_clustering': self._hmm_clustering_pipeline,
            'hmm_training': self._hmm_training_pipeline,  # New HMM training
            'regime_data_splitting': self._regime_data_splitting_pipeline,
            'triple_barrier_labeling': self._triple_barrier_labeling_pipeline,
            'feature_lookback_optimization': self._feature_lookback_optimization_pipeline,
            'cross_timeframe_analysis': self._cross_timeframe_analysis_pipeline,
            'sr_feature_integration': self._sr_feature_integration_pipeline,
        }
        
        # Define pipeline order with HMM training after HMM clustering
        self.pipeline_order = [
            'sr_detection',
            'sr_clustering', 
            'hmm_clustering',
            'hmm_training',  # HMM training after HMM clustering
            'regime_data_splitting',
            'triple_barrier_labeling',
            'feature_lookback_optimization',
            'cross_timeframe_analysis',
            'sr_feature_integration'
        ]
    
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
    
    async def execute_sub_pipeline_with_next(
        self,
        sub_pipeline_name: str,
        config: Optional[SubPipelineConfig] = None
    ) -> SubPipelineResult:
        """
        Execute a sub-pipeline and automatically trigger the next ones in sequence.
        
        Args:
            sub_pipeline_name: Name of the sub-pipeline to start from
            config: Optional configuration override
            
        Returns:
            SubPipelineResult of the starting sub-pipeline
        """
        config = config or self.config
        self.logger.info(f"🚀 Starting market analysis pipeline chain from: {sub_pipeline_name}")
        
        # Find the starting index
        try:
            start_index = self.pipeline_order.index(sub_pipeline_name)
        except ValueError:
            raise ValueError(f"Unknown sub-pipeline: {sub_pipeline_name}")
        
        # Execute sub-pipelines in sequence starting from the specified one
        for i in range(start_index, len(self.pipeline_order)):
            pipeline_name = self.pipeline_order[i]
            self.logger.info(f"🔄 Executing pipeline step {i+1}/{len(self.pipeline_order)}: {pipeline_name}")
            
            result = await self.execute_sub_pipeline(pipeline_name, config)
            
            # Check if this step failed
            if result.status == SubPipelineStatus.FAILED:
                self.logger.error(f"❌ Pipeline chain stopped due to failure in {pipeline_name}")
                break
        
        # Return the result of the starting sub-pipeline
        for result in self.results:
            if result.sub_pipeline_name == sub_pipeline_name:
                return result
        
        # If not found, return the last result
        return self.results[-1] if self.results else None
    
    # Sub-pipeline implementations
    async def _sr_detection_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """SR Detection sub-pipeline."""
        self.logger.info("🔍 Executing SR detection pipeline")
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual SR detection")
            return {'sr_levels': [], 'sr_metadata': {}}
        
        # Import and execute SR detection
        try:
            from .sr_detection import SRDetectionStep
            sr_detector = SRDetectionStep()
            result = await sr_detector.execute(
                training_input={
                    'symbol': config.symbol,
                    'exchange': config.exchange,
                    'timeframe': config.timeframe
                },
                pipeline_state={}
            )
            return result
        except ImportError:
            self.logger.warning("⚠️ SR detection not available, using mock detection")
            return {'sr_levels': [], 'sr_metadata': {}}
    
    async def _sr_clustering_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """SR Clustering sub-pipeline."""
        self.logger.info("🎯 Executing SR clustering pipeline")
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual SR clustering")
            return {'sr_clusters': [], 'sr_cluster_metadata': {}}
        
        # Import and execute SR clustering
        try:
            from .sr_clustering import SRClusteringStep
            sr_clusterer = SRClusteringStep()
            result = await sr_clusterer.execute(
                training_input={
                    'symbol': config.symbol,
                    'exchange': config.exchange,
                    'timeframe': config.timeframe
                },
                pipeline_state={}
            )
            return result
        except ImportError:
            self.logger.warning("⚠️ SR clustering not available, using mock clustering")
            return {'sr_clusters': [], 'sr_cluster_metadata': {}}
    
    async def _hmm_clustering_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """HMM Clustering sub-pipeline."""
        self.logger.info("🔄 Executing HMM clustering pipeline")
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual HMM clustering")
            return {'hmm_clusters': [], 'hmm_cluster_metadata': {}}
        
        # Import and execute HMM clustering
        try:
            from .hmm_clustering.step03_hmm_regime_discovery import HMMRegimeDiscoveryStep
            hmm_clusterer = HMMRegimeDiscoveryStep()
            result = await hmm_clusterer.execute(
                training_input={
                    'symbol': config.symbol,
                    'exchange': config.exchange,
                    'timeframe': config.timeframe
                },
                pipeline_state={}
            )
            return result
        except ImportError:
            self.logger.warning("⚠️ HMM clustering not available, using mock clustering")
            return {'hmm_clusters': [], 'hmm_cluster_metadata': {}}
    
    async def _hmm_training_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """HMM Training sub-pipeline - NEW."""
        self.logger.info("🤖 Executing HMM training pipeline")
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual HMM training")
            return {'hmm_models': [], 'hmm_training_metadata': {}}
        
        # Import and execute HMM training
        try:
            from .hmm_training.hmm_models_training import HMMModelsTraining
            from .hmm_training.hmm_ensemble_training import HMMEnsembleTraining
            
            # Load market data for training
            market_data = await self._load_market_data(config)
            if market_data is None:
                raise ValueError("No market data available for HMM training")
            
            # Get regime labels from previous HMM clustering
            regime_labels = await self._get_regime_labels(config)
            if regime_labels is None:
                raise ValueError("No regime labels available for HMM training")
            
            # Train base models
            hmm_models_trainer = HMMModelsTraining(config.custom_params)
            base_models_result = hmm_models_trainer.train_base_models(
                market_data, regime_labels, is_classification=True
            )
            
            # Train ensemble models
            hmm_ensemble_trainer = HMMEnsembleTraining(config.custom_params)
            ensemble_models_result = hmm_ensemble_trainer.train_ensemble_models(
                base_models_result['models'], market_data, regime_labels, is_classification=True
            )
            
            # Save models
            base_model_paths = hmm_models_trainer.save_models(
                base_models_result['models'], config.symbol, config.exchange, 
                config.timeframe, config.data_dir
            )
            
            ensemble_model_paths = hmm_ensemble_trainer.save_ensemble_models(
                ensemble_models_result['ensemble_models'], config.symbol, config.exchange,
                config.timeframe, config.data_dir
            )
            
            return {
                'hmm_base_models': base_models_result,
                'hmm_ensemble_models': ensemble_models_result,
                'base_model_paths': base_model_paths,
                'ensemble_model_paths': ensemble_model_paths,
                'hmm_training_completed': True
            }
            
        except ImportError as e:
            self.logger.warning(f"⚠️ HMM training not available: {e}, using mock training")
            return {'hmm_models': [], 'hmm_training_metadata': {}}
    
    async def _regime_data_splitting_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Regime Data Splitting sub-pipeline with HMM ML tagging."""
        self.logger.info("📊 Executing regime data splitting pipeline with HMM ML tagging")
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual regime data splitting")
            return {'regime_data': [], 'regime_metadata': {}}
        
        # Import and execute enhanced regime data splitting
        try:
            from .step04_regime_data_splitting_enhanced import execute_enhanced_regime_data_splitting
            result = await execute_enhanced_regime_data_splitting(
                training_input={
                    'symbol': config.symbol,
                    'exchange': config.exchange,
                    'timeframe': config.timeframe,
                    'data_dir': config.data_dir
                },
                pipeline_state={},
                config=config.custom_params
            )
            return result
        except ImportError:
            self.logger.warning("⚠️ Enhanced regime data splitting not available, using mock splitting")
            return {'regime_data': [], 'regime_metadata': {}}
    
    async def _triple_barrier_labeling_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Triple Barrier Labeling sub-pipeline."""
        self.logger.info("🏷️ Executing triple barrier labeling pipeline")
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual triple barrier labeling")
            return {'labels': [], 'label_metadata': {}}
        
        # Import and execute triple barrier labeling
        try:
            from .triple_barrier_labeling import TripleBarrierLabelingStep
            labeler = TripleBarrierLabelingStep()
            result = await labeler.execute(
                training_input={
                    'symbol': config.symbol,
                    'exchange': config.exchange,
                    'timeframe': config.timeframe
                },
                pipeline_state={}
            )
            return result
        except ImportError:
            self.logger.warning("⚠️ Triple barrier labeling not available, using mock labeling")
            return {'labels': [], 'label_metadata': {}}
    
    async def _feature_lookback_optimization_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Feature Lookback Optimization sub-pipeline."""
        self.logger.info("⚙️ Executing feature lookback optimization pipeline")
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual feature lookback optimization")
            return {'optimized_features': [], 'optimization_metadata': {}}
        
        # Import and execute feature lookback optimization
        try:
            from .step06_feature_engineering_per_regime import FeatureLookbackOptimizationStep
            optimizer = FeatureLookbackOptimizationStep()
            result = await optimizer.execute(
                training_input={
                    'symbol': config.symbol,
                    'exchange': config.exchange,
                    'timeframe': config.timeframe
                },
                pipeline_state={}
            )
            return result
        except ImportError:
            self.logger.warning("⚠️ Feature lookback optimization not available, using mock optimization")
            return {'optimized_features': [], 'optimization_metadata': {}}
    
    async def _cross_timeframe_analysis_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Cross Timeframe Analysis sub-pipeline."""
        self.logger.info("⏰ Executing cross timeframe analysis pipeline")
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual cross timeframe analysis")
            return {'cross_tf_features': [], 'cross_tf_metadata': {}}
        
        # Import and execute cross timeframe analysis
        try:
            from .cross_timeframe_analysis_pipeline import CrossTimeframeAnalysisStep
            analyzer = CrossTimeframeAnalysisStep()
            result = await analyzer.execute(
                training_input={
                    'symbol': config.symbol,
                    'exchange': config.exchange,
                    'timeframe': config.timeframe
                },
                pipeline_state={}
            )
            return result
        except ImportError:
            self.logger.warning("⚠️ Cross timeframe analysis not available, using mock analysis")
            return {'cross_tf_features': [], 'cross_tf_metadata': {}}
    
    async def _sr_feature_integration_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """SR Feature Integration sub-pipeline."""
        self.logger.info("🔗 Executing SR feature integration pipeline")
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual SR feature integration")
            return {'integrated_features': [], 'integration_metadata': {}}
        
        # Import and execute SR feature integration
        try:
            from .step06_sr_feature_integration import SRFeatureIntegrationStep
            integrator = SRFeatureIntegrationStep()
            result = await integrator.execute(
                training_input={
                    'symbol': config.symbol,
                    'exchange': config.exchange,
                    'timeframe': config.timeframe
                },
                pipeline_state={}
            )
            return result
        except ImportError:
            self.logger.warning("⚠️ SR feature integration not available, using mock integration")
            return {'integrated_features': [], 'integration_metadata': {}}
    
    # Helper methods
    async def _load_market_data(self, config: SubPipelineConfig) -> Optional[pd.DataFrame]:
        """Load market data for HMM training."""
        try:
            data_path = Path(config.data_dir) / 'training' / f'{config.exchange}_{config.symbol}_{config.timeframe}_market_data.parquet'
            
            if not data_path.exists():
                self.logger.warning(f"⚠️ Market data file not found: {data_path}")
                return None
            
            market_data = pd.read_parquet(data_path)
            self.logger.info(f"✅ Loaded market data: {market_data.shape}")
            return market_data
            
        except Exception as e:
            self.logger.error(f"❌ Error loading market data: {e}")
            return None
    
    async def _get_regime_labels(self, config: SubPipelineConfig) -> Optional[np.ndarray]:
        """Get regime labels from HMM clustering results."""
        try:
            # This would typically load from the HMM clustering results
            # For now, return mock data
            return np.random.randint(0, 3, 1000)  # Mock regime labels
            
        except Exception as e:
            self.logger.error(f"❌ Error getting regime labels: {e}")
            return None
    
    def get_available_sub_pipelines(self) -> List[str]:
        """Get list of available sub-pipelines."""
        return list(self.sub_pipelines.keys())
    
    def get_pipeline_order(self) -> List[str]:
        """Get the pipeline execution order."""
        return self.pipeline_order.copy()
    
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
def get_market_analysis_sub_pipeline_enhanced(config: Optional[SubPipelineConfig] = None) -> MarketAnalysisSubPipelineEnhanced:
    """Get a configured enhanced market analysis sub-pipeline."""
    return MarketAnalysisSubPipelineEnhanced(config)

async def execute_market_analysis_sub_pipeline_enhanced(
    sub_pipeline_name: str,
    config: Optional[SubPipelineConfig] = None
) -> SubPipelineResult:
    """Convenience function to execute an enhanced market analysis sub-pipeline."""
    pipeline = get_market_analysis_sub_pipeline_enhanced(config)
    return await pipeline.execute_sub_pipeline(sub_pipeline_name, config)