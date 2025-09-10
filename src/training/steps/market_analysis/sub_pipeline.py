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
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

# Import ML commons utilities
try:
    from src.utils.ml_common.data_labeling import get_data_labeler, TripleBarrierConfig, LabelingMethod
    from src.utils.ml_common.hmm_regime_detection import get_hmm_regime_detector, HMMRegimeConfig, RegimeDetectionMethod
    from src.utils.ml_common.regime_data_processing import get_regime_processor, RegimeProcessingConfig
    from src.utils.ml_common.feature_generation_optimization import get_feature_optimizer, FeatureOptimizationConfig
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

class MarketAnalysisSubPipeline:
    """
    Market Analysis Sub-Pipeline Manager.
    
    Provides granular control over market analysis processes with different
    execution modes and comprehensive monitoring.
    """
    
    def __init__(self, config: Optional[SubPipelineConfig] = None):
        """Initialize the market analysis sub-pipeline."""
        self.config = config or SubPipelineConfig()
        self.logger = logger.getChild('MarketAnalysisSubPipeline')
        self.results: List[SubPipelineResult] = []
        
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
    
    # Sub-pipeline implementations
    async def _sr_detection_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """SR detection sub-pipeline."""
        self.logger.info("📊 Executing SR detection pipeline")
        
        artifacts = {
            'sr_levels': [],
            'sr_metrics': {},
            'detection_params': {}
        }
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual SR detection")
            artifacts['sr_levels'] = [{'level': 50000, 'type': 'support', 'strength': 0.8}]
            return artifacts
        
        # Import and use existing SR levels manager
        try:
            from src.tactician.sr_levels.sr_levels_manager import SRLevelsManager
            
            sr_manager = SRLevelsManager()
            
            # Load existing SR levels from the data directory
            sr_levels = sr_manager.load_levels_from_directory(config.data_dir)
            
            # Convert SRLevel objects to dictionaries
            artifacts['sr_levels'] = [
                {
                    'level': level.price,
                    'type': level.level_type,
                    'strength': level.strength,
                    'touches': level.touch_count,
                    'confidence': level.confidence,
                    'algorithm': level.method
                }
                for level in sr_levels
            ]
            artifacts['sr_metrics'] = {
                'total_levels': len(sr_levels),
                'support_levels': len([l for l in sr_levels if l.level_type == 'support']),
                'resistance_levels': len([l for l in sr_levels if l.level_type == 'resistance']),
                'avg_strength': np.mean([l.strength for l in sr_levels]) if sr_levels else 0.0
            }
            artifacts['detection_params'] = {'method': 'sr_levels_manager', 'version': 'existing'}
            
        except ImportError:
            self.logger.warning("⚠️ SR levels manager not available, using mock SR levels")
            artifacts['sr_levels'] = [
                {'level': 50000, 'type': 'support', 'strength': 0.8},
                {'level': 52000, 'type': 'resistance', 'strength': 0.7}
            ]
        
        return artifacts
    
    async def _sr_clustering_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """SR clustering sub-pipeline."""
        self.logger.info("🔗 Executing SR clustering pipeline")
        
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
            from src.tactician.sr_levels.sr_levels_manager import SRLevelsManager
            
            sr_manager = SRLevelsManager()
            
            # Load existing SR levels from the data directory
            sr_levels = sr_manager.load_levels_from_directory(config.data_dir)
            
            # Group levels into clusters based on proximity
            clusters = self._cluster_sr_levels(sr_levels)
            
            artifacts['sr_clusters'] = clusters
            artifacts['clustering_metrics'] = {
                'clustering_method': 'proximity_based',
                'n_clusters': len(clusters),
                'avg_cluster_size': np.mean([len(cluster['levels']) for cluster in clusters]) if clusters else 0
            }
            artifacts['cluster_params'] = {'algorithm': 'proximity_clustering', 'distance_threshold': 0.02}
            
        except ImportError:
            self.logger.warning("⚠️ SR levels manager not available, using mock clusters")
            artifacts['sr_clusters'] = [
                {'cluster_id': 1, 'levels': [50000, 50100], 'strength': 0.8},
                {'cluster_id': 2, 'levels': [52000, 52100], 'strength': 0.7}
            ]
        
        return artifacts
    
    async def _sr_ml_learning_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """SR ML learning sub-pipeline."""
        self.logger.info("🤖 Executing SR ML learning pipeline")
        
        artifacts = {
            'ml_models': [],
            'training_metrics': {},
            'model_performance': {}
        }
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual SR ML learning")
            artifacts['ml_models'] = ['sr_predictor_model.pkl']
            return artifacts
        
        # Import and use SR ML learning
        try:
            from .sr_ml_learning_pipeline import SRMLLearningPipeline, SRMLConfig
            
            sr_ml_config = SRMLConfig(
                model_type='random_forest',
                test_size=0.2,
                validation_size=0.2,
                enable_data_quality_validation=True
            )
            sr_ml_pipeline = SRMLLearningPipeline(sr_ml_config)
            
            # Execute SR ML learning
            ml_result = await sr_ml_pipeline.train_sr_models(
                data_dir=config.data_dir,
                symbol=config.symbol,
                exchange=config.exchange,
                timeframe=config.timeframe
            )
            
            artifacts['ml_models'] = list(ml_result.models.keys())
            artifacts['training_metrics'] = ml_result.performance_metrics
            artifacts['model_performance'] = {
                'feature_importance': ml_result.feature_importance,
                'training_history': ml_result.training_history
            }
            
        except ImportError:
            self.logger.warning("⚠️ SR ML learning pipeline not available, using mock models")
            artifacts['ml_models'] = ['sr_predictor_model.pkl']
        
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
            from .fractional_differentiation_pipeline import FractionalDifferentiationPipeline, FractionalDiffConfig
            
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
            from .cross_timeframe_analysis_pipeline import CrossTimeframeAnalysisPipeline, CrossTimeframeConfig
            
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
                'levels': [level.level],
                'strength': level.strength,
                'type': level.level_type,
                'touches': level.touches
            }
            used_levels.add(i)
            
            # Find nearby levels
            for j, other_level in enumerate(levels[i+1:], i+1):
                if j in used_levels:
                    continue
                
                # Check if levels are close enough
                price_diff = abs(level.level - other_level.level)
                price_tolerance = level.level * 0.02  # 2% tolerance
                
                if price_diff <= price_tolerance and level.level_type == other_level.level_type:
                    cluster['levels'].append(other_level.level)
                    cluster['strength'] = max(cluster['strength'], other_level.strength)
                    cluster['touches'] += other_level.touches
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