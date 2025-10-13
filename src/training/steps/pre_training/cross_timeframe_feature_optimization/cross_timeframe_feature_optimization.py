"""
Cross-Timeframe Feature Optimization Component

This component optimizes cross-timeframe features using economic significance evaluation,
backtesting, and feature selection, following the pattern of FeatureLookbackOptimizationComponent.

Key Features:
- Economic significance evaluation for period selection
- Backtesting against financial targets
- Feature optimization and selection
- VectorBT-optimized performance
- Memory-efficient processing
"""

import json
import time
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Import base components
from ..base_pre_training_component import BasePreTrainingComponent, ComponentConfig, ComponentResult
from ..pipeline_state import PipelineState

# Import utility modules
from src.utils.common_operations import safe_dataframe_operation
from src.utils.common_utilities import CommonUtilities
from src.utils.math_validation import safe_divide, validate_finite
from src.utils.serialization_utils import UniversalSerializer, JSONSerializer, PickleSerializer

# Import cross-timeframe components
from ..interaction_feature_generator.feature_interaction_generation.enhanced_data_driven_period_selector import (
    EnhancedDataDrivenPeriodSelector, EnhancedPeriodSelectionConfig, EnhancedPeriodSelectionResult
)
from ..interaction_feature_generator.feature_interaction_generation.economic_period_evaluator import (
    EconomicPeriodEvaluator, EconomicEvaluationConfig, PeriodBacktestResult
)
from ...feature_generation.utils.cross_timeframe_interaction_features import CrossTimeframeFeatureGenerator

# Import optimization components
from .core.cross_timeframe_optimizer import CrossTimeframeOptimizer, CrossTimeframeOptimizationConfig
from .core.feature_backtester import FeatureBacktester, BacktestConfig
from .core.feature_selector import FeatureSelector, SelectionConfig

# Import VectorBT optimizations
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    )
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    VectorBTRollingOptimizer = None
    get_vectorbt_rolling_optimizer = None

# Import performance monitoring
from src.utils.performance_monitor import PerformanceMonitor
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
)

logger = logging.getLogger(__name__)


@dataclass
class CrossTimeframeOptimizationConfig(ComponentConfig):
    """Configuration for cross-timeframe feature optimization."""
    
    # Period selection configuration
    min_period: int = 1
    max_period: int = 50  # Optimized for 15m timeframe
    max_periods: int = 8
    min_data_points: int = 100
    
    # Economic evaluation configuration
    enable_economic_evaluation: bool = True
    min_economic_score: float = 0.4
    economic_weight: float = 0.6
    statistical_weight: float = 0.4
    
    # Backtesting configuration
    backtest_periods: int = 100
    min_backtest_periods: int = 50
    risk_free_rate: float = 0.02
    
    # Feature optimization configuration
    enable_feature_optimization: bool = True
    optimization_method: str = "mrmr"  # mrmr, grid_search, bayesian, random_search
    lookback_range: Tuple[int, int] = (5, 50)
    max_features: int = 20
    
    # Feature selection configuration
    enable_feature_selection: bool = True
    selection_method: str = "mutual_information"  # mutual_information, correlation, variance
    selection_threshold: float = 0.01
    max_correlation: float = 0.95
    
    # Performance optimization
    enable_vectorbt: bool = True
    enable_parallel: bool = True
    memory_efficient: bool = True
    chunk_size: int = 1000
    
    # Timeframe-specific configuration
    target_timeframe: str = "15m"
    enable_multi_timeframe: bool = True


@dataclass
class CrossTimeframeOptimizationResult:
    """Result from cross-timeframe feature optimization."""
    
    # Optimized features
    optimized_features: pd.DataFrame
    selected_features: List[str]
    feature_scores: Dict[str, float]
    
    # Period analysis
    optimal_periods: List[int]
    period_scores: Dict[int, float]
    economic_evaluation_results: Optional[Dict[str, Any]] = None
    
    # Optimization results
    optimization_scores: Dict[str, float]
    backtest_results: Optional[Dict[str, Any]] = None
    
    # Performance metrics
    total_execution_time: float
    memory_usage_mb: float
    successful_optimizations: int
    failed_optimizations: int
    
    # Configuration
    config: CrossTimeframeOptimizationConfig
    
    # Success indicators
    success: bool = True
    error_message: Optional[str] = None


class CrossTimeframeFeatureOptimizationComponent(BasePreTrainingComponent):
    """
    Cross-Timeframe Feature Optimization Component.
    
    Optimizes cross-timeframe features using economic significance evaluation,
    backtesting, and feature selection, following the pattern of FeatureLookbackOptimizationComponent.
    """
    
    def __init__(self, config: Optional[CrossTimeframeOptimizationConfig] = None):
        """Initialize the cross-timeframe feature optimization component."""
        tprint("🔧 Initializing CrossTimeframeFeatureOptimizationComponent...")
        
        super().__init__(config or CrossTimeframeOptimizationConfig())
        self.config = self.config  # Type hint
        
        # Initialize performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize enhanced period selector
        self.period_selector = None
        if self.config.enable_economic_evaluation:
            period_config = EnhancedPeriodSelectionConfig(
                min_period=self.config.min_period,
                max_period=self.config.max_period,
                max_periods=self.config.max_periods,
                min_data_points=self.config.min_data_points,
                enable_economic_evaluation=True,
                min_economic_score=self.config.min_economic_score,
                economic_weight=self.config.economic_weight,
                statistical_weight=self.config.statistical_weight,
                enable_vectorbt=self.config.enable_vectorbt,
                enable_parallel=self.config.enable_parallel,
                memory_efficient=self.config.memory_efficient
            )
            self.period_selector = EnhancedDataDrivenPeriodSelector(period_config)
            tprint("✅ Enhanced period selector initialized")
        
        # Initialize cross-timeframe feature generator
        self.feature_generator = CrossTimeframeFeatureGenerator()
        tprint("✅ Cross-timeframe feature generator initialized")
        
        # Initialize optimizer
        self.optimizer = None
        if self.config.enable_feature_optimization:
            from .core.cross_timeframe_optimizer import CrossTimeframeOptimizer, CrossTimeframeOptimizationConfig
            opt_config = CrossTimeframeOptimizationConfig(
                optimization_method=self.config.optimization_method,
                lookback_range=self.config.lookback_range,
                max_features=self.config.max_features,
                enable_vectorbt=self.config.enable_vectorbt,
                enable_parallel=self.config.enable_parallel,
                memory_efficient=self.config.memory_efficient
            )
            self.optimizer = CrossTimeframeOptimizer(opt_config)
            tprint("✅ Cross-timeframe optimizer initialized")
        
        # Initialize backtester
        self.backtester = None
        if self.config.enable_economic_evaluation:
            from .core.feature_backtester import FeatureBacktester, BacktestConfig
            backtest_config = BacktestConfig(
                backtest_periods=self.config.backtest_periods,
                min_backtest_periods=self.config.min_backtest_periods,
                risk_free_rate=self.config.risk_free_rate,
                enable_vectorbt=self.config.enable_vectorbt
            )
            self.backtester = FeatureBacktester(backtest_config)
            tprint("✅ Feature backtester initialized")
        
        # Initialize feature selector
        self.feature_selector = None
        if self.config.enable_feature_selection:
            from .core.feature_selector import FeatureSelector, SelectionConfig
            selection_config = SelectionConfig(
                selection_method=self.config.selection_method,
                selection_threshold=self.config.selection_threshold,
                max_correlation=self.config.max_correlation,
                enable_vectorbt=self.config.enable_vectorbt
            )
            self.feature_selector = FeatureSelector(selection_config)
            tprint("✅ Feature selector initialized")
        
        tprint("✅ CrossTimeframeFeatureOptimizationComponent initialized successfully")
    
    async def execute(self, data: Any, pipeline_state: PipelineState) -> ComponentResult:
        """
        Execute cross-timeframe feature optimization.
        
        Args:
            data: Input data for optimization
            pipeline_state: Current pipeline state
            
        Returns:
            ComponentResult with optimization results
        """
        tprint("🚀 Starting cross-timeframe feature optimization execution...")
        start_time = time.time()
        
        try:
            # Validate inputs
            pipeline_state = PipelineState.ensure(pipeline_state)
            
            # Load data if not provided
            if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                tprint("📥 Loading data from pipeline state...")
                data = await self._load_data_from_pipeline_state(pipeline_state)
            
            if data is None or data.empty:
                raise ValueError("No data available for optimization")
            
            tprint(f"📊 Data loaded: {data.shape}")
            
            # Step 1: Select optimal periods using economic significance
            tprint("🎯 Step 1: Selecting optimal periods...")
            optimal_periods = await self._select_optimal_periods(data, pipeline_state)
            tprint(f"✅ Selected periods: {optimal_periods}")
            
            # Step 2: Generate cross-timeframe features
            tprint("🔧 Step 2: Generating cross-timeframe features...")
            cross_timeframe_features = await self._generate_cross_timeframe_features(data, optimal_periods)
            tprint(f"✅ Generated {len(cross_timeframe_features)} cross-timeframe features")
            
            # Step 3: Backtest features for economic significance
            tprint("💰 Step 3: Backtesting features for economic significance...")
            backtest_results = await self._backtest_features(data, cross_timeframe_features)
            tprint(f"✅ Backtesting completed: {len(backtest_results)} features evaluated")
            
            # Step 4: Optimize features
            tprint("⚡ Step 4: Optimizing features...")
            optimized_features = await self._optimize_features(data, cross_timeframe_features, backtest_results)
            tprint(f"✅ Feature optimization completed: {len(optimized_features)} features optimized")
            
            # Step 5: Select best features
            tprint("🎯 Step 5: Selecting best features...")
            selected_features, feature_scores = await self._select_features(optimized_features, backtest_results)
            tprint(f"✅ Feature selection completed: {len(selected_features)} features selected")
            
            # Step 6: Create final result
            execution_time = time.time() - start_time
            memory_usage = self.performance_monitor.get_memory_usage()
            
            result = CrossTimeframeOptimizationResult(
                optimized_features=optimized_features,
                selected_features=selected_features,
                feature_scores=feature_scores,
                optimal_periods=optimal_periods,
                period_scores={},  # Will be filled by period selector
                economic_evaluation_results=backtest_results,
                optimization_scores=feature_scores,
                backtest_results=backtest_results,
                total_execution_time=execution_time,
                memory_usage_mb=memory_usage,
                successful_optimizations=len(selected_features),
                failed_optimizations=0,
                config=self.config,
                success=True
            )
            
            tprint_success(f"✅ Cross-timeframe feature optimization completed in {execution_time:.3f}s")
            tprint_info(f"📊 Final results: {len(selected_features)} features selected from {len(cross_timeframe_features)} generated")
            
            return ComponentResult(
                success=True,
                data=result,
                metadata={
                    'execution_time': execution_time,
                    'memory_usage_mb': memory_usage,
                    'selected_features_count': len(selected_features),
                    'generated_features_count': len(cross_timeframe_features),
                    'optimal_periods': optimal_periods
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            tprint_error(f"❌ Cross-timeframe feature optimization failed: {e}")
            
            return ComponentResult(
                success=False,
                data=None,
                metadata={
                    'execution_time': execution_time,
                    'error_message': str(e)
                }
            )
    
    async def _load_data_from_pipeline_state(self, pipeline_state: PipelineState) -> Optional[pd.DataFrame]:
        """Load data from pipeline state."""
        try:
            # Try to get data from various sources in pipeline state
            data_sources = [
                'data',
                'klines_data',
                'price_data',
                'market_data',
                'features'
            ]
            
            for source in data_sources:
                if source in pipeline_state and pipeline_state[source] is not None:
                    data = pipeline_state[source]
                    if isinstance(data, pd.DataFrame) and not data.empty:
                        tprint(f"📥 Data loaded from {source}: {data.shape}")
                        return data
            
            tprint_warning("⚠️ No data found in pipeline state")
            return None
            
        except Exception as e:
            tprint_error(f"❌ Failed to load data from pipeline state: {e}")
            return None
    
    async def _select_optimal_periods(self, data: pd.DataFrame, pipeline_state: PipelineState) -> List[int]:
        """Select optimal periods using economic significance evaluation."""
        try:
            if self.period_selector:
                result = self.period_selector.select_optimal_periods(data, self.config.target_timeframe)
                if result.success and result.optimal_periods:
                    return result.optimal_periods
                else:
                    tprint_warning("⚠️ Period selection failed, using default periods")
            
            # Fallback to default periods
            return [5, 10, 15, 20, 30, 40]
            
        except Exception as e:
            tprint_error(f"❌ Period selection failed: {e}")
            return [5, 10, 15, 20, 30, 40]
    
    async def _generate_cross_timeframe_features(self, data: pd.DataFrame, periods: List[int]) -> Dict[str, pd.Series]:
        """Generate cross-timeframe features using selected periods."""
        try:
            # Update feature generator with selected periods
            if hasattr(self.feature_generator, 'config'):
                self.feature_generator.config.momentum_timeframes = periods
                self.feature_generator.config.volatility_timeframes = periods
                self.feature_generator.config.volume_timeframes = periods
            
            # Generate features
            features = self.feature_generator.generate_cross_timeframe_features(data, data[['volume']])
            
            return features
            
        except Exception as e:
            tprint_error(f"❌ Cross-timeframe feature generation failed: {e}")
            return {}
    
    async def _backtest_features(self, data: pd.DataFrame, features: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Backtest features for economic significance."""
        try:
            if not self.backtester:
                tprint_warning("⚠️ Backtester not available, skipping backtesting")
                return {}
            
            backtest_results = {}
            
            for feature_name, feature_series in features.items():
                try:
                    # Align feature with price data
                    aligned_data = pd.concat([data['close'], feature_series], axis=1).dropna()
                    
                    if len(aligned_data) < self.config.min_backtest_periods:
                        continue
                    
                    # Backtest feature
                    result = self.backtester.backtest_feature(
                        aligned_data['close'], 
                        aligned_data[feature_name],
                        feature_name
                    )
                    
                    if result and result.get('success', False):
                        backtest_results[feature_name] = result
                        
                except Exception as e:
                    tprint_debug(f"⚠️ Backtesting failed for {feature_name}: {e}")
                    continue
            
            return backtest_results
            
        except Exception as e:
            tprint_error(f"❌ Feature backtesting failed: {e}")
            return {}
    
    async def _optimize_features(self, data: pd.DataFrame, features: Dict[str, pd.Series], backtest_results: Dict[str, Any]) -> Dict[str, pd.Series]:
        """Optimize features using the optimizer."""
        try:
            if not self.optimizer:
                tprint_warning("⚠️ Optimizer not available, returning original features")
                return features
            
            # Convert features to DataFrame for optimization
            features_df = pd.DataFrame(features)
            
            # Add target column (using close price returns)
            features_df['target'] = data['close'].pct_change()
            features_df = features_df.dropna()
            
            if features_df.empty:
                tprint_warning("⚠️ No valid data for optimization")
                return features
            
            # Optimize features
            optimized_result = self.optimizer.optimize_features(
                features_df,
                target_column='target',
                lookback_range=self.config.lookback_range
            )
            
            if optimized_result and optimized_result.get('success', False):
                optimized_features = optimized_result.get('optimized_features', {})
                return optimized_features
            else:
                tprint_warning("⚠️ Feature optimization failed, returning original features")
                return features
            
        except Exception as e:
            tprint_error(f"❌ Feature optimization failed: {e}")
            return features
    
    async def _select_features(self, features: Dict[str, pd.Series], backtest_results: Dict[str, Any]) -> Tuple[List[str], Dict[str, float]]:
        """Select best features based on backtest results and other criteria."""
        try:
            if not self.feature_selector:
                tprint_warning("⚠️ Feature selector not available, returning all features")
                return list(features.keys()), {name: 1.0 for name in features.keys()}
            
            # Convert features to DataFrame for selection
            features_df = pd.DataFrame(features)
            
            # Add target column
            features_df['target'] = features_df.index.to_series().apply(lambda x: 1.0)  # Dummy target
            features_df = features_df.dropna()
            
            if features_df.empty:
                tprint_warning("⚠️ No valid data for feature selection")
                return [], {}
            
            # Select features
            selection_result = self.feature_selector.select_features(
                features_df,
                target_column='target',
                max_features=self.config.max_features
            )
            
            if selection_result and selection_result.get('success', False):
                selected_features = selection_result.get('selected_features', [])
                feature_scores = selection_result.get('feature_scores', {})
                return selected_features, feature_scores
            else:
                tprint_warning("⚠️ Feature selection failed, returning all features")
                return list(features.keys()), {name: 1.0 for name in features.keys()}
            
        except Exception as e:
            tprint_error(f"❌ Feature selection failed: {e}")
            return list(features.keys()), {name: 1.0 for name in features.keys()}


# Export main classes
__all__ = [
    'CrossTimeframeFeatureOptimizationComponent',
    'CrossTimeframeOptimizationConfig',
    'CrossTimeframeOptimizationResult'
]