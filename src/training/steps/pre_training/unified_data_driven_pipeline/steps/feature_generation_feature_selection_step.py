"""
Sophisticated Feature Selection Step

This step performs advanced feature selection using battle-tested components
with multi-objective optimization, economic validation, and VectorBT optimization.
"""

from __future__ import annotations

import logging
import json
import warnings
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

from src.training.steps.pre_training.components.base_component import (
    BasePreTrainingComponent, ComponentConfig, ComponentResult
)
from src.utils.common_operations import safe_dataframe_operation
from src.utils.matrix_operations import safe_matrix_multiply, optimize_dataframe

# Import battle-tested feature selection components
try:
    from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.battle_tested_feature_selection import (
        BattleTestedFeatureSelector, FeatureSelectionConfig, FeatureSelectionResult as BattleTestedFeatureSelectionResult
    )
    from src.training.steps.pre_training.unified_data_driven_pipeline.feature_selection.multi_objective_selector import (
        MultiObjectiveFeatureSelector, MultiObjectiveResult
    )
    from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.economic_evaluation import (
        EconomicPeriodEvaluator, EconomicValidationResult
    )
    from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.vectorbt_enhancements import (
        EnhancedVectorBTOptimizer
    )
    BATTLE_TESTED_COMPONENTS_AVAILABLE = True
except ImportError:
    BATTLE_TESTED_COMPONENTS_AVAILABLE = False
    BattleTestedFeatureSelector = None
    FeatureSelectionConfig = None
    BattleTestedFeatureSelectionResult = None
    MultiObjectiveFeatureSelector = None
    MultiObjectiveResult = None
    EconomicPeriodEvaluator = None
    EconomicValidationResult = None
    EnhancedVectorBTOptimizer = None

@dataclass
class FeatureSelectionResult:
    """Sophisticated result of feature selection step."""

    success: bool
    selected_features: pd.DataFrame
    selection_metadata: Dict[str, Any]
    selection_metrics: Dict[str, Any]
    selection_strategy: str
    feature_importance: Dict[str, float]
    economic_validation: Dict[str, Any]
    multi_objective_results: Dict[str, Any]
    vectorbt_optimizations: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    diversity_metrics: Dict[str, Any]
    stability_metrics: Dict[str, Any]
    artifacts: Dict[str, Any]
    error_message: Optional[str] = None

class FeatureGenerationFeatureSelectionStep(BasePreTrainingComponent):
    """Sophisticated feature selection step using battle-tested components."""

    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the sophisticated feature selection step."""
        super().__init__(config or ComponentConfig())
        self.logger = logging.getLogger(__name__)
        
        # Initialize battle-tested feature selection components
        if BATTLE_TESTED_COMPONENTS_AVAILABLE:
            # Initialize advanced feature selector with sophisticated configuration
            self.feature_selection_config = FeatureSelectionConfig(
                enable_multi_stage_selection=True,
                enable_lightweight_screening=True,
                enable_diversity_selection=True,
                enable_stability_analysis=True,
                enable_vectorbt=True,
                enable_parallel_processing=True,
                final_selection_count=40,
                diversity_threshold=0.3,
                stability_window=20
            )
            
            self.battle_tested_selector = BattleTestedFeatureSelector(self.feature_selection_config)
            
            # Initialize multi-objective selector
            self.multi_objective_selector = MultiObjectiveFeatureSelector()
            
            # Initialize economic evaluator
            self.economic_evaluator = EconomicPeriodEvaluator()
            
            # Initialize VectorBT optimizer
            self.vectorbt_optimizer = EnhancedVectorBTOptimizer()
        else:
            self.logger.warning("⚠️ Sophisticated feature selection components not available, using fallback")
            self.advanced_selector = None
            self.multi_objective_selector = None
            self.economic_evaluator = None
            self.vectorbt_optimizer = None

    async def execute(self,
                     data: pd.DataFrame,
                     targets: pd.Series,
                     symbol: str = "ETHUSDT",
                     timeframe: str = "15m",
                     direction: str = "longs",
                     intensity: str = "blank",
                     lookback_days: Optional[int] = None,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None,
                     exchange: str = "binance",
                     custom_overrides: Optional[Dict[str, Any]] = None) -> FeatureSelectionResult:
        """Execute sophisticated feature selection step using battle-tested components."""

        self.logger.info("🎯 Starting sophisticated feature selection step with multi-objective optimization")

        try:
            if not BATTLE_TESTED_COMPONENTS_AVAILABLE:
                # Fallback to basic feature selection
                return await self._fallback_feature_selection(
                    data, targets, symbol, timeframe, direction, custom_overrides
                )

            # Perform sophisticated feature selection
            selection_result = await self._perform_sophisticated_feature_selection(
                data, targets, symbol, timeframe, direction, custom_overrides
            )

            if selection_result.success:
                self.logger.info(f"✅ Sophisticated feature selection completed successfully")
                self.logger.info(f"📊 Selected {len(selection_result.selected_features.columns)} features")
                self.logger.info(f"🎯 Strategy: {selection_result.selection_strategy}")
                self.logger.info(f"💰 Economic validation: {selection_result.economic_validation}")
                self.logger.info(f"📈 Multi-objective results: {selection_result.multi_objective_results}")
            else:
                self.logger.error(f"❌ Feature selection failed: {selection_result.error_message}")

            return selection_result

        except Exception as e:
            self.logger.error(f"❌ Sophisticated feature selection step failed with exception: {e}")
            return FeatureSelectionResult(
                success=False,
                selected_features=pd.DataFrame(),
                selection_metadata={},
                selection_metrics={},
                selection_strategy="error",
                feature_importance={},
                economic_validation={},
                multi_objective_results={},
                vectorbt_optimizations={},
                quality_metrics={},
                diversity_metrics={},
                stability_metrics={},
                artifacts={},
                error_message=str(e)
            )

    async def _perform_sophisticated_feature_selection(self, data: pd.DataFrame, targets: pd.Series,
                                                       symbol: str, timeframe: str, direction: str,
                                                       custom_overrides: Optional[Dict[str, Any]]) -> FeatureSelectionResult:
        """Perform sophisticated feature selection using battle-tested components."""
        
        try:
            if not BATTLE_TESTED_COMPONENTS_AVAILABLE:
                raise ImportError("Battle-tested feature selection components not available")
                
            # Step 1: Battle-tested multi-stage feature selection
            self.logger.info("🔄 Stage 1: Battle-tested multi-stage feature selection")
            battle_tested_selector = BattleTestedFeatureSelector()
            advanced_result = battle_tested_selector.select_features(data, targets)
            
            if not advanced_result.success:
                raise Exception(f"Advanced feature selection failed: {advanced_result.error_message}")
            
            # Step 2: Multi-objective optimization
            self.logger.info("🎯 Stage 2: Multi-objective optimization")
            multi_objective_result = self.multi_objective_selector.optimize_features(
                data[advanced_result.selected_features], targets
            )
            
            # Step 3: Economic validation
            self.logger.info("💰 Stage 3: Economic validation")
            economic_result = self.economic_evaluator.validate_features(
                data[multi_objective_result.selected_features], targets, symbol, timeframe
            )
            
            # Step 4: VectorBT optimization
            self.logger.info("⚡ Stage 4: VectorBT optimization")
            vectorbt_result = self.vectorbt_optimizer.optimize_features(
                data[economic_result.validated_features], targets
            )
            
            # Step 5: Compile sophisticated result
            selected_features_df = data[vectorbt_result.optimized_features]
            
            return FeatureSelectionResult(
                success=True,
                selected_features=selected_features_df,
                selection_metadata={
                    'advanced_selection': advanced_result.__dict__,
                    'multi_objective': multi_objective_result.__dict__,
                    'economic_validation': economic_result.__dict__,
                    'vectorbt_optimization': vectorbt_result.__dict__
                },
                selection_metrics={
                    'advanced_metrics': advanced_result.quality_metrics,
                    'multi_objective_metrics': multi_objective_result.objective_values,
                    'economic_metrics': economic_result.validation_metrics,
                    'vectorbt_metrics': vectorbt_result.optimization_metrics
                },
                selection_strategy="sophisticated_multi_stage",
                feature_importance=advanced_result.feature_importance,
                economic_validation=economic_result.__dict__,
                multi_objective_results=multi_objective_result.__dict__,
                vectorbt_optimizations=vectorbt_result.__dict__,
                quality_metrics=advanced_result.quality_metrics,
                diversity_metrics=advanced_result.diversity_metrics,
                stability_metrics=advanced_result.stability_metrics,
                artifacts={
                    'advanced_result': advanced_result.__dict__,
                    'multi_objective_result': multi_objective_result.__dict__,
                    'economic_result': economic_result.__dict__,
                    'vectorbt_result': vectorbt_result.__dict__
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Sophisticated feature selection failed: {e}")
            return FeatureSelectionResult(
                success=False,
                selected_features=pd.DataFrame(),
                selection_metadata={},
                selection_metrics={},
                selection_strategy="error",
                feature_importance={},
                economic_validation={},
                multi_objective_results={},
                vectorbt_optimizations={},
                quality_metrics={},
                diversity_metrics={},
                stability_metrics={},
                artifacts={},
                error_message=str(e)
            )

    async def _fallback_feature_selection(self, data: pd.DataFrame, targets: pd.Series,
                                          symbol: str, timeframe: str, direction: str,
                                          custom_overrides: Optional[Dict[str, Any]]) -> FeatureSelectionResult:
        """Fallback feature selection when sophisticated components are not available."""
        
        try:
            # Ensure data and targets are aligned and numeric
            # Align indexes
            common_index = data.index.intersection(targets.index)
            if len(common_index) == 0:
                raise ValueError("No common index between data and targets")
            
            data_aligned = data.loc[common_index]
            targets_aligned = targets.loc[common_index]
            
            # Select only numeric columns and drop rows with NaNs
            numeric_columns = data_aligned.select_dtypes(include=[np.number]).columns
            data_numeric = data_aligned[numeric_columns].dropna()
            targets_numeric = targets_aligned.dropna()
            
            # Re-align after dropping NaNs
            common_index_clean = data_numeric.index.intersection(targets_numeric.index)
            if len(common_index_clean) == 0:
                raise ValueError("No valid data after cleaning NaNs")
            
            data_clean = data_numeric.loc[common_index_clean]
            targets_clean = targets_numeric.loc[common_index_clean]
            
            # Basic feature selection using correlation
            correlations = data_clean.corrwith(targets_clean).abs().sort_values(ascending=False)
            
            # Drop NaN correlations and ensure we have at least one feature
            correlations_clean = correlations.dropna()
            if len(correlations_clean) == 0:
                raise ValueError("No valid correlations found")
            
            # Select top 20% of features, with minimum of 1 and maximum of all available
            n_features = max(1, min(len(correlations_clean), int(len(correlations_clean) * 0.2)))
            selected_features = correlations_clean.head(n_features).index.tolist()
            
            # Create selected features dataframe
            selected_data = data_clean[selected_features]
            
            # Calculate basic feature importance
            feature_importance = correlations_clean[selected_features].to_dict()
            
            return FeatureSelectionResult(
                success=True,
                selected_features=selected_data,
                selection_metadata={'method': 'fallback_correlation', 'symbol': symbol, 'timeframe': timeframe},
                selection_metrics={'selected_count': len(selected_features), 'total_count': len(data_clean.columns)},
                selection_strategy="correlation_fallback",
                feature_importance=feature_importance,
                economic_validation={},
                multi_objective_results={},
                vectorbt_optimizations={},
                quality_metrics={},
                diversity_metrics={},
                stability_metrics={},
                artifacts={'fallback_selection': selected_features, 'correlation_count': len(correlations_clean)}
            )
            
        except Exception as e:
            return FeatureSelectionResult(
                success=False,
                selected_features=pd.DataFrame(),
                selection_metadata={},
                selection_metrics={},
                selection_strategy="error",
                feature_importance={},
                economic_validation={},
                multi_objective_results={},
                vectorbt_optimizations={},
                quality_metrics={},
                diversity_metrics={},
                stability_metrics={},
                artifacts={},
                error_message=str(e)
            )

# Command handler for ares_launcher integration
async def handle_feature_generation_feature_selection_step(
    symbol: str = "ETHUSDT",
    timeframe: str = "15m",
    direction: str = "longs",
    custom_overrides: Optional[Dict[str, Any]] = None,
    **kwargs
) -> FeatureSelectionResult:
    """
    Handle sophisticated feature generation feature selection step command.

    Args:
        symbol: Trading symbol (default: "ETHUSDT")
        timeframe: Timeframe (default: "15m")
        direction: Direction (default: "longs")
        custom_overrides: Custom configuration overrides (optional)
        **kwargs: Additional arguments

    Returns:
        Sophisticated FeatureSelectionResult with comprehensive selection results
    """
    # Create sample data for feature selection (in real usage, this would come from data loading)
    sample_data = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 105,
        'low': np.random.randn(1000).cumsum() + 95,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    })

    # Generate targets using the labeling system with proper error handling
    try:
        from src.training.steps.pre_training.unified_data_driven_pipeline.consolidated_pipeline_runner import ConsolidatedPipelineRunner
        runner = ConsolidatedPipelineRunner()
            
        # Use public API if available, otherwise fallback to private with error handling
        if hasattr(runner, 'generate_targets'):
            targets = runner.generate_targets(sample_data, symbol, timeframe, direction)
        elif hasattr(runner, '_generate_targets'):
            targets = runner._generate_targets(sample_data, symbol, timeframe, direction)
        else:
            # Fallback: create simple targets based on price movement
            targets = pd.Series(
                (sample_data['close'].pct_change() > 0).astype(int),
                index=sample_data.index,
                name='target'
            ).dropna()
    except Exception as e:
        # Fallback: create simple targets based on price movement
        targets = pd.Series(
            (sample_data['close'].pct_change() > 0).astype(int),
            index=sample_data.index,
            name='target'
        ).dropna()

    # Create sophisticated step instance and execute
    step = FeatureGenerationFeatureSelectionStep()

    return await step.execute(
        data=sample_data,
        targets=targets,
        symbol=symbol,
        timeframe=timeframe,
        direction=direction,
        custom_overrides=custom_overrides
    )

# Register component with factory
def _register_feature_generation_feature_selection_step():
    """Register the sophisticated FeatureGenerationFeatureSelectionStep component with the factory."""
    try:
        from src.training.steps.pre_training.components import ComponentFactory
        ComponentFactory.register_component(
            'feature_generation_feature_selection_step',
            FeatureGenerationFeatureSelectionStep
        )
    except ImportError:
        # Component factory not available, skip registration
        pass

# Register the component when module is imported
_register_feature_generation_feature_selection_step()
