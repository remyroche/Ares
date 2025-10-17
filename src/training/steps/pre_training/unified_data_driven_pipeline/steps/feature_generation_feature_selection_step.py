"""
Sophisticated Feature Selection Step

This step performs advanced feature selection using battle-tested components
with multi-objective optimization, economic validation, and VectorBT optimization.
"""

from __future__ import annotations

import asyncio
import logging
import json
import warnings
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

from src.training.steps.pre_training.unified_data_driven_pipeline.core.modular_architecture import (
    ModularComponent
)

# Import tprint utilities
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

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

def _cols(obj: Any) -> List[str]:
    """Normalize selected_features to column names list."""
    if obj is None:
        return []
    if isinstance(obj, pd.DataFrame):
        return list(obj.columns)
    if hasattr(obj, "tolist"):
        return list(obj.tolist())
    if isinstance(obj, list):
        # Handle list of FeatureScore objects
        if obj and hasattr(obj[0], 'name'):
            return [item.name for item in obj if hasattr(item, 'name')]
        # Handle list of strings
        return list(obj)
    return list(obj)

def _safe_to_meta(obj: Any) -> Dict[str, Any]:
    """Safely convert object to serializable metadata."""
    if obj is None:
        return {}
    # Prefer a method if available
    for attr in ("to_dict", "model_dump", "dict"):
        if hasattr(obj, attr) and callable(getattr(obj, attr)):
            try:
                return getattr(obj, attr)()
            except Exception:
                pass
    # Fallback: shallow, serializable subset
    out = {}
    for k, v in getattr(obj, "__dict__", {}).items():
        if isinstance(v, (str, int, float, bool, type(None))):
            out[k] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (str, int, float, bool, type(None))) for x in v):
            out[k] = list(v)
        elif isinstance(v, dict) and all(isinstance(x, (str, int, float, bool, type(None))) for x in v.values()):
            out[k] = v
    return out

@dataclass
class FeatureGenerationFeatureSelectionStep(ModularComponent):
    """Sophisticated feature selection step using battle-tested components."""

    def __init__(self, name: str = "step", config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        """Initialize the sophisticated feature selection step."""
        super().__init__(name, config or {}, logger)
            name="feature_generation_feature_selection_step",
            config=config_dict,
            logger=logging.getLogger(__name__)
        )
        
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

    def _apply_overrides(self, overrides: Optional[Dict[str, Any]]):
        """Apply custom configuration overrides."""
        if not overrides or not BATTLE_TESTED_COMPONENTS_AVAILABLE:
            return
        for k, v in overrides.items():
            if hasattr(self.feature_selection_config, k):
                setattr(self.feature_selection_config, k, v)

    def _filter_data_by_parameters(self, data: pd.DataFrame, targets: pd.Series, 
                                  lookback_days: Optional[int], start_date: Optional[str], 
                                  end_date: Optional[str]) -> Tuple[pd.DataFrame, pd.Series]:
        """Filter data based on lookback_days, start_date, and end_date parameters."""
        if lookback_days is not None:
            # Use last N days of data
            data = data.tail(lookback_days)
            targets = targets.tail(lookback_days)
        
        if start_date is not None:
            try:
                start_dt = pd.to_datetime(start_date)
                data = data[data.index >= start_dt]
                targets = targets[targets.index >= start_dt]
            except Exception as e:
                self.logger.warning(f"Invalid start_date format: {e}")
        
        if end_date is not None:
            try:
                end_dt = pd.to_datetime(end_date)
                data = data[data.index <= end_dt]
                targets = targets[targets.index <= end_dt]
            except Exception as e:
                self.logger.warning(f"Invalid end_date format: {e}")
        
        return data, targets

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
            # Apply data filtering based on parameters
            data, targets = self._filter_data_by_parameters(data, targets, lookback_days, start_date, end_date)
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
            self.logger.error(f"❌ Sophisticated feature selection step failed with exception: {e}", exc_info=True)
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
            
            # Apply custom overrides if provided
            self._apply_overrides(custom_overrides)
                
            # Step 1: Battle-tested multi-stage feature selection
            self.logger.info("🔄 Stage 1: Battle-tested multi-stage feature selection")
            tprint_info("🎯 Starting Stage 1: Battle-tested multi-stage feature selection")
            tprint_debug(f"📊 Input data shape: {data.shape}")
            tprint_debug(f"📊 Target data shape: {targets.shape if targets is not None else 'None'}")
            
            advanced_result = await asyncio.to_thread(
                self.battle_tested_selector.select_features, data, targets
            )
            
            tprint_debug(f"📊 Stage 1 result: success={advanced_result.success}")
            if hasattr(advanced_result, 'selected_features'):
                tprint_debug(f"📊 Selected features count: {len(advanced_result.selected_features)}")
            
            if not advanced_result.success:
                tprint_error(f"❌ Stage 1 failed: {advanced_result.error_message}")
                raise Exception(f"Advanced feature selection failed: {advanced_result.error_message}")
            
            # Normalize selected features to column names
            cols1 = _cols(advanced_result.selected_features)
            df1 = data[cols1].copy()
            tprint_success(f"✅ Stage 1 completed: {len(cols1)} features selected")
            tprint_debug(f"📊 Selected features: {cols1}")
            
            # Step 2: Multi-objective optimization
            self.logger.info("🎯 Stage 2: Multi-objective optimization")
            tprint_info("🎯 Starting Stage 2: Multi-objective optimization")
            tprint_debug(f"📊 Input data shape: {df1.shape}")
            
            multi_objective_result = await asyncio.to_thread(
                self.multi_objective_selector.optimize_features, df1, targets
            )
            
            tprint_debug(f"📊 Stage 2 result: success={multi_objective_result.is_valid}")
            if hasattr(multi_objective_result, 'selected_features'):
                tprint_debug(f"📊 Selected features count: {len(multi_objective_result.selected_features)}")
            
            # Normalize multi-objective selected features
            cols2 = _cols(multi_objective_result.selected_features)
            df2 = df1[cols2].copy()
            tprint_success(f"✅ Stage 2 completed: {len(cols2)} features selected")
            tprint_debug(f"📊 Selected features: {cols2}")
            
            # Step 3: Economic validation
            self.logger.info("💰 Stage 3: Economic validation")
            tprint_info("💰 Starting Stage 3: Economic validation")
            tprint_debug(f"📊 Input data shape: {df2.shape}")
            tprint_debug(f"📊 Symbol: {symbol}, Timeframe: {timeframe}")
            
            economic_result = await asyncio.to_thread(
                self.economic_evaluator.validate_features, df2, targets, symbol, timeframe
            )
            
            tprint_debug(f"📊 Stage 3 result: success={economic_result.success}")
            if hasattr(economic_result, 'validated_features'):
                tprint_debug(f"📊 Validated features shape: {economic_result.validated_features.shape}")
            
            # Normalize economic validated features
            cols3 = _cols(economic_result.validated_features)
            df3 = df2[cols3].copy()
            tprint_success(f"✅ Stage 3 completed: {len(cols3)} features validated")
            tprint_debug(f"📊 Validated features: {cols3}")
            
            # Step 4: VectorBT optimization
            self.logger.info("⚡ Stage 4: VectorBT optimization")
            tprint_info("⚡ Starting Stage 4: VectorBT optimization")
            tprint_debug(f"📊 Input data shape: {df3.shape}")
            
            vectorbt_result = await asyncio.to_thread(
                self.vectorbt_optimizer.optimize_features, df3, targets
            )
            
            tprint_debug(f"📊 Stage 4 result: success={vectorbt_result.success}")
            if hasattr(vectorbt_result, 'optimized_features'):
                tprint_debug(f"📊 Optimized features shape: {vectorbt_result.optimized_features.shape}")
            
            # Normalize vectorbt optimized features
            cols4 = _cols(vectorbt_result.optimized_features)
            selected_features_df = df3[cols4].copy()
            tprint_success(f"✅ Stage 4 completed: {len(cols4)} features optimized")
            tprint_debug(f"📊 Final optimized features: {cols4}")
            
            # Final summary
            tprint_success("🎉 Feature selection pipeline completed successfully!")
            tprint_info(f"📊 Pipeline Summary:")
            tprint_info(f"   • Original features: {len(data.columns)}")
            tprint_info(f"   • Battle-tested features: {len(cols1)}")
            tprint_info(f"   • Multi-objective features: {len(cols2)}")
            tprint_info(f"   • Economic validated features: {len(cols3)}")
            tprint_info(f"   • VectorBT optimized features: {len(cols4)}")
            tprint_info(f"   • Final selected features: {len(selected_features_df.columns)}")
            tprint_info(f"   • Feature reduction: {len(data.columns) - len(selected_features_df.columns)} features removed")
            tprint_info(f"   • Reduction percentage: {((len(data.columns) - len(selected_features_df.columns)) / len(data.columns) * 100):.1f}%")
            
            return FeatureSelectionResult(
                success=True,
                selected_features=selected_features_df,
                selection_metadata={
                    'advanced_selection': _safe_to_meta(advanced_result),
                    'multi_objective': _safe_to_meta(multi_objective_result),
                    'economic_validation': _safe_to_meta(economic_result),
                    'vectorbt_optimization': _safe_to_meta(vectorbt_result)
                },
                selection_metrics={
                    'advanced_metrics': getattr(advanced_result, 'quality_metrics', {}),
                    'multi_objective_metrics': getattr(multi_objective_result, 'objective_values', {}),
                    'economic_metrics': getattr(economic_result, 'validation_metrics', {}),
                    'vectorbt_metrics': getattr(vectorbt_result, 'optimization_metrics', {})
                },
                selection_strategy="sophisticated_multi_stage",
                feature_importance=getattr(advanced_result, 'feature_importance', {}),
                economic_validation=_safe_to_meta(economic_result),
                multi_objective_results=_safe_to_meta(multi_objective_result),
                vectorbt_optimizations=_safe_to_meta(vectorbt_result),
                quality_metrics=getattr(advanced_result, 'quality_metrics', {}),
                diversity_metrics=getattr(advanced_result, 'diversity_metrics', {}),
                stability_metrics=getattr(advanced_result, 'stability_metrics', {}),
                artifacts={
                    'advanced_result': _safe_to_meta(advanced_result),
                    'multi_objective_result': _safe_to_meta(multi_objective_result),
                    'economic_result': _safe_to_meta(economic_result),
                    'vectorbt_result': _safe_to_meta(vectorbt_result)
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Sophisticated feature selection failed: {e}", exc_info=True)
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
            # Align data and targets first, then drop NaNs together
            df = pd.concat([data, targets.rename("target")], axis=1).dropna()
            if len(df) == 0:
                raise ValueError("No valid data after alignment and NaN removal")
            
            # Select only numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if "target" not in numeric_columns:
                raise ValueError("Target column is not numeric")
            
            # Remove target from feature columns
            feature_columns = [col for col in numeric_columns if col != "target"]
            if len(feature_columns) == 0:
                raise ValueError("No numeric feature columns found")
            
            X = df[feature_columns]
            y = df["target"]
            
            # Time-shift features to prevent target leakage
            X_shifted = X.shift(1).dropna()
            y_aligned = y.loc[X_shifted.index]
            
            if len(X_shifted) == 0:
                raise ValueError("No valid data after time-shifting")
            
            # Basic feature selection using correlation (on training period only)
            correlations = X_shifted.corrwith(y_aligned).abs().sort_values(ascending=False)
            
            # Drop NaN correlations and ensure we have at least one feature
            correlations_clean = correlations.dropna()
            if len(correlations_clean) == 0:
                raise ValueError("No valid correlations found")
            
            # Select top 20% of features, with minimum of 1 and maximum of all available
            n_features = max(1, min(len(correlations_clean), int(len(correlations_clean) * 0.2)))
            selected_features = correlations_clean.head(n_features).index.tolist()
            
            # Create selected features dataframe using original data (not shifted)
            selected_data = data[selected_features]
            
            # Calculate basic feature importance
            feature_importance = correlations_clean[selected_features].to_dict()
            
            return FeatureSelectionResult(
                success=True,
                selected_features=selected_data,
                selection_metadata={'method': 'fallback_correlation', 'symbol': symbol, 'timeframe': timeframe},
                selection_metrics={'selected_count': len(selected_features), 'total_count': len(data.columns)},
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
            self.logger.error(f"❌ Fallback feature selection failed: {e}", exc_info=True)
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

    # Required abstract methods from ModularComponent
    def _initialize_resources(self) -> bool:
        """Initialize component-specific resources."""
        try:
            # Initialize battle-tested feature selection components
            if BATTLE_TESTED_COMPONENTS_AVAILABLE:
                self.battle_tested_selector = BattleTestedFeatureSelector()
                self.multi_objective_selector = MultiObjectiveFeatureSelector()
                self.economic_evaluator = EconomicPeriodEvaluator()
                self.vectorbt_optimizer = EnhancedVectorBTOptimizer()
            else:
                self.battle_tested_selector = None
                self.multi_objective_selector = None
                self.economic_evaluator = None
                self.vectorbt_optimizer = None
            
            # Set initial state
            self.set_state('initialized_at', datetime.now().isoformat())
            self.set_state('selection_count', 0)
            return True
        except Exception as e:
            self.logger.error(f"Resource initialization failed: {e}")
            return False
    
    def _cleanup_resources(self) -> None:
        """Cleanup component-specific resources."""
        self.set_state('cleaned_up_at', datetime.now().isoformat())
        self.set_state('selection_count', 0)
    
    def _process_data(self, data: Any, **kwargs) -> Any:
        """Process data with component logic."""
        # Increment selection count
        count = self.get_state('selection_count', 0)
        self.set_state('selection_count', count + 1)
        
        # Basic processing - return data as-is for now
        # The actual feature selection is done in the execute method
        return data
    
    def _get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for this component."""
        return {
            'min_size': 100,
            'max_size': 1000000,
            'required_attributes': ['open', 'high', 'low', 'close', 'volume'],
            'data_types': ['pandas.DataFrame'],
            'max_nan_ratio': 0.1,
            'min_unique_values': 2
        }
    
    def _validate_component_specific(self, data: Any) -> Dict[str, Any]:
        """Validate data with component-specific rules."""
        errors = []
        warnings = []
        metadata = {}
        
        if isinstance(data, pd.DataFrame):
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                errors.append(f"Missing required columns: {missing_columns}")
            
            # Check data size
            if len(data) < 100:
                warnings.append("Data size is small (< 100 rows)")
            
            # Check for NaN values
            nan_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            if nan_ratio > 0.1:
                warnings.append(f"High NaN ratio: {nan_ratio:.2%}")
            
            metadata['data_shape'] = data.shape
            metadata['nan_ratio'] = nan_ratio
            metadata['columns'] = list(data.columns)
        
        return {'errors': errors, 'warnings': warnings, 'metadata': metadata}

# Command handler for ares_launcher integration
