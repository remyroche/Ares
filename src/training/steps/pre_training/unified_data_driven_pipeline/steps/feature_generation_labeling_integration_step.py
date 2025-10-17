"""
Feature Generation Labeling Integration Step

This step performs labeling integration as part of the unified data-driven pipeline
by calling the consolidated pipeline at the appropriate stage.
"""

from __future__ import annotations

import warnings
import logging
import json
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

from src.training.steps.pre_training.unified_data_driven_pipeline.consolidated_pipeline_runner import (
    run_labeling_integration_step
)
from src.training.steps.pre_training.components.base_component import (
    BasePreTrainingComponent, ComponentConfig, ComponentResult
)
from src.utils.common_operations import safe_dataframe_operation
from src.utils.matrix_operations import safe_matrix_multiply, optimize_dataframe

# Import tprint utilities for enhanced logging
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug,
        tprint_performance, tprint_step, tprint_result
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
    def tprint_performance(*args, **kwargs): print("PERFORMANCE:", *args, **kwargs)
    def tprint_step(*args, **kwargs): print("STEP:", *args, **kwargs)
    def tprint_result(*args, **kwargs): print("RESULT:", *args, **kwargs)

@dataclass
class LabelingIntegrationResult:
    """Result of labeling integration step."""

    success: bool
    integrated_labels: int
    integration_metadata: Dict[str, Any]
    artifacts: Dict[str, Any]
    error_message: Optional[str] = None

class FeatureGenerationLabelingIntegrationStep(BasePreTrainingComponent):
    """Labeling integration step that calls the consolidated pipeline."""

    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the labeling integration step."""
        super().__init__(config or ComponentConfig())
        self.logger = logging.getLogger(__name__)

    async def execute(self,
                     training_input: Optional[Dict[str, Any]] = None,
                     pipeline_state: Optional[Dict[str, Any]] = None,
                     data: Optional[Any] = None,
                     **kwargs) -> ComponentResult:
        """Execute labeling integration step using consolidated pipeline."""

        self.logger.info("🔗 Starting labeling integration step using consolidated pipeline")

        # Extract parameters from training_input or kwargs
        if training_input is None:
            # Extract from kwargs (called from component factory)
            if data is None:
                data = kwargs.get('data')
            symbol = kwargs.get('symbol', 'ETHUSDT')
            timeframe = kwargs.get('timeframe', '15m')
            direction = kwargs.get('direction', 'longs')
            intensity = kwargs.get('intensity', 'blank')
            lookback_days = kwargs.get('lookback_days')
            start_date = kwargs.get('start_date')
            end_date = kwargs.get('end_date')
            exchange = kwargs.get('exchange', 'binance')
            custom_overrides = kwargs.get('custom_overrides')
            # Create training_input dict for compatibility
            training_input = {
                'data': data,
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'intensity': intensity,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'custom_overrides': custom_overrides
            }
        else:
            # Extract from training_input (called from pipeline)
            data = training_input.get('data')
            symbol = training_input.get('symbol', 'ETHUSDT')
            timeframe = training_input.get('timeframe', '15m')
            direction = training_input.get('direction', 'longs')
            intensity = training_input.get('intensity', 'blank')
            lookback_days = training_input.get('lookback_days')
            start_date = training_input.get('start_date')
            end_date = training_input.get('end_date')
            exchange = training_input.get('exchange', 'binance')
            custom_overrides = training_input.get('custom_overrides')

        try:
            # Validate financial logic requirements before calling runner
            if not self._validate_financial_logic_requirements(timeframe, direction, custom_overrides):
                return ComponentResult(
                    success=False,
                    metadata={'artifacts': {}},
                    error_message='Financial logic validation failed'
                )
            
            # Call the consolidated pipeline runner
            result = await run_labeling_integration_step(
                data,
                symbol=symbol,
                timeframe=timeframe,
                direction=direction,
                intensity=intensity,
                lookback_days=lookback_days,
                start_date=start_date,
                end_date=end_date,
                exchange=exchange,
                custom_overrides=custom_overrides
            )

            # Normalize and validate runner result schema
            normalized_result = self._normalize_runner_result(result)
            
            # Convert result to ComponentResult
            component_result = ComponentResult(
                success=normalized_result['success'],
                metadata={
                    'integrated_labels': normalized_result.get('integrated_labels', 0),
                    'integration_metadata': normalized_result.get('integration_metadata', {}),
                    'artifacts': normalized_result.get('artifacts', {}),
                    **normalized_result.get('metadata', {})
                },
                error_message=normalized_result.get('error_message')
            )
            
            # Validate financial telemetry after runner execution
            if component_result.success:
                if not self._validate_financial_telemetry(component_result.metadata):
                    component_result.success = False
                    component_result.error_message = 'Financial telemetry validation failed'

            if component_result.success:
                self.logger.info(f"✅ Labeling integration completed successfully with {component_result.metadata.get('integrated_labels', 0)} integrated labels")
            else:
                self.logger.error(f"❌ Labeling integration failed: {component_result.error_message}")

            return component_result

        except Exception as e:
            self.logger.error(f"❌ Labeling integration step failed with exception: {e}")
            return ComponentResult(
                success=False,
                metadata={'artifacts': {}},
                error_message=str(e)
            )

    # Required abstract methods from BasePreTrainingComponent
    def process(self, data: Any) -> Any:
        """Process the input data and return the result with enhanced validation."""
        try:
            if TPRINT_AVAILABLE:
                tprint_info("🔗 Processing labeling integration step")
            else:
                print("INFO: Processing labeling integration step")
            
            # Convert data to DataFrame if needed
            if not isinstance(data, pd.DataFrame):
                if TPRINT_AVAILABLE:
                    tprint_warning("⚠️ Input data is not a DataFrame, attempting conversion")
                else:
                    print("WARNING: Input data is not a DataFrame, attempting conversion")
                data = pd.DataFrame(data)
            
            # Enhanced validation using the strict financial validation
            if not self.validate(data):
                if TPRINT_AVAILABLE:
                    tprint_error("❌ Data validation failed during processing")
                else:
                    print("ERROR: Data validation failed during processing")
                return None
            
            if TPRINT_AVAILABLE:
                tprint_success(f"✅ Processed {len(data)} rows for labeling integration")
            else:
                print(f"SUCCESS: Processed {len(data)} rows for labeling integration")
            
            return data
            
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Error processing labeling integration data: {e}")
            else:
                print(f"ERROR: Error processing labeling integration data: {e}")
            return None

    def validate(self, data: Any) -> bool:
        """Validate the input data for labeling integration with strict financial data requirements."""
        try:
            if data is None:
                if TPRINT_AVAILABLE:
                    tprint_error("❌ Validation failed: Data is None")
                else:
                    print("ERROR: Validation failed: Data is None")
                return False
            
            if not isinstance(data, pd.DataFrame):
                if TPRINT_AVAILABLE:
                    tprint_error("❌ Validation failed: Data is not a DataFrame")
                else:
                    print("ERROR: Validation failed: Data is not a DataFrame")
                return False
            
            if len(data) == 0:
                if TPRINT_AVAILABLE:
                    tprint_error("❌ Validation failed: Data is empty")
                else:
                    print("ERROR: Validation failed: Data is empty")
                return False
            
            # Check for required columns for labeling
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                if TPRINT_AVAILABLE:
                    tprint_error(f"❌ Validation failed: Missing required columns: {missing_columns}")
                else:
                    print(f"ERROR: Validation failed: Missing required columns: {missing_columns}")
                return False
            
            # Strict financial data validation
            if not self._validate_financial_data_structure(data):
                return False
            
            if TPRINT_AVAILABLE:
                tprint_success(f"✅ Validation passed: Data has {len(data)} rows with required columns")
            else:
                print(f"SUCCESS: Validation passed: Data has {len(data)} rows with required columns")
            
            return True
            
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Validation error: {e}")
            else:
                print(f"ERROR: Validation error: {e}")
            return False
    
    def _validate_financial_data_structure(self, data: pd.DataFrame) -> bool:
        """Validate financial data structure with strict requirements for 6-bar window logic."""
        try:
            # 1. Validate DatetimeIndex
            if not isinstance(data.index, pd.DatetimeIndex):
                if TPRINT_AVAILABLE:
                    tprint_error("❌ Financial validation failed: Index must be DatetimeIndex")
                else:
                    print("ERROR: Financial validation failed: Index must be DatetimeIndex")
                return False
            
            # 2. Validate monotonic increasing index
            if not data.index.is_monotonic_increasing:
                if TPRINT_AVAILABLE:
                    tprint_error("❌ Financial validation failed: Index must be monotonic increasing")
                else:
                    print("ERROR: Financial validation failed: Index must be monotonic increasing")
                return False
            
            # 3. Validate no duplicate timestamps
            if data.index.duplicated().any():
                if TPRINT_AVAILABLE:
                    tprint_error("❌ Financial validation failed: Index contains duplicate timestamps")
                else:
                    print("ERROR: Financial validation failed: Index contains duplicate timestamps")
                return False
            
            # 4. Validate 15-minute frequency
            if len(data) > 1:
                time_diffs = data.index.to_series().diff().dropna()
                expected_freq = pd.Timedelta(minutes=15)
                # Allow small tolerance for floating point precision
                tolerance = pd.Timedelta(seconds=1)
                if not all(abs(td - expected_freq) <= tolerance for td in time_diffs):
                    if TPRINT_AVAILABLE:
                        tprint_error("❌ Financial validation failed: Data must have 15-minute frequency")
                    else:
                        print("ERROR: Financial validation failed: Data must have 15-minute frequency")
                    return False
            
            # 5. Validate OHLCV data types and values
            ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in ohlcv_columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    if TPRINT_AVAILABLE:
                        tprint_error(f"❌ Financial validation failed: Column '{col}' must be numeric")
                    else:
                        print(f"ERROR: Financial validation failed: Column '{col}' must be numeric")
                    return False
                
                # Check for finite values
                if not np.isfinite(data[col]).all():
                    if TPRINT_AVAILABLE:
                        tprint_error(f"❌ Financial validation failed: Column '{col}' contains non-finite values")
                    else:
                        print(f"ERROR: Financial validation failed: Column '{col}' contains non-finite values")
                    return False
                
                # Check for positive values (except volume can be zero)
                if col != 'volume' and (data[col] <= 0).any():
                    if TPRINT_AVAILABLE:
                        tprint_error(f"❌ Financial validation failed: Column '{col}' contains non-positive values")
                    else:
                        print(f"ERROR: Financial validation failed: Column '{col}' contains non-positive values")
                    return False
                
                if col == 'volume' and (data[col] < 0).any():
                    if TPRINT_AVAILABLE:
                        tprint_error(f"❌ Financial validation failed: Column '{col}' contains negative values")
                    else:
                        print(f"ERROR: Financial validation failed: Column '{col}' contains negative values")
                    return False
            
            # 6. Validate OHLC relationships
            if not self._validate_ohlc_relationships(data):
                return False
            
            # 7. Validate sufficient data for 6-bar windows
            if len(data) < 6:
                if TPRINT_AVAILABLE:
                    tprint_error("❌ Financial validation failed: Need at least 6 bars for window analysis")
                else:
                    print("ERROR: Financial validation failed: Need at least 6 bars for window analysis")
                return False
            
            if TPRINT_AVAILABLE:
                tprint_success("✅ Financial data structure validation passed")
            else:
                print("SUCCESS: Financial data structure validation passed")
            
            return True
            
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Financial validation error: {e}")
            else:
                print(f"ERROR: Financial validation error: {e}")
            return False
    
    def _validate_ohlc_relationships(self, data: pd.DataFrame) -> bool:
        """Validate OHLC price relationships."""
        try:
            # High >= max(open, close)
            high_valid = (data['high'] >= np.maximum(data['open'], data['close'])).all()
            if not high_valid:
                if TPRINT_AVAILABLE:
                    tprint_error("❌ OHLC validation failed: High must be >= max(open, close)")
                else:
                    print("ERROR: OHLC validation failed: High must be >= max(open, close)")
                return False
            
            # Low <= min(open, close)
            low_valid = (data['low'] <= np.minimum(data['open'], data['close'])).all()
            if not low_valid:
                if TPRINT_AVAILABLE:
                    tprint_error("❌ OHLC validation failed: Low must be <= min(open, close)")
                else:
                    print("ERROR: OHLC validation failed: Low must be <= min(open, close)")
                return False
            
            return True
            
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ OHLC validation error: {e}")
            else:
                print(f"ERROR: OHLC validation error: {e}")
            return False

    # Required utility methods for BasePreTrainingComponent
    def safe_dataframe_operation(self, df, operation_func, *args, **kwargs):
        """Safe dataframe operation wrapper."""
        return safe_dataframe_operation(df, operation_func, *args, **kwargs)

    def safe_matrix_multiply(self, a, b):
        """Safe matrix multiplication."""
        return safe_matrix_multiply(a, b)

    def optimize_dataframe_for_matrix_ops(self, df):
        """Optimize dataframe for matrix operations."""
        return optimize_dataframe(df)
    
    def _validate_financial_logic_requirements(self, timeframe: str, direction: str, custom_overrides: Optional[Dict[str, Any]]) -> bool:
        """Validate financial logic requirements for 6-bar window analysis."""
        try:
            # 1. Validate timeframe is 15m for 6-bar window logic
            if timeframe != '15m':
                if TPRINT_AVAILABLE:
                    tprint_error(f"❌ Financial logic validation failed: Timeframe must be '15m' for 6-bar analysis, got '{timeframe}'")
                else:
                    print(f"ERROR: Financial logic validation failed: Timeframe must be '15m' for 6-bar analysis, got '{timeframe}'")
                return False
            
            # 2. Validate direction is allowed
            allowed_directions = {'longs', 'shorts', 'both'}
            if direction not in allowed_directions:
                if TPRINT_AVAILABLE:
                    tprint_error(f"❌ Financial logic validation failed: Direction must be one of {allowed_directions}, got '{direction}'")
                else:
                    print(f"ERROR: Financial logic validation failed: Direction must be one of {allowed_directions}, got '{direction}'")
                return False
            
            # 3. Validate custom overrides don't break 6-bar invariant
            if custom_overrides:
                if not self._validate_custom_overrides(custom_overrides):
                    return False
            
            if TPRINT_AVAILABLE:
                tprint_success("✅ Financial logic requirements validation passed")
            else:
                print("SUCCESS: Financial logic requirements validation passed")
            
            return True
            
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Financial logic validation error: {e}")
            else:
                print(f"ERROR: Financial logic validation error: {e}")
            return False
    
    def _validate_custom_overrides(self, custom_overrides: Dict[str, Any]) -> bool:
        """Validate custom overrides don't break 6-bar window invariant."""
        try:
            # Check for horizon_minutes that would break 6-bar logic
            if 'horizon_minutes' in custom_overrides:
                horizon_minutes = custom_overrides['horizon_minutes']
                if horizon_minutes != 90:  # 6 * 15 minutes
                    if TPRINT_AVAILABLE:
                        tprint_warning(f"⚠️ Custom horizon_minutes={horizon_minutes} may break 6-bar window logic (expected 90)")
                    else:
                        print(f"WARNING: Custom horizon_minutes={horizon_minutes} may break 6-bar window logic (expected 90)")
            
            # Check for evaluation_bars that would break 6-bar logic
            if 'evaluation_bars' in custom_overrides:
                evaluation_bars = custom_overrides['evaluation_bars']
                if evaluation_bars != 6:
                    if TPRINT_AVAILABLE:
                        tprint_warning(f"⚠️ Custom evaluation_bars={evaluation_bars} may break 6-bar window logic (expected 6)")
                    else:
                        print(f"WARNING: Custom evaluation_bars={evaluation_bars} may break 6-bar window logic (expected 6)")
            
            # Check for entry_price_model
            if 'entry_price_model' in custom_overrides:
                entry_price_model = custom_overrides['entry_price_model']
                allowed_models = {'next_bar_open', 'current_bar_close', 'next_bar_close'}
                if entry_price_model not in allowed_models:
                    if TPRINT_AVAILABLE:
                        tprint_error(f"❌ Invalid entry_price_model: {entry_price_model}. Must be one of {allowed_models}")
                    else:
                        print(f"ERROR: Invalid entry_price_model: {entry_price_model}. Must be one of {allowed_models}")
                    return False
            
            return True
            
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Custom overrides validation error: {e}")
            else:
                print(f"ERROR: Custom overrides validation error: {e}")
            return False
    
    def _normalize_runner_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize runner result with schema validation and .get() guards."""
        try:
            # Expected schema from runner
            expected_keys = {
                'success', 'integrated_labels', 'integration_metadata', 
                'artifacts', 'metadata', 'error_message'
            }
            
            # Check for missing keys and log differences
            actual_keys = set(result.keys())
            missing_keys = expected_keys - actual_keys
            unexpected_keys = actual_keys - expected_keys
            
            if missing_keys:
                if TPRINT_AVAILABLE:
                    tprint_warning(f"⚠️ Runner result missing keys: {missing_keys}")
                else:
                    print(f"WARNING: Runner result missing keys: {missing_keys}")
            
            if unexpected_keys:
                if TPRINT_AVAILABLE:
                    tprint_warning(f"⚠️ Runner result unexpected keys: {unexpected_keys}")
                else:
                    print(f"WARNING: Runner result unexpected keys: {unexpected_keys}")
            
            # Normalize with safe defaults
            normalized = {
                'success': result.get('success', False),
                'integrated_labels': result.get('integrated_labels', 0),
                'integration_metadata': result.get('integration_metadata', {}),
                'artifacts': result.get('artifacts', {}),
                'metadata': result.get('metadata', {}),
                'error_message': result.get('error_message')
            }
            
            return normalized
            
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Result normalization error: {e}")
            else:
                print(f"ERROR: Result normalization error: {e}")
            # Return safe defaults on error
            return {
                'success': False,
                'integrated_labels': 0,
                'integration_metadata': {},
                'artifacts': {},
                'metadata': {},
                'error_message': f'Result normalization failed: {e}'
            }
    
    def _validate_financial_telemetry(self, metadata: Dict[str, Any]) -> bool:
        """Validate financial telemetry from runner results."""
        try:
            integration_metadata = metadata.get('integration_metadata', {})
            
            # 1. Validate window_bars = 6
            window_bars = integration_metadata.get('window_bars')
            if window_bars is not None and window_bars != 6:
                if TPRINT_AVAILABLE:
                    tprint_error(f"❌ Telemetry validation failed: window_bars must be 6, got {window_bars}")
                else:
                    print(f"ERROR: Telemetry validation failed: window_bars must be 6, got {window_bars}")
                return False
            
            # 2. Validate excludes_current_bar = True
            excludes_current_bar = integration_metadata.get('excludes_current_bar')
            if excludes_current_bar is not None and excludes_current_bar is not True:
                if TPRINT_AVAILABLE:
                    tprint_error(f"❌ Telemetry validation failed: excludes_current_bar must be True, got {excludes_current_bar}")
                else:
                    print(f"ERROR: Telemetry validation failed: excludes_current_bar must be True, got {excludes_current_bar}")
                return False
            
            # 3. Validate entry_price_model
            entry_price_model = integration_metadata.get('entry_price_model')
            if entry_price_model is not None:
                allowed_models = {'next_bar_open', 'current_bar_close', 'next_bar_close'}
                if entry_price_model not in allowed_models:
                    if TPRINT_AVAILABLE:
                        tprint_error(f"❌ Telemetry validation failed: entry_price_model must be one of {allowed_models}, got {entry_price_model}")
                    else:
                        print(f"ERROR: Telemetry validation failed: entry_price_model must be one of {allowed_models}, got {entry_price_model}")
                    return False
            
            # 4. Validate first_touch_bar histogram
            first_touch_bar = integration_metadata.get('first_touch_bar')
            if first_touch_bar is not None:
                if isinstance(first_touch_bar, (list, np.ndarray)):
                    # Check for values outside 1-6 range
                    invalid_values = [x for x in first_touch_bar if x is not None and (x < 1 or x > 6)]
                    if invalid_values:
                        if TPRINT_AVAILABLE:
                            tprint_error(f"❌ Telemetry validation failed: first_touch_bar contains values outside 1-6 range: {invalid_values}")
                        else:
                            print(f"ERROR: Telemetry validation failed: first_touch_bar contains values outside 1-6 range: {invalid_values}")
                        return False
            
            # 5. Validate opportunity_rate_per_day is reasonable
            opportunity_rate = integration_metadata.get('opportunity_rate_per_day')
            if opportunity_rate is not None:
                if not (0 <= opportunity_rate <= 1):
                    if TPRINT_AVAILABLE:
                        tprint_warning(f"⚠️ Telemetry warning: opportunity_rate_per_day should be 0-1, got {opportunity_rate}")
                    else:
                        print(f"WARNING: Telemetry warning: opportunity_rate_per_day should be 0-1, got {opportunity_rate}")
            
            # 6. Validate net_hit_ratio_after_costs is not negative
            net_hit_ratio = integration_metadata.get('net_hit_ratio_after_costs')
            if net_hit_ratio is not None and net_hit_ratio < 0:
                if TPRINT_AVAILABLE:
                    tprint_warning(f"⚠️ Telemetry warning: net_hit_ratio_after_costs is negative: {net_hit_ratio}")
                else:
                    print(f"WARNING: Telemetry warning: net_hit_ratio_after_costs is negative: {net_hit_ratio}")
            
            if TPRINT_AVAILABLE:
                tprint_success("✅ Financial telemetry validation passed")
            else:
                print("SUCCESS: Financial telemetry validation passed")
            
            return True
            
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Telemetry validation error: {e}")
            else:
                print(f"ERROR: Telemetry validation error: {e}")
            return False

# Command handler for ares_launcher integration
async def handle_feature_generation_labeling_integration_step(
    symbol: str = "ETHUSDT",
    timeframe: str = "15m",
    direction: str = "longs",
    intensity: str = "blank",
    lookback_days: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    exchange: str = "binance",
    custom_overrides: Optional[Dict[str, Any]] = None,
    **kwargs
) -> ComponentResult:
    """
    Handle feature generation labeling integration step command.

    Args:
        symbol: Trading symbol (default: "ETHUSDT")
        timeframe: Timeframe (default: "15m")
        direction: Direction (default: "longs")
        intensity: Pipeline intensity (default: "blank")
        lookback_days: Lookback days (optional)
        start_date: Start date (optional)
        end_date: End date (optional)
        exchange: Exchange (default: "binance")
        custom_overrides: Custom configuration overrides (optional)
        **kwargs: Additional arguments

    Returns:
        ComponentResult with integration results
    """
    # Validate that data is provided in kwargs
    data = kwargs.get('data')
    if data is None:
        raise ValueError("Data must be provided for labeling integration step")
    
    # Create step instance and execute
    step = FeatureGenerationLabelingIntegrationStep()

    return await step.execute(
        data=data,
        symbol=symbol,
        timeframe=timeframe,
        direction=direction,
        intensity=intensity,
        lookback_days=lookback_days,
        start_date=start_date,
        end_date=end_date,
        exchange=exchange,
        custom_overrides=custom_overrides
    )

# Register component with factory
def _register_feature_generation_labeling_integration_step():
    """Register the feature generation labeling integration step component with the factory."""
    try:
        from src.training.steps.pre_training.components import ComponentFactory
        ComponentFactory.register_component(
            'feature_generation_labeling_integration_step',
            FeatureGenerationLabelingIntegrationStep
        )
    except ImportError:
        # Component factory not available, skip registration
        pass

# Register the component when module is imported
_register_feature_generation_labeling_integration_step()