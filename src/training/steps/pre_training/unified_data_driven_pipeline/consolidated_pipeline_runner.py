"""
Consolidated Pipeline Runner

This module provides functions to run the consolidated pipeline up to specific steps,
allowing the step files to call the consolidated pipeline at the proper places.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple, Union, Callable, Awaitable
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

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

# Ensure tprint_error is always available
if not TPRINT_AVAILABLE:
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)

from .consolidated_pipeline import (
    UnifiedDataDrivenPipeline,
    create_unified_pipeline,
    ConsolidatedPipelineResult
)
from .core.config import UnifiedPipelineConfig, create_default_config
from .core.simplified_config import (
    create_full_config,
    create_blank_config,
    create_light_config,
    create_config_by_intensity,
    PipelineIntensity
)

class ConsolidatedPipelineRunner:
    """Runner for executing consolidated pipeline up to specific steps."""

    def __init__(self, config: Optional[UnifiedPipelineConfig] = None) -> None:
        """
        Initialize the pipeline runner.

        Args:
            config: Optional pipeline configuration. If None, uses default config.

        Raises:
            ValueError: If config is invalid
            ImportError: If required dependencies are missing
        """
        try:
            tprint_step("🚀 Initializing ConsolidatedPipelineRunner")

            if config is None:
                tprint_info("📋 Using default configuration")
                self.config = create_default_config()
            else:
                tprint_info("📋 Using provided configuration")
                self.config = config

            # Validate configuration
            if not isinstance(self.config, UnifiedPipelineConfig):
                raise ValueError(f"Invalid config type: {type(self.config)}. Expected UnifiedPipelineConfig.")

            tprint_info("🔧 Creating unified pipeline")
            self.pipeline = create_unified_pipeline(self.config)

            if self.pipeline is None:
                raise RuntimeError("Failed to create unified pipeline")

            self.logger = logging.getLogger(__name__)
            tprint_success("✅ ConsolidatedPipelineRunner initialized successfully")

        except Exception as e:
            error_msg = f"Failed to initialize ConsolidatedPipelineRunner: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e

    async def run_data_validation_step(self,
                                     data: pd.DataFrame,
                                     symbol: str = "ETHUSDT",
                                     timeframe: str = "15m",
                                     direction: str = "longs",
                                     intensity: str = "blank",
                                     lookback_days: Optional[int] = None,
                                     start_date: Optional[str] = None,
                                     end_date: Optional[str] = None,
                                     exchange: str = "binance",
                                     custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run pipeline up to data validation step.

        Args:
            data: Input DataFrame with time series data
            symbol: Trading symbol (default: "ETHUSDT")
            timeframe: Time frame for analysis (default: "15m")
            direction: Trading direction (default: "longs")
            intensity: Pipeline intensity level (default: "blank")
            lookback_days: Optional lookback period in days
            start_date: Optional start date for analysis
            end_date: Optional end date for analysis
            exchange: Exchange name (default: "binance")
            custom_overrides: Optional configuration overrides

        Returns:
            Dict containing validation results with keys:
            - success: bool
            - data_quality_score: float
            - validation_metadata: Dict[str, Any]
            - artifacts: Dict[str, Any]
            - error_message: Optional[str]

        Raises:
            ValueError: If input data is invalid
            RuntimeError: If pipeline execution fails
        """
        try:
            tprint_step("🔍 Starting data validation step")
            tprint_info(f"📊 Data shape: {data.shape[0]} rows × {data.shape[1]} columns")
            tprint_info(f"🎯 Symbol: {symbol}, Timeframe: {timeframe}, Direction: {direction}")
            tprint_info(f"⚙️ Intensity: {intensity}, Exchange: {exchange}")

            # Validate input data
            if data is None or data.empty:
                raise ValueError("Input data cannot be None or empty")

            if not isinstance(data, pd.DataFrame):
                raise ValueError(f"Expected pandas DataFrame, got {type(data)}")

            # Configure pipeline based on intensity
            tprint_info("🔧 Configuring pipeline based on intensity")
            config = self._create_config_from_intensity(intensity, custom_overrides)
            self.pipeline = create_unified_pipeline(config)

            if self.pipeline is None:
                raise RuntimeError("Failed to create unified pipeline with new configuration")

            # Get labels from pipeline state (from previous labeling steps)
            targets = self._get_labels_from_pipeline_state(custom_overrides)

            if targets is None:
                raise ValueError("No labels found in pipeline state. Please ensure analyst_profit_labeler or tactician_entry_labeler runs before this step.")

            # Create pipeline state
            pipeline_state = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'step': 'data_validation'
            }

            tprint_info("🚀 Executing pipeline up to data validation")
            # Run pipeline up to data validation
            result = await self.pipeline.process(data, targets=targets, timeframe=timeframe, pipeline_state=pipeline_state)

            if result is None:
                raise RuntimeError("Pipeline returned None result")

            # Extract validation results
            validation_result = {
                'success': result.success,
                'data_quality_score': getattr(result, 'data_quality_score', 0.0),
                'validation_metadata': getattr(result, 'validation_metadata', {}),
                'artifacts': result.artifacts or {},
                'error_message': result.error_message if not result.success else None
            }

            if validation_result['success']:
                tprint_success(f"✅ Data validation completed successfully")
                tprint_result(f"📈 Data quality score: {validation_result['data_quality_score']:.3f}")
                tprint_info(f"📋 Validation metadata keys: {list(validation_result['validation_metadata'].keys())}")
                tprint_info(f"📦 Artifacts generated: {len(validation_result['artifacts'])}")
            else:
                tprint_error(f"❌ Data validation failed: {validation_result['error_message']}")

            # Generate human-readable report
            tprint_info("📄 Generating human-readable report")
            await self._generate_data_validation_report(validation_result, data)

            return validation_result

        except ValueError as e:
            error_msg = f"Invalid input for data validation step: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            self.logger.error(error_msg)
            return {
                'success': False,
                'error_message': error_msg,
                'artifacts': {},
                'data_quality_score': 0.0,
                'validation_metadata': {}
            }
        except RuntimeError as e:
            error_msg = f"Runtime error in data validation step: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            self.logger.error(error_msg)
            return {
                'success': False,
                'error_message': error_msg,
                'artifacts': {},
                'data_quality_score': 0.0,
                'validation_metadata': {}
            }
        except Exception as e:
            error_msg = f"Unexpected error in data validation step: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            self.logger.error(f"Data validation step failed: {e}", exc_info=True)
            return {
                'success': False,
                'error_message': error_msg,
                'artifacts': {},
                'data_quality_score': 0.0,
                'validation_metadata': {}
            }

    async def run_feature_generation_step(self,
                                        data: pd.DataFrame,
                                        symbol: str = "ETHUSDT",
                                        timeframe: str = "15m",
                                        direction: str = "longs",
                                        intensity: str = "blank",
                                        lookback_days: Optional[int] = None,
                                        start_date: Optional[str] = None,
                                        end_date: Optional[str] = None,
                                        exchange: str = "binance",
                                        custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run pipeline up to feature generation step."""
        try:
            # Configure pipeline based on intensity
            config = self._create_config_from_intensity(intensity, custom_overrides)
            self.pipeline = create_unified_pipeline(config)

            # Get labels from pipeline state (from previous labeling steps)
            targets = self._get_labels_from_pipeline_state(custom_overrides)

            if targets is None:
                raise ValueError("No labels found in pipeline state. Please ensure analyst_profit_labeler or tactician_entry_labeler runs before this step.")

            # Create pipeline state
            pipeline_state = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'step': 'feature_generation'
            }

            # Run pipeline up to feature generation
            result = await self.pipeline.process(data, targets=targets, timeframe=timeframe, pipeline_state=pipeline_state)

            # Extract feature generation results
            feature_result = {
                'success': result.success,
                'generated_features': getattr(result, 'generated_features', pd.DataFrame()),
                'feature_metadata': getattr(result, 'feature_metadata', {}),
                'generation_metrics': getattr(result, 'generation_metrics', {}),
                'artifacts': result.artifacts or {},
                'error_message': result.error_message if not result.success else None
            }

            # Generate human-readable report
            await self._generate_feature_generation_report(feature_result, data)

            return feature_result

        except Exception as e:
            self.logger.error(f"Feature generation step failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'artifacts': {},
                'generated_features': pd.DataFrame(),
                'feature_metadata': {},
                'generation_metrics': {}
            }

    async def run_feature_selection_step(self,
                                       data: pd.DataFrame,
                                       symbol: str = "ETHUSDT",
                                       timeframe: str = "15m",
                                       direction: str = "longs",
                                       intensity: str = "blank",
                                       lookback_days: Optional[int] = None,
                                       start_date: Optional[str] = None,
                                       end_date: Optional[str] = None,
                                       exchange: str = "binance",
                                       custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run pipeline up to feature selection step."""
        try:
            # Configure pipeline based on intensity
            config = self._create_config_from_intensity(intensity, custom_overrides)
            self.pipeline = create_unified_pipeline(config)

            # Create pipeline state
            pipeline_state = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'step': 'feature_selection'
            }

            # Get labels from pipeline state (from previous labeling steps)
            targets = self._get_labels_from_pipeline_state(custom_overrides)

            if targets is None:
                raise ValueError("No labels found in pipeline state. Please ensure analyst_profit_labeler or tactician_entry_labeler runs before this step.")

            # Run pipeline up to feature selection
            result = await self.pipeline.process(data, targets=targets, timeframe=timeframe, pipeline_state=pipeline_state)

            # Extract feature selection results
            selection_result = {
                'success': result.success,
                'selected_features': getattr(result, 'selected_features', pd.DataFrame()),
                'selection_metadata': getattr(result, 'selection_metadata', {}),
                'selection_metrics': getattr(result, 'selection_metrics', {}),
                'artifacts': result.artifacts or {},
                'error_message': result.error_message if not result.success else None
            }

            # Generate human-readable report
            await self._generate_feature_selection_report(selection_result, data)

            return selection_result

        except Exception as e:
            self.logger.error(f"Feature selection step failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'artifacts': {},
                'selected_features': pd.DataFrame(),
                'selection_metadata': {},
                'selection_metrics': {}
            }

    async def run_period_optimization_step(self,
                                         data: pd.DataFrame,
                                         symbol: str = "ETHUSDT",
                                         timeframe: str = "15m",
                                         direction: str = "longs",
                                         intensity: str = "blank",
                                         lookback_days: Optional[int] = None,
                                         start_date: Optional[str] = None,
                                         end_date: Optional[str] = None,
                                         exchange: str = "binance",
                                         custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run pipeline up to period optimization step."""
        try:
            # Configure pipeline based on intensity
            config = self._create_config_from_intensity(intensity, custom_overrides)
            self.pipeline = create_unified_pipeline(config)

            # Create pipeline state
            pipeline_state = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'step': 'period_optimization'
            }

            # Get labels from pipeline state (from previous labeling steps)
            targets = self._get_labels_from_pipeline_state(custom_overrides)

            if targets is None:
                raise ValueError("No labels found in pipeline state. Please ensure analyst_profit_labeler or tactician_entry_labeler runs before this step.")

            # Run pipeline up to period optimization
            result = await self.pipeline.process(data, targets=targets, timeframe=timeframe, pipeline_state=pipeline_state)

            # Extract period optimization results
            optimization_result = {
                'success': result.success,
                'optimal_periods': getattr(result, 'optimal_periods', {}),
                'optimization_metrics': getattr(result, 'optimization_metrics', {}),
                'artifacts': result.artifacts or {},
                'error_message': result.error_message if not result.success else None
            }

            # Generate human-readable report
            await self._generate_period_optimization_report(optimization_result, data)

            return optimization_result

        except Exception as e:
            self.logger.error(f"Period optimization step failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'artifacts': {},
                'optimal_periods': {},
                'optimization_metrics': {}
            }

    async def run_lookback_optimization_step(self,
                                           data: pd.DataFrame,
                                           symbol: str = "ETHUSDT",
                                           timeframe: str = "15m",
                                           direction: str = "longs",
                                           intensity: str = "blank",
                                           lookback_days: Optional[int] = None,
                                           start_date: Optional[str] = None,
                                           end_date: Optional[str] = None,
                                           exchange: str = "binance",
                                           custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run pipeline up to lookback optimization step."""
        try:
            # Configure pipeline based on intensity
            config = self._create_config_from_intensity(intensity, custom_overrides)
            self.pipeline = create_unified_pipeline(config)

            # Create pipeline state
            pipeline_state = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'step': 'lookback_optimization'
            }

            # Get labels from pipeline state (from previous labeling steps)
            targets = self._get_labels_from_pipeline_state(custom_overrides)

            if targets is None:
                raise ValueError("No labels found in pipeline state. Please ensure analyst_profit_labeler or tactician_entry_labeler runs before this step.")

            # Run pipeline up to lookback optimization
            result = await self.pipeline.process(data, targets=targets, timeframe=timeframe, pipeline_state=pipeline_state)

            # Extract lookback optimization results
            optimization_result = {
                'success': result.success,
                'optimal_lookbacks': getattr(result, 'optimal_lookbacks', {}),
                'optimization_metrics': getattr(result, 'optimization_metrics', {}),
                'artifacts': result.artifacts or {},
                'error_message': result.error_message if not result.success else None
            }

            # Generate human-readable report
            await self._generate_lookback_optimization_report(optimization_result, data)

            return optimization_result

        except Exception as e:
            self.logger.error(f"Lookback optimization step failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'artifacts': {},
                'optimal_lookbacks': {},
                'optimization_metrics': {}
            }

    async def run_period_lookback_optimization_step(self,
                                                  data: pd.DataFrame,
                                                  symbol: str = "ETHUSDT",
                                                  timeframe: str = "15m",
                                                  direction: str = "longs",
                                                  intensity: str = "blank",
                                                  lookback_days: Optional[int] = None,
                                                  start_date: Optional[str] = None,
                                                  end_date: Optional[str] = None,
                                                  exchange: str = "binance",
                                                  custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run pipeline up to concurrent period + lookback optimization step."""
        try:
            # Configure pipeline based on intensity
            config = self._create_config_from_intensity(intensity, custom_overrides)
            self.pipeline = create_unified_pipeline(config)

            # Create pipeline state
            pipeline_state = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'step': 'period_lookback_optimization'
            }

            # Get labels from pipeline state (from previous labeling steps)
            targets = self._get_labels_from_pipeline_state(custom_overrides)

            if targets is None:
                raise ValueError("No labels found in pipeline state. Please ensure analyst_profit_labeler or tactician_entry_labeler runs before this step.")

            # Run pipeline up to concurrent period + lookback optimization
            result = await self.pipeline.process(data, targets=targets, timeframe=timeframe, pipeline_state=pipeline_state)

            # Extract period + lookback optimization results
            optimization_result = {
                'success': result.success,
                'period_results': getattr(result, 'period_results', {}),
                'lookback_results': getattr(result, 'lookback_results', {}),
                'combined_results': getattr(result, 'combined_results', {}),
                'trading_defaults': getattr(result, 'trading_defaults', {}),
                'interaction_periods': getattr(result, 'interaction_periods', []),
                'artifacts': result.artifacts or {},
                'error_message': result.error_message if not result.success else None
            }

            # Generate human-readable report
            await self._generate_period_lookback_optimization_report(optimization_result, data)

            return optimization_result

        except Exception as e:
            self.logger.error(f"Concurrent period + lookback optimization step failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'artifacts': {},
                'period_results': {},
                'lookback_results': {},
                'combined_results': {},
                'trading_defaults': {},
                'interaction_periods': []
            }

    async def run_interaction_generation_step(self,
                                            data: pd.DataFrame,
                                            symbol: str = "ETHUSDT",
                                            timeframe: str = "15m",
                                            direction: str = "longs",
                                            intensity: str = "blank",
                                            lookback_days: Optional[int] = None,
                                            start_date: Optional[str] = None,
                                            end_date: Optional[str] = None,
                                            exchange: str = "binance",
                                            custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run pipeline up to interaction generation step."""
        try:
            # Configure pipeline based on intensity
            config = self._create_config_from_intensity(intensity, custom_overrides)
            self.pipeline = create_unified_pipeline(config)

            # Create pipeline state
            pipeline_state = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'step': 'interaction_generation'
            }

            # Get labels from pipeline state (from previous labeling steps)
            targets = self._get_labels_from_pipeline_state(custom_overrides)

            if targets is None:
                raise ValueError("No labels found in pipeline state. Please ensure analyst_profit_labeler or tactician_entry_labeler runs before this step.")

            # Run pipeline up to interaction generation
            result = await self.pipeline.process(data, targets=targets, timeframe=timeframe, pipeline_state=pipeline_state)

            # Extract interaction generation results
            interaction_result = {
                'success': result.success,
                'interaction_features': getattr(result, 'interaction_features', pd.DataFrame()),
                'interaction_metadata': getattr(result, 'interaction_metadata', {}),
                'generation_metrics': getattr(result, 'generation_metrics', {}),
                'artifacts': result.artifacts or {},
                'error_message': result.error_message if not result.success else None
            }

            # Generate human-readable report
            await self._generate_interaction_generation_report(interaction_result, data)

            return interaction_result

        except Exception as e:
            self.logger.error(f"Interaction generation step failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'artifacts': {},
                'interaction_features': pd.DataFrame(),
                'interaction_metadata': {},
                'generation_metrics': {}
            }

    async def run_vectorization_step(self,
                                   data: pd.DataFrame,
                                   symbol: str = "ETHUSDT",
                                   timeframe: str = "15m",
                                   direction: str = "longs",
                                   intensity: str = "blank",
                                   lookback_days: Optional[int] = None,
                                   start_date: Optional[str] = None,
                                   end_date: Optional[str] = None,
                                   exchange: str = "binance",
                                   custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run pipeline up to vectorization step."""
        try:
            # Configure pipeline based on intensity
            config = self._create_config_from_intensity(intensity, custom_overrides)
            self.pipeline = create_unified_pipeline(config)

            # Create pipeline state
            pipeline_state = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'step': 'vectorization'
            }

            # Get labels from pipeline state (from previous labeling steps)
            targets = self._get_labels_from_pipeline_state(custom_overrides)

            if targets is None:
                raise ValueError("No labels found in pipeline state. Please ensure analyst_profit_labeler or tactician_entry_labeler runs before this step.")

            # Run pipeline up to vectorization
            result = await self.pipeline.process(data, targets=targets, timeframe=timeframe, pipeline_state=pipeline_state)

            # Extract vectorization results
            vectorization_result = {
                'success': result.success,
                'vectorized_features': getattr(result, 'vectorized_features', pd.DataFrame()),
                'vectorization_metadata': getattr(result, 'vectorization_metadata', {}),
                'performance_metrics': getattr(result, 'performance_metrics', {}),
                'artifacts': result.artifacts or {},
                'error_message': result.error_message if not result.success else None
            }

            # Generate human-readable report
            await self._generate_vectorization_report(vectorization_result, data)

            return vectorization_result

        except Exception as e:
            self.logger.error(f"Vectorization step failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'artifacts': {},
                'vectorized_features': pd.DataFrame(),
                'vectorization_metadata': {},
                'performance_metrics': {}
            }

    async def run_labeling_integration_step(self,
                                          data: pd.DataFrame,
                                          symbol: str = "ETHUSDT",
                                          timeframe: str = "15m",
                                          direction: str = "longs",
                                          intensity: str = "blank",
                                          lookback_days: Optional[int] = None,
                                          start_date: Optional[str] = None,
                                          end_date: Optional[str] = None,
                                          exchange: str = "binance",
                                          custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run pipeline up to labeling integration step."""
        try:
            # Configure pipeline based on intensity
            config = self._create_config_from_intensity(intensity, custom_overrides)
            self.pipeline = create_unified_pipeline(config)

            # Create pipeline state
            pipeline_state = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'step': 'labeling_integration'
            }

            # Get labels from pipeline state (from previous labeling steps)
            targets = self._get_labels_from_pipeline_state(custom_overrides)

            if targets is None:
                raise ValueError("No labels found in pipeline state. Please ensure analyst_profit_labeler or tactician_entry_labeler runs before this step.")

            # Run pipeline up to labeling integration
            result = await self.pipeline.process(data, targets=targets, timeframe=timeframe, pipeline_state=pipeline_state)

            # Extract labeling integration results
            labeling_result = {
                'success': result.success,
                'labeled_data': getattr(result, 'labeled_data', pd.DataFrame()),
                'labeling_metadata': getattr(result, 'labeling_metadata', {}),
                'quality_metrics': getattr(result, 'quality_metrics', {}),
                'artifacts': result.artifacts or {},
                'error_message': result.error_message if not result.success else None
            }

            # Generate human-readable report
            await self._generate_labeling_integration_report(labeling_result, data)

            return labeling_result

        except Exception as e:
            self.logger.error(f"Labeling integration step failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'artifacts': {},
                'labeled_data': pd.DataFrame(),
                'labeling_metadata': {},
                'quality_metrics': {}
            }

    async def run_final_validation_step(self,
                                      data: pd.DataFrame,
                                      symbol: str = "ETHUSDT",
                                      timeframe: str = "15m",
                                      direction: str = "longs",
                                      intensity: str = "blank",
                                      lookback_days: Optional[int] = None,
                                      start_date: Optional[str] = None,
                                      end_date: Optional[str] = None,
                                      exchange: str = "binance",
                                      custom_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run pipeline up to final validation step."""
        try:
            # Configure pipeline based on intensity
            config = self._create_config_from_intensity(intensity, custom_overrides)
            self.pipeline = create_unified_pipeline(config)

            # Create pipeline state
            pipeline_state = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'step': 'final_validation'
            }

            # Get labels from pipeline state (from previous labeling steps)
            targets = self._get_labels_from_pipeline_state(custom_overrides)

            if targets is None:
                raise ValueError("No labels found in pipeline state. Please ensure analyst_profit_labeler or tactician_entry_labeler runs before this step.")

            # Run pipeline up to final validation
            result = await self.pipeline.process(data, targets=targets, timeframe=timeframe, pipeline_state=pipeline_state)

            # Extract final validation results
            validation_result = {
                'success': result.success,
                'final_dataset': getattr(result, 'final_dataset', pd.DataFrame()),
                'validation_summary': getattr(result, 'validation_summary', {}),
                'quality_metrics': getattr(result, 'quality_metrics', {}),
                'pipeline_summary': getattr(result, 'pipeline_summary', {}),
                'artifacts': result.artifacts or {},
                'error_message': result.error_message if not result.success else None
            }

            # Generate human-readable report
            await self._generate_final_validation_report(validation_result, data)

            return validation_result

        except Exception as e:
            self.logger.error(f"Final validation step failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'artifacts': {},
                'final_dataset': pd.DataFrame(),
                'validation_summary': {},
                'quality_metrics': {},
                'pipeline_summary': {}
            }

    def _get_labels_from_pipeline_state(self, pipeline_state: Optional[Dict[str, Any]]) -> Optional[pd.Series]:
        """
        Extract labels from pipeline state (from previous labeling steps).

        Args:
            pipeline_state: Pipeline state dictionary containing labels from previous steps

        Returns:
            Target series from previous steps, or None if not found
        """
        if not pipeline_state:
            return None

        # Try to get labels from various possible sources in pipeline state
        if 'labeled_data' in pipeline_state and 'target' in pipeline_state['labeled_data'].columns:
            return pipeline_state['labeled_data']['target']
        elif 'targets' in pipeline_state:
            return pipeline_state['targets']
        elif 'labels' in pipeline_state:
            return pipeline_state['labels']
        else:
            return None

    def _create_config_from_intensity(self, intensity: str, custom_overrides: Optional[Dict[str, Any]] = None) -> UnifiedPipelineConfig:
        """
        Create configuration based on intensity level.

        Args:
            intensity: Intensity level ("full", "blank", "light")
            custom_overrides: Optional configuration overrides

        Returns:
            Configured UnifiedPipelineConfig instance

        Raises:
            ValueError: If intensity is invalid
            RuntimeError: If config creation fails
        """
        try:
            tprint_info(f"🔧 Creating configuration for intensity: {intensity}")

            # Validate intensity parameter
            valid_intensities = {"full", "blank", "light"}
            if intensity not in valid_intensities:
                raise ValueError(f"Invalid intensity '{intensity}'. Must be one of: {valid_intensities}")

            # Create base configuration
            if intensity == "full":
                tprint_info("📋 Creating full intensity configuration (100%)")
                config = create_full_config()
            elif intensity == "blank":
                tprint_info("📋 Creating blank intensity configuration (25%)")
                config = create_blank_config()
            elif intensity == "light":
                tprint_info("📋 Creating light intensity configuration (10%)")
                config = create_light_config()
            else:
                tprint_warning(f"⚠️ Unknown intensity '{intensity}', falling back to blank")
                config = create_config_by_intensity(PipelineIntensity.BLANK)

            if config is None:
                raise RuntimeError("Failed to create configuration")

            # Apply custom overrides if provided
            if custom_overrides:
                tprint_info(f"🔧 Applying {len(custom_overrides)} custom overrides")
                for key, value in custom_overrides.items():
                    if hasattr(config, key):
                        old_value = getattr(config, key)
                        setattr(config, key, value)
                        tprint_debug(f"  - {key}: {old_value} → {value}")
                    else:
                        tprint_warning(f"  - Unknown config key: {key}")

            tprint_success(f"✅ Configuration created successfully for intensity: {intensity}")
            return config

        except ValueError as e:
            error_msg = f"Invalid intensity parameter: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            raise ValueError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to create configuration: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e

    async def _generate_data_validation_report(self, result: Dict[str, Any], data: pd.DataFrame) -> None:
        """
        Generate human-readable report for data validation step.

        Args:
            result: Validation result dictionary
            data: Input DataFrame

        Raises:
            OSError: If report file cannot be created
            ValueError: If result data is invalid
        """
        # Create outcomes directory
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report_filename = f"data_validation_report_{timestamp}.md"
        report_path = outcomes_dir / report_filename
        try:
            tprint_info("📄 Generating data validation report")

            # Validate inputs
            if not isinstance(result, dict):
                raise ValueError(f"Result must be a dictionary, got {type(result)}")

            if not isinstance(data, pd.DataFrame):
                raise ValueError(f"Data must be a pandas DataFrame, got {type(data)}")

            # Create outcomes directory
            outcomes_dir = Path("outcomes")
            outcomes_dir.mkdir(exist_ok=True)
            tprint_debug(f"📁 Using outcomes directory: {outcomes_dir.absolute()}")

            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"data_validation_report_{timestamp}.md"
            report_path = outcomes_dir / report_filename

            # Generate report content
            status_emoji = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            quality_score = result.get('data_quality_score', 0.0)
            error_msg = result.get('error_message', 'None')
            artifacts_count = len(result.get('artifacts', {}))

            report_content = f"""# Data Validation Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- **Status**: {status_emoji}
- **Data Shape**: {data.shape[0]} rows x {data.shape[1]} columns
- **Quality Score**: {quality_score:.3f}

## Validation Results
- **Success**: {result['success']}
- **Error Message**: {error_msg}
- **Artifacts Generated**: {artifacts_count}

## Data Quality Metrics
- **Rows**: {data.shape[0]:,}
- **Columns**: {data.shape[1]:,}
- **Memory Usage**: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB
- **Missing Values**: {data.isnull().sum().sum():,}

## Next Steps
1. Review validation results
2. Address any issues if present
3. Proceed to feature generation step

---
*Report generated by Consolidated Pipeline Runner*
"""

            # Write report
            tprint_debug(f"💾 Writing report to: {report_path}")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)

            # Add report to artifacts
            result['artifacts']['human_readable_report'] = str(report_path)

            tprint_success(f"📊 Human-readable report saved: {report_path}")

        except OSError as e:
            error_msg = f"Failed to create report file: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            self.logger.error(error_msg)
            raise OSError(error_msg) from e
        except ValueError as e:
            error_msg = f"Invalid data for report generation: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            self.logger.error(error_msg)
            raise ValueError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error generating report: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    async def _generate_feature_generation_report(self, result: Dict[str, Any], data: pd.DataFrame) -> None:
        """Generate human-readable report for feature generation step."""
        # Create outcomes directory
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report_filename = f"feature_generation_report_{timestamp}.md"
        report_path = outcomes_dir / report_filename

        # Generate report content
        report_content = f"""# Feature Generation Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- **Status**: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}
- **Data Shape**: {data.shape[0]} rows x {data.shape[1]} columns
- **Generated Features**: {len(result.get('generated_features', pd.DataFrame()).columns)}

## Generation Results
- **Success**: {result['success']}
- **Error Message**: {result.get('error_message', 'None')}
- **Artifacts Generated**: {len(result.get('artifacts', {}))}

## Next Steps
1. Review generated features
2. Proceed to feature selection step

---
*Report generated by Consolidated Pipeline Runner*
"""

        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)

        # Add report to artifacts
        result['artifacts']['human_readable_report'] = str(report_path)

        self.logger.info(f"📊 Human-readable report saved: {report_path}")

    async def _generate_feature_selection_report(self, result: Dict[str, Any], data: pd.DataFrame) -> None:
        """Generate human-readable report for feature selection step."""
        # Create outcomes directory
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report_filename = f"feature_selection_report_{timestamp}.md"
        report_path = outcomes_dir / report_filename

        # Generate report content
        report_content = f"""# Feature Selection Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- **Status**: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}
- **Data Shape**: {data.shape[0]} rows x {data.shape[1]} columns
- **Selected Features**: {len(result.get('selected_features', pd.DataFrame()).columns)}

## Selection Results
- **Success**: {result['success']}
- **Error Message**: {result.get('error_message', 'None')}
- **Artifacts Generated**: {len(result.get('artifacts', {}))}

## Next Steps
1. Review selected features
2. Proceed to period optimization step

---
*Report generated by Consolidated Pipeline Runner*
"""

        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)

        # Add report to artifacts
        result['artifacts']['human_readable_report'] = str(report_path)

        self.logger.info(f"📊 Human-readable report saved: {report_path}")

    async def _generate_period_optimization_report(self, result: Dict[str, Any], data: pd.DataFrame) -> None:
        """Generate human-readable report for period optimization step."""
        # Create outcomes directory
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report_filename = f"period_optimization_report_{timestamp}.md"
        report_path = outcomes_dir / report_filename

        # Generate report content
        report_content = f"""# Period Optimization Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- **Status**: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}
- **Data Shape**: {data.shape[0]} rows x {data.shape[1]} columns
- **Optimized Periods**: {len(result.get('optimal_periods', {}))}

## Optimization Results
- **Success**: {result['success']}
- **Error Message**: {result.get('error_message', 'None')}
- **Artifacts Generated**: {len(result.get('artifacts', {}))}

## Next Steps
1. Review optimized periods
2. Proceed to lookback optimization step

---
*Report generated by Consolidated Pipeline Runner*
"""

        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)

        # Add report to artifacts
        result['artifacts']['human_readable_report'] = str(report_path)

        self.logger.info(f"📊 Human-readable report saved: {report_path}")

    async def _generate_lookback_optimization_report(self, result: Dict[str, Any], data: pd.DataFrame) -> None:
        """Generate human-readable report for lookback optimization step."""
        # Create outcomes directory
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report_filename = f"lookback_optimization_report_{timestamp}.md"
        report_path = outcomes_dir / report_filename

        # Generate report content
        report_content = f"""# Lookback Optimization Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- **Status**: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}
- **Data Shape**: {data.shape[0]} rows x {data.shape[1]} columns
- **Optimized Lookbacks**: {len(result.get('optimal_lookbacks', {}))}

## Optimization Results
- **Success**: {result['success']}
- **Error Message**: {result.get('error_message', 'None')}
- **Artifacts Generated**: {len(result.get('artifacts', {}))}

## Next Steps
1. Review optimized lookbacks
2. Proceed to interaction generation step

---
*Report generated by Consolidated Pipeline Runner*
"""

        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)

        # Add report to artifacts
        result['artifacts']['human_readable_report'] = str(report_path)

        self.logger.info(f"📊 Human-readable report saved: {report_path}")

    async def _generate_interaction_generation_report(self, result: Dict[str, Any], data: pd.DataFrame) -> None:
        """Generate human-readable report for interaction generation step."""
        # Create outcomes directory
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report_filename = f"interaction_generation_report_{timestamp}.md"
        report_path = outcomes_dir / report_filename

        # Generate report content
        report_content = f"""# Interaction Generation Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- **Status**: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}
- **Data Shape**: {data.shape[0]} rows x {data.shape[1]} columns
- **Interaction Features**: {len(result.get('interaction_features', pd.DataFrame()).columns)}

## Generation Results
- **Success**: {result['success']}
- **Error Message**: {result.get('error_message', 'None')}
- **Artifacts Generated**: {len(result.get('artifacts', {}))}

## Next Steps
1. Review interaction features
2. Proceed to vectorization step

---
*Report generated by Consolidated Pipeline Runner*
"""

        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)

        # Add report to artifacts
        result['artifacts']['human_readable_report'] = str(report_path)

        self.logger.info(f"📊 Human-readable report saved: {report_path}")

    async def _generate_vectorization_report(self, result: Dict[str, Any], data: pd.DataFrame) -> None:
        """Generate human-readable report for vectorization step."""
        # Create outcomes directory
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report_filename = f"vectorization_report_{timestamp}.md"
        report_path = outcomes_dir / report_filename

        # Generate report content
        report_content = f"""# Vectorization Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- **Status**: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}
- **Data Shape**: {data.shape[0]} rows x {data.shape[1]} columns
- **Vectorized Features**: {len(result.get('vectorized_features', pd.DataFrame()).columns)}

## Vectorization Results
- **Success**: {result['success']}
- **Error Message**: {result.get('error_message', 'None')}
- **Artifacts Generated**: {len(result.get('artifacts', {}))}

## Next Steps
1. Review vectorized features
2. Proceed to labeling integration step

---
*Report generated by Consolidated Pipeline Runner*
"""

        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)

        # Add report to artifacts
        result['artifacts']['human_readable_report'] = str(report_path)

        self.logger.info(f"📊 Human-readable report saved: {report_path}")

    async def _generate_labeling_integration_report(self, result: Dict[str, Any], data: pd.DataFrame) -> None:
        """Generate human-readable report for labeling integration step."""
        # Create outcomes directory
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report_filename = f"labeling_integration_report_{timestamp}.md"
        report_path = outcomes_dir / report_filename

        # Generate report content
        report_content = f"""# Labeling Integration Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- **Status**: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}
- **Data Shape**: {data.shape[0]} rows x {data.shape[1]} columns
- **Labeled Data**: {len(result.get('labeled_data', pd.DataFrame()).columns)}

## Labeling Results
- **Success**: {result['success']}
- **Error Message**: {result.get('error_message', 'None')}
- **Artifacts Generated**: {len(result.get('artifacts', {}))}

## Next Steps
1. Review labeled data
2. Proceed to final validation step

---
*Report generated by Consolidated Pipeline Runner*
"""

        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)

        # Add report to artifacts
        result['artifacts']['human_readable_report'] = str(report_path)

        self.logger.info(f"📊 Human-readable report saved: {report_path}")

    async def _generate_final_validation_report(self, result: Dict[str, Any], data: pd.DataFrame) -> None:
        """Generate human-readable report for final validation step."""
        # Create outcomes directory
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report_filename = f"final_validation_report_{timestamp}.md"
        report_path = outcomes_dir / report_filename

        # Generate report content
        report_content = f"""# Final Validation Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
- **Status**: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}
- **Data Shape**: {data.shape[0]} rows x {data.shape[1]} columns
- **Final Dataset**: {len(result.get('final_dataset', pd.DataFrame()).columns)}

## Validation Results
- **Success**: {result['success']}
- **Error Message**: {result.get('error_message', 'None')}
- **Artifacts Generated**: {len(result.get('artifacts', {}))}

## Next Steps
1. Review final dataset
2. Use dataset for model training

---
*Report generated by Consolidated Pipeline Runner*
"""

        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)

        # Add report to artifacts
        result['artifacts']['human_readable_report'] = str(report_path)

        self.logger.info(f"📊 Human-readable report saved: {report_path}")

# Convenience functions for each step
async def run_data_validation_step(data: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
    """
    Run data validation step using consolidated pipeline.

    Args:
        data: Input DataFrame with time series data
        **kwargs: Additional arguments passed to the step

    Returns:
        Dict containing validation results

    Raises:
        ValueError: If input data is invalid
        RuntimeError: If pipeline execution fails
    """
    try:
        tprint_step("🚀 Starting data validation step (convenience function)")
        runner = ConsolidatedPipelineRunner()
        return await runner.run_data_validation_step(data, **kwargs)
    except Exception as e:
        error_msg = f"Convenience function failed for data validation: {str(e)}"
        tprint_error(f"❌ {error_msg}")
        raise RuntimeError(error_msg) from e

async def run_feature_generation_step(data: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
    """
    Run feature generation step using consolidated pipeline.

    Args:
        data: Input DataFrame with time series data
        **kwargs: Additional arguments passed to the step

    Returns:
        Dict containing feature generation results

    Raises:
        ValueError: If input data is invalid
        RuntimeError: If pipeline execution fails
    """
    try:
        tprint_step("🚀 Starting feature generation step (convenience function)")
        runner = ConsolidatedPipelineRunner()
        return await runner.run_feature_generation_step(data, **kwargs)
    except Exception as e:
        error_msg = f"Convenience function failed for feature generation: {str(e)}"
        tprint_error(f"❌ {error_msg}")
        raise RuntimeError(error_msg) from e

async def run_feature_selection_step(data: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
    """
    Run feature selection step using consolidated pipeline.

    Args:
        data: Input DataFrame with time series data
        **kwargs: Additional arguments passed to the step

    Returns:
        Dict containing feature selection results

    Raises:
        ValueError: If input data is invalid
        RuntimeError: If pipeline execution fails
    """
    try:
        tprint_step("🚀 Starting feature selection step (convenience function)")
        runner = ConsolidatedPipelineRunner()
        return await runner.run_feature_selection_step(data, **kwargs)
    except Exception as e:
        error_msg = f"Convenience function failed for feature selection: {str(e)}"
        tprint_error(f"❌ {error_msg}")
        raise RuntimeError(error_msg) from e

async def run_period_optimization_step(data: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
    """
    Run period optimization step using consolidated pipeline.

    Args:
        data: Input DataFrame with time series data
        **kwargs: Additional arguments passed to the step

    Returns:
        Dict containing period optimization results

    Raises:
        ValueError: If input data is invalid
        RuntimeError: If pipeline execution fails
    """
    try:
        tprint_step("🚀 Starting period optimization step (convenience function)")
        runner = ConsolidatedPipelineRunner()
        return await runner.run_period_optimization_step(data, **kwargs)
    except Exception as e:
        error_msg = f"Convenience function failed for period optimization: {str(e)}"
        tprint_error(f"❌ {error_msg}")
        raise RuntimeError(error_msg) from e

async def run_lookback_optimization_step(data: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
    """
    Run lookback optimization step using consolidated pipeline.

    Args:
        data: Input DataFrame with time series data
        **kwargs: Additional arguments passed to the step

    Returns:
        Dict containing lookback optimization results

    Raises:
        ValueError: If input data is invalid
        RuntimeError: If pipeline execution fails
    """
    try:
        tprint_step("🚀 Starting lookback optimization step (convenience function)")
        runner = ConsolidatedPipelineRunner()
        return await runner.run_lookback_optimization_step(data, **kwargs)
    except Exception as e:
        error_msg = f"Convenience function failed for lookback optimization: {str(e)}"
        tprint_error(f"❌ {error_msg}")
        raise RuntimeError(error_msg) from e

async def run_interaction_generation_step(data: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
    """
    Run interaction generation step using consolidated pipeline.

    Args:
        data: Input DataFrame with time series data
        **kwargs: Additional arguments passed to the step

    Returns:
        Dict containing interaction generation results

    Raises:
        ValueError: If input data is invalid
        RuntimeError: If pipeline execution fails
    """
    try:
        tprint_step("🚀 Starting interaction generation step (convenience function)")
        runner = ConsolidatedPipelineRunner()
        return await runner.run_interaction_generation_step(data, **kwargs)
    except Exception as e:
        error_msg = f"Convenience function failed for interaction generation: {str(e)}"
        tprint_error(f"❌ {error_msg}")
        raise RuntimeError(error_msg) from e

async def run_vectorization_step(data: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
    """
    Run vectorization step using consolidated pipeline.

    Args:
        data: Input DataFrame with time series data
        **kwargs: Additional arguments passed to the step

    Returns:
        Dict containing vectorization results

    Raises:
        ValueError: If input data is invalid
        RuntimeError: If pipeline execution fails
    """
    try:
        tprint_step("🚀 Starting vectorization step (convenience function)")
        runner = ConsolidatedPipelineRunner()
        return await runner.run_vectorization_step(data, **kwargs)
    except Exception as e:
        error_msg = f"Convenience function failed for vectorization: {str(e)}"
        tprint_error(f"❌ {error_msg}")
        raise RuntimeError(error_msg) from e

async def run_labeling_integration_step(data: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
    """
    Run labeling integration step using consolidated pipeline.

    Args:
        data: Input DataFrame with time series data
        **kwargs: Additional arguments passed to the step

    Returns:
        Dict containing labeling integration results

    Raises:
        ValueError: If input data is invalid
        RuntimeError: If pipeline execution fails
    """
    try:
        tprint_step("🚀 Starting labeling integration step (convenience function)")
        runner = ConsolidatedPipelineRunner()
        return await runner.run_labeling_integration_step(data, **kwargs)
    except Exception as e:
        error_msg = f"Convenience function failed for labeling integration: {str(e)}"
        tprint_error(f"❌ {error_msg}")
        raise RuntimeError(error_msg) from e

async def run_period_lookback_optimization_step(data: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
    """
    Run concurrent period + lookback optimization step using consolidated pipeline.

    Args:
        data: Input DataFrame with time series data
        **kwargs: Additional arguments passed to the step

    Returns:
        Dict containing period + lookback optimization results

    Raises:
        ValueError: If input data is invalid
        RuntimeError: If pipeline execution fails
    """
    try:
        tprint_step("🚀 Starting concurrent period + lookback optimization step (convenience function)")
        runner = ConsolidatedPipelineRunner()
        return await runner.run_period_lookback_optimization_step(data, **kwargs)
    except Exception as e:
        error_msg = f"Convenience function failed for period + lookback optimization: {str(e)}"
        tprint_error(f"❌ {error_msg}")
        raise RuntimeError(error_msg) from e

async def run_final_validation_step(data: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
    """
    Run final validation step using consolidated pipeline.

    Args:
        data: Input DataFrame with time series data
        **kwargs: Additional arguments passed to the step

    Returns:
        Dict containing final validation results

    Raises:
        ValueError: If input data is invalid
        RuntimeError: If pipeline execution fails
    """
    try:
        tprint_step("🚀 Starting final validation step (convenience function)")
        runner = ConsolidatedPipelineRunner()
        return await runner.run_final_validation_step(data, **kwargs)
    except Exception as e:
        error_msg = f"Convenience function failed for final validation: {str(e)}"
        tprint_error(f"❌ {error_msg}")
        raise RuntimeError(error_msg) from e
