"""
Tactician Pre-ML Orchestration - 5m Timeframe Processing

This module provides the pre-ML orchestration for Tactician models with:
1. Differentiated horizon labeling for 5m timeframe
2. Feature lookback optimization for 5m data
3. PID features generation for 5m timeframe
4. Final feature selection for Tactician models

The Tactician determines WHEN we trade using 5m timeframe data and Analyst signals.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
import traceback
import time
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success

# Import enhanced debugging utilities
try:
    from src.training.utils.debug_utilities import TrainingDebugger, create_enhanced_error_handler
    DEBUG_UTILITIES_AVAILABLE = True
except ImportError:
    DEBUG_UTILITIES_AVAILABLE = False

# Import universal validation integration
try:
    from src.utils.ml_common.training.universal_validation_integration import (
        get_validation_integrator, ValidationIntegrationConfig,
        intelligently_select_utilities, perform_data_leakage_check,
        perform_enhanced_validation, perform_complexity_analysis
    )
    UNIVERSAL_VALIDATION_AVAILABLE = True
except ImportError:
    UNIVERSAL_VALIDATION_AVAILABLE = False

# Import feature engineering utilities
try:
    from src.training.utils.feature_calculators import (
        calculate_technical_indicators, calculate_statistical_features,
        calculate_price_features, calculate_volume_features
    )
    FEATURE_CALCULATORS_AVAILABLE = True
except ImportError:
    FEATURE_CALCULATORS_AVAILABLE = False

# Import feature selection utilities
try:
    from src.training.utils.feature_selection.feature_selection_engine import FeatureSelectionEngine
    from src.training.utils.feature_selection.boruta_selection import BorutaFeatureSelector
    from src.training.utils.feature_selection.mutual_information_selection import MutualInformationSelector
    FEATURE_SELECTION_AVAILABLE = True
except ImportError:
    FEATURE_SELECTION_AVAILABLE = False

@dataclass
class TacticianPreMLOrchestrationConfig:
    """Configuration for Tactician pre-ML orchestration."""
    # Data configuration
    timeframe: str = "5m"
    analyst_confidence_threshold: float = 0.004  # Use Analyst signals above 0.4% confidence

    # Feature optimization
    enable_feature_optimization: bool = True
    max_lookback_periods: int = 60  # Shorter for 5m timeframe
    min_lookback_periods: int = 3

    # PID features
    enable_pid_features: bool = True
    pid_window_sizes: List[int] = None
    pid_polynomial_degrees: List[int] = None

    # Horizon labeling (shorter horizons for 5m timeframe)
    enable_horizon_labeling: bool = True
    horizon_periods: List[int] = None
    profit_thresholds: List[float] = None

    # Feature selection
    enable_feature_selection: bool = True
    max_features_per_model: int = 40  # Slightly fewer than Analyst
    feature_selection_method: str = "boruta"

    # Analyst integration
    require_analyst_signals: bool = True
    analyst_signal_columns: List[str] = None

    # Output configuration
    output_directory: str = "generated/tactician_pre_ml_orchestration"
    save_intermediate_results: bool = True

    def __post_init__(self):
        """Post-initialization setup."""
        if self.pid_window_sizes is None:
            self.pid_window_sizes = [3, 5, 10, 15, 20, 30]

        if self.pid_polynomial_degrees is None:
            self.pid_polynomial_degrees = [1, 2, 3]

        if self.horizon_periods is None:
            # 5m timeframe horizons: 15m, 1h, 3h, 6h (shorter than 15m horizons)
            self.horizon_periods = [3, 12, 36, 72]  # in 5m periods

        if self.profit_thresholds is None:
            self.profit_thresholds = [0.0005, 0.001, 0.002, 0.005]  # 0.05%, 0.1%, 0.2%, 0.5%

        if self.analyst_signal_columns is None:
            self.analyst_signal_columns = ['analyst_signal_long', 'analyst_signal_short', 'analyst_confidence']


@dataclass
class TacticianPreMLOrchestrationResult:
    """Result of Tactician pre-ML orchestration."""
    # Processing results
    differentiated_labels: pd.DataFrame = None
    optimized_features: pd.DataFrame = None
    pid_features: pd.DataFrame = None
    selected_features: pd.DataFrame = None

    # Analyst integration
    analyst_signals_used: pd.DataFrame = None
    filtered_analyst_signals: pd.DataFrame = None

    # Metadata
    processing_time: float = 0.0
    total_samples: int = 0
    selected_feature_count: int = 0

    # Status
    success: bool = False
    error_message: str = None


class TacticianPreMLOrchestrator:
    """
    Tactician Pre-ML Orchestrator for 5m timeframe processing.

    Handles differentiated horizon labeling, feature lookback optimization,
    PID features generation, and final feature selection for Tactician models.
    """

    def __init__(self, config: Optional[TacticianPreMLOrchestrationConfig] = None):
        """Initialize the Tactician pre-ML orchestrator."""
        self.config = config or TacticianPreMLOrchestrationConfig()
        self.logger = system_logger.getChild('TacticianPreMLOrchestrator')

        # Initialize debugging utilities
        if DEBUG_UTILITIES_AVAILABLE:
            self.debugger = TrainingDebugger("tactician_pre_ml_orchestration", config)
        else:
            self.debugger = None

        # Initialize validation integrator
        if UNIVERSAL_VALIDATION_AVAILABLE:
            validation_config = ValidationIntegrationConfig(
                enable_validation=True,
                enable_overfitting_detection=True,
                enable_temporal_validation=True,
                enable_data_leakage_prevention=True,
                enable_feature_drift_detection=True
            )
            self.validation_integrator = get_validation_integrator(validation_config)
        else:
            self.validation_integrator = None

        tprint_success("✅ TacticianPreMLOrchestrator initialized successfully")

    async def orchestrate_pre_ml_processing(
        self,
        market_data: pd.DataFrame,
        regime_labels: Optional[pd.Series] = None,
        analyst_signals: Optional[pd.DataFrame] = None
    ) -> TacticianPreMLOrchestrationResult:
        """
        Orchestrate the complete pre-ML processing for Tactician models.

        Args:
            market_data: Market data for 5m timeframe
            regime_labels: Regime labels for per-regime processing
            analyst_signals: Analyst signals with confidence scores

        Returns:
            TacticianPreMLOrchestrationResult with processed data
        """
        start_time = time.time()
        tprint_info("🚀 Starting Tactician pre-ML orchestration for 5m timeframe...")

        result = TacticianPreMLOrchestrationResult()

        try:
            # Step 0: Filter Analyst signals by confidence threshold
            if self.config.require_analyst_signals and analyst_signals is not None:
                tprint_info("🎯 Step 0: Filtering Analyst signals by confidence threshold...")
                analyst_result = await self._filter_analyst_signals(analyst_signals)
                if not analyst_result['success']:
                    raise ValueError(f"Analyst signal filtering failed: {analyst_result['error']}")

                result.analyst_signals_used = analyst_result['original_signals']
                result.filtered_analyst_signals = analyst_result['filtered_signals']
                tprint_success(f"✅ Filtered {analyst_result['filtered_count']} Analyst signals above {self.config.analyst_confidence_threshold} threshold")
            else:
                tprint_info("⚠️ No Analyst signals provided or filtering disabled")

            # Step 1: Apply differentiated horizon labeling
            tprint_info("📊 Step 1: Applying differentiated horizon labeling...")
            labeling_result = await self._apply_differentiated_horizon_labeling(market_data)
            if not labeling_result['success']:
                raise ValueError(f"Horizon labeling failed: {labeling_result['error']}")

            result.differentiated_labels = labeling_result['labels']
            tprint_success(f"✅ Applied differentiated labeling with {len(labeling_result['labels'])} samples")

            # Step 2: Optimize feature lookback periods
            tprint_info("🔍 Step 2: Optimizing feature lookback periods...")
            lookback_result = await self._optimize_feature_lookback_periods(market_data, result.differentiated_labels)
            if not lookback_result['success']:
                raise ValueError(f"Lookback optimization failed: {lookback_result['error']}")

            result.optimized_features = lookback_result['features']
            tprint_success(f"✅ Optimized lookback periods: {lookback_result['optimal_periods']}")

            # Step 3: Generate PID features
            tprint_info("🧬 Step 3: Generating PID features...")
            pid_result = await self._generate_pid_features(market_data, result.optimized_features)
            if not pid_result['success']:
                raise ValueError(f"PID feature generation failed: {pid_result['error']}")

            result.pid_features = pid_result['features']
            tprint_success(f"✅ Generated {pid_result['feature_count']} PID features")

            # Step 4: Select final features
            tprint_info("🔧 Step 4: Selecting final features...")
            selection_result = await self._select_final_features(result.pid_features, result.differentiated_labels)
            if not selection_result['success']:
                raise ValueError(f"Feature selection failed: {selection_result['error']}")

            result.selected_features = selection_result['features']
            result.selected_feature_count = selection_result['selected_count']
            tprint_success(f"✅ Selected {selection_result['selected_count']} final features")

            # Set success metadata
            result.processing_time = time.time() - start_time
            result.total_samples = len(result.differentiated_labels)
            result.success = True

            tprint_success(f"✅ Tactician pre-ML orchestration completed in {result.processing_time:.2f}s")
            return result

        except Exception as e:
            result.processing_time = time.time() - start_time
            result.error_message = str(e)
            result.success = False

            tprint_error(f"❌ Tactician pre-ML orchestration failed: {e}")
            self.logger.error(f"❌ Pre-ML orchestration failed: {e}", exc_info=True)
            raise

    async def _filter_analyst_signals(self, analyst_signals: pd.DataFrame) -> Dict[str, Any]:
        """Filter Analyst signals by confidence threshold."""
        try:
            tprint_info("🎯 Filtering Analyst signals by confidence threshold...")

            if analyst_signals is None or analyst_signals.empty:
                return {'success': False, 'error': 'No Analyst signals provided'}

            # Look for confidence columns
            confidence_cols = [col for col in analyst_signals.columns if 'confidence' in col.lower()]

            if not confidence_cols:
                # If no confidence column, assume all signals are valid
                tprint_warning("⚠️ No confidence columns found in Analyst signals, using all signals")
                return {
                    'success': True,
                    'original_signals': analyst_signals,
                    'filtered_signals': analyst_signals,
                    'filtered_count': len(analyst_signals)
                }

            # Filter by confidence threshold
            filtered_signals = analyst_signals.copy()

            for conf_col in confidence_cols:
                # Apply threshold filtering
                mask = analyst_signals[conf_col] >= self.config.analyst_confidence_threshold
                filtered_signals = filtered_signals[mask]

            tprint_success(f"✅ Filtered {len(filtered_signals)} signals above {self.config.analyst_confidence_threshold} threshold")
            return {
                'success': True,
                'original_signals': analyst_signals,
                'filtered_signals': filtered_signals,
                'filtered_count': len(filtered_signals)
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _apply_differentiated_horizon_labeling(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Apply differentiated horizon labeling for 5m timeframe."""
        try:
            tprint_info("🎯 Applying differentiated horizon labeling for 5m timeframe...")

            # Get price data for labeling
            if 'close' not in market_data.columns:
                return {'success': False, 'error': 'Close price data not available for labeling'}

            close_prices = market_data['close']

            # Initialize labels dataframe
            labels_df = pd.DataFrame(index=market_data.index)
            labels_df['timestamp'] = market_data.index

            # Apply different labeling strategies based on horizons (shorter for 5m)
            for i, horizon in enumerate(self.config.horizon_periods):
                threshold = self.config.profit_thresholds[i] if i < len(self.config.profit_thresholds) else 0.002

                # Calculate forward returns for this horizon
                future_prices = close_prices.shift(-horizon)
                forward_returns = (future_prices - close_prices) / close_prices

                # Create binary labels based on profit threshold
                labels_df[f'label_{horizon}p'] = (forward_returns > threshold).astype(int)

                # Also create magnitude labels for more nuanced training
                labels_df[f'magnitude_{horizon}p'] = forward_returns

            tprint_success(f"✅ Applied differentiated labeling for {len(self.config.horizon_periods)} horizons")
            return {
                'success': True,
                'labels': labels_df,
                'horizons': self.config.horizon_periods,
                'thresholds': self.config.profit_thresholds
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _optimize_feature_lookback_periods(self, market_data: pd.DataFrame, labels: pd.DataFrame) -> Dict[str, Any]:
        """Optimize feature lookback periods for 5m timeframe."""
        try:
            tprint_info("🔍 Optimizing feature lookback periods for 5m timeframe...")

            # Calculate base features with different lookback periods (shorter range for 5m)
            features_dict = {}

            # Test different lookback periods (shorter for 5m timeframe)
            for lookback in range(self.config.min_lookback_periods, self.config.max_lookback_periods + 1, 3):
                tprint_info(f"🧪 Testing lookback period: {lookback}")

                # Calculate features for this lookback period
                features_df = await self._calculate_features_for_lookback(market_data, lookback)

                # Evaluate feature importance/quality for this lookback
                quality_score = await self._evaluate_lookback_quality(features_df, labels, lookback)

                features_dict[lookback] = {
                    'features': features_df,
                    'quality_score': quality_score,
                    'feature_count': features_df.shape[1]
                }

            # Select optimal lookback period based on quality scores
            optimal_lookback = max(features_dict.keys(), key=lambda k: features_dict[k]['quality_score'])
            optimal_features = features_dict[optimal_lookback]['features']

            tprint_success(f"✅ Selected optimal lookback period: {optimal_lookback}")
            return {
                'success': True,
                'features': optimal_features,
                'optimal_periods': {optimal_lookback: features_dict[optimal_lookback]},
                'all_periods': features_dict
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _calculate_features_for_lookback(self, market_data: pd.DataFrame, lookback: int) -> pd.DataFrame:
        """Calculate features for a specific lookback period."""
        try:
            features_df = pd.DataFrame(index=market_data.index)

            if FEATURE_CALCULATORS_AVAILABLE:
                # Calculate technical indicators
                tech_features = calculate_technical_indicators(market_data, lookback)
                features_df = pd.concat([features_df, tech_features], axis=1)

                # Calculate statistical features
                stat_features = calculate_statistical_features(market_data, lookback)
                features_df = pd.concat([features_df, stat_features], axis=1)

                # Calculate price features
                price_features = calculate_price_features(market_data, lookback)
                features_df = pd.concat([features_df, price_features], axis=1)

                # Calculate volume features (if available)
                if 'volume' in market_data.columns:
                    vol_features = calculate_volume_features(market_data, lookback)
                    features_df = pd.concat([features_df, vol_features], axis=1)

            return features_df.fillna(0)  # Fill any NaN values

        except Exception as e:
            self.logger.warning(f"⚠️ Feature calculation failed for lookback {lookback}: {e}")
            return pd.DataFrame(index=market_data.index)  # Return empty dataframe

    async def _evaluate_lookback_quality(self, features: pd.DataFrame, labels: pd.DataFrame, lookback: int) -> float:
        """Evaluate the quality of features for a given lookback period."""
        try:
            # Simple correlation-based quality score
            quality_score = 0.0

            # Check correlation with labels (if labels are available)
            if not labels.empty and not features.empty:
                # Align indices
                common_indices = features.index.intersection(labels.index)
                if len(common_indices) > 50:  # Need minimum samples (lower for 5m)
                    aligned_features = features.loc[common_indices]
                    aligned_labels = labels.loc[common_indices]

                    # Calculate average correlation across label columns
                    correlations = []
                    for col in aligned_labels.columns:
                        if col.startswith('label_'):
                            corr_matrix = aligned_features.corrwith(aligned_labels[col])
                            correlations.append(corr_matrix.abs().mean())

                    quality_score = np.mean(correlations) if correlations else 0.0

            # Penalize too many NaN values
            nan_ratio = features.isnull().sum().sum() / (features.shape[0] * features.shape[1])
            quality_score *= (1 - nan_ratio)

            return quality_score

        except Exception as e:
            self.logger.warning(f"⚠️ Quality evaluation failed for lookback {lookback}: {e}")
            return 0.0

    async def _generate_pid_features(self, market_data: pd.DataFrame, base_features: pd.DataFrame) -> Dict[str, Any]:
        """Generate PID-based features for 5m timeframe."""
        try:
            tprint_info("🧬 Generating PID features for 5m timeframe...")

            pid_features_df = base_features.copy()

            if not self.config.enable_pid_features:
                return {'success': True, 'features': pid_features_df, 'feature_count': pid_features_df.shape[1]}

            # Calculate PID features for different window sizes and polynomial degrees (shorter windows for 5m)
            for window in self.config.pid_window_sizes:
                for degree in self.config.pid_polynomial_degrees:
                    try:
                        # Proportional component (current value)
                        pid_features_df[f'pid_prop_w{window}_d{degree}'] = market_data['close']

                        # Integral component (rolling mean)
                        pid_features_df[f'pid_int_w{window}_d{degree}'] = market_data['close'].rolling(window=window).mean()

                        # Derivative component (rate of change)
                        pid_features_df[f'pid_der_w{window}_d{degree}'] = market_data['close'].diff()

                        # Combined PID feature
                        pid_features_df[f'pid_combined_w{window}_d{degree}'] = (
                            pid_features_df[f'pid_prop_w{window}_d{degree}'] +
                            pid_features_df[f'pid_int_w{window}_d{degree}'] +
                            pid_features_df[f'pid_der_w{window}_d{degree}']
                        )

                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to calculate PID features for window {window}, degree {degree}: {e}")

            tprint_success(f"✅ Generated PID features with {pid_features_df.shape[1] - base_features.shape[1]} new features")
            return {
                'success': True,
                'features': pid_features_df,
                'feature_count': pid_features_df.shape[1]
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _select_final_features(self, features: pd.DataFrame, labels: pd.DataFrame) -> Dict[str, Any]:
        """Select final features for Tactician models."""
        try:
            tprint_info("🔧 Selecting final features for 5m timeframe...")

            if not self.config.enable_feature_selection:
                return {'success': True, 'features': features, 'selected_count': features.shape[1]}

            # Initialize feature selector
            if FEATURE_SELECTION_AVAILABLE:
                selector = FeatureSelectionEngine(
                    method=self.config.feature_selection_method,
                    max_features=self.config.max_features_per_model
                )

                # Select features based on labels
                if not labels.empty:
                    selected_features = await selector.select_features(features, labels)
                else:
                    selected_features = features.iloc[:, :self.config.max_features_per_model]  # Fallback selection

                tprint_success(f"✅ Selected {selected_features.shape[1]} features using {self.config.feature_selection_method}")
                return {
                    'success': True,
                    'features': selected_features,
                    'selected_count': selected_features.shape[1]
                }
            else:
                # Simple correlation-based selection as fallback
                if not labels.empty and not features.empty:
                    # Calculate correlations with label columns
                    correlations = {}
                    for col in features.columns:
                        correlations[col] = labels.iloc[:, 0].corr(features[col]) if len(labels.columns) > 0 else 0

                    # Select top features by absolute correlation
                    top_features = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
                    selected_cols = [col for col, _ in top_features[:self.config.max_features_per_model]]
                    selected_features = features[selected_cols]
                else:
                    selected_features = features

                tprint_success(f"✅ Selected {selected_features.shape[1]} features using correlation-based method")
                return {
                    'success': True,
                    'features': selected_features,
                    'selected_count': selected_features.shape[1]
                }

        except Exception as e:
            return {'success': False, 'error': str(e)}


# Convenience function for external usage
async def execute_tactician_pre_ml_orchestration(
    market_data: pd.DataFrame,
    regime_labels: Optional[pd.Series] = None,
    analyst_signals: Optional[pd.DataFrame] = None,
    config: Optional[TacticianPreMLOrchestrationConfig] = None
) -> TacticianPreMLOrchestrationResult:
    """
    Execute Tactician pre-ML orchestration.

    Args:
        market_data: Market data for 5m timeframe
        regime_labels: Regime labels for per-regime processing
        analyst_signals: Analyst signals with confidence scores
        config: Optional configuration

    Returns:
        TacticianPreMLOrchestrationResult with processed data
    """
    orchestrator = TacticianPreMLOrchestrator(config)
    return await orchestrator.orchestrate_pre_ml_processing(market_data, regime_labels, analyst_signals)