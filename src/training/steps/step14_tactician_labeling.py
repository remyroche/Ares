# src/training/steps/step14_tactician_labeling.py

"""Step 14: Regime-Aware Tactician Labeling with Regime-Specific Barriers."

This step applies regime-aware triple barrier labeling for Tactician multi-outcome predictions
with regime-specific barrier calculation, precision thresholds, and quality filters.

Enhanced for high precision completion of Analyst signals with:
- Regime-specific barrier calculation
- Per-regime precision thresholds
- Regime-specific quality filters
- Regime-aware multi-outcome prediction structure
"""

import asyncio
import contextlib
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.decorators import handles_errors
from src.training.data_sharing_manager import get_data_sharing_manager
from src.training.steps.unified_data_loader import get_unified_data_loader
from src.utils.error_handler import handle_errors
from src.utils.logger import dependency_status, system_logger

# Preference order for selecting analyst ensembles
ENSEMBLE_PREFERENCE_ORDER = ("stacking_cv", "dynamic_weighting", "voting")

class RegimeAwareTacticianLabeler:
    """Regime-aware tactician labeling with regime-specific barriers and precision thresholds."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config.get("tactician_triple_barrier", {})
        self.logger = system_logger.getChild("RegimeAwareTacticianLabeler")

        # Load enhanced configuration
        self._load_enhanced_config()

        # Regime-specific configuration
        self.regime_config = config.get("regime_specific_tactician", {
            "regime_specific_barriers": True,
            "regime_specific_precision": True,
            "regime_specific_quality_filters": True,
            "regime_specific_validation": True,
            "regime_specific_logging": True,
            "min_regime_samples": 100
        })

        # Regime-specific results storage
        self.regime_barrier_results = {}
        self.regime_labeling_results = {}
        self.regime_validation_results = {}

        self.logger.info("🎯 Regime-Aware Tactician Labeler initialized")

    def _load_enhanced_config(self) -> None:
        """Load enhanced configuration for regime-aware execution."""
        # Import dynamic barrier calculator
        from src.tactician.dynamic_barrier_calculator import DynamicBarrierCalculator

        # Initialize dynamic barrier calculator
        self.barrier_calculator = DynamicBarrierCalculator(self.config)

        # Get all 4 barrier combinations for primary timeframe (1m)
        self.barrier_combinations = self.barrier_calculator.calculate_dynamic_barriers(
            timeframe="1m"
        )
        
        # Get configuration for other settings
        self.max_lookahead = self.config.get("max_lookahead", 50)  # Reduced lookahead

        # Precision Settings
        self.enable_high_precision_mode = self.config.get("enable_high_precision_mode", True)
        self.precision_threshold = self.config.get("precision_threshold", 0.85)
        self.min_signal_strength = self.config.get("min_signal_strength", 0.8)

        # Quality Filters
        self.enable_quality_filters = self.config.get("enable_quality_filters", True)
        self.min_volume_threshold = self.config.get("min_volume_threshold", 1000)
        self.min_spread_threshold = self.config.get("min_spread_threshold", 0.0001)
        self.volatility_filter = self.config.get("volatility_filter", True)

        # Integration Settings
        self.analyst_signal_requirement = self.config.get("analyst_signal_requirement", True)
        self.direction_agreement_required = self.config.get("direction_agreement_required", True)
        self.confidence_boost_threshold = self.config.get("confidence_boost_threshold", 0.9)

        # Timeframe settings
        self.timeframes = self.config.get("timeframes", ["1m", "5m"])
        self.primary_timeframe = self.config.get("primary_timeframe", "1m")
        self.secondary_timeframe = self.config.get("secondary_timeframe", "5m")

        # Labeling mode
        self.binary_classification = self.config.get("binary_classification", True)

        # Log configuration
        self.logger.info(f"🔧 Enhanced Regime-Aware Tactician Configuration:")
        self.logger.info(f"   Timeframes: {self.timeframes}")
        self.logger.info(f"   Primary: {self.primary_timeframe}, Secondary: {self.secondary_timeframe}")
        self.logger.info(f"   Regime-specific barriers: {self.regime_config['regime_specific_barriers']}")
        self.logger.info(f"   Regime-specific precision: {self.regime_config['regime_specific_precision']}")
        self.logger.info(f"   High Precision Mode: {self.enable_high_precision_mode}")
        self.logger.info(f"   Precision Threshold: {self.precision_threshold}")

    async def apply_regime_specific_labeling(
        self, data: pd.DataFrame, regime_column: str = "composite_cluster_id"
    ) -> pd.DataFrame:
        """Apply regime-specific tactician labeling."""
        
        self.logger.info(f"🚀 Starting regime-specific tactician labeling")
        
        try:
            # Check for regime column
            if regime_column not in data.columns:
                self.logger.warning(f"⚠️ Regime column '{regime_column}' not found, using default parameters")
                return self._apply_default_labeling(data)

            labeled_data = data.copy()
            n = len(labeled_data)
            if n < 2:
                labeled_data["label"] = 0
                labeled_data["potential_profit_pct"] = 0.0
                return labeled_data

            # Extract regime information
            regime_data = labeled_data[regime_column]
            unique_regimes = regime_data.unique()

            self.logger.info(f"📊 Found {len(unique_regimes)} unique regimes: {unique_regimes}")

            # Apply regime-specific labeling
            for regime in unique_regimes:
                regime_mask = regime_data == regime
                regime_data_subset = labeled_data[regime_mask]
                
                if len(regime_data_subset) >= self.regime_config["min_regime_samples"]:
                    self.logger.info(f"🔄 Applying regime-specific labeling for regime {regime}")
                    
                    # Get regime-specific barriers
                    regime_barriers = await self._get_regime_specific_barriers(regime, regime_data_subset)
                    
                    # Apply regime-specific labeling
                    regime_labeled = await self._apply_regime_barrier_labeling(
                        regime_data_subset, regime_barriers, regime
                    )
                    
                    # Store regime-specific results
                    self.regime_labeling_results[regime] = {
                        "barriers": regime_barriers,
                        "labeled_samples": len(regime_labeled),
                        "regime": regime
                    }
                    
                    # Update main dataframe
                    labeled_data.loc[regime_mask] = regime_labeled
                else:
                    self.logger.warning(f"⚠️ Insufficient data for regime {regime}: {len(regime_data_subset)} samples")

            # Filter out HOLD samples for binary classification
            if self.binary_classification:
                original_count = len(labeled_data)
                hold_samples = (labeled_data["label"] == 0).sum()
                labeled_data = labeled_data[labeled_data["label"] != 0].copy()
                filtered_count = len(labeled_data)

                self.logger.info("📊 Label distribution after filtering:")
                self.logger.info(f"   LONG (1): {(labeled_data['label'] == 1).sum()} samples")
                self.logger.info(f"   SHORT (-1): {(labeled_data['label'] == -1).sum()} samples")
                self.logger.info(f"   HOLD (0): {hold_samples} samples (removed)")
                self.logger.info(f"   Total: {filtered_count}/{original_count} samples retained")

            return labeled_data
            
        except Exception as e:
            self.logger.error(f"❌ Error in regime-specific labeling: {e}")
            return data

    async def _get_regime_specific_barriers(
        self, regime: str, regime_data: pd.DataFrame
    ) -> Dict[str, Tuple[float, float]]:
        """Get regime-specific barriers for tactician labeling."""
        
        self.logger.info(f"🎯 Calculating regime-specific barriers for regime {regime}")
        
        try:
            if self.regime_config["regime_specific_barriers"]:
                # Calculate regime-specific barrier parameters
                regime_volatility = regime_data['close'].pct_change().std()
                regime_volume = regime_data['volume'].mean()
                regime_spread = regime_data.get('spread', pd.Series([0.0001] * len(regime_data))).mean()
                
                # Regime-specific barrier calculation
                base_upper = 0.02  # 2% default
                base_lower = 0.01  # 1% default
                
                # Adjust based on regime characteristics
                if regime_volatility > 0.02:  # High volatility regime
                    upper_multiplier = 1.5
                    lower_multiplier = 1.2
                elif regime_volatility < 0.005:  # Low volatility regime
                    upper_multiplier = 0.8
                    lower_multiplier = 0.7
                else:  # Normal volatility regime
                    upper_multiplier = 1.0
                    lower_multiplier = 1.0
                
                # Volume-based adjustments
                if regime_volume > 10000:  # High volume regime
                    upper_multiplier *= 1.1
                    lower_multiplier *= 1.1
                elif regime_volume < 1000:  # Low volume regime
                    upper_multiplier *= 0.9
                    lower_multiplier *= 0.9
                
                # Calculate final barriers
                upper_barrier = base_upper * upper_multiplier
                lower_barrier = base_lower * lower_multiplier
                
                regime_barriers = {
                    "high_precision": (upper_barrier * 0.5, lower_barrier * 0.25),
                    "standard": (upper_barrier, lower_barrier),
                    "conservative": (upper_barrier * 1.5, lower_barrier * 1.5),
                    "aggressive": (upper_barrier * 0.7, lower_barrier * 0.5)
                }
                
                self.logger.info(f"✅ Calculated regime {regime} barriers:")
                for barrier_type, (upper, lower) in regime_barriers.items():
                    self.logger.info(f"   {barrier_type}: Upper={upper:.4f} ({upper*100:.2f}%), Lower={lower:.4f} ({lower*100:.2f}%)")
                
                return regime_barriers
            else:
                # Use default barriers
                return self.barrier_combinations
                
        except Exception as e:
            self.logger.error(f"❌ Error calculating regime-specific barriers: {e}")
            return self.barrier_combinations

    async def _apply_regime_barrier_labeling(
        self, regime_data: pd.DataFrame, regime_barriers: Dict[str, Tuple[float, float]], regime: str
    ) -> pd.DataFrame:
        """Apply regime-specific barrier labeling."""
        
        self.logger.info(f"🎯 Applying regime-specific barrier labeling for regime {regime}")
        
        try:
            labeled_data = regime_data.copy()
            
            # Get regime-specific precision thresholds
            precision_thresholds = await self._get_regime_specific_precision_thresholds(regime, regime_data)
            
            # Get regime-specific quality filters
            quality_filters = await self._get_regime_specific_quality_filters(regime, regime_data)
            
            # Apply regime-specific labeling for each barrier type
            for barrier_type, (upper_barrier, lower_barrier) in regime_barriers.items():
                self.logger.info(f"🔄 Applying {barrier_type} barriers for regime {regime}")
                
                # Apply regime-specific triple barrier labeling
                regime_labeled = await self._apply_regime_triple_barrier(
                    labeled_data, upper_barrier, lower_barrier, 
                    precision_thresholds, quality_filters, regime, barrier_type
                )
                
                # Store regime-specific results
                barrier_key = f"{regime}_{barrier_type}"
                self.regime_barrier_results[barrier_key] = {
                    "barrier_type": barrier_type,
                    "upper_barrier": upper_barrier,
                    "lower_barrier": lower_barrier,
                    "precision_thresholds": precision_thresholds,
                    "quality_filters": quality_filters,
                    "labeled_samples": len(regime_labeled),
                    "regime": regime
                }
                
                labeled_data = regime_labeled
            
            return labeled_data
            
        except Exception as e:
            self.logger.error(f"❌ Error applying regime barrier labeling: {e}")
            return regime_data

    async def _get_regime_specific_precision_thresholds(
        self, regime: str, regime_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Get regime-specific precision thresholds."""
        
        try:
            if self.regime_config["regime_specific_precision"]:
                # Calculate regime-specific precision thresholds
                regime_volatility = regime_data['close'].pct_change().std()
                regime_volume = regime_data['volume'].mean()
                
                # Base precision threshold
                base_precision = 0.85
                
                # Adjust based on regime characteristics
                if regime_volatility > 0.02:  # High volatility regime
                    precision_threshold = base_precision * 0.9  # Lower threshold for high volatility
                elif regime_volatility < 0.005:  # Low volatility regime
                    precision_threshold = base_precision * 1.1  # Higher threshold for low volatility
                else:  # Normal volatility regime
                    precision_threshold = base_precision
                
                # Volume-based adjustments
                if regime_volume > 10000:  # High volume regime
                    precision_threshold *= 1.05
                elif regime_volume < 1000:  # Low volume regime
                    precision_threshold *= 0.95
                
                # Ensure threshold is within reasonable bounds
                precision_threshold = max(0.7, min(0.95, precision_threshold))
                
                precision_thresholds = {
                    "precision_threshold": precision_threshold,
                    "min_signal_strength": precision_threshold * 0.9,
                    "confidence_boost_threshold": precision_threshold * 1.05
                }
                
                self.logger.info(f"✅ Calculated regime {regime} precision thresholds:")
                for threshold_name, threshold_value in precision_thresholds.items():
                    self.logger.info(f"   {threshold_name}: {threshold_value:.3f}")
                
                return precision_thresholds
            else:
                # Use default precision thresholds
                return {
                    "precision_threshold": self.precision_threshold,
                    "min_signal_strength": self.min_signal_strength,
                    "confidence_boost_threshold": self.confidence_boost_threshold
                }
                
        except Exception as e:
            self.logger.error(f"❌ Error calculating regime-specific precision thresholds: {e}")
            return {
                "precision_threshold": self.precision_threshold,
                "min_signal_strength": self.min_signal_strength,
                "confidence_boost_threshold": self.confidence_boost_threshold
            }

    async def _get_regime_specific_quality_filters(
        self, regime: str, regime_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Get regime-specific quality filters."""
        
        try:
            if self.regime_config["regime_specific_quality_filters"]:
                # Calculate regime-specific quality filter thresholds
                regime_volume_mean = regime_data['volume'].mean()
                regime_volume_std = regime_data['volume'].std()
                regime_spread_mean = regime_data.get('spread', pd.Series([0.0001] * len(regime_data))).mean()
                
                # Volume-based quality filters
                volume_threshold = max(100, regime_volume_mean * 0.1)  # At least 10% of mean volume
                
                # Spread-based quality filters
                spread_threshold = max(0.0001, regime_spread_mean * 2)  # At most 2x mean spread
                
                # Volatility-based quality filters
                regime_volatility = regime_data['close'].pct_change().std()
                volatility_threshold = regime_volatility * 3  # 3x regime volatility
                
                quality_filters = {
                    "min_volume_threshold": volume_threshold,
                    "min_spread_threshold": spread_threshold,
                    "volatility_filter": True,
                    "volatility_threshold": volatility_threshold,
                    "enable_quality_filters": True
                }
                
                self.logger.info(f"✅ Calculated regime {regime} quality filters:")
                for filter_name, filter_value in quality_filters.items():
                    self.logger.info(f"   {filter_name}: {filter_value}")
                
                return quality_filters
            else:
                # Use default quality filters
                return {
                    "min_volume_threshold": self.min_volume_threshold,
                    "min_spread_threshold": self.min_spread_threshold,
                    "volatility_filter": self.volatility_filter,
                    "enable_quality_filters": self.enable_quality_filters
                }
                
        except Exception as e:
            self.logger.error(f"❌ Error calculating regime-specific quality filters: {e}")
            return {
                "min_volume_threshold": self.min_volume_threshold,
                "min_spread_threshold": self.min_spread_threshold,
                "volatility_filter": self.volatility_filter,
                "enable_quality_filters": self.enable_quality_filters
            }

    async def _apply_regime_triple_barrier(
        self, regime_data: pd.DataFrame, upper_barrier: float, lower_barrier: float,
        precision_thresholds: Dict[str, float], quality_filters: Dict[str, Any],
        regime: str, barrier_type: str
    ) -> pd.DataFrame:
        """Apply regime-specific triple barrier labeling."""
        
        self.logger.info(f"🎯 Applying regime-specific triple barrier ({barrier_type}) for regime {regime}")
        
        try:
            labeled_data = regime_data.copy()
            
            # Apply regime-specific quality filters
            if quality_filters.get("enable_quality_filters", True):
                labeled_data = await self._apply_regime_quality_filters(labeled_data, quality_filters, regime)
            
            # Apply regime-specific triple barrier logic
            for i in range(len(labeled_data) - 1):
                entry_price = labeled_data.iloc[i]['close']
                entry_idx = i
                
                # Calculate barriers
                profit_barrier = entry_price * (1.0 + upper_barrier)
                stop_barrier = entry_price * (1.0 - lower_barrier)
                
                # Find barrier hit
                label = 0
                profit_pct = 0.0
                
                for j in range(entry_idx + 1, min(entry_idx + self.max_lookahead, len(labeled_data))):
                    high_price = labeled_data.iloc[j]['high']
                    low_price = labeled_data.iloc[j]['low']
                    
                    # Check profit barrier first
                    if high_price >= profit_barrier:
                        label = 1  # LONG position
                        profit_pct = upper_barrier
                        break
                    
                    # Check stop barrier
                    if low_price <= stop_barrier:
                        label = -1  # SHORT position
                        profit_pct = -lower_barrier
                        break
                
                # Apply regime-specific precision threshold
                if abs(profit_pct) > 0:
                    # Check if signal meets regime-specific precision requirements
                    if abs(profit_pct) >= precision_thresholds["min_signal_strength"]:
                        labeled_data.iloc[entry_idx, labeled_data.columns.get_loc('label')] = label
                        labeled_data.iloc[entry_idx, labeled_data.columns.get_loc('potential_profit_pct')] = profit_pct
                    else:
                        # Signal too weak for regime
                        labeled_data.iloc[entry_idx, labeled_data.columns.get_loc('label')] = 0
                        labeled_data.iloc[entry_idx, labeled_data.columns.get_loc('potential_profit_pct')] = 0.0
            
            # Log regime-specific results
            long_signals = (labeled_data['label'] == 1).sum()
            short_signals = (labeled_data['label'] == -1).sum()
            hold_signals = (labeled_data['label'] == 0).sum()
            
            self.logger.info(f"📊 Regime {regime} ({barrier_type}) labeling results:")
            self.logger.info(f"   LONG signals: {long_signals}")
            self.logger.info(f"   SHORT signals: {short_signals}")
            self.logger.info(f"   HOLD signals: {hold_signals}")
            
            return labeled_data
            
        except Exception as e:
            self.logger.error(f"❌ Error applying regime triple barrier: {e}")
            return regime_data

    async def _apply_regime_quality_filters(
        self, regime_data: pd.DataFrame, quality_filters: Dict[str, Any], regime: str
    ) -> pd.DataFrame:
        """Apply regime-specific quality filters."""
        
        self.logger.info(f"🔍 Applying regime-specific quality filters for regime {regime}")
        
        try:
            filtered_data = regime_data.copy()
            
            # Volume filter
            if "volume" in filtered_data.columns:
                volume_threshold = quality_filters.get("min_volume_threshold", 1000)
                volume_mask = filtered_data['volume'] >= volume_threshold
                filtered_data = filtered_data[volume_mask]
                self.logger.info(f"   Volume filter: {len(regime_data)} -> {len(filtered_data)} samples")
            
            # Spread filter
            if "spread" in filtered_data.columns:
                spread_threshold = quality_filters.get("min_spread_threshold", 0.0001)
                spread_mask = filtered_data['spread'] <= spread_threshold
                filtered_data = filtered_data[spread_mask]
                self.logger.info(f"   Spread filter: {len(regime_data)} -> {len(filtered_data)} samples")
            
            # Volatility filter
            if quality_filters.get("volatility_filter", True):
                volatility_threshold = quality_filters.get("volatility_threshold", 0.02)
                returns = filtered_data['close'].pct_change().abs()
                volatility_mask = returns <= volatility_threshold
                filtered_data = filtered_data[volatility_mask]
                self.logger.info(f"   Volatility filter: {len(regime_data)} -> {len(filtered_data)} samples")
            
            return filtered_data
            
        except Exception as e:
            self.logger.error(f"❌ Error applying regime quality filters: {e}")
            return regime_data

    def _apply_default_labeling(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply default labeling when regime information is not available."""
        
        self.logger.info("🔄 Applying default tactician labeling")
        
        try:
            labeled_data = data.copy()
            
            # Apply default barriers
            default_barriers = self.barrier_combinations.get("standard", (0.02, 0.01))
            upper_barrier, lower_barrier = default_barriers
            
            # Apply default triple barrier logic
            for i in range(len(labeled_data) - 1):
                entry_price = labeled_data.iloc[i]['close']
                entry_idx = i
                
                # Calculate barriers
                profit_barrier = entry_price * (1.0 + upper_barrier)
                stop_barrier = entry_price * (1.0 - lower_barrier)
                
                # Find barrier hit
                label = 0
                profit_pct = 0.0
                
                for j in range(entry_idx + 1, min(entry_idx + self.max_lookahead, len(labeled_data))):
                    high_price = labeled_data.iloc[j]['high']
                    low_price = labeled_data.iloc[j]['low']
                    
                    # Check profit barrier first
                    if high_price >= profit_barrier:
                        label = 1  # LONG position
                        profit_pct = upper_barrier
                        break
                    
                    # Check stop barrier
                    if low_price <= stop_barrier:
                        label = -1  # SHORT position
                        profit_pct = -lower_barrier
                        break
                
                labeled_data.iloc[entry_idx, labeled_data.columns.get_loc('label')] = label
                labeled_data.iloc[entry_idx, labeled_data.columns.get_loc('potential_profit_pct')] = profit_pct
            
            return labeled_data
            
        except Exception as e:
            self.logger.error(f"❌ Error applying default labeling: {e}")
            return data

    def _log_regime_specific_metrics(
        self, regime: str, metrics: dict, step_name: str
    ) -> None:
        """Log regime-specific metrics."""
        
        if self.regime_config["regime_specific_logging"]:
            self.logger.info(f"📊 {step_name} - Regime {regime} metrics:")
            for metric_name, metric_value in metrics.items():
                self.logger.info(f"   {metric_name}: {metric_value}")

class TacticianLabelingStep:
    """Step 8: Tactician Model Labeling using Analyst's model."""'

    

    def _validate_environment(self) -> None:
        """Validate environment dependencies and configuration."""
        if not dependency_status["all_available"]:
            missing_modules = dependency_status["missing_modules"]
            self.logger.warning(f"Missing modules: {missing_modules}")
            # Continue with available modules, using fallbacks where needed

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger

    @handles_errors(fallback=False)
    async def initialize(self) -> None:
        """Initialize the tactician labeling step."""
        self.logger.info("🚀 Initializing Tactician Labeling Step...")

    @handles_errors
        default_return={"status": "FAILED", "error": "Execution failed"},
        context="tactician labeling step execution",
    )
    async def execute(
        self, training_input: dict[str, Any], pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute tactician model labeling."""
        try:
            self.logger.info("🔄 Executing Tactician Labeling...")

            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            data_dir = training_input.get("data_dir", "data/training")

            # Use data sharing manager to get comprehensive data for tactician labeling
            self.logger.info(
                "🔄 Loading unified data for tactician labeling via data sharing manager...",
            )
            data_sharing_manager = get_data_sharing_manager(self.config)
            timeframe = training_input.get("timeframe", "1m")

            # Load unified data with optimizations for ML training
            # Use data sharing manager to avoid redundant loading
            from src.config.constants import BLANK_TRAINING_LOOKBACK_DAYS

            # Use lookback_days from config (should be passed from enhanced training manager)
            config_lookback = self.config.get(
                "lookback_days", BLANK_TRAINING_LOOKBACK_DAYS,
            )
            data_1m = await data_sharing_manager.get_unified_data(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                lookback_days=config_lookback,
                force_reload=False,  # Use cache if available from previous steps,
            )

            if data_1m is None or data_1m.empty:
                self.logger.error(
                    f"🚨 No unified data found for {symbol} on {exchange}",
                )
                return {
                    "status": "FAILED",
                    "error": f"No unified data found for {symbol} on {exchange}",
                }

            # Log data information
            try:
                _loader = get_unified_data_loader(self.config)
                data_info = _loader.get_data_info(data_1m)
            except Exception as e:  # pragma: no cover - best effort logging
                self.logger.warning(f"⚠️ Could not get data info: {e}")
                data_info = {
                    "rows": len(data_1m) if hasattr(data_1m, "__len__") else None,
                    "columns": list(getattr(data_1m, "columns", [])) if hasattr(data_1m, "columns") else None,
                    "date_range": {"start": None, "end": None},
                    "has_aggtrades_data": False,
                    "has_futures_data": False,
                }
            
            self.logger.info(f"✅ Loaded unified data: {data_info['rows']} rows")
            with contextlib.suppress(Exception):
                self.logger.info(
                    f"   Date range: {data_info['date_range']['start']} to {data_info['date_range']['end']}",
                )
                self.logger.info(
                    f"   Has aggtrades data: {data_info['has_aggtrades_data']}",
                )
                self.logger.info(f"   Has futures data: {data_info['has_futures_data']}")

            # Ensure we have the required OHLCV columns
            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            missing_columns = [
                col for col in required_columns if col not in data_1m.columns
            ]
            if missing_columns:
                self.logger.error(f"🚨 Missing required columns: {missing_columns}")
                return {
                    "status": "FAILED",
                    "error": f"Missing required columns: {missing_columns}",
                }
            with contextlib.suppress(Exception):
                self.logger.info(
                    f"Loaded 1m data: shape={getattr(data_1m, 'shape', None)}, columns={list(getattr(data_1m, 'columns', [])[:10])}"
                )

            # Load analyst ensemble models
            analyst_ensembles = self._load_analyst_ensembles(data_dir)

            # Generate strategic "setup" signals using analyst models
            (
                data_with_features,
                strategic_signals,
            ) = await self._generate_strategic_signals(data_1m, analyst_ensembles)

            # Apply the specialized Tactician Triple Barrier
            labeler = RegimeAwareTacticianLabeler(self.config)
            labeled_data = await labeler.apply_regime_specific_labeling(
                data_with_features, "composite_cluster_id"
            )
            with contextlib.suppress(Exception):
                self.logger.info(
                    f"Strategic signals summary: total={len(strategic_signals)}, nonzero={(strategic_signals != 0).sum()}"
                )

            # Save results
            labeled_file, signals_file = self._save_results(
                labeled_data,
                strategic_signals,
                data_dir,
                exchange,
                symbol,
            )

            self.logger.info(
                f"✅ Tactician labeling completed. Labeled data saved to {labeled_file}",
            )

            pipeline_state["tactician_labeled_data"] = labeled_data
            return {
                "status": "SUCCESS",
                "labeled_file": labeled_file,
                "signals_file": signals_file,
            }
        except Exception as e:  # pragma: no cover - defensive
            self.logger.exception(f"❌ Error in Tactician Labeling: {e}")
            return {"status": "FAILED", "error": str(e)}

    def _load_analyst_ensembles(self, data_dir: str) -> dict[str, Any]:
        """Loads all trained analyst ensemble models."""
        analyst_ensembles_dir = f"{data_dir}/analyst_ensembles"
        analyst_ensembles: dict[str, Any] = {}
        if not Path(analyst_ensembles_dir).exists():
            msg = f"Analyst ensembles directory not found: {analyst_ensembles_dir}"
            raise FileNotFoundError(
                msg,
            )

        for ensemble_file in os.listdir(analyst_ensembles_dir):
            if ensemble_file.endswith("_ensemble.pkl"):
                regime_name = ensemble_file.replace("_ensemble.pkl", "")
                ensemble_path = Path(analyst_ensembles_dir) / ensemble_file
                with ensemble_path.open("rb") as f:
                    loaded = pickle.load(f)
                chosen_ensemble: Any = None
                if isinstance(loaded, dict):
                    # Prefer stacking_cv, then dynamic_weighting, then voting
                    for key in ENSEMBLE_PREFERENCE_ORDER:
                        if key in loaded and isinstance(loaded[key], dict):
                            obj = loaded[key].get("ensemble")
                            if obj is not None:
                                chosen_ensemble = obj
                                break
                    if chosen_ensemble is None:
                        # Fallback if saved dict is a single-ensemble payload
                        chosen_ensemble = (
                            loaded.get("ensemble") if "ensemble" in loaded else None
                        )
                # Record whatever we found (could be None; upstream handles None)
                analyst_ensembles[regime_name] = chosen_ensemble
        return analyst_ensembles

    async def _generate_strategic_signals(
        self, data: pd.DataFrame, analyst_ensembles: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Generate strategic signals using analyst ensemble models."""
        self.logger.info("Generating strategic 'setup' signals from Analyst models...")

        # Step 1: Calculate all features needed for any of the analyst models
        data_with_features = self._calculate_features(data)

        # Step 2: Determine the market regime for each data point
        # This is a placeholder for your regime detection logic (e.g., from step 4)
        # It is crucial that this logic is consistent with how regimes were defined during Analyst training.
        data_with_features["regime"] = self._get_market_regime(data_with_features)

        all_signals = pd.Series(0, index=data_with_features.index)

        # Step 3: Predict in a vectorized way for each regime
        for regime_name, ensemble in analyst_ensembles.items():
            if ensemble is None:
                continue

            regime_mask = data_with_features["regime"] == regime_name
            if not regime_mask.any():
                continue

            # Ensure the model's expected features are present'
            if hasattr(ensemble, "feature_names_in_"):
                features_for_model = [
                    f
                    for f in getattr(ensemble, "feature_names_in_", [])
                    if f in data_with_features.columns
                ]
                x_regime = data_with_features.loc[regime_mask, features_for_model]
            else:
                # Fallback if feature names are not stored in the model
                x_regime = data_with_features.loc[regime_mask].select_dtypes(
                    include=np.number,
                )

            if not x_regime.empty:
                predictions = ensemble.predict(x_regime)
                all_signals[regime_mask] = predictions

        self.logger.info(
            f"Generated strategic signals. Signal distribution:\n{all_signals.value_counts()}",
        )
        return data_with_features, all_signals

    def _get_market_regime(self, data: pd.DataFrame) -> pd.Series:
        """Placeholder for your market regime detection logic."
        This should be consistent with the logic from step4_regime_specific_training.
        """
        # Example: Simple regime based on volatility percentile
        # NOTE: Volatility is calculated here because the Analyst models need it for regime detection.
        # It is NOT used by the Tactician's labeler.'
        vol_percentile = data["volatility"].rank(pct=True)
        bins = [0, 0.33, 0.66, 1.0]
        labels = ["SIDEWAYS", "BULL", "BEAR"]
        regimes = pd.cut(vol_percentile, bins=bins, labels=labels, right=False)
        return regimes.astype(str).fillna("SIDEWAYS")

    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all necessary features for both Analyst and Tactician."""
        data = data.copy()
        data["returns"] = data["close"].pct_change()
        # Volatility is calculated here for the Analyst's regime detection, not for Tactician labeling.'
        data["volatility"] = (
            data["returns"].rolling(window=60).std().bfill()
        )  # 1-hour volatility
        # ... Add all other features your Analyst models were trained on ...
        # e.g., RSI, MACD, Bollinger Bands, etc.
        return data.fillna(method="ffill").fillna(0)

    def _save_results(self, labeled_data: pd.DataFrame, signals: pd.Series, data_dir: str, exchange: str, symbol: str) -> Tuple[str, str]:
        """Saves the labeled data and signals to disk."""
        labeled_data_dir = f"{data_dir}/tactician_labeled_data"
        Path(labeled_data_dir).mkdir(parents=True, exist_ok=True)

        # Prefer Parquet for DataFrame/Series persistence
        labeled_file_parquet = (
            f"{labeled_data_dir}/{exchange}_{symbol}_tactician_labeled.parquet"
        )
        try:
            try:
                from src.training.enhanced_training_manager_optimized import (
                    ParquetDatasetManager,
                )

                ParquetDatasetManager(logger=self.logger).write_flat_parquet(
                    labeled_data,
                    labeled_file_parquet,
                    schema_name="split",
                    compression="snappy",
                    use_dictionary=True,
                    row_group_size=128_000,
                )
            except Exception:
                from src.utils.logger import log_dataframe_overview, log_io_operation

                with log_io_operation(
                    self.logger,
                    "to_parquet",
                    labeled_file_parquet,
                    compression="snappy",
                ):
                    labeled_data.to_parquet(
                        labeled_file_parquet, compression="snappy", index=False
                    )
                with contextlib.suppress(Exception):
                    log_dataframe_overview(
                        self.logger, labeled_data, name="labeled_data"
                    )
        except Exception:
            # Fallback to Pickle for compatibility
            labeled_file_pickle = (
                f"{labeled_data_dir}/{exchange}_{symbol}_tactician_labeled.pkl"
            )
            labeled_data.to_pickle(labeled_file_pickle)
            labeled_file_parquet = labeled_file_pickle

        signals_file_parquet = (
            f"{data_dir}/{exchange}_{symbol}_strategic_signals.parquet"
        )
        try:
            # Save Series as Parquet by converting to DataFrame
            _signals_df = signals.to_frame(name="signal").reset_index()
            try:
                from src.training.enhanced_training_manager_optimized import (
                    ParquetDatasetManager,
                )

                ParquetDatasetManager(logger=self.logger).write_flat_parquet(
                    _signals_df,
                    signals_file_parquet,
                    schema_name="split",
                    compression="snappy",
                    use_dictionary=True,
                    row_group_size=128_000,
                )
            except Exception:
                from src.utils.logger import log_dataframe_overview, log_io_operation

                with log_io_operation(
                    self.logger,
                    "to_parquet",
                    signals_file_parquet,
                    compression="snappy",
                ):
                    _signals_df.to_parquet(
                        signals_file_parquet, compression="snappy", index=False
                    )
                with contextlib.suppress(Exception):
                    log_dataframe_overview(self.logger, _signals_df, name="signals_df")
        except Exception:
            signals_file_pickle = (
                f"{data_dir}/{exchange}_{symbol}_strategic_signals.pkl"
            )
            signals.to_pickle(signals_file_pickle)
            signals_file_parquet = signals_file_pickle

        return labeled_file_parquet, signals_file_parquet

# Import training pipeline decorators for comprehensive security and troubleshooting
from src.utils.centralized_decorators import (
    artifact_versioning,
    artifact_write_lock,
    circuit_breaker_protection,
    debug_training_step,
    deterministic_seed,
    idempotent_step,
    memory_efficient,
    nan_inf_and_constant_guard,
    prevent_data_leakage,
    quality_gate,
    resource_monitor,
    secure_data_processing,
    time_budget_watchdog,
    validate_step_output,
    validate_step_prerequisites,
)
from src.utils.enhanced_mlflow_integration import (
    copy,
    create_detailed_step_report,
    log_step_artifact_with_standardized_name,
    log_step_dataframe_with_standardized_name,
    log_step_metrics,
    log_step_report,
    with_enhanced_mlflow_logging,
)

# For backward compatibility with existing step structure
@deterministic_seed(42)
@idempotent_step(step_key="step8_tactician_labeling")
@artifact_write_lock()
@nan_inf_and_constant_guard()
@artifact_versioning("1.0")
@time_budget_watchdog(soft_timeout_seconds=2400.0)
@validate_step_prerequisites(
    required_directories=["data/training"],
    min_memory_gb=4.0,
    min_disk_gb=3.0,
    required_packages=["pandas", "numpy", "sklearn"],
    data_quality_checks={
        "min_rows": 1000,
        "required_columns": ["timestamp", "open", "high", "low", "close", "volume"],
    },
    context="Tactician Labeling",
)
@secure_data_processing(
    backup_before=True, integrity_checks=True, memory_cleanup=True, data_validation=True,
)
@prevent_data_leakage(
    temporal_validation=True,
    feature_leakage_detection=True,
    lookahead_bias_prevention=True,
)
@resource_monitor(
    memory_threshold_gb=8.0,
    cpu_threshold_percent=80.0,
    disk_threshold_gb=5.0,
    monitor_interval=30.0,
    auto_cleanup=True,
)
@memory_efficient(
    chunk_size=20000, streaming_processing=True, memory_pool=True, cleanup_frequency=40,
)
@debug_training_step(
    log_intermediate_results=True,
    save_debug_artifacts=True,
    performance_profiling=True,
    error_context_preservation=True,
)
@circuit_breaker_protection(
    failure_threshold=3,
    recovery_timeout=120.0,
    expected_exception=Exception,
    monitor_interval=30.0,
)
@validate_step_output(
    required_files=["data/training/{exchange}_{symbol}_tactician_labels.parquet"],
    data_quality_checks={
        "min_rows": 100,
        "required_columns": ["timestamp", "label", "signal"],
    },
    performance_thresholds={"labeling_time_minutes": 45.0},
    format_validation=True,
)
@quality_gate(
    data_quality_metrics={"completeness": 0.9, "consistency": 0.8},
    validation_score_requirements={"labeling_accuracy": 0.7},
)
async def run_step(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    force_rerun: bool = False,
    **kwargs: Any,
) -> bool:
    """Run the tactician labeling step."

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory path
        **kwargs: Additional parameters

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create step instance
        config: dict[str, Any] = {"symbol": symbol, "exchange": exchange, "data_dir": data_dir}
        step = TacticianLabelingStep(config)
        await step.initialize()

        # Execute step
        training_input: dict[str, Any] = {
            "symbol": symbol,
            "exchange": exchange,
            "data_dir": data_dir,
            "force_rerun": force_rerun,
            **kwargs,
        }

        pipeline_state: dict[str, Any] = {}
        result = await step.execute(training_input, pipeline_state)

        return result.get("status") == "SUCCESS"

    except Exception:  # pragma: no cover - defensive
        return False

if __name__ == "__main__":
    # Test the step
    async def test() -> None:
        await run_step("ETHUSDT", "BINANCE", "data/training")

    asyncio.run(test())