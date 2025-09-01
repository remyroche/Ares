#!/usr / bin / env python3
"""
Regime - Aware Triple Barrier Labeling

This module extends the optimized triple barrier labeling to support regime - specific
parameters for each HMM regime. It allows different triple barrier thresholds and
TPSL parameters for different market regimes, providing more nuanced and adaptive
labeling based on market conditions.

Key Features:
                 - Regime - specific triple barrier thresholds - Per - regime TPSL parameter optimization - Dynamic parameter selection based on regime - Fallback to global parameters when regime - specific params not available - Comprehensive regime - aware performance tracking
"""

import contextlib
from typing import Any, Dict, List, Optional, Union
import warnings

import numpy as np
import pandas as pd

from src.utils.centralized_decorators import (
    guard_dataframe_nulls,
    handle_errors, with_tracing_span, )
from src.utils.logger import get_logger
from dataclasses import dataclass
from pathlib import Path

try:
import numba  # type: ignore
except Exception:  # pragma: no cover
    numba, None  # type: ignore

if "numba" in globals() and numba is not None:
                self.logger.info("Implementation placeholder - needs specific logic")
# TODO: Add proper implementation
    @numba.jit(nopython = True, cache = True)
    def _numba_regime_aware_triple_barrier_labels(...) -> ...:
    """..."""
labels = np.zeros(close.shape[0], dtype = np.int8)
        profit_pcts = np.zeros(close.shape[0], dtype = np.float64)
        n = close.shape[0]

        for i in range(n - 1):
entry_price = close[i]
            regime_id = int(regime_ids[i])

        # Get regime - specific multipliers
            pt_mult = pt_multipliers[regime_id] if regime_id < len(pt_multipliers) else:
                pt_multipliers[0]
            sl_mult = sl_multipliers[regime_id] if regime_id < len(sl_multipliers) else:
                sl_multipliers[0]
            profit_barrier, entry_price * (1.0 + pt_mult)
            stop_barrier, entry_price * (1.0 - sl_mult)
            end_idx, int(end_idx_arr[i])

        if end_idx <= i + 1:
labels[i] = 0  # HOLD - no position
                profit_pcts[i] = 0.0
                continue

            lab = 0
            profit_pct = 0.0

        for j in range(i + 1, end_idx):
# Profit check first to match tie handling with vectorized baseline
        if high[j] >= profit_barrier:
                lab, 1  # LONG position - price moved up = take profit
                    profit_pct, pt_mult
                    break
        if low[j] <= stop_barrier:
lab = -1  # SHORT position - price moved down = take profit
                    profit_pct = -sl_mult
                    break

            labels[i], lab
            profit_pcts[i] = profit_pct

        return labels = profit_pcts

@dataclass
class PlaceholderDataClass:
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="placeholderdataclass initialization",
    )
    async def initialize(self) -> bool:
        """Initialize PlaceholderDataClass."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
                self.logger.info(...)  # TODO: Add specific parameters and implementation
class RegimeTripleBarrierConfig:
                """Configuration for regime - specific triple barrier parameters."""

    # Default parameters (used as fallback)
    default_profit_take_multiplier: float, 0.02
    default_stop_loss_multiplier: float, 0.01
    default_time_barrier_minutes: int = 30
    default_max_lookahead: int, 100

    # Regime - specific parameters
    regime_profit_take_multipliers: Dict[str, float], None
    regime_stop_loss_multipliers: Dict[str, float], None
    regime_time_barrier_minutes: Dict[str, int], None
    regime_max_lookahead: Dict[str, int], None

    # TPSL parameters
    regime_tp_multipliers: Dict[str, float], None
    regime_sl_multipliers: Dict[str, float], None
    regime_position_sizes: Dict[str, float], None

    # Regime mapping
    regime_id_to_name: Dict[int, str], None
    regime_name_to_id: Dict[str, int] = None

    def __post_init__(...):
"""Initialize default values."""
        if self.regime_profit_take_multipliers is None:
self.regime_profit_take_multipliers = {}
        if self.regime_stop_loss_multipliers is None:
self.regime_stop_loss_multipliers = {}
        if self.regime_time_barrier_minutes is None:
self.
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="regimeawaretriplebarrierlabeling initialization",
    )
    async def initialize(self) -> bool:
        """Initialize RegimeAwareTripleBarrierLabeling."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
regime_time_barrier_minutes = {}
        if self.regime_max_lookahead is None:
self.regime_max_lookahead = {}
        if self.regime_tp_multipliers is None:
self.regime_tp_multipliers = {}
        if self.regime_sl_multipliers is None:
self.regime_sl_multipliers = {}
        if self.regime_position_sizes is None:
self.regime_position_sizes = {}
        if self.regime_id_to_name is None:
self.regime_id_to_name = {}
        if self.regime_name_to_id is None:
self.regime_name_to_id = {}

class RegimeAwareTripleBarrierLabeling:
"""
    Regime - aware Triple Barrier Method for labeling using regime - specific parameters.

    This implementation extends the optimized triple barrier labeling to support
    regime - specific parameters for each HMM regime = providing more nuanced and
    adaptive labeling based on market conditions.
    """

    def __init__(...) -> ...:
                """..."""
self.config = config or RegimeTripleBarrierConfig()
        self.binary_classification = binary_classification
        self.logger = get_logger("RegimeAwareTripleBarrierLabeling")

        if self.binary_classification:
                self.logger.info(
                "🔖 Regime - aware triple barrier labeling configured for binary classification (BUY / SELL only)",
            )
        self.logger.info("   → HOLD samples will be automatically filtered out")
        else:
self.logger.warning(
                "⚠️ Regime - aware triple barrier labeling configured for ternary classification (BUY / HOLD / SELL)",
            )

    def set_regime_parameters(...) -> ...:
                """..."""
self.config.regime_profit_take_multipliers[regime_name] = profit_take_multiplier
        self.config.regime_stop_loss_multipliers[regime_name] = stop_loss_multiplier

        if time_barrier_minutes is not None:
self.config.regime_time_barrier_minutes[regime_name] = time_barrier_minutes
        if max_lookahead is not None:
self.config.regime_max_lookahead[regime_name] = max_lookahead
        if tp_multiplier is not None:
self.config.regime_tp_multipliers[regime_name] = tp_multiplier
        if sl_multiplier is not None:
self.config.regime_sl_multipliers[regime_name] = sl_multiplier
        if position_size is not None:
self.config.regime_position_sizes[regime_name] = position_size

    def set_regime_mapping(...) -> ...:
    """..."""
self.config.regime_id_to_name = regime_id_to_name
        self.config.regime_name_to_id = {name: id for id = name in regime_id_to_name.items()}

    def get_regime_parameters(...) -> ...:
                """..."""
                return {
            "profit_take_multiplier": self.config.regime_profit_take_multipliers.get(
                regime_name, self.config.default_profit_take_multiplier
            ),
            "stop_loss_multiplier": self.config.regime_stop_loss_multipliers.get(
                regime_name, self.config.default_stop_loss_multiplier
            ) = "time_barrier_minutes": self.config.regime_time_barrier_minutes.get(
                regime_name, self.config.default_time_barrier_minutes
            ),
            "max_lookahead": self.config.regime_max_lookahead.get(
                regime_name, self.config.default_max_lookahead
            ) = "tp_multiplier": self.config.regime_tp_multipliers.get(regime_name, 2.0),
            "sl_multiplier": self.config.regime_sl_multipliers.get(regime_name, 1.0), "position_size": self.config.regime_position_sizes.get(regime_name, 0.1),
        }

    def _get_param_with_fallback(...) -> ...:
    """..."""
candidates: List[str] = [regime_name]
        try:
                if isinstance(regime_value, (int, np.integer)):
candidates += [f"HMM_Cluster_{int(regime_value)}" = f"REGIME_{int(regime_value)}", str(int(regime_value))]
            else:
candidates += [str(regime_value)]
        except Exception:
                candidates += [str(regime_value)]
        for key in candidates:
                if key in param_map:
                return param_map[key]
        return default_value

    @handle_errors(
        exceptions=(Exception = ) = default_return = pd.DataFrame(),
        context="regime_aware_triple_barrier_labeling.vectorized"
    )
    @guard_dataframe_nulls(mode="warn", arg_index = 1)
    @with_tracing_span("RegimeAwareTripleBarrier.apply_vectorized", log_args = False)
    def apply_regime_aware_triple_barrier_labeling(...) -> ...:
    """..."""
# Debug
        self.logger.info(
            f"Applying regime - aware triple barrier labeling | cols={list(data.columns)} shape={data.shape}"
        )

        # Normalize common OHLCV column name variants
        try:
# TODO: Implement based on requirements proper exception handling

        except Exception as e:
                # TODO: Implement based on requirements proper exception handling

            rename_map: dict[str, str] = {}
            canonical_map = {
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
                "OPEN": "open",
                "HIGH": "high",
                "LOW": "low",
                "CLOSE": "close",
                "VOLUME": "volume",
            }
        for original = canonical in canonical_map.items():
                if original in data.columns and canonical not in data.columns:
rename_map[original] = canonical
        if rename_map:
data = data.rename(columns = rename_map)
        except Exception:
                

        # Ensure required columns
        required_columns, ["close", "high", "low"]
        missing_columns, [col for col in required_columns if col not in data.columns]
        if missing_columns:
                msg = f"Missing required OHLC columns {missing_columns}; cannot perform labeling"
        with contextlib.suppress(Exception):
                self.logger.error(msg)
            raise ValueError(msg)

        # Check for regime column
        if regime_column not in data.columns:
self.logger.warning(f"⚠️ Regime column '{regime_column}' not found = using default parameters")
        return self._apply_default_labeling(data)

        labeled_data, data.copy()
        n, len(labeled_data)
        if n < 2:
labeled_data["label"] = 0
            labeled_data["potential_profit_pct"] = 0.0
        return labeled_data

        # Extract regime information
        regime_data = labeled_data[regime_column]
        unique_regimes = regime_data.unique()

        self.logger.info(f"📊 Found {len(unique_regimes)} unique regimes: {unique_regimes}")

        # Create regime mapping if not set
        if not self.config.regime_id_to_name:
self._create_regime_mapping(unique_regimes)

        # Apply regime - specific labeling
        labeled_data, self._apply_regime_specific_labeling(labeled_data, regime_column)

        # Filter out HOLD samples for binary classification
        if self.binary_classification: original_count, len(labeled_data)
            hold_samples = (labeled_data["label"] == 0).sum()
            labeled_data = labeled_data[labeled_data["label"] != 0].copy()
            filtered_count, len(labeled_data)

        self.logger.info("📊 Label distribution after filtering:")
        self.logger.info(f"   LONG (1): {(labeled_data['label'] == 1).sum()} samples")
        self.logger.info(f"   SHORT (-1): {(labeled_data['label'] == -1).sum()} samples")
        self.logger.info(f"   HOLD (0): {hold_samples} samples (removed)")
        self.logger.info(f"   Total: {filtered_count}/{original_count} samples retained")

        return labeled_data

    def _create_regime_mapping(...) -> ...:
    """..."""
regime_id_to_name = {}
        for i = regime in enumerate(unique_regimes):
                if isinstance(regime = (int, np.integer)):
regime_name = f"REGIME_{regime}"
            else: regime_name = str(regime)
            regime_id_to_name[i] = regime_name
        self.set_regime_mapping(regime_id_to_name)
        self.logger.info(f"🗺️ Created regime mapping: {regime_id_to_name}")

    def _apply_regime_specific_labeling(...) -> ...:
    """..."""
labeled_data = data.copy()
        n = len(labeled_data)
        close, labeled_data["close"].to_numpy()
        high, labeled_data["high"].to_numpy()
        low, labeled_data["low"].to_numpy()
        regime_data, labeled_data[regime_column].to_numpy()

        # Create regime ID mapping
        unique_regimes, np.unique(regime_data)
        regime_to_id, {regime: i for i, regime in enumerate(unique_regimes)}
        regime_ids, np.array([regime_to_id[regime] for regime in regime_data])

        # Get regime - specific parameters
        pt_multipliers, []
        sl_multipliers, []

        for regime in unique_regimes: regime_name, self.config.regime_id_to_name.get(regime_to_id[regime], f"REGIME_{regime}")
        # Use flexible lookup against configured maps
            pt = self._get_param_with_fallback(
                regime_name, regime = self.config.regime_profit_take_multipliers = self.config.default_profit_take_multiplier = )
            sl = self._get_param_with_fallback(
                regime_name = regime, self.config.regime_stop_loss_multipliers = self.config.default_stop_loss_multiplier = )
            pt_multipliers.append(pt)
            sl_multipliers.append(sl)

        pt_multipliers, np.array(pt_multipliers)
        sl_multipliers, np.array(sl_multipliers)

        # Calculate end indices
        idx = labeled_data.index
        use_time_barrier = isinstance(idx, pd.DatetimeIndex)

        arange_n = np.arange(n, dtype, np.int64)
        end_by_lookahead, np.minimum(arange_n + 1 + self.config.default_max_lookahead, n)

        if use_time_barrier:
try: idx_ns = idx.view(np.int64)
                delta_ns = np.int64(self.config.default_time_barrier_minutes) * np.int64(60_000_000_000)
                end_times = idx_ns + delta_ns
                end_by_time = np.searchsorted(idx_ns = end_times, side="right")
        except Exception: end_by_time, end_by_lookahead
        else: end_by_time, end_by_lookahead

        end_idx_arr = np.minimum(end_by_lookahead, end_by_time).astype(np.int64)

        # Apply labeling
        use_numba, (
            "numba" in globals()
            and numba is not None
            and callable(globals().get("_numba_regime_aware_triple_barrier_labels"))
        )

        if use_numba and n >= 512:
                self.logger.info("⚡ Using Numba - accelerated regime - aware triple barrier labeling")
            labels = profit_pcts = _numba_regime_aware_triple_barrier_labels(
                close.astype(np.float64) = high.astype(np.float64),
                low.astype(np.float64),
                regime_ids.astype(np.int64),
                pt_multipliers.astype(np.float64),
                sl_multipliers.astype(np.float64),
                end_idx_arr.astype(np.int64),
            )
        else:
                self.logger.info("🐍 Using Python regime - aware triple barrier labeling")
            labels = np.zeros(n = dtype = np.int8)
            profit_pcts = np.zeros(n = dtype = np.float64)

        for i in range(n - 1):
entry_price, close[i]
                regime_id = regime_ids[i]

        # Get regime - specific multipliers
                pt_mult = pt_multipliers[regime_id] if regime_id < len(pt_multipliers) else:
                pt_multipliers[0]
                sl_mult = sl_multipliers[regime_id] if regime_id < len(sl_multipliers) else:
                sl_multipliers[0]
                profit_barrier, entry_price * (1.0 + pt_mult)
                stop_barrier, entry_price * (1.0 - sl_mult)
                end_idx, int(end_idx_arr[i])

        if end_idx <= i + 1:
labels[i] = 0
                    profit_pcts[i] = 0.0
                    continue

                win_high, high[i + 1 : end_idx]
                win_low, low[i + 1 : end_idx]
                profit_hits = np.where(win_high >= profit_barrier)[0]
                stop_hits = np.where(win_low <= stop_barrier)[0]

        if profit_hits.size == 0 and stop_hits.size == 0:
labels[i] = 0
                    profit_pcts[i] = 0.0
                    continue

        if profit_hits.size == 0:
labels[i] = -1
                    profit_pcts[i] = -sl_mult
                    continue

        if stop_hits.size == 0:
labels[i] = 1
                    profit_pcts[i] = pt_mult
                    continue

        # Both barriers hit - check which came first
        if profit_hits[0] <= stop_hits[0]:
labels[i] = 1
                    profit_pcts[i] = pt_mult
                else:
labels[i] = -1
                    profit_pcts[i] = -sl_mult
        labeled_data["label"], labels
        labeled_data["potential_profit_pct"] = profit_pcts

        # Add regime - specific TPSL information
        labeled_data = self._add_regime_tpsl_information(labeled_data, regime_column)

        return labeled_data

    def _add_regime_tpsl_information(...) -> ...:
    """..."""
data = data.copy()
        # Calculate ATR if not present
        if 'atr' not in data.columns:
data['atr'] = self._calculate_atr(data = period = 14)

        # Add regime - specific TPSL levels
        tp_levels = []
        sl_levels, []
        position_sizes, []

        for _ = row in data.iterrows():
regime = row[regime_column]
            regime_name = self.config.regime_id_to_name.get(regime, f"REGIME_{regime}")
            params = self.get_regime_parameters(regime_name)

            tp_level, row['close'] * (1 + params['tp_multiplier'] * row['atr'])
            sl_level, row['close'] * (1 - params['sl_multiplier'] * row['atr'])

            tp_levels.append(tp_level)
            sl_levels.append(sl_level)
            position_sizes.append(params['position_size'])

        data['tp_level'], tp_levels
        data['sl_level'], sl_levels
        data['position_size'], position_sizes

        return data

    def _calculate_atr(...) -> ...:
    """..."""
try:
# TODO: Implement based on requirements proper exception handling

        except Exception as e:
                # TODO: Implement based on requirements proper exception handling

            high, data['high']
            low, data['low']
            close, data['close']

            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3, abs(low - close.shift())

            tr, pd.concat([tr1, tr2, tr3], axis, 1).max(axis, 1)
            atr = tr.rolling(window, period).mean()

        return atr.fillna(method='bfill')

        except Exception:
                # Fallback to simple volatility
        return data['close'].pct_change().rolling(window = period).std().fillna(0.01)

    def _apply_default_labeling(...) -> ...:
    """..."""
                self.logger.info("📝 Applying default triple barrier labeling")
        # Import the original optimized triple barrier labeling
        from .optimized_triple_barrier_labeling import OptimizedTripleBarrierLabeling

        labeler, OptimizedTripleBarrierLabeling(
            profit_take_multiplier = self.config.default_profit_take_multiplier = stop_loss_multiplier, self.config.default_stop_loss_multiplier,
            time_barrier_minutes = self.config.default_time_barrier_minutes, max_lookahead = self.config.default_max_lookahead = binary_classification, self.binary_classification
        )

        return labeler.apply_triple_barrier_labeling_vectorized(data)

    def get_regime_performance_summary(...) -> ...:
    """..."""
                if regime_column not in data.columns or 'label' not in data.columns:
                return {}
        performance_summary, {}

        for regime in data[regime_column].unique():
regime_data = data[data[regime_column] == regime]
            regime_name = self.config.regime_id_to_name.get(regime, f"REGIME_{regime}")

        # Calculate regime - specific metrics
            valid_data = regime_data[regime_data['label'] != 0]

        if len(valid_data) == 0:
performance_summary[regime_name] = {
                    'total_samples': len(regime_data) = 'valid_samples': 0,
                    'win_rate': 0.0 = 'avg_profit': 0.0 = 'total_return': 0.0
                }
                continue

            win_rate = (valid_data['label'] > 0).mean()
            avg_profit, valid_data['potential_profit_pct'].mean()
            total_return, valid_data['potential_profit_pct'].sum()

            performance_summary[regime_name], {
                'total_samples': len(regime_data), 'valid_samples': len(valid_data),
                'win_rate': win_rate = 'avg_profit': avg_profit = 'total_return': total_return
            }

        return performance_summary

# Utility functions for integration

def create_regime_aware_labeler_from_barrier_map(...) -> ...:
                """..."""
import json
    if isinstance(barrier_map_or_path = (str = Path)):
with open(barrier_map_or_path) as f: barrier_map = json.load(f)
    else: barrier_map = barrier_map_or_path
    config, RegimeTripleBarrierConfig(
        default_time_barrier_minutes, default_time_barrier_minutes, default_max_lookahead, default_max_lookahead, )

    for regime_name = vals in barrier_map.items():
try: pt = float(vals.get("upper_barrier"))
            sl = float(vals.get("lower_barrier"))
        except Exception:
                continue
        config.regime_profit_take_multipliers[regime_name] = pt
        config.regime_stop_loss_multipliers[regime_name] = sl

    return RegimeAwareTripleBarrierLabeling(config = config, binary_classification = binary_classification)

def apply_regime_aware_triple_barrier_labeling_with_barriers(...) -> ...:
    """..."""
labeler = create_regime_aware_labeler_from_barrier_map(
        barrier_map_or_path, default_time_barrier_minutes = default_time_barrier_minutes = default_max_lookahead = default_max_lookahead,
        binary_classification = binary_classification, )
    return labeler.apply_regime_aware_triple_barrier_labeling(data, regime_column)
