#!/usr/bin/env python3
"""
Regime-Aware Triple Barrier Labeling

This module extends the optimized triple barrier labeling to support regime-specific
parameters for each HMM regime. It allows different triple barrier thresholds and
TPSL parameters for different market regimes, providing more nuanced and adaptive
labeling based on market conditions.

Key Features:
- Regime-specific triple barrier thresholds
- Per-regime TPSL parameter optimization
- Dynamic parameter selection based on regime
- Fallback to global parameters when regime-specific params not available
- Comprehensive regime-aware performance tracking
"""
import contextlib
from typing import Any, Dict, List, Optional, Union
import warnings

import numpy as np
import pandas as pd

from src.core.domain import (
    guard_dataframe_nulls,
    handle_errors,
    with_tracing_span
)
from src.utils.logger import get_logger

try:
    import numba  # type: ignore
except Exception:  # pragma: no cover
    numba = None  # type: ignore

if "numba" in globals() and numba is not None:
    pass  # TODO: Add proper implementation
    @numba.jit(nopython=True, cache=True)
    def _numba_regime_aware_triple_barrier_labels(
        close: np.ndarray, 
        high: np.ndarray, 
        low: np.ndarray, 
        regime_ids: np.ndarray,
        pt_multipliers: np.ndarray, 
        sl_multipliers: np.ndarray, 
        end_idx_arr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Numba-accelerated regime-aware triple barrier labeling with profit tracking."
        
        Args:
            close: Close prices
            high: High prices
            low: Low prices
            regime_ids: Regime IDs for each point
            pt_multipliers: Profit take multipliers for each regime
            sl_multipliers: Stop loss multipliers for each regime
            end_idx_arr: End indices for each point
            
        Returns:
            labels: 1 for LONG position, -1 for SHORT position, 0 for HOLD
            profit_pcts: Actual profit/loss percentages at barrier hits
        """
        labels = np.zeros(close.shape[0], dtype=np.int8)
        profit_pcts = np.zeros(close.shape[0], dtype=np.float64)
        n = close.shape[0]
        
        for i in range(n - 1):
            entry_price = close[i]
            regime_id = int(regime_ids[i])
            
            # Get regime-specific multipliers
            pt_mult = pt_multipliers[regime_id] if regime_id < len(pt_multipliers) else pt_multipliers[0]
            sl_mult = sl_multipliers[regime_id] if regime_id < len(sl_multipliers) else sl_multipliers[0]
            
            profit_barrier = entry_price * (1.0 + pt_mult)
            stop_barrier = entry_price * (1.0 - sl_mult)
            end_idx = int(end_idx_arr[i])
            
            if end_idx <= i + 1:
                labels[i] = 0  # HOLD - no position
                profit_pcts[i] = 0.0
                continue
                
            lab = 0
            profit_pct = 0.0
            
            for j in range(i + 1, end_idx):
                # Profit check first to match tie handling with vectorized baseline
                if high[j] >= profit_barrier:
                    lab = 1  # LONG position - price moved up, take profit
                    profit_pct = pt_mult
                    break
                if low[j] <= stop_barrier:
                    lab = -1  # SHORT position - price moved down, take profit
                    profit_pct = -sl_mult
                    break
                    
            labels[i] = lab
            profit_pcts[i] = profit_pct
            
        return labels, profit_pcts


@dataclass
class RegimeTripleBarrierConfig:
    """Configuration for regime-specific triple barrier parameters."""
    
    # Default parameters (used as fallback)
    default_profit_take_multiplier: float = 0.02
    default_stop_loss_multiplier: float = 0.01
    default_time_barrier_minutes: int = 30
    default_max_lookahead: int = 100
    
    # Regime-specific parameters
    regime_profit_take_multipliers: Dict[str, float] = None
    regime_stop_loss_multipliers: Dict[str, float] = None
    regime_time_barrier_minutes: Dict[str, int] = None
    regime_max_lookahead: Dict[str, int] = None
    
    # TPSL parameters
    regime_tp_multipliers: Dict[str, float] = None
    regime_sl_multipliers: Dict[str, float] = None
    regime_position_sizes: Dict[str, float] = None
    
    # Regime mapping
    regime_id_to_name: Dict[int, str] = None
    regime_name_to_id: Dict[str, int] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.regime_profit_take_multipliers is None:
            self.regime_profit_take_multipliers = {}
        if self.regime_stop_loss_multipliers is None:
            self.regime_stop_loss_multipliers = {}
        if self.regime_time_barrier_minutes is None:
            self.regime_time_barrier_minutes = {}
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
    Regime-aware Triple Barrier Method for labeling using regime-specific parameters.

    This implementation extends the optimized triple barrier labeling to support
    regime-specific parameters for each HMM regime, providing more nuanced and
    adaptive labeling based on market conditions.
    """
    def __init__(
        self, 
        config: Optional[RegimeTripleBarrierConfig] = None,
        binary_classification: bool = True,
    ) -> None:
        """Initialize the regime-aware triple barrier labeling."

        Args:
            config: Configuration with regime-specific parameters
            binary_classification: If True, only generate buy (1) and sell (-1) labels
            no hold (0) labels. If False, include hold labels (default: True)
        """
        self.config = config or RegimeTripleBarrierConfig()
        self.binary_classification = binary_classification
        self.logger = get_logger("RegimeAwareTripleBarrierLabeling")

        if self.binary_classification:
            self.logger.info(
                "🔖 Regime-aware triple barrier labeling configured for binary classification (BUY/SELL only)",
            )
            self.logger.info("   → HOLD samples will be automatically filtered out")
        else:
            self.logger.warning(
                "⚠️ Regime-aware triple barrier labeling configured for ternary classification (BUY/HOLD/SELL)",
            )

    def set_regime_parameters(
        self, 
        regime_name: str, 
        profit_take_multiplier: float,
        stop_loss_multiplier: float,
        time_barrier_minutes: Optional[int] = None,
        max_lookahead: Optional[int] = None,
        tp_multiplier: Optional[float] = None,
        sl_multiplier: Optional[float] = None,
        position_size: Optional[float] = None
    ) -> None:
        """Set regime-specific parameters."

        Args:
            regime_name: Name of the regime
            profit_take_multiplier: Profit take multiplier for this regime
            stop_loss_multiplier: Stop loss multiplier for this regime
            time_barrier_minutes: Time barrier in minutes for this regime
            max_lookahead: Maximum lookahead for this regime
            tp_multiplier: Take profit multiplier for this regime
            sl_multiplier: Stop loss multiplier for this regime
            position_size: Position size for this regime
        """
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

    def set_regime_mapping(self, regime_id_to_name: Dict[int, str]) -> None:
        """Set regime ID to name mapping."

        Args:
            regime_id_to_name: Dictionary mapping regime IDs to regime names
        """
        self.config.regime_id_to_name = regime_id_to_name
        self.config.regime_name_to_id = {name: id for id, name in regime_id_to_name.items()}

    def get_regime_parameters(
        self, 
        regime_name: str
    ) -> Dict[str, float]:
        """Get parameters for a specific regime."

        Args:
            regime_name: Name of the regime

        Returns:
            Dictionary with regime-specific parameters
        """
        return {
            "profit_take_multiplier": self.config.regime_profit_take_multipliers.get(
                regime_name, self.config.default_profit_take_multiplier
            ),
            "stop_loss_multiplier": self.config.regime_stop_loss_multipliers.get(
                regime_name, self.config.default_stop_loss_multiplier
            ),
            "time_barrier_minutes": self.config.regime_time_barrier_minutes.get(
                regime_name, self.config.default_time_barrier_minutes
            ),
            "max_lookahead": self.config.regime_max_lookahead.get(
                regime_name, self.config.default_max_lookahead
            ),
            "tp_multiplier": self.config.regime_tp_multipliers.get(regime_name, 2.0),
            "sl_multiplier": self.config.regime_sl_multipliers.get(regime_name, 1.0),
            "position_size": self.config.regime_position_sizes.get(regime_name, 0.1),
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return=pd.DataFrame(),
        context="regime_aware_triple_barrier_labeling.vectorized"
    )
    @guard_dataframe_nulls(mode="warn", arg_index=1)
    @with_tracing_span("RegimeAwareTripleBarrier.apply_vectorized", log_args=False)
    def apply_regime_aware_triple_barrier_labeling(
        self, 
        data: pd.DataFrame,
        regime_column: str = "composite_cluster_id"
    ) -> pd.DataFrame:
        """Apply regime-aware triple barrier labeling."

        Args:
            data: DataFrame with OHLCV data and regime information
            regime_column: Column containing regime labels

        Returns:
            DataFrame with regime-aware triple barrier labels
        """
        # Debug
        self.logger.info(
            f"Applying regime-aware triple barrier labeling | cols={list(data.columns)} shape={data.shape}"
        )

        # Normalize common OHLCV column name variants
        try:
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
            for original, canonical in canonical_map.items():
                if original in data.columns and canonical not in data.columns:
                    rename_map[original] = canonical
            if rename_map:
                data = data.rename(columns=rename_map)
        except Exception:
            pass

        # Ensure required columns
        required_columns = ["close", "high", "low"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            msg = f"Missing required OHLC columns {missing_columns}; cannot perform labeling"
            with contextlib.suppress(Exception):
                self.logger.error(msg)
            raise ValueError(msg)

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

        # Create regime mapping if not set
        if not self.config.regime_id_to_name:
            self._create_regime_mapping(unique_regimes)

        # Apply regime-specific labeling
        labeled_data = self._apply_regime_specific_labeling(labeled_data, regime_column)

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

    def _create_regime_mapping(self, unique_regimes: np.ndarray) -> None:
        """Create regime ID to name mapping."

        Args:
            unique_regimes: Array of unique regime values
        """
        regime_id_to_name = {}
        for i, regime in enumerate(unique_regimes):
            if isinstance(regime, (int, np.integer)):
                regime_name = f"REGIME_{regime}"
            else:
                regime_name = str(regime)
            regime_id_to_name[i] = regime_name
        
        self.set_regime_mapping(regime_id_to_name)
        self.logger.info(f"🗺️ Created regime mapping: {regime_id_to_name}")

    def _apply_regime_specific_labeling(
        self, 
        data: pd.DataFrame, 
        regime_column: str
    ) -> pd.DataFrame:
        """Apply regime-specific triple barrier labeling."

        Args:
            data: DataFrame with OHLCV and regime data
            regime_column: Column containing regime labels

        Returns:
            DataFrame with regime-specific labels
        """
        labeled_data = data.copy()
        n = len(labeled_data)
        
        close = labeled_data["close"].to_numpy()
        high = labeled_data["high"].to_numpy()
        low = labeled_data["low"].to_numpy()
        regime_data = labeled_data[regime_column].to_numpy()

        # Create regime ID mapping
        unique_regimes = np.unique(regime_data)
        regime_to_id = {regime: i for i, regime in enumerate(unique_regimes)}
        regime_ids = np.array([regime_to_id[regime] for regime in regime_data])

        # Get regime-specific parameters
        pt_multipliers = []
        sl_multipliers = []
        
        for regime in unique_regimes:
            regime_name = self.config.regime_id_to_name.get(regime_to_id[regime], f"REGIME_{regime}")
            params = self.get_regime_parameters(regime_name)
            pt_multipliers.append(params["profit_take_multiplier"])
            sl_multipliers.append(params["stop_loss_multiplier"])

        pt_multipliers = np.array(pt_multipliers)
        sl_multipliers = np.array(sl_multipliers)

        # Calculate end indices
        idx = labeled_data.index
        use_time_barrier = isinstance(idx, pd.DatetimeIndex)
        
        arange_n = np.arange(n, dtype=np.int64)
        end_by_lookahead = np.minimum(arange_n + 1 + self.config.default_max_lookahead, n)
        
        if use_time_barrier:
            try:
                idx_ns = idx.view(np.int64)
                delta_ns = np.int64(self.config.default_time_barrier_minutes) * np.int64(60_000_000_000)
                end_times = idx_ns + delta_ns
                end_by_time = np.searchsorted(idx_ns, end_times, side="right")
            except Exception:
                end_by_time = end_by_lookahead
        else:
            end_by_time = end_by_lookahead
            
        end_idx_arr = np.minimum(end_by_lookahead, end_by_time).astype(np.int64)

        # Apply labeling
        use_numba = (
            "numba" in globals()
            and numba is not None
            and callable(globals().get("_numba_regime_aware_triple_barrier_labels"))
        )
        
        if use_numba and n >= 512:
            self.logger.info("⚡ Using Numba-accelerated regime-aware triple barrier labeling")
            labels, profit_pcts = _numba_regime_aware_triple_barrier_labels(
                close.astype(np.float64),
                high.astype(np.float64),
                low.astype(np.float64),
                regime_ids.astype(np.int64),
                pt_multipliers.astype(np.float64),
                sl_multipliers.astype(np.float64),
                end_idx_arr.astype(np.int64),
            )
        else:
            self.logger.info("🐍 Using Python regime-aware triple barrier labeling")
            labels = np.zeros(n, dtype=np.int8)
            profit_pcts = np.zeros(n, dtype=np.float64)
            
            for i in range(n - 1):
                entry_price = close[i]
                regime_id = regime_ids[i]
                
                # Get regime-specific multipliers
                pt_mult = pt_multipliers[regime_id] if regime_id < len(pt_multipliers) else pt_multipliers[0]
                sl_mult = sl_multipliers[regime_id] if regime_id < len(sl_multipliers) else sl_multipliers[0]
                
                profit_barrier = entry_price * (1.0 + pt_mult)
                stop_barrier = entry_price * (1.0 - sl_mult)
                end_idx = int(end_idx_arr[i])
                
                if end_idx <= i + 1:
                    labels[i] = 0
                    profit_pcts[i] = 0.0
                    continue
                    
                win_high = high[i + 1 : end_idx]
                win_low = low[i + 1 : end_idx]
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

        labeled_data["label"] = labels
        labeled_data["potential_profit_pct"] = profit_pcts

        # Add regime-specific TPSL information
        labeled_data = self._add_regime_tpsl_information(labeled_data, regime_column)

        return labeled_data

    def _add_regime_tpsl_information(
        self, 
        data: pd.DataFrame, 
        regime_column: str
    ) -> pd.DataFrame:
        """Add regime-specific TPSL information to the data."

        Args:
            data: DataFrame with labels and regime information
            regime_column: Column containing regime labels

        Returns:
            DataFrame with regime-specific TPSL information
        """
        data = data.copy()
        
        # Calculate ATR if not present
        if 'atr' not in data.columns:
            data['atr'] = self._calculate_atr(data, period=14)
        
        # Add regime-specific TPSL levels
        tp_levels = []
        sl_levels = []
        position_sizes = []
        
        for _, row in data.iterrows():
            regime = row[regime_column]
            regime_name = self.config.regime_id_to_name.get(regime, f"REGIME_{regime}")
            params = self.get_regime_parameters(regime_name)
            
            tp_level = row['close'] * (1 + params['tp_multiplier'] * row['atr'])
            sl_level = row['close'] * (1 - params['sl_multiplier'] * row['atr'])
            
            tp_levels.append(tp_level)
            sl_levels.append(sl_level)
            position_sizes.append(params['position_size'])
        
        data['tp_level'] = tp_levels
        data['sl_level'] = sl_levels
        data['position_size'] = position_sizes
        
        return data

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."

        Args:
            data: DataFrame with OHLC data
            period: Period for ATR calculation

        Returns:
            Series with ATR values
        """
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            return atr.fillna(method='bfill')
            
        except Exception:
            # Fallback to simple volatility
            return data['close'].pct_change().rolling(window=period).std().fillna(0.01)

    def _apply_default_labeling(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply default triple barrier labeling when regime information is not available."

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with default labels
        """
        self.logger.info("📝 Applying default triple barrier labeling")
        
        # Import the original optimized triple barrier labeling
        from .optimized_triple_barrier_labeling import OptimizedTripleBarrierLabeling
        
        labeler = OptimizedTripleBarrierLabeling(
            profit_take_multiplier=self.config.default_profit_take_multiplier,
            stop_loss_multiplier=self.config.default_stop_loss_multiplier,
            time_barrier_minutes=self.config.default_time_barrier_minutes,
            max_lookahead=self.config.default_max_lookahead,
            binary_classification=self.binary_classification
        )
        
        return labeler.apply_triple_barrier_labeling_vectorized(data)

    def get_regime_performance_summary(self, data: pd.DataFrame, regime_column: str = "composite_cluster_id") -> Dict[str, Dict[str, float]]:
        """Get performance summary for each regime."

        Args:
            data: DataFrame with labels and regime information
            regime_column: Column containing regime labels

        Returns:
            Dictionary with performance metrics for each regime
        """
        if regime_column not in data.columns or 'label' not in data.columns:
            return {}
        
        performance_summary = {}
        
        for regime in data[regime_column].unique():
            regime_data = data[data[regime_column] == regime]
            regime_name = self.config.regime_id_to_name.get(regime, f"REGIME_{regime}")
            
            # Calculate regime-specific metrics
            valid_data = regime_data[regime_data['label'] != 0]
            
            if len(valid_data) == 0:
                performance_summary[regime_name] = {
                    'total_samples': len(regime_data),
                    'valid_samples': 0,
                    'win_rate': 0.0,
                    'avg_profit': 0.0,
                    'total_return': 0.0
                }
                continue
            
            win_rate = (valid_data['label'] > 0).mean()
            avg_profit = valid_data['potential_profit_pct'].mean()
            total_return = valid_data['potential_profit_pct'].sum()
            
            performance_summary[regime_name] = {
                'total_samples': len(regime_data),
                'valid_samples': len(valid_data),
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'total_return': total_return
            }
        
        return performance_summary


# Utility functions for integration
def create_regime_aware_labeler_from_optimization_results(
    optimization_results: Dict[str, Any]
) -> RegimeAwareTripleBarrierLabeling:
    """Create a regime-aware labeler from optimization results."

    Args:
        optimization_results: Results from regime-specific optimization

    Returns:
        Configured regime-aware triple barrier labeler
    """
    config = RegimeTripleBarrierConfig()
    
    # Set regime-specific parameters from optimization results
    for regime_name, result in optimization_results.items():
        if isinstance(result, dict) and 'triple_barrier_params' in result:
            tb_params = result['triple_barrier_params']
            tpsl_params = result.get('tpsl_params', {})
            
            config.regime_profit_take_multipliers[regime_name] = tb_params.get('profit_take_multiplier', 0.02)
            config.regime_stop_loss_multipliers[regime_name] = tb_params.get('stop_loss_multiplier', 0.01)
            config.regime_time_barrier_minutes[regime_name] = tb_params.get('time_barrier_minutes', 30)
            config.regime_max_lookahead[regime_name] = tb_params.get('max_lookahead', 100)
            
            config.regime_tp_multipliers[regime_name] = tpsl_params.get('tp_multiplier', 2.0)
            config.regime_sl_multipliers[regime_name] = tpsl_params.get('sl_multiplier', 1.0)
            config.regime_position_sizes[regime_name] = tpsl_params.get('position_size', 0.1)
    
    return RegimeAwareTripleBarrierLabeling(config)


def apply_regime_aware_triple_barrier_labeling(
    data: pd.DataFrame,
    optimization_results: Optional[Dict[str, Any]] = None,
    regime_column: str = "composite_cluster_id",
    binary_classification: bool = True
) -> pd.DataFrame:
    """Apply regime-aware triple barrier labeling to data."

    Args:
        data: DataFrame with OHLCV and regime data
        optimization_results: Optional optimization results for regime-specific parameters
        regime_column: Column containing regime labels
        binary_classification: Whether to use binary classification

    Returns:
        DataFrame with regime-aware labels
    """
    if optimization_results:
        labeler = create_regime_aware_labeler_from_optimization_results(optimization_results)
    else:
        labeler = RegimeAwareTripleBarrierLabeling(binary_classification=binary_classification)
    
    return labeler.apply_regime_aware_triple_barrier_labeling(data, regime_column)


def apply_regime_aware_triple_barrier_labeling_with_barriers(
    data: pd.DataFrame,
    barrier_map_or_path: Union[str, Dict[str, Any]],
    regime_column: str = "hmm_regime",
    binary_classification: bool = True,
    default_time_barrier_minutes: int = 30,
    default_max_lookahead: int = 100
) -> pd.DataFrame:
    """
    Apply regime-aware triple barrier labeling using a barrier map or path.
    
    This function is designed to work with the HMMRegimeBarrierOptimizer and
    provides the interface needed by the vectorized labeling orchestrator.
    
    Args:
        data: DataFrame with OHLCV and regime data
        barrier_map_or_path: Either a path to a barrier map file or a barrier map dictionary
        regime_column: Column containing regime labels
        binary_classification: Whether to use binary classification
        default_time_barrier_minutes: Default time barrier in minutes
        default_max_lookahead: Default maximum lookahead
        
    Returns:
        DataFrame with regime-aware labels
    """
    try:
        import json
        from pathlib import Path
        
        # Load barrier map if path is provided
        if isinstance(barrier_map_or_path, str):
            barrier_path = Path(barrier_map_or_path)
            if barrier_path.exists():
                with open(barrier_path, 'r') as f:
                    barrier_map = json.load(f)
            else:
                raise FileNotFoundError(f"Barrier map file not found: {barrier_map_or_path}")
        else:
            barrier_map = barrier_map_or_path
        
        # Validate barrier map
        if not isinstance(barrier_map, dict):
            raise ValueError("Invalid barrier map format")
        
        # Check if regime column exists
        if regime_column not in data.columns:
            raise ValueError(f"Regime column '{regime_column}' not found in data")
        
        # Create regime-aware labeler with barrier-specific configuration
        config = RegimeTripleBarrierConfig()
        
        # Configure regime-specific parameters from barrier map
        for regime_id, regime_config in barrier_map.items():
            if isinstance(regime_config, dict):
                # Set regime-specific parameters
                config.regime_profit_take_multipliers[regime_id] = regime_config.get(
                    'profit_take_multiplier', 0.002
                )
                config.regime_stop_loss_multipliers[regime_id] = regime_config.get(
                    'stop_loss_multiplier', 0.001
                )
                config.regime_time_barrier_minutes[regime_id] = regime_config.get(
                    'time_barrier_minutes', default_time_barrier_minutes
                )
                config.regime_max_lookahead[regime_id] = regime_config.get(
                    'max_lookahead', default_max_lookahead
                )
        
        # Create labeler with configuration
        labeler = RegimeAwareTripleBarrierLabeling(config)
        
        # Apply regime-aware labeling
        labeled_data = labeler.apply_regime_aware_triple_barrier_labeling(data, regime_column)
        
        # Add labeling metadata
        labeled_data['labeling_method'] = 'regime_aware_with_barriers'
        labeled_data['barrier_map_source'] = str(barrier_map_or_path) if isinstance(barrier_map_or_path, str) else 'dict'
        
        return labeled_data
        
    except Exception as e:
        # Log error and return original data with error indicator
        import logging
import copy

logger = logging.getLogger(__name__)
logger.error(f"❌ Error in regime-aware triple barrier labeling with barriers: {e}")
        
        # Return data with error indicator
        data_copy = data.copy()
        data_copy['label'] = 0  # Default to HOLD
        data_copy['labeling_method'] = 'error_fallback'
        data_copy['labeling_error'] = str(e)
        
        return data_copy