"""
XGB Meso Regime Step

This step implements the XGB Meso Regime logic as a specialization of the
HMM Macro Trend Step, focusing on meso-scale horizons (20-24 bars) and
enhanced binary labeling using profit/stop multipliers.
"""

from typing import Any, Dict
import pandas as pd
import numpy as np

from src.training.steps.hmm_macro_regime import HMMMLMacroTrendStep
from src.utils.tprint import tprint_info, tprint_warning

class XGBMesoRegimeStep(HMMMLMacroTrendStep):
    """
    Meso-regime specialist using XGBoost.
    Inherits from HMMMLMacroTrendStep but overrides configuration and labeling
    to target meso-scale horizons (20-32 bars) with precise event-based labeling.
    """

    def __init__(self, step_name: str = "xgb_meso_regime"):
        super().__init__(step_name)

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute with meso-specific configuration overrides."""

        # 1. Map Meso parameters to Alpha parameters
        # Default meso horizon: 24 bars (approx 6h on 15m, or 24h on 1h)
        # The prompt suggests 20-24 bars.
        meso_horizon = int(config.get("meso_event_horizon_bars", 24))

        # Force scaling factor to 1 so we have direct control over horizons
        config["macro_scaling_factor"] = 1

        # Override alpha horizons to focus strictly on the meso timeframe
        config["alpha_max_horizon_bars"] = meso_horizon
        config["alpha_macro_min_horizon_bars"] = meso_horizon
        config["alpha_macro_target_horizon_bars"] = meso_horizon

        # Use simple returns if specified (optional) but HMM macro defaults to log
        # We stick to defaults unless 'meso_event_min_ret' implies something else.

        tprint_info(f"🦎 XGBMesoRegimeStep initialized with meso_horizon={meso_horizon} bars")

        return await super().execute(config)

    def _compute_alpha_labels(
        self,
        aligned_df: pd.DataFrame,
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Override label computation to support 'meso' triple-barrier logic.

        If 'meso_event_profit_mult' and 'meso_event_stop_mult' are defined,
        we use path-dependent labeling (Triple Barrier).
        Otherwise, we use threshold-based logic on the horizon return using 'meso_event_min_ret'.
        """
        profit_mult = config.get("meso_event_profit_mult")
        stop_mult = config.get("meso_event_stop_mult")
        min_ret = float(config.get("meso_event_min_ret", 0.005)) # Default 0.5%

        # 1. Call super to get standard alpha fields (features + raw returns)
        # This populates 'alpha_forward_return_Xh' columns which we might use or overwrite
        df = super()._compute_alpha_labels(aligned_df, config)

        # 2. Apply Meso-Specific Labeling Logic
        if profit_mult is not None and stop_mult is not None:
            tprint_info(
                f"🦎 Applying Triple Barrier Labeling: "
                f"min_ret={min_ret}, profit_mult={profit_mult}, stop_mult={stop_mult}"
            )
            df = self._apply_triple_barrier_labels(df, config, min_ret, float(profit_mult), float(stop_mult))
        else:
            tprint_info(f"🦎 Applying Threshold-based Labeling: min_ret={min_ret}")
            df = self._apply_threshold_labels(df, config, min_ret)

        return df

    def _apply_threshold_labels(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
        min_ret: float
    ) -> pd.DataFrame:
        """
        Apply simple thresholding logic:
        Target = 1 if Return > min_ret
        Target = 0 if Return < -min_ret (or whatever the binary logic is)

        The prompt says: "emphasize events most similar to binary_label"
        Usually this means 1 for Up, 0 for Down.
        """
        # We need the return at the specific meso horizon
        meso_horizon = int(config.get("meso_event_horizon_bars", 24))

        # HMMMLMacroTrendStep computes 'alpha_forward_return_{h}h'.
        # Since we set alpha_max_horizon_bars = meso_horizon, the column should be there.
        # But HMMMLMacroTrendStep naming convention uses 'h' suffix which might be misleading
        # if bars != hours. It appends 'h' regardless of timeframe unit usually.
        # Let's check _compute_alpha_labels in parent.
        # It does: `col_name = f"alpha_forward_return_{h}h"` for h in range(1, max_h+1).

        target_col = f"alpha_forward_return_{meso_horizon}h"

        if target_col not in df.columns:
            # Fallback: recompute it here if missing
            close = df["close"].astype(float)
            if config.get("alpha_return_type", "log") == "simple":
                fwd_ret = close.shift(-meso_horizon) / close - 1.0
            else:
                fwd_ret = np.log(close.shift(-meso_horizon) / close)
            df[target_col] = fwd_ret

        # Apply thresholding
        # Standard classification: 1 if > min_ret, 0 if < -min_ret?
        # Or 1 if > min_ret, 0 otherwise?
        # MLMeanReversion uses: > threshold -> 0 (bullish), < -threshold -> 1 (bearish).
        # HMMMacroTrend uses: > 0 -> 1.

        # Let's align with HMMMacroTrend default direction (1 = Bullish/Up)
        # Target = 1 if ret > min_ret
        # Target = 0 if ret < -min_ret (if we want to exclude noise)
        # Or 0 if ret <= min_ret?

        # To "emphasize events", we likely want to ignore small moves (set to NaN).
        # This allows XGBoost (with OOF trainer) to learn from distinct events.

        ret_series = df[target_col]
        new_target = pd.Series(np.nan, index=df.index)

        # Long/Up events
        new_target[ret_series > min_ret] = 1.0

        # Short/Down events
        new_target[ret_series < -min_ret] = 0.0

        # Small moves remain NaN and will be dropped by dropna in parent/trainer
        # or handled by weights. HMMMLMacroTrendStep drops rows with NaN target.

        df["alpha_target"] = new_target

        # Update effective horizon name for metadata
        df["alpha_target_horizon_name"] = f"{meso_horizon}bars_threshold_{min_ret}"

        return df

    def _apply_triple_barrier_labels(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
        min_ret: float,
        profit_mult: float,
        stop_mult: float
    ) -> pd.DataFrame:
        """
        Apply Triple Barrier Method:
        1. Upper Barrier: entry * (1 + min_ret * profit_mult)
        2. Lower Barrier: entry * (1 - min_ret * stop_mult)
        3. Time Barrier: meso_event_horizon_bars

        Label 1 if Upper Barrier hit first.
        Label 0 if Lower Barrier hit first.
        Label 0 (or NaN?) if Time Barrier hit first (vertical barrier).
        """
        horizon = int(config.get("meso_event_horizon_bars", 24))

        close = df["close"].astype(float).values
        high = df["high"].astype(float).values if "high" in df.columns else close
        low = df["low"].astype(float).values if "low" in df.columns else close

        n = len(close)
        labels = np.full(n, np.nan)

        # Numba optimization would be ideal here, but sticking to numpy/python for simplicity/compatibility
        # We can iterate or use a rolling window approach. For 24 bars, iteration is okay-ish but slow for 100k rows.
        # Let's try to use a slightly optimized loop.

        # Pre-calculate barriers for all rows? No, barriers are relative to entry price.

        # Vectorized approximation:
        # We can use rolling max/min, but that tells us IF it hit, not WHICH hit first.
        # For full correctness, we need first-passage time.

        # Let's implement a Numba-friendly function if available, or a fast loop.
        # Since we are inside a class, let's define a static helper.

        try:
            from numba import njit

            @njit
            def compute_barriers(close_arr, high_arr, low_arr, horizon, min_ret, profit_mult, stop_mult):
                n_samples = len(close_arr)
                out_labels = np.full(n_samples, np.nan)

                # Dynamic thresholds based on volatility?
                # The prompt implies fixed multipliers on 'min_ret', but usually 'min_ret' is volatility-based?
                # "meso_event_min_ret" sounds like a scalar (e.g. 0.005).

                up_ret = min_ret * profit_mult
                down_ret = min_ret * stop_mult

                for i in range(n_samples - horizon):
                    entry = close_arr[i]
                    if entry <= 0: continue

                    hit_up = False
                    hit_down = False

                    upper_price = entry * (1.0 + up_ret)
                    lower_price = entry * (1.0 - down_ret)

                    for j in range(1, horizon + 1):
                        idx = i + j
                        if idx >= n_samples: break

                        # Check high for profit, low for stop (assuming Long direction logic)
                        # If we want symmetric or direction-agnostic, we need 'direction' param.
                        # HMMMLMacroTrendStep executes for a specific 'direction' (default 'long').
                        # Assuming Long for now (Label 1 = Profit Up).

                        # Did we hit stop?
                        if low_arr[idx] <= lower_price:
                            hit_down = True

                        # Did we hit profit?
                        if high_arr[idx] >= upper_price:
                            hit_up = True

                        if hit_down and hit_up:
                            # Hit both in same bar? Usually stop takes precedence or we consider Close.
                            # Standard Triple Barrier: if both, usually assume worst case (Stop) or check Open/Close path.
                            # Let's assume Stop hit first if Low < Lower.
                            out_labels[i] = 0.0
                            break
                        elif hit_down:
                            out_labels[i] = 0.0
                            break
                        elif hit_up:
                            out_labels[i] = 1.0
                            break

                    # If loop finishes without break -> Time Barrier
                    if not hit_down and not hit_up:
                        # Vertical barrier hit.
                        # Label depends on use case. Often 0 or separate class.
                        # For binary classification of "Successful Trade", this is 0 (Failure to profit).
                        out_labels[i] = 0.0

                return out_labels

            labels = compute_barriers(close, high, low, horizon, min_ret, profit_mult, stop_mult)

        except ImportError:
            # Python fallback (slow but functional)
            tprint_warning("Numba not available, using slower Python loop for Triple Barrier")
            up_ret = min_ret * profit_mult
            down_ret = min_ret * stop_mult

            for i in range(n - horizon):
                entry = close[i]
                if entry <= 0: continue

                upper = entry * (1 + up_ret)
                lower = entry * (1 - down_ret)

                window_high = high[i+1 : i+horizon+1]
                window_low = low[i+1 : i+horizon+1]

                # Find indices where barriers are crossed
                # argmax returns index of first True
                up_cross = np.where(window_high >= upper)[0]
                down_cross = np.where(window_low <= lower)[0]

                first_up = up_cross[0] if len(up_cross) > 0 else 9999
                first_down = down_cross[0] if len(down_cross) > 0 else 9999

                if first_up == 9999 and first_down == 9999:
                    labels[i] = 0.0 # Time barrier
                elif first_up < first_down:
                    labels[i] = 1.0 # Profit first
                else:
                    labels[i] = 0.0 # Stop first (or simultaneous)

        df["alpha_target"] = labels
        df["alpha_target_horizon_name"] = f"triple_barrier_{horizon}bars_{min_ret}"

        return df
