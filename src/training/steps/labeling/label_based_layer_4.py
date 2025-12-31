"""Layer 4 — Triple Barrier Trailing Profit & Sizing.

Layer2 is about learnability, layer3 about relation to target (IC, calibration),
layer4 is about position sizing. I want to trade it with a triple barrier method
that includes trailing profit.

This module implements:
1.  Triple Barrier Trailing Logic (Exit Strategy).
2.  Inverse Volatility Sizing (Position Sizing).
3.  Integration with Layer 5 via `layer4_prob` proxy generation.

Ensure compatibility with label_based_layer_5:
Layer 5 calculates Size = ((p - 0.5) / 0.5) ^ 2.
We reverse this to generate `layer4_prob` such that Layer 5 produces our desired
Inverse Volatility Size.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
from datetime import datetime
import json

from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
from scipy.stats import spearmanr


# Configuration Constants
STOP_LOSS_FLOOR = 0.004  # 0.3% Fees + 0.1% Spread Buffer
TARGET_VOLATILITY = 0.01  # 1% target volatility for sizing
VOLATILITY_SAFETY_FLOOR = 1e-4  # Prevent division by zero
HOME_RUN_MULTIPLIER = 3.0  # Multiplier for home run detection
WEIGHT_CLIP_MIN = 0.5  # Minimum weight clip
WEIGHT_CLIP_MAX = 2.0  # Maximum weight clip
NEUTRAL_PROB_THRESHOLD = 0.5  # Neutral probability threshold for gating
ZERO_SIZE_PROB = 0.4  # Probability that results in zero size


def triple_barrier_trailing_label(
    df: pd.DataFrame,
    events: pd.DatetimeIndex,
    volatility: pd.Series,
    horizon: int = 24,
    sl: float = 1.0,
    trailing_gap: Optional[float] = None,
    pt: Optional[float] = None,
    min_ret: float = 0.003
) -> pd.DataFrame:
    """
    Advanced Triple Barrier Labeler with Trailing Profit Logic.
    
    Implements a "Rising Floor" trade structure:
    - If trailing_gap is set: The Upper Barrier is removed (Infinity).
      The Lower Barrier (Stop Loss) ratchets up as price makes new highs.
    - If trailing_gap is None: Uses standard Fixed Upper/Lower Barriers.
    
    Args:
        df: DataFrame with 'close', 'high', 'low' columns.
        events: DatetimeIndex of signal entry times.
        volatility: Series of volatility (e.g., ATR or StdDev) aligned with df.
        horizon: Maximum holding period in bars (Vertical Barrier).
        sl: Initial Stop Loss multiplier (e.g., 1.0 * Volatility).
        trailing_gap: The distance (in Volatility units) the stop trails behind the High.
                      If None, defaults to Fixed Barrier logic.
        pt: Fixed Profit Target multiplier (Only used if trailing_gap is None).
        min_ret: Minimum return required to label as '1' (accounts for fees).

    Returns:
        DataFrame containing:
        - 'label': {-1, 0, 1}
        - 'ret': Raw return of the trade
        - 'weight': Sample weight based on Inverse Volatility
    """
    out = {}

    # 1. Config: Fee Floors
    # We enforce a minimum stop distance to prevent trading inside the spread/fees.
    # 0.3% Fees + 0.1% Spread Buffer = 0.4% Floor.
    barrier_type = "Trailing" if trailing_gap is not None else "Fixed"
    tprint_info(f"🔄 Triple Barrier: {barrier_type} mode, {len(events)} events, horizon={horizon}")

    # 2. Pre-fetch Data for Speed
    # Align volatility to events
    vol_s = volatility.reindex(events).ffill().bfill()

    # Extract arrays (Fast Numpy Access)
    closes = df['close'].values
    if 'high' in df.columns and 'low' in df.columns:
        highs = df['high'].values
        lows = df['low'].values
    else:
        # Fallback if OHLC not available (Not recommended for Trailing)
        highs = closes; lows = closes
        
    index = df.index
    n_bars = len(df)

    # 3. Main Event Loop
    for t in events:
        if t not in index: continue
        
        # Get Integer Location of Entry
        i_0 = index.get_loc(t)
        
        # Define Vertical Barrier (Time Expiry)
        i_1 = min(i_0 + horizon, n_bars - 1)
        if i_1 <= i_0: continue
        
        # Get Volatility at Entry
        curr_vol = vol_s[t]
        if curr_vol <= 0: curr_vol = VOLATILITY_SAFETY_FLOOR # Safety floor
        
        entry_price = closes[i_0]
        
        # --- A. Determine Safe Distances (Fee Floor Logic) ---
        raw_stop_dist = curr_vol * sl
        safe_stop_dist = max(raw_stop_dist, STOP_LOSS_FLOOR)
        
        # --- B. Trailing Stop Logic (The "Rising Floor") ---
        if trailing_gap is not None:
            # 1. Initialize Stop at Entry
            stop_price = entry_price * (1 - safe_stop_dist)
            max_price = entry_price # High Water Mark
            exit_idx = -1
            
            # Apply Fee Floor to the Gap as well
            raw_gap_dist = curr_vol * trailing_gap
            safe_gap_dist = max(raw_gap_dist, STOP_LOSS_FLOOR)

            # 2. Walk Forward Path (Bar by Bar)
            # Start at i_0 + 1 because we enter on Close of i_0
            exit_price = entry_price  # Initialize exit_price
            for k in range(i_0 + 1, i_1 + 1):
                c_low = lows[k]
                c_high = highs[k]
                
                # a. Check Stop Hit First (Pessimistic Assumption)
                # We check Low against Current Stop
                if c_low < stop_price:
                    exit_idx = k
                    exit_price = stop_price
                    break
                
                # b. Ratchet Logic (Optimistic Update)
                # If we survived the Low, check if High raised the ceiling
                if c_high > max_price:
                    max_price = c_high

                    # The Ratchet: Stop moves UP to (NewHigh - Gap)
                    # It NEVER moves down (max logic)
                    new_stop = max_price * (1 - safe_gap_dist)
                    stop_price = max(stop_price, new_stop)

            # 3. Determine Outcome
            if exit_idx != -1:
                # Stopped Out (Could be a Win if Stop Ratcheted above Entry)
                raw_ret = (exit_price / entry_price) - 1
            else:
                # Vertical Barrier Hit (Time Expired)
                raw_ret = (closes[i_1] / entry_price) - 1


        # --- C. Standard Fixed Barrier Logic ---
        else:
            # Implied R:R Ratio logic if PT is provided
            eff_pt = pt if pt is not None else 1.0
            rr_ratio = eff_pt / sl
            safe_target_dist = safe_stop_dist * rr_ratio
            
            trgt_price = entry_price * (1 + safe_target_dist)
            stop_price = entry_price * (1 - safe_stop_dist)

            # Simple Path Check
            raw_ret = 0.0
            path_slice_high = highs[i_0+1 : i_1+1]
            path_slice_low = lows[i_0+1 : i_1+1]
            path_slice_close = closes[i_0+1 : i_1+1]

            # Find first touch index
            # Fixed: Check if condition exists before using argmax
            up_mask = path_slice_high > trgt_price
            dn_mask = path_slice_low < stop_price
            
            has_up = up_mask.any()
            has_dn = dn_mask.any()
            
            if has_up:
                touch_up = np.argmax(up_mask)
            else:
                touch_up = None
                
            if has_dn:
                touch_dn = np.argmax(dn_mask)
            else:
                touch_dn = None

            if has_up and (not has_dn or (touch_dn is not None and touch_up < touch_dn)):
                raw_ret = safe_target_dist # Hit Target
            elif has_dn and (not has_up or (touch_up is not None and touch_dn < touch_up)):
                raw_ret = -safe_stop_dist # Hit Stop
            else:
                raw_ret = (path_slice_close[-1] / entry_price) - 1 # Time Expiry


        # --- D. Final Labeling & Weighting ---
        
        # 1. Label
        if raw_ret > min_ret:
            label = 1
        elif raw_ret < -min_ret:
            label = -1
        else:
            label = 0

        # 2. Sample Weight (Inverse Volatility)
        # We value stability. A win in low vol is worth more leverage than a win in high vol.
        # Logic: Target Vol (1%) / Current Vol
        weight = np.clip(TARGET_VOLATILITY / (curr_vol + VOLATILITY_SAFETY_FLOOR), WEIGHT_CLIP_MIN, WEIGHT_CLIP_MAX)
        
        # Bonus: Reward "Home Runs" (Outliers)
        # If the trade captured > 3 Sigma move, boost importance
        if label == 1 and abs(raw_ret) > (curr_vol * HOME_RUN_MULTIPLIER):
            weight *= 1.5


        if label != 0:
            out[t] = {
                'label': label,
                'ret': raw_ret,
                'weight': weight
            }

    return pd.DataFrame.from_dict(out, orient='index')


# ---------------------------------------------------------------------------
# Training Orchestration (Adapted for Layer 4 Sizing Calculation)
# ---------------------------------------------------------------------------

def train_layer4_oof(
    oof_df: pd.DataFrame,
    market_data: pd.DataFrame,
    l3_prob_col: str = 'meta_prob',
    target_col: str = 'target',
    return_col: str = 'realized_return',
    n_folds: int = 5,
    config: Optional[Dict[str, Any]] = None,
    # Deprecated parameters - kept for backward compatibility only
    l3_models_metadata: Optional[Dict] = None,
    l3_quantile_thresholds: Optional[List[float]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply Triple Barrier Trailing Logic and Volatility Sizing.
    
    This replaces the legacy Risk Filter model. Instead of training a model,
    it simulates the specific Triple Barrier strategy and sets 'layer4_prob'
    to achieve the desired Inverse Volatility Sizing in Layer 5.
    """
    
    tprint_info("="*60)
    tprint_info("🚀 LAYER 4: TRIPLE BARRIER TRAILING & POSITION SIZING")
    tprint_info("="*60)
    
    cfg = config or {}
    
    tprint_info(f"📊 Input data: {len(oof_df)} OOF samples, {len(market_data)} market data points")
    tprint_info(f"🎯 Using probability column: {l3_prob_col}")
    
    # 1. Prepare Data
    if oof_df.index.duplicated().any():
        tprint_warning(f"⚠️ OOF data has {oof_df.index.duplicated().sum()} duplicate indices. Keeping first.")
        oof_df = oof_df[~oof_df.index.duplicated(keep='first')]

    oof_aligned = oof_df.copy()
    
    # Ensure market data covers oof
    common_idx = oof_aligned.index.intersection(market_data.index)
    if len(common_idx) < len(oof_aligned) * 0.9:
        tprint_warning("⚠️ Market data coverage low for OOF events. Results may be inaccurate.")
    else:
        tprint_info(f"📊 Market data coverage: {len(common_idx)}/{len(oof_aligned)} ({len(common_idx)/len(oof_aligned):.1%})")
    
    df_eval = market_data.loc[common_idx]

    # 2. Config extraction
    sl = float(cfg.get('layer4_sl', 1.0))
    trailing_gap = cfg.get('layer4_trailing_gap')
    if trailing_gap is not None:
        trailing_gap = float(trailing_gap)
    else:
        # Default behavior: If not specified, use 1.5 as in user example, or allow None?
        # User said "trade it with a triple barrier method that includes trailing profit."
        # If config is missing, default to 1.5 for trailing gap to ensure trailing logic is active.
        trailing_gap = 1.5

    horizon = int(cfg.get('layer4_horizon', 48))
    pt = cfg.get('layer4_pt')
    if pt is not None: pt = float(pt)
    
    tprint_info(f"⚙️ Configuration: SL={sl}, Trailing Gap={trailing_gap}, Horizon={horizon}, PT={pt}")
    
    # 3. Volatility
    if 'volatility_1d' in oof_aligned.columns:
        vol = oof_aligned['volatility_1d']
        tprint_info("📈 Using volatility_1d from OOF data")
    elif 'volatility_1d' in market_data.columns:
        vol = market_data['volatility_1d'].reindex(oof_aligned.index).fillna(0.01)
        tprint_info("📈 Using volatility_1d from market data")
    else:
        vol = df_eval['close'].pct_change().rolling(24).std().reindex(oof_aligned.index).fillna(0.01)
        tprint_info("📈 Computed volatility from price changes")

    # 4. Run Triple Barrier
    # We run on ALL oof events to get potential sizing/returns
    tprint_info(f"🔄 Executing Triple Barrier: SL={sl}, Gap={trailing_gap}, Hz={horizon}")

    tb_results = triple_barrier_trailing_label(
        df=market_data,
        events=oof_aligned.index,
        volatility=vol,
        horizon=horizon,
        sl=sl,
        trailing_gap=trailing_gap,
        pt=pt
    )

    tprint_info(f"📊 Triple Barrier completed: {len(tb_results)} results")

    # 5. Integrate Results
    # Initialize columns
    tprint_info("🔧 Integrating triple barrier results...")
    oof_aligned['layer4_return'] = 0.0
    oof_aligned['layer4_weight'] = 0.0
    oof_aligned['layer4_prob'] = 0.5 # Default (Size 0)
    
    if not tb_results.empty:
        # Align results
        res_aligned = tb_results.reindex(oof_aligned.index)
        
        # Store Raw Results
        oof_aligned['layer4_return'] = res_aligned['ret'].fillna(0.0)
        oof_aligned['layer4_weight'] = res_aligned['weight'].fillna(0.0) # This includes Home Run bonus
        oof_aligned['layer4_label'] = res_aligned['label'].fillna(0)
        
        # Override realized_return for Layer 5 Backtest
        oof_aligned[return_col] = oof_aligned['layer4_return']
        
        tprint_info(f"📈 Results: {len(res_aligned)} aligned trades")
        tprint_info(f"📈 Average return: {oof_aligned['layer4_return'].mean():.4f}")
        tprint_info(f"⚖️ Average weight: {oof_aligned['layer4_weight'].mean():.3f}")
        
        # 6. Generate Compatible layer4_prob
        # Goal: Layer 5 Size = Weight (Inverse Vol)
        # Layer 5: Size = ((p - 0.5) / 0.5) ^ 2  (assuming gamma=2)
        # Inverse: p = 0.5 * sqrt(Size) + 0.5
        
        # We use the Base Weight (without Home Run bonus) for sizing to avoid lookahead.
        # Recalculate base weight locally
        curr_vol = vol
        base_weight = np.clip(TARGET_VOLATILITY / (curr_vol + VOLATILITY_SAFETY_FLOOR), WEIGHT_CLIP_MIN, WEIGHT_CLIP_MAX)
        
        tprint_info("🎯 Generating layer4_prob for Layer 5 compatibility...")
        
        # Gate: Only trade if Layer 3 Probability is high enough (if available)
        # Or if the user wants purely Vol Sizing, we assume L3 has already filtered the OOF set.
        # But OOF usually contains all samples.
        # We need a decision trigger.
        # Check if l3_prob_col exists
        if l3_prob_col in oof_aligned.columns:
            l3_probs = pd.to_numeric(oof_aligned[l3_prob_col], errors='coerce').fillna(0.5)
            # Simple Gate: If L3 > 0.5, apply Vol Sizing. Else Size 0.
            # Using 0.5 as neutral threshold.
            trade_mask = l3_probs > NEUTRAL_PROB_THRESHOLD
        else:
            # If no L3 prob, assume all events are valid (e.g. filtered upstream)
            trade_mask = pd.Series(True, index=oof_aligned.index)

        # Calculate P
        # Clip Size to max 1.0 because Layer 5 usually caps p at 1.0 -> Size 1.0
        # But user weight goes to 2.0.
        # If Layer 5 caps p at 1.0, max Size is 1.0.
        # To support Size > 1.0, Layer 5 needs to allow p > 1.0 or change formula.
        # Assuming Layer 5 is strict (p clipped 0-1), we clamp our target size to 1.0.
        target_size = np.minimum(base_weight, 1.0)
        
        derived_p = 0.5 * np.sqrt(target_size) + 0.5
        
        # Apply Gate
        final_p = np.where(trade_mask, derived_p, ZERO_SIZE_PROB) # 0.4 -> Size 0

        oof_aligned['layer4_prob'] = final_p
        
    else:
        tprint_warning("   Triple Barrier returned no results.")
    
    # 7. Metrics & Report
    metrics = {
        'l4_mean_ret': float(oof_aligned['layer4_return'].mean()),
        'l4_win_rate': float((oof_aligned['layer4_return'] > 0).mean()),
        'l4_avg_weight': float(oof_aligned['layer4_weight'].mean()),
        'l4_sl_param': sl,
        'l4_gap_param': trailing_gap
    }
    
    tprint_info(f"📈 Final Metrics: Mean Return={metrics['l4_mean_ret']:.4f}, Win Rate={metrics['l4_win_rate']:.1%}, Avg Weight={metrics['l4_avg_weight']:.3f}")
    
    _generate_report(oof_aligned, metrics, cfg)
    
    tprint_success(f"🎉 Layer 4 Complete! Processed {len(oof_aligned)} samples")
    return oof_aligned, metrics

def _safe_to_markdown(df: pd.DataFrame) -> str:
    """Fallback for to_markdown() if tabulate is missing."""
    try:
        return df.to_markdown(index=False)
    except Exception:
        cols = df.columns
        res = [" | " + " | ".join(map(str, cols)) + " | "]
        res.append(" | " + " | ".join(["---"] * len(cols)) + " | ")
        for _, row in df.iterrows():
            formatted_row = [f"{x:.4f}" if isinstance(x, (float, np.float64, np.float32)) else str(x) for x in row]
            res.append(" | " + " | ".join(formatted_row) + " | ")
        return "\n".join(res)

def _generate_report(df: pd.DataFrame, metrics: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """Generate final model performance report and CSV summary."""
    try:
        tprint_info("📝 Generating Layer 4 report...")
        outcomes_dir = Path(cfg.get('outcomes_dir', 'outcomes'))
        outcomes_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = outcomes_dir / f"layer4_report_{ts}.md"
        
        lines = ["# Layer 4 Triple Barrier & Sizing Report\n"]
        lines.append(f"Timestamp: {ts}\n\n")
        
        lines.append("## Configuration\n")
        lines.append(f"- SL: {metrics.get('l4_sl_param', 'N/A')}\n")
        lines.append(f"- Trailing Gap: {metrics.get('l4_gap_param', 'N/A')}\n\n")
        
        lines.append("## Performance Summary\n")
        lines.append(f"- Mean Return: {metrics.get('l4_mean_ret', 0.0):.6f}\n")
        lines.append(f"- Win Rate: {metrics.get('l4_win_rate', 0.0)*100:.2f}%\n")
        lines.append(f"- Avg Weight (Size): {metrics.get('l4_avg_weight', 1.0):.4f}\n")
        
        # Add Sizing Efficiency (De Prado aligned)
        # Spearman correlation between weight and executed return
        if 'layer4_weight' in df.columns and 'layer4_return' in df.columns:
            try:
                eff, _ = spearmanr(df['layer4_weight'], df['layer4_return'])
                lines.append(f"- Bet-Sizing Efficiency (SR): {eff:.4f}\n")
            except Exception:
                pass
        lines.append("\n")
        
        # Quintile Analysis Table
        lines.append("## Bet Size Quintiles Analysis\n")
        try:
            df['quintile'] = pd.qcut(df['layer4_weight'], 5, labels=['Q1(Small)', 'Q2', 'Q3', 'Q4', 'Q5(Large)'])
            q_stats = df.groupby('quintile').agg({
                'layer4_return': 'mean',
                'layer4_weight': ['count', 'mean'],
            })
            # Win rate per quintile
            q_stats['win_rate'] = df.groupby('quintile').apply(lambda x: (x['layer4_return'] > 0).mean())
            q_stats.columns = ['Mean Ret', 'Count', 'Avg Weight', 'Win Rate']
            lines.append(_safe_to_markdown(q_stats) + "\n\n")
        except Exception as e:
            lines.append(f"Failed to generate quintile analysis: {e}\n\n")
            
        with open(report_path, 'w') as f:
            f.writelines(lines)
        tprint_success(f"💾 Layer 4 Report saved to {report_path}")
        
    except Exception as e:
        tprint_error(f"❌ Failed to generate Layer 4 report: {e}")


# ==========================================
# Demo / Test Functions
# ==========================================

def _demo_triple_barrier():
    """Demo function for testing triple barrier logic."""
    # 1. Create Synthetic Data (Trending Regime)
    dates = pd.date_range('2024-01-01', periods=200, freq='15min')
    
    # Simulate a price path that goes up, pulls back, then goes up again
    # Perfect for testing trailing stops
    path = np.linspace(100, 105, 200) # Baseline trend
    noise = np.random.normal(0, 0.2, 200) # Volatility
    price_close = path + noise
    
    df = pd.DataFrame({
        'close': price_close,
        'high': price_close + 0.1, # Mock High
        'low': price_close - 0.1   # Mock Low
    }, index=dates)

    # 2. Calculate Volatility
    volatility = df['close'].pct_change().rolling(10).std().fillna(0.002)

    # 3. Define dummy events (e.g., every 20th bar)
    events = df.index[::20]

    print("--- Running Triple Barrier with Trailing Profit ---")

    # 4. Run Labeler
    # Scenario: Wide Trend Following
    # Stop = 1.0x Vol, Trail = 1.5x Vol
    labels = triple_barrier_trailing_label(
        df,
        events,
        volatility,
        horizon=48,
        sl=1.0,
        trailing_gap=1.5
    )

    print("\nResulting Labels:")
    print(labels.head())

    print("\nDistribution:")
    print(labels['label'].value_counts())

    print("\nMean Return of Winners:")
    if 1 in labels['label'].values:
        print(f"{labels[labels['label']==1]['ret'].mean():.4%}")


if __name__ == "__main__":
    _demo_triple_barrier()
