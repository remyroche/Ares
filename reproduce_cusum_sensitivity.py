import pandas as pd
import numpy as np
from src.training.steps.labeling.feature_generation_meta_labeling_step import generate_primary_signals
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_data(n_days=300):
    """Generate synthetic 15m price data with trends and noise."""
    n_bars = n_days * 96  # 15m bars
    
    # Generate random walk with drift changes (regimes)
    np.random.seed(42)
    returns = np.random.normal(0, 0.002, n_bars)
    
    # Add some trends
    trend_periods = 10
    trend_len = n_bars // trend_periods
    for i in range(trend_periods):
        direction = 1 if i % 2 == 0 else -1
        start = i * trend_len
        end = (i + 1) * trend_len
        drift = 0.0002 * direction
        returns[start:end] += drift
        
    price = 100 * np.exp(np.cumsum(returns))
    
    # Create DataFrame
    dates = pd.date_range(start='2024-01-01', periods=n_bars, freq='15min')
    df = pd.DataFrame({
        'close': price,
        'volume': np.random.lognormal(10, 1, n_bars)
    }, index=dates)
    
    return df

def test_sensitivity():
    df = generate_synthetic_data()
    logger.info(f"Generated {len(df)} bars of synthetic data")
    
    # 1. Baseline
    logger.info("Running Baseline (k=0.5, alpha=2.0)...")
    signals_base = generate_primary_signals(
        df,
        k=0.5,
        alpha=2.0,
        enable_dynamic_tuning=False
    )
    n_base = (signals_base['consensus'] != 0).sum()
    logger.info(f"Baseline events: {n_base}")
    
    # 2. Max Aggressive Relaxation
    logger.info("Running Max Aggressive (k=0.15, alpha=0.1, er_min=0.0)...")
    signals_aggressive = generate_primary_signals(
        df,
        k=0.15,
        alpha=0.1,
        er_min=0.00,
        enable_dynamic_tuning=False
    )
    n_aggressive = (signals_aggressive['consensus'] != 0).sum()
    logger.info(f"Max Aggressive events: {n_aggressive}")
    
    change = (n_aggressive - n_base) / n_base * 100
    logger.info(f"Change vs Baseline: {change:+.2f}%")
    
    if n_aggressive > n_base * 5.0:
        logger.info("SUCCESS: Max Aggressive parameters produced >5x events (Targeting ~6x).")
    else:
        logger.warning(f"FAILURE: Increase was only {change:.2f}%. Needed >400%.")

if __name__ == "__main__":
    test_sensitivity()
