#!/usr/bin/env python3
"""
Quick Layer 0 runner to calibrate Kalman filter parameters.
This should be run before meta_labeling_hpo_sample_weighted to generate layer0_summary files.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.training.steps.labeling.label_based_layer_0 import run_layer0_kalman_vwap
from src.utils.kline_parquet import KlinesParquetManager
import pandas as pd

def main():
    print("🚀 Running Layer 0 Kalman Filter Calibration...")
    
    # Configuration
    symbol = "ETHUSDT"
    timeframe = "15m"
    outcomes_dir = Path("outcomes")
    outcomes_dir.mkdir(exist_ok=True)
    
    config = {
        "symbol": symbol,
        "timeframe": timeframe,
        "execution_mode": "full",
        "layer0_n_trials": 50,  # Bayesian optimization trials
        "random_state": 42,
    }
    
    # Load market data
    print(f"📥 Loading {symbol} {timeframe} data...")
    manager = KlinesParquetManager()
    df = manager.load_klines(
        symbol=symbol.lower(),
        interval=timeframe,  # API uses 'interval' not 'timeframe'
        exchange="binance",
        last_n_days=1095  # 3 years
    )
    
    print(f"✅ Loaded {len(df)} bars from {df.index.min()} to {df.index.max()}")
    
    # Run Layer 0 optimization
    print("🔬 Optimizing Kalman filter parameters...")
    market_data_with_kalman, payload = run_layer0_kalman_vwap(
        symbol=symbol,
        timeframe=timeframe,
        market_data=df.copy(),
        config=config,
        outcomes_dir=outcomes_dir,
        run_optimization=True,
    )
    
    # Print results
    best_params = payload.get("best_params", {})
    print("\n✅ Layer 0 Calibration Complete!")
    print(f"   Q (process noise): {best_params.get('kalman_Q', 'N/A')}")
    print(f"   R (measurement noise): {best_params.get('kalman_R', 'N/A')}")
    print(f"   volume_weight: {best_params.get('volume_weight', 'N/A')}")
    print(f"   volume_adaptive: {best_params.get('volume_adaptive', 'N/A')}")
    
    print(f"\n📁 Output files saved to {outcomes_dir}/")
    print("   - layer0_summary_*.csv")
    print("   - layer0_report_*.md")
    print("   - layer0_kalman_bundle.joblib")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
