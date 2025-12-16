
import pandas as pd
import numpy as np
import sys
import os

# Ensure src is in path
sys.path.append(os.getcwd())

try:
    from src.training.steps.labeling.regime_leaf_feature_extractor import (
        build_regime_embedding_features,
        compute_regime_targets_from_ohlcv
    )
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def run_verification():
    # 1. Create Mock Market Data
    print("🛠️ Creating mock market data...")
    dates = pd.date_range("2024-01-01", periods=1000, freq="15min")
    np.random.seed(42)
    # Random walk
    close = 1000 * np.exp(np.random.normal(0, 0.001, size=len(dates)).cumsum())
    high = close * (1 + np.abs(np.random.normal(0, 0.0005, size=len(dates))))
    low = close * (1 - np.abs(np.random.normal(0, 0.0005, size=len(dates))))
    open_px = close * (1 + np.random.normal(0, 0.0002, size=len(dates))) # Approximate
    volume = np.random.lognormal(10, 1, size=len(dates))
    
    market_data = pd.DataFrame({
        "open": open_px,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    }, index=dates)
    
    # Ensure high is highest, low is lowest
    market_data["high"] = market_data[["open", "high", "low", "close"]].max(axis=1)
    market_data["low"] = market_data[["open", "high", "low", "close"]].min(axis=1)
    
    # 2. Test Feature Generation
    print("🧪 Testing build_regime_embedding_features...")
    try:
        cfg = {"close_col": "close", "volume_col": "volume"}
        features = build_regime_embedding_features(market_data, cfg)
        print(f"   Generated {features.shape[1]} features")
        
        # Check specific expert feature prefixes
        smc_cols = [c for c in features.columns if "smc_" in c]
        liq_cols = [c for c in features.columns if "liquidity_" in c or "reg_liq" in c] # Standard names used by generator?
        vf_cols = [c for c in features.columns if "vol_force_" in c]
        
        print(f"   SMC Features: {len(smc_cols)}")
        print(f"   Liquidity Features: {len(liq_cols)}")
        print(f"   Volume Force Features: {len(vf_cols)}")
        
        if len(smc_cols) > 0 and (len(liq_cols) > 0 or "reg_ohlcv__volume_log1p_z" in features.columns) and len(vf_cols) > 0:
            print("✅ Expert features present (Validation Passed)")
        else:
            print("⚠️ Some expert features missing (Check prefixes or generation logic)")
            # Note: volume_force might use different prefixes depending on config.
            
    except Exception as e:
        print(f"❌ Feature generation failed: {e}")
        import traceback
        traceback.print_exc()

    # 3. Test Target Generation
    print("🎯 Testing compute_regime_targets_from_ohlcv...")
    try:
        targets_cfg = {
            "close_col": "close", 
            "high_col": "high", 
            "low_col": "low",
            "macro_trend_horizons": [8, 16]
        }
        targets = compute_regime_targets_from_ohlcv(market_data, targets_cfg)
        
        if "regime_breakout" in targets.columns:
            n_hits = targets["regime_breakout"].sum()
            print(f"✅ 'regime_breakout' target found. Hits: {n_hits}")
        else:
            print("❌ 'regime_breakout' target MISSING")
            
    except Exception as e:
        print(f"❌ Target generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_verification()
