import numpy as np
import pandas as pd
from lgbm_based_mask_generation import (
    FeatureProcessor, 
    InteractionModel, 
    RuleExtractor, 
    RuleScorer, 
    run_lgbm_mask_generation
)
from extreme_price_movements.intraday_crypto_library import INTRADAY_TRIGGER_COLUMNS, LOCATION_FILTER_COLUMNS

def test_lgbm_hardening():
    print("Starting Hardened Interaction Detector Verification...")
    
    # 1. Generate Synthetic Data
    n_samples = 1000
    timestamps = np.repeat(np.arange(100), 10)
    symbol_codes = np.tile(np.arange(10), 100)
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'symbol': symbol_codes
    })
    
    # Create feature dict
    feat_dict = {}
    
    # Add some triggers
    t1, t2 = INTRADAY_TRIGGER_COLUMNS[0], INTRADAY_TRIGGER_COLUMNS[1]
    feat_dict[t1] = np.random.choice([0, 1], n_samples, p=[0.95, 0.05]).astype(float)
    feat_dict[t2] = feat_dict[t1].copy() # Duplicate for quality gate test
    
    # Add some locations
    l1 = LOCATION_FILTER_COLUMNS[0]
    feat_dict[l1] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2]).astype(float)
    
    # Add some regimes (continuous)
    feat_dict['regime_vol'] = np.random.normal(0, 1, n_samples)
    feat_dict['regime_trend'] = np.random.normal(0, 1, n_samples)
    
    # Target
    fwd_ret = 0.05 * feat_dict[t1] * feat_dict[l1] + np.random.normal(0, 0.01, n_samples)
    
    cfg = {
        "regime_cs_weight": 0.5,
        "min_feature_support": 5,
        "min_support_count_validation": 5,
        "sign_dead_zone": 1e-6,
        "seeds": [42],
        "output_dir": "./test_lgbm_outputs"
    }
    
    # 2. Test Feature Hardening
    print("\n[Testing Feature Hardening]")
    fp = FeatureProcessor()
    # Mocking RIDGE and TEST feature keys for the test
    import lgbm_based_mask_generation
    lgbm_based_mask_generation.RIDGE_FEATURE_COLS = ['regime_vol']
    lgbm_based_mask_generation.TEST_FEATURE_KEYS = ['regime_trend']
    
    X, metadata, audit = fp.prepare_features(feat_dict, timestamps, symbol_codes, cfg)
    
    print(f"Audit Status:\n{audit['status'].value_counts()}")
    # Verify t2 (duplicate) was dropped
    if t2 in audit[audit['status']=='dropped']['feature_name'].values:
        print(f"PASS: Duplicate feature {t2} dropped.")
    else:
        print(f"FAIL: Duplicate feature {t2} NOT dropped.")

    # 3. Test Full Pipeline
    print("\n[Testing Full Pipeline Orchestration]")
    registry = run_lgbm_mask_generation(data, feat_dict, fwd_ret, cfg)
    
    if not registry.empty:
        print(f"PASS: Registry generated with {len(registry)} rules.")
        print("\nTop Managed Rules:")
        print(registry[~registry['dominated_by_parent']].head(5))
        
        # Check for dominance markers
        dominated_count = registry['dominated_by_parent'].sum()
        print(f"Number of dominated rules pruned: {dominated_count}")
        
    else:
        print("FAIL: Registry is empty.")

if __name__ == "__main__":
    test_lgbm_hardening()
