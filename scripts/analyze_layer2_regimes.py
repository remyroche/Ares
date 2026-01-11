
import json
import pandas as pd
import numpy as np
import re

def parse_family_string(family_str):
    meta = {}
    
    # Infer Regime from Type
    if 'INVENTORY' in family_str: 
        meta['regime'] = 'Mean Reversion (Liquidity)'
    elif 'VOLUME_SURGE' in family_str or 'TREND' in family_str: 
        meta['regime'] = 'Breakout / Trend'
    elif 'VOLATILITY' in family_str or 'SHOCK' in family_str: 
        meta['regime'] = 'High Volatility'
    elif 'SURPRISE' in family_str: 
        meta['regime'] = 'Structural Break'
    else: 
        meta['regime'] = 'General / Unknown'
        
    return pd.Series(meta)

def analyze_layer2_regimes():
    log_file = "/Users/remyroche/Documents/Ares/outcomes/layer2_raw_metric_log_20260108_220716.json"
    print(f"Reading {log_file}...")
    
    try:
        with open(log_file, 'r') as f:
            data = json.load(f)
            
        df = pd.DataFrame(data)
        
        # Parse Regime
        df_meta = df['family'].apply(parse_family_string)
        df = pd.concat([df, df_meta], axis=1)
        
        print("\n=== Estimated Performance per Market Regime ===")
        regime_stats = df.groupby('regime')[['ic', 'lift', 'stability', 'f_stat']].agg(['mean', 'max', 'count'])
        print(regime_stats.round(4).to_string())
        
        print("\n=== Best Signal per Regime ===")
        for regime in df['regime'].unique():
            best = df[df['regime'] == regime].sort_values('ic', ascending=False).iloc[0]
            print(f"\nRegime: {regime}")
            print(f"  Best IC: {best['ic']:.4f}")
            print(f"  Best Lift: {best['lift']:.4f}")
            print(f"  Signal: {best['family'][:60]}...")

    except Exception as e:
        print(f"Analysis failed: {e}")

if __name__ == "__main__":
    analyze_layer2_regimes()
