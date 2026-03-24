#!/usr/bin/env python3
import pandas as pd
import os

feat_dir = '/Users/remyroche/Documents/Ares/data/features/20260321_140000'
path = os.path.join(feat_dir, 'symbol=BTC_USDT.parquet')
df = pd.read_parquet(path)
print(f'BTC_USDT.parquet: shape={df.shape}')
print(f'Columns (first 20): {list(df.columns[:20])}')
print(f'Non-null counts for first 20 cols:')
for col in df.columns[:20]:
    non_null = df[col].notna().sum()
    print(f'  {col}: {non_null}/{len(df)}')

# Check for specific features the miner needs
print('\nChecking specific features:')
for col in ['regime:trend_slope_12', 'LOC_45_AtRangeBreakoutZone_Long']:
    if col in df.columns:
        non_null = df[col].notna().sum()
        print(f'{col}: {non_null}/{len(df)} non-null')
    else:
        print(f'{col}: NOT FOUND')
