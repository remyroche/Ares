#!/usr/bin/env python3.11
"""Delete corrupted boolean feature columns from feature cache."""
import pandas as pd
from pathlib import Path
from extreme_price_movements.intraday_crypto_library import LOCATION_FILTER_COLUMNS, INTRADAY_TRIGGER_COLUMNS

def main():
    feat_dir = Path('/Users/remyroche/Documents/Ares/data/features/20260313_000000')
    parquet_files = list(feat_dir.glob('symbol=*.parquet'))
    cols_to_delete = list(LOCATION_FILTER_COLUMNS) + list(INTRADAY_TRIGGER_COLUMNS)

    print(f'Processing {len(parquet_files)} files...')
    print(f'Columns to delete: {len(cols_to_delete)}')
    
    deleted_count = 0
    skipped_count = 0
    for i, f in enumerate(parquet_files):
        try:
            df = pd.read_parquet(f)
            cols_in_file = [c for c in cols_to_delete if c in df.columns]
            if cols_in_file:
                df = df.drop(columns=cols_in_file)
                df.to_parquet(f, index=False)
                deleted_count += len(cols_in_file)
                print(f'  {f.name}: deleted {len(cols_in_file)} columns')
        except Exception as e:
            print(f'  {f.name}: ERROR - {str(e)[:100]}')
            skipped_count += 1

        if (i + 1) % 10 == 0:
            print(f'  Progress: {i+1}/{len(parquet_files)}')

    print(f'\nDone. Deleted {deleted_count} columns total. Skipped {skipped_count} corrupted files.')

if __name__ == '__main__':
    main()
