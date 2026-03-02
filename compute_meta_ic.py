import pandas as pd
import numpy as np
import glob

files = glob.glob("/Users/remyroche/Documents/Ares/data/artifacts/20260214_190000/meta_oof/*.parquet")
for f in sorted(files):
    df = pd.read_parquet(f)
    name = f.split('/')[-1].replace('meta_oof_', '').replace('.parquet', '')
    
    # Check what columns exist
    pred_col = 'meta_pred' if 'meta_pred' in df.columns else (
        'oof_prob' if 'oof_prob' in df.columns else (
            'pred' if 'pred' in df.columns else df.columns[-1]
        )
    )
    
    if 'y_ret' in df.columns and pred_col in df.columns:
        ic = df[['y_ret', pred_col]].corr(method='spearman').iloc[0, 1]
    else:
        ic = np.nan
        
    print(f"{name:30s} | N={len(df)} | IC={ic:.4f}")
