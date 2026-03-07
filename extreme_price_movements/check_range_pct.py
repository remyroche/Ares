import pandas as pd
import numpy as np
from pathlib import Path
from extreme_price_movements.data_store import load_latest_pipeline_features

feats = load_latest_pipeline_features("/Users/remyroche/Documents/Ares/data")
print(f"Loaded {len(feats)} features.")
for name in ["range_pct", "range_16h_pct", "range_12h_pct", "range_8h_pct", "range_24h_pct"]:
    if name in feats:
        df = feats[name]
        valid = df.notna().sum().sum()
        total = df.size
        print(f"{name}: {valid}/{total} valid ({valid/total:.2%})")
    else:
        print(f"{name}: NOT FOUND")
