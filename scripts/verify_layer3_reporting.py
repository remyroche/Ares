import pandas as pd
import numpy as np
from src.training.steps.labeling.layer3.core import layer3_analyst_lgbm
from pathlib import Path
import os

# Setup dummy data
n = 500
df = pd.DataFrame({
    'close': np.random.normal(100, 1, n).cumsum(),
    'volume': np.random.normal(1000, 100, n),
    'high': np.random.normal(102, 1, n),
    'low': np.random.normal(98, 1, n),
    'side': np.random.choice([-1, 1], n),
    'bin': np.random.choice([0, 1], n),
    'regime_label': np.random.choice(['Quiet', 'Trending', 'Chaos'], n)
}, index=pd.date_range('2023-01-01', periods=n, freq='15min'))

# Add some base model columns
base_cols = [f'prob_{i}' for i in range(5)]
for c in base_cols:
    df[c] = np.random.uniform(0, 1, n)

# Add some anchor and drift features to trigger structural reporting
df['anchor_0_pc1'] = np.random.normal(0, 1, n)
df['anchor_0_stability'] = np.random.uniform(0.8, 1.0, n)
df['anchor_1_pc1'] = np.random.normal(0, 1, n)
df['anchor_1_stability'] = np.random.uniform(0.5, 0.9, n)

config = {
    'symbol': 'BTCUSDT',
    'timeframe': '15min',
    'outcomes_dir': 'outcomes_test_reporting',
    'fast_mode': True,
    'comprehensive_metrics_enabled': True,
    'geometry_metrics': [
        {'geometry_id': 'geo_1', 'score': 0.85, 'auc': 0.62, 'ic': 0.05},
        {'geometry_id': 'geo_2', 'score': 0.75, 'auc': 0.58, 'ic': 0.03}
    ]
}

# Run the pipeline
try:
    l3_df, results = layer3_analyst_lgbm(
        oof_df=df,
        base_model_cols=['side'],
        target_col='bin',
        config=config
    )
    print("✅ Layer 3 pipeline executed successfully.")
    
    # Check if reports exist
    out_dir = Path('outcomes_test_reporting')
    reports = list(out_dir.glob('layer3_*.md'))
    print(f"📊 Found {len(reports)} markdown reports:")
    for r in reports:
        print(f"   - {r.name}")
        
    csvs = list(out_dir.glob('layer3_*.csv'))
    print(f"📊 Found {len(csvs)} CSV exports:")
    for c in csvs:
        print(f"   - {c.name}")
        
except Exception as e:
    print(f"❌ Layer 3 pipeline failed: {e}")
    import traceback
    traceback.print_exc()
