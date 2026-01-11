
import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
import sys

# Ensure project root is in python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_warning, tprint_error

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def diagnose_features(file_path: str):
    """
    Diagnose feature statistics from a parquet file.
    """
    tprint_info(f"🔍 Diagnosing features from: {file_path}")
    
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        tprint_error(f"Failed to read parquet file: {e}")
        return

    features = df.select_dtypes(include=[np.number])
    tprint_info(f"📊 Found {len(features.columns)} numeric features")

    stats = []
    for col in features.columns:
        series = features[col]
        n_zeros = (series == 0).sum()
        n_nans = series.isna().sum()
        variance = series.var()
        mean = series.mean()
        std = series.std()
        
        stat = {
            'feature': col,
            'mean': mean,
            'std': std,
            'min': series.min(),
            'max': series.max(),
            'zeros_pct': (n_zeros / len(series)) * 100,
            'nans_pct': (n_nans / len(series)) * 100,
            'variance': variance
        }
        stats.append(stat)

    stats_df = pd.DataFrame(stats)
    
    # Check for problematic features
    zero_variance = stats_df[stats_df['variance'] == 0]
    high_nans = stats_df[stats_df['nans_pct'] > 50]
    high_zeros = stats_df[stats_df['zeros_pct'] > 90]
    
    if not zero_variance.empty:
        tprint_warning(f"⚠️  {len(zero_variance)} features have ZERO variance:")
        for _, row in zero_variance.iterrows():
             print(f"   - {row['feature']}")
             
    if not high_nans.empty:
        tprint_warning(f"⚠️  {len(high_nans)} features have >50% NaNs:")
        for _, row in high_nans.head(5).iterrows():
             print(f"   - {row['feature']} ({row['nans_pct']:.1f}%)")

    if not high_zeros.empty:
        tprint_warning(f"⚠️  {len(high_zeros)} features have >90% Zeros:")
        for _, row in high_zeros.head(5).iterrows():
             print(f"   - {row['feature']} ({row['zeros_pct']:.1f}%)")

    # Save report
    output_path = Path("outcomes") / f"feature_diagnostics_{Path(file_path).stem}.csv"
    stats_df.to_csv(output_path, index=False)
    tprint_info(f"✅ Full feature stats saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose feature statistics")
    parser.add_argument("file_path", type=str, help="Path to parquet file containing features")
    args = parser.parse_args()
    
    setup_logging()
    diagnose_features(args.file_path)
