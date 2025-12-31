
import pandas as pd
import numpy as np
from pathlib import Path
from src.training.steps.labeling.orthogonal_label_generation import orthogonal_label_generation
from src.training.steps.labeling.label_based_layer_3 import layer3_analyst_lgbm
from src.training.steps.labeling.label_based_layer_4 import _generate_report
from src.utils.tprint import tprint_info

def run_verification():
    tprint_info("🚀 Starting Focused Verification of HPO Reporting...")
    
    # Create mock data (2000 bars for enough statistical power)
    n_bars = 2000
    dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='15min')
    df = pd.DataFrame({
        'close': np.cumsum(np.random.normal(0, 0.01, n_bars)) + 100,
        'volume': np.random.uniform(10, 100, n_bars),
        'high': np.random.uniform(101, 105, n_bars),
        'low': np.random.uniform(95, 99, n_bars),
        'open': np.random.uniform(99, 101, n_bars),
    }, index=dates)
    
    # Add required technical columns for HPO logic
    df['volatility_1d'] = df['close'].rolling(100).std().fillna(df['close'].std())
    df['returns'] = df['close'].pct_change().fillna(0)
    
    out_dir = Path("outcomes/verification_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Test Layer 2 Reporting
    tprint_info("🧪 Testing Layer 2 (Orthogonal Label Generation)...")
    try:
        # We need enough data for gates to pass, or at least for diagnostics to run.
        # orthogonal_label_generation(df) internally saves reporting.
        # We override cfg to point to our test dir if possible, but the function uses its own logic.
        # Let's just run it and see if the md report appears in outcomes/.
        geoms = orthogonal_label_generation(df)
        tprint_info(f"✅ Layer 2 finished. Generated {len(geoms)} geometries.")
    except Exception as e:
        tprint_info(f"❌ Layer 2 failed: {e}")

    # 3. Test Layer 3 Reporting
    tprint_info("🧪 Testing Layer 3 (Analyst Meta-Labeling)...")
    try:
        # Mock metrics and importance
        mock_metrics = [{'gid': 'G1', 'score': 0.6, 'auc': 0.6}, {'gid': 'G2', 'score': 0.55, 'auc': 0.58}]
        mock_mdi = {'feat1': 0.1, 'feat2': 0.05}
        mock_shap = {'feat1': 0.08, 'feat2': 0.03}
        mock_df = pd.DataFrame({'target': [1, 0, 1, 0]*25, 'meta_prob': np.random.uniform(0.4, 0.6, 100)})
        mock_cfg = {'symbol': 'TEST', 'timeframe': '15m'}
        
        # We call the reporting function directly since we can
        from src.training.steps.labeling.label_based_layer_3 import _generate_layer3_meta_report
        _generate_layer3_meta_report(mock_df, mock_metrics, mock_mdi, mock_shap, out_dir, "test_ts", mock_cfg)
        tprint_info("✅ Layer 3 report generated.")
    except Exception as e:
        tprint_info(f"❌ Layer 3 failed: {e}")

    # 4. Test Layer 4 Reporting
    tprint_info("🧪 Testing Layer 4 (Position Sizing)...")
    try:
        mock_l4_df = pd.DataFrame({
            'layer4_weight': np.random.uniform(0.5, 2.0, 100),
            'layer4_return': np.random.normal(0, 0.01, 100)
        })
        mock_l4_metrics = {
            'l4_mean_ret': 0.001,
            'l4_win_rate': 0.52,
            'l4_avg_weight': 1.2,
            'l4_sl_param': 1.5,
            'l4_gap_param': 0.2
        }
        _generate_report(mock_l4_df, mock_l4_metrics, {'outcomes_dir': str(out_dir)})
        tprint_info("✅ Layer 4 report generated.")
    except Exception as e:
        tprint_info(f"❌ Layer 4 failed: {e}")

    tprint_info("🏁 Verification Script Finished.")

if __name__ == "__main__":
    run_verification()
