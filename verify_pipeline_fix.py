import pandas as pd
import numpy as np
from unittest.mock import MagicMock
import sys

# Mock modules
sys.modules['econml'] = MagicMock()
sys.modules['econml.dml'] = MagicMock()
sys.modules['vectorbt'] = MagicMock()

# Import Layer 2
from src.training.steps.labeling.label_based_layer_2 import LabelBasedLayer2

def verify_pipeline_fix():
    print("🚀 Verifying Pipeline Fix: Base Feature Merging...")

    # 1. Create Mock Data
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='15min')
    df = pd.DataFrame({
        'open': np.random.rand(1000) * 100,
        'high': np.random.rand(1000) * 100,
        'low': np.random.rand(1000) * 100,
        'close': np.random.rand(1000) * 100,
        'volume': np.random.rand(1000) * 1000
    }, index=dates)

    df['high'] = df[['open', 'close']].max(axis=1) + 1.0
    df['low'] = df[['open', 'close']].min(axis=1) - 1.0
    df['volatility_1d'] = df['close'].pct_change().rolling(20).std()

    # 2. Setup Layer 2
    layer2 = LabelBasedLayer2(verbose=True)
    layer2.enable_causal_framework = True
    layer2.causal_discovery_enabled = True
    layer2.causal_specialists_enabled = True
    layer2.causal_surprise_enabled = True
    layer2.causal_engineering_enabled = True

    # Mock Regime Generation
    def mock_generate_regimes(df):
        n = len(df)
        regimes = ['Quiet'] * (n//2) + ['Trending'] * (n - n//2)
        layer2.regime_labels = pd.Series(regimes, index=df.index)
        return layer2.regime_labels
    layer2._generate_regimes = mock_generate_regimes

    # Mock Discovery (Returning 'close' as part of graph to force its inclusion)
    layer2._run_causal_discovery = MagicMock(return_value={'close': ['volume']})

    # Mock Specialists
    layer2._initialize_causal_specialists = MagicMock(return_value={'spec_1': pd.Series(np.random.rand(500), index=df.index[:500])})

    # Mock Events
    layer2._generate_causal_surprise_events = MagicMock(return_value=pd.DataFrame({'surprise': 1}, index=df.index[:10]))

    # Mock Engineering (Returns full DF + new features)
    def mock_engineering(df_in, graph):
        # Verify enrichment happened
        if 'spec_spec_1' not in df_in.columns:
            raise ValueError("Specialist columns missing in engineering input!")

        out = df_in.copy()
        out['engineered_feat'] = 1.0
        return out, {}
    layer2._apply_causal_feature_engineering = mock_engineering

    # Other mocks
    layer2._compute_causal_targets = MagicMock(return_value=pd.DataFrame({'target': 1}, index=df.index))
    layer2._train_causal_models = MagicMock(return_value=([], {}))
    layer2._run_causal_oof_analytics = MagicMock(return_value={})
    layer2._generate_causal_reports = MagicMock()
    layer2._save_artifacts = MagicMock()

    # 3. Run Pipeline
    try:
        results = layer2._run_causal_denoising_pipeline(df)
        print("   ✅ Pipeline Executed")

        merged_df = results.get('engineered_df') # In new logic, this is final_df
        cols = merged_df.columns

        # Verify Base Features are Merged (because 'close' is in graph)
        if 'Quiet_close' in cols and 'Trending_close' in cols:
            print("   ✅ Base Features (Quiet_close, Trending_close) Present")
        else:
            print(f"   ❌ Base Features MISSING. Columns: {[c for c in cols if 'close' in c]}")

        # Verify Engineered Features
        if 'Quiet_engineered_feat' in cols:
            print("   ✅ Engineered Features Present")
        else:
            print("   ❌ Engineered Features MISSING")

        # Verify Graph Keys
        graph = results['causal_graph']
        if 'Quiet_close' in graph:
            print("   ✅ Graph Keys Correct")
        else:
            print("   ❌ Graph Keys INCORRECT")

    except Exception as e:
        print(f"   ❌ Pipeline Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_pipeline_fix()
