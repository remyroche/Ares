
import sys
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

# Ensure src is in path
sys.path.append(os.getcwd())

from src.training.steps.labeling.de_prado_feature_engine import DePradoFeatureEngine, de_prado_feature_selection

def test_deprado_classification():
    print("\n🧪 Testing DePradoFeatureEngine (Classification)...")
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=200, 
        n_features=20, 
        n_informative=10, 
        n_redundant=5, 
        n_repeated=0, 
        random_state=42
    )
    
    feature_names = [f"feat_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y)
    
    # Initialize Engine with all new features
    engine = DePradoFeatureEngine(
        n_estimators=50, # Low for speed
        max_clusters=5,
        use_lgbm=True, # Test LGBM path
        use_regime_clustering=True,
        use_denoising=True,
        use_partial_corr=True,
        use_turnover_penalty=True,
        topk_freq_threshold=0.1 # Low threshold to ensure selection
    )
    
    selected_features = engine.run_selection(X_df, y_series)
    
    print(f"✅ Selected {len(selected_features)} features: {selected_features}")
    
    stats = engine.get_feature_stats()
    print("Stats head:")
    print(stats.head())
    
    assert len(selected_features) > 0, "Should select at least some features"
    assert 'CompositeScore' in stats.columns
    
def test_deprado_regression():
    print("\n🧪 Testing DePradoFeatureEngine (Regression)...")
    
    X, y = make_regression(
        n_samples=200,
        n_features=20,
        n_informative=10,
        random_state=42
    )
    
    feature_names = [f"reg_feat_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y)
    
    # Test ExtraTrees path (use_lgbm=False)
    engine = DePradoFeatureEngine(
        n_estimators=50,
        use_lgbm=False, 
        use_regime_clustering=False, # Disable to test branch
        use_denoising=False
    )
    
    selected_features = engine.run_selection(X_df, y_series)
    print(f"✅ Selected {len(selected_features)} features: {selected_features}")
    
    assert len(selected_features) > 0

if __name__ == "__main__":
    try:
        test_deprado_classification()
        test_deprado_regression()
        print("\n🎉 All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
