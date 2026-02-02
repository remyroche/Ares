import sys
import os
import shutil
import pandas as pd
import numpy as np

# Add root to path
sys.path.insert(0, os.path.abspath("."))

# Copy the file to root as a temporary module to allow joblib to import it
# without triggering the heavy __init__.py of the original package.
original_path = "src/training/steps/labeling/de_prado_feature_engine.py"
temp_path = "temp_deprado_engine.py"
if not os.path.exists(original_path):
    print(f"Error: {original_path} not found")
    exit(1)

shutil.copy(original_path, temp_path)

try:
    import temp_deprado_engine as deprado_mod
    DePradoFeatureEngine = deprado_mod.DePradoFeatureEngine
    de_prado_feature_selection = deprado_mod.de_prado_feature_selection

    def test_deprado_engine_multi_asset():
        print("Test: Multi-Asset Groups")
        # 1. Create Synthetic Data
        np.random.seed(42)
        n_samples = 200
        n_features = 10

        dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')

        # Asset A
        X_a = pd.DataFrame(np.random.randn(n_samples, n_features), index=dates, columns=[f'f_{i}' for i in range(n_features)])
        y_a = pd.Series(np.random.randint(0, 2, n_samples), index=dates)
        g_a = pd.Series(['A']*n_samples, index=dates)

        # Asset B (stacked)
        X_b = pd.DataFrame(np.random.randn(n_samples, n_features), index=dates, columns=[f'f_{i}' for i in range(n_features)])
        y_b = pd.Series(np.random.randint(0, 2, n_samples), index=dates)
        g_b = pd.Series(['B']*n_samples, index=dates)

        X = pd.concat([X_a, X_b])
        y = pd.concat([y_a, y_b])
        groups = pd.concat([g_a, g_b])

        # Make some features informative
        X['f_0'] = y.values * 0.5 + np.random.normal(0, 0.5, len(X)) # correlated
        X['f_1'] = X['f_0'] * 0.9 + np.random.normal(0, 0.1, len(X)) # redundant

        # 2. Initialize Engine
        engine = DePradoFeatureEngine(
            n_estimators=10,
            max_clusters=5,
            use_lgbm=True,
            use_regime_clustering=True,
            use_turnover_penalty=True
        )

        # 3. Run Selection with Groups
        print("Running selection...")
        selected = engine.run_selection(X, y, groups=groups)

        print(f"Selected features: {selected}")
        assert len(selected) > 0

        stats = engine.get_feature_stats()
        assert not stats.empty
        assert 'CompositeScore' in stats.columns

        # 4. Run Wrapper
        print("Running wrapper...")
        X_sel, eng = de_prado_feature_selection(X, y, groups=groups, n_estimators=10)
        assert X_sel.shape[1] == len(eng.get_selected_features())

    def test_deprado_engine_no_groups():
        print("\nTest: Single Asset (No Groups)")
        # 1. Create Synthetic Data (Single Asset)
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')

        X = pd.DataFrame(np.random.randn(n_samples, n_features), index=dates, columns=[f'f_{i}' for i in range(n_features)])
        y = pd.Series(np.random.randint(0, 2, n_samples), index=dates)

        engine = DePradoFeatureEngine(n_estimators=10, use_regime_clustering=True, use_turnover_penalty=True)

        selected = engine.run_selection(X, y)
        print(f"Selected features: {selected}")
        assert len(selected) > 0

    if __name__ == "__main__":
        try:
            test_deprado_engine_multi_asset()
            test_deprado_engine_no_groups()
            print("\nAll tests passed!")
        except Exception as e:
            print(f"\nTest failed: {e}")
            import traceback
            traceback.print_exc()
            exit(1)
finally:
    if os.path.exists(temp_path):
        os.remove(temp_path)
