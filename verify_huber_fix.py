
import numpy as np
import pandas as pd
from src.training.steps.labeling.irm_regime_pipeline import IRMLinearClassifier, IRMLinearRegressor
from src.utils.huber_regressor_for_trees import prepare_huber_teacher_outputs, _fit_huber

def test_huber_fallback():
    print("Testing Huber Fallback Sequence...")
    # Generate data with NO invariant signal (IRM should kill it) but some average signal
    # env 1: y = x + noise
    # env 2: y = -x + noise
    # Total: y = 0*x + noise (but Huber might pick up 0)
    
    X = np.random.randn(200, 2)
    y = np.random.randn(200)
    w = np.ones(200)
    
    # Force some correlation in the whole set
    y += 0.5 * X[:, 0]
    
    # env_indices representing asset regimes
    env_indices = [np.arange(0, 100), np.arange(100, 200)]
    
    print("\nStage 1: Testing Regressor Fallback")
    # This should trigger "Relaxed IRM" or "Standard Huber"
    model = _fit_huber(X, y, w, epsilon=1.35, alpha=0.1, max_iter=1000, irm_lambda=1.0, is_classification=False, verbose=True)
    print(f"Regressor Coef: {model.coef_}")
    assert np.max(np.abs(model.coef_)) > 0, "Regressor failed to recover signal"

    print("\nStage 2: Testing Classifier Fallback")
    y_bin = (y > 0).astype(int)
    model_c = _fit_huber(X, y_bin, w, epsilon=1.35, alpha=0.1, max_iter=1000, irm_lambda=1.0, is_classification=True, verbose=True)
    print(f"Classifier Coef: {model_c.coef_}")
    assert np.max(np.abs(model_c.coef_)) > 0, "Classifier failed to recover signal"
    
    if hasattr(model_c, 'predict_proba'):
        probs = model_c.predict_proba(X[:5])
        print(f"Classifier Probs Sample: {probs[:, 1]}")
        assert probs.shape == (5, 2), "Incorrect proba shape"

def test_prepare_huber_outputs():
    print("\nTesting prepare_huber_teacher_outputs with Classification")
    X = pd.DataFrame(np.random.randn(200, 5), columns=[f'feat_{i}' for i in range(5)])
    y = (X.sum(axis=1) > 0).astype(int)
    
    outputs = prepare_huber_teacher_outputs(
        X, y, 
        is_classification=True,
        n_time_splits=2,
        use_irm=False
    )
    
    print(f"Selected Features: {outputs['selected_features']}")
    print(f"Warm Start Mean: {outputs['warm_start']['train'].mean():.4f}")
    assert len(outputs['selected_features']) > 0, "No features selected in classification"
    assert outputs['warm_start']['train'].max() <= 1.0, "Probabilities exceed 1.0"
    assert outputs['warm_start']['train'].min() >= 0.0, "Probabilities below 0.0"

if __name__ == "__main__":
    try:
        test_huber_fallback()
        test_prepare_huber_outputs()
        print("\n✅ Verification SUCCESS")
    except Exception as e:
        print(f"\n❌ Verification FAILED: {e}")
        import traceback
        traceback.print_exc()
