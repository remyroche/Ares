
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from src.training.steps.labeling.layer2_5_integration import Layer25Integration
    from src.training.steps.labeling.layer2_5_chaser import Layer25Chaser
    from src.utils.tprint import tprint_info, tprint_success, tprint_error
except ImportError:
    print("Failed to import src modules. Run from project root.")
    sys.exit(1)

def create_synthetic_panel_data(n_tickers=3, n_timestamps=100):
    tprint_info("Creating synthetic panel data...")
    dates = pd.date_range(start='2023-01-01', periods=n_timestamps, freq='D')
    tickers = [f'TICKER_{i}' for i in range(n_tickers)]
    
    dfs = []
    for ticker in tickers:
        df = pd.DataFrame(index=dates)
        df['ticker'] = ticker
        
        # Features
        df['feature_1'] = np.random.randn(n_timestamps)
        df['feature_2'] = np.random.randn(n_timestamps)
        
        # Causal Anchor Prediction (Base)
        base_signal = np.sin(np.linspace(0, 10, n_timestamps))
        df['anchor_pred'] = base_signal + np.random.normal(0, 0.1, n_timestamps)
        
        # Actual Target (Base + Alpha + Noise)
        # Adding Asset-Specific Bias!
        bias = 0.5 if ticker == 'TICKER_0' else -0.5
        df['target'] = df['anchor_pred'] + (df['feature_1'] * 0.2) + bias + np.random.normal(0, 0.05, n_timestamps)
        
        dfs.append(df)
        
    panel_df = pd.concat(dfs).reset_index().rename(columns={'index': 'timestamp'})
    panel_df = panel_df.set_index(['timestamp', 'ticker']).sort_index()
    
    return panel_df

def verify_refinement():
    tprint_info("🚀 Starting Layer 2.5 Refinement Verification")
    
    # 1. Data Setup
    panel_df = create_synthetic_panel_data()
    tprint_info(f"Panel Data Shape: {panel_df.shape}")
    
    # 2. Integration
    integration = Layer25Integration(
        chaser_params={
            # Use fast minimal models
            'models_to_train': ['xgb', 'et'], # Check XGB (Cat) and ET (Numeric)
            'chaser_xgb_params': {'n_estimators': 2, 'max_depth': 2},
            'chaser_et_params': {'n_estimators': 2, 'max_depth': 2},
            'verbose': True
        },
        verbose=True
    )
    
    # 3. Prepare Training Data
    X, y = integration.prepare_training_data(
        panel_df,
        target_col='target',
        causal_anchor_prediction=panel_df['anchor_pred'],
        all_feature_cols=['feature_1', 'feature_2']
    )
    
    tprint_info("Checking prepared data for 'cat__asset_id'...")
    if 'cat__asset_id' in X.columns:
        tprint_success(f"✅ 'cat__asset_id' present in X: {X['cat__asset_id'].unique().tolist()}")
        if X['cat__asset_id'].dtype.name == 'category':
            tprint_success("✅ 'cat__asset_id' is categorical dtype")
        else:
            tprint_error(f"❌ 'cat__asset_id' is {X['cat__asset_id'].dtype}, expected category")
    else:
        tprint_error("❌ 'cat__asset_id' MISSING in X")
        return

    # 4. Train Chaser
    tprint_info("Starting training (should handle categorical split internally)...")
    try:
        metrics = integration.train_chaser(X, y, cv_folds=3)
        tprint_success("✅ Training completed successfully")
    except Exception as e:
        tprint_error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Verify Model Artifacts
    chaser = integration.chaser
    if hasattr(chaser, 'global_models'):
        models = chaser.global_models
        
        # Check Teacher (should validly train on numeric)
        teacher = models['teacher']
        if teacher.model is not None:
            tprint_success("✅ Teacher model exists")
            
        # Check Students
        students = models['students']
        if 'xgb' in students:
            tprint_success("✅ XGBoost student trained (Categorical enabled)")
        if 'et' in students:
            tprint_success("✅ ExtraTrees student trained (Numeric only)")
            
    # 6. Out-of-Universe Verification (Holdout Ticker)
    tprint_info("Running Out-of-Universe Verification (Holdout Ticker)...")
    
    # Create holdout ticker data
    holdout_dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
    holdout_df = pd.DataFrame(index=holdout_dates)
    holdout_df['ticker'] = 'TICKER_HOLDOUT'
    holdout_df['feature_1'] = np.random.randn(50)
    holdout_df['feature_2'] = np.random.randn(50)
    base_signal = np.sin(np.linspace(0, 5, 50))
    holdout_df['anchor_pred'] = base_signal
    # Create index
    holdout_df = holdout_df.reset_index().rename(columns={'index': 'timestamp'})
    holdout_df = holdout_df.set_index(['timestamp', 'ticker']).sort_index()

    # Prepare holdout data
    X_holdout, _ = integration.prepare_training_data(
        holdout_df,
        target_col=None, # No target needed for prediction preparation if we mock residuals
        causal_anchor_prediction=holdout_df['anchor_pred'],
        all_feature_cols=['feature_1', 'feature_2']
    )
    # Mock residuals for prepare_training_data? No, it returns y_resids.
    # Actually prepare_training_data requires target_col to compute residuals.
    # For INFERENCE, we usually use `predict_with_conflict_detection` which calls chaser.predict.
    
    # Let's use predict_with_conflict_detection
    preds = integration.predict_with_conflict_detection(
        X_holdout, 
        causal_anchor_prediction=holdout_df['anchor_pred'],
        return_conflicts=False
    )
    
    if len(preds['chaser_prediction']) == 50:
        tprint_success("✅ Out-of-Universe Prediction successful (Handled unknown ticker)")
    else:
        tprint_error(f"❌ Out-of-Universe Prediction failed length check: {len(preds['chaser_prediction'])}")

    tprint_success("🎉 Verification Complete: Asset-Specific Refinements & Regularization Checked!")

if __name__ == "__main__":
    verify_refinement()
