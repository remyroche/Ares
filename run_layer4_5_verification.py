
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# Add src to path
sys.path.append(os.getcwd())

from src.training.steps.labeling.label_based_layer_4 import Layer4RiskFilter, compute_final_score_product, compute_layer4_regime_features
from src.training.steps.labeling.label_based_layer_5 import Layer5PositionSizer
from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error

def run_verification():
    tprint_info("Starting Layer 4 & 5 Verification...")

    # 1. Load Layer 3 OOF Predictions
    l3_preds_path = Path("outcomes/layer3_oof_preds.csv")
    if not l3_preds_path.exists():
        tprint_error(f"Layer 3 OOF preds not found at {l3_preds_path}")
        return

    df = pd.read_csv(l3_preds_path, index_col=0, parse_dates=True)
    tprint_success(f"Loaded {len(df)} Layer 3 predictions.")

    # 2. Load Layer 4 Model
    l4_model_path = Path("outcomes/layer4_risk_filter.joblib")
    l4_model = None
    if l4_model_path.exists():
        try:
            l4_model = Layer4RiskFilter.load(str(l4_model_path))
            tprint_success(f"Loaded Layer 4 model from {l4_model_path}")
        except Exception as e:
            tprint_warning(f"Failed to load Layer 4 model: {e}")
    else:
        tprint_warning("Layer 4 model artifact not found. Will proceed without L4 filtering if possible.")

    # 3. Prepare Layer 4 Features
    # df should contain OHLCV and other features.
    # We need to ensure we have enough columns for compute_layer4_regime_features
    # It needs: close, high, low (volume optional?)
    
    # Check for required columns
    req_cols = ['close', 'high', 'low']
    missing = [c for c in req_cols if c not in df.columns]
    
    l4_probs = None
    
    if not missing and l4_model:
        tprint_info("Computing Layer 4 regime features...")
        try:
            feats = compute_layer4_regime_features(df)
            
            # Fix: Add l3_prob feature if model expects it
            if 'l3_prob' in l4_model.feature_names:
                tprint_info("Adding l3_prob to Layer 4 features...")
                feats['l3_prob'] = df['meta_prob']
            
            tprint_info(f"Computed {len(feats.columns)} L4 features.")
            
            # Predict
            l4_probs = l4_model.predict_proba(feats)
            df['l4_prob'] = l4_probs
            tprint_success("Generated Layer 4 probabilities.")
            
            # Compute Combined Score
            df['final_prob'] = compute_final_score_product(df['meta_prob'].values, l4_probs)
            
            # Diagnostics
            print("\n--- Probability Diagnostics ---")
            print(f"Meta Prob Q50: {df['meta_prob'].median():.4f}, Q95: {df['meta_prob'].quantile(0.95):.4f}, Max: {df['meta_prob'].max():.4f}")
            print(f"L4 Prob   Q50: {np.median(l4_probs):.4f}, Q95: {np.quantile(l4_probs, 0.95):.4f}, Max: {np.max(l4_probs):.4f}")
            print(f"Final Prob Q50: {df['final_prob'].median():.4f}, Q95: {df['final_prob'].quantile(0.95):.4f}, Max: {df['final_prob'].max():.4f}")
            
        except Exception as e:
            tprint_error(f"Error during Layer 4 execution: {e}")
            df['final_prob'] = df['meta_prob'] # Fallback
            print(f"Meta Prob Q50: {df['meta_prob'].median():.4f}, Q95: {df['meta_prob'].quantile(0.95):.4f}, Max: {df['meta_prob'].max():.4f}")


    # 4. Layer 5 Position Sizing
    tprint_info("Running Layer 5 Position Sizing...")
    
    # Rename/Prepare columns for Layer 5
    if 'l2_Consensus_target' in df.columns: 
         df['target'] = df['l2_Consensus_target']
    
    if 'realized_return' not in df.columns:
        tprint_info("Joining with sized_events/labeled_data to get realized_return...")
        found_ret = False
        
        # Try layer5_sized_events first
        for p in sorted(Path("outcomes").glob("layer5_sized_events_*.csv"), reverse=True):
             try:
                 ld = pd.read_csv(p, index_col=0, parse_dates=True)
                 # Check explicitly for realized_return or 'event_return'
                 ret_col = None
                 if 'realized_return' in ld.columns: ret_col = 'realized_return'
                 elif 'event_return' in ld.columns: ret_col = 'event_return'
                 
                 if ret_col:
                     common = df.index.intersection(ld.index)
                     if len(common) > 10: # Ensure valid overlap
                         df.loc[common, 'realized_return'] = ld.loc[common, ret_col]
                         found_ret = True
                         tprint_success(f"Joined realized_return from {p.name}")
                         break
             except: continue
             
        if not found_ret:
             tprint_error("Could not find realized_return for verification. Backtest will be limited.")
             return

    # Initialize Layer 5
    # Use parameters from default or config. 
    # For verification, we use standard defaults: p_min=0.51, gamma=1.0 to start?
    # Or better: p_min=0.5, p_max=0.9
    
    sizer = Layer5PositionSizer(
        oof_df=df,
        p_col='final_prob',
        return_col='realized_return',
        p_min=0.05, # Debugging: Very low threshold
        p_max=0.9,
        gamma=1.0,
        gate_mode='p_min',
    )
    
    metrics = sizer.run_backtest()
    
    tprint_success("Layer 5 Run Complete.")
    print("\n--- Layer 5 Verification Results ---")
    for k, v in metrics.items():
        if not isinstance(v, dict):
            print(f"{k}: {v}")
    
    # Save specific verification report
    with open("outcomes/verification_layer4_5_report.md", "w") as f:
        f.write("# Layer 4 & 5 Verification Report\n\n")
        for k, v in metrics.items():
             if not isinstance(v, dict):
                f.write(f"- {k}: {v}\n")

if __name__ == "__main__":
    run_verification()
