import pandas as pd
import numpy as np
from src.feature_generation.categories.layer3_specific_features import generate_layer3_features
from src.training.steps.labeling.layer3.feature_engineering import apply_layer3_feature_selection
from src.utils.tprint import tprint_info, tprint_success

def verify_layer3_refinement():
    tprint_info("🧪 Starting Layer 3 Refinement Verification...")
    
    # 1. Create dummy data
    n_bars = 500
    df = pd.DataFrame({
        'close': np.random.normal(100, 1, n_bars).cumsum(),
        'volume': np.random.normal(1000, 100, n_bars),
        'high': np.random.normal(102, 1, n_bars),
        'low': np.random.normal(98, 1, n_bars),
    }, index=pd.date_range('2023-01-01', periods=n_bars, freq='15min'))
    
    # Add dummy base models
    base_model_cols = [f'prob_{i}' for i in range(10)]
    for col in base_model_cols:
        df[col] = np.random.uniform(0, 1, n_bars)
        
    tprint_info(f"📊 Synthetic data created: {df.shape}")
    
    # 2. Test Feature Generation
    try:
        tprint_info("🏃 Running generate_layer3_features...")
        df_feat = generate_layer3_features(df, base_model_cols)
        
        # Check for expected columns
        anchor_cols = [c for c in df_feat.columns if 'anchor_' in c]
        tprint_info(f"   ✅ Found anchor columns: {anchor_cols}")
        
        if not any('pc1' in c for c in anchor_cols):
            raise ValueError("❌ Missing anchor_pc1 columns!")
        if not any('stability' in c for c in anchor_cols):
            raise ValueError("❌ Missing anchor_stability columns!")
        if not any('disagreement' in c for c in anchor_cols):
            raise ValueError("❌ Missing anchor_disagreement columns!")
            
        tprint_success("✅ Feature Generation successful.")
        
    except Exception as e:
        tprint_info(f"❌ Feature Generation failed: {e}")
        import traceback
        tprint_info(traceback.format_exc())
        return

    # 3. Test Feature Selection
    try:
        tprint_info("🏃 Running apply_layer3_feature_selection...")
        y = pd.Series(np.random.choice([0, 1], n_bars), index=df_feat.index)
        meta_features = [c for c in df_feat.columns if 'anchor_' in c or 'noise' in c]
        X = df_feat[meta_features]
        
        X_selected = apply_layer3_feature_selection(X, y, df_feat[base_model_cols])
        
        tprint_info(f"   ✅ Selection reduced features: {len(X.columns)} -> {len(X_selected.columns)}")
        tprint_success("✅ Feature Selection successful.")
        
    except Exception as e:
        tprint_info(f"❌ Feature Selection failed: {e}")
        import traceback
        tprint_info(traceback.format_exc())
        return

    tprint_success("🎉 All Layer 3 Refinement Verification steps passed!")

if __name__ == "__main__":
    verify_layer3_refinement()
