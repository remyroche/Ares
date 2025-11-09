#!/usr/bin/env python3
"""Test that the index fix works correctly"""
import pandas as pd
import numpy as np

# Simulate the scenario
print("🧪 Testing index fix...")

# Create protected_data with 44381 rows
protected_data = pd.DataFrame({
    'feature1': np.random.rand(44381),
    'feature2': np.random.rand(44381),
})

# Add regime prob columns - only first 480 rows have values, rest are NaN
regime_prob_cols = ['regime_0_prob', 'regime_1_prob', 'regime_2_prob']
for col in regime_prob_cols:
    protected_data[col] = np.nan
    protected_data.loc[:479, col] = np.random.rand(480)

print(f"📊 Protected data shape: {protected_data.shape}")
print(f"📊 Regime prob columns: {regime_prob_cols}")

# Apply the fix
valid_regime_mask = protected_data[regime_prob_cols].notna().any(axis=1)
valid_regime_idx = protected_data[valid_regime_mask].index

print(f"📊 Valid regime rows: {len(valid_regime_idx)} out of {len(protected_data)}")
print(f"📊 Valid regime index range: {valid_regime_idx.min()} to {valid_regime_idx.max()}")

# Create model predictions (480 values)
model_predictions = {
    'model1_regime_0_prob': np.random.rand(480),
    'model1_regime_1_prob': np.random.rand(480),
}

print(f"📊 Model predictions length: {len(list(model_predictions.values())[0])}")

# Try to create DataFrame
try:
    predictions_df = pd.DataFrame(model_predictions, index=valid_regime_idx)
    print(f"✅ SUCCESS! predictions_df shape: {predictions_df.shape}")
    print(f"✅ Index matches: {len(predictions_df) == 480}")
except ValueError as e:
    print(f"❌ FAILED: {e}")
