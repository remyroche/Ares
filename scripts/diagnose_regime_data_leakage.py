#!/usr/bin/env python3
"""
Diagnostic Script: Check Regime Data for Leakage and Alignment Issues

This script inspects the regime_assignments parquet file to identify:
1. Data leakage (features using future information)
2. Feature-label misalignment  
3. Feature quality issues
4. Temporal consistency problems

Run this to understand why HPO gives identical 0.8 scores.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import sys

def load_latest_regime_file(symbol="ETHUSDT"):
    """Load the most recent regime assignments file."""
    cache_path = Path("data_cache/sr_clustering") / symbol
    regime_files = list(cache_path.glob("regime_assignments_*.parquet"))
    
    if not regime_files:
        print(f"❌ No regime files found for {symbol}")
        return None
    
    latest = max(regime_files, key=lambda p: p.stat().st_mtime)
    print(f"📂 Loading: {latest.name}")
    return pd.read_parquet(latest)


def check_data_structure(df):
    """Check basic data structure."""
    print("\n" + "="*80)
    print("1️⃣  DATA STRUCTURE CHECK")
    print("="*80)
    
    print(f"\n📊 Shape: {df.shape}")
    print(f"📋 Columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns[:20], 1):
        print(f"  {i}. {col}")
    if len(df.columns) > 20:
        print(f"  ... and {len(df.columns) - 20} more columns")
    
    print(f"\n🔍 Column types:")
    print(df.dtypes.value_counts())
    
    print(f"\n📅 Index info:")
    print(f"  Type: {type(df.index)}")
    if hasattr(df.index, 'dtype'):
        print(f"  Dtype: {df.index.dtype}")
    print(f"  First: {df.index[0] if len(df) > 0 else 'N/A'}")
    print(f"  Last: {df.index[-1] if len(df) > 0 else 'N/A'}")
    
    return df


def check_regime_labels(df):
    """Check regime label characteristics."""
    print("\n" + "="*80)
    print("2️⃣  REGIME LABEL CHECK")
    print("="*80)
    
    if 'regime_id' not in df.columns:
        print("❌ 'regime_id' column not found!")
        return None
    
    labels = df['regime_id'].values
    unique_labels = np.unique(labels)
    
    print(f"\n🎯 Regime Statistics:")
    print(f"  Unique regimes: {len(unique_labels)}")
    print(f"  Regime IDs: {unique_labels}")
    
    print(f"\n📊 Regime Distribution:")
    for regime_id in unique_labels:
        count = np.sum(labels == regime_id)
        pct = count / len(labels) * 100
        print(f"  Regime {regime_id}: {count:,} samples ({pct:.1f}%)")
    
    # Check for regime transitions
    transitions = np.sum(labels[1:] != labels[:-1])
    print(f"\n🔄 Regime Transitions: {transitions:,} ({transitions/len(labels)*100:.2f}% of data)")
    
    # Check avg regime duration
    regime_lengths = []
    current_regime = labels[0]
    current_length = 1
    for label in labels[1:]:
        if label == current_regime:
            current_length += 1
        else:
            regime_lengths.append(current_length)
            current_regime = label
            current_length = 1
    regime_lengths.append(current_length)
    
    print(f"\n⏱️  Regime Duration Statistics:")
    print(f"  Mean: {np.mean(regime_lengths):.1f} samples")
    print(f"  Median: {np.median(regime_lengths):.1f} samples")
    print(f"  Min: {np.min(regime_lengths)} samples")
    print(f"  Max: {np.max(regime_lengths)} samples")
    
    if np.mean(regime_lengths) > 100:
        print(f"  ⚠️  Very long regimes - might indicate lack of transitions")
    
    return labels


def extract_features(df, feature_prefix):
    """Extract features for a given prefix (nas or tas)."""
    feature_cols = [col for col in df.columns if col.startswith(f"{feature_prefix}_feature_")]
    
    if not feature_cols:
        # Try array column
        array_col = f"{feature_prefix}_features"
        if array_col in df.columns:
            features = np.array(df[array_col].apply(np.array).tolist())
            return features, None
        return None, None
    
    features = df[feature_cols].values
    return features, feature_cols


def check_features(df, feature_prefix="nas"):
    """Check feature characteristics."""
    print("\n" + "="*80)
    print(f"3️⃣  {feature_prefix.upper()} FEATURE CHECK")
    print("="*80)

    try:
        features, feature_cols = extract_features(df, feature_prefix)

        if features is None:
            print(f"❌ No {feature_prefix.upper()} features found!")
            return None

        print(f"\n📏 Feature Shape: {features.shape}")
        print(f"  Samples: {features.shape[0]:,}")
        print(f"  Features: {features.shape[1]}")

        # Check for NaNs
        nan_count = np.isnan(features).sum()
        if nan_count > 0:
            print(f"  ⚠️  NaN values: {nan_count:,} ({nan_count/features.size*100:.2f}%)")
        else:
            print(f"  ✅ No NaN values")

        # Check variance
        feature_vars = np.var(features, axis=0)
        zero_var = np.sum(feature_vars == 0)
        low_var = np.sum(feature_vars < 1e-6)

        print(f"\n📊 Feature Variance:")
        print(f"  Mean variance: {np.mean(feature_vars):.6f}")
        print(f"  Zero variance features: {zero_var}/{features.shape[1]}")
        print(f"  Low variance features (<1e-6): {low_var}/{features.shape[1]}")

        if zero_var > 0:
            print(f"  ⚠️  {zero_var} features have ZERO variance!")

        # Check feature ranges
        print(f"\n📏 Feature Ranges:")
        print(f"  Min: {np.min(features):.6f}")
        print(f"  Max: {np.max(features):.6f}")
        print(f"  Mean: {np.mean(features):.6f}")
        print(f"  Std: {np.std(features):.6f}")

        return features

    except Exception as e:
        print(f"❌ {feature_prefix.upper()} features check failed: {e}")
        return None


def check_temporal_alignment(df, features, labels):
    """Check if features and labels are temporally aligned."""
    print("\n" + "="*80)
    print("4️⃣  TEMPORAL ALIGNMENT CHECK")
    print("="*80)
    
    # Check if there's a timestamp column
    timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower() or 'time' in col.lower()]
    
    if timestamp_cols:
        print(f"\n📅 Found timestamp columns: {timestamp_cols}")
        ts_col = timestamp_cols[0]
        timestamps = df[ts_col].values
        
        # Check for monotonic increase
        is_monotonic = np.all(timestamps[1:] >= timestamps[:-1])
        print(f"  Monotonic timestamps: {'✅ Yes' if is_monotonic else '❌ No - DATA ORDER ISSUE!'}")
        
        if not is_monotonic:
            print(f"  ⚠️  Timestamps are not in order - features/labels might be misaligned!")
    else:
        print(f"\n⚠️  No timestamp column found - cannot verify temporal alignment")
    
    # Check index
    if isinstance(df.index, pd.DatetimeIndex):
        print(f"\n📅 DatetimeIndex detected:")
        print(f"  Start: {df.index[0]}")
        print(f"  End: {df.index[-1]}")
        print(f"  Frequency: {pd.infer_freq(df.index) or 'irregular'}")
        
        is_monotonic = df.index.is_monotonic_increasing
        print(f"  Monotonic: {'✅ Yes' if is_monotonic else '❌ No - DATA ORDER ISSUE!'}")
    
    # Check for data leakage: are features from same timestamp as labels?
    print(f"\n🔍 Data Leakage Check:")
    print(f"  Feature shape[0]: {features.shape[0]}")
    print(f"  Labels shape[0]: {len(labels)}")
    print(f"  DataFrame shape[0]: {len(df)}")
    
    if features.shape[0] == len(labels) == len(df):
        print(f"  ✅ Lengths match - features and labels are aligned")
        print(f"\n  ⚠️  POTENTIAL ISSUE: Features from SAME timestamp as labels!")
        print(f"     This means you're trying to predict CURRENT regime, not FUTURE regime.")
        print(f"     Features should be from time T to predict regime at T+1.")
    else:
        print(f"  ❌ LENGTH MISMATCH - alignment issue!")


def test_prediction_capability(features, labels):
    """Test if features can actually predict labels."""
    print("\n" + "="*80)
    print("5️⃣  PREDICTION CAPABILITY TEST")
    print("="*80)
    
    print(f"\n🤖 Testing RandomForest with default parameters...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"  Train: {X_train.shape[0]:,} samples")
    print(f"  Test: {X_test.shape[0]:,} samples")
    
    # Train model
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    
    # Test predictions
    train_score = rf.score(X_train, y_train)
    test_score = rf.score(X_test, y_test)
    
    print(f"\n📊 Performance:")
    print(f"  Train accuracy: {train_score:.4f}")
    print(f"  Test accuracy: {test_score:.4f}")
    print(f"  Overfit gap: {train_score - test_score:.4f}")
    
    if train_score > 0.95:
        print(f"  ⚠️  VERY HIGH train accuracy - possible overfitting or leakage!")
    
    if test_score > 0.9:
        print(f"  ⚠️  VERY HIGH test accuracy - possible data leakage!")
    
    # Cross-validation
    print(f"\n🔄 Cross-Validation (3-fold):")
    cv_scores = cross_val_score(rf, features, labels, cv=3, scoring='accuracy')
    print(f"  Fold scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  Mean CV: {np.mean(cv_scores):.4f}")
    print(f"  Std CV: {np.std(cv_scores):.6f}")
    
    if np.std(cv_scores) < 0.001:
        print(f"  🚨 IDENTICAL CV SCORES - This is your problem!")
        print(f"     Either:")
        print(f"     1. Features have no signal")
        print(f"     2. CV is broken")
        print(f"     3. Data is too homogeneous")
    
    # Feature importances
    print(f"\n🔬 Feature Importance Analysis:")
    importances = rf.feature_importances_
    print(f"  Max importance: {np.max(importances):.4f}")
    print(f"  Mean importance: {np.mean(importances):.4f}")
    print(f"  Features >5% importance: {np.sum(importances > 0.05)}/{len(importances)}")
    print(f"  Features >1% importance: {np.sum(importances > 0.01)}/{len(importances)}")
    
    if np.max(importances) < 0.05:
        print(f"  🚨 ALL features have low importance - NO SIGNAL!")
    
    # Show top features
    top_indices = np.argsort(importances)[-5:][::-1]
    print(f"\n🏆 Top 5 Features:")
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i}. Feature {idx}: {importances[idx]:.4f}")
    
    # Predictions analysis
    y_pred = rf.predict(X_test)
    unique_preds = np.unique(y_pred)
    
    print(f"\n🎯 Prediction Analysis:")
    print(f"  Unique predictions: {len(unique_preds)}/{len(np.unique(labels))}")
    
    for label in np.unique(labels):
        pred_count = np.sum(y_pred == label)
        true_count = np.sum(y_test == label)
        print(f"  Predicted class {label}: {pred_count} (true: {true_count})")
    
    if len(unique_preds) < len(np.unique(labels)):
        print(f"  ⚠️  Model is not predicting all classes!")
    
    # Confusion matrix
    print(f"\n📋 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Check if diagonal dominant (perfect leakage indicator)
    if cm.shape[0] == cm.shape[1]:
        diag_sum = np.trace(cm)
        total = np.sum(cm)
        diag_pct = diag_sum / total
        print(f"\n  Diagonal percentage: {diag_pct*100:.1f}%")
        if diag_pct > 0.95:
            print(f"  🚨 NEAR-PERFECT diagonal - STRONG DATA LEAKAGE SUSPECTED!")


def main():
    """Run all diagnostics."""
    print("\n" + "="*100)
    print(" "*30 + "REGIME DATA DIAGNOSTIC REPORT")
    print("="*100)
    
    # Load data
    df = load_latest_regime_file("ETHUSDT")
    if df is None:
        sys.exit(1)
    
    # Run checks
    df = check_data_structure(df)
    labels = check_regime_labels(df)
    
    if labels is None:
        print("\n❌ Cannot continue without regime labels")
        sys.exit(1)
    
    # NAS/TAS features removed - legacy components no longer used

    # Add specific guidance for missing features
    if not any(col.startswith('regime_feature_') for col in df.columns):
        print(f"\n🚨 CRITICAL: No features found in regime_assignments file!")
        print(f"   The clustering pipeline needs to be fixed to save features.")
        print(f"   Run: python3 src/launcher/ares_launcher.py step05 sr_clustering")
        print(f"   This will generate a new parquet file with features included.")
    
    print("\n" + "="*100)
    print(" "*35 + "DIAGNOSTIC COMPLETE")
    print("="*100)
    
    print("\n💡 KEY FINDINGS TO LOOK FOR:")
    print("  1. 🚨 Identical CV scores (std < 0.001) → No signal or broken CV")
    print("  2. 🚨 All features low importance (<0.05) → No predictive signal")
    print("  3. 🚨 Near-perfect accuracy (>95%) → Data leakage")
    print("  4. 🚨 Features from same timestamp as labels → Temporal leakage")
    print("  5. 🚨 No regime transitions → Labels meaningless")
    print("\n")


if __name__ == "__main__":
    main()

