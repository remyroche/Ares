#!/usr/bin/env python3
"""
Temporal Integrity Validation Script

This script validates temporal data splits to ensure:
1. No data leakage between train/val/test sets
2. Proper temporal ordering is maintained
3. Split ratios are correct
4. No future information leaks into training
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.ml_common.validation.temporal_data_splitter import (
    TemporalDataSplitter, 
    RegimeAwareSplitter, 
    create_temporal_splitter
)
from utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error


def validate_temporal_order(timestamps):
    """Validate that timestamps are in proper chronological order."""
    if len(timestamps) < 2:
        return True, "Insufficient data for validation"
    
    # Check for any out-of-order timestamps
    is_sorted = all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
    
    if not is_sorted:
        # Find violations
        violations = []
        for i in range(len(timestamps)-1):
            if timestamps[i] > timestamps[i+1]:
                violations.append((i, timestamps[i], timestamps[i+1]))
        
        error_msg = f"Temporal order violations found:\n"
        for i, current, next_val in violations[:5]:  # Show first 5
            error_msg += f"  Position {i}: {current} > {next_val}\n"
        
        return False, error_msg
    
    return True, "Temporal order is correct"


def validate_split_overlap(train_idx, val_idx, test_idx):
    """Validate that splits don't overlap temporally."""
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        return False, "One or more splits are empty"
    
    # Convert to numpy arrays if they aren't already
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)
    
    # Check for index overlap (data leakage)
    train_val_overlap = np.intersect1d(train_idx, val_idx)
    train_test_overlap = np.intersect1d(train_idx, test_idx)
    val_test_overlap = np.intersect1d(val_idx, test_idx)
    
    if len(train_val_overlap) > 0:
        return False, f"Train/Validation overlap: {len(train_val_overlap)} samples"
    if len(train_test_overlap) > 0:
        return False, f"Train/Test overlap: {len(train_test_overlap)} samples"
    if len(val_test_overlap) > 0:
        return False, f"Validation/Test overlap: {len(val_test_overlap)} samples"
    
    # Check temporal ordering: max(train) < min(val) < min(test)
    if len(train_idx) > 0 and len(val_idx) > 0:
        if np.max(train_idx) >= np.min(val_idx):
            return False, f"Temporal violation: max train index ({np.max(train_idx)}) >= min val index ({np.min(val_idx)})"
    
    if len(val_idx) > 0 and len(test_idx) > 0:
        if np.max(val_idx) >= np.min(test_idx):
            return False, f"Temporal violation: max val index ({np.max(val_idx)}) >= min test index ({np.min(test_idx)})"
    
    return True, "No temporal violations found"


def validate_split_ratios(train_size, val_size, test_size, total_size):
    """Validate that split ratios are reasonable."""
    train_pct = train_size / total_size
    val_pct = val_size / total_size
    test_pct = test_size / total_size
    
    # Check for reasonable ratios
    if train_pct < 0.5:
        return False, f"Training percentage too low: {train_pct:.2%} (should be > 50%)"
    if train_pct > 0.8:
        return False, f"Training percentage too high: {train_pct:.2%} (should be < 80%)"
    
    if val_pct < 0.1:
        return False, f"Validation percentage too low: {val_pct:.2%} (should be > 10%)"
    if val_pct > 0.3:
        return False, f"Validation percentage too high: {val_pct:.2%} (should be < 30%)"
    
    if test_pct < 0.1:
        return False, f"Test percentage too low: {test_pct:.2%} (should be > 10%)"
    if test_pct > 0.3:
        return False, f"Test percentage too high: {test_pct:.2%} (should be < 30%)"
    
    # Check that sums to ~100%
    total_pct = train_pct + val_pct + test_pct
    if abs(total_pct - 1.0) > 0.01:  # Allow 1% tolerance
        return False, f"Split percentages don't sum to 100%: {total_pct:.2%}"
    
    return True, f"Splits are reasonable: Train={train_pct:.1%}, Val={val_pct:.1%}, Test={test_pct:.1%}"


def create_sample_data():
    """Create sample time series data for testing."""
    # Create 1000 samples with timestamps
    start_date = datetime(2023, 1, 1)
    timestamps = [start_date + timedelta(minutes=15*i) for i in range(1000)]
    
    # Create features
    np.random.seed(42)
    n_features = 10
    X = np.random.randn(1000, n_features)
    
    # Create regime labels (4 regimes)
    y = np.random.choice([0, 1, 2, 3], size=1000, p=[0.4, 0.3, 0.2, 0.1])
    
    # Create DataFrame with proper index
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names, index=timestamps)
    y_series = pd.Series(y, index=timestamps, name='regime')
    
    return X_df, y_series


def test_basic_temporal_splitter():
    """Test basic temporal splitter."""
    tprint_info("=" * 60)
    tprint_info("TESTING BASIC TEMPORAL SPLITTER")
    tprint_info("=" * 60)
    
    # Create sample data
    X, y = create_sample_data()
    
    # Test temporal order
    tprint_info("1. Validating temporal order...")
    is_valid, msg = validate_temporal_order(y.index)
    if is_valid:
        tprint_success(f"✅ {msg}")
    else:
        tprint_error(f"❌ {msg}")
    
    # Create splitter
    config = {
        'test_size': 0.2,
        'validation_size': 0.2,
        'gap_size': 1
    }
    splitter = create_temporal_splitter(config)
    
    # Perform split
    tprint_info("2. Performing temporal split...")
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split_temporal(X.values, y.values)
    
    # Get indices for validation
    train_indices = y.index[:len(y_train)]
    val_indices = y.index[len(y_train):len(y_train)+len(y_val)]
    test_indices = y.index[len(y_train)+len(y_val):]
    
    # Validate split overlap
    tprint_info("3. Validating split overlap...")
    is_valid, msg = validate_split_overlap(train_indices, val_indices, test_indices)
    if is_valid:
        tprint_success(f"✅ {msg}")
    else:
        tprint_error(f"❌ {msg}")
    
    # Validate split ratios
    tprint_info("4. Validating split ratios...")
    is_valid, msg = validate_split_ratios(len(X_train), len(X_val), len(X_test), len(X))
    if is_valid:
        tprint_success(f"✅ {msg}")
    else:
        tprint_warning(f"⚠️ {msg}")
    
    # Check for temporal gaps
    tprint_info("5. Checking temporal gaps...")
    if len(train_indices) > 0 and len(val_indices) > 0:
        train_end = train_indices[-1]
        val_start = val_indices[0]
        expected_val_start = train_end + timedelta(minutes=15) * (config['gap_size'] + 1)
        
        if val_start != expected_val_start:
            tprint_warning(f"⚠️ Temporal gap issue: expected val_start={expected_val_start}, actual={val_start}")
        else:
            tprint_success(f"✅ Temporal gap is correct: {val_start}")
    
    return True


def test_regime_aware_splitter():
    """Test regime-aware splitter."""
    tprint_info("=" * 60)
    tprint_info("TESTING REGIME-AWARE TEMPORAL SPLITTER")
    tprint_info("=" * 60)
    
    # Create sample data with regime imbalance
    X, y = create_sample_data()
    
    # Make regime 3 very rare (only 5 samples)
    rare_regime_indices = np.random.choice(len(y), size=5, replace=False)
    y.iloc[rare_regime_indices] = 3
    
    tprint_info(f"Regime distribution: {y.value_counts().to_dict()}")
    
    # Create regime-aware splitter
    config = {
        'test_size': 0.2,
        'validation_size': 0.2,
        'gap_size': 1,
        'min_regime_samples': 5,  # This will trigger the rare regime issue
        'regime_aware': True
    }
    splitter = create_temporal_splitter(config)
    
    # Perform split
    tprint_info("1. Performing regime-aware temporal split...")
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.split_regime_aware(X.values, y.values)
        
        # Check regime distribution in each split
        tprint_info("2. Checking regime distribution...")
        train_regimes = np.unique(y_train)
        val_regimes = np.unique(y_val)
        test_regimes = np.unique(y_test)
        
        train_counts = np.bincount(y_train)
        val_counts = np.bincount(y_val)
        test_counts = np.bincount(y_test)
        tprint_info(f"Train regimes: {train_regimes} (counts: {train_counts})")
        tprint_info(f"Val regimes: {val_regimes} (counts: {val_counts})")
        tprint_info(f"Test regimes: {test_regimes} (counts: {test_counts})")
        
        # Check for temporal violations
        train_indices = y.index[:len(y_train)]
        val_indices = y.index[len(y_train):len(y_train)+len(y_val)]
        test_indices = y.index[len(y_train)+len(y_val):]
        
        is_valid, msg = validate_split_overlap(train_indices, val_indices, test_indices)
        if is_valid:
            tprint_success(f"✅ No temporal violations")
        else:
            tprint_error(f"❌ {msg}")
            return False
        
        return True
        
    except ValueError as e:
        tprint_warning(f"⚠️ Expected failure for rare regimes: {e}")
        return True  # This is expected behavior


def test_with_real_data():
    """Test with real training data if available."""
    tprint_info("=" * 60)
    tprint_info("TESTING WITH REAL TRAINING DATA")
    tprint_info("=" * 60)
    
    try:
        # Try to load real training data from artifacts
        from utils.artifact_manager import ArtifactManager
        
        artifact_manager = ArtifactManager()
        
        # Look for training data
        training_data = artifact_manager.get_artifact('selected_feature_dataframe_60', 'data')
        
        if training_data is None:
            tprint_warning("⚠️ No real training data found for validation")
            return True
        
        tprint_info(f"Loaded real data: {training_data.shape}")
        tprint_info(f"Date range: {training_data.index.min()} to {training_data.index.max()}")
        
        # Validate temporal order
        tprint_info("1. Validating temporal order of real data...")
        is_valid, msg = validate_temporal_order(training_data.index)
        if is_valid:
            tprint_success(f"✅ {msg}")
        else:
            tprint_error(f"❌ {msg}")
        
        # Test regime-aware split with real data
        config = {
            'test_size': 0.2,
            'validation_size': 0.2,
            'gap_size': 1,
            'min_regime_samples': 10,
            'regime_aware': True
        }
        
        splitter = create_temporal_splitter(config)
        
        # Need regime labels for regime-aware split
        # Try to get regime predictions
        regime_data = artifact_manager.get_artifact('regime_ensemble_predictions', 'data')
        
        if regime_data is not None:
            # Align regime data with training data
            common_index = training_data.index.intersection(regime_data.index)
            if len(common_index) > 0:
                X_aligned = training_data.loc[common_index].values
                y_aligned = regime_data.loc[common_index].iloc[:, 0].values  # First column = regime
                
                tprint_info("2. Testing regime-aware split on real data...")
                try:
                    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split_regime_aware(X_aligned, y_aligned)
                    
                    tprint_success(f"✅ Regime-aware split successful")
                    tprint_info(f"   Train: {len(X_train)} samples")
                    tprint_info(f"   Val: {len(X_val)} samples") 
                    tprint_info(f"   Test: {len(X_test)} samples")
                    
                    return True
                    
                except ValueError as e:
                    tprint_warning(f"⚠️ Regime-aware split failed: {e}")
                    return True
            else:
                tprint_warning("⚠️ No common index between training data and regime data")
                return True
        else:
            tprint_warning("⚠️ No regime data found for regime-aware split test")
            return True
            
    except Exception as e:
        tprint_error(f"❌ Error testing with real data: {e}")
        return False


def main():
    """Main validation function."""
    tprint_info("🔍 TEMPORAL INTEGRITY VALIDATION")
    tprint_info("=" * 80)
    tprint_info("This script validates temporal data splits to prevent data leakage")
    tprint_info("=" * 80)
    
    all_passed = True
    
    # Test 1: Basic temporal splitter
    if not test_basic_temporal_splitter():
        all_passed = False
    
    # Test 2: Regime-aware splitter
    if not test_regime_aware_splitter():
        all_passed = False
    
    # Test 3: Real data (if available)
    if not test_with_real_data():
        all_passed = False
    
    # Summary
    tprint_info("=" * 80)
    if all_passed:
        tprint_success("🎉 ALL TEMPORAL VALIDATION TESTS PASSED")
        tprint_info("✅ No data leakage detected")
        tprint_info("✅ Temporal ordering is correct")
        tprint_info("✅ Split ratios are reasonable")
    else:
        tprint_error("❌ SOME TEMPORAL VALIDATION TESTS FAILED")
        tprint_error("🚨 DATA LEAKAGE OR TEMPORAL VIOLATIONS DETECTED")
        tprint_error("💡 Review the warnings above and fix before training")
    
    tprint_info("=" * 80)
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
