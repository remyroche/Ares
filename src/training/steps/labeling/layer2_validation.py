"""
Validation utilities for Layer 2 geometry assessment.

Provides comprehensive validation of geometry quality metrics to detect:
1. "God Features" - Features with suspiciously perfect predictive power
2. Event density issues - Too few events for reliable statistics  
3. Temporal instability - Performance degradation over time
4. Spurious AUC via permutation testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def validate_geometry_quality(
    geometry_uuid: str,
    feature_importance: pd.DataFrame,
    event_count: int,
    oof_auc: float,
    oof_ic: float,
    time_periods: Optional[List[str]] = None,
    period_aucs: Optional[List[float]] = None
) -> Dict[str, any]:
    """
    Comprehensive validation of a geometry's quality metrics.
    
    Args:
        geometry_uuid: Unique identifier for the geometry
        feature_importance: DataFrame with columns ['feature', 'importance']
        event_count: Total number of events for this geometry
        oof_auc: Out-of-fold AUC
        oof_ic: Out-of-fold Information Coefficient
        time_periods: Optional list of time period labels (e.g., ['2023', '2024', '2025'])
        period_aucs: Optional list of AUC values per time period
        
    Returns:
        Dict with validation results and warnings
    """
    warnings = []
    critical_issues = []
    validation_passed = True
    
    # ═══════════════════════════════════════════════════════════════════
    # TEST 1: EVENT DENSITY CHECK
    # ═══════════════════════════════════════════════════════════════════
    # Insufficient events lead to overfitting and unreliable statistics
    # De Prado 2018: Minimum 100 events for basic reliability
    # For AUC > 0.80, require at least 300 events for confidence
    
    if event_count < 50:
        critical_issues.append(f"❌ CRITICAL: Only {event_count} events - EXTREMELY unreliable")
        validation_passed = False
    elif event_count < 100:
        warnings.append(f"⚠️ WARNING: Only {event_count} events - high risk of overfitting")
    elif event_count < 300 and oof_auc > 0.80:
        warnings.append(f"⚠️ SUSPICIOUS: AUC={oof_auc:.3f} with only {event_count} events")
        warnings.append(f"   High AUC on low sample size suggests potential spuriousness")
    
    event_density_ok = event_count >= 100
    
    # ═══════════════════════════════════════════════════════════════════
    # TEST 2: "GOD FEATURE" DETECTION
    # ═══════════════════════════════════════════════════════════════════
    # A "God Feature" dominates predictions with >50% importance
    # This indicates either:
    # (a) Data leakage (feature contains future information)
    # (b) Proxy for non-causal relationship (e.g., absolute price level)
    # (c) Severe feature engineering bug
    
    god_features_detected = []
    
    if feature_importance is not None and len(feature_importance) > 0:
        # Normalize importance to sum to 1
        total_importance = feature_importance['importance'].sum()
        if total_importance > 0:
            feature_importance = feature_importance.copy()
            feature_importance['importance_pct'] = (
                feature_importance['importance'] / total_importance * 100
            )
            
            # Top 5 features for inspection
            top5 = feature_importance.nlargest(5, 'importance')
            
            # Check for God Features (>50% importance)
            for idx, row in top5.iterrows():
                feat_name = row['feature']
                feat_pct = row['importance_pct']
                
                if feat_pct > 50:
                    critical_issues.append(
                        f"❌ GOD FEATURE: '{feat_name}' has {feat_pct:.1f}% importance!"
                    )
                    critical_issues.append(
                        f"   This suggests data leakage or non-causal proxy"
                    )
                    god_features_detected.append(feat_name)
                    validation_passed = False
                elif feat_pct > 30:
                    warnings.append(
                        f"⚠️ DOMINANT FEATURE: '{feat_name}' has {feat_pct:.1f}% importance"
                    )
                    warnings.append(
                        f"   Verify this feature doesn't contain future information"
                    )
                    
            # Check for suspicious feature names
            suspicious_keywords = [
                'future', 'forward', 'label', 'target', 'return_t+',
                'price_next', 'vol_next', 'shift_-'  # negative shift = lookahead
            ]
            
            for idx, row in top5.iterrows():
                feat_name = row['feature'].lower()
                for keyword in suspicious_keywords:
                    if keyword in feat_name:
                        critical_issues.append(
                            f"❌ SUSPICIOUS FEATURE NAME: '{row['feature']}' contains '{keyword}'"
                        )
                        critical_issues.append(
                            f"   This may indicate lookahead bias!")
                        validation_passed = False
                        break
            
    else:
        warnings.append("⚠️ No feature importance available for validation")
    
    # ═══════════════════════════════════════════════════════════════════
    # TEST 3: TEMPORAL STABILITY
    # ═══════════════════════════════════════════════════════════════════
    # Strong performance that degrades over time suggests:
    # (a) Regime shift (model trained on past data doesn't generalize)
    # (b) Subtle lookahead (using information from specific time period)
    # (c) Overfitting to historical patterns
    
    temporal_stable = True
    if time_periods is not None and period_aucs is not None and len(period_aucs) >= 2:
        # Check if AUC degrades significantly in most recent period
        earliest_auc = period_aucs[0]
        latest_auc = period_aucs[-1]
        
        degradation = earliest_auc - latest_auc
        
        if degradation > 0.15:  # AUC drops by >0.15
            warnings.append(
                f"⚠️ TEMPORAL DEGRADATION: AUC dropped from {earliest_auc:.3f} to {latest_auc:.3f}"
            )
            warnings.append(
                f"   Model performance deteriorating over time - may not generalize"
            )
            temporal_stable = False
        
        # Check for extreme variance across periods
        auc_std = np.std(period_aucs)
        if auc_std > 0.1:
            warnings.append(
                f"⚠️ HIGH VARIANCE: AUC std={auc_std:.3f} across time periods"
            )
            warnings.append(
                f"   Inconsistent performance suggests overfitting or regime sensitivity"
            )
    
    # ═══════════════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ═══════════════════════════════════════════════════════════════════
    
    return {
        'validation_passed': validation_passed,
        'event_density_ok': event_density_ok,
        'god_features_detected': god_features_detected,
        'temporal_stable': temporal_stable,
        'warnings': warnings,
        'critical_issues': critical_issues,
        'event_count': event_count,
        'top5_features': top5[['feature', 'importance_pct']].to_dict('records') if feature_importance is not None and len(feature_importance) > 0 else []
    }


def run_permutation_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    real_auc: float,
    n_permutations: int = 1000,
    random_seed: int = 42
) -> Dict[str, any]:
    """
    Permutation test to assess whether observed AUC is significantly better than random.
    
    Shuffles labels N times and recomputes AUC. If real AUC falls within the 95% CI
    of shuffled AUCs, it's likely spurious.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        real_auc: Observed AUC (to avoid recomputing)
        n_permutations: Number of shuffle iterations
        random_seed: Random seed for reproducibility
        
    Returns:
        Dict with permutation test results
    """
    from sklearn.metrics import roc_auc_score
    
    np.random.seed(random_seed)
    
    shuffled_aucs = []
    
    for i in range(n_permutations):
        # Shuffle labels
        y_shuffled = np.random.permutation(y_true)
        
        # Check if shuffled labels are degenerate
        if len(np.unique(y_shuffled)) < 2:
            continue
            
        try:
            auc_shuffled = roc_auc_score(y_shuffled, y_pred)
            shuffled_aucs.append(auc_shuffled)
        except:
            # Skip if AUC computation fails
            continue
    
    if len(shuffled_aucs) < 10:
        # Not enough valid permutations
        return {
            'permutation_valid': False,
            'p_value': None,
            'mean_shuffled_auc': None,
            'ci_95_low': None,
            'ci_95_high': None,
            'is_spurious': None
        }
    
    shuffled_aucs = np.array(shuffled_aucs)
    
    # Compute p-value: fraction of shuffled AUCs >= real AUC
    p_value = np.mean(shuffled_aucs >= real_auc)
    
    # 95% confidence interval of shuffled AUCs
    ci_95_low, ci_95_high = np.percentile(shuffled_aucs, [2.5, 97.5])
    
    # Is real AUC within the shuffled distribution's 95% CI?
    is_spurious = (ci_95_low <= real_auc <= ci_95_high)
    
    return {
        'permutation_valid': True,
        'p_value': float(p_value),
        'mean_shuffled_auc': float(np.mean(shuffled_aucs)),
        'std_shuffled_auc': float(np.std(shuffled_aucs)),
        'ci_95_low': float(ci_95_low),
        'ci_95_high': float(ci_95_high),
        'is_spurious': is_spurious,
        'n_valid_permutations': len(shuffled_aucs)
    }


def print_validation_report(geometry_uuid: str, validation_result: Dict):
    """
    Print a formatted validation report to console.
    """
    print(f"\n{'═'*70}")
    print(f"🔍 VALIDATION REPORT: {geometry_uuid[:40]}")
    print(f"{'═'*70}")
    
    # Overall status
    if validation_result['validation_passed']:
        print("✅ VALIDATION: PASSED")
    else:
        print("❌ VALIDATION: FAILED")
    
    print(f"\n📊 Event Count: {validation_result['event_count']}")
    
    # Top 5 features
    if validation_result['top5_features']:
        print(f"\n🏆 Top 5 Features:")
        for i, feat in enumerate(validation_result['top5_features'], 1):
            print(f"   {i}. {feat['feature'][:50]}: {feat['importance_pct']:.1f}%")
    
    # Critical issues
    if validation_result['critical_issues']:
        print(f"\n{'─'*70}")
        print("❌ CRITICAL ISSUES:")
        for issue in validation_result['critical_issues']:
            print(f"   {issue}")
        print(f"{'─'*70}")
    
    # Warnings
    if validation_result['warnings']:
        print(f"\n⚠️  WARNINGS:")
        for warning in validation_result['warnings']:
            print(f"   {warning}")
    
    # God features
    if validation_result['god_features_detected']:
        print(f"\n🚨 GOD FEATURES DETECTED: {validation_result['god_features_detected']}")
    
    print(f"{'═'*70}\n")
