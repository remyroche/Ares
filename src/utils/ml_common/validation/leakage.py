"""
Data Leakage Detection Utilities

Provides utilities for detecting future information leakage in features and labels
for the Analyst→Tactician pipeline.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import ast
import inspect
from collections import defaultdict

from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.logger import system_logger


@dataclass
class LeakageDetectionResult:
    """Result of leakage detection validation."""
    has_leakage: bool
    leakage_sources: List[Dict[str, Any]]
    feature_analysis: Dict[str, Any]
    shift_analysis: Dict[str, Any]
    correlation_analysis: Dict[str, Any]
    recommendations: List[str]
    warnings: List[str]


def detect_negative_shifts(feature_expression: str) -> List[int]:
    """
    Detect negative shift operations in feature expressions.
    
    Args:
        feature_expression: String representation of feature calculation
    
    Returns:
        List of negative shift values found
    """
    negative_shifts = []
    
    try:
        # Parse the expression as an AST
        tree = ast.parse(feature_expression, mode='eval')
        
        def visit_node(node):
            if isinstance(node, ast.Call):
                # Check for .shift() calls
                if (isinstance(node.func, ast.Attribute) and 
                    node.func.attr == 'shift' and 
                    len(node.args) == 1):
                    arg = node.args[0]
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                        if arg.value < 0:
                            negative_shifts.append(arg.value)
                    elif isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                        if isinstance(arg.operand, ast.Constant) and isinstance(arg.operand.value, int):
                            negative_shifts.append(-arg.operand.value)
            elif isinstance(node, ast.BinOp):
                # Check for subtraction operations that might indicate negative shifts
                if isinstance(node.op, ast.Sub):
                    visit_node(node.left)
                    visit_node(node.right)
            else:
                # Visit child nodes
                for child in ast.iter_child_nodes(node):
                    visit_node(child)
        
        visit_node(tree)
        
    except (SyntaxError, ValueError) as e:
        tprint_warning(f"⚠️ Could not parse feature expression: {e}")
    
    return negative_shifts


def analyze_feature_shifts(X: pd.DataFrame, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Analyze shift patterns in feature columns.
    
    Args:
        X: Feature DataFrame
        feature_names: Optional list of specific features to analyze
    
    Returns:
        Dictionary with shift analysis results
    """
    if feature_names is None:
        feature_names = X.columns.tolist()
    
    shift_analysis = {
        'total_features': len(feature_names),
        'analyzed_features': 0,
        'suspicious_features': [],
        'shift_patterns': {},
        'lag_analysis': {}
    }
    
    for feature_name in feature_names:
        if feature_name not in X.columns:
            continue
            
        feature_series = X[feature_name]
        shift_analysis['analyzed_features'] += 1
        
        # Check for common leakage patterns
        suspicious_patterns = []
        
        # Pattern 1: Perfect correlation with future values (possible leakage)
        if len(feature_series) > 1:
            # Check correlation with 1-period forward shift
            forward_corr = feature_series.corr(feature_series.shift(-1))
            if not pd.isna(forward_corr) and abs(forward_corr) > 0.95:
                suspicious_patterns.append(f"High forward correlation: {forward_corr:.3f}")
        
        # Pattern 2: Non-monotonic changes that might indicate look-ahead
        if len(feature_series) > 10:
            # Check for sudden perfect predictions
            diff = feature_series.diff()
            perfect_changes = (diff.abs() > 0) & (diff.abs() < 1e-10)
            if perfect_changes.sum() > len(feature_series) * 0.1:  # More than 10% perfect changes
                suspicious_patterns.append(f"High frequency of perfect changes: {perfect_changes.sum()}/{len(feature_series)}")
        
        # Pattern 3: Check for lag patterns
        lag_correlations = {}
        for lag in range(1, min(6, len(feature_series) // 2)):
            lag_corr = feature_series.corr(feature_series.shift(lag))
            if not pd.isna(lag_corr):
                lag_correlations[f'lag_{lag}'] = lag_corr
        
        shift_analysis['lag_analysis'][feature_name] = lag_correlations
        
        if suspicious_patterns:
            shift_analysis['suspicious_features'].append({
                'feature': feature_name,
                'patterns': suspicious_patterns
            })
    
    return shift_analysis


def rolling_holdout_test(
    X: pd.DataFrame, 
    y: pd.Series,
    feature_builder_func: callable,
    test_indices: Optional[List[int]] = None,
    holdout_size: int = 100
) -> Dict[str, Any]:
    """
    Perform rolling holdout test to detect leakage.
    
    Args:
        X: Feature DataFrame
        y: Target series
        feature_builder_func: Function that builds features
        test_indices: Specific indices to test
        holdout_size: Size of holdout window
    
    Returns:
        Dictionary with test results
    """
    if test_indices is None:
        # Select random indices for testing
        test_indices = np.random.choice(len(X), size=min(holdout_size, len(X) // 4), replace=False)
    
    test_results = {
        'tested_indices': test_indices,
        'mismatches': [],
        'total_tests': len(test_indices),
        'mismatch_rate': 0.0
    }
    
    for idx in test_indices:
        try:
            # Build features with past-only constraint
            past_only_features = feature_builder_func(X.iloc[:idx+1], strict_past_only=True)
            
            # Build features naively (potential leakage)
            naive_features = feature_builder_func(X.iloc[:idx+1], strict_past_only=False)
            
            # Compare feature values
            if isinstance(past_only_features, pd.DataFrame) and isinstance(naive_features, pd.DataFrame):
                # Compare common columns
                common_cols = set(past_only_features.columns) & set(naive_features.columns)
                
                for col in common_cols:
                    past_val = past_only_features[col].iloc[-1] if len(past_only_features) > 0 else None
                    naive_val = naive_features[col].iloc[-1] if len(naive_features) > 0 else None
                    
                    # Check for significant differences
                    if (past_val is not None and naive_val is not None and 
                        not pd.isna(past_val) and not pd.isna(naive_val)):
                        
                        if abs(past_val - naive_val) > 1e-6:  # Significant difference
                            test_results['mismatches'].append({
                                'index': idx,
                                'feature': col,
                                'past_only_value': past_val,
                                'naive_value': naive_val,
                                'difference': abs(past_val - naive_val)
                            })
        
        except Exception as e:
            tprint_warning(f"⚠️ Rolling holdout test failed at index {idx}: {e}")
    
    test_results['mismatch_rate'] = len(test_results['mismatches']) / test_results['total_tests'] if test_results['total_tests'] > 0 else 0.0
    
    return test_results


def analyze_feature_label_correlation(X: pd.DataFrame, y: pd.Series, horizon_bars: int = 1) -> Dict[str, Any]:
    """
    Analyze correlations between features and labels to detect potential leakage.
    
    Args:
        X: Feature DataFrame
        y: Target series
        horizon_bars: Expected prediction horizon
    
    Returns:
        Dictionary with correlation analysis
    """
    correlation_analysis = {
        'feature_correlations': {},
        'suspicious_correlations': [],
        'high_correlation_features': [],
        'correlation_statistics': {}
    }
    
    # Align X and y on index
    aligned_data = pd.concat([X, y.rename('target')], axis=1, join='inner')
    
    if len(aligned_data) == 0:
        return correlation_analysis
    
    # Calculate correlations for each feature
    feature_correlations = {}
    correlations = []
    
    for feature_col in X.columns:
        if feature_col in aligned_data.columns:
            corr = aligned_data[feature_col].corr(aligned_data['target'])
            if not pd.isna(corr):
                feature_correlations[feature_col] = corr
                correlations.append(corr)
                
                # Flag suspicious correlations
                if abs(corr) > 0.8:  # Very high correlation
                    correlation_analysis['suspicious_correlations'].append({
                        'feature': feature_col,
                        'correlation': corr,
                        'abs_correlation': abs(corr),
                        'suspicious_reason': 'Very high correlation (|r| > 0.8)'
                    })
                elif abs(corr) > 0.6:  # High correlation
                    correlation_analysis['high_correlation_features'].append({
                        'feature': feature_col,
                        'correlation': corr
                    })
    
    correlation_analysis['feature_correlations'] = feature_correlations
    
    if correlations:
        correlation_analysis['correlation_statistics'] = {
            'mean_abs_correlation': np.mean(np.abs(correlations)),
            'max_abs_correlation': np.max(np.abs(correlations)),
            'min_abs_correlation': np.min(np.abs(correlations)),
            'std_abs_correlation': np.std(np.abs(correlations)),
            'features_above_0.5': sum(1 for c in correlations if abs(c) > 0.5),
            'features_above_0.8': sum(1 for c in correlations if abs(c) > 0.8)
        }
    
    return correlation_analysis


def assert_past_only(
    X: pd.DataFrame, 
    y: pd.Series, 
    horizon_bars: int = 1,
    feature_builder_func: Optional[callable] = None,
    strict_mode: bool = True
) -> LeakageDetectionResult:
    """
    Assert that features are built using only past information.
    
    Args:
        X: Feature DataFrame
        y: Target series
        horizon_bars: Expected prediction horizon
        feature_builder_func: Optional function for building features
        strict_mode: If True, fail on any detected leakage
    
    Returns:
        LeakageDetectionResult with detailed analysis
    """
    logger = system_logger.getChild('LeakageDetection')
    
    warnings = []
    leakage_sources = []
    recommendations = []
    
    # Analyze feature shifts
    shift_analysis = analyze_feature_shifts(X)
    
    # Analyze feature-label correlations
    correlation_analysis = analyze_feature_label_correlation(X, y, horizon_bars)
    
    # Rolling holdout test if feature builder is provided
    holdout_results = {}
    if feature_builder_func is not None:
        holdout_results = rolling_holdout_test(X, y, feature_builder_func)
        
        if holdout_results['mismatch_rate'] > 0.1:  # More than 10% mismatch
            leakage_sources.append({
                'type': 'rolling_holdout_test',
                'severity': 'high',
                'description': f"Rolling holdout test detected {holdout_results['mismatch_rate']:.1%} mismatch rate",
                'details': holdout_results
            })
            recommendations.append("Review feature building logic for potential look-ahead bias")
    
    # Check for suspicious correlations
    if correlation_analysis['suspicious_correlations']:
        for suspicious in correlation_analysis['suspicious_correlations']:
            leakage_sources.append({
                'type': 'suspicious_correlation',
                'severity': 'high',
                'description': f"Feature '{suspicious['feature']}' has suspicious correlation: {suspicious['correlation']:.3f}",
                'details': suspicious
            })
            recommendations.append(f"Investigate feature '{suspicious['feature']}' for potential leakage")
    
    # Check for suspicious shift patterns
    if shift_analysis['suspicious_features']:
        for suspicious in shift_analysis['suspicious_features']:
            leakage_sources.append({
                'type': 'suspicious_shift_pattern',
                'severity': 'medium',
                'description': f"Feature '{suspicious['feature']}' shows suspicious patterns",
                'details': suspicious
            })
            warnings.append(f"Feature '{suspicious['feature']}' may have shift issues")
    
    # Overall feature analysis
    feature_analysis = {
        'total_features': len(X.columns),
        'na_ratios': {col: X[col].isna().sum() / len(X) for col in X.columns},
        'dtypes': {col: str(X[col].dtype) for col in X.columns},
        'constant_features': [col for col in X.columns if X[col].nunique() <= 1],
        'high_na_features': [col for col in X.columns if X[col].isna().sum() / len(X) > 0.5]
    }
    
    # Determine if leakage is detected
    has_leakage = len([s for s in leakage_sources if s['severity'] == 'high']) > 0
    
    if strict_mode and has_leakage:
        tprint_error(f"❌ Data leakage detected: {len(leakage_sources)} sources found")
        for source in leakage_sources:
            tprint_error(f"   → {source['description']}")
    elif len(leakage_sources) > 0:
        tprint_warning(f"⚠️ Potential data leakage detected: {len(leakage_sources)} sources found")
        for source in leakage_sources:
            tprint_warning(f"   → {source['description']}")
    else:
        tprint_success("✅ No data leakage detected")
    
    return LeakageDetectionResult(
        has_leakage=has_leakage,
        leakage_sources=leakage_sources,
        feature_analysis=feature_analysis,
        shift_analysis=shift_analysis,
        correlation_analysis=correlation_analysis,
        recommendations=recommendations,
        warnings=warnings
    )


def validate_leakage_prevention(
    artifacts: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Validate that artifacts are free from data leakage.
    
    Args:
        artifacts: Dictionary containing DataFrames to validate
        config: Configuration for validation behavior
    
    Returns:
        Dictionary with validation results
    """
    if config is None:
        config = {
            'strict_mode': True,
            'horizon_bars': 1,
            'feature_builder_func': None,
            'check_correlations': True,
            'check_shifts': True
        }
    
    validation_results = {
        'success': True,
        'results': {},
        'overall_leakage_detected': False,
        'config': config
    }
    
    # Validate features if present
    if 'features' in artifacts and isinstance(artifacts['features'], pd.DataFrame):
        X = artifacts['features']
        y = artifacts.get('targets', pd.Series(dtype=float))
        
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]  # Take first column
        
        if not y.empty and len(y) == len(X):
            result = assert_past_only(
                X=X,
                y=y,
                horizon_bars=config.get('horizon_bars', 1),
                feature_builder_func=config.get('feature_builder_func'),
                strict_mode=config.get('strict_mode', True)
            )
            
            validation_results['results']['features'] = result
            if result.has_leakage:
                validation_results['overall_leakage_detected'] = True
                if config.get('strict_mode', True):
                    validation_results['success'] = False
    
    # Validate individual feature columns if specified
    feature_columns = config.get('feature_columns', [])
    for col_name in feature_columns:
        if col_name in artifacts and isinstance(artifacts[col_name], pd.Series):
            feature_series = artifacts[col_name]
            # Basic checks for the feature series
            if len(feature_series) > 1:
                # Check for perfect correlation with shifted versions (potential leakage)
                forward_corr = feature_series.corr(feature_series.shift(-1))
                if not pd.isna(forward_corr) and abs(forward_corr) > 0.95:
                    validation_results['overall_leakage_detected'] = True
                    if config.get('strict_mode', True):
                        validation_results['success'] = False
                    tprint_warning(f"⚠️ Feature '{col_name}' shows high forward correlation: {forward_corr:.3f}")
    
    return validation_results
