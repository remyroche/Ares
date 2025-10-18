"""
Window Quality Assessment Utilities

Provides utilities for validating opportunity windows in the Analyst→Tactician pipeline.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter

from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.logger import system_logger


@dataclass
class WindowQualityResult:
    """Result of window quality assessment."""
    is_valid: bool
    total_windows: int
    valid_windows: int
    invalid_windows: int
    overlap_count: int
    coverage_ratio: float
    window_statistics: Dict[str, Any]
    quality_issues: List[Dict[str, Any]]
    recommendations: List[str]


def validate_window_structure(window: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate the structure of a single window.
    
    Args:
        window: Window dictionary with keys 'start', 'end', 'anchor', 'direction'
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    required_keys = ['start', 'end', 'anchor', 'direction']
    errors = []
    
    # Check required keys
    missing_keys = [key for key in required_keys if key not in window]
    if missing_keys:
        errors.append(f"Missing required keys: {missing_keys}")
        return False, errors
    
    try:
        # Convert to timestamps for validation
        start_ts = pd.Timestamp(window['start'])
        end_ts = pd.Timestamp(window['end'])
        anchor_ts = pd.Timestamp(window['anchor'])
        
        # Check temporal ordering
        if start_ts >= end_ts:
            errors.append(f"Start timestamp ({start_ts}) >= end timestamp ({end_ts})")
        
        if not (start_ts <= anchor_ts <= end_ts):
            errors.append(f"Anchor timestamp ({anchor_ts}) not within window [{start_ts}, {end_ts}]")
        
        # Check direction
        direction = window['direction']
        if direction not in [-1, 1]:
            errors.append(f"Invalid direction: {direction} (must be -1 or 1)")
        
        # Check for minimum window size (at least 2 bars)
        time_diff = end_ts - start_ts
        if time_diff.total_seconds() < 60:  # Less than 1 minute
            errors.append(f"Window too short: {time_diff.total_seconds()} seconds")
        
    except (ValueError, TypeError) as e:
        errors.append(f"Invalid timestamp format: {e}")
    
    return len(errors) == 0, errors


def calculate_window_statistics(windows: List[Dict[str, Any]], data_index: Optional[pd.Index] = None) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for a list of windows.
    
    Args:
        windows: List of window dictionaries
        data_index: Optional data index for coverage calculation
    
    Returns:
        Dictionary with window statistics
    """
    if not windows:
        return {
            'total_windows': 0,
            'valid_windows': 0,
            'window_lengths': [],
            'directions': [],
            'coverage_ratio': 0.0,
            'overlap_count': 0,
            'anchor_distribution': {}
        }
    
    valid_windows = []
    window_lengths = []
    directions = []
    anchor_timestamps = []
    
    for window in windows:
        is_valid, _ = validate_window_structure(window)
        if is_valid:
            valid_windows.append(window)
            
            # Calculate window length
            start_ts = pd.Timestamp(window['start'])
            end_ts = pd.Timestamp(window['end'])
            length_seconds = (end_ts - start_ts).total_seconds()
            window_lengths.append(length_seconds)
            
            directions.append(window['direction'])
            anchor_timestamps.append(pd.Timestamp(window['anchor']))
    
    # Calculate overlap count
    overlap_count = 0
    if len(valid_windows) > 1:
        for i in range(len(valid_windows)):
            for j in range(i + 1, len(valid_windows)):
                win1 = valid_windows[i]
                win2 = valid_windows[j]
                
                start1 = pd.Timestamp(win1['start'])
                end1 = pd.Timestamp(win1['end'])
                start2 = pd.Timestamp(win2['start'])
                end2 = pd.Timestamp(win2['end'])
                
                # Check for overlap
                if not (end1 <= start2 or end2 <= start1):
                    overlap_count += 1
    
    # Calculate coverage ratio if data index is provided
    coverage_ratio = 0.0
    if data_index is not None and valid_windows:
        covered_indices = set()
        for window in valid_windows:
            start_ts = pd.Timestamp(window['start'])
            end_ts = pd.Timestamp(window['end'])
            
            # Find indices covered by this window
            try:
                start_idx = data_index.get_loc(start_ts)
                end_idx = data_index.get_loc(end_ts)
                
                if isinstance(start_idx, slice):
                    start_idx = start_idx.start
                if isinstance(end_idx, slice):
                    end_idx = end_idx.stop
                
                for idx in range(start_idx, min(end_idx + 1, len(data_index))):
                    covered_indices.add(idx)
            except KeyError:
                # Window timestamps not in data index
                continue
        
        coverage_ratio = len(covered_indices) / len(data_index) if len(data_index) > 0 else 0.0
    
    # Calculate statistics
    statistics = {
        'total_windows': len(windows),
        'valid_windows': len(valid_windows),
        'invalid_windows': len(windows) - len(valid_windows),
        'window_lengths': {
            'min_seconds': min(window_lengths) if window_lengths else 0,
            'max_seconds': max(window_lengths) if window_lengths else 0,
            'mean_seconds': np.mean(window_lengths) if window_lengths else 0,
            'median_seconds': np.median(window_lengths) if window_lengths else 0,
            'std_seconds': np.std(window_lengths) if window_lengths else 0,
            'count': len(window_lengths)
        },
        'directions': {
            'long_count': sum(1 for d in directions if d == 1),
            'short_count': sum(1 for d in directions if d == -1),
            'total_count': len(directions)
        },
        'coverage_ratio': coverage_ratio,
        'overlap_count': overlap_count,
        'anchor_distribution': dict(Counter(anchor_timestamps)),
        'unique_anchors': len(set(anchor_timestamps)),
        'duplicate_anchors': len(anchor_timestamps) - len(set(anchor_timestamps))
    }
    
    return statistics


def detect_window_quality_issues(windows: List[Dict[str, Any]], data_index: Optional[pd.Index] = None) -> List[Dict[str, Any]]:
    """
    Detect quality issues in window data.
    
    Args:
        windows: List of window dictionaries
        data_index: Optional data index for validation
    
    Returns:
        List of quality issues found
    """
    quality_issues = []
    
    if not windows:
        quality_issues.append({
            'type': 'no_windows',
            'severity': 'high',
            'description': 'No windows provided',
            'recommendation': 'Generate windows using profit labeling'
        })
        return quality_issues
    
    # Validate each window structure
    invalid_windows = []
    for i, window in enumerate(windows):
        is_valid, errors = validate_window_structure(window)
        if not is_valid:
            invalid_windows.append((i, errors))
    
    if invalid_windows:
        quality_issues.append({
            'type': 'invalid_window_structure',
            'severity': 'high',
            'description': f'{len(invalid_windows)} windows have invalid structure',
            'details': invalid_windows,
            'recommendation': 'Fix window structure validation errors'
        })
    
    # Check for index alignment if data_index is provided
    if data_index is not None:
        misaligned_windows = []
        for i, window in enumerate(windows):
            try:
                start_ts = pd.Timestamp(window['start'])
                end_ts = pd.Timestamp(window['end'])
                anchor_ts = pd.Timestamp(window['anchor'])
                
                # Check if timestamps exist in data index
                start_in_index = start_ts in data_index
                end_in_index = end_ts in data_index
                anchor_in_index = anchor_ts in data_index
                
                if not all([start_in_index, end_in_index, anchor_in_index]):
                    misaligned_windows.append({
                        'window_index': i,
                        'start_in_index': start_in_index,
                        'end_in_index': end_in_index,
                        'anchor_in_index': anchor_in_index
                    })
            except (ValueError, TypeError):
                # Already caught by structure validation
                continue
        
        if misaligned_windows:
            quality_issues.append({
                'type': 'index_misalignment',
                'severity': 'high',
                'description': f'{len(misaligned_windows)} windows have timestamps not in data index',
                'details': misaligned_windows,
                'recommendation': 'Ensure window timestamps exist in data index'
            })
    
    # Check for overlapping windows
    statistics = calculate_window_statistics(windows, data_index)
    if statistics['overlap_count'] > 0:
        quality_issues.append({
            'type': 'window_overlap',
            'severity': 'medium',
            'description': f'{statistics["overlap_count"]} window overlaps detected',
            'recommendation': 'Consider merging or filtering overlapping windows'
        })
    
    # Check for duplicate anchors
    if statistics['duplicate_anchors'] > 0:
        quality_issues.append({
            'type': 'duplicate_anchors',
            'severity': 'medium',
            'description': f'{statistics["duplicate_anchors"]} duplicate anchors found',
            'recommendation': 'Ensure unique anchors per window'
        })
    
    # Check for very short windows
    if statistics['window_lengths']['count'] > 0:
        min_length = statistics['window_lengths']['min_seconds']
        if min_length < 300:  # Less than 5 minutes
            quality_issues.append({
                'type': 'short_windows',
                'severity': 'low',
                'description': f'Some windows are very short (min: {min_length:.1f}s)',
                'recommendation': 'Consider minimum window length requirements'
            })
    
    # Check coverage ratio
    if statistics['coverage_ratio'] < 0.01:  # Less than 1% coverage
        quality_issues.append({
            'type': 'low_coverage',
            'severity': 'medium',
            'description': f'Low data coverage: {statistics["coverage_ratio"]:.1%}',
            'recommendation': 'Consider generating more windows for better coverage'
        })
    
    return quality_issues


def assess_windows(windows: List[Dict[str, Any]], index: Optional[pd.Index] = None) -> WindowQualityResult:
    """
    Comprehensive assessment of window quality.
    
    Args:
        windows: List of window dictionaries
        index: Optional data index for validation
    
    Returns:
        WindowQualityResult with comprehensive assessment
    """
    logger = system_logger.getChild('WindowAssessment')
    
    # Calculate statistics
    statistics = calculate_window_statistics(windows, index)
    
    # Detect quality issues
    quality_issues = detect_window_quality_issues(windows, index)
    
    # Generate recommendations
    recommendations = []
    
    if statistics['invalid_windows'] > 0:
        recommendations.append(f"Fix {statistics['invalid_windows']} invalid windows")
    
    if statistics['overlap_count'] > 0:
        recommendations.append("Resolve window overlaps by filtering or merging")
    
    if statistics['duplicate_anchors'] > 0:
        recommendations.append("Ensure unique anchors per window")
    
    if statistics['coverage_ratio'] < 0.05:  # Less than 5% coverage
        recommendations.append("Increase window generation for better data coverage")
    
    # Determine overall validity
    high_severity_issues = [issue for issue in quality_issues if issue['severity'] == 'high']
    is_valid = len(high_severity_issues) == 0
    
    result = WindowQualityResult(
        is_valid=is_valid,
        total_windows=statistics['total_windows'],
        valid_windows=statistics['valid_windows'],
        invalid_windows=statistics['invalid_windows'],
        overlap_count=statistics['overlap_count'],
        coverage_ratio=statistics['coverage_ratio'],
        window_statistics=statistics,
        quality_issues=quality_issues,
        recommendations=recommendations
    )
    
    # Log results
    if is_valid:
        tprint_success(f"✅ Window assessment passed: {statistics['valid_windows']}/{statistics['total_windows']} valid windows")
    else:
        tprint_error(f"❌ Window assessment failed: {len(high_severity_issues)} high-severity issues")
        for issue in high_severity_issues:
            tprint_error(f"   → {issue['description']}")
    
    # Log warnings for medium/low severity issues
    medium_low_issues = [issue for issue in quality_issues if issue['severity'] in ['medium', 'low']]
    for issue in medium_low_issues:
        tprint_warning(f"⚠️ {issue['description']}")
    
    return result


def validate_window_quality(
    artifacts: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Validate window quality in artifacts.
    
    Args:
        artifacts: Dictionary containing window data
        config: Configuration for validation behavior
    
    Returns:
        Dictionary with validation results
    """
    if config is None:
        config = {
            'require_min_windows': 1,
            'max_overlap_ratio': 0.1,
            'min_coverage_ratio': 0.01,
            'strict_mode': True
        }
    
    validation_results = {
        'success': True,
        'results': {},
        'config': config
    }
    
    # Look for opportunity windows in artifacts
    windows = []
    data_index = None
    
    # Check for opportunity_windows directly
    if 'opportunity_windows' in artifacts:
        windows = artifacts['opportunity_windows']
        if not isinstance(windows, list):
            windows = []
    
    # Check for windows in metadata
    if not windows and 'metadata' in artifacts:
        metadata = artifacts['metadata']
        if isinstance(metadata, dict) and 'opportunity_windows' in metadata:
            windows = metadata['opportunity_windows']
            if not isinstance(windows, list):
                windows = []
    
    # Get data index if available
    if 'data' in artifacts and isinstance(artifacts['data'], pd.DataFrame):
        data_index = artifacts['data'].index
    elif 'features' in artifacts and isinstance(artifacts['features'], pd.DataFrame):
        data_index = artifacts['features'].index
    
    # Perform assessment
    if windows:
        result = assess_windows(windows, data_index)
        validation_results['results']['windows'] = result
        
        # Check against configuration requirements
        if config.get('require_min_windows', 1) > 0 and result.valid_windows < config['require_min_windows']:
            validation_results['success'] = False
            tprint_error(f"❌ Insufficient valid windows: {result.valid_windows} < {config['require_min_windows']}")
        
        if config.get('strict_mode', True) and not result.is_valid:
            validation_results['success'] = False
        
        overlap_ratio = result.overlap_count / max(result.total_windows, 1)
        if config.get('max_overlap_ratio', 0.1) < overlap_ratio:
            validation_results['success'] = False
            tprint_error(f"❌ Too many window overlaps: {overlap_ratio:.1%} > {config['max_overlap_ratio']:.1%}")
        
        if result.coverage_ratio < config.get('min_coverage_ratio', 0.01):
            tprint_warning(f"⚠️ Low window coverage: {result.coverage_ratio:.1%} < {config['min_coverage_ratio']:.1%}")
    
    else:
        validation_results['success'] = False
        tprint_error("❌ No opportunity windows found in artifacts")
        validation_results['results']['windows'] = WindowQualityResult(
            is_valid=False,
            total_windows=0,
            valid_windows=0,
            invalid_windows=0,
            overlap_count=0,
            coverage_ratio=0.0,
            window_statistics={},
            quality_issues=[{'type': 'no_windows', 'severity': 'high', 'description': 'No windows found'}],
            recommendations=['Generate opportunity windows']
        )
    
    return validation_results
