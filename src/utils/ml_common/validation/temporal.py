"""
Temporal Alignment Validation Utilities

Provides utilities for validating temporal consistency across DataFrames and artifacts
in the Analyst→Tactician pipeline.
"""

import hashlib
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.logger import system_logger


@dataclass
class TemporalAlignmentResult:
    """Result of temporal alignment validation."""
    is_aligned: bool
    index_span_match: bool
    monotonic: bool
    no_duplicates: bool
    data_hash_match: bool
    drift_report: Dict[str, Any]
    errors: List[str]
    warnings: List[str]


def compute_data_hash(df: pd.DataFrame, exclude_columns: Optional[List[str]] = None) -> str:
    """Compute a hash of the data content for validation."""
    try:
        # Create a copy to avoid modifying original
        data_copy = df.copy()
        
        # Remove excluded columns
        if exclude_columns:
            for col in exclude_columns:
                if col in data_copy.columns:
                    data_copy = data_copy.drop(columns=[col])
        
        # Sort by index to ensure consistent hashing
        data_copy = data_copy.sort_index()
        
        # Convert to string representation for hashing
        data_str = data_copy.to_string()
        
        # Compute hash
        return hashlib.md5(data_str.encode()).hexdigest()
    except Exception as e:
        tprint_warning(f"⚠️ Failed to compute data hash: {e}")
        return "hash_computation_failed"


def check_index_properties(index: pd.Index) -> Dict[str, Any]:
    """Check properties of a pandas Index."""
    result = {
        'is_monotonic_increasing': index.is_monotonic_increasing,
        'is_monotonic_decreasing': index.is_monotonic_decreasing,
        'has_duplicates': index.has_duplicates,
        'duplicate_count': len(index) - len(index.drop_duplicates()) if index.has_duplicates else 0,
        'min_timestamp': index.min() if hasattr(index, 'min') else None,
        'max_timestamp': index.max() if hasattr(index, 'max') else None,
        'length': len(index),
        'dtype': str(index.dtype),
        'name': index.name
    }
    
    # Check for gaps if it's a DatetimeIndex
    if isinstance(index, pd.DatetimeIndex):
        if len(index) > 1:
            diffs = index.to_series().diff().dropna()
            result['median_frequency'] = diffs.median()
            result['frequency_std'] = diffs.std()
            result['frequency_consistency'] = (diffs.std() / diffs.mean()) < 0.1 if diffs.mean() > pd.Timedelta(0) else False
    
    return result


def compute_index_drift(index1: pd.Index, index2: pd.Index, name1: str = "index1", name2: str = "index2") -> Dict[str, Any]:
    """Compute drift between two indices."""
    drift_report = {
        'length_diff': len(index1) - len(index2),
        'length_ratio': len(index1) / len(index2) if len(index2) > 0 else float('inf'),
        'overlap_count': len(index1.intersection(index2)),
        'overlap_ratio': len(index1.intersection(index2)) / max(len(index1), len(index2)) if max(len(index1), len(index2)) > 0 else 0,
        'only_in_first': len(index1.difference(index2)),
        'only_in_second': len(index2.difference(index1)),
    }
    
    # Check timestamp alignment if both are DatetimeIndex
    if isinstance(index1, pd.DatetimeIndex) and isinstance(index2, pd.DatetimeIndex):
        if len(index1) > 0 and len(index2) > 0:
            drift_report['timestamp_offset'] = {
                'start_diff': (index1.min() - index2.min()).total_seconds(),
                'end_diff': (index1.max() - index2.max()).total_seconds(),
                'start_aligned': abs((index1.min() - index2.min()).total_seconds()) < 60,  # Within 1 minute
                'end_aligned': abs((index1.max() - index2.max()).total_seconds()) < 60,
            }
    
    return drift_report


def assert_aligned(
    df_list: List[pd.DataFrame], 
    names: Optional[List[str]] = None,
    require_exact_match: bool = True,
    tolerance_seconds: int = 60,
    check_data_hash: bool = False
) -> TemporalAlignmentResult:
    """
    Assert that a list of DataFrames are temporally aligned.
    
    Args:
        df_list: List of DataFrames to validate
        names: Optional names for each DataFrame for error reporting
        require_exact_match: If True, require exact index matching
        tolerance_seconds: Tolerance for timestamp alignment in seconds
        check_data_hash: If True, also validate data content hashes
    
    Returns:
        TemporalAlignmentResult with validation details
    """
    logger = system_logger.getChild('TemporalValidation')
    
    if not df_list:
        return TemporalAlignmentResult(
            is_aligned=False,
            index_span_match=False,
            monotonic=False,
            no_duplicates=False,
            data_hash_match=False,
            drift_report={},
            errors=["No DataFrames provided for validation"],
            warnings=[]
        )
    
    if names is None:
        names = [f"df_{i}" for i in range(len(df_list))]
    
    if len(df_list) != len(names):
        return TemporalAlignmentResult(
            is_aligned=False,
            index_span_match=False,
            monotonic=False,
            no_duplicates=False,
            data_hash_match=False,
            drift_report={},
            errors=[f"Mismatch between DataFrame count ({len(df_list)}) and names count ({len(names)})"],
            warnings=[]
        )
    
    errors = []
    warnings = []
    drift_report = {}
    
    # Check individual DataFrame properties
    index_properties = {}
    data_hashes = {}
    
    for i, (df, name) in enumerate(zip(df_list, names)):
        if df is None or df.empty:
            errors.append(f"DataFrame '{name}' is None or empty")
            continue
            
        # Check index properties
        index_props = check_index_properties(df.index)
        index_properties[name] = index_props
        
        # Check monotonicity
        if not index_props['is_monotonic_increasing']:
            errors.append(f"Index of '{name}' is not monotonic increasing")
        
        # Check duplicates
        if index_props['has_duplicates']:
            errors.append(f"Index of '{name}' has {index_props['duplicate_count']} duplicates")
        
        # Compute data hash if requested
        if check_data_hash:
            data_hashes[name] = compute_data_hash(df)
    
    # Check alignment between DataFrames
    if len(df_list) > 1:
        reference_name = names[0]
        reference_df = df_list[0]
        
        for i, (df, name) in enumerate(zip(df_list[1:], names[1:]), 1):
            if df is None or df.empty:
                continue
                
            # Compute drift
            drift = compute_index_drift(reference_df.index, df.index, reference_name, name)
            drift_report[f"{reference_name}_vs_{name}"] = drift
            
            # Check index span alignment
            if require_exact_match:
                if not df.index.equals(reference_df.index):
                    errors.append(f"Index of '{name}' does not exactly match reference '{reference_name}'")
            else:
                # Check tolerance-based alignment
                if drift['overlap_ratio'] < 0.95:  # Less than 95% overlap
                    errors.append(f"Index of '{name}' has insufficient overlap with '{reference_name}': {drift['overlap_ratio']:.2%}")
                
                # Check timestamp alignment
                if 'timestamp_offset' in drift:
                    ts_offset = drift['timestamp_offset']
                    if not ts_offset['start_aligned'] or not ts_offset['end_aligned']:
                        warnings.append(f"Timestamp alignment issues between '{name}' and '{reference_name}': start_aligned={ts_offset['start_aligned']}, end_aligned={ts_offset['end_aligned']}")
    
    # Check data hash consistency if requested
    data_hash_match = True
    if check_data_hash and len(data_hashes) > 1:
        reference_hash = data_hashes[names[0]]
        for name, hash_val in data_hashes.items():
            if hash_val != reference_hash:
                data_hash_match = False
                errors.append(f"Data hash mismatch: '{name}' hash differs from reference '{names[0]}'")
                break
    
    # Determine overall alignment
    is_aligned = len(errors) == 0
    index_span_match = len([e for e in errors if 'overlap' in e.lower() or 'exactly match' in e.lower()]) == 0
    monotonic = len([e for e in errors if 'monotonic' in e.lower()]) == 0
    no_duplicates = len([e for e in errors if 'duplicate' in e.lower()]) == 0
    
    result = TemporalAlignmentResult(
        is_aligned=is_aligned,
        index_span_match=index_span_match,
        monotonic=monotonic,
        no_duplicates=no_duplicates,
        data_hash_match=data_hash_match,
        drift_report=drift_report,
        errors=errors,
        warnings=warnings
    )
    
    if is_aligned:
        tprint_success(f"✅ Temporal alignment validation passed for {len(df_list)} DataFrames")
    else:
        tprint_error(f"❌ Temporal alignment validation failed: {len(errors)} errors, {len(warnings)} warnings")
        for error in errors:
            tprint_error(f"   → {error}")
        for warning in warnings:
            tprint_warning(f"   → {warning}")
    
    return result


def validate_temporal_consistency(
    artifacts: Dict[str, Any],
    artifact_names: List[str],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Validate temporal consistency across a collection of artifacts.
    
    Args:
        artifacts: Dictionary containing DataFrames to validate
        artifact_names: Names of artifacts to validate
        config: Configuration for validation behavior
    
    Returns:
        Dictionary with validation results
    """
    if config is None:
        config = {
            'require_exact_match': True,
            'tolerance_seconds': 60,
            'check_data_hash': False,
            'fail_fast': False
        }
    
    # Extract DataFrames
    df_list = []
    names = []
    
    for name in artifact_names:
        if name in artifacts:
            artifact = artifacts[name]
            if isinstance(artifact, pd.DataFrame):
                df_list.append(artifact)
                names.append(name)
            elif isinstance(artifact, dict) and 'data' in artifact:
                if isinstance(artifact['data'], pd.DataFrame):
                    df_list.append(artifact['data'])
                    names.append(name)
        else:
            tprint_warning(f"⚠️ Artifact '{name}' not found in artifacts dictionary")
    
    if not df_list:
        return {
            'success': False,
            'error': 'No valid DataFrames found in artifacts',
            'validated_artifacts': [],
            'result': None
        }
    
    # Perform validation
    result = assert_aligned(
        df_list=df_list,
        names=names,
        require_exact_match=config.get('require_exact_match', True),
        tolerance_seconds=config.get('tolerance_seconds', 60),
        check_data_hash=config.get('check_data_hash', False)
    )
    
    return {
        'success': result.is_aligned,
        'result': result,
        'validated_artifacts': names,
        'config': config
    }
