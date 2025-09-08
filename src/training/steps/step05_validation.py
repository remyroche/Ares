from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
"""
Step05 Validation Module

This module provides comprehensive validation capabilities for Step05 labeling,
including lookahead bias detection, data integrity checks, and temporal validation.
"""

import pandas as pd

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, validates
import logging
import numpy as np
import time

logger = system_logger.getChild('Step05Validation')

@dataclass
class ValidationResult:
    """Result of a validation check."""
    passed: bool
    score: float
    warnings: List[str]
    errors: List[str]
    recommendations: List[str]
    details: Dict[str, Any]

@dataclass
class LookaheadBiasResult:
    """Result of lookahead bias validation."""
    bias_detected: bool
    bias_score: float
    temporal_violations: int
    future_data_leakage: bool
    recommendations: List[str]
    details: Dict[str, Any]

class Step05Validator:
    """Comprehensive validator for Step05 labeling operations."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        self.validation_history = []
        
    @traced(span_name='validate_lookahead_bias')
    @validates()
    @handles_errors()
    def validate_lookahead_bias(self, data: pd.DataFrame, 
                              barrier_params: Dict[str, Any]) -> LookaheadBiasResult:
        """
        Validate that no lookahead bias exists in the labeling process.
        
        Args:
            data: DataFrame with price data and labels
            barrier_params: Triple barrier parameters
            
        Returns:
            LookaheadBiasResult with bias analysis
        """
        try:
            self.logger.info("🔍 Starting lookahead bias validation...")
            
            warnings = []
            errors = []
            recommendations = []
            details = {}
            
            # Check 1: Temporal ordering
            temporal_violations = self._check_temporal_ordering(data)
            details['temporal_violations'] = temporal_violations
            
            if temporal_violations > 0:
                errors.append(f"Found {temporal_violations} temporal ordering violations")
                recommendations.append("Ensure data is properly sorted by timestamp")
            
            # Check 2: Future data leakage in barrier calculations
            future_leakage = self._check_future_data_leakage(data, barrier_params)
            details['future_data_leakage'] = future_leakage
            
            if future_leakage:
                errors.append("Future data leakage detected in barrier calculations")
                recommendations.append("Review barrier calculation logic for temporal integrity")
            
            # Check 3: Label consistency with temporal constraints
            label_consistency = self._check_label_temporal_consistency(data, barrier_params)
            details['label_consistency'] = label_consistency
            
            if label_consistency['violations'] > 0:
                warnings.append(f"Found {label_consistency['violations']} label temporal inconsistencies")
                recommendations.append("Review labeling logic for temporal constraints")
            
            # Check 4: Barrier hit timing validation
            barrier_timing = self._validate_barrier_hit_timing(data, barrier_params)
            details['barrier_timing'] = barrier_timing
            
            if barrier_timing['invalid_hits'] > 0:
                warnings.append(f"Found {barrier_timing['invalid_hits']} invalid barrier hits")
                recommendations.append("Review barrier hit detection logic")
            
            # Calculate overall bias score
            bias_score = self._calculate_bias_score(details)
            bias_detected = bias_score > 0.1 or future_leakage or temporal_violations > 0
            
            result = LookaheadBiasResult(
                bias_detected=bias_detected,
                bias_score=bias_score,
                temporal_violations=temporal_violations,
                future_data_leakage=future_leakage,
                recommendations=recommendations,
                details=details
            )
            
            self.logger.info(f"✅ Lookahead bias validation completed. Bias score: {bias_score:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Lookahead bias validation failed: {e}")
            return LookaheadBiasResult(
                bias_detected=True,  # Assume bias if validation fails
                bias_score=1.0,
                temporal_violations=0,
                future_data_leakage=True,
                recommendations=["Fix validation errors before proceeding"],
                details={'error': str(e)}
            )
    
    def _check_temporal_ordering(self, data: pd.DataFrame) -> int:
        """Check if data is properly ordered by time."""
        if 'timestamp' not in data.columns and data.index.name != 'timestamp':
            return 0  # Can't check without timestamp
        
        timestamps = data.index if data.index.name == 'timestamp' else data['timestamp']
        
        # Check for any timestamp that is earlier than the previous one
        violations = 0
        for i in range(1, len(timestamps)):
            if timestamps.iloc[i] < timestamps.iloc[i-1]:
                violations += 1
        
        return violations
    
    def _check_future_data_leakage(self, data: pd.DataFrame, 
                                 barrier_params: Dict[str, Any]) -> bool:
        """Check for future data leakage in barrier calculations."""
        try:
            max_lookahead = barrier_params.get('max_lookahead', 100)
            time_barrier_minutes = barrier_params.get('time_barrier_minutes', 30)
            
            # Check if any label uses data beyond the allowed lookahead
            if 'label' not in data.columns:
                return False
            
            # For each labeled point, verify that the label was calculated
            # using only data within the allowed lookahead window
            for i in range(len(data) - max_lookahead):
                if pd.isna(data['label'].iloc[i]):
                    continue
                
                # Check if the label could have been influenced by future data
                # This is a simplified check - in practice, you'd need to trace
                # the actual barrier calculation logic
                future_window = data.iloc[i+1:i+max_lookahead+1]
                
                # If there are any significant price movements in the future window
                # that could have influenced the label, flag as potential leakage
                if len(future_window) > 0:
                    price_changes = future_window['close'].pct_change().abs()
                    if price_changes.max() > 0.01:  # 1% threshold
                        return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error checking future data leakage: {e}")
            return True  # Assume leakage if check fails
    
    def _check_label_temporal_consistency(self, data: pd.DataFrame,
                                        barrier_params: Dict[str, Any]) -> Dict[str, Any]:
        """Check if labels are temporally consistent."""
        violations = 0
        details = {}
        
        try:
            if 'label' not in data.columns:
                return {'violations': 0, 'details': {}}
            
            # Check for impossible label sequences
            labels = data['label'].dropna()
            
            # Check for labels that appear to be based on future information
            for i in range(len(labels) - 1):
                current_label = labels.iloc[i]
                next_label = labels.iloc[i + 1]
                
                # Simple heuristic: if current label is opposite to next label
                # and there's no price movement to justify it, flag as suspicious
                if current_label != 0 and next_label != 0 and current_label != next_label:
                    # Check if there's sufficient price movement to justify the change
                    if i + 1 < len(data):
                        price_change = abs(data['close'].iloc[i+1] - data['close'].iloc[i]) / data['close'].iloc[i]
                        if price_change < 0.001:  # Less than 0.1% change
                            violations += 1
            
            details['suspicious_transitions'] = violations
            return {'violations': violations, 'details': details}
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error checking label temporal consistency: {e}")
            return {'violations': 0, 'details': {'error': str(e)}}
    
    def _validate_barrier_hit_timing(self, data: pd.DataFrame,
                                   barrier_params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that barrier hits are temporally correct."""
        invalid_hits = 0
        details = {}
        
        try:
            if 'label' not in data.columns or 'close' not in data.columns:
                return {'invalid_hits': 0, 'details': {}}
            
            profit_take = barrier_params.get('profit_take_multiplier', 0.002)
            stop_loss = barrier_params.get('stop_loss_multiplier', 0.001)
            
            # Check each labeled point
            for i in range(len(data) - 1):
                if pd.isna(data['label'].iloc[i]):
                    continue
                
                label = data['label'].iloc[i]
                entry_price = data['close'].iloc[i]
                
                if label == 1:  # Profit take hit
                    # Check if profit take was actually hit in the future
                    future_prices = data['close'].iloc[i+1:i+101]  # Next 100 periods
                    profit_target = entry_price * (1 + profit_take)
                    
                    if not (future_prices >= profit_target).any():
                        invalid_hits += 1
                
                elif label == -1:  # Stop loss hit
                    # Check if stop loss was actually hit in the future
                    future_prices = data['close'].iloc[i+1:i+101]  # Next 100 periods
                    stop_target = entry_price * (1 - stop_loss)
                    
                    if not (future_prices <= stop_target).any():
                        invalid_hits += 1
            
            details['profit_take_validation'] = profit_take
            details['stop_loss_validation'] = stop_loss
            return {'invalid_hits': invalid_hits, 'details': details}
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error validating barrier hit timing: {e}")
            return {'invalid_hits': 0, 'details': {'error': str(e)}}
    
    def _calculate_bias_score(self, details: Dict[str, Any]) -> float:
        """Calculate overall bias score from validation details."""
        score = 0.0
        
        # Temporal violations weight
        temporal_violations = details.get('temporal_violations', 0)
        score += min(temporal_violations * 0.1, 0.5)
        
        # Future data leakage weight
        if details.get('future_data_leakage', False):
            score += 0.5
        
        # Label consistency weight
        label_consistency = details.get('label_consistency', {})
        violations = label_consistency.get('violations', 0)
        score += min(violations * 0.05, 0.3)
        
        # Barrier timing weight
        barrier_timing = details.get('barrier_timing', {})
        invalid_hits = barrier_timing.get('invalid_hits', 0)
        score += min(invalid_hits * 0.02, 0.2)
        
        return min(score, 1.0)
    
    @traced(span_name='validate_data_integrity')
    @validates()
    @handles_errors()
    def validate_data_integrity(self, data: pd.DataFrame) -> ValidationResult:
        """Validate data integrity for labeling operations."""
        try:
            self.logger.info("🔍 Starting data integrity validation...")
            
            warnings = []
            errors = []
            recommendations = []
            details = {}
            
            # Check 1: Required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                errors.append(f"Missing required columns: {missing_columns}")
                recommendations.append("Ensure all OHLCV data is present")
            
            details['missing_columns'] = missing_columns
            
            # Check 2: Data completeness
            if not data.empty:
                completeness = 1 - data.isnull().sum().sum() / (len(data) * len(data.columns))
                details['completeness'] = completeness
                
                if completeness < 0.95:
                    warnings.append(f"Data completeness is {completeness:.1%}, below 95% threshold")
                    recommendations.append("Review data collection process for missing values")
            
            # Check 3: OHLC consistency
            ohlc_errors = self._check_ohlc_consistency(data)
            details['ohlc_errors'] = ohlc_errors
            
            if ohlc_errors > 0:
                warnings.append(f"Found {ohlc_errors} OHLC consistency errors")
                recommendations.append("Review price data for invalid OHLC relationships")
            
            # Check 4: Volume validation
            volume_issues = self._check_volume_consistency(data)
            details['volume_issues'] = volume_issues
            
            if volume_issues > 0:
                warnings.append(f"Found {volume_issues} volume consistency issues")
                recommendations.append("Review volume data for anomalies")
            
            # Calculate overall score
            score = self._calculate_integrity_score(details)
            passed = len(errors) == 0 and score > 0.8
            
            result = ValidationResult(
                passed=passed,
                score=score,
                warnings=warnings,
                errors=errors,
                recommendations=recommendations,
                details=details
            )
            
            self.logger.info(f"✅ Data integrity validation completed. Score: {score:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Data integrity validation failed: {e}")
            return ValidationResult(
                passed=False,
                score=0.0,
                warnings=[],
                errors=[f"Validation failed: {str(e)}"],
                recommendations=["Fix validation errors before proceeding"],
                details={'error': str(e)}
            )
    
    def _check_ohlc_consistency(self, data: pd.DataFrame) -> int:
        """Check OHLC data consistency."""
        errors = 0
        
        try:
            if not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                return 0
            
            # Check: high >= max(open, close)
            high_violations = (data['high'] < data[['open', 'close']].max(axis=1)).sum()
            errors += high_violations
            
            # Check: low <= min(open, close)
            low_violations = (data['low'] > data[['open', 'close']].min(axis=1)).sum()
            errors += low_violations
            
            # Check: high >= low
            hl_violations = (data['high'] < data['low']).sum()
            errors += hl_violations
            
            return errors
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error checking OHLC consistency: {e}")
            return 0
    
    def _check_volume_consistency(self, data: pd.DataFrame) -> int:
        """Check volume data consistency."""
        issues = 0
        
        try:
            if 'volume' not in data.columns:
                return 0
            
            # Check for negative volumes
            negative_volumes = (data['volume'] < 0).sum()
            issues += negative_volumes
            
            # Check for extremely high volumes (outliers)
            if len(data) > 0:
                volume_q99 = data['volume'].quantile(0.99)
                volume_q01 = data['volume'].quantile(0.01)
                outlier_threshold = volume_q99 * 10  # 10x the 99th percentile
                
                outliers = (data['volume'] > outlier_threshold).sum()
                issues += outliers
            
            return issues
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error checking volume consistency: {e}")
            return 0
    
    def _calculate_integrity_score(self, details: Dict[str, Any]) -> float:
        """Calculate data integrity score."""
        score = 1.0
        
        # Deduct for missing columns
        missing_columns = details.get('missing_columns', [])
        score -= len(missing_columns) * 0.2
        
        # Deduct for completeness issues
        completeness = details.get('completeness', 1.0)
        score -= (1.0 - completeness) * 0.5
        
        # Deduct for OHLC errors
        ohlc_errors = details.get('ohlc_errors', 0)
        score -= min(ohlc_errors * 0.01, 0.3)
        
        # Deduct for volume issues
        volume_issues = details.get('volume_issues', 0)
        score -= min(volume_issues * 0.005, 0.2)
        
        return max(score, 0.0)
    
    @traced(span_name='validate_label_quality')
    @validates()
    @handles_errors()
    def validate_label_quality(self, data: pd.DataFrame) -> ValidationResult:
        """Validate the quality of generated labels."""
        try:
            self.logger.info("🔍 Starting label quality validation...")
            
            warnings = []
            errors = []
            recommendations = []
            details = {}
            
            if 'label' not in data.columns:
                errors.append("No label column found in data")
                return ValidationResult(
                    passed=False,
                    score=0.0,
                    warnings=warnings,
                    errors=errors,
                    recommendations=["Generate labels before validation"],
                    details=details
                )
            
            labels = data['label'].dropna()
            
            if len(labels) == 0:
                errors.append("No valid labels found")
                return ValidationResult(
                    passed=False,
                    score=0.0,
                    warnings=warnings,
                    errors=errors,
                    recommendations=["Check label generation process"],
                    details=details
                )
            
            # Check 1: Label distribution
            label_counts = labels.value_counts()
            details['label_distribution'] = label_counts.to_dict()
            
            # Check for extreme imbalance
            if len(label_counts) > 1:
                max_count = label_counts.max()
                min_count = label_counts.min()
                imbalance_ratio = max_count / min_count
                details['imbalance_ratio'] = imbalance_ratio
                
                if imbalance_ratio > 10:
                    warnings.append(f"Severe label imbalance detected (ratio: {imbalance_ratio:.1f})")
                    recommendations.append("Consider using balanced sampling or different labeling strategy")
            
            # Check 2: Label consistency
            consistency_score = self._calculate_label_consistency(labels)
            details['consistency_score'] = consistency_score
            
            if consistency_score < 0.7:
                warnings.append(f"Low label consistency score: {consistency_score:.3f}")
                recommendations.append("Review labeling logic for consistency issues")
            
            # Check 3: Temporal label patterns
            temporal_patterns = self._analyze_temporal_patterns(labels)
            details['temporal_patterns'] = temporal_patterns
            
            if temporal_patterns['suspicious_patterns'] > 0:
                warnings.append(f"Found {temporal_patterns['suspicious_patterns']} suspicious temporal patterns")
                recommendations.append("Review labeling for temporal bias")
            
            # Calculate overall score
            score = self._calculate_label_quality_score(details)
            passed = len(errors) == 0 and score > 0.7
            
            result = ValidationResult(
                passed=passed,
                score=score,
                warnings=warnings,
                errors=errors,
                recommendations=recommendations,
                details=details
            )
            
            self.logger.info(f"✅ Label quality validation completed. Score: {score:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Label quality validation failed: {e}")
            return ValidationResult(
                passed=False,
                score=0.0,
                warnings=[],
                errors=[f"Validation failed: {str(e)}"],
                recommendations=["Fix validation errors before proceeding"],
                details={'error': str(e)}
            )
    
    def _calculate_label_consistency(self, labels: pd.Series) -> float:
        """Calculate label consistency score."""
        try:
            # Simple consistency check: look for rapid label changes
            changes = (labels != labels.shift(1)).sum()
            total_labels = len(labels)
            
            # Consistency is higher when there are fewer rapid changes
            consistency = 1.0 - (changes / total_labels)
            return max(consistency, 0.0)
            
        except Exception:
            return 0.5  # Default moderate consistency
    
    def _analyze_temporal_patterns(self, labels: pd.Series) -> Dict[str, Any]:
        """Analyze temporal patterns in labels."""
        try:
            suspicious_patterns = 0
            
            # Check for alternating patterns (buy-sell-buy-sell)
            if len(labels) > 3:
                for i in range(len(labels) - 3):
                    pattern = labels.iloc[i:i+4].values
                    if len(set(pattern)) == 2 and pattern[0] == pattern[2] and pattern[1] == pattern[3]:
                        suspicious_patterns += 1
            
            return {
                'suspicious_patterns': suspicious_patterns,
                'total_patterns_checked': max(0, len(labels) - 3)
            }
            
        except Exception:
            return {'suspicious_patterns': 0, 'total_patterns_checked': 0}
    
    def _calculate_label_quality_score(self, details: Dict[str, Any]) -> float:
        """Calculate overall label quality score."""
        score = 1.0
        
        # Deduct for imbalance
        imbalance_ratio = details.get('imbalance_ratio', 1.0)
        if imbalance_ratio > 5:
            score -= min((imbalance_ratio - 5) * 0.1, 0.4)
        
        # Deduct for consistency issues
        consistency_score = details.get('consistency_score', 1.0)
        score -= (1.0 - consistency_score) * 0.3
        
        # Deduct for temporal patterns
        temporal_patterns = details.get('temporal_patterns', {})
        suspicious_patterns = temporal_patterns.get('suspicious_patterns', 0)
        total_patterns = temporal_patterns.get('total_patterns_checked', 1)
        
        if total_patterns > 0:
            pattern_ratio = suspicious_patterns / total_patterns
            score -= pattern_ratio * 0.3
        
        return max(score, 0.0)
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        return {
            'validation_history': self.validation_history,
            'total_validations': len(self.validation_history),
            'passed_validations': len([v for v in self.validation_history if v.get('passed', False)]),
            'timestamp': datetime.now().isoformat()
        }