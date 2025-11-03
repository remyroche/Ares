"""
Quality Score Data Validator

Comprehensive validation for quality score training data.
Ensures data quality before model training.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import json


class QualityDataValidator:
    """Validates quality score training data for production readiness."""
    
    def __init__(self, strict_mode: bool = True):
        """Initialize validator.
        
        Args:
            strict_mode: If True, raises exceptions on critical issues.
                        If False, only logs warnings.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.strict_mode = strict_mode
        
        # Define validation thresholds
        self.thresholds = {
            'min_samples': 100,
            'max_bounce_saturation': 0.30,  # Max 30% at max value
            'min_bounce_variance': 0.15,
            'max_nan_ratio': 0.01,  # Max 1% NaN values
            'max_inf_ratio': 0.0,   # No infinite values
            'min_correlation': 0.20,  # Min top feature correlation
            'min_strong_features': 3,  # Min features with >0.3 correlation
            'quality_range': (0.0, 1.0),
            'min_quality_variance': 0.20,
        }
    
    def validate_training_data(self, training_df: pd.DataFrame, 
                              timeframe: str = None) -> Dict[str, Any]:
        """Comprehensive validation of training data.
        
        Args:
            training_df: Training data DataFrame
            timeframe: Optional timeframe for context
            
        Returns:
            Validation report with issues, warnings, and statistics
            
        Raises:
            ValueError: If critical issues found and strict_mode=True
        """
        self.logger.info(f"\n{'='*80}")
        self.logger.info("🔍 VALIDATING QUALITY SCORE TRAINING DATA")
        self.logger.info(f"{'='*80}")
        if timeframe:
            self.logger.info(f"   Timeframe: {timeframe}")
        self.logger.info(f"   Samples: {len(training_df):,}")
        self.logger.info(f"   Strict mode: {self.strict_mode}")
        
        report = {
            'critical_issues': [],
            'warnings': [],
            'statistics': {},
            'validation_passed': True,
            'timeframe': timeframe
        }
        
        # Run all validation checks
        self._check_data_structure(training_df, report)
        self._check_required_columns(training_df, report)
        self._check_missing_values(training_df, report)
        self._check_value_ranges(training_df, report)
        self._check_distributions(training_df, report)
        self._check_correlations(training_df, report)
        self._check_sample_quality(training_df, report)
        
        # Determine if validation passed
        if report['critical_issues']:
            report['validation_passed'] = False
            self.logger.error(f"\n❌ VALIDATION FAILED: {len(report['critical_issues'])} critical issues")
            for issue in report['critical_issues']:
                self.logger.error(f"   • {issue}")
        else:
            self.logger.info(f"\n✅ VALIDATION PASSED")
        
        if report['warnings']:
            self.logger.warning(f"\n⚠️  {len(report['warnings'])} warnings:")
            for warning in report['warnings']:
                self.logger.warning(f"   • {warning}")
        
        # Print statistics
        self._print_statistics(report['statistics'])
        
        # Raise exception if critical issues and strict mode
        if self.strict_mode and not report['validation_passed']:
            raise ValueError(f"Data validation failed with {len(report['critical_issues'])} critical issues")
        
        return report
    
    def _check_data_structure(self, df: pd.DataFrame, report: Dict):
        """Check basic data structure."""
        try:
            # Check if DataFrame
            if not isinstance(df, pd.DataFrame):
                report['critical_issues'].append(f"Data is not a DataFrame: {type(df)}")
                return
            
            # Check if empty
            if len(df) == 0:
                report['critical_issues'].append("DataFrame is empty")
                return
            
            # Check minimum samples
            if len(df) < self.thresholds['min_samples']:
                report['critical_issues'].append(
                    f"Insufficient samples: {len(df)} < {self.thresholds['min_samples']}"
                )
            
            report['statistics']['total_samples'] = len(df)
            report['statistics']['total_columns'] = len(df.columns)
            
        except Exception as e:
            report['critical_issues'].append(f"Error checking data structure: {e}")
    
    def _check_required_columns(self, df: pd.DataFrame, report: Dict):
        """Check for required columns."""
        required_core = [
            'quality_score',
            'bounce_strength',
            'hold_strength',
            'trade_profit'
        ]
        
        required_enhanced = [
            'rejection_speed',
            'volume_quality'
        ]
        
        required_multi_outcome = [
            'bounce_quality',
            'hold_quality',
            'trade_quality',
            'speed_quality',
            'volume_confirmation_quality'
        ]
        
        # Check core columns
        missing_core = [c for c in required_core if c not in df.columns]
        if missing_core:
            report['critical_issues'].append(f"Missing core columns: {missing_core}")
        
        # Check enhanced columns (warnings only)
        missing_enhanced = [c for c in required_enhanced if c not in df.columns]
        if missing_enhanced:
            report['warnings'].append(f"Missing enhanced columns: {missing_enhanced}")
        
        # Check multi-outcome columns (warnings only)
        missing_multi = [c for c in required_multi_outcome if c not in df.columns]
        if missing_multi:
            report['warnings'].append(f"Missing multi-outcome columns: {missing_multi}")
        
        # Count feature columns
        feature_cols = [c for c in df.columns if c.startswith('feature_')]
        report['statistics']['feature_count'] = len(feature_cols)
        
        if len(feature_cols) < 50:
            report['warnings'].append(f"Low feature count: {len(feature_cols)} < 50")
    
    def _check_missing_values(self, df: pd.DataFrame, report: Dict):
        """Check for NaN and Inf values."""
        important_cols = [
            'quality_score', 'bounce_strength', 'hold_strength', 'trade_profit',
            'rejection_speed', 'volume_quality'
        ]
        
        for col in important_cols:
            if col not in df.columns:
                continue
            
            # Check NaN
            nan_count = df[col].isna().sum()
            nan_ratio = nan_count / len(df)
            
            if nan_ratio > self.thresholds['max_nan_ratio']:
                report['critical_issues'].append(
                    f"{col}: {nan_ratio*100:.2f}% NaN values (>{self.thresholds['max_nan_ratio']*100}%)"
                )
            elif nan_count > 0:
                report['warnings'].append(f"{col}: {nan_count} NaN values ({nan_ratio*100:.2f}%)")
            
            # Check Inf
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                report['critical_issues'].append(f"{col}: {inf_count} Inf values (not allowed)")
            
            report['statistics'][f'{col}_nan_count'] = int(nan_count)
            report['statistics'][f'{col}_inf_count'] = int(inf_count)
    
    def _check_value_ranges(self, df: pd.DataFrame, report: Dict):
        """Check that values are in expected ranges."""
        # Columns that should be [0, 1]
        bounded_cols = [
            'quality_score', 'bounce_strength', 'hold_strength',
            'rejection_speed', 'volume_quality',
            'bounce_quality', 'hold_quality', 'trade_quality', 'speed_quality'
        ]
        
        for col in bounded_cols:
            if col not in df.columns:
                continue
            
            col_min = df[col].min()
            col_max = df[col].max()
            
            # Check lower bound
            if col_min < 0:
                outliers = (df[col] < 0).sum()
                report['critical_issues'].append(
                    f"{col}: {outliers} negative values (min={col_min:.4f})"
                )
            
            # Check upper bound
            if col_max > 1.0:
                outliers = (df[col] > 1.0).sum()
                report['critical_issues'].append(
                    f"{col}: {outliers} values > 1.0 (max={col_max:.4f})"
                )
            
            report['statistics'][f'{col}_range'] = (float(col_min), float(col_max))
        
        # Trade profit can be negative but should be reasonable
        if 'trade_profit' in df.columns:
            tp_min = df['trade_profit'].min()
            tp_max = df['trade_profit'].max()
            
            if tp_min < -1.0:
                report['warnings'].append(f"trade_profit has extreme negative values: {tp_min:.2f}")
            if tp_max > 1.0:
                report['warnings'].append(f"trade_profit has extreme positive values: {tp_max:.2f}")
            
            report['statistics']['trade_profit_mean'] = float(df['trade_profit'].mean())
    
    def _check_distributions(self, df: pd.DataFrame, report: Dict):
        """Check for suspicious distributions."""
        
        # Check bounce strength saturation
        if 'bounce_strength' in df.columns:
            at_max = (df['bounce_strength'] >= 0.95).sum() / len(df)
            report['statistics']['bounce_saturation'] = float(at_max)
            
            if at_max > self.thresholds['max_bounce_saturation']:
                report['critical_issues'].append(
                    f"Bounce strength saturated: {at_max*100:.1f}% at max (>{self.thresholds['max_bounce_saturation']*100}%)"
                )
            
            bounce_std = df['bounce_strength'].std()
            report['statistics']['bounce_std'] = float(bounce_std)
            
            if bounce_std < self.thresholds['min_bounce_variance']:
                report['warnings'].append(
                    f"Low bounce variance: std={bounce_std:.3f} < {self.thresholds['min_bounce_variance']}"
                )
        
        # Check quality score distribution
        if 'quality_score' in df.columns:
            quality_std = df['quality_score'].std()
            report['statistics']['quality_std'] = float(quality_std)
            
            if quality_std < self.thresholds['min_quality_variance']:
                report['warnings'].append(
                    f"Low quality variance: std={quality_std:.3f} < {self.thresholds['min_quality_variance']}"
                )
            
            # Check for binary-like distribution
            at_extremes = ((df['quality_score'] <= 0.1).sum() + 
                          (df['quality_score'] >= 0.9).sum()) / len(df)
            report['statistics']['quality_at_extremes'] = float(at_extremes)
            
            if at_extremes > 0.5:
                report['warnings'].append(
                    f"Quality score distribution too binary: {at_extremes*100:.1f}% at extremes"
                )
            
            # Check for suspicious concentrations
            for val in [0.0, 0.2, 0.5, 1.0]:
                count = (np.abs(df['quality_score'] - val) < 0.01).sum()
                ratio = count / len(df)
                if ratio > 0.20:  # More than 20% at one value
                    report['warnings'].append(
                        f"Quality score concentrated at {val}: {ratio*100:.1f}%"
                    )
    
    def _check_correlations(self, df: pd.DataFrame, report: Dict):
        """Check feature correlations with quality score."""
        if 'quality_score' not in df.columns:
            return
        
        feature_cols = [c for c in df.columns if c.startswith('feature_')]
        
        if len(feature_cols) == 0:
            report['warnings'].append("No feature columns found")
            return
        
        try:
            # Calculate correlations
            correlations = df[feature_cols].corrwith(df['quality_score']).abs()
            correlations = correlations.dropna().sort_values(ascending=False)
            
            if len(correlations) == 0:
                report['critical_issues'].append("Could not calculate feature correlations")
                return
            
            top_corr = correlations.iloc[0]
            strong_features = (correlations > 0.3).sum()
            
            report['statistics']['top_correlation'] = float(top_corr)
            report['statistics']['top_feature'] = correlations.index[0]
            report['statistics']['strong_features_count'] = int(strong_features)
            
            # Validate correlation strength
            if top_corr < self.thresholds['min_correlation']:
                report['warnings'].append(
                    f"Weak top correlation: {top_corr:.3f} < {self.thresholds['min_correlation']}"
                )
            
            if strong_features < self.thresholds['min_strong_features']:
                report['warnings'].append(
                    f"Few strong features: {strong_features} < {self.thresholds['min_strong_features']}"
                )
            
            # Store top 10 correlations
            report['statistics']['top_10_correlations'] = {
                feat.replace('feature_', ''): float(corr) 
                for feat, corr in correlations.head(10).items()
            }
            
        except Exception as e:
            report['warnings'].append(f"Error calculating correlations: {e}")
    
    def _check_sample_quality(self, df: pd.DataFrame, report: Dict):
        """Check overall sample quality."""
        
        # Check for duplicate samples
        if 'date' in df.columns and 'symbol' in df.columns:
            duplicates = df.duplicated(subset=['date', 'symbol']).sum()
            if duplicates > 0:
                report['warnings'].append(f"Found {duplicates} duplicate samples")
                report['statistics']['duplicate_count'] = int(duplicates)
        
        # Check date range
        if 'date' in df.columns:
            try:
                date_range_days = (df['date'].max() - df['date'].min()).days
                report['statistics']['date_range_days'] = int(date_range_days)
                
                if date_range_days < 30:
                    report['warnings'].append(
                        f"Short date range: {date_range_days} days (recommend 90+)"
                    )
            except Exception as e:
                report['warnings'].append(f"Error checking date range: {e}")
        
        # Check for data imbalance
        if 'symbol' in df.columns:
            symbol_counts = df['symbol'].value_counts()
            report['statistics']['symbols'] = symbol_counts.to_dict()
            
            if len(symbol_counts) > 1:
                min_count = symbol_counts.min()
                max_count = symbol_counts.max()
                imbalance_ratio = max_count / min_count
                
                if imbalance_ratio > 5:
                    report['warnings'].append(
                        f"Symbol imbalance: {imbalance_ratio:.1f}x difference"
                    )
    
    def _print_statistics(self, stats: Dict):
        """Print validation statistics."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info("📊 VALIDATION STATISTICS")
        self.logger.info(f"{'='*80}")
        
        # Core stats
        if 'total_samples' in stats:
            self.logger.info(f"   Total samples: {stats['total_samples']:,}")
        if 'feature_count' in stats:
            self.logger.info(f"   Features: {stats['feature_count']}")
        
        # Quality stats
        if 'quality_std' in stats:
            self.logger.info(f"\n   Quality Score:")
            self.logger.info(f"      Std: {stats['quality_std']:.4f}")
            if 'quality_at_extremes' in stats:
                self.logger.info(f"      At extremes: {stats['quality_at_extremes']*100:.1f}%")
        
        # Bounce stats
        if 'bounce_saturation' in stats:
            self.logger.info(f"\n   Bounce Strength:")
            self.logger.info(f"      Saturation: {stats['bounce_saturation']*100:.1f}%")
            if 'bounce_std' in stats:
                self.logger.info(f"      Std: {stats['bounce_std']:.4f}")
        
        # Trade profit
        if 'trade_profit_mean' in stats:
            self.logger.info(f"\n   Trade Profit:")
            self.logger.info(f"      Mean: {stats['trade_profit_mean']:.4f}")
        
        # Correlations
        if 'top_correlation' in stats:
            self.logger.info(f"\n   Feature Correlations:")
            self.logger.info(f"      Top: {stats['top_correlation']:.4f} ({stats.get('top_feature', 'unknown')})")
            self.logger.info(f"      Strong (>0.3): {stats.get('strong_features_count', 0)}")
    
    def save_validation_report(self, report: Dict, output_path: str):
        """Save validation report to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSON-serializable format
        serializable_report = self._make_serializable(report)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        
        self.logger.info(f"\n✅ Validation report saved to: {output_file}")
    
    def _make_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def generate_quality_report(self, training_df: pd.DataFrame, 
                               output_path: str = None) -> str:
        """Generate comprehensive quality report.
        
        Args:
            training_df: Training data
            output_path: Optional path to save report
            
        Returns:
            Report as string
        """
        lines = []
        lines.append("="*80)
        lines.append("QUALITY SCORE DATA QUALITY REPORT")
        lines.append("="*80)
        lines.append("")
        
        # Basic info
        lines.append(f"Generated: {pd.Timestamp.now()}")
        lines.append(f"Samples: {len(training_df):,}")
        lines.append(f"Columns: {len(training_df.columns)}")
        lines.append("")
        
        # Component statistics
        components = ['bounce_strength', 'hold_strength', 'trade_profit', 
                     'rejection_speed', 'volume_quality']
        
        lines.append("COMPONENT STATISTICS:")
        lines.append("-"*80)
        
        for comp in components:
            if comp in training_df.columns:
                lines.append(f"\n{comp}:")
                lines.append(f"  Mean:   {training_df[comp].mean():.4f}")
                lines.append(f"  Median: {training_df[comp].median():.4f}")
                lines.append(f"  Std:    {training_df[comp].std():.4f}")
                lines.append(f"  Min:    {training_df[comp].min():.4f}")
                lines.append(f"  Max:    {training_df[comp].max():.4f}")
                lines.append(f"  NaN:    {training_df[comp].isna().sum()}")
        
        # Quality distribution
        if 'quality_score' in training_df.columns:
            lines.append("\n" + "-"*80)
            lines.append("QUALITY SCORE DISTRIBUTION:")
            lines.append("-"*80)
            q = training_df['quality_score']
            lines.append(f"  Mean:   {q.mean():.4f}")
            lines.append(f"  Median: {q.median():.4f}")
            lines.append(f"  Std:    {q.std():.4f}")
            lines.append(f"  IQR:    {q.quantile(0.75) - q.quantile(0.25):.4f}")
            lines.append(f"  Range:  [{q.min():.4f}, {q.max():.4f}]")
        
        # Feature correlations
        feature_cols = [c for c in training_df.columns if c.startswith('feature_')]
        if len(feature_cols) > 0 and 'quality_score' in training_df.columns:
            lines.append("\n" + "-"*80)
            lines.append("TOP FEATURE CORRELATIONS:")
            lines.append("-"*80)
            
            try:
                corr = training_df[feature_cols].corrwith(training_df['quality_score']).abs()
                corr = corr.dropna().sort_values(ascending=False).head(10)
                
                for i, (feat, val) in enumerate(corr.items(), 1):
                    lines.append(f"  {i:2d}. {feat.replace('feature_', ''):<40} {val:.4f}")
            except:
                lines.append("  Error calculating correlations")
        
        report_text = "\n".join(lines)
        
        # Save if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"\n✅ Quality report saved to: {output_path}")
        
        return report_text
    
    def validate_before_training(self, training_df: pd.DataFrame, 
                                 target_column: str = 'quality_score') -> bool:
        """Quick validation before model training.
        
        Args:
            training_df: Training data
            target_column: Target column to validate
            
        Returns:
            True if validation passed, False otherwise
        """
        self.logger.info(f"\n🔍 Quick validation before training...")
        
        # Critical checks only
        issues = []
        
        # Check data exists
        if training_df is None or len(training_df) == 0:
            issues.append("Empty training data")
        
        # Check target exists
        if target_column not in training_df.columns:
            issues.append(f"Target column '{target_column}' not found")
        
        # Check features exist
        feature_cols = [c for c in training_df.columns if c.startswith('feature_')]
        if len(feature_cols) == 0:
            issues.append("No feature columns found")
        
        # Check for NaN in target
        if target_column in training_df.columns:
            nan_in_target = training_df[target_column].isna().sum()
            if nan_in_target > 0:
                issues.append(f"{nan_in_target} NaN values in target column")
        
        # Check sample size
        if len(training_df) < self.thresholds['min_samples']:
            issues.append(f"Only {len(training_df)} samples (need {self.thresholds['min_samples']}+)")
        
        if issues:
            self.logger.error(f"❌ Validation failed:")
            for issue in issues:
                self.logger.error(f"   • {issue}")
            return False
        
        self.logger.info(f"✅ Quick validation passed")
        self.logger.info(f"   Samples: {len(training_df):,}")
        self.logger.info(f"   Features: {len(feature_cols)}")
        self.logger.info(f"   Target: {target_column}")
        
        return True


class DataQualityMonitor:
    """Monitor data quality over time and detect drift."""
    
    def __init__(self, baseline_data: Optional[pd.DataFrame] = None):
        """Initialize monitor.
        
        Args:
            baseline_data: Reference dataset for drift detection
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.baseline_data = baseline_data
        self.metrics_history = []
    
    def track_collection_metrics(self, training_df: pd.DataFrame, 
                                 duration: float, 
                                 timeframe: str) -> Dict[str, Any]:
        """Track metrics from a data collection run.
        
        Args:
            training_df: Collected training data
            duration: Collection duration in seconds
            timeframe: Timeframe used
            
        Returns:
            Metrics dictionary
        """
        from datetime import datetime
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'timeframe': timeframe,
            'samples_collected': len(training_df),
            'duration_seconds': duration,
            'samples_per_second': len(training_df) / duration if duration > 0 else 0,
        }
        
        # Quality metrics
        if 'bounce_strength' in training_df.columns:
            metrics['bounce_mean'] = float(training_df['bounce_strength'].mean())
            metrics['bounce_saturation'] = float((training_df['bounce_strength'] >= 0.95).sum() / len(training_df))
        
        if 'rejection_speed' in training_df.columns:
            metrics['rejection_speed_mean'] = float(training_df['rejection_speed'].mean())
        
        if 'volume_quality' in training_df.columns:
            metrics['volume_quality_mean'] = float(training_df['volume_quality'].mean())
        
        if 'quality_score' in training_df.columns:
            metrics['quality_mean'] = float(training_df['quality_score'].mean())
            metrics['quality_std'] = float(training_df['quality_score'].std())
        
        # Feature correlation
        feature_cols = [c for c in training_df.columns if c.startswith('feature_')]
        if len(feature_cols) > 0 and 'quality_score' in training_df.columns:
            try:
                correlations = training_df[feature_cols].corrwith(training_df['quality_score']).abs()
                correlations = correlations.dropna().sort_values(ascending=False)
                if len(correlations) > 0:
                    metrics['top_correlation'] = float(correlations.iloc[0])
                    metrics['strong_features'] = int((correlations > 0.3).sum())
            except:
                pass
        
        # Store in history
        self.metrics_history.append(metrics)
        
        # Check thresholds and generate alerts
        alerts = self._check_metric_thresholds(metrics)
        
        if alerts:
            self.logger.warning(f"\n⚠️  Collection alerts for {timeframe}:")
            for alert in alerts:
                self.logger.warning(f"   • {alert}")
        
        return metrics
    
    def _check_metric_thresholds(self, metrics: Dict) -> List[str]:
        """Check if metrics exceed thresholds."""
        alerts = []
        
        if metrics.get('bounce_saturation', 0) > 0.30:
            alerts.append(f"Bounce saturation: {metrics['bounce_saturation']*100:.1f}% (>30%)")
        
        if metrics.get('duration_seconds', 0) > 300:  # 5 minutes
            alerts.append(f"Slow collection: {metrics['duration_seconds']:.1f}s (>300s)")
        
        if metrics.get('samples_collected', 0) < 50:
            alerts.append(f"Few samples: {metrics['samples_collected']} (<50)")
        
        if metrics.get('top_correlation', 1.0) < 0.20:
            alerts.append(f"Weak correlations: {metrics['top_correlation']:.3f} (<0.20)")
        
        return alerts
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect if quality score distribution has drifted from baseline.
        
        Args:
            current_data: Current dataset to compare
            
        Returns:
            Drift report with statistics
        """
        if self.baseline_data is None:
            self.logger.warning("No baseline data set, cannot detect drift")
            return {'drift_detected': False, 'reason': 'no_baseline'}
        
        from scipy import stats
        
        drift_report = {
            'drift_detected': False,
            'drifted_metrics': [],
            'statistics': {}
        }
        
        # Compare distributions for key metrics
        metrics_to_check = ['bounce_strength', 'rejection_speed', 'volume_quality', 'quality_score']
        
        for metric in metrics_to_check:
            if metric not in current_data.columns or metric not in self.baseline_data.columns:
                continue
            
            try:
                # Kolmogorov-Smirnov test
                ks_stat, p_value = stats.ks_2samp(
                    self.baseline_data[metric].dropna(),
                    current_data[metric].dropna()
                )
                
                drift_report['statistics'][metric] = {
                    'ks_statistic': float(ks_stat),
                    'p_value': float(p_value),
                    'drifted': p_value < 0.05,  # Significant at 5% level
                    'baseline_mean': float(self.baseline_data[metric].mean()),
                    'current_mean': float(current_data[metric].mean()),
                    'mean_change': float(current_data[metric].mean() - self.baseline_data[metric].mean())
                }
                
                if p_value < 0.05:
                    drift_report['drift_detected'] = True
                    drift_report['drifted_metrics'].append(metric)
                    self.logger.warning(f"   Drift detected in {metric}: p={p_value:.4f}")
                
            except Exception as e:
                self.logger.debug(f"Error checking drift for {metric}: {e}")
        
        return drift_report
    
    def generate_quality_report(self, training_df: pd.DataFrame, 
                               output_path: str = None) -> str:
        """Generate comprehensive quality report.
        
        Args:
            training_df: Training data
            output_path: Optional path to save report
            
        Returns:
            Report as string
        """
        lines = []
        lines.append("="*80)
        lines.append("QUALITY SCORE DATA QUALITY REPORT")
        lines.append("="*80)
        lines.append("")
        
        # Basic info
        lines.append(f"Generated: {pd.Timestamp.now()}")
        lines.append(f"Samples: {len(training_df):,}")
        lines.append(f"Columns: {len(training_df.columns)}")
        lines.append("")
        
        # Component statistics
        components = ['bounce_strength', 'hold_strength', 'trade_profit', 
                     'rejection_speed', 'volume_quality']
        
        lines.append("COMPONENT STATISTICS:")
        lines.append("-"*80)
        
        for comp in components:
            if comp in training_df.columns:
                lines.append(f"\n{comp}:")
                lines.append(f"  Mean:   {training_df[comp].mean():.4f}")
                lines.append(f"  Median: {training_df[comp].median():.4f}")
                lines.append(f"  Std:    {training_df[comp].std():.4f}")
                lines.append(f"  Min:    {training_df[comp].min():.4f}")
                lines.append(f"  Max:    {training_df[comp].max():.4f}")
                lines.append(f"  NaN:    {training_df[comp].isna().sum()}")
        
        # Quality distribution
        if 'quality_score' in training_df.columns:
            lines.append("\n" + "-"*80)
            lines.append("QUALITY SCORE DISTRIBUTION:")
            lines.append("-"*80)
            q = training_df['quality_score']
            lines.append(f"  Mean:   {q.mean():.4f}")
            lines.append(f"  Median: {q.median():.4f}")
            lines.append(f"  Std:    {q.std():.4f}")
            lines.append(f"  IQR:    {q.quantile(0.75) - q.quantile(0.25):.4f}")
            lines.append(f"  Range:  [{q.min():.4f}, {q.max():.4f}]")
        
        # Feature correlations
        feature_cols = [c for c in training_df.columns if c.startswith('feature_')]
        if len(feature_cols) > 0 and 'quality_score' in training_df.columns:
            lines.append("\n" + "-"*80)
            lines.append("TOP FEATURE CORRELATIONS:")
            lines.append("-"*80)
            
            try:
                corr = training_df[feature_cols].corrwith(training_df['quality_score']).abs()
                corr = corr.dropna().sort_values(ascending=False).head(10)
                
                for i, (feat, val) in enumerate(corr.items(), 1):
                    lines.append(f"  {i:2d}. {feat.replace('feature_', ''):<40} {val:.4f}")
            except:
                lines.append("  Error calculating correlations")
        
        report_text = "\n".join(lines)
        
        # Save if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"\n✅ Quality report saved to: {output_path}")
        
        return report_text


# Convenience function
def validate_training_data(training_df: pd.DataFrame, 
                          timeframe: str = None,
                          strict: bool = True) -> Dict[str, Any]:
    """Validate training data before use.
    
    Args:
        training_df: Training data DataFrame
        timeframe: Optional timeframe context
        strict: If True, raises exception on critical issues
        
    Returns:
        Validation report
        
    Raises:
        ValueError: If critical issues and strict=True
    """
    validator = QualityDataValidator(strict_mode=strict)
    return validator.validate_training_data(training_df, timeframe)

