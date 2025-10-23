"""
Feature Consolidation and Redundancy Removal

This module handles feature consolidation, redundancy removal, and
multicollinearity screening for the feature comparison framework.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

logger = logging.getLogger(__name__)

class FeatureConsolidator:
    """
    Handles feature consolidation and redundancy removal.
    """
    
    def __init__(self, correlation_threshold: float = 0.95, vif_threshold: float = 10.0):
        """
        Initialize feature consolidator.
        
        Args:
            correlation_threshold: Threshold for correlation-based removal
            vif_threshold: Threshold for VIF-based removal
        """
        self.correlation_threshold = correlation_threshold
        self.vif_threshold = vif_threshold
        self.removed_features = {}
        self.consolidation_rules = self._define_consolidation_rules()
    
    def _define_consolidation_rules(self) -> Dict[str, List[str]]:
        """Define rules for feature consolidation."""
        return {
            # Keep only squared returns (more common in models)
            'returns_consolidation': {
                'keep': ['ret_sq_t1'],
                'remove': ['ret_abs_t1']  # Remove abs if both exist
            },
            
            # Momentum consolidation - keep explicit momentum definition
            'momentum_consolidation': {
                'keep': ['ret_mom_k1', 'ret_mom_k2', 'ret_mom_k3', 'ret_mom_k5'],
                'remove': ['ret_ma_w5', 'ret_ma_w10', 'ret_ma_w20', 'ret_ma_w50']  # Remove if same as momentum
            },
            
            # Acceleration consolidation - keep only one formulation
            'acceleration_consolidation': {
                'keep': ['ret_acc_k1'],
                'remove': ['ret_acceleration']  # Remove alternative formulation
            },
            
            # VWAP feature consolidation
            'vwap_consolidation': {
                'keep': ['vwap_basis_w20', 'rel_vwap_dev_w20'],
                'remove': ['vwap_basis', 'rel_vwap_dev']  # Remove non-standardized versions
            }
        }
    
    def consolidate_features(self, df: pd.DataFrame, version_name: str) -> pd.DataFrame:
        """
        Consolidate features by removing redundant ones.
        
        Args:
            df: Input DataFrame
            version_name: Name of the feature version
            
        Returns:
            Consolidated DataFrame
        """
        logger.info(f"Consolidating features for {version_name}")
        
        df_consolidated = df.copy()
        removed_features = []
        
        # Apply consolidation rules
        for rule_name, rule in self.consolidation_rules.items():
            keep_features = rule['keep']
            remove_features = rule['remove']
            
            # Check which features exist
            existing_keep = [f for f in keep_features if f in df_consolidated.columns]
            existing_remove = [f for f in remove_features if f in df_consolidated.columns]
            
            if existing_keep and existing_remove:
                # Remove redundant features
                df_consolidated = df_consolidated.drop(columns=existing_remove)
                removed_features.extend(existing_remove)
                logger.info(f"Removed {len(existing_remove)} redundant features: {existing_remove}")
        
        # Store removed features
        self.removed_features[version_name] = removed_features
        
        return df_consolidated
    
    def remove_multicollinearity(self, df: pd.DataFrame, version_name: str) -> pd.DataFrame:
        """
        Remove features with high multicollinearity.
        
        Args:
            df: Input DataFrame
            version_name: Name of the feature version
            
        Returns:
            DataFrame with multicollinearity removed
        """
        logger.info(f"Removing multicollinearity for {version_name}")
        
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_numeric = df[numeric_cols].dropna()
        
        if len(df_numeric) == 0:
            logger.warning("No numeric data available for multicollinearity analysis")
            return df
        
        removed_features = []
        
        # 1. Remove features with high correlation
        correlation_matrix = df_numeric.corr().abs()
        
        # Find pairs with high correlation
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if correlation_matrix.iloc[i, j] > self.correlation_threshold:
                    high_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        correlation_matrix.iloc[i, j]
                    ))
        
        # Remove one feature from each high correlation pair
        features_to_remove = set()
        for feat1, feat2, corr in high_corr_pairs:
            if feat1 not in features_to_remove and feat2 not in features_to_remove:
                # Remove the feature with lower variance
                var1 = df_numeric[feat1].var()
                var2 = df_numeric[feat2].var()
                if var1 < var2:
                    features_to_remove.add(feat1)
                else:
                    features_to_remove.add(feat2)
        
        # 2. Remove features with high VIF
        if len(df_numeric) > len(df_numeric.columns):
            vif_features = self._calculate_vif(df_numeric)
            high_vif_features = vif_features[vif_features['VIF'] > self.vif_threshold]['Feature'].tolist()
            features_to_remove.update(high_vif_features)
        
        # Remove features
        features_to_remove = list(features_to_remove)
        if features_to_remove:
            df_cleaned = df.drop(columns=features_to_remove)
            removed_features.extend(features_to_remove)
            logger.info(f"Removed {len(features_to_remove)} features due to multicollinearity")
        else:
            df_cleaned = df
            logger.info("No multicollinearity issues found")
        
        # Store removed features
        if version_name not in self.removed_features:
            self.removed_features[version_name] = []
        self.removed_features[version_name].extend(removed_features)
        
        return df_cleaned
    
    def _calculate_vif(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Variance Inflation Factor for features."""
        vif_data = []
        
        for i, feature in enumerate(df.columns):
            try:
                vif = variance_inflation_factor(df.values, i)
                vif_data.append({'Feature': feature, 'VIF': vif})
            except:
                vif_data.append({'Feature': feature, 'VIF': np.inf})
        
        return pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
    
    def winsorize_features(self, df: pd.DataFrame, lower: float = 0.005, upper: float = 0.995) -> pd.DataFrame:
        """
        Winsorize features to handle outliers.
        
        Args:
            df: Input DataFrame
            lower: Lower percentile for winsorization
            upper: Upper percentile for winsorization
            
        Returns:
            Winsorized DataFrame
        """
        logger.info(f"Winsorizing features at {lower*100:.1f}% and {upper*100:.1f}%")
        
        df_winsorized = df.copy()
        
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in df_winsorized.columns:
                lower_bound = df[col].quantile(lower)
                upper_bound = df[col].quantile(upper)
                
                df_winsorized[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df_winsorized
    
    def get_feature_stability_report(self, versions: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate feature stability report across versions.
        
        Args:
            versions: Dictionary with feature versions
            
        Returns:
            Stability report
        """
        report = {
            'version_comparison': {},
            'common_features': set(),
            'unique_features': {},
            'stability_metrics': {}
        }
        
        # Get all features across versions
        all_features = set()
        for version_name, df in versions.items():
            features = set(df.columns)
            all_features.update(features)
            report['unique_features'][version_name] = features
        
        # Find common features
        if versions:
            report['common_features'] = set.intersection(*[set(df.columns) for df in versions.values()])
        
        # Calculate stability metrics
        for version_name, df in versions.items():
            features = set(df.columns)
            
            # Feature count
            report['version_comparison'][version_name] = {
                'total_features': len(features),
                'common_features': len(features & report['common_features']),
                'unique_features': len(features - report['common_features'])
            }
            
            # Data quality metrics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            report['stability_metrics'][version_name] = {
                'n_samples': len(df),
                'n_numeric_features': len(numeric_cols),
                'missing_data_pct': df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100,
                'infinite_values': np.isinf(df[numeric_cols]).sum().sum(),
                'zero_variance_features': len([col for col in numeric_cols if df[col].var() == 0])
            }
        
        return report
    
    def get_consolidation_summary(self) -> Dict[str, Any]:
        """
        Get summary of feature consolidation.
        
        Returns:
            Consolidation summary
        """
        return {
            'consolidation_rules': self.consolidation_rules,
            'removed_features': self.removed_features,
            'total_removed': sum(len(features) for features in self.removed_features.values())
        }

class FeatureValidator:
    """
    Validates feature quality and consistency.
    """
    
    def __init__(self):
        """Initialize feature validator."""
        self.validation_results = {}
    
    def validate_features(self, df: pd.DataFrame, version_name: str) -> Dict[str, Any]:
        """
        Validate feature quality.
        
        Args:
            df: Input DataFrame
            version_name: Name of the feature version
            
        Returns:
            Validation results
        """
        logger.info(f"Validating features for {version_name}")
        
        validation = {
            'version_name': version_name,
            'data_quality': {},
            'feature_quality': {},
            'warnings': [],
            'errors': []
        }
        
        # Data quality checks
        validation['data_quality'] = {
            'n_samples': len(df),
            'n_features': len(df.columns),
            'missing_data_pct': df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100,
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Feature quality checks
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            validation['feature_quality'] = {
                'n_numeric_features': len(numeric_cols),
                'zero_variance_features': len([col for col in numeric_cols if df[col].var() == 0]),
                'infinite_values': np.isinf(df[numeric_cols]).sum().sum(),
                'constant_features': len([col for col in numeric_cols if df[col].nunique() == 1]),
                'high_missing_pct_features': len([col for col in numeric_cols if df[col].isnull().sum() / len(df) > 0.5])
            }
            
            # Check for problematic features
            for col in numeric_cols:
                if df[col].var() == 0:
                    validation['warnings'].append(f"Zero variance feature: {col}")
                
                if np.isinf(df[col]).sum() > 0:
                    validation['errors'].append(f"Infinite values in feature: {col}")
                
                if df[col].isnull().sum() / len(df) > 0.5:
                    validation['warnings'].append(f"High missing data in feature: {col}")
        
        # Store validation results
        self.validation_results[version_name] = validation
        
        return validation
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get validation summary across all versions.
        
        Returns:
            Validation summary
        """
        if not self.validation_results:
            return {}
        
        summary = {
            'total_versions': len(self.validation_results),
            'version_summaries': {},
            'overall_quality': {}
        }
        
        # Aggregate quality metrics
        all_warnings = []
        all_errors = []
        total_features = 0
        total_samples = 0
        
        for version_name, validation in self.validation_results.items():
            summary['version_summaries'][version_name] = {
                'n_features': validation['data_quality']['n_features'],
                'n_samples': validation['data_quality']['n_samples'],
                'warnings': len(validation['warnings']),
                'errors': len(validation['errors'])
            }
            
            all_warnings.extend(validation['warnings'])
            all_errors.extend(validation['errors'])
            total_features += validation['data_quality']['n_features']
            total_samples += validation['data_quality']['n_samples']
        
        summary['overall_quality'] = {
            'total_warnings': len(all_warnings),
            'total_errors': len(all_errors),
            'avg_features_per_version': total_features / len(self.validation_results),
            'avg_samples_per_version': total_samples / len(self.validation_results)
        }
        
        return summary