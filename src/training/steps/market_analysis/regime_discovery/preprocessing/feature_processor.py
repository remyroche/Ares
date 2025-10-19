"""
Feature Processor with Numerical Stability

Implements train-aware preprocessing with per-asset transformers,
correlation pruning, MI/HSIC pruning, and numerical stability floors.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr
import warnings

from ..config.regime_discovery_config import RegimeDiscoveryConfig

logger = logging.getLogger(__name__)


class FeatureProcessor:
    """
    Feature processor with numerical stability and per-asset transformers.
    
    Implements:
    - Winsorization with variance floors
    - Per-asset QuantileTransformer fitting
    - Correlation and MI/HSIC pruning
    - Missing value handling
    - Cold asset management
    """
    
    def __init__(self, config: RegimeDiscoveryConfig):
        """Initialize the feature processor."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Per-asset transformers
        self.asset_transformers = {}
        self.global_transformer = None
        self.asset_history_counts = {}
        
        # Processing metadata
        self.zero_variance_log = []
        self.dropped_features_log = []
        self.processing_metadata = {}
        
        self.logger.info("FeatureProcessor initialized")
    
    def process(self, features: Union[np.ndarray, pd.DataFrame], 
                fit: bool = False, 
                asset_id: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process features with numerical stability and per-asset transformers.
        
        Args:
            features: Input features (numpy array or DataFrame)
            fit: Whether to fit transformers (train mode)
            asset_id: Optional asset identifier for per-asset processing
            
        Returns:
            Tuple of (processed_features, metadata)
        """
        start_time = pd.Timestamp.now()
        
        try:
            # Convert to DataFrame if needed
            if isinstance(features, np.ndarray):
                features_df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(features.shape[1])])
            else:
                features_df = features.copy()
            
            self.logger.info(f"Processing {features_df.shape[1]} features for {features_df.shape[0]} samples")
            
            # Initialize metadata
            metadata = {
                'input_shape': features_df.shape,
                'processing_steps': [],
                'dropped_features': [],
                'zero_variance_features': [],
                'asset_id': asset_id,
                'fit_mode': fit
            }
            
            # Step 1: Handle missing values
            features_df = self._handle_missing_values(features_df)
            metadata['processing_steps'].append('missing_values_handled')
            
            # Step 2: Winsorization
            features_df = self._winsorize_features(features_df)
            metadata['processing_steps'].append('winsorization')
            
            # Step 3: Check for zero variance features
            zero_var_features = self._identify_zero_variance_features(features_df)
            if zero_var_features:
                features_df = features_df.drop(columns=zero_var_features)
                metadata['zero_variance_features'] = zero_var_features
                metadata['processing_steps'].append('zero_variance_removal')
                self.logger.warning(f"Dropped {len(zero_var_features)} zero-variance features")
            
            # Step 4: Correlation pruning
            features_df = self._prune_correlated_features(features_df, metadata)
            metadata['processing_steps'].append('correlation_pruning')
            
            # Step 5: MI/HSIC pruning (if enabled)
            if self.config.mi_threshold < 1.0:
                features_df = self._prune_mi_features(features_df, metadata)
                metadata['processing_steps'].append('mi_pruning')
            
            # Step 6: Apply transformer
            if fit:
                processed_features = self._fit_and_transform(features_df, asset_id)
            else:
                processed_features = self._transform_only(features_df, asset_id)
            
            metadata['processing_steps'].append('transformation')
            metadata['output_shape'] = processed_features.shape
            metadata['processing_time'] = (pd.Timestamp.now() - start_time).total_seconds()
            
            self.logger.info(f"✅ Feature processing completed: {processed_features.shape[1]} features in {metadata['processing_time']:.2f}s")
            
            return processed_features, metadata
            
        except Exception as e:
            self.logger.error(f"❌ Feature processing failed: {e}")
            raise
    
    def _handle_missing_values(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with forward-fill and median imputation."""
        # Forward fill first
        features_df = features_df.fillna(method='ffill')
        
        # Then median imputation for any remaining NaNs
        for col in features_df.columns:
            if features_df[col].isna().any():
                median_val = features_df[col].median()
                features_df[col] = features_df[col].fillna(median_val)
        
        return features_df
    
    def _winsorize_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Apply winsorization to limit extreme values."""
        lower_limit, upper_limit = self.config.winsorize_limits
        
        for col in features_df.columns:
            if features_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                # Compute quantiles
                lower_q = features_df[col].quantile(lower_limit)
                upper_q = features_df[col].quantile(upper_limit)
                
                # Clip values
                features_df[col] = features_df[col].clip(lower=lower_q, upper=upper_q)
        
        return features_df
    
    def _identify_zero_variance_features(self, features_df: pd.DataFrame) -> List[str]:
        """Identify features with zero or near-zero variance."""
        zero_var_features = []
        
        for col in features_df.columns:
            if features_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                variance = features_df[col].var()
                
                # Check for zero variance or variance below floor
                if variance == 0 or variance < self.config.variance_floor:
                    zero_var_features.append(col)
                    self.zero_variance_log.append({
                        'feature': col,
                        'variance': variance,
                        'threshold': self.config.variance_floor
                    })
        
        return zero_var_features
    
    def _prune_correlated_features(self, features_df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Remove highly correlated features."""
        if features_df.shape[1] <= 1:
            return features_df
        
        # Compute correlation matrix
        corr_matrix = features_df.corr().abs()
        
        # Find pairs above threshold
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > self.config.correlation_threshold:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        
        # Remove features (keep the one with higher variance)
        features_to_remove = set()
        for feat1, feat2, corr in high_corr_pairs:
            if feat1 not in features_to_remove and feat2 not in features_to_remove:
                var1 = features_df[feat1].var()
                var2 = features_df[feat2].var()
                
                if var1 >= var2:
                    features_to_remove.add(feat2)
                else:
                    features_to_remove.add(feat1)
        
        if features_to_remove:
            features_df = features_df.drop(columns=list(features_to_remove))
            metadata['dropped_features'].extend(list(features_to_remove))
            self.logger.info(f"Dropped {len(features_to_remove)} highly correlated features")
        
        return features_df
    
    def _prune_mi_features(self, features_df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Remove features with high mutual information (nonlinear duplicates)."""
        if features_df.shape[1] <= 1:
            return features_df
        
        try:
            # Compute MI matrix (this can be expensive)
            mi_matrix = np.zeros((features_df.shape[1], features_df.shape[1]))
            feature_names = features_df.columns.tolist()
            
            for i, col1 in enumerate(feature_names):
                for j, col2 in enumerate(feature_names):
                    if i != j:
                        # Compute MI between features
                        mi_val = mutual_info_regression(
                            features_df[[col1]], 
                            features_df[col2].values,
                            random_state=42
                        )[0]
                        mi_matrix[i, j] = mi_val
            
            # Find high MI pairs
            high_mi_pairs = []
            for i in range(len(feature_names)):
                for j in range(i+1, len(feature_names)):
                    if mi_matrix[i, j] > self.config.mi_threshold:
                        high_mi_pairs.append((feature_names[i], feature_names[j], mi_matrix[i, j]))
            
            # Remove features (keep the one with higher variance)
            features_to_remove = set()
            for feat1, feat2, mi_val in high_mi_pairs:
                if feat1 not in features_to_remove and feat2 not in features_to_remove:
                    var1 = features_df[feat1].var()
                    var2 = features_df[feat2].var()
                    
                    if var1 >= var2:
                        features_to_remove.add(feat2)
                    else:
                        features_to_remove.add(feat1)
            
            if features_to_remove:
                features_df = features_df.drop(columns=list(features_to_remove))
                metadata['dropped_features'].extend(list(features_to_remove))
                self.logger.info(f"Dropped {len(features_to_remove)} high-MI features")
            
        except Exception as e:
            self.logger.warning(f"MI pruning failed: {e}, skipping")
        
        return features_df
    
    def _fit_and_transform(self, features_df: pd.DataFrame, asset_id: Optional[str]) -> np.ndarray:
        """Fit transformer and transform features."""
        if self.config.per_asset_fitting and asset_id:
            return self._fit_per_asset_transformer(features_df, asset_id)
        else:
            return self._fit_global_transformer(features_df)
    
    def _transform_only(self, features_df: pd.DataFrame, asset_id: Optional[str]) -> np.ndarray:
        """Transform features using existing transformer."""
        if self.config.per_asset_fitting and asset_id and asset_id in self.asset_transformers:
            transformer = self.asset_transformers[asset_id]
        elif self.global_transformer is not None:
            transformer = self.global_transformer
        else:
            # No transformer available, return original features
            self.logger.warning("No transformer available, returning original features")
            return features_df.values
        
        try:
            return transformer.transform(features_df)
        except Exception as e:
            self.logger.error(f"Transformation failed: {e}")
            raise
    
    def _fit_per_asset_transformer(self, features_df: pd.DataFrame, asset_id: str) -> np.ndarray:
        """Fit per-asset transformer."""
        # Check if asset has sufficient history
        if asset_id not in self.asset_history_counts:
            self.asset_history_counts[asset_id] = 0
        self.asset_history_counts[asset_id] += len(features_df)
        
        # Use frozen global transformer for cold assets
        if self.asset_history_counts[asset_id] < self.config.min_history_for_asset_fit:
            self.logger.info(f"Cold asset {asset_id}: using global transformer "
                           f"({self.asset_history_counts[asset_id]} < "
                           f"{self.config.min_history_for_asset_fit} samples)")
            
            if self.global_transformer is None:
                # Fit global transformer first
                self.global_transformer = self._create_transformer()
                self.global_transformer.fit(features_df)
            
            return self.global_transformer.transform(features_df)
        
        # Fit per-asset transformer
        transformer = self._create_transformer()
        transformer.fit(features_df)
        self.asset_transformers[asset_id] = transformer
        
        return transformer.transform(features_df)
    
    def _fit_global_transformer(self, features_df: pd.DataFrame) -> np.ndarray:
        """Fit global transformer."""
        self.global_transformer = self._create_transformer()
        self.global_transformer.fit(features_df)
        
        return self.global_transformer.transform(features_df)
    
    def _create_transformer(self):
        """Create appropriate transformer based on config."""
        if self.config.quantile_transformer_output == 'normal':
            return QuantileTransformer(
                output_distribution='normal',
                n_quantiles=min(1000, 1000),  # Limit quantiles for memory
                random_state=self.config.random_state
            )
        else:
            return RobustScaler()
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processing results."""
        return {
            'zero_variance_features': len(self.zero_variance_log),
            'dropped_features': len(self.dropped_features_log),
            'asset_transformers': len(self.asset_transformers),
            'global_transformer_available': self.global_transformer is not None,
            'asset_history_counts': self.asset_history_counts,
            'processing_metadata': self.processing_metadata
        }
    
    def save_metadata(self) -> Dict[str, Any]:
        """Save processing metadata for reproducibility."""
        return {
            'zero_variance_features': self.zero_variance_log,
            'dropped_features': self.dropped_features_log,
            'variance_floor': self.config.variance_floor,
            'correlation_threshold': self.config.correlation_threshold,
            'mi_threshold': self.config.mi_threshold,
            'per_asset_fitting': self.config.per_asset_fitting,
            'min_history_for_asset_fit': self.config.min_history_for_asset_fit
        }
