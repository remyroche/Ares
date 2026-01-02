"""
Streamlined Layer 3 Causal Features

Essential causal meta-features for Layer 3 meta-learner.
Only 8 critical features to maximize benefit with minimum complexity.

Key Features:
1. Causal surprise flag
2. Specialist disagreement
3. Environment cluster
4. Meta confidence
5. Causal residual strength
6. Parent effect consistency
7. Intervention response
8. Causal variance ratio
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import warnings

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    # Fallback print functions
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")

class StreamlinedLayer3Features:
    """
    Streamlined Layer 3 causal features with only 8 essential features.
    
    Focuses on maximum causal insight with minimum computational overhead.
    """
    
    def __init__(
        self,
        surprise_threshold: float = 1.8,
        rolling_window: int = 20,
        n_clusters: int = 3,
        verbose: bool = True
    ):
        """
        Initialize Streamlined Layer 3 Features.
        
        Args:
            surprise_threshold: Threshold for causal surprise
            rolling_window: Window for rolling statistics
            n_clusters: Number of environment clusters
            verbose: Whether to print progress information
        """
        self.surprise_threshold = surprise_threshold
        self.rolling_window = rolling_window
        self.n_clusters = n_clusters
        self.verbose = verbose
        
        # Storage for computed features
        self.feature_cache_ = {}
        
    def generate_streamlined_features(
        self,
        df: pd.DataFrame,
        base_model_cols: List[str],
        target_col: str,
        specialist_surprises: Optional[Dict[str, np.ndarray]] = None,
        specialist_predictions: Optional[Dict[str, np.ndarray]] = None,
        custom_features: Optional[pd.DataFrame] = None,
        causal_effects: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Generate 8 streamlined Layer 3 causal features.
        
        Args:
            df: DataFrame with base model predictions
            base_model_cols: List of base model column names
            target_col: Target column name
            specialist_surprises: Surprise values from specialists
            specialist_predictions: Specialist prediction values
            custom_features: Custom regime features
            causal_effects: Causal effect estimates
            
        Returns:
            DataFrame with 8 streamlined causal features
        """
        try:
            if self.verbose:
                tprint_info("🎯 Generating Streamlined Layer 3 Causal Features...")
            
            features = pd.DataFrame(index=df.index)
            
            # Feature 1: Causal surprise flag
            features['causal_surprise_flag'] = self._compute_causal_surprise_flag(
                specialist_surprises, df
            )
            
            # Feature 2: Specialist disagreement
            features['specialist_disagreement'] = self._compute_specialist_disagreement(
                specialist_predictions, df
            )
            
            # Feature 3: Environment cluster
            features['environment_cluster'] = self._compute_environment_cluster(
                custom_features, df
            )
            
            # Feature 4: Meta confidence
            features['meta_confidence'] = self._compute_meta_confidence(
                df, base_model_cols, target_col
            )
            
            # Feature 5: Causal residual strength
            features['causal_residual_strength'] = self._compute_causal_residual_strength(
                df, base_model_cols, target_col, causal_effects
            )
            
            # Feature 6: Parent effect consistency
            features['parent_effect_consistency'] = self._compute_parent_effect_consistency(
                df, specialist_predictions
            )
            
            # Feature 7: Intervention response
            features['intervention_response'] = self._compute_intervention_response(
                df, custom_features
            )
            
            # Feature 8: Causal variance ratio
            features['causal_variance_ratio'] = self._compute_causal_variance_ratio(
                df, base_model_cols, target_col
            )
            
            # Cache features
            self.feature_cache_ = features
            
            if self.verbose:
                tprint_success(f"✅ Generated {len(features.columns)} streamlined Layer 3 features:")
                for col in features.columns:
                    tprint_info(f"   - {col}")
            
            return features
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Streamlined Layer 3 feature generation failed: {e}")
            return pd.DataFrame(index=df.index)
    
    def _compute_causal_surprise_flag(
        self,
        specialist_surprises: Optional[Dict[str, np.ndarray]],
        df: pd.DataFrame
    ) -> pd.Series:
        """Compute causal surprise flag from specialist surprises."""
        try:
            surprise_flags = pd.Series(0, index=df.index)
            
            if specialist_surprises:
                for spec_type, surprises in specialist_surprises.items():
                    if len(surprises) > 0:
                        # Create surprise series
                        surprise_series = pd.Series(0, index=df.index)
                        for i, surprise_val in enumerate(surprises):
                            if i < len(surprise_series):
                                surprise_series.iloc[i] = surprise_val
                        
                        # Flag surprises above threshold
                        surprise_flags = surprise_flags | (surprise_series > self.surprise_threshold)
            
            return surprise_flags.astype(int)
            
        except Exception:
            return pd.Series(0, index=df.index)
    
    def _compute_specialist_disagreement(
        self,
        specialist_predictions: Optional[Dict[str, np.ndarray]],
        df: pd.DataFrame
    ) -> pd.Series:
        """Compute specialist disagreement."""
        try:
            if specialist_predictions and len(specialist_predictions) >= 2:
                spec_preds_df = pd.DataFrame(specialist_predictions, index=df.index)
                spec_range = spec_preds_df.max(axis=1) - spec_preds_df.min(axis=1)
                return spec_range.fillna(0)
            else:
                return pd.Series(0, index=df.index)
                
        except Exception:
            return pd.Series(0, index=df.index)
    
    def _compute_environment_cluster(
        self,
        custom_features: Optional[pd.DataFrame],
        df: pd.DataFrame
    ) -> pd.Series:
        """Compute environment cluster."""
        try:
            env_vars = []
            
            # Get environment variables
            if 'volatility' in df.columns:
                env_vars.append('volatility')
            
            if custom_features is not None:
                for col in ['vol_regime_high', 'price_trend', 'vol_relative']:
                    if col in custom_features.columns:
                        env_vars.append(col)
            
            if len(env_vars) >= 2:
                try:
                    # Use available data
                    if env_vars[0] in df.columns:
                        env_data = df[env_vars].fillna(0)
                    else:
                        env_data = custom_features[env_vars].fillna(0)
                    
                    # K-means clustering
                    kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
                    env_clusters = kmeans.fit_predict(env_data)
                    return pd.Series(env_clusters, index=df.index)
                    
                except Exception:
                    return pd.Series(0, index=df.index)
            else:
                return pd.Series(0, index=df.index)
                
        except Exception:
            return pd.Series(0, index=df.index)
    
    def _compute_meta_confidence(
        self,
        df: pd.DataFrame,
        base_model_cols: List[str],
        target_col: str
    ) -> pd.Series:
        """Compute meta confidence from residual variance."""
        try:
            if target_col in df.columns and len(base_model_cols) > 0:
                residuals = []
                for base_col in base_model_cols:
                    if base_col in df.columns:
                        residual = df[target_col] - df[base_col]
                        std_residual = residual / (residual.rolling(self.rolling_window).std() + 1e-8)
                        residuals.append(std_residual.fillna(0))
                
                if residuals:
                    residuals_matrix = pd.concat(residuals, axis=1)
                    avg_residual = residuals_matrix.mean(axis=1)
                    residual_vol = avg_residual.rolling(self.rolling_window).std()
                    confidence = (1 / (1 + residual_vol)).fillna(0.5)
                    return confidence
                else:
                    return pd.Series(0.5, index=df.index)
            else:
                return pd.Series(0.5, index=df.index)
                
        except Exception:
            return pd.Series(0.5, index=df.index)
    
    def _compute_causal_residual_strength(
        self,
        df: pd.DataFrame,
        base_model_cols: List[str],
        target_col: str,
        causal_effects: Optional[Dict[str, float]]
    ) -> pd.Series:
        """Compute causal residual strength."""
        try:
            if target_col in df.columns and len(base_model_cols) > 0:
                # Compute average prediction
                valid_cols = [col for col in base_model_cols if col in df.columns]
                if valid_cols:
                    avg_prediction = df[valid_cols].mean(axis=1)
                    residuals = df[target_col] - avg_prediction
                    
                    # Strength as absolute residual
                    strength = np.abs(residuals)
                    
                    # Adjust by causal effects if available
                    if causal_effects:
                        avg_effect = np.mean(list(causal_effects.values()))
                        strength = strength * (1 + avg_effect)
                    
                    return strength.fillna(0)
                else:
                    return pd.Series(0, index=df.index)
            else:
                return pd.Series(0, index=df.index)
                
        except Exception:
            return pd.Series(0, index=df.index)
    
    def _compute_parent_effect_consistency(
        self,
        df: pd.DataFrame,
        specialist_predictions: Optional[Dict[str, np.ndarray]]
    ) -> pd.Series:
        """Compute parent effect consistency."""
        try:
            if specialist_predictions and len(specialist_predictions) >= 2:
                spec_preds_df = pd.DataFrame(specialist_predictions, index=df.index)
                
                # Consistency as inverse of variance
                pred_variance = spec_preds_df.var(axis=1)
                consistency = 1 / (1 + pred_variance)
                
                return consistency.fillna(0.5)
            else:
                return pd.Series(0.5, index=df.index)
                
        except Exception:
            return pd.Series(0.5, index=df.index)
    
    def _compute_intervention_response(
        self,
        df: pd.DataFrame,
        custom_features: Optional[pd.DataFrame]
    ) -> pd.Series:
        """Compute intervention response."""
        try:
            # Look for intervention-related features
            intervention_features = []
            
            if custom_features is not None:
                for col in custom_features.columns:
                    if any(keyword in col.lower() for keyword in ['intervention', 'shock', 'break']):
                        intervention_features.append(col)
            
            if intervention_features:
                # Use first intervention feature
                feature_data = custom_features[intervention_features[0]].fillna(0)
                
                # Response as absolute value
                response = np.abs(feature_data)
                
                return response
            else:
                # Fallback to volatility as intervention proxy
                if 'volatility' in df.columns:
                    return np.abs(df['volatility']).fillna(0)
                else:
                    return pd.Series(0, index=df.index)
                    
        except Exception:
            return pd.Series(0, index=df.index)
    
    def _compute_causal_variance_ratio(
        self,
        df: pd.DataFrame,
        base_model_cols: List[str],
        target_col: str
    ) -> pd.Series:
        """Compute causal variance ratio."""
        try:
            if target_col in df.columns and len(base_model_cols) > 0:
                valid_cols = [col for col in base_model_cols if col in df.columns]
                
                if valid_cols:
                    # Compute prediction variance
                    pred_variance = df[valid_cols].var(axis=1)
                    
                    # Compute target variance
                    target_variance = df[target_col].rolling(self.rolling_window).var()
                    
                    # Ratio
                    ratio = pred_variance / (target_variance + 1e-8)
                    
                    return ratio.fillna(1.0)
                else:
                    return pd.Series(1.0, index=df.index)
            else:
                return pd.Series(1.0, index=df.index)
                
        except Exception:
            return pd.Series(1.0, index=df.index)
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of the 8 streamlined features.
        
        Returns:
            Dictionary of feature descriptions
        """
        return {
            'causal_surprise_flag': 'Binary indicator of causal surprise events',
            'specialist_disagreement': 'Maximum disagreement between specialist predictions',
            'environment_cluster': 'Environment cluster ID (0-2)',
            'meta_confidence': 'Inverse of residual variance (confidence measure)',
            'causal_residual_strength': 'Strength of causal residuals',
            'parent_effect_consistency': 'Consistency of parent effects across specialists',
            'intervention_response': 'Response to interventions/shocks',
            'causal_variance_ratio': 'Ratio of prediction to target variance'
        }

# Convenience function
def quick_streamlined_layer3_features(
    df: pd.DataFrame,
    base_model_cols: List[str],
    target_col: str,
    **kwargs
) -> pd.DataFrame:
    """
    Quick streamlined Layer 3 feature generation.
    
    Args:
        df: DataFrame with base model predictions
        base_model_cols: List of base model column names
        target_col: Target column name
        **kwargs: Additional parameters
        
    Returns:
        DataFrame with 8 streamlined causal features
    """
    generator = StreamlinedLayer3Features(**kwargs)
    return generator.generate_streamlined_features(
        df, base_model_cols, target_col
    )
