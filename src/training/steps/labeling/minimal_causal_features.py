"""
Minimal Causal Features for Modern De Prado Implementation

Only the most essential causal features - 5-7 total maximum.
Focus on maximum benefit with minimum complexity.

Layer 3 Features (3-4):
1. causal_surprise_flag - Binary mechanism break indicator
2. specialist_disagreement - Max disagreement between specialists  
3. environment_cluster - Single environment ID
4. meta_confidence - Inverse of residual variance

Layer 4 Features (2-3):
1. invariance_score - Rolling coefficient stability
2. execution_friction - Volume volatility × price volatility
3. position_adjustment - Combined risk adjustment factor
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

def generate_minimal_layer3_features(
    df: pd.DataFrame,
    base_model_cols: List[str],
    target_col: str,
    specialist_surprises: Optional[Dict[str, np.ndarray]] = None,
    specialist_predictions: Optional[Dict[str, np.ndarray]] = None,
    custom_features: Optional[pd.DataFrame] = None,
    surprise_threshold: float = 1.8,
    rolling_window: int = 20,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate minimal Layer 3 causal features (3-4 features max).
    
    Args:
        df: DataFrame with base model predictions
        base_model_cols: List of base model column names
        target_col: Target column name
        specialist_surprises: Surprise values from specialists
        specialist_predictions: Specialist prediction values
        custom_features: Custom regime features
        surprise_threshold: Threshold for causal surprise detection
        rolling_window: Window for rolling statistics
        verbose: Whether to print progress information
        
    Returns:
        DataFrame with 3-4 essential causal features
    """
    try:
        features = pd.DataFrame(index=df.index)
        
        # 1. Causal surprise flag (binary mechanism break indicator)
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
                    surprise_flags = surprise_flags | (surprise_series > surprise_threshold)
        
        features['causal_surprise_flag'] = surprise_flags.astype(int)
        
        # 2. Specialist disagreement (max disagreement between specialists)
        if specialist_predictions and len(specialist_predictions) >= 2:
            spec_preds_df = pd.DataFrame(specialist_predictions, index=df.index)
            spec_range = spec_preds_df.max(axis=1) - spec_preds_df.min(axis=1)
            features['specialist_disagreement'] = spec_range.fillna(0)
        else:
            features['specialist_disagreement'] = 0
        
        # 3. Environment cluster (single environment ID)
        env_vars = []
        if 'volatility' in df.columns:
            env_vars.append('volatility')
        if custom_features is not None:
            for col in ['vol_regime_high', 'price_trend', 'vol_relative']:
                if col in custom_features.columns:
                    env_vars.append(col)
        
        if len(env_vars) >= 2:
            try:
                env_data = df[env_vars].fillna(0) if env_vars[0] in df.columns else custom_features[env_vars].fillna(0)
                
                # Simple k-means clustering (3 clusters)
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                env_clusters = kmeans.fit_predict(env_data)
                features['environment_cluster'] = env_clusters
            except Exception:
                features['environment_cluster'] = 0
        else:
            features['environment_cluster'] = 0
        
        # 4. Meta confidence (inverse of residual variance)
        if target_col in df.columns and len(base_model_cols) > 0:
            residuals = []
            for base_col in base_model_cols:
                if base_col in df.columns:
                    residual = df[target_col] - df[base_col]
                    std_residual = residual / (residual.rolling(rolling_window).std() + 1e-8)
                    residuals.append(std_residual.fillna(0))
            
            if residuals:
                residuals_matrix = pd.concat(residuals, axis=1)
                avg_residual = residuals_matrix.mean(axis=1)
                residual_vol = avg_residual.rolling(rolling_window).std()
                features['meta_confidence'] = (1 / (1 + residual_vol)).fillna(0.5)
            else:
                features['meta_confidence'] = 0.5
        else:
            features['meta_confidence'] = 0.5
        
        if verbose:
            tprint_success(f"✅ Generated {len(features.columns)} minimal Layer 3 features:")
            for col in features.columns:
                tprint_info(f"   - {col}")
        
        return features
        
    except Exception as e:
        if verbose:
            tprint_warning(f"⚠️ Minimal Layer 3 features failed: {e}")
        return pd.DataFrame(index=df.index)

def generate_minimal_layer4_features(
    df: pd.DataFrame,
    base_model_cols: List[str],
    target_col: str,
    market_data: Optional[pd.DataFrame] = None,
    volume_data: Optional[pd.Series] = None,
    invariance_window: int = 50,
    execution_window: int = 10,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate minimal Layer 4 causal features (2-3 features max).
    
    Args:
        df: DataFrame with features and predictions
        base_model_cols: List of base model column names
        target_col: Target column name
        market_data: Additional market data (OHLCV)
        volume_data: Volume data
        invariance_window: Window for invariance statistics
        execution_window: Window for execution features
        verbose: Whether to print progress information
        
    Returns:
        DataFrame with 2-3 essential causal risk features
    """
    try:
        features = pd.DataFrame(index=df.index)
        
        # 1. Invariance score (rolling coefficient stability)
        invariance_scores = []
        for base_col in base_model_cols:
            if base_col in df.columns and target_col in df.columns:
                # Rolling coefficient stability
                rolling_coefs = []
                
                for i in range(invariance_window, len(df)):
                    window_data = df.iloc[i-invariance_window:i]
                    
                    if len(window_data) >= 10:
                        try:
                            X = window_data[[base_col]].values
                            y = window_data[target_col].values
                            
                            model = LinearRegression()
                            model.fit(X, y)
                            rolling_coefs.append(model.coef_[0])
                        except Exception:
                            rolling_coefs.append(0)
                
                if rolling_coefs:
                    coef_series = pd.Series(rolling_coefs, index=df.index[invariance_window:])
                    coef_stability = 1 / (1 + coef_series.rolling(invariance_window).std())
                    invariance_scores.append(coef_stability.fillna(0.5))
        
        if invariance_scores:
            features['invariance_score'] = pd.concat(invariance_scores, axis=1).mean(axis=1)
        else:
            features['invariance_score'] = 0.5
        
        # 2. Execution friction (volume volatility × price volatility)
        execution_frictions = []
        
        # Volume volatility
        if volume_data is not None:
            volume_aligned = volume_data.reindex(df.index).fillna(0)
            volume_vol = volume_aligned.rolling(execution_window).std()
            execution_frictions.append(volume_vol.fillna(0))
        
        # Price volatility
        if market_data is not None and 'close' in market_data.columns:
            price_data = market_data['close'].reindex(df.index).fillna(method='ffill')
            price_vol = price_data.pct_change().rolling(execution_window).std()
            execution_frictions.append(price_vol.fillna(0))
        elif 'volatility' in df.columns:
            price_vol = df['volatility'].rolling(execution_window).std()
            execution_frictions.append(price_vol.fillna(0))
        
        if execution_frictions:
            # Combined friction (product of volatilities)
            features['execution_friction'] = pd.concat(execution_frictions, axis=1).prod(axis=1)
        else:
            features['execution_friction'] = 0.01
        
        # 3. Position adjustment factor (combined risk adjustment)
        # Combine invariance and friction into single adjustment
        invariance_adj = features['invariance_score']
        friction_adj = 1 / (1 + features['execution_friction'])
        
        features['position_adjustment'] = (invariance_adj * friction_adj).fillna(0.5)
        
        if verbose:
            tprint_success(f"✅ Generated {len(features.columns)} minimal Layer 4 features:")
            for col in features.columns:
                tprint_info(f"   - {col}")
        
        return features
        
    except Exception as e:
        if verbose:
            tprint_warning(f"⚠️ Minimal Layer 4 features failed: {e}")
        return pd.DataFrame(index=df.index)

# Convenience function for Layer 3
def quick_minimal_layer3_features(
    df: pd.DataFrame,
    base_model_cols: List[str],
    target_col: str
) -> pd.DataFrame:
    """
    Quick minimal Layer 3 features with default settings.
    
    Args:
        df: DataFrame with base model predictions
        base_model_cols: List of base model column names
        target_col: Target column name
        
    Returns:
        DataFrame with 3-4 minimal causal features
    """
    return generate_minimal_layer3_features(
        df=df,
        base_model_cols=base_model_cols,
        target_col=target_col,
        verbose=False
    )

# Convenience function for Layer 4
def quick_minimal_layer4_features(
    df: pd.DataFrame,
    base_model_cols: List[str],
    target_col: str
) -> pd.DataFrame:
    """
    Quick minimal Layer 4 features with default settings.
    
    Args:
        df: DataFrame with features and predictions
        base_model_cols: List of base model column names
        target_col: Target column name
        
    Returns:
        DataFrame with 2-3 minimal causal features
    """
    return generate_minimal_layer4_features(
        df=df,
        base_model_cols=base_model_cols,
        target_col=target_col,
        verbose=False
    )
