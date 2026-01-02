"""
Layer 4 Minimal Causal Features Integration Template

Template for integrating minimal causal features into Layer 4 position sizing.
Add this to your position sizing module to include causal risk features.

Usage:
    from .layer4_minimal_integration import add_minimal_layer4_features
    
    # In your position sizing function:
    position_data = add_minimal_layer4_features(
        df=position_data,
        base_model_cols=model_predictions,
        target_col='returns',
        market_data=ohlcv_data,
        volume_data=volume_series
    )
"""

import pandas as pd
from typing import List, Dict, Optional, Any

def add_minimal_layer4_features(
    df: pd.DataFrame,
    base_model_cols: List[str],
    target_col: str,
    market_data: Optional[pd.DataFrame] = None,
    volume_data: Optional[pd.Series] = None,
    config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Add minimal Layer 4 causal features to existing position sizing data.
    
    This is ADDITIVE - keeps all existing features and adds 2-3 causal features.
    
    Args:
        df: Existing position sizing DataFrame
        base_model_cols: List of base model prediction columns
        target_col: Target column name (e.g., 'returns')
        market_data: OHLCV market data
        volume_data: Volume data series
        config: Configuration dictionary
        
    Returns:
        DataFrame with existing features + 2-3 new causal features
    """
    try:
        from .minimal_causal_features import generate_minimal_layer4_features
        
        # Default configuration
        if config is None:
            config = {
                'layer4_minimal_causal_enabled': True,
                'invariance_window': 50,
                'execution_window': 10
            }
        
        # Check if enabled
        if not config.get('layer4_minimal_causal_enabled', True):
            return df
        
        # Generate minimal causal features (2-3 features only)
        minimal_causal_features = generate_minimal_layer4_features(
            df=df,
            base_model_cols=base_model_cols,
            target_col=target_col,
            market_data=market_data,
            volume_data=volume_data,
            invariance_window=config.get('invariance_window', 50),
            execution_window=config.get('execution_window', 10),
            verbose=True
        )
        
        # Add causal features to existing DataFrame (ADDITIVE)
        original_cols = len(df.columns)
        causal_cols_added = []
        
        for col in minimal_causal_features.columns:
            if col not in df.columns:
                df[col] = minimal_causal_features[col]
                causal_cols_added.append(col)
        
        print(f"✅ Added {len(causal_cols_added)} minimal Layer 4 causal features")
        print(f"   - New total: {len(df.columns)} features (+{len(causal_cols_added)} causal)")
        print(f"   - Causal features: {', '.join(causal_cols_added)}")
        
        return df
        
    except Exception as e:
        print(f"⚠️ Layer 4 minimal causal features failed: {e}")
        return df

# Quick integration function
def quick_add_layer4_causal(df, base_model_cols, target_col):
    """
    Quick integration with default settings.
    
    Args:
        df: DataFrame with features
        base_model_cols: List of base model columns
        target_col: Target column name
        
    Returns:
        DataFrame with added causal features
    """
    return add_minimal_layer4_features(
        df=df,
        base_model_cols=base_model_cols,
        target_col=target_col,
        config={'layer4_minimal_causal_enabled': True}
    )

# Example usage in position sizing:
"""
# In your position sizing module:
from .layer4_minimal_integration import add_minimal_layer4_features

def calculate_position_sizing(df, model_predictions, returns_col='returns'):
    # Your existing position sizing logic...
    
    # Add minimal causal features (ADDITIVE)
    df_with_causal = add_minimal_layer4_features(
        df=df,
        base_model_cols=model_predictions,
        target_col=returns_col,
        market_data=ohlcv_data,
        volume_data=volume_data
    )
    
    # Now df_with_causal has:
    # - All your existing position sizing features
    # - Plus 2-3 causal features: invariance_score, execution_friction, position_adjustment
    
    # Use causal features in your position sizing logic:
    position_adjustment = df_with_causal['position_adjustment']
    invariance_score = df_with_causal['invariance_score']
    execution_friction = df_with_causal['execution_friction']
    
    # Adjust position sizes based on causal insights:
    base_position_size = calculate_base_position(df_with_causal)
    adjusted_position = base_position_size * position_adjustment
    
    return adjusted_position
"""
