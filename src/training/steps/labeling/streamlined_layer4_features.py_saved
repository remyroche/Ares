"""
Streamlined Layer 4 Causal Features

Essential causal risk features for Layer 4 position sizing.
Only 6 critical features to maximize benefit with minimum complexity.

Key Features:
1. Invariance score
2. Execution friction
3. Position adjustment
4. Causal risk premium
5. Specialist consensus risk
6. Intervention impact
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
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

class StreamlinedLayer4Features:
    """
    Streamlined Layer 4 causal features with only 6 essential features.
    
    Focuses on maximum causal risk insight with minimum computational overhead.
    """
    
    def __init__(
        self,
        invariance_window: int = 50,
        execution_window: int = 10,
        risk_adjustment_factor: float = 1.0,
        verbose: bool = True
    ):
        """
        Initialize Streamlined Layer 4 Features.
        
        Args:
            invariance_window: Window for invariance statistics
            execution_window: Window for execution features
            risk_adjustment_factor: Factor for risk adjustment
            verbose: Whether to print progress information
        """
        self.invariance_window = invariance_window
        self.execution_window = execution_window
        self.risk_adjustment_factor = risk_adjustment_factor
        self.verbose = verbose
        
        # Storage for computed features
        self.feature_cache_ = {}
        
    def generate_streamlined_features(
        self,
        df: pd.DataFrame,
        base_model_cols: List[str],
        target_col: str,
        market_data: Optional[pd.DataFrame] = None,
        volume_data: Optional[pd.Series] = None,
        specialist_predictions: Optional[Dict[str, np.ndarray]] = None,
        causal_effects: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Generate 6 streamlined Layer 4 causal features.
        
        Args:
            df: DataFrame with features and predictions
            base_model_cols: List of base model column names
            target_col: Target column name
            market_data: Additional market data (OHLCV)
            volume_data: Volume data
            specialist_predictions: Specialist prediction values
            causal_effects: Causal effect estimates
            
        Returns:
            DataFrame with 6 streamlined causal features
        """
        try:
            if self.verbose:
                tprint_info("🎯 Generating Streamlined Layer 4 Causal Features...")
            
            features = pd.DataFrame(index=df.index)
            
            # Feature 1: Invariance score
            features['invariance_score'] = self._compute_invariance_score(
                df, base_model_cols, target_col
            )
            
            # Feature 2: Execution friction
            features['execution_friction'] = self._compute_execution_friction(
                df, market_data, volume_data
            )
            
            # Feature 3: Position adjustment
            features['position_adjustment'] = self._compute_position_adjustment(
                features['invariance_score'], features['execution_friction']
            )
            
            # Feature 4: Causal risk premium
            features['causal_risk_premium'] = self._compute_causal_risk_premium(
                df, base_model_cols, target_col, causal_effects
            )
            
            # Feature 5: Specialist consensus risk
            features['specialist_consensus_risk'] = self._compute_specialist_consensus_risk(
                specialist_predictions, df
            )
            
            # Feature 6: Intervention impact
            features['intervention_impact'] = self._compute_intervention_impact(
                df, market_data
            )
            
            # Cache features
            self.feature_cache_ = features
            
            if self.verbose:
                tprint_success(f"✅ Generated {len(features.columns)} streamlined Layer 4 features:")
                for col in features.columns:
                    tprint_info(f"   - {col}")
            
            return features
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Streamlined Layer 4 feature generation failed: {e}")
            return pd.DataFrame(index=df.index)
    
    def _compute_invariance_score(
        self,
        df: pd.DataFrame,
        base_model_cols: List[str],
        target_col: str
    ) -> pd.Series:
        """Compute invariance score (rolling coefficient stability)."""
        try:
            invariance_scores = []
            
            for base_col in base_model_cols:
                if base_col in df.columns and target_col in df.columns:
                    # Rolling coefficient stability
                    rolling_coefs = []
                    
                    for i in range(self.invariance_window, len(df)):
                        window_data = df.iloc[i-self.invariance_window:i]
                        
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
                        coef_series = pd.Series(rolling_coefs, index=df.index[self.invariance_window:])
                        coef_stability = 1 / (1 + coef_series.rolling(self.invariance_window).std())
                        invariance_scores.append(coef_stability.fillna(0.5))
            
            if invariance_scores:
                return pd.concat(invariance_scores, axis=1).mean(axis=1)
            else:
                return pd.Series(0.5, index=df.index)
                
        except Exception:
            return pd.Series(0.5, index=df.index)
    
    def _compute_execution_friction(
        self,
        df: pd.DataFrame,
        market_data: Optional[pd.DataFrame],
        volume_data: Optional[pd.Series]
    ) -> pd.Series:
        """Compute execution friction (volume volatility × price volatility)."""
        try:
            execution_frictions = []
            
            # Volume volatility
            if volume_data is not None:
                volume_aligned = volume_data.reindex(df.index).fillna(0)
                volume_vol = volume_aligned.rolling(self.execution_window).std()
                execution_frictions.append(volume_vol.fillna(0))
            
            # Price volatility
            if market_data is not None and 'close' in market_data.columns:
                price_data = market_data['close'].reindex(df.index).fillna(method='ffill')
                price_vol = price_data.pct_change().rolling(self.execution_window).std()
                execution_frictions.append(price_vol.fillna(0))
            elif 'volatility' in df.columns:
                price_vol = df['volatility'].rolling(self.execution_window).std()
                execution_frictions.append(price_vol.fillna(0))
            
            if execution_frictions:
                # Combined friction (product of volatilities)
                combined_friction = pd.concat(execution_frictions, axis=1).prod(axis=1)
                return combined_friction
            else:
                return pd.Series(0.01, index=df.index)
                
        except Exception:
            return pd.Series(0.01, index=df.index)
    
    def _compute_position_adjustment(
        self,
        invariance_score: pd.Series,
        execution_friction: pd.Series
    ) -> pd.Series:
        """Compute position adjustment factor."""
        try:
            # Combined adjustment (invariance × inverse_friction)
            invariance_adj = invariance_score.fillna(0.5)
            friction_adj = 1 / (1 + execution_friction.fillna(0.01))
            
            adjustment = invariance_adj * friction_adj * self.risk_adjustment_factor
            
            return adjustment.fillna(0.5)
            
        except Exception:
            return pd.Series(0.5, index=df.index)
    
    def _compute_causal_risk_premium(
        self,
        df: pd.DataFrame,
        base_model_cols: List[str],
        target_col: str,
        causal_effects: Optional[Dict[str, float]]
    ) -> pd.Series:
        """Compute causal risk premium."""
        try:
            if target_col in df.columns and len(base_model_cols) > 0:
                # Compute prediction error
                valid_cols = [col for col in base_model_cols if col in df.columns]
                
                if valid_cols:
                    avg_prediction = df[valid_cols].mean(axis=1)
                    prediction_error = df[target_col] - avg_prediction
                    
                    # Risk premium as absolute error
                    risk_premium = np.abs(prediction_error)
                    
                    # Adjust by causal effects if available
                    if causal_effects:
                        avg_effect = np.mean(list(causal_effects.values()))
                        risk_premium = risk_premium * (1 + abs(avg_effect))
                    
                    return risk_premium.fillna(0.01)
                else:
                    return pd.Series(0.01, index=df.index)
            else:
                return pd.Series(0.01, index=df.index)
                
        except Exception:
            return pd.Series(0.01, index=df.index)
    
    def _compute_specialist_consensus_risk(
        self,
        specialist_predictions: Optional[Dict[str, np.ndarray]],
        df: pd.DataFrame
    ) -> pd.Series:
        """Compute specialist consensus risk."""
        try:
            if specialist_predictions and len(specialist_predictions) >= 2:
                spec_preds_df = pd.DataFrame(specialist_predictions, index=df.index)
                
                # Risk as disagreement (variance)
                pred_variance = spec_preds_df.var(axis=1)
                risk_score = pred_variance
                
                return risk_score.fillna(0.01)
            else:
                return pd.Series(0.01, index=df.index)
                
        except Exception:
            return pd.Series(0.01, index=df.index)
    
    def _compute_intervention_impact(
        self,
        df: pd.DataFrame,
        market_data: Optional[pd.DataFrame]
    ) -> pd.Series:
        """Compute intervention impact."""
        try:
            # Look for intervention-related features in market data
            if market_data is not None:
                intervention_features = []
                
                for col in market_data.columns:
                    if any(keyword in col.lower() for keyword in ['volume', 'spread', 'range']):
                        intervention_features.append(col)
                
                if intervention_features:
                    # Use first intervention feature
                    feature_data = market_data[intervention_features[0]].reindex(df.index).fillna(0)
                    
                    # Impact as absolute change
                    impact = np.abs(feature_data.pct_change())
                    
                    return impact.fillna(0.01)
                else:
                    return pd.Series(0.01, index=df.index)
            else:
                # Fallback to volatility
                if 'volatility' in df.columns:
                    return df['volatility'].fillna(0.01)
                else:
                    return pd.Series(0.01, index=df.index)
                    
        except Exception:
            return pd.Series(0.01, index=df.index)
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of the 6 streamlined features.
        
        Returns:
            Dictionary of feature descriptions
        """
        return {
            'invariance_score': 'Rolling coefficient stability (higher = more stable)',
            'execution_friction': 'Combined volume and price volatility (market impact)',
            'position_adjustment': 'Combined risk adjustment factor',
            'causal_risk_premium': 'Risk premium based on causal effects',
            'specialist_consensus_risk': 'Risk from specialist disagreement',
            'intervention_impact': 'Impact of interventions on market'
        }
    
    def get_risk_adjustment_factors(self) -> Dict[str, float]:
        """
        Get risk adjustment factors for position sizing.
        
        Returns:
            Dictionary of adjustment factors
        """
        return {
            'invariance_weight': 1.0,
            'friction_weight': 1.0,
            'risk_premium_weight': 1.0,
            'consensus_weight': 1.0,
            'intervention_weight': 1.0,
            'overall_adjustment': self.risk_adjustment_factor
        }

# Convenience function
def quick_streamlined_layer4_features(
    df: pd.DataFrame,
    base_model_cols: List[str],
    target_col: str,
    **kwargs
) -> pd.DataFrame:
    """
    Quick streamlined Layer 4 feature generation.
    
    Args:
        df: DataFrame with features and predictions
        base_model_cols: List of base model column names
        target_col: Target column name
        **kwargs: Additional parameters
        
    Returns:
        DataFrame with 6 streamlined causal features
    """
    generator = StreamlinedLayer4Features(**kwargs)
    return generator.generate_streamlined_features(
        df, base_model_cols, target_col
    )
