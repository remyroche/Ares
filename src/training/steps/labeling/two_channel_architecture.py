"""
Two-Channel Model Architecture

Implements a two-channel feature architecture where:
- Signal Channel: Predictor + Trigger features (alpha generation)
- Context Channel: Context + Regime features (state conditioning)
- Interactions: signal × context cross-products (conditional activation)

This architecture allows:
1. Different signal types to have appropriate representation
2. Context features to gate/modulate signal predictions
3. Explicit interactions between signals and market state
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

try:
    from src.training.steps.labeling.causal_quality_assessment import SignalRole
except ImportError:
    class SignalRole(Enum):
        PREDICTOR = "predictor"
        TRIGGER = "trigger"
        INTERACTION = "interaction"
        CONTEXT = "context"

try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning
except ImportError:
    def tprint_info(msg): print(f"INFO: {msg}")
    def tprint_success(msg): print(f"SUCCESS: {msg}")
    def tprint_warning(msg): print(f"WARNING: {msg}")


@dataclass
class TwoChannelFeatures:
    """Container for two-channel feature separation."""
    signal_features: pd.DataFrame  # Predictors + Triggers
    context_features: pd.DataFrame  # Context + Regime
    interaction_features: pd.DataFrame  # signal × context
    feature_roles: Dict[str, SignalRole]  # Feature -> Role mapping
    
    @property
    def all_features(self) -> pd.DataFrame:
        """Combine all channels into single DataFrame."""
        return pd.concat([
            self.signal_features,
            self.context_features,
            self.interaction_features
        ], axis=1)
    
    @property
    def n_signal(self) -> int:
        return len(self.signal_features.columns)
    
    @property
    def n_context(self) -> int:
        return len(self.context_features.columns)
    
    @property
    def n_interaction(self) -> int:
        return len(self.interaction_features.columns)


class TwoChannelFeatureManager:
    """
    Manages two-channel feature architecture for meta-labeling models.
    
    Architecture:
    ┌────────────────────┐     ┌────────────────────┐
    │   Signal Channel   │     │  Context Channel   │
    │  (Predictors +     │     │  (Regime +         │
    │   Triggers)        │     │   State Features)  │
    └─────────┬──────────┘     └─────────┬──────────┘
              │                          │
              └──────────┬───────────────┘
                         │
              ┌──────────▼──────────┐
              │    Interactions     │
              │  signal × context   │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │   Combined Input    │
              │    to LightGBM      │
              └─────────────────────┘
    """
    
    # Feature name patterns for role detection
    SIGNAL_PATTERNS = [
        'surprise', 'flow', 'momentum', 'slope', 'curvature',
        'relaxation', 'decay', 'fragility', 'resilience', 'kyle',
        'pressure', 'imbalance', 'return', 'price', 'log_ratio',
        'residual', 'predictor', 'trigger', 'spike', 'shock',
        'causal_surprise', 'specialist', 'composite'
    ]
    
    CONTEXT_PATTERNS = [
        'regime', 'vol_regime', 'trend_regime', 'rv_z', 'vol_ratio',
        'volume_z', 'spread_proxy', 'trend_slope_z', 'trend_strength',
        'drawdown', 'hurst', 'vpin', 'rel_vol', 'time_since',
        'dist_from_vwap', 'vol_imbalance', 'context', 'state'
    ]
    
    def __init__(
        self,
        max_interactions: int = 50,
        interaction_top_k: int = 10,
        min_variance: float = 1e-6,
        verbose: bool = True
    ):
        """
        Initialize two-channel feature manager.
        
        Args:
            max_interactions: Maximum number of interaction features to generate
            interaction_top_k: Number of top features from each channel for interactions
            min_variance: Minimum variance to keep a feature
            verbose: Whether to print progress
        """
        self.max_interactions = max_interactions
        self.interaction_top_k = interaction_top_k
        self.min_variance = min_variance
        self.verbose = verbose
        self._feature_roles: Dict[str, SignalRole] = {}
    
    def process_features(
        self, 
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        generate_interactions: bool = True
    ) -> TwoChannelFeatures:
        """
        Process features into two-channel architecture.
        
        Args:
            X: Input feature DataFrame
            y: Optional target for feature importance ranking
            generate_interactions: Whether to generate signal×context interactions
            
        Returns:
            TwoChannelFeatures with separated channels
        """
        if self.verbose:
            tprint_info("🔀 Processing features into two-channel architecture...")
        
        # 1. Classify features by role
        self._classify_features(X)
        
        # 2. Separate into channels
        signal_cols = [c for c, r in self._feature_roles.items() 
                       if r in [SignalRole.PREDICTOR, SignalRole.TRIGGER, SignalRole.INTERACTION]]
        context_cols = [c for c, r in self._feature_roles.items() 
                        if r == SignalRole.CONTEXT]
        
        # Ensure columns exist
        signal_cols = [c for c in signal_cols if c in X.columns]
        context_cols = [c for c in context_cols if c in X.columns]
        
        signal_df = X[signal_cols] if signal_cols else pd.DataFrame(index=X.index)
        context_df = X[context_cols] if context_cols else pd.DataFrame(index=X.index)
        
        if self.verbose:
            tprint_info(f"   📊 Signal channel: {len(signal_cols)} features")
            tprint_info(f"   📊 Context channel: {len(context_cols)} features")
        
        # 3. Generate interactions
        if generate_interactions and len(signal_cols) > 0 and len(context_cols) > 0:
            interaction_df = self._generate_interactions(
                signal_df, context_df, y
            )
        else:
            interaction_df = pd.DataFrame(index=X.index)
        
        if self.verbose:
            tprint_success(f"   ✅ Two-channel processing complete: "
                          f"{len(signal_cols)}S + {len(context_cols)}C + "
                          f"{len(interaction_df.columns)}I = "
                          f"{len(signal_cols) + len(context_cols) + len(interaction_df.columns)} total")
        
        return TwoChannelFeatures(
            signal_features=signal_df,
            context_features=context_df,
            interaction_features=interaction_df,
            feature_roles=self._feature_roles.copy()
        )
    
    def _classify_features(self, X: pd.DataFrame) -> None:
        """Classify each feature by its role."""
        for col in X.columns:
            col_lower = col.lower()
            
            # Check context patterns first (more specific)
            if any(pat in col_lower for pat in self.CONTEXT_PATTERNS):
                self._feature_roles[col] = SignalRole.CONTEXT
            # Then signal patterns
            elif any(pat in col_lower for pat in self.SIGNAL_PATTERNS):
                # Distinguish between predictor and trigger
                if any(t in col_lower for t in ['trigger', 'spike', 'shock', 'event']):
                    self._feature_roles[col] = SignalRole.TRIGGER
                elif 'composite' in col_lower or '_int' in col_lower:
                    self._feature_roles[col] = SignalRole.INTERACTION
                else:
                    self._feature_roles[col] = SignalRole.PREDICTOR
            else:
                # Default to predictor for unknown features
                self._feature_roles[col] = SignalRole.PREDICTOR
    
    def _generate_interactions(
        self,
        signal_df: pd.DataFrame,
        context_df: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Generate signal × context interaction features.
        
        Uses top-k features from each channel based on variance or correlation with target.
        """
        if self.verbose:
            tprint_info("   🔗 Generating signal×context interactions...")
        
        # Select top features from each channel
        if y is not None and len(y) == len(signal_df):
            # Rank by absolute correlation with target
            signal_scores = signal_df.corrwith(y).abs().fillna(0)
            context_scores = context_df.corrwith(y).abs().fillna(0)
        else:
            # Rank by variance
            signal_scores = signal_df.var().fillna(0)
            context_scores = context_df.var().fillna(0)
        
        top_signals = signal_scores.nlargest(min(self.interaction_top_k, len(signal_scores))).index.tolist()
        top_contexts = context_scores.nlargest(min(self.interaction_top_k, len(context_scores))).index.tolist()
        
        # Generate interactions (multiplicative)
        interactions = {}
        n_generated = 0
        
        for sig_col in top_signals:
            for ctx_col in top_contexts:
                if n_generated >= self.max_interactions:
                    break
                    
                interaction_name = f"INT_{sig_col[:15]}×{ctx_col[:15]}"
                sig_vals = signal_df[sig_col].fillna(0)
                ctx_vals = context_df[ctx_col].fillna(0)
                
                # Multiplicative interaction (both normalized to [-1, 1] range)
                sig_norm = np.clip(sig_vals / (sig_vals.abs().max() + 1e-9), -1, 1)
                ctx_norm = np.clip(ctx_vals / (ctx_vals.abs().max() + 1e-9), -1, 1)
                
                interaction = sig_norm * ctx_norm
                
                # Only keep if has variance
                if interaction.var() > self.min_variance:
                    interactions[interaction_name] = interaction
                    n_generated += 1
            
            if n_generated >= self.max_interactions:
                break
        
        if self.verbose:
            tprint_info(f"   📊 Generated {len(interactions)} interaction features")
        
        return pd.DataFrame(interactions, index=signal_df.index)
    
    def get_feature_importance_by_channel(
        self,
        feature_importance: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Break down feature importance by channel.
        
        Args:
            feature_importance: Feature name -> importance score
            
        Returns:
            Dict with keys 'signal', 'context', 'interaction' containing feature importances
        """
        result = {
            'signal': {},
            'context': {},
            'interaction': {}
        }
        
        for feat, importance in feature_importance.items():
            if feat.startswith('INT_'):
                result['interaction'][feat] = importance
            elif feat in self._feature_roles:
                role = self._feature_roles[feat]
                if role == SignalRole.CONTEXT:
                    result['context'][feat] = importance
                else:
                    result['signal'][feat] = importance
            else:
                result['signal'][feat] = importance  # Default to signal
        
        return result
    
    def apply_context_gating(
        self,
        predictions: np.ndarray,
        context_features: pd.DataFrame,
        gating_threshold: float = 0.5
    ) -> np.ndarray:
        """
        Apply context-based gating to predictions.
        
        Reduces prediction confidence in unfavorable context regimes.
        
        Args:
            predictions: Raw model predictions (probabilities)
            context_features: Context feature DataFrame
            gating_threshold: Threshold for gating activation
            
        Returns:
            Gated predictions
        """
        if len(context_features.columns) == 0:
            return predictions
        
        # Simple gating: average context signal determines confidence
        context_signal = context_features.mean(axis=1).values
        
        # Normalize to [0.5, 1.5] range for soft gating
        gate = 0.5 + np.clip(context_signal, -0.5, 0.5) + 0.5
        
        # Apply gate to predictions
        gated = predictions * gate
        
        # Clip to valid probability range
        return np.clip(gated, 0, 1)


def apply_two_channel_architecture(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    max_interactions: int = 50,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Convenience function to apply two-channel architecture to features.
    
    Args:
        X: Input feature DataFrame
        y: Optional target for ranking
        max_interactions: Max number of interactions to generate
        verbose: Print progress
        
    Returns:
        DataFrame with signal + context + interaction features
    """
    manager = TwoChannelFeatureManager(
        max_interactions=max_interactions,
        verbose=verbose
    )
    
    two_channel = manager.process_features(X, y, generate_interactions=True)
    
    return two_channel.all_features
