"""
Layer-12: Model-Ready Feature Output
=====================================

This module packages validated Layer-11 features into a structured format
ready for ML model consumption.

Outputs:
- X: Feature matrix (Tier-weighted Layer-12 features)
- W: Tier-weight matrix (Tier-1/Tier-2 contributions)
- R: Regime information matrix (volatility/liquidity/trend states)
- y: Target labels

Quality Metrics Tracked:
- CI_score, PSR, IC, IC_IR, DSR, SPA_p, Dir_consistency
- Event counts, correlation, parent contribution ratio
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging

try:
    from src.utils.tprint import tprint_info, tprint_warning, tprint_success, tprint_error
except ImportError:
    def tprint_info(msg): print(f"ℹ️ {msg}")
    def tprint_warning(msg): print(f"⚠️ {msg}")
    def tprint_success(msg): print(f"✅ {msg}")
    def tprint_error(msg): print(f"❌ {msg}")

logger = logging.getLogger(__name__)


# =============================================================================
# QUALITY THRESHOLDS
# =============================================================================

QUALITY_THRESHOLDS = {
    'CI_score': {'min': 0.0, 'preferred': 0.05},
    'PSR': {'min': 0.5, 'max': 0.9},
    'IC': {'min': 0.05, 'preferred': 0.15},
    'IC_IR': {'min': 0.5, 'preferred': 1.0},
    'Dir_consistency': {'min': 0.6, 'preferred': 0.9},
    'DSR': {'min': 0.5, 'preferred': 0.8},
    'SPA_p': {'max': 0.1, 'preferred': 0.05},
    'min_events': {'tier1': 200, 'tier2': 100},
    'max_correlation': 0.95,
    'parent_contribution': {'min': 0.3, 'preferred': 0.5}
}

# Tier-weight thresholds (z-score based)
TIER_THRESHOLDS = {
    'tier1_z': 3.0,      # z > 3σ → Tier-1 (weight ~1.0)
    'tier2_z': 2.7,      # 2.7σ < z ≤ 3σ → Tier-2 (weight ~0.5)
    'tier1_quantile': 0.995,
    'tier2_quantile': 0.99
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Layer12Output:
    """
    Complete Layer-12 output structure for ML model consumption.
    
    Attributes:
        X: Feature matrix (time × features), tier-weighted
        W: Tier-weight matrix (time × features), values in [0, 1]
        R: Regime info matrix (time × regime_features)
        y: Target labels (time × 1)
        feature_metadata: Quality metrics per feature
        regime_metadata: Regime definitions
    """
    X: pd.DataFrame
    W: pd.DataFrame
    R: pd.DataFrame
    y: pd.Series
    feature_metadata: Dict[str, Dict[str, float]] = field(default_factory=dict)
    regime_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_weighted_features(self, normalize: bool = True) -> pd.DataFrame:
        """
        Compute X_final = X * W (element-wise).
        Optionally normalize before multiplication.
        """
        X_norm = self.X
        if normalize:
            X_norm = (self.X - self.X.mean()) / (self.X.std() + 1e-9)
        return X_norm * self.W
    
    def get_model_ready_data(self, include_regime: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get final model-ready feature matrix.
        
        Returns:
            X_final: Combined features (X*W + R if include_regime)
            y: Target labels
        """
        X_weighted = self.get_weighted_features(normalize=True)
        
        if include_regime and not self.R.empty:
            # Concatenate regime features
            X_final = pd.concat([X_weighted, self.R], axis=1)
        else:
            X_final = X_weighted
            
        return X_final, self.y
    
    def summary(self) -> str:
        """Generate summary report."""
        lines = [
            "=" * 60,
            "📦 LAYER-12 OUTPUT SUMMARY",
            "=" * 60,
            f"Features (X): {self.X.shape[1]} columns × {len(self.X)} rows",
            f"Tier Weights (W): {self.W.shape[1]} columns",
            f"Regime Info (R): {self.R.shape[1]} columns",
            f"Target (y): {len(self.y)} samples, {self.y.nunique()} classes",
            "",
            "📊 Feature Quality Summary:",
        ]
        
        if self.feature_metadata:
            # Aggregate metrics
            ics = [m.get('IC', 0) for m in self.feature_metadata.values()]
            dsrs = [m.get('DSR', 0) for m in self.feature_metadata.values()]
            lines.append(f"   Mean IC: {np.mean(ics):.4f}")
            lines.append(f"   Mean DSR: {np.mean(dsrs):.4f}")
            lines.append(f"   Features passing all gates: {sum(1 for m in self.feature_metadata.values() if m.get('passed', False))}")
        
        return "\n".join(lines)


# =============================================================================
# TIER-WEIGHT COMPUTATION
# =============================================================================

def compute_tier_weights(
    feature_values: pd.Series,
    method: str = 'zscore',
    adaptive_window: int = 100
) -> pd.Series:
    """
    Compute tier-weights for a feature based on signal extremeness.
    
    Args:
        feature_values: Raw feature values
        method: 'zscore' or 'quantile'
        adaptive_window: Rolling window for adaptive z-score
        
    Returns:
        Tier-weights in [0, 1]
    """
    weights = pd.Series(0.0, index=feature_values.index)
    
    if method == 'zscore':
        # Compute rolling z-score
        rolling_mean = feature_values.rolling(adaptive_window, min_periods=10).mean()
        rolling_std = feature_values.rolling(adaptive_window, min_periods=10).std()
        z_score = (feature_values - rolling_mean) / (rolling_std + 1e-9)
        
        # Apply tier thresholds
        tier1_mask = z_score.abs() >= TIER_THRESHOLDS['tier1_z']
        tier2_mask = (z_score.abs() >= TIER_THRESHOLDS['tier2_z']) & ~tier1_mask
        
        weights[tier1_mask] = 1.0
        weights[tier2_mask] = 0.5
        
    elif method == 'quantile':
        # Rolling quantile approach
        rolling_quantile_high = feature_values.rolling(adaptive_window).quantile(TIER_THRESHOLDS['tier1_quantile'])
        rolling_quantile_low = feature_values.rolling(adaptive_window).quantile(1 - TIER_THRESHOLDS['tier1_quantile'])
        
        tier1_mask = (feature_values >= rolling_quantile_high) | (feature_values <= rolling_quantile_low)
        
        rolling_q2_high = feature_values.rolling(adaptive_window).quantile(TIER_THRESHOLDS['tier2_quantile'])
        rolling_q2_low = feature_values.rolling(adaptive_window).quantile(1 - TIER_THRESHOLDS['tier2_quantile'])
        tier2_mask = ((feature_values >= rolling_q2_high) | (feature_values <= rolling_q2_low)) & ~tier1_mask
        
        weights[tier1_mask] = 1.0
        weights[tier2_mask] = 0.5
    
    return weights.fillna(0.0)


# =============================================================================
# REGIME EXTRACTION
# =============================================================================

def extract_regime_features(
    df: pd.DataFrame,
    volatility_window: int = 20,
    liquidity_window: int = 50,
    trend_window: int = 20
) -> pd.DataFrame:
    """
    Extract regime features from market data.
    
    Args:
        df: OHLCV dataframe
        volatility_window: Window for volatility regime
        liquidity_window: Window for liquidity state
        trend_window: Window for trend direction
        
    Returns:
        DataFrame with regime features
    """
    regime = pd.DataFrame(index=df.index)
    
    # 1. Volatility Regime (0=low, 1=medium, 2=high)
    if 'close' in df.columns:
        returns = df['close'].pct_change()
        rolling_vol = returns.rolling(volatility_window).std()
        vol_percentile = rolling_vol.rolling(250).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        
        regime['VOLATILITY_REGIME'] = pd.cut(
            vol_percentile,
            bins=[-np.inf, 0.33, 0.66, np.inf],
            labels=[0, 1, 2]
        ).astype(float).fillna(1)
    
    # 2. Liquidity State (normalized 0-1)
    if 'volume' in df.columns:
        rolling_vol_mean = df['volume'].rolling(liquidity_window).mean()
        rolling_vol_std = df['volume'].rolling(liquidity_window).std()
        liq_z = (df['volume'] - rolling_vol_mean) / (rolling_vol_std + 1e-9)
        # Normalize to 0-1 using sigmoid
        regime['LIQUIDITY_STATE'] = 1 / (1 + np.exp(-liq_z))
    
    # 3. Trend Direction (-1, 0, +1)
    if 'close' in df.columns:
        sma_short = df['close'].rolling(trend_window // 2).mean()
        sma_long = df['close'].rolling(trend_window).mean()
        trend_diff = sma_short - sma_long
        trend_std = trend_diff.rolling(trend_window * 2).std()
        trend_z = trend_diff / (trend_std + 1e-9)
        
        regime['TREND_DIRECTION'] = np.where(trend_z > 1, 1, np.where(trend_z < -1, -1, 0))
    
    # 4. Time-of-Day (optional, for intraday)
    if hasattr(df.index, 'hour'):
        regime['TIME_OF_DAY'] = df.index.hour / 24.0
    
    return regime.fillna(0)


# =============================================================================
# FEATURE QUALITY ASSESSMENT
# =============================================================================

def assess_feature_quality(
    feature_name: str,
    feature_values: pd.Series,
    target: pd.Series,
    tier_weights: pd.Series
) -> Dict[str, float]:
    """
    Compute quality metrics for a single feature.
    
    Returns:
        Dictionary of quality metrics
    """
    metrics = {'feature_name': feature_name}
    
    try:
        # Align data
        common_idx = feature_values.dropna().index.intersection(target.dropna().index)
        x = feature_values.loc[common_idx]
        y = target.loc[common_idx]
        w = tier_weights.loc[common_idx] if tier_weights is not None else None
        
        if len(x) < 50:
            metrics['passed'] = False
            metrics['reason'] = 'Insufficient samples'
            return metrics
        
        # IC (Spearman correlation)
        metrics['IC'] = x.corr(y, method='spearman')
        
        # IC_IR (IC stability)
        rolling_ic = x.rolling(100).corr(y)
        ic_mean = rolling_ic.mean()
        ic_std = rolling_ic.std()
        metrics['IC_IR'] = ic_mean / (ic_std + 1e-9) if ic_std > 0 else 0.0
        
        # Directional Consistency
        x_sign = np.sign(x.diff().fillna(0))
        y_sign = np.sign(y.diff().fillna(0))
        metrics['Dir_consistency'] = (x_sign == y_sign).mean()
        
        # Tier-1 / Tier-2 event counts
        if w is not None:
            metrics['tier1_events'] = (w >= 0.9).sum()
            metrics['tier2_events'] = ((w >= 0.4) & (w < 0.9)).sum()
        
        # Quality gate check
        passed = (
            metrics.get('IC', 0) >= QUALITY_THRESHOLDS['IC']['min'] and
            metrics.get('IC_IR', 0) >= QUALITY_THRESHOLDS['IC_IR']['min'] and
            metrics.get('Dir_consistency', 0) >= QUALITY_THRESHOLDS['Dir_consistency']['min']
        )
        metrics['passed'] = passed
        
    except Exception as e:
        metrics['passed'] = False
        metrics['error'] = str(e)
    
    return metrics


# =============================================================================
# MAIN LAYER-12 BUILDER
# =============================================================================

class Layer12Builder:
    """
    Builds Layer-12 Model-Ready Feature Output from validated candidates.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        target: pd.Series,
        verbose: bool = True
    ):
        """
        Initialize Layer-12 builder.
        
        Args:
            df: Market data (OHLCV)
            target: Target labels
            verbose: Enable logging
        """
        self.df = df
        self.target = target
        self.verbose = verbose
        
    def build(
        self,
        candidates: List[Dict],
        max_features: int = 100,
        max_correlation: float = 0.95,
        include_regime: bool = True
    ) -> Layer12Output:
        """
        Build Layer-12 output from validated candidates.
        
        Args:
            candidates: List of validated feature candidates
            max_features: Maximum features to include
            max_correlation: Maximum allowed pairwise correlation
            include_regime: Include regime features
            
        Returns:
            Layer12Output with X, W, R, y
        """
        if self.verbose:
            tprint_info(f"🏗️ Building Layer-12 Output from {len(candidates)} candidates...")
        
        # 1. Extract feature vectors and compute tier-weights
        feature_data = {}
        weight_data = {}
        metadata = {}
        
        for cand in candidates:
            try:
                family = cand.get('family', 'UNKNOWN')
                weight_vector = cand.get('weight_vector')
                
                if weight_vector is None or len(weight_vector) == 0:
                    continue
                
                # Ensure alignment with df index
                if isinstance(weight_vector, pd.Series):
                    aligned = weight_vector.reindex(self.df.index).fillna(0)
                else:
                    aligned = pd.Series(weight_vector, index=self.df.index[:len(weight_vector)]).reindex(self.df.index).fillna(0)
                
                # Compute tier-weights
                tier_weights = compute_tier_weights(aligned, method='zscore')
                
                # Assess quality
                quality = assess_feature_quality(family, aligned, self.target, tier_weights)
                
                if quality.get('passed', False):
                    feature_data[family] = aligned
                    weight_data[family] = tier_weights
                    metadata[family] = quality
                    
            except Exception as e:
                if self.verbose:
                    tprint_warning(f"   ⚠️ Failed to process {cand.get('family', 'UNKNOWN')}: {e}")
                continue
        
        if self.verbose:
            tprint_info(f"   ✅ {len(feature_data)} features passed quality gates")
        
        # 2. Correlation-based filtering
        if len(feature_data) > 1:
            feature_data, weight_data, metadata = self._filter_correlated(
                feature_data, weight_data, metadata, max_correlation
            )
            if self.verbose:
                tprint_info(f"   ✅ {len(feature_data)} features after correlation filter")
        
        # 3. Select top features by IC
        if len(feature_data) > max_features:
            sorted_features = sorted(
                metadata.items(),
                key=lambda x: abs(x[1].get('IC', 0)),
                reverse=True
            )[:max_features]
            selected = set(f[0] for f in sorted_features)
            feature_data = {k: v for k, v in feature_data.items() if k in selected}
            weight_data = {k: v for k, v in weight_data.items() if k in selected}
            metadata = {k: v for k, v in metadata.items() if k in selected}
            if self.verbose:
                tprint_info(f"   ✅ Selected top {max_features} features by IC")
        
        # 4. Build matrices
        X = pd.DataFrame(feature_data)
        W = pd.DataFrame(weight_data)
        
        # 5. Extract regime features
        R = pd.DataFrame()
        if include_regime:
            R = extract_regime_features(self.df)
            if self.verbose:
                tprint_info(f"   ✅ Extracted {R.shape[1]} regime features")
        
        # 6. Align target
        y = self.target.reindex(X.index).fillna(0)
        
        # 7. Build output
        output = Layer12Output(
            X=X,
            W=W,
            R=R,
            y=y,
            feature_metadata=metadata,
            regime_metadata={'volatility_window': 20, 'liquidity_window': 50, 'trend_window': 20}
        )
        
        if self.verbose:
            tprint_success(f"✅ Layer-12 Output Built: {X.shape[1]} features × {len(X)} samples")
        
        return output
    
    def _filter_correlated(
        self,
        feature_data: Dict[str, pd.Series],
        weight_data: Dict[str, pd.Series],
        metadata: Dict[str, Dict],
        max_corr: float
    ) -> Tuple[Dict, Dict, Dict]:
        """Remove features with correlation > max_corr."""
        df = pd.DataFrame(feature_data)
        corr_matrix = df.corr().abs()
        
        # Get upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation > threshold
        to_drop = set()
        for col in upper.columns:
            correlated = upper.index[(upper[col] > max_corr)].tolist()
            for c in correlated:
                # Keep the one with higher IC
                ic_col = abs(metadata.get(col, {}).get('IC', 0))
                ic_c = abs(metadata.get(c, {}).get('IC', 0))
                if ic_col >= ic_c:
                    to_drop.add(c)
                else:
                    to_drop.add(col)
        
        # Filter
        feature_data = {k: v for k, v in feature_data.items() if k not in to_drop}
        weight_data = {k: v for k, v in weight_data.items() if k not in to_drop}
        metadata = {k: v for k, v in metadata.items() if k not in to_drop}
        
        return feature_data, weight_data, metadata


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def build_layer12_output(
    df: pd.DataFrame,
    candidates: List[Dict],
    target: pd.Series,
    max_features: int = 100,
    verbose: bool = True
) -> Layer12Output:
    """
    Convenience function to build Layer-12 output.
    
    Args:
        df: Market data
        candidates: Validated feature candidates
        target: Target labels
        max_features: Maximum features to include
        verbose: Enable logging
        
    Returns:
        Layer12Output ready for ML model
    """
    builder = Layer12Builder(df, target, verbose=verbose)
    return builder.build(candidates, max_features=max_features)
