
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from scipy.stats import variation

class RegimeAssessor:
    """
    Assesses the quality and distinctness of identified market regimes.
    Used for the GMM 2.0 Feedback Loop (e.g., retrain if regimes are indistinct).
    """
    
    def __init__(self):
        pass
        
    def assess_regimes(
        self, 
        regime_features: pd.DataFrame, 
        returns: pd.Series
    ) -> Dict[str, Any]:
        """
        Calculate metrics for regime quality.
        
        Args:
            regime_features: DataFrame with REGIME_x probability columns
            returns: Series of returns (target)
            
        Returns:
            Dictionary with assessment metrics
        """
        if regime_features.empty or returns.empty:
            return {"quality_score": 0.0, "status": "empty"}
            
        # Align data
        common_idx = regime_features.index.intersection(returns.index)
        if len(common_idx) < 100:
             return {"quality_score": 0.0, "status": "insufficient_data"}
             
        df_regime = regime_features.loc[common_idx]
        y = returns.loc[common_idx]
        
        # Hard assignment for evaluation
        regime_cols = [c for c in df_regime.columns if c.startswith('REGIME_') and not c.endswith('_VELOCITY') and not c.endswith('_INTEGRITY')]
        if not regime_cols:
             return {"quality_score": 0.0, "status": "no_regime_cols"}
             
        assigned_regime = df_regime[regime_cols].idxmax(axis=1)
        
        # Metrics per regime
        sharpes = []
        kellys = []
        tail_risks = []
        counts = []
        vol_adj_corrs = []
        entropies = []
        
        regime_stats = {}
        
        # Global Volatility for reference
        global_vol = y.std() if len(y) > 1 else 0.0
        
        for r_col in regime_cols:
            mask = (assigned_regime == r_col)
            r_returns = y[mask]
            
            count = len(r_returns)
            counts.append(count)
            
            if count > 10:
                mean_ret = r_returns.mean()
                vol = r_returns.std()
                
                # Check for zero vol
                if vol < 1e-9:
                    sharpe = 0.0
                    kelly = 0.0
                    tail_risk = 0.0
                else:
                    # Annualized metrics (assuming 15m default, but scale invariant-ish for ratios)
                    # Used 15m=96 bars/day * 252
                    scale = 252 * 96 
                    sharpe = (mean_ret / vol) * np.sqrt(scale)
                    
                    # Kelly Criterion (Simple approximation: Mean/Var)
                    # f = mu / sigma^2
                    kelly = mean_ret / (vol ** 2)
                    
                    # Tail Risk: CVaR (Expected Shortfall) at 5%
                    tail_risk = r_returns[r_returns <= r_returns.quantile(0.05)].mean()
                    if np.isnan(tail_risk): tail_risk = 0.0
                
                # Vol-adjusted P&L correlation
                # Correlation of local returns with global returns (proxied or filtered)
                # Since r_returns is a subset, we can't correlate with 'global' directly easily row-wise
                # without the full index.
                # Proxy: Correlation of (Return / RegimeVol) vs Regime Prob?
                # User asks: "Vol-adjusted P&L correlation"
                # Interpret as: Correlation between Regime Probability and Vol-Adjusted Returns?
                # Or correlation of returns within regime?
                # Interpret: Auto-correlation of returns in this regime?
                # Let's use: Autocorrelation of returns (lag 1)
                ac1 = r_returns.autocorr(lag=1)
                vol_adj_corr = ac1 if not np.isnan(ac1) else 0.0
                
                # Regime-level entropy of returns (Histogram entropy)
                hist, bins = np.histogram(r_returns, bins='auto', density=True)
                # Normalize probabilities
                p_hist = hist / hist.sum()
                r_entropy = -np.sum(p_hist * np.log(p_hist + 1e-9))
                
            else:
                mean_ret = 0.0
                vol = 0.0
                sharpe = 0.0
                kelly = 0.0
                tail_risk = 0.0
                vol_adj_corr = 0.0
                r_entropy = 0.0
                
            sharpes.append(sharpe)
            kellys.append(kelly)
            tail_risks.append(tail_risk)
            vol_adj_corrs.append(vol_adj_corr)
            entropies.append(r_entropy)
            
            regime_stats[r_col] = {
                "count": count,
                "mean_return": mean_ret,
                "volatility": vol,
                "sharpe": sharpe,
                "kelly": kelly,
                "tail_risk": tail_risk,
                "vol_adj_corr": vol_adj_corr,
                "return_entropy": r_entropy
            }
            
        # Overall assessment - Coefficient of Variation (CV) between regimes
        # Used to measure distinctness.
        
        def calculate_cv(values):
            arr = np.array(values)
            if len(arr) <= 1: return 0.0
            mean_val = np.mean(arr)
            if abs(mean_val) < 1e-9: return 0.0 # Avoid div by zero
            return np.std(arr) / abs(mean_val)

        sharpe_cv = calculate_cv(sharpes)
        tail_cv = calculate_cv(tail_risks)
        entropy_cv = calculate_cv(entropies)
        
        # 2. Entropy of counts (Balance)
        probs = np.array(counts) / (sum(counts) + 1e-9)
        regime_entropy = -np.sum(probs * np.log(probs + 1e-9))
        
        # Decision
        quality_score = sharpe_cv  # Primary metric
        needs_retrain = abs(sharpe_cv) < 0.3 # Feedback loop trigger
        
        return {
            "regime_stats": regime_stats,
            "metrics_cv": {
                "sharpe_cv": sharpe_cv,
                "tail_risk_cv": tail_cv,
                "entropy_cv": entropy_cv
            },
            "sharpe_cv": sharpe_cv, # Top level for easy access
            "regime_entropy": regime_entropy,
            "quality_score": quality_score,
            "needs_retrain": needs_retrain
        }
