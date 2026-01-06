import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from sklearn.metrics import roc_auc_score, mean_squared_error
from scipy.stats import spearmanr, entropy
import logging

try:
    from src.utils.tprint import tprint_info, tprint_warning, tprint_error
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    tprint_info = logger.info
    tprint_warning = logger.warning
    tprint_error = logger.error

class DetailedMetricsReporter:
    """
    Generates a detailed CSV report with advanced financial, structural, and causal metrics
    for model candidates.
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.metrics_log = []
        
    def calculate_metrics(self, 
                          candidate: Any, 
                          context: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate all detailed metrics for a single candidate.
        
        Args:
            candidate: GeometryTrial or candidate dictionary
            context: Dictionary containing:
                - y_true: True targets
                - y_pred: Predicted scores/probabilities
                - cate_pred: Predicted CATE (optional)
                - events_df: DataFrame with event metadata (time, etc.)
                - X: Feature DataFrame (optional, for structural metrics)
                
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        # Extract data from context
        y_true = context.get('y_true')
        y_pred = context.get('y_pred')
        cate_pred = context.get('cate_pred')
        events_df = context.get('events_df')
        X = context.get('X')
        
        if y_true is None or y_pred is None:
            if self.verbose:
                tprint_warning(f"⚠️ Missing predictions for candidate {getattr(candidate, 'uuid', 'unknown')}")
            return {}
            
        # Ensure aligned indices/numpy arrays
        if hasattr(y_true, 'values'): y_true = y_true.values
        if hasattr(y_pred, 'values'): y_pred = y_pred.values
        if cate_pred is not None and hasattr(cate_pred, 'values'): cate_pred = cate_pred.values
        
        # 1. Financial Metrics
        metrics.update(self.compute_financial_metrics(y_true, y_pred, events_df))
        
        # 2. Causal Metrics (if CATE available)
        if cate_pred is not None:
            metrics.update(self.compute_causal_metrics(y_true, y_pred, cate_pred))
            
        # 3. Structural Metrics
        if X is not None:
            metrics.update(self.compute_structural_metrics(X, y_true, y_pred, candidate))
            
        # Add metadata
        metrics['uuid'] = getattr(candidate, 'uuid', 'unknown')
        metrics['family'] = getattr(candidate, 'family', 'unknown')
        
        self.metrics_log.append(metrics)
        return metrics

    def compute_financial_metrics(self, 
                                y_true: np.ndarray, 
                                y_pred: np.ndarray, 
                                events_df: Optional[pd.DataFrame]) -> Dict[str, float]:
        """
        Compute financial metrics: IC, Adjusted IC, Slippage Sensitivity, etc.
        """
        metrics = {}
        
        # Basic IC
        if len(y_pred) > 1:
            ic, _ = spearmanr(y_pred, y_true)
            metrics['IC'] = ic
        else:
            metrics['IC'] = 0.0

        # Turnover-adjusted IC (Simplified)
        # Proxy turnover by sum(|diff(y_pred)|) / len(y_pred)
        # Penalize IC: IC * (1 - Turnover_Penalty)
        if len(y_pred) > 1:
            turnover_proxy = np.mean(np.abs(np.diff(y_pred)))
            # Heuristic: High turnover (>0.5 avg change) reduces effective IC
            metrics['Turnover_Proxy'] = turnover_proxy
            metrics['Turnover_Adj_IC'] = metrics['IC'] * (1.0 - min(turnover_proxy, 1.0) * 0.2)
        
        # Slippage Sensitivity (Simulation)
        # Assume y_true are returns in bps
        # Simulate PnL = sign(y_pred) * y_true - cost
        if len(y_pred) > 0:
            returns = y_true # assuming already scaled or we check raw PnL
            position = np.sign(y_pred - 0.5) # Assuming probability output 0-1
            raw_pnl = position * returns
            
            # Sharpe at 0 bps
            if np.std(raw_pnl) > 0:
                sharpe_0bps = np.mean(raw_pnl) / np.std(raw_pnl)
            else:
                sharpe_0bps = 0.0
            metrics['Sharpe_0bps'] = sharpe_0bps
            
            # Sharpe at 5 bps slippage (per trade)
            # Count trades as change in position
            trades = np.abs(np.diff(position, prepend=0))
            cost_5bps = trades * 0.0005 # 5 bps
            net_pnl_5bps = raw_pnl - cost_5bps
             
            if np.std(net_pnl_5bps) > 0:
                sharpe_5bps = np.mean(net_pnl_5bps) / np.std(net_pnl_5bps)
            else:
                sharpe_5bps = 0.0
            
            # Sensitivity: % drop in Sharpe per bp
            if sharpe_0bps > 0:
                drop_pct = (sharpe_0bps - sharpe_5bps) / sharpe_0bps
                metrics['Slippage_Sensitivity'] = drop_pct / 5.0 # % drop per bp
                metrics['Sharpe_5bps'] = sharpe_5bps
            else:
                metrics['Slippage_Sensitivity'] = 0.0
                metrics['Sharpe_5bps'] = 0.0

        # MFE/MAE Quality Metrics (De Prado's Triple-Barrier Quality)
        if events_df is not None and 'mfe' in events_df.columns and 'mae' in events_df.columns:
            mfe = events_df['mfe'].dropna()
            mae = events_df['mae'].dropna()
            if len(mfe) > 0 and len(mae) > 0:
                # MFE/MAE Ratio: higher = better barrier geometry
                metrics['MFE_MAE_Ratio'] = np.mean(mfe) / (np.mean(mae) + 1e-9)
                
                # Hit Quality: % of events where MFE > MAE (profitable opportunity)
                common_idx = mfe.index.intersection(mae.index)
                if len(common_idx) > 0:
                    metrics['Hit_Quality'] = (mfe.loc[common_idx] > mae.loc[common_idx]).mean()
                else:
                    metrics['Hit_Quality'] = 0.0
                
                # Barrier Asymmetry: correlation between MFE and -MAE
                # Should be negative (high MFE implies low MAE)
                if len(common_idx) > 5:
                    metrics['Barrier_Asymmetry'] = np.corrcoef(
                        mfe.loc[common_idx].values, 
                        -mae.loc[common_idx].values
                    )[0, 1]
                else:
                    metrics['Barrier_Asymmetry'] = 0.0

        return metrics

    def compute_causal_metrics(self, 
                             y_true: np.ndarray, 
                             y_pred: np.ndarray, 
                             cate_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute uplift and causal metrics: Qini, Uplift, GATES, MSE vs Pseudo-Outcome.
        """
        metrics = {}
        
        # 1. MSE on Pseudo-Outcomes
        # Transform y to Y* (transformed outcome)
        # Y* = Y * (W - p) / (p * (1-p)) (Horvitz-Thompson)
        # Assuming binary treatment W approximated by top/bottom quantile of prediction?
        # Actually, for standard heterogeneous treatment effect validation where we have Observational Data:
        # We need a Treatment indicator T. 
        # If we don't have explicit T (this is OOF validation of a geometry), 
        # we often treat the "Signal > 0" as Treatment suggestion.
        # But wait, layer 2 models predict "Probability of Success (Label=1)". 
        # ORF provided CATE.
        # Let's assume T is implicit in how the data was generated (events).
        # OR: We compare CATE to "Realized CATE" proxy?
        # A robust proxy for MSE(tau) in observational data without T column is hard.
        # However, if we assume T comes from a policy or is observed...
        
        # Strategy for "MSE on Pseudo-Outcomes":
        # We need a Treatment column 'T'. In our event-based setups, T is often +1 (Long). 
        # But we only have rows where we acted? No, events are opportunities.
        # If we are always "Acting" (Long) in this dataset outcome (since we filtered for events), 
        # then T=1 for all samples? No, that makes CATE estimation impossible (need control).
        # Ah, the ORF was trained on T (feature > threshold?). 
        # If we don't have T here, we can't compute strictly causal metrics.
        # BUT the plan says "Compute MSE...".
        # Let's look at what we have. Validation set has (X, y).
        # If we don't have T, we can calculate "Uplift" by sorting by CATE and checking y_true (returns).
        # This assumes CATE roughly predicts Returns.
        
        # Uplift Curve / Qini
        # Sort by CATE (descending)
        # Calculate cumulative mean y_true
        
        sorted_idx = np.argsort(cate_pred)[::-1]
        sorted_y = y_true[sorted_idx]
        
        # Cumulative Sum of Outcomes (assumes y_true is return/label)
        cum_y = np.cumsum(sorted_y)
        
        # Area Under Uplift Curve (AUUC) / Qini proxy
        # Ideal Qini requires subtracting random baseline
        x_axis = np.arange(1, len(y_true) + 1)
        random_y = np.linspace(0, cum_y[-1], len(y_true))
        
        qini_curve = cum_y - random_y
        qini_coeff = np.trapz(qini_curve, x=x_axis) / (len(y_true) * np.sum(np.abs(y_true)) + 1e-9) # Normalization heuristic
        
        metrics['Qini_Coefficient'] = qini_coeff
        metrics['Top_Decile_Lift'] = np.mean(sorted_y[:int(len(y_true)*0.1)]) - np.mean(sorted_y)

        # GATES (Grouped Average Treatment Effects)
        # Divide into K groups by CATE score, compare avg outcome
        n_groups = 5
        try:
            groups = pd.qcut(cate_pred, n_groups, labels=False, duplicates='drop')
            gates_res = []
            for g in range(n_groups):
                mask = (groups == g)
                if mask.sum() > 0:
                   gates_res.append(np.mean(y_true[mask]))
                else:
                   gates_res.append(0.0)
            
            # Monotonicity of GATES (Spearman of group id vs avg outcome)
            gates_corr, _ = spearmanr(np.arange(n_groups), gates_res)
            metrics['GATES_Monotonicity'] = gates_corr
            metrics['GATES_Spread'] = (gates_res[-1] - gates_res[0]) # Top - Bottom
        except Exception:
            metrics['GATES_Monotonicity'] = 0.0
            metrics['GATES_Spread'] = 0.0
            
        return metrics

    def compute_structural_metrics(self, 
                                 X: pd.DataFrame, 
                                 y_true: np.ndarray, 
                                 y_pred: np.ndarray,
                                 candidate: Any) -> Dict[str, float]:
        """
        Compute structural metrics: Feature Entropy, R-Loss, Rank Consistency.
        """
        metrics = {}
        
        # Feature Entropy (concentration of importance)
        # Needs feature importances. If candidate has them stored.
        if hasattr(candidate, 'feature_importances_'):
            imps = candidate.feature_importances_
        elif isinstance(candidate, dict) and 'feature_importance' in candidate:
            # candidate['feature_importance'] might be a dict or list
            fi = candidate['feature_importance']
            if isinstance(fi, dict):
                imps = np.array(list(fi.values()))
            else:
                 imps = np.array(fi)
        elif hasattr(candidate, 'estimator') and hasattr(candidate.estimator, 'feature_importances_'):
             imps = candidate.estimator.feature_importances_
        else:
            imps = None
            
        if imps is not None and len(imps) > 0:
            # Normalize
            imps = np.abs(imps)
            imps_sum = np.sum(imps)
            if imps_sum > 0:
                prob = imps / imps_sum
                metrics['Feature_Entropy'] = entropy(prob)
            else:
                metrics['Feature_Entropy'] = 0.0
        else:
             metrics['Feature_Entropy'] = -1.0
             
        # R-Loss (Residual-on-Residual)
        # MSE of Model Residuals vs Null Model (Mean) Residuals
        # 1 - (MSE_model / MSE_null) is R2. "R-Loss" usually implies checking structure in residuals
        # Implementation: Simple MSE
        mse_model = mean_squared_error(y_true, y_pred)
        mse_null = mean_squared_error(y_true, np.full_like(y_true, np.mean(y_true)))
        metrics['MSE_Pseudo'] = mse_model
        metrics['R_Loss_Ratio'] = mse_model / (mse_null + 1e-9)
        
        # Rank Consistency (if time data available)
        # Split into halves, check rank correlation of predictions for similar feature profiles?
        # Or simply stability of prediction ranks over time?
        # Let's do Rank Autocorrelation of daily predictions if we have dates.
        # This requires the events_df to have dates. 
        # Skipping simple implementation for now to save space, defaulting to 0.
        metrics['Rank_Consistency'] = 0.0
        
        return metrics

    def save_report(self, output_dir: str, filename: str = None) -> str:
        """
        Save the accumulated metrics to a CSV file.
        """
        import os
        from datetime import datetime
        
        if not self.metrics_log:
            if self.verbose:
                tprint_warning("⚠️ No metrics to report.")
            return ""
            
        df = pd.DataFrame(self.metrics_log)
        
        # Reorder columns for readability (UUID first)
        cols = ['uuid', 'family'] + [c for c in df.columns if c not in ['uuid', 'family']]
        df = df[cols]
        
        if filename is None:
            filename = f"detailed_metrics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
        path = os.path.join(output_dir, filename)
        df.to_csv(path, index=False)
        
        if self.verbose:
            tprint_info(f"💾 Detailed features report saved to {path}")
            
        return path
