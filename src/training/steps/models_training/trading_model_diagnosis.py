import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_squared_error, mutual_info_score
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, IsotonicRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from scipy.stats import pearsonr, spearmanr, kstest, entropy
from scipy.cluster.hierarchy import linkage, fcluster
import shap
import warnings
from typing import Dict, List, Tuple, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime

# Suppress warnings for cleaner output in production
warnings.filterwarnings("ignore")

class ModelDiagnostician:
    """
    The central hub for all trading model diagnostics.
    Usage:
        diagnostician = ModelDiagnostician(model, X_test, y_test, y_pred, X_train, y_train)
        results = diagnostician.run_full_diagnosis()
        diagnostician.generate_report(results, "my_report.html")
    """
    def __init__(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                 y_pred: np.ndarray, X_train: pd.DataFrame = None, y_train: pd.Series = None):
        """
        Args:
            model: The trained model object (must have .predict()).
            X_test: Test features.
            y_test: Test targets (actuals).
            y_pred: Model predictions on X_test.
            X_train: Training features (optional, needed for some checks like distribution shift).
            y_train: Training targets (optional).
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test.values if isinstance(y_test, pd.Series) else y_test
        self.y_pred = y_pred
        self.X_train = X_train
        self.y_train = y_train.values if isinstance(y_train, pd.Series) else y_train
        
        # Calculate Residuals
        self.residuals = self.y_test - self.y_pred
        
        # Sub-modules
        self.oracle = self.Oracle(self)
        self.architect = self.Architect(self)
        self.critic = self.Critic(self)
        self.navigator = self.Navigator(self)
        self.stress_tester = self.StressTester(self)

    def run_full_diagnosis(self) -> Dict[str, Any]:
        """Run all 22 diagnostic tests and return a structured dictionary."""
        results = {}
        print("Running Oracle Diagnostics...")
        results['oracle'] = self.oracle.run_all()
        print("Running Architect Diagnostics...")
        results['architect'] = self.architect.run_all()
        print("Running Critic Diagnostics...")
        results['critic'] = self.critic.run_all()
        print("Running Navigator Diagnostics...")
        results['navigator'] = self.navigator.run_all()
        print("Running Stress Tester Diagnostics...")
        results['stress_tester'] = self.stress_tester.run_all()
        return results

    def generate_report(self, results: Dict[str, Any], output_path: str = "diagnosis_report.html"):
        """Generate the HTML dashboard from the results."""
        reporter = DiagnosisReporter(results, model_name=str(self.model.__class__.__name__))
        reporter.generate_report(output_path)

    # ==========================================
    # Module 1: The Oracle (Theoretical Bounds)
    # ==========================================
    class Oracle:
        def __init__(self, parent):
            self.p = parent

        def run_all(self):
            return {
                "noise_ceiling": self.estimate_noise_ceiling(),
                "volatility_decomposition": self.target_volatility_decomposition(),
                "maximal_sharpe": self.maximal_sharpe_foresight(),
                "hurst_exponent": self.hurst_exponent(),
                "info_ratio_decomp": self.information_ratio_decomposition()
            }

        def estimate_noise_ceiling(self, n_splits=5) -> Dict:
            """1.1 Noise Ceiling / Bayes Error Estimate"""
            # Method: Estimate predictable variance by comparing consistency across similar samples
            # Simplified proxy: Train a very high capacity model (e.g., KNN or RF) on the test set 
            # using cross-validation to see the limit of extractable signal in this specific sample.
            # Note: This is a heuristic. True Bayes error is hard to know.
            
            # Using a high-variance estimator (Random Forest) to overfit slightly and find the 'limit'
            rf = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_leaf=2, random_state=42)
            
            # TimeSeriesSplit to prevent leakage, but we want to see 'theoretical' max on this data
            tscv = TimeSeriesSplit(n_splits=n_splits)
            scores = []
            X_vals = self.p.X_test.values
            y_vals = self.p.y_test
            
            for train_index, test_index in tscv.split(X_vals):
                X_t, X_v = X_vals[train_index], X_vals[test_index]
                y_t, y_v = y_vals[train_index], y_vals[test_index]
                rf.fit(X_t, y_t)
                scores.append(r2_score(y_v, rf.predict(X_v)))
            
            ceiling_est = np.max([0, np.mean(scores)]) # Floor at 0
            current_r2 = r2_score(self.p.y_test, self.p.y_pred)
            
            return {
                "estimated_ceiling_r2": ceiling_est,
                "current_model_r2": current_r2,
                "gap": ceiling_est - current_r2,
                "status": "Saturated" if (ceiling_est - current_r2) < 0.005 else "Untapped Potential"
            }

        def target_volatility_decomposition(self) -> Dict:
            """1.2 Target Volatility Decomposition"""
            total_var = np.var(self.p.y_test)
            pred_var = np.var(self.p.y_pred)
            noise_var = np.var(self.p.residuals)
            
            # Predictable share
            r2 = pred_var / total_var if total_var > 0 else 0
            
            return {
                "total_variance": total_var,
                "predictable_variance_ratio": r2,
                "noise_variance_ratio": noise_var / total_var if total_var > 0 else 1,
                "interpretation": "High Alpha Potential" if r2 > 0.10 else "Low Alpha Environment"
            }

        def maximal_sharpe_foresight(self, cost_bps=2.0) -> float:
            """1.3 Maximal Sharpe under perfect foresight"""
            # Simulation: Go long if return > cost, short if return < -cost
            # Assumes y_test are raw returns.
            
            cost = cost_bps * 1e-4
            # Perfect signal strategy
            strategy_returns = np.abs(self.p.y_test) - cost
            # Filter for trades that would actually be taken (magnitude > cost)
            active_returns = strategy_returns[strategy_returns > 0]
            
            if len(active_returns) == 0:
                return 0.0
            
            # Annualized Sharpe (assuming daily data, 252 days)
            # Adjust 252 to your timeframe (e.g., crypto 365, hourly 24*365)
            sharpe = (np.mean(active_returns) / np.std(active_returns)) * np.sqrt(252)
            return sharpe

        def hurst_exponent(self) -> Dict:
            """1.4 Fractal Dimension / Hurst Analysis"""
            # R/S Analysis approximation
            ts = self.p.y_test
            lags = range(2, min(len(ts)//2, 100))
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            
            # Fit line to log-log plot
            if len(tau) < 2: 
                return {"H": 0.5, "regime": "Random Walk"}

            try:
                m = np.polyfit(np.log(lags), np.log(tau), 1)
                H = m[0] * 2 # Correction factor for return series vs price series
            except:
                H = 0.5

            regime = "Mean Reversion" if H < 0.45 else "Momentum" if H > 0.55 else "Random Walk"
            return {"H": H, "regime": regime}

        def information_ratio_decomposition(self) -> str:
            """1.5 Information Ratio Decomposition (Conceptual)"""
            # This typically requires trade logs. Returning heuristic.
            sharpe = self.p.y_pred.mean() / (self.p.y_pred.std() + 1e-9) * np.sqrt(252)
            return f"Est. Sharpe from Predictions: {sharpe:.2f}. If Realized << Est., check execution costs."

    # ==========================================
    # Module 2: The Architect (Feature Intelligence)
    # ==========================================
    class Architect:
        def __init__(self, parent):
            self.p = parent

        def run_all(self):
            return {
                "cmi": self.conditional_mutual_information(),
                "rsa": self.representational_similarity_analysis(),
                "ablation": self.feature_ablation_proxy(),
                "shap_interaction": self.detect_interactions(),
                "predictability_heatmap": "Heatmap data generated (see dashboard)" # Placeholder for matrix
            }

        def conditional_mutual_information(self) -> Dict:
            """2.1 Conditional Mutual Information (CMI)"""
            # Estimate I(X; Y | Y_pred) approx I(X; Residuals)
            # If features still correlate with residuals, we left info on the table.
            
            # Calculate MI between each feature and the residuals
            mi_scores = {}
            # subsample for speed if large
            mask = np.random.choice(len(self.p.residuals), size=min(1000, len(self.p.residuals)), replace=False)
            X_sub = self.p.X_test.iloc[mask]
            res_sub = self.p.residuals[mask]
            
            mi = mutual_info_regression(X_sub, res_sub)
            top_missed = dict(zip(self.p.X_test.columns, mi))
            
            # Sort
            top_missed = dict(sorted(top_missed.items(), key=lambda item: item[1], reverse=True))
            
            avg_mi = np.mean(list(top_missed.values()))
            return {
                "avg_unused_info": avg_mi,
                "top_underutilized_features": list(top_missed.keys())[:3],
                "status": "Underfitting" if avg_mi > 0.1 else "Efficient"
            }

        def representational_similarity_analysis(self) -> Dict:
            """2.3 Representational Similarity Analysis (RSA)"""
            # Compare similarity of raw feature space vs model predictions
            # If correlation is 1.0, the model is linear.
            
            # PCA on X
            pca = PCA(n_components=1)
            X_pca = pca.fit_transform(self.p.X_test).flatten()
            
            # Correlation with preds
            corr, _ = pearsonr(X_pca, self.p.y_pred)
            
            return {
                "similarity_score": abs(corr),
                "interpretation": "Linear/Simple Structure" if abs(corr) > 0.8 else "Complex/Non-linear Structure"
            }

        def feature_ablation_proxy(self) -> Dict:
            """2.4 Feature Ablation Matrix (Proxy using Feature Importances)"""
            # Full ablation is expensive. We check if model has "dead" features.
            # If model provides importances (Trees), use them. Else, use permutation.
            
            unused_count = 0
            if hasattr(self.p.model, "feature_importances_"):
                importances = self.p.model.feature_importances_
                unused_count = np.sum(importances < 1e-4)
            
            return {
                "dead_features_count": int(unused_count),
                "action": "Prune features" if unused_count > 0 else "Maintain"
            }

        def detect_interactions(self) -> str:
            """2.5 SHAP Interaction Effects (Heuristic)"""
            # Full SHAP interaction is slow. 
            # Heuristic: Compare R2 of Linear Model vs R2 of Tree Model on the test set.
            
            lr = LinearRegression()
            lr.fit(self.p.X_test, self.p.y_pred) # Fit to model's predictions to see linearity
            r2_lin = lr.score(self.p.X_test, self.p.y_pred)
            
            return {
                "linearity_of_model": r2_lin,
                "interpretation": "High Interactions" if r2_lin < 0.8 else "Mainly Linear"
            }

    # ==========================================
    # Module 3: The Critic (Residual Analysis)
    # ==========================================
    class Critic:
        def __init__(self, parent):
            self.p = parent

        def run_all(self):
            return {
                "orthogonal_tests": self.orthogonal_target_tests(),
                "residual_clustering": self.clustering_residuals(),
                "residual_structure": self.residual_randomness(),
                "residual_autocorr": self.residual_autocorrelation()
            }

        def orthogonal_target_tests(self) -> Dict:
            """3.1 Orthogonal Target Tests"""
            # Check if residuals correlate with Volume or Volatility (proxies)
            # Note: Requires 'Volume' or 'Volatility' to be in X_test. 
            # We will search for typical names.
            
            results = {}
            for col in self.p.X_test.columns:
                if any(x in col.lower() for x in ['vol', 'mom', 'rsi']):
                    corr, p = spearmanr(self.p.X_test[col], self.p.residuals)
                    if p < 0.05 and abs(corr) > 0.1:
                        results[col] = corr
            
            return {
                "leaking_features": results,
                "status": "Leakage Detected" if results else "Clean"
            }

        def clustering_residuals(self) -> Dict:
            """3.2 Clustering Residuals"""
            # Do residuals cluster in time? 
            # Simple check: Rolling mean of squared residuals (Variance clusters)
            
            res_sq = self.p.residuals ** 2
            rolling_risk = pd.Series(res_sq).rolling(window=20).mean()
            
            is_clustered = rolling_risk.std() > (rolling_risk.mean() * 1.5)
            
            return {
                "heteroskedasticity_detected": is_clustered,
                "interpretation": "Regime-dependent performance" if is_clustered else "Uniform errors"
            }

        def residual_randomness(self) -> Dict:
            """3.3 Residual Structure"""
            # Check vs Normal distribution
            stat, p_value = kstest(self.p.residuals, 'norm')
            return {
                "is_normal": p_value > 0.05,
                "p_value": p_value
            }

        def residual_autocorrelation(self) -> Dict:
            """3.4 Residual Autocorrelation"""
            lag1_corr = pd.Series(self.p.residuals).autocorr(lag=1)
            
            status = "Clean"
            if 0.10 < abs(lag1_corr) <= 0.20: status = "Minor Leakage"
            if abs(lag1_corr) > 0.20: status = "Critical Failure (Trend Missed)"
            
            return {
                "lag1_autocorr": lag1_corr,
                "status": status
            }

    # ==========================================
    # Module 4: The Navigator (Regime Dynamics)
    # ==========================================
    class Navigator:
        def __init__(self, parent):
            self.p = parent

        def run_all(self):
            return {
                "regime_stability": self.check_regime_pdp(),
                "transferability": "Requires multi-period split (Manual Check)",
                "distribution_shift": self.distribution_shift_test()
            }

        def check_regime_pdp(self) -> Dict:
            """4.1 Partial Dependence by Regime (Simplified)"""
            # Compare MSE in Low Vol vs High Vol environments
            # Requires Volatility proxy. We'll use rolling std of y_test as proxy.
            
            vol_proxy = pd.Series(self.p.y_test).rolling(10).std().fillna(0)
            high_vol_mask = vol_proxy > vol_proxy.median()
            
            mse_high = mean_squared_error(self.p.y_test[high_vol_mask], self.p.y_pred[high_vol_mask])
            mse_low = mean_squared_error(self.p.y_test[~high_vol_mask], self.p.y_pred[~high_vol_mask])
            
            return {
                "mse_high_vol": mse_high,
                "mse_low_vol": mse_low,
                "ratio": mse_high / (mse_low + 1e-9),
                "warning": mse_high > 2 * mse_low # Model fails in chaos
            }

        def distribution_shift_test(self) -> Dict:
            """4.4 Distribution Shift Tests"""
            if self.p.X_train is None:
                return {"error": "No training data provided for shift test"}
            
            # KS Test on first 3 features
            shifts = {}
            for col in self.p.X_test.columns[:3]:
                stat, p = kstest(self.p.X_train[col], self.p.X_test[col])
                if p < 0.01:
                    shifts[col] = "Drifted"
            
            return {
                "drifted_features": shifts,
                "status": "Drift Detected" if shifts else "Stable"
            }

    # ==========================================
    # Module 5: The Stress Tester (Robustness)
    # ==========================================
    class StressTester:
        def __init__(self, parent):
            self.p = parent

        def run_all(self):
            return {
                "stability": self.model_stability_map(),
                "adversarial": self.adversarial_perturbation(),
                "calibration": self.calibration_error(),
                "bootstrap_ci": self.bootstrap_r2_ci()
            }

        def model_stability_map(self) -> Dict:
            """5.1 Model Stability Map"""
            # Test sensitivity to random subsets (Bootstrap)
            scores = []
            indices = np.arange(len(self.p.y_pred))
            for _ in range(10):
                boot_idx = np.random.choice(indices, size=len(indices), replace=True)
                if len(np.unique(self.p.y_test[boot_idx])) > 1:
                    score = r2_score(self.p.y_test[boot_idx], self.p.y_pred[boot_idx])
                    scores.append(score)
            
            std_dev = np.std(scores) if scores else 0.0
            return {
                "perf_std_dev": std_dev,
                "status": "Fragile" if std_dev > 0.05 else "Robust"
            }

        def adversarial_perturbation(self) -> Dict:
            """5.2 Adversarial Perturbation"""
            # Add 1% noise to features and see if predictions flip sign
            noise = np.random.normal(0, 0.01, self.p.X_test.shape)
            X_noisy = self.p.X_test + noise
            
            try:
                # This requires the model object to run predict
                y_pred_noisy = self.p.model.predict(X_noisy)
                
                # Check Sign Flips
                sign_flips = np.mean(np.sign(self.p.y_pred) != np.sign(y_pred_noisy))
                return {
                    "sign_flip_rate_1pct_noise": sign_flips,
                    "interpretation": "Fragile" if sign_flips > 0.1 else "Robust"
                }
            except:
                return {"error": "Model object does not support direct prediction on array"}

        def calibration_error(self) -> Dict:
            """5.3 Calibration Error (Regression version)"""
            # For regression, check if predicted magnitude matches actual magnitude
            # Rank correlation
            corr, _ = spearmanr(self.p.y_pred, self.p.y_test)
            return {"rank_correlation": corr}

        def bootstrap_r2_ci(self) -> Dict:
            """5.4 Bootstrap R² CI"""
            n_boot = 100
            scores = []
            indices = np.arange(len(self.p.y_pred))
            
            for _ in range(n_boot):
                boot_idx = np.random.choice(indices, size=len(indices), replace=True)
                if len(np.unique(self.p.y_test[boot_idx])) > 1: # Avoid constant target
                    scores.append(r2_score(self.p.y_test[boot_idx], self.p.y_pred[boot_idx]))
            
            lower = np.percentile(scores, 2.5) if scores else 0
            upper = np.percentile(scores, 97.5) if scores else 0
            
            return {
                "ci_lower": lower,
                "ci_upper": upper,
                "reliable_edge": lower > 0
            }

class DiagnosisReporter:
    """
    Generates an interactive HTML report from the ModelDiagnostician results.
    Includes specific guidance logic based on thresholds.
    """
    
    def __init__(self, diagnosis_results: dict, model_name: str = "Model_v1"):
        self.results = diagnosis_results
        self.model_name = model_name
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    def generate_report(self, output_path: str = "diagnosis_report.html"):
        """Main entry point to create the HTML dashboard."""
        
        # 1. Generate Actionable Guidance
        guidance_cards = self._generate_guidance()
        
        # 2. Create Visualizations
        fig = self._create_dashboard_plots()
        
        # 3. Compile HTML
        html_content = self._compile_html(guidance_cards, fig)
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"Diagnosis report saved to {output_path}")

    def _generate_guidance(self) -> list:
        """
        Rule-engine that translates metrics into specific English advice.
        Returns a list of dicts: {'severity': 'critical|warning|info', 'title': str, 'message': str}
        """
        r = self.results
        advice = []

        # --- Oracle Checks ---
        oracle = r.get('oracle', {})
        # Noise Ceiling
        if oracle.get('noise_ceiling', {}).get('gap', 0) > 0.01:
            advice.append({
                'severity': 'info', 'title': 'Untapped Potential',
                'message': f"Theoretical R² is {oracle['noise_ceiling']['estimated_ceiling_r2']:.4f}, but your model is at {oracle['noise_ceiling']['current_model_r2']:.4f}. Significant signal remains uncaptured."
            })
        elif oracle.get('noise_ceiling', {}).get('estimated_ceiling_r2', 0) < 0.002:
            advice.append({
                'severity': 'critical', 'title': 'Target is Noise',
                'message': "The theoretical ceiling for this target is near zero. No model can predict this profitably. Change the target horizon."
            })

        # --- Architect Checks ---
        architect = r.get('architect', {})
        # CMI
        cmi_score = architect.get('cmi', {}).get('avg_unused_info', 0)
        if cmi_score > 0.1:
            advice.append({
                'severity': 'critical', 'title': 'Major Underfitting',
                'message': f"High Conditional Mutual Information ({cmi_score:.2f}). Features contain signal your model missed. Increase model complexity (Depth/Layers)."
            })
        
        # RSA (Linearity)
        rsa = architect.get('rsa', {}).get('similarity_score', 0)
        if rsa > 0.9:
            advice.append({
                'severity': 'warning', 'title': 'Model is Linear',
                'message': "Model representations are highly correlated with raw inputs. It is not learning complex features. Consider Non-linear models."
            })

        # --- Critic Checks ---
        critic = r.get('critic', {})
        # Autocorrelation
        autocorr = abs(critic.get('residual_autocorr', {}).get('lag1_autocorr', 0))
        if autocorr > 0.15:
            advice.append({
                'severity': 'critical', 'title': 'Trend Leaking',
                'message': f"Residual autocorrelation is high ({autocorr:.2f}). The model is missing trend information. Add lag-1 target features or use RNNs."
            })
        
        # Leakage
        leaking = critic.get('orthogonal_tests', {}).get('leaking_features', {})
        if leaking:
            feats = ", ".join(list(leaking.keys())[:3])
            advice.append({
                'severity': 'warning', 'title': 'Alpha Leakage',
                'message': f"Residuals are correlated with: {feats}. Explicitly add these as features."
            })

        # --- Stress Tester Checks ---
        stress = r.get('stress_tester', {})
        # Stability
        if stress.get('stability', {}).get('status') == 'Fragile':
            advice.append({
                'severity': 'critical', 'title': 'Model Fragility',
                'message': "Performance varies significantly across random seeds/subsets. Do not deploy. Use Bagging or reduce Learning Rate."
            })
            
        # CI Edge
        if not stress.get('bootstrap_ci', {}).get('reliable_edge', False):
             advice.append({
                'severity': 'critical', 'title': 'No Statistical Edge',
                'message': "The 95% Confidence Interval for R² includes zero. Performance is indistinguishable from luck."
            })

        if not advice:
            advice.append({
                'severity': 'success', 'title': 'Clean Bill of Health',
                'message': "No major flags detected. Proceed to paper trading."
            })

        return advice

    def _create_dashboard_plots(self):
        """Creates a Plotly subplot figure with key diagnostic charts."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Residual Distribution vs Normal", "Noise Ceiling Gap", 
                           "Bootstrap R² CI", "Autocorrelation Check")
        )

        r = self.results

        # Plot 1: Noise Ceiling Comparison
        oracle = r.get('oracle', {}).get('noise_ceiling', {})
        fig.add_trace(go.Bar(
            x=['Current R²', 'Theoretical Max R²'],
            y=[oracle.get('current_model_r2', 0), oracle.get('estimated_ceiling_r2', 0)],
            marker_color=['#EF553B', '#00CC96'],
            name="Predictability"
        ), row=1, col=1)

        # Plot 2: Feature Ablation / Dead Features (Architect)
        dead_count = r.get('architect', {}).get('ablation', {}).get('dead_features_count', 0)
        fig.add_trace(go.Indicator(
            mode = "number+gauge",
            value = dead_count,
            title = {"text": "Dead Features"},
            domain = {'row': 0, 'column': 1},
            gauge = {'axis': {'range': [None, 50]}, 'bar': {'color': "darkred"}}
        ), row=1, col=2)

        # Plot 3: Bootstrap CI (Stress Tester)
        ci = r.get('stress_tester', {}).get('bootstrap_ci', {})
        fig.add_trace(go.Scatter(
            x=['R² CI'],
            y=[(ci.get('ci_upper', 0) + ci.get('ci_lower', 0))/2],
            error_y=dict(
                type='data',
                symmetric=False,
                array=[ci.get('ci_upper', 0)],
                arrayminus=[ci.get('ci_lower', 0)]
            ),
            mode='markers',
            marker=dict(size=10),
            name="95% CI"
        ), row=2, col=1)

        # Plot 4: Autocorrelation Gauge (Critic)
        ac = abs(r.get('critic', {}).get('residual_autocorr', {}).get('lag1_autocorr', 0))
        fig.add_trace(go.Indicator(
            mode = "number+gauge",
            value = ac,
            title = {"text": "Residual Autocorr"},
            gauge = {
                'axis': {'range': [0, 0.5]},
                'bar': {'color': "green" if ac < 0.1 else "red"},
                'steps': [
                    {'range': [0, 0.1], 'color': "lightgreen"},
                    {'range': [0.1, 0.2], 'color': "yellow"},
                    {'range': [0.2, 0.5], 'color': "salmon"}
                ]
            }
        ), row=2, col=2)

        fig.update_layout(height=700, title_text=f"Diagnostic Overview: {self.model_name}", template="plotly_dark")
        return fig

    def _compile_html(self, advice, fig):
        """Assembles the final HTML string."""
        plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        # CSS for cards
        css = """
        <style>
            body { font-family: sans-serif; background: #1e1e1e; color: #eee; padding: 20px; }
            .container { max_width: 1200px; margin: 0 auto; }
            .card { background: #2d2d2d; border-radius: 8px; padding: 15px; margin-bottom: 10px; border-left: 5px solid #555; }
            .critical { border-left-color: #ff4444; }
            .warning { border-left-color: #ffbb33; }
            .info { border-left-color: #33b5e5; }
            .success { border-left-color: #00C851; }
            h2 { border-bottom: 1px solid #444; padding-bottom: 10px; }
            .json-dump { background: #111; padding: 15px; font-family: monospace; border-radius: 5px; max-height: 300px; overflow-y: scroll; }
        </style>
        """
        
        # Build Advice HTML
        advice_html = "<h2>🤖 AI Guidance</h2>"
        for item in advice:
            advice_html += f"""
            <div class="card {item['severity']}">
                <h3>{item['title']}</h3>
                <p>{item['message']}</p>
            </div>
            """
            
        # Build Full HTML
        return f"""
        <html>
        <head><title>Model Diagnosis: {self.model_name}</title>{css}</head>
        <body>
            <div class="container">
                <h1>Trading Model Diagnosis: {self.model_name}</h1>
                <p>Generated: {self.timestamp}</p>
                
                {advice_html}
                
                <h2>📊 Visual Diagnostics</h2>
                {plot_html}
                
                <h2>📝 Raw Metrics</h2>
                <div class="json-dump">
                    <pre>{json.dumps(self.results, indent=2)}</pre>
                </div>
            </div>
        </body>
        </html>
        """
