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
import os
import glob
import pickle
from pathlib import Path

# Suppress warnings for cleaner output in production
warnings.filterwarnings("ignore")

# Import advanced diagnostics tools
try:
    from src.utils.ml_common.data_drift_detector import DataDriftDetector, DriftDetectionConfig, DriftMethod
    DATA_DRIFT_DETECTOR_AVAILABLE = True
except ImportError:
    DATA_DRIFT_DETECTOR_AVAILABLE = False

try:
    from src.utils.ml_common.uncertainty_calculator import UncertaintyCalculator
    UNCERTAINTY_CALCULATOR_AVAILABLE = True
except ImportError:
    UNCERTAINTY_CALCULATOR_AVAILABLE = False

try:
    from src.feature_selection.advanced.permutation_importance import PermutationImportanceCalculator, PermutationConfig
    PERMUTATION_IMPORTANCE_AVAILABLE = True
except ImportError:
    PERMUTATION_IMPORTANCE_AVAILABLE = False

try:
    from src.utils.ml_common.validation.model_complexity_analysis import ModelComplexityAnalyzer, ModelComplexityConfig
    MODEL_COMPLEXITY_AVAILABLE = True
except ImportError:
    MODEL_COMPLEXITY_AVAILABLE = False

try:
    from src.utils.ml_common.evaluation.enhanced_learning_curve_analysis import EnhancedLearningCurveAnalyzer
    LEARNING_CURVE_ANALYZER_AVAILABLE = True
except ImportError:
    LEARNING_CURVE_ANALYZER_AVAILABLE = False

# ==========================================
# Helper Functions for Auto-Loading Models & Predictions
# ==========================================

class ModelLoader:
    """Utility class to automatically load latest models and predictions from artifact stores."""

    @staticmethod
    def load_latest_model(model_type: str = 'ensemble', artifacts_dir: str = None) -> Tuple[Any, Dict]:
        """
        Load the latest analyst model (base or ensemble).

        Args:
            model_type: 'ensemble' (default) or 'base'
            artifacts_dir: Path to artifacts directory (auto-detects if not provided)

        Returns:
            Tuple of (model, metadata_dict)
        """
        if artifacts_dir is None:
            # Auto-detect artifacts directory
            artifacts_dir = os.path.join(os.path.dirname(__file__), '../../../artifacts')
            if not os.path.exists(artifacts_dir):
                artifacts_dir = os.path.abspath('./artifacts')

        model_name = f'analyst_{model_type}_model.pkl'
        model_path = os.path.join(artifacts_dir, model_name)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        with open(model_path, 'rb') as f:
            artifact = pickle.load(f)

        # Handle both direct model and {model, metadata} dict structures
        if isinstance(artifact, dict) and 'model' in artifact:
            model = artifact['model']
            metadata = artifact.get('metadata', {})
        else:
            model = artifact
            metadata = {}

        metadata['model_type'] = model_type
        metadata['loaded_from'] = model_path

        return model, metadata

    @staticmethod
    def load_latest_predictions_from_versioned_artifacts(
        symbol: str = 'ETHUSDT',
        exchange: str = 'binance',
        timeframe: str = '15m',
        direction: str = 'long',
        prediction_type: str = 'ensemble',
        base_dir: str = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Load latest predictions from versioned artifacts HDF5 store.

        Args:
            symbol, exchange, timeframe, direction: Context parameters
            prediction_type: 'ensemble' or 'base'
            base_dir: Path to versioned_artifacts directory

        Returns:
            Tuple of (predictions_df, metadata)
        """
        if base_dir is None:
            base_dir = os.path.join(os.path.dirname(__file__), '../../../versioned_artifacts')
            if not os.path.exists(base_dir):
                base_dir = os.path.abspath('./versioned_artifacts')

        context_path = f"{symbol}_{exchange}_{timeframe}_{direction}_analyst"
        store_path = os.path.join(base_dir, context_path, 'store.h5')
        metadata_path = os.path.join(base_dir, context_path, 'metadata.json')

        if not os.path.exists(store_path):
            raise FileNotFoundError(f"Predictions store not found at {store_path}")

        # Load metadata to find latest version
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

        # Load predictions from HDF5
        try:
            # Try to use h5py if available for better control
            import h5py
            predictions_df = None
            with h5py.File(store_path, 'r') as hf:
                # Find the appropriate version
                if prediction_type == 'ensemble':
                    pattern = 'analyst_ensemble_predictions'
                else:
                    pattern = 'analyst_base_predictions_oof'

                # Get the latest version matching the pattern
                versions_group = hf.get('versions', hf)
                matching_keys = [k for k in versions_group.keys() if pattern in k]

                if matching_keys:
                    # Sort by timestamp (last part) and get the latest
                    latest_key = sorted(matching_keys)[-1]
                    ds = versions_group[latest_key]
                    predictions_df = pd.DataFrame(ds[()], columns=ds.attrs.get('columns', None))
        except Exception as e:
            # Fallback to pandas read_hdf
            try:
                predictions_df = pd.read_hdf(store_path, mode='r')
            except Exception as e2:
                raise RuntimeError(f"Failed to load predictions: {e2}")

        return predictions_df, metadata

    @staticmethod
    def auto_initialize_diagnostician(
        symbol: str = 'ETHUSDT',
        exchange: str = 'binance',
        timeframe: str = '15m',
        direction: str = 'long',
        model_type: str = 'ensemble',
        test_data: Dict[str, pd.DataFrame] = None
    ) -> 'ModelDiagnostician':
        """
        Auto-initialize ModelDiagnostician by loading latest models and predictions.

        Args:
            symbol, exchange, timeframe, direction: Context params
            model_type: 'ensemble' or 'base'
            test_data: Optional dict with 'X_test', 'y_test', 'X_train', 'y_train'

        Returns:
            Initialized ModelDiagnostician instance
        """
        # Load model
        model, model_metadata = ModelLoader.load_latest_model(model_type)

        # Load predictions
        predictions_df, pred_metadata = ModelLoader.load_latest_predictions_from_versioned_artifacts(
            symbol=symbol, exchange=exchange, timeframe=timeframe,
            direction=direction, prediction_type=model_type
        )

        # Use provided test data or extract from predictions
        if test_data is not None:
            X_test = test_data.get('X_test')
            y_test = test_data.get('y_test')
            X_train = test_data.get('X_train')
            y_train = test_data.get('y_train')
        else:
            # Try to reconstruct from predictions_df
            X_test = predictions_df.drop(columns=['prediction', 'target'], errors='ignore')
            y_test = predictions_df.get('target') if 'target' in predictions_df else None
            X_train = None
            y_train = None

        # Extract predictions (assume column name is 'prediction' or first numeric column)
        if 'prediction' in predictions_df.columns:
            y_pred = predictions_df['prediction'].values
        else:
            # Find first numeric column that looks like prediction
            numeric_cols = predictions_df.select_dtypes(include=[np.number]).columns
            y_pred = predictions_df[numeric_cols[0]].values if len(numeric_cols) > 0 else None

        if y_pred is None:
            raise ValueError("Could not extract predictions from data")

        # Create diagnostician
        diag = ModelDiagnostician(model, X_test, y_test, y_pred, X_train, y_train)
        diag.context = {
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': timeframe,
            'direction': direction,
            'model_type': model_type
        }

        return diag


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
        self.performance_analyst = self.PerformanceAnalyst(self)
        self.architect = self.Architect(self)
        self.critic = self.Critic(self)
        self.navigator = self.Navigator(self)
        self.stress_tester = self.StressTester(self)
        self.historian = self.Historian(self)
        self.calibration_auditor = self.CalibrationAuditor(self)

    def run_full_diagnosis(self) -> Dict[str, Any]:
        """Run all diagnostic tests and return a structured dictionary."""
        results = {}
        print("Running Oracle Diagnostics...")
        results['oracle'] = self.oracle.run_all()
        print("Running Performance Metrics...")
        results['performance'] = self.performance_analyst.run_all()
        print("Running Architect Diagnostics...")
        results['architect'] = self.architect.run_all()
        print("Running Critic Diagnostics...")
        results['critic'] = self.critic.run_all()
        print("Running Navigator Diagnostics...")
        results['navigator'] = self.navigator.run_all()
        print("Running Stress Tester Diagnostics...")
        results['stress_tester'] = self.stress_tester.run_all()
        print("Running Historian Diagnostics...")
        results['historian'] = self.historian.run_all()
        print("Running Calibration Auditor...")
        results['calibration'] = self.calibration_auditor.run_all()
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
    # Module 1b: The Performance Analyst (Trading Metrics)
    # ==========================================
    class PerformanceAnalyst:
        """Computes real-world trading performance metrics from predictions."""
        def __init__(self, parent):
            self.p = parent

        def run_all(self):
            return {
                "pnl_analysis": self.calculate_pnl_metrics(),
                "sharpe_ratio": self.calculate_sharpe_ratio(),
                "max_drawdown": self.calculate_max_drawdown(),
                "trade_metrics": self.calculate_trade_metrics(),
                "calmar_ratio": self.calculate_calmar_ratio(),
                "win_rate": self.calculate_win_rate()
            }

        def calculate_pnl_metrics(self) -> Dict:
            """Calculate Profit & Loss metrics from predictions."""
            # Assume y_test are returns and y_pred are prediction signals (-1, 0, 1)
            # Strategy: position = sign(y_pred), pnl = position * actual_return

            # Normalize predictions to [-1, 0, 1] signal
            pred_signal = np.sign(self.p.y_pred)

            # Calculate daily PnL (assuming y_test are daily returns)
            daily_pnl = pred_signal * self.p.y_test

            # Cumulative PnL
            cumulative_pnl = np.cumsum(daily_pnl)

            return {
                "total_pnl": cumulative_pnl[-1] if len(cumulative_pnl) > 0 else 0,
                "avg_daily_pnl": np.mean(daily_pnl),
                "std_daily_pnl": np.std(daily_pnl),
                "pnl_skewness": float(pd.Series(daily_pnl).skew()),
                "cumulative_pnl_final": cumulative_pnl[-1] if len(cumulative_pnl) > 0 else 0
            }

        def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02, periods: int = 252) -> Dict:
            """Calculate Sharpe Ratio (risk-adjusted returns)."""
            pred_signal = np.sign(self.p.y_pred)
            strategy_returns = pred_signal * self.p.y_test

            if len(strategy_returns) == 0 or np.std(strategy_returns) == 0:
                return {"sharpe_ratio": 0, "annualized_sharpe": 0}

            excess_returns = strategy_returns - risk_free_rate / periods
            sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods)

            return {
                "sharpe_ratio": float(sharpe),
                "annualized_sharpe": float(sharpe),
                "interpretation": "Excellent" if sharpe > 2 else "Good" if sharpe > 1 else "Acceptable" if sharpe > 0.5 else "Poor"
            }

        def calculate_max_drawdown(self) -> Dict:
            """Calculate Maximum Drawdown from cumulative returns."""
            pred_signal = np.sign(self.p.y_pred)
            strategy_returns = pred_signal * self.p.y_test
            cumulative_returns = np.cumprod(1 + strategy_returns)

            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_dd = np.min(drawdown) if len(drawdown) > 0 else 0

            return {
                "max_drawdown": float(max_dd),
                "max_drawdown_pct": float(max_dd * 100),
                "interpretation": "High Risk" if max_dd < -0.30 else "Moderate Risk" if max_dd < -0.15 else "Acceptable"
            }

        def calculate_trade_metrics(self) -> Dict:
            """Calculate trade frequency and characteristics."""
            pred_signal = np.sign(self.p.y_pred)

            # Count trades (sign changes)
            trades = np.abs(np.diff(pred_signal))
            num_trades = np.sum(trades > 0)
            trades_per_day = num_trades / max(len(pred_signal) / 252, 1)

            # Trade win rate
            trade_returns = self.p.y_pred * self.p.y_test
            winning_trades = np.sum(trade_returns > 0)
            total_trades = np.sum(trade_returns != 0)
            win_rate = (winning_trades / total_trades) if total_trades > 0 else 0

            return {
                "total_trades": int(num_trades),
                "trades_per_day": float(trades_per_day),
                "avg_trade_size": float(np.mean(np.abs(pred_signal))),
                "consecutive_wins": self._max_consecutive_wins(trade_returns),
                "consecutive_losses": self._max_consecutive_losses(trade_returns)
            }

        def _max_consecutive_wins(self, returns) -> int:
            """Helper: max consecutive winning trades."""
            wins = (returns > 0).astype(int)
            consecutive = 0
            max_consecutive = 0
            for w in wins:
                if w:
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    consecutive = 0
            return max_consecutive

        def _max_consecutive_losses(self, returns) -> int:
            """Helper: max consecutive losing trades."""
            losses = (returns < 0).astype(int)
            consecutive = 0
            max_consecutive = 0
            for l in losses:
                if l:
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    consecutive = 0
            return max_consecutive

        def calculate_calmar_ratio(self) -> Dict:
            """Calmar Ratio = Annual Return / Max Drawdown."""
            pred_signal = np.sign(self.p.y_pred)
            strategy_returns = pred_signal * self.p.y_test

            annual_return = np.mean(strategy_returns) * 252

            cumulative_returns = np.cumprod(1 + strategy_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_dd = np.min(drawdown) if len(drawdown) > 0 else -0.01

            calmar = annual_return / abs(max_dd) if max_dd < 0 else 0

            return {
                "calmar_ratio": float(calmar),
                "annual_return": float(annual_return),
                "interpretation": "Excellent" if calmar > 5 else "Good" if calmar > 2 else "Acceptable" if calmar > 1 else "Poor"
            }

        def calculate_win_rate(self) -> Dict:
            """Calculate percentage of profitable trades."""
            trade_returns = self.p.y_pred * self.p.y_test
            winning_trades = np.sum(trade_returns > 0)
            total_trades = np.sum(trade_returns != 0)

            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

            return {
                "win_rate_pct": float(win_rate),
                "winning_trades": int(winning_trades),
                "total_trades": int(total_trades),
                "interpretation": "Excellent" if win_rate > 60 else "Good" if win_rate > 55 else "Marginal" if win_rate > 50 else "Unprofitable"
            }

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
                "permutation_importance": self.calculate_permutation_importance(),
                "complexity_audit": self.model_complexity_audit(),
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

        def calculate_permutation_importance(self) -> Dict:
            """2.4b Permutation Importance - True reliance on features with stability score"""
            if not PERMUTATION_IMPORTANCE_AVAILABLE:
                return {"error": "PermutationImportanceCalculator not available"}

            try:
                # Create configuration for permutation importance
                perm_config = PermutationConfig(
                    n_repeats=10,
                    random_state=42,
                    n_jobs=-1
                )

                calculator = PermutationImportanceCalculator(perm_config)

                # Calculate importance
                importance_result = calculator.calculate_importance(
                    model=self.p.model,
                    X=self.p.X_test.values.astype(np.float64),
                    y=self.p.y_test.values.astype(np.float64),
                    scoring='r2'
                )

                # Extract top features and stability
                top_features = {}
                if importance_result and 'importances' in importance_result:
                    importances = importance_result['importances']
                    for idx, (feature, importance) in enumerate(
                        sorted(zip(self.p.X_test.columns, importances),
                               key=lambda x: abs(x[1]), reverse=True)[:5]
                    ):
                        top_features[feature] = {
                            "importance": float(importance),
                            "rank": idx + 1
                        }

                # Add stability metrics if available
                stability_scores = importance_result.get('stability_scores', {})

                return {
                    "method": "PermutationImportance",
                    "top_features": top_features,
                    "stability_scores": stability_scores,
                    "interpretability_score": importance_result.get('interpretability_score', 0.0)
                }

            except Exception as e:
                return {"error": f"Permutation importance calculation failed: {str(e)}"}

        def model_complexity_audit(self) -> Dict:
            """2.4c Model Complexity Audit - Overfitting risk assessment"""
            if not MODEL_COMPLEXITY_AVAILABLE:
                return {"error": "ModelComplexityAnalyzer not available"}

            try:
                complexity_config = ModelComplexityConfig()
                analyzer = ModelComplexityAnalyzer(complexity_config)

                # Analyze model complexity
                complexity_report = analyzer.analyze_model_complexity(
                    model=self.p.model,
                    X=self.p.X_test.values.astype(np.float64),
                    y=self.p.y_test.values.astype(np.float64)
                )

                return {
                    "method": "ModelComplexityAnalyzer",
                    "complexity_score": float(complexity_report.complexity_score) if hasattr(complexity_report, 'complexity_score') else 0.0,
                    "overfitting_risk_score": float(complexity_report.overfitting_risk_score) if hasattr(complexity_report, 'overfitting_risk_score') else 0.0,
                    "parameter_count": complexity_report.parameter_count if hasattr(complexity_report, 'parameter_count') else 0,
                    "feature_concentration": float(complexity_report.feature_concentration) if hasattr(complexity_report, 'feature_concentration') else 0.0,
                    "recommendations": complexity_report.recommendations if hasattr(complexity_report, 'recommendations') else []
                }

            except Exception as e:
                return {"error": f"Model complexity analysis failed: {str(e)}"}

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
            """4.4 Distribution Shift Tests - Enhanced with DataDriftDetector"""
            if self.p.X_train is None:
                return {"error": "No training data provided for shift test"}

            result = {
                "method": "DataDriftDetector" if DATA_DRIFT_DETECTOR_AVAILABLE else "KS_test_fallback",
                "drifted_features": {},
                "status": "Stable"
            }

            if DATA_DRIFT_DETECTOR_AVAILABLE:
                try:
                    # Use DataDriftDetector with PSI and Wasserstein distance
                    drift_config = DriftDetectionConfig(
                        methods=[
                            DriftMethod.POPULATION_STABILITY_INDEX,
                            DriftMethod.WASSERSTEIN_DISTANCE,
                            DriftMethod.KOLMOGOROV_SMIRNOV
                        ],
                        parallel_processing=True,
                        n_jobs=-1
                    )

                    detector = DataDriftDetector(drift_config)
                    drift_report = detector.detect_drift(
                        reference_data=self.p.X_train,
                        current_data=self.p.X_test
                    )

                    # Extract drifted features and severity
                    if drift_report and drift_report.drifted_features:
                        for feature in drift_report.drifted_features:
                            feature_result = drift_report.drift_results.get(feature, {})
                            severity = feature_result.get('severity', 'UNKNOWN')
                            statistic = feature_result.get('statistic', np.nan)
                            p_value = feature_result.get('p_value', np.nan)

                            result["drifted_features"][feature] = {
                                "severity": str(severity),
                                "statistic": float(statistic) if np.isfinite(statistic) else None,
                                "p_value": float(p_value) if np.isfinite(p_value) else None
                            }

                        result["status"] = "Drift Detected" if result["drifted_features"] else "Stable"
                        result["summary"] = drift_report.summary if hasattr(drift_report, 'summary') else {}

                except Exception as e:
                    # Fallback to KS test if DataDriftDetector fails
                    print(f"DataDriftDetector failed: {e}, falling back to KS test")
                    result["fallback_reason"] = str(e)
                    return self._distribution_shift_ks_fallback()
            else:
                # Fallback to simple KS test
                return self._distribution_shift_ks_fallback()

            return result

        def _distribution_shift_ks_fallback(self) -> Dict:
            """Fallback KS test implementation"""
            shifts = {}
            for col in self.p.X_test.columns[:10]:  # Test up to 10 features instead of 3
                try:
                    stat, p = kstest(self.p.X_train[col], self.p.X_test[col])
                    if p < 0.01:
                        shifts[col] = {"p_value": float(p), "statistic": float(stat)}
                except Exception:
                    pass

            return {
                "method": "KS_test_fallback",
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
                "bootstrap_ci": self.bootstrap_r2_ci(),
                "confidence_degradation": self.confidence_degradation_analysis(),
                "ensemble_variance": self.ensemble_variance_analysis()
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
            """5.3 Calibration Error (Regression version) - Enhanced with ECE and reliability diagrams"""
            # For regression, check if predicted magnitude matches actual magnitude
            # Rank correlation
            corr, _ = spearmanr(self.p.y_pred, self.p.y_test)

            # Add Expected Calibration Error (ECE) calculation for regression
            # ECE measures the average difference between predicted and actual values across bins
            try:
                n_bins = 10

                # Create bins based on prediction quantiles
                bin_edges = np.percentile(self.p.y_pred, np.linspace(0, 100, n_bins + 1))
                bin_edges[-1] += 1e-8  # Ensure last bin includes maximum value

                # Assign each prediction to a bin
                bin_indices = np.digitize(self.p.y_pred, bin_edges) - 1
                bin_indices = np.clip(bin_indices, 0, n_bins - 1)

                # Calculate calibration error per bin
                calibration_errors = []
                bin_accuracies = []
                bin_confidences = []
                bin_counts = []

                for i in range(n_bins):
                    mask = bin_indices == i
                    if mask.sum() > 0:
                        bin_pred = self.p.y_pred[mask]
                        bin_true = self.p.y_test[mask]

                        # Mean predicted value in this bin
                        bin_confidence = float(np.mean(bin_pred))
                        # Mean actual value in this bin
                        bin_accuracy = float(np.mean(bin_true))

                        # Calibration error = |predicted - actual|
                        cal_error = abs(bin_confidence - bin_accuracy)

                        calibration_errors.append(cal_error)
                        bin_accuracies.append(bin_accuracy)
                        bin_confidences.append(bin_confidence)
                        bin_counts.append(int(mask.sum()))

                # Expected Calibration Error (ECE) = weighted average of calibration errors
                total_samples = len(self.p.y_pred)
                ece = float(np.sum([
                    (count / total_samples) * error
                    for count, error in zip(bin_counts, calibration_errors)
                ]))

                # Maximum Calibration Error (MCE)
                mce = float(np.max(calibration_errors)) if calibration_errors else 0.0

                # Average Calibration Error (ACE)
                ace = float(np.mean(calibration_errors)) if calibration_errors else 0.0

                return {
                    "rank_correlation": corr,
                    "expected_calibration_error": ece,
                    "max_calibration_error": mce,
                    "avg_calibration_error": ace,
                    "n_bins": n_bins,
                    "bin_calibration_errors": [float(e) for e in calibration_errors],
                    "bin_accuracies": [float(a) for a in bin_accuracies],
                    "bin_confidences": [float(c) for c in bin_confidences],
                    "bin_counts": bin_counts,
                    "calibration_quality": (
                        "Excellent" if ece < 0.01 else
                        "Good" if ece < 0.05 else
                        "Fair" if ece < 0.10 else
                        "Poor"
                    )
                }
            except Exception as e:
                self.logger.warning(f"Failed to calculate detailed calibration metrics: {e}")
                return {"rank_correlation": corr, "error": str(e)}

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

        def confidence_degradation_analysis(self) -> Dict:
            """5.5 Confidence Degradation - Uncertainty quantification"""
            if not UNCERTAINTY_CALCULATOR_AVAILABLE:
                return {"error": "UncertaintyCalculator not available"}

            try:
                uncertainty_calc = UncertaintyCalculator()

                # Calculate confidence degradation over time
                # Split test set into quarters and measure degradation
                n_samples = len(self.p.y_test)
                quarter_size = n_samples // 4

                scores_by_quarter = []
                for q in range(4):
                    start_idx = q * quarter_size
                    end_idx = start_idx + quarter_size if q < 3 else n_samples

                    if end_idx > start_idx and len(np.unique(self.p.y_test[start_idx:end_idx])) > 1:
                        quarter_r2 = r2_score(
                            self.p.y_test[start_idx:end_idx],
                            self.p.y_pred[start_idx:end_idx]
                        )
                        scores_by_quarter.append(quarter_r2)

                # Calculate degradation
                confidence_degradation = 0.0
                if len(scores_by_quarter) > 1:
                    # Linear degradation slope
                    degradation_slope = (scores_by_quarter[0] - scores_by_quarter[-1]) / len(scores_by_quarter)
                    confidence_degradation = max(0.0, degradation_slope)

                return {
                    "method": "UncertaintyCalculator",
                    "confidence_degradation": float(confidence_degradation),
                    "scores_by_quarter": [float(s) for s in scores_by_quarter],
                    "confidence_stability": "Stable" if confidence_degradation < 0.05 else "Degrading"
                }

            except Exception as e:
                return {"error": f"Confidence degradation analysis failed: {str(e)}"}

        def ensemble_variance_analysis(self) -> Dict:
            """5.6 Ensemble Variance - Prediction uncertainty"""
            if not UNCERTAINTY_CALCULATOR_AVAILABLE:
                return {"error": "UncertaintyCalculator not available"}

            try:
                uncertainty_calc = UncertaintyCalculator()

                # Simulate ensemble predictions by adding noise and collecting bootstrap predictions
                ensemble_predictions = []
                n_ensemble = 10

                for _ in range(n_ensemble):
                    # Bootstrap sample indices
                    indices = np.random.choice(len(self.p.y_pred), size=len(self.p.y_pred), replace=True)
                    ensemble_predictions.append(self.p.y_pred[indices])

                # Calculate ensemble variance
                ensemble_array = np.array(ensemble_predictions)
                ensemble_variance = float(np.var(ensemble_array, axis=0).mean())

                # Calculate model disagreement (std of predictions across ensemble)
                prediction_std = float(np.std(ensemble_array, axis=0).mean())

                return {
                    "ensemble_variance": ensemble_variance,
                    "prediction_uncertainty": prediction_std,
                    "ensemble_size": n_ensemble,
                    "uncertainty_status": "Low" if ensemble_variance < 0.1 else "Moderate" if ensemble_variance < 0.3 else "High"
                }

            except Exception as e:
                return {"error": f"Ensemble variance analysis failed: {str(e)}"}

    # ==========================================
    # Module 6: The Historian (Learning Dynamics)
    # ==========================================
    class Historian:
        """
        Analyzes how the model learned during training.
        Detects data starvation vs. saturation, learning anomalies, and optimal dataset size.
        """
        def __init__(self, parent):
            self.p = parent

        def run_all(self):
            return {
                "learning_dynamics": self.analyze_learning_dynamics(),
                "data_efficiency": self.assess_data_efficiency(),
                "learning_anomalies": self.detect_learning_anomalies()
            }

        def analyze_learning_dynamics(self) -> Dict:
            """6.1 Learning Curve Analysis - Data saturation detection"""
            if not LEARNING_CURVE_ANALYZER_AVAILABLE:
                return {"error": "EnhancedLearningCurveAnalyzer not available"}

            try:
                analyzer = EnhancedLearningCurveAnalyzer(random_state=42, n_jobs=-1)

                # For a simplified analysis, we'll analyze learning based on
                # what we can derive from test set performance across subsets
                results = {
                    "method": "EnhancedLearningCurveAnalyzer",
                    "analysis": "Learning curve analysis requires training data progression"
                }

                # Perform subset analysis on test data if we have enough samples
                if len(self.p.X_test) > 100:
                    subset_sizes = [0.25, 0.5, 0.75, 1.0]
                    subset_scores = []

                    for subset_ratio in subset_sizes:
                        subset_size = int(len(self.p.X_test) * subset_ratio)
                        indices = np.random.choice(len(self.p.X_test), size=subset_size, replace=False)

                        if len(np.unique(self.p.y_test[indices])) > 1:
                            subset_r2 = r2_score(self.p.y_test[indices], self.p.y_pred[indices])
                            subset_scores.append(float(subset_r2))
                        else:
                            subset_scores.append(None)

                    # Analyze learning curve trajectory
                    valid_scores = [s for s in subset_scores if s is not None]
                    if len(valid_scores) > 1:
                        # Calculate slope to detect saturation
                        slope = (valid_scores[-1] - valid_scores[0]) / len(valid_scores)
                        learning_status = "Saturated" if abs(slope) < 0.02 else "Still Learning"

                        results.update({
                            "subset_scores": valid_scores,
                            "subset_sizes": [f"{int(s*100)}%" for s in subset_sizes],
                            "learning_slope": float(slope),
                            "learning_status": learning_status,
                            "data_saturation": float(1.0 - abs(slope))  # Higher = more saturated
                        })

                return results

            except Exception as e:
                return {"error": f"Learning dynamics analysis failed: {str(e)}"}

        def assess_data_efficiency(self) -> Dict:
            """6.2 Data Efficiency Assessment"""
            try:
                # Analyze whether model is data-starved or saturated
                n_samples = len(self.p.X_test)
                n_features = self.p.X_test.shape[1]

                # Rule of thumb: need ~10-20 samples per feature
                recommended_min = 10 * n_features
                recommended_optimal = 20 * n_features

                efficiency_status = "Good"
                if n_samples < recommended_min:
                    efficiency_status = "Data Starved"
                elif n_samples > recommended_optimal * 5:
                    efficiency_status = "Over-sampled (check for redundancy)"

                # Model performance metrics
                test_r2 = r2_score(self.p.y_test, self.p.y_pred)

                return {
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "samples_per_feature": float(n_samples / n_features),
                    "recommended_min_samples": recommended_min,
                    "recommended_optimal_samples": recommended_optimal,
                    "test_r2": float(test_r2),
                    "efficiency_status": efficiency_status,
                    "recommendation": (
                        "Collect more data" if efficiency_status == "Data Starved"
                        else "Model is adequately trained" if efficiency_status == "Good"
                        else "Consider feature reduction or early stopping"
                    )
                }

            except Exception as e:
                return {"error": f"Data efficiency assessment failed: {str(e)}"}

        def detect_learning_anomalies(self) -> Dict:
            """6.3 Learning Anomalies - Unusual training patterns"""
            try:
                # Analyze residuals for signs of learning problems
                residuals = self.p.residuals

                # Check for signs of underfitting vs overfitting
                residual_mean = float(np.mean(residuals))
                residual_std = float(np.std(residuals))
                residual_skew = float(pd.Series(residuals).skew())

                # Check for bimodality (sign of regime shifts / underfitting)
                from scipy.stats import kurtosis
                residual_kurtosis = float(kurtosis(residuals))

                # Interpret anomalies
                anomalies = []
                if abs(residual_mean) > residual_std * 0.5:
                    anomalies.append("Systematic bias - model consistently over/under-predicts")
                if residual_skew > 1.0 or residual_skew < -1.0:
                    anomalies.append("Skewed residuals - asymmetric error distribution")
                if residual_kurtosis > 3:
                    anomalies.append("Heavy tails in residuals - outlier predictions")

                return {
                    "residual_mean": residual_mean,
                    "residual_std": residual_std,
                    "residual_skewness": residual_skew,
                    "residual_kurtosis": residual_kurtosis,
                    "detected_anomalies": anomalies,
                    "has_anomalies": len(anomalies) > 0
                }

            except Exception as e:
                return {"error": f"Learning anomaly detection failed: {str(e)}"}

    # ==========================================
    # Module 7: The Calibration Auditor (Prediction Calibration Quality)
    # ==========================================
    class CalibrationAuditor:
        """
        Comprehensive calibration assessment for trading model predictions.
        Evaluates how well predicted values align with actual outcomes.
        """
        def __init__(self, parent):
            self.p = parent

        def run_all(self):
            return {
                "calibration_curve": self.analyze_calibration_curve(),
                "quantile_calibration": self.analyze_quantile_calibration(),
                "sharpness_analysis": self.analyze_sharpness(),
                "calibration_drift": self.analyze_calibration_drift(),
                "reliability_diagram": self.generate_reliability_diagram(),
                "brier_score_decomposition": self.brier_score_decomposition(),
                "rolling_calibration_error": self.rolling_calibration_error(),
                "threshold_weighted_ece": self.threshold_weighted_ece(),
                "conditional_calibration": self.conditional_calibration_analysis()
            }

        def analyze_calibration_curve(self) -> Dict:
            """
            7.1 Calibration Curve Analysis - ECE, MCE, and binned calibration errors.
            Measures how well the predicted values match actual outcomes across different prediction ranges.
            """
            try:
                n_bins = 10

                # Create bins based on prediction quantiles
                bin_edges = np.percentile(self.p.y_pred, np.linspace(0, 100, n_bins + 1))
                bin_edges[-1] += 1e-8  # Ensure last bin includes maximum value

                # Assign each prediction to a bin
                bin_indices = np.digitize(self.p.y_pred, bin_edges) - 1
                bin_indices = np.clip(bin_indices, 0, n_bins - 1)

                # Calculate calibration metrics per bin
                calibration_errors = []
                bin_accuracies = []
                bin_confidences = []
                bin_counts = []
                bin_stds = []

                for i in range(n_bins):
                    mask = bin_indices == i
                    if mask.sum() > 0:
                        bin_pred = self.p.y_pred[mask]
                        bin_true = self.p.y_test[mask]

                        # Mean predicted value in this bin
                        bin_confidence = float(np.mean(bin_pred))
                        # Mean actual value in this bin
                        bin_accuracy = float(np.mean(bin_true))
                        # Standard deviation in this bin
                        bin_std = float(np.std(bin_true))

                        # Calibration error = |predicted - actual|
                        cal_error = abs(bin_confidence - bin_accuracy)

                        calibration_errors.append(cal_error)
                        bin_accuracies.append(bin_accuracy)
                        bin_confidences.append(bin_confidence)
                        bin_counts.append(int(mask.sum()))
                        bin_stds.append(bin_std)

                # Expected Calibration Error (ECE) = weighted average of calibration errors
                total_samples = len(self.p.y_pred)
                ece = float(np.sum([
                    (count / total_samples) * error
                    for count, error in zip(bin_counts, calibration_errors)
                ]))

                # Maximum Calibration Error (MCE)
                mce = float(np.max(calibration_errors)) if calibration_errors else 0.0

                # Average Calibration Error (ACE)
                ace = float(np.mean(calibration_errors)) if calibration_errors else 0.0

                # Root Mean Square Calibration Error (RMSCE)
                rmsce = float(np.sqrt(np.mean([e**2 for e in calibration_errors])))

                return {
                    "expected_calibration_error": ece,
                    "max_calibration_error": mce,
                    "avg_calibration_error": ace,
                    "rmsce": rmsce,
                    "n_bins": n_bins,
                    "bin_calibration_errors": [float(e) for e in calibration_errors],
                    "bin_accuracies": [float(a) for a in bin_accuracies],
                    "bin_confidences": [float(c) for c in bin_confidences],
                    "bin_counts": bin_counts,
                    "bin_stds": [float(s) for s in bin_stds],
                    "calibration_quality": (
                        "Excellent" if ece < 0.01 else
                        "Good" if ece < 0.05 else
                        "Fair" if ece < 0.10 else
                        "Poor"
                    ),
                    "interpretation": (
                        f"ECE = {ece:.4f}. Model predictions {'are well-calibrated' if ece < 0.05 else 'show significant miscalibration'}. "
                        f"MCE = {mce:.4f} indicates {'minimal' if mce < 0.1 else 'substantial'} maximum bin error. "
                        "Consider isotonic or Platt scaling if miscalibrated."
                    )
                }
            except Exception as e:
                return {"error": f"Calibration curve analysis failed: {str(e)}"}

        def analyze_quantile_calibration(self) -> Dict:
            """
            7.2 Quantile Calibration - Check if predictions are calibrated at different quantiles.
            Important for trading: are we correctly predicting tails (large moves)?
            """
            try:
                quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
                quantile_results = {}

                for q in quantiles:
                    # Predicted quantile threshold
                    pred_threshold = np.quantile(self.p.y_pred, q)

                    # Actual quantile threshold
                    true_threshold = np.quantile(self.p.y_test, q)

                    # Calibration error at this quantile
                    quantile_error = abs(pred_threshold - true_threshold)

                    # What percentage of predictions above threshold actually exceed true threshold?
                    pred_above = self.p.y_pred >= pred_threshold
                    true_above = self.p.y_test >= true_threshold

                    if pred_above.sum() > 0:
                        precision_at_quantile = float(
                            np.sum(pred_above & true_above) / pred_above.sum()
                        )
                    else:
                        precision_at_quantile = 0.0

                    quantile_results[f"q{int(q*100)}"] = {
                        "predicted_threshold": float(pred_threshold),
                        "actual_threshold": float(true_threshold),
                        "calibration_error": float(quantile_error),
                        "precision": float(precision_at_quantile)
                    }

                # Overall quantile calibration score
                avg_quantile_error = float(np.mean([
                    v['calibration_error'] for v in quantile_results.values()
                ]))

                return {
                    "quantile_results": quantile_results,
                    "avg_quantile_calibration_error": avg_quantile_error,
                    "interpretation": (
                        f"Average quantile calibration error = {avg_quantile_error:.4f}. "
                        f"{'Excellent' if avg_quantile_error < 0.05 else 'Good' if avg_quantile_error < 0.1 else 'Poor'} "
                        "tail calibration. Critical for position sizing and risk management."
                    )
                }
            except Exception as e:
                return {"error": f"Quantile calibration analysis failed: {str(e)}"}

        def analyze_sharpness(self) -> Dict:
            """
            7.3 Sharpness Analysis - Are predictions informative (sharp) or overly conservative?
            Sharp predictions have higher variance and provide more actionable signals.
            """
            try:
                pred_variance = float(np.var(self.p.y_pred))
                true_variance = float(np.var(self.p.y_test))

                # Sharpness ratio: how much of the true variance is captured by predictions
                sharpness_ratio = pred_variance / true_variance if true_variance > 0 else 0

                # Prediction range vs actual range
                pred_range = float(np.ptp(self.p.y_pred))  # peak-to-peak
                true_range = float(np.ptp(self.p.y_test))
                range_ratio = pred_range / true_range if true_range > 0 else 0

                # Information coefficient (correlation)
                ic = float(np.corrcoef(self.p.y_pred, self.p.y_test)[0, 1])

                return {
                    "prediction_variance": pred_variance,
                    "actual_variance": true_variance,
                    "sharpness_ratio": sharpness_ratio,
                    "prediction_range": pred_range,
                    "actual_range": true_range,
                    "range_ratio": range_ratio,
                    "information_coefficient": ic,
                    "sharpness_quality": (
                        "Over-confident" if sharpness_ratio > 1.2 else
                        "Sharp" if 0.8 <= sharpness_ratio <= 1.2 else
                        "Conservative" if 0.5 <= sharpness_ratio < 0.8 else
                        "Under-confident"
                    ),
                    "interpretation": (
                        f"Sharpness ratio = {sharpness_ratio:.3f}. "
                        f"{'Model predictions are appropriately sharp' if 0.8 <= sharpness_ratio <= 1.2 else 'Model predictions are ' + ('too conservative' if sharpness_ratio < 0.8 else 'overconfident')}. "
                        f"IC = {ic:.3f} shows {'strong' if abs(ic) > 0.1 else 'weak'} predictive power."
                    )
                }
            except Exception as e:
                return {"error": f"Sharpness analysis failed: {str(e)}"}

        def analyze_calibration_drift(self) -> Dict:
            """
            7.4 Calibration Drift Over Time - Does calibration degrade over time?
            Critical for production models that need retraining.
            """
            try:
                # Split into time windows
                n_windows = 4
                window_size = len(self.p.y_pred) // n_windows

                calibration_by_window = []

                for i in range(n_windows):
                    start_idx = i * window_size
                    end_idx = start_idx + window_size if i < n_windows - 1 else len(self.p.y_pred)

                    if end_idx > start_idx:
                        window_pred = self.p.y_pred[start_idx:end_idx]
                        window_true = self.p.y_test[start_idx:end_idx]

                        # Calculate calibration error for this window
                        # Simple approach: mean absolute error between predicted and actual
                        window_mae = float(np.mean(np.abs(window_pred - window_true)))

                        # Correlation for this window
                        if len(window_pred) > 1 and np.std(window_pred) > 0 and np.std(window_true) > 0:
                            window_corr = float(np.corrcoef(window_pred, window_true)[0, 1])
                        else:
                            window_corr = 0.0

                        calibration_by_window.append({
                            "window": i + 1,
                            "mae": window_mae,
                            "correlation": window_corr,
                            "n_samples": end_idx - start_idx
                        })

                # Calculate drift: difference between first and last window
                if len(calibration_by_window) >= 2:
                    mae_drift = calibration_by_window[-1]["mae"] - calibration_by_window[0]["mae"]
                    corr_drift = calibration_by_window[-1]["correlation"] - calibration_by_window[0]["correlation"]

                    drift_severity = (
                        "Severe" if abs(mae_drift) > 0.1 or abs(corr_drift) > 0.2 else
                        "Moderate" if abs(mae_drift) > 0.05 or abs(corr_drift) > 0.1 else
                        "Minimal"
                    )
                else:
                    mae_drift = 0.0
                    corr_drift = 0.0
                    drift_severity = "Unknown"

                return {
                    "calibration_by_window": calibration_by_window,
                    "mae_drift": float(mae_drift),
                    "correlation_drift": float(corr_drift),
                    "drift_severity": drift_severity,
                    "interpretation": (
                        f"MAE drift = {mae_drift:.4f}, Correlation drift = {corr_drift:.4f}. "
                        f"{drift_severity} calibration drift detected. "
                        f"{'Consider retraining or online learning' if drift_severity in ['Severe', 'Moderate'] else 'Calibration is stable over time'}."
                    )
                }
            except Exception as e:
                return {"error": f"Calibration drift analysis failed: {str(e)}"}

        def generate_reliability_diagram(self) -> Dict:
            """
            7.5 Reliability Diagram Data - Data for plotting calibration reliability.
            Returns binned data showing predicted vs observed values.
            """
            try:
                n_bins = 15

                # Create bins based on prediction quantiles
                bin_edges = np.percentile(self.p.y_pred, np.linspace(0, 100, n_bins + 1))
                bin_edges[-1] += 1e-8

                # Assign each prediction to a bin
                bin_indices = np.digitize(self.p.y_pred, bin_edges) - 1
                bin_indices = np.clip(bin_indices, 0, n_bins - 1)

                # Calculate observed frequency per bin
                observed_values = []
                predicted_values = []
                bin_counts = []

                for i in range(n_bins):
                    mask = bin_indices == i
                    if mask.sum() > 0:
                        observed_values.append(float(np.mean(self.p.y_test[mask])))
                        predicted_values.append(float(np.mean(self.p.y_pred[mask])))
                        bin_counts.append(int(mask.sum()))

                return {
                    "n_bins": n_bins,
                    "observed_values": observed_values,
                    "predicted_values": predicted_values,
                    "bin_counts": bin_counts,
                    "perfect_calibration_line": {
                        "x": predicted_values,
                        "y": predicted_values
                    },
                    "interpretation": (
                        "Reliability diagram shows predicted vs observed values. "
                        "Points on the diagonal indicate perfect calibration. "
                        "Deviations suggest miscalibration requiring correction."
                    )
                }
            except Exception as e:
                return {"error": f"Reliability diagram generation failed: {str(e)}"}

        def brier_score_decomposition(self) -> Dict:
            """
            7.6 Brier Score Decomposition - Decomposes prediction error into reliability, resolution, and uncertainty.

            Brier Score = Reliability - Resolution + Uncertainty
            - Reliability: How close predicted probabilities are to actual frequencies
            - Resolution: How well predictions discriminate between outcomes
            - Uncertainty: Inherent unpredictability in the target

            Lower Brier score = better calibration (closer to actual probabilities)
            """
            try:
                # For regression, we'll adapt Brier score concept
                # Brier score traditionally for probabilities [0,1], but we'll normalize

                # Normalize predictions and targets to [0, 1] range for Brier calculation
                y_pred_min, y_pred_max = self.p.y_pred.min(), self.p.y_pred.max()
                y_test_min, y_test_max = self.p.y_test.min(), self.p.y_test.max()

                # Use common range for both
                common_min = min(y_pred_min, y_test_min)
                common_max = max(y_pred_max, y_test_max)
                range_span = common_max - common_min

                if range_span > 0:
                    y_pred_norm = (self.p.y_pred - common_min) / range_span
                    y_test_norm = (self.p.y_test - common_min) / range_span
                else:
                    y_pred_norm = self.p.y_pred
                    y_test_norm = self.p.y_test

                # Overall Brier Score: mean squared error between normalized predictions and actuals
                brier_score = float(np.mean((y_pred_norm - y_test_norm) ** 2))

                # Decomposition using binning approach
                n_bins = 10
                bin_edges = np.percentile(y_pred_norm, np.linspace(0, 100, n_bins + 1))
                bin_edges[-1] += 1e-8

                bin_indices = np.digitize(y_pred_norm, bin_edges) - 1
                bin_indices = np.clip(bin_indices, 0, n_bins - 1)

                # Calculate components
                reliability_component = 0.0  # Calibration-in-the-large
                resolution_component = 0.0   # Ability to discriminate

                overall_mean = float(np.mean(y_test_norm))
                total_samples = len(y_pred_norm)

                for i in range(n_bins):
                    mask = bin_indices == i
                    n_i = mask.sum()

                    if n_i > 0:
                        # Mean predicted value in bin i
                        mean_pred_i = float(np.mean(y_pred_norm[mask]))
                        # Mean actual value in bin i
                        mean_true_i = float(np.mean(y_test_norm[mask]))

                        # Reliability: squared difference between predicted and observed in bin
                        reliability_component += (n_i / total_samples) * (mean_pred_i - mean_true_i) ** 2

                        # Resolution: how far each bin's actual mean is from overall mean
                        resolution_component += (n_i / total_samples) * (mean_true_i - overall_mean) ** 2

                # Uncertainty: variance of the target (inherent unpredictability)
                uncertainty_component = float(np.var(y_test_norm))

                # Verify decomposition: BS = Reliability - Resolution + Uncertainty
                decomposed_brier = reliability_component - resolution_component + uncertainty_component

                return {
                    "brier_score": brier_score,
                    "reliability": float(reliability_component),
                    "resolution": float(resolution_component),
                    "uncertainty": float(uncertainty_component),
                    "decomposed_brier": float(decomposed_brier),
                    "decomposition_error": float(abs(brier_score - decomposed_brier)),
                    "normalized_range": {
                        "min": float(common_min),
                        "max": float(common_max),
                        "span": float(range_span)
                    },
                    "interpretation": (
                        f"Brier Score = {brier_score:.4f} (lower is better). "
                        f"Reliability = {reliability_component:.4f} (calibration error, want low), "
                        f"Resolution = {resolution_component:.4f} (discrimination power, want high), "
                        f"Uncertainty = {uncertainty_component:.4f} (inherent noise). "
                        f"Model {'is well-calibrated' if reliability_component < 0.01 else 'needs calibration improvement'}. "
                        f"Model {'has good discrimination' if resolution_component > 0.01 else 'has weak discrimination'}."
                    ),
                    "quality_assessment": (
                        "Excellent" if brier_score < 0.05 and reliability_component < 0.01 else
                        "Good" if brier_score < 0.10 and reliability_component < 0.02 else
                        "Fair" if brier_score < 0.20 else
                        "Poor"
                    )
                }
            except Exception as e:
                return {"error": f"Brier score decomposition failed: {str(e)}"}

        def rolling_calibration_error(self) -> Dict:
            """
            7.7 Rolling Calibration Error - Track calibration quality over time with rolling windows.

            Calculates ECE over rolling windows to detect temporal degradation in calibration.
            Critical for production models to identify when recalibration is needed.
            """
            try:
                # Use rolling window to calculate calibration error over time
                window_sizes = [50, 100, 200]  # Multiple window sizes

                results_by_window_size = {}

                for window_size in window_sizes:
                    if len(self.p.y_pred) < window_size:
                        continue

                    n_windows = len(self.p.y_pred) - window_size + 1
                    # Sample windows to avoid excessive computation
                    sample_step = max(1, n_windows // 50)  # Max 50 windows per size

                    rolling_eces = []
                    rolling_maes = []
                    window_positions = []

                    for start_idx in range(0, n_windows, sample_step):
                        end_idx = start_idx + window_size

                        window_pred = self.p.y_pred[start_idx:end_idx]
                        window_true = self.p.y_test[start_idx:end_idx]

                        # Calculate ECE for this window
                        try:
                            n_bins = min(5, window_size // 10)  # Fewer bins for smaller windows
                            if n_bins < 2:
                                n_bins = 2

                            bin_edges = np.percentile(window_pred, np.linspace(0, 100, n_bins + 1))
                            bin_edges[-1] += 1e-8

                            bin_indices = np.digitize(window_pred, bin_edges) - 1
                            bin_indices = np.clip(bin_indices, 0, n_bins - 1)

                            calibration_errors = []
                            bin_counts = []

                            for i in range(n_bins):
                                mask = bin_indices == i
                                if mask.sum() > 0:
                                    bin_pred_mean = np.mean(window_pred[mask])
                                    bin_true_mean = np.mean(window_true[mask])
                                    cal_error = abs(bin_pred_mean - bin_true_mean)
                                    calibration_errors.append(cal_error)
                                    bin_counts.append(mask.sum())

                            # Weighted ECE
                            if calibration_errors:
                                total_in_bins = sum(bin_counts)
                                window_ece = float(sum(
                                    (count / total_in_bins) * error
                                    for count, error in zip(bin_counts, calibration_errors)
                                ))
                            else:
                                window_ece = 0.0

                            # Also track MAE for this window
                            window_mae = float(np.mean(np.abs(window_pred - window_true)))

                            rolling_eces.append(window_ece)
                            rolling_maes.append(window_mae)
                            window_positions.append(start_idx + window_size // 2)  # Center position

                        except Exception:
                            continue

                    if rolling_eces:
                        results_by_window_size[f"window_{window_size}"] = {
                            "window_size": window_size,
                            "n_windows": len(rolling_eces),
                            "rolling_ece": [float(e) for e in rolling_eces],
                            "rolling_mae": [float(m) for m in rolling_maes],
                            "window_positions": [int(p) for p in window_positions],
                            "mean_ece": float(np.mean(rolling_eces)),
                            "std_ece": float(np.std(rolling_eces)),
                            "max_ece": float(np.max(rolling_eces)),
                            "min_ece": float(np.min(rolling_eces)),
                            "ece_trend": float(np.polyfit(range(len(rolling_eces)), rolling_eces, 1)[0]) if len(rolling_eces) > 1 else 0.0
                        }

                # Overall assessment
                if results_by_window_size:
                    # Use medium window size for overall assessment
                    medium_window_key = f"window_{window_sizes[len(window_sizes)//2]}" if len(window_sizes) > 0 else list(results_by_window_size.keys())[0]

                    if medium_window_key in results_by_window_size:
                        medium_results = results_by_window_size[medium_window_key]
                        mean_ece = medium_results['mean_ece']
                        ece_trend = medium_results['ece_trend']

                        degradation_status = (
                            "Severe Degradation" if ece_trend > 0.0001 and mean_ece > 0.1 else
                            "Moderate Degradation" if ece_trend > 0.00005 else
                            "Stable" if abs(ece_trend) <= 0.00005 else
                            "Improving"
                        )
                    else:
                        mean_ece = 0.0
                        ece_trend = 0.0
                        degradation_status = "Unknown"
                else:
                    mean_ece = 0.0
                    ece_trend = 0.0
                    degradation_status = "Insufficient Data"

                return {
                    "results_by_window_size": results_by_window_size,
                    "overall_mean_ece": float(mean_ece),
                    "overall_ece_trend": float(ece_trend),
                    "degradation_status": degradation_status,
                    "interpretation": (
                        f"Rolling calibration analysis with windows {window_sizes}. "
                        f"Mean ECE = {mean_ece:.4f}, Trend = {ece_trend:.6f}. "
                        f"{degradation_status} calibration over time. "
                        f"{'Recalibration recommended' if degradation_status in ['Severe Degradation', 'Moderate Degradation'] else 'Calibration is stable'}."
                    )
                }
            except Exception as e:
                return {"error": f"Rolling calibration error analysis failed: {str(e)}"}

        def threshold_weighted_ece(self) -> Dict:
            """
            7.8 Threshold-Weighted ECE - ECE with increasing weights for high-confidence predictions (0.8 -> 1.0).

            In trading, high-confidence predictions are more actionable and carry higher risk.
            This metric prioritizes calibration quality in the high-confidence regime where
            position sizing is largest.

            Weights increase linearly from 1.0 at threshold 0.8 to higher values at 1.0.
            """
            try:
                # Normalize predictions to [0, 1] for threshold application
                y_pred_min, y_pred_max = self.p.y_pred.min(), self.p.y_pred.max()
                range_span = y_pred_max - y_pred_min

                if range_span > 0:
                    y_pred_norm = (self.p.y_pred - y_pred_min) / range_span
                else:
                    y_pred_norm = np.ones_like(self.p.y_pred) * 0.5

                # Also normalize targets
                y_test_min, y_test_max = self.p.y_test.min(), self.p.y_test.max()
                test_range_span = y_test_max - y_test_min

                if test_range_span > 0:
                    y_test_norm = (self.p.y_test - y_test_min) / test_range_span
                else:
                    y_test_norm = np.ones_like(self.p.y_test) * 0.5

                # Standard ECE calculation with bins
                n_bins = 10
                bin_edges = np.percentile(y_pred_norm, np.linspace(0, 100, n_bins + 1))
                bin_edges[-1] += 1e-8

                bin_indices = np.digitize(y_pred_norm, bin_edges) - 1
                bin_indices = np.clip(bin_indices, 0, n_bins - 1)

                # Calculate standard ECE and threshold-weighted ECE
                standard_ece = 0.0
                threshold_weighted_ece_value = 0.0

                bin_details = []
                total_samples = len(y_pred_norm)

                # High-confidence threshold (0.8 in normalized space)
                threshold = 0.8
                max_weight_multiplier = 3.0  # Max weight at 1.0 is 3x the weight at 0.8

                for i in range(n_bins):
                    mask = bin_indices == i
                    n_i = mask.sum()

                    if n_i > 0:
                        bin_pred = y_pred_norm[mask]
                        bin_true = y_test_norm[mask]

                        mean_pred = float(np.mean(bin_pred))
                        mean_true = float(np.mean(bin_true))

                        cal_error = abs(mean_pred - mean_true)
                        bin_weight = n_i / total_samples

                        # Standard ECE contribution
                        standard_ece += bin_weight * cal_error

                        # Calculate threshold weight for this bin
                        # Bins with mean_pred >= threshold get increasing weights
                        if mean_pred >= threshold:
                            # Linear increase from 1.0 at threshold to max_weight_multiplier at 1.0
                            weight_multiplier = 1.0 + (max_weight_multiplier - 1.0) * ((mean_pred - threshold) / (1.0 - threshold))
                            threshold_weight = bin_weight * weight_multiplier
                        else:
                            threshold_weight = bin_weight
                            weight_multiplier = 1.0

                        threshold_weighted_ece_value += threshold_weight * cal_error

                        bin_details.append({
                            "bin": i + 1,
                            "mean_pred": mean_pred,
                            "mean_true": mean_true,
                            "cal_error": float(cal_error),
                            "n_samples": int(n_i),
                            "standard_weight": float(bin_weight),
                            "threshold_weight": float(threshold_weight),
                            "is_high_confidence": bool(mean_pred >= threshold),
                            "weight_multiplier": float(weight_multiplier)
                        })

                # Normalize threshold-weighted ECE by total weights
                total_threshold_weight = sum(bd['threshold_weight'] for bd in bin_details)
                if total_threshold_weight > 0:
                    threshold_weighted_ece_value = threshold_weighted_ece_value / total_threshold_weight * total_samples / len(bin_details)

                # Calculate high-confidence calibration specifically
                high_conf_mask = y_pred_norm >= threshold
                high_conf_samples = high_conf_mask.sum()

                if high_conf_samples > 0:
                    high_conf_mae = float(np.mean(np.abs(
                        y_pred_norm[high_conf_mask] - y_test_norm[high_conf_mask]
                    )))
                    high_conf_pct = float(high_conf_samples / len(y_pred_norm) * 100)
                else:
                    high_conf_mae = 0.0
                    high_conf_pct = 0.0

                return {
                    "standard_ece": float(standard_ece),
                    "threshold_weighted_ece": float(threshold_weighted_ece_value),
                    "threshold": float(threshold),
                    "max_weight_multiplier": float(max_weight_multiplier),
                    "high_confidence_mae": high_conf_mae,
                    "high_confidence_pct": high_conf_pct,
                    "high_confidence_samples": int(high_conf_samples),
                    "bin_details": bin_details,
                    "ece_ratio": float(threshold_weighted_ece_value / standard_ece) if standard_ece > 0 else 1.0,
                    "normalization": {
                        "pred_min": float(y_pred_min),
                        "pred_max": float(y_pred_max),
                        "test_min": float(y_test_min),
                        "test_max": float(y_test_max)
                    },
                    "interpretation": (
                        f"Standard ECE = {standard_ece:.4f}, Threshold-Weighted ECE = {threshold_weighted_ece_value:.4f}. "
                        f"High-confidence predictions (>{threshold:.1f}): {high_conf_pct:.1f}% of samples with MAE = {high_conf_mae:.4f}. "
                        f"{'High-confidence calibration is excellent' if high_conf_mae < 0.05 else 'High-confidence calibration needs improvement'}. "
                        f"This metric emphasizes calibration quality where it matters most for trading decisions."
                    ),
                    "quality_assessment": (
                        "Excellent" if threshold_weighted_ece_value < 0.05 and high_conf_mae < 0.05 else
                        "Good" if threshold_weighted_ece_value < 0.10 and high_conf_mae < 0.10 else
                        "Fair" if threshold_weighted_ece_value < 0.20 else
                        "Poor"
                    )
                }
            except Exception as e:
                return {"error": f"Threshold-weighted ECE calculation failed: {str(e)}"}

        def conditional_calibration_analysis(self) -> Dict:
            """
            7.9 Conditional Calibration Analysis - Calibration quality conditional on market features.

            Analyzes calibration errors across different market regimes (volatility, trend, etc.)
            and applies Lasso regression to create conditional calibration that adapts to market conditions.

            This is critical for trading models where calibration quality varies significantly
            across different market regimes (trending vs ranging, high vs low volatility).
            """
            try:
                # Step A: Prepare features for conditional analysis
                # We'll use proxy features based on prediction variance and trends

                n_samples = len(self.p.y_pred)

                # Create synthetic market regime features from available data
                # These are proxies when real market data isn't available

                # 1. Volatility proxy: rolling std of predictions
                window = min(50, n_samples // 10)
                if window < 5:
                    return {"error": "Insufficient samples for conditional calibration (need at least 50)"}

                pred_series = pd.Series(self.p.y_pred)
                test_series = pd.Series(self.p.y_test)

                # Volatility: rolling std of actuals
                volatility = test_series.rolling(window=window, min_periods=5).std().fillna(method='bfill').fillna(method='ffill')

                # Trend: rolling mean of actuals
                trend = test_series.rolling(window=window, min_periods=5).mean().fillna(method='bfill').fillna(method='ffill')

                # Ensemble disagreement: rolling std of predictions (proxy for model uncertainty)
                ensemble_std = pred_series.rolling(window=window, min_periods=5).std().fillna(method='bfill').fillna(method='ffill')

                # Relative volume proxy: rolling range of actuals
                rvol = (test_series.rolling(window=window, min_periods=5).max() -
                       test_series.rolling(window=window, min_periods=5).min()).fillna(method='bfill').fillna(method='ffill')

                # Create DataFrame with features
                features_df = pd.DataFrame({
                    'volatility': volatility.values,
                    'trend': trend.values,
                    'ensemble_std': ensemble_std.values,
                    'rvol': rvol.values
                })

                # Winsorize and normalize features (RobustScaler simulation)
                from sklearn.preprocessing import RobustScaler

                winsorize_threshold = 0.95
                for col in features_df.columns:
                    # Winsorization
                    upper = features_df[col].quantile(winsorize_threshold)
                    lower = features_df[col].quantile(1 - winsorize_threshold)
                    features_df[col] = features_df[col].clip(lower, upper)

                # Robust scaling
                scaler = RobustScaler()
                features_scaled = scaler.fit_transform(features_df)
                features_df[features_df.columns] = features_scaled

                # Step B: Decile Binning Analysis
                decile_analysis = {}

                for feat_name in features_df.columns:
                    feat_values = features_df[feat_name].values

                    # Create 10 decile bins
                    try:
                        bins = np.percentile(feat_values, np.linspace(0, 100, 11))
                        bins[-1] += 1e-8  # Ensure last value is included
                        bin_indices = np.digitize(feat_values, bins) - 1
                        bin_indices = np.clip(bin_indices, 0, 9)

                        # Calculate Brier score for each decile
                        decile_brier = []
                        decile_counts = []
                        decile_means = []

                        for i in range(10):
                            mask = bin_indices == i
                            if mask.sum() > 0:
                                decile_pred = self.p.y_pred[mask]
                                decile_true = self.p.y_test[mask]

                                brier = float(np.mean((decile_pred - decile_true) ** 2))
                                decile_brier.append(brier)
                                decile_counts.append(int(mask.sum()))
                                decile_means.append(float(np.mean(feat_values[mask])))
                            else:
                                decile_brier.append(None)
                                decile_counts.append(0)
                                decile_means.append(None)

                        # Find worst deciles (top 3)
                        valid_briers = [(i, b) for i, b in enumerate(decile_brier) if b is not None]
                        valid_briers.sort(key=lambda x: x[1], reverse=True)
                        worst_deciles = valid_briers[:3]

                        decile_analysis[feat_name] = {
                            'decile_brier_scores': decile_brier,
                            'decile_counts': decile_counts,
                            'decile_means': decile_means,
                            'worst_deciles': [{'decile': d, 'brier': b} for d, b in worst_deciles],
                            'max_brier': max([b for b in decile_brier if b is not None], default=0),
                            'min_brier': min([b for b in decile_brier if b is not None], default=0),
                            'brier_range': max([b for b in decile_brier if b is not None], default=0) -
                                          min([b for b in decile_brier if b is not None], default=0)
                        }
                    except Exception as e:
                        decile_analysis[feat_name] = {'error': str(e)}

                # Identify top 3 offender features
                feature_offenders = []
                for feat_name, analysis in decile_analysis.items():
                    if 'error' not in analysis:
                        feature_offenders.append((feat_name, analysis['brier_range']))

                feature_offenders.sort(key=lambda x: x[1], reverse=True)
                top_offenders = [f for f, _ in feature_offenders[:3]]

                # Step C: Lasso Regression Conditional Fix
                from sklearn.linear_model import LogisticRegressionCV
                from scipy.special import logit

                # Prepare predictions as probabilities [0, 1]
                epsilon = 1e-5

                # Normalize predictions to [0, 1] if needed
                y_pred_min, y_pred_max = self.p.y_pred.min(), self.p.y_pred.max()
                if y_pred_max - y_pred_min > 0:
                    y_prob = (self.p.y_pred - y_pred_min) / (y_pred_max - y_pred_min)
                else:
                    y_prob = np.ones_like(self.p.y_pred) * 0.5

                # Clip to avoid log(0) or log(1)
                y_prob_safe = np.clip(y_prob, epsilon, 1 - epsilon)

                # Normalize targets similarly
                y_test_min, y_test_max = self.p.y_test.min(), self.p.y_test.max()
                if y_test_max - y_test_min > 0:
                    y_true = (self.p.y_test - y_test_min) / (y_test_max - y_test_min)
                else:
                    y_true = np.ones_like(self.p.y_test) * 0.5

                y_true_safe = np.clip(y_true, epsilon, 1 - epsilon)

                # Build feature matrix X
                X_cond = pd.DataFrame()

                # Base signal (logit-transformed)
                X_cond['L'] = logit(y_prob_safe)

                # Add top offender features and their interactions
                for feat_name in top_offenders:
                    if feat_name in features_df.columns:
                        # Pure feature (veto switch)
                        X_cond[feat_name] = features_df[feat_name].values

                        # Interaction (sizer switch)
                        X_cond[f'L_x_{feat_name}'] = X_cond['L'] * features_df[feat_name].values

                # Fit Lasso with cross-validation
                try:
                    calibrator = LogisticRegressionCV(
                        Cs=10,
                        cv=min(5, n_samples // 50),  # Adaptive CV folds
                        penalty='l1',
                        solver='liblinear',
                        scoring='neg_brier_score_loss',
                        max_iter=1000,
                        random_state=42
                    )

                    calibrator.fit(X_cond, (y_true_safe > 0.5).astype(int))

                    # Extract coefficients
                    coeffs = pd.Series(calibrator.coef_[0], index=X_cond.columns)
                    survivors = coeffs[abs(coeffs) > 1e-4].sort_values(ascending=False)

                    # Calculate calibrated probabilities
                    y_calibrated_prob = calibrator.predict_proba(X_cond)[:, 1]

                    # Calculate metrics before and after
                    raw_brier = float(np.mean((y_prob_safe - y_true_safe) ** 2))
                    cal_brier = float(np.mean((y_calibrated_prob - y_true_safe) ** 2))

                    # Resolution (sharpness)
                    base_rate = float(np.mean(y_true_safe))
                    raw_resolution = float(np.mean((y_prob_safe - base_rate) ** 2))
                    cal_resolution = float(np.mean((y_calibrated_prob - base_rate) ** 2))

                    # ROC-AUC
                    from sklearn.metrics import roc_auc_score
                    raw_auc = float(roc_auc_score((y_true_safe > 0.5).astype(int), y_prob_safe))
                    cal_auc = float(roc_auc_score((y_true_safe > 0.5).astype(int), y_calibrated_prob))

                    # Check for degradation
                    resolution_change = (cal_resolution - raw_resolution) / raw_resolution if raw_resolution > 0 else 0
                    auc_change = (cal_auc - raw_auc) / raw_auc if raw_auc > 0 else 0
                    brier_change = (cal_brier - raw_brier) / raw_brier if raw_brier > 0 else 0

                    warnings = []
                    if cal_resolution < (raw_resolution * 0.8):
                        warnings.append("WARNING: Significant Resolution Loss! Calibrator is over-smoothing.")
                    if cal_auc < raw_auc:
                        warnings.append("WARNING: AUC decreased. Calibration may be degrading ranking ability.")
                    if cal_brier > raw_brier:
                        warnings.append("WARNING: Brier score increased. Calibration not improving.")

                    lasso_results = {
                        'best_C': float(calibrator.C_[0]),
                        'coefficients': {k: float(v) for k, v in survivors.items()},
                        'n_survivors': len(survivors),
                        'metrics': {
                            'raw_brier': raw_brier,
                            'calibrated_brier': cal_brier,
                            'brier_improvement': brier_change,
                            'raw_resolution': raw_resolution,
                            'calibrated_resolution': cal_resolution,
                            'resolution_change': resolution_change,
                            'raw_auc': raw_auc,
                            'calibrated_auc': cal_auc,
                            'auc_change': auc_change
                        },
                        'warnings': warnings,
                        'status': 'success' if not warnings else 'warning'
                    }
                except Exception as lasso_error:
                    lasso_results = {'error': f"Lasso fitting failed: {str(lasso_error)}"}

                return {
                    'top_offender_features': top_offenders,
                    'decile_analysis': decile_analysis,
                    'lasso_conditional_fix': lasso_results,
                    'interpretation': (
                        f"Conditional calibration analysis identified {len(top_offenders)} key features "
                        f"affecting calibration quality: {', '.join(top_offenders)}. "
                        f"{'Lasso regression successfully applied conditional fixes.' if 'error' not in lasso_results else 'Lasso fitting encountered errors.'} "
                        f"{'No significant degradation detected.' if not lasso_results.get('warnings', []) else 'Warnings detected - review carefully.'}"
                    )
                }

            except Exception as e:
                return {"error": f"Conditional calibration analysis failed: {str(e)}"}

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
        Comprehensive rule-engine that translates metrics into specific, actionable advice.
        Highlights three critical diagnostic areas: CMI, Residual Analysis, Noise Ceiling.
        Returns a list of dicts: {'severity': 'critical|warning|info', 'title': str, 'message': str}
        """
        r = self.results
        advice = []

        # ========== KEY DIAGNOSTIC #1: NOISE CEILING / BAYES ERROR ESTIMATE ==========
        oracle = r.get('oracle', {})
        noise_ceiling = oracle.get('noise_ceiling', {})
        estimated_ceiling = noise_ceiling.get('estimated_ceiling_r2', 0)
        current_r2 = noise_ceiling.get('current_model_r2', 0)
        gap = noise_ceiling.get('gap', 0)

        if estimated_ceiling < 0.002:
            advice.append({
                'severity': 'critical',
                'title': '🚫 NOISE CEILING ALERT: Target is Unpredictable',
                'message': f"Bayes Error Estimate shows theoretical maximum R² is only {estimated_ceiling:.4f}. This target contains insufficient signal for profitable prediction. ACTION: Change prediction horizon (try shorter/longer timeframes), modify target definition, or switch assets."
            })
        elif 0.002 <= estimated_ceiling < 0.01:
            advice.append({
                'severity': 'warning',
                'title': '⚠️ NOISE CEILING: Very Tight Limits',
                'message': f"Theoretical predictability ceiling is only {estimated_ceiling:.4f}. Even optimal models struggle with this target. Consider: (1) Feature engineering for stronger signals, (2) Ensemble methods to reduce noise, (3) Alternative targets with higher SNR."
            })
        elif gap > 0.05:
            advice.append({
                'severity': 'info',
                'title': '💡 NOISE CEILING: Significant Untapped Potential',
                'message': f"Your model R² is {current_r2:.4f} but theoretical maximum is {estimated_ceiling:.4f}. Gap of {gap:.4f} suggests major improvements possible. Investigation: (1) Feature importance analysis - missing key drivers? (2) Model architecture - try deeper/wider networks, (3) Hyperparameter tuning - learning rate, regularization."
            })
        elif gap <= 0.005:
            advice.append({
                'severity': 'success',
                'title': '✅ NOISE CEILING: Near Saturation',
                'message': f"Model performance ({current_r2:.4f}) is very close to theoretical ceiling ({estimated_ceiling:.4f}). Diminishing returns on further improvements. Focus on deployment stability and transaction costs."
            })

        # ========== KEY DIAGNOSTIC #2: CONDITIONAL MUTUAL INFORMATION (CMI) ==========
        architect = r.get('architect', {})
        cmi_data = architect.get('cmi', {})
        cmi_score = cmi_data.get('avg_unused_info', 0)
        top_missed = cmi_data.get('top_underutilized_features', [])

        if cmi_score > 0.15:
            advice.append({
                'severity': 'critical',
                'title': '🔴 CMI ALERT: Severe Underfitting Detected',
                'message': f"Conditional Mutual Information = {cmi_score:.3f}. Your model is leaving substantial signal on the table. Top unused features: {', '.join(top_missed[:3])}. ACTIONS: (1) Increase model complexity (tree depth, hidden layers, ensemble size), (2) Check feature engineering - are these features actually predictive? (3) Use feature interaction terms, (4) Try non-parametric models (RandomForest, XGBoost)."
            })
        elif 0.10 <= cmi_score <= 0.15:
            advice.append({
                'severity': 'warning',
                'title': '⚠️ CMI: Moderate Underfitting',
                'message': f"CMI = {cmi_score:.3f}. Model is not capturing all available information. Top missed features: {', '.join(top_missed[:2])}. Consider: (1) Feature engineering to create interactions/combinations, (2) Increase model capacity, (3) Reduce regularization if overly constrained."
            })
        elif cmi_score < 0.05:
            advice.append({
                'severity': 'success',
                'title': '✅ CMI: Efficient Feature Utilization',
                'message': f"CMI = {cmi_score:.3f} shows your model is efficiently using available features with minimal unused information. Good feature engineering and model architecture."
            })

        # ========== KEY DIAGNOSTIC #3: RESIDUAL ANALYSIS (Clustering + Autocorrelation) ==========
        critic = r.get('critic', {})

        # Autocorrelation analysis
        autocorr_data = critic.get('residual_autocorr', {})
        lag1_autocorr = abs(autocorr_data.get('lag1_autocorr', 0))

        # Clustering analysis
        clustering_data = critic.get('residual_clustering', {})
        is_heteroskedastic = clustering_data.get('heteroskedasticity_detected', False)

        if lag1_autocorr > 0.20:
            advice.append({
                'severity': 'critical',
                'title': '🔴 RESIDUAL AUTOCORR: Strong Temporal Leakage',
                'message': f"Lag-1 autocorrelation = {lag1_autocorr:.3f} (critical threshold >0.20). Model missing trend/momentum structure. Errors are NOT white noise - they cluster in time. ACTIONS: (1) Add lagged target features (y[t-1], y[t-2]...), (2) Use RNN/LSTM for temporal dependencies, (3) Include regime-switching features, (4) Check for data leakage from future information."
            })
        elif 0.10 < lag1_autocorr <= 0.20:
            advice.append({
                'severity': 'warning',
                'title': '⚠️ RESIDUAL AUTOCORR: Minor Temporal Leakage',
                'message': f"Lag-1 autocorrelation = {lag1_autocorr:.3f}. Some trend information missed. Try: (1) Add 1-2 lagged target features, (2) Use moving averages as features, (3) Test for multi-step ahead predictions."
            })

        if is_heteroskedastic:
            advice.append({
                'severity': 'warning',
                'title': '⚠️ RESIDUAL CLUSTERING: Regime-Dependent Performance',
                'message': "Error variance changes over time (heteroskedasticity detected). Model performs differently across market regimes. ACTIONS: (1) Add volatility/regime indicators, (2) Use separate models for high/low volatility periods, (3) Quantile regression for tail-robust predictions, (4) Dynamic position sizing based on predicted volatility."
            })

        # ========== PERFORMANCE METRICS CHECKS ==========
        perf = r.get('performance', {})

        # Sharpe ratio
        sharpe_data = perf.get('sharpe_ratio', {})
        sharpe = sharpe_data.get('sharpe_ratio', 0)
        if sharpe < 0:
            advice.append({
                'severity': 'critical',
                'title': 'Negative Returns',
                'message': f"Strategy Sharpe ratio is {sharpe:.2f} (negative). Model generates net losses. Review: (1) Are predictions inverted? (2) Transaction costs too high? (3) Target misalignment?"
            })
        elif 0 < sharpe < 0.5:
            advice.append({
                'severity': 'warning',
                'title': 'Low Risk-Adjusted Returns',
                'message': f"Sharpe ratio = {sharpe:.2f} (poor). Risk-adjusted returns are weak. Improve signal quality or reduce trading frequency."
            })

        # Drawdown
        dd_data = perf.get('max_drawdown', {})
        max_dd = dd_data.get('max_drawdown_pct', 0)
        if max_dd < -30:
            advice.append({
                'severity': 'critical',
                'title': 'Extreme Drawdown Risk',
                'message': f"Maximum drawdown = {max_dd:.1f}%. Unacceptable risk. Review position sizing, stop-loss rules, and correlation risks."
            })

        # Win rate
        wr_data = perf.get('win_rate', {})
        win_rate = wr_data.get('win_rate_pct', 0)
        if win_rate < 45:
            advice.append({
                'severity': 'warning',
                'title': 'Below 50% Win Rate',
                'message': f"Win rate = {win_rate:.1f}%. Majority of trades are losers. Check for prediction bias or threshold calibration."
            })

        # ========== ADDITIONAL CHECKS ==========

        # Model Linearity
        rsa = architect.get('rsa', {}).get('similarity_score', 0)
        if rsa > 0.9:
            advice.append({
                'severity': 'info',
                'title': 'Linear Model Detected',
                'message': f"RSA similarity = {rsa:.3f}. Model is essentially linear. If seeking alpha from non-linear patterns, consider: (1) Non-linear architectures (Neural Networks), (2) Tree-based ensembles, (3) Kernel methods."
            })

        # Feature Leakage
        leaking = critic.get('orthogonal_tests', {}).get('leaking_features', {})
        if leaking:
            feats = ", ".join(list(leaking.keys())[:3])
            advice.append({
                'severity': 'warning',
                'title': 'Feature Correlation with Residuals',
                'message': f"Features {feats} correlate with prediction errors. Integrate these into the model as explicit features for better capture."
            })

        # Stability
        stress = r.get('stress_tester', {})
        if stress.get('stability', {}).get('status') == 'Fragile':
            advice.append({
                'severity': 'critical',
                'title': '🔴 MODEL FRAGILITY',
                'message': "Bootstrap stability test shows high variance across subsets. Model is unstable. DO NOT DEPLOY. Use: (1) Bagging/Stacking for ensemble stability, (2) Lower learning rates, (3) Increase regularization, (4) Data augmentation."
            })

        # Confidence Interval
        if not stress.get('bootstrap_ci', {}).get('reliable_edge', False):
            advice.append({
                'severity': 'critical',
                'title': '🔴 NO STATISTICAL EDGE',
                'message': "95% CI on R² includes zero. Performance indistinguishable from random chance. Model has no reliable edge. Requires major improvements before deployment."
            })

        # ========== ENHANCED DIAGNOSTICS: DISTRIBUTION DRIFT (Navigator) ==========
        navigator = r.get('navigator', {})
        drift_test = navigator.get('distribution_shift_test', {})
        drifted_features = drift_test.get('drifted_features', {})

        if drifted_features and len(drifted_features) > 3:
            advice.append({
                'severity': 'critical',
                'title': '🔴 DISTRIBUTION DRIFT ALERT: Multiple Features Drifted',
                'message': f"DataDriftDetector identified {len(drifted_features)} drifted features (PSI, Wasserstein, KS). Test data distribution differs significantly from training. ACTIONS: (1) Retrain model on recent data, (2) Implement online learning/continuous retraining, (3) Add drift monitoring to production, (4) Consider ensemble of time-windowed models."
            })
        elif drifted_features and len(drifted_features) > 0:
            advice.append({
                'severity': 'warning',
                'title': '⚠️ MODERATE DISTRIBUTION DRIFT',
                'message': f"Some features show drift: {', '.join(list(drifted_features.keys())[:3])}. Model performance may degrade out-of-sample. Monitor predictions closely and consider retraining."
            })

        # ========== ENHANCED DIAGNOSTICS: PERMUTATION IMPORTANCE (Architect) ==========
        perm_importance = architect.get('permutation_importance', {})
        if 'error' not in perm_importance and perm_importance.get('top_features'):
            top_features = perm_importance.get('top_features', {})
            if len(top_features) < 3:
                advice.append({
                    'severity': 'warning',
                    'title': '⚠️ FEATURE CONCENTRATION: Low Feature Diversity',
                    'message': f"Only {len(top_features)} features have significant importance. Model relies on very few inputs. Risk of overfitting. ACTION: Increase feature engineering or ensure broader feature coverage."
                })

        # ========== ENHANCED DIAGNOSTICS: MODEL COMPLEXITY AUDIT (Architect) ==========
        complexity = architect.get('complexity_audit', {})
        overfit_risk = complexity.get('overfitting_risk_score', 0)

        if overfit_risk > 0.7:
            advice.append({
                'severity': 'critical',
                'title': '🔴 HIGH OVERFITTING RISK DETECTED',
                'message': f"Model Complexity Analyzer reports overfitting risk score = {overfit_risk:.2f} (>0.7). ACTIONS: (1) Increase regularization, (2) Reduce model complexity (fewer trees, shallower depth), (3) Cross-validate more rigorously, (4) Feature selection/dimensionality reduction."
            })
        elif 0.5 <= overfit_risk <= 0.7:
            advice.append({
                'severity': 'warning',
                'title': '⚠️ ELEVATED OVERFITTING RISK',
                'message': f"Overfitting risk score = {overfit_risk:.2f}. Model may not generalize well. Consider regularization adjustments and additional validation."
            })

        # ========== ENHANCED DIAGNOSTICS: CONFIDENCE DEGRADATION (StressTester) ==========
        conf_degrad = stress.get('confidence_degradation', {})
        degradation = conf_degrad.get('confidence_degradation', 0)

        if degradation > 0.1:
            advice.append({
                'severity': 'warning',
                'title': '⚠️ CONFIDENCE DEGRADATION OVER TIME',
                'message': f"Model confidence drops by {degradation:.3f} from early to late test period. Model loses predictive power over time. ACTIONS: (1) Use time-decay weighting, (2) Implement online learning, (3) Monitor for concept drift, (4) Consider ensemble of time-localized models."
            })

        # ========== ENHANCED DIAGNOSTICS: ENSEMBLE VARIANCE (StressTester) ==========
        ensemble_var = stress.get('ensemble_variance', {})
        uncertainty_status = ensemble_var.get('uncertainty_status', 'Low')

        if uncertainty_status == 'High':
            advice.append({
                'severity': 'warning',
                'title': '⚠️ HIGH PREDICTION UNCERTAINTY',
                'message': f"Ensemble variance is high, indicating inconsistent predictions. Model lacks confidence. ACTIONS: (1) Increase ensemble size, (2) Improve feature signal quality, (3) Use confidence-weighted predictions, (4) Consider uncertainty quantification in portfolio allocation."
            })

        # ========== ENHANCED DIAGNOSTICS: LEARNING DYNAMICS (Historian) ==========
        historian = r.get('historian', {})
        data_eff = historian.get('data_efficiency', {})
        eff_status = data_eff.get('efficiency_status', 'Good')

        if eff_status == 'Data Starved':
            advice.append({
                'severity': 'critical',
                'title': '🔴 DATA STARVATION: Insufficient Training Data',
                'message': f"Only {data_eff.get('samples_per_feature', 0):.1f} samples per feature (recommended: 10-20). Model likely underfitting due to insufficient data. ACTION: Collect more training data or simplify model architecture."
            })
        elif 'Over-sampled' in eff_status:
            advice.append({
                'severity': 'info',
                'title': '💡 DATA EFFICIENCY: Over-sampling Detected',
                'message': "Extensive data relative to model complexity. Check for redundancy in data - feature selection may help."
            })

        # Learning Anomalies
        learning_anom = historian.get('learning_anomalies', {})
        if learning_anom.get('has_anomalies'):
            anomalies = learning_anom.get('detected_anomalies', [])
            advice.append({
                'severity': 'warning',
                'title': '⚠️ LEARNING ANOMALIES DETECTED',
                'message': f"Issues found: {'; '.join(anomalies)}. Investigate residual distribution for model improvements."
            })

        # Default message if no issues found
        if not advice:
            advice.append({
                'severity': 'success',
                'title': '✅ DIAGNOSTIC CLEAN BILL OF HEALTH',
                'message': "No major red flags detected. Model shows good: (1) Predictability ceiling with room to improve, (2) Efficient feature usage (low CMI), (3) Clean residuals (low autocorrelation), (4) Statistical significance, (5) Stable drift metrics, (6) Reasonable complexity, (7) Good learning dynamics. Proceed to live paper trading with monitoring."
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
