import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.optimize import minimize
from typing import List, Tuple, Dict, Any, Optional

class StructuralRegimeGMM:
    def __init__(self, n_regimes=4):
        self.n_regimes = n_regimes
        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(n_components=n_regimes, covariance_type='full', random_state=42)

    def get_structural_features(self, df):
        """Derives the 3 pillars of market context: Vol, Volume, Trend."""
        f = pd.DataFrame(index=df.index)

        # Ensure we have required columns
        if 'close' not in df.columns or 'volume' not in df.columns:
             # Try case insensitive
             cols = {c.lower(): c for c in df.columns}
             if 'close' in cols and 'volume' in cols:
                 df = df.rename(columns={cols['close']: 'close', cols['volume']: 'volume'})
             else:
                 # Cannot compute
                 return pd.DataFrame()

        # 1. Volatility (Normalized)
        log_ret = np.log(df['close'] / df['close'].shift(1))
        f['volatility'] = log_ret.rolling(20).std()

        # 2. Volume Intensity (Relative to 50-bar average)
        vol_rolling_std = df['volume'].rolling(50).std() + 1e-9
        f['volume_z'] = (df['volume'] - df['volume'].rolling(50).mean()) / vol_rolling_std

        # 3. Trend Strength (Absolute return over 20 bars / volatility)
        denom = df['close'].rolling(20).std() * np.sqrt(20) + 1e-9
        f['trend_strength'] = np.abs(df['close'].diff(20)) / denom

        return f.dropna()

    def fit_predict(self, df):
        features = self.get_structural_features(df)
        if features.empty or len(features) < self.n_regimes * 2:
            return [], np.array([])

        scaled_features = self.scaler.fit_transform(features)

        # Fit and predict the 4 regimes
        clusters = self.gmm.fit_predict(scaled_features)

        # Create the env_indices list for IRM
        env_indices = []
        for i in range(self.n_regimes):
            # Map cluster assignments back to the original dataframe index positions
            cluster_indices = np.where(clusters == i)[0]

            # Aligning with the dropped NaNs from feature engineering
            # Using index search (can be optimized if needed, but robust)
            actual_indices = [df.index.get_loc(features.index[idx]) for idx in cluster_indices]
            env_indices.append(np.array(actual_indices))

        return env_indices, clusters

class IRMv1HuberRegressor:
    def __init__(self, irm_lambda=25.0, alpha=0.1, huber_epsilon=1.1):
        self.irm_lambda = irm_lambda
        self.alpha = alpha
        self.huber_epsilon = huber_epsilon
        self.coef_ = None

    def _huber_loss_and_grad(self, w, X, y):
        """Standard Huber Loss and its Gradient."""
        errors = (X @ w) - y
        abs_errors = np.abs(errors)

        # Loss calculation
        quadratic_mask = abs_errors <= self.huber_epsilon
        loss = np.where(quadratic_mask, 0.5 * errors**2,
                        self.huber_epsilon * (abs_errors - 0.5 * self.huber_epsilon))

        # Gradient calculation
        grad_mult = np.where(quadratic_mask, errors,
                             self.huber_epsilon * np.sign(errors))
        grad = (X.T @ grad_mult) / len(y)

        return np.mean(loss), grad

    def _objective(self, w, envs):
        """IRM-v1 Objective: ERM + Lambda * Var(Gradients) + L2."""
        total_loss = 0
        penalty = 0
        valid_envs = 0

        for X_e, y_e in envs:
            if len(y_e) == 0: continue
            valid_envs += 1
            loss_e, grad_e = self._huber_loss_and_grad(w, X_e, y_e)
            total_loss += loss_e
            # IRM-v1 Penalty: Squared norm of the gradient per environment
            penalty += np.sum(grad_e**2)

        if valid_envs == 0:
            return 0.0

        # Structural L2 Regularization
        l2_reg = self.alpha * np.sum(w**2)

        return (total_loss / valid_envs) + (self.irm_lambda * penalty) + l2_reg

    def fit(self, X, y, env_indices):
        # Prepare environment data
        envs = []
        for idx in env_indices:
            if len(idx) > 0:
                envs.append((X[idx], y[idx]))

        if not envs:
            self.coef_ = np.zeros(X.shape[1])
            return self

        # Initial guess (zeros)
        initial_w = np.zeros(X.shape[1])

        # Optimize
        res = minimize(self._objective, initial_w, args=(envs,), method='L-BFGS-B')
        self.coef_ = res.x
        return self

    def predict(self, X):
        if self.coef_ is None:
            return np.zeros(X.shape[0])
        return X @ self.coef_

def train_dual_chaser_audit(X, y, env_indices, irm_lambda=15.0, alpha=0.1, random_state=42):
    """
    X: Predictor features (Already scaled!)
    y: target
    env_indices: GMM-based structural regimes
    """
    # 1. THE STABLE CHASER (IRM-Huber)
    stable_chaser = IRMv1HuberRegressor(
        irm_lambda=irm_lambda,
        alpha=alpha
    ).fit(X, y, env_indices)

    # 2. THE AGGRESSIVE CHASER (Standard Ridge)
    aggressive_chaser = Ridge(alpha=1.0, random_state=random_state).fit(X, y)

    return stable_chaser, aggressive_chaser

def generate_sizer_features_v2(stable_chaser, agg_chaser, X):
    # X should be scaled numpy array or DataFrame
    if isinstance(X, pd.DataFrame):
        X_val = X.values
        index = X.index
    else:
        X_val = X
        index = None

    p_stable = stable_chaser.predict(X_val)
    p_agg = agg_chaser.predict(X_val)

    spread = np.abs(p_stable - p_agg)
    ratio = np.where(np.abs(p_agg) > 1e-9, p_stable / p_agg, 0.0)

    df_out = pd.DataFrame({'chaser_spread': spread, 'chaser_ratio': ratio})
    if index is not None:
        df_out.index = index

    return df_out
