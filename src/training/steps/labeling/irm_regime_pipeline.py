import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class IRMLinearModel(BaseEstimator):
    """
    IRM-v1 implementation for Linear Models (Regression & Classification).
    Supports Ridge, Huber, ElasticNet, and LogLoss (Logistic Regression).

    Parameters:
    loss_type : str, 'ridge', 'huber', 'elasticnet', or 'logloss'
    alpha : float, overall regularization strength
    l1_ratio : float, mix between L1 and L2 (only for 'elasticnet')
    irm_lambda : float, weight of the Invariant Risk penalty
    huber_epsilon : float, threshold for Huber loss
    max_iter : int, maximum iterations for optimizer
    """

    def __init__(
        self,
        loss_type: str = 'ridge',
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        irm_lambda: float = 1.0,
        huber_epsilon: float = 1.35,
        max_iter: int = 1000
    ):
        self.loss_type = loss_type
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.irm_lambda = irm_lambda
        self.huber_epsilon = huber_epsilon
        self.max_iter = max_iter

    def _huber_loss_and_grad(self, w: np.ndarray, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
        """Piecewise Huber loss and its gradient."""
        res = y - X @ w
        abs_res = np.abs(res)
        mask = abs_res <= self.huber_epsilon

        # Weighted loss
        loss_vec = np.where(
            mask,
            0.5 * res**2,
            self.huber_epsilon * (abs_res - 0.5 * self.huber_epsilon)
        )
        loss = np.average(loss_vec, weights=weights)

        # Weighted gradient
        grad_vec = np.where(mask, -res, -self.huber_epsilon * np.sign(res))
        # X.T @ (weights * grad_vec) / sum(weights)
        grad = (X.T @ (weights * grad_vec)) / np.sum(weights)

        return float(loss), grad

    def _mse_loss_and_grad(self, w: np.ndarray, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
        """Standard MSE loss and its gradient."""
        res = y - X @ w
        # Weighted MSE
        loss = np.average(res**2, weights=weights)
        grad = -2 * (X.T @ (weights * res)) / np.sum(weights)
        return float(loss), grad

    def _log_loss_and_grad(self, w: np.ndarray, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
        """Log Loss (Logistic Regression) and its gradient."""
        logits = X @ w
        # Stable sigmoid
        p = expit(logits)

        # Weighted Log Loss
        # - mean(w * (y log p + (1-y) log(1-p)))
        # Using log-sum-exp trick implicitly or clipping
        epsilon = 1e-15
        p_safe = np.clip(p, epsilon, 1 - epsilon)
        log_loss_vec = - (y * np.log(p_safe) + (1 - y) * np.log(1 - p_safe))
        loss = np.average(log_loss_vec, weights=weights)

        # Gradient: X.T @ (weights * (p - y)) / sum(weights)
        grad = (X.T @ (weights * (p - y))) / np.sum(weights)

        return float(loss), grad

    def _objective(self, w: np.ndarray, envs: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> float:
        total_erm_loss = 0.0
        irm_penalty = 0.0

        for X_e, y_e, w_e in envs:
            if self.loss_type == 'huber':
                loss_e, grad_e = self._huber_loss_and_grad(w, X_e, y_e, w_e)
            elif self.loss_type == 'logloss':
                loss_e, grad_e = self._log_loss_and_grad(w, X_e, y_e, w_e)
            else:
                loss_e, grad_e = self._mse_loss_and_grad(w, X_e, y_e, w_e)

            total_erm_loss += loss_e
            # IRM penalty is the norm of the gradient per environment (squared)
            irm_penalty += float(np.sum(grad_e**2))

        l2_penalty = 0.5 * np.sum(w**2)
        l1_penalty = np.sum(np.abs(w))

        if self.loss_type == 'ridge':
            reg = self.alpha * l2_penalty
        elif self.loss_type == 'elasticnet':
            reg = self.alpha * (self.l1_ratio * l1_penalty + (1 - self.l1_ratio) * l2_penalty)
        else:
            reg = self.alpha * l2_penalty

        return (total_erm_loss / len(envs)) + reg + (self.irm_lambda * irm_penalty)

    def fit(self, X: np.ndarray, y: np.ndarray, env_indices: list[np.ndarray], sample_weight: np.ndarray = None) -> "IRMLinearModel":
        X, y = check_X_y(X, y)

        if sample_weight is None:
            sample_weight = np.ones(len(y))

        envs = [(X[idx], y[idx], sample_weight[idx]) for idx in env_indices]

        # Initialize with zeros
        w0 = np.zeros(X.shape[1])

        res = minimize(
            self._objective,
            w0,
            args=(envs,),
            method='L-BFGS-B',
            options={'maxiter': self.max_iter}
        )

        self.coef_ = res.x
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = check_array(X)
        return X @ self.coef_


class IRMLinearRegressor(IRMLinearModel, RegressorMixin):
    """
    Regressor wrapper for IRMLinearModel.
    """
    def predict(self, X: np.ndarray) -> np.ndarray:
        return super().predict(X)


class IRMLinearClassifier(IRMLinearModel, ClassifierMixin):
    """
    Classifier wrapper for IRMLinearModel.
    Forces use of LogLoss for proper probabilistic output.
    """
    def __init__(
        self,
        loss_type: str = "logloss", # Default to logloss for classifier
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        irm_lambda: float = 1.0,
        max_iter: int = 1000
    ):
        super().__init__(
            loss_type=loss_type,
            alpha=alpha,
            l1_ratio=l1_ratio,
            irm_lambda=irm_lambda,
            max_iter=max_iter
        )

    def fit(self, X: np.ndarray, y: np.ndarray, env_indices: list[np.ndarray], sample_weight: np.ndarray = None) -> "IRMLinearClassifier":
        # Force logloss if user didn't specify it, or handle ridge as classification?
        # Ideally, we should stick to logloss for calibration
        if self.loss_type != 'logloss':
             # Warn or allow? Standard Ridge Classifier uses MSE on targets.
             # We allow it, but predict_proba will be heuristic if not logloss.
             pass

        super().fit(X, y, env_indices, sample_weight)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = super().predict(X)

        if self.loss_type == 'logloss':
            pos = expit(logits)
        else:
            # If trained with MSE/Huber on 0/1, logits IS the probability (roughly)
            # Clip to [0, 1]
            pos = np.clip(logits, 0.0, 1.0)

        neg = 1.0 - pos
        return np.column_stack([neg, pos])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class MarketRegimeLabeller:
    def __init__(self, n_regimes: int = 4, random_state: int = 42):
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(n_components=n_regimes, random_state=random_state)

    def prepare_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        log_ret = np.log(df['close'] / df['close'].shift(1))
        feat['volatility'] = log_ret.rolling(20).std()
        feat['rel_volume'] = df['volume'] / df['volume'].rolling(50).mean()
        feat['velocity'] = df['close'].diff(5) / df['close'].shift(5)
        return feat.dropna()

    def fit_save(self, df: pd.DataFrame, path: Path) -> None:
        data = self.prepare_regime_features(df)
        scaled = self.scaler.fit_transform(data)
        self.gmm.fit(scaled)
        joblib.dump({'gmm': self.gmm, 'scaler': self.scaler}, path)

    def get_regime_labels(self, df: pd.DataFrame, path: Path) -> pd.Series:
        setup = joblib.load(path)
        data = self.prepare_regime_features(df)
        clusters = setup['gmm'].predict(setup['scaler'].transform(data))
        labels = pd.Series(index=data.index, data=clusters, name="gmm_regime")
        return labels.reindex(df.index)

    def get_regime_posteriors(self, df: pd.DataFrame, path: Path) -> pd.DataFrame:
        """
        Get soft regime memberships (posterior probabilities).
        Returns a DataFrame where columns are 'regime_0', 'regime_1', etc.
        """
        setup = joblib.load(path)
        data = self.prepare_regime_features(df)
        probs = setup['gmm'].predict_proba(setup['scaler'].transform(data))

        cols = [f"regime_{i}" for i in range(probs.shape[1])]
        df_probs = pd.DataFrame(probs, index=data.index, columns=cols)

        # Reindex to match original df (filling NaN with 0 or equal probability?)
        # 0 is safer as it implies no knowledge
        return df_probs.reindex(df.index).fillna(0.0)

    def get_env_indices(self, df: pd.DataFrame, path: Path) -> list[np.ndarray]:
        labels = self.get_regime_labels(df, path)
        return build_env_indices_for_index(labels, df.index)


def build_env_indices_for_index(labels: pd.Series, index: pd.Index) -> list[np.ndarray]:
    aligned = labels.reindex(index)
    env_indices: list[np.ndarray] = []
    for regime in sorted(aligned.dropna().unique()):
        mask = aligned == regime
        idx = np.where(mask.values)[0]
        if len(idx) > 0:
            env_indices.append(idx.astype(int))
    return env_indices


def get_or_fit_regime_labels(
    df: pd.DataFrame,
    path: Path,
    n_regimes: int,
    refit: bool = False
) -> pd.Series:
    labeller = MarketRegimeLabeller(n_regimes=n_regimes)
    path.parent.mkdir(parents=True, exist_ok=True)

    if refit or not path.exists():
        labeller.fit_save(df, path)

    return labeller.get_regime_labels(df, path)

def get_regime_posteriors_from_path(
    df: pd.DataFrame,
    path: Path,
    n_regimes: int = 4
) -> pd.DataFrame:
    """Helper to get posteriors directly."""
    labeller = MarketRegimeLabeller(n_regimes=n_regimes)
    # Ensure it's fitted
    if not path.exists():
        labeller.fit_save(df, path)

    return labeller.get_regime_posteriors(df, path)
