import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class IRMLinearRegressor(BaseEstimator, RegressorMixin):
    """
    IRM-v1 implementation for Ridge, Huber, and ElasticNet.

    Parameters:
    loss_type : str, 'ridge', 'huber', or 'elasticnet'
    alpha : float, overall regularization strength (Ridge/Lasso component)
    l1_ratio : float, mix between L1 and L2 (only for 'elasticnet')
    irm_lambda : float, weight of the Invariant Risk penalty
    huber_epsilon : float, threshold for Huber loss
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

    def _huber_loss_and_grad(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        """Piecewise Huber loss and its gradient."""
        res = y - X @ w
        abs_res = np.abs(res)
        mask = abs_res <= self.huber_epsilon

        loss = np.where(
            mask,
            0.5 * res**2,
            self.huber_epsilon * (abs_res - 0.5 * self.huber_epsilon)
        )
        grad = np.where(mask, -res, -self.huber_epsilon * np.sign(res))
        return float(np.mean(loss)), (X.T @ grad) / len(y)

    def _mse_loss_and_grad(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        """Standard MSE loss and its gradient."""
        res = y - X @ w
        loss = float(np.mean(res**2))
        grad = -2 * (X.T @ res) / len(y)
        return loss, grad

    def _objective(self, w: np.ndarray, envs: list[tuple[np.ndarray, np.ndarray]]) -> float:
        total_erm_loss = 0.0
        irm_penalty = 0.0

        for X_e, y_e in envs:
            if self.loss_type == 'huber':
                loss_e, grad_e = self._huber_loss_and_grad(w, X_e, y_e)
            else:
                loss_e, grad_e = self._mse_loss_and_grad(w, X_e, y_e)

            total_erm_loss += loss_e
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

    def fit(self, X: np.ndarray, y: np.ndarray, env_indices: list[np.ndarray]) -> "IRMLinearRegressor":
        X, y = check_X_y(X, y)
        envs = [(X[idx], y[idx]) for idx in env_indices]
        res = minimize(
            self._objective,
            np.zeros(X.shape[1]),
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


class IRMLinearClassifier:
    def __init__(
        self,
        loss_type: str = "ridge",
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        irm_lambda: float = 1.0,
        max_iter: int = 1000
    ):
        self.loss_type = loss_type
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.irm_lambda = irm_lambda
        self.max_iter = max_iter
        self._model = IRMLinearRegressor(
            loss_type=loss_type,
            alpha=alpha,
            l1_ratio=l1_ratio,
            irm_lambda=irm_lambda,
            max_iter=max_iter
        )

    def fit(self, X: np.ndarray, y: np.ndarray, env_indices: list[np.ndarray]) -> "IRMLinearClassifier":
        self._model.fit(X, y, env_indices)
        self.coef_ = self._model.coef_
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self._model.predict(X)
        pos = 1.0 / (1.0 + np.exp(-logits))
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
