import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from src.utils.ml_common.gmm_semantic_sorting import GMMSemanticSorter

logger = logging.getLogger(__name__)


from src.utils.irm_linear_regressor import (
    IRMLinearModel,
    IRMLinearRegressor,
    IRMLinearClassifier,
    get_vol_env_indices
)




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
        sorter = GMMSemanticSorter(sort_by="first_feature")
        self.gmm, component_order = sorter.fit_and_sort(self.gmm, scaled)
        logger.info(
            "Sorted GMM components by volatility feature (order=%s)",
            component_order,
        )
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
        if isinstance(df.index, pd.MultiIndex):
            ts_level = 'timestamp' if 'timestamp' in df.index.names else df.index.names[0]
            target_ts = pd.to_datetime(df.index.get_level_values(ts_level))
            df_probs = df_probs.copy()
            if isinstance(df_probs.index, pd.MultiIndex):
                df_probs.index = pd.to_datetime(df_probs.index.get_level_values(ts_level))
            else:
                df_probs.index = pd.to_datetime(df_probs.index)
            if df_probs.index.has_duplicates:
                df_probs = df_probs.loc[~df_probs.index.duplicated(keep="last")]
            aligned = df_probs.reindex(target_ts).fillna(0.0)
            aligned.index = df.index
            return aligned

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
    
    # Robust directory creation: handle cases where a file exists at the parent path
    parent_dir = path.parent
    if parent_dir.exists() and not parent_dir.is_dir():
        logger.warning(f"File exists at directory path {parent_dir}, removing it.")
        parent_dir.unlink()
    
    parent_dir.mkdir(parents=True, exist_ok=True)

    if refit or not path.exists():
        labeller.fit_save(df, path)

    return labeller.get_regime_labels(df, path)

def get_regime_posteriors_from_path(
    df: pd.DataFrame,
    path: Path,
    n_regimes: int = 4
) -> pd.DataFrame:
    """Helper to get posteriors directly."""
    # Handle directory vs file path
    if path.is_dir():
        file_path = path / "layer2_teacher_gmm.pkl"
    else:
        file_path = path

    labeller = MarketRegimeLabeller(n_regimes=n_regimes)
    # Ensure it's fitted
    if not file_path.exists():
        # Robust directory creation: handle cases where a file exists at the parent path
        parent_dir = file_path.parent
        if parent_dir.exists() and not parent_dir.is_dir():
            logger.warning(f"File exists at directory path {parent_dir}, removing it.")
            parent_dir.unlink()
        parent_dir.mkdir(parents=True, exist_ok=True)
        labeller.fit_save(df, file_path)

    return labeller.get_regime_posteriors(df, file_path)
