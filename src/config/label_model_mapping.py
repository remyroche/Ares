from __future__ import annotations
from typing import Any
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
DEFAULT_PROBA_THRESHOLD = 0.5
LABEL_GROUPS: dict[str, dict[str, Any]] = {'STRONG_TREND_CONTINUATION': {'low': ('xgboost', {'max_depth': 5, 'eta': 0.1, 'subsample': 0.8}), 'high': ('logistic_regression', {'C': 1.0, 'penalty': 'l2'})}, 'MOMENTUM_ACCELERATION': {'low': ('xgboost', {'max_depth': 5, 'eta': 0.1, 'subsample': 0.8}), 'high': ('logistic_regression', {'C': 1.0, 'penalty': 'l2'})}, 'EUPHORIC_BUYING': {'low': ('xgboost', {'max_depth': 5, 'eta': 0.1, 'subsample': 0.8}), 'high': ('logistic_regression', {'C': 1.0, 'penalty': 'l2'})}, 'RANGE_MEAN_REVERSION': {'low': ('catboost', {'depth': 8, 'learning_rate': 0.05}), 'high': ('lightgbm', {'num_leaves': 64, 'feature_fraction': 0.8})}, 'FAILED_RETEST': {'low': ('catboost', {'depth': 8, 'learning_rate': 0.05}), 'high': ('lightgbm', {'num_leaves': 64, 'feature_fraction': 0.8})}, 'PRICE_REJECTING_VAH': {'low': ('catboost', {'depth': 8, 'learning_rate': 0.05}), 'high': ('lightgbm', {'num_leaves': 64, 'feature_fraction': 0.8})}, 'PRICE_REJECTING_VAL': {'low': ('catboost', {'depth': 8, 'learning_rate': 0.05}), 'high': ('lightgbm', {'num_leaves': 64, 'feature_fraction': 0.8})}, 'VOLATILITY_COMPRESSION': {'low': ('sgd_hinge', {'alpha': 0.0001}), 'high': ('xgboost', {'max_depth': 4, 'colsample_bytree': 0.7})}, 'BREAKOUT_SUCCESS': {'low': ('sgd_hinge', {'alpha': 0.0001}), 'high': ('xgboost', {'max_depth': 4, 'colsample_bytree': 0.7})}, 'IGNITION_BAR': {'low': ('sgd_hinge', {'alpha': 0.0001}), 'high': ('xgboost', {'max_depth': 4, 'colsample_bytree': 0.7})}, 'SPRING_ACTION': {'low': ('sgd_hinge', {'alpha': 0.0001}), 'high': ('xgboost', {'max_depth': 4, 'colsample_bytree': 0.7})}, 'PASSIVE_ABSORPTION': {'low': ('random_forest', {'n_estimators': 300, 'max_depth': 12}), 'high': ('catboost', {'depth': 7, 'learning_rate': 0.1})}, 'STOP_HUNT': {'low': ('random_forest', {'n_estimators': 300, 'max_depth': 12}), 'high': ('catboost', {'depth': 7, 'learning_rate': 0.1})}, 'BID_ASK_COMPRESSION': {'low': ('random_forest', {'n_estimators': 300, 'max_depth': 12}), 'high': ('catboost', {'depth': 7, 'learning_rate': 0.1})}, 'CHOP_WARNING': {'low': ('lightgbm', {'num_leaves': 48}), 'high': ('random_forest', {'n_estimators': 200, 'max_depth': 10})}, 'DULL_MARKET': {'low': ('lightgbm', {'num_leaves': 48}), 'high': ('random_forest', {'n_estimators': 200, 'max_depth': 10})}, 'HIGH_CONVICTION_SETUP': {'low': ('lightgbm', {'num_leaves': 48}), 'high': ('random_forest', {'n_estimators': 200, 'max_depth': 10})}, 'BULL_TRAP': {'low': ('catboost', {'depth': 8, 'learning_rate': 0.05}), 'high': ('xgboost', {'max_depth': 5, 'subsample': 0.8})}, 'BEAR_TRAP': {'low': ('catboost', {'depth': 8, 'learning_rate': 0.05}), 'high': ('xgboost', {'max_depth': 5, 'subsample': 0.8})}, 'FAKE_BREAKOUT': {'low': ('catboost', {'depth': 8, 'learning_rate': 0.05}), 'high': ('xgboost', {'max_depth': 5, 'subsample': 0.8})}, 'LIQUIDITY_DRAIN': {'low': ('catboost', {'depth': 7, 'learning_rate': 0.05}), 'high': ('lightgbm', {'num_leaves': 64})}, 'BID_WALL_REMOVAL': {'low': ('catboost', {'depth': 7, 'learning_rate': 0.05}), 'high': ('lightgbm', {'num_leaves': 64})}, 'OFFER_STACKING': {'low': ('catboost', {'depth': 7, 'learning_rate': 0.05}), 'high': ('lightgbm', {'num_leaves': 64})}, 'NEWS_SPIKE': {'low': ('random_forest', {'n_estimators': 250}), 'high': ('xgboost', {'max_depth': 4, 'subsample': 0.7})}, 'EARNINGS_SURPRISE_REACTION': {'low': ('random_forest', {'n_estimators': 250}), 'high': ('xgboost', {'max_depth': 4, 'subsample': 0.7})}, 'MACRO_DATA_RELEASE': {'low': ('random_forest', {'n_estimators': 250}), 'high': ('xgboost', {'max_depth': 4, 'subsample': 0.7})}, 'SP500_LEAD_LAG': {'low': ('lightgbm', {'num_leaves': 48}), 'high': ('sgd_elastic_net', {'alpha': 0.1, 'l1_ratio': 0.5})}, 'YIELD_CURVE_INVERSION_ALERT': {'low': ('lightgbm', {'num_leaves': 48}), 'high': ('sgd_elastic_net', {'alpha': 0.1, 'l1_ratio': 0.5})}, 'POC_SHIFT': {'low': ('random_forest', {'n_estimators': 300}), 'high': ('catboost', {'depth': 8, 'learning_rate': 0.05})}, 'HIGH_VOLUME_NODE_REJECTION': {'low': ('random_forest', {'n_estimators': 300}), 'high': ('catboost', {'depth': 8, 'learning_rate': 0.05})}, 'VOLATILITY_REGIME_CHANGE': {'low': ('lightgbm', {'num_leaves': 64}), 'high': ('hmm_gaussian', {'n_states': 4})}, 'TREND_TO_RANGE_TRANSITION': {'low': ('lightgbm', {'num_leaves': 64}), 'high': ('hmm_gaussian', {'n_states': 4})}, 'SR_TOUCH': {'low': ('catboost', {'depth': 8, 'learning_rate': 0.05, 'l2_leaf_reg': 3}), 'high': ('lightgbm', {'num_leaves': 64, 'feature_fraction': 0.8, 'learning_rate': 0.05})}, 'SR_BOUNCE': {'low': ('catboost', {'depth': 8, 'learning_rate': 0.05, 'l2_leaf_reg': 3}), 'high': ('lightgbm', {'num_leaves': 64, 'feature_fraction': 0.8, 'learning_rate': 0.05})}, 'SR_BREAK': {'low': ('catboost', {'depth': 8, 'learning_rate': 0.05, 'l2_leaf_reg': 3}), 'high': ('lightgbm', {'num_leaves': 64, 'feature_fraction': 0.8, 'learning_rate': 0.05})}, 'SR_FAKE_BREAK': {'low': ('catboost', {'depth': 8, 'learning_rate': 0.05, 'l2_leaf_reg': 3}), 'high': ('lightgbm', {'num_leaves': 64, 'feature_fraction': 0.8, 'learning_rate': 0.05})}}
LOW_TF = {'1m', '5m'}
HIGH_TF = {'15m', '30m'}

def _tf_band(timeframe: str) -> str:
    tf = timeframe.strip().lower()
    if tf in ('1m', '5m'):
        return 'low'
    if tf in ('15m', '30m'):
        return 'high'
    return 'high'

def get_model_choice_for_label(label: str, timeframe: str) -> tuple[str, dict[str, Any]]:
    """Return (model_key, params) for the given base label and timeframe.

    If label not in mapping, default to a conservative LightGBM.
    """
    base = label.strip().upper()
    band = _tf_band(timeframe)
    cfg = LABEL_GROUPS.get(base)
    if not cfg:
        return ('lightgbm', {'num_leaves': 48})
    key, params = cfg.get(band, cfg.get('high'))
    return (key, dict(params or {}))

def build_model(model_key: str, params: dict[str, Any]) -> Any:
    """Instantiate a model from a key and params. Returns a fitted-ready estimator.

    Supported keys: 'xgboost', 'lightgbm', 'catboost', 'random_forest',
    'sgd_hinge', 'sgd_elastic_net', 'logistic_regression', 'hmm_gaussian'.
    For hmm_gaussian, we return a lightweight wrapper with fit/predict_proba
    interface if possible, else fall back to LightGBM.
    """
    key = model_key.lower()
    try:
        mapping: dict[str, Any] = {'xgboost': xgb.XGBClassifier(n_estimators=int(params.get('n_estimators', 400)), max_depth=int(params.get('max_depth', 5)), learning_rate=float(params.get('eta', params.get('learning_rate', 0.1))), subsample=float(params.get('subsample', 0.8)), colsample_bytree=float(params.get('colsample_bytree', 0.8)), random_state=42, n_jobs=-1, tree_method=str(params.get('tree_method', 'hist')), verbosity=0), 'lightgbm': lgb.LGBMClassifier(n_estimators=int(params.get('n_estimators', 400)), learning_rate=float(params.get('learning_rate', 0.05)), max_depth=int(params.get('max_depth', -1)), num_leaves=int(params.get('num_leaves', 64)), feature_fraction=float(params.get('feature_fraction', 0.8)), subsample=float(params.get('subsample', 0.8)), colsample_bytree=float(params.get('colsample_bytree', 0.8)), random_state=42, n_jobs=-1, verbose=-1), 'catboost': CatBoostClassifier(iterations=int(params.get('iterations', 500)), learning_rate=float(params.get('learning_rate', params.get('lr', 0.05))), depth=int(params.get('depth', 8)), l2_leaf_reg=float(params.get('l2_leaf_reg', 3)), random_seed=42, verbose=False), 'random_forest': RandomForestClassifier(n_estimators=int(params.get('n_estimators', 300)), max_depth=int(params.get('max_depth', 12)), random_state=42, n_jobs=-1), 'sgd_hinge': SGDClassifier(loss='hinge', alpha=float(params.get('alpha', 0.0001)), max_iter=int(params.get('max_iter', 1000)), random_state=42), 'sgd_elastic_net': SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=float(params.get('alpha', 0.0001)), l1_ratio=float(params.get('l1_ratio', 0.5)), max_iter=int(params.get('max_iter', 1000)), random_state=42), 'logistic_regression': LogisticRegression(C=float(params.get('C', 1.0)), penalty=str(params.get('penalty', 'l2')), solver='liblinear' if params.get('penalty', 'l2') == 'l2' else 'saga', max_iter=1000, random_state=42)}
        if key == 'hmm_gaussian':
            try:

                class HMMWrapper:

                    def __init__(self, n_states: int=4) -> None:
                        self.hmm = GaussianHMM(n_components=n_states, covariance_type='diag', random_state=42)
                        self.decoder = LogisticRegression(max_iter=500, random_state=42)
                        self._fitted = False

                    def fit(self, x: Any, y: Any) -> HMMWrapper:
                        if isinstance(x, pd.DataFrame | pd.Series):
                            x_arr = x.to_numpy()
                        else:
                            x_arr = np.asarray(x)
                        self.hmm.fit(x_arr)
                        states = self.hmm.predict(x_arr)
                        self.decoder.fit(states.reshape(-1, 1), y)
                        self._fitted = True
                        return self

                    def predict_proba(self, x: Any) -> np.ndarray:
                        x_arr = x.values if hasattr(x, 'values') else np.asarray(x)
                        states = self.hmm.predict(x_arr)
                        return self.decoder.predict_proba(states.reshape(-1, 1))

                    def predict(self, x: Any) -> np.ndarray:
                        proba = self.predict_proba(x)
                        return (proba[:, -1] > DEFAULT_PROBA_THRESHOLD).astype(int)
                return HMMWrapper(n_states=int(params.get('n_states', 4)))
            except Exception:
                return mapping['lightgbm']
        return mapping.get(key, RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1))
    except Exception:
        return RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)

def select_model_for_label_timeframe(label: str, timeframe: str) -> None:
    key, params = get_model_choice_for_label(label, timeframe)
    return build_model(key, params)