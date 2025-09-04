from __future__ import annotations
import logging
import os
import os.path
from typing import Any
import numpy as np
import pandas as pd
from joblib import dump, load
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.config import CONFIG
from src.utils.logger import system_logger
from .regime_ensembles.volatile_regime_ensemble import VolatileRegimeEnsemble
from copy import copy
from typing import Dict, List, Optional, Union, Any, Tuple

class RegimePredictiveEnsembles:
    """
    Orchestrates the training and prediction workflows for all specialized ensembles.
    Now includes checkpointing for ensemble models and a sophisticated global meta-learner
    for final prediction combining outputs from all regime-specific ensembles and market context.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config.get('analyst', {})
        self.logger = system_logger.getChild('PredictiveEnsembles.Orchestrator')
        self.regime_ensembles = {'VOLATILE_REGIME': VolatileRegimeEnsemble(config, 'VolatileRegimeEnsemble')}
        self.model_storage_dir = os.path.join(CONFIG['CHECKPOINT_DIR'], 'analyst_models', 'ensembles')
        os.makedirs(self.model_storage_dir, exist_ok=True)
        self.global_meta_learner: LGBMClassifier | None = None
        self.global_meta_scaler: StandardScaler | None = None
        self.global_meta_label_encoder: LabelEncoder | None = None
        self.global_meta_learner_path = os.path.join(self.model_storage_dir, 'global_meta_learner.joblib')
        self.global_meta_scaler_path = os.path.join(self.model_storage_dir, 'global_meta_scaler.joblib')
        self.global_meta_label_encoder_path = os.path.join(self.model_storage_dir, 'global_meta_label_encoder.joblib')
        self._load_global_meta_learner()
        self.global_meta_config = self.config.get('global_meta_learner', {'n_estimators': 100, 'learning_rate': 0.1, 'num_leaves': 31, 'verbose': -1})
        self.overall_confidence_threshold = self.config.get('overall_confidence_threshold', 0.55)



    def _train_global_meta_learner(self, meta_learner_raw_data: list[dict[str, Any]]) -> None:
        """
        Trains the global meta-learner using outputs from individual ensembles
        and high-level market context.
        """
        self.logger.info('Training global meta-learner...')
        meta_df = pd.DataFrame(meta_learner_raw_data)
        meta_df.set_index('timestamp', inplace=True)
        meta_df.sort_index(inplace=True)
        meta_df = pd.get_dummies(meta_df, columns=['regime'], prefix='regime')
        all_regimes = list(self.regime_ensembles.keys())
        for r in all_regimes:
            meta_df[f'{r}_prediction'] = meta_df.apply(lambda row: row['prediction'] if row[f'regime_{r}'] == 1 else 'HOLD', axis=1)
            meta_df[f'{r}_confidence'] = meta_df.apply(lambda row: row['confidence'] if row[f'regime_{r}'] == 1 else 0.0, axis=1)
        meta_df.drop(columns=['prediction', 'confidence'], inplace=True)
        prediction_cols = [f'{r}_prediction' for r in all_regimes]
        for col in prediction_cols:
            meta_df = pd.get_dummies(meta_df, columns=[col], prefix=col)
        meta_features = [col for col in meta_df.columns if col.startswith(('regime_', 'BULL_TREND_confidence', 'BEAR_TREND_confidence', 'SIDEWAYS_RANGE_confidence', 'VOLATILE_REGIME_confidence')) or '_prediction_' in col]
        X_meta = meta_df[meta_features].copy()
        y_meta = meta_df['true_target'].copy()
        for col in meta_features:
            if col not in X_meta.columns:
                X_meta[col] = 0
        self.global_meta_label_encoder = LabelEncoder()
        y_encoded = self.global_meta_label_encoder.fit_transform(y_meta)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        best_model = None
        best_score = -np.inf
        best_scaler = None
        best_pca = None
        for train_index, val_index in skf.split(X_meta, y_encoded):
            X_train_raw, X_val_raw = (X_meta.iloc[train_index], X_meta.iloc[val_index])
            y_train, y_val = (y_encoded[train_index], y_encoded[val_index])
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_val = scaler.transform(X_val_raw)
            if self.global_meta_config.get('use_pca', False):
                n_components = min(self.global_meta_config.get('pca_components', 16), X_train.shape[1])
                pca = PCA(n_components=n_components)
                X_train = pca.fit_transform(X_train)
                X_val = pca.transform(X_val)
            else:
                pca = None
            model = LGBMClassifier(**self.global_meta_config, random_state=42)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[LGBMClassifier.early_stopping(10, verbose=False)])
            score = model.score(X_val, y_val)
            if score > best_score:
                best_score = score
                best_model = model
                best_scaler = scaler
                best_pca = pca
        self.global_meta_learner = best_model
        self.global_meta_scaler = best_scaler
        self.global_meta_pca = best_pca
        self.logger.info('Global meta-learner trained successfully.')
        self._save_global_meta_learner()

    def _predict_with_global_meta_learner(self, primary_regime: str, ensemble_predictions: dict[str, str], ensemble_confidences: dict[str, float], current_features: pd.DataFrame) -> tuple[str, float]:
        """
        Uses the trained global meta-learner to make the final prediction.
        """
        if not self.global_meta_learner or not self.global_meta_scaler or (not self.global_meta_label_encoder):
            self.logger.warning('Global meta-learner not trained/loaded. Defaulting to HOLD.')
            return ('HOLD', 0.0)
        meta_input_data = {'regime': primary_regime}
        all_regimes = list(self.regime_ensembles.keys())
        for r in all_regimes:
            pred = ensemble_predictions.get(r, 'HOLD')
            conf = ensemble_confidences.get(r, 0.0)
            meta_input_data[f'{r}_prediction'] = pred
            meta_input_data[f'{r}_confidence'] = conf
        meta_input_df = pd.DataFrame([meta_input_data])
        meta_input_df = pd.get_dummies(meta_input_df, columns=['regime'], prefix='regime')
        prediction_cols_for_dummies = [f'{r}_prediction' for r in all_regimes]
        for col in prediction_cols_for_dummies:
            meta_input_df = pd.get_dummies(meta_input_df, columns=[col], prefix=col)
        trained_features = self.global_meta_scaler.feature_names_in_ if hasattr(self.global_meta_scaler, 'feature_names_in_') else []
        missing_cols = list(set(trained_features) - set(meta_input_df.columns))
        if missing_cols:
            self.logger.warning(f'Missing meta features at inference: {missing_cols}')
        X_meta_live = meta_input_df.reindex(columns=trained_features)
        X_meta_live = X_meta_live.fillna(0)
        X_meta_live_scaled = self.global_meta_scaler.transform(X_meta_live)
        if hasattr(self, 'global_meta_pca') and self.global_meta_pca is not None:
            X_meta_live_scaled = self.global_meta_pca.transform(X_meta_live_scaled)
        proba = self.global_meta_learner.predict_proba(X_meta_live_scaled)[0]
        predicted_label_idx = np.argmax(proba)
        final_prediction = self.global_meta_label_encoder.inverse_transform([predicted_label_idx])[0]
        final_confidence = proba[predicted_label_idx]
        if final_confidence < self.overall_confidence_threshold:
            final_prediction = 'HOLD'
            self.logger.info(f'Global meta-learner confidence ({final_confidence:.2f}) below threshold ({self.overall_confidence_threshold}). Final decision: HOLD.')
        return (final_prediction, final_confidence)

    def _save_global_meta_learner(self) -> None:
        """Saves the global meta-learner and its scaler/encoder."""
        try:
            dump(self.global_meta_learner, self.global_meta_learner_path)
            dump(self.global_meta_scaler, self.global_meta_scaler_path)
            dump(self.global_meta_label_encoder, self.global_meta_label_encoder_path)
            self.logger.info('Global meta-learner, scaler, and label encoder saved successfully.')
        except Exception as e:
            self.logger.error(f'Error saving global meta-learner components: {e}')

    def _load_global_meta_learner(self) -> None:
        """Loads the global meta-learner and its scaler/encoder."""
        if os.path.exists(self.global_meta_learner_path) and os.path.exists(self.global_meta_scaler_path) and os.path.exists(self.global_meta_label_encoder_path):
            try:
                self.global_meta_learner = load(self.global_meta_learner_path)
                self.global_meta_scaler = load(self.global_meta_scaler_path)
                self.global_meta_label_encoder = load(self.global_meta_label_encoder_path)
                self.logger.info('Global meta-learner, scaler, and label encoder loaded.')
            except Exception as e:
                self.logger.error(f'Error loading global meta-learner components: {e}')
                self.global_meta_learner = None
                self.global_meta_scaler = None
                self.global_meta_label_encoder = None
        else:
            self.logger.info('Global meta-learner components not found. Will train on first run.')


    def _map_cluster_to_regime(self, cluster_id: int, timeframe: str='1m') -> str:
        """
        Maps HMM composite cluster IDs to regime ensemble names.
        Uses dynamic regime mapping based on Step 1.7 results.
        """
        if hasattr(self, 'dynamic_mapper') and self.dynamic_mapper:
            return self.dynamic_mapper.map_cluster_to_regime(cluster_id, timeframe)
        fallback_mapping = {-1: 'RARE_MARKET_CONDITIONS', 0: 'STRONG_BULL_TREND', 1: 'MODERATE_BULL_TREND', 2: 'WEAK_BULL_TREND', 3: 'STRONG_BEAR_TREND', 4: 'MODERATE_BEAR_TREND', 5: 'TIGHT_SIDEWAYS_RANGE', 6: 'WIDE_SIDEWAYS_RANGE', 7: 'ASCENDING_SIDEWAYS', 8: 'DESCENDING_SIDEWAYS', 9: 'HIGH_VOLATILITY_BULL', 10: 'HIGH_VOLATILITY_BEAR', 11: 'LOW_VOLATILITY_RANGE', 12: 'EXTREME_VOLATILITY', 13: 'BULL_TO_BEAR_TRANSITION', 14: 'BEAR_TO_BULL_TRANSITION', 15: 'TREND_TO_SIDEWAYS', 16: 'SIDEWAYS_TO_TREND', 17: 'ACCUMULATION_PHASE', 18: 'DISTRIBUTION_PHASE', 19: 'BREAKOUT_PREPARATION'}
        regime = fallback_mapping.get(cluster_id, f'UNKNOWN_REGIME_{cluster_id}')
        self.logger.debug(f'Mapped cluster_id {cluster_id} to regime {regime}')
        return regime

    def get_regime_expert(self, cluster_id: int) -> Any:
        """
        Get the appropriate regime expert based on composite_cluster_id.
        Returns the ensemble instance for the given cluster.
        """
        regime_name = self._map_cluster_to_regime(cluster_id)
        if regime_name in self.regime_ensembles:
            ensemble = self.regime_ensembles[regime_name]
            if not ensemble.trained:
                final_model_file_name = os.path.join(self.model_storage_dir, f'final_{regime_name.lower()}_ensemble.joblib')
                if not ensemble.load_model(final_model_file_name):
                    self.logger.warning(f'Could not load final model for {regime_name}. Returning None.')
                    return None
            return ensemble
        self.logger.warning(f'No ensemble found for regime {regime_name}')
        return None

    def get_current_regime_info(self, current_features: pd.DataFrame) -> dict[str, Any]:
        """
        Get comprehensive current regime information including cluster ID and expert.
        HMM composite clusters are paramount - no fallbacks allowed.
        """
        if current_features.empty:
            return {'cluster_id': -1, 'regime_name': 'UNKNOWN', 'expert': None, 'confidence': 0.0}
        if 'composite_cluster_id' not in current_features.columns:
            self.logger.error('🚨 HMM composite_cluster_id column is missing from current features')
            self.logger.error('   HMM composite clusters are paramount - no fallbacks allowed')
            return {'cluster_id': -1, 'regime_name': 'UNKNOWN', 'expert': None, 'confidence': 0.0}
        cluster_id = int(current_features['composite_cluster_id'].iloc[-1])
        regime_name = self._map_cluster_to_regime(cluster_id)
        expert = self.get_regime_expert(cluster_id)
        confidence = 0.0
        if 'intensity_cluster_' + str(cluster_id) in current_features.columns:
            confidence = float(current_features[f'intensity_cluster_{cluster_id}'].iloc[-1])
        return {'cluster_id': cluster_id, 'regime_name': regime_name, 'expert': expert, 'confidence': confidence, 'timestamp': current_features.index[-1] if not current_features.empty else None}

    def save_model(self, ensemble_instance: Any, path: str) -> None:
        """Saves a trained ensemble instance to a file."""
        try:
            dump(ensemble_instance, path)
            self.logger.info(f'Successfully saved trained ensemble to {path}')
        except Exception as e:
            self.logger.error(f'Error saving ensemble model to {path}: {e}')

    def load_model(self, ensemble_instance: Any, path: str) -> bool:
        """Loads a trained ensemble instance from a file."""
        if not os.path.exists(path):
            return False
        try:
            loaded_ensemble = load(path)
            ensemble_instance.__dict__.update(loaded_ensemble.__dict__)
            ensemble_instance.trained = True
            self.logger.info(f'Successfully loaded pre-trained ensemble from {path}')
            return True
        except Exception as e:
            self.logger.error(f'Error loading ensemble model from {path}: {e}')
            return False

    def load_weights(self, weights: dict[str, Any]) -> Any:
        """Loads updated weights into the ensembles for dynamic weighting."""
        for regime, ensemble_weights in weights.items():
            if regime in self.regime_ensembles:
                self.regime_ensembles[regime].ensemble_weights = ensemble_weights

    def get_current_weights(self) -> dict[str, Any]:
        """Returns the current weights of all ensembles."""
        return {regime: ens.ensemble_weights for regime, ens in self.regime_ensembles.items()}