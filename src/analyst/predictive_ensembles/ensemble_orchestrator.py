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

    def train_all_models(self, asset: str, prepared_data: pd.DataFrame, model_path_prefix: str | None=None) -> Any:
        """
        Orchestrates the training of all regime-specific ensembles.
        It splits the prepared data by regime and passes the relevant slice to each ensemble.
        After individual ensembles are trained, it trains a global meta-learner.

        Args:
            asset (str): The trading asset (e.g., "BTCUSDT").
            prepared_data (pd.DataFrame): The full prepared historical data with 'regime' and 'target' columns.
            model_path_prefix (str, optional): A prefix for saving models (e.g., includes fold_id).
        """
        self.logger.info(f'Orchestrator: Starting training for all ensembles for asset {asset} (prefix: {model_path_prefix})...')
        if 'composite_cluster_id' in prepared_data.columns:
            self.logger.info('🎯 Using HMM composite regime data for ensemble training (PARAMOUNT)')
            regime_column = 'composite_cluster_id'
            regime_prefix = 'hmm_composite_'
        else:
            self.logger.error('🚨 HMM composite_cluster_id column is missing from prepared data. Halting training.')
            self.logger.error('   HMM composite clusters are paramount - no fallbacks allowed')
            self.logger.error('   Please ensure step3_hmm_regime_discovery completed successfully')
            return
        if 'target' not in prepared_data.columns:
            self.logger.error("Prepared data is missing 'target' column. Halting training.")
            return
        meta_learner_data = []
        unique_regimes = prepared_data[regime_column].unique()
        self.logger.info(f'📊 Found {len(unique_regimes)} unique regimes: {unique_regimes}')
        for regime_id in unique_regimes:
            regime_key = f'{regime_prefix}{regime_id}'
            self.logger.info(f'--- Processing ensemble for {regime_key} ---')
            regime_data = prepared_data[prepared_data[regime_column] == regime_id]
            if regime_data.empty or len(regime_data['target'].unique()) < 2:
                self.logger.warning(f'Insufficient or single-class data for {regime_key}. Skipping training.')
                continue
            if regime_key not in self.regime_ensembles:
                self.logger.info(f'🆕 Creating new ensemble instance for {regime_key}')
                ensemble_instance = VolatileRegimeEnsemble(self.config, regime_key)
                self.regime_ensembles[regime_key] = ensemble_instance
            ensemble_instance = self.regime_ensembles[regime_key]
            historical_features = regime_data.drop(columns=['target', regime_column], errors='ignore')
            historical_targets = regime_data['target']
            model_file_name = f'{regime_key.lower()}_ensemble.joblib'
            if model_path_prefix:
                full_model_path = f'{model_path_prefix}{model_file_name}'
            else:
                full_model_path = os.path.join(self.model_storage_dir, f'final_{model_file_name}')
            if os.path.exists(full_model_path):
                self.logger.info(f'Attempting to load {regime_key} ensemble from {full_model_path}...')
                if ensemble_instance.load_model(full_model_path):
                    self.logger.info(f'Successfully loaded {regime_key} ensemble.')
                else:
                    self.logger.warning(f'Failed to load {regime_key} ensemble from {full_model_path}. Retraining.')
                    ensemble_instance.train_ensemble(historical_features, historical_targets)
            else:
                ensemble_instance.train_ensemble(historical_features, historical_targets)
            if ensemble_instance.trained:
                ensemble_instance.save_model(full_model_path)
                ensemble_predictions_on_full_data = ensemble_instance.get_prediction_on_historical_data(historical_features)
                for idx, row in ensemble_predictions_on_full_data.iterrows():
                    meta_learner_data.append({'timestamp': idx, 'regime': regime_key, 'prediction': row['prediction'], 'confidence': row['confidence'], 'true_target': prepared_data.loc[idx, 'target']})
            else:
                self.logger.warning(f'Ensemble {regime_key} was not trained/loaded successfully. Skipping for meta-learner.')
        if meta_learner_data:
            self._train_global_meta_learner(meta_learner_data)
        else:
            self.logger.warning('No data collected for global meta-learner training. Skipping.')

    def get_all_predictions(self, asset: str, current_features: pd.DataFrame, **kwargs) -> dict[str, Any]:
        """
        Gets a prediction by identifying the current regime and delegating to the
        appropriate trained ensemble. The final prediction is made by the global meta-learner.
        """
        regime_info = self.get_current_regime_info(current_features)
        primary_regime = regime_info['regime_name']
        current_expert = regime_info['expert']
        confidence = regime_info['confidence']
        ensemble_predictions_for_meta = {}
        ensemble_confidences_for_meta = {}
        combined_base_predictions = {}
        if current_expert is not None:
            try:
                prediction_output = current_expert.get_prediction(current_features, **kwargs)
                ensemble_predictions_for_meta[primary_regime] = prediction_output.get('prediction', 'HOLD')
                ensemble_confidences_for_meta[primary_regime] = prediction_output.get('confidence', confidence)
                if hasattr(current_expert, '_get_meta_features'):
                    base_preds_dict = current_expert._get_meta_features(current_features, is_live=True, **kwargs)
                    for model_name, pred_value in base_preds_dict.items():
                        unique_model_name = f'{primary_regime}_{model_name}'
                        combined_base_predictions[unique_model_name] = pred_value
                self.logger.info(f"Primary expert ({primary_regime}) prediction: {prediction_output.get('prediction', 'HOLD')} (confidence: {prediction_output.get('confidence', confidence):.3f})")
            except Exception as e:
                self.logger.exception(f'Error getting prediction from {primary_regime} expert: {e}')
                ensemble_predictions_for_meta[primary_regime] = 'HOLD'
                ensemble_confidences_for_meta[primary_regime] = 0.0
        for regime_key, ensemble_instance in self.regime_ensembles.items():
            if regime_key == primary_regime:
                continue
            if not ensemble_instance.trained:
                final_model_file_name = os.path.join(self.model_storage_dir, f'final_{regime_key.lower()}_ensemble.joblib')
                if not ensemble_instance.load_model(final_model_file_name):
                    self.logger.warning(f'Could not load final model for {regime_key}. Skipping its prediction.')
                    continue
            prediction_output = ensemble_instance.get_prediction(current_features, **kwargs)
            ensemble_predictions_for_meta[regime_key] = prediction_output.get('prediction', 'HOLD')
            ensemble_confidences_for_meta[regime_key] = prediction_output.get('confidence', 0.0)
            if hasattr(ensemble_instance, '_get_meta_features'):
                base_preds_dict = ensemble_instance._get_meta_features(current_features, is_live=True, **kwargs)
                for model_name, pred_value in base_preds_dict.items():
                    unique_model_name = f'{regime_key}_{model_name}'
                    combined_base_predictions[unique_model_name] = pred_value
        final_prediction, final_confidence = self._predict_with_global_meta_learner(primary_regime, ensemble_predictions_for_meta, ensemble_confidences_for_meta, current_features)
        current_ensemble_weights_snapshot = {regime: ens.ensemble_weights if hasattr(ens, 'ensemble_weights') else {} for regime, ens in self.regime_ensembles.items()}
        return {'prediction': final_prediction, 'confidence': final_confidence, 'regime': primary_regime, 'base_predictions': combined_base_predictions, 'ensemble_weights': current_ensemble_weights_snapshot}
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