import logging
import warnings
import numpy as np
import pandas as pd
import optuna
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import joblib # For saving/loading models
import os # For path manipulation

warnings.filterwarnings("ignore", category=UserWarning, module="arch")
optuna.logging.set_verbosity(optuna.logging.WARNING)

class BaseEnsemble:
    """
    Base class for all child ensembles to train highly optimized and robust models.
    Includes common utilities for training, prediction, and now, model persistence.
    """

    def __init__(self, config: dict, ensemble_name: str):
        self.config = config.get("analyst", {}).get(ensemble_name, {})
        self.ensemble_name = ensemble_name
        self.logger = logging.getLogger(self.__class__.__name__)
        self.models = {}
        self.meta_learner = None
        self.trained = False
        self.pca = None
        self.meta_feature_scaler = StandardScaler()
        self.best_meta_params = {}
        self.label_encoder = LabelEncoder()
        self.n_pca_components = self.config.get("n_pca_components", 15)
        self.use_smote = self.config.get("use_smote", True)
        self.tune_base_models = self.config.get("tune_base_models", True)
        self.ensemble_weights = {self.ensemble_name: 1.0} # Default initial weight

        # Unified Feature Lists
        self.sequence_features = [
            "close", "volume", "ADX", "MACD_HIST", "ATR", "volume_delta",
            "autoencoder_reconstruction_error", "oi_roc", "funding_rate"
        ]
        self.flat_features = [
            "ADX", "MACD_HIST", "ATR", "volume_delta", "rsi", "stoch_k",
            "autoencoder_reconstruction_error", "oi_roc", "funding_rate", 
            "total_liquidations", "liquidation_ratio", "bb_bandwidth", "position_in_range",
            "OBV", "CMF", "price_vs_vwap"
        ]
        self.order_flow_features = ['volume', 'volume_delta', 'cvd_slope', 'OBV', 'CMF']

    def train_ensemble(self, historical_features: pd.DataFrame, historical_targets: pd.Series):
        self.logger.info(f"Starting full training pipeline for {self.ensemble_name}...")
        if historical_features.empty:
            self.logger.warning(f"No historical features for {self.ensemble_name}. Skipping training.")
            return

        # Ensure all expected features are present, fill missing with 0.0
        for col in self.sequence_features + self.flat_features + self.order_flow_features:
            if col not in historical_features.columns:
                historical_features[col] = 0.0

        aligned_data = historical_features.join(historical_targets.rename("target")).dropna()
        if aligned_data.empty:
            self.logger.warning(f"Aligned data is empty for {self.ensemble_name} after dropping NaNs. Skipping training.")
            return

        # Encode targets
        try:
            y_encoded = self.label_encoder.fit_transform(aligned_data["target"])
        except ValueError as e:
            self.logger.error(f"Error encoding labels for {self.ensemble_name}: {e}. Skipping training.", exc_info=True)
            return

        self._train_base_models(aligned_data, y_encoded)
        
        # Prepare meta-features for meta-learner
        meta_features_train = self._get_meta_features(aligned_data, is_live=False)
        
        # Ensure meta_features_train is a DataFrame and has an index
        if not isinstance(meta_features_train, pd.DataFrame) or meta_features_train.empty:
            self.logger.warning(f"Meta-features are empty for {self.ensemble_name}. Cannot train meta-learner.")
            return

        # Align meta-features with targets
        # Re-align y_encoded to the index of meta_features_train
        y_meta_train = pd.Series(y_encoded, index=aligned_data.index).loc[meta_features_train.index].values
        X_meta_train = meta_features_train

        if X_meta_train.empty or len(np.unique(y_meta_train)) < 2:
            self.logger.warning(f"Insufficient or single-class data for meta-learner in {self.ensemble_name}. Skipping meta-learner training.")
            return
            
        self.logger.info("Applying PCA to meta-features...")
        X_meta_scaled = self.meta_feature_scaler.fit_transform(X_meta_train)
        n_components = min(self.n_pca_components, X_meta_scaled.shape[1])
        self.pca = PCA(n_components=n_components)
        X_meta_pca = self.pca.fit_transform(X_meta_scaled)
        X_meta_pca_df = pd.DataFrame(X_meta_pca, index=X_meta_train.index)

        self.logger.info("Tuning hyperparameters for meta-learner...")
        self.best_meta_params = self._tune_hyperparameters(LGBMClassifier, self._get_lgbm_search_space, X_meta_pca_df, y_meta_train)
        self._train_meta_learner(X_meta_pca_df, y_meta_train, self.best_meta_params)
        self.trained = True
        self.logger.info(f"Training pipeline for {self.ensemble_name} complete.")

    def get_prediction(self, current_features: pd.DataFrame, **kwargs) -> dict:
        if not self.trained:
            self.logger.warning(f"Ensemble {self.ensemble_name} not trained. Returning HOLD.")
            return {"prediction": "HOLD", "confidence": 0.0}
        
        # Ensure current_features has all expected columns, fill missing with 0.0
        for col in self.sequence_features + self.flat_features + self.order_flow_features:
            if col not in current_features.columns:
                current_features[col] = 0.0

        meta_features = self._get_meta_features(current_features, is_live=True, **kwargs)
        
        # Ensure meta_features contains all columns the scaler was fitted on
        # Reindex to match the scaler's feature_names_in_
        if hasattr(self.meta_feature_scaler, 'feature_names_in_'):
            # Create a DataFrame from the dictionary, then reindex
            meta_input_df = pd.DataFrame([meta_features])
            meta_input_df = meta_input_df.reindex(columns=self.meta_feature_scaler.feature_names_in_, fill_value=0)
        else:
            self.logger.error("Scaler not fitted with feature names. Cannot ensure correct feature order for prediction.")
            return {"prediction": "HOLD", "confidence": 0.0}

        meta_input_scaled = self.meta_feature_scaler.transform(meta_input_df)
        meta_input_pca = self.pca.transform(meta_input_scaled)
        return self._get_meta_prediction(meta_input_pca)

    def _train_with_smote(self, model, X, y):
        """Applies SMOTE to balance the dataset before training."""
        if self.use_smote and len(np.unique(y)) > 1:
            try:
                smote = SMOTE(random_state=42)
                X_res, y_res = smote.fit_resample(X, y)
                self.logger.info(f"Applied SMOTE: Original size {X.shape[0]}, Resampled size {X_res.shape[0]}")
                model.fit(X_res, y_res)
            except Exception as e:
                self.logger.warning(f"SMOTE failed: {e}. Training on original data.")
                model.fit(X, y)
        else:
            model.fit(X, y)
        return model

    def _tune_hyperparameters(self, model_class, search_space_func, X, y, n_trials=25):
        """Reusable Optuna hyperparameter tuning function."""
        if not self.tune_base_models:
            self.logger.info("Base model tuning is disabled. Using default parameters.")
            return {} # Return empty dict, meaning default params will be used

        def objective(trial):
            params = search_space_func(trial)
            model = model_class(**params, random_state=42, verbose=-1)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                model.fit(X_train, y_train)
                scores.append(model.score(X_val, y_val))
            return np.mean(scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)
        self.logger.info(f"Optuna best params for {model_class.__name__}: {study.best_params}")
        return study.best_params

    def _get_lgbm_search_space(self, trial):
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        }

    def _get_svm_search_space(self, trial):
        return {
            'C': trial.suggest_float('C', 1e-3, 1e2, log=True),
            'gamma': trial.suggest_float('gamma', 1e-4, 1e-1, log=True),
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly']),
            'probability': True
        }

    def _train_meta_learner(self, X, y, params):
        self.meta_learner = LGBMClassifier(**params, random_state=42, verbose=-1)
        self.meta_learner.fit(X, y)

    def _get_meta_prediction(self, meta_input_pca):
        if not self.meta_learner: return {"prediction": "HOLD", "confidence": 0.0}
        proba = self.meta_learner.predict_proba(meta_input_pca)[0]
        idx = np.argmax(proba)
        return {"prediction": self.label_encoder.inverse_transform([idx])[0], "confidence": proba[idx]}

    def save_model(self, path: str):
        """Saves the entire ensemble instance to a file."""
        try:
            # Save relevant components
            model_data = {
                'models': self.models,
                'meta_learner': self.meta_learner,
                'pca': self.pca,
                'meta_feature_scaler': self.meta_feature_scaler,
                'label_encoder': self.label_encoder,
                'trained': self.trained,
                'best_meta_params': self.best_meta_params,
                'ensemble_weights': self.ensemble_weights
            }
            joblib.dump(model_data, path)
            self.logger.info(f"Ensemble {self.ensemble_name} model saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving {self.ensemble_name} model to {path}: {e}", exc_info=True)

    def load_model(self, path: str) -> bool:
        """Loads the entire ensemble instance from a file."""
        if not os.path.exists(path):
            self.logger.warning(f"Ensemble {self.ensemble_name} model file not found at {path}. Cannot load.")
            self.trained = False
            return False
        try:
            model_data = joblib.load(path)
            self.models = model_data.get('models', {})
            self.meta_learner = model_data.get('meta_learner')
            self.pca = model_data.get('pca')
            self.meta_feature_scaler = model_data.get('meta_feature_scaler')
            self.label_encoder = model_data.get('label_encoder')
            self.trained = model_data.get('trained', False)
            self.best_meta_params = model_data.get('best_meta_params', {})
            self.ensemble_weights = model_data.get('ensemble_weights', {self.ensemble_name: 1.0})
            self.logger.info(f"Ensemble {self.ensemble_name} model loaded from {path}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading {self.ensemble_name} model from {path}: {e}", exc_info=True)
            self.trained = False
            return False

    def _train_base_models(self, aligned_data: pd.DataFrame, y_encoded: np.ndarray):
        raise NotImplementedError
    def _get_meta_features(self, df: pd.DataFrame, is_live: bool = False, **kwargs) -> pd.DataFrame or dict:
        raise NotImplementedError

