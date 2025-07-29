import logging
import warnings

import numpy as np
import pandas as pd
import optuna
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning, module="arch")
optuna.logging.set_verbosity(optuna.logging.WARNING)


class BaseEnsemble:
    """
    This base class uses Principal Component Analysis (PCA) to transform the
    meta-features into a smaller set of powerful, uncorrelated components.
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
        self.n_pca_components = self.config.get("n_pca_components", 10)

    def train_ensemble(self, historical_features: pd.DataFrame, historical_targets: pd.Series):
        """
        Main orchestration method for the entire training pipeline.
        """
        self.logger.info(f"Starting full training pipeline for {self.ensemble_name}...")
        if historical_features.empty: return

        aligned_data = historical_features.join(historical_targets.rename("target")).dropna()
        if aligned_data.empty: return

        y_encoded = self.label_encoder.fit_transform(aligned_data["target"])

        # 1. Train Base Models
        self._train_base_models(aligned_data, y_encoded)

        # 2. Generate Meta-Features
        meta_features_train = self._get_meta_features(aligned_data, is_live=False)
        
        aligned_meta = meta_features_train.join(pd.Series(y_encoded, index=aligned_data.index, name="target")).dropna()
        y_meta_train = aligned_meta["target"].values
        X_meta_train = aligned_meta.drop(columns=["target"])

        if X_meta_train.empty: return
            
        # 3. Scale and Apply PCA Transformation
        self.logger.info("Applying PCA to meta-features...")
        X_meta_scaled = self.meta_feature_scaler.fit_transform(X_meta_train)
        
        # Ensure n_components is not greater than the number of features
        n_components = min(self.n_pca_components, X_meta_scaled.shape[1])
        self.pca = PCA(n_components=n_components)
        X_meta_pca = self.pca.fit_transform(X_meta_scaled)
        self.logger.info(f"PCA transformed {X_meta_scaled.shape[1]} features into {self.pca.n_components_} components.")
        
        X_meta_pca_df = pd.DataFrame(X_meta_pca, index=X_meta_train.index)

        # 4. Hyperparameter Tuning on PCA components
        self.logger.info("Tuning hyperparameters for meta-learner on PCA components...")
        self.best_meta_params = self._tune_meta_learner_hyperparameters(X_meta_pca_df, y_meta_train)

        # 5. Train Final Meta-Learner
        self._train_meta_learner(X_meta_pca_df, y_meta_train, self.best_meta_params)
        self.trained = True
        self.logger.info(f"Training pipeline for {self.ensemble_name} complete.")

    def get_prediction(self, current_features: pd.DataFrame, **kwargs) -> dict:
        """
        Generates a prediction using the fully trained ensemble with PCA.
        """
        if not self.trained:
            return {"prediction": "HOLD", "confidence": 0.0}

        meta_features = self._get_meta_features(current_features, is_live=True, **kwargs)
        
        # Create a DataFrame with the same columns as during training
        meta_input_df = pd.DataFrame([meta_features], columns=self.meta_feature_scaler.feature_names_in_)
        
        # Scale and transform using the fitted scaler and PCA
        meta_input_scaled = self.meta_feature_scaler.transform(meta_input_df)
        meta_input_pca = self.pca.transform(meta_input_scaled)
        
        return self._get_meta_prediction(meta_input_pca)

    def _tune_meta_learner_hyperparameters(self, X, y):
        # (Implementation remains the same as before)
        def objective(trial):
            params = {
                'objective': 'multiclass', 'metric': 'multi_logloss',
                'num_class': len(np.unique(y)),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
            }
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            # Optuna works with numpy arrays
            X_np, y_np = X.values, y
            for train_idx, val_idx in cv.split(X_np, y_np):
                X_train, X_val = X_np[train_idx], X_np[val_idx]
                y_train, y_val = y_np[train_idx], y_np[val_idx]
                model = LGBMClassifier(**params, random_state=42, verbose=-1)
                model.fit(X_train, y_train)
                scores.append(model.score(X_val, y_val))
            return np.mean(scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50, n_jobs=-1)
        self.logger.info(f"Optuna best params: {study.best_params}")
        return study.best_params

    def _train_meta_learner(self, X, y, params):
        self.meta_learner = LGBMClassifier(**params, random_state=42, verbose=-1)
        self.meta_learner.fit(X, y)
        self.logger.info(f"{self.ensemble_name} meta-learner trained successfully.")

    def _get_meta_prediction(self, meta_input_pca):
        if not self.meta_learner:
            return {"prediction": "HOLD", "confidence": 0.0}
        
        proba = self.meta_learner.predict_proba(meta_input_pca)[0]
        idx = np.argmax(proba)
        
        prediction_label = self.label_encoder.inverse_transform([idx])[0]
        confidence = proba[idx]
        
        return {"prediction": prediction_label, "confidence": confidence}

    # --- Abstract methods to be implemented by child classes ---
    def _train_base_models(self, aligned_data: pd.DataFrame, y_encoded: np.ndarray):
        raise NotImplementedError("Child classes must implement _train_base_models.")

    def _get_meta_features(self, df: pd.DataFrame, is_live: bool = False, **kwargs) -> pd.DataFrame or dict:
        raise NotImplementedError("Child classes must implement _get_meta_features.")
