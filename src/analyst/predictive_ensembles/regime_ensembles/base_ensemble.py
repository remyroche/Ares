import logging
import warnings

import numpy as np
import pandas as pd
import optuna
from lightgbm import LGBMClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning, module="arch")
optuna.logging.set_verbosity(optuna.logging.WARNING)


class BaseEnsemble:
    """
    ## CHANGE: Upgraded to a full-featured, optimized training pipeline.
    ## This base class now orchestrates the entire ensemble training process, including:
    ## 1. Training of regime-specific base models (defined in child classes).
    ## 2. Generation of meta-features from base model outputs.
    ## 3. Recursive Feature Elimination (RFE) to select the most impactful meta-features.
    ## 4. Hyperparameter tuning with Optuna to find the best settings for the meta-learner.
    ## 5. Training of the final, optimized meta-learner.
    """

    def __init__(self, config: dict, ensemble_name: str):
        self.config = config.get("analyst", {}).get(ensemble_name, {})
        self.ensemble_name = ensemble_name
        self.logger = logging.getLogger(self.__class__.__name__)
        self.models = {}
        self.meta_learner = None
        self.trained = False
        self.selected_meta_features = []
        self.best_meta_params = {}
        self.label_encoder = LabelEncoder()

    def train_ensemble(self, historical_features: pd.DataFrame, historical_targets: pd.Series):
        """
        Main orchestration method for the entire training pipeline.
        """
        self.logger.info(f"Starting full training pipeline for {self.ensemble_name}...")
        if historical_features.empty or historical_targets.empty:
            self.logger.warning("No data provided for training. Aborting.")
            return

        aligned_data = historical_features.join(historical_targets.rename("target")).dropna()
        if aligned_data.empty:
            self.logger.warning("No data after alignment. Aborting.")
            return

        # Encode targets for all models
        y_encoded = self.label_encoder.fit_transform(aligned_data["target"])

        # 1. Train Base Models (implemented in child class)
        self._train_base_models(aligned_data, y_encoded)

        # 2. Generate Meta-Features (implemented in child class)
        meta_features_train = self._get_meta_features(aligned_data, is_live=False)
        
        # Align meta-features with targets
        aligned_meta = meta_features_train.join(pd.Series(y_encoded, index=aligned_data.index, name="target")).dropna()
        y_meta_train = aligned_meta["target"].values
        X_meta_train = aligned_meta.drop(columns=["target"])

        if X_meta_train.empty:
            self.logger.warning("Meta-features are empty. Aborting meta-learner training.")
            return
            
        # 3. Feature Selection
        self.logger.info("Performing feature selection for meta-learner...")
        self.selected_meta_features = self._perform_feature_selection(X_meta_train, y_meta_train)
        X_meta_selected = X_meta_train[self.selected_meta_features]

        # 4. Hyperparameter Tuning
        self.logger.info("Tuning hyperparameters for meta-learner...")
        self.best_meta_params = self._tune_meta_learner_hyperparameters(X_meta_selected, y_meta_train)

        # 5. Train Final Meta-Learner
        self._train_meta_learner(X_meta_selected, y_meta_train, self.best_meta_params)
        self.trained = True
        self.logger.info(f"Training pipeline for {self.ensemble_name} complete.")

    def get_prediction(self, current_features: pd.DataFrame, **kwargs) -> dict:
        """
        Generates a prediction using the fully trained ensemble.
        """
        if not self.trained:
            return {"prediction": "HOLD", "confidence": 0.0}

        meta_features = self._get_meta_features(current_features, is_live=True, **kwargs)
        
        # Ensure all selected features are present, fill with 0 if not
        meta_input_dict = {key: meta_features.get(key, 0.0) for key in self.selected_meta_features}
        meta_input = pd.DataFrame([meta_input_dict])
        
        return self._get_meta_prediction(meta_input)

    def _perform_feature_selection(self, X, y, n_features=15):
        """
        Selects the best features for the meta-learner using RFE.
        """
        estimator = LGBMClassifier(random_state=42, verbose=-1)
        # Ensure n_features is not greater than the number of available features
        n_features_to_select = min(n_features, X.shape[1])
        if n_features_to_select < 1:
             return X.columns.tolist() # Not enough features to select

        selector = RFE(estimator, n_features_to_select=n_features_to_select, step=1)
        selector = selector.fit(X, y)
        selected = X.columns[selector.support_].tolist()
        self.logger.info(f"RFE selected {len(selected)} features: {selected}")
        return selected

    def _tune_meta_learner_hyperparameters(self, X, y):
        """
        ## CHANGE: Added L1 and L2 regularization to the hyperparameter search.
        ## The Optuna study now tunes `reg_alpha` (L1) and `reg_lambda` (L2)
        ## to find the optimal level of regularization, making the final
        ## meta-learner more robust against overfitting.
        """
        def objective(trial):
            params = {
                'objective': 'multiclass',
                'metric': 'multi_logloss',
                'num_class': len(np.unique(y)),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),  # L1 Regularization
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True), # L2 Regularization
            }
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                model = LGBMClassifier(**params, random_state=42, verbose=-1)
                model.fit(X_train, y_train)
                scores.append(model.score(X_val, y_val))
            return np.mean(scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50, n_jobs=-1) # Use all available cores
        self.logger.info(f"Optuna best params: {study.best_params}")
        return study.best_params

    def _train_meta_learner(self, X, y, params):
        """
        Trains the final meta-learner with the selected features and best parameters.
        """
        self.meta_learner = LGBMClassifier(**params, random_state=42, verbose=-1)
        self.meta_learner.fit(X, y)
        self.logger.info(f"{self.ensemble_name} meta-learner trained successfully.")

    def _get_meta_prediction(self, meta_input_df):
        """
        Makes a final prediction using the trained meta-learner.
        """
        if not self.meta_learner:
            return {"prediction": "HOLD", "confidence": 0.0}
        
        proba = self.meta_learner.predict_proba(meta_input_df)[0]
        idx = np.argmax(proba)
        
        # Use the label encoder to transform the numeric prediction back to the original string label
        prediction_label = self.label_encoder.inverse_transform([idx])[0]
        confidence = proba[idx]
        
        return {"prediction": prediction_label, "confidence": confidence}

    # --- Abstract methods to be implemented by child classes ---
    def _train_base_models(self, aligned_data: pd.DataFrame, y_encoded: np.ndarray):
        """
        Placeholder for training regime-specific base models.
        Must be implemented by each child ensemble class.
        """
        raise NotImplementedError("Child classes must implement _train_base_models.")

    def _get_meta_features(self, df: pd.DataFrame, is_live: bool = False, **kwargs) -> pd.DataFrame or dict:
        """
        Placeholder for generating features for the meta-learner.
        Must be implemented by each child ensemble class.
        """
        raise NotImplementedError("Child classes must implement _get_meta_features.")
