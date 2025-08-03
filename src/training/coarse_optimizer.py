# src/training/coarse_optimizer.py

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import shap
from optuna.pruners import SuccessiveHalvingPruner
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.analyst.feature_engineering import (
    FeatureEngineeringEngine,
)  # Corrected import to FeatureEngineeringEngine
from src.analyst.ml_target_generator import MLTargetGenerator
from src.config import CONFIG
from src.database.sqlite_manager import SQLiteManager
from src.utils.logger import system_logger


class CoarseOptimizer:
    """
    Performs a coarse-grained optimization to prune features and narrow down
    hyperparameter ranges before the main, intensive training stage.
    """

    def __init__(
        self,
        db_manager: SQLiteManager,
        symbol: str,
        timeframe: str,
        optimal_target_params: dict,
        klines_data: pd.DataFrame,
        agg_trades_data: pd.DataFrame,
        futures_data: pd.DataFrame,
    ):
        """
        Initializes the CoarseOptimizer.

        Args:
            db_manager (SQLiteManager): The database manager (retained for consistency, but data passed directly).
            symbol (str): The trading symbol.
            timeframe (str): The data timeframe.
            optimal_target_params (dict): The optimal TP, SL, and holding period from Stage 1.
            klines_data (pd.DataFrame): Historical klines data.
            agg_trades_data (pd.DataFrame): Historical aggregated trades data.
            futures_data (pd.DataFrame): Historical futures data.
        """
        self.db_manager = db_manager
        self.symbol = symbol
        self.timeframe = timeframe
        self.optimal_target_params = optimal_target_params
        self.logger = system_logger.getChild("CoarseOptimizer")

        # Store the passed dataframes directly
        self.klines_data = klines_data
        self.agg_trades_data = agg_trades_data
        self.futures_data = futures_data

        self.data_with_targets = None
        self._prepare_data()

    def _prepare_data(self):
        """Loads data, adds features, and generates targets using the optimal parameters."""
        self.logger.info("Preparing data for coarse optimization...")

        if self.klines_data.empty:
            raise ValueError("No klines data provided for coarse optimization.")

        # Initialize FeatureEngineeringEngine
        feature_engineering = FeatureEngineeringEngine(CONFIG)  # Pass CONFIG

        # Ensure ATR is calculated on klines before passing to feature_engineering
        klines_data_copy = self.klines_data.copy()
        klines_data_copy.ta.atr(
            length=CONFIG["best_params"]["atr_period"],
            append=True,
            col_names=("ATR"),
        )

        # Generate all available features
        # Assuming SR levels are handled by FeatureEngineeringEngine or passed separately if needed
        # For coarse optimizer, we will pass an empty list for sr_levels if not explicitly provided
        sr_levels = []  # Coarse optimizer might not need precise SR levels for feature pruning

        self.data_with_features = feature_engineering.generate_all_features(
            klines_data_copy,
            self.agg_trades_data.copy(),
            self.futures_data.copy(),
            sr_levels,
        )

        if self.data_with_features.empty:
            raise ValueError(
                "Feature engineering resulted in empty DataFrame for coarse optimization.",
            )

        # Generate targets with the optimal parameters from Stage 1
        self.logger.info(
            f"Generating targets with pre-optimized params: {self.optimal_target_params}",
        )
        target_generator = MLTargetGenerator(config=CONFIG)  # Pass CONFIG
        self.data_with_targets = target_generator.generate_targets(
            features_df=self.data_with_features,
            leverage=CONFIG.get("tactician", {}).get(
                "initial_leverage",
                50,
            ),  # Use tactical leverage
        )
        self.data_with_targets.dropna(inplace=True)
        self.logger.info(
            f"Data prepared with {len(self.data_with_targets.columns)} initial features.",
        )

    def prune_features(self, top_n_percent: float = 0.5) -> list:
        """
        Prunes the feature set by training a LightGBM model and selecting
        features with the highest importance (using SHAP values and Mutual Information).

        Args:
            top_n_percent (float): The percentage of top features to keep (e.g., 0.5 for 50%).

        Returns:
            list: A list of the names of the most important features.
        """
        import time

        pruning_start_time = time.time()

        self.logger.info("🔍 FEATURE PRUNING PROCESS STARTED")
        self.logger.info(f"   Target: Keep top {top_n_percent * 100}% of features")
        self.logger.info(f"   Start time: {time.strftime('%H:%M:%S')}")

        # Features for pruning should be all non-target/non-label columns
        self.logger.info("📋 Step 1: Identifying features for pruning...")
        features = [
            col
            for col in self.data_with_targets.columns
            if col
            not in ["target", "reward", "risk", "target_sr", "Market_Regime_Label"]
        ]
        X = self.data_with_targets[features]
        y = self.data_with_targets["target"]

        self.logger.info(f"   Total features identified: {len(features)}")
        self.logger.info(f"   Data shape: X={X.shape}, y={y.shape}")
        self.logger.info(f"   Target distribution: {y.value_counts().to_dict()}")

        if X.empty:
            self.logger.error("❌ Feature set is empty for pruning.")
            return features

        # Calculate Mutual Information scores
        self.logger.info("🔍 Step 2: Calculating Mutual Information scores...")
        mi_start_time = time.time()

        from sklearn.feature_selection import mutual_info_classif

        # Handle categorical features for MI calculation
        X_mi = X.copy()
        categorical_features = X_mi.select_dtypes(
            include=["object", "category"],
        ).columns
        if len(categorical_features) > 0:
            # Convert categorical to numeric for MI calculation
            for col in categorical_features:
                X_mi[col] = pd.Categorical(X_mi[col]).codes

        # Calculate MI scores
        mi_scores = mutual_info_classif(X_mi, y, random_state=42)
        mi_feature_scores = dict(zip(features, mi_scores, strict=False))

        mi_duration = time.time() - mi_start_time
        self.logger.info(f"   ✅ MI calculation completed in {mi_duration:.2f} seconds")

        # Train a default LightGBM model
        self.logger.info("🤖 Step 3: Training LightGBM model for SHAP importance...")
        model_start_time = time.time()

        # Check number of unique classes in target
        n_classes = len(y.unique())
        self.logger.info(
            f"   Target has {n_classes} unique classes: {sorted(y.unique())}",
        )

        # Configure LightGBM based on number of classes
        if n_classes == 2:
            self.logger.info("   Configuring LightGBM for binary classification")
            lgb_model = lgb.LGBMClassifier(
                random_state=42,
                objective="binary",
                metric="binary_logloss",
            )
        else:
            self.logger.info(
                f"   Configuring LightGBM for multi-class classification ({n_classes} classes)",
            )
            lgb_model = lgb.LGBMClassifier(
                random_state=42,
                objective="multiclass",
                metric="multi_logloss",
                num_class=n_classes,
            )

        self.logger.info("   Training LightGBM classifier...")
        lgb_model.fit(X, y)

        model_duration = time.time() - model_start_time
        self.logger.info(
            f"   ✅ Model training completed in {model_duration:.2f} seconds",
        )

        # Use SHAP for robust feature importance calculation
        self.logger.info("🔍 Step 4: Calculating SHAP values for feature importance...")
        shap_start_time = time.time()

        self.logger.info("   Creating SHAP TreeExplainer...")
        explainer = shap.TreeExplainer(lgb_model)

        self.logger.info("   Computing SHAP values (this may take a while)...")
        # Use a sample for SHAP computation to improve performance
        sample_size = min(10000, len(X))  # Use up to 10k samples
        if len(X) > sample_size:
            sample_indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X.iloc[sample_indices]
            self.logger.info(
                f"   Using {sample_size} samples for SHAP computation (out of {len(X)} total)",
            )
        else:
            X_sample = X
            self.logger.info(f"   Using all {len(X)} samples for SHAP computation")

        shap_values = explainer.shap_values(X_sample)

        shap_duration = time.time() - shap_start_time
        self.logger.info(
            f"   ✅ SHAP computation completed in {shap_duration:.2f} seconds",
        )

        # Get global feature importance
        self.logger.info("📊 Step 4: Processing SHAP values...")
        processing_start_time = time.time()

        # For multi-class classification, shap_values is a list of arrays, one for each class.
        # We take the mean absolute SHAP value across all classes.
        if isinstance(shap_values, list):
            self.logger.info(
                f"   Multi-class classification detected: {len(shap_values)} classes",
            )
            global_shap_values = np.mean(
                [np.abs(s).mean(axis=0) for s in shap_values],
                axis=0,
            )
        else:
            self.logger.info("   Binary classification detected")
            global_shap_values = np.abs(shap_values).mean(axis=0)

        # Ensure we have scalar values for each feature
        if hasattr(global_shap_values, "__len__"):
            # Convert to list of scalar values
            importance_values = (
                global_shap_values.tolist()
                if hasattr(global_shap_values, "tolist")
                else list(global_shap_values)
            )
            # Ensure all values are scalars, not lists
            importance_values = [
                float(val)
                if not isinstance(val, (list, np.ndarray))
                else float(np.mean(val))
                for val in importance_values
            ]
            self.logger.info(
                f"   Processed {len(importance_values)} feature importance values",
            )
        else:
            # Single scalar value
            importance_values = [float(global_shap_values)] * len(X.columns)
            self.logger.info(
                f"   Using single importance value for all {len(X.columns)} features",
            )

        processing_duration = time.time() - processing_start_time
        self.logger.info(
            f"   ✅ SHAP processing completed in {processing_duration:.2f} seconds",
        )

        # Create feature importance DataFrame
        self.logger.info("📈 Step 5: Creating combined feature importance ranking...")
        ranking_start_time = time.time()

        # Combine SHAP and MI scores
        combined_scores = {}
        for feature in features:
            shap_score = mi_feature_scores.get(feature, 0.0)
            mi_score = mi_feature_scores.get(feature, 0.0)

            # Normalize scores to 0-1 range
            shap_normalized = (
                shap_score / max(importance_values) if max(importance_values) > 0 else 0
            )
            mi_normalized = mi_score / max(mi_scores) if max(mi_scores) > 0 else 0

            # Combined score (weighted average: 70% SHAP, 30% MI)
            combined_score = (0.7 * shap_normalized) + (0.3 * mi_normalized)
            combined_scores[feature] = combined_score

        feature_importance = pd.DataFrame(
            [
                (
                    feature,
                    combined_scores[feature],
                    mi_feature_scores[feature],
                    importance_values[i],
                )
                for i, feature in enumerate(X.columns)
            ],
            columns=["feature", "combined_score", "mi_score", "shap_score"],
        )
        feature_importance.sort_values(
            by="combined_score",
            ascending=False,
            inplace=True,
        )

        # Log top 10 most important features with both scores
        self.logger.info("   Top 10 most important features (Combined | MI | SHAP):")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            feature_name = str(row["feature"])
            combined_val = row["combined_score"]
            mi_val = row["mi_score"]
            shap_val = row["shap_score"]
            self.logger.info(
                f"     {i+1:2d}. {feature_name:<30} {combined_val:.4f} | {mi_val:.4f} | {shap_val:.4f}",
            )

        # Log bottom 10 least important features
        self.logger.info("   Bottom 10 least important features:")
        for i, (_, row) in enumerate(feature_importance.tail(10).iterrows()):
            feature_name = str(row["feature"])
            combined_val = row["combined_score"]
            self.logger.info(
                f"     {len(feature_importance)-9+i:2d}. {feature_name:<30} {combined_val:.4f}",
            )

        ranking_duration = time.time() - ranking_start_time
        self.logger.info(
            f"   ✅ Feature ranking completed in {ranking_duration:.2f} seconds",
        )

        # Prune the feature list
        self.logger.info("✂️  Step 6: Pruning features...")
        pruning_step_start_time = time.time()

        num_features_to_keep = int(len(feature_importance) * top_n_percent)
        pruned_features = feature_importance.head(num_features_to_keep)[
            "feature"
        ].tolist()

        pruning_step_duration = time.time() - pruning_step_start_time
        self.logger.info(
            f"   ✅ Feature pruning completed in {pruning_step_duration:.2f} seconds",
        )

        # Final summary
        total_duration = time.time() - pruning_start_time
        self.logger.info("🎉 FEATURE PRUNING PROCESS COMPLETED")
        self.logger.info(f"   Total duration: {total_duration:.2f} seconds")
        self.logger.info(
            f"   Features kept: {len(pruned_features)} out of {len(features)} ({len(pruned_features)/len(features)*100:.1f}%)",
        )
        self.logger.info(f"   Features removed: {len(features) - len(pruned_features)}")
        self.logger.info("   Performance breakdown:")
        self.logger.info(
            f"     - MI calculation: {mi_duration:.2f}s ({mi_duration/total_duration*100:.1f}%)",
        )
        self.logger.info(
            f"     - Model training: {model_duration:.2f}s ({model_duration/total_duration*100:.1f}%)",
        )
        self.logger.info(
            f"     - SHAP computation: {shap_duration:.2f}s ({shap_duration/total_duration*100:.1f}%)",
        )
        self.logger.info(
            f"     - Processing & ranking: {processing_duration + ranking_duration:.2f}s ({(processing_duration + ranking_duration)/total_duration*100:.1f}%)",
        )

        return pruned_features

    def find_hyperparameter_ranges(
        self,
        pruned_features: list,
        n_trials: int = 50,
    ) -> dict:
        """
        Runs a fast Optuna study with a pruner to find promising hyperparameter ranges.

        Args:
            pruned_features (list): The list of features to use for the search.
            n_trials (int): The number of trials for the Optuna study.

        Returns:
            dict: A dictionary of narrowed hyperparameter ranges for the final optimization.
        """
        self.logger.info(
            "Finding promising hyperparameter ranges with a coarse search...",
        )
        X = self.data_with_targets[pruned_features]
        y = self.data_with_targets["target"]
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        # Check number of unique classes in target
        n_classes = len(y.unique())
        self.logger.info(
            f"   Target has {n_classes} unique classes: {sorted(y.unique())}",
        )

        def objective(trial):
            # Configure LightGBM based on number of classes
            if n_classes == 2:
                param = {
                    "objective": "binary",
                    "metric": "binary_logloss",
                    "verbosity": -1,
                    "boosting_type": "gbdt",
                }
            else:
                param = {
                    "objective": "multiclass",
                    "metric": "multi_logloss",
                    "num_class": n_classes,
                    "verbosity": -1,
                    "boosting_type": "gbdt",
                }

            # Add hyperparameters
            param.update(
                {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                    "learning_rate": trial.suggest_float(
                        "learning_rate",
                        1e-3,
                        0.1,
                        log=True,
                    ),
                    "num_leaves": trial.suggest_int("num_leaves", 20, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                },
            )

            model = lgb.LGBMClassifier(**param)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )  # Use lgb.early_stopping
            preds = model.predict(X_val)
            accuracy = accuracy_score(y_val, preds)
            return accuracy

        # Use a pruner for efficiency
        study = optuna.create_study(
            direction="maximize",
            pruner=SuccessiveHalvingPruner(),
        )
        study.optimize(objective, n_trials=n_trials)

        # Analyze top trials to define ranges
        top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[
            : max(5, int(n_trials * 0.1))
        ]

        ranges = {}
        for param_name in top_trials[0].params.keys():
            values = [t.params[param_name] for t in top_trials]
            ranges[param_name] = {
                "low": min(values),
                "high": max(values),
                "type": "float" if isinstance(values[0], float) else "int",
            }
            # Add step for consistency, though not strictly used by gp_minimize
            if isinstance(values[0], float):
                ranges[param_name]["step"] = (
                    max(values) - min(values)
                ) / 10.0  # Example step
            else:  # int
                ranges[param_name]["step"] = 1

        self.logger.info(
            f"Coarse hyperparameter search complete. Found ranges: {ranges}",
        )
        return ranges

    def run(self) -> tuple[list, dict]:
        """
        Orchestrates the coarse optimization process.

        Returns:
            A tuple containing:
            - list: The list of pruned feature names.
            - dict: A dictionary of narrowed hyperparameter ranges.
        """
        self.logger.info("--- Starting Stage 2: Coarse Optimization & Pruning ---")
        pruning_config = CONFIG.get("MODEL_TRAINING", {}).get("feature_pruning", {})
        pruned_features = self.prune_features(
            top_n_percent=pruning_config.get("top_n_percent", 0.5),
        )

        hpo_config = CONFIG.get("MODEL_TRAINING", {}).get("coarse_hpo", {})
        narrowed_ranges = self.find_hyperparameter_ranges(
            pruned_features,
            n_trials=hpo_config.get("n_trials", 50),
        )
        self.logger.info("--- Stage 2 Complete ---")
        return pruned_features, narrowed_ranges
