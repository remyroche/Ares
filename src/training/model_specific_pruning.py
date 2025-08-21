# src/training/model_specific_pruning.py

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import Lasso

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger


class ModelSpecificPruning:
    """Model-specific feature pruning for different ML architectures.
    Tailored pruning strategies for each model type used in Steps 6, 6.5, 7, and 9.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config, config
        self.logger, system_logger.getChild("ModelSpecificPruning")

        # Model-specific pruning configuration
        pruning_config, config.get("feature_reduction", {}).get("pruning_strategies", {})

        self.neural_network_config, pruning_config.get("neural_networks", {
            "target_features": 80,
            "focus_on": ["non_linear", "interactions", "normalized"],
            "remove": ["highly_correlated", "low_variance"],
        })

        self.linear_model_config, pruning_config.get("linear_models", {
            "target_features": 60,
            "focus_on": ["linear", "uncorrelated", "interpretable"],
            "remove": ["interactions", "highly_correlated"],
        })

        self.ensemble_config, pruning_config.get("ensemble_models", {
            "target_features": 90,
            "focus_on": ["diverse", "different_info"],
            "remove": ["redundant", "low_importance"],
        })

        # Pruning metadata cache
        self.pruning_metadata = {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=(pd.DataFrame(), {}),
        context="neural network pruning",
    )
    def prune_for_neural_networks(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        model_type: str = "general",
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Prune features specifically for neural networks (CNN, TCN, Transformer).

        Neural networks benefit from:
        - Non-linear relationships
        - Interaction features
        - Normalized features
        - Diverse feature set

        Args:
            features_df: Input features DataFrame
            target: Target variable series
            model_type: Specific neural network type (CNN, TCN, Transformer)

        Returns:
            Tuple of (pruned_features_df, pruning_metadata)

        """
        try:
        self.logger.info(f"🧠 Pruning features for neural network: {model_type}")
            original_count, len(features_df.columns)
            target_features, self.neural_network_config["target_features"]

        # Step 1: Keep non-linear and interaction features
            non_linear_features, self._identify_non_linear_features(features_df, target)
            interaction_features, self._identify_interaction_features(features_df)
            normalized_features, self._identify_normalized_features(features_df)

        # Step 2: Remove highly correlated features (keep diverse set)
            uncorrelated_features, self._remove_highly_correlated_features(features_df, threshold=0.85)

        # Step 3: Combine and rank by importance
            preferred_features, list(set(non_linear_features + interaction_features + normalized_features))
            preferred_features = [f for f in preferred_features if f in uncorrelated_features]

        # Step 4: Add remaining features based on mutual information
            remaining_features = [f for f in uncorrelated_features if f not in preferred_features]
            mi_scores, mutual_info_classif(features_df[remaining_features], target, random_state=42)
            mi_ranking, pd.Series(mi_scores, index=remaining_features).sort_values(ascending=False)

        # Step 5: Select final features
            final_features, preferred_features + mi_ranking.head(target_features - len(preferred_features)).index.tolist()
            final_features, final_features[:target_features]  # Ensure we don't exceed target

            pruned_df, features_df[final_features]

            metadata = {
                "model_type": model_type,
                "original_features": original_count,
                "final_features": len(pruned_df.columns),
                "target_features": target_features,
                "non_linear_features": len(non_linear_features),
                "interaction_features": len(interaction_features),
                "normalized_features": len(normalized_features),
                "preferred_features": len(preferred_features),
                "pruning_strategy": "neural_network_optimized",
            }

        self.logger.info(f"✅ Neural network pruning: {original_count} -> {len(pruned_df.columns)} features")
        return pruned_df, metadata

        except Exception as e:
        self.logger.exception(f"❌ Neural network pruning failed: {e}")
            raise

    @handle_errors(
        exceptions=(Exception,),
        default_return=(pd.DataFrame(), {}),
        context="linear model pruning",
    )
    def prune_for_linear_models(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        model_type: str = "general",
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Prune features specifically for linear models (Logistic Regression, Ridge, Lasso).

        Linear models benefit from:
        - Linear relationships
        - Uncorrelated features
        - Interpretable features
        - Low multicollinearity

        Args:
            features_df: Input features DataFrame
            target: Target variable series
            model_type: Specific linear model type

        Returns:
            Tuple of (pruned_features_df, pruning_metadata)

        """
        try:
        self.logger.info(f"📊 Pruning features for linear model: {model_type}")
            original_count, len(features_df.columns)
            target_features, self.linear_model_config["target_features"]

        # Step 1: Remove interaction features (non-linear)
            linear_features, self._identify_linear_features(features_df, target)

        # Step 2: Remove highly correlated features (multicollinearity)
            uncorrelated_features, self._remove_highly_correlated_features(features_df[linear_features], threshold=0.7)

        # Step 3: Keep interpretable features
            interpretable_features, self._identify_interpretable_features(features_df[uncorrelated_features])

        # Step 4: Use Lasso for feature selection
            lasso_features, self._lasso_feature_selection(features_df[interpretable_features], target, target_features)

        # Step 5: Final selection based on mutual information
            mi_scores, mutual_info_classif(features_df[lasso_features], target, random_state=42)
            mi_ranking, pd.Series(mi_scores, index=lasso_features).sort_values(ascending=False)

            final_features, mi_ranking.head(target_features).index.tolist()
            pruned_df, features_df[final_features]

            metadata = {
                "model_type": model_type,
                "original_features": original_count,
                "final_features": len(pruned_df.columns),
                "target_features": target_features,
                "linear_features": len(linear_features),
                "interpretable_features": len(interpretable_features),
                "lasso_selected": len(lasso_features),
                "pruning_strategy": "linear_optimized",
            }

        self.logger.info(f"✅ Linear model pruning: {original_count} -> {len(pruned_df.columns)} features")
        return pruned_df, metadata

        except Exception as e:
        self.logger.exception(f"❌ Linear model pruning failed: {e}")
            raise

    @handle_errors(
        exceptions=(Exception,),
        default_return=(pd.DataFrame(), {}),
        context="ensemble model pruning",
    )
    def prune_for_ensemble_models(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        model_type: str = "general",
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Prune features specifically for ensemble models (LightGBM, XGBoost, Random Forest).

        Ensemble models benefit from:
        - Diverse feature set
        - Different information content
        - Balanced feature importance
        - Reduced redundancy

        Args:
            features_df: Input features DataFrame
            target: Target variable series
            model_type: Specific ensemble model type

        Returns:
            Tuple of (pruned_features_df, pruning_metadata)

        """
        try:
        self.logger.info(f"🌳 Pruning features for ensemble model: {model_type}")
            original_count, len(features_df.columns)
            target_features, self.ensemble_config["target_features"]

        # Step 1: Remove redundant features
            diverse_features, self._remove_redundant_features(features_df, target)

        # Step 2: Ensure feature diversity by category
            balanced_features, self._balance_feature_categories(diverse_features, target_features)

        # Step 3: Use ensemble-based feature selection
            ensemble_features, self._ensemble_feature_selection(features_df[balanced_features], target, target_features)

        # Step 4: Final optimization for ensemble diversity
            final_features, self._optimize_ensemble_diversity(features_df[ensemble_features], target, target_features)

            pruned_df, features_df[final_features]

            metadata = {
                "model_type": model_type,
                "original_features": original_count,
                "final_features": len(pruned_df.columns),
                "target_features": target_features,
                "diverse_features": len(diverse_features),
                "balanced_features": len(balanced_features),
                "ensemble_selected": len(ensemble_features),
                "pruning_strategy": "ensemble_optimized",
            }

        self.logger.info(f"✅ Ensemble model pruning: {original_count} -> {len(pruned_df.columns)} features")
        return pruned_df, metadata

        except Exception as e:
        self.logger.exception(f"❌ Ensemble model pruning failed: {e}")
            raise

    @handle_errors(
        exceptions=(Exception,),
        default_return=(pd.DataFrame(), {}),
        context="step6 hmm model pruning",
    )
    def prune_for_step6_hmm_models(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        timeframe: str,
        architecture: str,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Prune features specifically for Step 6 HMM-based models.

        Step 6 uses different architectures per timeframe:
        - 1m: CNN (neural network)
        - 5m: TCN (neural network)
        - 15m: Transformer (neural network)
        - 30m: LightGBM (ensemble)

        Args:
            features_df: Input features DataFrame
            target: Target variable series
            timeframe: Timeframe (1m, 5m, 15m, 30m)
            architecture: Model architecture (CNN, TCN, Transformer, LightGBM)

        Returns:
            Tuple of (pruned_features_df, pruning_metadata)

        """
        try:
        self.logger.info(f"🎯 Pruning features for Step 6 {timeframe} {architecture}")

        if architecture in ["CNN", "TCN", "Transformer"]:
        # Neural network pruning
        return self.prune_for_neural_networks(features_df, target, architecture)
        if architecture == "LightGBM":
        # Ensemble pruning
        return self.prune_for_ensemble_models(features_df, target, architecture)
        # Default to neural network pruning
        return self.prune_for_neural_networks(features_df, target, architecture)

        except Exception as e:
        self.logger.exception(f"❌ Step 6 pruning failed: {e}")
            raise

    @handle_errors(
        exceptions=(Exception,),
        default_return=(pd.DataFrame(), {}),
        context="step6.5 unified regime pruning",
    )
    def prune_for_step6_5_unified_regime(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Prune features specifically for Step 6.5 Unified Regime Intelligence.

        Step 6.5 uses MultiTimeframeHMMEncoder (Transformer-based) with:
        - Multi-timeframe HMM state analysis
        - Intensity-based regime transition prediction
        - Attention mechanisms

        Args:
            features_df: Input features DataFrame
            target: Target variable series

        Returns:
            Tuple of (pruned_features_df, pruning_metadata)

        """
        try:
        self.logger.info("🎯 Pruning features for Step 6.5 Unified Regime Intelligence")

        # Focus on regime-related features
            regime_features, self._identify_regime_features(features_df)
            intensity_features, self._identify_intensity_features(features_df)
            transition_features, self._identify_transition_features(features_df)

        # Combine regime-specific features
            preferred_features, list(set(regime_features + intensity_features + transition_features))

        # Use neural network pruning with regime focus
            pruned_df, metadata, self.prune_for_neural_networks(
                features_df[preferred_features], target, "MultiTimeframeHMMEncoder",
            )

            metadata.update({
                "regime_features": len(regime_features),
                "intensity_features": len(intensity_features),
                "transition_features": len(transition_features),
                "step": "6.5_unified_regime",
            })

        return pruned_df, metadata

        except Exception as e:
        self.logger.exception(f"❌ Step 6.5 pruning failed: {e}")
            raise

    @handle_errors(
        exceptions=(Exception,),
        default_return=(pd.DataFrame(), {}),
        context="step7 ensemble pruning",
    )
    def prune_for_step7_ensemble(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Prune features specifically for Step 7 Analyst Ensemble Creation.

        Step 7 creates ensembles from Step 6 models:
        - VotingClassifier
        - Multiple model types combined
        - Focus on diversity and complementarity

        Args:
            features_df: Input features DataFrame
            target: Target variable series

        Returns:
            Tuple of (pruned_features_df, pruning_metadata)

        """
        try:
        self.logger.info("🎯 Pruning features for Step 7 Analyst Ensemble")

        # Use ensemble pruning with focus on diversity
            pruned_df, metadata, self.prune_for_ensemble_models(
                features_df, target, "AnalystEnsemble",
            )

            metadata.update({
                "step": "7_analyst_ensemble",
                "ensemble_focus": "diversity_and_complementarity",
            })

        return pruned_df, metadata

        except Exception as e:
        self.logger.exception(f"❌ Step 7 pruning failed: {e}")
            raise

    @handle_errors(
        exceptions=(Exception,),
        default_return=(pd.DataFrame(), {}),
        context="step9 tactician pruning",
    )
    def prune_for_step9_tactician(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        model_type: str,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Prune features specifically for Step 9 Tactician Specialist Training.

        Step 9 uses multiple model types:
        - LightGBM
        - Calibrated Logistic Regression
        - XGBoost
        - CatBoost
        - Random Forest

        Args:
            features_df: Input features DataFrame
            target: Target variable series
            model_type: Specific model type

        Returns:
            Tuple of (pruned_features_df, pruning_metadata)

        """
        try:
        self.logger.info(f"🎯 Pruning features for Step 9 Tactician {model_type}")

        if model_type == "calibrated_logistic":
        # Linear model pruning
        return self.prune_for_linear_models(features_df, target, model_type)
        # Ensemble model pruning (LightGBM, XGBoost, CatBoost, Random Forest)
        return self.prune_for_ensemble_models(features_df, target, model_type)

        except Exception as e:
        self.logger.exception(f"❌ Step 9 pruning failed: {e}")
            raise

    # Helper methods for feature identification and selection

    def _identify_non_linear_features(self, features_df: pd.DataFrame, target: pd.Series) -> list[str]:
        """Identify features with non-linear relationships to target."""
        non_linear_features = []

        for col in features_df.columns:
        # Check for interaction features
        if "_x_" in col or "_div_" in col or "_ratio_" in col:
                non_linear_features.append(col)

        # Check for polynomial-like features
        if any(keyword in col.lower() for keyword in ["squared", "cubed", "power"]):
                non_linear_features.append(col)

        # Check for transformed features
        if any(keyword in col.lower() for keyword in ["log", "exp", "sqrt", "sin", "cos"]):
                non_linear_features.append(col)

        return non_linear_features

    def _identify_interaction_features(self, features_df: pd.DataFrame) -> list[str]:
        """Identify interaction features."""
        return [col for col in features_df.columns if "_x_" in col or "_div_" in col or "_ratio_" in col]

    def _identify_normalized_features(self, features_df: pd.DataFrame) -> list[str]:
        """Identify normalized features."""
        return [col for col in features_df.columns if any(keyword in col.lower() for keyword in ["_norm", "_z_score", "_standardized", "_scaled"])]

    def _identify_linear_features(self, features_df: pd.DataFrame, target: pd.Series) -> list[str]:
        """Identify features with linear relationships to target."""
        linear_features = []

        for col in features_df.columns:
        # Exclude interaction features
        if "_x_" in col or "_div_" in col or "_ratio_" in col:
                continue

        # Exclude transformed features
        if any(keyword in col.lower() for keyword in ["log", "exp", "sqrt", "sin", "cos", "squared", "cubed"]):
                continue

            linear_features.append(col)

        return linear_features

    def _identify_interpretable_features(self, features_df: pd.DataFrame) -> list[str]:
        """Identify interpretable features for linear models."""
        interpretable_features = []

        for col in features_df.columns:
        # Keep basic technical indicators
        if any(keyword in col.lower() for keyword in ["rsi", "macd", "sma", "ema", "atr", "adx", "cci", "mfi"]):
                interpretable_features.append(col)

        # Keep basic price/volume features
        if any(keyword in col.lower() for keyword in ["price", "volume", "returns", "volatility"]):
                interpretable_features.append(col)

        # Keep regime features
        if any(keyword in col.lower() for keyword in ["regime", "cluster", "state"]):
                interpretable_features.append(col)

        return interpretable_features

    def _identify_regime_features(self, features_df: pd.DataFrame) -> list[str]:
        """Identify regime-related features."""
        return [col for col in features_df.columns if any(keyword in col.lower() for keyword in ["regime", "cluster", "state", "composite"])]

    def _identify_intensity_features(self, features_df: pd.DataFrame) -> list[str]:
        """Identify intensity-related features."""
        return [col for col in features_df.columns if "intensity" in col.lower()]

    def _identify_transition_features(self, features_df: pd.DataFrame) -> list[str]:
        """Identify transition-related features."""
        return [col for col in features_df.columns if any(keyword in col.lower() for keyword in ["transition", "probability", "p_state"])]

    def _remove_highly_correlated_features(self, features_df: pd.DataFrame, threshold: float, 0.95) -> list[str]:
        """Remove highly correlated features."""
        corr_matrix, features_df.corr().abs()
        upper_tri, corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        features_to_keep = []
        for col in features_df.columns:
            high_corr_features, upper_tri[col][upper_tri[col] > threshold].index.tolist()
        if not high_corr_features:  # No high correlation found
                features_to_keep.append(col)

        return features_to_keep

    def _remove_redundant_features(self, features_df: pd.DataFrame, target: pd.Series) -> list[str]:
        """Remove redundant features for ensemble models."""
        # Use mutual information to identify redundant features
        mi_scores, mutual_info_classif(features_df, target, random_state=42)
        mi_ranking, pd.Series(mi_scores, index=features_df.columns).sort_values(ascending=False)

        # Keep top features and remove highly correlated ones
        top_features, mi_ranking.head(len(features_df.columns) // 2).index.tolist()
        return self._remove_highly_correlated_features(features_df[top_features], threshold=0.9)


    def _balance_feature_categories(self, features_df: pd.DataFrame, target_features: int) -> list[str]:
        """Balance features across categories for ensemble diversity."""
        categories = {
            "momentum": [],
            "volatility": [],
            "liquidity": [],
            "regime": [],
            "other": [],
        }

        for col in features_df.columns:
        if any(keyword in col.lower() for keyword in ["momentum", "rsi", "macd"]):
                categories["momentum"].append(col)
            elif any(keyword in col.lower() for keyword in ["volatility", "atr"]):
                categories["volatility"].append(col)
            elif any(keyword in col.lower() for keyword in ["liquidity", "volume"]):
                categories["liquidity"].append(col)
            elif any(keyword in col.lower() for keyword in ["regime", "cluster"]):
                categories["regime"].append(col)
            else:
                categories["other"].append(col)

        # Balance features across categories
        balanced_features = []
        features_per_category, target_features // len(categories)

        for features in categories.values():
            balanced_features.extend(features[:features_per_category])

        return balanced_features[:target_features]

    def _lasso_feature_selection(self, features_df: pd.DataFrame, target: pd.Series, target_features: int) -> list[str]:
        """Use Lasso for feature selection in linear models."""
        lasso, Lasso(alpha=0.01, random_state=42)
        lasso.fit(features_df, target)

        # Get features with non-zero coefficients
        selected_features, features_df.columns[lasso.coef_ != 0].tolist()

        # If too many features selected, use top by coefficient magnitude
        if len(selected_features) > target_features:
            coef_ranking, pd.Series(lasso.coef_, index=features_df.columns).abs().sort_values(ascending=False)
            selected_features, coef_ranking.head(target_features).index.tolist()

        return selected_features

    def _ensemble_feature_selection(self, features_df: pd.DataFrame, target: pd.Series, target_features: int) -> list[str]:
        """Use ensemble methods for feature selection."""
        # Use Random Forest for feature importance
        rf, RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(features_df, target)

        # Get feature importance ranking
        importance_ranking, pd.Series(rf.feature_importances_, index=features_df.columns).sort_values(ascending=False)

        return importance_ranking.head(target_features).index.tolist()

    def _optimize_ensemble_diversity(self, features_df: pd.DataFrame, target: pd.Series, target_features: int) -> list[str]:
        """Optimize feature diversity for ensemble models."""
        # Use multiple feature selection methods
        methods = [
            ("random_forest", RandomForestClassifier(n_estimators=100, random_state=42)),
            ("lightgbm", lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)),
            ("mutual_info", None),  # Will use mutual_info_classif
        ]

        feature_scores = {}
        for method_name, estimator in methods:
        if method_name == "mutual_info":
                scores, mutual_info_classif(features_df, target, random_state=42)
            else:
                estimator.fit(features_df, target)
                scores, estimator.feature_importances_

            feature_scores[method_name] = pd.Series(scores, index=features_df.columns)

        # Combine scores from different methods
        combined_scores, pd.DataFrame(feature_scores).mean(axis=1).sort_values(ascending=False)

        return combined_scores.head(target_features).index.tolist()
