# src/training/feature_selection_manager.py

import json
from datetime import datetime
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, mutual_info_classif

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger


class FeatureSelectionManager:
    """Feature Selection Manager for Step 2 - Reduces features from ~220 to 100
    with intelligent selection based on multiple criteria.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("FeatureSelectionManager")

        # Feature selection configuration
        self.target_features = config.get("feature_reduction", {}).get("step2_target_features", 100)
        self.variance_threshold = config.get("feature_reduction", {}).get("variance_threshold", 0.01)
        self.correlation_threshold = config.get("feature_reduction", {}).get("correlation_threshold", 0.95)
        self.mutual_info_threshold = config.get("feature_reduction", {}).get("mutual_info_threshold", 0.01)

        # Feature importance cache
        self.feature_importance_cache = {}
        self.selection_metadata = {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=(pd.DataFrame(), {}),
        context="feature selection step2",
    )
    def select_features_step2(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        symbol: str,
        exchange: str,
        data_dir: str,
        use_autoencoder_features: bool = True,
        use_regularization: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Multi-stage feature selection to reduce features to target count with autoencoder features and regularization.

        Args:
            features_df: Input features DataFrame
            target: Target variable series
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory for saving metadata
            use_autoencoder_features: Whether to include autoencoder features
            use_regularization: Whether to use regularization-aware selection

        Returns:
            Tuple of (selected_features_df, selection_metadata)

        """
        try:
            self.logger.info(f"🔍 Starting enhanced feature selection: {features_df.shape[1]} -> {self.target_features} features")
            
            # Stage 0: Add autoencoder features if enabled
            if use_autoencoder_features:
                features_df, stage0_metadata = self._stage0_autoencoder_features(features_df, target)
            else:
                stage0_metadata = {"autoencoder_features_added": 0}

            # Stage 1: Data quality filtering
            features_df, stage1_metadata = self._stage1_data_quality_filtering(features_df)

            # Stage 2: Variance-based filtering
            features_df, stage2_metadata = self._stage2_variance_filtering(features_df)

            # Stage 3: Correlation-based filtering
            features_df, stage3_metadata = self._stage3_correlation_filtering(features_df)

            # Stage 4: Mutual information ranking
            features_df, stage4_metadata = self._stage4_mutual_info_ranking(features_df, target)

            # Stage 5: Domain-specific selection
            features_df, stage5_metadata = self._stage5_domain_specific_selection(features_df, target)

            # Stage 6: Regularization-aware selection (if enabled)
            if use_regularization:
                features_df, stage6_metadata = self._stage6_regularization_aware_selection(features_df, target)
            else:
                stage6_metadata = {"regularization_applied": False}

            # Stage 7: Final ranking and selection
            features_df, stage7_metadata = self._stage7_final_selection(features_df, target)

            # Compile metadata
            selection_metadata = {
                "original_features": len(features_df.columns),
                "final_features": len(features_df.columns),
                "target_features": self.target_features,
                "stages": {
                    "stage0_autoencoder": stage0_metadata,
                    "stage1_data_quality": stage1_metadata,
                    "stage2_variance": stage2_metadata,
                    "stage3_correlation": stage3_metadata,
                    "stage4_mutual_info": stage4_metadata,
                    "stage5_domain_specific": stage5_metadata,
                    "stage6_regularization": stage6_metadata,
                    "stage7_final_selection": stage7_metadata,
                },
                "feature_categories": self._categorize_features(features_df.columns),
                "selection_timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "exchange": exchange,
            }

            # Save selection metadata
            self._save_selection_metadata(selection_metadata, symbol, exchange, data_dir)

            self.logger.info(f"✅ Feature selection completed: {len(features_df.columns)} features selected")
            return features_df, selection_metadata

        except Exception as e:
            self.logger.exception(f"❌ Feature selection failed: {e}")
            raise

    def _stage1_data_quality_filtering(self, features_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Stage 1: Remove features with poor data quality."""
        original_count = len(features_df.columns)

        # Remove features with too many NaN values (>10%)
        nan_ratio = features_df.isna().sum() / len(features_df)
        high_nan_features = nan_ratio[nan_ratio > 0.1].index.tolist()
        features_df = features_df.drop(columns=high_nan_features)

        # Remove features with infinite values
        inf_features = []
        for col in features_df.columns:
            if np.isinf(features_df[col]).any():
                inf_features.append(col)
        features_df = features_df.drop(columns=inf_features)

        # Fill remaining NaN values with forward fill then backward fill
        features_df = features_df.fillna(method="ffill").fillna(method="bfill").fillna(0)

        metadata = {
            "removed_high_nan": len(high_nan_features),
            "removed_infinite": len(inf_features),
            "features_after_stage": len(features_df.columns),
        }

        self.logger.info(f"Stage 1: Removed {original_count - len(features_df.columns)} low-quality features")
        return features_df, metadata

    def _stage2_variance_filtering(self, features_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Stage 2: Remove low-variance features."""
        len(features_df.columns)

        # Calculate variance for each feature
        variances = features_df.var()

        # Remove features with variance below threshold
        low_variance_features = variances[variances < self.variance_threshold].index.tolist()
        features_df = features_df.drop(columns=low_variance_features)

        metadata = {
            "removed_low_variance": len(low_variance_features),
            "variance_threshold": self.variance_threshold,
            "features_after_stage": len(features_df.columns),
        }

        self.logger.info(f"Stage 2: Removed {len(low_variance_features)} low-variance features")
        return features_df, metadata

    def _stage3_correlation_filtering(self, features_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Stage 3: Remove highly correlated features."""
        len(features_df.columns)

        # Calculate correlation matrix
        corr_matrix = features_df.corr().abs()

        # Find highly correlated feature pairs
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_pairs = []

        for col in upper_tri.columns:
            high_corr_features = upper_tri[col][upper_tri[col] > self.correlation_threshold].index.tolist()
            for feature in high_corr_features:
                high_corr_pairs.append((col, feature))

        # Remove one feature from each highly correlated pair
        features_to_remove = set()
        for feat1, feat2 in high_corr_pairs:
            # Keep the feature with higher variance
            var1 = features_df[feat1].var()
            var2 = features_df[feat2].var()
            if var1 < var2:
                features_to_remove.add(feat1)
            else:
                features_to_remove.add(feat2)

        features_df = features_df.drop(columns=list(features_to_remove))

        metadata = {
            "removed_high_correlation": len(features_to_remove),
            "correlation_threshold": self.correlation_threshold,
            "high_corr_pairs": len(high_corr_pairs),
            "features_after_stage": len(features_df.columns),
        }

        self.logger.info(f"Stage 3: Removed {len(features_to_remove)} highly correlated features")
        return features_df, metadata

    def _stage4_mutual_info_ranking(self, features_df: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Stage 4: Rank features by mutual information."""
        # Calculate mutual information scores
        mi_scores = mutual_info_classif(features_df, target, random_state=42)
        mi_ranking = pd.Series(mi_scores, index=features_df.columns).sort_values(ascending=False)

        # Store ranking for later use
        self.feature_importance_cache["mutual_info"] = mi_ranking

        metadata = {
            "top_10_mi_features": mi_ranking.head(10).index.tolist(),
            "mi_scores_range": (mi_ranking.min(), mi_ranking.max()),
            "features_after_stage": len(features_df.columns),
        }

        self.logger.info("Stage 4: Ranked features by mutual information")
        return features_df, metadata

    def _stage5_domain_specific_selection(self, features_df: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Stage 5: Domain-specific feature selection for financial data."""
        # Define feature categories and their importance weights
        # Note: Removed non-semantic categories (regime, lagged, normalized)
        feature_categories = {
            # Momentum/Trend indicators
            "momentum": [
                "momentum", "mom", "rsi", "macd", "cci", "roc", "willr", "stoch",
                "adx", "dmi", "kama", "tema", "dema", "hma", "wma", "vwma", "zlema",
                "ichimoku", "psar", "trix", "cmo", "tsi", "ppo", "pmo", "uo",
                "linreg", "lin_reg", "sma", "ema", "ma_", "moving_avg", "trend",
            ],
            # Volatility/range measures
            "volatility": [
                "volatility", "atr", "true_range", "truerange", "natr", "parkinson",
                "garman", "gk_vol", "garman_klass", "roll", "rvol", "realized_vol",
                "hv", "hist_vol", "historical_vol", "variance", "std", "bbands",
                "boll", "bollinger", "donch", "donchian", "keltner", "chop",
                "choppiness", "park_vol",
            ],
            # Liquidity/volume features
            "liquidity": [
                "liquidity", "volume", "tick_volume", "obv", "cmf", "mfi", "vwap",
                "pvi", "nvi", "efi", "delta_volume",
            ],
            # Microstructure/order book features
            "microstructure": [
                "microstructure", "order_flow", "orderflow", "ofi", "imbalance",
                "quote_imbalance", "spread", "bid_ask", "depth", "orderbook", "book",
                "microprice", "trade_count", "trade_frequency",
            ],
            # Wavelet/transform domain features
            "wavelet": ["wavelet", "dwt", "cwt", "wt_"],
            # Support/Resistance contextual features (sr_ prefix and related terms)
            "sr_distance": [
                "sr_", "sr_distance", "support", "resistance", "proximity",
                "breakout_probability", "rebounce_probability", "consolidation_probability",
                "sr_confidence", "multi_timeframe_sr_score",
            ],
            # Statistical descriptors
            "statistical": [
                "autocorr", "autocorrelation", "correl", "correlation", "entropy",
                "fractal", "hurst", "hjorth", "hj_", "kurtosis", "kurt", "skew",
                "skewness", "zscore", "z_score",
            ],
            # Candlestick pattern features
            "candlestick": [
                "cdl", "candlestick", "doji", "hammer", "engulf", "harami",
                "marubozu", "piercing", "shooting_star", "hanging_man",
                "three_black_crows", "three_white_soldiers", "morning_star", "evening_star",
                "dark_cloud",
            ],
            # Explicit interaction/composite features
            "interaction": ["_x_", "_div_", "_ratio_", "_over_", "_cross_", "interaction"],
        }

        # Calculate category importance scores
        category_scores = {}
        for category, keywords in feature_categories.items():
            category_features = [col for col in features_df.columns if any(keyword in col.lower() for keyword in keywords)]
            if category_features:
                mi_scores = self.feature_importance_cache["mutual_info"][category_features]
                category_scores[category] = mi_scores.mean()

        # Prioritize features from important categories
        prioritized_features = []
        for category, _score in sorted(category_scores.items(), key=lambda x: x[1], reverse=True):
            category_features = [col for col in features_df.columns if any(keyword in col.lower() for keyword in feature_categories[category])]
            prioritized_features.extend(category_features)

        # Ensure we don't exceed target features
        if len(prioritized_features) > self.target_features:
            prioritized_features = prioritized_features[:self.target_features]

        features_df = features_df[prioritized_features]

        metadata = {
            "category_scores": category_scores,
            "prioritized_categories": list(category_scores.keys()),
            "features_after_stage": len(features_df.columns),
        }

        self.logger.info("Stage 5: Applied domain-specific selection")
        return features_df, metadata

    def _stage6_final_selection(self, features_df: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Stage 6: Final feature selection using multiple methods (original method)."""
        if len(features_df.columns) <= self.target_features:
            # Already at or below target, return as is
            return features_df, {"final_selection": "no_change", "features_after_stage": len(features_df.columns)}

        # Use Recursive Feature Elimination with LightGBM
        estimator = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        rfe = RFE(estimator=estimator, n_features_to_select=self.target_features, step=1)

        # Fit RFE
        rfe.fit(features_df, target)

        # Get selected features
        selected_features = features_df.columns[rfe.support_].tolist()
        features_df = features_df[selected_features]

        metadata = {
            "final_selection": "rfe_lightgbm",
            "rfe_ranking": rfe.ranking_.tolist(),
            "features_after_stage": len(features_df.columns),
        }

        self.logger.info("Stage 6: Final selection using RFE-LightGBM")
        return features_df, metadata

    def _categorize_features(self, feature_names: list[str]) -> dict[str, list[str]]:
        """Categorize features by type."""
        categories = {
            "momentum": [],
            "volatility": [],
            "liquidity": [],
            "microstructure": [],
            "wavelet": [],
            "sr_distance": [],
            "statistical": [],
            "candlestick": [],
            "interaction": [],
            "transform": [],
            "other": [],
        }

        for feature in feature_names:
            feature_lower = feature.lower()
            categorized = False

            if any(keyword in feature_lower for keyword in [
                "momentum", "mom", "rsi", "macd", "cci", "roc", "willr", "stoch",
                "adx", "dmi", "kama", "tema", "dema", "hma", "wma", "vwma", "zlema",
                "ichimoku", "psar", "trix", "cmo", "tsi", "ppo", "pmo", "uo",
                "linreg", "lin_reg", "sma", "ema", "ma_", "moving_avg", "trend",
            ]):
                categories["momentum"].append(feature)
                categorized = True
            elif any(keyword in feature_lower for keyword in [
                "volatility", "atr", "true_range", "truerange", "natr", "parkinson",
                "garman", "gk_vol", "garman_klass", "roll", "rvol", "realized_vol",
                "hv", "hist_vol", "historical_vol", "variance", "std", "bbands",
                "boll", "bollinger", "donch", "donchian", "keltner", "chop",
                "choppiness", "park_vol",
            ]):
                categories["volatility"].append(feature)
                categorized = True
            elif any(keyword in feature_lower for keyword in [
                "liquidity", "volume", "tick_volume", "obv", "cmf", "mfi", "vwap",
                "pvi", "nvi", "efi", "delta_volume",
            ]):
                categories["liquidity"].append(feature)
                categorized = True
            elif any(keyword in feature_lower for keyword in [
                "microstructure", "order_flow", "orderflow", "ofi", "imbalance",
                "quote_imbalance", "spread", "bid_ask", "depth", "orderbook", "book",
                "microprice", "trade_count", "trade_frequency",
            ]):
                categories["microstructure"].append(feature)
                categorized = True
            elif any(keyword in feature_lower for keyword in ["wavelet", "dwt", "cwt", "wt_"]):
                categories["wavelet"].append(feature)
                categorized = True
            elif any(keyword in feature_lower for keyword in [
                "sr_", "sr_distance", "support", "resistance", "proximity",
                "breakout_probability", "rebounce_probability", "consolidation_probability",
                "sr_confidence", "multi_timeframe_sr_score",
            ]):
                categories["sr_distance"].append(feature)
                categorized = True
            elif any(keyword in feature_lower for keyword in ["cdl", "candlestick", "doji", "hammer", "engulf", "harami", "marubozu", "piercing", "shooting_star", "hanging_man", "three_black_crows", "three_white_soldiers", "morning_star", "evening_star", "dark_cloud"]):
                categories["candlestick"].append(feature)
                categorized = True
            elif any(keyword in feature_lower for keyword in [
                "autocorr", "autocorrelation", "correl", "correlation", "entropy",
                "fractal", "hurst", "hjorth", "hj_", "kurtosis", "kurt", "skew",
                "skewness", "zscore", "z_score",
            ]):
                categories["statistical"].append(feature)
                categorized = True
            elif any(keyword in feature_lower for keyword in ["_x_", "_div_", "_ratio_", "_over_", "_cross_", "interaction"]):
                categories["interaction"].append(feature)
                categorized = True
            elif any(keyword in feature_lower for keyword in ["fft", "fourier", "dct", "cosine", "sine", "transform_"]):
                categories["transform"].append(feature)
                categorized = True

            if not categorized:
                categories["other"].append(feature)

        return categories

    def _save_selection_metadata(self, metadata: dict[str, Any], symbol: str, exchange: str, data_dir: str) -> None:
        """Save feature selection metadata."""
        try:
            metadata_file = f"{data_dir}/{exchange}_{symbol}_feature_selection_metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            self.logger.info(f"💾 Feature selection metadata saved: {metadata_file}")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save feature selection metadata: {e}")

    def _stage0_autoencoder_features(self, features_df: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Stage 0: Add autoencoder features from the autoencoder feature generator."""
        try:
            self.logger.info("🔧 Stage 0: Adding autoencoder features...")
            
            # Import autoencoder feature generator
            from src.analyst.autoencoder_feature_generator import AutoencoderFeatureGenerator
            
            # Create autoencoder generator
            autoencoder_generator = AutoencoderFeatureGenerator()
            
            # Generate autoencoder features
            autoencoder_features = autoencoder_generator.generate_features(
                features_df=features_df,
                regime_name="default",
                labels=target.values,
                enable_analysis=True
            )
            
            # If autoencoder features were generated, add them
            if not autoencoder_features.empty and len(autoencoder_features.columns) > 0:
                # Add autoencoder features with prefix
                autoencoder_features = autoencoder_features.add_prefix("ae_")
                features_df = pd.concat([features_df, autoencoder_features], axis=1)
                
                self.logger.info(f"✅ Added {len(autoencoder_features.columns)} autoencoder features")
                stage_metadata = {
                    "autoencoder_features_added": len(autoencoder_features.columns),
                    "total_features_after_ae": len(features_df.columns)
                }
            else:
                self.logger.info("📊 No autoencoder features generated, continuing with base features")
                stage_metadata = {"autoencoder_features_added": 0}
                
        except Exception as e:
            self.logger.warning(f"⚠️ Autoencoder feature generation failed: {e}")
            self.logger.info("📊 Continuing without autoencoder features")
            stage_metadata = {"autoencoder_features_added": 0, "error": str(e)}
        
        return features_df, stage_metadata

    def _stage6_regularization_aware_selection(self, features_df: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Stage 6: Regularization-aware feature selection using pipeline regularization."""
        try:
            self.logger.info("🔧 Stage 6: Applying regularization-aware feature selection...")
            
            # Load regularization configuration from pipeline
            from src.training.regularization import RegularizationManager
            reg_manager = RegularizationManager()
            regularization_config = reg_manager.regularization_config
            
            # Apply regularization-aware feature selection
            if regularization_config:
                # Get regularization parameters
                l1_alpha = regularization_config.get('l1_alpha', 0.01)
                l2_alpha = regularization_config.get('l2_alpha', 0.001)
                
                # Calculate feature stability scores
                stability_scores = self._calculate_feature_stability(features_df, target)
                
                # Apply regularization penalty to feature importance
                if "mutual_info" in self.feature_importance_cache:
                    mi_scores = self.feature_importance_cache["mutual_info"]
                    
                    # Apply regularization penalty
                    regularization_penalty = 1.0 / (1.0 + l1_alpha + l2_alpha)
                    adjusted_scores = mi_scores * regularization_penalty
                    
                    # Select top features based on adjusted scores
                    top_features = adjusted_scores.nlargest(self.target_features).index.tolist()
                    features_df = features_df[top_features]
                    
                    stage_metadata = {
                        "regularization_applied": True,
                        "l1_alpha": l1_alpha,
                        "l2_alpha": l2_alpha,
                        "regularization_penalty": regularization_penalty,
                        "features_after_stage": len(features_df.columns)
                    }
                else:
                    stage_metadata = {"regularization_applied": False, "reason": "No mutual info scores available"}
            else:
                stage_metadata = {"regularization_applied": False, "reason": "No regularization config available"}
                
        except Exception as e:
            self.logger.warning(f"⚠️ Regularization-aware selection failed: {e}")
            stage_metadata = {"regularization_applied": False, "error": str(e)}
        
        return features_df, stage_metadata

    def _stage7_final_selection(self, features_df: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Stage 7: Final ranking and selection (renamed from stage6)."""
        # Use existing RFE-LightGBM selection logic
        return self._stage6_final_selection(features_df, target)

    def _calculate_feature_stability(self, features_df: pd.DataFrame, target: pd.Series) -> dict[str, float]:
        """Calculate feature stability scores using cross-validation."""
        stability_scores = {}
        
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.linear_model import LogisticRegression
            
            for feature in features_df.columns:
                try:
                    # Use single feature for prediction
                    X_single = features_df[[feature]]
                    
                    # Calculate cross-validation score
                    cv_scores = cross_val_score(
                        LogisticRegression(random_state=42),
                        X_single,
                        target,
                        cv=3,
                        scoring='accuracy'
                    )
                    
                    # Stability score is the mean CV score
                    stability_scores[feature] = np.mean(cv_scores)
                except Exception:
                    stability_scores[feature] = 0.0
                    
        except Exception as e:
            self.logger.warning(f"⚠️ Feature stability calculation failed: {e}")
        
        return stability_scores
