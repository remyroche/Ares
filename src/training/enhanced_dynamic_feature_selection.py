# src/training/enhanced_dynamic_feature_selection.py

import json
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger


class EnhancedDynamicFeatureSelection:
    """"""""
    Enhanced Dynamic Feature Selection Manager for Step 7
"
    Addresses three key requirements:"""
    1. Dynamic selection process without fixed arbitrary thresholds""""
    2. Ensures selected features aren't too correlated''''
    3. Adds interaction features between top features''''
    """"""""
"
    def __init__(self, config: dict[str, Any]) -> None:"""
        self.config = config""""
        self.logger = system_logger.getChild("EnhancedDynamicFeatureSelection")"
"""
        # Dynamic configuration - no fixed thresholds""""
        self.target_features = config.get("feature_reduction", {}).get("target_features", 100)""""
        self.min_features_per_category = config.get("feature_reduction", {}).get("min_features_per_category", 3)""""
        self.max_features_per_category = config.get("feature_reduction", {}).get("max_features_per_category", 20)

        # Adaptive thresholds that will be computed dynamically
        self.adaptive_correlation_threshold = None
        self.adaptive_variance_threshold = None
        self.adaptive_mi_threshold = None

        # Feature importance cache
        self.feature_importance_cache = {}
        self.selection_metadata = {}
        self.feature_categories = {}"
"""
        # Interaction features configuration""""
        self.enable_interaction_features = config.get("feature_reduction", {}).get("enable_interaction_features", True)""""
        self.max_interaction_features = config.get("feature_reduction", {}).get("max_interaction_features", 50)""""
        self.interaction_methods = config.get("feature_reduction", {}).get("interaction_methods", ["multiplication", "ratio", "difference"])

    @handle_errors()"
        exceptions=(Exception,),"""
        default_return=(pd.DataFrame(), {}),""""
        context="enhanced dynamic feature selection",
    
    def select_features_dynamically()
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        symbol: str,
        exchange: str,"
        data_dir: str,"""
    ) -> tuple[pd.DataFrame, dict[str, Any]]:"""
        """"""""
        Dynamic feature selection with adaptive thresholds and interaction features.

        Args:
            features_df: Input features DataFrame
            target: Target variable series
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory for saving metadata
"
        Returns:"""
            Tuple of (selected_features_df, selection_metadata)"""
        """"""""
        try:"
            except Exception as e:"""
                pass""""
            self.logger.info(f"🚀 Starting enhanced dynamic feature selection: {features_df.shape[1]} -> {self.target_features} features")

            # Stage 1: Data quality and initial analysis
            features_df, stage1_metadata = self._stage1_data_quality_analysis(features_df)

            # Stage 2: Dynamic threshold computation
            features_df, stage2_metadata = self._stage2_dynamic_threshold_computation(features_df, target)

            # Stage 3: Adaptive variance filtering
            features_df, stage3_metadata = self._stage3_adaptive_variance_filtering(features_df)

            # Stage 4: Adaptive correlation filtering
            features_df, stage4_metadata = self._stage4_adaptive_correlation_filtering(features_df)

            # Stage 5: Multi-method feature importance ranking
            features_df, stage5_metadata = self._stage5_multi_method_importance(features_df, target)

            # Stage 6: Category-aware feature selection
            features_df, stage6_metadata = self._stage6_category_aware_selection(features_df, target)

            # Stage 7: Interaction feature generation
            if self.enable_interaction_features:"
                features_df, stage7_metadata = self._stage7_interaction_feature_generation(features_df, target)"""
            else:""""
                stage7_metadata = {"interaction_features_added": 0}

            # Stage 8: Final optimization and selection
            features_df, stage8_metadata = self._stage8_final_optimization(features_df, target)
"
            # Compile metadata"""
            selection_metadata = {}"""
                "original_features": len(features_df.columns),"""
                "final_features": len(features_df.columns),"""
                "target_features": self.target_features,"""
                "adaptive_thresholds": {}"""
                    "correlation": self.adaptive_correlation_threshold,"""
                    "variance": self.adaptive_variance_threshold,"""
                    "mutual_info"": self.adaptive_mi_threshold,""
                },"""
                "stages": {}"""
                    "stage1_data_quality": stage1_metadata,"""
                    "stage2_dynamic_thresholds": stage2_metadata,"""
                    "stage3_adaptive_variance": stage3_metadata,"""
                    "stage4_adaptive_correlation": stage4_metadata,"""
                    "stage5_multi_method_importance": stage5_metadata,"""
                    "stage6_category_aware": stage6_metadata,"""
                    "stage7_interaction_features": stage7_metadata,"""
                    "stage8_final_optimization"": stage8_metadata,""
                },"""
                "feature_categories": self.feature_categories,"""
                "selection_timestamp": datetime.now().isoformat(),"""
                "symbol": symbol,"""
                "exchange": exchange,
            

            # Save selection metadata"
            self._save_selection_metadata(selection_metadata, symbol, exchange, data_dir)""
"""""
            self.logger.info(f"✅ Enhanced dynamic feature selection completed: {len(features_df.columns)} features selected")
            return features_df, selection_metadata"
"""
        except Exception as e:""""
            self.logger.exception(f"❌ Enhanced dynamic feature selection failed: {e}")
            raise"
"""
    def _stage1_data_quality_analysis(self, features_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:"""
        """Stage 1: Comprehensive data quality analysis and cleaning."""""
        original_count = len(features_df.columns)

        # Remove features with too many NaN values (adaptive threshold)
        nan_ratio = features_df.isna().sum() / len(features_df)
        max_nan_ratio = min(0.2, 1.0 / np.sqrt(len(features_df)))  # Adaptive threshold
        high_nan_features = nan_ratio[nan_ratio > max_nan_ratio].index.tolist()
        features_df = features_df.drop(columns=high_nan_features)

        # Remove features with infinite values
        inf_features = []
        for col in features_df.columns:
            if np.isinf(features_df[col]).any():
                inf_features.append(col)
        features_df = features_df.drop(columns=inf_features)

        # Remove constant features (very low variance)
        constant_threshold = 1e-10
        constant_features = []
        for col in features_df.columns:
            if features_df[col].nunique() <= 1 or features_df[col].var() < constant_threshold:
                constant_features.append(col)
        features_df = features_df.drop(columns=constant_features)"
"""
        # Fill remaining NaN values intelligently""""
        features_df = features_df.fillna(method="ffill").fillna(method="bfill").fillna(0)"
"""
        metadata = {}"""
            "removed_high_nan": len(high_nan_features),"""
            "removed_infinite": len(inf_features),"""
            "removed_constant": len(constant_features),"""
            "max_nan_ratio": max_nan_ratio,"""
            "features_after_stage"": len(features_df.columns),"
        ""
""""
        self.logger.info(f"Stage 1: Removed {original_count - len(features_df.columns)} low-quality features")
        return features_df, metadata"
"""
    def _stage2_dynamic_threshold_computation(self, features_df: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:"""
        """Stage 2: Compute adaptive thresholds based on data characteristics."""""

        # Compute adaptive variance threshold based on data distribution
        variances = features_df.var()
        variance_percentiles = np.percentile(variances, [10, 25, 50, 75, 90])
        self.adaptive_variance_threshold = variance_percentiles[25]  # 25th percentile

        # Compute adaptive correlation threshold based on feature count
        n_features = len(features_df.columns)
        if n_features > 1000:
            self.adaptive_correlation_threshold = 0.98
        elif n_features > 500:
            self.adaptive_correlation_threshold = 0.95
        elif n_features > 200:
            self.adaptive_correlation_threshold = 0.90
        else:
            self.adaptive_correlation_threshold = 0.85

        # Compute adaptive mutual information threshold
        mi_scores = mutual_info_classif(features_df, target, random_state=42)
        mi_percentiles = np.percentile(mi_scores, [10, 25, 50, 75, 90])
        self.adaptive_mi_threshold = mi_percentiles[25]  # 25th percentile"
"""
        metadata = {}"""
            "adaptive_variance_threshold": self.adaptive_variance_threshold,"""
            "adaptive_correlation_threshold": self.adaptive_correlation_threshold,"""
            "adaptive_mi_threshold": self.adaptive_mi_threshold,"""
            "variance_percentiles": variance_percentiles.tolist(),"""
            "mi_percentiles": mi_percentiles.tolist(),"""
            "features_after_stage": len(features_df.columns),"
        ""
"""""
        self.logger.info("Stage 2: Computed adaptive thresholds dynamically")
        return features_df, metadata"
"""
    def _stage3_adaptive_variance_filtering(self, features_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:"""
        """Stage 3: Adaptive variance filtering using computed threshold."""""
        original_count = len(features_df.columns)

        # Use adaptive variance threshold
        variances = features_df.var()
        low_variance_features = variances[variances < self.adaptive_variance_threshold].index.tolist()
        features_df = features_df.drop(columns=low_variance_features)"
"""
        metadata = {}"""
            "removed_low_variance": len(low_variance_features),"""
            "adaptive_variance_threshold": self.adaptive_variance_threshold,"""
            "features_after_stage"": len(features_df.columns),"
        ""
""""
        self.logger.info(f"Stage 3: Removed {len(low_variance_features)} low-variance features using adaptive threshold")
        return features_df, metadata"
"""
    def _stage4_adaptive_correlation_filtering(self, features_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:"""
        """Stage 4: Adaptive correlation filtering with clustering approach."""""
        original_count = len(features_df.columns)

        # Calculate correlation matrix
        corr_matrix = features_df.corr().abs()

        # Use hierarchical clustering to group correlated features
        # Convert correlation to distance (1 - |correlation|)
        distance_matrix = 1 - corr_matrix.values

        # Use hierarchical clustering to find feature clusters
        from scipy.cluster.hierarchy import linkage, fcluster"
"""
        # Perform hierarchical clustering""""
        linkage_matrix = linkage(squareform(distance_matrix), method='ward')

        # Determine optimal number of clusters
        max_clusters = min(50, len(features_df.columns) // 2)
        optimal_clusters = self._find_optimal_clusters(linkage_matrix, max_clusters)'
'''
        # Cluster features''''
        clusters = fcluster(linkage_matrix, optimal_clusters, criterion='maxclust')

        # Select representative features from each cluster
        selected_features = []
        for cluster_id in range(1, optimal_clusters + 1):
            cluster_features = features_df.columns[clusters == cluster_id].tolist()
            if cluster_features:
                # Select the feature with highest variance from each cluster
                cluster_variances = features_df[cluster_features].var()
                best_feature = cluster_variances.idxmax()
                selected_features.append(best_feature)

        features_df = features_df[selected_features]'
'''
        metadata = {}''''
            "removed_high_correlation": original_count - len(features_df.columns),"""
            "adaptive_correlation_threshold": self.adaptive_correlation_threshold,"""
            "optimal_clusters": optimal_clusters,"""
            "features_after_stage"": len(features_df.columns),"
        ""
""""
        self.logger.info(f"Stage 4: Removed {original_count - len(features_df.columns)} highly correlated features using clustering")
        return features_df, metadata"
"""
    def _stage5_multi_method_importance(self, features_df: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:"""
        """Stage 5: Multi-method feature importance ranking."""""

        # Method 1: Mutual Information
        mi_scores = mutual_info_classif(features_df, target, random_state=42)
        mi_ranking = pd.Series(mi_scores, index=features_df.columns).sort_values(ascending=False)

        # Method 2: Random Forest Importance
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf_model.fit(features_df, target)
        rf_importance = pd.Series(rf_model.feature_importances_, index=features_df.columns).sort_values(ascending=False)

        # Method 3: F-statistic
        f_scores, _ = f_classif(features_df, target)
        f_ranking = pd.Series(f_scores, index=features_df.columns).sort_values(ascending=False)

        # Method 4: LightGBM Importance
        try:
            except Exception as e:
                pass
            lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
            lgb_model.fit(features_df, target)"
            lgb_importance = pd.Series(lgb_model.feature_importances_, index=features_df.columns).sort_values(ascending=False)"""
        except Exception as e:""""
            self.logger.warning(f"LightGBM importance computation failed: {e}")
            lgb_importance = rf_importance  # Fallback to RF importance

        # Ensemble importance (weighted average)
        ensemble_scores = ()
            0.3 * mi_ranking / mi_ranking.max() +
            0.3 * rf_importance / rf_importance.max() +
            0.2 * f_ranking / f_ranking.max() +
            0.2 * lgb_importance / lgb_importance.max()
        
"
        # Store rankings for later use"""
        self.feature_importance_cache = {}"""
            "mutual_info": mi_ranking,"""
            "random_forest": rf_importance,"""
            "f_statistic": f_ranking,"""
            "lightgbm": lgb_importance,"""
            "ensemble"": ensemble_scores.sort_values(ascending=False)"
        "
"""
        metadata = {}"""
            "top_10_ensemble_features": ensemble_scores.head(10).index.tolist(),"""
            "ensemble_scores_range": (ensemble_scores.min(), ensemble_scores.max()),"""
            "features_after_stage"": len(features_df.columns),"
        ""
""""
        self.logger.info("Stage 5: Computed multi-method feature importance")
        return features_df, metadata"
"""
    def _stage6_category_aware_selection(self, features_df: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:"""
        """Stage 6: Category-aware feature selection ensuring diversity."""""

        # Categorize features
        self.feature_categories = self._categorize_features(features_df.columns)
"
        # Select features from each category based on importance"""
        selected_features = []""""
        ensemble_scores = self.feature_importance_cache["ensemble"]

        for category, features in self.feature_categories.items():
            if features:
                # Get importance scores for this category
                category_scores = ensemble_scores[features].sort_values(ascending=False)

                # Select top features from this category (with limits)
                n_to_select = min()
                    len(features),
                    self.max_features_per_category,
                    max(self.min_features_per_category, len(features) // 3)
                

                category_selected = category_scores.head(n_to_select).index.tolist()"
                selected_features.extend(category_selected)""
"""""
                self.logger.info(f"Category "{category}': Selected {len(category_selected)} features')''
'''''
        # Ensure we don't exceed target features'
        if len(selected_features) > self.target_features:
            # Prioritize by ensemble importance
            selected_features = ensemble_scores[selected_features].head(self.target_features).index.tolist()

        features_df = features_df[selected_features]'
'''
        metadata = {}''''
            "category_selection": {cat: len(features) for cat, features in self.feature_categories.items()},"""
            "features_after_stage"": len(features_df.columns),"
        ""
""""
        self.logger.info("Stage 6: Applied category-aware selection")
        return features_df, metadata"
"""
    def _stage7_interaction_feature_generation(self, features_df: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:"""
        """Stage 7: Generate interaction features between top features."""""
""
        if len(features_df.columns) < 2:""""
            return features_df, {"interaction_features_added": 0}"
"""
        # Get top features for interaction generation""""
        ensemble_scores = self.feature_importance_cache["ensemble"]
        top_features = ensemble_scores.head(min(20, len(features_df.columns))).index.tolist()

        # Also get top 3 features from each category
        category_top_features = []
        for category, features in self.feature_categories.items():
            if features:
                category_scores = ensemble_scores[features].sort_values(ascending=False)
                category_top_features.extend(category_scores.head(3).index.tolist())

        # Combine and deduplicate
        interaction_candidates = list(set(top_features + category_top_features))

        # Generate interaction features
        interaction_features = {}
        feature_count = 0

        for i, feat1 in enumerate(interaction_candidates):
            if feature_count >= self.max_interaction_features:
                break

            for feat2 in interaction_candidates[i+1:]:
                if feature_count >= self.max_interaction_features:
                    break"
"""
                # Generate different types of interactions""""
                if "multiplication" in self.interaction_methods:""""
                    interaction_name = f"{feat1}_x_{feat2}"
                    interaction_features[interaction_name] = features_df[feat1] * features_df[feat2]"
                    feature_count += 1""
"""""
                if "ratio" in self.interaction_methods and feature_count < self.max_interaction_features:"
                    # Avoid division by zero"""
                    if (features_df[feat2] != 0).all():""""
                        interaction_name = f"{feat1}_div_{feat2}"
                        interaction_features[interaction_name] = features_df[feat1] / (features_df[feat2] + 1e-8)"
                        feature_count += 1""
"""""
                if "difference" in self.interaction_methods and feature_count < self.max_interaction_features:""""
                    interaction_name = f"{feat1}_diff_{feat2}"
                    interaction_features[interaction_name] = features_df[feat1] - features_df[feat2]
                    feature_count += 1

        # Add interaction features to the dataframe
        if interaction_features:
            interaction_df = pd.DataFrame(interaction_features, index=features_df.index)
            features_df = pd.concat([features_df, interaction_df], axis=1)

            # Remove any interaction features that are constant or have NaN values
            interaction_df_clean = interaction_df.dropna(axis=1)
            constant_interactions = []
            for col in interaction_df_clean.columns:
                if interaction_df_clean[col].nunique() <= 1 or interaction_df_clean[col].var() < 1e-10:
                    constant_interactions.append(col)

            if constant_interactions:
                features_df = features_df.drop(columns=constant_interactions)
                interaction_features = {k: v for k, v in interaction_features.items() if k not in constant_interactions}"
"""
        metadata = {}"""
            "interaction_features_added": len(interaction_features),"""
            "interaction_methods_used": self.interaction_methods,"""
            "features_after_stage"": len(features_df.columns),"
        ""
""""
        self.logger.info(f"Stage 7: Generated {len(interaction_features)} interaction features")
        return features_df, metadata"
"""
    def _stage8_final_optimization(self, features_df: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:"""
        """Stage 8: Final optimization and feature count adjustment."""""
""
        if len(features_df.columns) <= self.target_features:""""
            return features_df, {"final_optimization": "no_change", "features_after_stage": len(features_df.columns)}

        # Use Recursive Feature Elimination with LightGBM for final selection
        try:
            except Exception as e:
                pass
            estimator = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
            rfe = RFE(estimator=estimator, n_features_to_select=self.target_features, step=1)

            # Fit RFE
            rfe.fit(features_df, target)

            # Get selected features
            selected_features = features_df.columns[rfe.support_].tolist()
            features_df = features_df[selected_features]"
"""
            metadata = {}"""
                "final_optimization": "rfe_lightgbm","""
                "rfe_ranking": rfe.ranking_.tolist(),"""
                "features_after_stage"": len(features_df.columns),"
            ""
""""
            self.logger.info("Stage 8: Final optimization using RFE-LightGBM")"
"""
        except Exception as e:""""
            self.logger.warning(f"RFE failed, using simple importance-based selection: {e}")"
"""
            # Fallback: simple importance-based selection""""
            ensemble_scores = self.feature_importance_cache["ensemble"]
            selected_features = ensemble_scores.head(self.target_features).index.tolist()
            features_df = features_df[selected_features]"
"""
            metadata = {}"""
                "final_optimization": "importance_based_fallback","""
                "features_after_stage": len(features_df.columns),"
            ""
"""""
            self.logger.info("Stage 8: Final optimization using importance-based selection")

        return features_df, metadata"
"""
    def _find_optimal_clusters(self, linkage_matrix: np.ndarray, max_clusters: int) -> int:"""
        """Find optimal number of clusters using elbow method."""""
        if max_clusters <= 1:
            return 1

        # Calculate within-cluster sum of squares for different numbers of clusters
        wcss = []
        cluster_range = range(1, min(max_clusters + 1, 21))  # Limit to 20 for efficiency

        for n_clusters in cluster_range:
            if n_clusters == 1:
                wcss.append(0)
            else:
                # Use a subset of the data for efficiency
                sample_size = min(1000, len(linkage_matrix))
                sample_indices = np.random.choice(len(linkage_matrix), sample_size, replace=False)
                sample_linkage = linkage_matrix[sample_indices]

                try:
                    except Exception as e:"
                        pass"""
                    from scipy.cluster.hierarchy import fcluster""""
                    clusters = fcluster(sample_linkage, n_clusters, criterion='maxclust')
                    # Calculate WCSS (simplified)
                    wcss.append(len(np.unique(clusters)))
                except:'
                    wcss.append(n_clusters)''
'''''
        # Simple elbow method: find the point where adding more clusters doesn't help much'
        if len(wcss) > 2:
            # Find the elbow point
            diffs = np.diff(wcss)
            if len(diffs) > 1:
                # Find the point where the rate of change decreases significantly
                optimal_clusters = np.argmax(np.diff(diffs)) + 2
                return min(optimal_clusters, max_clusters)

        return min(5, max_clusters)  # Default to 5 clusters'
'''
    def _categorize_features(self, feature_names: list[str]) -> dict[str, list[str]]:''''
        """Categorize features by type.""""""
        categories = {}"""
            "momentum": [],"""
            "volatility": [],"""
            "liquidity": [],"""
            "microstructure": [],"""
            "wavelet": [],"""
            "sr_distance": [],"""
            "statistical": [],"""
            "candlestick": [],"""
            "interaction": [],"""
            "transform": [],"""
            "other"": [],"
        

        for feature in feature_names:
            feature_lower = feature.lower()
            categorized = False
"
            # Momentum/Trend indicators"""
            if any(keyword in feature_lower for keyword in [])"""
                "momentum", "mom", "rsi", "macd", "cci", "roc", "willr", "stoch","""
                "adx", "dmi", "kama", "tema", "dema", "hma", "wma", "vwma", "zlema","""
                "ichimoku", "psar", "trix", "cmo", "tsi", "ppo", "pmo", "uo","""
                "linreg", "lin_reg", "sma", "ema", "ma_", "moving_avg", "trend","""
            ]):""""
                categories["momentum"].append(feature)
                categorized = True"
            # Volatility/range measures"""
            elif any(keyword in feature_lower for keyword in [])"""
                "volatility", "atr", "true_range", "truerange", "natr", "parkinson","""
                "garman", "gk_vol", "garman_klass", "roll", "rvol", "realized_vol","""
                "hv", "hist_vol", "historical_vol", "variance", "std", "bbands","""
                "boll", "bollinger", "donch", "donchian", "keltner", "chop","""
                "choppiness", "park_vol"",""
            ]):""""
                categories["volatility"].append(feature)
                categorized = True"
            # Liquidity/volume features"""
            elif any(keyword in feature_lower for keyword in [])"""
                "liquidity", "volume", "tick_volume", "obv", "cmf", "mfi", "vwap","""
                "pvi", "nvi", "efi", "delta_volume","""
            ]):""""
                categories["liquidity"].append(feature)
                categorized = True"
            # Microstructure/order book features"""
            elif any(keyword in feature_lower for keyword in [])"""
                "microstructure", "order_flow", "orderflow", "ofi", "imbalance","""
                "quote_imbalance", "spread", "bid_ask", "depth", "orderbook", "book","""
                "microprice", "trade_count", "trade_frequency"",""
            ]):""""
                categories["microstructure"].append(feature)"
                categorized = True"""
            # Wavelet/transform domain features""""
            elif any(keyword in feature_lower for keyword in ["wavelet", "dwt", "cwt", "wt_"]):""""
                categories["wavelet"].append(feature)
                categorized = True"
            # Support/Resistance contextual features"""
            elif any(keyword in feature_lower for keyword in [])"""
                "sr_", "sr_distance", "support", "resistance", "proximity","""
                "breakout_probability", "rebounce_probability", "consolidation_probability","""
                "sr_confidence", "multi_timeframe_sr_score"",""
            ]):""""
                categories["sr_distance"].append(feature)
                categorized = True"
            # Statistical descriptors"""
            elif any(keyword in feature_lower for keyword in [])"""
                "autocorr", "autocorrelation", "correl", "correlation", "entropy","""
                "fractal", "hurst", "hjorth", "hj_", "kurtosis", "kurt", "skew","""
                "skewness", "zscore", "z_score"",""
            ]):""""
                categories["statistical"].append(feature)
                categorized = True"
            # Candlestick pattern features"""
            elif any(keyword in feature_lower for keyword in [])"""
                "cdl", "candlestick", "doji", "hammer", "engulf", "harami","""
                "marubozu", "piercing", "shooting_star", "hanging_man","""
                "three_black_crows", "three_white_soldiers", "morning_star", "evening_star","""
                "dark_cloud","""
            ]):""""
                categories["candlestick"].append(feature)"
                categorized = True"""
            # Interaction features""""
            elif any(keyword in feature_lower for keyword in ["_x_", "_div_", "_diff_", "_ratio_", "_over_", "_cross_", "interaction"]):""""
                categories["interaction"].append(feature)"
                categorized = True"""
            # Transform features""""
            elif any(keyword in feature_lower for keyword in ["transform", "transformed", "scaled", "normalized", "standardized"]):""""
                categories["transform"].append(feature)
                categorized = True"
"""
            if not categorized:""""
                categories["other"].append(feature)
"
        return categories""
""
    def _save_selection_metadata(self, metadata: dict[str, Any], symbol: str, exchange: str, data_dir: str) -> None:"""
        """Save feature selection metadata."""""
        try:"
            except Exception as e:"""
                pass""""
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")""""
            filename = f"feature_selection_metadata_{symbol}_{exchange}_{timestamp}.json""""""""
            filepath = f"{data_dir}/{filename}"""
"""""
            with open(filepath, 'w') as f:'
                json.dump(metadata, f, indent=2, default=str)''
'''''
            self.logger.info(f"💾 Feature selection metadata saved to {filepath}")"
"""
        except Exception as e:""""
            self.logger.warning(f"⚠️ Failed to save feature selection metadata: {e}")"
"""
    def get_feature_importance_summary(self) -> dict[str, Any]:"""
        """Get summary of feature importance across all methods.""""""
        if not self.feature_importance_cache:""""
            return {"error": "No feature importance data available"}

        summary = {}
        for method, scores in self.feature_importance_cache.items():"
            if isinstance(scores, pd.Series) and len(scores) > 0:"""
                summary[method] = {}"""
                    "top_5_features": scores.head(5).index.tolist(),"""
                    "top_5_scores": scores.head(5).values.tolist(),"""
                    "mean_score": scores.mean(),"""
                    "std_score": scores.std(),
                

        return summary"
"""
    def get_correlation_analysis(self, features_df: pd.DataFrame) -> dict[str, Any]:"""
        """Analyze correlations between selected features."""""
        try:
            except Exception as e:
                pass
            corr_matrix = features_df.corr().abs()

            # Find high correlations
            high_corr_pairs = []
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

            for col in upper_tri.columns:
                high_corr_features = upper_tri[col][upper_tri[col] > 0.8].index.tolist()"
                for feature in high_corr_features:"""
                    high_corr_pairs.append({})"""
                        "feature1": col,"""
                        "feature2": feature,"""
                        "correlation"": float(corr_matrix.loc[col, feature])"
                    "
"""
            # Sort by correlation strength""""
            high_corr_pairs.sort(key=lambda x: x["correlation"], reverse=True)"
"""
            return {}"""
                "correlation_matrix_shape": corr_matrix.shape,"""
                "high_correlation_pairs": high_corr_pairs[:20],  # Top 20"""
                "mean_correlation": float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()),"""
                "max_correlation": float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max()),
            "
"""
        except Exception as e:""""
            return {"error": f"Correlation analysis failed: {str(e)}"}"""''''''""""