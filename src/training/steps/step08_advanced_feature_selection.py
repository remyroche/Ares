# src/training/steps/step08_advanced_feature_selection.py

"""Step 8: Advanced Feature Selection with Two-Phase Approach.

This step performs sophisticated feature selection using:
- Phase 1: mRMR and Random Forest to select top 150 features
- Phase 2: Boruta to generate multiple feature sets (100, 80, 60)
with regime-aware selection, time-series validation, and interpretability analysis.
"""

import asyncio
import json
import os
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import lightgbm as lgb

# Import if available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available - interpretability features will be limited")

try:
    from boruta import BorutaPy
    BORUTA_AVAILABLE = True
except ImportError:
    BORUTA_AVAILABLE = False
    warnings.warn("Boruta not available - will use alternative feature selection")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    warnings.warn("LIME not available - interpretability features will be limited")

from src.core.decorators import handles_errors
from src.utils.common_operations import ensure_directory, safe_json_dump
from src.utils.logger import system_logger
from src.utils.pipeline_standards import pipeline_standards


class Step08AdvancedFeatureSelection:
    """Advanced two-phase feature selection with regime awareness and interpretability."""
    
    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize Step 8 Advanced Feature Selection."""
        self.config = config
        self.logger = system_logger.getChild("Step08AdvancedFeatureSelection")
        self.standards = pipeline_standards
        
        # Step-specific configuration
        self.step_config = config.get("step08_advanced_feature_selection", {})
        self.output_dir = ensure_directory(self.step_config.get("output_dir", "data/selected_features"))
        
        # Phase 1 configuration (mRMR/RF)
        self.phase1_target_features = self.step_config.get("phase1_target_features", 150)
        self.enable_mrmr = self.step_config.get("enable_mrmr", True)
        self.enable_rf_importance = self.step_config.get("enable_rf_importance", True)
        
        # Phase 2 configuration (Boruta with redundancy)
        self.phase2_targets = self.step_config.get("phase2_targets", [100, 80, 60])
        self.boruta_max_iter = self.step_config.get("boruta_max_iter", 100)
        self.boruta_alpha = self.step_config.get("boruta_alpha", 0.05)
        
        # Redundancy configuration
        self.enable_redundancy_analysis = self.step_config.get("enable_redundancy_analysis", True)
        self.min_redundancy_correlation = self.step_config.get("min_redundancy_correlation", 0.7)
        self.redundancy_groups_per_concept = self.step_config.get("redundancy_groups_per_concept", 2)
        self.feature_concept_patterns = self.step_config.get("feature_concept_patterns", {
            "momentum": ["rsi", "macd", "momentum", "roc"],
            "volatility": ["bb_", "atr", "volatility", "std"],
            "volume": ["volume", "vwap", "obv", "mfi"],
            "trend": ["ema", "sma", "trend", "adx"],
            "microstructure": ["spread", "imbalance", "flow", "tick"],
            "regime": ["regime", "cluster", "state"],
            "support_resistance": ["sr_", "support", "resistance", "level"]
        })
        
        # Validation configuration
        self.n_splits_ts = self.step_config.get("n_splits_ts", 5)
        self.min_regime_samples = self.step_config.get("min_regime_samples", 100)
        
        # Interpretability configuration
        self.enable_shap = self.step_config.get("enable_shap", True) and SHAP_AVAILABLE
        self.enable_lime = self.step_config.get("enable_lime", True) and LIME_AVAILABLE
        self.n_lime_samples = self.step_config.get("n_lime_samples", 10)
        
        self.logger.info("🚀 Step 8 Advanced Feature Selection initialized")
        self.logger.info(f"   Phase 1 target: {self.phase1_target_features} features")
        self.logger.info(f"   Phase 2 targets: {self.phase2_targets}")
        self.logger.info(f"   Boruta available: {BORUTA_AVAILABLE}")
        self.logger.info(f"   SHAP available: {SHAP_AVAILABLE}")
        self.logger.info(f"   LIME available: {LIME_AVAILABLE}")
    
    @handles_errors(exceptions=(ValueError, RuntimeError), default_return=False)
    async def execute(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Execute Step 8: Advanced Feature Selection.
        
        Args:
            training_input: Input data from previous steps
            pipeline_state: Current pipeline state
            
        Returns:
            Updated pipeline state with selected features
        """
        try:
            start_time = datetime.now()
            self.logger.info("🚀 Starting Step 8: Advanced Feature Selection...")
            
            # Extract parameters
            symbol = training_input.get("symbol", "UNKNOWN")
            exchange = training_input.get("exchange", "UNKNOWN")
            timeframe = training_input.get("timeframe", "1m")
            
            # Load filtered features from step07
            filtered_train_path = f"data/training/{exchange}_{symbol}_{timeframe}_features_filtered_train.parquet"
            filtered_val_path = f"data/training/{exchange}_{symbol}_{timeframe}_features_filtered_val.parquet"
            
            if not os.path.exists(filtered_train_path):
                # Fallback to original features if filtered not available
                self.logger.warning("⚠️ Filtered features not found, using original features")
                filtered_train_path = f"data/training/{exchange}_{symbol}_{timeframe}_features_train.parquet"
                filtered_val_path = f"data/training/{exchange}_{symbol}_{timeframe}_features_val.parquet"
            
            self.logger.info(f"📊 Loading features from: {filtered_train_path}")
            
            # Load data
            df_train = pd.read_parquet(filtered_train_path)
            df_val = pd.read_parquet(filtered_val_path)
            df = pd.concat([df_train, df_val], ignore_index=True)
            
            self.logger.info(f"📈 Loaded {len(df)} rows with {len(df.columns)} columns")
            
            # Separate features and labels
            label_columns = ['target', 'direction', 'profit', 'outcome', 'returns', 'timestamp',
                           'open', 'high', 'low', 'close', 'volume']
            feature_columns = [col for col in df.columns if col not in label_columns]
            
            features_df = df[feature_columns]
            labels_df = df[[col for col in label_columns if col in df.columns]]
            
            # Extract target
            if 'target' in labels_df.columns:
                y = labels_df['target']
            elif 'direction' in labels_df.columns:
                y = labels_df['direction']
            else:
                raise ValueError("No target or direction column found")
            
            # Ensure binary target
            if y.dtype != int:
                y = (y > 0).astype(int)
            
            # Load regime labels if available
            regime_labels = None
            hmm_path = f"data/hmm_regimes/{exchange}_{symbol}_{timeframe}_composite_clusters.parquet"
            if os.path.exists(hmm_path):
                self.logger.info(f"🎭 Loading regime labels from: {hmm_path}")
                hmm_data = pd.read_parquet(hmm_path)
                if "composite_cluster_id" in hmm_data.columns:
                    regime_labels = hmm_data["composite_cluster_id"].iloc[:len(df)]
            
            # Phase 1: mRMR and Random Forest Selection
            self.logger.info("📊 Starting Phase 1: mRMR/RF Selection...")
            phase1_features, phase1_metadata = await self.phase1_mrmr_rf_selection(
                features_df, y, regime_labels
            )
            
            # Phase 2: Boruta Multi-Target Selection
            self.logger.info("🎯 Starting Phase 2: Boruta Multi-Target Selection...")
            phase2_results, interpretability_results = await self.phase2_boruta_multi_target(
                phase1_features, y, regime_labels
            )
            
            # Save results
            output_files = await self._save_selection_results(
                phase1_features, phase1_metadata, phase2_results, 
                interpretability_results, symbol, exchange, timeframe,
                df_train, df_val, labels_df
            )
            
            # Update pipeline state
            pipeline_state["step08_advanced_feature_selection"] = {
                "status": "completed",
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "output_files": output_files,
                "phase1_metadata": phase1_metadata,
                "phase2_results": {k: v for k, v in phase2_results.items() if k != 'features'},
                "interpretability_results": interpretability_results,
                "original_features": len(feature_columns),
                "phase1_features": len(phase1_features.columns),
                "phase2_feature_sets": {f"top_{k}": len(v['features']) for k, v in phase2_results.items()},
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe
            }
            
            self.logger.info("✅ Step 8: Advanced Feature Selection completed successfully")
            
            return pipeline_state
            
        except Exception as e:
            self.logger.error(f"❌ Step 8 failed: {str(e)}")
            pipeline_state["step08_advanced_feature_selection"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            return pipeline_state
    
    async def phase1_mrmr_rf_selection(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        regime_labels: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, dict[str, Any]]:
        """
        Phase 1: Select top 150 features using mRMR and Random Forest.
        
        Args:
            X: Feature dataframe
            y: Target series
            regime_labels: Optional regime labels
            
        Returns:
            Selected features and metadata
        """
        metadata = {}
        
        # 1. mRMR Selection
        mrmr_features = []
        if self.enable_mrmr:
            self.logger.info("🔍 Running mRMR selection...")
            mrmr_features = self._mrmr_selection(X, y, self.phase1_target_features)
            metadata['mrmr_features'] = mrmr_features
            self.logger.info(f"   mRMR selected {len(mrmr_features)} features")
        
        # 2. Random Forest importance with time-series validation
        rf_features = []
        if self.enable_rf_importance:
            self.logger.info("🌳 Running Random Forest selection with TS validation...")
            rf_features = self._time_series_rf_selection(X, y, self.phase1_target_features)
            metadata['rf_features'] = rf_features
            self.logger.info(f"   RF selected {len(rf_features)} features")
        
        # 3. Per-regime validation
        regime_validated_features = []
        if regime_labels is not None:
            self.logger.info("🎭 Validating features per regime...")
            candidate_features = list(set(mrmr_features) | set(rf_features))
            regime_validated_features = self._validate_features_per_regime(
                X, y, regime_labels, candidate_features
            )
            metadata['regime_validated_features'] = regime_validated_features
        
        # 4. Ensemble the results
        consensus_features = list(set(mrmr_features) & set(rf_features))
        metadata['consensus_features'] = consensus_features
        
        # Build final feature set
        final_features = list(consensus_features)
        remaining_slots = self.phase1_target_features - len(final_features)
        
        # Add regime-validated features
        for feature in regime_validated_features:
            if feature not in final_features and remaining_slots > 0:
                final_features.append(feature)
                remaining_slots -= 1
        
        # Add remaining mRMR features
        for feature in mrmr_features:
            if feature not in final_features and remaining_slots > 0:
                final_features.append(feature)
                remaining_slots -= 1
        
        # Add remaining RF features
        for feature in rf_features:
            if feature not in final_features and remaining_slots > 0:
                final_features.append(feature)
                remaining_slots -= 1
        
        # Ensure we have enough features
        if len(final_features) < self.phase1_target_features:
            # Add remaining features by mutual information
            mi_scores = mutual_info_classif(X, y, random_state=42)
            mi_ranking = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
            
            for feature in mi_ranking.index:
                if feature not in final_features and len(final_features) < self.phase1_target_features:
                    final_features.append(feature)
        
        metadata['final_features_count'] = len(final_features)
        metadata['consensus_ratio'] = len(consensus_features) / len(final_features) if final_features else 0
        metadata['regime_specific_additions'] = len([f for f in final_features if f in regime_validated_features])
        
        self.logger.info(f"✅ Phase 1 complete: {len(X.columns)} → {len(final_features)} features")
        self.logger.info(f"   Consensus features: {len(consensus_features)}")
        self.logger.info(f"   Regime-specific additions: {metadata['regime_specific_additions']}")
        
        return X[final_features], metadata
    
    def _mrmr_selection(self, X: pd.DataFrame, y: pd.Series, n_features: int) -> List[str]:
        """
        Minimum Redundancy Maximum Relevance feature selection.
        
        Args:
            X: Feature dataframe
            y: Target series
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        selected_features = []
        remaining_features = list(X.columns)
        
        # First feature: highest MI with target
        mi_scores = mutual_info_classif(X, y, random_state=42)
        first_feature_idx = np.argmax(mi_scores)
        selected_features.append(X.columns[first_feature_idx])
        remaining_features.remove(X.columns[first_feature_idx])
        
        # Iteratively add features
        while len(selected_features) < n_features and remaining_features:
            scores = {}
            
            for feature in remaining_features:
                # Relevance: MI with target
                relevance = mutual_info_classif(
                    X[[feature]], y, random_state=42
                )[0]
                
                # Redundancy: average MI with selected features
                redundancy = 0
                for selected in selected_features:
                    # Use correlation as proxy for MI between features (faster)
                    redundancy += abs(X[feature].corr(X[selected]))
                redundancy /= len(selected_features)
                
                # mRMR score
                scores[feature] = relevance - redundancy
            
            # Select feature with highest score
            best_feature = max(scores, key=scores.get)
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
        
        return selected_features
    
    def _time_series_rf_selection(self, X: pd.DataFrame, y: pd.Series, n_features: int) -> List[str]:
        """
        Random Forest feature selection with time-series cross-validation.
        
        Args:
            X: Feature dataframe
            y: Target series
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        # Train RF with time-series splits
        tscv = TimeSeriesSplit(n_splits=min(self.n_splits_ts, 3))  # Limit splits for speed
        feature_importances = np.zeros(X.shape[1])
        
        for train_idx, val_idx in tscv.split(X):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            
            # Train RF
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)
            
            # Accumulate importances
            feature_importances += rf.feature_importances_
        
        # Average importances
        feature_importances /= tscv.get_n_splits()
        
        # Select top features
        top_indices = np.argsort(feature_importances)[-n_features:]
        return X.columns[top_indices].tolist()
    
    def _validate_features_per_regime(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        regime_labels: pd.Series,
        candidate_features: List[str]
    ) -> List[str]:
        """
        Validate features perform well in each regime.
        
        Args:
            X: Feature dataframe
            y: Target series
            regime_labels: Regime labels
            candidate_features: Features to validate
            
        Returns:
            List of regime-validated features
        """
        regime_scores = {feature: [] for feature in candidate_features}
        
        for regime in np.unique(regime_labels):
            regime_mask = regime_labels == regime
            if regime_mask.sum() < self.min_regime_samples:
                continue
            
            X_regime = X[regime_mask]
            y_regime = y[regime_mask]
            
            # Evaluate each feature in this regime
            for feature in candidate_features:
                if feature in X_regime.columns:
                    # Simple univariate test
                    mi_score = mutual_info_classif(
                        X_regime[[feature]], y_regime, random_state=42
                    )[0]
                    regime_scores[feature].append(mi_score)
        
        # Select features that perform well across regimes
        validated_features = []
        for feature, scores in regime_scores.items():
            if scores and np.mean(scores) > 0.01:  # Threshold for relevance
                validated_features.append(feature)
        
        return validated_features
    
    async def phase2_boruta_multi_target(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        regime_labels: Optional[pd.Series] = None
    ) -> Tuple[dict[str, Any], dict[str, Any]]:
        """
        Phase 2: Boruta selection with redundancy analysis for multiple target sizes.
        
        Args:
            X: Feature dataframe (already filtered to ~150 features)
            y: Target series
            regime_labels: Optional regime labels
            
        Returns:
            Feature sets and interpretability results
        """
        feature_sets = {}
        
        # 1. Perform redundancy analysis
        redundancy_groups = {}
        feature_clusters = {}
        if self.enable_redundancy_analysis:
            self.logger.info("🔄 Analyzing feature redundancy...")
            redundancy_groups = self._analyze_feature_redundancy(X)
            self.logger.info(f"   Found {len(redundancy_groups)} redundancy groups")
            
            # Advanced clustering-based redundancy analysis
            self.logger.info("🔍 Performing hierarchical clustering for redundancy...")
            feature_clusters = self._hierarchical_feature_clustering(X)
            self.logger.info(f"   Identified {len(feature_clusters)} feature clusters")
        
        if BORUTA_AVAILABLE:
            # Run Boruta
            self.logger.info("🔍 Running Boruta for all-relevant features...")
            
            # Configure Boruta
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            boruta_selector = BorutaPy(
                rf,
                n_estimators='auto',
                alpha=self.boruta_alpha,
                max_iter=self.boruta_max_iter,
                random_state=42
            )
            
            # Fit Boruta
            boruta_selector.fit(X.values, y.values)
            
            # Get feature rankings
            feature_ranks = boruta_selector.ranking_
            feature_importance = pd.Series(
                1 / feature_ranks,  # Convert rank to importance
                index=X.columns
            ).sort_values(ascending=False)
            
            # Get confirmed features
            confirmed_features = X.columns[boruta_selector.support_].tolist()
            self.logger.info(f"   Boruta confirmed {len(confirmed_features)} features")
            
        else:
            # Fallback: Use LightGBM importance
            self.logger.warning("⚠️ Boruta not available, using LightGBM importance")
            
            lgb_model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            lgb_model.fit(X, y)
            
            feature_importance = pd.Series(
                lgb_model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)
            
            # Consider top 80% as "confirmed"
            threshold = feature_importance.quantile(0.2)
            confirmed_features = feature_importance[feature_importance > threshold].index.tolist()
        
        # Create feature sets for each target size
        for target_size in self.phase2_targets:
            self.logger.info(f"📊 Creating redundancy-aware feature set with {target_size} features...")
            
            # Select features with redundancy consideration
            if self.enable_redundancy_analysis and (redundancy_groups or feature_clusters):
                # Combine redundancy groups from both methods
                all_redundancy_groups = dict(redundancy_groups)
                
                # Add hierarchical clusters to redundancy groups
                for cluster_id, cluster_features in feature_clusters.items():
                    all_redundancy_groups[f'cluster_{cluster_id}'] = cluster_features
                
                # Use advanced selection with combined redundancy information
                selected_features = self._select_features_with_redundancy_advanced(
                    feature_importance,
                    all_redundancy_groups,
                    target_size,
                    confirmed_features,
                    boruta_selector if BORUTA_AVAILABLE else None
                )
            else:
                # Fallback to simple top selection
                selected_features = feature_importance.head(target_size).index.tolist()
            
            # Validate with time-series CV
            ts_validation = self._time_series_validate_features(
                X[selected_features], y, n_splits=self.n_splits_ts
            )
            
            # Validate per regime if available
            regime_validation = {}
            if regime_labels is not None:
                regime_validation = self._per_regime_validate_features(
                    X[selected_features], y, regime_labels
                )
            
            # Calculate redundancy statistics
            redundancy_stats = self._calculate_redundancy_stats(
                selected_features, redundancy_groups
            ) if redundancy_groups else {}
            
            feature_sets[target_size] = {
                'features': selected_features,
                'importance_scores': feature_importance[selected_features].to_dict(),
                'ts_validation': ts_validation,
                'regime_validation': regime_validation,
                'boruta_confirmed': len([f for f in selected_features if f in confirmed_features]),
                'boruta_confirmed_ratio': len([f for f in selected_features if f in confirmed_features]) / len(selected_features),
                'redundancy_stats': redundancy_stats
            }
            
            self.logger.info(f"   TS validation score: {ts_validation['mean_score']:.4f} ± {ts_validation['std_score']:.4f}")
            self.logger.info(f"   Boruta confirmed: {feature_sets[target_size]['boruta_confirmed']} features")
            
            if redundancy_stats:
                self.logger.info(f"   Redundancy groups: {redundancy_stats['groups_represented']}")
                self.logger.info(f"   Average redundancy: {redundancy_stats['average_redundancy']:.1f} features/group")
                self.logger.info(f"   Concept coverage: {sum(redundancy_stats['concept_coverage'].values())} features across {len([v for v in redundancy_stats['concept_coverage'].values() if v > 0])} concepts")
        
        # Generate interpretability analysis
        self.logger.info("🔮 Generating interpretability analysis...")
        interpretability_results = await self._generate_interpretability_report(
            X, y, feature_sets
        )
        
        return feature_sets, interpretability_results
    
    def _time_series_validate_features(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        n_splits: int = 5
    ) -> dict[str, Any]:
        """Time-series aware feature validation."""
        tscv = TimeSeriesSplit(n_splits=min(n_splits, 3))  # Limit for speed
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train simple model
            model = lgb.LGBMClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred)
            scores.append(score)
        
        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'scores': scores,
            'n_splits': len(scores)
        }
    
    def _per_regime_validate_features(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        regime_labels: pd.Series
    ) -> dict[str, float]:
        """Validate features perform well in each regime."""
        regime_scores = {}
        
        for regime in np.unique(regime_labels):
            regime_mask = regime_labels == regime
            if regime_mask.sum() < self.min_regime_samples:
                continue
            
            X_regime = X[regime_mask]
            y_regime = y[regime_mask]
            
            # Cross-validate within regime
            try:
                scores = cross_val_score(
                    lgb.LGBMClassifier(n_estimators=50, max_depth=5, verbose=-1),
                    X_regime, y_regime,
                    cv=min(3, len(np.unique(y_regime))),  # Handle small regimes
                    scoring='roc_auc'
                )
                regime_scores[f'regime_{regime}'] = scores.mean()
            except:
                # Skip if validation fails (e.g., single class in regime)
                continue
        
        return regime_scores
    
    async def _generate_interpretability_report(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        feature_sets: dict[int, dict[str, Any]]
    ) -> dict[str, Any]:
        """Generate SHAP/LIME interpretability analysis."""
        report = {}
        
        for size, feature_data in feature_sets.items():
            self.logger.info(f"🔍 Analyzing interpretability for {size}-feature set...")
            
            features = feature_data['features']
            X_subset = X[features]
            
            # Train model for interpretability
            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            model.fit(X_subset, y)
            
            feature_report = {}
            
            # SHAP analysis
            if self.enable_shap and SHAP_AVAILABLE:
                try:
                    explainer = shap.TreeExplainer(model)
                    
                    # Sample for efficiency
                    sample_size = min(1000, len(X_subset))
                    sample_idx = np.random.choice(len(X_subset), sample_size, replace=False)
                    X_sample = X_subset.iloc[sample_idx]
                    
                    shap_values = explainer.shap_values(X_sample)
                    
                    # Handle binary classification output
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]  # Use positive class
                    
                    # Calculate importance
                    shap_importance = pd.Series(
                        np.abs(shap_values).mean(axis=0),
                        index=features
                    ).sort_values(ascending=False)
                    
                    feature_report['shap_importance'] = shap_importance.head(20).to_dict()
                    
                    # Detect interactions
                    feature_report['feature_interactions'] = self._detect_feature_interactions(
                        shap_values, features
                    )
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ SHAP analysis failed: {e}")
                    feature_report['shap_error'] = str(e)
            
            # LIME analysis
            if self.enable_lime and LIME_AVAILABLE:
                try:
                    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                        X_subset.values,
                        feature_names=features,
                        class_names=['0', '1'],
                        mode='classification'
                    )
                    
                    # Sample explanations
                    sample_explanations = []
                    for i in range(min(self.n_lime_samples, len(X_subset))):
                        exp = lime_explainer.explain_instance(
                            X_subset.iloc[i].values,
                            model.predict_proba,
                            num_features=min(10, len(features))
                        )
                        sample_explanations.append(exp.as_list())
                    
                    feature_report['lime_explanations'] = sample_explanations[:3]  # Store first 3
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ LIME analysis failed: {e}")
                    feature_report['lime_error'] = str(e)
            
            # Model performance
            y_pred = model.predict_proba(X_subset)[:, 1]
            feature_report['model_performance'] = {
                'roc_auc': roc_auc_score(y, y_pred),
                'accuracy': accuracy_score(y, model.predict(X_subset)),
                'f1_score': f1_score(y, model.predict(X_subset))
            }
            
            report[f'feature_set_{size}'] = feature_report
        
        return report
    
    def _detect_feature_interactions(
        self, 
        shap_values: np.ndarray, 
        feature_names: List[str],
        top_k: int = 10
    ) -> List[Tuple[str, str, float]]:
        """Detect top feature interactions from SHAP values."""
        interactions = []
        
        # Calculate correlation of SHAP values between features
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        corr_matrix = shap_df.corr().abs()
        
        # Get top interactions (excluding diagonal)
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                interactions.append((
                    feature_names[i],
                    feature_names[j],
                    corr_matrix.iloc[i, j]
                ))
        
        # Sort and return top interactions
        interactions.sort(key=lambda x: x[2], reverse=True)
        return [(f1, f2, round(score, 3)) for f1, f2, score in interactions[:top_k]]
    
    def _hierarchical_feature_clustering(self, X: pd.DataFrame, n_clusters: int = None) -> dict[int, List[str]]:
        """
        Perform hierarchical clustering on features to identify redundant groups.
        Uses correlation distance and Ward linkage.
        
        Args:
            X: Feature dataframe
            n_clusters: Number of clusters (auto-determined if None)
            
        Returns:
            Dictionary mapping cluster IDs to feature lists
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Convert to distance matrix (1 - correlation)
        distance_matrix = 1 - corr_matrix
        
        # Convert to condensed distance matrix
        condensed_distances = squareform(distance_matrix, checks=False)
        
        # Perform hierarchical clustering
        Z = linkage(condensed_distances, method='ward')
        
        # Determine optimal number of clusters if not specified
        if n_clusters is None:
            # Use elbow method - cut at distance where gap is largest
            distances = Z[:, 2]
            gaps = np.diff(distances)
            optimal_idx = np.argmax(gaps) + 1
            distance_threshold = distances[optimal_idx]
            clusters = fcluster(Z, distance_threshold, criterion='distance')
        else:
            clusters = fcluster(Z, n_clusters, criterion='maxclust')
        
        # Group features by cluster
        feature_clusters = {}
        for idx, cluster_id in enumerate(clusters):
            if cluster_id not in feature_clusters:
                feature_clusters[cluster_id] = []
            feature_clusters[cluster_id].append(X.columns[idx])
        
        # Filter out singleton clusters
        feature_clusters = {k: v for k, v in feature_clusters.items() if len(v) > 1}
        
        return feature_clusters
    
    def _analyze_feature_redundancy(self, X: pd.DataFrame) -> dict[str, List[str]]:
        """
        Analyze feature redundancy to identify groups of correlated features.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Dictionary mapping group names to feature lists
        """
        redundancy_groups = {}
        
        # 1. Correlation-based redundancy
        corr_matrix = X.corr().abs()
        
        # Find highly correlated feature pairs
        high_corr_pairs = []
        for i in range(len(X.columns)):
            for j in range(i + 1, len(X.columns)):
                if corr_matrix.iloc[i, j] >= self.min_redundancy_correlation:
                    high_corr_pairs.append((X.columns[i], X.columns[j], corr_matrix.iloc[i, j]))
        
        # Group correlated features using connected components
        from collections import defaultdict
        feature_graph = defaultdict(set)
        for f1, f2, _ in high_corr_pairs:
            feature_graph[f1].add(f2)
            feature_graph[f2].add(f1)
        
        # Find connected components
        visited = set()
        corr_group_id = 0
        for feature in feature_graph:
            if feature not in visited:
                # BFS to find connected component
                component = set()
                queue = [feature]
                while queue:
                    current = queue.pop(0)
                    if current not in visited:
                        visited.add(current)
                        component.add(current)
                        queue.extend(feature_graph[current] - visited)
                
                if len(component) > 1:
                    redundancy_groups[f'corr_group_{corr_group_id}'] = list(component)
                    corr_group_id += 1
        
        # 2. Concept-based redundancy
        for concept, patterns in self.feature_concept_patterns.items():
            concept_features = []
            for feature in X.columns:
                feature_lower = feature.lower()
                if any(pattern in feature_lower for pattern in patterns):
                    concept_features.append(feature)
            
            if len(concept_features) > 1:
                # Only add if not already in correlation groups
                new_features = []
                for f in concept_features:
                    if not any(f in group for group in redundancy_groups.values()):
                        new_features.append(f)
                
                if len(new_features) > 1:
                    redundancy_groups[f'concept_{concept}'] = new_features
        
        return redundancy_groups
    
    def _select_features_with_redundancy(
        self, 
        feature_importance: pd.Series,
        redundancy_groups: dict[str, List[str]],
        target_size: int,
        confirmed_features: List[str]
    ) -> List[str]:
        """
        Select features considering redundancy to ensure robustness.
        
        Args:
            feature_importance: Feature importance scores
            redundancy_groups: Dictionary of redundancy groups
            target_size: Target number of features
            confirmed_features: Boruta-confirmed features
            
        Returns:
            List of selected features
        """
        selected_features = []
        used_groups = set()
        feature_to_groups = {}
        
        # Create reverse mapping
        for group_name, features in redundancy_groups.items():
            for feature in features:
                if feature not in feature_to_groups:
                    feature_to_groups[feature] = []
                feature_to_groups[feature].append(group_name)
        
        # First pass: Select best features, considering redundancy
        for feature in feature_importance.index:
            if len(selected_features) >= target_size:
                break
                
            # Check if feature belongs to any redundancy group
            if feature in feature_to_groups:
                groups = feature_to_groups[feature]
                
                # Count how many features from each group are already selected
                group_counts = {}
                for group in groups:
                    group_features = redundancy_groups[group]
                    count = sum(1 for f in selected_features if f in group_features)
                    group_counts[group] = count
                
                # Allow selection if we don't have enough redundancy for any group
                min_count = min(group_counts.values()) if group_counts else 0
                if min_count < self.redundancy_groups_per_concept:
                    selected_features.append(feature)
                    for group in groups:
                        used_groups.add(group)
            else:
                # Feature not in any redundancy group, select it
                selected_features.append(feature)
        
        # Second pass: Ensure minimum redundancy for important groups
        if len(selected_features) < target_size:
            for group_name, group_features in redundancy_groups.items():
                if len(selected_features) >= target_size:
                    break
                    
                # Count current features from this group
                current_count = sum(1 for f in selected_features if f in group_features)
                
                # Add more if needed
                if current_count < self.redundancy_groups_per_concept:
                    # Sort group features by importance
                    group_importance = feature_importance[
                        feature_importance.index.isin(group_features)
                    ].sort_values(ascending=False)
                    
                    for feature in group_importance.index:
                        if feature not in selected_features and len(selected_features) < target_size:
                            selected_features.append(feature)
                            current_count += 1
                            if current_count >= self.redundancy_groups_per_concept:
                                break
        
        # Third pass: Fill remaining slots with best available features
        while len(selected_features) < target_size:
            for feature in feature_importance.index:
                if feature not in selected_features:
                    selected_features.append(feature)
                    break
            else:
                break  # No more features available
        
        # Prioritize Boruta-confirmed features
        confirmed_selected = [f for f in selected_features if f in confirmed_features]
        unconfirmed_selected = [f for f in selected_features if f not in confirmed_features]
        
        # Reorder to put confirmed features first
        final_features = confirmed_selected + unconfirmed_selected
        
        return final_features[:target_size]
    
    def _select_features_with_redundancy_advanced(
        self, 
        feature_importance: pd.Series,
        all_redundancy_groups: dict[str, List[str]],
        target_size: int,
        confirmed_features: List[str],
        boruta_selector = None
    ) -> List[str]:
        """
        Advanced feature selection that combines Boruta's all-relevant features
        with redundancy reduction using multiple strategies.
        
        Args:
            feature_importance: Feature importance scores
            all_redundancy_groups: Combined redundancy groups from multiple methods
            target_size: Target number of features
            confirmed_features: Boruta-confirmed features
            boruta_selector: Fitted Boruta selector (optional)
            
        Returns:
            List of selected features with optimal redundancy
        """
        selected_features = []
        
        # Strategy 1: Start with Boruta-confirmed features, but limit redundancy
        if confirmed_features:
            # Group confirmed features by redundancy
            confirmed_by_group = {}
            ungrouped_confirmed = []
            
            for feature in confirmed_features:
                assigned = False
                for group_name, group_features in all_redundancy_groups.items():
                    if feature in group_features:
                        if group_name not in confirmed_by_group:
                            confirmed_by_group[group_name] = []
                        confirmed_by_group[group_name].append(feature)
                        assigned = True
                        break
                if not assigned:
                    ungrouped_confirmed.append(feature)
            
            # Add best confirmed features from each group (up to redundancy limit)
            for group_name, group_confirmed in confirmed_by_group.items():
                # Sort by importance within group
                group_importance = feature_importance[group_confirmed].sort_values(ascending=False)
                # Take up to redundancy_groups_per_concept from each group
                n_to_take = min(self.redundancy_groups_per_concept, len(group_importance))
                selected_features.extend(group_importance.head(n_to_take).index.tolist())
            
            # Add all ungrouped confirmed features (they're not redundant)
            selected_features.extend(ungrouped_confirmed)
        
        # Strategy 2: Use VIF (Variance Inflation Factor) for remaining features
        remaining_slots = target_size - len(selected_features)
        if remaining_slots > 0:
            # Get remaining features sorted by importance
            remaining_features = [f for f in feature_importance.index if f not in selected_features]
            
            # Calculate VIF-based selection
            vif_selected = self._select_low_vif_features(
                feature_importance[remaining_features],
                all_redundancy_groups,
                remaining_slots,
                selected_features
            )
            selected_features.extend(vif_selected)
        
        # Strategy 3: Ensure diversity across feature concepts
        if len(selected_features) < target_size:
            # Check concept coverage
            concept_coverage = {}
            for concept, patterns in self.feature_concept_patterns.items():
                concept_features = [f for f in selected_features 
                                  if any(p in f.lower() for p in patterns)]
                concept_coverage[concept] = len(concept_features)
            
            # Add features from underrepresented concepts
            for concept, count in sorted(concept_coverage.items(), key=lambda x: x[1]):
                if len(selected_features) >= target_size:
                    break
                
                if count < 2:  # Ensure at least 2 features per concept
                    patterns = self.feature_concept_patterns[concept]
                    concept_candidates = [f for f in feature_importance.index 
                                        if any(p in f.lower() for p in patterns) 
                                        and f not in selected_features]
                    
                    # Add best features from this concept
                    for feature in feature_importance[concept_candidates].sort_values(ascending=False).index:
                        if len(selected_features) < target_size:
                            selected_features.append(feature)
                            count += 1
                            if count >= 2:
                                break
        
        # Final adjustment: Replace low-importance redundant features
        if boruta_selector is not None and hasattr(boruta_selector, 'ranking_'):
            # Get Boruta rankings
            boruta_ranks = dict(zip(feature_importance.index, boruta_selector.ranking_))
            
            # Identify redundant features in selection
            redundant_pairs = []
            for i, f1 in enumerate(selected_features):
                for j, f2 in enumerate(selected_features[i+1:], i+1):
                    for group_features in all_redundancy_groups.values():
                        if f1 in group_features and f2 in group_features:
                            # Keep the one with better Boruta rank
                            if boruta_ranks.get(f1, float('inf')) > boruta_ranks.get(f2, float('inf')):
                                redundant_pairs.append((i, f1))  # Remove f1
                            else:
                                redundant_pairs.append((j, f2))  # Remove f2
                            break
            
            # Remove redundant features and replace with non-redundant ones
            removed_indices = set()
            for idx, feature in redundant_pairs:
                if idx not in removed_indices and len(selected_features) > target_size:
                    removed_indices.add(idx)
            
            # Remove in reverse order to maintain indices
            for idx in sorted(removed_indices, reverse=True):
                selected_features.pop(idx)
        
        return selected_features[:target_size]
    
    def _select_low_vif_features(
        self,
        candidate_importance: pd.Series,
        redundancy_groups: dict[str, List[str]],
        n_features: int,
        already_selected: List[str]
    ) -> List[str]:
        """
        Select features with low VIF (Variance Inflation Factor) to minimize multicollinearity.
        """
        selected = []
        
        for feature in candidate_importance.index:
            if len(selected) >= n_features:
                break
            
            # Check if adding this feature would create high multicollinearity
            redundancy_score = 0
            for group_name, group_features in redundancy_groups.items():
                if feature in group_features:
                    # Count how many from this group are already selected
                    existing_count = sum(1 for f in already_selected + selected if f in group_features)
                    redundancy_score += existing_count
            
            # Select if redundancy is acceptable
            if redundancy_score < self.redundancy_groups_per_concept:
                selected.append(feature)
        
        return selected
    
    def _calculate_redundancy_stats(
        self, 
        selected_features: List[str],
        redundancy_groups: dict[str, List[str]]
    ) -> dict[str, Any]:
        """
        Calculate redundancy statistics for selected features.
        
        Args:
            selected_features: List of selected features
            redundancy_groups: Dictionary of redundancy groups
            
        Returns:
            Dictionary of redundancy statistics
        """
        stats = {
            'groups_represented': 0,
            'average_redundancy': 0,
            'min_redundancy': float('inf'),
            'max_redundancy': 0,
            'concept_coverage': {},
            'group_feature_counts': {}
        }
        
        # Calculate group representation
        for group_name, group_features in redundancy_groups.items():
            count = sum(1 for f in selected_features if f in group_features)
            if count > 0:
                stats['groups_represented'] += 1
                stats['group_feature_counts'][group_name] = count
                stats['min_redundancy'] = min(stats['min_redundancy'], count)
                stats['max_redundancy'] = max(stats['max_redundancy'], count)
        
        # Calculate average redundancy
        if stats['group_feature_counts']:
            stats['average_redundancy'] = sum(stats['group_feature_counts'].values()) / len(stats['group_feature_counts'])
        else:
            stats['min_redundancy'] = 0
        
        # Calculate concept coverage
        for concept in self.feature_concept_patterns:
            concept_features = [f for f in selected_features 
                              if any(p in f.lower() for p in self.feature_concept_patterns[concept])]
            stats['concept_coverage'][concept] = len(concept_features)
        
        return stats
    
    async def _save_selection_results(
        self,
        phase1_features: pd.DataFrame,
        phase1_metadata: dict[str, Any],
        phase2_results: dict[int, dict[str, Any]],
        interpretability_results: dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        labels_df: pd.DataFrame
    ) -> dict[str, str]:
        """Save all selection results and create output datasets."""
        output_files = {}
        
        # Save phase 1 results
        phase1_path = os.path.join(
            self.output_dir,
            f"{exchange}_{symbol}_{timeframe}_phase1_features.json"
        )
        safe_json_dump({
            'features': phase1_features.columns.tolist(),
            'metadata': phase1_metadata,
            'timestamp': datetime.now().isoformat()
        }, phase1_path)
        output_files['phase1_results'] = phase1_path
        
        # Save phase 2 results for each target size
        for target_size, results in phase2_results.items():
            # Save feature list and metadata
            phase2_path = os.path.join(
                self.output_dir,
                f"{exchange}_{symbol}_{timeframe}_top{target_size}_features.json"
            )
            safe_json_dump({
                'features': results['features'],
                'importance_scores': results['importance_scores'],
                'validation': {
                    'ts_validation': results['ts_validation'],
                    'regime_validation': results['regime_validation']
                },
                'boruta_stats': {
                    'confirmed': results['boruta_confirmed'],
                    'confirmed_ratio': results['boruta_confirmed_ratio']
                },
                'timestamp': datetime.now().isoformat()
            }, phase2_path)
            output_files[f'top{target_size}_features'] = phase2_path
            
            # Create and save filtered datasets
            selected_features = results['features']
            
            # Split back to train/val
            train_size = len(df_train)
            
            # Create filtered train dataset
            train_features = phase1_features[selected_features].iloc[:train_size]
            train_data = pd.concat([train_features, labels_df.iloc[:train_size]], axis=1)
            train_path = os.path.join(
                self.output_dir,
                f"{exchange}_{symbol}_{timeframe}_top{target_size}_train.parquet"
            )
            train_data.to_parquet(train_path)
            output_files[f'top{target_size}_train'] = train_path
            
            # Create filtered val dataset
            val_features = phase1_features[selected_features].iloc[train_size:]
            val_data = pd.concat([val_features, labels_df.iloc[train_size:]], axis=1)
            val_path = os.path.join(
                self.output_dir,
                f"{exchange}_{symbol}_{timeframe}_top{target_size}_val.parquet"
            )
            val_data.to_parquet(val_path)
            output_files[f'top{target_size}_val'] = val_path
        
        # Save interpretability results
        interp_path = os.path.join(
            self.output_dir,
            f"{exchange}_{symbol}_{timeframe}_interpretability_report.json"
        )
        safe_json_dump(interpretability_results, interp_path)
        output_files['interpretability_report'] = interp_path
        
        # Save comprehensive selection report
        report_path = os.path.join(
            self.output_dir,
            f"{exchange}_{symbol}_{timeframe}_selection_report.json"
        )
        safe_json_dump({
            'phase1_summary': {
                'input_features': len(df_train.columns) - len(labels_df.columns),
                'output_features': len(phase1_features.columns),
                'consensus_features': len(phase1_metadata.get('consensus_features', [])),
                'regime_validated': phase1_metadata.get('regime_specific_additions', 0)
            },
            'phase2_summary': {
                f'top_{size}': {
                    'features': len(results['features']),
                    'ts_score': results['ts_validation']['mean_score'],
                    'boruta_confirmed': results['boruta_confirmed']
                }
                for size, results in phase2_results.items()
            },
            'timestamp': datetime.now().isoformat()
        }, report_path)
        output_files['selection_report'] = report_path
        
        self.logger.info(f"💾 Saved all selection results to {self.output_dir}")
        
        return output_files


# Step execution function
async def run_step(
    symbol: str,
    exchange: str,
    timeframe: str = "1m",
    data_dir: str = None,
    force_rerun: bool = False,
    **kwargs: Any,
) -> bool:
    """
    Run Step 8: Advanced Feature Selection.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        data_dir: Data directory
        force_rerun: Force rerun the step
        **kwargs: Additional arguments
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load configuration
        config_path = "config/training_config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Override with kwargs
        config.update(kwargs)
        
        # Create training input
        training_input = {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            "data_dir": data_dir or f"data/{exchange}/{symbol}",
        }
        
        # Initialize pipeline state
        pipeline_state = {}
        
        # Create and execute step
        step = Step08AdvancedFeatureSelection(config)
        result = await step.execute(training_input, pipeline_state)
        
        # Check if successful
        if result.get("step08_advanced_feature_selection", {}).get("status") == "completed":
            system_logger.info("✅ Step 8: Advanced Feature Selection completed successfully")
            return True
        else:
            system_logger.error("❌ Step 8: Advanced Feature Selection failed")
            return False
            
    except Exception as e:
        system_logger.error(f"❌ Error running Step 8: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    asyncio.run(run_step(
        symbol="BTCUSDT",
        exchange="binance",
        timeframe="1m",
        force_rerun=True
    ))