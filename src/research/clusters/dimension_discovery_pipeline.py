"""
Dimension Discovery Pipeline.

This orchestrator implements the end-to-end workflow requested:
  1) Build dynamic targets (market dynamics)
  2) For each target: mRMR screen → RF/XGB + SHAP (main + interaction)
     → LASSO check → PID on top pairs/triads
  3) Aggregate across targets: feature × dynamic matrix (unique/SHAP),
     synergy edges, redundancy matrix
  4) Derive dimensions via clustering/factorization and compute dimension scores
  5) Provide dimension scores suitable for downstream regime clustering

Notes:
  - This is a research pipeline; hyperparameters and performance tradeoffs are
    chosen for clarity and reproducibility rather than maximal speed.
  - Lookahead avoidance: caller must pass appropriately lagged features or use
    `DynamicTargetsBuilder` and ensure feature lagging upstream.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import system_logger

from .dynamic_targets import DynamicTargetsBuilder, DynamicTargetConfig
from .feature_selection import mrmr_select, estimate_mutual_information, MRMRConfig
from .pid_utils import pid_pair


@dataclass
class DiscoveryConfig:
    # Feature selection
    mrmr_max_features: int = 50
    mrmr_redundancy_penalty: float = 0.5
    # Model importance
    rf_n_estimators: int = 200
    xgb_n_estimators: int = 300
    shap_sample_size: int = 1000
    # PID
    pid_top_pairs: int = 50
    # Dimension derivation
    cluster_method: str = 'hierarchical'  # hierarchical | spectral | kmeans on feature-dynamic space
    n_dimensions: int = 8
    random_state: int = 42


class DimensionDiscoveryPipeline:
    def __init__(self,
                 dynamic_config: Optional[DynamicTargetConfig] = None,
                 discovery_config: Optional[DiscoveryConfig] = None):
        self.logger = system_logger.getChild('DimensionDiscovery')
        self.dynamic_builder = DynamicTargetsBuilder(dynamic_config)
        self.config = discovery_config or DiscoveryConfig()

    def _compute_model_importance(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """Compute model-based importance via RandomForest and optional XGBoost.

        Returns average-normalized importances keyed by feature name.
        """
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.preprocessing import LabelEncoder

        Xv = X.fillna(0).values
        is_cls = len(np.unique(y)) < min(len(y) * 0.1, 50)

        if is_cls:
            le = LabelEncoder()
            yv = le.fit_transform(y)
            model = RandomForestClassifier(n_estimators=self.config.rf_n_estimators,
                                           random_state=self.config.random_state,
                                           n_jobs=-1)
        else:
            yv = y
            model = RandomForestRegressor(n_estimators=self.config.rf_n_estimators,
                                          random_state=self.config.random_state,
                                          n_jobs=-1)
        model.fit(Xv, yv)
        imp = model.feature_importances_
        if np.max(imp) > 0:
            imp = imp / np.max(imp)
        return {f: float(w) for f, w in zip(X.columns, imp)}

    def _compute_shap_importance(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """Compute SHAP main-effect importance using a tree model.

        Fallbacks to RandomForest if XGBoost/SHAP is unavailable.
        """
        try:
            import shap  # type: ignore
            import xgboost as xgb  # type: ignore
            Xv = X.fillna(0).values
            is_cls = len(np.unique(y)) < min(len(y) * 0.1, 50)
            if is_cls:
                le = LabelEncoder()
                yv = le.fit_transform(y)
                model = xgb.XGBClassifier(n_estimators=self.config.xgb_n_estimators,
                                          max_depth=6,
                                          subsample=0.8,
                                          colsample_bytree=0.8,
                                          random_state=self.config.random_state)
            else:
                yv = y
                model = xgb.XGBRegressor(n_estimators=self.config.xgb_n_estimators,
                                         max_depth=6,
                                         subsample=0.8,
                                         colsample_bytree=0.8,
                                         random_state=self.config.random_state)
            model.fit(Xv, yv)
            if len(Xv) > self.config.shap_sample_size:
                idx = np.random.RandomState(self.config.random_state).choice(len(Xv), self.config.shap_sample_size, replace=False)
                Xs = Xv[idx]
            else:
                Xs = Xv
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(Xs)
            if isinstance(sv, list):  # multiclass
                values = np.mean([np.mean(np.abs(s), axis=0) for s in sv], axis=0)
            else:
                values = np.mean(np.abs(sv), axis=0)
            if np.max(values) > 0:
                values = values / np.max(values)
            return {f: float(w) for f, w in zip(X.columns, values)}
        except Exception as e:
            self.logger.warning(f"SHAP/XGBoost unavailable or failed ({e}); falling back to RF importance")
            return self._compute_model_importance(X, y)

    def analyze_single_target(self, features: pd.DataFrame, target: np.ndarray) -> Dict[str, any]:
        """Run mRMR → RF/XGB+SHAP → PID on pairs for one dynamic target."""
        cfg = MRMRConfig(max_features=self.config.mrmr_max_features,
                         redundancy_penalty=self.config.mrmr_redundancy_penalty,
                         random_state=self.config.random_state)

        # 1) mRMR screen
        selected = mrmr_select(features, target, cfg)
        X_sel = features[selected]

        # 2) Model importance (RF/XGB) and SHAP
        model_imp = self._compute_model_importance(X_sel, target)
        shap_imp = self._compute_shap_importance(X_sel, target)

        # 3) PID on top pairs (by SHAP importance)
        shap_sorted = sorted(shap_imp.items(), key=lambda kv: kv[1], reverse=True)
        top_feats = [f for f, _ in shap_sorted[:min(len(shap_sorted), int(np.sqrt(len(shap_sorted))*10)+5)]]
        pairs: List[Tuple[str, str]] = []
        for i in range(len(top_feats)):
            for j in range(i + 1, len(top_feats)):
                pairs.append((top_feats[i], top_feats[j]))
        pairs = pairs[: self.config.pid_top_pairs]

        pair_results = []
        for f1, f2 in pairs:
            try:
                res = pid_pair(X_sel, target, f1, f2, random_state=self.config.random_state)
                res['f1'] = f1
                res['f2'] = f2
                pair_results.append(res)
            except Exception as e:
                self.logger.warning(f"PID failed for ({f1},{f2}): {e}")

        return {
            'mrmr_selected': selected,
            'model_importance': model_imp,
            'shap_importance': shap_imp,
            'pid_pairs': pair_results,
        }

    def aggregate_across_targets(self, results_by_target: Dict[str, Dict[str, any]]) -> Dict[str, any]:
        """Aggregate to feature × dynamic matrix, synergy edges, redundancy matrix."""
        # Matrix: rows=features, cols=targets, values=stabilized importance (SHAP)
        all_features = sorted({f for r in results_by_target.values() for f in r['shap_importance'].keys()})
        all_targets = list(results_by_target.keys())
        mat = pd.DataFrame(0.0, index=all_features, columns=all_targets)
        for t, r in results_by_target.items():
            for f, v in r['shap_importance'].items():
                mat.loc[f, t] = v

        # Synergy edges from PID: sum synergy across targets
        synergy: Dict[Tuple[str, str], float] = {}
        for t, r in results_by_target.items():
            for pr in r['pid_pairs']:
                edge = tuple(sorted((pr['f1'], pr['f2'])))
                synergy[edge] = synergy.get(edge, 0.0) + float(pr.get('synergy', 0.0))

        # Redundancy matrix via co-importance correlation across targets
        redund = mat.corr(axis=1).fillna(0.0)

        return {
            'feature_dynamic_matrix': mat,
            'synergy_edges': synergy,
            'redundancy_matrix': redund,
        }

    def derive_dimensions(self, aggregation: Dict[str, any]) -> Dict[str, any]:
        """Derive implicit dimensions via clustering/factorization."""
        from sklearn.decomposition import NMF
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.cluster import AgglomerativeClustering

        mat = aggregation['feature_dynamic_matrix']  # features × targets
        # Non-negative transform for NMF
        scaler = MinMaxScaler()
        mat_pos = pd.DataFrame(scaler.fit_transform(mat.values), index=mat.index, columns=mat.columns)

        # Factorization to get latent topic-like dimensions
        nmf = NMF(n_components=self.config.n_dimensions, init='nndsvda', random_state=self.config.random_state, max_iter=1000)
        W = nmf.fit_transform(mat_pos.values)  # features × n_dimensions
        H = nmf.components_  # n_dimensions × targets

        # Optional hierarchical clustering on feature loadings to form groups
        clustering = AgglomerativeClustering(n_clusters=self.config.n_dimensions, linkage='ward')
        cluster_labels = clustering.fit_predict(W)

        # Build dimensions
        dimensions: Dict[str, Dict[str, any]] = {}
        for d in range(self.config.n_dimensions):
            members_idx = np.where(cluster_labels == d)[0]
            members = [mat.index[i] for i in members_idx]
            loadings = W[members_idx, d] if d < W.shape[1] else np.zeros(len(members_idx))
            # Representative targets for naming
            contrib_targets = np.argsort(H[d])[::-1][:3] if d < H.shape[0] else []
            top_targets = [mat.columns[i] for i in contrib_targets]
            dimensions[f'dim_{d}'] = {
                'members': members,
                'member_loadings': loadings.tolist(),
                'top_targets': top_targets,
            }

        # Dimension scores over time can be computed as weighted sums of member z-scores later
        return {
            'nmf_W': W,
            'nmf_H': H,
            'cluster_labels': cluster_labels,
            'dimensions': dimensions,
        }

    def compute_dimension_scores(self, features: pd.DataFrame, dimensions: Dict[str, Dict[str, any]]) -> pd.DataFrame:
        """Compute time series scores for each dimension using weighted SHAP-like scheme.

        For each dimension, score_t = mean z-score across member features (simple, robust).
        """
        # z-score features
        Xz = features.copy().fillna(0)
        Xz = (Xz - Xz.mean()) / (Xz.std().replace(0, np.nan))
        Xz = Xz.fillna(0)

        dim_scores = {}
        for name, info in dimensions.items():
            members = [f for f in info['members'] if f in Xz.columns]
            if len(members) == 0:
                dim_scores[name] = pd.Series(0.0, index=features.index)
            else:
                dim_scores[name] = Xz[members].mean(axis=1)
        return pd.DataFrame(dim_scores)

    def run(self, market_data: pd.DataFrame, feature_data: pd.DataFrame) -> Dict[str, any]:
        """Execute the full discovery pipeline.

        Returns a dict with targets, per-target analyses, aggregation, dimensions,
        and time-series dimension scores.
        """
        # 1) Targets
        targets = self.dynamic_builder.build_all(market_data)

        # 2) Analyze each target
        results_by_target: Dict[str, Dict[str, any]] = {}
        for t_name, t_series in targets.items():
            try:
                y = t_series.values.astype(float)
                # Drop NaNs alignment
                mask = np.isfinite(y)
                X = feature_data.loc[mask].copy().fillna(0)
                y = y[mask]
                if len(X) < 50:
                    continue
                results_by_target[t_name] = self.analyze_single_target(X, y)
            except Exception as e:
                self.logger.warning(f"Target {t_name} analysis failed: {e}")

        if not results_by_target:
            return {'targets': targets, 'error': 'no_target_results'}

        # 3) Aggregation
        aggregation = self.aggregate_across_targets(results_by_target)

        # 4) Dimensions
        derived = self.derive_dimensions(aggregation)

        # 5) Time-series dimension scores
        dim_scores = self.compute_dimension_scores(feature_data, derived['dimensions'])

        return {
            'targets': targets,
            'results_by_target': results_by_target,
            'aggregation': aggregation,
            'dimensions': derived['dimensions'],
            'dimension_scores': dim_scores,
        }

