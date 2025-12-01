import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.cluster.hierarchy import fcluster
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import fastcluster
import os
from datetime import datetime

# Limits the analysis to 6 months of data (even 3 months would be enough)
# 1/ EWMA Spearman IC + Spearman IC stability analysis pre-filters (higher weight on more recent windows)
# 2/ Hierarchical clustering (avoids colinearity, keep the best in class, reduces number of features)
# 3/ 2-steps RFE with light then strong LGBM

class FeatureSelector:
    def __init__(self, target_n_features):
        self.target_n_features = target_n_features
        self.ic_stats = None
        self.original_columns = None

    def select_features(self, X, y):
        self.original_columns = X.columns
        X = X.astype(np.float32)
        y = y.astype(np.float32)
    
        # --- Pre-filter: keep only last 6 months of data if longer ---
        bars_per_day = 24 * 4  # 15-min bars
        bars_per_month = 30 * bars_per_day
        bars_6_months = 6 * bars_per_month
    
        if X.shape[0] > bars_6_months:
            X = X.iloc[-bars_6_months:]
            y = y.iloc[-bars_6_months:]
    
        # Stage 1: EWMA-based Pre-Filters
        X_filtered = self.run_pre_filters(X, y)
    
        # Stage 2: Hierarchical Clustering
        X_clustered = self._hierarchical_clustering(X_filtered, y)
    
        # Stage 3: LGBM-Based RFE
        selected_features = self._lgbm_rfe(X_clustered, y)
    
        # Generate report
        self._generate_report(X_filtered)
    
        return selected_features

    
    def run_pre_filters(self, X, y):
        # --- 1. Remove constant/near-constant features ---
        to_drop = X.columns[
            (X.nunique() <= 1) |
            (X.apply(lambda col: col.value_counts(normalize=True).max()) > 0.99)
        ]
        X = X.drop(columns=to_drop)
    
        # --- 2. Remove features with > 5% NaN ---
        nan_cols = (X.isnull().sum() / len(X) > 0.05).index
        X = X.drop(columns=nan_cols)

        if X.shape[1] == 0:
            return X
    
        # --- 3. EWMA-based IC + stability analysis ---
        T_ic_window = 672  # 15-min bars per IC window
        K = 12             # number of windows
        hl_days = 60       # half-life in days
        bars_per_day = 24 * 4  # 15-min bars
        hl_samples = hl_days * bars_per_day
        alpha = 1 - np.exp(-np.log(2) / hl_samples)
        eps = 1e-9
    
        n_samples = X.shape[0]
        ic_series = []
    
        for i in range(K):
            start = n_samples - (K - i) * T_ic_window
            end = start + T_ic_window
            if start < 0:
                continue
            X_window = X.iloc[start:end]
            y_window = y.iloc[start:end]
            ic = self._calculate_spearman_correlation(X_window, y_window)
            ic_series.append(ic)
    
        ic_series = np.array(ic_series)
        if ic_series.shape[0] == 0:
            return X  # not enough data
    
        # EWMA IC calculations
        ic_ewma = pd.DataFrame(ic_series).ewm(alpha=alpha, adjust=False).mean().iloc[-1].values
        ic_ewm_var = pd.DataFrame(ic_series).ewm(alpha=alpha, adjust=False).var().iloc[-1].values
        ic_ewm_std = np.sqrt(ic_ewm_var)
    
        # Stability metrics
        ewma_sharpe = ic_ewma / (ic_ewm_std + eps)
        cv = ic_ewm_std / (np.abs(ic_ewma) + eps)
        positivity = (ic_series > 0).mean(axis=0)
    
        # Adaptive CUSUM
        cusum = np.cumsum(ic_series - ic_ewma, axis=0)
        cusum_recent = np.max(np.abs(cusum[-5:]), axis=0)
        cusum_weight = min(0.5, cusum_recent.std() / (cusum_recent.mean() + eps))
    
        # Normalize metrics
        ewma_sharpe_norm = pd.Series(ewma_sharpe, index=X.columns).rank(pct=True)
        cv_norm = pd.Series(cv, index=X.columns).rank(pct=True, ascending=False)
        positivity_norm = pd.Series(positivity, index=X.columns).rank(pct=True)
        cusum_norm = pd.Series(cusum_recent, index=X.columns).rank(pct=True, ascending=False)
    
        # Stability score
        stability_score = ewma_sharpe_norm - 0.8 * cv_norm + 0.5 * positivity_norm - cusum_weight * cusum_norm
    
        # Normalize EWMA IC
        ic_norm = pd.Series(ic_ewma, index=X.columns).rank(pct=True)

        # Calculate IC volatility 
        ic_volatility = np.std(ic_ewma) / (np.mean(np.abs(ic_ewma)) + eps)
        max_ic_weight = 0.7
        min_ic_weight = 0.3
        ic_volatility = np.clip(ic_volatility, 0, 1)  # 0 = stable, 1 = very volatile
        
        # Map to IC weight: higher volatility → lower IC weight
        w_ic = max_ic_weight - (max_ic_weight - min_ic_weight) * ic_volatility
        w_stability = 1 - w_ic

        # Combined score: stability + IC
        combined_score = w_stability * stability_score + w_ic * ic_norm
    
        self.ic_stats = {
            'stability_score': stability_score.values,
            'ic_ewma': ic_ewma,
            'ewma_sharpe': ewma_sharpe,
            'cv': cv,
            'positivity': positivity,
            'cusum_recent': cusum_recent,
            'combined_score': combined_score.values
        }
    
        # --- 4. Percentile + hard cap ---
        percentile_cut = 0.3  # keep top 70%
        candidate_features = combined_score[combined_score.rank(pct=True) > percentile_cut]
    
        max_features = 5 * self.target_n_features
        n_keep = min(len(candidate_features), max_features)
    
        # Select top features by combined score
        stable_features = candidate_features.nlargest(n_keep).index
    
        return X[stable_features]



    def _hierarchical_clustering(self, X, y):
        corr = np.corrcoef(rankdata(X.values, axis=0).T)
        dist = 1 - np.abs(corr)

        condensed_dist = dist[np.triu_indices(dist.shape[0], k=1)]
        Z = fastcluster.linkage(condensed_dist, method='average')

        if self.ic_stats is not None:
            stability_scores = pd.Series(self.ic_stats['stability_score'], index=self.original_columns)
            stability_scores = stability_scores[X.columns]
        else:
            ic_list = []
            for i in range(5):
                idx = np.random.choice(X.shape[0], size=int(0.15 * X.shape[0]), replace=False)
                X_sub = X.iloc[idx]
                y_sub = y.iloc[idx]
                ic_matrix = self._calculate_spearman_correlation(X_sub, y_sub)
                ic_list.append(ic_matrix)
            ic_array = np.vstack(ic_list)
            ic_mean = ic_array.mean(axis=0)
            ic_std = ic_array.std(axis=0)
            stability_scores = pd.Series(ic_mean - 0.5 * ic_std, index=X.columns)

        t = 0.4
        while True:
            clusters = fcluster(Z, t, criterion='distance')
            n_clusters = len(np.unique(clusters))
            if n_clusters / self.target_n_features > 3.5:
                t += 0.03
            elif n_clusters / self.target_n_features < 2.5:
                t -= 0.03
            else:
                break

        cluster_mapping = pd.DataFrame({'feature': X.columns, 'cluster': clusters})
        representative_features = []
        for cluster_id in cluster_mapping['cluster'].unique():
            features_in_cluster = cluster_mapping[cluster_mapping['cluster'] == cluster_id]['feature']
            best_feature = stability_scores.loc[features_in_cluster].idxmax()
            representative_features.append(best_feature)

        return X[representative_features]

    def _calculate_spearman_correlation(self, X, y):
        X_ranked = rankdata(X.values, axis=0)
        y_ranked = rankdata(y.values, axis=0)
        ranked_data = np.hstack([X_ranked, y_ranked.reshape(-1,1)])
        corr_matrix = np.corrcoef(ranked_data, rowvar=False)
        return corr_matrix[:-1, -1]

    def _generate_report(self, X):
        if self.ic_stats is None:
            return

        report_df = pd.DataFrame({
            'stability_score': self.ic_stats['stability_score'],
            'ewma_sharpe': self.ic_stats['ewma_sharpe'],
            'cv': self.ic_stats['cv'],
            'positivity': self.ic_stats['positivity'],
            'cusum_recent': self.ic_stats['cusum_recent']
        }, index=X.columns)

        os.makedirs('outcomes', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join('outcomes', f'feature_selection_report_{timestamp}.csv')
        report_df.to_csv(report_path)

    def _lgbm_rfe(self, X, y):
        features = list(X.columns)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fast RFE with shadow features
        X_train_shadow = X_train[features].copy()
        X_val_shadow = X_val[features].copy()
        n_shadow = min(20, len(features))
        for i in range(n_shadow):
            col_name = f'shadow_{i}'
            col_idx = i % len(features)
            X_train_shadow[col_name] = np.random.permutation(X_train_shadow.iloc[:, col_idx])
            X_val_shadow[col_name] = np.random.permutation(X_val_shadow.iloc[:, col_idx])

        model = lgb.LGBMRegressor(
            objective='regression',
            metric='rmse',
            boosting_type='goss',
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            min_child_samples=50,
            min_child_weight=1e-3,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=2,
            verbosity=-1,
            top_rate=0.1,
            other_rate=0.2
        )
        model.fit(X_train_shadow, y_train,
                  eval_set=[(X_val_shadow, y_val)],
                  callbacks=[lgb.early_stopping(5, verbose=False)])

        importances = pd.Series(model.feature_importances_, index=X_train_shadow.columns)
        shadow_importance = importances[importances.index.str.startswith('shadow')].mean()
        features = importances[importances > shadow_importance].index
        features = [f for f in features if not f.startswith('shadow')]

        # Thorough RFE
        model = lgb.LGBMRegressor(
            objective='regression',
            metric='rmse',
            boosting_type='gbdt',
            n_estimators=750,
            learning_rate=0.04,
            max_depth=6,
            min_child_samples=50,
            min_child_weight=1e-3,
            subsample=0.7,
            colsample_bytree=0.5,
            bagging_freq=1,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=2,
            verbosity=-1
        )
        model.fit(X_train[features], y_train,
                  eval_set=[(X_val[features], y_val)],
                  callbacks=[lgb.early_stopping(10, verbose=False)])

        importances = pd.Series(model.feature_importances_, index=features)
        p0 = len(features)
        pt = self.target_n_features
        alpha = 0.4
        min_drop = 1

        while len(features) > pt:
            remaining = len(features) - pt
            fraction = alpha * (remaining / (p0 - pt))
            n_to_drop = max(min_drop, int(np.ceil(fraction * remaining)))
            least_important = importances.loc[features].nsmallest(n_to_drop).index
            features = [f for f in features if f not in least_important]

        return features
