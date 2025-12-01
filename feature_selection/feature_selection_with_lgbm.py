import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.cluster.hierarchy import fcluster
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
import lightgbm as lgb
import fastcluster
from scipy.spatial.distance import squareform
import os
from datetime import datetime

class FeatureSelector:
    def __init__(self, target_n_features):
        self.target_n_features = target_n_features
        self.ic_stats = None
        self.original_columns = None

    def select_features(self, X, y):
        self.original_columns = X.columns
        """
        Selects the best features from the given data.
        """
        # Convert to float32
        X = X.astype(np.float32)
        y = y.astype(np.float32)

        # Stage 1: Cheap Pre-Filters
        X_filtered = self.run_pre_filters(X, y)

        # Stage 2: Hierarchical Clustering
        X_clustered = self._hierarchical_clustering(X_filtered, y)

        # Stage 3: LGBM-Based RFE
        selected_features = self._lgbm_rfe(X_clustered, y)

        # Generate report
        self._generate_report(X)

        return selected_features

    def run_pre_filters(self, X, y):
        """
        Performs the pre-filtering steps.
        """
        # 1.1 Remove constant or near-constant features, or features with over 5% NaN
        # Remove constant/near-constant features
        to_drop = []
        for col in X.columns:
            if X[col].nunique() <= 1:
                to_drop.append(col)
            elif X[col].value_counts(normalize=True).max() > 0.99:
                to_drop.append(col)
        
        X = X.drop(columns=to_drop)

        # Remove features with > 5% NaN
        nan_percents = X.isnull().sum() / len(X)
        nan_cols = nan_percents[nan_percents > 0.05].index
        X = X.drop(columns=nan_cols)

        # 1.2 Remove features with extremely low correlation to target or high instability
        n_samples = X.shape[0]
        sub_size = int(0.15 * n_samples)
        ic_list = []
        for i in range(5):
            idx = np.random.choice(n_samples, size=sub_size, replace=False)
            X_sub = X.iloc[idx]
            y_sub = y.iloc[idx]
            ic_matrix = self._calculate_spearman_correlation(X_sub, y_sub)
            ic_list.append(ic_matrix)

        ic_array = np.vstack(ic_list)
        ic_mean = ic_array.mean(axis=0)
        ic_std = ic_array.std(axis=0)
        sharpe_ratio = ic_mean - 0.5 * ic_std
        
        self.ic_stats = {
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'sharpe_ratio': sharpe_ratio
        }

        ic_mean_rank = pd.Series(ic_mean).rank(pct=True)
        ic_std_rank = pd.Series(ic_std).rank(pct=True, ascending=False)
        
        stable_features = X.columns[
            (ic_mean_rank > 0.2) & (ic_std_rank > 0.3) & (ic_std < 0.03)
        ]

        return X[stable_features]

    def _hierarchical_clustering(self, X, y):
        """
        Performs hierarchical clustering to select representative features.
        """
        # 2.1 Compute distance matrix
        ranked = self._fast_rank(X.values)
        corr = np.corrcoef(ranked.T)
        dist = 1 - np.abs(corr)

        # 2.2 Perform hierarchical clustering
        condensed_dist = dist[np.triu_indices(dist.shape[0], k=1)]
        Z = fastcluster.linkage(condensed_dist, method='ward')

        # 2.3 Use pre-calculated Sharpe Ratios
        if self.ic_stats is None:
            # Fallback if pre-filtering was not run
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
            sharpe_ratios = pd.Series(ic_mean - 0.5 * ic_std, index=X.columns)
        else:
            sharpe_ratios = pd.Series(self.ic_stats['sharpe_ratio'], index=self.original_columns)
            sharpe_ratios = sharpe_ratios[X.columns]

        # 2.4 Pick one representative per cluster
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
            best_feature = sharpe_ratios.loc[features_in_cluster].idxmax()
            representative_features.append(best_feature)
        
        return X[representative_features]

    def _calculate_spearman_correlation(self, X, y):
        """
        Calculates the Spearman correlation between features and the target.
        """
        X_ranked = self._fast_rank(X.values)
        y_ranked = self._fast_rank(y.values.reshape(-1, 1))
        
        # Combine and calculate correlation
        ranked_data = np.hstack([X_ranked, y_ranked])
        corr_matrix = np.corrcoef(ranked_data, rowvar=False)
        
        return corr_matrix[:-1, -1]

    def _generate_report(self, X):
        """
        Generates a CSV report with feature metrics.
        """
        if self.ic_stats is None:
            return

        report_df = pd.DataFrame({
            'ic_mean': self.ic_stats['ic_mean'],
            'ic_std': self.ic_stats['ic_std'],
            'sharpe_ratio': self.ic_stats['sharpe_ratio']
        }, index=X.columns)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join('outcomes', f'feature_selection_report_{timestamp}.csv')
        report_df.to_csv(report_path)

    def _lgbm_rfe(self, X, y, max_iterations=10):
        """
        Performs LGBM-based RFE.
        """
        features = list(X.columns)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3.1 Fast RFE
        # Train once with shadow features
        X_train_shadow = X_train[features].copy()
        n_shadow = min(20, len(features))
        for i in range(n_shadow):
            X_train_shadow[f'shadow_{i}'] = np.random.permutation(X_train_shadow.iloc[:, i % len(features)])

        model = lgb.LGBMRegressor(
            objective='regression',
            metric='rmse',
            boosting_type='goss',
            n_estimators=200,
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
                  eval_set=[(X_val[features], y_val)],
                  callbacks=[lgb.early_stopping(10, verbose=False)])
        
        importances = pd.Series(model.feature_importances_, index=X_train_shadow.columns)
        shadow_importance = importances[importances.index.str.startswith('shadow')].mean()
        
        features = importances[importances > shadow_importance].index
        features = [f for f in features if not f.startswith('shadow')]

        # 3.2 Thorough RFE
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
        
        while len(features) > self.target_n_features:
            n_to_drop = int(len(features) * 0.1)
            if n_to_drop == 0:
                n_to_drop = 1
            
            least_important = importances.loc[features].nsmallest(n_to_drop).index
            features = [f for f in features if f not in least_important]
            
        return features
        
    def _fast_rank(self, arr):
        n = arr.shape[0]
        return arr.argsort(axis=0).argsort(axis=0) / n

