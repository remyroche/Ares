#!/usr/bin/env python3
"""
HMM Regime Parameter Optimization using Optuna

This script optimizes Step 3 HMM regime discovery parameters to capture distinct market conditions
rather than predict regime transitions. It uses Optuna for hyperparameter optimization and
comprehensive evaluation metrics focused on market condition differentiation.

Usage:
    python optimize_hmm_regime_parameters.py --data_path path/to/feature_data.parquet
    python optimize_hmm_regime_parameters.py --config_path path/to/config.json --n_trials 100
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class HMMRegimeOptimizer:
    """Optimize HMM regime discovery parameters for distinct market condition capture."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.best_params = None
        self.best_score = -np.inf
        self.optimization_history = []
        self.study = None
        
    def create_objective_function(self, data: pd.DataFrame, 
                                feature_columns: List[str],
                                market_condition_columns: List[str]) -> callable:
        """Create the objective function for Optuna optimization."""
        
        def objective(trial: optuna.Trial) -> float:
            """Objective function to maximize market condition differentiation."""
            
            # Suggest HMM parameters
            params = self._suggest_hmm_parameters(trial)
            
            # Suggest clustering parameters
            clustering_params = self._suggest_clustering_parameters(trial)
            
            # Suggest feature engineering parameters
            feature_params = self._suggest_feature_parameters(trial)
            
            try:
                # Generate clusters with suggested parameters
                cluster_data = self._generate_clusters_with_params(
                    data, feature_columns, market_condition_columns, 
                    params, clustering_params, feature_params
                )
                
                # Evaluate cluster quality for market condition capture
                score = self._evaluate_market_condition_capture(
                    cluster_data, market_condition_columns
                )
                
                # Store trial information
                trial_info = {
                    'trial_number': trial.number,
                    'params': {**params, **clustering_params, **feature_params},
                    'score': score,
                    'timestamp': time.time()
                }
                self.optimization_history.append(trial_info)
                
                return score
                
            except Exception as e:
                print(f"⚠️ Trial {trial.number} failed: {e}")
                return -np.inf
        
        return objective
    
    def _suggest_hmm_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest HMM-specific parameters."""
        return {
            'n_components': trial.suggest_int('n_components', 2, 8),
            'covariance_type': trial.suggest_categorical('covariance_type', 
                                                       ['full', 'tied', 'diag', 'spherical']),
            'random_state': 42,
            'n_iter': trial.suggest_int('n_iter', 50, 200),
            'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True),
            'reg_covar': trial.suggest_float('reg_covar', 1e-6, 1e-3, log=True)
        }
    
    def _suggest_clustering_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest clustering algorithm parameters."""
        clustering_method = trial.suggest_categorical('clustering_method', 
                                                     ['kmeans', 'gaussian_mixture', 'hierarchical'])
        
        params = {'clustering_method': clustering_method}
        
        if clustering_method == 'kmeans':
            params.update({
                'n_clusters': trial.suggest_int('n_clusters', 3, 12),
                'init': trial.suggest_categorical('kmeans_init', ['k-means++', 'random']),
                'n_init': trial.suggest_int('kmeans_n_init', 5, 20),
                'max_iter': trial.suggest_int('kmeans_max_iter', 200, 500)
            })
        elif clustering_method == 'gaussian_mixture':
            params.update({
                'n_components': trial.suggest_int('gmm_n_components', 3, 12),
                'covariance_type': trial.suggest_categorical('gmm_covariance_type', 
                                                           ['full', 'tied', 'diag', 'spherical']),
                'init_params': trial.suggest_categorical('gmm_init_params', ['kmeans', 'random']),
                'max_iter': trial.suggest_int('gmm_max_iter', 100, 300)
            })
        
        return params
    
    def _suggest_feature_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest feature engineering parameters."""
        return {
            'use_pca': trial.suggest_categorical('use_pca', [True, False]),
            'n_pca_components': trial.suggest_int('n_pca_components', 5, 20) if trial.suggest_categorical('use_pca', [True, False]) else None,
            'feature_selection_method': trial.suggest_categorical('feature_selection', 
                                                                ['all', 'variance', 'correlation']),
            'correlation_threshold': trial.suggest_float('correlation_threshold', 0.7, 0.95),
            'variance_threshold': trial.suggest_float('variance_threshold', 0.01, 0.1),
            'scaling_method': trial.suggest_categorical('scaling_method', ['standard', 'robust', 'minmax'])
        }
    
    def _generate_clusters_with_params(self, data: pd.DataFrame, 
                                     feature_columns: List[str],
                                     market_condition_columns: List[str],
                                     hmm_params: Dict[str, Any],
                                     clustering_params: Dict[str, Any],
                                     feature_params: Dict[str, Any]) -> pd.DataFrame:
        """Generate clusters using the suggested parameters."""
        
        # Prepare features
        features = data[feature_columns].copy()
        
        # Handle missing values
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Feature selection
        features = self._apply_feature_selection(features, feature_params)
        
        # Feature scaling
        features_scaled = self._apply_scaling(features, feature_params)
        
        # Dimensionality reduction if requested
        if feature_params.get('use_pca', False):
            features_scaled = self._apply_pca(features_scaled, feature_params)
        
        # Generate clusters
        cluster_labels = self._apply_clustering(features_scaled, clustering_params)
        
        # Create result dataframe
        result_data = data.copy()
        result_data['composite_cluster_id'] = cluster_labels
        
        # Add market condition columns
        for col in market_condition_columns:
            if col in data.columns:
                result_data[f'market_{col}'] = data[col]
        
        return result_data
    
    def _apply_feature_selection(self, features: pd.DataFrame, 
                               feature_params: Dict[str, Any]) -> pd.DataFrame:
        """Apply feature selection based on parameters."""
        method = feature_params.get('feature_selection_method', 'all')
        
        if method == 'all':
            return features
        
        elif method == 'variance':
            from sklearn.feature_selection import VarianceThreshold
            threshold = feature_params.get('variance_threshold', 0.01)
            selector = VarianceThreshold(threshold=threshold)
            selected_features = selector.fit_transform(features)
            selected_columns = features.columns[selector.get_support()]
            return pd.DataFrame(selected_features, columns=selected_columns, index=features.index)
        
        elif method == 'correlation':
            threshold = feature_params.get('correlation_threshold', 0.8)
            corr_matrix = features.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
            return features.drop(columns=to_drop)
        
        return features
    
    def _apply_scaling(self, features: pd.DataFrame, 
                      feature_params: Dict[str, Any]) -> np.ndarray:
        """Apply scaling based on parameters."""
        method = feature_params.get('scaling_method', 'standard')
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        else:
            return features.values
        
        return scaler.fit_transform(features)
    
    def _apply_pca(self, features_scaled: np.ndarray, 
                  feature_params: Dict[str, Any]) -> np.ndarray:
        """Apply PCA if requested."""
        n_components = feature_params.get('n_pca_components', 10)
        n_components = min(n_components, features_scaled.shape[1])
        
        pca = PCA(n_components=n_components)
        return pca.fit_transform(features_scaled)
    
    def _apply_clustering(self, features_scaled: np.ndarray, 
                         clustering_params: Dict[str, Any]) -> np.ndarray:
        """Apply clustering algorithm based on parameters."""
        method = clustering_params.get('clustering_method', 'kmeans')
        
        if method == 'kmeans':
            kmeans = KMeans(
                n_clusters=clustering_params.get('n_clusters', 5),
                init=clustering_params.get('init', 'k-means++'),
                n_init=clustering_params.get('n_init', 10),
                max_iter=clustering_params.get('max_iter', 300),
                random_state=42
            )
            return kmeans.fit_predict(features_scaled)
        
        elif method == 'gaussian_mixture':
            gmm = GaussianMixture(
                n_components=clustering_params.get('n_components', 5),
                covariance_type=clustering_params.get('covariance_type', 'full'),
                init_params=clustering_params.get('init_params', 'kmeans'),
                max_iter=clustering_params.get('max_iter', 200),
                random_state=42
            )
            return gmm.fit_predict(features_scaled)
        
        else:
            # Default to kmeans
            kmeans = KMeans(n_clusters=5, random_state=42)
            return kmeans.fit_predict(features_scaled)
    
    def _evaluate_market_condition_capture(self, cluster_data: pd.DataFrame, 
                                         market_condition_columns: List[str]) -> float:
        """Evaluate how well clusters capture distinct market conditions."""
        
        if 'composite_cluster_id' not in cluster_data.columns:
            return -np.inf
        
        scores = []
        
        # 1. Market Condition Differentiation Score
        differentiation_score = self._calculate_market_differentiation_score(
            cluster_data, market_condition_columns
        )
        scores.append(differentiation_score)
        
        # 2. Cluster Quality Score
        quality_score = self._calculate_cluster_quality_score(cluster_data)
        scores.append(quality_score)
        
        # 3. Market Condition Consistency Score
        consistency_score = self._calculate_market_consistency_score(
            cluster_data, market_condition_columns
        )
        scores.append(consistency_score)
        
        # 4. Cluster Balance Score
        balance_score = self._calculate_cluster_balance_score(cluster_data)
        scores.append(balance_score)
        
        # 5. Market Condition Separation Score
        separation_score = self._calculate_market_separation_score(
            cluster_data, market_condition_columns
        )
        scores.append(separation_score)
        
        # Weighted combination (focus on market condition capture)
        weights = [0.4, 0.2, 0.2, 0.1, 0.1]  # Emphasize market differentiation
        final_score = np.average(scores, weights=weights)
        
        return final_score
    
    def _calculate_market_differentiation_score(self, cluster_data: pd.DataFrame, 
                                              market_condition_columns: List[str]) -> float:
        """Calculate how well clusters differentiate market conditions."""
        
        if not market_condition_columns:
            return 0.0
        
        differentiation_scores = []
        
        for col in market_condition_columns:
            if col not in cluster_data.columns:
                continue
            
            # Calculate average market condition value for each cluster
            cluster_means = cluster_data.groupby('composite_cluster_id')[col].mean()
            
            if len(cluster_means) < 2:
                continue
            
            # Calculate how different clusters are from each other
            differences = []
            for i, mean1 in cluster_means.items():
                for j, mean2 in cluster_means.items():
                    if i != j:
                        differences.append(abs(mean1 - mean2))
            
            if differences:
                # Normalize by the overall range of the market condition
                overall_range = cluster_data[col].max() - cluster_data[col].min()
                if overall_range > 0:
                    avg_difference = np.mean(differences) / overall_range
                    differentiation_scores.append(avg_difference)
        
        return np.mean(differentiation_scores) if differentiation_scores else 0.0
    
    def _calculate_cluster_quality_score(self, cluster_data: pd.DataFrame) -> float:
        """Calculate traditional cluster quality metrics."""
        
        # Prepare features for quality metrics
        feature_columns = [col for col in cluster_data.columns 
                          if col not in ['composite_cluster_id', 'timestamp', 'close', 'high', 'low', 'open', 'volume']]
        
        if len(feature_columns) < 2:
            return 0.0
        
        features = cluster_data[feature_columns[:min(10, len(feature_columns))]].fillna(0)
        cluster_labels = cluster_data['composite_cluster_id'].values
        
        try:
            # Silhouette score
            silhouette = silhouette_score(features, cluster_labels)
            
            # Calinski-Harabasz score
            calinski = calinski_harabasz_score(features, cluster_labels)
            
            # Davies-Bouldin score (lower is better, so invert)
            davies = davies_bouldin_score(features, cluster_labels)
            davies_normalized = 1.0 / (1.0 + davies)  # Convert to 0-1 scale
            
            # Combine scores
            quality_score = (silhouette + davies_normalized) / 2.0
            
            return quality_score
            
        except Exception:
            return 0.0
    
    def _calculate_market_consistency_score(self, cluster_data: pd.DataFrame, 
                                          market_condition_columns: List[str]) -> float:
        """Calculate how consistent market conditions are within clusters."""
        
        if not market_condition_columns:
            return 0.0
        
        consistency_scores = []
        
        for col in market_condition_columns:
            if col not in cluster_data.columns:
                continue
            
            # Calculate coefficient of variation within each cluster
            cluster_cvs = []
            for cluster_id in cluster_data['composite_cluster_id'].unique():
                cluster_data_subset = cluster_data[cluster_data['composite_cluster_id'] == cluster_id]
                if len(cluster_data_subset) > 1:
                    mean_val = cluster_data_subset[col].mean()
                    std_val = cluster_data_subset[col].std()
                    if mean_val != 0:
                        cv = std_val / abs(mean_val)
                        cluster_cvs.append(cv)
            
            if cluster_cvs:
                # Lower CV means more consistency, so invert
                avg_cv = np.mean(cluster_cvs)
                consistency = 1.0 / (1.0 + avg_cv)
                consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def _calculate_cluster_balance_score(self, cluster_data: pd.DataFrame) -> float:
        """Calculate how balanced the cluster sizes are."""
        
        cluster_sizes = cluster_data['composite_cluster_id'].value_counts()
        
        if len(cluster_sizes) < 2:
            return 0.0
        
        # Calculate coefficient of variation of cluster sizes
        mean_size = cluster_sizes.mean()
        std_size = cluster_sizes.std()
        
        if mean_size == 0:
            return 0.0
        
        cv = std_size / mean_size
        
        # Lower CV means more balanced, so invert
        balance_score = 1.0 / (1.0 + cv)
        
        return balance_score
    
    def _calculate_market_separation_score(self, cluster_data: pd.DataFrame, 
                                         market_condition_columns: List[str]) -> float:
        """Calculate how well clusters separate different market conditions."""
        
        if not market_condition_columns:
            return 0.0
        
        separation_scores = []
        
        for col in market_condition_columns:
            if col not in cluster_data.columns:
                continue
            
            # Calculate between-cluster variance vs within-cluster variance
            overall_mean = cluster_data[col].mean()
            total_ss = ((cluster_data[col] - overall_mean) ** 2).sum()
            
            between_ss = 0
            within_ss = 0
            
            for cluster_id in cluster_data['composite_cluster_id'].unique():
                cluster_data_subset = cluster_data[cluster_data['composite_cluster_id'] == cluster_id]
                cluster_mean = cluster_data_subset[col].mean()
                cluster_size = len(cluster_data_subset)
                
                between_ss += cluster_size * (cluster_mean - overall_mean) ** 2
                within_ss += ((cluster_data_subset[col] - cluster_mean) ** 2).sum()
            
            if within_ss > 0:
                f_ratio = between_ss / within_ss
                # Normalize to 0-1 scale
                separation = f_ratio / (1 + f_ratio)
                separation_scores.append(separation)
        
        return np.mean(separation_scores) if separation_scores else 0.0
    
    def optimize(self, data: pd.DataFrame, 
                feature_columns: List[str],
                market_condition_columns: List[str],
                n_trials: int = 100,
                timeout: Optional[int] = None,
                study_name: str = "hmm_regime_optimization") -> Dict[str, Any]:
        """Run the optimization process."""
        
        print(f"🚀 Starting HMM regime parameter optimization...")
        print(f"📊 Data shape: {data.shape}")
        print(f"🔧 Features: {len(feature_columns)}")
        print(f"📈 Market conditions: {len(market_condition_columns)}")
        print(f"🎯 Trials: {n_trials}")
        
        # Create study
        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            study_name=study_name
        )
        
        # Create objective function
        objective = self.create_objective_function(data, feature_columns, market_condition_columns)
        
        # Run optimization
        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        # Store best results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        print(f"\n✅ Optimization completed!")
        print(f"🏆 Best score: {self.best_score:.4f}")
        print(f"🔧 Best parameters: {self.best_params}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'study': self.study,
            'optimization_history': self.optimization_history
        }
    
    def generate_optimization_report(self, output_path: Optional[str] = None) -> str:
        """Generate a comprehensive optimization report."""
        
        if not self.study:
            return "No optimization study available."
        
        report = []
        report.append("# HMM Regime Parameter Optimization Report")
        report.append("")
        
        # Summary
        report.append(f"## Optimization Summary")
        report.append(f"- **Best Score**: {self.best_score:.4f}")
        report.append(f"- **Total Trials**: {len(self.study.trials)}")
        report.append(f"- **Completed Trials**: {len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        report.append("")
        
        # Best parameters
        report.append(f"## Best Parameters")
        for param, value in self.best_params.items():
            report.append(f"- **{param}**: {value}")
        report.append("")
        
        # Parameter importance
        try:
            importance = optuna.importance.get_param_importances(self.study)
            report.append(f"## Parameter Importance")
            for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                report.append(f"- **{param}**: {imp:.4f}")
            report.append("")
        except Exception as e:
            report.append(f"## Parameter Importance")
            report.append(f"Could not calculate parameter importance: {e}")
            report.append("")
        
        # Optimization history
        report.append(f"## Optimization History")
        report.append(f"| Trial | Score | Key Parameters |")
        report.append(f"|-------|-------|----------------|")
        
        for i, trial_info in enumerate(self.optimization_history[-10:]):  # Last 10 trials
            params = trial_info['params']
            key_params = f"n_components={params.get('n_components', 'N/A')}, " \
                        f"clustering_method={params.get('clustering_method', 'N/A')}"
            report.append(f"| {trial_info['trial_number']} | {trial_info['score']:.4f} | {key_params} |")
        report.append("")
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"📄 Report saved to: {output_path}")
        
        return report_text
    
    def create_optimization_visualizations(self, output_dir: Optional[str] = None) -> None:
        """Create visualizations for the optimization process."""
        
        if not self.study:
            print("No optimization study available for visualization.")
            return
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('HMM Regime Parameter Optimization Results', fontsize=16, fontweight='bold')
        
        # 1. Optimization history
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        scores = [t.value for t in completed_trials]
        trial_numbers = [t.number for t in completed_trials]
        
        axes[0, 0].plot(trial_numbers, scores, 'b-', alpha=0.7)
        axes[0, 0].scatter(trial_numbers, scores, c=scores, cmap='viridis', s=30, alpha=0.8)
        axes[0, 0].set_title('Optimization History')
        axes[0, 0].set_xlabel('Trial Number')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Parameter importance
        try:
            importance = optuna.importance.get_param_importances(self.study)
            params = list(importance.keys())
            values = list(importance.values())
            
            bars = axes[0, 1].barh(params, values)
            axes[0, 1].set_title('Parameter Importance')
            axes[0, 1].set_xlabel('Importance Score')
            
            # Color bars based on importance
            for bar, value in zip(bars, values):
                if value > 0.1:
                    bar.set_color('green')
                elif value > 0.05:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
        except Exception:
            axes[0, 1].text(0.5, 0.5, 'Parameter importance\nnot available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Parameter Importance')
        
        # 3. Score distribution
        axes[1, 0].hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].axvline(self.best_score, color='red', linestyle='--', 
                          label=f'Best Score: {self.best_score:.4f}')
        axes[1, 0].set_title('Score Distribution')
        axes[1, 0].set_xlabel('Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Key parameter relationships
        try:
            # Extract key parameters
            n_components_values = []
            clustering_methods = []
            scores_for_plot = []
            
            for trial in completed_trials:
                if trial.value is not None:
                    n_components_values.append(trial.params.get('n_components', 0))
                    clustering_methods.append(trial.params.get('clustering_method', 'unknown'))
                    scores_for_plot.append(trial.value)
            
            # Create scatter plot
            scatter = axes[1, 1].scatter(n_components_values, scores_for_plot, 
                                       c=range(len(scores_for_plot)), cmap='viridis', 
                                       s=50, alpha=0.7)
            axes[1, 1].set_title('Score vs Number of Components')
            axes[1, 1].set_xlabel('Number of Components')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add colorbar
            plt.colorbar(scatter, ax=axes[1, 1], label='Trial Order')
            
        except Exception:
            axes[1, 1].text(0.5, 0.5, 'Parameter relationship\nplot not available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Parameter Relationships')
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(output_path / 'optimization_results.png', dpi=300, bbox_inches='tight')
            print(f"📊 Visualizations saved to: {output_path / 'optimization_results.png'}")
        else:
            plt.show()
        
        plt.close()
    
    def save_optimization_results(self, output_path: str) -> None:
        """Save optimization results to file."""
        
        results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_history': self.optimization_history,
            'study_name': self.study.study_name if self.study else None
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Optimization results saved to: {output_path}")


def identify_market_condition_columns(data: pd.DataFrame) -> List[str]:
    """Identify columns that represent market conditions."""
    
    market_condition_keywords = [
        'volatility', 'momentum', 'volume', 'returns', 'price_change',
        'trend', 'regime', 'market', 'condition', 'state',
        'rsi', 'macd', 'bollinger', 'atr', 'adx'
    ]
    
    market_columns = []
    
    for col in data.columns:
        col_lower = col.lower()
        
        # Check if column name contains market condition keywords
        if any(keyword in col_lower for keyword in market_condition_keywords):
            market_columns.append(col)
        
        # Check if column represents price-based metrics
        elif any(metric in col_lower for metric in ['close', 'high', 'low', 'open']):
            continue  # Skip raw price data
        
        # Check if column represents technical indicators
        elif any(indicator in col_lower for indicator in ['sma', 'ema', 'bb', 'stoch', 'cci']):
            market_columns.append(col)
    
    return market_columns


def main():
    """Main function to run HMM parameter optimization."""
    parser = argparse.ArgumentParser(description="Optimize HMM regime discovery parameters")
    parser.add_argument("--data_path", type=str, required=True, help="Path to feature data parquet file")
    parser.add_argument("--config_path", type=str, help="Path to configuration JSON file")
    parser.add_argument("--output_dir", type=str, default="optimization_results", help="Output directory")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of optimization trials")
    parser.add_argument("--timeout", type=int, help="Optimization timeout in seconds")
    parser.add_argument("--study_name", type=str, default="hmm_regime_optimization", help="Study name")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load configuration
    config = {}
    if args.config_path:
        with open(args.config_path, 'r') as f:
            config = json.load(f)
    
    # Load data
    print(f"📂 Loading data from: {args.data_path}")
    data = pd.read_parquet(args.data_path)
    print(f"📊 Loaded {len(data)} samples with {len(data.columns)} columns")
    
    # Identify feature and market condition columns
    feature_columns = [col for col in data.columns 
                      if col not in ['timestamp', 'composite_cluster_id']]
    market_condition_columns = identify_market_condition_columns(data)
    
    print(f"🔧 Features: {len(feature_columns)}")
    print(f"📈 Market conditions: {len(market_condition_columns)}")
    print(f"📈 Market condition columns: {market_condition_columns[:10]}...")  # Show first 10
    
    # Initialize optimizer
    optimizer = HMMRegimeOptimizer(config)
    
    # Run optimization
    results = optimizer.optimize(
        data=data,
        feature_columns=feature_columns,
        market_condition_columns=market_condition_columns,
        n_trials=args.n_trials,
        timeout=args.timeout,
        study_name=args.study_name
    )
    
    # Generate report
    print("\n📄 Generating optimization report...")
    report = optimizer.generate_optimization_report(
        output_path=output_path / "optimization_report.md"
    )
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    optimizer.create_optimization_visualizations(output_path)
    
    # Save results
    optimizer.save_optimization_results(output_path / "optimization_results.json")
    
    # Print summary
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETED")
    print("="*60)
    print(f"🏆 Best Score: {results['best_score']:.4f}")
    print(f"🔧 Best Parameters: {results['best_params']}")
    print(f"📁 Results saved to: {output_path}")
    print("\n💡 Next steps:")
    print("1. Review the optimization report and visualizations")
    print("2. Apply the best parameters to your Step 3 HMM regime discovery")
    print("3. Validate the optimized clusters using the cluster validation tools")
    print("4. Integrate the best parameters into your pipeline configuration")


if __name__ == "__main__":
    main()