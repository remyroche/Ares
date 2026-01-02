"""
Enhanced Specialist Diagnostics Mixin - Independence Pattern

This mixin provides self-contained diagnostic capabilities for specialist models,
eliminating dependencies on the meta-labeling pipeline and get_specialist_models_outputs.
Enhanced with MI/HSIC analysis and orthogonality checks.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import logging

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform

from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.versioned_artifacts import VersionedArtifactStore


class SpecialistDiagnosticsMixinEnhanced:
    """
    Enhanced mixin providing independent diagnostic capabilities for specialist models.
    
    This mixin allows specialists to:
    - Load their own artifacts directly
    - Compute their own metrics without meta-labeling dependency
    - Generate their own diagnostic reports
    - Analyze feature importance and stability
    - Compute MI/HSIC scores to target
    - Ensure feature orthogonality
    - Produce single 0/1 scalar outputs
    """
    
    def _load_self_artifacts(self, symbol: str, exchange: str, timeframe: str, direction: str) -> Dict[str, Any]:
        """Load this specialist's own artifacts."""
        from src.utils.versioned_artifacts import VersionedArtifactStore
        artifact_store = VersionedArtifactStore("versioned_artifacts")
        
        # Load predictions
        artifact_name = f"{self.step_name}_{timeframe}"
        try:
            # Get current version view
            view = artifact_store.get_view()
            predictions_data = view.data
            tprint_success(f"✅ Loaded specialist predictions: {artifact_name}")
        except ValueError as e:
            if "No versions available" in str(e):
                tprint_warning(f"⚠️ No artifacts available for {artifact_name}")
                return {}
            else:
                tprint_error(f"❌ Failed to load predictions {artifact_name}: {e}")
                return {}
        except Exception as e:
            tprint_error(f"❌ Failed to load predictions {artifact_name}: {e}")
            return {}
        
        # Load model (skip for now as model loading needs different approach)
        model = None
        
        return {
            'predictions_data': predictions_data,
            'model': model,
            'metadata': predictions_data.get('metadata', {}) if predictions_data else {}
        }
    
    def _compute_hsic(self, X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> float:
        """
        Compute Hilbert-Schmidt Independence Criterion (HSIC).
        
        HSIC measures dependence between two variables using kernel methods.
        Higher values indicate stronger dependence.
        """
        # Ensure arrays are 1D
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
        
        n = X.shape[0]
        
        # RBF kernel function
        def rbf_kernel(X, Y=None, sigma=sigma):
            if Y is None:
                Y = X
            pairwise_dists = pdist(X, 'sqeuclidean')
            K = np.exp(-pairwise_dists / (2 * sigma ** 2))
            return squareform(K)
        
        # Compute centered kernel matrices
        K = rbf_kernel(X)
        L = rbf_kernel(Y)
        
        # Center the kernels
        H = np.eye(n) - np.ones((n, n)) / n
        K_centered = H @ K @ H
        L_centered = H @ L @ H
        
        # HSIC statistic
        hsic = np.trace(K_centered @ L_centered) / (n ** 2)
        
        return hsic
    
    def _compute_mutual_information_scores(self, features: pd.DataFrame, labels: pd.Series, 
                                        predictions: np.ndarray) -> Dict[str, Any]:
        """Compute MI and HSIC scores to target/context."""
        mi_scores = {}
        hsic_scores = {}
        
        # Clean data
        valid_idx = ~(features.isna().any(axis=1) | labels.isna())
        features_clean = features[valid_idx]
        labels_clean = labels[valid_idx]
        predictions_clean = predictions[valid_idx]
        
        if len(features_clean) < 100:
            tprint_warning("⚠️ Insufficient data for MI/HSIC analysis")
            return {'feature_mi': {}, 'feature_hsic': {}, 'prediction_mi': 0, 'prediction_hsic': 0}
        
        # MI to target (price movement context)
        for col in features_clean.columns:
            try:
                # Handle categorical vs continuous
                if len(np.unique(features_clean[col])) < 10:  # Likely categorical
                    mi = mutual_info_regression(
                        features_clean[[col]], 
                        labels_clean, 
                        discrete_features=True
                    )[0]
                else:  # Continuous
                    mi = mutual_info_regression(
                        features_clean[[col]], 
                        labels_clean, 
                        discrete_features=False
                    )[0]
                mi_scores[col] = mi
            except Exception as e:
                tprint_warning(f"⚠️ MI computation failed for {col}: {e}")
                mi_scores[col] = 0
        
        # HSIC to target (non-linear dependence)
        for col in features_clean.columns:
            try:
                hsic = self._compute_hsic(
                    features_clean[col].values, 
                    labels_clean.values,
                    sigma=np.std(features_clean[col].values)
                )
                hsic_scores[col] = hsic
            except Exception as e:
                tprint_warning(f"⚠️ HSIC computation failed for {col}: {e}")
                hsic_scores[col] = 0
        
        # Prediction MI to target
        try:
            pred_mi = mutual_info_regression(
                predictions_clean.reshape(-1, 1), 
                labels_clean
            )[0]
        except:
            pred_mi = 0
        
        # Prediction HSIC to target
        try:
            pred_hsic = self._compute_hsic(
                predictions_clean, 
                labels_clean.values,
                sigma=np.std(predictions_clean)
            )
        except:
            pred_hsic = 0
        
        return {
            'feature_mi': mi_scores,
            'feature_hsic': hsic_scores,
            'prediction_mi': pred_mi,
            'prediction_hsic': pred_hsic,
            'avg_feature_mi': np.mean(list(mi_scores.values())) if mi_scores else 0,
            'avg_feature_hsic': np.mean(list(hsic_scores.values())) if hsic_scores else 0
        }
    
    def _enforce_feature_orthogonality(self, features: pd.DataFrame, max_correlation: float = 0.7) -> Tuple[pd.DataFrame, set]:
        """Remove highly correlated features within specialist."""
        if len(features.columns) < 2:
            return features, set()
        
        # Compute correlation matrix
        corr_matrix = features.corr().abs()
        
        # Find correlations above threshold
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        high_corr_pairs = []
        for col1, col2 in upper_tri.stack().index:
            corr_val = upper_tri.loc[col1, col2]
            if not pd.isna(corr_val) and corr_val > max_correlation:
                high_corr_pairs.append((col1, col2, corr_val))
        
        # Remove one feature from each highly correlated pair
        features_to_drop = set()
        for col1, col2, corr in high_corr_pairs:
            # Keep the feature with higher MI to target (if available)
            if col1 not in features_to_drop and col2 not in features_to_drop:
                # Simple heuristic: drop the second feature
                features_to_drop.add(col2)
                tprint_info(f"🔄 Dropping {col2} (corr={corr:.3f} with {col1})")
        
        orthogonal_features = features.drop(columns=features_to_drop)
        
        if features_to_drop:
            tprint_info(f"📊 Orthogonalization: dropped {len(features_to_drop)} correlated features")
        
        return orthogonal_features, features_to_drop
    
    def _finalize_binary_output(self, probabilities: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Convert probabilities to single 0/1 scalar."""
        return (probabilities >= threshold).astype(int)
    
    def _compute_specialist_metrics(self, features: pd.DataFrame, labels: pd.Series, 
                                  predictions: np.ndarray, probabilities: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Compute comprehensive metrics for specialist performance with MI/HSIC analysis."""
        metrics = {}
        
        # Ensure binary predictions
        binary_predictions = self._finalize_binary_output(probabilities)
        
        # Classification metrics
        if len(np.unique(labels)) > 1:
            try:
                metrics['auc'] = roc_auc_score(labels, probabilities)
                metrics['accuracy'] = accuracy_score(labels, binary_predictions)
                metrics['brier_loss'] = brier_score_loss(labels, probabilities)
            except Exception as e:
                tprint_warning(f"⚠️ Classification metrics failed: {e}")
        
        # Regression metrics
        try:
            metrics['mse'] = mean_squared_error(labels, probabilities)
            metrics['r2'] = r2_score(labels, probabilities)
        except Exception as e:
            tprint_warning(f"⚠️ Regression metrics failed: {e}")
        
        # MI/HSIC metrics
        mi_scores = self._compute_mutual_information_scores(features, labels, binary_predictions)
        metrics.update({
            'prediction_mi_to_target': mi_scores['prediction_mi'],
            'prediction_hsic_to_target': mi_scores['prediction_hsic'],
            'avg_feature_mi': mi_scores['avg_feature_mi'],
            'avg_feature_hsic': mi_scores['avg_feature_hsic']
        })
        
        # Feature orthogonality metrics
        orthogonal_features, dropped_features = self._enforce_feature_orthogonality(features)
        metrics.update({
            'original_feature_count': len(features.columns),
            'orthogonal_feature_count': len(orthogonal_features.columns),
            'dropped_correlated_features': len(dropped_features)
        })
        
        return metrics, mi_scores
    
    def _compute_temporal_stability(self, features: pd.DataFrame, labels: pd.Series) -> Dict[str, float]:
        """Compute temporal stability using time series cross-validation."""
        stability_scores = []
        
        try:
            tscv = TimeSeriesSplit(n_splits=5)
            lr = LogisticRegression(random_state=42, max_iter=1000)
            
            for train_idx, val_idx in tscv.split(features):
                X_train = features.iloc[train_idx].fillna(0)
                y_train = labels.iloc[train_idx]
                X_val = features.iloc[val_idx].fillna(0)
                y_val = labels.iloc[val_idx]
                
                if len(np.unique(y_train)) > 1:
                    lr.fit(X_train, y_train)
                    
                    if len(np.unique(y_val)) > 1:
                        score = roc_auc_score(y_val, lr.predict_proba(X_val.fillna(0))[:, 1])
                        stability_scores.append(score)
            
            stability_mean = np.mean(stability_scores) if stability_scores else 0
            stability_cv = np.std(stability_scores) / np.mean(stability_scores) if stability_scores and np.mean(stability_scores) > 0 else 0
            
            return {
                'stability_mean': stability_mean,
                'stability_cv': stability_cv,
                'stability_scores': stability_scores
            }
        except Exception as e:
            tprint_warning(f"⚠️ Temporal stability computation failed: {e}")
            return {
                'stability_mean': 0,
                'stability_cv': 0,
                'stability_scores': []
            }
    
    def _generate_diagnostic_report(self, metrics: Dict[str, float], feature_importance: Dict[str, float], 
                                 stability: Dict[str, float], symbol: str, exchange: str, 
                                 timeframe: str, direction: str) -> str:
        """Generate comprehensive diagnostic report."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report = f"""# {self.step_name.title()} Diagnostic Report

**Symbol:** {symbol} | **Exchange:** {exchange} | **Timeframe:** {timeframe} | **Direction:** {direction}  
**Generated:** {timestamp}

## Performance Metrics

| Metric | Value |
|--------|-------|
| AUC | {metrics.get('auc', 0):.3f} |
| Accuracy | {metrics.get('accuracy', 0):.3f} |
| R² | {metrics.get('r2', 0):.3f} |
| Brier Loss | {metrics.get('brier_loss', 0):.3f} |
| MSE | {metrics.get('mse', 0):.3f} |

## Information Theory Metrics

| Metric | Value |
|--------|-------|
| Prediction MI to Target | {metrics.get('prediction_mi_to_target', 0):.4f} |
| Prediction HSIC to Target | {metrics.get('prediction_hsic_to_target', 0):.4f} |
| Average Feature MI | {metrics.get('avg_feature_mi', 0):.4f} |
| Average Feature HSIC | {metrics.get('avg_feature_hsic', 0):.4f} |

## Feature Orthogonality

| Metric | Value |
|--------|-------|
| Original Feature Count | {metrics.get('original_feature_count', 0)} |
| Orthogonal Feature Count | {metrics.get('orthogonal_feature_count', 0)} |
| Dropped Correlated Features | {metrics.get('dropped_correlated_features', 0)} |

## Temporal Stability

| Metric | Value |
|--------|-------|
| Stability Mean | {stability.get('stability_mean', 0):.3f} |
| Stability CV | {stability.get('stability_cv', 0):.3f} |

## Top 10 Feature Importance

"""
        
        # Add top features
        if feature_importance:
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for i, (feature, importance) in enumerate(top_features, 1):
                report += f"{i}. {feature}: {importance:.4f}\n"
        
        report += f"""
## Recommendations

"""
        
        # Add recommendations based on metrics
        if metrics.get('prediction_mi_to_target', 0) < 0.01:
            report += "- ⚠️ Low MI to target - consider feature engineering\n"
        
        if metrics.get('prediction_hsic_to_target', 0) < 0.01:
            report += "- ⚠️ Low HSIC to target - consider non-linear transformations\n"
        
        if metrics.get('dropped_correlated_features', 0) > 5:
            report += "- ⚠️ High feature correlation detected - orthogonalization applied\n"
        
        if stability.get('stability_mean', 0) < 0.55:
            report += "- ⚠️ Low temporal stability - consider regularization\n"
        
        if metrics.get('auc', 0) > 0.6 and metrics.get('prediction_mi_to_target', 0) > 0.02:
            report += "- ✅ Good performance with high information content\n"
        
        return report
    
    def _save_diagnostic_report(self, report: str, metrics: Dict[str, float], 
                              symbol: str, exchange: str, timeframe: str, direction: str) -> Tuple[str, str]:
        """Save diagnostic report and metrics."""
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"{self.step_name}_diagnostics_{symbol}_{timeframe}_{direction}_{timestamp}"
        
        # Save markdown report
        md_path = outcomes_dir / f"{base_name}.md"
        with open(md_path, 'w') as f:
            f.write(report)
        
        # Save metrics CSV
        metrics_df = pd.DataFrame([metrics])
        csv_path = outcomes_dir / f"{base_name}_metrics.csv"
        metrics_df.to_csv(csv_path, index=False)
        
        tprint_success(f"💾 Diagnostic report saved: {md_path}")
        tprint_success(f"💾 Metrics saved: {csv_path}")
        
        return str(md_path), str(csv_path)
    
    def run_diagnostics(self, symbol: str, exchange: str, timeframe: str, direction: str) -> Dict[str, Any]:
        """Run comprehensive diagnostics for this specialist."""
        try:
            tprint_info(f"🔍 Running diagnostics for {self.step_name}...")
            
            # Load artifacts
            artifacts = self._load_self_artifacts(symbol, exchange, timeframe, direction)
            if not artifacts:
                return {'success': False, 'error': 'Failed to load artifacts'}
            
            predictions_data = artifacts['predictions_data']
            if not isinstance(predictions_data, pd.DataFrame):
                return {'success': False, 'error': 'Invalid predictions data format'}
            
            # Extract features, labels, predictions
            feature_cols = [col for col in predictions_data.columns if col.endswith('_feature') or col in ['close', 'volume']]
            if not feature_cols:
                # Use all columns except predictions and labels
                exclude_cols = [col for col in predictions_data.columns if 'prediction' in col or 'label' in col or 'probability' in col]
                feature_cols = [col for col in predictions_data.columns if col not in exclude_cols]
            
            features = predictions_data[feature_cols].copy()
            labels = predictions_data.get(f'{self.step_name}_label', pd.Series())
            predictions = predictions_data.get(f'{self.step_name}_prediction', pd.Series())
            probabilities = predictions_data.get(f'{self.step_name}_probability', pd.Series())
            
            if labels.empty or predictions.empty:
                return {'success': False, 'error': 'Missing labels or predictions'}
            
            # Compute enhanced metrics
            metrics, mi_scores = self._compute_specialist_metrics(features, labels, predictions, probabilities)
            stability = self._compute_temporal_stability(features, labels)
            
            # Generate report
            report = self._generate_diagnostic_report(metrics, mi_scores, stability, symbol, exchange, timeframe, direction)
            
            # Save results
            md_path, csv_path = self._save_diagnostic_report(report, metrics, symbol, exchange, timeframe, direction)
            
            tprint_success(f"✅ {self.step_name} diagnostics completed")
            tprint_info(f"📊 AUC: {metrics.get('auc', 0):.3f}, MI: {metrics.get('prediction_mi_to_target', 0):.4f}")
            
            return {
                'success': True,
                'metrics': metrics,
                'feature_importance': mi_scores,
                'stability': stability,
                'report_path': md_path,
                'csv_path': csv_path
            }
            
        except Exception as e:
            tprint_error(f"❌ Diagnostics failed for {self.step_name}: {e}")
            return {'success': False, 'error': str(e)}
