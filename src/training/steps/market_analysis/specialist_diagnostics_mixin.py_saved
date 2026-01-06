"""
Specialist Diagnostics Mixin - Independence Pattern

This mixin provides self-contained diagnostic capabilities for specialist models,
eliminating dependencies on the meta-labeling pipeline and get_specialist_models_outputs.
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

from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.versioned_artifacts import VersionedArtifactStore


class SpecialistDiagnosticsMixin:
    """
    Mixin providing independent diagnostic capabilities for specialist models.
    
    This mixin allows specialists to:
    - Load their own artifacts directly
    - Compute their own metrics without meta-labeling dependency
    - Generate their own diagnostic reports
    - Analyze feature importance and stability
    """
    
    def _load_self_artifacts(self, symbol: str, exchange: str, timeframe: str, direction: str) -> Dict[str, Any]:
        """Load this specialist's own artifacts."""
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
    
    def _compute_specialist_metrics(self, features: pd.DataFrame, labels: pd.Series, 
                                  predictions: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive metrics for specialist performance."""
        metrics = {}
        
        # Classification metrics
        if len(np.unique(labels)) > 1:
            try:
                metrics['auc'] = roc_auc_score(labels, probabilities)
                metrics['accuracy'] = accuracy_score(labels, predictions)
                metrics['brier_loss'] = brier_score_loss(labels, probabilities)
            except Exception as e:
                tprint_warning(f"⚠️ Classification metrics failed: {e}")
        
        # Regression metrics
        try:
            metrics['mse'] = mean_squared_error(labels, probabilities)
            metrics['r2'] = r2_score(labels, probabilities)
        except Exception as e:
            tprint_warning(f"⚠️ Regression metrics failed: {e}")
        
        # Feature importance via correlation
        feature_importance = {}
        for col in features.columns:
            try:
                corr = np.corrcoef(features[col].fillna(0), labels)[0, 1]
                if np.isfinite(corr):
                    feature_importance[col] = abs(corr)
            except:
                continue
        
        if feature_importance:
            metrics['mean_feature_correlation'] = np.mean(list(feature_importance.values()))
            metrics['max_feature_correlation'] = np.max(list(feature_importance.values()))
        
        return metrics, feature_importance
    
    def _compute_temporal_stability(self, features: pd.DataFrame, labels: pd.Series,
                                  n_splits: int = 5) -> Dict[str, float]:
        """Compute temporal stability metrics using time series cross-validation."""
        if len(features) < n_splits * 100:
            return {'stability_mean': 0.0, 'stability_cv': 1.0}
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_scores = []
        
        for train_idx, val_idx in tscv.split(features):
            if len(train_idx) < 50 or len(val_idx) < 50:
                continue
                
            try:
                X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
                y_train, y_val = labels.iloc[train_idx], labels.iloc[val_idx]
                
                # Simple logistic regression for stability testing
                lr = LogisticRegression(random_state=42, max_iter=1000)
                lr.fit(X_train.fillna(0), y_train)
                
                if len(np.unique(y_val)) > 1:
                    score = roc_auc_score(y_val, lr.predict_proba(X_val.fillna(0))[:, 1])
                    fold_scores.append(score)
            except:
                continue
        
        if fold_scores:
            return {
                'stability_mean': np.mean(fold_scores),
                'stability_cv': np.std(fold_scores) / (np.mean(fold_scores) + 1e-8)
            }
        else:
            return {'stability_mean': 0.0, 'stability_cv': 1.0}
    
    def _generate_diagnostic_report(self, metrics: Dict[str, float], 
                                  feature_importance: Dict[str, float],
                                  stability: Dict[str, float],
                                  symbol: str, exchange: str, timeframe: str, direction: str) -> str:
        """Generate markdown diagnostic report."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report = f"""# {self.step_name.replace('_', ' ').title()} Diagnostic Report

**Generated:** {timestamp}  
**Symbol:** {symbol} | **Exchange:** {exchange} | **Timeframe:** {timeframe} | **Direction:** {direction}

## Performance Metrics

| Metric | Value |
|--------|-------|"""
        
        for metric, value in metrics.items():
            if isinstance(value, float):
                report += f"\n| {metric} | {value:.4f} |"
            else:
                report += f"\n| {metric} | {value} |"
        
        report += f"""

## Temporal Stability

| Metric | Value |
|--------|-------|"""
        
        for metric, value in stability.items():
            if isinstance(value, float):
                report += f"\n| {metric} | {value:.4f} |"
            else:
                report += f"\n| {metric} | {value} |"
        
        # Top features
        if feature_importance:
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            report += f"""

## Top 10 Features by Importance

| Feature | Importance |
|---------|------------|"""
            for feature, importance in top_features:
                report += f"\n| {feature} | {importance:.4f} |"
        
        report += f"""

## Summary

This specialist model {'performs' if metrics.get('auc', 0) > 0.5 else 'underperforms'} with AUC = {metrics.get('auc', 0):.3f}.
Temporal stability is {'good' if stability.get('stability_cv', 1) < 0.2 else 'poor'} with CV = {stability.get('stability_cv', 1):.3f}.

---
*Report generated by {self.step_name} independent diagnostics*
"""
        
        return report
    
    def _save_diagnostic_report(self, report: str, metrics: Dict[str, float],
                              symbol: str, exchange: str, timeframe: str, direction: str):
        """Save diagnostic report and metrics to files."""
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"{self.step_name}_diagnostics_{symbol}_{timeframe}_{direction}_{timestamp}"
        
        # Save markdown report
        md_path = outcomes_dir / f"{base_name}.md"
        with open(md_path, 'w') as f:
            f.write(report)
        
        # Save metrics as CSV
        csv_path = outcomes_dir / f"{base_name}.csv"
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(csv_path, index=False)
        
        tprint_success(f"✅ Diagnostic report saved: {md_path}")
        tprint_success(f"✅ Metrics saved: {csv_path}")
        
        return str(md_path), str(csv_path)
    
    def run_self_diagnostics(self, symbol: str, exchange: str, timeframe: str, direction: str) -> Dict[str, Any]:
        """Run complete self-contained diagnostics."""
        tprint_info(f"🔍 Running {self.step_name} self-diagnostics...")
        
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
        
        # Compute metrics
        metrics, feature_importance = self._compute_specialist_metrics(features, labels, predictions, probabilities)
        stability = self._compute_temporal_stability(features, labels)
        
        # Generate report
        report = self._generate_diagnostic_report(metrics, feature_importance, stability, symbol, exchange, timeframe, direction)
        
        # Save results
        md_path, csv_path = self._save_diagnostic_report(report, metrics, symbol, exchange, timeframe, direction)
        
        return {
            'success': True,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'stability': stability,
            'report_path': md_path,
            'csv_path': csv_path
        }
