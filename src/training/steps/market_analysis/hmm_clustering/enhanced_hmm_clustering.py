#!/usr/bin/env python3
"""
Enhanced HMM Clustering with Dynamic Parameter Optimization and Feature Selection

This module integrates all the improvements:
- Dynamic parameter optimization
- Systematic feature selection
- Ensemble weight optimization
- Enhanced feature engineering

Author: AI Assistant
Date: 2024-01-XX
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
import time
import logging
from pathlib import Path
import json

# Import our enhancement modules
from .parameter_optimization import ParameterOptimizer, OptimizationResult
from .feature_selection import FeatureSelector, EnhancedFeatureEngineer, FeatureSelectionResult
from .ensemble_optimization import EnsembleWeightOptimizer, OptimizationResult as EnsembleOptimizationResult

# HMM imports
try:
    from hmmlearn import hmm
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False

# Sklearn imports
try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from src.utils.logger import system_logger

@dataclass
class EnhancedHMMResult:
    """Result of enhanced HMM clustering"""
    hmm_predictions: np.ndarray
    kmeans_predictions: np.ndarray
    dbscan_predictions: np.ndarray
    ensemble_predictions: np.ndarray
    optimal_weights: Dict[str, float]
    selected_features: List[str]
    optimal_hmm_params: Dict[str, Any]
    feature_scores: pd.DataFrame
    clustering_metrics: Dict[str, float]
    execution_time: float

class EnhancedHMMClustering:
    """Enhanced HMM clustering with all improvements integrated"""
    
    def __init__(self, logger=None):
        self.logger = logger or system_logger.getChild('EnhancedHMMClustering')
        
        # Initialize components
        self.parameter_optimizer = ParameterOptimizer(self.logger)
        self.feature_selector = FeatureSelector(self.logger)
        self.feature_engineer = EnhancedFeatureEngineer(self.logger)
        self.ensemble_optimizer = EnsembleWeightOptimizer(self.logger)
        
        # Results storage
        self.results_history = []
    
    def fit_predict(self, df: pd.DataFrame, 
                   regime_labels: Optional[np.ndarray] = None,
                   n_features: int = 30,
                   use_comprehensive_features: bool = True,
                   optimize_parameters: bool = True,
                   optimize_ensemble: bool = True) -> EnhancedHMMResult:
        """
        Fit and predict using enhanced HMM clustering
        
        Args:
            df: Input DataFrame with OHLCV data
            regime_labels: True regime labels if available (for feature selection)
            n_features: Number of features to select
            use_comprehensive_features: Whether to use comprehensive feature engineering
            optimize_parameters: Whether to optimize HMM parameters
            optimize_ensemble: Whether to optimize ensemble weights
            
        Returns:
            EnhancedHMMResult with all clustering results
        """
        start_time = time.time()
        self.logger.info("🚀 Starting enhanced HMM clustering...")
        
        # Step 1: Feature Engineering
        if use_comprehensive_features:
            self.logger.info("🔧 Creating comprehensive feature set...")
            features = self.feature_engineer.create_comprehensive_features(df)
        else:
            # Use basic features
            features = self._create_basic_features(df)
        
        self.logger.info(f"✅ Created {len(features.columns)} features")
        
        # Step 2: Feature Selection
        if regime_labels is not None:
            self.logger.info("🔍 Performing feature selection...")
            feature_selection_result = self.feature_selector.comprehensive_feature_selection(
                features, regime_labels, n_features=n_features
            )
            selected_features = feature_selection_result.selected_features
            feature_scores = feature_selection_result.feature_scores
        else:
            # Use variance threshold selection if no labels available
            feature_selection_result = self.feature_selector.variance_threshold_selection(
                features, threshold=0.01
            )
            selected_features = feature_selection_result.selected_features[:n_features]
            feature_scores = feature_selection_result.feature_scores
        
        features_selected = features[selected_features]
        self.logger.info(f"✅ Selected {len(selected_features)} features")
        
        # Step 3: Parameter Optimization
        if optimize_parameters:
            self.logger.info("🔧 Optimizing HMM parameters...")
            param_result = self.parameter_optimizer.comprehensive_parameter_optimization(
                features_selected.values, use_optuna=True, n_trials=50
            )
            optimal_hmm_params = param_result.best_params
        else:
            # Use default parameters
            optimal_hmm_params = {
                'n_components': 4,
                'covariance_type': 'full',
                'n_iter': 100,
                'tol': 0.001
            }
        
        self.logger.info(f"✅ HMM parameters: {optimal_hmm_params}")
        
        # Step 4: Individual Algorithm Training
        self.logger.info("🧠 Training individual clustering algorithms...")
        
        # HMM
        hmm_predictions, hmm_model = self._train_hmm(features_selected.values, optimal_hmm_params)
        
        # K-means
        kmeans_predictions, kmeans_model = self._train_kmeans(features_selected.values)
        
        # DBSCAN
        dbscan_predictions, dbscan_model = self._train_dbscan(features_selected.values)
        
        # Step 5: Ensemble Weight Optimization
        if optimize_ensemble:
            self.logger.info("⚖️ Optimizing ensemble weights...")
            
            # Prepare results for ensemble optimization
            hmm_results = {'predictions': hmm_predictions, 'score': 0.5}
            kmeans_results = {'predictions': kmeans_predictions, 'score': 0.4}
            dbscan_results = {'predictions': dbscan_predictions, 'score': 0.3}
            
            # Optimize weights
            ensemble_result = self.ensemble_optimizer.multi_objective_optimization(
                hmm_results, kmeans_results, dbscan_results, features_selected.values
            )
            optimal_weights = ensemble_result.optimal_weights
        else:
            # Use equal weights
            optimal_weights = {'hmm': 0.4, 'kmeans': 0.3, 'dbscan': 0.3}
        
        self.logger.info(f"✅ Optimal weights: {optimal_weights}")
        
        # Step 6: Create Ensemble Predictions
        ensemble_predictions = self._create_ensemble_predictions(
            hmm_predictions, kmeans_predictions, dbscan_predictions, optimal_weights
        )
        
        # Step 7: Calculate Metrics
        clustering_metrics = self._calculate_clustering_metrics(
            features_selected.values, ensemble_predictions
        )
        
        execution_time = time.time() - start_time
        
        # Create result
        result = EnhancedHMMResult(
            hmm_predictions=hmm_predictions,
            kmeans_predictions=kmeans_predictions,
            dbscan_predictions=dbscan_predictions,
            ensemble_predictions=ensemble_predictions,
            optimal_weights=optimal_weights,
            selected_features=selected_features,
            optimal_hmm_params=optimal_hmm_params,
            feature_scores=feature_scores,
            clustering_metrics=clustering_metrics,
            execution_time=execution_time
        )
        
        self.results_history.append(result)
        self.logger.info(f"✅ Enhanced HMM clustering completed in {execution_time:.2f} seconds")
        
        return result
    
    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic features for HMM training"""
        features = pd.DataFrame()
        
        # Basic price features
        features['price_change'] = df['close'].pct_change()
        features['price_range'] = (df['high'] - df['low']) / df['close']
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(df['close'])
        features['macd'] = self._calculate_macd(df['close'])
        
        # Moving averages
        features['sma_20'] = df['close'].rolling(20).mean()
        features['price_vs_sma'] = (df['close'] - features['sma_20']) / features['sma_20']
        
        # Clean features
        features = features.fillna(0)
        
        return features
    
    def _train_hmm(self, features: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, Any]:
        """Train HMM model"""
        if not HMMLEARN_AVAILABLE:
            raise ImportError("hmmlearn not available")
        
        model = hmm.GaussianHMM(**params, random_state=42)
        model.fit(features)
        predictions = model.predict(features)
        
        return predictions, model
    
    def _train_kmeans(self, features: np.ndarray, n_clusters: int = 4) -> Tuple[np.ndarray, Any]:
        """Train K-means model"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn not available")
        
        model = KMeans(n_clusters=n_clusters, random_state=42)
        predictions = model.fit_predict(features)
        
        return predictions, model
    
    def _train_dbscan(self, features: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> Tuple[np.ndarray, Any]:
        """Train DBSCAN model"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn not available")
        
        model = DBSCAN(eps=eps, min_samples=min_samples)
        predictions = model.fit_predict(features)
        
        return predictions, model
    
    def _create_ensemble_predictions(self, hmm_pred: np.ndarray, kmeans_pred: np.ndarray, 
                                   dbscan_pred: np.ndarray, weights: Dict[str, float]) -> np.ndarray:
        """Create ensemble predictions"""
        # Normalize predictions to [0, 1] range for weighted combination
        hmm_norm = (hmm_pred - hmm_pred.min()) / (hmm_pred.max() - hmm_pred.min() + 1e-8)
        kmeans_norm = (kmeans_pred - kmeans_pred.min()) / (kmeans_pred.max() - kmeans_pred.min() + 1e-8)
        dbscan_norm = (dbscan_pred - dbscan_pred.min()) / (dbscan_pred.max() - dbscan_pred.min() + 1e-8)
        
        # Weighted combination
        ensemble_pred = (weights['hmm'] * hmm_norm + 
                        weights['kmeans'] * kmeans_norm + 
                        weights['dbscan'] * dbscan_norm)
        
        # Convert back to discrete clusters
        ensemble_clusters = np.round(ensemble_pred * (len(np.unique(hmm_pred)) - 1)).astype(int)
        
        return ensemble_clusters
    
    def _calculate_clustering_metrics(self, features: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
        """Calculate clustering quality metrics"""
        if not SKLEARN_AVAILABLE:
            return {}
        
        try:
            metrics = {
                'silhouette_score': silhouette_score(features, predictions),
                'calinski_harabasz_score': calinski_harabasz_score(features, predictions),
                'davies_bouldin_score': davies_bouldin_score(features, predictions)
            }
        except Exception as e:
            self.logger.warning(f"Error calculating clustering metrics: {e}")
            metrics = {}
        
        return metrics
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - 100 / (1 + rs)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        return ema_fast - ema_slow
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary of all clustering runs"""
        if not self.results_history:
            return {"message": "No clustering runs recorded"}
        
        summary = {
            "total_runs": len(self.results_history),
            "average_execution_time": np.mean([r.execution_time for r in self.results_history]),
            "average_silhouette_score": np.mean([r.clustering_metrics.get('silhouette_score', 0) for r in self.results_history]),
            "runs": []
        }
        
        for i, result in enumerate(self.results_history):
            summary["runs"].append({
                "run_id": i,
                "execution_time": result.execution_time,
                "n_features": len(result.selected_features),
                "clustering_metrics": result.clustering_metrics,
                "optimal_weights": result.optimal_weights
            })
        
        return summary
    
    def save_results(self, filepath: str) -> None:
        """Save clustering results to file"""
        results = {
            "results_history": [
                {
                    "hmm_predictions": r.hmm_predictions.tolist(),
                    "kmeans_predictions": r.kmeans_predictions.tolist(),
                    "dbscan_predictions": r.dbscan_predictions.tolist(),
                    "ensemble_predictions": r.ensemble_predictions.tolist(),
                    "optimal_weights": r.optimal_weights,
                    "selected_features": r.selected_features,
                    "optimal_hmm_params": r.optimal_hmm_params,
                    "clustering_metrics": r.clustering_metrics,
                    "execution_time": r.execution_time
                }
                for r in self.results_history
            ],
            "summary": self.get_results_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"💾 Results saved to {filepath}")

# Example usage and testing
def test_enhanced_hmm_clustering():
    """Test the enhanced HMM clustering functionality"""
    # Generate sample OHLCV data
    np.random.seed(42)
    n_samples = 1000
    
    # Create realistic price data
    prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.01)
    highs = prices + np.random.rand(n_samples) * 2
    lows = prices - np.random.rand(n_samples) * 2
    volumes = np.random.randint(1000, 10000, n_samples)
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1H'),
        'open': prices,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes
    })
    
    # Generate mock regime labels
    regime_labels = np.random.randint(0, 3, n_samples)
    
    # Test enhanced clustering
    clustering = EnhancedHMMClustering()
    
    print("Testing enhanced HMM clustering...")
    result = clustering.fit_predict(
        df, 
        regime_labels=regime_labels,
        n_features=20,
        use_comprehensive_features=True,
        optimize_parameters=True,
        optimize_ensemble=True
    )
    
    print(f"Execution time: {result.execution_time:.2f} seconds")
    print(f"Selected features: {len(result.selected_features)}")
    print(f"Optimal weights: {result.optimal_weights}")
    print(f"Clustering metrics: {result.clustering_metrics}")
    
    # Print summary
    print("\nResults Summary:")
    summary = clustering.get_results_summary()
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    test_enhanced_hmm_clustering()