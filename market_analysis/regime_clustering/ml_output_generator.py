#!/usr/bin/env python3
"""
ML Output Generator

This module generates ML-ready outputs from regime consolidation results,
including training datasets, cluster profiles, and feature mappings.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import json
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class MLTrainingDataset:
    """Container for ML training dataset."""
    
    features: pd.DataFrame
    cluster_labels: np.ndarray
    cluster_metadata: Dict[int, Dict[str, Any]]
    feature_names: List[str]
    cluster_names: List[str]
    
    @property
    def n_samples(self) -> int:
        return len(self.features)
    
    @property
    def n_features(self) -> int:
        return len(self.feature_names)
    
    @property
    def n_clusters(self) -> int:
        return len(np.unique(self.cluster_labels))

@dataclass
class ClusterProfile:
    """Profile for a single cluster."""
    
    cluster_id: int
    cluster_name: str
    sample_count: int
    sample_percentage: float
    regime_count: int
    
    # Feature characteristics
    centroid: Dict[str, float]
    feature_ranges: Dict[str, Tuple[float, float]]
    
    # Market interpretation
    market_regime: str
    regime_characteristics: Dict[str, Any]
    
    # Training suitability
    is_trainable: bool
    min_samples_required: int = 1000

class MLOutputGenerator:
    """
    Generator for ML-ready outputs from regime consolidation results.
    
    This class creates:
    - Training datasets for ML models
    - Cluster profiles and interpretations
    - Feature mappings and transformations
    - Validation and quality metrics
    """
    
    def __init__(self, output_dir: str = "market_analysis/regime_clustering/ml_outputs"):
        """Initialize the ML output generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger.getChild("MLOutputGenerator")
        self.logger.info(f"MLOutputGenerator initialized with output directory: {self.output_dir}")
    
    def generate_training_dataset(self, consolidation_result: Any, 
                                original_regime_data: pd.DataFrame) -> MLTrainingDataset:
        """
        Generate ML training dataset from consolidation results.
        
        Args:
            consolidation_result: Result from RegimeConsolidator.consolidate_regimes()
            original_regime_data: Original regime data from HMM discovery
            
        Returns:
            MLTrainingDataset ready for ML training
        """
        
        self.logger.info("Generating ML training dataset")
        
        # Create expanded dataset where each regime contributes samples
        expanded_features, expanded_labels = self._create_expanded_dataset(
            consolidation_result, original_regime_data
        )
        
        # Create cluster metadata
        cluster_metadata = self._create_cluster_metadata(consolidation_result)
        
        # Create feature and cluster names
        feature_names = ['momentum', 'volatility', 'volume', 'trend']
        cluster_names = [f"cluster_{i}" for i in range(len(consolidation_result.final_clusters))]
        
        # Create MLTrainingDataset
        training_dataset = MLTrainingDataset(
            features=expanded_features,
            cluster_labels=expanded_labels,
            cluster_metadata=cluster_metadata,
            feature_names=feature_names,
            cluster_names=cluster_names
        )
        
        self.logger.info(f"Generated training dataset: {training_dataset.n_samples} samples, "
                        f"{training_dataset.n_features} features, {training_dataset.n_clusters} clusters")
        
        return training_dataset
    
    def _create_expanded_dataset(self, consolidation_result: Any, 
                               original_regime_data: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Create expanded dataset where each regime contributes multiple samples."""
        
        expanded_features = []
        expanded_labels = []
        
        # Create mapping from regime names to cluster IDs
        regime_to_cluster = {}
        for cluster in consolidation_result.final_clusters:
            cluster_id = cluster['cluster_id']
            for regime_name in cluster['regime_names']:
                regime_to_cluster[regime_name] = cluster_id
        
        # Expand each regime into multiple samples
        for regime_name, regime_row in original_regime_data.iterrows():
            if regime_name not in regime_to_cluster:
                continue
            
            cluster_id = regime_to_cluster[regime_name]
            sample_count = regime_row['sample_count']
            
            # Create multiple samples for this regime (simulate the actual data)
            n_samples = min(sample_count, 1000)  # Cap at 1000 samples per regime for efficiency
            
            for _ in range(n_samples):
                # Add some noise to simulate real data variation
                noise_scale = 0.01  # 1% noise
                sample_features = [
                    regime_row['momentum_mean'] + np.random.normal(0, noise_scale),
                    regime_row['volatility_mean'] + np.random.normal(0, noise_scale),
                    regime_row['volume_mean'] + np.random.normal(0, noise_scale),
                    regime_row['trend_mean'] + np.random.normal(0, noise_scale)
                ]
                
                expanded_features.append(sample_features)
                expanded_labels.append(cluster_id)
        
        expanded_features_df = pd.DataFrame(
            expanded_features,
            columns=['momentum', 'volatility', 'volume', 'trend']
        )
        
        expanded_labels_array = np.array(expanded_labels)
        
        return expanded_features_df, expanded_labels_array
    
    def _create_cluster_metadata(self, consolidation_result: Any) -> Dict[int, Dict[str, Any]]:
        """Create metadata for each cluster."""
        
        cluster_metadata = {}
        
        for cluster in consolidation_result.final_clusters:
            cluster_id = cluster['cluster_id']
            
            # Calculate market regime interpretation
            market_regime = self._interpret_market_regime(cluster)
            
            cluster_metadata[cluster_id] = {
                'cluster_id': cluster_id,
                'sample_count': cluster['sample_count'],
                'regime_count': cluster['regime_count'],
                'centroid': cluster['centroid'],
                'feature_ranges': cluster['feature_ranges'],
                'market_regime': market_regime,
                'regime_names': cluster['regime_names'],
                'is_trainable': cluster['sample_count'] >= 1000
            }
        
        return cluster_metadata
    
    def _interpret_market_regime(self, cluster: Dict[str, Any]) -> str:
        """Interpret cluster characteristics as a market regime."""
        
        centroid = cluster['centroid']
        momentum = centroid[0]
        volatility = centroid[1]
        volume = centroid[2]
        trend = centroid[3]
        
        # Determine market regime based on feature values
        if volatility > 0.5:
            if momentum > 0.3:
                return "High_Volatility_Bull"
            elif momentum < -0.3:
                return "High_Volatility_Bear"
            else:
                return "High_Volatility_Sideways"
        elif volatility < -0.5:
            if momentum > 0.3:
                return "Low_Volatility_Bull"
            elif momentum < -0.3:
                return "Low_Volatility_Bear"
            else:
                return "Low_Volatility_Sideways"
        else:
            if momentum > 0.3:
                return "Moderate_Volatility_Bull"
            elif momentum < -0.3:
                return "Moderate_Volatility_Bear"
            else:
                return "Moderate_Volatility_Sideways"
    
    def generate_cluster_profiles(self, consolidation_result: Any) -> List[ClusterProfile]:
        """Generate detailed profiles for each cluster."""
        
        cluster_profiles = []
        
        for cluster in consolidation_result.final_clusters:
            # Calculate sample percentage
            total_samples = sum(c['sample_count'] for c in consolidation_result.final_clusters)
            sample_percentage = cluster['sample_count'] / total_samples
            
            # Create cluster profile
            profile = ClusterProfile(
                cluster_id=cluster['cluster_id'],
                cluster_name=f"cluster_{cluster['cluster_id']}",
                sample_count=cluster['sample_count'],
                sample_percentage=sample_percentage,
                regime_count=cluster['regime_count'],
                centroid={
                    'momentum': cluster['centroid'][0],
                    'volatility': cluster['centroid'][1],
                    'volume': cluster['centroid'][2],
                    'trend': cluster['centroid'][3]
                },
                feature_ranges=cluster['feature_ranges'],
                market_regime=self._interpret_market_regime(cluster),
                regime_characteristics={
                    'regime_names': cluster['regime_names'],
                    'feature_ranges': cluster['feature_ranges']
                },
                is_trainable=cluster['sample_count'] >= 1000
            )
            
            cluster_profiles.append(profile)
        
        # Sort by sample count (descending)
        cluster_profiles.sort(key=lambda x: x.sample_count, reverse=True)
        
        return cluster_profiles
    
    def save_ml_outputs(self, training_dataset: MLTrainingDataset, 
                       cluster_profiles: List[ClusterProfile],
                       consolidation_result: Any,
                       symbol: str, timeframe: str) -> Dict[str, Path]:
        """
        Save all ML outputs to files.
        
        Args:
            training_dataset: MLTrainingDataset to save
            cluster_profiles: List of ClusterProfile objects
            consolidation_result: Consolidation result for metadata
            symbol: Trading symbol
            timeframe: Data timeframe
            
        Returns:
            Dictionary mapping output types to file paths
        """
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        symbol_timeframe = f"{symbol}_{timeframe}"
        
        saved_files = {}
        
        # Save training dataset
        dataset_file = self.output_dir / f"training_dataset_{symbol_timeframe}_{timestamp}.csv"
        training_dataset.features.to_csv(dataset_file, index=False)
        saved_files['training_dataset'] = dataset_file
        
        # Save cluster labels
        labels_file = self.output_dir / f"cluster_labels_{symbol_timeframe}_{timestamp}.npy"
        np.save(labels_file, training_dataset.cluster_labels)
        saved_files['cluster_labels'] = labels_file
        
        # Save cluster metadata
        metadata_file = self.output_dir / f"cluster_metadata_{symbol_timeframe}_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(training_dataset.cluster_metadata, f, indent=2, default=str)
        saved_files['cluster_metadata'] = metadata_file
        
        # Save cluster profiles
        profiles_data = []
        for profile in cluster_profiles:
            profiles_data.append({
                'cluster_id': profile.cluster_id,
                'cluster_name': profile.cluster_name,
                'sample_count': profile.sample_count,
                'sample_percentage': profile.sample_percentage,
                'regime_count': profile.regime_count,
                'centroid': profile.centroid,
                'feature_ranges': profile.feature_ranges,
                'market_regime': profile.market_regime,
                'is_trainable': profile.is_trainable
            })
        
        profiles_file = self.output_dir / f"cluster_profiles_{symbol_timeframe}_{timestamp}.json"
        with open(profiles_file, 'w') as f:
            json.dump(profiles_data, f, indent=2, default=str)
        saved_files['cluster_profiles'] = profiles_file
        
        # Save comprehensive summary
        summary_file = self.output_dir / f"ml_outputs_summary_{symbol_timeframe}_{timestamp}.json"
        summary_data = {
            'metadata': {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': timestamp,
                'generated_at': datetime.now().isoformat()
            },
            'dataset_info': {
                'n_samples': training_dataset.n_samples,
                'n_features': training_dataset.n_features,
                'n_clusters': training_dataset.n_clusters,
                'feature_names': training_dataset.feature_names,
                'cluster_names': training_dataset.cluster_names
            },
            'consolidation_info': {
                'original_regime_count': consolidation_result.original_regime_count,
                'final_cluster_count': consolidation_result.final_cluster_count,
                'coverage_percentage': consolidation_result.coverage_percentage,
                'top_clusters_coverage': consolidation_result.top_clusters_coverage,
                'balance_score': consolidation_result.balance_score
            },
            'cluster_summary': [
                {
                    'cluster_id': profile.cluster_id,
                    'market_regime': profile.market_regime,
                    'sample_count': profile.sample_count,
                    'sample_percentage': profile.sample_percentage,
                    'is_trainable': profile.is_trainable
                }
                for profile in cluster_profiles
            ]
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        saved_files['summary'] = summary_file
        
        self.logger.info(f"Saved ML outputs to {len(saved_files)} files:")
        for output_type, file_path in saved_files.items():
            self.logger.info(f"  {output_type}: {file_path}")
        
        return saved_files
    
    def validate_training_dataset(self, training_dataset: MLTrainingDataset) -> Dict[str, Any]:
        """Validate training dataset quality for ML training."""
        
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'recommendations': [],
            'statistics': {}
        }
        
        # Basic statistics
        validation_results['statistics'] = {
            'n_samples': training_dataset.n_samples,
            'n_features': training_dataset.n_features,
            'n_clusters': training_dataset.n_clusters,
            'samples_per_cluster': {
                str(cluster_id): int(np.sum(training_dataset.cluster_labels == cluster_id))
                for cluster_id in np.unique(training_dataset.cluster_labels)
            }
        }
        
        # Check minimum samples per cluster
        min_samples_per_cluster = min(validation_results['statistics']['samples_per_cluster'].values())
        if min_samples_per_cluster < 100:
            validation_results['warnings'].append(f"Some clusters have very few samples: minimum = {min_samples_per_cluster}")
            validation_results['recommendations'].append("Consider merging small clusters or collecting more data")
        
        # Check class balance
        cluster_counts = validation_results['statistics']['samples_per_cluster']
        max_count = max(cluster_counts.values())
        min_count = min(cluster_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        if imbalance_ratio > 10:
            validation_results['warnings'].append(f"High class imbalance: ratio = {imbalance_ratio:.1f}")
            validation_results['recommendations'].append("Consider class balancing techniques for training")
        
        # Check feature distributions
        for feature_name in training_dataset.feature_names:
            if feature_name in training_dataset.features.columns:
                feature_std = training_dataset.features[feature_name].std()
                if feature_std < 0.001:
                    validation_results['warnings'].append(f"Very low variance in {feature_name}: {feature_std:.6f}")
                    validation_results['recommendations'].append(f"Consider removing or transforming {feature_name}")
        
        # Check for missing values
        missing_values = training_dataset.features.isnull().sum().sum()
        if missing_values > 0:
            validation_results['warnings'].append(f"Found {missing_values} missing values in features")
            validation_results['recommendations'].append("Handle missing values before training")
        
        # Overall validation
        if validation_results['warnings']:
            validation_results['is_valid'] = False
        
        return validation_results
    
    def create_feature_importance_report(self, training_dataset: MLTrainingDataset) -> pd.DataFrame:
        """Create feature importance report for the training dataset."""
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Prepare data
        X = training_dataset.features.values
        y = training_dataset.cluster_labels
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train random forest for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y)
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': training_dataset.feature_names,
            'importance': rf.feature_importances_,
            'rank': range(1, len(training_dataset.feature_names) + 1)
        }).sort_values('importance', ascending=False)
        
        # Reset rank after sorting
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        return importance_df
    
    def generate_training_recommendations(self, training_dataset: MLTrainingDataset,
                                        cluster_profiles: List[ClusterProfile]) -> Dict[str, Any]:
        """Generate recommendations for ML training."""
        
        recommendations = {
            'model_types': [],
            'preprocessing': [],
            'training_strategies': [],
            'evaluation_metrics': []
        }
        
        # Model type recommendations
        n_samples = training_dataset.n_samples
        n_clusters = training_dataset.n_clusters
        
        if n_samples > 10000:
            recommendations['model_types'].extend([
                'Random Forest Classifier',
                'Gradient Boosting Classifier',
                'Neural Networks'
            ])
        else:
            recommendations['model_types'].extend([
                'Random Forest Classifier',
                'Logistic Regression',
                'Support Vector Machine'
            ])
        
        # Preprocessing recommendations
        recommendations['preprocessing'].extend([
            'StandardScaler for feature normalization',
            'Handle class imbalance with SMOTE or class weights',
            'Feature selection based on importance scores'
        ])
        
        # Training strategy recommendations
        trainable_clusters = sum(1 for profile in cluster_profiles if profile.is_trainable)
        
        if trainable_clusters >= 15:
            recommendations['training_strategies'].extend([
                'Train separate models for each cluster',
                'Use ensemble methods combining cluster-specific models',
                'Implement hierarchical classification'
            ])
        else:
            recommendations['training_strategies'].extend([
                'Focus on top 10-15 clusters for training',
                'Use multi-class classification with all clusters',
                'Consider cluster merging for better model performance'
            ])
        
        # Evaluation metrics
        recommendations['evaluation_metrics'].extend([
            'Accuracy (overall)',
            'Precision, Recall, F1-score (per cluster)',
            'Confusion Matrix',
            'Classification Report',
            'Cross-validation scores'
        ])
        
        return recommendations


def create_ml_output_generator(output_dir: Optional[str] = None) -> MLOutputGenerator:
    """Create and return a new MLOutputGenerator instance."""
    if output_dir is None:
        output_dir = "market_analysis/regime_clustering/ml_outputs"
    
    return MLOutputGenerator(output_dir)