"""
ML Integration Framework for Enhanced Market Discovery

This module provides a unified interface to integrate all ML enhancements with the existing
clusters framework. It serves as the main entry point for ML-enhanced market regime discovery.

Key Components Integrated:
1. ML-Enhanced Discovery (autoencoders, LSTM, transformers, manifold learning)
2. Automated Feature Engineering (genetic programming, polynomial features, neural synthesis)
3. Adaptive Clustering (multi-criteria optimization, Bayesian optimization, reinforcement learning)
4. Advanced Feature Importance (SHAP, permutation importance, ensemble methods)
5. Regime Transition Prediction (LSTM, transformer models)

Usage:
    framework = MLIntegrationFramework()
    results = framework.complete_ml_discovery(market_data)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
import time
import warnings

from src.utils.logger import system_logger

# Import existing framework components
from .dimension_analyzer import MarketDimensionAnalyzer, DimensionAnalysisConfig
from .regime_clusterer import RegimeClusterer, ClusteringMethod
from .feature_importance import RegimeFeatureImportance, ImportanceMethod
from .validation_metrics import RegimeValidationMetrics

# Import new ML components
from .ml_enhanced_discovery import MLEnhancedDiscovery, MLDiscoveryMethod, MLDiscoveryConfig
from .automated_feature_engineering import AutomatedFeatureEngineer, FeatureEngineeringMethod, AutoFeatureConfig
from .adaptive_clustering import AdaptiveClusteringFramework, AdaptiveMethod, AdaptiveClusteringConfig


@dataclass
class MLIntegrationConfig:
    """Configuration for ML integration framework."""
    # Enable/disable components
    enable_ml_discovery: bool = True
    enable_automated_features: bool = True
    enable_adaptive_clustering: bool = True
    enable_transition_prediction: bool = True
    
    # ML Discovery settings
    ml_discovery_methods: List[str] = None
    ml_latent_dim: int = 10
    ml_epochs: int = 100
    
    # Feature engineering settings
    feature_engineering_methods: List[str] = None
    max_features: int = 500
    poly_degree: int = 2
    
    # Adaptive clustering settings
    adaptive_methods: List[str] = None
    min_clusters: int = 2
    max_clusters: int = 15
    
    # Performance settings
    n_jobs: int = -1
    random_state: int = 42
    verbose: bool = True
    
    # Output settings
    save_results: bool = True
    results_dir: str = "ml_discovery_results"
    
    def __post_init__(self):
        if self.ml_discovery_methods is None:
            self.ml_discovery_methods = ["autoencoder", "lstm_encoder", "manifold_learning"]
        if self.feature_engineering_methods is None:
            self.feature_engineering_methods = ["time_series_features", "domain_specific_features", "polynomial_features"]
        if self.adaptive_methods is None:
            self.adaptive_methods = ["multi_criteria", "ensemble_learning"]


class MLIntegrationFramework:
    """Main framework for ML-enhanced market discovery."""
    
    def __init__(self, config: MLIntegrationConfig = None):
        self.config = config or MLIntegrationConfig()
        self.logger = system_logger.getChild('MLIntegrationFramework')
        
        # Initialize components
        self._initialize_components()
        
        # Results storage
        self.discovery_results = {}
        self.performance_metrics = {}
        self.integration_history = []
    
    def _initialize_components(self):
        """Initialize all ML components."""
        
        # Traditional components
        self.dimension_analyzer = MarketDimensionAnalyzer()
        self.regime_clusterer = RegimeClusterer()
        self.feature_importance = RegimeFeatureImportance()
        self.validator = RegimeValidationMetrics()
        
        # ML Enhancement components
        if self.config.enable_ml_discovery:
            ml_config = MLDiscoveryConfig(
                latent_dim=self.config.ml_latent_dim,
                epochs=self.config.ml_epochs,
                random_state=self.config.random_state
            )
            self.ml_discovery = MLEnhancedDiscovery(ml_config)
        
        if self.config.enable_automated_features:
            feature_config = AutoFeatureConfig(
                max_features=self.config.max_features,
                poly_degree=self.config.poly_degree,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs
            )
            self.feature_engineer = AutomatedFeatureEngineer(feature_config)
        
        if self.config.enable_adaptive_clustering:
            clustering_config = AdaptiveClusteringConfig(
                min_clusters=self.config.min_clusters,
                max_clusters=self.config.max_clusters,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                verbose=self.config.verbose
            )
            self.adaptive_clusterer = AdaptiveClusteringFramework(clustering_config)
    
    def complete_ml_discovery(
        self,
        market_data: pd.DataFrame,
        target: Optional[np.ndarray] = None,
        price_columns: Optional[List[str]] = None,
        volume_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Complete ML-enhanced market discovery pipeline."""
        
        self.logger.info("🚀 Starting Complete ML-Enhanced Market Discovery")
        start_time = time.time()
        
        results = {
            'pipeline_stages': {},
            'performance_summary': {},
            'recommendations': {},
            'metadata': {
                'start_time': start_time,
                'config': self.config
            }
        }
        
        try:
            # Stage 1: Traditional Analysis (Baseline)
            self.logger.info("📊 Stage 1: Traditional Market Analysis")
            traditional_results = self._traditional_analysis(market_data)
            results['pipeline_stages']['traditional'] = traditional_results
            
            # Stage 2: Automated Feature Engineering
            if self.config.enable_automated_features:
                self.logger.info("🔧 Stage 2: Automated Feature Engineering")
                feature_results = self._automated_feature_engineering(
                    market_data, target, price_columns, volume_columns
                )
                results['pipeline_stages']['feature_engineering'] = feature_results
                enhanced_data = feature_results['enhanced_features']
            else:
                enhanced_data = market_data
            
            # Stage 3: ML-Enhanced Discovery
            if self.config.enable_ml_discovery:
                self.logger.info("🧠 Stage 3: ML-Enhanced Dimension Discovery")
                ml_discovery_results = self._ml_enhanced_discovery(enhanced_data, target)
                results['pipeline_stages']['ml_discovery'] = ml_discovery_results
            
            # Stage 4: Adaptive Clustering
            if self.config.enable_adaptive_clustering:
                self.logger.info("🎯 Stage 4: Adaptive Clustering")
                clustering_results = self._adaptive_clustering(enhanced_data)
                results['pipeline_stages']['adaptive_clustering'] = clustering_results
                
                # Use best clustering result for further analysis
                best_labels = clustering_results.get('best_labels')
                if best_labels is not None:
                    # Stage 5: Enhanced Validation
                    self.logger.info("✅ Stage 5: Enhanced Validation")
                    validation_results = self._enhanced_validation(enhanced_data, best_labels)
                    results['pipeline_stages']['validation'] = validation_results
                    
                    # Stage 6: Regime Transition Prediction
                    if self.config.enable_transition_prediction:
                        self.logger.info("🔮 Stage 6: Regime Transition Prediction")
                        transition_results = self._transition_prediction(enhanced_data, best_labels)
                        results['pipeline_stages']['transition_prediction'] = transition_results
            
            # Stage 7: Performance Summary and Recommendations
            self.logger.info("📈 Stage 7: Performance Analysis")
            performance_summary = self._analyze_performance(results)
            results['performance_summary'] = performance_summary
            
            recommendations = self._generate_recommendations(results)
            results['recommendations'] = recommendations
            
            # Metadata
            results['metadata']['end_time'] = time.time()
            results['metadata']['total_duration'] = results['metadata']['end_time'] - start_time
            results['metadata']['success'] = True
            
            self.logger.info(f"✅ Complete ML Discovery finished in {results['metadata']['total_duration']:.2f}s")
            
            # Save results if configured
            if self.config.save_results:
                self._save_results(results)
            
        except Exception as e:
            self.logger.error(f"❌ ML Discovery pipeline failed: {e}")
            results['metadata']['error'] = str(e)
            results['metadata']['success'] = False
        
        return results
    
    def _traditional_analysis(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Traditional market analysis using existing framework."""
        
        try:
            # Dimension analysis
            dimension_results = self.dimension_analyzer.analyze_all_dimensions(market_data)
            
            # Basic clustering
            clustering_results = self.regime_clusterer.run_all_methods(
                market_data.select_dtypes(include=[np.number]).fillna(0).values
            )
            
            # Feature importance (if we have regime labels)
            best_method, best_result = self.regime_clusterer.get_best_method()
            if best_result:
                importance_results = self.feature_importance.analyze_all_methods(
                    market_data, best_result.labels
                )
            else:
                importance_results = {}
            
            return {
                'dimension_analysis': dimension_results,
                'clustering_results': clustering_results,
                'feature_importance': importance_results,
                'success': True
            }
        
        except Exception as e:
            self.logger.warning(f"Traditional analysis failed: {e}")
            return {'error': str(e), 'success': False}
    
    def _automated_feature_engineering(
        self,
        market_data: pd.DataFrame,
        target: Optional[np.ndarray],
        price_columns: Optional[List[str]],
        volume_columns: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Automated feature engineering."""
        
        try:
            # Convert method names to enums
            methods = []
            for method_name in self.config.feature_engineering_methods:
                try:
                    methods.append(FeatureEngineeringMethod(method_name))
                except ValueError:
                    self.logger.warning(f"Unknown feature engineering method: {method_name}")
            
            # Apply feature engineering
            enhanced_features, metadata = self.feature_engineer.engineer_all_features(
                market_data, target, methods, price_columns, volume_columns
            )
            
            # Evaluate feature set if target is available
            if target is not None:
                evaluation = self.feature_engineer.evaluate_feature_set(enhanced_features, target)
                metadata['evaluation'] = evaluation
            
            return {
                'enhanced_features': enhanced_features,
                'original_feature_count': market_data.shape[1],
                'enhanced_feature_count': enhanced_features.shape[1],
                'metadata': metadata,
                'success': True
            }
        
        except Exception as e:
            self.logger.warning(f"Automated feature engineering failed: {e}")
            return {
                'enhanced_features': market_data,
                'error': str(e),
                'success': False
            }
    
    def _ml_enhanced_discovery(
        self,
        enhanced_data: pd.DataFrame,
        target: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """ML-enhanced dimension discovery."""
        
        try:
            ml_results = {}
            
            # Apply each ML discovery method
            for method_name in self.config.ml_discovery_methods:
                try:
                    method = MLDiscoveryMethod(method_name)
                    result = self.ml_discovery.discover_implicit_dimensions(enhanced_data, method)
                    ml_results[method_name] = result
                    
                    self.logger.info(f"✅ {method_name} discovery completed")
                    
                except ValueError:
                    self.logger.warning(f"Unknown ML discovery method: {method_name}")
                except Exception as e:
                    self.logger.warning(f"ML discovery method {method_name} failed: {e}")
                    ml_results[method_name] = {'error': str(e)}
            
            # Ensemble discovery if multiple methods succeeded
            successful_methods = [k for k, v in ml_results.items() if 'error' not in v]
            if len(successful_methods) > 1:
                try:
                    ensemble_result = self.ml_discovery.discover_implicit_dimensions(
                        enhanced_data, MLDiscoveryMethod.ENSEMBLE_DISCOVERY
                    )
                    ml_results['ensemble'] = ensemble_result
                except Exception as e:
                    self.logger.warning(f"Ensemble discovery failed: {e}")
            
            return {
                'ml_results': ml_results,
                'successful_methods': successful_methods,
                'n_successful_methods': len(successful_methods),
                'success': len(successful_methods) > 0
            }
        
        except Exception as e:
            self.logger.warning(f"ML-enhanced discovery failed: {e}")
            return {'error': str(e), 'success': False}
    
    def _adaptive_clustering(self, enhanced_data: pd.DataFrame) -> Dict[str, Any]:
        """Adaptive clustering with multiple methods."""
        
        try:
            # Convert method names to enums
            methods = []
            for method_name in self.config.adaptive_methods:
                try:
                    methods.append(AdaptiveMethod(method_name))
                except ValueError:
                    self.logger.warning(f"Unknown adaptive method: {method_name}")
            
            # Prepare data
            X = enhanced_data.select_dtypes(include=[np.number]).fillna(0).values
            
            # Compare adaptive methods
            comparison_results = self.adaptive_clusterer.compare_methods(X, methods)
            
            # Extract best result
            best_method = comparison_results.get('best_method')
            best_labels = None
            
            if best_method and best_method in comparison_results:
                best_labels = comparison_results[best_method].get('labels')
            
            return {
                'comparison_results': comparison_results,
                'best_method': best_method,
                'best_labels': best_labels,
                'success': best_labels is not None
            }
        
        except Exception as e:
            self.logger.warning(f"Adaptive clustering failed: {e}")
            return {'error': str(e), 'success': False}
    
    def _enhanced_validation(
        self,
        enhanced_data: pd.DataFrame,
        regime_labels: np.ndarray
    ) -> Dict[str, Any]:
        """Enhanced validation of discovered regimes."""
        
        try:
            # Traditional validation metrics
            validation_results = self.validator.validate_all_metrics(enhanced_data, regime_labels)
            
            # Additional ML-based validation
            ml_validation = self._ml_validation(enhanced_data, regime_labels)
            
            return {
                'traditional_validation': validation_results,
                'ml_validation': ml_validation,
                'success': True
            }
        
        except Exception as e:
            self.logger.warning(f"Enhanced validation failed: {e}")
            return {'error': str(e), 'success': False}
    
    def _ml_validation(
        self,
        enhanced_data: pd.DataFrame,
        regime_labels: np.ndarray
    ) -> Dict[str, Any]:
        """ML-based validation metrics."""
        
        try:
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
            
            X = enhanced_data.select_dtypes(include=[np.number]).fillna(0).values
            
            # Clustering quality metrics
            silhouette = silhouette_score(X, regime_labels)
            calinski_harabasz = calinski_harabasz_score(X, regime_labels)
            davies_bouldin = davies_bouldin_score(X, regime_labels)
            
            # Predictability of regimes
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            predictability_scores = cross_val_score(rf, X, regime_labels, cv=5)
            
            return {
                'silhouette_score': float(silhouette),
                'calinski_harabasz_score': float(calinski_harabasz),
                'davies_bouldin_score': float(davies_bouldin),
                'regime_predictability': {
                    'mean_accuracy': float(np.mean(predictability_scores)),
                    'std_accuracy': float(np.std(predictability_scores)),
                    'scores': predictability_scores.tolist()
                },
                'n_regimes': len(np.unique(regime_labels)),
                'regime_distribution': np.bincount(regime_labels).tolist()
            }
        
        except Exception as e:
            return {'error': str(e)}
    
    def _transition_prediction(
        self,
        enhanced_data: pd.DataFrame,
        regime_labels: np.ndarray
    ) -> Dict[str, Any]:
        """Regime transition prediction."""
        
        try:
            if hasattr(self.ml_discovery, 'predict_regime_transitions'):
                transition_results = self.ml_discovery.predict_regime_transitions(
                    enhanced_data, regime_labels
                )
                return {
                    'transition_results': transition_results,
                    'success': 'error' not in transition_results
                }
            else:
                return {'error': 'Transition prediction not available', 'success': False}
        
        except Exception as e:
            self.logger.warning(f"Transition prediction failed: {e}")
            return {'error': str(e), 'success': False}
    
    def _analyze_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall performance of the ML discovery pipeline."""
        
        performance = {
            'stage_success_rates': {},
            'feature_enhancement': {},
            'clustering_quality': {},
            'overall_score': 0.0
        }
        
        try:
            # Stage success rates
            stages = results.get('pipeline_stages', {})
            for stage_name, stage_result in stages.items():
                performance['stage_success_rates'][stage_name] = stage_result.get('success', False)
            
            # Feature enhancement analysis
            if 'feature_engineering' in stages:
                fe_results = stages['feature_engineering']
                original_count = fe_results.get('original_feature_count', 0)
                enhanced_count = fe_results.get('enhanced_feature_count', 0)
                
                performance['feature_enhancement'] = {
                    'original_features': original_count,
                    'enhanced_features': enhanced_count,
                    'enhancement_ratio': enhanced_count / original_count if original_count > 0 else 1.0,
                    'net_new_features': enhanced_count - original_count
                }
            
            # Clustering quality analysis
            if 'validation' in stages:
                val_results = stages['validation']
                ml_val = val_results.get('ml_validation', {})
                
                performance['clustering_quality'] = {
                    'silhouette_score': ml_val.get('silhouette_score', 0),
                    'regime_predictability': ml_val.get('regime_predictability', {}).get('mean_accuracy', 0),
                    'n_regimes': ml_val.get('n_regimes', 0)
                }
            
            # Calculate overall score
            success_rate = np.mean(list(performance['stage_success_rates'].values()))
            clustering_quality = performance.get('clustering_quality', {}).get('silhouette_score', 0)
            enhancement_ratio = performance.get('feature_enhancement', {}).get('enhancement_ratio', 1.0)
            
            performance['overall_score'] = (
                0.4 * success_rate +
                0.4 * max(0, clustering_quality) +
                0.2 * min(1.0, enhancement_ratio / 2.0)  # Normalize enhancement ratio
            )
        
        except Exception as e:
            self.logger.warning(f"Performance analysis failed: {e}")
            performance['error'] = str(e)
        
        return performance
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations based on discovery results."""
        
        recommendations = {
            'regime_modeling': 'unknown',
            'feature_strategy': 'unknown',
            'clustering_approach': 'unknown',
            'confidence_level': 'low',
            'action_items': []
        }
        
        try:
            performance = results.get('performance_summary', {})
            stages = results.get('pipeline_stages', {})
            
            # Overall performance assessment
            overall_score = performance.get('overall_score', 0)
            
            if overall_score > 0.7:
                recommendations['confidence_level'] = 'high'
            elif overall_score > 0.5:
                recommendations['confidence_level'] = 'medium'
            else:
                recommendations['confidence_level'] = 'low'
            
            # Regime modeling recommendation
            clustering_quality = performance.get('clustering_quality', {})
            silhouette_score = clustering_quality.get('silhouette_score', 0)
            n_regimes = clustering_quality.get('n_regimes', 0)
            
            if silhouette_score > 0.3 and n_regimes >= 2:
                recommendations['regime_modeling'] = 'train_separate_models'
                recommendations['action_items'].append(
                    f"Train separate ML models for {n_regimes} discovered regimes"
                )
            elif silhouette_score > 0.1:
                recommendations['regime_modeling'] = 'regime_aware_features'
                recommendations['action_items'].append(
                    "Use regime indicators as additional features in single model"
                )
            else:
                recommendations['regime_modeling'] = 'single_model'
                recommendations['action_items'].append(
                    "Use single model approach - regime separation insufficient"
                )
            
            # Feature strategy recommendation
            feature_enhancement = performance.get('feature_enhancement', {})
            enhancement_ratio = feature_enhancement.get('enhancement_ratio', 1.0)
            
            if enhancement_ratio > 3.0:
                recommendations['feature_strategy'] = 'feature_selection_needed'
                recommendations['action_items'].append(
                    "Apply feature selection - too many features generated"
                )
            elif enhancement_ratio > 1.5:
                recommendations['feature_strategy'] = 'enhanced_features_beneficial'
                recommendations['action_items'].append(
                    "Use enhanced feature set for improved performance"
                )
            else:
                recommendations['feature_strategy'] = 'original_features_sufficient'
                recommendations['action_items'].append(
                    "Original features appear sufficient - focus on other improvements"
                )
            
            # Clustering approach recommendation
            adaptive_results = stages.get('adaptive_clustering', {})
            best_method = adaptive_results.get('best_method')
            
            if best_method:
                recommendations['clustering_approach'] = best_method
                recommendations['action_items'].append(
                    f"Use {best_method} clustering for optimal regime discovery"
                )
            else:
                recommendations['clustering_approach'] = 'traditional_methods'
                recommendations['action_items'].append(
                    "Fallback to traditional clustering methods"
                )
            
            # ML discovery insights
            ml_discovery = stages.get('ml_discovery', {})
            successful_methods = ml_discovery.get('successful_methods', [])
            
            if successful_methods:
                recommendations['action_items'].append(
                    f"ML discovery successful with {len(successful_methods)} methods: {', '.join(successful_methods)}"
                )
            
        except Exception as e:
            self.logger.warning(f"Recommendation generation failed: {e}")
            recommendations['error'] = str(e)
        
        return recommendations
    
    def _save_results(self, results: Dict[str, Any]):
        """Save results to disk."""
        
        try:
            results_dir = Path(self.config.results_dir)
            results_dir.mkdir(exist_ok=True)
            
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            results_file = results_dir / f"ml_discovery_results_{timestamp}.json"
            
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            json_results = convert_numpy(results)
            
            with open(results_file, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
            
            self.logger.info(f"Results saved to {results_file}")
        
        except Exception as e:
            self.logger.warning(f"Failed to save results: {e}")
    
    def quick_discovery(
        self,
        market_data: pd.DataFrame,
        target: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Quick ML discovery with minimal configuration."""
        
        # Configure for speed
        quick_config = MLIntegrationConfig(
            ml_discovery_methods=["autoencoder"],
            feature_engineering_methods=["time_series_features"],
            adaptive_methods=["multi_criteria"],
            ml_epochs=20,
            max_features=100,
            verbose=False
        )
        
        # Create temporary framework
        quick_framework = MLIntegrationFramework(quick_config)
        
        return quick_framework.complete_ml_discovery(market_data, target)
    
    def get_feature_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract feature insights from discovery results."""
        
        insights = {
            'most_important_features': [],
            'discovered_dimensions': [],
            'feature_categories': {},
            'recommendations': []
        }
        
        try:
            stages = results.get('pipeline_stages', {})
            
            # Feature importance from traditional analysis
            traditional = stages.get('traditional', {})
            feature_importance = traditional.get('feature_importance', {})
            
            if feature_importance:
                # Extract top features (simplified)
                insights['most_important_features'] = ["feature_analysis_available"]
            
            # ML discovered dimensions
            ml_discovery = stages.get('ml_discovery', {})
            ml_results = ml_discovery.get('ml_results', {})
            
            for method, result in ml_results.items():
                if 'dimension_analysis' in result:
                    insights['discovered_dimensions'].append(f"{method}_dimensions")
            
            # Feature engineering results
            feature_eng = stages.get('feature_engineering', {})
            if feature_eng.get('success'):
                metadata = feature_eng.get('metadata', {})
                method_details = metadata.get('method_details', {})
                
                for method, details in method_details.items():
                    if 'error' not in details:
                        insights['feature_categories'][method] = 'successful'
        
        except Exception as e:
            insights['error'] = str(e)
        
        return insights


# Example usage and integration
def demonstrate_ml_integration():
    """Demonstrate the ML integration framework."""
    
    # Generate sample market data
    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Simulate market data with regime structure
    regime_changes = [250, 500, 750]
    data = {
        'price': 100 + np.cumsum(np.random.randn(n_samples) * 0.02),
        'volume': np.random.lognormal(10, 0.5, n_samples),
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples)
    }
    
    # Add regime-specific patterns
    for i, change_point in enumerate(regime_changes):
        start = change_point if i == 0 else regime_changes[i-1]
        end = regime_changes[i+1] if i < len(regime_changes) - 1 else n_samples
        
        if i % 2 == 0:
            data['feature_1'][start:end] += 1.5  # High momentum regime
        else:
            data['feature_2'][start:end] += 1.0  # High volatility regime
    
    market_data = pd.DataFrame(data, index=dates)
    
    # Create target (future returns)
    target = (market_data['price'].shift(-5) / market_data['price'] - 1).fillna(0)
    target_binary = (target > target.median()).astype(int)
    
    # Initialize framework
    config = MLIntegrationConfig(
        ml_epochs=20,  # Reduced for demo
        max_features=200,
        verbose=True,
        save_results=False
    )
    
    framework = MLIntegrationFramework(config)
    
    # Run complete discovery
    results = framework.complete_ml_discovery(
        market_data.iloc[:-5],  # Remove last 5 rows for target
        target_binary.iloc[:-5].values,
        price_columns=['price'],
        volume_columns=['volume']
    )
    
    print("🎯 ML Integration Framework Results:")
    print(f"Pipeline Success: {results['metadata']['success']}")
    print(f"Total Duration: {results['metadata']['total_duration']:.2f}s")
    
    # Performance summary
    performance = results.get('performance_summary', {})
    print(f"Overall Score: {performance.get('overall_score', 0):.3f}")
    
    # Recommendations
    recommendations = results.get('recommendations', {})
    print(f"Regime Modeling: {recommendations.get('regime_modeling', 'unknown')}")
    print(f"Confidence Level: {recommendations.get('confidence_level', 'unknown')}")
    
    # Action items
    action_items = recommendations.get('action_items', [])
    if action_items:
        print("\nAction Items:")
        for i, item in enumerate(action_items[:3], 1):
            print(f"{i}. {item}")
    
    return results


if __name__ == "__main__":
    # Run demonstration
    demo_results = demonstrate_ml_integration()
    
    # Quick discovery example
    print("\n" + "="*50)
    print("🚀 Quick Discovery Example:")
    
    # Generate smaller dataset for quick demo
    np.random.seed(42)
    quick_data = pd.DataFrame({
        'price': 100 + np.cumsum(np.random.randn(200) * 0.02),
        'volume': np.random.lognormal(8, 0.3, 200),
        'momentum': np.random.randn(200)
    })
    
    framework = MLIntegrationFramework()
    quick_results = framework.quick_discovery(quick_data)
    
    print(f"Quick Discovery Success: {quick_results['metadata']['success']}")
    print(f"Quick Discovery Duration: {quick_results['metadata']['total_duration']:.2f}s")