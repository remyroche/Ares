"""
Economic Validation Example for Data-Driven Clustering

This example demonstrates the comprehensive economic validation system
that integrates financial performance signals, volatility-aware clustering,
and advanced risk/distributional dimensions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
import time
from pathlib import Path

# Import economic validation components
from ..optimization.economic_validator import EconomicValidator, EconomicValidationConfig
from ..optimization.multi_objective_optimizer import MultiObjectiveOptimizer, MultiObjectiveConfig
from ..validation.regime_persistence_validator import RegimePersistenceValidator, RegimePersistenceConfig
from ..feature_engineering.advanced_financial_features import AdvancedFinancialFeatureEngineer, AdvancedFeatureConfig

# Import clustering components
from ..optimization.data_driven_clustering_optimizer import DataDrivenClusteringOptimizer
from ..config.data_driven_config import DataDrivenClusteringConfig

logger = logging.getLogger(__name__)

class EconomicValidationExample:
    """
    Example demonstrating comprehensive economic validation for clustering.
    
    This class shows how to integrate financial performance signals,
    volatility-aware clustering, and advanced risk dimensions into
    the data-driven clustering optimization process.
    """
    
    def __init__(self, config: Optional[DataDrivenClusteringConfig] = None):
        """Initialize the economic validation example."""
        self.config = config or DataDrivenClusteringConfig()
        self.results = {}
        
    def run_complete_economic_validation(self, 
                                       market_data: pd.DataFrame,
                                       features: np.ndarray,
                                       feature_names: List[str]) -> Dict[str, Any]:
        """
        Run complete economic validation example.
        
        Args:
            market_data: Market data for analysis
            features: Feature matrix
            feature_names: List of feature names
            
        Returns:
            Dictionary with comprehensive validation results
        """
        try:
            logger.info("🚀 Starting complete economic validation example...")
            
            results = {
                'start_time': time.time(),
                'steps_completed': [],
                'validation_results': {},
                'recommendations': {},
                'performance_metrics': {}
            }
            
            # Step 1: Advanced feature engineering
            logger.info("🔧 Step 1: Advanced feature engineering...")
            advanced_features_result = self._run_advanced_feature_engineering(market_data)
            results['steps_completed'].append('advanced_features')
            results['validation_results']['advanced_features'] = advanced_features_result
            
            # Step 2: Economic validation
            logger.info("💰 Step 2: Economic validation...")
            economic_validation_result = self._run_economic_validation(features, feature_names, market_data)
            results['steps_completed'].append('economic_validation')
            results['validation_results']['economic_validation'] = economic_validation_result
            
            # Step 3: Regime persistence validation
            logger.info("⏱️ Step 3: Regime persistence validation...")
            regime_persistence_result = self._run_regime_persistence_validation(features, feature_names, market_data)
            results['steps_completed'].append('regime_persistence')
            results['validation_results']['regime_persistence'] = regime_persistence_result
            
            # Step 4: Multi-objective optimization
            logger.info("🎯 Step 4: Multi-objective optimization...")
            multi_objective_result = self._run_multi_objective_optimization(features, feature_names, market_data)
            results['steps_completed'].append('multi_objective')
            results['validation_results']['multi_objective'] = multi_objective_result
            
            # Step 5: Generate recommendations
            logger.info("💡 Step 5: Generating recommendations...")
            recommendations = self._generate_economic_recommendations(results['validation_results'])
            results['recommendations'] = recommendations
            
            # Step 6: Calculate performance metrics
            logger.info("📈 Step 6: Calculating performance metrics...")
            performance_metrics = self._calculate_economic_performance_metrics(results)
            results['performance_metrics'] = performance_metrics
            
            results['end_time'] = time.time()
            results['total_time'] = results['end_time'] - results['start_time']
            
            logger.info(f"✅ Complete economic validation example finished in {results['total_time']:.2f} seconds")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Complete economic validation example failed: {e}")
            raise
    
    def _run_advanced_feature_engineering(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Run advanced feature engineering."""
        try:
            # Create advanced feature engineer
            feature_config = AdvancedFeatureConfig(
                enable_skewness_features=True,
                enable_kurtosis_features=True,
                enable_var_features=True,
                enable_cvar_features=True,
                enable_drawdown_features=True,
                enable_volatility_regimes=True,
                enable_volume_features=True,
                enable_technical_indicators=True
            )
            
            engineer = AdvancedFinancialFeatureEngineer(feature_config)
            
            # Engineer features
            features_array, feature_names, feature_categories = engineer.engineer_features(market_data)
            
            return {
                'features_array': features_array,
                'feature_names': feature_names,
                'feature_categories': feature_categories,
                'n_features': features_array.shape[1] if len(features_array.shape) > 1 else 0,
                'category_breakdown': {cat: len(features) for cat, features in feature_categories.items()},
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Advanced feature engineering failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _run_economic_validation(self, 
                               features: np.ndarray,
                               feature_names: List[str],
                               market_data: pd.DataFrame) -> Dict[str, Any]:
        """Run economic validation."""
        try:
            # Create economic validator
            economic_config = EconomicValidationConfig(
                enable_return_separation=True,
                enable_volatility_discrimination=True,
                enable_risk_metrics=True,
                enable_drawdown_metrics=True,
                enable_volume_metrics=True,
                enable_strategy_backtest=True,
                enable_statistical_tests=True
            )
            
            validator = EconomicValidator(economic_config)
            
            # Create sample cluster labels for validation
            from sklearn.cluster import KMeans
            n_clusters = min(5, features.shape[0] // 10)
            if n_clusters < 2:
                n_clusters = 2
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            
            # Perform economic validation
            economic_result = validator.validate_clustering(
                cluster_labels, market_data, features, feature_names
            )
            
            return {
                'overall_economic_score': economic_result.overall_economic_score,
                'return_separation_score': economic_result.return_separation_score,
                'volatility_discrimination_score': economic_result.volatility_discrimination_score,
                'risk_discrimination_score': economic_result.risk_discrimination_score,
                'drawdown_discrimination_score': economic_result.drawdown_discrimination_score,
                'volume_discrimination_score': economic_result.volume_discrimination_score,
                'strategy_performance_score': economic_result.strategy_performance_score,
                'validation_time': economic_result.validation_time,
                'n_clusters': economic_result.n_clusters,
                'n_samples': economic_result.n_samples,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Economic validation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _run_regime_persistence_validation(self, 
                                         features: np.ndarray,
                                         feature_names: List[str],
                                         market_data: pd.DataFrame) -> Dict[str, Any]:
        """Run regime persistence validation."""
        try:
            # Create regime persistence validator
            persistence_config = RegimePersistenceConfig(
                enable_lifespan_analysis=True,
                enable_transition_analysis=True,
                enable_economic_coherence=True,
                enable_volatility_persistence=True,
                enable_statistical_tests=True
            )
            
            validator = RegimePersistenceValidator(persistence_config)
            
            # Create sample cluster labels for validation
            from sklearn.cluster import KMeans
            n_clusters = min(5, features.shape[0] // 10)
            if n_clusters < 2:
                n_clusters = 2
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            
            # Perform persistence validation
            persistence_result = validator.validate_persistence(
                cluster_labels, market_data, features, feature_names
            )
            
            return {
                'overall_persistence_score': persistence_result.overall_persistence_score,
                'lifespan_score': persistence_result.lifespan_score,
                'transition_score': persistence_result.transition_score,
                'economic_coherence_score': persistence_result.economic_coherence_score,
                'volatility_persistence_score': persistence_result.volatility_persistence_score,
                'n_regimes': persistence_result.n_regimes,
                'n_transitions': persistence_result.n_transitions,
                'avg_regime_lifespan': persistence_result.avg_regime_lifespan,
                'validation_time': persistence_result.validation_time,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Regime persistence validation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _run_multi_objective_optimization(self, 
                                        features: np.ndarray,
                                        feature_names: List[str],
                                        market_data: pd.DataFrame) -> Dict[str, Any]:
        """Run multi-objective optimization."""
        try:
            # Create multi-objective optimizer
            multi_obj_config = MultiObjectiveConfig(
                enable_economic_validation=True,
                optimization_strategy='weighted_sum',
                max_iterations=50,
                population_size=20
            )
            
            optimizer = MultiObjectiveOptimizer(multi_obj_config)
            
            # Define parameter ranges
            parameter_ranges = {
                'similarity_threshold': (0.5, 0.95),
                'distance_threshold': (0.1, 0.5),
                'window_size': (50, 500),
                'smoothing_window': (3, 20)
            }
            
            # Create clustering function
            def clustering_func(features):
                from sklearn.cluster import KMeans
                n_clusters = min(5, features.shape[0] // 10)
                if n_clusters < 2:
                    n_clusters = 2
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                return kmeans.fit_predict(features)
            
            # Perform multi-objective optimization
            multi_obj_result = optimizer.optimize_parameters(
                parameter_ranges=parameter_ranges,
                clustering_func=clustering_func,
                market_data=market_data,
                features=features,
                feature_names=feature_names
            )
            
            return {
                'optimal_parameters': multi_obj_result.get('optimal_parameters', {}),
                'overall_score': multi_obj_result.get('overall_score', 0),
                'detailed_scores': multi_obj_result.get('detailed_scores', {}),
                'optimization_success': multi_obj_result.get('optimization_success', False),
                'convergence_info': multi_obj_result.get('convergence_info', {}),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Multi-objective optimization failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_economic_recommendations(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate economic recommendations based on validation results."""
        try:
            recommendations = {
                'feature_engineering': {},
                'economic_validation': {},
                'regime_persistence': {},
                'multi_objective': {},
                'overall': {}
            }
            
            # Advanced feature engineering recommendations
            if 'advanced_features' in validation_results:
                af_results = validation_results['advanced_features']
                if af_results.get('success', False):
                    recommendations['feature_engineering'] = {
                        'n_features_engineered': af_results.get('n_features', 0),
                        'category_breakdown': af_results.get('category_breakdown', {}),
                        'insights': [
                            f"Engineered {af_results.get('n_features', 0)} advanced financial features",
                            "Features include risk dimensions, volatility regimes, and volume analysis",
                            "Consider using these features for improved regime discovery"
                        ],
                        'implementation_notes': [
                            "Integrate advanced features into existing feature preparation pipeline",
                            "Use feature categories for targeted PCA weighting",
                            "Monitor feature importance for regime discrimination"
                        ]
                    }
            
            # Economic validation recommendations
            if 'economic_validation' in validation_results:
                ev_results = validation_results['economic_validation']
                if ev_results.get('success', False):
                    overall_score = ev_results.get('overall_economic_score', 0)
                    recommendations['economic_validation'] = {
                        'overall_score': overall_score,
                        'score_breakdown': {
                            'return_separation': ev_results.get('return_separation_score', 0),
                            'volatility_discrimination': ev_results.get('volatility_discrimination_score', 0),
                            'risk_discrimination': ev_results.get('risk_discrimination_score', 0),
                            'drawdown_discrimination': ev_results.get('drawdown_discrimination_score', 0),
                            'volume_discrimination': ev_results.get('volume_discrimination_score', 0),
                            'strategy_performance': ev_results.get('strategy_performance_score', 0)
                        },
                        'insights': self._analyze_economic_validation_scores(ev_results),
                        'implementation_notes': [
                            "Use economic validation scores to guide parameter optimization",
                            "Focus on improving low-scoring economic dimensions",
                            "Consider economic metrics as primary optimization objectives"
                        ]
                    }
            
            # Regime persistence recommendations
            if 'regime_persistence' in validation_results:
                rp_results = validation_results['regime_persistence']
                if rp_results.get('success', False):
                    recommendations['regime_persistence'] = {
                        'overall_score': rp_results.get('overall_persistence_score', 0),
                        'score_breakdown': {
                            'lifespan': rp_results.get('lifespan_score', 0),
                            'transition': rp_results.get('transition_score', 0),
                            'economic_coherence': rp_results.get('economic_coherence_score', 0),
                            'volatility_persistence': rp_results.get('volatility_persistence_score', 0)
                        },
                        'regime_statistics': {
                            'n_regimes': rp_results.get('n_regimes', 0),
                            'n_transitions': rp_results.get('n_transitions', 0),
                            'avg_lifespan': rp_results.get('avg_regime_lifespan', 0)
                        },
                        'insights': self._analyze_regime_persistence_scores(rp_results),
                        'implementation_notes': [
                            "Use persistence metrics to validate regime stability",
                            "Adjust temporal smoothing based on persistence scores",
                            "Monitor regime transitions for economic significance"
                        ]
                    }
            
            # Multi-objective optimization recommendations
            if 'multi_objective' in validation_results:
                mo_results = validation_results['multi_objective']
                if mo_results.get('success', False):
                    recommendations['multi_objective'] = {
                        'optimal_parameters': mo_results.get('optimal_parameters', {}),
                        'overall_score': mo_results.get('overall_score', 0),
                        'optimization_success': mo_results.get('optimization_success', False),
                        'insights': self._analyze_multi_objective_results(mo_results),
                        'implementation_notes': [
                            "Use optimized parameters in production clustering pipeline",
                            "Balance clustering quality with economic performance",
                            "Monitor parameter performance over time"
                        ]
                    }
            
            # Overall recommendations
            recommendations['overall'] = {
                'summary': "Economic validation completed successfully",
                'next_steps': [
                    "Integrate economic validation into production pipeline",
                    "Set up continuous monitoring of economic metrics",
                    "Implement adaptive parameter updates based on economic performance",
                    "Consider economic validation as primary success metric"
                ],
                'benefits': [
                    "Improved clustering quality through economic validation",
                    "Better adaptation to market conditions",
                    "Enhanced regime discovery with financial significance",
                    "Reduced reliance on purely statistical metrics"
                ]
            }
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return {'error': str(e)}
    
    def _analyze_economic_validation_scores(self, ev_results: Dict[str, Any]) -> List[str]:
        """Analyze economic validation scores."""
        insights = []
        
        overall_score = ev_results.get('overall_economic_score', 0)
        if overall_score > 0.7:
            insights.append("Excellent economic validation - clusters show strong financial significance")
        elif overall_score > 0.5:
            insights.append("Good economic validation - clusters show moderate financial significance")
        else:
            insights.append("Poor economic validation - clusters lack financial significance")
        
        # Analyze individual scores
        return_score = ev_results.get('return_separation_score', 0)
        if return_score > 0.6:
            insights.append("Strong return separation between clusters")
        else:
            insights.append("Weak return separation - consider improving feature selection")
        
        vol_score = ev_results.get('volatility_discrimination_score', 0)
        if vol_score > 0.6:
            insights.append("Good volatility discrimination between clusters")
        else:
            insights.append("Poor volatility discrimination - consider volatility-aware features")
        
        return insights
    
    def _analyze_regime_persistence_scores(self, rp_results: Dict[str, Any]) -> List[str]:
        """Analyze regime persistence scores."""
        insights = []
        
        overall_score = rp_results.get('overall_persistence_score', 0)
        if overall_score > 0.7:
            insights.append("Excellent regime persistence - regimes are stable and economically coherent")
        elif overall_score > 0.5:
            insights.append("Good regime persistence - regimes show moderate stability")
        else:
            insights.append("Poor regime persistence - regimes are unstable or lack economic coherence")
        
        # Analyze regime statistics
        n_regimes = rp_results.get('n_regimes', 0)
        avg_lifespan = rp_results.get('avg_regime_lifespan', 0)
        
        if n_regimes > 10:
            insights.append(f"High number of regimes ({n_regimes}) - consider merging similar regimes")
        elif n_regimes < 3:
            insights.append(f"Low number of regimes ({n_regimes}) - consider allowing more regime diversity")
        
        if avg_lifespan > 100:
            insights.append("Very long average regime lifespan - may indicate over-smoothing")
        elif avg_lifespan < 10:
            insights.append("Very short average regime lifespan - may indicate excessive regime changes")
        
        return insights
    
    def _analyze_multi_objective_results(self, mo_results: Dict[str, Any]) -> List[str]:
        """Analyze multi-objective optimization results."""
        insights = []
        
        overall_score = mo_results.get('overall_score', 0)
        if overall_score > 0.7:
            insights.append("Excellent multi-objective optimization - good balance of clustering quality and economic performance")
        elif overall_score > 0.5:
            insights.append("Good multi-objective optimization - reasonable balance of objectives")
        else:
            insights.append("Poor multi-objective optimization - consider adjusting objective weights")
        
        optimization_success = mo_results.get('optimization_success', False)
        if optimization_success:
            insights.append("Optimization converged successfully")
        else:
            insights.append("Optimization failed to converge - consider adjusting parameters or increasing iterations")
        
        optimal_params = mo_results.get('optimal_parameters', {})
        if optimal_params:
            insights.append(f"Optimal parameters found: {optimal_params}")
        
        return insights
    
    def _calculate_economic_performance_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate economic performance metrics."""
        try:
            metrics = {
                'total_validation_time': results.get('total_time', 0.0),
                'steps_completed': len(results.get('steps_completed', [])),
                'success_rate': 0.0,
                'average_economic_score': 0.0,
                'validation_breakdown': {}
            }
            
            # Calculate success rate
            validation_results = results.get('validation_results', {})
            successful_validations = 0
            total_validations = len(validation_results)
            
            economic_scores = []
            for step, step_results in validation_results.items():
                if step_results.get('success', False):
                    successful_validations += 1
                    
                    # Extract economic scores
                    if 'overall_economic_score' in step_results:
                        economic_scores.append(step_results['overall_economic_score'])
                    elif 'overall_persistence_score' in step_results:
                        economic_scores.append(step_results['overall_persistence_score'])
                    elif 'overall_score' in step_results:
                        economic_scores.append(step_results['overall_score'])
                
                # Store validation breakdown
                metrics['validation_breakdown'][step] = {
                    'success': step_results.get('success', False),
                    'score': step_results.get('overall_economic_score', 
                                            step_results.get('overall_persistence_score',
                                                           step_results.get('overall_score', 0))),
                    'time': step_results.get('validation_time', 0)
                }
            
            if total_validations > 0:
                metrics['success_rate'] = successful_validations / total_validations
            
            if economic_scores:
                metrics['average_economic_score'] = np.mean(economic_scores)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Performance metrics calculation failed: {e}")
            return {'error': str(e)}


def create_enhanced_market_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create enhanced market data with additional features for testing."""
    np.random.seed(42)
    
    # Generate synthetic market data
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='1H')
    
    # Generate price data with regime changes
    price = 100
    prices = [price]
    for i in range(1, n_samples):
        # Simulate regime changes
        if i % 200 == 0:  # Regime change every 200 periods
            regime_vol = np.random.uniform(0.01, 0.05)
        else:
            regime_vol = 0.02
        
        # Generate price with regime-specific volatility
        price_change = np.random.normal(0, regime_vol)
        price *= (1 + price_change)
        prices.append(price)
    
    # Generate volume data with regime-specific patterns
    volumes = []
    for i in range(n_samples):
        if i % 200 < 100:  # High volume regime
            volume = np.random.lognormal(10, 0.5)
        else:  # Low volume regime
            volume = np.random.lognormal(9, 0.3)
        volumes.append(volume)
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': volumes
    })
    
    data.set_index('timestamp', inplace=True)
    
    return data


def main():
    """Main function to run the economic validation example."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create enhanced sample data
    logger.info("Creating enhanced sample market data...")
    market_data = create_enhanced_market_data(1000)
    
    # Create sample features
    features, feature_names = create_sample_features(market_data)
    
    # Create example instance
    example = EconomicValidationExample()
    
    # Run complete economic validation example
    logger.info("Running complete economic validation example...")
    results = example.run_complete_economic_validation(market_data, features, feature_names)
    
    # Print results
    logger.info("=" * 80)
    logger.info("ECONOMIC VALIDATION EXAMPLE RESULTS")
    logger.info("=" * 80)
    
    logger.info(f"Total validation time: {results['total_time']:.2f} seconds")
    logger.info(f"Steps completed: {results['steps_completed']}")
    
    # Print validation results
    for step, step_results in results['validation_results'].items():
        logger.info(f"\n{step.upper()} VALIDATION:")
        if step_results.get('success', False):
            if 'overall_economic_score' in step_results:
                logger.info(f"  Economic Score: {step_results['overall_economic_score']:.4f}")
            if 'overall_persistence_score' in step_results:
                logger.info(f"  Persistence Score: {step_results['overall_persistence_score']:.4f}")
            if 'overall_score' in step_results:
                logger.info(f"  Overall Score: {step_results['overall_score']:.4f}")
        else:
            logger.error(f"  Error: {step_results.get('error', 'Unknown error')}")
    
    # Print recommendations
    logger.info("\nRECOMMENDATIONS:")
    for category, recs in results['recommendations'].items():
        if isinstance(recs, dict) and 'insights' in recs:
            logger.info(f"\n{category.upper()}:")
            for insight in recs['insights']:
                logger.info(f"  • {insight}")
    
    # Print performance metrics
    logger.info("\nPERFORMANCE METRICS:")
    metrics = results['performance_metrics']
    logger.info(f"  Success rate: {metrics.get('success_rate', 0):.1%}")
    logger.info(f"  Average economic score: {metrics.get('average_economic_score', 0):.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Economic validation example completed successfully!")
    logger.info("=" * 80)


def create_sample_features(market_data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Create sample features for testing."""
    np.random.seed(42)
    
    n_samples = len(market_data)
    
    # Generate synthetic features
    features = np.random.randn(n_samples, 20)
    feature_names = [
        'return_1h', 'return_4h', 'return_1d',
        'volatility_1h', 'volatility_4h', 'volatility_1d',
        'volume_1h', 'volume_4h', 'volume_1d',
        'rsi_14', 'rsi_21', 'rsi_50',
        'macd', 'macd_signal', 'macd_histogram',
        'bb_upper', 'bb_lower', 'bb_width',
        'atr_14', 'atr_21'
    ]
    
    return features, feature_names


if __name__ == "__main__":
    main()