"""
Advanced Markov Models Integration Example

This example demonstrates how to integrate the new advanced Markov models
(Markov-Switching and Hidden Semi-Markov) with the existing HMM framework
and clustering research infrastructure.

Key Integration Points:
1. Seamless integration with existing HMMIntegrationLayer
2. Compatibility with clustering validation metrics
3. Enhanced regime discovery pipeline
4. Economic validation and constraints
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import asyncio
from pathlib import Path

# Import existing framework components
from .integration_layer import HMMIntegrationLayer, IntegrationConfig, IntegrationMethod
from .advanced_markov_models import AdvancedMarkovIntegration, MarkovSwitchingConfig, SemiMarkovConfig
from .validation_metrics import RegimeValidationMetrics, ValidationConfig
from .dimension_analyzer import MarketDimensionAnalyzer, DimensionAnalysisConfig

from src.utils.logger import system_logger


class EnhancedMarkovIntegrationLayer:
    """
    Enhanced integration layer that combines traditional HMMs, clustering,
    and advanced Markov models for comprehensive regime analysis.
    """
    
    def __init__(self, config: Optional[IntegrationConfig] = None):
        self.config = config or IntegrationConfig()
        self.logger = system_logger.getChild('EnhancedMarkovIntegration')
        
        # Initialize components
        self.hmm_integration = HMMIntegrationLayer(config)
        self.advanced_markov = AdvancedMarkovIntegration()
        self.validation_metrics = RegimeValidationMetrics(ValidationConfig())
        self.dimension_analyzer = MarketDimensionAnalyzer(DimensionAnalysisConfig())
        
    async def run_comprehensive_regime_analysis(self, 
                                              market_data: pd.DataFrame,
                                              include_advanced_models: bool = True,
                                              validate_economic_constraints: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive regime analysis using all available methods.
        
        Args:
            market_data: Market data for analysis
            include_advanced_models: Whether to include advanced Markov models
            validate_economic_constraints: Whether to validate economic constraints
            
        Returns:
            Comprehensive analysis results
        """
        self.logger.info("🚀 Starting Comprehensive Regime Analysis")
        
        results = {
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'data_shape': market_data.shape,
            'methods_used': [],
            'performance_comparison': {},
            'recommendations': []
        }
        
        try:
            # Step 1: Run traditional HMM + clustering analysis
            self.logger.info("📊 Running traditional HMM + clustering analysis")
            traditional_results = await self.hmm_integration.run_integration_analysis(market_data)
            results['traditional_analysis'] = traditional_results.to_dict()
            results['methods_used'].extend(['hmm', 'clustering'])
            
            # Step 2: Run advanced Markov models if requested
            if include_advanced_models:
                self.logger.info("🔬 Running advanced Markov models")
                advanced_results = self.advanced_markov.run_advanced_regime_analysis(
                    market_data,
                    include_markov_switching=True,
                    include_semi_markov=True
                )
                results['advanced_analysis'] = advanced_results
                results['methods_used'].extend(advanced_results['models_run'])
            
            # Step 3: Cross-validate all methods
            self.logger.info("✅ Cross-validating regime detection methods")
            cross_validation = await self._cross_validate_methods(market_data, results)
            results['cross_validation'] = cross_validation
            
            # Step 4: Economic constraint validation
            if validate_economic_constraints:
                self.logger.info("💰 Validating economic constraints")
                economic_validation = self._validate_economic_constraints(results)
                results['economic_validation'] = economic_validation
            
            # Step 5: Performance comparison
            self.logger.info("📈 Comparing method performance")
            performance_comparison = self._compare_method_performance(results)
            results['performance_comparison'] = performance_comparison
            
            # Step 6: Generate integrated recommendations
            recommendations = self._generate_integrated_recommendations(results)
            results['recommendations'] = recommendations
            
            self.logger.info(f"✅ Comprehensive analysis completed using {len(results['methods_used'])} methods")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive regime analysis failed: {e}")
            raise
    
    async def _cross_validate_methods(self, 
                                    market_data: pd.DataFrame,
                                    results: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate different regime detection methods."""
        
        cross_validation = {
            'method_agreement': {},
            'regime_stability': {},
            'temporal_consistency': {}
        }
        
        # Collect regime assignments from different methods
        regime_assignments = {}
        
        # Traditional HMM
        if 'traditional_analysis' in results and results['traditional_analysis'].get('hmm_results'):
            hmm_regimes = results['traditional_analysis']['hmm_results'].get('regime_discovery', {}).get('regime_assignments', [])
            if hmm_regimes:
                regime_assignments['hmm'] = np.array(hmm_regimes)
        
        # Clustering
        if 'traditional_analysis' in results and results['traditional_analysis'].get('clustering_results'):
            clustering_regimes = results['traditional_analysis']['clustering_results'].get('regime_labels')
            if clustering_regimes is not None:
                regime_assignments['clustering'] = np.array(clustering_regimes)
        
        # Advanced Markov models
        if 'advanced_analysis' in results:
            if 'markov_switching' in results['advanced_analysis']:
                ms_regimes = results['advanced_analysis']['markov_switching'].get('regime_assignments')
                if ms_regimes is not None:
                    regime_assignments['markov_switching'] = np.array(ms_regimes)
            
            if 'hidden_semi_markov' in results['advanced_analysis']:
                hsmm_states = results['advanced_analysis']['hidden_semi_markov'].get('state_sequence')
                if hsmm_states is not None:
                    regime_assignments['hidden_semi_markov'] = np.array(hsmm_states)
        
        # Calculate pairwise agreement metrics
        if len(regime_assignments) > 1:
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
            
            methods = list(regime_assignments.keys())
            for i, method1 in enumerate(methods):
                for j, method2 in enumerate(methods[i+1:], i+1):
                    # Align lengths if necessary
                    min_len = min(len(regime_assignments[method1]), len(regime_assignments[method2]))
                    regimes1 = regime_assignments[method1][:min_len]
                    regimes2 = regime_assignments[method2][:min_len]
                    
                    agreement_key = f"{method1}_vs_{method2}"
                    cross_validation['method_agreement'][agreement_key] = {
                        'adjusted_rand_score': float(adjusted_rand_score(regimes1, regimes2)),
                        'normalized_mutual_info': float(normalized_mutual_info_score(regimes1, regimes2)),
                        'n_regimes_method1': len(np.unique(regimes1)),
                        'n_regimes_method2': len(np.unique(regimes2))
                    }
        
        # Temporal consistency analysis
        for method, regimes in regime_assignments.items():
            if len(regimes) > 1:
                # Calculate regime transition frequency
                transitions = np.sum(np.diff(regimes) != 0)
                transition_rate = transitions / len(regimes)
                
                # Calculate average regime duration
                regime_changes = np.diff(regimes.astype(int))
                regime_starts = np.where(regime_changes != 0)[0] + 1
                regime_starts = np.concatenate([[0], regime_starts])
                regime_ends = np.where(regime_changes != 0)[0] + 1
                regime_ends = np.concatenate([regime_ends, [len(regimes)]])
                
                durations = regime_ends - regime_starts
                
                cross_validation['temporal_consistency'][method] = {
                    'transition_rate': float(transition_rate),
                    'avg_regime_duration': float(np.mean(durations)),
                    'median_regime_duration': float(np.median(durations)),
                    'regime_duration_std': float(np.std(durations)),
                    'n_regime_episodes': len(durations)
                }
        
        return cross_validation
    
    def _validate_economic_constraints(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate regime results against economic constraints."""
        
        economic_validation = {
            'constraint_violations': [],
            'economic_plausibility': {},
            'regime_characteristics': {}
        }
        
        # Check traditional analysis
        if 'traditional_analysis' in results:
            traditional_validation = results['traditional_analysis'].get('validation_metrics', {})
            if traditional_validation:
                economic_validation['traditional_hmm'] = traditional_validation
        
        # Check advanced analysis
        if 'advanced_analysis' in results:
            # Markov-Switching validation
            if 'markov_switching' in results['advanced_analysis']:
                ms_validation = results['advanced_analysis']['markov_switching'].get('economic_validation', {})
                economic_validation['markov_switching'] = ms_validation
                
                if not ms_validation.get('constraints_satisfied', True):
                    economic_validation['constraint_violations'].extend(
                        ms_validation.get('violations', [])
                    )
            
            # Hidden Semi-Markov validation
            if 'hidden_semi_markov' in results['advanced_analysis']:
                hsmm_validation = results['advanced_analysis']['hidden_semi_markov'].get('duration_validation', {})
                economic_validation['hidden_semi_markov'] = hsmm_validation
                
                if not hsmm_validation.get('constraints_satisfied', True):
                    economic_validation['constraint_violations'].extend(
                        hsmm_validation.get('violations', [])
                    )
        
        # Overall economic plausibility assessment
        total_violations = len(economic_validation['constraint_violations'])
        if total_violations == 0:
            economic_validation['overall_plausibility'] = 'high'
        elif total_violations <= 2:
            economic_validation['overall_plausibility'] = 'medium'
        else:
            economic_validation['overall_plausibility'] = 'low'
        
        return economic_validation
    
    def _compare_method_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare performance of different regime detection methods."""
        
        performance_comparison = {
            'method_rankings': {},
            'best_method_overall': None,
            'method_strengths': {},
            'computational_efficiency': {}
        }
        
        # Collect performance metrics from different methods
        method_scores = {}
        
        # Traditional methods
        if 'traditional_analysis' in results:
            # HMM performance
            hmm_validation = results['traditional_analysis'].get('validation_metrics', {}).get('hmm', {})
            if hmm_validation:
                silhouette = hmm_validation.get('silhouette_score', {}).get('value', 0)
                temporal_consistency = hmm_validation.get('temporal_consistency', {}).get('value', 0)
                method_scores['hmm'] = {
                    'silhouette_score': silhouette,
                    'temporal_consistency': temporal_consistency,
                    'composite_score': 0.6 * silhouette + 0.4 * temporal_consistency
                }
            
            # Clustering performance
            clustering_validation = results['traditional_analysis'].get('validation_metrics', {}).get('clustering', {})
            if clustering_validation:
                silhouette = clustering_validation.get('silhouette_score', {}).get('value', 0)
                temporal_consistency = clustering_validation.get('temporal_consistency', {}).get('value', 0)
                method_scores['clustering'] = {
                    'silhouette_score': silhouette,
                    'temporal_consistency': temporal_consistency,
                    'composite_score': 0.6 * silhouette + 0.4 * temporal_consistency
                }
        
        # Advanced methods
        if 'advanced_analysis' in results:
            # Markov-Switching performance
            if 'markov_switching' in results['advanced_analysis']:
                ms_stats = results['advanced_analysis']['markov_switching'].get('regime_statistics', {})
                economic_validation = results['advanced_analysis']['markov_switching'].get('economic_validation', {})
                
                economic_score = 1.0 if economic_validation.get('constraints_satisfied', False) else 0.5
                regime_quality = len(ms_stats) * 0.1  # More regimes = potentially better granularity
                
                method_scores['markov_switching'] = {
                    'economic_plausibility': economic_score,
                    'regime_quality': min(1.0, regime_quality),
                    'composite_score': 0.7 * economic_score + 0.3 * min(1.0, regime_quality)
                }
            
            # Hidden Semi-Markov performance
            if 'hidden_semi_markov' in results['advanced_analysis']:
                hsmm_stats = results['advanced_analysis']['hidden_semi_markov'].get('state_statistics', {})
                duration_validation = results['advanced_analysis']['hidden_semi_markov'].get('duration_validation', {})
                
                duration_score = 1.0 if duration_validation.get('constraints_satisfied', False) else 0.5
                state_quality = len(hsmm_stats) * 0.1
                
                method_scores['hidden_semi_markov'] = {
                    'duration_plausibility': duration_score,
                    'state_quality': min(1.0, state_quality),
                    'composite_score': 0.7 * duration_score + 0.3 * min(1.0, state_quality)
                }
        
        # Rank methods by composite score
        if method_scores:
            sorted_methods = sorted(
                method_scores.items(),
                key=lambda x: x[1]['composite_score'],
                reverse=True
            )
            
            performance_comparison['method_rankings'] = {
                method: {'rank': rank + 1, **scores}
                for rank, (method, scores) in enumerate(sorted_methods)
            }
            
            performance_comparison['best_method_overall'] = sorted_methods[0][0]
        
        # Method strengths analysis
        performance_comparison['method_strengths'] = {
            'hmm': ['Established methodology', 'Good for sequential data', 'Probabilistic framework'],
            'clustering': ['Non-temporal patterns', 'Multiple algorithms', 'Dimensional analysis'],
            'markov_switching': ['Economic realism', 'Regime-dependent parameters', 'Structural breaks'],
            'hidden_semi_markov': ['Flexible durations', 'Realistic persistence', 'Better transitions']
        }
        
        return performance_comparison
    
    def _generate_integrated_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate integrated recommendations based on all analysis results."""
        
        recommendations = []
        
        # Performance-based recommendations
        best_method = results.get('performance_comparison', {}).get('best_method_overall')
        if best_method:
            recommendations.append(f"🏆 Best performing method: {best_method.upper()}")
        
        # Cross-validation recommendations
        cross_val = results.get('cross_validation', {})
        if cross_val.get('method_agreement'):
            agreements = cross_val['method_agreement']
            high_agreement_pairs = [
                pair for pair, metrics in agreements.items()
                if metrics['adjusted_rand_score'] > 0.5
            ]
            
            if high_agreement_pairs:
                recommendations.append(f"✅ High agreement found between: {', '.join(high_agreement_pairs)}")
            else:
                recommendations.append("⚠️ Low agreement between methods - consider ensemble approach")
        
        # Economic validation recommendations
        economic_val = results.get('economic_validation', {})
        if economic_val.get('overall_plausibility') == 'high':
            recommendations.append("💰 Regimes show high economic plausibility")
        elif economic_val.get('overall_plausibility') == 'low':
            recommendations.append("🔧 Consider parameter tuning for better economic realism")
        
        # Method-specific recommendations
        if 'advanced_analysis' in results:
            if 'markov_switching' in results['advanced_analysis']:
                recommendations.append("📊 Markov-Switching models provide regime-dependent parameters")
            
            if 'hidden_semi_markov' in results['advanced_analysis']:
                recommendations.append("⏱️ Hidden Semi-Markov models capture realistic regime durations")
        
        # Temporal consistency recommendations
        temporal_consistency = cross_val.get('temporal_consistency', {})
        unstable_methods = [
            method for method, metrics in temporal_consistency.items()
            if metrics.get('transition_rate', 0) > 0.1  # More than 10% transitions
        ]
        
        if unstable_methods:
            recommendations.append(f"⚠️ High regime instability in: {', '.join(unstable_methods)}")
        
        # Integration recommendations
        if len(results.get('methods_used', [])) > 2:
            recommendations.append("🔄 Consider ensemble of best-performing methods for robustness")
        
        if not recommendations:
            recommendations.append("📝 Complete analysis to generate specific recommendations")
        
        return recommendations


# Example usage and demonstration
async def demonstrate_enhanced_integration():
    """Demonstrate the enhanced Markov integration capabilities."""
    
    logger = system_logger.getChild('EnhancedIntegrationDemo')
    logger.info("🧪 Demonstrating Enhanced Markov Integration")
    
    # Generate synthetic market data with known regime structure
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    n_obs = len(dates)
    
    # Create realistic regime-switching market data
    true_regimes = np.zeros(n_obs, dtype=int)
    prices = np.zeros(n_obs)
    prices[0] = 100.0
    
    # Define realistic market regimes
    regime_periods = [
        (0, 300, 0),      # Bull market (low vol, positive returns)
        (300, 450, 1),    # Bear market (high vol, negative returns)
        (450, 500, 2),    # Crisis (extreme vol, mixed returns)
        (500, 800, 0),    # Recovery bull market
        (800, 950, 3),    # Consolidation (low vol, sideways)
        (950, n_obs, 0)   # Final bull phase
    ]
    
    for start, end, regime in regime_periods:
        true_regimes[start:end] = regime
        
        # Generate regime-specific returns
        if regime == 0:  # Bull market
            returns = np.random.normal(0.0008, 0.012, end - start)
        elif regime == 1:  # Bear market
            returns = np.random.normal(-0.0015, 0.025, end - start)
        elif regime == 2:  # Crisis
            returns = np.random.normal(-0.001, 0.045, end - start)
        else:  # Consolidation
            returns = np.random.normal(0.0002, 0.008, end - start)
        
        # Apply returns to prices
        for i, ret in enumerate(returns):
            if start + i < n_obs - 1:
                prices[start + i + 1] = prices[start + i] * (1 + ret)
    
    # Create comprehensive market data
    market_data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.normal(0, 0.001, n_obs)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.003, n_obs))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.003, n_obs))),
        'close': prices,
        'volume': np.random.lognormal(15, 0.3, n_obs)
    })
    
    logger.info(f"📊 Generated synthetic data: {len(market_data)} observations")
    logger.info(f"🎯 True regime distribution: {np.bincount(true_regimes)}")
    
    # Run comprehensive analysis
    enhanced_integration = EnhancedMarkovIntegrationLayer()
    
    try:
        results = await enhanced_integration.run_comprehensive_regime_analysis(
            market_data,
            include_advanced_models=True,
            validate_economic_constraints=True
        )
        
        logger.info("✅ Enhanced integration analysis completed")
        logger.info(f"📈 Methods used: {results['methods_used']}")
        logger.info(f"🏆 Best method: {results['performance_comparison'].get('best_method_overall', 'N/A')}")
        
        print("\n💡 Key Recommendations:")
        for rec in results['recommendations']:
            print(f"  {rec}")
        
        # Display method agreement
        if 'cross_validation' in results:
            print("\n🤝 Method Agreement Analysis:")
            agreements = results['cross_validation'].get('method_agreement', {})
            for pair, metrics in agreements.items():
                ari = metrics['adjusted_rand_score']
                nmi = metrics['normalized_mutual_info']
                print(f"  {pair}: ARI={ari:.3f}, NMI={nmi:.3f}")
        
        # Display performance rankings
        if 'performance_comparison' in results:
            print("\n🏅 Method Performance Rankings:")
            rankings = results['performance_comparison'].get('method_rankings', {})
            for method, data in sorted(rankings.items(), key=lambda x: x[1]['rank']):
                print(f"  {data['rank']}. {method.upper()}: {data['composite_score']:.3f}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Enhanced integration failed: {e}")
        raise


if __name__ == "__main__":
    # Run the demonstration
    import asyncio
    
    print("🚀 Enhanced Markov Integration Demonstration")
    print("=" * 60)
    
    results = asyncio.run(demonstrate_enhanced_integration())
    
    print("\n📊 Analysis Summary:")
    print(f"  Data points: {results['data_shape'][0]:,}")
    print(f"  Methods tested: {len(results['methods_used'])}")
    print(f"  Economic validation: {results.get('economic_validation', {}).get('overall_plausibility', 'N/A')}")
    
    print("\n✨ Enhanced integration demonstration completed!")