"""
Unified Regime Detection Example

This example demonstrates how to use the enhanced unified utilities
with both TAS and NAS systems for comprehensive regime detection and analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import TAS integration
from ...tas_regime.integration.tas_unified_integration import (
    TASUnifiedIntegration, TASUnifiedConfig, create_tas_unified_integration
)

# Import NAS integration
from ...nas_regime.integration.nas_unified_integration import (
    NASUnifiedIntegration, NASUnifiedConfig, create_nas_unified_integration
)

# Import unified utilities
from ..shared_utils import (
    EconomicEvaluationConfig, TradingViabilityConfig,
    OptimizationConfig, RegimeAnalysisConfig,
    create_unified_economic_evaluator,
    create_unified_trading_viability_evaluator,
    create_unified_multi_objective_optimizer,
    create_unified_regime_analyzer
)

logger = logging.getLogger(__name__)


class UnifiedRegimeDetectionExample:
    """
    Unified Regime Detection Example.
    
    Demonstrates comprehensive regime detection using both TAS and NAS systems
    with enhanced unified utilities.
    """
    
    def __init__(self):
        """Initialize the example."""
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize TAS integration
        self.tas_integration = create_tas_unified_integration()
        
        # Initialize NAS integration
        self.nas_integration = create_nas_unified_integration()
        
        self.logger.info("✅ Unified Regime Detection Example initialized")
    
    def run_comprehensive_example(self, 
                                market_data: Union[pd.DataFrame, np.ndarray],
                                comparison_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run comprehensive regime detection example.
        
        Args:
            market_data: Market data (OHLCV)
            comparison_config: Optional comparison configuration
            
        Returns:
            Comprehensive comparison results
        """
        try:
            self.logger.info("🚀 Starting comprehensive regime detection example...")
            start_time = time.time()
            
            # Run TAS analysis
            tas_results = self._run_tas_analysis(market_data)
            
            # Run NAS analysis
            nas_results = self._run_nas_analysis(market_data)
            
            # Compare results
            comparison_results = self._compare_results(tas_results, nas_results)
            
            # Generate insights
            insights = self._generate_insights(tas_results, nas_results, comparison_results)
            
            execution_time = time.time() - start_time
            
            results = {
                'success': True,
                'execution_time': execution_time,
                'tas_results': tas_results,
                'nas_results': nas_results,
                'comparison_results': comparison_results,
                'insights': insights,
                'timestamp': datetime.now()
            }
            
            self.logger.info(f"✅ Comprehensive example completed in {execution_time:.2f}s")
            self.logger.info(f"   TAS economic significance: {tas_results.get('economic_significance', {}).get('overall_score', 0.0):.3f}")
            self.logger.info(f"   NAS economic significance: {nas_results.get('economic_significance', {}).get('overall_score', 0.0):.3f}")
            self.logger.info(f"   TAS trading viability: {tas_results.get('trading_viability', {}).get('overall_score', 0.0):.3f}")
            self.logger.info(f"   NAS trading viability: {nas_results.get('trading_viability', {}).get('overall_score', 0.0):.3f}")
            
            return results
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Comprehensive example failed: {e}")
            
            return {
                'success': False,
                'execution_time': execution_time,
                'error': str(e)
            }
    
    def _run_tas_analysis(self, market_data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """Run TAS analysis."""
        try:
            self.logger.info("🌳 Running TAS analysis...")
            
            # Run TAS search and evaluation
            tas_results = self.tas_integration.search_and_evaluate(market_data)
            
            if not tas_results['success']:
                self.logger.error("TAS analysis failed")
                return {'error': 'TAS analysis failed'}
            
            # Extract evaluation results
            evaluation_results = tas_results.get('evaluation_results', {})
            
            return {
                'architecture_type': 'TAS',
                'success': True,
                'economic_significance': evaluation_results.get('economic_significance'),
                'trading_viability': evaluation_results.get('trading_viability'),
                'regime_analysis': evaluation_results.get('regime_analysis'),
                'multi_objective_optimization': evaluation_results.get('multi_objective_optimization'),
                'model_metadata': tas_results.get('model_metadata', {}),
                'execution_time': tas_results.get('execution_time', 0.0)
            }
            
        except Exception as e:
            self.logger.error(f"TAS analysis failed: {e}")
            return {'error': str(e)}
    
    def _run_nas_analysis(self, market_data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """Run NAS analysis."""
        try:
            self.logger.info("🧠 Running NAS analysis...")
            
            # Run NAS search and evaluation
            nas_results = self.nas_integration.search_and_evaluate(market_data)
            
            if not nas_results['success']:
                self.logger.error("NAS analysis failed")
                return {'error': 'NAS analysis failed'}
            
            # Extract evaluation results
            evaluation_results = nas_results.get('evaluation_results', {})
            
            return {
                'architecture_type': 'NAS',
                'success': True,
                'economic_significance': evaluation_results.get('economic_significance'),
                'trading_viability': evaluation_results.get('trading_viability'),
                'regime_analysis': evaluation_results.get('regime_analysis'),
                'multi_objective_optimization': evaluation_results.get('multi_objective_optimization'),
                'model_metadata': nas_results.get('model_metadata', {}),
                'execution_time': nas_results.get('execution_time', 0.0)
            }
            
        except Exception as e:
            self.logger.error(f"NAS analysis failed: {e}")
            return {'error': str(e)}
    
    def _compare_results(self, tas_results: Dict[str, Any], nas_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare TAS and NAS results."""
        try:
            self.logger.info("📊 Comparing results...")
            
            comparison = {
                'economic_significance': self._compare_economic_significance(tas_results, nas_results),
                'trading_viability': self._compare_trading_viability(tas_results, nas_results),
                'regime_analysis': self._compare_regime_analysis(tas_results, nas_results),
                'performance_metrics': self._compare_performance_metrics(tas_results, nas_results)
            }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Result comparison failed: {e}")
            return {'error': str(e)}
    
    def _compare_economic_significance(self, tas_results: Dict[str, Any], nas_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare economic significance results."""
        try:
            tas_economic = tas_results.get('economic_significance', {})
            nas_economic = nas_results.get('economic_significance', {})
            
            tas_score = tas_economic.get('overall_score', 0.0)
            nas_score = nas_economic.get('overall_score', 0.0)
            
            return {
                'tas_score': tas_score,
                'nas_score': nas_score,
                'difference': abs(tas_score - nas_score),
                'winner': 'TAS' if tas_score > nas_score else 'NAS',
                'tas_significance_level': tas_economic.get('significance_level', 'unknown'),
                'nas_significance_level': nas_economic.get('significance_level', 'unknown')
            }
            
        except Exception as e:
            self.logger.warning(f"Economic significance comparison failed: {e}")
            return {'error': str(e)}
    
    def _compare_trading_viability(self, tas_results: Dict[str, Any], nas_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare trading viability results."""
        try:
            tas_trading = tas_results.get('trading_viability', {})
            nas_trading = nas_results.get('trading_viability', {})
            
            tas_score = tas_trading.get('overall_score', 0.0)
            nas_score = nas_trading.get('overall_score', 0.0)
            
            return {
                'tas_score': tas_score,
                'nas_score': nas_score,
                'difference': abs(tas_score - nas_score),
                'winner': 'TAS' if tas_score > nas_score else 'NAS',
                'tas_viability_level': tas_trading.get('viability_level', 'unknown'),
                'nas_viability_level': nas_trading.get('viability_level', 'unknown')
            }
            
        except Exception as e:
            self.logger.warning(f"Trading viability comparison failed: {e}")
            return {'error': str(e)}
    
    def _compare_regime_analysis(self, tas_results: Dict[str, Any], nas_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare regime analysis results."""
        try:
            tas_regime = tas_results.get('regime_analysis', {})
            nas_regime = nas_results.get('regime_analysis', {})
            
            tas_stability = tas_regime.get('overall_stability', 0.0)
            nas_stability = nas_regime.get('overall_stability', 0.0)
            
            return {
                'tas_stability': tas_stability,
                'nas_stability': nas_stability,
                'difference': abs(tas_stability - nas_stability),
                'winner': 'TAS' if tas_stability > nas_stability else 'NAS'
            }
            
        except Exception as e:
            self.logger.warning(f"Regime analysis comparison failed: {e}")
            return {'error': str(e)}
    
    def _compare_performance_metrics(self, tas_results: Dict[str, Any], nas_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare performance metrics."""
        try:
            tas_time = tas_results.get('execution_time', 0.0)
            nas_time = nas_results.get('execution_time', 0.0)
            
            return {
                'tas_execution_time': tas_time,
                'nas_execution_time': nas_time,
                'time_difference': abs(tas_time - nas_time),
                'faster': 'TAS' if tas_time < nas_time else 'NAS',
                'speedup': max(tas_time, nas_time) / min(tas_time, nas_time) if min(tas_time, nas_time) > 0 else 1.0
            }
            
        except Exception as e:
            self.logger.warning(f"Performance metrics comparison failed: {e}")
            return {'error': str(e)}
    
    def _generate_insights(self, tas_results: Dict[str, Any], nas_results: Dict[str, Any], 
                          comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from the analysis."""
        try:
            insights = {
                'recommendations': [],
                'key_findings': [],
                'architecture_preferences': {},
                'use_cases': {}
            }
            
            # Economic significance insights
            economic_comparison = comparison_results.get('economic_significance', {})
            if economic_comparison.get('winner') == 'TAS':
                insights['recommendations'].append("TAS shows better economic significance for this dataset")
                insights['use_cases']['economic_focus'] = 'TAS'
            else:
                insights['recommendations'].append("NAS shows better economic significance for this dataset")
                insights['use_cases']['economic_focus'] = 'NAS'
            
            # Trading viability insights
            trading_comparison = comparison_results.get('trading_viability', {})
            if trading_comparison.get('winner') == 'TAS':
                insights['recommendations'].append("TAS shows better trading viability for this dataset")
                insights['use_cases']['trading_focus'] = 'TAS'
            else:
                insights['recommendations'].append("NAS shows better trading viability for this dataset")
                insights['use_cases']['trading_focus'] = 'NAS'
            
            # Regime stability insights
            regime_comparison = comparison_results.get('regime_analysis', {})
            if regime_comparison.get('winner') == 'TAS':
                insights['recommendations'].append("TAS shows better regime stability for this dataset")
                insights['use_cases']['stability_focus'] = 'TAS'
            else:
                insights['recommendations'].append("NAS shows better regime stability for this dataset")
                insights['use_cases']['stability_focus'] = 'NAS'
            
            # Performance insights
            performance_comparison = comparison_results.get('performance_metrics', {})
            if performance_comparison.get('faster') == 'TAS':
                insights['key_findings'].append("TAS is faster for this dataset")
                insights['use_cases']['speed_focus'] = 'TAS'
            else:
                insights['key_findings'].append("NAS is faster for this dataset")
                insights['use_cases']['speed_focus'] = 'NAS'
            
            # Overall recommendations
            tas_wins = sum(1 for key, value in insights['use_cases'].items() if value == 'TAS')
            nas_wins = sum(1 for key, value in insights['use_cases'].items() if value == 'NAS')
            
            if tas_wins > nas_wins:
                insights['architecture_preferences']['overall'] = 'TAS'
                insights['recommendations'].append("Overall, TAS appears to be better suited for this dataset")
            elif nas_wins > tas_wins:
                insights['architecture_preferences']['overall'] = 'NAS'
                insights['recommendations'].append("Overall, NAS appears to be better suited for this dataset")
            else:
                insights['architecture_preferences']['overall'] = 'HYBRID'
                insights['recommendations'].append("Consider using a hybrid approach combining both TAS and NAS")
            
            return insights
            
        except Exception as e:
            self.logger.warning(f"Insight generation failed: {e}")
            return {'error': str(e)}
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive report."""
        try:
            report = []
            report.append("# Unified Regime Detection Report")
            report.append(f"Generated at: {results.get('timestamp', 'Unknown')}")
            report.append(f"Execution time: {results.get('execution_time', 0.0):.2f} seconds")
            report.append("")
            
            # TAS Results
            tas_results = results.get('tas_results', {})
            report.append("## TAS Results")
            if tas_results.get('success'):
                economic = tas_results.get('economic_significance', {})
                trading = tas_results.get('trading_viability', {})
                regime = tas_results.get('regime_analysis', {})
                
                report.append(f"- Economic Significance: {economic.get('overall_score', 0.0):.3f} ({economic.get('significance_level', 'unknown')})")
                report.append(f"- Trading Viability: {trading.get('overall_score', 0.0):.3f} ({trading.get('viability_level', 'unknown')})")
                report.append(f"- Regime Stability: {regime.get('overall_stability', 0.0):.3f}")
            else:
                report.append("- TAS analysis failed")
            report.append("")
            
            # NAS Results
            nas_results = results.get('nas_results', {})
            report.append("## NAS Results")
            if nas_results.get('success'):
                economic = nas_results.get('economic_significance', {})
                trading = nas_results.get('trading_viability', {})
                regime = nas_results.get('regime_analysis', {})
                
                report.append(f"- Economic Significance: {economic.get('overall_score', 0.0):.3f} ({economic.get('significance_level', 'unknown')})")
                report.append(f"- Trading Viability: {trading.get('overall_score', 0.0):.3f} ({trading.get('viability_level', 'unknown')})")
                report.append(f"- Regime Stability: {regime.get('overall_stability', 0.0):.3f}")
            else:
                report.append("- NAS analysis failed")
            report.append("")
            
            # Comparison Results
            comparison = results.get('comparison_results', {})
            report.append("## Comparison Results")
            
            economic_comp = comparison.get('economic_significance', {})
            report.append(f"- Economic Significance Winner: {economic_comp.get('winner', 'Unknown')}")
            report.append(f"  - TAS: {economic_comp.get('tas_score', 0.0):.3f}")
            report.append(f"  - NAS: {economic_comp.get('nas_score', 0.0):.3f}")
            
            trading_comp = comparison.get('trading_viability', {})
            report.append(f"- Trading Viability Winner: {trading_comp.get('winner', 'Unknown')}")
            report.append(f"  - TAS: {trading_comp.get('tas_score', 0.0):.3f}")
            report.append(f"  - NAS: {trading_comp.get('nas_score', 0.0):.3f}")
            
            regime_comp = comparison.get('regime_analysis', {})
            report.append(f"- Regime Stability Winner: {regime_comp.get('winner', 'Unknown')}")
            report.append(f"  - TAS: {regime_comp.get('tas_stability', 0.0):.3f}")
            report.append(f"  - NAS: {regime_comp.get('nas_stability', 0.0):.3f}")
            report.append("")
            
            # Insights
            insights = results.get('insights', {})
            report.append("## Insights and Recommendations")
            
            recommendations = insights.get('recommendations', [])
            for rec in recommendations:
                report.append(f"- {rec}")
            
            key_findings = insights.get('key_findings', [])
            for finding in key_findings:
                report.append(f"- {finding}")
            
            overall_preference = insights.get('architecture_preferences', {}).get('overall', 'Unknown')
            report.append(f"- Overall Recommendation: {overall_preference}")
            
            return "\n".join(report)
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            return f"Report generation failed: {e}"


# Convenience functions
def run_unified_regime_detection_example(market_data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
    """Run the unified regime detection example."""
    example = UnifiedRegimeDetectionExample()
    return example.run_comprehensive_example(market_data)


def generate_sample_market_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate sample market data for testing."""
    np.random.seed(42)
    
    # Generate price data with regime changes
    prices = []
    volumes = []
    current_price = 100.0
    
    for i in range(n_samples):
        # Simulate regime changes
        if i % 200 == 0:
            regime = i // 200 % 3  # 3 regimes
        else:
            regime = i // 200 % 3
        
        # Different volatility for different regimes
        if regime == 0:
            volatility = 0.01
            trend = 0.0001
        elif regime == 1:
            volatility = 0.02
            trend = -0.0001
        else:
            volatility = 0.015
            trend = 0.00005
        
        # Generate price movement
        price_change = np.random.normal(trend, volatility)
        current_price *= (1 + price_change)
        
        # Generate OHLCV data
        open_price = current_price
        high_price = open_price * (1 + abs(np.random.normal(0, volatility/2)))
        low_price = open_price * (1 - abs(np.random.normal(0, volatility/2)))
        close_price = current_price
        volume = np.random.exponential(1000)
        
        prices.append([open_price, high_price, low_price, close_price, volume])
        volumes.append(volume)
    
    # Create DataFrame
    df = pd.DataFrame(prices, columns=['open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.date_range(start='2020-01-01', periods=n_samples, freq='1H')
    
    return df


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data
    market_data = generate_sample_market_data(1000)
    
    # Run example
    example = UnifiedRegimeDetectionExample()
    results = example.run_comprehensive_example(market_data)
    
    # Generate report
    report = example.generate_report(results)
    print(report)