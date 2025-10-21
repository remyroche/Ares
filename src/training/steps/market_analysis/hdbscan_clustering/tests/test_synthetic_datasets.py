"""
Synthetic Dataset Testing Framework

This module provides comprehensive testing with synthetic datasets to ensure
the data-driven clustering system reproduces previous economic-validation scores within tolerance.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import the clustering system components
from src.training.steps.market_analysis.hdbscan_clustering.optimization.data_driven_clustering_optimizer import (
    DataDrivenClusteringOptimizer
)
from src.training.steps.market_analysis.hdbscan_clustering.config.data_driven_config import (
    DataDrivenClusteringConfig, OptimizationStrategy, ValidationMetric
)
from src.training.steps.market_analysis.hdbscan_clustering.feature_engineering.advanced_financial_features import (
    AdvancedFinancialFeatureEngineer, AdvancedFeatureConfig
)
from src.training.steps.market_analysis.hdbscan_clustering.optimization.economic_validator import (
    EconomicValidator, EconomicValidationConfig
)
from src.training.steps.market_analysis.hdbscan_clustering.validation.regime_persistence_validator import (
    RegimePersistenceValidator, RegimePersistenceConfig
)
from src.training.steps.market_analysis.hdbscan_clustering.optimization.multi_objective_optimizer import (
    MultiObjectiveOptimizer, MultiObjectiveConfig
)

logger = logging.getLogger(__name__)


class SyntheticDatasetTester:
    """
    Comprehensive testing framework using synthetic datasets.
    
    Generates synthetic market data with known regime structures and tests
    the data-driven clustering system's ability to reproduce consistent results.
    """
    
    def __init__(self, 
                 tolerance: float = 0.05,  # 5% tolerance for score differences
                 test_data_dir: str = "synthetic_test_data",
                 baseline_dir: str = "synthetic_baseline"):
        """
        Initialize the synthetic dataset tester.
        
        Args:
            tolerance: Maximum allowed difference in scores (as fraction)
            test_data_dir: Directory containing synthetic test datasets
            baseline_dir: Directory containing baseline scores
        """
        self.tolerance = tolerance
        self.test_data_dir = Path(test_data_dir)
        self.baseline_dir = Path(baseline_dir)
        
        # Create directories if they don't exist
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.config = DataDrivenClusteringConfig()
        self.optimizer = DataDrivenClusteringOptimizer(self.config)
        self.feature_engineer = AdvancedFinancialFeatureEngineer(AdvancedFeatureConfig())
        self.economic_validator = EconomicValidator(EconomicValidationConfig())
        self.persistence_validator = RegimePersistenceValidator(RegimePersistenceConfig())
        self.multi_objective_optimizer = MultiObjectiveOptimizer(MultiObjectiveConfig())
        
        # Test results storage
        self.test_results: Dict[str, Any] = {}
        self.baseline_scores: Dict[str, Any] = {}
        
    def generate_synthetic_market_data(self, 
                                     scenario: str,
                                     n_samples: int = 1000,
                                     n_features: int = 50,
                                     n_regimes: int = 3,
                                     noise_level: float = 0.1,
                                     regime_persistence: float = 0.8,
                                     volatility_regimes: bool = True,
                                     trend_regimes: bool = True,
                                     volume_regimes: bool = True) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
        """
        Generate synthetic market data with known regime structure.
        
        Args:
            scenario: Name of the scenario (e.g., 'bull_market', 'crisis', 'sideways')
            n_samples: Number of samples to generate
            n_features: Number of features
            n_regimes: Number of underlying regimes
            noise_level: Level of noise to add
            regime_persistence: Probability of staying in same regime
            volatility_regimes: Whether to include volatility-based regimes
            trend_regimes: Whether to include trend-based regimes
            volume_regimes: Whether to include volume-based regimes
            
        Returns:
            Tuple of (market_data, true_regime_labels, scenario_metadata)
        """
        np.random.seed(42)  # For reproducibility
        
        # Generate regime labels with persistence
        regime_labels = np.zeros(n_samples, dtype=int)
        current_regime = 0
        
        for i in range(1, n_samples):
            if np.random.random() < regime_persistence:
                regime_labels[i] = current_regime
            else:
                current_regime = (current_regime + 1) % n_regimes
                regime_labels[i] = current_regime
        
        # Generate features based on regime characteristics
        features = np.zeros((n_samples, n_features))
        
        # Define regime characteristics based on scenario
        regime_characteristics = self._get_regime_characteristics(scenario, n_regimes)
        
        for regime in range(n_regimes):
            regime_mask = regime_labels == regime
            n_regime_samples = np.sum(regime_mask)
            
            if n_regime_samples > 0:
                char = regime_characteristics[regime]
                
                # Each regime has different characteristics
                regime_center = np.random.normal(char['mean'], char['std'], n_features)
                regime_scale = np.random.uniform(char['scale_min'], char['scale_max'], n_features)
                
                features[regime_mask] = np.random.normal(
                    regime_center, 
                    regime_scale, 
                    (n_regime_samples, n_features)
                )
        
        # Add noise
        features += np.random.normal(0, noise_level, features.shape)
        
        # Create market data DataFrame
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='1H')
        
        # Generate price data with regime-dependent characteristics
        prices = np.zeros(n_samples)
        prices[0] = 100.0
        
        for i in range(1, n_samples):
            regime = regime_labels[i]
            char = regime_characteristics[regime]
            
            # Different regimes have different return characteristics
            regime_returns = np.random.normal(
                char['return_mean'],  # Different mean returns per regime
                char['return_std'],   # Different volatility per regime
                1
            )
            prices[i] = prices[i-1] * (1 + regime_returns)
        
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        returns = np.concatenate([[0], returns])  # Add initial return
        
        # Create volume data with regime-dependent characteristics
        volumes = np.random.lognormal(
            regime_characteristics[regime_labels]['volume_mean'],
            regime_characteristics[regime_labels]['volume_std'],
            n_samples
        )
        
        # Create market data DataFrame
        market_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
            'close': prices,
            'volume': volumes,
            'returns': returns
        })
        
        # Add additional features
        market_data['volatility'] = market_data['returns'].rolling(20).std()
        market_data['volume_ma'] = market_data['volume'].rolling(20).mean()
        market_data['price_ma'] = market_data['close'].rolling(20).mean()
        
        # Scenario metadata
        scenario_metadata = {
            'scenario': scenario,
            'n_samples': n_samples,
            'n_features': n_features,
            'n_regimes': n_regimes,
            'noise_level': noise_level,
            'regime_persistence': regime_persistence,
            'regime_characteristics': regime_characteristics,
            'volatility_regimes': volatility_regimes,
            'trend_regimes': trend_regimes,
            'volume_regimes': volume_regimes
        }
        
        return market_data, regime_labels, scenario_metadata
    
    def _get_regime_characteristics(self, scenario: str, n_regimes: int) -> List[Dict[str, Any]]:
        """Get regime characteristics based on scenario."""
        if scenario == 'bull_market':
            return [
                {'mean': 0.1, 'std': 0.05, 'scale_min': 0.5, 'scale_max': 1.5, 
                 'return_mean': 0.002, 'return_std': 0.01, 'volume_mean': 5.5, 'volume_std': 0.3},
                {'mean': 0.2, 'std': 0.08, 'scale_min': 0.8, 'scale_max': 2.0, 
                 'return_mean': 0.001, 'return_std': 0.015, 'volume_mean': 5.8, 'volume_std': 0.4},
                {'mean': 0.15, 'std': 0.06, 'scale_min': 0.6, 'scale_max': 1.8, 
                 'return_mean': 0.0015, 'return_std': 0.012, 'volume_mean': 5.6, 'volume_std': 0.35}
            ][:n_regimes]
        
        elif scenario == 'crisis':
            return [
                {'mean': -0.1, 'std': 0.15, 'scale_min': 1.0, 'scale_max': 3.0, 
                 'return_mean': -0.005, 'return_std': 0.03, 'volume_mean': 6.2, 'volume_std': 0.6},
                {'mean': -0.05, 'std': 0.12, 'scale_min': 0.8, 'scale_max': 2.5, 
                 'return_mean': -0.002, 'return_std': 0.025, 'volume_mean': 6.0, 'volume_std': 0.5},
                {'mean': -0.08, 'std': 0.18, 'scale_min': 1.2, 'scale_max': 3.5, 
                 'return_mean': -0.003, 'return_std': 0.035, 'volume_mean': 6.3, 'volume_std': 0.7}
            ][:n_regimes]
        
        elif scenario == 'sideways':
            return [
                {'mean': 0.0, 'std': 0.08, 'scale_min': 0.7, 'scale_max': 1.3, 
                 'return_mean': 0.0001, 'return_std': 0.008, 'volume_mean': 5.2, 'volume_std': 0.2},
                {'mean': 0.02, 'std': 0.06, 'scale_min': 0.6, 'scale_max': 1.2, 
                 'return_mean': 0.0002, 'return_std': 0.006, 'volume_mean': 5.1, 'volume_std': 0.18},
                {'mean': -0.01, 'std': 0.07, 'scale_min': 0.65, 'scale_max': 1.25, 
                 'return_mean': -0.0001, 'return_std': 0.007, 'volume_mean': 5.15, 'volume_std': 0.19}
            ][:n_regimes]
        
        elif scenario == 'high_volatility':
            return [
                {'mean': 0.05, 'std': 0.2, 'scale_min': 1.0, 'scale_max': 3.0, 
                 'return_mean': 0.001, 'return_std': 0.02, 'volume_mean': 6.0, 'volume_std': 0.8},
                {'mean': -0.03, 'std': 0.25, 'scale_min': 1.2, 'scale_max': 3.5, 
                 'return_mean': -0.0005, 'return_std': 0.025, 'volume_mean': 6.2, 'volume_std': 0.9},
                {'mean': 0.01, 'std': 0.22, 'scale_min': 1.1, 'scale_max': 3.2, 
                 'return_mean': 0.0002, 'return_std': 0.022, 'volume_mean': 6.1, 'volume_std': 0.85}
            ][:n_regimes]
        
        else:  # Default scenario
            return [
                {'mean': 0.0, 'std': 0.1, 'scale_min': 0.5, 'scale_max': 2.0, 
                 'return_mean': 0.0, 'return_std': 0.01, 'volume_mean': 5.5, 'volume_std': 0.3},
                {'mean': 0.1, 'std': 0.12, 'scale_min': 0.6, 'scale_max': 2.2, 
                 'return_mean': 0.001, 'return_std': 0.012, 'volume_mean': 5.7, 'volume_std': 0.35},
                {'mean': -0.05, 'std': 0.15, 'scale_min': 0.8, 'scale_max': 2.5, 
                 'return_mean': -0.0005, 'return_std': 0.015, 'volume_mean': 5.8, 'volume_std': 0.4}
            ][:n_regimes]
    
    def run_synthetic_test(self, 
                          test_name: str,
                          scenario: str,
                          market_data: pd.DataFrame,
                          true_regime_labels: np.ndarray,
                          features: np.ndarray,
                          feature_names: List[str]) -> Dict[str, Any]:
        """
        Run synthetic test and return results.
        
        Args:
            test_name: Name of the test
            scenario: Scenario name
            market_data: Market data
            true_regime_labels: True regime labels
            features: Feature matrix
            feature_names: Feature names
            
        Returns:
            Dictionary containing test results
        """
        try:
            # Run data-driven optimization
            optimization_result = self.optimizer.optimize_all_parameters(
                market_data=market_data,
                features=features,
                feature_names=feature_names,
                clustering_func=lambda x: true_regime_labels  # Use true labels for testing
            )
            
            # Extract cluster labels from optimization result
            cluster_labels = optimization_result.optimal_parameters.get('cluster_labels', true_regime_labels)
            
            # Run economic validation
            economic_result = self.economic_validator.validate_clustering(
                cluster_labels=cluster_labels,
                market_data=market_data,
                features=features,
                feature_names=feature_names
            )
            
            # Run regime persistence validation
            persistence_result = self.persistence_validator.validate_persistence(
                cluster_labels=cluster_labels,
                market_data=market_data,
                features=features,
                feature_names=feature_names
            )
            
            # Compile results
            test_results = {
                'test_name': test_name,
                'scenario': scenario,
                'timestamp': datetime.now().isoformat(),
                'optimization': {
                    'success': optimization_result.success,
                    'overall_score': optimization_result.overall_score,
                    'optimal_parameters': optimization_result.optimal_parameters,
                    'optimization_summary': optimization_result.optimization_summary
                },
                'economic_validation': {
                    'overall_economic_score': economic_result.overall_economic_score,
                    'return_separation_score': economic_result.return_separation_score,
                    'volatility_discrimination_score': economic_result.volatility_discrimination_score,
                    'risk_discrimination_score': economic_result.risk_discrimination_score,
                    'drawdown_discrimination_score': economic_result.drawdown_discrimination_score,
                    'volume_discrimination_score': economic_result.volume_discrimination_score,
                    'strategy_performance_score': economic_result.strategy_performance_score,
                    'validation_time': economic_result.validation_time
                },
                'regime_persistence': {
                    'overall_persistence_score': persistence_result.overall_persistence_score,
                    'lifespan_score': persistence_result.lifespan_score,
                    'transition_score': persistence_result.transition_score,
                    'economic_coherence_score': persistence_result.economic_coherence_score,
                    'volatility_persistence_score': persistence_result.volatility_persistence_score,
                    'n_regimes': persistence_result.n_regimes,
                    'n_transitions': persistence_result.n_transitions,
                    'avg_regime_lifespan': persistence_result.avg_regime_lifespan,
                    'validation_time': persistence_result.validation_time
                },
                'data_info': {
                    'n_samples': len(market_data),
                    'n_features': features.shape[1] if len(features.shape) > 1 else 0,
                    'n_true_regimes': len(np.unique(true_regime_labels)),
                    'n_clusters': len(np.unique(cluster_labels)),
                    'noise_ratio': np.sum(cluster_labels == -1) / len(cluster_labels) if -1 in cluster_labels else 0.0
                }
            }
            
            return test_results
            
        except Exception as e:
            logger.error(f"Synthetic test failed for {test_name}: {e}")
            return {
                'test_name': test_name,
                'scenario': scenario,
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'success': False
            }
    
    def load_baseline_scores(self, baseline_file: str = "synthetic_baseline_scores.json") -> Dict[str, Any]:
        """
        Load baseline scores from file.
        
        Args:
            baseline_file: Name of baseline file
            
        Returns:
            Dictionary containing baseline scores
        """
        baseline_path = self.baseline_dir / baseline_file
        
        if baseline_path.exists():
            with open(baseline_path, 'r') as f:
                self.baseline_scores = json.load(f)
            logger.info(f"Loaded baseline scores from {baseline_path}")
        else:
            logger.warning(f"Baseline file {baseline_path} not found. Creating new baseline.")
            self.baseline_scores = {}
        
        return self.baseline_scores
    
    def save_baseline_scores(self, baseline_file: str = "synthetic_baseline_scores.json") -> None:
        """
        Save current scores as baseline.
        
        Args:
            baseline_file: Name of baseline file
        """
        baseline_path = self.baseline_dir / baseline_file
        
        with open(baseline_path, 'w') as f:
            json.dump(self.baseline_scores, f, indent=2)
        
        logger.info(f"Saved baseline scores to {baseline_path}")
    
    def compare_with_baseline(self, 
                            test_results: Dict[str, Any],
                            baseline_scores: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare test results with baseline scores.
        
        Args:
            test_results: Current test results
            baseline_scores: Baseline scores to compare against
            
        Returns:
            Dictionary containing comparison results
        """
        test_name = test_results['test_name']
        
        if test_name not in baseline_scores:
            return {
                'test_name': test_name,
                'status': 'NO_BASELINE',
                'message': f"No baseline found for test {test_name}",
                'differences': {}
            }
        
        baseline = baseline_scores[test_name]
        differences = {}
        max_difference = 0.0
        failed_metrics = []
        
        # Compare optimization scores
        for metric in ['success', 'overall_score']:
            if metric in test_results['optimization'] and metric in baseline['optimization']:
                current = test_results['optimization'][metric]
                baseline_val = baseline['optimization'][metric]
                
                if isinstance(baseline_val, (int, float)) and baseline_val != 0:
                    difference = abs(current - baseline_val) / abs(baseline_val)
                    differences[f'optimization.{metric}'] = {
                        'current': current,
                        'baseline': baseline_val,
                        'difference': difference,
                        'tolerance': self.tolerance
                    }
                    
                    max_difference = max(max_difference, difference)
                    
                    if difference > self.tolerance:
                        failed_metrics.append(f'optimization.{metric}')
        
        # Compare economic validation scores
        for metric in ['overall_economic_score', 'return_separation_score', 
                      'volatility_discrimination_score', 'risk_discrimination_score',
                      'drawdown_discrimination_score', 'volume_discrimination_score',
                      'strategy_performance_score']:
            
            if metric in test_results['economic_validation'] and metric in baseline['economic_validation']:
                current = test_results['economic_validation'][metric]
                baseline_val = baseline['economic_validation'][metric]
                
                if baseline_val != 0:
                    difference = abs(current - baseline_val) / abs(baseline_val)
                    differences[f'economic_validation.{metric}'] = {
                        'current': current,
                        'baseline': baseline_val,
                        'difference': difference,
                        'tolerance': self.tolerance
                    }
                    
                    max_difference = max(max_difference, difference)
                    
                    if difference > self.tolerance:
                        failed_metrics.append(f'economic_validation.{metric}')
        
        # Compare regime persistence scores
        for metric in ['overall_persistence_score', 'lifespan_score', 'transition_score',
                      'economic_coherence_score', 'volatility_persistence_score']:
            
            if metric in test_results['regime_persistence'] and metric in baseline['regime_persistence']:
                current = test_results['regime_persistence'][metric]
                baseline_val = baseline['regime_persistence'][metric]
                
                if baseline_val != 0:
                    difference = abs(current - baseline_val) / abs(baseline_val)
                    differences[f'regime_persistence.{metric}'] = {
                        'current': current,
                        'baseline': baseline_val,
                        'difference': difference,
                        'tolerance': self.tolerance
                    }
                    
                    max_difference = max(max_difference, difference)
                    
                    if difference > self.tolerance:
                        failed_metrics.append(f'regime_persistence.{metric}')
        
        # Determine overall status
        if max_difference <= self.tolerance:
            status = 'PASS'
            message = f"All metrics within tolerance ({self.tolerance:.1%})"
        else:
            status = 'FAIL'
            message = f"Metrics exceeded tolerance: {failed_metrics}"
        
        return {
            'test_name': test_name,
            'status': status,
            'message': message,
            'max_difference': max_difference,
            'tolerance': self.tolerance,
            'failed_metrics': failed_metrics,
            'differences': differences
        }
    
    def run_synthetic_test_suite(self, 
                               test_scenarios: List[Dict[str, Any]],
                               save_baseline: bool = False) -> Dict[str, Any]:
        """
        Run complete synthetic test suite.
        
        Args:
            test_scenarios: List of test scenario configurations
            save_baseline: Whether to save results as new baseline
            
        Returns:
            Dictionary containing all test results
        """
        logger.info(f"Starting synthetic test suite with {len(test_scenarios)} scenarios")
        
        # Load baseline scores
        self.load_baseline_scores()
        
        # Run tests
        all_results = {}
        comparison_results = {}
        
        for i, test_scenario in enumerate(test_scenarios):
            test_name = test_scenario['name']
            scenario = test_scenario['scenario']
            logger.info(f"Running test {i+1}/{len(test_scenarios)}: {test_name} ({scenario})")
            
            try:
                # Generate synthetic data
                market_data, true_labels, scenario_metadata = self.generate_synthetic_market_data(
                    scenario=scenario,
                    n_samples=test_scenario.get('n_samples', 1000),
                    n_features=test_scenario.get('n_features', 50),
                    n_regimes=test_scenario.get('n_regimes', 3),
                    noise_level=test_scenario.get('noise_level', 0.1),
                    regime_persistence=test_scenario.get('regime_persistence', 0.8)
                )
                
                # Generate features
                features, feature_names, feature_categories = self.feature_engineer.engineer_features(market_data)
                
                # Run synthetic test
                test_results = self.run_synthetic_test(
                    test_name=test_name,
                    scenario=scenario,
                    market_data=market_data,
                    true_regime_labels=true_labels,
                    features=features,
                    feature_names=feature_names
                )
                
                all_results[test_name] = test_results
                
                # Compare with baseline
                comparison = self.compare_with_baseline(test_results, self.baseline_scores)
                comparison_results[test_name] = comparison
                
                logger.info(f"Test {test_name}: {comparison['status']} - {comparison['message']}")
                
            except Exception as e:
                logger.error(f"Test {test_name} failed with error: {e}")
                all_results[test_name] = {
                    'test_name': test_name,
                    'scenario': scenario,
                    'error': str(e),
                    'success': False
                }
                comparison_results[test_name] = {
                    'test_name': test_name,
                    'status': 'ERROR',
                    'message': str(e)
                }
        
        # Save baseline if requested
        if save_baseline:
            self.baseline_scores = all_results
            self.save_baseline_scores()
            logger.info("Saved current results as new baseline")
        
        # Compile summary
        total_tests = len(test_scenarios)
        passed_tests = sum(1 for r in comparison_results.values() if r['status'] == 'PASS')
        failed_tests = sum(1 for r in comparison_results.values() if r['status'] == 'FAIL')
        error_tests = sum(1 for r in comparison_results.values() if r['status'] == 'ERROR')
        
        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'error_tests': error_tests,
            'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'tolerance': self.tolerance,
            'test_results': all_results,
            'comparison_results': comparison_results
        }
        
        logger.info(f"Synthetic test suite completed: {passed_tests}/{total_tests} passed")
        
        return summary


def create_default_test_scenarios() -> List[Dict[str, Any]]:
    """Create default test scenarios for synthetic testing."""
    return [
        {
            'name': 'bull_market_3_regimes',
            'scenario': 'bull_market',
            'n_samples': 1000,
            'n_features': 50,
            'n_regimes': 3,
            'noise_level': 0.1,
            'regime_persistence': 0.8
        },
        {
            'name': 'crisis_2_regimes',
            'scenario': 'crisis',
            'n_samples': 800,
            'n_features': 40,
            'n_regimes': 2,
            'noise_level': 0.2,
            'regime_persistence': 0.7
        },
        {
            'name': 'sideways_4_regimes',
            'scenario': 'sideways',
            'n_samples': 1200,
            'n_features': 60,
            'n_regimes': 4,
            'noise_level': 0.15,
            'regime_persistence': 0.75
        },
        {
            'name': 'high_volatility_3_regimes',
            'scenario': 'high_volatility',
            'n_samples': 900,
            'n_features': 45,
            'n_regimes': 3,
            'noise_level': 0.25,
            'regime_persistence': 0.65
        },
        {
            'name': 'default_2_regimes',
            'scenario': 'default',
            'n_samples': 600,
            'n_features': 30,
            'n_regimes': 2,
            'noise_level': 0.12,
            'regime_persistence': 0.85
        }
    ]


def run_synthetic_tests(tolerance: float = 0.05, 
                       save_baseline: bool = False,
                       test_scenarios: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run synthetic dataset tests.
    
    Args:
        tolerance: Maximum allowed difference in scores
        save_baseline: Whether to save results as new baseline
        test_scenarios: Custom test scenarios (uses default if None)
        
    Returns:
        Dictionary containing test results
    """
    if test_scenarios is None:
        test_scenarios = create_default_test_scenarios()
    
    tester = SyntheticDatasetTester(tolerance=tolerance)
    results = tester.run_synthetic_test_suite(test_scenarios, save_baseline=save_baseline)
    
    return results


if __name__ == "__main__":
    # Run synthetic tests
    print("Running synthetic dataset tests...")
    results = run_synthetic_tests(tolerance=0.05, save_baseline=False)
    
    print(f"\nSynthetic Test Results:")
    print(f"Total Tests: {results['total_tests']}")
    print(f"Passed: {results['passed_tests']}")
    print(f"Failed: {results['failed_tests']}")
    print(f"Errors: {results['error_tests']}")
    print(f"Pass Rate: {results['pass_rate']:.1%}")
    
    # Print detailed results for failed tests
    for test_name, comparison in results['comparison_results'].items():
        if comparison['status'] != 'PASS':
            print(f"\n{test_name}: {comparison['status']} - {comparison['message']}")
            if 'failed_metrics' in comparison and comparison['failed_metrics']:
                print(f"  Failed metrics: {comparison['failed_metrics']}")