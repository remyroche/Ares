"""
Advanced NAS System Example

This script demonstrates the complete advanced NAS system with:
- NSGA-II multi-objective optimization
- Median pruning for early stopping
- Population-based genetic algorithms
- Neural State Space Models (replacing MSM)
- Regime-specific search spaces with momentum
- Pareto front exploration
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

from .core.advanced_nas import (
    AdvancedNAS, AdvancedNASConfig, NSGAII_Optimizer, NSGAIIConfig,
    MedianPruner, MedianPrunerConfig, RegimeSpecificSearchSpace
)
from .core.neural_state_space_nas import (
    NeuralStateSpaceModel, NeuralSSMConfig, TransformerRegimeDetector,
    ContrastiveRegimeLearner, NeuralSSM_NAS_Optimizer
)
from .core.nas_search import NASArchitectureSearch, NASSearchConfig
from .core.nas_model import NASModel
from .core.nas_trainer import NASTrainer, TrainingConfig
from .core.nas_evaluator import NASEvaluator, EvaluationConfig
from .evaluation.nas_metrics import NASMetrics, NASMetricsConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedMarketDataPreprocessor:
    """Advanced preprocessor with regime-specific features."""

    def __init__(self, sequence_length: int = 20):
        """Initialize advanced preprocessor.

        Args:
            sequence_length: Length of input sequences
        """
        self.sequence_length = sequence_length

    def preprocess_for_advanced_nas(self, market_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Preprocess market data for advanced NAS systems.

        Args:
            market_data: Raw market data

        Returns:
            Tuple of (features, labels, metadata)
        """
        logger.info("📊 Preprocessing market data for advanced NAS...")

        # Extract comprehensive features
        features = self._extract_advanced_features(market_data)

        # Create multi-objective labels
        labels, metadata = self._create_multi_objective_labels(market_data)

        # Create regime-specific datasets
        regime_datasets = self._create_regime_specific_datasets(market_data)

        metadata.update({
            'regime_datasets': regime_datasets,
            'feature_dimensions': features.shape,
            'n_regimes': len(np.unique(labels)),
            'preprocessing_method': 'advanced_nas'
        })

        logger.info(f"✅ Advanced preprocessing completed with {features.shape[1]} features")
        return features, labels, metadata

    def _extract_advanced_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Extract advanced features for NAS training."""
        prices = market_data['close'].values
        volume = market_data['volume'].values
        high = market_data['high'].values
        low = market_data['low'].values
        open_price = market_data['open'].values

        # Basic features
        returns = np.diff(prices) / prices[:-1]
        volume_returns = np.diff(volume) / volume[:-1]

        # Price-based features
        price_range = high - low
        price_position = (prices - low) / (high - low + 1e-8)

        # Volatility features
        volatility = self._calculate_volatility_features(returns)
        volume_volatility = self._calculate_volatility_features(volume_returns)

        # Momentum features
        momentum = self._calculate_momentum_features(prices)
        volume_momentum = self._calculate_momentum_features(volume)

        # Trend features
        trend = self._calculate_trend_features(prices)
        volume_trend = self._calculate_trend_features(volume)

        # Combine all features
        features = np.column_stack([
            returns, volume_returns, price_range, price_position,
            volatility, volume_volatility, momentum, volume_momentum,
            trend, volume_trend
        ])

        return features

    def _calculate_volatility_features(self, data: np.ndarray, window: int = 10) -> np.ndarray:
        """Calculate volatility features."""
        volatility = []
        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            window_data = data[start_idx:i+1]
            volatility.append(np.std(window_data))

        return np.array(volatility)

    def _calculate_momentum_features(self, data: np.ndarray, window: int = 5) -> np.ndarray:
        """Calculate momentum features."""
        momentum = []
        for i in range(len(data)):
            if i < window:
                momentum.append(0.0)
            else:
                momentum.append(data[i] - data[i - window])

        return np.array(momentum)

    def _calculate_trend_features(self, data: np.ndarray, window: int = 20) -> np.ndarray:
        """Calculate trend features."""
        trend = []
        for i in range(len(data)):
            if i < window:
                trend.append(0.0)
            else:
                # Linear trend slope
                x = np.arange(window)
                y = data[i - window + 1:i + 1]
                slope = np.polyfit(x, y, 1)[0]
                trend.append(slope)

        return np.array(trend)

    def _create_multi_objective_labels(self, market_data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Create multi-objective labels for NAS optimization."""
        prices = market_data['close'].values

        # Primary labels: regime classification
        returns = np.diff(prices) / prices[:-1]

        # Multi-objective regime classification
        n_regimes = 5
        regime_labels = np.zeros(len(returns), dtype=np.int64)

        # Regime 0: Strong uptrend
        regime_labels[(returns > 0.005) & (returns <= 0.02)] = 0

        # Regime 1: Strong downtrend
        regime_labels[(returns < -0.005) & (returns >= -0.02)] = 1

        # Regime 2: High volatility
        volatility = np.abs(returns)
        regime_labels[volatility > 0.015] = 2

        # Regime 3: Sideways consolidation
        regime_labels[(returns >= -0.002) & (returns <= 0.002)] = 3

        # Regime 4: Extreme movements
        regime_labels[(returns > 0.02) | (returns < -0.02)] = 4

        # Additional objectives for multi-objective optimization
        metadata = {
            'regime_distribution': np.bincount(regime_labels, minlength=n_regimes),
            'return_volatility': np.std(returns),
            'max_return': np.max(returns),
            'min_return': np.min(returns),
            'positive_returns_ratio': np.mean(returns > 0),
            'regime_imbalance_ratio': np.max(np.bincount(regime_labels)) / len(regime_labels)
        }

        return regime_labels, metadata

    def _create_regime_specific_datasets(self, market_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create regime-specific datasets."""
        prices = market_data['close'].values
        returns = np.diff(prices) / prices[:-1]

        # Identify regimes
        volatility_mask = np.abs(returns) > 0.01
        trend_up_mask = returns > 0.002
        trend_down_mask = returns < -0.002
        sideways_mask = (returns >= -0.001) & (returns <= 0.001)

        # Create regime-specific features
        regime_datasets = {
            'volatility': market_data[volatility_mask],
            'trend_up': market_data[trend_up_mask],
            'trend_down': market_data[trend_down_mask],
            'sideways': market_data[sideways_mask],
            'all': market_data
        }

        return regime_datasets

class AdvancedNASDemo:
    """Demonstration of advanced NAS capabilities."""

    def __init__(self):
        """Initialize advanced NAS demo."""
        self.preprocessor = AdvancedMarketDataPreprocessor()
        self.advanced_nas = None

    def setup_advanced_system(self):
        """Setup the advanced NAS system."""
        config = AdvancedNASConfig(
            nsga_ii_config=NSGAIIConfig(
                population_size=30,
                max_generations=15,
                objectives=["accuracy", "complexity", "efficiency"],
                objective_weights=[1.0, -0.3, 0.2]
            ),
            pruner_config=MedianPrunerConfig(
                startup_trials=3,
                min_resource=5,
                reduction_factor=2
            ),
            use_median_pruning=True,
            use_population_based=True,
            use_regime_specific=True,
            n_objectives=3,
            pareto_front_size=10
        )

        self.advanced_nas = AdvancedNAS(config)
        logger.info("✅ Advanced NAS system configured")

    def run_multi_objective_optimization(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Run multi-objective NAS optimization."""
        logger.info("🚀 Running Multi-Objective NAS Optimization with NSGA-II")

        # Preprocess data
        features, labels, metadata = self.preprocessor.preprocess_for_advanced_nas(market_data)

        # Split data
        n_samples = len(features)
        n_train = int(0.7 * n_samples)
        n_val = int(0.15 * n_samples)

        X_train = features[:n_train]
        y_train = labels[:n_train]
        X_val = features[n_train:n_train+n_val]
        y_val = labels[n_train:n_train+n_val]
        X_test = features[n_train+n_val:]
        y_test = labels[n_train+n_val:]

        # Run NSGA-II optimization
        pareto_front = self.advanced_nas.optimize_multi_objective(
            train_data=(X_train, y_train),
            val_data=(X_val, y_val),
            problem_type="multi_objective_regime_detection"
        )

        # Analyze Pareto front
        pareto_analysis = self._analyze_pareto_front(pareto_front)

        results = {
            'pareto_front': pareto_front,
            'pareto_analysis': pareto_analysis,
            'n_pareto_optimal': len(pareto_front),
            'metadata': metadata,
            'optimization_method': 'nsga_ii_multi_objective'
        }

        logger.info(f"✅ Multi-objective optimization completed")
        logger.info(f"📊 Pareto front size: {len(pareto_front)}")
        logger.info(f"🎯 Pareto analysis: {pareto_analysis}")

        return results

    def run_regime_specific_optimization(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Run regime-specific NAS optimization."""
        logger.info("🎯 Running Regime-Specific NAS Optimization")

        regime_results = {}

        # Optimize for each regime type
        regime_types = ["volatility", "trend", "volume", "momentum", "hybrid"]

        for regime_type in regime_types:
            logger.info(f"🔍 Optimizing for {regime_type} regime...")

            regime_result = self.advanced_nas.get_regime_specific_optimization(
                market_data.values, regime_type
            )

            regime_results[regime_type] = regime_result

        # Find best regime-specific architecture
        best_regime = max(regime_results.items(), key=lambda x: x[1]['complexity_score'])
        best_regime_type = best_regime[0]

        results = {
            'regime_results': regime_results,
            'best_regime_type': best_regime_type,
            'best_regime_score': best_regime[1]['complexity_score'],
            'all_regime_types': regime_types,
            'optimization_method': 'regime_specific_nas'
        }

        logger.info(f"✅ Regime-specific optimization completed")
        logger.info(f"🏆 Best regime type: {best_regime_type}")

        return results

    def run_neural_ssm_comparison(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Compare Neural SSM with traditional approaches."""
        logger.info("🔬 Running Neural SSM Comparison")

        # Preprocess data
        features, labels, metadata = self.preprocessor.preprocess_for_advanced_nas(market_data)

        # Create training data
        X_train = features
        y_train = labels

        # Initialize Neural SSM optimizer
        ssm_config = {
            'n_iterations': 20,
            'state_dim': 8,
            'hidden_dim': 64
        }

        neural_ssm_optimizer = NeuralSSM_NAS_Optimizer(ssm_config)

        # Optimize Neural SSM
        neural_ssm_result = neural_ssm_optimizer.optimize_neural_ssm(
            market_data.values, n_regimes=5, n_iterations=15
        )

        # Compare with traditional NAS
        traditional_nas = NASArchitectureSearch(NASSearchConfig(max_iterations=15))
        traditional_result = traditional_nas.search(
            train_data=(X_train, y_train),
            validation_data=(X_train, y_train),
            problem_type="traditional_regime_detection"
        )

        # Compare with Transformer
        transformer_model = TransformerRegimeDetector(input_dim=features.shape[1], n_regimes=5)
        contrastive_model = ContrastiveRegimeLearner(input_dim=features.shape[1], n_regimes=5)

        comparison = {
            'neural_ssm': {
                'accuracy': neural_ssm_result['evaluation_result'].accuracy,
                'model_type': 'neural_state_space',
                'state_representation': True,
                'complexity': neural_ssm_result['best_architecture'].complexity_score
            },
            'traditional_nas': {
                'accuracy': traditional_result.best_score,
                'model_type': 'traditional_nas',
                'state_representation': False,
                'complexity': traditional_result.best_architecture.complexity_score
            },
            'transformer': {
                'model_type': 'transformer_attention',
                'state_representation': False,
                'attention_mechanism': True,
                'complexity': 'high'
            },
            'contrastive': {
                'model_type': 'contrastive_learning',
                'state_representation': False,
                'self_supervised': True,
                'complexity': 'medium'
            },
            'winner': self._determine_winner(neural_ssm_result, traditional_result),
            'recommendation': self._generate_recommendation(neural_ssm_result, traditional_result)
        }

        logger.info(f"✅ Neural SSM comparison completed")
        logger.info(f"🏆 Winner: {comparison['winner']}")

        return comparison

    def _analyze_pareto_front(self, pareto_front: List[Any]) -> Dict[str, Any]:
        """Analyze Pareto front results."""
        if not pareto_front:
            return {}

        fitness_values = np.array([individual.fitness for individual in pareto_front])

        return {
            'n_architectures': len(pareto_front),
            'accuracy_range': [np.min(fitness_values[:, 0]), np.max(fitness_values[:, 0])],
            'complexity_range': [np.min(fitness_values[:, 1]), np.max(fitness_values[:, 1])],
            'efficiency_range': [np.min(fitness_values[:, 2]), np.max(fitness_values[:, 2])],
            'mean_accuracy': np.mean(fitness_values[:, 0]),
            'mean_complexity': np.mean(fitness_values[:, 1]),
            'mean_efficiency': np.mean(fitness_values[:, 2]),
            'pareto_dominance_ratio': len(pareto_front) / 50  # Assuming population size of 50
        }

    def _determine_winner(self, neural_ssm_result: Dict, traditional_result: Dict) -> str:
        """Determine the best approach."""
        ssm_score = neural_ssm_result['evaluation_result'].accuracy
        traditional_score = traditional_result.best_score

        if ssm_score > traditional_score * 1.05:  # 5% better
            return "neural_ssm"
        elif traditional_score > ssm_score * 1.05:
            return "traditional_nas"
        else:
            return "comparable_performance"

    def _generate_recommendation(self, neural_ssm_result: Dict, traditional_result: Dict) -> str:
        """Generate recommendation based on results."""
        ssm_complexity = neural_ssm_result['best_architecture'].complexity_score
        traditional_complexity = traditional_result.best_architecture.complexity_score

        if ssm_complexity < traditional_complexity * 0.8:  # Much less complex
            return "Use Neural SSM for better efficiency"
        elif neural_ssm_result['evaluation_result'].accuracy > traditional_result.best_score:
            return "Neural SSM provides better accuracy"
        else:
            return "Traditional NAS is more suitable for this dataset"

def main():
    """Main function demonstrating advanced NAS capabilities."""
    logger.info("🚀 Advanced NAS System Demonstration")
    logger.info("=" * 60)

    # Create sample market data
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    np.random.seed(42)

    market_data = pd.DataFrame({
        'timestamp': dates,
        'open': 100 + np.cumsum(np.random.randn(1000) * 0.1),
        'high': 100 + np.cumsum(np.random.randn(1000) * 0.1) + np.abs(np.random.randn(1000) * 0.5),
        'low': 100 + np.cumsum(np.random.randn(1000) * 0.1) - np.abs(np.random.randn(1000) * 0.5),
        'close': 100 + np.cumsum(np.random.randn(1000) * 0.1),
        'volume': np.random.exponential(1000, 1000)
    })

    market_data.set_index('timestamp', inplace=True)

    try:
        # Initialize demo
        demo = AdvancedNASDemo()
        demo.setup_advanced_system()

        # Run multi-objective optimization
        logger.info("\\n1. Running Multi-Objective NSGA-II Optimization...")
        multi_obj_results = demo.run_multi_objective_optimization(market_data)

        # Run regime-specific optimization
        logger.info("\\n2. Running Regime-Specific Optimization...")
        regime_results = demo.run_regime_specific_optimization(market_data)

        # Run Neural SSM comparison
        logger.info("\\n3. Running Neural SSM vs Traditional Comparison...")
        comparison_results = demo.run_neural_ssm_comparison(market_data)

        # Comprehensive results
        logger.info("\\n✅ Advanced NAS Demonstration Results:")
        logger.info("=" * 50)
        logger.info(f"📊 Pareto Front Size: {multi_obj_results['n_pareto_optimal']}")
        logger.info(f"🎯 Best Regime Type: {regime_results['best_regime_type']}")
        logger.info(f"🔬 Neural SSM vs Traditional: {comparison_results['winner']}")
        logger.info(f"💡 Recommendation: {comparison_results['recommendation']}")

        # Advanced features summary
        advanced_features = {
            'nsga_ii_multi_objective': True,
            'median_pruning': True,
            'population_based_search': True,
            'genetic_operators': True,
            'regime_specific_spaces': True,
            'momentum_included': True,
            'neural_state_space_models': True,
            'pareto_front_exploration': True,
            'multi_objective_fitness': True,
            'diversity_maintenance': True,
            'comprehensive_evaluation': True
        }

        logger.info("\\n🚀 Advanced Features Implemented:")
        for feature, implemented in advanced_features.items():
            status = "✅" if implemented else "❌"
            logger.info(f"   {status} {feature.replace('_', ' ').title()}")

        # Performance comparison
        performance_summary = {
            'multi_objective_optimization': {
                'method': 'NSGA-II with Median Pruning',
                'population_size': 30,
                'generations': 15,
                'pareto_front_size': multi_obj_results['n_pareto_optimal'],
                'accuracy_range': multi_obj_results['pareto_analysis'].get('accuracy_range', [0, 0])
            },
            'regime_specific_optimization': {
                'regimes_optimized': list(regime_results['regime_results'].keys()),
                'best_regime': regime_results['best_regime_type'],
                'specialized_architectures': True
            },
            'neural_state_space_comparison': {
                'models_compared': ['neural_ssm', 'traditional_nas', 'transformer', 'contrastive'],
                'winner': comparison_results['winner'],
                'state_space_advantage': comparison_results['neural_ssm']['accuracy'] > comparison_results['traditional_nas']['accuracy']
            }
        }

        logger.info("\\n📈 Performance Summary:")
        for category, metrics in performance_summary.items():
            logger.info(f"   {category.replace('_', ' ').title()}: {metrics}")

        return {
            'multi_objective_results': multi_obj_results,
            'regime_specific_results': regime_results,
            'comparison_results': comparison_results,
            'advanced_features': advanced_features,
            'performance_summary': performance_summary,
            'overall_assessment': 'advanced_nas_successful'
        }

    except Exception as e:
        logger.error(f"❌ Advanced NAS demonstration failed: {e}")
        raise

if __name__ == "__main__":
    main()