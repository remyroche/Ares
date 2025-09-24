"""
Enhanced Analysis Integration Examples for ML Common

This module provides comprehensive examples of how to integrate the enhanced
analysis tools (learning curve analysis, bootstrap confidence intervals,
and adaptive regularization) with existing ml_common infrastructure.

Examples show:
1. Using enhanced analysis in existing training pipelines
2. Integrating with model evaluation and reporting
3. Leveraging adaptive regularization in model creation
4. Combining multiple analysis tools for comprehensive insights
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

# Import existing ml_common components
try:
    from src.utils.ml_common.models.model_factory import EnhancedModelFactory, ModelType
    from src.utils.ml_common.evaluation.evaluation_utils import EvaluationUtils
    from src.utils.ml_common.config.base_training_config import BaseTrainingConfig
    from src.utils.ml_common.training.training_utils import TrainingUtils
    ML_COMMON_AVAILABLE = True
except ImportError:
    ML_COMMON_AVAILABLE = False

# Import enhanced analysis tools
try:
    from src.utils.ml_common.evaluation.enhanced_learning_curve_analysis import EnhancedLearningCurveAnalyzer
    from src.utils.ml_common.evaluation.enhanced_bootstrap_confidence_intervals import EnhancedBootstrapConfidenceIntervalAnalyzer
    ENHANCED_ANALYSIS_AVAILABLE = True
except ImportError:
    ENHANCED_ANALYSIS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class EnhancedAnalysisConfig:
    """Configuration for enhanced analysis integration."""
    enable_learning_curve_analysis: bool = True
    enable_bootstrap_analysis: bool = True
    enable_adaptive_regularization: bool = True

    # Learning curve settings
    learning_curve_train_sizes: List[float] = None
    learning_curve_cv_folds: int = 5
    learning_curve_scoring: str = 'accuracy'

    # Bootstrap settings
    bootstrap_n_iterations: int = 100
    bootstrap_confidence_level: float = 0.95
    bootstrap_train_size: float = 0.7

    # Adaptive regularization settings
    adaptive_regime_labels: Optional[np.ndarray] = None

    def __post_init__(self):
        """Set default values."""
        if self.learning_curve_train_sizes is None:
            self.learning_curve_train_sizes = np.linspace(0.1, 1.0, 10).tolist()


class EnhancedAnalysisIntegrationManager:
    """
    Manager class for integrating enhanced analysis tools with ml_common infrastructure.

    This class provides a unified interface for:
    1. Creating models with adaptive regularization
    2. Performing learning curve analysis
    3. Running bootstrap confidence interval analysis
    4. Generating comprehensive reports
    """

    def __init__(self, config: EnhancedAnalysisConfig = None):
        """
        Initialize the enhanced analysis integration manager.

        Args:
            config: Configuration for enhanced analysis
        """
        self.config = config or EnhancedAnalysisConfig()

        # Initialize components
        self.evaluation_utils = EvaluationUtils()
        self.model_factory = EnhancedModelFactory()

        # Initialize enhanced analyzers
        self.learning_curve_analyzer = None
        self.bootstrap_analyzer = None

        if ENHANCED_ANALYSIS_AVAILABLE:
            self.learning_curve_analyzer = EnhancedLearningCurveAnalyzer(
                random_state=42, n_jobs=-1
            )
            self.bootstrap_analyzer = EnhancedBootstrapConfidenceIntervalAnalyzer(
                n_bootstrap=self.config.bootstrap_n_iterations,
                confidence_level=self.config.bootstrap_confidence_level,
                n_jobs=-1
            )

        logger.info("✅ Enhanced Analysis Integration Manager initialized")

    def create_model_with_enhanced_features(
        self,
        model_type: str,
        model_name: str,
        regime_labels: Optional[np.ndarray] = None,
        **custom_params
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Create model with adaptive regularization and enhanced features.

        Args:
            model_type: Type of model to create
            model_name: Name for the model
            regime_labels: Regime labels for adaptive regularization
            **custom_params: Additional model parameters

        Returns:
            Tuple of (model_instance, regularization_info_dict)
        """
        # Map string model type to ModelType enum
        try:
            model_type_enum = ModelType[model_type.upper()]
        except KeyError:
            raise ValueError(f"Unknown model type: {model_type}")

        # Use enhanced model factory with adaptive regularization
        model, reg_info = self.model_factory.create_model_with_adaptive_regularization(
            model_type_enum,
            model_name,
            regime_labels=regime_labels,
            **custom_params
        )

        return model, reg_info

    def perform_comprehensive_analysis(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_full: Optional[np.ndarray] = None,
        y_full: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive enhanced analysis on a trained model.

        Args:
            model: Trained model instance
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            X_full: Full feature matrix (for bootstrap analysis)
            y_full: Full target labels (for bootstrap analysis)

        Returns:
            Dictionary with comprehensive analysis results
        """
        results = {
            'enhanced_analysis_performed': ENHANCED_ANALYSIS_AVAILABLE,
            'learning_curve_analysis': None,
            'bootstrap_analysis': None,
            'combined_insights': [],
            'recommendations': []
        }

        if not ENHANCED_ANALYSIS_AVAILABLE:
            logger.warning("⚠️ Enhanced analysis tools not available")
            results['recommendations'].append('Enhanced analysis tools not available - consider installing HMM training enhancements')
            return results

        # Perform learning curve analysis
        if self.config.enable_learning_curve_analysis:
            learning_curve_results = self.evaluation_utils.analyze_learning_curves(
                model, X_train, y_train, X_test, y_test,
                train_sizes=self.config.learning_curve_train_sizes,
                cv_folds=self.config.learning_curve_cv_folds,
                scoring=self.config.learning_curve_scoring
            )

            if learning_curve_results:
                results['learning_curve_analysis'] = learning_curve_results
                results['combined_insights'].extend([
                    f"Learning Rate: {learning_curve_results.get('learning_rate', 'unknown')}",
                    f"Convergence Stability: {learning_curve_results.get('convergence_stability', 'unknown')}",
                    f"Overfitting Risk: {learning_curve_results.get('overfitting_risk', 'unknown')}",
                    f"Training Efficiency: {learning_curve_results.get('training_efficiency', 'unknown')}"
                ])
                results['recommendations'].extend(learning_curve_results.get('recommendations', []))

        # Perform bootstrap confidence interval analysis
        if self.config.enable_bootstrap_analysis and X_full is not None and y_full is not None:
            bootstrap_results = self.evaluation_utils.analyze_bootstrap_confidence_intervals(
                model, X_full, y_full,
                train_size=self.config.bootstrap_train_size,
                scoring_metrics=['accuracy', 'f1', 'precision', 'recall']
            )

            if bootstrap_results:
                results['bootstrap_analysis'] = bootstrap_results
                results['combined_insights'].extend([
                    f"Model Stability Score: {bootstrap_results.get('stability_score', 0):.3f}",
                    f"Stability Level: {bootstrap_results.get('stability_level', 'unknown')}",
                    f"Overfitting Probability: {bootstrap_results.get('overfitting_probability', 0):.1%}",
                    f"Overfitting Risk: {bootstrap_results.get('overfitting_risk', 'unknown')}"
                ])
                results['recommendations'].extend(bootstrap_results.get('recommendations', []))

        return results

    def analyze_multiple_models(
        self,
        models: List[Any],
        model_names: List[str],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_full: Optional[np.ndarray] = None,
        y_full: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Perform enhanced analysis on multiple models and compare results.

        Args:
            models: List of trained model instances
            model_names: Names for each model
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            X_full: Full feature matrix (for bootstrap analysis)
            y_full: Full target labels (for bootstrap analysis)

        Returns:
            Dictionary with multi-model analysis results
        """
        if len(models) != len(model_names):
            raise ValueError("Number of models must match number of model names")

        results = {
            'model_count': len(models),
            'individual_analyses': {},
            'comparison_summary': {},
            'best_model_analysis': {},
            'overall_recommendations': []
        }

        # Analyze each model
        for model, name in zip(models, model_names):
            logger.info(f"🔍 Analyzing model: {name}")
            analysis = self.perform_comprehensive_analysis(
                model, X_train, y_train, X_test, y_test, X_full, y_full
            )
            results['individual_analyses'][name] = analysis

        # Generate comparison summary
        results['comparison_summary'] = self._generate_comparison_summary(results['individual_analyses'])

        # Determine best model
        if self.bootstrap_analyzer:
            try:
                bootstrap_comparison = self.bootstrap_analyzer.compare_models_bootstrap(
                    models, model_names, X_full or X_test, y_full or y_test
                )
                results['best_model_analysis'] = {
                    'best_model': bootstrap_comparison.get('best_model'),
                    'best_score': bootstrap_comparison.get('best_score', 0.0),
                    'bootstrap_comparison': bootstrap_comparison
                }
            except Exception as e:
                logger.warning(f"⚠️ Bootstrap model comparison failed: {e}")

        # Generate overall recommendations
        results['overall_recommendations'] = self._generate_overall_recommendations(results)

        return results

    def _generate_comparison_summary(self, individual_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary comparison across all models."""
        summary = {
            'models_with_learning_curve_analysis': 0,
            'models_with_bootstrap_analysis': 0,
            'learning_curve_risk_distribution': {},
            'bootstrap_stability_distribution': {},
            'average_stability_score': 0.0,
            'average_overfitting_probability': 0.0
        }

        stability_scores = []
        overfitting_probs = []

        for model_name, analysis in individual_analyses.items():
            # Count available analyses
            if analysis.get('learning_curve_analysis'):
                summary['models_with_learning_curve_analysis'] += 1

            if analysis.get('bootstrap_analysis'):
                summary['models_with_bootstrap_analysis'] += 1

                # Collect bootstrap metrics
                bootstrap_analysis = analysis['bootstrap_analysis']
                stability_scores.append(bootstrap_analysis.get('stability_score', 0.0))
                overfitting_probs.append(bootstrap_analysis.get('overfitting_probability', 0.0))

                # Collect risk/stability distributions
                stability_level = bootstrap_analysis.get('stability_level', 'unknown')
                summary['bootstrap_stability_distribution'][stability_level] = \
                    summary['bootstrap_stability_distribution'].get(stability_level, 0) + 1

            # Collect learning curve metrics
            learning_curve_analysis = analysis.get('learning_curve_analysis')
            if learning_curve_analysis:
                overfitting_risk = learning_curve_analysis.get('overfitting_risk', 'unknown')
                summary['learning_curve_risk_distribution'][overfitting_risk] = \
                    summary['learning_curve_risk_distribution'].get(overfitting_risk, 0) + 1

        # Calculate averages
        if stability_scores:
            summary['average_stability_score'] = np.mean(stability_scores)
            summary['average_overfitting_probability'] = np.mean(overfitting_probs)

        return summary

    def _generate_overall_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate overall recommendations based on all analyses."""
        recommendations = []

        comparison_summary = results.get('comparison_summary', {})
        best_model_analysis = results.get('best_model_analysis', {})

        # Overall analysis availability
        if comparison_summary.get('models_with_learning_curve_analysis', 0) == 0:
            recommendations.append('Consider enabling learning curve analysis for better training insights')

        if comparison_summary.get('models_with_bootstrap_analysis', 0) == 0:
            recommendations.append('Consider enabling bootstrap analysis for statistical model evaluation')

        # Performance recommendations
        avg_stability = comparison_summary.get('average_stability_score', 0.0)
        if avg_stability < 0.6:
            recommendations.append('Overall model stability is low - consider ensemble methods')
        elif avg_stability < 0.8:
            recommendations.append('Overall model stability is moderate - consider hyperparameter optimization')

        # Best model recommendations
        if best_model_analysis.get('best_model'):
            best_model = best_model_analysis['best_model']
            recommendations.append(f"Best performing model: {best_model} - consider using as primary model")

        return recommendations


# Example usage functions
def example_basic_enhanced_analysis():
    """Example of basic enhanced analysis integration."""
    print("🔍 Example 1: Basic Enhanced Analysis Integration")
    print("=" * 60)

    if not ML_COMMON_AVAILABLE or not ENHANCED_ANALYSIS_AVAILABLE:
        print("❌ Required components not available")
        return

    # Create sample data
    np.random.seed(42)
    X_train = np.random.randn(1000, 20)
    y_train = np.random.randint(0, 3, 1000)
    X_test = np.random.randn(200, 20)
    y_test = np.random.randint(0, 3, 200)

    # Create integration manager
    config = EnhancedAnalysisConfig(
        enable_learning_curve_analysis=True,
        enable_bootstrap_analysis=True,
        enable_adaptive_regularization=False
    )
    manager = EnhancedAnalysisIntegrationManager(config)

    # Create and train a simple model
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Perform comprehensive analysis
    analysis_results = manager.perform_comprehensive_analysis(
        model, X_train, y_train, X_test, y_test, X_train, y_train
    )

    # Display results
    print("✅ Analysis completed:")
    print(f"  - Learning curve analysis: {'Available' if analysis_results['learning_curve_analysis'] else 'Not available'}")
    print(f"  - Bootstrap analysis: {'Available' if analysis_results['bootstrap_analysis'] else 'Not available'}")
    print(f"  - Combined insights: {len(analysis_results['combined_insights'])} insights")
    print(f"  - Recommendations: {len(analysis_results['recommendations'])} recommendations")

    if analysis_results['learning_curve_analysis']:
        lc_analysis = analysis_results['learning_curve_analysis']
        print(f"  - Learning rate: {lc_analysis.get('learning_rate', 'N/A')}")
        print(f"  - Overfitting risk: {lc_analysis.get('overfitting_risk', 'N/A')}")

    if analysis_results['bootstrap_analysis']:
        bs_analysis = analysis_results['bootstrap_analysis']
        print(f"  - Stability score: {bs_analysis.get('stability_score', 0):.3f}")
        print(f"  - Overfitting probability: {bs_analysis.get('overfitting_probability', 0):.1%}")


def example_adaptive_regularization_integration():
    """Example of adaptive regularization integration."""
    print("\n🔧 Example 2: Adaptive Regularization Integration")
    print("=" * 60)

    if not ML_COMMON_AVAILABLE:
        print("❌ ML Common components not available")
        return

    # Create sample regime labels for adaptive regularization
    np.random.seed(42)
    regime_labels = np.random.randint(0, 5, 1000)  # 5 regimes

    # Create integration manager with adaptive regularization
    config = EnhancedAnalysisConfig(
        enable_adaptive_regularization=True,
        adaptive_regime_labels=regime_labels
    )
    manager = EnhancedAnalysisIntegrationManager(config)

    # Create model with adaptive regularization
    model, reg_info = manager.create_model_with_enhanced_features(
        'RANDOM_FOREST_CLASSIFIER',
        'adaptive_rf_model',
        regime_labels=regime_labels,
        n_estimators=100,
        random_state=42
    )

    print("✅ Model created with adaptive regularization:")
    print(f"  - Dataset size category: {reg_info.get('dataset_size', 'unknown')}")
    print(f"  - Adaptive reg_alpha: {reg_info.get('reg_alpha', 0):.3f}")
    print(f"  - Adaptive reg_lambda: {reg_info.get('reg_lambda', 0):.3f}")
    print(f"  - Regime analysis: {reg_info.get('min_samples_per_regime', 0)} min samples per regime")


def example_multi_model_comparison():
    """Example of multi-model comparison with enhanced analysis."""
    print("\n📊 Example 3: Multi-Model Comparison with Enhanced Analysis")
    print("=" * 70)

    if not ML_COMMON_AVAILABLE or not ENHANCED_ANALYSIS_AVAILABLE:
        print("❌ Required components not available")
        return

    # Create sample data
    np.random.seed(42)
    X_train = np.random.randn(1000, 20)
    y_train = np.random.randint(0, 3, 1000)
    X_test = np.random.randn(200, 20)
    y_test = np.random.randint(0, 3, 200)

    # Create multiple models
    from sklearn.tree import DecisionTreeClassifier

    models = [
        RandomForestClassifier(n_estimators=100, random_state=42),
        RandomForestClassifier(n_estimators=50, random_state=42),
        DecisionTreeClassifier(random_state=42)
    ]

    model_names = ['RF_100', 'RF_50', 'DT']

    # Train models
    trained_models = []
    for model in models:
        model.fit(X_train, y_train)
        trained_models.append(model)

    # Create integration manager
    config = EnhancedAnalysisConfig(
        enable_learning_curve_analysis=True,
        enable_bootstrap_analysis=True
    )
    manager = EnhancedAnalysisIntegrationManager(config)

    # Perform multi-model analysis
    comparison_results = manager.analyze_multiple_models(
        trained_models, model_names, X_train, y_train, X_test, y_test, X_train, y_train
    )

    # Display results
    print("✅ Multi-model analysis completed:")
    print(f"  - Models analyzed: {comparison_results['model_count']}")
    print(f"  - Models with learning curve analysis: {comparison_results['comparison_summary'].get('models_with_learning_curve_analysis', 0)}")
    print(f"  - Models with bootstrap analysis: {comparison_results['comparison_summary'].get('models_with_bootstrap_analysis', 0)}")

    if comparison_results.get('best_model_analysis', {}).get('best_model'):
        best_model = comparison_results['best_model_analysis']['best_model']
        best_score = comparison_results['best_model_analysis']['best_score']
        print(f"  - Best model: {best_model} (score: {best_score:.3f})")

    print(f"  - Overall recommendations: {len(comparison_results['overall_recommendations'])}")
    for rec in comparison_results['overall_recommendations']:
        print(f"    • {rec}")


def example_hmm_training_integration():
    """Example of integrating enhanced analysis with HMM training."""
    print("\n🧠 Example 4: HMM Training Integration")
    print("=" * 50)

    print("This example shows how HMM training can leverage ml_common enhanced analysis:")
    print()
    print("1. Model Creation with Adaptive Regularization:")
    print("   ```python")
    print("   manager = EnhancedAnalysisIntegrationManager()")
    print("   model, reg_info = manager.create_model_with_enhanced_features(")
    print("       'RANDOM_FOREST_CLASSIFIER', 'hmm_rf_model',")
    print("       regime_labels=cluster_assignments")
    print("   )")
    print("   ```")
    print()
    print("2. Enhanced Analysis in Training Loop:")
    print("   ```python")
    print("   analysis_results = manager.perform_comprehensive_analysis(")
    print("       trained_model, X_train, y_train, X_test, y_test")
    print("   )")
    print("   # Access learning curve and bootstrap results")
    print("   learning_curve_risk = analysis_results['learning_curve_analysis']['overfitting_risk']")
    print("   bootstrap_stability = analysis_results['bootstrap_analysis']['stability_score']")
    print("   ```")
    print()
    print("3. Comprehensive Reporting:")
    print("   ```python")
    print("   # All analysis results integrated into existing reports")
    print("   report = generate_comprehensive_report(")
    print("       training_results, config, enhanced_analysis=analysis_results")
    print("   )")
    print("   ```")
    print()
    print("4. Integration Benefits:")
    print("   - ✅ Automatic adaptive regularization based on regime sizes")
    print("   - ✅ Statistical model stability assessment")
    print("   - ✅ Comprehensive overfitting detection")
    print("   - ✅ Actionable recommendations for model improvement")
    print("   - ✅ Backward compatibility with existing training pipelines")


def run_all_examples():
    """Run all integration examples."""
    print("🚀 Enhanced Analysis Integration Examples")
    print("=" * 60)
    print()

    try:
        example_basic_enhanced_analysis()
        example_adaptive_regularization_integration()
        example_multi_model_comparison()
        example_hmm_training_integration()

        print("\n" + "=" * 60)
        print("🎉 All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Example execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_examples()