"""
Enhanced TAS Example

This example demonstrates the enhanced Tree Architecture Search capabilities,
including modern tree algorithms, automated optimization, advanced evaluation,
and multi-objective optimization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Enhanced TAS components
try:
    from src.utils.nas_tas.core.tas_engine import TASEngine
    from src.utils.nas_tas.optimization.strategy_search import StrategySearchOptimizer, StrategySearchConfig
    ENHANCED_TAS_AVAILABLE = True
except ImportError:
    ENHANCED_TAS_AVAILABLE = False
    logger.warning("⚠️ Enhanced TAS not available")

try:
    from ..models.enhanced_tree_models import (
        EnhancedTreeModelFactory, TreeModelConfig, TreeModelType
    )
    ENHANCED_MODELS_AVAILABLE = True
except ImportError:
    ENHANCED_MODELS_AVAILABLE = False
    logger.warning("⚠️ Enhanced models not available")

try:
    from ..automl.tree_automl import (
        TreeAutoMLManager, AutoMLConfig, AutoMLResult
    )
    AUTOML_AVAILABLE = True
except ImportError:
    AUTOML_AVAILABLE = False
    logger.warning("⚠️ AutoML not available")

try:
    from ..evaluation.advanced_metrics import (
        AdvancedEvaluator, AdvancedEvaluationResult
    )
    ADVANCED_METRICS_AVAILABLE = True
except ImportError:
    ADVANCED_METRICS_AVAILABLE = False
    logger.warning("⚠️ Advanced metrics not available")

try:
    from ...shared_utils.evolutionary_search import (
        EvolutionaryAlgorithmManager, EvolutionaryConfig, EvolutionaryResult
    )
    EVOLUTIONARY_AVAILABLE = True
except ImportError:
    EVOLUTIONARY_AVAILABLE = False
    logger.warning("⚠️ Evolutionary search not available")


def create_sample_data(n_samples: int = 1000, n_features: int = 20, 
                      problem_type: str = "classification", 
                      noise: float = 0.1, random_state: int = 42) -> tuple:
    """Create sample data for demonstration."""
    np.random.seed(random_state)
    
    if problem_type == "classification":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=int(n_features * 0.7),
            n_redundant=int(n_features * 0.2),
            n_clusters_per_class=1,
            random_state=random_state
        )
    else:  # regression
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=int(n_features * 0.7),
            noise=noise,
            random_state=random_state
        )
    
    # Add some regime-like structure
    regime_labels = np.random.choice([0, 1, 2], size=n_samples, p=[0.4, 0.4, 0.2])
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    return X, y, regime_labels, feature_names


def demonstrate_enhanced_models():
    """Demonstrate enhanced tree models."""
    logger.info("🌳 Demonstrating Enhanced Tree Models...")
    
    if not ENHANCED_MODELS_AVAILABLE:
        logger.warning("⚠️ Enhanced models not available")
        return
    
    # Create sample data
    X, y, _, _ = create_sample_data(1000, 20, "classification")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test different model types
    model_types = [
        TreeModelType.XGBOOST,
        TreeModelType.LIGHTGBM,
        TreeModelType.CATBOOST,
        TreeModelType.RANDOM_FOREST,
        TreeModelType.EXTRA_TREES
    ]
    
    results = {}
    
    for model_type in model_types:
        try:
            logger.info(f"   Testing {model_type.value}...")
            
            # Create model configuration
            config = TreeModelConfig(
                model_type=model_type,
                params={
                    'n_estimators': 100,
                    'max_depth': 6,
                    'random_state': 42
                },
                is_classifier=True
            )
            
            # Create and train model
            model_factory = EnhancedTreeModelFactory(config)
            model_factory.fit(X_train, y_train)
            
            # Make predictions
            predictions = model_factory.predict(X_test)
            probabilities = model_factory.predict_proba(X_test)
            
            # Calculate accuracy
            accuracy = (predictions == y_test).mean()
            results[model_type.value] = accuracy
            
            logger.info(f"   {model_type.value} accuracy: {accuracy:.4f}")
            
        except Exception as e:
            logger.warning(f"   ⚠️ {model_type.value} failed: {e}")
            results[model_type.value] = 0.0
    
    # Display results
    logger.info("📊 Enhanced Models Results:")
    for model_type, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"   {model_type}: {accuracy:.4f}")


def demonstrate_automl():
    """Demonstrate AutoML capabilities."""
    logger.info("🤖 Demonstrating AutoML...")
    
    if not AUTOML_AVAILABLE:
        logger.warning("⚠️ AutoML not available")
        return
    
    # Create sample data
    X, y, _, _ = create_sample_data(1000, 20, "classification")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    try:
        # Create AutoML configuration
        config = AutoMLConfig(
            optimization_method="optuna",
            max_trials=50,
            timeout_seconds=300,
            model_types=["xgboost", "lightgbm", "catboost"],
            enable_ensemble=True,
            ensemble_method="voting"
        )
        
        # Create AutoML manager
        automl_manager = TreeAutoMLManager(config)
        
        # Run optimization
        logger.info("   Running AutoML optimization...")
        result = automl_manager.optimize(X_train, y_train, X_test, y_test)
        
        if result.success:
            logger.info(f"   Best model: {result.best_model_type}")
            logger.info(f"   Best score: {result.best_score:.4f}")
            logger.info(f"   Best parameters: {result.best_params}")
            logger.info(f"   Optimization time: {result.optimization_time:.2f}s")
        else:
            logger.warning(f"   AutoML failed: {result.error_message}")
            
    except Exception as e:
        logger.warning(f"   ⚠️ AutoML demonstration failed: {e}")


def demonstrate_evolutionary_search():
    """Demonstrate evolutionary search capabilities."""
    logger.info("🧬 Demonstrating Evolutionary Search...")
    
    if not EVOLUTIONARY_AVAILABLE:
        logger.warning("⚠️ Evolutionary search not available")
        return
    
    try:
        # Create sample data
        X, y, _, _ = create_sample_data(1000, 20, "classification")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define objective functions
        def accuracy_objective(params):
            try:
                # This is a simplified objective function
                # In practice, you would train and evaluate a model
                score = np.random.random()  # Placeholder
                return score
            except Exception:
                return 0.0
        
        def robustness_objective(params):
            try:
                # This is a simplified objective function
                # In practice, you would calculate robustness metrics
                score = np.random.random()  # Placeholder
                return score
            except Exception:
                return 0.0
        
        # Define parameter space
        parameter_space = {
            'n_estimators': {'type': 'integer', 'min': 50, 'max': 200},
            'max_depth': {'type': 'integer', 'min': 3, 'max': 10},
            'learning_rate': {'type': 'continuous', 'min': 0.01, 'max': 0.3}
        }
        
        # Create evolutionary algorithm manager
        config = EvolutionaryConfig(
            population_size=20,
            max_generations=10,
            use_nsga2=True
        )
        evolutionary_manager = EvolutionaryAlgorithmManager(config)
        
        # Run optimization
        logger.info("   Running evolutionary optimization...")
        result = evolutionary_manager.optimize_with_algorithm(
            [accuracy_objective, robustness_objective],
            parameter_space,
            "nsga2"
        )
        
        if result.success:
            logger.info(f"   Pareto front size: {len(result.pareto_front)}")
            logger.info(f"   Optimization time: {result.execution_time:.2f}s")
            logger.info(f"   Generations: {result.convergence_info.get('total_generations', 0)}")
        else:
            logger.warning(f"   Evolutionary search failed: {result.error_message}")
            
    except Exception as e:
        logger.warning(f"   ⚠️ Evolutionary search demonstration failed: {e}")


def demonstrate_advanced_metrics():
    """Demonstrate advanced evaluation metrics."""
    logger.info("📊 Demonstrating Advanced Metrics...")
    
    if not ADVANCED_METRICS_AVAILABLE:
        logger.warning("⚠️ Advanced metrics not available")
        return
    
    try:
        # Create sample returns data
        np.random.seed(42)
        n_samples = 1000
        returns = pd.Series(np.random.normal(0.001, 0.02, n_samples))
        regime_labels = np.random.choice([0, 1, 2], size=n_samples, p=[0.4, 0.4, 0.2])
        
        # Create advanced evaluator
        evaluator = AdvancedEvaluator()
        
        # Calculate metrics
        logger.info("   Calculating advanced metrics...")
        result = evaluator.evaluate(
            predictions=returns,  # Using returns as predictions for demo
            targets=returns,
            returns=returns,
            regime_labels=regime_labels
        )
        
        if result.success:
            logger.info(f"   Sharpe Ratio: {result.sharpe_ratio:.4f}")
            logger.info(f"   Sortino Ratio: {result.sortino_ratio:.4f}")
            logger.info(f"   Max Drawdown: {result.max_drawdown:.4f}")
            logger.info(f"   Hit Rate: {result.hit_rate:.4f}")
            logger.info(f"   Payoff Ratio: {result.payoff_ratio:.4f}")
            logger.info(f"   Multi-objective Score: {result.multi_objective_score:.4f}")
        else:
            logger.warning(f"   Advanced metrics failed: {result.error_message}")
            
    except Exception as e:
        logger.warning(f"   ⚠️ Advanced metrics demonstration failed: {e}")


def demonstrate_enhanced_tas():
    """Demonstrate complete Enhanced TAS pipeline."""
    logger.info("🚀 Demonstrating Enhanced TAS Pipeline...")
    
    if not ENHANCED_TAS_AVAILABLE:
        logger.warning("⚠️ Enhanced TAS not available")
        return
    
    try:
        # Create sample data
        X, y, regime_labels, feature_names = create_sample_data(1000, 20, "classification")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        # Create Enhanced TAS configuration
        config = EnhancedTASConfig(
            model_types=["xgboost", "lightgbm", "catboost"],
            enable_automl=True,
            enable_evolutionary_search=True,
            enable_advanced_metrics=True,
            enable_feature_engineering=True,
            enable_ensemble=True,
            max_search_time=300,  # 5 minutes for demo
            verbose=True
        )
        
        # Create Enhanced TAS engine
        engine = EnhancedTASEngine(config)
        
        # Run search
        logger.info("   Running Enhanced TAS search...")
        start_time = time.time()
        
        result = engine.search(
            X_train, y_train, X_val, y_val, X_test, y_test, regime_labels
        )
        
        search_time = time.time() - start_time
        
        # Display results
        if result.success:
            logger.info("✅ Enhanced TAS search completed successfully!")
            logger.info(f"   Best score: {result.best_score:.4f}")
            logger.info(f"   Search time: {search_time:.2f}s")
            logger.info(f"   Total evaluations: {result.total_evaluations}")
            logger.info(f"   Successful evaluations: {result.successful_evaluations}")
            logger.info(f"   Model rankings: {result.model_rankings}")
            
            if result.automl_result:
                logger.info(f"   AutoML best model: {result.automl_result.best_model_type}")
                logger.info(f"   AutoML best score: {result.automl_result.best_score:.4f}")
            
            if result.evolutionary_result:
                logger.info(f"   Evolutionary Pareto front size: {len(result.evolutionary_result.pareto_front)}")
            
            if result.advanced_evaluation:
                logger.info(f"   Advanced evaluation score: {result.advanced_evaluation.multi_objective_score:.4f}")
        else:
            logger.warning(f"❌ Enhanced TAS search failed: {result.error_message}")
            
    except Exception as e:
        logger.warning(f"   ⚠️ Enhanced TAS demonstration failed: {e}")


def create_visualization(result: EnhancedTASResult):
    """Create visualizations for Enhanced TAS results."""
    try:
        if not result.success:
            logger.warning("⚠️ Cannot create visualization for failed result")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Enhanced TAS Results', fontsize=16, fontweight='bold')
        
        # 1. Model Performance Comparison
        if result.model_rankings:
            models, scores = zip(*result.model_rankings)
            axes[0, 0].bar(models, scores, color='skyblue', alpha=0.7)
            axes[0, 0].set_title('Model Performance Comparison')
            axes[0, 0].set_xlabel('Model Type')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Search History
        if result.search_history:
            generations = list(range(len(result.search_history)))
            scores = [step.get('best_score', 0) for step in result.search_history]
            axes[0, 1].plot(generations, scores, marker='o', linewidth=2, markersize=6)
            axes[0, 1].set_title('Search Progress')
            axes[0, 1].set_xlabel('Generation')
            axes[0, 1].set_ylabel('Best Score')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Feature Importance
        if result.feature_importance:
            features = list(result.feature_importance.keys())
            importance = list(result.feature_importance.values())
            # Sort by importance
            sorted_data = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)
            features, importance = zip(*sorted_data[:10])  # Top 10 features
            
            axes[1, 0].barh(features, importance, color='lightcoral', alpha=0.7)
            axes[1, 0].set_title('Top 10 Feature Importance')
            axes[1, 0].set_xlabel('Importance')
        
        # 4. Multi-objective Results
        if result.pareto_front:
            objectives = ['accuracy', 'robustness', 'efficiency', 'interpretability']
            pareto_data = np.array([[ind.objectives[i] for i in range(len(objectives))] 
                                  for ind in result.pareto_front])
            
            # Create parallel coordinates plot
            for i, obj in enumerate(objectives):
                axes[1, 1].plot(pareto_data[:, i], label=obj, marker='o', alpha=0.7)
            axes[1, 1].set_title('Multi-objective Optimization')
            axes[1, 1].set_xlabel('Solution Index')
            axes[1, 1].set_ylabel('Objective Value')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Pareto front available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Multi-objective Optimization')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        logger.warning(f"⚠️ Visualization creation failed: {e}")


def main():
    """Main demonstration function."""
    logger.info("🎯 Enhanced TAS Demonstration")
    logger.info("=" * 50)
    
    # Check component availability
    logger.info("🔍 Checking component availability...")
    logger.info(f"   Enhanced TAS: {'✅' if ENHANCED_TAS_AVAILABLE else '❌'}")
    logger.info(f"   Enhanced Models: {'✅' if ENHANCED_MODELS_AVAILABLE else '❌'}")
    logger.info(f"   AutoML: {'✅' if AUTOML_AVAILABLE else '❌'}")
    logger.info(f"   Advanced Metrics: {'✅' if ADVANCED_METRICS_AVAILABLE else '❌'}")
    logger.info(f"   Evolutionary Search: {'✅' if EVOLUTIONARY_AVAILABLE else '❌'}")
    
    # Run demonstrations
    logger.info("\n" + "=" * 50)
    demonstrate_enhanced_models()
    
    logger.info("\n" + "=" * 50)
    demonstrate_automl()
    
    logger.info("\n" + "=" * 50)
    demonstrate_evolutionary_search()
    
    logger.info("\n" + "=" * 50)
    demonstrate_advanced_metrics()
    
    logger.info("\n" + "=" * 50)
    demonstrate_enhanced_tas()
    
    logger.info("\n" + "=" * 50)
    logger.info("🎉 Enhanced TAS demonstration completed!")
    
    # Note: Visualization would be created here if matplotlib is available
    # create_visualization(result)


if __name__ == "__main__":
    main()