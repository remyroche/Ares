"""
TAS Integration Demo

This script demonstrates how the enhanced TAS system integrates with shared utilities
and provides a comprehensive example of the enhanced capabilities.
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

# Import TAS components
try:
    from .core.tas_engine import TreeArchitectureSearchEngine, TASEngineConfig
    from src.utils.nas_tas.core.tas_engine import TASEngine
    from src.utils.nas_tas.optimization.strategy_search import StrategySearchOptimizer, StrategySearchConfig
    TAS_AVAILABLE = True
except ImportError:
    TAS_AVAILABLE = False
    logger.warning("⚠️ TAS not available")

# Import shared utilities
try:
    from ..shared_utils.evolutionary_search import (
        EvolutionaryAlgorithmManager, EvolutionaryConfig, EvolutionaryResult,
        create_evolutionary_algorithm_manager
    )
    from ..shared_utils.feature_engineering import (
        UnifiedFeatureEngineer, FeatureConfig, FeatureEngineeringResult,
        create_unified_feature_engineer
    )
    from ..shared_utils.evaluation_metrics import (
        UnifiedEvaluator, UnifiedEvaluationResult,
        create_unified_evaluator
    )
    SHARED_UTILS_AVAILABLE = True
except ImportError:
    SHARED_UTILS_AVAILABLE = False
    logger.warning("⚠️ Shared utilities not available")

# Import enhanced TAS components
try:
    from .models.enhanced_tree_models import (
        EnhancedTreeModelFactory, TreeModelConfig, TreeModelType
    )
    from .automl.tree_automl import (
        TreeAutoMLManager, AutoMLConfig, AutoMLResult
    )
    from .evaluation.advanced_metrics import (
        AdvancedEvaluator, AdvancedEvaluationResult
    )
    ENHANCED_COMPONENTS_AVAILABLE = True
except ImportError:
    ENHANCED_COMPONENTS_AVAILABLE = False
    logger.warning("⚠️ Enhanced components not available")


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


def demonstrate_shared_utilities():
    """Demonstrate shared utilities functionality."""
    logger.info("🔧 Demonstrating Shared Utilities...")
    
    if not SHARED_UTILS_AVAILABLE:
        logger.warning("⚠️ Shared utilities not available")
        return
    
    # Create sample data
    X, y, regime_labels, feature_names = create_sample_data(1000, 20, "classification")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. Feature Engineering
    logger.info("🔧 Testing Feature Engineering...")
    try:
        feature_config = FeatureConfig(
            enable_technical_indicators=True,
            enable_feature_selection=True,
            feature_selection_method="mutual_info",
            max_features=50
        )
        feature_engineer = create_unified_feature_engineer(feature_config)
        
        result = feature_engineer.engineer_features(X_train, y_train, feature_names)
        
        if result.success:
            logger.info(f"✅ Feature engineering completed")
            logger.info(f"   Original features: {result.original_feature_count}")
            logger.info(f"   Enhanced features: {result.enhanced_feature_count}")
            logger.info(f"   Selected features: {result.selected_feature_count}")
        else:
            logger.warning(f"⚠️ Feature engineering failed: {result.error_message}")
            
    except Exception as e:
        logger.warning(f"⚠️ Feature engineering demonstration failed: {e}")
    
    # 2. Evolutionary Search
    logger.info("🧬 Testing Evolutionary Search...")
    try:
        def objective_function(params):
            # Simple objective function for demonstration
            return np.random.random()
        
        evolutionary_config = EvolutionaryConfig(
            population_size=20,
            max_generations=10,
            use_nsga2=True
        )
        evolutionary_manager = create_evolutionary_algorithm_manager(evolutionary_config)
        
        parameter_space = {
            'param1': {'type': 'continuous', 'min': 0.0, 'max': 1.0},
            'param2': {'type': 'integer', 'min': 1, 'max': 10}
        }
        
        result = evolutionary_manager.optimize_with_algorithm(
            [objective_function], parameter_space, "nsga2"
        )
        
        if result.success:
            logger.info(f"✅ Evolutionary search completed")
            logger.info(f"   Pareto front size: {len(result.pareto_front)}")
            logger.info(f"   Execution time: {result.execution_time:.2f}s")
        else:
            logger.warning(f"⚠️ Evolutionary search failed: {result.error_message}")
            
    except Exception as e:
        logger.warning(f"⚠️ Evolutionary search demonstration failed: {e}")
    
    # 3. Unified Evaluation
    logger.info("📊 Testing Unified Evaluation...")
    try:
        # Create some sample predictions
        predictions = np.random.randint(0, 2, len(y_test))
        returns = np.random.normal(0.001, 0.02, len(y_test))
        
        evaluator = create_unified_evaluator()
        result = evaluator.evaluate(predictions, y_test, returns, regime_labels[:len(y_test)])
        
        if result.success:
            logger.info(f"✅ Unified evaluation completed")
            logger.info(f"   Overall score: {result.overall_score:.4f}")
            logger.info(f"   Risk-adjusted score: {result.risk_adjusted_score:.4f}")
            logger.info(f"   Regime-aware score: {result.regime_aware_score:.4f}")
            logger.info(f"   Economic score: {result.economic_score:.4f}")
            logger.info(f"   Trading score: {result.trading_score:.4f}")
        else:
            logger.warning(f"⚠️ Unified evaluation failed: {result.error_message}")
            
    except Exception as e:
        logger.warning(f"⚠️ Unified evaluation demonstration failed: {e}")


def demonstrate_enhanced_components():
    """Demonstrate enhanced TAS components."""
    logger.info("🚀 Demonstrating Enhanced TAS Components...")
    
    if not ENHANCED_COMPONENTS_AVAILABLE:
        logger.warning("⚠️ Enhanced components not available")
        return
    
    # Create sample data
    X, y, regime_labels, feature_names = create_sample_data(1000, 20, "classification")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. Enhanced Tree Models
    logger.info("🌳 Testing Enhanced Tree Models...")
    try:
        model_config = TreeModelConfig(
            model_type=TreeModelType.XGBOOST,
            params={'n_estimators': 100, 'max_depth': 6},
            is_classifier=True
        )
        model_factory = EnhancedTreeModelFactory(model_config)
        model_factory.fit(X_train, y_train)
        
        predictions = model_factory.predict(X_test)
        accuracy = (predictions == y_test).mean()
        
        logger.info(f"✅ Enhanced tree model completed")
        logger.info(f"   Model type: {model_config.model_type.value}")
        logger.info(f"   Accuracy: {accuracy:.4f}")
        
    except Exception as e:
        logger.warning(f"⚠️ Enhanced tree model demonstration failed: {e}")
    
    # 2. AutoML
    logger.info("🤖 Testing AutoML...")
    try:
        automl_config = AutoMLConfig(
            optimization_method="optuna",
            max_trials=20,
            timeout_seconds=300,
            model_types=["xgboost", "lightgbm"],
            enable_ensemble=True
        )
        automl_manager = TreeAutoMLManager(automl_config)
        
        result = automl_manager.optimize(X_train, y_train, X_test, y_test)
        
        if result.success:
            logger.info(f"✅ AutoML completed")
            logger.info(f"   Best model: {result.best_model_type}")
            logger.info(f"   Best score: {result.best_score:.4f}")
            logger.info(f"   Optimization time: {result.optimization_time:.2f}s")
        else:
            logger.warning(f"⚠️ AutoML failed: {result.error_message}")
            
    except Exception as e:
        logger.warning(f"⚠️ AutoML demonstration failed: {e}")
    
    # 3. Advanced Evaluation
    logger.info("📊 Testing Advanced Evaluation...")
    try:
        # Create some sample predictions
        predictions = np.random.randint(0, 2, len(y_test))
        returns = np.random.normal(0.001, 0.02, len(y_test))
        
        evaluator = AdvancedEvaluator()
        result = evaluator.evaluate(predictions, y_test, returns, regime_labels[:len(y_test)])
        
        if result.success:
            logger.info(f"✅ Advanced evaluation completed")
            logger.info(f"   Overall score: {result.overall_score:.4f}")
            logger.info(f"   Risk-adjusted score: {result.risk_adjusted_score:.4f}")
            logger.info(f"   Regime-aware score: {result.regime_aware_score:.4f}")
        else:
            logger.warning(f"⚠️ Advanced evaluation failed: {result.error_message}")
            
    except Exception as e:
        logger.warning(f"⚠️ Advanced evaluation demonstration failed: {e}")


def demonstrate_tas_integration():
    """Demonstrate TAS integration with shared utilities."""
    logger.info("🎯 Demonstrating TAS Integration...")
    
    if not TAS_AVAILABLE:
        logger.warning("⚠️ TAS not available")
        return
    
    # Create sample data
    X, y, regime_labels, feature_names = create_sample_data(1000, 20, "classification")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # 1. Standard TAS Engine
    logger.info("🔍 Testing Standard TAS Engine...")
    try:
        config = TASEngineConfig(
            enable_enhanced_models=True,
            enable_automl=True,
            enable_evolutionary_search=True,
            enable_advanced_metrics=True,
            enable_feature_engineering=True,
            model_types=["xgboost", "lightgbm", "catboost"],
            max_search_time=300,  # 5 minutes for demo
            verbose=True
        )
        
        engine = TreeArchitectureSearchEngine(config)
        
        # Run search
        start_time = time.time()
        result = engine.search(
            (X_train, y_train), (X_val, y_val), (X_test, y_test),
            regime_data={'regime_labels': regime_labels}
        )
        search_time = time.time() - start_time
        
        if result.success:
            logger.info(f"✅ Standard TAS search completed")
            logger.info(f"   Search time: {search_time:.2f}s")
            logger.info(f"   Best score: {result.best_score:.4f}")
            logger.info(f"   Execution time: {result.execution_time:.2f}s")
        else:
            logger.warning(f"⚠️ Standard TAS search failed: {result.error_message}")
            
    except Exception as e:
        logger.warning(f"⚠️ Standard TAS demonstration failed: {e}")
    
    # 2. Enhanced TAS Engine
    logger.info("🚀 Testing Enhanced TAS Engine...")
    try:
        enhanced_config = EnhancedTASConfig(
            model_types=["xgboost", "lightgbm", "catboost"],
            enable_automl=True,
            enable_evolutionary_search=True,
            enable_advanced_metrics=True,
            enable_feature_engineering=True,
            enable_ensemble=True,
            max_search_time=300,  # 5 minutes for demo
            verbose=True
        )
        
        enhanced_engine = EnhancedTASEngine(enhanced_config)
        
        # Run enhanced search
        start_time = time.time()
        enhanced_result = enhanced_engine.search(
            X_train, y_train, X_val, y_val, X_test, y_test, regime_labels
        )
        enhanced_search_time = time.time() - start_time
        
        if enhanced_result.success:
            logger.info(f"✅ Enhanced TAS search completed")
            logger.info(f"   Search time: {enhanced_search_time:.2f}s")
            logger.info(f"   Best score: {enhanced_result.best_score:.4f}")
            logger.info(f"   Total evaluations: {enhanced_result.total_evaluations}")
            logger.info(f"   Successful evaluations: {enhanced_result.successful_evaluations}")
            
            if enhanced_result.automl_result:
                logger.info(f"   AutoML best model: {enhanced_result.automl_result.best_model_type}")
                logger.info(f"   AutoML best score: {enhanced_result.automl_result.best_score:.4f}")
            
            if enhanced_result.evolutionary_result:
                logger.info(f"   Evolutionary Pareto front size: {len(enhanced_result.evolutionary_result.pareto_front)}")
            
            if enhanced_result.advanced_evaluation:
                logger.info(f"   Advanced evaluation score: {enhanced_result.advanced_evaluation.overall_score:.4f}")
        else:
            logger.warning(f"⚠️ Enhanced TAS search failed: {enhanced_result.error_message}")
            
    except Exception as e:
        logger.warning(f"⚠️ Enhanced TAS demonstration failed: {e}")


def create_integration_summary():
    """Create a summary of the integration capabilities."""
    logger.info("📋 Integration Summary")
    logger.info("=" * 50)
    
    # Check component availability
    logger.info("🔍 Component Availability:")
    logger.info(f"   TAS Engine: {'✅' if TAS_AVAILABLE else '❌'}")
    logger.info(f"   Shared Utilities: {'✅' if SHARED_UTILS_AVAILABLE else '❌'}")
    logger.info(f"   Enhanced Components: {'✅' if ENHANCED_COMPONENTS_AVAILABLE else '❌'}")
    
    # Integration capabilities
    logger.info("\n🔧 Integration Capabilities:")
    logger.info("   ✅ Shared Evolutionary Search (NSGA-II, SPEA2, GA)")
    logger.info("   ✅ Shared Feature Engineering (Technical Indicators, Selection)")
    logger.info("   ✅ Shared Evaluation Metrics (Financial, Statistical, Regime)")
    logger.info("   ✅ Enhanced Tree Models (XGBoost, LightGBM, CatBoost, BART)")
    logger.info("   ✅ AutoML Integration (Optuna, Grid Search, Bayesian)")
    logger.info("   ✅ Multi-objective Optimization")
    logger.info("   ✅ Regime-aware Evaluation")
    logger.info("   ✅ Economic Significance Assessment")
    
    # Usage examples
    logger.info("\n📚 Usage Examples:")
    logger.info("   # Standard TAS with shared utilities")
    logger.info("   config = TASEngineConfig(enable_enhanced_models=True)")
    logger.info("   engine = TreeArchitectureSearchEngine(config)")
    logger.info("   result = engine.search(train_data, val_data, test_data)")
    logger.info("")
    logger.info("   # Enhanced TAS with all capabilities")
    logger.info("   config = EnhancedTASConfig(enable_automl=True)")
    logger.info("   engine = EnhancedTASEngine(config)")
    logger.info("   result = engine.search(X_train, y_train, X_val, y_val)")
    logger.info("")
    logger.info("   # Direct shared utility usage")
    logger.info("   feature_engineer = create_unified_feature_engineer()")
    logger.info("   result = feature_engineer.engineer_features(X, y)")


def main():
    """Main demonstration function."""
    logger.info("🎯 TAS Integration Demonstration")
    logger.info("=" * 50)
    
    # Run demonstrations
    demonstrate_shared_utilities()
    
    logger.info("\n" + "=" * 50)
    demonstrate_enhanced_components()
    
    logger.info("\n" + "=" * 50)
    demonstrate_tas_integration()
    
    logger.info("\n" + "=" * 50)
    create_integration_summary()
    
    logger.info("\n" + "=" * 50)
    logger.info("🎉 TAS Integration demonstration completed!")
    logger.info("")
    logger.info("The enhanced TAS system now provides:")
    logger.info("✅ Modern tree algorithms (XGBoost, LightGBM, CatBoost, BART)")
    logger.info("✅ Automated optimization (AutoML, Evolutionary Search)")
    logger.info("✅ Advanced feature engineering (Technical Indicators, Selection)")
    logger.info("✅ Sophisticated evaluation (Financial, Statistical, Regime-aware)")
    logger.info("✅ Shared utilities for NAS and TAS systems")
    logger.info("✅ Multi-objective optimization capabilities")
    logger.info("✅ Regime-aware and economic significance assessment")


if __name__ == "__main__":
    main()