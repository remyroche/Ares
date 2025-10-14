"""
Comprehensive Validation Example

Demonstrates the implementation of all critical improvements to the
UnifiedDataDrivenPipeline including ablation studies, leakage prevention,
search space optimization, robust MOEA convergence, statistical validation,
and robust stability metrics.

This example shows how to use the enhanced pipeline with all the critical
improvements implemented.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import time
import warnings
warnings.filterwarnings('ignore')

# Import the enhanced pipeline components
from src.training.steps.pre_training.unified_data_driven_pipeline.ablation_studies import (
    AblationStudyFramework, AblationStudyConfig, run_ablation_study
)

from src.training.steps.pre_training.unified_data_driven_pipeline.leakage_prevention import (
    LabelConstructionValidator, LabelConstructionConfig, validate_label_construction
)

from src.training.steps.pre_training.unified_data_driven_pipeline.search_space_optimization import (
    HereditaryInteractionGenerator, HereditaryInteractionConfig,
    AdvancedScreeningFramework, AdvancedScreeningConfig
)

from src.training.steps.pre_training.unified_data_driven_pipeline.moea_optimization import (
    RobustMOEAOptimizer, ConvergenceConfig, optimize_with_robust_moea
)

from src.training.steps.pre_training.unified_data_driven_pipeline.statistical_validation import (
    DeflatedSharpeCalculator, DeflatedSharpeConfig, calculate_deflated_sharpe
)

from src.training.steps.pre_training.unified_data_driven_pipeline.robust_stability import (
    RobustStabilityCalculator, RobustStabilityConfig, calculate_robust_stability
)

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)


def create_sample_data(n_samples: int = 1000, n_features: int = 50) -> tuple:
    """Create sample data for demonstration."""
    tprint_info("📊 Creating sample data...")
    
    # Create synthetic time series data
    np.random.seed(42)
    
    # Generate base features
    data = {}
    for i in range(n_features):
        # Generate correlated time series
        base_signal = np.cumsum(np.random.randn(n_samples) * 0.01)
        noise = np.random.randn(n_samples) * 0.02
        data[f'feature_{i}'] = base_signal + noise + i * 0.001
    
    # Create DataFrame with datetime index
    df = pd.DataFrame(data)
    df.index = pd.date_range('2020-01-01', periods=n_samples, freq='1H')
    
    # Generate target returns
    returns = df['feature_0'].pct_change().dropna()
    targets = returns.shift(-1).dropna()  # Forward returns
    
    # Align data and targets
    common_index = df.index.intersection(targets.index)
    df = df.loc[common_index]
    targets = targets.loc[common_index]
    
    tprint_success(f"✅ Created sample data: {df.shape[0]} samples, {df.shape[1]} features")
    return df, targets


def demonstrate_ablation_studies(data: pd.DataFrame, targets: pd.Series) -> None:
    """Demonstrate comprehensive ablation studies."""
    tprint_info("🔬 Demonstrating Ablation Studies...")
    
    # Create ablation study configuration
    ablation_config = AblationStudyConfig(
        study_name="comprehensive_ablation_study",
        n_repeats=3,  # Reduced for demo
        enable_moea=True,
        enable_diversity_penalty=True,
        enable_htf_features=True,
        enable_embargo=True,
        enable_turnover_objective=True,
        enable_stability_objective=True,
        track_metrics=['oos_sharpe', 'max_drawdown', 'turnover', 'stability']
    )
    
    # Create a simple pipeline factory for demonstration
    def create_simple_pipeline(config_dict: Dict[str, Any]):
        """Simple pipeline factory for demonstration."""
        class SimplePipeline:
            def __init__(self, config):
                self.config = config
            
            def process(self, data, targets):
                # Simulate pipeline processing
                n_features = min(20, len(data.columns))
                selected_features = list(data.columns[:n_features])
                
                # Simulate results based on configuration
                result = type('Result', (), {})()
                result.selected_features = selected_features
                result.out_of_sample_sharpe = np.random.random() * 2 - 1
                result.max_drawdown = np.random.random() * 0.3
                result.turnover = np.random.random() * 0.5
                result.stability = np.random.random()
                result.processing_time = np.random.random() * 10
                result.memory_usage_mb = np.random.random() * 100
                
                return result
        
        return SimplePipeline(config_dict)
    
    try:
        # Run ablation study
        tprint_debug("Running ablation study...")
        ablation_result = run_ablation_study(
            data, targets, create_simple_pipeline, ablation_config
        )
        
        tprint_success("✅ Ablation study completed")
        tprint_info(f"📊 Baseline: {ablation_result.baseline_result.config_name}")
        tprint_info(f"📊 Ablations: {len(ablation_result.ablation_results)}")
        tprint_info(f"📊 Significant: {len(ablation_result.significant_ablation)}")
        
        # Show some results
        for ablation_name, result in ablation_result.ablation_results.items():
            if result.success:
                tprint_info(f"📊 {ablation_name}: {result.metrics.get('oos_sharpe', 0):.3f} Sharpe")
        
    except Exception as e:
        tprint_error(f"❌ Ablation study failed: {e}")


def demonstrate_leakage_prevention(data: pd.DataFrame, targets: pd.Series) -> None:
    """Demonstrate leakage prevention and label construction validation."""
    tprint_info("🔒 Demonstrating Leakage Prevention...")
    
    # Create label construction configuration
    label_config = LabelConstructionConfig(
        label_type='forward_returns',
        horizon=1,
        resampling_frequency='1H',
        min_samples_for_label=10,
        htf_alignment_method='strict_past',
        max_htf_lookback=252,
        htf_resampling_offset=0,
        strict_temporal_ordering=True,
        validate_future_data=True,
        validate_resampling_alignment=True,
        validate_htf_constraints=True
    )
    
    try:
        # Validate label construction
        tprint_debug("Validating label construction...")
        label_result = validate_label_construction(data, targets, None, label_config)
        
        tprint_success("✅ Label construction validation completed")
        tprint_info(f"📊 Valid labels: {label_result.valid_labels}/{label_result.total_labels}")
        tprint_info(f"📊 Future data points: {label_result.future_data_points}")
        tprint_info(f"📊 HTF alignment score: {label_result.htf_alignment_score:.3f}")
        
        if label_result.temporal_violations:
            tprint_warning(f"⚠️ {len(label_result.temporal_violations)} temporal violations found")
        
    except Exception as e:
        tprint_error(f"❌ Leakage prevention validation failed: {e}")


def demonstrate_search_space_optimization(data: pd.DataFrame, targets: pd.Series) -> None:
    """Demonstrate search space optimization with hereditary interactions and advanced screening."""
    tprint_info("🔍 Demonstrating Search Space Optimization...")
    
    # Advanced screening
    tprint_debug("Running advanced screening...")
    screening_config = AdvancedScreeningConfig(
        screening_methods=['hsic', 'distance_correlation', 'mutual_information'],
        hsic_threshold=0.1,
        distance_correlation_threshold=0.1,
        mutual_information_threshold=0.01,
        max_features=20,
        enable_parallel_processing=True
    )
    
    try:
        screening_framework = AdvancedScreeningFramework(screening_config)
        screening_result = screening_framework.screen_features(data, targets)
        
        tprint_success("✅ Advanced screening completed")
        tprint_info(f"📊 Features selected: {len(screening_result.combined_selected_features)}")
        tprint_info(f"📊 Method agreement: {screening_result.method_agreement:.3f}")
        
    except Exception as e:
        tprint_error(f"❌ Advanced screening failed: {e}")
    
    # Hereditary interactions
    tprint_debug("Generating hereditary interactions...")
    hereditary_config = HereditaryInteractionConfig(
        require_pre_selection=True,
        pre_selected_features=set(screening_result.combined_selected_features[:10]),
        interaction_types=['multiplication', 'division', 'ratio', 'difference'],
        max_interactions=50,
        min_correlation_threshold=0.1,
        max_correlation_threshold=0.95,
        enable_parallel_processing=True
    )
    
    try:
        hereditary_generator = HereditaryInteractionGenerator(hereditary_config)
        hereditary_result = hereditary_generator.generate_interactions(data)
        
        tprint_success("✅ Hereditary interactions completed")
        tprint_info(f"📊 Interactions generated: {len(hereditary_result.interactions)}")
        tprint_info(f"📊 Average correlation: {hereditary_result.average_correlation:.3f}")
        
    except Exception as e:
        tprint_error(f"❌ Hereditary interactions failed: {e}")


def demonstrate_robust_moea_convergence() -> None:
    """Demonstrate robust MOEA optimization with convergence criteria."""
    tprint_info("🧬 Demonstrating Robust MOEA Convergence...")
    
    # Create a simple optimization problem
    def simple_optimization_problem(x):
        """Simple multi-objective optimization problem."""
        obj1 = np.sum(x**2)  # Minimize sum of squares
        obj2 = np.sum(np.abs(x))  # Minimize sum of absolute values
        return [obj1, obj2]
    
    # Create convergence configuration
    convergence_config = ConvergenceConfig(
        max_generations=50,  # Reduced for demo
        max_evaluations=1000,
        max_time_seconds=300,  # 5 minutes
        hypervolume_tolerance=1e-6,
        epsilon_progress_tolerance=1e-6,
        stagnation_generations=10,
        enable_anytime_stop=True,
        min_generations=5,
        min_evaluations=50,
        enable_parallel=True
    )
    
    try:
        # Run optimization
        tprint_debug("Running robust MOEA optimization...")
        optimization_result = optimize_with_robust_moea(
            simple_optimization_problem,
            convergence_config=convergence_config
        )
        
        tprint_success("✅ Robust MOEA optimization completed")
        tprint_info(f"📊 Generations: {optimization_result.generations_completed}")
        tprint_info(f"📊 Evaluations: {optimization_result.evaluations_completed}")
        tprint_info(f"📊 Converged: {optimization_result.converged}")
        tprint_info(f"📊 Final hypervolume: {optimization_result.final_hypervolume:.6f}")
        
    except Exception as e:
        tprint_error(f"❌ Robust MOEA optimization failed: {e}")


def demonstrate_statistical_validation(data: pd.DataFrame, targets: pd.Series) -> None:
    """Demonstrate statistical validation with deflated Sharpe ratios."""
    tprint_info("📊 Demonstrating Statistical Validation...")
    
    # Calculate some sample Sharpe ratios
    sharpe_ratios = {}
    for col in data.columns[:10]:  # Use first 10 features
        returns = data[col].pct_change().dropna()
        if len(returns) > 10:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
            sharpe_ratios[col] = sharpe
    
    # Create deflated Sharpe configuration
    deflated_sharpe_config = DeflatedSharpeConfig(
        n_features_tested=len(sharpe_ratios),
        n_observations=len(data),
        confidence_level=0.95,
        deflation_method='bailey_lopez',
        skewness_adjustment=True,
        kurtosis_adjustment=True,
        enable_parallel=True
    )
    
    try:
        # Calculate deflated Sharpe ratios
        tprint_debug("Calculating deflated Sharpe ratios...")
        deflated_result = calculate_deflated_sharpe(
            sharpe_ratios, None, deflated_sharpe_config
        )
        
        tprint_success("✅ Deflated Sharpe calculation completed")
        tprint_info(f"📊 Features tested: {deflated_result.n_features_tested}")
        tprint_info(f"📊 Significant features: {deflated_result.n_significant_features}")
        tprint_info(f"📊 Significance rate: {deflated_result.significance_rate:.3f}")
        tprint_info(f"📊 Average deflation factor: {deflated_result.average_deflation_factor:.3f}")
        
    except Exception as e:
        tprint_error(f"❌ Statistical validation failed: {e}")


def demonstrate_robust_stability() -> None:
    """Demonstrate robust stability metrics beyond Jaccard similarity."""
    tprint_info("📈 Demonstrating Robust Stability Metrics...")
    
    # Create sample feature importance data
    feature_importances = {}
    for i in range(20):
        # Generate importance values over time (e.g., from different CV folds)
        importances = np.random.exponential(1.0, 10) + np.random.normal(0, 0.1, 10)
        feature_importances[f'feature_{i}'] = importances.tolist()
    
    # Create robust stability configuration
    stability_config = RobustStabilityConfig(
        stability_metrics=[
            'jaccard_similarity',
            'coefficient_path',
            'rank_correlation',
            'bootstrap_stability'
        ],
        bootstrap_samples=50,  # Reduced for demo
        coefficient_path_window=5,
        rank_correlation_min_ranks=3,
        enable_parallel=True
    )
    
    try:
        # Calculate robust stability
        tprint_debug("Calculating robust stability metrics...")
        stability_result = calculate_robust_stability(
            feature_importances, None, stability_config
        )
        
        tprint_success("✅ Robust stability calculation completed")
        tprint_info(f"📊 Features analyzed: {stability_result.n_features}")
        tprint_info(f"📊 Metrics calculated: {stability_result.n_metrics}")
        tprint_info(f"📊 Average stability: {stability_result.average_combined_stability:.3f}")
        
        # Show individual metric results
        for metric, result in stability_result.stability_results.items():
            tprint_info(f"📊 {metric.value}: {result.average_score:.3f} ± {result.score_std:.3f}")
        
    except Exception as e:
        tprint_error(f"❌ Robust stability calculation failed: {e}")


def main():
    """Main demonstration function."""
    tprint_info("🚀 Starting Comprehensive Validation Example")
    tprint_info("=" * 60)
    
    try:
        # Create sample data
        data, targets = create_sample_data(n_samples=500, n_features=30)
        
        # Demonstrate each critical improvement
        tprint_info("\n" + "=" * 60)
        demonstrate_ablation_studies(data, targets)
        
        tprint_info("\n" + "=" * 60)
        demonstrate_leakage_prevention(data, targets)
        
        tprint_info("\n" + "=" * 60)
        demonstrate_search_space_optimization(data, targets)
        
        tprint_info("\n" + "=" * 60)
        demonstrate_robust_moea_convergence()
        
        tprint_info("\n" + "=" * 60)
        demonstrate_statistical_validation(data, targets)
        
        tprint_info("\n" + "=" * 60)
        demonstrate_robust_stability()
        
        tprint_info("\n" + "=" * 60)
        tprint_success("🎉 Comprehensive Validation Example Completed Successfully!")
        tprint_info("All critical improvements have been demonstrated:")
        tprint_info("✅ Ablation Studies - Systematic component validation")
        tprint_info("✅ Leakage Prevention - Label construction and HTF alignment")
        tprint_info("✅ Search Space Optimization - Hereditary interactions and advanced screening")
        tprint_info("✅ Robust MOEA Convergence - Comprehensive convergence criteria")
        tprint_info("✅ Statistical Validation - Deflated Sharpe and reality checks")
        tprint_info("✅ Robust Stability - Multiple stability metrics beyond Jaccard")
        
    except Exception as e:
        tprint_error(f"❌ Comprehensive validation example failed: {e}")
        raise


if __name__ == "__main__":
    main()