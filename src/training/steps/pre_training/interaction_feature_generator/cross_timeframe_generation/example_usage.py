"""
Example Usage of Cross-Timeframe Feature Generation System

This example demonstrates how to use the complete cross-timeframe feature generation
and optimization pipeline for generating and optimizing higher temporal frequency (HTF) features.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = next(parent for parent in CURRENT_DIR.parents if (parent / "src").exists())

if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.tprint import tprint

from pipeline import CrossTimeframePipeline, PipelineConfig
from phase1_probe import Phase1HTFProbe
from phase2_optimization import Phase2Optimization
from regime_segmentation import RegimeSegmentation
from scoring_system import AdaptiveScoringSystem
from ehu_rih_assignment import EHU_RIH_Assignment
from knapsack_selection import KnapsackSelection
from htf_materialization import HTFMaterialization
from interaction_templates import HTFInteractionTemplates
from statistical_selection import StatisticalSelection
from evaluation import WalkForwardEvaluation
from monitoring import MonitoringSystem


def create_sample_data(n_samples: int = 10000) -> pd.DataFrame:
    """Create sample OHLCV data for demonstration."""
    tprint(f"🧪 Generating sample OHLCV data with {n_samples} rows...")
    np.random.seed(42)
    
    # Generate timestamps (5-minute intervals)
    start_time = datetime(2023, 1, 1, 9, 0)
    timestamps = [start_time + timedelta(minutes=5*i) for i in range(n_samples)]
    
    # Generate price data with trend and volatility
    returns = np.random.normal(0, 0.001, n_samples)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = []
    for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
        # Generate realistic OHLCV
        volatility = 0.002 + 0.001 * np.sin(i / 100)  # Time-varying volatility
        high = price * (1 + abs(np.random.normal(0, volatility)))
        low = price * (1 - abs(np.random.normal(0, volatility)))
        open_price = prices[i-1] if i > 0 else price
        close = price
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)

    tprint(
        "✅ Sample data generation complete",
        extra={
            "rows": len(df),
            "columns": list(df.columns),
            "start": df.index.min(),
            "end": df.index.max(),
        },
    )

    return df


def create_sample_targets(data: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Create sample target variables (log returns)."""
    tprint(
        "🧪 Creating sample targets",
        extra={"horizon": horizon, "rows": len(data)},
    )

    # Calculate log returns
    log_returns = np.log(data['close'] / data['close'].shift(1))

    # Forward-looking returns for prediction
    targets = log_returns.shift(-horizon)

    tprint(
        "✅ Sample targets created",
        extra={
            "min": float(targets.min()),
            "max": float(targets.max()),
            "non_null": int(targets.notna().sum()),
        },
    )

    return targets


def run_complete_pipeline_example():
    """Run the complete cross-timeframe pipeline example."""
    tprint("=" * 80)
    tprint("Cross-Timeframe Feature Generation Pipeline Example")
    tprint("=" * 80)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create sample data
    logger.info("Creating sample data...")
    ohlcv_data = create_sample_data(5000)
    targets = create_sample_targets(ohlcv_data, horizon=1)
    
    tprint(
        "📊 Sample dataset created",
        extra={
            "rows": len(ohlcv_data),
            "data_range": f"{ohlcv_data.index[0]} to {ohlcv_data.index[-1]}",
            "target_min": float(targets.min()),
            "target_max": float(targets.max()),
        },
    )
    
    # Configure pipeline
    config = PipelineConfig(
        base_timeframe_minutes=5,
        session_start_hour=9,
        session_end_hour=16,
        coarse_grid_min=15,
        coarse_grid_max=300,
        max_cost_ms=25.0,
        max_features=120,
        max_correlation=0.8,
        stability_resamples=50,  # Reduced for example
        fdr_q=0.1,
        walk_forward_folds=3,  # Reduced for example
        adaptive_penalties=True,
        dashboard_enabled=True
    )
    
    # Initialize pipeline
    logger.info("Initializing cross-timeframe pipeline...")
    pipeline = CrossTimeframePipeline(config)
    
    # Run complete pipeline
    logger.info("Running complete pipeline...")
    try:
        results = pipeline.run_pipeline(ohlcv_data, targets=targets)
        
        tprint("\n" + "=" * 60)
        tprint("PIPELINE RESULTS SUMMARY")
        tprint("=" * 60)
        
        # Phase-1 results
        phase1_results = results['phase1_results']
        tprint("\nPhase-1 HTF Probe:")
        tprint(f"  - Candidates evaluated: {len(phase1_results.get('candidates', []))}")
        tprint(f"  - Shortlisted candidates: {len(phase1_results.get('shortlisted_candidates', []))}")
        tprint(f"  - Early stopped families: {len(phase1_results.get('early_stopped_families', []))}")
        
        # Phase-2 results
        phase2_results = results['phase2_results']
        tprint("\nPhase-2 Optimization:")
        tprint(f"  - Optimized features: {len(phase2_results.get('optimized_features', []))}")
        tprint(f"  - Hierarchical results: {len(phase2_results.get('hierarchical_results', {}))}")
        
        # Knapsack selection
        selected_htfs = results['selected_htfs']
        tprint("\nKnapsack Selection (Stage 1):")
        tprint(f"  - Selected features: {len(selected_htfs.selected_features)}")
        tprint(f"  - Total utility: {selected_htfs.total_utility:.4f}")
        tprint(f"  - Total cost: {selected_htfs.total_cost:.2f} ms")
        tprint(f"  - Family coverage: {selected_htfs.family_coverage}")
        
        # Materialized HTFs
        materialized_htfs = results['materialized_htfs']
        tprint("\nHTF Materialization:")
        tprint(f"  - Materialized features: {len(materialized_htfs)}")
        
        # Interactions
        interactions = results['interactions']
        tprint("\nInteraction Generation:")
        tprint(f"  - Generated interactions: {len(interactions)}")
        
        # Final features
        final_features = results['final_features']
        tprint("\nStatistical Selection (Stage 2):")
        tprint(f"  - Final selected features: {len(final_features.selected_features)}")
        tprint(f"  - Selection method: {final_features.selection_method}")
        
        # Evaluation results
        evaluation_results = results['evaluation_results']
        tprint("\nWalk-Forward Evaluation:")
        tprint(f"  - Overall IC: {evaluation_results.overall_ic:.4f}")
        tprint(f"  - IC confidence interval: {evaluation_results.overall_ic_ci}")
        tprint(f"  - Number of folds: {len(evaluation_results.walk_forward_results)}")
        
        # Regime results
        if evaluation_results.regime_results:
            tprint("\nRegime-Specific Results:")
            for regime, metrics in evaluation_results.regime_results.items():
                ic_mean = metrics.get('ic_mean', 0.0)
                ic_count = metrics.get('ic_count', 0)
                tprint(f"  - {regime}: IC={ic_mean:.4f} (n={ic_count})")
        
        # Ablation results
        if evaluation_results.ablation_results:
            tprint("\nAblation Study Results:")
            for config_name, metrics in evaluation_results.ablation_results.items():
                ic = metrics.get('ic', 0.0)
                n_features = metrics.get('n_features', 0)
                tprint(f"  - {config_name}: IC={ic:.4f}, Features={n_features}")
        
        # SPA test
        if evaluation_results.spa_test_result:
            spa_result = evaluation_results.spa_test_result
            tprint("\nSPA Test:")
            tprint(f"  - Statistic: {spa_result.get('spa_statistic', 0.0):.4f}")
            tprint(f"  - P-value: {spa_result.get('p_value', 1.0):.4f}")
            tprint(f"  - Reject null: {spa_result.get('reject_null', False)}")

        tprint("\n" + "=" * 60)
        tprint("PIPELINE COMPLETED SUCCESSFULLY")
        tprint("=" * 60)
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


def run_individual_component_examples():
    """Run examples of individual components."""
    tprint("\n" + "=" * 80)
    tprint("Individual Component Examples")
    tprint("=" * 80)
    
    # Create sample data
    ohlcv_data = create_sample_data(1000)
    targets = create_sample_targets(ohlcv_data)
    
    # Example 1: Regime Segmentation
    tprint("\n1. Regime Segmentation Example")
    tprint("-" * 40)
    
    config = PipelineConfig()
    regime_segmentation = RegimeSegmentation(config)
    
    sessionized_data = {
        'aligned_data': ohlcv_data,
        'sessions': [],
        'base_timeframe': 5
    }
    
    regime_results = regime_segmentation.segment_regimes(sessionized_data, targets)
    tprint(f"  - Segments created: {len(regime_results['segments'])}")
    tprint(f"  - Change points detected: {len(regime_results['change_points'])}")
    
    # Example 2: Phase-1 HTF Probe
    tprint("\n2. Phase-1 HTF Probe Example")
    tprint("-" * 40)
    
    scoring_system = AdaptiveScoringSystem(config)
    phase1_probe = Phase1HTFProbe(config, scoring_system=scoring_system)
    phase1_results = phase1_probe.run_probe_stage(sessionized_data, regime_results, targets)
    tprint(f"  - Candidates evaluated: {len(phase1_results['candidates'])}")
    tprint(f"  - Shortlisted: {len(phase1_results['shortlisted_candidates'])}")
    
    # Example 3: EHU/RIH Assignment
    tprint("\n3. EHU/RIH Assignment Example")
    tprint("-" * 40)
    
    ehu_rih_assignment = EHU_RIH_Assignment(config)
    
    # Create mock phase2 results
    mock_phase2_results = {
        'optimized_features': [
            type('MockFeature', (), {
                'feature_name': 'test_feature',
                'family': 'trend_level_vol',
                'optimal_lookback': 60,
                'optimal_ic': 0.1,
                'optimal_se': 0.05
            })()
        ]
    }
    
    assignments = ehu_rih_assignment.assign_htf_features(mock_phase2_results, sessionized_data)
    tprint(f"  - Features assigned: {len(assignments)}")
    if assignments:
        tprint(f"  - EHU features: {sum(1 for a in assignments if a.update_style.value == 'ehu')}")
        tprint(f"  - RIH features: {sum(1 for a in assignments if a.update_style.value == 'rih')}")
    
    # Example 4: Monitoring System
    tprint("\n4. Monitoring System Example")
    tprint("-" * 40)
    
    monitoring = MonitoringSystem(config)
    
    # Create mock final features and evaluation results
    mock_final_features = ['feature1', 'feature2', 'feature3']
    mock_evaluation_summary = {
        'overall_ic': 0.12,
        'overall_ic_std': 0.03,
        'overall_ic_ci': (0.08, 0.16),
        'mean_sharpe': 1.05,
        'max_drawdown': -0.04,
        'feature_count': len(mock_final_features),
        'metadata': {'source': 'example_usage'}
    }

    monitoring.setup_monitoring(
        mock_final_features,
        mock_evaluation_summary
    )
    
    # Create mock performance metrics
    from monitoring import PerformanceMetrics
    mock_metrics = PerformanceMetrics(
        timestamp=datetime.now(),
        ic=0.15,
        ic_std=0.05,
        sharpe=1.2,
        max_drawdown=-0.05,
        feature_count=45,
        regime='low_vol',
        metadata={}
    )
    
    market_conditions = {
        'volatility_level': 0.6,
        'news_proximity': 0.2
    }
    
    update_result = monitoring.update_monitoring(mock_metrics, market_conditions)
    tprint(f"  - Monitoring update: {update_result['status']}")
    tprint(f"  - Alerts generated: {update_result['alerts_generated']}")
    
    # Get system status
    status = monitoring.get_system_status()
    tprint(f"  - System status: {status['status']}")
    tprint(f"  - Total metrics: {status['total_metrics']}")
    tprint(f"  - Total alerts: {status['total_alerts']}")


def demonstrate_configuration_options():
    """Demonstrate different configuration options."""
    tprint("\n" + "=" * 80)
    tprint("Configuration Options Examples")
    tprint("=" * 80)
    
    # Conservative configuration
    tprint("\n1. Conservative Configuration")
    tprint("-" * 40)
    
    conservative_config = PipelineConfig(
        base_timeframe_minutes=5,
        coarse_grid_min=30,
        coarse_grid_max=180,
        max_cost_ms=15.0,
        max_features=80,
        max_correlation=0.7,
        stability_resamples=100,
        fdr_q=0.05,
        walk_forward_folds=5,
        adaptive_penalties=True
    )
    
    tprint("Conservative settings:")
    tprint(f"  - Max cost: {conservative_config.max_cost_ms} ms")
    tprint(f"  - Max features: {conservative_config.max_features}")
    tprint(f"  - Max correlation: {conservative_config.max_correlation}")
    tprint(f"  - FDR threshold: {conservative_config.fdr_q}")
    
    # Aggressive configuration
    tprint("\n2. Aggressive Configuration")
    tprint("-" * 40)
    
    aggressive_config = PipelineConfig(
        base_timeframe_minutes=5,
        coarse_grid_min=15,
        coarse_grid_max=300,
        max_cost_ms=50.0,
        max_features=200,
        max_correlation=0.9,
        stability_resamples=50,
        fdr_q=0.2,
        walk_forward_folds=3,
        adaptive_penalties=True
    )
    
    tprint("Aggressive settings:")
    tprint(f"  - Max cost: {aggressive_config.max_cost_ms} ms")
    tprint(f"  - Max features: {aggressive_config.max_features}")
    tprint(f"  - Max correlation: {aggressive_config.max_correlation}")
    tprint(f"  - FDR threshold: {aggressive_config.fdr_q}")
    
    # High-frequency configuration
    tprint("\n3. High-Frequency Configuration")
    tprint("-" * 40)
    
    hf_config = PipelineConfig(
        base_timeframe_minutes=1,  # 1-minute base
        coarse_grid_min=5,
        coarse_grid_max=60,
        max_cost_ms=10.0,  # Lower latency requirements
        max_features=50,
        max_correlation=0.6,
        stability_resamples=200,
        fdr_q=0.01,
        walk_forward_folds=10,
        adaptive_penalties=True,
        hybrid_mode=True
    )
    
    tprint("High-frequency settings:")
    tprint(f"  - Base timeframe: {hf_config.base_timeframe_minutes} minutes")
    tprint(f"  - Max cost: {hf_config.max_cost_ms} ms")
    tprint(f"  - Max features: {hf_config.max_features}")
    tprint(f"  - Hybrid mode: {hf_config.hybrid_mode}")


def main():
    """Main function to run all examples."""
    tprint("Cross-Timeframe Feature Generation System")
    tprint("Comprehensive Example Usage")
    tprint("=" * 80)
    
    try:
        # Run complete pipeline example
        results = run_complete_pipeline_example()
        
        # Run individual component examples
        run_individual_component_examples()
        
        # Demonstrate configuration options
        demonstrate_configuration_options()
        
        tprint("\n" + "=" * 80)
        tprint("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        tprint("=" * 80)
        
        return results
        
    except Exception as e:
        tprint(f"\nError running examples: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
