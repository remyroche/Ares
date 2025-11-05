"""
Enhanced Statsmodel Clustering Example

This example demonstrates how to use the enhanced statsmodel clustering system
with comprehensive feature engineering, hybrid clustering, hierarchical optimization,
and quality assessment with CSV export.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings

# Import enhanced clustering components
try:
    from ..feature_engineering import (
        EnhancedFeatureEngineer, 
        CovarianceStabilizer,
        create_enhanced_feature_engineer,
        create_covariance_stabilizer
    )
    from ..clustering import (
        HybridClusteringEngine,
        create_hybrid_clustering_engine
    )
    from ..optimization import (
        HierarchicalParameterOptimizer,
        create_hierarchical_optimizer
    )
    from ..assessment import (
        QualityAssessmentIntegrator,
        create_quality_assessment_integrator
    )
    ENHANCED_CLUSTERING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Enhanced clustering components not available: {e}")
    ENHANCED_CLUSTERING_AVAILABLE = False

# Import utilities
try:
    from src.utils.tprint import (
        tprint_info, tprint_success, tprint_warning, tprint_error
    )
except ImportError:
    def tprint_info(msg): print(f'ℹ️  {msg}')
    def tprint_success(msg): print(f'✅ {msg}')
    def tprint_warning(msg): print(f'⚠️  {msg}')
    def tprint_error(msg): print(f'❌ {msg}')


def generate_sample_data(n_samples: int = 1000, n_assets: int = 50) -> pd.DataFrame:
    """
    Generate sample financial data for demonstration.
    
    Args:
        n_samples: Number of time periods
        n_assets: Number of assets
        
    Returns:
        DataFrame with OHLC price data
    """
    tprint_info(f"📊 Generating sample data: {n_samples} periods, {n_assets} assets")
    
    # Generate random price data
    np.random.seed(42)
    
    # Create base price series
    base_prices = 100 + np.cumsum(np.random.normal(0, 0.01, n_samples))
    
    # Generate OHLC data
    price_data = pd.DataFrame(index=pd.date_range('2020-01-01', periods=n_samples, freq='D'))
    
    for i in range(n_assets):
        asset_name = f'Asset_{i+1}'
        
        # Add noise to base prices for each asset
        asset_prices = base_prices * (1 + np.random.normal(0, 0.02, n_samples))
        
        # Generate OHLC
        high = asset_prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples)))
        low = asset_prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples)))
        open_price = low + np.random.uniform(0, 1, n_samples) * (high - low)
        close = asset_prices
        
        price_data[asset_name] = close
        price_data[f'{asset_name}_open'] = open_price
        price_data[f'{asset_name}_high'] = high
        price_data[f'{asset_name}_low'] = low
    
    # Generate volume data
    volume_data = pd.DataFrame(index=price_data.index)
    for i in range(n_assets):
        asset_name = f'Asset_{i+1}'
        volume = np.random.exponential(1000000, n_samples)
        volume_data[asset_name] = volume
    
    # Generate market data (for factor calculations)
    market_data = pd.DataFrame(index=price_data.index)
    market_data['market_return'] = np.random.normal(0, 0.01, n_samples)
    market_data['market_cap'] = np.random.exponential(1e9, n_samples)
    market_data['book_value'] = np.random.exponential(0.8e9, n_samples)
    market_data['earnings'] = np.random.exponential(0.1e9, n_samples)
    
    tprint_success("✅ Sample data generated")
    return price_data, volume_data, market_data


def define_parameter_space() -> dict:
    """Define parameter search space for optimization."""
    return {
        'n_asset_clusters': {
            'type': 'int',
            'low': 3,
            'high': 10
        },
        'n_regimes': {
            'type': 'int',
            'low': 2,
            'high': 5
        },
        'aggregation_method': {
            'type': 'categorical',
            'choices': ['pca', 'mean', 'weighted_mean']
        },
        'covariance_method': {
            'type': 'categorical',
            'choices': ['ledoit_wolf', 'exponential', 'shrunk']
        },
        'static_method': {
            'type': 'categorical',
            'choices': ['hierarchical', 'spectral', 'louvain']
        },
        'linkage_method': {
            'type': 'categorical',
            'choices': ['ward', 'complete', 'average']
        }
    }


def create_objective_function(price_data: pd.DataFrame, 
                         volume_data: pd.DataFrame,
                         market_data: pd.DataFrame):
    """
    Create objective function for optimization.
    
    Args:
        price_data: OHLC price data
        volume_data: Volume data
        market_data: Market data
        
    Returns:
        Objective function for optimization
    """
    def objective_function(params: dict, data: tuple) -> float:
        """
        Objective function combining multiple metrics.
        
        Args:
            params: Parameters to evaluate
            data: Tuple of (price_data, volume_data, market_data)
            
        Returns:
            Combined objective score (higher is better)
        """
        price_data, volume_data, market_data = data
        
        try:
            # 1. Feature engineering
            feature_engineer = create_enhanced_feature_engineer(
                include_raw_returns=True,
                include_log_returns=True,
                include_realized_vol=True,
                include_rolling_features=True,
                rolling_windows=[5, 10, 20],
                shift_periods=1,
                enable_rank_normalization=True
            )
            
            features = feature_engineer.extract_features(
                price_data=price_data,
                volume_data=volume_data,
                market_data=market_data
            )
            
            # 2. Hybrid clustering
            clustering_engine = create_hybrid_clustering_engine(
                static_method=params.get('static_method', 'hierarchical'),
                n_asset_clusters=params.get('n_asset_clusters', 5),
                n_regimes=params.get('n_regimes', 3),
                aggregation_method=params.get('aggregation_method', 'pca'),
                covariance_method=params.get('covariance_method', 'ledoit_wolf')
            )
            
            clustering_results = clustering_engine.fit_predict(
                returns=price_data['close'].pct_change(),
                features=features
            )
            
            # 3. Quality assessment
            quality_integrator = create_quality_assessment_integrator(
                output_dir="outcomes",
                include_datetime=True,
                integrate_with_cluster_assessor=True,
                enable_all_assessments=True
            )
            
            quality_results = quality_integrator.assess_quality(
                model=clustering_results['temporal_model'],
                data=features,
                regime_labels=clustering_results['regime_labels'],
                forward_returns=price_data['close'].pct_change().shift(-1),  # Next period returns
                timestamps=price_data.index,
                symbol="EXAMPLE"
            )
            
            # 4. Combine metrics into objective score
            # Use quality score as primary objective
            quality_score = quality_results.get('standard_quality_metrics', {}).get('quality_score', 0.0)
            
            # Add economic metrics if available
            economic_metrics = quality_results.get('economic_metrics', {})
            if economic_metrics:
                regime_sharpe = economic_metrics.get('regime_sharpe_ratio', 0.0)
                # Combine quality and economic metrics
                combined_score = 0.7 * quality_score + 0.3 * regime_sharpe
            else:
                combined_score = quality_score
            
            return combined_score
            
        except Exception as e:
            tprint_warning(f"⚠️ Objective function evaluation failed: {e}")
            return -np.inf
    
    return objective_function


def run_enhanced_clustering_example():
    """Run complete enhanced clustering example."""
    tprint_info("🚀 Starting Enhanced Statsmodel Clustering Example")
    
    try:
        # 1. Generate sample data
        price_data, volume_data, market_data = generate_sample_data(
            n_samples=500, n_assets=20
        )
        
        # 2. Define parameter space
        parameter_space = define_parameter_space()
        
        # 3. Create objective function
        objective_function = create_objective_function(
            price_data, volume_data, market_data
        )
        
        # 4. Create hierarchical optimizer
        optimizer = create_hierarchical_optimizer(
            objective_function=objective_function,
            parameter_space=parameter_space,
            stage1_method='bayesian',
            stage1_n_trials=20,  # Reduced for example
            stage2_method='bfgs',
            enable_economic_objectives=True
        )
        
        # 5. Run optimization
        optimization_results = optimizer.optimize(
            data=(price_data, volume_data, market_data)
        )
        
        # 6. Extract best parameters and results
        best_params = optimization_results['best_params']
        tprint_success(f"✅ Optimization complete. Best parameters: {best_params}")
        
        # 7. Run final clustering with best parameters
        feature_engineer = create_enhanced_feature_engineer(
            include_raw_returns=True,
            include_log_returns=True,
            include_realized_vol=True,
            include_rolling_features=True,
            rolling_windows=[5, 10, 20],
            shift_periods=1,
            enable_rank_normalization=True
        )
        
        features = feature_engineer.extract_features(
            price_data=price_data,
            volume_data=volume_data,
            market_data=market_data
        )
        
        clustering_engine = create_hybrid_clustering_engine(
            static_method=best_params.get('static_method', 'hierarchical'),
            n_asset_clusters=best_params.get('n_asset_clusters', 5),
            n_regimes=best_params.get('n_regimes', 3),
            aggregation_method=best_params.get('aggregation_method', 'pca'),
            covariance_method=best_params.get('covariance_method', 'ledoit_wolf')
        )
        
        final_results = clustering_engine.fit_predict(
            returns=price_data['close'].pct_change(),
            features=features
        )
        
        # 8. Comprehensive quality assessment
        quality_integrator = create_quality_assessment_integrator(
            output_dir="outcomes",
            include_datetime=True,
            integrate_with_cluster_assessor=True,
            enable_all_assessments=True
        )
        
        quality_results = quality_integrator.assess_quality(
            model=final_results['temporal_model'],
            data=features,
            regime_labels=final_results['regime_labels'],
            forward_returns=price_data['close'].pct_change().shift(-1),
            timestamps=price_data.index,
            symbol="EXAMPLE"
        )
        
        # 9. Display results summary
        tprint_success("✅ Enhanced clustering example complete!")
        tprint_info("📊 Results Summary:")
        tprint_info(f"   • Best parameters: {best_params}")
        tprint_info(f"   • Quality score: {quality_results.get('standard_quality_metrics', {}).get('quality_score', 'N/A')}")
        tprint_info(f"   • CSV reports: {quality_results.get('csv_reports', {})}")
        
        # 10. Return comprehensive results
        return {
            'best_parameters': best_params,
            'clustering_results': final_results,
            'quality_assessment': quality_results,
            'optimization_results': optimization_results
        }
        
    except Exception as e:
        tprint_error(f"❌ Enhanced clustering example failed: {e}")
        raise


if __name__ == "__main__":
    # Run the example
    results = run_enhanced_clustering_example()
    
    # Print final summary
    print("\n" + "="*60)
    print("ENHANCED STATSMODEL CLUSTERING EXAMPLE COMPLETE")
    print("="*60)
    print(f"Best Parameters: {results['best_parameters']}")
    print(f"Quality Score: {results['quality_assessment'].get('standard_quality_metrics', {}).get('quality_score', 'N/A')}")
    print(f"CSV Reports Generated: {list(results['quality_assessment'].get('csv_reports', {}).values())}")
    print("="*60)