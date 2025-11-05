"""
Complete Enhanced Sticky Finite HMM Pipeline Test

This test runs the full Sticky Finite HMM pipeline with all enhancements:
- sticky_finite_hmm_regime_discovery_step (BaseStep framework)
- sticky_finite_hmm_auto_tuner.py (Hierarchical optimization)
- enhanced_sticky_finite_hmm_clustering_integration.py (Feature generation)
- SVI Gradient, Rao-Blackwellization, Vectorized JIT Optimizations

Target: 2 years of ETHUSDT historical data with comprehensive feature engineering
"""

import sys
import os
import time
import warnings
from pathlib import Path
from typing import Dict, Any
import tempfile

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

warnings.filterwarnings('ignore')

async def run_complete_enhanced_pipeline(
    symbol: str = "ETHUSDT",
    timeframe: str = "1d", 
    years: int = 2,
    enable_auto_tuning: bool = True,
    tpe_trials: int = 50,  # Reduced for testing
    timeout: int = 1800,   # 30 minutes timeout
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run the complete enhanced Sticky Finite HMM pipeline with all optimizations.
    
    Args:
        symbol: Trading symbol (default: ETHUSDT)
        timeframe: Data timeframe (default: 1d)
        years: Years of historical data (default: 2)
        enable_auto_tuning: Whether to run auto-tuning (default: True)
        tpe_trials: Number of TPE optimization trials (default: 50)
        timeout: Timeout in seconds (default: 1800)
        verbose: Whether to print detailed progress (default: True)
        
    Returns:
        Dictionary containing all pipeline results
    """
    
    print("🚀 Starting Complete Enhanced Sticky Finite HMM Pipeline")
    print("=" * 80)
    print(f"📊 Symbol: {symbol}")
    print(f"⏰ Timeframe: {timeframe}")
    print(f"📅 Years: {years}")
    print(f"🔧 Auto-tuning: {enable_auto_tuning}")
    print(f"⚡ TPE Trials: {tpe_trials}")
    print(f"⏱️ Timeout: {timeout}s")
    print("=" * 80)
    
    start_time = time.time()
    results = {
        'pipeline_start': start_time,
        'symbol': symbol,
        'timeframe': timeframe,
        'years': years,
        'stages_completed': [],
        'stage_results': {},
        'errors': []
    }
    
    try:
        # STAGE 1: Data Loading with Enhanced Feature Integration
        print("\n🔍 STAGE 1: Data Loading & Feature Engineering")
        print("-" * 60)
        
        try:
            # Import required modules
            from src.utils.kline_parquet import KlineParquet, StorageConfig  # type: ignore
            from src.feature_generation.integration.enhanced_sticky_finite_hmm_clustering_integration import (  # type: ignore
                EnhancedStickyFiniteHMMClusteringIntegration
            )
            from datetime import datetime, timedelta
            
            # Initialize data loader
            storage_config = StorageConfig()
            kline_loader = KlineParquet(storage_config)
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365)
            
            print(f"📅 Loading data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Load historical data
            historical_data = kline_loader.load_klines(
                symbol=symbol,
                exchange="binance",
                interval=timeframe,
                start_time=start_date,
                end_time=end_date
            )
            
            if historical_data is None or len(historical_data) == 0:
                raise ValueError(f"No data loaded for {symbol} {timeframe}")
                
            print(f"✅ Loaded {len(historical_data)} data points")
            print(f"📊 Data columns: {list(historical_data.columns)}")
            
            # Initialize enhanced feature integration
            feature_integration = EnhancedStickyFiniteHMMClusteringIntegration(
                min_features=50,
                max_features=100,
                enable_comprehensive_features=True,
                enable_pca_reduction=True,
                pca_components=15,
                K=5,  # Default number of regimes
                n_mixtures=1,
                base_alpha=1.0,
                kappa=15.0,
                num_iters=100,
                lr=5e-3
            )
            
            print("🔧 Generating comprehensive features using enhanced pipeline...")
            
            # Generate features
            feature_results = feature_integration.generate_features_for_clustering(
                market_data=historical_data,
                symbol=symbol,
                exchange="binance", 
                timeframe=timeframe
            )
            
            if not feature_results or 'feature_matrix' not in feature_results:
                raise ValueError("Feature generation failed")
                
            feature_matrix = feature_results['feature_matrix']
            feature_names = feature_results.get('feature_names', [])
            
            print(f"✅ Feature generation completed")
            print(f"📈 Feature matrix shape: {feature_matrix.shape}")
            print(f"🔧 Number of features: {len(feature_names)}")
            
            results['stage_results']['data_loading'] = {
                'success': True,
                'data_points': len(historical_data),
                'feature_matrix_shape': feature_matrix.shape,
                'num_features': len(feature_names),
                'feature_names': feature_names[:10]  # First 10 features
            }
            results['stages_completed'].append('data_loading')
            
        except Exception as e:
            error_msg = f"Data loading failed: {str(e)}"
            print(f"❌ {error_msg}")
            results['errors'].append(error_msg)
            return results
            
        # STAGE 2: Auto-Tuning with Hierarchical Optimization
        if enable_auto_tuning:
            print("\n🎯 STAGE 2: Auto-Tuning with Hierarchical Optimization")
            print("-" * 60)
            
            try:
                from src.training.steps.market_analysis.sticky_finite_hmm_clustering.sticky_finite_hmm_auto_tuner import (  # type: ignore
                    run_sticky_finite_hmm_auto_tuning
                )
                
                print("🔧 Starting hierarchical auto-tuning...")
                print(f"⚡ TPE trials: {tpe_trials}")
                print(f"⏱️ Timeout: {timeout}s")
                
                # Run auto-tuning
                best_params, best_score, tuning_results = run_sticky_finite_hmm_auto_tuning(
                    market_data=historical_data,
                    symbol=symbol,
                    exchange="binance",
                    timeframe=timeframe,
                    use_hierarchical=True,
                    use_multi_objective=False,
                    tpe_trials=tpe_trials,
                    timeout=timeout,
                    verbose=verbose
                )
                
                print(f"✅ Auto-tuning completed")
                print(f"🎯 Best score: {best_score:.4f}")
                print(f"🔧 Best parameters: {list(best_params.keys())[:5]}...")
                
                results['stage_results']['auto_tuning'] = {
                    'success': True,
                    'best_score': best_score,
                    'best_params': best_params,
                    'tuning_summary': tuning_results.get('summary', {})
                }
                results['stages_completed'].append('auto_tuning')
                
                # Use best parameters for next stage
                optimized_params = best_params
                
            except Exception as e:
                error_msg = f"Auto-tuning failed: {str(e)}"
                print(f"⚠️ {error_msg}")
                print("🔄 Using default parameters for regime discovery...")
                results['errors'].append(error_msg)
                
                # Fallback to default parameters
                optimized_params = {
                    'K': 5,
                    'n_mixtures': 1,
                    'base_alpha': 1.0,
                    'kappa': 15.0,
                    'num_iters': 100,
                    'lr': 5e-3,
                    'min_features': 50,
                    'max_features': 100,
                    'pca_components': 15
                }
        else:
            print("\n⏭️ STAGE 2: Auto-Tuning (SKIPPED)")
            print("-" * 60)
            optimized_params = {
                'K': 5,
                'n_mixtures': 1,
                'base_alpha': 1.0,
                'kappa': 15.0,
                'num_iters': 100,
                'lr': 5e-3,
                'min_features': 50,
                'max_features': 100,
                'pca_components': 15
            }
            
        # STAGE 3: Enhanced Regime Discovery with All Optimizations
        print("\n🔬 STAGE 3: Enhanced Regime Discovery with All Optimizations")
        print("-" * 60)
        
        try:
            from src.training.steps.market_analysis.sticky_finite_hmm_clustering.sticky_finite_hmm_regime_discovery_step import (  # type: ignore
                StickyFiniteHMMRegimeDiscoveryStep
            )
            
            print("🚀 Initializing StickyFiniteHMMRegimeDiscoveryStep...")
            print("⚡ Enabling: SVI Gradient, Rao-Blackwellization, Vectorized JIT Optimizations")
            
            # Create output directory
            output_dir = Path(tempfile.mkdtemp(prefix="sticky_hmm_enhanced_"))
            print(f"📁 Output directory: {output_dir}")
            
            # Initialize regime discovery step with optimized parameters
            regime_step = StickyFiniteHMMRegimeDiscoveryStep(
                step_name="enhanced_regime_discovery",
                output_dir=str(output_dir)
            )
            
            # Prepare parameters with all optimizations
            step_params = {
                'K': optimized_params.get('K', 5),
                'n_mixtures': optimized_params.get('n_mixtures', 1),
                'base_alpha': optimized_params.get('base_alpha', 1.0),
                'kappa': optimized_params.get('kappa', 15.0),
                'num_iters': optimized_params.get('num_iters', 100),
                'lr': optimized_params.get('lr', 5e-3),
                'min_features': optimized_params.get('min_features', 50),
                'max_features': optimized_params.get('max_features', 100),
                'enable_pca': True,
                'pca_components': optimized_params.get('pca_components', 15),
                'compute_posteriors': True,
                'enable_svi_gradient': True,        # ✅ SVI Gradient optimization
                'enable_rao_blackwellization': True, # ✅ Rao-Blackwellization
                'enable_vectorized_jit': True,      # ✅ Vectorized JIT optimizations
                'early_stopping': True,
                'patience': 30,
                'random_state': 42
            }
            
            print("🔧 Running enhanced regime discovery with optimizations...")
            print(f"⚡ SVI Gradient: {step_params['enable_svi_gradient']}")
            print(f"🔬 Rao-Blackwellization: {step_params['enable_rao_blackwellization']}")
            print(f"🚀 Vectorized JIT: {step_params['enable_vectorized_jit']}")
            
            # Run the regime discovery step
            regime_results = await regime_step.run(
                market_data=historical_data,
                symbol=symbol,
                exchange="binance",
                timeframe=timeframe,
                **step_params
            )
            
            print("✅ Enhanced regime discovery completed")
            
            # Extract key results
            if regime_results and 'results' in regime_results:
                clustering_results = regime_results['results']
                
                results['stage_results']['regime_discovery'] = {
                    'success': True,
                    'n_regimes': clustering_results.get('n_clusters', 'N/A'),
                    'final_elbo': clustering_results.get('final_elbo', 'N/A'),
                    'quality_metrics': clustering_results.get('quality_metrics', {}),
                    'state_durations': clustering_results.get('state_durations', {}),
                    'transition_matrix_shape': clustering_results.get('transition_matrix', {}).get('shape', 'N/A'),
                    'optimizations_enabled': {
                        'svi_gradient': step_params['enable_svi_gradient'],
                        'rao_blackwellization': step_params['enable_rao_blackwellization'],
                        'vectorized_jit': step_params['enable_vectorized_jit']
                    }
                }
                results['stages_completed'].append('regime_discovery')
                
                print(f"🎯 Discovered {clustering_results.get('n_clusters', 'N/A')} regimes")
                print(f"📊 Final ELBO: {clustering_results.get('final_elbo', 'N/A')}")
                
            else:
                raise ValueError("Regime discovery returned no results")
                
        except Exception as e:
            error_msg = f"Regime discovery failed: {str(e)}"
            print(f"❌ {error_msg}")
            results['errors'].append(error_msg)
            
        # Final Summary
        total_time = time.time() - start_time
        results['pipeline_end'] = time.time()
        results['total_time'] = total_time
        results['stages_completed_count'] = len(results['stages_completed'])
        
        print("\n" + "=" * 80)
        print("🏁 COMPLETE ENHANCED PIPELINE SUMMARY")
        print("=" * 80)
        print(f"⏱️ Total time: {total_time:.2f}s")
        print(f"✅ Stages completed: {len(results['stages_completed'])}/{3}")
        print(f"📊 Data points processed: {results['stage_results'].get('data_loading', {}).get('data_points', 'N/A')}")
        print(f"🔧 Features generated: {results['stage_results'].get('data_loading', {}).get('num_features', 'N/A')}")
        
        if 'auto_tuning' in results['stages_completed']:
            best_score = results['stage_results'].get('auto_tuning', {}).get('best_score', 'N/A')
            print(f"🎯 Best tuning score: {best_score}")
            
        if 'regime_discovery' in results['stages_completed']:
            n_regimes = results['stage_results'].get('regime_discovery', {}).get('n_regimes', 'N/A')
            final_elbo = results['stage_results'].get('regime_discovery', {}).get('final_elbo', 'N/A')
            print(f"🎯 Regimes discovered: {n_regimes}")
            print(f"📊 Final ELBO: {final_elbo}")
            
        if results['errors']:
            print(f"⚠️ Errors encountered: {len(results['errors'])}")
            for error in results['errors']:
                print(f"   - {error}")
        else:
            print("🎉 All stages completed successfully!")
            
        print("=" * 80)
        
        return results
        
    except Exception as e:
        error_msg = f"Pipeline failed: {str(e)}"
        print(f"❌ {error_msg}")
        results['errors'].append(error_msg)
        results['pipeline_end'] = time.time()
        results['total_time'] = time.time() - start_time
        return results


async def main():
    """Main function to run the complete enhanced pipeline."""
    
    print("🚀 Sticky Finite HMM - Complete Enhanced Pipeline Test")
    print("🔬 Features: SVI Gradient, Rao-Blackwellization, Vectorized JIT")
    print("📊 Target: 2 years ETHUSDT historical data")
    print("🎯 Components: Regime Discovery + Auto-Tuner + Enhanced Integration")
    print()
    
    # Run the complete pipeline
    results = await run_complete_enhanced_pipeline(
        symbol="ETHUSDT",
        timeframe="1d",
        years=2,
        enable_auto_tuning=True,
        tpe_trials=30,  # Reduced for testing
        timeout=1200,   # 20 minutes
        verbose=True
    )
    
    # Save results summary
    import json
    output_file = "enhanced_pipeline_results.json"
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        else:
            return obj
    
    serializable_results = convert_numpy(results)
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)
        
    print(f"\n💾 Results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
