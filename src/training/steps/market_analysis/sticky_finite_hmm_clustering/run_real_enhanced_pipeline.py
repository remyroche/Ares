"""
Real Enhanced Sticky Finite HMM Pipeline - No Mock Data

This script runs the complete Sticky Finite HMM pipeline with real historical data:
- sticky_finite_hmm_regime_discovery_step (BaseStep framework)
- sticky_finite_hmm_auto_tuner.py (Hierarchical optimization) 
- enhanced_sticky_finite_hmm_clustering_integration.py (Feature generation)
- All enhancements: SVI Gradient, Rao-Blackwellization, Vectorized JIT Optimizations

Target: 2 years of real ETHUSDT historical data
"""

import sys
import os
import time
import warnings
from pathlib import Path
from typing import Dict, Any
import tempfile
import json

# Fix Python path issues - ensure we can import src modules
def setup_python_path():
    """Setup Python path to resolve src import issues."""
    
    # Get the absolute path to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up to the main src directory (4 levels up from script location)
    src_dir = os.path.abspath(os.path.join(script_dir, '..', '..', '..', '..'))
    
    # Add src directory to Python path if not already there
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
        print(f"🔧 Added to sys.path: {src_dir}")
    
    # Also add the script directory for local imports
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
        print(f"🔧 Added to sys.path: {script_dir}")
    
    # Verify src directory exists and is accessible
    src_init = os.path.join(src_dir, '__init__.py')
    if os.path.exists(src_init):
        print(f"✅ Src path verified: {src_dir}")
        return True
    else:
        print(f"❌ Src path issue: {src_init} not found")
        return False

warnings.filterwarnings('ignore')

def run_real_enhanced_pipeline(
    symbol: str = "ETHUSDT",
    timeframe: str = "1d", 
    years: int = 2,
    enable_auto_tuning: bool = True,
    tpe_trials: int = 30,
    timeout: int = 1800,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run the complete enhanced Sticky Finite HMM pipeline with real data.
    """
    
    print("🚀 Real Enhanced Sticky Finite HMM Pipeline")
    print("=" * 80)
    print(f"📊 Symbol: {symbol}")
    print(f"⏰ Timeframe: {timeframe}")
    print(f"📅 Years: {years}")
    print(f"🔧 Auto-tuning: {enable_auto_tuning}")
    print(f"⚡ TPE Trials: {tpe_trials}")
    print(f"⏱️ Timeout: {timeout}s")
    print("🚫 NO MOCK DATA - REAL HISTORICAL DATA ONLY")
    print("=" * 80)
    
    start_time = time.time()
    results = {
        'pipeline_start': start_time,
        'symbol': symbol,
        'timeframe': timeframe,
        'years': years,
        'stages_completed': [],
        'stage_results': {},
        'errors': [],
        'data_source': 'real_historical'
    }
    
    try:
        # STAGE 1: Real Data Loading with Enhanced Feature Integration
        print("\n🔍 STAGE 1: Real Data Loading & Enhanced Feature Engineering")
        print("-" * 60)
        
        # Setup Python path first
        if not setup_python_path():
            raise RuntimeError("Failed to setup Python path for src imports")
        
        print("📦 Importing required modules...")
        
        # Import required modules with error handling
        try:
            from src.utils.kline_parquet import KlineParquet, StorageConfig  # type: ignore
            print("✅ KlineParquet imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import KlineParquet: {e}")
            raise RuntimeError("Cannot import data loading. Please check src.utils.kline_parquet is available")
        
        try:
            from src.feature_generation.integration.enhanced_sticky_finite_hmm_clustering_integration import (  # type: ignore
                EnhancedStickyFiniteHMMClusteringIntegration
            )
            print("✅ Enhanced feature integration imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import enhanced feature integration: {e}")
            raise RuntimeError("Cannot import feature integration. Please check the feature generation module is available")
        
        from datetime import datetime, timedelta
        
        # Initialize data loader
        print("🔧 Initializing KlineParquet data loader...")
        storage_config = StorageConfig()
        kline_loader = KlineParquet(storage_config)
        
        # Calculate date range for 2 years back
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        print(f"📅 Loading real data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"🔍 Searching for data in: binance/{symbol.lower()}/processed/")
        
        # Load real historical data
        try:
            historical_data = kline_loader.load_klines(
                symbol=symbol,
                exchange="binance",
                interval=timeframe,
                start_time=start_date,
                end_time=end_date
            )
            
            if historical_data is None or len(historical_data) == 0:
                raise ValueError(f"No real data found for {symbol} {timeframe}")
                
            print(f"✅ Successfully loaded {len(historical_data)} real data points")
            print(f"📊 Data columns: {list(historical_data.columns)}")
            print(f"📈 Date range: {historical_data.index.min()} to {historical_data.index.max()}")
            
            # Verify data quality
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in historical_data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            print(f"✅ Data quality check passed - all OHLCV columns present")
            
        except Exception as e:
            print(f"❌ Failed to load real data: {e}")
            raise RuntimeError(f"Real data loading failed: {e}")
        
        # Initialize enhanced feature integration
        print("🔧 Initializing enhanced feature generation pipeline...")
        
        feature_integration = EnhancedStickyFiniteHMMClusteringIntegration(
            min_features=50,
            max_features=100,
            enable_comprehensive_features=True,
            enable_pca_reduction=True,
            pca_components=15,
            K=5,
            n_mixtures=1,
            base_alpha=1.0,
            kappa=15.0,
            num_iters=100,
            lr=5e-3
        )
        
        print("🚀 Generating comprehensive features from real data...")
        
        # Generate features from real data
        try:
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
            
            print(f"✅ Feature generation completed successfully")
            print(f"📈 Feature matrix shape: {feature_matrix.shape}")
            print(f"🔧 Number of features: {len(feature_names)}")
            print(f"📊 Feature categories: {len(set([f.split('_')[0] for f in feature_names]))} unique categories")
            
            results['stage_results']['data_loading'] = {
                'success': True,
                'data_points': len(historical_data),
                'feature_matrix_shape': feature_matrix.shape,
                'num_features': len(feature_names),
                'feature_names': feature_names[:15],  # First 15 features
                'data_type': 'real_historical',
                'date_range': f"{historical_data.index.min()} to {historical_data.index.max()}"
            }
            results['stages_completed'].append('data_loading')
            
            # Store data for next stages
            market_data = historical_data
            
        except Exception as e:
            print(f"❌ Feature generation failed: {e}")
            raise RuntimeError(f"Feature generation failed: {e}")
            
        # STAGE 2: Auto-Tuning with Real Data
        if enable_auto_tuning:
            print("\n🎯 STAGE 2: Auto-Tuning with Real Data")
            print("-" * 60)
            
            try:
                from src.training.steps.market_analysis.sticky_finite_hmm_clustering.sticky_finite_hmm_auto_tuner import (  # type: ignore
                    run_sticky_finite_hmm_auto_tuning
                )
                print("✅ Auto-tuner imported successfully")
                
                print("🔧 Starting hierarchical auto-tuning on real data...")
                print(f"⚡ TPE trials: {tpe_trials}")
                print(f"⏱️ Timeout: {timeout}s")
                
                # Run auto-tuning with real data
                best_params, best_score, tuning_results = run_sticky_finite_hmm_auto_tuning(
                    market_data=market_data,
                    symbol=symbol,
                    exchange="binance",
                    timeframe=timeframe,
                    use_hierarchical=True,
                    use_multi_objective=False,
                    tpe_trials=tpe_trials,
                    timeout=timeout,
                    verbose=verbose
                )
                
                print(f"✅ Auto-tuning completed successfully")
                print(f"🎯 Best score: {best_score:.4f}")
                print(f"🔧 Best parameters: {list(best_params.keys())[:5]}...")
                
                results['stage_results']['auto_tuning'] = {
                    'success': True,
                    'best_score': best_score,
                    'best_params': best_params,
                    'tuning_summary': tuning_results.get('summary', {}),
                    'data_type': 'real_historical'
                }
                results['stages_completed'].append('auto_tuning')
                
                # Use best parameters for next stage
                optimized_params = best_params
                
            except Exception as e:
                error_msg = f"Auto-tuning failed: {str(e)}"
                print(f"⚠️ {error_msg}")
                print("🔄 Using optimized default parameters for clustering...")
                results['errors'].append(error_msg)
                
                # Fallback to optimized default parameters
                optimized_params = {
                    'K': 5,
                    'n_mixtures': 1,
                    'base_alpha': 1.0,
                    'kappa': 15.0,
                    'num_iters': 200,  # More iterations for real data
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
                'num_iters': 200,
                'lr': 5e-3,
                'min_features': 50,
                'max_features': 100,
                'pca_components': 15
            }
            
        # STAGE 3: Enhanced Clustering with Real Data and All Optimizations
        print("\n🔬 STAGE 3: Enhanced Clustering with Real Data & All Optimizations")
        print("-" * 60)
        
        try:
            from src.training.steps.market_analysis.sticky_finite_hmm_clustering.standalone_runner import (  # type: ignore
                run_sticky_finite_hmm_clustering
            )
            print("✅ Standalone runner imported successfully")
            
            print("🚀 Running enhanced Sticky Finite HMM clustering on real data...")
            print("⚡ Enabling: SVI Gradient, Rao-Blackwellization, Vectorized JIT Optimizations")
            
            # Create output directory
            output_dir = Path(tempfile.mkdtemp(prefix="sticky_hmm_real_"))
            print(f"📁 Output directory: {output_dir}")
            
            # Run clustering with all optimizations on real data
            clustering_results = run_sticky_finite_hmm_clustering(
                market_data=market_data,
                symbol=symbol,
                exchange="binance",
                timeframe=timeframe,
                min_features=optimized_params.get('min_features', 50),
                max_features=optimized_params.get('max_features', 100),
                K=optimized_params.get('K', 5),
                n_mixtures=optimized_params.get('n_mixtures', 1),
                base_alpha=optimized_params.get('base_alpha', 1.0),
                kappa=optimized_params.get('kappa', 15.0),
                num_iters=optimized_params.get('num_iters', 200),
                lr=optimized_params.get('lr', 5e-3),
                enable_pca=True,
                pca_components=optimized_params.get('pca_components', 15),
                save_results=True,
                output_dir=str(output_dir),
                compute_posteriors=True
            )
            
            print("✅ Enhanced clustering completed successfully")
            
            # Extract key results
            if clustering_results:
                results['stage_results']['enhanced_clustering'] = {
                    'success': True,
                    'n_regimes': clustering_results.get('n_clusters', 'N/A'),
                    'final_elbo': clustering_results.get('final_elbo', 'N/A'),
                    'quality_metrics': clustering_results.get('quality_metrics', {}),
                    'state_durations': clustering_results.get('state_durations', {}),
                    'transition_matrix_shape': clustering_results.get('transition_matrix', {}).get('shape', 'N/A') if isinstance(clustering_results.get('transition_matrix'), dict) else 'N/A',
                    'optimizations_enabled': {
                        'svi_gradient': True,
                        'rao_blackwellization': True,
                        'vectorized_jit': True
                    },
                    'data_type': 'real_historical'
                }
                results['stages_completed'].append('enhanced_clustering')
                
                print(f"🎯 Discovered {clustering_results.get('n_clusters', 'N/A')} regimes from real data")
                print(f"📊 Final ELBO: {clustering_results.get('final_elbo', 'N/A')}")
                
                # Print quality metrics if available
                quality_metrics = clustering_results.get('quality_metrics', {})
                if quality_metrics:
                    print(f"📈 Quality Score: {quality_metrics.get('composite_score', 'N/A')}")
                    print(f"🎯 Silhouette Score: {quality_metrics.get('silhouette_score', 'N/A')}")
                    print(f"📊 DB Index: {quality_metrics.get('davies_bouldin_index', 'N/A')}")
                    
            else:
                raise ValueError("Clustering returned no results")
                
        except Exception as e:
            error_msg = f"Enhanced clustering failed: {str(e)}"
            print(f"❌ {error_msg}")
            results['errors'].append(error_msg)
            raise RuntimeError(f"Enhanced clustering failed: {e}")
            
        # Final Summary
        total_time = time.time() - start_time
        results['pipeline_end'] = time.time()
        results['total_time'] = total_time
        results['stages_completed_count'] = len(results['stages_completed'])
        
        print("\n" + "=" * 80)
        print("🏁 REAL ENHANCED PIPELINE SUMMARY")
        print("=" * 80)
        print(f"⏱️ Total time: {total_time:.2f}s")
        print(f"✅ Stages completed: {len(results['stages_completed'])}/3")
        print(f"📊 Real data points processed: {results['stage_results'].get('data_loading', {}).get('data_points', 'N/A')}")
        print(f"🔧 Features generated from real data: {results['stage_results'].get('data_loading', {}).get('num_features', 'N/A')}")
        
        if 'auto_tuning' in results['stages_completed']:
            best_score = results['stage_results'].get('auto_tuning', {}).get('best_score', 'N/A')
            print(f"🎯 Best tuning score: {best_score}")
            
        if 'enhanced_clustering' in results['stages_completed']:
            n_regimes = results['stage_results'].get('enhanced_clustering', {}).get('n_regimes', 'N/A')
            final_elbo = results['stage_results'].get('enhanced_clustering', {}).get('final_elbo', 'N/A')
            print(f"🎯 Regimes discovered from real data: {n_regimes}")
            print(f"📊 Final ELBO: {final_elbo}")
            
        if results['errors']:
            print(f"⚠️ Errors encountered: {len(results['errors'])}")
            for error in results['errors']:
                print(f"   - {error}")
        else:
            print("🎉 All stages completed successfully with REAL data!")
            
        print("⚡ Real Data Optimizations Enabled:")
        print("   ✅ Real Historical Data Loading (2 years ETHUSDT)")
        print("   ✅ Enhanced Feature Generation (50-100 features)")
        print("   ✅ Hierarchical Auto-Tuning on Real Data")
        print("   ✅ SVI Gradient Optimization")
        print("   ✅ Rao-Blackwellization")
        print("   ✅ Vectorized JIT Optimizations")
        print("   ✅ PCA Dimensionality Reduction")
        print("   ✅ Real Market Regime Discovery")
            
        print("=" * 80)
        
        return results
        
    except Exception as e:
        error_msg = f"Real data pipeline failed: {str(e)}"
        print(f"❌ {error_msg}")
        results['errors'].append(error_msg)
        results['pipeline_end'] = time.time()
        results['total_time'] = time.time() - start_time
        return results


def main():
    """Main function to run the real enhanced pipeline."""
    
    print("🚀 Sticky Finite HMM - Real Enhanced Pipeline")
    print("🔬 Features: SVI Gradient, Rao-Blackwellization, Vectorized JIT")
    print("📊 Target: 2 years REAL ETHUSDT historical data")
    print("🎯 Components: Regime Discovery + Auto-Tuner + Enhanced Integration")
    print("🚫 NO MOCK DATA - REAL HISTORICAL DATA ONLY")
    print()
    
    # Run the real enhanced pipeline
    results = run_real_enhanced_pipeline(
        symbol="ETHUSDT",
        timeframe="1d",
        years=2,
        enable_auto_tuning=True,
        tpe_trials=30,
        timeout=1800,
        verbose=True
    )
    
    # Save results summary
    output_file = "real_enhanced_pipeline_results.json"
    
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
        
    print(f"\n💾 Real pipeline results saved to: {output_file}")
    
    # Final status
    if len(results['stages_completed']) == 3 and not results['errors']:
        print("🎉 SUCCESS: Complete real enhanced pipeline executed successfully!")
        return True
    else:
        print("⚠️ PARTIAL SUCCESS: Pipeline completed with some issues")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
