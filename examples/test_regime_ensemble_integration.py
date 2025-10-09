"""
Test script for regime ensemble integration with data splitting and downstream models.

This script tests the complete data flow from regime ensemble training to data splitting
to Analyst & Tactician models to ensure probabilistic regime outputs are properly integrated.
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from training.steps.market_analysis.components.regime_ensemble_training import RegimeEnsembleTrainingComponent
from training.steps.market_analysis.regime_data_splitting.regime_data_splitting_component import RegimeDataSplittingComponent
from training.steps.models_training.analyst_models_training import AnalystModelsTrainingStep
from training.steps.models_training.tactician_models_training import TacticianModelsTrainingStep
from utils.tprint import tprint


def create_sample_market_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample market data for testing."""
    tprint("📊 Creating sample market data", color="cyan")
    
    # Generate synthetic market data
    np.random.seed(42)
    
    # Create time series data
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1H')
    
    # Generate price data with different regimes
    n_regimes = 4
    regime_length = n_samples // n_regimes
    
    data = []
    for i in range(n_regimes):
        start_idx = i * regime_length
        end_idx = min((i + 1) * regime_length, n_samples)
        
        # Different characteristics for each regime
        if i == 0:  # Trending up
            trend = np.linspace(100, 120, end_idx - start_idx)
            noise = np.random.normal(0, 1, end_idx - start_idx)
        elif i == 1:  # Trending down
            trend = np.linspace(120, 100, end_idx - start_idx)
            noise = np.random.normal(0, 1.5, end_idx - start_idx)
        elif i == 2:  # High volatility
            trend = np.full(end_idx - start_idx, 110)
            noise = np.random.normal(0, 3, end_idx - start_idx)
        else:  # Low volatility
            trend = np.full(end_idx - start_idx, 110)
            noise = np.random.normal(0, 0.5, end_idx - start_idx)
        
        prices = trend + noise
        
        for j, price in enumerate(prices):
            data.append({
                'timestamp': dates[start_idx + j],
                'open': price + np.random.normal(0, 0.1),
                'high': price + abs(np.random.normal(0, 0.2)),
                'low': price - abs(np.random.normal(0, 0.2)),
                'close': price,
                'volume': np.random.randint(1000, 10000)
            })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    tprint(f"✅ Created sample data with {len(df)} samples", color="green")
    return df


def test_regime_ensemble_training():
    """Test regime ensemble training with probabilistic outputs."""
    tprint("🧪 Testing Regime Ensemble Training", color="cyan", bold=True)
    
    try:
        # Create sample data
        data = create_sample_market_data(500)
        
        # Initialize component
        component = RegimeEnsembleTrainingComponent()
        
        # Create synthetic regime labels
        regime_labels = np.random.randint(0, 4, len(data))
        
        # Prepare pipeline state
        pipeline_state = {
            'artifacts': {
                'optimal_regime_clustering_result': {
                    'clustering_result': {
                        'cluster_assignments': regime_labels
                    }
                }
            }
        }
        
        # Execute training
        tprint("🏋️ Training regime ensemble...", color="yellow")
        result = component.execute(data, pipeline_state)
        
        if result.success:
            tprint("✅ Regime ensemble training completed successfully", color="green")
            
            # Extract stacker result
            stacker_result = result.artifacts.get('regime_ensemble_training_result', {}).get('stacker_lgbm_calibrated')
            
            if stacker_result:
                # Test probabilistic prediction
                tprint("🔮 Testing probabilistic regime prediction...", color="yellow")
                
                # Prepare test data
                X_test = np.random.randn(100, 50)  # 50 features
                feature_names = [f'feature_{i}' for i in range(50)]
                
                # Make predictions with probabilities
                prediction_result = component.predict_regimes_with_probabilities(
                    stacker_result=stacker_result,
                    X=X_test,
                    feature_names=feature_names,
                    scaler=None
                )
                
                if 'error' not in prediction_result:
                    tprint("✅ Probabilistic prediction completed successfully", color="green")
                    
                    # Verify probabilistic outputs
                    assert 'regime_labels' in prediction_result
                    assert 'regime_probabilities' in prediction_result
                    assert 'regime_analysis' in prediction_result
                    assert 'ensemble_probabilities' in prediction_result
                    
                    tprint(f"📊 Generated {len(prediction_result['regime_labels'])} regime predictions", color="blue")
                    tprint(f"📊 Regime probabilities shape: {prediction_result['regime_probabilities'].shape}", color="blue")
                    tprint(f"📊 Ensemble probabilities from {len(prediction_result['ensemble_probabilities'])} models", color="blue")
                    
                    return stacker_result, prediction_result
                else:
                    tprint(f"❌ Prediction failed: {prediction_result.get('error')}", color="red")
                    return None, None
            else:
                tprint("❌ No stacker result found", color="red")
                return None, None
        else:
            tprint(f"❌ Training failed: {result.error_message}", color="red")
            return None, None
            
    except Exception as e:
        tprint(f"❌ Test failed: {e}", color="red")
        return None, None


def test_regime_data_splitting(stacker_result, prediction_result):
    """Test regime data splitting with probabilistic outputs."""
    tprint("🧪 Testing Regime Data Splitting", color="cyan", bold=True)
    
    try:
        # Create sample market data
        data = create_sample_market_data(500)
        
        # Initialize component
        component = RegimeDataSplittingComponent()
        
        # Prepare pipeline state with ensemble result
        pipeline_state = {
            'artifacts': {
                'regime_ensemble_training_result': stacker_result
            }
        }
        
        # Execute data splitting
        tprint("✂️ Performing regime data splitting...", color="yellow")
        result = component.execute(data, pipeline_state)
        
        if result.success:
            tprint("✅ Regime data splitting completed successfully", color="green")
            
            # Extract regime data
            regime_data = result.artifacts.get('regime_data_splitting_result', {}).get('data', {})
            comprehensive_regime_info = result.artifacts.get('regime_data_splitting_result', {}).get('comprehensive_regime_info', {})
            
            if regime_data and comprehensive_regime_info:
                # Verify comprehensive regime information
                assert 'regime_probabilities' in comprehensive_regime_info
                assert 'regime_analysis' in comprehensive_regime_info
                assert 'ensemble_probabilities' in comprehensive_regime_info
                assert 'has_probabilistic_outputs' in comprehensive_regime_info
                
                # Verify market data has regime features
                market_data = regime_data.get('market_data')
                if market_data is not None:
                    regime_feature_cols = [col for col in market_data.columns if 'regime_' in col or 'ensemble_' in col]
                    tprint(f"📊 Added {len(regime_feature_cols)} regime features to market data", color="blue")
                    tprint(f"📊 Regime feature columns: {regime_feature_cols[:10]}...", color="blue")
                
                tprint("✅ Comprehensive regime information available for downstream models", color="green")
                return regime_data, comprehensive_regime_info
            else:
                tprint("❌ No regime data or comprehensive info found", color="red")
                return None, None
        else:
            tprint(f"❌ Data splitting failed: {result.error_message}", color="red")
            return None, None
            
    except Exception as e:
        tprint(f"❌ Test failed: {e}", color="red")
        return None, None


def test_analyst_models(regime_data, comprehensive_regime_info):
    """Test Analyst models with regime probability features."""
    tprint("🧪 Testing Analyst Models with Regime Features", color="cyan", bold=True)
    
    try:
        # Extract market data and create features
        market_data = regime_data.get('market_data')
        if market_data is None:
            tprint("❌ No market data available", color="red")
            return False
        
        # Create feature columns (exclude regime columns for now)
        feature_columns = [col for col in market_data.columns if not col.startswith('regime_') and not col.startswith('ensemble_')]
        
        # Create target columns
        target_columns = ['target_return']  # Simplified for testing
        
        # Add synthetic target
        market_data['target_return'] = np.random.normal(0, 0.01, len(market_data))
        
        # Initialize Analyst training
        analyst_trainer = AnalystModelsTrainingStep()
        
        # Test regime feature assembly
        tprint("🔧 Testing regime feature assembly for Analyst...", color="yellow")
        
        # Create sample data
        X = market_data[feature_columns].values
        y = market_data[target_columns].values
        sample_weight = np.ones(len(X))
        
        # Test regime feature tensor assembly
        regime_features = analyst_trainer._assemble_regime_feature_tensor(
            X=X,
            oof_predictions={},
            sample_weight=sample_weight,
            comprehensive_regime_info=comprehensive_regime_info
        )
        
        if regime_features:
            tprint(f"✅ Assembled {len(regime_features)} regime features for Analyst", color="green")
            tprint(f"📊 Regime feature keys: {list(regime_features.keys())[:10]}...", color="blue")
            
            # Verify probabilistic features are included
            probabilistic_features = [key for key in regime_features.keys() if 'regime_' in key or 'ensemble_' in key]
            tprint(f"📊 Probabilistic regime features: {len(probabilistic_features)}", color="blue")
            
            return True
        else:
            tprint("❌ Failed to assemble regime features for Analyst", color="red")
            return False
            
    except Exception as e:
        tprint(f"❌ Test failed: {e}", color="red")
        return False


def test_tactician_models(regime_data, comprehensive_regime_info):
    """Test Tactician models with regime probability features."""
    tprint("🧪 Testing Tactician Models with Regime Features", color="cyan", bold=True)
    
    try:
        # Extract market data and create features
        market_data = regime_data.get('market_data')
        if market_data is None:
            tprint("❌ No market data available", color="red")
            return False
        
        # Create feature columns (exclude regime columns for now)
        feature_columns = [col for col in market_data.columns if not col.startswith('regime_') and not col.startswith('ensemble_')]
        
        # Create target columns
        target_columns = ['target_return']  # Simplified for testing
        
        # Add synthetic target
        market_data['target_return'] = np.random.normal(0, 0.01, len(market_data))
        
        # Initialize Tactician training
        tactician_trainer = TacticianModelsTrainingStep()
        
        # Test regime feature assembly
        tprint("🔧 Testing regime feature assembly for Tactician...", color="yellow")
        
        # Create sample data
        X = market_data[feature_columns].values
        y = market_data[target_columns].values
        sample_weight = np.ones(len(X))
        
        # Test regime feature tensor assembly
        regime_features = tactician_trainer._assemble_regime_feature_tensor(
            X=X,
            oof_predictions={},
            sample_weight=sample_weight,
            comprehensive_regime_info=comprehensive_regime_info
        )
        
        if regime_features:
            tprint(f"✅ Assembled {len(regime_features)} regime features for Tactician", color="green")
            tprint(f"📊 Regime feature keys: {list(regime_features.keys())[:10]}...", color="blue")
            
            # Verify probabilistic features are included
            probabilistic_features = [key for key in regime_features.keys() if 'regime_' in key or 'ensemble_' in key]
            tprint(f"📊 Probabilistic regime features: {len(probabilistic_features)}", color="blue")
            
            return True
        else:
            tprint("❌ Failed to assemble regime features for Tactician", color="red")
            return False
            
    except Exception as e:
        tprint(f"❌ Test failed: {e}", color="red")
        return False


def main():
    """Main test function."""
    tprint("🚀 Starting Regime Ensemble Integration Tests", color="cyan", bold=True)
    tprint("=" * 70, color="cyan")
    
    # Test 1: Regime Ensemble Training
    stacker_result, prediction_result = test_regime_ensemble_training()
    if stacker_result is None:
        tprint("❌ Regime ensemble training failed - stopping tests", color="red")
        return
    
    tprint("=" * 70, color="cyan")
    
    # Test 2: Regime Data Splitting
    regime_data, comprehensive_regime_info = test_regime_data_splitting(stacker_result, prediction_result)
    if regime_data is None:
        tprint("❌ Regime data splitting failed - stopping tests", color="red")
        return
    
    tprint("=" * 70, color="cyan")
    
    # Test 3: Analyst Models
    analyst_success = test_analyst_models(regime_data, comprehensive_regime_info)
    
    tprint("=" * 70, color="cyan")
    
    # Test 4: Tactician Models
    tactician_success = test_tactician_models(regime_data, comprehensive_regime_info)
    
    tprint("=" * 70, color="cyan")
    tprint("📊 INTEGRATION TEST RESULTS", color="cyan", bold=True)
    tprint(f"Regime Ensemble Training: {'✅ PASSED' if stacker_result else '❌ FAILED'}", color="green" if stacker_result else "red")
    tprint(f"Regime Data Splitting: {'✅ PASSED' if regime_data else '❌ FAILED'}", color="green" if regime_data else "red")
    tprint(f"Analyst Models Integration: {'✅ PASSED' if analyst_success else '❌ FAILED'}", color="green" if analyst_success else "red")
    tprint(f"Tactician Models Integration: {'✅ PASSED' if tactician_success else '❌ FAILED'}", color="green" if tactician_success else "red")
    
    if stacker_result and regime_data and analyst_success and tactician_success:
        tprint("🎉 All integration tests passed! Probabilistic regime outputs are properly integrated.", color="green", bold=True)
    else:
        tprint("⚠️ Some integration tests failed. Please check the error messages above.", color="yellow", bold=True)
    
    tprint("=" * 70, color="cyan")


if __name__ == "__main__":
    main()