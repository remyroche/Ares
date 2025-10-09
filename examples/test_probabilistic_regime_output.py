"""
Test script for probabilistic regime output functionality.

This script demonstrates how to use the enhanced regime models training and ensemble training
components to generate comprehensive probabilistic outputs for each detected regime.
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from training.steps.market_analysis.components.regime_models_training import RegimeModelsTrainingComponent
from training.steps.market_analysis.components.regime_ensemble_training import RegimeEnsembleTrainingComponent
from utils.regime_probability_analyzer import RegimeProbabilityAnalyzer
from utils.tprint import tprint


def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
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


def test_regime_models_training():
    """Test the regime models training component with probabilistic outputs."""
    tprint("🧪 Testing Regime Models Training Component", color="cyan", bold=True)
    
    try:
        # Create sample data
        data = create_sample_data(500)
        
        # Initialize component
        component = RegimeModelsTrainingComponent()
        
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
        tprint("🏋️ Training regime models...", color="yellow")
        result = component.execute(data, pipeline_state)
        
        if result.success:
            tprint("✅ Regime models training completed successfully", color="green")
            
            # Extract models and scaler from results
            models = result.artifacts.get('regime_models_training_result', {}).get('models', {})
            scaler = result.artifacts.get('regime_models_training_result', {}).get('scaler')
            feature_names = result.artifacts.get('regime_models_training_result', {}).get('feature_names', [])
            
            if models and scaler is not None:
                # Test probabilistic prediction
                tprint("🔮 Testing probabilistic regime prediction...", color="yellow")
                
                # Prepare test data
                X_test = np.random.randn(100, len(feature_names))
                
                # Make predictions with probabilities
                prediction_result = component.predict_regimes_with_probabilities(
                    models=models,
                    scaler=scaler,
                    X=X_test,
                    feature_names=feature_names,
                    use_meta_learner=True
                )
                
                if 'error' not in prediction_result:
                    tprint("✅ Probabilistic prediction completed successfully", color="green")
                    
                    # Analyze results
                    analyzer = RegimeProbabilityAnalyzer()
                    analysis = analyzer.analyze_regime_predictions(
                        prediction_result, 
                        "Regime Models Training"
                    )
                    
                    # Generate report
                    report = analyzer.generate_comprehensive_report(analysis)
                    tprint("📝 Analysis Report:", color="cyan")
                    print(report[:500] + "..." if len(report) > 500 else report)
                    
                    return True
                else:
                    tprint(f"❌ Prediction failed: {prediction_result.get('error')}", color="red")
                    return False
            else:
                tprint("❌ No trained models or scaler found", color="red")
                return False
        else:
            tprint(f"❌ Training failed: {result.error_message}", color="red")
            return False
            
    except Exception as e:
        tprint(f"❌ Test failed: {e}", color="red")
        return False


def test_regime_ensemble_training():
    """Test the regime ensemble training component with probabilistic outputs."""
    tprint("🧪 Testing Regime Ensemble Training Component", color="cyan", bold=True)
    
    try:
        # Create sample data
        data = create_sample_data(500)
        
        # Initialize component
        component = RegimeEnsembleTrainingComponent()
        
        # Create synthetic regime labels
        regime_labels = np.random.randint(0, 4, len(data))
        
        # Prepare pipeline state with some base models
        pipeline_state = {
            'artifacts': {
                'optimal_regime_clustering_result': {
                    'clustering_result': {
                        'cluster_assignments': regime_labels
                    }
                },
                'nas_tas_models_training_result': {
                    'models': {
                        'catboost': None,  # Will be trained if not available
                        'random_forest': None,
                        'extra_tree': None
                    }
                }
            }
        }
        
        # Execute training
        tprint("🏋️ Training regime ensemble...", color="yellow")
        result = component.execute(data, pipeline_state)
        
        if result.success:
            tprint("✅ Regime ensemble training completed successfully", color="green")
            
            # Extract stacker result from results
            stacker_result = result.artifacts.get('regime_ensemble_training_result', {}).get('stacker_lgbm_calibrated')
            
            if stacker_result:
                # Test probabilistic prediction
                tprint("🔮 Testing ensemble probabilistic prediction...", color="yellow")
                
                # Prepare test data
                X_test = np.random.randn(100, 50)  # 50 features
                feature_names = [f'feature_{i}' for i in range(50)]
                
                # Make predictions with probabilities
                prediction_result = component.predict_regimes_with_probabilities(
                    stacker_result=stacker_result,
                    X=X_test,
                    feature_names=feature_names,
                    scaler=None  # No scaler for this test
                )
                
                if 'error' not in prediction_result:
                    tprint("✅ Ensemble probabilistic prediction completed successfully", color="green")
                    
                    # Analyze results
                    analyzer = RegimeProbabilityAnalyzer()
                    analysis = analyzer.analyze_regime_predictions(
                        prediction_result, 
                        "Regime Ensemble Training"
                    )
                    
                    # Generate report
                    report = analyzer.generate_comprehensive_report(analysis)
                    tprint("📝 Ensemble Analysis Report:", color="cyan")
                    print(report[:500] + "..." if len(report) > 500 else report)
                    
                    return True
                else:
                    tprint(f"❌ Ensemble prediction failed: {prediction_result.get('error')}", color="red")
                    return False
            else:
                tprint("❌ No stacker result found", color="red")
                return False
        else:
            tprint(f"❌ Ensemble training failed: {result.error_message}", color="red")
            return False
            
    except Exception as e:
        tprint(f"❌ Test failed: {e}", color="red")
        return False


def main():
    """Main test function."""
    tprint("🚀 Starting Probabilistic Regime Output Tests", color="cyan", bold=True)
    tprint("=" * 60, color="cyan")
    
    # Test regime models training
    models_success = test_regime_models_training()
    
    tprint("=" * 60, color="cyan")
    
    # Test regime ensemble training
    ensemble_success = test_regime_ensemble_training()
    
    tprint("=" * 60, color="cyan")
    tprint("📊 TEST RESULTS SUMMARY", color="cyan", bold=True)
    tprint(f"Regime Models Training: {'✅ PASSED' if models_success else '❌ FAILED'}", color="green" if models_success else "red")
    tprint(f"Regime Ensemble Training: {'✅ PASSED' if ensemble_success else '❌ FAILED'}", color="green" if ensemble_success else "red")
    
    if models_success and ensemble_success:
        tprint("🎉 All tests passed! Probabilistic regime output functionality is working correctly.", color="green", bold=True)
    else:
        tprint("⚠️ Some tests failed. Please check the error messages above.", color="yellow", bold=True)
    
    tprint("=" * 60, color="cyan")


if __name__ == "__main__":
    main()