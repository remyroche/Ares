#!/usr/bin/env python3
"""
Comprehensive Enhanced Specialists Metrics Collection
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import subprocess
import sys

def run_specialist_diagnostics():
    """Run diagnostics for all enhanced specialists to get comparable metrics."""
    
    # List of all enhanced specialists
    enhanced_specialists = [
        'enhanced_ml_momentum_persistence_step',
        'enhanced_ml_risk_regime_step', 
        'enhanced_ml_path_regime_step',
        'enhanced_ml_smc_regime_step',
        'enhanced_ml_volume_force_step',
        'enhanced_ml_volatility_burst_step',
        'enhanced_xgb_macro_regime_step',
        'enhanced_xgb_meso_regime_step',
        'enhanced_ml_liquidity_regime_step',
        'enhanced_ml_spectral_step',
        'enhanced_ml_microstructure_step',
        'enhanced_ml_candlestick_step',
        'enhanced_ml_reversion_regime_step'
    ]
    
    print("🚀 Running Comprehensive Enhanced Specialists Diagnostics")
    print("=" * 70)
    
    # Create a simple training script to get metrics
    training_script = '''
import sys
sys.path.append('/Users/remyroche/Documents/Ares')

from src.training.steps.market_analysis.enhanced_ml_momentum_persistence_step import EnhancedMLMomentumPersistenceStep
from src.training.steps.market_analysis.enhanced_ml_risk_regime_step import EnhancedMLRiskRegimeStep
from src.training.steps.market_analysis.enhanced_ml_path_regime_step import EnhancedMLPathRegimeStep
from src.training.steps.market_analysis.enhanced_ml_smc_regime_step import EnhancedMLSMCRegimeStep
from src.training.steps.market_analysis.enhanced_ml_volume_force_step import EnhancedMLVolumeForceStep
from src.training.steps.market_analysis.enhanced_ml_volatility_burst_step import EnhancedMLVolatilityBurstStep
from src.training.steps.market_analysis.enhanced_xgb_macro_regime_step import EnhancedXGBMacroRegimeStep
from src.training.steps.market_analysis.enhanced_xgb_meso_regime_step import EnhancedXGBMesoRegimeStep
from src.training.steps.market_analysis.enhanced_ml_liquidity_regime_step import EnhancedMLLiquidityRegimeStep
from src.training.steps.market_analysis.enhanced_ml_spectral_step import EnhancedMLSpectralStep
from src.training.steps.market_analysis.enhanced_ml_microstructure_step import EnhancedMLMicrostructureStep
from src.training.steps.market_analysis.enhanced_ml_candlestick_step import EnhancedMLCandlestickStep
from src.training.steps.market_analysis.enhanced_ml_reversion_regime_step import EnhancedMLReversionRegimeStep

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, mutual_info_score
from sklearn.feature_selection import mutual_info_regression
import time

def create_mock_data():
    """Create mock market data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=5000, freq='15T')
    df = pd.DataFrame({
        'open': 100 + np.random.randn(5000).cumsum() * 0.1,
        'high': 100 + np.random.randn(5000).cumsum() * 0.1 + np.random.random(5000) * 0.5,
        'low': 100 + np.random.randn(5000).cumsum() * 0.1 - np.random.random(5000) * 0.5,
        'close': 100 + np.random.randn(5000).cumsum() * 0.1,
        'volume': np.random.randint(1000, 10000, 5000)
    }, index=dates)
    df.index.name = 'timestamp'
    return df

def evaluate_specialist(specialist_class, specialist_name):
    """Evaluate a single specialist and return metrics."""
    try:
        print(f"🔧 Evaluating {specialist_name}...")
        
        # Initialize specialist
        specialist = specialist_class(specialist_name)
        
        # Create mock data
        market_data = create_mock_data()
        
        # Generate features
        try:
            enhanced_features = specialist._generate_enhanced_features(market_data)
        except AttributeError:
            # Try alternative method names
            method_names = [
                '_compute_enhanced_spectral_features',
                '_compute_enhanced_microstructure_features', 
                '_compute_enhanced_candlestick_features',
                '_generate_enhanced_reversion_features'
            ]
            
            enhanced_features = None
            for method_name in method_names:
                try:
                    method = getattr(specialist, method_name)
                    enhanced_features = method(market_data)
                    break
                except AttributeError:
                    continue
            
            if enhanced_features is None:
                return {'status': '❌ No feature method found'}
        
        if enhanced_features.empty:
            return {'status': '❌ No features generated'}
        
        # Create labels for evaluation
        returns = market_data['close'].pct_change()
        labels = (returns > returns.rolling(20).std() * 0.5).astype(int)
        
        # Align features and labels
        aligned_data = pd.concat([enhanced_features, labels], axis=1, join='inner')
        if aligned_data.empty:
            return {'status': '❌ No aligned data'}
        
        X = aligned_data.iloc[:, :-1]
        y = aligned_data.iloc[:, -1]
        
        # Calculate metrics
        metrics = {}
        
        # MI Score (using mutual_info_regression for continuous features)
        try:
            X_clean = X.fillna(0)
            mi_scores = mutual_info_regression(X_clean, y)
            metrics['mi_score'] = float(np.mean(mi_scores))
        except:
            metrics['mi_score'] = 0.0
        
        # R² Score
        try:
            from sklearn.metrics import r2_score
            # Simple linear regression for R²
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            lr.fit(X.fillna(0), y)
            y_pred = lr.predict(X.fillna(0))
            metrics['r2_score'] = float(r2_score(y, y_pred))
        except:
            metrics['r2_score'] = 0.0
        
        # AUC Score (if binary classification possible)
        try:
            # Use a simple classifier
            from sklearn.linear_model import LogisticRegression
            lr_clf = LogisticRegression(random_state=42)
            lr_clf.fit(X.fillna(0), y)
            y_pred_proba = lr_clf.predict_proba(X.fillna(0))[:, 1]
            metrics['auc_score'] = float(roc_auc_score(y, y_pred_proba))
        except:
            metrics['auc_score'] = 0.5
        
        # Feature count
        metrics['n_features'] = int(len(X.columns))
        metrics['n_samples'] = int(len(X))
        
        # Status
        if metrics['mi_score'] > 0.02:
            metrics['status'] = '✅ Success'
        elif metrics['mi_score'] > 0.01:
            metrics['status'] = '⚠️ Moderate'
        else:
            metrics['status'] = '❌ Low MI'
        
        return metrics
        
    except Exception as e:
        return {'status': f'❌ Error: {str(e)}'}  # Fix the f-string syntax error

# Main evaluation
specialist_classes = {
    'enhanced_ml_momentum_persistence_step': EnhancedMLMomentumPersistenceStep,
    'enhanced_ml_risk_regime_step': EnhancedMLRiskRegimeStep,
    'enhanced_ml_path_regime_step': EnhancedMLPathRegimeStep,
    'enhanced_ml_smc_regime_step': EnhancedMLSMCRegimeStep,
    'enhanced_ml_volume_force_step': EnhancedMLVolumeForceStep,
    'enhanced_ml_volatility_burst_step': EnhancedMLVolatilityBurstStep,
    'enhanced_xgb_macro_regime_step': EnhancedXGBMacroRegimeStep,
    'enhanced_xgb_meso_regime_step': EnhancedXGBMesoRegimeStep,
    'enhanced_ml_liquidity_regime_step': EnhancedMLLiquidityRegimeStep,
    'enhanced_ml_spectral_step': EnhancedMLSpectralStep,
    'enhanced_ml_microstructure_step': EnhancedMLMicrostructureStep,
    'enhanced_ml_candlestick_step': EnhancedMLCandlestickStep,
    'enhanced_ml_reversion_regime_step': EnhancedMLReversionRegimeStep
}

results = {}
for name, cls in specialist_classes.items():
    metrics = evaluate_specialist(cls, name)
    results[name] = metrics
    print(f"✅ {name}: {metrics.get('status', 'Unknown')}")

# Save results
import json
with open('/tmp/enhanced_specialists_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)

print("🎉 Evaluation complete!")
'''
    
    # Save and run the evaluation script
    with open('/tmp/evaluate_enhanced_specialists.py', 'w') as f:
        f.write(training_script)
    
    try:
        result = subprocess.run([
            sys.executable, '/tmp/evaluate_enhanced_specialists.py'
        ], capture_output=True, text=True, cwd='/Users/remyroche/Documents/Ares')
        
        print("📊 Evaluation Results:")
        print(result.stdout)
        
        if result.stderr:
            print("⚠️ Errors:")
            print(result.stderr)
        
        # Load and display results
        try:
            with open('/tmp/enhanced_specialists_metrics.json', 'r') as f:
                results = json.load(f)
            
            print("\n🎯 Complete Enhanced Specialists Metrics")
            print("=" * 80)
            print(f"{'Specialist':<35} {'MI Score':<12} {'AUC Score':<12} {'R² Score':<12} {'Features':<10} {'Status':<15}")
            print("-" * 80)
            
            # Sort by MI score
            sorted_results = sorted(results.items(), 
                                 key=lambda x: x[1].get('mi_score', 0), 
                                 reverse=True)
            
            for name, metrics in sorted_results:
                mi_score = metrics.get('mi_score', 0)
                auc_score = metrics.get('auc_score', 0.5)
                r2_score = metrics.get('r2_score', 0)
                n_features = metrics.get('n_features', 0)
                status = metrics.get('status', 'Unknown')
                
                print(f"{name:<35} {mi_score:<12.4f} {auc_score:<12.4f} {r2_score:<12.4f} {n_features:<10} {status:<15}")
            
            return results
            
        except Exception as e:
            print(f"❌ Failed to load results: {e}")
            return {}
            
    except Exception as e:
        print(f"❌ Failed to run evaluation: {e}")
        return {}

if __name__ == "__main__":
    run_specialist_diagnostics()
