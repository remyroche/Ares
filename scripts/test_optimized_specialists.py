#!/usr/bin/env python3
"""
Test Optimized Risk and Path Risk Specialists
"""

import sys
sys.path.append('/Users/remyroche/Documents/Ares')

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LogisticRegression
import time

def create_mock_data():
    """Create realistic mock market data."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=10000, freq='15T')
    
    # Create realistic price series with trends and volatility
    returns = np.random.randn(10000) * 0.001
    # Add some autocorrelation
    returns[1:] += 0.3 * returns[:-1]
    # Add occasional volatility spikes
    vol_spikes = np.random.random(10000) < 0.02
    returns[vol_spikes] *= np.random.uniform(2, 5, vol_spikes.sum())
    
    # Generate prices
    price = 100 + np.cumsum(returns)
    
    # Generate OHLC from returns
    high_noise = np.abs(np.random.randn(10000) * 0.0005)
    low_noise = np.abs(np.random.randn(10000) * 0.0005)
    
    df = pd.DataFrame({
        'open': price,
        'high': price + high_noise,
        'low': price - low_noise,
        'close': price + np.random.randn(10000) * 0.0002,
        'volume': np.random.randint(1000, 10000, 10000)
    }, index=dates)
    df.index.name = 'timestamp'
    
    return df

def evaluate_specialist(specialist_class, specialist_name, market_data):
    """Evaluate a single specialist and return comprehensive metrics."""
    try:
        print(f"🔧 Evaluating {specialist_name}...")
        start_time = time.time()
        
        # Initialize specialist
        specialist = specialist_class(specialist_name)
        
        # Generate features
        try:
            # Import SpecialistType
            from src.training.steps.market_analysis.specialist_data_standard import SpecialistType
            
            # Determine specialist type
            if 'risk_regime' in specialist_name.lower():
                specialist_type = SpecialistType.RISK_REGIME
            elif 'path_regime' in specialist_name.lower():
                specialist_type = SpecialistType.PATH_REGIME
            elif 'momentum_persistence' in specialist_name.lower():
                specialist_type = SpecialistType.MOMENTUM_PERSISTENCE
            elif 'smc_regime' in specialist_name.lower():
                specialist_type = SpecialistType.SMC_REGIME
            elif 'volume_force' in specialist_name.lower():
                specialist_type = SpecialistType.VOLUME_FORCE
            elif 'volatility_burst' in specialist_name.lower():
                specialist_type = SpecialistType.VOLATILITY_BURST
            else:
                specialist_type = SpecialistType.VOLUME_FORCE  # Default
            
            enhanced_features = specialist._generate_enhanced_features(market_data, specialist_type)
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
                    print(f"  ✅ Used {method_name}")
                    break
                except AttributeError:
                    continue
            
            if enhanced_features is None:
                return {'status': '❌ No feature method found'}
        
        if enhanced_features.empty:
            return {'status': '❌ No features generated'}
        
        # Create labels using specialist's label creation method
        if hasattr(specialist, '_create_risk_labels'):
            labels = specialist._create_risk_labels(market_data)
        elif hasattr(specialist, '_create_path_labels'):
            labels = specialist._create_path_labels(market_data)
        else:
            # Default labels
            returns = market_data['close'].pct_change()
            labels = (returns > returns.rolling(20).std() * 0.5).astype(int)
        
        # Align features and labels
        aligned_data = pd.concat([enhanced_features, labels], axis=1, join='inner')
        if aligned_data.empty:
            return {'status': '❌ No aligned data'}
        
        X = aligned_data.iloc[:, :-1].fillna(0)
        y = aligned_data.iloc[:, -1]
        
        # Calculate comprehensive metrics
        metrics = {}
        
        # MI Score
        try:
            mi_scores = mutual_info_regression(X, y)
            metrics['mi_score'] = float(np.mean(mi_scores))
            metrics['mi_max'] = float(np.max(mi_scores))
        except:
            metrics['mi_score'] = 0.0
            metrics['mi_max'] = 0.0
        
        # R² Score
        try:
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            lr.fit(X, y)
            y_pred = lr.predict(X)
            from sklearn.metrics import r2_score
            metrics['r2_score'] = float(r2_score(y, y_pred))
        except:
            metrics['r2_score'] = 0.0
        
        # AUC Score
        try:
            lr_clf = LogisticRegression(random_state=42, max_iter=1000)
            lr_clf.fit(X, y)
            y_pred_proba = lr_clf.predict_proba(X)[:, 1]
            metrics['auc_score'] = float(roc_auc_score(y, y_pred_proba))
        except:
            metrics['auc_score'] = 0.5
        
        # Label statistics
        metrics['label_balance'] = float(y.mean())
        metrics['label_count'] = int(len(y))
        metrics['positive_count'] = int(y.sum())
        
        # Feature statistics
        metrics['n_features'] = int(len(X.columns))
        metrics['n_samples'] = int(len(X))
        
        # Processing time
        metrics['processing_time'] = float(time.time() - start_time)
        
        # Status
        if metrics['mi_score'] > 0.02:
            metrics['status'] = '✅ Success'
        elif metrics['mi_score'] > 0.01:
            metrics['status'] = '⚠️ Moderate'
        else:
            metrics['status'] = '❌ Low MI'
        
        return metrics
        
    except Exception as e:
        return {'status': f'❌ Error: {str(e)}'}

def main():
    """Main evaluation function."""
    print("🚀 Testing Optimized Risk and Path Risk Specialists")
    print("=" * 60)
    
    # Import specialists
    try:
        from src.training.steps.market_analysis.ml_risk_regime_step_enhanced import EnhancedMLRiskRegimeStep
        from src.training.steps.market_analysis.ml_path_regime_step_enhanced import EnhancedMLPathRegimeStep
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # Create test data
    print("📊 Creating test data...")
    market_data = create_mock_data()
    print(f"✅ Created {len(market_data)} rows of test data")
    
    # Test specialists
    specialists = {
        'EnhancedMLRiskRegimeStep': EnhancedMLRiskRegimeStep,
        'EnhancedMLPathRegimeStep': EnhancedMLPathRegimeStep
    }
    
    results = {}
    
    for name, cls in specialists.items():
        metrics = evaluate_specialist(cls, name, market_data)
        results[name] = metrics
        
        # Print immediate results
        print(f"\n📊 {name} Results:")
        print(f"   Status: {metrics.get('status', 'Unknown')}")
        if 'mi_score' in metrics:
            print(f"   MI Score: {metrics['mi_score']:.4f}")
        if 'auc_score' in metrics:
            print(f"   AUC Score: {metrics['auc_score']:.4f}")
        if 'r2_score' in metrics:
            print(f"   R² Score: {metrics['r2_score']:.4f}")
        if 'n_features' in metrics:
            print(f"   Features: {metrics['n_features']}")
        if 'label_balance' in metrics:
            print(f"   Label Balance: {metrics['label_balance']:.3f}")
        if 'processing_time' in metrics:
            print(f"   Processing Time: {metrics['processing_time']:.2f}s")
    
    # Summary comparison
    print(f"\n🎯 Optimization Summary:")
    print("=" * 60)
    print(f"{'Specialist':<25} {'MI Score':<12} {'AUC Score':<12} {'R² Score':<12} {'Status':<15}")
    print("-" * 60)
    
    for name, metrics in results.items():
        mi_score = metrics.get('mi_score', 0)
        auc_score = metrics.get('auc_score', 0.5)
        r2_score = metrics.get('r2_score', 0)
        status = metrics.get('status', 'Unknown')
        
        print(f"{name:<25} {mi_score:<12.4f} {auc_score:<12.4f} {r2_score:<12.4f} {status:<15}")
    
    # Improvement assessment
    print(f"\n📈 Improvement Assessment:")
    
    risk_mi = results.get('EnhancedMLRiskRegimeStep', {}).get('mi_score', 0)
    path_mi = results.get('EnhancedMLPathRegimeStep', {}).get('mi_score', 0)
    
    print(f"Risk Specialist MI: {risk_mi:.4f} (Target: >0.02)")
    print(f"Path Risk Specialist MI: {path_mi:.4f} (Target: >0.02)")
    
    if risk_mi > 0.02:
        print(f"✅ Risk specialist meets target!")
    else:
        improvement_needed = 0.02 - risk_mi
        print(f"⚠️ Risk specialist needs {improvement_needed:.4f} MI improvement")
    
    if path_mi > 0.02:
        print(f"✅ Path risk specialist meets target!")
    else:
        improvement_needed = 0.02 - path_mi
        print(f"⚠️ Path risk specialist needs {improvement_needed:.4f} MI improvement")
    
    return results

if __name__ == "__main__":
    main()
