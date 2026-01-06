#!/usr/bin/env python3
"""
Simple Enhanced Specialists Test - Focused on Risk and Path Risk specialists
"""

import sys
sys.path.append('/Users/remyroche/Documents/Ares')

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import r2_score
import time
import json
from datetime import datetime

# Import only the working specialists
try:
    from src.training.steps.market_analysis.ml_risk_regime_step_enhanced import EnhancedMLRiskRegimeStep
    from src.training.steps.market_analysis.ml_path_regime_step_enhanced import EnhancedMLPathRegimeStep
    from src.training.steps.market_analysis.specialist_data_standard import SpecialistType
    print("✅ Successfully imported enhanced specialists")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def create_realistic_market_data():
    """Create realistic market data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=10000, freq='15T')
    
    # Create realistic price series with trends, volatility, and patterns
    returns = np.random.randn(10000) * 0.001
    
    # Add autocorrelation and momentum
    returns[1:] += 0.2 * returns[:-1]
    returns[2:] += 0.1 * returns[:-2]
    
    # Add regime-dependent volatility
    volatility_regime = np.random.choice([0.5, 1.0, 2.0], size=10000, p=[0.3, 0.5, 0.2])
    volatility_regime = pd.Series(volatility_regime).rolling(100).mean().fillna(1.0).values
    returns *= volatility_regime
    
    # Add occasional volatility spikes
    vol_spikes = np.random.random(10000) < 0.01
    returns[vol_spikes] *= np.random.uniform(3, 8, vol_spikes.sum())
    
    # Add trend components
    trend = np.sin(np.linspace(0, 50, 10000)) * 0.0002
    returns += trend
    
    # Generate prices
    price = 100 + np.cumsum(returns)
    
    # Generate OHLC with realistic spreads
    spread = np.abs(np.random.randn(10000) * 0.0003) * price
    high_noise = np.abs(np.random.randn(10000) * 0.0005) * price
    low_noise = np.abs(np.random.randn(10000) * 0.0005) * price
    
    df = pd.DataFrame({
        'open': price,
        'high': price + high_noise + spread/2,
        'low': price - low_noise - spread/2,
        'close': price + np.random.randn(10000) * 0.0001 * price,
        'volume': np.random.randint(5000, 50000, 10000)
    }, index=dates)
    df.index.name = 'timestamp'
    
    return df

def evaluate_specialist_simple(specialist_class, specialist_name, market_data):
    """Simple evaluation of a specialist with key metrics."""
    try:
        print(f"🔧 Evaluating {specialist_name}...")
        start_time = time.time()
        
        # Initialize specialist
        specialist = specialist_class(specialist_name)
        
        # Determine specialist type
        if 'risk_regime' in specialist_name.lower():
            specialist_type = SpecialistType.RISK_REGIME
        elif 'path_regime' in specialist_name.lower():
            specialist_type = SpecialistType.PATH_REGIME
        else:
            specialist_type = SpecialistType.VOLUME_FORCE  # Default
        
        # Generate features
        try:
            enhanced_features = specialist._generate_enhanced_features(market_data, specialist_type)
        except Exception as e:
            print(f"  ❌ Feature generation failed: {e}")
            return {'status': f'❌ Feature generation failed: {e}'}
        
        if enhanced_features.empty:
            return {'status': '❌ No features generated'}
        
        # Create labels using specialist's label creation method
        try:
            if hasattr(specialist, '_create_risk_labels'):
                labels = specialist._create_risk_labels(market_data)
            elif hasattr(specialist, '_create_path_labels'):
                labels = specialist._create_path_labels(market_data)
            else:
                # Default labels
                returns = market_data['close'].pct_change()
                labels = (returns > returns.rolling(20).std() * 0.5).astype(int)
        except Exception as e:
            print(f"  ❌ Label creation failed: {e}")
            return {'status': f'❌ Label creation failed: {e}'}
        
        # Align features and labels
        aligned_data = pd.concat([enhanced_features, labels], axis=1, join='inner')
        if aligned_data.empty:
            return {'status': '❌ No aligned data'}
        
        X = aligned_data.iloc[:, :-1].fillna(0)
        y = aligned_data.iloc[:, -1]
        
        # Calculate key metrics
        metrics = {}
        
        # Basic statistics
        metrics['n_samples'] = int(len(X))
        metrics['n_features'] = int(len(X.columns))
        metrics['positive_labels'] = int(y.sum())
        metrics['negative_labels'] = int(len(y) - y.sum())
        metrics['label_balance'] = float(y.mean())
        
        # MI Score
        try:
            mi_scores = mutual_info_regression(X, y)
            metrics['mi_score'] = float(np.mean(mi_scores))
            metrics['mi_max'] = float(np.max(mi_scores))
            metrics['mi_above_02'] = int(np.sum(mi_scores > 0.02))
        except Exception as e:
            metrics.update({'mi_score': 0.0, 'mi_max': 0.0, 'mi_above_02': 0})
        
        # R² Score
        try:
            lr = LinearRegression()
            lr.fit(X, y)
            y_pred = lr.predict(X)
            metrics['r2_score'] = float(r2_score(y, y_pred))
        except:
            metrics['r2_score'] = 0.0
        
        # AUC Score
        try:
            if len(np.unique(y)) > 1:
                lr_clf = LogisticRegression(random_state=42, max_iter=1000)
                lr_clf.fit(X, y)
                y_pred_proba = lr_clf.predict_proba(X)[:, 1]
                metrics['auc_score'] = float(roc_auc_score(y, y_pred_proba))
            else:
                metrics['auc_score'] = 0.5
        except:
            metrics['auc_score'] = 0.5
        
        # Processing time
        metrics['processing_time'] = float(time.time() - start_time)
        
        # Status determination
        if metrics['mi_score'] > 0.02:
            metrics['status'] = '✅ Success'
        elif metrics['mi_score'] > 0.01:
            metrics['status'] = '⚠️ Moderate'
        elif metrics['mi_score'] > 0.005:
            metrics['status'] = '⚠️ Low'
        else:
            metrics['status'] = '❌ Very Low'
        
        return metrics
        
    except Exception as e:
        return {'status': f'❌ Error: {str(e)}'}

def main():
    """Main training and evaluation function."""
    print("🚀 Enhanced Risk & Path Risk Specialists Training")
    print("=" * 60)
    
    # Create test data
    print("📊 Creating realistic market data...")
    market_data = create_realistic_market_data()
    print(f"✅ Created {len(market_data)} rows of realistic test data")
    
    # Define specialists to test
    specialists = {
        'EnhancedMLRiskRegimeStep': EnhancedMLRiskRegimeStep,
        'EnhancedMLPathRegimeStep': EnhancedMLPathRegimeStep
    }
    
    # Evaluate specialists
    all_results = {}
    
    for name, cls in specialists.items():
        print(f"\n{'='*50}")
        metrics = evaluate_specialist_simple(cls, name, market_data)
        all_results[name] = metrics
        
        # Print immediate results
        print(f"📊 {name} Results:")
        print(f"   Status: {metrics.get('status', 'Unknown')}")
        
        key_metrics = ['mi_score', 'auc_score', 'r2_score', 'n_features', 'label_balance', 'processing_time']
        for metric in key_metrics:
            if metric in metrics:
                if metric in ['mi_score', 'auc_score', 'r2_score', 'label_balance']:
                    print(f"   {metric.replace('_', ' ').title()}: {metrics[metric]:.4f}")
                elif metric in ['n_features', 'processing_time']:
                    print(f"   {metric.replace('_', ' ').title()}: {metrics[metric]}")
        
        # Additional detailed metrics
        if 'mi_above_02' in metrics:
            print(f"   Features with MI > 0.02: {metrics['mi_above_02']}")
    
    # Summary table
    print(f"\n{'='*70}")
    print("🎯 ENHANCED SPECIALISTS PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    
    # Table header
    header = f"{'Specialist':<30} {'MI Score':<10} {'AUC':<8} {'R²':<8} {'Features':<10} {'Balance':<10} {'Status':<12}"
    print(header)
    print("-" * len(header))
    
    # Sort by MI score
    sorted_results = sorted(all_results.items(), key=lambda x: x[1].get('mi_score', 0), reverse=True)
    
    for name, metrics in sorted_results:
        mi_score = metrics.get('mi_score', 0)
        auc_score = metrics.get('auc_score', 0.5)
        r2_score = metrics.get('r2_score', 0)
        n_features = metrics.get('n_features', 0)
        label_balance = metrics.get('label_balance', 0)
        status = metrics.get('status', 'Unknown')
        
        print(f"{name:<30} {mi_score:<10.4f} {auc_score:<8.4f} {r2_score:<8.4f} {n_features:<10} {label_balance:<10.3f} {status:<12}")
    
    # Performance analysis
    print(f"\n{'='*70}")
    print("📈 PERFORMANCE ANALYSIS")
    print(f"{'='*70}")
    
    mi_scores = [m.get('mi_score', 0) for m in all_results.values() if m.get('mi_score', 0) > 0]
    if mi_scores:
        print(f"Total specialists evaluated: {len(all_results)}")
        print(f"Average MI score: {np.mean(mi_scores):.4f}")
        print(f"Best MI score: {np.max(mi_scores):.4f}")
        print(f"Worst MI score: {np.min(mi_scores):.4f}")
        
        # Target achievement
        above_target = sum(1 for m in all_results.values() if m.get('mi_score', 0) > 0.02)
        print(f"Specialists above MI target (0.02): {above_target}/{len(all_results)} ({above_target/len(all_results)*100:.1f}%)")
        
        moderate = sum(1 for m in all_results.values() if 0.01 < m.get('mi_score', 0) <= 0.02)
        print(f"Specialists with moderate MI (0.01-0.02): {moderate}/{len(all_results)} ({moderate/len(all_results)*100:.1f}%)")
        
        low = sum(1 for m in all_results.values() if m.get('mi_score', 0) <= 0.01)
        print(f"Specialists with low MI (≤0.01): {low}/{len(all_results)} ({low/len(all_results)*100:.1f}%)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"outcomes/enhanced_specialists_simple_results_{timestamp}.json"
    
    try:
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n💾 Results saved to: {results_file}")
    except Exception as e:
        print(f"\n⚠️ Could not save results file: {e}")
    
    return all_results

if __name__ == "__main__":
    main()
