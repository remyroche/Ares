#!/usr/bin/env python3
"""
Comprehensive Enhanced Specialists Training with Detailed Metrics
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

# Import all enhanced specialists
try:
    from src.training.steps.market_analysis.ml_momentum_persistence_step_enhanced import EnhancedMLMomentumPersistenceStep
    from src.training.steps.market_analysis.ml_risk_regime_step_enhanced import EnhancedMLRiskRegimeStep
    from src.training.steps.market_analysis.ml_path_regime_step_enhanced import EnhancedMLPathRegimeStep
    from src.training.steps.market_analysis.ml_smc_regime_step_enhanced import EnhancedMLSMCRegimeStep
    from src.training.steps.market_analysis.ml_volume_force_step_enhanced import EnhancedMLVolumeForceStep
    from src.training.steps.market_analysis.ml_volatility_burst_step_enhanced import EnhancedMLVolatilityBurstStep
    from src.training.steps.market_analysis.xgb_macro_regime_step_enhanced import EnhancedXGBMacroRegimeStep
    from src.training.steps.market_analysis.xgb_meso_regime_step_enhanced import EnhancedXGBMesoRegimeStep
    from src.training.steps.market_analysis.ml_liquidity_regime_step_enhanced import EnhancedMLLiquidityRegimeStep
    from src.training.steps.market_analysis.ml_spectral_step_enhanced import EnhancedMLSpectralStep
    from src.training.steps.market_analysis.ml_microstructure_step_enhanced import EnhancedMLMicrostructureStep
    from src.training.steps.market_analysis.ml_candlestick_step_enhanced import EnhancedMLCandlestickStep
    from src.training.steps.market_analysis.ml_reversion_regime_step_enhanced import EnhancedMLReversionRegimeStep
    from src.training.steps.market_analysis.specialist_data_standard import SpecialistType
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def create_realistic_market_data():
    """Create realistic market data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=20000, freq='15T')
    
    # Create realistic price series with trends, volatility, and patterns
    returns = np.random.randn(20000) * 0.001
    
    # Add autocorrelation and momentum
    returns[1:] += 0.2 * returns[:-1]
    returns[2:] += 0.1 * returns[:-2]
    
    # Add regime-dependent volatility
    volatility_regime = np.random.choice([0.5, 1.0, 2.0], size=20000, p=[0.3, 0.5, 0.2])
    volatility_regime = pd.Series(volatility_regime).rolling(100).mean().fillna(1.0).values
    returns *= volatility_regime
    
    # Add occasional volatility spikes
    vol_spikes = np.random.random(20000) < 0.01
    returns[vol_spikes] *= np.random.uniform(3, 8, vol_spikes.sum())
    
    # Add trend components
    trend = np.sin(np.linspace(0, 50, 20000)) * 0.0002
    returns += trend
    
    # Generate prices
    price = 100 + np.cumsum(returns)
    
    # Generate OHLC with realistic spreads
    spread = np.abs(np.random.randn(20000) * 0.0003) * price
    high_noise = np.abs(np.random.randn(20000) * 0.0005) * price
    low_noise = np.abs(np.random.randn(20000) * 0.0005) * price
    
    df = pd.DataFrame({
        'open': price,
        'high': price + high_noise + spread/2,
        'low': price - low_noise - spread/2,
        'close': price + np.random.randn(20000) * 0.0001 * price,
        'volume': np.random.randint(5000, 50000, 20000)
    }, index=dates)
    df.index.name = 'timestamp'
    
    return df

def evaluate_specialist_comprehensive(specialist_class, specialist_name, market_data):
    """Comprehensive evaluation of a specialist with detailed metrics."""
    try:
        print(f"🔧 Evaluating {specialist_name}...")
        start_time = time.time()
        
        # Initialize specialist
        specialist = specialist_class(specialist_name)
        
        # Determine specialist type
        if 'momentum_persistence' in specialist_name.lower():
            specialist_type = SpecialistType.MOMENTUM_PERSISTENCE
        elif 'risk_regime' in specialist_name.lower():
            specialist_type = SpecialistType.RISK_REGIME
        elif 'path_regime' in specialist_name.lower():
            specialist_type = SpecialistType.PATH_REGIME
        elif 'smc_regime' in specialist_name.lower():
            specialist_type = SpecialistType.SMC_REGIME
        elif 'volume_force' in specialist_name.lower():
            specialist_type = SpecialistType.VOLUME_FORCE
        elif 'volatility_burst' in specialist_name.lower():
            specialist_type = SpecialistType.VOLATILITY_BURST
        elif 'liquidity_regime' in specialist_name.lower():
            specialist_type = SpecialistType.LIQUIDITY_REGIME
        elif 'reversion_regime' in specialist_name.lower():
            specialist_type = SpecialistType.REVERSION_REGIME
        elif 'macro_regime' in specialist_name.lower():
            specialist_type = SpecialistType.MACRO_REGIME
        elif 'meso_regime' in specialist_name.lower():
            specialist_type = SpecialistType.MESO_REGIME
        elif 'spectral' in specialist_name.lower():
            specialist_type = SpecialistType.SPECTRAL
        elif 'microstructure' in specialist_name.lower():
            specialist_type = SpecialistType.MICROSTRUCTURE
        elif 'candlestick' in specialist_name.lower():
            specialist_type = SpecialistType.CANDLESTICK
        else:
            specialist_type = SpecialistType.VOLUME_FORCE  # Default
        
        # Generate features
        try:
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
        
        # Basic statistics
        metrics['n_samples'] = int(len(X))
        metrics['n_features'] = int(len(X.columns))
        metrics['positive_labels'] = int(y.sum())
        metrics['negative_labels'] = int(len(y) - y.sum())
        metrics['label_balance'] = float(y.mean())
        
        # MI Score (detailed)
        try:
            mi_scores = mutual_info_regression(X, y)
            metrics['mi_score'] = float(np.mean(mi_scores))
            metrics['mi_max'] = float(np.max(mi_scores))
            metrics['mi_min'] = float(np.min(mi_scores))
            metrics['mi_std'] = float(np.std(mi_scores))
            metrics['mi_above_01'] = int(np.sum(mi_scores > 0.01))
            metrics['mi_above_02'] = int(np.sum(mi_scores > 0.02))
            metrics['mi_above_05'] = int(np.sum(mi_scores > 0.05))
        except Exception as e:
            metrics.update({
                'mi_score': 0.0, 'mi_max': 0.0, 'mi_min': 0.0, 'mi_std': 0.0,
                'mi_above_01': 0, 'mi_above_02': 0, 'mi_above_05': 0
            })
        
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
            if len(np.unique(y)) > 1:  # Check if we have both classes
                lr_clf = LogisticRegression(random_state=42, max_iter=1000)
                lr_clf.fit(X, y)
                y_pred_proba = lr_clf.predict_proba(X)[:, 1]
                metrics['auc_score'] = float(roc_auc_score(y, y_pred_proba))
                
                # Additional classification metrics
                y_pred_class = lr_clf.predict(X)
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                metrics['accuracy'] = float(accuracy_score(y, y_pred_class))
                metrics['precision'] = float(precision_score(y, y_pred_class, zero_division=0))
                metrics['recall'] = float(recall_score(y, y_pred_class, zero_division=0))
                metrics['f1_score'] = float(f1_score(y, y_pred_class, zero_division=0))
            else:
                metrics.update({
                    'auc_score': 0.5, 'accuracy': 0.5, 'precision': 0.0, 
                    'recall': 0.0, 'f1_score': 0.0
                })
        except:
            metrics.update({
                'auc_score': 0.5, 'accuracy': 0.5, 'precision': 0.0, 
                'recall': 0.0, 'f1_score': 0.0
            })
        
        # Feature quality metrics
        try:
            # Feature variance
            feature_var = X.var()
            metrics['feature_variance_mean'] = float(feature_var.mean())
            metrics['feature_variance_std'] = float(feature_var.std())
            metrics['zero_variance_features'] = int(np.sum(feature_var < 1e-8))
            
            # Feature correlation
            corr_matrix = X.corr()
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_pairs = (upper_triangle.abs() > 0.9).sum().sum()
            metrics['high_correlation_pairs'] = int(high_corr_pairs)
            metrics['mean_absolute_correlation'] = float(upper_triangle.abs().mean())
            
        except:
            metrics.update({
                'feature_variance_mean': 0.0, 'feature_variance_std': 0.0, 'zero_variance_features': 0,
                'high_correlation_pairs': 0, 'mean_absolute_correlation': 0.0
            })
        
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
    print("🚀 Comprehensive Enhanced Specialists Training & Evaluation")
    print("=" * 80)
    
    # Create test data
    print("📊 Creating realistic market data...")
    market_data = create_realistic_market_data()
    print(f"✅ Created {len(market_data)} rows of realistic test data")
    print(f"📈 Data range: {market_data.index[0]} to {market_data.index[-1]}")
    
    # Define all enhanced specialists
    specialists = {
        'EnhancedMLMomentumPersistenceStep': EnhancedMLMomentumPersistenceStep,
        'EnhancedMLRiskRegimeStep': EnhancedMLRiskRegimeStep,
        'EnhancedMLPathRegimeStep': EnhancedMLPathRegimeStep,
        'EnhancedMLSMCRegimeStep': EnhancedMLSMCRegimeStep,
        'EnhancedMLVolumeForceStep': EnhancedMLVolumeForceStep,
        'EnhancedMLVolatilityBurstStep': EnhancedMLVolatilityBurstStep,
        'EnhancedXGBMacroRegimeStep': EnhancedXGBMacroRegimeStep,
        'EnhancedXGBMesoRegimeStep': EnhancedXGBMesoRegimeStep,
        'EnhancedMLLiquidityRegimeStep': EnhancedMLLiquidityRegimeStep,
        'EnhancedMLSpectralStep': EnhancedMLSpectralStep,
        'EnhancedMLMicrostructureStep': EnhancedMLMicrostructureStep,
        'EnhancedMLCandlestickStep': EnhancedMLCandlestickStep,
        'EnhancedMLReversionRegimeStep': EnhancedMLReversionRegimeStep
    }
    
    # Evaluate all specialists
    all_results = {}
    
    for name, cls in specialists.items():
        print(f"\n{'='*60}")
        metrics = evaluate_specialist_comprehensive(cls, name, market_data)
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
        if 'high_correlation_pairs' in metrics:
            print(f"   High correlation pairs: {metrics['high_correlation_pairs']}")
    
    # Comprehensive summary table
    print(f"\n{'='*80}")
    print("🎯 COMPREHENSIVE ENHANCED SPECIALISTS PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    
    # Table header
    header = f"{'Specialist':<35} {'MI Score':<10} {'AUC':<8} {'R²':<8} {'Features':<10} {'Balance':<10} {'Status':<12}"
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
        
        print(f"{name:<35} {mi_score:<10.4f} {auc_score:<8.4f} {r2_score:<8.4f} {n_features:<10} {label_balance:<10.3f} {status:<12}")
    
    # Performance analysis
    print(f"\n{'='*80}")
    print("📈 PERFORMANCE ANALYSIS")
    print(f"{'='*80}")
    
    mi_scores = [m.get('mi_score', 0) for m in all_results.values() if m.get('mi_score', 0) > 0]
    if mi_scores:
        print(f"Total specialists evaluated: {len(all_results)}")
        print(f"Average MI score: {np.mean(mi_scores):.4f}")
        print(f"Best MI score: {np.max(mi_scores):.4f}")
        print(f"Worst MI score: {np.min(mi_scores):.4f}")
        print(f"MI score std: {np.std(mi_scores):.4f}")
        
        # Target achievement
        above_target = sum(1 for m in all_results.values() if m.get('mi_score', 0) > 0.02)
        print(f"Specialists above MI target (0.02): {above_target}/{len(all_results)} ({above_target/len(all_results)*100:.1f}%)")
        
        moderate = sum(1 for m in all_results.values() if 0.01 < m.get('mi_score', 0) <= 0.02)
        print(f"Specialists with moderate MI (0.01-0.02): {moderate}/{len(all_results)} ({moderate/len(all_results)*100:.1f}%)")
        
        low = sum(1 for m in all_results.values() if m.get('mi_score', 0) <= 0.01)
        print(f"Specialists with low MI (≤0.01): {low}/{len(all_results)} ({low/len(all_results)*100:.1f}%)")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"outcomes/enhanced_specialists_comprehensive_results_{timestamp}.json"
    
    try:
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n💾 Detailed results saved to: {results_file}")
    except Exception as e:
        print(f"\n⚠️ Could not save results file: {e}")
    
    return all_results

if __name__ == "__main__":
    main()
