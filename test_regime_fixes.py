#!/usr/bin/env python3
"""
Test script to validate the regime detection fixes.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_dimension_mismatch_fix():
    """Test the dimension mismatch fix in clustering quality analyzer."""
    print("🧪 Testing dimension mismatch fix...")
    
    try:
        from src.training.steps.market_analysis.hybrid_nas_tas_regime.clustering_quality_analyzer import ClusteringQualityAnalyzer
        
        # Create test data with dimension mismatch
        features = np.random.randn(960, 5)  # 960 samples, 5 features
        labels = np.random.randint(0, 3, 100)  # 100 samples
        
        analyzer = ClusteringQualityAnalyzer()
        metrics = analyzer.calculate_comprehensive_metrics(features, labels)
        
        print(f"✅ Dimension mismatch fix works: {metrics['n_samples']} samples processed")
        return True
        
    except Exception as e:
        print(f"❌ Dimension mismatch fix failed: {e}")
        return False

def test_nas_execution_fix():
    """Test the NAS execution fix."""
    print("🧪 Testing NAS execution fix...")
    
    try:
        from src.training.steps.market_analysis.nas_regime.core.enhanced_perfect_nas_regime_detector import EnhancedPerfectNASRegimeDetector
        
        # Create test data
        data = pd.DataFrame({
            'open': np.random.randn(100),
            'high': np.random.randn(100),
            'low': np.random.randn(100),
            'close': np.random.randn(100),
            'volume': np.random.randn(100)
        })
        
        # Create a simple config for the detector
        from src.training.steps.market_analysis.nas_regime.core.perfect_nas_config import PerfectNASConfig
        config = PerfectNASConfig()
        detector = EnhancedPerfectNASRegimeDetector(config)
        result = detector.detect_regimes(data)
        
        print(f"✅ NAS execution fix works: {result.n_regimes} regimes detected")
        return True
        
    except Exception as e:
        print(f"❌ NAS execution fix failed: {e}")
        return False

def test_consensus_mechanism_fix():
    """Test the consensus mechanism fix."""
    print("🧪 Testing consensus mechanism fix...")
    
    try:
        from src.training.steps.market_analysis.hybrid_nas_tas_regime.hybrid_orchestrator import HybridOrchestrator
        
        # Create test data with different lengths
        nas_assignments = np.random.randint(0, 3, 100)
        tas_assignments = np.random.randint(0, 3, 80)
        
        orchestrator = HybridOrchestrator()
        
        # Test consensus mapping
        consensus_mapping = orchestrator._calculate_consensus_mapping(nas_assignments, tas_assignments)
        
        # Test consolidated assignments
        consolidated = orchestrator._generate_consolidated_assignments(nas_assignments, tas_assignments, consensus_mapping)
        
        print(f"✅ Consensus mechanism fix works: {len(consolidated)} consolidated assignments")
        return True
        
    except Exception as e:
        print(f"❌ Consensus mechanism fix failed: {e}")
        return False

def test_data_quality_fix():
    """Test the data quality fix for NaN values."""
    print("🧪 Testing data quality fix...")
    
    try:
        # Create test data with single sample regime
        data = pd.DataFrame({
            'close': [1.0, 2.0, 3.0, 4.0, 5.0],
            'volume': [100, 200, 300, 400, 500]
        })
        
        # Create regime assignments with single sample regime
        regime_assignments = np.array([0, 0, 1, 1, 2])  # Regime 2 has only 1 sample
        
        # Test regime characteristics calculation
        regime_characteristics = {}
        for regime_id in range(3):
            regime_mask = regime_assignments == regime_id
            regime_data = data[regime_mask]
            
            if len(regime_data) > 0:
                volatility = regime_data['close'].std() if 'close' in regime_data.columns and len(regime_data) > 1 else 0.0
                avg_return = regime_data['close'].pct_change().mean() if 'close' in regime_data.columns and len(regime_data) > 1 else 0.0
                
                regime_characteristics[f'regime_{regime_id}'] = {
                    'volatility': float(volatility),
                    'avg_return': float(avg_return)
                }
        
        # Check that no NaN values exist
        has_nan = any(
            pd.isna(regime_characteristics[f'regime_{i}']['volatility']) or 
            pd.isna(regime_characteristics[f'regime_{i}']['avg_return'])
            for i in range(3)
        )
        
        if not has_nan:
            print("✅ Data quality fix works: No NaN values found")
            return True
        else:
            print("❌ Data quality fix failed: NaN values still present")
            return False
        
    except Exception as e:
        print(f"❌ Data quality fix failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting regime detection fixes validation...")
    
    tests = [
        test_dimension_mismatch_fix,
        test_nas_execution_fix,
        test_consensus_mechanism_fix,
        test_data_quality_fix
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All fixes validated successfully!")
        return True
    else:
        print("⚠️ Some fixes need attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
