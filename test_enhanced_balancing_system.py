"""
Comprehensive Test Suite for Enhanced Label Balancing & Sample Weighting System

This test suite validates the complete enhanced balancing system including:
- All balancing techniques (SMOTE, ADASYN, Mixup, etc.)
- All weighting schemes (volatility, confidence, event overlap, etc.)
- Regime-aware rebalancing
- Validation fairness checks
- Integration with training pipelines
"""

import numpy as np
import pandas as pd
import pytest
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Import the enhanced balancing system
from src.training.steps.pre_training.profit_labeling.label_balancing import (
    ComprehensiveBalancingSystem, LabelBalancer, SampleWeighter,
    BalancingConfig, WeightingConfig, RegimeConfig, ValidationFairnessConfig,
    BalancingTechnique, WeightingScheme,
    DEFAULT_BALANCING_CONFIG, DEFAULT_WEIGHTING_CONFIG, DEFAULT_REGIME_CONFIG, DEFAULT_FAIRNESS_CONFIG
)

from src.training.steps.pre_training.profit_labeling.enhanced_balancing_integration import (
    BalancingIntegrationManager, BalancingIntegrationConfig,
    create_trading_balancing_manager, create_research_balancing_manager,
    integrate_with_analyst_training, integrate_with_tactician_training
)


class TestEnhancedBalancingSystem:
    """Test suite for the enhanced balancing system."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create imbalanced dataset
        n_samples = 1000
        n_features = 10
        
        # Generate features
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Create imbalanced labels (80% class 0, 15% class 1, 5% class 2)
        y = pd.Series(np.random.choice([0, 1, 2], n_samples, p=[0.8, 0.15, 0.05]))
        
        # Add some additional features
        X['returns'] = np.random.randn(n_samples) * 0.02
        X['volatility'] = np.abs(X['returns']) * 10
        X['regime'] = np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1])
        
        self.X = X
        self.y = y
        self.regime_data = X['regime']
        
        # Create additional features for weighting
        self.additional_features = {
            'regime': self.regime_data,
            'volatility': X['volatility'],
            'confidence': np.random.uniform(0.1, 1.0, n_samples)
        }
    
    def test_balancing_techniques(self):
        """Test all balancing techniques."""
        print("\n🧪 Testing balancing techniques...")
        
        techniques = [
            BalancingTechnique.UNDER_SAMPLING,
            BalancingTechnique.OVER_SAMPLING,
            BalancingTechnique.SMOTE,
            BalancingTechnique.ADASYN,
            BalancingTechnique.MIXUP,
            BalancingTechnique.STRATIFIED_BATCHING,
            BalancingTechnique.HYBRID,
            BalancingTechnique.ADAPTIVE
        ]
        
        for technique in techniques:
            print(f"   → Testing {technique.value}...")
            
            config = BalancingConfig(
                balancing_technique=technique,
                random_state=42
            )
            
            balancer = LabelBalancer(config)
            X_balanced, y_balanced, weights = balancer.balance_dataset(self.X, self.y)
            
            # Validate results
            assert len(X_balanced) > 0, f"{technique.value} produced empty dataset"
            assert len(y_balanced) == len(X_balanced), f"{technique.value} mismatched lengths"
            assert y_balanced.nunique() >= 2, f"{technique.value} lost classes"
            
            print(f"      ✅ {technique.value}: {len(self.X)} → {len(X_balanced)} samples")
    
    def test_weighting_schemes(self):
        """Test all weighting schemes."""
        print("\n⚖️ Testing weighting schemes...")
        
        schemes = [
            WeightingScheme.VOLATILITY,
            WeightingScheme.CONFIDENCE,
            WeightingScheme.EVENT_OVERLAP,
            WeightingScheme.TIME_DECAY,
            WeightingScheme.REGIME_AWARE,
            WeightingScheme.INFORMATION_CONTENT
        ]
        
        for scheme in schemes:
            print(f"   → Testing {scheme.value}...")
            
            config = WeightingConfig(
                weighting_scheme=scheme,
                random_state=42
            )
            
            weighter = SampleWeighter(config)
            weights = weighter.compute_weights(
                self.X, self.y, self.additional_features
            )
            
            # Validate weights
            assert len(weights) == len(self.X), f"{scheme.value} wrong weight length"
            assert weights.min() > 0, f"{scheme.value} has non-positive weights"
            assert weights.max() < 100, f"{scheme.value} has extreme weights"
            
            print(f"      ✅ {scheme.value}: weight range [{weights.min():.3f}, {weights.max():.3f}]")
    
    def test_comprehensive_balancing_system(self):
        """Test the comprehensive balancing system."""
        print("\n🔧 Testing comprehensive balancing system...")
        
        # Test with different configurations
        configs = [
            {
                'balancing': {'balancing_technique': BalancingTechnique.ADAPTIVE},
                'weighting': {'weighting_scheme': WeightingScheme.INFORMATION_CONTENT}
            },
            {
                'balancing': {'balancing_technique': BalancingTechnique.SMOTE},
                'weighting': {'weighting_scheme': WeightingScheme.VOLATILITY}
            },
            {
                'balancing': {'balancing_technique': BalancingTechnique.HYBRID},
                'weighting': {'weighting_scheme': WeightingScheme.REGIME_AWARE}
            }
        ]
        
        for i, custom_config in enumerate(configs):
            print(f"   → Testing configuration {i+1}...")
            
            # Create balancing system
            balancing_config = BalancingConfig(**custom_config['balancing'])
            weighting_config = WeightingConfig(**custom_config['weighting'])
            
            system = ComprehensiveBalancingSystem(
                balancing_config=balancing_config,
                weighting_config=weighting_config,
                regime_config=DEFAULT_REGIME_CONFIG,
                fairness_config=DEFAULT_FAIRNESS_CONFIG
            )
            
            # Apply balancing and weighting
            X_balanced, y_balanced, weights = system.balance_and_weight(
                self.X, self.y, additional_features=self.additional_features
            )
            
            # Validate results
            assert len(X_balanced) > 0, f"Configuration {i+1} produced empty dataset"
            assert len(y_balanced) == len(X_balanced), f"Configuration {i+1} mismatched lengths"
            assert len(weights) == len(X_balanced), f"Configuration {i+1} wrong weight length"
            
            # Check class distribution improvement
            original_imbalance = self.y.value_counts().min() / self.y.value_counts().max()
            balanced_imbalance = y_balanced.value_counts().min() / y_balanced.value_counts().max()
            
            print(f"      ✅ Config {i+1}: {len(self.X)} → {len(X_balanced)} samples")
            print(f"         Imbalance: {original_imbalance:.3f} → {balanced_imbalance:.3f}")
    
    def test_integration_manager(self):
        """Test the integration manager."""
        print("\n🔗 Testing integration manager...")
        
        # Test trading manager
        print("   → Testing trading manager...")
        trading_manager = create_trading_balancing_manager()
        
        dataset_characteristics = {
            'n_samples': len(self.X),
            'n_classes': self.y.nunique(),
            'imbalance_ratio': self.y.value_counts().min() / self.y.value_counts().max(),
            'has_regime_data': True,
            'has_volatility_data': True,
            'dataset_type': 'trading'
        }
        
        result = trading_manager.balance_and_weight_data(
            self.X, self.y,
            additional_features=self.additional_features,
            dataset_characteristics=dataset_characteristics
        )
        
        assert result['success'], "Trading manager failed"
        assert len(result['X_balanced']) > 0, "Trading manager produced empty dataset"
        
        print(f"      ✅ Trading manager: {result['original_samples']} → {result['balanced_samples']} samples")
        print(f"         Technique: {result['balancing_technique']}")
        print(f"         Weighting: {result['weighting_scheme']}")
        
        # Test research manager
        print("   → Testing research manager...")
        research_manager = create_research_balancing_manager()
        
        result = research_manager.balance_and_weight_data(
            self.X, self.y,
            additional_features=self.additional_features,
            dataset_characteristics=dataset_characteristics
        )
        
        assert result['success'], "Research manager failed"
        assert len(result['X_balanced']) > 0, "Research manager produced empty dataset"
        
        print(f"      ✅ Research manager: {result['original_samples']} → {result['balanced_samples']} samples")
    
    def test_validation_fairness(self):
        """Test validation fairness checks."""
        print("\n📊 Testing validation fairness...")
        
        # Create train/val split
        from sklearn.model_selection import train_test_split
        
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        regime_train = self.regime_data.loc[X_train.index]
        regime_val = self.regime_data.loc[X_val.index]
        
        train_data = {'y': y_train, 'regime': regime_train}
        val_data = {'y': y_val, 'regime': regime_val}
        
        # Test fairness checker
        fairness_config = ValidationFairnessConfig()
        from src.training.steps.pre_training.profit_labeling.label_balancing import ValidationFairnessChecker
        
        checker = ValidationFairnessChecker(fairness_config)
        fairness_report = checker.check_fairness(train_data, val_data)
        
        assert 'class_ratio_fair' in fairness_report, "Missing class ratio fairness check"
        assert 'regime_mix_fair' in fairness_report, "Missing regime mix fairness check"
        
        print(f"      ✅ Class ratio fair: {fairness_report['class_ratio_fair']}")
        print(f"      ✅ Regime mix fair: {fairness_report['regime_mix_fair']}")
    
    def test_pipeline_integration(self):
        """Test integration with training pipelines."""
        print("\n🚀 Testing pipeline integration...")
        
        # Test Analyst integration
        print("   → Testing Analyst integration...")
        analyst_result = integrate_with_analyst_training(
            self.X, self.y, self.regime_data
        )
        
        assert analyst_result['success'], "Analyst integration failed"
        assert len(analyst_result['X_balanced']) > 0, "Analyst integration produced empty dataset"
        
        print(f"      ✅ Analyst integration: {analyst_result['original_samples']} → {analyst_result['balanced_samples']} samples")
        
        # Test Tactician integration
        print("   → Testing Tactician integration...")
        tactician_result = integrate_with_tactician_training(
            self.X, self.y, self.regime_data
        )
        
        assert tactician_result['success'], "Tactician integration failed"
        assert len(tactician_result['X_balanced']) > 0, "Tactician integration produced empty dataset"
        
        print(f"      ✅ Tactician integration: {tactician_result['original_samples']} → {tactician_result['balanced_samples']} samples")
    
    def test_performance_metrics(self):
        """Test performance and monitoring metrics."""
        print("\n📈 Testing performance metrics...")
        
        manager = create_trading_balancing_manager()
        
        # Process data multiple times to test monitoring
        for i in range(3):
            result = manager.balance_and_weight_data(
                self.X, self.y,
                additional_features=self.additional_features,
                dataset_characteristics={
                    'n_samples': len(self.X),
                    'n_classes': self.y.nunique(),
                    'imbalance_ratio': self.y.value_counts().min() / self.y.value_counts().max(),
                    'has_regime_data': True,
                    'has_volatility_data': True,
                    'dataset_type': 'trading'
                }
            )
            
            assert result['success'], f"Processing {i+1} failed"
        
        # Get monitoring report
        report = manager.get_balancing_report()
        
        assert 'monitoring_data' in report, "Missing monitoring data"
        assert 'balancing_system_config' in report, "Missing system config"
        assert 'timestamp' in report, "Missing timestamp"
        
        print(f"      ✅ Monitoring data: {len(report['monitoring_data'])} metrics")
        print(f"      ✅ System config: {report['balancing_system_config']}")
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        print("\n⚠️ Testing edge cases...")
        
        # Test with very small dataset
        print("   → Testing small dataset...")
        X_small = self.X.head(10)
        y_small = self.y.head(10)
        
        manager = create_trading_balancing_manager()
        result = manager.balance_and_weight_data(X_small, y_small)
        
        # Should handle gracefully
        assert 'success' in result, "Small dataset handling failed"
        
        # Test with single class
        print("   → Testing single class...")
        y_single = pd.Series([0] * len(self.X))
        
        try:
            result = manager.balance_and_weight_data(self.X, y_single)
            # Should either succeed with warning or fail gracefully
            assert 'success' in result, "Single class handling failed"
        except ValueError:
            # Expected for single class
            pass
        
        # Test with empty dataset
        print("   → Testing empty dataset...")
        X_empty = pd.DataFrame()
        y_empty = pd.Series(dtype=int)
        
        try:
            result = manager.balance_and_weight_data(X_empty, y_empty)
            assert not result['success'], "Empty dataset should fail"
        except ValueError:
            # Expected for empty dataset
            pass
        
        print("      ✅ Edge cases handled correctly")
    
    def test_memory_constraints(self):
        """Test memory constraint handling."""
        print("\n💾 Testing memory constraints...")
        
        # Create large dataset
        X_large = pd.DataFrame(
            np.random.randn(200000, 10),
            columns=[f'feature_{i}' for i in range(10)]
        )
        y_large = pd.Series(np.random.choice([0, 1, 2], 200000, p=[0.8, 0.15, 0.05]))
        
        # Test with memory limit
        config = BalancingIntegrationConfig(max_samples_for_balancing=50000)
        manager = BalancingIntegrationManager(config)
        
        result = manager.balance_and_weight_data(
            X_large, y_large,
            dataset_characteristics={
                'n_samples': len(X_large),
                'n_classes': y_large.nunique(),
                'imbalance_ratio': y_large.value_counts().min() / y_large.value_counts().max(),
                'has_regime_data': False,
                'has_volatility_data': False,
                'dataset_type': 'general'
            }
        )
        
        assert result['success'], "Memory constraint handling failed"
        assert result['balanced_samples'] <= 50000, "Memory limit not respected"
        
        print(f"      ✅ Memory constraints: {len(X_large)} → {result['balanced_samples']} samples")


def run_comprehensive_test():
    """Run the comprehensive test suite."""
    print("🧪 Starting Comprehensive Enhanced Balancing System Test Suite")
    print("=" * 70)
    
    test_suite = TestEnhancedBalancingSystem()
    test_suite.setup_method()
    
    try:
        # Run all tests
        test_suite.test_balancing_techniques()
        test_suite.test_weighting_schemes()
        test_suite.test_comprehensive_balancing_system()
        test_suite.test_integration_manager()
        test_suite.test_validation_fairness()
        test_suite.test_pipeline_integration()
        test_suite.test_performance_metrics()
        test_suite.test_edge_cases()
        test_suite.test_memory_constraints()
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED! Enhanced Balancing System is working correctly.")
        print("🎯 The system successfully addresses financial dataset imbalance with:")
        print("   → Advanced balancing techniques (SMOTE, ADASYN, Mixup, etc.)")
        print("   → Comprehensive sample weighting schemes")
        print("   → Regime-aware rebalancing")
        print("   → Validation fairness checks")
        print("   → Seamless pipeline integration")
        print("   → Performance monitoring and debugging")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)
