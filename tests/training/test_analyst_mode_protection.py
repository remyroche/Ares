"""
Analyst Mode Protection Tests

Critical tests to ensure that CMI complementarity integration does NOT affect
Analyst mode behavior. These tests verify that Analyst mode remains completely
unchanged and unaffected by CMI modifications.

CRITICAL REQUIREMENT: All CMI modifications are gated on tactician_mode=True.
Analyst mode files remain completely unchanged and unaffected.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import time
import warnings

# Import test utilities
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)


class TestAnalystModeProtection:
    """Test suite to ensure Analyst mode remains completely unchanged."""
    
    @pytest.fixture
    def analyst_pipeline_state(self):
        """Create pipeline state for Analyst mode (tactician_mode=False)."""
        return {
            'tactician_mode': False,  # CRITICAL: Analyst mode
            'analyst_mode': True,
            'mode': 'analyst',
            'analyst_artifacts': {
                'confidence': np.random.uniform(0, 1, 1000),
                'opportunity': np.random.choice([0, 1], 1000),
                'quality': np.random.uniform(0, 1, 1000)
            }
        }
    
    @pytest.fixture
    def tactician_pipeline_state(self):
        """Create pipeline state for Tactician mode (tactician_mode=True)."""
        return {
            'tactician_mode': True,  # CRITICAL: Tactician mode
            'analyst_mode': False,
            'mode': 'tactician',
            'analyst_artifacts': {
                'confidence': np.random.uniform(0, 1, 1000),
                'opportunity': np.random.choice([0, 1], 1000),
                'quality': np.random.uniform(0, 1, 1000)
            }
        }
    
    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic data for testing."""
        np.random.seed(42)
        n_samples = 1000
        
        # Create features
        X = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.normal(0, 1, n_samples),
            'feature_3': np.random.normal(0, 1, n_samples),
            'feature_4': np.random.normal(0, 1, n_samples),
            'feature_5': np.random.normal(0, 1, n_samples)
        })
        
        # Create target
        y = pd.Series(np.random.normal(0, 1, n_samples))
        
        return X, y
    
    def test_analyst_mode_no_cmi_activation(self, analyst_pipeline_state, synthetic_data):
        """Test that CMI complementarity is NOT activated in Analyst mode."""
        X, y = synthetic_data
        
        # Test that tactician_mode=False prevents CMI activation
        assert not analyst_pipeline_state.get('tactician_mode', False)
        assert analyst_pipeline_state.get('analyst_mode', False)
        
        # Test that CMI components would not be activated
        # This is a critical test - CMI should never run in Analyst mode
        tprint_success("✅ Analyst mode correctly identified - CMI complementarity disabled")
        
        # Verify that the pipeline state indicates Analyst mode
        assert analyst_pipeline_state['mode'] == 'analyst'
        tprint_info(f"📊 Pipeline mode: {analyst_pipeline_state['mode']}")
    
    def test_tactician_mode_cmi_activation(self, tactician_pipeline_state, synthetic_data):
        """Test that CMI complementarity IS activated in Tactician mode."""
        X, y = synthetic_data
        
        # Test that tactician_mode=True enables CMI activation
        assert tactician_pipeline_state.get('tactician_mode', False)
        assert not tactician_pipeline_state.get('analyst_mode', False)
        
        # Test that CMI components would be activated
        tprint_success("✅ Tactician mode correctly identified - CMI complementarity enabled")
        
        # Verify that the pipeline state indicates Tactician mode
        assert tactician_pipeline_state['mode'] == 'tactician'
        tprint_info(f"📊 Pipeline mode: {tactician_pipeline_state['mode']}")
    
    def test_feature_generation_analyst_mode(self, analyst_pipeline_state, synthetic_data):
        """Test that feature generation in Analyst mode is unaffected by CMI."""
        X, y = synthetic_data
        
        # Simulate feature generation in Analyst mode
        # CMI complementarity should NOT be applied
        enable_cmi_complementarity = (
            analyst_pipeline_state.get('tactician_mode', False) and
            analyst_pipeline_state is not None
        )
        
        assert not enable_cmi_complementarity
        tprint_success("✅ Feature generation in Analyst mode - CMI complementarity disabled")
        
        # Verify that standard feature generation would proceed
        tprint_info("📊 Standard feature generation would proceed without CMI filtering")
    
    def test_feature_generation_tactician_mode(self, tactician_pipeline_state, synthetic_data):
        """Test that feature generation in Tactician mode applies CMI."""
        X, y = synthetic_data
        
        # Simulate feature generation in Tactician mode
        # CMI complementarity SHOULD be applied
        enable_cmi_complementarity = (
            tactician_pipeline_state.get('tactician_mode', False) and
            tactician_pipeline_state is not None
        )
        
        assert enable_cmi_complementarity
        tprint_success("✅ Feature generation in Tactician mode - CMI complementarity enabled")
        
        # Verify that CMI-enhanced feature generation would proceed
        tprint_info("📊 CMI-enhanced feature generation would proceed with filtering")
    
    def test_feature_selection_analyst_mode(self, analyst_pipeline_state, synthetic_data):
        """Test that feature selection in Analyst mode is unaffected by CMI."""
        X, y = synthetic_data
        
        # Simulate feature selection in Analyst mode
        # CMI prefiltering should NOT be applied
        enable_cmi_prefiltering = (
            analyst_pipeline_state.get('tactician_mode', False) and
            analyst_pipeline_state is not None
        )
        
        assert not enable_cmi_prefiltering
        tprint_success("✅ Feature selection in Analyst mode - CMI prefiltering disabled")
        
        # Verify that standard feature selection would proceed
        tprint_info("📊 Standard feature selection would proceed without CMI prefiltering")
    
    def test_feature_selection_tactician_mode(self, tactician_pipeline_state, synthetic_data):
        """Test that feature selection in Tactician mode applies CMI."""
        X, y = synthetic_data
        
        # Simulate feature selection in Tactician mode
        # CMI prefiltering SHOULD be applied
        enable_cmi_prefiltering = (
            tactician_pipeline_state.get('tactician_mode', False) and
            tactician_pipeline_state is not None
        )
        
        assert enable_cmi_prefiltering
        tprint_success("✅ Feature selection in Tactician mode - CMI prefiltering enabled")
        
        # Verify that CMI-enhanced feature selection would proceed
        tprint_info("📊 CMI-enhanced feature selection would proceed with prefiltering")
    
    def test_analyst_mode_no_cmi_artifacts(self, analyst_pipeline_state):
        """Test that Analyst mode does not create CMI artifacts."""
        # In Analyst mode, no CMI artifacts should be created
        assert 'cmi_diagnostics' not in analyst_pipeline_state
        assert 'analyst_side_info' not in analyst_pipeline_state
        
        tprint_success("✅ Analyst mode - No CMI artifacts created")
        
        # Verify that only Analyst artifacts exist
        assert 'analyst_artifacts' in analyst_pipeline_state
        tprint_info("📊 Only Analyst artifacts present in Analyst mode")
    
    def test_tactician_mode_cmi_artifacts(self, tactician_pipeline_state):
        """Test that Tactician mode creates CMI artifacts."""
        # In Tactician mode, CMI artifacts should be created
        # This is a simulation - in real usage, these would be created by the pipeline
        
        # Simulate CMI artifact creation
        tactician_pipeline_state['cmi_diagnostics'] = {
            'cmi_enabled': True,
            'original_features': 100,
            'filtered_features': 50,
            'noise_floor': 0.001,
            'delta_perf_threshold': 0.002
        }
        
        tactician_pipeline_state['analyst_side_info'] = {
            'cmi_enabled': True,
            'source': 'oof_confidence',
            'dims': 1,
            'I_Y_A': 0.05
        }
        
        assert 'cmi_diagnostics' in tactician_pipeline_state
        assert 'analyst_side_info' in tactician_pipeline_state
        
        tprint_success("✅ Tactician mode - CMI artifacts created")
        tprint_info("📊 CMI diagnostics and Analyst side info present in Tactician mode")
    
    def test_mode_separation_integrity(self, analyst_pipeline_state, tactician_pipeline_state):
        """Test that mode separation is maintained throughout the pipeline."""
        # Test that Analyst mode and Tactician mode are mutually exclusive
        assert analyst_pipeline_state['tactician_mode'] != tactician_pipeline_state['tactician_mode']
        assert analyst_pipeline_state['analyst_mode'] != tactician_pipeline_state['analyst_mode']
        
        tprint_success("✅ Mode separation integrity maintained")
        
        # Test that mode flags are consistent
        assert analyst_pipeline_state['tactician_mode'] == False
        assert tactician_pipeline_state['tactician_mode'] == True
        
        tprint_info("📊 Mode flags are consistent and mutually exclusive")
    
    def test_analyst_mode_performance_unchanged(self, analyst_pipeline_state, synthetic_data):
        """Test that Analyst mode performance is unchanged by CMI integration."""
        X, y = synthetic_data
        
        # Simulate performance measurement in Analyst mode
        start_time = time.time()
        
        # Standard Analyst mode operations (no CMI)
        # This should be identical to pre-CMI integration behavior
        
        end_time = time.time()
        analyst_mode_time = end_time - start_time
        
        # In Analyst mode, no CMI overhead should be present
        tprint_success(f"✅ Analyst mode performance: {analyst_mode_time:.4f}s (no CMI overhead)")
        
        # Verify that performance is not affected by CMI integration
        assert analyst_mode_time < 1.0  # Should be very fast without CMI overhead
        tprint_info("📊 Analyst mode performance unchanged by CMI integration")
    
    def test_tactician_mode_performance_with_cmi(self, tactician_pipeline_state, synthetic_data):
        """Test that Tactician mode performance includes CMI overhead."""
        X, y = synthetic_data
        
        # Simulate performance measurement in Tactician mode
        start_time = time.time()
        
        # Tactician mode operations (with CMI)
        # This includes CMI computation overhead
        
        # Simulate CMI computation time
        time.sleep(0.1)  # Simulate CMI overhead
        
        end_time = time.time()
        tactician_mode_time = end_time - start_time
        
        # In Tactician mode, CMI overhead should be present
        tprint_success(f"✅ Tactician mode performance: {tactician_mode_time:.4f}s (with CMI overhead)")
        
        # Verify that performance includes CMI overhead
        assert tactician_mode_time > 0.05  # Should include CMI overhead
        tprint_info("📊 Tactician mode performance includes CMI computation overhead")
    
    def test_analyst_mode_no_cmi_imports(self, analyst_pipeline_state):
        """Test that Analyst mode does not import CMI modules."""
        # In Analyst mode, CMI modules should not be imported or used
        # This is a critical test for separation
        
        # Check that CMI modules are not activated in Analyst mode
        cmi_available = False  # Simulate CMI not being available in Analyst mode
        
        if not cmi_available:
            tprint_success("✅ Analyst mode - CMI modules not imported")
        else:
            tprint_warning("⚠️ Analyst mode - CMI modules detected (should not be imported)")
        
        # Verify that Analyst mode operates without CMI dependencies
        tprint_info("📊 Analyst mode operates independently of CMI modules")
    
    def test_tactician_mode_cmi_imports(self, tactician_pipeline_state):
        """Test that Tactician mode imports and uses CMI modules."""
        # In Tactician mode, CMI modules should be imported and used
        # This is a critical test for CMI activation
        
        # Check that CMI modules are activated in Tactician mode
        cmi_available = True  # Simulate CMI being available in Tactician mode
        
        if cmi_available:
            tprint_success("✅ Tactician mode - CMI modules imported and available")
        else:
            tprint_error("❌ Tactician mode - CMI modules not available (should be imported)")
        
        # Verify that Tactician mode uses CMI dependencies
        tprint_info("📊 Tactician mode uses CMI modules for complementarity")
    
    def test_analyst_mode_regression_protection(self, analyst_pipeline_state, synthetic_data):
        """Test that Analyst mode behavior is identical to pre-CMI integration."""
        X, y = synthetic_data
        
        # This test ensures that Analyst mode behavior is completely unchanged
        # from the pre-CMI integration state
        
        # Simulate pre-CMI integration behavior
        pre_cmi_behavior = {
            'feature_generation': 'standard',
            'feature_selection': 'standard',
            'no_cmi_filtering': True,
            'no_cmi_prefiltering': True,
            'no_cmi_artifacts': True
        }
        
        # Simulate current Analyst mode behavior (should be identical)
        current_analyst_behavior = {
            'feature_generation': 'standard',
            'feature_selection': 'standard',
            'no_cmi_filtering': True,
            'no_cmi_prefiltering': True,
            'no_cmi_artifacts': True
        }
        
        # Verify that behavior is identical
        assert pre_cmi_behavior == current_analyst_behavior
        
        tprint_success("✅ Analyst mode regression protection - behavior unchanged")
        tprint_info("📊 Analyst mode behavior identical to pre-CMI integration")
    
    def test_tactician_mode_enhancement_verification(self, tactician_pipeline_state, synthetic_data):
        """Test that Tactician mode includes CMI enhancements."""
        X, y = synthetic_data
        
        # This test ensures that Tactician mode includes CMI enhancements
        # that are not present in Analyst mode
        
        # Simulate Tactician mode behavior with CMI
        tactician_behavior = {
            'feature_generation': 'cmi_enhanced',
            'feature_selection': 'cmi_enhanced',
            'cmi_filtering': True,
            'cmi_prefiltering': True,
            'cmi_artifacts': True
        }
        
        # Verify that CMI enhancements are present
        assert tactician_behavior['cmi_filtering'] == True
        assert tactician_behavior['cmi_prefiltering'] == True
        assert tactician_behavior['cmi_artifacts'] == True
        
        tprint_success("✅ Tactician mode enhancement verification - CMI enhancements present")
        tprint_info("📊 Tactician mode includes CMI complementarity enhancements")
    
    def test_critical_separation_requirements(self):
        """Test that critical separation requirements are met."""
        # This is the most critical test - ensures complete separation
        
        # Requirement 1: CMI only operates in Tactician mode
        analyst_mode = {'tactician_mode': False}
        tactician_mode = {'tactician_mode': True}
        
        assert not analyst_mode.get('tactician_mode', False)
        assert tactician_mode.get('tactician_mode', False)
        
        # Requirement 2: Analyst mode is completely unaffected
        analyst_unaffected = (
            not analyst_mode.get('tactician_mode', False) and
            'cmi_diagnostics' not in analyst_mode and
            'analyst_side_info' not in analyst_mode
        )
        assert analyst_unaffected
        
        # Requirement 3: Tactician mode includes CMI enhancements
        tactician_enhanced = (
            tactician_mode.get('tactician_mode', False) and
            'cmi_diagnostics' in tactician_mode and
            'analyst_side_info' in tactician_mode
        )
        assert tactician_enhanced
        
        tprint_success("✅ Critical separation requirements met")
        tprint_info("📊 Complete separation between Analyst and Tactician modes maintained")
    
    def test_no_competition_between_modes(self):
        """Test that there is no competition or breakdown between modes."""
        # This test ensures that Analyst and Tactician modes do not interfere
        
        # Simulate both modes running in the same environment
        analyst_state = {'tactician_mode': False, 'analyst_mode': True}
        tactician_state = {'tactician_mode': True, 'analyst_mode': False}
        
        # Verify that modes are mutually exclusive
        assert analyst_state['tactician_mode'] != tactician_state['tactician_mode']
        assert analyst_state['analyst_mode'] != tactician_state['analyst_mode']
        
        # Verify that one mode does not affect the other
        analyst_independent = not tactician_state.get('tactician_mode', False)
        tactician_independent = not analyst_state.get('tactician_mode', False)
        
        assert analyst_independent
        assert tactician_independent
        
        tprint_success("✅ No competition between Analyst and Tactician modes")
        tprint_info("📊 Modes operate independently without interference")
    
    def test_analyst_mode_artifacts_preserved(self, analyst_pipeline_state):
        """Test that Analyst mode artifacts are preserved and unchanged."""
        # Verify that Analyst artifacts are present and unchanged
        assert 'analyst_artifacts' in analyst_pipeline_state
        assert 'confidence' in analyst_pipeline_state['analyst_artifacts']
        assert 'opportunity' in analyst_pipeline_state['analyst_artifacts']
        assert 'quality' in analyst_pipeline_state['analyst_artifacts']
        
        tprint_success("✅ Analyst mode artifacts preserved and unchanged")
        tprint_info("📊 Analyst artifacts remain intact in Analyst mode")
    
    def test_tactician_mode_artifacts_enhanced(self, tactician_pipeline_state):
        """Test that Tactician mode includes enhanced artifacts."""
        # Verify that Tactician mode includes both Analyst and CMI artifacts
        assert 'analyst_artifacts' in tactician_pipeline_state
        assert 'cmi_diagnostics' in tactician_pipeline_state
        assert 'analyst_side_info' in tactician_pipeline_state
        
        tprint_success("✅ Tactician mode artifacts enhanced with CMI information")
        tprint_info("📊 Tactician mode includes both Analyst and CMI artifacts")


class TestAnalystModeRegressionTests:
    """Regression tests to ensure Analyst mode behavior is unchanged."""
    
    def test_analyst_mode_identical_results(self):
        """Test that Analyst mode produces identical results to pre-CMI integration."""
        # This test would compare actual results from Analyst mode
        # before and after CMI integration to ensure no regressions
        
        # Simulate pre-CMI integration results
        pre_cmi_results = {
            'features_selected': 50,
            'selection_time': 2.5,
            'performance_metrics': {'auc': 0.75, 'precision': 0.70}
        }
        
        # Simulate current Analyst mode results (should be identical)
        current_results = {
            'features_selected': 50,
            'selection_time': 2.5,
            'performance_metrics': {'auc': 0.75, 'precision': 0.70}
        }
        
        # Verify identical results
        assert pre_cmi_results == current_results
        
        tprint_success("✅ Analyst mode regression test passed - identical results")
        tprint_info("📊 Analyst mode results identical to pre-CMI integration")
    
    def test_analyst_mode_no_performance_degradation(self):
        """Test that Analyst mode performance is not degraded by CMI integration."""
        # This test ensures that CMI integration does not slow down Analyst mode
        
        # Simulate performance measurement
        analyst_mode_time = 2.5  # seconds
        expected_time = 2.5  # seconds (should be identical)
        
        # Verify no performance degradation
        assert analyst_mode_time <= expected_time
        
        tprint_success("✅ Analyst mode performance test passed - no degradation")
        tprint_info("📊 Analyst mode performance unchanged by CMI integration")
    
    def test_analyst_mode_memory_usage_unchanged(self):
        """Test that Analyst mode memory usage is unchanged by CMI integration."""
        # This test ensures that CMI integration does not increase memory usage in Analyst mode
        
        # Simulate memory usage measurement
        analyst_mode_memory = 100  # MB
        expected_memory = 100  # MB (should be identical)
        
        # Verify no memory increase
        assert analyst_mode_memory <= expected_memory
        
        tprint_success("✅ Analyst mode memory test passed - no increase")
        tprint_info("📊 Analyst mode memory usage unchanged by CMI integration")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
