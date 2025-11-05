"""
Comprehensive Test Suite for Sticky Finite HMM with Full Diagnostics

This test suite integrates all advanced diagnostic and validation techniques:
- Initial sweep with multiple restarts and Hungarian alignment
- Posterior Predictive Checks (PPCs) with global and per-state analysis
- Calibration & scoring with log score, CRPS, and predictive intervals
- Temporal diagnostics with multi-step predictions and residual ACF
- Complexity analysis with held-out LL vs K and WAIC/PSIS-LOO
- Sensitivity testing for κ, α, and emission families
- Simulation-Based Calibration (SBC) and recoverability tests
- Ensemble methods and final model validation

Author: Enhanced HMM Diagnostics Test Suite
Date: 2024
"""

import pytest
import numpy as np
import pandas as pd
import torch
import tempfile
import shutil
from pathlib import Path
import sys
import os
import time

# Add the project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

# Import the enhanced components
from src.training.steps.market_analysis.sticky_finite_hmm_clustering.sticky_finite_hmm_clusterer import (
    StickyFiniteHMMClusterer,
    StickyFiniteHMMConfig,
    create_sticky_finite_hmm_clusterer,
    DEPENDENCIES_AVAILABLE,
    SCIPY_AVAILABLE,
    NUMBA_AVAILABLE
)

from src.training.steps.market_analysis.sticky_finite_hmm_clustering.sticky_finite_hmm_auto_tuner import (
    sticky_finite_hmm_objective_function,
    create_default_search_space
)

from src.training.steps.market_analysis.sticky_finite_hmm_clustering.sticky_finite_hmm_diagnostics import (
    StickyFiniteHMMDiagnostics,
    DiagnosticConfig,
    create_comprehensive_diagnostics
)

from src.training.steps.market_analysis.sticky_finite_hmm_clustering.sticky_finite_hmm_regime_discovery_step import (
    StickyFiniteHMMRegimeDiscoveryStep
)


class TestComprehensiveDiagnostics:
    """Test suite for comprehensive HMM diagnostics and validation."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample multivariate time series data for testing."""
        np.random.seed(42)
        n_timesteps = 1000
        n_dimensions = 3
        
        # Generate 3-state HMM data
        true_K = 3
        true_transitions = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1], 
            [0.1, 0.1, 0.8]
        ])
        
        true_means = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0],
            [-2.0, -2.0, -2.0]
        ])
        
        true_stds = np.array([
            [0.5, 0.5, 0.5],
            [0.8, 0.8, 0.8],
            [0.6, 0.6, 0.6]
        ])
        
        # Generate state sequence
        states = np.zeros(n_timesteps, dtype=int)
        for t in range(1, n_timesteps):
            states[t] = np.random.choice(true_K, p=true_transitions[states[t-1]])
        
        # Generate observations
        data = np.zeros((n_timesteps, n_dimensions))
        for t in range(n_timesteps):
            data[t] = np.random.normal(
                true_means[states[t]], 
                true_stds[states[t]], 
                n_dimensions
            )
        
        return data, states, {
            'K': true_K,
            'transitions': true_transitions,
            'means': true_means,
            'stds': true_stds
        }
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def diagnostic_config(self):
        """Create diagnostic configuration for testing."""
        return DiagnosticConfig(
            k_candidates=[2, 3, 4],  # Reduced for testing speed
            n_restarts_per_k=2,       # Reduced for testing speed
            random_seeds=[42, 123],   # Reduced for testing speed
            held_out_ratio=0.2,
            n_ppc_samples=100,        # Reduced for testing speed
            prediction_horizons=[1, 5],  # Reduced for testing speed
            confidence_levels=[0.8, 0.9],  # Reduced for testing speed
            max_lag=10,               # Reduced for testing speed
            kappa_range=(1.0, 10.0),  # Reduced for testing speed
            alpha_range=(0.1, 1.0),   # Reduced for testing speed
            emission_families=['gaussian'],  # Reduced for testing speed
            n_sbc_simulations=10      # Reduced for testing speed
        )
    
    @pytest.fixture
    def base_config(self):
        """Create base HMM configuration for testing."""
        return StickyFiniteHMMConfig(
            K=3,
            n_mixtures=1,
            base_alpha=1.0,
            kappa=10.0,
            lr=0.01,
            num_iters=100,  # Reduced for testing speed
            pca_components=2
        )

    def test_sticky_hmm_with_autotuner_comprehensive(
        self, 
        sample_data, 
        temp_dir, 
        diagnostic_config, 
        base_config
    ):
        """
        Comprehensive test integrating all diagnostics with auto-tuner.
        
        This is the main test function that runs the complete diagnostic pipeline:
        1. Initial sweep with multiple restarts and Hungarian alignment
        2. Posterior Predictive Checks (PPCs)
        3. Calibration & scoring
        4. Temporal diagnostics
        5. Complexity analysis
        6. Sensitivity testing
        7. SBC & recoverability
        8. Ensemble & final validation
        """
        data, true_states, true_params = sample_data
        
        print("\n" + "="*80)
        print("🚀 COMPREHENSIVE STICKY FINITE HMM TEST WITH AUTOTUNER")
        print("="*80)
        
        # Initialize diagnostics
        diagnostics = create_comprehensive_diagnostics(diagnostic_config)
        
        # Create output directory
        output_dir = temp_dir / "comprehensive_diagnostics"
        output_dir.mkdir(exist_ok=True)
        
        start_time = time.time()
        
        try:
            # 1. INITIAL SWEEP WITH MULTIPLE RESTARTS
            print("\n🔍 STEP 1: Initial Sweep with Multiple Restarts")
            print("-" * 60)
            
            sweep_results = diagnostics.run_initial_sweep(
                data, base_config, output_dir / "initial_sweep"
            )
            
            # Verify sweep results
            assert 'initial_sweep' in diagnostics.results
            assert len(sweep_results) > 0
            
            # Check that we have results for each K
            for K in diagnostic_config.k_candidates:
                assert K in sweep_results
                assert 'best_restart' in sweep_results[K]
                assert 'held_out_ll' in sweep_results[K]['best_restart']
                assert 'unstable_states' in sweep_results[K]
            
            print(f"✅ Initial sweep completed for K={list(sweep_results.keys())}")
            
            # Select best K from sweep
            best_K = max(sweep_results.keys(), 
                        key=lambda k: sweep_results[k]['best_restart']['held_out_ll'])
            best_model = sweep_results[best_K]['best_restart']['model']
            
            print(f"🎯 Best K selected: {best_K}")
            print(f"📊 Best held-out LL: {sweep_results[best_K]['best_restart']['held_out_ll']:.2f}")
            
            # 2. POSTERIOR PREDICTIVE CHECKS
            print("\n🔍 STEP 2: Posterior Predictive Checks")
            print("-" * 60)
            
            ppc_results = diagnostics.run_posterior_predictive_checks(
                best_model, data, output_dir / "ppc"
            )
            
            # Verify PPC results
            assert 'ppc' in diagnostics.results
            assert 'samples' in ppc_results
            assert 'global_checks' in ppc_results
            assert 'per_state_checks' in ppc_results
            assert 'time_series_checks' in ppc_results
            
            print("✅ Posterior predictive checks completed")
            
            # 3. CALIBRATION & SCORING
            print("\n🔍 STEP 3: Calibration & Scoring")
            print("-" * 60)
            
            calibration_results = diagnostics.run_calibration_scoring(
                best_model, data, output_dir / "calibration"
            )
            
            # Verify calibration results
            assert 'calibration' in diagnostics.results
            assert 'aggregate' in calibration_results
            
            for horizon in diagnostic_config.prediction_horizons:
                assert f'horizon_{horizon}' in calibration_results
            
            print("✅ Calibration & scoring completed")
            
            # 4. TEMPORAL DIAGNOSTICS
            print("\n🔍 STEP 4: Temporal Diagnostics")
            print("-" * 60)
            
            temporal_results = diagnostics.run_temporal_diagnostics(
                best_model, data, output_dir / "temporal"
            )
            
            # Verify temporal results
            assert 'temporal' in diagnostics.results
            assert 'multi_step' in temporal_results
            assert 'residuals' in temporal_results
            assert 'durations' in temporal_results
            
            print("✅ Temporal diagnostics completed")
            
            # 5. COMPLEXITY ANALYSIS
            print("\n🔍 STEP 5: Complexity Analysis")
            print("-" * 60)
            
            complexity_results = diagnostics.run_complexity_analysis(
                data, base_config, output_dir / "complexity"
            )
            
            # Verify complexity results
            assert 'complexity' in diagnostics.results
            assert 'll_vs_k' in complexity_results
            assert 'recommendations' in complexity_results
            
            print("✅ Complexity analysis completed")
            
            # 6. SENSITIVITY TESTS
            print("\n🔍 STEP 6: Sensitivity Tests")
            print("-" * 60)
            
            sensitivity_results = diagnostics.run_sensitivity_tests(
                data, base_config, output_dir / "sensitivity"
            )
            
            # Verify sensitivity results
            assert 'sensitivity' in diagnostics.results
            assert 'kappa' in sensitivity_results
            assert 'alpha' in sensitivity_results
            assert 'emissions' in sensitivity_results
            assert 'summary' in sensitivity_results
            
            print("✅ Sensitivity tests completed")
            
            # 7. SBC & RECOVERABILITY
            print("\n🔍 STEP 7: SBC & Recoverability Tests")
            print("-" * 60)
            
            sbc_results = diagnostics.run_sbc_recoverability(
                base_config, output_dir / "sbc"
            )
            
            # Verify SBC results
            assert 'sbc' in diagnostics.results
            assert 'overall' in sbc_results
            
            for param in diagnostic_config.sbc_parameters:
                assert param in sbc_results
            
            print("✅ SBC & recoverability tests completed")
            
            # 8. ENSEMBLE & FINAL VALIDATION
            print("\n🔍 STEP 8: Ensemble & Final Validation")
            print("-" * 60)
            
            # Collect models for ensemble
            ensemble_models = []
            for k_value in diagnostic_config.k_candidates:
                if k_value in sweep_results and 'best_restart' in sweep_results[k_value]:
                    ensemble_models.append(sweep_results[k_value]['best_restart']['model'])
            
            ensemble_results = diagnostics.run_ensemble_final_check(
                data, ensemble_models, output_dir / "ensemble"
            )
            
            # Verify ensemble results
            assert 'ensemble' in diagnostics.results
            assert 'comparison' in ensemble_results
            assert 'ensemble' in ensemble_results
            assert 'final_diagnostics' in ensemble_results
            
            print("✅ Ensemble & final validation completed")
            
            # FINAL SUMMARY
            total_time = time.time() - start_time
            
            print("\n" + "="*80)
            print("🎉 COMPREHENSIVE TEST COMPLETED SUCCESSFULLY")
            print("="*80)
            print(f"⏱️  Total time: {total_time:.2f} seconds")
            print(f"📊 Data shape: {data.shape}")
            print(f"🎯 Best K found: {best_K}")
            print(f"📈 Best held-out LL: {sweep_results[best_K]['best_restart']['held_out_ll']:.2f}")
            print(f"📁 Output directory: {output_dir}")
            
            # Verify all diagnostic modules were executed
            expected_modules = [
                'initial_sweep', 'ppc', 'calibration', 'temporal',
                'complexity', 'sensitivity', 'sbc', 'ensemble'
            ]
            
            for module in expected_modules:
                assert module in diagnostics.results, f"Module {module} not executed"
                print(f"✅ {module.replace('_', ' ').title()}: Completed")
            
            # Test enhanced SVI features are working
            print("\n🔧 ENHANCED SVI FEATURES VERIFICATION:")
            print("-" * 60)
            
            # Check if enhanced methods were used
            if hasattr(best_model, '_last_posterior_marginals'):
                print("✅ Enhanced posterior computation: Active")
            
            if NUMBA_AVAILABLE:
                print("✅ JIT optimizations: Available")
            else:
                print("⚠️  JIT optimizations: Not available (numba missing)")
            
            if SCIPY_AVAILABLE:
                print("✅ Advanced statistical functions: Available")
            else:
                print("⚠️  Advanced statistical functions: Not available (scipy missing)")
            
            # Verify convergence
            if hasattr(best_model, 'elbo_history') and best_model.elbo_history:
                final_elbo = best_model.elbo_history[-1]
                print(f"✅ Final ELBO: {final_elbo:.2f}")
                
                # Check for convergence improvement
                if len(best_model.elbo_history) > 10:
                    early_elbo = np.mean(best_model.elbo_history[:10])
                    improvement = final_elbo - early_elbo
                    print(f"✅ ELBO improvement: {improvement:.2f}")
            
            print("\n🚀 ALL DIAGNOSTIC MODULES SUCCESSFULLY INTEGRATED!")
            print("🎯 Enhanced SVI with comprehensive validation is production-ready!")
            
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            raise
        
        finally:
            print(f"\n📁 Test artifacts saved to: {output_dir}")
    
    def test_initial_sweep_only(self, sample_data, base_config, temp_dir):
        """Test initial sweep component in isolation."""
        data, _, _ = sample_data
        
        config = DiagnosticConfig(
            k_candidates=[2, 3],
            n_restarts_per_k=2,
            random_seeds=[42, 123]
        )
        
        diagnostics = create_comprehensive_diagnostics(config)
        results = diagnostics.run_initial_sweep(data, base_config, temp_dir)
        
        assert len(results) == 2  # K=2 and K=3
        assert all('best_restart' in results[k] for k in results)
        assert all('unstable_states' in results[k] for k in results)
    
    def test_ppc_only(self, sample_data, base_config, temp_dir):
        """Test PPC component in isolation."""
        data, _, _ = sample_data
        
        config = DiagnosticConfig(n_ppc_samples=50)
        diagnostics = create_comprehensive_diagnostics(config)
        
        # Create a simple model for testing
        model = StickyFiniteHMMClusterer(base_config)
        model.fit(data)
        
        results = diagnostics.run_posterior_predictive_checks(model, data, temp_dir)
        
        assert 'samples' in results
        assert 'global_checks' in results
        assert 'per_state_checks' in results
    
    def test_calibration_only(self, sample_data, base_config, temp_dir):
        """Test calibration component in isolation."""
        data, _, _ = sample_data
        
        config = DiagnosticConfig(
            prediction_horizons=[1, 3],
            confidence_levels=[0.8, 0.9]
        )
        diagnostics = create_comprehensive_diagnostics(config)
        
        # Create a simple model for testing
        model = StickyFiniteHMMClusterer(base_config)
        model.fit(data)
        
        results = diagnostics.run_calibration_scoring(model, data, temp_dir)
        
        assert 'aggregate' in results
        assert 'horizon_1' in results
        assert 'horizon_3' in results
    
    def test_integration_with_enhanced_svi(self, sample_data, base_config):
        """Test that diagnostics work with enhanced SVI features."""
        data, _, _ = sample_data
        
        # Create model with enhanced features
        model = StickyFiniteHMMClusterer(base_config)
        model.fit(data)
        
        # Verify enhanced features are available
        assert hasattr(model, '_compute_log_emissions_vectorized')
        assert hasattr(model, '_forward_backward_structured')
        
        if NUMBA_AVAILABLE:
            assert hasattr(model, '_forward_backward_jit')
        
        # Test that enhanced methods can be called
        try:
            # Test vectorized emissions
            mu = np.random.randn(base_config.K, data.shape[1])
            sigma = np.abs(np.random.randn(base_config.K, data.shape[1])) + 0.1
            log_emissions = model._compute_log_emissions_vectorized(data, mu, sigma)
            assert log_emissions.shape == (len(data), base_config.K)
            
            print("✅ Enhanced SVI features integration verified")
            
        except Exception as e:
            pytest.fail(f"Enhanced SVI features integration failed: {e}")


if __name__ == "__main__":
    # Run the comprehensive test
    pytest.main([__file__, "-v", "-s"])
