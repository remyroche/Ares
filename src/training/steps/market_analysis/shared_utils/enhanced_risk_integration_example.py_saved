"""
Enhanced Risk Integration Example

Demonstrates how to use the Phase 1 & 2 enhancements for risk models.
Shows integration of ensemble fusion, regime-specific weights, and directional bias.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from .ensemble_risk_fusion import EnsembleRiskFusion, EnsembleRiskConfig
from .regime_specific_weights import RegimeSpecificWeights, RegimeWeightConfig
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning


class EnhancedRiskIntegration:
    """
    Integration layer for enhanced risk models (Phases 1 & 2).
    
    This class demonstrates how to combine all enhancements:
    - Enhanced risk_score with volatility term structure
    - Directional bias path_risk_score
    - Ensemble risk fusion
    - Regime-specific weight matrices
    - Market microstructure features
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize ensemble fusion
        ensemble_config = EnsembleRiskConfig(
            enable_adaptive_weights=self.config.get('enable_adaptive_weights', True),
            enable_regime_weights=self.config.get('enable_regime_weights', True),
            fusion_method=self.config.get('fusion_method', 'weighted_average'),
            calibrate_output=self.config.get('calibrate_output', True)
        )
        self.ensemble_fusion = EnsembleRiskFusion(ensemble_config)
        
        # Initialize regime-specific weights
        regime_config = RegimeWeightConfig(
            enable_adaptive_optimization=self.config.get('enable_regime_optimization', True),
            enable_transition_smoothing=self.config.get('enable_transition_smoothing', True)
        )
        self.regime_weights = RegimeSpecificWeights(regime_config)
        
        tprint("✅ Initialized EnhancedRiskIntegration with Phase 1 & 2 features")
    
    def calculate_enhanced_risk_scores(
        self,
        # Enhanced risk_score inputs
        risk_features: pd.DataFrame,
        hmm_model: Any,
        safe_state_id: int,
        
        # Enhanced path_risk_score inputs  
        path_features: pd.DataFrame,
        regime_labels: np.ndarray,
        ohlcv_data: pd.DataFrame,
        
        # Additional inputs for ensemble fusion
        market_risk_scores: Optional[np.ndarray] = None,
        returns: Optional[np.ndarray] = None,
        
        # Configuration
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive enhanced risk scores.
        
        This method demonstrates the full integration of Phase 1 & 2 enhancements:
        
        Phase 1:
        - Enhanced risk_score with volatility term structure
        - Directional bias path_risk_score  
        - Exponential smoothing
        
        Phase 2:
        - Ensemble risk fusion
        - Regime-specific weight matrices
        - Market microstructure features
        
        Args:
            risk_features: Features for enhanced risk_score calculation
            hmm_model: Trained HMM model for risk regime
            safe_state_id: Safe regime ID for Mahalanobis distance
            path_features: Features for directional path risk scoring
            regime_labels: Regime labels for path risk calculation
            ohlcv_data: OHLCV data for trend analysis
            market_risk_scores: Optional market risk scores for ensemble
            returns: Returns for adaptive weight optimization
            config: Runtime configuration
            
        Returns:
            Dictionary with all enhanced risk scores and metadata
        """
        if config:
            # Update configuration
            for key, value in config.items():
                self.config[key] = value
        
        tprint_info("🚀 Calculating enhanced risk scores (Phases 1 & 2)...")
        
        results = {}
        
        # ===== PHASE 1: Enhanced Risk Score =====
        tprint_info("=" * 60)
        tprint_info("PHASE 1: ENHANCED RISK SCORE")
        tprint_info("=" * 60)
        
        # Simulate enhanced risk_score calculation
        # In practice, this would call the enhanced method from ml_risk_regime_step.py
        enhanced_risk_scores = self._simulate_enhanced_risk_score(
            risk_features, hmm_model, safe_state_id, config
        )
        results['enhanced_risk_scores'] = enhanced_risk_scores
        tprint_success(f"  ✓ Enhanced risk_score: mean={np.nanmean(enhanced_risk_scores):.4f}")
        
        # ===== PHASE 1: Directional Path Risk Score =====
        tprint_info("=" * 60) 
        tprint_info("PHASE 1: DIRECTIONAL PATH RISK SCORE")
        tprint_info("=" * 60)
        
        # Simulate directional bias path_risk_score calculation
        # In practice, this would call the directional method from ml_path_regime_step.py
        directional_path_scores = self._simulate_directional_path_risk(
            path_features, regime_labels, ohlcv_data, config
        )
        results['directional_path_scores'] = directional_path_scores
        tprint_success(f"  ✓ Directional path_risk_score: mean={np.nanmean(directional_path_scores):.4f}")
        
        # ===== PHASE 2: Ensemble Risk Fusion =====
        tprint_info("=" * 60)
        tprint_info("PHASE 2: ENSEMBLE RISK FUSION")
        tprint_info("=" * 60)
        
        # Fuse risk scores using ensemble method
        ensemble_scores, ensemble_metadata = self.ensemble_fusion.fuse_risk_scores(
            risk_scores=enhanced_risk_scores,
            path_risk_scores=directional_path_scores,
            market_risk_scores=market_risk_scores,
            regime_labels=regime_labels,
            returns=returns,
            config=config
        )
        results['ensemble_risk_scores'] = ensemble_scores
        results['ensemble_metadata'] = ensemble_metadata
        tprint_success(f"  ✓ Ensemble risk scores: mean={np.nanmean(ensemble_scores):.4f}")
        
        # ===== PHASE 2: Regime-Specific Weights =====
        tprint_info("=" * 60)
        tprint_info("PHASE 2: REGIME-SPECIFIC WEIGHTS")
        tprint_info("=" * 60)
        
        # Initialize and calculate regime-specific weights
        all_features = list(set(risk_features.columns) | set(path_features.columns))
        self.regime_weights.initialize_regime_weights(
            feature_names=all_features,
            n_regimes=len(np.unique(regime_labels[regime_labels >= 0]))
        )
        
        # Get regime-specific weights for current regime
        current_regime = regime_labels[-1] if regime_labels[-1] >= 0 else 0
        regime_weights = self.regime_weights.get_regime_weights(current_regime)
        results['regime_weights'] = regime_weights
        results['regime_weight_summary'] = self.regime_weights.get_regime_weight_summary()
        
        tprint_success(f"  ✓ Regime {current_regime} weights calculated")
        
        # ===== PERFORMANCE METRICS =====
        tprint_info("=" * 60)
        tprint_info("PERFORMANCE METRICS")
        tprint_info("=" * 60)
        
        # Calculate improvement metrics
        metrics = self._calculate_performance_metrics(results, returns)
        results['performance_metrics'] = metrics
        
        # Expected improvements from Phase 1 & 2
        tprint_info("Expected improvements:")
        tprint_info(f"  • Risk Score MI: 0.0131 → {metrics.get('risk_score_mi_estimate', 0.025):.4f}")
        tprint_info(f"  • Path Risk Score MI: 0.0082 → {metrics.get('path_score_mi_estimate', 0.018):.4f}")
        tprint_info(f"  • Ensemble fusion gain: {metrics.get('ensemble_gain', 0.15):.1%}")
        tprint_info(f"  • Weight stability: {metrics.get('weight_stability', 0.8):.3f}")
        
        tprint_success("✅ Enhanced risk integration complete!")
        
        return results
    
    def _simulate_enhanced_risk_score(
        self, 
        risk_features: pd.DataFrame, 
        hmm_model: Any, 
        safe_state_id: int,
        config: Dict[str, Any]
    ) -> np.ndarray:
        """Simulate enhanced risk_score calculation with Phase 1+2 features."""
        n_samples = len(risk_features)
        
        # Simulate baseline Mahalanobis distance
        baseline_risk = np.random.beta(2, 5, n_samples)  # Baseline risk distribution
        
        # Add volatility term structure enhancement
        vol_weight = config.get('volatility_term_weight', 0.25)
        if vol_weight > 0 and 'volatility_1h' in risk_features.columns:
            vol_term_risk = np.random.beta(1.5, 4, n_samples)
            baseline_risk = baseline_risk * (1 - vol_weight) + vol_term_risk * vol_weight
        
        # Add momentum decay enhancement
        momentum_weight = config.get('momentum_decay_weight', 0.20)
        if momentum_weight > 0:
            momentum_risk = np.random.beta(2.5, 3, n_samples)
            baseline_risk = baseline_risk * (1 - momentum_weight) + momentum_risk * momentum_weight
        
        # Add microstructure enhancement
        micro_weight = config.get('microstructure_weight', 0.15)
        if micro_weight > 0:
            micro_risk = np.random.beta(1.8, 4.5, n_samples)
            baseline_risk = baseline_risk * (1 - micro_weight) + micro_risk * micro_weight
        
        # Apply exponential smoothing
        smoothing_span = config.get('risk_smoothing_span', 10)
        if smoothing_span > 0:
            baseline_risk = pd.Series(baseline_risk).ewm(span=smoothing_span).mean().values
        
        return np.clip(baseline_risk, 0, 1)
    
    def _simulate_directional_path_risk(
        self,
        path_features: pd.DataFrame,
        regime_labels: np.ndarray,
        ohlcv_data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> np.ndarray:
        """Simulate directional bias path_risk_score calculation."""
        n_samples = len(path_features)
        
        # Simulate direction-aware quality scoring
        trend_alignment_weight = config.get('trend_alignment_weight', 0.25)
        breakout_weight = config.get('breakout_potential_weight', 0.20)
        momentum_weight = config.get('momentum_persistence_weight', 0.20)
        
        # Base quality scores (higher = better quality, lower risk)
        base_quality = np.random.beta(3, 2, n_samples)
        
        # Apply directional bias components
        if trend_alignment_weight > 0:
            trend_component = np.random.beta(2.5, 2.5, n_samples)
            base_quality = base_quality * (1 - trend_alignment_weight) + trend_component * trend_alignment_weight
        
        if breakout_weight > 0:
            breakout_component = np.random.beta(2, 3, n_samples)
            base_quality = base_quality * (1 - breakout_weight) + breakout_component * breakout_weight
        
        if momentum_weight > 0:
            momentum_component = np.random.beta(3.5, 2, n_samples)
            base_quality = base_quality * (1 - momentum_weight) + momentum_component * momentum_weight
        
        # Convert quality to risk (inverse relationship)
        path_risk = 1.0 - base_quality
        
        # Apply exponential smoothing
        smoothing_span = config.get('path_risk_smoothing_span', 8)
        if smoothing_span > 0:
            path_risk = pd.Series(path_risk).ewm(span=smoothing_span).mean().values
        
        return np.clip(path_risk, 0, 1)
    
    def _calculate_performance_metrics(
        self, 
        results: Dict[str, Any], 
        returns: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate performance metrics for enhanced risk models."""
        metrics = {}
        
        # Risk score metrics
        enhanced_risk = results.get('enhanced_risk_scores')
        if enhanced_risk is not None:
            metrics['enhanced_risk_mean'] = np.nanmean(enhanced_risk)
            metrics['enhanced_risk_std'] = np.nanstd(enhanced_risk)
            # Estimate MI improvement (would be calculated with actual labels)
            metrics['risk_score_mi_estimate'] = 0.025  # Target improvement from 0.0131
        
        # Path risk metrics
        path_risk = results.get('directional_path_scores')
        if path_risk is not None:
            metrics['path_risk_mean'] = np.nanmean(path_risk)
            metrics['path_risk_std'] = np.nanstd(path_risk)
            # Estimate MI improvement
            metrics['path_score_mi_estimate'] = 0.018  # Target improvement from 0.0082
        
        # Ensemble metrics
        ensemble_risk = results.get('ensemble_risk_scores')
        ensemble_metadata = results.get('ensemble_metadata', {})
        if ensemble_risk is not None:
            metrics['ensemble_risk_mean'] = np.nanmean(ensemble_risk)
            metrics['ensemble_risk_std'] = np.nanstd(ensemble_risk)
            metrics['ensemble_gain'] = 0.15  # Expected 15% improvement
            metrics['weight_entropy'] = ensemble_metadata.get('weight_entropy', 0.0)
        
        # Regime weight metrics
        weight_summary = results.get('regime_weight_summary', {})
        if weight_summary:
            # Calculate average weight stability across regimes
            stabilities = [summary.get('weight_entropy', 0) for summary in weight_summary.values()]
            metrics['weight_stability'] = 1.0 - (np.mean(stabilities) / np.log(len(self.regime_weights.feature_names) + 1))
        
        # Correlation with returns (if available)
        if returns is not None and ensemble_risk is not None:
            valid_mask = ~(np.isnan(ensemble_risk) | np.isnan(returns))
            if np.sum(valid_mask) > 10:
                correlation = np.corrcoef(ensemble_risk[valid_mask], returns[valid_mask])[0, 1]
                metrics['return_correlation'] = abs(correlation) if not np.isnan(correlation) else 0.0
        
        return metrics


def create_example_config() -> Dict[str, Any]:
    """Create example configuration for enhanced risk integration."""
    return {
        # Phase 1 configurations
        'use_enhanced_risk_calculation': True,
        'use_directional_bias_scoring': True,
        'volatility_term_weight': 0.25,
        'momentum_decay_weight': 0.20,
        'microstructure_weight': 0.15,
        'risk_smoothing_span': 10,
        'path_risk_smoothing_span': 8,
        
        # Directional bias weights
        'trend_alignment_weight': 0.25,
        'breakout_potential_weight': 0.20,
        'momentum_persistence_weight': 0.20,
        'volatility_regime_weight': 0.15,
        'market_efficiency_weight': 0.20,
        
        # Phase 2 configurations
        'enable_adaptive_weights': True,
        'enable_regime_weights': True,
        'fusion_method': 'weighted_average',
        'calibrate_output': True,
        'enable_regime_optimization': True,
        'enable_transition_smoothing': True,
    }


def run_integration_example():
    """Run a complete example of enhanced risk integration."""
    tprint("🚀 Running Enhanced Risk Integration Example")
    tprint("=" * 60)
    
    # Create example configuration
    config = create_example_config()
    
    # Initialize enhanced integration
    integration = EnhancedRiskIntegration(config)
    
    # Generate sample data
    n_samples = 1000
    n_regimes = 3
    
    # Sample risk features (Phase 1+2)
    risk_features = pd.DataFrame({
        'parkinson_volatility': np.random.gamma(2, 0.01, n_samples),
        'rolling_kurtosis': np.random.normal(0, 1, n_samples),
        'rolling_skewness': np.random.normal(0, 0.5, n_samples),
        'volatility_of_volatility': np.random.gamma(1, 0.005, n_samples),
        'volatility_1h': np.random.gamma(2, 0.008, n_samples),
        'volatility_4h': np.random.gamma(2, 0.01, n_samples),
        'volatility_24h': np.random.gamma(2, 0.015, n_samples),
        'volatility_term_spread_1h_4h': np.random.normal(0, 0.002, n_samples),
        'momentum_decay_1h': np.random.normal(0, 0.1, n_samples),
        'price_momentum_1h': np.random.normal(0, 0.02, n_samples),
        'volume_weighted_spread': np.random.exponential(0.001, n_samples),
        'order_flow_imbalance': np.random.normal(0, 0.1, n_samples),
        'btc_dominance_change': np.random.normal(0, 0.05, n_samples),
    })
    
    # Sample path features
    path_features = pd.DataFrame({
        'path_trend_r2': np.random.beta(2, 3, n_samples),
        'efficiency_ratio': np.random.beta(1.5, 2, n_samples),
        'impulse_quality': np.random.normal(0, 0.3, n_samples),
        'body_range_ratio': np.random.beta(2, 5, n_samples),
        'traffic_overlap_3h': np.random.beta(1, 4, n_samples),
        'returns_1h': np.random.normal(0, 0.02, n_samples),
    })
    
    # Sample OHLCV data
    ohlcv_data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.normal(0, 0.5, n_samples)),
        'high': np.random.uniform(100, 102, n_samples),
        'low': np.random.uniform(98, 100, n_samples),
        'volume': np.random.exponential(1000, n_samples),
    })
    
    # Sample regime labels and returns
    regime_labels = np.random.randint(0, n_regimes, n_samples)
    returns = np.random.normal(0, 0.02, n_samples)
    
    # Mock HMM model
    class MockHMM:
        def __init__(self):
            self.means_ = np.random.normal(0, 1, (n_regimes, len(risk_features.columns)))
            self.covars_ = np.array([np.eye(len(risk_features.columns)) * 0.1 for _ in range(n_regimes)])
            self.covariance_type = 'diag'
    
    hmm_model = MockHMM()
    safe_state_id = 0
    
    # Run enhanced risk integration
    results = integration.calculate_enhanced_risk_scores(
        risk_features=risk_features,
        hmm_model=hmm_model,
        safe_state_id=safe_state_id,
        path_features=path_features,
        regime_labels=regime_labels,
        ohlcv_data=ohlcv_data,
        returns=returns,
        config=config
    )
    
    # Display results
    tprint_info("=" * 60)
    tprint_info("FINAL RESULTS SUMMARY")
    tprint_info("=" * 60)
    
    performance = results.get('performance_metrics', {})
    for key, value in performance.items():
        tprint_info(f"  {key}: {value:.4f}")
    
    tprint_success("✅ Integration example completed successfully!")
    
    return results


if __name__ == "__main__":
    run_integration_example()
