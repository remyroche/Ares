"""
NAS Regime Analyzer

Implementation for NAS regime analysis.
"""

from rich.console import Console
from rich import print as tprint

tprint("🔍 [NAS_REGIME_ANALYZER] Loading NAS Regime Analyzer module")
tprint("🔍 [NAS_REGIME_ANALYZER] Module path: /workspace/src/training/steps/market_analysis/nas_clustering/core/nas_regime_analyzer.py")
tprint("🔍 [NAS_REGIME_ANALYZER] Purpose: Implementation for NAS regime analysis")
tprint("🔍 [NAS_REGIME_ANALYZER] Status: Starting module import")

import numpy as np
tprint("🔍 [NAS_REGIME_ANALYZER] ✓ NumPy imported successfully")

from typing import Dict, List, Any, Optional, Tuple
tprint("🔍 [NAS_REGIME_ANALYZER] ✓ Typing imports completed")

from dataclasses import dataclass
tprint("🔍 [NAS_REGIME_ANALYZER] ✓ Dataclasses imported successfully")

from enum import Enum
tprint("🔍 [NAS_REGIME_ANALYZER] ✓ Enum imported successfully")

import time
tprint("🔍 [NAS_REGIME_ANALYZER] ✓ Time module imported successfully")

tprint("🔍 [NAS_REGIME_ANALYZER] All imports completed successfully")

class AnalysisType(Enum):
    """Types of regime analysis."""
    tprint("🔍 [ANALYSIS_TYPE] Defining AnalysisType enum")
    PERFORMANCE = "performance"
    tprint("🔍 [ANALYSIS_TYPE] ✓ PERFORMANCE defined")
    STABILITY = "stability"
    tprint("🔍 [ANALYSIS_TYPE] ✓ STABILITY defined")
    TRANSITION = "transition"
    tprint("🔍 [ANALYSIS_TYPE] ✓ TRANSITION defined")
    CORRELATION = "correlation"
    tprint("🔍 [ANALYSIS_TYPE] ✓ CORRELATION defined")
    tprint("🔍 [ANALYSIS_TYPE] All analysis types defined successfully")

@dataclass
class AnalysisConfig:
    """Configuration for regime analysis."""
    tprint("🔍 [ANALYSIS_CONFIG] Defining AnalysisConfig dataclass")
    analysis_types: List[AnalysisType]
    tprint("🔍 [ANALYSIS_CONFIG] ✓ analysis_types field defined")
    window_size: int = 50
    tprint("🔍 [ANALYSIS_CONFIG] ✓ window_size field defined (default: 50)")
    correlation_threshold: float = 0.7
    tprint("🔍 [ANALYSIS_CONFIG] ✓ correlation_threshold field defined (default: 0.7)")
    stability_threshold: float = 0.8
    tprint("🔍 [ANALYSIS_CONFIG] ✓ stability_threshold field defined (default: 0.8)")
    performance_metrics: List[str] = None
    tprint("🔍 [ANALYSIS_CONFIG] ✓ performance_metrics field defined (default: None)")
    tprint("🔍 [ANALYSIS_CONFIG] All configuration fields defined successfully")

class NASRegimeAnalyzer:
    """NAS Regime Analyzer for comprehensive regime analysis."""

    def __init__(self, config: AnalysisConfig):
        """Initialize NAS regime analyzer.

        Args:
            config: Analysis configuration
        """
        tprint("🔍 [NAS_REGIME_ANALYZER_INIT] Initializing NASRegimeAnalyzer")
        tprint(f"🔍 [NAS_REGIME_ANALYZER_INIT] Config received: {config}")
        tprint(f"🔍 [NAS_REGIME_ANALYZER_INIT] Config type: {type(config)}")
        tprint(f"🔍 [NAS_REGIME_ANALYZER_INIT] Analysis types: {config.analysis_types}")
        tprint(f"🔍 [NAS_REGIME_ANALYZER_INIT] Window size: {config.window_size}")
        tprint(f"🔍 [NAS_REGIME_ANALYZER_INIT] Correlation threshold: {config.correlation_threshold}")
        tprint(f"🔍 [NAS_REGIME_ANALYZER_INIT] Stability threshold: {config.stability_threshold}")

        self.config = config
        tprint("🔍 [NAS_REGIME_ANALYZER_INIT] ✓ Config assigned to self.config")

        self.analysis_results = []
        tprint("🔍 [NAS_REGIME_ANALYZER_INIT] ✓ analysis_results initialized as empty list")

        self.regime_patterns = {}
        tprint("🔍 [NAS_REGIME_ANALYZER_INIT] ✓ regime_patterns initialized as empty dict")

        self.analysis_metrics = {}
        tprint("🔍 [NAS_REGIME_ANALYZER_INIT] ✓ analysis_metrics initialized as empty dict")

        tprint("🔍 [NAS_REGIME_ANALYZER_INIT] Initialization complete!")

    def analyze_regimes(self, data: np.ndarray, regimes: List[Dict],
                       architectures: List[Dict]) -> Dict:
        """Analyze regimes and their relationships with architectures.

        Args:
            data: Input data
            regimes: List of regime information
            architectures: List of architecture specifications

        Returns:
            Dictionary containing analysis results
        """
        tprint("🔍 [NAS_REGIME_ANALYZER_ANALYZE] Starting regime analysis")
        tprint(f"🔍 [NAS_REGIME_ANALYZER_ANALYZE] Data shape: {data.shape}")
        tprint(f"🔍 [NAS_REGIME_ANALYZER_ANALYZE] Data type: {type(data)}")
        tprint(f"🔍 [NAS_REGIME_ANALYZER_ANALYZE] Data dtype: {data.dtype}")
        tprint(f"🔍 [NAS_REGIME_ANALYZER_ANALYZE] Data min: {np.min(data):.6f}")
        tprint(f"🔍 [NAS_REGIME_ANALYZER_ANALYZE] Data max: {np.max(data):.6f}")
        tprint(f"🔍 [NAS_REGIME_ANALYZER_ANALYZE] Data mean: {np.mean(data):.6f}")
        tprint(f"🔍 [NAS_REGIME_ANALYZER_ANALYZE] Data std: {np.std(data):.6f}")
        tprint(f"🔍 [NAS_REGIME_ANALYZER_ANALYZE] Number of regimes: {len(regimes)}")
        tprint(f"🔍 [NAS_REGIME_ANALYZER_ANALYZE] Number of architectures: {len(architectures)}")
        tprint(f"🔍 [NAS_REGIME_ANALYZER_ANALYZE] Analysis types: {self.config.analysis_types}")

        start_time = time.time()
        tprint(f"🔍 [NAS_REGIME_ANALYZER_ANALYZE] Start time recorded: {start_time}")

        try:
            tprint("🔍 [NAS_REGIME_ANALYZER_ANALYZE] Starting try block")
            analysis_results = {}
            tprint("🔍 [NAS_REGIME_ANALYZER_ANALYZE] ✓ Analysis results dictionary initialized")

            # Performance analysis
            tprint("🔍 [NAS_REGIME_ANALYZER_ANALYZE] Checking for performance analysis...")
            if AnalysisType.PERFORMANCE in self.config.analysis_types:
                tprint("🔍 [NAS_REGIME_ANALYZER_ANALYZE] ✓ Performance analysis enabled - starting analysis")
                performance_results = self._analyze_performance(data, regimes, architectures)
                analysis_results['performance'] = performance_results
                tprint(f"🔍 [NAS_REGIME_ANALYZER_ANALYZE] ✓ Performance analysis completed: {performance_results}")
            else:
                tprint("🔍 [NAS_REGIME_ANALYZER_ANALYZE] Performance analysis disabled")

            # Stability analysis
            tprint("🔍 [NAS_REGIME_ANALYZER_ANALYZE] Checking for stability analysis...")
            if AnalysisType.STABILITY in self.config.analysis_types:
                tprint("🔍 [NAS_REGIME_ANALYZER_ANALYZE] ✓ Stability analysis enabled - starting analysis")
                stability_results = self._analyze_stability(data, regimes)
                analysis_results['stability'] = stability_results
                tprint(f"🔍 [NAS_REGIME_ANALYZER_ANALYZE] ✓ Stability analysis completed: {stability_results}")
            else:
                tprint("🔍 [NAS_REGIME_ANALYZER_ANALYZE] Stability analysis disabled")

            # Transition analysis
            tprint("🔍 [NAS_REGIME_ANALYZER_ANALYZE] Checking for transition analysis...")
            if AnalysisType.TRANSITION in self.config.analysis_types:
                tprint("🔍 [NAS_REGIME_ANALYZER_ANALYZE] ✓ Transition analysis enabled - starting analysis")
                transition_results = self._analyze_transitions(regimes)
                analysis_results['transitions'] = transition_results
                tprint(f"🔍 [NAS_REGIME_ANALYZER_ANALYZE] ✓ Transition analysis completed: {transition_results}")
            else:
                tprint("🔍 [NAS_REGIME_ANALYZER_ANALYZE] Transition analysis disabled")

            # Correlation analysis
            tprint("🔍 [NAS_REGIME_ANALYZER_ANALYZE] Checking for correlation analysis...")
            if AnalysisType.CORRELATION in self.config.analysis_types:
                tprint("🔍 [NAS_REGIME_ANALYZER_ANALYZE] ✓ Correlation analysis enabled - starting analysis")
                correlation_results = self._analyze_correlations(data, regimes, architectures)
                analysis_results['correlations'] = correlation_results
                tprint(f"🔍 [NAS_REGIME_ANALYZER_ANALYZE] ✓ Correlation analysis completed: {correlation_results}")
            else:
                tprint("🔍 [NAS_REGIME_ANALYZER_ANALYZE] Correlation analysis disabled")

            # Record analysis
            analysis_time = time.time() - start_time
            tprint(f"🔍 [NAS_REGIME_ANALYZER_ANALYZE] Analysis completed in {analysis_time:.4f}s")

            analysis_record = {
                'analysis_results': analysis_results,
                'analysis_time': analysis_time,
                'timestamp': time.time()
            }
            tprint("🔍 [NAS_REGIME_ANALYZER_ANALYZE] Creating analysis record...")
            self.analysis_results.append(analysis_record)
            tprint(f"🔍 [NAS_REGIME_ANALYZER_ANALYZE] ✓ Analysis record added to history (total: {len(self.analysis_results)})")

            result = {
                'analysis_results': analysis_results,
                'analysis_time': analysis_time
            }
            tprint(f"🔍 [NAS_REGIME_ANALYZER_ANALYZE] ✓ Analysis completed successfully")
            tprint(f"🔍 [NAS_REGIME_ANALYZER_ANALYZE] Result: {result}")
            return result

        except Exception as e:
            tprint(f"🔍 [NAS_REGIME_ANALYZER_ANALYZE] ❌ Exception occurred: {e}")
            tprint(f"🔍 [NAS_REGIME_ANALYZER_ANALYZE] Exception type: {type(e)}")
            analysis_time = time.time() - start_time
            tprint(f"🔍 [NAS_REGIME_ANALYZER_ANALYZE] Analysis time before error: {analysis_time:.4f}s")

            error_result = {
                'error': str(e),
                'analysis_time': analysis_time
            }
            tprint(f"🔍 [NAS_REGIME_ANALYZER_ANALYZE] Returning error result: {error_result}")
            return error_result

    def _analyze_performance(self, data: np.ndarray, regimes: List[Dict],
                           architectures: List[Dict]) -> Dict:
        """Analyze performance of architectures across regimes."""
        performance_results = {}

        for architecture in architectures:
            arch_performance = {}

            for regime in regimes:
                regime_type = regime['regime_type']
                regime_data = regime.get('window_data', [])

                if len(regime_data) > 0:
                    # Simulate performance calculation
                    performance = self._calculate_architecture_performance(
                        architecture, regime_data, regime_type
                    )
                    arch_performance[regime_type.value] = performance

            performance_results[str(architecture)] = arch_performance

        return performance_results

    def _calculate_architecture_performance(self, architecture: Dict,
                                          data: np.ndarray, regime_type: Any) -> float:
        """Calculate architecture performance for specific regime."""
        # Simulate performance calculation
        base_performance = np.random.random()

        # Adjust performance based on regime type
        if hasattr(regime_type, 'value'):
            regime_name = regime_type.value
        else:
            regime_name = str(regime_type)

        if 'volatility' in regime_name:
            # High volatility regimes
            performance = base_performance * 1.2
        elif 'trending' in regime_name:
            # Trending regimes
            performance = base_performance * 1.1
        elif 'stable' in regime_name:
            # Stable regimes
            performance = base_performance * 0.9
        else:
            # Default performance
            performance = base_performance

        return performance

    def _analyze_stability(self, data: np.ndarray, regimes: List[Dict]) -> Dict:
        """Analyze stability of regimes."""
        stability_results = {}

        # Calculate regime stability metrics
        regime_durations = []
        regime_volatilities = []

        for regime in regimes:
            regime_durations.append(1)  # Simplified duration
            regime_volatilities.append(regime.get('volatility', 0))

        # Calculate stability metrics
        duration_stability = 1.0 / (1.0 + np.std(regime_durations))
        volatility_stability = 1.0 / (1.0 + np.std(regime_volatilities))

        stability_results = {
            'duration_stability': duration_stability,
            'volatility_stability': volatility_stability,
            'overall_stability': (duration_stability + volatility_stability) / 2,
            'regime_count': len(regimes),
            'avg_duration': np.mean(regime_durations),
            'avg_volatility': np.mean(regime_volatilities)
        }

        return stability_results

    def _analyze_transitions(self, regimes: List[Dict]) -> Dict:
        """Analyze regime transitions."""
        transition_results = {}

        # Count transitions
        transition_counts = {}
        transition_types = []

        for i in range(1, len(regimes)):
            current_regime = regimes[i]['regime_type']
            previous_regime = regimes[i-1]['regime_type']

            if current_regime != previous_regime:
                transition_key = f"{previous_regime.value} -> {current_regime.value}"
                transition_counts[transition_key] = transition_counts.get(transition_key, 0) + 1
                transition_types.append(transition_key)

        # Calculate transition probabilities
        total_transitions = len(transition_types)
        transition_probabilities = {
            transition: count / total_transitions
            for transition, count in transition_counts.items()
        }

        transition_results = {
            'transition_counts': transition_counts,
            'transition_probabilities': transition_probabilities,
            'total_transitions': total_transitions,
            'transition_rate': total_transitions / len(regimes) if regimes else 0
        }

        return transition_results

    def _analyze_correlations(self, data: np.ndarray, regimes: List[Dict],
                            architectures: List[Dict]) -> Dict:
        """Analyze correlations between regimes and architectures."""
        correlation_results = {}

        # Extract regime features
        regime_features = []
        for regime in regimes:
            features = [
                regime.get('volatility', 0),
                regime.get('trend', 0),
                regime.get('price_change', 0)
            ]
            regime_features.append(features)

        regime_features = np.array(regime_features)

        # Calculate correlations
        if len(regime_features) > 1:
            # Volatility vs trend correlation
            volatility_trend_corr = np.corrcoef(
                regime_features[:, 0], regime_features[:, 1]
            )[0, 1]

            # Volatility vs price change correlation
            volatility_price_corr = np.corrcoef(
                regime_features[:, 0], regime_features[:, 2]
            )[0, 1]

            # Trend vs price change correlation
            trend_price_corr = np.corrcoef(
                regime_features[:, 1], regime_features[:, 2]
            )[0, 1]
        else:
            volatility_trend_corr = 0
            volatility_price_corr = 0
            trend_price_corr = 0

        correlation_results = {
            'volatility_trend_correlation': volatility_trend_corr,
            'volatility_price_correlation': volatility_price_corr,
            'trend_price_correlation': trend_price_corr,
            'regime_feature_matrix': regime_features.tolist()
        }

        return correlation_results

    def get_analysis_results(self) -> List[Dict]:
        """Get analysis results history."""
        return self.analysis_results

    def get_regime_patterns(self) -> Dict:
        """Get detected regime patterns."""
        return self.regime_patterns

    def get_analysis_metrics(self) -> Dict:
        """Get analysis metrics."""
        return self.analysis_metrics

    def get_regime_summary(self, data: np.ndarray, regimes: List[Dict]) -> Dict:
        """Get summary of regime analysis."""
        if not regimes:
            return {}

        # Calculate basic statistics
        regime_types = [regime['regime_type'] for regime in regimes]
        regime_counts = {}
        for regime_type in regime_types:
            regime_counts[regime_type] = regime_counts.get(regime_type, 0) + 1

        # Calculate regime durations
        regime_durations = []
        current_regime = regime_types[0]
        duration = 1

        for i in range(1, len(regime_types)):
            if regime_types[i] == current_regime:
                duration += 1
            else:
                regime_durations.append(duration)
                current_regime = regime_types[i]
                duration = 1

        regime_durations.append(duration)

        return {
            'total_regimes': len(regimes),
            'regime_counts': regime_counts,
            'regime_percentages': {
                regime: count / len(regimes) * 100
                for regime, count in regime_counts.items()
            },
            'avg_duration': np.mean(regime_durations),
            'duration_std': np.std(regime_durations),
            'most_common_regime': max(regime_counts, key=regime_counts.get),
            'regime_diversity': len(set(regime_types))
        }
