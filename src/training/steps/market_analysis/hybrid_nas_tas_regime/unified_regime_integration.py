"""
Unified Regime Integration for Hybrid TAS-NAS System

This module provides integration between the existing hybrid regime system
and the new unified regime detection system, allowing seamless switching
and combination of approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time
from datetime import datetime

# Import tprint for comprehensive logging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

# Import unified regime system
try:
    from src.utils.ml_common.nas_tas_unified import (
        UnifiedRegimeDetector, UnifiedRegimeConfig, UnifiedRegimeResult,
        RegimeDetectionMethod, OptimizationStrategy, EconomicEvaluationMode
    )
    UNIFIED_SYSTEM_AVAILABLE = True
except ImportError:
    UNIFIED_SYSTEM_AVAILABLE = False

# Import existing hybrid components
try:
    from .core.hybrid_regime_detector import HybridRegimeDetector
    from .shared_utils.search_strategies import SearchStrategyManager
    from .shared_utils.analysis_components import SharedClusteringUtilities
    from .shared_utils.position_aware_trading import PositionAwareTradingAnalyzer
    HYBRID_SYSTEM_AVAILABLE = True
except ImportError:
    HYBRID_SYSTEM_AVAILABLE = False

# Import TAS and NAS components for comparison
try:
    from src.training.steps.market_analysis.tas_regime.core.tas_regime_detector import (
        TASRegimeDetector, TASRegimeResult
    )
    from src.training.steps.market_analysis.nas_regime.core.perfect_nas_regime_detector import (
        PerfectNASRegimeDetector, PerfectNASResult
    )
    LEGACY_SYSTEMS_AVAILABLE = True
except ImportError:
    LEGACY_SYSTEMS_AVAILABLE = False

logger = logging.getLogger(__name__)

class UnifiedRegimeIntegration:
    """
    Integration layer between unified regime system and existing hybrid system.
    
    Provides seamless switching between different regime detection approaches
    and comprehensive comparison capabilities.
    """
    
    def __init__(self, config: Optional[UnifiedRegimeConfig] = None):
        """Initialize unified regime integration."""
        tprint_info("🚀 Initializing Unified Regime Integration")
        
        self.config = config or UnifiedRegimeConfig.create_production_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize systems
        self.unified_detector = None
        self.hybrid_detector = None
        self.tas_detector = None
        self.nas_detector = None
        
        self._initialize_systems()
        
        # Performance comparison tracking
        self.comparison_metrics = {
            'unified': {'accuracy': [], 'efficiency': [], 'economic_score': []},
            'hybrid': {'accuracy': [], 'efficiency': [], 'economic_score': []},
            'tas': {'accuracy': [], 'efficiency': [], 'economic_score': []},
            'nas': {'accuracy': [], 'efficiency': [], 'economic_score': []}
        }
        
        tprint_success("✅ Unified Regime Integration initialized successfully")
    
    def _initialize_systems(self):
        """Initialize all available regime detection systems."""
        
        # Initialize unified system
        if UNIFIED_SYSTEM_AVAILABLE:
            try:
                self.unified_detector = UnifiedRegimeDetector(self.config)
                tprint_success("✅ Unified regime detector initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Unified detector initialization failed: {e}")
                self.unified_detector = None
        
        # Initialize hybrid system
        if HYBRID_SYSTEM_AVAILABLE:
            try:
                # Create hybrid config from unified config
                hybrid_config = self._create_hybrid_config()
                self.hybrid_detector = HybridRegimeDetector(hybrid_config)
                tprint_success("✅ Hybrid regime detector initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Hybrid detector initialization failed: {e}")
                self.hybrid_detector = None
        
        # Initialize legacy systems for comparison
        if LEGACY_SYSTEMS_AVAILABLE:
            try:
                # Initialize TAS detector
                tas_config = self._create_tas_config()
                self.tas_detector = TASRegimeDetector(tas_config)
                tprint_success("✅ TAS regime detector initialized")
                
                # Initialize NAS detector
                nas_config = self._create_nas_config()
                self.nas_detector = PerfectNASRegimeDetector(nas_config)
                tprint_success("✅ NAS regime detector initialized")
                
            except Exception as e:
                tprint_warning(f"⚠️ Legacy systems initialization failed: {e}")
                self.tas_detector = None
                self.nas_detector = None
    
    def _create_hybrid_config(self) -> Dict[str, Any]:
        """Create hybrid configuration from unified config."""
        return {
            'n_regimes': self.config.n_regimes,
            'primary_timeframe': self.config.primary_timeframe,
            'enable_tas': self.config.should_use_tas(),
            'enable_nas': self.config.should_use_nas(),
            'economic_evaluation': self.config.economic_evaluation.value,
            'optimization_strategy': self.config.optimization_strategy.value,
            'max_execution_time': self.config.max_execution_time,
            'enable_hardware_optimization': self.config.enable_hardware_optimization
        }
    
    def _create_tas_config(self):
        """Create TAS configuration from unified config."""
        from src.training.steps.market_analysis.tas_regime.core.tas_regime_config import TASRegimeConfig
        
        return TASRegimeConfig(
            n_regimes=self.config.n_regimes,
            primary_timeframe=self.config.primary_timeframe,
            min_regime_samples=self.config.min_regime_samples,
            max_regime_samples=self.config.max_regime_samples,
            enable_economic_evaluation=True,
            economic_significance_threshold=self.config.economic_significance_threshold,
            trading_viability_threshold=self.config.trading_viability_threshold,
            max_execution_time=self.config.max_execution_time
        )
    
    def _create_nas_config(self):
        """Create NAS configuration from unified config."""
        from src.training.steps.market_analysis.nas_regime.core.perfect_nas_config import PerfectNASConfig
        
        return PerfectNASConfig(
            n_regimes=self.config.n_regimes,
            primary_timeframe=self.config.primary_timeframe,
            min_regime_duration=self.config.min_regime_samples,
            max_regime_duration=self.config.max_regime_samples,
            accuracy_threshold=self.config.target_accuracy,
            economic_significance_threshold=self.config.economic_significance_threshold,
            trading_viability_threshold=self.config.trading_viability_threshold,
            max_execution_time=self.config.max_execution_time
        )
    
    def compare_all_systems(self,
                           market_data: Union[pd.DataFrame, np.ndarray],
                           timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Compare all available regime detection systems.
        
        Args:
            market_data: Market data for regime detection
            timestamps: Optional timestamps
            
        Returns:
            Dictionary with comparison results from all systems
        """
        tprint_info("🔄 Comparing all regime detection systems")
        
        results = {}
        start_time = time.time()
        
        # Run unified system
        if self.unified_detector:
            try:
                tprint_info("🧠 Running unified system")
                unified_start = time.time()
                unified_result = self.unified_detector.detect_regimes(market_data, timestamps)
                unified_time = time.time() - unified_start
                
                results['unified'] = {
                    'result': unified_result,
                    'execution_time': unified_time,
                    'success': unified_result.success,
                    'accuracy': self._calculate_accuracy_metric(unified_result),
                    'efficiency': 1.0 / (unified_time + 1e-8),
                    'economic_score': np.mean(unified_result.economic_significance_scores) if len(unified_result.economic_significance_scores) > 0 else 0.0
                }
                tprint_success(f"✅ Unified system completed in {unified_time:.2f}s")
                
            except Exception as e:
                tprint_error(f"❌ Unified system failed: {e}")
                results['unified'] = {'error': str(e), 'success': False}
        
        # Run hybrid system
        if self.hybrid_detector:
            try:
                tprint_info("🔀 Running hybrid system")
                hybrid_start = time.time()
                hybrid_result = self.hybrid_detector.detect_regimes(market_data, timestamps)
                hybrid_time = time.time() - hybrid_start
                
                results['hybrid'] = {
                    'result': hybrid_result,
                    'execution_time': hybrid_time,
                    'success': hybrid_result.success if hasattr(hybrid_result, 'success') else True,
                    'accuracy': self._calculate_accuracy_metric(hybrid_result),
                    'efficiency': 1.0 / (hybrid_time + 1e-8),
                    'economic_score': self._extract_economic_score(hybrid_result)
                }
                tprint_success(f"✅ Hybrid system completed in {hybrid_time:.2f}s")
                
            except Exception as e:
                tprint_error(f"❌ Hybrid system failed: {e}")
                results['hybrid'] = {'error': str(e), 'success': False}
        
        # Run TAS system
        if self.tas_detector:
            try:
                tprint_info("🌲 Running TAS system")
                tas_start = time.time()
                tas_result = self.tas_detector.detect_regimes(market_data, timestamps)
                tas_time = time.time() - tas_start
                
                results['tas'] = {
                    'result': tas_result,
                    'execution_time': tas_time,
                    'success': tas_result.success,
                    'accuracy': np.mean(tas_result.regime_stability_scores) if len(tas_result.regime_stability_scores) > 0 else 0.0,
                    'efficiency': 1.0 / (tas_time + 1e-8),
                    'economic_score': np.mean(tas_result.economic_significance_scores) if len(tas_result.economic_significance_scores) > 0 else 0.0
                }
                tprint_success(f"✅ TAS system completed in {tas_time:.2f}s")
                
            except Exception as e:
                tprint_error(f"❌ TAS system failed: {e}")
                results['tas'] = {'error': str(e), 'success': False}
        
        # Run NAS system
        if self.nas_detector:
            try:
                tprint_info("🧠 Running NAS system")
                nas_start = time.time()
                nas_result = self.nas_detector.detect_regimes(market_data, timestamps)
                nas_time = time.time() - nas_start
                
                results['nas'] = {
                    'result': nas_result,
                    'execution_time': nas_time,
                    'success': nas_result.success,
                    'accuracy': np.mean(nas_result.regime_stability_scores) if len(nas_result.regime_stability_scores) > 0 else 0.0,
                    'efficiency': 1.0 / (nas_time + 1e-8),
                    'economic_score': np.mean(nas_result.economic_significance_scores) if len(nas_result.economic_significance_scores) > 0 else 0.0
                }
                tprint_success(f"✅ NAS system completed in {nas_time:.2f}s")
                
            except Exception as e:
                tprint_error(f"❌ NAS system failed: {e}")
                results['nas'] = {'error': str(e), 'success': False}
        
        total_time = time.time() - start_time
        tprint_success(f"✅ All systems comparison completed in {total_time:.2f}s")
        
        # Update comparison metrics
        self._update_comparison_metrics(results)
        
        # Add summary
        results['summary'] = self._generate_comparison_summary(results)
        
        return results
    
    def _calculate_accuracy_metric(self, result) -> float:
        """Calculate accuracy metric from result."""
        try:
            if hasattr(result, 'regime_stability_scores') and len(result.regime_stability_scores) > 0:
                return np.mean(result.regime_stability_scores)
            elif hasattr(result, 'accuracy'):
                return result.accuracy
            else:
                return 0.5  # Default accuracy
        except Exception:
            return 0.5
    
    def _extract_economic_score(self, result) -> float:
        """Extract economic score from result."""
        try:
            if hasattr(result, 'economic_significance_scores') and len(result.economic_significance_scores) > 0:
                return np.mean(result.economic_significance_scores)
            elif hasattr(result, 'economic_score'):
                return result.economic_score
            else:
                return 0.5  # Default economic score
        except Exception:
            return 0.5
    
    def _update_comparison_metrics(self, results: Dict[str, Any]):
        """Update comparison metrics with new results."""
        for system_name, system_results in results.items():
            if system_name == 'summary' or 'error' in system_results:
                continue
            
            if system_results['success']:
                self.comparison_metrics[system_name]['accuracy'].append(system_results['accuracy'])
                self.comparison_metrics[system_name]['efficiency'].append(system_results['efficiency'])
                self.comparison_metrics[system_name]['economic_score'].append(system_results['economic_score'])
    
    def _generate_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of comparison results."""
        summary = {
            'total_systems_tested': 0,
            'successful_systems': 0,
            'failed_systems': 0,
            'best_accuracy': {'system': None, 'score': 0.0},
            'best_efficiency': {'system': None, 'score': 0.0},
            'best_economic_score': {'system': None, 'score': 0.0},
            'recommended_system': None
        }
        
        for system_name, system_results in results.items():
            if system_name == 'summary':
                continue
            
            summary['total_systems_tested'] += 1
            
            if 'error' in system_results:
                summary['failed_systems'] += 1
                continue
            
            summary['successful_systems'] += 1
            
            # Track best performers
            if system_results['accuracy'] > summary['best_accuracy']['score']:
                summary['best_accuracy'] = {'system': system_name, 'score': system_results['accuracy']}
            
            if system_results['efficiency'] > summary['best_efficiency']['score']:
                summary['best_efficiency'] = {'system': system_name, 'score': system_results['efficiency']}
            
            if system_results['economic_score'] > summary['best_economic_score']['score']:
                summary['best_economic_score'] = {'system': system_name, 'score': system_results['economic_score']}
        
        # Determine recommended system based on optimization strategy
        if self.config.optimization_strategy == OptimizationStrategy.ACCURACY_FIRST:
            summary['recommended_system'] = summary['best_accuracy']['system']
        elif self.config.optimization_strategy == OptimizationStrategy.PERFORMANCE_FIRST:
            summary['recommended_system'] = summary['best_efficiency']['system']
        elif self.config.optimization_strategy == OptimizationStrategy.ECONOMIC_FOCUSED:
            summary['recommended_system'] = summary['best_economic_score']['system']
        else:  # BALANCED
            # Calculate balanced score
            best_balanced_system = None
            best_balanced_score = 0.0
            
            for system_name, system_results in results.items():
                if system_name == 'summary' or 'error' in system_results:
                    continue
                
                balanced_score = (
                    0.4 * system_results['accuracy'] +
                    0.3 * system_results['efficiency'] +
                    0.3 * system_results['economic_score']
                )
                
                if balanced_score > best_balanced_score:
                    best_balanced_score = balanced_score
                    best_balanced_system = system_name
            
            summary['recommended_system'] = best_balanced_system
        
        return summary
    
    def get_comparison_metrics(self) -> Dict[str, Any]:
        """Get historical comparison metrics."""
        metrics = {}
        
        for system_name, system_metrics in self.comparison_metrics.items():
            if system_metrics['accuracy']:
                metrics[system_name] = {
                    'avg_accuracy': np.mean(system_metrics['accuracy']),
                    'avg_efficiency': np.mean(system_metrics['efficiency']),
                    'avg_economic_score': np.mean(system_metrics['economic_score']),
                    'total_runs': len(system_metrics['accuracy']),
                    'std_accuracy': np.std(system_metrics['accuracy']),
                    'std_efficiency': np.std(system_metrics['efficiency']),
                    'std_economic_score': np.std(system_metrics['economic_score'])
                }
            else:
                metrics[system_name] = {
                    'avg_accuracy': 0.0,
                    'avg_efficiency': 0.0,
                    'avg_economic_score': 0.0,
                    'total_runs': 0,
                    'std_accuracy': 0.0,
                    'std_efficiency': 0.0,
                    'std_economic_score': 0.0
                }
        
        return metrics
    
    def recommend_best_system(self) -> str:
        """Recommend the best performing system based on historical data."""
        metrics = self.get_comparison_metrics()
        
        if not metrics:
            return "unified"  # Default recommendation
        
        best_system = None
        best_score = 0.0
        
        for system_name, system_metrics in metrics.items():
            if system_metrics['total_runs'] == 0:
                continue
            
            # Calculate recommendation score based on optimization strategy
            if self.config.optimization_strategy == OptimizationStrategy.ACCURACY_FIRST:
                score = system_metrics['avg_accuracy']
            elif self.config.optimization_strategy == OptimizationStrategy.PERFORMANCE_FIRST:
                score = system_metrics['avg_efficiency']
            elif self.config.optimization_strategy == OptimizationStrategy.ECONOMIC_FOCUSED:
                score = system_metrics['avg_economic_score']
            else:  # BALANCED
                score = (
                    0.4 * system_metrics['avg_accuracy'] +
                    0.3 * system_metrics['avg_efficiency'] +
                    0.3 * system_metrics['avg_economic_score']
                )
            
            if score > best_score:
                best_score = score
                best_system = system_name
        
        return best_system or "unified"
    
    def run_recommended_system(self,
                              market_data: Union[pd.DataFrame, np.ndarray],
                              timestamps: Optional[np.ndarray] = None) -> Any:
        """Run the recommended best performing system."""
        recommended_system = self.recommend_best_system()
        
        tprint_info(f"🎯 Running recommended system: {recommended_system}")
        
        if recommended_system == "unified" and self.unified_detector:
            return self.unified_detector.detect_regimes(market_data, timestamps)
        elif recommended_system == "hybrid" and self.hybrid_detector:
            return self.hybrid_detector.detect_regimes(market_data, timestamps)
        elif recommended_system == "tas" and self.tas_detector:
            return self.tas_detector.detect_regimes(market_data, timestamps)
        elif recommended_system == "nas" and self.nas_detector:
            return self.nas_detector.detect_regimes(market_data, timestamps)
        else:
            # Fallback to unified system
            if self.unified_detector:
                return self.unified_detector.detect_regimes(market_data, timestamps)
            else:
                raise RuntimeError("No regime detection systems available")