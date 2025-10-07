"""
Main Orchestrator for Data-Driven Lookback Optimization System

This module orchestrates the three-stage Bayesian optimization system:
1. IC Surface Estimation with HAC standard errors
2. Walk-Forward Stability Testing with purged CV
3. Hierarchical Bayesian Shrinkage across families and symbols

The system replaces hardcoded lookback ceilings with data-driven inference
while maintaining production constraints and latency requirements.
"""

import logging
import time
import traceback
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import warnings
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd

# Import all stage modules
from .config import LookbackOptimizationConfig, FamilyType, create_default_config
from .ic_surface import ICSurfaceEstimator, ICSurfaceResult
from .wf_stability import StabilityTester, StabilityResult, MultiFamilyStabilityTester
from .hierarchical import HierarchicalBayesianShrinkage, HierarchicalResult, MultiSymbolHierarchicalShrinkage
from .decision import LookbackDecisionMaker, DecisionResult, MultiFamilyDecisionMaker
from .feature_families import MultiFamilyFeatureGenerator, FeatureResult

# Import utilities
try:
    from src.utils.tprint import tprint, tprint_info, tprint_error, tprint_warning, tprint_success
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Complete result of the lookback optimization system."""
    ic_surface_results: Dict[str, Dict[FamilyType, ICSurfaceResult]]
    stability_results: Dict[str, Dict[FamilyType, StabilityResult]]
    hierarchical_results: Dict[str, HierarchicalResult]
    decisions: Dict[str, Dict[FamilyType, DecisionResult]]
    feature_results: Dict[str, Dict[FamilyType, FeatureResult]]
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'ic_surface_results': {
                symbol: {family.value: result.to_dict() for family, result in symbol_results.items()}
                for symbol, symbol_results in self.ic_surface_results.items()
            },
            'stability_results': {
                symbol: {family.value: result.to_dict() for family, result in symbol_results.items()}
                for symbol, symbol_results in self.stability_results.items()
            },
            'hierarchical_results': {
                symbol: result.to_dict() for symbol, result in self.hierarchical_results.items()
            },
            'decisions': {
                symbol: {family.value: result.to_dict() for family, result in symbol_results.items()}
                for symbol, symbol_results in self.decisions.items()
            },
            'feature_results': {
                symbol: {family.value: result.to_dict() for family, result in symbol_results.items()}
                for symbol, symbol_results in self.feature_results.items()
            },
            'execution_time': self.execution_time,
            'success': self.success,
            'error_message': self.error_message
        }


class LookbackOptimizationOrchestrator:
    """Main orchestrator for the lookback optimization system."""
    
    def __init__(self, config: Optional[LookbackOptimizationConfig] = None):
        self.config = config or create_default_config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize stage components
        self.ic_estimator = ICSurfaceEstimator(self.config)
        self.stability_tester = MultiFamilyStabilityTester(self.config)
        self.hierarchical_shrinkage = MultiSymbolHierarchicalShrinkage(self.config)
        self.decision_maker = MultiFamilyDecisionMaker(self.config)
        self.feature_generator = MultiFamilyFeatureGenerator(self.config)
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def optimize_lookbacks(self, 
                          data: Dict[str, pd.DataFrame],
                          targets: Dict[str, np.ndarray],
                          feature_names: Optional[Dict[FamilyType, str]] = None) -> OptimizationResult:
        """Run the complete lookback optimization pipeline."""
        start_time = time.time()
        
        try:
            tprint_success("🚀 Starting data-driven lookback optimization system...")
            
            # Validate inputs
            self._validate_inputs(data, targets)
            
            # Set default feature names
            if feature_names is None:
                feature_names = {family: f"{family.value}_feature" for family in FamilyType}
            
            # Stage 1: IC Surface Estimation
            tprint_info("📊 Stage 1: Estimating IC surfaces with HAC standard errors...")
            ic_surface_results = self._run_stage_1(data, targets, feature_names)
            
            # Stage 2: Walk-Forward Stability Testing
            tprint_info("🔄 Stage 2: Testing stability with purged walk-forward validation...")
            stability_results = self._run_stage_2(data, targets, ic_surface_results, feature_names)
            
            # Stage 3: Hierarchical Bayesian Shrinkage
            tprint_info("🎯 Stage 3: Applying hierarchical Bayesian shrinkage...")
            hierarchical_results = self._run_stage_3(ic_surface_results, stability_results)
            
            # Decision Making
            tprint_info("🤔 Making lookback decisions with hysteresis...")
            decisions = self._make_decisions(ic_surface_results, stability_results, 
                                          hierarchical_results, data, targets, feature_names)
            
            # Feature Generation
            tprint_info("⚙️ Generating optimized features...")
            feature_results = self._generate_features(data, decisions, feature_names)
            
            # Save results
            if self.config.save_intermediate_results:
                self._save_results(ic_surface_results, stability_results, 
                                 hierarchical_results, decisions, feature_results)
            
            execution_time = time.time() - start_time
            
            result = OptimizationResult(
                ic_surface_results=ic_surface_results,
                stability_results=stability_results,
                hierarchical_results=hierarchical_results,
                decisions=decisions,
                feature_results=feature_results,
                execution_time=execution_time,
                success=True
            )
            
            tprint_success(f"✅ Lookback optimization completed successfully in {execution_time:.3f}s")
            self._print_summary(result)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = f"Lookback optimization failed: {str(e)}"
            
            self.logger.error(error_message)
            self.logger.error(f"Error details: {traceback.format_exc()}")
            
            return OptimizationResult(
                ic_surface_results={},
                stability_results={},
                hierarchical_results={},
                decisions={},
                feature_results={},
                execution_time=execution_time,
                success=False,
                error_message=error_message
            )
    
    def _validate_inputs(self, data: Dict[str, pd.DataFrame], targets: Dict[str, np.ndarray]) -> None:
        """Validate input data and targets."""
        if not data:
            raise ValueError("No data provided")
        
        if not targets:
            raise ValueError("No targets provided")
        
        # Check that all symbols have both data and targets
        for symbol in data.keys():
            if symbol not in targets:
                raise ValueError(f"No target provided for symbol {symbol}")
            
            if len(data[symbol]) != len(targets[symbol]):
                raise ValueError(f"Data and target length mismatch for symbol {symbol}")
        
        # Check minimum data requirements
        for symbol, df in data.items():
            if len(df) < 1000:
                raise ValueError(f"Insufficient data for symbol {symbol}: {len(df)} < 1000")
    
    def _run_stage_1(self, data: Dict[str, pd.DataFrame], targets: Dict[str, np.ndarray],
                    feature_names: Dict[FamilyType, str]) -> Dict[str, Dict[FamilyType, ICSurfaceResult]]:
        """Run Stage 1: IC Surface Estimation."""
        results = {}
        
        for symbol, symbol_data in data.items():
            symbol_results = {}
            symbol_target = targets[symbol]
            
            tprint_info(f"Processing {symbol}...")
            
            for family in FamilyType:
                try:
                    feature_name = feature_names[family]
                    
                    ic_result = self.ic_estimator.estimate_surface(
                        symbol_data, symbol_target, family, feature_name
                    )
                    
                    symbol_results[family] = ic_result
                    
                except Exception as e:
                    self.logger.warning(f"Failed to estimate IC surface for {symbol}-{family.value}: {e}")
                    continue
            
            results[symbol] = symbol_results
        
        return results
    
    def _run_stage_2(self, data: Dict[str, pd.DataFrame], targets: Dict[str, np.ndarray],
                    ic_surface_results: Dict[str, Dict[FamilyType, ICSurfaceResult]],
                    feature_names: Dict[FamilyType, str]) -> Dict[str, Dict[FamilyType, StabilityResult]]:
        """Run Stage 2: Walk-Forward Stability Testing."""
        results = {}
        
        for symbol, symbol_data in data.items():
            symbol_ic_results = ic_surface_results.get(symbol, {})
            symbol_target = targets[symbol]
            
            if not symbol_ic_results:
                continue
            
            tprint_info(f"Testing stability for {symbol}...")
            
            try:
                symbol_stability_results = self.stability_tester.test_all_families(
                    symbol_data, symbol_target, symbol_ic_results, feature_names
                )
                
                results[symbol] = symbol_stability_results
                
            except Exception as e:
                self.logger.warning(f"Failed to test stability for {symbol}: {e}")
                continue
        
        return results
    
    def _run_stage_3(self, ic_surface_results: Dict[str, Dict[FamilyType, ICSurfaceResult]],
                    stability_results: Dict[str, Dict[FamilyType, StabilityResult]]) -> Dict[str, HierarchicalResult]:
        """Run Stage 3: Hierarchical Bayesian Shrinkage."""
        try:
            hierarchical_results = self.hierarchical_shrinkage.apply_multi_symbol_shrinkage(
                ic_surface_results, stability_results
            )
            
            return hierarchical_results
            
        except Exception as e:
            self.logger.warning(f"Hierarchical shrinkage failed: {e}")
            return {}
    
    def _make_decisions(self, ic_surface_results: Dict[str, Dict[FamilyType, ICSurfaceResult]],
                       stability_results: Dict[str, Dict[FamilyType, StabilityResult]],
                       hierarchical_results: Dict[str, HierarchicalResult],
                       data: Dict[str, pd.DataFrame],
                       targets: Dict[str, np.ndarray],
                       feature_names: Dict[FamilyType, str]) -> Dict[str, Dict[FamilyType, DecisionResult]]:
        """Make lookback decisions for all symbol-family combinations."""
        try:
            decisions = self.decision_maker.make_all_decisions(
                ic_surface_results, stability_results, hierarchical_results,
                data, targets, feature_names
            )
            
            return decisions
            
        except Exception as e:
            self.logger.warning(f"Decision making failed: {e}")
            return {}
    
    def _generate_features(self, data: Dict[str, pd.DataFrame],
                          decisions: Dict[str, Dict[FamilyType, DecisionResult]],
                          feature_names: Dict[FamilyType, str]) -> Dict[str, Dict[FamilyType, FeatureResult]]:
        """Generate optimized features for all symbol-family combinations."""
        try:
            feature_results = self.feature_generator.generate_all_symbols_features(
                data, decisions, feature_names
            )
            
            return feature_results
            
        except Exception as e:
            self.logger.warning(f"Feature generation failed: {e}")
            return {}
    
    def _save_results(self, ic_surface_results: Dict[str, Dict[FamilyType, ICSurfaceResult]],
                     stability_results: Dict[str, Dict[FamilyType, StabilityResult]],
                     hierarchical_results: Dict[str, HierarchicalResult],
                     decisions: Dict[str, Dict[FamilyType, DecisionResult]],
                     feature_results: Dict[str, Dict[FamilyType, FeatureResult]]) -> None:
        """Save intermediate results to disk."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save IC surface results
            ic_path = os.path.join(self.config.output_dir, f"ic_surface_results_{timestamp}.json")
            with open(ic_path, 'w') as f:
                json.dump({
                    symbol: {family.value: result.to_dict() for family, result in symbol_results.items()}
                    for symbol, symbol_results in ic_surface_results.items()
                }, f, indent=2)
            
            # Save stability results
            stability_path = os.path.join(self.config.output_dir, f"stability_results_{timestamp}.json")
            with open(stability_path, 'w') as f:
                json.dump({
                    symbol: {family.value: result.to_dict() for family, result in symbol_results.items()}
                    for symbol, symbol_results in stability_results.items()
                }, f, indent=2)
            
            # Save hierarchical results
            hierarchical_path = os.path.join(self.config.output_dir, f"hierarchical_results_{timestamp}.json")
            with open(hierarchical_path, 'w') as f:
                json.dump({
                    symbol: result.to_dict() for symbol, result in hierarchical_results.items()
                }, f, indent=2)
            
            # Save decisions
            decisions_path = os.path.join(self.config.output_dir, f"decisions_{timestamp}.json")
            with open(decisions_path, 'w') as f:
                json.dump({
                    symbol: {family.value: result.to_dict() for family, result in symbol_results.items()}
                    for symbol, symbol_results in decisions.items()
                }, f, indent=2)
            
            # Save feature results
            feature_path = os.path.join(self.config.output_dir, f"feature_results_{timestamp}.json")
            with open(feature_path, 'w') as f:
                json.dump({
                    symbol: {family.value: result.to_dict() for family, result in symbol_results.items()}
                    for symbol, symbol_results in feature_results.items()
                }, f, indent=2)
            
            tprint_info(f"Results saved to {self.config.output_dir}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save results: {e}")
    
    def _print_summary(self, result: OptimizationResult) -> None:
        """Print optimization summary."""
        tprint_success("📊 OPTIMIZATION SUMMARY")
        tprint_success(f"Execution time: {result.execution_time:.3f}s")
        tprint_success(f"Symbols processed: {len(result.ic_surface_results)}")
        
        # Count features by type
        decision_counts = {'discrete': 0, 'blend': 0, 'default': 0, 'inactive': 0}
        for symbol_decisions in result.decisions.values():
            for decision in symbol_decisions.values():
                decision_type = decision.lookback_spec.decision_type.value
                decision_counts[decision_type] += 1
        
        tprint_success(f"Decision types: {decision_counts}")
        
        # Quality metrics
        if result.feature_results:
            all_quality_scores = []
            for symbol_results in result.feature_results.values():
                for feature_result in symbol_results.values():
                    all_quality_scores.append(feature_result.quality_score)
            
            if all_quality_scores:
                avg_quality = np.mean(all_quality_scores)
                tprint_success(f"Average feature quality: {avg_quality:.3f}")
    
    def generate_comprehensive_report(self, result: OptimizationResult) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        report = {
            'execution_summary': {
                'success': result.success,
                'execution_time': result.execution_time,
                'error_message': result.error_message,
                'symbols_processed': len(result.ic_surface_results),
                'families_processed': len(FamilyType)
            },
            'stage_1_summary': self._summarize_ic_surfaces(result.ic_surface_results),
            'stage_2_summary': self._summarize_stability(result.stability_results),
            'stage_3_summary': self._summarize_hierarchical(result.hierarchical_results),
            'decision_summary': self._summarize_decisions(result.decisions),
            'feature_summary': self._summarize_features(result.feature_results),
            'recommendations': self._generate_recommendations(result)
        }
        
        return report
    
    def _summarize_ic_surfaces(self, ic_results: Dict[str, Dict[FamilyType, ICSurfaceResult]]) -> Dict[str, Any]:
        """Summarize IC surface results."""
        summary = {
            'total_estimations': 0,
            'successful_estimations': 0,
            'average_optimal_ic': 0.0,
            'family_performance': {}
        }
        
        all_ics = []
        
        for family in FamilyType:
            family_ics = []
            for symbol_results in ic_results.values():
                if family in symbol_results:
                    family_ics.append(symbol_results[family].optimal_ic)
                    summary['total_estimations'] += 1
                    summary['successful_estimations'] += 1
            
            if family_ics:
                summary['family_performance'][family.value] = {
                    'average_ic': np.mean(family_ics),
                    'std_ic': np.std(family_ics),
                    'count': len(family_ics)
                }
                all_ics.extend(family_ics)
        
        if all_ics:
            summary['average_optimal_ic'] = np.mean(all_ics)
        
        return summary
    
    def _summarize_stability(self, stability_results: Dict[str, Dict[FamilyType, StabilityResult]]) -> Dict[str, Any]:
        """Summarize stability results."""
        summary = {
            'total_tests': 0,
            'stable_families': 0,
            'blend_recommended': 0,
            'unstable_families': 0,
            'average_stability_score': 0.0
        }
        
        all_stability_scores = []
        
        for symbol_results in stability_results.values():
            for result in symbol_results.values():
                summary['total_tests'] += 1
                all_stability_scores.append(result.stability_score)
                
                if result.recommendation == "stable":
                    summary['stable_families'] += 1
                elif result.recommendation == "blend_recommended":
                    summary['blend_recommended'] += 1
                else:
                    summary['unstable_families'] += 1
        
        if all_stability_scores:
            summary['average_stability_score'] = np.mean(all_stability_scores)
        
        return summary
    
    def _summarize_hierarchical(self, hierarchical_results: Dict[str, HierarchicalResult]) -> Dict[str, Any]:
        """Summarize hierarchical shrinkage results."""
        summary = {
            'total_shrinkage_applications': len(hierarchical_results),
            'average_shrinkage_factor': 0.0,
            'convergence_issues': 0
        }
        
        all_shrinkage_factors = []
        
        for result in hierarchical_results.values():
            shrinkage_factors = list(result.shrinkage_factors.values())
            all_shrinkage_factors.extend(shrinkage_factors)
            
            if 'error' in result.convergence_diagnostics:
                summary['convergence_issues'] += 1
        
        if all_shrinkage_factors:
            summary['average_shrinkage_factor'] = np.mean(all_shrinkage_factors)
        
        return summary
    
    def _summarize_decisions(self, decisions: Dict[str, Dict[FamilyType, DecisionResult]]) -> Dict[str, Any]:
        """Summarize decision results."""
        summary = {
            'total_decisions': 0,
            'decision_type_counts': {'discrete': 0, 'blend': 0, 'default': 0, 'inactive': 0},
            'average_confidence': 0.0,
            'families_with_changes': 0
        }
        
        all_confidence_scores = []
        families_with_changes = set()
        
        for symbol_results in decisions.values():
            for family, decision in symbol_results.items():
                summary['total_decisions'] += 1
                decision_type = decision.lookback_spec.decision_type.value
                summary['decision_type_counts'][decision_type] += 1
                
                all_confidence_scores.append(decision.lookback_spec.confidence_score)
                
                if decision.change_magnitude > 0.1:
                    families_with_changes.add(family)
        
        if all_confidence_scores:
            summary['average_confidence'] = np.mean(all_confidence_scores)
        
        summary['families_with_changes'] = len(families_with_changes)
        
        return summary
    
    def _summarize_features(self, feature_results: Dict[str, Dict[FamilyType, FeatureResult]]) -> Dict[str, Any]:
        """Summarize feature generation results."""
        summary = {
            'total_features_generated': 0,
            'average_generation_time': 0.0,
            'average_quality_score': 0.0,
            'total_memory_usage_mb': 0.0
        }
        
        all_generation_times = []
        all_quality_scores = []
        total_memory = 0.0
        
        for symbol_results in feature_results.values():
            for result in symbol_results.values():
                summary['total_features_generated'] += 1
                all_generation_times.append(result.generation_time)
                all_quality_scores.append(result.quality_score)
                total_memory += result.memory_usage_mb
        
        if all_generation_times:
            summary['average_generation_time'] = np.mean(all_generation_times)
        if all_quality_scores:
            summary['average_quality_score'] = np.mean(all_quality_scores)
        
        summary['total_memory_usage_mb'] = total_memory
        
        return summary
    
    def _generate_recommendations(self, result: OptimizationResult) -> List[str]:
        """Generate recommendations based on optimization results."""
        recommendations = []
        
        # Check for inactive families
        inactive_count = 0
        for symbol_decisions in result.decisions.values():
            for decision in symbol_decisions.values():
                if decision.lookback_spec.decision_type.value == 'inactive':
                    inactive_count += 1
        
        if inactive_count > 0:
            recommendations.append(f"Consider removing {inactive_count} inactive families")
        
        # Check for low quality features
        if result.feature_results:
            low_quality_count = 0
            for symbol_results in result.feature_results.values():
                for feature_result in symbol_results.values():
                    if feature_result.quality_score < 0.3:
                        low_quality_count += 1
            
            if low_quality_count > 0:
                recommendations.append(f"Review {low_quality_count} low-quality features")
        
        # Check for high memory usage
        if result.feature_results:
            total_memory = sum(
                sum(feature_result.memory_usage_mb for feature_result in symbol_results.values())
                for symbol_results in result.feature_results.values()
            )
            
            if total_memory > 1000:  # More than 1GB
                recommendations.append(f"High memory usage: {total_memory:.1f}MB - consider optimization")
        
        return recommendations