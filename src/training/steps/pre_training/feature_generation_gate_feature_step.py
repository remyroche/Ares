"""
Feature Generation Gate Feature Step

This step implements gate feature protection and quality validation for the feature generation pipeline.
Gate features act as quality gates and protection mechanisms to ensure feature quality before proceeding
to final feature selection and validation.

Features:
- Quality gate validation
- Correlation gate validation
- Variance gate validation
- Feature importance gating
- Comprehensive gate reporting
- Gate state persistence
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path
try:
    from contextlib import nullcontext
except ImportError:
    # Fallback for older Python versions
    from contextlib import contextmanager
    @contextmanager
    def nullcontext():
        yield

# Import BaseStep and step registry
from src.training.steps.base_step import BaseStep

# Import gate feature integration
from src.training.steps.pre_training.gate_feature_integration import (
    GateFeaturePipelineManager,
    GateFeatureConfig,
    GateFeatureResult,
    GateStatus,
    GateFeatureType
)

# Import hardware optimization tools
from src.utils.hardware.unified_hardware_manager import (
    UnifiedHardwareManager,
    HardwareConfig,
    WorkloadType,
    OptimizationLevel
)
from src.utils.memory_management.streaming_data_processor import StreamingDataProcessor, StreamingConfig
from src.feature_selection.caching.intelligent_feature_cache import IntelligentFeatureCache, CacheConfig

# Import VectorBT optimization components
from src.feature_generation.utils.unified_vectorization_manager import (
    UnifiedVectorizationManager,
    VectorizationConfig,
    get_unified_vectorization_manager
)
from src.feature_generation.utils.vectorbt_rolling_optimizer import (
    VectorBTRollingOptimizer,
    get_vectorbt_rolling_optimizer
)

# Import hardware optimization components
try:
    from src.utils.hardware.unified_hardware_manager import WorkloadType, OptimizationLevel
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    WorkloadType = None
    OptimizationLevel = None

# Note: Hardware optimization components are optional for gate feature validation

# Import utilities
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error
from src.utils.artifact_manager import ArtifactManager

logger = logging.getLogger(__name__)


class FeatureGenerationGateFeatureStep(BaseStep):
    """
    Gate feature protection step for the feature generation pipeline.

    This step evaluates gate features to ensure quality and protection mechanisms
    are in place before proceeding to final feature selection.
    """

    def __init__(self, step_name: str = "feature_generation_gate_feature_step"):
        """Initialize the gate feature step."""
        super().__init__(step_name)
        self.gate_manager: Optional[GateFeaturePipelineManager] = None
        
        # VectorBT optimization components
        self.vectorization_manager: Optional[UnifiedVectorizationManager] = None
        self.rolling_optimizer: Optional[VectorBTRollingOptimizer] = None
        self.vectorization_config: Optional[VectorizationConfig] = None

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute gate feature protection and validation.

        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol
                - exchange: Exchange name
                - timeframe: Timeframe
                - execution_mode: Execution mode (light, full, etc.)
                - gate_config: Optional gate configuration overrides

        Returns:
            Dict containing execution results and artifacts
        """
        try:
            tprint_info(f"🛡️ Starting {self.step_name} execution...")

            # Initialize hardware optimization
            self._initialize_hardware_optimization(config)
            
            # Initialize VectorBT optimization components
            self._initialize_vectorbt_optimization(config)

            # Get required data from previous steps
            features_df = self._get_artifact('labeled_dataframe')
            targets = self._get_artifact('targets')

            if features_df is None or targets is None:
                raise ValueError("Required artifacts 'labeled_dataframe' and 'targets' not found")

            # Check if we need streaming processing for large datasets
            needs_streaming = len(features_df) > 50000  # Threshold for streaming

            # Setup gate configuration
            gate_config = self._setup_gate_config(config)

            # Initialize gate manager
            self.gate_manager = GateFeaturePipelineManager(gate_config)

            # Optimize for feature engineering workload
            if self.hardware_manager:
                self.hardware_manager.optimize_for_workload(
                    WorkloadType.FEATURE_ENGINEERING,
                    OptimizationLevel.BALANCED
                )

            # Enable gate protection
            self.gate_manager.enable_gate_protection()

            # Evaluate gate features with optimization
            tprint_info("🔍 Evaluating gate features...")
            start_time = datetime.now()

            # Check cache for existing results
            cache_key = self._generate_cache_key(features_df, targets, gate_config)
            cached_results = None

            if self.feature_cache:
                cached_results = self.feature_cache.get(cache_key)
                if cached_results:
                    self.performance_metrics['cache_hits'] += 1
                    tprint_info("💾 Using cached gate results")
                    gate_results = cached_results
                else:
                    self.performance_metrics['cache_misses'] += 1

            if cached_results is None:
                # Evaluate with VectorBT-optimized processing
                if self.vectorization_manager and self.rolling_optimizer:
                    gate_results = self._evaluate_gates_vectorized(features_df, targets)
                elif needs_streaming and self.streaming_processor:
                    gate_results = self._evaluate_gates_streaming(features_df, targets)
                else:
                    gate_results = self.gate_manager.evaluate_gate_features(features_df, targets)

                # Cache the results
                if self.feature_cache:
                    self.feature_cache.set(cache_key, gate_results, ttl_seconds=3600)

            # Track performance
            processing_time = (datetime.now() - start_time).total_seconds()
            self.performance_metrics['total_processing_time'] += processing_time
            
            # Track VectorBT optimization metrics
            if self.vectorization_manager:
                vectorbt_stats = self.vectorization_manager.get_performance_stats()
                self.performance_metrics['vectorbt_operations'] = vectorbt_stats.get('vectorbt_operations', 0)
                self.performance_metrics['vectorbt_usage_rate'] = vectorbt_stats.get('vectorbt_usage_rate', 0)
                self.performance_metrics['memory_optimizations'] = vectorbt_stats.get('memory_optimizations', 0)
                self.performance_metrics['cache_hit_rate'] = vectorbt_stats.get('cache_hit_rate', 0)

            # Check gate results and determine success
            success = self._evaluate_gate_results(gate_results, config)

            if not success:
                tprint_warning("⚠️ Gate validation completed with warnings")

            # Generate artifacts
            artifacts = self._generate_artifacts(gate_results, config)

            # Create comprehensive outcome report
            outcome_report = self._create_outcome_report(gate_results, config)

            # Save artifacts
            saved_artifacts = []
            for artifact_name, artifact_data in artifacts.items():
                artifact_path = self._save_artifact(
                    artifact_data,
                    artifact_name,
                    artifact_type="data"
                )
                saved_artifacts.append({
                    'name': artifact_name,
                    'path': artifact_path,
                    'type': 'data'
                })

            # Calculate metrics
            metrics = self._calculate_metrics(gate_results, config)

            # Save outcome report
            report_path = self._save_artifact(
                outcome_report,
                "gate_feature_outcome_report",
                artifact_type="report"
            )

            # Add performance metrics to the result
            metrics.update(self.performance_metrics)

            # Get system status if hardware manager is available
            if self.hardware_manager:
                system_status = self.hardware_manager.get_system_status()
                metrics['system_status'] = system_status

            execution_result = {
                'success': success,
                'artifacts': saved_artifacts,
                'metrics': metrics,
                'gate_results': [result.__dict__ for result in gate_results],
                'outcome_report_path': report_path,
                'execution_time': 0.0,  # Will be set by base class
                'performance_metrics': self.performance_metrics
            }

            if success:
                tprint_success(f"✅ {self.step_name} completed successfully")
                if self.performance_metrics['cache_hits'] > 0:
                    tprint_info(f"💾 Cache performance: {self.performance_metrics['cache_hits']} hits, {self.performance_metrics['cache_misses']} misses")
                if self.performance_metrics.get('vectorbt_operations', 0) > 0:
                    tprint_info(f"⚡ VectorBT operations: {self.performance_metrics['vectorbt_operations']}")
                if self.performance_metrics.get('vectorbt_usage_rate', 0) > 0:
                    tprint_info(f"📊 VectorBT usage rate: {self.performance_metrics['vectorbt_usage_rate']:.1%}")
                if self.performance_metrics.get('memory_optimizations', 0) > 0:
                    tprint_info(f"🧠 Memory optimizations: {self.performance_metrics['memory_optimizations']}")
            else:
                tprint_warning(f"⚠️ {self.step_name} completed with warnings")

            return execution_result

        except Exception as e:
            error_msg = f"Gate feature step failed: {str(e)}"
            tprint_error(error_msg)
            logger.error(error_msg, exc_info=True)

            # Cleanup hardware resources
            self._cleanup_hardware_resources()

            return {
                'success': False,
                'error': error_msg,
                'artifacts': [],
                'metrics': self.performance_metrics,
                'execution_time': 0.0
            }

    def _setup_gate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Setup gate feature configuration."""
        default_config = {
            # Gate configuration
            'enable_gate_protection': True,
            'max_gate_features_per_base': 3,
            'min_gate_ic_improvement': 0.005,
            'min_gate_stability': 0.4,
            'max_nan_ratio': 0.3,
            'min_variance_threshold': 1e-8,
            'max_correlation_threshold': 0.95,
            'min_data_points': 100,
            'min_ic_threshold': 0.01,
            'max_ic_decay': 0.5,
            'min_sharpe_ratio': 0.5,
            'enable_feature_importance_gates': True,
            'enable_correlation_gates': True,
            'enable_variance_gates': True,
            'enable_outlier_gates': True,
            'enable_gate_monitoring': True,
            'gate_monitoring_frequency': 100,
            'enable_gate_reporting': True,
            'enable_gate_persistence': True,
            'gate_state_file': f"gate_feature_state_{config.get('symbol', 'unknown')}.json"
        }

        # Override with user configuration if provided
        if 'gate_config' in config:
            default_config.update(config['gate_config'])

        return default_config

    def _evaluate_gate_results(self, gate_results: List[GateFeatureResult], config: Dict[str, Any]) -> bool:
        """Evaluate gate results and determine overall success."""
        if not gate_results:
            return True  # No gates to evaluate

        failed_gates = [r for r in gate_results if r.status == GateStatus.FAILED]
        warning_gates = [r for r in gate_results if r.status == GateStatus.WARNING]

        if failed_gates:
            tprint_error(f"❌ {len(failed_gates)} gate(s) failed:")
            for gate in failed_gates:
                tprint_error(f"  - {gate.feature_name} ({gate.gate_type.value}): {gate.message}")
            return False

        if warning_gates:
            tprint_warning(f"⚠️ {len(warning_gates)} gate(s) have warnings:")
            for gate in warning_gates:
                tprint_warning(f"  - {gate.feature_name} ({gate.gate_type.value}): {gate.message}")

        # Check if execution mode allows continuing with warnings
        execution_mode = config.get('execution_mode', 'light')
        if execution_mode == 'strict' and warning_gates:
            tprint_error("❌ Strict mode: Cannot continue with gate warnings")
            return False

        return True

    def _generate_artifacts(self, gate_results: List[GateFeatureResult], config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate artifacts from gate evaluation results."""
        artifacts = {}

        # Gate results summary
        gate_summary = {
            'total_gates': len(gate_results),
            'passed_gates': len([r for r in gate_results if r.status.value == 'passed']),
            'failed_gates': len([r for r in gate_results if r.status.value == 'failed']),
            'warning_gates': len([r for r in gate_results if r.status.value == 'warning']),
            'gate_details': [result.__dict__ for result in gate_results],
            'timestamp': datetime.now().isoformat(),
            'symbol': config.get('symbol', 'unknown'),
            'execution_mode': config.get('execution_mode', 'light')
        }
        artifacts['gate_results_summary'] = gate_summary

        # Gate performance metrics
        if gate_results:
            performance_metrics = {
                'avg_gate_score': np.mean([r.score for r in gate_results]),
                'min_gate_score': np.min([r.score for r in gate_results]),
                'max_gate_score': np.max([r.score for r in gate_results]),
                'gate_score_std': np.std([r.score for r in gate_results]),
                'quality_gates_passed': len([r for r in gate_results
                                          if r.gate_type == GateFeatureType.QUALITY_GATE and r.status == GateStatus.PASSED]),
                'correlation_gates_passed': len([r for r in gate_results
                                               if r.gate_type == GateFeatureType.CORRELATION_GATE and r.status == GateStatus.PASSED]),
                'variance_gates_passed': len([r for r in gate_results
                                            if r.gate_type == GateFeatureType.VARIANCE_GATE and r.status == GateStatus.PASSED])
            }
            artifacts['gate_performance_metrics'] = performance_metrics

        # Gate state (if persistence is enabled)
        if self.gate_manager and self.gate_manager.config.enable_gate_persistence:
            gate_state = {
                'active_gates': {k: v.__dict__ for k, v in self.gate_manager.state.active_gates.items()},
                'gate_history': [r.__dict__ for r in self.gate_manager.state.gate_history[-10:]],  # Last 10 results
                'configuration': self.gate_manager.config.__dict__,
                'enabled': self.gate_manager.is_gate_protection_enabled()
            }
            artifacts['gate_state'] = gate_state

        return artifacts

    def _calculate_metrics(self, gate_results: List[GateFeatureResult], config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics for the gate evaluation."""
        metrics = {
            'total_gates_evaluated': len(gate_results),
            'execution_timestamp': datetime.now().isoformat(),
            'symbol': config.get('symbol', 'unknown'),
            'exchange': config.get('exchange', 'binance'),
            'timeframe': config.get('timeframe', '15m'),
            'execution_mode': config.get('execution_mode', 'light')
        }

        if gate_results:
            metrics.update({
                'gates_passed': len([r for r in gate_results if r.status == GateStatus.PASSED]),
                'gates_failed': len([r for r in gate_results if r.status == GateStatus.FAILED]),
                'gates_with_warnings': len([r for r in gate_results if r.status == GateStatus.WARNING]),
                'average_gate_score': float(np.mean([r.score for r in gate_results])),
                'gate_score_standard_deviation': float(np.std([r.score for r in gate_results]))
            })

        return metrics

    def _create_outcome_report(self, gate_results: List[GateFeatureResult], config: Dict[str, Any]) -> str:
        """Create comprehensive outcome report."""
        try:
            report = f"""# Gate Feature Protection Outcome Report

**Execution Details:**
- **Symbol:** {config.get('symbol', 'unknown')}
- **Exchange:** {config.get('exchange', 'binance')}
- **Timeframe:** {config.get('timeframe', '15m')}
- **Execution Mode:** {config.get('execution_mode', 'light')}
- **Timestamp:** {datetime.now().isoformat()}

## Gate Evaluation Summary

**Total Gates Evaluated:** {len(gate_results)}

**Results Breakdown:**
"""

            if gate_results:
                passed = len([r for r in gate_results if r.status == GateStatus.PASSED])
                failed = len([r for r in gate_results if r.status == GateStatus.FAILED])
                warnings = len([r for r in gate_results if r.status == GateStatus.WARNING])

                report += f"- ✅ **Passed:** {passed}\n"
                report += f"- ❌ **Failed:** {failed}\n"
                report += f"- ⚠️ **Warnings:** {warnings}\n"

                if gate_results:
                    avg_score = np.mean([r.score for r in gate_results])
                    report += f"- 📊 **Average Score:** {avg_score:.4f}\n"

                report += "\n## Detailed Gate Results\n"

                for result in gate_results:
                    status_icon = {
                        GateStatus.PASSED: '✅',
                        GateStatus.FAILED: '❌',
                        GateStatus.WARNING: '⚠️',
                        GateStatus.SKIPPED: '⏭️'
                    }.get(result.status, '❓')

                    report += f"\n### {status_icon} {result.gate_type.value.replace('_', ' ').title()}\n"
                    report += f"- **Feature:** {result.feature_name}\n"
                    report += f"- **Score:** {result.score:.4f}\n"
                    report += f"- **Threshold:** {result.threshold:.4f}\n"
                    report += f"- **Status:** {result.status.value.upper()}\n"
                    if result.message:
                        report += f"- **Message:** {result.message}\n"
            else:
                report += "- No gates were evaluated (gate protection may be disabled)\n"

            # Configuration section
            if self.gate_manager:
                report += "\n## Gate Configuration\n"
                gate_status = self.gate_manager.get_gate_status()
                for key, value in gate_status.items():
                    if key != 'configuration':  # Skip nested configuration for brevity
                        report += f"- **{key}:** {value}\n"

                # Add key configuration items
                report += f"- **Max Gate Features Per Base:** {self.gate_manager.config.max_gate_features_per_base}\n"
                report += f"- **Min Gate IC Improvement:** {self.gate_manager.config.min_gate_ic_improvement}\n"
                report += f"- **Min Gate Stability:** {self.gate_manager.config.min_gate_stability}\n"

            report += f"""

## Generated Artifacts
- Gate results summary (JSON)
- Gate performance metrics (JSON)
- Gate state (JSON) - if persistence enabled
- This outcome report (Markdown)

---
*Generated by Feature Generation Gate Feature Step at {datetime.now().isoformat()}*
"""

            return report

        except Exception as e:
            tprint_error(f"⚠️ Failed to create outcome report: {e}")
            return f"# Gate Feature Outcome Report\n\nError creating report: {str(e)}"

    def _initialize_vectorbt_optimization(self, config: Dict[str, Any]):
        """Initialize VectorBT optimization components."""
        try:
            tprint_info("🚀 Initializing VectorBT optimization components...")
            
            # Setup VectorBT configuration with hardware optimization
            self.vectorization_config = VectorizationConfig(
                enable_vectorbt=True,
                enable_gpu=config.get('enable_gpu', False),
                enable_parallel=True,
                enable_hardware_optimization=True,
                workload_type=WorkloadType.FEATURE_ENGINEERING if HARDWARE_AVAILABLE else None,
                optimization_level=OptimizationLevel.BALANCED if HARDWARE_AVAILABLE else None,
                memory_efficient=True,
                max_memory_gb=config.get('max_memory_gb', 8.0),
                chunk_size=config.get('chunk_size', 1000),
                enable_monitoring=True,
                enable_profiling=config.get('enable_profiling', False),
                batch_size=config.get('batch_size', 10000),
                enable_batch_processing=True,
                rolling_optimization_threshold=config.get('rolling_optimization_threshold', 1000),
                enable_rolling_optimization=True
            )
            
            # Initialize unified vectorization manager
            self.vectorization_manager = get_unified_vectorization_manager(self.vectorization_config)
            
            # Initialize rolling optimizer with hardware optimization
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=self.vectorization_config.enable_gpu,
                enable_parallel=self.vectorization_config.enable_parallel,
                memory_efficient=self.vectorization_config.memory_efficient,
                chunk_size=self.vectorization_config.chunk_size,
                fast_fail=True,
                enable_logging=True,
                enable_hardware_optimization=True,
                workload_type=WorkloadType.FEATURE_ENGINEERING
            )
            
            tprint_success("✅ VectorBT optimization components initialized")
            
        except Exception as e:
            tprint_warning(f"⚠️ VectorBT optimization initialization failed: {e}")
            self.vectorization_manager = None
            self.rolling_optimizer = None

    def _evaluate_gates_vectorized(self, features_df: pd.DataFrame, targets: pd.Series) -> List[GateFeatureResult]:
        """Evaluate gate features using VectorBT-optimized operations."""
        try:
            tprint_info("⚡ Evaluating gates with VectorBT optimization...")
            
            gate_results = []
            
            # Use hardware optimization context
            with self.vectorization_manager.hardware_optimization_context(
                WorkloadType.FEATURE_ENGINEERING if HARDWARE_AVAILABLE else None
            ) if self.vectorization_manager else nullcontext():
                
                # Optimize DataFrame for VectorBT processing
                if self.vectorization_manager:
                    optimized_df = self.vectorization_manager.optimize_dataframe(features_df)
                else:
                    optimized_df = features_df
            
            # Quality gates with VectorBT rolling operations
            quality_gates = self._evaluate_quality_gates_vectorized(optimized_df, targets)
            gate_results.extend(quality_gates)
            
            # Correlation gates with VectorBT correlation functions
            correlation_gates = self._evaluate_correlation_gates_vectorized(optimized_df, targets)
            gate_results.extend(correlation_gates)
            
            # Variance gates with VectorBT variance calculations
            variance_gates = self._evaluate_variance_gates_vectorized(optimized_df, targets)
            gate_results.extend(variance_gates)
            
            # Outlier gates with VectorBT outlier detection
            outlier_gates = self._evaluate_outlier_gates_vectorized(optimized_df, targets)
            gate_results.extend(outlier_gates)
            
            tprint_success(f"✅ VectorBT gate evaluation completed: {len(gate_results)} gates evaluated")
            return gate_results
            
        except Exception as e:
            tprint_error(f"❌ VectorBT gate evaluation failed: {e}")
            # Fallback to standard gate evaluation
            return self.gate_manager.evaluate_gate_features(features_df, targets)

    def _evaluate_quality_gates_vectorized(self, features_df: pd.DataFrame, targets: pd.Series) -> List[GateFeatureResult]:
        """Evaluate quality gates using VectorBT rolling operations."""
        results = []
        
        try:
            # Use VectorBT rolling operations for statistical calculations
            for column in features_df.columns:
                if self.rolling_optimizer:
                    # Rolling mean and std for quality assessment
                    rolling_mean = self.rolling_optimizer.rolling_mean(features_df[column], window=20)
                    rolling_std = self.rolling_optimizer.rolling_std(features_df[column], window=20)
                    
                    # Calculate quality score based on stability
                    stability_score = 1.0 - (rolling_std / (rolling_mean.abs() + 1e-8)).mean()
                    
                    # Create quality gate result
                    from src.training.steps.pre_training.gate_feature_integration import GateFeatureResult, GateStatus, GateFeatureType
                    
                    status = GateStatus.PASSED if stability_score > 0.4 else GateStatus.WARNING
                    result = GateFeatureResult(
                        feature_name=f"{column}_quality",
                        gate_type=GateFeatureType.QUALITY_GATE,
                        score=stability_score,
                        threshold=0.4,
                        status=status,
                        message=f"Quality gate: stability={stability_score:.4f}"
                    )
                    results.append(result)
                    
        except Exception as e:
            tprint_warning(f"⚠️ Quality gate evaluation failed: {e}")
            
        return results

    def _evaluate_correlation_gates_vectorized(self, features_df: pd.DataFrame, targets: pd.Series) -> List[GateFeatureResult]:
        """Evaluate correlation gates using VectorBT correlation functions."""
        results = []
        
        try:
            # Use VectorBT for correlation calculations
            for column in features_df.columns:
                if self.rolling_optimizer:
                    # Rolling correlation with targets
                    rolling_corr = self.rolling_optimizer.rolling_corr(
                        features_df[column], targets, window=20
                    )
                    
                    # Calculate correlation stability
                    corr_stability = 1.0 - rolling_corr.std()
                    avg_correlation = rolling_corr.mean()
                    
                    # Create correlation gate result
                    from src.training.steps.pre_training.gate_feature_integration import GateFeatureResult, GateStatus, GateFeatureType
                    
                    status = GateStatus.PASSED if corr_stability > 0.3 else GateStatus.WARNING
                    result = GateFeatureResult(
                        feature_name=f"{column}_correlation",
                        gate_type=GateFeatureType.CORRELATION_GATE,
                        score=corr_stability,
                        threshold=0.3,
                        status=status,
                        message=f"Correlation gate: stability={corr_stability:.4f}, avg_corr={avg_correlation:.4f}"
                    )
                    results.append(result)
                    
        except Exception as e:
            tprint_warning(f"⚠️ Correlation gate evaluation failed: {e}")
            
        return results

    def _evaluate_variance_gates_vectorized(self, features_df: pd.DataFrame, targets: pd.Series) -> List[GateFeatureResult]:
        """Evaluate variance gates using VectorBT variance calculations."""
        results = []
        
        try:
            # Use VectorBT for variance calculations
            for column in features_df.columns:
                if self.rolling_optimizer:
                    # Rolling variance
                    rolling_var = self.rolling_optimizer.rolling_var(features_df[column], window=20)
                    
                    # Calculate variance stability
                    var_stability = 1.0 - (rolling_var.std() / (rolling_var.mean() + 1e-8))
                    
                    # Create variance gate result
                    from src.training.steps.pre_training.gate_feature_integration import GateFeatureResult, GateStatus, GateFeatureType
                    
                    status = GateStatus.PASSED if var_stability > 0.5 else GateStatus.WARNING
                    result = GateFeatureResult(
                        feature_name=f"{column}_variance",
                        gate_type=GateFeatureType.VARIANCE_GATE,
                        score=var_stability,
                        threshold=0.5,
                        status=status,
                        message=f"Variance gate: stability={var_stability:.4f}"
                    )
                    results.append(result)
                    
        except Exception as e:
            tprint_warning(f"⚠️ Variance gate evaluation failed: {e}")
            
        return results

    def _evaluate_outlier_gates_vectorized(self, features_df: pd.DataFrame, targets: pd.Series) -> List[GateFeatureResult]:
        """Evaluate outlier gates using VectorBT outlier detection."""
        results = []
        
        try:
            # Use VectorBT for outlier detection
            for column in features_df.columns:
                if self.rolling_optimizer:
                    # Rolling mean and std for outlier detection
                    rolling_mean = self.rolling_optimizer.rolling_mean(features_df[column], window=20)
                    rolling_std = self.rolling_optimizer.rolling_std(features_df[column], window=20)
                    
                    # Calculate outlier ratio
                    z_scores = (features_df[column] - rolling_mean) / (rolling_std + 1e-8)
                    outlier_ratio = (z_scores.abs() > 3).mean()
                    
                    # Create outlier gate result
                    from src.training.steps.pre_training.gate_feature_integration import GateFeatureResult, GateStatus, GateFeatureType
                    
                    status = GateStatus.PASSED if outlier_ratio < 0.05 else GateStatus.WARNING
                    result = GateFeatureResult(
                        feature_name=f"{column}_outliers",
                        gate_type=GateFeatureType.OUTLIER_GATE,
                        score=1.0 - outlier_ratio,
                        threshold=0.95,
                        status=status,
                        message=f"Outlier gate: ratio={outlier_ratio:.4f}"
                    )
                    results.append(result)
                    
        except Exception as e:
            tprint_warning(f"⚠️ Outlier gate evaluation failed: {e}")
            
        return results


# Register the step
def register_feature_generation_gate_feature_step():
    """Register the feature generation gate feature step."""
    from src.training.steps.base_step import step_registry

    step_registry.register("feature_generation_gate_feature_step", FeatureGenerationGateFeatureStep)
    tprint("✅ Feature generation gate feature step registered", "SUCCESS")


# Auto-register when module is imported
register_feature_generation_gate_feature_step()







