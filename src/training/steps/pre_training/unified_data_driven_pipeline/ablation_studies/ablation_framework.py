"""
Comprehensive Ablation Study Framework

Implements systematic ablation studies to validate the contribution of different
pipeline components and configurations.

Key Features:
- Systematic ablation of pipeline components
- Statistical significance testing
- Performance delta calculations
- Comprehensive reporting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import logging
import time
from enum import Enum
from pathlib import Path
import json
import warnings
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon, mannwhitneyu

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

logger = logging.getLogger(__name__)


class AblationType(Enum):
    """Types of ablation studies."""
    COMPONENT = "component"
    CONFIGURATION = "configuration"
    STATISTICAL = "statistical"
    PERFORMANCE = "performance"


class StatisticalTest(Enum):
    """Statistical tests for ablation studies."""
    TTEST_PAIRED = "ttest_paired"
    WILCOXON = "wilcoxon"
    MANNWHITNEY = "mannwhitney"
    BOOTSTRAP = "bootstrap"


@dataclass
class AblationStudyConfig:
    """Configuration for ablation studies."""
    
    # Study parameters
    study_name: str = "ablation_study"
    random_seed: int = 42
    n_repeats: int = 5  # Number of repeats for statistical robustness
    
    # Ablation components to test
    enable_moea: bool = True
    enable_diversity_penalty: bool = True
    enable_htf_features: bool = True
    enable_embargo: bool = True
    enable_turnover_objective: bool = True
    enable_stability_objective: bool = True
    enable_lightweight_screening: bool = True
    enable_hereditary_interactions: bool = True
    enable_advanced_screening: bool = True
    
    # Metrics to track
    track_metrics: List[str] = field(default_factory=lambda: [
        'oos_sharpe', 'max_drawdown', 'turnover', 'stability',
        'diversity', 'mutual_information', 'profit_centered',
        'feature_count', 'processing_time', 'memory_usage'
    ])
    
    # Statistical testing
    statistical_tests: List[StatisticalTest] = field(default_factory=lambda: [
        StatisticalTest.TTEST_PAIRED,
        StatisticalTest.WILCOXON,
        StatisticalTest.BOOTSTRAP
    ])
    significance_level: float = 0.05
    bootstrap_samples: int = 1000
    
    # Output settings
    save_results: bool = True
    output_dir: Optional[str] = None
    generate_plots: bool = True
    generate_report: bool = True


@dataclass
class AblationResult:
    """Result from a single ablation configuration."""
    
    # Configuration
    config_name: str
    config_params: Dict[str, Any]
    
    # Performance metrics
    metrics: Dict[str, float]
    metrics_std: Dict[str, float] = field(default_factory=dict)
    
    # Statistical significance
    is_significant: bool = False
    p_value: float = 1.0
    effect_size: float = 0.0
    
    # Metadata
    processing_time: float = 0.0
    memory_usage_mb: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    
    # Raw results for statistical analysis
    raw_metrics: List[Dict[str, float]] = field(default_factory=list)


@dataclass
class AblationDelta:
    """Delta between baseline and ablation result."""
    
    metric_name: str
    baseline_value: float
    ablation_value: float
    delta_absolute: float
    delta_relative: float
    delta_percentage: float
    
    # Statistical significance
    is_significant: bool = False
    p_value: float = 1.0
    effect_size: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)


@dataclass
class AblationReport:
    """Comprehensive ablation study report."""
    
    # Study metadata
    study_name: str
    study_config: AblationStudyConfig
    timestamp: str
    
    # Results
    baseline_result: AblationResult
    ablation_results: Dict[str, AblationResult]
    deltas: Dict[str, List[AblationDelta]]
    
    # Statistical summary
    significant_ablation: List[str]
    effect_sizes: Dict[str, float]
    
    # Performance summary
    performance_summary: Dict[str, Any]
    
    # Recommendations
    recommendations: List[str]


class AblationStudyFramework:
    """
    Comprehensive framework for conducting ablation studies.
    
    This class provides systematic ablation of pipeline components to validate
    their contribution to overall performance.
    """
    
    def __init__(self, config: Optional[AblationStudyConfig] = None):
        """
        Initialize the ablation study framework.
        
        Args:
            config: Configuration for the ablation study
        """
        self.config = config or AblationStudyConfig()
        self.logger = logger
        
        # Set random seed for reproducibility
        np.random.seed(self.config.random_seed)
        
        # Initialize results storage
        self.results = {}
        self.deltas = {}
        
        tprint_info("🔬 Ablation Study Framework initialized")
        tprint_debug(f"📊 Study: {self.config.study_name}")
        tprint_debug(f"📊 Repeats: {self.config.n_repeats}")
        tprint_debug(f"📊 Metrics: {len(self.config.track_metrics)}")
    
    def run_ablation_study(self, 
                          data: pd.DataFrame, 
                          targets: pd.Series,
                          pipeline_factory: callable) -> AblationReport:
        """
        Run comprehensive ablation study.
        
        Args:
            data: Input data for the study
            targets: Target variable
            pipeline_factory: Factory function to create pipeline instances
            
        Returns:
            AblationReport with comprehensive results
        """
        start_time = time.time()
        
        tprint_info("🔬 Starting comprehensive ablation study...")
        tprint_debug(f"📊 Data shape: {data.shape}")
        tprint_debug(f"📊 Target length: {len(targets)}")
        
        try:
            # Step 1: Run baseline configuration
            tprint_info("Step 1: Running baseline configuration...")
            baseline_result = self._run_baseline(data, targets, pipeline_factory)
            
            # Step 2: Run ablation configurations
            tprint_info("Step 2: Running ablation configurations...")
            ablation_results = self._run_ablation_configurations(
                data, targets, pipeline_factory, baseline_result
            )
            
            # Step 3: Calculate deltas and statistical significance
            tprint_info("Step 3: Calculating deltas and statistical significance...")
            deltas = self._calculate_deltas(baseline_result, ablation_results)
            
            # Step 4: Generate comprehensive report
            tprint_info("Step 4: Generating comprehensive report...")
            report = self._generate_report(baseline_result, ablation_results, deltas)
            
            # Step 5: Save results if requested
            if self.config.save_results:
                self._save_results(report)
            
            total_time = time.time() - start_time
            tprint_success(f"✅ Ablation study completed in {total_time:.2f}s")
            tprint_info(f"📊 Baseline: {baseline_result.config_name}")
            tprint_info(f"📊 Ablations: {len(ablation_results)}")
            tprint_info(f"📊 Significant: {len(report.significant_ablation)}")
            
            return report
            
        except Exception as e:
            tprint_error(f"❌ Ablation study failed: {e}")
            raise
    
    def _run_baseline(self, 
                     data: pd.DataFrame, 
                     targets: pd.Series,
                     pipeline_factory: callable) -> AblationResult:
        """Run baseline configuration with all features enabled."""
        tprint_debug("Running baseline configuration...")
        
        # Create baseline configuration
        baseline_config = self._create_baseline_config()
        
        # Run multiple times for statistical robustness
        raw_results = []
        processing_times = []
        memory_usage = []
        
        for repeat in range(self.config.n_repeats):
            try:
                # Create pipeline with baseline config
                pipeline = pipeline_factory(baseline_config)
                
                # Run pipeline
                start_time = time.time()
                result = pipeline.process(data, targets)
                processing_time = time.time() - start_time
                
                # Extract metrics
                metrics = self._extract_metrics(result)
                raw_results.append(metrics)
                processing_times.append(processing_time)
                
                # Estimate memory usage (simplified)
                memory_usage.append(self._estimate_memory_usage(data, result))
                
            except Exception as e:
                tprint_warning(f"⚠️ Baseline repeat {repeat} failed: {e}")
                continue
        
        if not raw_results:
            raise RuntimeError("All baseline runs failed")
        
        # Calculate statistics
        metrics = self._calculate_metric_statistics(raw_results)
        metrics_std = self._calculate_metric_std(raw_results)
        
        return AblationResult(
            config_name="baseline",
            config_params=baseline_config,
            metrics=metrics,
            metrics_std=metrics_std,
            processing_time=np.mean(processing_times),
            memory_usage_mb=np.mean(memory_usage),
            raw_metrics=raw_results,
            success=True
        )
    
    def _run_ablation_configurations(self, 
                                   data: pd.DataFrame, 
                                   targets: pd.Series,
                                   pipeline_factory: callable,
                                   baseline_result: AblationResult) -> Dict[str, AblationResult]:
        """Run all ablation configurations."""
        ablation_configs = self._create_ablation_configurations()
        results = {}
        
        for config_name, config_params in ablation_configs.items():
            tprint_debug(f"Running ablation: {config_name}")
            
            try:
                # Run multiple times for statistical robustness
                raw_results = []
                processing_times = []
                memory_usage = []
                
                for repeat in range(self.config.n_repeats):
                    try:
                        # Create pipeline with ablation config
                        pipeline = pipeline_factory(config_params)
                        
                        # Run pipeline
                        start_time = time.time()
                        result = pipeline.process(data, targets)
                        processing_time = time.time() - start_time
                        
                        # Extract metrics
                        metrics = self._extract_metrics(result)
                        raw_results.append(metrics)
                        processing_times.append(processing_time)
                        
                        # Estimate memory usage
                        memory_usage.append(self._estimate_memory_usage(data, result))
                        
                    except Exception as e:
                        tprint_warning(f"⚠️ Ablation {config_name} repeat {repeat} failed: {e}")
                        continue
                
                if not raw_results:
                    tprint_warning(f"⚠️ All runs failed for ablation: {config_name}")
                    continue
                
                # Calculate statistics
                metrics = self._calculate_metric_statistics(raw_results)
                metrics_std = self._calculate_metric_std(raw_results)
                
                results[config_name] = AblationResult(
                    config_name=config_name,
                    config_params=config_params,
                    metrics=metrics,
                    metrics_std=metrics_std,
                    processing_time=np.mean(processing_times),
                    memory_usage_mb=np.mean(memory_usage),
                    raw_metrics=raw_results,
                    success=True
                )
                
                tprint_success(f"✅ Ablation {config_name} completed")
                
            except Exception as e:
                tprint_error(f"❌ Ablation {config_name} failed: {e}")
                results[config_name] = AblationResult(
                    config_name=config_name,
                    config_params=config_params,
                    metrics={},
                    success=False,
                    error_message=str(e)
                )
        
        return results
    
    def _create_baseline_config(self) -> Dict[str, Any]:
        """Create baseline configuration with all features enabled."""
        return {
            'enable_moea': True,
            'enable_diversity_penalty': True,
            'enable_htf_features': True,
            'enable_embargo': True,
            'enable_turnover_objective': True,
            'enable_stability_objective': True,
            'enable_lightweight_screening': True,
            'enable_hereditary_interactions': True,
            'enable_advanced_screening': True
        }
    
    def _create_ablation_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Create ablation configurations for testing."""
        configs = {}
        
        # Ablation 1: Without MOEA (greedy single-objective)
        if self.config.enable_moea:
            configs['no_moea'] = {
                'enable_moea': False,
                'enable_diversity_penalty': True,
                'enable_htf_features': True,
                'enable_embargo': True,
                'enable_turnover_objective': True,
                'enable_stability_objective': True,
                'enable_lightweight_screening': True,
                'enable_hereditary_interactions': True,
                'enable_advanced_screening': True
            }
        
        # Ablation 2: Without diversity penalty
        if self.config.enable_diversity_penalty:
            configs['no_diversity_penalty'] = {
                'enable_moea': True,
                'enable_diversity_penalty': False,
                'enable_htf_features': True,
                'enable_embargo': True,
                'enable_turnover_objective': True,
                'enable_stability_objective': True,
                'enable_lightweight_screening': True,
                'enable_hereditary_interactions': True,
                'enable_advanced_screening': True
            }
        
        # Ablation 3: Without HTF features
        if self.config.enable_htf_features:
            configs['no_htf_features'] = {
                'enable_moea': True,
                'enable_diversity_penalty': True,
                'enable_htf_features': False,
                'enable_embargo': True,
                'enable_turnover_objective': True,
                'enable_stability_objective': True,
                'enable_lightweight_screening': True,
                'enable_hereditary_interactions': True,
                'enable_advanced_screening': True
            }
        
        # Ablation 4: Without embargo
        if self.config.enable_embargo:
            configs['no_embargo'] = {
                'enable_moea': True,
                'enable_diversity_penalty': True,
                'enable_htf_features': True,
                'enable_embargo': False,
                'enable_turnover_objective': True,
                'enable_stability_objective': True,
                'enable_lightweight_screening': True,
                'enable_hereditary_interactions': True,
                'enable_advanced_screening': True
            }
        
        # Ablation 5: Without turnover objective
        if self.config.enable_turnover_objective:
            configs['no_turnover_objective'] = {
                'enable_moea': True,
                'enable_diversity_penalty': True,
                'enable_htf_features': True,
                'enable_embargo': True,
                'enable_turnover_objective': False,
                'enable_stability_objective': True,
                'enable_lightweight_screening': True,
                'enable_hereditary_interactions': True,
                'enable_advanced_screening': True
            }
        
        # Ablation 6: Without stability objective
        if self.config.enable_stability_objective:
            configs['no_stability_objective'] = {
                'enable_moea': True,
                'enable_diversity_penalty': True,
                'enable_htf_features': True,
                'enable_embargo': True,
                'enable_turnover_objective': True,
                'enable_stability_objective': False,
                'enable_lightweight_screening': True,
                'enable_hereditary_interactions': True,
                'enable_advanced_screening': True
            }
        
        # Ablation 7: Without lightweight screening
        if self.config.enable_lightweight_screening:
            configs['no_lightweight_screening'] = {
                'enable_moea': True,
                'enable_diversity_penalty': True,
                'enable_htf_features': True,
                'enable_embargo': True,
                'enable_turnover_objective': True,
                'enable_stability_objective': True,
                'enable_lightweight_screening': False,
                'enable_hereditary_interactions': True,
                'enable_advanced_screening': True
            }
        
        # Ablation 8: Without hereditary interactions
        if self.config.enable_hereditary_interactions:
            configs['no_hereditary_interactions'] = {
                'enable_moea': True,
                'enable_diversity_penalty': True,
                'enable_htf_features': True,
                'enable_embargo': True,
                'enable_turnover_objective': True,
                'enable_stability_objective': True,
                'enable_lightweight_screening': True,
                'enable_hereditary_interactions': False,
                'enable_advanced_screening': True
            }
        
        # Ablation 9: Without advanced screening
        if self.config.enable_advanced_screening:
            configs['no_advanced_screening'] = {
                'enable_moea': True,
                'enable_diversity_penalty': True,
                'enable_htf_features': True,
                'enable_embargo': True,
                'enable_turnover_objective': True,
                'enable_stability_objective': True,
                'enable_lightweight_screening': True,
                'enable_hereditary_interactions': True,
                'enable_advanced_screening': False
            }
        
        return configs
    
    def _extract_metrics(self, result: Any) -> Dict[str, float]:
        """Extract metrics from pipeline result."""
        metrics = {}
        
        # Extract available metrics from result object
        if hasattr(result, 'selected_features'):
            metrics['feature_count'] = len(result.selected_features)
        
        if hasattr(result, 'out_of_sample_sharpe'):
            metrics['oos_sharpe'] = result.out_of_sample_sharpe
        
        if hasattr(result, 'max_drawdown'):
            metrics['max_drawdown'] = result.max_drawdown
        
        if hasattr(result, 'turnover'):
            metrics['turnover'] = result.turnover
        
        if hasattr(result, 'stability'):
            metrics['stability'] = result.stability
        
        if hasattr(result, 'diversity'):
            metrics['diversity'] = result.diversity
        
        if hasattr(result, 'mutual_information'):
            metrics['mutual_information'] = result.mutual_information
        
        if hasattr(result, 'profit_centered'):
            metrics['profit_centered'] = result.profit_centered
        
        if hasattr(result, 'processing_time'):
            metrics['processing_time'] = result.processing_time
        
        if hasattr(result, 'memory_usage_mb'):
            metrics['memory_usage'] = result.memory_usage_mb
        
        # Fill missing metrics with defaults
        for metric in self.config.track_metrics:
            if metric not in metrics:
                metrics[metric] = 0.0
        
        return metrics
    
    def _calculate_metric_statistics(self, raw_results: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate mean statistics from raw results."""
        if not raw_results:
            return {}
        
        metrics = {}
        for metric in self.config.track_metrics:
            values = [result.get(metric, 0.0) for result in raw_results]
            metrics[metric] = np.mean(values) if values else 0.0
        
        return metrics
    
    def _calculate_metric_std(self, raw_results: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate standard deviation from raw results."""
        if not raw_results:
            return {}
        
        metrics_std = {}
        for metric in self.config.track_metrics:
            values = [result.get(metric, 0.0) for result in raw_results]
            metrics_std[metric] = np.std(values) if values else 0.0
        
        return metrics_std
    
    def _estimate_memory_usage(self, data: pd.DataFrame, result: Any) -> float:
        """Estimate memory usage in MB."""
        # Simple estimation based on data size and result complexity
        data_memory = data.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Add estimated result memory
        result_memory = 0.0
        if hasattr(result, 'selected_features'):
            result_memory += len(result.selected_features) * 0.001  # Rough estimate
        
        return data_memory + result_memory
    
    def _calculate_deltas(self, 
                         baseline_result: AblationResult,
                         ablation_results: Dict[str, AblationResult]) -> Dict[str, List[AblationDelta]]:
        """Calculate deltas between baseline and ablation results."""
        deltas = {}
        
        for ablation_name, ablation_result in ablation_results.items():
            if not ablation_result.success:
                continue
            
            ablation_deltas = []
            
            for metric in self.config.track_metrics:
                baseline_value = baseline_result.metrics.get(metric, 0.0)
                ablation_value = ablation_result.metrics.get(metric, 0.0)
                
                # Calculate deltas
                delta_absolute = ablation_value - baseline_value
                delta_relative = delta_absolute / baseline_value if baseline_value != 0 else 0.0
                delta_percentage = delta_relative * 100
                
                # Statistical significance testing
                is_significant, p_value, effect_size, ci = self._test_statistical_significance(
                    baseline_result.raw_metrics, ablation_result.raw_metrics, metric
                )
                
                delta = AblationDelta(
                    metric_name=metric,
                    baseline_value=baseline_value,
                    ablation_value=ablation_value,
                    delta_absolute=delta_absolute,
                    delta_relative=delta_relative,
                    delta_percentage=delta_percentage,
                    is_significant=is_significant,
                    p_value=p_value,
                    effect_size=effect_size,
                    confidence_interval=ci
                )
                
                ablation_deltas.append(delta)
            
            deltas[ablation_name] = ablation_deltas
        
        return deltas
    
    def _test_statistical_significance(self, 
                                     baseline_metrics: List[Dict[str, float]],
                                     ablation_metrics: List[Dict[str, float]],
                                     metric: str) -> Tuple[bool, float, float, Tuple[float, float]]:
        """Test statistical significance of metric differences."""
        try:
            # Extract metric values
            baseline_values = [m.get(metric, 0.0) for m in baseline_metrics]
            ablation_values = [m.get(metric, 0.0) for m in ablation_metrics]
            
            if len(baseline_values) < 2 or len(ablation_values) < 2:
                return False, 1.0, 0.0, (0.0, 0.0)
            
            # Paired t-test
            if len(baseline_values) == len(ablation_values):
                t_stat, p_value = ttest_rel(baseline_values, ablation_values)
            else:
                t_stat, p_value = ttest_rel(baseline_values[:len(ablation_values)], ablation_values)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(baseline_values) + np.var(ablation_values)) / 2)
            effect_size = (np.mean(ablation_values) - np.mean(baseline_values)) / pooled_std if pooled_std > 0 else 0.0
            
            # Confidence interval (simplified)
            mean_diff = np.mean(ablation_values) - np.mean(baseline_values)
            std_diff = np.std(np.array(ablation_values) - np.array(baseline_values))
            ci = (mean_diff - 1.96 * std_diff, mean_diff + 1.96 * std_diff)
            
            is_significant = p_value < self.config.significance_level
            
            return is_significant, p_value, effect_size, ci
            
        except Exception as e:
            self.logger.warning(f"Statistical test failed for metric {metric}: {e}")
            return False, 1.0, 0.0, (0.0, 0.0)
    
    def _generate_report(self, 
                        baseline_result: AblationResult,
                        ablation_results: Dict[str, AblationResult],
                        deltas: Dict[str, List[AblationDelta]]) -> AblationReport:
        """Generate comprehensive ablation report."""
        
        # Find significant ablations
        significant_ablation = []
        effect_sizes = {}
        
        for ablation_name, ablation_deltas in deltas.items():
            significant_metrics = [d for d in ablation_deltas if d.is_significant]
            if significant_metrics:
                significant_ablation.append(ablation_name)
                effect_sizes[ablation_name] = np.mean([d.effect_size for d in significant_metrics])
        
        # Performance summary
        performance_summary = {
            'baseline_metrics': baseline_result.metrics,
            'ablation_count': len(ablation_results),
            'significant_count': len(significant_ablation),
            'total_metrics_tested': len(self.config.track_metrics),
            'study_duration': time.time()
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(deltas, significant_ablation)
        
        return AblationReport(
            study_name=self.config.study_name,
            study_config=self.config,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            baseline_result=baseline_result,
            ablation_results=ablation_results,
            deltas=deltas,
            significant_ablation=significant_ablation,
            effect_sizes=effect_sizes,
            performance_summary=performance_summary,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, 
                                 deltas: Dict[str, List[AblationDelta]],
                                 significant_ablation: List[str]) -> List[str]:
        """Generate recommendations based on ablation results."""
        recommendations = []
        
        # Analyze significant ablations
        for ablation_name in significant_ablation:
            ablation_deltas = deltas[ablation_name]
            
            # Find metrics with largest effects
            largest_effects = sorted(ablation_deltas, key=lambda x: abs(x.effect_size), reverse=True)[:3]
            
            for delta in largest_effects:
                if delta.is_significant and abs(delta.effect_size) > 0.5:  # Large effect
                    if delta.delta_percentage > 0:
                        recommendations.append(
                            f"Consider enabling {ablation_name.replace('no_', '')} - "
                            f"improves {delta.metric_name} by {delta.delta_percentage:.1f}%"
                        )
                    else:
                        recommendations.append(
                            f"Consider disabling {ablation_name.replace('no_', '')} - "
                            f"degrades {delta.metric_name} by {abs(delta.delta_percentage):.1f}%"
                        )
        
        # Add general recommendations
        if not significant_ablation:
            recommendations.append("No significant differences found - current configuration is optimal")
        
        if len(significant_ablation) > len(deltas) * 0.5:
            recommendations.append("Many components show significant effects - consider systematic optimization")
        
        return recommendations
    
    def _save_results(self, report: AblationReport) -> None:
        """Save ablation study results."""
        if not self.config.output_dir:
            output_dir = Path(f"ablation_results/{self.config.study_name}")
        else:
            output_dir = Path(self.config.output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save report as JSON
        report_file = output_dir / f"{self.config.study_name}_report.json"
        
        # Convert report to serializable format
        report_dict = {
            'study_name': report.study_name,
            'timestamp': report.timestamp,
            'baseline_result': {
                'config_name': report.baseline_result.config_name,
                'metrics': report.baseline_result.metrics,
                'success': report.baseline_result.success
            },
            'ablation_results': {
                name: {
                    'config_name': result.config_name,
                    'metrics': result.metrics,
                    'success': result.success
                } for name, result in report.ablation_results.items()
            },
            'significant_ablation': report.significant_ablation,
            'effect_sizes': report.effect_sizes,
            'recommendations': report.recommendations
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        tprint_success(f"✅ Results saved to {report_file}")


# Convenience functions
def run_ablation_study(data: pd.DataFrame, 
                      targets: pd.Series,
                      pipeline_factory: callable,
                      config: Optional[AblationStudyConfig] = None) -> AblationReport:
    """
    Convenience function to run ablation study.
    
    Args:
        data: Input data
        targets: Target variable
        pipeline_factory: Factory function to create pipeline instances
        config: Ablation study configuration
        
    Returns:
        AblationReport with results
    """
    framework = AblationStudyFramework(config)
    return framework.run_ablation_study(data, targets, pipeline_factory)


# Export main classes and functions
__all__ = [
    'AblationStudyFramework',
    'AblationStudyConfig',
    'AblationResult',
    'AblationDelta',
    'AblationReport',
    'AblationType',
    'StatisticalTest',
    'run_ablation_study'
]