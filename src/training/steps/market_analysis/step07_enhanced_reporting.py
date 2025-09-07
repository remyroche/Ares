"""
Enhanced Reporting System for Step 7 Matrix Operations

This module provides comprehensive reporting capabilities for step07_enhanced_matrix_operations
with detailed metrics, performance analytics, matrix quality assessments,
and computational efficiency analysis.
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import warnings

# Avoid circular import - import these functions when needed
# from src.training.reports import save_training_report, CentralizedReportManager
from src.utils.logger import system_logger

logger = system_logger.getChild('Step07EnhancedReporting')


@dataclass
class MatrixOperationMetrics:
    """Comprehensive metrics for matrix operations."""
    matrix_dimensions: Tuple[int, int]
    matrix_density: float
    matrix_condition_number: float
    matrix_rank: int
    matrix_sparsity_ratio: float
    computation_time_seconds: float
    memory_usage_mb: float
    gpu_acceleration_used: bool
    numba_optimization_used: bool
    parallel_processing_used: bool
    matrix_stability_score: float
    numerical_precision_score: float


@dataclass
class ComputationalPerformanceMetrics:
    """Performance metrics for computational operations."""
    total_operations: int
    operations_per_second: float
    memory_bandwidth_mb_s: float
    cache_hit_rate: float
    floating_point_operations: int
    instructions_per_cycle: float
    branch_misprediction_rate: float
    execution_efficiency_score: float
    optimization_gain_percentage: float
    resource_utilization_score: float


@dataclass
class GPUAccelerationMetrics:
    """Metrics for GPU/MPS acceleration performance."""
    gpu_available: bool
    gpu_memory_used_mb: float
    gpu_utilization_percentage: float
    gpu_kernel_launch_time_ms: float
    gpu_memory_transfer_time_ms: float
    gpu_compute_time_ms: float
    gpu_acceleration_factor: float
    gpu_memory_efficiency_score: float
    gpu_compute_efficiency_score: float


@dataclass
class MatrixQualityAssessment:
    """Quality assessment for matrix operations."""
    matrix_well_conditioned: bool
    numerical_stability_score: float
    computation_accuracy_score: float
    matrix_orthogonality_score: float
    eigenvalue_distribution_score: float
    singular_value_distribution_score: float
    matrix_energy_concentration: float
    noise_to_signal_ratio: float
    quality_warnings: List[str]
    quality_improvements: List[str]


@dataclass
class OptimizationEffectiveness:
    """Analysis of optimization effectiveness."""
    baseline_performance: float
    optimized_performance: float
    performance_improvement_percentage: float
    memory_usage_reduction_percentage: float
    time_complexity_improvement: str
    space_complexity_improvement: str
    scalability_score: float
    optimization_robustness_score: float
    optimization_recommendations: List[str]


@dataclass
class MatrixOperationResults:
    """Results from matrix operations processing."""
    operations_completed: int
    operations_failed: int
    success_rate_percentage: float
    total_computation_time_seconds: float
    average_operation_time_ms: float
    peak_memory_usage_mb: float
    matrix_transformations_applied: List[str]
    computational_methods_used: List[str]
    optimization_techniques_applied: List[str]


class Step07EnhancedReporter:
    """
    Enhanced reporting system for Step 7 Matrix Operations.

    Provides comprehensive metrics including:
    - Matrix operation performance analytics
    - GPU/MPS acceleration effectiveness
    - Computational efficiency assessment
    - Matrix quality and stability analysis
    - Optimization effectiveness evaluation
    - Memory usage and resource utilization
    """

    def __init__(self, output_dir: str = "src/training/reports/step07"):
        """
        Initialize the Step07 enhanced reporter.

        Args:
            output_dir: Directory to save reports and visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = system_logger.getChild('Step07EnhancedReporter')

        # Initialize report manager (avoid circular import)
        try:
            from src.training.reports import CentralizedReportManager
            self.report_manager = CentralizedReportManager()
        except (ImportError, TypeError):
            self.logger.warning("Could not import CentralizedReportManager, using fallback")
            self.report_manager = None

    def generate_comprehensive_report(self,
                                    matrix_results: Dict[str, Any],
                                    performance_data: Dict[str, Any],
                                    computational_metrics: Dict[str, Any],
                                    gpu_metrics: Dict[str, Any],
                                    optimization_results: Dict[str, Any],
                                    symbol: str,
                                    exchange: str,
                                    timeframe: str,
                                    step_type: str = "enhanced_matrix_operations") -> Dict[str, Any]:
        """
        Generate comprehensive report with all metrics and analyses.

        Args:
            matrix_results: Results from matrix operations
            performance_data: Performance metrics during execution
            computational_metrics: Computational efficiency metrics
            gpu_metrics: GPU/MPS acceleration metrics
            optimization_results: Optimization effectiveness results
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe analyzed
            step_type: Type of matrix operations ("enhanced_matrix_operations" or "model_training")

        Returns:
            Comprehensive report dictionary
        """
        try:
            self.logger.info("🔍 Generating comprehensive Step07 (Matrix Operations) report...")

            # Generate all report sections
            report = {
                'metadata': self._generate_metadata(symbol, exchange, timeframe, step_type),
                'matrix_operation_metrics': self._generate_matrix_operation_metrics(matrix_results),
                'computational_performance': self._generate_computational_performance(computational_metrics),
                'gpu_acceleration_analysis': self._generate_gpu_acceleration_analysis(gpu_metrics),
                'matrix_quality_assessment': self._generate_matrix_quality_assessment(matrix_results),
                'optimization_effectiveness': self._generate_optimization_effectiveness(optimization_results),
                'memory_resource_analysis': self._generate_memory_resource_analysis(performance_data),
                'computational_efficiency_insights': self._generate_computational_efficiency_insights(computational_metrics, gpu_metrics),
                'matrix_stability_analysis': self._generate_matrix_stability_analysis(matrix_results),
                'performance_benchmarking': self._generate_performance_benchmarking(performance_data, computational_metrics),
                'optimization_recommendations': self._generate_optimization_recommendations(optimization_results, gpu_metrics),
                'visualization_data': self._generate_visualization_data(matrix_results, performance_data, computational_metrics)
            }

            self.logger.info("✅ Comprehensive Step07 report generated successfully")
            return report

        except Exception as e:
            self.logger.error(f"❌ Failed to generate comprehensive report: {e}")
            # Return minimal report on error
            return {
                'metadata': self._generate_metadata(symbol, exchange, timeframe, step_type),
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _generate_metadata(self, symbol: str, exchange: str, timeframe: str, step_type: str) -> Dict[str, Any]:
        """Generate report metadata."""
        step_descriptions = {
            'enhanced_matrix_operations': 'Enhanced Matrix Operations - Market Analysis',
            'model_training': 'Enhanced Matrix Operations - Model Training'
        }

        return {
            'report_type': f'step07_{step_type}_enhanced',
            'version': '1.0.0',
            'generated_at': datetime.now().isoformat(),
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': timeframe,
            'step_name': 'Step 7',
            'step_type': step_type,
            'description': step_descriptions.get(step_type, 'Enhanced Matrix Operations Analysis')
        }

    def _generate_matrix_operation_metrics(self, matrix_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate matrix operation metrics."""
        try:
            if not matrix_results:
                return {'error': 'No matrix results available'}

            # Extract matrix properties
            matrix_data = matrix_results.get('matrix_data', {})
            if isinstance(matrix_data, dict) and 'shape' in matrix_data:
                shape = matrix_data['shape']
                dimensions = (shape[0], shape[1]) if len(shape) >= 2 else (0, 0)
            else:
                dimensions = (0, 0)

            # Calculate matrix properties
            density = matrix_results.get('density', 0.0)
            condition_number = matrix_results.get('condition_number', float('inf'))
            rank = matrix_results.get('rank', 0)
            sparsity_ratio = 1.0 - density

            metrics = MatrixOperationMetrics(
                matrix_dimensions=dimensions,
                matrix_density=density,
                matrix_condition_number=condition_number,
                matrix_rank=rank,
                matrix_sparsity_ratio=sparsity_ratio,
                computation_time_seconds=matrix_results.get('computation_time', 0.0),
                memory_usage_mb=matrix_results.get('memory_usage', 0.0),
                gpu_acceleration_used=matrix_results.get('gpu_used', False),
                numba_optimization_used=matrix_results.get('numba_used', False),
                parallel_processing_used=matrix_results.get('parallel_used', False),
                matrix_stability_score=self._calculate_matrix_stability_score(condition_number),
                numerical_precision_score=self._calculate_numerical_precision_score(matrix_results)
            )

            return {
                'metrics': asdict(metrics),
                'matrix_properties': self._analyze_matrix_properties(metrics),
                'computational_characteristics': self._analyze_computational_characteristics(metrics)
            }

        except Exception as e:
            self.logger.warning(f"Could not generate matrix operation metrics: {e}")
            return {'error': str(e)}

    def _generate_computational_performance(self, computational_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate computational performance analysis."""
        try:
            metrics = ComputationalPerformanceMetrics(
                total_operations=computational_metrics.get('total_operations', 0),
                operations_per_second=computational_metrics.get('operations_per_second', 0.0),
                memory_bandwidth_mb_s=computational_metrics.get('memory_bandwidth', 0.0),
                cache_hit_rate=computational_metrics.get('cache_hit_rate', 0.0),
                floating_point_operations=computational_metrics.get('flops', 0),
                instructions_per_cycle=computational_metrics.get('ipc', 0.0),
                branch_misprediction_rate=computational_metrics.get('branch_misprediction', 0.0),
                execution_efficiency_score=computational_metrics.get('efficiency_score', 0.0),
                optimization_gain_percentage=computational_metrics.get('optimization_gain', 0.0),
                resource_utilization_score=computational_metrics.get('resource_utilization', 0.0)
            )

            return {
                'metrics': asdict(metrics),
                'performance_analysis': self._analyze_performance_characteristics(metrics),
                'efficiency_assessment': self._assess_computational_efficiency(metrics),
                'bottleneck_identification': self._identify_computational_bottlenecks(metrics)
            }

        except Exception as e:
            self.logger.warning(f"Could not generate computational performance: {e}")
            return {'error': str(e)}

    def _generate_gpu_acceleration_analysis(self, gpu_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate GPU acceleration analysis."""
        try:
            metrics = GPUAccelerationMetrics(
                gpu_available=gpu_metrics.get('gpu_available', False),
                gpu_memory_used_mb=gpu_metrics.get('memory_used', 0.0),
                gpu_utilization_percentage=gpu_metrics.get('utilization', 0.0),
                gpu_kernel_launch_time_ms=gpu_metrics.get('kernel_launch_time', 0.0),
                gpu_memory_transfer_time_ms=gpu_metrics.get('memory_transfer_time', 0.0),
                gpu_compute_time_ms=gpu_metrics.get('compute_time', 0.0),
                gpu_acceleration_factor=gpu_metrics.get('acceleration_factor', 1.0),
                gpu_memory_efficiency_score=gpu_metrics.get('memory_efficiency', 0.0),
                gpu_compute_efficiency_score=gpu_metrics.get('compute_efficiency', 0.0)
            )

            return {
                'metrics': asdict(metrics),
                'acceleration_analysis': self._analyze_gpu_acceleration_effectiveness(metrics),
                'memory_transfer_analysis': self._analyze_gpu_memory_transfers(metrics),
                'kernel_performance_analysis': self._analyze_gpu_kernel_performance(metrics)
            }

        except Exception as e:
            self.logger.warning(f"Could not generate GPU acceleration analysis: {e}")
            return {'error': str(e)}

    def _generate_matrix_quality_assessment(self, matrix_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate matrix quality assessment."""
        try:
            # Calculate quality metrics
            condition_number = matrix_results.get('condition_number', float('inf'))
            well_conditioned = condition_number < 1000  # Threshold for well-conditioned matrix

            # Calculate stability score based on condition number
            stability_score = max(0, 100 - min(condition_number / 10, 100))

            assessment = MatrixQualityAssessment(
                matrix_well_conditioned=well_conditioned,
                numerical_stability_score=stability_score,
                computation_accuracy_score=self._calculate_computation_accuracy(matrix_results),
                matrix_orthogonality_score=matrix_results.get('orthogonality_score', 0.0),
                eigenvalue_distribution_score=self._calculate_eigenvalue_distribution_score(matrix_results),
                singular_value_distribution_score=self._calculate_singular_value_distribution_score(matrix_results),
                matrix_energy_concentration=matrix_results.get('energy_concentration', 0.0),
                noise_to_signal_ratio=matrix_results.get('noise_to_signal_ratio', 0.0),
                quality_warnings=self._identify_matrix_quality_warnings(matrix_results),
                quality_improvements=self._suggest_matrix_quality_improvements(matrix_results)
            )

            return {
                'assessment': asdict(assessment),
                'quality_score': self._calculate_overall_quality_score(assessment),
                'stability_analysis': self._analyze_matrix_stability(assessment),
                'numerical_analysis': self._analyze_numerical_properties(assessment)
            }

        except Exception as e:
            self.logger.warning(f"Could not generate matrix quality assessment: {e}")
            return {'error': str(e)}

    def _generate_optimization_effectiveness(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimization effectiveness analysis."""
        try:
            baseline_perf = optimization_results.get('baseline_performance', 0.0)
            optimized_perf = optimization_results.get('optimized_performance', 0.0)

            if baseline_perf > 0:
                improvement_pct = ((optimized_perf - baseline_perf) / baseline_perf) * 100
            else:
                improvement_pct = 0.0

            effectiveness = OptimizationEffectiveness(
                baseline_performance=baseline_perf,
                optimized_performance=optimized_perf,
                performance_improvement_percentage=improvement_pct,
                memory_usage_reduction_percentage=optimization_results.get('memory_reduction', 0.0),
                time_complexity_improvement=optimization_results.get('time_complexity', 'Unknown'),
                space_complexity_improvement=optimization_results.get('space_complexity', 'Unknown'),
                scalability_score=optimization_results.get('scalability_score', 0.0),
                optimization_robustness_score=optimization_results.get('robustness_score', 0.0),
                optimization_recommendations=optimization_results.get('recommendations', [])
            )

            return {
                'effectiveness': asdict(effectiveness),
                'improvement_analysis': self._analyze_optimization_improvements(effectiveness),
                'scalability_assessment': self._assess_optimization_scalability(effectiveness),
                'robustness_evaluation': self._evaluate_optimization_robustness(effectiveness)
            }

        except Exception as e:
            self.logger.warning(f"Could not generate optimization effectiveness: {e}")
            return {'error': str(e)}

    def _generate_memory_resource_analysis(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate memory and resource utilization analysis."""
        try:
            return {
                'memory_utilization': self._analyze_memory_utilization(performance_data),
                'resource_efficiency': self._analyze_resource_efficiency(performance_data),
                'memory_optimization_effectiveness': self._analyze_memory_optimization(performance_data),
                'resource_allocation_analysis': self._analyze_resource_allocation(performance_data)
            }

        except Exception as e:
            self.logger.warning(f"Could not generate memory resource analysis: {e}")
            return {'error': str(e)}

    def _generate_computational_efficiency_insights(self, computational_metrics: Dict[str, Any], gpu_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate computational efficiency insights."""
        try:
            return {
                'efficiency_analysis': self._analyze_computational_efficiency(computational_metrics),
                'hardware_utilization': self._analyze_hardware_utilization(gpu_metrics),
                'parallelization_effectiveness': self._analyze_parallelization_effectiveness(computational_metrics),
                'optimization_impact': self._analyze_optimization_impact(computational_metrics, gpu_metrics)
            }

        except Exception as e:
            self.logger.warning(f"Could not generate computational efficiency insights: {e}")
            return {'error': str(e)}

    def _generate_matrix_stability_analysis(self, matrix_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate matrix stability analysis."""
        try:
            return {
                'stability_assessment': self._assess_matrix_stability(matrix_results),
                'numerical_robustness': self._analyze_numerical_robustness(matrix_results),
                'conditioning_analysis': self._analyze_matrix_conditioning(matrix_results),
                'perturbation_sensitivity': self._analyze_perturbation_sensitivity(matrix_results)
            }

        except Exception as e:
            self.logger.warning(f"Could not generate matrix stability analysis: {e}")
            return {'error': str(e)}

    def _generate_performance_benchmarking(self, performance_data: Dict[str, Any], computational_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance benchmarking analysis."""
        try:
            return {
                'benchmark_comparison': self._compare_performance_benchmarks(performance_data),
                'efficiency_benchmarks': self._establish_efficiency_benchmarks(computational_metrics),
                'scalability_benchmarks': self._establish_scalability_benchmarks(performance_data),
                'optimization_benchmarks': self._establish_optimization_benchmarks(performance_data, computational_metrics)
            }

        except Exception as e:
            self.logger.warning(f"Could not generate performance benchmarking: {e}")
            return {'error': str(e)}

    def _generate_optimization_recommendations(self, optimization_results: Dict[str, Any], gpu_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimization recommendations."""
        try:
            return {
                'performance_optimizations': self._recommend_performance_optimizations(optimization_results),
                'memory_optimizations': self._recommend_memory_optimizations(optimization_results),
                'hardware_optimizations': self._recommend_hardware_optimizations(gpu_metrics),
                'algorithmic_improvements': self._recommend_algorithmic_improvements(optimization_results),
                'implementation_priority': self._prioritize_implementation_recommendations(optimization_results, gpu_metrics)
            }

        except Exception as e:
            self.logger.warning(f"Could not generate optimization recommendations: {e}")
            return {'error': str(e)}

    def _generate_visualization_data(self, matrix_results: Dict[str, Any], performance_data: Dict[str, Any], computational_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data for visualizations."""
        try:
            viz_data = {
                'matrix_properties_chart': self._prepare_matrix_properties_data(matrix_results),
                'performance_timeline': self._prepare_performance_timeline_data(performance_data),
                'computational_efficiency_chart': self._prepare_computational_efficiency_data(computational_metrics),
                'optimization_comparison': self._prepare_optimization_comparison_data(performance_data),
                'memory_usage_analysis': self._prepare_memory_usage_data(performance_data),
                'gpu_acceleration_comparison': self._prepare_gpu_acceleration_data(matrix_results),
                'matrix_stability_dashboard': self._prepare_matrix_stability_data(matrix_results),
                'resource_utilization_heatmap': self._prepare_resource_utilization_data(performance_data)
            }

            return viz_data

        except Exception as e:
            self.logger.warning(f"Could not generate visualization data: {e}")
            return {'error': str(e)}

    def save_comprehensive_report(self, report: Dict[str, Any], base_filename: str = "step07_enhanced_report") -> Dict[str, str]:
        """
        Save comprehensive report in multiple formats.

        Args:
            report: The comprehensive report dictionary
            base_filename: Base filename for saved files

        Returns:
            Dictionary mapping format types to file paths
        """
        try:
            self.logger.info("💾 Saving comprehensive Step07 report...")

            saved_files = {}
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Save JSON report
            json_path = self._save_json_report(report, timestamp, base_filename)
            saved_files['json'] = str(json_path)

            # Save Markdown report
            md_path = self._save_markdown_report(report, timestamp, base_filename)
            saved_files['markdown'] = str(md_path)

            # Save CSV data
            csv_path = self._save_csv_report(report, timestamp, base_filename)
            saved_files['csv'] = str(csv_path)

            # Generate and save visualizations
            try:
                self._generate_visualizations(report, timestamp, base_filename)
                saved_files['visualizations'] = str(self.output_dir / f"{base_filename}_visualizations_{timestamp}")
            except Exception as e:
                self.logger.warning(f"Could not generate visualizations: {e}")

            # Use centralized report manager if available
            if self.report_manager:
                try:
                    from src.training.reports import save_training_report
                    report_path = save_training_report(
                        report_data=report,
                        step_name="step07",
                        symbol=report.get('metadata', {}).get('symbol', 'unknown'),
                        exchange=report.get('metadata', {}).get('exchange', 'unknown'),
                        timeframe=report.get('metadata', {}).get('timeframe', 'unknown'),
                        report_type="enhanced_matrix_operations_analysis"
                    )
                    saved_files['centralized'] = str(report_path)
                except Exception as e:
                    self.logger.warning(f"Could not save to centralized reports: {e}")

            self.logger.info(f"✅ Step07 enhanced report saved successfully: {saved_files}")
            return saved_files

        except Exception as e:
            self.logger.error(f"❌ Failed to save comprehensive report: {e}")
            return {'error': str(e)}

    # Helper methods for analysis and calculations
    def _calculate_matrix_stability_score(self, condition_number: float) -> float:
        """Calculate matrix stability score based on condition number."""
        if condition_number == 0 or np.isinf(condition_number):
            return 0.0
        # Lower condition number = higher stability
        return max(0, 100 - min(condition_number / 10, 100))

    def _calculate_numerical_precision_score(self, matrix_results: Dict[str, Any]) -> float:
        """Calculate numerical precision score."""
        # Based on various numerical indicators
        precision_indicators = [
            matrix_results.get('numerical_stability', 0.5),
            matrix_results.get('computation_accuracy', 0.5),
            1.0 - matrix_results.get('error_rate', 0.0),
            matrix_results.get('precision_score', 0.5)
        ]
        return np.mean(precision_indicators) * 100

    def _calculate_computation_accuracy(self, matrix_results: Dict[str, Any]) -> float:
        """Calculate computation accuracy score."""
        # Based on error analysis and numerical stability
        error_rate = matrix_results.get('error_rate', 0.0)
        stability = matrix_results.get('numerical_stability', 0.5)
        return (1.0 - error_rate) * stability * 100

    def _calculate_eigenvalue_distribution_score(self, matrix_results: Dict[str, Any]) -> float:
        """Calculate eigenvalue distribution score."""
        eigenvalues = matrix_results.get('eigenvalues', [])
        if not eigenvalues:
            return 50.0

        # Score based on eigenvalue distribution characteristics
        eigen_array = np.array(eigenvalues)
        if len(eigen_array) == 0:
            return 50.0

        # Check for clustering, spread, and conditioning
        eigen_std = np.std(eigen_array)
        eigen_mean = np.mean(eigen_array)

        if eigen_mean == 0:
            return 50.0

        # Coefficient of variation as stability indicator
        cv = eigen_std / abs(eigen_mean)
        return max(0, 100 - cv * 100)

    def _calculate_singular_value_distribution_score(self, matrix_results: Dict[str, Any]) -> float:
        """Calculate singular value distribution score."""
        singular_values = matrix_results.get('singular_values', [])
        if not singular_values:
            return 50.0

        sv_array = np.array(singular_values)
        if len(sv_array) == 0 or sv_array[0] == 0:
            return 50.0

        # Condition number from singular values
        condition_number = sv_array[0] / sv_array[-1] if sv_array[-1] > 0 else float('inf')
        return max(0, 100 - min(condition_number / 10, 100))

    def _identify_matrix_quality_warnings(self, matrix_results: Dict[str, Any]) -> List[str]:
        """Identify matrix quality warnings."""
        warnings = []

        condition_number = matrix_results.get('condition_number', float('inf'))
        if condition_number > 1000:
            warnings.append(f"High condition number ({condition_number:.1f}) indicates ill-conditioned matrix")

        density = matrix_results.get('density', 1.0)
        if density < 0.1:
            warnings.append(f"Very sparse matrix (density: {density:.1%}) may affect computational efficiency")

        rank = matrix_results.get('rank', 0)
        shape = matrix_results.get('shape', [0, 0])
        if len(shape) >= 2 and rank < min(shape):
            warnings.append(f"Matrix is rank-deficient (rank: {rank}, expected: {min(shape)})")

        return warnings

    def _suggest_matrix_quality_improvements(self, matrix_results: Dict[str, Any]) -> List[str]:
        """Suggest matrix quality improvements."""
        improvements = []

        condition_number = matrix_results.get('condition_number', float('inf'))
        if condition_number > 100:
            improvements.append("Consider matrix regularization or preconditioning")

        if matrix_results.get('density', 1.0) < 0.5:
            improvements.append("Consider sparse matrix optimizations")

        if matrix_results.get('numerical_stability', 0.5) < 0.8:
            improvements.append("Implement numerical stabilization techniques")

        return improvements

    def _calculate_overall_quality_score(self, assessment: Any) -> float:
        """Calculate overall matrix quality score."""
        scores = [
            assessment.numerical_stability_score,
            assessment.computation_accuracy_score,
            assessment.matrix_orthogonality_score,
            assessment.eigenvalue_distribution_score,
            assessment.singular_value_distribution_score
        ]

        return np.mean(scores)

    # Additional helper methods would be implemented here
    # These are simplified stubs for the full implementation

    def _analyze_matrix_properties(self, metrics: Any) -> Dict[str, Any]:
        """Analyze matrix properties."""
        return {'properties': 'simplified analysis'}

    def _analyze_computational_characteristics(self, metrics: Any) -> Dict[str, Any]:
        """Analyze computational characteristics."""
        return {'characteristics': 'simplified analysis'}

    def _analyze_performance_characteristics(self, metrics: Any) -> Dict[str, Any]:
        """Analyze performance characteristics."""
        return {'characteristics': 'simplified analysis'}

    def _assess_computational_efficiency(self, metrics: Any) -> Dict[str, Any]:
        """Assess computational efficiency."""
        return {'efficiency': 'simplified assessment'}

    def _identify_computational_bottlenecks(self, metrics: Any) -> List[str]:
        """Identify computational bottlenecks."""
        return ['bottleneck 1', 'bottleneck 2']

    def _analyze_gpu_acceleration_effectiveness(self, metrics: Any) -> Dict[str, Any]:
        """Analyze GPU acceleration effectiveness."""
        return {'effectiveness': 'simplified analysis'}

    def _analyze_gpu_memory_transfers(self, metrics: Any) -> Dict[str, Any]:
        """Analyze GPU memory transfers."""
        return {'transfers': 'simplified analysis'}

    def _analyze_gpu_kernel_performance(self, metrics: Any) -> Dict[str, Any]:
        """Analyze GPU kernel performance."""
        return {'performance': 'simplified analysis'}

    def _analyze_matrix_stability(self, assessment: Any) -> Dict[str, Any]:
        """Analyze matrix stability."""
        return {'stability': 'simplified analysis'}

    def _analyze_numerical_properties(self, assessment: Any) -> Dict[str, Any]:
        """Analyze numerical properties."""
        return {'properties': 'simplified analysis'}

    def _analyze_optimization_improvements(self, effectiveness: Any) -> Dict[str, Any]:
        """Analyze optimization improvements."""
        return {'improvements': 'simplified analysis'}

    def _assess_optimization_scalability(self, effectiveness: Any) -> Dict[str, Any]:
        """Assess optimization scalability."""
        return {'scalability': 'simplified assessment'}

    def _evaluate_optimization_robustness(self, effectiveness: Any) -> Dict[str, Any]:
        """Evaluate optimization robustness."""
        return {'robustness': 'simplified evaluation'}

    def _analyze_memory_utilization(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze memory utilization."""
        return {'utilization': 'simplified analysis'}

    def _analyze_resource_efficiency(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze resource efficiency."""
        return {'efficiency': 'simplified analysis'}

    def _analyze_memory_optimization(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze memory optimization."""
        return {'optimization': 'simplified analysis'}

    def _analyze_resource_allocation(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze resource allocation."""
        return {'allocation': 'simplified analysis'}

    def _analyze_computational_efficiency(self, computational_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze computational efficiency."""
        return {'efficiency': 'simplified analysis'}

    def _analyze_hardware_utilization(self, gpu_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze hardware utilization."""
        return {'utilization': 'simplified analysis'}

    def _analyze_parallelization_effectiveness(self, computational_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze parallelization effectiveness."""
        return {'effectiveness': 'simplified analysis'}

    def _analyze_optimization_impact(self, computational_metrics: Dict[str, Any], gpu_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze optimization impact."""
        return {'impact': 'simplified analysis'}

    def _assess_matrix_stability(self, matrix_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess matrix stability."""
        return {'stability': 'simplified assessment'}

    def _analyze_numerical_robustness(self, matrix_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze numerical robustness."""
        return {'robustness': 'simplified analysis'}

    def _analyze_matrix_conditioning(self, matrix_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze matrix conditioning."""
        return {'conditioning': 'simplified analysis'}

    def _analyze_perturbation_sensitivity(self, matrix_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze perturbation sensitivity."""
        return {'sensitivity': 'simplified analysis'}

    def _compare_performance_benchmarks(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare performance benchmarks."""
        return {'benchmarks': 'simplified comparison'}

    def _establish_efficiency_benchmarks(self, computational_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Establish efficiency benchmarks."""
        return {'benchmarks': 'simplified benchmarks'}

    def _establish_scalability_benchmarks(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Establish scalability benchmarks."""
        return {'benchmarks': 'simplified benchmarks'}

    def _establish_optimization_benchmarks(self, performance_data: Dict[str, Any], computational_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Establish optimization benchmarks."""
        return {'benchmarks': 'simplified benchmarks'}

    def _recommend_performance_optimizations(self, optimization_results: Dict[str, Any]) -> List[str]:
        """Recommend performance optimizations."""
        return ['performance optimization 1', 'performance optimization 2']

    def _recommend_memory_optimizations(self, optimization_results: Dict[str, Any]) -> List[str]:
        """Recommend memory optimizations."""
        return ['memory optimization 1', 'memory optimization 2']

    def _recommend_hardware_optimizations(self, gpu_metrics: Dict[str, Any]) -> List[str]:
        """Recommend hardware optimizations."""
        return ['hardware optimization 1', 'hardware optimization 2']

    def _recommend_algorithmic_improvements(self, optimization_results: Dict[str, Any]) -> List[str]:
        """Recommend algorithmic improvements."""
        return ['algorithmic improvement 1', 'algorithmic improvement 2']

    def _prioritize_implementation_recommendations(self, optimization_results: Dict[str, Any], gpu_metrics: Dict[str, Any]) -> List[str]:
        """Prioritize implementation recommendations."""
        return ['priority 1', 'priority 2']

    # Visualization helper methods
    def _prepare_matrix_properties_data(self, matrix_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for matrix properties visualization."""
        return {
            'dimensions': matrix_results.get('shape', [0, 0]),
            'density': matrix_results.get('density', 0.0),
            'condition_number': matrix_results.get('condition_number', 0.0),
            'rank': matrix_results.get('rank', 0)
        }

    def _prepare_performance_timeline_data(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for performance timeline visualization."""
        return {
            'timeline': performance_data.get('performance_timeline', [])
        }

    def _prepare_computational_efficiency_data(self, computational_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for computational efficiency visualization."""
        return {
            'operations_per_second': computational_metrics.get('operations_per_second', 0),
            'efficiency_score': computational_metrics.get('efficiency_score', 0),
            'cache_hit_rate': computational_metrics.get('cache_hit_rate', 0)
        }

    def _prepare_optimization_comparison_data(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for optimization comparison visualization."""
        return {
            'baseline': performance_data.get('baseline_performance', 0),
            'optimized': performance_data.get('optimized_performance', 0),
            'improvement': performance_data.get('improvement_percentage', 0)
        }

    def _prepare_memory_usage_data(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for memory usage visualization."""
        return {
            'peak_memory': performance_data.get('peak_memory', 0),
            'average_memory': performance_data.get('average_memory', 0),
            'memory_efficiency': performance_data.get('memory_efficiency', 0)
        }

    def _prepare_gpu_acceleration_data(self, matrix_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for GPU acceleration visualization."""
        return {
            'gpu_used': matrix_results.get('gpu_used', False),
            'acceleration_factor': matrix_results.get('acceleration_factor', 1.0),
            'gpu_efficiency': matrix_results.get('gpu_efficiency', 0)
        }

    def _prepare_matrix_stability_data(self, matrix_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for matrix stability visualization."""
        return {
            'stability_score': matrix_results.get('stability_score', 0),
            'condition_number': matrix_results.get('condition_number', 0),
            'numerical_stability': matrix_results.get('numerical_stability', 0)
        }

    def _prepare_resource_utilization_data(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for resource utilization visualization."""
        return {
            'cpu_utilization': performance_data.get('cpu_utilization', []),
            'memory_utilization': performance_data.get('memory_utilization', []),
            'gpu_utilization': performance_data.get('gpu_utilization', [])
        }

    def _save_json_report(self, report: Dict[str, Any], timestamp: str, base_filename: str) -> Path:
        """Save report as JSON."""
        file_path = self.output_dir / f"{base_filename}_{timestamp}.json"
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        self.logger.info(f"📄 JSON report saved to: {file_path}")
        return file_path

    def _save_markdown_report(self, report: Dict[str, Any], timestamp: str, base_filename: str) -> Path:
        """Save report as Markdown."""
        file_path = self.output_dir / f"{base_filename}_{timestamp}.md"

        md_content = self._generate_markdown_content(report)

        with open(file_path, 'w') as f:
            f.write(md_content)
        self.logger.info(f"📄 Markdown report saved to: {file_path}")
        return file_path

    def _save_csv_report(self, report: Dict[str, Any], timestamp: str, base_filename: str) -> Path:
        """Save report data as CSV files."""
        csv_dir = self.output_dir / f"{base_filename}_data_{timestamp}"
        csv_dir.mkdir(exist_ok=True)

        try:
            # Save matrix operation metrics
            matrix_data = report.get('matrix_operation_metrics', {})
            if matrix_data and 'metrics' in matrix_data:
                matrix_df = pd.DataFrame([matrix_data['metrics']])
                matrix_df.to_csv(csv_dir / 'matrix_operation_metrics.csv', index=False)

            # Save computational performance
            comp_data = report.get('computational_performance', {})
            if comp_data and 'metrics' in comp_data:
                comp_df = pd.DataFrame([comp_data['metrics']])
                comp_df.to_csv(csv_dir / 'computational_performance.csv', index=False)

            # Save GPU acceleration metrics
            gpu_data = report.get('gpu_acceleration_analysis', {})
            if gpu_data and 'metrics' in gpu_data:
                gpu_df = pd.DataFrame([gpu_data['metrics']])
                gpu_df.to_csv(csv_dir / 'gpu_acceleration_metrics.csv', index=False)

        except Exception as e:
            self.logger.warning(f"Could not save CSV data: {e}")

        self.logger.info(f"📄 CSV data saved to: {csv_dir}")
        return csv_dir

    def _generate_visualizations(self, report: Dict[str, Any], timestamp: str, base_filename: str) -> None:
        """Generate and save visualizations."""
        try:
            viz_dir = self.output_dir / f"{base_filename}_visualizations_{timestamp}"
            viz_dir.mkdir(exist_ok=True)

            viz_data = report.get('visualization_data', {})

            # Generate matrix properties chart
            if 'matrix_properties_chart' in viz_data:
                self._create_matrix_properties_chart(viz_data['matrix_properties_chart'], viz_dir)

            # Generate performance timeline
            if 'performance_timeline' in viz_data:
                self._create_performance_timeline(viz_data['performance_timeline'], viz_dir)

            # Generate computational efficiency chart
            if 'computational_efficiency_chart' in viz_data:
                self._create_computational_efficiency_chart(viz_data['computational_efficiency_chart'], viz_dir)

            # Generate comprehensive summary dashboard
            try:
                self._create_comprehensive_summary_dashboard(report, viz_dir)
            except Exception as e:
                self.logger.warning(f"Could not create comprehensive summary dashboard: {e}")

            # Generate GPU performance comparison (if GPU data available)
            gpu_data = report.get('gpu_acceleration_analysis', {})
            if gpu_data and 'metrics' in gpu_data and gpu_data['metrics'].get('gpu_available', False):
                try:
                    self._create_gpu_performance_comparison_chart(gpu_data['metrics'], viz_dir)
                except Exception as e:
                    self.logger.warning(f"Could not create GPU performance comparison: {e}")

            # Generate matrix quality radar chart
            quality_data = report.get('matrix_quality_assessment', {})
            if quality_data and 'assessment' in quality_data:
                try:
                    self._create_matrix_quality_radar_chart(quality_data['assessment'], viz_dir)
                except Exception as e:
                    self.logger.warning(f"Could not create matrix quality radar chart: {e}")

            # Generate optimization effectiveness chart
            opt_data = report.get('optimization_effectiveness', {})
            if opt_data and 'effectiveness' in opt_data:
                try:
                    self._create_optimization_effectiveness_chart(opt_data['effectiveness'], viz_dir)
                except Exception as e:
                    self.logger.warning(f"Could not create optimization effectiveness chart: {e}")

            self.logger.info(f"📊 Visualizations saved to: {viz_dir}")

        except Exception as e:
            self.logger.warning(f"Could not generate visualizations: {e}")

    def _generate_markdown_content(self, report: Dict[str, Any]) -> str:
        """Generate comprehensive markdown report content."""
        md_lines = []

        # Header
        metadata = report.get('metadata', {})
        step_type = metadata.get('step_type', 'unknown')
        md_lines.extend([
            "# Step 7 Enhanced Matrix Operations - Comprehensive Analysis Report",
            "",
            f"**Generated:** {metadata.get('generated_at', 'Unknown')}",
            f"**Symbol:** {metadata.get('symbol', 'Unknown')}",
            f"**Exchange:** {metadata.get('exchange', 'Unknown')}",
            f"**Timeframe:** {metadata.get('timeframe', 'Unknown')}",
            f"**Step Description:** {metadata.get('description', 'Enhanced Matrix Operations Analysis')}",
            "",
        ])

        # Executive Summary
        md_lines.extend(self._generate_executive_summary_section(report))

        # Performance Summary
        md_lines.extend(self._generate_performance_summary_section(report))

        # Matrix Operation Metrics
        md_lines.extend(self._generate_matrix_metrics_section(report))

        # Computational Performance
        md_lines.extend(self._generate_computational_performance_section(report))

        # GPU Acceleration Analysis
        md_lines.extend(self._generate_gpu_acceleration_section(report))

        # Matrix Quality Assessment
        md_lines.extend(self._generate_matrix_quality_section(report))

        # Optimization Effectiveness
        md_lines.extend(self._generate_optimization_effectiveness_section(report))

        # Risk Assessment
        md_lines.extend(self._generate_risk_assessment_section(report))

        # Optimization Recommendations
        md_lines.extend(self._generate_optimization_recommendations_section(report))

        # Alerts and Recommendations
        md_lines.extend(self._generate_alerts_section(report))

        md_lines.append("")
        return "\n".join(md_lines)

    def _generate_executive_summary_section(self, report: Dict[str, Any]) -> List[str]:
        """Generate executive summary section."""
        lines = [
            "## 🚀 Executive Summary",
            "",
            "This comprehensive report provides detailed analysis of Step 7: Enhanced Matrix Operations with advanced performance optimizations and computational efficiency analysis.",
            "",
        ]

        # Key highlights
        matrix_data = report.get('matrix_operation_metrics', {})
        comp_data = report.get('computational_performance', {})
        gpu_data = report.get('gpu_acceleration_analysis', {})

        if matrix_data and 'metrics' in matrix_data:
            metrics = matrix_data['metrics']
            lines.extend([
                "### 📊 Key Metrics Overview",
                f"- **Matrix Dimensions:** {metrics.get('matrix_dimensions', (0, 0))}",
                f"- **Computation Time:** {metrics.get('computation_time_seconds', 0):.2f} seconds",
                f"- **Matrix Stability:** {metrics.get('matrix_stability_score', 0):.1f}%",
            ])

        if comp_data and 'metrics' in comp_data:
            metrics = comp_data['metrics']
            lines.extend([
                f"- **Operations per Second:** {metrics.get('operations_per_second', 0):,.0f}",
                f"- **Efficiency Score:** {metrics.get('execution_efficiency_score', 0):.1f}%",
            ])

        if gpu_data and 'metrics' in gpu_data:
            metrics = gpu_data['metrics']
            if metrics.get('gpu_available', False):
                lines.extend([
                    f"- **GPU Acceleration:** {metrics.get('gpu_acceleration_factor', 1):.1f}x speedup",
                    "",
                ])
            else:
                lines.append("")
        else:
            lines.append("")

        return lines

    def _generate_performance_summary_section(self, report: Dict[str, Any]) -> List[str]:
        """Generate performance summary section."""
        lines = [
            "## 📈 Performance Summary",
            "",
        ]

        matrix_data = report.get('matrix_operation_metrics', {})
        comp_data = report.get('computational_performance', {})
        gpu_data = report.get('gpu_acceleration_analysis', {})

        if matrix_data and 'metrics' in matrix_data:
            metrics = matrix_data['metrics']
            lines.extend([
                f"- **Matrix Dimensions:** {metrics.get('matrix_dimensions', (0, 0))}",
                f"- **Matrix Density:** {metrics.get('matrix_density', 0):.1%}",
                f"- **Computation Time:** {metrics.get('computation_time_seconds', 0):.2f} seconds",
                f"- **Memory Usage:** {metrics.get('memory_usage_mb', 0):.1f} MB",
                f"- **Matrix Stability Score:** {metrics.get('matrix_stability_score', 0):.1f}%",
                f"- **Numerical Precision Score:** {metrics.get('numerical_precision_score', 0):.1f}%",
                "",
            ])

        if comp_data and 'metrics' in comp_data:
            metrics = comp_data['metrics']
            lines.extend([
                "### ⚡ Computational Efficiency",
                f"- **Operations per Second:** {metrics.get('operations_per_second', 0):,.0f}",
                f"- **Memory Bandwidth:** {metrics.get('memory_bandwidth_mb_s', 0):.1f} MB/s",
                f"- **Cache Hit Rate:** {metrics.get('cache_hit_rate', 0):.1%}",
                f"- **FLOPS:** {metrics.get('floating_point_operations', 0):,.0f}",
                f"- **Efficiency Score:** {metrics.get('execution_efficiency_score', 0):.1f}%",
                f"- **Optimization Gain:** {metrics.get('optimization_gain_percentage', 0):.1f}%",
                "",
            ])

        if gpu_data and 'metrics' in gpu_data:
            metrics = gpu_data['metrics']
            if metrics.get('gpu_available', False):
                lines.extend([
                    "### 🎮 GPU Acceleration",
                    f"- **GPU Available:** ✅ Yes",
                    f"- **GPU Memory Used:** {metrics.get('gpu_memory_used_mb', 0):.1f} MB",
                    f"- **GPU Utilization:** {metrics.get('gpu_utilization_percentage', 0):.1f}%",
                    f"- **Acceleration Factor:** {metrics.get('gpu_acceleration_factor', 1):.1f}x",
                    f"- **GPU Efficiency Score:** {metrics.get('gpu_compute_efficiency_score', 0):.1f}%",
                    "",
                ])
            else:
                lines.append("### 🎮 GPU Acceleration\n\n- **GPU Available:** ❌ No\n")

        return lines

    def _generate_matrix_metrics_section(self, report: Dict[str, Any]) -> List[str]:
        """Generate matrix operation metrics section."""
        lines = [
            "## 🔢 Matrix Operation Metrics",
            "",
        ]

        matrix_data = report.get('matrix_operation_metrics', {})
        if matrix_data and 'metrics' in matrix_data:
            metrics = matrix_data['metrics']
            lines.extend([
                "### 📊 Core Metrics",
                "| Metric | Value | Status |",
                "|--------|-------|--------|",
                f"| Matrix Dimensions | {metrics.get('matrix_dimensions', (0, 0))} | {'✅ Large' if max(metrics.get('matrix_dimensions', (0, 0))) > 1000 else '⚠️ Small'} |",
                f"| Matrix Density | {metrics.get('matrix_density', 0):.1%} | {'✅ Dense' if metrics.get('matrix_density', 0) > 0.5 else '⚠️ Sparse'} |",
                f"| Condition Number | {metrics.get('matrix_condition_number', 0):.1f} | {'✅ Well-conditioned' if metrics.get('matrix_condition_number', float('inf')) < 1000 else '❌ Ill-conditioned'} |",
                f"| Matrix Rank | {metrics.get('matrix_rank', 0)} | {'✅ Full rank' if metrics.get('matrix_rank', 0) == min(metrics.get('matrix_dimensions', (0, 0))) else '⚠️ Rank deficient'} |",
                f"| Computation Time | {metrics.get('computation_time_seconds', 0):.2f}s | {'✅ Fast' if metrics.get('computation_time_seconds', 0) < 10 else '⚠️ Slow'} |",
                f"| Memory Usage | {metrics.get('memory_usage_mb', 0):.1f}MB | {'✅ Efficient' if metrics.get('memory_usage_mb', 0) < 1000 else '⚠️ High usage'} |",
                "",
                "### 🎯 Quality Scores",
                f"- **Matrix Stability Score:** {metrics.get('matrix_stability_score', 0):.1f}%",
                f"- **Numerical Precision Score:** {metrics.get('numerical_precision_score', 0):.1f}%",
                f"- **Matrix Sparsity Ratio:** {metrics.get('matrix_sparsity_ratio', 0):.1%}",
                "",
                "### ⚙️ Optimization Features",
                f"- **GPU Acceleration:** {'✅ Enabled' if metrics.get('gpu_acceleration_used', False) else '❌ Disabled'}",
                f"- **Numba Optimization:** {'✅ Enabled' if metrics.get('numba_optimization_used', False) else '❌ Disabled'}",
                f"- **Parallel Processing:** {'✅ Enabled' if metrics.get('parallel_processing_used', False) else '❌ Disabled'}",
                "",
            ])

        return lines

    def _generate_computational_performance_section(self, report: Dict[str, Any]) -> List[str]:
        """Generate computational performance section."""
        lines = [
            "## ⚡ Computational Performance Analysis",
            "",
        ]

        comp_data = report.get('computational_performance', {})
        if comp_data and 'metrics' in comp_data:
            metrics = comp_data['metrics']
            lines.extend([
                "### 📈 Performance Metrics",
                f"- **Total Operations:** {metrics.get('total_operations', 0):,}",
                f"- **Operations per Second:** {metrics.get('operations_per_second', 0):,.0f}",
                f"- **Memory Bandwidth:** {metrics.get('memory_bandwidth_mb_s', 0):.1f} MB/s",
                f"- **Floating Point Operations:** {metrics.get('floating_point_operations', 0):,.0f} FLOPS",
                f"- **Instructions per Cycle:** {metrics.get('instructions_per_cycle', 0):.2f} IPC",
                "",
                "### 🎯 Efficiency Metrics",
                f"- **Cache Hit Rate:** {metrics.get('cache_hit_rate', 0):.1%}",
                f"- **Execution Efficiency Score:** {metrics.get('execution_efficiency_score', 0):.1f}%",
                f"- **Branch Misprediction Rate:** {metrics.get('branch_misprediction_rate', 0):.1%}",
                f"- **Optimization Gain:** {metrics.get('optimization_gain_percentage', 0):.1f}%",
                f"- **Resource Utilization Score:** {metrics.get('resource_utilization_score', 0):.1f}%",
                "",
            ])

            # Performance analysis insights
            perf_analysis = comp_data.get('performance_analysis', {})
            if perf_analysis:
                lines.extend([
                    "### 🔍 Performance Analysis",
                    f"- **Bottleneck Identification:** {perf_analysis.get('bottlenecks', 'Analysis pending')}",
                    f"- **Optimization Opportunities:** {perf_analysis.get('opportunities', 'Analysis pending')}",
                    "",
                ])

        return lines

    def _generate_gpu_acceleration_section(self, report: Dict[str, Any]) -> List[str]:
        """Generate GPU acceleration analysis section."""
        lines = [
            "## 🎮 GPU Acceleration Analysis",
            "",
        ]

        gpu_data = report.get('gpu_acceleration_analysis', {})
        if gpu_data and 'metrics' in gpu_data:
            metrics = gpu_data['metrics']
            if metrics.get('gpu_available', False):
                lines.extend([
                    "### ✅ GPU Status: Available",
                    "",
                    "#### 📊 GPU Performance Metrics",
                    f"- **GPU Memory Used:** {metrics.get('gpu_memory_used_mb', 0):.1f} MB",
                    f"- **GPU Utilization:** {metrics.get('gpu_utilization_percentage', 0):.1f}%",
                    f"- **GPU Kernel Launch Time:** {metrics.get('gpu_kernel_launch_time_ms', 0):.2f} ms",
                    f"- **GPU Memory Transfer Time:** {metrics.get('gpu_memory_transfer_time_ms', 0):.2f} ms",
                    f"- **GPU Compute Time:** {metrics.get('gpu_compute_time_ms', 0):.2f} ms",
                    "",
                    "#### 🚀 Acceleration Results",
                    f"- **Acceleration Factor:** {metrics.get('gpu_acceleration_factor', 1):.1f}x",
                    f"- **Memory Efficiency Score:** {metrics.get('gpu_memory_efficiency_score', 0):.1f}%",
                    f"- **Compute Efficiency Score:** {metrics.get('gpu_compute_efficiency_score', 0):.1f}%",
                    "",
                ])

                # Acceleration analysis
                accel_analysis = gpu_data.get('acceleration_analysis', {})
                if accel_analysis:
                    lines.extend([
                        "#### 🔍 Acceleration Analysis",
                        f"- **Effectiveness:** {accel_analysis.get('effectiveness', 'Analysis pending')}",
                        f"- **Bottlenecks:** {accel_analysis.get('bottlenecks', 'Analysis pending')}",
                        "",
                    ])
            else:
                lines.extend([
                    "### ❌ GPU Status: Not Available",
                    "",
                    "GPU acceleration is not available on this system. Consider the following alternatives:",
                    "",
                    "- Use CPU-based parallel processing",
                    "- Optimize memory access patterns",
                    "- Consider cloud-based GPU instances for large-scale computations",
                    "",
                ])

        return lines

    def _generate_matrix_quality_section(self, report: Dict[str, Any]) -> List[str]:
        """Generate matrix quality assessment section."""
        lines = [
            "## 🔍 Matrix Quality Assessment",
            "",
        ]

        quality_data = report.get('matrix_quality_assessment', {})
        if quality_data and 'assessment' in quality_data:
            assessment = quality_data['assessment']
            lines.extend([
                "### 📊 Quality Metrics",
                f"- **Well Conditioned:** {'✅ Yes' if assessment.get('matrix_well_conditioned', False) else '❌ No'}",
                f"- **Numerical Stability Score:** {assessment.get('numerical_stability_score', 0):.1f}%",
                f"- **Computation Accuracy Score:** {assessment.get('computation_accuracy_score', 0):.1f}%",
                f"- **Matrix Orthogonality Score:** {assessment.get('matrix_orthogonality_score', 0):.1f}%",
                f"- **Eigenvalue Distribution Score:** {assessment.get('eigenvalue_distribution_score', 0):.1f}%",
                f"- **Singular Value Distribution Score:** {assessment.get('singular_value_distribution_score', 0):.1f}%",
                f"- **Matrix Energy Concentration:** {assessment.get('matrix_energy_concentration', 0):.1f}%",
                f"- **Noise to Signal Ratio:** {assessment.get('noise_to_signal_ratio', 0):.1f}%",
                "",
                f"**Overall Quality Score:** {quality_data.get('quality_score', 0):.1f}%",
                "",
            ])

            # Quality warnings
            warnings = assessment.get('quality_warnings', [])
            if warnings:
                lines.extend([
                    "### ⚠️ Quality Warnings",
                ])
                for warning in warnings[:5]:  # Limit to 5 warnings
                    lines.append(f"- {warning}")
                lines.append("")

            # Quality improvements
            improvements = assessment.get('quality_improvements', [])
            if improvements:
                lines.extend([
                    "### 💡 Quality Improvements",
                ])
                for improvement in improvements[:5]:  # Limit to 5 improvements
                    lines.append(f"- {improvement}")
                lines.append("")

        return lines

    def _generate_optimization_effectiveness_section(self, report: Dict[str, Any]) -> List[str]:
        """Generate optimization effectiveness section."""
        lines = [
            "## 🎯 Optimization Effectiveness",
            "",
        ]

        opt_data = report.get('optimization_effectiveness', {})
        if opt_data and 'effectiveness' in opt_data:
            effectiveness = opt_data['effectiveness']
            lines.extend([
                "### 📈 Performance Comparison",
                f"- **Baseline Performance:** {effectiveness.get('baseline_performance', 0):.2f}",
                f"- **Optimized Performance:** {effectiveness.get('optimized_performance', 0):.2f}",
                f"- **Performance Improvement:** {effectiveness.get('performance_improvement_percentage', 0):.1f}%",
                "",
                "### 💾 Resource Optimization",
                f"- **Memory Usage Reduction:** {effectiveness.get('memory_usage_reduction_percentage', 0):.1f}%",
                f"- **Time Complexity Improvement:** {effectiveness.get('time_complexity_improvement', 'Unknown')}",
                f"- **Space Complexity Improvement:** {effectiveness.get('space_complexity_improvement', 'Unknown')}",
                "",
                "### 🎯 Optimization Metrics",
                f"- **Scalability Score:** {effectiveness.get('scalability_score', 0):.1f}%",
                f"- **Robustness Score:** {effectiveness.get('optimization_robustness_score', 0):.1f}%",
                "",
            ])

            # Optimization recommendations
            recommendations = effectiveness.get('optimization_recommendations', [])
            if recommendations:
                lines.extend([
                    "### 💡 Optimization Recommendations",
                ])
                for rec in recommendations[:5]:  # Limit to 5 recommendations
                    lines.append(f"- {rec}")
                lines.append("")

        return lines

    def _generate_risk_assessment_section(self, report: Dict[str, Any]) -> List[str]:
        """Generate risk assessment section."""
        lines = [
            "## ⚠️ Risk Assessment",
            "",
        ]

        # Calculate overall risk level based on various factors
        risk_level = "MEDIUM"  # Default
        risk_factors = []

        # Assess risks from different components
        matrix_data = report.get('matrix_operation_metrics', {})
        if matrix_data and 'metrics' in matrix_data:
            metrics = matrix_data['metrics']
            if metrics.get('matrix_condition_number', float('inf')) > 10000:
                risk_factors.append("Extremely high condition number indicates severe numerical instability")
                risk_level = "HIGH"
            elif metrics.get('matrix_condition_number', float('inf')) > 1000:
                risk_factors.append("High condition number may cause numerical instability")
                if risk_level == "MEDIUM":
                    risk_level = "MEDIUM-HIGH"

            if metrics.get('matrix_stability_score', 50) < 60:
                risk_factors.append("Low matrix stability score indicates potential numerical issues")
                risk_level = "HIGH"

        comp_data = report.get('computational_performance', {})
        if comp_data and 'metrics' in comp_data:
            metrics = comp_data['metrics']
            if metrics.get('execution_efficiency_score', 50) < 70:
                risk_factors.append("Low computational efficiency may impact performance")
                if risk_level == "MEDIUM":
                    risk_level = "MEDIUM-HIGH"

        quality_data = report.get('matrix_quality_assessment', {})
        if quality_data and 'assessment' in quality_data:
            assessment = quality_data['assessment']
            if not assessment.get('matrix_well_conditioned', True):
                risk_factors.append("Matrix is ill-conditioned, results may be unreliable")
                risk_level = "HIGH"

            if assessment.get('numerical_stability_score', 50) < 70:
                risk_factors.append("Low numerical stability may affect computation accuracy")
                if risk_level == "MEDIUM":
                    risk_level = "MEDIUM-HIGH"

        lines.extend([
            f"**Overall Risk Level:** {risk_level}",
            "",
        ])

        if risk_factors:
            lines.extend([
                "### 🚨 Key Risk Factors",
            ])
            for factor in risk_factors:
                lines.append(f"- {factor}")
            lines.append("")

        # Mitigation strategies
        lines.extend([
            "### 🛡️ Risk Mitigation Strategies",
            "- Implement robust numerical validation checks",
            "- Use regularization techniques for ill-conditioned matrices",
            "- Monitor numerical stability throughout computations",
            "- Implement fallback algorithms for high-risk operations",
            "- Validate results against known test cases",
            "",
        ])

        return lines

    def _generate_optimization_recommendations_section(self, report: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations section."""
        lines = [
            "## 🔧 Optimization Recommendations",
            "",
        ]

        # Performance optimizations
        perf_opt_data = report.get('optimization_recommendations', {})
        if perf_opt_data and 'performance_optimizations' in perf_opt_data:
            optimizations = perf_opt_data['performance_optimizations']
            if optimizations:
                lines.extend([
                    "### ⚡ Performance Optimizations",
                ])
                for opt in optimizations[:5]:  # Limit to 5 recommendations
                    lines.append(f"- {opt}")
                lines.append("")

        # Memory optimizations
        if perf_opt_data and 'memory_optimizations' in perf_opt_data:
            optimizations = perf_opt_data['memory_optimizations']
            if optimizations:
                lines.extend([
                    "### 💾 Memory Optimizations",
                ])
                for opt in optimizations[:3]:  # Limit to 3 recommendations
                    lines.append(f"- {opt}")
                lines.append("")

        # Algorithmic improvements
        if perf_opt_data and 'algorithmic_improvements' in perf_opt_data:
            improvements = perf_opt_data['algorithmic_improvements']
            if improvements:
                lines.extend([
                    "### 🧮 Algorithmic Improvements",
                ])
                for imp in improvements[:3]:  # Limit to 3 improvements
                    lines.append(f"- {imp}")
                lines.append("")

        # Hardware optimizations
        gpu_data = report.get('gpu_acceleration_analysis', {})
        if gpu_data and 'metrics' in gpu_data:
            metrics = gpu_data['metrics']
            if not metrics.get('gpu_available', False):
                lines.extend([
                    "### 🎮 Hardware Optimization Opportunities",
                    "- Consider GPU acceleration for large matrix operations",
                    "- Evaluate cloud-based GPU instances for intensive computations",
                    "- Optimize CPU cache utilization for better performance",
                    "",
                ])

        return lines

    def _generate_alerts_section(self, report: Dict[str, Any]) -> List[str]:
        """Generate alerts and recommendations section."""
        lines = [
            "## 🚨 Alerts and Recommendations",
            "",
        ]

        alerts = []

        # Check for critical issues
        matrix_data = report.get('matrix_operation_metrics', {})
        if matrix_data and 'metrics' in matrix_data:
            metrics = matrix_data['metrics']
            if metrics.get('matrix_condition_number', 0) > 10000:
                alerts.append("🚨 **CRITICAL:** Extremely high condition number detected - results may be unreliable")
            elif metrics.get('matrix_condition_number', 0) > 1000:
                alerts.append("⚠️ **WARNING:** High condition number may cause numerical instability")

            if metrics.get('matrix_stability_score', 100) < 50:
                alerts.append("🚨 **CRITICAL:** Matrix stability score is critically low")

            if metrics.get('computation_time_seconds', 0) > 300:  # 5 minutes
                alerts.append("⚠️ **WARNING:** Computation time is excessively high")

        comp_data = report.get('computational_performance', {})
        if comp_data and 'metrics' in comp_data:
            metrics = comp_data['metrics']
            if metrics.get('execution_efficiency_score', 100) < 60:
                alerts.append("⚠️ **WARNING:** Computational efficiency is below acceptable levels")

            if metrics.get('branch_misprediction_rate', 0) > 0.1:
                alerts.append("⚠️ **WARNING:** High branch misprediction rate detected")

        gpu_data = report.get('gpu_acceleration_analysis', {})
        if gpu_data and 'metrics' in gpu_data:
            metrics = gpu_data['metrics']
            if not metrics.get('gpu_available', False):
                alerts.append("💡 **INFO:** GPU acceleration not available - consider hardware upgrade for better performance")

        quality_data = report.get('matrix_quality_assessment', {})
        if quality_data and 'assessment' in quality_data:
            assessment = quality_data['assessment']
            if not assessment.get('matrix_well_conditioned', True):
                alerts.append("🚨 **CRITICAL:** Matrix is ill-conditioned - numerical results may be inaccurate")

            if assessment.get('numerical_stability_score', 100) < 70:
                alerts.append("⚠️ **WARNING:** Numerical stability is compromised")

        if alerts:
            lines.extend(alerts)
            lines.append("")
        else:
            lines.extend([
                "✅ No critical alerts detected",
                "",
            ])

        # General recommendations
        lines.extend([
            "### 💡 General Recommendations",
            "- Regularly monitor matrix condition numbers and numerical stability",
            "- Implement comprehensive validation checks for matrix operations",
            "- Consider matrix preconditioning for ill-conditioned problems",
            "- Optimize memory access patterns for better cache performance",
            "- Evaluate GPU acceleration opportunities for computational bottlenecks",
            "",
        ])

        return lines

    def _create_matrix_properties_chart(self, data: Dict[str, Any], viz_dir: Path) -> None:
        """Create matrix properties visualization."""
        try:
            properties = ['Density', 'Stability', 'Conditioning']
            values = [
                data.get('density', 0) * 100,
                data.get('stability_score', 50),
                max(0, 100 - min(data.get('condition_number', 1000) / 10, 100))
            ]

            plt.figure(figsize=(10, 6))
            bars = plt.bar(properties, values, color=['skyblue', 'lightgreen', 'lightcoral'])

            plt.title('Matrix Properties Analysis')
            plt.ylabel('Score (%)')
            plt.ylim(0, 100)

            # Add value labels
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{value:.1f}%', ha='center', va='bottom')

            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(viz_dir / 'matrix_properties.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.warning(f"Could not create matrix properties chart: {e}")

    def _create_performance_timeline(self, data: Dict[str, Any], viz_dir: Path) -> None:
        """Create performance timeline visualization."""
        try:
            timeline = data.get('timeline', [])
            if timeline:
                plt.figure(figsize=(12, 6))
                plt.plot(range(len(timeline)), timeline, 'b-', linewidth=2, alpha=0.7)
                plt.title('Performance Timeline')
                plt.xlabel('Time Steps')
                plt.ylabel('Performance Metric')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(viz_dir / 'performance_timeline.png', dpi=300, bbox_inches='tight')
                plt.close()

        except Exception as e:
            self.logger.warning(f"Could not create performance timeline: {e}")

    def _create_computational_efficiency_chart(self, data: Dict[str, Any], viz_dir: Path) -> None:
        """Create computational efficiency visualization."""
        try:
            metrics = ['Operations/sec', 'Efficiency', 'Cache Hit Rate']
            values = [
                data.get('operations_per_second', 0) / 1000,  # Scale down for display
                data.get('efficiency_score', 0),
                data.get('cache_hit_rate', 0) * 100
            ]

            plt.figure(figsize=(10, 6))
            bars = plt.bar(metrics, values, color=['blue', 'green', 'orange'])

            plt.title('Computational Efficiency Analysis')
            plt.ylabel('Score')
            plt.grid(True, alpha=0.3)

            # Add value labels
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                        f'{value:.1f}', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(viz_dir / 'computational_efficiency.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.warning(f"Could not create computational efficiency chart: {e}")

    def _create_comprehensive_summary_dashboard(self, report: Dict[str, Any], viz_dir: Path) -> None:
        """Create comprehensive summary dashboard for matrix operations."""
        try:
            plt.figure(figsize=(16, 12))

            # Matrix metrics overview
            plt.subplot(3, 3, 1)
            matrix_data = report.get('matrix_operation_metrics', {})
            if matrix_data and 'metrics' in matrix_data:
                metrics = matrix_data['metrics']
                matrix_scores = [
                    metrics.get('matrix_stability_score', 50),
                    metrics.get('numerical_precision_score', 50),
                    min(100, 100 - metrics.get('matrix_condition_number', 1000) / 10),
                    metrics.get('matrix_density', 0.5) * 100
                ]
                matrix_labels = ['Stability', 'Precision', 'Conditioning', 'Density']

                plt.barh(matrix_labels, matrix_scores, color='lightblue', alpha=0.7)
                plt.title('Matrix Metrics Overview', fontsize=11, fontweight='bold')
                plt.xlim(0, 100)

            # Computational performance
            plt.subplot(3, 3, 2)
            comp_data = report.get('computational_performance', {})
            if comp_data and 'metrics' in comp_data:
                metrics = comp_data['metrics']
                comp_scores = [
                    metrics.get('execution_efficiency_score', 50),
                    metrics.get('resource_utilization_score', 50),
                    metrics.get('cache_hit_rate', 0.5) * 100,
                    max(0, 100 - metrics.get('branch_misprediction_rate', 0.05) * 1000)
                ]
                comp_labels = ['Efficiency', 'Resource Use', 'Cache Hit', 'Branch Pred.']

                plt.barh(comp_labels, comp_scores, color='lightgreen', alpha=0.7)
                plt.title('Computational Performance', fontsize=11, fontweight='bold')
                plt.xlim(0, 100)

            # GPU acceleration status
            plt.subplot(3, 3, 3)
            gpu_data = report.get('gpu_acceleration_analysis', {})
            if gpu_data and 'metrics' in gpu_data:
                metrics = gpu_data['metrics']
                if metrics.get('gpu_available', False):
                    gpu_scores = [
                        metrics.get('gpu_utilization_percentage', 0),
                        metrics.get('gpu_acceleration_factor', 1) * 20,  # Scale for display
                        metrics.get('gpu_compute_efficiency_score', 50),
                        metrics.get('gpu_memory_efficiency_score', 50)
                    ]
                    gpu_labels = ['GPU Utilization', 'Acceleration', 'Compute Eff.', 'Memory Eff.']
                    colors = ['green', 'blue', 'orange', 'purple']
                else:
                    gpu_scores = [0, 0, 0, 0]
                    gpu_labels = ['GPU Not Available'] * 4
                    colors = ['red'] * 4

                plt.barh(gpu_labels, gpu_scores, color=colors, alpha=0.7)
                plt.title('GPU Acceleration Status', fontsize=11, fontweight='bold')
                plt.xlim(0, 100)

            # Matrix quality assessment
            plt.subplot(3, 3, 4)
            quality_data = report.get('matrix_quality_assessment', {})
            if quality_data and 'assessment' in quality_data:
                assessment = quality_data['assessment']
                quality_scores = [
                    assessment.get('numerical_stability_score', 50),
                    assessment.get('computation_accuracy_score', 50),
                    assessment.get('matrix_orthogonality_score', 50),
                    assessment.get('eigenvalue_distribution_score', 50)
                ]
                quality_labels = ['Stability', 'Accuracy', 'Orthogonality', 'Eigenvalues']

                plt.pie(quality_scores, labels=quality_labels, autopct='%1.1f%%', startangle=90)
                plt.title('Matrix Quality Distribution', fontsize=11, fontweight='bold')

            # Risk assessment gauge
            plt.subplot(3, 3, 5)
            risk_level = "MEDIUM"
            risk_score = 50

            # Calculate risk score
            if matrix_data and 'metrics' in matrix_data:
                condition_num = metrics.get('matrix_condition_number', 1000)
                if condition_num > 10000:
                    risk_score = 90
                    risk_level = "CRITICAL"
                elif condition_num > 1000:
                    risk_score = 70
                    risk_level = "HIGH"
                else:
                    risk_score = 30
                    risk_level = "LOW"

            plt.pie([risk_score, 100-risk_score], colors=['red', 'lightgray'], startangle=90, counterclock=False)
            plt.text(0, 0, f'{risk_level}\n{risk_score}%', ha='center', va='center', fontsize=12, fontweight='bold')
            plt.title('Risk Assessment', fontsize=11, fontweight='bold')

            # Optimization effectiveness
            plt.subplot(3, 3, 6)
            opt_data = report.get('optimization_effectiveness', {})
            if opt_data and 'effectiveness' in opt_data:
                effectiveness = opt_data['effectiveness']
                opt_scores = [
                    effectiveness.get('performance_improvement_percentage', 0),
                    effectiveness.get('memory_usage_reduction_percentage', 0),
                    effectiveness.get('scalability_score', 50),
                    effectiveness.get('optimization_robustness_score', 50)
                ]
                opt_labels = ['Performance', 'Memory', 'Scalability', 'Robustness']

                plt.plot(opt_labels, opt_scores, 'o-', linewidth=2, markersize=8, color='purple')
                plt.title('Optimization Effectiveness', fontsize=11, fontweight='bold')
                plt.ylim(0, 100)
                plt.grid(True, alpha=0.3)

            # Performance timeline
            plt.subplot(3, 3, 7)
            if comp_data and 'metrics' in comp_data:
                metrics = comp_data['metrics']
                timeline_labels = ['Start', 'Processing', 'Optimization', 'Complete']
                timeline_values = [
                    0,
                    metrics.get('computation_time_seconds', 0) * 0.4,
                    metrics.get('computation_time_seconds', 0) * 0.8,
                    metrics.get('computation_time_seconds', 0)
                ]

                plt.plot(timeline_labels, timeline_values, 's-', linewidth=2, markersize=8, color='navy')
                plt.title('Process Timeline', fontsize=11, fontweight='bold')
                plt.ylabel('Time (seconds)')
                plt.grid(True, alpha=0.3)

            # Key metrics summary
            plt.subplot(3, 3, 8)
            key_metrics = {}
            if matrix_data and 'metrics' in matrix_data:
                key_metrics['Matrix Rank'] = metrics.get('matrix_rank', 0)
                key_metrics['Operations/sec'] = metrics.get('operations_per_second', 0) / 1000 if comp_data and 'metrics' in comp_data else 0

            if key_metrics:
                labels = list(key_metrics.keys())
                values = list(key_metrics.values())

                plt.bar(labels, values, color='teal', alpha=0.7)
                plt.title('Key Metrics Summary', fontsize=11, fontweight='bold')
                plt.xticks(rotation=45, ha='right')

                # Add value labels
                for i, v in enumerate(values):
                    plt.text(i, v + max(values)*0.01 if values else 0.1, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')

            # Overall status indicator
            plt.subplot(3, 3, 9)
            status_score = 75  # Default good status

            # Calculate overall status
            issues = 0
            if matrix_data and 'metrics' in matrix_data:
                if metrics.get('matrix_condition_number', 1000) > 1000:
                    issues += 1
                if metrics.get('matrix_stability_score', 50) < 70:
                    issues += 1

            if comp_data and 'metrics' in comp_data:
                if metrics.get('execution_efficiency_score', 50) < 70:
                    issues += 1

            status_score = max(0, 100 - issues * 20)

            status_color = 'green' if status_score >= 80 else 'orange' if status_score >= 60 else 'red'
            plt.pie([status_score, 100-status_score], colors=[status_color, 'lightgray'], startangle=90, counterclock=False)
            plt.text(0, 0, f'Overall\n{status_score}%', ha='center', va='center', fontsize=12, fontweight='bold')
            plt.title('System Health', fontsize=11, fontweight='bold')

            plt.tight_layout()
            plt.savefig(viz_dir / 'comprehensive_summary_dashboard.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.warning(f"Could not create comprehensive summary dashboard: {e}")

    def _create_gpu_performance_comparison_chart(self, gpu_metrics: Dict[str, Any], viz_dir: Path) -> None:
        """Create GPU performance comparison visualization."""
        try:
            plt.figure(figsize=(12, 8))

            # GPU metrics comparison
            metrics = ['Memory Transfer', 'Kernel Launch', 'Compute Time', 'Utilization']
            gpu_values = [
                gpu_metrics.get('gpu_memory_transfer_time_ms', 0),
                gpu_metrics.get('gpu_kernel_launch_time_ms', 0),
                gpu_metrics.get('gpu_compute_time_ms', 0),
                gpu_metrics.get('gpu_utilization_percentage', 0)
            ]

            # CPU baseline (estimated)
            cpu_values = [
                gpu_values[0] * 5,  # Memory transfer slower on CPU
                gpu_values[1] * 2,  # Kernel launch overhead
                gpu_values[2] * 10, # Compute slower on CPU
                80  # Typical CPU utilization
            ]

            x = np.arange(len(metrics))
            width = 0.35

            plt.bar(x - width/2, gpu_values, width, label='GPU', alpha=0.7, color='green')
            plt.bar(x + width/2, cpu_values, width, label='CPU (Estimated)', alpha=0.7, color='blue')

            plt.title('GPU vs CPU Performance Comparison', fontsize=14, fontweight='bold')
            plt.xlabel('Performance Metrics')
            plt.ylabel('Time (ms) / Utilization (%)')
            plt.xticks(x, metrics)
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')

            # Add speedup annotation
            speedup = gpu_metrics.get('gpu_acceleration_factor', 1)
            plt.annotate(f'GPU Speedup: {speedup:.1f}x',
                        xy=(0.02, 0.98), xycoords='axes fraction',
                        fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

            plt.tight_layout()
            plt.savefig(viz_dir / 'gpu_performance_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.warning(f"Could not create GPU performance comparison: {e}")

    def _create_matrix_quality_radar_chart(self, assessment: Dict[str, Any], viz_dir: Path) -> None:
        """Create matrix quality radar chart."""
        try:
            plt.figure(figsize=(10, 8))

            # Quality metrics for radar chart
            categories = ['Stability', 'Accuracy', 'Orthogonality', 'Eigenvalues', 'Singular Values', 'Energy Conc.']
            values = [
                assessment.get('numerical_stability_score', 50),
                assessment.get('computation_accuracy_score', 50),
                assessment.get('matrix_orthogonality_score', 50),
                assessment.get('eigenvalue_distribution_score', 50),
                assessment.get('singular_value_distribution_score', 50),
                assessment.get('matrix_energy_concentration', 50)
            ]

            # Close the radar chart
            values += values[:1]
            categories += categories[:1]

            angles = [n / float(len(categories[:-1])) * 2 * 3.14159 for n in range(len(categories[:-1]))]
            angles += angles[:1]

            plt.polar(angles, values, 'o-', linewidth=2, label='Quality Metrics')
            plt.fill(angles, values, alpha=0.25)
            plt.xticks(angles[:-1], categories[:-1])
            plt.title('Matrix Quality Assessment Radar', fontsize=14, fontweight='bold')
            plt.ylim(0, 100)

            # Add quality score in center
            overall_quality = np.mean(values[:-1])
            plt.text(0, 0, f'Overall\n{overall_quality:.1f}%', ha='center', va='center',
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

            plt.tight_layout()
            plt.savefig(viz_dir / 'matrix_quality_radar.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.warning(f"Could not create matrix quality radar chart: {e}")

    def _create_optimization_effectiveness_chart(self, effectiveness: Dict[str, Any], viz_dir: Path) -> None:
        """Create optimization effectiveness visualization."""
        try:
            plt.figure(figsize=(12, 8))

            # Before/After comparison
            plt.subplot(2, 2, 1)
            baseline = effectiveness.get('baseline_performance', 1)
            optimized = effectiveness.get('optimized_performance', 1)
            improvement = effectiveness.get('performance_improvement_percentage', 0)

            labels = ['Baseline', 'Optimized']
            values = [baseline, optimized]
            colors = ['red', 'green']

            bars = plt.bar(labels, values, color=colors, alpha=0.7)
            plt.title('Performance Before/After Optimization', fontsize=12, fontweight='bold')
            plt.ylabel('Performance Score')
            plt.grid(True, alpha=0.3, axis='y')

            # Add improvement annotation
            plt.annotate(f'Improvement: {improvement:.1f}%',
                        xy=(1, optimized), xytext=(0.6, optimized + 0.1),
                        arrowprops=dict(arrowstyle='->', color='green'),
                        fontsize=10, fontweight='bold')

            # Memory usage comparison
            plt.subplot(2, 2, 2)
            memory_reduction = effectiveness.get('memory_usage_reduction_percentage', 0)

            plt.pie([100-memory_reduction, memory_reduction],
                   labels=['Memory Usage', 'Reduction'],
                   colors=['red', 'green'], autopct='%1.1f%%', startangle=90)
            plt.title('Memory Optimization Impact', fontsize=12, fontweight='bold')

            # Scalability and robustness
            plt.subplot(2, 2, 3)
            scalability = effectiveness.get('scalability_score', 50)
            robustness = effectiveness.get('optimization_robustness_score', 50)

            metrics = ['Scalability', 'Robustness']
            scores = [scalability, robustness]
            colors = ['blue', 'purple']

            plt.bar(metrics, scores, color=colors, alpha=0.7)
            plt.title('Optimization Quality Metrics', fontsize=12, fontweight='bold')
            plt.ylabel('Score (%)')
            plt.ylim(0, 100)
            plt.grid(True, alpha=0.3, axis='y')

            # Add value labels
            for i, score in enumerate(scores):
                plt.text(i, score + 1, f'{score:.1f}%', ha='center', va='bottom', fontweight='bold')

            # Optimization recommendations impact
            plt.subplot(2, 2, 4)
            recommendations = effectiveness.get('optimization_recommendations', [])

            if recommendations:
                # Count recommendation categories
                perf_count = sum(1 for rec in recommendations if 'performance' in rec.lower())
                memory_count = sum(1 for rec in recommendations if 'memory' in rec.lower())
                algo_count = sum(1 for rec in recommendations if 'algorithm' in rec.lower())

                categories = ['Performance', 'Memory', 'Algorithm']
                counts = [perf_count, memory_count, algo_count]

                plt.bar(categories, counts, color=['orange', 'green', 'blue'], alpha=0.7)
                plt.title('Optimization Recommendations', fontsize=12, fontweight='bold')
                plt.ylabel('Number of Recommendations')
                plt.grid(True, alpha=0.3, axis='y')

                # Add value labels
                for i, count in enumerate(counts):
                    plt.text(i, count + 0.1, str(count), ha='center', va='bottom', fontweight='bold')
            else:
                plt.text(0.5, 0.5, 'No specific\nrecommendations', ha='center', va='center',
                        transform=plt.gca().transAxes, fontsize=12)
                plt.title('Optimization Recommendations', fontsize=12, fontweight='bold')

            plt.tight_layout()
            plt.savefig(viz_dir / 'optimization_effectiveness.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.warning(f"Could not create optimization effectiveness chart: {e}")

