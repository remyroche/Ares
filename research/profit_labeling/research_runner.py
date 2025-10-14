"""
Research Runner for Multi-Horizon Profit Labeling Analysis

This module provides a comprehensive research runner that orchestrates the entire
profit labeling analysis workflow, similar to the research runners used in 
HMM clustering research. It coordinates all analysis components and generates
complete research reports.

Key Research Workflows:
1. Complete Heuristic Analysis Pipeline
2. Comprehensive Validation Testing
3. Parameter Optimization Studies
4. Comparative Analysis Studies
5. End-to-End Research Reports
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
from datetime import datetime
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.utils.logger import get_logger
from src.training.steps.pre_training.multi_horizon_profit_labeler import (
    MultiHorizonProfitLabeler, 
    MultiHorizonConfig
)

from .heuristic_analyzer import (
    HeuristicAnalyzer, 
    HeuristicAnalysisConfig, 
    HeuristicAnalysisResult,
    analyze_profit_labeling_heuristics
)
from .labeling_validator import (
    LabelingValidator,
    ValidationConfig,
    ValidationResult,
    validate_profit_labeling
)
from .parameter_optimizer import (
    ParameterOptimizer,
    OptimizationConfig,
    OptimizationResult,
    OptimizationMethod,
    OptimizationObjective,
    optimize_labeling_parameters
)
from .labeling_visualizer import (
    LabelingVisualizer,
    VisualizationConfig,
    create_profit_labeling_visualizations
)


class ResearchWorkflow(Enum):
    """Enumeration of research workflows."""
    HEURISTIC_ANALYSIS = "heuristic_analysis"
    VALIDATION_TESTING = "validation_testing"
    PARAMETER_OPTIMIZATION = "parameter_optimization"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    COMPLETE_PIPELINE = "complete_pipeline"


@dataclass
class ResearchConfig:
    """Configuration for research workflows."""
    # Workflow selection
    workflows: List[ResearchWorkflow] = field(default_factory=lambda: [
        ResearchWorkflow.COMPLETE_PIPELINE
    ])
    
    # Component configurations
    heuristic_config: Optional[HeuristicAnalysisConfig] = None
    validation_config: Optional[ValidationConfig] = None
    optimization_config: Optional[OptimizationConfig] = None
    visualization_config: Optional[VisualizationConfig] = None
    
    # Data handling
    train_test_split: float = 0.7
    validation_split: float = 0.3
    min_data_size: int = 2000
    
    # Output settings
    output_dir: str = "profit_labeling_research"
    generate_reports: bool = True
    generate_visualizations: bool = True
    save_intermediate_results: bool = True
    
    # Performance settings
    parallel_processing: bool = True
    max_workers: int = 4
    timeout_minutes: int = 60
    
    # Experimental settings
    random_seed: int = 42
    bootstrap_runs: int = 3
    comparative_baselines: List[str] = field(default_factory=lambda: [
        'random', 'simple_threshold'
    ])


@dataclass
class ResearchResults:
    """Container for comprehensive research results."""
    workflow_type: ResearchWorkflow
    heuristic_results: Optional[Dict[str, HeuristicAnalysisResult]] = None
    validation_results: Optional[Dict[str, ValidationResult]] = None
    optimization_results: Optional[Dict[str, OptimizationResult]] = None
    visualization_paths: Optional[Dict[str, Path]] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class ResearchRunner:
    """
    Comprehensive research runner for multi-horizon profit labeling analysis.
    
    This class orchestrates complete research workflows for analyzing and
    optimizing the profit labeling system. It coordinates all analysis
    components and generates comprehensive research reports.
    
    Key Features:
    1. **Complete Pipeline**: End-to-end analysis from data to insights
    2. **Modular Workflows**: Individual analysis components or full pipeline
    3. **Parallel Processing**: Efficient execution of compute-intensive tasks
    4. **Comprehensive Reporting**: Detailed reports with visualizations
    5. **Reproducible Research**: Consistent results with seed management
    """
    
    def __init__(self, config: Optional[ResearchConfig] = None):
        """Initialize the research runner."""
        self.config = config or ResearchConfig()
        self.logger = get_logger('ResearchRunner')
        
        # Research state
        self.research_results: Dict[str, ResearchResults] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        # Set random seed for reproducibility
        np.random.seed(self.config.random_seed)
        
        self.logger.info('🔬 Research Runner initialized')
        self.logger.info(f'   → Workflows: {[w.value for w in self.config.workflows]}')
        self.logger.info(f'   → Output directory: {self.config.output_dir}')
        
    def run_research(self, 
                    market_data: pd.DataFrame,
                    custom_labeling_config: Optional[MultiHorizonConfig] = None) -> Dict[str, ResearchResults]:
        """
        Run comprehensive research workflows.
        
        Args:
            market_data: Market data for analysis
            custom_labeling_config: Custom labeling configuration (optional)
            
        Returns:
            Dictionary of research results by workflow type
        """
        self.logger.info('🚀 Starting comprehensive research workflows')
        
        # Validate input data
        self._validate_input_data(market_data)
        
        # Prepare output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Record execution start
        execution_start = time.time()
        
        try:
            # Run each selected workflow
            for workflow in self.config.workflows:
                self.logger.info(f'📊 Running {workflow.value} workflow')
                
                workflow_start = time.time()
                result = self._run_single_workflow(
                    workflow, market_data, custom_labeling_config, output_dir
                )
                workflow_time = time.time() - workflow_start
                
                result.execution_time = workflow_time
                result.metadata['workflow'] = workflow.value
                
                self.research_results[workflow.value] = result
                
                self.logger.info(f'✅ {workflow.value} completed in {workflow_time:.1f}s')
            
            # Generate comprehensive report
            if self.config.generate_reports:
                self._generate_comprehensive_report(output_dir)
            
            # Record execution history
            total_time = time.time() - execution_start
            self.execution_history.append({
                'timestamp': datetime.now(),
                'workflows': [w.value for w in self.config.workflows],
                'execution_time': total_time,
                'data_size': len(market_data),
                'results_count': len(self.research_results)
            })
            
            self.logger.info(f'🎉 Research completed in {total_time:.1f}s')
            
            return self.research_results
            
        except Exception as e:
            self.logger.error(f'❌ Research failed: {e}')
            raise
    
    def _validate_input_data(self, market_data: pd.DataFrame):
        """Validate input market data."""
        if len(market_data) < self.config.min_data_size:
            raise ValueError(f"Insufficient data: need {self.config.min_data_size}, got {len(market_data)}")
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in market_data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for data quality issues
        if market_data.isnull().sum().sum() > len(market_data) * 0.1:
            self.logger.warning('⚠️ High percentage of missing values in data')
        
        self.logger.info(f'✅ Data validation passed: {len(market_data)} samples')
    
    def _run_single_workflow(self,
                           workflow: ResearchWorkflow,
                           market_data: pd.DataFrame,
                           custom_labeling_config: Optional[MultiHorizonConfig],
                           output_dir: Path) -> ResearchResults:
        """Run a single research workflow."""
        
        if workflow == ResearchWorkflow.HEURISTIC_ANALYSIS:
            return self._run_heuristic_analysis_workflow(
                market_data, custom_labeling_config, output_dir
            )
        elif workflow == ResearchWorkflow.VALIDATION_TESTING:
            return self._run_validation_testing_workflow(
                market_data, custom_labeling_config, output_dir
            )
        elif workflow == ResearchWorkflow.PARAMETER_OPTIMIZATION:
            return self._run_parameter_optimization_workflow(
                market_data, custom_labeling_config, output_dir
            )
        elif workflow == ResearchWorkflow.COMPARATIVE_ANALYSIS:
            return self._run_comparative_analysis_workflow(
                market_data, custom_labeling_config, output_dir
            )
        elif workflow == ResearchWorkflow.COMPLETE_PIPELINE:
            return self._run_complete_pipeline_workflow(
                market_data, custom_labeling_config, output_dir
            )
        else:
            raise ValueError(f"Unknown workflow: {workflow}")
    
    def _run_heuristic_analysis_workflow(self,
                                       market_data: pd.DataFrame,
                                       custom_labeling_config: Optional[MultiHorizonConfig],
                                       output_dir: Path) -> ResearchResults:
        """Run heuristic analysis workflow."""
        self.logger.info('🔍 Running heuristic analysis workflow')
        
        # Configure heuristic analysis
        heuristic_config = self.config.heuristic_config or HeuristicAnalysisConfig()
        
        # Run analysis
        analyzer = HeuristicAnalyzer(heuristic_config)
        heuristic_results = analyzer.analyze_labeling_heuristics(
            market_data, custom_labeling_config
        )
        
        # Generate visualizations
        visualization_paths = {}
        if self.config.generate_visualizations:
            visualizer = LabelingVisualizer(self.config.visualization_config)
            visualization_paths = visualizer.visualize_heuristic_analysis(
                heuristic_results, output_dir / "heuristic_analysis"
            )
        
        # Save results
        if self.config.save_intermediate_results:
            results_path = output_dir / "heuristic_analysis_results.json"
            analyzer.save_results(results_path)
        
        return ResearchResults(
            workflow_type=ResearchWorkflow.HEURISTIC_ANALYSIS,
            heuristic_results=heuristic_results,
            visualization_paths=visualization_paths,
            metadata={'config': heuristic_config.__dict__}
        )
    
    def _run_validation_testing_workflow(self,
                                       market_data: pd.DataFrame,
                                       custom_labeling_config: Optional[MultiHorizonConfig],
                                       output_dir: Path) -> ResearchResults:
        """Run validation testing workflow."""
        self.logger.info('🔬 Running validation testing workflow')
        
        # Configure validation
        validation_config = self.config.validation_config or ValidationConfig()
        
        # Run validation
        validator = LabelingValidator(validation_config)
        validation_results = validator.validate_labeling_quality(
            market_data, None, custom_labeling_config
        )
        
        # Generate visualizations
        visualization_paths = {}
        if self.config.generate_visualizations:
            visualizer = LabelingVisualizer(self.config.visualization_config)
            visualization_paths = visualizer.visualize_validation_results(
                validation_results, output_dir / "validation_testing"
            )
        
        # Save results
        if self.config.save_intermediate_results:
            results_path = output_dir / "validation_results.json"
            validator.save_validation_results(results_path)
        
        return ResearchResults(
            workflow_type=ResearchWorkflow.VALIDATION_TESTING,
            validation_results=validation_results,
            visualization_paths=visualization_paths,
            metadata={'config': validation_config.__dict__}
        )
    
    def _run_parameter_optimization_workflow(self,
                                           market_data: pd.DataFrame,
                                           custom_labeling_config: Optional[MultiHorizonConfig],
                                           output_dir: Path) -> ResearchResults:
        """Run parameter optimization workflow."""
        self.logger.info('🎯 Running parameter optimization workflow')
        
        # Configure optimization
        optimization_config = self.config.optimization_config or OptimizationConfig()
        
        # Split data for optimization
        split_idx = int(len(market_data) * self.config.train_test_split)
        train_data = market_data.iloc[:split_idx]
        test_data = market_data.iloc[split_idx:]
        
        # Run optimization
        optimizer = ParameterOptimizer(optimization_config)
        
        if self.config.parallel_processing and len(market_data) > 5000:
            # Run multiple optimization methods in parallel
            methods = [
                OptimizationMethod.GRID_SEARCH,
                OptimizationMethod.RANDOM_SEARCH,
                OptimizationMethod.BAYESIAN_OPTIMIZATION
            ]
            optimization_results = optimizer.compare_optimization_methods(
                train_data, methods
            )
        else:
            # Run single optimization
            optimization_result = optimizer.optimize_parameters(train_data, test_data)
            optimization_results = {
                f"{optimization_config.method.value}_{optimization_config.objective.value}": 
                optimization_result
            }
        
        # Generate visualizations
        visualization_paths = {}
        if self.config.generate_visualizations:
            visualizer = LabelingVisualizer(self.config.visualization_config)
            visualization_paths = visualizer.visualize_optimization_results(
                optimization_results, output_dir / "parameter_optimization"
            )
        
        # Save results
        if self.config.save_intermediate_results:
            results_path = output_dir / "optimization_results.json"
            optimizer.save_optimization_results(results_path)
        
        return ResearchResults(
            workflow_type=ResearchWorkflow.PARAMETER_OPTIMIZATION,
            optimization_results=optimization_results,
            visualization_paths=visualization_paths,
            metadata={'config': optimization_config.__dict__}
        )
    
    def _run_comparative_analysis_workflow(self,
                                         market_data: pd.DataFrame,
                                         custom_labeling_config: Optional[MultiHorizonConfig],
                                         output_dir: Path) -> ResearchResults:
        """Run comparative analysis workflow."""
        self.logger.info('📊 Running comparative analysis workflow')
        
        # This workflow compares different labeling configurations
        results = {}
        
        # Test different configurations
        test_configs = self._generate_test_configurations(custom_labeling_config)
        
        for config_name, config in test_configs.items():
            self.logger.info(f'   → Testing configuration: {config_name}')
            
            # Run heuristic analysis for this configuration
            analyzer = HeuristicAnalyzer(self.config.heuristic_config)
            heuristic_results = analyzer.analyze_labeling_heuristics(market_data, config)
            
            # Run validation for this configuration  
            validator = LabelingValidator(self.config.validation_config)
            validation_results = validator.validate_labeling_quality(
                market_data, None, config
            )
            
            results[config_name] = {
                'heuristic_results': heuristic_results,
                'validation_results': validation_results,
                'config': config
            }
        
        # Generate comparative visualizations
        visualization_paths = {}
        if self.config.generate_visualizations:
            visualization_paths = self._create_comparative_visualizations(
                results, output_dir / "comparative_analysis"
            )
        
        return ResearchResults(
            workflow_type=ResearchWorkflow.COMPARATIVE_ANALYSIS,
            heuristic_results=results,  # Store all results here
            visualization_paths=visualization_paths,
            metadata={'configurations_tested': list(test_configs.keys())}
        )
    
    def _run_complete_pipeline_workflow(self,
                                      market_data: pd.DataFrame,
                                      custom_labeling_config: Optional[MultiHorizonConfig],
                                      output_dir: Path) -> ResearchResults:
        """Run complete research pipeline."""
        self.logger.info('🔄 Running complete research pipeline')
        
        # Run all components in sequence
        pipeline_results = {}
        
        # 1. Heuristic Analysis
        self.logger.info('   → Step 1: Heuristic Analysis')
        heuristic_result = self._run_heuristic_analysis_workflow(
            market_data, custom_labeling_config, output_dir
        )
        pipeline_results['heuristic'] = heuristic_result
        
        # 2. Validation Testing
        self.logger.info('   → Step 2: Validation Testing')
        validation_result = self._run_validation_testing_workflow(
            market_data, custom_labeling_config, output_dir
        )
        pipeline_results['validation'] = validation_result
        
        # 3. Parameter Optimization
        self.logger.info('   → Step 3: Parameter Optimization')
        optimization_result = self._run_parameter_optimization_workflow(
            market_data, custom_labeling_config, output_dir
        )
        pipeline_results['optimization'] = optimization_result
        
        # 4. Generate integrated visualizations
        if self.config.generate_visualizations:
            self.logger.info('   → Step 4: Integrated Visualizations')
            visualization_paths = self._create_integrated_visualizations(
                pipeline_results, output_dir / "complete_pipeline"
            )
        else:
            visualization_paths = {}
        
        return ResearchResults(
            workflow_type=ResearchWorkflow.COMPLETE_PIPELINE,
            heuristic_results=heuristic_result.heuristic_results,
            validation_results=validation_result.validation_results,
            optimization_results=optimization_result.optimization_results,
            visualization_paths=visualization_paths,
            metadata={'pipeline_steps': ['heuristic', 'validation', 'optimization']}
        )
    
    def _generate_test_configurations(self, 
                                    base_config: Optional[MultiHorizonConfig]) -> Dict[str, MultiHorizonConfig]:
        """Generate test configurations for comparative analysis."""
        configs = {}
        
        # Base configuration
        base = base_config or MultiHorizonConfig()
        configs['baseline'] = base
        
        # Conservative configuration (smaller targets, shorter horizons)
        conservative = MultiHorizonConfig()
        conservative.profit_targets = {
            'micro': 0.002,   # 0.2%
            'small': 0.003,   # 0.3%
            'medium': 0.005,  # 0.5%
            'good': 0.008     # 0.8%
        }
        conservative.time_horizons = {
            'immediate': 1,   # 5 minutes
            'short': 2        # 10 minutes
        }
        configs['conservative'] = conservative
        
        # Aggressive configuration (larger targets, longer horizons)
        aggressive = MultiHorizonConfig()
        aggressive.profit_targets = {
            'micro': 0.005,   # 0.5%
            'small': 0.008,   # 0.8%
            'medium': 0.012,  # 1.2%
            'good': 0.020     # 2.0%
        }
        aggressive.time_horizons = {
            'immediate': 3,   # 15 minutes
            'short': 6        # 30 minutes
        }
        configs['aggressive'] = aggressive
        
        # Quality-focused configuration
        quality_focused = MultiHorizonConfig()
        quality_focused.enable_quality_scoring = True
        quality_focused.speed_weight = 0.2
        quality_focused.risk_weight = 0.5
        quality_focused.profitability_weight = 0.3
        configs['quality_focused'] = quality_focused
        
        return configs
    
    def _create_comparative_visualizations(self,
                                         comparative_results: Dict[str, Any],
                                         output_dir: Path) -> Dict[str, Path]:
        """Create visualizations for comparative analysis."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # This would create comparison charts between different configurations
        # For now, return empty dict as placeholder
        return {}
    
    def _create_integrated_visualizations(self,
                                        pipeline_results: Dict[str, ResearchResults],
                                        output_dir: Path) -> Dict[str, Path]:
        """Create integrated visualizations for complete pipeline."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        visualization_paths = {}
        
        # Combine all visualization paths from individual steps
        for step_name, result in pipeline_results.items():
            if result.visualization_paths:
                for viz_name, viz_path in result.visualization_paths.items():
                    visualization_paths[f"{step_name}_{viz_name}"] = viz_path
        
        # Create integrated dashboard if possible
        if self.config.generate_visualizations:
            try:
                visualizer = LabelingVisualizer(self.config.visualization_config)
                dashboard_path = visualizer.create_comprehensive_research_dashboard(
                    heuristic_results=pipeline_results.get('heuristic', ResearchResults(ResearchWorkflow.HEURISTIC_ANALYSIS)).heuristic_results,
                    validation_results=pipeline_results.get('validation', ResearchResults(ResearchWorkflow.VALIDATION_TESTING)).validation_results,
                    optimization_results=pipeline_results.get('optimization', ResearchResults(ResearchWorkflow.PARAMETER_OPTIMIZATION)).optimization_results,
                    output_dir=output_dir
                )
                
                if dashboard_path:
                    visualization_paths['integrated_dashboard'] = Path(dashboard_path)
                    
            except Exception as e:
                self.logger.warning(f'Failed to create integrated dashboard: {e}')
        
        return visualization_paths
    
    def _generate_comprehensive_report(self, output_dir: Path):
        """Generate comprehensive research report."""
        self.logger.info('📝 Generating comprehensive research report')
        
        report_lines = [
            "# Multi-Horizon Profit Labeling Research Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            f"Completed {len(self.research_results)} research workflows",
            ""
        ]
        
        # Add summary statistics
        total_execution_time = sum(result.execution_time for result in self.research_results.values())
        report_lines.extend([
            f"**Total Execution Time**: {total_execution_time:.1f} seconds",
            f"**Workflows Completed**: {', '.join(self.research_results.keys())}",
            ""
        ])
        
        # Add detailed results for each workflow
        for workflow_name, result in self.research_results.items():
            report_lines.extend([
                f"## {workflow_name.replace('_', ' ').title()} Results",
                f"**Execution Time**: {result.execution_time:.1f} seconds",
                ""
            ])
            
            # Add specific results based on workflow type
            if result.heuristic_results:
                report_lines.extend([
                    "### Heuristic Analysis",
                    f"- Analyzed {len(result.heuristic_results)} heuristic components",
                    ""
                ])
            
            if result.validation_results:
                significant_count = sum(1 for r in result.validation_results.values() 
                                      if r.is_significant)
                report_lines.extend([
                    "### Validation Results",
                    f"- Validated {len(result.validation_results)} components",
                    f"- {significant_count} statistically significant results",
                    ""
                ])
            
            if result.optimization_results:
                best_result = max(result.optimization_results.values(), 
                                key=lambda x: x.best_score)
                report_lines.extend([
                    "### Optimization Results",
                    f"- Best score: {best_result.best_score:.4f}",
                    f"- Best method: {best_result.method.value}",
                    ""
                ])
        
        # Add recommendations section
        report_lines.extend([
            "## Key Recommendations",
            ""
        ])
        
        recommendations = self._generate_research_recommendations()
        for rec in recommendations:
            report_lines.append(f"- {rec}")
        
        # Save report
        report_path = output_dir / "comprehensive_research_report.md"
        with open(report_path, 'w') as f:
            f.write("\n".join(report_lines))
        
        self.logger.info(f'📄 Research report saved to {report_path}')
    
    def _generate_research_recommendations(self) -> List[str]:
        """Generate research recommendations based on results."""
        recommendations = []
        
        # Analyze results and generate specific recommendations
        for workflow_name, result in self.research_results.items():
            if result.heuristic_results:
                # Analyze heuristic results for recommendations
                target_effectiveness = [r for r in result.heuristic_results.values() 
                                      if 'effectiveness' in str(r.analysis_type)]
                if target_effectiveness:
                    avg_effectiveness = np.mean([r.metric_value for r in target_effectiveness])
                    if avg_effectiveness < 0.3:
                        recommendations.append("Consider revising target/horizon combinations - low effectiveness detected")
            
            if result.validation_results:
                # Analyze validation results
                significant_ratio = sum(1 for r in result.validation_results.values() 
                                      if r.is_significant) / len(result.validation_results)
                if significant_ratio < 0.5:
                    recommendations.append("Labeling methodology needs improvement - low statistical significance")
            
            if result.optimization_results:
                # Analyze optimization results
                best_scores = [r.best_score for r in result.optimization_results.values()]
                if max(best_scores) < 0.6:
                    recommendations.append("Parameter optimization shows limited improvement - consider alternative approaches")
        
        # Generic recommendations if no specific ones found
        if not recommendations:
            recommendations.extend([
                "Research completed successfully with acceptable results",
                "Consider running additional validation with different market conditions",
                "Monitor labeling performance in live trading environments"
            ])
        
        return recommendations
    
    def save_research_results(self, output_path: Union[str, Path]):
        """Save all research results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to serializable format
        serializable_results = {}
        for workflow_name, result in self.research_results.items():
            serializable_results[workflow_name] = {
                'workflow_type': result.workflow_type.value,
                'execution_time': result.execution_time,
                'metadata': result.metadata,
                'timestamp': result.timestamp.isoformat(),
                'has_heuristic_results': result.heuristic_results is not None,
                'has_validation_results': result.validation_results is not None,
                'has_optimization_results': result.optimization_results is not None,
                'visualization_count': len(result.visualization_paths) if result.visualization_paths else 0
            }
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump({
                'research_results_summary': serializable_results,
                'execution_history': self.execution_history,
                'config': {
                    'workflows': [w.value for w in self.config.workflows],
                    'output_dir': self.config.output_dir,
                    'random_seed': self.config.random_seed
                }
            }, f, indent=2)
        
        self.logger.info(f'💾 Research results saved to {output_path}')
    
    def run_quick_analysis(self, 
                          market_data: pd.DataFrame,
                          analysis_type: str = "heuristic") -> ResearchResults:
        """Run quick analysis for rapid insights."""
        self.logger.info(f'⚡ Running quick {analysis_type} analysis')
        
        # Create temporary config for quick analysis
        quick_config = ResearchConfig()
        quick_config.generate_visualizations = False
        quick_config.save_intermediate_results = False
        
        if analysis_type == "heuristic":
            quick_config.workflows = [ResearchWorkflow.HEURISTIC_ANALYSIS]
        elif analysis_type == "validation":
            quick_config.workflows = [ResearchWorkflow.VALIDATION_TESTING]
        elif analysis_type == "optimization":
            quick_config.workflows = [ResearchWorkflow.PARAMETER_OPTIMIZATION]
        else:
            quick_config.workflows = [ResearchWorkflow.HEURISTIC_ANALYSIS]
        
        # Store original config and temporarily replace
        original_config = self.config
        self.config = quick_config
        
        try:
            # Run analysis
            results = self.run_research(market_data)
            return list(results.values())[0]  # Return first (and only) result
        finally:
            # Restore original config
            self.config = original_config


# Convenience functions
def run_profit_labeling_research(market_data: pd.DataFrame,
                                workflows: Optional[List[ResearchWorkflow]] = None,
                                output_dir: str = "profit_labeling_research",
                                config: Optional[ResearchConfig] = None) -> Dict[str, ResearchResults]:
    """Convenience function to run profit labeling research."""
    if config is None:
        config = ResearchConfig()
        
    if workflows is not None:
        config.workflows = workflows
        
    config.output_dir = output_dir
    
    runner = ResearchRunner(config)
    return runner.run_research(market_data)


def run_quick_profit_labeling_analysis(market_data: pd.DataFrame,
                                     analysis_type: str = "heuristic") -> ResearchResults:
    """Convenience function for quick analysis."""
    runner = ResearchRunner()
    return runner.run_quick_analysis(market_data, analysis_type)


# Example usage and testing
if __name__ == '__main__':
    # Example research workflow
    print('🧪 Testing Multi-Horizon Profit Labeling Research Framework')
    
    # Generate sample data
    dates = pd.date_range('2024-01-01', periods=2000, freq='5min')
    np.random.seed(42)
    
    # Generate realistic price data
    base_price = 100.0
    returns = np.random.normal(0.0001, 0.002, 2000)
    prices = [base_price]
    
    for ret in returns[:-1]:
        prices.append(prices[-1] * (1 + ret))
    
    sample_data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, 2000)
    }, index=dates)
    
    # Ensure OHLC consistency
    for i in range(len(sample_data)):
        sample_data.loc[sample_data.index[i], 'high'] = max(sample_data.iloc[i][['open', 'high', 'low', 'close']])
        sample_data.loc[sample_data.index[i], 'low'] = min(sample_data.iloc[i][['open', 'high', 'low', 'close']])
    
    print(f'📊 Generated sample data: {len(sample_data)} samples')
    
    # Run quick heuristic analysis
    print('\n🔍 Running quick heuristic analysis...')
    quick_result = run_quick_profit_labeling_analysis(sample_data, "heuristic")
    print(f'✅ Quick analysis completed in {quick_result.execution_time:.1f}s')
    
    if quick_result.heuristic_results:
        print(f'   → Analyzed {len(quick_result.heuristic_results)} heuristic components')
    
    print('\n🎉 Research framework test completed successfully!')
