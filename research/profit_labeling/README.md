# Multi-Horizon Profit Labeling Research Framework

A comprehensive research framework for analyzing and optimizing the multi-horizon profit labeling system from a data-driven perspective, similar to how we analyze HMM clustering. This framework provides systematic tools to examine labeling heuristics, validate labeling quality, optimize parameters, and generate actionable insights.

## 🎯 Overview

This research framework addresses the key question: **"How effective are our profit labeling heuristics, and how can we optimize them for better ML model performance?"**

The framework provides data-driven analysis of the multi-horizon profit labeling system [[memory:8334081]], examining:

- **Heuristic Effectiveness**: Are our labeling rules actually predictive?
- **Parameter Sensitivity**: How do parameter changes affect labeling quality?
- **Validation Robustness**: Are our labels statistically significant and stable?
- **Optimization Opportunities**: What parameter configurations work best?

## 🏗️ Architecture

The framework consists of 5 main components, following the same pattern as our HMM clustering research:

```
src/research/profit_labeling/
├── __init__.py                     # Main exports and framework entry point
├── heuristic_analyzer.py           # Data-driven heuristic effectiveness analysis
├── labeling_validator.py           # Comprehensive labeling quality validation
├── parameter_optimizer.py          # Systematic parameter optimization
├── labeling_visualizer.py          # Publication-quality visualizations
├── research_runner.py              # Complete research workflow orchestration
└── README.md                       # This documentation
```

### Component Overview

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **HeuristicAnalyzer** | Analyze labeling heuristics effectiveness | Target/horizon analysis, quality scoring validation, composite score analysis |
| **LabelingValidator** | Validate labeling quality and consistency | Statistical significance, temporal stability, bias detection |
| **ParameterOptimizer** | Optimize labeling parameters systematically | Grid search, Bayesian optimization, multi-objective optimization |
| **LabelingVisualizer** | Create research visualizations | Static charts, interactive dashboards, publication-ready outputs |
| **ResearchRunner** | Orchestrate complete research workflows | End-to-end analysis, comparative studies, automated reporting |

## 🚀 Quick Start

### Basic Usage

```python
import pandas as pd
from research.profit_labeling import (
    HeuristicAnalyzer,
    LabelingValidator, 
    ParameterOptimizer,
    LabelingVisualizer,
    ResearchRunner
)

# Load your market data (OHLCV format)
market_data = pd.read_csv('your_market_data.csv')

# Option 1: Run complete research pipeline
runner = ResearchRunner()
results = runner.run_research(market_data)

# Option 2: Run individual components
analyzer = HeuristicAnalyzer()
heuristic_results = analyzer.analyze_labeling_heuristics(market_data)

validator = LabelingValidator()
validation_results = validator.validate_labeling_quality(market_data)

optimizer = ParameterOptimizer()
optimization_results = optimizer.optimize_parameters(market_data)

# Generate visualizations
visualizer = LabelingVisualizer()
charts = visualizer.visualize_heuristic_analysis(heuristic_results, "output/")
```

### Quick Analysis

```python
from research.profit_labeling import run_quick_profit_labeling_analysis

# Run quick heuristic analysis
result = run_quick_profit_labeling_analysis(market_data, "heuristic")
print(f"Analysis completed in {result.execution_time:.1f}s")
print(f"Found {len(result.heuristic_results)} heuristic components")
```

## 📊 Research Workflows

### 1. Heuristic Analysis Workflow

Analyzes the effectiveness of profit labeling heuristics:

```python
from research.profit_labeling import HeuristicAnalyzer, HeuristicAnalysisConfig

# Configure analysis
config = HeuristicAnalysisConfig(
    analyze_target_combinations=True,
    analyze_quality_scoring=True,
    analyze_composite_scores=True,
    bootstrap_samples=1000,
    confidence_level=0.95
)

# Run analysis
analyzer = HeuristicAnalyzer(config)
results = analyzer.analyze_labeling_heuristics(market_data)

# Generate report
report = analyzer.generate_analysis_report()
print(report)
```

**Key Analysis Areas:**
- **Target/Horizon Effectiveness**: Which profit target and time horizon combinations work best?
- **Quality Scoring Validation**: Are quality scores actually predictive of outcomes?
- **Composite Score Analysis**: Do composite scores add value over individual components?
- **Parameter Sensitivity**: How sensitive are results to parameter changes?

### 2. Validation Testing Workflow

Validates labeling quality and consistency:

```python
from research.profit_labeling import LabelingValidator, ValidationConfig

# Configure validation
config = ValidationConfig(
    validate_consistency=True,
    validate_stability=True,
    validate_predictiveness=True,
    validate_significance=True,
    validate_bias=True,
    significance_level=0.05,
    bootstrap_iterations=1000
)

# Run validation
validator = LabelingValidator(config)
results = validator.validate_labeling_quality(market_data)

# Generate report
report = validator.generate_validation_report()
print(report)
```

**Key Validation Areas:**
- **Label Consistency**: Are similar market conditions labeled similarly?
- **Temporal Stability**: Do labels remain stable over time?
- **Predictive Validity**: Do labels predict future outcomes?
- **Statistical Significance**: Are labeling patterns statistically valid?
- **Bias Detection**: Are there systematic biases in labeling?

### 3. Parameter Optimization Workflow

Systematically optimizes labeling parameters:

```python
from research.profit_labeling import (
    ParameterOptimizer, 
    OptimizationConfig, 
    OptimizationMethod, 
    OptimizationObjective
)

# Configure optimization
config = OptimizationConfig(
    method=OptimizationMethod.BAYESIAN_OPTIMIZATION,
    objective=OptimizationObjective.PREDICTIVE_POWER,
    profit_targets_range={
        'micro': (0.002, 0.005),
        'small': (0.003, 0.008),
        'medium': (0.005, 0.012),
        'good': (0.008, 0.020)
    },
    time_horizons_range={
        'immediate': (1, 4),
        'short': (2, 8)
    },
    bayesian_iterations=50
)

# Run optimization
optimizer = ParameterOptimizer(config)
result = optimizer.optimize_parameters(market_data)

print(f"Best score: {result.best_score:.4f}")
print(f"Best parameters: {result.best_params}")
```

**Optimization Features:**
- **Multiple Methods**: Grid search, random search, Bayesian optimization
- **Multi-Objective**: Balance predictive power, stability, and economic value
- **Constraint Handling**: Respect economic and statistical constraints
- **Parallel Processing**: Efficient parameter space exploration

### 4. Complete Research Pipeline

Run end-to-end research with automated reporting:

```python
from research.profit_labeling import ResearchRunner, ResearchConfig, ResearchWorkflow

# Configure complete pipeline
config = ResearchConfig(
    workflows=[ResearchWorkflow.COMPLETE_PIPELINE],
    generate_reports=True,
    generate_visualizations=True,
    output_dir="profit_labeling_research"
)

# Run complete research
runner = ResearchRunner(config)
results = runner.run_research(market_data)

# Results include heuristic analysis, validation, optimization, and visualizations
complete_result = results['complete_pipeline']
print(f"Research completed in {complete_result.execution_time:.1f}s")
```

## 🔬 Research Applications

### Market Condition Analysis

Analyze how labeling effectiveness varies across different market conditions:

```python
# Split data by volatility regime
high_vol_data = market_data[market_data['volatility'] > market_data['volatility'].quantile(0.7)]
low_vol_data = market_data[market_data['volatility'] < market_data['volatility'].quantile(0.3)]

# Analyze each regime
analyzer = HeuristicAnalyzer()
high_vol_results = analyzer.analyze_labeling_heuristics(high_vol_data)
low_vol_results = analyzer.analyze_labeling_heuristics(low_vol_data)

# Compare effectiveness across regimes
print("High Volatility Results:", high_vol_results)
print("Low Volatility Results:", low_vol_results)
```

### Parameter Sensitivity Studies

Study how sensitive labeling is to parameter changes:

```python
# Test different profit target configurations
configs_to_test = {
    'conservative': {'micro': 0.002, 'small': 0.003, 'medium': 0.005},
    'moderate': {'micro': 0.003, 'small': 0.005, 'medium': 0.007},
    'aggressive': {'micro': 0.005, 'small': 0.008, 'medium': 0.012}
}

results = {}
for config_name, targets in configs_to_test.items():
    # Create labeling config with these targets
    labeling_config = MultiHorizonConfig()
    labeling_config.profit_targets.update(targets)
    
    # Analyze effectiveness
    analyzer = HeuristicAnalyzer()
    results[config_name] = analyzer.analyze_labeling_heuristics(
        market_data, labeling_config
    )

# Compare results
for config_name, result in results.items():
    print(f"{config_name}: {result}")
```

### Comparative Optimization Studies

Compare different optimization approaches:

```python
from research.profit_labeling import compare_labeling_optimization_methods

# Compare multiple optimization methods
methods = [
    OptimizationMethod.GRID_SEARCH,
    OptimizationMethod.RANDOM_SEARCH,
    OptimizationMethod.BAYESIAN_OPTIMIZATION
]

comparison_results = compare_labeling_optimization_methods(
    market_data, 
    methods=methods,
    objective=OptimizationObjective.PREDICTIVE_POWER
)

# Analyze which method works best
for method_name, result in comparison_results.items():
    print(f"{method_name}: Score {result.best_score:.4f}, Time {result.metadata.get('optimization_time', 0):.1f}s")
```

## 📈 Visualization System

### Static Visualizations

Generate publication-quality static charts:

```python
from research.profit_labeling import LabelingVisualizer, VisualizationConfig

# Configure visualization
config = VisualizationConfig(
    output_format="png",
    output_dpi=300,
    figure_size=(12, 8),
    style_theme="seaborn-v0_8"
)

# Generate visualizations
visualizer = LabelingVisualizer(config)

# Heuristic analysis charts
heuristic_charts = visualizer.visualize_heuristic_analysis(
    heuristic_results, "output/heuristic_analysis"
)

# Validation results charts
validation_charts = visualizer.visualize_validation_results(
    validation_results, "output/validation_results"
)

# Optimization results charts
optimization_charts = visualizer.visualize_optimization_results(
    optimization_results, "output/optimization_results"
)
```

### Interactive Dashboards

Create interactive research dashboards:

```python
# Create comprehensive interactive dashboard
dashboard_path = visualizer.create_comprehensive_research_dashboard(
    heuristic_results=heuristic_results,
    validation_results=validation_results,
    optimization_results=optimization_results,
    market_data=market_data,
    labeled_data=labeled_data,
    output_dir="research_dashboard"
)

print(f"Interactive dashboard created: {dashboard_path}")
# Open in browser to explore results interactively
```

## ⚙️ Configuration

### Comprehensive Configuration Example

```python
from research.profit_labeling import (
    ResearchConfig, 
    HeuristicAnalysisConfig,
    ValidationConfig,
    OptimizationConfig,
    VisualizationConfig
)

# Heuristic analysis configuration
heuristic_config = HeuristicAnalysisConfig(
    analyze_target_combinations=True,
    analyze_quality_scoring=True,
    analyze_composite_scores=True,
    analyze_parameter_sensitivity=True,
    min_samples_per_analysis=1000,
    bootstrap_samples=1000,
    confidence_level=0.95,
    min_predictive_power=0.55,
    compare_to_random=True
)

# Validation configuration
validation_config = ValidationConfig(
    validate_consistency=True,
    validate_stability=True,
    validate_predictiveness=True,
    validate_significance=True,
    validate_bias=True,
    significance_level=0.05,
    bootstrap_iterations=1000,
    cv_folds=5,
    min_economic_significance=0.001
)

# Optimization configuration
optimization_config = OptimizationConfig(
    method=OptimizationMethod.BAYESIAN_OPTIMIZATION,
    objective=OptimizationObjective.COMPOSITE_SCORE,
    profit_targets_range={
        'micro': (0.002, 0.005),
        'small': (0.003, 0.008),
        'medium': (0.005, 0.012),
        'good': (0.008, 0.020)
    },
    time_horizons_range={
        'immediate': (1, 4),
        'short': (2, 8)
    },
    quality_weights_range={
        'speed_weight': (0.1, 0.5),
        'risk_weight': (0.2, 0.6),
        'profitability_weight': (0.1, 0.5)
    },
    bayesian_iterations=100,
    parallel_processing=True
)

# Visualization configuration
visualization_config = VisualizationConfig(
    output_format="png",
    output_dpi=300,
    figure_size=(15, 10),
    style_theme="seaborn-v0_8",
    interactive_charts=True,
    show_confidence_intervals=True,
    include_annotations=True
)

# Complete research configuration
research_config = ResearchConfig(
    workflows=[ResearchWorkflow.COMPLETE_PIPELINE],
    heuristic_config=heuristic_config,
    validation_config=validation_config,
    optimization_config=optimization_config,
    visualization_config=visualization_config,
    generate_reports=True,
    generate_visualizations=True,
    parallel_processing=True,
    output_dir="comprehensive_profit_labeling_research"
)
```

## 📊 Output and Results

### Analysis Reports

The framework generates comprehensive reports including:

1. **Heuristic Analysis Report**
   - Target/horizon combination effectiveness
   - Quality scoring consistency analysis
   - Composite score coherence evaluation
   - Parameter sensitivity findings

2. **Validation Report**
   - Label consistency metrics
   - Temporal stability analysis
   - Predictive validity assessment
   - Statistical significance tests
   - Bias detection results

3. **Optimization Report**
   - Best parameter configurations
   - Method comparison results
   - Convergence analysis
   - Performance improvement quantification

4. **Comprehensive Research Report**
   - Executive summary with key findings
   - Detailed results from all components
   - Actionable recommendations
   - Technical appendices

### Visualization Outputs

- **Static Charts**: High-quality matplotlib/seaborn visualizations
- **Interactive Dashboards**: Plotly-based interactive exploration tools
- **Publication-Ready**: Configurable styling and export formats (PNG, PDF, SVG)

### Data Exports

- **JSON**: Structured results for programmatic access
- **CSV**: Tabular data for spreadsheet analysis
- **Markdown**: Human-readable reports with formatting
- **HTML**: Interactive dashboards and web reports

## 🔧 Advanced Usage

### Custom Analysis Metrics

Extend the framework with custom analysis metrics:

```python
from research.profit_labeling.heuristic_analyzer import (
    HeuristicAnalyzer, 
    AnalysisMetric, 
    HeuristicAnalysisResult
)

class CustomHeuristicAnalyzer(HeuristicAnalyzer):
    def _analyze_custom_metric(self, labeled_data, config):
        # Implement custom analysis logic
        custom_score = self._calculate_custom_score(labeled_data)
        
        return HeuristicAnalysisResult(
            analysis_type=AnalysisMetric.CUSTOM,
            metric_value=custom_score,
            interpretation=f"Custom analysis shows score: {custom_score:.3f}",
            recommendations=self._generate_custom_recommendations(custom_score),
            metadata={'custom_analysis': True}
        )

# Use custom analyzer
custom_analyzer = CustomHeuristicAnalyzer()
custom_results = custom_analyzer.analyze_labeling_heuristics(market_data)
```

### Custom Optimization Objectives

Define custom optimization objectives:

```python
from research.profit_labeling.parameter_optimizer import ParameterOptimizer

class CustomParameterOptimizer(ParameterOptimizer):
    def _calculate_custom_objective_score(self, labeled_data, market_data):
        # Implement custom objective function
        # Example: Risk-adjusted returns with maximum drawdown penalty
        
        opportunities = labeled_data['overall_opportunity'].fillna(0)
        returns = market_data['close'].pct_change().shift(-1).fillna(0)
        
        # Strategy returns based on opportunities
        signals = (opportunities > opportunities.quantile(0.7)).astype(int)
        strategy_returns = signals * returns
        
        # Calculate risk-adjusted score with drawdown penalty
        if strategy_returns.std() > 0:
            sharpe = strategy_returns.mean() / strategy_returns.std()
            max_drawdown = self._calculate_max_drawdown(strategy_returns)
            
            # Penalize high drawdown
            drawdown_penalty = max(0, max_drawdown - 0.05) * 10  # Penalty for >5% drawdown
            risk_adjusted_score = sharpe - drawdown_penalty
            
            return max(0.0, min(2.0, risk_adjusted_score + 1.0)) / 2.0
        
        return 0.0
    
    def _calculate_max_drawdown(self, returns):
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())

# Use custom optimizer
custom_optimizer = CustomParameterOptimizer()
custom_result = custom_optimizer.optimize_parameters(market_data)
```

### Batch Processing

Process multiple datasets or configurations:

```python
import glob
from pathlib import Path

def batch_analyze_profit_labeling(data_directory: str, output_directory: str):
    """Analyze multiple datasets in batch."""
    
    data_files = glob.glob(f"{data_directory}/*.csv")
    runner = ResearchRunner()
    
    batch_results = {}
    
    for data_file in data_files:
        print(f"Processing {data_file}...")
        
        # Load data
        market_data = pd.read_csv(data_file)
        
        # Create output subdirectory
        file_name = Path(data_file).stem
        file_output_dir = Path(output_directory) / file_name
        
        # Update config for this file
        config = ResearchConfig(output_dir=str(file_output_dir))
        runner.config = config
        
        # Run analysis
        try:
            results = runner.run_research(market_data)
            batch_results[file_name] = results
            print(f"✅ Completed {file_name}")
        except Exception as e:
            print(f"❌ Failed {file_name}: {e}")
    
    return batch_results

# Run batch analysis
batch_results = batch_analyze_profit_labeling("data/", "batch_results/")
```

## 🔍 Integration with Existing Systems

### Integration with Ares Pipeline

```python
# In your existing training pipeline
from research.profit_labeling import run_quick_profit_labeling_analysis

def enhanced_market_analysis_step(market_data):
    """Enhanced market analysis with profit labeling research."""
    
    # Run existing market analysis
    existing_results = run_existing_market_analysis(market_data)
    
    # Add profit labeling research
    labeling_research = run_quick_profit_labeling_analysis(
        market_data, "heuristic"
    )
    
    # Combine results
    enhanced_results = {
        'existing_analysis': existing_results,
        'labeling_research': labeling_research,
        'recommendations': generate_combined_recommendations(
            existing_results, labeling_research
        )
    }
    
    return enhanced_results
```

### Integration with HMM Research

Compare profit labeling with HMM clustering approaches:

```python
from research.clusters import RegimeClusterer
from research.profit_labeling import HeuristicAnalyzer

def compare_labeling_vs_hmm(market_data):
    """Compare profit labeling effectiveness with HMM clustering."""
    
    # Run HMM clustering analysis
    hmm_clusterer = RegimeClusterer()
    hmm_results = hmm_clusterer.run_all_methods(market_data.values)
    
    # Run profit labeling analysis
    labeling_analyzer = HeuristicAnalyzer()
    labeling_results = labeling_analyzer.analyze_labeling_heuristics(market_data)
    
    # Compare effectiveness
    comparison = {
        'hmm_best_score': hmm_results.best_result.metrics.get('silhouette_score', 0),
        'labeling_predictive_power': np.mean([
            r.metadata.get('predictive_power', 0) 
            for r in labeling_results.values()
        ]),
        'recommendation': 'hmm' if hmm_results.best_result.metrics.get('silhouette_score', 0) > 0.5 else 'labeling'
    }
    
    return comparison
```

## 📚 Examples

### Complete Research Example

```python
#!/usr/bin/env python3
"""
Complete Multi-Horizon Profit Labeling Research Example

This example demonstrates the full research workflow for analyzing
and optimizing profit labeling heuristics.
"""

import pandas as pd
import numpy as np
from research.profit_labeling import *

def main():
    print("🔬 Multi-Horizon Profit Labeling Research Example")
    
    # 1. Load or generate sample data
    print("\n📊 Loading market data...")
    market_data = load_sample_market_data()  # Your data loading function
    print(f"   → Loaded {len(market_data)} samples")
    
    # 2. Run complete research pipeline
    print("\n🚀 Running complete research pipeline...")
    
    config = ResearchConfig(
        workflows=[ResearchWorkflow.COMPLETE_PIPELINE],
        generate_reports=True,
        generate_visualizations=True,
        output_dir="example_research_output"
    )
    
    runner = ResearchRunner(config)
    results = runner.run_research(market_data)
    
    # 3. Analyze results
    print("\n📈 Analyzing results...")
    complete_result = results['complete_pipeline']
    
    print(f"   → Research completed in {complete_result.execution_time:.1f}s")
    
    if complete_result.heuristic_results:
        print(f"   → Heuristic analysis: {len(complete_result.heuristic_results)} components analyzed")
        
        # Find best performing target/horizon combination
        target_results = {k: v for k, v in complete_result.heuristic_results.items() 
                         if 'effectiveness' in k}
        if target_results:
            best_target = max(target_results.items(), key=lambda x: x[1].metric_value)
            print(f"   → Best target/horizon: {best_target[0]} (score: {best_target[1].metric_value:.3f})")
    
    if complete_result.validation_results:
        significant_count = sum(1 for r in complete_result.validation_results.values() 
                              if r.is_significant)
        print(f"   → Validation: {significant_count}/{len(complete_result.validation_results)} tests significant")
    
    if complete_result.optimization_results:
        best_opt = max(complete_result.optimization_results.values(), 
                      key=lambda x: x.best_score)
        print(f"   → Optimization: Best score {best_opt.best_score:.4f} with {best_opt.method.value}")
    
    # 4. Generate summary recommendations
    print("\n💡 Key Recommendations:")
    recommendations = generate_research_recommendations(complete_result)
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    print(f"\n✅ Research completed! Results saved to: {config.output_dir}")
    print(f"   📄 View the comprehensive report: {config.output_dir}/comprehensive_research_report.md")
    
    return results

def load_sample_market_data():
    """Generate sample market data for demonstration."""
    dates = pd.date_range('2024-01-01', periods=3000, freq='5min')
    np.random.seed(42)
    
    # Generate realistic price data with trends and volatility clustering
    base_price = 100.0
    returns = np.random.normal(0.0001, 0.002, 3000)
    
    # Add volatility clustering
    vol_persistence = 0.9
    volatility = np.zeros(3000)
    volatility[0] = 0.002
    
    for i in range(1, 3000):
        volatility[i] = vol_persistence * volatility[i-1] + (1 - vol_persistence) * 0.002
        returns[i] = np.random.normal(0.0001, volatility[i])
    
    # Generate prices
    prices = [base_price]
    for ret in returns[:-1]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, 3000)
    }, index=dates)
    
    # Ensure OHLC consistency
    for i in range(len(data)):
        data.loc[data.index[i], 'high'] = max(data.iloc[i][['open', 'high', 'low', 'close']])
        data.loc[data.index[i], 'low'] = min(data.iloc[i][['open', 'high', 'low', 'close']])
    
    return data

def generate_research_recommendations(result):
    """Generate actionable recommendations from research results."""
    recommendations = []
    
    # Analyze heuristic results
    if result.heuristic_results:
        target_effectiveness = [r.metric_value for r in result.heuristic_results.values() 
                              if 'effectiveness' in str(r.analysis_type)]
        if target_effectiveness:
            avg_effectiveness = np.mean(target_effectiveness)
            if avg_effectiveness < 0.3:
                recommendations.append("Consider revising target/horizon combinations - low effectiveness detected")
            elif avg_effectiveness > 0.7:
                recommendations.append("Target/horizon combinations show good effectiveness - maintain current approach")
    
    # Analyze validation results
    if result.validation_results:
        significant_ratio = sum(1 for r in result.validation_results.values() 
                              if r.is_significant) / len(result.validation_results)
        if significant_ratio < 0.5:
            recommendations.append("Improve labeling methodology - low statistical significance detected")
        elif significant_ratio > 0.8:
            recommendations.append("Labeling shows strong statistical validity - suitable for ML training")
    
    # Analyze optimization results
    if result.optimization_results:
        best_scores = [r.best_score for r in result.optimization_results.values()]
        if max(best_scores) > 0.7:
            recommendations.append("Parameter optimization shows significant improvement potential")
        
        # Method-specific recommendations
        method_performance = {r.method.value: r.best_score for r in result.optimization_results.values()}
        best_method = max(method_performance.items(), key=lambda x: x[1])
        recommendations.append(f"Use {best_method[0]} for future parameter optimization")
    
    # Default recommendations
    if not recommendations:
        recommendations.extend([
            "Research completed successfully with acceptable results",
            "Monitor labeling performance in live trading environments",
            "Consider periodic re-optimization as market conditions change"
        ])
    
    return recommendations

if __name__ == '__main__':
    main()
```

## 🤝 Contributing

This research framework is designed to be extensible. You can contribute by:

1. **Adding New Analysis Metrics**: Implement additional heuristic analysis methods
2. **New Validation Tests**: Develop domain-specific validation measures  
3. **Optimization Algorithms**: Add specialized optimization approaches
4. **Visualization Types**: Create new chart types and dashboards
5. **Research Workflows**: Implement new research methodologies

## 🔗 Integration Points

This framework integrates seamlessly with:

- **Multi-Horizon Profit Labeler**: `src/training/steps/market_analysis/multi_horizon_profit_labeler.py`
- **HMM Clustering Research**: `src/research/clusters/`
- **Feature Engineering**: `src/feature_engineering_roadmap/`
- **Training Pipeline**: `src/training/steps/`
- **Logging System**: `src/utils/logger`

## 📄 License

This framework is part of the larger Ares trading system and follows the same licensing terms.

## 📞 Support

For questions or issues with the profit labeling research framework:

1. Check the examples in `research_runner.py`
2. Review the component documentation in each module
3. Examine the integration examples for usage patterns
4. Refer to the HMM clustering research framework for similar patterns

---

**Happy Profit Labeling Research! 🎯📊🤖**

> *"In God we trust. All others must bring data."* - W. Edwards Deming

This framework brings data-driven rigor to profit labeling analysis, ensuring your heuristics are not just intuitive, but statistically validated and optimally configured for maximum ML model performance.
