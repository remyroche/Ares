# Enhanced Code Mapping Pipeline - Complexity Integration

## Overview

To enhance the existing code mapping pipeline with our new comprehensive complexity analysis capabilities, we need to integrate the new `code_complexity` module with the existing `map_code_interactions.py` script.

## Required Imports to Add

### 1. Enhanced Complexity Analysis Imports

Add these imports to the top of `map_code_interactions.py`:

```python
# Enhanced complexity analysis imports
from code_complexity.complexity_pipeline import ComplexityPipeline, ComplexityMetrics, DirectoryMetrics
from code_complexity.config.complexity_config import ComplexityConfig
from code_complexity.analyzers.pyexamine_analyzer import PyExamineAnalyzer
from code_complexity.analyzers.radon_analyzer import RadonAnalyzer
from code_complexity.analyzers.xenon_analyzer import XenonAnalyzer
from code_complexity.utils.report_generator import ReportGenerator
from code_complexity.utils.file_utils import FileUtils
```

### 2. Additional Utility Imports

```python
# Enhanced reporting and visualization
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import numpy as np
```

## Enhanced CodeInteractionMapper Class

### New Methods to Add

```python
class CodeInteractionMapper:
    def __init__(self, project_root: str):
        # ... existing initialization ...
        
        # Initialize enhanced complexity pipeline
        self.complexity_pipeline = ComplexityPipeline()
        self.complexity_config = ComplexityConfig()
        
    def analyze_enhanced_complexity(self):
        """Enhanced complexity analysis using the new pipeline."""
        print("\n[5/7] Running enhanced complexity analysis...")
        
        # Run comprehensive complexity analysis
        complexity_results = self.complexity_pipeline.run_full_analysis(str(self.project_root))
        
        # Store results
        self.results["enhanced_complexity"] = complexity_results
        
        # Print enhanced summary
        self._print_enhanced_complexity_summary(complexity_results)
        
    def analyze_complexity_correlations(self):
        """Analyze correlations between complexity and other metrics."""
        print("\n[6/7] Analyzing complexity correlations...")
        
        correlations = self._calculate_complexity_correlations()
        self.results["complexity_correlations"] = correlations
        
        # Print correlation summary
        self._print_correlation_summary(correlations)
        
    def generate_complexity_visualizations(self):
        """Generate visualizations for complexity analysis."""
        print("\n[7/7] Generating complexity visualizations...")
        
        visualizations = self._create_complexity_visualizations()
        self.results["complexity_visualizations"] = visualizations
        
        print(f"  - Generated {len(visualizations)} visualization files")
```

### Enhanced Analysis Methods

```python
    def _print_enhanced_complexity_summary(self, results):
        """Print comprehensive complexity summary."""
        file_analysis = results.get('file_analysis', {})
        directory_analysis = results.get('directory_analysis', {})
        
        if file_analysis:
            scores = [m.get('combined_score', 0) for m in file_analysis.values() 
                     if m.get('combined_score') is not None]
            
            if scores:
                print(f"  - Files analyzed: {len(scores)}")
                print(f"  - Average complexity score: {sum(scores)/len(scores):.3f}")
                print(f"  - Highest complexity: {max(scores):.3f}")
                print(f"  - Lowest complexity: {min(scores):.3f}")
                
                # Tool-specific metrics
                pyexamine_scores = [m.get('pyexamine_score') for m in file_analysis.values() 
                                  if m.get('pyexamine_score') is not None]
                radon_cc_scores = [m.get('radon_cc') for m in file_analysis.values() 
                                 if m.get('radon_cc') is not None]
                xenon_scores = [m.get('xenon_score') for m in file_analysis.values() 
                              if m.get('xenon_score') is not None]
                
                if pyexamine_scores:
                    print(f"  - PyExamine average: {sum(pyexamine_scores)/len(pyexamine_scores):.3f}")
                if radon_cc_scores:
                    print(f"  - Radon CC average: {sum(radon_cc_scores)/len(radon_cc_scores):.2f}")
                if xenon_scores:
                    print(f"  - Xenon average: {sum(xenon_scores)/len(xenon_scores):.2f}")
                    
    def _calculate_complexity_correlations(self):
        """Calculate correlations between complexity and other metrics."""
        correlations = {}
        
        # Get complexity data
        enhanced_complexity = self.results.get("enhanced_complexity", {})
        file_analysis = enhanced_complexity.get('file_analysis', {})
        
        # Get other metrics
        dependencies = self.results.get("dependencies", {})
        call_graph = self.results.get("call_graph", {})
        
        # Calculate correlations
        for file_path, complexity_metrics in file_analysis.items():
            file_correlations = {}
            
            # Dependency correlation
            if file_path in dependencies.get('modules', {}):
                dep_count = len(dependencies['modules'][file_path].get('dependencies', []))
                complexity_score = complexity_metrics.get('combined_score', 0)
                file_correlations['dependency_count'] = dep_count
                file_correlations['complexity_score'] = complexity_score
                
            # Call graph correlation
            if file_path in call_graph.get('functions', {}):
                function_count = len(call_graph['functions'][file_path])
                file_correlations['function_count'] = function_count
                
            correlations[file_path] = file_correlations
            
        return correlations
        
    def _create_complexity_visualizations(self):
        """Create complexity visualization files."""
        visualizations = []
        
        # Complexity distribution histogram
        self._create_complexity_distribution_plot()
        visualizations.append("complexity_distribution.png")
        
        # Complexity vs dependencies scatter plot
        self._create_complexity_dependencies_plot()
        visualizations.append("complexity_dependencies.png")
        
        # Tool comparison heatmap
        self._create_tool_comparison_heatmap()
        visualizations.append("tool_comparison_heatmap.png")
        
        return visualizations
        
    def _create_complexity_distribution_plot(self):
        """Create complexity distribution histogram."""
        enhanced_complexity = self.results.get("enhanced_complexity", {})
        file_analysis = enhanced_complexity.get('file_analysis', {})
        
        scores = [m.get('combined_score', 0) for m in file_analysis.values() 
                 if m.get('combined_score') is not None]
        
        if scores:
            plt.figure(figsize=(10, 6))
            plt.hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.xlabel('Complexity Score')
            plt.ylabel('Number of Files')
            plt.title('Distribution of Code Complexity Scores')
            plt.grid(True, alpha=0.3)
            plt.savefig('complexity_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
```

## Enhanced Main Function

```python
def main():
    """Enhanced main function with complexity integration."""
    parser = argparse.ArgumentParser(description="Enhanced Code Interaction Mapping with Complexity Analysis")
    parser.add_argument("project_root", help="Root directory of the project")
    parser.add_argument("--output", "-o", default="code_interaction_map", help="Output file prefix")
    parser.add_argument("--include-complexity", action="store_true", 
                       help="Include enhanced complexity analysis")
    parser.add_argument("--complexity-tools", nargs="+", 
                       choices=["pyexamine", "radon", "xenon"],
                       default=["pyexamine", "radon", "xenon"],
                       help="Complexity analysis tools to use")
    parser.add_argument("--generate-visualizations", action="store_true",
                       help="Generate complexity visualization plots")
    
    args = parser.parse_args()
    
    # Initialize mapper
    mapper = CodeInteractionMapper(args.project_root)
    
    # Configure complexity tools if specified
    if args.include_complexity:
        mapper.complexity_config.enable_pyexamine = "pyexamine" in args.complexity_tools
        mapper.complexity_config.enable_radon = "radon" in args.complexity_tools
        mapper.complexity_config.enable_xenon = "xenon" in args.complexity_tools
    
    # Run analysis steps
    mapper.analyze_dependencies()
    mapper.analyze_call_graph()
    mapper.analyze_architecture()
    mapper.analyze_imports()
    
    # Enhanced complexity analysis
    if args.include_complexity:
        mapper.analyze_enhanced_complexity()
        mapper.analyze_complexity_correlations()
        
        if args.generate_visualizations:
            mapper.generate_complexity_visualizations()
    else:
        # Original complexity analysis
        mapper.analyze_complexity()
    
    mapper.analyze_dead_code()
    
    # Generate enhanced reports
    mapper.generate_enhanced_reports(args.output)
```

## Enhanced Report Generation

```python
    def generate_enhanced_reports(self, output_prefix: str):
        """Generate enhanced reports with complexity integration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate comprehensive JSON report
        json_file = f"{output_prefix}_enhanced_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nEnhanced analysis results saved to: {json_file}")
        
        # Generate complexity-specific reports
        if "enhanced_complexity" in self.results:
            complexity_reports = self._generate_complexity_reports(output_prefix, timestamp)
            print(f"Complexity reports generated: {', '.join(complexity_reports)}")
        
        # Generate HTML report with complexity integration
        html_file = f"{output_prefix}_enhanced_{timestamp}.html"
        self._generate_enhanced_html_report(html_file)
        print(f"Enhanced HTML report: {html_file}")
        
    def _generate_complexity_reports(self, output_prefix: str, timestamp: str):
        """Generate complexity-specific reports."""
        reports = []
        
        # Use the complexity pipeline's report generator
        report_generator = ReportGenerator(self.complexity_config)
        complexity_results = self.results.get("enhanced_complexity", {})
        
        # Generate all report formats
        report_generator.generate_reports(complexity_results)
        
        # List generated reports
        reports_dir = Path("reports")
        if reports_dir.exists():
            for report_file in reports_dir.glob(f"complexity_*_{timestamp}.*"):
                reports.append(str(report_file))
                
        return reports
```

## Configuration Enhancements

### Enhanced Configuration File

Create `config/enhanced_complexity_config.yaml`:

```yaml
# Enhanced Complexity Analysis Configuration
complexity_analysis:
  enabled: true
  tools:
    pyexamine:
      enabled: true
      timeout: 30
    radon:
      enabled: true
      timeout: 30
    xenon:
      enabled: true
      timeout: 30
  
  thresholds:
    complexity: 0.5
    cyclomatic_complexity: 10
    maintainability: 50
    
  visualization:
    enabled: true
    formats: ["png", "svg"]
    dpi: 300
    
  correlation_analysis:
    enabled: true
    metrics: ["dependencies", "call_graph", "architecture"]
```

## Integration Benefits

### 1. **Comprehensive Complexity Analysis**
- Multi-tool integration (PyExamine, Radon, Xenon)
- Per-file and per-directory analysis
- Combined scoring system

### 2. **Enhanced Correlations**
- Complexity vs. dependency relationships
- Complexity vs. function call patterns
- Architecture complexity analysis

### 3. **Rich Visualizations**
- Complexity distribution plots
- Tool comparison heatmaps
- Correlation scatter plots

### 4. **Improved Reporting**
- Multiple output formats
- Integrated complexity metrics
- Enhanced HTML reports

### 5. **Better Decision Making**
- Identify high-complexity areas
- Understand complexity drivers
- Prioritize refactoring efforts

## Usage Examples

```bash
# Run enhanced analysis with all complexity tools
python map_code_interactions.py /path/to/project --include-complexity

# Run with specific complexity tools
python map_code_interactions.py /path/to/project --include-complexity --complexity-tools radon xenon

# Generate visualizations
python map_code_interactions.py /path/to/project --include-complexity --generate-visualizations

# Full enhanced analysis
python map_code_interactions.py /path/to/project --include-complexity --generate-visualizations --output enhanced_analysis
```

This integration will significantly enhance the code mapping pipeline by providing comprehensive complexity analysis capabilities that complement the existing dependency, call graph, and architecture analysis.