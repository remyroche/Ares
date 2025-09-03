# Code Quality Visualization Guide

This guide explains how to use the visual mapping tools in the code_quality directory to create comprehensive visualizations of your codebase.

## Overview

The visualization tools provide multiple ways to visualize code quality metrics:

1. **Dependency Graphs** - Visualize module dependencies and relationships
2. **Complexity Heatmaps** - Identify complex areas of code
3. **Function Call Networks** - Understand function relationships
4. **Interactive Dashboards** - Explore metrics dynamically
5. **Module Interaction Matrices** - See module coupling at a glance

## Installation

First, install the required visualization dependencies:

```bash
cd code_quality
pip install -r requirements.txt
```

## Quick Start

### Generate Sample Visualizations

To see what the tools can do, run the demo:

```bash
python examples/visual_mapping_demo.py
```

This creates sample visualizations in `code_quality/demo_visualizations/`.

### Visualize Your Code

1. **Using the enhanced visualization script:**

```bash
# Generate visualizations from analysis results
python visualize_interactions.py --input analysis_results.json

# Or generate sample visualizations
python visualize_interactions.py --sample
```

2. **Using the code interaction mapper:**

```bash
# Map all interactions in your project
python map_code_interactions.py --project-root /path/to/your/project
```

## Visualization Types

### 1. Dependency Graphs

Shows how modules depend on each other:

- **Dependency Network**: Overall module relationships
- **Circular Dependencies**: Highlights circular imports
- **Module Hierarchy**: Layered view of dependencies

```python
from visualizers import DependencyGraphVisualizer

dep_viz = DependencyGraphVisualizer()
fig, metadata = dep_viz.create_dependency_graph(dependencies)
dep_viz.save_figure(fig, "my_dependencies")
```

### 2. Complexity Heatmaps

Visualizes code complexity metrics:

- **Heatmap**: Shows complexity across files and metrics
- **Treemap**: Size represents lines of code, color shows complexity
- **Bubble Chart**: Multi-dimensional complexity view

```python
from visualizers import ComplexityHeatmapVisualizer

complexity_viz = ComplexityHeatmapVisualizer()
fig, metadata = complexity_viz.create_complexity_heatmap(complexity_data)
complexity_viz.save_figure(fig, "complexity_analysis")
```

### 3. Function Call Networks

Maps function relationships:

- **Call Graph**: Shows which functions call others
- **Entry/Exit Points**: Identifies key functions
- **Hub Functions**: Shows central functions

```python
from visualizers import InteractionNetworkVisualizer

network_viz = InteractionNetworkVisualizer()
fig, metadata = network_viz.create_function_call_network(call_graph)
network_viz.save_figure(fig, "function_network")
```

### 4. Interactive Visualizations

Creates interactive HTML visualizations:

- **Interactive Networks**: Explore relationships dynamically
- **Dashboards**: Comprehensive metrics overview
- **Comparison Views**: Track changes over time

```python
from visualizers import DashboardGenerator

dashboard_gen = DashboardGenerator()
dashboard_file = dashboard_gen.generate_quality_dashboard(analysis_results)
```

## Output Formats

All visualizations are saved in multiple formats:

- **PNG**: High-quality raster images
- **PDF**: Vector format for publications
- **SVG**: Scalable vector graphics for web
- **HTML**: Interactive visualizations

## Customization

### Colors and Themes

Modify color schemes in the visualizers:

```python
# Use different colormaps
colors = viz.create_color_map(values, cmap_name='viridis')
```

### Layout Options

Different layout algorithms for networks:

```python
# Spring layout (default)
pos = nx.spring_layout(graph)

# Hierarchical layout
pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')

# Circular layout
pos = nx.circular_layout(graph)
```

### Dashboard Customization

Modify dashboard appearance by editing CSS in `dashboard_generator.py`.

## Integration with CI/CD

Generate visualizations in your CI pipeline:

```yaml
# Example GitHub Actions workflow
- name: Generate Code Quality Visualizations
  run: |
    python code_quality/map_code_interactions.py
    python code_quality/visualize_interactions.py --input results.json
```

## Examples

### Complete Analysis with Visualizations

```python
#!/usr/bin/env python3
"""Run complete code analysis with visualizations."""

from map_code_interactions import CodeInteractionMapper
from visualize_interactions import visualize_code_interactions

# Analyze code
mapper = CodeInteractionMapper("/path/to/project")
results = mapper.run()

# Generate visualizations
visual_files = visualize_code_interactions(results)
print(f"Generated {len(visual_files)} visualizations")
```

### Custom Visualization Pipeline

```python
#!/usr/bin/env python3
"""Custom visualization pipeline."""

from visualizers import *

# Load your analysis results
import json
with open('analysis_results.json', 'r') as f:
    results = json.load(f)

# Create custom visualizations
output_dir = "my_visualizations"

# 1. Dependencies only
dep_viz = DependencyGraphVisualizer(output_dir)
if 'dependencies' in results:
    fig, _ = dep_viz.create_dependency_graph(
        results['dependencies']['modules'],
        "My Project Dependencies"
    )
    dep_viz.save_figure(fig, "dependencies")

# 2. Complexity focus
complexity_viz = ComplexityHeatmapVisualizer(output_dir)
if 'complexity' in results:
    # Create multiple complexity views
    for metric in ['cyclomatic_complexity', 'maintainability_index']:
        fig = complexity_viz.create_treemap_visualization(
            results['complexity']['files'],
            metric,
            f"{metric.title()} Treemap"
        )
        complexity_viz.save_figure(fig, f"{metric}_treemap")

# 3. Interactive dashboard
dashboard_gen = DashboardGenerator(output_dir)
dashboard_file = dashboard_gen.generate_quality_dashboard(
    results,
    "My Project Quality Dashboard"
)
```

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Install all requirements with `pip install -r requirements.txt`
2. **Graphviz not found**: Install system graphviz: `apt-get install graphviz` or `brew install graphviz`
3. **Large codebases**: Use sampling or filtering for better performance

### Performance Tips

- For large projects, limit the number of files analyzed
- Use the `--exclude` flag to skip unnecessary directories
- Generate visualizations incrementally

## Best Practices

1. **Regular Generation**: Generate visualizations regularly to track trends
2. **Version Control**: Keep visualizations in a separate directory
3. **Documentation**: Include key visualizations in your project documentation
4. **Review Process**: Use visualizations in code reviews to identify issues

## Advanced Usage

### Batch Processing

Process multiple projects:

```bash
for project in project1 project2 project3; do
    python map_code_interactions.py --project-root $project
    python visualize_interactions.py --input $project/results.json
done
```

### Historical Analysis

Track complexity over time:

```python
# Generate timeline visualizations
historical_data = load_historical_results()
fig = complexity_viz.create_complexity_timeline(historical_data)
```

### Custom Metrics

Add your own metrics to visualizations:

```python
# Extend the visualizers
class MyCustomVisualizer(CodeVisualizer):
    def create_custom_visualization(self, data):
        # Your visualization logic
        pass
```

## Conclusion

The visual mapping tools provide powerful ways to understand your codebase. Use them to:

- Identify architectural issues
- Find complexity hotspots
- Understand dependencies
- Track quality over time
- Communicate with stakeholders

For more examples, see the `examples/` directory.