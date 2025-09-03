"""
Dashboard Generator

Creates interactive HTML dashboards for code quality metrics.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import base64
from io import BytesIO


class DashboardGenerator:
    """Generates interactive HTML dashboards for code quality visualization."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the dashboard generator.
        
        Args:
            output_dir: Directory to save dashboards
        """
        self.output_dir = Path(output_dir) if output_dir else Path("code_quality/dashboards")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_quality_dashboard(self, 
                                 analysis_results: Dict[str, Any],
                                 project_name: str = "Code Quality Dashboard") -> str:
        """
        Generate a comprehensive code quality dashboard.
        
        Args:
            analysis_results: Complete analysis results
            project_name: Name of the project
            
        Returns:
            Path to generated HTML file
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Extract metrics
        metrics = self._extract_metrics(analysis_results)
        
        # Generate HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{project_name}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        {self._get_dashboard_css()}
    </style>
</head>
<body>
    <div class="dashboard">
        <header>
            <h1>{project_name}</h1>
            <p class="timestamp">Generated: {timestamp}</p>
        </header>
        
        <div class="metrics-grid">
            {self._generate_metric_cards(metrics)}
        </div>
        
        <div class="charts-container">
            <div class="chart-section">
                <h2>Code Complexity Overview</h2>
                <div id="complexity-chart"></div>
            </div>
            
            <div class="chart-section">
                <h2>Dependency Network</h2>
                <div id="dependency-network"></div>
            </div>
            
            <div class="chart-section">
                <h2>Quality Metrics Distribution</h2>
                <canvas id="metrics-distribution"></canvas>
            </div>
            
            <div class="chart-section">
                <h2>File Complexity Heatmap</h2>
                <div id="complexity-heatmap"></div>
            </div>
        </div>
        
        <div class="issues-container">
            <h2>Quality Issues</h2>
            {self._generate_issues_table(analysis_results)}
        </div>
        
        <div class="recommendations">
            <h2>Recommendations</h2>
            {self._generate_recommendations(analysis_results)}
        </div>
    </div>
    
    <script>
        {self._generate_dashboard_javascript(analysis_results)}
    </script>
</body>
</html>
"""
        
        # Save dashboard
        filename = f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(filepath)
    
    def generate_comparison_dashboard(self, 
                                    current_results: Dict[str, Any],
                                    previous_results: Dict[str, Any],
                                    project_name: str = "Code Quality Comparison") -> str:
        """
        Generate a dashboard comparing current vs previous results.
        
        Args:
            current_results: Current analysis results
            previous_results: Previous analysis results
            project_name: Name of the project
            
        Returns:
            Path to generated HTML file
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate changes
        changes = self._calculate_changes(current_results, previous_results)
        
        # Generate HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{project_name}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        {self._get_dashboard_css()}
        {self._get_comparison_css()}
    </style>
</head>
<body>
    <div class="dashboard">
        <header>
            <h1>{project_name}</h1>
            <p class="timestamp">Generated: {timestamp}</p>
        </header>
        
        <div class="comparison-summary">
            <h2>Overall Changes</h2>
            {self._generate_change_summary(changes)}
        </div>
        
        <div class="charts-container">
            <div class="chart-section">
                <h2>Complexity Trends</h2>
                <div id="complexity-trends"></div>
            </div>
            
            <div class="chart-section">
                <h2>Quality Score Changes</h2>
                <div id="quality-changes"></div>
            </div>
            
            <div class="chart-section">
                <h2>Issue Count Comparison</h2>
                <div id="issue-comparison"></div>
            </div>
        </div>
        
        <div class="detailed-changes">
            <h2>Detailed Changes</h2>
            {self._generate_detailed_changes(changes)}
        </div>
    </div>
    
    <script>
        {self._generate_comparison_javascript(current_results, previous_results, changes)}
    </script>
</body>
</html>
"""
        
        # Save dashboard
        filename = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(filepath)
    
    def _get_dashboard_css(self) -> str:
        """Get the base CSS for dashboards."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #f5f7fa;
            color: #2d3748;
            line-height: 1.6;
        }
        
        .dashboard {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            margin-bottom: 30px;
        }
        
        header h1 {
            font-size: 32px;
            font-weight: 700;
            color: #1a202c;
            margin-bottom: 10px;
        }
        
        .timestamp {
            color: #718096;
            font-size: 14px;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .metric-card {
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        }
        
        .metric-card h3 {
            font-size: 14px;
            font-weight: 500;
            color: #718096;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
        }
        
        .metric-value {
            font-size: 36px;
            font-weight: 700;
            color: #1a202c;
            line-height: 1;
        }
        
        .metric-change {
            margin-top: 10px;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .metric-change.positive {
            color: #48bb78;
        }
        
        .metric-change.negative {
            color: #f56565;
        }
        
        .charts-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .chart-section {
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        
        .chart-section h2 {
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 20px;
            color: #2d3748;
        }
        
        .issues-container {
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            margin-bottom: 30px;
        }
        
        .issues-container h2 {
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 20px;
            color: #2d3748;
        }
        
        .issues-table {
            width: 100%;
            border-collapse: collapse;
        }
        
        .issues-table th {
            background: #f7fafc;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            font-size: 14px;
            color: #4a5568;
            border-bottom: 2px solid #e2e8f0;
        }
        
        .issues-table td {
            padding: 12px;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .severity-high {
            color: #f56565;
            font-weight: 600;
        }
        
        .severity-medium {
            color: #ed8936;
            font-weight: 500;
        }
        
        .severity-low {
            color: #4299e1;
        }
        
        .recommendations {
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        
        .recommendations h2 {
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 20px;
            color: #2d3748;
        }
        
        .recommendation-item {
            padding: 15px;
            margin-bottom: 15px;
            background: #f7fafc;
            border-left: 4px solid #4299e1;
            border-radius: 4px;
        }
        
        .recommendation-item h4 {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 5px;
            color: #2d3748;
        }
        
        .recommendation-item p {
            color: #4a5568;
            font-size: 14px;
        }
        """
    
    def _get_comparison_css(self) -> str:
        """Get additional CSS for comparison dashboards."""
        return """
        .comparison-summary {
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            margin-bottom: 30px;
        }
        
        .change-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .change-item {
            padding: 15px;
            background: #f7fafc;
            border-radius: 8px;
            text-align: center;
        }
        
        .change-item h4 {
            font-size: 14px;
            color: #718096;
            margin-bottom: 10px;
        }
        
        .change-value {
            font-size: 24px;
            font-weight: 700;
        }
        
        .change-value.improved {
            color: #48bb78;
        }
        
        .change-value.degraded {
            color: #f56565;
        }
        
        .change-value.neutral {
            color: #718096;
        }
        
        .detailed-changes {
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        """
    
    def _extract_metrics(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from analysis results."""
        metrics = {
            'total_files': 0,
            'total_functions': 0,
            'average_complexity': 0,
            'total_issues': 0,
            'code_coverage': 0,
            'maintainability_index': 0
        }
        
        # Extract from different analyzers
        if 'complexity' in analysis_results:
            complexity = analysis_results['complexity']
            metrics['total_files'] = len(complexity.get('files', {}))
            metrics['average_complexity'] = complexity.get('average_complexity', 0)
        
        if 'call_graph' in analysis_results:
            call_graph = analysis_results['call_graph']
            metrics['total_functions'] = len(call_graph.get('functions', {}))
        
        if 'issues' in analysis_results:
            metrics['total_issues'] = len(analysis_results['issues'])
        
        return metrics
    
    def _generate_metric_cards(self, metrics: Dict[str, Any]) -> str:
        """Generate HTML for metric cards."""
        cards_html = ""
        
        metric_configs = [
            ('total_files', 'Total Files', '📁'),
            ('total_functions', 'Total Functions', '🔧'),
            ('average_complexity', 'Avg Complexity', '📊'),
            ('total_issues', 'Total Issues', '⚠️'),
            ('maintainability_index', 'Maintainability', '✨')
        ]
        
        for key, label, icon in metric_configs:
            value = metrics.get(key, 0)
            if isinstance(value, float):
                value_str = f"{value:.2f}"
            else:
                value_str = str(value)
            
            cards_html += f"""
            <div class="metric-card">
                <h3>{icon} {label}</h3>
                <div class="metric-value">{value_str}</div>
            </div>
            """
        
        return cards_html
    
    def _generate_issues_table(self, analysis_results: Dict[str, Any]) -> str:
        """Generate HTML table for issues."""
        issues = analysis_results.get('issues', [])
        
        if not issues:
            return "<p>No issues found! 🎉</p>"
        
        table_html = """
        <table class="issues-table">
            <thead>
                <tr>
                    <th>File</th>
                    <th>Issue</th>
                    <th>Severity</th>
                    <th>Line</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for issue in issues[:20]:  # Show first 20 issues
            severity_class = f"severity-{issue.get('severity', 'low').lower()}"
            table_html += f"""
                <tr>
                    <td>{self._format_path(issue.get('file', 'Unknown'))}</td>
                    <td>{issue.get('message', 'No description')}</td>
                    <td class="{severity_class}">{issue.get('severity', 'Low')}</td>
                    <td>{issue.get('line', '-')}</td>
                </tr>
            """
        
        table_html += """
            </tbody>
        </table>
        """
        
        if len(issues) > 20:
            table_html += f"<p style='margin-top: 10px; color: #718096;'>... and {len(issues) - 20} more issues</p>"
        
        return table_html
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> str:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Check complexity
        if 'complexity' in analysis_results:
            avg_complexity = analysis_results['complexity'].get('average_complexity', 0)
            if avg_complexity > 10:
                recommendations.append({
                    'title': 'Reduce Code Complexity',
                    'description': f'Average complexity is {avg_complexity:.1f}. Consider refactoring complex functions.'
                })
        
        # Check for circular dependencies
        if 'dependencies' in analysis_results:
            circular = analysis_results['dependencies'].get('circular_imports', [])
            if circular:
                recommendations.append({
                    'title': 'Resolve Circular Dependencies',
                    'description': f'Found {len(circular)} circular dependencies that should be resolved.'
                })
        
        # Check for isolated functions
        if 'call_graph' in analysis_results:
            isolated = analysis_results['call_graph'].get('isolated_functions', [])
            if isolated:
                recommendations.append({
                    'title': 'Review Isolated Functions',
                    'description': f'Found {len(isolated)} isolated functions that might be unused.'
                })
        
        # Generate HTML
        if not recommendations:
            recommendations.append({
                'title': 'Great Job!',
                'description': 'Your code quality looks good. Keep up the excellent work!'
            })
        
        html = ""
        for rec in recommendations:
            html += f"""
            <div class="recommendation-item">
                <h4>{rec['title']}</h4>
                <p>{rec['description']}</p>
            </div>
            """
        
        return html
    
    def _generate_dashboard_javascript(self, analysis_results: Dict[str, Any]) -> str:
        """Generate JavaScript for interactive charts."""
        return f"""
        // Complexity Chart
        const complexityData = {json.dumps(self._prepare_complexity_data(analysis_results))};
        
        Plotly.newPlot('complexity-chart', complexityData.data, complexityData.layout);
        
        // Dependency Network
        const dependencyData = {json.dumps(self._prepare_dependency_data(analysis_results))};
        
        Plotly.newPlot('dependency-network', dependencyData.data, dependencyData.layout);
        
        // Metrics Distribution
        const ctx = document.getElementById('metrics-distribution').getContext('2d');
        const metricsChart = new Chart(ctx, {json.dumps(self._prepare_metrics_chart(analysis_results))});
        
        // Complexity Heatmap
        const heatmapData = {json.dumps(self._prepare_heatmap_data(analysis_results))};
        
        Plotly.newPlot('complexity-heatmap', heatmapData.data, heatmapData.layout);
        """
    
    def _prepare_complexity_data(self, analysis_results: Dict[str, Any]) -> Dict:
        """Prepare data for complexity chart."""
        complexity = analysis_results.get('complexity', {})
        files = list(complexity.get('files', {}).keys())[:20]
        values = [complexity['files'][f].get('complexity', 0) for f in files]
        
        return {
            'data': [{
                'x': [Path(f).name for f in files],
                'y': values,
                'type': 'bar',
                'marker': {'color': 'rgba(66, 153, 225, 0.8)'}
            }],
            'layout': {
                'title': '',
                'xaxis': {'title': 'Files'},
                'yaxis': {'title': 'Complexity Score'},
                'height': 400
            }
        }
    
    def _prepare_dependency_data(self, analysis_results: Dict[str, Any]) -> Dict:
        """Prepare data for dependency network."""
        dependencies = analysis_results.get('dependencies', {})
        modules = dependencies.get('modules', {})
        
        # Create nodes and edges
        nodes = list(modules.keys())[:30]
        edges = []
        
        for module in nodes:
            for dep in modules.get(module, {}).get('dependencies', []):
                if dep in nodes:
                    edges.append({'source': nodes.index(module), 'target': nodes.index(dep)})
        
        # Create network visualization data
        edge_trace = {
            'x': [],
            'y': [],
            'mode': 'lines',
            'line': {'width': 0.5, 'color': '#888'},
            'hoverinfo': 'none'
        }
        
        node_trace = {
            'x': [],
            'y': [],
            'mode': 'markers+text',
            'text': nodes,
            'textposition': 'top center',
            'marker': {
                'size': 10,
                'color': 'rgba(66, 153, 225, 0.8)'
            }
        }
        
        # Simple circular layout
        import math
        n = len(nodes)
        for i in range(n):
            angle = 2 * math.pi * i / n
            node_trace['x'].append(math.cos(angle))
            node_trace['y'].append(math.sin(angle))
        
        return {
            'data': [edge_trace, node_trace],
            'layout': {
                'showlegend': False,
                'hovermode': 'closest',
                'height': 500,
                'xaxis': {'showgrid': False, 'zeroline': False, 'showticklabels': False},
                'yaxis': {'showgrid': False, 'zeroline': False, 'showticklabels': False}
            }
        }
    
    def _prepare_metrics_chart(self, analysis_results: Dict[str, Any]) -> Dict:
        """Prepare data for metrics distribution chart."""
        return {
            'type': 'doughnut',
            'data': {
                'labels': ['High Complexity', 'Medium Complexity', 'Low Complexity'],
                'datasets': [{
                    'data': [10, 30, 60],  # Example data
                    'backgroundColor': [
                        'rgba(245, 101, 101, 0.8)',
                        'rgba(237, 137, 54, 0.8)',
                        'rgba(72, 187, 120, 0.8)'
                    ]
                }]
            },
            'options': {
                'responsive': True,
                'maintainAspectRatio': False
            }
        }
    
    def _prepare_heatmap_data(self, analysis_results: Dict[str, Any]) -> Dict:
        """Prepare data for complexity heatmap."""
        complexity = analysis_results.get('complexity', {})
        files = list(complexity.get('files', {}).keys())[:15]
        metrics = ['complexity', 'lines', 'functions']
        
        z_data = []
        for metric in metrics:
            row = []
            for file in files:
                value = complexity['files'].get(file, {}).get(metric, 0)
                row.append(value)
            z_data.append(row)
        
        return {
            'data': [{
                'z': z_data,
                'x': [Path(f).name for f in files],
                'y': metrics,
                'type': 'heatmap',
                'colorscale': 'RdYlGn',
                'reversescale': True
            }],
            'layout': {
                'height': 400,
                'xaxis': {'title': 'Files'},
                'yaxis': {'title': 'Metrics'}
            }
        }
    
    def _calculate_changes(self, current: Dict[str, Any], previous: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate changes between current and previous results."""
        changes = {
            'complexity': {
                'current': current.get('complexity', {}).get('average_complexity', 0),
                'previous': previous.get('complexity', {}).get('average_complexity', 0)
            },
            'issues': {
                'current': len(current.get('issues', [])),
                'previous': len(previous.get('issues', []))
            },
            'files': {
                'current': len(current.get('complexity', {}).get('files', {})),
                'previous': len(previous.get('complexity', {}).get('files', {}))
            }
        }
        
        # Calculate percentages
        for metric in changes:
            curr = changes[metric]['current']
            prev = changes[metric]['previous']
            if prev > 0:
                changes[metric]['change'] = ((curr - prev) / prev) * 100
            else:
                changes[metric]['change'] = 0
        
        return changes
    
    def _generate_change_summary(self, changes: Dict[str, Any]) -> str:
        """Generate HTML for change summary."""
        html = '<div class="change-grid">'
        
        for metric, data in changes.items():
            change = data.get('change', 0)
            if change > 0:
                change_class = 'degraded' if metric in ['complexity', 'issues'] else 'improved'
                symbol = '↑'
            elif change < 0:
                change_class = 'improved' if metric in ['complexity', 'issues'] else 'degraded'
                symbol = '↓'
            else:
                change_class = 'neutral'
                symbol = '→'
            
            html += f"""
            <div class="change-item">
                <h4>{metric.title()}</h4>
                <div class="change-value {change_class}">
                    {symbol} {abs(change):.1f}%
                </div>
            </div>
            """
        
        html += '</div>'
        return html
    
    def _generate_detailed_changes(self, changes: Dict[str, Any]) -> str:
        """Generate detailed changes table."""
        # This would show file-by-file changes
        return """
        <p>Detailed file-by-file analysis would appear here, showing:</p>
        <ul>
            <li>Files with increased complexity</li>
            <li>New issues introduced</li>
            <li>Fixed issues</li>
            <li>Improved functions</li>
        </ul>
        """
    
    def _generate_comparison_javascript(self, current: Dict[str, Any], 
                                      previous: Dict[str, Any], 
                                      changes: Dict[str, Any]) -> str:
        """Generate JavaScript for comparison charts."""
        return f"""
        // Complexity Trends
        const trendsData = {{
            data: [{{
                x: ['Previous', 'Current'],
                y: [{previous.get('complexity', {}).get('average_complexity', 0)}, 
                    {current.get('complexity', {}).get('average_complexity', 0)}],
                type: 'scatter',
                mode: 'lines+markers',
                marker: {{size: 10}}
            }}],
            layout: {{
                title: '',
                xaxis: {{title: 'Version'}},
                yaxis: {{title: 'Average Complexity'}},
                height: 400
            }}
        }};
        
        Plotly.newPlot('complexity-trends', trendsData.data, trendsData.layout);
        
        // Quality Changes
        const qualityData = {{
            data: [{{
                x: ['Files', 'Functions', 'Issues'],
                y: [{changes['files']['change']}, 0, {changes['issues']['change']}],
                type: 'bar',
                marker: {{
                    color: ['rgba(72, 187, 120, 0.8)', 'rgba(66, 153, 225, 0.8)', 'rgba(245, 101, 101, 0.8)']
                }}
            }}],
            layout: {{
                title: '',
                xaxis: {{title: 'Metric'}},
                yaxis: {{title: 'Change (%)'}},
                height: 400
            }}
        }};
        
        Plotly.newPlot('quality-changes', qualityData.data, qualityData.layout);
        
        // Issue Comparison
        const issueData = {{
            data: [{{
                labels: ['High', 'Medium', 'Low'],
                values: [5, 15, 30],  // Example data
                type: 'pie',
                name: 'Previous',
                domain: {{x: [0, 0.48]}}
            }}, {{
                labels: ['High', 'Medium', 'Low'],
                values: [3, 20, 35],  // Example data
                type: 'pie',
                name: 'Current',
                domain: {{x: [0.52, 1]}}
            }}],
            layout: {{
                title: '',
                height: 400,
                annotations: [
                    {{text: 'Previous', x: 0.18, y: 0, font: {{size: 20}}, showarrow: false}},
                    {{text: 'Current', x: 0.82, y: 0, font: {{size: 20}}, showarrow: false}}
                ]
            }}
        }};
        
        Plotly.newPlot('issue-comparison', issueData.data, issueData.layout);
        """
    
    def _format_path(self, path: str, max_length: int = 40) -> str:
        """Format file path for display."""
        if len(path) <= max_length:
            return path
        
        parts = path.split('/')
        if len(parts) > 2:
            return f".../{'/'.join(parts[-2:])}"
        return f"...{path[-max_length:]}"