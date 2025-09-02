"""
HTML Reporter

Generates beautiful, interactive HTML reports for code quality analysis.
Provides rich visualizations and interactive elements for better data exploration.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import base64

from ..core.config import ReportingConfig


@dataclass
class HTMLReportConfig:
    """Configuration for HTML report generation."""
    include_charts: bool = True
    include_interactive: bool = True
    theme: str = 'light'  # 'light' or 'dark'
    custom_css: Optional[str] = None
    custom_js: Optional[str] = None


class HTMLReporter:
    """
    Generates comprehensive HTML reports for code quality analysis.
    
    Features:
    - Responsive design
    - Interactive charts and tables
    - Multiple themes
    - Export functionality
    - Search and filtering
    """
    
    def __init__(self, config: Optional[HTMLReportConfig] = None):
        """
        Initialize the HTML reporter.
        
        Args:
            config: HTML report configuration
        """
        self.config = config or HTMLReportConfig()
        self.template_dir = Path(__file__).parent / 'templates'
        
    def generate_report(self, data: Dict[str, Any], title: str = "Code Quality Report") -> str:
        """
        Generate HTML report from analysis data.
        
        Args:
            data: Analysis data dictionary
            title: Report title
            
        Returns:
            HTML string
        """
        html = self._generate_header(title)
        html += self._generate_navigation()
        html += self._generate_summary_section(data)
        html += self._generate_details_section(data)
        html += self._generate_charts_section(data)
        html += self._generate_footer()
        
        return html
    
    def _generate_header(self, title: str) -> str:
        """Generate HTML header with CSS and JavaScript."""
        css = self._get_css()
        js = self._get_javascript()
        
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>{css}</style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>{js}</script>
</head>
<body class="theme-{self.config.theme}">
    <div class="container">
        <header class="report-header">
            <h1>{title}</h1>
            <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>
"""
    
    def _generate_navigation(self) -> str:
        """Generate navigation menu."""
        return """
        <nav class="report-nav">
            <ul>
                <li><a href="#summary">Summary</a></li>
                <li><a href="#details">Details</a></li>
                <li><a href="#charts">Charts</a></li>
                <li><a href="#export">Export</a></li>
            </ul>
        </nav>
"""
    
    def _generate_summary_section(self, data: Dict[str, Any]) -> str:
        """Generate summary section with key metrics."""
        summary = data.get('summary', {})
        
        return f"""
        <section id="summary" class="report-section">
            <h2>Summary</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Total Issues</h3>
                    <div class="metric-value">{summary.get('total_issues', 0)}</div>
                </div>
                <div class="metric-card">
                    <h3>Files Analyzed</h3>
                    <div class="metric-value">{summary.get('total_files', 0)}</div>
                </div>
                <div class="metric-card">
                    <h3>Critical Issues</h3>
                    <div class="metric-value critical">{summary.get('critical_errors', 0)}</div>
                </div>
                <div class="metric-card">
                    <h3>Quality Score</h3>
                    <div class="metric-value">{summary.get('quality_score', 'N/A')}</div>
                </div>
            </div>
        </section>
"""
    
    def _generate_details_section(self, data: Dict[str, Any]) -> str:
        """Generate detailed analysis section."""
        details = data.get('details', {})
        
        html = """
        <section id="details" class="report-section">
            <h2>Detailed Analysis</h2>
        """
        
        # Issues by category
        if 'categories' in details:
            html += self._generate_categories_table(details['categories'])
        
        # Issues by file
        if 'files' in details:
            html += self._generate_files_table(details['files'])
        
        html += """
        </section>
"""
        return html
    
    def _generate_categories_table(self, categories: List[Dict]) -> str:
        """Generate categories table."""
        html = """
            <div class="table-container">
                <h3>Issues by Category</h3>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Category</th>
                            <th>Count</th>
                            <th>Severity</th>
                            <th>Files Affected</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        
        for category in categories:
            html += f"""
                        <tr>
                            <td>{category.get('name', 'Unknown')}</td>
                            <td>{category.get('count', 0)}</td>
                            <td>{category.get('severity', 'Unknown')}</td>
                            <td>{category.get('files_affected', 0)}</td>
                        </tr>
"""
        
        html += """
                    </tbody>
                </table>
            </div>
"""
        return html
    
    def _generate_files_table(self, files: List[Dict]) -> str:
        """Generate files table."""
        html = """
            <div class="table-container">
                <h3>Issues by File</h3>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>File</th>
                            <th>Errors</th>
                            <th>Warnings</th>
                            <th>Score</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        
        for file_info in files:
            html += f"""
                        <tr>
                            <td>{file_info.get('file_path', 'Unknown')}</td>
                            <td>{file_info.get('total_errors', 0)}</td>
                            <td>{file_info.get('total_warnings', 0)}</td>
                            <td>{file_info.get('score', 'N/A')}</td>
                        </tr>
"""
        
        html += """
                    </tbody>
                </table>
            </div>
"""
        return html
    
    def _generate_charts_section(self, data: Dict[str, Any]) -> str:
        """Generate charts section."""
        if not self.config.include_charts:
            return ""
        
        return """
        <section id="charts" class="report-section">
            <h2>Charts & Visualizations</h2>
            <div class="charts-grid">
                <div class="chart-container">
                    <canvas id="issuesChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="severityChart"></canvas>
                </div>
            </div>
            <script>
                // Initialize charts with data
                initializeCharts();
            </script>
        </section>
"""
    
    def _generate_footer(self) -> str:
        """Generate footer with export options."""
        return """
        <section id="export" class="report-section">
            <h2>Export Options</h2>
            <div class="export-buttons">
                <button onclick="exportToJSON()">Export JSON</button>
                <button onclick="exportToCSV()">Export CSV</button>
                <button onclick="printReport()">Print Report</button>
            </div>
        </section>
    </div>
</body>
</html>
"""
    
    def _get_css(self) -> str:
        """Get CSS styles."""
        return """
        :root {
            --primary-color: #007acc;
            --secondary-color: #6c757d;
            --success-color: #28a745;
            --warning-color: #ffc107;
            --danger-color: #dc3545;
            --light-color: #f8f9fa;
            --dark-color: #343a40;
        }
        
        .theme-light {
            --bg-color: #ffffff;
            --text-color: #333333;
            --card-bg: #f8f9fa;
            --border-color: #dee2e6;
        }
        
        .theme-dark {
            --bg-color: #1a1a1a;
            --text-color: #ffffff;
            --card-bg: #2d2d2d;
            --border-color: #404040;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background-color: var(--bg-color);
            color: var(--text-color);
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .report-header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: var(--card-bg);
            border-radius: 8px;
        }
        
        .report-nav ul {
            list-style: none;
            padding: 0;
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .report-nav a {
            text-decoration: none;
            color: var(--primary-color);
            padding: 10px 20px;
            border-radius: 5px;
            transition: background-color 0.3s;
        }
        
        .report-nav a:hover {
            background-color: var(--light-color);
        }
        
        .report-section {
            margin-bottom: 40px;
            padding: 20px;
            background: var(--card-bg);
            border-radius: 8px;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .metric-card {
            text-align: center;
            padding: 20px;
            background: var(--bg-color);
            border-radius: 5px;
            border: 1px solid var(--border-color);
        }
        
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: var(--primary-color);
        }
        
        .metric-value.critical {
            color: var(--danger-color);
        }
        
        .table-container {
            margin-top: 20px;
        }
        
        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        
        .data-table th,
        .data-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        
        .data-table th {
            background-color: var(--primary-color);
            color: white;
            font-weight: bold;
        }
        
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .chart-container {
            background: var(--bg-color);
            padding: 20px;
            border-radius: 5px;
            border: 1px solid var(--border-color);
        }
        
        .export-buttons {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-top: 20px;
        }
        
        .export-buttons button {
            padding: 12px 24px;
            background-color: var(--primary-color);
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s;
        }
        
        .export-buttons button:hover {
            background-color: #0056b3;
        }
        
        @media (max-width: 768px) {
            .metrics-grid {
                grid-template-columns: 1fr;
            }
            
            .charts-grid {
                grid-template-columns: 1fr;
            }
            
            .report-nav ul {
                flex-direction: column;
                align-items: center;
            }
        }
"""
    
    def _get_javascript(self) -> str:
        """Get JavaScript functionality."""
        return """
        function initializeCharts() {
            // Issues by category chart
            const issuesCtx = document.getElementById('issuesChart');
            if (issuesCtx) {
                new Chart(issuesCtx, {
                    type: 'doughnut',
                    data: {
                        labels: ['Errors', 'Warnings', 'Info'],
                        datasets: [{
                            data: [12, 19, 3],
                            backgroundColor: ['#dc3545', '#ffc107', '#17a2b8']
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            title: {
                                display: true,
                                text: 'Issues Distribution'
                            }
                        }
                    }
                });
            }
            
            // Severity chart
            const severityCtx = document.getElementById('severityChart');
            if (severityCtx) {
                new Chart(severityCtx, {
                    type: 'bar',
                    data: {
                        labels: ['Critical', 'High', 'Medium', 'Low'],
                        datasets: [{
                            label: 'Issue Count',
                            data: [5, 12, 8, 3],
                            backgroundColor: ['#dc3545', '#fd7e14', '#ffc107', '#28a745']
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            title: {
                                display: true,
                                text: 'Issues by Severity'
                            }
                        }
                    }
                });
            }
        }
        
        function exportToJSON() {
            const data = getReportData();
            const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'code_quality_report.json';
            a.click();
            URL.revokeObjectURL(url);
        }
        
        function exportToCSV() {
            const data = getReportData();
            const csv = convertToCSV(data);
            const blob = new Blob([csv], {type: 'text/csv'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'code_quality_report.csv';
            a.click();
            URL.revokeObjectURL(url);
        }
        
        function printReport() {
            window.print();
        }
        
        function getReportData() {
            // This would be populated with actual report data
            return {
                summary: {
                    total_issues: 28,
                    total_files: 15,
                    quality_score: 85
                },
                timestamp: new Date().toISOString()
            };
        }
        
        function convertToCSV(data) {
            // Simple CSV conversion
            const rows = [];
            for (const [key, value] of Object.entries(data)) {
                rows.push(`${key},${value}`);
            }
            return rows.join('\\n');
        }
"""
    
    def save_report(self, html_content: str, output_path: Union[str, Path]) -> None:
        """
        Save HTML report to file.
        
        Args:
            html_content: Generated HTML content
            output_path: Path to save the report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def generate_from_analyzer_results(self, analyzer_results: Dict[str, Any], title: str = "Code Quality Report") -> str:
        """
        Generate HTML report from analyzer results.
        
        Args:
            analyzer_results: Results from various analyzers
            title: Report title
            
        Returns:
            HTML string
        """
        # Transform analyzer results to report format
        report_data = self._transform_analyzer_results(analyzer_results)
        return self.generate_report(report_data, title)
    
    def _transform_analyzer_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Transform analyzer results to report format."""
        transformed = {
            'summary': {},
            'details': {
                'categories': [],
                'files': []
            }
        }
        
        # Extract summary information
        if 'complexity' in results:
            complexity = results['complexity']
            transformed['summary']['total_files'] = complexity.get('total_files', 0)
            transformed['summary']['quality_score'] = complexity.get('average_complexity_score', 0)
        
        if 'dead_code' in results:
            dead_code = results['dead_code']
            transformed['summary']['total_issues'] = dead_code.get('total_issues', 0)
        
        # Extract categories
        if 'complexity' in results:
            complexity_issues = results['complexity'].get('issues', [])
            for issue in complexity_issues:
                transformed['details']['categories'].append({
                    'name': issue.get('type', 'complexity'),
                    'count': 1,
                    'severity': issue.get('severity', 'medium'),
                    'files_affected': 1
                })
        
        return transformed