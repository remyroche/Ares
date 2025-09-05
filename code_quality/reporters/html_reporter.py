#!/usr/bin/env python3
"""HTML report generator for code analysis results."""

from datetime import datetime
from typing import Dict, Any
import time


class HTMLReporter:
    """Generates HTML reports from analysis results."""
    
    def __init__(self):
        """Initialize the HTML reporter."""
        self.template = self._get_base_template()
    
    def generate_from_analyzer_results(self, results: Dict[str, Any], title: str = "Code Analysis Report") -> str:
        """Generate HTML report from analyzer results."""
        html = self.template.format(
            title=title,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            content=self._generate_content(results)
        )
        return html

    def _generate_header(self, title: str) -> str:
        """Generate HTML header with CSS and JavaScript."""

        return f"""

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; padding-bottom: 20px; border-bottom: 2px solid #007acc; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f0f8ff; border-radius: 5px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #007acc; }}
        .metric-label {{ font-size: 14px; color: #666; }}
        .timestamp {{ text-align: center; color: #666; font-size: 0.9em; margin-top: 30px; }}
    </style>
</head>
<body>
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

    def _generate_summary_section(self, data: dict[str, Any]) -> str:
        """Generate summary section with key metrics."""
        summary = data.get("summary", {})

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
from .exceptions import (
<div class="metric-value critical">{summary.get('critical_errors', 0)}</div>
)
                </div>
                <div class="metric-card">
                    <h3>Quality Score</h3>
                    <div class="metric-value">{summary.get('quality_score', 'N/A')}</div>
                </div>
            </div>
        </section>
"""

    def _generate_details_section(self, data: dict[str, Any]) -> str:
        """Generate detailed analysis section."""
        details = data.get("details", {})

        html = """
        <section id="details" class="report-section">
            <h2>Detailed Analysis</h2>
        """

        # Issues by category
        if "categories" in details:
            html += self._generate_categories_table(details["categories"])

        # Issues by file
        if "files" in details:
            html += self._generate_files_table(details["files"])

        html += """
        </section>
"""
        return html

    def _generate_categories_table(self, categories: list[dict]) -> str:
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

    def _generate_files_table(self, files: list[dict]) -> str:
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
)
                            <td>{file_info.get('score', 'N/A')}</td>
                        </tr>
"""

        html += """
                    </tbody>
                </table>
            </div>
"""
        return html

    def _generate_charts_section(self, data: dict[str, Any]) -> str:
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
)
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
)
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
        <div class="section">
            <h2>�� Analysis Summary</h2>
            <div class="metric">
                <div class="metric-value">0</div>
                <div class="metric-label">Total Issues</div>
            </div>
            <div class="metric">
                <div class="metric-value">0</div>
                <div class="metric-label">Files Analyzed</div>
            </div>
        </div>
        """
    
    def _generate_dead_code_section(self, dead_code: Dict[str, Any]) -> str:
        """Generate dead code analysis section."""
        return """
        <div class="section">
            <h2>💀 Dead Code Analysis</h2>
            <p>Dead code analysis results will be displayed here.</p>
        </div>
        """
