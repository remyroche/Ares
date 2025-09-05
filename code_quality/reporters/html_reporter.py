#!/usr/bin/env python3
"""HTML report generator for code analysis results."""

from datetime import datetime
from typing import Dict, Any


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
    
    def _get_base_template(self) -> str:
        """Get the base HTML template."""
        return """
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
        <div class="header">
            <h1>🔍 {title}</h1>
            <p>Generated on {timestamp}</p>
        </div>
        {content}
    </div>
</body>
</html>
"""
    
    def _generate_content(self, results: Dict[str, Any]) -> str:
        """Generate the main content of the report."""
        content = ""
        
        # Summary section
        content += self._generate_summary_section(results)
        
        # Dead code section
        if "dead_code" in results:
            content += self._generate_dead_code_section(results["dead_code"])
        
        return content
    
    def _generate_summary_section(self, results: Dict[str, Any]) -> str:
        """Generate summary section."""
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
