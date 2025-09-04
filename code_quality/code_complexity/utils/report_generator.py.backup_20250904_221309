"""
Report Generator for Code Complexity Analysis
"""

import os
import json
import logging
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate reports for complexity analysis results"""
    
    def __init__(self, config):
        """Initialize report generator"""
        self.config = config
        
    def generate_reports(self, results: Dict[str, Any]):
        """Generate all types of reports"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Generate JSON report
        self._generate_json_report(results, timestamp)
        
        # Generate Markdown report
        self._generate_markdown_report(results, timestamp)
        
        # Generate HTML report
        self._generate_html_report(results, timestamp)
        
        # Generate summary report
        self._generate_summary_report(results, timestamp)
        
    def _generate_json_report(self, results: Dict[str, Any], timestamp: str):
        """Generate JSON report"""
        output_file = os.path.join(self.config.reports_dir, f'complexity_report_{timestamp}.json')
        
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"JSON report generated: {output_file}")
        except Exception as e:
            logger.error(f"Error generating JSON report: {e}")
            
    def _generate_markdown_report(self, results: Dict[str, Any], timestamp: str):
        """Generate Markdown report"""
        output_file = os.path.join(self.config.reports_dir, f'complexity_report_{timestamp}.md')
        
        try:
            with open(output_file, 'w') as f:
                f.write(self._create_markdown_content(results))
            logger.info(f"Markdown report generated: {output_file}")
        except Exception as e:
            logger.error(f"Error generating Markdown report: {e}")
            
    def _generate_html_report(self, results: Dict[str, Any], timestamp: str):
        """Generate HTML report"""
        output_file = os.path.join(self.config.reports_dir, f'complexity_report_{timestamp}.html')
        
        try:
            with open(output_file, 'w') as f:
                f.write(self._create_html_content(results))
            logger.info(f"HTML report generated: {output_file}")
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            
    def _generate_summary_report(self, results: Dict[str, Any], timestamp: str):
        """Generate summary report"""
        output_file = os.path.join(self.config.reports_dir, f'complexity_summary_{timestamp}.md')
        
        try:
            with open(output_file, 'w') as f:
                f.write(self._create_summary_content(results))
            logger.info(f"Summary report generated: {output_file}")
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            
    def _create_markdown_content(self, results: Dict[str, Any]) -> str:
        """Create Markdown report content"""
        content = []
        
        # Header
        content.append("# Code Complexity Analysis Report")
        content.append(f"**Generated:** {results.get('analysis_timestamp', 'Unknown')}")
        content.append(f"**Target:** {results.get('target_path', 'Unknown')}")
        content.append("")
        
        # File Analysis Section
        if 'file_analysis' in results and results['file_analysis']:
            content.append("## File-Level Analysis")
            content.append("")
            
            # Sort files by combined score (descending)
            files = results['file_analysis']
            sorted_files = sorted(files.items(), 
                                key=lambda x: x[1].get('combined_score', 0), 
                                reverse=True)
            
            content.append("| File | Combined Score | PyExamine | Radon CC | Radon MI | Xenon |")
            content.append("|------|----------------|-----------|----------|----------|-------|")
            
            for file_path, metrics in sorted_files:
                content.append(f"| {file_path} | "
                             f"{metrics.get('combined_score', 'N/A'):.3f} | "
                             f"{metrics.get('pyexamine_score', 'N/A')} | "
                             f"{metrics.get('radon_cc', 'N/A')} | "
                             f"{metrics.get('radon_mi', 'N/A')} | "
                             f"{metrics.get('xenon_score', 'N/A')} |")
            content.append("")
            
        # Directory Analysis Section
        if 'directory_analysis' in results and results['directory_analysis']:
            content.append("## Directory-Level Analysis")
            content.append("")
            
            for dir_path, metrics in results['directory_analysis'].items():
                content.append(f"### {dir_path}")
                content.append("")
                content.append(f"- **Files Analyzed:** {metrics.get('total_files_analyzed', 0)}/{metrics.get('file_count', 0)}")
                content.append(f"- **Average Complexity:** {metrics.get('average_complexity', 0):.3f}")
                content.append(f"- **Max Complexity:** {metrics.get('max_complexity', 0):.3f}")
                content.append(f"- **Min Complexity:** {metrics.get('min_complexity', 0):.3f}")
                content.append("")
                
                # Complexity distribution
                distribution = metrics.get('complexity_distribution', {})
                content.append("**Complexity Distribution:**")
                content.append(f"- Low (≥0.7): {distribution.get('low', 0)} files")
                content.append(f"- Medium (0.4-0.7): {distribution.get('medium', 0)} files")
                content.append(f"- High (<0.4): {distribution.get('high', 0)} files")
                content.append("")
                
        return "\n".join(content)
        
    def _create_html_content(self, results: Dict[str, Any]) -> str:
        """Create HTML report content"""
        timestamp = results.get('analysis_timestamp', 'Unknown')
        target = results.get('target_path', 'Unknown')
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Code Complexity Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .high-complexity {{ background-color: #ffebee; }}
        .medium-complexity {{ background-color: #fff3e0; }}
        .low-complexity {{ background-color: #e8f5e8; }}
        .summary {{ background-color: #f5f5f5; padding: 20px; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>Code Complexity Analysis Report</h1>
    <div class="summary">
        <p><strong>Generated:</strong> {timestamp}</p>
        <p><strong>Target:</strong> {target}</p>
    </div>
"""
        
        # File Analysis Table
        if 'file_analysis' in results and results['file_analysis']:
            html += "<h2>File-Level Analysis</h2>"
            html += "<table>"
            html += "<tr><th>File</th><th>Combined Score</th><th>PyExamine</th><th>Radon CC</th><th>Radon MI</th><th>Xenon</th></tr>"
            
            files = results['file_analysis']
            sorted_files = sorted(files.items(), 
                                key=lambda x: x[1].get('combined_score', 0), 
                                reverse=True)
            
            for file_path, metrics in sorted_files:
                score = metrics.get('combined_score', 0)
                css_class = self._get_complexity_class(score)
                
                html += f"<tr class='{css_class}'>"
                html += f"<td>{file_path}</td>"
                html += f"<td>{score:.3f if score else 'N/A'}</td>"
                html += f"<td>{metrics.get('pyexamine_score', 'N/A')}</td>"
                html += f"<td>{metrics.get('radon_cc', 'N/A')}</td>"
                html += f"<td>{metrics.get('radon_mi', 'N/A')}</td>"
                html += f"<td>{metrics.get('xenon_score', 'N/A')}</td>"
                html += "</tr>"
                
            html += "</table>"
            
        # Directory Analysis
        if 'directory_analysis' in results and results['directory_analysis']:
            html += "<h2>Directory-Level Analysis</h2>"
            
            for dir_path, metrics in results['directory_analysis'].items():
                html += f"<h3>{dir_path}</h3>"
                html += "<div class='summary'>"
                html += f"<p><strong>Files Analyzed:</strong> {metrics.get('total_files_analyzed', 0)}/{metrics.get('file_count', 0)}</p>"
                html += f"<p><strong>Average Complexity:</strong> {metrics.get('average_complexity', 0):.3f}</p>"
                html += f"<p><strong>Max Complexity:</strong> {metrics.get('max_complexity', 0):.3f}</p>"
                html += f"<p><strong>Min Complexity:</strong> {metrics.get('min_complexity', 0):.3f}</p>"
                
                distribution = metrics.get('complexity_distribution', {})
                html += "<p><strong>Complexity Distribution:</strong></p>"
                html += "<ul>"
                html += f"<li>Low (≥0.7): {distribution.get('low', 0)} files</li>"
                html += f"<li>Medium (0.4-0.7): {distribution.get('medium', 0)} files</li>"
                html += f"<li>High (<0.4): {distribution.get('high', 0)} files</li>"
                html += "</ul>"
                html += "</div>"
                
        html += "</body></html>"
        return html
        
    def _create_summary_content(self, results: Dict[str, Any]) -> str:
        """Create summary report content"""
        content = []
        
        content.append("# Code Complexity Analysis Summary")
        content.append(f"**Generated:** {results.get('analysis_timestamp', 'Unknown')}")
        content.append(f"**Target:** {results.get('target_path', 'Unknown')}")
        content.append("")
        
        # Overall Statistics
        file_analysis = results.get('file_analysis', {})
        directory_analysis = results.get('directory_analysis', {})
        
        if file_analysis:
            scores = [m.get('combined_score', 0) for m in file_analysis.values() if m.get('combined_score') is not None]
            
            if scores:
                content.append("## Overall Statistics")
                content.append(f"- **Total Files Analyzed:** {len(scores)}")
                content.append(f"- **Average Complexity Score:** {sum(scores)/len(scores):.3f}")
                content.append(f"- **Highest Complexity:** {max(scores):.3f}")
                content.append(f"- **Lowest Complexity:** {min(scores):.3f}")
                content.append("")
                
                # Complexity distribution
                low_count = len([s for s in scores if s >= 0.7])
                medium_count = len([s for s in scores if 0.4 <= s < 0.7])
                high_count = len([s for s in scores if s < 0.4])
                
                content.append("## Complexity Distribution")
                content.append(f"- **Low Complexity (≥0.7):** {low_count} files ({low_count/len(scores)*100:.1f}%)")
                content.append(f"- **Medium Complexity (0.4-0.7):** {medium_count} files ({medium_count/len(scores)*100:.1f}%)")
                content.append(f"- **High Complexity (<0.4):** {high_count} files ({high_count/len(scores)*100:.1f}%)")
                content.append("")
                
        # Top 10 most complex files
        if file_analysis:
            sorted_files = sorted(file_analysis.items(), 
                                key=lambda x: x[1].get('combined_score', 0), 
                                reverse=True)
            
            content.append("## Top 10 Most Complex Files")
            content.append("")
            
            for i, (file_path, metrics) in enumerate(sorted_files[:10], 1):
                score = metrics.get('combined_score', 0)
                content.append(f"{i}. **{file_path}** - Score: {score:.3f}")
                
        return "\n".join(content)
        
    def _get_complexity_class(self, score: float) -> str:
        """Get CSS class based on complexity score"""
        if score >= 0.7:
            return "low-complexity"
        elif score >= 0.4:
            return "medium-complexity"
        else:
            return "high-complexity"