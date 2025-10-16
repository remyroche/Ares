#!/usr/bin/env python3
"""
Create visualizations for dead code analysis results.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
from pathlib import Path

def create_visualizations():
    """Create various visualizations of the dead code analysis results."""
    
    # Load the report
    with open('/workspace/dead_code_analysis_report.json', 'r') as f:
        report = json.load(f)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Issues by Type (Pie Chart)
    ax1 = plt.subplot(2, 3, 1)
    types = list(report['issues_by_type'].keys())
    counts = list(report['issues_by_type'].values())
    colors = ['#ff9999', '#66b3ff']
    wedges, texts, autotexts = ax1.pie(counts, labels=types, autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Dead Code Issues by Type', fontsize=14, fontweight='bold')
    
    # 2. Confidence Distribution (Bar Chart)
    ax2 = plt.subplot(2, 3, 2)
    confidence_levels = list(report['confidence_distribution'].keys())
    confidence_counts = list(report['confidence_distribution'].values())
    bars = ax2.bar(confidence_levels, confidence_counts, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
    ax2.set_title('Issues by Confidence Level', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Issues')
    ax2.set_xlabel('Confidence Level')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{int(height):,}', ha='center', va='bottom')
    
    # 3. Severity Distribution (Bar Chart)
    ax3 = plt.subplot(2, 3, 3)
    severity_levels = list(report['issues_by_severity'].keys())
    severity_counts = [len(issues) for issues in report['issues_by_severity'].values()]
    colors_severity = ['#ff6b6b', '#ffa726', '#66bb6a']
    bars = ax3.bar(severity_levels, severity_counts, color=colors_severity)
    ax3.set_title('Issues by Severity Level', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Number of Issues')
    ax3.set_xlabel('Severity Level')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{int(height):,}', ha='center', va='bottom')
    
    # 4. Top 15 Files with Most Issues (Horizontal Bar Chart)
    ax4 = plt.subplot(2, 3, 4)
    file_issue_counts = [(file_path, len(issues)) for file_path, issues in report['issues_by_file'].items()]
    file_issue_counts.sort(key=lambda x: x[1], reverse=True)
    top_files = file_issue_counts[:15]
    
    file_names = [Path(f[0]).name for f in top_files]
    file_counts = [f[1] for f in top_files]
    
    y_pos = np.arange(len(file_names))
    bars = ax4.barh(y_pos, file_counts, color='#ff9f43')
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(file_names, fontsize=8)
    ax4.set_xlabel('Number of Issues')
    ax4.set_title('Top 15 Files with Most Issues', fontsize=14, fontweight='bold')
    ax4.invert_yaxis()
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax4.text(width + 1, bar.get_y() + bar.get_height()/2.,
                f'{int(width)}', ha='left', va='center', fontsize=8)
    
    # 5. Issues Distribution by Directory (Treemap-like visualization)
    ax5 = plt.subplot(2, 3, 5)
    
    # Group issues by directory
    dir_issues = {}
    for file_path, issues in report['issues_by_file'].items():
        dir_path = str(Path(file_path).parent)
        if dir_path not in dir_issues:
            dir_issues[dir_path] = 0
        dir_issues[dir_path] += len(issues)
    
    # Get top 10 directories
    top_dirs = sorted(dir_issues.items(), key=lambda x: x[1], reverse=True)[:10]
    dir_names = [Path(d[0]).name for d in top_dirs]
    dir_counts = [d[1] for d in top_dirs]
    
    bars = ax5.bar(range(len(dir_names)), dir_counts, color='#a55eea')
    ax5.set_xticks(range(len(dir_names)))
    ax5.set_xticklabels(dir_names, rotation=45, ha='right', fontsize=8)
    ax5.set_ylabel('Number of Issues')
    ax5.set_title('Issues by Directory (Top 10)', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    # 6. Summary Statistics (Text Box)
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate additional statistics
    total_issues = report['total_issues']
    total_functions = report['summary']['total_functions_analyzed']
    total_classes = report['summary']['total_classes_analyzed']
    unused_functions = report['summary']['unused_functions']
    unused_classes = report['summary']['unused_classes']
    
    unused_func_rate = (unused_functions / total_functions * 100) if total_functions > 0 else 0
    unused_class_rate = (unused_classes / total_classes * 100) if total_classes > 0 else 0
    
    high_conf_issues = report['confidence_distribution']['high']
    medium_conf_issues = report['confidence_distribution']['medium']
    
    summary_text = f"""
    DEAD CODE ANALYSIS SUMMARY
    
    📊 OVERALL STATISTICS
    • Total Issues Found: {total_issues:,}
    • Unused Functions: {unused_functions:,} ({unused_func_rate:.1f}%)
    • Unused Classes: {unused_classes:,} ({unused_class_rate:.1f}%)
    
    📈 CONFIDENCE BREAKDOWN
    • High Confidence: {high_conf_issues:,}
    • Medium Confidence: {medium_conf_issues:,}
    
    🎯 ANALYSIS SCOPE
    • Functions Analyzed: {total_functions:,}
    • Classes Analyzed: {total_classes:,}
    • Files with Issues: {len(report['issues_by_file']):,}
    
    ⚠️ SEVERITY DISTRIBUTION
    • High Severity: {len(report['issues_by_severity']['high']):,}
    • Medium Severity: {len(report['issues_by_severity']['medium']):,}
    • Low Severity: {len(report['issues_by_severity']['low']):,}
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_path = '/workspace/dead_code_analysis_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # Create a detailed HTML report
    create_html_report(report)
    
    plt.show()

def create_html_report(report):
    """Create a detailed HTML report."""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dead Code Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; color: #333; border-bottom: 3px solid #007acc; padding-bottom: 20px; margin-bottom: 30px; }}
            .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
            .stat-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
            .stat-number {{ font-size: 2em; font-weight: bold; margin-bottom: 5px; }}
            .stat-label {{ font-size: 0.9em; opacity: 0.9; }}
            .section {{ margin-bottom: 30px; }}
            .section h2 {{ color: #333; border-left: 4px solid #007acc; padding-left: 15px; }}
            .file-list {{ max-height: 400px; overflow-y: auto; border: 1px solid #ddd; border-radius: 5px; }}
            .file-item {{ padding: 10px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; }}
            .file-item:nth-child(even) {{ background-color: #f9f9f9; }}
            .file-name {{ font-weight: bold; color: #007acc; }}
            .issue-count {{ background-color: #ff6b6b; color: white; padding: 2px 8px; border-radius: 15px; font-size: 0.8em; }}
            .high-confidence {{ background-color: #ffebee; border-left: 4px solid #f44336; padding: 10px; margin: 5px 0; border-radius: 4px; }}
            .issue-item {{ margin: 5px 0; padding: 8px; background-color: #f8f9fa; border-radius: 4px; border-left: 3px solid #007acc; }}
            .confidence-high {{ border-left-color: #f44336; }}
            .confidence-medium {{ border-left-color: #ff9800; }}
            .confidence-low {{ border-left-color: #4caf50; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔍 Dead Code Analysis Report</h1>
                <p>Comprehensive analysis of unused code in the codebase</p>
            </div>
            
            <div class="summary">
                <div class="stat-card">
                    <div class="stat-number">{report['total_issues']:,}</div>
                    <div class="stat-label">Total Issues</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{report['issues_by_type']['unused_function']:,}</div>
                    <div class="stat-label">Unused Functions</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{report['issues_by_type']['unused_class']:,}</div>
                    <div class="stat-label">Unused Classes</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{report['confidence_distribution']['high']:,}</div>
                    <div class="stat-label">High Confidence</div>
                </div>
            </div>
            
            <div class="section">
                <h2>📊 Analysis Summary</h2>
                <p><strong>Total Functions Analyzed:</strong> {report['summary']['total_functions_analyzed']:,}</p>
                <p><strong>Total Classes Analyzed:</strong> {report['summary']['total_classes_analyzed']:,}</p>
                <p><strong>Unused Function Rate:</strong> {(report['summary']['unused_functions'] / report['summary']['total_functions_analyzed'] * 100):.1f}%</p>
                <p><strong>Unused Class Rate:</strong> {(report['summary']['unused_classes'] / report['summary']['total_classes_analyzed'] * 100):.1f}%</p>
            </div>
            
            <div class="section">
                <h2>📁 Files with Most Issues (Top 20)</h2>
                <div class="file-list">
    """
    
    # Add top files
    file_issue_counts = [(file_path, len(issues)) for file_path, issues in report['issues_by_file'].items()]
    file_issue_counts.sort(key=lambda x: x[1], reverse=True)
    
    for file_path, count in file_issue_counts[:20]:
        file_name = Path(file_path).name
        html_content += f"""
                    <div class="file-item">
                        <span class="file-name">{file_name}</span>
                        <span class="issue-count">{count}</span>
                    </div>
        """
    
    html_content += """
                </div>
            </div>
            
            <div class="section">
                <h2>🎯 High Confidence Issues (Sample)</h2>
                <p>These are the most reliable findings with confidence >= 80%:</p>
    """
    
    # Add high confidence issues
    high_conf_issues = [i for i in report['detailed_issues'] if i['confidence'] >= 80]
    for issue in high_conf_issues[:20]:
        confidence_class = 'confidence-high' if issue['confidence'] >= 80 else 'confidence-medium'
        html_content += f"""
                <div class="issue-item {confidence_class}">
                    <strong>{Path(issue['file_path']).name}:{issue['line_number']}</strong> - {issue['description']}<br>
                    <small>Confidence: {issue['confidence']:.1f}% | Severity: {issue['severity']}</small>
                </div>
        """
    
    html_content += """
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    with open('/workspace/dead_code_analysis_report.html', 'w') as f:
        f.write(html_content)
    
    print("HTML report saved to: /workspace/dead_code_analysis_report.html")

if __name__ == "__main__":
    try:
        create_visualizations()
        print("✅ Visualizations created successfully!")
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()