#!/usr/bin/env python3
"""
Create simple visualizations for dead code analysis results using basic Python libraries.
"""

import json
from collections import Counter
from pathlib import Path

def create_simple_visualizations():
    """Create simple text-based visualizations and reports."""
    
    # Load the report
    with open('/workspace/dead_code_analysis_report.json', 'r') as f:
        report = json.load(f)
    
    print("🔍 DEAD CODE ANALYSIS VISUALIZATION")
    print("=" * 60)
    
    # 1. Overall Statistics
    print("\n📊 OVERALL STATISTICS")
    print("-" * 30)
    print(f"Total Issues Found: {report['total_issues']:,}")
    print(f"Unused Functions: {report['issues_by_type']['unused_function']:,}")
    print(f"Unused Classes: {report['issues_by_type']['unused_class']:,}")
    print(f"Files with Issues: {len(report['issues_by_file']):,}")
    
    # 2. Confidence Distribution (Text Bar Chart)
    print("\n🎯 CONFIDENCE DISTRIBUTION")
    print("-" * 30)
    confidence_data = report['confidence_distribution']
    max_count = max(confidence_data.values())
    
    for level, count in confidence_data.items():
        bar_length = int((count / max_count) * 50)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"{level:6}: {bar} {count:,}")
    
    # 3. Severity Distribution (Text Bar Chart)
    print("\n⚠️ SEVERITY DISTRIBUTION")
    print("-" * 30)
    severity_data = {k: len(v) for k, v in report['issues_by_severity'].items()}
    max_count = max(severity_data.values())
    
    for severity, count in severity_data.items():
        bar_length = int((count / max_count) * 50)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"{severity:6}: {bar} {count:,}")
    
    # 4. Top Files with Most Issues (Text Table)
    print("\n📁 TOP 20 FILES WITH MOST ISSUES")
    print("-" * 60)
    file_issue_counts = [(file_path, len(issues)) for file_path, issues in report['issues_by_file'].items()]
    file_issue_counts.sort(key=lambda x: x[1], reverse=True)
    
    print(f"{'Rank':<4} {'File Name':<40} {'Issues':<8}")
    print("-" * 60)
    for i, (file_path, count) in enumerate(file_issue_counts[:20], 1):
        file_name = Path(file_path).name
        if len(file_name) > 40:
            file_name = file_name[:37] + "..."
        print(f"{i:<4} {file_name:<40} {count:<8}")
    
    # 5. Issues by Directory (Text Table)
    print("\n📂 ISSUES BY DIRECTORY (Top 15)")
    print("-" * 50)
    
    # Group issues by directory
    dir_issues = {}
    for file_path, issues in report['issues_by_file'].items():
        dir_path = str(Path(file_path).parent)
        if dir_path not in dir_issues:
            dir_issues[dir_path] = 0
        dir_issues[dir_path] += len(issues)
    
    top_dirs = sorted(dir_issues.items(), key=lambda x: x[1], reverse=True)[:15]
    
    print(f"{'Directory':<35} {'Issues':<8}")
    print("-" * 50)
    for dir_path, count in top_dirs:
        dir_name = Path(dir_path).name
        if len(dir_name) > 35:
            dir_name = dir_name[:32] + "..."
        print(f"{dir_name:<35} {count:<8}")
    
    # 6. High Confidence Issues Sample
    print("\n🎯 HIGH CONFIDENCE ISSUES (Sample - Top 15)")
    print("-" * 70)
    high_conf_issues = [i for i in report['detailed_issues'] if i['confidence'] >= 80]
    
    print(f"{'File':<25} {'Line':<6} {'Function/Class':<20} {'Confidence':<10}")
    print("-" * 70)
    for issue in high_conf_issues[:15]:
        file_name = Path(issue['file_path']).name
        if len(file_name) > 25:
            file_name = file_name[:22] + "..."
        
        name = issue['name']
        if len(name) > 20:
            name = name[:17] + "..."
        
        print(f"{file_name:<25} {issue['line_number']:<6} {name:<20} {issue['confidence']:<10.1f}%")
    
    # 7. Analysis Summary
    print("\n💡 ANALYSIS SUMMARY")
    print("-" * 30)
    total_functions = report['summary']['total_functions_analyzed']
    total_classes = report['summary']['total_classes_analyzed']
    unused_functions = report['summary']['unused_functions']
    unused_classes = report['summary']['unused_classes']
    
    unused_func_rate = (unused_functions / total_functions * 100) if total_functions > 0 else 0
    unused_class_rate = (unused_classes / total_classes * 100) if total_classes > 0 else 0
    
    print(f"Functions Analyzed: {total_functions:,}")
    print(f"Classes Analyzed: {total_classes:,}")
    print(f"Unused Function Rate: {unused_func_rate:.1f}%")
    print(f"Unused Class Rate: {unused_class_rate:.1f}%")
    
    # 8. Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("-" * 30)
    print("1. Focus on high-confidence issues first (712 issues with 80%+ confidence)")
    print("2. Review files with most issues for potential refactoring")
    print("3. Consider removing unused utility functions and helper classes")
    print("4. Check if unused code might be needed for future features")
    print("5. Use version control to track changes when removing code")
    
    # Create a simple HTML report
    create_simple_html_report(report)

def create_simple_html_report(report):
    """Create a simple HTML report without external dependencies."""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dead Code Analysis Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f7fa; }}
            .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 20px; margin-bottom: 30px; }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
            .stat-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 10px; text-align: center; }}
            .stat-number {{ font-size: 2.5em; font-weight: bold; margin-bottom: 10px; }}
            .stat-label {{ font-size: 1em; opacity: 0.9; }}
            .section {{ margin-bottom: 40px; }}
            .section h2 {{ color: #2c3e50; border-left: 4px solid #3498db; padding-left: 15px; margin-bottom: 20px; }}
            .table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            .table th, .table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            .table th {{ background-color: #f8f9fa; font-weight: bold; color: #495057; }}
            .table tr:nth-child(even) {{ background-color: #f8f9fa; }}
            .table tr:hover {{ background-color: #e9ecef; }}
            .confidence-high {{ color: #dc3545; font-weight: bold; }}
            .confidence-medium {{ color: #fd7e14; font-weight: bold; }}
            .confidence-low {{ color: #28a745; font-weight: bold; }}
            .recommendations {{ background-color: #e8f4fd; padding: 20px; border-radius: 8px; border-left: 4px solid #3498db; }}
            .recommendations ul {{ margin: 0; padding-left: 20px; }}
            .recommendations li {{ margin-bottom: 8px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔍 Dead Code Analysis Report</h1>
                <p>Comprehensive analysis of unused code in the codebase</p>
                <p><strong>Generated:</strong> {Path().cwd()}</p>
            </div>
            
            <div class="stats-grid">
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
                <table class="table">
                    <tr><td><strong>Total Functions Analyzed</strong></td><td>{report['summary']['total_functions_analyzed']:,}</td></tr>
                    <tr><td><strong>Total Classes Analyzed</strong></td><td>{report['summary']['total_classes_analyzed']:,}</td></tr>
                    <tr><td><strong>Unused Function Rate</strong></td><td>{(report['summary']['unused_functions'] / report['summary']['total_functions_analyzed'] * 100):.1f}%</td></tr>
                    <tr><td><strong>Unused Class Rate</strong></td><td>{(report['summary']['unused_classes'] / report['summary']['total_classes_analyzed'] * 100):.1f}%</td></tr>
                    <tr><td><strong>Files with Issues</strong></td><td>{len(report['issues_by_file']):,}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>📁 Files with Most Issues (Top 20)</h2>
                <table class="table">
                    <thead>
                        <tr><th>Rank</th><th>File Name</th><th>Issues</th></tr>
                    </thead>
                    <tbody>
    """
    
    # Add top files
    file_issue_counts = [(file_path, len(issues)) for file_path, issues in report['issues_by_file'].items()]
    file_issue_counts.sort(key=lambda x: x[1], reverse=True)
    
    for i, (file_path, count) in enumerate(file_issue_counts[:20], 1):
        file_name = Path(file_path).name
        html_content += f"""
                        <tr>
                            <td>{i}</td>
                            <td>{file_name}</td>
                            <td>{count:,}</td>
                        </tr>
        """
    
    html_content += """
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2>🎯 High Confidence Issues (Sample)</h2>
                <p>These are the most reliable findings with confidence >= 80%:</p>
                <table class="table">
                    <thead>
                        <tr><th>File</th><th>Line</th><th>Function/Class</th><th>Confidence</th><th>Severity</th></tr>
                    </thead>
                    <tbody>
    """
    
    # Add high confidence issues
    high_conf_issues = [i for i in report['detailed_issues'] if i['confidence'] >= 80]
    for issue in high_conf_issues[:20]:
        confidence_class = 'confidence-high' if issue['confidence'] >= 80 else 'confidence-medium'
        html_content += f"""
                        <tr>
                            <td>{Path(issue['file_path']).name}</td>
                            <td>{issue['line_number']}</td>
                            <td>{issue['name']}</td>
                            <td class="{confidence_class}">{issue['confidence']:.1f}%</td>
                            <td>{issue['severity']}</td>
                        </tr>
        """
    
    html_content += """
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2>💡 Recommendations</h2>
                <div class="recommendations">
                    <ul>
                        <li><strong>Focus on high-confidence issues first:</strong> Start with the 712 issues that have 80%+ confidence</li>
                        <li><strong>Review files with most issues:</strong> Prioritize files with the highest number of unused code</li>
                        <li><strong>Remove unused utility functions:</strong> Many helper functions appear to be unused</li>
                        <li><strong>Check for future needs:</strong> Verify if unused code might be needed for planned features</li>
                        <li><strong>Use version control:</strong> Track changes when removing code to enable rollback if needed</li>
                        <li><strong>Consider refactoring:</strong> Some unused code might indicate opportunities for better code organization</li>
                    </ul>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    with open('/workspace/dead_code_analysis_report.html', 'w') as f:
        f.write(html_content)
    
    print("\n✅ HTML report saved to: /workspace/dead_code_analysis_report.html")

if __name__ == "__main__":
    create_simple_visualizations()