#!/usr/bin/env python3
"""
Comprehensive Placeholder Analysis Report Generator
Analyzes placeholder detection results and provides detailed insights
"""

import json
import os
from collections import defaultdict, Counter
from pathlib import Path
import re
from datetime import datetime

def load_placeholder_data(json_file):
    """Load placeholder detection results from JSON file."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None

def analyze_placeholder_patterns(placeholders):
    """Analyze patterns in placeholder usage."""
    patterns = {
        'comment_patterns': defaultdict(int),
        'value_patterns': defaultdict(int),
        'function_patterns': defaultdict(int),
        'file_patterns': defaultdict(int),
        'severity_distribution': defaultdict(int),
        'line_number_ranges': defaultdict(int)
    }
    
    for placeholder in placeholders:
        # Comment patterns
        if placeholder['placeholder_type'].startswith('comment_'):
            pattern_type = placeholder['placeholder_type'].replace('comment_', '')
            patterns['comment_patterns'][pattern_type] += 1
        
        # Value patterns
        elif placeholder['placeholder_type'] == 'value_placeholder':
            content = placeholder['content']
            if 'placeholder' in content:
                patterns['value_patterns'][content] += 1
        
        # Function patterns
        elif placeholder['placeholder_type'] in ['stub_function', 'empty_function', 'placeholder_return']:
            patterns['function_patterns'][placeholder['placeholder_type']] += 1
        
        # File patterns
        file_path = placeholder['file_path']
        file_dir = os.path.dirname(file_path)
        patterns['file_patterns'][file_dir] += 1
        
        # Severity distribution
        patterns['severity_distribution'][placeholder['severity']] += 1
        
        # Line number ranges
        line_num = placeholder['line']
        if line_num <= 10:
            patterns['line_number_ranges']['1-10'] += 1
        elif line_num <= 50:
            patterns['line_number_ranges']['11-50'] += 1
        elif line_num <= 100:
            patterns['line_number_ranges']['51-100'] += 1
        elif line_num <= 500:
            patterns['line_number_ranges']['101-500'] += 1
        else:
            patterns['line_number_ranges']['500+'] += 1
    
    return patterns

def analyze_file_categories(placeholders):
    """Categorize files by type and placeholder density."""
    file_analysis = defaultdict(lambda: {
        'total_placeholders': 0,
        'by_type': defaultdict(int),
        'by_severity': defaultdict(int),
        'placeholder_lines': set()
    })
    
    for placeholder in placeholders:
        file_path = placeholder['file_path']
        file_analysis[file_path]['total_placeholders'] += 1
        file_analysis[file_path]['by_type'][placeholder['placeholder_type']] += 1
        file_analysis[file_path]['by_severity'][placeholder['severity']] += 1
        file_analysis[file_path]['placeholder_lines'].add(placeholder['line'])
    
    # Convert sets to counts for JSON serialization
    for file_info in file_analysis.values():
        file_info['placeholder_lines'] = len(file_info['placeholder_lines'])
    
    return file_analysis

def identify_critical_files(file_analysis, threshold=10):
    """Identify files with high placeholder density."""
    critical_files = []
    
    for file_path, info in file_analysis.items():
        if info['total_placeholders'] >= threshold:
            critical_files.append({
                'file_path': file_path,
                'total_placeholders': info['total_placeholders'],
                'high_severity': info['by_severity'].get('high', 0),
                'medium_severity': info['by_severity'].get('medium', 0),
                'low_severity': info['by_severity'].get('low', 0)
            })
    
    # Sort by total placeholders (descending)
    critical_files.sort(key=lambda x: x['total_placeholders'], reverse=True)
    return critical_files

def analyze_placeholder_contexts(placeholders):
    """Analyze the context and content of placeholders."""
    contexts = {
        'common_phrases': Counter(),
        'numeric_values': Counter(),
        'boolean_values': Counter(),
        'string_values': Counter(),
        'function_names': Counter(),
        'class_names': Counter()
    }
    
    for placeholder in placeholders:
        content = placeholder['content']
        context = placeholder['context']
        
        # Extract common phrases
        if 'placeholder' in content.lower():
            contexts['common_phrases']['placeholder'] += 1
        if 'stub' in content.lower():
            contexts['common_phrases']['stub'] += 1
        if 'implement' in content.lower():
            contexts['common_phrases']['implement'] += 1
        if 'future' in content.lower():
            contexts['common_phrases']['future'] += 1
        
        # Extract numeric values
        numeric_matches = re.findall(r'\b\d+(?:\.\d+)?\b', context)
        for match in numeric_matches:
            contexts['numeric_values'][match] += 1
        
        # Extract boolean values
        if 'True' in context:
            contexts['boolean_values']['True'] += 1
        if 'False' in context:
            contexts['boolean_values']['False'] += 1
        
        # Extract string values
        string_matches = re.findall(r'["\']([^"\']*)["\']', context)
        for match in string_matches:
            if match:  # Skip empty strings
                contexts['string_values'][match] += 1
        
        # Extract function names from context
        func_matches = re.findall(r'def\s+(\w+)', context)
        for match in func_matches:
            contexts['function_names'][match] += 1
        
        # Extract class names from context
        class_matches = re.findall(r'class\s+(\w+)', context)
        for match in class_matches:
            contexts['class_names'][match] += 1
    
    return contexts

def generate_recommendations(analysis_data):
    """Generate actionable recommendations based on analysis."""
    recommendations = {
        'immediate_actions': [],
        'short_term_goals': [],
        'long_term_strategies': [],
        'code_quality_improvements': [],
        'testing_improvements': [],
        'documentation_needs': []
    }
    
    # Immediate actions based on high severity items
    high_severity_count = analysis_data['patterns']['severity_distribution'].get('high', 0)
    if high_severity_count > 0:
        recommendations['immediate_actions'].append(
            f"Address {high_severity_count} high-severity placeholders (FIXME, BUG, HACK, XXX)"
        )
    
    # Short term goals
    stub_count = analysis_data['patterns']['function_patterns'].get('stub_function', 0)
    if stub_count > 0:
        recommendations['short_term_goals'].append(
            f"Implement {stub_count} stub functions that currently return placeholder values"
        )
    
    empty_func_count = analysis_data['patterns']['function_patterns'].get('empty_function', 0)
    if empty_func_count > 0:
        recommendations['short_term_goals'].append(
            f"Complete {empty_func_count} empty function implementations"
        )
    
    # Long term strategies
    total_placeholders = analysis_data['summary_stats']['total_placeholders']
    if total_placeholders > 100:
        recommendations['long_term_strategies'].append(
            "Establish coding standards to prevent placeholder accumulation"
        )
        recommendations['long_term_strategies'].append(
            "Implement automated placeholder detection in CI/CD pipeline"
        )
    
    # Code quality improvements
    if analysis_data['patterns']['value_patterns']:
        recommendations['code_quality_improvements'].append(
            "Replace hardcoded placeholder values with configuration parameters"
        )
    
    # Testing improvements
    if analysis_data['patterns']['function_patterns'].get('stub_function', 0) > 0:
        recommendations['testing_improvements'].append(
            "Add comprehensive tests for functions currently returning placeholder values"
        )
    
    # Documentation needs
    if analysis_data['patterns']['comment_patterns'].get('implement', 0) > 0:
        recommendations['documentation_needs'].append(
            "Document implementation requirements for marked functions"
        )
    
    return recommendations

def generate_html_report(analysis_data, output_file):
    """Generate a comprehensive HTML report."""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Comprehensive Placeholder Analysis Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                border-bottom: 3px solid #007acc;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }}
            .header h1 {{
                color: #007acc;
                margin: 0;
                font-size: 2.5em;
            }}
            .header p {{
                color: #666;
                font-size: 1.2em;
                margin: 10px 0;
            }}
            .summary-stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .stat-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }}
            .stat-card h3 {{
                margin: 0 0 10px 0;
                font-size: 1.5em;
            }}
            .stat-card .number {{
                font-size: 2.5em;
                font-weight: bold;
                margin: 10px 0;
            }}
            .section {{
                margin: 30px 0;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 8px;
                border-left: 4px solid #007acc;
            }}
            .section h2 {{
                color: #007acc;
                margin-top: 0;
                border-bottom: 2px solid #e9ecef;
                padding-bottom: 10px;
            }}
            .chart-container {{
                margin: 20px 0;
                padding: 20px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            .severity-indicator {{
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 8px;
            }}
            .severity-high {{ background-color: #dc3545; }}
            .severity-medium {{ background-color: #ffc107; }}
            .severity-low {{ background-color: #28a745; }}
            .critical-files {{
                background: #fff3cd;
                border: 1px solid #ffeaa7;
                border-radius: 8px;
                padding: 15px;
                margin: 15px 0;
            }}
            .recommendations {{
                background: #d1ecf1;
                border: 1px solid #bee5eb;
                border-radius: 8px;
                padding: 15px;
                margin: 15px 0;
            }}
            .recommendations h4 {{
                color: #0c5460;
                margin-top: 0;
            }}
            .recommendations ul {{
                margin: 10px 0;
                padding-left: 20px;
            }}
            .recommendations li {{
                margin: 8px 0;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
                background: white;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #007acc;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .progress-bar {{
                width: 100%;
                background-color: #e9ecef;
                border-radius: 10px;
                overflow: hidden;
                height: 20px;
            }}
            .progress-fill {{
                height: 100%;
                background: linear-gradient(90deg, #28a745, #ffc107, #dc3545);
                transition: width 0.3s ease;
            }}
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔍 Comprehensive Placeholder Analysis Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Analysis of {analysis_data['summary_stats']['total_files_analyzed']} Python files</p>
            </div>
            
            <div class="summary-stats">
                <div class="stat-card">
                    <h3>📁 Files Analyzed</h3>
                    <div class="number">{analysis_data['summary_stats']['total_files_analyzed']}</div>
                </div>
                <div class="stat-card">
                    <h3>⚠️ Files with Placeholders</h3>
                    <div class="number">{analysis_data['summary_stats']['files_with_placeholders']}</div>
                </div>
                <div class="stat-card">
                    <h3>🔍 Total Placeholders</h3>
                    <div class="number">{analysis_data['summary_stats']['total_placeholders']}</div>
                </div>
                <div class="stat-card">
                    <h3>📊 Avg per File</h3>
                    <div class="number">{analysis_data['summary_stats']['average_placeholders_per_file']:.2f}</div>
                </div>
            </div>
            
            <div class="section">
                <h2>📈 Severity Distribution</h2>
                <div class="chart-container">
                    <canvas id="severityChart" width="400" height="200"></canvas>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {analysis_data['summary_stats']['files_with_placeholders_percentage']}%"></div>
                </div>
                <p><strong>Files with placeholders:</strong> {analysis_data['summary_stats']['files_with_placeholders_percentage']:.1f}% of analyzed files</p>
            </div>
            
            <div class="section">
                <h2>🚨 Critical Files (10+ Placeholders)</h2>
                <div class="critical-files">
                    <p><strong>Found {len(analysis_data['critical_files'])} files with high placeholder density:</strong></p>
                    <table>
                        <thead>
                            <tr>
                                <th>File Path</th>
                                <th>Total Placeholders</th>
                                <th>High Severity</th>
                                <th>Medium Severity</th>
                                <th>Low Severity</th>
                            </tr>
                        </thead>
                        <tbody>
    """
    
    for file_info in analysis_data['critical_files'][:20]:  # Show top 20
        html_content += f"""
                            <tr>
                                <td>{file_info['file_path']}</td>
                                <td>{file_info['total_placeholders']}</td>
                                <td><span class="severity-indicator severity-high"></span>{file_info['high_severity']}</td>
                                <td><span class="severity-indicator severity-medium"></span>{file_info['medium_severity']}</td>
                                <td><span class="severity-indicator severity-low"></span>{file_info['low_severity']}</td>
                            </tr>
        """
    
    html_content += """
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div class="section">
                <h2>📊 Placeholder Type Analysis</h2>
                <div class="chart-container">
                    <canvas id="typeChart" width="400" height="200"></canvas>
                </div>
            </div>
            
            <div class="section">
                <h2>🎯 Top Recommendations</h2>
                <div class="recommendations">
    """
    
    for category, items in analysis_data['recommendations'].items():
        if items:
            html_content += f"""
                    <h4>{category.replace('_', ' ').title()}</h4>
                    <ul>
            """
            for item in items:
                html_content += f"<li>{item}</li>"
            html_content += "</ul>"
    
    html_content += """
                </div>
            </div>
            
            <div class="section">
                <h2>📋 Detailed Statistics</h2>
                <div class="chart-container">
                    <h3>Comment Pattern Distribution</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Pattern Type</th>
                                <th>Count</th>
                                <th>Percentage</th>
                            </tr>
                        </thead>
                        <tbody>
    """
    
    total_comment_placeholders = sum(analysis_data['patterns']['comment_patterns'].values())
    for pattern, count in sorted(analysis_data['patterns']['comment_patterns'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_comment_placeholders * 100) if total_comment_placeholders > 0 else 0
        html_content += f"""
                            <tr>
                                <td>{pattern.upper()}</td>
                                <td>{count}</td>
                                <td>{percentage:.1f}%</td>
                            </tr>
        """
    
    html_content += """
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <script>
            // Severity Chart
            const severityCtx = document.getElementById('severityChart').getContext('2d');
            new Chart(severityCtx, {{
                type: 'doughnut',
                data: {{
                    labels: ['High', 'Medium', 'Low'],
                    datasets: [{{
                        data: [
                            """ + str(analysis_data['patterns']['severity_distribution'].get('high', 0)) + """,
                            """ + str(analysis_data['patterns']['severity_distribution'].get('medium', 0)) + """,
                            """ + str(analysis_data['patterns']['severity_distribution'].get('low', 0)) + """
                        ],
                        backgroundColor: ['#dc3545', '#ffc107', '#28a745'],
                        borderWidth: 2,
                        borderColor: '#fff'
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{
                            position: 'bottom'
                        }},
                        title: {{
                            display: true,
                            text: 'Placeholder Severity Distribution'
                        }}
                    }}
                }}
            }});
            
            // Type Chart
            const typeCtx = document.getElementById('typeChart').getContext('2d');
            new Chart(typeCtx, {{
                type: 'bar',
                data: {{
                    labels: ['Comment', 'Value', 'Function', 'Assignment', 'Return'],
                    datasets: [{{
                        label: 'Count',
                        data: [
                            """ + str(sum(analysis_data['patterns']['comment_patterns'].values())) + """,
                            """ + str(analysis_data['patterns']['value_patterns'].get('value_placeholder', 0)) + """,
                            """ + str(sum(analysis_data['patterns']['function_patterns'].values())) + """,
                            """ + str(analysis_data['patterns']['value_patterns'].get('placeholder_assignment', 0)) + """,
                            """ + str(analysis_data['patterns']['value_patterns'].get('placeholder_return', 0)) + """
                        ],
                        backgroundColor: 'rgba(0, 122, 204, 0.8)',
                        borderColor: 'rgba(0, 122, 204, 1)',
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        y: {{
                            beginAtZero: true
                        }}
                    }},
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Placeholder Type Distribution'
                        }}
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report generated: {output_file}")

def main():
    """Main function to generate comprehensive report."""
    json_file = "comprehensive_placeholder_report.json"
    
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found. Please run the placeholder detector first.")
        return
    
    print("Loading placeholder detection data...")
    data = load_placeholder_data(json_file)
    
    if not data:
        print("Failed to load data. Exiting.")
        return
    
    print("Analyzing placeholder patterns...")
    patterns = analyze_placeholder_patterns(data['placeholders'])
    
    print("Analyzing file categories...")
    file_analysis = analyze_file_categories(data['placeholders'])
    
    print("Identifying critical files...")
    critical_files = identify_critical_files(file_analysis, threshold=10)
    
    print("Analyzing placeholder contexts...")
    contexts = analyze_placeholder_contexts(data['placeholders'])
    
    print("Generating recommendations...")
    recommendations = generate_recommendations({
        'patterns': patterns,
        'summary_stats': data['summary_stats'],
        'critical_files': critical_files
    })
    
    # Compile comprehensive analysis
    comprehensive_analysis = {
        'summary_stats': data['summary_stats'],
        'patterns': patterns,
        'file_analysis': file_analysis,
        'critical_files': critical_files,
        'contexts': contexts,
        'recommendations': recommendations,
        'generation_timestamp': datetime.now().isoformat()
    }
    
    # Save comprehensive analysis
    output_file = "comprehensive_placeholder_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_analysis, f, indent=2, default=str)
    
    print(f"Comprehensive analysis saved to: {output_file}")
    
    # Generate HTML report
    html_file = "comprehensive_placeholder_report.html"
    generate_html_report(comprehensive_analysis, html_file)
    
    # Print summary
    print("\n" + "="*80)
    print("COMPREHENSIVE PLACEHOLDER ANALYSIS SUMMARY")
    print("="*80)
    print(f"Total files analyzed: {data['summary_stats']['total_files_analyzed']}")
    print(f"Files with placeholders: {data['summary_stats']['files_with_placeholders']}")
    print(f"Total placeholders found: {data['summary_stats']['total_placeholders']}")
    print(f"Critical files (10+ placeholders): {len(critical_files)}")
    print(f"High severity items: {patterns['severity_distribution'].get('high', 0)}")
    print(f"Medium severity items: {patterns['severity_distribution'].get('medium', 0)}")
    print(f"Low severity items: {patterns['severity_distribution'].get('low', 0)}")
    
    print("\nTop 5 critical files:")
    for i, file_info in enumerate(critical_files[:5], 1):
        print(f"{i}. {file_info['file_path']}: {file_info['total_placeholders']} placeholders")
    
    print(f"\nReports generated:")
    print(f"- JSON analysis: {output_file}")
    print(f"- HTML report: {html_file}")

if __name__ == "__main__":
    main()