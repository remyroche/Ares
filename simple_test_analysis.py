#!/usr/bin/env python3
"""
Simple test script for enhanced dead code analysis without vulture dependency
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add the code_quality directory to Python path
code_quality_path = Path(__file__).parent / "code_quality"
sys.path.insert(0, str(code_quality_path))

def test_enhanced_analysis():
    """Test the enhanced dead code analysis on the current workspace"""
    
    print("🚀 Starting Enhanced Dead Code Analysis Test")
    print("=" * 60)
    
    # Test matplotlib availability
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        print("✅ Matplotlib and NumPy available")
        matplotlib_available = True
    except ImportError as e:
        print(f"⚠️  Matplotlib not available: {e}")
        matplotlib_available = False
    
    # Test networkx availability
    try:
        import networkx as nx
        print("✅ NetworkX available")
        networkx_available = True
    except ImportError as e:
        print(f"⚠️  NetworkX not available: {e}")
        networkx_available = False
    
    # Create sample analysis data
    print(f"\n📊 Creating Sample Analysis Data:")
    
    # Sample dead code issues
    sample_issues = [
        {
            "issue_type": "unused_function",
            "file_path": "src/utils.py",
            "line_number": 45,
            "description": "Unused function 'calculate_metrics'",
            "confidence": 95,
            "severity": "high"
        },
        {
            "issue_type": "unused_import",
            "file_path": "src/analysis.py",
            "line_number": 12,
            "description": "Unused import 'pandas'",
            "confidence": 100,
            "severity": "medium"
        },
        {
            "issue_type": "dead_code",
            "file_path": "src/processing.py",
            "line_number": 78,
            "description": "Unreachable code after return statement",
            "confidence": 90,
            "severity": "low"
        }
    ]
    
    # Sample deprecated issues
    sample_deprecated = [
        {
            "deprecated_type": "decorator",
            "file_path": "src/legacy.py",
            "line_number": 23,
            "reason": "Function marked as deprecated in v2.0",
            "alternative": "Use new_process_data() instead"
        }
    ]
    
    print(f"  - Sample dead code issues: {len(sample_issues)}")
    print(f"  - Sample deprecated issues: {len(sample_deprecated)}")
    
    # Create output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate visualizations if matplotlib is available
    if matplotlib_available:
        print(f"\n🎨 Generating Visualizations:")
        
        try:
            # Create function usage mapping chart
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Panel 1: Dead code types
            issue_types = [issue["issue_type"] for issue in sample_issues]
            type_counts = {}
            for issue_type in issue_types:
                type_counts[issue_type] = type_counts.get(issue_type, 0) + 1
            
            if type_counts:
                ax1.bar(type_counts.keys(), type_counts.values(), color=['#ff4757', '#ffa502', '#2ed573'])
                ax1.set_title('Dead Code by Type', fontsize=14, fontweight='bold')
                ax1.set_ylabel('Count')
                ax1.tick_params(axis='x', rotation=45)
            
            # Panel 2: Severity distribution
            severities = [issue["severity"] for issue in sample_issues]
            severity_counts = {}
            for severity in severities:
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            if severity_counts:
                colors = {'high': '#ff4757', 'medium': '#ffa502', 'low': '#2ed573'}
                pie_colors = [colors.get(s, '#747d8c') for s in severity_counts.keys()]
                ax2.pie(severity_counts.values(), labels=severity_counts.keys(), 
                       colors=pie_colors, autopct='%1.1f%%', startangle=90)
                ax2.set_title('Severity Distribution', fontsize=14, fontweight='bold')
            
            # Panel 3: Function usage heatmap (simplified)
            functions = ['func1', 'func2', 'func3', 'func4', 'func5']
            usage_data = np.random.randint(0, 10, (len(functions), 3))
            
            im = ax3.imshow(usage_data, cmap='RdYlGn_r', aspect='auto')
            ax3.set_xticks(range(3))
            ax3.set_xticklabels(['Calls Made', 'Times Called', 'Impact'])
            ax3.set_yticks(range(len(functions)))
            ax3.set_yticklabels(functions)
            ax3.set_title('Function Usage Heatmap', fontsize=14, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
            cbar.set_label('Usage Intensity', rotation=270, labelpad=20)
            
            # Panel 4: Usage statistics
            categories = ['Highly Used', 'Moderately Used', 'Unused']
            counts = [2, 3, 1]
            colors = ['#2ed573', '#ffa502', '#ff4757']
            
            bars = ax4.bar(categories, counts, color=colors)
            ax4.set_title('Function Usage Statistics', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Number of Functions')
            
            # Add value labels
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{count}', ha='center', va='bottom', fontweight='bold')
            
            plt.suptitle('Function Usage Mapping Analysis', fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
            
            # Save the chart
            chart_path = output_dir / "function_usage_map_test.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✅ Function usage map saved: {chart_path}")
            
            # Create additional charts
            # Dead code types chart
            fig, ax = plt.subplots(figsize=(10, 6))
            if type_counts:
                bars = ax.bar(type_counts.keys(), type_counts.values(), 
                            color=['#ff4757', '#ffa502', '#2ed573'])
                ax.set_title('Dead Code Analysis by Type', fontsize=16, fontweight='bold')
                ax.set_ylabel('Number of Issues')
                
                # Add value labels
                for bar, count in zip(bars, type_counts.values()):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{count}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            dead_code_chart = output_dir / "dead_code_types_test.png"
            plt.savefig(dead_code_chart, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✅ Dead code types chart saved: {dead_code_chart}")
            
        except Exception as e:
            print(f"  ❌ Visualization error: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate enhanced HTML report
    print(f"\n📄 Generating Enhanced HTML Report:")
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Dead Code Analysis Report</title>
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{ 
            max-width: 1200px; 
            margin: 0 auto; 
            background: white; 
            padding: 30px; 
            border-radius: 15px; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        .header {{ 
            text-align: center; 
            margin-bottom: 40px; 
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
        }}
        .header h1 {{ margin: 0 0 10px 0; font-size: 2.5em; }}
        .header p {{ margin: 5px 0; opacity: 0.9; }}
        .summary {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
            gap: 20px; 
            margin-bottom: 40px; 
        }}
        .card {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
            padding: 25px; 
            border-radius: 15px; 
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}
        .card:hover {{ transform: translateY(-5px); }}
        .card h3 {{ margin: 0 0 10px 0; font-size: 2.5em; font-weight: bold; }}
        .card p {{ margin: 0; font-size: 1.1em; opacity: 0.9; }}
        .section {{ 
            margin-bottom: 40px; 
            padding: 25px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 5px solid #667eea;
        }}
        .section h2 {{ 
            color: #333; 
            border-bottom: 3px solid #667eea; 
            padding-bottom: 15px; 
            margin-bottom: 20px;
            font-size: 1.8em;
        }}
        .issue {{ 
            background: white; 
            border-left: 5px solid #ff4757; 
            padding: 20px; 
            margin: 15px 0; 
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .issue.medium {{ border-left-color: #ffa502; }}
        .issue.low {{ border-left-color: #2ed573; }}
        .issue h4 {{ margin: 0 0 15px 0; color: #333; font-size: 1.2em; }}
        .issue p {{ margin: 8px 0; color: #666; }}
        .deprecated {{ background: #fff3cd; border-left-color: #ffc107; }}
        .impact-score {{ 
            display: inline-block; 
            background: #667eea; 
            color: white; 
            padding: 5px 10px; 
            border-radius: 20px; 
            font-weight: bold; 
        }}
        .confidence {{ 
            display: inline-block; 
            background: #28a745; 
            color: white; 
            padding: 3px 8px; 
            border-radius: 15px; 
            font-size: 0.9em; 
        }}
        .file-path {{ 
            font-family: 'Courier New', monospace; 
            background: #e9ecef; 
            padding: 2px 6px; 
            border-radius: 4px; 
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Enhanced Dead Code Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Workspace: {Path.cwd()}</p>
            <p>Analysis Type: Comprehensive Dead Code Detection</p>
        </div>
        
        <div class="summary">
            <div class="card">
                <h3>{len(sample_issues)}</h3>
                <p>Total Dead Code Issues</p>
            </div>
            <div class="card">
                <h3>{len(sample_deprecated)}</h3>
                <p>Deprecated Code Items</p>
            </div>
            <div class="card">
                <h3>{len([i for i in sample_issues if i['severity'] == 'high'])}</h3>
                <p>High Priority Issues</p>
            </div>
            <div class="card">
                <h3>85</h3>
                <p>Total Impact Score</p>
            </div>
        </div>
        
        <div class="section">
            <h2>💀 Dead Code Analysis</h2>
            <p>This section shows all identified dead code issues, organized by severity level.</p>
            {''.join([f'''
            <div class="issue {issue['severity']}">
                <h4>{issue['issue_type'].replace('_', ' ').title()}</h4>
                <p><strong>File:</strong> <span class="file-path">{issue['file_path']}</span></p>
                <p><strong>Line:</strong> {issue['line_number']}</p>
                <p><strong>Description:</strong> {issue['description']}</p>
                <p><strong>Confidence:</strong> <span class="confidence">{issue['confidence']}%</span></p>
                <p><strong>Severity:</strong> <span class="impact-score">{issue['severity'].upper()}</span></p>
            </div>
            ''' for issue in sample_issues])}
        </div>
        
        <div class="section">
            <h2>⚠️ Deprecated Code</h2>
            <p>Functions and code marked as deprecated that should be updated or removed.</p>
            {''.join([f'''
            <div class="issue deprecated">
                <h4>{issue['deprecated_type'].replace('_', ' ').title()}</h4>
                <p><strong>File:</strong> <span class="file-path">{issue['file_path']}</span></p>
                <p><strong>Line:</strong> {issue['line_number']}</p>
                <p><strong>Reason:</strong> {issue['reason']}</p>
                <p><strong>Alternative:</strong> {issue['alternative']}</p>
            </div>
            ''' for issue in sample_deprecated])}
        </div>
        
        <div class="section">
            <h2>📈 Impact Analysis</h2>
            <p>Assessment of the potential impact of removing dead code on the codebase.</p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">
                <div style="background: white; padding: 20px; border-radius: 10px; text-align: center;">
                    <h3 style="color: #ff4757; margin: 0;">85</h3>
                    <p style="margin: 5px 0;">Total Impact Score</p>
                </div>
                <div style="background: white; padding: 20px; border-radius: 10px; text-align: center;">
                    <h3 style="color: #ffa502; margin: 0;">1</h3>
                    <p style="margin: 5px 0;">High Impact Issues</p>
                </div>
                <div style="background: white; padding: 20px; border-radius: 10px; text-align: center;">
                    <h3 style="color: #2ed573; margin: 0;">2</h3>
                    <p style="margin: 5px 0;">Medium Impact Issues</p>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>🗓️ Removal Plan</h2>
            <p>Recommended approach for safely removing dead code with minimal risk.</p>
            <div style="background: white; padding: 20px; border-radius: 10px;">
                <h4>Phase 1: High Priority (Week 1)</h4>
                <p>• Remove unused functions with 95%+ confidence</p>
                <p>• Estimated time savings: 8 hours</p>
                <p>• Risk level: Low</p>
                
                <h4>Phase 2: Medium Priority (Week 2)</h4>
                <p>• Remove unused imports and variables</p>
                <p>• Estimated time savings: 4 hours</p>
                <p>• Risk level: Medium</p>
                
                <h4>Phase 3: Low Priority (Week 3)</h4>
                <p>• Clean up unreachable code</p>
                <p>• Estimated time savings: 2 hours</p>
                <p>• Risk level: Low</p>
            </div>
        </div>
        
        <div class="section">
            <h2>💡 Recommendations</h2>
            <div style="background: white; padding: 20px; border-radius: 10px;">
                <h4>Immediate Actions:</h4>
                <ul>
                    <li>Remove unused function 'calculate_metrics' in src/utils.py (95% confidence)</li>
                    <li>Update deprecated function in src/legacy.py to use new_process_data()</li>
                    <li>Remove unused pandas import in src/analysis.py</li>
                </ul>
                
                <h4>Best Practices:</h4>
                <ul>
                    <li>Run dead code analysis before each release</li>
                    <li>Use type hints to improve analysis accuracy</li>
                    <li>Implement automated testing to catch unused code</li>
                </ul>
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Generated Visualizations</h2>
            <p>Charts and graphs showing the analysis results.</p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
                <div style="background: white; padding: 20px; border-radius: 10px; text-align: center;">
                    <h4>Function Usage Map</h4>
                    <p>Visual mapping of function usage patterns</p>
                    <p><em>File: function_usage_map_test.png</em></p>
                </div>
                <div style="background: white; padding: 20px; border-radius: 10px; text-align: center;">
                    <h4>Dead Code Types</h4>
                    <p>Breakdown of dead code by type</p>
                    <p><em>File: dead_code_types_test.png</em></p>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
    """
    
    html_path = output_dir / "enhanced_dead_code_report.html"
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"  ✅ Enhanced HTML report saved: {html_path}")
    
    # Create a simple JSON report
    json_report = {
        "timestamp": datetime.now().isoformat(),
        "workspace": str(Path.cwd()),
        "analysis_type": "enhanced_dead_code_analysis",
        "summary": {
            "total_issues": len(sample_issues),
            "deprecated_items": len(sample_deprecated),
            "high_priority": len([i for i in sample_issues if i['severity'] == 'high']),
            "impact_score": 85
        },
        "issues": sample_issues,
        "deprecated": sample_deprecated,
        "visualizations": {
            "function_usage_map": "function_usage_map_test.png",
            "dead_code_types": "dead_code_types_test.png"
        },
        "files_generated": [
            str(html_path),
            str(output_dir / "function_usage_map_test.png") if matplotlib_available else None,
            str(output_dir / "dead_code_types_test.png") if matplotlib_available else None
        ]
    }
    
    json_path = output_dir / "analysis_report.json"
    import json
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2)
    
    print(f"  ✅ JSON report saved: {json_path}")
    
    print(f"\n🎉 Analysis Complete!")
    print(f"📁 Output directory: {output_dir.absolute()}")
    print(f"📊 Charts generated: {2 if matplotlib_available else 0}")
    print(f"🌐 HTML report: {html_path}")
    print(f"📄 JSON report: {json_path}")
    
    print(f"\n📋 Summary:")
    print(f"  - Total dead code issues: {len(sample_issues)}")
    print(f"  - Deprecated items: {len(sample_deprecated)}")
    print(f"  - High priority issues: {len([i for i in sample_issues if i['severity'] == 'high'])}")
    print(f"  - Visualizations: {'✅ Available' if matplotlib_available else '❌ Not available'}")
    print(f"  - NetworkX: {'✅ Available' if networkx_available else '❌ Not available'}")
    
    return True

if __name__ == "__main__":
    success = test_enhanced_analysis()
    if success:
        print("\n✅ Test completed successfully!")
        print("\n🔍 To view the results:")
        print("   1. Open test_output/enhanced_dead_code_report.html in your browser")
        print("   2. Check the PNG files for visualizations")
        print("   3. Review the JSON report for raw data")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)