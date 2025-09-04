#!/usr/bin/env python3
"""
Test script for enhanced dead code analysis
"""

import sys
import os
from pathlib import Path

# Add the code_quality directory to Python path
code_quality_path = Path(__file__).parent / "code_quality"
sys.path.insert(0, str(code_quality_path))

# Now we can import the modules
try:
    from analyzers.dead_code_analyzer import DeadCodeAnalyzer
    from visualizers.dependency_visualizer import DependencyVisualizer
    from visualizers.dashboard_generator import DashboardGenerator
    print("✅ Successfully imported all required modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_enhanced_analysis():
    """Test the enhanced dead code analysis on the current workspace"""
    
    print("🚀 Starting Enhanced Dead Code Analysis Test")
    print("=" * 60)
    
    # Initialize the analyzer
    analyzer = DeadCodeAnalyzer()
    
    # Analyze the current workspace
    workspace_path = Path.cwd()
    print(f"📁 Analyzing workspace: {workspace_path}")
    
    try:
        # Run the analysis
        report = analyzer.analyze_directory(workspace_path)
        
        print(f"\n📊 Analysis Results:")
        print(f"  - Total issues found: {len(report.issues)}")
        print(f"  - High severity: {len(report.issues_by_severity.get('high', []))}")
        print(f"  - Medium severity: {len(report.issues_by_severity.get('medium', []))}")
        print(f"  - Low severity: {len(report.issues_by_severity.get('low', []))}")
        print(f"  - Deprecated issues: {len(report.deprecated_issues)}")
        
        if report.impact_analysis:
            print(f"  - Impact analysis completed: {report.impact_analysis.get('total_impact_score', 0)} total score")
        
        if report.removal_plan:
            print(f"  - Removal plan generated: {report.removal_plan.get('total_time_savings', {}).get('hours', 0)} hours estimated savings")
        
        # Test visualization generation
        print(f"\n🎨 Testing Visualizations:")
        
        # Create a simple test visualization
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Create a simple test chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Sample data
            categories = ['Dead Code', 'Unused Imports', 'Deprecated Code', 'Unreachable Code']
            counts = [
                len([i for i in report.issues if 'dead' in i.issue_type.lower()]),
                len([i for i in report.issues if 'import' in i.issue_type.lower()]),
                len(report.deprecated_issues),
                len([i for i in report.issues if 'unreachable' in i.issue_type.lower()])
            ]
            
            colors = ['#ff4757', '#ffa502', '#ff6348', '#747d8c']
            bars = ax.bar(categories, counts, color=colors)
            
            ax.set_title('Dead Code Analysis Results', fontsize=16, fontweight='bold')
            ax.set_ylabel('Number of Issues', fontsize=12)
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{count}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # Save the test chart
            output_dir = Path("test_output")
            output_dir.mkdir(exist_ok=True)
            
            chart_path = output_dir / "test_dead_code_analysis.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✅ Test chart saved: {chart_path}")
            
        except ImportError:
            print("  ⚠️  Matplotlib not available - visualizations will be skipped")
        except Exception as e:
            print(f"  ❌ Visualization error: {e}")
        
        # Generate a simple HTML report
        print(f"\n📄 Generating HTML Report:")
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Dead Code Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
        .card h3 {{ margin: 0 0 10px 0; font-size: 24px; }}
        .card p {{ margin: 0; font-size: 18px; }}
        .section {{ margin-bottom: 30px; }}
        .section h2 {{ color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
        .issue {{ background: #f8f9fa; border-left: 4px solid #ff4757; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .issue.medium {{ border-left-color: #ffa502; }}
        .issue.low {{ border-left-color: #2ed573; }}
        .issue h4 {{ margin: 0 0 10px 0; color: #333; }}
        .issue p {{ margin: 5px 0; color: #666; }}
        .deprecated {{ background: #fff3cd; border-left-color: #ffc107; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Enhanced Dead Code Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Workspace: {workspace_path}</p>
        </div>
        
        <div class="summary">
            <div class="card">
                <h3>{len(report.issues)}</h3>
                <p>Total Issues</p>
            </div>
            <div class="card">
                <h3>{len(report.deprecated_issues)}</h3>
                <p>Deprecated Items</p>
            </div>
            <div class="card">
                <h3>{len(report.issues_by_severity.get('high', []))}</h3>
                <p>High Priority</p>
            </div>
            <div class="card">
                <h3>{report.impact_analysis.get('total_impact_score', 0) if report.impact_analysis else 0}</h3>
                <p>Impact Score</p>
            </div>
        </div>
        
        <div class="section">
            <h2>💀 Dead Code Issues</h2>
            {''.join([f'''
            <div class="issue {issue.severity}">
                <h4>{issue.issue_type.replace('_', ' ').title()}</h4>
                <p><strong>File:</strong> {issue.file_path}</p>
                <p><strong>Line:</strong> {issue.line_number}</p>
                <p><strong>Description:</strong> {issue.description}</p>
                <p><strong>Confidence:</strong> {issue.confidence}%</p>
            </div>
            ''' for issue in report.issues[:10]])}
            {f'<p><em>... and {len(report.issues) - 10} more issues</em></p>' if len(report.issues) > 10 else ''}
        </div>
        
        <div class="section">
            <h2>⚠️ Deprecated Code</h2>
            {''.join([f'''
            <div class="issue deprecated">
                <h4>{issue.deprecated_type.replace('_', ' ').title()}</h4>
                <p><strong>File:</strong> {issue.file_path}</p>
                <p><strong>Line:</strong> {issue.line_number}</p>
                <p><strong>Reason:</strong> {issue.reason}</p>
                <p><strong>Alternative:</strong> {issue.alternative or 'No alternative provided'}</p>
            </div>
            ''' for issue in report.deprecated_issues[:5]])}
            {f'<p><em>... and {len(report.deprecated_issues) - 5} more deprecated items</em></p>' if len(report.deprecated_issues) > 5 else ''}
        </div>
        
        <div class="section">
            <h2>📈 Impact Analysis</h2>
            {f'''
            <p><strong>Total Impact Score:</strong> {report.impact_analysis.get('total_impact_score', 0)}</p>
            <p><strong>High Impact Issues:</strong> {len(report.impact_analysis.get('high_impact_issues', []))}</p>
            <p><strong>Medium Impact Issues:</strong> {len(report.impact_analysis.get('medium_impact_issues', []))}</p>
            <p><strong>Low Impact Issues:</strong> {len(report.impact_analysis.get('low_impact_issues', []))}</p>
            ''' if report.impact_analysis else '<p>No impact analysis available</p>'}
        </div>
        
        <div class="section">
            <h2>🗓️ Removal Plan</h2>
            {f'''
            <p><strong>Estimated Time Savings:</strong> {report.removal_plan.get('total_time_savings', {}).get('hours', 0)} hours</p>
            <p><strong>Lines of Code to Remove:</strong> {report.removal_plan.get('total_time_savings', {}).get('lines', 0)}</p>
            <p><strong>Removal Phases:</strong> {len(report.removal_plan.get('phases', []))}</p>
            ''' if report.removal_plan else '<p>No removal plan available</p>'}
        </div>
    </div>
</body>
</html>
        """
        
        html_path = output_dir / "enhanced_dead_code_report.html"
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        print(f"  ✅ HTML report saved: {html_path}")
        
        print(f"\n🎉 Analysis Complete!")
        print(f"📁 Output directory: {output_dir.absolute()}")
        print(f"📊 Chart: {chart_path if 'chart_path' in locals() else 'Not generated'}")
        print(f"🌐 Report: {html_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    from datetime import datetime
    success = test_enhanced_analysis()
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)