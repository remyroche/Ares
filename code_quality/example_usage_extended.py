"""
Extended Example Usage for Code Quality Tools

Demonstrates the new components:
- Complexity Analysis
- Dead Code Analysis  
- Error Reporting
- HTML Reporting
- Trend Tracking
"""

from pathlib import Path
import tempfile
import os

# Import the new components
from code_quality import (
    # New analyzers
    ComplexityAnalyzer,
    DeadCodeAnalyzer,
    
    # New reporters
    ErrorReporter,
    HTMLReporter,
    TrendReporter,
    
    # Quick access functions
    analyze_complexity,
    analyze_dead_code,
    generate_error_report,
    generate_html_report,
    track_quality_trends
)


def example_complexity_analysis():
    """Demonstrate complexity analysis."""
    print("🔍 Running Complexity Analysis...")
    
    # Analyze current directory
    try:
        complexity_results = analyze_complexity(".")
        print(f"✅ Complexity analysis completed for {len(complexity_results)} files")
        
        # Get summary
        analyzer = ComplexityAnalyzer()
        summary = analyzer.get_complexity_summary(complexity_results)
        
        print(f"📊 Complexity Summary:")
        print(f"  - Total files: {summary['total_files']}")
        print(f"  - Total functions: {summary['total_functions']}")
        print(f"  - Total classes: {summary['total_classes']}")
        print(f"  - High complexity functions: {summary['high_complexity_functions']}")
        print(f"  - Average complexity score: {summary['average_complexity_score']:.2f}")
        
        # Find issues
        issues = analyzer.find_complexity_issues(complexity_results)
        if issues:
            print(f"⚠️  Found {len(issues)} complexity issues:")
            for issue in issues[:3]:  # Show first 3
                print(f"    - {issue['file']}:{issue['line']} {issue['name']} (complexity: {issue['complexity']})")
        
        return complexity_results
        
    except Exception as e:
        print(f"❌ Complexity analysis failed: {e}")
        return {}


def example_dead_code_analysis():
    """Demonstrate dead code analysis."""
    print("\n🧹 Running Dead Code Analysis...")
    
    try:
        dead_code_results = analyze_dead_code(".")
        print(f"✅ Dead code analysis completed")
        
        # Get summary
        analyzer = DeadCodeAnalyzer()
        summary = analyzer.get_dead_code_summary(dead_code_results)
        
        print(f"📊 Dead Code Summary:")
        print(f"  - Total issues: {summary['total_issues']}")
        print(f"  - Files affected: {summary['files_affected']}")
        print(f"  - High confidence issues: {summary['high_confidence_issues']}")
        
        # Get recommendations
        recommendations = analyzer.generate_cleanup_recommendations(dead_code_results)
        if recommendations:
            print(f"💡 Recommendations:")
            for rec in recommendations[:3]:  # Show first 3
                print(f"    - {rec}")
        
        return dead_code_results
        
    except Exception as e:
        print(f"❌ Dead code analysis failed: {e}")
        return {}


def example_error_reporting(complexity_results, dead_code_results):
    """Demonstrate comprehensive error reporting."""
    print("\n📋 Generating Error Report...")
    
    try:
        # Create error reporter
        reporter = ErrorReporter()
        
        # Add results from different analyzers
        if complexity_results:
            complexity_issues = []
            for file_path, module in complexity_results.items():
                for func in module.functions:
                    if func.complexity > 10:  # High complexity threshold
                        complexity_issues.append({
                            'file_path': file_path,
                            'line_number': func.lineno,
                            'type': 'error',
                            'category': 'complexity',
                            'description': f"High complexity function: {func.name}",
                            'severity': 'high'
                        })
            
            if complexity_issues:
                reporter.add_complexity_issues(complexity_issues)
        
        if dead_code_results:
            dead_code_issues = []
            for issue in dead_code_results.issues_by_severity.get('high', []):
                dead_code_issues.append({
                    'file_path': issue.file_path,
                    'line_number': issue.line_number,
                    'type': 'warning',
                    'category': issue.issue_type,
                    'description': issue.description,
                    'severity': issue.severity
                })
            
            if dead_code_issues:
                reporter.add_dead_code_issues(dead_code_issues)
        
        # Generate report
        error_report = reporter.generate_report(".")
        print(f"✅ Error report generated")
        print(f"📊 Error Summary:")
        print(f"  - Total errors: {error_report.summary.total_errors}")
        print(f"  - Total warnings: {error_report.summary.total_warnings}")
        print(f"  - Files with errors: {error_report.summary.files_with_errors}")
        
        return error_report
        
    except Exception as e:
        print(f"❌ Error reporting failed: {e}")
        return None


def example_html_reporting(complexity_results, dead_code_results):
    """Demonstrate HTML report generation."""
    print("\n🌐 Generating HTML Report...")
    
    try:
        # Prepare data for HTML report
        report_data = {
            'complexity': complexity_results,
            'dead_code': dead_code_results
        }
        
        # Generate HTML report
        html_content = generate_html_report(report_data, "Code Quality Analysis Report")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(html_content)
            temp_file = f.name
        
        print(f"✅ HTML report generated and saved to: {temp_file}")
        print(f"🌐 Open this file in your browser to view the report")
        
        return temp_file
        
    except Exception as e:
        print(f"❌ HTML report generation failed: {e}")
        return None


def example_trend_tracking():
    """Demonstrate trend tracking."""
    print("\n📈 Setting up Trend Tracking...")
    
    try:
        # Create trend reporter
        trend_reporter = TrendReporter()
        
        # Add some sample data points
        sample_metrics = {
            'total_files': 25,
            'total_issues': 15,
            'quality_score': 85.5,
            'complexity_score': 78.2,
            'dead_code_issues': 8
        }
        
        trend_reporter.add_data_point(sample_metrics, "example_project")
        print(f"✅ Added data point for trend tracking")
        
        # Show available projects
        projects = trend_reporter.get_project_list()
        print(f"📊 Available projects: {projects}")
        
        return trend_reporter
        
    except Exception as e:
        print(f"❌ Trend tracking setup failed: {e}")
        return None


def main():
    """Run all examples."""
    print("🚀 Code Quality Tools - Extended Example")
    print("=" * 50)
    
    # Run complexity analysis
    complexity_results = example_complexity_analysis()
    
    # Run dead code analysis
    dead_code_results = example_dead_code_analysis()
    
    # Generate error report
    error_report = example_error_reporting(complexity_results, dead_code_results)
    
    # Generate HTML report
    html_file = example_html_reporting(complexity_results, dead_code_results)
    
    # Set up trend tracking
    trend_reporter = example_trend_tracking()
    
    print("\n🎉 All examples completed!")
    print("\n📝 Summary of what was demonstrated:")
    print("  ✅ Complexity analysis with Radon")
    print("  ✅ Dead code detection with Vulture")
    print("  ✅ Comprehensive error reporting")
    print("  ✅ Beautiful HTML report generation")
    print("  ✅ Trend tracking setup")
    
    if html_file:
        print(f"\n🌐 HTML report available at: {html_file}")
    
    print("\n💡 Next steps:")
    print("  - Run these analyses on your own code")
    print("  - Customize the configuration")
    print("  - Set up automated trend tracking")
    print("  - Integrate with your CI/CD pipeline")


if __name__ == "__main__":
    main()