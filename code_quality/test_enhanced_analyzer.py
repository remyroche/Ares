#!/usr/bin/env python3
"""
Test script for the Enhanced Dead Code Analyzer

Demonstrates the capabilities of the enhanced analyzer with:
- DeadCodeRemover integration
- PyCG call graph analysis
- NetworkX dependency analysis
- Enhanced AST analysis
"""

import sys
from pathlib import Path
import logging

# Add code_quality to path
sys.path.insert(0, str(Path(__file__).parent))

from analyzers.enhanced_dead_code_analyzer import EnhancedDeadCodeAnalyzer
from core.config import AnalysisConfig

def setup_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_enhanced_analyzer():
    """Test the enhanced dead code analyzer."""
    print("🔍 Testing Enhanced Dead Code Analyzer")
    print("=" * 50)
    
    # Setup
    setup_logging()
    config = AnalysisConfig()
    analyzer = EnhancedDeadCodeAnalyzer(config)
    
    # Test on a smaller directory first
    test_directory = Path("/workspace/src/utils")
    
    if not test_directory.exists():
        print(f"❌ Test directory {test_directory} does not exist")
        return
    
    print(f"📁 Analyzing directory: {test_directory}")
    print(f"🔧 Available tools:")
    print(f"   - DeadCodeRemover: {'✅' if analyzer.deadcode_available else '❌'}")
    print(f"   - PyCG: {'✅' if analyzer.pycg_available else '❌'}")
    print(f"   - NetworkX: ✅")
    print(f"   - Enhanced AST: ✅")
    print()
    
    try:
        # Run analysis
        print("🚀 Starting analysis...")
        report = analyzer.analyze_directory(test_directory)
        
        # Display results
        print("\n📊 Analysis Results:")
        print(f"   Total Issues: {report.total_issues}")
        print(f"   Issues by Type: {report.issues_by_type}")
        print(f"   Issues by Severity: {dict(report.issues_by_severity)}")
        print(f"   Issues by Tool: {dict(report.issues_by_tool)}")
        print(f"   Call Graph Nodes: {report.call_graph.number_of_nodes()}")
        print(f"   Call Graph Edges: {report.call_graph.number_of_edges()}")
        print(f"   Dependency Graph Nodes: {report.dependency_graph.number_of_nodes()}")
        print(f"   Dependency Graph Edges: {report.dependency_graph.number_of_edges()}")
        
        # Show sample issues
        print("\n🔍 Sample Issues:")
        issue_count = 0
        for file_path, issues in report.issues_by_file.items():
            if issue_count >= 5:  # Show only first 5 files
                break
            print(f"\n   📄 {file_path}:")
            for issue in issues[:3]:  # Show first 3 issues per file
                print(f"      Line {issue.line_number}: {issue.description}")
                print(f"         Tool: {issue.tool_source}, Confidence: {issue.confidence}%")
            issue_count += 1
        
        # Generate visualizations
        output_dir = Path("/workspace/code_quality/test_output")
        print(f"\n🎨 Generating visualizations in {output_dir}...")
        analyzer.generate_visualization(report, output_dir)
        
        # Export results
        output_file = output_dir / "enhanced_analysis_results.json"
        print(f"💾 Exporting results to {output_file}...")
        analyzer.export_results(report, output_file)
        
        print("\n✅ Analysis complete!")
        print(f"📁 Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

def test_individual_components():
    """Test individual components of the enhanced analyzer."""
    print("\n🧪 Testing Individual Components")
    print("=" * 50)
    
    config = AnalysisConfig()
    analyzer = EnhancedDeadCodeAnalyzer(config)
    
    # Test AST analysis on a single file
    test_file = Path("/workspace/src/utils/data_quality_framework.py")
    if test_file.exists():
        print(f"🔍 Testing AST analysis on {test_file}")
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                source = f.read()
            
            import ast
            tree = ast.parse(source)
            issues = analyzer._analyze_file_ast(tree, test_file)
            
            print(f"   Found {len(issues)} issues")
            for issue in issues[:3]:
                print(f"   - {issue.description} (confidence: {issue.confidence}%)")
                
        except Exception as e:
            print(f"   ❌ AST analysis failed: {e}")
    
    # Test call graph building
    print(f"\n🔗 Testing call graph building...")
    try:
        test_files = [Path("/workspace/src/utils/data_quality_framework.py")]
        analyzer._build_comprehensive_call_graph(test_files)
        
        print(f"   Call graph nodes: {analyzer.call_graph.number_of_nodes()}")
        print(f"   Call graph edges: {analyzer.call_graph.number_of_edges()}")
        print(f"   Dependency graph nodes: {analyzer.dependency_graph.number_of_nodes()}")
        print(f"   Dependency graph edges: {analyzer.dependency_graph.number_of_edges()}")
        
    except Exception as e:
        print(f"   ❌ Call graph building failed: {e}")

if __name__ == "__main__":
    test_enhanced_analyzer()
    test_individual_components()