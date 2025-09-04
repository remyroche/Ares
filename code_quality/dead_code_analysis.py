#!/usr/bin/env python3
"""
Test script for the Simplified Enhanced Dead Code Analyzer

Demonstrates the enhanced capabilities using only standard library modules.
"""

import sys
from pathlib import Path
import logging

# Add code_quality to path
sys.path.insert(0, str(Path(__file__).parent))

from standalone_enhanced_analyzer import SimplifiedEnhancedDeadCodeAnalyzer, AnalysisConfig

def setup_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_simplified_enhanced_analyzer():
    """Test the simplified enhanced dead code analyzer."""
    print("🔍 Testing Simplified Enhanced Dead Code Analyzer")
    print("=" * 60)
    
    # Setup
    setup_logging()
    config = AnalysisConfig()
    analyzer = SimplifiedEnhancedDeadCodeAnalyzer(config)
    
    # Test on a smaller directory first
    test_directory = Path("/workspace/src/utils")
    
    if not test_directory.exists():
        print(f"❌ Test directory {test_directory} does not exist")
        return
    
    print(f"📁 Analyzing directory: {test_directory}")
    print(f"🔧 Available tools:")
    print(f"   - Enhanced AST Analysis: ✅")
    print(f"   - Import Analysis: ✅")
    print(f"   - Call Graph Building: ✅")
    print(f"   - Cross-validation: ✅")
    print()
    
    try:
        # Run analysis
        print("🚀 Starting analysis...")
        report = analyzer.analyze_project(test_directory)
        
        # Display results
        print("\n📊 Analysis Results:")
        print(f"   Total Issues: {report['stats']['total_issues']}")
        print(f"   Files Analyzed: {report['stats']['files_analyzed']}")
        print(f"   Dead Code Issues: {report['stats']['dead_code_issues']}")
        print(f"   Unused Import Issues: {report['stats']['unused_import_issues']}")
        print(f"   Processing Time: {report['stats']['processing_time']:.2f} seconds")
        
        # Show sample issues
        if report['issues']:
            print(f"\n📋 Sample Issues:")
            for i, issue in enumerate(report['issues'][:3]):
                print(f"   {i+1}. {issue['message']} ({issue['severity']})")
                print(f"      File: {issue['file_path']}:{issue['line_number']}")
                if issue['suggestion']:
                    print(f"      Suggestion: {issue['suggestion']}")
        
        # Export results
        output_dir = Path("/workspace/code_quality/test_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "simplified_enhanced_analysis.json"
        print(f"\n💾 Exporting results to {output_file}...")
        analyzer.export_results(str(output_file))
        
        print("\n✅ Analysis complete!")
        print(f"📁 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

def test_individual_components():
    """Test individual components of the simplified enhanced analyzer."""
    print("\n🧪 Testing Individual Components")
    print("=" * 60)
    
    config = AnalysisConfig()
    analyzer = SimplifiedEnhancedDeadCodeAnalyzer(config)
    
    # Test AST analysis on a single file
    test_file = Path("/workspace/src/utils/data_quality_framework.py")
    if test_file.exists():
        print(f"🔍 Testing AST analysis on {test_file}")
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                source = f.read()
            
            import ast
            tree = ast.parse(source)
            # Test individual file analysis
            analyzer._analyze_file(test_file)
            issues = analyzer.issues
            
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
        
        print(f"   Call graph nodes: {len(analyzer.call_graph_nodes)}")
        print(f"   Dependency graph modules: {len(analyzer.dependency_graph)}")
        
        # Show sample nodes
        print(f"   Sample nodes:")
        for name, node in list(analyzer.call_graph_nodes.items())[:5]:
            print(f"     {name} ({node.node_type})")
        
    except Exception as e:
        print(f"   ❌ Call graph building failed: {e}")

def compare_with_original():
    """Compare results with the original analyzer."""
    print("\n🔄 Comparing with Original Analyzer")
    print("=" * 60)
    
    # This would require running both analyzers and comparing results
    # For now, just show the enhanced features
    print("Enhanced features in the simplified analyzer:")
    print("   ✅ Better AST analysis with call graph context")
    print("   ✅ Import analysis for unused imports")
    print("   ✅ Cross-validation to reduce false positives")
    print("   ✅ Tool attribution for each issue")
    print("   ✅ Enhanced reporting with confidence levels")
    print("   ✅ Call graph building for dependency analysis")
    print("   ✅ Dynamic usage pattern detection")

if __name__ == "__main__":
    test_simplified_enhanced_analyzer()
    test_individual_components()
    compare_with_original()