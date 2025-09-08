#!/usr/bin/env python3
"""
Test script for the enhanced dead code analyzers.

This script demonstrates the new Multi-Modal and Context-Aware
dead code analysis capabilities.
"""

import sys
from pathlib import Path

# Add the code_quality directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from analyzers.multi_modal_dead_code_analyzer import MultiModalDeadCodeAnalyzer
from analyzers.context_aware_dead_code_analyzer import ContextAwareDeadCodeAnalyzer
from analyzers.framework_detector import FrameworkDetector
from analyzers.pattern_analyzer import PatternAnalyzer
from core.config import AnalysisConfig


def test_framework_detector():
    """Test the framework detector."""
    print("=" * 60)
    print("Testing Framework Detector")
    print("=" * 60)
    
    detector = FrameworkDetector()
    project_root = Path(__file__).parent
    
    try:
        context = detector.detect_frameworks(project_root)
        
        print(f"Project type: {context.project_type}")
        print(f"Frameworks detected: {len(context.frameworks)}")
        
        for framework in context.frameworks:
            print(f"  - {framework.framework_name}: {framework.confidence:.2f} confidence")
            print(f"    Files: {len(framework.files_involved)}")
            print(f"    Patterns: {', '.join(framework.patterns_found[:3])}")
        
        print(f"Development patterns: {', '.join(context.development_patterns)}")
        print(f"Build tools: {', '.join(context.build_tools)}")
        print(f"Testing frameworks: {', '.join(context.testing_frameworks)}")
        
        return context
        
    except Exception as e:
        print(f"Framework detection failed: {e}")
        return None


def test_pattern_analyzer():
    """Test the pattern analyzer."""
    print("\n" + "=" * 60)
    print("Testing Pattern Analyzer")
    print("=" * 60)
    
    analyzer = PatternAnalyzer()
    project_root = Path(__file__).parent
    
    try:
        result = analyzer.analyze_patterns(project_root)
        
        print(f"Design patterns: {len(result.design_patterns)}")
        print(f"Framework patterns: {len(result.framework_patterns)}")
        print(f"Anti-patterns: {len(result.anti_patterns)}")
        print(f"Usage patterns: {len(result.usage_patterns)}")
        
        if result.recommendations:
            print(f"\nRecommendations:")
            for i, rec in enumerate(result.recommendations, 1):
                print(f"  {i}. {rec}")
        
        return result
        
    except Exception as e:
        print(f"Pattern analysis failed: {e}")
        return None


def test_multi_modal_analyzer():
    """Test the multi-modal analyzer."""
    print("\n" + "=" * 60)
    print("Testing Multi-Modal Dead Code Analyzer")
    print("=" * 60)
    
    config = AnalysisConfig()
    analyzer = MultiModalDeadCodeAnalyzer(config)
    project_root = Path(__file__).parent
    
    try:
        result = analyzer.analyze(project_root)
        
        print(f"Total analyzers: {result.total_analyzers}")
        print(f"Successful analyzers: {result.successful_analyzers}")
        print(f"Dead functions: {len(result.combined_dead_functions)}")
        print(f"Dead classes: {len(result.combined_dead_classes)}")
        print(f"Dead imports: {len(result.combined_dead_imports)}")
        print(f"Overall confidence: {result.consensus_scores.get('overall_confidence', 0):.2f}")
        print(f"Execution time: {result.execution_time:.2f} seconds")
        
        # Show some examples
        if result.combined_dead_functions:
            print(f"\nExample dead functions:")
            for func in result.combined_dead_functions[:3]:
                print(f"  - {func['name']} in {func['file']} (confidence: {func['confidence']:.2f})")
        
        return result
        
    except Exception as e:
        print(f"Multi-modal analysis failed: {e}")
        return None


def test_context_aware_analyzer():
    """Test the context-aware analyzer."""
    print("\n" + "=" * 60)
    print("Testing Context-Aware Dead Code Analyzer")
    print("=" * 60)
    
    config = AnalysisConfig()
    analyzer = ContextAwareDeadCodeAnalyzer(config)
    project_root = Path(__file__).parent
    
    try:
        result = analyzer.analyze(project_root)
        
        print(f"Primary framework: {result.framework_context.primary_framework.framework_name if result.framework_context.primary_framework else 'None'}")
        print(f"Framework confidence: {result.framework_context.primary_framework.confidence if result.framework_context.primary_framework else 0.0:.2f}")
        print(f"Context-aware dead functions: {len(result.context_aware_dead_functions)}")
        print(f"Context-aware dead classes: {len(result.context_aware_dead_classes)}")
        print(f"Context-aware dead imports: {len(result.context_aware_dead_imports)}")
        print(f"False positives filtered: {result.false_positives_filtered}")
        print(f"Context awareness score: {result.context_insights.get('context_effectiveness', {}).get('context_awareness_score', 0.0):.2f}")
        print(f"Overall confidence: {result.confidence_scores.get('overall', 0.0):.2f}")
        print(f"Execution time: {result.execution_time:.2f} seconds")
        
        if result.recommendations:
            print(f"\nRecommendations:")
            for i, rec in enumerate(result.recommendations, 1):
                print(f"  {i}. {rec}")
        
        return result
        
    except Exception as e:
        print(f"Context-aware analysis failed: {e}")
        return None


def main():
    """Run all tests."""
    print("Enhanced Dead Code Analyzers Test Suite")
    print("=" * 80)
    
    # Test individual components
    framework_context = test_framework_detector()
    pattern_result = test_pattern_analyzer()
    multi_modal_result = test_multi_modal_analyzer()
    context_aware_result = test_context_aware_analyzer()
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    tests_passed = 0
    total_tests = 4
    
    if framework_context:
        tests_passed += 1
        print("✅ Framework Detector: PASSED")
    else:
        print("❌ Framework Detector: FAILED")
    
    if pattern_result:
        tests_passed += 1
        print("✅ Pattern Analyzer: PASSED")
    else:
        print("❌ Pattern Analyzer: FAILED")
    
    if multi_modal_result:
        tests_passed += 1
        print("✅ Multi-Modal Analyzer: PASSED")
    else:
        print("❌ Multi-Modal Analyzer: FAILED")
    
    if context_aware_result:
        tests_passed += 1
        print("✅ Context-Aware Analyzer: PASSED")
    else:
        print("❌ Context-Aware Analyzer: FAILED")
    
    print(f"\nTests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The enhanced analyzers are working correctly.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
    
    return tests_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)