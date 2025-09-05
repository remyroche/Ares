#!/usr/bin/env python3
"""
Pattern Analysis Script for Code Quality Issues

This script analyzes the detailed reports to identify patterns that can be automated.
"""

import json
import re
from collections import defaultdict, Counter
from pathlib import Path
import typing

def analyze_data_flow_patterns():
    """Analyze data flow issues for automation patterns."""
    print("="*60)
    print("DATA FLOW PATTERN ANALYSIS")
    print("="*60)
    
    # Load data flow report
    data_flow_file = "/workspace/code_quality/reports/interaction_mapping/data_flow_20250905_135425.json"
    
    try:
        with open(data_flow_file, 'r') as f:
            data = json.load(f)
        
        print(f"Total Issues: {data['results']['total_issues']}")
        print(f"Files Analyzed: {data['results']['stats']['files_analyzed']}")
        print(f"Files with Issues: {data['results']['stats']['files_with_issues']}")
        
        # Analyze issue types
        issues_by_type = data['results']['issues_by_type']
        print("\nIssue Breakdown:")
        for issue_type, count in issues_by_type.items():
            percentage = (count / data['results']['total_issues']) * 100
            print(f"  {issue_type}: {count} ({percentage:.1f}%)")
        
        # Analyze complexity distribution
        complexity_dist = data['results']['complexity_distribution']
        print(f"\nComplexity Distribution:")
        print(f"  Low: {complexity_dist['low']}")
        print(f"  Medium: {complexity_dist['medium']}")
        print(f"  High: {complexity_dist['high']}")
        
        return issues_by_type
        
    except Exception as e:
        print(f"Error loading data flow report: {e}")
        return {}

def analyze_circular_dependencies():
    """Analyze circular dependencies for patterns."""
    print("\n" + "="*60)
    print("CIRCULAR DEPENDENCY ANALYSIS")
    print("="*60)
    
    # Load call graph report
    call_graph_file = "/workspace/code_quality/reports/interaction_mapping/call_graph_20250905_135425.json"
    
    try:
        with open(call_graph_file, 'r') as f:
            data = json.load(f)
        
        print(f"Total Functions: {data['total_functions']}")
        print(f"Max Call Depth: {data['max_call_depth']}")
        print(f"Circular Calls: {data['circular_calls']}")
        
        # Analyze circular call patterns
        circular_calls = data['results']['circular_calls']
        print(f"\nCircular Call Patterns:")
        
        # Group by pattern type
        patterns = defaultdict(int)
        for call in circular_calls:
            if ' -> ' in call:
                source, target = call.split(' -> ')
                if source == target:
                    patterns['self_recursive'] += 1
                elif source.startswith('_') and target.startswith('_'):
                    patterns['private_methods'] += 1
                elif 'init' in source.lower() or 'init' in target.lower():
                    patterns['initialization'] += 1
                else:
                    patterns['other'] += 1
        
        for pattern, count in patterns.items():
            print(f"  {pattern}: {count}")
        
        return circular_calls
        
    except Exception as e:
        print(f"Error loading call graph report: {e}")
        return []

def analyze_dependency_patterns():
    """Analyze dependency patterns."""
    print("\n" + "="*60)
    print("DEPENDENCY PATTERN ANALYSIS")
    print("="*60)
    
    # Load dependency report
    dep_file = "/workspace/code_quality/reports/interaction_mapping/dependency_analysis_20250905_135425.json"
    
    try:
        with open(dep_file, 'r') as f:
            data = json.load(f)
        
        print(f"Total Modules: {data['results']['total_modules']}")
        print(f"Total Dependencies: {data['results']['total_dependencies']}")
        print(f"Internal Dependencies: {data['results']['internal_dependencies']}")
        print(f"External Dependencies: {data['results']['external_dependencies']}")
        
        # Analyze most common dependencies
        all_deps = []
        for module_data in data['results']['modules'].values():
            all_deps.extend(module_data['dependencies'])
        
        dep_counter = Counter(all_deps)
        print(f"\nTop 10 Most Common Dependencies:")
        for dep, count in dep_counter.most_common(10):
            print(f"  {dep}: {count}")
        
        return dep_counter
        
    except Exception as e:
        print(f"Error loading dependency report: {e}")
        return Counter()

def generate_automation_recommendations(issues_by_type, circular_calls, dep_counter):
    """Generate automation recommendations based on patterns."""
    print("\n" + "="*60)
    print("AUTOMATION RECOMMENDATIONS")
    print("="*60)
    
    print("\n1. UNUSED VARIABLES/PARAMETERS (11,753 issues):")
    print("   Patterns identified:")
    print("   - Unused parameters: 6,299 (45.4%)")
    print("   - Unused variables: 5,454 (39.3%)")
    print("   ")
    print("   Automation Strategy:")
    print("   - Create AST-based analyzer to identify unused parameters")
    print("   - Use 'unused' prefix for intentionally unused parameters")
    print("   - Implement auto-removal for clearly unused variables")
    print("   - Add linting rules to prevent future unused variables")
    
    print("\n2. POTENTIAL NONE ACCESS (1,293 issues):")
    print("   Automation Strategy:")
    print("   - Add None checks before variable access")
    print("   - Use optional chaining patterns")
    print("   - Implement type hints with Optional[]")
    print("   - Add runtime None validation")
    
    print("\n3. CIRCULAR DEPENDENCIES (140 issues):")
    print("   Patterns identified:")
    print("   - Self-recursive calls")
    print("   - Private method circular calls")
    print("   - Initialization circular dependencies")
    print("   ")
    print("   Automation Strategy:")
    print("   - Implement dependency injection patterns")
    print("   - Create interface abstractions")
    print("   - Use lazy loading for circular dependencies")
    print("   - Refactor initialization order")
    
    print("\n4. VARIABLE SHADOWING (410 issues):")
    print("   Automation Strategy:")
    print("   - Rename shadowed variables")
    print("   - Use more specific variable names")
    print("   - Implement scope-aware naming conventions")
    
    print("\n5. UNVALIDATED INPUT (402 issues):")
    print("   Automation Strategy:")
    print("   - Add input validation decorators")
    print("   - Implement type checking")
    print("   - Use schema validation libraries")
    print("   - Add runtime validation")

def main():
    """Main analysis function."""
    print("CODE QUALITY PATTERN ANALYSIS")
    print("Analyzing reports for automation opportunities...")
    
    # Analyze different aspects
    issues_by_type = analyze_data_flow_patterns()
    circular_calls = analyze_circular_dependencies()
    dep_counter = analyze_dependency_patterns()
    
    # Generate recommendations
    generate_automation_recommendations(issues_by_type, circular_calls, dep_counter)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()