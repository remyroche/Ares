#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Enhanced Dead Code Detection Demo

Demonstrates the enhanced dead code detection capabilities including:
- Deprecated code detection
- Dynamic import analysis
- Conditional dead code detection
- Impact analysis
- Dependency-aware removal planning
"""

import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analyzers.dead_code_analyzer import DeadCodeAnalyzer
from core.config import get_default_config
import logging


def create_sample_code():
    """Create sample code files to demonstrate dead code detection."""
    sample_dir = Path("sample_dead_code")
    sample_dir.mkdir(exist_ok = True)
    
    # Sample file with various dead code patterns
    sample_file = sample_dir / "sample_dead_code.py"
    sample_file.write_text('''
"""
Sample file demonstrating various dead code patterns.
"""

import os
import json  # This import is unused
from typing import List, Dict  # Dict is unused

# Deprecated function with decorator
@deprecated(reason="Use new_function instead", version="2.0", alternative="new_function")
def old_function():
    """This function is deprecated."""
    return "old"

# Function that raises deprecation warning
def deprecated_function():
    """This function is deprecated."""
    raise DeprecationWarning("This function will be removed in v2.0")

# Unused function
def unused_function():
    """This function is never called."""
    return "unused"

# Unused variable
unused_variable = "never used"

# Function with unused parameter
def function_with_unused_param(used_param, unused_param):
    """Function with an unused parameter."""
    return used_param

# Function with dynamic import
def dynamic_import_example():
    """Function using dynamic imports."""
    module = __import__('os')
    return module.path

# Function with importlib
def importlib_example():
    """Function using importlib."""
    module = importlib.import_module('json')
    return module

# Function with unreachable code
def unreachable_code_example():
    """Function with unreachable code."""
    return "this will be returned"
    tprint("This will never be executed")  # Unreachable code

# Function with conditional dead code
def conditional_dead_code():
    """Function with conditional dead code."""
    if False:  # This condition is always False
        tprint("This code is never executed")
    return "executed"

# Used function
def used_function():
    """This function is actually used."""
    return "used"

# Main function that uses some functions
def main():
    """Main function."""
    result = used_function()
    old_function()  # Using deprecated function
    deprecated_function()  # Using deprecated function
    return result

if __name__ == "__main__":
    main()
''')

    # Create a second file with imports
    second_file = sample_dir / "imports_example.py"
    second_file.write_text('''
"""
File demonstrating import patterns.
"""

from sample_dead_code import used_function, old_function
import sys  # This import is unused

import importlib




def example_usage():
    """Example of using imported functions."""
    result = used_function()
    old_function()  # Using deprecated function
    return result
''')

    return sample_dir


def demonstrate_enhanced_dead_code_detection():
    """Demonstrate enhanced dead code detection capabilities."""
    tprint("ENHANCED DEAD CODE DETECTION DEMO")
    tprint("=" * 50)
    
    # Create sample code
    sample_dir = create_sample_code()
    tprint(f"Created sample code in: {sample_dir}")
    
    # Initialize analyzer
    analyzer = DeadCodeAnalyzer(config)
    
    tprint(f"\nAnalyzing directory: {sample_dir}")
    tprint("-" * 30)
    
    # Run comprehensive analysis
    report = analyzer.analyze_directory(sample_dir)
    
    # Display results
    tprint(f"\nDEAD CODE ANALYSIS RESULTS")
    tprint("=" * 30)
    tprint(f"Total Issues: {report.total_issues}")
    tprint(f"Deprecated Issues: {len(report.deprecated_issues or [])}")
    tprint(f"Potential Lines Removed: {report.potential_savings.get('total_lines', 0)}")
    
    # Show issues by type
    tprint(f"\nIssues by Type:")
    for issue_type, count in report.issues_by_type.items():
        tprint(f"  {issue_type}: {count}")
    
    # Show issues by severity
    tprint(f"\nIssues by Severity:")
    for severity, issues in report.issues_by_severity.items():
        tprint(f"  {severity}: {len(issues)}")
    
    # Show deprecated code
    if report.deprecated_issues:
        tprint(f"\nDEPRECATED CODE DETECTED:")
        tprint("-" * 25)
        for issue in report.deprecated_issues:
            tprint(f"  {issue.file_path}:{issue.line_number}")
            tprint(f"    Type: {issue.deprecated_type}")
            tprint(f"    Description: {issue.description}")
            tprint(f"    Reason: {issue.deprecation_reason}")
            if issue.removal_version:
                tprint(f"    Removal Version: {issue.removal_version}")
            if issue.alternative:
                tprint(f"    Alternative: {issue.alternative}")
            tprint()
    
    # Show impact analysis
    if report.impact_analysis:
        tprint(f"\nIMPACT ANALYSIS:")
        tprint("-" * 15)
        impact = report.impact_analysis
        tprint(f"High Impact Issues: {len(impact.get('high_impact', []))}")
        tprint(f"Medium Impact Issues: {len(impact.get('medium_impact', []))}")
        tprint(f"Low Impact Issues: {len(impact.get('low_impact', []))}")
        tprint(f"Total Impact Score: {impact.get('total_impact_score', 0)}")
        
        # Show removal order
        removal_order = impact.get('removal_order', [])
        if removal_order:
            tprint(f"\nRecommended Removal Order (Top 5):")
            for i, issue in enumerate(removal_order[:5], 1):
                tprint(f"  {i}. {issue.file_path}:{issue.line_number} - {issue.description}")
                tprint(f"     Impact: {issue.removal_impact}, Confidence: {issue.confidence}%")
        
        # Show dependency analysis
        if "dependency_analysis" in impact:
            dep_analysis = impact["dependency_analysis"]
            tprint(f"\nDEPENDENCY ANALYSIS:")
            tprint("-" * 20)
            tprint(f"Dependency Chains: {len(dep_analysis.get('dependency_chains', []))}")
            tprint(f"Risky Removals: {len(dep_analysis.get('risky_removals', []))}")
            tprint(f"Removal Groups: {len(dep_analysis.get('removal_groups', []))}")
            
            # Show risky removals
            risky_removals = dep_analysis.get('risky_removals', [])
            if risky_removals:
                tprint(f"\nRisky Removals:")
                for removal in risky_removals[:3]:  # Show top 3
                    issue = removal.get('issue', {})
                    tprint(f"  {issue.get('file_path', '')}:{issue.get('line_number', '')}")
                    tprint(f"    Risk: {removal.get('risk_level', 'unknown')}")
                    tprint(f"    Reason: {removal.get('risk_reason', 'unknown')}")
        
        # Show removal plan
        if "removal_plan" in impact:
            removal_plan = impact["removal_plan"]
            tprint(f"\nREMOVAL PLAN:")
            tprint("-" * 12)
            time_savings = removal_plan.get('estimated_time_savings', {})
            tprint(f"Estimated Time Savings: {time_savings.get('estimated_hours_saved', 0):.1f} hours")
            tprint(f"Estimated Days Saved: {time_savings.get('estimated_days_saved', 0):.1f} days")
            tprint(f"Lines to Remove: {time_savings.get('total_lines_removed', 0)}")
            
            # Show removal phases
            phases = removal_plan.get('removal_phases', [])
            if phases:
                tprint(f"\nRemoval Phases:")
                for phase in phases:
                    tprint(f"  Phase {phase.get('phase', '')}: {phase.get('name', '')}")
                    tprint(f"    Description: {phase.get('description', '')}")
                    tprint(f"    Effort: {phase.get('estimated_effort', '')}")
                    tprint(f"    Risk: {phase.get('risk_level', '')}")
            
            # Show risk assessment
            risk_assessment = removal_plan.get('risk_assessment', {})
            tprint(f"\nRisk Assessment:")
            tprint(f"  Total Risks: {risk_assessment.get('total_risks', 0)}")
            tprint(f"  High Risk: {risk_assessment.get('high_risk_count', 0)}")
            tprint(f"  Medium Risk: {risk_assessment.get('medium_risk_count', 0)}")
            tprint(f"  Recommended Approach: {risk_assessment.get('recommended_approach', 'unknown')}")
            
            # Show recommendations
            recommendations = removal_plan.get('recommendations', [])
            if recommendations:
                tprint(f"\nRecommendations:")
                for rec in recommendations:
                    tprint(f"  • {rec}")
    
    # Generate cleanup recommendations
    tprint(f"\nCLEANUP RECOMMENDATIONS:")
    tprint("-" * 25)
    recommendations = analyzer.generate_cleanup_recommendations(report)
    for rec in recommendations:
        tprint(f"  {rec}")
    
    # Export results
    tprint(f"\nEXPORTING RESULTS:")
    tprint("-" * 18)
    
    # Export to JSON
    json_file = sample_dir / "dead_code_analysis.json"
    json_content = analyzer.export_issues(report, "json")
    json_file.write_text(json_content)
    tprint(f"  JSON report: {json_file}")
    
    # Export to text
    text_file = sample_dir / "dead_code_analysis.txt"
    text_content = analyzer.export_issues(report, "text")
    text_file.write_text(text_content)
    tprint(f"  Text report: {text_file}")
    
    tprint(f"\nDemo complete! Check the files in {sample_dir} for detailed results.")


if __name__ == "__main__":
    demonstrate_enhanced_dead_code_detection()