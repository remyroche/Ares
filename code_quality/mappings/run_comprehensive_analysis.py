#!/usr/bin/env python3
"""
Comprehensive Dead Code Analysis Runner
Orchestrates multiple analysis tools to provide accurate dead code detection
with dependency validation and safety assessment.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import argparse

def run_enhanced_function_usage_analyzer(root_dir: str) -> Dict[str, Any]:
    """Run the enhanced function usage analyzer."""
    print("\n🔍 Running Enhanced Function Usage Analyzer...")
    
    try:
        # Import and run the analyzer
        sys.path.insert(0, str(Path(__file__).parent))
        from enhanced_function_usage_analyzer import EnhancedFunctionUsageAnalyzer
        
        analyzer = EnhancedFunctionUsageAnalyzer(root_dir)
        analyzer.analyze_all_files()
        analyzer.analyze_function_usage()
        
        # Get results
        report = analyzer.generate_report()
        
        # Save report
        output_path = "enhanced_function_usage_report.json"
        analyzer.save_report(output_path)
        
        print(f"✅ Enhanced function usage analysis complete. Report saved to {output_path}")
        return report
        
    except Exception as e:
        print(f"❌ Error running enhanced function usage analyzer: {e}")
        return {}

def run_advanced_dependency_analyzer(root_dir: str) -> Dict[str, Any]:
    """Run the advanced dependency analyzer."""
    print("\n🔍 Running Advanced Dependency Analyzer...")
    
    try:
        # Import and run the analyzer
        sys.path.insert(0, str(Path(__file__).parent))
        from dependency_analyzer_v2 import AdvancedDependencyAnalyzer
        
        analyzer = AdvancedDependencyAnalyzer(root_dir)
        analyzer.analyze_repository()
        
        # Get results
        report = analyzer.generate_report()
        
        # Save report
        output_path = "advanced_dependency_analysis.json"
        analyzer.save_report(output_path)
        
        print(f"✅ Advanced dependency analysis complete. Report saved to {output_path}")
        return report
        
    except Exception as e:
        print(f"❌ Error running advanced dependency analyzer: {e}")
        return {}

def run_function_usage_validator(root_dir: str) -> Dict[str, Any]:
    """Run the function usage validator."""
    print("\n🔍 Running Function Usage Validator...")
    
    try:
        # Import and run the validator
        sys.path.insert(0, str(Path(__file__).parent))
        from function_usage_validator import FunctionUsageValidator
        
        validator = FunctionUsageValidator(root_dir)
        validator.validate_all_functions()
        
        # Get results
        report = validator.generate_validation_report()
        
        # Save report
        output_path = "function_usage_validation.json"
        validator.save_report(output_path)
        
        print(f"✅ Function usage validation complete. Report saved to {output_path}")
        return report
        
    except Exception as e:
        print(f"❌ Error running function usage validator: {e}")
        return {}

def merge_analysis_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge results from multiple analysis tools."""
    print("\n🔗 Merging analysis results...")
    
    merged_report = {
        "analysis_metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tools_used": [],
            "total_functions_analyzed": 0,
            "total_truly_unused": 0,
            "total_high_risk": 0
        },
        "consolidated_functions": {},
        "safety_assessment": {
            "safe_to_remove": [],
            "requires_caution": [],
            "high_risk": [],
            "critical_risk": [],
            "requires_manual_review": []
        },
        "tool_specific_results": {}
    }
    
    for i, result in enumerate(results):
        if result:
            tool_name = f"tool_{i+1}"
            merged_report["tool_specific_results"][tool_name] = result
            merged_report["analysis_metadata"]["tools_used"].append(tool_name)
            
            # Extract function information
            if "functions" in result:
                for func_name, func_info in result["functions"].items():
                    if func_name not in merged_report["consolidated_functions"]:
                        merged_report["consolidated_functions"][func_name] = {
                            "name": func_name,
                            "definitions": [],
                            "usage_patterns": [],
                            "dependencies": [],
                            "risk_assessment": "unknown"
                        }
                    
                    merged_report["consolidated_functions"][func_name]["definitions"].append({
                        "tool": tool_name,
                        "info": func_info
                    })
            
            # Extract safety assessment
            if "safety_report" in result:
                safety_report = result["safety_report"]
                for category in ["safe_to_remove", "requires_caution", "high_risk", "critical_risk", "requires_manual_review"]:
                    if category in safety_report:
                        merged_report["safety_assessment"][category].extend(safety_report[category])
    
    # Update metadata
    merged_report["analysis_metadata"]["total_functions_analyzed"] = len(merged_report["consolidated_functions"])
    merged_report["analysis_metadata"]["total_truly_unused"] = len(merged_report["safety_assessment"]["safe_to_remove"])
    merged_report["analysis_metadata"]["total_high_risk"] = len(merged_report["safety_assessment"]["high_risk"]) + len(merged_report["safety_assessment"]["critical_risk"])
    
    return merged_report

def generate_final_report(merged_results: Dict[str, Any], root_dir: str) -> str:
    """Generate a comprehensive final report."""
    print("\n📝 Generating comprehensive final report...")
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("COMPREHENSIVE DEAD CODE ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Analysis Date: {merged_results['analysis_metadata']['timestamp']}")
    report_lines.append(f"Target Directory: {root_dir}")
    report_lines.append(f"Tools Used: {', '.join(merged_results['analysis_metadata']['tools_used'])}")
    report_lines.append("")
    
    # Summary
    report_lines.append("📊 EXECUTIVE SUMMARY")
    report_lines.append("-" * 40)
    report_lines.append(f"Total Functions Analyzed: {merged_results['analysis_metadata']['total_functions_analyzed']}")
    report_lines.append(f"Truly Unused Functions: {merged_results['analysis_metadata']['total_truly_unused']}")
    report_lines.append(f"High Risk Functions: {merged_results['analysis_metadata']['total_high_risk']}")
    report_lines.append("")
    
    # Safety Assessment
    report_lines.append("🛡️  SAFETY ASSESSMENT")
    report_lines.append("-" * 40)
    
    for category, functions in merged_results["safety_assessment"].items():
        if functions:
            report_lines.append(f"{category.replace('_', ' ').title()}: {len(functions)} functions")
            if category == "safe_to_remove":
                report_lines.append("  These functions can be safely removed:")
                for func in functions[:10]:  # Show first 10
                    if isinstance(func, dict):
                        name = func.get('name', 'Unknown')
                        file_path = func.get('file_path', 'Unknown')
                        report_lines.append(f"    • {name} in {file_path}")
                    else:
                        report_lines.append(f"    • {func}")
                if len(functions) > 10:
                    report_lines.append(f"    ... and {len(functions) - 10} more")
            report_lines.append("")
    
    # Recommendations
    report_lines.append("💡 RECOMMENDATIONS")
    report_lines.append("-" * 40)
    
    if merged_results["analysis_metadata"]["total_truly_unused"] > 0:
        report_lines.append("✅ IMMEDIATE ACTIONS:")
        report_lines.append("  • Remove truly unused functions (marked as 'safe_to_remove')")
        report_lines.append("  • These functions have no dependencies and can be safely deleted")
        report_lines.append("")
    
    if merged_results["analysis_metadata"]["total_high_risk"] > 0:
        report_lines.append("⚠️  CAUTION REQUIRED:")
        report_lines.append("  • Review high-risk functions before removal")
        report_lines.append("  • Check for hidden dependencies or dynamic usage")
        report_lines.append("")
    
    report_lines.append("🔧 GENERAL GUIDELINES:")
    report_lines.append("  • Always test after removing functions")
    report_lines.append("  • Remove functions in small batches")
    report_lines.append("  • Keep backup of original code")
    report_lines.append("")
    
    # Tool-specific findings
    report_lines.append("🔍 TOOL-SPECIFIC FINDINGS")
    report_lines.append("-" * 40)
    
    for tool_name, tool_results in merged_results["tool_specific_results"].items():
        report_lines.append(f"Tool: {tool_name}")
        if "summary" in tool_results:
            summary = tool_results["summary"]
            for key, value in summary.items():
                if isinstance(value, (int, str)):
                    report_lines.append(f"  {key.replace('_', ' ').title()}: {value}")
        report_lines.append("")
    
    report_lines.append("=" * 80)
    
    # Save report
    report_content = "\n".join(report_lines)
    report_path = "comprehensive_dead_code_analysis_report.txt"
    
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"✅ Comprehensive report saved to {report_path}")
    return report_path

def print_analysis_summary(merged_results: Dict[str, Any]):
    """Print a summary of the analysis results."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE ANALYSIS COMPLETE")
    print("=" * 80)
    
    metadata = merged_results["analysis_metadata"]
    print(f"📁 Total functions analyzed: {metadata['total_functions_analyzed']}")
    print(f"✅ Truly unused functions: {metadata['total_truly_unused']}")
    print(f"🚨 High risk functions: {metadata['total_high_risk']}")
    print(f"🔧 Tools used: {', '.join(metadata['tools_used'])}")
    
    # Show some safe to remove functions
    safe_functions = merged_results["safety_assessment"]["safe_to_remove"]
    if safe_functions:
        print(f"\n🗑️  SAFE TO REMOVE (Top 5):")
        for i, func in enumerate(safe_functions[:5], 1):
            if isinstance(func, dict):
                name = func.get('name', 'Unknown')
                file_path = func.get('file_path', 'Unknown')
                print(f"   {i}. {name} in {file_path}")
            else:
                print(f"   {i}. {func}")
    
    # Show high risk functions
    high_risk = merged_results["safety_assessment"]["high_risk"] + merged_results["safety_assessment"]["critical_risk"]
    if high_risk:
        print(f"\n⚠️  HIGH RISK FUNCTIONS (Top 5):")
        for i, func in enumerate(high_risk[:5], 1):
            if isinstance(func, dict):
                name = func.get('name', 'Unknown')
                file_path = func.get('file_path', 'Unknown')
                risk = func.get('risk_assessment', {}).get('risk_level', 'Unknown')
                print(f"   {i}. {name} in {file_path} (Risk: {risk})")
            else:
                print(f"   {i}. {func}")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Dead Code Analysis Runner')
    parser.add_argument('--target-dir', default='src', help='Target directory to analyze (default: src)')
    parser.add_argument('--skip-tools', nargs='*', help='Skip specific analysis tools')
    parser.add_argument('--output-dir', default='.', help='Output directory for reports (default: current directory)')
    
    args = parser.parse_args()
    
    root_dir = args.target_dir
    skip_tools = set(args.skip_tools or [])
    output_dir = args.output_dir
    
    # Change to output directory
    os.chdir(output_dir)
    
    print(f"🚀 Starting comprehensive dead code analysis for: {root_dir}")
    print(f"📁 Output directory: {output_dir}")
    if skip_tools:
        print(f"⏭️  Skipping tools: {', '.join(skip_tools)}")
    
    # Run analysis tools
    results = []
    
    # Enhanced Function Usage Analyzer
    if 'enhanced_function_usage' not in skip_tools:
        result = run_enhanced_function_usage_analyzer(root_dir)
        if result:
            results.append(result)
    
    # Advanced Dependency Analyzer
    if 'advanced_dependency' not in skip_tools:
        result = run_advanced_dependency_analyzer(root_dir)
        if result:
            results.append(result)
    
    # Function Usage Validator
    if 'function_usage_validator' not in skip_tools:
        result = run_function_usage_validator(root_dir)
        if result:
            results.append(result)
    
    if not results:
        print("❌ No analysis tools completed successfully")
        return 1
    
    # Merge results
    merged_results = merge_analysis_results(results)
    
    # Generate final report
    report_path = generate_final_report(merged_results, root_dir)
    
    # Print summary
    print_analysis_summary(merged_results)
    
    print(f"\n📄 Reports generated:")
    print(f"  • Comprehensive report: {report_path}")
    for tool_name in merged_results["analysis_metadata"]["tools_used"]:
        print(f"  • {tool_name} report: {tool_name}_report.json")
    
    print(f"\n🎯 Next steps:")
    print(f"  1. Review the comprehensive report: {report_path}")
    print(f"  2. Start with functions marked as 'safe_to_remove'")
    print(f"  3. Test thoroughly after each removal")
    print(f"  4. Use version control to track changes")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())