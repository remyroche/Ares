#!/usr/bin/env python3
"""
Quick Summary - Provides overview of codebase state and cleanup recommendations.
"""

import json
import os
from pathlib import Path

def load_latest_report():
    """Load the most recent analysis report."""
    reports = [
        "unused_code_report.json",
        "focused_usage_report.json", 
        "enhanced_dependency_report.json"
    ]
    
    for report in reports:
        if os.path.exists(report):
            try:
                with open(report, 'r') as f:
                    return json.load(f), report
            except:
                continue
    
    return None, None

def print_quick_summary():
    """Print a quick summary of the current state."""
    data, report_name = load_latest_report()
    
    if not data:
        print("❌ No analysis reports found. Run the analyzers first:")
        print("   python3 unused_code_analyzer.py src/")
        print("   python3 focused_usage_analyzer.py src/")
        return
    
    print(f"📊 QUICK SUMMARY (from {report_name})")
    print("=" * 60)
    
    if "summary" in data:
        summary = data["summary"]
        
        # Print key metrics
        if "total_functions" in summary:
            print(f"🔧 Functions: {summary.get('total_functions', 'N/A')}")
        if "total_classes" in summary:
            print(f"🏗️  Classes: {summary.get('total_classes', 'N/A')}")
        if "unused_functions" in summary:
            print(f"🗑️  Unused functions: {summary.get('unused_functions', 'N/A')}")
        if "unused_classes" in summary:
            print(f"🗑️  Unused classes: {summary.get('unused_classes', 'N/A')}")
        if "files_with_syntax_errors" in summary:
            print(f"⚠️  Files with syntax errors: {summary.get('files_with_syntax_errors', 'N/A')}")
    
    # Print cleanup recommendations
    print(f"\n💡 QUICK CLEANUP ACTIONS:")
    print(f"   1. Fix syntax errors in entry point files")
    print(f"   2. Remove unused optimization classes")
    print(f"   3. Clean up unused training step validators")
    print(f"   4. Remove example files that aren't core functionality")
    
    print(f"\n🚀 TOOLS AVAILABLE:")
    print(f"   • unused_code_analyzer.py - Find all unused code")
    print(f"   • focused_usage_analyzer.py - Focused analysis")
    print(f"   • enhanced_dependency_analyzer.py - Dependency mapping")
    print(f"   • function_call_analyzer.py - Function call analysis")
    
    print(f"\n📁 ORGANIZATION:")
    print(f"   All tools are in the code_quality/ folder")
    print(f"   Run from the code_quality directory")
    print(f"   Use ../src as the target directory")

if __name__ == "__main__":
    print_quick_summary()