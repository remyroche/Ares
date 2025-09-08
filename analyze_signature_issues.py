#!/usr/bin/env python3
"""
Analyze signature issues and create a systematic fix plan
"""

import json
from collections import defaultdict, Counter

def analyze_signature_issues():
    """Analyze the signature analysis results and create a fix plan."""

    with open('/Users/remyroche/Documents/Ares/signature_analysis_results.json', 'r') as f:
        data = json.load(f)

    print("🔍 SIGNATURE ANALYSIS RESULTS")
    print("=" * 60)
    print(f"📊 Total files analyzed: {data['summary']['total_files_analyzed']}")
    print(f"📋 Total issues: {data['summary']['total_issues']}")
    print(f"🔄 Signature changes: {data['summary']['signature_changes']}")
    print(f"⚠️  Compatibility issues: {data['summary']['compatibility_issues']}")
    print(f"❌ Missing functions: {data['summary']['missing_functions']}")
    print(f"⚠️  Unused functions: {data['summary']['unused_functions']}")
    print()

    # Analyze compatibility issues (most critical)
    if 'compatibility_issues' in data.get('issues', {}):
        print("🚨 COMPATIBILITY ISSUES (CRITICAL - FIX FIRST)")
        print("-" * 50)

        issues = data['issues']['compatibility_issues']

        # Group by error type
        error_patterns = Counter()
        for issue in issues:
            message = issue.get('message', '')
            if 'Missing required arguments:' in message:
                error_patterns['Missing required arguments'] += 1
            elif 'Too many positional arguments:' in message:
                error_patterns['Too many positional arguments'] += 1
            elif 'Unknown keyword argument:' in message:
                error_patterns['Unknown keyword argument'] += 1
            else:
                error_patterns['Other'] += 1

        print("Error pattern breakdown:")
        for pattern, count in error_patterns.most_common():
            print(f"  • {pattern}: {count} issues")

        # Show top 10 most critical issues
        print("\nTop 10 Critical Issues:")
        for i, issue in enumerate(issues[:10]):
            print(f"  {i+1}. Line {issue['line']}: {issue['message']}")
            if 'details' in issue and 'call' in issue['details']:
                call_info = issue['details']['call']
                if 'args' in call_info and call_info['args']:
                    print(f"      Call args: {call_info['args']}")
                if 'keywords' in call_info and call_info['keywords']:
                    print(f"      Call kwargs: {call_info['keywords']}")

    # Analyze missing functions
    if 'missing_functions' in data.get('issues', {}):
        print("\n❌ MISSING FUNCTIONS")
        print("-" * 30)

        missing = data['issues']['missing_functions']
        for issue in missing:
            print(f"  • Line {issue['line']}: {issue['message']}")

    # Analyze signature changes
    if 'signature_changes' in data.get('issues', {}):
        print("\n🔄 SIGNATURE CHANGES")
        print("-" * 25)

        changes = data['issues']['signature_changes']
        for issue in changes:
            func_name = issue.get('details', {}).get('function_name', 'unknown')
            print(f"  • Line {issue['line']}: {func_name} - {issue['message']}")
            if 'details' in issue and 'differences' in issue['details']:
                for diff in issue['details']['differences'][:2]:  # Show first 2 differences
                    print(f"      {diff}")

    print("\n" + "=" * 60)
    print("🎯 FIX PRIORITIES:")
    print("1. 🔴 Critical: Fix compatibility issues (missing arguments)")
    print("2. 🟡 High: Fix missing functions")
    print("3. 🟢 Medium: Address signature changes")
    print("4. 🔵 Low: Review unused functions")
    print("=" * 60)

if __name__ == '__main__':
    analyze_signature_issues()
