#!/usr/bin/env python3
"""
Simple runner for the improved signature analyzer
"""

import sys
import json
from pathlib import Path

# Add the code_quality directory to the path
sys.path.insert(0, str(Path(__file__).parent / 'code_quality'))

def run_signature_analysis():
    """Run the improved signature analyzer on the src directory."""

    # Import the analyzer
    from analyzers.improved_signature_analyzer import ImprovedSignatureAnalyzer

    # Create a simple config
    from core.config import AnalysisConfig, CodeQualityConfig

    analysis_config = AnalysisConfig()
    config = CodeQualityConfig(
        project_root="/Users/remyroche/Documents/Ares",
        analysis_config=analysis_config
    )

    print('🔍 Running improved signature analyzer...')
    print('=' * 60)

    # Fix the config structure issue
    config.analysis = config.analysis_config

    analyzer = ImprovedSignatureAnalyzer(config)

    # Start with a single file first to test
    print('Testing on a single file first...')
    single_results = analyzer.analyze_files(['/Users/remyroche/Documents/Ares/src/training/steps/data_collection/data_preparation/step02_5_sr_optimization.py'])

    if single_results['summary']['total_issues'] > 0:
        print('Found issues in the test file!')
        results = single_results  # Use single file results
    else:
        # If no issues in the test file, run on the full directory
        print('No issues in test file, running full analysis...')
        results = analyzer.analyze_directory('/Users/remyroche/Documents/Ares/src')

    print('✅ Analysis completed!')
    print(f'📊 Total files analyzed: {results["summary"]["total_files_analyzed"]}')
    print(f'📋 Total issues found: {results["summary"]["total_issues"]}')

    if results['summary']['total_issues'] > 0:
        print(f'🔄 Signature changes: {results["summary"]["signature_changes"]}')
        print(f'⚠️  Compatibility issues: {results["summary"]["compatibility_issues"]}')
        print(f'❌ Missing functions: {results["summary"]["missing_functions"]}')
        print(f'⚠️  Unused functions: {results["summary"]["unused_functions"]}')

        # Save detailed results
        output_file = '/Users/remyroche/Documents/Ares/signature_analysis_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f'💾 Detailed results saved to: {output_file}')

        # Show top issues
        print('\n🔥 Top Issues:')
        print('-' * 40)

        # Show compatibility issues (most critical)
        if 'compatibility_issues' in results.get('issues', {}):
            issues = results['issues']['compatibility_issues']
            if issues:
                print('🚨 COMPATIBILITY ISSUES:')
                for i, issue in enumerate(issues[:5]):  # Show first 5
                    print(f'  {i+1}. {issue["file"]}:{issue["line"]} - {issue["message"]}')

        # Show missing functions
        if 'missing_functions' in results.get('issues', {}):
            issues = results['issues']['missing_functions']
            if issues:
                print('\n❌ MISSING FUNCTIONS:')
                for i, issue in enumerate(issues[:5]):  # Show first 5
                    print(f'  {i+1}. {issue["file"]}:{issue["line"]} - {issue["message"]}')

    else:
        print('🎉 No signature issues found!')

    return results

if __name__ == '__main__':
    run_signature_analysis()
