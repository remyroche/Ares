#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Import Analysis Runner for Ares Repository

This script runs comprehensive import analysis using only the import analyzers
from the code_quality module to identify unused imports and import-related dead code.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json
import logging

# Add the code_quality directory to the Python path
code_quality_path = Path(__file__).parent / "code_quality"
sys.path.insert(0, str(code_quality_path))

try:
    from code_quality.analyzers.enhanced_import_analysis import EnhancedImportAnalyzer, IssueType, IssueSeverity
    from code_quality.analyzers.import_verifier_analyzer import ImportVerifierAnalyzer
    from code_quality.core.config import get_default_config
    tprint("✓ Successfully imported import analyzers")
except ImportError as e:
    tprint(f"❌ Failed to import analyzers: {e}")
    tprint("Please ensure you're running this from the Ares repository root")
    sys.exit(1)

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('imports_analysis.log'),
            logging.StreamHandler()
        ]
    )

def run_enhanced_import_analysis(target_path: Path, output_dir: Path):
    """Run enhanced import analysis."""
    tprint("\n🔍 Running Enhanced Import Analysis...")
    tprint("=" * 60)

    config = {
        'ignore_patterns': ['__pycache__', '.git', 'node_modules', '.venv', 'venv', '.pytest_cache'],
        'max_file_size': 1024 * 1024,  # 1MB
        'encoding': 'utf-8'
    }
    analyzer = EnhancedImportAnalyzer(config)

    try:
        tprint(f"Analyzing directory: {target_path}")
        results = analyzer.analyze_directory(str(target_path))

        # Convert results to serializable format
        serializable_results = []
        for result in results:
            result_dict = {
                'file_path': result.file_path,
                'total_issues': result.total_issues,
                'issues_by_severity': {k.value: v for k, v in result.issues_by_severity.items()},
                'issues_by_type': {k.value: v for k, v in result.issues_by_type.items()},
                'issues': []
            }

            for issue in result.issues:
                issue_dict = {
                    'type': issue.type.value,
                    'severity': issue.severity.value,
                    'name': issue.name,
                    'line': issue.line,
                    'column': issue.column,
                    'message': issue.message,
                    'context': issue.context,
                    'file_path': issue.file_path,
                    'suggestions': issue.suggestions
                }
                result_dict['issues'].append(issue_dict)

            serializable_results.append(result_dict)

        # Save detailed JSON report
        json_file = output_dir / "enhanced_import_analysis.json"
        with open(json_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'analyzer': 'enhanced_import',
                'target_path': str(target_path),
                'total_files': len(results),
                'results': serializable_results
            }, f, indent=2)

        tprint(f"✅ Enhanced import analysis complete. Report saved to: {json_file}")
        return results

    except Exception as e:
        tprint(f"❌ Enhanced import analysis failed: {e}")
        return None

def run_import_verifier_analysis(target_path: Path, output_dir: Path):
    """Run import verifier analysis."""
    tprint("\n🔍 Running Import Verifier Analysis...")
    tprint("=" * 60)

    config = {
        'ignore_patterns': ['__pycache__', '.git', 'node_modules', '.venv', 'venv', '.pytest_cache'],
        'max_file_size': 1024 * 1024,  # 1MB
        'encoding': 'utf-8'
    }
    analyzer = ImportVerifierAnalyzer(config)

    try:
        tprint(f"Analyzing directory: {target_path}")
        results = analyzer.analyze_directory(str(target_path))

        # Save JSON report
        json_file = output_dir / "import_verifier_analysis.json"
        with open(json_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'analyzer': 'import_verifier',
                'target_path': str(target_path),
                'import_status': results.get('import_status', {}),
                'stats': results.get('stats', {})
            }, f, indent=2)

        tprint(f"✅ Import verifier analysis complete. Report saved to: {json_file}")
        return results

    except Exception as e:
        tprint(f"❌ Import verifier analysis failed: {e}")
        return None

def generate_import_summary_report(enhanced_results, verifier_results, output_dir: Path):
    """Generate a summary report focusing on import-related dead code."""
    tprint("\n📊 Generating Import Analysis Summary Report...")
    tprint("=" * 60)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'target_repository': 'Ares',
        'analysis_type': 'import_analysis_only',
        'summary': {}
    }

    # Process enhanced import analysis results
    if enhanced_results:
        unused_imports = []
        duplicate_imports = []
        other_import_issues = []

        for result in enhanced_results:
            for issue in result.issues:
                if issue.type == IssueType.UNUSED_IMPORT:
                    unused_imports.append({
                        'file': issue.file_path,
                        'name': issue.name,
                        'line': issue.line,
                        'message': issue.message
                    })
                elif issue.type == IssueType.DUPLICATE_IMPORT:
                    duplicate_imports.append({
                        'file': issue.file_path,
                        'name': issue.name,
                        'line': issue.line,
                        'message': issue.message
                    })
                else:
                    other_import_issues.append({
                        'file': issue.file_path,
                        'type': issue.type.value,
                        'name': issue.name,
                        'line': issue.line,
                        'message': issue.message
                    })

        summary['summary']['enhanced_analysis'] = {
            'total_files_analyzed': len(enhanced_results),
            'unused_imports': len(unused_imports),
            'duplicate_imports': len(duplicate_imports),
            'other_import_issues': len(other_import_issues),
            'unused_imports_list': unused_imports[:50],  # Limit for readability
            'duplicate_imports_list': duplicate_imports[:20]
        }

    # Process import verifier results
    if verifier_results:
        import_status = verifier_results.get('import_status', {})
        unused_files = []
        used_files = []

        for file_path, status in import_status.items():
            if not status.get('is_imported', False):
                unused_files.append({
                    'file': file_path,
                    'module_name': status.get('module_name', ''),
                    'reason': status.get('reason', 'Not imported by any other file')
                })
            else:
                used_files.append({
                    'file': file_path,
                    'module_name': status.get('module_name', ''),
                    'import_count': status.get('import_count', 0)
                })

        summary['summary']['import_verifier'] = {
            'total_files_analyzed': len(import_status),
            'unused_files': len(unused_files),
            'used_files': len(used_files),
            'unused_files_list': unused_files[:30],  # Limit for readability
        }

    # Save summary
    summary_file = output_dir / "imports_analysis_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Generate text summary
    text_summary = f"""
IMPORT ANALYSIS SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 80}

TARGET REPOSITORY: Ares
ANALYSIS TYPE: Import Analysis Only
"""

    if enhanced_results:
        enhanced = summary['summary']['enhanced_analysis']
        text_summary += f"""

ENHANCED IMPORT ANALYSIS RESULTS:
{'-' * 40}
Total Files Analyzed: {enhanced['total_files_analyzed']}
Unused Imports Found: {enhanced['unused_imports']}
Duplicate Imports Found: {enhanced['duplicate_imports']}
Other Import Issues: {enhanced['other_import_issues']}

Top Unused Imports:
"""
        for i, unused in enumerate(enhanced['unused_imports_list'][:10], 1):
            text_summary += f"{i}. {unused['file']}:{unused['line']} - {unused['name']}\n"

        if enhanced['duplicate_imports_list']:
            text_summary += f"\nDuplicate Imports:\n"
            for i, dup in enumerate(enhanced['duplicate_imports_list'][:5], 1):
                text_summary += f"{i}. {dup['file']}:{dup['line']} - {dup['name']}\n"

    if verifier_results:
        verifier = summary['summary']['import_verifier']
        text_summary += f"""

IMPORT VERIFIER ANALYSIS RESULTS:
{'-' * 40}
Total Files Analyzed: {verifier['total_files_analyzed']}
Unused Files (not imported): {verifier['unused_files']}
Used Files (imported by others): {verifier['used_files']}

Unused Files (potential dead code):
"""
        for i, unused in enumerate(verifier['unused_files_list'][:15], 1):
            text_summary += f"{i}. {unused['file']} ({unused['module_name']})\n"

    text_summary += f"""
{'=' * 80}
RECOMMENDATIONS:
{'=' * 80}

1. Focus on unused imports first - they can be safely removed
2. Review duplicate imports for consolidation opportunities
3. Examine unused files carefully before removal
4. Consider if unused files are entry points, utilities, or future features

For detailed reports, check the JSON files in: {output_dir}
"""

    text_file = output_dir / "imports_analysis_summary.txt"
    with open(text_file, 'w') as f:
        f.write(text_summary)

    tprint(f"✅ Import analysis summary generated: {summary_file}")
    tprint(f"✅ Text summary generated: {text_file}")

    # Print summary to console
    tprint("\n" + text_summary)

def main():
    """Main function to run import analysis."""
    tprint("🔧 IMPORT ANALYSIS FOR ARES REPOSITORY")
    tprint("=" * 80)

    # Setup logging
    setup_logging()

    # Define paths
    repo_root = Path(__file__).parent
    target_path = repo_root / "src"  # Focus on the main source code
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = repo_root / f"imports_analysis_{timestamp}"
    output_dir.mkdir(exist_ok=True)

    tprint(f"Repository root: {repo_root}")
    tprint(f"Target directory: {target_path}")
    tprint(f"Output directory: {output_dir}")
    tprint()

    if not target_path.exists():
        tprint(f"❌ Target directory does not exist: {target_path}")
        sys.exit(1)

    # Run enhanced import analysis
    enhanced_results = run_enhanced_import_analysis(target_path, output_dir)

    # Run import verifier analysis
    verifier_results = run_import_verifier_analysis(target_path, output_dir)

    # Generate summary
    generate_import_summary_report(enhanced_results, verifier_results, output_dir)

    tprint("\n🎉 IMPORT ANALYSIS COMPLETE!")
    tprint(f"📁 All reports saved to: {output_dir}")
    tprint("\nNext steps:")
    tprint("1. Review the summary report for unused imports")
    tprint("2. Examine duplicate imports for consolidation")
    tprint("3. Check unused files for potential dead code")
    tprint("4. Consider the impact before removing any imports")

if __name__ == "__main__":
    main()
