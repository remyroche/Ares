#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Dead Code Analysis Runner for Ares Repository

This script runs comprehensive dead code analysis on the entire repository
using the enhanced dead code analyzers from the code_quality module.
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
    from analyzers.enhanced_dead_code_analyzer import EnhancedDeadCodeAnalyzer
    from analyzers.truly_enhanced_dead_code_analyzer import TrulyEnhancedDeadCodeAnalyzer
    from core.config import get_default_config
    tprint("✓ Successfully imported dead code analyzers")
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
            logging.FileHandler('dead_code_analysis.log'),
            logging.StreamHandler()
        ]
    )

def run_enhanced_analysis(target_path: Path, output_dir: Path):
    """Run enhanced dead code analysis."""
    tprint("\n🔍 Running Enhanced Dead Code Analysis...")
    tprint("=" * 60)

    config = get_default_config()
    analyzer = EnhancedDeadCodeAnalyzer(config)

    try:
        tprint(f"Analyzing directory: {target_path}")
        report = analyzer.analyze_directory(str(target_path))

        # Save JSON report
        json_file = output_dir / "enhanced_dead_code_analysis.json"
        with open(json_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'analyzer': 'enhanced',
                'target_path': str(target_path),
                'total_issues': report.total_issues,
                'issues_by_type': report.issues_by_type,
                'issues_by_severity': report.issues_by_severity,
                'high_confidence_issues': len([i for i in report.issues_by_severity.get('high', [])]),
                'medium_confidence_issues': len([i for i in report.issues_by_severity.get('medium', [])]),
                'low_confidence_issues': len([i for i in report.issues_by_severity.get('low', [])]),
            }, f, indent=2, default=str)

        tprint(f"✅ Enhanced analysis complete. Report saved to: {json_file}")
        return report

    except Exception as e:
        tprint(f"❌ Enhanced analysis failed: {e}")
        return None

def run_truly_enhanced_analysis(target_path: Path, output_dir: Path):
    """Run truly enhanced dead code analysis."""
    tprint("\n🔬 Running Truly Enhanced Dead Code Analysis...")
    tprint("=" * 60)

    config = get_default_config()
    analyzer = TrulyEnhancedDeadCodeAnalyzer(config)

    try:
        tprint(f"Analyzing directory: {target_path}")
        report = analyzer.analyze_directory(str(target_path))

        # Save JSON report
        json_file = output_dir / "truly_enhanced_dead_code_analysis.json"
        with open(json_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'analyzer': 'truly_enhanced',
                'target_path': str(target_path),
                'total_issues': report.total_issues,
                'high_confidence_issues': report.high_confidence_issues,
                'medium_confidence_issues': report.medium_confidence_issues,
                'low_confidence_issues': report.low_confidence_issues,
                'issues_by_type': report.issues_by_type,
                'false_positives_filtered': report.false_positives_filtered,
                'consensus_issues': report.consensus_issues,
            }, f, indent=2, default=str)

        tprint(f"✅ Truly enhanced analysis complete. Report saved to: {json_file}")
        return report

    except Exception as e:
        tprint(f"❌ Truly enhanced analysis failed: {e}")
        return None

def generate_summary_report(enhanced_report, truly_enhanced_report, output_dir: Path):
    """Generate a summary report combining both analyses."""
    tprint("\n📊 Generating Summary Report...")
    tprint("=" * 60)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'target_repository': 'Ares',
        'analysis_summary': {}
    }

    if enhanced_report:
        summary['analysis_summary']['enhanced'] = {
            'total_issues': enhanced_report.total_issues,
            'issues_by_type': enhanced_report.issues_by_type,
            'issues_by_severity': enhanced_report.issues_by_severity,
        }

    if truly_enhanced_report:
        summary['analysis_summary']['truly_enhanced'] = {
            'total_issues': truly_enhanced_report.total_issues,
            'high_confidence_issues': truly_enhanced_report.high_confidence_issues,
            'medium_confidence_issues': truly_enhanced_report.medium_confidence_issues,
            'low_confidence_issues': truly_enhanced_report.low_confidence_issues,
            'false_positives_filtered': truly_enhanced_report.false_positives_filtered,
            'consensus_issues': truly_enhanced_report.consensus_issues,
        }

    # Save summary
    summary_file = output_dir / "dead_code_analysis_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Generate text summary
    text_summary = f"""
DEAD CODE ANALYSIS SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 80}

TARGET REPOSITORY: Ares
ANALYSIS DATE: {datetime.now().isoformat()}

{'ENHANCED ANALYZER RESULTS:' if enhanced_report else 'ENHANCED ANALYZER: NOT RUN'}
{'-' * 40}
"""

    if enhanced_report:
        text_summary += f"""
Total Issues: {enhanced_report.total_issues}
Issues by Type: {enhanced_report.issues_by_type}
Issues by Severity: {enhanced_report.issues_by_severity}
"""

    analyzer_status = 'TRULY ENHANCED ANALYZER RESULTS:' if truly_enhanced_report else 'TRULY ENHANCED ANALYZER: NOT RUN'
    text_summary += f"\n{analyzer_status}\n{'-' * 40}\n"

    if truly_enhanced_report:
        text_summary += f"""
Total Issues: {truly_enhanced_report.total_issues}
High Confidence Issues: {truly_enhanced_report.high_confidence_issues}
Medium Confidence Issues: {truly_enhanced_report.medium_confidence_issues}
Low Confidence Issues: {truly_enhanced_report.low_confidence_issues}
False Positives Filtered: {truly_enhanced_report.false_positives_filtered}
Consensus Issues: {truly_enhanced_report.consensus_issues}
"""

    text_summary += f"""
{'=' * 80}
RECOMMENDATIONS:
{'=' * 80}

1. Focus on high-confidence issues first for safe removal
2. Review medium-confidence issues manually before removal
3. Be cautious with low-confidence issues (may be false positives)
4. Consider the impact of removing code on the overall system
5. Test thoroughly after any code removal

For detailed reports, check the JSON files in: {output_dir}
"""

    text_file = output_dir / "dead_code_analysis_summary.txt"
    with open(text_file, 'w') as f:
        f.write(text_summary)

    tprint(f"✅ Summary report generated: {summary_file}")
    tprint(f"✅ Text summary generated: {text_file}")

    # Print summary to console
    tprint("\n" + text_summary)

def main():
    """Main function to run dead code analysis."""
    tprint("🚀 DEAD CODE ANALYSIS FOR ARES REPOSITORY")
    tprint("=" * 80)

    # Setup logging
    setup_logging()

    # Define paths
    repo_root = Path(__file__).parent
    target_path = repo_root / "src"  # Focus on the main source code
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = repo_root / f"dead_code_analysis_{timestamp}"
    output_dir.mkdir(exist_ok=True)

    tprint(f"Repository root: {repo_root}")
    tprint(f"Target directory: {target_path}")
    tprint(f"Output directory: {output_dir}")
    tprint()

    if not target_path.exists():
        tprint(f"❌ Target directory does not exist: {target_path}")
        sys.exit(1)

    # Run enhanced analysis
    enhanced_report = run_enhanced_analysis(target_path, output_dir)

    # Run truly enhanced analysis
    truly_enhanced_report = run_truly_enhanced_analysis(target_path, output_dir)

    # Generate summary
    generate_summary_report(enhanced_report, truly_enhanced_report, output_dir)

    tprint("\n🎉 DEAD CODE ANALYSIS COMPLETE!")
    tprint(f"📁 All reports saved to: {output_dir}")
    tprint("\nNext steps:")
    tprint("1. Review the summary report for an overview")
    tprint("2. Examine detailed JSON reports for specific issues")
    tprint("3. Consider the confidence levels before removing code")
    tprint("4. Test thoroughly after any code removal")

if __name__ == "__main__":
    main()
