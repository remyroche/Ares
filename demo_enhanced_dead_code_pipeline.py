#!/usr/bin/env python3
"""
Enhanced Dead Code Pipeline Demo

This script demonstrates how to use the enhanced Dead Code Pipeline
that integrates with Interaction Mapping Pipeline outputs to provide
more accurate dead code detection with reduced false positives.

Usage:
    python demo_enhanced_dead_code_pipeline.py [--disable-interaction-mapping]

Features demonstrated:
- Integration with interaction mapping for cross-file usage analysis
- False positive reduction using call graph analysis
- Enhanced reporting with confidence levels
- Entry point analysis for better accuracy
"""

import sys
import time
from pathlib import Path

# Add the code_quality directory to the path
sys.path.insert(0, str(Path(__file__).parent / "code_quality"))
sys.path.insert(0, str(Path(__file__).parent / "code_quality" / "pipelines"))

from pipelines.dead_code_pipeline import DeadCodePipeline


def demo_enhanced_vs_standard():
    """Compare enhanced dead code analysis with and without interaction mapping."""

    print("="*80)
    print("ENHANCED DEAD CODE PIPELINE DEMO")
    print("="*80)
    print(f"Project root: {Path.cwd()}")
    print()

    # Demo 1: Standard dead code analysis (no interaction mapping)
    print("🔍 DEMO 1: Standard Dead Code Analysis (Static Only)")
    print("-" * 60)

    pipeline_standard = DeadCodePipeline(
        project_root=str(Path.cwd()),
        enable_plugins=False,
        use_interaction_mapping=False  # Disable interaction mapping
    )

    start_time = time.time()
    results_standard = pipeline_standard.run_enhanced_dead_code_analysis()
    standard_time = time.time() - start_time

    print("Standard analysis completed:")
    print(f"  ⏱️  Execution time: {standard_time:.2f} seconds")
    print(f"  📊 Total issues found: {results_standard.get('total_issues', 0)}")
    print(f"  🎯 High confidence issues: {results_standard.get('high_confidence_issues', 0)}")
    print(f"  🔗 Interaction enhanced: {results_standard.get('interaction_enhanced', False)}")
    print()

    # Demo 2: Enhanced dead code analysis (with interaction mapping)
    print("🚀 DEMO 2: Enhanced Dead Code Analysis (With Interaction Mapping)")
    print("-" * 60)

    pipeline_enhanced = DeadCodePipeline(
        project_root=str(Path.cwd()),
        enable_plugins=False,
        use_interaction_mapping=True  # Enable interaction mapping
    )

    start_time = time.time()
    results_enhanced = pipeline_enhanced.run_enhanced_dead_code_analysis()
    enhanced_time = time.time() - start_time

    print("Enhanced analysis completed:")
    print(f"  ⏱️  Execution time: {enhanced_time:.2f} seconds")
    print(f"  📊 Total issues found: {results_enhanced.get('total_issues', 0)}")
    print(f"  🎯 High confidence issues: {results_enhanced.get('high_confidence_issues', 0)}")
    print(f"  🔗 Interaction enhanced: {results_enhanced.get('interaction_enhanced', False)}")
    print(f"  ✅ False positives removed: {results_enhanced.get('false_positives_removed', 0)}")
    print()

    # Comparison
    print("📈 COMPARISON RESULTS")
    print("-" * 60)

    time_improvement = ((standard_time - enhanced_time) / standard_time * 100) if standard_time > 0 else 0
    accuracy_improvement = results_enhanced.get('false_positives_removed', 0)

    print("Performance comparison:")
    print(f"  ⏱️  Time improvement: {time_improvement:.2f}%")
    print(f"  🎯 Accuracy improvement: {accuracy_improvement} false positives removed")

    if accuracy_improvement > 0:
        print("\n🎉 SUCCESS: Enhanced analysis removed false positives!")
        print("   The interaction mapping integration successfully identified")
        print("   functions/classes that were incorrectly flagged as unused.")
    else:
        print("\nℹ️  INFO: No false positives were found to remove.")
        print("   This could mean the codebase is already well-analyzed or")
        print("   there are genuine unused functions/classes.")

    print()
    print("="*80)
    print("DEMO COMPLETED")
    print("="*80)

    return results_standard, results_enhanced


def show_enhanced_features():
    """Show the key features of the enhanced dead code pipeline."""

    print("\n🔧 ENHANCED DEAD CODE PIPELINE FEATURES")
    print("-" * 60)

    features = [
        ("🔗 Cross-File Usage Analysis", "Uses interaction mapping to verify function/class usage across the entire codebase"),
        ("📞 Call Graph Integration", "Analyzes call graphs to identify entry points and reachable functions"),
        ("🎯 Entry Point Detection", "Automatically identifies main entry points and their call chains"),
        ("❌ False Positive Reduction", "Validates potentially unused code against interaction data"),
        ("📊 Enhanced Confidence Scoring", "Provides confidence levels for dead code findings"),
        ("📈 Comprehensive Reporting", "Generates detailed reports combining dead code and interaction analysis"),
        ("⚡ Smart Filtering", "Filters out special functions (main, __init__, etc.) and base classes"),
        ("🔍 Transitive Usage Detection", "Identifies functions used indirectly through call chains")
    ]

    for feature, description in features:
        print(f"  {feature}")
        print(f"    {description}")
        print()

    print("💡 USAGE TIP:")
    print("  To disable interaction mapping and use only static analysis:")
    print("  python demo_enhanced_dead_code_pipeline.py --disable-interaction-mapping")


def main():
    """Main demo function."""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Dead Code Pipeline Demo")
    parser.add_argument(
        "--disable-interaction-mapping",
        action="store_true",
        help="Disable interaction mapping enhancement for comparison"
    )

    args = parser.parse_args()

    try:
        if args.disable_interaction_mapping:
            print("🔧 Running standard dead code analysis (interaction mapping disabled)")
            pipeline = DeadCodePipeline(
                project_root=str(Path.cwd()),
                use_interaction_mapping=False
            )
            results = pipeline.run_enhanced_dead_code_analysis()
            print("Standard analysis completed successfully!")
        else:
            # Run the full comparison demo
            show_enhanced_features()
            demo_enhanced_vs_standard()

    except Exception as e:
        print(f"❌ Error running demo: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
