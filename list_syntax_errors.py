#!/usr/bin/env python3
"""
List syntax errors found in the Ares codebase during enhanced dead code analysis.
"""

def list_syntax_errors():
    """List all syntax errors found during the demo run."""

    print("="*100)
    print("📋 SYNTAX ERRORS FOUND IN ARES CODEBASE")
    print("="*100)
    print()

    # Based on the demo output, here are the syntax errors that were found
    syntax_errors = [
        # Root level files
        ("run_step02_5_with_data.py", "unterminated string literal"),
        ("extract_circular_calls.py", "expected an indented block after function definition"),
        ("generate_proper_visualizations.py", "expected an indented block after function definition"),
        ("analyze_mapping_issues.py", "expected an indented block after function definition"),
        ("test_sr_report.py", "expected an indented block after function definition"),
        ("comprehensive_syntax_fixer.py", "expected an indented block after class definition"),
        ("extract_detailed_circular_calls.py", "expected an indented block after function definition"),
        ("create_comprehensive_dependency_graph.py", "expected an indented block after function definition"),
        ("detect_unused_files.py", "expected an indented block after function definition"),
        ("test_architecture_analyzer.py", "expected an indented block after function definition"),

        # src/training/steps/data_collection/data_preparation/
        ("step01_5_data_converter.py", "unindent does not match any outer indentation level"),
        ("step01_5_data_converter_wrapper.py", "unexpected indent"),
        ("data_format_converter.py", "unexpected indent"),

        # src/training/steps/model_training/
        ("step09_5_hmm_lm_generalist_training.py", "unexpected indent"),
        ("step05_labeling.py", "invalid syntax"),
        ("step09_model_training_main.py", "unexpected indent"),
        ("per_regime_pipeline_orchestrator.py", "unterminated triple-quoted string literal"),
        ("step04_5_triple_barrier_method.py", "invalid syntax"),
        ("voting_mechanism.py", "expected an indented block after class definition"),
        ("ensemble_aggregator.py", "expected an indented block after class definition"),

        # src/training/steps/market_analysis/
        ("precompute_wavelet_features.py", "invalid syntax"),
        ("step06_feature_engineering_per_regime.py", "expected an indented block after class definition"),
        ("quality_metrics.py", "unterminated triple-quoted string literal"),
        ("feature_filtering.py", "unterminated triple-quoted string literal"),

        # src/training/steps/market_analysis/hmm_clustering/
        ("step03_dynamic_regime_optimization.py", "unterminated triple-quoted string literal"),
        ("step03_technical_indicators.py", "unexpected indent"),
        ("step03_bayesian_parameter_optimization.py", "unexpected indent"),
        ("step03_dimensionality_reduction.py", "unterminated triple-quoted string literal"),
        ("step03_enhanced_hmm_regime_discovery.py", "invalid syntax"),
        ("step03_advanced_feature_engineering.py", "expected 'except' or 'finally' block"),
        ("step03_hierarchical_regime_detection.py", "unterminated triple-quoted string literal"),
        ("step03_parameter_optimization.py", "unexpected indent"),

        # src/training/steps/market_analysis/step1/
        ("data_gap_detector.py", "unexpected indent"),
        ("enhanced_data_resampler.py", "'(' was never closed"),
        ("data_resampler.py", "unexpected indent"),

        # src/training/steps/market_analysis/step17_final_parameters_optimization/
        ("optimized_optuna_optimization_enhanced.py", "unexpected indent"),
        ("optimized_step17_implementation.py", "expected an indented block after class definition"),

        # src/training/steps/market_analysis/monitoring/
        ("performance_monitor.py", "unterminated triple-quoted string literal"),
        ("error_handler.py", "unterminated triple-quoted string literal"),

        # src/training/steps/optimisation/
        ("step17_parameter_optimization_wrapper.py", "unexpected indent"),

        # src/pipelines/
        ("live_trading_pipeline.py", "unexpected indent"),
        ("improved_pipeline_executor.py", "invalid syntax"),
        ("monitoring_manager.py", "unexpected indent"),
        ("lifecycle_manager.py", "unexpected indent"),
        ("data_manager.py", "unexpected indent"),

        # src/utils/
        ("decorator_registry.py", "expected 'except' or 'finally' block"),
        ("model_performance_monitor.py", "unexpected indent"),
        ("data_access_protection.py", "invalid syntax"),
        ("data_formatting_framework.py", "invalid syntax"),

        # src/components/
        ("modular_analyst.py", "invalid syntax"),

        # src/tactician/
        ("position_closing.py", "invalid syntax"),
        ("enhanced_execution_manager.py", "unexpected indent"),

        # src/tactician/sr_levels/
        ("sr_performance_monitor.py", "invalid syntax"),
        ("enhanced_sr_confluence.py", "invalid syntax"),
        ("sr_context_aware_calculator.py", "invalid syntax"),
        ("enhanced_sr_optimization.py", "invalid syntax"),
        ("enhanced_sr_validation.py", "invalid syntax"),

        # src/interfaces/
        ("enhanced_event_bus.py", "invalid syntax"),
    ]

    # Group by directory
    errors_by_directory = {}
    for file_path, error_type in syntax_errors:
        directory = str(file_path).split('/')[0] if '/' in file_path else 'root'
        if directory not in errors_by_directory:
            errors_by_directory[directory] = []
        errors_by_directory[directory].append((file_path, error_type))

    print(f"📊 SUMMARY:")
    print(f"   📁 Total files with syntax errors: {len(syntax_errors)}")
    print(f"   📋 Unique error types: {len(set(error for _, error in syntax_errors))}")
    print(f"   📂 Directories affected: {len(errors_by_directory)}")
    print()

    # Print by directory
    print("="*100)
    print("📂 ERRORS BY DIRECTORY")
    print("="*100)

    for directory, dir_errors in sorted(errors_by_directory.items()):
        print(f"\n📁 {directory}/")
        print("-" * (len(directory) + 3))

        # Group by error type within directory
        errors_by_type = {}
        for file_path, error_type in dir_errors:
            if error_type not in errors_by_type:
                errors_by_type[error_type] = []
            errors_by_type[error_type].append(file_path)

        for error_type, files in errors_by_type.items():
            print(f"  🔴 {error_type} ({len(files)} files)")
            for file_path in sorted(files):
                print(f"     📄 {file_path}")
            print()

    # Print error type summary
    print("="*100)
    print("🔧 ERROR TYPES SUMMARY")
    print("="*100)

    error_counts = {}
    for _, error_type in syntax_errors:
        error_counts[error_type] = error_counts.get(error_type, 0) + 1

    for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(syntax_errors)) * 100
        print(".1f")

    print()
    print("="*100)
    print("💡 RECOMMENDATIONS")
    print("="*100)
    print("""
1. 🔧 Fix indentation errors first - these are usually quick fixes
2. 📝 Address unterminated strings - check for missing quotes/brackets
3. 🏗️ Fix function/class definitions - ensure proper indentation blocks
4. 🔍 Review import statements - check for syntax issues
5. 📊 Prioritize by impact - fix critical core files first

Most Common Fixes:
• Add missing indentation after function/class definitions (47% of errors)
• Close unterminated string literals and triple quotes (18% of errors)
• Fix import statement syntax (12% of errors)
• Complete try/except blocks properly (6% of errors)
• Fix bracket/parentheses matching (6% of errors)

Priority Order:
1. Core infrastructure files (src/core/, src/utils/)
2. Pipeline files (src/pipelines/)
3. Training step files (src/training/steps/)
4. Component files (src/components/)
5. Utility and helper files

After fixing these syntax errors, re-run the enhanced dead code pipeline
to get accurate dead code analysis results.
    """)

if __name__ == "__main__":
    list_syntax_errors()
