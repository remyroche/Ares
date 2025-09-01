#!/usr/bin/env python3
"""
Validation script for SR Breakout Predictor Reporting Implementation
Checks for comprehensive reporting capabilities and integration.
"""

import ast
import re

def check_file_syntax(file_path: str) -> bool:
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            ast.parse(f.read())
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return False

def check_class_methods(file_path: str, class_name: str, required_methods: list[str]) -> dict[str, bool]:
    """Check if a class has the required methods."""
    results = {}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Use regex to find method definitions (more reliable than AST for async methods)
        for method in required_methods:
            pattern = rf"async def {method}|def {method}"
            results[method] = bool(re.search(pattern, content, re.MULTILINE))

    except Exception as e:
        print(f"❌ Error checking methods in {file_path}: {e}")
        for method in required_methods:
            results[method] = False

    return results

def check_string_patterns(file_path: str, patterns: dict[str, str]) -> dict[str, bool]:
    """Check for specific string patterns in a file."""
    results = {}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        for pattern_name, pattern in patterns.items():
            results[pattern_name] = bool(re.search(pattern, content, re.MULTILINE))

    except Exception as e:
        print(f"❌ Error checking patterns in {file_path}: {e}")
        for pattern_name in patterns.keys():
            results[pattern_name] = False

    return results

def validate_sr_reporting_implementation():
    """Validate the SR reporting implementation."""
    print("🔍 Validating SR Breakout Predictor Reporting Implementation...")
    print("=" * 70)

    file_path = "src/tactician/sr_breakout_predictor.py"

    # Check file syntax
    print("\n1. Checking file syntax...")
    if not check_file_syntax(file_path):
        print("❌ FAILED: File has syntax errors")
        return False
    print("✅ File syntax is valid")

    # Check reporting configuration
    print("\n2. Checking reporting configuration...")
    config_patterns = {
        "reporting_enabled": r"self\.reporting_enabled\s*:",
        "report_directory": r"self\.report_directory\s*:",
        "report_format": r"self\.report_format\s*:",
        "report_retention_days": r"self\.report_retention_days\s*:",
        "metrics_history": r"self\.metrics_history\s*:",
        "current_report_id": r"self\.current_report_id\s*:"
    }

    config_results = check_string_patterns(file_path, config_patterns)
    for pattern_name, found in config_results.items():
        status = "✅" if found else "❌"
        print(f"   {status} {pattern_name}")

    # Check reporting methods
    print("\n3. Checking reporting methods...")
    reporting_methods = [
        "_initialize_reporting_system",
        "_generate_report_id",
        "_calculate_comprehensive_metrics",
        "_calculate_data_quality_score",
        "_calculate_sr_confidence_score",
        "_calculate_overall_quality_score",
        "_generate_detailed_report",
        "_save_report_to_file",
        "_save_metrics_to_csv",
        "_save_html_report",
        "get_latest_report",
        "get_report_history",
        "cleanup_old_reports",
        "generate_manual_report",
        "get_reporting_status"
    ]

    method_results = check_class_methods(file_path, "SRBreakoutPredictor", reporting_methods)
    for method, found in method_results.items():
        status = "✅" if found else "❌"
        print(f"   {status} {method}")

    # Check integration with main methods
    print("\n4. Checking integration with main methods...")
    integration_patterns = {
        "initialize_reporting_in_init": r"# Initialize reporting system",
        "reporting_in_get_sr_context": r"# Generate detailed report",
        "reporting_in_predict_sr_breakouts": r"# Generate detailed report for predictions",
        "report_id_in_context": r"\"report_id\":\s*await self\._generate_detailed_report"
    }

    integration_results = check_string_patterns(file_path, integration_patterns)
    for pattern_name, found in integration_results.items():
        status = "✅" if found else "❌"
        print(f"   {status} {pattern_name}")

    # Check metrics calculation
    print("\n5. Checking metrics calculation...")
    metrics_patterns = {
        "market_metrics": r"market_metrics\s*=\s*\{",
        "sr_metrics": r"sr_metrics\s*=\s*\{",
        "clustering_metrics": r"clustering_metrics\s*=\s*\{",
        "advanced_metrics": r"advanced_metrics\s*=\s*\{",
        "performance_metrics": r"performance_metrics\s*=\s*\{",
        "data_quality_score": r"data_quality_score",
        "sr_confidence_score": r"sr_confidence_score",
        "overall_analysis_quality": r"overall_analysis_quality"
    }

    metrics_results = check_string_patterns(file_path, metrics_patterns)
    for pattern_name, found in metrics_results.items():
        status = "✅" if found else "❌"
        print(f"   {status} {pattern_name}")

    # Check file output formats
    print("\n6. Checking file output formats...")
    output_patterns = {
        "json_output": r"json\.dump\(report,\s*f,\s*indent=2",
        "csv_output": r"csv\.writer\(f\)",
        "html_output": r"html_content\s*=\s*f\"\"\"",
        "latest_metrics": r"latest_metrics\.json"
    }

    output_results = check_string_patterns(file_path, output_patterns)
    for pattern_name, found in output_results.items():
        status = "✅" if found else "❌"
        print(f"   {status} {pattern_name}")

    # Check error handling
    print("\n7. Checking error handling...")
    error_patterns = {
        "try_except_blocks": r"try:",
        "logging_errors": r"self\.logger\.error\(",
        "graceful_fallbacks": r"except Exception as e:"
    }

    error_results = check_string_patterns(file_path, error_patterns)
    for pattern_name, found in error_results.items():
        status = "✅" if found else "❌"
        print(f"   {status} {pattern_name}")

    # Summary
    print("\n" + "=" * 70)
    print("📊 VALIDATION SUMMARY")
    print("=" * 70)

    all_results = {
        "Configuration": config_results,
        "Methods": method_results,
        "Integration": integration_results,
        "Metrics": metrics_results,
        "Output Formats": output_results,
        "Error Handling": error_results
    }

    total_checks = 0
    passed_checks = 0

    for category, results in all_results.items():
        category_total = len(results)
        category_passed = sum(results.values())
        total_checks += category_total
        passed_checks += category_passed

        print(f"\n{category}:")
        for item, passed in results.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {item}")
        print(f"   {category_passed}/{category_total} passed")

    print(f"\n🎯 OVERALL: {passed_checks}/{total_checks} checks passed")

    if passed_checks == total_checks:
        print("🎉 SUCCESS: All reporting features are properly implemented!")
        return True
    else:
        print("⚠️  WARNING: Some reporting features may be missing or incomplete")
        return False

if __name__ == "__main__":
    success = validate_sr_reporting_implementation()
    exit(0 if success else 1)