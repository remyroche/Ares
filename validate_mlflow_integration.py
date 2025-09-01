#!/usr/bin/env python3
"""
Validation script to check MLflow integration completeness and metadata validation.

This script validates:
1. Enhanced MLflow imports in all steps
2. Decorator presence in execute methods
3. Artifact logging method presence
4. Metadata completeness
5. Standardized naming patterns
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json

# Define all pipeline steps
ALL_STEPS = [
    "step1_data_collection.py",
    "step2_data_reading.py",
    "step2_5_sr_optimization.py",
    "step3_hmm_regime_discovery.py",
    "step4_triple_barrier_method.py",
    "step5_labeling.py",
    "step6_feature_engineering.py",
    "step7_enhanced_matrix_operations.py",
    "step8_regime_data_splitting.py",
    "step9_hmm_based_training.py",
    "step9_5_hmm_lm_generalist_training.py",
    "step9_5_multi_timeframe_hmm_ensemble.py",
    "step10_unified_regime_intelligence.py",
    "step11_analyst_creation.py",
    "step12_analyst_enhancement.py",
    "step13_analyst_ensemble_creation.py",
    "step14_tactician_labeling.py",
    "step15_tactician_specialist_training.py",
    "step16_confidence_calibration.py",
    "step17_final_parameters_optimization.py",
    "step18_walk_forward_validation.py",
    "step19_monte_carlo_validation.py",
    "step20_ab_testing.py",
    "step21_saving.py",
]

# Required metadata fields
REQUIRED_METADATA_FIELDS = [
    "asset",
    "exchange",
    "lookback_period",
    "project_version",
    "date"
]

# Required MLflow integration components
REQUIRED_COMPONENTS = [
    "enhanced_mlflow_integration",
    "with_enhanced_mlflow_logging",
    "log_step_report",
    "create_detailed_step_report",
    "log_step_metrics",
    "log_step_dataframe_with_standardized_name",
    "log_step_artifact_with_standardized_name"
]


def extract_step_number(filename: str) -> str:
    """Extract step number from filename."""
    match = re.search(r'step(\d+(?:_\d+)?)', filename)
    if match:
        return match.group(1)
    return "unknown"


def find_execute_methods(file_path: Path) -> List[Tuple[str, int]]:
    """Find all execute methods in a step file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        methods = []
        lines = content.split('\n')

        # Look for different execute method patterns
        patterns = [
            r'async def execute\s*\([^)]*\)\s*->[^:]*:',
            r'def execute\s*\([^)]*\)\s*->[^:]*:',
            r'async def execute_[a-zA-Z_]*\s*\([^)]*\)\s*->[^:]*:',
            r'def execute_[a-zA-Z_]*\s*\([^)]*\)\s*->[^:]*:',
            r'async def run_step\s*\([^)]*\)\s*->[^:]*:',
            r'def run_step\s*\([^)]*\)\s*->[^:]*:',
        ]

        for i, line in enumerate(lines):
            for pattern in patterns:
                if re.search(pattern, line):
                    methods.append((line.strip(), i))

        return methods

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []


def validate_mlflow_imports(file_path: Path) -> Dict[str, bool]:
    """Validate MLflow imports in a step file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        results = {}

        # Check for enhanced MLflow integration imports
        results["enhanced_mlflow_integration"] = "from src.utils.enhanced_mlflow_integration import" in content

        # Check for specific imports
        for component in REQUIRED_COMPONENTS[1:]:  # Skip the first one as it's the module name
            results[component] = component in content

        return results

    except Exception as e:
        print(f"Error validating imports in {file_path}: {e}")
        return {component: False for component in REQUIRED_COMPONENTS}


def validate_mlflow_decorator(file_path: Path) -> Dict[str, bool]:
    """Validate MLflow decorator presence in execute methods."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        results = {}

        # Check if decorator exists anywhere in the file
        results["decorator_present"] = "@with_enhanced_mlflow_logging" in content

        # Find execute methods
        execute_methods = find_execute_methods(file_path)
        results["execute_methods_found"] = len(execute_methods) > 0

        # Check if decorator is applied to execute methods
        if execute_methods:
            lines = content.split('\n')
            decorated_methods = 0

            for method_line, line_num in execute_methods:
                # Check if decorator is present before this method
                decorator_found = False
                for i in range(line_num - 1, max(0, line_num - 10), -1):
                    if lines[i].strip().startswith('@with_enhanced_mlflow_logging'):
                        decorator_found = True
                        break
                    elif lines[i].strip() and not lines[i].strip().startswith('@'):
                        break

                if decorator_found:
                    decorated_methods += 1

            results["methods_decorated"] = decorated_methods
            results["all_methods_decorated"] = decorated_methods == len(execute_methods)
        else:
            results["methods_decorated"] = 0
            results["all_methods_decorated"] = False

        return results

    except Exception as e:
        print(f"Error validating decorator in {file_path}: {e}")
        return {
            "decorator_present": False,
            "execute_methods_found": False,
            "methods_decorated": 0,
            "all_methods_decorated": False
        }


def validate_artifact_logging_method(file_path: Path) -> Dict[str, bool]:
    """Validate artifact logging method presence."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        step_num = extract_step_number(file_path.name)
        method_name = f"_log_step{step_num}_artifacts_and_report"

        results = {}
        results["method_present"] = method_name in content

        # Check for specific components in the method
        if results["method_present"]:
            results["create_detailed_step_report_call"] = "create_detailed_step_report" in content
            results["log_step_report_call"] = "log_step_report" in content
            results["log_step_metrics_call"] = "log_step_metrics" in content
        else:
            results["create_detailed_step_report_call"] = False
            results["log_step_report_call"] = False
            results["log_step_metrics_call"] = False

        return results

    except Exception as e:
        print(f"Error validating artifact logging method in {file_path}: {e}")
        return {
            "method_present": False,
            "create_detailed_step_report_call": False,
            "log_step_report_call": False,
            "log_step_metrics_call": False
        }


def validate_metadata_completeness(file_path: Path) -> Dict[str, bool]:
    """Validate metadata completeness in artifact logging methods."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        step_num = extract_step_number(file_path.name)
        method_name = f"_log_step{step_num}_artifacts_and_report"

        results = {}

        # Check if method exists
        if method_name not in content:
            return {field: False for field in REQUIRED_METADATA_FIELDS}

        # Extract the method content - look for the entire method
        method_start = content.find(method_name)
        if method_start == -1:
            return {field: False for field in REQUIRED_METADATA_FIELDS}

        # Find method end by looking for the next method that's at the same indentation level
        lines = content.split('\n')
        method_start_line = content[:method_start].count('\n')

        method_end = len(content)
        for i in range(method_start_line + 1, len(lines)):
            line = lines[i]
            if line.strip().startswith('def ') and line.strip() != method_name:
                # Found next method, calculate end position
                method_end = content.find(line, method_start)
                break

        method_content = content[method_start:method_end]

        # Check for metadata fields with more flexible patterns
        results["asset"] = '"asset"' in method_content or 'asset' in method_content
        results["exchange"] = '"exchange"' in method_content or 'exchange' in method_content
        results["lookback_period"] = '"lookback_period"' in method_content or 'lookback_period' in method_content
        results["project_version"] = '"project_version"' in method_content or 'project_version' in method_content
        results["date"] = '"date"' in method_content or 'datetime.now()' in method_content

        return results

    except Exception as e:
        print(f"Error validating metadata in {file_path}: {e}")
        return {field: False for field in REQUIRED_METADATA_FIELDS}


def validate_standardized_naming(file_path: Path) -> Dict[str, bool]:
    """Validate standardized naming patterns."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        results = {}

        # Check for standardized naming functions
        results["log_step_dataframe_with_standardized_name"] = "log_step_dataframe_with_standardized_name" in content
        results["log_step_artifact_with_standardized_name"] = "log_step_artifact_with_standardized_name" in content

        # Check for standardized naming patterns in strings and comments
        results["standardized_naming_pattern"] = (
            re.search(r'[A-Z]+_[A-Z]+_\d{8}_\d{4}_\d+', content) is not None or
            'Standardized naming pattern' in content or
            'exchange_token_date_hourminute_NumberOfStep_Artifact' in content
        )

        return results

    except Exception as e:
        print(f"Error validating standardized naming in {file_path}: {e}")
        return {
            "log_step_dataframe_with_standardized_name": False,
            "log_step_artifact_with_standardized_name": False,
            "standardized_naming_pattern": False
        }


def validate_step_file(file_path: Path) -> Dict[str, Any]:
    """Validate a single step file."""
    print(f"\n🔍 Validating {file_path.name}...")

    results = {
        "file_exists": file_path.exists(),
        "step_number": extract_step_number(file_path.name),
        "imports": {},
        "decorator": {},
        "artifact_logging": {},
        "metadata": {},
        "standardized_naming": {},
    }

    if not results["file_exists"]:
        print(f"❌ File not found: {file_path.name}")
        return results

    # Validate imports
    results["imports"] = validate_mlflow_imports(file_path)

    # Validate decorator
    results["decorator"] = validate_mlflow_decorator(file_path)

    # Validate artifact logging method
    results["artifact_logging"] = validate_artifact_logging_method(file_path)

    # Validate metadata completeness
    results["metadata"] = validate_metadata_completeness(file_path)

    # Validate standardized naming
    results["standardized_naming"] = validate_standardized_naming(file_path)

    return results


def calculate_integration_score(results: Dict[str, Any]) -> float:
    """Calculate integration completeness score (0-100)."""
    if not results["file_exists"]:
        return 0.0

    score = 0.0
    total_checks = 0

    # Imports (20 points)
    import_checks = len(results["imports"])
    import_score = sum(results["imports"].values()) / import_checks if import_checks > 0 else 0
    score += import_score * 20
    total_checks += 20

    # Decorator (20 points)
    decorator_checks = len(results["decorator"])
    decorator_score = sum(results["decorator"].values()) / decorator_checks if decorator_checks > 0 else 0
    score += decorator_score * 20
    total_checks += 20

    # Artifact logging (30 points)
    artifact_checks = len(results["artifact_logging"])
    artifact_score = sum(results["artifact_logging"].values()) / artifact_checks if artifact_checks > 0 else 0
    score += artifact_score * 30
    total_checks += 30

    # Metadata (20 points)
    metadata_checks = len(results["metadata"])
    metadata_score = sum(results["metadata"].values()) / metadata_checks if metadata_checks > 0 else 0
    score += metadata_score * 20
    total_checks += 20

    # Standardized naming (10 points)
    naming_checks = len(results["standardized_naming"])
    naming_score = sum(results["standardized_naming"].values()) / naming_checks if naming_checks > 0 else 0
    score += naming_score * 10
    total_checks += 10

    return (score / total_checks) * 100 if total_checks > 0 else 0.0


def generate_validation_report(all_results: Dict[str, Dict[str, Any]]) -> str:
    """Generate a comprehensive validation report."""
    report = []
    report.append("# MLflow Integration Validation Report")
    report.append("")
    report.append(f"Generated: {__import__('datetime').datetime.now().isoformat()}")
    report.append("")

    # Summary statistics
    total_steps = len(all_results)
    completed_steps = sum(1 for results in all_results.values() if calculate_integration_score(results) >= 90)
    partial_steps = sum(1 for results in all_results.values() if 50 <= calculate_integration_score(results) < 90)
    incomplete_steps = sum(1 for results in all_results.values() if calculate_integration_score(results) < 50)

    report.append("## Summary")
    report.append("")
    report.append(f"- **Total Steps**: {total_steps}")
    report.append(f"- **Fully Integrated** (90-100%): {completed_steps}")
    report.append(f"- **Partially Integrated** (50-89%): {partial_steps}")
    report.append(f"- **Incomplete** (<50%): {incomplete_steps}")
    report.append(f"- **Overall Completion**: {(completed_steps + partial_steps) / total_steps * 100:.1f}%")
    report.append("")

    # Detailed results
    report.append("## Detailed Results")
    report.append("")

    for step_file, results in all_results.items():
        score = calculate_integration_score(results)
        status = "✅ Complete" if score >= 90 else "⚠️ Partial" if score >= 50 else "❌ Incomplete"

        report.append(f"### {step_file} - {status} ({score:.1f}%)")
        report.append("")

        if not results["file_exists"]:
            report.append("- ❌ File not found")
            report.append("")
            continue

        # Imports
        import_score = sum(results["imports"].values()) / len(results["imports"]) * 100
        report.append(f"- **Imports**: {import_score:.1f}%")
        for import_name, present in results["imports"].items():
            status_icon = "✅" if present else "❌"
            report.append(f"  - {status_icon} {import_name}")

        # Decorator
        decorator_score = sum(results["decorator"].values()) / len(results["decorator"]) * 100
        report.append(f"- **Decorator**: {decorator_score:.1f}%")
        for decorator_name, present in results["decorator"].items():
            status_icon = "✅" if present else "❌"
            report.append(f"  - {status_icon} {decorator_name}")

        # Artifact logging
        artifact_score = sum(results["artifact_logging"].values()) / len(results["artifact_logging"]) * 100
        report.append(f"- **Artifact Logging**: {artifact_score:.1f}%")
        for artifact_name, present in results["artifact_logging"].items():
            status_icon = "✅" if present else "❌"
            report.append(f"  - {status_icon} {artifact_name}")

        # Metadata
        metadata_score = sum(results["metadata"].values()) / len(results["metadata"]) * 100
        report.append(f"- **Metadata**: {metadata_score:.1f}%")
        for metadata_name, present in results["metadata"].items():
            status_icon = "✅" if present else "❌"
            report.append(f"  - {status_icon} {metadata_name}")

        # Standardized naming
        naming_score = sum(results["standardized_naming"].values()) / len(results["standardized_naming"]) * 100
        report.append(f"- **Standardized Naming**: {naming_score:.1f}%")
        for naming_name, present in results["standardized_naming"].items():
            status_icon = "✅" if present else "❌"
            report.append(f"  - {status_icon} {naming_name}")

        report.append("")

    return "\n".join(report)


def main():
    """Main validation function."""
    steps_dir = Path("src/training/steps")

    if not steps_dir.exists():
        print(f"❌ Steps directory not found: {steps_dir}")
        return

    print("🔍 Starting MLflow integration validation...")
    print(f"📁 Steps directory: {steps_dir}")
    print(f"📋 Steps to validate: {len(ALL_STEPS)}")

    all_results = {}

    for step_file in ALL_STEPS:
        file_path = steps_dir / step_file
        all_results[step_file] = validate_step_file(file_path)

    # Calculate overall statistics
    total_steps = len(all_results)
    completed_steps = sum(1 for results in all_results.values() if calculate_integration_score(results) >= 90)
    partial_steps = sum(1 for results in all_results.values() if 50 <= calculate_integration_score(results) < 90)
    incomplete_steps = sum(1 for results in all_results.values() if calculate_integration_score(results) < 50)

    # Print summary
    print("\n" + "="*60)
    print("📊 VALIDATION SUMMARY")
    print("="*60)
    print(f"Total Steps: {total_steps}")
    print(f"Fully Integrated (90-100%): {completed_steps}")
    print(f"Partially Integrated (50-89%): {partial_steps}")
    print(f"Incomplete (<50%): {incomplete_steps}")
    print(f"Overall Completion: {(completed_steps + partial_steps) / total_steps * 100:.1f}%")

    # Print detailed results
    print("\n" + "="*60)
    print("📋 DETAILED RESULTS")
    print("="*60)

    for step_file, results in all_results.items():
        score = calculate_integration_score(results)
        status = "✅ Complete" if score >= 90 else "⚠️ Partial" if score >= 50 else "❌ Incomplete"
        print(f"{status} {step_file} ({score:.1f}%)")

    # Generate and save detailed report
    report = generate_validation_report(all_results)

    with open("mlflow_integration_validation_report.md", "w") as f:
        f.write(report)

    print(f"\n📄 Detailed report saved to: mlflow_integration_validation_report.md")

    # Save JSON results
    with open("mlflow_integration_validation_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"📊 JSON results saved to: mlflow_integration_validation_results.json")


if __name__ == "__main__":
    main()