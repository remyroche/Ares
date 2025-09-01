#!/usr/bin/env python3
"""
Comprehensive script to complete the integration of the remaining 16 steps.
"""

import re
from pathlib import Path

# Steps that need full integration
STEPS_TO_INTEGRATE = [
    "step3_hmm_regime_discovery.py",
    "step6_feature_engineering.py",
    "step9_hmm_based_training.py",
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

# Template for artifact logging method
ARTIFACT_LOGGING_METHOD_TEMPLATE = '''
    async def _log_step{step_num}_artifacts_and_report(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
        step_results: dict[str, Any] = None
    ) -> None:
        """Log step {step_num} artifacts and create detailed report."""
        try:
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            timeframe = training_input.get("timeframe", "1m")

            # Collect execution metadata
            execution_metadata = {{
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": 0.0,  # Will be calculated if available
                "memory_usage_mb": 0.0,  # Will be calculated if available
                "cpu_usage_percent": 0.0,  # Will be calculated if available
                "data_quality_score": 1.0 if pipeline_state.get("step{step_num}_completed", False) else 0.5,
                "processing_efficiency": 1.0 if pipeline_state.get("step{step_num}_completed", False) else 0.0,
            }}

            # Collect artifacts generated
            artifacts_generated = []
            if pipeline_state.get("step{step_num}_completed", False):
                # Add expected artifacts for step {step_num}
                artifacts_generated.extend([
                    f"{{exchange}}_{{symbol}}_{{timeframe}}_step{step_num}_results.parquet",
                    f"{{exchange}}_{{symbol}}_{{timeframe}}_step{step_num}_metrics.json",
                ])

            # Collect metrics
            metrics_calculated = {{
                "step{step_num}_success": 1.0 if pipeline_state.get("step{step_num}_completed", False) else 0.0,
                "total_artifacts_generated": len(artifacts_generated),
            }}

            # Add step-specific metrics if available
            if step_results:
                for key, value in step_results.items():
                    if isinstance(value, (int, float)):
                        metrics_calculated[f"step{step_num}_{key}"] = float(value)

            # Create training input for report
            training_input = {{
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "asset": symbol,  # Use symbol as asset
                "lookback_period": training_input.get("lookback_days", 1095),
                "project_version": self.config.get("project_version", "1.0.0"),
            }}

            # Create step data for report
            step_data = {{
                "step_results": step_results,
                "pipeline_state": pipeline_state,
            }}

            # Create detailed report
            report_data = create_detailed_step_report(
                step_name="step{step_num}",
                step_data=step_data,
                training_input=training_input,
                execution_metadata=execution_metadata,
                artifacts_generated=artifacts_generated,
                metrics_calculated=metrics_calculated,
                errors_encountered=[] if pipeline_state.get("step{step_num}_completed", False) else ["Step {step_num} failed"]
            )

            # Log the report
            report_name = log_step_report(
                config=self.config,
                step_name="step{step_num}",
                report_data=report_data,
                report_type="step{step_num}_report",
                additional_metadata={{
                    "step{step_num}_success": pipeline_state.get("step{step_num}_completed", False),
                    "timeframe": timeframe,
                    "asset": symbol,
                    "lookback_period": training_input.get("lookback_days", 1095),
                    "project_version": self.config.get("project_version", "1.0.0"),
                }}
            )
            self.logger.info(f"✅ Logged step {step_num} report: {{report_name}}")

            # Log metrics
            log_step_metrics(
                config=self.config,
                step_name="step{step_num}",
                metrics=metrics_calculated,
                additional_metadata={{
                    "metrics_type": "step{step_num}_performance",
                    "timeframe": timeframe,
                    "asset": symbol,
                    "lookback_period": training_input.get("lookback_days", 1095),
                    "project_version": self.config.get("project_version", "1.0.0"),
                }}
            )

            # Log step-specific summary if available
            if step_results:
                summary_report_name = log_step_report(
                    config=self.config,
                    step_name="step{step_num}",
                    report_data=step_results,
                    report_type="step{step_num}_summary",
                    additional_metadata={{
                        "step{step_num}_success": True,
                        "timeframe": timeframe,
                        "asset": symbol,
                        "lookback_period": training_input.get("lookback_days", 1095),
                        "project_version": self.config.get("project_version", "1.0.0"),
                    }}
                )
                self.logger.info(f"✅ Logged step {step_num} summary: {{summary_report_name}}")

            self.logger.info("✅ Step {step_num} artifacts and reports logged successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to log step {step_num} artifacts and reports: {{e}}")
            # Don't fail the step if MLflow logging fails
'''


def extract_step_number(filename: str) -> str:
    """Extract step number from filename."""
    match = re.search(r'step(\d+(?:_\d+)?)', filename)
    if match:
        return match.group(1)
    return "unknown"


def find_execute_methods(file_path: Path) -> List[tuple[str, int]]:
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


def add_mlflow_imports(file_path: Path) -> bool:
    """Add MLflow imports to a step file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if imports already exist
        if "from src.utils.enhanced_mlflow_integration import" in content:
            print(f"✅ MLflow imports already exist in {file_path.name}")
            return True

        # Find the right place to add imports (after existing imports)
        lines = content.split('\n')
        import_end = 0

        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                import_end = i + 1
            elif line.strip() and not line.strip().startswith('#'):
                break

        # Add MLflow imports
        mlflow_imports = [
            "",
            "from src.utils.enhanced_mlflow_integration import (",
            "    with_enhanced_mlflow_logging,",
            "    log_step_report,",
            "    create_detailed_step_report,",
            "    log_step_metrics,",
            "    log_step_dataframe_with_standardized_name,",
            "    log_step_artifact_with_standardized_name",
            ")",
            ""
        ]

        lines[import_end:import_end] = mlflow_imports
        new_content = '\n'.join(lines)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"✅ Added MLflow imports to {file_path.name}")
        return True

    except Exception as e:
        print(f"❌ Failed to add MLflow imports to {file_path.name}: {e}")
        return False


def add_mlflow_decorator(file_path: Path) -> bool:
    """Add MLflow decorator to execute methods."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if decorator already exists
        if "@with_enhanced_mlflow_logging" in content:
            print(f"✅ MLflow decorator already exists in {file_path.name}")
            return True

        # Find execute methods
        execute_methods = find_execute_methods(file_path)

        if not execute_methods:
            print(f"⚠️ Could not find execute methods in {file_path.name}")
            return False

        # Extract step number
        step_num = extract_step_number(file_path.name)
        decorator = f'    @with_enhanced_mlflow_logging("step{step_num}")'

        # Add decorator to each execute method
        lines = content.split('\n')
        changes_made = False

        for method_line, line_num in execute_methods:
            # Check if decorator is already present before this method
            decorator_present = False
            for i in range(line_num - 1, max(0, line_num - 10), -1):
                if lines[i].strip().startswith('@with_enhanced_mlflow_logging'):
                    decorator_present = True
                    break
                elif lines[i].strip() and not lines[i].strip().startswith('@'):
                    break

            if not decorator_present:
                # Find the right position (before other decorators)
                insert_pos = line_num
                for i in range(line_num - 1, -1, -1):
                    if lines[i].strip().startswith('@'):
                        insert_pos = i
                    elif lines[i].strip() and not lines[i].strip().startswith('#'):
                        break

                lines.insert(insert_pos, decorator)
                changes_made = True

        if changes_made:
            new_content = '\n'.join(lines)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            print(f"✅ Added MLflow decorator to {file_path.name}")
            return True
        else:
            print(f"✅ Decorator already applied to all methods in {file_path.name}")
            return True

    except Exception as e:
        print(f"❌ Failed to add MLflow decorator to {file_path.name}: {e}")
        return False


def add_artifact_logging_call(file_path: Path) -> bool:
    """Add call to artifact logging method before return statement."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find execute methods and their return statements
        lines = content.split('\n')
        step_num = extract_step_number(file_path.name)
        method_name = f"_log_step{step_num}_artifacts_and_report"

        # Look for return statements in execute methods
        changes_made = False

        for i, line in enumerate(lines):
            if 'async def execute' in line or 'def execute' in line or 'async def run_step' in line or 'def run_step' in line:
                # Found execute method, look for return statement
                in_execute = True
                for j in range(i + 1, len(lines)):
                    if lines[j].strip().startswith('return '):
                        # Add call before return
                        call_line = f'            # Log artifacts and create detailed report'
                        call_line2 = f'            await self.{method_name}(training_input, pipeline_state, result)'
                        lines.insert(j, call_line2)
                        lines.insert(j, call_line)
                        changes_made = True
                        break
                    elif lines[j].strip() and not lines[j].strip().startswith(' ') and not lines[j].strip().startswith('#'):
                        # Found another method, execute method ended
                        break

        if changes_made:
            new_content = '\n'.join(lines)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            print(f"✅ Added artifact logging call to {file_path.name}")
            return True
        else:
            print(f"⚠️ Could not find return statements in execute methods in {file_path.name}")
            return False

    except Exception as e:
        print(f"❌ Failed to add artifact logging call to {file_path.name}: {e}")
        return False


def add_artifact_logging_method(file_path: Path) -> bool:
    """Add artifact logging method to step file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if method already exists
        step_num = extract_step_number(file_path.name)
        method_name = f"_log_step{step_num}_artifacts_and_report"

        if method_name in content:
            print(f"✅ Artifact logging method already exists in {file_path.name}")
            return True

        # Find the end of the file to add the method
        lines = content.split('\n')

        # Add the artifact logging method at the end
        method_content = ARTIFACT_LOGGING_METHOD_TEMPLATE.format(step_num=step_num)

        lines.append(method_content)
        new_content = '\n'.join(lines)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"✅ Added artifact logging method to {file_path.name}")
        return True

    except Exception as e:
        print(f"❌ Failed to add artifact logging method to {file_path.name}: {e}")
        return False


def integrate_step_file(file_path: Path) -> Dict[str, bool]:
    """Integrate a single step file with enhanced MLflow logging."""
    results = {
        "imports": False,
        "decorator": False,
        "call": False,
        "method": False
    }

    print(f"\n🔄 Integrating {file_path.name}...")

    # Add MLflow imports
    results["imports"] = add_mlflow_imports(file_path)

    # Add MLflow decorator
    results["decorator"] = add_mlflow_decorator(file_path)

    # Add artifact logging call
    results["call"] = add_artifact_logging_call(file_path)

    # Add artifact logging method
    results["method"] = add_artifact_logging_method(file_path)

    return results


def main():
    """Main function to integrate all remaining step files."""
    steps_dir = Path("src/training/steps")

    if not steps_dir.exists():
        print(f"❌ Steps directory not found: {steps_dir}")
        return

    print("🚀 Starting comprehensive integration of remaining pipeline steps...")
    print(f"📁 Steps directory: {steps_dir}")
    print(f"📋 Steps to integrate: {len(STEPS_TO_INTEGRATE)}")

    results = {}

    for step_file in STEPS_TO_INTEGRATE:
        file_path = steps_dir / step_file

        if not file_path.exists():
            print(f"⚠️ Step file not found: {step_file}")
            continue

        results[step_file] = integrate_step_file(file_path)

    # Print summary
    print("\n" + "="*60)
    print("📊 INTEGRATION SUMMARY")
    print("="*60)

    successful_integrations = 0
    total_steps = len(results)

    for step_file, step_results in results.items():
        success_count = sum(step_results.values())
        total_count = len(step_results)

        if success_count == total_count:
            print(f"✅ {step_file}: All integrations successful")
            successful_integrations += 1
        elif success_count > 0:
            print(f"⚠️ {step_file}: Partial success ({success_count}/{total_count})")
        else:
            print(f"❌ {step_file}: All integrations failed")

    print(f"\n🎯 Overall: {successful_integrations}/{total_steps} steps fully integrated")

    if successful_integrations == total_steps:
        print("🎉 All remaining steps successfully integrated with enhanced MLflow logging!")
    else:
        print("⚠️ Some steps may need manual review")


if __name__ == "__main__":
    main()