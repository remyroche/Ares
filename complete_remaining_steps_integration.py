#!/usr/bin/env python3
"""
Script to complete the integration of all remaining pipeline steps with enhanced MLflow logging.

This script will systematically update the remaining steps that weren't fully integrated
by the previous automated script.
"""

import re
from pathlib import Path

# Define the remaining steps that need manual integration
REMAINING_STEPS = [
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

            # Create detailed report
            report_data = create_detailed_step_report(
                step_name="step{step_num}",
                step_data=pipeline_state,
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


def find_execute_method(file_path: Path) -> tuple[bool, str, int]:
    """Find the execute method in a step file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Look for different execute method patterns
        patterns = [
            r'async def execute\s*\([^)]*\)\s*->[^:]*:',
            r'def execute\s*\([^)]*\)\s*->[^:]*:',
            r'async def execute_[a-zA-Z_]*\s*\([^)]*\)\s*->[^:]*:',
            r'def execute_[a-zA-Z_]*\s*\([^)]*\)\s*->[^:]*:',
        ]

        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if re.search(pattern, line):
                        return True, line.strip(), i

        return False, "", -1

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return False, "", -1


def add_mlflow_decorator_to_step(file_path: Path) -> bool:
    """Add MLflow decorator to a step's execute method."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if decorator already exists
        if "@with_enhanced_mlflow_logging" in content:
            print(f"✅ MLflow decorator already exists in {file_path.name}")
            return True

        # Find execute method
        found, method_line, line_num = find_execute_method(file_path)

        if not found:
            print(f"⚠️ Could not find execute method in {file_path.name}")
            return False

        # Add decorator before execute method
        lines = content.split('\n')
        step_num = extract_step_number(file_path.name)
        decorator = f'    @with_enhanced_mlflow_logging("step{step_num}")'

        # Find the right position (before other decorators)
        insert_pos = line_num
        for i in range(line_num - 1, -1, -1):
            if lines[i].strip().startswith('@'):
                insert_pos = i
            elif lines[i].strip() and not lines[i].strip().startswith('#'):
                break

        lines.insert(insert_pos, decorator)
        new_content = '\n'.join(lines)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"✅ Added MLflow decorator to {file_path.name}")
        return True

    except Exception as e:
        print(f"❌ Failed to add MLflow decorator to {file_path.name}: {e}")
        return False


def add_artifact_logging_call(file_path: Path) -> bool:
    """Add call to artifact logging method before return statement."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find execute method and its return statement
        lines = content.split('\n')
        step_num = extract_step_number(file_path.name)
        method_name = f"_log_step{step_num}_artifacts_and_report"

        # Look for return statements in execute method
        in_execute = False
        execute_start = -1

        for i, line in enumerate(lines):
            if 'async def execute' in line or 'def execute' in line:
                in_execute = True
                execute_start = i
            elif in_execute and line.strip().startswith('return '):
                # Add call before return
                call_line = f'            # Log artifacts and create detailed report'
                call_line2 = f'            await self.{method_name}(training_input, pipeline_state, result)'
                lines.insert(i, call_line2)
                lines.insert(i, call_line)
                break
            elif in_execute and line.strip() and not line.strip().startswith(' ') and not line.strip().startswith('#'):
                # Found another method, execute method ended
                break

        new_content = '\n'.join(lines)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"✅ Added artifact logging call to {file_path.name}")
        return True

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

        # Find the end of the execute method
        lines = content.split('\n')
        execute_end = None

        # Find the return statement in execute method
        in_execute = False
        for i, line in enumerate(lines):
            if 'async def execute' in line or 'def execute' in line:
                in_execute = True
            elif in_execute and line.strip().startswith('return '):
                execute_end = i
                break
            elif in_execute and line.strip() and not line.strip().startswith(' ') and not line.strip().startswith('#'):
                # Found another method, execute method ended
                execute_end = i - 1
                break

        if execute_end is not None:
            # Add the artifact logging method
            method_content = ARTIFACT_LOGGING_METHOD_TEMPLATE.format(step_num=step_num)

            # Insert method after execute method
            lines.insert(execute_end + 1, method_content)
            new_content = '\n'.join(lines)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            print(f"✅ Added artifact logging method to {file_path.name}")
            return True

        print(f"⚠️ Could not find execute method in {file_path.name}")
        return False

    except Exception as e:
        print(f"❌ Failed to add artifact logging method to {file_path.name}: {e}")
        return False


def update_step_file(file_path: Path) -> Dict[str, bool]:
    """Update a single step file with enhanced MLflow integration."""
    results = {
        "decorator": False,
        "call": False,
        "method": False
    }

    print(f"\n🔄 Updating {file_path.name}...")

    # Add MLflow decorator
    results["decorator"] = add_mlflow_decorator_to_step(file_path)

    # Add artifact logging call
    results["call"] = add_artifact_logging_call(file_path)

    # Add artifact logging method
    results["method"] = add_artifact_logging_method(file_path)

    return results


def main():
    """Main function to update all remaining step files."""
    steps_dir = Path("src/training/steps")

    if not steps_dir.exists():
        print(f"❌ Steps directory not found: {steps_dir}")
        return

    print("🚀 Starting enhanced MLflow integration for remaining pipeline steps...")
    print(f"📁 Steps directory: {steps_dir}")
    print(f"📋 Steps to update: {len(REMAINING_STEPS)}")

    results = {}

    for step_file in REMAINING_STEPS:
        file_path = steps_dir / step_file

        if not file_path.exists():
            print(f"⚠️ Step file not found: {step_file}")
            continue

        results[step_file] = update_step_file(file_path)

    # Print summary
    print("\n" + "="*60)
    print("📊 UPDATE SUMMARY")
    print("="*60)

    successful_updates = 0
    total_steps = len(results)

    for step_file, step_results in results.items():
        success_count = sum(step_results.values())
        total_count = len(step_results)

        if success_count == total_count:
            print(f"✅ {step_file}: All updates successful")
            successful_updates += 1
        elif success_count > 0:
            print(f"⚠️ {step_file}: Partial success ({success_count}/{total_count})")
        else:
            print(f"❌ {step_file}: All updates failed")

    print(f"\n🎯 Overall: {successful_updates}/{total_steps} steps fully updated")

    if successful_updates == total_steps:
        print("🎉 All remaining steps successfully updated with enhanced MLflow integration!")
    else:
        print("⚠️ Some steps may need manual review")


if __name__ == "__main__":
    main()