#!/usr/bin/env python3
"""
Syntax Error Count Analysis Script
Counts the specific number of syntax errors in each listed file.
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Tuple

class SyntaxErrorCounter:
    def __init__(self):
        self.error_counts = {}
        self.error_details = {}
        
    def analyze_file(self, file_path: str) -> Tuple[int, List[Dict]]:
        """Analyze a single file and count syntax errors."""
        if not os.path.exists(file_path):
            return 0, [{"error": "File not found"}]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to parse with AST
            ast.parse(content)
            return 0, []  # No syntax errors
            
        except SyntaxError as e:
            # Count this as 1 syntax error
            return 1, [{
                "line": e.lineno,
                "column": e.offset,
                "message": str(e.msg),
                "type": "syntax_error"
            }]
        except Exception as e:
            # Count this as 1 error
            return 1, [{
                "line": 0,
                "column": 0,
                "message": str(e),
                "type": "other_error"
            }]
    
    def analyze_files(self, file_list: List[str]) -> Dict[str, Dict]:
        """Analyze multiple files and return error counts."""
        results = {}
        
        for file_path in file_list:
            print(f"🔍 Analyzing: {file_path}")
            error_count, error_details = self.analyze_file(file_path)
            
            results[file_path] = {
                "error_count": error_count,
                "error_details": error_details,
                "status": "ERROR" if error_count > 0 else "OK"
            }
            
            if error_count > 0:
                print(f"  ❌ {error_count} syntax error(s)")
                for error in error_details:
                    if "line" in error and error["line"] > 0:
                        print(f"    Line {error['line']}: {error['message']}")
            else:
                print(f"  ✅ No syntax errors")
        
        return results
    
    def generate_summary(self, results: Dict[str, Dict]) -> str:
        """Generate a summary report."""
        total_files = len(results)
        files_with_errors = sum(1 for r in results.values() if r["error_count"] > 0)
        total_errors = sum(r["error_count"] for r in results.values())
        
        summary_lines = [
            "# Syntax Error Count Analysis - Core Source Files",
            "",
            f"**Total Files Analyzed**: {total_files}",
            f"**Files with Errors**: {files_with_errors}",
            f"**Total Syntax Errors**: {total_errors}",
            "",
            "## Detailed Results",
            ""
        ]
        
        # Group by category
        categories = {
            "Core Components": [],
            "Training System": [],
            "Training Steps": [],
            "Utility Modules": []
        }
        
        for file_path, result in results.items():
            if "supervisor" in file_path or "tactician" in file_path:
                categories["Core Components"].append((file_path, result))
            elif "training" in file_path and "steps" not in file_path:
                categories["Training System"].append((file_path, result))
            elif "training/steps" in file_path:
                categories["Training Steps"].append((file_path, result))
            elif "utils" in file_path:
                categories["Utility Modules"].append((file_path, result))
        
        for category, files in categories.items():
            if files:
                summary_lines.append(f"### {category}")
                summary_lines.append("")
                
                for file_path, result in files:
                    status_icon = "❌" if result["error_count"] > 0 else "✅"
                    summary_lines.append(f"{status_icon} **{file_path}**: {result['error_count']} error(s)")
                    
                    if result["error_count"] > 0 and result["error_details"]:
                        for error in result["error_details"]:
                            if "line" in error and error["line"] > 0:
                                summary_lines.append(f"  - Line {error['line']}: {error['message']}")
                            else:
                                summary_lines.append(f"  - {error['message']}")
                
                summary_lines.append("")
        
        # Summary statistics
        summary_lines.append("## Summary Statistics")
        summary_lines.append("")
        summary_lines.append(f"- **Core Components**: {len(categories['Core Components'])} files")
        summary_lines.append(f"- **Training System**: {len(categories['Training System'])} files")
        summary_lines.append(f"- **Training Steps**: {len(categories['Training Steps'])} files")
        summary_lines.append(f"- **Utility Modules**: {len(categories['Utility Modules'])} files")
        summary_lines.append("")
        summary_lines.append(f"**Overall Error Rate**: {(files_with_errors/total_files)*100:.1f}%")
        
        return "\n".join(summary_lines)

def main():
    """Main function to analyze the specified files."""
    
    # List of files to analyze
    files_to_analyze = [
        # Core Components
        "src/supervisor/global_portfolio_manager.py",
        "src/tactician/sr_weight_optimizer.py",
        "src/tactician/sr_breakout_predictor.py",
        
        # Training System
        "src/training/model_trainer.py",
        "src/training/enhanced_training_manager.py",
        "src/training/step_orchestrator.py",
        
        # Training Steps
        "src/training/steps/step9_5_hmm_lm_generalist_training.py",
        "src/training/steps/step2_5_sr_optimization.py",
        "src/training/steps/step10_unified_regime_intelligence.py",
        "src/training/steps/step14_tactician_labeling.py",
        "src/training/steps/step4_triple_barrier_method.py",
        "src/training/steps/step5_labeling.py",
        "src/training/steps/step2_data_reading.py",
        "src/training/steps/step16_confidence_calibration.py",
        "src/training/steps/step21_saving.py",
        "src/training/steps/step3_hmm_regime_discovery.py",
        "src/training/steps/vectorized_advanced_feature_engineering.py",
        "src/training/steps/step6_feature_engineering_validator.py",
        "src/training/steps/step19_monte_carlo_validation.py",
        "src/training/steps/step9_5_multi_timeframe_hmm_ensemble.py",
        "src/training/steps/step17_final_parameters_optimization.py",
        "src/training/steps/step12_analyst_enhancement.py",
        "src/training/steps/step18_walk_forward_validation.py",
        "src/training/steps/step15_tactician_specialist_training.py",
        "src/training/steps/step7_enhanced_matrix_operations.py",
        "src/training/steps/step17_final_parameters_optimization/sr_optuna_optimization.py",
        
        # Utility Modules
        "src/utils/observability.py",
        "src/utils/step_dependency_validator.py",
        "src/utils/enhanced_validation_decorators.py",
        "src/utils/model_performance_monitor.py",
        "src/utils/enhanced_config_management.py",
        "src/utils/enhanced_memory_management.py",
        "src/utils/centralized_decorators_v2.py",
        "src/utils/validator_orchestrator.py",
        "src/utils/enhanced_data_quality_validator.py",
        "src/utils/enhanced_error_handling.py",
        "src/utils/prometheus_metrics.py"
    ]
    
    print("🔍 Starting Syntax Error Count Analysis...")
    print("=" * 60)
    
    counter = SyntaxErrorCounter()
    results = counter.analyze_files(files_to_analyze)
    
    print("\n" + "=" * 60)
    print("📊 ANALYSIS COMPLETE")
    print("=" * 60)
    
    # Generate and save detailed report
    summary = counter.generate_summary(results)
    
    with open("syntax_error_count_report.md", "w") as f:
        f.write(summary)
    
    print("📄 Detailed report saved to: syntax_error_count_report.md")
    
    # Print summary
    total_files = len(results)
    files_with_errors = sum(1 for r in results.values() if r["error_count"] > 0)
    total_errors = sum(r["error_count"] for r in results.values())
    
    print(f"\n📈 SUMMARY:")
    print(f"Total Files: {total_files}")
    print(f"Files with Errors: {files_with_errors}")
    print(f"Total Syntax Errors: {total_errors}")
    print(f"Error Rate: {(files_with_errors/total_files)*100:.1f}%")

if __name__ == "__main__":
    main()