#!/usr/bin/env python3
"""
Script to run the sequential fixer across the entire codebase
and generate a comprehensive report.
"""
import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add code_quality to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code_quality'))

from code_quality.fixers.sequential_fixer import SequentialFixer
from code_quality.core.config import get_default_config

def main():
    """Run sequential fixer and generate report."""
    # Set up output directory for reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"sequential_fixer_reports_{timestamp}"
    
    print(f"Starting Sequential Fixer Run at {datetime.now()}")
    print(f"Reports will be saved to: {output_dir}")
    
    # Get default configuration
    config = get_default_config()
    
    # Create sequential fixer instance
    fixer = SequentialFixer(config)
    
    # Run on the src directory (main codebase)
    target_dir = "src"
    
    try:
        # Run the pipeline with all steps
        results = fixer.run_pipeline(
            target=target_dir,
            output_dir=output_dir,
            create_backups=True,  # Create backups before fixing
            run_pre_commit=False  # Skip pre-commit for now
        )
        
        # Save results to a separate summary file
        summary_file = f"sequential_fixer_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {summary_file}")
        
        # Generate text report
        report_file = f"sequential_fixer_report_{timestamp}.txt"
        generate_text_report(results, report_file)
        print(f"Text report saved to: {report_file}")
        
        return results
        
    except Exception as e:
        print(f"Error running sequential fixer: {e}")
        raise

def generate_text_report(results, filename):
    """Generate a human-readable text report."""
    with open(filename, 'w') as f:
        f.write("SEQUENTIAL FIXER COMPREHENSIVE REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        # Pipeline info
        info = results.get("pipeline_info", {})
        f.write(f"Target: {info.get('target', 'Unknown')}\n")
        f.write(f"Total Files: {info.get('total_files', 0)}\n")
        f.write(f"Timestamp: {info.get('timestamp', 'Unknown')}\n")
        f.write(f"Duration: {info.get('duration', 0):.2f} seconds\n\n")
        
        # Summary
        summary = results.get("summary", {})
        f.write("OVERALL STATUS: " + summary.get("overall_status", "Unknown").upper() + "\n\n")
        
        # Step statuses
        f.write("Step Statuses:\n")
        for step, status in summary.get("step_statuses", {}).items():
            f.write(f"  - {step.replace('_', ' ').title()}: {status.upper()}\n")
        f.write("\n")
        
        # Metrics
        metrics = summary.get("metrics", {})
        f.write("Key Metrics:\n")
        f.write(f"  - Files Processed: {metrics.get('files_processed', 0)}\n")
        f.write(f"  - Syntax Errors: {metrics.get('syntax_errors', 0)}\n")
        f.write(f"  - Valid Files: {metrics.get('valid_files', 0)}\n")
        f.write(f"  - Invalid Files: {metrics.get('invalid_files', 0)}\n")
        f.write(f"  - Linter Issues: {metrics.get('linter_issues', 0)}\n")
        f.write(f"  - Import Issues: {metrics.get('import_issues', 0)}\n")
        f.write(f"  - Signature Issues: {metrics.get('signature_issues', 0)}\n")
        f.write("\n")
        
        # Recommendations
        recommendations = summary.get("recommendations", [])
        if recommendations:
            f.write("Recommendations:\n")
            for i, rec in enumerate(recommendations, 1):
                f.write(f"  {i}. [{rec.get('priority', '').upper()}] {rec.get('message', '')}\n")
            f.write("\n")
        
        # Detailed step results
        f.write("\nDETAILED STEP RESULTS\n")
        f.write("-" * 70 + "\n")
        
        step_results = results.get("step_results", {})
        
        # Auto-fix results
        if "auto_fix" in step_results:
            f.write("\n1. AUTO-FIX RESULTS\n")
            auto_fix = step_results["auto_fix"]
            f.write(f"   Status: {auto_fix.get('status', 'Unknown')}\n")
            f.write(f"   Successful Tools: {', '.join(auto_fix.get('successful_tools', []))}\n")
            f.write(f"   Failed Tools: {', '.join(auto_fix.get('failed_tools', []))}\n")
        
        # Linter results
        if "linter_analysis" in step_results:
            f.write("\n2. LINTER ANALYSIS\n")
            linter = step_results["linter_analysis"]
            if linter.get("status") == "success":
                results_data = linter.get("results", {})
                f.write(f"   Total Issues: {results_data.get('total_issues', 0)}\n")
                f.write(f"   Files with Issues: {results_data.get('total_files_with_issues', 0)}\n")
                f.write(f"   Errors: {results_data.get('total_errors', 0)}\n")
                f.write(f"   Warnings: {results_data.get('total_warnings', 0)}\n")
        
        # Syntax validation results
        if "syntax_validation" in step_results:
            f.write("\n3. SYNTAX VALIDATION\n")
            syntax = step_results["syntax_validation"]
            if syntax.get("status") == "success":
                results_data = syntax.get("results", {}).get("summary", {})
                f.write(f"   Valid Files: {results_data.get('valid_files', 0)}\n")
                f.write(f"   Invalid Files: {results_data.get('invalid_files', 0)}\n")
                f.write(f"   AST Parseable: {results_data.get('ast_parseable_files', 0)}\n")
                f.write(f"   Compilable: {results_data.get('compilable_files', 0)}\n")
        
        # Import analysis results
        if "import_analysis" in step_results:
            f.write("\n4. IMPORT ANALYSIS\n")
            imports = step_results["import_analysis"]
            if imports.get("status") == "success":
                results_data = imports.get("results", {}).get("summary", {})
                f.write(f"   Total Imports: {results_data.get('total_imports', 0)}\n")
                f.write(f"   Duplicate Imports: {results_data.get('duplicate_imports', 0)}\n")
                f.write(f"   Circular Dependencies: {results_data.get('circular_dependencies', 0)}\n")
                f.write(f"   Conflicting Imports: {results_data.get('conflicting_imports', 0)}\n")
        
        # Signature analysis results
        if "signature_analysis" in step_results:
            f.write("\n5. SIGNATURE ANALYSIS\n")
            signatures = step_results["signature_analysis"]
            if signatures.get("status") == "success":
                results_data = signatures.get("results", {}).get("summary", {})
                f.write(f"   Total Functions: {results_data.get('total_functions', 0)}\n")
                f.write(f"   Total Function Calls: {results_data.get('total_function_calls', 0)}\n")
                f.write(f"   Signature Changes: {results_data.get('signature_changes', 0)}\n")
                f.write(f"   Compatibility Issues: {results_data.get('compatibility_issues', 0)}\n")

if __name__ == "__main__":
    main()