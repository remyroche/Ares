#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Sequential Auto-Fix - Simple Function-Based Version

Runs syntax fixing, linter analysis, and AST/compilation checking in sequence
with improved dependency handling and error management.
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import dependency manager for safe imports
from utils.dependency_manager import dependency_manager, safe_import

# Safe imports with fallbacks
ImportAnalyzer, _ = safe_import("analyzers.import_analyzer", None)
SignatureAnalyzer, _ = safe_import("analyzers.improved_signature_analyzer", None)
LinterAnalyzer, _ = safe_import("analyzers.linter_analyzer", None)
SyntaxValidator, _ = safe_import("analyzers.syntax_validator", None)
AutoFixer, _ = safe_import("fixers.auto_fixer", None)

# Core imports with fallbacks
try:
    from core.config import AnalysisConfig, CodeQualityConfig, get_default_config
except ImportError:
    CodeQualityConfig = None
    get_default_config = None

try:
    from utils.file_utils import find_python_files
except ImportError:
    find_python_files = None


def run_sequential_fixes(
    project_root: str = "/workspace/src",
    config: Optional[CodeQualityConfig] = None,
    save_report: bool = True,
    print_report: bool = True,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run sequential auto-fix pipeline.
    
    Args:
        project_root: Root directory of the project
        config: Configuration object (optional)
        save_report: Whether to save the report to file
        print_report: Whether to print the report to console
        output_dir: Directory to save reports
        
    Returns:
        Dict containing fix results
    """
    tprint("Starting sequential auto-fix pipeline...")
    
    # Get config with fallback
    if config is None:
        if get_default_config:
            config = get_default_config()
        else:
            config = _create_fallback_config()
    
    # Print dependency status
    dependency_manager.print_dependency_status()
    
    results = {
        "pipeline_info": {
            "pipeline_name": "sequential_fixer",
            "timestamp": datetime.now().isoformat(),
            "project_root": project_root,
            "config_used": str(type(config).__name__)
        },
        "steps": {},
        "summary": {
            "total_files_processed": 0,
            "total_issues_found": 0,
            "total_issues_fixed": 0,
            "execution_time_seconds": 0
        }
    }
    
    start_time = time.time()
    
    try:
        # Step 1: Auto-fix syntax and style issues
        tprint("\n" + "="*60)
        tprint("Step 1: Auto-fix syntax and style issues")
        tprint("="*60)
        
        if AutoFixer:
            fixer = AutoFixer(config)
            fix_results = fixer.run_auto_fix(project_root)
            results["steps"]["auto_fix"] = fix_results
            results["summary"]["total_issues_fixed"] += fix_results.get("issues_fixed", 0)
        else:
            results["steps"]["auto_fix"] = {"status": "skipped", "reason": "AutoFixer not available"}
        
        # Step 2: Run linter analysis
        tprint("\n" + "="*60)
        tprint("Step 2: Linter analysis and error reporting")
        tprint("="*60)
        
        if LinterAnalyzer:
            linter = LinterAnalyzer(config)
            linter_results = linter.analyze_directory(project_root)
            results["steps"]["linter_analysis"] = linter_results
            results["summary"]["total_issues_found"] += linter_results.get("total_issues", 0)
        else:
            results["steps"]["linter_analysis"] = {"status": "skipped", "reason": "LinterAnalyzer not available"}
        
        # Step 3: Validate AST parsing and compilation
        tprint("\n" + "="*60)
        tprint("Step 3: AST parsing and compilation validation")
        tprint("="*60)
        
        if SyntaxValidator:
            validator = SyntaxValidator(config)
            validation_results = validator.validate_directory(project_root)
            results["steps"]["syntax_validation"] = validation_results
        else:
            results["steps"]["syntax_validation"] = {"status": "skipped", "reason": "SyntaxValidator not available"}
        
        # Step 4: Analyze imports
        tprint("\n" + "="*60)
        tprint("Step 4: Import analysis")
        tprint("="*60)
        
        if ImportAnalyzer:
            import_analyzer = ImportAnalyzer(config)
            import_results = import_analyzer.analyze_directory(project_root)
            results["steps"]["import_analysis"] = import_results
        else:
            results["steps"]["import_analysis"] = {"status": "skipped", "reason": "ImportAnalyzer not available"}
        
        # Step 5: Analyze function signatures
        tprint("\n" + "="*60)
        tprint("Step 5: Function signature analysis")
        tprint("="*60)
        
        if SignatureAnalyzer:
            signature_analyzer = SignatureAnalyzer(config)
            signature_results = signature_analyzer.analyze_directory(project_root)
            results["steps"]["signature_analysis"] = signature_results
        else:
            results["steps"]["signature_analysis"] = {"status": "skipped", "reason": "SignatureAnalyzer not available"}
        
        # Calculate execution time
        execution_time = time.time() - start_time
        results["summary"]["execution_time_seconds"] = execution_time
        
        # Print summary if requested
        if print_report:
            _print_fix_summary(results)
        
        # Save report if requested
        if save_report:
            report_path = _save_fix_report(results, output_dir)
            results["report_path"] = str(report_path)
        
        tprint(f"\nSequential auto-fix pipeline completed in {execution_time:.2f} seconds")
        return results
        
    except Exception as e:
        tprint(f"Error during sequential auto-fix: {e}")
        results["error"] = str(e)
        results["summary"]["execution_time_seconds"] = time.time() - start_time
        return results


def _create_fallback_config():
    """Create a fallback configuration when core config is not available."""
    class FallbackConfig:
        def __init__(self):
            self.auto_fix = type('AutoFix', (), {
                'tools': ['isort', 'autoflake', 'pyupgrade', 'yesqa'],
                'aggressive': False,
                'max_line_length': 120
            })()
            self.linter = type('Linter', (), {
                'enabled': True,
                'tools': ['flake8', 'pylint']
            })()
            self.validation = type('Validation', (), {
                'enabled': True,
                'strict': False
            })()
    
    return FallbackConfig()


def _print_fix_summary(results: Dict[str, Any]):
    """Print a summary of the fix results."""
    tprint(f"\n{'='*60}")
    tprint("SEQUENTIAL AUTO-FIX SUMMARY")
    tprint(f"{'='*60}")
    
    summary = results.get("summary", {})
    tprint(f"Total files processed: {summary.get('total_files_processed', 0)}")
    tprint(f"Total issues found: {summary.get('total_issues_found', 0)}")
    tprint(f"Total issues fixed: {summary.get('total_issues_fixed', 0)}")
    tprint(f"Execution time: {summary.get('execution_time_seconds', 0):.2f} seconds")
    
    tprint(f"\nStep Results:")
    for step_name, step_result in results.get("steps", {}).items():
        status = step_result.get("status", "completed")
        tprint(f"  {step_name}: {status}")
    
    tprint(f"{'='*60}")


def _save_fix_report(results: Dict[str, Any], output_dir: Optional[str] = None) -> Path:
    """Save fix results to a JSON file."""
    if output_dir is None:
        output_dir = Path.cwd() / "code_quality" / "reports"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"sequential_fix_results_{timestamp}.json"
    
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    tprint(f"Fix report saved to: {report_path}")
    return report_path


# Convenience function for backward compatibility
def run_sequential_fixer(
    project_root: str = "/workspace/src",
    config: Optional[CodeQualityConfig] = None,
    save_report: bool = True,
    print_report: bool = True,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Alias for run_sequential_fixes for backward compatibility."""
    return run_sequential_fixes(project_root, config, save_report, print_report, output_dir)