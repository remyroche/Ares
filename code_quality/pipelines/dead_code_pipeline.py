#!/usr/bin/env python3
"""
Dead Code Analysis Pipeline

Specialized pipeline for dead code detection and analysis including:
- Unused imports detection
- Dead function detection
- Unreachable code detection
- Unused variables detection
- Dead code removal automation
"""

import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import dead code analyzers (ONLY dead code-related)
from analyzers.enhanced_dead_code_analyzer import EnhancedDeadCodeAnalyzer
from analyzers.undefined_names_analyzer import UndefinedNamesAnalyzer
from analyzers.enhanced_import_analysis import EnhancedImportAnalyzer

# Note: AutoFixDeadCode and DeadCodeAnalyzer were removed as they were redundant
# The enhanced_dead_code_analyzer provides all necessary functionality

# Import core components
from core.config import get_default_config
from plugins.plugin_registry import PluginRegistry
from plugins.plugin_manager import PluginManager


class DeadCodePipeline:
    """Specialized pipeline for dead code analysis."""
    
    def __init__(self, project_root: str = None, enable_plugins: bool = True):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        self.enable_plugins = enable_plugins
        
        # Initialize analyzers
        self.config = get_default_config()
        self.dead_code_analyzer = EnhancedDeadCodeAnalyzer(self.config)
        self.undefined_names_analyzer = UndefinedNamesAnalyzer(self.config)
        self.import_analyzer = EnhancedImportAnalyzer(None)  # Use default config
        
        # Initialize auto-fixer
        # self.auto_fixer = AutoFixDeadCode()  # Removed during cleanup
        
        # Initialize plugin system
        if self.enable_plugins:
            self.plugin_registry = PluginRegistry()
            self.plugin_manager = PluginManager(self.plugin_registry)
            self._register_dead_code_plugins()
        
        # Setup reports directory
        self.reports_dir = self.project_root / "code_quality" / "reports" / "dead_code"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def _register_dead_code_plugins(self):
        """Register dead code analysis plugins."""
        try:
            # Register dead code analysis plugins
            from plugins.autoflake_fixer import AutoflakeFixer
            from plugins.creosote_analyzer import CreosoteAnalyzer
            from plugins.fawltydeps_analyzer import FawltyDepsAnalyzer
            
            self.plugin_registry.register_plugin(AutoflakeFixer)
            self.plugin_registry.register_plugin(CreosoteAnalyzer)
            self.plugin_registry.register_plugin(FawltyDepsAnalyzer)
            
            print(f"✅ Registered {len(self.plugin_registry.list_plugins())} dead code plugins")
        except ImportError as e:
            print(f"⚠️  Warning: Could not register some plugins: {e}")
    
    def run_basic_dead_code_analysis(self) -> Dict[str, Any]:
        """Run basic dead code analysis using Vulture."""
        print("\n" + "="*60)
        print("Running Basic Dead Code Analysis")
        print("="*60)
        
        try:
            results = self.dead_code_analyzer.analyze_dead_code(str(self.project_root))
            
            # Generate dead code report
            dead_code_report = {
                "timestamp": self.timestamp,
                "analysis_type": "basic_dead_code",
                "project_root": str(self.project_root),
                "total_issues": len(results.get("issues", [])),
                "dead_code_issues": len([i for i in results.get("issues", []) 
                                       if i.get("type") == "dead_code"]),
                "unreachable_code_issues": len([i for i in results.get("issues", []) 
                                              if i.get("type") == "unreachable_code"]),
                "results": results
            }
            
            # Save report
            report_path = self.reports_dir / f"basic_dead_code_{self.timestamp}.json"
            with open(report_path, "w") as f:
                json.dump(dead_code_report, f, indent=2)
            
            return {
                "status": "completed",
                "report_path": str(report_path),
                "total_issues": dead_code_report["total_issues"],
                "dead_code_count": dead_code_report["dead_code_issues"],
                "unreachable_code_count": dead_code_report["unreachable_code_issues"],
                "results": results
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_enhanced_dead_code_analysis(self) -> Dict[str, Any]:
        """Run enhanced dead code analysis with reduced false positives."""
        print("\n" + "="*60)
        print("Running Enhanced Dead Code Analysis")
        print("="*60)
        
        try:
            report = self.dead_code_analyzer.analyze_directory(str(self.project_root))
            
            # Convert report to dictionary format
            results = {
                "total_issues": report.total_issues,
                "issues_by_type": report.issues_by_type,
                "issues_by_file": {str(k): v for k, v in report.issues_by_file.items()},
                "issues_by_severity": {str(k): v for k, v in report.issues_by_severity.items()},
                "issues_by_tool": {str(k): v for k, v in report.issues_by_tool.items()},
                "confidence_distribution": report.confidence_distribution,
                "potential_savings": report.potential_savings,
                "false_positives_filtered": report.false_positives_filtered,
                "impact_analysis": report.impact_analysis
            }
            
            # Generate enhanced dead code report
            enhanced_report = {
                "timestamp": self.timestamp,
                "analysis_type": "enhanced_dead_code",
                "project_root": str(self.project_root),
                "total_issues": report.total_issues,
                "high_confidence_issues": len([i for i in report.issues_by_severity.get("high", [])]),
                "cross_file_usage_checked": True,  # Enhanced analyzer always checks cross-file usage
                "results": results
            }
            
            # Save report
            report_path = self.reports_dir / f"enhanced_dead_code_{self.timestamp}.json"
            with open(report_path, "w") as f:
                json.dump(enhanced_report, f, indent=2)
            
            return {
                "status": "completed",
                "report_path": str(report_path),
                "total_issues": enhanced_report["total_issues"],
                "high_confidence_issues": enhanced_report["high_confidence_issues"],
                "results": results
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_unused_imports_analysis(self) -> Dict[str, Any]:
        """Run unused imports analysis."""
        print("\n" + "="*60)
        print("Running Unused Imports Analysis")
        print("="*60)
        
        try:
            results = self.import_analyzer.analyze_unused_imports(str(self.project_root))
            
            # Generate unused imports report
            unused_imports_report = {
                "timestamp": self.timestamp,
                "analysis_type": "unused_imports",
                "project_root": str(self.project_root),
                "total_unused_imports": len(results.get("unused_imports", [])),
                "files_with_unused_imports": len(set(i.get("file", "") for i in results.get("unused_imports", []))),
                "results": results
            }
            
            # Save report
            report_path = self.reports_dir / f"unused_imports_{self.timestamp}.json"
            with open(report_path, "w") as f:
                json.dump(unused_imports_report, f, indent=2)
            
            return {
                "status": "completed",
                "report_path": str(report_path),
                "total_unused_imports": unused_imports_report["total_unused_imports"],
                "files_affected": unused_imports_report["files_with_unused_imports"],
                "results": results
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_undefined_names_analysis(self) -> Dict[str, Any]:
        """Run undefined names analysis."""
        print("\n" + "="*60)
        print("Running Undefined Names Analysis")
        print("="*60)
        
        try:
            results = self.undefined_names_analyzer.analyze_undefined_names(str(self.project_root))
            
            # Generate undefined names report
            undefined_names_report = {
                "timestamp": self.timestamp,
                "analysis_type": "undefined_names",
                "project_root": str(self.project_root),
                "total_undefined_names": len(results.get("undefined_names", [])),
                "files_with_undefined_names": len(set(i.get("file", "") for i in results.get("undefined_names", []))),
                "results": results
            }
            
            # Save report
            report_path = self.reports_dir / f"undefined_names_{self.timestamp}.json"
            with open(report_path, "w") as f:
                json.dump(undefined_names_report, f, indent=2)
            
            return {
                "status": "completed",
                "report_path": str(report_path),
                "total_undefined_names": undefined_names_report["total_undefined_names"],
                "files_affected": undefined_names_report["files_with_undefined_names"],
                "results": results
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_auto_fix_dead_code(self) -> Dict[str, Any]:
        """Run automated dead code fixing."""
        print("\n" + "="*60)
        print("Running Automated Dead Code Fixing")
        print("="*60)
        
        # Auto-fixer is not available, return disabled status
        return {
            "status": "disabled",
            "message": "Auto-fixer functionality is not available in this version",
            "fixes_applied": 0,
            "files_modified": 0,
            "fixes_failed": 0
        }
    
    def run_plugin_analysis(self) -> Dict[str, Any]:
        """Run plugin-based dead code analysis."""
        if not self.enable_plugins:
            return {"status": "disabled", "message": "Plugins are disabled"}
        
        print("\n" + "="*60)
        print("Running Plugin-Based Dead Code Analysis")
        print("="*60)
        
        try:
            plugin_results = {}
            
            # Get dead code related plugins
            dead_code_plugins = [
                plugin for plugin in self.plugin_registry.list_plugins()
                if any(keyword in plugin.lower() for keyword in ["autoflake", "creosote", "fawltydeps"])
            ]
            
            for plugin_name in dead_code_plugins:
                try:
                    result = self.plugin_manager.execute_plugin(
                        plugin_name, 
                        {"project_root": str(self.project_root), "analysis_type": "dead_code"}
                    )
                    plugin_results[plugin_name] = result
                except Exception as e:
                    plugin_results[plugin_name] = {"status": "error", "error": str(e)}
            
            return {
                "status": "completed",
                "plugins_executed": len(plugin_results),
                "results": plugin_results
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_all_dead_code_analysis(self) -> Dict[str, Any]:
        """Run comprehensive dead code analysis."""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE DEAD CODE ANALYSIS PIPELINE")
        print(f"{'='*80}")
        print(f"Project root: {self.project_root}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Plugins enabled: {self.enable_plugins}")
        
        total_start = time.time()
        
        # Run all dead code analyses
        self.results["basic_dead_code"] = self.run_basic_dead_code_analysis()
        self.results["enhanced_dead_code"] = self.run_enhanced_dead_code_analysis()
        self.results["unused_imports"] = self.run_unused_imports_analysis()
        self.results["undefined_names"] = self.run_undefined_names_analysis()
        self.results["auto_fix"] = self.run_auto_fix_dead_code()
        self.results["plugin_analysis"] = self.run_plugin_analysis()
        
        # Generate summary
        total_time = time.time() - total_start
        self.results["summary"] = {
            "timestamp": self.timestamp,
            "project_root": str(self.project_root),
            "total_execution_time": total_time,
            "analysis_categories": len(self.results) - 1,  # Exclude summary
            "plugins_enabled": self.enable_plugins,
            "status": "completed"
        }
        
        # Save comprehensive report
        report_path = self.reports_dir / f"dead_code_analysis_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n{'='*80}")
        print("DEAD CODE ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Reports saved to: {self.reports_dir}")
        print(f"Main report: {report_path}")
        
        return self.results


def main():
    """Main entry point for the dead code pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Dead Code Analysis Pipeline - Comprehensive dead code detection and removal"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root directory (default: current directory)"
    )
    parser.add_argument(
        "--disable-plugins",
        action="store_true",
        help="Disable plugin system"
    )
    parser.add_argument(
        "--analysis-type",
        type=str,
        choices=["basic", "enhanced", "auto_fix", "all"],
        default="enhanced",
        help="Type of dead code analysis to perform (default: enhanced)"
    )
    parser.add_argument(
        "--auto-fix",
        action="store_true",
        help="Enable automatic fixing of dead code"
    )
    
    args = parser.parse_args()
    
    pipeline = DeadCodePipeline(
        project_root=args.project_root,
        enable_plugins=not args.disable_plugins
    )
    
    if args.analysis_type == "all":
        results = pipeline.run_all_dead_code_analysis()
    elif args.analysis_type == "basic":
        results = pipeline.run_basic_dead_code_analysis()
    elif args.analysis_type == "enhanced":
        results = pipeline.run_enhanced_dead_code_analysis()
        # If auto-fix is requested, also run auto-fix
        if args.auto_fix:
            auto_fix_results = pipeline.run_auto_fix_dead_code()
            results["auto_fix"] = auto_fix_results
    elif args.analysis_type == "unused_imports":
        results = pipeline.run_unused_imports_analysis()
    elif args.analysis_type == "undefined_names":
        results = pipeline.run_undefined_names_analysis()
    elif args.analysis_type == "auto_fix":
        results = pipeline.run_auto_fix_dead_code()
    
    print(f"\nDead code pipeline completed with status: {results.get('status', 'unknown')}")


if __name__ == "__main__":
    main()