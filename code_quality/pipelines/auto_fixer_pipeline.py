#!/usr/bin/env python3
"""
Auto-Fixer Pipeline

Specialized pipeline for automated code fixing including:
- Import fixes
- Syntax fixes
- Type hint fixes
- Formatting fixes
- Dead code removal
- Code style fixes
"""

import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import main fixers (ONLY auto-fixing related)
from fixers.auto_fixer import AutoFixer
from fixers.conservative_auto_fixer import ConservativeAutoFixer
from fixers.sequential_fixer_fixed import SequentialFixer

# Import script-based fixers (ONLY auto-fixing related)
from scripts.advanced_syntax_fixer import AdvancedSyntaxFixer
from scripts.enhanced_type_hints import TypeHintEnhancer, EnhancedTypeHintAdder
from scripts.robust_async_fixer import RobustAsyncFixer
from scripts.fix_missing_imports import ImportFixer
from scripts.detect_circular_imports import ImportAnalyzer
from scripts.fix_async_await import AsyncAwaitFixer, AsyncPatternFixer

# Import comprehensive fixers (ONLY auto-fixing related)
# Note: Some comprehensive fixers may not be available

# Import undefined names fixers (ONLY auto-fixing related)
from fixers.undefined_names_fixers.fix_undefined_names import UndefinedNamesFixer
# Note: Other undefined names fixers were removed as redundant during cleanup

# Import plugin fixers (ONLY auto-fixing related)
from plugins.black_fixer import BlackFixer
from plugins.ruff_fixer import RuffFixer
from plugins.autopep8_fixer import Autopep8Fixer
from plugins.isort_fixer import IsortFixer
from plugins.autoflake_fixer import AutoflakeFixer
from plugins.docformatter_fixer import DocformatterFixer
from plugins.flynt_fixer import FlyntFixer
from plugins.future_annotations_fixer import FutureAnnotationsFixer
from plugins.import_hygiene_fixer import ImportHygieneFixer
from plugins.pyupgrade_fixer import PyupgradeFixer
from plugins.unify_fixer import UnifyFixer
from plugins.yapf_fixer import YapfFixer
from plugins.yesqa_fixer import YesqaFixer

# Import core components
from core.config import get_default_config
from plugins.plugin_registry import PluginRegistry
from plugins.plugin_manager import PluginManager


class AutoFixerPipeline:
    """Specialized pipeline for automated code fixing."""
    
    def __init__(self, project_root: str = None, enable_plugins: bool = True, conservative: bool = False, balanced: bool = False):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        self.enable_plugins = enable_plugins
        self.conservative = conservative
        self.balanced = balanced
        
        # Initialize fixers
        self.config = get_default_config()
        if conservative:
            self.auto_fixer = ConservativeAutoFixer(self.config)
        elif balanced:
            # Use regular AutoFixer but with balanced settings
            self.auto_fixer = AutoFixer(self.config)
            # Override config for balanced mode
            self.config.max_fixes_per_file = 10  # Limit fixes per file
            self.config.skip_complex_files = True  # Skip very complex files
            self.config.enable_syntax_fixes = True  # Enable syntax fixes
            self.config.enable_import_fixes = True  # Enable import fixes
        else:
            self.auto_fixer = AutoFixer(self.config)
        
        self.sequential_fixer = SequentialFixer(self.config)
        
        # Initialize script-based fixers
        self.syntax_fixer = AdvancedSyntaxFixer(str(self.project_root))
        self.type_hint_enhancer = TypeHintEnhancer()
        self.async_fixer = RobustAsyncFixer(str(self.project_root))
        self.circular_import_detector = ImportAnalyzer(str(self.project_root))
        self.type_hint_adder = EnhancedTypeHintAdder(str(self.project_root))
        self.missing_import_fixer = ImportFixer(str(self.project_root))
        self.async_await_fixer = AsyncAwaitFixer(set())  # Empty set for now
        self.async_pattern_fixer = AsyncPatternFixer(str(self.project_root))
        
        # Initialize comprehensive fixers
        # Note: Some comprehensive fixers may not be available
        
        # Initialize plugin system
        if self.enable_plugins:
            try:
                self.plugin_registry = PluginRegistry()
                self.plugin_manager = PluginManager(self.plugin_registry)
                self._register_fixer_plugins()
            except Exception as e:
                print(f"⚠️  Warning: Could not initialize plugin system: {e}")
                self.enable_plugins = False
        
        # Setup reports directory
        self.reports_dir = self.project_root / "code_quality" / "reports" / "auto_fixer"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def _register_fixer_plugins(self):
        """Register auto-fixer plugins."""
        try:
            # Register formatting and fixing plugins
            from plugins.black_fixer import BlackFixer
            from plugins.isort_fixer import IsortFixer
            from plugins.autopep8_fixer import Autopep8Fixer
            from plugins.autoflake_fixer import AutoflakeFixer
            from plugins.docformatter_fixer import DocformatterFixer
            from plugins.flynt_fixer import FlyntFixer
            from plugins.future_annotations_fixer import FutureAnnotationsFixer
            from plugins.import_hygiene_fixer import ImportHygieneFixer
            from plugins.pyupgrade_fixer import PyupgradeFixer
            from plugins.unify_fixer import UnifyFixer
            from plugins.yapf_fixer import YapfFixer
            from plugins.yesqa_fixer import YesqaFixer
            
            self.plugin_registry.register_plugin(BlackFixer)
            self.plugin_registry.register_plugin(IsortFixer)
            self.plugin_registry.register_plugin(Autopep8Fixer)
            self.plugin_registry.register_plugin(AutoflakeFixer)
            self.plugin_registry.register_plugin(DocformatterFixer)
            self.plugin_registry.register_plugin(FlyntFixer)
            self.plugin_registry.register_plugin(FutureAnnotationsFixer)
            self.plugin_registry.register_plugin(ImportHygieneFixer)
            self.plugin_registry.register_plugin(PyupgradeFixer)
            self.plugin_registry.register_plugin(UnifyFixer)
            self.plugin_registry.register_plugin(YapfFixer)
            self.plugin_registry.register_plugin(YesqaFixer)
            
            print(f"✅ Registered {len(self.plugin_registry.list_plugins())} auto-fixer plugins")
        except ImportError as e:
            print(f"⚠️  Warning: Could not register some plugins: {e}")
    
    def run_import_fixes(self) -> Dict[str, Any]:
        """Run comprehensive import fixes with enhanced auto-detection."""
        print("\n" + "="*60)
        print("Running Enhanced Import Fixes with Auto-Detection")
        print("="*60)
        
        try:
            results = {}
            
            # Get all Python files
            file_paths = list(self.project_root.glob("**/*.py"))
            print(f"🔍 Analyzing {len(file_paths)} Python files for import issues...")
            
            # Enhanced auto-detection import fixes (primary method)
            print("🚀 Running enhanced auto-detection for missing imports...")
            auto_detection_results = self.missing_import_fixer.auto_fix_all_files(
                [str(f) for f in file_paths], 
                dry_run=False
            )
            results["auto_detection_fixes"] = auto_detection_results
            
            # Only run traditional fixes if auto-detection found issues or failed
            auto_fixed = auto_detection_results.get("fixed", 0)
            auto_failed = auto_detection_results.get("failed", 0)
            
            if auto_fixed == 0 and auto_failed > 0:
                print("🔧 Auto-detection had issues, running traditional missing import fixes...")
                try:
                    missing_results = self.missing_import_fixer.fix_all_imports(dry_run=False)
                    results["traditional_import_fixes"] = missing_results
                except Exception as e:
                    print(f"⚠️  Traditional import fixes also failed: {e}")
                    results["traditional_import_fixes"] = {"status": "error", "error": str(e)}
            else:
                print("✅ Auto-detection completed successfully, skipping traditional fixes")
                results["traditional_import_fixes"] = {"status": "skipped", "reason": "auto_detection_successful"}
            
            # Import placement correction (run after import fixes)
            print("🔧 Correcting import placement...")
            try:
                from scripts.fix_incorrect_imports import find_incorrect_imports, fix_incorrect_imports
                
                # Find files with incorrect import placement
                files_with_issues = []
                for file_path in file_paths:
                    incorrect_imports = find_incorrect_imports(str(file_path))
                    if incorrect_imports:
                        files_with_issues.append((str(file_path), incorrect_imports))
                
                # Fix incorrect imports
                placement_fixed = 0
                for file_path, _ in files_with_issues:
                    if fix_incorrect_imports(file_path):
                        placement_fixed += 1
                
                results["import_placement_fixes"] = {
                    "files_with_incorrect_placement": len(files_with_issues),
                    "files_fixed": placement_fixed,
                    "status": "success"
                }
                
                if files_with_issues:
                    print(f"  ✅ Fixed import placement in {placement_fixed} files")
                else:
                    print(f"  ✅ No incorrect import placement found")
                    
            except Exception as e:
                print(f"⚠️  Import placement correction failed: {e}")
                results["import_placement_fixes"] = {"status": "error", "error": str(e)}
            
            # Circular import detection (always run)
            print("🔄 Detecting circular imports...")
            try:
                circular_report = self.circular_import_detector.generate_report()
                results["circular_import_fixes"] = {
                    "circular_imports_found": circular_report.get("circular_imports", {}).get("count", 0),
                    "cycles": circular_report.get("circular_imports", {}).get("cycles", [])
                }
            except Exception as e:
                print(f"⚠️  Circular import detection failed: {e}")
                results["circular_import_fixes"] = {"status": "error", "error": str(e)}
            
            # Enhanced reporting
            total_auto_fixed = auto_detection_results.get("fixed", 0)
            total_auto_failed = auto_detection_results.get("failed", 0)
            total_traditional_fixed = results["traditional_import_fixes"].get("fixes_applied", 0) if isinstance(results["traditional_import_fixes"], dict) else 0
            total_placement_fixed = results["import_placement_fixes"].get("files_fixed", 0) if isinstance(results["import_placement_fixes"], dict) else 0
            total_circular_found = results["circular_import_fixes"].get("circular_imports_found", 0) if isinstance(results["circular_import_fixes"], dict) else 0
            
            print(f"\n📊 Import Fixes Summary:")
            print(f"  ✅ Auto-detection fixes: {total_auto_fixed} files")
            print(f"  ❌ Auto-detection failures: {total_auto_failed} files")
            print(f"  🔧 Traditional fixes: {total_traditional_fixed} files")
            print(f"  📍 Import placement fixes: {total_placement_fixed} files")
            print(f"  🔄 Circular imports found: {total_circular_found}")
            
            # Show module breakdown for auto-detection
            module_counts = auto_detection_results.get("module_counts", {})
            if module_counts:
                print(f"\n📈 Imports added by module:")
                for module, count in sorted(module_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"    {module}: {count} files")
            
            # Show specific files that were fixed
            fixed_files = auto_detection_results.get("fixed_files", [])
            if fixed_files:
                print(f"\n📁 Files fixed by auto-detection:")
                for file_path in fixed_files[:10]:  # Show first 10
                    print(f"    ✓ {file_path}")
                if len(fixed_files) > 10:
                    print(f"    ... and {len(fixed_files) - 10} more files")
            
            # Generate comprehensive import fixes report
            import_fixes_report = {
                "timestamp": self.timestamp,
                "analysis_type": "enhanced_import_fixes",
                "project_root": str(self.project_root),
                "auto_detection_fixes": total_auto_fixed,
                "auto_detection_failures": total_auto_failed,
                "traditional_import_fixes": total_traditional_fixed,
                "import_placement_fixes": total_placement_fixed,
                "circular_imports_found": total_circular_found,
                "module_breakdown": module_counts,
                "files_analyzed": len(file_paths),
                "files_fixed": fixed_files,
                "results": results
            }
            
            # Save report
            report_path = self.reports_dir / f"enhanced_import_fixes_{self.timestamp}.json"
            with open(report_path, "w") as f:
                json.dump(import_fixes_report, f, indent=2)
            
            print(f"\n💾 Report saved to: {report_path}")
            
            return {
                "status": "completed",
                "report_path": str(report_path),
                "auto_detection_fixes": total_auto_fixed,
                "auto_detection_failures": total_auto_failed,
                "traditional_import_fixes": total_traditional_fixed,
                "circular_imports_found": total_circular_found,
                "module_breakdown": module_counts,
                "files_analyzed": len(file_paths),
                "files_fixed": fixed_files,
                "results": results
            }
        except Exception as e:
            print(f"❌ Enhanced import fixes failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def run_enhanced_auto_detection(self, file_pattern: str = "**/*.py", dry_run: bool = False) -> Dict[str, Any]:
        """Run enhanced auto-detection for missing imports only."""
        print("\n" + "="*60)
        print("Running Enhanced Auto-Detection for Missing Imports")
        print("="*60)
        
        try:
            # Get file paths based on pattern
            file_paths = list(self.project_root.glob(file_pattern))
            print(f"🔍 Analyzing {len(file_paths)} Python files...")
            
            # Run auto-detection
            auto_detection_results = self.missing_import_fixer.auto_fix_all_files(
                [str(f) for f in file_paths], 
                dry_run=dry_run
            )
            
            # Enhanced reporting
            total_files = auto_detection_results.get("files_to_fix", 0)
            total_imports = auto_detection_results.get("imports_to_add", 0)
            module_counts = auto_detection_results.get("module_counts", {})
            
            if dry_run:
                print(f"\n📊 Auto-Detection Summary (DRY RUN):")
                print(f"  📁 Files analyzed: {len(file_paths)}")
                print(f"  🎯 Files with missing imports: {total_files}")
                print(f"  📦 Total imports to add: {total_imports}")
            else:
                print(f"\n📊 Auto-Detection Summary:")
                print(f"  📁 Files analyzed: {len(file_paths)}")
                print(f"  ✅ Files fixed: {auto_detection_results.get('fixed', 0)}")
                print(f"  ❌ Files failed: {auto_detection_results.get('failed', 0)}")
            
            # Show module breakdown
            if module_counts:
                print(f"\n📈 Imports by module:")
                for module, count in sorted(module_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"    {module}: {count} files")
            
            # Generate report
            auto_detection_report = {
                "timestamp": self.timestamp,
                "analysis_type": "enhanced_auto_detection",
                "project_root": str(self.project_root),
                "file_pattern": file_pattern,
                "dry_run": dry_run,
                "files_analyzed": len(file_paths),
                "files_with_missing_imports": total_files,
                "total_imports_to_add": total_imports,
                "module_breakdown": module_counts,
                "results": auto_detection_results
            }
            
            # Save report
            report_suffix = "dry_run" if dry_run else "applied"
            report_path = self.reports_dir / f"auto_detection_{report_suffix}_{self.timestamp}.json"
            with open(report_path, "w") as f:
                json.dump(auto_detection_report, f, indent=2)
            
            return {
                "status": "completed",
                "report_path": str(report_path),
                "files_analyzed": len(file_paths),
                "files_with_missing_imports": total_files,
                "total_imports_to_add": total_imports,
                "module_breakdown": module_counts,
                "dry_run": dry_run,
                "results": auto_detection_results
            }
        except Exception as e:
            print(f"❌ Enhanced auto-detection failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def run_syntax_fixes(self) -> Dict[str, Any]:
        """Run syntax fixes."""
        print("\n" + "="*60)
        print("Running Syntax Fixes")
        print("="*60)
        
        try:
            results = {}
            
            # Advanced syntax fixes
            syntax_results = self.syntax_fixer.fix_syntax_issues(str(self.project_root))
            results["advanced_syntax_fixes"] = syntax_results
            
            # Bulk syntax cleanup
            bulk_results = self.bulk_cleanup.cleanup_syntax(str(self.project_root))
            results["bulk_syntax_cleanup"] = bulk_results
            
            # Generate syntax fixes report
            syntax_fixes_report = {
                "timestamp": self.timestamp,
                "analysis_type": "syntax_fixes",
                "project_root": str(self.project_root),
                "syntax_fixes": syntax_results.get("fixes_applied", 0),
                "bulk_cleanup_fixes": bulk_results.get("fixes_applied", 0),
                "results": results
            }
            
            # Save report
            report_path = self.reports_dir / f"syntax_fixes_{self.timestamp}.json"
            with open(report_path, "w") as f:
                json.dump(syntax_fixes_report, f, indent=2)
            
            return {
                "status": "completed",
                "report_path": str(report_path),
                "total_syntax_fixes": (syntax_results.get("fixes_applied", 0) + 
                                     bulk_results.get("fixes_applied", 0)),
                "results": results
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_type_hint_fixes(self) -> Dict[str, Any]:
        """Run type hint fixes."""
        print("\n" + "="*60)
        print("Running Type Hint Fixes")
        print("="*60)
        
        try:
            results = {}
            
            # Enhanced type hints
            enhanced_results = self.type_hint_enhancer.enhance_type_hints(str(self.project_root))
            results["enhanced_type_hints"] = enhanced_results
            
            # Add type hints
            add_results = self.type_hint_adder.add_type_hints(str(self.project_root))
            results["add_type_hints"] = add_results
            
            # Generate type hint fixes report
            type_hint_fixes_report = {
                "timestamp": self.timestamp,
                "analysis_type": "type_hint_fixes",
                "project_root": str(self.project_root),
                "enhanced_type_hints": enhanced_results.get("fixes_applied", 0),
                "added_type_hints": add_results.get("fixes_applied", 0),
                "results": results
            }
            
            # Save report
            report_path = self.reports_dir / f"type_hint_fixes_{self.timestamp}.json"
            with open(report_path, "w") as f:
                json.dump(type_hint_fixes_report, f, indent=2)
            
            return {
                "status": "completed",
                "report_path": str(report_path),
                "total_type_hint_fixes": (enhanced_results.get("fixes_applied", 0) + 
                                        add_results.get("fixes_applied", 0)),
                "results": results
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_async_fixes(self) -> Dict[str, Any]:
        """Run async/await fixes."""
        print("\n" + "="*60)
        print("Running Async/Await Fixes")
        print("="*60)
        
        try:
            results = {}
            
            # Robust async fixes
            robust_results = self.async_fixer.fix_async_issues(str(self.project_root))
            results["robust_async_fixes"] = robust_results
            
            # Async/await fixes
            await_results = self.async_await_fixer.fix_async_await(str(self.project_root))
            results["async_await_fixes"] = await_results
            
            # Generate async fixes report
            async_fixes_report = {
                "timestamp": self.timestamp,
                "analysis_type": "async_fixes",
                "project_root": str(self.project_root),
                "robust_async_fixes": robust_results.get("fixes_applied", 0),
                "async_await_fixes": await_results.get("fixes_applied", 0),
                "results": results
            }
            
            # Save report
            report_path = self.reports_dir / f"async_fixes_{self.timestamp}.json"
            with open(report_path, "w") as f:
                json.dump(async_fixes_report, f, indent=2)
            
            return {
                "status": "completed",
                "report_path": str(report_path),
                "total_async_fixes": (robust_results.get("fixes_applied", 0) + 
                                    await_results.get("fixes_applied", 0)),
                "results": results
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_dead_code_fixes(self) -> Dict[str, Any]:
        """Run dead code fixes."""
        print("\n" + "="*60)
        print("Running Dead Code Fixes")
        print("="*60)
        
        try:
            results = self.auto_dead_code_fixer.auto_fix_dead_code(str(self.project_root))
            
            # Generate dead code fixes report
            dead_code_fixes_report = {
                "timestamp": self.timestamp,
                "analysis_type": "dead_code_fixes",
                "project_root": str(self.project_root),
                "fixes_applied": results.get("fixes_applied", 0),
                "files_modified": results.get("files_modified", 0),
                "fixes_failed": results.get("fixes_failed", 0),
                "results": results
            }
            
            # Save report
            report_path = self.reports_dir / f"dead_code_fixes_{self.timestamp}.json"
            with open(report_path, "w") as f:
                json.dump(dead_code_fixes_report, f, indent=2)
            
            return {
                "status": "completed",
                "report_path": str(report_path),
                "fixes_applied": dead_code_fixes_report["fixes_applied"],
                "files_modified": dead_code_fixes_report["files_modified"],
                "results": results
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_plugin_fixes(self) -> Dict[str, Any]:
        """Run plugin-based fixes."""
        if not self.enable_plugins:
            return {"status": "disabled", "message": "Plugins are disabled"}
        
        print("\n" + "="*60)
        print("Running Plugin-Based Fixes")
        print("="*60)
        
        try:
            plugin_results = {}
            
            # Get formatting and fixing plugins
            fixer_plugins = self.plugin_registry.get_plugins_by_category(PluginCategory.FORMATTING)
            fixer_plugins.extend(self.plugin_registry.get_plugins_by_category(PluginCategory.SYNTAX))
            
            for plugin_name in fixer_plugins:
                try:
                    result = self.plugin_manager.execute_plugin(
                        plugin_name, 
                        {"project_root": str(self.project_root), "fix_type": "auto_fix"}
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
    
    def run_sequential_fixes(self) -> Dict[str, Any]:
        """Run sequential fixes for comprehensive code improvement."""
        print("\n" + "="*60)
        print("Running Sequential Fixes")
        print("="*60)
        
        try:
            results = self.sequential_fixer.fix_all_issues(str(self.project_root))
            
            # Generate sequential fixes report
            sequential_fixes_report = {
                "timestamp": self.timestamp,
                "analysis_type": "sequential_fixes",
                "project_root": str(self.project_root),
                "total_fixes": results.get("total_fixes", 0),
                "files_processed": results.get("files_processed", 0),
                "fixes_by_category": results.get("fixes_by_category", {}),
                "results": results
            }
            
            # Save report
            report_path = self.reports_dir / f"sequential_fixes_{self.timestamp}.json"
            with open(report_path, "w") as f:
                json.dump(sequential_fixes_report, f, indent=2)
            
            return {
                "status": "completed",
                "report_path": str(report_path),
                "total_fixes": sequential_fixes_report["total_fixes"],
                "files_processed": sequential_fixes_report["files_processed"],
                "results": results
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_all_auto_fixes(self) -> Dict[str, Any]:
        """Run comprehensive auto-fixing."""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE AUTO-FIXER PIPELINE")
        print(f"{'='*80}")
        print(f"Project root: {self.project_root}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Conservative mode: {self.conservative}")
        print(f"Balanced mode: {self.balanced}")
        print(f"Plugins enabled: {self.enable_plugins}")
        
        total_start = time.time()
        
        # Run all auto-fixes (enhanced auto-detection is included in import_fixes)
        self.results["import_fixes"] = self.run_import_fixes()
        self.results["syntax_fixes"] = self.run_syntax_fixes()
        self.results["type_hint_fixes"] = self.run_type_hint_fixes()
        self.results["async_fixes"] = self.run_async_fixes()
        self.results["dead_code_fixes"] = self.run_dead_code_fixes()
        self.results["plugin_fixes"] = self.run_plugin_fixes()
        self.results["sequential_fixes"] = self.run_sequential_fixes()
        
        # Generate summary
        total_time = time.time() - total_start
        self.results["summary"] = {
            "timestamp": self.timestamp,
            "project_root": str(self.project_root),
            "total_execution_time": total_time,
            "fix_categories": len(self.results) - 1,  # Exclude summary
            "conservative_mode": self.conservative,
            "balanced_mode": self.balanced,
            "plugins_enabled": self.enable_plugins,
            "status": "completed"
        }
        
        # Save comprehensive report
        report_path = self.reports_dir / f"auto_fixer_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n{'='*80}")
        print("AUTO-FIXER PIPELINE COMPLETE")
        print(f"{'='*80}")
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Reports saved to: {self.reports_dir}")
        print(f"Main report: {report_path}")
        
        return self.results


def main():
    """Main entry point for the auto-fixer pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Auto-Fixer Pipeline - Comprehensive automated code fixing"
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
        "--conservative",
        action="store_true",
        help="Use conservative auto-fixing (safer but fewer fixes)"
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Use balanced auto-fixing (moderate safety with more fixes)"
    )
    parser.add_argument(
        "--fix-type",
        type=str,
        choices=["imports", "auto_detection", "syntax", "type_hints", "async", "dead_code", "all"],
        default="imports",
        help="Type of fixes to apply (default: imports)"
    )
    parser.add_argument(
        "--file-pattern",
        type=str,
        default="**/*.py",
        help="File pattern for auto-detection (default: **/*.py)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (show what would be fixed without making changes)"
    )
    
    args = parser.parse_args()
    
    pipeline = AutoFixerPipeline(
        project_root=args.project_root,
        enable_plugins=not args.disable_plugins,
        conservative=args.conservative,
        balanced=args.balanced
    )
    
    if args.fix_type == "all":
        results = pipeline.run_all_auto_fixes()
    elif args.fix_type == "imports":
        results = pipeline.run_import_fixes()
    elif args.fix_type == "auto_detection":
        results = pipeline.run_enhanced_auto_detection(
            file_pattern=args.file_pattern,
            dry_run=args.dry_run
        )
    elif args.fix_type == "syntax":
        results = pipeline.run_syntax_fixes()
    elif args.fix_type == "type_hints":
        results = pipeline.run_type_hint_fixes()
    elif args.fix_type == "async":
        results = pipeline.run_async_fixes()
    elif args.fix_type == "dead_code":
        results = pipeline.run_dead_code_fixes()
    elif args.fix_type == "plugins":
        results = pipeline.run_plugin_fixes()
    elif args.fix_type == "sequential":
        results = pipeline.run_sequential_fixes()
    
    print(f"\nAuto-fixer pipeline completed with status: {results.get('status', 'unknown')}")


if __name__ == "__main__":
    main()