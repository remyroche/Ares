#!/usr/bin/env python3
"""
Enhanced Import Fixing Pipeline

This pipeline integrates all import fixing scripts and provides comprehensive import
fixing capabilities for the code quality system. It includes:

1. Common imports fixing
2. Common undefined names fixing
3. Import issues fixing
4. Missing imports fixing (multiple variants)
5. Parameter undefined names fixing
6. Remaining imports fixing
7. Simple undefined names fixing
8. Top undefined names fixing
9. Undefined names fixing
10. Comprehensive import fixing
11. Targeted import fixing
12. Intelligent import fixing

All import fixing operations are executed with proper error handling, reporting,
and integration with the plugin system.
"""

import ast
import json
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from base_pipeline import BasePipeline, PipelineConfig
from plugins import PluginManager, PluginContext, PluginResult


@dataclass
class ImportFixResult:
    """Result of an import fixing operation."""
    fixer_name: str
    status: str  # 'success', 'failed', 'skipped', 'error'
    duration: float
    files_processed: int
    fixes_applied: int
    errors_fixed: int
    output: str
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImportFixingConfig(PipelineConfig):
    """Configuration for import fixing pipeline."""
    fix_timeout: int = 600
    parallel_fixes: bool = True
    max_fix_workers: int = 4
    include_common_imports: bool = True
    include_undefined_names: bool = True
    include_missing_imports: bool = True
    include_remaining_imports: bool = True
    include_comprehensive_fixes: bool = True
    include_targeted_fixes: bool = True
    include_intelligent_fixes: bool = True
    dry_run: bool = False
    backup_files: bool = True
    verbose_output: bool = False


class EnhancedImportFixingPipeline(BasePipeline):
    """Enhanced import fixing pipeline with comprehensive import fixing capabilities."""
    
    def __init__(self, config: ImportFixingConfig):
        super().__init__(config)
        self.config = config
        self.fix_results: List[ImportFixResult] = []
        self.import_fixers = self._discover_import_fixers()
        
    def _discover_import_fixers(self) -> Dict[str, str]:
        """Discover all import fixing scripts in the code_quality directory."""
        import_fixers = {}
        code_quality_dir = Path(__file__).parent.parent
        
        # Main import fixing scripts
        main_fixers = {
            'fix_common_imports_final': 'fix_common_imports_final.py',
            'fix_common_undefined_names': 'fix_common_undefined_names.py',
            'fix_import_issues': 'fix_import_issues.py',
            'fix_missing_imports_only': 'fix_missing_imports_only.py',
            'fix_missing_imports_targeted': 'fix_missing_imports_targeted.py',
            'fix_parameter_undefined_names': 'fix_parameter_undefined_names.py',
            'fix_remaining_imports': 'fix_remaining_imports.py',
            'fix_remaining_imports_final': 'fix_remaining_imports_final.py',
            'fix_simple_undefined_names': 'fix_simple_undefined_names.py',
            'fix_top_undefined_names': 'fix_top_undefined_names.py',
            'fix_undefined_names': 'fix_undefined_names.py',
            'comprehensive_import_fixer': 'comprehensive_import_fixer.py',
            'targeted_import_fixer': 'targeted_import_fixer.py',
        }
        
        # Verify scripts exist and add to import_fixers
        for fixer_name, script_name in main_fixers.items():
            script_path = code_quality_dir / script_name
            if script_path.exists():
                import_fixers[fixer_name] = str(script_path)
            else:
                self.logger.warning(f"Import fixer script not found: {script_path}")
        
        return import_fixers
    
    def run_import_fixer(self, fixer_name: str, script_path: str, args: List[str] = None) -> ImportFixResult:
        """Run a single import fixing script and return the result."""
        start_time = time.time()
        
        try:
            self.logger.info(f"Running import fixer: {fixer_name}")
            
            # Prepare command
            cmd = [sys.executable, script_path]
            if args:
                cmd.extend(args)
            
            # Add common arguments
            if self.config.dry_run:
                cmd.append('--dry-run')
            if self.config.verbose_output:
                cmd.append('--verbose')
            
            # Run the import fixing script
            result = subprocess.run(
                cmd,
                cwd=str(Path(__file__).parent.parent),
                capture_output=True,
                text=True,
                timeout=self.config.fix_timeout
            )
            
            duration = time.time() - start_time
            
            # Parse output for statistics
            files_processed = 0
            fixes_applied = 0
            errors_fixed = 0
            
            # Try to extract statistics from output
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if 'files processed' in line.lower():
                    try:
                        files_processed = int(line.split()[0])
                    except (ValueError, IndexError):
                        pass
                elif 'fixes applied' in line.lower():
                    try:
                        fixes_applied = int(line.split()[0])
                    except (ValueError, IndexError):
                        pass
                elif 'errors fixed' in line.lower():
                    try:
                        errors_fixed = int(line.split()[0])
                    except (ValueError, IndexError):
                        pass
            
            if result.returncode == 0:
                status = 'success'
                error = None
            else:
                status = 'failed'
                error = result.stderr
            
            return ImportFixResult(
                fixer_name=fixer_name,
                status=status,
                duration=duration,
                files_processed=files_processed,
                fixes_applied=fixes_applied,
                errors_fixed=errors_fixed,
                output=result.stdout,
                error=error,
                details={
                    'returncode': result.returncode,
                    'script_path': script_path,
                    'command': ' '.join(cmd)
                }
            )
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return ImportFixResult(
                fixer_name=fixer_name,
                status='error',
                duration=duration,
                files_processed=0,
                fixes_applied=0,
                errors_fixed=0,
                output='',
                error=f'Import fixer timed out after {self.config.fix_timeout} seconds',
                details={'script_path': script_path}
            )
        except Exception as e:
            duration = time.time() - start_time
            return ImportFixResult(
                fixer_name=fixer_name,
                status='error',
                duration=duration,
                files_processed=0,
                fixes_applied=0,
                errors_fixed=0,
                output='',
                error=str(e),
                details={'script_path': script_path}
            )
    
    def fix_common_imports(self) -> ImportFixResult:
        """Fix common imports."""
        if not self.config.include_common_imports:
            return ImportFixResult(
                fixer_name='fix_common_imports',
                status='skipped',
                duration=0.0,
                files_processed=0,
                fixes_applied=0,
                errors_fixed=0,
                output='Common imports fixing disabled',
                error=None
            )
        
        if 'fix_common_imports_final' not in self.import_fixers:
            return ImportFixResult(
                fixer_name='fix_common_imports',
                status='skipped',
                duration=0.0,
                files_processed=0,
                fixes_applied=0,
                errors_fixed=0,
                output='Import fixer script not found',
                error='Script not available'
            )
        
        return self.run_import_fixer(
            'fix_common_imports',
            self.import_fixers['fix_common_imports_final']
        )
    
    def fix_common_undefined_names(self) -> ImportFixResult:
        """Fix common undefined names."""
        if not self.config.include_undefined_names:
            return ImportFixResult(
                fixer_name='fix_common_undefined_names',
                status='skipped',
                duration=0.0,
                files_processed=0,
                fixes_applied=0,
                errors_fixed=0,
                output='Undefined names fixing disabled',
                error=None
            )
        
        if 'fix_common_undefined_names' not in self.import_fixers:
            return ImportFixResult(
                fixer_name='fix_common_undefined_names',
                status='skipped',
                duration=0.0,
                files_processed=0,
                fixes_applied=0,
                errors_fixed=0,
                output='Import fixer script not found',
                error='Script not available'
            )
        
        return self.run_import_fixer(
            'fix_common_undefined_names',
            self.import_fixers['fix_common_undefined_names']
        )
    
    def fix_import_issues(self) -> ImportFixResult:
        """Fix import issues."""
        if 'fix_import_issues' not in self.import_fixers:
            return ImportFixResult(
                fixer_name='fix_import_issues',
                status='skipped',
                duration=0.0,
                files_processed=0,
                fixes_applied=0,
                errors_fixed=0,
                output='Import fixer script not found',
                error='Script not available'
            )
        
        return self.run_import_fixer(
            'fix_import_issues',
            self.import_fixers['fix_import_issues']
        )
    
    def fix_missing_imports(self) -> ImportFixResult:
        """Fix missing imports."""
        if not self.config.include_missing_imports:
            return ImportFixResult(
                fixer_name='fix_missing_imports',
                status='skipped',
                duration=0.0,
                files_processed=0,
                fixes_applied=0,
                errors_fixed=0,
                output='Missing imports fixing disabled',
                error=None
            )
        
        if 'fix_missing_imports_only' not in self.import_fixers:
            return ImportFixResult(
                fixer_name='fix_missing_imports',
                status='skipped',
                duration=0.0,
                files_processed=0,
                fixes_applied=0,
                errors_fixed=0,
                output='Import fixer script not found',
                error='Script not available'
            )
        
        return self.run_import_fixer(
            'fix_missing_imports',
            self.import_fixers['fix_missing_imports_only']
        )
    
    def fix_missing_imports_targeted(self) -> ImportFixResult:
        """Fix missing imports with targeted approach."""
        if not self.config.include_missing_imports:
            return ImportFixResult(
                fixer_name='fix_missing_imports_targeted',
                status='skipped',
                duration=0.0,
                files_processed=0,
                fixes_applied=0,
                errors_fixed=0,
                output='Missing imports fixing disabled',
                error=None
            )
        
        if 'fix_missing_imports_targeted' not in self.import_fixers:
            return ImportFixResult(
                fixer_name='fix_missing_imports_targeted',
                status='skipped',
                duration=0.0,
                files_processed=0,
                fixes_applied=0,
                errors_fixed=0,
                output='Import fixer script not found',
                error='Script not available'
            )
        
        return self.run_import_fixer(
            'fix_missing_imports_targeted',
            self.import_fixers['fix_missing_imports_targeted']
        )
    
    def fix_parameter_undefined_names(self) -> ImportFixResult:
        """Fix parameter undefined names."""
        if not self.config.include_undefined_names:
            return ImportFixResult(
                fixer_name='fix_parameter_undefined_names',
                status='skipped',
                duration=0.0,
                files_processed=0,
                fixes_applied=0,
                errors_fixed=0,
                output='Undefined names fixing disabled',
                error=None
            )
        
        if 'fix_parameter_undefined_names' not in self.import_fixers:
            return ImportFixResult(
                fixer_name='fix_parameter_undefined_names',
                status='skipped',
                duration=0.0,
                files_processed=0,
                fixes_applied=0,
                errors_fixed=0,
                output='Import fixer script not found',
                error='Script not available'
            )
        
        return self.run_import_fixer(
            'fix_parameter_undefined_names',
            self.import_fixers['fix_parameter_undefined_names']
        )
    
    def fix_remaining_imports(self) -> ImportFixResult:
        """Fix remaining imports."""
        if not self.config.include_remaining_imports:
            return ImportFixResult(
                fixer_name='fix_remaining_imports',
                status='skipped',
                duration=0.0,
                files_processed=0,
                fixes_applied=0,
                errors_fixed=0,
                output='Remaining imports fixing disabled',
                error=None
            )
        
        if 'fix_remaining_imports' not in self.import_fixers:
            return ImportFixResult(
                fixer_name='fix_remaining_imports',
                status='skipped',
                duration=0.0,
                files_processed=0,
                fixes_applied=0,
                errors_fixed=0,
                output='Import fixer script not found',
                error='Script not available'
            )
        
        return self.run_import_fixer(
            'fix_remaining_imports',
            self.import_fixers['fix_remaining_imports']
        )
    
    def fix_remaining_imports_final(self) -> ImportFixResult:
        """Fix remaining imports (final pass)."""
        if not self.config.include_remaining_imports:
            return ImportFixResult(
                fixer_name='fix_remaining_imports_final',
                status='skipped',
                duration=0.0,
                files_processed=0,
                fixes_applied=0,
                errors_fixed=0,
                output='Remaining imports fixing disabled',
                error=None
            )
        
        if 'fix_remaining_imports_final' not in self.import_fixers:
            return ImportFixResult(
                fixer_name='fix_remaining_imports_final',
                status='skipped',
                duration=0.0,
                files_processed=0,
                fixes_applied=0,
                errors_fixed=0,
                output='Import fixer script not found',
                error='Script not available'
            )
        
        return self.run_import_fixer(
            'fix_remaining_imports_final',
            self.import_fixers['fix_remaining_imports_final']
        )
    
    def fix_simple_undefined_names(self) -> ImportFixResult:
        """Fix simple undefined names."""
        if not self.config.include_undefined_names:
            return ImportFixResult(
                fixer_name='fix_simple_undefined_names',
                status='skipped',
                duration=0.0,
                files_processed=0,
                fixes_applied=0,
                errors_fixed=0,
                output='Undefined names fixing disabled',
                error=None
            )
        
        if 'fix_simple_undefined_names' not in self.import_fixers:
            return ImportFixResult(
                fixer_name='fix_simple_undefined_names',
                status='skipped',
                duration=0.0,
                files_processed=0,
                fixes_applied=0,
                errors_fixed=0,
                output='Import fixer script not found',
                error='Script not available'
            )
        
        return self.run_import_fixer(
            'fix_simple_undefined_names',
            self.import_fixers['fix_simple_undefined_names']
        )
    
    def fix_top_undefined_names(self) -> ImportFixResult:
        """Fix top undefined names."""
        if not self.config.include_undefined_names:
            return ImportFixResult(
                fixer_name='fix_top_undefined_names',
                status='skipped',
                duration=0.0,
                files_processed=0,
                fixes_applied=0,
                errors_fixed=0,
                output='Undefined names fixing disabled',
                error=None
            )
        
        if 'fix_top_undefined_names' not in self.import_fixers:
            return ImportFixResult(
                fixer_name='fix_top_undefined_names',
                status='skipped',
                duration=0.0,
                files_processed=0,
                fixes_applied=0,
                errors_fixed=0,
                output='Import fixer script not found',
                error='Script not available'
            )
        
        return self.run_import_fixer(
            'fix_top_undefined_names',
            self.import_fixers['fix_top_undefined_names']
        )
    
    def fix_undefined_names(self) -> ImportFixResult:
        """Fix undefined names."""
        if not self.config.include_undefined_names:
            return ImportFixResult(
                fixer_name='fix_undefined_names',
                status='skipped',
                duration=0.0,
                files_processed=0,
                fixes_applied=0,
                errors_fixed=0,
                output='Undefined names fixing disabled',
                error=None
            )
        
        if 'fix_undefined_names' not in self.import_fixers:
            return ImportFixResult(
                fixer_name='fix_undefined_names',
                status='skipped',
                duration=0.0,
                files_processed=0,
                fixes_applied=0,
                errors_fixed=0,
                output='Import fixer script not found',
                error='Script not available'
            )
        
        return self.run_import_fixer(
            'fix_undefined_names',
            self.import_fixers['fix_undefined_names']
        )
    
    def comprehensive_import_fix(self) -> ImportFixResult:
        """Comprehensive import fixing."""
        if not self.config.include_comprehensive_fixes:
            return ImportFixResult(
                fixer_name='comprehensive_import_fix',
                status='skipped',
                duration=0.0,
                files_processed=0,
                fixes_applied=0,
                errors_fixed=0,
                output='Comprehensive fixes disabled',
                error=None
            )
        
        if 'comprehensive_import_fixer' not in self.import_fixers:
            return ImportFixResult(
                fixer_name='comprehensive_import_fix',
                status='skipped',
                duration=0.0,
                files_processed=0,
                fixes_applied=0,
                errors_fixed=0,
                output='Import fixer script not found',
                error='Script not available'
            )
        
        return self.run_import_fixer(
            'comprehensive_import_fix',
            self.import_fixers['comprehensive_import_fixer']
        )
    
    def targeted_import_fix(self) -> ImportFixResult:
        """Targeted import fixing."""
        if not self.config.include_targeted_fixes:
            return ImportFixResult(
                fixer_name='targeted_import_fix',
                status='skipped',
                duration=0.0,
                files_processed=0,
                fixes_applied=0,
                errors_fixed=0,
                output='Targeted fixes disabled',
                error=None
            )
        
        if 'targeted_import_fixer' not in self.import_fixers:
            return ImportFixResult(
                fixer_name='targeted_import_fix',
                status='skipped',
                duration=0.0,
                files_processed=0,
                fixes_applied=0,
                errors_fixed=0,
                output='Import fixer script not found',
                error='Script not available'
            )
        
        return self.run_import_fixer(
            'targeted_import_fix',
            self.import_fixers['targeted_import_fixer']
        )
    
    def run_all_import_fixes(self) -> Dict[str, Any]:
        """Run all available import fixes."""
        start_time = time.time()
        
        self.logger.info("Starting comprehensive import fixing")
        
        # Run all import fixing methods
        fix_methods = [
            self.fix_common_imports,
            self.fix_common_undefined_names,
            self.fix_import_issues,
            self.fix_missing_imports,
            self.fix_missing_imports_targeted,
            self.fix_parameter_undefined_names,
            self.fix_remaining_imports,
            self.fix_remaining_imports_final,
            self.fix_simple_undefined_names,
            self.fix_top_undefined_names,
            self.fix_undefined_names,
            self.comprehensive_import_fix,
            self.targeted_import_fix,
        ]
        
        results = []
        for fix_method in fix_methods:
            try:
                result = fix_method()
                results.append(result)
                self.fix_results.append(result)
            except Exception as e:
                error_result = ImportFixResult(
                    fixer_name=fix_method.__name__,
                    status='error',
                    duration=0.0,
                    files_processed=0,
                    fixes_applied=0,
                    errors_fixed=0,
                    output='',
                    error=str(e)
                )
                results.append(error_result)
                self.fix_results.append(error_result)
        
        # Calculate summary statistics
        total_fixers = len(results)
        successful_fixers = len([r for r in results if r.status == 'success'])
        failed_fixers = len([r for r in results if r.status == 'failed'])
        skipped_fixers = len([r for r in results if r.status == 'skipped'])
        error_fixers = len([r for r in results if r.status == 'error'])
        total_files_processed = sum(r.files_processed for r in results)
        total_fixes_applied = sum(r.fixes_applied for r in results)
        total_errors_fixed = sum(r.errors_fixed for r in results)
        total_duration = sum(r.duration for r in results)
        
        summary = {
            'total_fixers': total_fixers,
            'successful_fixers': successful_fixers,
            'failed_fixers': failed_fixers,
            'skipped_fixers': skipped_fixers,
            'error_fixers': error_fixers,
            'success_rate': (successful_fixers / total_fixers * 100) if total_fixers > 0 else 0,
            'total_files_processed': total_files_processed,
            'total_fixes_applied': total_fixes_applied,
            'total_errors_fixed': total_errors_fixed,
            'total_duration': total_duration,
            'execution_time': time.time() - start_time,
            'fix_results': [
                {
                    'fixer_name': r.fixer_name,
                    'status': r.status,
                    'duration': r.duration,
                    'files_processed': r.files_processed,
                    'fixes_applied': r.fixes_applied,
                    'errors_fixed': r.errors_fixed,
                    'error': r.error,
                    'details': r.details
                }
                for r in results
            ]
        }
        
        self.logger.info(f"Import fixing completed: {successful_fixers}/{total_fixers} successful")
        
        return summary
    
    def execute(self) -> Dict[str, Any]:
        """Execute the enhanced import fixing pipeline."""
        start_time = time.time()
        
        self.logger.info("Starting Enhanced Import Fixing Pipeline")
        
        # Run all import fixes
        fix_summary = self.run_all_import_fixes()
        
        # Execute plugins if available
        plugin_results = {}
        if self.plugin_manager:
            try:
                context = PluginContext(
                    project_root=self.config.project_root,
                    output_dir=self.config.output_dir,
                    fix_results=self.fix_results
                )
                plugin_results = self.plugin_manager.execute_pipeline(
                    "enhanced_import_fixing_pipeline",
                    context
                )
            except Exception as e:
                self.logger.warning(f"Plugin execution failed: {e}")
        
        # Generate final results
        results = {
            'pipeline_name': 'enhanced_import_fixing_pipeline',
            'execution_time': time.time() - start_time,
            'fix_summary': fix_summary,
            'plugin_results': plugin_results,
            'configuration': {
                'fix_timeout': self.config.fix_timeout,
                'parallel_fixes': self.config.parallel_fixes,
                'max_fix_workers': self.config.max_fix_workers,
                'include_common_imports': self.config.include_common_imports,
                'include_undefined_names': self.config.include_undefined_names,
                'include_missing_imports': self.config.include_missing_imports,
                'include_remaining_imports': self.config.include_remaining_imports,
                'include_comprehensive_fixes': self.config.include_comprehensive_fixes,
                'include_targeted_fixes': self.config.include_targeted_fixes,
                'include_intelligent_fixes': self.config.include_intelligent_fixes,
                'dry_run': self.config.dry_run,
                'backup_files': self.config.backup_files,
                'verbose_output': self.config.verbose_output
            }
        }
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save import fixing results to output directory."""
        output_file = self.config.output_dir / f"enhanced_import_fixing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Import fixing results saved to: {output_file}")


def main():
    """Main entry point for the enhanced import fixing pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Import Fixing Pipeline")
    parser.add_argument("--project-root", type=str, default="/workspace", help="Project root directory")
    parser.add_argument("--output-dir", type=str, default="/workspace/code_quality/reports", help="Output directory")
    parser.add_argument("--fix-timeout", type=int, default=600, help="Fix timeout in seconds")
    parser.add_argument("--parallel-fixes", action="store_true", help="Enable parallel fix execution")
    parser.add_argument("--max-fix-workers", type=int, default=4, help="Maximum fix workers")
    parser.add_argument("--include-common-imports", action="store_true", help="Include common imports fixing")
    parser.add_argument("--include-undefined-names", action="store_true", help="Include undefined names fixing")
    parser.add_argument("--include-missing-imports", action="store_true", help="Include missing imports fixing")
    parser.add_argument("--include-remaining-imports", action="store_true", help="Include remaining imports fixing")
    parser.add_argument("--include-comprehensive-fixes", action="store_true", help="Include comprehensive fixes")
    parser.add_argument("--include-targeted-fixes", action="store_true", help="Include targeted fixes")
    parser.add_argument("--include-intelligent-fixes", action="store_true", help="Include intelligent fixes")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--backup-files", action="store_true", help="Backup files before fixing")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Create configuration
    config = ImportFixingConfig(
        project_root=Path(args.project_root),
        output_dir=Path(args.output_dir),
        fix_timeout=args.fix_timeout,
        parallel_fixes=args.parallel_fixes,
        max_fix_workers=args.max_fix_workers,
        include_common_imports=args.include_common_imports,
        include_undefined_names=args.include_undefined_names,
        include_missing_imports=args.include_missing_imports,
        include_remaining_imports=args.include_remaining_imports,
        include_comprehensive_fixes=args.include_comprehensive_fixes,
        include_targeted_fixes=args.include_targeted_fixes,
        include_intelligent_fixes=args.include_intelligent_fixes,
        dry_run=args.dry_run,
        backup_files=args.backup_files,
        verbose_output=args.verbose
    )
    
    # Create and run pipeline
    pipeline = EnhancedImportFixingPipeline(config)
    results = pipeline.execute()
    
    # Print summary
    fix_summary = results['fix_summary']
    print(f"\n{'='*60}")
    print("ENHANCED IMPORT FIXING PIPELINE RESULTS")
    print(f"{'='*60}")
    print(f"Total Fixers: {fix_summary['total_fixers']}")
    print(f"Successful: {fix_summary['successful_fixers']}")
    print(f"Failed: {fix_summary['failed_fixers']}")
    print(f"Skipped: {fix_summary['skipped_fixers']}")
    print(f"Errors: {fix_summary['error_fixers']}")
    print(f"Success Rate: {fix_summary['success_rate']:.1f}%")
    print(f"Files Processed: {fix_summary['total_files_processed']}")
    print(f"Fixes Applied: {fix_summary['total_fixes_applied']}")
    print(f"Errors Fixed: {fix_summary['total_errors_fixed']}")
    print(f"Total Duration: {fix_summary['total_duration']:.2f}s")
    print(f"Execution Time: {fix_summary['execution_time']:.2f}s")
    print(f"{'='*60}")
    
    # Print individual fixer results
    if args.verbose:
        print("\nIndividual Fixer Results:")
        for result in fix_summary['fix_results']:
            status_icon = "✅" if result['status'] == 'success' else "❌" if result['status'] == 'failed' else "⏭️" if result['status'] == 'skipped' else "⚠️"
            print(f"  {status_icon} {result['fixer_name']}: {result['status']} ({result['duration']:.2f}s)")
            print(f"    Files: {result['files_processed']}, Fixes: {result['fixes_applied']}, Errors: {result['errors_fixed']}")
            if result['error']:
                print(f"    Error: {result['error']}")


if __name__ == "__main__":
    main()