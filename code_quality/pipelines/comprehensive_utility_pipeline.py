#!/usr/bin/env python3
"""
Comprehensive Utility Pipeline

This pipeline integrates all utility scripts from the scripts/ directory and provides
comprehensive utility capabilities for the code quality system. It includes:

1. Type hints addition and enhancement
2. Advanced syntax fixing
3. Apply all fixes
4. Bulk syntax cleanup
5. Circular imports detection
6. Interaction extraction and summary
7. Final code fixes
8. Async/await fixing
9. Common syntax patterns fixing
10. Missing imports fixing
11. Simple interaction mapping
12. Master code quality management
13. Robust async fixing

All utility operations are executed with proper error handling, reporting,
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
class UtilityResult:
    """Result of a utility operation."""
    utility_name: str
    status: str  # 'success', 'failed', 'skipped', 'error'
    duration: float
    files_processed: int
    operations_performed: int
    output: str
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UtilityConfig(PipelineConfig):
    """Configuration for utility pipeline."""
    utility_timeout: int = 300
    parallel_utilities: bool = True
    max_utility_workers: int = 4
    include_type_hints: bool = True
    include_syntax_fixing: bool = True
    include_import_fixing: bool = True
    include_async_fixing: bool = True
    include_interaction_mapping: bool = True
    include_code_cleanup: bool = True
    include_circular_imports: bool = True
    include_master_quality: bool = True
    dry_run: bool = False
    backup_files: bool = True
    verbose_output: bool = False


class ComprehensiveUtilityPipeline(BasePipeline):
    """Comprehensive utility pipeline with all utility script integration."""
    
    def __init__(self, config: UtilityConfig):
        super().__init__(config)
        self.config = config
        self.utility_results: List[UtilityResult] = []
        self.utility_scripts = self._discover_utility_scripts()
        
    def _discover_utility_scripts(self) -> Dict[str, str]:
        """Discover all utility scripts in the scripts/ directory."""
        utility_scripts = {}
        scripts_dir = Path(__file__).parent.parent / "scripts"
        
        # Main utility scripts
        main_utilities = {
            'add_type_hints': 'add_type_hints.py',
            'advanced_syntax_fixer': 'advanced_syntax_fixer.py',
            'apply_all_fixes': 'apply_all_fixes.py',
            'bulk_syntax_cleanup': 'bulk_syntax_cleanup.py',
            'detect_circular_imports': 'detect_circular_imports.py',
            'enhanced_type_hints': 'enhanced_type_hints.py',
            'extract_interactions': 'extract_interactions.py',
            'final_code_fixes': 'final_code_fixes.py',
            'fix_async_await': 'fix_async_await.py',
            'fix_common_syntax_patterns': 'fix_common_syntax_patterns.py',
            'fix_missing_imports': 'fix_missing_imports.py',
            'interaction_summary': 'interaction_summary.py',
            'master_code_quality': 'master_code_quality.py',
            'robust_async_fixer': 'robust_async_fixer.py',
            'simple_interaction_mapper': 'simple_interaction_mapper.py',
        }
        
        # Verify scripts exist and add to utility_scripts
        for utility_name, script_name in main_utilities.items():
            script_path = scripts_dir / script_name
            if script_path.exists():
                utility_scripts[utility_name] = str(script_path)
            else:
                self.logger.warning(f"Utility script not found: {script_path}")
        
        return utility_scripts
    
    def run_utility_script(self, utility_name: str, script_path: str, args: List[str] = None) -> UtilityResult:
        """Run a single utility script and return the result."""
        start_time = time.time()
        
        try:
            self.logger.info(f"Running utility: {utility_name}")
            
            # Prepare command
            cmd = [sys.executable, script_path]
            if args:
                cmd.extend(args)
            
            # Add common arguments
            if self.config.dry_run:
                cmd.append('--dry-run')
            if self.config.verbose_output:
                cmd.append('--verbose')
            
            # Run the utility script
            result = subprocess.run(
                cmd,
                cwd=str(Path(__file__).parent.parent),
                capture_output=True,
                text=True,
                timeout=self.config.utility_timeout
            )
            
            duration = time.time() - start_time
            
            # Parse output for statistics
            files_processed = 0
            operations_performed = 0
            
            # Try to extract statistics from output
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if 'files processed' in line.lower():
                    try:
                        files_processed = int(line.split()[0])
                    except (ValueError, IndexError):
                        pass
                elif 'operations performed' in line.lower() or 'fixes applied' in line.lower():
                    try:
                        operations_performed = int(line.split()[0])
                    except (ValueError, IndexError):
                        pass
            
            if result.returncode == 0:
                status = 'success'
                error = None
            else:
                status = 'failed'
                error = result.stderr
            
            return UtilityResult(
                utility_name=utility_name,
                status=status,
                duration=duration,
                files_processed=files_processed,
                operations_performed=operations_performed,
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
            return UtilityResult(
                utility_name=utility_name,
                status='error',
                duration=duration,
                files_processed=0,
                operations_performed=0,
                output='',
                error=f'Utility timed out after {self.config.utility_timeout} seconds',
                details={'script_path': script_path}
            )
        except Exception as e:
            duration = time.time() - start_time
            return UtilityResult(
                utility_name=utility_name,
                status='error',
                duration=duration,
                files_processed=0,
                operations_performed=0,
                output='',
                error=str(e),
                details={'script_path': script_path}
            )
    
    def add_type_hints(self) -> UtilityResult:
        """Add type hints to code."""
        if not self.config.include_type_hints:
            return UtilityResult(
                utility_name='add_type_hints',
                status='skipped',
                duration=0.0,
                files_processed=0,
                operations_performed=0,
                output='Type hints addition disabled',
                error=None
            )
        
        if 'add_type_hints' not in self.utility_scripts:
            return UtilityResult(
                utility_name='add_type_hints',
                status='skipped',
                duration=0.0,
                files_processed=0,
                operations_performed=0,
                output='Utility script not found',
                error='Script not available'
            )
        
        return self.run_utility_script(
            'add_type_hints',
            self.utility_scripts['add_type_hints']
        )
    
    def enhanced_type_hints(self) -> UtilityResult:
        """Enhanced type hints addition."""
        if not self.config.include_type_hints:
            return UtilityResult(
                utility_name='enhanced_type_hints',
                status='skipped',
                duration=0.0,
                files_processed=0,
                operations_performed=0,
                output='Type hints addition disabled',
                error=None
            )
        
        if 'enhanced_type_hints' not in self.utility_scripts:
            return UtilityResult(
                utility_name='enhanced_type_hints',
                status='skipped',
                duration=0.0,
                files_processed=0,
                operations_performed=0,
                output='Utility script not found',
                error='Script not available'
            )
        
        return self.run_utility_script(
            'enhanced_type_hints',
            self.utility_scripts['enhanced_type_hints']
        )
    
    def advanced_syntax_fixer(self) -> UtilityResult:
        """Advanced syntax fixing."""
        if not self.config.include_syntax_fixing:
            return UtilityResult(
                utility_name='advanced_syntax_fixer',
                status='skipped',
                duration=0.0,
                files_processed=0,
                operations_performed=0,
                output='Syntax fixing disabled',
                error=None
            )
        
        if 'advanced_syntax_fixer' not in self.utility_scripts:
            return UtilityResult(
                utility_name='advanced_syntax_fixer',
                status='skipped',
                duration=0.0,
                files_processed=0,
                operations_performed=0,
                output='Utility script not found',
                error='Script not available'
            )
        
        return self.run_utility_script(
            'advanced_syntax_fixer',
            self.utility_scripts['advanced_syntax_fixer']
        )
    
    def apply_all_fixes(self) -> UtilityResult:
        """Apply all fixes."""
        if 'apply_all_fixes' not in self.utility_scripts:
            return UtilityResult(
                utility_name='apply_all_fixes',
                status='skipped',
                duration=0.0,
                files_processed=0,
                operations_performed=0,
                output='Utility script not found',
                error='Script not available'
            )
        
        return self.run_utility_script(
            'apply_all_fixes',
            self.utility_scripts['apply_all_fixes']
        )
    
    def bulk_syntax_cleanup(self) -> UtilityResult:
        """Bulk syntax cleanup."""
        if not self.config.include_code_cleanup:
            return UtilityResult(
                utility_name='bulk_syntax_cleanup',
                status='skipped',
                duration=0.0,
                files_processed=0,
                operations_performed=0,
                output='Code cleanup disabled',
                error=None
            )
        
        if 'bulk_syntax_cleanup' not in self.utility_scripts:
            return UtilityResult(
                utility_name='bulk_syntax_cleanup',
                status='skipped',
                duration=0.0,
                files_processed=0,
                operations_performed=0,
                output='Utility script not found',
                error='Script not available'
            )
        
        return self.run_utility_script(
            'bulk_syntax_cleanup',
            self.utility_scripts['bulk_syntax_cleanup']
        )
    
    def detect_circular_imports(self) -> UtilityResult:
        """Detect circular imports."""
        if not self.config.include_circular_imports:
            return UtilityResult(
                utility_name='detect_circular_imports',
                status='skipped',
                duration=0.0,
                files_processed=0,
                operations_performed=0,
                output='Circular imports detection disabled',
                error=None
            )
        
        if 'detect_circular_imports' not in self.utility_scripts:
            return UtilityResult(
                utility_name='detect_circular_imports',
                status='skipped',
                duration=0.0,
                files_processed=0,
                operations_performed=0,
                output='Utility script not found',
                error='Script not available'
            )
        
        return self.run_utility_script(
            'detect_circular_imports',
            self.utility_scripts['detect_circular_imports']
        )
    
    def extract_interactions(self) -> UtilityResult:
        """Extract interactions."""
        if not self.config.include_interaction_mapping:
            return UtilityResult(
                utility_name='extract_interactions',
                status='skipped',
                duration=0.0,
                files_processed=0,
                operations_performed=0,
                output='Interaction mapping disabled',
                error=None
            )
        
        if 'extract_interactions' not in self.utility_scripts:
            return UtilityResult(
                utility_name='extract_interactions',
                status='skipped',
                duration=0.0,
                files_processed=0,
                operations_performed=0,
                output='Utility script not found',
                error='Script not available'
            )
        
        return self.run_utility_script(
            'extract_interactions',
            self.utility_scripts['extract_interactions']
        )
    
    def final_code_fixes(self) -> UtilityResult:
        """Final code fixes."""
        if 'final_code_fixes' not in self.utility_scripts:
            return UtilityResult(
                utility_name='final_code_fixes',
                status='skipped',
                duration=0.0,
                files_processed=0,
                operations_performed=0,
                output='Utility script not found',
                error='Script not available'
            )
        
        return self.run_utility_script(
            'final_code_fixes',
            self.utility_scripts['final_code_fixes']
        )
    
    def fix_async_await(self) -> UtilityResult:
        """Fix async/await issues."""
        if not self.config.include_async_fixing:
            return UtilityResult(
                utility_name='fix_async_await',
                status='skipped',
                duration=0.0,
                files_processed=0,
                operations_performed=0,
                output='Async fixing disabled',
                error=None
            )
        
        if 'fix_async_await' not in self.utility_scripts:
            return UtilityResult(
                utility_name='fix_async_await',
                status='skipped',
                duration=0.0,
                files_processed=0,
                operations_performed=0,
                output='Utility script not found',
                error='Script not available'
            )
        
        return self.run_utility_script(
            'fix_async_await',
            self.utility_scripts['fix_async_await']
        )
    
    def robust_async_fixer(self) -> UtilityResult:
        """Robust async fixing."""
        if not self.config.include_async_fixing:
            return UtilityResult(
                utility_name='robust_async_fixer',
                status='skipped',
                duration=0.0,
                files_processed=0,
                operations_performed=0,
                output='Async fixing disabled',
                error=None
            )
        
        if 'robust_async_fixer' not in self.utility_scripts:
            return UtilityResult(
                utility_name='robust_async_fixer',
                status='skipped',
                duration=0.0,
                files_processed=0,
                operations_performed=0,
                output='Utility script not found',
                error='Script not available'
            )
        
        return self.run_utility_script(
            'robust_async_fixer',
            self.utility_scripts['robust_async_fixer']
        )
    
    def fix_common_syntax_patterns(self) -> UtilityResult:
        """Fix common syntax patterns."""
        if not self.config.include_syntax_fixing:
            return UtilityResult(
                utility_name='fix_common_syntax_patterns',
                status='skipped',
                duration=0.0,
                files_processed=0,
                operations_performed=0,
                output='Syntax fixing disabled',
                error=None
            )
        
        if 'fix_common_syntax_patterns' not in self.utility_scripts:
            return UtilityResult(
                utility_name='fix_common_syntax_patterns',
                status='skipped',
                duration=0.0,
                files_processed=0,
                operations_performed=0,
                output='Utility script not found',
                error='Script not available'
            )
        
        return self.run_utility_script(
            'fix_common_syntax_patterns',
            self.utility_scripts['fix_common_syntax_patterns']
        )
    
    def fix_missing_imports(self) -> UtilityResult:
        """Fix missing imports."""
        if not self.config.include_import_fixing:
            return UtilityResult(
                utility_name='fix_missing_imports',
                status='skipped',
                duration=0.0,
                files_processed=0,
                operations_performed=0,
                output='Import fixing disabled',
                error=None
            )
        
        if 'fix_missing_imports' not in self.utility_scripts:
            return UtilityResult(
                utility_name='fix_missing_imports',
                status='skipped',
                duration=0.0,
                files_processed=0,
                operations_performed=0,
                output='Utility script not found',
                error='Script not available'
            )
        
        return self.run_utility_script(
            'fix_missing_imports',
            self.utility_scripts['fix_missing_imports']
        )
    
    def interaction_summary(self) -> UtilityResult:
        """Generate interaction summary."""
        if not self.config.include_interaction_mapping:
            return UtilityResult(
                utility_name='interaction_summary',
                status='skipped',
                duration=0.0,
                files_processed=0,
                operations_performed=0,
                output='Interaction mapping disabled',
                error=None
            )
        
        if 'interaction_summary' not in self.utility_scripts:
            return UtilityResult(
                utility_name='interaction_summary',
                status='skipped',
                duration=0.0,
                files_processed=0,
                operations_performed=0,
                output='Utility script not found',
                error='Script not available'
            )
        
        return self.run_utility_script(
            'interaction_summary',
            self.utility_scripts['interaction_summary']
        )
    
    def master_code_quality(self) -> UtilityResult:
        """Master code quality management."""
        if not self.config.include_master_quality:
            return UtilityResult(
                utility_name='master_code_quality',
                status='skipped',
                duration=0.0,
                files_processed=0,
                operations_performed=0,
                output='Master code quality disabled',
                error=None
            )
        
        if 'master_code_quality' not in self.utility_scripts:
            return UtilityResult(
                utility_name='master_code_quality',
                status='skipped',
                duration=0.0,
                files_processed=0,
                operations_performed=0,
                output='Utility script not found',
                error='Script not available'
            )
        
        return self.run_utility_script(
            'master_code_quality',
            self.utility_scripts['master_code_quality']
        )
    
    def simple_interaction_mapper(self) -> UtilityResult:
        """Simple interaction mapping."""
        if not self.config.include_interaction_mapping:
            return UtilityResult(
                utility_name='simple_interaction_mapper',
                status='skipped',
                duration=0.0,
                files_processed=0,
                operations_performed=0,
                output='Interaction mapping disabled',
                error=None
            )
        
        if 'simple_interaction_mapper' not in self.utility_scripts:
            return UtilityResult(
                utility_name='simple_interaction_mapper',
                status='skipped',
                duration=0.0,
                files_processed=0,
                operations_performed=0,
                output='Utility script not found',
                error='Script not available'
            )
        
        return self.run_utility_script(
            'simple_interaction_mapper',
            self.utility_scripts['simple_interaction_mapper']
        )
    
    def run_all_utilities(self) -> Dict[str, Any]:
        """Run all available utilities."""
        start_time = time.time()
        
        self.logger.info("Starting comprehensive utility execution")
        
        # Run all utility methods
        utility_methods = [
            self.add_type_hints,
            self.enhanced_type_hints,
            self.advanced_syntax_fixer,
            self.apply_all_fixes,
            self.bulk_syntax_cleanup,
            self.detect_circular_imports,
            self.extract_interactions,
            self.final_code_fixes,
            self.fix_async_await,
            self.robust_async_fixer,
            self.fix_common_syntax_patterns,
            self.fix_missing_imports,
            self.interaction_summary,
            self.master_code_quality,
            self.simple_interaction_mapper,
        ]
        
        results = []
        for utility_method in utility_methods:
            try:
                result = utility_method()
                results.append(result)
                self.utility_results.append(result)
            except Exception as e:
                error_result = UtilityResult(
                    utility_name=utility_method.__name__,
                    status='error',
                    duration=0.0,
                    files_processed=0,
                    operations_performed=0,
                    output='',
                    error=str(e)
                )
                results.append(error_result)
                self.utility_results.append(error_result)
        
        # Calculate summary statistics
        total_utilities = len(results)
        successful_utilities = len([r for r in results if r.status == 'success'])
        failed_utilities = len([r for r in results if r.status == 'failed'])
        skipped_utilities = len([r for r in results if r.status == 'skipped'])
        error_utilities = len([r for r in results if r.status == 'error'])
        total_files_processed = sum(r.files_processed for r in results)
        total_operations_performed = sum(r.operations_performed for r in results)
        total_duration = sum(r.duration for r in results)
        
        summary = {
            'total_utilities': total_utilities,
            'successful_utilities': successful_utilities,
            'failed_utilities': failed_utilities,
            'skipped_utilities': skipped_utilities,
            'error_utilities': error_utilities,
            'success_rate': (successful_utilities / total_utilities * 100) if total_utilities > 0 else 0,
            'total_files_processed': total_files_processed,
            'total_operations_performed': total_operations_performed,
            'total_duration': total_duration,
            'execution_time': time.time() - start_time,
            'utility_results': [
                {
                    'utility_name': r.utility_name,
                    'status': r.status,
                    'duration': r.duration,
                    'files_processed': r.files_processed,
                    'operations_performed': r.operations_performed,
                    'error': r.error,
                    'details': r.details
                }
                for r in results
            ]
        }
        
        self.logger.info(f"Utility execution completed: {successful_utilities}/{total_utilities} successful")
        
        return summary
    
    def execute(self) -> Dict[str, Any]:
        """Execute the comprehensive utility pipeline."""
        start_time = time.time()
        
        self.logger.info("Starting Comprehensive Utility Pipeline")
        
        # Run all utilities
        utility_summary = self.run_all_utilities()
        
        # Execute plugins if available
        plugin_results = {}
        if self.plugin_manager:
            try:
                context = PluginContext(
                    project_root=self.config.project_root,
                    output_dir=self.config.output_dir,
                    utility_results=self.utility_results
                )
                plugin_results = self.plugin_manager.execute_pipeline(
                    "comprehensive_utility_pipeline",
                    context
                )
            except Exception as e:
                self.logger.warning(f"Plugin execution failed: {e}")
        
        # Generate final results
        results = {
            'pipeline_name': 'comprehensive_utility_pipeline',
            'execution_time': time.time() - start_time,
            'utility_summary': utility_summary,
            'plugin_results': plugin_results,
            'configuration': {
                'utility_timeout': self.config.utility_timeout,
                'parallel_utilities': self.config.parallel_utilities,
                'max_utility_workers': self.config.max_utility_workers,
                'include_type_hints': self.config.include_type_hints,
                'include_syntax_fixing': self.config.include_syntax_fixing,
                'include_import_fixing': self.config.include_import_fixing,
                'include_async_fixing': self.config.include_async_fixing,
                'include_interaction_mapping': self.config.include_interaction_mapping,
                'include_code_cleanup': self.config.include_code_cleanup,
                'include_circular_imports': self.config.include_circular_imports,
                'include_master_quality': self.config.include_master_quality,
                'dry_run': self.config.dry_run,
                'backup_files': self.config.backup_files,
                'verbose_output': self.config.verbose_output
            }
        }
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save utility results to output directory."""
        output_file = self.config.output_dir / f"comprehensive_utility_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Utility results saved to: {output_file}")


def main():
    """Main entry point for the comprehensive utility pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Utility Pipeline")
    parser.add_argument("--project-root", type=str, default="/workspace", help="Project root directory")
    parser.add_argument("--output-dir", type=str, default="/workspace/code_quality/reports", help="Output directory")
    parser.add_argument("--utility-timeout", type=int, default=300, help="Utility timeout in seconds")
    parser.add_argument("--parallel-utilities", action="store_true", help="Enable parallel utility execution")
    parser.add_argument("--max-utility-workers", type=int, default=4, help="Maximum utility workers")
    parser.add_argument("--include-type-hints", action="store_true", help="Include type hints utilities")
    parser.add_argument("--include-syntax-fixing", action="store_true", help="Include syntax fixing utilities")
    parser.add_argument("--include-import-fixing", action="store_true", help="Include import fixing utilities")
    parser.add_argument("--include-async-fixing", action="store_true", help="Include async fixing utilities")
    parser.add_argument("--include-interaction-mapping", action="store_true", help="Include interaction mapping utilities")
    parser.add_argument("--include-code-cleanup", action="store_true", help="Include code cleanup utilities")
    parser.add_argument("--include-circular-imports", action="store_true", help="Include circular imports utilities")
    parser.add_argument("--include-master-quality", action="store_true", help="Include master quality utilities")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--backup-files", action="store_true", help="Backup files before processing")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Create configuration
    config = UtilityConfig(
        project_root=Path(args.project_root),
        output_dir=Path(args.output_dir),
        utility_timeout=args.utility_timeout,
        parallel_utilities=args.parallel_utilities,
        max_utility_workers=args.max_utility_workers,
        include_type_hints=args.include_type_hints,
        include_syntax_fixing=args.include_syntax_fixing,
        include_import_fixing=args.include_import_fixing,
        include_async_fixing=args.include_async_fixing,
        include_interaction_mapping=args.include_interaction_mapping,
        include_code_cleanup=args.include_code_cleanup,
        include_circular_imports=args.include_circular_imports,
        include_master_quality=args.include_master_quality,
        dry_run=args.dry_run,
        backup_files=args.backup_files,
        verbose_output=args.verbose
    )
    
    # Create and run pipeline
    pipeline = ComprehensiveUtilityPipeline(config)
    results = pipeline.execute()
    
    # Print summary
    utility_summary = results['utility_summary']
    print(f"\n{'='*60}")
    print("COMPREHENSIVE UTILITY PIPELINE RESULTS")
    print(f"{'='*60}")
    print(f"Total Utilities: {utility_summary['total_utilities']}")
    print(f"Successful: {utility_summary['successful_utilities']}")
    print(f"Failed: {utility_summary['failed_utilities']}")
    print(f"Skipped: {utility_summary['skipped_utilities']}")
    print(f"Errors: {utility_summary['error_utilities']}")
    print(f"Success Rate: {utility_summary['success_rate']:.1f}%")
    print(f"Files Processed: {utility_summary['total_files_processed']}")
    print(f"Operations Performed: {utility_summary['total_operations_performed']}")
    print(f"Total Duration: {utility_summary['total_duration']:.2f}s")
    print(f"Execution Time: {utility_summary['execution_time']:.2f}s")
    print(f"{'='*60}")
    
    # Print individual utility results
    if args.verbose:
        print("\nIndividual Utility Results:")
        for result in utility_summary['utility_results']:
            status_icon = "✅" if result['status'] == 'success' else "❌" if result['status'] == 'failed' else "⏭️" if result['status'] == 'skipped' else "⚠️"
            print(f"  {status_icon} {result['utility_name']}: {result['status']} ({result['duration']:.2f}s)")
            print(f"    Files: {result['files_processed']}, Operations: {result['operations_performed']}")
            if result['error']:
                print(f"    Error: {result['error']}")


if __name__ == "__main__":
    main()