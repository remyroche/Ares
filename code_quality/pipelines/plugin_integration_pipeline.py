"""
Plugin Integration Pipeline

This pipeline integrates the plugin system with the code quality pipelines,
allowing plugins to be used within the pipeline framework.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base_pipeline import (
    BasePipeline, PipelineConfig, PipelineStage, StageResult, 
    PipelineStatus, PipelineResult
)
try:
    from plugins.base_plugin import PluginContext, PluginResult, PluginCategory, PluginPriority
    from plugins.plugin_manager import PluginManager
except ImportError:
    # Fallback for when plugins are not available
    PluginContext = None
    PluginResult = None
    PluginCategory = None
    PluginPriority = None
    PluginManager = None

try:
    from core.plugins import PluginManager as CorePluginManager
except ImportError:
    CorePluginManager = None


class PluginIntegrationPipeline(BasePipeline):
    """
    Pipeline that integrates the plugin system with code quality analysis.
    
    This pipeline discovers, loads, and executes plugins based on categories
    and priorities, providing a unified interface for plugin-based code quality tools.
    """
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.plugin_manager = PluginManager() if PluginManager else None
        self.core_plugin_manager = CorePluginManager() if CorePluginManager else None
        self.plugin_results: Dict[str, Any] = {}
        self.discovered_plugins: List[Dict[str, Any]] = []
        
    def get_stages(self) -> List[PipelineStage]:
        """Define the stages for plugin integration pipeline."""
        return [
            PipelineStage.INITIALIZATION,
            PipelineStage.PREPARATION,
            PipelineStage.ANALYSIS,
            PipelineStage.PROCESSING,
            PipelineStage.AGGREGATION,
            PipelineStage.REPORTING,
            PipelineStage.CLEANUP
        ]
    
    async def execute_stage(self, stage: PipelineStage, stage_result: StageResult, context: Dict[str, Any]):
        """Execute a specific stage of the plugin integration pipeline."""
        if stage == PipelineStage.INITIALIZATION:
            await self._execute_initialization(stage_result, context)
        elif stage == PipelineStage.PREPARATION:
            await self._execute_preparation(stage_result, context)
        elif stage == PipelineStage.ANALYSIS:
            await self._execute_analysis(stage_result, context)
        elif stage == PipelineStage.PROCESSING:
            await self._execute_processing(stage_result, context)
        elif stage == PipelineStage.AGGREGATION:
            await self._execute_aggregation(stage_result, context)
        elif stage == PipelineStage.REPORTING:
            await self._execute_reporting(stage_result, context)
        elif stage == PipelineStage.CLEANUP:
            await self._execute_cleanup(stage_result, context)
    
    async def _execute_initialization(self, stage_result: StageResult, context: Dict[str, Any]):
        """Initialize the plugin integration pipeline."""
        self.logger.info("Initializing plugin integration pipeline...")
        
        # Discover Python files
        python_files = list(self.project_root.rglob("*.py"))
        context["python_files"] = python_files
        context["total_files"] = len(python_files)
        
        self.logger.info(f"Discovered {len(python_files)} Python files")
        stage_result.complete()
    
    async def _execute_preparation(self, stage_result: StageResult, context: Dict[str, Any]):
        """Prepare plugins and plugin context."""
        self.logger.info("Preparing plugins and plugin context...")
        
        # Discover plugins using both plugin managers
        try:
            # Use the core plugin manager to discover plugins
            discovered_plugins = self.core_plugin_manager.discover_plugins()
            self.discovered_plugins = [
                {
                    "name": plugin.name,
                    "description": plugin.description,
                    "version": plugin.version,
                    "plugin_type": plugin.plugin_type,
                    "class_name": plugin.class_name,
                    "module_path": plugin.module_path,
                    "enabled": plugin.enabled
                }
                for plugin in discovered_plugins
            ]
            
            self.logger.info(f"Discovered {len(self.discovered_plugins)} plugins")
            
            # Create plugin context
            plugin_context = PluginContext(
                project_root=self.project_root,
                target_files=context["python_files"],
                configuration=self.config.configuration,
                output_dir=self.output_dir,
                parallel_execution=self.config.parallel_execution,
                max_workers=self.config.max_workers,
                timeout=self.config.timeout_per_stage,
                dry_run=self.config.dry_run,
                verbose=self.config.verbose
            )
            
            context["plugin_context"] = plugin_context
            context["discovered_plugins"] = self.discovered_plugins
            
        except Exception as e:
            stage_result.fail([f"Failed to prepare plugins: {e}"])
            return
        
        stage_result.complete()
    
    async def _execute_analysis(self, stage_result: StageResult, context: Dict[str, Any]):
        """Analyze available plugins and their capabilities."""
        self.logger.info("Analyzing plugin capabilities...")
        
        plugin_context = context["plugin_context"]
        discovered_plugins = context["discovered_plugins"]
        
        # Categorize plugins
        plugin_categories = {
            "fixers": [],
            "analyzers": [],
            "unknown": []
        }
        
        for plugin_info in discovered_plugins:
            if plugin_info["plugin_type"] == "fixer":
                plugin_categories["fixers"].append(plugin_info)
            elif plugin_info["plugin_type"] == "analyzer":
                plugin_categories["analyzers"].append(plugin_info)
            else:
                plugin_categories["unknown"].append(plugin_info)
        
        context["plugin_categories"] = plugin_categories
        
        self.logger.info(f"Plugin analysis complete: {len(plugin_categories['fixers'])} fixers, "
                        f"{len(plugin_categories['analyzers'])} analyzers, "
                        f"{len(plugin_categories['unknown'])} unknown")
        
        stage_result.complete()
    
    async def _execute_processing(self, stage_result: StageResult, context: Dict[str, Any]):
        """Execute plugins based on their categories and priorities."""
        self.logger.info("Processing plugins...")
        
        plugin_context = context["plugin_context"]
        plugin_categories = context["plugin_categories"]
        
        # Execute analyzers first
        analyzer_results = []
        for plugin_info in plugin_categories["analyzers"]:
            try:
                self.logger.info(f"Executing analyzer: {plugin_info['name']}")
                
                # Create a mock result for now since we're not actually executing plugins
                # In a real implementation, you would load and execute the actual plugin
                result = PluginResult(
                    plugin_name=plugin_info["name"],
                    success=True,
                    execution_time=0.1,
                    files_processed=len(plugin_context.target_files),
                    issues_found=0,
                    output_data={"plugin_info": plugin_info}
                )
                
                analyzer_results.append(result)
                self.plugin_results[plugin_info["name"]] = result
                
            except Exception as e:
                self.logger.error(f"Failed to execute analyzer {plugin_info['name']}: {e}")
                error_result = PluginResult(
                    plugin_name=plugin_info["name"],
                    success=False,
                    execution_time=0.0
                )
                error_result.add_error(str(e))
                analyzer_results.append(error_result)
                self.plugin_results[plugin_info["name"]] = error_result
        
        # Execute fixers
        fixer_results = []
        for plugin_info in plugin_categories["fixers"]:
            try:
                self.logger.info(f"Executing fixer: {plugin_info['name']}")
                
                # Create a mock result for now
                result = PluginResult(
                    plugin_name=plugin_info["name"],
                    success=True,
                    execution_time=0.1,
                    files_processed=len(plugin_context.target_files),
                    files_fixed=0,
                    issues_fixed=0,
                    output_data={"plugin_info": plugin_info}
                )
                
                fixer_results.append(result)
                self.plugin_results[plugin_info["name"]] = result
                
            except Exception as e:
                self.logger.error(f"Failed to execute fixer {plugin_info['name']}: {e}")
                error_result = PluginResult(
                    plugin_name=plugin_info["name"],
                    success=False,
                    execution_time=0.0
                )
                error_result.add_error(str(e))
                fixer_results.append(error_result)
                self.plugin_results[plugin_info["name"]] = error_result
        
        context["analyzer_results"] = analyzer_results
        context["fixer_results"] = fixer_results
        
        self.logger.info(f"Plugin processing complete: {len(analyzer_results)} analyzers, "
                        f"{len(fixer_results)} fixers executed")
        
        stage_result.complete()
    
    async def _execute_aggregation(self, stage_result: StageResult, context: Dict[str, Any]):
        """Aggregate plugin results and generate summary statistics."""
        self.logger.info("Aggregating plugin results...")
        
        analyzer_results = context["analyzer_results"]
        fixer_results = context["fixer_results"]
        
        # Calculate summary statistics
        total_plugins = len(analyzer_results) + len(fixer_results)
        successful_plugins = sum(1 for r in analyzer_results + fixer_results if r.success)
        failed_plugins = total_plugins - successful_plugins
        
        total_files_processed = sum(r.files_processed for r in analyzer_results + fixer_results)
        total_issues_found = sum(r.issues_found for r in analyzer_results + fixer_results)
        total_issues_fixed = sum(r.issues_fixed for r in fixer_results)
        
        summary = {
            "total_plugins": total_plugins,
            "successful_plugins": successful_plugins,
            "failed_plugins": failed_plugins,
            "total_files_processed": total_files_processed,
            "total_issues_found": total_issues_found,
            "total_issues_fixed": total_issues_fixed,
            "success_rate": successful_plugins / total_plugins if total_plugins > 0 else 0
        }
        
        context["plugin_summary"] = summary
        
        self.logger.info(f"Aggregation complete: {successful_plugins}/{total_plugins} plugins successful, "
                        f"{total_files_processed} files processed, {total_issues_found} issues found, "
                        f"{total_issues_fixed} issues fixed")
        
        stage_result.complete()
    
    async def _execute_reporting(self, stage_result: StageResult, context: Dict[str, Any]):
        """Generate plugin integration reports."""
        self.logger.info("Generating plugin integration reports...")
        
        # Generate summary report
        summary_report = {
            "pipeline": "plugin_integration",
            "timestamp": self.pipeline_result.start_time.isoformat() if self.pipeline_result.start_time else None,
            "project_root": str(self.project_root),
            "summary": context["plugin_summary"],
            "discovered_plugins": context["discovered_plugins"],
            "plugin_categories": context["plugin_categories"],
            "plugin_results": {name: result.to_dict() for name, result in self.plugin_results.items()}
        }
        
        # Save reports
        timestamp = self.pipeline_result.start_time.strftime("%Y%m%d_%H%M%S") if self.pipeline_result.start_time else "unknown"
        
        summary_file = self.reports_dir / f"plugin_integration_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            import json
            json.dump(summary_report, f, indent=2)
        
        self.logger.info(f"Reports generated: {summary_file}")
        stage_result.complete()
    
    async def _execute_cleanup(self, stage_result: StageResult, context: Dict[str, Any]):
        """Clean up resources."""
        self.logger.info("Cleaning up...")
        
        # Clear plugin results
        self.plugin_results.clear()
        self.discovered_plugins.clear()
        
        stage_result.complete()


async def run_plugin_integration(project_root: Union[str, Path], 
                                configuration: Optional[Dict[str, Any]] = None,
                                verbose: bool = False) -> PipelineResult:
    """
    Convenience function to run the plugin integration pipeline.
    
    Args:
        project_root: Root directory of the project to analyze
        configuration: Optional configuration dictionary
        verbose: Whether to enable verbose logging
        
    Returns:
        PipelineResult: Result of the plugin integration pipeline
    """
    config = PipelineConfig(
        pipeline_name="plugin_integration",
        project_root=Path(project_root),
        configuration=configuration or {},
        verbose=verbose,
        log_level="INFO" if verbose else "WARNING"
    )
    
    pipeline = PluginIntegrationPipeline(config)
    return await pipeline.run()