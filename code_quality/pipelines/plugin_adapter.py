"""
Plugin Adapter

This module provides adapters to integrate existing plugins with the pipeline system.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base_pipeline import BasePipeline, PipelineConfig, PipelineStage, StageResult, PipelineStatus, PipelineResult
try:
    from plugins.base_plugin import PluginContext, PluginResult, PluginCategory, PluginPriority
    from plugins.plugin_manager import PluginManager
except ImportError:
    PluginContext = None
    PluginResult = None
    PluginCategory = None
    PluginPriority = None
    PluginManager = None

try:
    from core.plugins import PluginManager as CorePluginManager, BaseCodeFixer, BaseCodeAnalyzer
except ImportError:
    CorePluginManager = None
    BaseCodeFixer = None
    BaseCodeAnalyzer = None


class PluginToPipelineAdapter:
    """
    Adapter to convert plugin execution into pipeline stages.
    
    This adapter allows existing plugins to be executed within the pipeline framework
    by converting plugin execution into pipeline stages.
    """
    
    def __init__(self, plugin_manager: Optional[PluginManager] = None):
        self.plugin_manager = plugin_manager or PluginManager()
        self.core_plugin_manager = CorePluginManager()
        self.logger = logging.getLogger(__name__)
    
    def create_plugin_context(self, project_root: Path, target_files: List[Path], 
                            configuration: Dict[str, Any]) -> PluginContext:
        """Create a plugin context from pipeline parameters."""
        return PluginContext(
            project_root=project_root,
            target_files=target_files,
            configuration=configuration,
            parallel_execution=False,  # Can be configured
            max_workers=4,
            timeout=300,
            dry_run=False,
            verbose=False
        )
    
    async def execute_plugin_as_stage(self, plugin_name: str, context: PluginContext) -> PluginResult:
        """Execute a plugin and return the result."""
        try:
            # Try to get plugin from the new plugin manager first
            plugin = self.plugin_manager.registry.get_plugin(plugin_name)
            if plugin:
                return self.plugin_manager.execute_plugin(plugin_name, context)
            
            # Fall back to core plugin manager
            plugin = self.core_plugin_manager.get_plugin(plugin_name)
            if plugin:
                # Execute the plugin directly
                start_time = asyncio.get_event_loop().time()
                result = plugin.execute(context)
                execution_time = asyncio.get_event_loop().time() - start_time
                
                # Convert to PluginResult if needed
                if not isinstance(result, PluginResult):
                    plugin_result = PluginResult(
                        plugin_name=plugin_name,
                        success=True,
                        execution_time=execution_time,
                        output_data=result if isinstance(result, dict) else {"result": result}
                    )
                    return plugin_result
                
                return result
            
            # Plugin not found
            error_result = PluginResult(
                plugin_name=plugin_name,
                success=False,
                execution_time=0.0
            )
            error_result.add_error(f"Plugin '{plugin_name}' not found")
            return error_result
            
        except Exception as e:
            error_result = PluginResult(
                plugin_name=plugin_name,
                success=False,
                execution_time=0.0
            )
            error_result.add_error(f"Plugin execution failed: {e}")
            return error_result
    
    def get_available_plugins(self) -> Dict[str, Dict[str, Any]]:
        """Get all available plugins from both plugin managers."""
        plugins = {}
        
        # Get plugins from core plugin manager
        try:
            discovered_plugins = self.core_plugin_manager.discover_plugins()
            for plugin_info in discovered_plugins:
                plugins[plugin_info.name] = {
                    "name": plugin_info.name,
                    "description": plugin_info.description,
                    "version": plugin_info.version,
                    "plugin_type": plugin_info.plugin_type,
                    "class_name": plugin_info.class_name,
                    "module_path": plugin_info.module_path,
                    "enabled": plugin_info.enabled,
                    "source": "core_plugin_manager"
                }
        except Exception as e:
            self.logger.warning(f"Failed to discover plugins from core plugin manager: {e}")
        
        # Get plugins from new plugin manager
        try:
            registered_plugins = self.plugin_manager.registry.list_plugins()
            for plugin_name in registered_plugins:
                if plugin_name not in plugins:  # Don't override core plugins
                    plugin = self.plugin_manager.registry.get_plugin(plugin_name)
                    if plugin:
                        metadata = plugin.get_metadata()
                        plugins[plugin_name] = {
                            "name": metadata.name,
                            "description": metadata.description,
                            "version": metadata.version,
                            "plugin_type": metadata.category.value,
                            "class_name": plugin.__class__.__name__,
                            "module_path": plugin.__class__.__module__,
                            "enabled": True,
                            "source": "plugin_manager"
                        }
        except Exception as e:
            self.logger.warning(f"Failed to get plugins from plugin manager: {e}")
        
        return plugins


class PluginBasedPipeline(BasePipeline):
    """
    Base pipeline class that integrates with the plugin system.
    
    This pipeline can execute plugins as part of its stages, providing
    a bridge between the plugin system and the pipeline framework.
    """
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.plugin_adapter = PluginToPipelineAdapter()
        self.plugin_results: Dict[str, PluginResult] = {}
        self.available_plugins: Dict[str, Dict[str, Any]] = {}
    
    def get_available_plugins(self) -> Dict[str, Dict[str, Any]]:
        """Get all available plugins."""
        if not self.available_plugins:
            self.available_plugins = self.plugin_adapter.get_available_plugins()
        return self.available_plugins
    
    async def execute_plugin_stage(self, plugin_name: str, stage_result: StageResult, 
                                 context: Dict[str, Any]) -> PluginResult:
        """Execute a plugin as part of a pipeline stage."""
        self.logger.info(f"Executing plugin: {plugin_name}")
        
        # Create plugin context
        plugin_context = self.plugin_adapter.create_plugin_context(
            project_root=self.project_root,
            target_files=context.get("target_files", []),
            configuration=self.config.configuration
        )
        
        # Execute plugin
        result = await self.plugin_adapter.execute_plugin_as_stage(plugin_name, plugin_context)
        
        # Store result
        self.plugin_results[plugin_name] = result
        
        # Update stage result
        if result.success:
            stage_result.add_metric(f"{plugin_name}_success", True)
            stage_result.add_metric(f"{plugin_name}_execution_time", result.execution_time)
            stage_result.add_metric(f"{plugin_name}_files_processed", result.files_processed)
            stage_result.add_metric(f"{plugin_name}_issues_found", result.issues_found)
            stage_result.add_metric(f"{plugin_name}_issues_fixed", result.issues_fixed)
        else:
            stage_result.add_metric(f"{plugin_name}_success", False)
            stage_result.add_metric(f"{plugin_name}_execution_time", result.execution_time)
            for error in result.errors:
                stage_result.add_error(f"{plugin_name}: {error}")
        
        return result
    
    async def execute_plugins_by_category(self, category: str, stage_result: StageResult, 
                                        context: Dict[str, Any]) -> List[PluginResult]:
        """Execute all plugins in a specific category."""
        available_plugins = self.get_available_plugins()
        
        # Filter plugins by category
        category_plugins = [
            name for name, info in available_plugins.items()
            if info.get("plugin_type") == category and info.get("enabled", True)
        ]
        
        results = []
        for plugin_name in category_plugins:
            result = await self.execute_plugin_stage(plugin_name, stage_result, context)
            results.append(result)
        
        return results
    
    async def execute_plugins_by_type(self, plugin_type: str, stage_result: StageResult, 
                                    context: Dict[str, Any]) -> List[PluginResult]:
        """Execute all plugins of a specific type (fixer, analyzer, etc.)."""
        available_plugins = self.get_available_plugins()
        
        # Filter plugins by type
        type_plugins = [
            name for name, info in available_plugins.items()
            if info.get("plugin_type") == plugin_type and info.get("enabled", True)
        ]
        
        results = []
        for plugin_name in type_plugins:
            result = await self.execute_plugin_stage(plugin_name, stage_result, context)
            results.append(result)
        
        return results


class PluginExecutionPipeline(PluginBasedPipeline):
    """
    Pipeline that executes plugins based on configuration.
    
    This pipeline can execute specific plugins or categories of plugins
    as part of a code quality analysis workflow.
    """
    
    def __init__(self, config: PipelineConfig, plugins_to_execute: Optional[List[str]] = None):
        super().__init__(config)
        self.plugins_to_execute = plugins_to_execute or []
    
    def get_stages(self) -> List[PipelineStage]:
        """Define the stages for plugin execution pipeline."""
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
        """Execute a specific stage of the plugin execution pipeline."""
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
        """Initialize the plugin execution pipeline."""
        self.logger.info("Initializing plugin execution pipeline...")
        
        # Discover Python files
        python_files = list(self.project_root.rglob("*.py"))
        context["target_files"] = python_files
        context["total_files"] = len(python_files)
        
        self.logger.info(f"Discovered {len(python_files)} Python files")
        stage_result.complete()
    
    async def _execute_preparation(self, stage_result: StageResult, context: Dict[str, Any]):
        """Prepare plugins for execution."""
        self.logger.info("Preparing plugins for execution...")
        
        # Get available plugins
        available_plugins = self.get_available_plugins()
        context["available_plugins"] = available_plugins
        
        # Determine which plugins to execute
        if self.plugins_to_execute:
            plugins_to_run = [
                name for name in self.plugins_to_execute 
                if name in available_plugins
            ]
        else:
            # Execute all enabled plugins
            plugins_to_run = [
                name for name, info in available_plugins.items()
                if info.get("enabled", True)
            ]
        
        context["plugins_to_run"] = plugins_to_run
        
        self.logger.info(f"Prepared {len(plugins_to_run)} plugins for execution")
        stage_result.complete()
    
    async def _execute_analysis(self, stage_result: StageResult, context: Dict[str, Any]):
        """Analyze plugin capabilities and dependencies."""
        self.logger.info("Analyzing plugin capabilities...")
        
        available_plugins = context["available_plugins"]
        plugins_to_run = context["plugins_to_run"]
        
        # Categorize plugins
        plugin_categories = {
            "fixers": [],
            "analyzers": [],
            "other": []
        }
        
        for plugin_name in plugins_to_run:
            plugin_info = available_plugins[plugin_name]
            plugin_type = plugin_info.get("plugin_type", "other")
            
            if plugin_type in plugin_categories:
                plugin_categories[plugin_type].append(plugin_name)
            else:
                plugin_categories["other"].append(plugin_name)
        
        context["plugin_categories"] = plugin_categories
        
        self.logger.info(f"Plugin analysis complete: {len(plugin_categories['fixers'])} fixers, "
                        f"{len(plugin_categories['analyzers'])} analyzers, "
                        f"{len(plugin_categories['other'])} other")
        
        stage_result.complete()
    
    async def _execute_processing(self, stage_result: StageResult, context: Dict[str, Any]):
        """Execute the plugins."""
        self.logger.info("Executing plugins...")
        
        plugin_categories = context["plugin_categories"]
        
        # Execute analyzers first
        analyzer_results = await self.execute_plugins_by_type("analyzer", stage_result, context)
        
        # Then execute fixers
        fixer_results = await self.execute_plugins_by_type("fixer", stage_result, context)
        
        context["analyzer_results"] = analyzer_results
        context["fixer_results"] = fixer_results
        
        self.logger.info(f"Plugin execution complete: {len(analyzer_results)} analyzers, "
                        f"{len(fixer_results)} fixers executed")
        
        stage_result.complete()
    
    async def _execute_aggregation(self, stage_result: StageResult, context: Dict[str, Any]):
        """Aggregate plugin results."""
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
        
        self.logger.info(f"Aggregation complete: {successful_plugins}/{total_plugins} plugins successful")
        
        stage_result.complete()
    
    async def _execute_reporting(self, stage_result: StageResult, context: Dict[str, Any]):
        """Generate plugin execution reports."""
        self.logger.info("Generating plugin execution reports...")
        
        # Generate summary report
        summary_report = {
            "pipeline": "plugin_execution",
            "timestamp": self.pipeline_result.start_time.isoformat() if self.pipeline_result.start_time else None,
            "project_root": str(self.project_root),
            "summary": context["plugin_summary"],
            "available_plugins": context["available_plugins"],
            "plugins_executed": context["plugins_to_run"],
            "plugin_results": {name: result.to_dict() for name, result in self.plugin_results.items()}
        }
        
        # Save reports
        timestamp = self.pipeline_result.start_time.strftime("%Y%m%d_%H%M%S") if self.pipeline_result.start_time else "unknown"
        
        summary_file = self.reports_dir / f"plugin_execution_summary_{timestamp}.json"
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
        self.available_plugins.clear()
        
        stage_result.complete()


async def run_plugin_execution(project_root: Union[str, Path], 
                             plugins_to_execute: Optional[List[str]] = None,
                             configuration: Optional[Dict[str, Any]] = None,
                             verbose: bool = False) -> PipelineResult:
    """
    Convenience function to run the plugin execution pipeline.
    
    Args:
        project_root: Root directory of the project to analyze
        plugins_to_execute: Optional list of specific plugins to execute
        configuration: Optional configuration dictionary
        verbose: Whether to enable verbose logging
        
    Returns:
        PipelineResult: Result of the plugin execution pipeline
    """
    config = PipelineConfig(
        pipeline_name="plugin_execution",
        project_root=Path(project_root),
        configuration=configuration or {},
        verbose=verbose,
        log_level="INFO" if verbose else "WARNING"
    )
    
    pipeline = PluginExecutionPipeline(config, plugins_to_execute)
    return await pipeline.run()