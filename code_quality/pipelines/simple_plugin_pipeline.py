"""
Simple Plugin Pipeline

A simplified pipeline that integrates with the existing plugin system
without complex dependencies.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base_pipeline import (
    BasePipeline, PipelineConfig, PipelineStage, StageResult, 
    PipelineStatus, PipelineResult
)


class SimplePluginPipeline(BasePipeline):
    """
    Simple pipeline that integrates with the existing plugin system.
    
    This pipeline discovers and reports on available plugins without
    complex execution dependencies.
    """
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config, "simple_plugin")
        self.discovered_plugins: List[Dict[str, Any]] = []
        self.plugin_categories: Dict[str, List[str]] = {}
        
    def get_stages(self) -> List[PipelineStage]:
        """Define the stages for simple plugin pipeline."""
        return [
            PipelineStage.INITIALIZATION,
            PipelineStage.PREPARATION,
            PipelineStage.ANALYSIS,
            PipelineStage.PROCESSING,
            PipelineStage.AGGREGATION,
            PipelineStage.REPORTING,
            PipelineStage.CLEANUP
        ]
    
    async def execute_stage(self, stage: PipelineStage, context: Dict[str, Any]) -> StageResult:
        """Execute a specific stage of the simple plugin pipeline."""
        stage_result = StageResult(
            stage=stage,
            status=PipelineStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
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
            
            stage_result.complete()
        except Exception as e:
            stage_result.fail([str(e)])
        
        return stage_result
    
    async def _execute_initialization(self, stage_result: StageResult, context: Dict[str, Any]):
        """Initialize the simple plugin pipeline."""
        self.logger.info("Initializing simple plugin pipeline...")
        
        # Discover Python files
        python_files = list(self.config.project_root.rglob("*.py"))
        context["python_files"] = python_files
        context["total_files"] = len(python_files)
        
        self.logger.info(f"Discovered {len(python_files)} Python files")
        stage_result.complete()
    
    async def _execute_preparation(self, stage_result: StageResult, context: Dict[str, Any]):
        """Prepare plugin discovery."""
        self.logger.info("Preparing plugin discovery...")
        
        # Discover plugin files
        plugin_files = list((self.config.project_root / "plugins").rglob("*.py"))
        context["plugin_files"] = plugin_files
        
        self.logger.info(f"Discovered {len(plugin_files)} plugin files")
        stage_result.complete()
    
    async def _execute_analysis(self, stage_result: StageResult, context: Dict[str, Any]):
        """Analyze plugin files and extract information."""
        self.logger.info("Analyzing plugin files...")
        
        plugin_files = context["plugin_files"]
        discovered_plugins = []
        
        for plugin_file in plugin_files:
            try:
                # Extract basic information from plugin files
                plugin_info = self._analyze_plugin_file(plugin_file)
                if plugin_info:
                    discovered_plugins.append(plugin_info)
            except Exception as e:
                self.logger.warning(f"Failed to analyze plugin file {plugin_file}: {e}")
        
        self.discovered_plugins = discovered_plugins
        context["discovered_plugins"] = discovered_plugins
        
        self.logger.info(f"Analyzed {len(discovered_plugins)} plugins")
        stage_result.complete()
    
    def _analyze_plugin_file(self, plugin_file: Path) -> Optional[Dict[str, Any]]:
        """Analyze a plugin file to extract basic information."""
        try:
            with open(plugin_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract class names
            class_names = []
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('class ') and '(' in line:
                    class_name = line.split('(')[0].replace('class ', '').strip()
                    class_names.append(class_name)
            
            # Determine plugin type based on filename and content
            plugin_type = "unknown"
            if "fixer" in plugin_file.name.lower():
                plugin_type = "fixer"
            elif "analyzer" in plugin_file.name.lower():
                plugin_type = "analyzer"
            elif "Fixer" in content and "class" in content:
                plugin_type = "fixer"
            elif "Analyzer" in content and "class" in content:
                plugin_type = "analyzer"
            elif "BaseCodeFixer" in content:
                plugin_type = "fixer"
            elif "BaseCodeAnalyzer" in content:
                plugin_type = "analyzer"
            
            # Extract description if available
            description = ""
            for line in content.split('\n'):
                if '"""' in line and not line.strip().startswith('"""'):
                    # Look for docstring
                    start = content.find('"""')
                    if start != -1:
                        end = content.find('"""', start + 3)
                        if end != -1:
                            description = content[start + 3:end].strip()
                            break
            
            return {
                "name": plugin_file.stem,
                "file_path": str(plugin_file),
                "class_names": class_names,
                "plugin_type": plugin_type,
                "description": description[:100] + "..." if len(description) > 100 else description,
                "size_bytes": plugin_file.stat().st_size
            }
            
        except Exception as e:
            self.logger.debug(f"Failed to analyze plugin file {plugin_file}: {e}")
            return None
    
    async def _execute_processing(self, stage_result: StageResult, context: Dict[str, Any]):
        """Process plugin information."""
        self.logger.info("Processing plugin information...")
        
        discovered_plugins = context["discovered_plugins"]
        
        # Categorize plugins
        plugin_categories = {
            "fixers": [],
            "analyzers": [],
            "unknown": []
        }
        
        for plugin in discovered_plugins:
            plugin_type = plugin.get("plugin_type", "unknown")
            if plugin_type == "fixer":
                plugin_categories["fixers"].append(plugin)
            elif plugin_type == "analyzer":
                plugin_categories["analyzers"].append(plugin)
            else:
                plugin_categories["unknown"].append(plugin)
        
        self.plugin_categories = plugin_categories
        context["plugin_categories"] = plugin_categories
        
        self.logger.info(f"Processed {len(discovered_plugins)} plugins into categories")
        stage_result.complete()
    
    async def _execute_aggregation(self, stage_result: StageResult, context: Dict[str, Any]):
        """Aggregate plugin information."""
        self.logger.info("Aggregating plugin information...")
        
        plugin_categories = context["plugin_categories"]
        
        # Calculate summary statistics
        total_plugins = sum(len(plugins) for plugins in plugin_categories.values())
        fixers_count = len(plugin_categories["fixers"])
        analyzers_count = len(plugin_categories["analyzers"])
        unknown_count = len(plugin_categories["unknown"])
        
        summary = {
            "total_plugins": total_plugins,
            "fixers": fixers_count,
            "analyzers": analyzers_count,
            "unknown": unknown_count,
            "plugin_categories": plugin_categories
        }
        
        context["plugin_summary"] = summary
        
        self.logger.info(f"Aggregation complete: {total_plugins} total plugins "
                        f"({fixers_count} fixers, {analyzers_count} analyzers, {unknown_count} unknown)")
        
        stage_result.complete()
    
    async def _execute_reporting(self, stage_result: StageResult, context: Dict[str, Any]):
        """Generate plugin discovery reports."""
        self.logger.info("Generating plugin discovery reports...")
        
        # Generate summary report
        summary_report = {
            "pipeline": "simple_plugin",
            "timestamp": self.result.start_time.isoformat() if self.result.start_time else None,
            "project_root": str(self.config.project_root),
            "summary": context["plugin_summary"],
            "discovered_plugins": context["discovered_plugins"],
            "plugin_categories": context["plugin_categories"]
        }
        
        # Save reports
        timestamp = self.result.start_time.strftime("%Y%m%d_%H%M%S") if self.result.start_time else "unknown"
        
        summary_file = self.config.output_dir / f"simple_plugin_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            import json
            json.dump(summary_report, f, indent=2)
        
        self.logger.info(f"Reports generated: {summary_file}")
        stage_result.complete()
    
    async def _execute_cleanup(self, stage_result: StageResult, context: Dict[str, Any]):
        """Clean up resources."""
        self.logger.info("Cleaning up...")
        
        # Clear plugin data
        self.discovered_plugins.clear()
        self.plugin_categories.clear()
        
        stage_result.complete()


async def run_simple_plugin_discovery(project_root: Union[str, Path], 
                                    configuration: Optional[Dict[str, Any]] = None,
                                    verbose: bool = False) -> PipelineResult:
    """
    Convenience function to run the simple plugin discovery pipeline.
    
    Args:
        project_root: Root directory of the project to analyze
        configuration: Optional configuration dictionary
        verbose: Whether to enable verbose logging
        
    Returns:
        PipelineResult: Result of the simple plugin discovery pipeline
    """
    config = PipelineConfig(
        project_root=Path(project_root),
        verbose=verbose,
        log_level="INFO" if verbose else "WARNING"
    )
    
    pipeline = SimplePluginPipeline(config)
    return await pipeline.run()