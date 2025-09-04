"""
Plugin Manager

Manages plugin execution, lifecycle, and coordination.
"""

import asyncio
import concurrent.futures
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Any
from .base_plugin import BasePlugin, PluginContext, PluginResult, PluginCategory, PluginPriority
from .plugin_registry import PluginRegistry
from .exceptions import PluginError, PluginExecutionError, PluginTimeoutError


class PluginManager:
    """
    Manages plugin execution and coordination.
    """
    
    def __init__(self, registry: Optional[PluginRegistry] = None):
        self.registry = registry or PluginRegistry()
        self.execution_history: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_execution_time": 0.0,
            "plugin_stats": {}
        }
    
    def execute_plugin(self, 
                      plugin_name: str, 
                      context: PluginContext,
                      timeout: Optional[int] = None) -> PluginResult:
        """
        Execute a single plugin.
        
        Args:
            plugin_name: Name of the plugin to execute
            context: Plugin execution context
            timeout: Optional timeout in seconds
            
        Returns:
            PluginResult: Result of plugin execution
            
        Raises:
            PluginExecutionError: If plugin execution fails
            PluginTimeoutError: If plugin execution times out
        """
        start_time = time.time()
        
        try:
            # Get plugin instance
            plugin = self.registry.get_plugin(plugin_name)
            
            # Check if plugin is available
            if not plugin.is_available():
                missing_deps = plugin.validate_dependencies()
                raise PluginExecutionError(
                    plugin_name, 
                    Exception(f"Plugin dependencies not met: {missing_deps}")
                )
            
            # Execute plugin with timeout
            if timeout:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(plugin.execute, context)
                    try:
                        result = future.result(timeout=timeout)
                    except concurrent.futures.TimeoutError:
                        raise PluginTimeoutError(plugin_name, timeout)
            else:
                result = plugin.execute(context)
            
            # Update metrics
            self._update_metrics(plugin_name, result, time.time() - start_time)
            
            # Record execution
            self._record_execution(plugin_name, context, result, time.time() - start_time)
            
            return result
            
        except (PluginExecutionError, PluginTimeoutError):
            raise
        except Exception as e:
            execution_time = time.time() - start_time
            error_result = PluginResult(
                plugin_name=plugin_name,
                success=False,
                execution_time=execution_time
            )
            error_result.add_error(str(e))
            
            self._update_metrics(plugin_name, error_result, execution_time)
            self._record_execution(plugin_name, context, error_result, execution_time)
            
            raise PluginExecutionError(plugin_name, e)
    
    def execute_plugins_sequential(self, 
                                  plugin_names: List[str], 
                                  context: PluginContext,
                                  timeout_per_plugin: Optional[int] = None) -> List[PluginResult]:
        """
        Execute multiple plugins sequentially.
        
        Args:
            plugin_names: List of plugin names to execute
            context: Plugin execution context
            timeout_per_plugin: Optional timeout per plugin in seconds
            
        Returns:
            List[PluginResult]: Results of plugin executions
        """
        results = []
        
        for plugin_name in plugin_names:
            try:
                result = self.execute_plugin(plugin_name, context, timeout_per_plugin)
                results.append(result)
                
                # Stop on critical failure if configured
                if (not result.success and 
                    self.registry.get_plugin(plugin_name).get_metadata().priority == PluginPriority.CRITICAL):
                    break
                    
            except (PluginExecutionError, PluginTimeoutError) as e:
                # Create error result
                error_result = PluginResult(
                    plugin_name=plugin_name,
                    success=False,
                    execution_time=0.0
                )
                error_result.add_error(str(e))
                results.append(error_result)
                
                # Stop on critical failure
                if self.registry.get_plugin(plugin_name).get_metadata().priority == PluginPriority.CRITICAL:
                    break
        
        return results
    
    def execute_plugins_parallel(self, 
                                plugin_names: List[str], 
                                context: PluginContext,
                                max_workers: Optional[int] = None,
                                timeout_per_plugin: Optional[int] = None) -> List[PluginResult]:
        """
        Execute multiple plugins in parallel.
        
        Args:
            plugin_names: List of plugin names to execute
            context: Plugin execution context
            max_workers: Maximum number of worker threads
            timeout_per_plugin: Optional timeout per plugin in seconds
            
        Returns:
            List[PluginResult]: Results of plugin executions
        """
        if max_workers is None:
            max_workers = min(len(plugin_names), 4)
        
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_plugin = {
                executor.submit(self.execute_plugin, plugin_name, context, timeout_per_plugin): plugin_name
                for plugin_name in plugin_names
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_plugin):
                plugin_name = future_to_plugin[future]
                try:
                    result = future.result()
                    results.append(result)
                except (PluginExecutionError, PluginTimeoutError) as e:
                    error_result = PluginResult(
                        plugin_name=plugin_name,
                        success=False,
                        execution_time=0.0
                    )
                    error_result.add_error(str(e))
                    results.append(error_result)
        
        # Sort results by original plugin order
        plugin_order = {name: i for i, name in enumerate(plugin_names)}
        results.sort(key=lambda r: plugin_order.get(r.plugin_name, 999))
        
        return results
    
    def execute_plugins_by_category(self, 
                                   category: PluginCategory, 
                                   context: PluginContext,
                                   parallel: bool = False,
                                   max_workers: Optional[int] = None,
                                   timeout_per_plugin: Optional[int] = None) -> List[PluginResult]:
        """
        Execute all plugins in a category.
        
        Args:
            category: Plugin category to execute
            context: Plugin execution context
            parallel: Whether to execute plugins in parallel
            max_workers: Maximum number of worker threads (for parallel execution)
            timeout_per_plugin: Optional timeout per plugin in seconds
            
        Returns:
            List[PluginResult]: Results of plugin executions
        """
        plugin_names = self.registry.list_plugins_by_category(category)
        
        if not plugin_names:
            return []
        
        # Get execution order
        ordered_plugins = self.registry.get_execution_order(plugin_names)
        
        if parallel:
            return self.execute_plugins_parallel(
                ordered_plugins, context, max_workers, timeout_per_plugin
            )
        else:
            return self.execute_plugins_sequential(
                ordered_plugins, context, timeout_per_plugin
            )
    
    def execute_plugins_by_priority(self, 
                                   priority: PluginPriority, 
                                   context: PluginContext,
                                   parallel: bool = False,
                                   max_workers: Optional[int] = None,
                                   timeout_per_plugin: Optional[int] = None) -> List[PluginResult]:
        """
        Execute all plugins with a specific priority.
        
        Args:
            priority: Plugin priority to execute
            context: Plugin execution context
            parallel: Whether to execute plugins in parallel
            max_workers: Maximum number of worker threads (for parallel execution)
            timeout_per_plugin: Optional timeout per plugin in seconds
            
        Returns:
            List[PluginResult]: Results of plugin executions
        """
        plugin_names = self.registry.list_plugins_by_priority(priority)
        
        if not plugin_names:
            return []
        
        if parallel:
            return self.execute_plugins_parallel(
                plugin_names, context, max_workers, timeout_per_plugin
            )
        else:
            return self.execute_plugins_sequential(
                plugin_names, context, timeout_per_plugin
            )
    
    def execute_pipeline(self, 
                        plugin_names: Optional[List[str]] = None,
                        categories: Optional[List[PluginCategory]] = None,
                        priorities: Optional[List[PluginPriority]] = None,
                        context: Optional[PluginContext] = None,
                        parallel: bool = False,
                        max_workers: Optional[int] = None,
                        timeout_per_plugin: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute a complete pipeline of plugins.
        
        Args:
            plugin_names: Specific plugins to execute
            categories: Categories of plugins to execute
            priorities: Priorities of plugins to execute
            context: Plugin execution context
            parallel: Whether to execute plugins in parallel
            max_workers: Maximum number of worker threads
            timeout_per_plugin: Optional timeout per plugin in seconds
            
        Returns:
            Dict[str, Any]: Pipeline execution results
        """
        if context is None:
            raise ValueError("Plugin context is required")
        
        start_time = time.time()
        
        # Determine which plugins to execute
        plugins_to_execute = set()
        
        if plugin_names:
            plugins_to_execute.update(plugin_names)
        
        if categories:
            for category in categories:
                plugins_to_execute.update(self.registry.list_plugins_by_category(category))
        
        if priorities:
            for priority in priorities:
                plugins_to_execute.update(self.registry.list_plugins_by_priority(priority))
        
        if not plugins_to_execute:
            plugins_to_execute = set(self.registry.list_plugins())
        
        # Get execution order
        ordered_plugins = self.registry.get_execution_order(list(plugins_to_execute))
        
        # Execute plugins
        if parallel:
            results = self.execute_plugins_parallel(
                ordered_plugins, context, max_workers, timeout_per_plugin
            )
        else:
            results = self.execute_plugins_sequential(
                ordered_plugins, context, timeout_per_plugin
            )
        
        # Calculate summary
        total_time = time.time() - start_time
        successful_plugins = sum(1 for r in results if r.success)
        failed_plugins = len(results) - successful_plugins
        total_files_processed = sum(r.files_processed for r in results)
        total_issues_found = sum(r.issues_found for r in results)
        total_issues_fixed = sum(r.issues_fixed for r in results)
        
        return {
            "pipeline_info": {
                "total_plugins": len(results),
                "successful_plugins": successful_plugins,
                "failed_plugins": failed_plugins,
                "total_execution_time": total_time,
                "parallel_execution": parallel,
                "max_workers": max_workers
            },
            "results": [result.to_dict() for result in results],
            "summary": {
                "total_files_processed": total_files_processed,
                "total_issues_found": total_issues_found,
                "total_issues_fixed": total_issues_fixed,
                "success_rate": successful_plugins / len(results) if results else 0
            }
        }
    
    def get_execution_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get execution history.
        
        Args:
            limit: Optional limit on number of executions to return
            
        Returns:
            List[Dict[str, Any]]: Execution history
        """
        if limit:
            return self.execution_history[-limit:]
        return self.execution_history.copy()
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get execution metrics.
        
        Returns:
            Dict[str, Any]: Execution metrics
        """
        return self.metrics.copy()
    
    def get_plugin_stats(self, plugin_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific plugin.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Dict[str, Any]: Plugin statistics
        """
        return self.metrics["plugin_stats"].get(plugin_name, {
            "executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_execution_time": 0.0,
            "average_execution_time": 0.0,
            "last_execution": None
        })
    
    def clear_history(self) -> None:
        """Clear execution history and reset metrics."""
        self.execution_history.clear()
        self.metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_execution_time": 0.0,
            "plugin_stats": {}
        }
    
    def _update_metrics(self, plugin_name: str, result: PluginResult, execution_time: float) -> None:
        """Update execution metrics."""
        self.metrics["total_executions"] += 1
        self.metrics["total_execution_time"] += execution_time
        
        if result.success:
            self.metrics["successful_executions"] += 1
        else:
            self.metrics["failed_executions"] += 1
        
        # Update plugin-specific stats
        if plugin_name not in self.metrics["plugin_stats"]:
            self.metrics["plugin_stats"][plugin_name] = {
                "executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "total_execution_time": 0.0,
                "average_execution_time": 0.0,
                "last_execution": None
            }
        
        stats = self.metrics["plugin_stats"][plugin_name]
        stats["executions"] += 1
        stats["total_execution_time"] += execution_time
        stats["average_execution_time"] = stats["total_execution_time"] / stats["executions"]
        stats["last_execution"] = result.timestamp.isoformat()
        
        if result.success:
            stats["successful_executions"] += 1
        else:
            stats["failed_executions"] += 1
    
    def _record_execution(self, plugin_name: str, context: PluginContext, 
                         result: PluginResult, execution_time: float) -> None:
        """Record execution in history."""
        execution_record = {
            "plugin_name": plugin_name,
            "timestamp": result.timestamp.isoformat(),
            "execution_time": execution_time,
            "success": result.success,
            "files_processed": result.files_processed,
            "files_fixed": result.files_fixed,
            "files_failed": result.files_failed,
            "issues_found": result.issues_found,
            "issues_fixed": result.issues_fixed,
            "errors": result.errors,
            "warnings": result.warnings,
            "context": {
                "project_root": str(context.project_root),
                "target_files": [str(f) for f in context.target_files],
                "parallel_execution": context.parallel_execution,
                "dry_run": context.dry_run
            }
        }
        
        self.execution_history.append(execution_record)