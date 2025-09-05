#!/usr/bin/env python3
"""
Enhanced Import Analyzer Plugin

This plugin integrates the enhanced import and undefined variable analyzer
with the plugin system for extensible functionality.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from plugins.base_plugin import BasePlugin, PluginResult, PluginContext
from plugins.plugin_registry import PluginCategory, PluginPriority
from analyzers.enhanced_import_analysis import (
    EnhancedImportAndUndefinedAnalyzer,
    IssueSeverity,
    IssueType
)


class EnhancedImportAnalyzerPlugin(BasePlugin):
    """
    Plugin for enhanced import and undefined variable analysis.
    
    This plugin provides comprehensive analysis of import issues and undefined variables
    with improved accuracy and detailed reporting.
    """
    
    def __init__(self):
        """Initialize the enhanced import analyzer plugin."""
        super().__init__()
        self.name = "enhanced_import_analyzer"
        self.version = "1.0.0"
        self.description = "Enhanced import and undefined variable analysis with improved accuracy"
        self.category = PluginCategory.ANALYSIS
        self.priority = PluginPriority.HIGH
        self.dependencies = []
        self.analyzer = None
    
    def initialize(self, context: PluginContext) -> bool:
        """Initialize the plugin with the given context."""
        try:
            # Create analyzer configuration
            config = {
                'ignore_patterns': context.configuration.get('ignore_patterns', [
                    '__pycache__', '.git', 'node_modules', '.venv', 'venv', '.pytest_cache'
                ]),
                'max_issues_per_file': context.configuration.get('max_issues_per_file', 100),
                'min_severity': IssueSeverity.LOW
            }
            
            # Initialize the analyzer
            self.analyzer = EnhancedImportAndUndefinedAnalyzer(
                project_root=context.project_root,
                config=config
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced import analyzer plugin: {e}")
            return False
    
    def execute(self, context: PluginContext) -> PluginResult:
        """Execute the enhanced import analysis."""
        try:
            if not self.analyzer:
                return PluginResult(
                    success=False,
                    message="Plugin not properly initialized",
                    data={}
                )
            
            # Run comprehensive analysis
            results = self.analyzer.run_comprehensive_analysis(context.project_root)
            
            # Process results
            processed_results = {
                "analysis_type": "enhanced_import_analysis",
                "timestamp": results.get("summary", {}).get("timestamp"),
                "target_path": results.get("summary", {}).get("target_path"),
                "execution_time": results.get("summary", {}).get("total_execution_time"),
                "summary": results.get("summary", {}),
                "files_analyzed": len(results.get("files", {})),
                "total_issues": results.get("summary", {}).get("total_issues", 0),
                "import_issues": results.get("summary", {}).get("import_issues", 0),
                "undefined_issues": results.get("summary", {}).get("undefined_issues", 0),
                "recommendations": results.get("summary", {}).get("recommendations", [])
            }
            
            # Get high-priority issues
            high_priority = self.analyzer.get_high_priority_issues()
            processed_results["high_priority_issues"] = high_priority
            
            # Get statistics
            stats = self.analyzer.get_issue_statistics()
            processed_results["statistics"] = stats
            
            # Determine success based on issues found
            total_issues = processed_results["total_issues"]
            success = total_issues == 0 or total_issues <= 10  # Allow some issues
            
            message = f"Analysis completed: {total_issues} issues found"
            if total_issues == 0:
                message = "Analysis completed: No issues found"
            elif total_issues <= 10:
                message = f"Analysis completed: {total_issues} issues found (acceptable level)"
            else:
                message = f"Analysis completed: {total_issues} issues found (requires attention)"
            
            return PluginResult(
                success=success,
                message=message,
                data=processed_results
            )
            
        except Exception as e:
            self.logger.error(f"Enhanced import analyzer plugin execution failed: {e}")
            return PluginResult(
                success=False,
                message=f"Plugin execution failed: {str(e)}",
                data={"error": str(e)}
            )
    
    def cleanup(self) -> None:
        """Clean up plugin resources."""
        self.analyzer = None
    
    def get_plugin_info(self) -> Dict[str, Any]:
        """Get detailed plugin information."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "category": self.category.value,
            "priority": self.priority.value,
            "dependencies": self.dependencies,
            "capabilities": [
                "import_analysis",
                "undefined_variable_detection",
                "issue_classification",
                "severity_assessment",
                "recommendation_generation"
            ],
            "supported_file_types": [".py"],
            "configuration_options": {
                "ignore_patterns": "List of directory patterns to ignore",
                "max_issues_per_file": "Maximum issues to report per file",
                "min_severity": "Minimum severity level to report"
            }
        }
    
    def validate_configuration(self, config: Dict[str, Any]) -> List[str]:
        """Validate plugin configuration."""
        errors = []
        
        # Validate ignore_patterns
        if "ignore_patterns" in config:
            if not isinstance(config["ignore_patterns"], list):
                errors.append("ignore_patterns must be a list")
        
        # Validate max_issues_per_file
        if "max_issues_per_file" in config:
            if not isinstance(config["max_issues_per_file"], int) or config["max_issues_per_file"] < 1:
                errors.append("max_issues_per_file must be a positive integer")
        
        # Validate min_severity
        if "min_severity" in config:
            valid_severities = ["low", "medium", "high", "critical"]
            if config["min_severity"] not in valid_severities:
                errors.append(f"min_severity must be one of: {valid_severities}")
        
        return errors


# Plugin registration
def register_plugin() -> EnhancedImportAnalyzerPlugin:
    """Register the enhanced import analyzer plugin."""
    return EnhancedImportAnalyzerPlugin()