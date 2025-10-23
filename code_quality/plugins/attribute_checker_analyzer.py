"""
Attribute Checker Analyzer Plugin

Advanced plugin for detecting missing methods and attributes in Python classes.
Uses enhanced attribute access analysis to identify potential issues with reduced false positives.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

# Add paths for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
grandparent_dir = parent_dir.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(grandparent_dir))

from plugins.base_plugin import PluginMetadata, PluginCategory, PluginPriority, BasePlugin
from core.plugins import BaseCodeAnalyzer


class AttributeCheckerAnalyzer(BaseCodeAnalyzer):
    """Enhanced attribute checker plugin for detecting missing methods and attributes."""

    def __init__(self, config: Dict[str, Any] = None):
        self.name = "AttributeChecker"
        self.description = "Advanced attribute access analysis for detecting missing methods and attributes"
        self.version = "2.0.0"
        super().__init__(config)

        # Import attribute checker components
        try:
            from attribute_checker import AttributeChecker, check_file
            self.AttributeChecker = AttributeChecker
            self.check_file = check_file
            self.available = True
        except ImportError as e:
            self.logger.warning(f"Could not import attribute checker: {e}")
            self.available = False

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description

    def get_version(self) -> str:
        return self.version

    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name=self.name,
            version=self.version,
            description=self.description,
            author="Code Quality Pipeline",
            category=PluginCategory.ANALYSIS,
            priority=PluginPriority.HIGH,
            dependencies=[],
            tags={"analysis", "attributes", "methods", "classes", "static-analysis"},
            min_python_version="3.8",
            required_packages=[],
            optional_packages=[],
            configuration_schema={
                "type": "object",
                "properties": {
                    "exclude_common_attrs": {
                        "type": "boolean",
                        "default": True,
                        "description": "Exclude common external attributes"
                    },
                    "severity_filter": {
                        "type": "string",
                        "enum": ["all", "warnings", "errors"],
                        "default": "all",
                        "description": "Filter results by severity level"
                    }
                }
            }
        )

    def can_analyze(self, file_path: str) -> bool:
        """Check if this analyzer can process the given file."""
        if not self.available:
            return False

        return file_path.endswith(".py")

    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Analyze a Python file for attribute access issues."""
        if not self.available:
            return {
                "success": False,
                "tool": "attribute_checker",
                "file": file_path,
                "error": "Attribute checker not available",
                "issues": [],
                "total_issues": 0,
                "stdout": "",
                "stderr": "Attribute checker dependencies not available"
            }

        try:
            # Extract classes from the file
            classes = self._extract_classes_from_file(file_path)

            if not classes:
                return {
                    "success": True,
                    "tool": "attribute_checker",
                    "file": file_path,
                    "message": "No classes found in file",
                    "issues": [],
                    "total_issues": 0,
                    "stdout": "No classes to analyze",
                    "stderr": ""
                }

            all_issues = []
            total_issues = 0

            # Analyze each class in the file
            for class_name in classes:
                result = self.check_file(file_path, class_name)

                if result['status'] == 'success':
                    # Add class information to each issue
                    for issue in result.get('missing_items', []):
                        issue['class'] = class_name
                        issue['file'] = file_path
                        all_issues.append(issue)

                    total_issues += result.get('missing_count', 0)

            # Apply severity filtering
            severity_filter = self.get_config("severity_filter", "all")
            if severity_filter != "all":
                if severity_filter == "errors":
                    all_issues = [issue for issue in all_issues if issue.get('severity') == 'error']
                elif severity_filter == "warnings":
                    all_issues = [issue for issue in all_issues if issue.get('severity') == 'warning']

            return {
                "success": True,
                "tool": "attribute_checker",
                "file": file_path,
                "message": f"Found {len(all_issues)} potential attribute access issues",
                "issues": all_issues,
                "total_issues": len(all_issues),
                "stdout": f"Analyzed {len(classes)} classes, found {len(all_issues)} issues",
                "stderr": "",
                "analysis_details": {
                    "classes_analyzed": len(classes),
                    "class_names": classes,
                    "severity_filter": severity_filter
                }
            }

        except Exception as e:
            return {
                "success": False,
                "tool": "attribute_checker",
                "file": file_path,
                "error": str(e),
                "issues": [],
                "total_issues": 0,
                "stdout": "",
                "stderr": f"Analysis failed: {e}"
            }

    def _extract_classes_from_file(self, file_path: str) -> List[str]:
        """Extract class names from a Python file."""
        try:
            import ast

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)
            classes = []

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)

            return classes

        except Exception as e:
            self.logger.warning(f"Could not extract classes from {file_path}: {e}")
            return []

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get a summary of what this analyzer does."""
        return {
            "name": self.get_name(),
            "description": self.get_description(),
            "version": self.get_version(),
            "supported_extensions": self.get_supported_extensions(),
            "capabilities": [
                "Detects missing method calls",
                "Identifies undefined attribute access",
                "Provides severity classification",
                "Filters out common external attributes",
                "Supports multiple classes per file"
            ],
            "configuration_options": [
                "exclude_common_attrs: Filter out common external attributes",
                "severity_filter: Filter by error/warning severity"
            ]
        }
