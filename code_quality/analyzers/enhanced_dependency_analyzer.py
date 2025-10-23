from src.utils.tprint import tprint

from typing import Dict, List, Any, Optional
"""
Enhanced Dependency Analyzer

Combines FawltyDeps and Creosote for comprehensive dependency analysis.
Replaces the basic import analysis in the pipeline.
"""

import json
import time
from pathlib import Path

from plugins.fawltydeps_analyzer import FawltyDepsAnalyzer
from plugins.creosote_analyzer import CreosoteAnalyzer
from plugins.base_plugin import PluginContext, PluginResult


class EnhancedDependencyAnalyzer:
    """
    Enhanced dependency analyzer using FawltyDeps and Creosote.
    
    This analyzer provides comprehensive dependency analysis by combining:
    - FawltyDeps: Identifies undeclared and unused dependencies
    - Creosote: Identifies unused dependencies with virtual environment awareness
    """
    
    def __init__(self, project_root: str, configuration: Optional[Dict[str, Any]] = None):
        self.project_root = Path(project_root)
        self.configuration = configuration or {}
        
        # Initialize plugins
        self.fawltydeps_plugin = FawltyDepsAnalyzer(
            configuration=self.configuration.get("fawltydeps", {})
        )
        self.creosote_plugin = CreosoteAnalyzer(
            configuration=self.configuration.get("creosote", {})
        )
        
        # Results storage
        self.results = {
            "fawltydeps": None,
            "creosote": None,
            "combined": {},
            "summary": {}
        }
    
    def analyze_project(self) -> Dict[str, Any]:
        """
        Analyze project dependencies using both FawltyDeps and Creosote.
        
        Returns:
            Dict containing comprehensive dependency analysis results
        """
        tprint("\n" + "="*60)
        tprint("Running Enhanced Dependency Analysis")
        tprint("="*60)
        
        start_time = time.time()
        
        # Create plugin context
        context = self._create_plugin_context()
        
        # Run FawltyDeps analysis
        if self.fawltydeps_plugin.is_available():
            tprint("Running FawltyDeps analysis...")
            self.results["fawltydeps"] = self.fawltydeps_plugin.execute(context)
        else:
            tprint("Warning: FawltyDeps not available, skipping...")
            self.results["fawltydeps"] = self._create_unavailable_result("fawltydeps")
        
        # Run Creosote analysis
        if self.creosote_plugin.is_available():
            tprint("Running Creosote analysis...")
            self.results["creosote"] = self.creosote_plugin.execute(context)
        else:
            tprint("Warning: Creosote not available, skipping...")
            self.results["creosote"] = self._create_unavailable_result("creosote")
        
        # Combine results
        self.results["combined"] = self._combine_results()
        
        # Generate summary
        self.results["summary"] = self._generate_summary(time.time() - start_time)
        
        return self.results
    
    def _create_plugin_context(self) -> PluginContext:
        """Create plugin context for analysis."""
        # Get all Python files in project
        python_files = list(self.project_root.rglob("*.py"))
        
        return PluginContext(
            project_root=self.project_root,
            target_files=python_files,
            configuration=self.configuration,
            timeout=300,  # 5 minutes timeout
            dry_run=False,
            verbose=True
        )
    
    def _create_unavailable_result(self, plugin_name: str) -> PluginResult:
        """Create a result for unavailable plugins."""
        return PluginResult(
            plugin_name=plugin_name,
            success=False,
            execution_time=0.0
        )
    
    def _combine_results(self) -> Dict[str, Any]:
        """Combine results from both plugins."""
        combined = {
            "undeclared_deps": set(),
            "unused_deps": set(),
            "all_issues": [],
            "dependency_files": [],
            "analysis_coverage": {}
        }
        
        # Process FawltyDeps results
        if self.results["fawltydeps"] and self.results["fawltydeps"].success:
            fawltydeps_data = self.results["fawltydeps"].output_data
            
            # Add undeclared dependencies
            undeclared = fawltydeps_data.get("undeclared_deps", [])
            combined["undeclared_deps"].update(undeclared)
            
            # Add unused dependencies
            unused = fawltydeps_data.get("unused_deps", [])
            combined["unused_deps"].update(unused)
            
            # Add to all issues
            for dep in undeclared:
                combined["all_issues"].append({
                    "dependency": dep,
                    "issue_type": "undeclared",
                    "source": "fawltydeps",
                    "severity": "high"
                })
            
            for dep in unused:
                combined["all_issues"].append({
                    "dependency": dep,
                    "issue_type": "unused",
                    "source": "fawltydeps",
                    "severity": "medium"
                })
            
            combined["analysis_coverage"]["fawltydeps"] = True
        
        # Process Creosote results
        if self.results["creosote"] and self.results["creosote"].success:
            creosote_data = self.results["creosote"].output_data
            
            # Add unused dependencies (Creosote focuses on unused)
            unused = creosote_data.get("unused_deps", [])
            combined["unused_deps"].update(unused)
            
            # Add to all issues (avoid duplicates)
            existing_unused = {issue["dependency"] for issue in combined["all_issues"] 
                            if issue["issue_type"] == "unused"}
            
            for dep in unused:
                if dep not in existing_unused:
                    combined["all_issues"].append({
                        "dependency": dep,
                        "issue_type": "unused",
                        "source": "creosote",
                        "severity": "medium"
                    })
            
            combined["analysis_coverage"]["creosote"] = True
        
        # Convert sets to lists for JSON serialization
        combined["undeclared_deps"] = list(combined["undeclared_deps"])
        combined["unused_deps"] = list(combined["unused_deps"])
        
        return combined
    
    def _generate_summary(self, total_time: float) -> Dict[str, Any]:
        """Generate analysis summary."""
        summary = {
            "total_execution_time": total_time,
            "tools_used": [],
            "total_issues": 0,
            "issue_breakdown": {
                "undeclared": 0,
                "unused": 0
            },
            "recommendations": []
        }
        
        # Count tools used
        if self.results["fawltydeps"] and self.results["fawltydeps"].success:
            summary["tools_used"].append("fawltydeps")
        
        if self.results["creosote"] and self.results["creosote"].success:
            summary["tools_used"].append("creosote")
        
        # Count issues
        if self.results["combined"]:
            combined = self.results["combined"]
            summary["total_issues"] = len(combined["all_issues"])
            summary["issue_breakdown"]["undeclared"] = len(combined["undeclared_deps"])
            summary["issue_breakdown"]["unused"] = len(combined["unused_deps"])
        
        # Generate recommendations
        if self.results["combined"]:
            combined = self.results["combined"]
            
            if combined["undeclared_deps"]:
                summary["recommendations"].append({
                    "type": "add_dependencies",
                    "message": f"Add {len(combined['undeclared_deps'])} undeclared dependencies to your dependency files",
                    "dependencies": combined["undeclared_deps"]
                })
            
            if combined["unused_deps"]:
                summary["recommendations"].append({
                    "type": "remove_dependencies",
                    "message": f"Consider removing {len(combined['unused_deps'])} unused dependencies",
                    "dependencies": combined["unused_deps"]
                })
        
        return summary
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive dependency analysis report."""
        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "project_root": str(self.project_root),
            "analysis_results": self.results,
            "summary": self.results["summary"],
            "recommendations": self.results["summary"].get("recommendations", [])
        }
    
    def save_report(self, output_dir: Path, filename: str = "enhanced_dependency_analysis") -> Dict[str, Path]:
        """Save analysis report to files."""
        output_dir.mkdir(exist_ok=True)
        
        # Generate report
        report = self.generate_report()
        
        # Convert PluginResult objects to dictionaries for JSON serialization
        def convert_plugin_results(obj):
            if hasattr(obj, 'to_dict'):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {k: convert_plugin_results(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_plugin_results(item) for item in obj]
            else:
                return obj
        
        report_serializable = convert_plugin_results(report)
        
        # Save JSON report
        json_path = output_dir / f"{filename}.json"
        with open(json_path, "w") as f:
            json.dump(report_serializable, f, indent=2)
        
        # Save markdown report
        md_path = output_dir / f"{filename}.md"
        with open(md_path, "w") as f:
            f.write(self._generate_markdown_report(report))
        
        return {
            "json": json_path,
            "markdown": md_path
        }
    
    def _generate_markdown_report(self, report: Dict[str, Any]) -> str:
        """Generate markdown report."""
        md = []
        md.append("# Enhanced Dependency Analysis Report")
        md.append(f"**Generated:** {report['timestamp']}")
        md.append(f"**Project:** {report['project_root']}")
        md.append("")
        
        # Summary
        summary = report["summary"]
        md.append("## Summary")
        md.append(f"- **Total Issues:** {summary['total_issues']}")
        md.append(f"- **Tools Used:** {', '.join(summary['tools_used'])}")
        md.append(f"- **Execution Time:** {summary['total_execution_time']:.2f} seconds")
        md.append("")
        
        # Issue breakdown
        md.append("## Issue Breakdown")
        breakdown = summary["issue_breakdown"]
        md.append(f"- **Undeclared Dependencies:** {breakdown['undeclared']}")
        md.append(f"- **Unused Dependencies:** {breakdown['unused']}")
        md.append("")
        
        # Detailed results
        if report["analysis_results"]["combined"]:
            combined = report["analysis_results"]["combined"]
            
            if combined["undeclared_deps"]:
                md.append("## Undeclared Dependencies")
                md.append("These dependencies are imported but not declared in your dependency files:")
                for dep in combined["undeclared_deps"]:
                    md.append(f"- `{dep}`")
                md.append("")
            
            if combined["unused_deps"]:
                md.append("## Unused Dependencies")
                md.append("These dependencies are declared but not used in your code:")
                for dep in combined["unused_deps"]:
                    md.append(f"- `{dep}`")
                md.append("")
        
        # Recommendations
        if summary["recommendations"]:
            md.append("## Recommendations")
            for rec in summary["recommendations"]:
                md.append(f"### {rec['type'].replace('_', ' ').title()}")
                md.append(rec["message"])
                if "dependencies" in rec:
                    for dep in rec["dependencies"]:
                        md.append(f"- `{dep}`")
                md.append("")
        
        return "\n".join(md)