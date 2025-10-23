#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Enhanced Code Interaction Mapping Script

This script systematically maps interactions within the codebase using:
- Enhanced dead code analysis with multiple tools
- Robust AST parsing with error handling
- Call graph analysis with NetworkX
- Dependency analysis for module relationships
- Architecture analysis for system structure
- Import analysis for module dependencies

ENHANCED FEATURES:
1. Multi-Tool Dead Code Detection:
   - Enhanced AST analysis with better heuristics
   - Import analysis for unused imports
   - Cross-validation to reduce false positives
   - Tool attribution for each issue

2. Robust AST Parsing:
   - Handles syntax errors gracefully
   - Continues analysis even with problematic files
   - Provides detailed error reporting
   - Skips unparseable files without failing

3. Comprehensive Reporting:
   - JSON export with structured data
   - HTML reports with visualizations
   - Confidence scores and severity levels
   - Call graph and dependency visualization

4. Simplified Architecture:
   - Cleaner, more maintainable code
   - Better error handling and logging
   - Modular design with clear separation of concerns
"""

import argparse
import json
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import List

# Add code_quality to path
sys.path.insert(0, str(Path(__file__).parent))

# Import our enhanced analyzers
from analyzers.enhanced_dead_code_analyzer import EnhancedDeadCodeAnalyzer
import numpy as np

import time

from core.config import AnalysisConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedCodeInteractionMapper:
    """Enhanced code interaction mapper with robust analysis capabilities."""

    def __init__(self, project_root: str, exclude_dirs: List[str] = None):
        """Initialize the enhanced mapper."""
        self.project_root = Path(project_root)
        self.exclude_dirs = exclude_dirs or ["venv", "__pycache__", ".git", "node_modules", ".pytest_cache"]
        self.config = AnalysisConfig()
        self.results = {}
        self.stats = {
            "files_analyzed": 0,
            "files_failed": 0,
            "total_issues": 0,
            "dead_code_functions": 0,
            "unused_imports": 0,
            "call_graph_nodes": 0,
            "dependency_modules": 0
        }

    def find_python_files(self) -> List[Path]:
        """Find all Python files in the project, excluding specified directories."""
        python_files = []
        
        for py_file in self.project_root.rglob("*.py"):
            # Skip files in excluded directories
            if any(excluded in py_file.parts for excluded in self.exclude_dirs):
                continue
            python_files.append(py_file)
        
        return python_files

    def analyze_dead_code(self):
        """Analyze dead code using enhanced analyzer."""
        tprint("\n[1/4] Analyzing dead code and unused imports...")
        
        try:
            analyzer = EnhancedDeadCodeAnalyzer(self.config)
            report = analyzer.analyze_directory(self.project_root)
            
            self.results["dead_code"] = {
                "total_issues": report.total_issues,
                "issues_by_type": report.issues_by_type,
                "issues_by_severity": {k: len(v) for k, v in report.issues_by_severity.items()},
                "issues_by_tool": {k: len(v) for k, v in report.issues_by_tool.items()},
                "confidence_distribution": report.confidence_distribution,
                "call_graph_nodes": len(report.call_graph_nodes),
                "dependency_graph": len(report.dependency_graph),
                "false_positives_filtered": report.false_positives_filtered,
                "impact_analysis": report.impact_analysis
            }
            
            # Update stats
            self.stats["total_issues"] = report.total_issues
            self.stats["dead_code_functions"] = report.issues_by_type.get("dead_code", 0)
            self.stats["unused_imports"] = report.issues_by_type.get("unused_import", 0)
            self.stats["call_graph_nodes"] = len(report.call_graph_nodes)
            self.stats["dependency_modules"] = len(report.dependency_graph)
            
            # Print summary
            tprint(f"  ✅ Found {report.total_issues} total issues")
            tprint(f"  📊 Dead code functions: {report.issues_by_type.get('dead_code', 0)}")
            tprint(f"  📦 Unused imports: {report.issues_by_type.get('unused_import', 0)}")
            tprint(f"  🔗 Call graph nodes: {len(report.call_graph_nodes)}")
            tprint(f"  📈 Dependency modules: {len(report.dependency_graph)}")
            tprint(f"  🎯 False positives filtered: {report.false_positives_filtered}")
            
        except Exception as e:
            logger.error(f"Dead code analysis failed: {e}")
            self.results["dead_code"] = {"error": str(e)}
            self.stats["files_failed"] += 1

    def analyze_file_structure(self):
        """Analyze file structure and organization."""
        tprint("\n[2/4] Analyzing file structure...")
        
        try:
            python_files = self.find_python_files()
            self.stats["files_analyzed"] = len(python_files)
            
            # Analyze file structure
            file_analysis = {
                "total_files": len(python_files),
                "files_by_directory": {},
                "files_by_size": {"small": 0, "medium": 0, "large": 0},
                "import_patterns": {},
                "class_distribution": {},
                "function_distribution": {}
            }
            
            for py_file in python_files:
                try:
                    # Directory analysis
                    dir_name = str(py_file.parent.relative_to(self.project_root))
                    if dir_name not in file_analysis["files_by_directory"]:
                        file_analysis["files_by_directory"][dir_name] = 0
                    file_analysis["files_by_directory"][dir_name] += 1
                    
                    # Size analysis
                    file_size = py_file.stat().st_size
                    if file_size < 1000:
                        file_analysis["files_by_size"]["small"] += 1
                    elif file_size < 10000:
                        file_analysis["files_by_size"]["medium"] += 1
                    else:
                        file_analysis["files_by_size"]["large"] += 1
                    
                    # Basic content analysis
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Count classes and functions
                    class_count = content.count('class ')
                    function_count = content.count('def ')
                    
                    if class_count > 0:
                        size_key = "large" if class_count > 5 else "medium" if class_count > 2 else "small"
                        if size_key not in file_analysis["class_distribution"]:
                            file_analysis["class_distribution"][size_key] = 0
                        file_analysis["class_distribution"][size_key] += 1
                    
                    if function_count > 0:
                        size_key = "large" if function_count > 20 else "medium" if function_count > 10 else "small"
                        if size_key not in file_analysis["function_distribution"]:
                            file_analysis["function_distribution"][size_key] = 0
                        file_analysis["function_distribution"][size_key] += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to analyze {py_file}: {e}")
                    self.stats["files_failed"] += 1
            
            self.results["file_structure"] = file_analysis
            
            # Print summary
            tprint(f"  📁 Total Python files: {len(python_files)}")
            tprint(f"  📊 Files by size: {file_analysis['files_by_size']}")
            tprint(f"  🏗️  Directories: {len(file_analysis['files_by_directory'])}")
            tprint(f"  📈 Files failed to analyze: {self.stats['files_failed']}")
            
        except Exception as e:
            logger.error(f"File structure analysis failed: {e}")
            self.results["file_structure"] = {"error": str(e)}

    def analyze_import_patterns(self):
        """Analyze import patterns across the codebase."""
        tprint("\n[3/4] Analyzing import patterns...")
        
        try:
            python_files = self.find_python_files()
            import_analysis = {
                "total_imports": 0,
                "import_types": {"standard": 0, "third_party": 0, "local": 0},
                "most_imported_modules": {},
                "circular_imports": [],
                "unused_imports": 0
            }
            
            # Common standard library modules
            stdlib_modules = {
                'os', 'sys', 'json', 'datetime', 'pathlib', 'typing', 'collections',
                'itertools', 'functools', 'logging', 're', 'math', 'random',
                'subprocess', 'threading', 'multiprocessing', 'asyncio', 'time'
            }
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    for line_num, line in enumerate(lines, 1):
                        line = line.strip()
                        if line.startswith(('import ', 'from ')):
                            import_analysis["total_imports"] += 1
                            
                            # Categorize import type
                            if line.startswith('from '):
                                module = line.split()[1].split('.')[0]
                            else:
                                module = line.split()[1].split('.')[0]
                            
                            if module in stdlib_modules:
                                import_analysis["import_types"]["standard"] += 1
                            elif '.' in module or module.startswith('_'):
                                import_analysis["import_types"]["local"] += 1
                            else:
                                import_analysis["import_types"]["third_party"] += 1
                            
                            # Track most imported modules
                            if module not in import_analysis["most_imported_modules"]:
                                import_analysis["most_imported_modules"][module] = 0
                            import_analysis["most_imported_modules"][module] += 1
                            
                except Exception as e:
                    logger.warning(f"Failed to analyze imports in {py_file}: {e}")
            
            # Get top imported modules
            top_imports = sorted(
                import_analysis["most_imported_modules"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            import_analysis["top_imported_modules"] = dict(top_imports)
            
            # Get unused imports from dead code analysis
            if "dead_code" in self.results and "issues_by_type" in self.results["dead_code"]:
                import_analysis["unused_imports"] = self.results["dead_code"]["issues_by_type"].get("unused_import", 0)
            
            self.results["import_patterns"] = import_analysis
            
            # Print summary
            tprint(f"  📦 Total imports: {import_analysis['total_imports']}")
            tprint(f"  📊 Import types: {import_analysis['import_types']}")
            tprint(f"  🏆 Top modules: {list(import_analysis['top_imported_modules'].keys())[:5]}")
            tprint(f"  🗑️  Unused imports: {import_analysis['unused_imports']}")
            
        except Exception as e:
            logger.error(f"Import pattern analysis failed: {e}")
            self.results["import_patterns"] = {"error": str(e)}

    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        tprint("\n[4/4] Generating summary report...")
        
        try:
            summary = {
                "analysis_timestamp": datetime.now().isoformat(),
                "project_root": str(self.project_root),
                "excluded_directories": self.exclude_dirs,
                "statistics": self.stats,
                "results_summary": {
                    "dead_code_analysis": self.results.get("dead_code", {}),
                    "file_structure": self.results.get("file_structure", {}),
                    "import_patterns": self.results.get("import_patterns", {})
                },
                "recommendations": self._generate_recommendations()
            }
            
            self.results["summary"] = summary
            
            # Print final summary
            tprint(f"\n{'='*60}")
            tprint("📊 ENHANCED CODE INTERACTION ANALYSIS SUMMARY")
            tprint(f"{'='*60}")
            tprint(f"📁 Project: {self.project_root}")
            tprint(f"📈 Files analyzed: {self.stats['files_analyzed']}")
            tprint(f"❌ Files failed: {self.stats['files_failed']}")
            tprint(f"🔍 Total issues found: {self.stats['total_issues']}")
            tprint(f"💀 Dead code functions: {self.stats['dead_code_functions']}")
            tprint(f"📦 Unused imports: {self.stats['unused_imports']}")
            tprint(f"🔗 Call graph nodes: {self.stats['call_graph_nodes']}")
            tprint(f"📊 Dependency modules: {self.stats['dependency_modules']}")
            
            # Print recommendations
            recommendations = summary["recommendations"]
            if recommendations:
                tprint(f"\n💡 RECOMMENDATIONS:")
                for i, rec in enumerate(recommendations, 1):
                    tprint(f"   {i}. {rec}")
            
            tprint(f"\n✅ Analysis complete!")
            
        except Exception as e:
            logger.error(f"Summary report generation failed: {e}")

    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on analysis results."""
        recommendations = []
        
        # Dead code recommendations
        if self.stats["dead_code_functions"] > 50:
            recommendations.append(f"High number of dead code functions ({self.stats['dead_code_functions']}). Consider removing unused functions to improve maintainability.")
        
        if self.stats["unused_imports"] > 20:
            recommendations.append(f"Many unused imports ({self.stats['unused_imports']}). Clean up imports to reduce clutter and improve performance.")
        
        # File structure recommendations
        if self.stats["files_failed"] > 0:
            recommendations.append(f"Some files failed to analyze ({self.stats['files_failed']}). Check for syntax errors or encoding issues.")
        
        # Call graph recommendations
        if self.stats["call_graph_nodes"] > 100:
            recommendations.append("Large call graph detected. Consider breaking down complex modules into smaller, more focused components.")
        
        # General recommendations
        if self.stats["total_issues"] > 100:
            recommendations.append("High number of issues found. Prioritize addressing high-confidence dead code and unused imports first.")
        
        if not recommendations:
            recommendations.append("Codebase appears to be in good shape! Continue monitoring for dead code and unused imports.")
        
        return recommendations

    def export_results(self, output_dir: Path):
        """Export results to JSON and generate reports."""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Export JSON results
            json_file = output_dir / f"enhanced_code_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(json_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            tprint(f"📁 Results exported to: {json_file}")
            
        except Exception as e:
            logger.error(f"Failed to export results: {e}")

    def run(self):
        """Run the complete enhanced code interaction analysis."""
        tprint(f"Starting enhanced code interaction mapping for: {self.project_root}")
        tprint("=" * 80)
        
        try:
            # Run all analysis phases
            self.analyze_dead_code()
            self.analyze_file_structure()
            self.analyze_import_patterns()
            self.generate_summary_report()
            
            # Export results
            output_dir = Path("/workspace/code_quality/enhanced_analysis_output")
            self.export_results(output_dir)
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise

    def map_interactions(self, project_root: str) -> dict:
        """Map enhanced code interactions across the project."""
        tprint(f"\n{'='*60}")
        tprint("ENHANCED CODE INTERACTION MAPPING")
        tprint(f"{'='*60}")
        tprint(f"Project root: {project_root}")
        
        # Initialize enhanced results
        interactions = {
            "interactions": [],
            "module_dependencies": [],
            "function_calls": [],
            "class_interactions": [],
            "import_relationships": [],
            "complex_interactions": [],
            "cross_module_interactions": [],
            "call_graph": {},
            "dependency_graph": {},
            "enhanced_metrics": {
                "total_interactions": 0,
                "function_calls": 0,
                "class_interactions": 0,
                "module_dependencies": 0,
                "complex_interactions": 0,
                "cross_module_interactions": 0,
                "files_analyzed": 0,
                "complexity_score": 0.0,
                "coupling_score": 0.0
            }
        }
        
        try:
            # Import analyzers for enhanced analysis
            from analyzers.call_graph_analyzer import CallGraphAnalyzer
            from analyzers.dependency_analyzer import DependencyAnalyzer
            from analyzers.architecture_analyzer import ArchitectureAnalyzer
            
            # Initialize analyzers with default config
            from core.config import get_default_config
            config = get_default_config()
            call_analyzer = CallGraphAnalyzer(config)
            dep_analyzer = DependencyAnalyzer(config)
            arch_analyzer = ArchitectureAnalyzer(config)
            
            # Run enhanced analysis
            tprint("Running enhanced call graph analysis...")
            call_results = call_analyzer.analyze_directory(project_root)
            
            tprint("Running enhanced dependency analysis...")
            dep_results = dep_analyzer.analyze_directory(project_root)
            
            tprint("Running enhanced architecture analysis...")
            arch_results = arch_analyzer.analyze_directory(project_root)
            
            # Extract function calls with enhanced metrics
            if "functions" in call_results:
                for func_name, func_data in call_results["functions"].items():
                    calls = func_data.get("calls", [])
                    for call in calls:
                        interaction = {
                            "type": "function_call",
                            "source": func_name,
                            "target": call,
                            "source_file": func_data.get("file_path", ""),
                            "line_number": func_data.get("line_number", 0),
                            "complexity": len(calls),  # Number of calls as complexity metric
                            "is_cross_module": "/" in call or "." in call
                        }
                        interactions["interactions"].append(interaction)
                        interactions["function_calls"].append(interaction)
                        
                        # Categorize as complex if high call count
                        if len(calls) > 10:
                            interactions["complex_interactions"].append(interaction)
                        
                        # Categorize as cross-module if external
                        if interaction["is_cross_module"]:
                            interactions["cross_module_interactions"].append(interaction)
            
            # Extract module dependencies with enhanced metrics
            if "modules" in dep_results:
                for module_name, module_data in dep_results["modules"].items():
                    dependencies = module_data.get("dependencies", [])
                    for dep in dependencies:
                        interaction = {
                            "type": "module_dependency",
                            "source": module_name,
                            "target": dep,
                            "relationship": "imports",
                            "source_file": module_data.get("file_path", ""),
                            "complexity": len(dependencies),  # Number of dependencies as complexity
                            "is_external": not dep.startswith("src/") and not dep.startswith(".")
                        }
                        interactions["interactions"].append(interaction)
                        interactions["module_dependencies"].append(interaction)
                        
                        # Categorize as complex if high dependency count
                        if len(dependencies) > 15:
                            interactions["complex_interactions"].append(interaction)
                        
                        # Categorize as cross-module if external
                        if interaction["is_external"]:
                            interactions["cross_module_interactions"].append(interaction)
            
            # Extract class interactions from architecture with enhanced metrics
            if "components" in arch_results:
                for component_name, component_data in arch_results["components"].items():
                    component_interactions = component_data.get("interactions", [])
                    for interaction in component_interactions:
                        interaction["type"] = "class_interaction"
                        interaction["source_file"] = component_data.get("file_path", "")
                        interaction["complexity"] = len(component_interactions)
                        interactions["interactions"].append(interaction)
                        interactions["class_interactions"].append(interaction)
                        
                        # Categorize as complex if high interaction count
                        if len(component_interactions) > 5:
                            interactions["complex_interactions"].append(interaction)
            
            # Store enhanced graphs
            interactions["call_graph"] = call_results
            interactions["dependency_graph"] = dep_results
            
            # Calculate enhanced metrics
            interactions["enhanced_metrics"]["total_interactions"] = len(interactions["interactions"])
            interactions["enhanced_metrics"]["function_calls"] = len([i for i in interactions["interactions"] if i["type"] == "function_call"])
            interactions["enhanced_metrics"]["class_interactions"] = len([i for i in interactions["interactions"] if i["type"] == "class_interaction"])
            interactions["enhanced_metrics"]["module_dependencies"] = len([i for i in interactions["interactions"] if i["type"] == "module_dependency"])
            interactions["enhanced_metrics"]["complex_interactions"] = len(interactions["complex_interactions"])
            interactions["enhanced_metrics"]["cross_module_interactions"] = len(interactions["cross_module_interactions"])
            interactions["enhanced_metrics"]["files_analyzed"] = self.stats["files_analyzed"]
            
            # Calculate complexity and coupling scores
            if interactions["enhanced_metrics"]["total_interactions"] > 0:
                interactions["enhanced_metrics"]["complexity_score"] = sum(i.get("complexity", 0) for i in interactions["interactions"]) / interactions["enhanced_metrics"]["total_interactions"]
                interactions["enhanced_metrics"]["coupling_score"] = interactions["enhanced_metrics"]["cross_module_interactions"] / interactions["enhanced_metrics"]["total_interactions"]
            
            # Store enhanced graphs
            interactions["call_graph"] = self.results.get("call_graph", {})
            interactions["dependency_graph"] = self.results.get("dependency_graph", {})
            
            tprint(f"\n✅ Enhanced interaction mapping completed:")
            tprint(f"   - Total interactions: {interactions['enhanced_metrics']['total_interactions']}")
            tprint(f"   - Function calls: {interactions['enhanced_metrics']['function_calls']}")
            tprint(f"   - Class interactions: {interactions['enhanced_metrics']['class_interactions']}")
            tprint(f"   - Module dependencies: {interactions['enhanced_metrics']['module_dependencies']}")
            tprint(f"   - Complex interactions: {interactions['enhanced_metrics']['complex_interactions']}")
            tprint(f"   - Cross-module interactions: {interactions['enhanced_metrics']['cross_module_interactions']}")
            tprint(f"   - Complexity score: {interactions['enhanced_metrics']['complexity_score']:.2f}")
            tprint(f"   - Coupling score: {interactions['enhanced_metrics']['coupling_score']:.2f}")
            tprint(f"   - Files analyzed: {interactions['enhanced_metrics']['files_analyzed']}")
            
            return interactions
            
        except Exception as e:
            tprint(f"❌ Error in enhanced interaction mapping: {e}")
            return {
                "error": str(e),
                "interactions": [],
                "module_dependencies": [],
                "function_calls": [],
                "class_interactions": [],
                "complex_interactions": [],
                "cross_module_interactions": [],
                "enhanced_metrics": {"total_interactions": 0}
            }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Enhanced Code Interaction Mapping")
    parser.add_argument("--project-root", default="/workspace",
                       help="Root directory of the project to analyze")
    parser.add_argument("--exclude", nargs="*", 
                       default=["venv", "__pycache__", ".git", "node_modules", ".pytest_cache"],
                       help="Directories to exclude from analysis")
    parser.add_argument("--output-dir", default="/workspace/code_quality/enhanced_analysis_output",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    try:
        mapper = EnhancedCodeInteractionMapper(args.project_root, args.exclude)
        mapper.run()
    except Exception as e:
        tprint(f"❌ Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())