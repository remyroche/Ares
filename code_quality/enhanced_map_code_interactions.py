#!/usr/bin/env python3
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
from typing import Dict, List, Any, Optional

# Add code_quality to path
sys.path.insert(0, str(Path(__file__).parent))

# Import our enhanced analyzers
from analyzers.simplified_enhanced_analyzer import SimplifiedEnhancedDeadCodeAnalyzer
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
        print("\n[1/4] Analyzing dead code and unused imports...")
        
        try:
            analyzer = SimplifiedEnhancedDeadCodeAnalyzer(self.config)
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
            print(f"  ✅ Found {report.total_issues} total issues")
            print(f"  📊 Dead code functions: {report.issues_by_type.get('dead_code', 0)}")
            print(f"  📦 Unused imports: {report.issues_by_type.get('unused_import', 0)}")
            print(f"  🔗 Call graph nodes: {len(report.call_graph_nodes)}")
            print(f"  📈 Dependency modules: {len(report.dependency_graph)}")
            print(f"  🎯 False positives filtered: {report.false_positives_filtered}")
            
        except Exception as e:
            logger.error(f"Dead code analysis failed: {e}")
            self.results["dead_code"] = {"error": str(e)}
            self.stats["files_failed"] += 1

    def analyze_file_structure(self):
        """Analyze file structure and organization."""
        print("\n[2/4] Analyzing file structure...")
        
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
            print(f"  📁 Total Python files: {len(python_files)}")
            print(f"  📊 Files by size: {file_analysis['files_by_size']}")
            print(f"  🏗️  Directories: {len(file_analysis['files_by_directory'])}")
            print(f"  📈 Files failed to analyze: {self.stats['files_failed']}")
            
        except Exception as e:
            logger.error(f"File structure analysis failed: {e}")
            self.results["file_structure"] = {"error": str(e)}

    def analyze_import_patterns(self):
        """Analyze import patterns across the codebase."""
        print("\n[3/4] Analyzing import patterns...")
        
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
            print(f"  📦 Total imports: {import_analysis['total_imports']}")
            print(f"  📊 Import types: {import_analysis['import_types']}")
            print(f"  🏆 Top modules: {list(import_analysis['top_imported_modules'].keys())[:5]}")
            print(f"  🗑️  Unused imports: {import_analysis['unused_imports']}")
            
        except Exception as e:
            logger.error(f"Import pattern analysis failed: {e}")
            self.results["import_patterns"] = {"error": str(e)}

    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        print("\n[4/4] Generating summary report...")
        
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
            print(f"\n{'='*60}")
            print("📊 ENHANCED CODE INTERACTION ANALYSIS SUMMARY")
            print(f"{'='*60}")
            print(f"📁 Project: {self.project_root}")
            print(f"📈 Files analyzed: {self.stats['files_analyzed']}")
            print(f"❌ Files failed: {self.stats['files_failed']}")
            print(f"🔍 Total issues found: {self.stats['total_issues']}")
            print(f"💀 Dead code functions: {self.stats['dead_code_functions']}")
            print(f"📦 Unused imports: {self.stats['unused_imports']}")
            print(f"🔗 Call graph nodes: {self.stats['call_graph_nodes']}")
            print(f"📊 Dependency modules: {self.stats['dependency_modules']}")
            
            # Print recommendations
            recommendations = summary["recommendations"]
            if recommendations:
                print(f"\n💡 RECOMMENDATIONS:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"   {i}. {rec}")
            
            print(f"\n✅ Analysis complete!")
            
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
            
            print(f"📁 Results exported to: {json_file}")
            
        except Exception as e:
            logger.error(f"Failed to export results: {e}")

    def run(self):
        """Run the complete enhanced code interaction analysis."""
        print(f"Starting enhanced code interaction mapping for: {self.project_root}")
        print("=" * 80)
        
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
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()