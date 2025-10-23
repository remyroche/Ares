#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Simplified Code Interaction Mapping Script

This is a refactored version of the original map_code_interactions.py that:
- Separates concerns into different modules
- Uses composition instead of doing everything in one class
- Reduces complexity from 675 to manageable levels
- Maintains all original functionality
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add code_quality to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our new modular components
from analyzers.dependency_analyzer import DependencyAnalyzer
from analyzers.call_graph_analyzer import CallGraphAnalyzer
from analyzers.architecture_analyzer import ArchitectureAnalyzer
from analyzers.import_analyzer import ImportAnalyzer
from analyzers.complexity_analyzer import ComplexityAnalyzer
from analyzers.enhanced_dead_code_analyzer import EnhancedDeadCodeAnalyzer
from reporters.html_reporter import HTMLReporter
from reporters.text_reporter import TextReporter
from visualizers.chart_generator import ChartGenerator
from core.config import AnalysisConfig
import time


class SimplifiedCodeInteractionMapper:
    """Simplified code interaction mapper using composition."""
    
    def __init__(self, project_root: str):
        """Initialize the mapper with all required components."""
        self.project_root = Path(project_root)
        self.config = AnalysisConfig()
        
        # Initialize analyzers
        self.dependency_analyzer = DependencyAnalyzer(self.config)
        self.call_graph_analyzer = CallGraphAnalyzer(self.config)
        self.architecture_analyzer = ArchitectureAnalyzer(self.config)
        self.import_analyzer = ImportAnalyzer(self.config)
        self.complexity_analyzer = ComplexityAnalyzer(self.config)
        self.dead_code_analyzer = EnhancedDeadCodeAnalyzer(self.config)
        
        # Initialize reporters
        self.html_reporter = HTMLReporter()
        self.text_reporter = TextReporter()
        
        # Results storage
        self.results = {}
        self.stats = {
            "files_analyzed": 0,
            "files_failed": 0,
            "total_issues": 0,
            "dead_code_functions": 0,
            "unused_imports": 0,
            "call_graph_nodes": 0
        }
    
    def run_analysis(self):
        """Run the complete analysis pipeline."""
        tprint(f"Starting simplified code interaction mapping for: {self.project_root}")
        tprint("=" * 80)
        
        try:
            # Run dead code analysis (main focus)
            self._analyze_dead_code()
            
            # Run other analyses if needed
            self._analyze_dependencies()
            self._analyze_call_graph()
            self._analyze_architecture()
            self._analyze_imports()
            self._analyze_complexity()
            
            # Generate reports
            self._generate_reports()
            
            tprint("\n" + "=" * 80)
            tprint("SIMPLIFIED CODE INTERACTION MAPPING COMPLETE!")
            tprint("=" * 80)
            self._print_summary()
            
            return {
                "project_root": str(self.project_root),
                "stats": self.stats,
                "results": self.results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            tprint(f"❌ Analysis failed: {e}")
            return {
                "project_root": str(self.project_root),
                "stats": self.stats,
                "results": {},
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _analyze_dead_code(self):
        """Analyze dead code using the enhanced analyzer."""
        tprint("\n[1/6] Analyzing dead code and deprecated patterns...")
        
        try:
            report = self.dead_code_analyzer.analyze_directory(self.project_root)
            
            # Store results
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
            
            tprint(f"  ✅ Enhanced analysis complete:")
            tprint(f"     - Total issues found: {report.total_issues}")
            tprint(f"     - Dead code functions: {report.issues_by_type.get('dead_code', 0)}")
            tprint(f"     - Unused imports: {report.issues_by_type.get('unused_import', 0)}")
            tprint(f"     - False positives filtered: {report.false_positives_filtered}")
            
        except Exception as e:
            tprint(f"  ❌ Dead code analysis failed: {e}")
            self.results["dead_code"] = {"error": str(e)}
            self.stats["files_failed"] += 1
    
    def _analyze_dependencies(self):
        """Analyze module dependencies."""
        tprint("\n[2/6] Analyzing module dependencies...")
        try:
            self.results["dependencies"] = self.dependency_analyzer.analyze_directory(str(self.project_root))
            deps = self.results["dependencies"]
            tprint(f"  - Found {len(deps.get('modules', {}))} modules")
            tprint(f"  - Total dependencies: {deps.get('total_dependencies', 0)}")
        except Exception as e:
            tprint(f"  ❌ Dependency analysis failed: {e}")
            self.results["dependencies"] = {"error": str(e)}
    
    def _analyze_call_graph(self):
        """Analyze function call relationships."""
        tprint("\n[3/6] Analyzing function call graph...")
        try:
            self.results["call_graph"] = self.call_graph_analyzer.analyze_directory(str(self.project_root))
            cg = self.results["call_graph"]
            tprint(f"  - Found {len(cg.get('functions', {}))} functions")
            tprint(f"  - Total function calls: {cg.get('total_calls', 0)}")
        except Exception as e:
            tprint(f"  ❌ Call graph analysis failed: {e}")
            self.results["call_graph"] = {"error": str(e)}
    
    def _analyze_architecture(self):
        """Analyze system architecture."""
        tprint("\n[4/6] Analyzing system architecture...")
        try:
            self.results["architecture"] = self.architecture_analyzer.analyze_directory(str(self.project_root))
            arch = self.results["architecture"]
            tprint(f"  - Found {len(arch.get('components', {}))} components")
        except Exception as e:
            tprint(f"  ❌ Architecture analysis failed: {e}")
            self.results["architecture"] = {"error": str(e)}
    
    def _analyze_imports(self):
        """Analyze import relationships."""
        tprint("\n[5/6] Analyzing import relationships...")
        try:
            self.results["imports"] = self.import_analyzer.analyze_directory(str(self.project_root))
            imps = self.results["imports"]
            tprint(f"  - Total imports: {imps.get('total_imports', 0)}")
        except Exception as e:
            tprint(f"  ❌ Import analysis failed: {e}")
            self.results["imports"] = {"error": str(e)}
    
    def _analyze_complexity(self):
        """Analyze code complexity."""
        tprint("\n[6/6] Analyzing code complexity...")
        try:
            self.results["complexity"] = self.complexity_analyzer.analyze_directory(str(self.project_root))
            comp = self.results["complexity"]
            tprint(f"  - Average complexity: {comp.get('average_complexity', 0):.2f}")
        except Exception as e:
            tprint(f"  ❌ Complexity analysis failed: {e}")
            self.results["complexity"] = {"error": str(e)}
    
    def _generate_reports(self):
        """Generate all reports."""
        tprint("\n[7/7] Generating reports...")
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("code_quality/simplified_analysis_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate JSON report
        json_file = output_dir / f"simplified_analysis_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        tprint(f"  📁 JSON report saved to: {json_file}")
        
        # Generate HTML report
        html_content = self.html_reporter.generate_from_analyzer_results(
            self.results, 
            title="Simplified Code Interaction Analysis"
        )
        html_file = output_dir / f"simplified_analysis_{timestamp}.html"
        with open(html_file, 'w') as f:
            f.write(html_content)
        tprint(f"  📁 HTML report saved to: {html_file}")
        
        # Generate text report
        text_content = self.text_reporter.generate_detailed_report(self.results)
        text_file = output_dir / f"simplified_analysis_{timestamp}.txt"
        with open(text_file, 'w') as f:
            f.write(text_content)
        tprint(f"  📁 Text report saved to: {text_file}")
    
    def _print_summary(self):
        """Print analysis summary."""
        tprint(f"\n📊 Analysis Summary:")
        tprint(f"   - Total issues found: {self.stats['total_issues']}")
        tprint(f"   - Dead code functions: {self.stats['dead_code_functions']}")
        tprint(f"   - Unused imports: {self.stats['unused_imports']}")
        tprint(f"   - Call graph nodes: {self.stats['call_graph_nodes']}")
        tprint(f"   - Files analyzed: {self.stats['files_analyzed']}")


def main():
    """Main entry point for simplified code interaction mapping."""
    parser = argparse.ArgumentParser(
        description="Simplified Code Interaction Mapping - Dead Code Analysis and Dependency Mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze current workspace
  python map_code_interactions_simplified.py
  
  # Analyze specific project
  python map_code_interactions_simplified.py --project-root /path/to/project
        """
    )
    
    parser.add_argument("--project-root", default="/workspace",
                       help="Root directory of the project to analyze")
    parser.add_argument("--output", default="simplified_code_interactions_report.json",
                       help="Output file for the analysis report")

    args = parser.parse_args()

    mapper = SimplifiedCodeInteractionMapper(args.project_root)
    results = mapper.run_analysis()
    
    # Save results to file
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    tprint(f"📊 Analysis complete! Report saved to: {args.output}")


if __name__ == "__main__":
    main()
