#!/usr/bin/env python3
"""
Code Interaction Mapping Script

This script systematically maps interactions within the codebase using:
- Dependency analysis to understand module relationships
- Call graph analysis to visualize function calls
- Architecture analysis for system structure
- Import analysis for module dependencies
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add code_quality to path
sys.path.insert(0, str(Path(__file__).parent))

from analyzers.architecture_analyzer import ArchitectureAnalyzer as ArchitectureAnalyzer_3
from analyzers.call_graph_analyzer import CallGraphAnalyzer as CallGraphAnalyzer_analyzers_call_graph_analyzer
from analyzers.complexity_analyzer import ComplexityAnalyzer as ComplexityAnalyzer_analyzers_complexity_analyzer
from analyzers.dependency_analyzer import DependencyAnalyzer as DependencyAnalyzer_analyzers_dependency_analyzer
from analyzers.import_analyzer import ImportAnalyzer as ImportAnalyzer_analyzers_import_analyzer
from reporters.html_reporter import HTMLReporter

from core.config import get_default_config


class CodeInteractionMapper:
    """Maps all interactions within a codebase."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.config = get_default_config()
        self.results = {}

    def analyze_dependencies(self):
        """Analyze module dependencies."""
        print("\n[1/5] Analyzing module dependencies...")
        analyzer = DependencyAnalyzer(self.config)
        self.results["dependencies"] = analyzer.analyze_directory(str(self.project_root))

        # Print summary
        deps = self.results["dependencies"]
        print(f"  - Found {len(deps.get('modules', {}))} modules")
        print(f"  - Total dependencies: {sum(len(m.get('dependencies', [])) for m in deps.get('modules', {}).values())}")

    def analyze_call_graph(self):
        """Analyze function call relationships."""
        print("\n[2/5] Analyzing function call graph...")
        analyzer = CallGraphAnalyzer(self.config)
        self.results["call_graph"] = analyzer.analyze_directory(str(self.project_root))

        # Print summary
        cg = self.results["call_graph"]
        print(f"  - Found {len(cg.get('functions', {}))} functions")
        print(f"  - Total function calls: {sum(len(f.get('calls', [])) for f in cg.get('functions', {}).values())}")

    def analyze_architecture(self):
        """Analyze system architecture."""
        print("\n[3/5] Analyzing system architecture...")
        analyzer = ArchitectureAnalyzer(self.config)
        self.results["architecture"] = analyzer.analyze_directory(str(self.project_root))

        # Print summary
        arch = self.results["architecture"]
        print(f"  - Identified {len(arch.get('layers', []))} architectural layers")
        print(f"  - Found {len(arch.get('components', {}))} components")

    def analyze_imports(self):
        """Analyze import relationships."""
        print("\n[4/5] Analyzing import relationships...")
        analyzer = ImportAnalyzer(self.config)
        self.results["imports"] = analyzer.analyze_directory(str(self.project_root))

        # Print summary
        imps = self.results["imports"]
        print(f"  - Total imports: {sum(len(f.get('imports', [])) for f in imps.get('files', {}).values())}")
        print(f"  - Circular imports: {len(imps.get('circular_imports', []))}")

    def analyze_complexity(self):
        """Analyze code complexity for context."""
        print("\n[5/5] Analyzing code complexity...")
        analyzer = ComplexityAnalyzer(self.config)
        self.results["complexity"] = analyzer.analyze_directory(str(self.project_root))

        # Print summary
        comp = self.results["complexity"]
        print(f"  - Average cyclomatic complexity: {comp.get('average_complexity', 0):.2f}")
        print(f"  - Files with high complexity: {len([f for f in comp.get('files', {}).values() if f.get('complexity', 0) > 10])}")

    def generate_interaction_report(self):
        """Generate comprehensive interaction report."""
        print("\n[6/6] Generating interaction reports...")

        # Create reports directory with datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reports_dir = Path("code_quality/visualizers/reports") / f"report_{timestamp}"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  - Output directory: {reports_dir}")

        # Save raw JSON data
        json_file = reports_dir / f"interactions_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"  - Saved raw data: {json_file}")

        # Generate text summary
        summary_file = reports_dir / f"interactions_summary_{timestamp}.txt"
        with open(summary_file, "w") as f:
            f.write("CODE INTERACTION MAPPING SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            # Dependencies section
            f.write("MODULE DEPENDENCIES\n")
            f.write("-" * 40 + "\n")
            deps = self.results.get("dependencies", {})
            for module, info in deps.get("modules", {}).items():
                if info.get("dependencies"):
                    f.write(f"\n{module}:\n")
                    f.writelines(f"  → {dep}\n" for dep in info["dependencies"])

            # Call graph section
            f.write("\n\nFUNCTION CALL RELATIONSHIPS\n")
            f.write("-" * 40 + "\n")
            cg = self.results.get("call_graph", {})
            for func, info in cg.get("functions", {}).items():
                if info.get("calls"):
                    f.write(f"\n{func}:\n")
                    f.writelines(f"  → {call}\n" for call in info["calls"])

            # Architecture section
            f.write("\n\nARCHITECTURAL COMPONENTS\n")
            f.write("-" * 40 + "\n")
            arch = self.results.get("architecture", {})
            for component, info in arch.get("components", {}).items():
                f.write(f"\n{component}:\n")
                f.write(f"  Type: {info.get('type', 'unknown')}\n")
                f.write(f"  Dependencies: {', '.join(info.get('dependencies', []))}\n")

            # Import relationships
            f.write("\n\nIMPORT RELATIONSHIPS\n")
            f.write("-" * 40 + "\n")
            imps = self.results.get("imports", {})
            for file, info in imps.get("files", {}).items():
                if info.get("imports"):
                    f.write(f"\n{file}:\n")
                    f.writelines(f"  ← {imp.get('module', 'unknown')}\n" for imp in info["imports"])

            # Circular imports
            circular = imps.get("circular_imports", [])
            if circular:
                f.write("\n\nCIRCULAR IMPORTS DETECTED\n")
                f.write("-" * 40 + "\n")
                f.writelines(f"  • {' → '.join(cycle)}\n" for cycle in circular)

        print(f"  - Saved summary: {summary_file}")

        # Generate HTML report
        html_reporter = HTMLReporter()
        html_file = reports_dir / f"interactions_{timestamp}.html"
        html_content = html_reporter.generate_from_analyzer_results(
            self.results,
            title="Code Interaction Mapping Report",
        )
        with open(html_file, "w") as f:
            f.write(html_content)
        print(f"  - Saved HTML report: {html_file}")

        # Generate visual diagrams
        try:
            visual_files = self._generate_visual_diagrams(reports_dir, timestamp)
            if visual_files:
                print(f"  - Generated {len(visual_files)} visual diagrams")
        except Exception as e:
            print(f"  - Could not generate visual diagrams: {e}")

        return {
            "json": str(json_file),
            "summary": str(summary_file),
            "html": str(html_file),
            "report_dir": str(reports_dir),
            "timestamp": timestamp
        }

    def _generate_visual_diagrams(self, output_dir: Path, timestamp: str):
        """Generate visual diagrams of interactions using the new visualization system."""
        from visualizers import (
            DependencyGraphVisualizer,
            ComplexityHeatmapVisualizer,
            InteractionNetworkVisualizer,
            DashboardGenerator
        )
        
        # Create visualizers
        dep_viz = DependencyGraphVisualizer(str(output_dir))
        complexity_viz = ComplexityHeatmapVisualizer(str(output_dir))
        network_viz = InteractionNetworkVisualizer(str(output_dir))
        dashboard_gen = DashboardGenerator(str(output_dir))
        
        generated_files = []
        
        # Generate dependency visualizations
        if 'dependencies' in self.results:
            deps = self.results['dependencies'].get('modules', {})
            if deps:
                # Main dependency graph
                fig, metadata = dep_viz.create_dependency_graph(deps, "Module Dependencies")
                files = dep_viz.save_figure(fig, f"dependencies_{timestamp}")
                generated_files.extend(files)
                
                # Circular dependencies
                circular = self.results['dependencies'].get('circular_imports', [])
                if circular:
                    fig = dep_viz.create_circular_dependency_visualization(circular)
                    files = dep_viz.save_figure(fig, f"circular_deps_{timestamp}")
                    generated_files.extend(files)
                
                print(f"  - Generated dependency visualizations")
        
        # Generate complexity visualizations
        if 'complexity' in self.results:
            complexity_data = self.results['complexity'].get('files', {})
            if complexity_data:
                # Complexity heatmap
                fig, _ = complexity_viz.create_complexity_heatmap(complexity_data)
                files = complexity_viz.save_figure(fig, f"complexity_heatmap_{timestamp}")
                generated_files.extend(files)
                
                print(f"  - Generated complexity visualizations")
        
        # Generate function call network
        if 'call_graph' in self.results:
            call_graph = self.results['call_graph'].get('functions', {})
            if call_graph:
                # Function network
                fig, _ = network_viz.create_function_call_network(call_graph)
                files = network_viz.save_figure(fig, f"function_network_{timestamp}")
                generated_files.extend(files)
                
                # Interactive network
                html_file = network_viz.create_interactive_network(
                    call_graph,
                    title="Interactive Function Network"
                )
                generated_files.append(html_file)
                
                print(f"  - Generated function network visualizations")
        
        # Generate comprehensive dashboard
        dashboard_file = dashboard_gen.generate_quality_dashboard(
            self.results,
            "Code Interaction Analysis Dashboard"
        )
        generated_files.append(dashboard_file)
        print(f"  - Generated interactive dashboard: {Path(dashboard_file).name}")
        
        return generated_files

    def run(self):
        """Run the complete interaction mapping."""
        print(f"Starting code interaction mapping for: {self.project_root}")
        print("=" * 80)

        # Run all analyses
        self.analyze_dependencies()
        self.analyze_call_graph()
        self.analyze_architecture()
        self.analyze_imports()
        self.analyze_complexity()

        # Generate reports
        report_files = self.generate_interaction_report()

        print("\n" + "=" * 80)
        print("CODE INTERACTION MAPPING COMPLETE!")
        print("=" * 80)
        print(f"\nAll reports saved to: {report_files.get('report_dir', 'reports')}")
        print("\nGenerated files:")
        for report_type, file_path in report_files.items():
            if report_type not in ['report_dir', 'timestamp']:
                print(f"  - {report_type.upper()}: {Path(file_path).name}")

        return report_files


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Map code interactions within a Python project")
    parser.add_argument("--project-root", default="/workspace",
                       help="Root directory of the project to analyze")
    parser.add_argument("--exclude", nargs="*", default=["venv", "__pycache__", ".git"],
                       help="Directories to exclude from analysis")

    args = parser.parse_args()

    mapper = CodeInteractionMapper(args.project_root)
    mapper.run()


if __name__ == "__main__":
    main()
