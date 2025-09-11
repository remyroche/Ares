#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Demonstration Script: Enhanced Import Analysis

This script demonstrates how to use the import_verifier_pipeline.py to enhance
code detection and graphs in the code_quality/pipelines directory.

Usage:
    python3 pipelines/demo_enhanced_import_analysis.py
    python3 pipelines/demo_enhanced_import_analysis.py --project-root /path/to/project
    python3 pipelines/demo_enhanced_import_analysis.py --demo-type basic
    python3 pipelines/demo_enhanced_import_analysis.py --demo-type advanced
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from analysis_functions import run_import_verification, run_enhanced_import_analysis
from visualizers.import_network_visualizer import ImportNetworkVisualizer


class ImportAnalysisDemo:
    """Demonstration class for enhanced import analysis capabilities."""
    
    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.demo_results = {}
        
    def run_basic_demo(self) -> Dict[str, Any]:
        """Run basic import verification demonstration."""
        tprint("🔍 Running Basic Import Verification Demo")
        tprint("=" * 60)
        
        # Run import verification using simple function
        results = run_import_verification(
            project_root=str(self.project_root),
            target_directory=str(self.project_root),
            save_report=True,
            print_report=True
        )
        
        # Extract key insights
        summary = results.get("summary", {})
        import_status = results.get("import_status", {})
        
        insights = {
            "total_files": summary.get("total_files", 0),
            "imported_files": summary.get("imported_files", 0),
            "unimported_files": summary.get("unimported_files", 0),
            "import_percentage": summary.get("import_percentage", 0),
            "most_imported_file": summary.get("most_imported_file", {}),
            "least_imported_file": summary.get("least_imported_file", {})
        }
        
        # Show top 5 most imported files
        top_imported = pipeline.get_most_imported_files(results, 5)
        
        tprint(f"\n📊 Key Insights:")
        tprint(f"  • Total files analyzed: {insights['total_files']}")
        tprint(f"  • Files imported by others: {insights['imported_files']}")
        tprint(f"  • Files NOT imported: {insights['unimported_files']}")
        tprint(f"  • Import percentage: {insights['import_percentage']:.1f}%")
        
        if top_imported:
            tprint(f"\n🏆 Top 5 Most Imported Files:")
            for i, file_info in enumerate(top_imported, 1):
                tprint(f"  {i}. {Path(file_info['file_path']).name} ({file_info['import_count']} imports)")
        
        self.demo_results["basic_demo"] = {
            "results": results,
            "insights": insights,
            "top_imported": top_imported
        }
        
        return results
    
    def run_advanced_demo(self) -> Dict[str, Any]:
        """Run advanced enhanced import analysis demonstration."""
        tprint("\n🚀 Running Advanced Enhanced Import Analysis Demo")
        tprint("=" * 60)
        
        # Run comprehensive analysis using simple function
        results = run_enhanced_import_analysis(
            project_root=str(self.project_root),
            target_directory=str(self.project_root),
            save_report=True,
            print_report=True,
            create_visualizations=True
        )
        
        # Extract enhanced insights
        enhanced_detection = results.get("enhanced_detection", {})
        issues = enhanced_detection.get("issues", {})
        recommendations = enhanced_detection.get("recommendations", [])
        
        tprint(f"\n🔍 Enhanced Code Detection Results:")
        tprint(f"  • Unused modules: {len(issues.get('unused_modules', []))}")
        tprint(f"  • Orphaned files: {len(issues.get('orphaned_files', []))}")
        tprint(f"  • Circular dependencies: {len(issues.get('circular_dependencies', []))}")
        tprint(f"  • High coupling modules: {len(issues.get('high_coupling_modules', []))}")
        tprint(f"  • Critical dependencies: {len(issues.get('critical_dependencies', []))}")
        
        if recommendations:
            tprint(f"\n💡 Top Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(rec.get("priority", ""), "⚪")
                tprint(f"  {i}. {priority_emoji} {rec.get('title', 'Unknown')}")
                tprint(f"     {rec.get('description', '')}")
        
        # Show visualizations created
        visualizations = results.get("visualizations", {})
        if visualizations:
            tprint(f"\n📊 Visualizations Created:")
            for viz_name, viz_info in visualizations.items():
                if isinstance(viz_info, dict) and "files" in viz_info:
                    tprint(f"  • {viz_name}: {len(viz_info['files'])} files")
                elif isinstance(viz_info, dict) and "html_file" in viz_info:
                    tprint(f"  • {viz_name}: Interactive HTML")
        
        self.demo_results["advanced_demo"] = {
            "results": results,
            "issues": issues,
            "recommendations": recommendations,
            "visualizations": visualizations
        }
        
        return results
    
    def run_visualization_demo(self) -> Dict[str, Any]:
        """Run visualization demonstration."""
        tprint("\n🎨 Running Visualization Demo")
        tprint("=" * 60)
        
        # First get import verification data
        results = run_import_verification(
            project_root=str(self.project_root),
            target_directory=str(self.project_root),
            save_report=False,
            print_report=False
        )
        
        # Create visualizations
        visualizer = ImportNetworkVisualizer()
        
        tprint("Creating import network visualization...")
        try:
            fig, metadata = visualizer.create_import_network_from_verifier_data(
                results, "Demo Import Network Analysis"
            )
            if fig:
                saved_files = visualizer.save_figure(fig, "demo_import_network")
                tprint(f"  ✅ Import network saved to: {saved_files[0] if saved_files else 'N/A'}")
        except Exception as e:
            tprint(f"  ❌ Error creating import network: {e}")
        
        tprint("Creating import heatmap...")
        try:
            heatmap_fig = visualizer.create_import_heatmap(
                results, "Demo Import Heatmap"
            )
            if heatmap_fig:
                saved_files = visualizer.save_figure(heatmap_fig, "demo_import_heatmap")
                tprint(f"  ✅ Import heatmap saved to: {saved_files[0] if saved_files else 'N/A'}")
        except Exception as e:
            tprint(f"  ❌ Error creating heatmap: {e}")
        
        tprint("Creating circular dependency analysis...")
        try:
            circular_fig = visualizer.create_circular_dependency_analysis(
                results, "Demo Circular Dependency Analysis"
            )
            if circular_fig:
                saved_files = visualizer.save_figure(circular_fig, "demo_circular_dependencies")
                tprint(f"  ✅ Circular dependency analysis saved to: {saved_files[0] if saved_files else 'N/A'}")
        except Exception as e:
            tprint(f"  ❌ Error creating circular dependency analysis: {e}")
        
        tprint("Creating interactive network...")
        try:
            html_file = visualizer.create_interactive_import_network(
                results, "Demo Interactive Import Network"
            )
            tprint(f"  ✅ Interactive network saved to: {html_file}")
        except Exception as e:
            tprint(f"  ❌ Error creating interactive network: {e}")
        
        self.demo_results["visualization_demo"] = {
            "results": results,
            "visualizations_created": True
        }
        
        return results
    
    def run_custom_analysis_demo(self) -> Dict[str, Any]:
        """Run custom analysis demonstration."""
        tprint("\n🔧 Running Custom Analysis Demo")
        tprint("=" * 60)
        
        # Get import verification data
        results = run_import_verification(
            project_root=str(self.project_root),
            target_directory=str(self.project_root),
            save_report=False,
            print_report=False
        )
        
        # Perform custom analysis
        import_status = results.get("import_status", {})
        summary = results.get("summary", {})
        
        # Custom analysis: Find potential refactoring candidates
        refactoring_candidates = []
        for file_path, status in import_status.items():
            import_count = status.get("import_count", 0)
            is_imported = status.get("is_imported", False)
            
            # Files with high import count but low usage might be candidates for refactoring
            if import_count > 3 and not is_imported:
                refactoring_candidates.append({
                    "file": file_path,
                    "import_count": import_count,
                    "reason": "High import count but not imported by others"
                })
        
        # Custom analysis: Find potential utility modules
        utility_modules = []
        for file_path, status in import_status.items():
            import_count = status.get("import_count", 0)
            is_imported = status.get("is_imported", False)
            
            # Files imported by many others might be utility modules
            if import_count > 5 and is_imported:
                utility_modules.append({
                    "file": file_path,
                    "import_count": import_count,
                    "reason": "Imported by many files - potential utility module"
                })
        
        tprint(f"🔍 Custom Analysis Results:")
        tprint(f"  • Potential refactoring candidates: {len(refactoring_candidates)}")
        tprint(f"  • Potential utility modules: {len(utility_modules)}")
        
        if refactoring_candidates:
            tprint(f"\n📝 Top Refactoring Candidates:")
            for i, candidate in enumerate(refactoring_candidates[:3], 1):
                tprint(f"  {i}. {Path(candidate['file']).name}")
                tprint(f"     Import count: {candidate['import_count']}")
                tprint(f"     Reason: {candidate['reason']}")
        
        if utility_modules:
            tprint(f"\n🛠️  Top Utility Modules:")
            for i, module in enumerate(utility_modules[:3], 1):
                tprint(f"  {i}. {Path(module['file']).name}")
                tprint(f"     Import count: {module['import_count']}")
                tprint(f"     Reason: {module['reason']}")
        
        self.demo_results["custom_analysis_demo"] = {
            "results": results,
            "refactoring_candidates": refactoring_candidates,
            "utility_modules": utility_modules
        }
        
        return results
    
    def generate_demo_report(self) -> str:
        """Generate a comprehensive demo report."""
        report_lines = [
            "# Enhanced Import Analysis Demo Report",
            "=" * 50,
            f"Project Root: {self.project_root}",
            f"Demo Results: {len(self.demo_results)} demos completed",
            "",
            "## Demo Summary",
            ""
        ]
        
        for demo_name, demo_data in self.demo_results.items():
            report_lines.append(f"### {demo_name.replace('_', ' ').title()}")
            
            if demo_name == "basic_demo":
                insights = demo_data.get("insights", {})
                report_lines.extend([
                    f"- Total files analyzed: {insights.get('total_files', 0)}",
                    f"- Import percentage: {insights.get('import_percentage', 0):.1f}%",
                    f"- Most imported file: {insights.get('most_imported_file', {}).get('file', 'N/A')}",
                    ""
                ])
            
            elif demo_name == "advanced_demo":
                issues = demo_data.get("issues", {})
                recommendations = demo_data.get("recommendations", [])
                report_lines.extend([
                    f"- Issues found: {sum(len(issue_list) for issue_list in issues.values())}",
                    f"- Recommendations: {len(recommendations)}",
                    f"- Visualizations created: {len(demo_data.get('visualizations', {}))}",
                    ""
                ])
            
            elif demo_name == "visualization_demo":
                report_lines.extend([
                    "- Import network visualization created",
                    "- Import heatmap created", 
                    "- Circular dependency analysis created",
                    "- Interactive network created",
                    ""
                ])
            
            elif demo_name == "custom_analysis_demo":
                refactoring = demo_data.get("refactoring_candidates", [])
                utilities = demo_data.get("utility_modules", [])
                report_lines.extend([
                    f"- Refactoring candidates: {len(refactoring)}",
                    f"- Utility modules: {len(utilities)}",
                    ""
                ])
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_file = self.project_root / "demo_report.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        return str(report_file)
    
    def run_all_demos(self) -> Dict[str, Any]:
        """Run all demonstration types."""
        tprint("🎯 Enhanced Import Analysis - Complete Demo")
        tprint("=" * 80)
        tprint(f"Project Root: {self.project_root}")
        tprint("=" * 80)
        
        all_results = {}
        
        # Run all demos
        try:
            all_results["basic"] = self.run_basic_demo()
        except Exception as e:
            tprint(f"❌ Basic demo failed: {e}")
            all_results["basic"] = {"error": str(e)}
        
        try:
            all_results["advanced"] = self.run_advanced_demo()
        except Exception as e:
            tprint(f"❌ Advanced demo failed: {e}")
            all_results["advanced"] = {"error": str(e)}
        
        try:
            all_results["visualization"] = self.run_visualization_demo()
        except Exception as e:
            tprint(f"❌ Visualization demo failed: {e}")
            all_results["visualization"] = {"error": str(e)}
        
        try:
            all_results["custom_analysis"] = self.run_custom_analysis_demo()
        except Exception as e:
            tprint(f"❌ Custom analysis demo failed: {e}")
            all_results["custom_analysis"] = {"error": str(e)}
        
        # Generate report
        try:
            report_file = self.generate_demo_report()
            tprint(f"\n📄 Demo report saved to: {report_file}")
        except Exception as e:
            tprint(f"❌ Failed to generate report: {e}")
        
        tprint("\n🎉 Demo completed!")
        return all_results


def main():
    """Main function for demonstration script."""
    parser = argparse.ArgumentParser(
        description="Enhanced Import Analysis Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all demos
  python3 pipelines/demo_enhanced_import_analysis.py
  
  # Run specific demo type
  python3 pipelines/demo_enhanced_import_analysis.py --demo-type basic
  python3 pipelines/demo_enhanced_import_analysis.py --demo-type advanced
  python3 pipelines/demo_enhanced_import_analysis.py --demo-type visualization
  python3 pipelines/demo_enhanced_import_analysis.py --demo-type custom
  
  # Run on specific project
  python3 pipelines/demo_enhanced_import_analysis.py --project-root /path/to/project
        """
    )
    
    parser.add_argument("--project-root", type=str, help="Project root directory")
    parser.add_argument("--demo-type", choices=["basic", "advanced", "visualization", "custom", "all"], 
                       default="all", help="Type of demo to run")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Create demo instance
    demo = ImportAnalysisDemo(args.project_root)
    
    # Run selected demo
    if args.demo_type == "all":
        results = demo.run_all_demos()
    elif args.demo_type == "basic":
        results = demo.run_basic_demo()
    elif args.demo_type == "advanced":
        results = demo.run_advanced_demo()
    elif args.demo_type == "visualization":
        results = demo.run_visualization_demo()
    elif args.demo_type == "custom":
        results = demo.run_custom_analysis_demo()
    
    # Save results
    results_file = Path(args.project_root or ".") / "demo_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    if args.verbose:
        tprint(f"\n📊 Results saved to: {results_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())