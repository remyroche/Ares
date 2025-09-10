#!/usr/bin/env python3
"""
Import Verifier Pipeline - Pipeline for verifying file import status.

This pipeline uses the ImportVerifierAnalyzer to check which files are imported
by others and provides a simple yes/no answer for each file's import status.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from pipelines.base_pipeline import BasePipeline, PipelineConfig
from analyzers.import_verifier_analyzer import ImportVerifierAnalyzer
# Import visualizers with fallback
try:
    from visualizers.dependency_graph import DependencyGraphVisualizer
    from visualizers.interaction_network import InteractionNetworkVisualizer
    VISUALIZERS_AVAILABLE = True
except ImportError:
    VISUALIZERS_AVAILABLE = False
    DependencyGraphVisualizer = None
    InteractionNetworkVisualizer = None


class ImportVerifierPipeline(BasePipeline):
    """Pipeline for verifying file import status."""
    
    def __init__(self, project_root: Optional[Union[str, Path]] = None, 
                 config: Optional[PipelineConfig] = None,
                 enable_plugins: bool = False,  # Disable plugins for this simple pipeline
                 pipeline_name: str = "import_verifier") -> None:
        """Initialize the import verifier pipeline."""
        super().__init__(project_root, config, enable_plugins, pipeline_name)
        
        # Initialize the analyzer
        self.analyzer = ImportVerifierAnalyzer(self.config.__dict__)
        
        # Initialize visualizers for enhanced analysis (if available)
        if VISUALIZERS_AVAILABLE:
            self.dependency_visualizer = DependencyGraphVisualizer(str(self.reports_dir / "dependency_graphs"))
            self.interaction_visualizer = InteractionNetworkVisualizer(str(self.reports_dir / "interaction_networks"))
        else:
            self.dependency_visualizer = None
            self.interaction_visualizer = None
        
        self.logger.info(f"Initialized ImportVerifierPipeline for project: {self.project_root}")
    
    def run(self, target_directory: Optional[str] = None, 
            save_report: bool = True, 
            print_report: bool = True,
            create_visualizations: bool = False) -> Dict[str, Any]:
        """
        Run the import verification analysis.
        
        Args:
            target_directory: Directory to analyze (defaults to project root)
            save_report: Whether to save the report to file
            print_report: Whether to print the report to console
            create_visualizations: Whether to create enhanced visualizations
            
        Returns:
            Dict containing analysis results
        """
        self.logger.info("Starting import verification analysis...")
        
        # Use target directory or project root
        analysis_dir = target_directory or str(self.project_root)
        
        try:
            # Run the analysis
            results = self.analyzer.analyze_directory(analysis_dir)
            
            # Add pipeline metadata
            results["pipeline_info"] = {
                "pipeline_name": self.__class__.__name__,
                "timestamp": self.timestamp,
                "project_root": str(self.project_root),
                "analysis_directory": analysis_dir,
                "analyzer_used": "ImportVerifierAnalyzer"
            }
            
            # Create enhanced visualizations if requested and available
            visualizations = {}
            if create_visualizations and VISUALIZERS_AVAILABLE:
                self.logger.info("Creating enhanced visualizations...")
                visualizations = self._create_enhanced_visualizations(results)
                results["visualizations"] = visualizations
            elif create_visualizations and not VISUALIZERS_AVAILABLE:
                self.logger.warning("Visualizations requested but visualizer dependencies not available")
                results["visualizations"] = {"error": "Visualizer dependencies not available"}
            
            # Print report if requested
            if print_report:
                self.analyzer.print_simple_report(results)
            
            # Save report if requested
            if save_report:
                report_path = self._save_report(results, "import_verification")
                self.logger.info(f"Report saved to: {report_path}")
                results["report_path"] = str(report_path)
            
            # Update metrics
            self.metrics["files_processed"] = results.get("summary", {}).get("total_files", 0)
            self.metrics["successful_executions"] += 1
            
            self.logger.info("Import verification analysis completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error during import verification analysis: {e}")
            self.metrics["failed_executions"] += 1
            return {
                "error": str(e),
                "error_type": type(e).__name__,
                "pipeline_info": {
                    "pipeline_name": self.__class__.__name__,
                    "timestamp": self.timestamp,
                    "project_root": str(self.project_root),
                    "analysis_directory": analysis_dir
                }
            }
    
    def get_unimported_files(self, results: Dict[str, Any]) -> List[str]:
        """
        Get list of files that are not imported by any other files.
        
        Args:
            results: Results from the run() method
            
        Returns:
            List of file paths that are not imported
        """
        unimported_files = []
        import_status = results.get("import_status", {})
        
        for file_path, status in import_status.items():
            if not status.get("is_imported", False):
                unimported_files.append(file_path)
        
        return unimported_files
    
    def get_imported_files(self, results: Dict[str, Any]) -> List[str]:
        """
        Get list of files that are imported by other files.
        
        Args:
            results: Results from the run() method
            
        Returns:
            List of file paths that are imported
        """
        imported_files = []
        import_status = results.get("import_status", {})
        
        for file_path, status in import_status.items():
            if status.get("is_imported", False):
                imported_files.append(file_path)
        
        return imported_files
    
    def get_most_imported_files(self, results: Dict[str, Any], top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most imported files.
        
        Args:
            results: Results from the run() method
            top_n: Number of top files to return
            
        Returns:
            List of dicts containing file info and import count
        """
        import_status = results.get("import_status", {})
        
        # Sort by import count
        sorted_files = sorted(
            import_status.items(),
            key=lambda x: x[1].get("import_count", 0),
            reverse=True
        )
        
        top_files = []
        for file_path, status in sorted_files[:top_n]:
            top_files.append({
                "file_path": file_path,
                "import_count": status.get("import_count", 0),
                "imported_by": status.get("imported_by", []),
                "module_name": status.get("module_name", "")
            })
        
        return top_files
    
    def generate_summary_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary report.
        
        Args:
            results: Results from the run() method
            
        Returns:
            String containing the summary report
        """
        summary = results.get("summary", {})
        pipeline_info = results.get("pipeline_info", {})
        
        report_lines = [
            "="*80,
            "IMPORT VERIFICATION SUMMARY REPORT",
            "="*80,
            f"Pipeline: {pipeline_info.get('pipeline_name', 'Unknown')}",
            f"Timestamp: {pipeline_info.get('timestamp', 'Unknown')}",
            f"Project Root: {pipeline_info.get('project_root', 'Unknown')}",
            f"Analysis Directory: {pipeline_info.get('analysis_directory', 'Unknown')}",
            "",
            "STATISTICS:",
            f"  Total files analyzed: {summary.get('total_files', 0)}",
            f"  Files imported by others: {summary.get('imported_files', 0)}",
            f"  Files NOT imported by others: {summary.get('unimported_files', 0)}",
            f"  Import percentage: {summary.get('import_percentage', 0):.1f}%",
            ""
        ]
        
        # Add most imported file info
        most_imported = summary.get("most_imported_file", {})
        if most_imported.get("file"):
            report_lines.extend([
                "MOST IMPORTED FILE:",
                f"  File: {most_imported['file']}",
                f"  Import count: {most_imported['import_count']}",
                ""
            ])
        
        # Add least imported file info
        least_imported = summary.get("least_imported_file", {})
        if least_imported.get("file"):
            report_lines.extend([
                "LEAST IMPORTED FILE:",
                f"  File: {least_imported['file']}",
                f"  Import count: {least_imported['import_count']}",
                ""
            ])
        
        # Add top 5 most imported files with details
        top_files = self.get_most_imported_files(results, 5)
        if top_files:
            report_lines.extend([
                "TOP 5 MOST IMPORTED FILES:",
                ""
            ])
            for i, file_info in enumerate(top_files, 1):
                report_lines.append(f"  {i}. {file_info['file_path']} ({file_info['import_count']} imports)")
                # Show which files import this one
                imported_by = file_info.get('imported_by', [])
                if imported_by:
                    report_lines.append(f"     Imported by:")
                    for importer in sorted(imported_by)[:5]:  # Show first 5 importers
                        try:
                            rel_importer = Path(importer).relative_to(Path.cwd())
                            report_lines.append(f"       • {rel_importer}")
                        except ValueError:
                            report_lines.append(f"       • {importer}")
                    if len(imported_by) > 5:
                        report_lines.append(f"       ... and {len(imported_by) - 5} more")
                report_lines.append("")
        
        report_lines.extend([
            "="*80,
            "End of Report",
            "="*80
        ])
        
        return "\n".join(report_lines)
    
    def _create_enhanced_visualizations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create enhanced visualizations using import verification data.
        
        Args:
            results: Results from import verification analysis
            
        Returns:
            Dictionary of created visualizations
        """
        visualizations = {}
        
        if not VISUALIZERS_AVAILABLE:
            return {"error": "Visualizer dependencies not available"}
        
        try:
            # Extract import relationships for dependency graph
            import_status = results.get("import_status", {})
            dependencies = {}
            
            # Build dependency relationships from import data
            for file_path, status in import_status.items():
                imported_by = status.get("imported_by", [])
                if imported_by:
                    dependencies[file_path] = imported_by
            
            # Create enhanced dependency graph
            if dependencies:
                self.logger.info("Creating enhanced dependency graph...")
                try:
                    fig, metadata = self.dependency_visualizer.create_enhanced_dependency_graph_with_imports(
                        dependencies, results, "Enhanced Import Dependency Graph"
                    )
                    if fig:
                        saved_files = self.dependency_visualizer.save_figure(fig, "enhanced_dependency_graph")
                        visualizations["enhanced_dependency_graph"] = {
                            "files": saved_files,
                            "metadata": metadata
                        }
                except Exception as e:
                    self.logger.warning(f"Could not create enhanced dependency graph: {e}")
            
            # Create enhanced interaction network
            self.logger.info("Creating enhanced interaction network...")
            try:
                # Use import relationships as interactions
                interactions = {}
                for file_path, status in import_status.items():
                    imported_by = status.get("imported_by", [])
                    if imported_by:
                        interactions[file_path] = imported_by
                
                if interactions:
                    fig, metadata = self.interaction_visualizer.create_enhanced_interaction_network_with_imports(
                        interactions, results, "Enhanced Import Interaction Network"
                    )
                    if fig:
                        saved_files = self.interaction_visualizer.save_figure(fig, "enhanced_interaction_network")
                        visualizations["enhanced_interaction_network"] = {
                            "files": saved_files,
                            "metadata": metadata
                        }
            except Exception as e:
                self.logger.warning(f"Could not create enhanced interaction network: {e}")
            
            # Create circular dependency visualization
            advanced_analysis = results.get("advanced_analysis", {})
            circular_imports = advanced_analysis.get("circular_imports", [])
            
            if circular_imports:
                self.logger.info("Creating circular dependency visualization...")
                try:
                    fig = self.dependency_visualizer.create_circular_dependency_visualization(
                        circular_imports, "Import Circular Dependencies"
                    )
                    if fig:
                        saved_files = self.dependency_visualizer.save_figure(fig, "circular_dependencies")
                        visualizations["circular_dependencies"] = {
                            "files": saved_files,
                            "type": "circular_dependency_analysis"
                        }
                except Exception as e:
                    self.logger.warning(f"Could not create circular dependency visualization: {e}")
            
        except Exception as e:
            self.logger.error(f"Error creating enhanced visualizations: {e}")
            visualizations["error"] = str(e)
        
        return visualizations


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Import Verifier Pipeline")
    parser.add_argument("--project-root", type=str, help="Project root directory")
    parser.add_argument("--target-dir", type=str, help="Target directory to analyze")
    parser.add_argument("--no-print", action="store_true", help="Don't print report to console")
    parser.add_argument("--no-save", action="store_true", help="Don't save report to file")
    parser.add_argument("--create-visualizations", action="store_true", help="Create enhanced visualizations")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = ImportVerifierPipeline(
        project_root=args.project_root,
        enable_plugins=False
    )
    
    if args.verbose:
        pipeline.logger.setLevel("DEBUG")
    
    # Run analysis
    results = pipeline.run(
        target_directory=args.target_dir,
        save_report=not args.no_save,
        print_report=not args.no_print,
        create_visualizations=args.create_visualizations
    )
    
    # Print summary if not printing full report
    if args.no_print:
        summary = pipeline.generate_summary_report(results)
        print(summary)
    
    # Exit with error code if there was an error
    if "error" in results:
        sys.exit(1)


if __name__ == "__main__":
    main()