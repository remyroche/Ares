#!/usr/bin/env python3
"""
Enhanced Import Analysis Pipeline

This pipeline combines the ImportVerifierAnalyzer with other code quality analyzers
to provide comprehensive import analysis and enhanced code detection capabilities.
It creates sophisticated graphs and visualizations of import relationships.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from pipelines.base_pipeline import BasePipeline, PipelineConfig
from analyzers.import_verifier_analyzer import ImportVerifierAnalyzer
from analyzers.dependency_analyzer import DependencyAnalyzer
from analyzers.complexity_analyzer import ComplexityAnalyzer
from analyzers.dead_code_analyzer import DeadCodeAnalyzer
from visualizers.import_network_visualizer import ImportNetworkVisualizer
from visualizers.dependency_graph import DependencyGraphVisualizer
from visualizers.interaction_network import InteractionNetworkVisualizer


class EnhancedImportAnalysisPipeline(BasePipeline):
    """Enhanced pipeline for comprehensive import analysis and code detection."""
    
    def __init__(self, project_root: Optional[Union[str, Path]] = None, 
                 config: Optional[PipelineConfig] = None,
                 enable_plugins: bool = True,
                 pipeline_name: str = "enhanced_import_analysis") -> None:
        """Initialize the enhanced import analysis pipeline."""
        super().__init__(project_root, config, enable_plugins, pipeline_name)
        
        # Initialize analyzers
        self.import_verifier = ImportVerifierAnalyzer(self.config.__dict__)
        self.dependency_analyzer = DependencyAnalyzer(self.config.__dict__)
        self.complexity_analyzer = ComplexityAnalyzer(self.config.__dict__)
        self.dead_code_analyzer = DeadCodeAnalyzer(self.config.__dict__)
        
        # Initialize visualizers
        self.import_visualizer = ImportNetworkVisualizer(str(self.reports_dir / "import_networks"))
        self.dependency_visualizer = DependencyGraphVisualizer(str(self.reports_dir / "dependency_graphs"))
        self.interaction_visualizer = InteractionNetworkVisualizer(str(self.reports_dir / "interaction_networks"))
        
        self.logger.info(f"Initialized EnhancedImportAnalysisPipeline for project: {self.project_root}")
    
    def run(self, target_directory: Optional[str] = None, 
            save_report: bool = True, 
            print_report: bool = True,
            create_visualizations: bool = True) -> Dict[str, Any]:
        """
        Run the enhanced import analysis.
        
        Args:
            target_directory: Directory to analyze (defaults to project root)
            save_report: Whether to save the report to file
            print_report: Whether to print the report to console
            create_visualizations: Whether to create visualizations
            
        Returns:
            Dict containing comprehensive analysis results
        """
        self.logger.info("Starting enhanced import analysis...")
        
        # Use target directory or project root
        analysis_dir = target_directory or str(self.project_root)
        
        try:
            # Step 1: Import verification analysis
            self.logger.info("Running import verification analysis...")
            import_results = self.import_verifier.analyze_directory(analysis_dir)
            
            # Step 2: Dependency analysis
            self.logger.info("Running dependency analysis...")
            dependency_results = self.dependency_analyzer.analyze_directory(analysis_dir)
            
            # Step 3: Complexity analysis
            self.logger.info("Running complexity analysis...")
            complexity_results = self.complexity_analyzer.analyze_directory(analysis_dir)
            
            # Step 4: Dead code analysis
            self.logger.info("Running dead code analysis...")
            dead_code_results = self.dead_code_analyzer.analyze_directory(analysis_dir)
            
            # Step 5: Enhanced code detection using import patterns
            self.logger.info("Running enhanced code detection...")
            enhanced_detection = self._enhanced_code_detection(import_results, dependency_results, complexity_results)
            
            # Step 6: Create visualizations
            visualizations = {}
            if create_visualizations:
                self.logger.info("Creating visualizations...")
                visualizations = self._create_visualizations(import_results, dependency_results, enhanced_detection)
            
            # Step 7: Generate comprehensive report
            comprehensive_results = {
                "pipeline_info": {
                    "pipeline_name": self.__class__.__name__,
                    "timestamp": self.timestamp,
                    "project_root": str(self.project_root),
                    "analysis_directory": analysis_dir,
                    "analyzers_used": [
                        "ImportVerifierAnalyzer",
                        "DependencyAnalyzer", 
                        "ComplexityAnalyzer",
                        "DeadCodeAnalyzer"
                    ]
                },
                "import_analysis": import_results,
                "dependency_analysis": dependency_results,
                "complexity_analysis": complexity_results,
                "dead_code_analysis": dead_code_results,
                "enhanced_detection": enhanced_detection,
                "visualizations": visualizations,
                "summary": self._generate_comprehensive_summary(
                    import_results, dependency_results, complexity_results, 
                    dead_code_results, enhanced_detection
                )
            }
            
            # Print report if requested
            if print_report:
                self._print_comprehensive_report(comprehensive_results)
            
            # Save report if requested
            if save_report:
                report_path = self._save_report(comprehensive_results, "enhanced_import_analysis")
                self.logger.info(f"Report saved to: {report_path}")
                comprehensive_results["report_path"] = str(report_path)
            
            # Update metrics
            self.metrics["files_processed"] = import_results.get("summary", {}).get("total_files", 0)
            self.metrics["successful_executions"] += 1
            
            self.logger.info("Enhanced import analysis completed successfully")
            return comprehensive_results
            
        except Exception as e:
            self.logger.error(f"Error during enhanced import analysis: {e}")
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
    
    def _enhanced_code_detection(self, import_results: Dict[str, Any], 
                               dependency_results: Dict[str, Any],
                               complexity_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced code detection using import patterns and relationships.
        
        Args:
            import_results: Results from import verification
            dependency_results: Results from dependency analysis
            complexity_results: Results from complexity analysis
            
        Returns:
            Enhanced detection results
        """
        import_status = import_results.get("import_status", {})
        advanced_analysis = import_results.get("advanced_analysis", {})
        
        # Detect potential issues using import patterns
        issues = {
            "unused_modules": [],
            "orphaned_files": [],
            "circular_dependencies": [],
            "high_coupling_modules": [],
            "low_cohesion_modules": [],
            "potential_refactoring_candidates": [],
            "critical_dependencies": [],
            "bottleneck_modules": []
        }
        
        # 1. Identify unused modules (not imported by any production code)
        for file_path, status in import_status.items():
            if not status.get("is_imported", False):
                issues["unused_modules"].append({
                    "file": file_path,
                    "reason": "Not imported by any other files",
                    "severity": "medium"
                })
            elif status.get("only_imported_by_non_production", False):
                issues["orphaned_files"].append({
                    "file": file_path,
                    "reason": "Only imported by non-production files",
                    "severity": "low"
                })
        
        # 2. Identify circular dependencies
        circular_imports = advanced_analysis.get("circular_imports", [])
        for cycle in circular_imports:
            issues["circular_dependencies"].append({
                "cycle": cycle,
                "files": [Path(f).name for f in cycle],
                "severity": "high",
                "impact": "Can cause import errors and tight coupling"
            })
        
        # 3. Identify high coupling modules (many imports)
        import_counts = {}
        for file_path, status in import_status.items():
            import_count = status.get("import_count", 0)
            if import_count > 5:  # Threshold for high coupling
                issues["high_coupling_modules"].append({
                    "file": file_path,
                    "import_count": import_count,
                    "severity": "medium",
                    "suggestion": "Consider breaking down into smaller modules"
                })
        
        # 4. Identify critical dependencies (files that many others depend on)
        critical_paths = advanced_analysis.get("critical_paths", {})
        high_impact_files = critical_paths.get("high_impact_files", [])
        for file_path, count in high_impact_files:
            if count > 3:  # Threshold for critical dependency
                issues["critical_dependencies"].append({
                    "file": file_path,
                    "dependent_count": count,
                    "severity": "high",
                    "impact": "Changes to this file will affect many others"
                })
        
        # 5. Identify potential refactoring candidates
        # Files with high complexity and many imports
        for file_path, status in import_status.items():
            import_count = status.get("import_count", 0)
            # This would need complexity data integration
            if import_count > 3:  # Simple heuristic for now
                issues["potential_refactoring_candidates"].append({
                    "file": file_path,
                    "import_count": import_count,
                    "reason": "High import count suggests potential for refactoring",
                    "severity": "low"
                })
        
        # 6. Identify bottleneck modules (files that import many others)
        # This would require analyzing what each file imports, not just what imports it
        # For now, we'll use a simple heuristic
        for file_path, status in import_status.items():
            if status.get("import_count", 0) > 8:  # High threshold
                issues["bottleneck_modules"].append({
                    "file": file_path,
                    "import_count": status.get("import_count", 0),
                    "severity": "medium",
                    "suggestion": "This module may be doing too much"
                })
        
        # Calculate summary statistics
        total_issues = sum(len(issue_list) for issue_list in issues.values())
        high_severity_issues = sum(
            len([issue for issue in issue_list if issue.get("severity") == "high"])
            for issue_list in issues.values()
        )
        medium_severity_issues = sum(
            len([issue for issue in issue_list if issue.get("severity") == "medium"])
            for issue_list in issues.values()
        )
        low_severity_issues = sum(
            len([issue for issue in issue_list if issue.get("severity") == "low"])
            for issue_list in issues.values()
        )
        
        return {
            "issues": issues,
            "summary": {
                "total_issues": total_issues,
                "high_severity": high_severity_issues,
                "medium_severity": medium_severity_issues,
                "low_severity": low_severity_issues,
                "issue_categories": {
                    category: len(issue_list) for category, issue_list in issues.items()
                }
            },
            "recommendations": self._generate_recommendations(issues)
        }
    
    def _create_visualizations(self, import_results: Dict[str, Any], 
                             dependency_results: Dict[str, Any],
                             enhanced_detection: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create comprehensive visualizations.
        
        Args:
            import_results: Import verification results
            dependency_results: Dependency analysis results
            enhanced_detection: Enhanced detection results
            
        Returns:
            Dictionary of created visualizations
        """
        visualizations = {}
        
        try:
            # 1. Import network visualization
            self.logger.info("Creating import network visualization...")
            fig, metadata = self.import_visualizer.create_import_network_from_verifier_data(
                import_results, "Enhanced Import Network Analysis"
            )
            if fig:
                saved_files = self.import_visualizer.save_figure(fig, "import_network_analysis")
                visualizations["import_network"] = {
                    "files": saved_files,
                    "metadata": metadata
                }
            
            # 2. Interactive import network
            try:
                html_file = self.import_visualizer.create_interactive_import_network(
                    import_results, "Interactive Import Network"
                )
                visualizations["interactive_import_network"] = {
                    "html_file": html_file,
                    "type": "interactive"
                }
            except Exception as e:
                self.logger.warning(f"Could not create interactive visualization: {e}")
            
            # 3. Import heatmap
            self.logger.info("Creating import heatmap...")
            heatmap_fig = self.import_visualizer.create_import_heatmap(
                import_results, "Import Relationship Heatmap"
            )
            if heatmap_fig:
                saved_files = self.import_visualizer.save_figure(heatmap_fig, "import_heatmap")
                visualizations["import_heatmap"] = {
                    "files": saved_files,
                    "type": "heatmap"
                }
            
            # 4. Circular dependency analysis
            self.logger.info("Creating circular dependency analysis...")
            circular_fig = self.import_visualizer.create_circular_dependency_analysis(
                import_results, "Circular Dependency Analysis"
            )
            if circular_fig:
                saved_files = self.import_visualizer.save_figure(circular_fig, "circular_dependencies")
                visualizations["circular_dependencies"] = {
                    "files": saved_files,
                    "type": "analysis"
                }
            
            # 5. Dependency graph (if dependency data is available)
            if dependency_results and "dependencies" in dependency_results:
                self.logger.info("Creating dependency graph...")
                try:
                    dep_fig, dep_metadata = self.dependency_visualizer.create_dependency_graph(
                        dependency_results["dependencies"], "Module Dependencies"
                    )
                    if dep_fig:
                        saved_files = self.dependency_visualizer.save_figure(dep_fig, "dependency_graph")
                        visualizations["dependency_graph"] = {
                            "files": saved_files,
                            "metadata": dep_metadata
                        }
                except Exception as e:
                    self.logger.warning(f"Could not create dependency graph: {e}")
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}")
            visualizations["error"] = str(e)
        
        return visualizations
    
    def _generate_recommendations(self, issues: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on detected issues."""
        recommendations = []
        
        # High priority recommendations
        if issues["circular_dependencies"]:
            recommendations.append({
                "priority": "high",
                "category": "circular_dependencies",
                "title": "Resolve Circular Dependencies",
                "description": f"Found {len(issues['circular_dependencies'])} circular dependency cycles",
                "action": "Review and refactor modules to break circular imports",
                "impact": "High - prevents import errors and improves maintainability"
            })
        
        if issues["critical_dependencies"]:
            recommendations.append({
                "priority": "high",
                "category": "critical_dependencies", 
                "title": "Review Critical Dependencies",
                "description": f"Found {len(issues['critical_dependencies'])} highly coupled modules",
                "action": "Consider breaking down critical modules or adding interfaces",
                "impact": "High - reduces risk of cascading changes"
            })
        
        # Medium priority recommendations
        if issues["unused_modules"]:
            recommendations.append({
                "priority": "medium",
                "category": "unused_modules",
                "title": "Remove Unused Modules",
                "description": f"Found {len(issues['unused_modules'])} unused modules",
                "action": "Remove or integrate unused modules to reduce codebase size",
                "impact": "Medium - reduces maintenance burden"
            })
        
        if issues["high_coupling_modules"]:
            recommendations.append({
                "priority": "medium",
                "category": "high_coupling",
                "title": "Reduce Module Coupling",
                "description": f"Found {len(issues['high_coupling_modules'])} highly coupled modules",
                "action": "Refactor modules to reduce dependencies",
                "impact": "Medium - improves modularity and testability"
            })
        
        # Low priority recommendations
        if issues["potential_refactoring_candidates"]:
            recommendations.append({
                "priority": "low",
                "category": "refactoring",
                "title": "Consider Refactoring",
                "description": f"Found {len(issues['potential_refactoring_candidates'])} refactoring candidates",
                "action": "Review and refactor modules with high import counts",
                "impact": "Low - improves code organization"
            })
        
        return recommendations
    
    def _generate_comprehensive_summary(self, import_results: Dict[str, Any],
                                      dependency_results: Dict[str, Any],
                                      complexity_results: Dict[str, Any],
                                      dead_code_results: Dict[str, Any],
                                      enhanced_detection: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive summary of all analyses."""
        import_summary = import_results.get("summary", {})
        detection_summary = enhanced_detection.get("summary", {})
        
        return {
            "total_files_analyzed": import_summary.get("total_files", 0),
            "import_statistics": {
                "imported_files": import_summary.get("imported_files", 0),
                "unimported_files": import_summary.get("unimported_files", 0),
                "import_percentage": import_summary.get("import_percentage", 0),
                "circular_dependencies": len(import_results.get("advanced_analysis", {}).get("circular_imports", []))
            },
            "issue_statistics": {
                "total_issues": detection_summary.get("total_issues", 0),
                "high_severity": detection_summary.get("high_severity", 0),
                "medium_severity": detection_summary.get("medium_severity", 0),
                "low_severity": detection_summary.get("low_severity", 0)
            },
            "code_quality_metrics": {
                "unused_modules": len(enhanced_detection.get("issues", {}).get("unused_modules", [])),
                "orphaned_files": len(enhanced_detection.get("issues", {}).get("orphaned_files", [])),
                "high_coupling_modules": len(enhanced_detection.get("issues", {}).get("high_coupling_modules", [])),
                "critical_dependencies": len(enhanced_detection.get("issues", {}).get("critical_dependencies", []))
            },
            "recommendations_count": len(enhanced_detection.get("recommendations", [])),
            "visualizations_created": len(enhanced_detection.get("visualizations", {}))
        }
    
    def _print_comprehensive_report(self, results: Dict[str, Any]) -> None:
        """Print a comprehensive report to console."""
        print("\n" + "="*100)
        print("ENHANCED IMPORT ANALYSIS REPORT")
        print("="*100)
        
        pipeline_info = results.get("pipeline_info", {})
        print(f"Pipeline: {pipeline_info.get('pipeline_name', 'Unknown')}")
        print(f"Timestamp: {pipeline_info.get('timestamp', 'Unknown')}")
        print(f"Project Root: {pipeline_info.get('project_root', 'Unknown')}")
        print(f"Analysis Directory: {pipeline_info.get('analysis_directory', 'Unknown')}")
        
        # Summary statistics
        summary = results.get("summary", {})
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"  Total files analyzed: {summary.get('total_files_analyzed', 0)}")
        
        import_stats = summary.get("import_statistics", {})
        print(f"  Imported files: {import_stats.get('imported_files', 0)}")
        print(f"  Unimported files: {import_stats.get('unimported_files', 0)}")
        print(f"  Import percentage: {import_stats.get('import_percentage', 0):.1f}%")
        print(f"  Circular dependencies: {import_stats.get('circular_dependencies', 0)}")
        
        issue_stats = summary.get("issue_statistics", {})
        print(f"\n🔍 ISSUE DETECTION:")
        print(f"  Total issues found: {issue_stats.get('total_issues', 0)}")
        print(f"  High severity: {issue_stats.get('high_severity', 0)}")
        print(f"  Medium severity: {issue_stats.get('medium_severity', 0)}")
        print(f"  Low severity: {issue_stats.get('low_severity', 0)}")
        
        # Code quality metrics
        quality_metrics = summary.get("code_quality_metrics", {})
        print(f"\n📈 CODE QUALITY METRICS:")
        print(f"  Unused modules: {quality_metrics.get('unused_modules', 0)}")
        print(f"  Orphaned files: {quality_metrics.get('orphaned_files', 0)}")
        print(f"  High coupling modules: {quality_metrics.get('high_coupling_modules', 0)}")
        print(f"  Critical dependencies: {quality_metrics.get('critical_dependencies', 0)}")
        
        # Recommendations
        enhanced_detection = results.get("enhanced_detection", {})
        recommendations = enhanced_detection.get("recommendations", [])
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS ({len(recommendations)} total):")
            for i, rec in enumerate(recommendations, 1):
                priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(rec.get("priority", ""), "⚪")
                print(f"  {i}. {priority_emoji} {rec.get('title', 'Unknown')}")
                print(f"     {rec.get('description', '')}")
                print(f"     Action: {rec.get('action', '')}")
                print()
        
        # Visualizations
        visualizations = results.get("visualizations", {})
        if visualizations:
            print(f"\n📊 VISUALIZATIONS CREATED:")
            for viz_name, viz_info in visualizations.items():
                if isinstance(viz_info, dict) and "files" in viz_info:
                    print(f"  • {viz_name}: {len(viz_info['files'])} files")
                elif isinstance(viz_info, dict) and "html_file" in viz_info:
                    print(f"  • {viz_name}: Interactive HTML")
        
        print("\n" + "="*100)
        print("End of Enhanced Import Analysis Report")
        print("="*100)


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Import Analysis Pipeline")
    parser.add_argument("--project-root", type=str, help="Project root directory")
    parser.add_argument("--target-dir", type=str, help="Target directory to analyze")
    parser.add_argument("--no-print", action="store_true", help="Don't print report to console")
    parser.add_argument("--no-save", action="store_true", help="Don't save report to file")
    parser.add_argument("--no-visualizations", action="store_true", help="Don't create visualizations")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = EnhancedImportAnalysisPipeline(
        project_root=args.project_root,
        enable_plugins=True
    )
    
    if args.verbose:
        pipeline.logger.setLevel("DEBUG")
    
    # Run analysis
    results = pipeline.run(
        target_directory=args.target_dir,
        save_report=not args.no_save,
        print_report=not args.no_print,
        create_visualizations=not args.no_visualizations
    )
    
    # Exit with error code if there was an error
    if "error" in results:
        sys.exit(1)


if __name__ == "__main__":
    main()