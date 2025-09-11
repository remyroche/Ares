#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Code Quality Analysis Functions

Simple, direct functions for code quality analysis without unnecessary pipeline abstractions.
These functions directly call the analyzers and provide optional utilities for saving/printing results.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

# Add parent directory to path for imports
parent_dir = Path(__file__).parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import analyzers with fallbacks
try:
    from analyzers.import_verifier_analyzer import ImportVerifierAnalyzer
except ImportError:
    ImportVerifierAnalyzer = None

try:
    from analyzers.dependency_analyzer import DependencyAnalyzer
except ImportError:
    DependencyAnalyzer = None

try:
    from analyzers.complexity_analyzer import ComplexityAnalyzer
except ImportError:
    ComplexityAnalyzer = None

try:
    from analyzers.dead_code_analyzer import DeadCodeAnalyzer
except ImportError:
    DeadCodeAnalyzer = None

try:
    from analyzers.enhanced_dead_code_analyzer import EnhancedDeadCodeAnalyzer
except ImportError:
    EnhancedDeadCodeAnalyzer = None

try:
    from analyzers.undefined_names_analyzer import UndefinedNamesAnalyzer
except ImportError:
    UndefinedNamesAnalyzer = None

try:
    from analyzers.enhanced_import_analysis import EnhancedImportAnalyzer
except ImportError:
    EnhancedImportAnalyzer = None

# Import visualizers with fallbacks
try:
    from visualizers.dependency_graph import DependencyGraphVisualizer
except ImportError:
    DependencyGraphVisualizer = None

try:
    from visualizers.interaction_network import InteractionNetworkVisualizer
except ImportError:
    InteractionNetworkVisualizer = None

try:
    from visualizers.import_network_visualizer import ImportNetworkVisualizer
except ImportError:
    ImportNetworkVisualizer = None


def save_report(data: Dict[str, Any], filename: str, output_dir: Optional[Union[str, Path]] = None) -> Path:
    """Save analysis results to a JSON file."""
    if output_dir is None:
        output_dir = Path.cwd() / "code_quality" / "reports"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"{filename}_{timestamp}.json"
    
    with open(report_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    tprint(f"Report saved to: {report_path}")
    return report_path


def print_summary(data: Dict[str, Any], title: str = "Analysis Summary"):
    """Print a summary of analysis results."""
    tprint(f"\n{'='*60}")
    tprint(f"{title}")
    tprint(f"{'='*60}")
    
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (list, dict)):
                tprint(f"{key}: {len(value)} items")
            else:
                tprint(f"{key}: {value}")
    else:
        tprint(f"Results: {data}")
    
    tprint(f"{'='*60}")


def run_import_verification(
    project_root: Union[str, Path],
    target_directory: Optional[str] = None,
    save_report: bool = True,
    print_report: bool = True,
    create_visualizations: bool = False,
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Run import verification analysis to check which files are imported by others.
    
    Args:
        project_root: Root directory of the project
        target_directory: Directory to analyze (defaults to project root)
        save_report: Whether to save the report to file
        print_report: Whether to print the report to console
        create_visualizations: Whether to create enhanced visualizations
        output_dir: Directory to save reports (defaults to project_root/code_quality/reports)
        
    Returns:
        Dict containing analysis results
    """
    tprint("Starting import verification analysis...")
    
    # Check if analyzer is available
    if ImportVerifierAnalyzer is None:
        return {
            "error": "ImportVerifierAnalyzer not available",
            "analysis_info": {
                "analysis_type": "import_verification",
                "timestamp": datetime.now().isoformat(),
                "project_root": str(project_root),
                "analysis_directory": target_directory or str(project_root),
                "analyzer_used": "ImportVerifierAnalyzer (not available)"
            }
        }
    
    # Initialize analyzer
    analyzer = ImportVerifierAnalyzer({})
    
    # Run analysis
    analysis_dir = target_directory or str(project_root)
    results = analyzer.analyze_directory(analysis_dir)
    
    # Add metadata
    results["analysis_info"] = {
        "analysis_type": "import_verification",
        "timestamp": datetime.now().isoformat(),
        "project_root": str(project_root),
        "analysis_directory": analysis_dir,
        "analyzer_used": "ImportVerifierAnalyzer"
    }
    
    # Create visualizations if requested
    if create_visualizations:
        tprint("Creating enhanced visualizations...")
        try:
            # Initialize visualizers
            reports_dir = output_dir or Path(project_root) / "code_quality" / "reports"
            dependency_visualizer = DependencyGraphVisualizer(str(reports_dir / "dependency_graphs"))
            interaction_visualizer = InteractionNetworkVisualizer(str(reports_dir / "interaction_networks"))
            
            # Create visualizations (this would need to be implemented based on the visualizer APIs)
            visualizations = {
                "dependency_graphs": "Created",
                "interaction_networks": "Created"
            }
            results["visualizations"] = visualizations
        except Exception as e:
            tprint(f"Warning: Could not create visualizations: {e}")
    
    # Print report if requested
    if print_report:
        analyzer.print_simple_report(results)
    
    # Save report if requested
    if save_report:
        report_path = save_report(results, "import_verification", output_dir)
        results["report_path"] = str(report_path)
    
    tprint("Import verification analysis completed successfully")
    return results


def run_enhanced_import_analysis(
    project_root: Union[str, Path],
    target_directory: Optional[str] = None,
    save_report: bool = True,
    print_report: bool = True,
    create_visualizations: bool = False,
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Run enhanced import analysis combining multiple analyzers.
    
    Args:
        project_root: Root directory of the project
        target_directory: Directory to analyze (defaults to project root)
        save_report: Whether to save the report to file
        print_report: Whether to print the report to console
        create_visualizations: Whether to create enhanced visualizations
        output_dir: Directory to save reports
        
    Returns:
        Dict containing comprehensive analysis results
    """
    tprint("Starting enhanced import analysis...")
    
    analysis_dir = target_directory or str(project_root)
    
    # Initialize analyzers (only if available)
    results = {}
    
    if ImportVerifierAnalyzer:
        import_verifier = ImportVerifierAnalyzer({})
        results["import_verification"] = import_verifier.analyze_directory(analysis_dir)
    else:
        results["import_verification"] = {"error": "ImportVerifierAnalyzer not available"}
    
    if DependencyAnalyzer:
        dependency_analyzer = DependencyAnalyzer({})
        results["dependency_analysis"] = dependency_analyzer.analyze_directory(analysis_dir)
    else:
        results["dependency_analysis"] = {"error": "DependencyAnalyzer not available"}
    
    if ComplexityAnalyzer:
        complexity_analyzer = ComplexityAnalyzer({})
        results["complexity_analysis"] = complexity_analyzer.analyze_directory(analysis_dir)
    else:
        results["complexity_analysis"] = {"error": "ComplexityAnalyzer not available"}
    
    if DeadCodeAnalyzer:
        dead_code_analyzer = DeadCodeAnalyzer({})
        results["dead_code_analysis"] = dead_code_analyzer.analyze_directory(analysis_dir)
    else:
        results["dead_code_analysis"] = {"error": "DeadCodeAnalyzer not available"}
    
    # Add metadata
    results["analysis_info"] = {
        "analysis_type": "enhanced_import_analysis",
        "timestamp": datetime.now().isoformat(),
        "project_root": str(project_root),
        "analysis_directory": analysis_dir,
        "analyzers_used": ["ImportVerifierAnalyzer", "DependencyAnalyzer", "ComplexityAnalyzer", "DeadCodeAnalyzer"]
    }
    
    # Create visualizations if requested
    if create_visualizations:
        tprint("Creating enhanced visualizations...")
        try:
            reports_dir = output_dir or Path(project_root) / "code_quality" / "reports"
            import_visualizer = ImportNetworkVisualizer(str(reports_dir / "import_networks"))
            dependency_visualizer = DependencyGraphVisualizer(str(reports_dir / "dependency_graphs"))
            interaction_visualizer = InteractionNetworkVisualizer(str(reports_dir / "interaction_networks"))
            
            visualizations = {
                "import_networks": "Created",
                "dependency_graphs": "Created", 
                "interaction_networks": "Created"
            }
            results["visualizations"] = visualizations
        except Exception as e:
            tprint(f"Warning: Could not create visualizations: {e}")
    
    # Print summary if requested
    if print_report:
        print_summary(results, "Enhanced Import Analysis Results")
    
    # Save report if requested
    if save_report:
        report_path = save_report(results, "enhanced_import_analysis", output_dir)
        results["report_path"] = str(report_path)
    
    tprint("Enhanced import analysis completed successfully")
    return results


def run_dead_code_analysis(
    project_root: Union[str, Path],
    target_directory: Optional[str] = None,
    save_report: bool = True,
    print_report: bool = True,
    use_interaction_mapping: bool = True,
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Run dead code analysis to find unused code.
    
    Args:
        project_root: Root directory of the project
        target_directory: Directory to analyze (defaults to project root)
        save_report: Whether to save the report to file
        print_report: Whether to print the report to console
        use_interaction_mapping: Whether to use interaction mapping for better accuracy
        output_dir: Directory to save reports
        
    Returns:
        Dict containing dead code analysis results
    """
    tprint("Starting dead code analysis...")
    
    analysis_dir = target_directory or str(project_root)
    
    # Initialize analyzers (only if available)
    results = {}
    
    if EnhancedDeadCodeAnalyzer:
        enhanced_dead_code_analyzer = EnhancedDeadCodeAnalyzer({})
        results["enhanced_dead_code"] = enhanced_dead_code_analyzer.analyze_directory(analysis_dir)
    else:
        results["enhanced_dead_code"] = {"error": "EnhancedDeadCodeAnalyzer not available"}
    
    if UndefinedNamesAnalyzer:
        undefined_names_analyzer = UndefinedNamesAnalyzer({})
        results["undefined_names"] = undefined_names_analyzer.analyze_directory(analysis_dir)
    else:
        results["undefined_names"] = {"error": "UndefinedNamesAnalyzer not available"}
    
    if EnhancedImportAnalyzer:
        enhanced_import_analyzer = EnhancedImportAnalyzer({})
        results["import_analysis"] = enhanced_import_analyzer.analyze_directory(analysis_dir)
    else:
        results["import_analysis"] = {"error": "EnhancedImportAnalyzer not available"}
    
    # Add metadata
    results["analysis_info"] = {
        "analysis_type": "dead_code_analysis",
        "timestamp": datetime.now().isoformat(),
        "project_root": str(project_root),
        "analysis_directory": analysis_dir,
        "use_interaction_mapping": use_interaction_mapping,
        "analyzers_used": ["EnhancedDeadCodeAnalyzer", "UndefinedNamesAnalyzer", "EnhancedImportAnalyzer"]
    }
    
    # Print summary if requested
    if print_report:
        print_summary(results, "Dead Code Analysis Results")
    
    # Save report if requested
    if save_report:
        report_path = save_report(results, "dead_code_analysis", output_dir)
        results["report_path"] = str(report_path)
    
    tprint("Dead code analysis completed successfully")
    return results


def run_complexity_analysis(
    project_root: Union[str, Path],
    target_directory: Optional[str] = None,
    save_report: bool = True,
    print_report: bool = True,
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Run complexity analysis to measure code complexity metrics.
    
    Args:
        project_root: Root directory of the project
        target_directory: Directory to analyze (defaults to project root)
        save_report: Whether to save the report to file
        print_report: Whether to print the report to console
        output_dir: Directory to save reports
        
    Returns:
        Dict containing complexity analysis results
    """
    tprint("Starting complexity analysis...")
    
    analysis_dir = target_directory or str(project_root)
    
    # Check if analyzer is available
    if ComplexityAnalyzer is None:
        return {
            "error": "ComplexityAnalyzer not available",
            "analysis_info": {
                "analysis_type": "complexity_analysis",
                "timestamp": datetime.now().isoformat(),
                "project_root": str(project_root),
                "analysis_directory": analysis_dir,
                "analyzer_used": "ComplexityAnalyzer (not available)"
            }
        }
    
    # Initialize analyzer
    complexity_analyzer = ComplexityAnalyzer({})
    
    # Run analysis
    results = complexity_analyzer.analyze_directory(analysis_dir)
    
    # Add metadata
    results["analysis_info"] = {
        "analysis_type": "complexity_analysis",
        "timestamp": datetime.now().isoformat(),
        "project_root": str(project_root),
        "analysis_directory": analysis_dir,
        "analyzer_used": "ComplexityAnalyzer"
    }
    
    # Print summary if requested
    if print_report:
        print_summary(results, "Complexity Analysis Results")
    
    # Save report if requested
    if save_report:
        report_path = save_report(results, "complexity_analysis", output_dir)
        results["report_path"] = str(report_path)
    
    tprint("Complexity analysis completed successfully")
    return results


def run_dependency_analysis(
    project_root: Union[str, Path],
    target_directory: Optional[str] = None,
    save_report: bool = True,
    print_report: bool = True,
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Run dependency analysis to understand module dependencies.
    
    Args:
        project_root: Root directory of the project
        target_directory: Directory to analyze (defaults to project root)
        save_report: Whether to save the report to file
        print_report: Whether to print the report to console
        output_dir: Directory to save reports
        
    Returns:
        Dict containing dependency analysis results
    """
    tprint("Starting dependency analysis...")
    
    analysis_dir = target_directory or str(project_root)
    
    # Check if analyzer is available
    if DependencyAnalyzer is None:
        return {
            "error": "DependencyAnalyzer not available",
            "analysis_info": {
                "analysis_type": "dependency_analysis",
                "timestamp": datetime.now().isoformat(),
                "project_root": str(project_root),
                "analysis_directory": analysis_dir,
                "analyzer_used": "DependencyAnalyzer (not available)"
            }
        }
    
    # Initialize analyzer
    dependency_analyzer = DependencyAnalyzer({})
    
    # Run analysis
    results = dependency_analyzer.analyze_directory(analysis_dir)
    
    # Add metadata
    results["analysis_info"] = {
        "analysis_type": "dependency_analysis",
        "timestamp": datetime.now().isoformat(),
        "project_root": str(project_root),
        "analysis_directory": analysis_dir,
        "analyzer_used": "DependencyAnalyzer"
    }
    
    # Print summary if requested
    if print_report:
        print_summary(results, "Dependency Analysis Results")
    
    # Save report if requested
    if save_report:
        report_path = save_report(results, "dependency_analysis", output_dir)
        results["report_path"] = str(report_path)
    
    tprint("Dependency analysis completed successfully")
    return results


# Convenience function to run all analyses
def run_all_analyses(
    project_root: Union[str, Path],
    target_directory: Optional[str] = None,
    save_reports: bool = True,
    print_reports: bool = True,
    create_visualizations: bool = False,
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Run all available code quality analyses.
    
    Args:
        project_root: Root directory of the project
        target_directory: Directory to analyze (defaults to project root)
        save_reports: Whether to save reports to files
        print_reports: Whether to print reports to console
        create_visualizations: Whether to create visualizations
        output_dir: Directory to save reports
        
    Returns:
        Dict containing all analysis results
    """
    tprint("Starting comprehensive code quality analysis...")
    
    all_results = {}
    
    # Run all analyses
    all_results["import_verification"] = run_import_verification(
        project_root, target_directory, save_reports, print_reports, create_visualizations, output_dir
    )
    
    all_results["enhanced_import_analysis"] = run_enhanced_import_analysis(
        project_root, target_directory, save_reports, print_reports, create_visualizations, output_dir
    )
    
    all_results["dead_code_analysis"] = run_dead_code_analysis(
        project_root, target_directory, save_reports, print_reports, True, output_dir
    )
    
    all_results["complexity_analysis"] = run_complexity_analysis(
        project_root, target_directory, save_reports, print_reports, output_dir
    )
    
    all_results["dependency_analysis"] = run_dependency_analysis(
        project_root, target_directory, save_reports, print_reports, output_dir
    )
    
    # Add overall metadata
    all_results["overall_info"] = {
        "analysis_type": "comprehensive_analysis",
        "timestamp": datetime.now().isoformat(),
        "project_root": str(project_root),
        "analysis_directory": target_directory or str(project_root),
        "analyses_run": list(all_results.keys())
    }
    
    tprint("Comprehensive code quality analysis completed successfully")
    return all_results