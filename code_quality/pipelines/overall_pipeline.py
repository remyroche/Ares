#!/usr/bin/env python3
"""
Overall Pipeline - Master Orchestrator

This is the main pipeline that orchestrates all other specialized pipelines.
It provides a single entry point to run comprehensive code quality analysis.

Usage:
    python pipelines/overall_pipeline.py
    python pipelines/overall_pipeline.py --project-root /path/to/project
    python pipelines/overall_pipeline.py --pipelines complexity,dead_code
    python pipelines/overall_pipeline.py --all
"""

import sys
import argparse
import subprocess
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import core components (ONLY orchestration related)
from core.config import get_default_config
from utils.dependency_manager import DependencyManager
from utils.file_utils import find_python_files, get_directory_stats
from utils.progress import ProgressTracker
from utils.report_aggregator import ReportAggregator

# Import CLI and runners (ONLY orchestration related)
from cli import main as cli_main
from run_full_pipeline import run_full_pipeline
from quick_start import quick_validate

# Import plugin system (ONLY orchestration related)
from plugins.plugin_manager import PluginManager
from plugins.plugin_registry import PluginRegistry
from plugins.base_plugin import PluginCategory, PluginPriority


class OverallPipeline:
    """Master orchestrator for all code quality pipelines."""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.pipelines_dir = Path(__file__).parent
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        
        # Define available pipelines with their commands
        self.available_pipelines = {
            "complexity": {
                "script": "complexity_pipeline.py",
                "default_args": ["--analysis-type", "cyclomatic"],
                "description": "Complexity analysis (cyclomatic, cognitive, maintainability, metrics)"
            },
            "dead_code": {
                "script": "dead_code_pipeline.py", 
                "default_args": ["--analysis-type", "enhanced", "--auto-fix"],
                "description": "Dead code detection and removal"
            },
            "auto_fixer": {
                "script": "auto_fixer_pipeline.py",
                "default_args": ["--fix-type", "imports", "--conservative"],
                "description": "Automatic code fixing (imports, syntax, type hints, etc.)"
            },
            "interaction_mapping": {
                "script": "interaction_mapping_pipeline.py",
                "default_args": ["--analysis-type", "call_graph"],
                "description": "Code interaction and dependency mapping"
            },
            "import_free_analysis": {
                "script": "import_free_analysis_pipeline.py",
                "default_args": ["--analysis-type", "syntax"],
                "description": "Import-free code analysis (syntax, structure, patterns)"
            },
            "unified_enhanced": {
                "script": "pipeline_unified_enhanced.py",
                "default_args": [],
                "description": "Comprehensive analysis with imports"
            }
        }
    
    def run_pipeline(self, pipeline_name: str, custom_args: List[str] = None) -> Dict[str, Any]:
        """Run a specific pipeline."""
        if pipeline_name not in self.available_pipelines:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")
        
        pipeline_info = self.available_pipelines[pipeline_name]
        script_path = self.pipelines_dir / pipeline_info["script"]
        
        if not script_path.exists():
            raise FileNotFoundError(f"Pipeline script not found: {script_path}")
        
        # Build command
        cmd = [sys.executable, str(script_path)]
        cmd.extend(["--project-root", str(self.project_root)])
        
        # Add default args or custom args
        if custom_args:
            cmd.extend(custom_args)
        else:
            cmd.extend(pipeline_info["default_args"])
        
        print(f"🚀 Running {pipeline_name} pipeline...")
        print(f"   Command: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            # Run the pipeline
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.pipelines_dir.parent
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            pipeline_result = {
                "pipeline": pipeline_name,
                "status": "success" if result.returncode == 0 else "failed",
                "return_code": result.returncode,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(cmd)
            }
            
            if result.returncode == 0:
                print(f"✅ {pipeline_name} pipeline completed successfully ({duration:.2f}s)")
            else:
                print(f"❌ {pipeline_name} pipeline failed ({duration:.2f}s)")
                if result.stderr:
                    print(f"   Error: {result.stderr.strip()}")
            
            return pipeline_result
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            pipeline_result = {
                "pipeline": pipeline_name,
                "status": "error",
                "return_code": -1,
                "duration": duration,
                "error": str(e),
                "command": " ".join(cmd)
            }
            
            print(f"💥 {pipeline_name} pipeline crashed ({duration:.2f}s)")
            print(f"   Error: {e}")
            
            return pipeline_result
    
    def run_pipelines(self, pipeline_names: List[str], custom_args: Dict[str, List[str]] = None) -> Dict[str, Any]:
        """Run multiple pipelines."""
        if custom_args is None:
            custom_args = {}
        
        print(f"🎯 Running {len(pipeline_names)} pipelines on {self.project_root}")
        print("=" * 60)
        
        start_time = time.time()
        results = {
            "overall_start_time": start_time,
            "project_root": str(self.project_root),
            "pipelines_requested": pipeline_names,
            "pipeline_results": {},
            "summary": {
                "total_pipelines": len(pipeline_names),
                "successful": 0,
                "failed": 0,
                "total_duration": 0.0
            }
        }
        
        for i, pipeline_name in enumerate(pipeline_names, 1):
            print(f"\n[{i}/{len(pipeline_names)}] {pipeline_name.upper()}")
            print("-" * 40)
            
            custom_pipeline_args = custom_args.get(pipeline_name, None)
            pipeline_result = self.run_pipeline(pipeline_name, custom_pipeline_args)
            results["pipeline_results"][pipeline_name] = pipeline_result
            
            # Update summary
            if pipeline_result["status"] == "success":
                results["summary"]["successful"] += 1
            else:
                results["summary"]["failed"] += 1
        
        end_time = time.time()
        results["overall_end_time"] = end_time
        results["summary"]["total_duration"] = end_time - start_time
        
        # Print summary
        self.print_summary(results)
        
        return results
    
    def run_all_pipelines(self) -> Dict[str, Any]:
        """Run all available pipelines."""
        all_pipeline_names = list(self.available_pipelines.keys())
        return self.run_pipelines(all_pipeline_names)
    
    def print_summary(self, results: Dict[str, Any]):
        """Print execution summary."""
        print("\n" + "=" * 60)
        print("📊 OVERALL PIPELINE EXECUTION SUMMARY")
        print("=" * 60)
        
        summary = results["summary"]
        print(f"Project: {results['project_root']}")
        print(f"Total pipelines: {summary['total_pipelines']}")
        print(f"Successful: {summary['successful']} ✅")
        print(f"Failed: {summary['failed']} ❌")
        print(f"Total duration: {summary['total_duration']:.2f} seconds")
        
        if summary['failed'] > 0:
            print(f"\n❌ Failed pipelines:")
            for pipeline_name, result in results["pipeline_results"].items():
                if result["status"] != "success":
                    print(f"   - {pipeline_name}: {result.get('error', 'Unknown error')}")
        
        print(f"\n📄 Detailed results saved to: overall_pipeline_results_{self.timestamp}.json")
    
    def save_results(self, results: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """Save results to JSON file."""
        if not output_file:
            output_file = f"overall_pipeline_results_{self.timestamp}.json"
        
        output_path = self.project_root / output_file
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return str(output_path)
    
    def list_pipelines(self):
        """List all available pipelines."""
        print("📋 Available Pipelines:")
        print("=" * 50)
        
        for name, info in self.available_pipelines.items():
            print(f"• {name}")
            print(f"  Description: {info['description']}")
            print(f"  Script: {info['script']}")
            print(f"  Default args: {' '.join(info['default_args'])}")
            print()


def main():
    """Main function for overall pipeline."""
    parser = argparse.ArgumentParser(
        description="Overall Pipeline - Master orchestrator for all code quality pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all pipelines
  python pipelines/overall_pipeline.py --all
  
  # Run specific pipelines
  python pipelines/overall_pipeline.py --pipelines complexity,dead_code,auto_fixer
  
  # Run on specific project
  python pipelines/overall_pipeline.py --project-root /path/to/project --all
  
  # List available pipelines
  python pipelines/overall_pipeline.py --list
  
  # Run with custom arguments
  python pipelines/overall_pipeline.py --pipelines complexity --custom-args complexity:--analysis-type,metrics
        """
    )
    
    parser.add_argument("--project-root", "-p",
                       help="Project root directory (default: current directory)")
    parser.add_argument("--pipelines", "-t",
                       help="Comma-separated list of pipelines to run")
    parser.add_argument("--all", action="store_true",
                       help="Run all available pipelines")
    parser.add_argument("--list", action="store_true",
                       help="List all available pipelines")
    parser.add_argument("--output", "-o",
                       help="Output file for results (default: auto-generated)")
    parser.add_argument("--custom-args",
                       help="Custom arguments for specific pipelines (format: pipeline:arg1,arg2)")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = OverallPipeline(args.project_root)
    
    # Handle list command
    if args.list:
        pipeline.list_pipelines()
        return 0
    
    # Determine which pipelines to run
    if args.all:
        pipeline_names = list(pipeline.available_pipelines.keys())
    elif args.pipelines:
        pipeline_names = [p.strip() for p in args.pipelines.split(",")]
        # Validate pipeline names
        invalid_pipelines = [p for p in pipeline_names if p not in pipeline.available_pipelines]
        if invalid_pipelines:
            print(f"Error: Unknown pipelines: {', '.join(invalid_pipelines)}")
            print("Use --list to see available pipelines")
            return 1
    else:
        print("Error: Must specify --pipelines or --all")
        print("Use --list to see available pipelines")
        return 1
    
    # Parse custom arguments
    custom_args = {}
    if args.custom_args:
        for arg_spec in args.custom_args.split(";"):
            if ":" in arg_spec:
                pipeline_name, args_str = arg_spec.split(":", 1)
                pipeline_name = pipeline_name.strip()
                if pipeline_name in pipeline.available_pipelines:
                    custom_args[pipeline_name] = [arg.strip() for arg in args_str.split(",")]
    
    try:
        # Run pipelines
        results = pipeline.run_pipelines(pipeline_names, custom_args)
        
        # Save results
        output_file = pipeline.save_results(results, args.output)
        
        return 0 if results["summary"]["failed"] == 0 else 1
        
    except Exception as e:
        print(f"Error running overall pipeline: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())