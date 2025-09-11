#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Demo Script for Code Quality Pipelines

This script demonstrates how to use the various theme-based pipeline classes
for comprehensive code quality analysis.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .base_pipeline import PipelineConfig
from .syntax_validation_pipeline import run_syntax_validation
from .import_analysis_pipeline import run_import_analysis
from .import_free_analysis_pipeline import run_import_free_analysis
from .dead_code_analysis_pipeline import run_dead_code_analysis
from .code_graph_pipeline import run_code_graph_analysis
from .complexity_analysis_pipeline import run_complexity_analysis
from .auto_fixer_pipeline import run_auto_fixer
from .function_import_analysis_pipeline import run_function_import_analysis


async def demo_all_pipelines(project_root: str = "/workspace"):
    """Demonstrate all pipeline types."""
    tprint("🚀 Code Quality Pipelines Demo")
    tprint("=" * 80)
    
    # Configuration
    config = PipelineConfig(
        project_root=project_root,
        parallel_execution=True,
        max_workers=2,
        verbose=True
    )
    
    pipelines = [
        ("Syntax Validation", run_syntax_validation),
        ("Import Analysis", run_import_analysis),
        ("Import-Free Analysis", run_import_free_analysis),
        ("Dead Code Analysis", run_dead_code_analysis),
        ("Code Graph Analysis", run_code_graph_analysis),
        ("Complexity Analysis", run_complexity_analysis),
        ("Function Import Analysis", run_function_import_analysis),
        ("Auto-Fixer (Dry Run)", lambda root, **kwargs: run_auto_fixer(root, dry_run=True, **kwargs))
    ]
    
    results = {}
    
    for name, pipeline_func in pipelines:
        tprint(f"\n🔍 Running {name} Pipeline...")
        tprint("-" * 50)
        
        try:
            # Create a copy of config dict without project_root to avoid argument conflicts
            config_dict = config.__dict__.copy()
            config_dict.pop('project_root', None)
            result = await pipeline_func(project_root, **config_dict)
            results[name] = result
            
            tprint(f"✅ {name} completed successfully")
            tprint(f"   Status: {result.status.value}")
            tprint(f"   Duration: {result.duration_seconds:.2f}s")
            tprint(f"   Stages: {len(result.stages)}")
            
            if result.errors:
                tprint(f"   Errors: {len(result.errors)}")
                for error in result.errors[:3]:  # Show first 3 errors
                    tprint(f"     - {error}")
            
            if result.warnings:
                tprint(f"   Warnings: {len(result.warnings)}")
                for warning in result.warnings[:3]:  # Show first 3 warnings
                    tprint(f"     - {warning}")
                    
        except Exception as e:
            tprint(f"❌ {name} failed: {e}")
            results[name] = None
    
    # Summary
    tprint(f"\n📊 Pipeline Execution Summary")
    tprint("=" * 80)
    
    successful = sum(1 for r in results.values() if r is not None)
    total = len(results)
    
    tprint(f"Successful: {successful}/{total}")
    tprint(f"Failed: {total - successful}/{total}")
    
    for name, result in results.items():
        if result:
            tprint(f"✅ {name}: {result.duration_seconds:.2f}s")
        else:
            tprint(f"❌ {name}: Failed")
    
    return results


async def demo_single_pipeline(pipeline_name: str, project_root: str = "/workspace"):
    """Demonstrate a single pipeline with detailed output."""
    tprint(f"🔍 Running {pipeline_name} Pipeline")
    tprint("=" * 80)
    
    config = PipelineConfig(
        project_root=project_root,
        verbose=True,
        log_level="INFO"
    )
    
    pipeline_map = {
        "syntax": run_syntax_validation,
        "import": run_import_analysis,
        "import-free": run_import_free_analysis,
        "dead-code": run_dead_code_analysis,
        "graph": run_code_graph_analysis,
        "complexity": run_complexity_analysis,
        "function-import": run_function_import_analysis,
        "autofixer": lambda root, **kwargs: run_auto_fixer(root, dry_run=True, **kwargs)
    }
    
    if pipeline_name not in pipeline_map:
        tprint(f"❌ Unknown pipeline: {pipeline_name}")
        tprint(f"Available pipelines: {', '.join(pipeline_map.keys())}")
        return None
    
    try:
        # Create a copy of config dict without project_root to avoid argument conflicts
        config_dict = config.__dict__.copy()
        config_dict.pop('project_root', None)
        result = await pipeline_map[pipeline_name](project_root, **config_dict)
        
        tprint(f"\n📊 {pipeline_name.title()} Pipeline Results")
        tprint("-" * 50)
        tprint(f"Status: {result.status.value}")
        tprint(f"Duration: {result.duration_seconds:.2f}s")
        tprint(f"Stages: {len(result.stages)}")
        
        tprint(f"\n📋 Stage Details:")
        for stage in result.stages:
            tprint(f"  {stage.stage.value}: {stage.status.value} ({stage.duration_seconds:.2f}s)")
            if stage.errors:
                for error in stage.errors:
                    tprint(f"    ❌ {error}")
            if stage.warnings:
                for warning in stage.warnings:
                    tprint(f"    ⚠️  {warning}")
        
        if result.errors:
            tprint(f"\n❌ Pipeline Errors:")
            for error in result.errors:
                tprint(f"  - {error}")
        
        if result.warnings:
            tprint(f"\n⚠️  Pipeline Warnings:")
            for warning in result.warnings:
                tprint(f"  - {warning}")
        
        return result
        
    except Exception as e:
        tprint(f"❌ Pipeline failed: {e}")
        return None


def main():
    """Main entry point for the demo script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo Code Quality Pipelines")
    parser.add_argument("--pipeline", "-p", help="Run specific pipeline (syntax, import, import-free, dead-code, graph, complexity, autofixer)")
    parser.add_argument("--project-root", "-r", default="/workspace", help="Project root directory")
    parser.add_argument("--all", "-a", action="store_true", help="Run all pipelines")
    
    args = parser.parse_args()
    
    if args.all:
        asyncio.run(demo_all_pipelines(args.project_root))
    elif args.pipeline:
        asyncio.run(demo_single_pipeline(args.pipeline, args.project_root))
    else:
        tprint("Please specify --pipeline <name> or --all")
        tprint("Available pipelines: syntax, import, import-free, dead-code, graph, complexity, function-import, autofixer")


if __name__ == "__main__":
    main()