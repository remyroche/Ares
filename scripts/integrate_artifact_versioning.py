#!/usr/bin/env python3
"""
Integration Script: Complete Artifact Versioning Integration

This script systematically integrates the artifact versioning system across all
sub-pipeline stages and steps in the Ares pipeline.
"""

import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PipelineStage:
    """Represents a pipeline stage with its sub-pipelines."""
    name: str
    directory: str
    sub_pipelines: List[str]


class ArtifactVersioningIntegrator:
    """Integrates artifact versioning across all pipeline stages."""
    
    def __init__(self, workspace_root: str = "/workspace"):
        """Initialize the integrator."""
        self.workspace_root = Path(workspace_root)
        self.stages = self._define_pipeline_stages()
        
        # Import statements to add
        self.imports = [
            "from src.utils.enhanced_artifact_manager import get_artifact_manager",
            "from src.utils.artifact_pickup_utils import get_artifact_pickup_utils", 
            "from src.utils.version_manager import get_version_manager"
        ]
        
        # Initialization code to add to __init__ methods
        self.init_code = [
            "        # Initialize artifact and version managers",
            "        self.artifact_manager = get_artifact_manager()",
            "        self.pickup_utils = get_artifact_pickup_utils()",
            "        self.version_manager = get_version_manager()"
        ]
    
    def _define_pipeline_stages(self) -> List[PipelineStage]:
        """Define all pipeline stages and their sub-pipelines."""
        return [
            PipelineStage(
                name="DATA_COLLECTION",
                directory="src/training/steps/data_collection",
                sub_pipelines=[
                    "data_download", "data_conversion", "data_validation", 
                    "data_preparation", "feature_engineering", "data_quality_check",
                    "data_storage", "data_monitoring", "data_integration", "data_export"
                ]
            ),
            PipelineStage(
                name="MARKET_ANALYSIS", 
                directory="src/training/steps/market_analysis",
                sub_pipelines=[
                    "sr_detection", "sr_clustering", "sr_ml_learning", "hmm_clustering",
                    "hmm_regime_discovery", "regime_data_splitting", "triple_barrier_labeling",
                    "feature_lookback_optimization", "fractional_differentiation", "cross_timeframe_analysis"
                ]
            ),
            PipelineStage(
                name="MODEL_TRAINING",
                directory="src/training/steps/model_training", 
                sub_pipelines=[
                    "general_model_training", "analyst_model_training", "tactician_model_training",
                    "hmm_training", "ensemble_training", "multi_timeframe_training",
                    "regime_specific_training", "model_validation", "model_persistence", "model_evaluation"
                ]
            ),
            PipelineStage(
                name="BACKTESTING",
                directory="src/training/steps/backtesting",
                sub_pipelines=[
                    "basic_backtesting_pre", "final_parameters_optimization", "basic_backtesting_post",
                    "walk_forward_validation", "monte_carlo_simulation", "ab_testing",
                    "performance_analytics", "risk_analysis", "trade_analysis", "portfolio_analysis", "reporting"
                ]
            )
        ]
    
    def find_python_files(self, directory: Path) -> List[Path]:
        """Find all Python files in a directory."""
        python_files = []
        if directory.exists():
            for file_path in directory.rglob("*.py"):
                # Skip __pycache__ and test files for now
                if "__pycache__" not in str(file_path) and "test_" not in file_path.name:
                    python_files.append(file_path)
        return python_files
    
    def needs_integration(self, file_path: Path) -> bool:
        """Check if a file needs artifact versioning integration."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if it already has the imports
            if any(import_stmt in content for import_stmt in self.imports):
                return False
            
            # Check if it has artifact-related operations
            artifact_patterns = [
                r'\.to_parquet\(',
                r'\.to_pickle\(',
                r'\.to_json\(',
                r'joblib\.dump\(',
                r'pickle\.dump\(',
                r'pd\.read_parquet\(',
                r'pd\.read_pickle\(',
                r'joblib\.load\(',
                r'pickle\.load\(',
                r'save.*\.pkl',
                r'save.*\.parquet',
                r'save.*\.json',
                r'load.*\.pkl',
                r'load.*\.parquet',
                r'load.*\.json'
            ]
            
            return any(re.search(pattern, content) for pattern in artifact_patterns)
            
        except Exception as e:
            print(f"Error checking {file_path}: {e}")
            return False
    
    def add_imports(self, content: str) -> str:
        """Add artifact versioning imports to file content."""
        lines = content.split('\n')
        
        # Find the last import statement
        last_import_idx = -1
        for i, line in enumerate(lines):
            if line.strip().startswith(('import ', 'from ')):
                last_import_idx = i
        
        if last_import_idx >= 0:
            # Insert imports after the last import
            for import_stmt in reversed(self.imports):
                lines.insert(last_import_idx + 1, import_stmt)
        else:
            # No imports found, add at the beginning
            for import_stmt in reversed(self.imports):
                lines.insert(0, import_stmt)
        
        return '\n'.join(lines)
    
    def add_initialization(self, content: str) -> str:
        """Add artifact manager initialization to __init__ methods."""
        lines = content.split('\n')
        
        # Find __init__ methods
        for i, line in enumerate(lines):
            if re.match(r'\s*def __init__\(', line):
                # Find the end of the __init__ method
                init_end = self._find_method_end(lines, i)
                
                # Check if initialization code is already present
                init_content = '\n'.join(lines[i:init_end])
                if 'self.artifact_manager = get_artifact_manager()' in init_content:
                    continue
                
                # Add initialization code before the end of __init__
                for init_line in reversed(self.init_code):
                    lines.insert(init_end, init_line)
        
        return '\n'.join(lines)
    
    def _find_method_end(self, lines: List[str], start_idx: int) -> int:
        """Find the end of a method definition."""
        indent_level = len(lines[start_idx]) - len(lines[start_idx].lstrip())
        
        for i in range(start_idx + 1, len(lines)):
            line = lines[i]
            if line.strip() == '':
                continue
            
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent_level and line.strip():
                return i
        
        return len(lines)
    
    def update_artifact_operations(self, content: str) -> str:
        """Update artifact save/load operations to use versioned system."""
        # Pattern replacements for common artifact operations
        replacements = [
            # Save operations
            (r'(\w+)\.to_parquet\(([^)]+)\)', r'self.artifact_manager.save_artifact(\1, "\2", ".parquet", "artifacts")'),
            (r'(\w+)\.to_pickle\(([^)]+)\)', r'self.artifact_manager.save_artifact(\1, "\2", ".pkl", "artifacts")'),
            (r'(\w+)\.to_json\(([^)]+)\)', r'self.artifact_manager.save_artifact(\1, "\2", ".json", "artifacts")'),
            
            # Load operations
            (r'pd\.read_parquet\(([^)]+)\)', r'self.pickup_utils.load_most_recent_artifact("data", "artifacts", extension=".parquet")[0]'),
            (r'pd\.read_pickle\(([^)]+)\)', r'self.pickup_utils.load_most_recent_artifact("data", "artifacts", extension=".pkl")[0]'),
        ]
        
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content)
        
        return content
    
    def integrate_file(self, file_path: Path) -> bool:
        """Integrate artifact versioning into a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Skip if already integrated
            if any(import_stmt in original_content for import_stmt in self.imports):
                return False
            
            # Add imports
            content = self.add_imports(original_content)
            
            # Add initialization
            content = self.add_initialization(content)
            
            # Update artifact operations (basic patterns)
            content = self.update_artifact_operations(content)
            
            # Write back if changed
            if content != original_content:
                # Create backup
                backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
                shutil.copy2(file_path, backup_path)
                
                # Write updated content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"✅ Integrated: {file_path}")
                return True
            else:
                print(f"⏭️ No changes needed: {file_path}")
                return False
                
        except Exception as e:
            print(f"❌ Error integrating {file_path}: {e}")
            return False
    
    def integrate_stage(self, stage: PipelineStage) -> Dict[str, int]:
        """Integrate artifact versioning for a pipeline stage."""
        print(f"\n🔄 Integrating {stage.name} stage...")
        
        stage_dir = self.workspace_root / stage.directory
        python_files = self.find_python_files(stage_dir)
        
        results = {
            "total_files": len(python_files),
            "integrated": 0,
            "skipped": 0,
            "errors": 0
        }
        
        for file_path in python_files:
            if self.needs_integration(file_path):
                if self.integrate_file(file_path):
                    results["integrated"] += 1
                else:
                    results["skipped"] += 1
            else:
                results["skipped"] += 1
        
        print(f"📊 {stage.name} Results: {results['integrated']} integrated, {results['skipped']} skipped")
        return results
    
    def integrate_all_stages(self) -> Dict[str, Dict[str, int]]:
        """Integrate artifact versioning across all pipeline stages."""
        print("🚀 Starting Complete Artifact Versioning Integration")
        print("=" * 60)
        
        all_results = {}
        
        for stage in self.stages:
            stage_results = self.integrate_stage(stage)
            all_results[stage.name] = stage_results
        
        return all_results
    
    def generate_integration_report(self, results: Dict[str, Dict[str, int]]) -> str:
        """Generate a comprehensive integration report."""
        report = []
        report.append("📊 Artifact Versioning Integration Report")
        report.append("=" * 50)
        
        total_integrated = 0
        total_skipped = 0
        total_errors = 0
        
        for stage_name, stage_results in results.items():
            report.append(f"\n📁 {stage_name} Stage:")
            report.append(f"  ✅ Integrated: {stage_results['integrated']}")
            report.append(f"  ⏭️ Skipped: {stage_results['skipped']}")
            report.append(f"  ❌ Errors: {stage_results['errors']}")
            
            total_integrated += stage_results['integrated']
            total_skipped += stage_results['skipped']
            total_errors += stage_results['errors']
        
        report.append(f"\n📈 Overall Summary:")
        report.append(f"  ✅ Total Integrated: {total_integrated}")
        report.append(f"  ⏭️ Total Skipped: {total_skipped}")
        report.append(f"  ❌ Total Errors: {total_errors}")
        
        report.append(f"\n🔧 Next Steps:")
        report.append("1. Review integrated files for any manual adjustments needed")
        report.append("2. Test pipeline execution with new artifact system")
        report.append("3. Update any hardcoded file paths to use pickup utilities")
        report.append("4. Remove backup files after successful testing")
        
        return "\n".join(report)


def main():
    """Main integration function."""
    integrator = ArtifactVersioningIntegrator()
    
    # Run integration
    results = integrator.integrate_all_stages()
    
    # Generate report
    report = integrator.generate_integration_report(results)
    
    # Save report
    report_file = "artifact_versioning_integration_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\n📋 Integration report saved to: {report_file}")
    print("\n" + report)


if __name__ == "__main__":
    main()