#!/usr/bin/env python3
"""
Analysis script to identify files that are called vs not called when launching ares_launcher from step1.

This script will:
1. Trace the execution flow from ares_launcher.py when step1 is specified
2. Identify all Python files that are imported/executed
3. Compare against all Python files in the project
4. Generate a report of unused files
"""

import os
import sys
import importlib
import ast
import subprocess
from pathlib import Path
from typing import Set, List, Dict
import json

class ExecutionAnalyzer:
    def __init__(self, project_root: str = "/workspace"):
        self.project_root = Path(project_root)
        self.called_files: Set[str] = set()
        self.all_python_files: Set[str] = set()
        self.import_graph: Dict[str, List[str]] = {}
        
    def find_all_python_files(self) -> None:
        """Find all Python files in the project."""
        print("🔍 Finding all Python files...")
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip certain directories
            if any(skip_dir in root for skip_dir in ['.git', '__pycache__', '.pytest_cache', 'node_modules']):
                continue
                
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    relative_path = file_path.relative_to(self.project_root)
                    self.all_python_files.add(str(relative_path))
        
        print(f"📊 Found {len(self.all_python_files)} Python files")
    
    def analyze_ares_launcher_flow(self) -> None:
        """Analyze the execution flow starting from ares_launcher.py with step1."""
        print("🚀 Analyzing ares_launcher.py execution flow...")
        
        # Start with ares_launcher.py
        self.called_files.add("ares_launcher.py")
        
        # Analyze imports in ares_launcher.py
        self._analyze_file_imports("ares_launcher.py")
        
        # Follow the step1 execution path
        self._follow_step1_execution_path()
    
    def _analyze_file_imports(self, file_path: str) -> None:
        """Analyze imports in a specific file."""
        try:
            full_path = self.project_root / file_path
            if not full_path.exists():
                return
                
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST to find imports
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self._add_imported_file(alias.name, file_path)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        self._add_imported_file(node.module, file_path)
                        
        except Exception as e:
            print(f"⚠️ Error analyzing {file_path}: {e}")
    
    def _add_imported_file(self, module_name: str, source_file: str) -> None:
        """Add an imported file to the called files set."""
        # Convert module name to file path
        if module_name.startswith('src.'):
            # Handle src imports
            module_path = module_name.replace('.', '/') + '.py'
            if module_path.startswith('src/'):
                self.called_files.add(module_path)
                if source_file not in self.import_graph:
                    self.import_graph[source_file] = []
                self.import_graph[source_file].append(module_path)
        elif module_name.startswith('scripts.'):
            # Handle scripts imports
            module_path = module_name.replace('.', '/') + '.py'
            self.called_files.add(module_path)
            if source_file not in self.import_graph:
                self.import_graph[source_file] = []
            self.import_graph[source_file].append(module_path)
    
    def _follow_step1_execution_path(self) -> None:
        """Follow the specific execution path for step1."""
        print("📋 Following step1 execution path...")
        
        # Based on the analysis of ares_launcher.py, these are the key files called for step1:
        step1_related_files = [
            "src/training/step_orchestrator.py",
            "src/training/enhanced_training_manager.py",
            "src/training/steps/step01_data_collection.py",
            "src/training/steps/step01_5_data_converter.py",
            "src/training/steps/step02_feature_engineering.py",
            "src/training/steps/step03_hmm_regime_discovery.py",
            "src/training/steps/step04_regime_data_splitting.py",
            "src/training/steps/step05_triple_barrier_method.py",
            "src/training/steps/step06_feature_generation.py",
            "src/training/steps/step07_matrix_feature_selection.py",
            "src/training/steps/step08_tactician_labeling.py",
            "src/training/steps/step09_tactician_specialist_training.py",
            "src/training/steps/step10_confidence_calibration.py",
            "src/training/steps/step11_final_parameters_optimization.py",
            "src/training/steps/step12_walk_forward_validation.py",
            "src/training/steps/step13_monte_carlo_validation.py",
            "src/training/steps/step14_ab_testing.py",
            "src/training/steps/step15_saving.py",
            "src/config/__init__.py",
            "src/utils/logger.py",
            "src/utils/error_handler.py",
            "src/utils/comprehensive_logger.py",
            "src/utils/signal_handler.py",
            "src/utils/observability.py",
            "src/database/sqlite_manager.py",
            "src/training/progress_manager.py",
            "src/training/enhanced_training_manager_optimized.py",
            "src/utils/validator_orchestrator.py",
            "src/utils/step_dependency_validator.py",
            "src/utils/training_pipeline_decorators.py",
            "src/utils/model_performance_monitor.py",
            "src/config/computational_optimization.py",
            "src/training/optimization/computational_optimization_manager.py",
            "src/training/steps/multi_timeframe_training/multi_timeframe_training_manager.py",
        ]
        
        # Add these files to called_files
        for file_path in step1_related_files:
            self.called_files.add(file_path)
        
        # Analyze imports in each of these files
        for file_path in step1_related_files:
            self._analyze_file_imports(file_path)
    
    def get_unused_files(self) -> Set[str]:
        """Get files that are not called in the step1 execution."""
        return self.all_python_files - self.called_files
    
    def generate_report(self) -> None:
        """Generate a comprehensive report."""
        print("\n" + "="*80)
        print("📊 EXECUTION ANALYSIS REPORT")
        print("="*80)
        
        print(f"📁 Total Python files found: {len(self.all_python_files)}")
        print(f"🚀 Files called during step1 execution: {len(self.called_files)}")
        print(f"❌ Files NOT called: {len(self.get_unused_files())}")
        
        print("\n" + "="*80)
        print("📋 FILES CALLED DURING STEP1 EXECUTION")
        print("="*80)
        
        called_files_sorted = sorted(self.called_files)
        for file_path in called_files_sorted:
            print(f"✅ {file_path}")
        
        print("\n" + "="*80)
        print("❌ FILES NOT CALLED DURING STEP1 EXECUTION")
        print("="*80)
        
        unused_files = self.get_unused_files()
        unused_files_sorted = sorted(unused_files)
        
        # Categorize unused files
        categories = {
            "validation_files": [],
            "test_files": [],
            "utility_files": [],
            "step_files": [],
            "other_files": []
        }
        
        for file_path in unused_files_sorted:
            if "validator" in file_path.lower():
                categories["validation_files"].append(file_path)
            elif "test" in file_path.lower():
                categories["test_files"].append(file_path)
            elif "utils" in file_path.lower() or "utility" in file_path.lower():
                categories["utility_files"].append(file_path)
            elif "step" in file_path.lower():
                categories["step_files"].append(file_path)
            else:
                categories["other_files"].append(file_path)
        
        for category, files in categories.items():
            if files:
                print(f"\n📂 {category.upper().replace('_', ' ')} ({len(files)} files):")
                for file_path in sorted(files):
                    print(f"   ❌ {file_path}")
        
        # Save detailed report to file
        self._save_detailed_report()
    
    def _save_detailed_report(self) -> None:
        """Save a detailed report to JSON file."""
        report = {
            "summary": {
                "total_files": len(self.all_python_files),
                "called_files": len(self.called_files),
                "unused_files": len(self.get_unused_files())
            },
            "called_files": sorted(self.called_files),
            "unused_files": sorted(self.get_unused_files()),
            "import_graph": self.import_graph
        }
        
        with open("step1_execution_analysis.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: step1_execution_analysis.json")

def main():
    """Main execution function."""
    print("🔍 Starting step1 execution analysis...")
    
    analyzer = ExecutionAnalyzer()
    
    # Step 1: Find all Python files
    analyzer.find_all_python_files()
    
    # Step 2: Analyze ares_launcher execution flow
    analyzer.analyze_ares_launcher_flow()
    
    # Step 3: Generate report
    analyzer.generate_report()
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()