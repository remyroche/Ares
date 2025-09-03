#!/usr/bin/env python3
"""Quick-start script for migration tasks.

This script helps developers quickly start working on migration tasks
by setting up the environment and providing helper functions.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import shutil
import subprocess


class MigrationHelper:
    """Helper class for migration tasks."""
    
    def __init__(self):
        self.root_dir = Path(__file__).parent.parent.parent
        self.steps_dir = self.root_dir / "src" / "training" / "steps"
        self.templates_dir = self.steps_dir / "migration_templates"
        
    def list_pending_steps(self) -> List[Dict[str, str]]:
        """List all steps pending migration."""
        pending = []
        
        step_info = {
            "07": ("enhanced_matrix_operations", "model_training"),
            "08": ("feature_selection", "model_training"),
            "09": ("hmm_based_training", "model_training"),
            "09_5": ("multi_timeframe_hmm_ensemble", "model_training"),
            "10": ("unified_regime_intelligence", "model_training"),
            "11": ("analyst_creation", "model_training"),
            "12": ("analyst_enhancement", "model_training"),
            "13": ("analyst_ensemble_creation", "model_training"),
            "14": ("tactician_labeling", "model_training"),
            "15": ("tactician_specialist_training", "model_training"),
            "21": ("saving", "model_training"),
        }
        
        for step_num, (step_name, category) in step_info.items():
            old_path = self.steps_dir / f"step{step_num}_{step_name}.py"
            new_path = self.steps_dir / category / f"step{step_num}_{step_name}.py"
            
            if old_path.exists() and not new_path.exists():
                file_size = old_path.stat().st_size / 1024  # KB
                line_count = sum(1 for _ in open(old_path))
                
                pending.append({
                    "step": step_num,
                    "name": step_name,
                    "category": category,
                    "old_path": str(old_path),
                    "new_path": str(new_path),
                    "size_kb": file_size,
                    "lines": line_count
                })
                
        return sorted(pending, key=lambda x: x["step"])
    
    def start_migration(self, step_num: str) -> bool:
        """Start migration for a specific step."""
        # Find step info
        pending = self.list_pending_steps()
        step_info = next((s for s in pending if s["step"] == step_num), None)
        
        if not step_info:
            print(f"❌ Step {step_num} not found in pending migrations")
            return False
        
        print(f"\n🚀 Starting migration for Step {step_num}: {step_info['name']}")
        print(f"   Size: {step_info['size_kb']:.1f} KB ({step_info['lines']} lines)")
        
        # Create category directory if needed
        category_dir = self.steps_dir / step_info["category"]
        category_dir.mkdir(exist_ok=True)
        
        # Check for template
        template_path = self.templates_dir / f"step{step_num}_{step_info['name']}_template.py"
        
        if template_path.exists():
            print(f"   ✅ Found template: {template_path}")
            
            # Copy template to new location
            new_path = Path(step_info["new_path"])
            shutil.copy(template_path, new_path)
            print(f"   📝 Created new file: {new_path}")
            
            # Open both files for reference
            self._open_files_for_editing(Path(step_info["old_path"]), new_path)
            
            return True
        else:
            print(f"   ⚠️  No template found, creating basic structure...")
            self._create_basic_migration(step_info)
            return True
    
    def _create_basic_migration(self, step_info: Dict[str, str]) -> None:
        """Create a basic migration file."""
        from migration_script import create_step_template
        
        template = create_step_template(
            step_info["step"],
            step_info["category"],
            step_info["name"]
        )
        
        new_path = Path(step_info["new_path"])
        with open(new_path, 'w') as f:
            f.write(template)
        
        print(f"   📝 Created basic template: {new_path}")
        self._open_files_for_editing(Path(step_info["old_path"]), new_path)
    
    def _open_files_for_editing(self, old_path: Path, new_path: Path) -> None:
        """Open files in the default editor."""
        # Try to open in VS Code if available
        try:
            subprocess.run(["code", str(old_path), str(new_path)], check=True)
            print(f"   📂 Opened files in VS Code")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"   📂 Ready for editing:")
            print(f"      Old: {old_path}")
            print(f"      New: {new_path}")
    
    def analyze_file_structure(self, file_path: Path) -> Dict[str, any]:
        """Analyze the structure of a Python file to help with refactoring."""
        import ast
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return {"error": "Failed to parse file"}
        
        # Analyze the AST
        classes = []
        functions = []
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                classes.append({
                    "name": node.name,
                    "line": node.lineno,
                    "methods": methods,
                    "method_count": len(methods)
                })
            elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                functions.append({
                    "name": node.name,
                    "line": node.lineno
                })
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(node.lineno)
        
        return {
            "classes": classes,
            "functions": functions,
            "import_lines": len(imports),
            "total_lines": len(content.splitlines())
        }
    
    def create_component_file(self, step_num: str, component_name: str) -> Path:
        """Create a component file for a step."""
        # Find step info
        pending = self.list_pending_steps()
        step_info = next((s for s in pending if s["step"] == step_num), None)
        
        if not step_info:
            raise ValueError(f"Step {step_num} not found")
        
        # Create component file
        category_dir = self.steps_dir / step_info["category"]
        component_path = category_dir / f"{component_name}.py"
        
        if component_path.exists():
            print(f"⚠️  Component file already exists: {component_path}")
            return component_path
        
        # Create basic component template
        template = f'''"""Components for Step {step_num}: {step_info['name']}.

This module contains extracted components from the original step implementation.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from src.utils.logger import system_logger


class {component_name.title().replace("_", "")}:
    """Component extracted from Step {step_num}."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize component.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("{component_name}")
        
    # TODO: Add methods extracted from original step
'''
        
        with open(component_path, 'w') as f:
            f.write(template)
        
        print(f"✅ Created component file: {component_path}")
        return component_path
    
    def run_tests(self, step_num: Optional[str] = None) -> bool:
        """Run tests for a specific step or all migrated steps."""
        test_cmd = ["pytest", "-v"]
        
        if step_num:
            test_cmd.append(f"src/training/tests/test_step{step_num}_*.py")
        else:
            test_cmd.append("src/training/tests/")
        
        try:
            result = subprocess.run(test_cmd, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
            return result.returncode == 0
        except FileNotFoundError:
            print("❌ pytest not found. Please install: pip install pytest")
            return False


def main():
    """Main entry point for migration quick-start."""
    helper = MigrationHelper()
    
    print("🚀 Training System Migration Quick-Start")
    print("=" * 50)
    
    # List pending migrations
    pending = helper.list_pending_steps()
    
    print("\n📋 Pending Migrations:")
    print(f"{'Step':<6} {'Name':<30} {'Category':<20} {'Size':<10} {'Lines':<10}")
    print("-" * 80)
    
    for step in pending:
        size_str = f"{step['size_kb']:.1f} KB"
        print(f"{step['step']:<6} {step['name']:<30} {step['category']:<20} {size_str:<10} {step['lines']:<10}")
    
    # Get user choice
    print("\n" + "=" * 50)
    print("Commands:")
    print("  migrate <step_num>  - Start migrating a specific step")
    print("  analyze <step_num>  - Analyze file structure")
    print("  component <step_num> <name> - Create component file")
    print("  test [step_num]     - Run tests")
    print("  exit                - Exit")
    
    while True:
        try:
            command = input("\n> ").strip().split()
            
            if not command:
                continue
                
            if command[0] == "exit":
                break
                
            elif command[0] == "migrate" and len(command) > 1:
                helper.start_migration(command[1])
                
            elif command[0] == "analyze" and len(command) > 1:
                step_num = command[1]
                step = next((s for s in pending if s["step"] == step_num), None)
                if step:
                    analysis = helper.analyze_file_structure(Path(step["old_path"]))
                    print(f"\n📊 Analysis of Step {step_num}:")
                    print(f"   Classes: {len(analysis.get('classes', []))}")
                    for cls in analysis.get('classes', []):
                        print(f"     - {cls['name']} ({cls['method_count']} methods)")
                    print(f"   Functions: {len(analysis.get('functions', []))}")
                    print(f"   Total lines: {analysis.get('total_lines', 0)}")
                else:
                    print(f"❌ Step {step_num} not found")
                    
            elif command[0] == "component" and len(command) > 2:
                helper.create_component_file(command[1], command[2])
                
            elif command[0] == "test":
                step_num = command[1] if len(command) > 1 else None
                helper.run_tests(step_num)
                
            else:
                print("❌ Invalid command")
                
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()