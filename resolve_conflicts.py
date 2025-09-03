#!/usr/bin/env python3
"""Resolve merge conflicts by keeping our syntax fixes and main's formatting."""

import re
import os

def resolve_conflict(content, file_path):
    """Resolve conflicts in a file by intelligently merging both versions."""
    
    # For simple import conflicts, prefer the version with proper spacing
    if "config.py" in file_path:
        # Keep the version with proper spacing (origin/main version)
        content = content.replace(
            """<<<<<<< HEAD
from copy import copy
import asyncio
=======

from copy import copy
import asyncio

>>>>>>> origin/main""",
            """
from copy import copy
import asyncio
"""
        )
    
    # For validation files, we need to keep our syntax fixes
    if "validation/step" in file_path:
        # These files had important syntax fixes - need to handle carefully
        conflicts = re.findall(r'<<<<<<< HEAD.*?>>>>>>> origin/main', content, re.DOTALL)
        for conflict in conflicts:
            # If the conflict involves our import fixes, keep our version
            if "from src.training.decorators import" in conflict:
                our_version = conflict.split("=======")[0].replace("<<<<<<< HEAD\n", "")
                content = content.replace(conflict, our_version)
    
    # General approach: remove conflict markers and try to merge intelligently
    lines = content.split('\n')
    result_lines = []
    in_conflict = False
    conflict_start = -1
    our_lines = []
    their_lines = []
    in_theirs = False
    
    for i, line in enumerate(lines):
        if line.strip() == "<<<<<<< HEAD":
            in_conflict = True
            conflict_start = i
            our_lines = []
            their_lines = []
            in_theirs = False
        elif line.strip() == "=======":
            in_theirs = True
        elif line.strip() == ">>>>>>> origin/main":
            # Resolve this conflict
            in_conflict = False
            
            # Choose the better version based on content
            # If both are similar, prefer theirs for formatting
            our_content = '\n'.join(our_lines)
            their_content = '\n'.join(their_lines)
            
            # If one is empty and the other isn't, take the non-empty one
            if not our_content.strip() and their_content.strip():
                result_lines.extend(their_lines)
            elif our_content.strip() and not their_content.strip():
                result_lines.extend(our_lines)
            # If they're essentially the same (just formatting), take theirs
            elif our_content.replace(' ', '').replace('\n', '') == their_content.replace(' ', '').replace('\n', ''):
                result_lines.extend(their_lines)
            # For import statements, check for our syntax fixes
            elif "import" in our_content and "from copy import copy" in our_content:
                # This is one of our syntax fixes, keep it
                result_lines.extend(our_lines)
            else:
                # Default to their version for other cases
                result_lines.extend(their_lines)
            
            our_lines = []
            their_lines = []
            in_theirs = False
        elif in_conflict:
            if in_theirs:
                their_lines.append(line)
            else:
                our_lines.append(line)
        else:
            result_lines.append(line)
    
    return '\n'.join(result_lines)

# Files with conflicts
conflict_files = [
    "src/config.py",
    "src/exchange/binance.py",
    "src/monitoring/fractional_performance_tracker.py",
    "src/monitoring/surrogate_optimization_monitor.py",
    "src/supervisor/main.py",
    "src/tactician/fully_migrated_tactician.py",
    "src/tactician/sr_detection_optimization.py",
    "src/tactician/tactics_orchestrator.py",
    "src/training/core/stage_registry.py",
    "src/training/integration_guide.py",
    "src/training/steps/model_training/step04_5_triple_barrier_method.py",
    "src/training/steps/validation/step17_final_parameters_optimization.py",
    "src/training/steps/validation/step18_walk_forward_validation.py",
    "src/training/steps/validation/step19_monte_carlo_validation.py",
    "src/training/steps/validation_components/base_validation_step.py",
    "src/training/steps/validation_components/confidence_calibration_step.py",
    "src/utils/confidence.py"
]

# Resolve conflicts in each file
for file_path in conflict_files:
    full_path = f"/workspace/{file_path}"
    if os.path.exists(full_path):
        print(f"Resolving conflicts in {file_path}...")
        with open(full_path, 'r') as f:
            content = f.read()
        
        if "<<<<<<< HEAD" in content:
            resolved_content = resolve_conflict(content, file_path)
            with open(full_path, 'w') as f:
                f.write(resolved_content)
            print(f"  ✓ Resolved")
        else:
            print(f"  - No conflicts found")

print("\nDone resolving conflicts!")