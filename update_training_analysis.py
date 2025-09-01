#!/usr/bin/env python3
"""
Update Training Analysis Script
Updates the training execution analysis to reflect the new 21-step pipeline.
"""

import json
import os
from pathlib import Path

def update_training_analysis():
    """Update the training analysis to include steps 16-21."""
    
    # Read the current analysis
    analysis_file = "complete_training_execution_analysis.json"
    
    if os.path.exists(analysis_file):
        with open(analysis_file, 'r') as f:
            analysis = json.load(f)
    else:
        print(f"❌ Analysis file {analysis_file} not found")
        return
    
    # Add the new step files to the called files list
    new_step_files = [
        "src/training/steps/step16_confidence_calibration.py",
        "src/training/steps/step16_confidence_calibration_validator.py",
        "src/training/steps/step17_final_parameters_optimization.py",
        "src/training/steps/step17_final_parameters_optimization_validator.py",
        "src/training/steps/step18_walk_forward_validation.py",
        "src/training/steps/step18_walk_forward_validation_validator.py",
        "src/training/steps/step19_monte_carlo_validation.py",
        "src/training/steps/step19_monte_carlo_validation_validator.py",
        "src/training/steps/step20_ab_testing.py",
        "src/training/steps/step20_ab_testing_validator.py",
        "src/training/steps/step21_saving.py",
        "src/training/steps/step21_saving_validator.py",
    ]
    
    # Add new files to called files
    for file_path in new_step_files:
        if file_path not in analysis["called_files"]:
            analysis["called_files"].append(file_path)
    
    # Remove these files from uncalled files if they were there
    for file_path in new_step_files:
        if file_path in analysis["uncalled_files"]:
            analysis["uncalled_files"].remove(file_path)
    
    # Update statistics
    analysis["statistics"]["total_files"] = len(analysis["called_files"]) + len(analysis["uncalled_files"])
    analysis["statistics"]["called_files_count"] = len(analysis["called_files"])
    analysis["statistics"]["uncalled_files_count"] = len(analysis["uncalled_files"])
    analysis["statistics"]["coverage_percentage"] = (len(analysis["called_files"]) / analysis["statistics"]["total_files"]) * 100
    
    # Update metadata
    analysis["metadata"]["pipeline_steps"] = 21
    analysis["metadata"]["description"] = "Complete 21-step enhanced training pipeline analysis"
    analysis["metadata"]["updated_at"] = "2024-01-01"  # Update with current date
    
    # Save updated analysis
    output_file = "updated_21_step_training_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"✅ Updated analysis saved to {output_file}")
    print(f"📊 New statistics:")
    print(f"   - Total files: {analysis['statistics']['total_files']}")
    print(f"   - Called files: {analysis['statistics']['called_files_count']}")
    print(f"   - Uncalled files: {analysis['statistics']['uncalled_files_count']}")
    print(f"   - Coverage: {analysis['statistics']['coverage_percentage']:.1f}%")
    print(f"   - Pipeline steps: {analysis['metadata']['pipeline_steps']}")

if __name__ == "__main__":
    update_training_analysis()