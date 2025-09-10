#!/usr/bin/env python3
"""
Documentation Update Script for Simplified Infrastructure

This script updates all documentation to reflect the new simplified infrastructure.
"""

import os
import re
from pathlib import Path
from typing import List, Dict

def update_documentation():
    """Update all documentation files."""
    
    # Files to update
    doc_files = [
        'README.md',
        'docs/README.md',
        'docs/architecture.md',
        'docs/api_reference.md'
    ]
    
    # Update mappings
    updates = {
        'BaseStep': 'SimplifiedStepBase',
        'step1_data_collection': 'simplified_step1_data_collection',
        'step05_labeling': 'simplified_step5_labeling',
        'AdvancedFeatureEngineeringStep': 'unified_feature_engineering',
        'HMMBasedTraining': 'unified_model_training',
        'M1MemoryOptimizer': 'unified_optimization'
    }
    
    for doc_file in doc_files:
        if Path(doc_file).exists():
            update_file(doc_file, updates)

def update_file(file_path: str, updates: Dict[str, str]):
    """Update a single documentation file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    for old, new in updates.items():
        content = content.replace(old, new)
    
    with open(file_path, 'w') as f:
        f.write(content)

if __name__ == "__main__":
    update_documentation()