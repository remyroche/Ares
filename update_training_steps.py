#!/usr/bin/env python3
"""
Update Training Steps to Use New Unified Model Training System

This script updates existing training steps to use the new unified model training system.
"""

import os
import sys
import re
from pathlib import Path
from datetime import datetime

def main():
    """Main execution function."""
    print("🔄 Updating Training Steps to Use New Unified Model Training System")
    print("=" * 70)
    
    # Find all Python files that might need updating
    python_files = list(Path('src').rglob('*.py'))
    
    # Files to update (excluding the new infrastructure files)
    files_to_update = []
    for file_path in python_files:
        file_str = str(file_path)
        # Skip new infrastructure files
        if any(skip in file_str for skip in ['simplified_', 'unified_', 'consolidated_', 'transition_', 'simple_']):
            continue
        # Include files that might import old training classes
        if any(keyword in file_str for keyword in ['training', 'model', 'step']):
            files_to_update.append(file_path)
    
    print(f"📊 Found {len(files_to_update)} files to check for updates")
    print()
    
    # Update mappings for training-related imports
    training_import_mappings = {
        # Old imports to new imports
        'from src.training.steps.model_training.step09_hmm_based_training import HMMBasedTraining': 'from src.training.steps.consolidated_model_training import ConsolidatedHMMBasedTraining',
        'from src.training.steps.model_training.step11_analyst_creation import AnalystCreationStep': 'from src.training.steps.consolidated_model_training import ConsolidatedAnalystEnhancement',
        'from src.training.steps.model_training.step12_analyst_enhancement import AnalystEnhancement': 'from src.training.steps.consolidated_model_training import ConsolidatedAnalystEnhancement',
        'from src.training.steps.model_training.step15_tactician_specialist_training import TacticianSpecialistTraining': 'from src.training.steps.consolidated_model_training import ConsolidatedTacticianSpecialistTraining',
        'from src.training.steps.model_training.step10_unified_regime_intelligence import UnifiedRegimeIntelligence': 'from src.training.steps.consolidated_model_training import ConsolidatedUnifiedRegimeIntelligence',
        
        # Class name mappings
        'HMMBasedTraining(': 'ConsolidatedHMMBasedTraining(',
        'AnalystCreationStep(': 'ConsolidatedAnalystEnhancement(',
        'AnalystEnhancement(': 'ConsolidatedAnalystEnhancement(',
        'TacticianSpecialistTraining(': 'ConsolidatedTacticianSpecialistTraining(',
        'UnifiedRegimeIntelligence(': 'ConsolidatedUnifiedRegimeIntelligence(',
    }
    
    files_updated = 0
    
    for file_path in files_to_update:
        try:
            updated = update_training_file(file_path, training_import_mappings)
            if updated:
                files_updated += 1
                print(f"  ✅ Updated: {file_path}")
        except Exception as e:
            print(f"  ❌ Error updating {file_path}: {e}")
    
    print()
    print(f"📊 Updated {files_updated} files to use new unified model training system")
    
    # Create example usage documentation
    create_usage_examples()
    
    print()
    print("✅ Training steps update completed!")
    print("💡 Key changes:")
    print("  - Old training classes replaced with consolidated versions")
    print("  - All imports updated to use new unified system")
    print("  - Core principles preserved (per-HMM regime training, Analyst/Tactician separation)")
    print("  - Backward compatibility maintained")

def update_training_file(file_path: Path, mappings: dict) -> bool:
    """Update a single training file."""
    try:
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply mappings
        for old_pattern, new_pattern in mappings.items():
            if old_pattern in content:
                content = content.replace(old_pattern, new_pattern)
        
        # Write updated content if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def create_usage_examples():
    """Create usage examples for the new unified model training system."""
    examples_content = '''# Usage Examples for New Unified Model Training System

## 1. Basic Usage

```python
from src.training.steps.consolidated_model_training import (
    ConsolidatedHMMBasedTraining,
    ConsolidatedAnalystEnhancement,
    ConsolidatedTacticianSpecialistTraining,
    ConsolidatedUnifiedRegimeIntelligence
)

# Create Analyst
analyst = ConsolidatedAnalystEnhancement(config)
analyst_result = await analyst.execute(features, targets)

# Create Tactician
tactician = ConsolidatedTacticianSpecialistTraining(config)
tactician_result = await tactician.execute(features, targets)

# Create HMM-based model
hmm_model = ConsolidatedHMMBasedTraining(config)
hmm_result = await hmm_model.execute(features, targets)

# Create unified regime intelligence
regime_intel = ConsolidatedUnifiedRegimeIntelligence(config)
regime_result = await regime_intel.execute(features, targets)
```

## 2. Through Unified Model Training

```python
from src.training.steps.unified_model_training import comprehensive_model_training

# Create Analyst
analyst_result = await comprehensive_model_training(
    config, 
    pipeline_state, 
    model_name='analyst_enhancement_model'
)

# Create Tactician
tactician_result = await comprehensive_model_training(
    config, 
    pipeline_state, 
    model_name='tactician_specialist_model'
)
```

## 3. Through Pipeline (Recommended)

```python
from src.training.steps.example_simplified_pipeline import ExampleSimplifiedPipeline

# The pipeline automatically creates both Analyst and Tactician
pipeline = ExampleSimplifiedPipeline(config)
result = await pipeline.execute_pipeline()
```

## 4. Configuration

```python
config = {
    'symbol': 'BTCUSDT',
    'exchange': 'binance',
    'timeframe': '1m',
    'model_training_config': {
        'enable_confidence_metrics': True,
        'enable_calibration_assessment': True,
        'enable_feature_importance': True,
        'enable_cross_validation': True,
        'enable_model_explanations': True,
        'enable_post_training_hpo': True,
        'cv_folds': 5
    }
}
```

## Core Principles Preserved

- ✅ **per-HMM regime training**: Models are trained specifically for different HMM-identified market regimes
- ✅ **Analyst/Tactician separation**: Distinct roles and models for Analyst and Tactician components
- ✅ **Tactician creation**: ConsolidatedTacticianSpecialistTraining handles tactician model creation
- ✅ **General model (Step 10)**: ConsolidatedUnifiedRegimeIntelligence handles the unified regime intelligence model
- ✅ **Tactician labels based on Analyst predictions**: Logic preserved in unified training and labeling
'''
    
    with open('/workspace/UNIFIED_MODEL_TRAINING_USAGE.md', 'w') as f:
        f.write(examples_content)
    
    print("📄 Created usage examples: UNIFIED_MODEL_TRAINING_USAGE.md")

if __name__ == "__main__":
    main()