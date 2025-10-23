#!/usr/bin/env python3
"""
Update core VectorBT imports to use the new production module.

This script updates the most critical files that use VectorBT to import
from the new production-ready module instead of direct imports.
"""

import os
import re
from pathlib import Path

def update_file_imports(file_path: str) -> bool:
    """Update VectorBT imports in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Pattern 1: Replace direct vectorbt imports
        content = re.sub(
            r'import vectorbt as vbt',
            'from src.vectorbt import vbt',
            content
        )
        
        # Pattern 2: Replace from vectorbt imports
        content = re.sub(
            r'from vectorbt\.generic import',
            'from src.vectorbt import',
            content
        )
        
        content = re.sub(
            r'from vectorbt\.returns import',
            'from src.vectorbt import',
            content
        )
        
        content = re.sub(
            r'from vectorbt\.portfolio import',
            'from src.vectorbt import',
            content
        )
        
        content = re.sub(
            r'from vectorbt\.indicators\.basic import',
            'from src.vectorbt import',
            content
        )
        
        # Pattern 3: Replace try/except blocks with direct imports
        try_except_pattern = r'try:\s*\n\s*import vectorbt as vbt.*?\nexcept ImportError:.*?\n\s*VECTORBT_AVAILABLE = (True|False)'
        
        def replace_try_except(match):
            return "from src.vectorbt import vbt, rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov, scale, rank, zscore, winsorize, clip, quantile, Portfolio, PortfolioFactory, Returns, RSI, MACD, BBANDS, ATR, STOCH, VECTORBT_AVAILABLE"
        
        content = re.sub(try_except_pattern, replace_try_except, content, flags=re.DOTALL)
        
        # Pattern 4: Remove VECTORBT_AVAILABLE checks (always True in production)
        content = re.sub(r'if VECTORBT_AVAILABLE:', 'if True:  # VectorBT always available')
        content = re.sub(r'if not VECTORBT_AVAILABLE:', 'if False:  # VectorBT always available')
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def main():
    """Update core VectorBT files."""
    
    # Core files that need updating
    core_files = [
        "src/analyst/analyst.py",
        "src/analyst/unified_regime_classifier.py",
        "src/analyst/unified_regime_classifier_sr_optimized.py",
        "src/analyst/unified_regime_classifier_fractal_simplified.py",
        "src/analyst/feature_engineering_utils.py",
        "src/analyst/unified_regime_classifier_fractal_enhanced.py",
        "src/analyst/enhanced_regime_predictor.py",
        "src/analyst/candlestick_pattern_analyzer.py",
        "src/analyst/multi_timeframe_feature_engineering.py",
        "src/analyst/autoencoder_feature_generator.py",
        "src/analyst/location_classifier_optimization.py",
        "src/analyst/meta_labeling_system.py",
        "src/analyst/feature_engineering_orchestrator.py",
        "src/analyst/unified_regime_classifier_sr_focused.py",
        "src/analyst/advanced_feature_engineering.py",
        "src/training/steps/model_training/tactician_ensemble_training.py",
        "src/utils/ml_common/optimization/consolidated_hpo.py",
        "src/utils/vectorbt_batch_processor.py",
        "src/features_common/utils.py",
        "src/feature_generation/utils/vectorbt_rolling_optimizer.py",
    ]
    
    updated_count = 0
    
    for file_path in core_files:
        if os.path.exists(file_path):
            print(f"Updating {file_path}...")
            if update_file_imports(file_path):
                print(f"✅ Updated {file_path}")
                updated_count += 1
            else:
                print(f"⚠️ No changes needed for {file_path}")
        else:
            print(f"❌ File not found: {file_path}")
    
    print(f"\n🎉 Updated {updated_count} files successfully!")

if __name__ == "__main__":
    main()