import os
from pathlib import Path

specialist_files = [
    "src/training/steps/market_analysis/ml_liquidity_regime_step_enhanced.py",
    "src/training/steps/market_analysis/ml_path_regime_step_enhanced.py",
    "src/training/steps/market_analysis/ml_reversion_regime_step_enhanced.py",
    "src/training/steps/market_analysis/ml_risk_regime_step_enhanced.py",
    "src/training/steps/market_analysis/ml_smc_regime_step_enhanced.py",
    "src/training/steps/market_analysis/ml_volume_force_step_enhanced.py",
    "src/training/steps/market_analysis/xgb_macro_regime_step_enhanced.py",
    "src/training/steps/market_analysis/xgb_meso_regime_step_enhanced.py"
]

def fix_imports(file_path):
    print(f"Fixing imports in {file_path}...")
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Ensure tprint is imported
    if "from src.utils.tprint import (" in content:
        if "tprint," not in content and "tprint\n" not in content and "tprint)" not in content:
            content = content.replace("from src.utils.tprint import (", "from src.utils.tprint import (\n    tprint,")
    
    # Fix MIOptimizedFeaturePipeline in VolumeForce
    if "ml_volume_force_step_enhanced.py" in str(file_path):
        if "from src.training.steps.market_analysis.enhanced_feature_generators import MIOptimizedFeaturePipeline" not in content:
            content = content.replace("from src.training.steps.market_analysis.specialist_interface import SpecialistDataInterface", 
                                     "from src.training.steps.market_analysis.enhanced_feature_generators import MIOptimizedFeaturePipeline\nfrom src.training.steps.market_analysis.specialist_interface import SpecialistDataInterface")

    # Fix the call sites for _create_standardized_output
    # Specifically for Liquidity, Path, Reversion which were reported missing arguments
    if any(x in str(file_path) for x in ["ml_liquidity", "ml_path", "ml_reversion"]):
        # It seems the call site might be broken or the signature doesn't match what's expected.
        # The error said missing 7 required positional arguments: 'labels', 'predictions', 'probabilities', 'symbol', 'exchange', 'timeframe', and 'direction'
        # Let's check the code for these files.
        pass

    with open(file_path, 'w') as f:
        f.write(content)

for f_rel in specialist_files:
    f_path = Path(f_rel)
    if f_path.exists():
        fix_imports(f_path)

