"""Fix all incomplete imports in feature generation categories."""
import re
from pathlib import Path

# All files that need fixing
files_with_errors = [
    'src/feature_generation/categories/advanced_regime_features.py',
    'src/feature_generation/categories/autoencoder.py',
    'src/feature_generation/categories/candlestick_pattern.py',
    'src/feature_generation/categories/cross_timeframe.py',
    'src/feature_generation/categories/entropy.py',
    'src/feature_generation/categories/optimized_volatility.py',
    'src/feature_generation/categories/oscillator.py',
    'src/feature_generation/categories/regime_structural_trend.py',
    'src/feature_generation/categories/regime_volume.py',
    'src/feature_generation/categories/regime_volatility.py',
    'src/feature_generation/categories/support_resistance.py',
    'src/feature_generation/categories/volatility.py',
    'src/feature_generation/categories/volume.py',
    'src/feature_generation/categories/regime_statistical.py',
]

for file_path in files_with_errors:
    path = Path(file_path)
    if not path.exists():
        print(f"❌ Not found: {file_path}")
        continue
    
    content = path.read_text()
    
    # Pattern 1: Incomplete import with try block
    if 'from ..core.feature_generator import (\n' in content[:500]:
        # Find and replace the incomplete import structure
        new_content = re.sub(
            r'from \.\.core\.feature_generator import \(\s*\n',
            'from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory\n',
            content,
            count=1
        )
        
        # Remove orphaned lines after except block
        new_content = re.sub(
            r'(except ImportError:\s*\n\s*OPTIMIZATION_AVAILABLE = False)\s*\n(\s*FeatureGenerator,?\s*\n\s*FeatureConfig,?\s*\n\s*FeatureCategory,?\s*\n\s*VectorizedFeatureGenerator\s*\n\))',
            r'\1',
            new_content
        )
        
        path.write_text(new_content)
        print(f"✅ Fixed: {file_path}")
    else:
        print(f"⚠️ Already OK or different pattern: {file_path}")

print("\n✅ All files processed!")
