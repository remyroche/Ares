"""Mass fix all incomplete imports."""
import re
from pathlib import Path

files = [
    'src/feature_generation/categories/autoencoder.py',
    'src/feature_generation/categories/candlestick_pattern.py',
    'src/feature_generation/categories/cross_timeframe.py',
    'src/feature_generation/categories/entropy.py',
    'src/feature_generation/categories/optimized_volatility.py',
    'src/feature_generation/categories/oscillator.py',
    'src/feature_generation/categories/regime_structural_trend.py',
    'src/feature_generation/categories/regime_volatility.py',
    'src/feature_generation/categories/support_resistance.py',
    'src/feature_generation/categories/volatility.py',
    'src/feature_generation/categories/volume.py',
    'src/feature_generation/categories/regime_statistical.py',
]

for file_path in files:
    path = Path(file_path)
    if not path.exists():
        continue
    
    content = path.read_text()
    
    # Pattern: incomplete import followed by try block
    pattern = r'from \.\.core\.feature_generator import \(\s*\n\s*#'
    if re.search(pattern, content):
        # Replace incomplete import
        content = re.sub(
            r'from \.\.core\.feature_generator import \(\s*\n',
            'from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory\n',
            content,
            count=1
        )
        
        # Remove orphaned imports after except block  
        # Multi-line pattern to catch various formats
        content = re.sub(
            r'(except ImportError:\s*\n\s*OPTIMIZATION_AVAILABLE = False)\s*\n+(\s*FeatureGenerator[^)]*\n\))',
            r'\1',
            content,
            flags=re.MULTILINE | re.DOTALL
        )
        
        path.write_text(content)
        print(f"✅ {file_path}")

print("Done!")
