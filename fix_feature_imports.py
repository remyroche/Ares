"""Fix incomplete imports in feature generation categories."""
import re
from pathlib import Path

files_to_fix = [
    'src/feature_generation/categories/regime_volume.py',
    'src/feature_generation/categories/interaction.py',
    'src/feature_generation/categories/trend.py',
    'src/feature_generation/categories/acceleration.py',
    'src/feature_generation/categories/regime_statistical.py',
]

# Pattern to match the broken import structure
pattern = re.compile(
    r'from \.\.core\.feature_generator import \(\s*\n\s*#.*\ntry:',
    re.MULTILINE
)

replacement = '''from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

# Optimization utilities
try:'''

for file_path in files_to_fix:
    path = Path(file_path)
    if not path.exists():
        print(f"❌ File not found: {file_path}")
        continue
    
    content = path.read_text()
    
    # Check if file has the problematic pattern
    if 'from ..core.feature_generator import (' in content and '\ntry:\n' in content[:500]:
        # Fix the import
        new_content = re.sub(
            r'from \.\.core\.feature_generator import \(\s*\n+\s*# Optimization',
            'from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory\n\n# Optimization',
            content
        )
        
        # Remove orphaned closing parenthesis after except block
        new_content = re.sub(
            r'(except ImportError:\s*\n\s*OPTIMIZATION_AVAILABLE = False)\s*\n\s*FeatureGenerator,\s*\n\s*FeatureConfig,\s*\n\s*FeatureCategory,\s*\n\s*VectorizedFeatureGenerator\s*\n\)',
            r'\1',
            new_content
        )
        
        path.write_text(new_content)
        print(f"✅ Fixed: {file_path}")
    else:
        print(f"⚠️ Pattern not found in: {file_path}")

print("\n✅ All files processed!")
