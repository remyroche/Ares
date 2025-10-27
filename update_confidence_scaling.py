#!/usr/bin/env python3
"""
Script to update confidence scaling in the volatility aware labeler.
"""

import re

def update_confidence_scaling():
    """Update the confidence scaling logic in the volatility aware labeler."""
    
    # Read the file
    with open('src/training/steps/pre_training/profit_labeling/volatility_aware_labeler.py', 'r') as f:
        content = f.read()
    
    # Define the old and new patterns
    old_pattern = r'# - Reduced confidence when close to targets \(within 20% of threshold\)\n        # - Increased confidence when hitting targets and beyond \(120%\+ threshold\)\n        proximity_factor = np\.where\(\n            np\.abs\(distance\) < 1\.2,  # Close to target\n            np\.abs\(distance\) / 1\.2,  # Linear scaling from 0 to 1\n            np\.where\(\n                np\.abs\(distance\) >= 1\.2,  # Hit target and beyond\n                1\.0 \+ \(np\.abs\(distance\) - 1\.2\) \* 0\.5,  # Enhanced confidence: 1\.0 \+ 0\.5 \* excess\n                1\.0  # Fallback\n            \)\n        \)'
    
    new_pattern = '''# - 0.5 confidence for 75% of target
        # - 1.0 confidence for 100% of target
        # - 1.5 confidence for 200% of target
        # Linear scaling: confidence = 0.5 + 0.4 * (distance - 0.75)
        proximity_factor = np.where(
            np.abs(distance) < 0.75,  # Below 75% of target
            0.5 * np.abs(distance) / 0.75,  # Linear scaling from 0 to 0.5
            np.where(
                np.abs(distance) >= 0.75,  # 75% and above
                0.5 + 0.4 * (np.abs(distance) - 0.75),  # Linear scaling: 0.5 + 0.4 * (distance - 0.75)
                0.5  # Fallback
            )
        )'''
    
    # Replace all occurrences
    updated_content = re.sub(old_pattern, new_pattern, content, flags=re.MULTILINE)
    
    # Write back to file
    with open('src/training/steps/pre_training/profit_labeling/volatility_aware_labeler.py', 'w') as f:
        f.write(updated_content)
    
    print("✅ Updated confidence scaling in volatility aware labeler")

if __name__ == "__main__":
    update_confidence_scaling()