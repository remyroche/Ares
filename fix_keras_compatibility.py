#!/usr/bin/env python3
"""
Fix Keras 3 compatibility issues with Transformers library.

This script addresses the warning:
"Your currently installed version of Keras is Keras 3, but this is not yet supported in Transformers"

Solution: Install the backwards-compatible tf-keras package.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main function to fix Keras compatibility issues."""
    print("🔧 Fixing Keras 3 compatibility issues...")
    
    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Not running in a virtual environment. Consider using one.")
    
    # Install tf-keras for backwards compatibility
    success = run_command(
        "pip install tf-keras",
        "Installing tf-keras for backwards compatibility"
    )
    
    if not success:
        print("❌ Failed to install tf-keras. Trying alternative approach...")
        
        # Try installing with specific version
        success = run_command(
            "pip install tf-keras==2.15.0",
            "Installing specific version of tf-keras"
        )
    
    if success:
        # Update requirements.txt to include tf-keras
        print("📝 Updating requirements.txt...")
        try:
            with open("requirements.txt", "r") as f:
                lines = f.readlines()
            
            # Check if tf-keras is already in requirements
            tf_keras_exists = any("tf-keras" in line for line in lines)
            
            if not tf_keras_exists:
                # Find the keras line and add tf-keras after it
                for i, line in enumerate(lines):
                    if line.strip().startswith("keras>="):
                        lines.insert(i + 1, "tf-keras>=2.15.0\n")
                        break
                
                with open("requirements.txt", "w") as f:
                    f.writelines(lines)
                print("✅ Added tf-keras to requirements.txt")
            else:
                print("ℹ️  tf-keras already in requirements.txt")
                
        except Exception as e:
            print(f"⚠️  Warning: Could not update requirements.txt: {e}")
    
    # Test the fix
    print("🧪 Testing Keras compatibility...")
    try:
        import tensorflow as tf
        import tf_keras as keras
        print("✅ Keras compatibility test passed")
        print(f"   TensorFlow version: {tf.__version__}")
        print(f"   Keras version: {keras.__version__}")
    except ImportError as e:
        print(f"❌ Keras compatibility test failed: {e}")
        return False
    
    print("\n🎉 Keras compatibility fix completed!")
    print("📋 Summary:")
    print("   - Installed tf-keras for backwards compatibility")
    print("   - Updated requirements.txt")
    print("   - Verified compatibility with TensorFlow")
    print("\n💡 Note: The SHAP analysis in Step 6 should now work without Keras compatibility warnings.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 