#!/usr/bin/env python3
"""
Export Poetry dependencies to requirements.txt for ChatGPT Codex compatibility.
This script reads from poetry.lock and creates a requirements.txt file.
"""

import subprocess
import sys
import os

def export_poetry_requirements():
    """Export Poetry dependencies to requirements.txt"""
    try:
        # Check if Poetry is available
        result = subprocess.run(['poetry', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Poetry found: {result.stdout.strip()}")
        
        # Export requirements
        print("📦 Exporting Poetry dependencies to requirements.txt...")
        subprocess.run(['poetry', 'export', '--format=requirements.txt', 
                       '--output=requirements_poetry.txt', '--without-hashes'], 
                      check=True)
        
        print("✅ Requirements exported to requirements_poetry.txt")
        
        # Also create a simple requirements.txt for Codex
        print("📦 Creating Codex-compatible requirements.txt...")
        subprocess.run(['poetry', 'export', '--format=requirements.txt', 
                       '--output=requirements.txt', '--without-hashes', '--without-urls'], 
                      check=True)
        
        print("✅ Codex-compatible requirements.txt created")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Poetry command: {e}")
        return False
    except FileNotFoundError:
        print("❌ Poetry not found. Please install Poetry first.")
        return False

def main():
    """Main function"""
    print("🚀 Exporting Poetry dependencies for ChatGPT Codex...")
    
    if export_poetry_requirements():
        print("🎉 Successfully exported dependencies!")
        
        # Show the first few lines of the requirements file
        if os.path.exists('requirements.txt'):
            print("\n📋 First 10 lines of requirements.txt:")
            with open('requirements.txt', 'r') as f:
                lines = f.readlines()[:10]
                for line in lines:
                    print(f"  {line.strip()}")
            print("  ...")
    else:
        print("❌ Failed to export dependencies")
        sys.exit(1)

if __name__ == "__main__":
    main()