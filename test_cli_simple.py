#!/usr/bin/env python3
"""
Simple test for CLI with existing data using subprocess
"""

import sys
import subprocess
from pathlib import Path

def test_cli_simple():
    """Test CLI with existing data using subprocess."""
    try:
        # Find existing data file
        data_file = None
        for file_path in Path("artifacts").rglob("*.parquet"):
            if "BTCUSDT" in str(file_path) or "ETHUSDT" in str(file_path):
                data_file = file_path
                break
        
        if not data_file:
            print("❌ No existing data file found")
            return False
        
        print(f"✅ Using data file: {data_file}")
        
        # Create output directory
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        
        # Run CLI command
        cmd = [
            "python3", "-m", "src.training.steps.market_analysis.statsmodel_clustering.cli",
            "cluster",
            "--symbol", "BTCUSDT",
            "--data-file", str(data_file),
            "--regimes", "3",
            "--output-dir", str(output_dir)
        ]
        
        print(f"🔬 Running command: {' '.join(cmd)}")
        
        # Run command without timeout to allow batch processing
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path.cwd()
        )
        
        # Print output
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        # Check result
        if result.returncode == 0:
            print("✅ CLI test completed successfully")
            return True
        else:
            print(f"❌ CLI test failed with exit code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ CLI test timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ CLI test failed with exception: {e}")
        return False

def main():
    """Run simple CLI test."""
    print("🧪 Running simple CLI test...\n")
    
    result = test_cli_simple()
    
    if result:
        print("\n🎉 Simple CLI test passed!")
        return 0
    else:
        print("\n❌ Simple CLI test failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)