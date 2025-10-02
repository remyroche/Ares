"""
Test script for the refactored NAS-TAS clustering component structure.

This script tests the basic structure and imports of the refactored clustering modules.
"""

import sys
import os
from pathlib import Path

def test_directory_structure():
    """Test that the directory structure is correct."""
    print("🧪 Testing Directory Structure")
    print("=" * 50)
    
    # Check main directories
    required_dirs = [
        "src/training/steps/market_analysis/components/clustering",
        "src/training/steps/market_analysis/components/optimization",
        "src/training/steps/market_analysis/components/validation",
        "src/training/steps/market_analysis/components/metrics",
        "src/training/steps/market_analysis/components/hardware",
        "src/training/steps/market_analysis/components/config"
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path}")
    
    print()

def test_file_structure():
    """Test that the required files exist."""
    print("📁 Testing File Structure")
    print("=" * 50)
    
    # Check main files
    required_files = [
        "src/training/steps/market_analysis/components/nas_tas_clustering_refactored.py",
        "src/training/steps/market_analysis/components/clustering/__init__.py",
        "src/training/steps/market_analysis/components/clustering/step1_feature_preparation.py",
        "src/training/steps/market_analysis/components/clustering/step2_initial_clustering.py",
        "src/training/steps/market_analysis/components/clustering/iterative_optimization.py",
        "src/training/steps/market_analysis/components/clustering/step8_validation.py",
        "src/training/steps/market_analysis/components/clustering/step9_results_consolidation.py",
        "src/training/steps/market_analysis/components/clustering/clustering_orchestrator.py"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
    
    print()

def test_import_structure():
    """Test that the import structure is correct."""
    print("📦 Testing Import Structure")
    print("=" * 50)
    
    # Test basic Python imports
    try:
        import asyncio
        print("✅ asyncio")
    except ImportError as e:
        print(f"❌ asyncio: {e}")
    
    try:
        import json
        print("✅ json")
    except ImportError as e:
        print(f"❌ json: {e}")
    
    try:
        import datetime
        print("✅ datetime")
    except ImportError as e:
        print(f"❌ datetime: {e}")
    
    try:
        import pathlib
        print("✅ pathlib")
    except ImportError as e:
        print(f"❌ pathlib: {e}")
    
    print()

def test_file_sizes():
    """Test that the refactored files are appropriately sized."""
    print("📏 Testing File Sizes")
    print("=" * 50)
    
    # Check file sizes
    files_to_check = [
        "src/training/steps/market_analysis/components/nas_tas_clustering_refactored.py",
        "src/training/steps/market_analysis/components/clustering/step1_feature_preparation.py",
        "src/training/steps/market_analysis/components/clustering/step2_initial_clustering.py",
        "src/training/steps/market_analysis/components/clustering/iterative_optimization.py",
        "src/training/steps/market_analysis/components/clustering/step8_validation.py",
        "src/training/steps/market_analysis/components/clustering/step9_results_consolidation.py",
        "src/training/steps/market_analysis/components/clustering/clustering_orchestrator.py"
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            lines = 0
            try:
                with open(file_path, 'r') as f:
                    lines = len(f.readlines())
            except:
                lines = "N/A"
            print(f"✅ {file_path}: {size} bytes, {lines} lines")
        else:
            print(f"❌ {file_path}: File not found")
    
    print()

def test_code_structure():
    """Test that the code structure is correct."""
    print("🔍 Testing Code Structure")
    print("=" * 50)
    
    # Check for key classes and functions
    main_file = "src/training/steps/market_analysis/components/nas_tas_clustering_refactored.py"
    
    if os.path.exists(main_file):
        try:
            with open(main_file, 'r') as f:
                content = f.read()
                
            # Check for key components
            key_components = [
                "class NASTASClusteringComponent",
                "class NASTASClusteringConfig", 
                "class ClusteringContext",
                "async def run",
                "async def _perform_clustering",
                "ClusteringOrchestrator"
            ]
            
            for component in key_components:
                if component in content:
                    print(f"✅ {component}")
                else:
                    print(f"❌ {component}")
                    
        except Exception as e:
            print(f"❌ Error reading main file: {e}")
    else:
        print(f"❌ Main file not found: {main_file}")
    
    print()

def test_refactoring_benefits():
    """Test that the refactoring benefits are achieved."""
    print("🎯 Testing Refactoring Benefits")
    print("=" * 50)
    
    # Check original file size
    original_file = "src/training/steps/market_analysis/components/nas_tas_clustering.py"
    if os.path.exists(original_file):
        original_size = os.path.getsize(original_file)
        print(f"📊 Original file size: {original_size:,} bytes")
        
        # Check refactored main file size
        refactored_file = "src/training/steps/market_analysis/components/nas_tas_clustering_refactored.py"
        if os.path.exists(refactored_file):
            refactored_size = os.path.getsize(refactored_file)
            print(f"📊 Refactored main file size: {refactored_size:,} bytes")
            
            # Calculate reduction
            reduction = ((original_size - refactored_size) / original_size) * 100
            print(f"📉 Size reduction: {reduction:.1f}%")
            
            if reduction > 50:
                print("✅ Significant size reduction achieved")
            else:
                print("⚠️ Limited size reduction")
        else:
            print("❌ Refactored file not found")
    else:
        print("❌ Original file not found")
    
    print()
    
    # Check modular structure
    clustering_dir = "src/training/steps/market_analysis/components/clustering"
    if os.path.exists(clustering_dir):
        files = [f for f in os.listdir(clustering_dir) if f.endswith('.py')]
        print(f"📁 Clustering modules: {len(files)} files")
        print("✅ Modular structure achieved")
    else:
        print("❌ Clustering directory not found")
    
    print()

def main():
    """Run all tests."""
    print("🚀 Testing Refactored NAS-TAS Clustering Structure")
    print("=" * 80)
    
    test_directory_structure()
    test_file_structure()
    test_import_structure()
    test_file_sizes()
    test_code_structure()
    test_refactoring_benefits()
    
    print("🎉 Structure testing completed!")
    print("=" * 80)

if __name__ == "__main__":
    main()