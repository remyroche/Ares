#!/usr/bin/env python3
"""
Test the complete code quality pipeline.
This script tests the full pipeline: analyze code -> generate data -> create visualizations
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add code_quality to path
sys.path.insert(0, str(Path(__file__).parent))

# Now we can import our modules
from map_code_interactions import CodeInteractionMapper


def test_pipeline():
    """Test the complete pipeline on a sample project."""
    print("TESTING CODE QUALITY PIPELINE")
    print("=" * 80)
    print()
    
    # Use the code_quality directory itself as test subject
    project_root = Path(__file__).parent
    
    print(f"Testing on: {project_root}")
    print(f"This will analyze the code_quality tools themselves!")
    print()
    
    try:
        # Create the mapper
        mapper = CodeInteractionMapper(str(project_root))
        
        # Run the complete pipeline
        # This will:
        # 1. Check/analyze the code
        # 2. Generate data
        # 3. Create visualizations
        results = mapper.run()
        
        print("\n" + "=" * 80)
        print("PIPELINE TEST COMPLETE!")
        print("=" * 80)
        
        # Verify outputs exist
        if 'report_dir' in results:
            report_dir = Path(results['report_dir'])
            print(f"\nReport directory: {report_dir}")
            
            # List all generated files
            print("\nGenerated files:")
            for file in report_dir.iterdir():
                print(f"  - {file.name}")
            
            # Check visualizations subdirectory if it exists
            viz_dir = report_dir
            if viz_dir.exists():
                viz_files = list(viz_dir.glob("*.png")) + list(viz_dir.glob("*.html"))
                if viz_files:
                    print(f"\nVisualization files ({len(viz_files)}):")
                    for vf in viz_files:
                        print(f"  - {vf.name}")
        
        print("\n✅ Pipeline test successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization_only():
    """Test just the visualization part with sample data."""
    print("\nTESTING VISUALIZATION MODULE")
    print("=" * 80)
    
    try:
        from visualize_interactions import create_sample_visualizations
        
        print("Generating sample visualizations...")
        files = create_sample_visualizations()
        
        print(f"\n✅ Generated {len(files)} visualization files")
        return True
        
    except Exception as e:
        print(f"\n❌ Visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test the full pipeline
    pipeline_ok = test_pipeline()
    
    print("\n" + "-" * 80 + "\n")
    
    # Also test visualization module separately
    viz_ok = test_visualization_only()
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Full Pipeline: {'✅ PASSED' if pipeline_ok else '❌ FAILED'}")
    print(f"Visualization: {'✅ PASSED' if viz_ok else '❌ FAILED'}")
    
    if pipeline_ok and viz_ok:
        print("\n🎉 All tests passed! The pipeline is working correctly.")
        print("\nThe outputs are stored in:")
        print("  - code_quality/visualizers/reports/report_YYYYMMDD_HHMMSS/")
        print("  - code_quality/visualizations/ (for sample visualizations)")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")