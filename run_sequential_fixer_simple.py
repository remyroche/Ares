#!/usr/bin/env python3
"""
Simple runner for sequential_fixer that bypasses rich library dependency issues
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock the rich library imports to prevent import errors
class MockProgress:
    def __init__(self, *args, **kwargs):
        pass
    
    def add_task(self, *args, **kwargs):
        return 0
    
    def update(self, *args, **kwargs):
        pass
    
    def advance(self, *args, **kwargs):
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass

class MockConsole:
    def print(self, *args, **kwargs):
        print(args[0] if args else "")
    
    def log(self, *args, **kwargs):
        print(args[0] if args else "")

class MockLive:
    def __init__(self, *args, **kwargs):
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def update(self, *args, **kwargs):
        pass

# Create mock modules
mock_rich = type(sys)('rich')
mock_rich.console = type(sys)('console')
mock_rich.console.Console = MockConsole
mock_rich.console.Group = lambda *args: None
mock_rich.layout = type(sys)('layout')
mock_rich.layout.Layout = lambda *args: None
mock_rich.live = type(sys)('live')
mock_rich.live.Live = MockLive
mock_rich.panel = type(sys)('panel')
mock_rich.panel.Panel = lambda *args: None
mock_rich.progress = type(sys)('progress')
mock_rich.progress.Progress = MockProgress
mock_rich.progress.BarColumn = lambda: None
mock_rich.progress.MofNCompleteColumn = lambda: None
mock_rich.progress.SpinnerColumn = lambda: None
mock_rich.progress.TaskProgressColumn = lambda: None
mock_rich.progress.TextColumn = lambda *args: None
mock_rich.progress.TimeElapsedColumn = lambda: None
mock_rich.progress.TimeRemainingColumn = lambda: None
mock_rich.table = type(sys)('table')
mock_rich.table.Table = lambda: None
mock_rich.box = None

sys.modules['rich'] = mock_rich
sys.modules['rich.console'] = mock_rich.console
sys.modules['rich.layout'] = mock_rich.layout
sys.modules['rich.live'] = mock_rich.live
sys.modules['rich.panel'] = mock_rich.panel
sys.modules['rich.progress'] = mock_rich.progress
sys.modules['rich.table'] = mock_rich.table

# Now we can import the sequential fixer
from code_quality.fixers.sequential_fixer import SequentialFixer
from code_quality.core.config import get_default_config

def main():
    # Configure for a simpler run
    config = get_default_config()
    
    # Make the fixer even more conservative
    if config.auto_fix:
        config.auto_fix.enabled = True
        config.auto_fix.tools = ["isort", "autoflake", "pyupgrade"]  # Only the safest tools
        config.auto_fix.aggressive = False
        config.auto_fix.max_line_length = 120
    
    # Run the fixer
    fixer = SequentialFixer(config)
    
    print("Starting Sequential Fixer Analysis...")
    print("=" * 70)
    
    try:
        results = fixer.run_pipeline(
            target=".",
            output_dir="./sequential_fixer_reports",
            create_backups=False,  # No backups to avoid file modifications
            run_pre_commit=False
        )
        
        # Save results
        output_dir = Path("./sequential_fixer_reports")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"sequential_fixer_report_{timestamp}.json"
        
        with open(report_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nReport saved to: {report_file}")
        
        # Print summary
        if "summary" in results:
            summary = results["summary"]
            print("\n" + "=" * 70)
            print("SUMMARY")
            print("=" * 70)
            print(f"Overall Status: {summary.get('overall_status', 'unknown')}")
            
            if "metrics" in summary:
                print("\nMetrics:")
                for key, value in summary["metrics"].items():
                    print(f"  {key}: {value}")
            
            if "recommendations" in summary:
                print("\nRecommendations:")
                for i, rec in enumerate(summary["recommendations"], 1):
                    print(f"  {i}. [{rec.get('priority', '')}] {rec.get('message', '')}")
        
    except Exception as e:
        print(f"\nError running sequential fixer: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())