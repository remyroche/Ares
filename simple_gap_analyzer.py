#!/usr/bin/env python3
"""
Simple Gap Analyzer for ETHUSDT Data

Directly analyzes the downloaded parquet files for within-day gaps
without using the complex existing infrastructure.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger

logger = system_logger.getChild('SimpleGapAnalyzer')

class SimpleGapAnalyzer:
    """Simple gap analyzer for ETHUSDT 1-minute data."""
    
    def __init__(self, data_dir: str = "historical_data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "binance" / "ethusdt" / "raw"
        
        logger.info("✅ Simple Gap Analyzer initialized")
    
    def analyze_file_gaps(self, file_path: Path) -> dict:
        """Analyze gaps within a single parquet file."""
        try:
            # Read the parquet file
            df = pd.read_parquet(file_path)
            
            if 'timestamp' not in df.columns:
                df = df.reset_index()
            
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Calculate expected intervals (1 minute = 60 seconds)
            expected_interval = pd.Timedelta(minutes=1)
            
            # Find gaps by calculating time differences
            time_diffs = df['timestamp'].diff()
            
            # Identify gaps (intervals > 1.1 minutes to account for small delays)
            gap_threshold = pd.Timedelta(minutes=1.1)
            gaps = time_diffs[time_diffs > gap_threshold]
            
            gap_info = {
                'file': file_path.name,
                'total_records': len(df),
                'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
                'expected_records': len(df),
                'gaps_found': len(gaps),
                'gap_details': []
            }
            
            # Analyze each gap
            for idx, gap in gaps.items():
                prev_time = df.loc[idx-1, 'timestamp']
                curr_time = df.loc[idx, 'timestamp']
                gap_duration = curr_time - prev_time
                
                gap_info['gap_details'].append({
                    'gap_index': idx,
                    'start_time': prev_time,
                    'end_time': curr_time,
                    'duration_minutes': gap_duration.total_seconds() / 60,
                    'missing_records': int(gap_duration.total_seconds() / 60) - 1
                })
            
            return gap_info
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return {
                'file': file_path.name,
                'error': str(e),
                'gaps_found': 0
            }
    
    def analyze_all_files(self) -> dict:
        """Analyze gaps in all ETHUSDT raw data files."""
        if not self.raw_dir.exists():
            logger.error(f"Raw directory not found: {self.raw_dir}")
            return {'error': 'Raw directory not found'}
        
        # Find all parquet files
        parquet_files = list(self.raw_dir.glob("ethusdt_1m_*.parquet"))
        
        if not parquet_files:
            logger.warning("No parquet files found in raw directory")
            return {'error': 'No parquet files found'}
        
        logger.info(f"📊 Analyzing {len(parquet_files)} files for gaps...")
        
        all_gaps = []
        total_gaps = 0
        total_missing_records = 0
        
        for file_path in sorted(parquet_files):
            logger.info(f"🔍 Analyzing {file_path.name}...")
            gap_info = self.analyze_file_gaps(file_path)
            
            if 'error' not in gap_info:
                all_gaps.append(gap_info)
                total_gaps += gap_info['gaps_found']
                
                for gap in gap_info['gap_details']:
                    total_missing_records += gap['missing_records']
        
        # Summary
        summary = {
            'files_analyzed': len(parquet_files),
            'files_with_gaps': len([g for g in all_gaps if g['gaps_found'] > 0]),
            'total_gaps': total_gaps,
            'total_missing_records': total_missing_records,
            'file_details': all_gaps
        }
        
        return summary
    
    def print_gap_analysis(self, analysis: dict):
        """Print a formatted gap analysis report."""
        if 'error' in analysis:
            print(f"❌ Error: {analysis['error']}")
            return
        
        print("🔍 ETHUSDT Within-Day Gap Analysis")
        print("=" * 50)
        print(f"📊 Files analyzed: {analysis['files_analyzed']}")
        print(f"📈 Files with gaps: {analysis['files_with_gaps']}")
        print(f"🕳️ Total gaps found: {analysis['total_gaps']}")
        print(f"📉 Total missing records: {analysis['total_missing_records']:,}")
        
        if analysis['total_gaps'] > 0:
            print(f"\n📋 Files with gaps:")
            for file_info in analysis['file_details']:
                if file_info['gaps_found'] > 0:
                    print(f"  📁 {file_info['file']}")
                    print(f"    📅 Range: {file_info['date_range']}")
                    print(f"    🕳️ Gaps: {file_info['gaps_found']}")
                    
                    for gap in file_info['gap_details'][:3]:  # Show first 3 gaps
                        print(f"      ⏰ {gap['start_time']} → {gap['end_time']} ({gap['duration_minutes']:.1f} min, {gap['missing_records']} missing)")
                    
                    if len(file_info['gap_details']) > 3:
                        print(f"      ... and {len(file_info['gap_details']) - 3} more gaps")
        else:
            print("✅ No gaps found in any files!")
    
    def get_gap_summary_by_file(self, analysis: dict) -> dict:
        """Get a summary of gaps by file for detailed analysis."""
        if 'error' in analysis:
            return analysis
        
        summary = {
            'files_with_gaps': [],
            'files_without_gaps': [],
            'largest_gaps': [],
            'total_missing_time_minutes': 0
        }
        
        for file_info in analysis['file_details']:
            if file_info['gaps_found'] > 0:
                summary['files_with_gaps'].append({
                    'file': file_info['file'],
                    'gaps': file_info['gaps_found'],
                    'missing_records': sum(gap['missing_records'] for gap in file_info['gap_details']),
                    'date_range': file_info['date_range']
                })
                
                # Track largest gaps
                for gap in file_info['gap_details']:
                    summary['largest_gaps'].append({
                        'file': file_info['file'],
                        'duration_minutes': gap['duration_minutes'],
                        'missing_records': gap['missing_records'],
                        'start_time': gap['start_time'],
                        'end_time': gap['end_time']
                    })
                    summary['total_missing_time_minutes'] += gap['duration_minutes']
            else:
                summary['files_without_gaps'].append(file_info['file'])
        
        # Sort largest gaps by duration
        summary['largest_gaps'].sort(key=lambda x: x['duration_minutes'], reverse=True)
        
        return summary

def main():
    """Main function to analyze gaps."""
    print("🔍 ETHUSDT Within-Day Gap Analysis")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = SimpleGapAnalyzer()
    
    # Analyze gaps
    print("\n📊 Step 1: Analyzing existing data for gaps...")
    analysis = analyzer.analyze_all_files()
    
    # Print results
    analyzer.print_gap_analysis(analysis)
    
    # Get detailed summary
    if 'error' not in analysis:
        print(f"\n📋 Detailed Summary:")
        summary = analyzer.get_gap_summary_by_file(analysis)
        
        print(f"  📁 Files with gaps: {len(summary['files_with_gaps'])}")
        print(f"  📁 Files without gaps: {len(summary['files_without_gaps'])}")
        print(f"  ⏱️ Total missing time: {summary['total_missing_time_minutes']:.1f} minutes")
        
        if summary['largest_gaps']:
            print(f"\n🔍 Top 5 largest gaps:")
            for i, gap in enumerate(summary['largest_gaps'][:5], 1):
                print(f"  {i}. {gap['file']}: {gap['duration_minutes']:.1f} min ({gap['missing_records']} records)")
                print(f"     {gap['start_time']} → {gap['end_time']}")

if __name__ == "__main__":
    main()
