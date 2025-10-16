#!/usr/bin/env python3
"""
Analyze and Fill Within-Day Gaps

Uses the existing data collection infrastructure to detect and fill gaps
in the downloaded ETHUSDT 1-minute data.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger
from src.training.steps.data_collection.unified_gap_filler import UnifiedGapFiller
from src.training.steps.data_collection.data_preparation.data_gap_detector import DataGapDetector

logger = system_logger.getChild('GapAnalyzer')

class ETHUSDTGapAnalyzer:
    """Analyzes and fills gaps in ETHUSDT 1-minute data using existing infrastructure."""
    
    def __init__(self, data_dir: str = "historical_data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "binance" / "ethusdt" / "raw"
        
        # Initialize gap detection and filling components
        self.gap_detector = DataGapDetector(str(self.data_dir))
        self.gap_filler = UnifiedGapFiller(str(self.data_dir))
        
        logger.info("✅ ETHUSDT Gap Analyzer initialized")
    
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
    
    async def fill_gaps_automatically(self, analysis: dict):
        """Use the existing gap filler to automatically fill detected gaps."""
        if 'error' in analysis or analysis['total_gaps'] == 0:
            logger.info("No gaps to fill")
            return
        
        logger.info(f"🔧 Attempting to fill {analysis['total_gaps']} gaps...")
        
        try:
            # Use the existing gap filler infrastructure
            # This will use the unified gap filler which can download missing data
            gap_fill_results = await self.gap_filler.fill_gaps_async(
                symbol="ETHUSDT",
                exchange="binance",
                data_type="klines",
                interval="1m"
            )
            
            logger.info(f"✅ Gap filling completed: {gap_fill_results}")
            return gap_fill_results
            
        except Exception as e:
            logger.error(f"Error during gap filling: {e}")
            return {'error': str(e)}

async def main():
    """Main function to analyze and fill gaps."""
    print("🔍 ETHUSDT Within-Day Gap Analysis and Filling")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = ETHUSDTGapAnalyzer()
    
    # Analyze gaps
    print("\n📊 Step 1: Analyzing existing data for gaps...")
    analysis = analyzer.analyze_all_files()
    
    # Print results
    analyzer.print_gap_analysis(analysis)
    
    # Fill gaps if any found
    if 'error' not in analysis and analysis['total_gaps'] > 0:
        print(f"\n🔧 Step 2: Attempting to fill {analysis['total_gaps']} gaps...")
        fill_results = await analyzer.fill_gaps_automatically(analysis)
        
        if 'error' not in fill_results:
            print("✅ Gap filling completed successfully!")
        else:
            print(f"❌ Gap filling failed: {fill_results['error']}")
    else:
        print("\n✅ No gaps found - data is complete!")

if __name__ == "__main__":
    asyncio.run(main())
