"""
SR Data Collector - 100% Data-Driven

Walk-forward data collection from historical data with artifact manager integration.
Combines candidate generation, feature extraction, and target labeling.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from datetime import datetime

from src.training.steps.pre_training.utils.artifact_manager import artifact_context
from src.training.steps.sr_detection_ml.candidate_level_generator import DataDrivenLevelGenerator
from src.training.steps.sr_detection_ml.raw_feature_generator import RawFeatureGenerator
from src.training.steps.sr_detection_ml.outcome_target_generator import OutcomeTargetGenerator

logger = logging.getLogger(__name__)


class SRDataCollector:
    """
    Collect SR level training data using walk-forward methodology.
    
    Philosophy: Generate data from pure price behavior, no heuristics.
    """
    
    def __init__(self, fast_mode: bool = True):
        """
        Initialize data collector.
        
        Args:
            fast_mode: If True, uses faster target generation (40 targets vs 135)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize generators
        self.level_generator = DataDrivenLevelGenerator(order=1)
        self.feature_generator = RawFeatureGenerator()
        self.target_generator = OutcomeTargetGenerator(fast_mode=fast_mode)
    
    def collect_training_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        sample_every_n_bars: int = 10
    ) -> pd.DataFrame:
        """
        Collect SR level training data using walk-forward approach.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            exchange: Exchange name (e.g., 'binance')
            timeframe: Timeframe (e.g., '1h')
            start_date: Start date for collection
            end_date: End date for collection
            sample_every_n_bars: Sample frequency (default: every 10 bars)
        
        Returns:
            DataFrame with features and targets for all level candidates
        """
        self.logger.info(f"📊 Collecting SR training data for {symbol} {exchange} {timeframe}")
        self.logger.info(f"   Period: {start_date} to {end_date}")
        self.logger.info(f"   Sample frequency: every {sample_every_n_bars} bars")
        
        # Load historical OHLCV data
        ohlcv_data = self._load_historical_data(symbol, exchange, timeframe, start_date, end_date)
        
        if ohlcv_data is None or len(ohlcv_data) < 300:
            raise ValueError(f"Insufficient data for {symbol} {exchange} {timeframe}")
        
        self.logger.info(f"✅ Loaded {len(ohlcv_data)} historical bars")
        self.logger.info(f"   Date range: {ohlcv_data.index.min()} to {ohlcv_data.index.max()}")
        
        # Walk forward through time
        training_samples = []
        
        # Need minimum history for features (200 bars) and future for targets (100 bars)
        min_history = 200
        min_future = 100
        
        # Sample points
        sample_indices = range(min_history, len(ohlcv_data) - min_future, sample_every_n_bars)
        
        self.logger.info(f"🔄 Processing {len(sample_indices)} sample points...")
        
        for current_idx in tqdm(sample_indices, desc="Collecting samples"):
            try:
                # Get historical and future data
                historical = ohlcv_data.iloc[:current_idx]
                
                # Generate all candidate levels from historical data
                candidates = self.level_generator.generate_all_candidates(historical)
                
                if not candidates:
                    continue
                
                # Process each candidate
                for level in candidates:
                    try:
                        # Extract exhaustive features
                        features = self.feature_generator.generate_exhaustive_features(
                            level['price'],
                            level['idx'],
                            historical
                        )
                        
                        # Generate all outcome targets
                        targets = self.target_generator.generate_all_targets(
                            level['price'],
                            level['idx'],
                            ohlcv_data  # Pass full data so it can access future
                        )
                        
                        # Skip if no valid targets
                        if not targets:
                            continue
                        
                        # Combine into training sample
                        sample = {
                            'date': ohlcv_data.index[current_idx],
                            'symbol': symbol,
                            'exchange': exchange,
                            'timeframe': timeframe,
                            'level_price': level['price'],
                            'level_idx': level['idx'],
                            'level_type': level['type'],
                            'current_idx': current_idx,
                            **features,
                            **targets
                        }
                        
                        training_samples.append(sample)
                        
                    except Exception as e:
                        self.logger.debug(f"Failed to process level at {level['price']}: {e}")
                        continue
            
            except Exception as e:
                self.logger.warning(f"Failed to process sample at index {current_idx}: {e}")
                continue
        
        if not training_samples:
            raise ValueError("No training samples collected!")
        
        # Convert to DataFrame
        df = pd.DataFrame(training_samples)
        
        self.logger.info(f"\n✅ Training data collection complete!")
        self.logger.info(f"   Total samples: {len(df):,}")
        self.logger.info(f"   Unique dates: {df['date'].nunique()}")
        self.logger.info(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Count features and targets
        feature_cols = [c for c in df.columns if any(c.startswith(p) for p in [
            'dist_', 'crosses_', 'vol_', 'ret_', 'range_', 'atr_', 'time_at_', 'close_'
        ])]
        target_cols = [c for c in df.columns if any(c.startswith(p) for p in [
            'max_', 'touch_', 'break_', 'reversal_', 'vol_change', 'volume_surge', 
            'net_move', 'bars_to', 'vol_spike', 'volume_spike'
        ])]
        
        self.logger.info(f"   Features: {len(feature_cols)}")
        self.logger.info(f"   Targets: {len(target_cols)}")
        
        # Save with artifact manager
        self._save_with_artifact_manager(df, symbol, exchange, timeframe, feature_cols, target_cols)
        
        return df
    
    def _load_historical_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Load historical OHLCV data from processed directory.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
        
        Returns:
            DataFrame with OHLCV data or None if not found
        """
        try:
            # Path to processed data
            base_path = Path(f"historical_data/{exchange}/{symbol.lower()}/processed/{symbol.lower()}_{timeframe}")
            
            if not base_path.exists():
                self.logger.error(f"Data directory not found: {base_path}")
                return None
            
            # Load all parquet files
            parquet_files = list(base_path.glob("**/*.parquet"))
            
            if not parquet_files:
                self.logger.error(f"No parquet files found in {base_path}")
                return None
            
            self.logger.info(f"Loading from {len(parquet_files)} parquet file(s)...")
            
            # Read all files
            dfs = []
            for file in parquet_files:
                try:
                    df = pd.read_parquet(file)
                    
                    # Convert to timezone-naive immediately to avoid comparison issues
                    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    
                    dfs.append(df)
                except Exception as e:
                    self.logger.warning(f"Failed to load {file}: {e}")
            
            if not dfs:
                return None
            
            # Concatenate (all dataframes now timezone-naive)
            data = pd.concat(dfs, axis=0)
            
            # Ensure required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required):
                self.logger.error(f"Missing required columns. Found: {data.columns.tolist()}")
                return None
            
            # Sort by index
            data = data.sort_index()
            
            # Filter date range (timezone-naive now)
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            data = data[(data.index >= start_dt) & (data.index <= end_dt)]
            
            # Remove duplicates
            data = data[~data.index.duplicated(keep='first')]
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load historical data: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _save_with_artifact_manager(
        self,
        df: pd.DataFrame,
        symbol: str,
        exchange: str,
        timeframe: str,
        feature_cols: List[str],
        target_cols: List[str]
    ):
        """
        Save training data using artifact manager.
        
        Args:
            df: Training data DataFrame
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            feature_cols: List of feature column names
            target_cols: List of target column names
        """
        try:
            with artifact_context(
                symbol=symbol,
                exchange=exchange,
                information="sr_ml_training",
                timeframe=timeframe
            ) as am:
                # Separate features and targets
                metadata_cols = ['date', 'symbol', 'exchange', 'timeframe', 'level_price', 
                               'level_idx', 'level_type', 'current_idx']
                
                # Create separate DataFrames
                features_df = df[feature_cols].copy()
                targets_df = df[target_cols].copy()
                metadata_df = df[metadata_cols].copy()
                
                # Save joint parquet
                filepath = am.create_joint_parquet_file(
                    step_name="sr_training_data",
                    ohlcv_data=metadata_df,  # Use metadata as 'ohlcv' slot
                    features_data=features_df,
                    labels_data=targets_df,
                    key='joint_dataset'
                )
                
                self.logger.info(f"✅ Saved training data via artifact manager: {filepath}")
                
        except Exception as e:
            self.logger.warning(f"Failed to save via artifact manager: {e}")
            # Fallback: save locally
            self._save_locally(df, symbol, exchange, timeframe)
    
    def _save_locally(
        self,
        df: pd.DataFrame,
        symbol: str,
        exchange: str,
        timeframe: str
    ):
        """Fallback: save training data locally."""
        try:
            output_dir = Path("data_cache/sr_ml_training")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"sr_training_{symbol}_{exchange}_{timeframe}_{timestamp}.parquet"
            filepath = output_dir / filename
            
            df.to_parquet(filepath, index=False)
            
            self.logger.info(f"✅ Saved training data locally: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save locally: {e}")

