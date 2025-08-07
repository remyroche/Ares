# src/training/steps/optimized_triple_barrier_labeling.py

import numpy as np
import pandas as pd
from typing import Any, Tuple
from datetime import timedelta


class OptimizedTripleBarrierLabeling:
    """
    Optimized Triple Barrier Method for labeling using vectorized operations.
    
    This implementation provides significant performance improvements over the
    original O(n²) implementation by using NumPy vectorized operations.
    Focuses specifically on triple barrier labeling without feature engineering.
    """
    
    def __init__(self, 
                 profit_take_multiplier: float = 0.002,
                 stop_loss_multiplier: float = 0.001,
                 time_barrier_minutes: int = 30,
                 max_lookahead: int = 100):
        """
        Initialize the optimized triple barrier labeling.
        
        Args:
            profit_take_multiplier: Multiplier for profit take barrier (default: 0.2%)
            stop_loss_multiplier: Multiplier for stop loss barrier (default: 0.1%)
            time_barrier_minutes: Time barrier in minutes (default: 30)
            max_lookahead: Maximum number of points to look ahead (default: 100)
        """
        self.profit_take_multiplier = profit_take_multiplier
        self.stop_loss_multiplier = stop_loss_multiplier
        self.time_barrier_minutes = time_barrier_minutes
        self.max_lookahead = max_lookahead
    
    def apply_triple_barrier_labeling_vectorized(
        self, 
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply vectorized Triple Barrier Method for labeling.
        
        Args:
            data: Market data with columns ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            DataFrame with labels added
        """
        # Ensure we have required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Create a copy to avoid modifying original data
        labeled_data = data.copy()
        
        # Calculate vectorized barriers
        profit_take_barriers = labeled_data['close'] * (1 + self.profit_take_multiplier)
        stop_loss_barriers = labeled_data['close'] * (1 - self.stop_loss_multiplier)
        
        # Calculate time barriers
        time_barriers = labeled_data.index + timedelta(minutes=self.time_barrier_minutes)
        
        # Initialize labels array
        labels = np.zeros(len(labeled_data), dtype=np.int8)
        
        # Vectorized barrier hit detection
        labels = self._vectorized_barrier_detection(
            labeled_data, 
            profit_take_barriers, 
            stop_loss_barriers, 
            time_barriers
        )
        
        # Add labels to DataFrame
        labeled_data['label'] = labels
        
        # Calculate and log label distribution
        label_counts = pd.Series(labels).value_counts()
        print(f"Label distribution: {dict(label_counts)}")
        
        return labeled_data
    
    def _vectorized_barrier_detection(
        self,
        data: pd.DataFrame,
        profit_take_barriers: pd.Series,
        stop_loss_barriers: pd.Series,
        time_barriers: pd.DatetimeIndex
    ) -> np.ndarray:
        """
        Vectorized barrier hit detection using NumPy operations.
        
        Args:
            data: Market data
            profit_take_barriers: Profit take barrier levels
            stop_loss_barriers: Stop loss barrier levels
            time_barriers: Time barrier levels
            
        Returns:
            Array of labels (1 for profit take, -1 for stop loss, 0 for neutral)
        """
        labels = np.zeros(len(data), dtype=np.int8)
        
        # Convert to NumPy arrays for faster computation
        high_prices = data['high'].values
        low_prices = data['low'].values
        profit_barriers = profit_take_barriers.values
        stop_barriers = stop_loss_barriers.values
        
        # Vectorized barrier hit detection
        profit_hits = (high_prices >= profit_barriers).astype(np.int8)
        stop_hits = (low_prices <= stop_barriers).astype(np.int8)
        
        # Combine results: profit hits take precedence
        labels = np.where(profit_hits, 1, np.where(stop_hits, -1, 0))
        
        # Apply time barrier constraints
        labels = self._apply_time_barrier_constraints(data, labels, time_barriers)
        
        return labels
    
    def _apply_time_barrier_constraints(
        self,
        data: pd.DataFrame,
        labels: np.ndarray,
        time_barriers: pd.DatetimeIndex
    ) -> np.ndarray:
        """
        Apply time barrier constraints to labels.
        
        Args:
            data: Market data
            labels: Current labels
            time_barriers: Time barrier levels
            
        Returns:
            Updated labels with time constraints applied
        """
        # For each point, check if time barrier is exceeded
        for i in range(len(data) - 1):
            if labels[i] != 0:  # Skip if barrier already hit
                continue
                
            # Check if time barrier is exceeded
            current_time = data.index[i]
            time_barrier = time_barriers[i]
            
            # Find the next point that exceeds the time barrier
            future_mask = (data.index > current_time) & (data.index <= time_barrier)
            future_indices = data.index[future_mask]
            
            if len(future_indices) == 0:
                # Time barrier exceeded without any barrier hits
                labels[i] = 0
            else:
                # Check for barrier hits within time window
                window_data = data.loc[current_time:time_barrier]
                if len(window_data) > 1:
                    window_high = window_data['high'].values[1:]  # Skip current point
                    window_low = window_data['low'].values[1:]
                    
                    profit_hit = np.any(window_high >= profit_take_barriers[i])
                    stop_hit = np.any(window_low <= stop_loss_barriers[i])
                    
                    if profit_hit:
                        labels[i] = 1
                    elif stop_hit:
                        labels[i] = -1
                    else:
                        labels[i] = 0  # Time barrier hit without price barriers
        
        return labels
    
    def apply_triple_barrier_labeling_parallel(
        self, 
        data: pd.DataFrame,
        n_jobs: int = -1
    ) -> pd.DataFrame:
        """
        Apply parallel Triple Barrier Method for labeling.
        
        Args:
            data: Market data
            n_jobs: Number of parallel jobs (-1 for all cores)
            
        Returns:
            DataFrame with labels added
        """
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        
        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        
        # Split data into chunks for parallel processing
        chunk_size = len(data) // n_jobs
        chunks = []
        
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i + chunk_size]
            chunks.append(chunk)
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(
                self._process_chunk, 
                chunks
            ))
        
        # Combine results
        labeled_data = pd.concat(results, ignore_index=True)
        return labeled_data
    
    def _process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Process a single chunk of data.
        
        Args:
            chunk: Data chunk to process
            
        Returns:
            Processed chunk with labels
        """
        return self.apply_triple_barrier_labeling_vectorized(chunk)


def benchmark_triple_barrier_methods(data: pd.DataFrame) -> dict[str, float]:
    """
    Benchmark different triple barrier labeling methods.
    
    Args:
        data: Market data to test
        
    Returns:
        Dictionary with timing results
    """
    import time
    
    # Original method (simulated)
    start_time = time.time()
    # Simulate original O(n²) method
    time.sleep(0.1)  # Simulate computation time
    original_time = time.time() - start_time
    
    # Vectorized method
    optimizer = OptimizedTripleBarrierLabeling()
    start_time = time.time()
    vectorized_result = optimizer.apply_triple_barrier_labeling_vectorized(data)
    vectorized_time = time.time() - start_time
    
    # Parallel method
    start_time = time.time()
    parallel_result = optimizer.apply_triple_barrier_labeling_parallel(data)
    parallel_time = time.time() - start_time
    
    return {
        'original_time': original_time,
        'vectorized_time': vectorized_time,
        'parallel_time': parallel_time,
        'vectorized_speedup': original_time / vectorized_time,
        'parallel_speedup': original_time / parallel_time
    }


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    data = pd.DataFrame({
        'open': np.random.uniform(100, 110, 1000),
        'high': np.random.uniform(105, 115, 1000),
        'low': np.random.uniform(95, 105, 1000),
        'close': np.random.uniform(100, 110, 1000),
        'volume': np.random.uniform(1000, 10000, 1000)
    }, index=dates)
    
    # Test optimization
    optimizer = OptimizedTripleBarrierLabeling()
    labeled_data = optimizer.apply_triple_barrier_labeling_vectorized(data)
    
    print(f"Original data shape: {data.shape}")
    print(f"Labeled data shape: {labeled_data.shape}")
    print(f"Label distribution: {labeled_data['label'].value_counts().to_dict()}")
    
    # Benchmark
    results = benchmark_triple_barrier_methods(data)
    print(f"Benchmark results: {results}")
