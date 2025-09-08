#!/usr/bin/env python3
"""
Step07 Enhanced Matrix Operations - Standalone Version

This is a completely standalone version that works without any external dependencies.
It provides basic matrix operations functionality using only Python standard library.
"""

import sys
import os
import time
import json
import math
import statistics
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Step07Standalone')

class StandaloneMatrixOperations:
    """Standalone matrix operations using only Python standard library."""
    
    def __init__(self):
        self.logger = logger.getChild('StandaloneMatrixOps')
        self.logger.info("🔢 Initialized standalone matrix operations")
    
    def compute_correlation_matrix(self, data: List[List[float]]) -> List[List[float]]:
        """Compute correlation matrix using standard library only."""
        if not data or len(data) == 0:
            return []
        
        n_features = len(data[0])
        n_samples = len(data)
        
        # Initialize correlation matrix
        corr_matrix = [[0.0 for _ in range(n_features)] for _ in range(n_features)]
        
        for i in range(n_features):
            for j in range(n_features):
                if i == j:
                    corr_matrix[i][j] = 1.0
                else:
                    # Extract columns
                    col_i = [row[i] for row in data]
                    col_j = [row[j] for row in data]
                    
                    # Compute correlation
                    corr_matrix[i][j] = self._compute_correlation(col_i, col_j)
        
        return corr_matrix
    
    def compute_covariance_matrix(self, data: List[List[float]]) -> List[List[float]]:
        """Compute covariance matrix using standard library only."""
        if not data or len(data) == 0:
            return []
        
        n_features = len(data[0])
        n_samples = len(data)
        
        # Initialize covariance matrix
        cov_matrix = [[0.0 for _ in range(n_features)] for _ in range(n_features)]
        
        # Compute means
        means = []
        for j in range(n_features):
            col = [row[j] for row in data]
            means.append(statistics.mean(col))
        
        # Compute covariances
        for i in range(n_features):
            for j in range(n_features):
                if i == j:
                    # Variance
                    col = [row[i] for row in data]
                    cov_matrix[i][j] = statistics.variance(col)
                else:
                    # Covariance
                    col_i = [row[i] for row in data]
                    col_j = [row[j] for row in data]
                    cov_matrix[i][j] = self._compute_covariance(col_i, col_j, means[i], means[j])
        
        return cov_matrix
    
    def _compute_correlation(self, x: List[float], y: List[float]) -> float:
        """Compute Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) == 0:
            return 0.0
        
        n = len(x)
        
        # Compute means
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)
        
        # Compute correlation
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        
        sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(n))
        sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(n))
        
        denominator = math.sqrt(sum_sq_x * sum_sq_y)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _compute_covariance(self, x: List[float], y: List[float], mean_x: float, mean_y: float) -> float:
        """Compute covariance between two variables."""
        if len(x) != len(y) or len(x) == 0:
            return 0.0
        
        n = len(x)
        return sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / (n - 1)
    
    def compute_feature_statistics(self, data: List[List[float]]) -> Dict[str, List[float]]:
        """Compute basic statistics for each feature."""
        if not data or len(data) == 0:
            return {}
        
        n_features = len(data[0])
        stats = {
            'mean': [],
            'std': [],
            'min': [],
            'max': [],
            'count': []
        }
        
        for j in range(n_features):
            col = [row[j] for row in data]
            
            stats['mean'].append(statistics.mean(col))
            stats['std'].append(statistics.stdev(col) if len(col) > 1 else 0.0)
            stats['min'].append(min(col))
            stats['max'].append(max(col))
            stats['count'].append(len(col))
        
        return stats
    
    def compute_feature_importance(self, data: List[List[float]], target: List[float]) -> List[float]:
        """Compute feature importance using correlation with target."""
        if not data or len(data) == 0 or len(target) == 0:
            return []
        
        n_features = len(data[0])
        importance = []
        
        for j in range(n_features):
            col = [row[j] for row in data]
            corr = abs(self._compute_correlation(col, target))
            importance.append(corr)
        
        return importance

class StandaloneStep07:
    """Standalone Step07 implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.step_id = '07'
        self.step_name = 'standalone_matrix_operations'
        self.logger = logger.getChild('StandaloneStep07')
        self.matrix_ops = StandaloneMatrixOperations()
        
        self.logger.info("🔢 Initialized standalone Step07")
    
    def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute standalone matrix operations."""
        self.logger.info("🚀 Starting standalone matrix operations...")
        
        try:
            # Get data
            data_dict = self._get_data_to_process(pipeline_state)
            if not data_dict:
                self.logger.error("❌ No data available for processing")
                return pipeline_state
            
            # Process each split
            matrix_results = {}
            for split_name, data in data_dict.items():
                self.logger.info(f"🧮 Processing {split_name} split...")
                matrices = self._compute_matrices_standalone(data)
                matrix_results[split_name] = matrices
            
            # Update pipeline state
            pipeline_state.update({
                'matrix_results': matrix_results,
                'step07_standalone_completed': True
            })
            
            self.logger.info("✅ Standalone matrix operations completed")
            return pipeline_state
            
        except Exception as e:
            self.logger.error(f"❌ Error in standalone matrix operations: {e}")
            return pipeline_state
    
    def _get_data_to_process(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get data to process."""
        # Try to get engineered data
        if 'engineered_data' in pipeline_state:
            return pipeline_state['engineered_data']
        
        # Try to get advanced features
        if 'advanced_features' in pipeline_state:
            advanced_features = pipeline_state['advanced_features']
            data_dict = {}
            
            for split in ['train', 'val', 'test']:
                if split in advanced_features:
                    path = advanced_features[split]
                    if isinstance(path, str) and Path(path).exists():
                        try:
                            data_dict[split] = self._load_data_from_file(path)
                        except Exception as e:
                            self.logger.warning(f"⚠️ Failed to load {split} data: {e}")
            
            if data_dict:
                return data_dict
        
        # Fallback to individual data keys
        data_dict = {}
        for split in ['train', 'val', 'test']:
            if f'{split}_data' in pipeline_state:
                data_dict[split] = pipeline_state[f'{split}_data']
        
        return data_dict
    
    def _load_data_from_file(self, file_path: str) -> List[List[float]]:
        """Load data from file (basic CSV support)."""
        try:
            data = []
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        # Basic CSV parsing
                        values = line.strip().split(',')
                        try:
                            row = [float(v) for v in values]
                            data.append(row)
                        except ValueError:
                            # Skip non-numeric rows
                            continue
            return data
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load data from {file_path}: {e}")
            return []
    
    def _compute_matrices_standalone(self, data: Any) -> Dict[str, Any]:
        """Compute matrices using standalone operations."""
        matrices = {}
        
        try:
            # Convert data to list of lists if needed
            if isinstance(data, list):
                matrix_data = data
            elif hasattr(data, 'values'):  # pandas DataFrame
                matrix_data = data.values.tolist()
            else:
                self.logger.warning("⚠️ Unsupported data type, skipping matrix computations")
                return matrices
            
            if not matrix_data or len(matrix_data) == 0:
                self.logger.warning("⚠️ No data available for matrix computation")
                return matrices
            
            # Compute correlation matrix
            try:
                corr_matrix = self.matrix_ops.compute_correlation_matrix(matrix_data)
                matrices['correlation_matrix'] = corr_matrix
                self.logger.info(f"✅ Computed correlation matrix: {len(corr_matrix)}x{len(corr_matrix[0]) if corr_matrix else 0}")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to compute correlation matrix: {e}")
            
            # Compute covariance matrix
            try:
                cov_matrix = self.matrix_ops.compute_covariance_matrix(matrix_data)
                matrices['covariance_matrix'] = cov_matrix
                self.logger.info(f"✅ Computed covariance matrix: {len(cov_matrix)}x{len(cov_matrix[0]) if cov_matrix else 0}")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to compute covariance matrix: {e}")
            
            # Compute feature statistics
            try:
                feature_stats = self.matrix_ops.compute_feature_statistics(matrix_data)
                matrices['feature_statistics'] = feature_stats
                self.logger.info(f"✅ Computed feature statistics for {len(feature_stats.get('mean', []))} features")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to compute feature statistics: {e}")
                
        except Exception as e:
            self.logger.error(f"❌ Error in standalone matrix computation: {e}")
        
        return matrices
    
    def validate_inputs(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate inputs."""
        errors = []
        
        # Check for data
        has_data = (
            'engineered_data' in pipeline_state or
            'advanced_features' in pipeline_state or
            any(f'{split}_data' in pipeline_state for split in ['train', 'val', 'test'])
        )
        
        if not has_data:
            errors.append('No data available for processing')
        
        return len(errors) == 0, errors
    
    def get_required_inputs(self) -> List[str]:
        """Get required inputs."""
        return ['engineered_data or split data']
    
    def get_produced_outputs(self) -> List[str]:
        """Get produced outputs."""
        return ['matrix_results']

def create_standalone_step07(config: Dict[str, Any]) -> StandaloneStep07:
    """Create a standalone Step07 step instance."""
    return StandaloneStep07(config)

def test_standalone_step07():
    """Test the standalone Step07 implementation."""
    print("🧪 Testing Standalone Step07")
    print("=" * 40)
    
    # Create test data
    test_data = [
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0, 6.0],
        [4.0, 5.0, 6.0, 7.0],
        [5.0, 6.0, 7.0, 8.0]
    ]
    
    # Create step
    config = {'matrix_operations_config': {'batch_size': 1000}}
    step = create_standalone_step07(config)
    
    print(f"✅ Created step: {step.step_name}")
    print(f"📊 Required inputs: {step.get_required_inputs()}")
    print(f"📤 Produced outputs: {step.get_produced_outputs()}")
    
    # Test matrix operations
    print("\n🧮 Testing matrix operations...")
    matrix_ops = StandaloneMatrixOperations()
    
    # Test correlation matrix
    corr_matrix = matrix_ops.compute_correlation_matrix(test_data)
    print(f"✅ Correlation matrix: {len(corr_matrix)}x{len(corr_matrix[0]) if corr_matrix else 0}")
    
    # Test covariance matrix
    cov_matrix = matrix_ops.compute_covariance_matrix(test_data)
    print(f"✅ Covariance matrix: {len(cov_matrix)}x{len(cov_matrix[0]) if cov_matrix else 0}")
    
    # Test feature statistics
    stats = matrix_ops.compute_feature_statistics(test_data)
    print(f"✅ Feature statistics: {len(stats.get('mean', []))} features")
    
    # Test execution
    print("\n🚀 Testing step execution...")
    training_input = {'symbol': 'BTCUSDT', 'exchange': 'binance', 'timeframe': '1h'}
    pipeline_state = {'engineered_data': {'train': test_data}}
    
    result = step.execute(training_input, pipeline_state)
    
    if 'step07_standalone_completed' in result:
        print("✅ Step execution completed successfully")
        print(f"📊 Matrix results: {list(result.get('matrix_results', {}).keys())}")
    else:
        print("❌ Step execution failed")
    
    return True

if __name__ == "__main__":
    test_standalone_step07()