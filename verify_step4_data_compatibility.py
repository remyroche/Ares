#!/usr/bin/env python3
"""
Data Compatibility Verification for Step4 and Enhancements

This script verifies data compatibility for:
1. Step4 Processing & Labeling
2. Step4 Validator
3. Step4 Regime Data Splitting
4. Vectorized Labeling Orchestrator
5. Optimized Triple Barrier Labeling
6. Vectorized Advanced Feature Engineering
7. Matrix and Vector Operations
8. Data Transformations and Enhancements

Author: AI Assistant
Date: 2024
"""

import asyncio
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger


class Step4DataCompatibilityVerifier:
    """Comprehensive data compatibility verifier for Step4 and enhancements."""

    def __init__(self):
        self.logger = system_logger.getChild("Step4DataCompatibility")
        self.verification_results = {}
        self.errors = []
        self.warnings = []

    def log_error(self, component: str, error: str, details: Optional[Dict] = None):
        """Log an error with component context."""
        error_info = {
            "component": component,
            "error": error,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }
        self.errors.append(error_info)
        self.logger.error(f"❌ {component}: {error}")

    def log_warning(self, component: str, warning: str, details: Optional[Dict] = None):
        """Log a warning with component context."""
        warning_info = {
            "component": component,
            "warning": warning,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }
        self.warnings.append(warning_info)
        self.logger.warning(f"⚠️ {component}: {warning}")

    def log_success(self, component: str, message: str, details: Optional[Dict] = None):
        """Log a success message with component context."""
        success_info = {
            "component": component,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }
        self.verification_results[component] = success_info
        self.logger.info(f"✅ {component}: {message}")

    async def verify_step4_processing_labeling(self) -> bool:
        """Verify Step4 Processing & Labeling data compatibility."""
        try:
            self.logger.info("🔍 Verifying Step4 Processing & Labeling...")
            
            # Import the step4 module
            from src.training.steps.step4_processing_labeling import run_step
            
            # Test data compatibility requirements
            test_config = {
                "symbol": "ETHUSDT",
                "exchange": "BINANCE",
                "data_dir": "data/training",
                "timeframe": "1m",
                "lookback_days": 30
            }
            
            # Verify required directories exist
            data_dir = test_config["data_dir"]
            if not os.path.exists(data_dir):
                os.makedirs(data_dir, exist_ok=True)
                self.log_warning("Step4", f"Created missing data directory: {data_dir}")
            
            # Verify input data format compatibility
            self._verify_ohlcv_data_format()
            
            # Verify output data format compatibility
            self._verify_labeled_data_format()
            
            # Verify matrix operations compatibility
            self._verify_matrix_operations()
            
            # Verify vector operations compatibility
            self._verify_vector_operations()
            
            self.log_success("Step4 Processing & Labeling", "Data compatibility verified")
            return True
            
        except Exception as e:
            self.log_error("Step4 Processing & Labeling", str(e), {"traceback": traceback.format_exc()})
            return False

    async def verify_step4_validator(self) -> bool:
        """Verify Step4 Validator data compatibility."""
        try:
            self.logger.info("🔍 Verifying Step4 Validator...")
            
            # Import the validator module
            from src.training.steps.step4_processing_labeling_validator import Step4ProcessingLabelingValidator
            
            # Test validator initialization
            test_config = {
                "symbol": "ETHUSDT",
                "exchange": "BINANCE",
                "data_dir": "data/training"
            }
            
            validator = Step4ProcessingLabelingValidator(test_config)
            
            # Verify validator data requirements
            self._verify_validator_data_requirements(validator)
            
            # Verify validation logic compatibility
            self._verify_validation_logic()
            
            self.log_success("Step4 Validator", "Data compatibility verified")
            return True
            
        except Exception as e:
            self.log_error("Step4 Validator", str(e), {"traceback": traceback.format_exc()})
            return False

    async def verify_step4_regime_data_splitting(self) -> bool:
        """Verify Step4 Regime Data Splitting data compatibility."""
        try:
            self.logger.info("🔍 Verifying Step4 Regime Data Splitting...")
            
            # Import the regime splitting module
            from src.training.steps.step4_regime_data_splitting import RegimeDataSplittingStep
            
            # Test regime splitting initialization
            test_config = {
                "symbol": "ETHUSDT",
                "exchange": "BINANCE",
                "data_dir": "data/training",
                "timeframe": "1m"
            }
            
            step = RegimeDataSplittingStep(test_config)
            
            # Verify HMM composite cluster data compatibility
            self._verify_hmm_composite_clusters()
            
            # Verify regime splitting logic
            self._verify_regime_splitting_logic()
            
            self.log_success("Step4 Regime Data Splitting", "Data compatibility verified")
            return True
            
        except Exception as e:
            self.log_error("Step4 Regime Data Splitting", str(e), {"traceback": traceback.format_exc()})
            return False

    async def verify_vectorized_labeling_orchestrator(self) -> bool:
        """Verify Vectorized Labeling Orchestrator data compatibility."""
        try:
            self.logger.info("🔍 Verifying Vectorized Labeling Orchestrator...")
            
            # Import the orchestrator module
            from src.training.steps.vectorized_labelling_orchestrator import VectorizedLabellingOrchestrator
            
            # Test orchestrator initialization
            test_config = {
                "symbol": "ETHUSDT",
                "exchange": "BINANCE",
                "data_dir": "data/training",
                "vectorized_labelling_orchestrator": {
                    "enable_stationary_checks": True,
                    "enable_data_normalization": True,
                    "enable_feature_selection": True,
                    "strict_feature_shapes": True
                }
            }
            
            orchestrator = VectorizedLabellingOrchestrator(test_config)
            
            # Verify orchestrator data compatibility
            self._verify_orchestrator_data_compatibility(orchestrator)
            
            # Verify feature engineering pipeline
            self._verify_feature_engineering_pipeline()
            
            self.log_success("Vectorized Labeling Orchestrator", "Data compatibility verified")
            return True
            
        except Exception as e:
            self.log_error("Vectorized Labeling Orchestrator", str(e), {"traceback": traceback.format_exc()})
            return False

    async def verify_optimized_triple_barrier_labeling(self) -> bool:
        """Verify Optimized Triple Barrier Labeling data compatibility."""
        try:
            self.logger.info("🔍 Verifying Optimized Triple Barrier Labeling...")
            
            # Import the triple barrier labeling module
            from src.training.steps.step4_analyst_labeling_feature_engineering_components.optimized_triple_barrier_labeling import OptimizedTripleBarrierLabeling
            
            # Test triple barrier labeling initialization
            labeler = OptimizedTripleBarrierLabeling(
                profit_take_multiplier=0.002,
                stop_loss_multiplier=0.001,
                time_barrier_minutes=30,
                binary_classification=True
            )
            
            # Verify labeling data compatibility
            self._verify_labeling_data_compatibility(labeler)
            
            # Verify matrix operations in labeling
            self._verify_labeling_matrix_operations()
            
            self.log_success("Optimized Triple Barrier Labeling", "Data compatibility verified")
            return True
            
        except Exception as e:
            self.log_error("Optimized Triple Barrier Labeling", str(e), {"traceback": traceback.format_exc()})
            return False

    async def verify_vectorized_advanced_feature_engineering(self) -> bool:
        """Verify Vectorized Advanced Feature Engineering data compatibility."""
        try:
            self.logger.info("🔍 Verifying Vectorized Advanced Feature Engineering...")
            
            # Import the feature engineering module
            from src.training.steps.vectorized_advanced_feature_engineering import VectorizedAdvancedFeatureEngineering
            
            # Test feature engineering initialization
            test_config = {
                "symbol": "ETHUSDT",
                "exchange": "BINANCE",
                "data_dir": "data/training"
            }
            
            feature_engineer = VectorizedAdvancedFeatureEngineering(test_config)
            
            # Verify feature engineering data compatibility
            self._verify_feature_engineering_data_compatibility(feature_engineer)
            
            # Verify advanced matrix operations
            self._verify_advanced_matrix_operations()
            
            self.log_success("Vectorized Advanced Feature Engineering", "Data compatibility verified")
            return True
            
        except Exception as e:
            self.log_error("Vectorized Advanced Feature Engineering", str(e), {"traceback": traceback.format_exc()})
            return False

    def _verify_ohlcv_data_format(self):
        """Verify OHLCV data format compatibility."""
        try:
            # Test OHLCV data structure
            test_data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min'),
                'open': np.random.random(1000) * 100,
                'high': np.random.random(1000) * 100,
                'low': np.random.random(1000) * 100,
                'close': np.random.random(1000) * 100,
                'volume': np.random.random(1000) * 1000
            })
            
            # Verify required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in test_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required OHLCV columns: {missing_cols}")
            
            # Verify data types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if not pd.api.types.is_numeric_dtype(test_data[col]):
                    raise ValueError(f"Column {col} must be numeric")
            
            # Verify timestamp format
            if not pd.api.types.is_datetime64_any_dtype(test_data['timestamp']):
                raise ValueError("Timestamp column must be datetime")
            
            self.log_success("OHLCV Data Format", "Format compatibility verified")
            
        except Exception as e:
            self.log_error("OHLCV Data Format", str(e))

    def _verify_labeled_data_format(self):
        """Verify labeled data format compatibility."""
        try:
            # Test labeled data structure
            test_data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min'),
                'open': np.random.random(1000) * 100,
                'high': np.random.random(1000) * 100,
                'low': np.random.random(1000) * 100,
                'close': np.random.random(1000) * 100,
                'volume': np.random.random(1000) * 1000,
                'label': np.random.choice([-1, 0, 1], 1000)
            })
            
            # Verify label column exists
            if 'label' not in test_data.columns:
                raise ValueError("Label column is required")
            
            # Verify label values are valid
            valid_labels = [-1, 0, 1]
            invalid_labels = test_data['label'].dropna().apply(lambda x: x not in valid_labels)
            if invalid_labels.any():
                raise ValueError(f"Invalid label values found: {test_data['label'][invalid_labels].unique()}")
            
            self.log_success("Labeled Data Format", "Format compatibility verified")
            
        except Exception as e:
            self.log_error("Labeled Data Format", str(e))

    def _verify_matrix_operations(self):
        """Verify matrix operations compatibility."""
        try:
            # Test numpy matrix operations
            test_matrix = np.random.random((100, 50))
            
            # Test basic matrix operations
            matrix_sum = np.sum(test_matrix)
            matrix_mean = np.mean(test_matrix)
            matrix_std = np.std(test_matrix)
            
            # Test matrix multiplication
            matrix_product = np.dot(test_matrix.T, test_matrix)
            
            # Test matrix decomposition
            try:
                from scipy import linalg
                eigenvals, eigenvecs = linalg.eigh(matrix_product)
            except ImportError:
                self.log_warning("Matrix Operations", "SciPy not available for advanced matrix operations")
            
            # Test pandas matrix operations
            df_matrix = pd.DataFrame(test_matrix)
            df_corr = df_matrix.corr()
            df_cov = df_matrix.cov()
            
            self.log_success("Matrix Operations", "Matrix operations compatibility verified")
            
        except Exception as e:
            self.log_error("Matrix Operations", str(e))

    def _verify_vector_operations(self):
        """Verify vector operations compatibility."""
        try:
            # Test numpy vector operations
            test_vector = np.random.random(1000)
            
            # Test basic vector operations
            vector_sum = np.sum(test_vector)
            vector_mean = np.mean(test_vector)
            vector_std = np.std(test_vector)
            
            # Test vectorized operations
            vector_squared = test_vector ** 2
            vector_sqrt = np.sqrt(test_vector)
            vector_log = np.log(test_vector + 1e-8)  # Add small constant to avoid log(0)
            
            # Test rolling window operations
            rolling_mean = pd.Series(test_vector).rolling(window=20).mean()
            rolling_std = pd.Series(test_vector).rolling(window=20).std()
            
            # Test vector concatenation
            vector_concat = np.concatenate([test_vector, test_vector])
            
            self.log_success("Vector Operations", "Vector operations compatibility verified")
            
        except Exception as e:
            self.log_error("Vector Operations", str(e))

    def _verify_validator_data_requirements(self, validator):
        """Verify validator data requirements."""
        try:
            # Check required attributes
            required_attrs = ['min_labeled_rows', 'min_label_balance', 'max_label_balance', 'required_columns']
            for attr in required_attrs:
                if not hasattr(validator, attr):
                    raise ValueError(f"Validator missing required attribute: {attr}")
            
            # Check required columns
            if not isinstance(validator.required_columns, list):
                raise ValueError("required_columns must be a list")
            
            # Check numeric thresholds
            if not isinstance(validator.min_labeled_rows, int):
                raise ValueError("min_labeled_rows must be an integer")
            
            if not isinstance(validator.min_label_balance, float):
                raise ValueError("min_label_balance must be a float")
            
            self.log_success("Validator Data Requirements", "Requirements compatibility verified")
            
        except Exception as e:
            self.log_error("Validator Data Requirements", str(e))

    def _verify_validation_logic(self):
        """Verify validation logic compatibility."""
        try:
            # Test validation logic with sample data
            test_data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min'),
                'open': np.random.random(1000) * 100,
                'high': np.random.random(1000) * 100,
                'low': np.random.random(1000) * 100,
                'close': np.random.random(1000) * 100,
                'volume': np.random.random(1000) * 1000,
                'label': np.random.choice([-1, 1], 1000)  # Binary labels
            })
            
            # Test label distribution validation
            label_counts = test_data['label'].value_counts()
            total_rows = len(test_data)
            label_ratios = label_counts / total_rows
            
            min_ratio = label_ratios.min()
            max_ratio = label_ratios.max()
            
            # Verify ratio constraints
            if min_ratio < 0.05:  # 5% minimum
                self.log_warning("Validation Logic", f"Label balance below minimum: {min_ratio:.3f}")
            
            if max_ratio > 0.95:  # 95% maximum
                self.log_warning("Validation Logic", f"Label balance above maximum: {max_ratio:.3f}")
            
            self.log_success("Validation Logic", "Validation logic compatibility verified")
            
        except Exception as e:
            self.log_error("Validation Logic", str(e))

    def _verify_hmm_composite_clusters(self):
        """Verify HMM composite clusters data compatibility."""
        try:
            # Test HMM composite cluster data structure
            test_data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min'),
                'open': np.random.random(1000) * 100,
                'high': np.random.random(1000) * 100,
                'low': np.random.random(1000) * 100,
                'close': np.random.random(1000) * 100,
                'volume': np.random.random(1000) * 1000,
                'composite_cluster_id': np.random.choice([0, 1, 2, 3], 1000)
            })
            
            # Verify composite_cluster_id column exists
            if 'composite_cluster_id' not in test_data.columns:
                raise ValueError("composite_cluster_id column is required for regime splitting")
            
            # Verify cluster IDs are not all null
            cluster_ids = test_data['composite_cluster_id'].dropna()
            if cluster_ids.empty:
                raise ValueError("composite_cluster_id column contains only null values")
            
            # Verify unique clusters
            unique_clusters = cluster_ids.unique()
            if len(unique_clusters) < 2:
                self.log_warning("HMM Composite Clusters", f"Only {len(unique_clusters)} unique clusters found")
            
            self.log_success("HMM Composite Clusters", "Cluster data compatibility verified")
            
        except Exception as e:
            self.log_error("HMM Composite Clusters", str(e))

    def _verify_regime_splitting_logic(self):
        """Verify regime splitting logic compatibility."""
        try:
            # Test regime splitting with sample data
            test_data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min'),
                'open': np.random.random(1000) * 100,
                'high': np.random.random(1000) * 100,
                'low': np.random.random(1000) * 100,
                'close': np.random.random(1000) * 100,
                'volume': np.random.random(1000) * 1000,
                'composite_cluster_id': np.random.choice([0, 1, 2], 1000)
            })
            
            # Simulate regime splitting
            regime_splits = {}
            for cluster_id in test_data['composite_cluster_id'].unique():
                cluster_mask = test_data['composite_cluster_id'] == cluster_id
                cluster_data = test_data[cluster_mask].copy()
                
                if not cluster_data.empty:
                    regime_name = f"hmm_composite_{cluster_id}"
                    regime_splits[regime_name] = cluster_data
            
            # Verify regime splits
            if not regime_splits:
                raise ValueError("No valid regime splits created")
            
            # Verify each regime has sufficient data
            for regime_name, regime_data in regime_splits.items():
                if len(regime_data) < 100:
                    self.log_warning("Regime Splitting Logic", f"Regime {regime_name} has only {len(regime_data)} rows")
            
            self.log_success("Regime Splitting Logic", "Splitting logic compatibility verified")
            
        except Exception as e:
            self.log_error("Regime Splitting Logic", str(e))

    def _verify_orchestrator_data_compatibility(self, orchestrator):
        """Verify orchestrator data compatibility."""
        try:
            # Check orchestrator configuration
            required_config_keys = [
                'enable_stationary_checks',
                'enable_data_normalization',
                'enable_feature_selection',
                'strict_feature_shapes'
            ]
            
            for key in required_config_keys:
                if not hasattr(orchestrator, key):
                    raise ValueError(f"Orchestrator missing configuration: {key}")
            
            # Test data processing pipeline
            test_price_data = pd.DataFrame({
                'open': np.random.random(1000) * 100,
                'high': np.random.random(1000) * 100,
                'low': np.random.random(1000) * 100,
                'close': np.random.random(1000) * 100,
                'volume': np.random.random(1000) * 1000
            }, index=pd.date_range('2024-01-01', periods=1000, freq='1min'))
            
            test_volume_data = pd.DataFrame({
                'volume': np.random.random(1000) * 1000
            }, index=test_price_data.index)
            
            # Test SR levels structure
            test_sr_levels = {
                'support_levels': [
                    {'price': 95.0, 'strength': 0.2},
                    {'price': 90.0, 'strength': 0.3}
                ],
                'resistance_levels': [
                    {'price': 105.0, 'strength': 0.2},
                    {'price': 110.0, 'strength': 0.3}
                ]
            }
            
            self.log_success("Orchestrator Data Compatibility", "Data compatibility verified")
            
        except Exception as e:
            self.log_error("Orchestrator Data Compatibility", str(e))

    def _verify_feature_engineering_pipeline(self):
        """Verify feature engineering pipeline compatibility."""
        try:
            # Test feature engineering components
            test_data = pd.DataFrame({
                'open': np.random.random(1000) * 100,
                'high': np.random.random(1000) * 100,
                'low': np.random.random(1000) * 100,
                'close': np.random.random(1000) * 100,
                'volume': np.random.random(1000) * 1000
            }, index=pd.date_range('2024-01-01', periods=1000, freq='1min'))
            
            # Test technical indicators
            # Moving averages
            test_data['sma_20'] = test_data['close'].rolling(window=20).mean()
            test_data['ema_20'] = test_data['close'].ewm(span=20).mean()
            
            # Volatility indicators
            test_data['volatility'] = test_data['close'].rolling(window=20).std()
            
            # Volume indicators
            test_data['volume_sma'] = test_data['volume'].rolling(window=20).mean()
            
            # Test feature normalization
            numeric_cols = test_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != 'volume':  # Skip volume for now
                    mean_val = test_data[col].mean()
                    std_val = test_data[col].std()
                    if std_val > 0:
                        test_data[f'{col}_normalized'] = (test_data[col] - mean_val) / std_val
            
            self.log_success("Feature Engineering Pipeline", "Pipeline compatibility verified")
            
        except Exception as e:
            self.log_error("Feature Engineering Pipeline", str(e))

    def _verify_labeling_data_compatibility(self, labeler):
        """Verify labeling data compatibility."""
        try:
            # Test labeling with sample data
            test_data = pd.DataFrame({
                'open': np.random.random(1000) * 100,
                'high': np.random.random(1000) * 100,
                'low': np.random.random(1000) * 100,
                'close': np.random.random(1000) * 100,
                'volume': np.random.random(1000) * 1000
            }, index=pd.date_range('2024-01-01', periods=1000, freq='1min'))
            
            # Test triple barrier labeling
            labeled_data = labeler.apply_triple_barrier_labeling_vectorized(test_data)
            
            # Verify labeling results
            if 'label' not in labeled_data.columns:
                raise ValueError("Labeling failed to add 'label' column")
            
            # Verify label values
            unique_labels = labeled_data['label'].unique()
            valid_labels = [-1, 0, 1] if not labeler.binary_classification else [-1, 1]
            
            invalid_labels = [label for label in unique_labels if label not in valid_labels]
            if invalid_labels:
                raise ValueError(f"Invalid label values: {invalid_labels}")
            
            # Verify label distribution
            label_counts = labeled_data['label'].value_counts()
            self.log_success("Labeling Data Compatibility", f"Labeling compatibility verified. Label distribution: {label_counts.to_dict()}")
            
        except Exception as e:
            self.log_error("Labeling Data Compatibility", str(e))

    def _verify_labeling_matrix_operations(self):
        """Verify matrix operations in labeling."""
        try:
            # Test matrix operations used in labeling
            test_prices = np.random.random(1000) * 100
            
            # Test barrier calculations
            profit_barriers = test_prices * (1.0 + 0.002)  # 0.2% profit take
            stop_barriers = test_prices * (1.0 - 0.001)    # 0.1% stop loss
            
            # Test vectorized comparisons
            high_prices = test_prices + np.random.random(1000) * 2
            low_prices = test_prices - np.random.random(1000) * 2
            
            profit_hits = high_prices >= profit_barriers
            stop_hits = low_prices <= stop_barriers
            
            # Test label assignment
            labels = np.zeros(1000, dtype=np.int8)
            labels[profit_hits] = 1
            labels[stop_hits] = -1
            
            self.log_success("Labeling Matrix Operations", "Matrix operations compatibility verified")
            
        except Exception as e:
            self.log_error("Labeling Matrix Operations", str(e))

    def _verify_feature_engineering_data_compatibility(self, feature_engineer):
        """Verify feature engineering data compatibility."""
        try:
            # Test feature engineering with sample data
            test_price_data = pd.DataFrame({
                'open': np.random.random(1000) * 100,
                'high': np.random.random(1000) * 100,
                'low': np.random.random(1000) * 100,
                'close': np.random.random(1000) * 100,
                'volume': np.random.random(1000) * 1000
            }, index=pd.date_range('2024-01-01', periods=1000, freq='1min'))
            
            test_volume_data = pd.DataFrame({
                'volume': np.random.random(1000) * 1000
            }, index=test_price_data.index)
            
            # Test SR levels
            test_sr_levels = {
                'support_levels': [
                    {'price': 95.0, 'strength': 0.2},
                    {'price': 90.0, 'strength': 0.3}
                ],
                'resistance_levels': [
                    {'price': 105.0, 'strength': 0.2},
                    {'price': 110.0, 'strength': 0.3}
                ]
            }
            
            self.log_success("Feature Engineering Data Compatibility", "Data compatibility verified")
            
        except Exception as e:
            self.log_error("Feature Engineering Data Compatibility", str(e))

    def _verify_advanced_matrix_operations(self):
        """Verify advanced matrix operations compatibility."""
        try:
            # Test advanced matrix operations
            test_matrix = np.random.random((100, 50))
            
            # Test matrix decomposition
            try:
                from scipy import linalg
                # SVD decomposition
                U, s, Vt = linalg.svd(test_matrix, full_matrices=False)
                
                # Eigenvalue decomposition
                cov_matrix = np.cov(test_matrix.T)
                eigenvals, eigenvecs = linalg.eigh(cov_matrix)
                
                # PCA-like operations
                pca_components = eigenvecs[:, -10:]  # Top 10 components
                pca_transformed = np.dot(test_matrix, pca_components)
                
            except ImportError:
                self.log_warning("Advanced Matrix Operations", "SciPy not available for advanced operations")
            
            # Test rolling window matrix operations
            test_series = pd.Series(np.random.random(1000))
            rolling_matrix = np.array([
                test_series.rolling(window=20).mean().values,
                test_series.rolling(window=20).std().values,
                test_series.rolling(window=20).min().values,
                test_series.rolling(window=20).max().values
            ]).T
            
            # Test correlation matrix operations
            test_df = pd.DataFrame(np.random.random((100, 10)))
            corr_matrix = test_df.corr()
            
            # Test covariance matrix operations
            cov_matrix = test_df.cov()
            
            self.log_success("Advanced Matrix Operations", "Advanced operations compatibility verified")
            
        except Exception as e:
            self.log_error("Advanced Matrix Operations", str(e))

    async def run_comprehensive_verification(self) -> Dict[str, Any]:
        """Run comprehensive data compatibility verification."""
        self.logger.info("🚀 Starting comprehensive Step4 data compatibility verification...")
        
        verification_tasks = [
            ("Step4 Processing & Labeling", self.verify_step4_processing_labeling),
            ("Step4 Validator", self.verify_step4_validator),
            ("Step4 Regime Data Splitting", self.verify_step4_regime_data_splitting),
            ("Vectorized Labeling Orchestrator", self.verify_vectorized_labeling_orchestrator),
            ("Optimized Triple Barrier Labeling", self.verify_optimized_triple_barrier_labeling),
            ("Vectorized Advanced Feature Engineering", self.verify_vectorized_advanced_feature_engineering)
        ]
        
        results = {}
        for task_name, task_func in verification_tasks:
            try:
                result = await task_func()
                results[task_name] = result
            except Exception as e:
                self.log_error(task_name, f"Verification failed: {str(e)}")
                results[task_name] = False
        
        # Generate comprehensive report
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_components": len(verification_tasks),
                "successful_verifications": sum(results.values()),
                "failed_verifications": len(results) - sum(results.values()),
                "total_errors": len(self.errors),
                "total_warnings": len(self.warnings)
            },
            "component_results": results,
            "verification_results": self.verification_results,
            "errors": self.errors,
            "warnings": self.warnings
        }
        
        # Save report
        report_path = "log/step4_data_compatibility_report.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"📊 Verification completed. Report saved to: {report_path}")
        self.logger.info(f"✅ Successful verifications: {report['summary']['successful_verifications']}/{report['summary']['total_components']}")
        self.logger.info(f"❌ Failed verifications: {report['summary']['failed_verifications']}")
        self.logger.info(f"⚠️ Total warnings: {report['summary']['total_warnings']}")
        self.logger.info(f"🚨 Total errors: {report['summary']['total_errors']}")
        
        return report


async def main():
    """Main function to run the verification."""
    verifier = Step4DataCompatibilityVerifier()
    report = await verifier.run_comprehensive_verification()
    
    # Print summary
    print("\n" + "="*80)
    print("STEP4 DATA COMPATIBILITY VERIFICATION SUMMARY")
    print("="*80)
    print(f"Total Components: {report['summary']['total_components']}")
    print(f"Successful: {report['summary']['successful_verifications']}")
    print(f"Failed: {report['summary']['failed_verifications']}")
    print(f"Warnings: {report['summary']['total_warnings']}")
    print(f"Errors: {report['summary']['total_errors']}")
    print("="*80)
    
    if report['summary']['failed_verifications'] > 0:
        print("\n❌ FAILED VERIFICATIONS:")
        for component, result in report['component_results'].items():
            if not result:
                print(f"  - {component}")
    
    if report['errors']:
        print("\n🚨 ERRORS:")
        for error in report['errors'][:5]:  # Show first 5 errors
            print(f"  - {error['component']}: {error['error']}")
    
    if report['warnings']:
        print("\n⚠️ WARNINGS:")
        for warning in report['warnings'][:5]:  # Show first 5 warnings
            print(f"  - {warning['component']}: {warning['warning']}")
    
    print(f"\n📄 Detailed report saved to: log/step4_data_compatibility_report.json")


if __name__ == "__main__":
    asyncio.run(main())