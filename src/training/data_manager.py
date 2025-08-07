# src/training/data_manager.py

import os
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import system_logger


class UnifiedDataManager:
    """
    Unified data management system for the training pipeline.
    
    This class creates and manages a single, efficient database with all labels
    and features needed by subsequent training steps, with proper time-based splitting.
    """

    def __init__(self, data_dir: str, symbol: str, exchange: str, lookback_days: int = 730):
        self.data_dir = data_dir
        self.symbol = symbol
        self.exchange = exchange
        self.lookback_days = lookback_days
        self.logger = system_logger.getChild("UnifiedDataManager")
        
        # Database file paths
        self.unified_db_dir = os.path.join(data_dir, "unified_database")
        os.makedirs(self.unified_db_dir, exist_ok=True)
        
        self.database_file = os.path.join(
            self.unified_db_dir, f"{exchange}_{symbol}_unified_dataset.pkl"
        )
        self.metadata_file = os.path.join(
            self.unified_db_dir, f"{exchange}_{symbol}_dataset_metadata.json"
        )
        
        # Split files
        self.train_file = os.path.join(data_dir, f"{exchange}_{symbol}_train_data.pkl")
        self.validation_file = os.path.join(data_dir, f"{exchange}_{symbol}_validation_data.pkl")
        self.test_file = os.path.join(data_dir, f"{exchange}_{symbol}_test_data.pkl")

    def create_unified_database(
        self, 
        labeled_data: pd.DataFrame,
        strategic_signals: pd.Series = None,
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Dict[str, Any]:
        """
        Create a unified database with proper time-based data splitting.
        
        Args:
            labeled_data: Complete dataset with features and labels
            strategic_signals: Strategic signals from analyst models
            train_ratio: Proportion of data for training (default 80%)
            validation_ratio: Proportion of data for validation (default 10%)
            test_ratio: Proportion of data for testing (default 10%)
            
        Returns:
            Dict containing database creation results
        """
        try:
            self.logger.info("🔄 Creating unified database with time-based splits...")
            
            # Validate ratios
            if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-6:
                raise ValueError("Train, validation, and test ratios must sum to 1.0")
            
            # Ensure data is sorted by time
            if not labeled_data.index.is_monotonic_increasing:
                labeled_data = labeled_data.sort_index()
            
            # Apply lookback filtering based on configuration
            labeled_data = self._apply_lookback_filter(labeled_data)
            
            # Create time-based splits
            train_data, validation_data, test_data = self._create_time_based_splits(
                labeled_data, train_ratio, validation_ratio, test_ratio
            )
            
            # Add metadata to each split
            metadata = self._create_metadata(labeled_data, train_data, validation_data, test_data)
            
            # Save unified database
            self._save_unified_database(labeled_data, metadata)
            
            # Save individual splits for easy access
            self._save_data_splits(train_data, validation_data, test_data)
            
            # Save strategic signals if provided
            if strategic_signals is not None:
                self._save_strategic_signals(strategic_signals)
            
            self.logger.info("✅ Unified database created successfully")
            
            return {
                "unified_database_file": self.database_file,
                "metadata_file": self.metadata_file,
                "train_file": self.train_file,
                "validation_file": self.validation_file,
                "test_file": self.test_file,
                "metadata": metadata,
                "status": "SUCCESS"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error creating unified database: {e}")
            raise

    def _apply_lookback_filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply lookback period filtering to the data.
        
        Args:
            data: Input DataFrame with datetime index
            
        Returns:
            Filtered DataFrame
        """
        if self.lookback_days and self.lookback_days > 0:
            self.logger.info(f"📅 Applying lookback filter: {self.lookback_days} days")
            
            # Calculate cutoff date
            end_date = data.index.max()
            start_date = end_date - timedelta(days=self.lookback_days)
            
            # Filter data
            filtered_data = data[data.index >= start_date].copy()
            
            self.logger.info(
                f"📊 Data filtered from {len(data)} to {len(filtered_data)} samples "
                f"(from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})"
            )
            
            return filtered_data
        
        return data

    def _create_time_based_splits(
        self,
        data: pd.DataFrame,
        train_ratio: float,
        validation_ratio: float,
        test_ratio: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create time-based data splits ensuring temporal ordering.
        
        Args:
            data: Complete dataset
            train_ratio: Training data ratio
            validation_ratio: Validation data ratio
            test_ratio: Test data ratio
            
        Returns:
            Tuple of (train_data, validation_data, test_data)
        """
        total_samples = len(data)
        
        # Calculate split indices (time-based, not random)
        train_end_idx = int(total_samples * train_ratio)
        validation_end_idx = int(total_samples * (train_ratio + validation_ratio))
        
        # Create splits maintaining temporal order
        train_data = data.iloc[:train_end_idx].copy()
        validation_data = data.iloc[train_end_idx:validation_end_idx].copy()
        test_data = data.iloc[validation_end_idx:].copy()
        
        # Ensure we have minimum viable datasets
        min_samples = 100
        if len(train_data) < min_samples:
            raise ValueError(f"Training set too small: {len(train_data)} samples (minimum: {min_samples})")
        if len(validation_data) < 50:
            self.logger.warning(f"Validation set small: {len(validation_data)} samples")
        if len(test_data) < 50:
            self.logger.warning(f"Test set small: {len(test_data)} samples")
        
        self.logger.info(
            f"📊 Time-based splits created:\n"
            f"  • Training: {len(train_data)} samples ({len(train_data)/total_samples:.1%})\n"
            f"  • Validation: {len(validation_data)} samples ({len(validation_data)/total_samples:.1%})\n"
            f"  • Test: {len(test_data)} samples ({len(test_data)/total_samples:.1%})"
        )
        
        return train_data, validation_data, test_data

    def _create_metadata(
        self,
        full_data: pd.DataFrame,
        train_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        test_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Create comprehensive metadata for the dataset."""
        
        # Identify different types of columns
        feature_columns = [
            col for col in full_data.columns 
            if not any(label in col.lower() for label in ['label', 'target', 'signal'])
        ]
        label_columns = [
            col for col in full_data.columns 
            if any(label in col.lower() for label in ['label', 'target'])
        ]
        
        metadata = {
            "creation_date": datetime.now().isoformat(),
            "symbol": self.symbol,
            "exchange": self.exchange,
            "lookback_days": self.lookback_days,
            "total_samples": len(full_data),
            "date_range": {
                "start": full_data.index.min().isoformat(),
                "end": full_data.index.max().isoformat()
            },
            "splits": {
                "train": {
                    "samples": len(train_data),
                    "start_date": train_data.index.min().isoformat(),
                    "end_date": train_data.index.max().isoformat(),
                    "ratio": len(train_data) / len(full_data)
                },
                "validation": {
                    "samples": len(validation_data),
                    "start_date": validation_data.index.min().isoformat() if len(validation_data) > 0 else None,
                    "end_date": validation_data.index.max().isoformat() if len(validation_data) > 0 else None,
                    "ratio": len(validation_data) / len(full_data)
                },
                "test": {
                    "samples": len(test_data),
                    "start_date": test_data.index.min().isoformat() if len(test_data) > 0 else None,
                    "end_date": test_data.index.max().isoformat() if len(test_data) > 0 else None,
                    "ratio": len(test_data) / len(full_data)
                }
            },
            "columns": {
                "total": len(full_data.columns),
                "features": feature_columns,
                "labels": label_columns,
                "feature_count": len(feature_columns),
                "label_count": len(label_columns)
            },
            "label_distributions": {}
        }
        
        # Add label distributions for each label column
        for label_col in label_columns:
            if label_col in full_data.columns:
                distribution = full_data[label_col].value_counts().to_dict()
                metadata["label_distributions"][label_col] = {
                    "full_dataset": distribution,
                    "train": train_data[label_col].value_counts().to_dict() if label_col in train_data.columns else {},
                    "validation": validation_data[label_col].value_counts().to_dict() if label_col in validation_data.columns else {},
                    "test": test_data[label_col].value_counts().to_dict() if label_col in test_data.columns else {}
                }
        
        return metadata

    def _save_unified_database(self, data: pd.DataFrame, metadata: Dict[str, Any]) -> None:
        """Save the unified database and metadata."""
        
        # Save main database
        with open(self.database_file, "wb") as f:
            pickle.dump(data, f)
        
        # Save metadata
        import json
        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"💾 Unified database saved to {self.database_file}")
        self.logger.info(f"📋 Metadata saved to {self.metadata_file}")

    def _save_data_splits(
        self,
        train_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        test_data: pd.DataFrame
    ) -> None:
        """Save individual data splits for easy access by subsequent steps."""
        
        # Save training data
        with open(self.train_file, "wb") as f:
            pickle.dump(train_data, f)
        
        # Save validation data
        with open(self.validation_file, "wb") as f:
            pickle.dump(validation_data, f)
        
        # Save test data
        with open(self.test_file, "wb") as f:
            pickle.dump(test_data, f)
        
        self.logger.info(f"💾 Data splits saved:")
        self.logger.info(f"  • Training: {self.train_file}")
        self.logger.info(f"  • Validation: {self.validation_file}")
        self.logger.info(f"  • Test: {self.test_file}")

    def _save_strategic_signals(self, strategic_signals: pd.Series) -> None:
        """Save strategic signals with metadata."""
        signals_file = os.path.join(
            self.unified_db_dir, f"{self.exchange}_{self.symbol}_strategic_signals.pkl"
        )
        
        with open(signals_file, "wb") as f:
            pickle.dump(strategic_signals, f)
        
        self.logger.info(f"💾 Strategic signals saved to {signals_file}")

    def load_data_split(self, split_type: str) -> pd.DataFrame:
        """
        Load a specific data split.
        
        Args:
            split_type: One of 'train', 'validation', 'test', or 'full'
            
        Returns:
            DataFrame for the requested split
        """
        split_files = {
            'train': self.train_file,
            'validation': self.validation_file,
            'test': self.test_file,
            'full': self.database_file
        }
        
        if split_type not in split_files:
            raise ValueError(f"Invalid split_type: {split_type}. Must be one of {list(split_files.keys())}")
        
        file_path = split_files[split_type]
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data split file not found: {file_path}")
        
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        self.logger.info(f"📂 Loaded {split_type} data: {len(data)} samples")
        return data

    def get_metadata(self) -> Dict[str, Any]:
        """Load and return dataset metadata."""
        if not os.path.exists(self.metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")
        
        import json
        with open(self.metadata_file, "r") as f:
            metadata = json.load(f)
        
        return metadata

    def update_data_split(self, split_type: str, updated_data: pd.DataFrame) -> None:
        """
        Update a specific data split (useful for steps like step 8 that modify data).
        
        Args:
            split_type: One of 'train', 'validation', 'test'
            updated_data: Updated DataFrame
        """
        split_files = {
            'train': self.train_file,
            'validation': self.validation_file,
            'test': self.test_file
        }
        
        if split_type not in split_files:
            raise ValueError(f"Invalid split_type: {split_type}. Must be one of {list(split_files.keys())}")
        
        file_path = split_files[split_type]
        
        with open(file_path, "wb") as f:
            pickle.dump(updated_data, f)
        
        self.logger.info(f"💾 Updated {split_type} data: {len(updated_data)} samples")

    def get_features_and_labels(
        self, 
        split_type: str, 
        label_column: str = 'tactician_label'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get features and labels for a specific split.
        
        Args:
            split_type: Data split to load
            label_column: Name of the label column
            
        Returns:
            Tuple of (features_df, labels_series)
        """
        data = self.load_data_split(split_type)
        
        if label_column not in data.columns:
            raise ValueError(f"Label column '{label_column}' not found in data")
        
        # Separate features and labels
        feature_columns = [col for col in data.columns if col != label_column]
        
        X = data[feature_columns]
        y = data[label_column]
        
        return X, y

    def validate_database_integrity(self) -> Dict[str, Any]:
        """
        Validate the integrity of the unified database and splits.
        
        Returns:
            Dict containing validation results
        """
        try:
            validation_results = {
                "status": "SUCCESS",
                "issues": [],
                "warnings": []
            }
            
            # Check if all files exist
            required_files = [
                self.database_file,
                self.metadata_file,
                self.train_file,
                self.validation_file,
                self.test_file
            ]
            
            for file_path in required_files:
                if not os.path.exists(file_path):
                    validation_results["issues"].append(f"Missing file: {file_path}")
            
            if validation_results["issues"]:
                validation_results["status"] = "FAILED"
                return validation_results
            
            # Load and validate data
            metadata = self.get_metadata()
            full_data = self.load_data_split('full')
            train_data = self.load_data_split('train')
            validation_data = self.load_data_split('validation')
            test_data = self.load_data_split('test')
            
            # Check data consistency
            total_splits = len(train_data) + len(validation_data) + len(test_data)
            if total_splits != len(full_data):
                validation_results["issues"].append(
                    f"Split sizes don't match full dataset: {total_splits} vs {len(full_data)}"
                )
            
            # Check temporal ordering
            if not self._check_temporal_ordering(train_data, validation_data, test_data):
                validation_results["issues"].append("Temporal ordering violated in splits")
            
            # Check for sufficient data in each split
            if len(train_data) < 100:
                validation_results["warnings"].append(f"Small training set: {len(train_data)} samples")
            if len(validation_data) < 50:
                validation_results["warnings"].append(f"Small validation set: {len(validation_data)} samples")
            
            # Update status based on issues
            if validation_results["issues"]:
                validation_results["status"] = "FAILED"
            elif validation_results["warnings"]:
                validation_results["status"] = "WARNING"
            
            return validation_results
            
        except Exception as e:
            return {
                "status": "FAILED",
                "issues": [f"Validation error: {str(e)}"],
                "warnings": []
            }

    def _check_temporal_ordering(
        self, 
        train_data: pd.DataFrame, 
        validation_data: pd.DataFrame, 
        test_data: pd.DataFrame
    ) -> bool:
        """Check if the temporal ordering is maintained across splits."""
        try:
            if len(validation_data) > 0 and len(test_data) > 0:
                return (
                    train_data.index.max() <= validation_data.index.min() and
                    validation_data.index.max() <= test_data.index.min()
                )
            elif len(validation_data) > 0:
                return train_data.index.max() <= validation_data.index.min()
            elif len(test_data) > 0:
                return train_data.index.max() <= test_data.index.min()
            return True
        except Exception:
            return False