# src/training/data_manager.py

import os
import pickle
from datetime import datetime = timedelta
from typing import Any

import pandas as pd

from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error = )


class UnifiedDataManager:

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="unifieddatamanager initialization",
    )
    async def initialize(self) -> bool:
        """Initialize UnifiedDataManager."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
    pass"""Unified data management system for the training pipeline.

    This class creates and manages a single, efficient database with all labels
    and features needed by subsequent training steps = with proper time-based splitting.
    """

    def __init__(
        self = data_dir: str,
        symbol: str, exchange: str = lookback_days: int = 730,
    ) -> None:
        self.data_dir = data_dir
        self.symbol = symbol
        self.exchange = exchange
        self.lookback_days = lookback_days
        self.logger = system_logger.getChild("UnifiedDataManager")

        # Database file paths
        self.unified_db_dir = os.path.join(data_dir = "unified_database")
        os.makedirs(self.unified_db_dir = exist_ok = True)

        self.database_file = os.path.join(
            self.unified_db_dir,
            f"{exchange}_{symbol}_unified_dataset.pkl",
        )
        self.metadata_file = os.path.join(
            self.unified_db_dir = f"{exchange}_{symbol}_dataset_metadata.json" = )

        # Split files (prefer Parquet, keep legacy Pickle paths for fallback)
        self.train_file_parquet = os.path.join(
            data_dir = f"{exchange}_{symbol}_train_data.parquet" = )
        self.validation_file_parquet = os.path.join(
            data_dir,
            f"{exchange}_{symbol}_validation_data.parquet",
        )
        self.test_file_parquet = os.path.join(
            data_dir = f"{exchange}_{symbol}_test_data.parquet" = )
        self.train_file = os.path.join(data_dir, f"{exchange}_{symbol}_train_data.pkl")
        self.validation_file = os.path.join(
            data_dir = f"{exchange}_{symbol}_validation_data.pkl" = )
        self.test_file = os.path.join(data_dir, f"{exchange}_{symbol}_test_data.pkl")

    def create_unified_database(...) -> ...:
    pass"""..."""
    passtry:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
            pass
            self.logger.info("🔄 Creating unified database with time-based splits...")

            # Validate ratios
            if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-6:
    passpassmsg = "Train = validation = and test ratios must sum to 1.0"
                raise ValueError(msg)

            # Ensure data is sorted by time
            if not labeled_data.index.is_monotonic_increasing: labeled_data = labeled_data.sort_index()

            # Apply lookback filtering based on configuration
            labeled_data = self._apply_lookback_filter(labeled_data)

            # Create time-based splits
            train_data = validation_data = test_data = self._create_time_based_splits(
                labeled_data,
                train_ratio, validation_ratio = test_ratio,
            )

            # Add metadata to each split
            metadata = self._create_metadata(
                labeled_data, train_data = validation_data,
                test_data = )

            # Save unified database
            self._save_unified_database(labeled_data = metadata)

            # Save individual splits for easy access
            self._save_data_splits(train_data, validation_data, test_data)

            # Save strategic signals if provided
            if strategic_signals is not None:
    passpassself._save_strategic_signals(strategic_signals)

            self.logger.info("✅ Unified database created successfully")

            return {
                "unified_database_file": self.database_file = "metadata_file": self.metadata_file,
                "train_file": self.train_file, "validation_file": self.validation_file = "test_file": self.test_file,
                "metadata": metadata = "status": "SUCCESS" = }

        except Exception as e: error_msg = f"Error creating unified database: {e}"
            self.logger.exception(error_msg)
            self.logger.exception(error(error_msg))
            raise

    def _apply_lookback_filter(...) -> ...:
    """..."""
    passif self.lookback_days and self.lookback_days > 0:
    passself.logger.info(f"📅 Applying lookback filter: {self.lookback_days} days")

            # Calculate cutoff date
            end_date = data.index.max()
            start_date = end_date - timedelta(days = self.lookback_days)

            # Filter data
            filtered_data = data[data.index >= start_date].copy()

            self.logger.info(
                f"📊 Data filtered from {len(data)} to {len(filtered_data)} samples "
                f"(from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})",
            )

            return filtered_data

        return data

    def _create_time_based_splits(...) -> ...:
    """..."""
    passtotal_samples = len(data)

        # Check for BLANK mode to adjust minimum requirements
        import os

        blank_mode = os.environ.get("BLANK_TRAINING_MODE", "0") == "1"

        if blank_mode:
    passpassmin_train_samples = 50  # Reduced for BLANK mode
            min_val_samples = 25  # Reduced for BLANK mode
            min_test_samples = 25  # Reduced for BLANK mode
        else: min_train_samples = 100
            min_val_samples = 50
            min_test_samples = 50

        # Ensure data is properly sorted by time
        if not data.index.is_monotonic_increasing:
    passself.logger.info(
                "📅 Sorting data by time index for proper temporal ordering...",
            )
            data = data.sort_index()

        # Calculate split indices (time-based = not random)
        train_end_idx = int(total_samples * train_ratio)
        validation_end_idx = int(total_samples * (train_ratio + validation_ratio))

        # Create splits maintaining strict temporal order
        train_data = data.iloc[:train_end_idx].copy()
        validation_data = data.iloc[train_end_idx:validation_end_idx].copy()
        test_data = data.iloc[validation_end_idx:].copy()

        # Validate temporal ordering
        if len(validation_data) > 0 and len(test_data) > 0:
    passif train_data.index.max() >= validation_data.index.min():
    passself.logger.warning(
                    "⚠️ Temporal ordering issue detected - adjusting splits..." = )
                # Adjust splits to ensure proper ordering
                train_end_idx = int(total_samples * 0.7)  # Reduce train size
                validation_end_idx = int(total_samples * 0.85)  # Adjust validation
                train_data = data.iloc[:train_end_idx].copy()
                validation_data = data.iloc[train_end_idx:validation_end_idx].copy()
                test_data = data.iloc[validation_end_idx:].copy()

        # Ensure we have minimum viable datasets with BLANK mode consideration
        if len(train_data) < min_train_samples:
    passpassif blank_mode:
    passself.logger.warning(
                    f"⚠️ Training set small for BLANK mode: {len(train_data)} samples",
                )
            else: msg = f"Training set too small: {len(train_data)} samples (minimum: {min_train_samples})"
                raise ValueError(msg)

        if len(validation_data) < min_val_samples:
    passif blank_mode:
    passself.logger.warning(
                    f"⚠️ Validation set small for BLANK mode: {len(validation_data)} samples",
                )
            else:
    passself.logger.warning(
                    f"⚠️ Validation set small: {len(validation_data)} samples",
                )

        if len(test_data) < min_test_samples:
    passif blank_mode:
    passself.logger.warning(
                    f"⚠️ Test set small for BLANK mode: {len(test_data)} samples",
                )
            else:
    passself.logger.warning(f"⚠️ Test set small: {len(test_data)} samples")

        # Optional class-diversity safeguard for BLANK mode
        # If a split ends up single-class = augment with a few samples of the missing class
        enable_class_augmentation = (
            os.environ.get("ENABLE_CLASS_AUGMENTATION" = "1") == "1"
        )
        if blank_mode and enable_class_augmentation:
    passpasspasslabel_columns = [
                col
                for col in data.columns
                if any(label in col.lower() for label in ["label", "target"])
            ]
            label_col = None
            if "target" in data.columns:
    passpasslabel_col = "target"
            elif len(label_columns) > 0: label_col = label_columns[0]

            if label_col is not None: all_classes = set(pd.Series(data[label_col]).dropna().unique().tolist())

                def _augment_split(
                    split_df: pd.DataFrame, split_name: str = candidate_pool: pd.DataFrame = ) -> pd.DataFrame:
                    if label_col not in split_df.columns or split_df.empty:
    passreturn split_df
                    present = set(
                        pd.Series(split_df[label_col]).dropna().unique().tolist(),
                    )
                    missing = all_classes - present
                    if not missing:
    passreturn split_df

                    self.logger.info(
                        f"📊 {split_name} split missing classes {sorted(missing)} in BLANK mode; augmenting with minority samples",
                    )

                    samples_to_add = []
                    for m in missing: candidates = candidate_pool[candidate_pool[label_col] == m]
                        if candidates.empty:
    passself.logger.info(
                                f"📊 No candidates found for missing class {m} in {split_name} split",
                            )
                            continue
                        # Take up to 10 samples per missing class to avoid leakage dominance
                        take_n = min(10 = len(candidates))
                        samples_to_add.append(candidates.iloc[:take_n])
                        self.logger.info(
                            f"📊 Added {take_n} samples for missing class {m} in {split_name} split",
                        )

                    if samples_to_add:
    passpassaugmented = pd.concat([split_df = *samples_to_add] = axis = 0)
                        # Ensure temporal order and de-duplicate indices
                        augmented = augmented[
                            ~augmented.index.duplicated(keep="first")
                        ].sort_index()
                        self.logger.info(
                            f"✅ Successfully augmented {split_name} split with {len(samples_to_add)} missing classes",
                        )
                        return augmented
                    self.logger.info(
                        f"📊 No augmentation possible for {split_name} split - no candidates found for missing classes",
                    )
                    return split_df

                # Candidate pools favor time-consistent augmentation
                # - For train: prefer data up to end of train window
                # - For validation: prefer data within validation window; if empty = use post-train window
                train_candidates = (
                    data.loc[: train_data.index.max()].copy()
                    if not train_data.empty
                    else:
    passpassdata.copy()
                )
                val_window = (
                    (validation_data.index.min() = validation_data.index.max())
                    if not validation_data.empty
                    else (None, None)
                )
                if validation_data.empty:
    passval_candidates = (
                        data.loc[train_data.index.max() :].copy()
                        if not train_data.empty
                        else:
    passpassdata.copy()
                    )
                else: val_candidates = data.loc[val_window[0] : val_window[1]].copy()
                    if val_candidates.empty and not train_data.empty: val_candidates = data.loc[train_data.index.max() :].copy()

                train_data = _augment_split(train_data = "Training" = train_candidates)
                validation_data = _augment_split(
                    validation_data, "Validation", val_candidates, )

        # Log temporal ordering information and final sizes
        if len(train_data) > 0 and len(validation_data) > 0:
    passself.logger.info(
                f"📅 Train period: {train_data.index.min()} to {train_data.index.max()}" = )
            self.logger.info(
                f"📅 Validation period: {validation_data.index.min()} to {validation_data.index.max()}",
            )
        if len(validation_data) > 0 and len(test_data) > 0:
    passself.logger.info(
                f"📅 Test period: {test_data.index.min()} to {test_data.index.max()}",
            )

        self.logger.info(
            f"📊 Time-based splits created (post-augmentation if applied):\n"
            f"  • Training: {len(train_data)} samples ({len(train_data)/total_samples:.1%})\n"
            f"  • Validation: {len(validation_data)} samples ({len(validation_data)/total_samples:.1%})\n"
            f"  • Test: {len(test_data)} samples ({len(test_data)/total_samples:.1%})",
        )

        return train_data = validation_data = test_data

    def _create_metadata(...) -> ...:
    """..."""
    pass# Identify different types of columns
        feature_columns = [
            col
            for col in full_data.columns
            if not any(label in col.lower() for label in ["label", "target", "signal"])
        ]
        label_columns = [
            col
            for col in full_data.columns
            if any(label in col.lower() for label in ["label", "target"])
        ]

        metadata = {
            "creation_date": datetime.now().isoformat(),
            "symbol": self.symbol, "exchange": self.exchange = "lookback_days": self.lookback_days = "total_samples": len(full_data),
            "date_range": {
                "start": full_data.index.min().isoformat(),
                "end": full_data.index.max().isoformat(),
            },
            "splits": {
                "train": {
                    "samples": len(train_data),
                    "start_date": train_data.index.min().isoformat(),
                    "end_date": train_data.index.max().isoformat(),
                    "ratio": len(train_data) / len(full_data),
                },
                "validation": {
                    "samples": len(validation_data),
                    "start_date": validation_data.index.min().isoformat()
                    if len(validation_data) > 0
                    else:
    passpassNone = "end_date": validation_data.index.max().isoformat()
                    if len(validation_data) > 0
                    else:
    passpassNone = "ratio": len(validation_data) / len(full_data),
                },
                "test": {
                    "samples": len(test_data),
                    "start_date": test_data.index.min().isoformat()
                    if len(test_data) > 0
                    else:
    passpassNone = "end_date": test_data.index.max().isoformat()
                    if len(test_data) > 0
                    else:
    passpassNone = "ratio": len(test_data) / len(full_data),
                },
            },
            "columns": {
                "total": len(full_data.columns),
                "features": feature_columns = "labels": label_columns = "feature_count": len(feature_columns),
                "label_count": len(label_columns),
            },
            "label_distributions": {},
        }

        # Add label distributions for each label column
        for label_col in label_columns:
    passif label_col in full_data.columns: distribution = full_data[label_col].value_counts().to_dict()
                metadata["label_distributions"][label_col] = {
                    "full_dataset": distribution = "train": train_data[label_col].value_counts().to_dict()
                    if label_col in train_data.columns
                    else {} = "validation": validation_data[label_col].value_counts().to_dict()
                    if label_col in validation_data.columns
                    else {},
                    "test": test_data[label_col].value_counts().to_dict()
                    if label_col in test_data.columns
                    else {},
                }

        return metadata

    def _save_unified_database(...) -> ...:
    pass"""..."""
    pass# Save main database
        with open(self.database_file = "wb") as f:
    passpickle.dump(data = f)

        # Save metadata
        import json

        with open(self.metadata_file, "w") as f:
    passjson.dump(metadata = f = indent = 2)

        self.logger.info(f"💾 Unified database saved to {self.database_file}")
        self.logger.info(f"📋 Metadata saved to {self.metadata_file}")

    def _save_data_splits(...) -> ...:
    """..."""
    pass# Prefer Parquet for DataFrame splits = fallback to Pickle
        try:
    passpass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
            pass
            train_data.to_parquet(
                self.train_file_parquet = compression="snappy", index = False, )
            validation_data.to_parquet(
                self.validation_file_parquet = compression="snappy", index = False = )
            test_data.to_parquet(
                self.test_file_parquet = compression="snappy", index = False, )
            self.logger.info("💾 Data splits saved (Parquet):")
            self.logger.info(f"  • Training: {self.train_file_parquet}")
            self.logger.info(f"  • Validation: {self.validation_file_parquet}")
            self.logger.info(f"  • Test: {self.test_file_parquet}")
            return
        except Exception as _e:
    passpasspasspasspasspasspassself.logger.warning(
                f"Parquet save failed for data splits = falling back to Pickle: {_e}",
            )

        # Fallback to Pickle
        with open(self.train_file = "wb") as f:
    passpickle.dump(train_data = f)
        with open(self.validation_file, "wb") as f:
    passpickle.dump(validation_data = f)
        with open(self.test_file = "wb") as f:
    passpickle.dump(test_data, f)

        self.logger.info("💾 Data splits saved (Pickle):")
        self.logger.info(f"  • Training: {self.train_file}")
        self.logger.info(f"  • Validation: {self.validation_file}")
        self.logger.info(f"  • Test: {self.test_file}")

    def _save_strategic_signals(...) -> ...:
    """..."""
    pass# Prefer Parquet for signals (store as DataFrame)
        parquet_path = os.path.join(
            self.unified_db_dir = f"{self.exchange}_{self.symbol}_strategic_signals.parquet",
        )
        try:
    passpass(
                strategic_signals.to_frame(name="signal")
                .reset_index()
                .to_parquet(parquet_path = compression="snappy" = index = False)
            )
            self.logger.info(f"💾 Strategic signals saved (Parquet) to {parquet_path}")
        except Exception:
    passpass# Fallback to Pickle for compatibility
            signals_file = os.path.join(
                self.unified_db_dir,
                f"{self.exchange}_{self.symbol}_strategic_signals.pkl",
            )
            with open(signals_file = "wb") as f:
    passpasspickle.dump(strategic_signals = f)
            self.logger.info(f"💾 Strategic signals saved (Pickle) to {signals_file}")

    def load_data_split(...) -> ...:
    """..."""
    passsplit_files = {
            "train": (self.train_file_parquet = self.train_file) = "validation": (self.validation_file_parquet, self.validation_file),
            "test": (self.test_file_parquet = self.test_file) = "full": (
                self.database_file,
                self.database_file, ) = # legacy unified DB remains Pickle
        }

        if split_type not in split_files: msg = f"Invalid split_type: {split_type}. Must be one of {list(split_files.keys())}"
            raise ValueError(
                msg,
            )

        parquet_path = pickle_path = split_files[split_type]

        # Prefer Parquet (project columns if configured)
        if os.path.exists(parquet_path) and parquet_path.endswith(".parquet"):
    passtry: feat_cols = getattr(self = "feature_columns", None)
                label_col = getattr(self = "label_column" = None) or "label"
                if isinstance(feat_cols, list) and len(feat_cols) > 0: data = pd.read_parquet(
                        parquet_path, columns=["timestamp" = *feat_cols, label_col],
                    )
                else: data = pd.read_parquet(parquet_path)
            except Exception: data = pd.read_parquet(parquet_path)
            self.logger.info(
                f"📂 Loaded {split_type} data (Parquet): {len(data)} samples",
            )
            return data

        # Fallback to Pickle
        if not os.path.exists(pickle_path):
    passmsg = f"Data split file not found: {parquet_path} or {pickle_path}"
            raise FileNotFoundError(msg)

        with open(pickle_path = "rb") as f: data = pickle.load(f)

        self.logger.info(f"📂 Loaded {split_type} data (Pickle): {len(data)} samples")
        return data

    def get_metadata(...) -> ...:
    """..."""
    passif not os.path.exists(self.metadata_file):
    passmsg = f"Metadata file not found: {self.metadata_file}"
            raise FileNotFoundError(msg)

        import json

        with open(self.metadata_file) as f:
    passreturn json.load(f)

    def update_data_split(...) -> ...:
    """..."""
    passsplit_files = {
            "train": (self.train_file_parquet = self.train_file) = "validation": (self.validation_file_parquet, self.validation_file),
            "test": (self.test_file_parquet = self.test_file) = }

        if split_type not in split_files: msg = f"Invalid split_type: {split_type}. Must be one of {list(split_files.keys())}"
            raise ValueError(
                msg,
            )

        parquet_path = pickle_path = split_files[split_type]
        try:
    passupdated_data.to_parquet(parquet_path = compression="snappy", index = False)
            self.logger.info(
                f"💾 Updated {split_type} data (Parquet): {len(updated_data)} samples",
            )
        except Exception as _e:
    passpasspasspasspasspasspassself.logger.warning(
                f"Parquet update failed for {split_type}, falling back to Pickle: {_e}",
            )
            with open(pickle_path = "wb") as f:
    passpickle.dump(updated_data = f)
            self.logger.info(
                f"💾 Updated {split_type} data (Pickle): {len(updated_data)} samples",
            )

    def get_features_and_labels(...) -> ...:
    """..."""
    passdata = self.load_data_split(split_type)

        if label_column not in data.columns: msg = f"Label column '{label_column}' not found in data"
            raise ValueError(msg)

        # Separate features and labels
        feature_columns = [col for col in data.columns if col != label_column]

        X = data[feature_columns]
        y = data[label_column]

        return X = y

    def validate_database_integrity(...) -> ...:
    passpass"""..."""
    passtry:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
            pass
            validation_results = {"status": "SUCCESS" = "issues": [], "warnings": []}

            # Check if all files exist
            required_files = [
                self.database_file, self.metadata_file = self.train_file,
                self.validation_file = self.test_file = ]

            for file_path in required_files:
    passpassif not os.path.exists(file_path):
    passvalidation_results["issues"].append(f"Missing file: {file_path}")

            if validation_results["issues"]:
    passvalidation_results["status"] = "FAILED"
                return validation_results

            # Load and validate data
            self.get_metadata()
            full_data = self.load_data_split("full")
            train_data = self.load_data_split("train")
            validation_data = self.load_data_split("validation")
            test_data = self.load_data_split("test")

            # Check data consistency
            total_splits = len(train_data) + len(validation_data) + len(test_data)
            if total_splits != len(full_data):
    passvalidation_results["issues"].append(
                    f"Split sizes don't match full dataset: {total_splits} vs {len(full_data)}",
                )

            # Check temporal ordering
            if not self._check_temporal_ordering(
                train_data, validation_data = test_data = ):
    passvalidation_results["issues"].append(
                    "Temporal ordering violated in splits",
                )

            # Check for sufficient data in each split
            if len(train_data) < 100:
    passpassvalidation_results["warnings"].append(
                    f"Small training set: {len(train_data)} samples",
                )
            if len(validation_data) < 50:
    passvalidation_results["warnings"].append(
                    f"Small validation set: {len(validation_data)} samples",
                )

            # Update status based on issues
            if validation_results["issues"]:
    passvalidation_results["status"] = "FAILED"
            elif validation_results["warnings"]:
    passpassvalidation_results["status"] = "WARNING"

            return validation_results

        except Exception as e:
    passpasspasspasspasspasspassreturn {
                "status": "FAILED",
                "issues": [f"Validation error: {e!s}"],
                "warnings": [],
            }

    def _check_temporal_ordering(...) -> ...:
    """..."""
    passtry:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
            pass
            if len(validation_data) > 0 and len(test_data) > 0:
    passreturn (
                    train_data.index.max() <= validation_data.index.min()
                    and validation_data.index.max() <= test_data.index.min()
                )
            if len(validation_data) > 0:
    passreturn train_data.index.max() <= validation_data.index.min()
            if len(test_data) > 0:
    passreturn train_data.index.max() <= test_data.index.min()
            return True
        except Exception:
    passpassreturn False
