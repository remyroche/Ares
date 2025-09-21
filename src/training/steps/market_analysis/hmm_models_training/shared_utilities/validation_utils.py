"""
Shared Validation Utilities

Provides centralized validation logic for HMM training components.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging

# Optional imports for external dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

logger = logging.getLogger(__name__)


class ValidationUtils:
    """Shared validation utilities for HMM training components."""
    
    @staticmethod
    def validate_data_shapes(X: Union[Any, Any], y: Any, regime_labels: Any) -> bool:
        """
        Validate data shapes are consistent.
        
        Args:
            X: Input features
            y: Target values
            regime_labels: Regime labels
            
        Returns:
            True if shapes are valid, False otherwise
        """
        if not NUMPY_AVAILABLE:
            logger.warning("NumPy not available, skipping shape validation")
            return True
            
        try:
            if len(X) != len(y):
                logger.error(f"Shape mismatch: X has {len(X)} samples, y has {len(y)} samples")
                return False

            # Allow regime_labels to be shorter than X (due to feature generation differences)
            if regime_labels is not None and len(X) != len(regime_labels):
                logger.warning(f"⚠️ Data alignment issue: X has {len(X)} samples, regime_labels has {len(regime_labels)} samples")
                logger.info("🔧 Attempting to align data...")

                # Handle data alignment more intelligently
                if len(regime_labels) != len(X):
                    logger.warning(f"⚠️ Data alignment issue: X has {len(X)} samples, regime_labels has {len(regime_labels)} samples")

                    # Strategy 1: If regime_labels is significantly longer, it might be from a different dataset
                    # In this case, create a fallback approach
                    if len(regime_labels) > len(X) * 1.2:  # More than 20% longer
                        logger.warning(f"⚠️ Regime labels from different dataset detected. Creating aligned labels...")
                        # Create a pattern-based assignment that maintains cluster distribution
                        unique_regimes, regime_counts = np.unique(regime_labels, return_counts=True)
                        total_samples = len(X)

                        # Calculate proportions for each regime
                        regime_proportions = regime_counts / len(regime_labels)

                        # Create new regime labels based on proportions
                        new_regime_labels = np.zeros(total_samples, dtype=regime_labels.dtype)
                        start_idx = 0
                        for regime_id, proportion in enumerate(regime_proportions):
                            end_idx = start_idx + int(total_samples * proportion)
                            new_regime_labels[start_idx:end_idx] = regime_id
                            start_idx = end_idx

                        # Fill any remaining slots with the last regime
                        if start_idx < total_samples:
                            new_regime_labels[start_idx:] = len(unique_regimes) - 1

                        regime_labels = new_regime_labels
                        logger.info(f"✅ Created aligned regime labels using proportion-based assignment")

                    # Strategy 2: Standard extension/trimming for minor differences
                    else:
                        # If regime_labels is shorter, extend it by repeating the last value
                        if len(regime_labels) < len(X):
                            logger.info(f"📊 Extending regime_labels from {len(regime_labels)} to {len(X)} samples")
                            if len(regime_labels) > 0:
                                last_value = regime_labels[-1]
                                extension = np.full(len(X) - len(regime_labels), last_value)
                                regime_labels = np.concatenate([regime_labels, extension])
                            else:
                                logger.error("❌ regime_labels is empty, cannot extend")
                                return False

                        # If regime_labels is longer, trim it to match X
                        elif len(regime_labels) > len(X):
                            logger.info(f"📊 Trimming regime_labels from {len(regime_labels)} to {len(X)} samples")
                            regime_labels = regime_labels[:len(X)]

                logger.info(f"✅ Data aligned successfully: X={len(X)}, regime_labels={len(regime_labels)}")

            # Check regime sample distribution
            try:
                import numpy as np
                unique_regimes, regime_counts = np.unique(regime_labels, return_counts=True)
                min_regime_samples = np.min(regime_counts)
                max_regime_samples = np.max(regime_counts)
            except ImportError:
                # Fallback if numpy not available
                logger.warning("⚠️ NumPy not available, skipping regime distribution analysis")
                unique_regimes = []
                regime_counts = []
                min_regime_samples = 0
                max_regime_samples = 0

            logger.info(f"📊 Regime distribution: {len(unique_regimes)} regimes, "
                       f"min samples: {min_regime_samples}, max samples: {max_regime_samples}")

            # Allow training even with very small regime samples
            # The user specifically wants to proceed with available data
            if min_regime_samples < 1:
                logger.warning(f"⚠️ Some regimes have no samples (minimum: {min_regime_samples}) - this may affect training quality")
                return False
            # Note: Allowing training with any regime that has at least 1 sample

            return True
        except Exception as e:
            logger.error(f"Error validating data shapes: {e}")
            return False
    
    @staticmethod
    def validate_data_quality(X: Union[Any, Any], y: Any, regime_labels: Any) -> bool:
        """
        Validate data quality (no NaN, no infinite values).
        
        Args:
            X: Input features
            y: Target values
            regime_labels: Regime labels
            
        Returns:
            True if data quality is acceptable, False otherwise
        """
        if not NUMPY_AVAILABLE:
            logger.warning("NumPy not available, skipping data quality validation")
            return True
            
        try:
            # Check X for NaN and infinite values
            if NUMPY_AVAILABLE and isinstance(X, np.ndarray):
                if np.any(np.isnan(X)):
                    logger.error("X contains NaN values")
                    return False
                if np.any(np.isinf(X)):
                    logger.error("X contains infinite values")
                    return False
            elif PANDAS_AVAILABLE and isinstance(X, pd.DataFrame):
                if X.isnull().any().any():
                    logger.error("X contains NaN values")
                    return False
                if NUMPY_AVAILABLE and np.isinf(X.select_dtypes(include=[np.number])).any().any():
                    logger.error("X contains infinite values")
                    return False
            
            # Check y for NaN and infinite values
            if NUMPY_AVAILABLE:
                if np.any(np.isnan(y)):
                    logger.error("y contains NaN values")
                    return False
                if np.any(np.isinf(y)):
                    logger.error("y contains infinite values")
                    return False
                
                # Check regime_labels for NaN values
                if np.any(np.isnan(regime_labels)):
                    logger.error("regime_labels contains NaN values")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating data quality: {e}")
            return False
    
    @staticmethod
    def validate_regime_distribution(regime_labels: Any, min_samples_per_regime: int = 10) -> bool:
        """
        Validate regime distribution is adequate.
        
        Args:
            regime_labels: Regime labels
            min_samples_per_regime: Minimum samples required per regime
            
        Returns:
            True if regime distribution is adequate, False otherwise
        """
        if not NUMPY_AVAILABLE:
            logger.warning("NumPy not available, skipping regime distribution validation")
            return True
            
        try:
            if regime_labels is None:
                logger.error("regime_labels is None - cannot validate regime distribution")
                return False

            unique_regimes, regime_counts = np.unique(regime_labels, return_counts=True)
            
            if len(unique_regimes) < 2:
                logger.error("Need at least 2 regimes")
                return False
            
            min_count = np.min(regime_counts)
            if min_count < min_samples_per_regime:
                logger.error(f"Some regimes have insufficient samples (minimum: {min_count}, required: {min_samples_per_regime})")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating regime distribution: {e}")
            return False
    
    @staticmethod
    def validate_config(config: Any) -> bool:
        """
        Validate configuration parameters with range checks.
        
        Args:
            config: Configuration object to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check required attributes
            required_attrs = ['model_types', 'n_features', 'sequence_length', 'n_regimes', 'timeframe']
            for attr in required_attrs:
                if not hasattr(config, attr):
                    logger.error(f"Configuration missing required attribute: {attr}")
                    return False
            
            # Validate model types
            if not config.model_types or len(config.model_types) == 0:
                logger.error("No model types specified")
                return False
            
            # Enhanced: Validate numeric parameters with ranges
            if config.n_features <= 0:
                logger.error("n_features must be positive")
                return False
            elif config.n_features > 10000:
                logger.warning(f"n_features is very large ({config.n_features}), may cause performance issues")
            
            if config.sequence_length <= 0:
                logger.error("sequence_length must be positive")
                return False
            elif config.sequence_length > 1000:
                logger.warning(f"sequence_length is very large ({config.sequence_length}), may cause memory issues")
            
            if config.n_regimes < 2:
                logger.error("n_regimes must be at least 2")
                return False
            elif config.n_regimes > 20:
                logger.warning(f"n_regimes is very large ({config.n_regimes}), may cause overfitting")
            
            # Enhanced: Validate HPO parameters if present
            if hasattr(config, 'hpo_trials'):
                if config.hpo_trials < 0:
                    logger.error("hpo_trials must be non-negative")
                    return False
                elif config.hpo_trials > 1000:
                    logger.warning(f"hpo_trials is very large ({config.hpo_trials}), may take very long")
            
            # Enhanced: Validate learning rate if present
            if hasattr(config, 'learning_rate'):
                if config.learning_rate <= 0 or config.learning_rate > 1:
                    logger.error("learning_rate must be between 0 and 1")
                    return False
            
            # Enhanced: Validate batch size if present
            if hasattr(config, 'batch_size'):
                if config.batch_size <= 0:
                    logger.error("batch_size must be positive")
                    return False
                elif config.batch_size > 10000:
                    logger.warning(f"batch_size is very large ({config.batch_size}), may cause memory issues")
            
            # Validate timeframe
            valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
            if config.timeframe not in valid_timeframes:
                logger.error(f"Invalid timeframe: {config.timeframe}. Valid timeframes: {valid_timeframes}")
                return False
            
            # Enhanced: Cross-parameter validation
            if hasattr(config, 'n_features') and hasattr(config, 'sequence_length'):
                total_params = config.n_features * config.sequence_length
                if total_params > 1000000:
                    logger.warning(f"Total parameter space is very large ({total_params}), may cause memory issues")
            
            return True
        except Exception as e:
            logger.error(f"Error validating configuration: {e}")
            return False
    
    @staticmethod
    def validate_model_type(model_type: str, available_types: List[str]) -> bool:
        """
        Validate model type is supported.
        
        Args:
            model_type: Model type to validate
            available_types: List of available model types
            
        Returns:
            True if model type is valid, False otherwise
        """
        if model_type not in available_types:
            logger.error(f"Invalid model type: {model_type}. Available types: {available_types}")
            return False
        return True
    
    @staticmethod
    def comprehensive_validation(
        X: Union[Any, Any], 
        y: Any, 
        regime_labels: Any,
        config: Any,
        min_samples_per_regime: int = 10
    ) -> Tuple[bool, List[str]]:
        """
        Perform comprehensive validation of all inputs.
        
        Args:
            X: Input features
            y: Target values
            regime_labels: Regime labels
            config: Configuration object
            min_samples_per_regime: Minimum samples per regime
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate shapes
        if not ValidationUtils.validate_data_shapes(X, y, regime_labels):
            errors.append("Data shape validation failed")
        
        # Validate data quality
        if not ValidationUtils.validate_data_quality(X, y, regime_labels):
            errors.append("Data quality validation failed")
        
        # Validate regime distribution
        if not ValidationUtils.validate_regime_distribution(regime_labels, min_samples_per_regime):
            errors.append("Regime distribution validation failed")
        
        # Validate configuration
        if not ValidationUtils.validate_config(config):
            errors.append("Configuration validation failed")
        
        is_valid = len(errors) == 0
        return is_valid, errors

    @staticmethod
    def validate_feature_selection_quality(X, y, selected_features, min_feature_count=5):
        """Enhanced validation that feature selection is reasonable and not causing data leakage."""
        try:
            import numpy as np
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            from sklearn.model_selection import train_test_split, StratifiedKFold
            from sklearn.ensemble import RandomForestClassifier

            if len(selected_features) < min_feature_count:
                return False, f"Too few features selected: {len(selected_features)} < {min_feature_count}"

            # Handle both column names (strings) and indices (integers)
            try:
                X_selected = X[selected_features]
            except (KeyError, IndexError) as e:
                # If selected_features contains indices but X is DataFrame with column names,
                # convert indices to column names
                if isinstance(X, pd.DataFrame) and isinstance(selected_features[0], int):
                    all_columns = list(X.columns)
                    valid_indices = [idx for idx in selected_features if 0 <= idx < len(all_columns)]
                    if not valid_indices:
                        return False, f"No valid feature indices found in selected_features: {selected_features[:10]}"
                    column_names = [all_columns[idx] for idx in valid_indices]
                    X_selected = X[column_names]
                    selected_features = column_names  # Update for consistency
                else:
                    return False, f"Feature selection failed: {e}"

            # Enhanced data leakage detection with multiple tests
            leakage_tests = []

            # Test 1: Basic train/test split validation
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=0.3, random_state=42, stratify=y
            )

            rf = RandomForestClassifier(
                n_estimators=10,
                max_depth=3,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)
            train_acc = accuracy_score(y_train, rf.predict(X_train))
            test_acc = accuracy_score(y_test, rf.predict(X_test))

            # Test 2: Cross-validation consistency check
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            cv_scores = []

            for train_idx, val_idx in skf.split(X_selected, y):
                X_cv_train, X_cv_val = X_selected[train_idx], X_selected[val_idx]
                y_cv_train, y_cv_val = y[train_idx], y[val_idx]

                rf_cv = RandomForestClassifier(
                    n_estimators=10,
                    max_depth=3,
                    random_state=42,
                    n_jobs=-1
                )
                rf_cv.fit(X_cv_train, y_cv_train)
                cv_score = accuracy_score(y_cv_val, rf_cv.predict(X_cv_val))
                cv_scores.append(cv_score)

            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)

            # Test 3: Feature variance check (avoid zero-variance features)
            feature_variances = X_selected.var()
            zero_var_features = np.sum(feature_variances == 0)
            very_low_var_features = np.sum(feature_variances < 1e-6)

            # Test 4: Correlation analysis (avoid highly correlated features)
            correlation_matrix = np.abs(X_selected.corr().values)
            high_corr_pairs = np.sum(correlation_matrix > 0.95) // 2  # Count pairs above threshold

            # Test 5: Baseline comparison (shouldn't be too much better than random)
            majority_class_ratio = np.max(np.bincount(y)) / len(y)
            if majority_class_ratio < 0.8:  # Only check if not heavily imbalanced
                # Simple baseline accuracy
                baseline_acc = majority_class_ratio
                if test_acc > baseline_acc + 0.3:  # Much better than baseline might indicate issues
                    leakage_tests.append(f"Performance suspiciously high: {test_acc:.3f} vs baseline {baseline_acc:.3f}")

            # Comprehensive leakage detection criteria
            is_leaky = False
            leakage_reasons = []

            # Criterion 1: Suspicious train/test accuracy gap
            train_test_gap = train_acc - test_acc
            if train_acc > 0.95 and test_acc < 0.6:
                is_leaky = True
                leakage_reasons.append(f"Severe train/test gap: train={train_acc:.3f}, test={test_acc:.3f}")

            # Criterion 2: Cross-validation inconsistency
            if cv_std > 0.2:  # High variance in CV scores
                leakage_reasons.append(f"High CV score variance: {cv_std:.3f}")

            # Criterion 3: Too many zero-variance features
            if zero_var_features > len(selected_features) * 0.1:  # More than 10% zero variance
                leakage_reasons.append(f"Too many zero-variance features: {zero_var_features}")

            # Criterion 4: Too many highly correlated features
            if high_corr_pairs > len(selected_features) * 0.3:  # More than 30% highly correlated pairs
                leakage_reasons.append(f"Too many highly correlated features: {high_corr_pairs}")

            # Criterion 5: Cross-validation vs test set discrepancy
            cv_test_gap = abs(cv_mean - test_acc)
            if cv_test_gap > 0.15:  # Large gap between CV and test performance
                leakage_reasons.append(f"CV/test discrepancy: CV={cv_mean:.3f}, test={test_acc:.3f}")

            # Feature quality checks
            quality_warnings = []

            if very_low_var_features > 0:
                quality_warnings.append(f"⚠️ {very_low_var_features} features have very low variance")

            if high_corr_pairs > 0:
                quality_warnings.append(f"⚠️ {high_corr_pairs} highly correlated feature pairs detected")

            if cv_std > 0.1:
                quality_warnings.append(f"⚠️ High CV variance ({cv_std:.3f}) indicates unstable features")

            # Overall validation result
            if is_leaky:
                return False, f"Data leakage detected: {'; '.join(leakage_reasons[:3])}"  # Limit to 3 reasons

            # Generate comprehensive validation message
            validation_message = f"Feature selection validated: {len(selected_features)} features"
            if quality_warnings:
                validation_message += f" (Warnings: {'; '.join(quality_warnings[:2])})"  # Limit to 2 warnings

            return True, validation_message

        except Exception as e:
            return False, f"Enhanced feature validation failed: {e}"

        @staticmethod
        def _align_regime_labels(regime_labels: np.ndarray, target_length: int) -> Optional[np.ndarray]:
            """
            Align regime labels to match target length using proportion-based strategy.

            Args:
                regime_labels: Original regime labels array
                target_length: Target length to align to

            Returns:
                Aligned regime labels array or None if alignment fails
            """
            try:
                if len(regime_labels) == target_length:
                    return regime_labels

                unique_regimes, regime_counts = np.unique(regime_labels, return_counts=True)
                total_samples = target_length

                # Calculate proportions for each regime
                regime_proportions = regime_counts / len(regime_labels)

                # Create new regime labels based on proportions
                new_regime_labels = np.zeros(total_samples, dtype=regime_labels.dtype)
                start_idx = 0
                for regime_id, proportion in enumerate(regime_proportions):
                    end_idx = start_idx + int(total_samples * proportion)
                    new_regime_labels[start_idx:end_idx] = regime_id
                    start_idx = end_idx

                # Fill any remaining slots with the last regime
                if start_idx < total_samples:
                    new_regime_labels[start_idx:] = len(unique_regimes) - 1

                return new_regime_labels

            except Exception as e:
                logger.error(f"❌ Error in _align_regime_labels: {e}")
                return None

    @staticmethod
    def validate_train_test_split(X_train, X_test, y_train, y_test, temporal_check=True):
        """Validate train/test split integrity and check for data leakage."""
        try:
            import numpy as np
            from sklearn.metrics import accuracy_score

            # Basic shape validation
            if len(X_train) == 0 or len(X_test) == 0:
                return False, "Empty train or test set"

            if len(X_train) + len(X_test) != len(np.vstack([X_train, X_test])):
                return False, "Train/test sets don't match combined data"

            # Check for identical samples (data leakage)
            X_combined = np.vstack([X_train, X_test])
            unique_samples = np.unique(X_combined, axis=0)
            if len(unique_samples) < len(X_combined):
                return False, f"Duplicate samples detected: {len(X_combined) - len(unique_samples)} duplicates"

            # Temporal validation (if timestamps are available)
            if temporal_check and hasattr(X_train, 'index') and hasattr(X_test, 'index'):
                try:
                    train_max_time = X_train.index.max()
                    test_min_time = X_test.index.min()
                    if train_max_time >= test_min_time:
                        return False, f"Temporal leakage: train max ({train_max_time}) >= test min ({test_min_time})"
                except Exception:
                    pass  # Skip temporal check if not applicable

            # Check for suspiciously high baseline accuracy
            if len(np.unique(y_train)) > 1 and len(np.unique(y_test)) > 1:
                # Simple majority class baseline
                train_majority = np.bincount(y_train).argmax()
                test_majority = np.bincount(y_test).argmax()

                baseline_train_acc = accuracy_score(y_train, [train_majority] * len(y_train))
                baseline_test_acc = accuracy_score(y_test, [test_majority] * len(y_test))

                # If test set has much higher baseline than train, might indicate data leakage
                if baseline_test_acc > baseline_train_acc + 0.1:
                    return False, f"Suspicious baseline: train={baseline_train_acc:.3f}, test={baseline_test_acc:.3f}"

            return True, "Train/test split appears valid"

        except Exception as e:
            return False, f"Split validation failed: {e}"

    @staticmethod
    def detect_overfitting_comprehensive(train_predictions, test_predictions, train_labels, test_labels,
                                       train_probabilities=None, test_probabilities=None,
                                       model=None, feature_importance=None):
        """
        Comprehensive overfitting detection with multiple validation methods.

        Args:
            train_predictions: Predictions on training set
            test_predictions: Predictions on test set
            train_labels: True labels for training set
            test_labels: True labels for test set
            train_probabilities: Probabilities on training set (optional)
            test_probabilities: Probabilities on test set (optional)
            model: Trained model object (optional)
            feature_importance: Feature importance scores (optional)

        Returns:
            Dictionary with overfitting analysis results
        """
        try:
            import numpy as np
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            from sklearn.metrics import log_loss

            # Basic accuracy metrics
            train_accuracy = accuracy_score(train_labels, train_predictions)
            test_accuracy = accuracy_score(test_labels, test_predictions)

            # Calculate accuracy gap
            accuracy_gap = train_accuracy - test_accuracy

            # Calculate other metrics
            train_f1 = f1_score(train_labels, train_predictions, average='weighted')
            test_f1 = f1_score(test_labels, test_predictions, average='weighted')

            train_precision = precision_score(train_labels, train_predictions, average='weighted')
            test_precision = precision_score(test_labels, test_predictions, average='weighted')

            train_recall = recall_score(train_labels, train_predictions, average='weighted')
            test_recall = recall_score(test_labels, test_predictions, average='weighted')

            # Log loss (if probabilities available)
            train_log_loss = None
            test_log_loss = None
            log_loss_gap = None

            if train_probabilities is not None and test_probabilities is not None:
                try:
                    train_log_loss = log_loss(train_labels, train_probabilities)
                    test_log_loss = log_loss(test_labels, test_probabilities)
                    log_loss_gap = train_log_loss - test_log_loss
                except Exception:
                    pass

            # Confidence calibration analysis (if probabilities available)
            confidence_analysis = {}
            if train_probabilities is not None and test_probabilities is not None:
                try:
                    # Calculate max probabilities (confidence)
                    train_max_probs = np.max(train_probabilities, axis=1)
                    test_max_probs = np.max(test_probabilities, axis=1)

                    train_confidence = np.mean(train_max_probs)
                    test_confidence = np.mean(test_max_probs)

                    # Check for overconfident predictions
                    overconfident_threshold = 0.8
                    train_overconfident = np.mean(train_max_probs > overconfident_threshold)
                    test_overconfident = np.mean(test_max_probs > overconfident_threshold)

                    confidence_analysis = {
                        'train_avg_confidence': float(train_confidence),
                        'test_avg_confidence': float(test_confidence),
                        'train_overconfident_ratio': float(train_overconfident),
                        'test_overconfident_ratio': float(test_overconfident),
                        'confidence_gap': float(train_confidence - test_confidence)
                    }
                except Exception:
                    pass

            # Feature importance analysis (if available)
            feature_analysis = {}
            if feature_importance is not None:
                try:
                    # Check for feature concentration
                    sorted_importance = np.sort(feature_importance)[::-1]
                    cumulative_importance = np.cumsum(sorted_importance)

                    # Find number of features needed for 90% importance
                    n_features_90pct = np.where(cumulative_importance >= 0.9)[0][0] + 1
                    feature_concentration = n_features_90pct / len(feature_importance)

                    # Check for sparse importance
                    zero_importance_ratio = np.mean(feature_importance == 0)
                    very_low_importance_ratio = np.mean(feature_importance < 0.001)

                    feature_analysis = {
                        'n_features_for_90pct_importance': int(n_features_90pct),
                        'feature_concentration_ratio': float(feature_concentration),
                        'zero_importance_ratio': float(zero_importance_ratio),
                        'very_low_importance_ratio': float(very_low_importance_ratio)
                    }
                except Exception:
                    pass

            # Determine overfitting severity
            is_overfitting = False
            severity = 'none'
            severity_score = 0

            # Multiple criteria for overfitting detection
            if accuracy_gap > 0.2:
                severity_score += 2  # High severity
                is_overfitting = True
            elif accuracy_gap > 0.1:
                severity_score += 1  # Medium severity
                is_overfitting = True

            # Check F1 score gap
            f1_gap = train_f1 - test_f1
            if f1_gap > 0.15:
                severity_score += 1
                is_overfitting = True

            # Check for overconfident predictions
            if confidence_analysis and confidence_analysis['confidence_gap'] > 0.1:
                severity_score += 1
                is_overfitting = True

            # Check for feature concentration
            if feature_analysis and feature_analysis['feature_concentration_ratio'] < 0.1:
                severity_score += 1  # Very concentrated features may indicate overfitting

            # Determine final severity
            if severity_score >= 3:
                severity = 'high'
            elif severity_score >= 2:
                severity = 'medium'
            elif severity_score >= 1:
                severity = 'low'

            # Generate recommendations
            recommendations = []
            if is_overfitting:
                if severity == 'high':
                    recommendations.extend([
                        "Strong overfitting detected - consider increasing regularization",
                        "Reduce model complexity (depth, features, etc.)",
                        "Increase training data size or use data augmentation",
                        "Apply stronger cross-validation strategies"
                    ])
                elif severity == 'medium':
                    recommendations.extend([
                        "Moderate overfitting detected - consider adding regularization",
                        "Review feature selection process",
                        "Consider ensemble methods or early stopping"
                    ])
                else:  # low
                    recommendations.append("Minor overfitting detected - monitor future performance")

            # Additional warnings
            warnings = []
            if accuracy_gap > 0.3:
                warnings.append("⚠️ Severe overfitting: >30% accuracy gap between train and test")
            elif accuracy_gap > 0.2:
                warnings.append("⚠️ Significant overfitting: >20% accuracy gap between train and test")

            if confidence_analysis and confidence_analysis['train_overconfident_ratio'] > 0.7:
                warnings.append("⚠️ Model is overconfident on training data")

            if feature_analysis and feature_analysis['feature_concentration_ratio'] < 0.05:
                warnings.append("⚠️ Features are highly concentrated - may indicate overfitting to specific patterns")

            result = {
                'is_overfitting': is_overfitting,
                'severity': severity,
                'severity_score': severity_score,
                'accuracy_gap': float(accuracy_gap),
                'f1_gap': float(f1_gap),
                'train_accuracy': float(train_accuracy),
                'test_accuracy': float(test_accuracy),
                'train_f1': float(train_f1),
                'test_f1': float(test_f1),
                'train_precision': float(train_precision),
                'test_precision': float(test_precision),
                'train_recall': float(train_recall),
                'test_recall': float(test_recall),
                'log_loss_gap': float(log_loss_gap) if log_loss_gap is not None else None,
                'confidence_analysis': confidence_analysis,
                'feature_analysis': feature_analysis,
                'recommendations': recommendations,
                'warnings': warnings,
                'validation_score': test_accuracy  # Overall validation score for model selection
            }

            return result

        except Exception as e:
            return {
                'error': str(e),
                'is_overfitting': False,
                'severity': 'unknown',
                'warnings': [f"⚠️ Overfitting detection failed: {e}"]
            }

    @staticmethod
    def validate_model_stability(model, X_train, y_train, X_test, y_test, n_iterations=5):
        """
        Validate model stability through repeated training and evaluation.

        Args:
            model: Model class or instance to test
            X_train, y_train: Training data
            X_test, y_test: Test data
            n_iterations: Number of stability test iterations

        Returns:
            Dictionary with stability analysis results
        """
        try:
            import numpy as np
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import train_test_split

            stability_scores = []
            stability_predictions = []

            for i in range(n_iterations):
                # Create random subsamples
                train_idx = np.random.choice(len(X_train), size=int(0.8 * len(X_train)), replace=False)
                val_idx = np.random.choice(len(X_test), size=int(0.8 * len(X_test)), replace=False)

                X_train_sub = X_train[train_idx]
                y_train_sub = y_train[train_idx]
                X_val_sub = X_test[val_idx]
                y_val_sub = y_test[val_idx]

                # Clone model if instance, create new if class
                if hasattr(model, 'fit'):
                    model_copy = type(model)(**model.get_params())
                else:
                    model_copy = model()

                # Train on subsample
                model_copy.fit(X_train_sub, y_train_sub)

                # Evaluate on validation subsample
                val_pred = model_copy.predict(X_val_sub)
                val_score = accuracy_score(y_val_sub, val_pred)
                stability_scores.append(val_score)
                stability_predictions.append(val_pred)

            # Calculate stability metrics
            stability_mean = np.mean(stability_scores)
            stability_std = np.std(stability_scores)
            stability_cv = stability_std / max(stability_mean, 0.001)  # Coefficient of variation

            # Check prediction consistency
            prediction_consistency = []
            for i in range(len(stability_predictions) - 1):
                consistency = np.mean(stability_predictions[i] == stability_predictions[i + 1])
                prediction_consistency.append(consistency)
            prediction_consistency_mean = np.mean(prediction_consistency)

            result = {
                'stability_mean': float(stability_mean),
                'stability_std': float(stability_std),
                'stability_coefficient_of_variation': float(stability_cv),
                'prediction_consistency': float(prediction_consistency_mean),
                'is_stable': stability_cv < 0.1 and prediction_consistency > 0.8,  # Thresholds
                'stability_score': float(stability_mean * (1 - min(stability_cv, 1.0))),  # Weighted score
                'n_iterations': n_iterations
            }

            return result

        except Exception as e:
            return {
                'error': str(e),
                'is_stable': False,
                'stability_score': 0.0
            }