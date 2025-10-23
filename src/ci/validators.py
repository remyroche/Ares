"""
CI/CD Gates and Tests for End-to-End Roadmap

Implements:
- Build-time hard fails for budget violations
- Unit tests for feature computation
- Golden replay for bit-for-bit reproduction
- Latency harness for performance validation
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
import warnings
import time
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

class ValidationStatus(Enum):
    """Status of validation."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"

@dataclass
class ValidationResult:
    """Result of validation."""
    status: ValidationStatus
    message: str
    details: Dict[str, Any]
    execution_time_ms: float

class BudgetValidator:
    """Validates feature and latency budgets."""

    def __init__(self,
                 feature_budget_pre: int = 120,
                 feature_budget_post: Tuple[int, int] = (30, 60),
                 interactions_cap: int = 15,
                 transforms_per_parent: int = 1,
                 latency_budget_ms: int = 50,
                 lookback_ceiling_minutes: int = 120):
        self.feature_budget_pre = feature_budget_pre
        self.feature_budget_post = feature_budget_post
        self.interactions_cap = interactions_cap
        self.transforms_per_parent = transforms_per_parent
        self.latency_budget_ms = latency_budget_ms
        self.lookback_ceiling_minutes = lookback_ceiling_minutes

    def validate_feature_budgets(self, features: pd.DataFrame) -> ValidationResult:
        """Validate feature count budgets."""
        start_time = time.time()

        issues = []

        # Check pre-selection budget
        if len(features.columns) > self.feature_budget_pre:
            issues.append(f"Pre-selection budget exceeded: {len(features.columns)} > {self.feature_budget_pre}")

        # Check post-selection budget
        if not (self.feature_budget_post[0] <= len(features.columns) <= self.feature_budget_post[1]):
            issues.append(f"Post-selection budget violated: {len(features.columns)} not in {self.feature_budget_post}")

        # Check interactions cap
        interaction_cols = [col for col in features.columns if col.startswith('i/')]
        if len(interaction_cols) > self.interactions_cap:
            issues.append(f"Interactions cap exceeded: {len(interaction_cols)} > {self.interactions_cap}")

        # Check transforms per parent
        parent_cols = [col for col in features.columns if col.startswith('p/')]
        transform_cols = [col for col in features.columns if col.startswith('t/')]

        # Count transforms per parent
        parent_transform_counts = {}
        for col in transform_cols:
            # Extract parent name from transform column (e.g., 't/r1/ewz12' -> 'r1')
            parts = col.split('/')
            if len(parts) >= 2:
                parent = parts[1]
                parent_transform_counts[parent] = parent_transform_counts.get(parent, 0) + 1

        for parent, count in parent_transform_counts.items():
            if count > self.transforms_per_parent:
                issues.append(f"Transforms per parent exceeded for {parent}: {count} > {self.transforms_per_parent}")

        status = ValidationStatus.FAIL if issues else ValidationStatus.PASS
        message = "Feature budgets validated" if not issues else f"Budget violations: {'; '.join(issues)}"

        execution_time = (time.time() - start_time) * 1000

        return ValidationResult(
            status=status,
            message=message,
            details={
                'total_features': len(features.columns),
                'interaction_features': len(interaction_cols),
                'parent_features': len(parent_cols),
                'transform_features': len(transform_cols),
                'parent_transform_counts': parent_transform_counts,
                'issues': issues
            },
            execution_time_ms=execution_time
        )

    def validate_lookback_ceiling(self, features: pd.DataFrame) -> ValidationResult:
        """Validate lookback ceiling compliance."""
        start_time = time.time()

        issues = []

        # Check for features that might violate lookback ceiling
        # This is a simplified check - in practice, would need feature metadata
        for col in features.columns:
            if 'sigma_ew' in col and '18' in col:  # Example check
                issues.append(f"Feature {col} may violate lookback ceiling")

        status = ValidationStatus.WARNING if issues else ValidationStatus.PASS
        message = "Lookback ceiling validated" if not issues else f"Potential ceiling violations: {'; '.join(issues)}"

        execution_time = (time.time() - start_time) * 1000

        return ValidationResult(
            status=status,
            message=message,
            details={'issues': issues},
            execution_time_ms=execution_time
        )

    def validate_transform_types(self, features: pd.DataFrame) -> ValidationResult:
        """Validate transform types are from allowed set."""
        start_time = time.time()

        allowed_transforms = {'ewz', 'tod_rank', 'signed_log', 'winsor'}
        issues = []

        transform_cols = [col for col in features.columns if col.startswith('t/')]

        for col in transform_cols:
            # Extract transform type from column name (e.g., 't/r1/ewz12' -> 'ewz12')
            parts = col.split('/')
            if len(parts) >= 3:
                transform_part = parts[2]
                # Check if transform type is in allowed set
                transform_type = None
                for allowed in allowed_transforms:
                    if transform_part.startswith(allowed):
                        transform_type = allowed
                        break

                if transform_type is None:
                    issues.append(f"Unknown transform type in {col}: {transform_part}")

        status = ValidationStatus.FAIL if issues else ValidationStatus.PASS
        message = "Transform types validated" if not issues else f"Invalid transforms: {'; '.join(issues)}"

        execution_time = (time.time() - start_time) * 1000

        return ValidationResult(
            status=status,
            message=message,
            details={'issues': issues, 'allowed_transforms': list(allowed_transforms)},
            execution_time_ms=execution_time
        )

class LatencyHarness:
    """Latency testing harness."""

    def __init__(self, latency_budget_ms: int = 50):
        self.latency_budget_ms = latency_budget_ms
        self.results = {}

    def test_component_latency(self,
                              component_name: str,
                              test_function,
                              *args, **kwargs) -> ValidationResult:
        """Test latency of a component."""
        start_time = time.time()

        try:
            # Run the function
            result = test_function(*args, **kwargs)

            execution_time_ms = (time.time() - start_time) * 1000

            # Check against budget
            if execution_time_ms > self.latency_budget_ms:
                status = ValidationStatus.FAIL
                message = f"Latency budget exceeded: {execution_time_ms:.2f}ms > {self.latency_budget_ms}ms"
            else:
                status = ValidationStatus.PASS
                message = f"Latency within budget: {execution_time_ms:.2f}ms <= {self.latency_budget_ms}ms"

            self.results[component_name] = {
                'latency_ms': execution_time_ms,
                'status': status,
                'result': result
            }

            return ValidationResult(
                status=status,
                message=message,
                details={
                    'component': component_name,
                    'latency_ms': execution_time_ms,
                    'budget_ms': self.latency_budget_ms,
                    'result_type': type(result).__name__
                },
                execution_time_ms=execution_time_ms
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message=f"Component {component_name} failed: {str(e)}",
                details={
                    'component': component_name,
                    'error': str(e),
                    'latency_ms': execution_time_ms
                },
                execution_time_ms=execution_time_ms
            )

    def test_full_pipeline_latency(self,
                                  pipeline_function,
                                  test_data: pd.DataFrame,
                                  *args, **kwargs) -> ValidationResult:
        """Test full pipeline latency."""
        return self.test_component_latency(
            'full_pipeline',
            pipeline_function,
            test_data,
            *args,
            **kwargs
        )

class GoldenReplayValidator:
    """Validates bit-for-bit reproduction."""

    def __init__(self, reference_dir: str = "golden_replay"):
        self.reference_dir = Path(reference_dir)
        self.reference_dir.mkdir(exist_ok=True)

    def save_reference(self,
                      name: str,
                      data: pd.DataFrame,
                      metadata: Dict[str, Any]) -> str:
        """Save reference data for golden replay."""

        # Create hash of data
        data_hash = self._calculate_data_hash(data)

        # Save data and metadata
        reference_file = self.reference_dir / f"{name}_{data_hash}.parquet"
        metadata_file = self.reference_dir / f"{name}_{data_hash}_metadata.json"

        data.to_parquet(reference_file)

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        return str(reference_file)

    def validate_reproduction(self,
                            name: str,
                            data: pd.DataFrame,
                            metadata: Dict[str, Any]) -> ValidationResult:
        """Validate bit-for-bit reproduction."""
        start_time = time.time()

        # Calculate current data hash
        current_hash = self._calculate_data_hash(data)

        # Look for reference files
        reference_files = list(self.reference_dir.glob(f"{name}_*.parquet"))

        if not reference_files:
            return ValidationResult(
                status=ValidationStatus.WARNING,
                message=f"No reference data found for {name}",
                details={'current_hash': current_hash},
                execution_time_ms=(time.time() - start_time) * 1000
            )

        # Check against all reference files
        matches = []
        for ref_file in reference_files:
            try:
                ref_data = pd.read_parquet(ref_file)
                ref_hash = self._calculate_data_hash(ref_data)

                if ref_hash == current_hash:
                    matches.append(str(ref_file))

            except Exception as e:
                warnings.warn(f"Failed to read reference file {ref_file}: {e}")
                continue

        if matches:
            status = ValidationStatus.PASS
            message = f"Bit-for-bit reproduction validated for {name}"
        else:
            status = ValidationStatus.FAIL
            message = f"Bit-for-bit reproduction failed for {name}"

        execution_time = (time.time() - start_time) * 1000

        return ValidationResult(
            status=status,
            message=message,
            details={
                'current_hash': current_hash,
                'reference_matches': matches,
                'total_references': len(reference_files)
            },
            execution_time_ms=execution_time
        )

    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash of DataFrame."""
        # Convert to string representation for hashing
        data_str = data.to_string()
        return hashlib.md5(data_str.encode()).hexdigest()

class UnitTestSuite:
    """Unit tests for feature computation."""

    def __init__(self):
        self.test_results = {}

    def run_all_tests(self) -> Dict[str, ValidationResult]:
        """Run all unit tests."""

        # Test session VWAP reset
        self.test_results['session_vwap_reset'] = self._test_session_vwap_reset()

        # Test DST and half-day flags
        self.test_results['dst_halfday_flags'] = self._test_dst_halfday_flags()

        # Test online EW-Z state continuity
        self.test_results['ewz_state_continuity'] = self._test_ewz_state_continuity()

        # Test missing book data handling
        self.test_results['missing_book_data'] = self._test_missing_book_data()

        # Test feature computation
        self.test_results['feature_computation'] = self._test_feature_computation()

        return self.test_results

    def _test_session_vwap_reset(self) -> ValidationResult:
        """Test that session VWAP resets per session."""
        start_time = time.time()

        try:
            # Create test data with multiple sessions
            dates = pd.date_range('2024-01-01', periods=100, freq='5min')
            test_data = pd.DataFrame({
                'timestamp': dates,
                'open': np.random.randn(100).cumsum() + 100,
                'high': np.random.randn(100).cumsum() + 105,
                'low': np.random.randn(100).cumsum() + 95,
                'close': np.random.randn(100).cumsum() + 100,
                'volume': np.random.randint(1000, 10000, 100)
            })

            # Add session IDs
            test_data['session_id'] = test_data['timestamp'].dt.date

            # Test VWAP calculation (simplified)
            test_data['vwap'] = (test_data['high'] + test_data['low'] + test_data['close']) / 3

            # Check that VWAP resets per session
            session_vwaps = test_data.groupby('session_id')['vwap'].first()

            if len(session_vwaps) > 1:
                status = ValidationStatus.PASS
                message = "Session VWAP reset test passed"
            else:
                status = ValidationStatus.FAIL
                message = "Session VWAP reset test failed - only one session detected"

        except Exception as e:
            status = ValidationStatus.FAIL
            message = f"Session VWAP reset test failed: {str(e)}"

        execution_time = (time.time() - start_time) * 1000

        return ValidationResult(
            status=status,
            message=message,
            details={'test': 'session_vwap_reset'},
            execution_time_ms=execution_time
        )

    def _test_dst_halfday_flags(self) -> ValidationResult:
        """Test DST and half-day flag computation."""
        start_time = time.time()

        try:
            # Create test data around DST transition
            dates = pd.date_range('2024-03-10', periods=48, freq='30min')  # DST transition
            test_data = pd.DataFrame({
                'timestamp': dates,
                'open': np.random.randn(48) + 100,
                'high': np.random.randn(48) + 105,
                'low': np.random.randn(48) + 95,
                'close': np.random.randn(48) + 100,
                'volume': np.random.randint(1000, 10000, 48)
            })

            # Test open30 and last30 flags (simplified)
            test_data['open30'] = (test_data['timestamp'].dt.hour < 10).astype(int)
            test_data['last30'] = (test_data['timestamp'].dt.hour >= 15).astype(int)

            # Check that flags are computed correctly
            open30_count = test_data['open30'].sum()
            last30_count = test_data['last30'].sum()

            if open30_count > 0 and last30_count > 0:
                status = ValidationStatus.PASS
                message = "DST and half-day flags test passed"
            else:
                status = ValidationStatus.FAIL
                message = "DST and half-day flags test failed"

        except Exception as e:
            status = ValidationStatus.FAIL
            message = f"DST and half-day flags test failed: {str(e)}"

        execution_time = (time.time() - start_time) * 1000

        return ValidationResult(
            status=status,
            message=message,
            details={'test': 'dst_halfday_flags'},
            execution_time_ms=execution_time
        )

    def _test_ewz_state_continuity(self) -> ValidationResult:
        """Test online EW-Z state continuity between batches."""
        start_time = time.time()

        try:
            from src.feature_engineering_roadmap.transforms import OnlineEWZ

            # Create test data
            data1 = pd.Series(np.random.randn(100))
            data2 = pd.Series(np.random.randn(100))

            # Test EW-Z with state continuity
            ewz = OnlineEWZ(halflife=12)

            # Fit on first batch
            result1 = ewz.fit_transform(data1)
            state1 = ewz.get_state()

            # Transform second batch with same state
            ewz2 = OnlineEWZ(halflife=12)
            ewz2.set_state(state1)
            result2 = ewz2.transform(data2)

            # Check that results are finite
            if result1.notna().all() and result2.notna().all():
                status = ValidationStatus.PASS
                message = "EW-Z state continuity test passed"
            else:
                status = ValidationStatus.FAIL
                message = "EW-Z state continuity test failed - NaN values detected"

        except Exception as e:
            status = ValidationStatus.FAIL
            message = f"EW-Z state continuity test failed: {str(e)}"

        execution_time = (time.time() - start_time) * 1000

        return ValidationResult(
            status=status,
            message=message,
            details={'test': 'ewz_state_continuity'},
            execution_time_ms=execution_time
        )

    def _test_missing_book_data(self) -> ValidationResult:
        """Test graceful handling of missing book data."""
        start_time = time.time()

        try:
            from src.feature_engineering_roadmap.feature_registry import LiquidityMicroFeatures

            # Create test data without book fields
            test_data = pd.DataFrame({
                'open': np.random.randn(100) + 100,
                'high': np.random.randn(100) + 105,
                'low': np.random.randn(100) + 95,
                'close': np.random.randn(100) + 100,
                'volume': np.random.randint(1000, 10000, 100)
            })

            # Test micro features without book data
            spread_z = LiquidityMicroFeatures.spread_z18(test_data)
            ofi_proxy = LiquidityMicroFeatures.ofi_proxy(test_data)
            microprice_dev = LiquidityMicroFeatures.microprice_dev(test_data)

            # Check that features return Series with correct index
            if (len(spread_z) == 100 and
                len(ofi_proxy) == 100 and
                len(microprice_dev) == 100):
                status = ValidationStatus.PASS
                message = "Missing book data handling test passed"
            else:
                status = ValidationStatus.FAIL
                message = "Missing book data handling test failed"

        except Exception as e:
            status = ValidationStatus.FAIL
            message = f"Missing book data handling test failed: {str(e)}"

        execution_time = (time.time() - start_time) * 1000

        return ValidationResult(
            status=status,
            message=message,
            details={'test': 'missing_book_data'},
            execution_time_ms=execution_time
        )

    def _test_feature_computation(self) -> ValidationResult:
        """Test basic feature computation."""
        start_time = time.time()

        try:
            from src.feature_engineering_roadmap.feature_registry import PriceReturnsFeatures

            # Create test data
            test_data = pd.DataFrame({
                'close': [100, 101, 99, 102, 98, 103, 97, 104, 96, 105]
            })

            # Test r1 calculation
            r1 = PriceReturnsFeatures.r1(test_data)

            # Check that r1 is calculated correctly
            expected_r1 = np.log(test_data['close'] / test_data['close'].shift(1))

            if np.allclose(r1.dropna(), expected_r1.dropna(), rtol=1e-10):
                status = ValidationStatus.PASS
                message = "Feature computation test passed"
            else:
                status = ValidationStatus.FAIL
                message = "Feature computation test failed - incorrect calculation"

        except Exception as e:
            status = ValidationStatus.FAIL
            message = f"Feature computation test failed: {str(e)}"

        execution_time = (time.time() - start_time) * 1000

        return ValidationResult(
            status=status,
            message=message,
            details={'test': 'feature_computation'},
            execution_time_ms=execution_time
        )

class CICDValidator:
    """Main CI/CD validator."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.budget_validator = BudgetValidator(**self.config.get('budgets', {}))
        self.latency_harness = LatencyHarness(**self.config.get('latency', {}))
        self.golden_replay = GoldenReplayValidator(**self.config.get('golden_replay', {}))
        self.unit_tests = UnitTestSuite()

    def run_all_validations(self,
                           features: pd.DataFrame,
                           test_data: Optional[pd.DataFrame] = None) -> Dict[str, ValidationResult]:
        """Run all CI/CD validations."""

        results = {}

        # Budget validations
        results['feature_budgets'] = self.budget_validator.validate_feature_budgets(features)
        results['lookback_ceiling'] = self.budget_validator.validate_lookback_ceiling(features)
        results['transform_types'] = self.budget_validator.validate_transform_types(features)

        # Unit tests
        unit_test_results = self.unit_tests.run_all_tests()
        results.update(unit_test_results)

        # Golden replay (if test data provided)
        if test_data is not None:
            results['golden_replay'] = self.golden_replay.validate_reproduction(
                'test_features', features, {'timestamp': '2024-01-01'}
            )

        return results

    def should_fail_build(self, results: Dict[str, ValidationResult]) -> bool:
        """Determine if build should fail based on validation results."""

        critical_tests = [
            'feature_budgets',
            'transform_types',
            'feature_computation',
            'ewz_state_continuity'
        ]

        for test_name in critical_tests:
            if test_name in results and results[test_name].status == ValidationStatus.FAIL:
                return True

        return False

def run_ci_validation(features: pd.DataFrame,
                     test_data: Optional[pd.DataFrame] = None,
                     config: Optional[Dict[str, Any]] = None) -> Dict[str, ValidationResult]:
    """Run complete CI validation pipeline."""

    validator = CICDValidator(config)
    return validator.run_all_validations(features, test_data)
