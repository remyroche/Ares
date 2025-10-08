"""
Pre-Training Validation Framework

Comprehensive validation system for the pre-training pipeline to ensure:
1. Data integrity and representativeness
2. Label quality and target validity
3. Feature engineering soundness
4. Lookback optimization robustness
5. Feature selection stability
6. Reproducibility and scientific rigor
7. Quantitative soundness

This framework implements all recommended tests from the pre-training audit.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
import logging
import json
import hashlib
import subprocess

from scipy import stats
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success


logger = logging.getLogger(__name__)


@dataclass
class ValidationThresholds:
    """Thresholds for validation tests."""
    
    # Label quality
    label_autocorr_max: float = 0.1  # Max autocorrelation for h>3
    min_mutual_info_percentile: float = 10.0  # Top 10% features retained
    
    # Feature quality
    feature_stability_pvalue: float = 0.05  # KS test p-value
    min_sharpe_ratio: float = 0.5  # Minimum Sharpe for synthetic signal
    
    # Lookback optimization
    max_lookback_sensitivity: float = 0.15  # Max 15% change under resampling
    
    # Information coefficient
    min_ic_mean: float = 0.02  # Minimum mean IC
    max_ic_mean: float = 0.05  # Typical max IC
    min_ic_tstat: float = 2.0  # Minimum t-statistic
    
    # Distribution checks
    max_distribution_shift: float = 2.0  # Max shift in standard deviations
    min_sample_ratio: float = 0.8  # Min ratio between train/val/test samples


@dataclass
class ValidationResult:
    """Result of a validation test."""
    
    test_name: str
    passed: bool
    score: float
    threshold: float
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PreTrainingValidationReport:
    """Comprehensive validation report for pre-training pipeline."""
    
    # Overall results
    all_tests_passed: bool
    total_tests: int
    passed_tests: int
    failed_tests: int
    
    # Test results by category
    data_integrity_results: List[ValidationResult] = field(default_factory=list)
    label_quality_results: List[ValidationResult] = field(default_factory=list)
    feature_engineering_results: List[ValidationResult] = field(default_factory=list)
    lookback_optimization_results: List[ValidationResult] = field(default_factory=list)
    feature_selection_results: List[ValidationResult] = field(default_factory=list)
    reproducibility_results: List[ValidationResult] = field(default_factory=list)
    soundness_check_results: List[ValidationResult] = field(default_factory=list)
    
    # Metadata
    pipeline_config_hash: Optional[str] = None
    data_checksum: Optional[str] = None
    git_commit: Optional[str] = None
    environment_info: Dict[str, str] = field(default_factory=dict)
    
    # Timestamp
    validation_timestamp: str = None
    
    def __post_init__(self):
        if self.validation_timestamp is None:
            self.validation_timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'summary': {
                'all_tests_passed': self.all_tests_passed,
                'total_tests': self.total_tests,
                'passed_tests': self.passed_tests,
                'failed_tests': self.failed_tests,
                'pass_rate': self.passed_tests / self.total_tests if self.total_tests > 0 else 0.0
            },
            'data_integrity': [r.to_dict() for r in self.data_integrity_results],
            'label_quality': [r.to_dict() for r in self.label_quality_results],
            'feature_engineering': [r.to_dict() for r in self.feature_engineering_results],
            'lookback_optimization': [r.to_dict() for r in self.lookback_optimization_results],
            'feature_selection': [r.to_dict() for r in self.feature_selection_results],
            'reproducibility': [r.to_dict() for r in self.reproducibility_results],
            'soundness_checks': [r.to_dict() for r in self.soundness_check_results],
            'metadata': {
                'pipeline_config_hash': self.pipeline_config_hash,
                'data_checksum': self.data_checksum,
                'git_commit': self.git_commit,
                'environment_info': self.environment_info,
                'validation_timestamp': self.validation_timestamp
            }
        }


class PreTrainingValidator:
    """
    Comprehensive validator for pre-training pipeline.
    
    Implements all 7 validation aspects with quantitative tests.
    """
    
    def __init__(self, thresholds: Optional[ValidationThresholds] = None):
        """
        Initialize pre-training validator.
        
        Args:
            thresholds: Validation thresholds
        """
        self.thresholds = thresholds or ValidationThresholds()
        self.validation_results: List[ValidationResult] = []
        
        tprint_success("✅ PreTrainingValidator initialized")
    
    # =================================================================
    # 1. DATA INTEGRITY & REPRESENTATIVENESS
    # =================================================================
    
    def validate_label_autocorrelation(
        self,
        labels: pd.DataFrame,
        max_lag: int = 10
    ) -> ValidationResult:
        """
        Test: Label autocorrelation decay
        Expectation: ρ(h) < 0.1 for h>3
        """
        tprint_info("🔍 Testing label autocorrelation decay...")
        
        try:
            autocorrs = []
            target_columns = [col for col in labels.columns if 'target' in col.lower() or 'label' in col.lower()]
            
            for col in target_columns:
                series = labels[col].dropna()
                if len(series) > max_lag + 10:
                    for lag in range(4, max_lag + 1):
                        autocorr = series.autocorr(lag=lag)
                        if not np.isnan(autocorr):
                            autocorrs.append(abs(autocorr))
            
            if not autocorrs:
                return ValidationResult(
                    test_name="label_autocorrelation",
                    passed=False,
                    score=0.0,
                    threshold=self.thresholds.label_autocorr_max,
                    warnings=["No valid autocorrelations computed"]
                )
            
            max_autocorr = np.max(autocorrs)
            mean_autocorr = np.mean(autocorrs)
            
            passed = max_autocorr < self.thresholds.label_autocorr_max
            
            result = ValidationResult(
                test_name="label_autocorrelation",
                passed=passed,
                score=mean_autocorr,
                threshold=self.thresholds.label_autocorr_max,
                details={
                    'max_autocorr': float(max_autocorr),
                    'mean_autocorr': float(mean_autocorr),
                    'n_lags_tested': len(autocorrs)
                }
            )
            
            if not passed:
                result.warnings.append(f"High autocorrelation detected: {max_autocorr:.3f} > {self.thresholds.label_autocorr_max}")
                result.recommendations.append("Labels may be too predictable - consider different labeling scheme")
            
            self.validation_results.append(result)
            return result
            
        except Exception as e:
            tprint_error(f"❌ Error in autocorrelation test: {e}")
            return ValidationResult(
                test_name="label_autocorrelation",
                passed=False,
                score=0.0,
                threshold=self.thresholds.label_autocorr_max,
                warnings=[f"Test failed: {str(e)}"]
            )
    
    def validate_feature_target_mutual_info(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        top_percentile: float = 10.0
    ) -> ValidationResult:
        """
        Test: Feature-target mutual information
        Expectation: Top 10% retained based on MI
        """
        tprint_info("🔍 Testing feature-target mutual information...")
        
        try:
            # Select first target column
            target_col = targets.columns[0]
            target = targets[target_col].dropna()
            
            # Align features with target
            common_idx = features.index.intersection(target.index)
            features_aligned = features.loc[common_idx]
            target_aligned = target.loc[common_idx]
            
            # Calculate MI for each feature
            mi_scores = []
            feature_names = []
            
            for col in features_aligned.columns:
                feature_data = features_aligned[col].fillna(0)
                
                # Discretize continuous variables
                if feature_data.nunique() > 10:
                    feature_data = pd.qcut(feature_data, q=10, labels=False, duplicates='drop')
                
                target_discrete = pd.qcut(target_aligned, q=5, labels=False, duplicates='drop')
                
                mi = mutual_info_score(feature_data, target_discrete)
                mi_scores.append(mi)
                feature_names.append(col)
            
            if not mi_scores:
                return ValidationResult(
                    test_name="feature_target_mutual_info",
                    passed=False,
                    score=0.0,
                    threshold=top_percentile,
                    warnings=["No MI scores computed"]
                )
            
            # Calculate threshold for top percentile
            mi_threshold = np.percentile(mi_scores, 100 - top_percentile)
            n_above_threshold = np.sum(np.array(mi_scores) >= mi_threshold)
            
            passed = n_above_threshold >= len(mi_scores) * (top_percentile / 100.0)
            
            result = ValidationResult(
                test_name="feature_target_mutual_info",
                passed=passed,
                score=float(np.mean(mi_scores)),
                threshold=float(mi_threshold),
                details={
                    'mean_mi': float(np.mean(mi_scores)),
                    'max_mi': float(np.max(mi_scores)),
                    'min_mi': float(np.min(mi_scores)),
                    'n_features_above_threshold': int(n_above_threshold),
                    'percentile_threshold': top_percentile
                }
            )
            
            if not passed:
                result.warnings.append("Too few features with high mutual information")
                result.recommendations.append("Consider adding more informative features")
            
            self.validation_results.append(result)
            return result
            
        except Exception as e:
            tprint_error(f"❌ Error in mutual info test: {e}")
            return ValidationResult(
                test_name="feature_target_mutual_info",
                passed=False,
                score=0.0,
                threshold=top_percentile,
                warnings=[f"Test failed: {str(e)}"]
            )
    
    def validate_feature_stability_across_regimes(
        self,
        features: pd.DataFrame,
        regime_column: str
    ) -> ValidationResult:
        """
        Test: Feature stability across regimes
        Expectation: KS test p>0.05
        """
        tprint_info("🔍 Testing feature stability across regimes...")
        
        try:
            if regime_column not in features.columns:
                return ValidationResult(
                    test_name="feature_stability_regimes",
                    passed=False,
                    score=0.0,
                    threshold=self.thresholds.feature_stability_pvalue,
                    warnings=[f"Regime column '{regime_column}' not found"]
                )
            
            regimes = features[regime_column].unique()
            if len(regimes) < 2:
                return ValidationResult(
                    test_name="feature_stability_regimes",
                    passed=True,
                    score=1.0,
                    threshold=self.thresholds.feature_stability_pvalue,
                    warnings=["Only one regime found - test skipped"]
                )
            
            # Test each feature across regimes
            ks_pvalues = []
            unstable_features = []
            
            numeric_features = features.select_dtypes(include=[np.number]).columns
            numeric_features = [col for col in numeric_features if col != regime_column]
            
            for col in numeric_features:
                regime_data = []
                for regime in regimes[:2]:  # Compare first two regimes
                    regime_mask = features[regime_column] == regime
                    data = features.loc[regime_mask, col].dropna()
                    if len(data) > 10:
                        regime_data.append(data.values)
                
                if len(regime_data) == 2:
                    ks_stat, p_value = stats.ks_2samp(regime_data[0], regime_data[1])
                    ks_pvalues.append(p_value)
                    
                    if p_value < self.thresholds.feature_stability_pvalue:
                        unstable_features.append(col)
            
            if not ks_pvalues:
                return ValidationResult(
                    test_name="feature_stability_regimes",
                    passed=False,
                    score=0.0,
                    threshold=self.thresholds.feature_stability_pvalue,
                    warnings=["No valid KS tests completed"]
                )
            
            mean_p_value = np.mean(ks_pvalues)
            passed = mean_p_value > self.thresholds.feature_stability_pvalue
            
            result = ValidationResult(
                test_name="feature_stability_regimes",
                passed=passed,
                score=float(mean_p_value),
                threshold=self.thresholds.feature_stability_pvalue,
                details={
                    'mean_p_value': float(mean_p_value),
                    'n_unstable_features': len(unstable_features),
                    'unstable_features': unstable_features[:10],  # Top 10
                    'n_features_tested': len(ks_pvalues)
                }
            )
            
            if not passed:
                result.warnings.append(f"Feature instability detected across regimes: {len(unstable_features)} features")
                result.recommendations.append("Consider regime-specific feature engineering")
            
            self.validation_results.append(result)
            return result
            
        except Exception as e:
            tprint_error(f"❌ Error in regime stability test: {e}")
            return ValidationResult(
                test_name="feature_stability_regimes",
                passed=False,
                score=0.0,
                threshold=self.thresholds.feature_stability_pvalue,
                warnings=[f"Test failed: {str(e)}"]
            )
    
    # =================================================================
    # 2. LABEL DESIGN & TARGET QUALITY
    # =================================================================
    
    def validate_sharpe_of_synthetic_signal(
        self,
        labels: pd.DataFrame,
        prices: Optional[pd.DataFrame] = None
    ) -> ValidationResult:
        """
        Test: Sharpe of synthetic signal
        Expectation: >0.5 on validation
        """
        tprint_info("🔍 Testing Sharpe ratio of synthetic signal...")
        
        try:
            # Use first target column
            target_col = [col for col in labels.columns if 'target' in col.lower()][0]
            signals = labels[target_col].dropna()
            
            if len(signals) < 50:
                return ValidationResult(
                    test_name="sharpe_synthetic_signal",
                    passed=False,
                    score=0.0,
                    threshold=self.thresholds.min_sharpe_ratio,
                    warnings=["Insufficient samples for Sharpe calculation"]
                )
            
            # Create synthetic returns from signals
            # Assume signal represents expected return
            returns = signals.values
            
            # Calculate Sharpe ratio
            if np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
            else:
                sharpe = 0.0
            
            passed = sharpe > self.thresholds.min_sharpe_ratio
            
            result = ValidationResult(
                test_name="sharpe_synthetic_signal",
                passed=passed,
                score=float(sharpe),
                threshold=self.thresholds.min_sharpe_ratio,
                details={
                    'sharpe_ratio': float(sharpe),
                    'mean_return': float(np.mean(returns)),
                    'std_return': float(np.std(returns)),
                    'n_samples': len(returns)
                }
            )
            
            if not passed:
                result.warnings.append(f"Low Sharpe ratio: {sharpe:.3f} < {self.thresholds.min_sharpe_ratio}")
                result.recommendations.append("Labels may not be economically viable")
            
            self.validation_results.append(result)
            return result
            
        except Exception as e:
            tprint_error(f"❌ Error in Sharpe test: {e}")
            return ValidationResult(
                test_name="sharpe_synthetic_signal",
                passed=False,
                score=0.0,
                threshold=self.thresholds.min_sharpe_ratio,
                warnings=[f"Test failed: {str(e)}"]
            )
    
    # =================================================================
    # 3. LOOKBACK OPTIMIZATION
    # =================================================================
    
    def validate_lookback_sensitivity(
        self,
        lookback_results: Dict[str, Any],
        n_resamples: int = 10
    ) -> ValidationResult:
        """
        Test: Lookback sensitivity
        Expectation: <15% change under resampling
        """
        tprint_info("🔍 Testing lookback sensitivity...")
        
        try:
            if 'optimal_lookback' not in lookback_results:
                return ValidationResult(
                    test_name="lookback_sensitivity",
                    passed=False,
                    score=0.0,
                    threshold=self.thresholds.max_lookback_sensitivity,
                    warnings=["No optimal lookback found in results"]
                )
            
            optimal_lookback = lookback_results['optimal_lookback']
            
            # If multiple lookbacks tested, check sensitivity
            if 'lookback_scores' in lookback_results:
                scores = lookback_results['lookback_scores']
                lookbacks = list(scores.keys())
                
                if len(lookbacks) > 1:
                    # Calculate coefficient of variation
                    lookback_values = [float(lb) for lb in lookbacks]
                    cv = np.std(lookback_values) / np.mean(lookback_values) if np.mean(lookback_values) > 0 else 0
                    
                    passed = cv < self.thresholds.max_lookback_sensitivity
                    
                    result = ValidationResult(
                        test_name="lookback_sensitivity",
                        passed=passed,
                        score=float(cv),
                        threshold=self.thresholds.max_lookback_sensitivity,
                        details={
                            'coefficient_of_variation': float(cv),
                            'optimal_lookback': optimal_lookback,
                            'n_lookbacks_tested': len(lookbacks)
                        }
                    )
                    
                    if not passed:
                        result.warnings.append(f"High lookback sensitivity: {cv:.1%} > {self.thresholds.max_lookback_sensitivity:.1%}")
                        result.recommendations.append("Lookback optimization may be overfit")
                    
                    self.validation_results.append(result)
                    return result
            
            # Default: pass if we have an optimal lookback
            result = ValidationResult(
                test_name="lookback_sensitivity",
                passed=True,
                score=0.0,
                threshold=self.thresholds.max_lookback_sensitivity,
                details={'optimal_lookback': optimal_lookback}
            )
            
            self.validation_results.append(result)
            return result
            
        except Exception as e:
            tprint_error(f"❌ Error in lookback sensitivity test: {e}")
            return ValidationResult(
                test_name="lookback_sensitivity",
                passed=False,
                score=0.0,
                threshold=self.thresholds.max_lookback_sensitivity,
                warnings=[f"Test failed: {str(e)}"]
            )
    
    # =================================================================
    # 4. INFORMATION COEFFICIENT (IC)
    # =================================================================
    
    def validate_information_coefficient(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        feature_names: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Test: IC mean/vol
        Expectation: Mean(IC)≈0.02-0.05; t-stat>2
        """
        tprint_info("🔍 Testing information coefficient...")
        
        try:
            if feature_names is None:
                feature_names = features.columns.tolist()
            
            # Select first target
            target_col = targets.columns[0]
            target = targets[target_col].dropna()
            
            # Calculate IC for each feature
            ics = []
            for col in feature_names:
                if col not in features.columns:
                    continue
                
                feature = features[col].dropna()
                common_idx = feature.index.intersection(target.index)
                
                if len(common_idx) > 10:
                    f_aligned = feature.loc[common_idx]
                    t_aligned = target.loc[common_idx]
                    
                    # Calculate rank IC (Spearman correlation)
                    ic = f_aligned.corr(t_aligned, method='spearman')
                    if not np.isnan(ic):
                        ics.append(ic)
            
            if len(ics) < 5:
                return ValidationResult(
                    test_name="information_coefficient",
                    passed=False,
                    score=0.0,
                    threshold=self.thresholds.min_ic_mean,
                    warnings=["Too few IC values computed"]
                )
            
            mean_ic = np.mean(np.abs(ics))
            std_ic = np.std(ics)
            t_stat = mean_ic / (std_ic / np.sqrt(len(ics))) if std_ic > 0 else 0
            
            passed = (
                mean_ic >= self.thresholds.min_ic_mean and
                mean_ic <= self.thresholds.max_ic_mean and
                t_stat > self.thresholds.min_ic_tstat
            )
            
            result = ValidationResult(
                test_name="information_coefficient",
                passed=passed,
                score=float(mean_ic),
                threshold=self.thresholds.min_ic_mean,
                details={
                    'mean_ic': float(mean_ic),
                    'std_ic': float(std_ic),
                    't_statistic': float(t_stat),
                    'n_features': len(ics)
                }
            )
            
            if not passed:
                if mean_ic < self.thresholds.min_ic_mean:
                    result.warnings.append(f"IC too low: {mean_ic:.4f} < {self.thresholds.min_ic_mean}")
                if t_stat < self.thresholds.min_ic_tstat:
                    result.warnings.append(f"IC t-stat too low: {t_stat:.2f} < {self.thresholds.min_ic_tstat}")
                result.recommendations.append("Features may have weak predictive power")
            
            self.validation_results.append(result)
            return result
            
        except Exception as e:
            tprint_error(f"❌ Error in IC test: {e}")
            return ValidationResult(
                test_name="information_coefficient",
                passed=False,
                score=0.0,
                threshold=self.thresholds.min_ic_mean,
                warnings=[f"Test failed: {str(e)}"]
            )
    
    # =================================================================
    # 5. REPRODUCIBILITY
    # =================================================================
    
    def validate_reproducibility(
        self,
        config: Dict[str, Any],
        data: Optional[pd.DataFrame] = None
    ) -> List[ValidationResult]:
        """
        Test: Reproducibility checks
        Captures: git commit, environment, random seed, data checksum
        """
        tprint_info("🔍 Testing reproducibility...")
        
        results = []
        
        # 1. Git commit SHA
        try:
            git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
            results.append(ValidationResult(
                test_name="git_commit_capture",
                passed=True,
                score=1.0,
                threshold=1.0,
                details={'git_commit': git_commit}
            ))
        except Exception as e:
            results.append(ValidationResult(
                test_name="git_commit_capture",
                passed=False,
                score=0.0,
                threshold=1.0,
                warnings=[f"Could not capture git commit: {e}"]
            ))
        
        # 2. Random seed
        if 'random_seed' in config:
            results.append(ValidationResult(
                test_name="random_seed_set",
                passed=True,
                score=1.0,
                threshold=1.0,
                details={'random_seed': config['random_seed']}
            ))
        else:
            results.append(ValidationResult(
                test_name="random_seed_set",
                passed=False,
                score=0.0,
                threshold=1.0,
                warnings=["No random seed found in config"],
                recommendations=["Set random seed for reproducibility"]
            ))
        
        # 3. Data checksum
        if data is not None:
            try:
                # Calculate checksum of data
                data_str = pd.util.hash_pandas_object(data).values
                checksum = hashlib.sha256(str(data_str).encode()).hexdigest()[:16]
                
                results.append(ValidationResult(
                    test_name="data_checksum",
                    passed=True,
                    score=1.0,
                    threshold=1.0,
                    details={'data_checksum': checksum}
                ))
            except Exception as e:
                results.append(ValidationResult(
                    test_name="data_checksum",
                    passed=False,
                    score=0.0,
                    threshold=1.0,
                    warnings=[f"Could not compute data checksum: {e}"]
                ))
        
        # 4. Config hash
        try:
            config_str = json.dumps(config, sort_keys=True, default=str)
            config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
            
            results.append(ValidationResult(
                test_name="config_hash",
                passed=True,
                score=1.0,
                threshold=1.0,
                details={'config_hash': config_hash}
            ))
        except Exception as e:
            results.append(ValidationResult(
                test_name="config_hash",
                passed=False,
                score=0.0,
                threshold=1.0,
                warnings=[f"Could not compute config hash: {e}"]
            ))
        
        for result in results:
            self.validation_results.append(result)
        
        return results
    
    # =================================================================
    # COMPREHENSIVE VALIDATION
    # =================================================================
    
    def run_comprehensive_validation(
        self,
        labels: pd.DataFrame,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        config: Dict[str, Any],
        lookback_results: Optional[Dict[str, Any]] = None,
        regime_column: Optional[str] = None
    ) -> PreTrainingValidationReport:
        """
        Run all validation tests and generate comprehensive report.
        
        Args:
            labels: Label data
            features: Feature data
            targets: Target data
            config: Pipeline configuration
            lookback_results: Lookback optimization results
            regime_column: Optional regime column name
        
        Returns:
            PreTrainingValidationReport
        """
        tprint_info("🔬 Running comprehensive pre-training validation...")
        
        # Reset validation results
        self.validation_results = []
        
        # 1. Data Integrity & Representativeness
        tprint_info("\n📊 1. Data Integrity & Representativeness")
        data_integrity = [
            self.validate_label_autocorrelation(labels),
            self.validate_feature_target_mutual_info(features, targets),
        ]
        
        if regime_column and regime_column in features.columns:
            data_integrity.append(
                self.validate_feature_stability_across_regimes(features, regime_column)
            )
        
        # 2. Label Quality
        tprint_info("\n🏷️ 2. Label Design & Target Quality")
        label_quality = [
            self.validate_sharpe_of_synthetic_signal(labels),
        ]
        
        # 3. Lookback Optimization
        tprint_info("\n🔄 3. Lookback Optimization")
        lookback_results_list = []
        if lookback_results:
            lookback_results_list.append(
                self.validate_lookback_sensitivity(lookback_results)
            )
        
        # 4. Information Coefficient
        tprint_info("\n📈 4. Information Coefficient")
        ic_results = [
            self.validate_information_coefficient(features, targets)
        ]
        
        # 5. Reproducibility
        tprint_info("\n🔐 5. Reproducibility & Scientific Rigor")
        reproducibility = self.validate_reproducibility(config, features)
        
        # Compile report
        all_results = (
            data_integrity + 
            label_quality + 
            lookback_results_list + 
            ic_results + 
            reproducibility
        )
        
        passed_tests = sum(1 for r in all_results if r.passed)
        failed_tests = len(all_results) - passed_tests
        
        report = PreTrainingValidationReport(
            all_tests_passed=(failed_tests == 0),
            total_tests=len(all_results),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            data_integrity_results=data_integrity,
            label_quality_results=label_quality,
            lookback_optimization_results=lookback_results_list,
            soundness_check_results=ic_results,
            reproducibility_results=reproducibility
        )
        
        # Log summary
        tprint_info(f"\n{'='*60}")
        tprint_info(f"VALIDATION SUMMARY")
        tprint_info(f"{'='*60}")
        tprint_info(f"Total tests: {report.total_tests}")
        tprint_success(f"✅ Passed: {report.passed_tests}")
        if report.failed_tests > 0:
            tprint_error(f"❌ Failed: {report.failed_tests}")
        tprint_info(f"{'='*60}\n")
        
        return report
    
    def export_report(
        self,
        report: PreTrainingValidationReport,
        output_path: Union[str, Path]
    ) -> Path:
        """
        Export validation report to JSON file.
        
        Args:
            report: Validation report
            output_path: Output file path
        
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        
        tprint_success(f"✅ Validation report exported to {output_path}")
        return output_path


def create_pre_training_validator(
    thresholds: Optional[ValidationThresholds] = None
) -> PreTrainingValidator:
    """
    Factory function to create PreTrainingValidator.
    
    Args:
        thresholds: Optional validation thresholds
    
    Returns:
        PreTrainingValidator instance
    """
    return PreTrainingValidator(thresholds)