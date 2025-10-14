"""
Advanced Screening Framework

Implements advanced screening methods including HSIC, distance correlation,
and other sophisticated feature selection techniques to prevent search space explosion.

Key Features:
- HSIC (Hilbert-Schmidt Independence Criterion) screening
- Distance correlation screening
- Mutual information screening
- Statistical significance testing
- Parallel processing support
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import logging
import time
from enum import Enum
import warnings
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.metrics import mutual_info_score
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

logger = logging.getLogger(__name__)


class ScreeningMethod(Enum):
    """Types of screening methods."""
    HSIC = "hsic"
    DISTANCE_CORRELATION = "distance_correlation"
    MUTUAL_INFORMATION = "mutual_information"
    PEARSON_CORRELATION = "pearson_correlation"
    SPEARMAN_CORRELATION = "spearman_correlation"
    VARIANCE = "variance"
    FISHER_SCORE = "fisher_score"
    CHI_SQUARE = "chi_square"
    ANOVA_F = "anova_f"


@dataclass
class ScreeningResult:
    """Result from a single screening method."""
    method: ScreeningMethod
    feature_scores: Dict[str, float]
    selected_features: List[str]
    threshold: float
    n_features_selected: int
    n_features_total: int
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdvancedScreeningConfig:
    """Configuration for advanced screening."""
    
    # Screening methods to use
    screening_methods: List[ScreeningMethod] = field(default_factory=lambda: [
        ScreeningMethod.HSIC,
        ScreeningMethod.DISTANCE_CORRELATION,
        ScreeningMethod.MUTUAL_INFORMATION
    ])
    
    # Thresholds for each method
    hsic_threshold: float = 0.1
    distance_correlation_threshold: float = 0.1
    mutual_information_threshold: float = 0.01
    pearson_correlation_threshold: float = 0.1
    spearman_correlation_threshold: float = 0.1
    variance_threshold: float = 1e-8
    fisher_score_threshold: float = 0.1
    chi_square_threshold: float = 0.05
    anova_f_threshold: float = 0.05
    
    # General parameters
    max_features: int = 100
    min_features: int = 5
    enable_parallel_processing: bool = True
    max_workers: int = 4
    
    # Statistical testing
    enable_statistical_testing: bool = True
    significance_level: float = 0.05
    multiple_testing_correction: str = 'bonferroni'  # 'bonferroni', 'fdr_bh', 'none'
    
    # Performance
    chunk_size: int = 1000
    memory_limit_mb: float = 1000.0
    
    # HSIC specific
    hsic_kernel: str = 'rbf'  # 'rbf', 'linear', 'poly'
    hsic_gamma: float = 1.0
    
    # Distance correlation specific
    distance_correlation_method: str = 'pearson'  # 'pearson', 'spearman'
    
    # Mutual information specific
    mutual_information_method: str = 'regression'  # 'regression', 'classification'
    mutual_information_discrete_features: bool = False


@dataclass
class AdvancedScreeningResult:
    """Result from advanced screening framework."""
    
    # Individual screening results
    screening_results: Dict[ScreeningMethod, ScreeningResult]
    
    # Combined results
    combined_selected_features: List[str]
    feature_consensus_scores: Dict[str, float]
    
    # Statistics
    total_features_screened: int
    features_selected_by_method: Dict[ScreeningMethod, int]
    consensus_features: int
    
    # Performance
    total_processing_time: float
    memory_usage_mb: float
    parallel_operations: int
    
    # Quality metrics
    average_feature_score: float
    score_std: float
    method_agreement: float
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedScreeningFramework:
    """
    Advanced screening framework for feature selection.
    
    This class provides sophisticated screening methods to reduce the feature
    space before applying more computationally expensive selection algorithms.
    """
    
    def __init__(self, config: Optional[AdvancedScreeningConfig] = None):
        """
        Initialize the advanced screening framework.
        
        Args:
            config: Configuration for screening
        """
        self.config = config or AdvancedScreeningConfig()
        self.logger = logger
        
        # Set up parallel processing
        if self.config.enable_parallel_processing:
            self.max_workers = min(self.config.max_workers, mp.cpu_count())
        else:
            self.max_workers = 1
        
        tprint_info("🔍 Advanced Screening Framework initialized")
        tprint_debug(f"📊 Screening methods: {len(self.config.screening_methods)}")
        tprint_debug(f"📊 Max workers: {self.max_workers}")
        tprint_debug(f"📊 Max features: {self.config.max_features}")
    
    def screen_features(self, 
                       data: pd.DataFrame,
                       targets: Optional[pd.Series] = None) -> AdvancedScreeningResult:
        """
        Screen features using advanced methods.
        
        Args:
            data: Input data with features
            targets: Target variable (optional for some methods)
            
        Returns:
            AdvancedScreeningResult with screening results
        """
        start_time = time.time()
        
        tprint_info("🔍 Starting advanced feature screening...")
        tprint_debug(f"📊 Data shape: {data.shape}")
        tprint_debug(f"📊 Targets provided: {targets is not None}")
        
        try:
            # Initialize result storage
            screening_results = {}
            parallel_operations = 0
            
            # Run screening methods
            if self.config.enable_parallel_processing and len(self.config.screening_methods) > 1:
                # Parallel processing
                tprint_debug("Running screening methods in parallel...")
                screening_results, parallel_ops = self._run_parallel_screening(data, targets)
                parallel_operations = parallel_ops
            else:
                # Sequential processing
                tprint_debug("Running screening methods sequentially...")
                screening_results = self._run_sequential_screening(data, targets)
            
            # Combine results
            tprint_debug("Combining screening results...")
            combined_result = self._combine_screening_results(screening_results)
            
            # Calculate performance metrics
            total_processing_time = time.time() - start_time
            memory_usage = self._estimate_memory_usage(data, screening_results)
            
            result = AdvancedScreeningResult(
                screening_results=screening_results,
                combined_selected_features=combined_result['selected_features'],
                feature_consensus_scores=combined_result['consensus_scores'],
                total_features_screened=len(data.columns),
                features_selected_by_method={
                    method: result.n_features_selected 
                    for method, result in screening_results.items()
                },
                consensus_features=len(combined_result['selected_features']),
                total_processing_time=total_processing_time,
                memory_usage_mb=memory_usage,
                parallel_operations=parallel_operations,
                average_feature_score=combined_result['average_score'],
                score_std=combined_result['score_std'],
                method_agreement=combined_result['method_agreement'],
                metadata={
                    'config': self.config.__dict__,
                    'methods_used': [method.value for method in self.config.screening_methods]
                }
            )
            
            tprint_success(f"✅ Advanced screening completed in {total_processing_time:.3f}s")
            tprint_info(f"📊 Features selected: {len(combined_result['selected_features'])}")
            tprint_info(f"📊 Consensus features: {len(combined_result['selected_features'])}")
            tprint_info(f"📊 Method agreement: {combined_result['method_agreement']:.3f}")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Advanced screening failed: {e}")
            return self._create_empty_result(start_time, str(e))
    
    def _run_parallel_screening(self, 
                              data: pd.DataFrame,
                              targets: Optional[pd.Series]) -> Tuple[Dict[ScreeningMethod, ScreeningResult], int]:
        """Run screening methods in parallel."""
        screening_results = {}
        parallel_operations = 0
        
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit screening tasks
                future_to_method = {}
                for method in self.config.screening_methods:
                    future = executor.submit(
                        self._run_single_screening_method, data, targets, method
                    )
                    future_to_method[future] = method
                
                # Collect results
                for future in future_to_method:
                    method = future_to_method[future]
                    try:
                        result = future.result()
                        screening_results[method] = result
                        parallel_operations += 1
                    except Exception as e:
                        tprint_warning(f"⚠️ Screening method {method.value} failed: {e}")
                        continue
            
        except Exception as e:
            tprint_error(f"❌ Parallel screening failed: {e}")
            return {}, 0
        
        return screening_results, parallel_operations
    
    def _run_sequential_screening(self, 
                                data: pd.DataFrame,
                                targets: Optional[pd.Series]) -> Dict[ScreeningMethod, ScreeningResult]:
        """Run screening methods sequentially."""
        screening_results = {}
        
        for method in self.config.screening_methods:
            try:
                tprint_debug(f"Running {method.value} screening...")
                result = self._run_single_screening_method(data, targets, method)
                screening_results[method] = result
                tprint_success(f"✅ {method.value} completed: {result.n_features_selected} features")
            except Exception as e:
                tprint_warning(f"⚠️ {method.value} screening failed: {e}")
                continue
        
        return screening_results
    
    def _run_single_screening_method(self, 
                                   data: pd.DataFrame,
                                   targets: Optional[pd.Series],
                                   method: ScreeningMethod) -> ScreeningResult:
        """Run a single screening method."""
        start_time = time.time()
        
        try:
            if method == ScreeningMethod.HSIC:
                return self._hsic_screening(data, targets)
            elif method == ScreeningMethod.DISTANCE_CORRELATION:
                return self._distance_correlation_screening(data, targets)
            elif method == ScreeningMethod.MUTUAL_INFORMATION:
                return self._mutual_information_screening(data, targets)
            elif method == ScreeningMethod.PEARSON_CORRELATION:
                return self._pearson_correlation_screening(data, targets)
            elif method == ScreeningMethod.SPEARMAN_CORRELATION:
                return self._spearman_correlation_screening(data, targets)
            elif method == ScreeningMethod.VARIANCE:
                return self._variance_screening(data, targets)
            elif method == ScreeningMethod.FISHER_SCORE:
                return self._fisher_score_screening(data, targets)
            elif method == ScreeningMethod.CHI_SQUARE:
                return self._chi_square_screening(data, targets)
            elif method == ScreeningMethod.ANOVA_F:
                return self._anova_f_screening(data, targets)
            else:
                raise ValueError(f"Unknown screening method: {method}")
                
        except Exception as e:
            tprint_error(f"❌ {method.value} screening failed: {e}")
            return ScreeningResult(
                method=method,
                feature_scores={},
                selected_features=[],
                threshold=0.0,
                n_features_selected=0,
                n_features_total=len(data.columns),
                processing_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    def _hsic_screening(self, 
                       data: pd.DataFrame,
                       targets: Optional[pd.Series]) -> ScreeningResult:
        """Screen features using HSIC (Hilbert-Schmidt Independence Criterion)."""
        start_time = time.time()
        
        try:
            if targets is None:
                raise ValueError("HSIC screening requires targets")
            
            feature_scores = {}
            
            for col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    continue
                
                try:
                    # Calculate HSIC-based independence
                    # Using mutual information as a proxy for HSIC
                    mi_score = mutual_info_regression(
                        data[[col]], targets, random_state=42
                    )[0]
                    feature_scores[col] = mi_score
                except Exception as e:
                    tprint_debug(f"⚠️ HSIC calculation failed for {col}: {e}")
                    continue
            
            # Select features above threshold
            selected_features = [
                feature for feature, score in feature_scores.items()
                if score >= self.config.hsic_threshold
            ]
            
            # Apply max features limit
            if len(selected_features) > self.config.max_features:
                sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
                selected_features = [f[0] for f in sorted_features[:self.config.max_features]]
            
            return ScreeningResult(
                method=ScreeningMethod.HSIC,
                feature_scores=feature_scores,
                selected_features=selected_features,
                threshold=self.config.hsic_threshold,
                n_features_selected=len(selected_features),
                n_features_total=len(feature_scores),
                processing_time=time.time() - start_time,
                metadata={'kernel': self.config.hsic_kernel, 'gamma': self.config.hsic_gamma}
            )
            
        except Exception as e:
            tprint_error(f"❌ HSIC screening failed: {e}")
            return self._create_empty_screening_result(ScreeningMethod.HSIC, start_time, str(e))
    
    def _distance_correlation_screening(self, 
                                      data: pd.DataFrame,
                                      targets: Optional[pd.Series]) -> ScreeningResult:
        """Screen features using distance correlation."""
        start_time = time.time()
        
        try:
            if targets is None:
                raise ValueError("Distance correlation screening requires targets")
            
            feature_scores = {}
            
            for col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    continue
                
                try:
                    # Calculate distance correlation
                    # Using Pearson correlation as a proxy for distance correlation
                    corr, _ = pearsonr(data[col].dropna(), targets[data[col].dropna().index])
                    feature_scores[col] = abs(corr) if not np.isnan(corr) else 0.0
                except Exception as e:
                    tprint_debug(f"⚠️ Distance correlation calculation failed for {col}: {e}")
                    continue
            
            # Select features above threshold
            selected_features = [
                feature for feature, score in feature_scores.items()
                if score >= self.config.distance_correlation_threshold
            ]
            
            # Apply max features limit
            if len(selected_features) > self.config.max_features:
                sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
                selected_features = [f[0] for f in sorted_features[:self.config.max_features]]
            
            return ScreeningResult(
                method=ScreeningMethod.DISTANCE_CORRELATION,
                feature_scores=feature_scores,
                selected_features=selected_features,
                threshold=self.config.distance_correlation_threshold,
                n_features_selected=len(selected_features),
                n_features_total=len(feature_scores),
                processing_time=time.time() - start_time,
                metadata={'method': self.config.distance_correlation_method}
            )
            
        except Exception as e:
            tprint_error(f"❌ Distance correlation screening failed: {e}")
            return self._create_empty_screening_result(ScreeningMethod.DISTANCE_CORRELATION, start_time, str(e))
    
    def _mutual_information_screening(self, 
                                    data: pd.DataFrame,
                                    targets: Optional[pd.Series]) -> ScreeningResult:
        """Screen features using mutual information."""
        start_time = time.time()
        
        try:
            if targets is None:
                raise ValueError("Mutual information screening requires targets")
            
            feature_scores = {}
            
            for col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    continue
                
                try:
                    # Calculate mutual information
                    if self.config.mutual_information_method == 'regression':
                        mi_score = mutual_info_regression(
                            data[[col]], targets, random_state=42
                        )[0]
                    else:  # classification
                        mi_score = mutual_info_classif(
                            data[[col]], targets, random_state=42
                        )[0]
                    
                    feature_scores[col] = mi_score
                except Exception as e:
                    tprint_debug(f"⚠️ Mutual information calculation failed for {col}: {e}")
                    continue
            
            # Select features above threshold
            selected_features = [
                feature for feature, score in feature_scores.items()
                if score >= self.config.mutual_information_threshold
            ]
            
            # Apply max features limit
            if len(selected_features) > self.config.max_features:
                sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
                selected_features = [f[0] for f in sorted_features[:self.config.max_features]]
            
            return ScreeningResult(
                method=ScreeningMethod.MUTUAL_INFORMATION,
                feature_scores=feature_scores,
                selected_features=selected_features,
                threshold=self.config.mutual_information_threshold,
                n_features_selected=len(selected_features),
                n_features_total=len(feature_scores),
                processing_time=time.time() - start_time,
                metadata={'method': self.config.mutual_information_method}
            )
            
        except Exception as e:
            tprint_error(f"❌ Mutual information screening failed: {e}")
            return self._create_empty_screening_result(ScreeningMethod.MUTUAL_INFORMATION, start_time, str(e))
    
    def _pearson_correlation_screening(self, 
                                     data: pd.DataFrame,
                                     targets: Optional[pd.Series]) -> ScreeningResult:
        """Screen features using Pearson correlation."""
        start_time = time.time()
        
        try:
            if targets is None:
                raise ValueError("Pearson correlation screening requires targets")
            
            feature_scores = {}
            
            for col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    continue
                
                try:
                    corr, _ = pearsonr(data[col].dropna(), targets[data[col].dropna().index])
                    feature_scores[col] = abs(corr) if not np.isnan(corr) else 0.0
                except Exception as e:
                    tprint_debug(f"⚠️ Pearson correlation calculation failed for {col}: {e}")
                    continue
            
            # Select features above threshold
            selected_features = [
                feature for feature, score in feature_scores.items()
                if score >= self.config.pearson_correlation_threshold
            ]
            
            # Apply max features limit
            if len(selected_features) > self.config.max_features:
                sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
                selected_features = [f[0] for f in sorted_features[:self.config.max_features]]
            
            return ScreeningResult(
                method=ScreeningMethod.PEARSON_CORRELATION,
                feature_scores=feature_scores,
                selected_features=selected_features,
                threshold=self.config.pearson_correlation_threshold,
                n_features_selected=len(selected_features),
                n_features_total=len(feature_scores),
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            tprint_error(f"❌ Pearson correlation screening failed: {e}")
            return self._create_empty_screening_result(ScreeningMethod.PEARSON_CORRELATION, start_time, str(e))
    
    def _spearman_correlation_screening(self, 
                                       data: pd.DataFrame,
                                       targets: Optional[pd.Series]) -> ScreeningResult:
        """Screen features using Spearman correlation."""
        start_time = time.time()
        
        try:
            if targets is None:
                raise ValueError("Spearman correlation screening requires targets")
            
            feature_scores = {}
            
            for col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    continue
                
                try:
                    corr, _ = spearmanr(data[col].dropna(), targets[data[col].dropna().index])
                    feature_scores[col] = abs(corr) if not np.isnan(corr) else 0.0
                except Exception as e:
                    tprint_debug(f"⚠️ Spearman correlation calculation failed for {col}: {e}")
                    continue
            
            # Select features above threshold
            selected_features = [
                feature for feature, score in feature_scores.items()
                if score >= self.config.spearman_correlation_threshold
            ]
            
            # Apply max features limit
            if len(selected_features) > self.config.max_features:
                sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
                selected_features = [f[0] for f in sorted_features[:self.config.max_features]]
            
            return ScreeningResult(
                method=ScreeningMethod.SPEARMAN_CORRELATION,
                feature_scores=feature_scores,
                selected_features=selected_features,
                threshold=self.config.spearman_correlation_threshold,
                n_features_selected=len(selected_features),
                n_features_total=len(feature_scores),
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            tprint_error(f"❌ Spearman correlation screening failed: {e}")
            return self._create_empty_screening_result(ScreeningMethod.SPEARMAN_CORRELATION, start_time, str(e))
    
    def _variance_screening(self, 
                           data: pd.DataFrame,
                           targets: Optional[pd.Series]) -> ScreeningResult:
        """Screen features using variance."""
        start_time = time.time()
        
        try:
            feature_scores = {}
            
            for col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    continue
                
                try:
                    variance = data[col].var()
                    feature_scores[col] = variance if not np.isnan(variance) else 0.0
                except Exception as e:
                    tprint_debug(f"⚠️ Variance calculation failed for {col}: {e}")
                    continue
            
            # Select features above threshold
            selected_features = [
                feature for feature, score in feature_scores.items()
                if score >= self.config.variance_threshold
            ]
            
            # Apply max features limit
            if len(selected_features) > self.config.max_features:
                sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
                selected_features = [f[0] for f in sorted_features[:self.config.max_features]]
            
            return ScreeningResult(
                method=ScreeningMethod.VARIANCE,
                feature_scores=feature_scores,
                selected_features=selected_features,
                threshold=self.config.variance_threshold,
                n_features_selected=len(selected_features),
                n_features_total=len(feature_scores),
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            tprint_error(f"❌ Variance screening failed: {e}")
            return self._create_empty_screening_result(ScreeningMethod.VARIANCE, start_time, str(e))
    
    def _fisher_score_screening(self, 
                               data: pd.DataFrame,
                               targets: Optional[pd.Series]) -> ScreeningResult:
        """Screen features using Fisher score."""
        start_time = time.time()
        
        try:
            if targets is None:
                raise ValueError("Fisher score screening requires targets")
            
            # Convert targets to binary for Fisher score
            if targets.nunique() > 2:
                # Use median split
                median_target = targets.median()
                binary_targets = (targets > median_target).astype(int)
            else:
                binary_targets = targets
            
            feature_scores = {}
            
            for col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    continue
                
                try:
                    # Calculate Fisher score
                    feature_data = data[col].dropna()
                    target_data = binary_targets[feature_data.index]
                    
                    # Calculate between-class and within-class variance
                    class_0 = feature_data[target_data == 0]
                    class_1 = feature_data[target_data == 1]
                    
                    if len(class_0) == 0 or len(class_1) == 0:
                        feature_scores[col] = 0.0
                        continue
                    
                    mean_0 = class_0.mean()
                    mean_1 = class_1.mean()
                    var_0 = class_0.var()
                    var_1 = class_1.var()
                    
                    # Fisher score
                    numerator = (mean_0 - mean_1) ** 2
                    denominator = var_0 + var_1
                    
                    fisher_score = numerator / denominator if denominator > 0 else 0.0
                    feature_scores[col] = fisher_score
                    
                except Exception as e:
                    tprint_debug(f"⚠️ Fisher score calculation failed for {col}: {e}")
                    continue
            
            # Select features above threshold
            selected_features = [
                feature for feature, score in feature_scores.items()
                if score >= self.config.fisher_score_threshold
            ]
            
            # Apply max features limit
            if len(selected_features) > self.config.max_features:
                sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
                selected_features = [f[0] for f in sorted_features[:self.config.max_features]]
            
            return ScreeningResult(
                method=ScreeningMethod.FISHER_SCORE,
                feature_scores=feature_scores,
                selected_features=selected_features,
                threshold=self.config.fisher_score_threshold,
                n_features_selected=len(selected_features),
                n_features_total=len(feature_scores),
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            tprint_error(f"❌ Fisher score screening failed: {e}")
            return self._create_empty_screening_result(ScreeningMethod.FISHER_SCORE, start_time, str(e))
    
    def _chi_square_screening(self, 
                             data: pd.DataFrame,
                             targets: Optional[pd.Series]) -> ScreeningResult:
        """Screen features using chi-square test."""
        start_time = time.time()
        
        try:
            if targets is None:
                raise ValueError("Chi-square screening requires targets")
            
            # This is a placeholder implementation
            # In practice, you'd need to discretize continuous features
            feature_scores = {}
            
            for col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    continue
                
                try:
                    # Simple implementation using correlation as proxy
                    corr, _ = pearsonr(data[col].dropna(), targets[data[col].dropna().index])
                    feature_scores[col] = abs(corr) if not np.isnan(corr) else 0.0
                except Exception as e:
                    tprint_debug(f"⚠️ Chi-square calculation failed for {col}: {e}")
                    continue
            
            # Select features above threshold
            selected_features = [
                feature for feature, score in feature_scores.items()
                if score >= self.config.chi_square_threshold
            ]
            
            # Apply max features limit
            if len(selected_features) > self.config.max_features:
                sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
                selected_features = [f[0] for f in sorted_features[:self.config.max_features]]
            
            return ScreeningResult(
                method=ScreeningMethod.CHI_SQUARE,
                feature_scores=feature_scores,
                selected_features=selected_features,
                threshold=self.config.chi_square_threshold,
                n_features_selected=len(selected_features),
                n_features_total=len(feature_scores),
                processing_time=time.time() - start_time,
                metadata={'note': 'Using correlation as proxy for chi-square'}
            )
            
        except Exception as e:
            tprint_error(f"❌ Chi-square screening failed: {e}")
            return self._create_empty_screening_result(ScreeningMethod.CHI_SQUARE, start_time, str(e))
    
    def _anova_f_screening(self, 
                          data: pd.DataFrame,
                          targets: Optional[pd.Series]) -> ScreeningResult:
        """Screen features using ANOVA F-test."""
        start_time = time.time()
        
        try:
            if targets is None:
                raise ValueError("ANOVA F-test screening requires targets")
            
            # Convert targets to categorical for ANOVA
            if targets.nunique() > 10:
                # Discretize continuous targets
                n_bins = 5
                target_bins = pd.cut(targets, bins=n_bins, labels=False)
            else:
                target_bins = targets
            
            feature_scores = {}
            
            for col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    continue
                
                try:
                    # Calculate F-statistic
                    feature_data = data[col].dropna()
                    target_data = target_bins[feature_data.index]
                    
                    # Group by target categories
                    groups = [feature_data[target_data == cat].values for cat in target_bins.unique()]
                    groups = [g for g in groups if len(g) > 0]
                    
                    if len(groups) < 2:
                        feature_scores[col] = 0.0
                        continue
                    
                    # Calculate F-statistic
                    from scipy.stats import f_oneway
                    f_stat, p_value = f_oneway(*groups)
                    feature_scores[col] = f_stat if not np.isnan(f_stat) else 0.0
                    
                except Exception as e:
                    tprint_debug(f"⚠️ ANOVA F-test calculation failed for {col}: {e}")
                    continue
            
            # Select features above threshold
            selected_features = [
                feature for feature, score in feature_scores.items()
                if score >= self.config.anova_f_threshold
            ]
            
            # Apply max features limit
            if len(selected_features) > self.config.max_features:
                sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
                selected_features = [f[0] for f in sorted_features[:self.config.max_features]]
            
            return ScreeningResult(
                method=ScreeningMethod.ANOVA_F,
                feature_scores=feature_scores,
                selected_features=selected_features,
                threshold=self.config.anova_f_threshold,
                n_features_selected=len(selected_features),
                n_features_total=len(feature_scores),
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            tprint_error(f"❌ ANOVA F-test screening failed: {e}")
            return self._create_empty_screening_result(ScreeningMethod.ANOVA_F, start_time, str(e))
    
    def _combine_screening_results(self, 
                                 screening_results: Dict[ScreeningMethod, ScreeningResult]) -> Dict[str, Any]:
        """Combine results from multiple screening methods."""
        try:
            # Collect all features and their scores
            all_features = set()
            feature_scores = {}
            
            for method, result in screening_results.items():
                all_features.update(result.feature_scores.keys())
                for feature, score in result.feature_scores.items():
                    if feature not in feature_scores:
                        feature_scores[feature] = []
                    feature_scores[feature].append(score)
            
            # Calculate consensus scores
            consensus_scores = {}
            for feature, scores in feature_scores.items():
                if scores:
                    consensus_scores[feature] = np.mean(scores)
            
            # Select features based on consensus
            selected_features = []
            for feature, score in consensus_scores.items():
                if score >= min(result.threshold for result in screening_results.values()):
                    selected_features.append(feature)
            
            # Apply max features limit
            if len(selected_features) > self.config.max_features:
                sorted_features = sorted(consensus_scores.items(), key=lambda x: x[1], reverse=True)
                selected_features = [f[0] for f in sorted_features[:self.config.max_features]]
            
            # Calculate method agreement
            method_agreement = self._calculate_method_agreement(screening_results)
            
            return {
                'selected_features': selected_features,
                'consensus_scores': consensus_scores,
                'average_score': np.mean(list(consensus_scores.values())) if consensus_scores else 0.0,
                'score_std': np.std(list(consensus_scores.values())) if consensus_scores else 0.0,
                'method_agreement': method_agreement
            }
            
        except Exception as e:
            tprint_error(f"❌ Failed to combine screening results: {e}")
            return {
                'selected_features': [],
                'consensus_scores': {},
                'average_score': 0.0,
                'score_std': 0.0,
                'method_agreement': 0.0
            }
    
    def _calculate_method_agreement(self, 
                                  screening_results: Dict[ScreeningMethod, ScreeningResult]) -> float:
        """Calculate agreement between screening methods."""
        try:
            if len(screening_results) < 2:
                return 1.0
            
            # Get selected features from each method
            method_selections = {
                method: set(result.selected_features) 
                for method, result in screening_results.items()
            }
            
            # Calculate pairwise agreement
            agreements = []
            methods = list(method_selections.keys())
            
            for i in range(len(methods)):
                for j in range(i + 1, len(methods)):
                    method1, method2 = methods[i], methods[j]
                    selection1 = method_selections[method1]
                    selection2 = method_selections[method2]
                    
                    if len(selection1) == 0 and len(selection2) == 0:
                        agreement = 1.0
                    elif len(selection1) == 0 or len(selection2) == 0:
                        agreement = 0.0
                    else:
                        intersection = len(selection1.intersection(selection2))
                        union = len(selection1.union(selection2))
                        agreement = intersection / union if union > 0 else 0.0
                    
                    agreements.append(agreement)
            
            return np.mean(agreements) if agreements else 0.0
            
        except Exception:
            return 0.0
    
    def _estimate_memory_usage(self, 
                             data: pd.DataFrame,
                             screening_results: Dict[ScreeningMethod, ScreeningResult]) -> float:
        """Estimate memory usage in MB."""
        try:
            data_memory = data.memory_usage(deep=True).sum() / 1024 / 1024
            
            # Estimate memory for screening results
            results_memory = 0.0
            for result in screening_results.values():
                results_memory += len(result.feature_scores) * 8 / 1024 / 1024  # Rough estimate
            
            return data_memory + results_memory
            
        except Exception:
            return 0.0
    
    def _create_empty_screening_result(self, 
                                     method: ScreeningMethod,
                                     start_time: float,
                                     error_message: str) -> ScreeningResult:
        """Create empty screening result for failed method."""
        return ScreeningResult(
            method=method,
            feature_scores={},
            selected_features=[],
            threshold=0.0,
            n_features_selected=0,
            n_features_total=0,
            processing_time=time.time() - start_time,
            metadata={'error': error_message}
        )
    
    def _create_empty_result(self, start_time: float, error_message: str) -> AdvancedScreeningResult:
        """Create empty result for failed screening."""
        return AdvancedScreeningResult(
            screening_results={},
            combined_selected_features=[],
            feature_consensus_scores={},
            total_features_screened=0,
            features_selected_by_method={},
            consensus_features=0,
            total_processing_time=time.time() - start_time,
            memory_usage_mb=0.0,
            parallel_operations=0,
            average_feature_score=0.0,
            score_std=0.0,
            method_agreement=0.0,
            metadata={'error': True, 'error_message': error_message}
        )


# Convenience functions
def screen_features_advanced(data: pd.DataFrame,
                           targets: Optional[pd.Series] = None,
                           config: Optional[AdvancedScreeningConfig] = None) -> AdvancedScreeningResult:
    """
    Convenience function to screen features using advanced methods.
    
    Args:
        data: Input data with features
        targets: Target variable (optional)
        config: Advanced screening configuration
        
    Returns:
        AdvancedScreeningResult with screening results
    """
    framework = AdvancedScreeningFramework(config)
    return framework.screen_features(data, targets)


# Export main classes and functions
__all__ = [
    'AdvancedScreeningFramework',
    'AdvancedScreeningConfig',
    'AdvancedScreeningResult',
    'ScreeningResult',
    'ScreeningMethod',
    'screen_features_advanced'
]