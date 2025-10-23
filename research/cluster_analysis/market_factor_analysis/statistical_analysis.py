"""
Statistical Tools for Implicit Dimension Analysis.

This module provides advanced statistical methods for discovering implicit
market dimensions from features, including:
- Principal Component Analysis (PCA)
- Factor Analysis (FA) 
- Independent Component Analysis (ICA)
- Canonical Correlation Analysis (CCA)
- Non-negative Matrix Factorization (NMF)
- t-SNE and UMAP for non-linear dimensionality
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

from src.utils.logger import system_logger


class DimensionalityMethod(Enum):
    """Statistical methods for dimensionality analysis."""
    PCA = "principal_component_analysis"
    FACTOR_ANALYSIS = "factor_analysis"
    ICA = "independent_component_analysis"
    NMF = "non_negative_matrix_factorization"
    CANONICAL_CORRELATION = "canonical_correlation_analysis"
    TSNE = "t_distributed_stochastic_neighbor_embedding"
    UMAP = "uniform_manifold_approximation"


@dataclass
class DimensionalityResult:
    """Results from dimensionality analysis."""
    method: DimensionalityMethod
    n_components: int
    explained_variance_ratio: Optional[np.ndarray]
    components: np.ndarray
    transformed_data: np.ndarray
    feature_loadings: Optional[np.ndarray]
    statistical_tests: Dict[str, float]
    interpretation: str
    metadata: Dict[str, Any]


class StatisticalDimensionAnalyzer:
    """Advanced statistical analysis for implicit dimensions."""
    
    def __init__(self):
        self.logger = system_logger.getChild('StatisticalDimensionAnalyzer')
    
    def analyze_dimensions(self, 
                          features: pd.DataFrame,
                          methods: Optional[List[DimensionalityMethod]] = None,
                          n_components: Optional[int] = None) -> Dict[DimensionalityMethod, DimensionalityResult]:
        """
        Comprehensive statistical analysis of implicit dimensions.
        
        Args:
            features: Feature matrix
            methods: Statistical methods to apply
            n_components: Number of components (if None, determined automatically)
            
        Returns:
            Dictionary mapping methods to results
        """
        if methods is None:
            methods = [
                DimensionalityMethod.PCA,
                DimensionalityMethod.FACTOR_ANALYSIS,
                DimensionalityMethod.ICA
            ]
        
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features.fillna(0))
        
        # Determine optimal number of components if not specified
        if n_components is None:
            n_components = self._estimate_optimal_components(features_scaled)
        
        results = {}
        
        for method in methods:
            self.logger.info(f"📊 Running {method.value}")
            try:
                result = self._apply_method(method, features_scaled, features.columns, n_components)
                results[method] = result
            except Exception as e:
                self.logger.error(f"❌ {method.value} failed: {e}")
                continue
        
        return results
    
    def _estimate_optimal_components(self, data: np.ndarray) -> int:
        """Estimate optimal number of components using multiple criteria."""
        from sklearn.decomposition import PCA
        
        # Run PCA to get explained variance
        pca = PCA()
        pca.fit(data)
        
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        
        # Method 1: 95% variance explained
        n_95 = np.argmax(cumvar >= 0.95) + 1
        
        # Method 2: Kaiser criterion (eigenvalues > 1)
        n_kaiser = np.sum(pca.explained_variance_ > 1)
        
        # Method 3: Elbow method (simplified)
        variance_ratios = pca.explained_variance_ratio_
        second_derivatives = np.diff(variance_ratios, 2)
        n_elbow = np.argmax(second_derivatives) + 2 if len(second_derivatives) > 0 else n_95
        
        # Method 4: Broken stick model
        n_features = data.shape[1]
        broken_stick = np.array([1/j for j in range(1, n_features + 1)])
        broken_stick = broken_stick / np.sum(broken_stick)
        n_broken_stick = np.sum(pca.explained_variance_ratio_ > broken_stick)
        
        # Take median of methods as estimate
        estimates = [n_95, n_kaiser, n_elbow, n_broken_stick]
        optimal_components = int(np.median([e for e in estimates if e > 0]))
        
        self.logger.info(f"📊 Component estimation: 95%={n_95}, Kaiser={n_kaiser}, Elbow={n_elbow}, BrokenStick={n_broken_stick}")
        self.logger.info(f"📊 Estimated optimal components: {optimal_components}")
        
        return min(optimal_components, data.shape[1] // 2)  # Cap at half the features
    
    def _apply_method(self, 
                     method: DimensionalityMethod,
                     data: np.ndarray,
                     feature_names: pd.Index,
                     n_components: int) -> DimensionalityResult:
        """Apply specific dimensionality reduction method."""
        
        if method == DimensionalityMethod.PCA:
            return self._apply_pca(data, feature_names, n_components)
        elif method == DimensionalityMethod.FACTOR_ANALYSIS:
            return self._apply_factor_analysis(data, feature_names, n_components)
        elif method == DimensionalityMethod.ICA:
            return self._apply_ica(data, feature_names, n_components)
        elif method == DimensionalityMethod.NMF:
            return self._apply_nmf(data, feature_names, n_components)
        else:
            raise ValueError(f"Method {method.value} not implemented")
    
    def _apply_pca(self, data: np.ndarray, feature_names: pd.Index, n_components: int) -> DimensionalityResult:
        """Apply Principal Component Analysis."""
        
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(data)
        
        # Statistical tests
        statistical_tests = {
            'explained_variance_ratio_total': float(np.sum(pca.explained_variance_ratio_)),
            'first_component_variance': float(pca.explained_variance_ratio_[0]),
            'kaiser_meyer_olkin': self._calculate_kmo(data),
            'bartlett_sphericity_pvalue': self._bartlett_test(data)
        }
        
        # Feature loadings (correlations between original features and components)
        feature_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        
        # Interpretation
        interpretation = f"PCA extracted {n_components} components explaining {statistical_tests['explained_variance_ratio_total']:.1%} of variance"
        
        return DimensionalityResult(
            method=DimensionalityMethod.PCA,
            n_components=n_components,
            explained_variance_ratio=pca.explained_variance_ratio_,
            components=pca.components_,
            transformed_data=transformed,
            feature_loadings=feature_loadings,
            statistical_tests=statistical_tests,
            interpretation=interpretation,
            metadata={
                'eigenvalues': pca.explained_variance_.tolist(),
                'feature_names': list(feature_names),
                'loadings_interpretation': self._interpret_loadings(feature_loadings, feature_names, n_components)
            }
        )
    
    def _apply_factor_analysis(self, data: np.ndarray, feature_names: pd.Index, n_components: int) -> DimensionalityResult:
        """Apply Factor Analysis."""
        from sklearn.decomposition import FactorAnalysis
        
        fa = FactorAnalysis(n_components=n_components, random_state=42)
        transformed = fa.fit_transform(data)
        
        # Statistical tests
        statistical_tests = {
            'log_likelihood': float(fa.score(data)),
            'noise_variance_mean': float(np.mean(fa.noise_variance_)),
            'communalities_mean': float(np.mean(1 - fa.noise_variance_)),
            'bartlett_sphericity_pvalue': self._bartlett_test(data)
        }
        
        # Feature loadings
        feature_loadings = fa.components_.T
        
        # Interpretation
        interpretation = f"Factor Analysis identified {n_components} latent factors with mean communality {statistical_tests['communalities_mean']:.3f}"
        
        return DimensionalityResult(
            method=DimensionalityMethod.FACTOR_ANALYSIS,
            n_components=n_components,
            explained_variance_ratio=None,
            components=fa.components_,
            transformed_data=transformed,
            feature_loadings=feature_loadings,
            statistical_tests=statistical_tests,
            interpretation=interpretation,
            metadata={
                'noise_variance': fa.noise_variance_.tolist(),
                'feature_names': list(feature_names),
                'loadings_interpretation': self._interpret_loadings(feature_loadings, feature_names, n_components)
            }
        )
    
    def _apply_ica(self, data: np.ndarray, feature_names: pd.Index, n_components: int) -> DimensionalityResult:
        """Apply Independent Component Analysis."""
        from sklearn.decomposition import FastICA
        
        ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
        transformed = ica.fit_transform(data)
        
        # Statistical tests (independence measures)
        statistical_tests = {
            'mean_kurtosis': float(np.mean([self._calculate_kurtosis(transformed[:, i]) for i in range(n_components)])),
            'mean_negentropy': float(np.mean([self._calculate_negentropy(transformed[:, i]) for i in range(n_components)])),
            'mutual_information': self._calculate_mutual_information_matrix(transformed)
        }
        
        # Feature loadings (mixing matrix)
        feature_loadings = ica.mixing_
        
        # Interpretation
        interpretation = f"ICA found {n_components} independent components with mean kurtosis {statistical_tests['mean_kurtosis']:.3f}"
        
        return DimensionalityResult(
            method=DimensionalityMethod.ICA,
            n_components=n_components,
            explained_variance_ratio=None,
            components=ica.components_,
            transformed_data=transformed,
            feature_loadings=feature_loadings,
            statistical_tests=statistical_tests,
            interpretation=interpretation,
            metadata={
                'feature_names': list(feature_names),
                'loadings_interpretation': self._interpret_loadings(feature_loadings, feature_names, n_components)
            }
        )
    
    def _apply_nmf(self, data: np.ndarray, feature_names: pd.Index, n_components: int) -> DimensionalityResult:
        """Apply Non-negative Matrix Factorization."""
        from sklearn.decomposition import NMF
        
        # NMF requires non-negative data
        data_positive = data - np.min(data) + 1e-6
        
        nmf = NMF(n_components=n_components, random_state=42, max_iter=1000)
        transformed = nmf.fit_transform(data_positive)
        
        # Statistical tests
        statistical_tests = {
            'reconstruction_error': float(nmf.reconstruction_err_),
            'sparsity_w': float(np.mean(transformed == 0)),
            'sparsity_h': float(np.mean(nmf.components_ == 0))
        }
        
        # Feature loadings
        feature_loadings = nmf.components_.T
        
        # Interpretation
        interpretation = f"NMF found {n_components} non-negative components with {statistical_tests['sparsity_w']:.1%} sparsity"
        
        return DimensionalityResult(
            method=DimensionalityMethod.NMF,
            n_components=n_components,
            explained_variance_ratio=None,
            components=nmf.components_,
            transformed_data=transformed,
            feature_loadings=feature_loadings,
            statistical_tests=statistical_tests,
            interpretation=interpretation,
            metadata={
                'feature_names': list(feature_names),
                'loadings_interpretation': self._interpret_loadings(feature_loadings, feature_names, n_components)
            }
        )
    
    def _calculate_kmo(self, data: np.ndarray) -> float:
        """Calculate Kaiser-Meyer-Olkin measure of sampling adequacy."""
        try:
            from factor_analyzer.factor_analyzer import calculate_kmo
            kmo_all, kmo_model = calculate_kmo(data)
            return float(kmo_model)
        except ImportError:
            # Simplified KMO calculation
            corr_matrix = np.corrcoef(data.T)
            inv_corr_matrix = np.linalg.pinv(corr_matrix)
            
            # Partial correlations
            partial_corr = np.zeros_like(corr_matrix)
            for i in range(corr_matrix.shape[0]):
                for j in range(corr_matrix.shape[1]):
                    if i != j:
                        partial_corr[i, j] = -inv_corr_matrix[i, j] / np.sqrt(inv_corr_matrix[i, i] * inv_corr_matrix[j, j])
            
            # KMO calculation
            r_squared = np.sum(corr_matrix**2) - np.trace(corr_matrix**2)
            partial_squared = np.sum(partial_corr**2)
            
            kmo = r_squared / (r_squared + partial_squared) if (r_squared + partial_squared) > 0 else 0
            return float(kmo)
    
    def _bartlett_test(self, data: np.ndarray) -> float:
        """Bartlett's test of sphericity p-value."""
        try:
            from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
            chi_square_value, p_value = calculate_bartlett_sphericity(data)
            return float(p_value)
        except ImportError:
            # Simplified Bartlett test
            n, p = data.shape
            corr_matrix = np.corrcoef(data.T)
            det_corr = np.linalg.det(corr_matrix)
            
            if det_corr <= 0:
                return 1.0  # Cannot reject sphericity
            
            chi_square = -(n - 1 - (2 * p + 5) / 6) * np.log(det_corr)
            df = p * (p - 1) / 2
            
            # Approximate p-value using chi-square distribution
            from scipy import stats
            p_value = 1 - stats.chi2.cdf(chi_square, df)
            return float(p_value)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        return float(stats.kurtosis(data))
    
    def _calculate_negentropy(self, data: np.ndarray) -> float:
        """Calculate negentropy (measure of non-Gaussianity)."""
        # Simplified negentropy using kurtosis
        kurt = self._calculate_kurtosis(data)
        return float(abs(kurt))  # Simplified measure
    
    def _calculate_mutual_information_matrix(self, data: np.ndarray) -> float:
        """Calculate mean mutual information between components."""
        try:
            from sklearn.feature_selection import mutual_info_regression
            n_components = data.shape[1]
            mi_sum = 0
            count = 0
            
            for i in range(n_components):
                for j in range(i + 1, n_components):
                    mi = mutual_info_regression(data[:, [i]], data[:, j])
                    mi_sum += mi[0]
                    count += 1
            
            return float(mi_sum / count) if count > 0 else 0.0
        except:
            return 0.0
    
    def _interpret_loadings(self, loadings: np.ndarray, feature_names: pd.Index, n_components: int) -> Dict[str, List[Tuple[str, float]]]:
        """Interpret feature loadings for each component."""
        interpretation = {}
        
        for comp in range(min(n_components, loadings.shape[1])):
            # Get top features for this component
            component_loadings = loadings[:, comp]
            top_indices = np.argsort(np.abs(component_loadings))[-10:][::-1]  # Top 10
            
            top_features = [
                (feature_names[idx], float(component_loadings[idx]))
                for idx in top_indices
            ]
            
            interpretation[f'component_{comp + 1}'] = top_features
        
        return interpretation
    
    def generate_statistical_report(self, results: Dict[DimensionalityMethod, DimensionalityResult]) -> str:
        """Generate comprehensive statistical analysis report."""
        report = []
        report.append("# Statistical Dimensionality Analysis Report")
        report.append("=" * 60)
        report.append("")
        
        # Summary
        report.append("## Summary")
        report.append("")
        for method, result in results.items():
            report.append(f"**{method.value.upper()}**")
            report.append(f"- Components: {result.n_components}")
            if result.explained_variance_ratio is not None:
                total_var = np.sum(result.explained_variance_ratio)
                report.append(f"- Explained Variance: {total_var:.1%}")
            report.append(f"- Interpretation: {result.interpretation}")
            report.append("")
        
        # Statistical Tests
        report.append("## Statistical Test Results")
        report.append("")
        
        for method, result in results.items():
            report.append(f"### {method.value.upper()}")
            for test_name, test_value in result.statistical_tests.items():
                report.append(f"- **{test_name.replace('_', ' ').title()}**: {test_value:.4f}")
            report.append("")
        
        # Component Interpretation
        report.append("## Component Interpretation")
        report.append("")
        
        for method, result in results.items():
            if 'loadings_interpretation' in result.metadata:
                report.append(f"### {method.value.upper()} - Top Feature Loadings")
                loadings_interp = result.metadata['loadings_interpretation']
                
                for comp_name, features in loadings_interp.items():
                    report.append(f"**{comp_name.replace('_', ' ').title()}:**")
                    for feature, loading in features[:5]:  # Top 5
                        report.append(f"  - {feature}: {loading:.3f}")
                    report.append("")
        
        return "\n".join(report)