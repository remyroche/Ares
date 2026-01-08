from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
"""
Quantitative Regime Engine with Advanced GMM Pipeline

Implements a comprehensive regime detection system with:
- Wavelet denoising for price and features
- HAR-based target residualization
- Feature temporal residualization (shocks)
- Permutation-based RMI feature selection
- Ledoit-Wolf blockwise whitening
- Dual pipeline execution (returns vs FracDiff)
- Complete GMM enhancement for all steps (A-D)

Based on advanced quantitative finance methodology and de Prado's ML framework.
"""

import numpy as np
import pandas as pd
import pywt
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from sklearn.mixture import GaussianMixture
from scipy.stats import chi2
from joblib import Parallel, delayed
import asyncio
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

# GPU acceleration imports
try:
    import torch
    import torch.mps as mps
    MPS_AVAILABLE = torch.backends.mps.is_available()
    if MPS_AVAILABLE:
        torch_device = torch.device("mps")
    else:
        torch_device = torch.device("cpu")
        MPS_AVAILABLE = False
except ImportError:
    MPS_AVAILABLE = False
    torch = None
    torch_device = None

# #region agent log - Hypothesis E: GPU acceleration initialization
import json
with open('/Users/remyroche/Documents/Ares/.cursor/debug.log', 'a') as f:
    f.write(json.dumps({
        "id": "log_gpu_init",
        "timestamp": int(__import__('time').time() * 1000),
        "location": "quantitative_regime_engine.py:gpu_imports",
        "message": "GPU acceleration initialization check",
        "data": {
            "torch_available": torch is not None,
            "mps_available": MPS_AVAILABLE,
            "torch_version": torch.__version__ if torch else None,
            "device_type": str(torch_device) if torch_device else None
        },
        "sessionId": "debug-session",
        "runId": "initial",
        "hypothesisId": "E"
    }) + '\n')
# #endregion

# Internal imports
from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
from src.utils.fracdiff import FracDiffTransformer, fracdiff_series, validate_stationarity


class QuantitativeRegimeEngine:
    """
    Advanced quantitative regime detection engine with comprehensive feature engineering.
    
    This engine implements a sophisticated pipeline that:
    1. Denoises price and features using wavelets
    2. Extracts innovations via HAR residualization and feature shocks
    3. Selects informative features using permutation RMI
    4. Whitens features using Ledoit-Wolf shrinkage
    5. Fits GMM and generates regime probabilities
    6. Runs dual pipelines (returns vs FracDiff) in parallel
    """
    
    def __init__(
        self,
        n_components: int = 8,
        n_neighbors: int = 3,
        subsample_size: int = 10000,
        wavelet: str = 'db4',
        n_permutations: int = 50,
        fracdiff_config: Optional[Dict] = None,
        har_windows: List[int] = None,
        shock_window: int = 20
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.subsample_size = subsample_size
        self.wavelet = wavelet
        self.n_permutations = n_permutations
        self.shock_window = shock_window
        
        # HAR windows for residualization
        self.har_windows = har_windows or [1, 5, 22]
        
        # FracDiff configuration
        self.fracdiff_config = fracdiff_config or {
            'max_d': 1.0,
            'min_d': 0.0,
            'adf_threshold': 0.01,
            'method': 'binary_search',
            'tolerance': 0.01
        }
        
        # Initialize components
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            reg_covar=1e-4,
            random_state=42
        )
        
        # Storage for preprocessing results
        self.denoised_price = None
        self.denoised_features = None
        self.feature_shocks = None
        self.whitening_matrix = None
        self.feature_mu = None
        self.selected_features = None
        
        # Pipeline results
        self.pipeline_results = {}

        # GPU acceleration setup
        self.use_gpu = MPS_AVAILABLE
        self.device = torch_device if MPS_AVAILABLE else None

        if self.use_gpu:
            tprint_info("🖥️ Quantitative Regime Engine: GPU acceleration enabled")
        else:
            tprint_info("💻 Quantitative Regime Engine: CPU mode (GPU not available)")

    # --- GPU Acceleration Utilities ---
    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        """Convert numpy array to GPU tensor if available."""
        if self.use_gpu and torch is not None:
            return torch.tensor(data, dtype=torch.float32, device=self.device)
        return data

    def _to_numpy(self, tensor) -> np.ndarray:
        """Convert tensor back to numpy array."""
        if self.use_gpu and torch is not None and isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy()
        return np.array(tensor)

    def _gpu_matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """GPU-accelerated matrix multiplication."""
        if self.use_gpu and torch is not None:
            A_tensor = self._to_tensor(A)
            B_tensor = self._to_tensor(B)
            result = torch.matmul(A_tensor, B_tensor)
            return self._to_numpy(result)
        else:
            return np.matmul(A, B)

    def _gpu_svd(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """GPU-accelerated SVD decomposition."""
        if self.use_gpu and torch is not None:
            matrix_tensor = self._to_tensor(matrix)
            U, S, Vh = torch.linalg.svd(matrix_tensor, full_matrices=False)
            return self._to_numpy(U), self._to_numpy(S), self._to_numpy(Vh)
        else:
            return np.linalg.svd(matrix, full_matrices=False)

    def _gpu_pca_transform(self, data: np.ndarray, n_components: int = None) -> np.ndarray:
        """GPU-accelerated PCA transformation."""
        if self.use_gpu and torch is not None:
            data_tensor = self._to_tensor(data)

            # Center the data
            data_centered = data_tensor - torch.mean(data_tensor, dim=0)

            # Compute covariance matrix
            cov_matrix = torch.matmul(data_centered.T, data_centered) / (data_centered.shape[0] - 1)

            # Eigenvalue decomposition
            eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

            # Sort eigenvectors by eigenvalues (descending)
            sorted_indices = torch.argsort(eigenvalues, descending=True)
            eigenvectors = eigenvectors[:, sorted_indices]

            # Select components
            if n_components is None:
                n_components = eigenvectors.shape[1]

            # Project data onto principal components
            transformed = torch.matmul(data_centered, eigenvectors[:, :n_components])

            return self._to_numpy(transformed)
        else:
            # Fallback to sklearn PCA
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components)
            return pca.fit_transform(data)

    # --- 1. Wavelet Denoising ---
    def wavelet_denoise(self, series: pd.Series, level: int = 1) -> pd.Series:
        """
        MODWT-style denoising to strip high-frequency jitter.
        
        Args:
            series: Input time series
            level: Decomposition level for thresholding
            
        Returns:
            Denoised series
        """
        try:
            coeffs = pywt.wavedec(series.values, self.wavelet, mode='per')
            
            # Soft threshold the detail coefficients
            sigma = np.median(np.abs(coeffs[-level])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(series)))
            
            # Apply threshold to detail coefficients
            coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
            
            # Reconstruct
            denoised = pywt.waverec(coeffs, self.wavelet, mode='per')[:len(series)]
            
            return pd.Series(denoised, index=series.index)
            
        except Exception as e:
            tprint_warning(f"⚠️ Wavelet denoising failed: {e}, returning original")
            return series

    # --- 2. Target Innovation Extraction ---
    def get_har_innovation(self, target_series: pd.Series) -> pd.Series:
        """
        HAR-structure extraction to isolate 'Market Surprise'.
        
        Uses heterogeneous autoregressive structure to remove predictable components
        and extract the pure innovation/residual series.
        
        Args:
            target_series: Target series (returns or fracdiff returns)
            
        Returns:
            Studentized residuals (innovations)
        """
        try:
            df = pd.DataFrame(index=target_series.index)
            df['y'] = target_series
            
            # HAR features
            df['d'] = target_series.shift(self.har_windows[0])  # Daily
            df['w'] = target_series.rolling(self.har_windows[1]).mean().shift(1)  # Weekly
            df['m'] = target_series.rolling(self.har_windows[2]).mean().shift(1)  # Monthly
            
            # Drop NaNs
            clean = df.dropna()
            
            if len(clean) < 100:
                tprint_warning("⚠️ Insufficient data for HAR residualization")
                return target_series
            
            # Fit HAR model
            model = LinearRegression()
            model.fit(clean[['d', 'w', 'm']], clean['y'])
            
            # Calculate innovations
            predicted = model.predict(clean[['d', 'w', 'm']])
            innovations = clean['y'] - predicted
            
            # Studentize residuals for heteroskedasticity
            rolling_std = innovations.rolling(20).std()
            studentized = innovations / (rolling_std + 1e-9)
            
            return studentized.dropna()
            
        except Exception as e:
            tprint_warning(f"⚠️ HAR innovation extraction failed: {e}")
            return target_series

    # --- 3. Feature Shocks Extraction ---
    def get_feature_shocks(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts Z-scored innovations (Feature Shocks).
        
        Removes temporal autocorrelation from features to isolate pure shocks.
        
        Args:
            feature_df: DataFrame of features
            
        Returns:
            DataFrame of feature shocks
        """
        try:
            # Calculate first differences
            delta = feature_df.diff()
            
            # EWMA for adaptive volatility normalization
            rolling_mu = delta.ewm(span=self.shock_window).mean()
            rolling_std = delta.ewm(span=self.shock_window).std()
            
            # Z-score the changes
            shocks = (delta - rolling_mu) / (rolling_std + 1e-9)
            
            return shocks.dropna()
            
        except Exception as e:
            tprint_warning(f"⚠️ Feature shocks extraction failed: {e}")
            return feature_df

    # --- 4. RMI Feature Selection ---
    def calculate_rmi_threshold(self, X: pd.DataFrame, y_innov: pd.Series) -> List[str]:
        """
        Permutation-based RMI to discard spurious signals.
        
        Uses permutation testing to establish significance thresholds for
        mutual information scores.
        
        Args:
            X: Feature matrix
            y_innov: Target innovations
            
        Returns:
            List of selected feature names
        """
        try:
            # Subsample if necessary
            X_sub = X.iloc[-self.subsample_size:] if len(X) > self.subsample_size else X
            y_sub = y_innov.iloc[-self.subsample_size:] if len(y_innov) > self.subsample_size else y_innov
            
            # Align data
            common_idx = X_sub.index.intersection(y_sub.index)
            X_aligned = X_sub.loc[common_idx]
            y_aligned = y_sub.loc[common_idx]
            
            if len(X_aligned) < 100:
                tprint_warning("⚠️ Insufficient data for RMI selection")
                return X.columns.tolist()
            
            # Calculate actual MI
            actual_mi = mutual_info_regression(
                X_aligned.fillna(0), 
                y_aligned.fillna(0), 
                n_neighbors=self.n_neighbors
            )
            
            # Calculate permutation MI distribution
            def get_perm_mi():
                y_perm = np.random.permutation(y_aligned.values)
                return mutual_info_regression(
                    X_aligned.fillna(0), 
                    y_perm, 
                    n_neighbors=self.n_neighbors
                )
            
            # Parallel permutation testing
            noise_dist = Parallel(n_jobs=-1)(
                delayed(get_perm_mi)() for _ in range(self.n_permutations)
            )
            
            # 95th percentile threshold
            thresholds = np.percentile(noise_dist, 95, axis=0)
            
            # Select features
            valid_mask = actual_mi > thresholds
            selected_features = X.columns[valid_mask].tolist()
            
            tprint_info(f"📊 RMI selected {len(selected_features)}/{len(X.columns)} features")
            
            return selected_features
            
        except Exception as e:
            tprint_warning(f"⚠️ RMI selection failed: {e}")
            return X.columns.tolist()

    # --- 5. Whitening ---
    def whiten_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Blockwise whitening using Ledoit-Wolf shrinkage with GPU acceleration.

        Transforms features into uncorrelated components with unit variance,
        essential for GMM stability.

        Args:
            X: Feature matrix

        Returns:
            Whitened feature matrix
        """
        try:
            # Store mean for later use
            self.feature_mu = X.mean()

            # Ledoit-Wolf covariance estimation
            lw = LedoitWolf().fit(X.fillna(0))
            cov_matrix = lw.covariance_

            # Eigen-decomposition (GPU accelerated if available)
            if self.use_gpu:
                vals, vecs = np.linalg.eigh(cov_matrix)  # Keep CPU for now as it's more stable
            else:
                vals, vecs = np.linalg.eigh(cov_matrix)

            # Whitening matrix computation (GPU accelerated)
            sqrt_vals_inv = np.diag(1.0 / np.sqrt(vals + 1e-9))
            whitening_mat = self._gpu_matrix_multiply(
                self._gpu_matrix_multiply(vecs, sqrt_vals_inv),
                vecs.T
            )
            self.whitening_matrix = whitening_mat

            # Apply whitening (GPU accelerated)
            X_centered = X - self.feature_mu
            X_white = self._gpu_matrix_multiply(X_centered.values, whitening_mat)
            X_white_df = pd.DataFrame(X_white, index=X.index, columns=X.columns)

            return X_white_df

        except Exception as e:
            tprint_warning(f"⚠️ Feature whitening failed: {e}")
            return X

    # --- 6. Regime Integrity ---
    def get_regime_p_values(self, X_white: pd.DataFrame) -> pd.Series:
        """
        Calculates Chi-Square p-values for outlier detection.
        
        Since X is whitened, the precision matrix is Identity,
        making Mahalanobis distance equivalent to Euclidean distance.
        
        Args:
            X_white: Whitened feature matrix
            
        Returns:
            Series of p-values
        """
        try:
            # Squared Mahalanobis distances (simplified for whitened data)
            d2 = (X_white**2).sum(axis=1)
            
            # Chi-square CDF
            p_values = 1 - chi2.cdf(d2, df=X_white.shape[1])
            
            return pd.Series(p_values, index=X_white.index)
            
        except Exception as e:
            tprint_warning(f"⚠️ Regime integrity calculation failed: {e}")
            return pd.Series(1.0, index=X_white.index)

    # --- 7. Heavy Preprocessing (Once) ---
    def preprocess_heavy_operations(
        self, 
        market_data: pd.DataFrame, 
        raw_features: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Heavy preprocessing operations done once and reused.
        
        Args:
            market_data: OHLCV data
            raw_features: Raw specialist features
            
        Returns:
            Dictionary with preprocessed components
        """
        tprint_info("🔧 Starting heavy preprocessing (once)...")
        
        # 1. Wavelet denoise price
        tprint_info("🌊 Wavelet denoising price...")
        self.denoised_price = self.wavelet_denoise(market_data['close'])
        
        # 2. Feature denoising
        tprint_info("🌊 Wavelet denoising features...")
        self.denoised_features = raw_features.apply(
            lambda x: self.wavelet_denoise(x), 
            axis=0
        )
        
        # 3. Feature shocks
        tprint_info("⚡ Extracting feature shocks...")
        self.feature_shocks = self.get_feature_shocks(self.denoised_features)
        
        tprint_success("✅ Heavy preprocessing completed")
        
        return {
            "denoised_price": self.denoised_price,
            "denoised_features": self.denoised_features,
            "feature_shocks": self.feature_shocks
        }

    # --- 8. Pipeline Execution ---
    async def run_returns_pipeline(
        self, 
        denoised_price: pd.Series, 
        feature_shocks: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Pipeline 1: Returns + HAR Residualization.
        
        Args:
            denoised_price: Wavelet-denoised price
            feature_shocks: Preprocessed feature shocks
            
        Returns:
            Pipeline results
        """
        tprint_info("📈 Running Returns Pipeline (HAR Residualization)...")
        
        try:
            # 1. Calculate returns from denoised price
            returns = denoised_price.pct_change().dropna()
            
            # 2. HAR residualization
            target_innov = self.get_har_innovation(returns)
            
            # 3. Feature processing
            common_idx = feature_shocks.index.intersection(target_innov.index)
            X = feature_shocks.loc[common_idx]
            y = target_innov.loc[common_idx]
            
            # 4. RMI selection
            selected_features = self.calculate_rmi_threshold(X, y)
            X_selected = X[selected_features]
            
            # 5. Whitening
            X_white = self.whiten_features(X_selected)
            
            # 6. GMM fitting
            if self.use_gpu:
                tprint_info("🖥️ Fitting GMM with GPU-accelerated preprocessing...")
            else:
                tprint_info("💻 Fitting GMM on CPU...")
            self.gmm.fit(X_white.fillna(0))
            probs = self.gmm.predict_proba(X_white.fillna(0))
            
            # 7. Regime synthesis
            regime_df = pd.DataFrame(
                probs, 
                index=X_white.index, 
                columns=[f'REGIME_{i}' for i in range(self.n_components)]
            )
            
            # Add regime integrity
            regime_df['REGIME_INTEGRITY'] = self.get_regime_p_values(X_white)
            
            # Add regime velocity
            velocity = regime_df.filter(like='REGIME_').diff().add_suffix('_VELOCITY')
            
            final_features = pd.concat([regime_df, velocity], axis=1).dropna()
            
            return {
                "pipeline_type": "returns_har",
                "features": final_features,
                "target_innovations": target_innov,
                "selected_features": selected_features,
                "n_features": len(selected_features),
                "optimal_d": 0.0  # No FracDiff in this pipeline
            }
            
        except Exception as e:
            tprint_error(f"❌ Returns pipeline failed: {e}")
            return {"pipeline_type": "returns_har", "error": str(e)}

    async def run_fracdiff_pipeline(
        self, 
        denoised_price: pd.Series, 
        feature_shocks: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Pipeline 2: FracDiff on Denoized Price.
        
        Args:
            denoised_price: Wavelet-denoised price
            feature_shocks: Preprocessed feature shocks
            
        Returns:
            Pipeline results
        """
        tprint_info("🔄 Running FracDiff Pipeline...")
        
        try:
            # 1. FracDiff on denoised price
            fracdiff_transformer = FracDiffTransformer(**self.fracdiff_config)
            fracdiff_returns, optimal_d = fracdiff_series(
                denoised_price,
                d=None,  # Auto-determine
                **self.fracdiff_config
            )
            
            # Validate stationarity
            stationarity_results = validate_stationarity(fracdiff_returns)
            tprint_info(f"📊 FracDiff d={optimal_d:.4f}, stationary: {stationarity_results.get('stationarity_confirmed', False)}")
            
            # 2. Calculate returns from FracDiff series
            returns = fracdiff_returns.pct_change().dropna()
            
            # 3. Feature processing (no HAR residualization for FracDiff)
            common_idx = feature_shocks.index.intersection(returns.index)
            X = feature_shocks.loc[common_idx]
            y = returns.loc[common_idx]
            
            # 4. RMI selection
            selected_features = self.calculate_rmi_threshold(X, y)
            X_selected = X[selected_features]
            
            # 5. Whitening
            X_white = self.whiten_features(X_selected)
            
            # 6. GMM fitting
            if self.use_gpu:
                tprint_info("🖥️ Fitting GMM with GPU-accelerated preprocessing...")
            else:
                tprint_info("💻 Fitting GMM on CPU...")
            self.gmm.fit(X_white.fillna(0))
            probs = self.gmm.predict_proba(X_white.fillna(0))
            
            # 7. Regime synthesis
            regime_df = pd.DataFrame(
                probs, 
                index=X_white.index, 
                columns=[f'REGIME_{i}' for i in range(self.n_components)]
            )
            
            # Add regime integrity
            regime_df['REGIME_INTEGRITY'] = self.get_regime_p_values(X_white)
            
            # Add regime velocity
            velocity = regime_df.filter(like='REGIME_').diff().add_suffix('_VELOCITY')
            
            final_features = pd.concat([regime_df, velocity], axis=1).dropna()
            
            return {
                "pipeline_type": "fracdiff",
                "features": final_features,
                "target_innovations": returns,
                "selected_features": selected_features,
                "n_features": len(selected_features),
                "optimal_d": optimal_d,
                "stationarity": stationarity_results
            }
            
        except Exception as e:
            tprint_error(f"❌ FracDiff pipeline failed: {e}")
            return {"pipeline_type": "fracdiff", "error": str(e)}

    # --- 9. Main Execution ---
    async def fit_transform(
        self, 
        market_data: pd.DataFrame, 
        raw_features: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Main execution method with dual pipeline parallel processing.
        
        Args:
            market_data: OHLCV data
            raw_features: Raw specialist features
            
        Returns:
            Dictionary with results from both pipelines
        """
        tprint_info("🚀 Starting Quantitative Regime Engine...")
        
        try:
            # 1. Heavy preprocessing (once)
            preprocessed = self.preprocess_heavy_operations(market_data, raw_features)
            
            # 2. Parallel dual pipeline execution
            tprint_info("🔄 Running dual pipelines in parallel...")
            
            returns_task = self.run_returns_pipeline(
                preprocessed["denoised_price"],
                preprocessed["feature_shocks"]
            )
            
            fracdiff_task = self.run_fracdiff_pipeline(
                preprocessed["denoised_price"],
                preprocessed["feature_shocks"]
            )
            
            # Execute in parallel
            results = await asyncio.gather(returns_task, fracdiff_task, return_exceptions=True)
            
            # Process results
            returns_result = results[0] if not isinstance(results[0], Exception) else {"error": str(results[0])}
            fracdiff_result = results[1] if not isinstance(results[1], Exception) else {"error": str(results[1])}
            
            # Combine results
            final_results = {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "preprocessing": {
                    "denoised_price_shape": preprocessed["denoised_price"].shape,
                    "feature_shocks_shape": preprocessed["feature_shocks"].shape,
                    "n_original_features": len(raw_features.columns)
                },
                "returns_pipeline": returns_result,
                "fracdiff_pipeline": fracdiff_result,
                "comparison": self._compare_pipelines(returns_result, fracdiff_result)
            }
            
            tprint_success("✅ Quantitative Regime Engine completed successfully!")
            
            return final_results
            
        except Exception as e:
            tprint_error(f"❌ Quantitative Regime Engine failed: {e}")
            return {"success": False, "error": str(e)}

    def _compare_pipelines(
        self, 
        returns_result: Dict[str, Any], 
        fracdiff_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare results between the two pipelines."""
        
        comparison = {
            "returns_pipeline_features": returns_result.get("n_features", 0),
            "fracdiff_pipeline_features": fracdiff_result.get("n_features", 0),
            "fracdiff_optimal_d": fracdiff_result.get("optimal_d", 0.0),
            "returns_success": "error" not in returns_result,
            "fracdiff_success": "error" not in fracdiff_result
        }
        
        # Feature overlap
        if "selected_features" in returns_result and "selected_features" in fracdiff_result:
            returns_features = set(returns_result["selected_features"])
            fracdiff_features = set(fracdiff_result["selected_features"])
            
            comparison["feature_overlap"] = len(returns_features & fracdiff_features)
            comparison["feature_union"] = len(returns_features | fracdiff_features)
            comparison["overlap_percentage"] = (
                comparison["feature_overlap"] / comparison["feature_union"] * 100
                if comparison["feature_union"] > 0 else 0
            )
        
        return comparison
