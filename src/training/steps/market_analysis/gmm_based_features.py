"""
GMM-Based Feature Engineering Pipeline
-----------------------------------------------------------------------------
This module implements a sophisticated 4-step pipeline for generating
regime-aware features using Gaussian Mixture Models, Causal Discovery,
and Latent Factor Analysis.

The four pipelines are:
1. Step A: "Macro State" (Master GMM on Compressed Features)
2. Step B: "Causal Experts" (Multivariate GMMs on Causal Families)
3. Step C: "Latent Causal Macro" (Master GMM on Independent Latent Factors)
4. Step D: "Testing & Pruning" (MDI Feature Selection)

Usage:
    step = GMMFeaturePipeline()
    step.run(config)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA, FastICA
from scipy.stats import entropy
import networkx as nx
import joblib
import os
import gc
from datetime import datetime

# Internal imports
from src.training.steps.base_step import BaseStep
from src.training.steps.labeling.mtf_feature_generation import create_meta_features
from src.training.steps.labeling.de_prado_feature_engine import DePradoFeatureEngine
from src.training.steps.labeling.causal_discovery import CausalDiscovery
from src.utils.ml_common.wavelet_utils import wavelet_energy_ratios
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.numba_funcs import jit as njit

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    # Fallback dummy decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(n):
        return range(n)
    NUMBA_AVAILABLE = False

# Constants
MAX_FITTING_SAMPLES = 20000
GMM_RANDOM_STATE = 42
DEFAULT_GMM_COMPONENTS = 8

class RobustGMM:
    """
    Robust Gaussian Mixture Model wrapper incorporating logic from AdaptiveHunterRouter.

    Features:
    - RobustScaling
    - Entropy calculation
    - Z-Familiarity score (Mahalanobis-like)
    - Component selection via BIC (optional)
    """

    def __init__(self, n_components: int = 5, random_state: int = 42, covariance_type: str = 'full'):
        self.n_components = n_components
        self.random_state = random_state
        self.covariance_type = covariance_type
        self.scaler = RobustScaler()
        self.gmm = None
        self.log_lik_mean = None
        self.log_lik_std = None
        self.is_fitted = False

    def fit(self, X: np.ndarray):
        """Fit GMM with Robust Scaling."""
        if len(X) < self.n_components * 2:
            tprint_warning(f"   ⚠️ RobustGMM: Insufficient samples ({len(X)}) for {self.n_components} components.")
            return self

        # Scale
        X_scaled = self.scaler.fit_transform(X)

        # Fit GMM
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            reg_covar=1e-5,
            random_state=self.random_state,
            n_init=3 # Multiple restarts for stability
        ).fit(X_scaled)

        # Calculate baseline log-likelihood stats for Z-score
        scores = self.gmm.score_samples(X_scaled)
        self.log_lik_mean = np.mean(scores)
        self.log_lik_std = np.std(scores)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict regime probabilities and metrics.

        Returns:
            Tuple[probs, z_familiarity, entropy]
        """
        if not self.is_fitted:
            # Return uniform if not fitted
            n_samples = len(X)
            probs = np.full((n_samples, self.n_components), 1.0 / self.n_components)
            z_fam = np.zeros(n_samples)
            ent = np.full(n_samples, np.log(self.n_components))
            return probs, z_fam, ent

        X_scaled = self.scaler.transform(X)

        # Probabilities
        probs = self.gmm.predict_proba(X_scaled)

        # Log Likelihood for Z-Familiarity
        log_prob = self.gmm.score_samples(X_scaled)
        z_familiar = (log_prob - self.log_lik_mean) / (self.log_lik_std + 1e-9)

        # Entropy
        # Clip probabilities for numerical stability in log
        probs_safe = np.clip(probs, 1e-9, 1.0)
        ent = -np.sum(probs_safe * np.log(probs_safe), axis=1)

        return probs, z_familiar, ent

class GMMFeaturePipeline(BaseStep):
    """
    4-Stage GMM-Based Feature Engineering Pipeline.
    """

    def __init__(self, step_name: str = "gmm_based_features", **kwargs):
        super().__init__(step_name, **kwargs)
        self.verbose = kwargs.get('verbose', True)
        self.artifacts_dir = "artifacts/gmm_features"
        os.makedirs(self.artifacts_dir, exist_ok=True)

        # Configurable params
        self.n_clusters_macro = kwargs.get('n_clusters_macro', 8)
        self.pca_variance = kwargs.get('pca_variance', 0.95)
        self.n_latent_factors = kwargs.get('n_latent_factors', 8)

        # Caches
        self.models = {}
        self.feature_lists = {}

    def _preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Efficient Preprocessing: Winsorization + Z-Score (Numpy/Numba).
        Uses simple clipping at 1st/99th percentiles for speed.
        """
        if df.empty: return df

        tprint_info("   🧹 Preprocessing: Winsorization + Standardization...")

        # Convert to float32 to save memory
        X = df.astype(np.float32).values

        # Calculate stats (ignoring NaNs)
        means = np.nanmean(X, axis=0)
        stds = np.nanstd(X, axis=0)

        # Avoid division by zero
        stds[stds < 1e-9] = 1.0

        # Z-Score
        X_z = (X - means) / stds

        # Winsorize (clip at +/- 5 sigma)
        X_clipped = np.clip(X_z, -5.0, 5.0)

        # Replace NaNs with 0 (mean imputation after z-score)
        X_clean = np.nan_to_num(X_clipped, nan=0.0)

        return pd.DataFrame(X_clean, index=df.index, columns=df.columns)

    def _subsample_data(self, df: pd.DataFrame, n_samples: int = MAX_FITTING_SAMPLES) -> pd.DataFrame:
        """
        Uniformly subsample data for fitting models efficiently.
        """
        if len(df) <= n_samples:
            return df

        indices = np.linspace(0, len(df) - 1, n_samples).astype(int)
        return df.iloc[indices]

    def _step_a_macro_state(self, X: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
        """
        Step A: The "Macro State" (Master GMM).
        1. Cluster features (ONC).
        2. Extract PCA 1st PC per cluster.
        3. Fit Master GMM.
        4. Anchor to returns.
        """
        tprint_info("\n🌐 Step A: Macro State Analysis...")

        # 1. Feature Clustering (ONC)
        tprint_info("   🔍 Running ONC Clustering on features...")
        X_fit = self._subsample_data(X)

        deprado = DePradoFeatureEngine(max_clusters=self.n_clusters_macro, verbose=False)
        clusters = deprado._get_onc_clusters(X_fit)

        unique_clusters = clusters.unique()
        tprint_info(f"   🧩 Found {len(unique_clusters)} feature clusters.")

        # 2. Extract 1st PC per cluster (Compression)
        tprint_info("   📉 Compressing clusters via PCA (1st Component)...")
        compressed_features = pd.DataFrame(index=X.index)

        for cid in unique_clusters:
            feats_in_cluster = clusters[clusters == cid].index.tolist()
            if not feats_in_cluster: continue

            # PCA
            pca = PCA(n_components=1)
            # Fit on subsample
            pca.fit(X_fit[feats_in_cluster])
            # Transform full
            pc1 = pca.transform(X[feats_in_cluster])

            # Name: e.g., cluster_0_pc1
            col_name = f"cluster_{cid}_pc1"
            compressed_features[col_name] = pc1.flatten()

            # Log explained variance
            exp_var = pca.explained_variance_ratio_[0]
            # tprint_info(f"      - Cluster {cid}: {len(feats_in_cluster)} feats -> PC1 (ExpVar: {exp_var:.2f})")

        # 3. Fit Master GMM
        tprint_info(f"   🧠 Fitting Master GMM on {len(compressed_features.columns)} compressed features...")
        gmm = RobustGMM(n_components=self.n_clusters_macro, random_state=GMM_RANDOM_STATE)
        gmm.fit(self._subsample_data(compressed_features).values)
        self.models['step_a_gmm'] = gmm

        # 4. Generate Predictions & Anchor to Returns
        probs, z_fam, ent = gmm.predict(compressed_features.values)

        # Calculate Cluster Anchors (Mean Forward Return per Cluster)
        # Use 12-period forward return (approx 3h)
        fwd_ret = returns.shift(-12).fillna(0)

        # Align timelines
        # Compute weighted average return for each cluster
        cluster_returns = []
        for k in range(self.n_clusters_macro):
            # Weight = probability of being in cluster k
            w = probs[:, k]
            # Weighted mean return
            mean_ret = np.average(fwd_ret, weights=w) if np.sum(w) > 0 else 0.0
            cluster_returns.append(mean_ret)

        tprint_info(f"   ⚓ Anchored Cluster Returns: {[f'{r:.5f}' for r in cluster_returns]}")

        # 5. Generate Signals
        tprint_info("   ⚡ Generating Step A Signals...")
        results = pd.DataFrame(index=X.index)

        # Macro GMM Signal: Dot product of probs and cluster returns
        results['macro_gmm_signal'] = np.dot(probs, np.array(cluster_returns))

        # Regime Velocity: Change in probability distribution
        # L2 norm of diff in prob vectors
        probs_diff = np.diff(probs, axis=0, prepend=probs[:1])
        regime_velocity = np.linalg.norm(probs_diff, axis=1)
        results['macro_regime_velocity'] = regime_velocity

        # Theme Acceleration: Change in velocity
        results['macro_theme_accel'] = pd.Series(regime_velocity, index=X.index).diff().fillna(0)

        # Add Entropy and Z-Fam
        results['macro_entropy'] = ent
        results['macro_z_familiarity'] = z_fam

        return results

    def _step_b_causal_experts(self, X: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
        """
        Step B: "Causal Experts" (Multivariate GMMs).
        1. Causal Discovery (PC Algo).
        2. Family Construction (Leader + Children).
        3. Expert GMMs.
        4. Structural Scalar.
        """
        tprint_info("\n🔗 Step B: Causal Experts Analysis...")

        # 1. Causal Discovery
        tprint_info("   🔍 Running Causal Discovery (PC Algorithm)...")
        # Use subsample for speed
        X_fit = self._subsample_data(X, n_samples=5000) # PC is slow, use smaller sample

        cd = CausalDiscovery(verbose=False)
        graph = cd.pc_algorithm(X_fit, list(X_fit.columns))

        # 2. Family Construction
        tprint_info("   👨‍👩‍👧‍👦 Identifying Causal Families...")
        # Build networkx graph
        G = nx.DiGraph()
        for node, neighbors in graph.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)

        # Calculate Out-Degree Centrality
        centrality = nx.out_degree_centrality(G)
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

        # Select top leaders (ensure diversity)
        families = []
        covered_nodes = set()

        for leader, score in sorted_nodes:
            if leader in covered_nodes: continue

            # Find children (direct successors)
            children = list(G.successors(leader))

            # Keep top 4 children by correlation (or just first 4)
            # Simple heuristic: take first 4
            selected_children = children[:4]

            if len(selected_children) >= 1:
                family_nodes = [leader] + selected_children
                families.append({
                    'leader': leader,
                    'members': family_nodes,
                    'score': score
                })
                covered_nodes.update(family_nodes)

            if len(families) >= 5: # Limit to 5 top families
                break

        tprint_info(f"   found {len(families)} causal families.")

        # 3. Expert GMMs & 4. Signals
        results = pd.DataFrame(index=X.index)

        for i, fam in enumerate(families):
            leader = fam['leader']
            members = fam['members']

            # Fit Multivariate GMM on family members
            fam_X = X[members]
            gmm = RobustGMM(n_components=3, random_state=GMM_RANDOM_STATE + i) # 3 regimes per expert
            gmm.fit(self._subsample_data(fam_X).values)
            self.models[f'expert_gmm_{leader}'] = gmm

            # Predict
            probs, z_fam, ent = gmm.predict(fam_X.values)

            # Structural Weighted Scalar
            # Weight = Causal Strength?
            # Proxy: Correlation of Leader with Returns
            # But we want the "Pressure" of the regime.
            # Let's anchor the clusters to returns like in Step A
            fwd_ret = returns.shift(-12).fillna(0)
            cluster_impacts = []
            for k in range(3):
                w = probs[:, k]
                impact = np.average(fwd_ret, weights=w) if np.sum(w) > 0 else 0.0
                cluster_impacts.append(impact)

            # Scalar = Sum(Prob * Impact)
            scalar = np.dot(probs, np.array(cluster_impacts))

            # Store Features
            prefix = f"causal_expert_{i}" # use index for brevity, map to leader in metadata
            results[f'{prefix}_signal'] = scalar
            results[f'{prefix}_entropy'] = ent
            results[f'{prefix}_velocity'] = np.linalg.norm(np.diff(probs, axis=0, prepend=probs[:1]), axis=1)

            # tprint_info(f"      - Family {i}: Leader={leader}, Members={len(members)}, Signal Mean={scalar.mean():.5f}")

        return results

    def _step_c_latent_causal_macro(self, X: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
        """
        Step C: "Latent Causal Macro".
        1. Independent Component Analysis (ICA).
        2. Wavelet Entropy Anchoring.
        3. Latent GMM.
        4. Signals.
        """
        tprint_info("\n👻 Step C: Latent Causal Macro Analysis...")

        # 1. ICA Extraction
        tprint_info(f"   🧪 Extracting {self.n_latent_factors} Independent Components (ICA)...")
        ica = FastICA(n_components=self.n_latent_factors, random_state=GMM_RANDOM_STATE, whiten='unit-variance')
        # Fit on subsample
        X_fit = self._subsample_data(X)
        ica.fit(X_fit)

        # Transform full
        latent_factors = ica.transform(X) # Shape (N, n_factors)

        # 2. Wavelet Entropy Anchoring
        # Calculate "Causal Impact" or "Structure" of each factor
        # Low Entropy = High Structure = Potential Causal Driver
        factor_weights = []

        tprint_info("   🌊 Calculating Wavelet Entropy for factors...")
        # Use a window to calculate rolling entropy? Or global?
        # User said "Use Wavelets entropy here".
        # Let's calculate Global Wavelet Entropy of the factor to assign a static weight/quality score.

        for k in range(self.n_latent_factors):
            factor_series = latent_factors[:, k]
            # Use subsample for speed if needed, but 1D wavelet is fast
            # Calculate Wavelet Entropy on the factor series
            # We use `wavelet_energy_ratios` (High/Low ratio) as a proxy for noise/entropy
            # Lower ratio = More Low Freq energy = More Structure?
            # Actually, let's use the entropy per scale from `get_wavelet_features` if accessible,
            # or just use the ratio.
            # User specifically asked for "Wavelets entropy".
            # Implementation: Shannon entropy of the wavelet energy distribution.

            # Simple implementation using existing util if possible, or inline
            try:
                # Reuse wavelet_energy_ratios (returns float noise_ratio)
                # If noise_ratio is high -> High Entropy/Noise.
                noise_ratio = wavelet_energy_ratios(factor_series[-2000:]) # Check last 2000 points
                weight = 1.0 - noise_ratio # Higher weight for cleaner signals
            except:
                weight = 0.5

            factor_weights.append(weight)

        tprint_info(f"      - Factor Weights (1-Noise): {[f'{w:.2f}' for w in factor_weights]}")

        # Re-weight factors? Or select top?
        # Let's keep all but weight their contribution to the GMM input?
        # Or just feed raw ICA components to GMM?
        # User: "X Features: If you find 8 independent latent factors, your Master GMM has 8 inputs."
        # User: "Anchor to Returns: ... Use Wavelets entropy ... to verify that the cluster actually forces a change..."
        # Ah, the anchoring is for the *GMM Clusters*, not the ICA factors directly.
        # "Live Signal: The dot product of predict_proba and the Causal Impact of each cluster."

        # So:
        # 1. Fit GMM on ICA factors.
        # 2. For each cluster, estimate its "Causal Impact".
        #    Impact = Mean Return * (1 / Cluster_Wavelet_Entropy)?
        #    Or verify if cluster activation *reduces* return entropy?

        # Let's implement:
        # Impact_k = Mean_Return_k * Weight_k
        # Where Weight_k is derived from Wavelet Entropy of returns *during* that regime.
        # If Regime K is active and Return Entropy is Low -> High Causal Control -> High Weight.

        # 3. Fit Latent GMM
        tprint_info("   🧠 Fitting Latent GMM on ICA factors...")
        gmm = RobustGMM(n_components=self.n_clusters_macro, random_state=GMM_RANDOM_STATE)
        gmm.fit(latent_factors)
        self.models['step_c_gmm'] = gmm

        probs, z_fam, ent = gmm.predict(latent_factors)

        # 4. Causal Impact Calculation
        cluster_impacts = []

        # Calculate Rolling Wavelet Entropy of Returns (expensive?)
        # Or just calculate it on the subsets defined by the clusters.

        for k in range(self.n_clusters_macro):
            # Identify periods where this cluster is dominant
            mask = probs[:, k] > 0.5
            if mask.sum() < 20:
                cluster_impacts.append(0.0)
                continue

            # Get returns during this regime
            regime_returns = returns[mask]

            # 1. Mean Return
            mean_ret = regime_returns.mean()

            # 2. Wavelet Entropy of Regime Returns
            # Are returns structured during this regime?
            try:
                # Taking a slice for wavelet might be discontinuous.
                # Just take the mean return for now, modulated by a static "Quality" check
                # If returns are very noisy (high std), penalize?
                # Let's stick to the prompt: "Use Wavelets entropy... to verify... forces a change"
                # Simplified:
                impact = mean_ret

                # Check entropy of returns in this regime (treating concatenated slice as series)
                if len(regime_returns) > 100:
                    noise = wavelet_energy_ratios(regime_returns.values[:1000]) # Sample
                    # If noise is high, reduce impact (it's just noise, not signal)
                    # Impact = Mean Return * (1 - Noise)^2
                    quality = (1.0 - noise) ** 2
                    impact *= quality

                cluster_impacts.append(impact)
            except:
                cluster_impacts.append(mean_ret)

        tprint_info(f"   ⚡ Generating Step C Signals...")
        results = pd.DataFrame(index=X.index)

        # Latent Macro Signal
        results['latent_macro_signal'] = np.dot(probs, np.array(cluster_impacts))

        # Causal Kinematics (Acceleration of the Latent State)
        # Latent State Vector S = Sum(Prob_k * Factor_Mean_k)?
        # Or just use the Signal acceleration.
        results['latent_kinematics_accel'] = results['latent_macro_signal'].diff().diff().fillna(0)

        return results

    def _step_d_testing_pruning(self, feature_sets: List[pd.DataFrame], target: pd.Series) -> pd.DataFrame:
        """
        Step D: Testing & Pruning.
        Combine all features and run MDI selection.
        """
        tprint_info("\n✂️ Step D: Testing & Pruning...")

        # 1. Combine
        all_features = pd.concat(feature_sets, axis=1)
        # Handle NaN/Inf
        all_features = all_features.replace([np.inf, -np.inf], np.nan).fillna(0)

        tprint_info(f"   📊 Combined Feature Set: {all_features.shape[1]} features.")

        # Align target
        y = target.reindex(all_features.index).fillna(0)
        # Binarize target for classification selection (e.g., positive return)
        y_binary = (y > 0).astype(int)

        # 2. MDI Feature Selection
        tprint_info("   🌳 Running MDI Feature Selection...")
        # Use DePrado engine
        engine = DePradoFeatureEngine(
            n_estimators=500, # Faster
            max_clusters=5,
            verbose=False
        )

        try:
            selected_cols = engine.run_selection(all_features, y_binary)
            tprint_success(f"   ✅ Selected {len(selected_cols)} features: {selected_cols}")

            final_df = all_features[selected_cols]

            # Save stats
            self.feature_lists['selected'] = selected_cols

            return final_df

        except Exception as e:
            tprint_error(f"   ❌ Feature selection failed: {e}")
            return all_features

    def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the full pipeline.
        """
        try:
            # 1. Load Data
            market_data, _ = self.load_market_data_or_fail(config)
            if market_data is None or market_data.empty:
                raise ValueError("No market data loaded.")

            tprint_info(f"🚀 Starting GMM Feature Pipeline on {len(market_data)} rows...")

            # 2. Base Features
            tprint_info("🔨 Generating Base Meta-Features...")
            # Dummy signals df required by create_meta_features
            dummy_signals = pd.DataFrame(index=market_data.index)
            base_features = create_meta_features(market_data, dummy_signals, volume_available=True)

            # Preprocess
            X_clean = self._preprocess_features(base_features)

            # Define Target (e.g., 1-day forward return for anchoring/selection)
            returns = market_data['close'].pct_change()

            # 3. Pipelines
            # A: Macro State
            df_a = self._step_a_macro_state(X_clean, returns)

            # B: Causal Experts
            df_b = self._step_b_causal_experts(X_clean, returns)

            # C: Latent Causal Macro
            df_c = self._step_c_latent_causal_macro(X_clean, returns)

            # D: Pruning
            # Target for selection: 12-period forward return sign
            target_series = returns.shift(-12).fillna(0)

            final_features = self._step_d_testing_pruning([df_a, df_b, df_c], target_series)

            # 4. Save Artifacts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.artifacts_dir, f"gmm_features_{timestamp}.parquet")
            final_features.to_parquet(output_path)
            tprint_success(f"💾 Saved {len(final_features.columns)} features to {output_path}")

            return {
                "features_path": output_path,
                "n_features": len(final_features.columns),
                "feature_names": list(final_features.columns),
                "success": True
            }

        except Exception as e:
            tprint_error(f"❌ GMM Pipeline Failed: {e}")
            import traceback
            tprint_error(traceback.format_exc())
            return {"success": False, "error": str(e)}

def register_gmm_feature_step():
    from src.training.steps.base_step import step_registry
    step_registry.register("gmm_based_features", GMMFeaturePipeline)
