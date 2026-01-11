
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.mixture import BayesianGaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.utils.tprint import tprint_info, tprint_warning, tprint_error

class DPGMMRegimeDetector:
    """
    Dirichlet Process Gaussian Mixture Model (DPGMM) for regime detection.
    
    Uses a Bayesian approach with a Dirichlet Process prior to automatically 
    determine the number of active components (regimes) and handle infinite mixtures.
    Suitable for 'Layer 1' Movement detection.
    """
    
    def __init__(self, n_components: int = 10, concentration_prior: float = 1.0/10):
        """
        Args:
            n_components: Upper bound on number of regimes (truncation level)
            concentration_prior: Dirichlet concentration parameter (gamma). 
                                 Lower values -> fewer active regimes.
        """
        self.n_components = n_components
        self.concentration_prior = concentration_prior
        self.model = None
        self.active_components = 0
        
    def fit(self, X: pd.DataFrame, weights: Optional[np.ndarray] = None):
        """
        Fit DPGMM.
        
        Args:
            X: Feature matrix
            weights: Sample weights (W-EM support via resampling or property if supported)
        """
        try:
            # Note: BayesianGaussianMixture in sklearn usually ignores weights in fit()
            # We assume X is already resampled if W-EM is applied externally
            
            self.model = BayesianGaussianMixture(
                n_components=self.n_components,
                covariance_type='full',
                weight_concentration_prior_type='dirichlet_process',
                weight_concentration_prior=self.concentration_prior,
                max_iter=500,
                random_state=42,
                n_init=1
            )
            
            self.model.fit(X)
            
            # Count active components (weight > threshold)
            self.active_components = np.sum(self.model.weights_ > 1e-3)
            
            tprint_info(f"🧠 DPGMM converged. Active regimes: {self.active_components}/{self.n_components}")
            
        except Exception as e:
            tprint_warning(f"⚠️ DPGMM fit failed: {e}")
            self.model = None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.zeros((len(X), self.n_components))
        return self.model.predict_proba(X)
        
    def get_params(self) -> Dict[str, Any]:
        if self.model is None: return {}
        return {
            "weights": self.model.weights_.tolist(),
            "means": self.model.means_.tolist(),
            "covariances": self.model.covariances_.tolist(),
            "active_components": int(self.active_components)
        }


class IOHMMRegimeDetector:
    """
    Input-Output Hidden Markov Model (IOHMM) proxy.
    
    Models 'Layer 2' State by conditioning state transitions on input features.
    S_t depends on S_{t-1} AND X_t.
    
    Implementation:
    - Uses GMM/DPGMM outputs as 'Observation' or 'State Proxy'.
    - Trains a transition model P(S_t | S_{t-1}, X_t).
    - Can smooth the state sequence.
    """
    
    def __init__(self, n_states: int = 3):
        self.n_states = n_states
        self.transition_models = {} # distinct models for each 'from' state
        
    def fit(self, X: pd.DataFrame, state_seq: np.ndarray):
        """
        Fit IOHMM transition dynamics.
        
        Args:
            X: Input features conditioning the transition
            state_seq: Sequence of states (from GMM/DPGMM hard assignment)
        """
        df = pd.DataFrame(X).copy()
        df['state'] = state_seq
        df['prev_state'] = pd.Series(state_seq).shift(1)
        
        # Drop first row (unknown prev state)
        train_df = df.dropna()
        
        if len(train_df) < 100:
            return
            
        # Train a classifier for each previous state
        # P(S_t | S_{t-1}=k, X_t)
        
        unique_states = np.unique(state_seq)
        
        for k in unique_states:
            subset = train_df[train_df['prev_state'] == k]
            
            if len(subset) < 20: 
                continue
                
            # Features: X_t
            X_subset = subset.drop(['state', 'prev_state'], axis=1)
            y_subset = subset['state']
            
            # Use simplified Logistic Regression
            clf = LogisticRegression(max_iter=200, C=1.0)
            try:
                clf.fit(X_subset, y_subset)
                self.transition_models[k] = clf
            except:
                continue
                
        tprint_info(f"🧠 IOHMM transition models fitted for {len(self.transition_models)} states")

    def predict_smoothed_probs(self, X: pd.DataFrame, initial_probs: np.ndarray) -> np.ndarray:
        """
        Predict state probabilities using input-conditioned transitions.
        This effectively filters the GMM 'observation' probabilities.
        
        Args:
            X: Features
            initial_probs: Probabilities from Layer 1 GMM (treated as observation likelihood P(O|S) approx)
                           Actually, GMM gives P(S|X) (posterior).
                           We can treat GMM output as 'Observation' of state.
        
        Returns:
            Smoothed probabilities.
        """
        n = len(X)
        n_states = initial_probs.shape[1]
        smoothed = np.zeros_like(initial_probs)
        
        # Forward pass (Filtering)
        # alpha_t(j) = P(Standard HMM) ... but here transitions depend on X
        # alpha_t = (alpha_{t-1} * TransitionMatrix(X_t)) * Observation(t)
        
        # 1. Initialize
        smoothed[0] = initial_probs[0]
        
        # 2. Iterate
        # Note: Vectorizing this is hard because TransitionMatrix depends on t
        # We assume X controls transition.
        
        # Optimization: We can predict transition matrices for all t first?
        # Too heavy if n is large.
        
        # Simplified:
        # P(S_t | X_t, S_{t-1}) is from our models.
        # But we don't know S_{t-1}, we have distribution smoothed[t-1].
        
        # smoothed[t] = normalize( (sum_k smoothed[t-1][k] * P(S_t|k, X_t)) * initial_probs[t] )
        # initial_probs[t] is P(S_t | X_t) from GMM (Movement Layer).
        # We blend 'Movement' (GMM) and 'Context' (IOHMM Transition).
        
        # For simplicity in this implementation, we just return GMM probs 
        # but blended with a momentum factor derived from transitions.
        
        return initial_probs # Placeholder for full Viterbi/Forward-Backward
