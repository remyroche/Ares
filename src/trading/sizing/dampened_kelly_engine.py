"""
Dampened Kelly Engine - Production-Hardened Kelly Criterion Implementation

Unified dampened Kelly sizing for both position sizing AND leverage calculation,
with Bayesian posterior estimation, ensemble uncertainty (ESS/entropy), regime
conditioning, adaptive binning fallback, and comprehensive risk management.

Key Features:
- Unified logic for position sizing and leverage
- Modular lambda_eff computation (ESS, entropy, variance components)
- Kelly fraction clipping (never exceed max_kelly_fraction)
- Drawdown dampening (scales with current drawdown)
- Exploration floors (minimum position/leverage for exploration)
- Thread-safe versioned config for hot-swapping
- Comprehensive metadata and reason codes
"""

import numpy as np
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum

from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from src.utils.tprint import tprint_info, tprint_warning, tprint_error

logger = system_logger.getChild('DampenedKellyEngine')


class ReasonCode(Enum):
    """Reason codes for sizing decisions."""
    INSUFFICIENT_SAMPLES = "insufficient_samples"
    ENTROPY_VETO = "entropy_veto"
    ESS_LOW = "ess_low"
    HARD_CAP_APPLIED = "hard_cap_applied"
    REGIME_UNKNOWN = "regime_unknown"
    THROTTLED_BY_COOLDOWN = "throttled_by_cooldown"
    PORTFOLIO_LIMIT_REACHED = "portfolio_limit_reached"
    BIN_MERGED = "bin_merged"
    DRAWDOWN_DAMPENING = "drawdown_dampening"
    CORRELATION_ADJUSTED = "correlation_adjusted"
    KELLY_FRACTION_CLIPPED = "kelly_fraction_clipped"
    R_INSTABILITY_DETECTED = "r_instability_detected"


@dataclass
class KellyConfigVersion:
    """Versioned configuration for thread-safe hot-swapping."""
    version: int
    timestamp: datetime
    params: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'version': self.version,
            'timestamp': self.timestamp.isoformat(),
            'params': self.params.copy()
        }


@dataclass
class KellyResult:
    """Result from dampened Kelly calculation."""
    # Core outputs
    f_final: float  # Final position size fraction
    leverage_final: float  # Final leverage
    
    # Intermediate calculations
    f_kelly: float  # Raw Kelly fraction
    leverage_kelly: float  # Raw Kelly leverage
    lambda_eff: float  # Effective dampening factor
    
    # Posterior statistics
    posterior_mean: float
    posterior_var: float
    
    # Ensemble uncertainty
    ess: float  # Effective sample size
    entropy: float  # Ensemble entropy
    
    # Bin information
    regime_id: Optional[int]
    bin_sample_count: int
    bin_merge_level: int  # 0=exact, 1=regime-merged, 2=coarse, 3=prior
    bin_last_updated: Optional[datetime]
    is_bin_stale: bool
    
    # Realized R statistics
    r_realized_mean: float
    r_realized_std: float
    r_conservative: float  # Conservative R used (25th percentile)
    
    # Adjustments applied
    dd_dampening_factor: float  # Drawdown dampening multiplier
    kelly_fraction_clip_applied: bool
    correlation_adjusted: bool
    
    # Configuration
    config_version: int
    
    # Metadata
    reason_codes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'f_final': self.f_final,
            'leverage_final': self.leverage_final,
            'f_kelly': self.f_kelly,
            'leverage_kelly': self.leverage_kelly,
            'lambda_eff': self.lambda_eff,
            'posterior_mean': self.posterior_mean,
            'posterior_var': self.posterior_var,
            'ess': self.ess,
            'entropy': self.entropy,
            'regime_id': self.regime_id,
            'bin_sample_count': self.bin_sample_count,
            'bin_merge_level': self.bin_merge_level,
            'r_realized_mean': self.r_realized_mean,
            'r_realized_std': self.r_realized_std,
            'r_conservative': self.r_conservative,
            'dd_dampening_factor': self.dd_dampening_factor,
            'kelly_fraction_clip_applied': self.kelly_fraction_clip_applied,
            'correlation_adjusted': self.correlation_adjusted,
            'config_version': self.config_version,
            'reason_codes': self.reason_codes,
            'metadata': self.metadata
        }


class DampenedKellyEngine:
    """
    Production-hardened dampened Kelly engine with unified position/leverage logic.
    
    Implements sophisticated Kelly criterion with:
    - Bayesian posterior estimation (Beta distribution)
    - Ensemble uncertainty integration (ESS, entropy)
    - Regime-aware parameters
    - Adaptive bin merging fallback
    - Realized R tracking
    - Thread-safe hot-swappable config
    - Comprehensive risk management
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize dampened Kelly engine.
        
        Args:
            config: Configuration dictionary with regime-aware parameters
        """
        self.config = config
        self.logger = logger.getChild('Engine')
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Versioned configuration
        self._config_version = 1
        self._config_history: List[KellyConfigVersion] = []
        self._current_config = self._create_config_version(config)
        self._config_history.append(self._current_config)
        
        # Extract configuration sections
        self.regime_params = config.get('regime_params', {})
        self.global_fallback = config.get('global_fallback', {})
        self.lambda_eff_components = config.get('lambda_eff_components', {})
        self.safety_limits = config.get('safety_limits', {})
        self.r_tracking = config.get('r_tracking', {})
        
        # Lambda_eff component parameters
        self.ess_sigmoid_kappa = self.lambda_eff_components.get('ess_sigmoid_kappa', 0.1)
        self.entropy_scale = self.lambda_eff_components.get('entropy_scale', 0.5)
        self.variance_penalty = self.lambda_eff_components.get('variance_penalty', 2.0)
        
        # Safety limits (hot-swappable, except fixed risk parameters)
        self.max_leverage = self.safety_limits.get('max_leverage', 3.0)
        self.max_per_trade_pct = self.safety_limits.get('max_per_trade_pct', 0.15)
        self.max_exposure_per_asset = self.safety_limits.get('max_exposure_per_asset', 0.3)
        self.high_leverage_threshold = self.safety_limits.get('high_leverage_threshold', 2.0)
        self.max_acceptable_drawdown = self.safety_limits.get('max_acceptable_drawdown', 0.15)
        
        # Fixed risk parameters (not optimized - strategic choices)
        self.f_floor = 0.005  # Exploration floor for position sizing
        self.max_kelly_fraction = 0.33  # Risk cap: 1/3 Kelly for robustness
        
        # R tracking parameters
        self.use_realized_r = self.r_tracking.get('use_realized_r', True)
        self.r_percentile = self.r_tracking.get('r_percentile', 25)
        self.r_instability_threshold = self.r_tracking.get('r_instability_threshold', 2.0)
        self.r_instability_prior_boost = self.r_tracking.get('r_instability_prior_boost', 2.0)
        
        tprint_info("✅ Dampened Kelly Engine initialized")
        self.logger.info(f"Initialized with config version {self._config_version}")
    
    def _create_config_version(self, params: Dict[str, Any]) -> KellyConfigVersion:
        """Create a versioned config snapshot."""
        return KellyConfigVersion(
            version=self._config_version,
            timestamp=datetime.now(),
            params=params.copy()
        )
    
    @handles_errors
    def update_config(self, new_params: Dict[str, Any]) -> int:
        """
        Hot-swap configuration (thread-safe).
        
        Args:
            new_params: New configuration parameters
            
        Returns:
            New config version number
        """
        with self._lock:
            self._config_version += 1
            self._current_config = self._create_config_version(new_params)
            self._config_history.append(self._current_config)
            
            # Update hot-swappable limits
            if 'safety_limits' in new_params:
                limits = new_params['safety_limits']
                self.max_leverage = limits.get('max_leverage', self.max_leverage)
                self.max_per_trade_pct = limits.get('max_per_trade_pct', self.max_per_trade_pct)
                self.max_exposure_per_asset = limits.get('max_exposure_per_asset', self.max_exposure_per_asset)
                self.max_kelly_fraction = limits.get('max_kelly_fraction', self.max_kelly_fraction)
                self.max_acceptable_drawdown = limits.get('max_acceptable_drawdown', self.max_acceptable_drawdown)
            
            self.logger.info(f"Config updated to version {self._config_version}")
            tprint_info(f"🔄 Kelly config updated to version {self._config_version}")
            
            return self._config_version
    
    def get_config_version(self) -> int:
        """Get current config version."""
        with self._lock:
            return self._config_version
    
    def get_regime_params(self, regime_id: Optional[int]) -> Dict[str, Any]:
        """
        Get regime-specific parameters with fallback to global.
        
        Args:
            regime_id: Regime identifier (None for unknown)
            
        Returns:
            Parameter dictionary for the regime
        """
        if regime_id is None:
            return self.global_fallback.copy()
        
        regime_key = f"regime_{regime_id}"
        if regime_key in self.regime_params:
            return self.regime_params[regime_key].copy()
        
        # Fallback to global if regime not found
        self.logger.warning(f"Regime {regime_id} not found, using global fallback")
        return self.global_fallback.copy()
    
    @staticmethod
    def calculate_system_half_life_params(
        system_half_life: float,
        target_samples: int = 200
    ) -> Tuple[float, float]:
        """
        Calculate decay_theta and prior_alpha from a single system half-life parameter.
        
        The system_half_life represents the number of trades after which the system's
        belief should be 50% based on historical data.
        
        Args:
            system_half_life: Number of trades for 50% belief decay (e.g., 200)
            target_samples: Target samples for half-life calculation
            
        Returns:
            Tuple of (decay_theta, prior_alpha)
        """
        # decay_theta: exponential decay factor where 0.5 = theta^system_half_life
        # Solving: 0.5 = theta^N => theta = 0.5^(1/N)
        decay_theta = 0.5 ** (1.0 / system_half_life)
        
        # prior_alpha: Bayesian prior strength
        # Higher half-life = trust old data more = higher prior
        # Map half-life [100, 300] to prior_alpha [10, 50]
        # Using linear interpolation
        prior_alpha = 10.0 + (system_half_life - 100.0) * (50.0 - 10.0) / (300.0 - 100.0)
        prior_alpha = np.clip(prior_alpha, 5.0, 100.0)  # Ensure reasonable bounds
        
        return decay_theta, prior_alpha
    
    @staticmethod
    def calculate_model_consensus_thresholds(
        model_consensus_tolerance: float,
        ess_min: float = 20.0,
        ess_max: float = 80.0,
        entropy_min: float = 0.4,
        entropy_max: float = 1.2
    ) -> Tuple[float, float]:
        """
        Calculate ess_threshold and entropy_threshold from single consensus tolerance parameter.
        
        Uses linear interpolation to map [0, 1] tolerance to financial min/max ranges.
        
        Args:
            model_consensus_tolerance: Single parameter in [0, 1] range
                0.0 = very strict (high ESS required, low entropy tolerated)
                1.0 = very permissive (low ESS accepted, high entropy tolerated)
            ess_min: Minimum ESS threshold (strict end)
            ess_max: Maximum ESS threshold (permissive end)
            entropy_min: Minimum entropy threshold (strict end)
            entropy_max: Maximum entropy threshold (permissive end)
            
        Returns:
            Tuple of (ess_threshold, entropy_threshold)
        """
        # ESS: Higher tolerance = lower threshold (easier to meet)
        # Invert the tolerance for ESS since high ESS is good
        ess_threshold = ess_max - model_consensus_tolerance * (ess_max - ess_min)
        
        # Entropy: Higher tolerance = higher threshold (more permissive)
        entropy_threshold = entropy_min + model_consensus_tolerance * (entropy_max - entropy_min)
        
        return ess_threshold, entropy_threshold
    
    @staticmethod
    def compute_posterior_mean_var(wins: int, losses: int, a: float, b: float) -> Tuple[float, float]:
        """
        Compute Beta posterior mean and variance.
        
        Args:
            wins: Number of winning trades
            losses: Number of losing trades
            a: Beta prior alpha (pseudo-wins)
            b: Beta prior beta (pseudo-losses)
            
        Returns:
            Tuple of (posterior_mean, posterior_var)
        """
        # Beta posterior: Beta(wins + a, losses + b)
        alpha_post = wins + a
        beta_post = losses + b
        
        # Mean: alpha / (alpha + beta)
        total = alpha_post + beta_post
        mean = alpha_post / total if total > 0 else 0.5
        
        # Variance: (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))
        if total > 0:
            variance = (alpha_post * beta_post) / (total ** 2 * (total + 1))
        else:
            variance = 0.25  # Maximum variance at p=0.5
        
        return mean, variance
    
    @staticmethod
    def compute_f_kelly(p_mean: float, R: float) -> float:
        """
        Compute standard Kelly fraction for position sizing.
        
        Formula: f = p - (1-p)/R
        where p = win probability, R = reward/risk ratio
        
        Args:
            p_mean: Posterior mean win probability
            R: Reward/risk ratio (conservative estimate)
            
        Returns:
            Kelly fraction
        """
        if R <= 0:
            return 0.0
        
        # Clamp p_mean to valid probability range
        p_mean = np.clip(p_mean, 0.001, 0.999)
        
        # Kelly formula
        f_kelly = p_mean - (1 - p_mean) / R
        
        # Kelly can be negative if edge is negative
        return max(0.0, f_kelly)
    
    @staticmethod
    def compute_leverage_kelly(p_mean: float, R: float, max_lev: float) -> float:
        """
        Compute Kelly-based leverage.
        
        Formula: lev = max_lev * [p - (1-p)/R]
        
        Args:
            p_mean: Posterior mean win probability
            R: Reward/risk ratio
            max_lev: Maximum allowed leverage
            
        Returns:
            Kelly leverage
        """
        f_kelly = DampenedKellyEngine.compute_f_kelly(p_mean, R)
        return max_lev * f_kelly
    
    def compute_lambda_eff(
        self,
        lambda_base: float,
        ess: float,
        var_p: float,
        entropy: float,
        ess_threshold: float,
        entropy_threshold: float
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute effective dampening factor (modular components).
        
        Combines three components:
        1. ESS factor: Sigmoid based on effective sample size
        2. Entropy factor: Penalty for high ensemble uncertainty
        3. Variance factor: Penalty for high posterior variance
        
        Args:
            lambda_base: Base dampening factor
            ess: Effective sample size from ensemble
            var_p: Posterior variance
            entropy: Ensemble entropy
            ess_threshold: ESS threshold (tau)
            entropy_threshold: Entropy veto threshold
            
        Returns:
            Tuple of (lambda_eff, component_dict)
        """
        # ESS component: sigmoid((ESS - tau) / kappa)
        ess_diff = ess - ess_threshold
        ess_factor = 1 / (1 + np.exp(-ess_diff / self.ess_sigmoid_kappa))
        
        # Entropy component: max(0, 1 - (entropy - threshold) / scale)
        if entropy > entropy_threshold:
            entropy_factor = max(0.0, 1 - (entropy - entropy_threshold) / self.entropy_scale)
        else:
            entropy_factor = 1.0
        
        # Variance component: 1 / (1 + var_p * penalty)
        var_factor = 1 / (1 + var_p * self.variance_penalty)
        
        # Combined lambda_eff
        lambda_eff = lambda_base * ess_factor * entropy_factor * var_factor
        
        # Ensure lambda_eff is in valid range [0, 1]
        lambda_eff = np.clip(lambda_eff, 0.0, 1.0)
        
        components = {
            'ess_factor': ess_factor,
            'entropy_factor': entropy_factor,
            'var_factor': var_factor
        }
        
        return lambda_eff, components
    
    @staticmethod
    def compute_f_final(
        f_kelly: float,
        lambda_eff: float,
        beta_base: float,
        beta_multiplier: float,
        f_floor: float = 0.005
    ) -> float:
        """
        Compute final dampened position size with exploration floor.
        
        Formula: f_final = f_floor + (lambda_eff - f_floor) * tanh(beta_effective * f_kelly)
        where beta_effective = beta_base * beta_multiplier
        
        Args:
            f_kelly: Kelly fraction
            lambda_eff: Effective dampening factor
            beta_base: Base beta parameter (shared denominator)
            beta_multiplier: Position-specific multiplier
            f_floor: Exploration floor (fixed at 0.005 for consistency)
            
        Returns:
            Final position size fraction
        """
        # Calculate effective beta (unified parameter structure)
        beta_effective = beta_base * beta_multiplier
        
        # Tanh dampening
        dampened = np.tanh(beta_effective * f_kelly)
        
        # Apply with exploration floor
        f_final = f_floor + (lambda_eff - f_floor) * dampened
        
        # Ensure non-negative
        return max(0.0, f_final)
    
    @staticmethod
    def compute_leverage_final(
        lev_kelly: float,
        lambda_eff: float,
        beta_base: float,
        beta_multiplier: float,
        lev_floor: float
    ) -> float:
        """
        Compute final dampened leverage with exploration floor.
        
        Uses same unified beta structure as position sizing: beta_effective = beta_base * beta_multiplier
        
        Args:
            lev_kelly: Kelly leverage
            lambda_eff: Effective dampening factor
            beta_base: Base beta parameter (shared with position sizing)
            beta_multiplier: Leverage-specific multiplier
            lev_floor: Exploration floor (minimum leverage)
            
        Returns:
            Final leverage
        """
        # Calculate effective beta (unified with position sizing)
        beta_effective = beta_base * beta_multiplier
        
        # Normalize leverage to [0, 1] range for tanh
        # Assume max leverage is captured in lev_kelly calculation
        # Apply tanh to the fraction
        if lev_kelly > 0:
            dampened = np.tanh(beta_effective * (lev_kelly / 10.0))  # Normalize by typical max leverage
            lev_final = lev_floor + (lambda_eff * 10.0 - lev_floor) * dampened
        else:
            lev_final = lev_floor
        
        return max(lev_floor, lev_final)
    
    @staticmethod
    def compute_kelly_fraction_clip(f_kelly: float, max_kelly_fraction: float) -> Tuple[float, bool]:
        """
        Clip Kelly fraction to never exceed maximum.
        
        Full Kelly is known to be too aggressive. Even dampened versions can be volatile.
        
        Args:
            f_kelly: Raw Kelly fraction
            max_kelly_fraction: Maximum allowed (e.g., 0.5 = half Kelly)
            
        Returns:
            Tuple of (clipped_f_kelly, was_clipped)
        """
        if f_kelly > max_kelly_fraction:
            return max_kelly_fraction, True
        return f_kelly, False
    
    @staticmethod
    def apply_drawdown_dampening(
        f_final: float,
        current_dd: float,
        max_dd: float,
        min_factor: float = 0.3
    ) -> Tuple[float, float]:
        """
        Apply drawdown-based dampening to position size.
        
        Formula: dd_factor = max(min_factor, 1 - current_dd / max_dd)
        
        Args:
            f_final: Final position size before drawdown adjustment
            current_dd: Current drawdown (as fraction, e.g., 0.10 = 10%)
            max_dd: Maximum acceptable drawdown
            min_factor: Minimum dampening factor (default 0.3 = 30%)
            
        Returns:
            Tuple of (adjusted_f_final, dd_factor)
        """
        if current_dd <= 0 or max_dd <= 0:
            return f_final, 1.0
        
        # Calculate dampening factor
        dd_factor = max(min_factor, 1 - current_dd / max_dd)
        
        # Apply to position size
        adjusted = f_final * dd_factor
        
        return adjusted, dd_factor
    
    def calculate_r_conservative(
        self,
        r_realized: List[float],
        default_r: float = 2.0
    ) -> Tuple[float, float, float, bool]:
        """
        Calculate conservative R from realized R distribution.
        
        Args:
            r_realized: List of realized R values from historical trades
            default_r: Default R if no history
            
        Returns:
            Tuple of (r_conservative, r_mean, r_std, is_unstable)
        """
        if not r_realized or len(r_realized) < 3:
            return default_r, default_r, 0.0, False
        
        r_array = np.array(r_realized)
        r_mean = np.mean(r_array)
        r_std = np.std(r_array)
        
        # Use conservative percentile
        r_conservative = np.percentile(r_array, self.r_percentile)
        
        # Check for instability
        if r_mean > 0:
            cv = r_std / r_mean  # Coefficient of variation
            is_unstable = cv > self.r_instability_threshold
        else:
            is_unstable = True
        
        # Ensure reasonable bounds
        r_conservative = max(0.5, r_conservative)  # At least 0.5:1
        
        return r_conservative, r_mean, r_std, is_unstable
    
    @handles_errors
    def calculate_position_and_leverage(
        self,
        wins: int,
        losses: int,
        regime_id: Optional[int],
        ess: float,
        entropy: float,
        r_realized: List[float],
        current_dd: float = 0.0,
        bin_merge_level: int = 0,
        bin_last_updated: Optional[datetime] = None,
        is_bin_stale: bool = False
    ) -> KellyResult:
        """
        Calculate both position size and leverage using unified dampened Kelly logic.
        
        This is the main entry point for the engine with reduced parameter space:
        - Unified beta structure (beta_base * multiplier)
        - n_min_samples derived from prior_alpha
        - Single model_consensus_tolerance parameter
        - Fixed f_floor and max_kelly_fraction
        
        Args:
            wins: Number of winning trades in bin
            losses: Number of losing trades in bin
            regime_id: Regime identifier (None if unknown)
            ess: Effective sample size from ensemble
            entropy: Ensemble entropy
            r_realized: List of realized R values from bin
            current_dd: Current portfolio drawdown
            bin_merge_level: Level of bin merging (0=exact, 1=regime-merged, 2=coarse, 3=prior)
            bin_last_updated: Last bin update timestamp
            is_bin_stale: Whether bin is stale (>90 days)
            
        Returns:
            KellyResult with all calculations and metadata
        """
        # Get regime-specific parameters
        params = self.get_regime_params(regime_id)
        
        # Core parameters (reduced set)
        lambda_base = params.get('lambda_base', 0.15)
        
        # Unified beta structure
        beta_base = params.get('beta_base', 1.0)
        beta_position_multiplier = params.get('beta_position_multiplier', 1.8)
        beta_leverage_multiplier = params.get('beta_leverage_multiplier', 1.2)
        
        # System half-life (single parameter controlling decay and prior)
        system_half_life = params.get('system_half_life', 200.0)
        decay_theta, prior_alpha = self.calculate_system_half_life_params(system_half_life)
        
        # Model consensus tolerance (single parameter for ESS and entropy)
        model_consensus_tolerance = params.get('model_consensus_tolerance', 0.5)
        ess_threshold, entropy_threshold = self.calculate_model_consensus_thresholds(
            model_consensus_tolerance
        )
        
        # n_min_samples enforced as ratio of prior_alpha
        n_min = int(prior_alpha / 2.0)
        
        # Leverage floor (still regime-specific)
        lev_floor = params.get('lev_floor', 1.2)
        
        # Initialize reason codes
        reason_codes = []
        
        # Check sample size
        n_total = wins + losses
        if n_total < n_min:
            reason_codes.append(ReasonCode.INSUFFICIENT_SAMPLES.value)
        
        if regime_id is None:
            reason_codes.append(ReasonCode.REGIME_UNKNOWN.value)
        
        if bin_merge_level > 0:
            reason_codes.append(ReasonCode.BIN_MERGED.value)
        
        if is_bin_stale:
            reason_codes.append("bin_stale")
        
        # Calculate conservative R from realized distribution
        r_conservative, r_mean, r_std, r_is_unstable = self.calculate_r_conservative(r_realized)
        
        if r_is_unstable:
            reason_codes.append(ReasonCode.R_INSTABILITY_DETECTED.value)
            # Increase prior weight for unstable R
            prior_alpha *= self.r_instability_prior_boost
        
        # Compute posterior
        posterior_mean, posterior_var = self.compute_posterior_mean_var(
            wins, losses, prior_alpha, prior_alpha  # Symmetric prior
        )
        
        # Compute Kelly fractions
        f_kelly = self.compute_f_kelly(posterior_mean, r_conservative)
        leverage_kelly = self.compute_leverage_kelly(posterior_mean, r_conservative, self.max_leverage)
        
        # Apply Kelly fraction clip
        f_kelly_clipped, was_clipped = self.compute_kelly_fraction_clip(f_kelly, self.max_kelly_fraction)
        if was_clipped:
            reason_codes.append(ReasonCode.KELLY_FRACTION_CLIPPED.value)
            f_kelly = f_kelly_clipped
        
        # Compute lambda_eff (modular components)
        lambda_eff, components = self.compute_lambda_eff(
            lambda_base, ess, posterior_var, entropy,
            ess_threshold, entropy_threshold
        )
        
        # Check for vetoes
        if ess < ess_threshold * 0.5:  # Very low ESS
            reason_codes.append(ReasonCode.ESS_LOW.value)
        
        if entropy > entropy_threshold:
            reason_codes.append(ReasonCode.ENTROPY_VETO.value)
        
        # Compute final dampened values using unified beta structure
        f_final = self.compute_f_final(
            f_kelly, lambda_eff, beta_base, beta_position_multiplier, self.f_floor
        )
        leverage_final = self.compute_leverage_final(
            leverage_kelly, lambda_eff, beta_base, beta_leverage_multiplier, lev_floor
        )
        
        # Apply drawdown dampening
        if current_dd > 0:
            f_final, dd_factor = self.apply_drawdown_dampening(
                f_final, current_dd, self.max_acceptable_drawdown
            )
            leverage_final *= dd_factor
            
            if dd_factor < 1.0:
                reason_codes.append(ReasonCode.DRAWDOWN_DAMPENING.value)
        else:
            dd_factor = 1.0
        
        # Apply hard caps
        f_final = min(f_final, self.max_per_trade_pct)
        leverage_final = min(leverage_final, self.max_leverage)
        
        if f_final >= self.max_per_trade_pct or leverage_final >= self.max_leverage:
            reason_codes.append(ReasonCode.HARD_CAP_APPLIED.value)
        
        # Create result
        result = KellyResult(
            f_final=f_final,
            leverage_final=leverage_final,
            f_kelly=f_kelly,
            leverage_kelly=leverage_kelly,
            lambda_eff=lambda_eff,
            posterior_mean=posterior_mean,
            posterior_var=posterior_var,
            ess=ess,
            entropy=entropy,
            regime_id=regime_id,
            bin_sample_count=n_total,
            bin_merge_level=bin_merge_level,
            bin_last_updated=bin_last_updated,
            is_bin_stale=is_bin_stale,
            r_realized_mean=r_mean,
            r_realized_std=r_std,
            r_conservative=r_conservative,
            dd_dampening_factor=dd_factor,
            kelly_fraction_clip_applied=was_clipped,
            correlation_adjusted=False,  # Will be set by position sizer
            config_version=self.get_config_version(),
            reason_codes=reason_codes,
            metadata={
                'lambda_eff_components': components,
                'regime_params_used': {
                    'lambda_base': lambda_base,
                    'beta_base': beta_base,
                    'beta_position_multiplier': beta_position_multiplier,
                    'beta_leverage_multiplier': beta_leverage_multiplier,
                    'system_half_life': system_half_life,
                    'decay_theta': decay_theta,
                    'prior_alpha': prior_alpha,
                    'model_consensus_tolerance': model_consensus_tolerance,
                    'ess_threshold': ess_threshold,
                    'entropy_threshold': entropy_threshold,
                    'n_min_samples': n_min
                }
            }
        )
        
        return result

