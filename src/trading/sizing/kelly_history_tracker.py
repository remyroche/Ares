"""
Kelly History Tracker - Adaptive Binning with Hierarchical Fallback

Tracks historical trade outcomes in 3D bins (score, volatility, regime) with:
- Adaptive bin merging when samples are insufficient
- Realized R tracking and distribution analysis
- Temporal integrity (purging, embargo periods)
- Regime-adaptive decay rates
- Stale bin detection

Three-level fallback hierarchy:
1. Exact bin (score_bucket, vol_bucket, regime_id)
2. Regime-agnostic (score_bucket, vol_bucket, all regimes)
3. Coarser bins (merge adjacent buckets)
4. Global prior (if all else fails)
"""

import numpy as np
import pickle
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path

from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
from src.printing import tprint

logger = system_logger.getChild('KellyHistoryTracker')


@dataclass
class BinData:
    """Data for a single bin."""
    wins: int = 0
    losses: int = 0
    r_realized: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)
    last_updated: Optional[datetime] = None
    merge_level: int = 0  # 0=exact, 1=regime-merged, 2=coarse, 3=prior
    is_stale: bool = False
    
    def total_samples(self) -> int:
        """Get total number of samples."""
        return self.wins + self.losses
    
    def win_rate(self) -> float:
        """Calculate win rate."""
        total = self.total_samples()
        return self.wins / total if total > 0 else 0.5
    
    def add_trade(
        self,
        is_win: bool,
        r_realized: float,
        timestamp: datetime,
        decay_theta: Optional[float] = None
    ) -> None:
        """
        Add a trade outcome to the bin.
        
        Args:
            is_win: Whether trade was a win
            r_realized: Realized reward/risk ratio
            timestamp: Trade timestamp
            decay_theta: Optional exponential decay factor
        """
        if decay_theta is not None and 0 < decay_theta < 1:
            # Apply exponential decay to existing counts
            self.wins *= decay_theta
            self.losses *= decay_theta
        
        # Add new trade
        if is_win:
            self.wins += 1
        else:
            self.losses += 1
        
        self.r_realized.append(r_realized)
        self.timestamps.append(timestamp)
        self.last_updated = timestamp
    
    def check_stale(self, current_time: datetime, stale_days: int = 90) -> bool:
        """
        Check if bin is stale (not updated recently).
        
        Args:
            current_time: Current timestamp
            stale_days: Days after which bin is considered stale
            
        Returns:
            True if stale
        """
        if self.last_updated is None:
            return True
        
        age = (current_time - self.last_updated).days
        self.is_stale = age > stale_days
        return self.is_stale
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'wins': self.wins,
            'losses': self.losses,
            'r_realized': self.r_realized.copy(),
            'timestamps': [ts.isoformat() for ts in self.timestamps],
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'merge_level': self.merge_level,
            'is_stale': self.is_stale
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BinData':
        """Create from dictionary."""
        timestamps = [datetime.fromisoformat(ts) for ts in data.get('timestamps', [])]
        last_updated = None
        if data.get('last_updated'):
            last_updated = datetime.fromisoformat(data['last_updated'])
        
        return cls(
            wins=data.get('wins', 0),
            losses=data.get('losses', 0),
            r_realized=data.get('r_realized', []).copy(),
            timestamps=timestamps,
            last_updated=last_updated,
            merge_level=data.get('merge_level', 0),
            is_stale=data.get('is_stale', False)
        )


class KellyHistoryTracker:
    """
    Adaptive Kelly history tracker with hierarchical bin fallback.
    
    Features:
    - 3D binning: (score_bucket, volatility_bucket, regime_id)
    - Adaptive merging: regime-agnostic → coarse → prior
    - Realized R tracking per bin
    - Temporal integrity: purging and embargo
    - Regime-adaptive decay rates
    - Stale bin detection and management
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize history tracker.

        Args:
            config: Configuration dictionary with binning and temporal settings
        """
        tprint(f"KellyHistoryTracker.__init__ called with config keys: {list(config.keys())}")

        self.config = config
        self.logger = logger.getChild('Tracker')

        # Extract configuration
        binning_config = config.get('binning', {})
        self.score_bins = np.array(binning_config.get('score_bins', [0.5, 0.6, 0.7, 0.8, 0.9]))
        self.volatility_bins = np.array(binning_config.get('volatility_bins', [0.005, 0.01, 0.02, 0.04]))
        self.enable_adaptive_merging = binning_config.get('enable_adaptive_merging', True)
        self.stale_bin_days = binning_config.get('stale_bin_days', 90)
        tprint(f"Binning config: {len(self.score_bins)} score bins, {len(self.volatility_bins)} vol bins, adaptive_merging={self.enable_adaptive_merging}, stale_days={self.stale_bin_days}")

        # Temporal integrity
        temporal_config = config.get('temporal', {})
        self.enable_purging = temporal_config.get('enable_purging', True)
        self.embargo_pct = temporal_config.get('embargo_pct_of_train', 0.05)
        self.overlap_detection = temporal_config.get('overlap_detection', True)
        tprint(f"Temporal integrity: purging={self.enable_purging}, embargo_pct={self.embargo_pct}, overlap_detection={self.overlap_detection}")

        # Regime-specific parameters
        self.regime_params = config.get('regime_params', {})
        tprint(f"Regime params: {len(self.regime_params)} regimes configured")

        # Storage: bins[regime_id][bin_key] = BinData
        self.bins: Dict[str, Dict[str, BinData]] = defaultdict(lambda: defaultdict(BinData))

        # Metadata
        self.version = 1
        self.created_at = datetime.now()
        self.last_saved = None

        # Regime stability tracking (for adaptive decay)
        self.regime_switches = []  # List of (timestamp, old_regime, new_regime)
        self.regime_stability_window = 1000  # bars for stability calculation
        tprint(f"Regime stability window: {self.regime_stability_window} bars")

        tprint_info("✅ Kelly History Tracker initialized")
        self.logger.info(f"Bins: {len(self.score_bins)} score × {len(self.volatility_bins)} vol")
    
    def _get_bin_key(self, score_bucket: int, vol_bucket: int) -> str:
        """Create bin key from bucket indices."""
        return f"s{score_bucket}_v{vol_bucket}"
    
    def _digitize_score(self, score: float) -> int:
        """Convert score to bucket index."""
        return int(np.digitize(score, self.score_bins))
    
    def _digitize_volatility(self, volatility: float) -> int:
        """Convert volatility to bucket index."""
        return int(np.digitize(volatility, self.volatility_bins))
    
    def _get_regime_key(self, regime_id: Optional[int]) -> str:
        """Convert regime ID to string key."""
        if regime_id is None:
            return "regime_unknown"
        return f"regime_{regime_id}"
    
    def _calculate_regime_stability(self, regime_id: int) -> float:
        """
        Calculate regime stability metric.
        
        Returns value between 0 and 1, where 1 = very stable (low turnover)
        
        Args:
            regime_id: Regime to check
            
        Returns:
            Stability metric (0-1)
        """
        if len(self.regime_switches) < 2:
            return 1.0  # Assume stable if insufficient data
        
        # Count switches involving this regime in recent window
        recent_switches = [
            s for s in self.regime_switches[-self.regime_stability_window:]
            if s[1] == regime_id or s[2] == regime_id
        ]
        
        # Stability = 1 - (switches / window_size)
        stability = 1.0 - len(recent_switches) / self.regime_stability_window
        return max(0.0, min(1.0, stability))
    
    def _get_decay_theta(self, regime_id: Optional[int]) -> float:
        """
        Get regime-adaptive decay rate.
        
        Stable regimes: slower decay (θ=0.98)
        Volatile regimes: faster decay (θ=0.90)
        
        Args:
            regime_id: Regime identifier
            
        Returns:
            Decay theta
        """
        if regime_id is None:
            return 0.90  # Default for unknown regime
        
        regime_key = self._get_regime_key(regime_id)
        regime_config = self.regime_params.get(regime_key, {})
        
        # If explicitly configured, use that
        if 'decay_theta' in regime_config:
            return regime_config['decay_theta']
        
        # Otherwise, adapt based on stability
        stability = self._calculate_regime_stability(regime_id)
        
        # Map stability [0, 1] to theta [0.90, 0.98]
        theta = 0.90 + stability * 0.08
        return theta
    
    @handles_errors
    def update_bin(
        self,
        score: float,
        volatility: float,
        regime_id: Optional[int],
        is_win: bool,
        r_realized: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Update bin with new trade outcome.

        Args:
            score: Model confidence score
            volatility: Market volatility (ATR or similar)
            regime_id: Regime identifier
            is_win: Whether trade was a win
            r_realized: Realized reward/risk ratio
            timestamp: Trade timestamp (defaults to now)
        """
        tprint(f"update_bin called: score={score:.3f}, volatility={volatility:.4f}, regime_id={regime_id}, is_win={is_win}, r_realized={r_realized:.2f}")

        if timestamp is None:
            timestamp = datetime.now()

        # Digitize to buckets
        score_bucket = self._digitize_score(score)
        vol_bucket = self._digitize_volatility(volatility)
        tprint(f"Digitized to buckets: score_bucket={score_bucket}, vol_bucket={vol_bucket}")

        # Get regime key
        regime_key = self._get_regime_key(regime_id)

        # Create bin key
        bin_key = self._get_bin_key(score_bucket, vol_bucket)
        tprint(f"Bin location: {regime_key}/{bin_key}")

        # Get decay rate for this regime
        decay_theta = self._get_decay_theta(regime_id)
        tprint(f"Decay theta for regime {regime_id}: {decay_theta:.4f}")

        # Update bin
        before_samples = self.bins[regime_key][bin_key].total_samples()
        self.bins[regime_key][bin_key].add_trade(is_win, r_realized, timestamp, decay_theta)
        after_samples = self.bins[regime_key][bin_key].total_samples()

        tprint(f"Bin updated: {regime_key}/{bin_key}, samples {before_samples:.1f}->{after_samples:.1f}, win={is_win}, R={r_realized:.2f}")
        self.logger.debug(f"Updated bin {regime_key}/{bin_key}: win={is_win}, R={r_realized:.2f}")
    
    def track_regime_switch(
        self,
        timestamp: datetime,
        old_regime: Optional[int],
        new_regime: Optional[int]
    ) -> None:
        """
        Track regime switch for stability calculation.
        
        Args:
            timestamp: When switch occurred
            old_regime: Previous regime
            new_regime: New regime
        """
        if old_regime != new_regime:
            self.regime_switches.append((timestamp, old_regime, new_regime))
            
            # Keep only recent history
            if len(self.regime_switches) > self.regime_stability_window * 2:
                self.regime_switches = self.regime_switches[-self.regime_stability_window:]
            
            self.logger.info(f"Regime switch: {old_regime} → {new_regime}")
    
    @handles_errors
    def lookup_bin(
        self,
        score: float,
        volatility: float,
        regime_id: Optional[int],
        n_min: int,
        current_time: Optional[datetime] = None
    ) -> Tuple[BinData, int]:
        """
        Lookup bin with adaptive fallback hierarchy.

        Hierarchy:
        1. Exact bin (score_bucket, vol_bucket, regime_id)
        2. Regime-agnostic (merge across regimes for same score/vol)
        3. Coarser bins (merge adjacent buckets)
        4. Global prior (return empty BinData with merge_level=3)

        Args:
            score: Model confidence score
            volatility: Market volatility
            regime_id: Regime identifier
            n_min: Minimum samples required
            current_time: Current time for staleness check

        Returns:
            Tuple of (BinData, merge_level)
        """
        tprint(f"lookup_bin called: score={score:.3f}, volatility={volatility:.4f}, regime_id={regime_id}, n_min={n_min}")

        if current_time is None:
            current_time = datetime.now()

        # Digitize to buckets
        score_bucket = self._digitize_score(score)
        vol_bucket = self._digitize_volatility(volatility)
        tprint(f"Digitized to buckets: score_bucket={score_bucket}, vol_bucket={vol_bucket}")

        # Level 0: Exact bin
        regime_key = self._get_regime_key(regime_id)
        bin_key = self._get_bin_key(score_bucket, vol_bucket)

        if regime_key in self.bins and bin_key in self.bins[regime_key]:
            bin_data = self.bins[regime_key][bin_key]
            bin_data.check_stale(current_time, self.stale_bin_days)
            samples = bin_data.total_samples()
            tprint(f"Level 0 - Exact bin found: {regime_key}/{bin_key}, samples={samples:.1f}, required={n_min}")

            if samples >= n_min:
                bin_data.merge_level = 0
                tprint(f"lookup_bin returning exact bin at level 0 with {samples:.1f} samples")
                return bin_data, 0

        # Level 1: Regime-agnostic merge (if adaptive merging enabled)
        if self.enable_adaptive_merging:
            merged_bin = self._merge_across_regimes(score_bucket, vol_bucket)
            samples = merged_bin.total_samples()
            tprint(f"Level 1 - Regime-agnostic merge: samples={samples:.1f}, required={n_min}")
            if samples >= n_min:
                merged_bin.merge_level = 1
                merged_bin.check_stale(current_time, self.stale_bin_days)
                self.logger.debug(f"Using regime-agnostic bin: {samples} samples")
                tprint(f"lookup_bin returning regime-agnostic bin at level 1 with {samples:.1f} samples")
                return merged_bin, 1

            # Level 2: Coarser bins (merge adjacent buckets)
            merged_bin = self._merge_coarse_bins(score_bucket, vol_bucket, regime_id)
            samples = merged_bin.total_samples()
            tprint(f"Level 2 - Coarse bin merge: samples={samples:.1f}, required={n_min}")
            if samples >= n_min:
                merged_bin.merge_level = 2
                merged_bin.check_stale(current_time, self.stale_bin_days)
                self.logger.debug(f"Using coarse bin: {samples} samples")
                tprint(f"lookup_bin returning coarse bin at level 2 with {samples:.1f} samples")
                return merged_bin, 2

        # Level 3: Global prior (fallback with no data)
        empty_bin = BinData(merge_level=3)
        self.logger.warning(f"Insufficient data, using global prior for s={score:.2f}, v={volatility:.4f}, r={regime_id}")
        tprint(f"lookup_bin returning global prior at level 3 (no data available)")
        return empty_bin, 3
    
    def _merge_across_regimes(self, score_bucket: int, vol_bucket: int) -> BinData:
        """
        Merge bin data across all regimes for given score/vol buckets.
        
        Args:
            score_bucket: Score bucket index
            vol_bucket: Volatility bucket index
            
        Returns:
            Merged BinData
        """
        merged = BinData()
        bin_key = self._get_bin_key(score_bucket, vol_bucket)
        
        for regime_key in self.bins:
            if bin_key in self.bins[regime_key]:
                bin_data = self.bins[regime_key][bin_key]
                merged.wins += bin_data.wins
                merged.losses += bin_data.losses
                merged.r_realized.extend(bin_data.r_realized)
                merged.timestamps.extend(bin_data.timestamps)
                
                if bin_data.last_updated:
                    if merged.last_updated is None or bin_data.last_updated > merged.last_updated:
                        merged.last_updated = bin_data.last_updated
        
        return merged
    
    def _merge_coarse_bins(
        self,
        score_bucket: int,
        vol_bucket: int,
        regime_id: Optional[int]
    ) -> BinData:
        """
        Merge adjacent bins to create coarser granularity.
        
        Merges ±1 bucket in both score and volatility dimensions.
        
        Args:
            score_bucket: Center score bucket
            vol_bucket: Center volatility bucket
            regime_id: Regime (or None)
            
        Returns:
            Merged BinData
        """
        merged = BinData()
        regime_key = self._get_regime_key(regime_id)
        
        # Search adjacent buckets (±1 in each dimension)
        for s_offset in [-1, 0, 1]:
            for v_offset in [-1, 0, 1]:
                s_bucket = score_bucket + s_offset
                v_bucket = vol_bucket + v_offset
                
                # Skip invalid buckets
                if s_bucket < 0 or v_bucket < 0:
                    continue
                
                bin_key = self._get_bin_key(s_bucket, v_bucket)
                
                if regime_key in self.bins and bin_key in self.bins[regime_key]:
                    bin_data = self.bins[regime_key][bin_key]
                    merged.wins += bin_data.wins
                    merged.losses += bin_data.losses
                    merged.r_realized.extend(bin_data.r_realized)
                    merged.timestamps.extend(bin_data.timestamps)
                    
                    if bin_data.last_updated:
                        if merged.last_updated is None or bin_data.last_updated > merged.last_updated:
                            merged.last_updated = bin_data.last_updated
        
        return merged
    
    def purge_overlapping_trades(
        self,
        train_end: datetime,
        test_start: datetime,
        max_trade_duration_days: int = 7
    ) -> int:
        """
        Purge trades that started in training but ended in test period.
        
        This prevents temporal leakage in walk-forward validation.
        
        Args:
            train_end: End of training period
            test_start: Start of test period
            max_trade_duration_days: Maximum expected trade duration
            
        Returns:
            Number of trades purged
        """
        if not self.enable_purging:
            return 0
        
        purged_count = 0
        cutoff = train_end - timedelta(days=max_trade_duration_days)
        
        for regime_key in self.bins:
            for bin_key in self.bins[regime_key]:
                bin_data = self.bins[regime_key][bin_key]
                
                # Find trades in the overlap window
                keep_indices = [
                    i for i, ts in enumerate(bin_data.timestamps)
                    if ts <= cutoff or ts >= test_start
                ]
                
                if len(keep_indices) < len(bin_data.timestamps):
                    # Purge overlapping trades
                    purged = len(bin_data.timestamps) - len(keep_indices)
                    purged_count += purged
                    
                    # Reconstruct lists
                    bin_data.timestamps = [bin_data.timestamps[i] for i in keep_indices]
                    bin_data.r_realized = [bin_data.r_realized[i] for i in keep_indices]
                    
                    # Recalculate wins/losses (conservative: remove proportionally)
                    if len(bin_data.timestamps) > 0:
                        keep_ratio = len(keep_indices) / (len(keep_indices) + purged)
                        bin_data.wins *= keep_ratio
                        bin_data.losses *= keep_ratio
                    else:
                        bin_data.wins = 0
                        bin_data.losses = 0
        
        if purged_count > 0:
            self.logger.info(f"Purged {purged_count} overlapping trades")
        
        return purged_count
    
    def get_embargo_period(self, train_window_days: int) -> int:
        """
        Calculate embargo period in days.
        
        Args:
            train_window_days: Training window length
            
        Returns:
            Embargo period in days
        """
        return max(1, int(train_window_days * self.embargo_pct))
    
    def check_staleness_all_bins(self, current_time: Optional[datetime] = None) -> Dict[str, int]:
        """
        Check staleness across all bins.
        
        Args:
            current_time: Current time (defaults to now)
            
        Returns:
            Dictionary with staleness statistics
        """
        if current_time is None:
            current_time = datetime.now()
        
        total_bins = 0
        stale_bins = 0
        
        for regime_key in self.bins:
            for bin_key in self.bins[regime_key]:
                total_bins += 1
                bin_data = self.bins[regime_key][bin_key]
                if bin_data.check_stale(current_time, self.stale_bin_days):
                    stale_bins += 1
        
        return {
            'total_bins': total_bins,
            'stale_bins': stale_bins,
            'stale_pct': stale_bins / total_bins if total_bins > 0 else 0.0
        }
    
    def get_bin_coverage_stats(self) -> Dict[str, Any]:
        """
        Get statistics about bin coverage.
        
        Returns:
            Dictionary with coverage statistics
        """
        total_bins_possible = len(self.score_bins) * len(self.volatility_bins) * len(self.regime_params)
        filled_bins = sum(len(regime_bins) for regime_bins in self.bins.values())
        
        sample_counts = []
        for regime_key in self.bins:
            for bin_key in self.bins[regime_key]:
                sample_counts.append(self.bins[regime_key][bin_key].total_samples())
        
        return {
            'total_possible_bins': total_bins_possible,
            'filled_bins': filled_bins,
            'coverage_pct': filled_bins / total_bins_possible if total_bins_possible > 0 else 0.0,
            'mean_samples_per_bin': np.mean(sample_counts) if sample_counts else 0.0,
            'median_samples_per_bin': np.median(sample_counts) if sample_counts else 0.0,
            'min_samples': np.min(sample_counts) if sample_counts else 0,
            'max_samples': np.max(sample_counts) if sample_counts else 0
        }
    
    @handles_errors
    def save_to_file(self, filepath: Path) -> None:
        """
        Save tracker to pickle file.
        
        Args:
            filepath: Path to save file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for serialization
        data = {
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'last_saved': datetime.now().isoformat(),
            'config': self.config,
            'bins': {
                regime_key: {
                    bin_key: bin_data.to_dict()
                    for bin_key, bin_data in regime_bins.items()
                }
                for regime_key, regime_bins in self.bins.items()
            },
            'regime_switches': [
                (ts.isoformat(), old_r, new_r)
                for ts, old_r, new_r in self.regime_switches
            ]
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        self.last_saved = datetime.now()
        tprint_success(f"✅ Saved Kelly bins to {filepath}")
        self.logger.info(f"Saved bins to {filepath}")
    
    @classmethod
    @handles_errors
    def load_from_file(cls, filepath: Path) -> 'KellyHistoryTracker':
        """
        Load tracker from pickle file.
        
        Args:
            filepath: Path to load from
            
        Returns:
            Loaded KellyHistoryTracker instance
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Create instance
        tracker = cls(data['config'])
        tracker.version = data['version']
        tracker.created_at = datetime.fromisoformat(data['created_at'])
        tracker.last_saved = datetime.fromisoformat(data['last_saved']) if data.get('last_saved') else None
        
        # Load bins
        for regime_key, regime_bins in data['bins'].items():
            for bin_key, bin_dict in regime_bins.items():
                tracker.bins[regime_key][bin_key] = BinData.from_dict(bin_dict)
        
        # Load regime switches
        if 'regime_switches' in data:
            tracker.regime_switches = [
                (datetime.fromisoformat(ts), old_r, new_r)
                for ts, old_r, new_r in data['regime_switches']
            ]
        
        tprint_info(f"✅ Loaded Kelly bins from {filepath}")
        logger.info(f"Loaded bins from {filepath}")
        
        return tracker

