"""
Matrix Profile Pure Price Pattern Discovery

This module uses matrix profile analysis to discover recurring price motifs
(frequently occurring price movement subsequences) in pure price action data.

Key Approach:
1. Calculate matrix profile of price return series
2. Identify top motifs (most frequently occurring price patterns)
3. Analyze motif contexts and outcomes
4. Convert motifs to mathematical pattern definitions
5. Generate gradient-based targets for motif patterns

Matrix Profile Benefits:
- Finds exact recurring subsequences
- Parameter-free motif discovery
- Handles variable-length patterns
- Identifies seasonal/cyclical price behaviors
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

from src.utils.logger import system_logger


class MotifPatternType(Enum):
    """Types of matrix profile discovered patterns."""
    RECURRING_MOTIF = "recurring_motif"
    SEASONAL_MOTIF = "seasonal_motif"
    DISCORD_PATTERN = "discord_pattern"  # Rare/unusual patterns


@dataclass
class MatrixProfilePattern:
    """Matrix profile discovered price pattern."""
    pattern_id: str
    motif_type: MotifPatternType
    description: str
    binary_labels: pd.Series
    intensity_gradients: pd.Series
    motif_length: int
    occurrence_count: int
    motif_distance: float  # How similar motif occurrences are
    example_sequence: List[float]
    mathematical_approximation: str
    
    @property
    def frequency(self) -> float:
        return self.binary_labels.sum() / len(self.binary_labels)
    
    @property
    def is_significant(self) -> bool:
        return self.occurrence_count >= 5 and self.frequency >= 0.02


class MatrixProfilePriceDiscovery:
    """Matrix profile-based discovery of pure price patterns."""
    
    def __init__(self):
        self.logger = system_logger.getChild('MatrixProfileDiscovery')
    
    def discover_matrix_profile_patterns(self, 
                                       prices: pd.Series,
                                       motif_length: int = 20,
                                       max_motifs: int = 10) -> List[MatrixProfilePattern]:
        """
        Discover price patterns using matrix profile analysis.
        
        NOTE: This is a conceptual implementation. For production use:
        ```python
        import stumpy
        
        # Calculate matrix profile
        mp = stumpy.stump(price_returns, m=motif_length)
        
        # Find motifs
        motifs = stumpy.motifs(price_returns, mp, max_motifs=max_motifs)
        ```
        """
        
        self.logger.info(f"📊 Discovering matrix profile patterns (motif_length={motif_length})")
        
        # Prepare price returns for analysis
        returns = prices.pct_change().fillna(0)
        
        if len(returns) < motif_length * 10:
            self.logger.warning("Insufficient data for matrix profile analysis")
            return []
        
        # Simulate matrix profile calculation
        # (In real implementation, use stumpy library)
        motifs, discords = self._simulate_matrix_profile_analysis(returns, motif_length, max_motifs)
        
        discovered_patterns = []
        
        # Process motifs (recurring patterns)
        for motif_id, motif_info in enumerate(motifs):
            pattern = self._create_motif_pattern(
                motif_info, motif_id, prices, returns, MotifPatternType.RECURRING_MOTIF
            )
            if pattern and pattern.is_significant:
                discovered_patterns.append(pattern)
        
        # Process discords (rare patterns)
        for discord_id, discord_info in enumerate(discords):
            pattern = self._create_motif_pattern(
                discord_info, discord_id, prices, returns, MotifPatternType.DISCORD_PATTERN
            )
            if pattern and pattern.is_significant:
                discovered_patterns.append(pattern)
        
        self.logger.info(f"✅ Matrix profile discovery completed: {len(discovered_patterns)} patterns")
        return discovered_patterns
    
    def _simulate_matrix_profile_analysis(self, 
                                        returns: pd.Series,
                                        motif_length: int,
                                        max_motifs: int) -> Tuple[List[Dict], List[Dict]]:
        """
        Simulate matrix profile analysis.
        
        Real implementation would use:
        ```python
        import stumpy
        
        # Calculate matrix profile
        mp = stumpy.stump(returns.values, m=motif_length)
        
        # Find motifs and discords
        motifs = stumpy.motifs(returns.values, mp, max_motifs=max_motifs)
        discords = stumpy.discords(returns.values, mp, max_discords=3)
        ```
        """
        
        self.logger.info("🔍 Simulating matrix profile analysis...")
        
        # Simulate by finding similar subsequences manually
        motifs = []
        discords = []
        
        # Extract all subsequences
        subsequences = []
        for i in range(len(returns) - motif_length + 1):
            subseq = returns.iloc[i:i+motif_length].values
            subsequences.append((i, subseq))
        
        # Find motifs by comparing subsequences
        used_indices = set()
        
        for i, (start_idx, subseq) in enumerate(subsequences):
            if start_idx in used_indices:
                continue
            
            # Find similar subsequences
            similar_indices = [start_idx]
            
            for j, (other_start_idx, other_subseq) in enumerate(subsequences):
                if other_start_idx != start_idx and other_start_idx not in used_indices:
                    # Calculate similarity (Euclidean distance)
                    distance = np.linalg.norm(subseq - other_subseq)
                    
                    # If similar enough, consider it same motif
                    if distance < np.std(subseq) * 0.5:  # Similarity threshold
                        similar_indices.append(other_start_idx)
                        used_indices.add(other_start_idx)
            
            # If found enough similar sequences, it's a motif
            if len(similar_indices) >= 3:
                motif_info = {
                    'indices': similar_indices,
                    'sequence': subseq,
                    'avg_distance': np.mean([
                        np.linalg.norm(subseq - subsequences[idx][1]) 
                        for idx in range(len(subsequences))
                        if subsequences[idx][0] in similar_indices
                    ]),
                    'occurrence_count': len(similar_indices)
                }
                motifs.append(motif_info)
                
                for idx in similar_indices:
                    used_indices.add(idx)
        
        # Find discords (most unusual subsequences)
        remaining_subsequences = [
            (start_idx, subseq) for start_idx, subseq in subsequences
            if start_idx not in used_indices
        ]
        
        if remaining_subsequences:
            # Calculate how unusual each remaining subsequence is
            unusualness_scores = []
            
            for start_idx, subseq in remaining_subsequences:
                # Calculate distance to all other subsequences
                distances = []
                for other_start_idx, other_subseq in subsequences:
                    if other_start_idx != start_idx:
                        distance = np.linalg.norm(subseq - other_subseq)
                        distances.append(distance)
                
                # Unusualness = minimum distance to any other subsequence
                unusualness = min(distances) if distances else 0
                unusualness_scores.append((unusualness, start_idx, subseq))
            
            # Top 3 most unusual
            unusualness_scores.sort(reverse=True)
            
            for unusualness, start_idx, subseq in unusualness_scores[:3]:
                discord_info = {
                    'indices': [start_idx],
                    'sequence': subseq,
                    'unusualness_score': unusualness,
                    'occurrence_count': 1
                }
                discords.append(discord_info)
        
        return motifs, discords
    
    def _create_motif_pattern(self, 
                            motif_info: Dict,
                            pattern_id: int,
                            prices: pd.Series,
                            returns: pd.Series,
                            motif_type: MotifPatternType) -> Optional[MatrixProfilePattern]:
        """Create pattern from motif information."""
        
        indices = motif_info['indices']
        motif_sequence = motif_info['sequence']
        occurrence_count = motif_info['occurrence_count']
        
        if len(indices) < 3:  # Need minimum occurrences
            return None
        
        # Create pattern labels
        pattern_labels = pd.Series(0, index=prices.index)
        intensity_gradients = pd.Series(0.0, index=prices.index)
        
        # Calculate intensity based on motif quality
        if motif_type == MotifPatternType.RECURRING_MOTIF:
            base_intensity = 1.0 - (motif_info['avg_distance'] / np.std(motif_sequence))
            base_intensity = max(0.3, min(base_intensity, 1.0))
        else:  # Discord pattern
            base_intensity = min(motif_info['unusualness_score'] / np.std(motif_sequence), 1.0)
        
        for idx in indices:
            if idx < len(pattern_labels):
                pattern_labels.iloc[idx] = 1
                intensity_gradients.iloc[idx] = base_intensity
        
        # Generate description and mathematical approximation
        description, math_approximation = self._analyze_motif_characteristics(
            motif_sequence, occurrence_count, motif_type, pattern_id
        )
        
        return MatrixProfilePattern(
            pattern_id=f"motif_{pattern_id}",
            motif_type=motif_type,
            description=description,
            binary_labels=pattern_labels,
            intensity_gradients=intensity_gradients,
            motif_length=len(motif_sequence),
            occurrence_count=occurrence_count,
            motif_distance=motif_info.get('avg_distance', 0),
            example_sequence=motif_sequence.tolist(),
            mathematical_approximation=math_approximation
        )
    
    def _analyze_motif_characteristics(self, 
                                     motif_sequence: np.ndarray,
                                     occurrence_count: int,
                                     motif_type: MotifPatternType,
                                     pattern_id: int) -> Tuple[str, str]:
        """Analyze characteristics of discovered motif."""
        
        # Calculate motif characteristics
        total_movement = np.sum(motif_sequence)
        max_movement = np.max(np.abs(motif_sequence))
        volatility = np.std(motif_sequence)
        
        # Determine pattern characteristics
        if abs(total_movement) > 0.02:
            direction = "upward" if total_movement > 0 else "downward"
            if volatility < 0.01:
                pattern_style = "smooth"
            else:
                pattern_style = "volatile"
            
            description = f"Recurring {pattern_style} {direction} price movement pattern"
            math_approximation = f"Directional motif: total_movement ≈ {total_movement:.3f}, volatility ≈ {volatility:.3f}"
            
        elif max_movement > 0.03:
            description = f"Recurring high-volatility price movement pattern"
            math_approximation = f"Volatile motif: max_movement ≈ {max_movement:.3f}, volatility ≈ {volatility:.3f}"
            
        elif volatility < 0.005:
            description = f"Recurring low-volatility consolidation pattern"
            math_approximation = f"Consolidation motif: volatility ≈ {volatility:.3f}"
            
        else:
            description = f"Recurring mixed price movement pattern"
            math_approximation = f"Mixed motif: characteristics vary"
        
        # Add occurrence information
        description += f" (occurs {occurrence_count} times)"
        
        if motif_type == MotifPatternType.DISCORD_PATTERN:
            description = description.replace("Recurring", "Rare")
            math_approximation += " (discord - rare pattern)"
        
        return description, math_approximation
    
    def generate_matrix_profile_report(self, 
                                     discovered_patterns: List[MatrixProfilePattern]) -> str:
        """Generate comprehensive matrix profile discovery report."""
        
        report = []
        report.append("# Matrix Profile Pure Price Pattern Discovery Report")
        report.append("=" * 60)
        report.append("")
        report.append("**Method**: Matrix Profile Motif Discovery")
        report.append("**Focus**: Recurring pure price movement subsequences")
        report.append("**Innovation**: Parameter-free discovery of exact price patterns")
        report.append("")
        
        # Summary
        total_patterns = len(discovered_patterns)
        significant_patterns = sum(1 for p in discovered_patterns if p.is_significant)
        
        if total_patterns > 0:
            avg_occurrences = np.mean([p.occurrence_count for p in discovered_patterns])
            total_occurrences = sum(p.occurrence_count for p in discovered_patterns)
        else:
            avg_occurrences = 0
            total_occurrences = 0
        
        report.append("## Matrix Profile Discovery Summary")
        report.append("")
        report.append(f"- **Total Motifs Discovered**: {total_patterns}")
        report.append(f"- **Significant Motifs**: {significant_patterns}")
        report.append(f"- **Average Occurrences per Motif**: {avg_occurrences:.1f}")
        report.append(f"- **Total Pattern Occurrences**: {total_occurrences}")
        report.append("")
        
        # Motif analysis
        for pattern in discovered_patterns:
            if pattern.is_significant:
                report.append(f"### {pattern.pattern_id}")
                report.append("")
                report.append(f"**Type**: {pattern.motif_type.value}")
                report.append(f"**Description**: {pattern.description}")
                report.append(f"**Occurrences**: {pattern.occurrence_count}")
                report.append(f"**Frequency**: {pattern.frequency:.3f} ({pattern.frequency*100:.1f}% of periods)")
                report.append(f"**Motif Length**: {pattern.motif_length} periods")
                report.append(f"**Max Intensity**: {pattern.intensity_gradients.max():.3f}")
                report.append(f"**Mathematical Approximation**: {pattern.mathematical_approximation}")
                report.append("")
                
                # Example sequence
                if pattern.example_sequence:
                    report.append(f"**Example Price Movement Sequence**:")
                    formatted_sequence = [f"{val:.4f}" for val in pattern.example_sequence[:10]]
                    report.append(f"   [{', '.join(formatted_sequence)}...]")
                    report.append("")
        
        # Implementation guidance
        report.append("## Real Implementation with STUMPY")
        report.append("")
        report.append("### Installation")
        report.append("```bash")
        report.append("pip install stumpy")
        report.append("```")
        report.append("")
        
        report.append("### Implementation")
        report.append("```python")
        report.append("import stumpy")
        report.append("import numpy as np")
        report.append("")
        report.append("# Prepare price returns")
        report.append("returns = prices.pct_change().dropna()")
        report.append("")
        report.append("# Calculate matrix profile")
        report.append("mp = stumpy.stump(returns.values, m=motif_length)")
        report.append("")
        report.append("# Find top motifs")
        report.append("motifs = stumpy.motifs(returns.values, mp, max_motifs=10)")
        report.append("")
        report.append("# Analyze each motif")
        report.append("for motif in motifs:")
        report.append("    motif_indices = motif[1]  # Indices of motif occurrences")
        report.append("    motif_sequence = returns.iloc[motif[0]:motif[0]+motif_length]")
        report.append("    ")
        report.append("    # Create pattern definition")
        report.append("    pattern_definition = analyze_motif_pattern(motif_sequence)")
        report.append("    ")
        report.append("    # Generate gradient targets")
        report.append("    binary_labels, intensities = create_motif_targets(motif_indices)")
        report.append("```")
        report.append("")
        
        report.append("### Discord Discovery")
        report.append("```python")
        report.append("# Find discords (rare patterns)")
        report.append("discords = stumpy.discords(returns.values, mp, max_discords=5)")
        report.append("")
        report.append("for discord in discords:")
        report.append("    discord_sequence = returns.iloc[discord[0]:discord[0]+motif_length]")
        report.append("    unusual_pattern = analyze_discord_pattern(discord_sequence)")
        report.append("```")
        
        return "\n".join(report)
    
    def _simulate_matrix_profile_analysis(self, 
                                        returns: pd.Series,
                                        motif_length: int,
                                        max_motifs: int) -> Tuple[List[Dict], List[Dict]]:
        """Simulate matrix profile analysis (conceptual implementation)."""
        
        # Extract all subsequences
        subsequences = []
        for i in range(len(returns) - motif_length + 1):
            subseq = returns.iloc[i:i+motif_length].values
            subsequences.append((i, subseq))
        
        # Find motifs by similarity
        motifs = []
        used_indices = set()
        
        for i, (start_idx, subseq) in enumerate(subsequences):
            if start_idx in used_indices or len(motifs) >= max_motifs:
                continue
            
            # Find similar subsequences
            similar_indices = [start_idx]
            
            for j, (other_start_idx, other_subseq) in enumerate(subsequences):
                if (other_start_idx != start_idx and 
                    other_start_idx not in used_indices and
                    abs(other_start_idx - start_idx) > motif_length):  # No overlap
                    
                    # Calculate normalized distance
                    distance = np.linalg.norm(subseq - other_subseq) / np.linalg.norm(subseq)
                    
                    if distance < 0.3:  # Similarity threshold
                        similar_indices.append(other_start_idx)
            
            # If found enough similar sequences, it's a motif
            if len(similar_indices) >= 3:
                # Calculate average distance between motif instances
                distances = []
                for idx1 in similar_indices:
                    for idx2 in similar_indices:
                        if idx1 != idx2:
                            seq1 = returns.iloc[idx1:idx1+motif_length].values
                            seq2 = returns.iloc[idx2:idx2+motif_length].values
                            distance = np.linalg.norm(seq1 - seq2)
                            distances.append(distance)
                
                avg_distance = np.mean(distances) if distances else 0
                
                motif_info = {
                    'indices': similar_indices,
                    'sequence': subseq,
                    'avg_distance': avg_distance,
                    'occurrence_count': len(similar_indices)
                }
                motifs.append(motif_info)
                
                # Mark indices as used
                for idx in similar_indices:
                    used_indices.add(idx)
        
        # Find discords (most unusual patterns)
        discords = []
        remaining_subsequences = [
            (start_idx, subseq) for start_idx, subseq in subsequences
            if start_idx not in used_indices
        ]
        
        if remaining_subsequences:
            # Calculate unusualness scores
            unusualness_scores = []
            
            for start_idx, subseq in remaining_subsequences:
                # Calculate minimum distance to any other subsequence
                min_distance = float('inf')
                
                for other_start_idx, other_subseq in subsequences:
                    if other_start_idx != start_idx:
                        distance = np.linalg.norm(subseq - other_subseq)
                        min_distance = min(min_distance, distance)
                
                unusualness_scores.append((min_distance, start_idx, subseq))
            
            # Top 3 most unusual
            unusualness_scores.sort(reverse=True)
            
            for unusualness, start_idx, subseq in unusualness_scores[:3]:
                discord_info = {
                    'indices': [start_idx],
                    'sequence': subseq,
                    'unusualness_score': unusualness,
                    'occurrence_count': 1
                }
                discords.append(discord_info)
        
        return motifs, discords
    
    def _create_motif_pattern(self,
                            motif_info: Dict,
                            pattern_id: int,
                            prices: pd.Series,
                            returns: pd.Series,
                            motif_type: MotifPatternType) -> Optional[MatrixProfilePattern]:
        """Create pattern from motif information."""
        
        indices = motif_info['indices']
        motif_sequence = motif_info['sequence']
        occurrence_count = motif_info['occurrence_count']
        
        # Create pattern labels and intensities
        pattern_labels = pd.Series(0, index=prices.index)
        intensity_gradients = pd.Series(0.0, index=prices.index)
        
        # Calculate intensity based on motif type
        if motif_type == MotifPatternType.RECURRING_MOTIF:
            # Higher occurrence count = higher intensity
            base_intensity = min(occurrence_count / 10.0, 1.0)
            # Lower average distance = higher intensity
            distance_factor = 1.0 - min(motif_info['avg_distance'] / np.std(motif_sequence), 1.0)
            intensity = base_intensity * distance_factor
        else:  # Discord
            # Unusualness score determines intensity
            intensity = min(motif_info['unusualness_score'] / np.std(motif_sequence), 1.0)
        
        intensity = max(0.1, intensity)  # Minimum intensity
        
        for idx in indices:
            if idx < len(pattern_labels):
                pattern_labels.iloc[idx] = 1
                intensity_gradients.iloc[idx] = intensity
        
        # Generate description and approximation
        description, math_approximation = self._analyze_motif_price_behavior(
            motif_sequence, occurrence_count, motif_type, pattern_id
        )
        
        return MatrixProfilePattern(
            pattern_id=f"motif_{pattern_id}",
            motif_type=motif_type,
            description=description,
            binary_labels=pattern_labels,
            intensity_gradients=intensity_gradients,
            motif_length=len(motif_sequence),
            occurrence_count=occurrence_count,
            motif_distance=motif_info.get('avg_distance', 0),
            example_sequence=motif_sequence.tolist(),
            mathematical_approximation=math_approximation
        )
    
    def _analyze_motif_price_behavior(self, 
                                    motif_sequence: np.ndarray,
                                    occurrence_count: int,
                                    motif_type: MotifPatternType,
                                    pattern_id: int) -> Tuple[str, str]:
        """Analyze price behavior characteristics of motif."""
        
        # Calculate motif characteristics
        total_return = np.sum(motif_sequence)
        max_return = np.max(np.abs(motif_sequence))
        volatility = np.std(motif_sequence)
        
        # Identify pattern shape
        if abs(total_return) > 0.02:
            direction = "upward" if total_return > 0 else "downward"
            if volatility < 0.01:
                shape = "smooth"
            else:
                shape = "volatile"
            
            description = f"Recurring {shape} {direction} price movement"
            math_approximation = f"{direction.title()} motif: return ≈ {total_return:.3f}, volatility ≈ {volatility:.3f}"
            
        elif max_return > 0.025:
            description = f"Recurring high-volatility price oscillation"
            math_approximation = f"Volatile motif: max_movement ≈ {max_return:.3f}"
            
        else:
            description = f"Recurring low-volatility price pattern"
            math_approximation = f"Low-vol motif: volatility ≈ {volatility:.3f}"
        
        # Add frequency information
        description += f" (found {occurrence_count} times)"
        
        if motif_type == MotifPatternType.DISCORD_PATTERN:
            description = description.replace("Recurring", "Rare")
            math_approximation += " (unusual pattern)"
        
        return description, math_approximation


class MatrixProfileOrchestrator:
    """Main orchestrator for matrix profile pattern discovery."""
    
    def __init__(self):
        self.logger = system_logger.getChild('MatrixProfileOrchestrator')
        self.discoverer = MatrixProfilePriceDiscovery()
    
    def run_complete_matrix_profile_analysis(self, 
                                           prices: pd.Series,
                                           motif_lengths: List[int] = [10, 15, 20, 25]) -> Dict[str, List[MatrixProfilePattern]]:
        """Run complete matrix profile analysis with multiple motif lengths."""
        
        self.logger.info("📊 Starting complete matrix profile analysis")
        
        all_results = {}
        
        for motif_length in motif_lengths:
            self.logger.info(f"🔍 Analyzing motif length {motif_length}")
            
            try:
                patterns = self.discoverer.discover_matrix_profile_patterns(
                    prices, motif_length=motif_length
                )
                all_results[f"length_{motif_length}"] = patterns
                
                significant_count = sum(1 for p in patterns if p.is_significant)
                self.logger.info(f"   ✅ Found {significant_count}/{len(patterns)} significant patterns")
                
            except Exception as e:
                self.logger.error(f"   ❌ Failed for length {motif_length}: {e}")
                all_results[f"length_{motif_length}"] = []
        
        total_significant = sum(
            sum(1 for p in patterns if p.is_significant)
            for patterns in all_results.values()
        )
        
        self.logger.info(f"🎯 Matrix profile analysis completed: {total_significant} total significant patterns")
        return all_results
    
    def export_matrix_profile_targets(self, 
                                    results: Dict[str, List[MatrixProfilePattern]]) -> Dict[str, pd.DataFrame]:
        """Export matrix profile patterns as ML targets."""
        
        exports = {
            'binary_labels': {},
            'intensity_gradients': {},
            'combined': {}
        }
        
        # Combine all significant patterns
        all_significant_patterns = []
        for length_results in results.values():
            all_significant_patterns.extend([p for p in length_results if p.is_significant])
        
        if not all_significant_patterns:
            return {k: pd.DataFrame() for k in exports.keys()}
        
        # Binary labels
        binary_data = {}
        for pattern in all_significant_patterns:
            binary_data[pattern.pattern_id] = pattern.binary_labels
        
        exports['binary_labels'] = pd.DataFrame(binary_data)
        
        # Intensity gradients
        intensity_data = {}
        for pattern in all_significant_patterns:
            intensity_data[f"{pattern.pattern_id}_intensity"] = pattern.intensity_gradients
        
        exports['intensity_gradients'] = pd.DataFrame(intensity_data)
        
        # Combined
        combined_data = {**binary_data, **intensity_data}
        exports['combined'] = pd.DataFrame(combined_data)
        
        return exports


def get_matrix_profile_implementation_guide() -> Dict[str, str]:
    """Get implementation guide for real matrix profile analysis."""
    
    return {
        "Installation": """
        pip install stumpy
        # STUMPY is the premier library for matrix profile analysis
        """,
        
        "Basic Implementation": """
        import stumpy
        import numpy as np
        import pandas as pd
        
        # Prepare data
        returns = prices.pct_change().dropna()
        
        # Calculate matrix profile
        mp = stumpy.stump(returns.values, m=20)  # 20-period motifs
        
        # Find top motifs
        motifs = stumpy.motifs(returns.values, mp, max_motifs=10)
        
        # Find discords (rare patterns)
        discords = stumpy.discords(returns.values, mp, max_discords=5)
        """,
        
        "Pattern Analysis": """
        # Analyze each motif
        for motif in motifs:
            motif_idx = motif[0]  # Index of first occurrence
            motif_indices = motif[1]  # All occurrence indices
            
            # Extract motif sequence
            motif_sequence = returns.iloc[motif_idx:motif_idx+20]
            
            # Analyze motif characteristics
            total_movement = motif_sequence.sum()
            volatility = motif_sequence.std()
            
            # Create pattern definition
            if abs(total_movement) > 0.02:
                pattern_type = "directional_motif"
            elif volatility > 0.03:
                pattern_type = "volatile_motif"
            else:
                pattern_type = "consolidation_motif"
        """,
        
        "Gradient Target Generation": """
        # Create ML targets from motifs
        binary_labels = pd.Series(0, index=prices.index)
        intensity_gradients = pd.Series(0.0, index=prices.index)
        
        for motif in motifs:
            motif_indices = motif[1]
            motif_quality = 1.0 / (motif[2] + 1e-6)  # Distance-based quality
            
            for idx in motif_indices:
                if idx < len(binary_labels):
                    binary_labels.iloc[idx] = 1
                    intensity_gradients.iloc[idx] = min(motif_quality, 1.0)
        """
    }


def run_matrix_profile_discovery_example():
    """Example of matrix profile-based pure price pattern discovery."""
    
    print("Matrix Profile Pure Price Pattern Discovery")
    print("==========================================")
    print()
    print("📊 MATRIX PROFILE APPROACH:")
    print("   1. Find exact recurring price movement subsequences")
    print("   2. Parameter-free motif discovery")
    print("   3. Identify seasonal/cyclical price behaviors")
    print("   4. Discover rare/unusual price patterns (discords)")
    print("   5. Generate gradient-based intensity targets")
    print()
    print("Expected discoveries:")
    print("- Recurring price movement motifs")
    print("- Seasonal price patterns")
    print("- Unusual price behaviors (discords)")
    print("- Exact subsequence matches")
    print()
    
    implementation_guide = get_matrix_profile_implementation_guide()
    print("Implementation requirements:")
    print(implementation_guide["Installation"])
    print()
    print("Usage:")
    print("```python")
    print("orchestrator = MatrixProfileOrchestrator()")
    print("results = orchestrator.run_complete_matrix_profile_analysis(prices)")
    print("targets = orchestrator.export_matrix_profile_targets(results)")
    print("```")


if __name__ == "__main__":
    run_matrix_profile_discovery_example()