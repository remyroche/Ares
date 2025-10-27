"""
Enhanced Chunked Processing System

This module provides intelligent chunked processing for HDBSCAN clustering
with proper temporal continuity, cluster merging, and memory management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import gc
import warnings

logger = logging.getLogger(__name__)


@dataclass
class ChunkProcessingConfig:
    """Configuration for chunked processing."""
    enable_chunked_processing: bool = True
    chunk_size: int = 1000
    chunk_overlap: float = 0.1  # 10% overlap between chunks
    enable_temporal_continuity: bool = True
    merge_similar_clusters: bool = True
    similarity_threshold: float = 0.8
    max_memory_gb: float = 8.0
    enable_garbage_collection: bool = True
    progress_callback: Optional[callable] = None


class ClusterSimilarityCalculator:
    """Calculator for cluster similarity and merging."""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_cluster_similarity(self, cluster1_features: np.ndarray, 
                                   cluster2_features: np.ndarray) -> float:
        """
        Calculate similarity between two clusters.
        
        Args:
            cluster1_features: Feature matrix for cluster 1
            cluster2_features: Feature matrix for cluster 2
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        try:
            # Calculate centroid similarity
            centroid1 = np.mean(cluster1_features, axis=0)
            centroid2 = np.mean(cluster2_features, axis=0)
            
            # Cosine similarity between centroids
            centroid_similarity = cosine_similarity(
                centroid1.reshape(1, -1), 
                centroid2.reshape(1, -1)
            )[0, 0]
            
            # Calculate distribution similarity (if both clusters have enough points)
            if len(cluster1_features) > 1 and len(cluster2_features) > 1:
                # Use covariance matrix similarity
                cov1 = np.cov(cluster1_features.T)
                cov2 = np.cov(cluster2_features.T)
                
                # Frobenius norm similarity
                cov_similarity = 1.0 - np.linalg.norm(cov1 - cov2) / (
                    np.linalg.norm(cov1) + np.linalg.norm(cov2) + 1e-8
                )
                
                # Combine centroid and distribution similarity
                overall_similarity = 0.7 * centroid_similarity + 0.3 * cov_similarity
            else:
                overall_similarity = centroid_similarity
            
            return max(0.0, min(1.0, overall_similarity))
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate cluster similarity: {e}")
            return 0.0
    
    def find_similar_clusters(self, cluster_features: Dict[int, np.ndarray]) -> List[Tuple[int, int, float]]:
        """
        Find similar clusters that should be merged.
        
        Args:
            cluster_features: Dictionary mapping cluster labels to feature matrices
            
        Returns:
            List of (cluster1, cluster2, similarity) tuples
        """
        similar_pairs = []
        cluster_labels = list(cluster_features.keys())
        
        for i, label1 in enumerate(cluster_labels):
            for label2 in cluster_labels[i+1:]:
                similarity = self.calculate_cluster_similarity(
                    cluster_features[label1],
                    cluster_features[label2]
                )
                
                if similarity >= self.similarity_threshold:
                    similar_pairs.append((label1, label2, similarity))
        
        # Sort by similarity (highest first)
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return similar_pairs
    
    def merge_clusters(self, cluster_labels: np.ndarray, 
                      merge_pairs: List[Tuple[int, int, float]]) -> np.ndarray:
        """
        Merge similar clusters in the label array.
        
        Args:
            cluster_labels: Original cluster labels
            merge_pairs: List of (cluster1, cluster2, similarity) tuples to merge
            
        Returns:
            Updated cluster labels with merged clusters
        """
        merged_labels = cluster_labels.copy()
        
        # Create mapping for merged clusters
        cluster_mapping = {}
        for label1, label2, similarity in merge_pairs:
            # Use the smaller label as the target
            target_label = min(label1, label2)
            source_label = max(label1, label2)
            
            # Map source to target
            cluster_mapping[source_label] = target_label
            
            # Update any existing mappings
            for existing_source, existing_target in cluster_mapping.items():
                if existing_target == source_label:
                    cluster_mapping[existing_source] = target_label
        
        # Apply mappings
        for source_label, target_label in cluster_mapping.items():
            merged_labels[merged_labels == source_label] = target_label
        
        return merged_labels


class TemporalContinuityManager:
    """Manager for maintaining temporal continuity across chunks."""
    
    def __init__(self, overlap_ratio: float = 0.1):
        self.overlap_ratio = overlap_ratio
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_chunk_boundaries(self, data_size: int, chunk_size: int) -> List[Tuple[int, int]]:
        """
        Create chunk boundaries with overlap for temporal continuity.
        
        Args:
            data_size: Total size of the dataset
            chunk_size: Size of each chunk
            
        Returns:
            List of (start, end) tuples for each chunk
        """
        if not self.overlap_ratio > 0:
            # No overlap - simple chunking
            boundaries = []
            for start in range(0, data_size, chunk_size):
                end = min(start + chunk_size, data_size)
                boundaries.append((start, end))
            return boundaries
        
        # Calculate overlap size
        overlap_size = int(chunk_size * self.overlap_ratio)
        
        boundaries = []
        start = 0
        
        while start < data_size:
            end = min(start + chunk_size, data_size)
            boundaries.append((start, end))
            
            # Move start position with overlap
            if end < data_size:
                start = end - overlap_size
            else:
                break
        
        return boundaries
    
    def align_chunk_labels(self, chunk_labels: np.ndarray, 
                          global_labels: np.ndarray,
                          chunk_start: int, chunk_end: int) -> np.ndarray:
        """
        Align chunk labels with global labels using overlap region.
        
        Args:
            chunk_labels: Labels from current chunk
            global_labels: Global labels array
            chunk_start: Start index of current chunk
            chunk_end: End index of current chunk
            
        Returns:
            Aligned chunk labels
        """
        if len(global_labels) == 0:
            return chunk_labels
        
        # Find overlap region
        overlap_start = max(0, chunk_start)
        overlap_end = min(len(global_labels), chunk_end)
        
        if overlap_start >= overlap_end:
            return chunk_labels
        
        # Get overlap labels from both global and chunk
        global_overlap = global_labels[overlap_start:overlap_end]
        chunk_overlap_start = overlap_start - chunk_start
        chunk_overlap_end = overlap_end - chunk_start
        chunk_overlap = chunk_labels[chunk_overlap_start:chunk_overlap_end]
        
        # Find label mapping
        label_mapping = self._find_label_mapping(global_overlap, chunk_overlap)
        
        # Apply mapping to chunk labels
        aligned_labels = chunk_labels.copy()
        for chunk_label, global_label in label_mapping.items():
            aligned_labels[chunk_labels == chunk_label] = global_label
        
        return aligned_labels
    
    def _find_label_mapping(self, global_labels: np.ndarray, 
                           chunk_labels: np.ndarray) -> Dict[int, int]:
        """
        Find mapping between chunk labels and global labels.
        
        Args:
            global_labels: Labels from global array
            chunk_labels: Labels from chunk array
            
        Returns:
            Dictionary mapping chunk labels to global labels
        """
        mapping = {}
        
        # Find unique labels in both arrays
        global_unique = np.unique(global_labels[global_labels != -1])
        chunk_unique = np.unique(chunk_labels[chunk_labels != -1])
        
        # Create label correspondence matrix
        correspondence_matrix = np.zeros((len(chunk_unique), len(global_unique)))
        
        for i, chunk_label in enumerate(chunk_unique):
            for j, global_label in enumerate(global_unique):
                # Count co-occurrences
                chunk_mask = chunk_labels == chunk_label
                global_mask = global_labels == global_label
                co_occurrences = np.sum(chunk_mask & global_mask)
                
                # Normalize by minimum cluster size
                min_size = min(np.sum(chunk_mask), np.sum(global_mask))
                if min_size > 0:
                    correspondence_matrix[i, j] = co_occurrences / min_size
        
        # Find best mappings
        for i, chunk_label in enumerate(chunk_unique):
            best_global_idx = np.argmax(correspondence_matrix[i])
            best_correspondence = correspondence_matrix[i, best_global_idx]
            
            if best_correspondence > 0.5:  # Threshold for mapping
                global_label = global_unique[best_global_idx]
                mapping[chunk_label] = global_label
        
        return mapping


class MemoryManager:
    """Manager for memory usage and garbage collection."""
    
    def __init__(self, max_memory_gb: float = 8.0, enable_gc: bool = True):
        self.max_memory_gb = max_memory_gb
        self.enable_gc = enable_gc
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def check_memory_usage(self) -> float:
        """Check current memory usage in GB."""
        try:
            import psutil
            process = psutil.Process()
            memory_gb = process.memory_info().rss / (1024 ** 3)
            return memory_gb
        except ImportError:
            self.logger.warning("psutil not available for memory monitoring")
            return 0.0
    
    def should_trigger_gc(self) -> bool:
        """Check if garbage collection should be triggered."""
        if not self.enable_gc:
            return False
        
        memory_gb = self.check_memory_usage()
        return memory_gb > self.max_memory_gb * 0.8  # Trigger at 80% of max memory
    
    def cleanup_memory(self):
        """Clean up memory by triggering garbage collection."""
        if self.enable_gc:
            gc.collect()
            self.logger.debug("Memory cleanup triggered")


class EnhancedChunkedProcessor:
    """Enhanced chunked processor with temporal continuity and intelligent merging."""
    
    def __init__(self, config: ChunkProcessingConfig):
        self.config = config
        self.similarity_calculator = ClusterSimilarityCalculator(config.similarity_threshold)
        self.temporal_manager = TemporalContinuityManager(config.chunk_overlap)
        self.memory_manager = MemoryManager(config.max_memory_gb, config.enable_garbage_collection)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def process_chunks(self, features: np.ndarray, 
                      clustering_func: callable,
                      timestamps: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Process data in chunks with temporal continuity.
        
        Args:
            features: Feature matrix to cluster
            clustering_func: Function to perform clustering on a chunk
            timestamps: Optional timestamps for temporal analysis
            
        Returns:
            Dictionary containing clustering results and metadata
        """
        if not self.config.enable_chunked_processing or len(features) <= self.config.chunk_size:
            # Process entire dataset at once
            return self._process_single_chunk(features, clustering_func, timestamps)
        
        start_time = time.time()
        
        # Create chunk boundaries
        chunk_boundaries = self.temporal_manager.create_chunk_boundaries(
            len(features), self.config.chunk_size
        )
        
        self.logger.info(f"Processing {len(features)} samples in {len(chunk_boundaries)} chunks")
        
        # Process chunks
        chunk_results = []
        global_labels = np.full(len(features), -1, dtype=int)
        global_cluster_features = {}
        next_global_label = 0
        
        for i, (chunk_start, chunk_end) in enumerate(chunk_boundaries):
            self.logger.info(f"Processing chunk {i+1}/{len(chunk_boundaries)}: {chunk_start}-{chunk_end}")
            
            # Check memory usage
            if self.memory_manager.should_trigger_gc():
                self.memory_manager.cleanup_memory()
            
            # Process current chunk
            chunk_features = features[chunk_start:chunk_end]
            chunk_timestamps = timestamps[chunk_start:chunk_end] if timestamps is not None else None
            
            chunk_result = self._process_single_chunk(
                chunk_features, clustering_func, chunk_timestamps
            )
            
            # Align labels with global labels
            if i > 0 and self.config.enable_temporal_continuity:
                chunk_result['labels'] = self.temporal_manager.align_chunk_labels(
                    chunk_result['labels'], global_labels, chunk_start, chunk_end
                )
            
            # Update global labels
            global_labels[chunk_start:chunk_end] = chunk_result['labels']
            
            # Store cluster features for similarity analysis
            for label in np.unique(chunk_result['labels']):
                if label != -1:
                    cluster_mask = chunk_result['labels'] == label
                    cluster_features = chunk_features[cluster_mask]
                    
                    if label in global_cluster_features:
                        # Append to existing cluster
                        global_cluster_features[label] = np.vstack([
                            global_cluster_features[label], cluster_features
                        ])
                    else:
                        # Create new cluster
                        global_cluster_features[label] = cluster_features
            
            chunk_results.append(chunk_result)
            
            # Progress callback
            if self.config.progress_callback:
                self.config.progress_callback(i + 1, len(chunk_boundaries))
        
        # Merge similar clusters
        if self.config.merge_similar_clusters and len(global_cluster_features) > 1:
            self.logger.info("Merging similar clusters across chunks")
            similar_pairs = self.similarity_calculator.find_similar_clusters(global_cluster_features)
            
            if similar_pairs:
                self.logger.info(f"Found {len(similar_pairs)} similar cluster pairs to merge")
                global_labels = self.similarity_calculator.merge_clusters(global_labels, similar_pairs)
        
        # Final cleanup
        self.memory_manager.cleanup_memory()
        
        # Compile results
        processing_time = time.time() - start_time
        
        results = {
            'labels': global_labels,
            'n_clusters': len(np.unique(global_labels[global_labels != -1])),
            'n_noise_points': np.sum(global_labels == -1),
            'noise_ratio': np.sum(global_labels == -1) / len(global_labels),
            'chunk_results': chunk_results,
            'processing_time': processing_time,
            'chunk_count': len(chunk_boundaries),
            'merged_clusters': len(similar_pairs) if 'similar_pairs' in locals() else 0
        }
        
        self.logger.info(f"Chunked processing completed in {processing_time:.2f}s")
        self.logger.info(f"Final clusters: {results['n_clusters']}, Noise ratio: {results['noise_ratio']:.3f}")
        
        return results
    
    def _process_single_chunk(self, features: np.ndarray, 
                             clustering_func: callable,
                             timestamps: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Process a single chunk of data."""
        try:
            # Perform clustering
            cluster_result = clustering_func(features)
            
            # Extract labels
            if isinstance(cluster_result, dict):
                labels = cluster_result.get('labels', np.full(len(features), -1))
            else:
                labels = cluster_result
            
            # Calculate basic metrics
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels[unique_labels != -1])
            n_noise_points = np.sum(labels == -1)
            noise_ratio = n_noise_points / len(labels) if len(labels) > 0 else 0.0
            
            return {
                'labels': labels,
                'n_clusters': n_clusters,
                'n_noise_points': n_noise_points,
                'noise_ratio': noise_ratio,
                'unique_labels': unique_labels
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process chunk: {e}")
            return {
                'labels': np.full(len(features), -1),
                'n_clusters': 0,
                'n_noise_points': len(features),
                'noise_ratio': 1.0,
                'unique_labels': np.array([-1])
            }
    
    def validate_chunked_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate results from chunked processing.
        
        Args:
            results: Results from chunked processing
            
        Returns:
            Validation results
        """
        validation = {
            'passed': True,
            'issues': [],
            'recommendations': []
        }
        
        # Check for reasonable number of clusters
        n_clusters = results.get('n_clusters', 0)
        if n_clusters == 0:
            validation['issues'].append("No clusters found")
            validation['recommendations'].append("Consider adjusting clustering parameters")
        elif n_clusters > 20:
            validation['issues'].append(f"Too many clusters: {n_clusters}")
            validation['recommendations'].append("Consider increasing min_cluster_size")
        
        # Check noise ratio
        noise_ratio = results.get('noise_ratio', 0.0)
        if noise_ratio > 0.5:
            validation['issues'].append(f"High noise ratio: {noise_ratio:.3f}")
            validation['recommendations'].append("Consider reducing min_cluster_size or min_samples")
        
        # Check processing time
        processing_time = results.get('processing_time', 0.0)
        if processing_time > 300:  # 5 minutes
            validation['issues'].append(f"Long processing time: {processing_time:.2f}s")
            validation['recommendations'].append("Consider reducing chunk size or using faster clustering")
        
        # Overall validation
        validation['passed'] = len(validation['issues']) == 0
        
        return validation


def create_chunked_processor(config: ChunkProcessingConfig) -> EnhancedChunkedProcessor:
    """Factory function to create a chunked processor."""
    return EnhancedChunkedProcessor(config)