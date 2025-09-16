"""
Unified Progress Reporter

Provides consistent progress reporting across all HMM training components.
"""

from typing import List, Optional, Dict, Any
import time
import logging

# Optional imports for external dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

logger = logging.getLogger(__name__)


class ProgressReporter:
    """Unified progress reporting for training operations."""
    
    def __init__(self, total_models: int, show_progress: bool = True):
        """
        Initialize progress reporter.
        
        Args:
            total_models: Total number of models to train
            show_progress: Whether to show real-time progress
        """
        self.total_models = total_models
        self.completed_models = 0
        self.start_time = time.time()
        self.model_times = []
        self.successful_models = 0
        self.failed_models = 0
        self.show_progress = show_progress
        self.model_results = []
    
    def update_progress(self, model_name: str, success: bool, training_time: float, 
                       accuracy: Optional[float] = None, error_message: Optional[str] = None) -> None:
        """
        Update progress after each model training.
        
        Args:
            model_name: Name of the model
            success: Whether training was successful
            training_time: Time taken for training
            accuracy: Model accuracy (if successful)
            error_message: Error message (if failed)
        """
        self.completed_models += 1
        self.model_times.append(training_time)
        
        if success:
            self.successful_models += 1
        else:
            self.failed_models += 1
        
        # Store result details
        result = {
            'model_name': model_name,
            'success': success,
            'training_time': training_time,
            'accuracy': accuracy,
            'error_message': error_message,
            'timestamp': time.time()
        }
        self.model_results.append(result)
        
        if self.show_progress:
            self._print_progress(model_name, success, training_time, accuracy)
    
    def _print_progress(self, model_name: str, success: bool, training_time: float, 
                       accuracy: Optional[float] = None) -> None:
        """Print progress information."""
        progress_percent = (self.completed_models / self.total_models) * 100
        if self.model_times:
            if NUMPY_AVAILABLE:
                avg_time = np.mean(self.model_times)
            else:
                avg_time = sum(self.model_times) / len(self.model_times)
        else:
            avg_time = 0
        eta = avg_time * (self.total_models - self.completed_models)
        
        status = "✅" if success else "❌"
        accuracy_str = f" (acc: {accuracy:.4f})" if accuracy is not None else ""
        
        print(f"\r{status} {model_name}{accuracy_str} | Progress: {progress_percent:.1f}% | "
              f"Success: {self.successful_models}/{self.completed_models} | "
              f"ETA: {eta:.1f}s", end="", flush=True)
    
    def finish_report(self) -> Dict[str, Any]:
        """
        Generate final progress report.
        
        Returns:
            Dictionary with summary statistics
        """
        total_time = time.time() - self.start_time
        
        if self.show_progress:
            print(f"\n\n🎯 Training Summary:")
            print(f"   Total time: {total_time:.2f}s")
            if self.model_times:
                if NUMPY_AVAILABLE:
                    print(f"   Average time per model: {np.mean(self.model_times):.2f}s")
                    print(f"   Fastest model: {np.min(self.model_times):.2f}s")
                    print(f"   Slowest model: {np.max(self.model_times):.2f}s")
                else:
                    print(f"   Average time per model: {sum(self.model_times)/len(self.model_times):.2f}s")
                    print(f"   Fastest model: {min(self.model_times):.2f}s")
                    print(f"   Slowest model: {max(self.model_times):.2f}s")
            print(f"   Successful models: {self.successful_models}/{self.total_models}")
            print(f"   Success rate: {(self.successful_models/self.total_models)*100:.1f}%")
        
        # Calculate additional statistics
        successful_times = [r['training_time'] for r in self.model_results if r['success']]
        failed_times = [r['training_time'] for r in self.model_results if not r['success']]
        
        summary = {
            'total_time': total_time,
            'total_models': self.total_models,
            'successful_models': self.successful_models,
            'failed_models': self.failed_models,
            'success_rate': (self.successful_models / self.total_models) * 100,
            'average_training_time': (sum(self.model_times) / len(self.model_times)) if self.model_times else 0,
            'fastest_training_time': min(self.model_times) if self.model_times else 0,
            'slowest_training_time': max(self.model_times) if self.model_times else 0,
            'successful_training_times': successful_times,
            'failed_training_times': failed_times,
            'model_results': self.model_results
        }
        
        return summary
    
    def get_current_progress(self) -> Dict[str, Any]:
        """Get current progress information."""
        progress_percent = (self.completed_models / self.total_models) * 100
        if self.model_times:
            if NUMPY_AVAILABLE:
                avg_time = np.mean(self.model_times)
            else:
                avg_time = sum(self.model_times) / len(self.model_times)
        else:
            avg_time = 0
        eta = avg_time * (self.total_models - self.completed_models)
        
        return {
            'completed_models': self.completed_models,
            'total_models': self.total_models,
            'progress_percent': progress_percent,
            'successful_models': self.successful_models,
            'failed_models': self.failed_models,
            'average_time': avg_time,
            'eta': eta
        }
    
    def reset(self) -> None:
        """Reset progress reporter for new training session."""
        self.completed_models = 0
        self.start_time = time.time()
        self.model_times = []
        self.successful_models = 0
        self.failed_models = 0
        self.model_results = []