"""Path Manager Module.

Handles path generation and management for artifacts with step categorization.
"""

from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from .logger import system_logger


# Step category mapping for organized artifact storage
STEP_CATEGORIES = {
    'data_collection': ['step01', 'data_downloader', 'klines_downloading_processing'],
    'market_analysis': ['step02', 'market_analysis', 'sr_detection', 'regime_discovery'],
    'pre_training': ['step02_5', 'feature_generation', 'pre_training'],
    'models_training': ['step03', 'model_training', 'analyst_models', 'tactician_models'],
    'backtesting': ['step04', 'backtesting', 'real_parameters_optimization']
}


class PathManager:
    """Handles path generation and management for artifacts."""
    
    def __init__(self, base_dir: Path):
        """Initialize path manager.
        
        Args:
            base_dir: Base directory for artifacts
        """
        self.base_dir = base_dir
        self.logger = system_logger.getChild("PathManager")
        
        # Current context
        self._current_step_name: Optional[str] = None
        self._current_symbol: Optional[str] = None
        self._current_exchange: Optional[str] = None
        self._current_datetime: Optional[datetime] = None
        self._current_information: Optional[str] = None
        self._current_direction: str = "long"
        self._current_model: str = "Analyst"
        
        # Path configuration
        self.include_symbol_in_filename: bool = True
        self.include_exchange_in_filename: bool = True
        self.include_datetime_in_filename: bool = True
        self.include_information_in_filename: bool = True
        self.include_direction_in_filename: bool = True
        self.include_model_in_filename: bool = True
    
    def set_context(self, step_name: str, symbol: Optional[str] = None, 
                   exchange: Optional[str] = None, datetime_param: Optional[datetime] = None, 
                   information: Optional[str] = None, direction: str = "long", 
                   model: str = "Analyst") -> None:
        """Set the current context for path generation.
        
        Args:
            step_name: Name of the current step
            symbol: Trading symbol
            exchange: Exchange name
            datetime: Current datetime
            information: Additional information
            direction: Trading direction
            model: Model name
        """
        self._current_step_name = step_name
        self._current_symbol = symbol
        self._current_exchange = exchange
        self._current_datetime = datetime_param or datetime.now()
        self._current_information = information
        self._current_direction = direction
        self._current_model = model
        
        self.logger.debug(f"Context set: step={step_name}, symbol={symbol}, exchange={exchange}")
    
    def get_step_category(self, step_name: str) -> str:
        """Determine the category for a step based on its name.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Step category
        """
        step_name_lower = step_name.lower()
        for category, patterns in STEP_CATEGORIES.items():
            if any(pattern.lower() in step_name_lower for pattern in patterns):
                return category
        return 'pre_training'  # Default fallback
    
    def generate_filename(self, key: str, step_name: str, file_extension: str = "parquet") -> str:
        """Generate enhanced filename with context information.
        
        Args:
            key: Artifact key
            step_name: Step name
            file_extension: File extension
            
        Returns:
            Generated filename
        """
        parts = []
        
        # Add information prefix if configured and available
        if self.include_information_in_filename and self._current_information:
            parts.append(self._current_information)
        
        # Add step name
        parts.append(step_name)
        
        # Add key
        parts.append(key)
        
        # Add symbol if configured and available
        if self.include_symbol_in_filename and self._current_symbol:
            parts.append(self._current_symbol)
        
        # Add exchange if configured and available
        if self.include_exchange_in_filename and self._current_exchange:
            parts.append(self._current_exchange)
        
        # Add direction if configured
        if self.include_direction_in_filename and self._current_direction:
            parts.append(self._current_direction)
        
        # Add model if configured
        if self.include_model_in_filename and self._current_model:
            parts.append(self._current_model)
        
        # Add datetime if configured
        if self.include_datetime_in_filename and self._current_datetime:
            datetime_str = self._current_datetime.strftime("%Y%m%d_%H%M%S")
            parts.append(datetime_str)
        
        # Join parts with underscores and add extension
        filename = "_".join(parts) + f".{file_extension}"
        
        self.logger.debug(f"Generated filename: {filename}")
        return filename
    
    def get_artifact_path(self, step_name: str, key: str, file_extension: str = "parquet") -> Path:
        """Get full path for an artifact with proper directory structure.
        
        Args:
            step_name: Name of the step
            key: Artifact key
            file_extension: File extension
            
        Returns:
            Full path to the artifact
        """
        # Determine step category
        step_category = self.get_step_category(step_name)
        
        # Create directory structure: artifacts/step_category/symbol/exchange/direction/model/step_name/
        path_parts = [self.base_dir, step_category]
        
        if self._current_symbol:
            path_parts.append(self._current_symbol)
        
        if self._current_exchange:
            path_parts.append(self._current_exchange)
        
        if self._current_direction:
            path_parts.append(self._current_direction)
        
        if self._current_model:
            path_parts.append(self._current_model)
        
        path_parts.append(step_name)
        
        step_dir = Path(*path_parts)
        step_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        filename = self.generate_filename(key, step_name, file_extension)
        full_path = step_dir / filename
        
        self.logger.debug(f"Generated artifact path: {full_path}")
        return full_path
    
    def find_artifact(self, step_name: str, key: str, artifact_type: str = "data") -> Optional[Path]:
        """Find an artifact using multiple search strategies.
        
        Args:
            step_name: Name of the step
            key: Artifact key
            artifact_type: Type of artifact
            
        Returns:
            Path to the artifact if found, None otherwise
        """
        # Primary: Try step-category structure
        step_category = self.get_step_category(step_name)
        artifact_path = self._find_artifact_in_category(step_category, key, artifact_type)
        
        if artifact_path and artifact_path.exists():
            return artifact_path
        
        # Fallback 1: Direct search in base directory
        fallback_path = self._find_artifact_fallback(key, artifact_type)
        if fallback_path and fallback_path.exists():
            return fallback_path
        
        # Fallback 2: Fuzzy search
        fuzzy_path = self._find_artifact_fuzzy(key, artifact_type)
        if fuzzy_path and fuzzy_path.exists():
            return fuzzy_path
        
        return None
    
    def _find_artifact_in_category(self, step_category: str, key: str, 
                                  artifact_type: str) -> Optional[Path]:
        """Find artifact in step-category structure."""
        try:
            category_dir = self.base_dir / step_category
            if not category_dir.exists():
                return None
            
            # Search recursively for the artifact
            for file_path in category_dir.rglob(f"*{key}*"):
                if file_path.is_file() and self._is_correct_file_type(file_path, artifact_type):
                    return file_path
            
            return None
        except Exception as e:
            self.logger.warning(f"Failed to search in category {step_category}: {e}")
            return None
    
    def _find_artifact_fallback(self, key: str, artifact_type: str) -> Optional[Path]:
        """Find artifact in fallback search."""
        try:
            if not self.base_dir.exists():
                return None
            
            # Search patterns
            search_patterns = [
                f"*{key}*",
                f"*{key}*.parquet",
                f"*{key}*.csv",
                f"*{key}*.pkl",
                f"*{key}*.json",
            ]
            
            for pattern in search_patterns:
                for file_path in self.base_dir.rglob(pattern):
                    if file_path.is_file() and self._is_correct_file_type(file_path, artifact_type):
                        if key.lower() in file_path.name.lower():
                            return file_path
            
            return None
        except Exception as e:
            self.logger.warning(f"Failed to search in fallback: {e}")
            return None
    
    def _find_artifact_fuzzy(self, key: str, artifact_type: str) -> Optional[Path]:
        """Find artifact using fuzzy matching."""
        try:
            if not self.base_dir.exists():
                return None
            
            # Search in all subdirectories
            for file_path in self.base_dir.rglob("*"):
                if file_path.is_file() and self._is_correct_file_type(file_path, artifact_type):
                    if self._is_similar_name(key, file_path.stem):
                        return file_path
            
            return None
        except Exception as e:
            self.logger.warning(f"Failed to search with fuzzy matching: {e}")
            return None
    
    def _is_correct_file_type(self, file_path: Path, artifact_type: str) -> bool:
        """Check if the file type matches the expected artifact type."""
        try:
            file_extension = file_path.suffix.lower()
            
            # Map artifact types to expected file extensions
            type_mappings = {
                "data": [".parquet", ".csv", ".json"],
                "model": [".pkl", ".joblib", ".h5", ".onnx"],
                "metadata": [".json", ".yaml", ".yml"],
                "image": [".png", ".jpg", ".jpeg", ".svg"],
                "text": [".txt", ".md", ".log"]
            }
            
            expected_extensions = type_mappings.get(artifact_type, [".parquet", ".csv", ".json", ".pkl"])
            return file_extension in expected_extensions
        except Exception:
            return True  # Default to True if we can't determine
    
    def _is_similar_name(self, name1: str, name2: str) -> bool:
        """Check if two names are similar (for fuzzy matching)."""
        try:
            # Simple similarity check
            name1_clean = name1.lower().replace('_', '').replace('-', '')
            name2_clean = name2.lower().replace('_', '').replace('-', '')
            
            # Check if one is contained in the other
            if name1_clean in name2_clean or name2_clean in name1_clean:
                return True
            
            # Check for common patterns
            common_patterns = ['data', 'model', 'result', 'output', 'input']
            for pattern in common_patterns:
                if pattern in name1_clean and pattern in name2_clean:
                    return True
            
            return False
        except Exception:
            return False
    
    def ensure_directories(self) -> None:
        """Ensure all step category directories exist."""
        try:
            # Ensure base directory exists
            self.base_dir.mkdir(parents=True, exist_ok=True)
            
            # Ensure all step category directories exist
            for category in STEP_CATEGORIES.keys():
                category_dir = self.base_dir / category
                category_dir.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Ensured directory exists: {category_dir}")
            
            self.logger.info(f"All step category directories ensured in: {self.base_dir}")
        except Exception as e:
            self.logger.error(f"Failed to ensure step category directories: {e}")
            raise