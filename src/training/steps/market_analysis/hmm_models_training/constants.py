"""
Constants for HMM Models Training

Centralized constants to avoid magic numbers and improve maintainability.
"""

# Validation thresholds
class ValidationThresholds:
    """Validation threshold constants."""
    
    # Basic validation level
    BASIC_MIN_SAMPLES = 100
    BASIC_MIN_SAMPLES_PER_REGIME = 5
    BASIC_MAX_MISSING_RATIO = 0.5
    BASIC_MAX_INFINITE_RATIO = 0.1
    BASIC_MIN_FEATURE_VARIANCE = 1e-8
    BASIC_MAX_CORRELATION_THRESHOLD = 0.99
    
    # Standard validation level
    STANDARD_MIN_SAMPLES = 1000
    STANDARD_MIN_SAMPLES_PER_REGIME = 50
    STANDARD_MAX_MISSING_RATIO = 0.1
    STANDARD_MAX_INFINITE_RATIO = 0.01
    STANDARD_MIN_FEATURE_VARIANCE = 1e-6
    STANDARD_MAX_CORRELATION_THRESHOLD = 0.95
    
    # Strict validation level
    STRICT_MIN_SAMPLES = 5000
    STRICT_MIN_SAMPLES_PER_REGIME = 200
    STRICT_MAX_MISSING_RATIO = 0.01
    STRICT_MAX_INFINITE_RATIO = 0.001
    STRICT_MIN_FEATURE_VARIANCE = 1e-4
    STRICT_MAX_CORRELATION_THRESHOLD = 0.9


# Training configuration limits
class TrainingLimits:
    """Training configuration limits."""
    
    MAX_FEATURES = 10000
    MAX_SEQUENCE_LENGTH = 1000
    MAX_REGIMES = 20
    MAX_HPO_TRIALS = 1000
    MAX_BATCH_SIZE = 10000
    MAX_TOTAL_PARAMS = 1000000
    
    # Memory thresholds
    HIGH_MEMORY_THRESHOLD = 80  # Percentage
    CRITICAL_MEMORY_THRESHOLD = 85  # Percentage
    
    # Timeout settings
    LIGHT_TRAINING_TIMEOUT = 60  # seconds
    STANDARD_TRAINING_TIMEOUT = 300  # seconds
    
    # Performance thresholds
    MIN_ACCEPTABLE_ACCURACY = 0.5
    GOOD_ACCURACY_THRESHOLD = 0.7
    HIGH_VARIANCE_THRESHOLD = 0.1


# Circuit breaker settings
class CircuitBreakerSettings:
    """Circuit breaker configuration constants."""
    
    DEFAULT_FAILURE_THRESHOLD = 3
    DEFAULT_TIMEOUT = 300  # seconds
    HALF_OPEN_RETRY_TIMEOUT = 60  # seconds


# Model factory settings
class ModelFactorySettings:
    """Model factory configuration constants."""
    
    # Default model parameters
    DEFAULT_N_ESTIMATORS = 100
    DEFAULT_LEARNING_RATE = 0.1
    DEFAULT_MAX_DEPTH = 6
    DEFAULT_RANDOM_STATE = 42
    DEFAULT_MAX_ITER = 1000
    
    # Model-specific settings
    LIGHTGBM_VERBOSITY = -1
    XGBOOST_VERBOSITY = 0
    ELASTIC_NET_L1_RATIO = 0.5


# Temporal consistency settings
class TemporalConsistencySettings:
    """Temporal consistency validation constants."""
    
    MIN_STABILITY_SCORE = 0.7
    MAX_RAPID_TRANSITION_RATIO = 0.1
    MIN_REGIME_DURATION = 3  # samples
    
    # Regime balance thresholds
    MIN_REGIME_BALANCE = 0.1
    GOOD_REGIME_BALANCE = 0.3


# Reporting settings
class ReportingSettings:
    """Reporting configuration constants."""
    
    MAX_TOP_FEATURES = 10
    PREVIEW_SAMPLE_COUNT = 100
    
    # Quality score weights
    PERFORMANCE_WEIGHT = 0.4
    RELIABILITY_WEIGHT = 0.3
    FEATURE_WEIGHT = 0.2
    REGIME_WEIGHT = 0.1


# File and path constants
class PathConstants:
    """File and path constants."""
    
    DEFAULT_OUTPUT_DIR = "artifacts"
    MODELS_SUBDIR = "models"
    REPORTS_SUBDIR = "reports"
    
    # File extensions
    MODEL_EXTENSION = ".pkl"
    METADATA_EXTENSION = ".json"
    REPORT_EXTENSION = ".json"


# Logging constants
class LoggingConstants:
    """Logging configuration constants."""
    
    # Log levels
    DEBUG_LEVEL = "DEBUG"
    INFO_LEVEL = "INFO"
    WARNING_LEVEL = "WARNING"
    ERROR_LEVEL = "ERROR"
    CRITICAL_LEVEL = "CRITICAL"
    
    # Common log messages
    INITIALIZATION_SUCCESS = "✅ Component initialized successfully"
    INITIALIZATION_FAILED = "❌ Component initialization failed"
    VALIDATION_PASSED = "✅ Validation passed"
    VALIDATION_FAILED = "❌ Validation failed"
    TRAINING_STARTED = "🚀 Training started"
    TRAINING_COMPLETED = "✅ Training completed successfully"
    TRAINING_FAILED = "❌ Training failed"
    
    # Progress indicators
    PROGRESS_INDICATOR = "🔄"
    SUCCESS_INDICATOR = "✅"
    WARNING_INDICATOR = "⚠️"
    ERROR_INDICATOR = "❌"
    INFO_INDICATOR = "ℹ️"
    CRITICAL_INDICATOR = "🚨"


# Feature processing constants
class FeatureProcessingConstants:
    """Feature processing configuration constants."""
    
    # Data quality thresholds
    HIGH_NULL_THRESHOLD = 0.1
    ACCEPTABLE_NULL_THRESHOLD = 0.05
    
    # Feature selection ratios
    HIGH_SELECTION_RATIO = 0.5
    OPTIMAL_SELECTION_RATIO = 0.3
    
    # Correlation thresholds
    HIGH_CORRELATION_THRESHOLD = 0.95
    MODERATE_CORRELATION_THRESHOLD = 0.8


# Hardware optimization constants
class HardwareConstants:
    """Hardware optimization constants."""
    
    # GPU thresholds
    GPU_MIN_SAMPLES = 10000
    GPU_MEMORY_LIMIT = 8  # GB
    
    # CPU optimization
    MAX_CPU_CORES = 16
    OPTIMAL_THREAD_COUNT = 8
    
    # Memory optimization
    MEMORY_CLEANUP_THRESHOLD = 1000  # MB
    GARBAGE_COLLECTION_INTERVAL = 100  # operations