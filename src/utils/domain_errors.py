"""Domain-specific error types for normalization and validation."""

from typing import Any, Dict, Optional


class DomainError(Exception):
    """Base domain error class for all domain-specific exceptions."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.is_initialized = False

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.message}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message='{self.message}', context={self.context})"


class DataValidationError(DomainError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "data_validation"


class SchemaValidationError(DomainError):
    """Raised when schema validation fails."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "schema_validation"


class VectorizationError(DomainError):
    """Raised when vectorization operations fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "vectorization"


class ExternalServiceError(DomainError):
    """Raised when external service calls fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "external_service"


class OperationTimeoutError(DomainError):
    """Raised when operations exceed timeout limits."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "timeout"


class NotFoundError(DomainError):
    """Raised when requested resources are not found."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "not_found"


class AuthenticationError(DomainError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "authentication"


class AuthorizationError(DomainError):
    """Raised when authorization fails."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "authorization"


class ConfigurationError(DomainError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "configuration"


class ResourceExhaustedError(DomainError):
    """Raised when system resources are exhausted."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "resource_exhausted"


class InvalidStateError(DomainError):
    """Raised when an operation is performed in an invalid state."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "invalid_state"


class CircularDependencyError(DomainError):
    """Raised when circular dependencies are detected."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "circular_dependency"


class DataIntegrityError(DomainError):
    """Raised when data integrity constraints are violated."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "data_integrity"


class ValidationError(DomainError):
    """Generic validation error."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "validation"


class ProcessingError(DomainError):
    """Raised when data processing fails."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "processing"


class SerializationError(DomainError):
    """Raised when serialization/deserialization fails."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "serialization"


class NetworkError(DomainError):
    """Raised when network operations fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "network"


class FileSystemError(DomainError):
    """Raised when file system operations fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "file_system"


class DatabaseError(DomainError):
    """Raised when database operations fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "database"


class CacheError(DomainError):
    """Raised when cache operations fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "cache"


class QueueError(DomainError):
    """Raised when queue operations fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "queue"


class LockError(DomainError):
    """Raised when lock operations fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "lock"


class RateLimitError(DomainError):
    """Raised when rate limits are exceeded."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "rate_limit"


class QuotaExceededError(DomainError):
    """Raised when quotas are exceeded."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "quota_exceeded"


class VersionConflictError(DomainError):
    """Raised when version conflicts occur."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "version_conflict"


class DependencyError(DomainError):
    """Raised when dependencies are missing or incompatible."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "dependency"


class FormatError(DomainError):
    """Raised when data format is invalid."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "format"


class EncodingError(DomainError):
    """Raised when encoding/decoding fails."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "encoding"


class CompressionError(DomainError):
    """Raised when compression/decompression fails."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "compression"


class EncryptionError(DomainError):
    """Raised when encryption/decryption fails."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "encryption"


class ChecksumError(DomainError):
    """Raised when checksum validation fails."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "checksum"


class SignatureError(DomainError):
    """Raised when signature validation fails."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "signature"


class CertificateError(DomainError):
    """Raised when certificate validation fails."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "certificate"


class TokenError(DomainError):
    """Raised when token validation fails."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "token"


class SessionError(DomainError):
    """Raised when session operations fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "session"


class ConnectionError(DomainError):
    """Raised when connection operations fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "connection"


class ProtocolError(DomainError):
    """Raised when protocol violations occur."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "protocol"


class MessageError(DomainError):
    """Raised when message processing fails."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "message"


class EventError(DomainError):
    """Raised when event processing fails."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "event"


class WorkflowError(DomainError):
    """Raised when workflow execution fails."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "workflow"


class PipelineError(DomainError):
    """Raised when pipeline execution fails."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "pipeline"


class TaskError(DomainError):
    """Raised when task execution fails."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "task"


class JobError(DomainError):
    """Raised when job execution fails."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "job"


class SchedulerError(DomainError):
    """Raised when scheduler operations fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "scheduler"


class MonitorError(DomainError):
    """Raised when monitoring operations fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "monitor"


class AlertError(DomainError):
    """Raised when alert operations fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "alert"


class MetricError(DomainError):
    """Raised when metric operations fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "metric"


class LogError(DomainError):
    """Raised when logging operations fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "log"


class TraceError(DomainError):
    """Raised when tracing operations fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "trace"


class AuditError(DomainError):
    """Raised when audit operations fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "audit"


class ComplianceError(DomainError):
    """Raised when compliance checks fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "compliance"


class SecurityError(DomainError):
    """Raised when security checks fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "security"


class PrivacyError(DomainError):
    """Raised when privacy checks fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "privacy"


class GovernanceError(DomainError):
    """Raised when governance checks fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "governance"


class PolicyError(DomainError):
    """Raised when policy checks fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "policy"


class RuleError(DomainError):
    """Raised when rule checks fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "rule"


class ConstraintError(DomainError):
    """Raised when constraint checks fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "constraint"


class BusinessRuleError(DomainError):
    """Raised when business rule checks fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "business_rule"


class DomainRuleError(DomainError):
    """Raised when domain rule checks fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "domain_rule"


class ApplicationRuleError(DomainError):
    """Raised when application rule checks fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "application_rule"


class SystemRuleError(DomainError):
    """Raised when system rule checks fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "system_rule"


class UserRuleError(DomainError):
    """Raised when user rule checks fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "user_rule"


class DataRuleError(DomainError):
    """Raised when data rule checks fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "data_rule"


class ProcessRuleError(DomainError):
    """Raised when process rule checks fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "process_rule"


class WorkflowRuleError(DomainError):
    """Raised when workflow rule checks fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "workflow_rule"


class PipelineRuleError(DomainError):
    """Raised when pipeline rule checks fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "pipeline_rule"


class TaskRuleError(DomainError):
    """Raised when task rule checks fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "task_rule"


class JobRuleError(DomainError):
    """Raised when job rule checks fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "job_rule"


class SchedulerRuleError(DomainError):
    """Raised when scheduler rule checks fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "scheduler_rule"


class MonitorRuleError(DomainError):
    """Raised when monitor rule checks fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "monitor_rule"


class AlertRuleError(DomainError):
    """Raised when alert rule checks fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "alert_rule"


class MetricRuleError(DomainError):
    """Raised when metric rule checks fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "metric_rule"


class LogRuleError(DomainError):
    """Raised when log rule checks fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "log_rule"


class TraceRuleError(DomainError):
    """Raised when trace rule checks fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "trace_rule"


class AuditRuleError(DomainError):
    """Raised when audit rule checks fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "audit_rule"


class ComplianceRuleError(DomainError):
    """Raised when compliance rule checks fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "compliance_rule"


class SecurityRuleError(DomainError):
    """Raised when security rule checks fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "security_rule"


class PrivacyRuleError(DomainError):
    """Raised when privacy rule checks fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "privacy_rule"


class GovernanceRuleError(DomainError):
    """Raised when governance rule checks fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "governance_rule"


class PolicyRuleError(DomainError):
    """Raised when policy rule checks fail."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "policy_rule"


# Export all error types
__all__ = [
    "DomainError",
    "DataValidationError",
    "SchemaValidationError",
    "VectorizationError",
    "ExternalServiceError",
    "OperationTimeoutError",
    "NotFoundError",
    "AuthenticationError",
    "AuthorizationError",
    "ConfigurationError",
    "ResourceExhaustedError",
    "InvalidStateError",
    "CircularDependencyError",
    "DataIntegrityError",
    "ValidationError",
    "ProcessingError",
    "SerializationError",
    "NetworkError",
    "FileSystemError",
    "DatabaseError",
    "CacheError",
    "QueueError",
    "LockError",
    "RateLimitError",
    "QuotaExceededError",
    "VersionConflictError",
    "DependencyError",
    "FormatError",
    "EncodingError",
    "CompressionError",
    "EncryptionError",
    "ChecksumError",
    "SignatureError",
    "CertificateError",
    "TokenError",
    "SessionError",
    "ConnectionError",
    "ProtocolError",
    "MessageError",
    "EventError",
    "WorkflowError",
    "PipelineError",
    "TaskError",
    "JobError",
    "SchedulerError",
    "MonitorError",
    "AlertError",
    "MetricError",
    "LogError",
    "TraceError",
    "AuditError",
    "ComplianceError",
    "SecurityError",
    "PrivacyError",
    "GovernanceError",
    "PolicyError",
    "RuleError",
    "ConstraintError",
    "BusinessRuleError",
    "DomainRuleError",
    "ApplicationRuleError",
    "SystemRuleError",
    "UserRuleError",
    "DataRuleError",
    "ProcessRuleError",
    "WorkflowRuleError",
    "PipelineRuleError",
    "TaskRuleError",
    "JobRuleError",
    "SchedulerRuleError",
    "MonitorRuleError",
    "AlertRuleError",
    "MetricRuleError",
    "LogRuleError",
    "TraceRuleError",
    "AuditRuleError",
    "ComplianceRuleError",
    "SecurityRuleError",
    "PrivacyRuleError",
    "GovernanceRuleError",
    "PolicyRuleError",
]
