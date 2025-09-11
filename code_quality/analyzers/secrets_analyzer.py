#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Secrets and API Keys Analyzer

This analyzer specifically looks for hardcoded secrets, API keys, passwords, and other sensitive information
that should not be committed to version control. It ignores general security issues like hasattr() usage.

Focus areas:
1. Hardcoded API keys
2. Database passwords
3. Secret tokens
4. Private keys
5. Authentication credentials
6. Sensitive configuration values
"""

import ast
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class SecretType(Enum):
    """Types of secrets found."""
    API_KEY = "api_key"
    PASSWORD = "password"
    SECRET_TOKEN = "secret_token"
    PRIVATE_KEY = "private_key"
    DATABASE_URL = "database_url"
    AUTH_TOKEN = "auth_token"
    ACCESS_KEY = "access_key"
    SECRET_KEY = "secret_key"
    CREDENTIALS = "credentials"
    HARDCODED_SECRET = "hardcoded_secret"


class SecretSeverity(Enum):
    """Severity levels for secrets."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecretIssue:
    """Represents a found secret."""
    type: SecretType
    severity: SecretSeverity
    name: str
    line: int
    column: int = 0
    context: str = ""
    file_path: str = ""
    description: str = ""
    suggestions: List[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class SecretsAnalysisResult:
    """Results from secrets analysis."""
    file_path: str
    secrets: List[SecretIssue] = field(default_factory=list)
    execution_time: float = 0.0
    error: Optional[str] = None
    
    @property
    def total_secrets(self) -> int:
        return len(self.secrets)
    
    @property
    def critical_secrets(self) -> int:
        return len([s for s in self.secrets if s.severity == SecretSeverity.CRITICAL])
    
    @property
    def high_secrets(self) -> int:
        return len([s for s in self.secrets if s.severity == SecretSeverity.HIGH])


class SecretsAnalyzer:
    """Analyzer for hardcoded secrets and API keys."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the secrets analyzer."""
        self.config = config or {}
        
        # Patterns for different types of secrets
        self.secret_patterns = {
            SecretType.API_KEY: [
                r'api[_-]?key\s*[:=]\s*["\']([^"\']+)["\']',
                r'apikey\s*[:=]\s*["\']([^"\']+)["\']',
                r'api_key\s*[:=]\s*["\']([^"\']+)["\']',
                r'API_KEY\s*[:=]\s*["\']([^"\']+)["\']',
            ],
            SecretType.PASSWORD: [
                r'password\s*[:=]\s*["\']([^"\']+)["\']',
                r'passwd\s*[:=]\s*["\']([^"\']+)["\']',
                r'pwd\s*[:=]\s*["\']([^"\']+)["\']',
                r'PASSWORD\s*[:=]\s*["\']([^"\']+)["\']',
            ],
            SecretType.SECRET_TOKEN: [
                r'secret[_-]?token\s*[:=]\s*["\']([^"\']+)["\']',
                r'secrettoken\s*[:=]\s*["\']([^"\']+)["\']',
                r'SECRET_TOKEN\s*[:=]\s*["\']([^"\']+)["\']',
                r'token\s*[:=]\s*["\']([^"\']+)["\']',
            ],
            SecretType.PRIVATE_KEY: [
                r'private[_-]?key\s*[:=]\s*["\']([^"\']+)["\']',
                r'privatekey\s*[:=]\s*["\']([^"\']+)["\']',
                r'PRIVATE_KEY\s*[:=]\s*["\']([^"\']+)["\']',
                r'-----BEGIN PRIVATE KEY-----',
                r'-----BEGIN RSA PRIVATE KEY-----',
            ],
            SecretType.DATABASE_URL: [
                r'database[_-]?url\s*[:=]\s*["\']([^"\']+)["\']',
                r'db[_-]?url\s*[:=]\s*["\']([^"\']+)["\']',
                r'DATABASE_URL\s*[:=]\s*["\']([^"\']+)["\']',
                r'mysql://[^"\']+',
                r'postgresql://[^"\']+',
                r'mongodb://[^"\']+',
            ],
            SecretType.AUTH_TOKEN: [
                r'auth[_-]?token\s*[:=]\s*["\']([^"\']+)["\']',
                r'authtoken\s*[:=]\s*["\']([^"\']+)["\']',
                r'AUTH_TOKEN\s*[:=]\s*["\']([^"\']+)["\']',
                r'bearer[_-]?token\s*[:=]\s*["\']([^"\']+)["\']',
            ],
            SecretType.ACCESS_KEY: [
                r'access[_-]?key\s*[:=]\s*["\']([^"\']+)["\']',
                r'accesskey\s*[:=]\s*["\']([^"\']+)["\']',
                r'ACCESS_KEY\s*[:=]\s*["\']([^"\']+)["\']',
                r'aws[_-]?access[_-]?key\s*[:=]\s*["\']([^"\']+)["\']',
            ],
            SecretType.SECRET_KEY: [
                r'secret[_-]?key\s*[:=]\s*["\']([^"\']+)["\']',
                r'secretkey\s*[:=]\s*["\']([^"\']+)["\']',
                r'SECRET_KEY\s*[:=]\s*["\']([^"\']+)["\']',
                r'flask[_-]?secret[_-]?key\s*[:=]\s*["\']([^"\']+)["\']',
            ],
        }
        
        # Common false positive patterns to ignore
        self.ignore_patterns = [
            r'password\s*[:=]\s*["\']\s*["\']',  # Empty password
            r'api[_-]?key\s*[:=]\s*["\']\s*["\']',  # Empty API key
            r'secret\s*[:=]\s*["\']\s*["\']',  # Empty secret
            r'#.*password',  # Comments
            r'#.*api.*key',  # Comments
            r'#.*secret',  # Comments
            r'example\.com',  # Example URLs
            r'localhost',  # Local development
            r'127\.0\.0\.1',  # Local IP
            r'your[_-]?api[_-]?key',  # Placeholder text
            r'your[_-]?password',  # Placeholder text
            r'your[_-]?secret',  # Placeholder text
        ]
        
        # Minimum length for secrets (to avoid false positives)
        self.min_secret_length = 8
    
    def analyze_file(self, file_path: str) -> SecretsAnalysisResult:
        """Analyze a file for hardcoded secrets."""
        start_time = time.time()
        result = SecretsAnalysisResult(file_path=file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                self._analyze_line(line, line_num, result)
            
        except Exception as e:
            result.error = str(e)
        
        result.execution_time = time.time() - start_time
        return result
    
    def _analyze_line(self, line: str, line_num: int, result: SecretsAnalysisResult) -> None:
        """Analyze a single line for secrets."""
        line_lower = line.lower()
        
        # Skip comments and empty lines
        if line.strip().startswith('#') or not line.strip():
            return
        
        # Check for each type of secret
        for secret_type, patterns in self.secret_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    secret_value = match.group(1) if match.groups() else match.group(0)
                    
                    # Skip if it's an ignore pattern
                    if self._should_ignore(line, secret_value):
                        continue
                    
                    # Skip if secret is too short
                    if len(secret_value) < self.min_secret_length:
                        continue
                    
                    # Determine severity
                    severity = self._determine_severity(secret_type, secret_value)
                    
                    # Create secret issue
                    secret_issue = SecretIssue(
                        type=secret_type,
                        severity=severity,
                        name=secret_value[:20] + "..." if len(secret_value) > 20 else secret_value,
                        line=line_num,
                        column=match.start(),
                        context=line.strip(),
                        file_path=result.file_path,
                        description=f"Hardcoded {secret_type.value.replace('_', ' ')} found",
                        suggestions=self._get_suggestions(secret_type),
                        confidence=self._calculate_confidence(secret_value)
                    )
                    
                    result.secrets.append(secret_issue)
    
    def _should_ignore(self, line: str, secret_value: str) -> bool:
        """Check if this should be ignored as a false positive."""
        for ignore_pattern in self.ignore_patterns:
            if re.search(ignore_pattern, line, re.IGNORECASE):
                return True
        
        # Ignore common placeholder values
        placeholder_values = [
            'your_api_key', 'your_password', 'your_secret', 'your_token',
            'api_key_here', 'password_here', 'secret_here', 'token_here',
            'replace_me', 'change_me', 'update_me', 'fill_me',
            'example', 'test', 'demo', 'sample', 'placeholder'
        ]
        
        if secret_value.lower() in placeholder_values:
            return True
        
        return False
    
    def _determine_severity(self, secret_type: SecretType, secret_value: str) -> SecretSeverity:
        """Determine the severity of a found secret."""
        if secret_type in [SecretType.PRIVATE_KEY, SecretType.DATABASE_URL]:
            return SecretSeverity.CRITICAL
        elif secret_type in [SecretType.API_KEY, SecretType.SECRET_KEY, SecretType.AUTH_TOKEN]:
            return SecretSeverity.HIGH
        elif secret_type in [SecretType.PASSWORD, SecretType.SECRET_TOKEN]:
            return SecretSeverity.MEDIUM
        else:
            return SecretSeverity.LOW
    
    def _calculate_confidence(self, secret_value: str) -> float:
        """Calculate confidence that this is a real secret."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence for longer secrets
        if len(secret_value) > 20:
            confidence += 0.2
        if len(secret_value) > 40:
            confidence += 0.2
        
        # Increase confidence for secrets with special characters
        if re.search(r'[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]', secret_value):
            confidence += 0.1
        
        # Increase confidence for secrets with numbers
        if re.search(r'\d', secret_value):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _get_suggestions(self, secret_type: SecretType) -> List[str]:
        """Get suggestions for handling the found secret."""
        suggestions = [
            "Remove hardcoded secret from source code",
            "Use environment variables instead",
            "Use a secure configuration management system",
            "Consider using a secrets management service"
        ]
        
        if secret_type == SecretType.API_KEY:
            suggestions.extend([
                "Store API key in environment variable: export API_KEY=your_key",
                "Use a .env file (and add it to .gitignore)",
                "Consider using API key rotation"
            ])
        elif secret_type == SecretType.PASSWORD:
            suggestions.extend([
                "Store password in environment variable: export PASSWORD=your_password",
                "Use a secure password manager",
                "Consider using OAuth or other authentication methods"
            ])
        elif secret_type == SecretType.DATABASE_URL:
            suggestions.extend([
                "Store database URL in environment variable: export DATABASE_URL=your_url",
                "Use connection pooling with environment-based configuration",
                "Consider using a database connection service"
            ])
        
        return suggestions


def analyze_secrets(file_path: str, config: Optional[Dict[str, Any]] = None) -> SecretsAnalysisResult:
    """Analyze a file for hardcoded secrets."""
    analyzer = SecretsAnalyzer(config)
    return analyzer.analyze_file(file_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = analyze_secrets(sys.argv[1])
        tprint(f"Found {result.total_secrets} secrets in {sys.argv[1]}")
        tprint(f"Critical: {result.critical_secrets}, High: {result.high_secrets}")
        for secret in result.secrets:
            tprint(f"  {secret.severity.value}: {secret.type.value} (line {secret.line}) - {secret.name}")
    else:
        tprint("Usage: python secrets_analyzer.py <file_path>")