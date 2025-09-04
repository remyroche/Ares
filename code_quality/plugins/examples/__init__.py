"""
Example Plugins

This module contains example plugins demonstrating how to use the plugin system.
"""

from .syntax_plugin import SyntaxFixerPlugin
from .import_plugin import ImportFixerPlugin
from .linter_plugin import LinterPlugin
from .security_plugin import SecurityScannerPlugin

__all__ = [
    'SyntaxFixerPlugin',
    'ImportFixerPlugin', 
    'LinterPlugin',
    'SecurityScannerPlugin'
]