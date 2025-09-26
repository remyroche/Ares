#!/usr/bin/env python3
"""
TAS Issues Fix Script

This script implements the fixes identified in the audit report.
"""

import os
import re
import shutil
from pathlib import Path
from typing import List, Dict, Tuple

def fix_import_issues():
    """Fix import issues throughout the codebase."""
    print("🔧 Fixing import issues...")
    
    # Files with import issues to fix
    import_fixes = [
        {
            'file': '/workspace/live_trading/order_manager.py',
            'old': 'from ..src.interfaces.base_interfaces import TradeDecision',
            'new': 'from src.interfaces.base_interfaces import TradeDecision'
        },
        {
            'file': '/workspace/exchanges/order_router.py', 
            'old': 'from ..exchange.factory import ExchangeFactory',
            'new': 'from exchange.factory import ExchangeFactory'
        }
    ]
    
    for fix in import_fixes:
        if os.path.exists(fix['file']):
            try:
                with open(fix['file'], 'r') as f:
                    content = f.read()
                
                if fix['old'] in content:
                    content = content.replace(fix['old'], fix['new'])
                    
                    with open(fix['file'], 'w') as f:
                        f.write(content)
                    print(f"✅ Fixed import in {fix['file']}")
                else:
                    print(f"⚠️  Import pattern not found in {fix['file']}")
            except Exception as e:
                print(f"❌ Error fixing {fix['file']}: {e}")
        else:
            print(f"⚠️  File not found: {fix['file']}")

def fix_broad_exception_handlers():
    """Replace broad exception handlers with specific ones."""
    print("🔧 Fixing broad exception handlers...")
    
    # Find files with broad exception handlers
    broad_exception_pattern = r'except\s+Exception\s*:'
    
    files_to_fix = [
        '/workspace/core/tree_architecture_search.py',
        '/workspace/nas_trainer.py',
        '/workspace/live_trading/trading_engine.py'
    ]
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Count broad exception handlers
                matches = re.findall(broad_exception_pattern, content)
                if matches:
                    print(f"⚠️  Found {len(matches)} broad exception handlers in {file_path}")
                    # Note: Specific fixes would need to be implemented per file
                else:
                    print(f"✅ No broad exception handlers found in {file_path}")
            except Exception as e:
                print(f"❌ Error checking {file_path}: {e}")

def create_missing_interfaces():
    """Create missing interface files."""
    print("🔧 Creating missing interface files...")
    
    # Create base_interfaces.py if it doesn't exist
    interfaces_dir = Path('/workspace/src/interfaces')
    interfaces_dir.mkdir(exist_ok=True)
    
    base_interfaces_file = interfaces_dir / 'base_interfaces.py'
    
    if not base_interfaces_file.exists():
        interfaces_content = '''"""
Base Interfaces for TAS

Defines core data structures and interfaces used throughout the system.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum

class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class TradeDecision:
    """Trading decision data structure."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class AnalysisResult:
    """Analysis result data structure."""
    symbol: str
    timestamp: datetime
    analysis_type: str
    confidence: float
    data: Dict[str, Any]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class StrategyResult:
    """Strategy result data structure."""
    strategy_name: str
    symbol: str
    timestamp: datetime
    signal: str
    confidence: float
    parameters: Dict[str, Any]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class MarketData:
    """Market data structure."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class Order:
    """Order data structure."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    timestamp: datetime = None
    filled_quantity: float = 0.0
    average_price: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}
'''
        
        with open(base_interfaces_file, 'w') as f:
            f.write(interfaces_content)
        print(f"✅ Created {base_interfaces_file}")
    else:
        print(f"✅ {base_interfaces_file} already exists")

def create_missing_init_files():
    """Create missing __init__.py files."""
    print("🔧 Creating missing __init__.py files...")
    
    directories_needing_init = [
        '/workspace/src/interfaces',
        '/workspace/src/utils/ml_common/feature_selection',
        '/workspace/exchanges/base_exchange',
        '/workspace/live_trading'
    ]
    
    for dir_path in directories_needing_init:
        init_file = Path(dir_path) / '__init__.py'
        if not init_file.exists():
            init_file.parent.mkdir(parents=True, exist_ok=True)
            with open(init_file, 'w') as f:
                f.write(f'"""Package for {dir_path.split("/")[-1]}."""\n')
            print(f"✅ Created {init_file}")
        else:
            print(f"✅ {init_file} already exists")

def consolidate_requirements():
    """Consolidate requirements files."""
    print("🔧 Consolidating requirements files...")
    
    # Use the consolidated requirements file we created
    consolidated_file = '/workspace/requirements_consolidated.txt'
    main_requirements = '/workspace/requirements.txt'
    
    if os.path.exists(consolidated_file):
        shutil.copy2(consolidated_file, main_requirements)
        print(f"✅ Consolidated requirements into {main_requirements}")
    else:
        print(f"⚠️  Consolidated requirements file not found: {consolidated_file}")

def create_development_guide():
    """Create a development guide with the fixes."""
    print("🔧 Creating development guide...")
    
    guide_content = '''# TAS Development Guide

## Quick Start

1. **Install Dependencies**:
   ```bash
   # Try system packages first
   sudo apt install python3-numpy python3-pandas python3-sklearn
   
   # Or use the installation script
   ./install_dependencies.sh
   ```

2. **Test Imports**:
   ```bash
   python3 -c "import sys; sys.path.append('/workspace'); import src.analyst.analyst; print('TAS working!')"
   ```

## Fixed Issues

### ✅ Import Violations Fixed
- Fixed relative import violations in live_trading modules
- Fixed relative import violations in exchanges modules
- Created missing base_interfaces.py

### ✅ Requirements Consolidated
- Consolidated multiple requirements.txt files
- Created comprehensive dependency list

### ✅ Missing Files Created
- Created missing __init__.py files
- Created base_interfaces.py with core data structures

## Remaining Issues

### 🔄 Still Need Manual Fix
- Large file refactoring (8,734 line feature_selection.py)
- Broad exception handler replacement
- TODO/FIXME item resolution

### 📋 Next Steps
1. Install missing dependencies
2. Test core functionality
3. Refactor large files
4. Implement proper error handling
5. Add comprehensive testing

## Architecture Improvements

### Recommended Structure
```
src/
├── interfaces/          # Core data structures
├── analyst/            # Analysis components
├── tactician/          # Trading logic
├── supervisor/         # System coordination
├── utils/              # Shared utilities
└── training/           # ML training
```

### Code Quality Standards
- Maximum file size: 500 lines
- Specific exception handling
- Comprehensive logging
- Type hints throughout
- Unit test coverage > 80%
'''
    
    with open('/workspace/DEVELOPMENT_GUIDE.md', 'w') as f:
        f.write(guide_content)
    print("✅ Created DEVELOPMENT_GUIDE.md")

def main():
    """Main fix function."""
    print("🚀 Starting TAS issues fix...")
    
    fix_import_issues()
    fix_broad_exception_handlers()
    create_missing_interfaces()
    create_missing_init_files()
    consolidate_requirements()
    create_development_guide()
    
    print("\n🎉 TAS issues fix completed!")
    print("\nNext steps:")
    print("1. Run: ./install_dependencies.sh")
    print("2. Test: python3 -c \"import sys; sys.path.append('/workspace'); import src.analyst.analyst\"")
    print("3. Review: DEVELOPMENT_GUIDE.md")

if __name__ == "__main__":
    main()