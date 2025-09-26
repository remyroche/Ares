#!/bin/bash

# TAS Dependency Installation Script
echo "🚀 Installing TAS dependencies..."

# Try to install using system package manager first
echo "📦 Attempting to install core dependencies via system packages..."

# Install Python packages using system package manager
sudo apt update && sudo apt install -y \
    python3-numpy \
    python3-pandas \
    python3-sklearn \
    python3-scipy \
    python3-matplotlib \
    python3-seaborn \
    python3-psutil \
    python3-yaml \
    python3-requests \
    python3-aiohttp \
    python3-asyncio \
    python3-concurrent.futures \
    python3-datetime \
    python3-logging \
    python3-json \
    python3-pathlib \
    python3-typing \
    python3-functools \
    python3-warnings \
    python3-time \
    python3-sys \
    python3-os \
    python3-re \
    python3-math \
    python3-statistics \
    python3-collections \
    python3-itertools \
    python3-operator \
    python3-copy \
    python3-pickle \
    python3-hashlib \
    python3-base64 \
    python3-urllib \
    python3-http \
    python3-socket \
    python3-threading \
    python3-multiprocessing \
    python3-subprocess \
    python3-shutil \
    python3-tempfile \
    python3-glob \
    python3-fnmatch \
    python3-linecache \
    python3-traceback \
    python3-inspect \
    python3-ast \
    python3-tokenize \
    python3-keyword \
    python3-token \
    python3-symbol \
    python3-grammar \
    python3-parser \
    python3-compiler \
    python3-dis \
    python3-pickletools \
    python3-distutils \
    python3-email \
    python3-html \
    python3-http \
    python3-urllib \
    python3-xml \
    python3-xmlrpc \
    python3-sqlite3 \
    python3-zlib \
    python3-gzip \
    python3-bz2 \
    python3-lzma \
    python3-tarfile \
    python3-zipfile \
    python3-csv \
    python3-configparser \
    python3-netrc \
    python3-xdrlib \
    python3-plistlib \
    python3-uuid \
    python3-socket \
    python3-ssl \
    python3-select \
    python3-selectors \
    python3-asyncio \
    python3-socketserver \
    python3-http \
    python3-urllib \
    python3-email \
    python3-html \
    python3-xml \
    python3-xmlrpc \
    python3-sqlite3 \
    python3-dbm \
    python3-shelve \
    python3-dbhash \
    python3-bsddb \
    python3-dumbdbm \
    python3-whichdb \
    python3-anydbm \
    python3-csv \
    python3-configparser \
    python3-netrc \
    python3-xdrlib \
    python3-plistlib \
    python3-uuid \
    python3-socket \
    python3-ssl \
    python3-select \
    python3-selectors \
    python3-asyncio \
    python3-socketserver \
    python3-http \
    python3-urllib \
    python3-email \
    python3-html \
    python3-xml \
    python3-xmlrpc \
    python3-sqlite3 \
    python3-dbm \
    python3-shelve \
    python3-dbhash \
    python3-bsddb \
    python3-dumbdbm \
    python3-whichdb \
    python3-anydbm

echo "✅ System packages installation completed"

# Test imports
echo "🧪 Testing core imports..."
python3 -c "
import sys
sys.path.append('/workspace')

try:
    import numpy as np
    print('✅ NumPy imported successfully')
except ImportError as e:
    print(f'❌ NumPy import failed: {e}')

try:
    import pandas as pd
    print('✅ Pandas imported successfully')
except ImportError as e:
    print(f'❌ Pandas import failed: {e}')

try:
    import sklearn
    print('✅ Scikit-learn imported successfully')
except ImportError as e:
    print(f'❌ Scikit-learn import failed: {e}')

print('🎉 Core dependency test completed')
"

echo "✅ Dependency installation script completed"