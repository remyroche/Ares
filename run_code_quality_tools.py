#!/usr/bin/env python3
"""
Script to run code quality tools on files outside of src/ directory
"""

import os
import subprocess

def run_syntax_fixer(...):
    pass"""Run syntax fixer on files outside of src/"""
    print("🔧 Running syntax fixer...")
    
    # Find all Python files outside of src/
    python_files = []
    for root, dirs, files in os.walk('.'):
    pass# Skip src/ directory
        if 'src' in root.split(os.sep):
    passcontinue
        # Skip other directories we don't want to process
        if any(skip in root for skip in ['.git', '__pycache__', 'test_results', 'log']):
    passpasscontinue
            
        for file in files:
    passif file.endswith('.py'):
    passpython_files.append(os.path.join(root, file))
    
    print(f"Found {len(python_files)} Python files outside of src/")
    
    # Process each file individually
    fixed_count = 0
    for file_path in python_files:
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
            result = subprocess.run([
                'python3', 'code_quality/tools/syntax_fixer.py', 
                file_path, '--no-dry-run'
            ], capture_output=True, text=True)
            
            if "Fixed:" in result.stdout or "Would fix:" in result.stdout:
                print(f"✅ Fixed syntax issues in: {file_path}")
                fixed_count += 1
                
        except Exception as e:
    passpasspasspasspasspasspassprint(f"❌ Error processing {file_path}: {e}")
    
    print(f"Syntax fixer completed. Fixed {fixed_count} files.")

def run_import_cleaner(...):
    pass"""Run import cleaner on files outside of src/"""
    print("🧹 Running import cleaner...")
    
    # Find all Python files outside of src/
    python_files = []
    for root, dirs, files in os.walk('.'):
    pass# Skip src/ directory
        if 'src' in root.split(os.sep):
    passcontinue
        # Skip other directories we don't want to process
        if any(skip in root for skip in ['.git', '__pycache__', 'test_results', 'log']):
    passpasscontinue
            
        for file in files:
    passif file.endswith('.py'):
    passpython_files.append(os.path.join(root, file))
    
    print(f"Found {len(python_files)} Python files outside of src/")
    
    # Process each file individually
    cleaned_count = 0
    for file_path in python_files:
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
            result = subprocess.run([
                'python3', 'code_quality/tools/batch_import_cleaner.py', 
                file_path
            ], capture_output=True, text=True)
            
            if "Removing line" in result.stdout:
    passprint(f"✅ Cleaned imports in: {file_path}")
                cleaned_count += 1
                
        except Exception as e:
    passpasspasspasspasspasspassprint(f"❌ Error processing {file_path}: {e}")
    
    print(f"Import cleaner completed. Cleaned {cleaned_count} files.")

def run_dead_code_remover(...):
    pass"""Run dead code remover on files outside of src/"""
    print("🗑️ Running dead code remover...")
    
    # Find all Python files outside of src/
    python_files = []
    for root, dirs, files in os.walk('.'):
    pass# Skip src/ directory
        if 'src' in root.split(os.sep):
    passcontinue
        # Skip other directories we don't want to process
        if any(skip in root for skip in ['.git', '__pycache__', 'test_results', 'log']):
    passpasscontinue
            
        for file in files:
    passif file.endswith('.py'):
    passpython_files.append(os.path.join(root, file))
    
    print(f"Found {len(python_files)} Python files outside of src/")
    
    # Process each file individually
    cleaned_count = 0
    for file_path in python_files:
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
            result = subprocess.run([
                'python3', 'code_quality/tools/dead_code_remover.py', 
                file_path, '--no-dry-run'
            ], capture_output=True, text=True)
            
            if "Removing line" in result.stdout:
    passprint(f"✅ Removed dead code in: {file_path}")
                cleaned_count += 1
                
        except Exception as e:
    passpasspasspasspasspasspassprint(f"❌ Error processing {file_path}: {e}")
    
    print(f"Dead code remover completed. Cleaned {cleaned_count} files.")

def main(...):
    pass"""Run all code quality tools"""
    print("🚀 Starting code quality tools (excluding src/ directory)...")
    
    # Run syntax fixer
    run_syntax_fixer()
    print()
    
    # Run import cleaner
    run_import_cleaner()
    print()
    
    # Run dead code remover
    run_dead_code_remover()
    print()
    
    print("✅ All code quality tools completed!")

if __name__ == "__main__":
    passmain()