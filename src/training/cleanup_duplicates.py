"""Script to identify and clean up duplicate and backup files in the training module.

This script helps identify files that should be removed or consolidated.
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple
import hashlib
import json


def calculate_file_hash(filepath: Path) -> str:
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def find_duplicate_files(directory: Path) -> Dict[str, List[Path]]:
    """Find duplicate files based on content hash."""
    file_hashes = {}
    
    for filepath in directory.rglob("*.py"):
        if filepath.is_file():
            file_hash = calculate_file_hash(filepath)
            if file_hash not in file_hashes:
                file_hashes[file_hash] = []
            file_hashes[file_hash].append(filepath)
    
    # Filter to only include hashes with duplicates
    duplicates = {k: v for k, v in file_hashes.items() if len(v) > 1}
    return duplicates


def identify_backup_files(directory: Path) -> List[Path]:
    """Identify files that appear to be backups."""
    backup_patterns = [
        "*backup*", "*_backup*", "*_old*", "*_deprecated*",
        "*_temp*", "*_tmp*", "*_copy*", "*_original*"
    ]
    
    backup_files = []
    for pattern in backup_patterns:
        backup_files.extend(directory.rglob(pattern))
    
    return list(set(backup_files))


def identify_enhanced_versions(directory: Path) -> Dict[str, List[Path]]:
    """Identify files with multiple enhanced/optimized versions."""
    enhanced_groups = {}
    
    for filepath in directory.rglob("*.py"):
        if filepath.is_file():
            filename = filepath.stem
            
            # Check for enhanced/optimized versions
            base_name = filename
            for suffix in ["_enhanced", "_optimized", "_improved", "_v2", "_new"]:
                if filename.endswith(suffix):
                    base_name = filename[:-len(suffix)]
                    break
            
            if base_name not in enhanced_groups:
                enhanced_groups[base_name] = []
            enhanced_groups[base_name].append(filepath)
    
    # Filter to only include groups with multiple versions
    multiple_versions = {k: v for k, v in enhanced_groups.items() if len(v) > 1}
    return multiple_versions


def analyze_file_sizes(directory: Path) -> List[Tuple[Path, int]]:
    """Identify exceptionally large files."""
    large_files = []
    
    for filepath in directory.rglob("*.py"):
        if filepath.is_file():
            size = filepath.stat().st_size
            lines = len(filepath.read_text().splitlines())
            if lines > 1000:  # Files over 1000 lines
                large_files.append((filepath, lines))
    
    return sorted(large_files, key=lambda x: x[1], reverse=True)


def generate_cleanup_report(training_dir: Path) -> Dict[str, any]:
    """Generate a comprehensive cleanup report."""
    # Convert duplicate files paths to strings
    duplicate_files_str = {}
    for hash_val, files in find_duplicate_files(training_dir).items():
        duplicate_files_str[hash_val] = [str(f) for f in files]
    
    report = {
        "duplicate_files": duplicate_files_str,
        "backup_files": [str(f) for f in identify_backup_files(training_dir)],
        "enhanced_versions": {k: [str(f) for f in v] for k, v in identify_enhanced_versions(training_dir).items()},
        "large_files": [(str(f), lines) for f, lines in analyze_file_sizes(training_dir)]
    }
    
    return report


def main():
    """Main function to run the cleanup analysis."""
    training_dir = Path("src/training")
    
    print("🔍 Analyzing training directory for cleanup opportunities...\n")
    
    report = generate_cleanup_report(training_dir)
    
    # Print duplicate files
    if report["duplicate_files"]:
        print("📁 DUPLICATE FILES (same content):")
        print("-" * 60)
        for hash_val, files in report["duplicate_files"].items():
            print(f"\nDuplicate group (hash: {hash_val[:8]}...):")
            for f in files:
                print(f"  - {f}")
    
    # Print backup files
    if report["backup_files"]:
        print("\n💾 BACKUP FILES:")
        print("-" * 60)
        for f in report["backup_files"]:
            print(f"  - {f}")
    
    # Print enhanced versions
    if report["enhanced_versions"]:
        print("\n🔄 MULTIPLE VERSIONS:")
        print("-" * 60)
        for base_name, files in report["enhanced_versions"].items():
            if len(files) > 1:
                print(f"\n{base_name}:")
                for f in files:
                    print(f"  - {f}")
    
    # Print large files
    if report["large_files"]:
        print("\n📏 LARGE FILES (>1000 lines):")
        print("-" * 60)
        for f, lines in report["large_files"][:10]:  # Top 10
            print(f"  - {f}: {lines} lines")
    
    # Save detailed report
    report_path = Path("src/training/cleanup_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 Detailed report saved to: {report_path}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("-" * 60)
    print("1. Remove backup files identified above")
    print("2. Consolidate enhanced versions into single implementations")
    print("3. Break down large files into smaller, focused modules")
    print("4. Use version control instead of keeping backup copies")
    print("5. Follow the new module structure in MODULE_STRUCTURE.md")


if __name__ == "__main__":
    main()