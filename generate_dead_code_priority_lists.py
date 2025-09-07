#!/usr/bin/env python3
"""
Generate prioritized dead code cleanup lists from interaction mapping results.
Automatically creates lists for different thresholds: 20+, 15+, 10+, 5+, 1+ dead items.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def load_dead_code_data(report_file: Path) -> dict:
    """Load dead code cleanup recommendations from JSON file."""
    try:
        with open(report_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading {report_file}: {e}")
        return {}


def categorize_files_by_dead_code_count(dead_code_data: dict) -> dict:
    """Categorize files by number of dead code items."""
    files_with_dead_code = dead_code_data.get('files_with_dead_code', {})
    
    categories = {
        '20_plus': [],    # 20+ dead items
        '15_plus': [],    # 15+ dead items  
        '10_plus': [],    # 10+ dead items
        '5_plus': [],     # 5+ dead items
        '1_plus': []      # 1+ dead items
    }
    
    for file_path, dead_items in files_with_dead_code.items():
        count = len(dead_items)
        
        if count >= 20:
            categories['20_plus'].append((file_path, count, dead_items))
        elif count >= 15:
            categories['15_plus'].append((file_path, count, dead_items))
        elif count >= 10:
            categories['10_plus'].append((file_path, count, dead_items))
        elif count >= 5:
            categories['5_plus'].append((file_path, count, dead_items))
        elif count >= 1:
            categories['1_plus'].append((file_path, count, dead_items))
    
    # Sort each category by count (descending)
    for category in categories:
        categories[category].sort(key=lambda x: x[1], reverse=True)
    
    return categories


def generate_priority_report(categories: dict, output_dir: Path):
    """Generate comprehensive priority report with different thresholds."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate individual files for each threshold
    thresholds = [
        (20, '20_plus', '🔴 CRITICAL'),
        (15, '15_plus', '🟠 HIGH'),
        (10, '10_plus', '🟡 MEDIUM'),
        (5, '5_plus', '🟢 LOW'),
        (1, '1_plus', '🔵 MINIMAL')
    ]
    
    for threshold, category_key, priority_label in thresholds:
        files = categories[category_key]
        
        if not files:
            continue
            
        # Generate individual threshold file
        threshold_file = output_dir / f"dead_code_priority_{threshold}_plus_{timestamp}.txt"
        with open(threshold_file, 'w') as f:
            f.write(f"DEAD CODE CLEANUP PRIORITY LIST - {priority_label} ({threshold}+ dead items)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total files: {len(files)}\n")
            f.write(f"Total dead items: {sum(count for _, count, _ in files)}\n\n")
            
            for i, (file_path, count, dead_items) in enumerate(files, 1):
                f.write(f"{i:3d}. {file_path}\n")
                f.write(f"     Dead items: {count}\n")
                
                # Show first 5 dead items
                f.write("     Items:\n")
                for item in dead_items[:5]:
                    f.write(f"       - {item}\n")
                
                if len(dead_items) > 5:
                    f.write(f"       ... and {len(dead_items) - 5} more\n")
                
                f.write("\n")
        
        print(f"✅ Generated {priority_label} priority list: {threshold_file}")
        print(f"   📁 {len(files)} files with {threshold}+ dead items")


def generate_summary_report(categories: dict, output_dir: Path):
    """Generate a comprehensive summary report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = output_dir / f"dead_code_cleanup_summary_{timestamp}.md"
    
    with open(summary_file, 'w') as f:
        f.write("# Dead Code Cleanup Priority Summary\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall statistics
        total_files = sum(len(files) for files in categories.values())
        total_dead_items = sum(
            sum(count for _, count, _ in files) 
            for files in categories.values()
        )
        
        f.write("## Overall Statistics\n\n")
        f.write(f"- **Total files with dead code**: {total_files}\n")
        f.write(f"- **Total dead items**: {total_dead_items}\n\n")
        
        # Priority breakdown
        f.write("## Priority Breakdown\n\n")
        
        thresholds = [
            (20, '20_plus', '🔴 CRITICAL', 'Immediate action required'),
            (15, '15_plus', '🟠 HIGH', 'High priority cleanup'),
            (10, '10_plus', '🟡 MEDIUM', 'Medium priority cleanup'),
            (5, '5_plus', '🟢 LOW', 'Low priority cleanup'),
            (1, '1_plus', '🔵 MINIMAL', 'Minimal priority cleanup')
        ]
        
        for threshold, category_key, priority_label, description in thresholds:
            files = categories[category_key]
            if files:
                total_items = sum(count for _, count, _ in files)
                f.write(f"### {priority_label} Priority ({threshold}+ dead items)\n\n")
                f.write(f"**{description}**\n\n")
                f.write(f"- Files: {len(files)}\n")
                f.write(f"- Total dead items: {total_items}\n")
                f.write(f"- Average per file: {total_items/len(files):.1f}\n\n")
                
                # Top 10 files
                f.write("**Top 10 files:**\n\n")
                for i, (file_path, count, _) in enumerate(files[:10], 1):
                    f.write(f"{i:2d}. `{file_path}` - {count} dead items\n")
                
                if len(files) > 10:
                    f.write(f"    ... and {len(files) - 10} more files\n")
                
                f.write("\n")
        
        # Cleanup recommendations
        f.write("## Cleanup Recommendations\n\n")
        f.write("### Phase 1: Critical Files (20+ dead items)\n")
        f.write("- Start with files that have the most dead code\n")
        f.write("- Focus on one file at a time\n")
        f.write("- Test thoroughly after each cleanup\n\n")
        
        f.write("### Phase 2: High Priority Files (15+ dead items)\n")
        f.write("- Continue with high-impact files\n")
        f.write("- Group similar files for batch cleanup\n\n")
        
        f.write("### Phase 3: Medium Priority Files (10+ dead items)\n")
        f.write("- Clean up remaining significant dead code\n")
        f.write("- Consider refactoring opportunities\n\n")
        
        f.write("### Phase 4: Low Priority Files (5+ dead items)\n")
        f.write("- Final cleanup phase\n")
        f.write("- Focus on code quality improvements\n\n")
        
        f.write("### Phase 5: Minimal Priority Files (1+ dead items)\n")
        f.write("- Polish and optimization\n")
        f.write("- Consider if cleanup is worth the effort\n\n")
    
    print(f"✅ Generated summary report: {summary_file}")


def generate_cleanup_script(categories: dict, output_dir: Path):
    """Generate a Python script for automated cleanup."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_file = output_dir / f"automated_dead_code_cleanup_{timestamp}.py"
    
    with open(script_file, 'w') as f:
        f.write('#!/usr/bin/env python3\n')
        f.write('"""\n')
        f.write('Automated dead code cleanup script.\n')
        f.write('Generated from interaction mapping analysis.\n')
        f.write('"""\n\n')
        f.write('import ast\n')
        f.write('import os\n')
        f.write('from pathlib import Path\n')
        f.write('from typing import List, Dict, Any\n\n')
        f.write('class DeadCodeCleaner:\n')
        f.write('    """Automated dead code cleaner."""\n\n')
        f.write('    def __init__(self):\n')
        f.write('        self.cleaned_files = []\n')
        f.write('        self.errors = []\n\n')
        f.write('    def clean_file(self, file_path: str, dead_items: List[str]) -> bool:\n')
        f.write('        """Clean dead code from a single file."""\n')
        f.write('        try:\n')
        f.write('            # Implementation would go here\n')
        f.write('            print(f"Cleaning {{file_path}} - {{len(dead_items)}} dead items")\n')
        f.write('            return True\n')
        f.write('        except Exception as e:\n')
        f.write('            print(f"Error cleaning {{file_path}}: {{e}}")\n')
        f.write('            return False\n\n')
        f.write('    def run_cleanup(self):\n')
        f.write('        """Run the cleanup process."""\n')
        f.write('        print("Starting automated dead code cleanup...")\n\n')
        
        # Add cleanup targets
        for threshold, category_key, priority_label, _ in [
            (20, '20_plus', 'CRITICAL', 'Immediate action required'),
            (15, '15_plus', 'HIGH', 'High priority cleanup'),
            (10, '10_plus', 'MEDIUM', 'Medium priority cleanup'),
            (5, '5_plus', 'LOW', 'Low priority cleanup'),
            (1, '1_plus', 'MINIMAL', 'Minimal priority cleanup')
        ]:
            files = categories[category_key]
            if files:
                f.write(f'        # {priority_label} Priority ({threshold}+ dead items)\n')
                f.write(f'        print("\\n{priority_label} Priority Files ({threshold}+ dead items):")\n')
                for file_path, count, dead_items in files[:5]:  # Top 5 for script
                    f.write(f'        self.clean_file("{file_path}", {dead_items})\n')
                f.write('\n')
        
        f.write('        print("\\nCleanup completed!")\n')
        f.write('        print(f"Cleaned files: {len(self.cleaned_files)}")\n')
        f.write('        print(f"Errors: {len(self.errors)}")\n\n')
        f.write('if __name__ == "__main__":\n')
        f.write('    cleaner = DeadCodeCleaner()\n')
        f.write('    cleaner.run_cleanup()\n')
    
    print(f"✅ Generated cleanup script: {script_file}")


def main():
    """Main function to generate priority lists."""
    # Find the most recent dead code cleanup report
    reports_dir = Path("code_quality/reports/interaction_mapping")
    
    if not reports_dir.exists():
        print("❌ Reports directory not found. Run the interaction mapping pipeline first.")
        return
    
    # Find the most recent dead code cleanup report
    cleanup_files = list(reports_dir.glob("dead_code_cleanup_recommendations_*.json"))
    if not cleanup_files:
        print("❌ No dead code cleanup reports found. Run the interaction mapping pipeline first.")
        return
    
    # Use the most recent file
    latest_file = max(cleanup_files, key=lambda f: f.stat().st_mtime)
    print(f"📁 Using dead code report: {latest_file}")
    
    # Load data
    dead_code_data = load_dead_code_data(latest_file)
    if not dead_code_data:
        return
    
    # Categorize files
    categories = categorize_files_by_dead_code_count(dead_code_data)
    
    # Create output directory
    output_dir = Path("dead_code_cleanup_priority_lists")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n📊 DEAD CODE PRIORITY ANALYSIS:")
    print(f"   🔴 Critical (20+): {len(categories['20_plus'])} files")
    print(f"   🟠 High (15+): {len(categories['15_plus'])} files")
    print(f"   🟡 Medium (10+): {len(categories['10_plus'])} files")
    print(f"   🟢 Low (5+): {len(categories['5_plus'])} files")
    print(f"   🔵 Minimal (1+): {len(categories['1_plus'])} files")
    
    # Generate reports
    generate_priority_report(categories, output_dir)
    generate_summary_report(categories, output_dir)
    generate_cleanup_script(categories, output_dir)
    
    print(f"\n✅ All priority lists generated in: {output_dir}")
    print(f"📋 Review the summary report for cleanup recommendations")


if __name__ == "__main__":
    main()
