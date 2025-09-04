#!/usr/bin/env python3
"""
Command Line Interface for Code Complexity Analysis Pipeline
"""

import argparse
import sys
import os
from pathlib import Path

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

from complexity_pipeline import ComplexityPipeline
from config.complexity_config import ComplexityConfig


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Code Complexity Analysis Pipeline - Combines PyExamine, Radon, and Xenon',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single file
  python cli.py analyze /path/to/file.py
  
  # Analyze a directory
  python cli.py analyze /path/to/directory
  
  # Use custom configuration
  python cli.py analyze /path/to/code --config custom_config.yaml
  
  # Generate specific output formats
  python cli.py analyze /path/to/code --format html --format markdown
  
  # Check tool availability
  python cli.py check-tools
  
  # Generate configuration template
  python cli.py generate-config --output my_config.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze code complexity')
    analyze_parser.add_argument('target', help='Target file or directory to analyze')
    analyze_parser.add_argument('--config', '-c', help='Path to configuration file')
    analyze_parser.add_argument('--output', '-o', help='Output file name for results')
    analyze_parser.add_argument('--format', '-f', choices=['json', 'html', 'markdown', 'summary'], 
                               action='append', default=['json'], 
                               help='Output format (can be specified multiple times)')
    analyze_parser.add_argument('--verbose', '-v', action='store_true', 
                               help='Enable verbose logging')
    analyze_parser.add_argument('--quiet', '-q', action='store_true', 
                               help='Suppress output except errors')
    
    # Check tools command
    check_parser = subparsers.add_parser('check-tools', help='Check if analysis tools are available')
    check_parser.add_argument('--verbose', '-v', action='store_true', 
                             help='Show detailed tool information')
    
    # Generate config command
    config_parser = subparsers.add_parser('generate-config', help='Generate configuration template')
    config_parser.add_argument('--output', '-o', default='complexity_config.yaml',
                              help='Output configuration file name')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
        
    try:
        if args.command == 'analyze':
            return run_analysis(args)
        elif args.command == 'check-tools':
            return check_tools(args)
        elif args.command == 'generate-config':
            return generate_config(args)
        else:
            parser.print_help()
            return 1
            
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose if hasattr(args, 'verbose') else False:
            import traceback
            traceback.print_exc()
        return 1


def run_analysis(args):
    """Run complexity analysis"""
    import logging
    from datetime import datetime
    
    # Configure logging
    log_level = logging.ERROR if args.quiet else (logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    # Initialize pipeline
    pipeline = ComplexityPipeline(args.config)
    
    # Run analysis
    print(f"Starting complexity analysis on: {args.target}")
    results = pipeline.run_full_analysis(args.target)
    
    # Save results
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'complexity_analysis_{timestamp}.json'
        
    pipeline.save_results(results, output_file)
    
    # Print summary
    if not args.quiet:
        print_summary(results)
        
    print(f"Analysis completed. Results saved to: {output_file}")
    return 0


def check_tools(args):
    """Check if analysis tools are available"""
    from analyzers.pyexamine_analyzer import PyExamineAnalyzer
    from analyzers.radon_analyzer import RadonAnalyzer
    from analyzers.xenon_analyzer import XenonAnalyzer
    
    config = ComplexityConfig()
    
    tools = [
        ('PyExamine', PyExamineAnalyzer(config)),
        ('Radon', RadonAnalyzer(config)),
        ('Xenon', XenonAnalyzer(config))
    ]
    
    print("Checking analysis tools availability:")
    print("=" * 40)
    
    all_available = True
    
    for tool_name, analyzer in tools:
        is_available = analyzer.is_available()
        status = "✓ Available" if is_available else "✗ Not available"
        print(f"{tool_name:12} {status}")
        
        if not is_available:
            all_available = False
            
        if args.verbose and not is_available:
            print(f"  Install with: pip install {tool_name.lower()}")
            
    print("=" * 40)
    
    if all_available:
        print("All tools are available!")
        return 0
    else:
        print("Some tools are missing. Install them to use all features.")
        return 1


def generate_config(args):
    """Generate configuration template"""
    config = ComplexityConfig()
    config.save_config(args.output)
    print(f"Configuration template saved to: {args.output}")
    return 0


def print_summary(results):
    """Print analysis summary"""
    print("\n" + "=" * 50)
    print("ANALYSIS SUMMARY")
    print("=" * 50)
    
    file_analysis = results.get('file_analysis', {})
    directory_analysis = results.get('directory_analysis', {})
    
    if file_analysis:
        scores = [m.get('combined_score', 0) for m in file_analysis.values() 
                 if m.get('combined_score') is not None]
        
        if scores:
            print(f"Files analyzed: {len(scores)}")
            print(f"Average complexity: {sum(scores)/len(scores):.3f}")
            print(f"Highest complexity: {max(scores):.3f}")
            print(f"Lowest complexity: {min(scores):.3f}")
            
            # Complexity distribution
            low_count = len([s for s in scores if s >= 0.7])
            medium_count = len([s for s in scores if 0.4 <= s < 0.7])
            high_count = len([s for s in scores if s < 0.4])
            
            print(f"\nComplexity distribution:")
            print(f"  Low (≥0.7):    {low_count:3d} files ({low_count/len(scores)*100:5.1f}%)")
            print(f"  Medium (0.4-0.7): {medium_count:3d} files ({medium_count/len(scores)*100:5.1f}%)")
            print(f"  High (<0.4):   {high_count:3d} files ({high_count/len(scores)*100:5.1f}%)")
            
    if directory_analysis:
        print(f"\nDirectories analyzed: {len(directory_analysis)}")
        for dir_path, metrics in directory_analysis.items():
            print(f"  {dir_path}: {metrics.get('total_files_analyzed', 0)} files")
            
    print("=" * 50)


if __name__ == '__main__':
    sys.exit(main())