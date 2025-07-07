#!/usr/bin/env python3
"""
Best ROMs Analyzer
Analyzes the enhanced generalization analysis JSON to find and display the best performing ROMs
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


def load_analysis_results(json_file: Path) -> Dict:
    """Load the enhanced analysis results from JSON file"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return None


def analyze_best_roms(data: Dict, top_n: int = 10) -> List[Dict]:
    """Find the top N best performing ROMs"""
    
    if 'results' not in data:
        print("Error: No 'results' found in JSON data")
        return []
    
    results = data['results']
    
    # Sort by overall success rate (generalization score)
    sorted_roms = sorted(results, 
                        key=lambda x: x['enhanced_analysis']['overall_success_rate'], 
                        reverse=True)
    
    return sorted_roms[:top_n]


def display_rom_details(rom: Dict, rank: int) -> None:
    """Display detailed information about a ROM"""
    
    filename = rom['filename']
    orig = rom['original_discovery']
    enhanced = rom['enhanced_analysis']
    classification = rom['classification']
    
    print(f"\n{rank:2d}. {filename}")
    print("=" * (len(filename) + 4))
    
    # Basic performance
    success_rate = enhanced['overall_success_rate']
    data_points = f"{enhanced['successful_data_points']}/{enhanced['total_data_points']}"
    print(f"   Overall Success Rate: {success_rate:.1%}")
    print(f"   Data Points: {data_points}")
    print(f"   Subsequences Tested: {enhanced['subsequences_tested']}")
    
    # Original discovery info
    print(f"   Original Discovery: {orig['range']} ({orig['length']} elements, {orig['direction']})")
    
    # Classification
    if classification['consistent_sorting']:
        print(f"   ✓ CONSISTENT SORTING ALGORITHM")
    elif classification['pattern_specific']:
        print(f"   ✗ PATTERN-SPECIFIC ONLY")
    else:
        print(f"   ~ PARTIAL GENERALIZATION ({classification['generalization_score']:.1%})")
    
    # Best subsequence performance
    best_subseq = enhanced.get('best_subsequence')
    if best_subseq:
        range_name, best_rate, best_stats = best_subseq
        print(f"   Best Subsequence: {range_name} ({best_rate:.1%} success)")
        print(f"                     Worked on {best_stats['success']}/{best_stats['total']} test patterns")
    
    # Subsequence breakdown (show top 3 performing subsequences)
    subseq_performance = enhanced.get('subsequence_performance', {})
    if subseq_performance:
        # Sort subsequences by success rate
        sorted_subseqs = sorted(subseq_performance.items(), 
                               key=lambda x: x[1]['success'] / x[1]['total'] if x[1]['total'] > 0 else 0, 
                               reverse=True)
        
        print(f"   Top 3 Subsequences:")
        for i, (range_name, stats) in enumerate(sorted_subseqs[:3]):
            rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
            print(f"      {i+1}. {range_name}: {rate:.1%} ({stats['success']}/{stats['total']})")


def create_summary_report(data: Dict, best_roms: List[Dict], output_file: Path = None) -> None:
    """Create a summary report of the best ROMs"""
    
    meta = data.get('meta', {})
    
    report_lines = []
    report_lines.append("BEST ROMs ANALYSIS REPORT")
    report_lines.append("=" * 50)
    report_lines.append("")
    
    # Overall statistics
    report_lines.append("OVERALL STATISTICS:")
    report_lines.append(f"   Total ROMs analyzed: {meta.get('total_roms', 'unknown')}")
    report_lines.append(f"   Total data points: {meta.get('total_data_points', 'unknown'):,}")
    report_lines.append(f"   Consistent algorithms: {meta.get('consistent_algorithms', 'unknown')}")
    report_lines.append(f"   Partial generalizers: {meta.get('partial_generalizers', 'unknown')}")
    
    if meta.get('total_data_points', 0) > 0:
        overall_rate = meta.get('successful_data_points', 0) / meta.get('total_data_points', 1)
        report_lines.append(f"   Overall success rate: {overall_rate:.1%}")
    
    report_lines.append("")
    
    # Best performers summary
    report_lines.append(f"TOP {len(best_roms)} PERFORMERS:")
    report_lines.append("-" * 30)
    
    for i, rom in enumerate(best_roms):
        rate = rom['enhanced_analysis']['overall_success_rate']
        points = f"{rom['enhanced_analysis']['successful_data_points']}/{rom['enhanced_analysis']['total_data_points']}"
        filename = rom['filename']
        
        # Truncate long filenames for readability
        display_name = filename if len(filename) <= 50 else filename[:47] + "..."
        
        report_lines.append(f"{i+1:2d}. {rate:5.1%} | {points:>7} | {display_name}")
    
    report_lines.append("")
    
    # Analysis insights
    if best_roms:
        best_rate = best_roms[0]['enhanced_analysis']['overall_success_rate']
        worst_rate = best_roms[-1]['enhanced_analysis']['overall_success_rate']
        
        report_lines.append("KEY INSIGHTS:")
        report_lines.append(f"   Best performance: {best_rate:.1%}")
        report_lines.append(f"   Worst in top {len(best_roms)}: {worst_rate:.1%}")
        report_lines.append(f"   Performance range: {worst_rate:.1%} - {best_rate:.1%}")
        
        # Count how many achieved different thresholds
        thresholds = [0.5, 0.4, 0.3, 0.2]
        for threshold in thresholds:
            count = len([r for r in best_roms if r['enhanced_analysis']['overall_success_rate'] >= threshold])
            report_lines.append(f"   ROMs >={threshold:.0%}: {count}")
    
    # Print to console
    for line in report_lines:
        print(line)
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        print(f"\nReport saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze enhanced generalization analysis results to find best ROMs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python best_roms_analyzer.py enhanced_generalization_analysis.json
  python best_roms_analyzer.py results.json --top 20 --detailed
  python best_roms_analyzer.py results.json --summary-only --report best_roms_report.txt

This script analyzes the JSON output from the enhanced ROM generalization tester
to identify and display the best performing ROMs with detailed statistics.
        """
    )
    
    parser.add_argument('json_file', type=Path,
                       help='JSON file with enhanced analysis results')
    parser.add_argument('--top', '-n', type=int, default=10,
                       help='Number of top ROMs to show (default: 10)')
    parser.add_argument('--detailed', '-d', action='store_true',
                       help='Show detailed information for each ROM')
    parser.add_argument('--summary-only', '-s', action='store_true',
                       help='Show only summary report, not individual ROM details')
    parser.add_argument('--report', '-r', type=Path,
                       help='Save summary report to file')
    parser.add_argument('--min-rate', type=float, default=0.0,
                       help='Minimum success rate to include (default: 0.0)')
    
    args = parser.parse_args()
    
    # Validate input
    if not args.json_file.exists():
        print(f"Error: File not found: {args.json_file}")
        return 1
    
    # Load data
    print(f"Loading analysis results from: {args.json_file}")
    data = load_analysis_results(args.json_file)
    if not data:
        return 1
    
    # Find best ROMs
    print(f"Finding top {args.top} ROMs...")
    best_roms = analyze_best_roms(data, args.top)
    
    if not best_roms:
        print("No ROMs found in the data!")
        return 1
    
    # Filter by minimum rate if specified
    if args.min_rate > 0:
        filtered_roms = [rom for rom in best_roms 
                        if rom['enhanced_analysis']['overall_success_rate'] >= args.min_rate]
        print(f"Filtered to {len(filtered_roms)} ROMs with ≥{args.min_rate:.1%} success rate")
        best_roms = filtered_roms
    
    # Create summary report
    create_summary_report(data, best_roms, args.report)
    
    # Show detailed ROM information if requested and not summary-only
    if args.detailed and not args.summary_only:
        print(f"\nDETAILED ROM ANALYSIS:")
        print("=" * 40)
        
        for i, rom in enumerate(best_roms):
            display_rom_details(rom, i + 1)
    
    elif not args.summary_only:
        # Show basic info for each ROM
        print(f"\nTOP {len(best_roms)} ROMs (use --detailed for more info):")
        print("-" * 60)
        
        for i, rom in enumerate(best_roms):
            rate = rom['enhanced_analysis']['overall_success_rate']
            points = f"{rom['enhanced_analysis']['successful_data_points']}/{rom['enhanced_analysis']['total_data_points']}"
            filename = rom['filename']
            
            # Show best subsequence
            best_subseq = rom['enhanced_analysis'].get('best_subsequence')
            best_info = ""
            if best_subseq:
                range_name, best_rate, _ = best_subseq
                best_info = f" | Best: {range_name} ({best_rate:.1%})"
            
            print(f"{i+1:2d}. {rate:5.1%} | {points:>7}{best_info}")
            print(f"    {filename}")
    
    return 0


if __name__ == "__main__":
    exit(main())