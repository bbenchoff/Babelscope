#!/usr/bin/env python3
"""
Batch Sorting ROM Tester for Babelscope
Tests all discovered sorting ROMs with comprehensive analysis
"""

import os
import glob
import json
import time
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Optional imports for enhanced features
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available, some analysis features disabled")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, visualizations disabled")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Import your enhanced emulator
import sys
sys.path.append('emulators')
from sorting_emulator import EnhancedChip8Emulator, SortingDetector

class BatchSortingTester:
    """Comprehensive batch tester for sorting ROMs"""
    
    def __init__(self, 
                 rom_directory: str,
                 output_directory: str = "sorting_analysis_results",
                 test_cycles: int = 100000,
                 parallel_workers: int = 4):
        
        self.rom_directory = Path(rom_directory)
        self.output_directory = Path(output_directory)
        self.test_cycles = test_cycles
        self.parallel_workers = parallel_workers
        
        # Create output directories
        self.output_directory.mkdir(exist_ok=True)
        (self.output_directory / "logs").mkdir(exist_ok=True)
        (self.output_directory / "reports").mkdir(exist_ok=True)
        (self.output_directory / "visualizations").mkdir(exist_ok=True)
        (self.output_directory / "verified_sorts").mkdir(exist_ok=True)
        
        # Test array configurations - all exactly 8 bytes
        self.test_arrays = {
            'random_small': np.array([5, 2, 8, 1, 9, 3, 7, 4], dtype=np.uint8),
            'random_medium': np.array([8, 3, 6, 1, 4, 7, 2, 5], dtype=np.uint8),
            'reverse_sorted': np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=np.uint8),
            'nearly_sorted': np.array([1, 2, 4, 3, 5, 6, 8, 7], dtype=np.uint8),  # Only 2 swaps needed
            'duplicates': np.array([3, 1, 3, 2, 1, 2, 3, 1], dtype=np.uint8),
            'worst_case': np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=np.uint8),      # Worst case for bubble sort
            'already_sorted': np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint8), # Control test
        }
        
        self.results = []
        
    def find_rom_files(self, pattern: str = "*.ch8") -> List[Path]:
        """Find all ROM files in the directory"""
        rom_files = list(self.rom_directory.glob(pattern))
        rom_files.extend(self.rom_directory.glob("*.rom"))
        rom_files.extend(self.rom_directory.glob("*.bin"))
        
        print(f"Found {len(rom_files)} ROM files in {self.rom_directory}")
        return rom_files
    
    def test_single_rom(self, rom_path: Path, test_array_name: str, test_array: np.ndarray) -> Dict[str, Any]:
        """Test a single ROM with a specific test array"""
        
        rom_id = rom_path.stem
        test_id = f"{rom_id}_{test_array_name}"
        
        try:
            # Create emulator with logging
            log_file = self.output_directory / "logs" / f"{test_id}.log"
            emulator = EnhancedChip8Emulator(
                enable_sorting_detection=True,
                debug_file=str(log_file)
            )
            
            # Setup test array
            emulator.setup_sorting_test(test_array.copy())
            
            # Load ROM
            with open(rom_path, 'rb') as f:
                rom_data = f.read()
            
            rom_hash = hashlib.sha256(rom_data).hexdigest()[:16]
            emulator.load_rom(rom_data)
            
            # Run emulation
            start_time = time.time()
            emulator.run(max_cycles=self.test_cycles)
            execution_time = time.time() - start_time
            
            # Analyze results
            sorting_analysis = emulator.analyze_sorting_results()
            enhanced_stats = emulator.get_enhanced_stats()
            
            # Compile comprehensive result
            result = {
                'rom_file': str(rom_path),
                'rom_id': rom_id,
                'rom_hash': rom_hash,
                'rom_size': len(rom_data),
                'test_array_name': test_array_name,
                'test_array': test_array.tolist(),
                'execution_time': execution_time,
                'cycles_executed': enhanced_stats['cycles_executed'],
                'instructions_executed': enhanced_stats['instructions_executed'],
                'crashed': emulator.crashed,
                'halted': emulator.halt,
                
                # Sorting specific results
                'initial_array': sorting_analysis.get('initial_array', []).tolist() if 'initial_array' in sorting_analysis else [],
                'final_array': sorting_analysis.get('final_array', []).tolist() if 'final_array' in sorting_analysis else [],
                'is_sorted': sorting_analysis.get('is_sorted', False),
                'is_reverse_sorted': sorting_analysis.get('is_reverse_sorted', False),
                'sortedness_measure': sorting_analysis.get('sortedness_measure', 0.0),
                'initial_sortedness': sorting_analysis.get('initial_sortedness', 0.0),
                'improvement': sorting_analysis.get('improvement', 0.0),
                'algorithm_type': sorting_analysis.get('algorithm_type', 'unknown'),
                
                # Memory access patterns
                'sort_reads': sorting_analysis.get('activity', {}).get('sort_reads', 0),
                'sort_writes': sorting_analysis.get('activity', {}).get('sort_writes', 0),
                'total_comparisons': sorting_analysis.get('activity', {}).get('total_comparisons', 0),
                'total_swaps': sorting_analysis.get('activity', {}).get('total_swaps', 0),
                'unique_addresses_read': sorting_analysis.get('activity', {}).get('unique_addresses_read', 0),
                'unique_addresses_written': sorting_analysis.get('activity', {}).get('unique_addresses_written', 0),
                
                # Enhanced stats
                'complexity_score': enhanced_stats.get('complexity_score', 0.0),
                'branch_taken_ratio': enhanced_stats.get('branch_taken_ratio', 0.0),
                'memory_region_accesses': enhanced_stats.get('memory_region_accesses', 0),
                'subroutine_calls': enhanced_stats.get('subroutine_calls', 0),
                'infinite_loops_detected': enhanced_stats.get('infinite_loops_detected', 0),
                
                # Classification
                'success': sorting_analysis.get('is_sorted', False) or sorting_analysis.get('improvement', 0) > 0.1,
                'timestamp': time.time()
            }
            
            return result
            
        except Exception as e:
            return {
                'rom_file': str(rom_path),
                'rom_id': rom_id,
                'test_array_name': test_array_name,
                'error': str(e),
                'success': False,
                'timestamp': time.time()
            }
    
    def test_rom_wrapper(self, args: Tuple[Path, str, np.ndarray]) -> Dict[str, Any]:
        """Wrapper for parallel processing"""
        rom_path, test_array_name, test_array = args
        return self.test_single_rom(rom_path, test_array_name, test_array)
    
    def run_batch_test(self, rom_files: List[Path], use_parallel: bool = True) -> List[Dict[str, Any]]:
        """Run batch testing on all ROMs with all test arrays"""
        
        # Prepare all test combinations
        test_jobs = []
        for rom_path in rom_files:
            for test_name, test_array in self.test_arrays.items():
                test_jobs.append((rom_path, test_name, test_array))
        
        print(f"Running {len(test_jobs)} test combinations...")
        print(f"ROMs: {len(rom_files)}, Test arrays: {len(self.test_arrays)}")
        
        results = []
        
        if use_parallel and self.parallel_workers > 1:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=self.parallel_workers) as executor:
                future_to_job = {executor.submit(self.test_rom_wrapper, job): job for job in test_jobs}
                
                completed = 0
                for future in as_completed(future_to_job):
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    if completed % 10 == 0:
                        print(f"Completed {completed}/{len(test_jobs)} tests...")
                        
                    # Save intermediate results every 50 tests
                    if completed % 50 == 0:
                        self.save_intermediate_results(results)
        else:
            # Sequential execution
            for i, job in enumerate(test_jobs):
                result = self.test_rom_wrapper(job)
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"Completed {i + 1}/{len(test_jobs)} tests...")
        
        return results
    
    def save_intermediate_results(self, results: List[Dict[str, Any]]):
        """Save intermediate results to prevent data loss"""
        intermediate_file = self.output_directory / "intermediate_results.json"
        with open(intermediate_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Comprehensive analysis of all test results"""
        
        # Filter successful results
        successful_results = [r for r in results if r.get('success', False)]
        sorted_results = [r for r in results if r.get('is_sorted', False)]
        improved_results = [r for r in results if r.get('improvement', 0) > 0.1]
        
        analysis = {
            'summary': {
                'total_tests': len(results),
                'successful_tests': len(successful_results),
                'sorted_results': len(sorted_results),
                'improved_results': len(improved_results),
                'success_rate': len(successful_results) / len(results) if results else 0,
                'sorting_rate': len(sorted_results) / len(results) if results else 0,
            },
            
            'algorithm_types': {},
            'test_array_performance': {},
            'top_performers': [],
            'statistics': {}
        }
        
        # Algorithm type analysis (without pandas)
        algorithm_counts = {}
        for r in results:
            algo_type = r.get('algorithm_type', 'unknown')
            algorithm_counts[algo_type] = algorithm_counts.get(algo_type, 0) + 1
        analysis['algorithm_types'] = algorithm_counts
        
        # Analysis by test array (without pandas)
        for test_name in self.test_arrays.keys():
            test_results = [r for r in results if r.get('test_array_name') == test_name]
            if test_results:
                successes = len([r for r in test_results if r.get('success', False)])
                sorted_count = len([r for r in test_results if r.get('is_sorted', False)])
                improvements = [r.get('improvement', 0) for r in test_results if 'improvement' in r]
                complexities = [r.get('complexity_score', 0) for r in test_results if 'complexity_score' in r]
                
                analysis['test_array_performance'][test_name] = {
                    'total_tests': len(test_results),
                    'successes': successes,
                    'sorted': sorted_count,
                    'avg_improvement': sum(improvements) / len(improvements) if improvements else 0,
                    'avg_complexity': sum(complexities) / len(complexities) if complexities else 0,
                }
        
        # Top performers
        if successful_results:
            # Sort by improvement score
            top_by_improvement = sorted(successful_results, 
                                      key=lambda x: x.get('improvement', 0), 
                                      reverse=True)[:10]
            
            analysis['top_performers'] = [
                {
                    'rom_id': r['rom_id'],
                    'test_array': r['test_array_name'],
                    'improvement': r.get('improvement', 0),
                    'algorithm_type': r.get('algorithm_type', 'unknown'),
                    'is_sorted': r.get('is_sorted', False),
                    'complexity_score': r.get('complexity_score', 0)
                }
                for r in top_by_improvement
            ]
        
        # Statistical analysis (without pandas)
        improvements = [r.get('improvement', 0) for r in results if 'improvement' in r]
        complexities = [r.get('complexity_score', 0) for r in results if 'complexity_score' in r]
        exec_times = [r.get('execution_time', 0) for r in results if 'execution_time' in r]
        
        if improvements:
            analysis['statistics'] = {
                'improvement_mean': sum(improvements) / len(improvements),
                'improvement_std': (sum((x - sum(improvements)/len(improvements))**2 for x in improvements) / len(improvements))**0.5,
                'improvement_max': max(improvements),
                'complexity_mean': sum(complexities) / len(complexities) if complexities else 0,
                'execution_time_mean': sum(exec_times) / len(exec_times) if exec_times else 0,
            }
        
        return analysis
    
    def generate_report(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]):
        """Generate comprehensive HTML report"""
        
        report_file = self.output_directory / "reports" / "sorting_analysis_report.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Babelscope Sorting Algorithm Analysis Report</title>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #333; }}
                .summary {{ background: #f0f0f0; padding: 20px; border-radius: 8px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background: white; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .success {{ color: green; font-weight: bold; }}
                .failure {{ color: red; }}
                .algorithm-type {{ background: #e6f3ff; padding: 3px 8px; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <h1>Babelscope Sorting Algorithm Analysis Report</h1>
            <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="summary">
                <h2>Summary Statistics</h2>
                <div class="metric">
                    <h3>{analysis['summary']['total_tests']}</h3>
                    <p>Total Tests</p>
                </div>
                <div class="metric">
                    <h3>{analysis['summary']['sorted_results']}</h3>
                    <p>Successful Sorts</p>
                </div>
                <div class="metric">
                    <h3>{analysis['summary']['success_rate']:.1%}</h3>
                    <p>Success Rate</p>
                </div>
                <div class="metric">
                    <h3>{len(analysis['algorithm_types'])}</h3>
                    <p>Algorithm Types Found</p>
                </div>
            </div>
            
            <h2>Algorithm Types Discovered</h2>
            <table>
                <tr><th>Algorithm Type</th><th>Count</th><th>Percentage</th></tr>
        """
        
        for algo_type, count in analysis['algorithm_types'].items():
            percentage = (count / analysis['summary']['total_tests']) * 100
            html_content += f"""
                <tr>
                    <td><span class="algorithm-type">{algo_type}</span></td>
                    <td>{count}</td>
                    <td>{percentage:.1f}%</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Top Performing ROMs</h2>
            <table>
                <tr><th>ROM ID</th><th>Test Array</th><th>Improvement</th><th>Algorithm Type</th><th>Sorted</th></tr>
        """
        
        for performer in analysis['top_performers']:
            sorted_badge = '<span class="success">YES</span>' if performer['is_sorted'] else 'NO'
            html_content += f"""
                <tr>
                    <td>{performer['rom_id']}</td>
                    <td>{performer['test_array']}</td>
                    <td>{performer['improvement']:.3f}</td>
                    <td><span class="algorithm-type">{performer['algorithm_type']}</span></td>
                    <td>{sorted_badge}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Test Array Performance</h2>
            <table>
                <tr><th>Test Array</th><th>Total Tests</th><th>Successes</th><th>Success Rate</th><th>Avg Improvement</th></tr>
        """
        
        for test_name, perf in analysis['test_array_performance'].items():
            success_rate = (perf['successes'] / perf['total_tests']) * 100 if perf['total_tests'] > 0 else 0
            html_content += f"""
                <tr>
                    <td>{test_name}</td>
                    <td>{perf['total_tests']}</td>
                    <td>{perf['successes']}</td>
                    <td>{success_rate:.1f}%</td>
                    <td>{perf['avg_improvement']:.3f}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Report generated: {report_file}")
    
    def create_visualizations(self, results: List[Dict[str, Any]]):
        """Create visualization plots (only if matplotlib is available)"""
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available, skipping visualizations")
            return
            
        if not results:
            print("No data to visualize")
            return
        
        # Set up plotting style
        if HAS_SEABORN:
            plt.style.use('seaborn-v0_8')
        else:
            plt.style.use('default')
        fig_size = (12, 8)
        
        # 1. Success rate by test array
        test_array_success = {}
        for result in results:
            test_name = result.get('test_array_name', 'unknown')
            if test_name not in test_array_success:
                test_array_success[test_name] = {'total': 0, 'success': 0}
            test_array_success[test_name]['total'] += 1
            if result.get('success', False):
                test_array_success[test_name]['success'] += 1
        
        if test_array_success:
            plt.figure(figsize=fig_size)
            test_names = list(test_array_success.keys())
            success_rates = [test_array_success[name]['success'] / test_array_success[name]['total'] 
                           for name in test_names]
            
            plt.bar(test_names, success_rates)
            plt.title('Success Rate by Test Array Type')
            plt.ylabel('Success Rate')
            plt.xlabel('Test Array Type')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.output_directory / "visualizations" / "success_by_array.png", dpi=300)
            plt.close()
        
        # 2. Improvement distribution
        improvements = [r.get('improvement', 0) for r in results if 'improvement' in r and r.get('improvement') is not None]
        if improvements:
            plt.figure(figsize=fig_size)
            plt.hist(improvements, bins=50, alpha=0.7, edgecolor='black')
            plt.title('Distribution of Sorting Improvement Scores')
            plt.xlabel('Improvement Score')
            plt.ylabel('Frequency')
            plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No Improvement')
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.output_directory / "visualizations" / "improvement_distribution.png", dpi=300)
            plt.close()
        
        # 3. Algorithm types pie chart
        algo_counts = {}
        for result in results:
            algo_type = result.get('algorithm_type', 'unknown')
            algo_counts[algo_type] = algo_counts.get(algo_type, 0) + 1
        
        if algo_counts:
            plt.figure(figsize=(10, 8))
            plt.pie(algo_counts.values(), labels=algo_counts.keys(), autopct='%1.1f%%')
            plt.title('Distribution of Detected Algorithm Types')
            plt.tight_layout()
            plt.savefig(self.output_directory / "visualizations" / "algorithm_types.png", dpi=300)
            plt.close()
        
        # 4. Complexity vs Improvement scatter
        complexity_scores = []
        improvement_scores = []
        is_sorted_list = []
        
        for result in results:
            if 'complexity_score' in result and 'improvement' in result:
                complexity_scores.append(result.get('complexity_score', 0))
                improvement_scores.append(result.get('improvement', 0))
                is_sorted_list.append(result.get('is_sorted', False))
        
        if complexity_scores and improvement_scores:
            plt.figure(figsize=fig_size)
            colors = ['green' if sorted_flag else 'red' for sorted_flag in is_sorted_list]
            plt.scatter(complexity_scores, improvement_scores, alpha=0.6, c=colors)
            plt.xlabel('Complexity Score')
            plt.ylabel('Improvement Score')
            plt.title('Complexity vs Improvement (Green=Sorted, Red=Not Sorted)')
            plt.tight_layout()
            plt.savefig(self.output_directory / "visualizations" / "complexity_vs_improvement.png", dpi=300)
            plt.close()
        
        print(f"Visualizations saved to {self.output_directory / 'visualizations'}")
    
    def copy_verified_sorts(self, results: List[Dict[str, Any]]):
        """Copy ROMs that successfully sorted to a separate directory"""
        import shutil
        
        verified_dir = self.output_directory / "verified_sorts"
        
        sorted_roms = [r for r in results if r.get('is_sorted', False)]
        
        for result in sorted_roms:
            try:
                src_path = Path(result['rom_file'])
                dst_path = verified_dir / f"{result['rom_id']}_{result['test_array_name']}.ch8"
                shutil.copy2(src_path, dst_path)
                
                # Create metadata file
                metadata = {
                    'original_file': str(src_path),
                    'test_array_name': result['test_array_name'],
                    'algorithm_type': result.get('algorithm_type', 'unknown'),
                    'improvement': result.get('improvement', 0),
                    'complexity_score': result.get('complexity_score', 0),
                    'initial_array': result.get('initial_array', []),
                    'final_array': result.get('final_array', [])
                }
                
                metadata_path = verified_dir / f"{result['rom_id']}_{result['test_array_name']}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                    
            except Exception as e:
                print(f"Error copying {result['rom_file']}: {e}")
        
        print(f"Copied {len(sorted_roms)} verified sorting ROMs to {verified_dir}")
    
    def run_complete_analysis(self, rom_pattern: str = "*.ch8"):
        """Run the complete analysis pipeline"""
        
        print("=" * 60)
        print("BABELSCOPE SORTING ROM BATCH ANALYSIS")
        print("=" * 60)
        
        # Find ROM files
        rom_files = self.find_rom_files(rom_pattern)
        if not rom_files:
            print(f"No ROM files found in {self.rom_directory}")
            return
        
        print(f"Testing {len(rom_files)} ROMs with {len(self.test_arrays)} test arrays each")
        print(f"Total test combinations: {len(rom_files) * len(self.test_arrays)}")
        print(f"Estimated time: {(len(rom_files) * len(self.test_arrays) * 2) // 60} minutes")
        print()
        
        # Run batch testing
        start_time = time.time()
        results = self.run_batch_test(rom_files)
        total_time = time.time() - start_time
        
        # Save raw results
        results_file = self.output_directory / "raw_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Analyze results
        print("\nAnalyzing results...")
        analysis = self.analyze_results(results)
        
        # Save analysis
        analysis_file = self.output_directory / "analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Generate report
        print("Generating report...")
        self.generate_report(results, analysis)
        
        # Create visualizations
        print("Creating visualizations...")
        self.create_visualizations(results)
        
        # Copy verified sorting ROMs
        print("Copying verified sorting ROMs...")
        self.copy_verified_sorts(results)
        
        # Print summary
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Total execution time: {total_time:.1f} seconds")
        print(f"Tests completed: {len(results)}")
        print(f"Successful sorts: {analysis['summary']['sorted_results']}")
        print(f"Success rate: {analysis['summary']['success_rate']:.1%}")
        print(f"Results saved to: {self.output_directory}")
        print("\nTop algorithm types found:")
        for algo_type, count in list(analysis['algorithm_types'].items())[:5]:
            print(f"  {algo_type}: {count}")
        
        return results, analysis


def main():
    parser = argparse.ArgumentParser(description="Batch test sorting ROMs from Babelscope")
    parser.add_argument("rom_directory", help="Directory containing ROM files")
    parser.add_argument("-o", "--output", default="sorting_analysis_results", 
                       help="Output directory for results")
    parser.add_argument("-c", "--cycles", type=int, default=100000,
                       help="Maximum cycles to run each ROM")
    parser.add_argument("-w", "--workers", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("-p", "--pattern", default="*.ch8",
                       help="File pattern to match ROM files")
    parser.add_argument("--no-parallel", action="store_true",
                       help="Disable parallel processing")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.rom_directory):
        print(f"Error: ROM directory '{args.rom_directory}' does not exist")
        return 1
    
    # Create tester
    tester = BatchSortingTester(
        rom_directory=args.rom_directory,
        output_directory=args.output,
        test_cycles=args.cycles,
        parallel_workers=args.workers if not args.no_parallel else 1
    )
    
    # Run analysis
    try:
        results, analysis = tester.run_complete_analysis(args.pattern)
        print(f"\nAnalysis complete! Check {args.output} for detailed results.")
        return 0
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        return 1
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1


if __name__ == "__main__":
    exit(main())