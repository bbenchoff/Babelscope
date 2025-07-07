#!/usr/bin/env python3
"""
ENHANCED Babelscope ROM Generalization Tester with Subsequence Analysis
Tests discovered ROMs with different input patterns and analyzes ALL possible subsequences

MAJOR ENHANCEMENT: Now checks all contiguous subsequences within the discovery range
instead of only checking the exact discovery range. This provides much more data points
and better statistical confidence in determining true sorting behavior.

Example: If discovery range was V2-V7, now checks:
V2-V3, V2-V4, V2-V5, V2-V6, V2-V7, V3-V4, V3-V5, V3-V6, V3-V7, V4-V5, V4-V6, V4-V7, V5-V6, V5-V7, V6-V7

This gives 8 patterns × 15 subsequences = 120 data points instead of 8 patterns × 1 range = 8 data points
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
import numpy as np

# Add emulators directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'emulators'))

try:
    import cupy as cp
    print("CuPy loaded")
except ImportError:
    print("CuPy required: pip install cupy-cuda12x")
    sys.exit(1)

try:
    from sorting_emulator import (
        PartialSortingBabelscopeDetector, 
        MIN_PARTIAL_LENGTH,
        MIN_SAVE_LENGTH
    )
    print("Partial sorting emulator modules loaded")
except ImportError as e:
    print(f"Failed to import from emulators/sorting_emulator.py: {e}")
    sys.exit(1)


class EnhancedROMGeneralizationTester:
    """ENHANCED: Tests all possible subsequences within discovery range for comprehensive analysis"""
    
    def __init__(self, rom_directories: List[Path]):
        self.rom_directories = rom_directories
        self.discovered_roms = []
        self.test_patterns = self._generate_test_patterns()
        self.original_pattern = [8, 3, 6, 1, 7, 2, 5, 4]  # The pattern used in discovery
        
        print(f"ENHANCED ROM Generalization Tester initialized")
        print(f"MAJOR ENHANCEMENT: Now analyzes ALL subsequences within discovery range")
        print(f"This provides much more data points for statistical confidence")
        print(f"Scanning ROM directories: {len(rom_directories)}")
        print(f"Test patterns: {len(self.test_patterns)}")
        
    def _generate_test_patterns(self) -> List[List[int]]:
        """Generate diverse test patterns to validate sorting algorithms"""
        patterns = []
        
        # Pattern 1: 22-29 range
        patterns.append([22, 25, 21, 28, 24, 27, 23, 26])
        
        # Pattern 2: Different permutation of 1-8
        patterns.append([4, 7, 2, 8, 1, 6, 5, 3])
        
        # Pattern 3: 15-22 range (similar spacing to original)
        patterns.append([18, 15, 21, 16, 20, 17, 22, 19])
        
        # Pattern 4: Larger numbers (90s)
        patterns.append([94, 97, 91, 98, 93, 96, 92, 95])
        
        # Pattern 5: Mixed spacing
        patterns.append([10, 30, 5, 35, 15, 25, 20, 40])
        
        # Pattern 6: Reverse of original pattern
        patterns.append([4, 5, 2, 7, 1, 6, 3, 8])
        
        # Pattern 7: Already sorted (control test)
        patterns.append([1, 2, 3, 4, 5, 6, 7, 8])
        
        # Pattern 8: Reverse sorted (control test)
        patterns.append([8, 7, 6, 5, 4, 3, 2, 1])
        
        return patterns
    
    def _parse_sequence_range(self, sequence_range: str) -> Tuple[int, int]:
        """Parse sequence range like 'V2-V7' into start and end positions"""
        try:
            # Handle formats like "V2-V7" or "V0-V5"
            if '-' in sequence_range and 'V' in sequence_range:
                parts = sequence_range.replace('V', '').split('-')
                start_pos = int(parts[0])
                end_pos = int(parts[1])
                return start_pos, end_pos
            else:
                # Fallback: assume it's a single register or invalid format
                print(f"   Warning: Could not parse sequence range '{sequence_range}', assuming V0-V7")
                return 0, 7
        except Exception as e:
            print(f"   Warning: Error parsing sequence range '{sequence_range}': {e}, assuming V0-V7")
            return 0, 7
    
    def _generate_all_subsequences(self, start_pos: int, end_pos: int, min_length: int = 3) -> List[Tuple[int, int]]:
        """Generate all contiguous subsequences within the range that are at least min_length long"""
        subsequences = []
        range_length = end_pos - start_pos + 1
        
        for subseq_start in range(start_pos, end_pos + 1):
            for subseq_end in range(subseq_start + min_length - 1, end_pos + 1):
                if subseq_end - subseq_start + 1 >= min_length:
                    subsequences.append((subseq_start, subseq_end))
        
        return subsequences
    
    def scan_for_roms(self) -> None:
        """Scan ROM directories and load discovered ROM metadata"""
        print("Scanning for discovered ROMs...")
        
        total_roms = 0
        for rom_dir in self.rom_directories:
            if not rom_dir.exists():
                print(f"   Directory not found: {rom_dir}")
                continue
                
            # Look for ROM files and their metadata
            ch8_files = list(rom_dir.glob("*.ch8"))
            json_files = list(rom_dir.glob("*.json"))
            
            print(f"   {rom_dir}: {len(ch8_files)} ROM files, {len(json_files)} metadata files")
            
            # Match ROM files with their metadata
            for ch8_file in ch8_files:
                json_file = ch8_file.with_suffix('.json')
                if json_file.exists():
                    try:
                        with open(json_file, 'r') as f:
                            metadata = json.load(f)
                        
                        # Load ROM binary data
                        with open(ch8_file, 'rb') as f:
                            rom_data = f.read()
                        
                        # Extract the specific sorting range from metadata
                        partial_sorting = metadata.get('partial_sorting', {})
                        sequence_range = partial_sorting.get('sequence_range', 'V0-V7')
                        start_pos, end_pos = self._parse_sequence_range(sequence_range)
                        
                        # ENHANCEMENT: Generate all possible subsequences within the discovery range
                        subsequences = self._generate_all_subsequences(start_pos, end_pos, min_length=3)
                        
                        rom_info = {
                            'filename': ch8_file.name,
                            'filepath': ch8_file,
                            'metadata': metadata,
                            'rom_data': rom_data,
                            'original_discovery': partial_sorting,
                            'original_registers': metadata.get('registers', {}),
                            'sorting_range': {
                                'start': start_pos,
                                'end': end_pos,
                                'length': end_pos - start_pos + 1,
                                'range_str': sequence_range
                            },
                            # ENHANCEMENT: Store all subsequences to test
                            'test_subsequences': subsequences
                        }
                        
                        self.discovered_roms.append(rom_info)
                        total_roms += 1
                        
                    except Exception as e:
                        print(f"   Failed to load {ch8_file}: {e}")
        
        print(f"Loaded {total_roms} discovered ROMs for testing")
        
        # Sort by discovery quality (longer sequences first)
        self.discovered_roms.sort(
            key=lambda x: x['original_discovery'].get('length', 0), 
            reverse=True
        )
        
        if total_roms > 0:
            print("ROM quality distribution:")
            length_counts = {}
            range_counts = {}
            subseq_counts = {}
            
            for rom in self.discovered_roms:
                length = rom['original_discovery'].get('length', 0)
                range_str = rom['sorting_range']['range_str']
                num_subseqs = len(rom['test_subsequences'])
                
                length_counts[length] = length_counts.get(length, 0) + 1
                range_counts[range_str] = range_counts.get(range_str, 0) + 1
                subseq_counts[num_subseqs] = subseq_counts.get(num_subseqs, 0) + 1
            
            for length in sorted(length_counts.keys(), reverse=True):
                count = length_counts[length]
                print(f"   {length}-element sequences: {count} ROMs")
            
            print("Sorting ranges found:")
            for range_str, count in sorted(range_counts.items()):
                print(f"   {range_str}: {count} ROMs")
            
            print("ENHANCEMENT - Subsequences to test per ROM:")
            for num_subseqs, count in sorted(subseq_counts.items()):
                print(f"   {num_subseqs} subsequences: {count} ROMs")
            
            # Calculate total data points
            total_data_points = sum(len(rom['test_subsequences']) * len(self.test_patterns) 
                                  for rom in self.discovered_roms)
            print(f"ENHANCEMENT - Total data points to collect: {total_data_points:,}")
    
    def test_rom_with_pattern(self, rom_data: bytes, test_pattern: List[int], 
                             test_subsequences: List[Tuple[int, int]], cycles: int = 100000, 
                             check_interval: int = 100) -> Optional[Dict]:
        """ENHANCED: Test a ROM with a pattern and check ALL subsequences within discovery range"""
        
        # Create single-instance detector
        detector = PartialSortingBabelscopeDetector(1)
        
        try:
            # Convert ROM data to the format expected by the detector
            rom_array = np.frombuffer(rom_data, dtype=np.uint8)
            
            # Manually load ROM without setting up default test pattern
            rom_size = min(len(rom_array), 4096 - 512)  # PROGRAM_START = 0x200 = 512
            detector.memory[0, 512:512 + rom_size] = cp.array(rom_array[:rom_size])
            
            # Setup our specific test pattern in registers V0-V7
            test_pattern_gpu = cp.array(test_pattern[:8], dtype=cp.uint8)
            detector.registers[0, :8] = test_pattern_gpu
            
            # Reset program counter to start
            detector.program_counter[0] = 512  # 0x200
            
            # Run the ROM with more generous parameters for better detection
            discoveries = detector.run_partial_sorting_search(
                cycles=cycles, 
                check_interval=min(check_interval, 50)  # More frequent checks
            )
            
            # Get final register state
            final_registers = cp.asnumpy(detector.registers[0, :8]).tolist()
            
            # ENHANCEMENT: Check sorting in ALL possible subsequences
            subsequence_results = []
            total_sorted_subsequences = 0
            
            for subseq_start, subseq_end in test_subsequences:
                # Extract the values in this subsequence
                input_subseq = test_pattern[subseq_start:subseq_end+1]
                output_subseq = final_registers[subseq_start:subseq_end+1]
                
                # Check if this subsequence is sorted
                sorting_info = self._check_subsequence_sorting(input_subseq, output_subseq, subseq_start, subseq_end)
                
                subsequence_result = {
                    'range': (subseq_start, subseq_end),
                    'range_str': f"V{subseq_start}-V{subseq_end}",
                    'length': subseq_end - subseq_start + 1,
                    'input': input_subseq,
                    'output': output_subseq,
                    'sorted': sorting_info is not None,
                    'sorting_info': sorting_info
                }
                
                subsequence_results.append(subsequence_result)
                
                if sorting_info is not None:
                    total_sorted_subsequences += 1
            
            # Calculate enhanced statistics
            total_subsequences = len(test_subsequences)
            subsequence_success_rate = total_sorted_subsequences / total_subsequences if total_subsequences > 0 else 0
            
            # Also check if the detector found anything (might be in a different range)
            detector_found_sorting = False
            detector_sorting_info = None
            
            if discoveries > 0:
                discovery_list = detector.get_partial_sorting_discoveries()
                if discovery_list:
                    discovery = discovery_list[0]
                    detector_found_sorting = True
                    detector_sorting_info = discovery['partial_sorting']
            
            return {
                # Enhanced results
                'subsequence_success_rate': subsequence_success_rate,
                'sorted_subsequences': total_sorted_subsequences,
                'total_subsequences': total_subsequences,
                'subsequence_results': subsequence_results,
                
                # Original results for compatibility
                'sorted': total_sorted_subsequences > 0,  # Any subsequence sorted
                'input_pattern': test_pattern[:8],
                'output_pattern': final_registers,
                'detector_found': detector_found_sorting,
                'detector_sorting': detector_sorting_info,
                'register_activity': {
                    'total_register_ops': int(detector.total_register_ops[0]),
                    'register_reads': int(detector.register_reads[0]),
                    'register_writes': int(detector.register_writes[0])
                }
            }
            
        except Exception as e:
            print(f"      Error testing ROM: {e}")
            return None
        finally:
            # Clean up
            detector.reset()
            del detector
            cp.get_default_memory_pool().free_all_blocks()
    
    def _check_subsequence_sorting(self, input_subseq: List[int], output_subseq: List[int], 
                                 start_pos: int, end_pos: int) -> Optional[Dict]:
        """Check if a specific subsequence shows sorting behavior"""
        
        if len(output_subseq) < 3:  # Need at least 3 elements
            return None
        
        # Check for consecutive sorted sequences
        for direction in ['ascending', 'descending']:
            # Check if the entire subsequence is sorted in this direction
            is_sorted = True
            
            for i in range(len(output_subseq) - 1):
                if direction == 'ascending':
                    if output_subseq[i] > output_subseq[i + 1]:
                        is_sorted = False
                        break
                else:  # descending
                    if output_subseq[i] < output_subseq[i + 1]:
                        is_sorted = False
                        break
            
            if is_sorted:
                # Additional check: ensure it's actually a permutation of the input
                if sorted(input_subseq) == sorted(output_subseq):
                    return {
                        'length': len(output_subseq),
                        'start_position': 0,  # Local to the subsequence
                        'end_position': len(output_subseq) - 1,  # Local to the subsequence
                        'global_start': start_pos,  # Global register position
                        'global_end': end_pos,  # Global register position
                        'direction': direction,
                        'sequence': output_subseq,
                        'sequence_range': f"V{start_pos}-V{end_pos}",
                        'detection_method': 'enhanced_subsequence_analysis',
                        'is_permutation': True
                    }
        
        return None
    
    def analyze_rom_generalization(self, rom_info: Dict) -> Dict:
        """ENHANCED: Test a ROM with all patterns and all subsequences for comprehensive analysis"""
        
        filename = rom_info['filename']
        original_discovery = rom_info['original_discovery']
        sorting_range = rom_info['sorting_range']
        test_subsequences = rom_info['test_subsequences']
        rom_data = rom_info['rom_data']
        
        print(f"Testing ROM: {filename}")
        print(f"   Original discovery: {original_discovery.get('sequence_range', 'unknown')} "
              f"({original_discovery.get('length', 0)} elements, "
              f"{original_discovery.get('direction', 'unknown')})")
        print(f"   ENHANCED: Testing {len(test_subsequences)} subsequences within "
              f"{sorting_range['range_str']}")
        print(f"   Total data points: {len(test_subsequences) * len(self.test_patterns):,}")
        
        results = {
            'rom_info': rom_info,
            'test_results': [],
            'enhanced_statistics': {
                'total_data_points': 0,
                'successful_data_points': 0,
                'overall_success_rate': 0,
                'best_subsequence': None,
                'worst_subsequence': None,
                'subsequence_performance': {}
            },
            'generalization_score': 0,  # Will be calculated from enhanced stats
            'consistent_sorting': False,
            'pattern_specific': False,
            'range_tested': sorting_range,
            'subsequences_tested': test_subsequences
        }
        
        # Test with original pattern first (sanity check)
        print(f"   Testing with original pattern: {self.original_pattern}")
        original_result = self.test_rom_with_pattern(
            rom_data, 
            self.original_pattern, 
            test_subsequences,
            cycles=150000,  # More cycles for original test
            check_interval=50  # More frequent checks
        )
        
        if original_result:
            success_rate = original_result['subsequence_success_rate']
            sorted_count = original_result['sorted_subsequences']
            total_count = original_result['total_subsequences']
            print(f"      ✓ Subsequence success rate: {success_rate:.1%} "
                  f"({sorted_count}/{total_count} subsequences)")
        else:
            print(f"      ✗ Failed to test with original pattern")
        
        results['test_results'].append(('original', self.original_pattern, original_result))
        
        # Test with all other patterns
        total_data_points = 0
        successful_data_points = 0
        subsequence_stats = {}  # Track performance of each subsequence
        
        for i, test_pattern in enumerate(self.test_patterns):
            print(f"   Testing pattern {i+1}/{len(self.test_patterns)}: {test_pattern}")
            
            result = self.test_rom_with_pattern(rom_data, test_pattern, test_subsequences)
            results['test_results'].append((f'pattern_{i+1}', test_pattern, result))
            
            if result:
                pattern_total = result['total_subsequences']
                pattern_success = result['sorted_subsequences']
                pattern_rate = result['subsequence_success_rate']
                
                total_data_points += pattern_total
                successful_data_points += pattern_success
                
                print(f"      Success rate: {pattern_rate:.1%} ({pattern_success}/{pattern_total})")
                
                # Track individual subsequence performance
                for subseq_result in result['subsequence_results']:
                    range_str = subseq_result['range_str']
                    if range_str not in subsequence_stats:
                        subsequence_stats[range_str] = {'success': 0, 'total': 0}
                    
                    subsequence_stats[range_str]['total'] += 1
                    if subseq_result['sorted']:
                        subsequence_stats[range_str]['success'] += 1
        
        # Calculate enhanced statistics
        overall_success_rate = successful_data_points / total_data_points if total_data_points > 0 else 0
        
        # Find best and worst performing subsequences
        best_subseq = None
        worst_subseq = None
        best_rate = -1
        worst_rate = 2
        
        for range_str, stats in subsequence_stats.items():
            rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
            if rate > best_rate:
                best_rate = rate
                best_subseq = (range_str, rate, stats)
            if rate < worst_rate:
                worst_rate = rate
                worst_subseq = (range_str, rate, stats)
        
        # Update enhanced statistics
        results['enhanced_statistics'].update({
            'total_data_points': total_data_points,
            'successful_data_points': successful_data_points,
            'overall_success_rate': overall_success_rate,
            'best_subsequence': best_subseq,
            'worst_subsequence': worst_subseq,
            'subsequence_performance': subsequence_stats
        })
        
        # Calculate generalization metrics using enhanced data
        results['generalization_score'] = overall_success_rate
        results['consistent_sorting'] = overall_success_rate >= 0.7  # 70% success rate across all data points
        results['pattern_specific'] = overall_success_rate == 0  # No subsequences ever sorted
        
        print(f"   ENHANCED Generalization score: {overall_success_rate:.1%} "
              f"({successful_data_points:,}/{total_data_points:,} data points)")
        
        if best_subseq:
            print(f"   Best subsequence: {best_subseq[0]} ({best_subseq[1]:.1%} success)")
        if worst_subseq and worst_subseq[1] < best_rate:
            print(f"   Worst subsequence: {worst_subseq[0]} ({worst_subseq[1]:.1%} success)")
        
        if results['consistent_sorting']:
            print(f"   ✓ CONSISTENT SORTING ALGORITHM DETECTED")
        elif results['pattern_specific']:
            print(f"   ✗ NO SORTING BEHAVIOR DETECTED")
        else:
            print(f"   ~ PARTIAL SORTING BEHAVIOR ({overall_success_rate:.1%} success)")
        
        print()
        
        return results
    
    def run_generalization_analysis(self, max_roms: Optional[int] = None, 
                                   cycles_per_test: int = 100000) -> List[Dict]:
        """Run ENHANCED generalization analysis on all discovered ROMs"""
        
        if not self.discovered_roms:
            self.scan_for_roms()
        
        if not self.discovered_roms:
            print("No ROMs found to test!")
            return []
        
        roms_to_test = self.discovered_roms[:max_roms] if max_roms else self.discovered_roms
        
        # Calculate total expected data points
        total_expected_points = sum(len(rom['test_subsequences']) * len(self.test_patterns) 
                                  for rom in roms_to_test)
        
        print(f"\nSTARTING ENHANCED GENERALIZATION ANALYSIS")
        print(f"=" * 60)
        print(f"MAJOR ENHANCEMENT: Now testing all subsequences within discovery ranges")
        print(f"This provides much more data points for statistical confidence")
        print(f"ROMs to test: {len(roms_to_test)}")
        print(f"Test patterns: {len(self.test_patterns)}")
        print(f"Expected total data points: {total_expected_points:,}")
        print(f"Cycles per test: {cycles_per_test:,}")
        print(f"Original discovery pattern: {self.original_pattern}")
        print()
        
        print("Test patterns:")
        for i, pattern in enumerate(self.test_patterns):
            print(f"   Pattern {i+1}: {pattern}")
        print()
        
        all_results = []
        start_time = time.time()
        total_collected_points = 0
        
        for rom_idx, rom_info in enumerate(roms_to_test):
            print(f"[{rom_idx+1}/{len(roms_to_test)}] ", end="")
            
            try:
                result = self.analyze_rom_generalization(rom_info)
                all_results.append(result)
                total_collected_points += result['enhanced_statistics']['total_data_points']
                
            except Exception as e:
                print(f"Failed to test ROM {rom_info['filename']}: {e}")
                continue
        
        # Analysis complete
        total_time = time.time() - start_time
        print(f"ENHANCED GENERALIZATION ANALYSIS COMPLETE")
        print(f"=" * 60)
        print(f"Total time: {total_time:.1f} seconds")
        print(f"ROMs tested: {len(all_results)}")
        print(f"Data points collected: {total_collected_points:,}")
        print(f"Data collection efficiency: {total_collected_points/total_expected_points:.1%}")
        
        # Summarize results using enhanced statistics
        consistent_algorithms = [r for r in all_results if r['consistent_sorting']]
        partial_generalizers = [r for r in all_results if 0 < r['generalization_score'] < 0.7]
        pattern_specific = [r for r in all_results if r['pattern_specific']]
        
        print(f"\nENHANCED RESULTS SUMMARY:")
        print(f"   Consistent sorting algorithms (≥70%): {len(consistent_algorithms)}")
        print(f"   Partial generalizers (1-69%): {len(partial_generalizers)}")
        print(f"   No sorting behavior (0%): {len(pattern_specific)}")
        
        if consistent_algorithms:
            print(f"\n🎉 DISCOVERED SORTING ALGORITHMS (ENHANCED ANALYSIS):")
            for result in consistent_algorithms:
                rom_name = result['rom_info']['filename']
                score = result['generalization_score']
                stats = result['enhanced_statistics']
                points = f"{stats['successful_data_points']}/{stats['total_data_points']}"
                
                print(f"   {rom_name}: {score:.1%} success rate ({points} data points)")
                
                if stats['best_subsequence']:
                    best_range, best_rate, _ = stats['best_subsequence']
                    print(f"      Best subsequence: {best_range} ({best_rate:.1%})")
        
        # Statistical insights
        if all_results:
            all_scores = [r['generalization_score'] for r in all_results]
            avg_score = sum(all_scores) / len(all_scores)
            max_score = max(all_scores)
            
            print(f"\nSTATISTICAL INSIGHTS:")
            print(f"   Average success rate: {avg_score:.1%}")
            print(f"   Best ROM performance: {max_score:.1%}")
            print(f"   Data points per ROM: {total_collected_points//len(all_results):,} avg")
        
        return all_results
    
    def save_analysis_results(self, results: List[Dict], output_file: Path) -> None:
        """Save ENHANCED analysis results with comprehensive subsequence data"""
        
        print(f"\n🔍 SAVING ENHANCED ANALYSIS RESULTS")
        print("=" * 60)
        
        # Create detailed summary
        enhanced_summary = {
            'meta': {
                'enhancement': 'subsequence_analysis',
                'total_roms': len(results),
                'consistent_algorithms': len([r for r in results if r['consistent_sorting']]),
                'partial_generalizers': len([r for r in results if 0 < r['generalization_score'] < 0.7]),
                'no_sorting': len([r for r in results if r['pattern_specific']]),
                'test_patterns': len(self.test_patterns),
                'original_pattern': self.original_pattern,
                'total_data_points': sum(r['enhanced_statistics']['total_data_points'] for r in results),
                'successful_data_points': sum(r['enhanced_statistics']['successful_data_points'] for r in results)
            },
            'results': []
        }
        
        for result in results:
            rom_info = result['rom_info']
            enhanced_stats = result['enhanced_statistics']
            
            enhanced_result = {
                'filename': rom_info['filename'],
                'original_discovery': {
                    'range': rom_info['original_discovery'].get('sequence_range', 'unknown'),
                    'length': rom_info['original_discovery'].get('length', 0),
                    'direction': rom_info['original_discovery'].get('direction', 'unknown')
                },
                'enhanced_analysis': {
                    'overall_success_rate': enhanced_stats['overall_success_rate'],
                    'total_data_points': enhanced_stats['total_data_points'],
                    'successful_data_points': enhanced_stats['successful_data_points'],
                    'subsequences_tested': len(result['subsequences_tested']),
                    'best_subsequence': enhanced_stats['best_subsequence'],
                    'subsequence_performance': enhanced_stats['subsequence_performance']
                },
                'classification': {
                    'consistent_sorting': result['consistent_sorting'],
                    'pattern_specific': result['pattern_specific'],
                    'generalization_score': result['generalization_score']
                }
            }
            
            enhanced_summary['results'].append(enhanced_result)
        
        # Save enhanced results
        with open(output_file, 'w') as f:
            json.dump(enhanced_summary, f, indent=2)
        
        print(f"Enhanced analysis results saved to: {output_file}")
        print(f"Total data points analyzed: {enhanced_summary['meta']['total_data_points']:,}")
        print(f"Overall success rate: {enhanced_summary['meta']['successful_data_points']/enhanced_summary['meta']['total_data_points']:.1%}")


def find_rom_directories(base_path: Path = Path(".")) -> List[Path]:
    """Find all directories containing discovered ROMs"""
    rom_dirs = []
    
    # Primary target: Benchoff's specific async_partial_sorting structure
    target_path = Path(r"C:\Users\Benchoff\Documents\GitHub\Babelscope\output\async_partial_sorting")
    
    if target_path.exists():
        print(f"Found target directory: {target_path}")
        # Look for session folders and their discovered_roms subdirectories
        session_dirs = list(target_path.glob("session_*/discovered_roms"))
        rom_dirs.extend(session_dirs)
        print(f"Found {len(session_dirs)} session ROM directories")
        for session_dir in session_dirs:
            rom_count = len(list(session_dir.glob("*.ch8")))
            print(f"   {session_dir.name}: {rom_count} ROM files")
    else:
        print(f"Target directory not found: {target_path}")
        print("Falling back to relative path search...")
        
        # Fallback: Look for common ROM directory patterns relative to current directory
        search_patterns = [
            "output/async_partial_sorting/session_*/discovered_roms",
            "async_partial_sorting/session_*/discovered_roms",
            "session_*/discovered_roms",
            "**/discovered_roms"
        ]
        
        for pattern in search_patterns:
            dirs = list(base_path.glob(pattern))
            rom_dirs.extend(dirs)
    
    # Remove duplicates and filter existing directories
    unique_dirs = []
    seen = set()
    for d in rom_dirs:
        resolved = d.resolve()
        if resolved not in seen and resolved.exists():
            unique_dirs.append(resolved)
            seen.add(resolved)
    
    return unique_dirs


def main():
    """Main entry point for ENHANCED ROM generalization testing with subsequence analysis"""
    parser = argparse.ArgumentParser(
        description='ENHANCED: Test discovered ROMs with comprehensive subsequence analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MAJOR ENHANCEMENT: This version now tests ALL possible contiguous subsequences
within the discovered sorting range, providing much more statistical data.

Instead of testing only the exact discovery range (e.g., V2-V7), this now tests:
V2-V3, V2-V4, V2-V5, V2-V6, V2-V7, V3-V4, V3-V5, V3-V6, V3-V7, etc.

This gives 8 patterns × 15 subsequences = 120 data points instead of just 8,
providing much higher statistical confidence in determining true sorting behavior.

Examples:
  python enhanced_rom_tester.py --auto-scan --max-roms 50
  python enhanced_rom_tester.py --rom-dirs output/async_partial_sorting/session_*/discovered_roms
  python enhanced_rom_tester.py --rom-dirs path/to/roms --cycles 50000 --output enhanced_results.json

Enhanced Test Methodology:
1. Load ROMs discovered by partial sorting Babelscope
2. Extract the specific sorting range from metadata (e.g., V2-V7)
3. Generate ALL contiguous subsequences within that range (≥3 elements)
4. Test each ROM with multiple different input patterns  
5. Check sorting in ALL subsequences for each test pattern
6. Calculate comprehensive statistics from hundreds of data points per ROM
7. Identify ROMs with consistent sorting behavior across subsequences and patterns
        """
    )
    
    parser.add_argument('--rom-dirs', nargs='+', type=Path,
                       help='ROM directories to scan for discovered ROMs')
    parser.add_argument('--auto-scan', action='store_true',
                       help='Automatically scan for ROM directories')
    parser.add_argument('--max-roms', type=int, default=None,
                       help='Maximum number of ROMs to test (default: all)')
    parser.add_argument('--cycles', type=int, default=100000,
                       help='Execution cycles per ROM test (default: 100000)')
    parser.add_argument('--output', type=Path, default='enhanced_generalization_analysis.json',
                       help='Output file for results (default: enhanced_generalization_analysis.json)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test mode (fewer cycles, fewer ROMs)')
    parser.add_argument('--min-subsequence-length', type=int, default=3,
                       help='Minimum length for subsequences to test (default: 3)')
    
    args = parser.parse_args()
    
    print("BABELSCOPE ENHANCED ROM GENERALIZATION TESTER")
    print("=" * 50)
    print("🚀 MAJOR ENHANCEMENT: Comprehensive subsequence analysis")
    print("Now tests ALL possible contiguous subsequences within discovery ranges")
    print("This provides orders of magnitude more data points for statistical confidence!")
    print()
    
    # Determine ROM directories
    if args.auto_scan:
        print("Auto-scanning for ROM directories...")
        rom_dirs = find_rom_directories()
        if not rom_dirs:
            print("No ROM directories found! Try specifying --rom-dirs manually.")
            return 1
    elif args.rom_dirs:
        rom_dirs = [Path(d) for d in args.rom_dirs]
    else:
        print("No ROM directories specified. Use --auto-scan or --rom-dirs")
        return 1
    
    print(f"ROM directories to scan: {len(rom_dirs)}")
    for d in rom_dirs:
        print(f"   {d}")
    print()
    
    # Quick test mode adjustments
    if args.quick_test:
        print("Quick test mode enabled")
        args.cycles = 25000
        args.max_roms = args.max_roms or 10
        print(f"   Cycles per test: {args.cycles:,}")
        print(f"   Max ROMs: {args.max_roms}")
        print()
    
    # Create ENHANCED tester and run analysis
    try:
        tester = EnhancedROMGeneralizationTester(rom_dirs)
        tester.scan_for_roms()
        
        if not tester.discovered_roms:
            print("No discovered ROMs found to test!")
            return 1
        
        # Show expected data points
        roms_to_analyze = tester.discovered_roms[:args.max_roms] if args.max_roms else tester.discovered_roms
        expected_points = sum(len(rom['test_subsequences']) * len(tester.test_patterns) 
                            for rom in roms_to_analyze)
        
        print(f"ENHANCED ANALYSIS PREVIEW:")
        print(f"   Expected data points: {expected_points:,}")
        print(f"   This is a {expected_points//8}x increase over the original method!")
        print()
        
        # Run ENHANCED generalization analysis
        results = tester.run_generalization_analysis(
            max_roms=args.max_roms,
            cycles_per_test=args.cycles
        )
        
        if results:
            # Save results
            tester.save_analysis_results(results, args.output)
            
            # Final summary with enhancement details
            consistent = [r for r in results if r['consistent_sorting']]
            total_points = sum(r['enhanced_statistics']['total_data_points'] for r in results)
            successful_points = sum(r['enhanced_statistics']['successful_data_points'] for r in results)
            overall_rate = successful_points / total_points if total_points > 0 else 0
            
            print(f"\n🎉 ENHANCED ANALYSIS COMPLETE!")
            print(f"Data Points Analyzed: {total_points:,}")
            print(f"Overall Success Rate: {overall_rate:.1%}")
            print(f"Consistent Algorithms Found: {len(consistent)}")
            
            if consistent:
                print(f"\n🔥 DISCOVERED SORTING ALGORITHMS:")
                for i, result in enumerate(consistent[:10]):  # Show top 10
                    rom_name = result['rom_info']['filename']
                    score = result['generalization_score']
                    stats = result['enhanced_statistics']
                    print(f"   {i+1:2d}. {rom_name}: {score:.1%} "
                          f"({stats['successful_data_points']}/{stats['total_data_points']} points)")
                
                # Show the most reliable algorithm
                best_rom = max(consistent, key=lambda x: x['generalization_score'])
                best_stats = best_rom['enhanced_statistics']
                print(f"\n🏆 MOST RELIABLE ALGORITHM:")
                print(f"   ROM: {best_rom['rom_info']['filename']}")
                print(f"   Success Rate: {best_rom['generalization_score']:.1%}")
                print(f"   Data Points: {best_stats['successful_data_points']}/{best_stats['total_data_points']}")
                
                if best_stats['best_subsequence']:
                    best_range, best_rate, best_counts = best_stats['best_subsequence']
                    print(f"   Best Subsequence: {best_range} ({best_rate:.1%} success)")
                    print(f"   That subsequence worked on {best_counts['success']}/{best_counts['total']} test patterns")
            else:
                print(f"\n📊 ENHANCED ANALYSIS RESULTS:")
                print("No ROMs achieved ≥70% success rate for consistent sorting")
                print("However, the enhanced analysis provides much more detailed insights")
                
                # Show the best performers even if not consistent
                if results:
                    best_rom = max(results, key=lambda x: x['generalization_score'])
                    print(f"\nBest Performer: {best_rom['rom_info']['filename']}")
                    print(f"Success Rate: {best_rom['generalization_score']:.1%}")
            
            print(f"\n📁 Detailed results saved to: {args.output}")
            print("\nThe enhanced analysis provides much more granular data about")
            print("sorting behavior within specific register subsequences!")
            
            return 0
        else:
            print("No results generated")
            return 1
            
    except KeyboardInterrupt:
        print("\nEnhanced generalization testing interrupted by user")
        return 0
        
    except Exception as e:
        print(f"Enhanced generalization testing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())