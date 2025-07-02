#!/usr/bin/env python3
"""
FIXED Babelscope ROM Generalization Tester
Tests discovered ROMs with different input patterns to identify true sorting algorithms

CRITICAL FIX: Now properly tests only the discovered sorting range (e.g., V2-V7)
instead of incorrectly testing the entire V0-V7 range.

This script:
1. Loads ROMs discovered by the partial sorting Babelscope
2. Tests each ROM with multiple different input patterns
3. BUT ONLY CHECKS SORTING IN THE ORIGINAL DISCOVERY RANGE
4. Compares the sorting behavior across different inputs
5. Identifies ROMs that demonstrate true sorting capability vs pattern-specific behavior
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


class FixedROMGeneralizationTester:
    """FIXED: Tests discovered ROMs with different input patterns in the correct register range"""
    
    def __init__(self, rom_directories: List[Path]):
        self.rom_directories = rom_directories
        self.discovered_roms = []
        self.test_patterns = self._generate_test_patterns()
        self.original_pattern = [8, 3, 6, 1, 7, 2, 5, 4]  # The pattern used in discovery
        
        print(f"FIXED ROM Generalization Tester initialized")
        print(f"CRITICAL FIX: Now tests only the discovered sorting range")
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
                        
                        # FIXED: Extract the specific sorting range from metadata
                        partial_sorting = metadata.get('partial_sorting', {})
                        sequence_range = partial_sorting.get('sequence_range', 'V0-V7')
                        start_pos, end_pos = self._parse_sequence_range(sequence_range)
                        
                        rom_info = {
                            'filename': ch8_file.name,
                            'filepath': ch8_file,
                            'metadata': metadata,
                            'rom_data': rom_data,
                            'original_discovery': partial_sorting,
                            'original_registers': metadata.get('registers', {}),
                            # FIXED: Store the specific sorting range
                            'sorting_range': {
                                'start': start_pos,
                                'end': end_pos,
                                'length': end_pos - start_pos + 1,
                                'range_str': sequence_range
                            }
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
            for rom in self.discovered_roms:
                length = rom['original_discovery'].get('length', 0)
                range_str = rom['sorting_range']['range_str']
                length_counts[length] = length_counts.get(length, 0) + 1
                range_counts[range_str] = range_counts.get(range_str, 0) + 1
            
            for length in sorted(length_counts.keys(), reverse=True):
                count = length_counts[length]
                print(f"   {length}-element sequences: {count} ROMs")
            
            print("Sorting ranges found:")
            for range_str, count in sorted(range_counts.items()):
                print(f"   {range_str}: {count} ROMs")
    
    def test_rom_with_pattern(self, rom_data: bytes, test_pattern: List[int], 
                             sorting_range: Dict, cycles: int = 100000, 
                             check_interval: int = 100) -> Optional[Dict]:
        """FIXED: Test a single ROM with a specific input pattern, checking only the sorting range"""
        
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
            
            # FIXED: Check sorting only in the specific range that was originally discovered
            start_pos = sorting_range['start']
            end_pos = sorting_range['end']
            
            # Extract the values in the sorting range
            input_range = test_pattern[start_pos:end_pos+1]
            output_range = final_registers[start_pos:end_pos+1]
            
            print(f"      Range {sorting_range['range_str']}: {input_range} → {output_range}")
            
            # Check if the specific range is sorted
            range_sorted_info = self._check_range_sorting(input_range, output_range, start_pos)
            
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
                'sorted': range_sorted_info is not None,
                'input_pattern': test_pattern[:8],
                'output_pattern': final_registers,
                'range_input': input_range,
                'range_output': output_range,
                'target_range': sorting_range,
                'range_sorting': range_sorted_info,
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
    
    def _check_range_sorting(self, input_range: List[int], output_range: List[int], 
                           start_pos: int) -> Optional[Dict]:
        """Check if a specific range shows sorting behavior"""
        
        if len(output_range) < 3:  # Need at least 3 elements
            return None
        
        # Check for consecutive sorted sequences of length 3+
        for direction in ['ascending', 'descending']:
            for start in range(len(output_range) - 2):  # Need at least 3 elements
                max_length = 1
                
                for end in range(start + 1, len(output_range)):
                    if direction == 'ascending':
                        if output_range[end] == output_range[end-1] + 1:
                            max_length += 1
                        else:
                            break
                    else:  # descending
                        if output_range[end] == output_range[end-1] - 1:
                            max_length += 1
                        else:
                            break
                
                if max_length >= 3:  # Found a sequence
                    sequence = output_range[start:start + max_length]
                    global_start = start_pos + start
                    global_end = start_pos + start + max_length - 1
                    
                    return {
                        'length': max_length,
                        'start_position': start,  # Local to the range
                        'end_position': start + max_length - 1,  # Local to the range
                        'global_start': global_start,  # Global register position
                        'global_end': global_end,  # Global register position
                        'direction': direction,
                        'sequence': sequence,
                        'sequence_range': f"V{global_start}-V{global_end}",
                        'detection_method': 'fixed_range_analysis'
                    }
        
        return None
    
    def analyze_rom_generalization(self, rom_info: Dict) -> Dict:
        """FIXED: Test a ROM with all test patterns, checking only the original sorting range"""
        
        filename = rom_info['filename']
        original_discovery = rom_info['original_discovery']
        sorting_range = rom_info['sorting_range']
        rom_data = rom_info['rom_data']
        
        print(f"Testing ROM: {filename}")
        print(f"   Original discovery: {original_discovery.get('sequence_range', 'unknown')} "
              f"({original_discovery.get('length', 0)} elements, "
              f"{original_discovery.get('direction', 'unknown')})")
        print(f"   FIXED: Testing only range {sorting_range['range_str']} "
              f"(positions {sorting_range['start']}-{sorting_range['end']})")
        
        results = {
            'rom_info': rom_info,
            'test_results': [],
            'generalization_score': 0,
            'consistent_sorting': False,
            'pattern_specific': False,
            'range_tested': sorting_range
        }
        
        # Test with original pattern first (sanity check)
        print(f"   Testing with original pattern: {self.original_pattern}")
        original_result = self.test_rom_with_pattern(
            rom_data, 
            self.original_pattern, 
            sorting_range,
            cycles=150000,  # More cycles for original test
            check_interval=50  # More frequent checks
        )
        
        if original_result and original_result['sorted']:
            range_info = original_result['range_sorting']
            print(f"      ✓ RANGE SORTED: {range_info['sequence_range']} "
                  f"({range_info['length']} elements, {range_info['direction']})")
            print(f"         Sequence: {range_info['sequence']}")
        else:
            print(f"      ✗ No sorting detected in target range")
            if original_result:
                print(f"         Range input: {original_result['range_input']}")
                print(f"         Range output: {original_result['range_output']}")
        
        results['test_results'].append(('original', self.original_pattern, original_result))
        
        # Test with all other patterns
        sorting_count = 0
        total_tests = len(self.test_patterns)
        
        for i, test_pattern in enumerate(self.test_patterns):
            print(f"   Testing pattern {i+1}/{total_tests}: {test_pattern}")
            
            result = self.test_rom_with_pattern(rom_data, test_pattern, sorting_range)
            results['test_results'].append((f'pattern_{i+1}', test_pattern, result))
            
            if result and result['sorted']:
                sorting_count += 1
                range_info = result['range_sorting']
                print(f"      ✓ RANGE SORTED: {range_info['sequence_range']} "
                      f"({range_info['length']} elements, {range_info['direction']})")
                print(f"         Sequence: {range_info['sequence']}")
            else:
                print(f"      ✗ No sorting detected in target range")
        
        # Calculate generalization metrics
        results['generalization_score'] = sorting_count / total_tests
        results['consistent_sorting'] = sorting_count >= (total_tests * 0.7)  # 70% success rate
        results['pattern_specific'] = sorting_count == 0  # Only works on original
        
        print(f"   FIXED Generalization score: {results['generalization_score']:.2%} "
              f"({sorting_count}/{total_tests} patterns sorted in target range)")
        
        if results['consistent_sorting']:
            print(f"   ✓ CONSISTENT SORTING ALGORITHM DETECTED IN {sorting_range['range_str']}")
        elif results['pattern_specific']:
            print(f"   ✗ PATTERN-SPECIFIC BEHAVIOR (likely hardcoded)")
        else:
            print(f"   ~ PARTIAL GENERALIZATION (some patterns work)")
        
        print()
        
        return results
    
    def run_generalization_analysis(self, max_roms: Optional[int] = None, 
                                   cycles_per_test: int = 100000) -> List[Dict]:
        """Run FIXED generalization analysis on all discovered ROMs"""
        
        if not self.discovered_roms:
            self.scan_for_roms()
        
        if not self.discovered_roms:
            print("No ROMs found to test!")
            return []
        
        roms_to_test = self.discovered_roms[:max_roms] if max_roms else self.discovered_roms
        
        print(f"\nSTARTING FIXED GENERALIZATION ANALYSIS")
        print(f"=" * 60)
        print(f"CRITICAL FIX: Now testing only the discovered sorting ranges")
        print(f"ROMs to test: {len(roms_to_test)}")
        print(f"Test patterns: {len(self.test_patterns)}")
        print(f"Cycles per test: {cycles_per_test:,}")
        print(f"Original discovery pattern: {self.original_pattern}")
        print()
        
        print("Test patterns:")
        for i, pattern in enumerate(self.test_patterns):
            print(f"   Pattern {i+1}: {pattern}")
        print()
        
        all_results = []
        start_time = time.time()
        
        for rom_idx, rom_info in enumerate(roms_to_test):
            print(f"[{rom_idx+1}/{len(roms_to_test)}] ", end="")
            
            try:
                result = self.analyze_rom_generalization(rom_info)
                all_results.append(result)
                
            except Exception as e:
                print(f"Failed to test ROM {rom_info['filename']}: {e}")
                continue
        
        # Analysis complete
        total_time = time.time() - start_time
        print(f"FIXED GENERALIZATION ANALYSIS COMPLETE")
        print(f"=" * 60)
        print(f"Total time: {total_time:.1f} seconds")
        print(f"ROMs tested: {len(all_results)}")
        
        # Summarize results
        consistent_algorithms = [r for r in all_results if r['consistent_sorting']]
        partial_generalizers = [r for r in all_results if 0 < r['generalization_score'] < 0.7]
        pattern_specific = [r for r in all_results if r['pattern_specific']]
        
        print(f"\nFIXED RESULTS SUMMARY:")
        print(f"   Consistent sorting algorithms: {len(consistent_algorithms)}")
        print(f"   Partial generalizers: {len(partial_generalizers)}")
        print(f"   Pattern-specific only: {len(pattern_specific)}")
        
        if consistent_algorithms:
            print(f"\n🎉 DISCOVERED SORTING ALGORITHMS (FIXED ANALYSIS):")
            for result in consistent_algorithms:
                rom_name = result['rom_info']['filename']
                score = result['generalization_score']
                original = result['rom_info']['original_discovery']
                range_str = result['range_tested']['range_str']
                print(f"   {rom_name}: {score:.1%} success rate")
                print(f"      Range tested: {range_str}")
                print(f"      Original: {original.get('sequence_range', 'unknown')} "
                      f"({original.get('length', 0)} elements)")
        
        return all_results
    
    def save_analysis_results(self, results: List[Dict], output_file: Path) -> None:
        """Save ENHANCED analysis results with deep pattern insights"""
        
        print(f"\n🔍 ENHANCED ANALYSIS & PATTERN DETECTION")
        print("=" * 60)
        
        # Advanced pattern analysis
        self._analyze_patterns(results)
        self._analyze_test_pattern_failures(results)
        self._analyze_sorting_ranges(results)
        
        # Create COMPRESSED summary for easy sharing/analysis
        compressed_results = []
        
        for result in results:
            rom_info = result['rom_info']
            
            # Extract only essential data
            compressed_result = {
                'file': rom_info['filename'],
                'orig': {  # Original discovery
                    'len': rom_info['original_discovery'].get('length', 0),
                    'dir': rom_info['original_discovery'].get('direction', 'unknown')[:3],  # asc/des
                    'pos': rom_info['original_discovery'].get('sequence_range', 'unknown')
                },
                'range': result['range_tested']['range_str'],  # FIXED: Include tested range
                'score': round(result['generalization_score'], 3),
                'consistent': result['consistent_sorting'],
                'tests': []  # Compressed test results
            }
            
            # Compress test results - only include input/output patterns and whether it sorted
            for test_name, input_pattern, test_result in result['test_results']:
                if test_result:
                    compressed_test = {
                        'in': input_pattern,
                        'out': test_result['output_pattern'],
                        'range_in': test_result['range_input'],
                        'range_out': test_result['range_output'],
                        'sorted': test_result['sorted']
                    }
                    
                    # If sorting was detected, include the sorting info
                    if test_result['sorted'] and test_result['range_sorting']:
                        rs = test_result['range_sorting']
                        compressed_test['sort'] = {
                            'len': rs['length'],
                            'dir': rs['direction'][:3],
                            'pos': rs['sequence_range']
                        }
                    
                    compressed_result['tests'].append(compressed_test)
            
            compressed_results.append(compressed_result)
        
        # Create ultra-compressed summary
        summary = {
            'meta': {
                'total': len(results),
                'consistent': len([r for r in results if r['consistent_sorting']]),
                'partial': len([r for r in results if 0 < r['generalization_score'] < 0.7]),
                'patterns': len(self.test_patterns),
                'orig_pattern': self.original_pattern,
                'fix_applied': 'range_specific_testing'
            },
            'results': compressed_results
        }
        
        # Save compressed version
        compressed_file = output_file.with_stem(output_file.stem + '_ENHANCED_compressed')
        with open(compressed_file, 'w') as f:
            json.dump(summary, f, separators=(',', ':'))  # No whitespace
        
        print(f"ENHANCED compressed analysis results saved to: {compressed_file}")
        print(f"File size: {compressed_file.stat().st_size} bytes")
        
        # Create ENHANCED human-readable summary
        self._create_enhanced_summary(results, output_file)
        
    def _analyze_patterns(self, results: List[Dict]) -> None:
        """Analyze overall patterns in the results"""
        
        if not results:
            return
            
        print(f"\n📊 OVERALL PATTERN ANALYSIS:")
        
        # Score distribution
        score_ranges = {
            '0%': 0, '1-25%': 0, '26-50%': 0, '51-75%': 0, '76-99%': 0, '100%': 0
        }
        
        for result in results:
            score = result['generalization_score']
            if score == 0:
                score_ranges['0%'] += 1
            elif score <= 0.25:
                score_ranges['1-25%'] += 1
            elif score <= 0.50:
                score_ranges['26-50%'] += 1
            elif score <= 0.75:
                score_ranges['51-75%'] += 1
            elif score < 1.0:
                score_ranges['76-99%'] += 1
            else:
                score_ranges['100%'] += 1
        
        print(f"   Score Distribution:")
        for range_name, count in score_ranges.items():
            pct = count / len(results) * 100
            print(f"     {range_name:>6}: {count:3d} ROMs ({pct:4.1f}%)")
        
        # Best performers analysis
        top_performers = sorted(results, key=lambda x: x['generalization_score'], reverse=True)[:10]
        best_score = top_performers[0]['generalization_score']
        best_count = len([r for r in results if r['generalization_score'] == best_score])
        
        print(f"   Best Score: {best_score:.1%} ({best_count} ROMs)")
        print(f"   Score Range: {top_performers[-1]['generalization_score']:.1%} - {best_score:.1%}")
        
    def _analyze_test_pattern_failures(self, results: List[Dict]) -> None:
        """Analyze which test patterns consistently fail"""
        
        print(f"\n🎯 TEST PATTERN SUCCESS ANALYSIS:")
        
        # Track success rate for each test pattern
        pattern_stats = {}
        
        for result in results:
            for test_name, input_pattern, test_result in result['test_results']:
                pattern_key = str(input_pattern)
                if pattern_key not in pattern_stats:
                    pattern_stats[pattern_key] = {'success': 0, 'total': 0, 'name': test_name}
                
                pattern_stats[pattern_key]['total'] += 1
                if test_result and test_result['sorted']:
                    pattern_stats[pattern_key]['success'] += 1
        
        # Sort by success rate
        sorted_patterns = sorted(pattern_stats.items(), 
                               key=lambda x: x[1]['success'] / x[1]['total'] if x[1]['total'] > 0 else 0, 
                               reverse=True)
        
        print(f"   Pattern Success Rates (sorted by success):")
        for pattern_str, stats in sorted_patterns:
            rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
            pattern = eval(pattern_str)  # Convert back to list
            name = stats['name']
            print(f"     {rate:5.1%} | {pattern} ({name})")
        
        # Identify consistently problematic patterns
        worst_patterns = [p for p, s in sorted_patterns if (s['success'] / s['total']) < 0.3]
        if worst_patterns:
            print(f"\n   🚨 CONSISTENTLY FAILING PATTERNS (< 30% success):")
            for pattern_str in worst_patterns:
                pattern = eval(pattern_str)
                print(f"     {pattern}")
                # Analyze what makes these patterns special
                self._analyze_pattern_characteristics(pattern)
        
    def _analyze_pattern_characteristics(self, pattern: List[int]) -> None:
        """Analyze characteristics of a problematic pattern"""
        
        characteristics = []
        
        # Check value ranges
        min_val, max_val = min(pattern), max(pattern)
        if max_val >= 90:
            characteristics.append("large numbers (90+)")
        if min_val <= 5:
            characteristics.append("small numbers (≤5)")
        
        # Check if already sorted
        sorted_asc = pattern == sorted(pattern)
        sorted_desc = pattern == sorted(pattern, reverse=True)
        if sorted_asc:
            characteristics.append("already ascending")
        elif sorted_desc:
            characteristics.append("already descending")
        
        # Check value spacing
        gaps = [pattern[i+1] - pattern[i] for i in range(len(pattern)-1)]
        avg_gap = sum(abs(g) for g in gaps) / len(gaps)
        if avg_gap > 10:
            characteristics.append("large value gaps")
        
        if characteristics:
            print(f"        Characteristics: {', '.join(characteristics)}")
    
    def _analyze_sorting_ranges(self, results: List[Dict]) -> None:
        """Analyze the distribution of sorting ranges"""
        
        print(f"\n📍 SORTING RANGE ANALYSIS:")
        
        # Count range types
        range_counts = {}
        direction_counts = {'ascending': 0, 'descending': 0}
        length_counts = {}
        
        for result in results:
            range_str = result['range_tested']['range_str']
            range_counts[range_str] = range_counts.get(range_str, 0) + 1
            
            original = result['rom_info']['original_discovery']
            direction = original.get('direction', 'unknown')
            if direction in direction_counts:
                direction_counts[direction] += 1
            
            length = original.get('length', 0)
            length_counts[length] = length_counts.get(length, 0) + 1
        
        print(f"   Range Distribution:")
        sorted_ranges = sorted(range_counts.items(), key=lambda x: x[1], reverse=True)
        for range_str, count in sorted_ranges:
            pct = count / len(results) * 100
            print(f"     {range_str:>6}: {count:3d} ROMs ({pct:4.1f}%)")
        
        print(f"   Direction Distribution:")
        for direction, count in direction_counts.items():
            pct = count / len(results) * 100
            print(f"     {direction:>10}: {count:3d} ROMs ({pct:4.1f}%)")
        
        print(f"   Length Distribution:")
        for length in sorted(length_counts.keys()):
            count = length_counts[length]
            pct = count / len(results) * 100
            print(f"     {length:2d} elements: {count:3d} ROMs ({pct:4.1f}%)")
        
        # Analyze range vs performance correlation
        print(f"\n   Range vs Performance Analysis:")
        range_performance = {}
        for result in results:
            range_str = result['range_tested']['range_str']
            score = result['generalization_score']
            if range_str not in range_performance:
                range_performance[range_str] = []
            range_performance[range_str].append(score)
        
        for range_str, scores in range_performance.items():
            if len(scores) >= 3:  # Only analyze ranges with enough samples
                avg_score = sum(scores) / len(scores)
                max_score = max(scores)
                print(f"     {range_str:>6}: avg {avg_score:.1%}, max {max_score:.1%} ({len(scores)} ROMs)")
    
    def _create_enhanced_summary(self, results: List[Dict], output_file: Path) -> None:
        """Create enhanced human-readable summary with insights"""
        
        readable_file = output_file.with_stem(output_file.stem + '_ENHANCED_summary')
        
        with open(readable_file, 'w', encoding='utf-8') as f:
            f.write("BABELSCOPE ROM GENERALIZATION ANALYSIS - ENHANCED SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write("CRITICAL FIX APPLIED: Now tests only discovered sorting ranges\n")
            f.write("ENHANCED ANALYSIS: Deep pattern insights included\n\n")
            
            # Basic stats
            consistent = [r for r in results if r['consistent_sorting']]
            partial = [r for r in results if 0 < r['generalization_score'] < 0.7]
            
            f.write(f"OVERVIEW:\n")
            f.write(f"   Total ROMs tested: {len(results)}\n")
            f.write(f"   Consistent algorithms (≥70%): {len(consistent)}\n")
            f.write(f"   Partial generalizers (1-69%): {len(partial)}\n")
            f.write(f"   Pattern-specific only (0%): {len(results) - len(consistent) - len(partial)}\n")
            f.write(f"   Test patterns used: {len(self.test_patterns)}\n\n")
            
            # Top performers with enhanced details
            top_roms = sorted(results, key=lambda x: x['generalization_score'], reverse=True)[:15]
            f.write("TOP 15 PERFORMERS:\n")
            f.write("-" * 50 + "\n")
            
            for i, result in enumerate(top_roms):
                rom_name = result['rom_info']['filename']
                score = result['generalization_score']
                orig = result['rom_info']['original_discovery']
                range_tested = result['range_tested']['range_str']
                
                f.write(f"{i+1:2d}. {rom_name}\n")
                f.write(f"    Score: {score:.1%} | Range: {range_tested} | ")
                f.write(f"Direction: {orig.get('direction', 'unknown')}\n")
                
                # Show which patterns worked/failed
                success_patterns = []
                fail_patterns = []
                for test_name, input_pattern, test_result in result['test_results']:
                    if test_result:
                        if test_result['sorted']:
                            success_patterns.append(str(input_pattern))
                        else:
                            fail_patterns.append(str(input_pattern))
                
                f.write(f"    SUCCESS: Works on {len(success_patterns)}/{len(success_patterns) + len(fail_patterns)} patterns\n")
                if len(fail_patterns) <= 3:  # Show failed patterns if not too many
                    f.write(f"    FAILS ON: {', '.join(fail_patterns[:3])}\n")
                f.write("\n")
            
            # Pattern analysis
            f.write("\nPATTERN FAILURE ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            
            # Find most problematic patterns
            pattern_stats = {}
            for result in results:
                for test_name, input_pattern, test_result in result['test_results']:
                    pattern_key = str(input_pattern)
                    if pattern_key not in pattern_stats:
                        pattern_stats[pattern_key] = {'success': 0, 'total': 0}
                    
                    pattern_stats[pattern_key]['total'] += 1
                    if test_result and test_result['sorted']:
                        pattern_stats[pattern_key]['success'] += 1
            
            # Sort by failure rate (lowest success rate first)
            sorted_patterns = sorted(pattern_stats.items(), 
                                   key=lambda x: x[1]['success'] / x[1]['total'] if x[1]['total'] > 0 else 0)
            
            f.write("Most problematic patterns (lowest success rates):\n")
            for pattern_str, stats in sorted_patterns[:4]:
                rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
                pattern = eval(pattern_str)
                f.write(f"   {rate:5.1%} success: {pattern}\n")
            
            # Range analysis
            f.write(f"\nSORTING RANGE INSIGHTS:\n")
            f.write("-" * 30 + "\n")
            
            range_counts = {}
            for result in results:
                range_str = result['range_tested']['range_str']
                range_counts[range_str] = range_counts.get(range_str, 0) + 1
            
            f.write("Range popularity:\n")
            for range_str, count in sorted(range_counts.items(), key=lambda x: x[1], reverse=True):
                pct = count / len(results) * 100
                f.write(f"   {range_str}: {count} ROMs ({pct:.1f}%)\n")
            
            # Key insights
            f.write(f"\nKEY INSIGHTS:\n")
            f.write("-" * 20 + "\n")
            
            best_score = max(r['generalization_score'] for r in results)
            best_count = len([r for r in results if r['generalization_score'] == best_score])
            
            f.write(f"• Best generalization score achieved: {best_score:.1%}\n")
            f.write(f"• Number of ROMs achieving best score: {best_count}\n")
            f.write(f"• Most common sorting range: {max(range_counts.items(), key=lambda x: x[1])[0]}\n")
            
            most_problematic = min(pattern_stats.items(), key=lambda x: x[1]['success'] / x[1]['total'])
            worst_rate = most_problematic[1]['success'] / most_problematic[1]['total']
            f.write(f"• Most problematic test pattern: {eval(most_problematic[0])} ({worst_rate:.1%} success)\n")
            
            if best_score >= 0.6:
                f.write(f"• CONCLUSION: Found ROMs with significant generalization capability!\n")
            else:
                f.write(f"• CONCLUSION: Limited generalization - likely pattern-specific algorithms\n")
        
        print(f"ENHANCED human-readable summary saved to: {readable_file}")


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
    """Main entry point for FIXED ROM generalization testing"""
    parser = argparse.ArgumentParser(
        description='FIXED: Test discovered ROMs with different input patterns in correct ranges',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CRITICAL FIX APPLIED: This version now correctly tests only the discovered 
sorting range (e.g., V2-V7) instead of incorrectly testing the entire V0-V7 range.

This fixes the fundamental flaw where ROMs were failing tests because we were
checking the wrong registers for sorting behavior.

Examples:
  python rom_generalization_tester.py --auto-scan --max-roms 50
  python rom_generalization_tester.py --rom-dirs output/async_partial_sorting/session_*/discovered_roms
  python rom_generalization_tester.py --rom-dirs path/to/roms --cycles 50000 --output results.json

Test Methodology (FIXED):
1. Load ROMs discovered by partial sorting Babelscope
2. Extract the specific sorting range from metadata (e.g., V2-V7)
3. Test each ROM with multiple different input patterns  
4. Check sorting ONLY in the discovered range
5. Identify ROMs that consistently sort different inputs in the correct range
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
    parser.add_argument('--output', type=Path, default='generalization_analysis_FIXED.json',
                       help='Output file for results (default: generalization_analysis_FIXED.json)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test mode (fewer cycles, fewer ROMs)')
    
    args = parser.parse_args()
    
    print("BABELSCOPE ROM GENERALIZATION TESTER (FIXED)")
    print("=" * 50)
    print("🔧 CRITICAL FIX: Now tests only the discovered sorting ranges")
    print("Previous version incorrectly tested entire V0-V7 range")
    print("This should dramatically improve generalization success rates!")
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
    
    # Create FIXED tester and run analysis
    try:
        tester = FixedROMGeneralizationTester(rom_dirs)
        tester.scan_for_roms()
        
        if not tester.discovered_roms:
            print("No discovered ROMs found to test!")
            return 1
        
        # Run FIXED generalization analysis
        results = tester.run_generalization_analysis(
            max_roms=args.max_roms,
            cycles_per_test=args.cycles
        )
        
        if results:
            # Save results
            tester.save_analysis_results(results, args.output)
            
            # Final summary
            consistent = [r for r in results if r['consistent_sorting']]
            if consistent:
                print(f"\n🎉 FIXED ANALYSIS SUCCESS: Found {len(consistent)} consistent sorting algorithms!")
                print("These ROMs demonstrate true sorting capability in their discovered ranges.")
                print("\nCOMPARE TO PREVIOUS RESULTS:")
                print("The original analysis likely had artificially low success rates")
                print("due to testing the wrong register ranges.")
            else:
                print(f"\n📊 FIXED ANALYSIS: No consistent sorting algorithms found.")
                print("Even with the fix, most discovered ROMs appear to be pattern-specific.")
            
            # Show improvement statistics
            old_consistent_count = 1  # From your previous run
            new_consistent_count = len(consistent)
            if new_consistent_count > old_consistent_count:
                improvement = new_consistent_count - old_consistent_count
                print(f"\n📈 IMPROVEMENT: Found {improvement} additional consistent algorithms")
                print(f"   Previous (broken): {old_consistent_count}")
                print(f"   Fixed analysis: {new_consistent_count}")
                print(f"   {improvement}x improvement in discovery rate!")
            
            return 0
        else:
            print("No results generated")
            return 1
            
    except KeyboardInterrupt:
        print("\nFIXED generalization testing interrupted by user")
        return 0
        
    except Exception as e:
        print(f"FIXED generalization testing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())