#!/usr/bin/env python3
"""
Babelscope ROM Generalization Tester
Tests discovered ROMs with different input patterns to identify true sorting algorithms

This script:
1. Loads ROMs discovered by the partial sorting Babelscope
2. Tests each ROM with multiple different input patterns
3. Compares the sorting behavior across different inputs
4. Identifies ROMs that demonstrate true sorting capability vs pattern-specific behavior
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


class ROMGeneralizationTester:
    """Tests discovered ROMs with different input patterns to validate sorting algorithms"""
    
    def __init__(self, rom_directories: List[Path]):
        self.rom_directories = rom_directories
        self.discovered_roms = []
        self.test_patterns = self._generate_test_patterns()
        self.original_pattern = [8, 3, 6, 1, 7, 2, 5, 4]  # The pattern used in discovery
        
        print(f"ROM Generalization Tester initialized")
        print(f"Scanning ROM directories: {len(rom_directories)}")
        print(f"Test patterns: {len(self.test_patterns)}")
        
    def _generate_test_patterns(self) -> List[List[int]]:
        """Generate diverse test patterns to validate sorting algorithms"""
        patterns = []
        
        # Pattern 1: Your suggested 22-29 range
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
                        
                        rom_info = {
                            'filename': ch8_file.name,
                            'filepath': ch8_file,
                            'metadata': metadata,
                            'rom_data': rom_data,
                            'original_discovery': metadata.get('partial_sorting', {}),
                            'original_registers': metadata.get('registers', {})
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
            for rom in self.discovered_roms:
                length = rom['original_discovery'].get('length', 0)
                length_counts[length] = length_counts.get(length, 0) + 1
            
            for length in sorted(length_counts.keys(), reverse=True):
                count = length_counts[length]
                print(f"   {length}-element sequences: {count} ROMs")
    
    def test_rom_with_pattern(self, rom_data: bytes, test_pattern: List[int], 
                             cycles: int = 100000, check_interval: int = 100) -> Optional[Dict]:
        """Test a single ROM with a specific input pattern"""
        
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
            
            # Get final register state regardless of whether sorting was detected
            final_registers = cp.asnumpy(detector.registers[0, :8]).tolist()
            
            if discoveries > 0:
                discovery_list = detector.get_partial_sorting_discoveries()
                if discovery_list:
                    discovery = discovery_list[0]
                    return {
                        'sorted': True,
                        'input_pattern': test_pattern[:8],
                        'output_pattern': discovery['final_registers'],
                        'partial_sorting': discovery['partial_sorting'],
                        'sort_cycle': discovery['sort_cycle'],
                        'register_activity': discovery['register_activity']
                    }
            
            # No sorting detected by the algorithm, but check manually for any obvious patterns
            manual_check = self._manual_sorting_check(test_pattern[:8], final_registers)
            
            return {
                'sorted': False,
                'input_pattern': test_pattern[:8],
                'output_pattern': final_registers,
                'partial_sorting': manual_check,
                'sort_cycle': None,
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
    
    def _manual_sorting_check(self, input_pattern: List[int], output_pattern: List[int]) -> Optional[Dict]:
        """Manually check for sorting patterns that the detector might have missed"""
        
        # Check for consecutive sorted sequences of length 3+
        for direction in ['ascending', 'descending']:
            for start in range(len(output_pattern) - 2):  # Need at least 3 elements
                max_length = 1
                
                for end in range(start + 1, len(output_pattern)):
                    if direction == 'ascending':
                        if output_pattern[end] == output_pattern[end-1] + 1:
                            max_length += 1
                        else:
                            break
                    else:  # descending
                        if output_pattern[end] == output_pattern[end-1] - 1:
                            max_length += 1
                        else:
                            break
                
                if max_length >= 3:  # Found a sequence
                    sequence = output_pattern[start:start + max_length]
                    return {
                        'length': max_length,
                        'start_position': start,
                        'end_position': start + max_length - 1,
                        'direction': direction,
                        'sequence': sequence,
                        'sequence_range': f"V{start}-V{start + max_length - 1}",
                        'detection_method': 'manual_post_analysis'
                    }
        
        return None
    
    def analyze_rom_generalization(self, rom_info: Dict) -> Dict:
        """Test a ROM with all test patterns and analyze its generalization ability"""
        
        filename = rom_info['filename']
        original_discovery = rom_info['original_discovery']
        rom_data = rom_info['rom_data']
        
        print(f"Testing ROM: {filename}")
        print(f"   Original discovery: {original_discovery.get('sequence_range', 'unknown')} "
              f"({original_discovery.get('length', 0)} elements, "
              f"{original_discovery.get('direction', 'unknown')})")
        
        results = {
            'rom_info': rom_info,
            'test_results': [],
            'generalization_score': 0,
            'consistent_sorting': False,
            'pattern_specific': False
        }
        
        # Test with original pattern first (sanity check) - use longer cycles for better reproduction
        print(f"   Testing with original pattern: {self.original_pattern}")
        original_result = self.test_rom_with_pattern(
            rom_data, 
            self.original_pattern, 
            cycles=150000,  # More cycles for original test
            check_interval=50  # More frequent checks
        )
        
        if original_result and (original_result['sorted'] or original_result['partial_sorting']):
            if original_result['sorted']:
                print(f"      ✓ Sorted: {original_result['partial_sorting']['sequence_range']} "
                      f"({original_result['partial_sorting']['length']} elements)")
            else:
                print(f"      ~ Manual detection: {original_result['partial_sorting']['sequence_range']} "
                      f"({original_result['partial_sorting']['length']} elements)")
        else:
            print(f"      ✗ No sorting detected")
            if original_result:
                print(f"         Input: {original_result['input_pattern']}")
                print(f"         Output: {original_result['output_pattern']}")
                activity = original_result.get('register_activity', {})
                if activity:
                    print(f"         Register activity: {activity.get('register_writes', 0)} writes")
        
        results['test_results'].append(('original', self.original_pattern, original_result))
        
        # Test with all other patterns
        sorting_count = 0
        total_tests = len(self.test_patterns)
        
        for i, test_pattern in enumerate(self.test_patterns):
            print(f"   Testing pattern {i+1}/{total_tests}: {test_pattern}")
            
            result = self.test_rom_with_pattern(rom_data, test_pattern)
            results['test_results'].append((f'pattern_{i+1}', test_pattern, result))
            
            if result and (result['sorted'] or result['partial_sorting']):
                sorting_count += 1
                if result['sorted']:
                    print(f"      ✓ Sorted: {result['partial_sorting']['sequence_range']} "
                          f"({result['partial_sorting']['length']} elements)")
                else:
                    print(f"      ~ Manual: {result['partial_sorting']['sequence_range']} "
                          f"({result['partial_sorting']['length']} elements)")
            else:
                print(f"      ✗ No sorting detected")
        
        # Calculate generalization metrics
        results['generalization_score'] = sorting_count / total_tests
        results['consistent_sorting'] = sorting_count >= (total_tests * 0.7)  # 70% success rate
        results['pattern_specific'] = sorting_count == 0  # Only works on original
        
        print(f"   Generalization score: {results['generalization_score']:.2%} "
              f"({sorting_count}/{total_tests} patterns sorted)")
        
        if results['consistent_sorting']:
            print(f"   ✓ CONSISTENT SORTING ALGORITHM DETECTED")
        elif results['pattern_specific']:
            print(f"   ✗ PATTERN-SPECIFIC BEHAVIOR (likely hardcoded)")
        else:
            print(f"   ~ PARTIAL GENERALIZATION (some patterns work)")
        
        print()
        
        return results
    
    def run_generalization_analysis(self, max_roms: Optional[int] = None, 
                                   cycles_per_test: int = 100000) -> List[Dict]:
        """Run generalization analysis on all discovered ROMs"""
        
        if not self.discovered_roms:
            self.scan_for_roms()
        
        if not self.discovered_roms:
            print("No ROMs found to test!")
            return []
        
        roms_to_test = self.discovered_roms[:max_roms] if max_roms else self.discovered_roms
        
        print(f"\nSTARTING GENERALIZATION ANALYSIS")
        print(f"=" * 60)
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
        print(f"GENERALIZATION ANALYSIS COMPLETE")
        print(f"=" * 60)
        print(f"Total time: {total_time:.1f} seconds")
        print(f"ROMs tested: {len(all_results)}")
        
        # Summarize results
        consistent_algorithms = [r for r in all_results if r['consistent_sorting']]
        partial_generalizers = [r for r in all_results if 0 < r['generalization_score'] < 0.7]
        pattern_specific = [r for r in all_results if r['pattern_specific']]
        
        print(f"\nRESULTS SUMMARY:")
        print(f"   Consistent sorting algorithms: {len(consistent_algorithms)}")
        print(f"   Partial generalizers: {len(partial_generalizers)}")
        print(f"   Pattern-specific only: {len(pattern_specific)}")
        
        if consistent_algorithms:
            print(f"\n🎉 DISCOVERED SORTING ALGORITHMS:")
            for result in consistent_algorithms:
                rom_name = result['rom_info']['filename']
                score = result['generalization_score']
                original = result['rom_info']['original_discovery']
                print(f"   {rom_name}: {score:.1%} success rate")
                print(f"      Original: {original.get('sequence_range', 'unknown')} "
                      f"({original.get('length', 0)} elements)")
        
        return all_results
    
    def save_analysis_results(self, results: List[Dict], output_file: Path) -> None:
        """Save generalization analysis results to file"""
        
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
                        'sorted': test_result['sorted'] or (test_result['partial_sorting'] is not None)
                    }
                    
                    # If sorting was detected, include the sorting info
                    if test_result['sorted'] and test_result['partial_sorting']:
                        ps = test_result['partial_sorting']
                        compressed_test['sort'] = {
                            'len': ps['length'],
                            'dir': ps['direction'][:3],
                            'pos': ps['sequence_range']
                        }
                    elif test_result['partial_sorting']:  # Manual detection
                        ps = test_result['partial_sorting']
                        compressed_test['sort'] = {
                            'len': ps['length'],
                            'dir': ps['direction'][:3],
                            'pos': ps['sequence_range'],
                            'manual': True
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
                'orig_pattern': self.original_pattern
            },
            'results': compressed_results
        }
        
        # Save compressed version
        compressed_file = output_file.with_stem(output_file.stem + '_compressed')
        with open(compressed_file, 'w') as f:
            json.dump(summary, f, separators=(',', ':'))  # No whitespace
        
        print(f"Compressed analysis results saved to: {compressed_file}")
        print(f"File size: {compressed_file.stat().st_size} bytes")
        
        # Also save a human-readable summary
        readable_file = output_file.with_stem(output_file.stem + '_summary')
        with open(readable_file, 'w') as f:
            f.write("BABELSCOPE ROM GENERALIZATION ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total ROMs tested: {len(results)}\n")
            f.write(f"Consistent algorithms: {len([r for r in results if r['consistent_sorting']])}\n")
            f.write(f"Partial generalizers: {len([r for r in results if 0 < r['generalization_score'] < 0.7])}\n")
            f.write(f"Test patterns: {len(self.test_patterns)}\n\n")
            
            # List top performers
            top_roms = sorted(results, key=lambda x: x['generalization_score'], reverse=True)[:10]
            f.write("TOP 10 PERFORMERS:\n")
            for i, result in enumerate(top_roms):
                rom_name = result['rom_info']['filename']
                score = result['generalization_score']
                orig = result['rom_info']['original_discovery']
                f.write(f"{i+1:2d}. {rom_name}\n")
                f.write(f"    Score: {score:.1%}, Original: {orig.get('sequence_range', 'unknown')} "
                        f"({orig.get('length', 0)} elements, {orig.get('direction', 'unknown')})\n")
            
            # Show which patterns work best
            f.write(f"\nPATTERN SUCCESS RATES:\n")
            pattern_success = {}
            for result in results:
                for test_name, input_pattern, test_result in result['test_results']:
                    pattern_key = str(input_pattern)
                    if pattern_key not in pattern_success:
                        pattern_success[pattern_key] = {'success': 0, 'total': 0}
                    pattern_success[pattern_key]['total'] += 1
                    if test_result and (test_result['sorted'] or test_result['partial_sorting']):
                        pattern_success[pattern_key]['success'] += 1
            
            for pattern, stats in pattern_success.items():
                rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
                f.write(f"  {pattern}: {rate:.1%} ({stats['success']}/{stats['total']})\n")
        
        print(f"Human-readable summary saved to: {readable_file}")
        
        # Original detailed file (optional)
        if len(results) <= 50:  # Only save detailed for small datasets
            # Prepare data for JSON serialization (detailed version)
            serializable_results = []
            
            for result in results:
                # Convert bytes to hex string for ROM data and fix Path objects
                rom_info = result['rom_info'].copy()
                rom_info['rom_data'] = rom_info['rom_data'].hex()
                # Convert WindowsPath to string
                if 'filepath' in rom_info:
                    rom_info['filepath'] = str(rom_info['filepath'])
                
                serializable_result = {
                    'rom_info': rom_info,
                    'test_results': result['test_results'],
                    'generalization_score': result['generalization_score'],
                    'consistent_sorting': result['consistent_sorting'],
                    'pattern_specific': result['pattern_specific']
                }
                
                serializable_results.append(serializable_result)
            
            # Create detailed summary
            detailed_summary = {
                'analysis_info': {
                    'timestamp': time.time(),
                    'total_roms_tested': len(results),
                    'test_patterns': self.test_patterns,
                    'original_pattern': self.original_pattern,
                    'consistent_algorithms': len([r for r in results if r['consistent_sorting']]),
                    'partial_generalizers': len([r for r in results if 0 < r['generalization_score'] < 0.7]),
                    'pattern_specific': len([r for r in results if r['pattern_specific']])
                },
                'results': serializable_results
            }
            
            # Save detailed file
            with open(output_file, 'w') as f:
                json.dump(detailed_summary, f, indent=2)
            
            print(f"Detailed analysis results saved to: {output_file}")
        else:
            print(f"Skipped detailed output (too many ROMs - use compressed version)")
        
        print(f"\nFOR SHARING/ANALYSIS: Use {compressed_file.name}")
        print(f"Size: {compressed_file.stat().st_size} bytes")


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
    """Main entry point for ROM generalization testing"""
    parser = argparse.ArgumentParser(
        description='Test discovered ROMs with different input patterns to identify true sorting algorithms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script tests your discovered ROMs from the Babelscope partial sorting runs
with different input patterns to determine which ones are true sorting algorithms
versus pattern-specific rearrangements.

Examples:
  python rom_generalization_tester.py --auto-scan --max-roms 50
  python rom_generalization_tester.py --rom-dirs output/async_partial_sorting/session_*/discovered_roms
  python rom_generalization_tester.py --rom-dirs path/to/roms --cycles 50000 --output results.json

Test Methodology:
1. Load ROMs discovered by partial sorting Babelscope
2. Test each ROM with multiple different input patterns  
3. Compare sorting behavior across patterns
4. Identify ROMs that consistently sort different inputs
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
    parser.add_argument('--output', type=Path, default='generalization_analysis.json',
                       help='Output file for results (default: generalization_analysis.json)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test mode (fewer cycles, fewer ROMs)')
    
    args = parser.parse_args()
    
    print("BABELSCOPE ROM GENERALIZATION TESTER")
    print("=" * 50)
    print("Testing discovered ROMs with different input patterns")
    print("to identify true sorting algorithms vs pattern-specific behavior")
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
    
    # Create tester and run analysis
    try:
        tester = ROMGeneralizationTester(rom_dirs)
        tester.scan_for_roms()
        
        if not tester.discovered_roms:
            print("No discovered ROMs found to test!")
            return 1
        
        # Run generalization analysis
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
                print(f"\n🎉 SUCCESS: Found {len(consistent)} consistent sorting algorithms!")
                print("These ROMs demonstrate true sorting capability across different inputs.")
            else:
                print(f"\n📊 No consistent sorting algorithms found.")
                print("Most discovered ROMs appear to be pattern-specific rearrangements.")
            
            return 0
        else:
            print("No results generated")
            return 1
            
    except KeyboardInterrupt:
        print("\nGeneralization testing interrupted by user")
        return 0
        
    except Exception as e:
        print(f"Generalization testing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())