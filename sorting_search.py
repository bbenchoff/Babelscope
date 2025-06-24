#!/usr/bin/env python3
"""
Babelscope Sorting Algorithm Search Runner
Top-level script to search for emergent sorting algorithms in random CHIP-8 code

Saves all output to output/sorting/ directory with comprehensive debug logs
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import argparse

# Add emulators and generators directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'emulators'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'generators'))

try:
    from parallel_chip8_sorting import CUDASortingDetector
    print("✅ Successfully imported CUDA Sorting Detector")
except ImportError as e:
    print(f"❌ Failed to import sorting detector: {e}")
    print("Make sure emulators/parallel_chip8_sorting.py exists")
    sys.exit(1)

try:
    from cuda_rom_generator import generate_random_roms_cuda
    print("✅ Successfully imported CUDA ROM Generator")
    USE_CUDA_GENERATOR = True
except ImportError as e:
    print(f"⚠️  CUDA ROM Generator not available: {e}")
    print("Falling back to CPU generation")
    USE_CUDA_GENERATOR = False

class SortingSearchRunner:
    """
    Top-level runner for Babelscope sorting algorithm search
    Manages output directories, logging, and comprehensive result tracking
    """
    
    def __init__(self, output_base_dir: str = "output/sorting"):
        self.output_base_dir = Path(output_base_dir)
        self.session_start = datetime.now()
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
        
        # Create output directory structure
        self.output_dir = self.output_base_dir / f"session_{self.session_id}"
        self.roms_dir = self.output_dir / "roms"
        self.logs_dir = self.output_dir / "logs"
        
        for dir_path in [self.output_dir, self.roms_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize log files
        self.debug_log_path = self.logs_dir / "debug.txt"
        self.summary_log_path = self.logs_dir / "summary.json"
        self.discoveries_log_path = self.logs_dir / "discoveries.json"
        
        # Session statistics
        self.session_stats = {
            'session_id': self.session_id,
            'start_time': self.session_start.isoformat(),
            'total_roms_tested': 0,
            'total_batches': 0,
            'total_sorts_found': 0,
            'total_execution_time': 0.0,
            'discoveries': [],
            'batch_history': []
        }
        
        self._init_debug_log()
        print(f"🔢 Sorting Search Session: {self.session_id}")
        print(f"📁 Output directory: {self.output_dir}")
    
    def _init_debug_log(self):
        """Initialize debug log file with session header"""
        with open(self.debug_log_path, 'w', encoding='utf-8') as f:
            f.write("BABELSCOPE SORTING ALGORITHM SEARCH - DEBUG LOG\n")
            f.write("=" * 60 + "\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Start Time: {self.session_start}\n")
            f.write(f"Output Directory: {self.output_dir}\n")
            f.write("=" * 60 + "\n\n")
    
    def _log_debug(self, message: str, also_print: bool = True):
        """Write message to debug log and optionally print"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        with open(self.debug_log_path, 'a', encoding='utf-8') as f:
            f.write(log_message + "\n")
        
        if also_print:
            print(log_message)
    
    def _save_session_summary(self):
        """Save comprehensive session summary"""
        # Update final stats
        self.session_stats['end_time'] = datetime.now().isoformat()
        self.session_stats['total_execution_time'] = time.time() - time.mktime(self.session_start.timetuple())
        
        # Calculate rates
        if self.session_stats['total_execution_time'] > 0:
            self.session_stats['roms_per_second'] = self.session_stats['total_roms_tested'] / self.session_stats['total_execution_time']
            self.session_stats['batches_per_hour'] = self.session_stats['total_batches'] / (self.session_stats['total_execution_time'] / 3600)
        
        # Discovery rate
        if self.session_stats['total_sorts_found'] > 0:
            self.session_stats['discovery_rate'] = self.session_stats['total_roms_tested'] / self.session_stats['total_sorts_found']
        else:
            self.session_stats['discovery_rate'] = None
        
        # Save summary
        with open(self.summary_log_path, 'w', encoding='utf-8') as f:
            json.dump(self.session_stats, f, indent=2)
        
        # Save discoveries separately
        with open(self.discoveries_log_path, 'w', encoding='utf-8') as f:
            json.dump(self.session_stats['discoveries'], f, indent=2)
    
    def _process_batch_results(self, batch_num: int, batch_results: Dict, 
                              execution_time: float, detector: CUDASortingDetector) -> int:
        """Process and log results from a batch"""
        sorts_found = batch_results['sorts_found']
        arrays_accessed = batch_results['arrays_accessed']
        
        # Log batch summary
        self._log_debug(f"Batch {batch_num} Results:")
        self._log_debug(f"  Execution time: {execution_time:.2f}s")
        self._log_debug(f"  Arrays accessed: {arrays_accessed}")
        self._log_debug(f"  Sorting algorithms found: {sorts_found}")
        self._log_debug(f"  Total array reads: {batch_results['total_array_reads']}")
        self._log_debug(f"  Total array writes: {batch_results['total_array_writes']}")
        self._log_debug(f"  Total comparisons: {batch_results['total_comparisons']}")
        self._log_debug(f"  Total swaps: {batch_results['total_swaps']}")
        
        # Process discoveries
        discoveries_saved = 0
        if sorts_found > 0:
            self._log_debug(f"🎯 SORTING ALGORITHMS DISCOVERED:")
            
            for i, discovery in enumerate(batch_results['discoveries']):
                instance_id = discovery['instance_id']
                direction = discovery['sort_direction']
                cycle = discovery['sort_cycle']
                final_array = discovery['final_array']
                
                self._log_debug(f"  {i+1}. Instance {instance_id}: {direction.upper()} sort")
                self._log_debug(f"     Sorted at cycle: {cycle:,}")
                self._log_debug(f"     Array reads: {discovery['array_reads']}")
                self._log_debug(f"     Array writes: {discovery['array_writes']}")
                self._log_debug(f"     Comparisons: {discovery['comparisons']}")
                self._log_debug(f"     Swaps: {discovery['swaps']}")
                self._log_debug(f"     Final array: {final_array}")
                
                # Save ROM and analysis
                rom_filename = self._save_discovery_rom(
                    detector, instance_id, discovery, batch_num, i+1
                )
                
                # Add to session discoveries
                discovery_record = {
                    'batch': batch_num,
                    'discovery_number': i+1,
                    'instance_id': instance_id,
                    'direction': direction,
                    'cycle': cycle,
                    'array_reads': discovery['array_reads'],
                    'array_writes': discovery['array_writes'],
                    'comparisons': discovery['comparisons'],
                    'swaps': discovery['swaps'],
                    'final_array': final_array,
                    'rom_filename': rom_filename,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.session_stats['discoveries'].append(discovery_record)
                discoveries_saved += 1
        
        # Record batch in history
        batch_record = {
            'batch_number': batch_num,
            'execution_time': execution_time,
            'arrays_accessed': arrays_accessed,
            'sorts_found': sorts_found,
            'total_array_reads': batch_results['total_array_reads'],
            'total_array_writes': batch_results['total_array_writes'],
            'total_comparisons': batch_results['total_comparisons'],
            'total_swaps': batch_results['total_swaps'],
            'timestamp': datetime.now().isoformat()
        }
        
        self.session_stats['batch_history'].append(batch_record)
        self._log_debug("")  # Empty line for readability
        
        return discoveries_saved
    
    def _save_discovery_rom(self, detector: CUDASortingDetector, instance_id: int, 
                           discovery: Dict, batch_num: int, discovery_num: int) -> str:
        """Save discovered sorting ROM with comprehensive analysis"""
        import cupy as cp
        import hashlib
        
        # Extract ROM data
        rom_data = cp.asnumpy(detector.memory[instance_id, 0x200:])
        
        # Find actual ROM end (first long stretch of zeros)
        rom_end = len(rom_data)
        zero_count = 0
        for i in range(len(rom_data)):
            if rom_data[i] == 0:
                zero_count += 1
                if zero_count > 64:
                    rom_end = max(100, i - 63)  # Ensure minimum ROM size
                    break
            else:
                zero_count = 0
        
        rom_data = rom_data[:rom_end]
        rom_hash = hashlib.sha256(rom_data.tobytes()).hexdigest()[:12]
        
        # Create filenames
        direction = discovery['sort_direction']
        cycle = discovery['sort_cycle']
        
        base_filename = f"SORT_{direction.upper()}_B{batch_num:03d}D{discovery_num:02d}_I{instance_id:05d}_C{cycle}_{rom_hash}"
        rom_filename = f"{base_filename}.ch8"
        analysis_filename = f"{base_filename}_ANALYSIS.txt"
        
        rom_path = self.roms_dir / rom_filename
        analysis_path = self.roms_dir / analysis_filename
        
        # Save ROM binary
        with open(rom_path, 'wb') as f:
            f.write(rom_data.tobytes())
        
        # Save comprehensive analysis
        with open(analysis_path, 'w', encoding='utf-8') as f:
            f.write("BABELSCOPE SORTING ALGORITHM DISCOVERY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("DISCOVERY INFORMATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Batch Number: {batch_num}\n")
            f.write(f"Discovery Number: {discovery_num}\n")
            f.write(f"Instance ID: {instance_id}\n")
            f.write(f"Discovery Time: {datetime.now()}\n\n")
            
            f.write("SORTING ALGORITHM DETAILS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Sort Direction: {direction.upper()}\n")
            f.write(f"Sorting Completed at Cycle: {cycle:,}\n")
            f.write(f"Final Sorted Array: {discovery['final_array']}\n")
            f.write(f"Array Location: 0x300-0x307 (768-775 decimal)\n\n")
            
            f.write("ALGORITHM STATISTICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Array Reads: {discovery['array_reads']}\n")
            f.write(f"Array Writes: {discovery['array_writes']}\n")
            f.write(f"Comparison Operations: {discovery['comparisons']}\n")
            f.write(f"Swap Operations: {discovery['swaps']}\n")
            f.write(f"Total Memory Operations: {discovery['array_reads'] + discovery['array_writes']}\n\n")
            
            f.write("ROM INFORMATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"ROM Size: {len(rom_data)} bytes\n")
            f.write(f"ROM Hash: {rom_hash}\n")
            f.write(f"ROM Filename: {rom_filename}\n\n")
            
            f.write("ALGORITHM CLASSIFICATION:\n")
            f.write("-" * 30 + "\n")
            
            # Classify the sorting algorithm based on operations
            reads = discovery['array_reads']
            writes = discovery['array_writes']
            comps = discovery['comparisons']
            swaps = discovery['swaps']
            
            if reads <= 16 and writes <= 16 and comps <= 20:
                algorithm_type = "Efficient sorting (possible quicksort or optimized algorithm)"
            elif reads > 50 or writes > 50:
                algorithm_type = "Brute-force or bubble-sort style algorithm"
            elif swaps > comps:
                algorithm_type = "Swap-heavy algorithm (possible selection sort)"
            elif comps > swaps * 3:
                algorithm_type = "Comparison-heavy algorithm (possible insertion sort)"
            else:
                algorithm_type = "Unknown sorting methodology"
            
            f.write(f"Likely Algorithm Type: {algorithm_type}\n")
            f.write(f"Efficiency Rating: ")
            if reads + writes < 30:
                f.write("HIGH (efficient sorting)\n")
            elif reads + writes < 60:
                f.write("MEDIUM (moderately efficient)\n")
            else:
                f.write("LOW (brute-force approach)\n")
        
        self._log_debug(f"💾 Saved: {rom_filename}")
        return rom_filename
    
    def run_search(self, batch_size: int = 20000, cycles: int = 50000, 
                   max_batches: int = None, continuous: bool = False,
                   check_interval: int = 500):
        """Run the sorting algorithm search"""
        
        self._log_debug(f"🔢 Starting Sorting Algorithm Search")
        self._log_debug(f"Batch size: {batch_size:,} ROMs")
        self._log_debug(f"Cycles per ROM: {cycles:,}")
        self._log_debug(f"Check interval: {check_interval}")
        self._log_debug(f"Mode: {'Continuous' if continuous else f'{max_batches} batches'}")
        self._log_debug("")
        
        # Initialize detector
        try:
            detector = CUDASortingDetector(batch_size)
        except Exception as e:
            self._log_debug(f"❌ Failed to initialize detector: {e}")
            return False
        
        batch_count = 0
        search_start_time = time.time()
        
        try:
            while True:
                # Check stopping conditions
                if not continuous and max_batches and batch_count >= max_batches:
                    self._log_debug(f"🛑 Reached maximum batches ({max_batches})")
                    break
                
                batch_count += 1
                self._log_debug(f"🔢 Starting Batch {batch_count}")
                
                # Generate ROMs using CUDA generator
                batch_start = time.time()
                try:
                    if USE_CUDA_GENERATOR:
                        rom_data_list = generate_random_roms_cuda(batch_size)
                        self._log_debug(f"Generated {len(rom_data_list):,} random ROMs on GPU")
                    else:
                        # Fallback to simple CPU generation
                        self._log_debug(f"Generating {batch_size:,} ROMs on CPU...")
                        rom_data_list = []
                        for i in range(batch_size):
                            rom_data = np.random.randint(0, 256, size=3584, dtype=np.uint8)
                            rom_data_list.append(rom_data)
                        self._log_debug(f"Generated {len(rom_data_list):,} random ROMs on CPU")
                except Exception as e:
                    self._log_debug(f"❌ ROM generation failed: {e}")
                    continue
                
                # Load ROMs with sorting arrays
                try:
                    detector.load_roms_with_sort_arrays(rom_data_list)
                    self._log_debug(f"Loaded ROMs with pre-seeded sort arrays")
                except Exception as e:
                    self._log_debug(f"❌ ROM loading failed: {e}")
                    continue
                
                # Run sorting detection
                try:
                    sorts_found = detector.run_sorting_detection(
                        cycles=cycles,
                        sort_check_interval=check_interval
                    )
                except Exception as e:
                    self._log_debug(f"❌ Detection failed: {e}")
                    continue
                
                batch_time = time.time() - batch_start
                
                # Get comprehensive results
                try:
                    # Try validated results first, fall back to regular results
                    if hasattr(detector, 'get_validated_results'):
                        batch_results = detector.get_validated_results()
                    elif hasattr(detector, 'validate_sorting_results'):
                        batch_results = detector.validate_sorting_results()
                    else:
                        # Fallback to regular results
                        batch_results = detector.get_sorting_results()
                        self._log_debug("⚠️  Using unvalidated results - false positives possible")
                except Exception as e:
                    self._log_debug(f"❌ Failed to get results: {e}")
                    continue
                
                # Process and save results
                discoveries_saved = self._process_batch_results(
                    batch_count, batch_results, batch_time, detector
                )
                
                # Update session statistics
                self.session_stats['total_roms_tested'] += batch_size
                self.session_stats['total_batches'] = batch_count
                self.session_stats['total_sorts_found'] += sorts_found
                
                # Progress report
                total_time = time.time() - search_start_time
                rate = self.session_stats['total_roms_tested'] / total_time if total_time > 0 else 0
                
                self._log_debug(f"📊 Session Progress:")
                self._log_debug(f"  Total ROMs tested: {self.session_stats['total_roms_tested']:,}")
                self._log_debug(f"  Total batches: {batch_count}")
                self._log_debug(f"  Total sorts found: {self.session_stats['total_sorts_found']}")
                self._log_debug(f"  Processing rate: {rate:.0f} ROMs/sec")
                self._log_debug(f"  Session time: {total_time/3600:.2f} hours")
                
                if self.session_stats['total_sorts_found'] > 0:
                    discovery_rate = self.session_stats['total_roms_tested'] // self.session_stats['total_sorts_found']
                    self._log_debug(f"  Discovery rate: 1 in {discovery_rate:,} ROMs")
                
                self._log_debug("=" * 60)
                
                # Save progress
                self._save_session_summary()
        
        except KeyboardInterrupt:
            self._log_debug(f"🛑 Search interrupted by user at batch {batch_count}")
        except Exception as e:
            self._log_debug(f"❌ Unexpected error: {e}")
        
        # Final summary
        total_time = time.time() - search_start_time
        final_rate = self.session_stats['total_roms_tested'] / total_time if total_time > 0 else 0
        
        self._log_debug("")
        self._log_debug("🏁 SORTING ALGORITHM SEARCH COMPLETE")
        self._log_debug("=" * 60)
        self._log_debug(f"Session ID: {self.session_id}")
        self._log_debug(f"Total batches: {batch_count}")
        self._log_debug(f"Total ROMs tested: {self.session_stats['total_roms_tested']:,}")
        self._log_debug(f"Total execution time: {total_time/3600:.2f} hours")
        self._log_debug(f"Average rate: {final_rate:.0f} ROMs/sec")
        self._log_debug(f"🎯 TOTAL SORTING ALGORITHMS FOUND: {self.session_stats['total_sorts_found']}")
        
        if self.session_stats['total_sorts_found'] > 0:
            discovery_rate = self.session_stats['total_roms_tested'] // self.session_stats['total_sorts_found']
            self._log_debug(f"🔢 Final discovery rate: 1 per {discovery_rate:,} ROMs")
            self._log_debug(f"📁 ROMs saved to: {self.roms_dir}")
        
        self._log_debug(f"📋 Debug log: {self.debug_log_path}")
        self._log_debug(f"📊 Summary: {self.summary_log_path}")
        
        # Final save
        self._save_session_summary()
        
        return self.session_stats['total_sorts_found'] > 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Babelscope Sorting Algorithm Search')
    parser.add_argument('--batch-size', type=int, default=20000,
                       help='ROMs per batch (default: 20000)')
    parser.add_argument('--cycles', type=int, default=50000,
                       help='Execution cycles per ROM (default: 50000)')
    parser.add_argument('--batches', type=int, default=1,
                       help='Number of batches (default: 1)')
    parser.add_argument('--continuous', action='store_true',
                       help='Run continuous search')
    parser.add_argument('--check-interval', type=int, default=500,
                       help='Sort check interval (default: 500)')
    parser.add_argument('--output-dir', type=str, default='output/sorting',
                       help='Output directory (default: output/sorting)')
    
    args = parser.parse_args()
    
    print("🔢 Babelscope Sorting Algorithm Search")
    print("=" * 50)
    print("🎯 Searching for emergent sorting algorithms in random CHIP-8 code")
    print("📊 Pre-seeds 8 integers at memory 0x300, detects when sorted")
    print()
    
    # Create runner and start search
    runner = SortingSearchRunner(args.output_dir)
    
    success = runner.run_search(
        batch_size=args.batch_size,
        cycles=args.cycles,
        max_batches=args.batches,
        continuous=args.continuous,
        check_interval=args.check_interval
    )
    
    if success:
        print(f"🎉 Search completed successfully!")
        print(f"📁 Results saved to: {runner.output_dir}")
        return 0
    else:
        print(f"❌ Search encountered errors")
        return 1


if __name__ == "__main__":
    exit(main())