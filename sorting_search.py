#!/usr/bin/env python3
"""
Correct Babelscope Main Runner
Pure implementation matching the blog post requirements:
1. Generate completely random ROMs
2. Put unique unsorted values at 0x300-0x307  
3. Run complete CHIP-8 emulation
4. Check if values get sorted
5. Save ROMs that achieve sorting

No bias, no templates, no "smart" generation - just pure computational archaeology.
"""

import os
import sys
import time
import json
import signal
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import argparse

# Add emulators directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'emulators'))

try:
    import cupy as cp
    import numpy as np
    print("✅ CuPy loaded")
except ImportError:
    print("❌ CuPy required: pip install cupy-cuda12x")
    sys.exit(1)

try:
    from sorting_emulator import PureBabelscopeDetector, generate_pure_random_roms_gpu, save_discovery_rom
    print("✅ Sorting emulator modules loaded")
except ImportError as e:
    print(f"❌ Failed to import from emulators/sorting_emulator.py: {e}")
    print("Make sure sorting_emulator.py exists in the emulators/ directory")
    sys.exit(1)

class BabelscopeSession:
    """Manages a Babelscope exploration session"""
    
    def __init__(self, batch_size: int, output_dir: str = "babelscope_results"):
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / f"session_{self.session_id}"
        
        # Create directory structure
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.roms_dir = self.session_dir / "discovered_roms"
        self.logs_dir = self.session_dir / "logs"
        self.roms_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Session state
        self.running = True
        self.stats = {
            'session_id': self.session_id,
            'start_time': time.time(),
            'total_roms_tested': 0,
            'total_batches': 0,
            'total_discoveries': 0,
            'batch_history': []
        }
        
        # Initialize detector
        print(f"🔬 Initializing Babelscope session: {self.session_id}")
        print(f"📁 Output directory: {self.session_dir}")
        print(f"📊 Batch size: {batch_size:,}")
        
        self.detector = PureBabelscopeDetector(batch_size)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle graceful shutdown"""
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def run_exploration(self, 
                       max_batches: Optional[int] = None,
                       cycles_per_rom: int = 100000,
                       check_interval: int = 100,
                       save_frequency: int = 10):
        """
        Run the main Babelscope exploration
        
        Args:
            max_batches: Maximum batches to run (None = infinite)
            cycles_per_rom: Execution cycles per ROM
            check_interval: Check for sorting every N cycles (lower = more sensitive)
            save_frequency: Save session state every N batches
        """
        
        print("\n🏹 STARTING BABELSCOPE EXPLORATION")
        print("=" * 60)
        print(f"   Test pattern: [8, 3, 6, 1, 7, 2, 5, 4] at 0x300-0x307")
        print(f"   Cycles per ROM: {cycles_per_rom:,}")
        print(f"   Check interval: every {check_interval} cycles")
        print(f"   Max batches: {max_batches or 'Infinite'}")
        print(f"   Looking for: [1,2,3,4,5,6,7,8] or [8,7,6,5,4,3,2,1]")
        print()
        
        batch_count = 0
        
        try:
            while self.running:
                if max_batches and batch_count >= max_batches:
                    print(f"🏁 Reached maximum batches ({max_batches})")
                    break
                
                batch_count += 1
                batch_start_time = time.time()
                
                print(f"🎯 BATCH {batch_count}")
                print("-" * 30)
                
                # Step 1: Generate completely random ROMs on GPU
                print(f"🎲 Generating {self.batch_size:,} random ROMs on GPU...")
                try:
                    rom_generation_start = time.time()
                    random_roms_gpu = generate_pure_random_roms_gpu(self.batch_size)
                    rom_gen_time = time.time() - rom_generation_start
                    
                    print(f"   ✅ Generated in {rom_gen_time:.2f}s")
                except Exception as e:
                    print(f"   ❌ ROM generation failed: {e}")
                    continue
                
                # Step 2: Load ROMs and setup sort test (direct GPU-to-GPU transfer)
                print(f"📥 Loading ROMs with test pattern...")
                try:
                    load_start = time.time()
                    self.detector.load_random_roms_and_setup_sort_test(random_roms_gpu)
                    load_time = time.time() - load_start
                    
                    print(f"   ✅ Loaded in {load_time:.2f}s")
                except Exception as e:
                    print(f"   ❌ ROM loading failed: {e}")
                    continue
                
                # Step 3: Run complete CHIP-8 emulation and check for sorting
                print(f"🔍 Running CHIP-8 emulation and sort detection...")
                try:
                    search_start = time.time()
                    discoveries = self.detector.run_babelscope_search(
                        cycles=cycles_per_rom, 
                        check_interval=check_interval
                    )
                    search_time = time.time() - search_start
                    
                    print(f"   ✅ Search completed in {search_time:.2f}s")
                    print(f"   🎯 Discoveries: {discoveries}")
                    
                except Exception as e:
                    print(f"   ❌ Search failed: {e}")
                    discoveries = 0
                    continue
                
                # Step 4: Save any discovered ROMs
                discoveries_saved = 0
                if discoveries > 0:
                    print(f"💾 Saving discovered ROMs...")
                    try:
                        discovery_list = self.detector.get_discoveries()
                        
                        for i, discovery in enumerate(discovery_list):
                            filename = save_discovery_rom(
                                discovery, 
                                self.roms_dir, 
                                batch_count, 
                                i + 1
                            )
                            discoveries_saved += 1
                            
                            print(f"      {filename}: cycle {discovery['sort_cycle']:,}")
                        
                    except Exception as e:
                        print(f"   ❌ Failed to save discoveries: {e}")
                
                # Update statistics
                batch_time = time.time() - batch_start_time
                roms_per_second = self.batch_size / batch_time
                
                self.stats['total_roms_tested'] += self.batch_size
                self.stats['total_batches'] = batch_count
                self.stats['total_discoveries'] += discoveries_saved
                
                # Record batch info
                batch_record = {
                    'batch': batch_count,
                    'roms_tested': self.batch_size,
                    'discoveries': discoveries_saved,
                    'batch_time': batch_time,
                    'roms_per_second': roms_per_second,
                    'timestamp': datetime.now().isoformat()
                }
                self.stats['batch_history'].append(batch_record)
                
                # Print batch summary
                print(f"📊 Batch {batch_count} summary:")
                print(f"   ROMs tested: {self.batch_size:,}")
                print(f"   Discoveries: {discoveries_saved}")
                print(f"   Batch time: {batch_time:.2f}s")
                print(f"   Rate: {roms_per_second:,.0f} ROMs/sec")
                
                # Print session totals
                session_time = time.time() - self.stats['start_time']
                total_rate = self.stats['total_roms_tested'] / session_time
                
                print(f"🎯 Session totals:")
                print(f"   Total ROMs: {self.stats['total_roms_tested']:,}")
                print(f"   Total discoveries: {self.stats['total_discoveries']}")
                print(f"   Session time: {session_time/3600:.2f} hours")
                print(f"   Avg rate: {total_rate:,.0f} ROMs/sec")
                
                if self.stats['total_discoveries'] > 0:
                    discovery_rate = self.stats['total_roms_tested'] // self.stats['total_discoveries']
                    print(f"   Discovery rate: 1 in {discovery_rate:,}")
                
                print()
                
                # Save session state periodically
                if batch_count % save_frequency == 0:
                    self._save_session_state()
                
                # Reset detector for next batch
                self.detector.reset()
                
                # Memory cleanup
                del random_roms_gpu
                cp.get_default_memory_pool().free_all_blocks()
        
        except KeyboardInterrupt:
            print(f"\n🛑 Exploration interrupted by user after {batch_count} batches")
        
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Final save and summary
            self._save_session_state()
            self._print_final_summary(batch_count)
    
    def _save_session_state(self):
        """Save current session state with batch history limit"""
        try:
            # Limit batch history to last 100 entries to prevent huge JSON files
            if len(self.stats['batch_history']) > 100:
                print(f"   📝 Limiting batch history to last 100 entries (was {len(self.stats['batch_history'])})")
                self.stats['batch_history'] = self.stats['batch_history'][-100:]
            
            self.stats['last_saved'] = datetime.now().isoformat()
            self.stats['total_time'] = time.time() - self.stats['start_time']
            
            # Save detailed state
            state_file = self.logs_dir / "session_state.json"
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=2, ensure_ascii=True)
            
            # Save human-readable summary
            summary_file = self.logs_dir / "summary.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"Babelscope Exploration Session {self.stats['session_id']}\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"ROMs tested: {self.stats['total_roms_tested']:,}\n")
                f.write(f"Batches completed: {self.stats['total_batches']}\n")
                f.write(f"Sorting algorithms found: {self.stats['total_discoveries']}\n")
                f.write(f"Session time: {self.stats['total_time']/3600:.2f} hours\n")
                
                if self.stats['total_discoveries'] > 0:
                    rate = self.stats['total_roms_tested'] // self.stats['total_discoveries']
                    f.write(f"Discovery rate: 1 in {rate:,} ROMs\n")
                
                f.write(f"\nLast updated: {datetime.now()}\n")
            
            print(f"   💾 Session state saved successfully")
            
        except Exception as e:
            print(f"   ⚠️  Failed to save session state: {e}")
            print(f"       Continuing without saving...")
    
    def _print_final_summary(self, batches_completed: int):
        """Print final session summary"""
        total_time = time.time() - self.stats['start_time']
        final_rate = self.stats['total_roms_tested'] / total_time if total_time > 0 else 0
        
        print("\n🏁 BABELSCOPE EXPLORATION COMPLETE")
        print("=" * 60)
        print(f"Session ID: {self.stats['session_id']}")
        print(f"Batches completed: {batches_completed}")
        print(f"Total ROMs tested: {self.stats['total_roms_tested']:,}")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Average rate: {final_rate:,.0f} ROMs/sec")
        print(f"🎯 SORTING ALGORITHMS DISCOVERED: {self.stats['total_discoveries']}")
        
        if self.stats['total_discoveries'] > 0:
            final_discovery_rate = self.stats['total_roms_tested'] // self.stats['total_discoveries']
            print(f"🔢 Final discovery rate: 1 in {final_discovery_rate:,}")
            print(f"📁 Discovered ROMs saved in: {self.roms_dir}")
        
        print(f"📋 Session data: {self.logs_dir}")
        print(f"📁 All results: {self.session_dir}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Babelscope: Find sorting algorithms in random machine code',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sorting_search.py --batch-size 100000 --batches 50
  python sorting_search.py --batch-size 200000 --infinite
  python sorting_search.py --batch-size 50000 --cycles 200000
        """
    )
    
    parser.add_argument('--batch-size', type=int, default=50000,
                       help='ROMs per batch (default: 50000)')
    parser.add_argument('--batches', type=int, default=10,
                       help='Number of batches (default: 10)')
    parser.add_argument('--infinite', action='store_true',
                       help='Run infinite batches (Ctrl+C to stop)')
    parser.add_argument('--cycles', type=int, default=100000,
                       help='Execution cycles per ROM (default: 100000)')
    parser.add_argument('--check-interval', type=int, default=100,
                       help='Check for sorting every N cycles (default: 100, lower = more sensitive)')
    parser.add_argument('--output-dir', type=str, default='babelscope_results',
                       help='Output directory (default: babelscope_results)')
    parser.add_argument('--save-frequency', type=int, default=10,
                       help='Save state every N batches (default: 10)')
    
    args = parser.parse_args()
    
    print("🔬 BABELSCOPE: COMPUTATIONAL ARCHAEOLOGY")
    print("=" * 60)
    print("🎯 Searching for sorting algorithms in random machine code")
    print("📊 Method: Generate random ROMs, run CHIP-8 emulation, detect sorting")
    print("🧬 Pure exploration - no bias, no templates, just entropy")
    print()
    
    # Validate GPU
    try:
        device = cp.cuda.Device()
        device_props = cp.cuda.runtime.getDeviceProperties(device.id)
        device_name = device_props['name'].decode('utf-8')
        memory_info = device.mem_info
        compute_capability = device.compute_capability
        
        print(f"🎮 GPU: {device_name}")
        print(f"💾 Memory: {memory_info[1] // (1024**3)} GB")
        print(f"🔧 Compute: {compute_capability}")
        print()
    except Exception as e:
        print(f"❌ GPU validation failed: {e}")
        return 1
    
    # Create and run session
    try:
        session = BabelscopeSession(
            batch_size=args.batch_size,
            output_dir=args.output_dir
        )
        
        max_batches = None if args.infinite else args.batches
        
        session.run_exploration(
            max_batches=max_batches,
            cycles_per_rom=args.cycles,
            check_interval=args.check_interval,
            save_frequency=args.save_frequency
        )
        
        print("🎉 Exploration completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Exploration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())