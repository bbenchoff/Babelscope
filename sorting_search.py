#!/usr/bin/env python3
"""
Enhanced Babelscope Main Runner - Multi-Location Detection
Massive improvement: Instead of checking one 8-byte location, now checks
hundreds of locations simultaneously, increasing discovery probability by ~480x!

Pure implementation matching the enhanced blog post requirements:
1. Generate completely random ROMs
2. Put unique unsorted values at MULTIPLE 8-byte chunks (0x600-0xF00)
3. Run complete CHIP-8 emulation
4. Check if ANY location gets sorted
5. Save ROMs that achieve sorting with location metadata

Enhancement: ~480x better discovery rate through multi-location monitoring!
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
    from sorting_emulator import (
        EnhancedBabelscopeDetector, 
        generate_pure_random_roms_gpu, 
        save_enhanced_discovery_rom,
        SORT_LOCATIONS_COUNT,
        SORT_SEARCH_START,
        SORT_SEARCH_END
    )
    print("✅ Enhanced sorting emulator modules loaded")
    print(f"✅ Multi-location enhancement: {SORT_LOCATIONS_COUNT} locations per ROM")
except ImportError as e:
    print(f"❌ Failed to import from emulators/sorting_emulator.py: {e}")
    print("Available items in sorting_emulator:")
    try:
        import sorting_emulator
        print([name for name in dir(sorting_emulator) if not name.startswith('_')])
    except:
        pass
    print("Make sure the updated sorting_emulator.py exists in the emulators/ directory")
    sys.exit(1)

class GlobalStatsManager:
    """Manages persistent global statistics across all Babelscope runs"""
    
    def __init__(self, base_dir: Path = Path(".")):
        self.stats_file = base_dir / "babelscope_global_stats.json"
        self.stats = self._load_global_stats()
    
    def _load_global_stats(self) -> Dict:
        """Load global stats from file, create if doesn't exist"""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r') as f:
                    stats = json.load(f)
                print(f"📊 Loaded global stats: {stats['total_roms_tested']:,} ROMs tested across {stats['total_sessions']} sessions")
                return stats
            except Exception as e:
                print(f"⚠️ Error loading global stats: {e}, creating new stats file")
        
        # Create new stats file
        return {
            'total_roms_tested': 0,
            'total_location_checks': 0,
            'total_discoveries': 0,
            'total_sessions': 0,
            'total_batches': 0,
            'total_runtime_hours': 0.0,
            'first_run': datetime.now().isoformat(),
            'last_run': datetime.now().isoformat(),
            'sessions': []
        }
    
    def start_session(self, session_id: str, batch_size: int) -> None:
        """Record the start of a new session"""
        self.stats['total_sessions'] += 1
        self.stats['last_run'] = datetime.now().isoformat()
        
        session_info = {
            'session_id': session_id,
            'start_time': datetime.now().isoformat(),
            'batch_size': batch_size,
            'status': 'running'
        }
        
        self.stats['sessions'].append(session_info)
        self._save_stats()
    
    def update_session_progress(self, session_id: str, roms_tested: int, location_checks: int, 
                              batches: int, discoveries: int, runtime_hours: float) -> None:
        """Update progress for current session"""
        # Update global totals
        session_stats = self.stats['sessions'][-1] if self.stats['sessions'] else {}
        
        # Calculate incremental changes
        prev_roms = session_stats.get('roms_tested', 0)
        prev_checks = session_stats.get('location_checks', 0)
        prev_batches = session_stats.get('batches', 0)
        prev_discoveries = session_stats.get('discoveries', 0)
        prev_runtime = session_stats.get('runtime_hours', 0.0)
        
        # Add incremental changes to global totals
        self.stats['total_roms_tested'] += (roms_tested - prev_roms)
        self.stats['total_location_checks'] += (location_checks - prev_checks)
        self.stats['total_batches'] += (batches - prev_batches)
        self.stats['total_discoveries'] += (discoveries - prev_discoveries)
        self.stats['total_runtime_hours'] += (runtime_hours - prev_runtime)
        
        # Update current session
        if self.stats['sessions']:
            self.stats['sessions'][-1].update({
                'roms_tested': roms_tested,
                'location_checks': location_checks,
                'batches': batches,
                'discoveries': discoveries,
                'runtime_hours': runtime_hours,
                'last_update': datetime.now().isoformat()
            })
        
        self._save_stats()
    
    def finish_session(self, session_id: str) -> None:
        """Mark session as completed"""
        if self.stats['sessions']:
            self.stats['sessions'][-1]['status'] = 'completed'
            self.stats['sessions'][-1]['end_time'] = datetime.now().isoformat()
        
        self._save_stats()
    
    def _save_stats(self) -> None:
        """Save stats to file with atomic write for safety"""
        try:
            # Use a temporary file and atomic move to prevent corruption
            temp_file = self.stats_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
            
            # Atomic move (works on both Windows and Unix)
            if os.name == 'nt':  # Windows
                if self.stats_file.exists():
                    self.stats_file.unlink()  # Windows requires explicit delete before replace
            temp_file.replace(self.stats_file)
        except Exception as e:
            print(f"⚠️ Error saving global stats: {e}")
            # Clean up temp file if it exists
            if temp_file.exists():
                temp_file.unlink()
    
    def print_global_summary(self) -> None:
        """Print a nice summary of all-time stats"""
        print("\n" + "="*70)
        print("🌍 BABELSCOPE GLOBAL STATISTICS")
        print("="*70)
        print(f"🎯 Total ROMs tested: {self.stats['total_roms_tested']:,}")
        print(f"🔍 Total location-checks: {self.stats['total_location_checks']:,}")
        print(f"🏆 Total discoveries: {self.stats['total_discoveries']}")
        print(f"📊 Total sessions: {self.stats['total_sessions']}")
        print(f"⚡ Total batches: {self.stats['total_batches']:,}")
        print(f"⏱️  Total runtime: {self.stats['total_runtime_hours']:.1f} hours")
        
        if self.stats['total_discoveries'] > 0:
            discovery_rate = self.stats['total_roms_tested'] // self.stats['total_discoveries']
            print(f"📈 All-time discovery rate: 1 in {discovery_rate:,} ROMs")
        
        if self.stats['total_runtime_hours'] > 0:
            avg_rate = self.stats['total_roms_tested'] / self.stats['total_runtime_hours']
            print(f"🚀 Average processing rate: {avg_rate:,.0f} ROMs/hour")
        
        print(f"📅 First run: {self.stats['first_run'][:10]}")
        print(f"📅 Last run: {self.stats['last_run'][:10]}")
        print("="*70 + "\n")


class EnhancedBabelscopeSession:
    """Manages an Enhanced Multi-Location Babelscope exploration session"""
    
    def __init__(self, batch_size: int, output_dir: str = "output/sorting"):
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
        
        # Initialize global stats manager
        self.global_stats = GlobalStatsManager()
        
        # Session state
        self.running = True
        self.stats = {
            'session_id': self.session_id,
            'enhancement_type': 'multi_location',
            'locations_per_rom': SORT_LOCATIONS_COUNT,
            'search_range': f"0x{SORT_SEARCH_START:03X}-0x{SORT_SEARCH_END:03X}",
            'discovery_multiplier': SORT_LOCATIONS_COUNT,
            'start_time': time.time(),
            'total_roms_tested': 0,
            'total_location_checks': 0,
            'total_batches': 0,
            'total_discoveries': 0,
            'batch_history': []
        }
        
        # Initialize enhanced detector
        print(f"🔬 Initializing Enhanced Multi-Location Babelscope session: {self.session_id}")
        print(f"📁 Output directory: {self.session_dir}")
        print(f"📊 Batch size: {batch_size:,}")
        print(f"🎯 Enhancement: {SORT_LOCATIONS_COUNT} locations per ROM (~{SORT_LOCATIONS_COUNT}x discovery rate)")
        
        # Print global stats
        self.global_stats.print_global_summary()
        
        # Register this session
        self.global_stats.start_session(self.session_id, batch_size)
        
        self.detector = EnhancedBabelscopeDetector(batch_size)
        
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
        Run the main Enhanced Multi-Location Babelscope exploration
        
        Args:
            max_batches: Maximum batches to run (None = infinite)
            cycles_per_rom: Execution cycles per ROM
            check_interval: Check for sorting every N cycles (lower = more sensitive)
            save_frequency: Save session state every N batches
        """
        
        print("\n🏹 STARTING ENHANCED MULTI-LOCATION BABELSCOPE EXPLORATION")
        print("=" * 70)
        print(f"   Enhancement: Multi-location detection (~{SORT_LOCATIONS_COUNT}x discovery rate)")
        print(f"   Test pattern: [8, 3, 6, 1, 7, 2, 5, 4] at {SORT_LOCATIONS_COUNT} locations")
        print(f"   Search range: 0x{SORT_SEARCH_START:03X} to 0x{SORT_SEARCH_END:03X}")
        print(f"   Cycles per ROM: {cycles_per_rom:,}")
        print(f"   Check interval: every {check_interval} cycles")
        print(f"   Max batches: {max_batches or 'Infinite'}")
        print(f"   Looking for: [1,2,3,4,5,6,7,8] or [8,7,6,5,4,3,2,1] at ANY location")
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
                
                # Step 2: Load ROMs and setup enhanced multi-location test
                print(f"📥 Loading ROMs with multi-location test pattern...")
                try:
                    load_start = time.time()
                    self.detector.load_random_roms_and_setup_multilocation_test(random_roms_gpu)
                    load_time = time.time() - load_start
                    
                    print(f"   ✅ Loaded in {load_time:.2f}s")
                    print(f"   🎯 Monitoring {SORT_LOCATIONS_COUNT} locations per ROM")
                except Exception as e:
                    print(f"   ❌ ROM loading failed: {e}")
                    continue
                
                # Step 3: Run enhanced CHIP-8 emulation with multi-location detection
                print(f"🔍 Running enhanced CHIP-8 emulation and multi-location sort detection...")
                try:
                    search_start = time.time()
                    discoveries = self.detector.run_enhanced_babelscope_search(
                        cycles=cycles_per_rom, 
                        check_interval=check_interval
                    )
                    search_time = time.time() - search_start
                    
                    effective_checks = self.batch_size * SORT_LOCATIONS_COUNT
                    checks_per_second = effective_checks / search_time
                    
                    print(f"   ✅ Search completed in {search_time:.2f}s")
                    print(f"   🎯 Discoveries: {discoveries}")
                    print(f"   📊 Location-checks: {effective_checks:,} ({checks_per_second:,.0f}/sec)")
                    
                except Exception as e:
                    print(f"   ❌ Search failed: {e}")
                    discoveries = 0
                    effective_checks = 0
                    continue
                
                # Step 4: Save any discovered ROMs with enhanced metadata
                discoveries_saved = 0
                if discoveries > 0:
                    print(f"💾 Saving discovered ROMs with location metadata...")
                    try:
                        discovery_list = self.detector.get_enhanced_discoveries()
                        
                        for i, discovery in enumerate(discovery_list):
                            filename = save_enhanced_discovery_rom(
                                discovery, 
                                self.roms_dir, 
                                batch_count, 
                                i + 1
                            )
                            discoveries_saved += 1
                            
                            location = discovery['sort_location']
                            direction = discovery['sort_direction']
                            cycle = discovery['sort_cycle']
                            
                            print(f"      {filename}: 0x{location:03X} {direction} @ cycle {cycle:,}")
                        
                    except Exception as e:
                        print(f"   ❌ Failed to save discoveries: {e}")
                
                # Update statistics
                batch_time = time.time() - batch_start_time
                roms_per_second = self.batch_size / batch_time
                effective_checks_this_batch = self.batch_size * SORT_LOCATIONS_COUNT
                
                self.stats['total_roms_tested'] += self.batch_size
                self.stats['total_location_checks'] += effective_checks_this_batch
                self.stats['total_batches'] = batch_count
                self.stats['total_discoveries'] += discoveries_saved
                
                # Update global stats
                session_time = time.time() - self.stats['start_time']
                self.global_stats.update_session_progress(
                    self.session_id,
                    self.stats['total_roms_tested'],
                    self.stats['total_location_checks'],
                    self.stats['total_batches'],
                    self.stats['total_discoveries'],
                    session_time / 3600  # Convert to hours
                )
                
                # Record batch info
                batch_record = {
                    'batch': batch_count,
                    'roms_tested': self.batch_size,
                    'location_checks': effective_checks_this_batch,
                    'discoveries': discoveries_saved,
                    'batch_time': batch_time,
                    'roms_per_second': roms_per_second,
                    'location_checks_per_second': effective_checks_this_batch / batch_time,
                    'timestamp': datetime.now().isoformat()
                }
                self.stats['batch_history'].append(batch_record)
                
                # Print batch summary
                print(f"📊 Batch {batch_count} summary:")
                print(f"   ROMs tested: {self.batch_size:,}")
                print(f"   Location-checks: {effective_checks_this_batch:,}")
                print(f"   Discoveries: {discoveries_saved}")
                print(f"   Batch time: {batch_time:.2f}s")
                print(f"   ROM rate: {roms_per_second:,.0f} ROMs/sec")
                print(f"   Location-check rate: {effective_checks_this_batch / batch_time:,.0f} checks/sec")
                
                # Print session totals
                session_time = time.time() - self.stats['start_time']
                total_rom_rate = self.stats['total_roms_tested'] / session_time
                total_check_rate = self.stats['total_location_checks'] / session_time
                
                print(f"🎯 Session totals:")
                print(f"   Total ROMs: {self.stats['total_roms_tested']:,}")
                print(f"   Total location-checks: {self.stats['total_location_checks']:,}")
                print(f"   Total discoveries: {self.stats['total_discoveries']}")
                print(f"   Session time: {session_time/3600:.2f} hours")
                print(f"   Avg ROM rate: {total_rom_rate:,.0f} ROMs/sec")
                print(f"   Avg location-check rate: {total_check_rate:,.0f} checks/sec")
                
                if self.stats['total_discoveries'] > 0:
                    rom_discovery_rate = self.stats['total_roms_tested'] // self.stats['total_discoveries']
                    check_discovery_rate = self.stats['total_location_checks'] // self.stats['total_discoveries']
                    print(f"   Discovery rate: 1 in {rom_discovery_rate:,} ROMs")
                    print(f"   Effective discovery rate: 1 in {check_discovery_rate:,} location-checks")
                    improvement = SORT_LOCATIONS_COUNT
                    print(f"   Enhancement factor: ~{improvement}x better than single-location")
                
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
            # Mark session as completed and final save
            self.global_stats.finish_session(self.session_id)
            self._save_session_state()
            self._print_final_summary(batch_count)
            
            # Print updated global stats
            print("\n" + "🌍 UPDATED GLOBAL STATISTICS:")
            self.global_stats.print_global_summary()
    
    def _save_session_state(self):
        """Save current session state with enhanced metrics"""
        try:
            # Limit batch history to last 100 entries to prevent huge JSON files
            if len(self.stats['batch_history']) > 100:
                print(f"   📝 Limiting batch history to last 100 entries (was {len(self.stats['batch_history'])})")
                self.stats['batch_history'] = self.stats['batch_history'][-100:]
            
            self.stats['last_saved'] = datetime.now().isoformat()
            self.stats['total_time'] = time.time() - self.stats['start_time']
            
            # Calculate enhancement metrics
            if self.stats['total_discoveries'] > 0:
                self.stats['rom_discovery_rate'] = self.stats['total_roms_tested'] // self.stats['total_discoveries']
                self.stats['location_discovery_rate'] = self.stats['total_location_checks'] // self.stats['total_discoveries']
            
            # Save detailed state
            state_file = self.logs_dir / "enhanced_session_state.json"
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=2, ensure_ascii=True)
            
            # Save human-readable summary
            summary_file = self.logs_dir / "enhanced_summary.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"Enhanced Multi-Location Babelscope Session {self.stats['session_id']}\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"Enhancement: Multi-location detection (~{SORT_LOCATIONS_COUNT}x discovery rate)\n")
                f.write(f"Search range: {self.stats['search_range']}\n")
                f.write(f"Locations per ROM: {self.stats['locations_per_rom']}\n\n")
                f.write(f"ROMs tested: {self.stats['total_roms_tested']:,}\n")
                f.write(f"Location-checks performed: {self.stats['total_location_checks']:,}\n")
                f.write(f"Batches completed: {self.stats['total_batches']}\n")
                f.write(f"Sorting algorithms found: {self.stats['total_discoveries']}\n")
                f.write(f"Session time: {self.stats['total_time']/3600:.2f} hours\n")
                
                if self.stats['total_discoveries'] > 0:
                    f.write(f"ROM discovery rate: 1 in {self.stats['total_roms_tested'] // self.stats['total_discoveries']:,} ROMs\n")
                    f.write(f"Location-check discovery rate: 1 in {self.stats['total_location_checks'] // self.stats['total_discoveries']:,} checks\n")
                    f.write(f"Enhancement effectiveness: ~{SORT_LOCATIONS_COUNT}x improvement\n")
                
                f.write(f"\nLast updated: {datetime.now()}\n")
            
            print(f"   💾 Enhanced session state saved successfully")
            
        except Exception as e:
            print(f"   ⚠️  Failed to save session state: {e}")
            print(f"       Continuing without saving...")
    
    def _print_final_summary(self, batches_completed: int):
        """Print final enhanced session summary"""
        total_time = time.time() - self.stats['start_time']
        final_rom_rate = self.stats['total_roms_tested'] / total_time if total_time > 0 else 0
        final_check_rate = self.stats['total_location_checks'] / total_time if total_time > 0 else 0
        
        print("\n🏁 ENHANCED MULTI-LOCATION BABELSCOPE EXPLORATION COMPLETE")
        print("=" * 70)
        print(f"Session ID: {self.stats['session_id']}")
        print(f"Enhancement: Multi-location detection (~{SORT_LOCATIONS_COUNT}x discovery rate)")
        print(f"Batches completed: {batches_completed}")
        print(f"Total ROMs tested: {self.stats['total_roms_tested']:,}")
        print(f"Total location-checks: {self.stats['total_location_checks']:,}")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Average ROM rate: {final_rom_rate:,.0f} ROMs/sec")
        print(f"Average location-check rate: {final_check_rate:,.0f} checks/sec")
        print(f"🎯 SORTING ALGORITHMS DISCOVERED: {self.stats['total_discoveries']}")
        
        if self.stats['total_discoveries'] > 0:
            final_rom_discovery_rate = self.stats['total_roms_tested'] // self.stats['total_discoveries']
            final_check_discovery_rate = self.stats['total_location_checks'] // self.stats['total_discoveries']
            print(f"🔢 Final ROM discovery rate: 1 in {final_rom_discovery_rate:,}")
            print(f"🔢 Final location-check discovery rate: 1 in {final_check_discovery_rate:,}")
            print(f"⚡ Enhancement effectiveness: ~{SORT_LOCATIONS_COUNT}x improvement over single-location")
            print(f"📁 Discovered ROMs saved in: {self.roms_dir}")
        else:
            expected_improvement = self.stats['total_location_checks'] / self.stats['total_roms_tested']
            print(f"📊 No discoveries, but searched {expected_improvement:.0f}x more locations than single-location method")
            print(f"📊 Equivalent to testing {self.stats['total_location_checks']:,} single-location ROMs")
        
        print(f"📋 Session data: {self.logs_dir}")
        print(f"📁 All results: {self.session_dir}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Enhanced Babelscope: Find sorting algorithms with multi-location detection (~480x better)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sorting_search.py --batch-size 100000 --batches 50
  python sorting_search.py --batch-size 200000 --infinite
  python sorting_search.py --batch-size 50000 --cycles 200000

Enhancement: Multi-location detection monitors ~480 locations per ROM instead of 1,
dramatically increasing discovery probability!
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
    parser.add_argument('--output-dir', type=str, default='output/sorting',
                       help='Output directory (default: output/sorting)')
    parser.add_argument('--save-frequency', type=int, default=10,
                       help='Save state every N batches (default: 10)')
    
    args = parser.parse_args()
    
    print("🔬 ENHANCED MULTI-LOCATION BABELSCOPE: COMPUTATIONAL ARCHAEOLOGY")
    print("=" * 70)
    print("🎯 Searching for sorting algorithms in random machine code")
    print("⚡ ENHANCEMENT: Multi-location detection (~480x better discovery rate)")
    print("📊 Method: Generate random ROMs, run CHIP-8 emulation, detect sorting at ANY location")
    print("🧬 Pure exploration - no bias, no templates, just enhanced entropy detection")
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
    
    # Show enhancement details
    print("🚀 MULTI-LOCATION ENHANCEMENT DETAILS:")
    print(f"   Locations monitored per ROM: {SORT_LOCATIONS_COUNT}")
    print(f"   Search range: 0x{SORT_SEARCH_START:03X} to 0x{SORT_SEARCH_END:03X}")
    print(f"   Discovery probability multiplier: ~{SORT_LOCATIONS_COUNT}x")
    print(f"   Expected discovery rate improvement: Massive!")
    print()
    
    # Create and run enhanced session
    try:
        session = EnhancedBabelscopeSession(
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
        
        print("🎉 Enhanced exploration completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Enhanced exploration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())