#!/usr/bin/env python3
"""
View Interesting ROMs
Loads and displays all the interesting ROMs found by the random generator
Uses the single-instance CHIP-8 emulator for interactive viewing
"""

import os
import sys
import glob
import time
from pathlib import Path

# Add emulators directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'emulators'))

try:
    from chip8 import Chip8Emulator, test_rom_file
    print("CHIP-8 emulator loaded successfully")
except ImportError:
    print("Error: Could not import CHIP-8 emulator")
    print("Make sure chip8.py is in the emulators/ directory")
    sys.exit(1)


def find_interesting_roms(base_dir: str = "output") -> list:
    """Find all interesting ROM files"""
    rom_files = []
    
    # Look in several possible locations
    possible_dirs = [
        os.path.join(base_dir, "interesting_roms"),
        "output/interesting_roms", 
        "interesting_roms",
        base_dir
    ]
    
    for rom_dir in possible_dirs:
        if os.path.exists(rom_dir):
            print(f"Searching in: {os.path.abspath(rom_dir)}")
            # Find .ch8 files
            pattern = os.path.join(rom_dir, "*.ch8")
            found_files = glob.glob(pattern)
            rom_files.extend(found_files)
            break
    
    if not rom_files:
        print("No interesting ROMs found!")
        print("Make sure you've run test_random_roms.py with --save-interesting first")
        return []
    
    # Sort by filename for consistent order
    rom_files.sort()
    return rom_files


def parse_rom_filename(filename: str) -> dict:
    """Parse ROM filename to extract metadata"""
    basename = Path(filename).stem
    parts = basename.split('_')
    
    info = {
        'filename': basename,
        'instructions': 'unknown',
        'pixels': 'unknown', 
        'density': 'unknown'
    }
    
    try:
        for part in parts:
            if part.startswith('inst'):
                info['instructions'] = part[4:]  # Remove 'inst' prefix
            elif part.startswith('pix'):
                info['pixels'] = part[3:]  # Remove 'pix' prefix
            elif part.startswith('dens'):
                info['density'] = part[4:]  # Remove 'dens' prefix
    except:
        pass  # Keep default values if parsing fails
    
    return info


def view_rom_with_auto_close(rom_path: str, timeout: float = 3.0):
    """View a ROM and auto-close if it crashes"""
    print(f"\n{'='*60}")
    print(f"Loading ROM: {Path(rom_path).name}")
    
    # Parse filename for info
    info = parse_rom_filename(rom_path)
    print(f"Instructions: {info['instructions']}, Pixels: {info['pixels']}, Density: {info['density']}")
    
    try:
        # Create emulator and load ROM
        emulator = Chip8Emulator()
        emulator.load_rom(rom_path)
        
        # Run for a bit to see if it crashes immediately
        emulator.run(max_cycles=1000)
        
        if emulator.crashed:
            print(f"ROM crashed immediately - skipping display")
            return True
        
        print("ROM running successfully - opening display...")
        print("(Close window manually, or it will auto-close if ROM crashes)")
        
        # Use the interactive mode but check for crashes
        emulator_result = test_rom_file(
            rom_path, 
            interactive=True, 
            scale=10,
            show_display=True
        )
        
        return emulator_result is not None
        
    except Exception as e:
        print(f"Error running ROM {rom_path}: {e}")
        return False


def view_rom_static(rom_path: str, cycles: int = 50000):
    """View a ROM's final state after running for specified cycles"""
    print(f"\n{'='*60}")
    print(f"Running ROM: {Path(rom_path).name}")
    
    # Parse filename for info
    info = parse_rom_filename(rom_path)
    print(f"Instructions executed: {info['instructions']}")
    print(f"Pixels drawn: {info['pixels']}")
    print(f"Pixel density: {info['density']}")
    
    try:
        # Run ROM non-interactively and show final display
        emulator = test_rom_file(
            rom_path,
            cycles=cycles,
            interactive=False,
            scale=8,
            show_display=True
        )
        
        return emulator is not None
        
    except Exception as e:
        print(f"Error running ROM {rom_path}: {e}")
        return False


def view_all_roms_slideshow(rom_files: list, delay: float = 3.0):
    """Show all ROMs in slideshow mode"""
    print(f"\nStarting slideshow mode...")
    print(f"Will show each ROM for {delay} seconds")
    print("Press Ctrl+C to stop slideshow")
    
    try:
        for i, rom_path in enumerate(rom_files):
            print(f"\n[{i+1}/{len(rom_files)}] {Path(rom_path).name}")
            
            # Create emulator and run ROM
            emulator = Chip8Emulator()
            emulator.load_rom(rom_path)
            emulator.run(max_cycles=50000)
            
            # Show display state briefly
            display = emulator.get_display()
            pixels_set = int(display.sum())
            
            print(f"Final state: {pixels_set} pixels active")
            if pixels_set > 0:
                print("Display output:")
                for row in display:
                    line = ''.join('██' if pixel else '  ' for pixel in row)
                    print(line)
            else:
                print("(No visual output)")
            
            time.sleep(delay)
            
    except KeyboardInterrupt:
        print("\nSlideshow stopped by user")


def main():
    print("Interesting ROMs Viewer - Auto Display Mode")
    print("=" * 50)
    
    # Find interesting ROMs
    rom_files = find_interesting_roms()
    
    if not rom_files:
        return 1
    
    print(f"\nFound {len(rom_files)} interesting ROMs")
    print("Will open each ROM automatically and show display")
    print("Crashed ROMs will be skipped automatically")
    print("Working ROMs will open - close window to continue")
    print("=" * 50)
    
    crashed_count = 0
    working_count = 0
    
    # Automatically open all ROMs one by one
    for i, rom_path in enumerate(rom_files):
        print(f"\n[{i+1}/{len(rom_files)}] Processing: {Path(rom_path).name}")
        
        # Parse filename for info
        info = parse_rom_filename(rom_path)
        print(f"Instructions: {info['instructions']}, Pixels: {info['pixels']}, Density: {info['density']}")
        
        try:
            # Check if ROM crashes immediately (without debug output)
            emulator = Chip8Emulator(debug_file=None)  # No debug file
            emulator.load_rom(rom_path)
            
            # Run for a small number of cycles to test
            emulator.run(max_cycles=1000)
            
            if emulator.crashed:
                print(f"❌ ROM crashed - skipping display")
                crashed_count += 1
                continue
            
            print(f"✓ ROM working - opening display window...")
            print("  Close window to continue to next ROM")
            
            # Open in interactive mode for working ROMs (without debug)
            emulator_result = test_rom_file(
                rom_path, 
                interactive=True, 
                scale=10,
                show_display=True,
                debug=False  # Disable debug output
            )
            
            working_count += 1
            print(f"ROM {i+1} completed")
            
        except Exception as e:
            print(f"❌ Error with ROM {rom_path}: {e}")
            crashed_count += 1
            continue
    
    print(f"\n" + "="*50)
    print(f"Summary:")
    print(f"Total ROMs: {len(rom_files)}")
    print(f"Working ROMs displayed: {working_count}")
    print(f"Crashed ROMs skipped: {crashed_count}")
    print(f"All ROMs processed!")
    return 0


if __name__ == "__main__":
    exit(main())