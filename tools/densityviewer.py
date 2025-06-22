#!/usr/bin/env python3
"""
Analyze interesting ROMs and run the top 10 by pixel density
"""

import os
import re
import sys
import glob
import time
from pathlib import Path
from typing import List, Tuple, Dict
import argparse

# Add the emulators directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'emulators'))

try:
    from chip8 import Chip8Emulator
    print("Single-instance emulator loaded")
except ImportError:
    print("Error: Could not import CHIP-8 emulator")
    print("Make sure chip8.py is in the emulators/ directory")
    sys.exit(1)


def extract_density_from_filename(filename: str) -> float:
    """
    Extract density from filename
    Looks for pattern _dens0.139 or _densX.XXX in the filename
    """
    # Look for _dens followed by a decimal number, handling edge cases
    pattern = r'_dens(\d+(?:\.\d+)?)'
    match = re.search(pattern, filename)
    
    if match:
        density_str = match.group(1)
        # Remove any trailing dots
        density_str = density_str.rstrip('.')
        try:
            return float(density_str)
        except ValueError:
            print(f"Warning: Could not parse density '{density_str}' in filename: {filename}")
            return 0.0
    else:
        print(f"Warning: Could not find density in filename: {filename}")
        return 0.0


def parse_rom_filename(filename: str) -> Dict:
    """
    Parse ROM filename to extract basic info and density
    """
    density = extract_density_from_filename(filename)
    
    # Try to extract other info if available
    rom_id = None
    instructions = None
    pixels = None
    ca_likelihood = 0
    
    # Look for ROM ID
    id_match = re.search(r'random_(\d+)', filename)
    if id_match:
        rom_id = int(id_match.group(1))
    
    # Look for instructions
    inst_match = re.search(r'_inst(\d+)', filename)
    if inst_match:
        instructions = int(inst_match.group(1))
    
    # Look for pixels
    pix_match = re.search(r'_pix(\d+)', filename)
    if pix_match:
        pixels = int(pix_match.group(1))
    
    # Look for CA likelihood
    ca_match = re.search(r'_CA(\d+)', filename)
    if ca_match:
        ca_likelihood = int(ca_match.group(1))
    
    return {
        'filename': filename,
        'density': density,
        'rom_id': rom_id,
        'instructions': instructions,
        'pixels': pixels,
        'ca_likelihood': ca_likelihood
    }


def find_interesting_roms(rom_dir: str) -> List[Dict]:
    """Find all interesting ROM files and parse their metadata"""
    rom_files = []
    
    # Look for .ch8 files
    pattern = os.path.join(rom_dir, "*.ch8")
    files = glob.glob(pattern)
    
    print(f"Found {len(files)} ROM files in {rom_dir}")
    
    for filepath in files:
        filename = os.path.basename(filepath)
        metadata = parse_rom_filename(filename)
        
        if metadata and metadata['density'] > 0:
            metadata['filepath'] = filepath
            rom_files.append(metadata)
        else:
            print(f"Skipped: {filename}")
    
    return rom_files


def analyze_rom_with_emulator(rom_path: str, cycles: int = 1000000) -> Dict:
    """Run a ROM in the emulator and return analysis"""
    try:
        # Load ROM data
        with open(rom_path, 'rb') as f:
            rom_data = f.read()
        
        # Create and run emulator
        emulator = Chip8Emulator()
        emulator.load_rom(rom_data)
        
        # Track execution
        start_instructions = emulator.stats.get('instructions_executed', 0)
        
        # Run for specified cycles
        for _ in range(cycles):
            if emulator.crashed or emulator.halt:
                break
            if not emulator.step():
                break
        
        # Get final state
        final_instructions = emulator.stats.get('instructions_executed', 0)
        display = emulator.display
        pixels_set = sum(sum(row) for row in display)
        total_pixels = len(display) * len(display[0])
        
        return {
            'instructions_executed': final_instructions - start_instructions,
            'pixels_set': pixels_set,
            'pixel_density': pixels_set / total_pixels,
            'crashed': emulator.crashed,
            'halted': emulator.halt,
            'final_pc': emulator.program_counter,
            'display_state': display
        }
        
    except Exception as e:
        print(f"Error analyzing ROM {rom_path}: {e}")
        return None


def print_rom_analysis(metadata: Dict, analysis: Dict = None):
    """Print detailed analysis of a ROM"""
    print(f"Filename: {metadata['filename']}")
    print(f"Density: {metadata['density']:.3f}")
    
    if metadata['rom_id']:
        print(f"ROM ID: {metadata['rom_id']:06d}")
    if metadata['instructions']:
        print(f"Instructions: {metadata['instructions']:,}")
    if metadata['pixels']:
        print(f"Pixels: {metadata['pixels']}")
    if metadata['ca_likelihood'] > 0:
        print(f"CA Likelihood: {metadata['ca_likelihood']}%")
    
    if analysis:
        print(f"Re-run analysis:")
        print(f"  Instructions executed: {analysis['instructions_executed']:,}")
        print(f"  Final pixels set: {analysis['pixels_set']}")
        print(f"  Final density: {analysis['pixel_density']:.3f}")
        print(f"  Final PC: 0x{analysis['final_pc']:03X}")
        print(f"  Crashed: {analysis['crashed']}")
        print(f"  Halted: {analysis['halted']}")
    
    print("-" * 50)


def run_rom_interactive(rom_path: str) -> bool:
    """Run a ROM interactively with display window"""
    try:
        # Load ROM data
        with open(rom_path, 'rb') as f:
            rom_data = f.read()
        
        # Create emulator and load ROM
        emulator = Chip8Emulator(debug_file=None)  # No debug output
        emulator.load_rom(rom_data)
        
        # Test if ROM crashes immediately
        emulator.run(max_cycles=1000)
        
        if emulator.crashed:
            print(f"❌ ROM crashed immediately - skipping display")
            return False
        
        print(f"✓ ROM working - opening display window...")
        print("  Close window to continue to next ROM")
        
        # Import test_rom_file function
        from chip8 import test_rom_file
        
        # Open in interactive mode
        emulator_result = test_rom_file(
            rom_path, 
            interactive=True, 
            scale=10,
            show_display=True,
            debug=False
        )
        
        return emulator_result is not None
        
    except Exception as e:
        print(f"❌ Error running ROM {rom_path}: {e}")
        return False


def run_roms_sequence(roms: List[Dict], start_from: int = 0):
    """Run ROMs in sequence with interactive display"""
    print(f"\nStarting ROM sequence from #{start_from + 1}...")
    print("Each ROM will open in a window - close to continue to next")
    print("Press Ctrl+C to stop sequence")
    print("=" * 60)
    
    working_count = 0
    crashed_count = 0
    
    try:
        for i, rom in enumerate(roms[start_from:], start_from):
            print(f"\n[{i+1}/{len(roms)}] Processing: {rom['filename']}")
            print(f"Density: {rom['density']:.3f}")
            
            if rom['rom_id']:
                print(f"ROM ID: {rom['rom_id']:06d}")
            if rom['ca_likelihood'] > 0:
                print(f"🔬 CA Likelihood: {rom['ca_likelihood']}%")
            
            success = run_rom_interactive(rom['filepath'])
            
            if success:
                working_count += 1
                print(f"✓ ROM {i+1} completed")
            else:
                crashed_count += 1
            
            # Small delay between ROMs
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print(f"\n🛑 Sequence stopped by user at ROM {i+1}")
    
    print(f"\n" + "="*60)
    print(f"Summary:")
    print(f"ROMs processed: {i+1 - start_from}")
    print(f"Working ROMs displayed: {working_count}")
    print(f"Crashed ROMs skipped: {crashed_count}")
    print(f"Sequence completed!")


def main():
    parser = argparse.ArgumentParser(description="Analyze interesting ROMs by pixel density")
    parser.add_argument("--rom-dir", type=str, default="output/interesting_roms", 
                       help="Directory containing interesting ROMs")
    parser.add_argument("--top", type=int, default=10, 
                       help="Number of top ROMs to analyze")
    parser.add_argument("--cycles", type=int, default=1000000, 
                       help="Cycles to run each ROM for re-analysis")
    parser.add_argument("--list-only", action="store_true", 
                       help="Only list ROMs, don't re-run them")
    parser.add_argument("--ca-only", action="store_true", 
                       help="Only analyze ROMs with CA patterns")
    parser.add_argument("--interactive", action="store_true",
                       help="Run ROMs in sequence with interactive display")
    parser.add_argument("--start-from", type=int, default=0,
                       help="Start sequence from this ROM number (0-based)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.rom_dir):
        print(f"ROM directory not found: {args.rom_dir}")
        return 1
    
    # Find all interesting ROMs
    roms = find_interesting_roms(args.rom_dir)
    
    if not roms:
        print("No ROM files found!")
        return 1
    
    # Filter for CA-only if requested
    if args.ca_only:
        roms = [rom for rom in roms if rom['ca_likelihood'] > 0]
        print(f"Filtered to {len(roms)} ROMs with CA patterns")
    
    # Sort by density (highest first)
    roms.sort(key=lambda x: x['density'], reverse=True)
    
    # Interactive sequence mode
    if args.interactive:
        if args.top < len(roms):
            roms = roms[:args.top]
            print(f"Running top {args.top} ROMs by density in interactive sequence")
        else:
            print(f"Running all {len(roms)} ROMs in interactive sequence")
        
        run_roms_sequence(roms, args.start_from)
        return 0
    
    # Normal analysis mode
    print(f"\nTop {args.top} ROMs by pixel density:")
    print("=" * 60)
    
    # Show top N ROMs
    top_roms = roms[:args.top]
    
    for i, rom in enumerate(top_roms, 1):
        print(f"\n#{i}")
        
        if args.list_only:
            print_rom_analysis(rom)
        else:
            # Re-run the ROM for fresh analysis
            print(f"Running ROM: {rom['filename']}")
            analysis = analyze_rom_with_emulator(rom['filepath'], args.cycles)
            print_rom_analysis(rom, analysis)
    
    # Show summary statistics
    print(f"\nSummary of all {len(roms)} ROMs:")
    print(f"Highest density: {max(rom['density'] for rom in roms):.3f}")
    print(f"Lowest density: {min(rom['density'] for rom in roms):.3f}")
    print(f"Average density: {sum(rom['density'] for rom in roms) / len(roms):.3f}")
    
    ca_roms = [rom for rom in roms if rom['ca_likelihood'] > 0]
    if ca_roms:
        print(f"ROMs with CA patterns: {len(ca_roms)}")
        print(f"Highest CA likelihood: {max(rom['ca_likelihood'] for rom in ca_roms)}%")
    
    return 0


if __name__ == "__main__":
    exit(main())