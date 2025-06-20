#!/usr/bin/env python3
"""
Test Random CHIP-8 ROMs
Loads random ROM data and tests which ones actually execute without crashing
Following the Finite Atari Machine approach of computational archaeology
"""

import os
import sys
import glob
import argparse
import time
import numpy as np
import cupy as cp
from pathlib import Path
from typing import List, Dict, Tuple

# Add the emulators and generators directories to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'emulators'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'generators'))

try:
    from mega_kernel_chip8 import MegaKernelChip8Emulator as ParallelChip8Emulator
    print("Using mega-kernel emulator")
except ImportError:
    try:
        from parallel_chip8 import ParallelChip8Emulator
        print("Using standard parallel emulator")
    except ImportError:
        print("Error: No emulator available")
        sys.exit(1)

try:
    from random_chip8_generator import PureRandomChip8Generator
    print("Pure random ROM generator available")
except ImportError:
    print("Warning: Pure random ROM generator not available")
    PureRandomChip8Generator = None


def load_rom_files(rom_dir: str) -> List[Tuple[str, bytes]]:
    """Load all ROM files from a directory"""
    rom_files = []
    
    # Look for .ch8 and .bin files
    for pattern in ["*.ch8", "*.bin"]:
        files = glob.glob(os.path.join(rom_dir, pattern))
        for filepath in files:
            try:
                with open(filepath, 'rb') as f:
                    rom_data = f.read()
                rom_files.append((os.path.basename(filepath), rom_data))
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    
    return sorted(rom_files)


def test_rom_batch(rom_data_list: List[bytes], cycles: int = 1000000) -> List[Dict]:
    """
    Test a batch of ROMs in parallel
    """
    print(f"Testing batch of {len(rom_data_list)} ROMs...")
    
    # Create emulator for the batch
    emulator = ParallelChip8Emulator(len(rom_data_list))
    
    # Load all ROMs
    emulator.load_roms(rom_data_list)
    
    # Run all ROMs
    start_time = time.time()
    emulator.run(cycles=cycles)
    execution_time = time.time() - start_time
    
    print(f"Batch execution completed in {execution_time:.2f} seconds")
    
    # Analyze results for each ROM
    results = []
    displays = emulator.get_displays()
    
    for i in range(len(rom_data_list)):
        # Get state for this instance
        final_pc = int(emulator.program_counter[i])
        crashed = bool(emulator.crashed[i])
        halted = bool(emulator.halted[i])
        waiting_for_key = bool(emulator.waiting_for_key[i])
        
        # Analyze display
        display = displays[i]
        pixels_set = int(np.sum(display > 0))
        total_pixels = display.shape[0] * display.shape[1]
        pixel_density = pixels_set / total_pixels
        
        has_output = pixels_set > 0
        has_structure = pixel_density > 0.05 and pixel_density < 0.5
        
        # Get per-instance stats
        instructions = int(emulator.stats['instructions_executed'][i])
        display_writes = int(emulator.stats['display_writes'][i])
        pixels_drawn = int(emulator.stats['pixels_drawn'][i])
        
        result = {
            'execution_time': execution_time / len(rom_data_list),  # Approximate
            'final_pc': final_pc,
            'crashed': crashed,
            'halted': halted,
            'waiting_for_key': waiting_for_key,
            'completed_normally': not crashed and not waiting_for_key,
            'instructions_executed': instructions,
            'display_writes': display_writes,
            'pixels_drawn': pixels_drawn,
            'final_pixel_count': pixels_set,
            'pixel_density': pixel_density,
            'has_output': has_output,
            'has_structure': has_structure,
            'interesting': (not crashed and not waiting_for_key) and (has_output and has_structure)
        }
        
        results.append(result)
    
    return results


def save_interesting_roms(rom_files: List[Tuple[str, bytes]], results: List[Dict], 
                         output_dir: str = "interesting_roms"):
    """Save ROMs that produced interesting results and generate screenshots"""
    os.makedirs(output_dir, exist_ok=True)
    
    saved_count = 0
    for (filename, rom_data), result in zip(rom_files, results):
        if result['interesting']:
            # Create descriptive filename
            base_name = Path(filename).stem
            new_filename = (f"{base_name}_"
                          f"inst{result['instructions_executed']}_"
                          f"pix{result['final_pixel_count']}_"
                          f"dens{result['pixel_density']:.3f}")
            
            # Save ROM file
            rom_path = os.path.join(output_dir, f"{new_filename}.ch8")
            with open(rom_path, 'wb') as f:
                f.write(rom_data)
            
            # Generate screenshot by running the ROM again
            print(f"Generating screenshot for {new_filename}...")
            try:
                # Create single-instance emulator
                emulator = ParallelChip8Emulator(1)
                emulator.load_single_rom(rom_data)
                
                # Run for the same number of cycles as the original test
                emulator.run(cycles=1000000)
                
                # Get display and convert to numpy
                display = emulator.get_displays()[0]  # Shape: (32, 64)
                if hasattr(display, 'get'):  # CuPy array
                    display_np = display.get()
                else:  # Already numpy
                    display_np = display
                
                # Convert to PIL Image
                from PIL import Image
                display_img = (display_np * 255).astype(np.uint8)
                
                # Scale up 8x for visibility (256x512 pixels)
                scale_factor = 8
                scaled_img = np.repeat(np.repeat(display_img, scale_factor, axis=0), scale_factor, axis=1)
                
                # Save as PNG
                img = Image.fromarray(scaled_img, mode='L')
                screenshot_path = os.path.join(output_dir, f"{new_filename}.png")
                img.save(screenshot_path)
                
            except Exception as e:
                print(f"Error generating screenshot for {new_filename}: {e}")
            
            saved_count += 1
    
    print(f"Saved {saved_count} interesting ROMs with screenshots to {output_dir}/")
    return saved_count


def generate_and_test_random_batch(generator, num_roms: int = 10000, 
                                 test_cycles: int = 1000000) -> Tuple[int, int]:
    """Generate pure random ROMs and test them immediately"""
    print(f"Generating and testing {num_roms:,} pure random ROMs...")
    
    # Generate pure random ROMs - no filtering at all
    roms = generator.generate_batch(num_roms)
    
    # Convert to list of bytes
    rom_data_list = [cp.asnumpy(rom).tobytes() for rom in roms]
    
    # Test them all
    results = test_rom_batch(rom_data_list, cycles=test_cycles)
    
    # Count results
    interesting_count = sum(1 for r in results if r['interesting'])
    completed_count = sum(1 for r in results if r['completed_normally'])
    has_output_count = sum(1 for r in results if r['has_output'])
    crashed_count = sum(1 for r in results if r['crashed'])
    
    print(f"Results: {crashed_count} crashed, {completed_count} completed normally, "
          f"{has_output_count} had visual output, {interesting_count} were interesting")
    
    # Save ROMs if there are interesting ones
    if interesting_count > 0:
        # Create fake filenames for the ROM data
        rom_files = [(f"random_{i:06d}.ch8", rom_data) for i, rom_data in enumerate(rom_data_list)]
        save_interesting_roms(rom_files, results, "output/interesting_roms")
    
    return completed_count, interesting_count


def print_summary_stats(results: List[Dict]):
    """Print summary statistics of ROM testing"""
    total = len(results)
    crashed = sum(1 for r in results if r['crashed'])
    completed = sum(1 for r in results if r['completed_normally'])
    has_output = sum(1 for r in results if r['has_output'])
    interesting = sum(1 for r in results if r['interesting'])
    
    print("\nSummary Statistics:")
    print("=" * 50)
    print(f"Total ROMs tested:      {total:,}")
    print(f"Crashed:                {crashed:,} ({crashed/total*100:.1f}%)")
    print(f"Completed normally:     {completed:,} ({completed/total*100:.1f}%)")
    print(f"Produced visual output: {has_output:,} ({has_output/total*100:.1f}%)")
    print(f"Interesting:            {interesting:,} ({interesting/total*100:.1f}%)")
    
    if completed > 0:
        avg_instructions = np.mean([r['instructions_executed'] for r in results if r['completed_normally']])
        print(f"Avg instructions (completed): {avg_instructions:,.0f}")
    
    if has_output > 0:
        avg_pixels = np.mean([r['final_pixel_count'] for r in results if r['has_output']])
        print(f"Avg pixels drawn (visual): {avg_pixels:.0f}")


def main():
    parser = argparse.ArgumentParser(description="Test random CHIP-8 ROMs")
    parser.add_argument("--rom-dir", type=str, help="Directory containing ROM files to test")
    parser.add_argument("--generate", type=int, help="Generate N random ROMs and test them")
    parser.add_argument("--continuous", action="store_true", help="Run continuously in batches")
    parser.add_argument("--cycles", type=int, default=1000000, help="Cycles to run each ROM")
    parser.add_argument("--batch-size", type=int, default=10000, help="Batch size for continuous mode")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--save-interesting", action="store_true", help="Save interesting ROMs")
    
    args = parser.parse_args()
    
    print("Random CHIP-8 ROM Tester - Computational Archaeology")
    print("=" * 60)
    
    os.makedirs(args.output, exist_ok=True)
    
    if args.continuous:
        # Continuous mode - run forever in batches
        if PureRandomChip8Generator is None:
            print("Error: Pure random ROM generator not available")
            return 1
        
        print(f"Running continuously with batches of {args.batch_size:,} ROMs")
        print("Press Ctrl+C to stop")
        print("=" * 60)
        
        generator = PureRandomChip8Generator(rom_size=3584)
        
        completed_total = 0
        interesting_total = 0
        batches_tested = 0
        start_time = time.time()
        
        try:
            while True:  # Run forever
                batch_start = time.time()
                
                completed, interesting = generate_and_test_random_batch(
                    generator, args.batch_size, args.cycles
                )
                
                completed_total += completed
                interesting_total += interesting
                batches_tested += 1
                
                batch_time = time.time() - batch_start
                total_time = time.time() - start_time
                total_roms = batches_tested * args.batch_size
                
                print(f"Batch {batches_tested}: {completed}/{args.batch_size} completed, "
                      f"{interesting} interesting ({batch_time:.1f}s)")
                print(f"Running totals: {total_roms:,} tested, {completed_total} completed, "
                      f"{interesting_total} interesting")
                
                # Show rate statistics
                rate = total_roms / total_time if total_time > 0 else 0
                success_rate = completed_total / total_roms * 100 if total_roms > 0 else 0
                interesting_rate = interesting_total / total_roms * 100 if total_roms > 0 else 0
                
                print(f"Rates: {rate:.0f} ROMs/sec, {success_rate:.3f}% completion, "
                      f"{interesting_rate:.4f}% interesting")
                print("=" * 60)
                
        except KeyboardInterrupt:
            print(f"\nStopped after {batches_tested} batches")
            print(f"Final results:")
            print(f"Total ROMs tested: {batches_tested * args.batch_size:,}")
            print(f"Completed normally: {completed_total}")
            print(f"Interesting: {interesting_total}")
            
            total_time = time.time() - start_time
            rate = (batches_tested * args.batch_size) / total_time if total_time > 0 else 0
            print(f"Average rate: {rate:.0f} ROMs/second")
        
    elif args.generate:
        # Original single-run mode
        if PureRandomChip8Generator is None:
            print("Error: Pure random ROM generator not available")
            return 1
        
        print(f"Generating {args.generate:,} pure random ROMs...")
        generator = PureRandomChip8Generator(rom_size=3584)
        
        completed_total = 0
        interesting_total = 0
        batches_tested = 0
        
        try:
            total_tested = 0
            while total_tested < args.generate:
                batch_size = min(20000, args.generate - total_tested)
                completed, interesting = generate_and_test_random_batch(
                    generator, batch_size, args.cycles
                )
                
                completed_total += completed
                interesting_total += interesting
                batches_tested += 1
                total_tested += batch_size
                
                print(f"Batch {batches_tested}: {completed}/{batch_size} completed, "
                      f"{interesting} interesting")
                print(f"Running totals: {completed_total} completed, "
                      f"{interesting_total} interesting")
                
        except KeyboardInterrupt:
            print("\nTesting stopped by user")
        
        print(f"\nFinal results after testing {total_tested} ROMs:")
        print(f"Completed normally: {completed_total}")
        print(f"Interesting: {interesting_total}")
        
    elif args.rom_dir:
        # Test existing ROM files
        if not os.path.exists(args.rom_dir):
            print(f"ROM directory not found: {args.rom_dir}")
            return 1
        
        rom_files = load_rom_files(args.rom_dir)
        if not rom_files:
            print(f"No ROM files found in {args.rom_dir}")
            return 1
        
        print(f"Found {len(rom_files)} ROM files")
        
        all_results = []
        rom_data_list = [rom_data for _, rom_data in rom_files]
        
        for i in range(0, len(rom_data_list), args.batch_size):
            end_idx = min(i + args.batch_size, len(rom_data_list))
            batch_data = rom_data_list[i:end_idx]
            
            print(f"Testing batch {i//args.batch_size + 1} "
                  f"(ROMs {i+1}-{end_idx})...")
            
            batch_results = test_rom_batch(batch_data, cycles=args.cycles)
            all_results.extend(batch_results)
        
        print_summary_stats(all_results)
        
        if args.save_interesting:
            interesting_output = os.path.join(args.output, "interesting_roms")
            save_interesting_roms(rom_files, all_results, interesting_output)
    
    else:
        # Interactive mode
        print("Options:")
        print("1. Generate and test random ROMs (single run)")
        print("2. Run continuously in batches") 
        print("3. Test existing ROM files")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            if PureRandomChip8Generator is None:
                print("Error: Pure random ROM generator not available")
                return 1
            
            num_roms = int(input("Number of ROMs to generate (default 10000): ") or "10000")
            generator = PureRandomChip8Generator(rom_size=3584)
            completed, interesting = generate_and_test_random_batch(
                generator, num_roms, args.cycles
            )
            print(f"Final: {completed} completed, {interesting} interesting")
            
        elif choice == "2":
            if PureRandomChip8Generator is None:
                print("Error: Pure random ROM generator not available")
                return 1
            
            batch_size = int(input("Batch size (default 10000): ") or "10000")
            print(f"Running continuously with batches of {batch_size:,} ROMs")
            print("Press Ctrl+C to stop")
            
            generator = PureRandomChip8Generator(rom_size=3584)
            completed_total = 0
            interesting_total = 0
            batches_tested = 0
            start_time = time.time()
            
            try:
                while True:
                    completed, interesting = generate_and_test_random_batch(
                        generator, batch_size, args.cycles
                    )
                    
                    completed_total += completed
                    interesting_total += interesting
                    batches_tested += 1
                    
                    total_time = time.time() - start_time
                    total_roms = batches_tested * batch_size
                    rate = total_roms / total_time if total_time > 0 else 0
                    
                    print(f"Batch {batches_tested}: {completed}/{batch_size} completed, "
                          f"{interesting} interesting")
                    print(f"Totals: {total_roms:,} tested, {completed_total} completed, "
                          f"{interesting_total} interesting ({rate:.0f} ROMs/sec)")
                    print("=" * 40)
                    
            except KeyboardInterrupt:
                print(f"\nStopped after {batches_tested} batches")
                
        elif choice == "3":
            rom_dir = input("ROM directory path: ").strip()
            if os.path.exists(rom_dir):
                rom_files = load_rom_files(rom_dir)
                if rom_files:
                    rom_data_list = [rom_data for _, rom_data in rom_files]
                    results = test_rom_batch(rom_data_list, cycles=args.cycles)
                    print_summary_stats(results)
                else:
                    print("No ROM files found")
            else:
                print("Directory not found")
        
        else:
            print("Invalid choice")
            return 1
    
    print("\nTesting completed!")
    return 0


if __name__ == "__main__":
    exit(main())