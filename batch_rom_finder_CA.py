#!/usr/bin/env python3
"""
Test Random CHIP-8 ROMs with CUDA CA Detection
High-performance batch ROM testing with integrated CUDA cellular automata detection
Focuses on finding high-quality CA patterns (60%+ likelihood)
"""

import os
import sys
import glob
import argparse
import time
import numpy as np
import cupy as cp
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Add the emulators and generators directories to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'emulators'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'generators'))

try:
    from parallel_chip8_CA import MegaKernelChip8Emulator as ParallelChip8Emulator
    print("Using enhanced CUDA CA emulator")
except ImportError:
    try:
        from parallel_chip8 import ParallelChip8Emulator
        print("Warning: Using standard emulator - no CUDA CA detection")
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


def test_rom_batch_with_cuda_ca(rom_data_list: List[bytes], cycles: int = 1000000,
                                ca_detection_interval: int = 500, ca_threshold: float = 60.0) -> Tuple[List[Dict], List[Tuple[bytes, Dict]]]:
    """Test a batch of ROMs with CUDA CA detection - silent execution"""
    
    # Create enhanced emulator with CUDA CA detection (no debug output)
    emulator = ParallelChip8Emulator(
        len(rom_data_list),
        ca_detection_interval=ca_detection_interval,
        ca_threshold=ca_threshold
    )
    
    # Load and run ROMs (silent)
    emulator.load_roms(rom_data_list)
    emulator.run(cycles=cycles)
    
    # Get CA detection results from CUDA
    ca_results = emulator.get_ca_results()
    
    # Filter for high-quality CA patterns only
    high_quality_ca_instances = [
        ca for ca in ca_results['ca_instances'] 
        if ca['ca_likelihood'] >= 60.0  # Strict 60% threshold
    ]
    
    # Analyze results for each ROM
    results = []
    displays = emulator.get_displays()
    ca_roms = []
    
    # Create mapping of instance_id to high-quality CA analysis
    ca_by_instance = {ca['instance_id']: ca for ca in high_quality_ca_instances}
    
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
        
        # Check if this instance has HIGH-QUALITY CA patterns
        ca_analysis = ca_by_instance.get(i)
        has_ca = ca_analysis is not None
        
        # Only add ROMs with 60%+ CA likelihood
        if has_ca:
            ca_roms.append((rom_data_list[i], ca_analysis))
        
        result = {
            'execution_time': 0,  # Not needed for CA focus
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
            'interesting': (not crashed and not waiting_for_key) and (has_output and has_structure),
            'has_ca': has_ca,
            'ca_analysis': ca_analysis
        }
        
        results.append(result)
    
    return results, ca_roms


def save_ca_roms(rom_files: List[Tuple[str, bytes]], results: List[Dict], 
                output_dir: str = "ca_roms"):
    """Save high-quality CA ROMs (60%+ likelihood) with minimal output"""
    os.makedirs(output_dir, exist_ok=True)
    
    saved_count = 0
    
    for (filename, rom_data), result in zip(rom_files, results):
        if result.get('has_ca', False):
            ca_analysis = result['ca_analysis']
            
            # Create descriptive filename with CA prefix
            base_name = Path(filename).stem
            
            new_filename = (f"CA_likelihood{ca_analysis['ca_likelihood']:.0f}"
                          f"_inst{result['instructions_executed']}"
                          f"_pix{result['final_pixel_count']}"
                          f"_dens{result['pixel_density']:.3f}"
                          f"_loop{ca_analysis['hot_loop_range'][0]:03X}-{ca_analysis['hot_loop_range'][1]:03X}"
                          f"_{base_name}")
            
            # Save ROM file
            rom_path = os.path.join(output_dir, f"{new_filename}.ch8")
            with open(rom_path, 'wb') as f:
                f.write(rom_data)
            
            # Generate screenshot silently
            try:
                screenshot_emulator = ParallelChip8Emulator(1)
                screenshot_emulator.load_single_rom(rom_data)
                screenshot_emulator.run(cycles=1000000)
                
                display = screenshot_emulator.get_displays()[0]
                if hasattr(display, 'get'):
                    display_np = display.get()
                else:
                    display_np = display
                
                from PIL import Image
                display_img = (display_np * 255).astype(np.uint8)
                scale_factor = 8
                scaled_img = np.repeat(np.repeat(display_img, scale_factor, axis=0), scale_factor, axis=1)
                
                img = Image.fromarray(scaled_img, mode='L')
                screenshot_path = os.path.join(output_dir, f"{new_filename}.png")
                img.save(screenshot_path)
                
                # Save detailed CA analysis
                ca_report_path = os.path.join(output_dir, f"{new_filename}_CUDA_CA_ANALYSIS.txt")
                with open(ca_report_path, 'w') as f:
                    f.write(f"HIGH-QUALITY CUDA Cellular Automata Analysis\n")
                    f.write(f"=" * 60 + "\n\n")
                    f.write(f"ROM: {new_filename}.ch8\n")
                    f.write(f"Detection Method: Real-time CUDA kernel analysis\n")
                    f.write(f"CA Likelihood: {ca_analysis['ca_likelihood']:.1f}% (HIGH CONFIDENCE)\n")
                    f.write(f"Hot Loop Range: 0x{ca_analysis['hot_loop_range'][0]:03X}-0x{ca_analysis['hot_loop_range'][1]:03X}\n")
                    f.write(f"Instance ID: {ca_analysis['instance_id']}\n\n")
                    f.write(f"Execution Statistics:\n")
                    f.write(f"- Instructions executed: {result['instructions_executed']:,}\n")
                    f.write(f"- Display writes: {result['display_writes']}\n")
                    f.write(f"- Pixels drawn: {result['pixels_drawn']}\n")
                    f.write(f"- Final pixel count: {result['final_pixel_count']}\n")
                    f.write(f"- Pixel density: {result['pixel_density']:.3f}\n")
                    f.write(f"- Completed normally: {'Yes' if result['completed_normally'] else 'No'}\n")
                    f.write(f"- Final PC: 0x{result['final_pc']:03X}\n\n")
                    f.write(f"Quality Assessment:\n")
                    f.write(f"This ROM shows strong evidence of cellular automata patterns\n")
                    f.write(f"with {ca_analysis['ca_likelihood']:.1f}% confidence. Only ROMs with\n")
                    f.write(f"60%+ likelihood are saved, making this a high-quality candidate.\n")
                
            except Exception:
                pass  # Silent failure for screenshots
            
            saved_count += 1
    
    return saved_count


def generate_and_test_random_batch_ca(generator, num_roms: int = 10000, 
                                     test_cycles: int = 1000000,
                                     ca_detection_interval: int = 500,
                                     ca_threshold: float = 60.0) -> Tuple[int, int, int]:
    """Generate and test ROMs with silent execution, 60%+ CA threshold"""
    
    # Generate ROMs
    roms = generator.generate_batch(num_roms)
    rom_data_list = [cp.asnumpy(rom).tobytes() for rom in roms]
    
    # Test with CUDA CA detection (silent)
    results, ca_roms = test_rom_batch_with_cuda_ca(
        rom_data_list, 
        cycles=test_cycles,
        ca_detection_interval=ca_detection_interval,
        ca_threshold=ca_threshold
    )
    
    # Count results
    interesting_count = sum(1 for r in results if r['interesting'])
    completed_count = sum(1 for r in results if r['completed_normally'])
    has_output_count = sum(1 for r in results if r['has_output'])
    crashed_count = sum(1 for r in results if r['crashed'])
    ca_count = len(ca_roms)
    
    # Save high-quality CA ROMs only
    if ca_count > 0:
        rom_files = [(f"random_{i:06d}.ch8", rom_data) for i, rom_data in enumerate(rom_data_list)]
        save_ca_roms(rom_files, results, "output/ca_roms")
    
    return completed_count, interesting_count, ca_count


def print_summary_stats(results: List[Dict]):
    """Print concise summary statistics"""
    total = len(results)
    crashed = sum(1 for r in results if r['crashed'])
    completed = sum(1 for r in results if r['completed_normally'])
    has_output = sum(1 for r in results if r['has_output'])
    interesting = sum(1 for r in results if r['interesting'])
    ca_roms = sum(1 for r in results if r.get('has_ca', False))
    
    print(f"\nSummary: {total:,} ROMs tested")
    print(f"Crashed: {crashed:,} ({crashed/total*100:.1f}%)")
    print(f"Completed: {completed:,} ({completed/total*100:.1f}%)")
    print(f"Visual output: {has_output:,} ({has_output/total*100:.1f}%)")
    print(f"Interesting: {interesting:,} ({interesting/total*100:.1f}%)")
    print(f"HIGH-QUALITY CA (60%+): {ca_roms:,} ({ca_roms/total*100:.6f}%)")
    
    if ca_roms > 0:
        ca_likelihoods = [r['ca_analysis']['ca_likelihood'] for r in results if r.get('has_ca', False)]
        avg_ca_likelihood = np.mean(ca_likelihoods)
        max_ca_likelihood = np.max(ca_likelihoods)
        print(f"CA likelihood (avg/max): {avg_ca_likelihood:.1f}%/{max_ca_likelihood:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Find high-quality CA patterns in random CHIP-8 ROMs")
    parser.add_argument("--rom-dir", type=str, help="Directory containing ROM files to test")
    parser.add_argument("--generate", type=int, help="Generate N random ROMs and test them")
    parser.add_argument("--continuous", action="store_true", help="Run continuously in batches")
    parser.add_argument("--cycles", type=int, default=1000000, help="Cycles to run each ROM")
    parser.add_argument("--batch-size", type=int, default=10000, help="Batch size for continuous mode")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--ca-threshold", type=float, default=60.0, help="CA likelihood threshold (60-100)")
    parser.add_argument("--ca-interval", type=int, default=500, help="CA detection interval (cycles)")
    
    args = parser.parse_args()
    
    print("CHIP-8 CA Hunter - High-Quality CA Detection (60%+ threshold)")
    print("=" * 60)
    
    os.makedirs(args.output, exist_ok=True)
    
    if args.continuous:
        if PureRandomChip8Generator is None:
            print("Error: Pure random ROM generator not available")
            return 1
        
        print(f"Continuous CA hunting: {args.batch_size:,} ROMs per batch, 60%+ threshold")
        print("Press Ctrl+C to stop\n")
        
        generator = PureRandomChip8Generator(rom_size=3584)
        
        completed_total = 0
        interesting_total = 0
        ca_total = 0
        batches_tested = 0
        start_time = time.time()
        
        try:
            while True:
                batch_start = time.time()
                
                completed, interesting, ca_count = generate_and_test_random_batch_ca(
                    generator, 
                    args.batch_size, 
                    args.cycles,
                    ca_detection_interval=args.ca_interval,
                    ca_threshold=args.ca_threshold
                )
                
                completed_total += completed
                interesting_total += interesting
                ca_total += ca_count
                batches_tested += 1
                
                batch_time = time.time() - batch_start
                total_time = time.time() - start_time
                total_roms = batches_tested * args.batch_size
                
                # Concise batch summary
                print(f"Batch {batches_tested}: {completed}/{args.batch_size} completed, "
                      f"{interesting} interesting, {ca_count} HIGH-QUALITY CA ({batch_time:.1f}s)")
                
                # Running totals
                rate = total_roms / total_time if total_time > 0 else 0
                ca_rate = ca_total / total_roms * 100 if total_roms > 0 else 0
                
                print(f"Totals: {total_roms:,} tested, {ca_total} HIGH-QUALITY CA "
                      f"({rate:.0f} ROMs/sec, {ca_rate:.6f}% CA rate)")
                
                if ca_total > 0:
                    discovery_rate = total_roms // ca_total
                    print(f"Discovery rate: 1 high-quality CA per {discovery_rate:,} ROMs")
                
                print()  # Empty line for readability
                
        except KeyboardInterrupt:
            print(f"\nStopped after {batches_tested} batches")
            print(f"Final: {batches_tested * args.batch_size:,} ROMs tested, {ca_total} HIGH-QUALITY CA found")
            
            if ca_total > 0:
                total_time = time.time() - start_time
                rate = (batches_tested * args.batch_size) / total_time if total_time > 0 else 0
                discovery_rate = (batches_tested * args.batch_size) // ca_total
                print(f"Rate: {rate:.0f} ROMs/sec, 1 high-quality CA per {discovery_rate:,} ROMs")
        
    elif args.generate:
        if PureRandomChip8Generator is None:
            print("Error: Pure random ROM generator not available")
            return 1
        
        print(f"Testing {args.generate:,} ROMs for high-quality CA patterns...")
        generator = PureRandomChip8Generator(rom_size=3584)
        
        completed_total = 0
        interesting_total = 0
        ca_total = 0
        
        try:
            total_tested = 0
            while total_tested < args.generate:
                batch_size = min(20000, args.generate - total_tested)
                completed, interesting, ca_count = generate_and_test_random_batch_ca(
                    generator, 
                    batch_size, 
                    args.cycles,
                    ca_detection_interval=args.ca_interval,
                    ca_threshold=args.ca_threshold
                )
                
                completed_total += completed
                interesting_total += interesting
                ca_total += ca_count
                total_tested += batch_size
                
                print(f"Progress: {total_tested}/{args.generate} tested, "
                      f"{ca_count} HIGH-QUALITY CA in this batch")
                
        except KeyboardInterrupt:
            print("\nTesting stopped by user")
        
        print(f"\nFinal: {total_tested} ROMs tested, {ca_total} HIGH-QUALITY CA found")
        if ca_total > 0:
            discovery_rate = total_tested // ca_total
            print(f"Discovery rate: 1 high-quality CA per {discovery_rate:,} ROMs")
        
    elif args.rom_dir:
        if not os.path.exists(args.rom_dir):
            print(f"ROM directory not found: {args.rom_dir}")
            return 1
        
        rom_files = load_rom_files(args.rom_dir)
        if not rom_files:
            print(f"No ROM files found in {args.rom_dir}")
            return 1
        
        print(f"Testing {len(rom_files)} existing ROMs for high-quality CA patterns...")
        
        all_results = []
        rom_data_list = [rom_data for _, rom_data in rom_files]
        
        for i in range(0, len(rom_data_list), args.batch_size):
            end_idx = min(i + args.batch_size, len(rom_data_list))
            batch_data = rom_data_list[i:end_idx]
            
            batch_results, _ = test_rom_batch_with_cuda_ca(
                batch_data, 
                cycles=args.cycles,
                ca_detection_interval=args.ca_interval,
                ca_threshold=args.ca_threshold
            )
            all_results.extend(batch_results)
        
        print_summary_stats(all_results)
        
        ca_count = sum(1 for r in all_results if r.get('has_ca', False))
        if ca_count > 0:
            save_ca_roms(rom_files, all_results, os.path.join(args.output, "ca_roms"))
            print(f"Saved {ca_count} HIGH-QUALITY CA ROMs")
    
    else:
        print("Interactive mode:")
        print("1. Generate ROMs and hunt for high-quality CA")
        print("2. Continuous CA hunting") 
        print("3. Test existing ROMs")
        
        choice = input("Choice (1-3): ").strip()
        
        if choice == "1":
            if PureRandomChip8Generator is None:
                print("Error: Pure random ROM generator not available")
                return 1
            
            num_roms = int(input("Number of ROMs (default 10000): ") or "10000")
            generator = PureRandomChip8Generator(rom_size=3584)
            completed, interesting, ca_count = generate_and_test_random_batch_ca(
                generator, num_roms, args.cycles, args.ca_interval, args.ca_threshold
            )
            print(f"Result: {completed} completed, {interesting} interesting, {ca_count} HIGH-QUALITY CA")
            
        elif choice == "2":
            if PureRandomChip8Generator is None:
                print("Error: Pure random ROM generator not available")
                return 1
            
            batch_size = int(input("Batch size (default 10000): ") or "10000")
            print(f"Continuous hunting with {batch_size:,} ROMs per batch")
            print("Press Ctrl+C to stop\n")
            
            generator = PureRandomChip8Generator(rom_size=3584)
            ca_total = 0
            batches_tested = 0
            start_time = time.time()
            
            try:
                while True:
                    completed, interesting, ca_count = generate_and_test_random_batch_ca(
                        generator, batch_size, args.cycles, args.ca_interval, args.ca_threshold
                    )
                    
                    ca_total += ca_count
                    batches_tested += 1
                    
                    total_roms = batches_tested * batch_size
                    rate = total_roms / (time.time() - start_time)
                    
                    print(f"Batch {batches_tested}: {ca_count} HIGH-QUALITY CA found")
                    print(f"Total: {total_roms:,} tested, {ca_total} HIGH-QUALITY CA ({rate:.0f} ROMs/sec)")
                    
                    if ca_total > 0:
                        print(f"Rate: 1 high-quality CA per {total_roms // ca_total:,} ROMs")
                    print()
                    
            except KeyboardInterrupt:
                print(f"\nStopped: {ca_total} HIGH-QUALITY CA found in {batches_tested * batch_size:,} ROMs")
                
        elif choice == "3":
            rom_dir = input("ROM directory path: ").strip()
            if os.path.exists(rom_dir):
                rom_files = load_rom_files(rom_dir)
                if rom_files:
                    rom_data_list = [rom_data for _, rom_data in rom_files]
                    results, _ = test_rom_batch_with_cuda_ca(
                        rom_data_list, 
                        cycles=args.cycles,
                        ca_detection_interval=args.ca_interval,
                        ca_threshold=args.ca_threshold
                    )
                    print_summary_stats(results)
                    
                    ca_count = sum(1 for r in results if r.get('has_ca', False))
                    if ca_count > 0:
                        save_ca_roms(rom_files, results, "output/ca_roms")
                        print(f"Saved {ca_count} HIGH-QUALITY CA ROMs")
                else:
                    print("No ROM files found")
            else:
                print("Directory not found")
        
        else:
            print("Invalid choice")
            return 1
    
    print("\nCA hunting completed!")
    return 0


if __name__ == "__main__":
    exit(main())